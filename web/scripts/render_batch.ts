// Playwright batch render worker (design doc 14: web/scripts/render_batch.ts).
//
// Protocol (JSONL over stdin/stdout + files):
//   stdin : one JSON per line: {kind:"job", job:RenderJob} ... then EOF
//   RenderJob = { job_id, scenes: [{sample_id, scene, camera_ids, passes,
//                                    out_dir}], render_url }
//   stdout: one JSON per line: {kind:"progress", job_id, done, total}
//                          or {kind:"sample_done", job_id, sample_id, files}
//                          or {kind:"error", job_id, sample_id, message}
//   exit non-zero on fatal errors; per-sample errors reported and skipped.
import { chromium, type Browser, type BrowserContext, type Page } from "playwright";
import { PNG } from "pngjs";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";

const MIME: Record<string, string> = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".png": "image/png",
};

function startStaticServer(root: string): Promise<{ url: string; close: () => void }> {
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent((req.url ?? "/").split("?")[0]);
    if (p === "/") p = "/render.html";
    const file = path.join(root, p);
    if (!file.startsWith(path.resolve(root))) {
      res.writeHead(403).end();
      return;
    }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404).end(); return; }
      res.writeHead(200, {
        "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream",
      });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({
        url: `http://127.0.0.1:${addr.port}/render.html`,
        close: () => server.close(),
      });
    });
  });
}

interface RgbaImage {
  width: number;
  height: number;
  b64: string;
}

interface RenderJobItem {
  sample_id: string;
  scene: unknown;
  camera_ids: string[];
  passes: string[];
  out_dir: string;
  bev_range?: { x_min: number; x_max: number; y_min: number; y_max: number };
  bev_mask_size?: [number, number];
}

interface RenderJob {
  job_id: string;
  scenes: RenderJobItem[];
  render_url: string;
}

interface JobResult {
  sample_id: string;
  files: string[];
  errors?: string[];
}

function b64ToBytes(b64: string): Uint8Array {
  return new Uint8Array(Buffer.from(b64, "base64"));
}

function writePng(file: string, img: RgbaImage, stripAlpha = true,
                  binary = false): number {
  const png = new PNG({ width: img.width, height: img.height });
  png.data.set(b64ToBytes(img.b64));
  if (binary) {
    for (let i = 0; i < png.data.length; i += 4) {
      const on = png.data[i] > 0 || png.data[i + 1] > 0 || png.data[i + 2] > 0;
      const v = on ? 255 : 0;
      png.data[i] = v; png.data[i + 1] = v; png.data[i + 2] = v;
      png.data[i + 3] = 255;
    }
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, PNG.sync.write(png));
    return img.width * img.height;
  }
  if (stripAlpha) {
    for (let i = 3; i < png.data.length; i += 4) png.data[i] = 255;
  }
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, PNG.sync.write(png));
  return img.width * img.height;
}

async function renderOne(page: Page, item: RenderJobItem): Promise<JobResult> {
  const files: string[] = [];
  const out = item.out_dir;
  const res = await page.evaluate(
    ([scene, cameraIds, passes, bevRange, bevMaskSize]) =>
      (window as unknown as {
        toyRenderer: {
          renderSample: (req: unknown) => unknown;
        };
      }).toyRenderer.renderSample({
        scene, cameraIds, passes, bevRange, bevMaskSize,
      }),
    [item.scene, item.camera_ids, item.passes, item.bev_range, item.bev_mask_size]
  ) as {
    cameras: Record<string, {
      rgbRgba?: RgbaImage;
      visibleInstanceRgba?: RgbaImage;
      amodalPerArrowRgba?: Record<string, RgbaImage>;
      calibration: unknown;
    }>;
    bevInstanceRgba?: RgbaImage;
  };
  for (const [camId, cam] of Object.entries(res.cameras)) {
    if (cam.rgbRgba) {
      const f = path.join(out, "rgb", `${camId}.png`);
      writePng(f, cam.rgbRgba);
      files.push(f);
    }
    if (cam.visibleInstanceRgba) {
      const f = path.join(out, "masks", `${camId}_visible.png`);
      writePng(f, cam.visibleInstanceRgba);
      files.push(f);
    }
    if (cam.amodalPerArrowRgba) {
      for (const [id, img] of Object.entries(cam.amodalPerArrowRgba)) {
        const f = path.join(out, "masks", "amodal", camId, `${id}.png`);
        writePng(f, img, true, true);
        files.push(f);
      }
    }
  }
  if (res.bevInstanceRgba) {
    const f = path.join(out, "masks", "bev_instance.png");
    writePng(f, res.bevInstanceRgba);
    files.push(f);
  }
  return { sample_id: item.sample_id, files };
}

async function runJob(browser: Browser, job: RenderJob): Promise<void> {
  const context = await browser.newContext();
  const page = await context.newPage();
  page.on("pageerror", (e) => {
    process.stdout.write(
      JSON.stringify({ kind: "error", job_id: job.job_id, message: String(e) }) + "\n"
    );
  });
  await page.goto(job.render_url, { waitUntil: "load" });
  await page.waitForFunction(() =>
    (window as unknown as { toyRenderer?: unknown }).toyRenderer != null
  , { timeout: 30000 });
  let done = 0;
  const total = job.scenes.length;
  for (const item of job.scenes) {
    try {
      const r = await renderOne(page, item);
      process.stdout.write(
        JSON.stringify({ kind: "sample_done", job_id: job.job_id, ...r }) + "\n"
      );
    } catch (e) {
      process.stdout.write(
        JSON.stringify({
          kind: "error", job_id: job.job_id, sample_id: item.sample_id,
          message: String(e),
        }) + "\n"
      );
    }
    done += 1;
    process.stdout.write(
      JSON.stringify({ kind: "progress", job_id: job.job_id, done, total }) + "\n"
    );
  }
  await context.close();
}

async function main() {
  const raw = await new Promise<string>((resolve) => {
    let buf = "";
    process.stdin.setEncoding("utf-8");
    process.stdin.on("data", (d) => (buf += d));
    process.stdin.on("end", () => resolve(buf));
  });
  const jobs: RenderJob[] = raw.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  if (!jobs.length) {
    process.stderr.write("no jobs on stdin\n");
    process.exit(2);
  }
  const execPath = process.env.ARROW_CHROME_BIN || undefined;
  const browser = await chromium.launch({
    headless: true,
    ...(execPath ? { executablePath: execPath } : {}),
  });
  for (const job of jobs) {
    const url = job.render_url.startsWith("file:")
      ? (await startStaticServer(path.dirname(job.render_url.replace("file://", "")))).url
      : job.render_url;
    await runJob(browser, { ...job, render_url: url });
  }
  await browser.close();
  process.exit(0);
}

main().catch((e) => {
  process.stderr.write(String(e) + "\n");
  process.exit(1);
});
