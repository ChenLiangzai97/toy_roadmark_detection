// Unit checks for the detection overlay projection helpers (run by
// pytest via tsx). Verifies world-frame heading direction through the
// DEFAULT camera: +x must draw UP (v decreases), +y (left) must draw
// LEFT (u decreases), behind-camera points are rejected.
import { headingSegment, projGround } from "../src/shared/detection";

const K = [[166.277, 0, 160], [0, 166.277, 96], [0, 0, 1]];
const T = [
  [0, -1, 0, 0],
  [-0.207912, 0, -0.978148, 1.662551],
  [0.978148, 0, -0.207912, 0.352655],
  [0, 0, 0, 1],
];
const cam = { K, T_C_E: T, width: 320, height: 192 };

function assert(cond: boolean, msg: string) {
  if (!cond) { console.error("FAIL:", msg); process.exit(1); }
}

// center projection sanity: a forward point lands mid-image, below the
// horizon (v > 96) on the ground plane
const p = projGround(cam, 10, 0);
assert(p !== null, "forward ground point must project");
const pu = p as [number, number];
assert(pu[0] > 150 && pu[0] < 170, `u ~160, got ${pu[0]}`);
// ground point at 10 m: depression ~9.6 deg < 12 deg pitch, so it sits
// above the optical axis but below the horizon (v_h = 96 - fy*tan12 ~ 60)
assert(pu[1] > 60 && pu[1] < 96, `ground between horizon and center, got ${pu[1]}`);

// heading +x (forward): endpoint must be UP (smaller v) in the image
const segX = headingSegment(cam, 10, 0, [1, 0]);
assert(segX !== null, "+x segment must exist");
assert(!!segX && segX[1][1] < segX[0][1],
  `+x must draw UP: v0=${segX?.[0][1]} v1=${segX?.[1][1]}`);

// heading +y (left): endpoint must be LEFT (smaller u)
const segY = headingSegment(cam, 10, 0, [0, 1]);
assert(segY !== null, "+y segment must exist");
assert(!!segY && segY[1][0] < segY[0][0],
  `+y must draw LEFT: u0=${segY?.[0][0]} u1=${segY?.[1][0]}`);

// heading -y (right): endpoint must be RIGHT (larger u)
const segR = headingSegment(cam, 10, 0, [0, -1]);
assert(segR !== null && segR[1][0] > segR[0][0], "-y must draw RIGHT");

// behind camera -> null
assert(projGround(cam, -5, 0) === null, "behind-camera must be null");
assert(headingSegment(cam, -5, 0, [1, 0]) === null,
  "segment anchored behind camera must be null");

console.log("detection_check OK");
