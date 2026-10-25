# Toy Roadmark Detection

用可编辑的三维仿真场景，快速验证地面箭头 BEV 检测头及结构调整。

**当前阶段：P0–P7 已实现并有自动化验收；快模式精度目标未达标，原因待验证（工作假设与证据见下）。**

- [完整开发设计书](docs/ARROW_TOY_DESIGN.md)
- [实现决策与差异记录](docs/IMPLEMENTATION_NOTES.md)
- [实现与验证报告](docs/REPORT.md)（最终结论/复现/失败限制）
- 目标链路：单帧图像 → Tiny CNN → 地面几何投影与 BEV 特征 → 32-query decoder → 类型、中心、朝向。
- 被遮挡箭头忽略监督，并处理未匹配 query 的背景忽略。

## 最短路径（从零复现）

```bash
# 0) 环境：Python 3.11、Node 22+；GPU 可选（CUDA 12.x + 驱动 530+）
# torch 单独安装（本仓库验证过 cu121 wheel；CPU-only 可装 torch==2.5.1）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]                     # 运行+测试依赖（含 jsonschema/playwright）
npm --prefix web ci                       # 前端依赖（锁定在 package-lock.json）
# 浏览器二选一（缺一会在浏览器集成测试处明确失败，不会静默跳过）：
npx --prefix web playwright install chromium
#   或已有 Chrome for Testing：export ARROW_CHROME_BIN=/path/to/chrome

# 1) 构建 + 启动编辑器（http://127.0.0.1:8765）
npm --prefix web run build
python -m arrow_toy.editor_server --host 127.0.0.1 --port 8765

# 2) 生成、审计并缓存数据（约 11 分钟，7648 样本；配置 configs/data_v1.yaml）
python -m arrow_toy.render --config configs/data_v1.yaml
python -m arrow_toy.data.audit --root data/arrow_v1
python -m arrow_toy.data.prepare_cache --root data/arrow_v1

# 3) 关键语义验证（含 JS/Python 投影一致性、遮挡、忽略监督）
python -m pytest tests/ -q

# 4) 小样本拟合诊断（P4 门槛；预期 P/R→1）
python -m arrow_toy.train --config configs/overfit.yaml --run-dir runs/overfit

# 5) 性能基准 + 10 分钟快模式基线与单变量对比
python -m arrow_toy.benchmark --config configs/quick.yaml --output runs/benchmark.json
python -m arrow_toy.train --config configs/quick.yaml --run-dir runs/baseline_s42
python -m arrow_toy.train --config configs/quick.yaml \
    --override configs/ablations/no_query_sa.yaml --run-dir runs/no_sa_s42

# 6) 完整评测（阈值在 val 固定，test 不调参）与结果对比、叠图
python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split test_clean
python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split test_occlusion
python -m arrow_toy.evaluation.compare --runs runs/baseline_s42 runs/no_sa_s42 \
    --output runs/comparison
python -m arrow_toy.evaluation.visualize --run-dir runs/baseline_s42 --split val
```

浏览器不可下载时的备选：设置 `ARROW_CHROME_BIN` 指向任意 Chrome for Testing
二进制（本仓库验证过 156.0.8078.4）。

## 结果速览（本机实测，详见 docs/IMPLEMENTATION_NOTES.md）

固定步数主表（仅完整 1500 步配对；修正评测器、val 选阈值、test 冻结）：

| run | 种子 | f1@0.5 | mAP@0.5 | 宏F1 | test_clean | 墙钟 | status |
|---|---|---|---|---|---|---|---|
| overfit32_v2（诊断） | 42 | 0.984 | 1.000 | 0.985 | — | 54.4s | success |
| baseline A（SA 开） | 42 | 0.293 | 0.170 | 0.306 | 0.297 | 543.9s | success |
| baseline A | 44 | 0.355 | 0.264 | 0.376 | 0.386 | 544.9s | success |
| no_sa B（SA 关） | 42 | 0.427 | 0.283 | 0.427 | 0.415 | 543.0s | success |
| no_sa B | 44 | 0.451 | 0.349 | 0.443 | 0.457 | 543.0s | success |

- 主表均值：A = 0.324±0.044，B = 0.439±0.017（n=2 完整配对）。B 在全部
  完整配对中占优。
- 种子 43 对为诊断观察：A 侧 run 在预算内仅完成 1369 步
  （budget_incomplete），按设计 10.4 不进入固定步数结构结论；其 B 侧
  完整（f1 0.438），与主表方向一致。
- lr 5e-4 变体（种子 42，完整）：f1 0.251 —— 劣于默认 1e-3。

- 时间/显存达标（≤600 s、1.12 GB）；**精度目标（f1≥0.85、mAP≥0.75、角≤10°）未达标**；
  中心误差 0.21–0.26 m（≤0.35 达标）、空帧误报 ≤0.054（≤0.10 达标）。budget_incomplete
  的 run 不计为预算通过。
- 精度瓶颈的工作假设（未定证）：sparse 头 + 一对一匹配在固定 1500 步
  预算下收敛不足。支持证据：overfit 1000 步即 0.984；oracle-BEV 探针
  （真值直接写入 token）1200 步也仅 ~0.55；关 SA 每个完整配对都更好。
  消融 E（dense 头，同 BEV 特征）为设计的下一步单变量验证。
- 本机 RTX 4080 实测性能仅为正常水平的 ~1/5–1/10（驱动报 ERR/not-ready），
  已在文档中记录；这影响墙钟余量但不影响固定步数的可比性。

## 目录约定

- `data/arrow_v1/` 数据集（samples/splits/cache/audit.json/manifest.json）
- `runs/<name>/` 训练产物（resolved_config/environment/metrics/result/
  last.pt/best.pt/predictions_val/figures）
- `docs/ARROW_TOY_DESIGN.md` 设计书（契约与验收标准）
- `docs/IMPLEMENTATION_NOTES.md` 所有实现差异、问题与证据
