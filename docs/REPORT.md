# 实现与验证报告

- 实现 commit：12da307（基准：7710335 = 空仓库设计文档）
- 从零安装验证：全新 venv（torch 2.5.1+cu121 wheel + `pip install -e .[dev]`）安装、40/40 测试、CLI dry-run 与上述 P4 复现链全部通过（依赖清单见 pyproject.toml：运行含 jsonschema，dev 含 playwright）。
- 环境与 GPU：RTX 4080 16 GB（本机实测性能约为正常水平 1/5–1/10，见下）、驱动 535.288.01、torch 2.5.1+cu121、Python 3.11.5、Node 24、Chrome for Testing 156；前端/依赖版本锁定于 web/package-lock.json 与 docs/IMPLEMENTATION_NOTES.md
- 数据版本/哈希及 split：data/arrow_v1，data_version `a065b96fe99c9dbd`（sha256(manifest+audit) 前 16 位）；overfit 48 / train 6040（含 40 类别对照）/ val 600 / test_clean 600 / test_occlusion 200 / test_pose 200 / stress 8；audit 通过（无泄漏、0 不变式违反、完整性检查 0 问题）
- 生效配置：各 run 目录 `resolved_config.yaml|json`
- 实际网络参数量：177,739 trainable（< 0.5M 目标）
- 与设计书的差异及理由：见 docs/IMPLEMENTATION_NOTES.md（要点：pass 返回 raw RGBA 由 Node 端 pngjs 编码、模板并集用图元集合表达、渲染器修 three.js 镜像视锥退化、11.2 抑制按模板轮廓、本机 GPU 降速事实）

## 已完成阶段

| 阶段 | 结果 | 证据/产物 |
|---|---|---|
| P0 契约 | 通过 | tests/test_projection.py（JS↔Python ≤0.5px，含偏移主点）、test_bev.py、test_templates.py；schemas/、assets/arrows/ |
| P1 最小生成 | 通过 | tests/test_render_semantics.py（BEV 方向/镜像/对齐）、test_generation_contract.py（无/部分/完全遮挡标签证据） |
| P2 编辑器 | 通过 | tests/test_editor_flow.py（放置/撤销重做/相机/保存重开一致/非法导入拒绝/预览逐像素=离线渲染/9样本跨块任务/取消）、test_api_validation.py（三入口同步 400） |
| P3 数据 | 通过 | 7648 样本零错误渲染；audit.json 全过（含完整性抽样复核）；cache 1.4 GB（版本绑定） |
| P4 模型 | 通过 | runs/overfit32_v2：32 场景 1000 步 54.4 s，f1 0.984 / mAP 1.000 / 中心 0.021 m / 角 0.52°；全新 venv 按本文命令复现 runs/overfit32_verify：f1 1.000 / mAP 1.000 / 中心 0.020 m / 角 0.64°（差异来自 cudnn autotune 非确定性，同过门槛）；8+ 语义测试 |
| P5 快模式 | 部分通过 | 时间/显存实测达标；精度目标未达标（见下） |
| P6 对比 | 两种子受控对比成立 | runs/comparison/comparison.md（主表=完整 1500 步配对） |
| P7 交接 | 本报告 + README + AGENTS.md + IMPLEMENTATION_NOTES.md | 见"复现命令" |

## 快模式结果（主表：完整 1500 步 A/B 配对，种子 42/44）

| 版本/种子 | 完成步数 | f1@0.5 | mAP@0.5 | 宏F1 | 中心误差 m | 角误差 ° | 墙钟 s | 显存 GB | status |
|---|---|---|---|---|---|---|---|---|---|
| A 基线 s42 | 1500 | 0.293 | 0.170 | 0.306 | 0.254 | 16.4 | 543.9 | 1.12 | success |
| A 基线 s44 | 1500 | 0.355 | 0.264 | 0.376 | 0.230 | 17.9 | 544.9 | 1.12 | success |
| B 关SA s42 | 1500 | 0.427 | 0.283 | 0.427 | 0.260 | 17.9 | 543.0 | 1.12 | success |
| B 关SA s44 | 1500 | 0.451 | 0.349 | 0.443 | 0.213 | 12.9 | 543.0 | 1.12 | success |

- A = 0.324±0.044；B = 0.439±0.017（样本标准差，n=2）。B 赢得全部完整配对且方差更小。
- 诊断表（不入结构结论）：A/s43 1369 步 budget_incomplete；B/s43 完整 0.438；lr 5e-4 变体 0.251。

## 目标达成情况（设计 10.5）

| 指标 | 目标 | 实测 | 结论 |
|---|---|---|---|
| mAP@0.5 | ≥0.75 | 0.17–0.35 | 未达标 |
| 宏平均 F1@0.5 | ≥0.85 | 0.31–0.44 | 未达标 |
| 中心误差 | ≤0.35 m | 0.21–0.26 m | 达标 |
| 角误差 | ≤10° | 12.9–17.9° | 未达标 |
| 空帧误报 | ≤0.10/帧 | ≤0.054 | 达标 |
| 600 s 含最终 val/save | ≤600 s | 543–547 s | 达标（截断 run 不计） |
| 峰值显存 | ≤12 GB | 1.12 GB | 达标 |

## 关键语义检查

投影（JS↔Python）、遮挡（amodal 不变/visible 变化/忽略规则）、背景忽略（三情形单测）、匹配（GT 序不变、重复唯一正配、容量>32 拒绝）、梯度（backbone→BEV→decoder→三分支）、数据划分（group 无泄漏 + 完整性抽样复核 + 版本门槛）、保存加载（ckpt 一致性）：全部通过，40/40 测试（tests/）。

## 失败与限制

1. 精度未达工程目标。工作假设（非定证）：sparse 头 + 一对一匹配在固定 1500 步（≈8 epoch）预算下收敛不足。证据：overfit 32 场景 1000 步即 0.984；oracle-BEV 探针（真值类别+占用直接写入 token）1200 步 × batch 64 也仅 ~0.55；关 SA 每个种子都更好。设计的下一步单变量实验是消融 E（dense 头已实现，独立补充运行中）。
2. 本机 RTX 4080 实测降速 ~5–10×（4096³ bf16 matmul 9.1 ms，正常 ~2 ms；nvidia-smi 报 ERR/not-ready）。影响墙钟余量与中途验证次数，不影响固定步数可比性；换正常 GPU 时步时延余量可恢复。
3. 遮挡集 40/200 帧遮挡物未命中目标（生成几何近似所致），标签如实描述渲染结果；如需更高命中率需改进遮挡物放置算法。
4. 位置覆盖 12/32 粗格（车道式布局所致），类别与位置已解耦（类别均衡按实例）。
5. 多相机训练融合、车道转向任务、掉头类箭头：未实现（设计明确保留给后续版本）。
   注：mAP/宏F1（101 点 AP）与 11.2 忽略区抑制均已实现并有回归测试。

## 复现命令

```bash
# 安装（torch 用 cu121 wheel；本机验证 2.5.1+cu121）
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .[dev]
npm --prefix web ci
# 浏览器：npx --prefix web playwright install chromium 或设 ARROW_CHROME_BIN

npm --prefix web run build                       # 前端构建（提交前必跑）
python -m pytest tests/ -q                       # 40 项语义/集成测试

python -m arrow_toy.editor_server --port 8765    # 编辑器 http://127.0.0.1:8765
python -m arrow_toy.render --config configs/data_v1.yaml          # 生成 ~11 min
python -m arrow_toy.data.audit --root data/arrow_v1               # 审计（置 completed）
python -m arrow_toy.data.prepare_cache --root data/arrow_v1       # 缓存（版本绑定）

# P4 门槛（32 场景子集，与 runs/overfit32_v2 对应）：
head -32 data/arrow_v1/splits/overfit.txt > data/arrow_v1/splits/overfit32.txt
python3 -c "from arrow_toy.data.prepare_cache import build_cache; import pathlib; \
build_cache(pathlib.Path('data/arrow_v1'), 'overfit32')"
python -m arrow_toy.train --config configs/overfit32.yaml --run-dir runs/overfit32_v2
python -m arrow_toy.benchmark --config configs/quick.yaml --output runs/benchmark.json
python -m arrow_toy.train --config configs/quick.yaml --run-dir runs/baseline_s42
python -m arrow_toy.train --config configs/quick.yaml --override configs/ablations/no_query_sa.yaml --run-dir runs/no_sa_s42

# 最短评测链（只依赖上面已训练的 s42 两个 run）：
python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split val          # 阈值选择
python -m arrow_toy.evaluation.evaluate --run-dir runs/baseline_s42 --split test_clean   # 冻结阈值
python -m arrow_toy.evaluation.compare \
    --runs runs/baseline_s42 runs/no_sa_s42 --output runs/comparison
python -m arrow_toy.evaluation.visualize --run-dir runs/baseline_s42 --split val

# 完整对比实验（生成主表其余 run 后再执行；每条都需 ~9.5 分钟训练 + 评测）：
python3 - <<'PY'
import yaml, pathlib
for seed in (43, 44):
    cfg = yaml.safe_load(open("configs/quick.yaml")); cfg["seed"] = seed
    pathlib.Path("configs/experiments").mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(f"configs/experiments/quick_s{seed}.yaml", "w"))
cfg = yaml.safe_load(open("configs/quick.yaml")); cfg["train"]["lr"] = 0.0005
yaml.safe_dump(cfg, open("configs/experiments/quick_lr5e4.yaml", "w"))
PY
for S in 43 44; do
  python -m arrow_toy.train --config configs/experiments/quick_s$S.yaml --run-dir runs/baseline_s$S
  python -m arrow_toy.train --config configs/experiments/quick_s$S.yaml \
      --override configs/ablations/no_query_sa.yaml --run-dir runs/no_sa_s$S
  for RUN in baseline_s$S no_sa_s$S; do
    python -m arrow_toy.evaluation.evaluate --run-dir runs/$RUN --split val
    python -m arrow_toy.evaluation.evaluate --run-dir runs/$RUN --split test_clean
    python -m arrow_toy.evaluation.evaluate --run-dir runs/$RUN --split test_occlusion
  done
done
python -m arrow_toy.train --config configs/experiments/quick_lr5e4.yaml --run-dir runs/lr5e4_s42
python -m arrow_toy.evaluation.evaluate --run-dir runs/lr5e4_s42 --split val
python -m arrow_toy.evaluation.compare \
    --runs runs/baseline_s42 runs/no_sa_s42 runs/baseline_s44 runs/no_sa_s44 \
           runs/baseline_s43 runs/no_sa_s43 runs/lr5e4_s42 --output runs/comparison
```

产物位置：`data/arrow_v1/`（samples/splits/cache/audit.json/manifest.json）、`runs/<name>/`（resolved_config、environment（含 data_version 与 commit）、metrics.jsonl、result.json、last/best.pt、eval_*.json、predictions_*.jsonl、figures/）。

## 后续建议

1. 依据（未定证）：先跑完消融 E（同 BEV 特征 dense 头）——若 dense 头在同等预算下显著更好，则支持"sparse 头收敛是瓶颈"的假设，下一步做去噪 query/辅助监督的单变量实验；若 dense 同样不行，则瓶颈在特征，优先加深 backbone 或用 lite profile 换更多步数。
2. 换正常 4080 复测墙钟，恢复设计中途验证节奏。
3. 角误差不达标部分与位置精度耦合（中心 0.21–0.26 m）；提高位置精度（offset 回归或多尺度 BEV）可能同时改善两项。
