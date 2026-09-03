# Grid-Aware Pricing for York EV Charging Hubs

完整中文运行与案例说明见 [`RUN_AND_CASE_STUDY_GUIDE_CN.md`](RUN_AND_CASE_STUDY_GUIDE_CN.md)。

本仓库实现 York EV 充电站定价 case study 的完整仿真与实验流程，包括 BPR 路由、随机需求、跨时段 fluid queue、LP 能源调度、outside-cost 推断、MAPPO/IPPO、benchmark、sensitivity，以及计划扩展的 Fig.3–8 和 Table I。

## 证据边界

参考论文 PDF 的正文到 Section IV 和 Fig.2 为止。本工程生成的 Fig.3–8 与 Table I 是按照 case-study plan 实现的 **planned Section V extension**，不属于源论文已有图表。所有报告条目均写入：

```text
planned_extension_not_present_in_source_pdf
```

生成结果也不能替代正式实验：发布级结论至少需要充分训练的 3 个独立 training seeds，以及共享 scenario seeds 的随机多场景冻结评估。短 smoke run 只用于验证程序链路。

## 环境与安装

根工程要求 Python 3.12 或更高版本，并依赖 NumPy、SciPy、PyTorch、PyYAML、pandas、Matplotlib 和 NetworkX：

```bash
cd /efs/guyh29/grid_aware_pricing
python3.12 -m pip install -e '.[test]'
```

安装后使用统一 CLI：

```bash
grid-pricing --help
```

如果不安装 editable package，可在仓库根目录使用：

```bash
PYTHONPATH=src python -m grid_aware_pricing.cli --help
```

## York 数据与 queue-v2

根工程直接从 `york_ev_case_study_runnable.zip` 读取六个运行 CSV 和 `scenario_config.yaml`，不会解压、复制或修改源数据。当前 York 环境使用：

- `environment_schema_version: 2`
- `queue_semantics: dynamic_fluid_carryover_v1`
- ZIP SHA-256：由加载器计算并写入 config、checkpoint 和结果 manifest

Queue-v2 的主要语义：

- residual wait 和 queued energy 跨时段结转；
- 历史 backlog 与本期新 arrivals 合并为 pending work；
- base case 不执行 finite waiting-space 截断；
- `queue_capacity_vehicles` 仅保留为 optional-extension provenance；
- 只有 admitted energy 进入 LP dispatch；
- admitted 后未供给的能量记为 unmet，不返回 queue；
- 历史 backlog 不作为新的用户选择重复进入 outside-cost 推断。

York local observation 为 16 维，global state 为 136 维。旧 York checkpoint 与 queue-v1 输出不兼容，必须重新训练；synthetic legacy checkpoint 在网络维度匹配时仍可加载。

## 快速验证

### 1. 运行测试

```bash
pytest -q tests
```

### 2. 验证 York 数据和 fixed-tariff sanity

```bash
grid-pricing validate-data \
  --config configs/york_smoke.yaml \
  --output-dir outputs/york_queue_v2/validation
```

`validate_data.json` 应满足 12/12 structural checks，并包含以下 queue-v2 golden values：

| 指标 | 期望值 |
|---|---:|
| Total potential demand | 382.830 |
| Peak hourly demand | 103.800 |
| Maximum residual wait | 75.996 min |
| Peak queued energy | 649.458 kWh |
| Minimum admitted-load supply margin | 152.400 kWh |
| Queue cleared from | 2024-09-20 20:00:00 |

这些值是固定价 deterministic replay 的程序校验值，不是 proposed policy 的性能结果。正的 admitted-load supply margin 表明当前 base case 不支持“Proposed 降低 unmet energy”的结论；如需研究该效果，应新增明确校准的 config，而不是修改原始 CSV。

## Smoke 工作流

以下命令用于验证训练、checkpoint、benchmark 和报告链路，不用于论文结论：

```bash
ROOT=outputs/york_queue_v2/smoke

for METHOD in proposed mappo_no_inference ippo; do
  grid-pricing train \
    --config configs/york_smoke.yaml \
    --method "$METHOD" \
    --seed 5 \
    --episodes 3 \
    --episodes-per-update 1 \
    --output-dir "$ROOT/training/$METHOD/seed_5"
done

grid-pricing benchmark \
  --config configs/york_smoke.yaml \
  --output-dir "$ROOT/benchmark" \
  --checkpoint proposed="$ROOT/training/proposed/seed_5/best_checkpoint.pt" \
  --checkpoint mappo_no_inference="$ROOT/training/mappo_no_inference/seed_5/best_checkpoint.pt" \
  --checkpoint ippo="$ROOT/training/ippo/seed_5/best_checkpoint.pt" \
  --seeds 101 202 303 \
  --eval-episodes 3 \
  --reference-budget 10 \
  --unilateral-budget 0

grid-pricing report \
  --config configs/york_smoke.yaml \
  --results-dir "$ROOT" \
  --output-dir "$ROOT/report"
```

Deterministic benchmark 会在 Fig.6–8 和 Table I 中被标记为 partial，因为它不包含随机多场景证据。

## 正式训练与评估

`configs/york.yaml` 默认配置 3 个 training seeds：11、29、47。建议每种训练方法分别保存到独立目录：

```bash
ROOT=outputs/york_queue_v2/full

for METHOD in proposed mappo_no_inference ippo; do
  for SEED in 11 29 47; do
    grid-pricing train \
      --config configs/york.yaml \
      --method "$METHOD" \
      --seed "$SEED" \
      --output-dir "$ROOT/training/$METHOD/seed_$SEED"
  done
done
```

训练完成后，用冻结 checkpoint、共同 scenario seeds 和 stochastic demand 做 benchmark：

```bash
grid-pricing benchmark \
  --config configs/york.yaml \
  --output-dir "$ROOT/benchmark" \
  --checkpoint proposed=PATH_TO_PROPOSED_CHECKPOINT \
  --checkpoint mappo_no_inference=PATH_TO_MAPPO_NO_INFERENCE_CHECKPOINT \
  --checkpoint ippo=PATH_TO_IPPO_CHECKPOINT \
  --seeds 101 202 303 \
  --eval-episodes 30 \
  --reference-budget 500 \
  --unilateral-budget 100 \
  --stochastic
```

六方法 benchmark 包括：

1. Fixed Tariff
2. Myopic Local Pricing
3. IPPO
4. MAPPO-No-Inference
5. Proposed Dual-Layer MAPPO
6. Centralized coordinate-search reference

第 6 项是预算受限的 centralized reference，不是 exact oracle，也不是 upper bound。报告使用：

```text
centralized_reference_difference = reference return - method return
```

`exact_oracle_gap` 保持为空。

## Sensitivity 与 ablation

Proposed checkpoint 可用于 sensitivity：

```bash
grid-pricing sensitivity \
  --config configs/york.yaml \
  --checkpoint PATH_TO_PROPOSED_CHECKPOINT \
  --method proposed \
  --episodes 30 \
  --stochastic \
  --output-dir "$ROOT/sensitivity"
```

内置水平包括：

- inverse-cost sensitivity：0.22、0.30、0.55
- demand multiplier：0.85、1.00、1.15
- grid-cap multiplier：0.85、1.00、1.15
- hidden true outside cost：14.0、16.5、19.0 GBP

Ablation 需要各自训练的 `no_traffic`、`no_energy` 和 `known_preference` checkpoint，不能用 Proposed checkpoint 冒充。

## 报告输出与对应章节

```bash
grid-pricing report \
  --config configs/york.yaml \
  --results-dir "$ROOT" \
  --output-dir "$ROOT/report"
```

| 输出 | 对应计划章节 | 标题 |
|---|---|---|
| Fig.3 | Planned Section V-A | York study area and infrastructure |
| Fig.4 | Planned Section V-B | Learning convergence and preference estimation |
| Fig.5 | Planned Section V-C | Dynamic prices and demand allocation |
| Fig.6 | Planned Section V-C/V-D | Cross-period queue congestion and service quality |
| Fig.7 | Planned Section V-D | Energy dispatch and grid interaction |
| Fig.8 | Planned Section V-E | Access/detour–profit trade-off and signed centralized-reference difference |
| Table I | Planned Section V-E | Main benchmark results |

`figures_manifest.json` 记录每张图的章节、标题、源文件、状态、缺失证据和限制。若 training seeds、stochastic evaluation、queue 字段或 artifact provenance 不足，图会标记 partial/blocked 并带红色水印。

Benchmark bundle 必须通过以下一致性检查才能进入 Fig.5–8 和 Table I：

- output schema version；
- York environment schema；
- queue semantics；
- ZIP SHA-256；
- CSV SHA-256、行数和列顺序。

当前发布包不包含旧 queue-v1 结果；manifest 校验也会阻止历史 queue-v1 结果被静默混入 queue-v2 报告。

## Standalone ZIP 固定价参考

`york_ev_case_study_runnable.zip` 内还有一个零第三方依赖的固定价参考程序。它与根工程是两个不同入口。只有在解压 ZIP 后，才运行：

```bash
unzip york_ev_case_study_runnable.zip -d /tmp/york_ev_case_study_runnable
cd /tmp/york_ev_case_study_runnable
python3 run_case_study.py
```

该程序生成 standalone baseline 的 SVG Fig.1–4，不是根工程 planned Section V 的 Fig.3–8。零第三方依赖和 Python 3.9+ 的说明仅适用于这个 standalone baseline，不适用于根工程。

## 输出目录约定

Queue-v2 新结果应写入 `outputs/york_queue_v2/` 下的独立目录。保留的 `verification_20260901` 是 smoke 验证包，不应被新训练覆盖；manifest 校验会阻止不兼容 benchmark bundle 进入新报告。
