# Grid-Aware Pricing 运行与 Case Study 指南

## 1. 工程范围

根工程已经实现完整 York 仿真与实验链路：

1. ZIP 内六个 CSV 与 `scenario_config.yaml` 的直接读取和 provenance 校验；
2. BPR 路由、OD→hub→destination 路径和 candidate mask；
3. deterministic expected demand 与 stochastic demand；
4. queue-v2 跨时段 residual wait / queued-energy carryover；
5. PV、grid、battery、SOC 约束下的 LP dispatch；
6. 隐藏 outside-cost 的聚合选择推断；
7. MAPPO、IPPO、Fixed Tariff、Myopic Local Pricing；
8. budgeted centralized coordinate-search reference；
9. benchmark、sensitivity、ablation、Fig.3–8 和 Table I 报告。

参考论文 PDF 只包含到 Section IV 和 Fig.2。根工程生成的 Fig.3–8 与 Table I 是计划中的 Section V 扩展，manifest 中固定标记：

```text
planned_extension_not_present_in_source_pdf
```

不能把生成数字描述成源论文已报告的结果。

## 2. 根工程安装

根工程要求 Python 3.12+ 和科学计算/机器学习依赖：

```bash
cd /efs/guyh29/grid_aware_pricing
python3.12 -m pip install -e '.[test]'
grid-pricing --help
```

未安装 CLI 时可以使用：

```bash
PYTHONPATH=src python -m grid_aware_pricing.cli --help
```

后文均使用 `grid-pricing`。

## 3. Queue-v2 模型语义

York 当前 schema：

```text
environment_schema_version = 2
queue_semantics = dynamic_fluid_carryover_v1
observation_dim = 16
global_state_dim = 136
```

每站每期计算：

- historical equivalent vehicles；
- total pending vehicles；
- admission ratio / pressure；
- admitted vehicles；
- queued energy at period start；
- pending / admitted / next queued energy；
- next residual wait；
- vehicle-backlog equivalence 和 energy conservation residual。

Base case 不执行 finite waiting-space truncation。只有 admitted energy 进入 dispatch；dispatch 后的 unmet energy 不返回 queue。历史 backlog 不重复计入本期 choice counts。

旧 York checkpoint 缺少 queue-v2 metadata 或网络维度不一致，加载时会明确要求重新训练。旧 York CSV 也不能与新结果混合生成报告。

## 4. 数据与 fixed-tariff sanity

运行：

```bash
grid-pricing validate-data \
  --config configs/york_smoke.yaml \
  --output-dir outputs/york_queue_v2/validation
```

成功标准：

```text
12/12 structural checks passed
Total potential demand:                 382.830
Peak hourly demand:                     103.800
Maximum residual wait:                   75.996 min
Peak queued energy:                     649.458 kWh
Minimum admitted-load supply margin:    152.400 kWh
Queue cleared from: 2024-09-20 20:00:00
```

这些是 queue implementation 的 golden checks，不是策略性能。20:00 起 residual wait 和 queued energy 必须清零并保持为零。

当前 minimum admitted-load supply margin 为正，因此 base case 不支持“Proposed 减少 unmet energy”的结论。若以后研究该结论，应新增显式校准 config，并让所有方法共享同一环境；不得修改 ZIP 内原始 CSV 来制造结果。

## 5. 单方法训练和 checkpoint 验证

最小 Proposed smoke：

```bash
grid-pricing train \
  --config configs/york_smoke.yaml \
  --method proposed \
  --seed 5 \
  --episodes 3 \
  --episodes-per-update 1 \
  --output-dir outputs/york_queue_v2/smoke/training/proposed/seed_5
```

主要输出：

- `training_metrics.csv`
- `period_hub.csv`
- `best_checkpoint.pt`
- `final_checkpoint.pt`
- `checkpoint.pt`
- `run_manifest.json`
- `ppo_diagnostics.png`

`run_manifest.json` 应记录 output schema、environment schema、queue semantics 和 York ZIP hash。

冻结 checkpoint 评估：

```bash
grid-pricing evaluate \
  --config configs/york_smoke.yaml \
  --method proposed \
  --checkpoint outputs/york_queue_v2/smoke/training/proposed/seed_5/best_checkpoint.pt \
  --episodes 3 \
  --seeds 101 202 303 \
  --output-dir outputs/york_queue_v2/smoke/evaluation/proposed
```

默认 evaluation 使用 deterministic Beta mean，并冻结 lower-layer estimator；每个 episode 从 checkpoint 中相同的 outside estimate 开始。只有机制演示才使用 `--online-lower-layer`。

## 6. 六方法 benchmark

先分别训练 Proposed、MAPPO-No-Inference 和 IPPO：

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
```

运行短 benchmark：

```bash
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
```

方法包括：

1. Fixed Tariff；
2. Myopic Local Pricing；
3. IPPO；
4. MAPPO-No-Inference；
5. Proposed Dual-Layer MAPPO；
6. Centralized coordinate-search reference。

第 6 项不是 `Centralized Welfare Oracle`。它是预算受限、非精确、非 upper bound 的 reference。报告指标为：

```text
centralized_reference_difference = reference return - method return
```

`exact_oracle_gap` 不可用并保持 NaN。

加入 `--stochastic` 才会运行随机需求。正式 case study 应使用 stochastic evaluation、共同 scenario seeds、足够多 evaluation episodes，并为三个学习方法提供至少 3 个独立 training seeds。Deterministic benchmark 只用于机制和程序检查，报告会自动标为 partial。

## 7. Sensitivity 和 ablation

### Sensitivity

```bash
grid-pricing sensitivity \
  --config configs/york.yaml \
  --checkpoint PATH_TO_PROPOSED_CHECKPOINT \
  --method proposed \
  --episodes 30 \
  --stochastic \
  --output-dir outputs/york_queue_v2/full/sensitivity
```

内置参数网格：

| Axis | Levels |
|---|---|
| inverse_cost_sensitivity | 0.22, 0.30, 0.55 |
| demand_multiplier | 0.85, 1.00, 1.15 |
| grid_cap_multiplier | 0.85, 1.00, 1.15 |
| true_outside_cost | 14.0, 16.5, 19.0 GBP |

Sensitivity 输出保留完整 period-hub queue trajectory，而不是只保存 episode 平均值。

### Ablation

```bash
grid-pricing ablation \
  --config configs/york.yaml \
  --checkpoint no_traffic=PATH_TO_NO_TRAFFIC_CHECKPOINT \
  --checkpoint no_energy=PATH_TO_NO_ENERGY_CHECKPOINT \
  --checkpoint known_preference=PATH_TO_KNOWN_PREFERENCE_CHECKPOINT \
  --episodes 30 \
  --stochastic \
  --output-dir outputs/york_queue_v2/full/ablation
```

每个 ablation 必须使用按对应 method 训练的 checkpoint。缺失或不兼容 checkpoint 会记录为 missing/incompatible，不会用 Proposed checkpoint 代替。

## 8. 报告生成、章节和缺失证据

```bash
grid-pricing report \
  --config configs/york.yaml \
  --results-dir outputs/york_queue_v2/full \
  --output-dir outputs/york_queue_v2/full/report
```

根工程报告对应关系：

| 编号 | 计划章节 | 标题 | 主要证据要求 |
|---|---|---|---|
| Fig.3 | Planned Section V-A | York study area and infrastructure | ZIP map/ATC/OD/hub data |
| Fig.4 | Planned Section V-B | Learning convergence and preference estimation | 3 个充分训练的 Proposed seeds |
| Fig.5 | Planned Section V-C | Dynamic prices and demand allocation | Proposed online mechanism replay |
| Fig.6 | Planned Section V-C/V-D | Cross-period queue and service quality | queue-v2 trajectory、3 seeds、stochastic evaluation |
| Fig.7 | Planned Section V-D | Energy dispatch and grid interaction | pending/admitted/served/unmet energy 与 dispatch fields |
| Fig.8 | Planned Section V-E | Access/detour–profit trade-off | 六方法、queue/wait encoding、signed reference difference |
| Table I | Planned Section V-E | Main benchmark results | queue-v2、energy、access、inference、reference metrics |

Fig.6 展示 mean/p95/max wait、15-minute violation、system queued-energy carryover、minimum admission ratio 和 peak admission pressure。

Fig.7 明确区分 pending、admitted、served 和 unmet energy。

Fig.8 的点大小表示 peak queued energy，颜色表示 15-minute wait violation rate；独立 panel 展示 signed centralized-reference difference，并明确不是 oracle gap。

Table I 包含 pending/admitted requests、admission ratio、admitted/pending full-service ratio、peak/final queue、admission pressure、queue clearance 和 conservation residual。

`figures_manifest.json` 会逐图记录：

- 对应章节、图号和标题；
- source artifacts；
- complete / partial / blocked；
- missing data、experiments、seeds 或 evidence；
- limitations；
- output/environment schema、queue semantics 和 ZIP hash；
- benchmark artifact validation status。

报告只接受 manifest 绑定的 `benchmark_episodes.csv` 和 `benchmark_period_hub.csv`。schema、queue semantics、ZIP hash、CSV hash、行数或列顺序任一不匹配时，Fig.5–8 和 Table I 不会静默使用该 bundle。

## 9. Standalone ZIP 参考程序

根目录 `york_ev_case_study_runnable.zip` 内含一个独立固定价 reference。它只用于验证 packaged queue equations 和 golden CSV，不是完整 MARL 工程。

解压后运行：

```bash
unzip york_ev_case_study_runnable.zip -d /tmp/york_ev_case_study_runnable
cd /tmp/york_ev_case_study_runnable
python3 run_case_study.py
python3 -m unittest discover -s scripts -p 'test_*.py'
python3 scripts/smoke_test.py
```

只有这个 standalone baseline 可以使用 Python 3.9+ 且零第三方依赖。其 SVG Fig.1–4 与根工程 planned Section V Fig.3–8 是两套不同编号，不能混用。

## 10. 输出管理与结论限制

- Queue-v2 新结果统一写入 `outputs/york_queue_v2/`；
- 不删除、不覆盖旧 outputs；
- 不修改 ZIP 内六个输入 CSV；
- 不把 smoke checkpoint 当作收敛模型；
- 不把 weighted hub-profit objective 称为一般社会福利；
- 不把 coordinate search 称为 exact oracle 或 upper bound；
- 不在当前正供给余量 base case 中声称 Proposed 降低 unmet energy；
- 不将 deterministic repeated seeds 描述为独立 stochastic samples。
