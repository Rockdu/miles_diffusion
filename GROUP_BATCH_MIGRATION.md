# 组 batch 重构迁移方案

把「组 microbatch / train-data dispatch」从 `TrainRayActor` 上移到 `RolloutManager` 的重构,
采用**两阶段、按模型、CI 门控**的方式上线:先零行为变更地接入,再逐个模型切到理想默认策略。

---

## 0. 为什么这样做

重构把 train 侧的组 batch 逻辑换成了 flat-train-pair 设计。它与 legacy(grid tiling)在两处分组上有差异:

| 维度 | legacy 策略 | 重构理想默认 |
|---|---|---|
| DP 切分 | 按样本 stride `range(rank,N,dp)` | 连续块(更简单,locality 更好) |
| 文本 cond padding | 每个 optim window 统一 collate(window-max) | 每个 microbatch 本地 collate(local-max) |

**关键:这两处差异是「确定但不同」的分组选择,不是 bug。** 已实锤:重构在 parity 模式下,
组装进模型前的张量与 legacy **逐位相同**(见下方单测 + step-1 bitwise 实测)。

**为什么不能用端到端训练对齐做门控**:训练反向的 attention dQ 用 atomicAdd 累加,硬件级非确定
(见 `determinism/BACKWARD_NONDETERMINISM.md`),即使同配置两次跑 reward 也有 ~1% 散布。
所以**对齐验证必须放在确定的「数据组装层」做强单测**,而非 flaky 的 e2e 比较。

---

## 1. 已就绪的组件

**代码**(分支 `verify/baseline-batch-parity`):
- `RolloutTrainDataConverter` / `TrainDataDPSplitter`(`miles/utils/train_data_utils.py`)
- 两个分组旋钮:
  - `--diffusion-train-dp-split {contiguous, baseline_stride}`(`baseline_stride` = legacy 对齐)
  - `--diffusion-train-cond-pad-window`(开 = legacy 的 window padding)

**强单测**(`tests/test_grouping_parity.py`,确定、CPU、CI 友好):
- **L1 DP 切分**:`baseline_stride` == legacy `range(rank,N,dp)`;并断言 contiguous 确实不同(flag 非 no-op)
- **L2 Converter**:展平 train-pair 的 latent/next_latent/timestep/log_prob_old/advantage/debug
  == 直接按 `sde_step_indices` 索引 trajectory;sample-major 顺序、scheduler meta 正确
- **L3 Cond padding**:`collate(pad_to_len=window_max)` == legacy「window-collate 再 tile-slice」逐位相同;
  并断言不开 flag 时 seq_len 确实不同
- **L4 黄金等价**(待补,见 §4):整条 refactor 组装 == legacy `_build_train_grids`→`_forward_tile` 取输入

运行:`cd <repo> && PYTHONPATH=<repo> python -m pytest tests/test_grouping_parity.py -q`

---

## 2. 第一阶段 — 接入,默认 legacy 对齐(零行为变更)

**目标**:合入重构代码,但**默认走 legacy 对齐策略**,所有现有模型的训练在组装层与改前**逐位一致**。

- **默认值**:`dp_split = baseline_stride`、`cond_pad_window = True`(= legacy 对齐)。
- **按模型默认机制**(建议实现):在 `TrainPipelineConfig` 加两个字段,CLI 用 sentinel 默认 `auto` 表示「听 config 的」:
  ```python
  class TrainPipelineConfig:
      train_dp_split: str = "baseline_stride"      # 第一阶段:全模型 = legacy
      train_cond_pad_window: bool = True
  ```
  actor/rollout:`args.X if 用户显式传了 else config.X`。
- **CI 门控**:L1–L4 单测每个 PR 必跑(模型无关的数据层测试,**永久守住 legacy 等价契约**)。
- **结果**:现有训练**无行为变化**(组装层逐位一致;残留只有 legacy 本就有的 bf16 反向噪声)。

---

## 3. 第二阶段 — 按模型逐个切到理想默认

对每个模型(qwen_image、sd3、…),**一次一个**:

1. **CI 健康检查**:用理想默认(`contiguous` + per-microbatch padding)对该模型跑 smoke/短训
   —— 断言能跑通、loss/reward 合理、无 NaN、microbatch count 跨 DP 一致(`validate_same_microbatch_counts_across_dp`)。
2. **(可选强检查)**:该模型的 cond 结构是否需要 padding 对齐的等价处理(L3 的 per-model 版)。
3. **翻转**该模型 config 的 `train_dp_split → contiguous`、`train_cond_pad_window → False`。
4. **保留** legacy 模式(flag 可显式回退),用于精确复现/调试与回滚。

**门控的依据**:
- legacy 等价由 L1–L4 单测**持续**保证(确定、与模型无关)。
- 切到理想默认是**有意的分组变更**(microbatch 组成不同 → 数值不同但数学等价的梯度),
  所以这一步**不是去复现 legacy**,而是确认该模型在新分组下训练健康。
- **不要求 e2e bitwise**——那被 bf16 反向非确定性污染,不是有效门控。

---

## 4. 待补:L4 黄金等价测试(最强门控,建议加进第一阶段 CI)

把 legacy 的输入组装抽成纯函数后,直接对比两条路径喂给模型的张量:

- 抽取 legacy `_build_train_grids` + `_forward_tile` 的**输入组装**(latents_flat/timesteps/log_prob_old/advantage/cond)为可测函数(顺带是个让组装可测的小重构)。
- 构造固定 mock samples → 跑 legacy 组装(stride DP + window cond)与 refactor 组装(baseline_stride + cond-pad-window)→ per-microbatch/tile **`torch.equal`**。
- 这会把「step-1 bitwise」从一次性的训练实测,固化成 CI 里确定可跑的单测。

---

## 5. 风险与回滚

- **第一阶段**:legacy 对齐默认 + 单测 → 可证无行为变更;风险极低。
- **第二阶段**:按模型门控、flag 可回退、一次一个 → 影响面受控,任意模型可单独回滚到 legacy。
- **已知非阻塞项**:重构训练路径暂未接入 main 新增的 diffusion KL loss(`--diffusion-kl-beta`);
  OCR 等 `kl_beta=0` 配置不受影响,需要时回填。

---

## 附:相关文件(分支 `verify/baseline-batch-parity`,B200 本地)

- 测试:`tests/test_grouping_parity.py`(L1–L3)、`test_dp_split_parity.py`(L1 独立版,可并入)
- 代码:`miles/utils/train_data_utils.py`、`miles/backends/fsdp_utils/{actor.py,configs/qwen_image.py}`、`miles/utils/arguments.py`、`miles/ray/rollout.py`
- run 脚本:`scripts/run-ocr-2gpu-{baseline,bitwise}-parity.sh`
- 背景:`determinism/BACKWARD_NONDETERMINISM.md`(为什么 e2e 不能做门控)
