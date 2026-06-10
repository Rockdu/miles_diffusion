# 反向传播非确定性 —— 实锤因素汇总

**背景**：在 2 卡 OCR qwen-image GRPO 训练上,组 batch 重构分支与 docker 默认 miles-d 基线的前几步对齐后,reward 仍有 ~1% 的逐 rollout 漂移。逐层排查后定位:漂移**不是重构的 bug**,而是**训练反向传播的硬件级非确定性**经 RL 反馈放大。本文档汇总所有**已用代码/实测坐实(实锤)**的非确定性因素,附录给出每个因素的复现方法。

平台:NVIDIA B200 (sm_100, cc 10.0),bf16 前向,fp32 master,FSDP2,2 卡 colocate。

---

## 0. 总结论

| # | 实锤因素 | 现象 | 量级 | 可消除? |
|---|---|---|---|---|
| 1 | **torch SDPA 反向 dQ**(训练实际用的) | 同输入两次跑,dQ 不逐位一致 | maxrel ≈ 1.2e-3 | 仅 MATH 后端/全局确定模式(6.4× 慢) |
| 2 | **FlashAttention-4 反向 dQ** | 同上 | maxrel ≈ 8.8e-4 | `deterministic=True`(sm_100 有效) |
| 3 | FA2/FA3 反向 dQ | 同 atomic-dQ 模式 | — | `deterministic=True`(本机未装,按源码模式) |
| A | 训练前向 = SDPA NATIVE,**非** FA | 架构事实 | — | diffusers 不支持 FA4 |
| B | 训练**前向确定、反向不确定** | run1==run2 前向逐位同、grad_norm 不同 | grad_norm ~0.55% | 见 #1 |
| C | 重构前向**逐位等于基线** | bitwise parity run step-1 全等 | 残留=#1 的 floor | —(本就无 bug) |

**一句话**:reward 漂移的根 = **attention 反向 dQ 通过 atomicAdd 累加,完成顺序由硬件调度决定 → 每次跑梯度略不同**。这发生在**单卡内部的反向**,与卡数、与 2 卡 DP 归约(两数相加、精确无序)**无关**。重构本身的前向可逐位复刻基线,无 bug。

---

## 1. 因素 1 — torch SDPA 反向 dQ 非确定(训练实际路径)

**这是训练真正用的 kernel。** diffusers 的 Qwen transformer 走 `dispatch_attention_fn`,本环境解析为 `AttentionBackendName.NATIVE` = `F.scaled_dot_product_attention`(见因素 A)。

实测(B200,bf16,shape B,H,S,D=8,24,1024,128,同 seed 跑两次):
```
forward  bit-identical: True       ← 前向确定
dQ       bit-identical: False  maxrel=1.24e-03   ← 反向 dQ 非确定
dK       bit-identical: True
dV       bit-identical: True
```
- 只有 **dQ** 非确定(dQ 跨 key-block 用 atomic 累加);dK/dV 确定。
- `use_deterministic_algorithms(True, warn_only=True)` **不能修复**(warn_only 只警告,不强制换 MATH 后端)。
- 强制 `sdpa_kernel(MATH)` 可确定,但 **8.43ms vs 1.32ms(6.4×)**,且 O(S²) 显存。

→ 复现:[附录 A.1](#a1) `sdpa_backward_determinism.py` / [A.4](#a4) `sdpa_math.py`

---

## 2. 因素 2 — FlashAttention-4 反向 dQ 非确定(rollout 用的家族)

FA4 = pip 包 `flash-attn-4` 4.0.0b13,装成 `flash_attn/cute/`(CuTe-DSL Blackwell 版),API 为 `flash_attn.cute.flash_attn_func`。sglang rollout 引擎在 Blackwell 上用 FA4(日志 `Using FlashAttention (FA4 for blackwell)`)。

**源码实锤**(`flash_attn/cute/`):
- `interface.py:1238` — `_flash_attn_bwd(..., deterministic: bool = False, ...)` → 反向有 deterministic 开关,**默认 False**
- `interface.py:1286` — `assert deterministic is False, "deterministic backward not supported on SM 12.0"` → sm_120 上不支持确定反向
- `flash_bwd.py:1004` — `utils.atomic_add_fp32(acc_dQ_atomic[i], ...)` → **dQ 用 atomic add(默认非确定)**
- `flash_bwd.py:1135-1158` — **GQA(qhead_per_kvhead>1)时 dK/dV 也走 atomic add**

**经验实锤**(B200,同 seed 跑两次):
```
deterministic=False(默认):  dQ bit-identical=False  maxrel=8.80e-04 ; dK/dV 确定
deterministic=True:          dQ/dK/dV 全部 bit-identical ✓        ← flag 在 sm_100 有效
```

→ 结论:**FA4 默认反向非确定(dQ 恒 atomic;GQA 下 dK/dV 也 atomic),`deterministic=True` 可消除(sm_100 有效,sm_120 不支持)**。

→ 复现:[附录 A.2](#a2) `fa4_backward_determinism.py`;源码 [附录 A.5](#a5)

---

## 3. 因素 3 — FA2 / FA3 反向 dQ(同模式,本机未硬验)

FA2/FA3 用相同的 atomic-dQ 反向模式,均提供 `deterministic` flag(上游 Dao-AILab/flash-attention 文档)。本 B200 环境**未安装可运行的 FA2/FA3**(`flash_attn_interface`/`flash_attn_3` 不可导入,diffusers `_CAN_USE_FLASH_ATTN[_3]=False`),故仅按源码模式列出,**未在本机经验实锤**。前向均确定(故 rollout 纯推理可逐位复现)。

---

## 附录 A — 每个因素的实锤复现方法

所有脚本保存在复现分支 `verify/baseline-batch-parity` 的 `determinism/` 目录(见附录 B)。运行环境:容器 `sglang-miles-rockdu` @ ion-b200,`CUDA_VISIBLE_DEVICES=2`。

### <a id="a1"></a>A.1 SDPA 反向 dQ 非确定
```bash
ssh ion-b200 "docker exec sglang-miles-rockdu bash -lc \
  'cd /root/miles_verify && CUDA_VISIBLE_DEVICES=2 python determinism/sdpa_backward_determinism.py'"
```
原理:同 seed 构造 q/k/v,`F.scaled_dot_product_attention` 前向+反向跑两次,`torch.equal` 比 dQ/dK/dV。预期:forward 逐位同,dQ 不同(~1.2e-3),dK/dV 同。脚本同时验证 `use_deterministic_algorithms(True, warn_only=True)` 不修复。

### <a id="a2"></a>A.2 FA4 反向 dQ 非确定 + deterministic flag 有效
```bash
ssh ion-b200 "docker exec sglang-miles-rockdu bash -lc \
  'cd /root/miles_verify && CUDA_VISIBLE_DEVICES=2 python determinism/fa4_backward_determinism.py'"
```
原理:`from flash_attn.cute import flash_attn_func`,`deterministic=False/True` 各跑两次比 dQ/dK/dV。预期:False 时 dQ 不同(~8.8e-4),True 时全逐位同。

### <a id="a3"></a>A.3 训练实际后端 = SDPA NATIVE(非 FA)
```bash
ssh ion-b200 "docker exec sglang-miles-rockdu bash -lc \
  'cd /root/miles_diffusion && CUDA_VISIBLE_DEVICES=2 python determinism/check_diffusers_attn_backend.py'"
```
预期:`active_backend = AttentionBackendName.NATIVE`,`_CAN_USE_FLASH_ATTN=False`,`DIFFUSERS_ATTN_BACKEND=<unset>`,`deterministic_algorithms=False`。

### <a id="a4"></a>A.4 SDPA MATH 后端可确定但慢 6.4×
```bash
ssh ion-b200 "docker exec sglang-miles-rockdu bash -lc \
  'cd /root/miles_verify && CUDA_VISIBLE_DEVICES=2 python determinism/sdpa_math_backend.py'"
```
预期:DEFAULT 后端 dQ 非确定 ~1.3ms;MATH 后端 dQ 确定 ~8.4ms。

### <a id="a5"></a>A.5 FA4 源码 atomic / deterministic 信号
```bash
ssh ion-b200 "docker exec sglang-miles-rockdu bash -lc '
  C=/usr/local/lib/python3.12/dist-packages/flash_attn/cute
  grep -nE \"deterministic\" \$C/interface.py | head
  grep -niE \"atomic_add_fp32|acc_dQ_atomic|acc_dV_atomic|acc_dK_atomic\" \$C/flash_bwd.py'"
```
预期:interface.py 见 `deterministic: bool = False` 与 SM12.0 assert;flash_bwd.py:1004 见 dQ 的 `atomic_add_fp32`,1135-1158 见 GQA 下 dK/dV 的 atomic。

### <a id="a6"></a>A.6 端到端佐证 — 训练前向确定、反向不确定
两次同配置训练 run(默认 contiguous bf16,同 seed)的 step-1:loss/approx_kl/clipfrac/log_prob_new/log_prob_old/model_output_diff **逐位相同**,但 grad_norm 不同(`1.278063e-03` vs `1.270997e-03`,~0.55%)。日志:
`/root/miles_diffusion/logs/refactor_run.log`(run1)与 `.../refactor_run2.log`(run2),`grep "[train step 1] rollout=0"`。

### <a id="a7"></a>A.7 端到端佐证 — 重构前向逐位等于基线
验证分支 `--diffusion-train-dp-split baseline_stride --diffusion-train-cond-pad-window` 跑出的 step-1 与基线 `..._074438` 逐位相同(loss 4.470348e-07、approx_kl 1.202395e-09、clipfrac 0.05078125、model_output_mean_abs_diff 0.01966679 全等),仅 grad_norm 差 ~0.1%(即因素 1 的 floor)。脚本 `scripts/run-ocr-2gpu-bitwise-parity.sh`。数据级证明:`python test_dp_split_parity.py`。

---

## 附录 B — 复现分支与代码清单（保留）

**分支**：`verify/baseline-batch-parity`(worktree `/root/miles_verify`,基于 rebase 到 origin/main 的重构分支;**仅 B200 本地,未 push**）。

**代码改动（用于把重构组 batch 精确对齐 legacy,隔离"分组差异 vs bug"）**：
- `miles/utils/train_data_utils.py` — `TrainDataDPSplitter` 加 `baseline_stride` 模式(复刻 legacy `range(rank,N,dp)` 样本 stride 分配）
- `miles/utils/arguments.py` — flag `--diffusion-train-dp-split {contiguous,baseline_stride}`、`--diffusion-train-cond-pad-window`
- `miles/backends/fsdp_utils/actor.py` — `_train_core` 按 optim 窗口算 window joint-max,`_forward_train_pair_batch` 加 `cond_pad_len`;模块级 `_window_cond_pad_len`
- `miles/backends/fsdp_utils/configs/qwen_image.py` — `collate_cond_for_sample_batch` 加 `pad_to_len`
- `miles/ray/rollout.py` — 接 dp-split mode + `[reward stats]` stdout instrumentation(复刻基线日志解析）

**确定性复现脚本（`determinism/`）**：
- `sdpa_backward_determinism.py`（因素 1）
- `fa4_backward_determinism.py`（因素 2）
- `sdpa_math_backend.py`（因素 1 缓解 / A.4）
- `check_diffusers_attn_backend.py`（因素 A）

**数据级单测**：`test_dp_split_parity.py`（baseline_stride 复刻 legacy DP 分配，确定性、无 GPU）

**run 脚本（`scripts/`）**：`run-ocr-2gpu-baseline-parity.sh`、`run-ocr-2gpu-bitwise-parity.sh`

---

## 附录 C — 「能不能开 deterministic flag」与 backend 统一

- diffusers（本版）attention backend：FA2、FA3、原生 SDPA(cudnn/efficient/flash/math)、sage、flex、xformers、npu、xla、aiter ——**无 FA4**。本环境 `_CAN_USE_FLASH_ATTN[_3]=False`,实际只能用 SDPA。
- SDPA **无 per-call deterministic 参数**;要确定只能 MATH 后端(6.4× 慢 + O(S²) 显存)或全局 `use_deterministic_algorithms(True)`(同样逼 SDPA→math,且可能在别的 op 报错)。
- diffusers 的 deterministic flag **只接在 FA2/FA3 路径**,本环境用不到。
- **干净且统一 train/rollout 的路**:给 Qwen transformer 写一个调用 `flash_attn.cute.flash_attn_func(..., deterministic=True)` 的自定义 attention processor —— 已实测 sm_100 有效。代价:集成工作 + 确定反向的性能损耗。
- 取舍:~1e-3 dQ 抖动 → ~1% reward 散布属 RL 正常方差;determinism 主要换精确可复现/调试。对 train/rollout 一致性,更大杠杆是**统一 kernel(FA4)**而非 determinism 本身。
