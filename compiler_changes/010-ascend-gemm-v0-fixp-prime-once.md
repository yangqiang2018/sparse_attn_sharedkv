# 010 · Ascend `gemm_v0_fixp` 加 `prime_drain`：L0AB `M_MTE1` flag 一次 prime（= `AllocEventID`/`FreeEventID`）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | `wip/gemm-v0-fixp-prime-once`（**独立基于 `ascendc_pto`**，含 001–009） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`gemm_v0_fixp` 加 `bool prime_drain=true`，M_MTE1 prime/drain gate 在 `if(prime_drain)`）、`tilelang/language/ascend.py`（绑定加 `prime_drain`）、`src/target/codegen_ascend.cc`（`GemmFixpOpCodegen` emit args[9]）、`src/op/ascend.cc`（`ascend_gemm_v0_fixp` `set_num_inputs` 9→10） |
| **是否必须** | 是 —— 忠实复刻要求 QK/PV 的 L0AB ping-pong flag 由调用方在整条 cube 循环**一次** prime（= `AllocEventID`）、**一次** drain（= `FreeEventID`），而不是每个 `gemm_v0_fixp` 调用自 prime/drain；这是删 `DEBUG_SERIAL` barrier、拿回跨调用/跨迭代 fixpipe∥mma 重叠的前置 |
| **是否兼容** | 是 —— `prime_drain=true` 默认 → PV（008）及一切现有 caller 逐字节不变（仍自 prime/drain）；回归 example 走 `gemm_v0`（非 fixp），不受影响 |
| **状态** | ⏳ 待 NPU 验证（SWA 5/5 + 回归两 example）后合入 `ascendc_pto`。这是「忠实 cube 续做」增量 2（增量 1 = 合并共享 cL0，已 5/5 PASS，纯内核无编译器改动） |

---

## 1. 算子为什么需要它

参考 `SWACubeBlock`（`swa_block_cube.h`）把 L0A/L0B ping-pong 的两个 `M_MTE1`
flag（`L0AB_EVENT0/1`）当作**整条 kernel 生命周期**的资源：`AllocEventID()`
（:225-226）在所有 `ComputeMm1`/`ComputeMm2` 调用**之前一次性** `SetFlag<M_MTE1>`
两槽，`FreeEventID()`（:239-240）在所有调用**之后一次性** `WaitFlag<M_MTE1>` 两槽。
每个 `ComputeMm1`/`ComputeMm2` **内部从不碰 L0AB flag 的生命周期**——只在 kL0 tile
循环里逐槽 `WaitFlag<M_MTE1>(abL0BufIter%2)` ... `SetFlag<M_MTE1>(...)`（:563/583/845/916）。

我现在的 `gemm_v0_fixp`（004/008/009）相反：**每次调用**开头自 prime（`SetFlag<M_MTE1>(L0AB_EVENT)`、
`+1`），结尾自 drain（`WaitFlag<M_MTE1>(L0AB_EVENT)`、`+1`）。背靠背的 QK 调用、PV
调用因此在每个调用边界都重新 prime+drain 一遍 L0AB 环 → 一处不忠实，也是 QK↔PV 之间
多余的 L0AB 串行边界，挡住「QK 末 fixpipe ∥ PV 首 mma」的核内重叠。

## 2. 现象 / 缺口

`gemm_v0_fixp` 是自包含原语：自 prime/drain L0AB 环。两个独立调用（QK + PV）各做一遍
prime/drain，无法表达参考「调用方一次 prime、一次 drain，原语只逐槽 Wait/Set」的结构。

## 3. 根因

L0AB `M_MTE1` 环的 prime/drain 写死在模板首尾（`common.h` `gemm_v0_fixp`），调用方无从
把它提到 cube 循环外只做一次。

**★event-id 冲突（一次 prime 暴露的隐患，已修）★**:模板原本把 `M_MTE1`/`MTE1_M` 的
L0AB ping-pong 和首尾的 `MTE2_MTE1`/`MTE1_MTE2` 自配对 fence **共用同一个 event id
`L0AB_EVENT=0`**。在原「每调用 prime/drain」下没问题——`M_MTE1(0)` 在调用尾被 drain 清空,
之后才用 `MTE1_MTE2(0)`,从不并发。但一旦把 prime/drain 提到循环外、`M_MTE1(0/1)` **整条
cube 循环常驻 SET**,下一个 gemm 调用首部的 `SetFlag<MTE2_MTE1>(0)` 就和常驻的 `M_MTE1(0)`
**撞同一个物理 flag 寄存器** → 同步错乱/卡死。参考正是因此把 `M_MTE1` 放在专属
`EVENT_ID3/4`(`L0AB_EVENT0/1`),与 L1 flag(`EVENT_ID1/2/5/6/7`)**完全不相交**。

## 4. 为什么不能在内核侧解决

prime/drain 的 `SetFlag`/`WaitFlag<M_MTE1>` 在**模板内部**首尾发出；内核调不进模板内部去
删它们。逐槽的 `WaitFlag<M_MTE1>(L0AB_EVENT+pp)` 也在模板的 tile 循环里。要让调用方接管
L0AB 环的生命周期，必须在原语上开一个开关（`prime_drain`），并让模板在关时跳过首尾
prime/drain。一次 prime/drain 本身（`SetFlag<M_MTE1>` 两槽）可以用现有 `T.set_flag("m","mte1",ev)`
在内核发出（`.upper()` → `HardEvent::M_MTE1`），无需新原语。

## 5. 修法

**`common.h` `gemm_v0_fixp`**：加 `bool prime_drain = true`（尾随默认参）。
- 新增 `const uint32_t mmEv = prime_drain ? L0AB_EVENT : L0AB_MM_EVENT;`（`L0AB_MM_EVENT=4`，
  新常量）。**所有** `M_MTE1`/`MTE1_M` 的 prime/drain/逐槽 Wait/Set 都改用 `mmEv(+pp)`。
  `prime_drain=true` → `mmEv=L0AB_EVENT(=0)`，逐字节不变；`prime_drain=false` → `mmEv=4`，
  把常驻的 `M_MTE1(4/5)` 与首尾自配对 fence(留在 `L0AB_EVENT=0`)错开,根治 §3 的 event-id
  冲突。`{4,5}` 在 SWA 内核空闲(KV flag 用 `{2,3}`、fence 用 `{0,1}`)。
- 首部 `SetFlag<HardEvent::M_MTE1>(mmEv)`、`(mmEv+1)` gate 在 `if (prime_drain)`。
- 尾部 `WaitFlag<HardEvent::M_MTE1>(mmEv)`、`(mmEv+1)` gate 在 `if (prime_drain)`。
- 首部 `MTE2_MTE1` 自配对 fence、尾部 `MTE1_MTE2` 自配对 fence **保留在 `L0AB_EVENT=0`**
  （护 A 矩阵 q_l1/p_l1 的 MTE2，留待增量 3 的 QP/KV L1 环 + 反向 flag 替代）。
- 逐槽 tile 循环里的 `WaitFlag<M_MTE1>(mmEv+pp)` / `SetFlag<MTE1_M>(mmEv+pp)` /
  `WaitFlag<MTE1_M>(mmEv+pp)` / `SetFlag<M_MTE1>(mmEv+pp)`——护 L0A/L0B buffer 复用，随 `mmEv`
  迁移（`M_MTE1` 与 `MTE1_M` 是同一 L0AB 槽的正/反向，必须共用 id；tile 内顺序化,不并发）。

**内核（算子仓 `kernel.py`，`prime_drain=False` 模式）**：
- cube `with T.Scope("C")` 的 `for j` 之前：`T.set_flag("m","mte1",0)`、`T.set_flag("m","mte1",1)`
  （= `AllocEventID` 的 M_MTE1 两槽）。
- `for j` 之后：`T.wait_flag("m","mte1",0)`、`T.wait_flag("m","mte1",1)`（= `FreeEventID`）。
- QK、PV 的 `T.gemm_v0_fixp(...)` 加 `prime_drain=False`。

**为什么 `abL0BufIter` 不需要持久**：SWA 的 cadence 让每个 `gemm_v0_fixp` 调用恰好把
`abL0BufIter` 推进 **4**（偶数）——QK 4 个 kL0 tile、PV 4 个 N-tile 各 1 个 kL0 tile。
所以每个调用的起点 `abL0BufIter % 2` 恒为 0，模板里**局部 `tileIdx` 每调用从 0 起 ≡
持久 `abL0BufIter`**。`cL0BufIter` 的跨迭代复用仍由 `DEBUG_SERIAL` barrier
（`PipeBarrier<PIPE_ALL>`，drain 全 pipe）掩盖；持久 `cL0BufIter` 留待增量 3 删 barrier 时。

**绑定/codegen/`set_num_inputs`** 照既有范式透传（尾随默认参，arg[9]=prime_drain；
`set_num_inputs` 9→10）。

## 6. 忠实性（对照 `swa_block_cube.h`）

| TileLang | Ascend C 参考 |
|---|---|
| 内核 cube 循环前 `set_flag("m","mte1",0/1)` 一次 | `AllocEventID()` `SetFlag<M_MTE1>(L0AB_EVENT0/1)`（:225-226，整条 kernel 一次） |
| 内核 cube 循环后 `wait_flag("m","mte1",0/1)` 一次 | `FreeEventID()` `WaitFlag<M_MTE1>(L0AB_EVENT0/1)`（:239-240） |
| `gemm_v0_fixp(prime_drain=false)` 内部不 prime/drain L0AB | `ComputeMm1`/`ComputeMm2` 内部只逐槽 `Wait/Set<M_MTE1>`（:563/583/845/916），不碰生命周期 |
| 模板逐槽 `Wait/Set<M_MTE1>(L0AB_EVENT+pp)` | `WaitFlag/SetFlag<M_MTE1>(Mte1MmABEventId(abL0BufIter%2))` |

## 7. 兼容性证据

1. **默认值保兼容**：`prime_drain=true` 默认 → 模板首尾仍自 prime/drain，PV（008）及任何
   现有/未来 caller 逐字节不变。
2. **回归**：`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 走 `gemm_v0`（非
   `gemm_v0_fixp`），完全不受影响；仍需复跑确认不破坏其它原语。
3. **唯一新 caller**：SWA 的 QK/PV 是唯一传 `prime_drain=false` 的 caller；M_MTE1 event 0/1
   在内核 scope 空闲（KV pipe-flag 用 event 2/3 的 `MTE2_MTE1`，不同 `HardEvent` 通道）。

## 8. 必要性

忠实复刻 `AllocEventID`/`FreeEventID` 的 L0AB 环生命周期；并且是删 `DEBUG_SERIAL` barrier
（增量 3，拿回跨迭代 fixpipe∥mma 重叠 = prefill 提速的主路）的前置——barrier 删掉后，L0AB
环必须由调用方在循环外一次 prime/drain（不能每调用 prime/drain，否则跨迭代环被打断）。
