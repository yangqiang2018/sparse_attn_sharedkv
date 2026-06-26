# 011 · Ascend `gemm_v0_fixp` 加 `flush_last`/`do_fixpipe`：per-K-chunk 累加（内核驱动 GM→L1 链式切块 = `ComputeMm1` kL1 切分）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | `wip/gemm-v0-fixp-kchunk`（**独立基于 `ascendc_pto`**，含 001–010） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`gemm_v0_fixp` 加 `flush_last`/`do_fixpipe` + `kL0split` 改运行期 `k_actual`）、`tilelang/language/ascend.py`（绑定加 2 参）、`src/target/codegen_ascend.cc`（`GemmFixpOpCodegen` emit args[10][11]）、`src/op/ascend.cc`（`ascend_gemm_v0_fixp` `set_num_inputs` 10→12） |
| **是否必须** | 是 —— 用户拍板「全切块」忠实复刻：参考的 `ComputeMm1` 把 QK 的收缩轴 D=512 切成 2 个 256 的 kL1 D-半，**逐半 GM→L1 载入不同 L1 环槽、累加进同一 cL0 后单次 Fixpipe**（block_cube.h:341-450/575-604）。要让内核驱动这条「载一个 D-chunk → 调 gemm 累加」，`gemm_v0_fixp` 必须能 per-K-chunk 调用：非末 chunk 不 flush/不 fixpipe、续累加 |
| **是否兼容** | 是 —— `flush_last=true`/`do_fixpipe=true` 默认 → 现有 caller 单次调用逐字节不变；`kL0split` 改用运行期 `k_actual`，但现 caller `k_actual==K` 故 `kL0split` 不变 |
| **状态** | ✅ 已合入 `ascendc_pto`（`Merge 011` `a191c7d7`）：NPU SWA 快测 5/5 PASS + 回归（paged_flash_attn_bhsd + sparse_flash_attn_developer）全过。这是「全切块/环感知 cube」**Layer ①**（QK 的 Q/K 各切 256+256 + per-chunk gemm 累加），已 5/5；Layer ②(PV V 4 切片)③(反向 flag)④(持久迭代器)⑤(删 barrier)续做 |

> **踩坑备忘（合入过程）**：算子侧 Layer① 调试时撞上 `LowerTileOp` 编译期 SIGSEGV，根因**不在本编译器改动**，而是内核把 `D2 = D//2` 写在了 `@T.prim_func` body 内 → TVMScript 变符号 `Var` → 用作 L1 buffer 维（非 IntImm）→ `makeBufferWithLayout` 裸解引用崩。修法 = 把 `D2` 挪到 prim_func 外层闭包常量。定位用 RelWithDebInfo build + gdb + `std::cerr`（`LOG(INFO)` 被静默）+ `pytest -s`（否则崩时捕获丢失）。已沉淀进 tilelang 插件 skills（pitfalls + debugging）。**编译器侧 `makeBufferWithLayout` 对非 IntImm 维应 ICHECK 而非 segfault，留作独立健壮性硬化。**

---

## 1. 算子为什么需要它

参考 `ComputeMm1`（QK）把 headDim=512 当收缩轴,切成 `kL1Loops=2` 个 256 的 D-半
（`kL1Size=256`,block_cube.h:341-345）。每个 kL1:`DataCopyPA(actHeadDim=256, startPos.dIdx=kL1*256)`
把一个 256 D-半载入一个独立 KV 环槽（:387-398）,然后 `for kL0` 做 mma 累加进**同一个**
`cL0TensorPingPong` 槽（`cmatrixInitVal=(kL1==0&&kL0==0)`、`unitFlag=(最后)?0b11:0b10`,:575-579）,
**两个 kL1 半都累加完后才一次 Fixpipe**（`if(kL1==1) Fixpipe`,:591-604）。Q 同理切 2 个 256 半
（`CopyInMm1AToL1(headSize=256, headOffset=kL1*256)`,:549/552）。

我现在的内核是**整块载入**(Q[64,512]、K[win,512] 一次)+ 单次 `gemm_v0_fixp`,不是参考的
链式切块。要忠实复刻,内核要驱动 kL1 循环:载一个 D-半 → 调 gemm 累加;但 `gemm_v0_fixp`
只能「一次做完整段 K + flush + fixpipe」,无法被分成多段累加。

## 2. 现象 / 缺口

`gemm_v0_fixp` 末 kL0 恒 `0b11`（flush）并恒做 fixpipe,且 `kL0split` 按编译期 `K` 算。无法表达
「这段只是 D 的一个 chunk,先累加别 flush,后面还有 chunk」。

## 3. 根因

unitFlag 的 flush 时机（`0b11` vs `0b10`）和 fixpipe 写出都写死在模板里,按「这次调用就是全部 K」
假设;`kL0split=(K+127)/128` 用编译期 K,一个 256 chunk 会多跑空 tile。

## 4. 为什么不能在内核侧解决

unitFlag 的 `0b10/0b11`、cL0 累加（`cmatrixInitVal`/`cmatrixSource`）、fixpipe 都在**模板内部**的
kL0 循环里;内核调不进去。要 per-chunk 累加,只能在原语上开关「这是不是最后一个 chunk」。

## 5. 修法

**`common.h` `gemm_v0_fixp`**：加 `bool flush_last=true, bool do_fixpipe=true`。
- `kL0split` 改用**运行期 `k_actual`**：`kL0split=(k_actual+127)/128`。chunk 传 `k_actual=256`→2 tile;
  whole QK `k_actual=512`→4、PV `k_actual=win`→1（现 caller `k_actual==K`,`kL0split` 不变,字节兼容）。
- `unitFlag = (flush_last && kL0Idx==kL0split-1) ? 0b11 : 0b10`：`flush_last=false`（非末 chunk）→ 每个
  tile 都 `0b10`,cL0 续累加;只有末 chunk 末 tile `0b11`。
- fixpipe `copy_l0c_to_gm` gate 在 `if(do_fixpipe)`：非末 chunk 不写出,cL0 留着累加;末 chunk 一次写出
  全累加结果（= 参考 `if(kL1==1) Fixpipe`）。

**内核（`kernel.py` Layer ①）**：QK 的 Q/K 各切 2 个 256 D-半:
- `q_l1[2,G,256]`（2 个 Q 半,`T.copy(Q[tok,:,h*256:(h+1)*256], q_l1[h])`,= `CopyInMm1AToL1 headOffset`）、
  `kq_l1[2,BI,256]`（2 个 K 半,`T.copy_pa(act_head_dim=256, d_idx=h*256)`,= `DataCopyPA actHeadDim/dIdx`）。
- 两次 `gemm_v0_fixp` 进同一 `cl0_base=0` 槽:chunk0 `init=True, flush_last=False, do_fixpipe=False`;
  chunk1 `init=False, flush_last=True, do_fixpipe=True`。`k_actual=256`。
- PV 暂保持整块 V（Layer ② 再切 4 片）。

## 6. 忠实性（对照 `block_cube.h` `ComputeMm1`）

| TileLang | Ascend C 参考 |
|---|---|
| 2 个 chunk 进同一 cL0,chunk0 `init/flush_last/do_fixpipe`=T/F/F、chunk1=F/T/T | kL1=0/1 累加同一 cL0,`cmatrixInitVal=(kL1==0&&kL0==0)`、末 kL1 末 kL0 `0b11`+Fixpipe(:575-604) |
| `k_actual=256` per chunk,`kL0split=2` | `kL1Size=256`,`kL0Loops=2`(:342-345) |
| Q/K 各 256+256 D-半进各自 L1 槽 | `CopyInMm1AToL1 headOffset=kL1*256`(:549/552) / `DataCopyPA dIdx=kL1*256,actHeadDim=256`(:387-398) |

## 7. 兼容性证据

1. **默认值保兼容**：`flush_last=true`/`do_fixpipe=true` → 末 tile `0b11`+fixpipe,单次调用与改前逐字节同;
   `k_actual==K` 时 `kL0split` 不变。PV（008）/ whole-QK（009/010）caller 不变。
2. **回归**：`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 走 `gemm_v0`,不受影响;仍复跑确认。
3. **唯一新用法**：SWA QK 的 2-chunk 是唯一传 `flush_last=false`/`do_fixpipe=false` 的 caller。

## 8. 必要性

「全切块」忠实复刻 `ComputeMm1` 的 kL1 D-半链式累加的前提;是环感知 cube（3-slot KV 环 + 4-slot QP 环
+ 反向 flag + 持久迭代器 + 删 barrier）Layer ① 的 matmul 基础。
