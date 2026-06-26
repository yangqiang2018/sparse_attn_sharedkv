# 012 · Ascend `gemm_v0_fixp` 加 `dst_row_stride`：per-N-tile 输出写进宽 dst 的列段（内核驱动 `ComputeMm2` nL1 输出切分）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | `wip/gemm-v0-fixp-dststride`（**独立基于 `ascendc_pto`**，含 001–011） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`gemm_v0_fixp` 加 `dst_row_stride`，fixpipe `realDstN = dst_row_stride ? dst_row_stride : N`）、`tilelang/language/ascend.py`（绑定加 1 参）、`src/target/codegen_ascend.cc`（`GemmFixpOpCodegen` emit args[12]）、`src/op/ascend.cc`（`set_num_inputs` 12→13） |
| **是否必须** | 是 —— Layer ②「全切块」忠实复刻要求**内核驱动** PV 的 per-nL1 输出（每片 V → 一个 [G,128] 输出 tile），而每个 tile 要写进 [G,512] 输出的第 i 个列段、GM 行间距仍是完整的 512。现 gemm 用 `N=dst.shape[-1]` 当行间距，切片 dst（[G,128]）会用错 128 → 写错 GM 位置 |
| **是否兼容** | 是 —— `dst_row_stride=0` 默认 → fixpipe `realDstN=N`，现有 caller（整块 dst）逐字节不变 |
| **状态** | ⏳ 待 NPU 验证（SWA 5/5 + 回归）后合入 `ascendc_pto`。这是「全切块/环感知 cube」**Layer ②**（PV 的 V 按输出 D 切 4 片 + 内核驱动 per-N-tile gemm）;③(反向 flag)④(持久迭代器)⑤(删 barrier)续做 |

---

## 1. 算子为什么需要它

参考 `ComputeMm2`（PV）按**输出 D** 切 `nL1Loops=4` 个 128 的 tile，每个 nL1:载入对应的 V D-切片
`[window, 128]` → mma → Fixpipe 到 `mm2ResGm[... + nL1*N_SPLIT_SIZE]`，`fixParams.dstStride = nSize`
（= 完整输出宽 512，block_cube.h:927-936）。即**每个 [G,128] 输出 tile 写进 [G,512] 输出的第 nL1 个
列段,而 GM 两行间距是完整的 512**。

Layer ② 忠实复刻 = 内核驱动这 4 个 per-nL1（每片 V 一个 gemm 调用,对称于 Layer ① 的 QK 按 D-chunk）。
要让 `gemm_v0_fixp` 写一个 [G,128] tile 进 [G,512] 输出的第 i 列段,需要 dst 的**完整行间距 512**。

## 2. 现象 / 缺口

`gemm_v0_fixp` 的 fixpipe `copy_l0c_to_gm<...,M,nTile>(dst[nL0Idx*nTile], C, realDstN=N, ...)` 用模板
`N`(= `dst.shape[-1]`)当 GM 行间距(`LayoutGM{tailM, realDstN}`)。若把一个 `[G,128]` 列切片
`workspace_o[..., i*128:(i+1)*128]` 当 dst,`N=128` → 行间距用成 128(应为 512)→ 第 g 行写到
`base + g*128` 而非 `base + g*512`,**写错 GM 位置、行相互覆盖**。

## 3. 根因

fixpipe 的 GM 行间距写死等于 `N`(= dst 末维)。整块 dst(`[G,512]`,N=512)时对;窄列切片 dst 时
`N` 不再是真实行间距。

## 4. 为什么不能在内核侧解决

行间距是 `copy_l0c_to_gm` 的 `LayoutGM` 参数,在**模板内部**;内核传不进去。要写「窄列段 + 宽行间距」
只能在原语上给一个独立的行间距参数。

## 5. 修法

**`common.h` `gemm_v0_fixp`**：加 `uint32_t dst_row_stride = 0`。fixpipe:
```cpp
uint32_t realDstN = dst_row_stride ? dst_row_stride : N;
copy_l0c_to_gm<T2, T2, LayoutGM, M, nTile>(dst[nL0Idx*nTile], C[c_base], realDstN, 0, fixN, 0b11);
```
默认 0 → `realDstN=N`(整块 dst,逐字节不变)。

**内核（`kernel.py` Layer ②）**：PV 的 V 切 `PV_NUM=4` 个输出-D 片(`v_l1[PV_NUM,BI,128]`,
`copy_pa(act_head_dim=128, d_idx=i*128)`)+ 4 次 per-N-tile gemm:
```python
for i in range(PV_NUM):
    T.gemm_v0_fixp(p_l1, v_l1[i,:,:], cL0,
                   workspace_o[cid,bufm,:, i*128:(i+1)*128],
                   k_actual=winm, init=True, cl0_base=i+1,
                   prime_drain=False, dst_row_stride=D)   # D=512
```
`cl0_base=i+1` → cL0 槽 1,0,1,0(= 008 ping-pong)。每片自带 fixpipe(`init=True`,单 K tile)。

绑定/codegen/`set_num_inputs` 照范式透传(arg [12]=dst_row_stride;`set_num_inputs` 12→13)。

## 6. 忠实性（对照 `block_cube.h` `ComputeMm2`）

| TileLang | Ascend C 参考 |
|---|---|
| 4 次 per-N-tile gemm,每片 V → 一个 [G,128] 输出 tile | `for nL1(4)`,每片 V → Fixpipe 一个 tile(:647/924-936) |
| `dst=workspace_o[...,i*128:..]` + `dst_row_stride=D` | `Fixpipe(mm2ResGm[...+nL1*128], fixParams.dstStride=nSize)`(:930/934) |
| `act_head_dim=128, d_idx=i*128` 载 V 的第 i 个 D-片 | `DataCopyPA(..., dIdx=nL1*N_SPLIT)`(每片 V 进环槽) |
| `cl0_base=i+1` → cL0 槽 1,0,1,0 | 连续 `cL0BufIter` 横跨 4 个 nL1(:833-940) |

## 7. 兼容性证据

1. **默认值保兼容**：`dst_row_stride=0` → `realDstN=N`,整块 dst 的现有 caller(QK/PV-whole/008)逐字节不变。
2. **回归**：`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 走 `gemm_v0`,不受影响;仍复跑确认。
3. **唯一新用法**：SWA PV 的 4-tile 调用是唯一传 `dst_row_stride!=0`(= 窄列切片 dst)的 caller。

## 8. 必要性

Layer ② 忠实复刻 `ComputeMm2` 的 nL1 输出切分的前提;是环感知 cube（3-slot KV 环 + 反向 flag + 持久
迭代器 + 删 barrier）的 PV 侧基础。
