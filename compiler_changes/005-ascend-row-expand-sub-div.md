# 005 · Ascend 新增 `row_expand_sub` / `row_expand_div` 原语(行广播 Sub/Div,消掉非忠实的 [M,N] 广播缓冲)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-gemm-v0-fixp` · `410cafbd`(叠在 004 之上,使一次重编同时含 gemm_v0_fixp + row_expand) |
| **改动文件** | `src/tl_templates/ascend/common.h`(模板 `row_expand_div`/`row_expand_sub`)、`src/op/ascend.{cc,h}`(builtin)、`src/target/codegen_ascend.{cc,h}`(dispatch + `RowExpandCodegen`)、`src/transform/common/operation_config.h`(读写 + PIPE_V)、`tilelang/language/ascend_tile.py`(`T.tile.row_expand_{div,sub}` 绑定) |
| **是否必须** | 是 —— 不改的话向量侧只能 materialize `[M,N]` 广播缓冲,撑爆向量 UB(196352B) |
| **是否兼容** | 是 —— 纯新增,既有 `sub`/`div`/`row_expand_mul` 逐字不变 |
| **状态** | 待容器从源码重编后验证(SWA 快测 + msprof) |

---

## 1. 算子为什么需要它

SWA 向量侧两处要"每行一个标量、沿列广播"的逐元素运算,与 Ascend C 一一对应:

- **softmax 减最大值**:`s[i, :] -= rowmax[i]`(Ascend C 在 `SoftmaxFlashV2` 内做,行广播)。
- **输出归一化**:`o[i, :] /= denom[i]`(Ascend C `RowDivs`,`swa_block_vector.h:722-766`)。

Ascend C 用**行广播指令**做:先 `Brcb` 把 `[M,1]` 的逐行标量铺成 `[M, blk]`,再 `Div`/`Sub` 带
`BinaryRepeatParams{src1BlkStride=0, src1RepStride=1}`,**从不 materialize `[M,N]` 广播缓冲**。

## 2. 现象

cube/vector 软件流水版去掉 L0C 越界(004)后,向量侧运行期崩:

```
VEC instruction error: the ub address out of bounds. ... aivector error
```

## 3. 根因

我向量侧为了做 `s - max`、`o / denom`,用 `T.tile.broadcast` materialize 了
`m_2d[G2,BI]`(16KB)与 `den_2d[G2,D]`(64KB)两个 `[M,N]` 广播缓冲,再逐元素 `sub`/`div`。
串行版能跑,是因为内存规划把活跃期不相交的 softmax/output 缓冲**复用**了;但 cube/vector
流水让 `softmax(任务 j)` 与 `output(任务 j-1)` **同时存活**,规划器无法复用,向量 UB 累计
~206KB > 196352B(191.75KB)→ UB 地址越界。

这 ~82KB 的 `m_2d+den_2d` 是 **Ascend C 根本不分配的、为绕过 TileLang 缺行广播而 materialize
的多余缓冲** —— 是非忠实的绕行,UB 越界只是它的症状。诊断的第一步就该是"我的向量 UB 布局
和 Ascend C 一样吗",答案是否,故修法是**消除这处不忠实**,而不是把它"塞进去"(早先的 D 列
分块就是错误的绕行打补丁,已回退)。

## 4. 为什么不能在内核侧解决

`T.tile.sub`/`T.tile.div`(`ascend_tile.py` 的 `binary_op`)只有三条 src1 路径:① BufferLoad
标量(只取 `indices[0]`,仅 1D scalar)② PrimExpr/float 标量 ③ BufferRegion 要求 size 完全
相等(逐元素)。**没有 `[M,1]→[M,N]` 行广播路径**。已有的行广播原语 `row_expand_mul` 只接了
**PTO** codegen(非-PTO 是 `LOG(FATAL)`,`codegen_ascend.cc`),而本算子走非-PTO 路径。所以内核
侧无法表达行广播 sub/div —— 属于 *"TileLang 表达不了→加原语"*。

## 5. 修法

新增非-PTO 原语 `row_expand_div` / `row_expand_sub`,**逐字复刻 Ascend C `RowDivs`**:

```cpp
template <typename T, uint32_t M, uint32_t N>
row_expand_div(dst, src0, src1_col, tmp) {        // tmp: [M, 32/sizeof(T)] 暂存
  Brcb(tmp, src1_col, (M+BLK-1)/BLK, {1, BLK});   // [M,1] -> [M,blk]
  BinaryRepeatParams rp{src0BlkStride=1, src1BlkStride=0, dstBlkStride=1,
                        src0RepStride=N/BLK, src1RepStride=1, dstRepStride=N/BLK};
  for (i in [0, N/MASK)) Div(dst[i*MASK], src0[i*MASK], tmp, MASK, M, rp);  // 行广播
}
```

`row_expand_sub` 同构,`Div`→`Sub`。`BLK=32/sizeof(T)`(f32=8)、`MASK=256/sizeof(T)`(f32=64),
与参考的 `FP32_BLOCK_ELEMENT_NUM`/`FP32_REPEAT_ELEMENT_NUM` 一致。只移植了参考的**行-repeat
首支**(`N/MASK <= M`、`N%MASK==0`、行 pitch == N),SWA 调用满足(M=32, N∈{128,512}),
两条 `static_assert` 守卫。

内核侧:删 `m_2d`/`den_2d`,加 `brcb_m`/`brcb_d`(各 `[G2,8]`=1KB);`s - max` → 一句
`T.tile.row_expand_sub(s_ub, s_ub, m_i[buf], brcb_m)`;`o / denom` → 一句
`T.tile.row_expand_div(o_ub, o_ub, denom[bufm], brcb_d)`,整 D 一次过(回退 D 分块)。

## 6. 为什么它是兼容性修改(及待验证项)

- **纯新增**:`sub`/`div`/`row_expand_mul` 及所有既有调用逐字不变;新原语是独立 builtin +
  非-PTO codegen 分支(复用现有 `PrintOpCall`)+ `operation_config` PIPE_V 条目。
- combinecv 无需改(向量是默认归类,既有 `T.tile.*` 向量 op 都不在 cube 白名单里)。
- 对抗式审查(C++ 模板对照参考 RowDivs、5 层 arg 布局、原地 dst==src0、类型/编译):无
  compile-break、无 numerical-bug。
- **PTO 路径不涉及**(本算子非-PTO);若日后 PTO 需要,另接 `codegen_ascend_pto.cc`。

待容器重编后验证:SWA 快测通过(UB 不再越界 + 数值对)、回归 `paged_flash_attn_bhsd` /
`sparse_flash_attn_developer` 仍通过。

## 7. 必要性与通用性

**必要性**:不改则向量侧只能 materialize `[M,N]` 广播缓冲,在 cube/vector 流水下撑爆向量 UB
(§2/§3);内核侧躲不掉(§4:无行广播路径,且 `row_expand_mul` 非-PTO 不可用)。阻断性。

**通用性**:行广播逐元素是**通用向量能力**,与 sparse_attn_sharedkv 无关 —— 任何"逐行标量
× / − / 沿列广播"(softmax 的减 max/除 sum、layernorm/rmsnorm 的按行 scale、注意力归一化等)
都需要它,且都不应 materialize `[M,N]` 缓冲。这正是 Ascend C 全线用 `RowDivs`/`RowMuls` 的原因;
本改动把同款能力以兼容新增原语的形式带给非-PTO 路径(补齐 `row_expand_mul` 只在 PTO 的缺口)。

## 8. 忠实性说明

改完后向量侧**不再有任何 Ascend C 没有的 `[M,N]` 广播缓冲**,`s-max`/`o/denom` 与 Ascend C
的行广播 `Sub`/`RowDivs` 指令层面对齐,向量 UB 预算(~124KB)由"删掉非忠实缓冲"得到,而非
列分块"塞进去"。这是 *"绝不绕行、正面突破"* 的要求:撞到 TileLang 缺行广播 → 加原语复刻
Ascend C,而不是在内核里发明绕路。
