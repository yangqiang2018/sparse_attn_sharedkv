# 006 · Ascend `gemm_v0`/`mma` 增加运行期 `n_actual`(变长 N 输出列,= Ascend C `ComputeMm1` 的窗口长 N)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-gemm-v0-nactual` · `b255a071`(**独立基于 `ascendc_pto`**,与 004/005 无关、可单独合入) |
| **改动文件** | `src/tl_templates/ascend/common.h`(`mma`、`gemm_v0`)、`src/op/ascend.{cc}`(`ascend_gemm_v0` set_num_inputs 5→6)、`src/target/codegen_ascend.cc`(`GemmOpCodegen` 多发一参)、`tilelang/language/ascend.py`(`gemm_v0` 绑定加 `n_actual`) |
| **是否必须** | 是 —— SWA 的「全链路变长 N」忠实复刻里,QK 必须按实际窗口长 N 计算 |
| **是否兼容** | 是 —— `n_actual` 默认 `=N`,所有现有调用(及 `gemm_v0_fixp`)逐字节不变 |
| **状态** | 待容器从源码重编后验证(SWA 快测 + 回归) |

---

## 1. 算子为什么需要它

忠实复刻里,QK(`S = Q @ Kᵀ`)的 N 维 = KV 窗口列数。Ascend C 的 `ComputeMm1` 用**实际窗口长
`N=actualSingleProcessSInnerSize`**(精确窗口)做,**不是固定 128 + 掩码**。我现在的 QK 固定
`N=BI=128`(算满 128 列,再靠掩码把窗口外列置 -inf),既多算了列、又用了 Ascend C 没有的掩码
——是「没有变长 N」的绕行。变长 N 让 QK 只算 `win` 列,逐指令对齐参考。

## 2. 现象 / 缺口

QK 当前 `T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True)`:N 是模板常量 BI,`mma` 把
`mmadParams.n` 设成模板 N,`copy_l1_to_l0b` 也搬满 N 列。无法只算运行期窗口长。

## 3. 根因

`gemm_v0`/`mma` 的 N 是**编译期模板参数**:`mma<M,N>` 里 `mmadParams.n = N`(模板);
`copy_l1_to_l0b<...>(l0b, B[...], kSize, N)` 第 4 运行期实参也是模板 N。没有「运行期输出列数」
这条通道(K 已经是运行期了,N 还不是)。

## 4. 为什么不能在内核侧解决

内核无法让 `mma` 只写前 `win` 列 / `copy_l1_to_l0b` 只搬 `win` 列 —— 这由 N 模板常量决定。
属于 *"TileLang 表达不了 → 加运行期参数(原语能力补齐)"*,与已合入的 PV `k_actual`(运行期 K)
完全同性质、对偶(K 是收缩长,N 是输出列)。

## 5. 修法

- **`mma`**:签名加 `uint32_t n_actual = N`(默认 = 模板 N),`mmadParams.n = n_actual`。K 已是
  运行期,N 照搬同一套。默认值保证所有现有 `mma` 调用(含 `gemm_v0_fixp`)逐字节不变。
- **`gemm_v0`**:签名加 `uint32_t n_actual = N`。**仅 `transpose_B` 单 N-tile 路径(QK)生效**:
  `copy_l1_to_l0b<T1,N,K,true>(l0b, B[...], kSize, n_actual)` 只搬 `n_actual` 列;`mma` 传
  `transpose_B ? n_actual : nTile`(只算 `n_actual` 列)。**模板 N/K、L0B/L0C 的 stride/偏移全部
  保持编译期上界 BI 不变**(只改「实算多少列」,不改物理布局)——与 PV `k_actual` 只换 `kSize`、
  不换 K-stride 完全对偶。**非 transpose 的 N-tiling 路径不受影响**(每 tile 仍用编译期 `nTile`)。
- **kernel.py**:QK 传 `n_actual=win`(当前任务窗口长)。**掩码(createvecindex/compare/select)
  暂时保留**:`acc_s_l0c[:, win:BI]` 现在是 `mma` 没写的 L0C 旧值,掩码把这些列置 -inf,softmax
  仍按 BI 跑时不被污染。等后续 007(softmax 也按 winm 缩列)落地,reduce 不再读这些列,才真正
  去掉掩码。

## 6. 为什么它是兼容性修改(及待验证项)

- **纯新增/默认等价**:`n_actual` 默认 `=N`;不传时 `mmadParams.n=N`、`copy_l1_to_l0b` 搬满 N
  列,与改动前**逐字节相同**。`gemm_v0_fixp` 调用 `mma` 不传 `n_actual`(默认 N),不受影响。
- 接线只新增:绑定加可选形参、builtin inputs 5→6、`GemmOpCodegen` 末尾多打一个等于编译期 N 的
  常量(不传时)。
- 待容器重编后验证:SWA 快测仍 PASS(QK 只算窗口 + 掩码兜底,数值不变);回归
  `paged_flash_attn_bhsd` / `sparse_flash_attn_developer` 仍通过(它们传默认 `n_actual=N`)。

## 7. 必要性与通用性

**必要性**:QK 按窗口长算是「全链路变长 N」忠实复刻的必需一环;不改则 QK 永远算满 128 列、
且依赖非忠实的掩码。**通用性**:运行期 N 是**矩阵乘原语的通用能力**(与 k_actual 对偶)——任何
需要「输出列数运行期可变」的 gemm(变长序列 attention、动态 shape matmul)都受益,且对固定-N
调用零影响。这把 `gemm_v0` 拉齐到「M 模板、K 运行期、N 运行期」的完整变长能力。
