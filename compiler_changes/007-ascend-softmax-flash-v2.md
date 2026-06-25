# 007 · Ascend 新增 `softmax_flash_v2` 原语(逐指令复刻 AscendC `SoftmaxFlashV2`，变长 N softmax 不掩码)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-softmax-flashv2`（**独立基于 `ascendc_pto`=`cc98641d`**，与 006 无关、可单独合入） |
| **改动文件** | `src/tl_templates/ascend/common.h`（新增 `softmax_flash_v2` 模板 + CFG 常量）、`src/op/ascend.{h,cc}`（注册 builtin）、`src/target/codegen_ascend.{h,cc}`（dispatch + `SoftmaxFlashV2OpCodegen`）、`src/transform/common/operation_config.h`（两条 PIPE_V 配置）、`tilelang/language/ascend_tile.py`（`T.tile.softmax_flash_v2` 绑定） |
| **是否必须** | 是 —— SWA softmax 的「变长 N / 删掉掩码三件套」忠实复刻里，AscendC 用 `SoftmaxFlashV2` 的双量 `SoftMaxShapeInfo` 在窗口列上 reduce，拆开的 `reduce_max/reduce_sum` 表达不了 |
| **是否兼容** | 是 —— 纯新增 opaque 原语 + 新模板（仅被调用时实例化）+ 新 config 键，对现有算子零影响 |
| **状态** | 待容器从源码重编后验证（SWA 快测 + 回归） |

---

## 1. 算子为什么需要它

忠实复刻里，SWA 的 softmax 是 **一条 AscendC 库调用**
`SoftmaxFlashV2<float, true, true, false, false, CFG_WITHOUT_BRC>(...)`
（`sparse_attn_sharedkv_swa_block_vector.h:349-355`），配 `SoftMaxShapeInfo srcShape{dealRowCount,
columnCount, dealRowCount, actualColumnCount}`：

- `columnCount = actualSingleProcessSInnerSizeAlign`（对齐窗口宽 = 结果 buffer 行 stride）；
- `actualColumnCount = actualSingleProcessSInnerSize`（**真实窗口长 winm**）。

库内部 **只 reduce `actualColumnCount` 个有效列**，把 `[actualColumnCount, columnCount)` 的对齐
padding 自动排除——这就是 Ascend C 排除「窗口外 / 对齐尾」列的机制本身：**不掩码、不压实、不置 -inf**。

我们之前把 softmax 拆成了 `reduce_max → row_expand_sub → exp → reduce_sum` 加 sink 项。问题在于
AscendC 高阶 `ReduceMax/ReduceSum(srcShape={M,N})` 把 src 当 **连续 [M,N]**（行 stride=N），**只有
一个 N、没有「对齐 stride + 有效列」双量通道**：把 N 换成运行期 winm 会按 stride=winm 读第 2 行（应是
128）→ 错位。要正确，要么压实 buffer + 中和 ≤7 对齐尾（= 发明 Ascend C 没有的步骤、绕行），要么
**忠实复刻 Ascend C 的那条 `SoftmaxFlashV2`**。本改动选后者。

## 2. 现象 / 缺口

`T.reduce_max / T.reduce_sum` 走 `tl.ascend_reduce`，把 `M,N` 烧进模板字符串 `reduce_max<T,M,N,dim>`，
`N` 既是模板列数也是 src 行 stride（连续假设）。没有「对齐宽度（stride）+ 运行期有效列数
（actualColumnCount）」双量。TileLang/编译器侧没有任何原语能发出 `SoftmaxFlashV2`。

## 3. 根因

`SoftmaxFlashV2` 是 AscendC 高阶库函数，需要构造 `SoftMaxShapeInfo`（双量）、调
`SoftMaxFlashV2TilingFunc` 推 tiling、再以 6 个模板布尔位 + 10 个运行期参数调用。TileLang 没有
对应原语，**属于「TileLang 表达不了 → 加原语」**。

## 4. 为什么不能在内核侧解决

内核侧拿不到「对齐 stride + actualColumnCount」双量 reduce 能力；用现有 reduce 只能在固定列上算，
必须靠掩码三件套（createvecindex/compare/select）把窗口外列置 -inf，再在全 128 列上 reduce——这正是
Ascend C 没有的、要被消除的非忠实写法。

## 5. 修法（5 处接线，005/006 同款）

- **`common.h` 模板**：新增 `template<typename T, uint32_t M, uint32_t N> softmax_flash_v2(...)`，
  内部 **逐字复刻** 参考的三步：构造 `SoftMaxShapeInfo srcShape{M, N, M, actual_col}`、调
  `SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), tmp.GetSize(), true, false)`、调
  `SoftmaxFlashV2<T, true, true, false, false, kSoftmaxFlashV2CfgWithoutBrc>(dst, sum, max, src,
  expmax, in_sum, in_max, tmp, tiling, srcShape)`。CFG 常量
  `{false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC}`（max/sum/exp 输出 `(m,1)` 不广播）。
  **关键**：`N` = buffer 行 stride（编译期，我们 `s_ub` 是 `[M,N]`=`[G2,BI]`），`actual_col` = 运行期
  窗口长 winm（≤N）。复杂度全藏进模板，codegen 只发一行调用。
- **`op/ascend.{h,cc}`**：`TIR_DEFINE_TL_BUILTIN(ascend_softmax_flash_v2).set_num_inputs(10)`（opaque）；
  inputs = `[0]`模板名、`[1..8]` buffer（dst,sum,max,expmax,src,in_sum,in_max,tmp）、`[9]` actual_col。
- **`codegen_ascend.{h,cc}`**：dispatch `ascend_softmax_flash_v2()` → `SoftmaxFlashV2OpCodegen`，handler
  `PrintOpCall(op, "tl::ascend::"+模板名, {1,9}, {9,10})`（8 buffer + 1 运行期标量，仿 `RowExpandCodegen`）。
- **`operation_config.h`**：加两条（与 `gemm_v0_fixp` 双键同理，覆盖 TIR-builtin 与 call_extern 两条
  lookup）：`tl.ascend_softmax_flash_v2`（1-based）与 `softmax_flash_v2`（0-based），dst/sum/max/expmax
  写、src/in_sum/in_max 读、tmp 写，`PIPE_V`。
- **`ascend_tile.py`**：`softmax_flash_v2(dst, out_sum, out_max, expmax, src, in_sum, in_max, tmp,
  actual_col)` 绑定，`M,N` 取自 dst 末两维，`actual_col` 作运行期标量挂 `call_intrin` 尾（006 范式）。

内核侧（`kernel.py`，算子仓）：softmax 段换成
`mul scale → T.tile.softmax_flash_v2(s_ub, denom[buf], m_i[buf], expmax_ub, s_ub, ones_ub, sink_ub,
softmax_tmp, winm)`，删掉 createvecindex/compare/select + reduce_max/tile.max/row_expand_sub/exp/
reduce_sum/psink。`in_max=sink`、`in_sum=1.0`，输出的 `m_i/denom` 与旧路径逐位一致。

## 6. 为什么它是兼容性修改（及待验证项）

- **纯新增 opaque 原语**：dispatch 每个 `same_as` 是独立分支，新 op 只命中自己；现有 op 判定不变。
- **模板按需实例化**：`softmax_flash_v2` C++ 模板只在被发出的调用点实例化；无任何现存内核调用它，故
  现存算子编译产物字节级不变（与 004/005/006 同构）。`kernel_operator.h`（提供 `SoftmaxFlashV2`）已由
  catlass 头传递包含（现有 `ReduceMax/Broadcast/Brcb` 即来自它）。
- **新 config 键无影响**：`GetOperationConfig()` 按 key 精确查找，新键仅在归一化名匹配时命中。
- **PTO 路径不涉及**：softmax 走非-PTO `codegen_ascend.cc`；`pto_*` 与 `codegen_ascend_pto.cc` 不动。
- 待容器重编后验证：SWA 快测仍 PASS（数值与掩码版逐位一致）；回归
  `paged_flash_attn_bhsd` / `sparse_flash_attn_developer` 仍通过（不调用本原语）。

## 7. 必要性与通用性

**必要性**：AscendC SWA 的 softmax 就是 `SoftmaxFlashV2`；不加这个原语，TileLang 侧只能靠掩码三件套
绕行、且 reduce 算满 128 列，无法逐指令复刻「变长 N、窗口列上 reduce」。**通用性**：`softmax_flash_v2`
是 **flash-attention 在线 softmax 的通用原语**（变长 N、sink 种子、running max/sum、expMax 修正），任何
flash/sliding-window/变长序列 attention 都可复用；对不调用它的算子零影响。
