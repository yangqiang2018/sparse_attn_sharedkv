# 014 · Ascend 非-PTO `copy_gm_to_ub` / `copy_ub_to_gm` 支持运行期 inner-extent

> 一句话:`src/op/ascend.cc` 里这两个 copy 的 **codegen 模板列维**误用了 region 的
> *运行期 extent* 当**编译期模板参数**;改成「extent 是常量就用 extent(不变),是运行期
> 表达式则回退用 buffer 的编译期 shape」。运行期实际宽度本就由 `maskShapeN` 函数参
> (`validCol_*`)承载,所以这是**可证零回归**的纯修复——只改「以前根本编译不过」的运行期
> inner-slice 这一路,所有现有调用(extent 全是常量)字节不变。

## 现象 / 动机

cfa(scenario 2)的 vector softmax,实际窗口宽 `tw_a`(cmp tile ≈ 64、ori ≈ 128)远小于
buffer 的 `S2_BASE=512`。参考 `block_vector.h` DealBmm1 只处理 `columnCount =
actualSingleProcessSInnerSizeAlign`(实际宽),我却按满 512 列 load `ws_s` / copy `ws_p`
(msprof:vector mte2 1722 vs AC 954,1.8×)。把 `T.copy` 改成运行期切片 `[..., 0:tw_a]`
追平实际宽度时,**编译直接挂**:

```
/tmp/...cpp:214:51: error: use of undeclared identifier 'T'
```

## 根因(逐路径核实)

`copy_gm_to_ub` 模板声明(`common.h:268`)= `template <typename T, uint32_t dstN,
uint32_t dstM = 1>`,**`dstN`(列维)是编译期模板参**;运行期实际宽度由函数参
`maskShapeN`(= lowering 算出的 `validCol_dst`,`ascend.cc:412-427` 的
`compute_valid_extent`)承载。但模板名构建处:

| 位置 | 事实 | 结论 |
|---|---|---|
| `ascend.cc:220`(gm2ub) | `ss << dst_extents[last]` —— 用 **region 运行期 extent** 填 `dstN` 模板参(旁边 `// ss << dst->shape[...]` 是被注释掉的旧正确做法) | 运行期切片时把非常量表达式塞进模板 → 非法 C++(`T` 是下游解析错) |
| `ascend.cc:233`(ub2gm) | 同样 `src_extents[last]` | 同上 |
| `compute_blocklen`(`ascend.cc:150`,即 `dstM` 维) | 仅当 `extents[size-2]` 是 `IntImmNode` 才用 extent,否则返回 `buf->shape[size-2]` | **已是运行期安全**,无需改 |
| `validCol_dst`→`maskShapeN` | 始终用 region extent(`compute_valid_extent` clamp 到 shape) | 运行期实际宽度**已正确下传**,模板修好即通 |

> 即:运行期宽度的「数据通路」(maskShapeN)早就对,只差模板列维不该用 extent。SWA / 现有
> 算子的所有 GM↔UB copy 都是满维(extent==shape,编译期常量),故从不触发;一旦 inner-slice
> 用运行期宽度就撞这个 bug。

## 为何不能用现有原语 / 内核侧绕

- `copy_gm_to_ub` 已有 `maskShapeN` 运行期参,**但 `T.copy` 没法只传 mask 不切 buffer**:
  不切 → extent==shape==512(满宽,无收益);切 `0:tw_a` → 触发模板 bug。死结只能在 codegen 解。
- 无其它原语能发「运行期宽度、连续 strided GM↔UB」(`copy_l0c_to_gm` 是 L0C→GM;`copy_pa` 是
  paged gather)。故按铁律「现有原语表达不了 → 改编译器」。

## 修法(`src/op/ascend.cc`,2 处,可证零回归)

gm2ub(原 `ss << dst_extents[last]`)/ ub2gm(原 `ss << src_extents[last]`)各改成:

```cpp
PrimExpr tmpl_n = <region>_extents[<buf>->shape.size() - 1];
if (!tmpl_n->IsInstance<IntImmNode>())          // 运行期 extent
  tmpl_n = <buf>->shape[<buf>->shape.size() - 1]; // 回退编译期 shape
ss << tmpl_n;
```

- **常量 extent(所有现有调用)**:`IsInstance<IntImmNode>()` 为真 → 仍用 extent → 模板字符串
  **逐字节不变**。
- **运行期 extent(以前必崩,无现存调用)**:用 buffer shape 当模板列维(编译期合法),实际窗口宽
  走 `maskShapeN` → DataCopyPad `blockLen=maskShapeN`、dst stride 跳 `(dstN−maskShapeN)`,
  正确地把 `tw_a` 列写进 `S2_BASE`-strided buffer 的前 `tw_a` 列。

## 兼容性论证

现有算子全部能编译 ⇒ 它们从不用运行期 inner-extent(否则早崩,如本次)⇒ 本修改对它们
**完全不触达**(走 IntImm 分支,字节不变)。回归只需确认 SWA + examples 仍过(= 印证它们没用
运行期 inner-extent),非「行为改变」验证。

## 验证

- 回归:`swa_prefill/swa_decode` 正确性 + 选几个 `examples/` smoke(bench_test)。
- 功能:cfa `T.copy(..., 0:tw_a)` 编过 + `cfa_prefill_fast` 正确性 + msprof vector mte2 下降。
