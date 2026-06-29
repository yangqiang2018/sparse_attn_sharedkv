# 013 · Ascend 新增非-PTO `row_expand_mul_nd`(行广播乘 = 忠实 RowMuls)

> 一句话:给**非-PTO**(`tilelang_ascend`)codegen 补一个行广播乘原语
> `row_expand_mul_nd`(Brcb + Mul),= Ascend C 的 `RowMuls`。它是 cfa/scfa 的
> FlashAttention **PV rescale**(把上一 KV-tile 的部分和按 `exp(m_old−m_new)` 行缩放)
> 唯一缺的指令。与 005 的 `row_expand_sub/div` 同族、同实现方式;**纯新增,PTO 路径
> 与其 examples/HISA 调用方字节不变**。

## 现象 / 动机

cfa(scenario 2)/scfa(scenario 3)的多-KV-tile 在线 softmax,在 PV 阶段需要按
flash rescale 因子 `expmax`(每行一个标量)对上一 tile 的部分输出做**行广播乘**
再累加(`block_vector.h` `DealBmm2ResBaseBlock` 的 `if(!isFirstSInnerLoop)` 分支:
`Brcb(expUb, softmaxExpUb) → RowMuls(prevPV, expUb) → Add(curPV, prevPV)`)。

SWA(scenario 1)是单 KV-tile 退化路径,`isFirstSInnerLoop==isLastS2Loop==true`,
**从不进 rescale 分支**,所以一直没用到行广播乘。反退化做 cfa/scfa 时这条是硬需求。

## 根因(逐路径核实)

| 路径 | 事实 | 结论 |
|---|---|---|
| 内核想发的指令 | 参考 `RowMuls` = `Brcb([M,1]→[M,blk])` + `Mul`(`src1BlkStride=0`/`src1RepStride=1`),与 `RowDivs`(005 已复刻)只差 Div↔Mul | 需要一条行广播乘 |
| 现有 `row_expand_mul`(ascend_tile.py) | 是 **PTO 专用**(`TROWEXPANDMUL`);examples/HISA 两处在用 | 不能改它 |
| 非-PTO codegen 的 `RowExpandMulCodegen` | `codegen_ascend.cc:2079` 直接 `LOG(FATAL) << "TROWEXPANDMUL is only supported in the PTO codegen path."` | **SWA/cfa/scfa 走的非-PTO 路径发不出行广播乘** |
| `common.h` | 只有 `row_expand_div`(:1175)/`row_expand_sub`(:1207),**无 `row_expand_mul`** | 非-PTO 模板缺失 |

> SWA 永远走 `target.build.tilelang_ascend` = `CodeGenTileLangAscend`(非-PTO),故撞
> `LOG(FATAL)`。这是 structure 层 scoping 漏判、指令级核对(往 codegen 挖)才发现的真缺口。

## 为何不能在内核侧 / 用现有原语绕

- **不能用 `row_expand_div` 凑**:`prev * expmax` 写成 `prev / (1/expmax)` 需要额外
  reciprocal + Div,**指令序列与参考的 Brcb+Mul 不同** = 绕行、不忠实。
- **不能改 PTO `row_expand_mul`**:它服务 PTO 算子(examples/HISA),改其 op/binding/
  codegen 会让那些算子字节变(违反兼容铁律)。

## 修法(5 处,均纯新增 / 兼容)

1. **`src/tl_templates/ascend/common.h`**:加 `row_expand_mul<T,M,N>` 模板,**逐字节复制
   `row_expand_div`、仅 `AscendC::Div`→`AscendC::Mul`**(同 `Brcb`+`BinaryRepeatParams`
   `src1BlkStride=0`/`src1RepStride=1`/`src0RepStride=src1RepStride=N/BLK`)。`tl::ascend`
   命名空间内,与 `TROWEXPANDMUL` 符号不冲突。
2. **`src/op/ascend.{cc,h}`**:注册**新 op** `ascend_row_expand_mul_nd`(`set_num_inputs(5)`,
   = div/sub)。**PTO 的 `ascend_row_expand_mul` 原封不动。**
3. **`src/target/codegen_ascend.cc`**:把 `ascend_row_expand_mul_nd()` 加进 div/sub 的
   `RowExpandCodegen` 分支(:623,通用 printer:args[0]=模板名、args[1..4]=buffer)。
   `RowExpandMulCodegen`(PTO 那条 FATAL)不动。
4. **`tilelang/language/ascend_tile.py`**:加 `row_expand_mul_nd` 函数 =
   `_row_expand_binary("row_expand_mul", "tl.ascend_row_expand_mul_nd", …)`(在 Python 层
   从 `dst.shape[-2:]` 算 M,N 烤进模板名 `row_expand_mul<T,M,N>`,= div/sub 同款)。

内核侧 `T.tile.row_expand_mul_nd(dst, src0, src1_col, tmp)` → 发出
`tl::ascend::row_expand_mul<T,M,N>(...)` = `Brcb`+`Mul`,逐条等于参考 `RowMuls`。

## 兼容性证据

- **PTO 路径零改动**:`ascend_row_expand_mul`(op)、`row_expand_mul`(ascend_tile.py:2067)、
  PTO `RowExpandMulCodegen`(`TROWEXPANDMUL_row_vec`)、`examples/HISA/*` 调用方全未触碰 →
  PTO codegen 输出字节不变。
- **非-PTO 现有 op 零改动**:div/sub 的模板/op/codegen 不变;codegen 改动只是在既有
  `else if` 追加一个 `|| ascend_row_expand_mul_nd()`(纯加分支,不改 div/sub 行为)。
- **新增物只在被调用时激活**:`row_expand_mul_nd` 不被任何现有算子调用 → 现有 SWA/
  paged_flash/sparse_flash/HISA 均不命中。
- **回归**:`examples/HISA/block_sparse_mqa_attn_expert_test.py`(PTO `row_expand_mul`)+
  `examples/flash_attention/paged_flash_attn_bhsd.py` + `examples/developer_mode/
  sparse_flash_attn_developer.py` 数值/字节不变即证兼容。

## 必要性

cfa/scfa 的多-tile 在线 softmax 的 **PV rescale** 是其核心(把上一 KV-tile 的部分输出
按 `expmax` 重缩放再累加);忠实复刻要求发出与参考 `RowMuls` **逐条相同**的 `Brcb+Mul`,
非-PTO 路径此前 `LOG(FATAL)`。这是 cfa/scfa 反退化的硬地基。

## 忠实性

逐指令对齐 `block_vector.h` `DealBmm2ResBaseBlock`:`RowMuls(prevPV, expUb)` =
`Brcb(expUb, [M,1])` + `Mul(... src1RepStride=1)`。新模板与 005 的 `row_expand_div`
同实现(只 Div→Mul),= Ascend C `RowMuls`/`RowDivs` 共用的行广播范式,非发明新方法。

## 状态

⏳ 待 NPU:容器 rebuild(`USE_ASCEND=True pip install -e . --force-reinstall --no-deps`)
+ 回归(HISA / paged_flash / sparse_flash 字节不变)。功能验证随 cfa 反退化(首个调用方)。
编译器 `wip/row-expand-mul-nd`(基于 `ascendc_pto`=`64fd7752`)。
