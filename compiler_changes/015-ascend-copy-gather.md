# 015 · 新增非-PTO `copy_gm_to_ub_gather` 原语(忠实 scfa V0 CopyInKv 稀疏块 gather)

> 一句话:scfa(scenario 3)的 V0 段要把 topk 选中的**两个非相邻 KV 块用一条
> `DataCopyPad` 融合搬进 packed UB merge 缓冲**(参考 `CopyInKv`),其 `srcStride` 是
> 两块之间的**运行期字节间隙**。现有 `T.copy`(`AscendCopy`)的 `srcStride` 恒等于源
> buffer 的编译期内维(`compute_strideN = src->shape[last]`)、`blockCount` 来自 slice
> 的行 extent —— 表达不了「运行期间隙 + 运行期块数」的 gather。故**新增一个纯加法
> 的薄前端原语** `tl.ascend_copy_gather`,把 `DataCopyExtParams` 四个字段直接暴露给
> 内核,device 模板与参考 `DataCopyPad` 逐字段 1:1。全新 builtin/模板/key,不碰任何
> 现有分支 → **结构性零回归**(= 013 同款加法)。

## 动机 / 参考指令

参考 `op_kernel/arch32/sparse_attn_sharedkv_scfa_block_vector.h` 的 `CopyInKv`
(快路径,@620-642):

```cpp
DataCopyExtParams intriParams;
intriParams.blockLen   = sparseBlockSize * headDim * sizeof(KV_T); // 每块字节数
intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);    // 运行期 1 / 2
intriParams.dstStride  = 0;                                        // UB 内 packed
intriParams.srcStride  = keySrcStride;     // 运行期 = 两块 topk 间隙(字节,见 @600-612)
DataCopyPadExtParams<KV_T> padParams;      // 默认构造 = 不 pad
DataCopyPad(kvMergUb_[...], cmpKvGm_[startGmOffset], intriParams, padParams);
```

两块由 `GetRealS2Idx`(topk 查表)定位、地址不相邻,`keySrcStride` 是它们的运行期
字节间隙。整条指令的意义 = **一条 DMA 把两个稀疏块 gather 进 merge 缓冲**(快路径;
间隙溢出/为负/越界时回退成 `CopyInSingleKv` 两条单块拷贝)。

## 为何现有原语 / 已做编译器改(014)做不到(逐路径核实)

| 候选 | 事实 | 结论 |
|---|---|---|
| `T.copy`(`AscendCopy::Lower`,`src/op/ascend.cc:215/455`) | `strideN = compute_strideN(src) = src->shape[last]`(编译期内维);`maskShapeM = validRow_src`(slice 行 extent 的 clamp) | `srcStride` 被钉死成 buffer 内维,**无法注入运行期 topk 间隙**;`blockCount` 来自 slice 几何,无法给运行期 2 |
| 014(运行期 inner-extent) | 只解决「模板列维误用 region extent」,数据通路仍是连续 slice | 与 gather 的「两间隔块」正交,无关 |
| `copy_pa`(`common.h:101`) | GM→**L1** 的 Nd2Nz paged gather(cube) | 错 dest(L1 非 UB)、错 layout(Nz 非 ND) |
| 两条 `T.copy`(逐块) | = 参考的 `CopyInSingleKv` **回退路径**(2 条指令) | 功能可行但**不忠实快路径**(快路径是 1 条融合 DMA);流水/指令结构不一致 |

→ 满足铁律「现有原语 + 已做编译器改真发不出 → 才加原语」。

> 注:`CopyOutMrgeResult`(scatter)是 `srcStride=dstStride=0` 的连续 UB→GM 拷贝、
> 只 `blockCount` 运行期,**理论上现有 `T.copy` + 014(运行期 row-extent)可能就能发**,
> 故本次**不**为它加原语 —— 先在内核用 `T.copy` 实现并 dump codegen 核实;若发不出再
> 单独补(016)。本原语只解决铁定发不出的 gather 快路径。

## 实现(纯加法,7 处接线,照 `copy_pa` 范式)

`copy_pa` 是完美模板:**front-end `_retrieve_ptr` 在 Python 侧把运行期标量偏移折进
`access_ptr` → 直接 codegen(无 Lower)→ 调 device 模板**。

| 文件 | 改动 |
|---|---|
| `src/tl_templates/ascend/common.h`(`copy_ub_to_gm` 之后) | + `copy_gm_to_ub_gather<T>` device 模板 |
| `src/op/ascend.h`(`ascend_copy_pa` 之后) | + `TVM_DLL const Op &ascend_copy_gather();` |
| `src/op/ascend.cc`(`ascend_copy_pa` builtin 之后) | + `TIR_DEFINE_TL_BUILTIN(ascend_copy_gather).set_num_inputs(-1)` |
| `src/target/codegen_ascend.cc`(dispatch + 处理函数) | + `else if (...ascend_copy_gather())` 分支 + `CopyGatherCodegen`(照 `CopyPACodegen`,2 buf + 标量) |
| `src/target/codegen_ascend.h` | + `void CopyGatherCodegen(const CallNode *op);` 声明 |
| `src/transform/common/operation_config.h`(`copy_pa` 之后) | + `{"copy_gm_to_ub_gather", {{{0,"write"},{1,"read"}}, "PIPE_MTE2"}}` |
| `tilelang/language/ascend.py`(`copy_pa` 之后) | + `def copy_gather(dst, src, block_count, block_len_bytes, src_stride_bytes, dst_stride=0)` |

device 模板与参考 `DataCopyPad` 逐字段 1:1(无单位换算、无 Duplicate pad):

```cpp
template <typename T>
CATLASS_DEVICE void
copy_gm_to_ub_gather(LocalTensor<T> dstTensor, GlobalTensor<T> srcTensor,
                     uint16_t blockCount, uint32_t blockLenBytes,
                     uint32_t srcStrideBytes, uint32_t dstStride = 0) {
  AscendC::DataCopyExtParams dataCopyParams(blockCount, blockLenBytes,
                                            srcStrideBytes, dstStride, 0);
  AscendC::DataCopyPadExtParams<T> padParams;        // 默认 = 不 pad(== 参考)
  AscendC::DataCopyPad(dstTensor, srcTensor, dataCopyParams, padParams);
}
```

内核侧把 `blockLenBytes = sparseBlockSize*headDim*sizeof`、`srcStrideBytes = keySrcStride`、
`block_count = 1/2`、`dst_stride = 0` 按参考原样算好传入;`dst`/`src` 的运行期标量偏移
(`mergeMte3Idx%2*OFFSET + (mte2Size-mte3Size)*headDim` / `startGmOffset`)经
`_retrieve_ptr` 折进指针 → codegen `args[2]` PrintExpr 发出。

## 兼容性论证(零回归)

- **全新** TIR op(`tl.ascend_copy_gather`)、device 模板(`copy_gm_to_ub_gather`)、
  codegen 分支、`operation_config` key、前端函数 —— 没有任何现有 key / 分支 / 模板被
  触达,现有算子(SWA / CFA / examples)的 codegen 字节不变。
- device 模板内 `DataCopyExtParams(...)` + 默认 `DataCopyPadExtParams<T>` 与参考
  `CopyInKv` 快路径逐字段一致,无单位换算、无额外 `Duplicate`/flag。
- 回归只需确认 SWA + CFA + 几个 examples smoke 仍编过 + 数值正确(= 印证未触达)。

## 验证

- 回归:`swa_prefill / swa_decode`、`cfa_prefill_fast` 正确性 + 选几个 `examples/` smoke。
- 功能:scfa V0 内核段用 `T.copy_gather(...)` 编过 + dump codegen 核实发出
  `tl::ascend::copy_gm_to_ub_gather<half>(kvMergUb[...], cmpKvGm[...], 2, blockLen, keySrcStride, 0)`
  + `scfa_prefill_fast` 正确性。
