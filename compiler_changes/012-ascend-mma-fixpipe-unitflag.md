# 012 · Ascend `T.mma` / `T.copy(L0C→GM)` 接通 `unitFlag`(内核驱动 fixpipe∥mma 融合)

> 一句话:给内核可调的 `T.mma` 和 `T.copy(L0C→GM)` 各加一个可选 `unit_flag` 尾参,把
> 「硬件 mma→fixpipe 流水(`cL0TensorPingPong` 重叠)」从 `gemm_v0_fixp` **模板内部**暴露到**内核层**,
> 让 Layer ③ 的「环感知 gemm 拆分」能在内核里逐 tile 驱动 `mma + 融合-带-行间距 fixpipe`。默认参保持
> 所有现有调用方字节不变。

## 现象 / 动机

Layer ③ 要把 PV/QK 从单体 `gemm_v0_fixp` **解构**到内核层逐 tile 驱动(因为每个 V D-切片要进**独立的
KV L1 环槽** `kvL1[kvL1BufIter%3]`,而单体 gemm 的 B 操作数是一块连续 buffer,内部 `nL0split` 循环读不了
3 个分离的环槽 —— 这是参考 `ComputeMm2` 的真实结构:每个 `(nL1, k1)` 子块进一个环槽)。解构后内核需要:

```
T.copy(V_ring_slot → l0b)            # L1→L0b
T.mma(p_l0a, v_l0b, cL0[slot], ...)  # 单 tile mma
T.copy(cL0[slot], workspace_o[:, tile*128:..])  # fixpipe 写 GM 列段
```

但 `008` 的 **fixpipe∥mma 核内重叠**靠的是硬件 `unitFlag`(`Mmad` 0b11 + `Fixpipe` 0b11),且 mma 与
fixpipe 之间**无软件 flag**(纯硬件流水建立依赖)。一旦拆到内核用 `T.mma` + `T.copy(L0C→GM)`,两个绑定的
codegen 都把 `unitFlag` 落成默认 **0** → fixpipe 退化成独立搬运:**要么丢 008 的重叠(慢),要么(无软件
flag 时)cL0 RAW 竞争(错)**。

## 根因(逐路径核实)

| 路径 | 事实 | 结论 |
|---|---|---|
| C++ 模板 `tl::ascend::mma` | `common.h:188` 签名 `mma<T1,T2,M,N>(A,B,C,init,K, n_actual=N, unitFlag=0)` —— 早有 `unitFlag` | 模板支持 |
| `tl::ascend::copy_l0c_to_gm` | `common.h:227` 签名 `(dst,src, realDstN=1, realTailM=0, realTailN=0, unitFlag=0)`,`tileCopier(dst,src,unitFlag)` 直插 `FixpipeParams` | 模板支持 |
| 内核 `T.mma` 的 codegen | `codegen_ascend.cc` `MmaCodegen` **硬编码**只发 `(A,B,C,init,K)`(args[4]/[5]) | `unitFlag`/`n_actual` 取默认 → **接不通** |
| 内核 `T.copy(L0C→GM)` 的 codegen | `CopyCodegen` 的 `kCopyOpExtraArgs["copy_l0c_to_gm"]==3`(只发 realDstN/realTailM/realTailN);`AscendCopy::Lower` 的 l0c2gm 块不发第 4 个 runtime 参 | `unitFlag` 永不发 → **接不通** |

> 关键确认:**列段行间距不需要新参**。`AscendCopy::Lower` 的 l0c2gm 用 `compute_strideN(dst)` 从 dst
> buffer 末维推出 `realDstN`(= 参考 `fixParams.dstStride = nSize`,PV 为 512),写 `workspace_o[:, tile*128:..]`
> 列段时行间距天然正确。早先撤销的旧 012(`dst_row_stride`)正因为多此一举而被否。本 012 只补 `unitFlag`。

## 为何内核侧 / 3rdparty 不能绕

- **内核侧表达不了**:`unitFlag` 是 `Mmad`/`Fixpipe` 的硬件流水标志,只能在 codegen 发指令时落参;
  TileLang 内核层没有任何现有途径让 `T.mma`/`T.copy` 传出它。
- **不碰 3rdparty**:`catlass` 的 `Mmad`/`Fixpipe`/`CopyL0CToGmTla` 已经支持 `unitFlag`(`FixpipeParams.unitFlag`),
  无需改;我们只改自家包装模板的**调用方暴露**(codegen 发参 + 绑定加 kwarg)。
- **SWA 走非-pto codegen**:`rt_mod_ascend.cc` 的 `target.build.tilelang_ascend` → `CodeGenTileLangAscend`
  (`codegen_ascend.cc`,发 `tl::ascend::` 模板);008–011 改 `tl::ascend::gemm_v0_fixp`/`mma` 且对 SWA 生效,
  即坐实 SWA 用这条路径。`codegen_ascend_pto.cc`(`target.model=="pto"`)用 `pto-isa`(**3rdparty**)的另一套
  `mma<...,M,N,K>(A,B,C,init)` 模板,SWA 不走、也**不应改**(3rdparty)。故本改只动非-pto codegen。

## 修法(5 处,均兼容)

1. **`tilelang/language/customize.py` `npu_gemm`(= `T.mma`)**:加 `n_actual=None, unit_flag=None`;两者皆 `None`
   时发 legacy 6 参 `mma<...>(A,B,C,init,K)`(现有 examples 调用方字节不变),否则补 `(n_actual or N, unit_flag or 0)`。
   **另加 `k_actual=None`**:覆盖从 `A` 末维推的 runtime `K`(操作数保持整块,只收缩 `k_actual` 列)。修
   `T.mma(p_l0a[pp,:,0:winm], …)` 的 `int() argument must be … not 'Var'` —— 符号切片操作数让 `access_ptr` 对
   `winm` 调 `int(Var)`;改传整块 + `k_actual=winm`(= `gemm_v0_fixp` 的 `kSize`/`k_actual`)。纯 Python(`K` 本
   就在 codegen 的 `args[5]` runtime 位,无需改 codegen),默认 `None` 保持现有调用方不变。
2. **`src/op/ascend.cc` `ascend_mma`**:`set_num_inputs(6)→(-1)`(变参,容 6 或 8)。
3. **`src/target/codegen_ascend.cc` `MmaCodegen`**:发完 args[4]/[5] 后,`for i in [6, args.size())` 续发尾参
   (6 参调用方无尾参 → 不变)。
4. **`tilelang/language/copy.py` `npu_copy_v2`(= `T.copy`)**:加 `unit_flag=None`;非 `None` 时把它作第 6 个
   位置参追加进 `tl.ascend_copy`。
5. **`src/op/ascend.{h,cc}` `AscendCopy`** + **`codegen_ascend.cc` `CopyCodegen`**:`AscendCopy` 加 `PrimExpr unitFlag`
   成员(ctor 从可选 args[5] 解析,缺省 `Integer(0)`);`Lower` 的 l0c2gm 块把 `unitFlag` 作第 4 个 runtime 参
   push(在原 dead 参之前);`kCopyOpExtraArgs["copy_l0c_to_gm"] 3→4`。

## 兼容性证据

- **`T.mma`**:`set_num_inputs(-1)` 容旧的 6 参;`MmaCodegen` 仅在 `args.size()>6` 时发尾参 → 现有
  `T.mma(A,B,C,init=...)`(sparse_flash / HISA examples)发 `mma<...>(A,B,C,init,K)` **逐字节不变**。
- **`T.copy(L0C→GM)`**:现有调用方 `unit_flag=None` → 不追加第 6 位置参 → `AscendCopy` 解析 `unitFlag=0`。
  `CopyCodegen` 表升到 4 后,所有 `copy_l0c_to_gm` 显式发 `, 0`;C++ 模板 `unitFlag` 默认本就是 0 → **行为等价**
  (仅生成文本多一个 `, 0`)。`atomic_add_l0c_to_gm` 是独立 op、子串不与 `copy_l0c_to_gm` 命中、表项保持 3 →
  不受影响。
- **回归**:`examples/flash_attention/paged_flash_attn_bhsd.py` + `examples/developer_mode/sparse_flash_attn_developer.py`
  均含内核级 `T.copy(L0C→GM)`(gemm_v0 输出),验证数值不变即证字节兼容。

## 必要性

- 不接通 `unitFlag` 就无法在内核层忠实复刻 `ComputeMm2`/`ComputeMm1` 的 `cL0TensorPingPong`
  (fixpipe∥mma 重叠);而把 V 切片进 KV 环槽(Layer ③ 的核心)**要求**逐 tile 由内核驱动 mma+fixpipe,
  单体 `gemm_v0_fixp` 做不到(其 B 是连续单 buffer、`realDstN` 硬编码为模板 N,per-tile 调用会写错 dstStride
  —— 正是旧 012 试图用 `dst_row_stride` 补救、被否的死胡同)。
- 这是「可复用拆分原语」而非「给单体 gemm 堆参数」:`T.mma`(单次 mma)+ `T.copy(L0C→GM, unit_flag)`
  (融合-带-行间距 fixpipe,行间距走 `compute_strideN`)是通用 cube 拆分件,后续 QK/PV/其它 cube 算子皆可复用。

## 忠实性

逐指令对齐 `block_cube.h`:mma 的 `unitFlag = (kL0==last) ? 0b11 : 0b10`(:910),fixpipe `unitFlag=0b11`
(:932)且 `fixParams.nSize == mmadParams.n`(由内核把 dst 列段切成 mma 实际写的 `n` 列保证),mma 与
fixpipe 之间**无软件 flag**。解构后的内核指令流与单体 `gemm_v0_fixp` 内层逐字节同构,只是循环体从 C++ 模板
移到 TileLang 内核(为插入 KV 环槽载入)。

## 状态

⏳ 待合入 `ascendc_pto`(独立 wip 分支 → `--no-ff` merge);随后内核侧 PV 单缓冲解构 + `get_kernel_source`
验「mma→fixpipe 之间无 set_flag/wait_flag/pipe_barrier」+ SWA 5/5 + 回归。
