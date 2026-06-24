# 003 · Ascend 新增 `copy_pa` 原语 —— 分页 KV 直读进 L1（= Ascend C 的 `DataCopyPA`）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | 独立合并单元待建 `wip/ascend-copy-pa`（基于 `ascendc_pto`）；当前在编译分支 `wip/build-001-002`（含 001+002+003，commit `5bdc6165`） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`copy_pa` 模板）、`src/op/ascend.{cc,h}`（builtin）、`src/target/codegen_ascend.{cc,h}`（handler）、`src/transform/common/operation_config.h`、`src/transform/ascend_combinecv.cc`（pipe/读写 + cube 归类） |
| **是否必须** | 是 —— 不加的话 KV 只能逐行 gather 走 GM workspace,是和 Ascend C 性能差距(~6×)的最大来源 |
| **是否兼容** | 是 —— **纯新增**(一个新 op + 新模板),不改任何现有 op / codegen 路径 |
| **状态** | 已在容器重编 + 验证:**SWA 正确性通过**,**性能 11050us→5330us(↓2.1×)**;待抽成独立分支合入 `ascendc_pto` |

---

## 1. 为什么需要它（msprof 实测）

device 侧 msprof 对比(swa_prefill,每次 kernel):

| | Ascend C | TileLang(003 前) |
|---|---|---|
| Task Duration | **1838 us** | **11050 us(6×)** |
| aic_mac(矩阵乘) | 470 | 458 ← 计算量一样 |
| aic_mte2(cube GM→L1) | **1563（占 99%）** | 653 |
| aiv_scalar(gather 寻址) | 640 | **2853（4.5×）** |
| aiv_mte2 / aiv_mte3(向量 GM 读/写) | 338 / 213 | **2787 / 3122（8× / 15×）** |

差距**全在数据搬运**:Ascend C 用 `DataCopyPA` 在 **cube** 上把分页 KV 窗口**直读进 L1**
(`swa_block_cube.h::ComputeMm1/Mm2`),cube 的 mte2 占满、和 scalar/fixpipe overlap,
vector 很轻;TileLang 反过来——cube 空等,vector 被**逐行 gather(GM→UB→GM)+
`workspace_kv` GM 中转**拖死,且各 pipe 串行不 overlap。

**003 落地后实测**(device 侧,swa_prefill):Task Duration **11050us → 5330us(↓2.1×)**,
其中 `aiv_mte2` 2787→**282(↓10×)**、`aiv_mte3` 3122→**197(↓16×)**——gather 那趟 GM
搬运彻底消失。现在 TileLang 是 Ascend C(1838us)的 **~2.9×**(原 6×)。剩余差距主要在
`aiv_scalar`(2347us,主要是固定 BI=128 + 掩码引入的逐列位置标量计算)和 cube/vector 的
overlap,留待后续优化。

## 2. 为什么不能用现有 TileLang 原语在内核侧写出来

`DataCopyPA` 的核心是一个**数据相关的按页 `while` 循环**:

```c
while (已拷行数 < window) {
    page      = blockTable[s2Idx / blockSize];          // 间接寻址
    copyRows  = blockSize - (s2Idx % blockSize);        // 本页能拷多少行(运行期)
    DataCopy(L1[已拷行数], kv[page * kvStride + ...], Nd2Nz(copyRows 行));
    s2Idx += copyRows;  已拷行数 += copyRows;
}
```

要在内核侧用现有原语写出它,有**三处当前 TileLang 表达不了**:

1. **循环次数是运行期的。** 窗口跨几页取决于 `s2Idx`(=`ori_left`,每个 token 不同),
   而 `T.serial(N)` 要求 `N` 是**编译期常量**。即使把页数放宽到一个编译期上界(窗口 ≤128、
   blockSize 已知 ⇒ 至多 2 页)做**固定展开**,也躲不过下面两条。
2. **每页拷贝的行数是运行期的**(`blockSize - s2Idx%blockSize`)。`T.copy` 的搬运
   extent 必须**编译期已知**(它要据此算 `DataCopy`/`Nd2Nz` 的参数),没法接受一个
   运行期行数。
3. **给 matmul 的子窗口偏移是运行期的。** 即便把整页(定长)搬进 L1 再让 matmul 取
   `[brow : brow+window]`,这个起点 `brow` 是运行期的,而 `gemm_v0` **拒绝切片/运行期
   偏移操作数**(本项目早先已撞到 "Unsupported BufferLoad")。

`T.copy` 本身只按 src/dst 的 **scope** 自动选 `copy_gm_to_l1` 等,**根本没有 block table
这个操作数**,表达不了分页间接寻址。所以这不是"懒得用现有原语",而是现有原语在
**控制流(运行期循环)+ 搬运 extent(运行期)+ 间接寻址(block table)** 三个维度上都不够。
这正是项目规则里的「**TileLang 表达不了→加原语**」。

> 退一步的"内核侧批量 copy"(假设窗口不跨页 ⇒ 一次定长 `T.copy(kv[page, brow:brow+128])`)
> 只在 blockSize 足够大、窗口恰好不跨页的**退化配置**下成立,且依赖上面第 3 条仍表达不了的
> 运行期 `brow` 切片;它也**不忠实**于参考的分页 dataflow(参考是按页循环、可跨页)。

## 2b. 这个原语通用吗?（不是只有本算子用）

**通用。** `DataCopyPA` 是 paged-attention 的**通用 KV 加载操作**,在参考仓库里被**多个
attention 算子共用**(`grep DataCopyPA ops-transformer/` 命中):`sparse_attn_sharedkv`、
`sparse_flash_attention`、`kv_quant_sparse_flash_attention` 等。分页 KV cache + block table
是 PagedAttention 的标准布局,"把分页窗口直读进 L1"是任何 paged 注意力 cube 侧都要做的事。

并且——**把一个 C 模板封装成原语,正是 TileLang Ascend 后端本来的工作方式**:
`gemm_v0`、`copy_gm_to_l1`、`copy_l1_to_l0a`、所有 `T.tile.*` **无一例外**都是
`common.h` 里的 C 模板,经 builtin + codegen + Python 绑定暴露成 `T.xxx`。`copy_pa` 与它们
**同构**,不是特例,也不是"把某算子的私有逻辑塞进编译器"。

## 3. 修法（纯新增一个 `copy_pa` 原语）

`DataCopyPA` 是参考实现自己的 helper(`sparse_attn_sharedkv_common.h`,不是 catlass),
逻辑**忠实照搬**进 TileLang:

- **`common.h::copy_pa<T>`**:PA_ND 分支的忠实移植——`while (copyFinishRowCnt <
  copyRowNum)`:`blockTableGm.GetValue` 拿页号、算页内偏移、`AscendC::DataCopy`
  (Nd2Nz)把该页内的连续行 GM→L1、推进。窗口落在一页内就是**一次 DataCopy**。
- **`tl.ascend_copy_pa` builtin**(`op/ascend.{cc,h}`,变参)。
- **`CopyPACodegen`**(`codegen_ascend.{cc,h}`):发 `tl::ascend::copy_pa<T>(dst[off],
  kv[off], blockTable[off], 标量参数…)`,buffer 操作数的取法与 `GemmOpCodegen` 一致。
- **登记分类**:`operation_config.h` 加 `{"copy_pa", {{{0,"write"},{1,"read"},
  {2,"read"}}, "PIPE_MTE2"}}`(dst 写、kv/table 读、MTE2 流水),`ascend_combinecv.cc`
  的 `callnodeMapPos_` 加 `{"copy_pa","cube"}`,使 auto-sync 正确同步 L1 写、CV 拆分把它
  放到 cube。
- **`T.copy_pa` Python 绑定**(`language/ascend.py`,照 `gemm_v0` 构造 `tir.Call`)。

内核侧:删掉整段逐行 gather 和 `workspace_kv`(算子从 16 个参数减到 15、`workspace_idx`
变 `[13,14,15]`),cube 直接:

```python
win = ori_right - ori_left          # 窗口行数 (<= BI)
T.copy_pa(kv_l1, ori_kv, ori_block_table,
          ori_block_size, N2, D, ori_block_size*N2*D, ori_table_len, D,
          win, BI, b, 0, ori_left, 0)
```

`copy_row_num = win`,kv_l1 的 `[win:BI)` 行不写、其 QK 列在 softmax 里被 mask 成 -inf
(与原 kernel 一致;参考实现是直接用变长 N=win)。

## 4. 为什么兼容

**纯新增**:新 op `ascend_copy_pa` + 新模板 `copy_pa` + 新 codegen 分支 + 两张表各加一行。
没有改动任何现有 op、模板、codegen 路径,因此对其它算子零影响。待容器重编后与 001/002
同批跑回归(`paged_flash_attn_bhsd.py` / `sparse_flash_attn_developer.py`)确认。

## 5. 忠实性

`copy_pa` 是参考 `DataCopyPA` 的逐行移植,内核因此与 Ascend C 的 cube dataflow 一致
(KV 直读进 L1、无 gather、无 workspace_kv)。这是「TileLang 表达不了→加原语」,不是绕行、
不是发明新方法。
