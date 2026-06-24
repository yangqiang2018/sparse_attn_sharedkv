# 003 · Ascend 新增 `copy_pa` 原语 —— 分页 KV 直读进 L1（= Ascend C 的 `DataCopyPA`）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | 独立合并单元待建 `wip/ascend-copy-pa`（基于 `ascendc_pto`）；当前在编译分支 `wip/build-001-002`（含 001+002+003，commit `5bdc6165`） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`copy_pa` 模板）、`src/op/ascend.{cc,h}`（builtin）、`src/target/codegen_ascend.{cc,h}`（handler）、`src/transform/common/operation_config.h`、`src/transform/ascend_combinecv.cc`（pipe/读写 + cube 归类） |
| **是否必须** | 是 —— 不加的话 KV 只能逐行 gather 走 GM workspace,是和 Ascend C 性能差距(~6×)的最大来源 |
| **是否兼容** | 是 —— **纯新增**(一个新 op + 新模板),不改任何现有 op / codegen 路径 |
| **状态** | 待容器从源码重编 + SWA 正确性 + 性能复测;通过后抽成独立分支合入 `ascendc_pto` |

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

## 2. 为什么内核侧/现有原语解决不了

`T.copy` 按 src/dst 的 scope 自动选 `copy_gm_to_l1` 等,**表达不了 block table 的间接
寻址**;而 `DataCopyPA` 的核心是一个**数据相关的按页 while 循环**(查 block table 拿页号、
每页一次 `Nd2Nz` 的 `DataCopy`、运行期行数),TileLang 的静态 `T.serial` + 静态 copy
extent 都表达不了。所以这是「TileLang 表达不了→加原语」。

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
