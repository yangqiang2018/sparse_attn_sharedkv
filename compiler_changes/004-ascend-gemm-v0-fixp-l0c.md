# 004 · Ascend 新增 `gemm_v0_fixp` 原语(按 N-tile 即时 fixpipe,根治 PV 的 L0C 越界)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-gemm-v0-fixp` · `03cc14b6`(基于 `ascendc_pto`) |
| **改动文件** | `src/tl_templates/ascend/common.h`(新模板 `gemm_v0_fixp`)、`src/op/ascend.{cc,h}`(builtin `ascend_gemm_v0_fixp`)、`src/target/codegen_ascend.{cc,h}`(dispatch + `GemmFixpOpCodegen`)、`src/transform/common/operation_config.h`(读写 + PIPE)、`src/transform/ascend_combinecv.cc`(cube 归类)、`tilelang/language/ascend.py`(`T.gemm_v0_fixp` 绑定) |
| **是否必须** | 是 —— 不改的话 SWA 的 cube/vector 软件流水会因 L0C 越界在运行期崩溃 |
| **是否兼容** | 是 —— 纯新增原语,`gemm_v0` 与所有既有调用逐字节不变 |
| **状态** | 待容器从源码重编后验证(SWA 快测 + msprof + 回归例子) |

---

## 1. 算子为什么需要它

忠实复刻的 SWA cube/vector 软件流水(每核任务循环,`QK(任务 i) ∥ softmax(i-1) ∥ PV(i-2)`
跨核重叠,对齐 Ascend C 的 `ProcessBalance`/`PreloadPipeline`)需要在 cube 上**同时**持有
多个任务的 L0C 结果。而 PV(`O = P @ V`,`<M=gSize=64, N=headDim=512, K=窗口≤128>`)的
输出 `O[64,512] float32 = 128KB` 自己就占满整片 L0C(物理上限 128KB)。

Ascend C 的 `ComputeMm2`(`swa_block_cube.h:622-`)**从不**把整块 O 留在 L0C:它按
`N_SPLIT_SIZE=128` 切 N,每个 `[m,128]` tile 在 K 累加结束(`kL1==最后一轮`)时**立即
`Fixpipe` 搬到 GM**(`swa_block_cube.h:591-605`),并用 `cL0TensorPingPong`(2×64KB 半块)
做 fixpipe∥mma 重叠。所以 Ascend C 的 O 在 L0C 任一时刻只占一个 tile。这是它本来的
tiling,不是可选优化。

## 2. 现象

cube/vector 流水版(算子 `wip/swa-cv-pipeline` 的 `ab9a577`)运行期崩溃:

```
The operation address of L0C exceeds the maximum range of L0C.
... aic error ... blk:11 ... fftsplus aicore error
```

## 3. 根因

内核里 `acc_s_l0c=[64,128]f32=32KB`(QK 结果)与 `acc_o_l0c=[64,512]f32=128KB`(PV 结果)
都在 kernel scope 声明。串行版本(`8e2f8df`)能跑,是因为内存规划把两者**复用**到同一段
L0C(活跃期不相交:QK 结果搬到 `workspace_s` 后 `acc_s_l0c` 即释放,PV 才用 `acc_o_l0c`),
峰值 = `max(32,128) = 128KB`,刚好不溢出。

但默认内存规划走 `PlanMemoryForScopeLinear`(`ascend_memory_planning.cc`,开关
`tl.ascend_memory_planning` 默认关),它按声明顺序排**不相交**地址、不做活跃区间复用;在
流水版里 `with T.Scope("C")` + `for j` 循环 + 条件分支撑大了活跃区间,规划器放弃复用,把
`acc_s` 与 `acc_o` 排到 `32KB + 128KB = 160KB > 128KB`。且 linear 路径 `check_overflow=false`,
无编译期守卫,只能运行期以"L0C address exceeds"崩出来。

根本问题:`gemm_v0`(002 的 N 切分)**只切 L0B、不切 L0C** —— 4 个 `[64,128]` 列带在
`gemm_v0` 返回前**同时驻留**,fixpipe 由调用方返回后单独 `T.copy(acc_o_l0c, workspace_o)`
做,所以 `acc_o_l0c` 自己就是整块 128KB。

## 4. 为什么不能在内核侧解决

要把 O 的 L0C 降到单个 `[64,128]` tile,必须按 N-tile 算一块、搬一块、复用 L0C。但:

- `gemm_v0` **拒绝切片操作数**(`kv_l1[:, n*128:...]` 报 "Unsupported BufferLoad"),所以
  内核无法把 V 的 N 子块喂给 `gemm_v0` 自己循环 N(这正是 002 当初把 N 切分下沉进
  `gemm_v0` 的原因)。
- 用 4 次 `copy_pa`(各加载一个 D 子块到独立 buffer)+ 4 次 `gemm_v0` 也可绕,但 Ascend C
  的 `ComputeMm2` 是"V 整窗口一次进 L1、再按 N-tile 从 L1→L0B"——4 次 `DataCopyPA` 既不
  忠实(参考只 1 次),又把分页加载的标量开销 ×4。
- 把 fixpipe 留作内核侧 `T.copy`、只让 gemm 写小 L0C 也不行:gemm 必须在内部按 tile 算完
  就搬,fixpipe 与 N-tile 循环是耦合的,无法拆成"gemm 出整块 + 内核单独 copy"。

所以"按 N-tile 即时 fixpipe + 单 L0C slot 复用"属于**矩阵乘原语本身的职责**,应在原语层
解决 —— 这正是 *"TileLang 表达不了→加原语,绝不在内核里发明绕路"* 的要求。

## 5. 修法

新增独立原语 `gemm_v0_fixp`(不改 `gemm_v0`)。它以 `gemm_v0`(含 002 的 N/K 切分与
L0A/L0B ping-pong)为骨架,叠加:

1. **L0C 累加器 `C` 变成单个 `[M, nTile]` slot**(PV:`[64,128]=32KB`)。所有 N-tile 复用
   它(`mma(..., C[0], ...)`)。
2. **每个 N-tile K 累加结束后,立即 `copy_l0c_to_gm(dst[nL0Idx*nTile], C[0], realDstN=N)`**
   把这一列带搬到 GM 目的张量(= `workspace_o`),与 `ComputeMm2` 的 per-tile `Fixpipe`
   一一对应。
3. **`M_FIX`/`FIX_M` 握手**保护单 slot 复用:`mma` 写 `C[0]` → `M_FIX` → fixpipe 读 `C[0]`
   → `FIX_M` → 下一 N-tile 的 `mma` 才能覆写 `C[0]`。

签名:`gemm_v0_fixp<T1, T2, LayoutGM, M, N, K, transpose_A, transpose_B>(A_l1, B_l1,
C_l0c, dst_gm, l0a, l0b, clear)`。`M,N` 取自 GM 目的张量,`K` 取自 `A`。

L0C 预算修复后:`acc_s_l0c 32KB + acc_o_l0c 32KB = 64KB`,即使流水里不复用也远低于 128KB。

> 当前为**单 slot**实现(求正确、修崩溃):fixpipe 与下一 tile 的 mma 串行,未做 Ascend C 的
> 2×64KB cL0 ping-pong 重叠。fixpipe∥mma 的 ping-pong 重叠留作后续(对应 cube `aic_fixpipe`
> 比率从 ~0.14 拉向 Ascend C 的 ~0.99)。

## 6. 为什么它是兼容性修改(及待验证项)

- **纯新增**:`gemm_v0` 模板与全部既有调用逐字节不变;新原语是独立 builtin/codegen 分支。
- 接线只新增条目(`operation_config` 两表把 `dst` 标 write、PIPE_M;`ascend_combinecv` 归
  cube),不改既有 op 的行为。
- 对抗式审查(C++ 模板 helper 签名、事件配对无死锁/泄漏、端到端参数顺序)结论:无
  compile-break、无 logic-bug。
- **PTO 路径未补**:只有非-PTO codegen(`codegen_ascend.cc`)加了 `GemmFixpOpCodegen`。本
  算子走默认 `target="auto"` → 非-PTO,不受影响;且无其它算子使用该新原语,故**不破坏任何
  现有算子**。若日后切 `target="pto"` 需在 `codegen_ascend_pto.cc` 补对应 handler。

待容器从源码重编后验证:
- SWA 快测通过(L0C 不再越界);
- `examples/flash_attention/paged_flash_attn_bhsd.py` 仍 `Kernel Output Match!`;
- `examples/developer_mode/sparse_flash_attn_developer.py` 仍 `Test Passed!`。

## 7. 忠实性说明

`gemm_v0_fixp` 让 PV 变成"按 N-tile 算完即 fixpipe 搬出、L0C 只占单 tile",与 Ascend C
`ComputeMm2` 的 per-tile `Fixpipe`(`swa_block_cube.h:591-605`)逐指令对齐。内核里不再有
"整块 O 留 L0C 再单独 copy"这一相对参考的偏离。这是 002 N 切分之上的**忠实收尾**:002
切 L0B、004 切 L0C + 即时搬出,合起来才完整复刻参考的 matmul 切分。

## 8. 必要性与通用性

**必要性(为什么非改不可)。** PV 的 `O[64,512]f32 = 128KB` 整块留 L0C 就占满整片 L0C;
SWA 的 cube/vector 软件流水要让多个任务的 L0C 结果共存(`acc_s 32KB + acc_o 128KB = 160KB`
> 128KB),必然越界(§2/§3 实测崩溃)。内核侧躲不掉(§4:`gemm_v0` 不收切片操作数、L1→L1
不支持、4×copy_pa 不忠实)。要忠实复刻 `ComputeMm2` 的 per-tile fixpipe、又要能跑,**只能
在 gemm 原语里把 fixpipe 下沉**。这是阻断性的。

**通用性(不止本算子)。** 这是**矩阵乘原语的通用能力**,与 sparse_attn_sharedkv 无关:
**任何**输出 `[M,N]` 在 L0C 偏大(`M*N*sizeof(accum)` 逼近/超过 L0C 128KB)的 gemm 都需要
"按 N-tile 即时 fixpipe、L0C 只留单 tile"——典型如 head_dim 较大的 attention PV、或任意大
M/N 的 matmul。在此修复前,`gemm_v0` 把整块结果钉在 L0C,大输出要么放不下、要么逼调用方
为流水/复用纠结。004 与 002 互补、共同把 `gemm_v0` 的能力补齐:**002 把 B-tile 约束在
L0B(切 N 的载入)、004 把 C-tile 约束在 L0C(切 N 的搬出)**,合起来 `gemm_v0` 系列才能处理
任意大的 M/N——这正是 catlass 自己的分块 mmad(`block_mmad_*`,逐 tile fixpipe)一直具备的
通用行为,004 只是把同款能力以**兼容性新增原语**的形式带给非-PTO 路径。流水/重叠也是通用
收益(per-tile fixpipe 可与下一 tile 的 mma 重叠,拉高 fixpipe 占用率)。
