# 002 · Ascend `gemm_v0` —— 增加 N 方向切分(对齐 Ascend C 的 matmul 切分）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/ascend-gemm-n-tiling` · `26116e27`（**独立基于 `ascendc_pto`**，与 001 互不依赖，可单独合并） |
| **改动文件** | `src/tl_templates/ascend/common.h`（`gemm_v0` 模板） |
| **是否必须** | 是 —— 不改的话 SWA 的 PV 矩阵乘要么溢出 L0B 崩溃，要么只能在内核里切 K 绕行（而绕行又引出同步/数值问题） |
| **是否兼容** | 是 —— 小 N / `transpose_B` 路径逐字节不变；大 N `transpose_B=false` 原本会溢出（没有可用调用），只新增能力 |
| **状态** | 已验证：SWA 快速正确性测试通过（`1 passed`）。待回归例子复核后合入 `ascendc_pto`（与 001 分两次独立合并） |

---

## 1. 算子为什么需要切 N

忠实复刻的 SWA 第二个矩阵乘 PV（`O = P @ V`，`swa_block_cube.h::ComputeMm2`）
是 `<M=gSize=64, N=headDim=512, K=s2(窗口)≤128>`。Ascend C 的 matmul 模板对
**N 方向按 `N_SPLIT_SIZE=128` 切分**（`ComputeMm2` 里 `nL1Loops = 512/128 = 4`），
每个 N-tile 单独把 `(K x 128)` 的 V 子块载入 L0B、单独 fixpipe 出去。这是
Ascend C 本来的切分，不是可选优化。

## 2. 现象

TileLang 的 `gemm_v0`（`common.h`）**只切 K（`kL0Size=128`）不切 N**：每次 mma 把
**整个 N** 的 B 操作数载入 L0B，slot 大小 `N * kL0Size`。PV 的 `N=512, kL0Size=128`：

```
B-tile = N * kL0Size * sizeof(half) = 512 * 128 * 2 = 128KB  >  L0B 64KB
```

→ L0B 溢出，运行期 aicore `CCU instruction address check error` 崩溃。

为了绕开，之前在**内核**里把 PV 的 K 切成两半（每半 `512*64*2=64KB` 勉强装下）、
再为对齐跨核同步点而把 `workspace_s/p` 的读写拆成列半（strided UB 子块拷贝）。结果
strided UB 子块拷贝行步长不对，softmax 的 `s_ub` 被逐行错位污染，LSE 出现
**逐行单调恶化**（row0 完美 → row31 全错）的数值错误。**绕行链最终在数值上崩了。**

## 3. 根因

`gemm_v0` 模板的循环只有 K 维（`kL0Idx`），mma 一次吃满整个 N。当 `N * kL0Size *
sizeof` 超过 L0B 单 slot 预算时必然溢出。这是 `gemm_v0` 的能力缺口 —— 它从未被需要
切 N 的算子用过（既有示例的 N 都 ≤128）。

## 4. 为什么不能在内核侧解决

`gemm_v0` **拒绝切片操作数**（`kv_l1[:, n0:n0+128]` 报 "Unsupported BufferLoad"），
所以内核无法把 V 的 N 子块喂给 `gemm_v0`；要切 N 只能预先把每个 D 子块 gather/拷贝
成独立完整 buffer，这会引入大量 strided 拷贝和额外的 workspace 读、再带出跨核同步点
数目不匹配等一连串绕行（且 L1→L1 拷贝在 codegen 里根本 `not implemented yet`）。
**切 N 属于矩阵乘原语本身的职责，应在 `gemm_v0` 修复。**

## 5. 修法

在 `gemm_v0` 内部对 N 加一层切分循环（`nL0Idx`），逐 N-tile：① 载入该 tile 的
`(kL0Size x nTile)` B 子块到 L0B；② mma 写入 L0C 对应的列带。N-tile 大小取
`nTile = min(N, 32KB / (kL0Size * sizeof(T)))`（half ⇒ 128，正好对齐 Ascend C 的
`N_SPLIT_SIZE`）。**只对 `transpose_B=false` 且 `N > nTile` 的路径真正切**；
`transpose_B`（QK，N=block_I≤128）与所有小 N 调用 `nL0split==1`，与原实现逐字节相同。

子块偏移**直接取自 catlass 的 `tla` 分形布局**（不是猜的）：

```cpp
// L0C 列偏移（tla::MakeLayoutL0C 的 N1 stride = RoundUp16(M)*16）
cNOffset = nL0Idx * nTile * roundUp16(M);
// L1 的 B 是 zN 布局（tla::MakeLayout<zN> 的 C1 stride = RoundUp16(K)*ELE_PER_C0）
bNOffset = nL0Idx * nTile * roundUp16(K);
```

两者都与 `gemm_v0` 原有的 K 偏移 `B[kL0Idx*16*kL0Size]`（zN 的 K-row stride）自洽，
也与 catlass 自己的分块 mmad（`block_mmad_pingpong_tla.hpp` 用
`GetTile(L0C, MakeCoord(0, nPart), …)` 逐 N-tile 写）等价。L0B slot 改为
`pp * (nTile * kL0Size)`，配合 K 的 ping-pong，峰值 `2*nTile*kL0Size*2 = 64KB` 正好
装下 L0B。

**流水化（对齐 Ascend C 的 overlap，非串行）**：把 `(N-tile, K-tile)` 两层循环用
一个 `tileIdx` 拍平，L0A/L0B 的 ping-pong **只 prime/drain 一次**、在整条 tile 序列
上连续滚动 —— 当前 tile 的 mma 在执行时，下一个 tile 的 L1→L0 载入已经写进另一个
ping-pong buffer，所以 N-tile 与 K 真正 overlap（最早一版「逐 N-tile drain」会把 tile
串起来，是没逐指令复刻 Ascend C matmul 流水的，已改正）。`nL0split==1` 时
`tileIdx==kL0Idx`，与原 K ping-pong 逐字节相同。

## 6. 为什么它是兼容性修改（及待验证项）

设计上：
- `transpose_B=true`：`nTile==N`、`nL0split==1`，循环体与原实现**逐字节相同**。
- `transpose_B=false` 且 `N ≤ nTile`（既有示例都是）：同样 `nL0split==1`，**不变**。
- `transpose_B=false` 且 `N > nTile`：原实现必然溢出 L0B（没有能正常工作的调用），
  本改动只是**新增**「之前根本跑不了」的能力，不改变任何现有行为。
- `static_assert(transpose_B || N % nTile == 0)`：大 N 非整除会编译期报错而非静默出错。

待容器从源码重编后验证（与 001 同批）：
- `examples/flash_attention/paged_flash_attn_bhsd.py` 应仍 `Kernel Output Match!`
- `examples/developer_mode/sparse_flash_attn_developer.py` 应仍 `Test Passed!`
- SWA 快速正确性测试通过（PV 单次 matmul、跨核同步 1:1）。

## 7. 忠实性说明

切 N 后，TileLang 的 QK/PV 都变成**单次 `gemm_v0`，切分与 Ascend C 的
`ComputeMm1`（K 切 4×128、N=窗口）/`ComputeMm2`（N 切 4×128、K=窗口）完全一致**，
内核里**不再有切 K 两半、列半同步配平、strided 拷贝等任何绕行**。这正是
*"TileLang 表达不了→加原语/修编译器，绝不在内核里发明绕路"* 的要求。

## 8. 必要性与通用性

**必要性(为什么非改不可)。** PV 矩阵乘是 `<M=64, N=D=512, K=窗口>`。`gemm_v0` 原本
**只切 K 不切 N**,一次 mma 把整个 N 的 B 操作数载入 L0B = `512*128*2 = 128KB > L0B 64KB`
→ **溢出、cube 运行期崩**。而 Ascend C 的 `ComputeMm2` 本来就按 `N_SPLIT_SIZE=128` 切 N。
要忠实复刻、又要能跑,**必须让 gemm 切 N**。内核侧切不了(§4:`gemm_v0` 拒绝切片操作数、
L1→L1 拷贝 codegen 不支持);唯一的内核侧替代(切 K 两半)又引出跨核同步点不匹配、列半
strided 拷贝,最终把数值搞崩。所以这一处**只能在 gemm 原语里修**。

**通用性(不止本算子)。** N 切分是**矩阵乘原语的通用能力**,与 sparse_attn_sharedkv 无关:
**任何** B-tile(`N*kL0Size*elem`)超过 L0B 预算的 gemm 都需要它——在此修复前,`gemm_v0`
**根本做不了大 N 的 matmul**(必溢出 L0B)。修复是**兼容性的**:小 N / `transpose_B` 调用
`nL0split==1`、`tileIdx==kL0Idx`,与原实现**逐字节相同**;它只是**新增**了"以前根本跑不了
的大 N matmul"这一类能力(如任何 head_dim 较大的 attention PV)。catlass 自己的分块 mmad
(`block_mmad_pingpong_tla.hpp`)本就通用地切 M/N/K,本修复只是把 `gemm_v0` 拉齐到同一
水平。流水化(prime/drain 一次、连续 ping-pong)也是通用的 overlap 改进。
