# 009 · Ascend `gemm_v0_fixp` K-累加 + `n_actual` + `cl0_base` + `dbg_barrier`（QK 走统一 fixp 路径 = `ComputeMm1`，与 PV 共享 cL0；内置卡死诊断开关）

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支** | `wip/gemm-v0-fixp-kaccum`（**独立基于 `ascendc_pto`**，含 001–008） |
| **改动文件** | `src/tl_templates/ascend/common.h`(`gemm_v0_fixp`)、`tilelang/language/ascend.py`(绑定加 `n_actual`/`cl0_base`/`dbg_barrier`)、`src/target/codegen_ascend.cc`(`GemmFixpOpCodegen` 多发三参)、`src/op/ascend.cc`(`ascend_gemm_v0_fixp` `set_num_inputs` 7→10) |
| **是否必须** | 是 —— 忠实复刻要求 QK(`ComputeMm1`)与 PV(`ComputeMm2`)走**同一套** matmul 结构(K-累加 → 融合 fixpipe → `unitFlag` → 共享 cL0),QK 不能再用 `gemm_v0` 留驻 + 单独拷的绕行 |
| **是否兼容** | 是 —— PV/现有 caller:`K≤128`、`n_actual=N`、`cl0_base=0`、`dbg_barrier=false` 全默认 → 逐字节不变;回归两 example 不走 `gemm_v0_fixp`(SWA 专用) |
| **状态** | ⏳ 调试中。**必要性已确认**:cfa/scfa 的 QK N=512(cmp 满块=s2BaseSize)→ 多 N-tile → resident L0C(M·N·4 ≤512KB)远超 128KB → **per-N-tile 融合 fixpipe 是容量功能必需**(非仅性能/保真),故 009 必做。**核心未决**:多-K `unitFlag`(0b10 累加)卡死;NPU 实测 `dbg_mode=1`(每 mma 前 PipeBarrier 全串行)**仍卡** → 排除 M 流水排序,病根 = `0b10` 值本身。现用 `dbg_mode=2`(0b00 中间)探。 |

> 这是**回收编号后的新 009**。上一版 009(同名 K-累加,无 `dbg_barrier`)在单个 QK 调用内多-K
> `unitFlag` 累加处卡死、远程无法定位(模板不在 dump 里),已废弃删分支。本版加 `dbg_barrier`
> 把可观测性焊进原语,作为整条忠实 cube 重建(退回 parity 8061a9e)的 matmul 基础。

---

## 1. 算子为什么需要它

参考里 QK(`ComputeMm1`)和 PV(`ComputeMm2`)是**同一套** matmul 结构:K-累加 → 融合 Fixpipe
到 GM → `unitFlag` → 共享 cL0 ping-pong。之前 QK 用 `gemm_v0`(结果留驻 L0C + 内核里单独 `T.copy`
搬出),**不是融合 fixpipe**,是绕行。忠实复刻要求 QK 也走 fixp 路径,并与 PV 共用一个 cL0TensorPingPong。

## 2. 现象 / 缺口

008 的 `gemm_v0_fixp` 只服务 PV——PV 的 K=窗口≤128,单 K-tile,所以 008 里有
`static_assert(K <= kL0Size=128)`、`kSize=k_actual`、`cL0BufIter` 从 0 起。QK 的 K=headDim=512
需要 4 个 K-tile 累加、输出列 = 窗口宽 < N=BI、且要与 PV 接续同一个 cL0 旋转——008 都做不到。

## 3. 根因(含未决卡死)

`gemm_v0_fixp`(008)模板假设单 K-tile;`transpose_B` 路径没有运行期列数 `n_actual`;`cL0BufIter`
每次调用从 0 起,QK 与 PV 各调一次 → cL0 旋转不接续。

**★未决卡死(已逐步缩小)★**:把 QK 接上多-K `unitFlag` 累加(4 个 mma:0b10,0b10,0b10,0b11 +
Fixpipe 0b11)后 NPU **卡死**,发生在**单个 QK 调用内部**。逐条排除:
- 分开/共享 cL0 都卡、tiny-tile 屏障运行期化、所有 set/wait_flag 收支平衡 → 不是 cL0、不是 flag 死锁。
- **NPU 实测 `dbg_mode=1`(每个 mma 前 `PipeBarrier<PIPE_M>` 全串行化 M 流水,= 能工作的 `gemm_v0` 做法)
  仍卡** → **排除「缺 M 流水排序」**。
- `cmatrixSource` **不是**病根:`gemm_v0` 用 0b00 做同样的多-K 累加且数值正确 → 默认 `cmatrixSource` 已是
  `false`。
剩下唯一变量:**`unitFlag=0b10` 这个值本身**。`gemm_v0`(0b00,工作)、PV 008(0b11 单 mma,工作)都不走
0b10;QK 是唯一走 0b10 的。参考用 0b10 能跑 → 是咱们 catlass 对 0b10 的发射/硬件交互有问题。
**当前探针 `dbg_mode=2`**:中间 K-tile 改 0b00(纯累加)、只末 tile 0b11+fixpipe,让「末 mma 0b11 +
fixpipe 0b11」配对与已验证的 PV 完全一致,数值不变。若 dodge 掉卡死 → 0b00-中间 即为修法。

## 4. 为什么不能在内核侧解决

K 累加(跨 4 个 K-tile 把 D=512 收缩进同一 cL0 slot、首 tile `init`、末 tile `unitFlag` flush、再
fixpipe)在**模板内部的 kL0 循环**里;内核调不进去。`unitFlag` 是硬件 mma↔fixpipe 流水位,只能在
原语里发。诊断这个流水挂死也只能在原语里加屏障开关——模板不在 `SAS_DUMP_SRC` 的 dump 里,内核侧无从观测。

## 5. 修法

`gemm_v0_fixp`(`common.h`):
- **去掉 `static_assert(K <= kL0Size)`**;`kL0` 循环真 K-累加:per-tile
  `kRemain = k_actual - kL0Idx*128; kSize = min(128, kRemain)`,`initflag` 只在首 K-tile、`unitFlag`
  末 K-tile = 0b11(其余 0b10),累加进同一 cL0 slot,再做 fixpipe。PV(`kL0split==1`)→ `kSize=k_actual`
  单 tile,与 008 同。
- **加 `uint32_t n_actual = N`**:`transpose_B` 的输出列数(QK 窗口宽),用在 mma 的 n 与 `transpose_B`
  的 L0B 载入;非 transpose 的 PV 路径仍用 `nTile`,逐字节不变。tiny-tile `PipeBarrier<PIPE_M>` 改用
  **运行期** `mmaN = transpose_B ? n_actual : nTile`(参考 cube.h:580 用运行期 `mmadParams.n`;对 QK
  小窗口编译期 `nTile` 会漏屏障)。
- **加 `uint32_t cl0_base = 0`**:`c_base = ((cl0_base + cL0BufIter) & 1) * (M*nTile)`,让 QK 与 PV
  共用一个 cL0 旋转(QK 传一个槽、PV 接续下一个槽),= 参考的单一 `cL0BufIter` 横跨 Mm1+Mm2。默认 0 = 原行为。
- **加 `uint32_t dbg_mode = 0`(诊断位掩码,非忠实特性)**:位 0(1)= 每个 mma 前发 `PipeBarrier<PIPE_M>`
  (= `gemm_v0` 做法,实测不解决卡死 → 排除 M 流水排序);位 1(2)= 中间 K-tile 用 `unitFlag=0b00`(纯累加、
  不挂 unit flag),只末 tile `0b11`,使「末 mma+fixpipe」配对与已验证的 PV 一致、数值不变(用于 dodge 0b10
  卡死)。默认 0 = 忠实(0b10 中间)、字节不变。`set_num_inputs` 7→10,arg [9]=dbg_mode。

绑定/codegen/`set_num_inputs` 照既有范式透传(尾随默认参,arg [7]=n_actual、[8]=cl0_base、[9]=dbg_barrier;
`set_num_inputs` 7→10)。

## 6. 忠实性（对照 `block_cube.h` `ComputeMm1`/`ComputeMm2`）

| TileLang | Ascend C 参考 |
|---|---|
| `kL0` 循环 K-累加,`initflag` 首 tile,`unitFlag`=末 tile?0b11:0b10 | `cmatrixInitVal=(kL1==0&&kL0==0)`,`unitFlag=(末)?0b11:0b10`(cube.h:575-578) |
| 末 K 后 fixpipe(0b11)到 dst | `if(kL1==末) Fixpipe(unitFlag=0b11)`(cube.h:591-605) |
| tiny-tile barrier 用运行期 `mmaN` | `if((m/16)*(mmadParams.n/16)<10) PipeBarrier<PIPE_M>`(cube.h:580) |
| `n_actual` = transpose_B 列数 | `nL1SizeAlign` = 窗口宽 |
| `cl0_base` 让 QK/PV 共享一个 cL0、连续旋转 | QK/PV 共用**同一个** `cL0TensorPingPong` + 一个 `cL0BufIter`(cube.h:97/559/833) |
| `dbg_barrier`(诊断,非忠实) | 参考无;参考靠全局 prime 一次的持续 `abL0BufIter`(`AllocEventID` cube.h:225-226 + Mte1MmABEventId)|

## 7. 兼容性证据

1. **默认值保兼容**:PV `K≤128` → `kL0split==1` → `kSize=k_actual`(单 tile,与 008 同);`n_actual=N`、
   `cl0_base=0`、`dbg_barrier=false` 全默认 → PV 逐字节不变。
2. **回归**:`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 不调 `gemm_v0_fixp`(SWA 专用),
   不受影响;仍需复跑确认不破坏其它原语。
3. **单 caller 扩展**:SWA QK 是唯一新走 K>128 路径的 caller,与 PV(K≤128)共用模板、互不影响。

## 8. 必要性

QK 忠实复刻(融合 fixpipe + unitFlag + 与 PV 同结构 + 共享 cL0)的前提;`dbg_barrier` 是在本地无 NPU、
模板不可 dump 的条件下,坐实多-K unitFlag 卡死病根的唯一可观测手段。
