# 009 · Ascend `gemm_v0_fixp` K-累加 + `n_actual`(QK 走统一 fixp 路径 = Ascend C `ComputeMm1`)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/gemm-v0-fixp-kaccum`（**独立基于 `ascendc_pto`**,含 001–008;0b0f8994 + tiny-tile 屏障修复 1c7c124a） |
| **改动文件** | `src/tl_templates/ascend/common.h`(`gemm_v0_fixp`)、`tilelang/language/ascend.py`(`gemm_v0_fixp` 绑定加 `n_actual`)、`src/target/codegen_ascend.cc`(`GemmFixpOpCodegen` 多发一参)、`src/op/ascend.cc`(`ascend_gemm_v0_fixp` set_num_inputs 7→8) |
| **是否必须** | 是 —— 忠实复刻要求 QK 与 PV 走**同一套 matmul 结构**(参考 `ComputeMm1`/`ComputeMm2`),QK 不能再用 `gemm_v0` 留驻 + 单独拷 |
| **是否兼容** | 是 —— PV(K≤128,kL0split=1,`n_actual=N`)逐字节不变;回归两 example PASS |
| **状态** | 待 NPU 验证(首测因 tiny-tile 屏障 bug 卡死,1c7c124a 修复后重测中) |

---

## 1. 算子为什么需要它

参考里 QK(`ComputeMm1`)和 PV(`ComputeMm2`)是**同一套** matmul 结构:K-累加 → 融合
Fixpipe 到 GM → `unitFlag` → 共享 cL0 ping-pong。我之前 QK 用 `gemm_v0`(结果留驻 L0C +
内核里单独 `T.copy` 搬出),**不是融合 fixpipe**,是绕行。忠实复刻要求 QK 也走 fixp 路径。

## 2. 现象 / 缺口

008 给 `gemm_v0_fixp` 的 cL0 ping-pong 只服务 PV——PV 的 K=窗口≤128,单 K-tile,所以 008
里有 `static_assert(K <= kL0Size=128)`。QK 的 K=headDim=512,需要 4 个 K-tile 累加,用不了。

## 3. 根因

`gemm_v0_fixp` 模板假设单 K-tile(`kL0split==1`),没有跨 K-tile 的累加;`transpose_B`
路径也没有运行期列数 `n_actual`(QK 的输出列 = 窗口宽 < N=BI)。

## 4. 为什么不能在内核侧解决

K 累加(跨 4 个 K-tile 把 D=512 收缩进同一个 cL0 slot,首 tile `init`、末 tile `unitFlag`
flush、然后 fixpipe)在**模板内部的 kL0 循环**里;内核调不进去。属于
*“TileLang 表达不了 → 扩展原语能力”*,与已合入的 `gemm_v0` 的 K-累加 / `n_actual` 同性质。

## 5. 修法

`gemm_v0_fixp`(`common.h`):
- **去掉 `static_assert(K <= kL0Size)`**;`kL0` 循环真正 K-累加:per-tile
  `kSize = min(kL0Size, k_actual - kL0Idx*kL0Size)`,`initflag` 只在首 K-tile、`unitFlag`
  在末 K-tile = 0b11(其余 0b10),4 个 tile 累加进同一 cL0 slot,再做该 N-tile 的 fixpipe。
- **加 `uint32_t n_actual = N`**:`transpose_B` 的输出列数(= QK 窗口宽),用在 mma 的 n 与
  `transpose_B` 的 L0B 载入;非 transpose 的 PV 路径仍用 `nTile`,逐字节不变。
- **关键修复(1c7c124a)**:mma 后的 tiny-tile `PipeBarrier<PIPE_M>` 原写成
  `if constexpr ((M/16)*(nTile/16)<10)`——对 PV(mma 的 n = nTile = 128)恰好对,但对 **QK
  (mma 的 n = `n_actual` = win_align,可以很小)就错了**。参考 `ComputeMm1`(cube.h:580)用的是
  **运行期 `mmadParams.n`**。小窗口时该补的屏障没补 → 硬件 PIPE_M hazard → cube **卡死**。
  改成运行期 `mmaN = transpose_B ? n_actual : nTile`,与参考一致。

绑定/codegen/set_num_inputs 照 `n_actual`/`k_actual` 既有范式透传(尾随默认参,arg [7]=n_actual)。

## 6. 忠实性(对照 `block_cube.h` `ComputeMm1`)

| TileLang | Ascend C 参考 |
|---|---|
| `kL0` 循环 K-累加,`initflag` 首 tile,`unitFlag`=末 tile?0b11:0b10 | `cmatrixInitVal=(kL1==0&&kL0==0)`,`unitFlag=(kL1==末&&kL0==末)?0b11:0b10`(cube.h:575-578) |
| 末 K 后 fixpipe(0b11)到 dst | `if(kL1==末) Fixpipe(unitFlag=0b11)`(cube.h:591-605) |
| tiny-tile barrier 用运行期 `mmaN` | `if((mmadParams.m/16)*(mmadParams.n/16)<10) PipeBarrier<PIPE_M>`(cube.h:580) |
| `n_actual` = transpose_B 列数 | `nL1SizeAlign` = 窗口宽 |

## 7. 兼容性证据

1. **默认值保兼容**:`n_actual` 默认 N;PV K≤128 → `kL0split==1` → `kSize=k_actual`(单 tile,
   与 008 同),`mmaN=nTile`(运行期分支 == 原 constexpr)。PV 逐字节不变。
2. **回归**:`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 均 PASS(证 `gemm_v0` /
   `gemm_v0_fixp` 的扩展不破坏其它算子)。
3. **单 caller 扩展**:SWA QK 是唯一新走 K>128 路径的 caller,与 PV(K≤128)共用模板、互不影响。

## 8. 必要性

QK 忠实复刻(融合 fixpipe + unitFlag + 与 PV 同结构)的前提;也是后续"删 barrier + 共享 cL0
ping-pong"的基础(QK 必须先在 fixp/unitFlag 路径上,才能与 PV 共享 cL0 的硬件排序)。
