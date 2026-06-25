# 008 · Ascend `gemm_v0_fixp` 2-slot L0C ping-pong + 接通 `unitFlag`(= Ascend C `cL0TensorPingPong`,fixpipe∥mma 核内重叠)

| | |
|---|---|
| **编译器仓库** | `yangqiang2018/tilelang-ascend-2` |
| **分支 / 提交** | `wip/gemm-v0-fixp-l0c-pingpong`(**独立基于 `ascendc_pto`**,与 006/007 无关、可单独合入) |
| **改动文件** | `src/tl_templates/ascend/common.h`(`mma` 接通 unitFlag、`copy_l0c_to_gm` 加 unitFlag 透传、`gemm_v0_fixp` 改 2-slot cL0 ping-pong)、`tilelang/language/ascend.py`(`gemm_v0_fixp` docstring) |
| **是否必须** | 是 —— cube 核内 fixpipe 与 mma 串行是 1.28× 的主杠杆之一;忠实复刻 Ascend C 的 `cL0TensorPingPong` 必须有第二个 L0C slot + 硬件 `unitFlag` |
| **是否兼容** | 是 —— `mma`/`copy_l0c_to_gm` 的 `unitFlag` 默认 0(关),所有现有 caller 逐字节不变;`gemm_v0_fixp` 全树**仅 1 个 caller**(算子 kernel.py);不改 catlass |
| **状态** | 待容器从源码重编后验证(SWA 快测 ×5 + 回归 + msprof) |

---

## 1. 算子为什么需要它

当前 ~1.28× Ascend C,差距全在 **cube 核内各 pipe 不重叠**(aicore≈2314us,但最大单 pipe
aic_mte2≈980us,各 pipe 串着跑;Ascend C 各 pipe ~0.99 全并发,aicore=1578us)。其中 PV
(`P @ V`,`gemm_v0_fixp`)的 **fixpipe(把第 i 个 N-tile 搬出 L0C)与 mma(算第 i+1 个 N-tile)
被强制串行**。Ascend C 的 `ComputeMm2` 用 **`cL0TensorPingPong`(两个 L0C slot)+ 硬件 `unitFlag`**
让它们重叠(`block_cube.h:833-940`)。这是忠实复刻里 cube 核内 overlap 的核心结构,之前为求正确
被我简化成单 slot。

## 2. 现象 / 缺口

`gemm_v0_fixp` 现用**单个 L0C slot `C[0]`**,每个 N-tile 之间用
`SetFlag<M_FIX>;WaitFlag<M_FIX>`(mma→fixpipe)+ `SetFlag<FIX_M>;WaitFlag<FIX_M>`(复用 slot)
两道**阻塞**软握手 —— 下一个 mma 必须等本 tile 的 fixpipe 完全排空才能开始,fixpipe 与 mma 零重叠。

## 3. 根因

Ascend C 的重叠**不靠**软 flag,靠两条硬件机制:`Mmad` 的 `unitFlag`(`0b10` 累加不 flush /
`0b11` 末次 flush)+ `Fixpipe` 的 `unitFlag=0b11`,围绕 cL0 **没有任何** `SetFlag/WaitFlag`
(`ComputeMm2:910/932`)。而我们的编译器:

- `mma` 模板里 `mmadParams.unitFlag = unitFlag;` **被注释掉**(`common.h:197`)—— unitFlag 原语
  当前**表达不了**;
- `copy_l0c_to_gm` 里 `tileCopier(dst, src, 0)` **把 unitFlag 硬编码成 0**(`common.h:230`);
- `gemm_v0_fixp` 只有一个 `C[0]` slot,且用阻塞软握手。

## 4. 为什么不能在内核侧解决

第二个 L0C slot 的寻址(`C[c_base]`)、`unitFlag` 的传递,都在 **`mma`/`copy_l0c_to_gm`/
`gemm_v0_fixp` 模板内部**,内核(kernel.py)只能传一个 `acc_o_l0c` 操作数、调不进模板体。属于
*“TileLang 表达不了 → 把 unitFlag 这个已存在但被掐断的原语接通(原语能力补齐)”*。

**关键:不改 catlass。** catlass 早已支持 unitFlag —— `CopyL0CToGmTla::operator()(..., uint8_t
unitFlag = 0)` 透传进 `FixpipeParamsV220.unitFlag`(`copy_l0c_to_gm.hpp:226/239`),`TileMmadTla`
同样透传(`tile_mmad.hpp:93/99`)。所以只需在我们自己的 `common.h` 包装里把这条通道打开,**零
catlass 改动**。

## 5. 修法

- **`mma`**(`common.h:197`):取消注释 `mmadParams.unitFlag = unitFlag;`。签名里 `unitFlag`
  默认 0,所有现有 caller(`gemm_v0`、各 example)传 0 = 关 = 当前行为,逐字节不变。
- **`copy_l0c_to_gm`**(`common.h`):签名加 `uint8_t unitFlag = 0`,把 `tileCopier(dst, src, 0)`
  改成 `tileCopier(dst, src, unitFlag)`。默认 0 → 现有 caller(含内核 QK 的 acc_s 搬出)不变。
- **`gemm_v0_fixp`**(`common.h`):逐指令对齐 `ComputeMm2`:
  - `C` 改为 2-slot `[2, M, nTile]` L0C ping-pong;每个 N-tile 选
    `c_base = (cL0BufIter & 1) * (M * nTile)`(= `cL0TensorPingPong[cL0BufIter%2]`),`cL0BufIter++`;
  - mma 传 `unitFlag = (末 K sub-tile) ? 0b11 : 0b10`(本算子 K≤128 单 tile → 恒 0b11);
  - fixpipe(`copy_l0c_to_gm`)传 `unitFlag = 0b11`;
  - **删掉** N-tile 间的 `M_FIX/FIX_M` 软握手 + 调用开头的 prior-fixpipe drain;
  - mma **前**的无条件 `PipeBarrier<PIPE_M>` 移到 mma **后**并条件化为
    `if constexpr ((M/16)*(nTile/16) < 10)`(对齐 `ComputeMm2:902-914`:`WaitFlag<MTE1_M>` 后直接
    `Mmad`,前置 PipeBarrier 会排空 M 流水、破坏重叠;后置仅小 tile,本算子 4×8=32 编译期消去)。
- **内核**(`kernel.py`):`acc_o_l0c = T.alloc_L0C([2, G, BI], accum_dtype)`(原 `[G, BI]`);PV
  调用站点不变(仍传 `acc_o_l0c` 单操作数,`M/N/K` 由绑定从 `dst`/`A` 推、不从 `C` 推)。

## 6. 忠实性(逐指令对照 `block_cube.h` `ComputeMm2`)

| TileLang | Ascend C 参考 |
|---|---|
| `C[(cL0BufIter&1)*(M*nTile)]`,`cL0BufIter++` | `cL0TensorPingPong[(cL0BufIter%2)*(L0C_PP_SIZE/4)]`,`cL0BufIter++`(`:833-835, 940`) |
| `mma(... unitFlag=(末K)?0b11:0b10)` | `mmadParams.unitFlag = ((k1==末)&&(kL0==末)) ? 0b11 : 0b10`(`:910`) |
| `copy_l0c_to_gm(... 0b11)` | `fixParams.unitFlag = 0b11`(`:932`) |
| `WaitFlag<MTE1_M>` → `mma` 直连,无前置 barrier | `WaitFlag<MTE1_M>` → `Mmad` 直连(`:902-912`) |
| 后置 `if constexpr ((M/16)*(nTile/16)<10) PipeBarrier<PIPE_M>` | `if ((m/16)*(n/16)<10) PipeBarrier<PIPE_M>`(`:913-914`) |
| cL0 周围无软 flag,靠 unitFlag | 同(`:902-936` 无 M_FIX/FIX_M) |

## 7. 兼容性证据

1. **默认值保兼容**:`mma`/`copy_l0c_to_gm` 的 `unitFlag` 默认 0(= 当前 Mmad/Fixpipe 行为)。
   现有 `gemm_v0` 及所有 example caller 传 0,生成代码逐字节不变。
2. **单 caller**:`gemm_v0_fixp` 全树仅 `sparse_attn_sharedkv/kernel.py` 一处调用;`acc_o_l0c`
   改 2-slot 与之 lockstep,无第三方受影响。
3. **数值 bit-identical**:每个 N-tile 仍是独立 `init + accumulate(kSize=k_actual) + fixpipe` 到
   各自 `dst` 列带,无跨 tile 累加;改动只动「用哪个 slot / fixpipe 何时排空」,数学不变。
4. **不改 catlass**:仅打开 catlass 已暴露的 unitFlag 通道。
5. **回归**:`paged_flash_attn_bhsd.py`、`sparse_flash_attn_developer.py` 用 `gemm_v0`(unitFlag=0,
   未触及),必须仍 PASS,证明 `mma`/`copy_l0c_to_gm` 的兼容性。

## 8. L0C 预算

L0C = 128KB。`acc_s_l0c`(QK)= `[G,BI]` = 64·128·4 = 32KB;`acc_o_l0c`(PV)= `[2,G,BI]` =
2·32KB = 64KB;两者内核作用域共驻 = **96KB ≤ 128KB**(余 32KB)。

## 9. 跨调用安全(当前依赖,待 (B) 收口)

`gemm_v0_fixp` 内 `cL0BufIter` 是局部变量(每次调用从 0 起,不像参考的持久成员)。调用**内** 4 个
N-tile 的重叠由 unitFlag 硬件保证(= 参考 tile 循环本身);跨调用的 FIX 排空当前依赖内核
`kernel.py:359` PV 之后的 `T.barrier_all()`。**待后续「depth-3 preload」改动移除该 barrier_all 时**,
需补跨调用 FIX 收口或把 cL0BufIter 持久化。

## 10. 验证

容器重编 `wip/gemm-v0-fixp-l0c-pingpong` → 先 `get_kernel_source()` 看 codegen(确认 fixpipe(i) 与
mma(i+1) 间无阻塞 wait、`copy_l0c_to_gm` 读 `C[c_base]`、unitFlag 落位)→ SWA 快测 ×5(flaky-NaN
史,跑多次)→ 两个回归 → msprof(看 `aic_fixpipe`∥`aic_mac` 重叠、cube aicore < 2314us)。
