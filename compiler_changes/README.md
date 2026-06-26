# 编译器修改记录

本目录记录为了**忠实复刻 `sparse_attn_sharedkv`**（TileLang 逐指令复刻 Ascend C
实现）而对**编译器**（`tilelang-ascend`,仓库 `yangqiang2018/tilelang-ascend-2`,
集成分支 `ascendc_pto`）所做、并已合入集成分支的每一处修改。

为什么要有这个记录：本项目的规则是 *“TileLang 表达不了→加原语;编译器有 bug→修
编译器”* —— 当忠实的逐指令复刻撞到 TileLang/其编译器表达不了的东西、或撞到真正
的编译器 bug 时,我们去修编译器,而不是在内核里发明绕路写法。并且**每一处对编译器
的修改都必须是兼容性修改**（绝不能因此让其它算子出问题）。每一处合入的修改在此对应
一篇文档,完整记录来龙去脉:现象、根因、为什么不能在内核侧解决、修法、以及证明它兼容
的证据。

文档用**中文**书写,一处修改一篇,按合入顺序编号:

集成分支为 `ascendc_pto`(本仓库无 `main`)。001/002/003 已各自作为一次**独立 merge**
合入 `ascendc_pto`(`Merge 001` `11d28d8d` → `Merge 002` `2f146c1c` → `Merge 003`
`e4fc9e1c`),对应 wip 分支已删除。验证:`git log --graph --oneline --merges ascendc_pto`。

| 编号 | 标题 | 合并提交 | 状态 |
|---|---|---|---|
| 001 | [Ascend codegen:整数 max/min 输出为三元表达式](001-ascend-codegen-integer-minmax-ternary.md) | `Merge 001` `11d28d8d`(原 `0e53a8ad`) | ✅ 已合入 `ascendc_pto` |
| 002 | [Ascend `gemm_v0`:增加 N 方向切分（对齐 Ascend C）](002-ascend-gemm-v0-n-tiling.md) | `Merge 002` `2f146c1c`(原 `26116e27`) | ✅ 已合入 `ascendc_pto` |
| 003 | [Ascend 新增 `copy_pa` 原语（分页 KV 直读进 L1）](003-ascend-copy-pa-paged-kv-load.md) | `Merge 003` `e4fc9e1c` | ✅ 已合入 `ascendc_pto`(SWA 正确 + 11050→5330us + 回归过) |
| 004 | [Ascend 新增 `gemm_v0_fixp` 原语（按 N-tile 即时 fixpipe + 运行期 k_actual 变长 K，根治 PV 的 L0C 越界与 0×NaN）](004-ascend-gemm-v0-fixp-l0c.md) | `Merge 004` `44cd1d9e` | ✅ 已合入 `ascendc_pto`(SWA 快测 PASS + 回归过) |
| 005 | [Ascend 新增 `row_expand_sub`/`row_expand_div` 原语（行广播 Sub/Div，消掉非忠实的 [M,N] 广播缓冲）](005-ascend-row-expand-sub-div.md) | `Merge 005` `cc98641d` | ✅ 已合入 `ascendc_pto` |
| 006 | [Ascend `gemm_v0`/`mma` 增加运行期 `n_actual`（变长 N 输出列，= Ascend C `ComputeMm1` 窗口长 N）](006-ascend-gemm-v0-n-actual.md) | `Merge 006` `2a7662a1`(原 `b255a071`) | ✅ 已合入 `ascendc_pto`(NPU 5/5 PASS + 回归过) |
| 007 | [Ascend 新增 `softmax_flash_v2` 原语（逐指令复刻 AscendC `SoftmaxFlashV2`，变长 N softmax 不掩码）](007-ascend-softmax-flash-v2.md) | `wip/ascend-softmax-flashv2` `cf38e5fa`(快进合入) | ✅ 已合入 `ascendc_pto`(NPU 5/5 PASS + 回归过) |
| 008 | [Ascend `gemm_v0_fixp` 2-slot L0C ping-pong + 接通 `unitFlag`（= Ascend C `cL0TensorPingPong`，fixpipe∥mma 核内重叠）](008-ascend-gemm-v0-fixp-l0c-pingpong.md) | `Merge 008` `dddf3413`(原 `ca15c716`) | ✅ 已合入 `ascendc_pto`(NPU 5/5 PASS + 回归 + 1.03× parity) |
| 009 | [Ascend `gemm_v0_fixp` K-累加 + `n_actual` + `cl0_base` + `mma` 补 `cmatrixSource` + fixpipe `nSize`（QK 走统一 fixp 路径 = `ComputeMm1`，与 PV 共享 cL0）](009-ascend-gemm-v0-fixp-kaccum.md) | `Merge 009` `9e7300f7`（原 `d61b5127`） | ✅ 已合入 `ascendc_pto`(SWA 5/5 PASS + 回归 + prefill 持平/decode 1.65×) |
| 010 | [Ascend `gemm_v0_fixp` 加 `prime_drain`：L0AB `M_MTE1` flag 一次 prime（= `AllocEventID`/`FreeEventID`）+ 专属 event id（`L0AB_MM_EVENT`，根治常驻 flag 冲突）](010-ascend-gemm-v0-fixp-prime-once.md) | `wip/gemm-v0-fixp-prime-once` `9c024295` | ⏳ 待 NPU 验证(SWA 5/5 + 回归)后合入 `ascendc_pto` |

> 001–008 均已合入 `ascendc_pto`(`dddf3413`)。006/007 做「全链路变长 N」忠实复刻;008 给 PV
> `gemm_v0_fixp` 加 `cL0TensorPingPong`(2-slot L0C + 硬件 `unitFlag`,fixpipe∥mma 核内重叠,= `ComputeMm2`,
> NPU 验证 1.03× parity)。
>
> **009 = 整条忠实 cube 流水的 matmul 基础**(退回 parity 8061a9e 重建):给 `gemm_v0_fixp` 加 K-累加 +
> `n_actual` + `cl0_base`,让 QK 也走同一 fused-fixpipe / `unitFlag` / 共享 cL0 路径(= `ComputeMm1`);
> 并内置 `dbg_barrier` 诊断开关坐实「多-K `unitFlag`(0b10)在单个 QK 调用内卡死」的病根。内核侧 3-slot
> KV 环 + 子块化 + QP 环 + 共享 cL0 + 零 barrier 在此原语之上做。（注:这是回收编号后的新 009;同名旧版
> 无 `dbg_barrier`、卡死无法定位已删。）

> 各修改互不依赖、各自一个 commit、各自基于 `ascendc_pto`，**逐个独立合并**，每次合并都是一个自洽的修复。004 是 002（N 切分切 L0B）之上的忠实收尾（切 L0C + 即时搬出），但仍是独立的兼容性新增原语。
