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
| 004 | [Ascend 新增 `gemm_v0_fixp` 原语（按 N-tile 即时 fixpipe，根治 PV 的 L0C 越界）](004-ascend-gemm-v0-fixp-l0c.md) | `wip/ascend-gemm-v0-fixp` `03cc14b6` | ⏳ 待 NPU 验证后合入 `ascendc_pto` |
| 005 | [Ascend 新增 `row_expand_sub`/`row_expand_div` 原语（行广播 Sub/Div，消掉非忠实的 [M,N] 广播缓冲）](005-ascend-row-expand-sub-div.md) | `wip/ascend-gemm-v0-fixp` `410cafbd` | ⏳ 待 NPU 验证后合入 `ascendc_pto` |

> 各修改互不依赖、各自一个 commit、各自基于 `ascendc_pto`，**逐个独立合并**，每次合并都是一个自洽的修复。004 是 002（N 切分切 L0B）之上的忠实收尾（切 L0C + 即时搬出），但仍是独立的兼容性新增原语。
