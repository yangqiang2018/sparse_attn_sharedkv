# Compiler change log

This directory records every **compiler** modification (to `tilelang-ascend`,
repo `yangqiang2018/tilelang-ascend-2`, branch `ascendc_pto`) that was made
**for the faithful TileLang port of `sparse_attn_sharedkv`** and merged into the
compiler's integration branch.

Why this log exists: the rule for this project is *"TileLang 表达不了→加原语；
编译器有 bug→修编译器"* — i.e. when a faithful instruction-by-instruction port
hits something TileLang/its compiler cannot express or a genuine compiler bug,
we fix the compiler rather than inventing a kernel-side workaround. Every such
compiler change **must be compatibility-preserving** (it must not break any
other operator). Each merged change gets one document here capturing the full
story: the symptom, the root cause, why it could not be solved in the kernel,
the fix, and the evidence that it is compatible.

One file per change, numbered in merge order:

| # | Title | Compiler commit / branch | Status |
|---|---|---|---|
| 001 | [Ascend codegen: emit integer max/min as a ternary](001-ascend-codegen-integer-minmax-ternary.md) | `wip/ascend-codegen-int-minmax-ternary` (`0e53a8ad`) | verified compatible; pending merge to `ascendc_pto` |
