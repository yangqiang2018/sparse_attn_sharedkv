"""临时探针: dump 编译后的 CFA Ascend C, 查手拼 softmax 的 -inf mask(fill)为何没落到
[tw_a:512](lse 偏大 = reduce_max 读到 padding 残留)。跑: python dump_cfa_cg.py"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from api import _KERNEL_CACHE, sparse_attn_sharedkv  # noqa: E402
from metadata import sparse_attn_sharedkv_metadata  # noqa: E402
from test_sparse_attn_sharedkv import _build_case, _call_metadata_then_sharedkv  # noqa: E402
from test_sparse_attn_sharedkv_fast import FAST_SCENARIOS  # noqa: E402

cfg = FAST_SCENARIOS["cfa_prefill_fast"]
case = _build_case(cfg, torch.bfloat16)
# build + run(数值错没关系,只要 kernel 编出来进 cache)
_call_metadata_then_sharedkv(
    case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
)
assert _KERNEL_CACHE, "no kernel built"
func = list(_KERNEL_CACHE.values())[-1]
src = func.get_kernel_source()
with open("/tmp/cfa_cg.cpp", "w") as f:
    f.write(src)
lines = src.splitlines()
print(f"wrote /tmp/cfa_cg.cpp; total lines: {len(lines)}")

# 1) 所有 fill(Duplicate)行 —— mask 应该是对 512 宽的一条
print("\n=== Duplicate / Fill 行 ===")
for i, ln in enumerate(lines):
    if re.search(r"Duplicate|Fill\b", ln):
        print(f"{i:5d}: {ln.strip()[:130]}")

# 2) 第一处 ReduceMax 前 45 行(手拼 softmax 的 mask+load+mul 区)
first_rm = next(
    (i for i, l in enumerate(lines) if re.search(r"ReduceMax|WholeReduceMax", l)),
    None,
)
print(f"\n=== 第一处 ReduceMax @ {first_rm}, 前 45 行(mask/load/mul 区)===")
if first_rm is not None:
    for i in range(max(0, first_rm - 45), first_rm + 3):
        print(f"{i:5d}: {lines[i].rstrip()[:130]}")
