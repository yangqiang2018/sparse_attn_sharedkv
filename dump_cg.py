"""Dump one scenario's compiled Ascend C and grep the front-end paged KV load
(the copy_pa replacement): where the page counter is declared / initialised /
accumulated, what copy_gm_to_l1 calls come out, and the loop + flag structure
around them.

The front-end reconstruction is the same source in every builder, but the
enclosing structure differs, so dumping per scenario is what separates
"the runtime-slice GM->L1 copy itself is wrong" from "the enclosing ring /
runtime-if is wrong":

    swa   slot = h (0/1, no ring), whole window in one load, no is_ori branch
    cfa   slot = (kv_iter+h)%3 (runtime), cb band loop, runtime `is_ori`
    scfa  as cfa, plus the topk/gather V0 path

Run:  python dump_cg.py [swa|cfa|scfa]
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from api import _KERNEL_CACHE, sparse_attn_sharedkv  # noqa: E402
from metadata import sparse_attn_sharedkv_metadata  # noqa: E402
from test_sparse_attn_sharedkv import _build_case, _call_metadata_then_sharedkv  # noqa: E402
from test_sparse_attn_sharedkv_fast import FAST_SCENARIOS  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "cfa"
name = f"{which}_prefill_fast"
assert name in FAST_SCENARIOS, f"{name} not in {list(FAST_SCENARIOS)}"

cfg = FAST_SCENARIOS[name]
case = _build_case(cfg, torch.bfloat16)
# build + run (a wrong result is fine here -- we only need the kernel compiled
# and in the cache).
_call_metadata_then_sharedkv(
    case, cfg, sparse_attn_sharedkv, sparse_attn_sharedkv_metadata
)
assert _KERNEL_CACHE, "no kernel built"
func = list(_KERNEL_CACHE.values())[-1]
src = func.get_kernel_source()
out = f"/tmp/{which}_cg.cpp"
with open(out, "w") as f:
    f.write(src)
lines = src.splitlines()
print(f"wrote {out}; total lines: {len(lines)}")

# 1) the page counter: declared once at kernel scope, or re-initialised per load?
print("\n=== pa_cur / pa_done (declaration / init / accumulate / use) ===")
for i, ln in enumerate(lines):
    if re.search(r"\bpa_done\b|\bpa_cur\b", ln):
        print(f"{i:5d}: {ln.rstrip()[:150]}")

# 2) what the runtime row-slice actually emits (template args + realTail*)
print("\n=== copy_gm_to_l1 (front-end paged load) ===")
for i, ln in enumerate(lines):
    if "copy_gm_to_l1" in ln:
        print(f"{i:5d}: {ln.strip()[:190]}")

# 3) copy_pa still in the kernel (cmp path / PV) -- the reference to compare against
print("\n=== copy_pa (still-unconverted call sites) ===")
for i, ln in enumerate(lines):
    if "copy_pa" in ln:
        print(f"{i:5d}: {ln.strip()[:190]}")

# 4) full nesting + flags around the first paged load: keep original indentation,
#    so the loop / runtime-if depth the counter sits at is readable at a glance.
first = next((i for i, ln in enumerate(lines) if "copy_gm_to_l1" in ln), None)
print(f"\n=== first copy_gm_to_l1 @ {first}: 60 before / 10 after (raw indent) ===")
if first is not None:
    for i in range(max(0, first - 60), min(len(lines), first + 10)):
        mark = ">>" if i == first else "  "
        print(f"{mark}{i:5d}: {lines[i].rstrip()[:150]}")
