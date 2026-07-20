"""临时探针: dump 编译后的 CFA Ascend C, 查前端拼 paged load 的计数器(pa_done/pa_cur)
在 cfa 的**运行期** `if is_ori` 分支里是怎么 lowering 的 —— swa 的 is_ori 是编译期 True
(前端拼 PASS), cfa 是运行期(FAIL), 嫌疑是 alloc_var 计数器在 runtime-if 内没正确
声明/初始化/累加。跑: python dump_cfa_cg.py"""

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

# 1) 计数器 pa_done / pa_cur 的每一处: 声明? 初始化? 累加? 还是裸用未初始化?
print("\n=== pa_done / pa_cur 的所有出现(声明/init/累加/使用)===")
for i, ln in enumerate(lines):
    if re.search(r"\bpa_done\b|\bpa_cur\b", ln):
        print(f"{i:5d}: {ln.strip()[:150]}")

# 2) 前端拼发出的 GM->L1 paged copy
print("\n=== copy_gm_to_l1 行(前端拼的 paged load)===")
for i, ln in enumerate(lines):
    if "copy_gm_to_l1" in ln:
        print(f"{i:5d}: {ln.strip()[:170]}")

# 3) 第一处 copy_gm_to_l1 的上下文: 看它外面的 runtime if / for 结构和计数器位置
first_c = next((i for i, ln in enumerate(lines) if "copy_gm_to_l1" in ln), None)
print(f"\n=== 第一处 copy_gm_to_l1 @ {first_c}, 前 40 行上下文(runtime if/for 结构)===")
if first_c is not None:
    for i in range(max(0, first_c - 40), min(len(lines), first_c + 4)):
        print(f"{i:5d}: {lines[i].rstrip()[:150]}")
