import torch

from sparse_flash_mla_api import sparse_flash_mla
from sparse_flash_mla_golden import build_case, run_case, check_lse, check_result

_COMMON = dict(
    layout_q="TND",
    B=1,
    S1=8192,
    T1=8192,
    N1=64,
    N2=1,
    D=512,
    block_num1=65,
    block_size1=128,
    cu_seqlens_q=[0, 8192],
    seqused_kv=[8192],
    softmax_scale=0.04419417,
    ori_win_left=127,
    ori_win_right=0,
    ori_mask_mode=4,
    cmp_mask_mode=3,
)

FULL_SCENARIOS = {
    "swa": dict(_COMMON, scenario=1, K=0, block_num2=1, block_size2=1, cmp_ratio=4),
    "hca": dict(_COMMON, scenario=2, K=512, block_num2=17, block_size2=128, cmp_ratio=128),
    "csa": dict(_COMMON, scenario=3, K=512, block_num2=17, block_size2=128, cmp_ratio=4),
}


def main():
    dtype = torch.bfloat16
    for name, cfg in FULL_SCENARIOS.items():
        case = build_case(cfg, dtype)
        out, lse = run_case(case, cfg, sparse_flash_mla)
        check_lse(lse.cpu(), case["cpu_ref_lse"], dtype)
        check_result(out.cpu(), case["cpu_ref"])
        print(f"[{name}] 8K passed")
    print("Kernel Output Match!")


if __name__ == "__main__":
    main()
