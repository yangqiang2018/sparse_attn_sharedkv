"""pytest 配置:补回 test_sparse_attn_sharedkv.py 文档(line 24-28/224)依赖、但此前
未提交进仓的 ``--runslow`` 选项与 ``slow`` 标记。

prefill 用例(S1=8192)的 CPU golden 要跑几分钟,被标 ``slow`` 默认跳过;加 ``--runslow``
才跑。decode(S1=1)等 SMALL_CASES 不带标记,默认就跑。``test_*_fast.py`` 是独立的小 case
文件,不受影响。
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run slow cases (large S1=8192; CPU golden takes minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large case (S1=8192); needs --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow; pass --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
