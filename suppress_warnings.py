#!/usr/bin/env python3
"""
抑制常见的transformers警告
"""

import warnings
import logging
from transformers import logging as transformers_logging

def suppress_warnings():
    """抑制transformers相关警告"""

    # 抑制特定警告
    warnings.filterwarnings("ignore",
        message=".*padding tokens aren't masked.*",
        category=UserWarning
    )

    warnings.filterwarnings("ignore",
        message=".*Setting `pad_token_id` to `eos_token_id`.*",
        category=UserWarning
    )

    # 设置transformers日志级别
    transformers_logging.set_verbosity_error()

    # 或者使用logging模块
    logging.getLogger("transformers").setLevel(logging.ERROR)

    print("✅ 已抑制transformers警告")

if __name__ == "__main__":
    print("配置警告抑制...")
    suppress_warnings()
    print("\n说明：")
    print("1. attention_mask警告已被抑制")
    print("2. pad_token_id警告已被抑制")
    print("这些警告不会影响功能，只是提醒信息")