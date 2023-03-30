# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

from fairseq.data.encoders import register_tokenizer


@register_tokenizer('cjk')
class CJKTokenizer(object):

    def __init__(self, source_lang=None, target_lang=None):
        self.CHAR_SPACE_PATTERN1 = r"([\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF]\s+)"
        self.CHAR_SPACE_PATTERN2 = r"(\s+[\u2E80-\u9FFF\uA000-\uA4FF\uAC00-\uD7FF\uF900-\uFAFF])"

    def encode(self, x: str) -> str:
        return x

    def decode(self, x: str) -> str:
        def _strip(matched):
            return matched.group(1).strip()
        x = re.sub(self.CHAR_SPACE_PATTERN1, _strip, x)
        x = re.sub(self.CHAR_SPACE_PATTERN2, _strip, x)
        return x.strip()
