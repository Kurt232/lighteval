# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

from lighteval.metrics.metrics import SimpleEvalMatch


"""
Simple Eval
"""


def compare_strings(
    gold: str,
    pred: str,
):
    return SimpleEvalMatch().compute(
        golds=[gold],
        predictions=[pred],
    )


# Test basic multiple choice answer extraction
@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("C", "Answer: C", 1),
        ("C", "'Answer: C'", 1),
        # Test answer with reasoning
        (
            "B",
            "Let's think step by step. It's not A because it doesn't make sense, therefore I think it's B. Answer: B",
            1,
        ),
        ("D", "The Answer: D, doesn't makese nsense for answer to be A or B", 1),
    ],
)
def test_extraction_abc(gold, pred, expected):
    assert compare_strings(gold, pred) == expected
