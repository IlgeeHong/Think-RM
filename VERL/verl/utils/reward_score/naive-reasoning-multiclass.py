# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

def extract_multiclass_strict(text):
    score_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(score_blocks) < 1:
        return None
    score_content = score_blocks[-1].strip()
    matches = re.findall(r"(-3|-2|-1|1|2|3)", score_content)
    if not matches:
        return None
    return int(matches[-1])


def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos):
    accuracy_rewards = []
    for content, sol in zip(solution_strs, ground_truths):
        try:
            gold_label = int(sol)
        except ValueError:
            gold_label = None
        comp_label = extract_multiclass_strict(content)
        if gold_label is None or comp_label is None:
            acc_reward = 0.0
        elif gold_label == comp_label:
            acc_reward = 1.0
        else:
            if abs(comp_label) > 3:
                acc_reward = 0.0
            else:
                # Loose accuracy criterion: award 0.5 if both scores have the same sign.
                if (gold_label > 0 and comp_label > 0) or (gold_label < 0 and comp_label < 0):
                    acc_reward = 0.5
                else:
                    acc_reward = 0.0
        accuracy_rewards.append(acc_reward)
    return accuracy_rewards