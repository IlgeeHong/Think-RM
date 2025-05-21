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

def extract_alphabet_block_simple(text):
    # Find all <answer>...</answer> blocks in the text.
    alphabet_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(alphabet_blocks) != 1:
        return None
    alphabet_content = alphabet_blocks[0].strip()
    matches = re.findall(r"(A|B)", alphabet_content)
    if not matches:
        return None
    # Convert the last extracted number to an integer.
    return str(matches[-1])

def compute_score_batched(data_sources, solution_strs, ground_truths, extra_infos):
    """
    This is a demonstration of how the batched reward function should look like.
    Typically, you want to use batched reward to speed up the process with parallelization
    """
    full_responses = ["<think>\n" + solution_str for solution_str in solution_strs]
    pattern = r"^<think>\s*(.*?)\s*</think>\s*<rationale>\s*(.*?)\s*</rationale>\s*<answer>\s*(.*?)\s*</answer>$"
    # pattern = r"^<think>\s*(.*?)\s*</think>\s*<rationale>\s*(.*?)\s*</rationale>\s*<answer>\s*(.*?)\s*</answer>(?:\s*<\|eot_id\|>)*$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in full_responses]
    format_rewards = [1.0 if match else 0.0 for match in matches]
    accuracy_rewards = []
    for content, sol in zip(solution_strs, ground_truths):
        try:
            gold_label = str(sol)
        except ValueError:
            gold_label = None
        comp_label = extract_alphabet_block_simple(content)
        if gold_label is None or comp_label is None:
            acc_reward = 0.0
        elif gold_label == comp_label:
            acc_reward = 1.0
        else:
            acc_reward = 0.0
        accuracy_rewards.append(acc_reward)
    # print("strict_format_rewards: ", format_rewards)
    # print("accuracy_rewards: ", accuracy_rewards)
    return [
        format_reward + accuracy_reward for format_reward, accuracy_reward in zip(format_rewards, accuracy_rewards)
    ]