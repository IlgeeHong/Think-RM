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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os, sys
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse

import torch
sys.path.append(os.path.dirname(__file__))
from prompt_templates import NaiveReasoningMulticlassPromptTemplate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/hs2-naive-reasoning-multiclass')
    parser.add_argument('--skip_idx_dir', default='VERL/skip-indices/hs2-naive-reasoning-multiclass')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'nvidia/HelpSteer2'
    skip_idx = torch.load(f"{args.skip_idx_dir}/skip_idx.pt").tolist()

    dataset = datasets.load_dataset(data_source, data_dir="preference")

    train_dataset = dataset['train'].filter(lambda x: x["split"]=="train").filter(lambda x: x["preference_strength"]!=0).filter(lambda example, idx: idx not in skip_idx, with_indices=True)
    test_dataset = dataset['train'].filter(lambda x: x["split"]!="train").filter(lambda x: x["preference_strength"]!=0)

    prompt_template = NaiveReasoningMulticlassPromptTemplate()
    system_msg = prompt_template.get_system_msg()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            context = example.pop('prompt')
            response1 = example.pop('response_1')
            response2 = example.pop('response_2')
            user_msg = prompt_template.get_user_msg(
                context=context.rstrip(),
                response1=response1.rstrip(),
                response2=response2.rstrip(),
            )
            solution = example['preference_strength']
            data = {
                "data_source": data_source,
                "prompt": [
                    {'role': 'system', 'content': system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "ability": "rm",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": int(solution)
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)