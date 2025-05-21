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

import os
import datasets
from verl.utils.hdfs_io import copy, makedirs
import argparse
from datasets import concatenate_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', type=str, default='~/data/mixed')
    parser.add_argument('--frac', type=float, default=0.01)
    parser.add_argument('--hdfs_dir', type=str, required=False, default=None)

    args = parser.parse_args()

    data_source = 'allenai/RLVR-GSM-MATH-IF-Mixed-Constraints'
    dataset = datasets.load_dataset(data_source)
    
    # Subset each dataset
    math_dataset = dataset['train'].shuffle(seed=42).filter(lambda x: x["dataset"] == "math").select(range(int(args.frac * len(dataset['train']))))
    gsm8k_dataset = dataset['train'].shuffle(seed=42).filter(lambda x: x["dataset"] == "gsm8k").select(range(int(args.frac * len(dataset['train']))))

    # Concatenate them into one
    train_dataset = concatenate_datasets([math_dataset, gsm8k_dataset]).shuffle(seed=42)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            data = {
                "data_source": data_source,
                "prompt": [{'content': "Please reason step by step, and put your final answer within \\boxed{}.", 'role': 'system'}] + example["messages"],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": example["ground_truth"]  # should not be used
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    # test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    # test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)