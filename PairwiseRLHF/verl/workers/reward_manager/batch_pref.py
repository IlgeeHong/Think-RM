# Copyright 2025 Individual Contributor: Mert Unsal
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

import torch
from verl import DataProto
from collections import defaultdict
#######################################################
from itertools import combinations
#######################################################
# class BatchRewardManager:
class BatchPrefRewardManager:
    def __init__(self, tokenizer, num_examine, compute_score, reward_fn_key='data_source', **reward_kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    # def verify(self, data):
    #     prompt_ids = data.batch['prompts']
    #     response_ids = data.batch['responses']
    #     attention_mask = data.batch['attention_mask']

    #     prompt_len = prompt_ids.shape[-1]
    #     valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

    #     responses_str = []
    #     for i in range(len(data)):
    #         valid_len = valid_response_lengths[i]
    #         valid_response_ids = response_ids[i][:valid_len]
    #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    #         responses_str.append(response_str)

    #     ground_truths = [item.non_tensor_batch['reward_model'].get('ground_truth', None) for item in data]
    #     data_sources = data.non_tensor_batch[self.reward_fn_key]
    #     extras = data.non_tensor_batch.get('extra_info', [None] * len(data))

    #     scores = self.compute_score(data_sources=data_sources,
    #                                 solution_strs=responses_str,
    #                                 ground_truths=ground_truths,
    #                                 extra_infos=extras,
    #                                 **self.reward_kwargs)

    #     return scores

    def verify_pairwise(self, data):
        """
        Returns
        -------
        i_idx, j_id: torch.Tensor   # (N_pairs, ) global indices (i, j)
        d_ij: torch.Tensor   # (N_pairs,) reward differences r_i - r_j
        """

        # 1) Decode every prompt and response
        prompt_ids = data.batch['prompts']      # (B, prompt_len)
        response_ids = data.batch['responses']    # (B, max_seq_len)
        attention_mask = data.batch['attention_mask']

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1) # without padding

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)
        
        # extract prompt strings
        # hardcoded prompt slice to remove "\nuser\n\n" -> "\n" and "assistant\n"
        # prompt_strs = [
        #     self.tokenizer.decode(p_ids, skip_special_tokens=True).strip()[6:-9]
        #     for p_ids in prompt_ids
        # ]
        prompt_strs = [
            self.tokenizer.decode(p_ids, skip_special_tokens=True).strip()
            for p_ids in prompt_ids
        ]
        
        # 2) Group indices that share the same UID (same prompt)
        uids = data.non_tensor_batch['uid']
        groups = defaultdict(list)
        for idx, uid in enumerate(uids):
            groups[uid].append(idx)
        # uid -> global index

        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get('extra_info', [None] * len(data))

        # 3) Enumerate every unordered pairs (i, j) within each group
        # we only compute the upper triangle (i < j) and take minus for the lower triangle
        pair_i, pair_j = [], []
        for idx_list in groups.values():
            for i_idx, j_idx in combinations(idx_list, 2):
                pair_i.append(i_idx)
                pair_j.append(j_idx)
        pairs = list(zip(pair_i, pair_j))
        # at this point we have the unordered pairs only (pair_i < pair_j) where i, j share the same prompt

        # 4) Build batched arguments — now including the shared prompt
        prompt_extended = [prompt_strs[i] for i, _ in pairs]
        resp_pairs = [(responses_str[i], responses_str[j]) for i, j in pairs]
        data_sources_extended = [data_sources[i] for i, _ in pairs]
        extras_extended = [extras[i] for i, _ in pairs]

        # 5) Call compute_score — sharded to multiple LLMs
        # note that diff_vals = r_j - r_i
        # shape: (nC2 * G, ) where G = B / n -> ((B * (n-1)) / 2, )
        diff_vals = self.compute_score(prompts = prompt_extended,
                                       data_sources = data_sources_extended,
                                       response_pairs = resp_pairs,
                                       extra_infos = extras_extended,
                                       **self.reward_kwargs)
        
        # 6) Return tensors
        d_ij = (-1) * torch.as_tensor(diff_vals, dtype=torch.float32, device=prompt_ids.device)
        i_idx = torch.tensor(pair_i, dtype=torch.long, device=prompt_ids.device)
        j_idx = torch.tensor(pair_j, dtype=torch.long, device=prompt_ids.device)

        return i_idx, j_idx, d_ij


    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     if return_dict:
        #         return {"reward_tensor": data.batch['rm_scores']}
        #     else:
        #         return data.batch['rm_scores']
        # reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # reward_diff_tensor = torch.zeros((data.batch['responses'].shape[0], data.batch['responses'].shape[0]), dtype=torch.float32)
        prompt_ids = data.batch['prompts']
        device = prompt_ids.device
        reward_diff_tensor = torch.full((data.batch['responses'].shape[0], data.batch['responses'].shape[0]), float('nan'), dtype=torch.float32, device=device)
        reward_extra_info = defaultdict(list)
        # prompt_ids = data.batch['prompts']
        # prompt_len = prompt_ids.shape[-1]
        # attention_mask = data.batch['attention_mask']
        # valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        # data_sources = data.non_tensor_batch[self.reward_fn_key]

        # groups, pair_i, pair_j, d_ij = self.verify_pairwise(data)
        i_idx, j_idx, d_ij = self.verify_pairwise(data)

        # fill the upper triangle
        reward_diff_tensor[i_idx, j_idx] = d_ij

        # # 1) Check strictly upper‐triangular (zeros on and below diagonal)
        # is_upper = torch.all(reward_diff_tensor == torch.triu(reward_diff_tensor, diagonal=1))
        # print(f"Is strictly upper‐triangular? {is_upper}")

        # fill the lower triangle
        reward_diff_tensor[j_idx, i_idx] = -d_ij

        # fill diagonal with 0
        reward_diff_tensor.fill_diagonal_(0.0)
        print(torch.nansum(reward_diff_tensor, dim=1))
        # import sys
        # sys.exit("bye")

        if return_dict:
            return {"reward_diff_tensor": reward_diff_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_diff_tensor
        
        # return groups, pair_i, pair_j, d_ij

        # scores = self.verify(data)
        # rewards = []
        # already_printed = {}

        # for i in range(len(data)):
        #     length = valid_response_lengths[i].item()
        #     score = scores[i]

        #     if isinstance(score, dict):
        #         reward = score["score"]
        #         for key, value in score.items():
        #             reward_extra_info[key].append(value)
        #     else:
        #         reward = score

        #     rewards.append(reward)
        #     reward_tensor[i, length - 1] = reward

        #     data_source = data_sources[i]
        #     if already_printed.get(data_source, 0) < self.num_examine:
        #         response_str = self.tokenizer.decode(data.batch['responses'][i][:length], skip_special_tokens=True)
        #         prompt_str = self.tokenizer.decode(data.batch['prompts'][i], skip_special_tokens=True)
        #         ground_truth = data[i].non_tensor_batch['reward_model'].get('ground_truth', None)
        #         print("[prompt]", prompt_str)
        #         print("[response]", response_str)
        #         print("[ground_truth]", ground_truth)
        #         print("[score]", scores[i])
        #         already_printed[data_source] = already_printed.get(data_source, 0) + 1

        # data.batch['acc'] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        # if return_dict:
        #     return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        # else:
        #     return reward_tensor
