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

from verl.utils.reward_score.prompt_templates import NoThinkMulticlassPromptTemplate
import requests
import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

def chat_vllm_server(server_url, prompt_template, context, response1, response2, model_name, temperature, max_tokens, n):
    """
    Send to whatever server_url you pass.
    """
    system_msg = prompt_template.get_system_msg()
    user_msg   = prompt_template.get_user_msg(
        context=context.rstrip(),
        response1=response1.rstrip(),
        response2=response2.rstrip()
    )
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": n,
    }
    resp = requests.post(server_url, json=payload)
    resp.raise_for_status()

    generations = []
    try:
        for choice in resp.json()["choices"]:
            generations.append(choice["message"]["content"])
    except Exception as e:
        print("Failed to decode response:", e)
        generations.append("<answer>0</answer>")
    return generations 


def extract_multiclass_strict(text):
    score_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(score_blocks) < 1:
        return None
    score_content = score_blocks[-1].strip()
    matches = re.findall(r"(-3|-2|-1|1|2|3)", score_content)
    if not matches:
        return None
    return int(matches[-1])

def _compute_chunk(server_url, prompt_template, prompts, response_pairs, model_name, temperature=0.6, max_tokens=2048, n=1):
    """
    Send one chunk of (prompts, response_pairs) to a single server.
    Returns a list of int scores, same length as prompts.
    """
    sub_scores = []
    for prompt, (r1, r2) in zip(prompts, response_pairs):
        texts = chat_vllm_server(server_url, prompt_template, prompt, r1, r2, model_name, temperature, max_tokens, n)
        raw_scores = [
            s if (s := extract_multiclass_strict(t)) is not None else 0
            for t in texts
        ]
        print(raw_scores)
        sub_scores.append(sum(raw_scores)/len(raw_scores))
    return sub_scores

def compute_pref_batched(prompts, data_sources, response_pairs, extra_infos, model_name, vllm_server_ip="localhost", start_port=8000, n_port=8, temperature=0.6, max_tokens=2048, n=1):
    """
    prompts         : List[str], length = M
    response_pairs  : List[(str,str)], length = M
    server_urls     : List[str] of length n_servers

    Returns: List[int] of length M, in the same order as prompts.
    """
    M = len(prompts)
    n_servers = n_port
    server_urls = [
        f"http://{vllm_server_ip}:{start_port + i}/v1/chat/completions"
        for i in range(n_port)
    ]
    chunk_size = math.ceil(M / n_servers)
    template = NoThinkMulticlassPromptTemplate()
    

    # Prepare chunks
    chunks = []
    for i, url in enumerate(server_urls):
        start = i * chunk_size
        end   = min(start + chunk_size, M)
        if start >= end:
            break
        chunks.append((url, start, end,
                       prompts[start:end],
                       response_pairs[start:end]))

    # Fire off each chunk in its own thread
    results = [None] * M
    with ThreadPoolExecutor(max_workers=n_servers) as exe:
        future_to_range = {
            exe.submit(_compute_chunk, url, template, p_chunk, rp_chunk, model_name, temperature, max_tokens, n): (start, end)
            for url, start, end, p_chunk, rp_chunk in chunks
        }
        for fut in as_completed(future_to_range):
            start, end = future_to_range[fut]
            sub_scores = fut.result()
            results[start:end] = sub_scores

    return results