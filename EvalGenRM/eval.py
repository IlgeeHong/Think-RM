import time, json, os, sys
from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
import re
import warnings
import random
from collections import defaultdict
from transformers import AutoTokenizer
from utils import extract_simple_binary_strict, extract_multiclass_strict, extract_binary_strict, extract_scoring_strict
from utils import format_accuracy_table
from prompt_templates import NoThinkBinaryPromptTemplate, NoThinkMulticlassPromptTemplate, SimpleBinaryPromptTemplate, SimpleMulticlassPromptTemplate, MTBenchPromptTemplate, JudgeLRMPromptTemplate, ReasoningBinaryPromptTemplate, ReasoningMulticlassPromptTemplate, NaiveReasoningBinaryPromptTemplate, NaiveReasoningMulticlassPromptTemplate, StepbyStepReasoningBinaryPromptTemplate
from jinja2 import Template
from rewardbench_utils import SUBSET_MAPPING, EXAMPLE_COUNTS, calculate_scores_per_section
# TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'Assistant' %}{% else %}{% set role = 'User' %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}"
TEMPLATE = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ message['content'] | trim + '\n'}}{% else %}<extra_id_1>{{ 'Assistant' + '\n' if message['role'] == 'assistant' else 'User' + '\n' }}{{ message['content'] | trim + '\n'}}{% endif %}{% endfor %}"

MULTI_TURN_TEMPLATE = Template(TEMPLATE)

warnings.filterwarnings("ignore")
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=16384)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="THU-KEG/RM-Bench")
    parser.add_argument('--template', type=str, default="reasoning-binary")
    parser.add_argument('--save_dir', type=str, default="/path/to/eval/dir/data/template")
    parser.add_argument("--custom_chat_template", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=1e8)
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_name = os.path.basename(args.model_path)
    data_name = os.path.basename(args.dataset)
    save_dir = os.path.join(args.save_dir, data_name)
    OUT_NAME = f'{model_name}_{args.seed}.json'
    print("Evaluation Model: ", model_name)

    ### Choose prompt template
    if args.template == "simple-binary":
        prompt_template = SimpleBinaryPromptTemplate()
    elif args.template == "simple-multiclass":
        prompt_template = SimpleMulticlassPromptTemplate()
    elif args.template == "nothink-binary":
        prompt_template = NoThinkBinaryPromptTemplate()
    elif args.template == "nothink-multiclass":
        prompt_template = NoThinkMulticlassPromptTemplate()
    elif args.template == "mtbench-binary":
        prompt_template = MTBenchPromptTemplate()
    elif args.template == "judgelrm-scoring":
        prompt_template = JudgeLRMPromptTemplate()
    elif args.template == "reasoning-multiclass":
        prompt_template = ReasoningMulticlassPromptTemplate()
    elif args.template == "reasoning-binary":
        prompt_template = ReasoningBinaryPromptTemplate()
    elif args.template == "naive-reasoning-multiclass":
        prompt_template = NaiveReasoningMulticlassPromptTemplate()
    elif args.template == "naive-reasoning-binary":
        prompt_template = NaiveReasoningBinaryPromptTemplate()
    elif args.template == "stepbystep-reasoning-binary":
        prompt_template = StepbyStepReasoningBinaryPromptTemplate()
    else:
        raise ValueError("The template specified is not supported")
    print("Prompt Template: ", args.template)
    
    ### Load tokenizer
    tokenizer = load_hf_tokenizer(model_name_or_path=args.model_path)
    if args.custom_chat_template:
        tokenizer.chat_template = args.custom_chat_template
    
    ### Load model
    model = LLM(model=args.model_path, tensor_parallel_size=args.world_size, seed=args.seed, enable_prefix_caching=False, enforce_eager=True)
    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature, stop=tokenizer.eos_token, seed=args.seed)
    print("Sampling Params: ", sampling_params)

    ### Data Preprocessing
    eval_data, vllm_input = [], []
    system_msg = prompt_template.get_system_msg()
    if "HelpSteer2" in args.dataset:
        data = load_dataset(args.dataset, data_dir="preference")['train'].filter(lambda x: x["split"]!="train").filter(lambda x: x["preference_strength"]!=0)
        data = data.select(range(min(args.max_samples, len(data))))
        for example in data:
            user_msg = prompt_template.get_user_msg(context=example['prompt'].rstrip(), response1=example['response_1'].rstrip(), response2=example['response_2'].rstrip())
            eval_data.append({
                "system_msg": system_msg,
                "user_msg": user_msg, 
                "label": 'A' if example["preference_strength"] < 0 else 'B',
                "preference": example["preference_strength"],
                "domain": "validation"
                })
            formatted_prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}], add_generation_prompt=True, tokenize=False)
            vllm_input.append(formatted_prompt)
    elif "hh-rlhf" in args.dataset:
        data = load_dataset(args.dataset, split="test")
        data = data.select(range(min(args.max_samples, len(data))))
        for example in data:
            if random.random() < 0.5:
                user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=example["chosen"].rstrip(), response2=example["rejected"].rstrip())
                eval_data.append({
                    "system_msg": system_msg,
                    "user_msg": user_msg, 
                    "label": 'A',
                    "domain": "validation"
                })
            else:
                user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=example["rejected"].rstrip(), response2=example["chosen"].rstrip())
                eval_data.append({
                    "system_msg": system_msg,
                    "user_msg": user_msg, 
                    "label": 'B',
                    "domain": "validation"
                })
            formatted_prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}], add_generation_prompt=True, tokenize=False)
            vllm_input.append(formatted_prompt)
    elif "HelpSteer3" in args.dataset:
        data = load_dataset(args.dataset)["validation"].filter(lambda x: x["overall_preference"] != 0)
        data = data.select(range(min(args.max_samples, len(data))))
        for example in data:
            # context = MULTI_TURN_TEMPLATE.render(messages=example["context"])
            if len(example["context"]) == 1:
                context = example["context"][0]['content']    
            else:
                context = MULTI_TURN_TEMPLATE.render(messages=example["context"])
            ### prompt, label (binary), domain
            user_msg = prompt_template.get_user_msg(context=context.rstrip(), response1=example['response1'].rstrip(), response2=example['response2'].rstrip())
            eval_data.append({
                    "system_msg": system_msg,
                    "user_msg": user_msg, 
                    "label": 'A' if example["overall_preference"] < 0 else 'B',
                    "preference": example['overall_preference'],
                    "domain": example["domain"]
                    })
            formatted_prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}], add_generation_prompt=True, tokenize=False)
            vllm_input.append(formatted_prompt)
    elif "RM-Bench" in args.dataset:
        data = load_dataset(args.dataset)["train"]
        data = data.select(range(min(args.max_samples, len(data))))
        for example in data:
            combinations = []
            for i, chosen_resp in enumerate(example["chosen"]):
                for j, rejected_resp in enumerate(example["rejected"]):
                    # context = MULTI_TURN_TEMPLATE.render(messages=[{'role': 'user', 'content': example["prompt"]}])
                    if random.random() < 0.5:
                        user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=chosen_resp.rstrip(), response2=rejected_resp.rstrip())
                        # user_msg = prompt_template.get_user_msg(context=context.rstrip(), response1=chosen_resp.rstrip(), response2=rejected_resp.rstrip())
                        combinations.append({
                            "system_msg": system_msg,
                            "user_msg": user_msg, 
                            "label": 'A',
                            "difficulty": 'normal' if i == j else ('hard' if i < j else 'easy'),
                            "domain": 'safety' if 'safety' in example["domain"] else example["domain"]
                        })
                    else:
                        user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=rejected_resp.rstrip(), response2=chosen_resp.rstrip())
                        # user_msg = prompt_template.get_user_msg(context=context.rstrip(), response1=rejected_resp.rstrip(), response2=chosen_resp.rstrip())
                        combinations.append({
                            "system_msg": system_msg,
                            "user_msg": user_msg, 
                            "label": 'B',
                            "difficulty": 'normal' if i == j else ('hard' if i < j else 'easy'),
                            "domain": 'safety' if 'safety' in example["domain"] else example["domain"]
                        })
                    formatted_prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}], add_generation_prompt=True, tokenize=False)
                    vllm_input.append(formatted_prompt)
            eval_data.extend(combinations)
    elif "reward-bench" in args.dataset:
        data = load_dataset(args.dataset, split="filtered")
        data = data.select(range(min(args.max_samples, len(data))))
        # dict_keys(['prompt', 'chosen', 'chosen_model', 'rejected', 'rejected_model', 'subset', 'id'])
        for example in data:
            if random.random() < 0.5:
                user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=example["chosen"].rstrip(), response2=example["rejected"].rstrip())
                eval_data.append({
                    "system_msg": system_msg,
                    "user_msg": user_msg, 
                    "label": 'A',
                    "domain": example["subset"]
                })
            else:
                user_msg = prompt_template.get_user_msg(context=example["prompt"].rstrip(), response1=example["rejected"].rstrip(), response2=example["chosen"].rstrip())
                eval_data.append({
                    "system_msg": system_msg,
                    "user_msg": user_msg, 
                    "label": 'B',
                    "domain": example["subset"]
                })
            formatted_prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': system_msg}, {'role': 'user', 'content': user_msg}], add_generation_prompt=True, tokenize=False)
            vllm_input.append(formatted_prompt)
    else:
        raise ValueError("The dataset specified is not supported.")
        
    print("Formatted Prompt:\n\n", vllm_input[1])
    print('Total Length of Data: ', len(data))
    print('Total Length of Evaluation Data: ', len(eval_data))
    print('Total Length of vLLM Input: ', len(vllm_input))

    ### Generation
    print("Starting Generation...")
    start=time.time()
    outputs = model.generate(vllm_input, sampling_params)
    results = [
    f"<think>\n{it.outputs[0].text.strip()}" if "</think>" in it.outputs[0].text.strip() 
    else it.outputs[0].text.strip()
    for it in outputs
    ]
    timediff=time.time()-start
    print(f"time elapsed: {timediff}")
    print("Response:\n\n", results[1])
    
    ### Parse prediction and check matching format
    if 'simple-binary' in args.template:
        parse_func_strict = extract_simple_binary_strict
    elif 'multiclass' in args.template:
        parse_func_strict = extract_multiclass_strict
    elif 'binary' in args.template:
        parse_func_strict = extract_binary_strict
    elif 'judgelrm' in args.template:
        parse_func_strict = extract_scoring_strict
    
    strict_domain_scores = defaultdict(list)
    flexible_domain_scores = defaultdict(list)
    if "RM-Bench" in args.dataset:
        strict_difficulty_scores = defaultdict(list)
        flexible_difficulty_scores = defaultdict(list)
    
    invalid_output = 0
    total_len = 0
    for idx, result in enumerate(results):
        eval_data[idx]["response"] = result
        eval_data[idx]["token_length"] = len(tokenizer(result, add_special_tokens=False).input_ids)
        total_len += eval_data[idx]["token_length"]
        if parse_func_strict(result) is None:
            invalid_output += 1
            if idx == 0:
                print(result)
        eval_data[idx]["strict_predict"] = parse_func_strict(result)

        strict_domain_scores[eval_data[idx]["domain"]].append(1.0 if eval_data[idx]["strict_predict"] == eval_data[idx]["label"] else 0.0)
        flexible_domain_scores[eval_data[idx]["domain"]].append(1.0 if eval_data[idx]["strict_predict"] == eval_data[idx]["label"] else (0.5 if eval_data[idx]["strict_predict"] == None else 0.0))

        if "RM-Bench" in args.dataset:
            strict_difficulty_scores[eval_data[idx]["difficulty"]].append(1.0 if eval_data[idx]["strict_predict"] == eval_data[idx]["label"] else 0.0)
            flexible_difficulty_scores[eval_data[idx]["difficulty"]].append(1.0 if eval_data[idx]["strict_predict"] == eval_data[idx]["label"] else (0.5 if eval_data[idx]["strict_predict"] == None else 0.0))
        
    invalid_output_ratio = invalid_output / len(eval_data)
    avg_response_length = total_len / len(eval_data)
    print(f"Percentage of Invalid Output:  {invalid_output_ratio * 100:.2f}%")
    print(f"Average Response Length: {avg_response_length:.2f}")
    
    print(f"***{model_name}***")
    strict_mean_per_domain = {domain: sum(correct) / len(correct) * 100 for domain, correct in strict_domain_scores.items()}
    flexible_mean_per_domain = {domain: sum(correct) / len(correct) * 100 for domain, correct in flexible_domain_scores.items()}
    if "RM-Bench" in args.dataset:
        strict_mean_per_difficulty = {difficulty: sum(correct) / len(correct) * 100 for difficulty, correct in strict_difficulty_scores.items()}
        flexible_mean_per_difficulty = {difficulty: sum(correct) / len(correct) * 100 for difficulty, correct in flexible_difficulty_scores.items()}
    if "reward-bench" in args.dataset:
        strict_mean_per_domain = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, strict_mean_per_domain)
        flexible_mean_per_domain = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, flexible_mean_per_domain)
    # Compute overall average
    strict_all_scores = [score for correct in strict_domain_scores.values() for score in correct] 
    flexible_all_scores = [score for correct in flexible_domain_scores.values() for score in correct]
    strict_overall_average = sum(strict_all_scores) / len(strict_all_scores) * 100
    flexible_overall_average = sum(flexible_all_scores) / len(flexible_all_scores) * 100
    if "RM-Bench" in args.dataset:
        strict_all_scores2 = [score for correct in strict_difficulty_scores.values() for score in correct]
        flexible_all_scores2 = [score for correct in flexible_difficulty_scores.values() for score in correct]
        strict_overall_average2 = sum(strict_all_scores2) / len(strict_all_scores2) * 100
        flexible_overall_average2 = sum(flexible_all_scores2) / len(flexible_all_scores2) * 100
        # Assert both averages are equal
        assert strict_overall_average == strict_overall_average2, "Mismatch between overall averages computed from domains and difficulties!"
        assert flexible_overall_average == flexible_overall_average2, "Mismatch between overall averages computed from domains and difficulties!"
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, OUT_NAME), 'w') as f:
        json.dump(eval_data, f, indent=4)

    if "HelpSteer2" in args.dataset or "HelpSteer3" in args.dataset or "hh-rlhf" in args.dataset:
        result = format_accuracy_table(strict_mean_per_domain, flexible_mean_per_domain, strict_overall_average, flexible_overall_average, "Domain", invalid_output_ratio, avg_response_length)
        print(result)
        with open(os.path.join(save_dir, "domain_accuracy_table.txt"), 'w') as f:
            f.write(result)
    elif "RM-Bench" in args.dataset:
        result1 = format_accuracy_table(strict_mean_per_domain, flexible_mean_per_domain, strict_overall_average, flexible_overall_average, "Domain", invalid_output_ratio, avg_response_length)
        result2 = format_accuracy_table(strict_mean_per_difficulty, flexible_mean_per_difficulty, strict_overall_average, flexible_overall_average, "Difficulty", invalid_output_ratio, avg_response_length)
        print(result1)
        print(result2)
        with open(os.path.join(save_dir, "domain_accuracy_table.txt"), 'w') as f:
            f.write(result1)
        with open(os.path.join(save_dir, "difficulty_accuracy_table.txt"), 'w') as f:
            f.write(result2)
    elif "reward-bench" in args.dataset:
        result = format_accuracy_table(strict_mean_per_domain, flexible_mean_per_domain, strict_overall_average, flexible_overall_average, "Section", invalid_output_ratio, avg_response_length)
        print(result)
        with open(os.path.join(save_dir, "section_accuracy_table.txt"), 'w') as f:
            f.write(result)

def load_hf_tokenizer(
        model_name_or_path, 
        revision=None,
        tokenizer_name_or_path=None, 
        use_fast_tokenizer=True,
        padding_side="left"
    ):
        if not tokenizer_name_or_path:
            tokenizer_name_or_path = model_name_or_path
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, revision=revision)
        except:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, revision=revision)
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer
    
if __name__ == "__main__":
    main()