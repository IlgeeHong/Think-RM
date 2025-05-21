import torch
import re

def reward_func(queries, prompts, labels):

    len_think_prompt = len("<think>\n")
    responses = [content[len(prompt)-len_think_prompt:] for content, prompt in zip(queries, prompts)]
    print("response:\n\n", responses[0])
    print("label: ", labels[0])
    
    # Compute soft format rewards: award 1/3 point for each required tag block found.
    # soft_format_rewards = torch.tensor([soft_format_score(content) for content in responses])
    ### strict pattern, any extra text before or after these tags will cause the match to fail
    # pattern = r"^<think>\s*.*?\s*</think>\s*<rationale>\s*.*?\s*</rationale>\s*<answer>\s*.*?\s*</answer>"
    pattern = r"^<think>\s*(.*?)\s*</think>\s*<rationale>\s*(.*?)\s*</rationale>\s*<answer>\s*(.*?)\s*</answer>(?:\s*<\|eot_id\|>)*$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in responses]
    strict_format_rewards = torch.tensor([1.0 if match else 0.0 for match in matches])
    
    accuracy_rewards = []
    for content, sol in zip(responses, labels):
        # Parse the gold score as an integer.
        try:
            gold_score = int(sol)
        except ValueError:
            gold_score = None
        
        # Extract the computed score from the <answer> block.
        comp_score = extract_score_block(content)
        # If extraction fails or the gold score is invalid, assign an accuracy reward of 0.
        if gold_score is None or comp_score == {}:
            acc_reward = 0.0
        elif comp_score == gold_score:
            acc_reward = 1.0
        else:
            # Check if the computed score is in the allowed range: -3 to 3.
            if abs(comp_score) > 3:
                acc_reward = 0.0
            else:
                # Loose accuracy criterion: award 0.5 if both scores have the same sign.
                if (gold_score > 0 and comp_score > 0) or (gold_score < 0 and comp_score < 0):
                    acc_reward = 0.5
                else:
                    acc_reward = 0.0
        accuracy_rewards.append(acc_reward)
    
    accuracy_rewards = torch.tensor(accuracy_rewards)
    
    print("strict_format_rewards: ", strict_format_rewards)
    print("accuracy_rewards: ", accuracy_rewards)
    # total_rewards = soft_format_rewards + accuracy_rewards
    total_rewards = strict_format_rewards + accuracy_rewards

    return total_rewards

def soft_format_score(text):
    """
    Returns a soft format score based on the presence of the required tag blocks.
    Awards:
      - 1/3 point if exactly one <think>...</think> block is found.
      - 1/3 point if exactly one <rationale>...</rationale> block is found.
      - 1/3 point if exactly one <answer>...</answer> block is found.
    If there is more than one occurrence for any tag type, the reward for that tag is 0.
    Maximum soft format score = 1.
    
    Additionally, prints a message for each tag type.
    """
    total_score = 0.0

    # Process <think> tag
    think_matches = re.findall(r"<think>\s*.*?\s*</think>", text, re.DOTALL)
    count_think = len(think_matches)
    if count_think == 1:
        score_think = 1/3
        print(f"<think>...</think> tag is found exactly once. Score for <think>: {score_think:.3f}")
    else:
        score_think = 0.0
        # if count_think == 0:
        #     print("<think>...</think> tag is not found.")
        # else:
        #     print(f"<think>...</think> tag is found {count_think} times. Score for <think>: 0")
    total_score += score_think

    # Process <rationale> tag
    rationale_matches = re.findall(r"<rationale>\s*.*?\s*</rationale>", text, re.DOTALL)
    count_rationale = len(rationale_matches)
    if count_rationale == 1:
        score_rationale = 1/3
        print(f"<rationale>...</rationale> tag is found exactly once. Score for <rationale>: {score_rationale:.3f}")
    else:
        score_rationale = 0.0
        # if count_rationale == 0:
        #     print("<rationale>...</rationale> tag is not found.")
        # else:
        #     print(f"<rationale>...</rationale> tag is found {count_rationale} times. Score for <rationale>: 0")
    total_score += score_rationale

    # Process <answer> tag
    answer_matches = re.findall(r"<answer>\s*.*?\s*</answer>", text, re.DOTALL)
    count_answer = len(answer_matches)
    if count_answer == 1:
        score_answer = 1/3
        print(f"<answer>...</answer> tag is found exactly once. Score for <answer>: {score_answer:.3f}")
    else:
        score_answer = 0.0
        # if count_answer == 0:
        #     print("<answer>...</answer> tag is not found.")
        # else:
        #     print(f"<answer>...</answer> tag is found {count_answer} times. Score for <answer>: 0")
    total_score += score_answer

    return total_score



def extract_score_block(text):
    """
    Extracts the final <answer>...</answer> block from the text, ensuring that:
      - There is exactly one <answer>...</answer> block.
      - It appears at the very end of the text (only whitespace allowed after the block).
    
    Inside the <answer> tag, extra text is allowed. However, this function searches for a contiguous block 
    containing a \boxed{} expression.
    
    It returns the integer extracted from the last occurrence of \boxed{<number>} in that block.
    If the format is not as expected, returns an empty dictionary.
    """
    # Find all <answer>...</answer> blocks in the text.
    score_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(score_blocks) != 1:
        return {}
    
    # # Ensure the <answer> block is at the very end of the text (only whitespace allowed after).
    # if not re.search(r"<answer>.*?</answer>\s*$", text, re.DOTALL):
    #     return {}
    
    # Get the content inside the <answer> tag.
    score_content = score_blocks[0]
    
    # Use regex to extract all occurrences of \boxed{<number>} (allowing a leading minus sign).
    numbers = re.findall(r"\\boxed\{(-?\d+)\}", score_content)
    if not numbers:
        return {}
    
    # Convert the last extracted number to an integer.
    return int(numbers[-1])
