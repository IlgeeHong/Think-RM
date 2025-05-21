import torch
import re

def reward_func(queries, prompts, labels):

    print(queries[0])

    responses = [content[len(prompt):] for content, prompt in zip(queries, prompts)]

    # Compute soft format rewards: 1/3 point for each required tag block found.
    soft_format_rewards = torch.tensor([soft_format_score(content) for content in responses])
    
    attribute_rewards = []
    # Expected attribute keys.
    expected_keys = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    
    for content, sol in zip(responses, labels):
        # Extract the gold scores from the plain multiline format.
        gold_scores = extract_plain_score_block(sol)
        # Extract the computed scores from the <final answer> block.
        comp_scores = extract_score_block(content)
        
        # If extraction fails (i.e. empty dict returned), assign attribute reward as 0.
        if gold_scores == {} or comp_scores == {}:
            attr_reward = 0.0
        else:
            # Award 0.2 for each attribute that exactly matches.
            attr_reward = sum(
                0.2 for key in expected_keys if gold_scores.get(key) == comp_scores.get(key)
            )
        attribute_rewards.append(attr_reward)
    
    attribute_rewards = torch.tensor(attribute_rewards)
    total_rewards = soft_format_rewards + attribute_rewards
    return total_rewards

def soft_format_score(text):
    """
    Returns a soft format score based on the presence of the required tag blocks.
    Awards:
      - 1/3 point if exactly one <think>...</think> block is found.
      - 1/3 point if exactly one <rationale>...</rationale> block is found.
      - 1/3 point if exactly one <final answer>...</final answer> block is found.
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

    # Process <final answer> tag
    final_answer_matches = re.findall(r"<final answer>\s*.*?\s*</final answer>", text, re.DOTALL)
    count_final_answer = len(final_answer_matches)
    if count_final_answer == 1:
        score_final_answer = 1/3
        print(f"<final answer>...</final answer> tag is found exactly once. Score for <final answer>: {score_final_answer:.3f}")
    else:
        score_final_answer = 0.0
        # if count_final_answer == 0:
        #     print("<final answer>...</final answer> tag is not found.")
        # else:
        #     print(f"<final answer>...</final answer> tag is found {count_final_answer} times. Score for <final answer>: 0")
    total_score += score_final_answer

    return total_score



def extract_plain_score_block(text):
    """
    Extracts scores from text in the multiline format:
    
      helpfulness: \boxed{score}
      correctness: \boxed{score}
      coherence: \boxed{score}
      complexity: \boxed{score}
      verbosity: \boxed{score}
      
    If the format is exactly as expected (exactly 5 lines in order),
    returns a dictionary mapping each key to its score.
    Otherwise, returns an empty dictionary.
    """
    expected_order = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    lines = text.strip().splitlines()
    scores = {}
    pattern = re.compile(r"^(\w+):\s*\\boxed\{(\d+)\}\s*$")
    for line, expected_key in zip(lines, expected_order):
        match = pattern.match(line)
        if not match:
            return {}
        key, score = match.group(1), int(match.group(2))
        if key != expected_key:
            return {}
        scores[key] = score
    return scores

def extract_score_block(text):
    """
    Extracts the final <final answer>...</final answer> block from the text, ensuring that:
      - There is exactly one <final answer>...</final answer> block.
      - It appears at the very end of the text (only whitespace allowed after).
    
    Inside the <final answer> tag, extra text is allowed. However, this function searches for a contiguous block 
    containing the following five lines (in order):
    
      helpfulness: \boxed{score}
      correctness: \boxed{score}
      coherence: \boxed{score}
      complexity: \boxed{score}
      verbosity: \boxed{score}
    
    If found, returns a dictionary mapping each key to its integer score.
    Otherwise, returns an empty dictionary.
    """
    # Find all <final answer>...</final answer> blocks.
    score_blocks = re.findall(r"<final answer>(.*?)</final answer>", text, re.DOTALL)
    if len(score_blocks) != 1:
        return {}
    
    # Ensure the <final answer> block is at the very end (only whitespace allowed after).
    if not re.search(r"<final answer>.*?</final answer>\s*$", text, re.DOTALL):
        return {}
    
    # Get the content inside the <final answer> tag.
    score_content = score_blocks[0]
    
    # Use a regex to search for the required five lines in order.
    pattern = re.compile(
        r"helpfulness:\s*\\boxed\{(\d+)\}\s*.*?"
        r"correctness:\s*\\boxed\{(\d+)\}\s*.*?"
        r"coherence:\s*\\boxed\{(\d+)\}\s*.*?"
        r"complexity:\s*\\boxed\{(\d+)\}\s*.*?"
        r"verbosity:\s*\\boxed\{(\d+)\}",
        re.DOTALL
    )
    
    m = pattern.search(score_content)
    if not m:
        return {}
    
    scores = {
        "helpfulness": int(m.group(1)),
        "correctness": int(m.group(2)),
        "coherence": int(m.group(3)),
        "complexity": int(m.group(4)),
        "verbosity": int(m.group(5)),
    }
    return scores
