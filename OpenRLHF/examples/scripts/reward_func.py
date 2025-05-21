import torch
import re
# def reward_func(queries, prompts, labels):
#     # queries is prompts + responses
#     # labels is answers
#     print(queries)
#     return torch.randn(len(queries))

def reward_func(queries, prompts, labels):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<score>\n.*?\n</score>$"
    pattern = r"^<think>\s*.*?\s*</think>\s*<rationale>\s*.*?\s*</rationale>\s*<final answer>\s*.*?\s*</final answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in queries]
    ### strict format rewards
    strict_format_rewards = torch.tensor([1.0 if match else 0.0 for match in matches])
    ### soft format rewards
    soft_format_rewards = torch.tensor([count_tags(content) for content in queries])
    ### strict accuracy rewards
    ### soft accuracy rewards
    strict_accuracy_rewards = []
    soft_accuracy_rewards = []
    for content, sol in zip(queries, labels):
        # Parse the gold solution using last_match.
        gold_scores = extract_plain_score_block(sol)
        if not gold_scores:
            print("Failed to parse gold solution:", sol)
            strict_accuracy_reward = 1.0
            soft_accuracy_reward = 0.0
        else:
            # Parse the completion output using last_match.
            comp_scores = extract_score_block(content)
            strict_accuracy_reward = 1.0 if comp_scores == gold_scores else 0.0
            # Initialize reward; 0.2 is awarded for each of the 5 criteria.
            correct_count = sum(1 for criterion in gold_scores if comp_scores.get(criterion) == gold_scores[criterion])
            soft_accuracy_reward = correct_count * 0.2
        strict_accuracy_rewards.append(strict_accuracy_reward)
        soft_accuracy_rewards.append(soft_accuracy_reward)
    strict_accuracy_rewards = torch.tensor(strict_accuracy_rewards)
    soft_accuracy_rewards = torch.tensor(soft_accuracy_rewards)
    total_rewards = strict_format_rewards + soft_format_rewards + strict_accuracy_rewards + soft_accuracy_rewards
    # print(total_rewards)
    # import sys
    # print("queries shape", len(queries))
    # print(labels)
    # print(format_rewards)
    # sys.exit("bye")
    return total_rewards

def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>") == 1:
            count += 0.125
        if text.count("</think>") == 1:
            count += 0.125
        if text.count("<rationale>") == 1:
            count += 0.125
        if text.count("</rationale>") == 1:
            count += 0.125
        if text.count("<final answer>") == 1:
            count += 0.125
        if text.count("</final answer>") == 1:
            count += 0.125
        return count

def extract_plain_score_block(text):
    """
    Extracts scores from text that is directly in the multiline format:
    
      helpfulness: [[score]]
      correctness: [[score]]
      coherence: [[score]]
      complexity: [[score]]
      verbosity: [[score]]
      
    If the format is exactly as expected (exactly 5 lines in order), returns a dictionary mapping each key to its score.
    Otherwise, returns an empty dictionary.
    """
    expected_order = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    lines = text.strip().splitlines()
    if len(lines) != 5:
        return {}
    scores = {}
    pattern = re.compile(r"^(\w+):\s*\[\[(\d+)\]\]\s*$")
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
    Extracts the final <score>...</score> block from the text, ensuring that:
      - There is exactly one <score>...</score> block.
      - It appears at the very end of the text (only whitespace is allowed after the block).
    
    Inside the <score> tag, extra text is allowed. However, this function searches for a contiguous block 
    of text containing the following five lines (in order):
    
      helpfulness: [[score]]
      correctness: [[score]]
      coherence: [[score]]
      complexity: [[score]]
      verbosity: [[score]]
    
    If found, returns a dictionary mapping each key to its integer score. Otherwise, returns an empty dictionary.
    """
    # Find all <final answer>...</final answer> blocks in the text.
    score_blocks = re.findall(r"<final answer>(.*?)</final answer>", text, re.DOTALL)
    if len(score_blocks) != 1:
        return {}
    
    # Ensure the <final answer> block is at the very end of the text (only whitespace allowed after).
    if not re.search(r"<final answer>.*?</final answer>\s*\Z", text, re.DOTALL):
        return {}
    
    # Get the content inside the <score> tag.
    score_content = score_blocks[0]
    
    # Use a regex to search for the required five lines in order.
    # We allow any text (including newlines) between the lines.
    pattern = re.compile(
        r"helpfulness:\s*\[\[(\d+)\]\].*?"  # helpfulness line
        r"correctness:\s*\[\[(\d+)\]\].*?"   # correctness line
        r"coherence:\s*\[\[(\d+)\]\].*?"     # coherence line
        r"complexity:\s*\[\[(\d+)\]\].*?"     # complexity line
        r"verbosity:\s*\[\[(\d+)\]\]",        # verbosity line
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
# def extract_scores(text, extraction_mode="last_match"):
#     """
#     Extracts scores from a text that may contain one or more candidate blocks.
    
#     A valid block exactly follows this format (order can be arbitrary but all keys must appear):
    
#       helpfulness: [[score]]
#       correctness: [[score]]
#       coherence: [[score]]
#       complexity: [[score]]
#       verbosity: [[score]]
    
#     Blocks are assumed to be separated by one or more blank lines.
    
#     Depending on the extraction_mode:
#       - "first_match": returns the first valid block.
#       - "last_match": returns the last valid block.
      
#     If no valid block is found, returns an empty dictionary.
#     """
#     expected_keys = {'helpfulness', 'correctness', 'coherence', 'complexity', 'verbosity'}
#     # Split the text into candidate blocks by blank lines.
#     blocks = text.strip().split("\n\n")
#     valid_blocks = []
    
#     # Regex pattern to match a line in the format: key: [[score]]
#     pattern = re.compile(r"^(\w+):\s*\[\[(\d+)\]\]\s*$")
    
#     for block in blocks:
#         lines = block.strip().splitlines()
#         if len(lines) != 5:
#             continue  # Not a valid block since it doesn't have exactly 5 lines.
#         scores = {}
#         valid = True
#         for line in lines:
#             match = pattern.match(line)
#             if not match:
#                 valid = False
#                 break
#             key, score = match.group(1), int(match.group(2))
#             if key in scores:
#                 valid = False
#                 break
#             scores[key] = score
#         if valid and set(scores.keys()) == expected_keys:
#             valid_blocks.append(scores)
    
#     if not valid_blocks:
#         return {}
    
#     if extraction_mode == "first_match":
#         return valid_blocks[0]
#     elif extraction_mode == "last_match":
#         return valid_blocks[-1]
#     else:
#         # Default behavior if an unknown extraction_mode is provided.
#         return valid_blocks[0]