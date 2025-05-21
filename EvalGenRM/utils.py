import re

def extract_simple_binary_strict(text):
    label_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(label_blocks) < 1:
        return None
    label_content = label_blocks[-1].strip()
    matches = re.findall(r"(1|2)", label_content)
    if not matches:
        return None
    label = int(matches[-1])
    if label == 1:
        return 'A'
    elif label == 2:
        return 'B'
    else:
        return None

def extract_simple_binary_flexible(text):
    matches = re.findall(r"(1|2)", text)
    if not matches:
        return None
    label = int(matches[-1])
    if label == 1:
        return 'A'
    elif label == 2:
        return 'B'
    else:
        return None

def extract_multiclass_strict(text):
    score_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(score_blocks) < 1:
        return None
    score_content = score_blocks[-1].strip()
    matches = re.findall(r"(-3|-2|-1|1|2|3)", score_content)
    if not matches:
        return None
    score = int(matches[-1])
    if score < 0:
        return 'A'
    elif score > 0:
        return 'B'
    else:
        return None

def extract_multiclass_flexible(text):
    matches = re.findall(r"(-3|-2|-1|1|2|3)", text)
    if not matches:
        return None
    score = int(matches[-1])
    if score < 0:
        return 'A'
    elif score > 0:
        return 'B'
    else:
        return None

def extract_binary_strict(text):
    alphabet_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(alphabet_blocks) < 1:
        return None
    alphabet_content = alphabet_blocks[-1].strip()
    matches = re.findall(r"(A|B)", alphabet_content)
    if not matches:
        return None
    return str(matches[-1])

def extract_binary_flexible(text):
    matches = re.findall(r"(A|B)", text)
    if not matches:
        return None
    return str(matches[-1])

def extract_scoring_strict(text):
    score_blocks = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if len(score_blocks) < 2:
        return None
    
    first, second = [block.strip() for block in score_blocks[-2:]]
    first_matches = re.findall(r"\b(?:10|[1-9])\b", first)
    second_matches = re.findall(r"\b(?:10|[1-9])\b", second)

    if not first_matches or not second_matches:
        return None

    first_score = int(first_matches[-1])
    second_score = int(second_matches[-1])

    if first_score > second_score:
        return 'A'
    elif first_score < second_score:
        return 'B'
    else:
        return None

def extract_scoring_flexible(text):
    
    matches = re.findall(r"\b(?:10|[1-9])\b", text)

    if len(matches) < 2:
        return None

    first_score = int(matches[-2])
    second_score = int(matches[-1])

    if first_score > second_score:
        return 'A'
    elif first_score < second_score:
        return 'B'
    else:
        return None

def format_accuracy_table(
    strict_mean: dict,
    flexible_mean: dict,
    overall_strict: float,
    overall_flexible: float,
    prefix: str = "Domain",
    invalid_output_ratio: float = None,
    avg_response_length: float = None
) -> str:
    """
    Formats a table of strict and flexible accuracies per domain.

    Parameters:
    - strict_mean (dict): Domain -> strict accuracy float
    - flexible_mean (dict): Domain -> flexible accuracy float
    - overall_strict (float): Overall strict accuracy
    - overall_flexible (float): Overall flexible accuracy
    - prefix (str): Label for the domain/category column
    - invalid_output_ratio (float): Ratio of invalid outputs (0.0 ~ 1.0)
    - avg_response_length (float): Average length of generated responses

    Returns:
    - str: Formatted accuracy table
    """
    lines = []

    if invalid_output_ratio is not None:
        lines.append(f"Percentage of Invalid Output: {invalid_output_ratio * 100:.2f}%")
    if avg_response_length is not None:
        lines.append(f"Ave. Response Length: {avg_response_length:.2f}")
    if lines:
        lines.append("")  # Add a blank line before the table
    
    lines.extend([
        "-------------------------------------------------------------",
        f"{prefix:<13} | Strict Accuracy (%)  | Flexible Accuracy (%) ",
        "-------------------------------------------------------------",
    ])

    all_domains = sorted(set(strict_mean) | set(flexible_mean))

    for domain in all_domains:
        strict_val = strict_mean.get(domain, float("nan"))
        flexible_val = flexible_mean.get(domain, float("nan"))
        lines.append(f"{domain.capitalize():<13} | {strict_val:>19.2f} | {flexible_val:>23.2f}")

    lines.extend([
        "-------------------------------------------------------------",
        f"{'Overall':<13} | {overall_strict:>19.2f} | {overall_flexible:>23.2f}",
        "-------------------------------------------------------------"
    ])

    return "\n".join(lines)