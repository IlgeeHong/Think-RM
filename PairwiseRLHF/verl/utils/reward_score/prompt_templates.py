# prompt_templates.py

class BasePromptTemplate:
    def __init__(self, system_msg: str, user_msg: str):
        self.system_msg = system_msg
        self.user_msg = user_msg

    def get_user_msg(self, **kwargs) -> str:
        """Fill the template with keyword arguments."""
        return self.user_msg.format(**kwargs)
    
    def get_system_msg(self) -> str:
        """Fill the template with keyword arguments."""
        return self.system_msg

class NoThinkBinaryPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (    
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is superior. "
        "Begin your evaluation by comparing the two responses and provide a detailed explanation, explicitly referencing the criteria. "
        "After providing your explanation, output your final verdict by strictly following this format: " 
        "<answer>A</answer> if assistant A is better, and <answer>B</answer> if assistant B is better."
        )
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class NoThinkMulticlassPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is better and how much better it is using the scale below:\n\n"
        "-3 if Assistant A's response is much better than Assistant B's response\n"
        "-2 if Assistant A's response is better than Assistant B's response\n"
        "-1 if Assistant A's response is slightly better than Assistant B's response\n"
        "1 if Assistant B's response is slightly better than Assistant A's response\n"
        "2 if Assistant B's response is better than Assistant A's response\n"
        "3 if Assistant B's response is much better than Assistant A's response\n\n"
        "Begin your evaluation by comparing the two responses and provide a detailed explanation, explicitly referencing the criteria. "
        "After providing your explanation, output your final score inside the <answer></answer> tag."
        )
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class SimpleBinaryPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "Please act as an impartial judge and evaluate the quality of the two responses to the context displayed below. "
        "You should choose the response that follows the user's instructions and answers the user's question better. "
        "Your evaluation should consider following factors:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "Begin your evaluation by comparing the two responses and provide a short explanation. "
        "After providing your explanation, output your final verdict by strictly following this format: " 
        "<answer>1</answer> if @Response 1 is better, and <answer>2</answer> if @Response 2 is better."
        )
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of @Response 1]\n"
        "{response1}\n"
        "[The End of @Response 1]\n\n"
        "[The Start of @Response 2]\n"
        "{response2}\n"
        "[The End of @Response 2]"
        )
        super().__init__(system_msg, user_msg)

class SimpleMulticlassPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "Please act as an impartial judge and evaluate the quality of the two responses to the context displayed below. "
        "You should choose the response that follows the user's instructions and answers the user's question better, and how much better it is using the scale below:\n\n"
        "-3 if Assistant A's response is much better than Assistant B's response\n"
        "-2 if Assistant A's response is better than Assistant B's response\n"
        "-1 if Assistant A's response is slightly better than Assistant B's response\n"
        "1 if Assistant B's response is slightly better than Assistant A's response\n"
        "2 if Assistant B's response is better than Assistant A's response\n"
        "3 if Assistant B's response is much better than Assistant A's response\n\n"
        "Your evaluation should consider following factors:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "Begin your evaluation by comparing the two responses and provide a short explanation. "
        "After providing your explanation, output your final score inside the <answer></answer> tag."
        )
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of @Response 1]\n"
        "{response1}\n"
        "[The End of @Response 1]\n\n"
        "[The Start of @Response 2]\n"
        "{response2}\n"
        "[The End of @Response 2]"
        )
        super().__init__(system_msg, user_msg)

class MTBenchPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the chat context displayed below. "
        "You should choose the assistant that follows the user's instructions and answers the user's question better. " 
        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. " 
        "Begin your evaluation by comparing the two responses and provide a short explanation. " 
        "Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. " 
        "Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. " 
        "Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: " 
        "\\boxed{{A}} if assistant A is better, and \\boxed{{B}} if assistant B is better."
        )
        user_msg = (
        "[Chat Context]\n"
        "{context}\n\n"
        "[The Start of Assistant A's Answer]\n"
        "{response1}\n"
        "[The End of Assistant A's Answer]\n\n"
        "[The Start of Assistant B's Answer]\n"
        "{response2}\n"
        "[The End of Assistant B's Answer]"
        )
        super().__init__(system_msg, user_msg)

class JudgeLRMPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are a helpful assistant. The assistant first performs a detailed, "
        "step-by-step reasoning process in its mind and then provides the user with "
        "the answer. The reasoning process and answer are enclosed within <think> "
        "</think> and <answer> </answer> tags, respectively, i.e., <think> detailed "
        "reasoning process here, explaining each step of your evaluation for both "
        "assistants </think><answer> answer here </answer>. Now the user asks you "
        "to judge the performance of two AI assistants in response to the question. "
        "Score assistants 1-10 (higher=better). Criteria includes helpfulness, "
        "relevance, accuracy, and level of detail. Avoid order, length, style or "
        "other bias. After thinking, when you finally reach a conclusion, clearly "
        "provide your evaluation scores within <answer> </answer> tags, i.e., for "
        "example, <answer>3</answer><answer>5</answer>."
        )
        user_msg = (
        "[Question]\n"
        "{context}\n"
        "[Assistant 1's Answer]\n"
        "{response1}\n"
        "[Assistant 2's Answer]\n"
        "{response2}"
        )
        super().__init__(system_msg, user_msg)
        
class ReasoningBinaryPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is superior. "
        "Your evaluation must follow this exact structure:\n\n"
        "<think>\n"
        "Your internal thought here\n" #"Your internal thought here\n"
        "</think>\n\n"
        "<rationale>\n"
        "Your concise rationale here\n" #explicitly referencing the criteria
        "</rationale>\n\n"
        "<answer>\n"
        "Your final verdict here, A or B only\n"
        "</answer>")
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class NaiveReasoningBinaryPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is superior. "
        "Begin your evaluation by thinking through the problem step by step. Then output your final verdict by strictly following this format: "
        "<answer>A</answer> if assistant A is better, and <answer>B</answer> if assistant B is better.")
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class ReasoningMulticlassPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is better and how much better it is using the scale below:\n\n"
        "-3 if Assistant A's response is much better than Assistant B's response\n"
        "-2 if Assistant A's response is better than Assistant B's response\n"
        "-1 if Assistant A's response is slightly better than Assistant B's response\n"
        "1 if Assistant B's response is slightly better than Assistant A's response\n"
        "2 if Assistant B's response is better than Assistant A's response\n"
        "3 if Assistant B's response is much better than Assistant A's response\n\n"
        "Your evaluation must follow this exact structure:\n\n"
        "<think>\n"
        "Your internal thought here\n" #"Your internal thought here\n"
        "</think>\n\n"
        "<rationale>\n"
        "Your concise rationale here\n" #explicitly referencing the criteria
        "</rationale>\n\n"
        "<answer>\n"
        "Your preference score here (numerical value only)\n"
        "</answer>")
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class NaiveReasoningMulticlassPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should be based on the following six criteria:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "After carefully considering these criteria, determine which assistant's response is better and how much better it is using the scale below:\n\n"
        "-3 if Assistant A's response is much better than Assistant B's response\n"
        "-2 if Assistant A's response is better than Assistant B's response\n"
        "-1 if Assistant A's response is slightly better than Assistant B's response\n"
        "1 if Assistant B's response is slightly better than Assistant A's response\n"
        "2 if Assistant B's response is better than Assistant A's response\n"
        "3 if Assistant B's response is much better than Assistant A's response\n\n"
        "Begin your evaluation by thinking through the problem step by step. Then output your final score inside the <answer></answer> tag.")
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)

class StepbyStepReasoningBinaryPromptTemplate(BasePromptTemplate):
    def __init__(self):
        system_msg = (
        "You are an impartial judge, tasked with evaluating the quality of the two AI assistants' responses to the context displayed below. "
        "Your evaluation should adhere to the following steps:\n\n"
        "[Step 1] Analyze the context of the user's question or instruction and identify the user's intent. Then answer the following question: "
        "What is the user ultimately trying to achieve with their question or instruction?\n\n"
        "[Step 2] Describe what a good response should look like.\n\n"
        "[Step 3] Based on [Step 1] and [Step 2], apply the following criteria to evaluate the two responses:\n\n"
        "- Helpfulness: Overall helpfulness of the response to the user's question or instruction.\n"
        "- Correctness: Inclusion of all pertinent facts without errors.\n"
        "- Coherence: Consistency and clarity of expression.\n"
        "- Complexity: Intellectual depth required to write response (i.e., whether the response can be written by anyone with basic language competency or requires deep domain expertise).\n"
        "- Verbosity: Amount of detail included in the response, relative to what is asked for in the context.\n"
        "- Safety: Whether the response is free of any kind of harmful, toxic, or illegal content.\n\n"
        "[Step 4] Based on [Step 3], determine which assistant's response is superior. "
        "Output your final verdict by strictly following this format: "
        "<answer>A</answer> if assistant A is better, and <answer>B</answer> if assistant B is better.")
        user_msg = (
        "[The Start of Context]\n"
        "{context}\n"
        "[The End of Context]\n\n"
        "[The Start of Assistant A's Response]\n"
        "{response1}\n"
        "[The End of Assistant A's Response]\n\n"
        "[The Start of Assistant B's Response]\n"
        "{response2}\n"
        "[The End of Assistant B's Response]"
        )
        super().__init__(system_msg, user_msg)