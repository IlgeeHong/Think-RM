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