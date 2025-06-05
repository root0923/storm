import dspy
import re
from typing import Union, List
from .chinese_utils import clean_chinese_expert_output, clean_chinese_output
from ...logging_wrapper import LoggingWrapper


class GenerateExpertGeneral(dspy.Signature):
    """您需要选择一组多样化的专家，适合受邀参加关于给定主题的圆桌讨论。
    每位专家应该代表与该主题相关的不同观点、角色或立场。
    您可以使用提供的主题背景信息作为灵感。为每位专家添加其专业知识的描述以及他们在讨论中的重点。
    输出中无需包含发言人姓名。
    严格按照以下格式：
    1. [发言人1角色]：[发言人1简短描述]
    2. [发言人2角色]：[发言人2简短描述]
    """

    topic = dspy.InputField(prefix="关注的主题：", format=str)
    background_info = dspy.InputField(
        prefix="关于主题的背景信息：\n", format=str
    )
    topN = dspy.InputField(prefix="需要的发言人数量：", format=str)
    experts = dspy.OutputField(format=str)


class GenerateExpertWithFocus(dspy.Signature):
    """
    您需要选择一组适合就特定焦点进行圆桌讨论的发言人。
    您可以考虑邀请在该主题上持相反立场的发言人；代表不同利益方的发言人；确保所选发言人与提供的特定背景和场景直接相关。
    例如，如果讨论焦点是关于某特定大学的近期事件，可以考虑邀请学生、教职员工、报道该事件的记者、大学官员和当地社区成员。
    使用提供的主题背景信息作为灵感。为每位发言人添加其兴趣的描述以及他们在讨论中的重点。
    输出中无需包含发言人姓名。
    严格按照以下格式：
    1. [发言人1角色]：[发言人1简短描述]
    2. [发言人2角色]：[发言人2简短描述]
    """

    topic = dspy.InputField(prefix="关注的主题：", format=str)
    background_info = dspy.InputField(prefix="背景信息：\n", format=str)
    focus = dspy.InputField(prefix="讨论焦点：", format=str)
    topN = dspy.InputField(prefix="需要的发言人数量：", format=str)
    experts = dspy.OutputField(format=str)


class GenerateExpertModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.generate_expert_general = dspy.Predict(GenerateExpertGeneral)
        self.generate_expert_w_focus = dspy.ChainOfThought(GenerateExpertWithFocus)

    def trim_background(self, background: str, max_words: int = 100):
        words = background.split()
        cur_len = len(words)
        if cur_len <= max_words:
            return background
        trimmed_words = words[: min(cur_len, max_words)]
        trimmed_background = " ".join(trimmed_words)
        return f"{trimmed_background} [rest content omitted]."

    def forward(
        self, topic: str, num_experts: int, background_info: str = "", focus: str = ""
    ):
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            while True:
                if not focus:
                    output = self.generate_expert_general(
                        topic=topic, background_info=background_info, topN=num_experts
                    ).experts
                else:
                    background_info = self.trim_background(
                        background=background_info, max_words=100
                    )
                    output = self.generate_expert_w_focus(
                        topic=topic,
                        background_info=background_info,
                        focus=focus,
                        topN=num_experts,
                    ).experts
                
                text = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
                text = re.sub(r'<think>', '', text)
                output = re.sub(r'</think>', '', text)
                
                # 使用新的中文处理函数
                expert_list = clean_chinese_expert_output(output)
                
                if expert_list:
                    break
        
        return dspy.Prediction(experts=expert_list, raw_output=output)