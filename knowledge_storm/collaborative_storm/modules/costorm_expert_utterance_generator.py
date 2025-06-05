import dspy
from typing import Union

from .callback import BaseCallbackHandler
from .collaborative_storm_utils import (
    trim_output_after_hint,
    extract_and_remove_citations,
    keep_first_and_last_paragraph,
)
from .chinese_utils import clean_chinese_output
from ...utils import ArticleTextProcessing
from .grounded_question_answering import AnswerQuestionModule
from .grounded_question_generation import ConvertUtteranceStyle
from ...dataclass import ConversationTurn
from ...logging_wrapper import LoggingWrapper


class GenExpertActionPlanning(dspy.Signature):
    """
    【重要】：请使用简体中文进行思考和回答。
    
    您是圆桌对话中的受邀发言人。您的任务是为助手做一个简短备注，以帮助您准备下一轮对话。
    您将得到我们正在讨论的主题、您的专业领域以及对话历史。
    查看对话历史，特别是最近几轮，然后让您的助手用以下方式之一为您准备材料：
    1. Original Question：向其他发言人提出新问题
    2. Further Details：提供额外信息
    3. Information Request：向其他发言人请求信息
    4. Potential Answer：提供可能的解决方案或答案

    【严格禁止】：
    - 不得出现<think>标签或英文思考内容
    - 不得输出内部思考过程
    - 直接给出备注内容

    严格要求：必须使用简体中文回答，严格按照此格式：[贡献类型]：[一句话描述]。例如，Further Details：[描述]
    """

    topic = dspy.InputField(prefix="讨论主题：", format=str)
    expert = dspy.InputField(prefix="您受邀的身份：", format=str)
    summary = dspy.InputField(prefix="讨论历史：\n", format=str)
    last_utterance = dspy.InputField(
        prefix="对话中的最后一次发言：\n", format=str
    )
    resposne = dspy.OutputField(
        prefix="现在给出您的备注。以[Original Question, Further Details, Information Request, Potential Answer]之一开头，并附上一句话描述\n",
        format=str,
    )

class CoStormExpertUtteranceGenerationModule(dspy.Module):
    def __init__(
        self,
        action_planning_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        utterance_polishing_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        answer_question_module: AnswerQuestionModule,
        logging_wrapper: LoggingWrapper,
        callback_handler: BaseCallbackHandler = None,
    ):
        self.action_planning_lm = action_planning_lm
        self.utterance_polishing_lm = utterance_polishing_lm
        self.expert_action = dspy.Predict(GenExpertActionPlanning)
        self.change_style = dspy.Predict(ConvertUtteranceStyle)
        self.answer_question_module = answer_question_module
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def parse_action(self, action):
        action_types = [
            "Original Question",
            "Further Details",
            "Information Request",
            "Potential Answer",
        ]
        for action_type in action_types:
            # 检查英文冒号
            if f"{action_type}:" in action:
                return action_type, trim_output_after_hint(action, f"{action_type}:")
            # 检查中文冒号
            elif f"{action_type}：" in action:
                return action_type, trim_output_after_hint(action, f"{action_type}：")
            # 检查方括号+英文冒号
            elif f"[{action_type}]:" in action:
                return action_type, trim_output_after_hint(action, f"[{action_type}]:")
            # 检查方括号+中文冒号
            elif f"[{action_type}]：" in action:
                return action_type, trim_output_after_hint(action, f"[{action_type}]：")        
        return "Undefined", ""

    def polish_utterance(
        self, conversation_turn: ConversationTurn, last_conv_turn: ConversationTurn
    ):
        # change utterance style
        action_type = conversation_turn.utterance_type
        with self.logging_wrapper.log_event(
            "RoundTableConversationModule.ConvertUtteranceStyle"
        ):
            with dspy.settings.context(
                lm=self.utterance_polishing_lm, show_guidelines=False
            ):
                action_string = (
                    f"{action_type}：{conversation_turn.claim_to_make}"
                )
                if action_type in ["Original Question", "Information Request"]:
                    action_string = f"{action_type}"
                last_expert_utterance_wo_citation, _ = extract_and_remove_citations(
                    last_conv_turn.utterance
                )
                trimmed_last_expert_utterance = keep_first_and_last_paragraph(
                    last_expert_utterance_wo_citation
                )
                utterance = self.change_style(
                    expert=conversation_turn.role,
                    action=action_string,
                    prev=trimmed_last_expert_utterance,
                    content=conversation_turn.raw_utterance,
                ).utterance
                
                # 🔴 清理生成的utterance中的think标签和元认知内容
                utterance = clean_chinese_output(utterance)
                utterance = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(utterance)
                
            conversation_turn.utterance = utterance

    def forward(
        self,
        topic: str,
        current_expert: str,
        conversation_summary: str,
        last_conv_turn: ConversationTurn,
    ):
        last_utterance, _ = extract_and_remove_citations(last_conv_turn.utterance)
        if last_conv_turn.utterance_type in [
            "Original Question",
            "Information Request",
        ]:
            action_type = "Potential Answer"
            action_content = last_utterance
        else:
            with self.logging_wrapper.log_event(
                "CoStormExpertUtteranceGenerationModule: GenExpertActionPlanning"
            ):
                with dspy.settings.context(
                    lm=self.action_planning_lm, show_guidelines=False
                ):
                    action = self.expert_action(
                        topic=topic,
                        expert=current_expert,
                        summary=conversation_summary,
                        last_utterance=last_utterance,
                    ).resposne
                    
                    # 🔴 清理expert_action输出中的think标签
                    action = clean_chinese_output(action)
                    
                action_type, action_content = self.parse_action(action)

        if self.callback_handler is not None:
            self.callback_handler.on_expert_action_planning_end()
        # get response
        conversation_turn = ConversationTurn(
            role=current_expert, raw_utterance="", utterance_type=action_type
        )

        if action_type == "Undefined":
            raise Exception(f"unexpected output: {action}")
        elif action_type in ["Further Details", "Potential Answer"]:
            with self.logging_wrapper.log_event(
                "RoundTableConversationModule: QuestionAnswering"
            ):
                grounded_answer = self.answer_question_module(
                    topic=topic,
                    question=action_content,
                    mode="brief",
                    style="对话性且简洁",
                    callback_handler=self.callback_handler,
                )
            conversation_turn.claim_to_make = action_content
            # 🔴 清理grounded_answer.response中的think标签
            conversation_turn.raw_utterance = clean_chinese_output(grounded_answer.response)
            conversation_turn.queries = grounded_answer.queries
            conversation_turn.raw_retrieved_info = grounded_answer.raw_retrieved_info
            conversation_turn.cited_info = grounded_answer.cited_info
        elif action_type in ["Original Question", "Information Request"]:
            # 🔴 清理action_content中的think标签
            conversation_turn.raw_utterance = clean_chinese_output(action_content)

        return dspy.Prediction(conversation_turn=conversation_turn)
