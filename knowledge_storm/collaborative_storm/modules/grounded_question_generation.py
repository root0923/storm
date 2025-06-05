"""
This module handles question generation within the Co-STORM framework, specifically designed to support the Moderator role.

The Moderator generates insightful, thought-provoking questions that introduce new directions into the conversation. 
By leveraging uncited or unused snippets of information retrieved during the discussion, the Moderator ensures the conversation remains dynamic and avoids repetitive or overly niche topics.

For more detailed information, refer to Section 3.5 of the Co-STORM paper: https://www.arxiv.org/pdf/2408.15232.
"""

import dspy
from typing import List, Union

from .collaborative_storm_utils import (
    format_search_results,
    extract_and_remove_citations,
    keep_first_and_last_paragraph,
    extract_cited_storm_info,
)
from ...dataclass import ConversationTurn, KnowledgeBase
from ...interface import Information
from .chinese_utils import clean_chinese_output
import re


class KnowledgeBaseSummmary(dspy.Signature):
    """您的任务是对圆桌对话中已讨论的内容给出简要总结。内容按主题分层组织成层次化章节。
    您将看到这些章节，其中"#"表示章节级别。
    """

    topic = dspy.InputField(prefix="主题：", format=str)
    structure = dspy.InputField(prefix="树形结构：\n", format=str)
    output = dspy.OutputField(prefix="现在给出简要总结：\n", format=str)


class ConvertUtteranceStyle(dspy.Signature):
    """
    您是圆桌对话中的受邀发言人。
    您的任务是让问题或回应更具对话性和吸引力，以促进对话的流畅进行。
    注意这是正在进行的对话，所以无需开场白和结束语。提供前一位发言人的发言仅为了让对话更自然。
    注意不要编造信息，并保持引用索引如[1]不变。
    """

    expert = dspy.InputField(prefix="您受邀的身份：", format=str)
    action = dspy.InputField(
        prefix="您想通过以下方式为对话做出贡献：", format=str
    )
    prev = dspy.InputField(prefix="前一位发言人说：", format=str)
    content = dspy.InputField(
        prefix="您想说的问题或回应：", format=str
    )
    utterance = dspy.OutputField(
        prefix="您的发言（尽可能保留信息并附上引用，在不丢失信息的前提下偏向简短回答）：",
        format=str,
    )



class GroundedQuestionGeneration(dspy.Signature):
    """您的任务是为圆桌对话找到下一个讨论焦点。您将得到之前的对话总结和一些可能帮助您发现新讨论焦点的信息。
    注意新的讨论焦点应该为讨论带来新的角度和观点，避免重复。新的讨论焦点应该基于可用信息，并推动当前讨论的边界以进行更广泛的探索。
    新的讨论焦点应该与对话中的最后一次发言有自然的连接。
    使用[1][2]等行内引用来支撑您的问题。
    """

    topic = dspy.InputField(prefix="主题：", format=str)
    summary = dspy.InputField(prefix="讨论历史：\n", format=str)
    information = dspy.InputField(prefix="可用信息：\n", format=str)
    last_utterance = dspy.InputField(
        prefix="对话中的最后一次发言：\n", format=str
    )
    output = dspy.OutputField(
        prefix="现在用一句话问题的格式给出下一个讨论焦点：\n",
        format=str,
    )


class GroundedQuestionGenerationModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.gen_focus = dspy.Predict(GroundedQuestionGeneration)
        self.polish_style = dspy.Predict(ConvertUtteranceStyle)
        self.gen_summary = dspy.Predict(KnowledgeBaseSummmary)

    def forward(
        self,
        topic: str,
        knowledge_base: KnowledgeBase,
        last_conv_turn: ConversationTurn,
        unused_snippets: List[Information],
    ):
        information, index_to_information_mapping = format_search_results(
            unused_snippets, info_max_num_words=1000
        )
        summary = knowledge_base.get_knowledge_base_summary()
        last_utterance, _ = extract_and_remove_citations(last_conv_turn.utterance)
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            raw_utterance = self.gen_focus(
                topic=topic,
                summary=summary,
                information=information,
                last_utterance=last_utterance,
            ).output
            
            # 🔴 清理raw_utterance中的think标签
            raw_utterance = clean_chinese_output(raw_utterance, role_context="Moderator")
            
            if '**' in raw_utterance:
                matches = re.findall(r'\*\*(.*?)\*\*', raw_utterance, flags=re.DOTALL)
                raw_utterance = ' '.join(m.strip() for m in matches)
            utterance = self.polish_style(
                expert="圆桌对话主持人",
                action="从上一次发言自然过渡到新问题。",
                prev=keep_first_and_last_paragraph(last_utterance),
                content=raw_utterance,
            ).utterance
            
            # 🔴 清理polished utterance中的think标签
            utterance = clean_chinese_output(utterance, role_context="Moderator")
            if '**' in raw_utterance:
                matches = re.findall(r'\*\*(.*?)\*\*', utterance, flags=re.DOTALL)
                utterance = ' '.join(m.strip() for m in matches)
            
            cited_searched_results = extract_cited_storm_info(
                response=utterance, index_to_storm_info=index_to_information_mapping
            )
            return dspy.Prediction(
                raw_utterance=raw_utterance,
                utterance=utterance,
                cited_info=cited_searched_results,
            )
