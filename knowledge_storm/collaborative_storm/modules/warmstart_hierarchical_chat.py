"""
Warm starts the Co-STORM system by conducting a background information search to establish a shared conceptual space with the user.
 
This stage functions as a mini-STORM, where multiple LLM agents are spawned with different perspectives to engage in multi-round conversations. 
The knowledge base (represented as a mind map) is initialized using the information gathered during these exchanges.

Additionally, the system generates a first draft of the report, which is then used to create a concise and engaging conversation. 
The synthesized conversation is presented to the user to help them quickly catch up on the system's current knowledge about the topic.
"""

import dspy
import concurrent.futures
from threading import Lock
from typing import List, Optional, Union, TYPE_CHECKING

from .callback import BaseCallbackHandler
from .collaborative_storm_utils import _get_answer_question_module_instance
from .expert_generation import GenerateExpertModule
from .grounded_question_answering import AnswerQuestionModule
from .chinese_utils import clean_chinese_output
from ...dataclass import ConversationTurn, KnowledgeBase, KnowledgeNode
from ...interface import LMConfigs
from ...logging_wrapper import LoggingWrapper
from ...storm_wiki.modules.outline_generation import WritePageOutline
from ...utils import ArticleTextProcessing as AP
import re


if TYPE_CHECKING:
    from ..engine import RunnerArgument


class WarmStartModerator(dspy.Signature):
    """
    您是圆桌讨论的主持人。目标是与多位专家聊天，讨论主题的事实和背景，让观众熟悉该主题。
    您将看到主题、您已经问过的问题历史，以及您正在讨论的当前专家。
    基于这些信息，为当前专家生成下一个问题以推进讨论。

    输出应该只包含给当前专家的下一个问题。不要包含任何其他信息或前言。
    """

    topic = dspy.InputField(prefix="圆桌讨论的主题：", format=str)
    history = dspy.InputField(
        prefix="您已经交流过的专家：", format=str
    )
    current_expert = dspy.InputField(prefix="您正在交流的专家：", format=str)
    question = dspy.OutputField(
        prefix="给您正在交流的专家的下一个问题：", format=str
    )


class SectionToConvTranscript(dspy.Signature):
    """
    给您一个关于特定主题的简要报告章节。您的任务是将这个章节转换为圆桌对话的引人入胜的开场讨论。
    目标是帮助参与者和观众快速理解关键信息。
    问题和答案都应该用面向观众的圆桌讨论语调。

    具体而言，您需要：
    1. 生成一个引人入胜的问题，利用章节名称和主题来开启内容讨论。
    2. 提供一个简洁而引人入胜的答案（包含原文的所有行内引用），源于该章节，作为指引，避免过多细节。
    
    【严格禁止】：
    - 不得出现<think>标签或英文思考内容
    - 不得输出内部思考过程
    - 直接给出问题和答案内容
    """

    topic = dspy.InputField(prefix="主题：", format=str)
    section_name = dspy.InputField(prefix="章节名称：", format=str)
    section_content = dspy.InputField(prefix="章节内容：", format=str)
    question = dspy.OutputField(prefix="现在只给出引人入胜的问题。\n问题：")
    answer = dspy.OutputField(
        prefix="现在只给出引人入胜的答案，包含原文的所有行内引用。\n答案："
    )


class ReportToConversation(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.section_to_conv_transcript = dspy.Predict(SectionToConvTranscript)

    def forward(self, knowledge_base: KnowledgeBase):
        def process_node(node, topic):
            with dspy.settings.context(lm=self.engine, show_guidelines=False):
                output = self.section_to_conv_transcript(
                    topic=topic,
                    section_name=node.get_path_from_root(),
                    section_content=node.synthesize_output,
                )
                # 🔴 立即清理LLM输出中的think标签
                question = clean_chinese_output(output.question.replace("问题：", "").strip())
                answer = clean_chinese_output(output.answer.replace("答案：", "").strip())
                return question, answer

        conversations = []
        nodes = knowledge_base.collect_all_nodes()
        nodes = [node for node in nodes if node.name != "root" and node.content]
        topic = knowledge_base.topic

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_node = {
                executor.submit(process_node, node, topic): node for node in nodes
            }
            for future in concurrent.futures.as_completed(future_to_node):
                node = future_to_node[future]
                question, answer = future.result()
                
                conversations.append(
                    ConversationTurn(
                        role="背景讨论主持人",
                        raw_utterance=question,
                        utterance_type="Original Question",
                        utterance=question,
                        cited_info=[
                            knowledge_base.info_uuid_to_info_dict[idx]
                            for idx in AP.parse_citation_indices(question)
                        ],
                    )
                )
                conversations.append(
                    ConversationTurn(
                        role="背景讨论专家",
                        raw_utterance=answer,
                        utterance_type="Potential Answer",
                        utterance=answer,
                        cited_info=[
                            knowledge_base.info_uuid_to_info_dict[idx]
                            for idx in AP.parse_citation_indices(answer)
                        ],
                    )
                )
        return conversations


class WarmStartConversation(dspy.Module):
    def __init__(
        self,
        question_asking_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        generate_expert_module: GenerateExpertModule,
        answer_question_module: AnswerQuestionModule,
        logging_wrapper: LoggingWrapper,
        max_num_experts: int = 3,
        max_turn_per_experts: int = 2,
        max_thread: int = 3,
        callback_handler: BaseCallbackHandler = None,
    ):
        self.ask_question = dspy.Predict(WarmStartModerator)
        self.max_num_experts = max_num_experts
        self.max_turn_per_experts = max_turn_per_experts
        self.question_asking_lm = question_asking_lm
        self.answer_question_module = answer_question_module
        self.max_thread = max_thread
        self.generate_experts_module = generate_expert_module
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def format_dialogue_question_history_string(
        self, conversation_history: List[ConversationTurn]
    ):
        output = []
        for idx, turn in enumerate(conversation_history):
            info = turn.claim_to_make if turn.claim_to_make else turn.utterance
            output.append(f"{idx + 1}: {info}")
        return "\n".join(output)

    def generate_warmstart_experts(self, topic: str):
        background_seeking_dialogue = self.get_background_info(topic=topic)
        background_info = background_seeking_dialogue.utterance
        gen_expert_output = self.generate_experts_module(
            topic=topic,
            background_info=background_info,
            num_experts=self.max_num_experts,
        )
        return gen_expert_output.experts, background_seeking_dialogue

    def get_background_info(self, topic: str):
        question = f"关于{topic}的背景信息"
    
        answer = self.answer_question_module(
            topic=topic, question=question, mode="extensive", style="对话性"
        )

        # 🟢 清理think标签和英文思考内容
        cleaned_response = clean_chinese_output(answer.response)

        return ConversationTurn(
            role="背景信息研究员",
            raw_utterance=cleaned_response,
            utterance_type="Questioning",
            claim_to_make=question,
            queries=answer.queries,
            raw_retrieved_info=answer.raw_retrieved_info,
            cited_info=answer.cited_info,
        )

    def forward(self, topic: str):
        with self.logging_wrapper.log_event(
            "warm start, perspective guided QA: identify experts"
        ):
            # do background research, generate some experts
            experts, background_seeking_dialogue = self.generate_warmstart_experts(
                topic=topic
            )
        # init list to store the dialogue history
        conversation_history: List[ConversationTurn] = []
        lock = Lock()

        # hierarchical chat: chat with one expert. Generate question, get answer
        def process_expert(expert):
            # 支持中英文冒号分割
            colon_pattern = r'[：:]'
            if re.search(colon_pattern, expert):
                parts = re.split(colon_pattern, expert, 1)
                expert_name = parts[0].strip()
                expert_description = parts[1].strip() if len(parts) > 1 else ""
            else:
                expert_name = expert.strip()
                expert_description = ""
            for idx in range(self.max_turn_per_experts):
                with self.logging_wrapper.log_event(
                    f"warm start, perspective guided QA: expert {expert_name}; turn {idx + 1}"
                ):
                    try:
                        with lock:
                            history = self.format_dialogue_question_history_string(
                                conversation_history
                            )
                        with dspy.settings.context(lm=self.question_asking_lm):
                            question = self.ask_question(
                                topic=topic, history=history, current_expert=expert
                            ).question
                            # 🟢 清理think标签和英文思考内容
                            question = clean_chinese_output(question)
                        answer = self.answer_question_module(
                            topic=topic,
                            question=question,
                            mode="brief",
                            style="对话性",
                        )
                        
                        # 🟢 清理think标签和英文思考内容
                        cleaned_response = clean_chinese_output(answer.response)
                        
                        conversation_turn = ConversationTurn(
                            role=expert,
                            claim_to_make=question,
                            raw_utterance=cleaned_response,
                            utterance_type="Support",
                            queries=answer.queries,
                            raw_retrieved_info=answer.raw_retrieved_info,
                            cited_info=answer.cited_info,
                        )
                        if self.callback_handler is not None:
                            self.callback_handler.on_warmstart_update(
                                message="\n".join(
                                    [
                                        f"Finish browsing {url}"
                                        for url in [
                                            i.url for i in answer.raw_retrieved_info
                                        ]
                                    ]
                                )
                            )
                        with lock:
                            conversation_history.append(conversation_turn)
                    except Exception as e:
                        print(f"Error processing expert {expert}: {e}")

        # multi-thread conversation
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_thread
        ) as executor:
            futures = [
                executor.submit(process_expert, expert)
                for expert in experts[: min(len(experts), self.max_num_experts)]
            ]
            concurrent.futures.wait(futures)

        conversation_history = [background_seeking_dialogue] + conversation_history

        return dspy.Prediction(
            conversation_history=conversation_history, experts=experts
        )


class GenerateWarmStartOutline(dspy.Signature):
    """根据圆桌讨论生成类维基百科报告的大纲。您将看到对话中的讨论要点和相应查询。
    您将获得一个草案大纲，可以从中获得一些灵感。不要包含给定讨论历史中未提及的章节。
    
    重要说明：
    - 讨论历史是以"问题："开头的对话记录，不是章节标题
    - 您需要根据这些讨论内容来生成简洁的章节标题
    - 章节标题应该简洁明了，如"概述"、"技术架构"、"应用场景"等
    
    格式要求：
    1. 使用"#"表示章节标题，"##"表示子章节标题，"###"表示子子章节标题，依此类推。
    2. 不要包含任何附加信息或解释性文字。
    3. 从大纲中排除主题名称。
    4. 章节标题要简洁，避免冗长的描述性语句。
    大纲的组织应采用维基百科风格。
    """

    topic = dspy.InputField(prefix="讨论的主题：", format=str)
    draft = dspy.InputField(prefix="您可以参考的草案大纲：", format=str)
    conv = dspy.InputField(prefix="讨论历史：\n", format=str)
    outline = dspy.OutputField(
        prefix='请生成简洁的维基百科风格大纲（使用"# 标题"表示章节标题，"## 标题"表示子章节标题...）：\n',
        format=str,
    )


class GenerateWarmStartOutlineModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.gen_outline = dspy.Predict(GenerateWarmStartOutline)
        self.draft_outline = dspy.Predict(WritePageOutline)

    def extract_questions_and_queries(self, conv: List[ConversationTurn]):
        context = []
        for turn in conv:
            focus = turn.claim_to_make
            queries = turn.queries
            queries_string = "\n".join(
                f"  - {query}" for query in queries  # 使用更清晰的格式
            )
            string = f"问题：{focus}\n相关查询：\n{queries_string}"  # 简化格式
            context.append(string)
        return "\n\n".join(context)  # 用双换行分隔不同的问题

    def get_draft_outline(self, topic: str):
        with dspy.settings.context(lm=self.engine):
            return self.draft_outline(topic=topic).outline

    def forward(self, topic: str, conv: List[ConversationTurn]):
        discussion_history = self.extract_questions_and_queries(conv)
        draft_outline = self.get_draft_outline(topic=topic)
        # 🟢 清理think标签和英文思考内容
        draft_outline = clean_chinese_output(draft_outline)
        with dspy.settings.context(lm=self.engine):
            outline = self.gen_outline(
                topic=topic, draft=draft_outline, conv=discussion_history
            ).outline
            outline = AP.clean_up_outline(outline)
        return dspy.Prediction(outline=outline, draft_outline=draft_outline)


class WarmStartModule:
    def __init__(
        self,
        lm_config: LMConfigs,
        runner_argument: "RunnerArgument",
        logging_wrapper: LoggingWrapper,
        rm: Optional[dspy.Retrieve] = None,
        callback_handler: BaseCallbackHandler = None,
    ):
        generate_expert_module = GenerateExpertModule(
            engine=lm_config.discourse_manage_lm
        )
        self.warmstart_conv = WarmStartConversation(
            question_asking_lm=lm_config.question_asking_lm,
            generate_expert_module=generate_expert_module,
            answer_question_module=_get_answer_question_module_instance(
                lm_config=lm_config,
                runner_argument=runner_argument,
                logging_wrapper=logging_wrapper,
                rm=rm,
                deepsearcher_api_url=runner_argument.deepsearcher_api_url,
            ),
            max_num_experts=runner_argument.warmstart_max_num_experts,
            max_turn_per_experts=runner_argument.warmstart_max_turn_per_experts,
            max_thread=runner_argument.warmstart_max_thread,
            logging_wrapper=logging_wrapper,
            callback_handler=callback_handler,
        )
        self.warmstart_outline_gen_module = GenerateWarmStartOutlineModule(
            engine=lm_config.warmstart_outline_gen_lm
        )
        self.report_to_conversation = ReportToConversation(lm_config.knowledge_base_lm)
        self.logging_wrapper = logging_wrapper
        self.callback_handler = callback_handler

    def initiate_warm_start(self, topic: str, knowledge_base: KnowledgeBase):
        """
        Initiates a warm start process for the given topic by generating a warm start conversation and inserting the
        resulting information into a knowledge base.

        Args:
            topic (str): The topic for which to initiate the warm start process.

        Returns:
            Tuple[List[ConversationTurn], List[str], KnowledgeBase]:
                - A list of ConversationTurn instances representing the conversation history.
                - A list of strings representing the experts involved in the conversation.
                - A KnowledgeBase instance containing the organized information.
        """
        warm_start_conversation_history: List[ConversationTurn] = []
        warm_start_experts = None
        # get warm start conversations
        with self.logging_wrapper.log_event("warm start: perspective guided QA"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    message="Start getting familiar with the topic by chatting with multiple LLM experts (Step 1 / 4)"
                )
            warm_start_result = self.warmstart_conv(topic=topic)
            warm_start_conversation_history = warm_start_result.conversation_history
            warm_start_experts = warm_start_result.experts

        # get warm start conv outline
        with self.logging_wrapper.log_event("warm start: outline generation"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    "Organizing collected information (Step 2 / 4)"
                )
            warm_start_outline_output = self.warmstart_outline_gen_module(
                topic=topic, conv=warm_start_conversation_history
            )
        # init knowledge base
        with self.logging_wrapper.log_event("warm start: insert into knowledge base"):
            if self.callback_handler is not None:
                self.callback_handler.on_warmstart_update(
                    "Inserting collected information into knowledge base (Step 3 / 4)"
                )
            knowledge_base.insert_from_outline_string(
                outline_string=warm_start_outline_output.outline
            )
            # insert information to knowledge base
            for turn in warm_start_conversation_history:
                knowledge_base.update_from_conv_turn(
                    conv_turn=turn, allow_create_new_node=False
                )
        # knowledge base to report
        if self.callback_handler is not None:
            self.callback_handler.on_warmstart_update(
                "Synthesizing background information discussion utterances (Step 4 / 4)"
            )
        knowledge_base.to_report()

        # generate engaging conversations
        engaging_conversations = self.report_to_conversation(knowledge_base)
        return (
            warm_start_conversation_history,
            engaging_conversations,
            warm_start_experts,
        )
