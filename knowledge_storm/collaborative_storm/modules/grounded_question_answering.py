import dspy
from typing import Union, List

from .callback import BaseCallbackHandler
from .collaborative_storm_utils import (
    trim_output_after_hint,
    format_search_results,
    extract_cited_storm_info,
    separate_citations,
)
from ...logging_wrapper import LoggingWrapper
from ...utils import ArticleTextProcessing
from ...interface import Information
from .chinese_utils import clean_chinese_output


class QuestionToQuery(dspy.Signature):
    """您想使用Google搜索来回答问题或支持一个观点。您会在搜索框中输入什么？
    这个问题是在关于某个主题的圆桌讨论中提出的。这个问题可能聚焦于主题本身，也可能不是。
    请按以下格式写出您将使用的查询：
    - 查询 1
    - 查询 2
    ...
    - 查询 n"""

    topic = dspy.InputField(prefix="主题背景：", format=str)
    question = dspy.InputField(
        prefix="我想收集关于以下内容的信息：", format=str
    )
    queries = dspy.OutputField(prefix="查询：\n", format=str)

class AnswerQuestion(dspy.Signature):
    """您是一位能够有效利用信息的专家。您已经收集了相关信息，现在将使用这些信息来形成回应。
    
    【重要】：请严格使用简体中文进行思考和回答，不要使用英文思考。
    
    让您的回应尽可能具有信息性，并确保每个句子都有收集到的信息支撑。
    如果[收集的信息]与[主题]和[问题]没有直接关系，请基于可用信息提供最相关的答案，并解释任何限制或差距。
    使用[1]、[2]、...、[n]进行行内引用（例如："美国的首都是华盛顿特区[1][3]。"）。
    您不需要在最后包含参考文献或来源部分。写作风格应该是正式的，但要对话性。
    
    【严格禁止】：
    - 不得出现英文内容或<think>标签
    - 不得输出内部思考过程（如"我需要处理用户提供的"、"首先我需要"等）
    - 不得分析任务要求或解释如何组织信息
    - 直接给出回应内容，不要描述您的处理过程
    
    严格要求：必须使用简体中文回答，直接给出有实质内容的回应。
    """

    topic = dspy.InputField(prefix="您正在讨论的主题：", format=str)
    question = dspy.InputField(prefix="您想提供见解的问题：", format=str)
    info = dspy.InputField(prefix="收集的信息：\n", format=str)
    style = dspy.InputField(prefix="您的回应风格应该是：", format=str)
    answer = dspy.OutputField(
        prefix="现在用简体中文给出您的回应。（尽量使用尽可能多的不同来源，不要编造信息。）",
        format=str,
    )

class AnswerQuestionModule(dspy.Module):
    def __init__(
        self,
        retriever: dspy.Retrieve,
        max_search_queries: int,
        question_answering_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        logging_wrapper: LoggingWrapper,
    ):
        super().__init__()
        self.question_answering_lm = question_answering_lm
        self.question_to_query = dspy.Predict(QuestionToQuery)
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.retriever = retriever
        self.max_search_queries = max_search_queries
        self.logging_wrapper = logging_wrapper

    def retrieve_information(self, topic, question):
        # decompose question to queries
        with self.logging_wrapper.log_event(
            f"AnswerQuestionModule.question_to_query ({hash(question)})"
        ):
            with dspy.settings.context(lm=self.question_answering_lm):
                queries = self.question_to_query(topic=topic, question=question).queries
            
            # 🟢 新增：首先清理think标签和英文思考内容
            queries = clean_chinese_output(queries)
            
            # 支持中文和英文hint
            queries = trim_output_after_hint(queries, hint="查询：")
            if "查询：" not in queries:
                queries = trim_output_after_hint(queries, hint="Queries:")
            
            # 改进的查询解析逻辑
            query_lines = queries.split("\n")
            cleaned_queries = []
            
            for line in query_lines:
                # 移除空行
                line = line.strip()
                if not line:
                    continue
                    
                # 移除各种标记符号
                line = line.replace("- ", "").replace("• ", "").replace("* ", "")
                line = line.replace("查询", "").replace("Query", "")
                
                # 移除数字编号
                import re
                line = re.sub(r'^\d+\.?\s*', '', line)
                line = re.sub(r'^[一二三四五六七八九十]+\.?\s*', '', line)
                
                # 移除引号
                line = line.strip('"').strip('"').strip("'").strip()
                
                # 移除think标签或其他HTML标签
                line = re.sub(r'<[^>]+>', '', line)
                
                # 只保留非空且有意义的查询
                if line and len(line) > 2 and not line.startswith('<'):
                    cleaned_queries.append(line)
            
            # 限制查询数量
            queries = cleaned_queries[:self.max_search_queries]
        
        self.logging_wrapper.add_query_count(count=len(queries))
        with self.logging_wrapper.log_event(
            f"AnswerQuestionModule.retriever.retrieve ({hash(question)})"
        ):
            # retrieve information using retriever
            searched_results: List[Information] = self.retriever.retrieve(
                list(set(queries)), exclude_urls=[]
            )
        # update storm information meta to include the question
        for storm_info in searched_results:
            storm_info.meta["question"] = question
        return queries, searched_results

    def forward(
        self,
        topic: str,
        question: str,
        mode: str = "brief",
        style: str = "conversational",
        callback_handler: BaseCallbackHandler = None,
    ):
        """
        Processes a topic and question to generate a response with relevant information and citations.

        Args:
            topic (str): The topic of interest.
            question (str): The specific question related to the topic.
            mode (str, optional): Mode of summarization. 'brief' takes only the first snippet of each Information.
                                'extensive' adds snippets iteratively until the word limit is reached. Defaults to 'brief'.

        Returns:
            dspy.Prediction: An object containing the following:
                - question (str): the question to answer
                - queries (List[str]): List of query strings used for information retrieval.
                - raw_retrieved_info (List[Information]): List of Information instances retrieved.
                - cited_info (Dict[int, Information]): Dictionary of cited Information instances, indexed by their citation number.
                - response (str): The generated response string with inline citations.
        """
        # retrieve information
        if callback_handler is not None:
            callback_handler.on_expert_information_collection_start()
        queries, searched_results = self.retrieve_information(
            topic=topic, question=question
        )
        if callback_handler is not None:
            callback_handler.on_expert_information_collection_end(searched_results)
        # format information string for answer generation
        info_text, index_to_information_mapping = format_search_results(
            searched_results, mode=mode
        )
        answer = "Sorry, there is insufficient information to answer the question."
        # generate answer to the question
        if info_text:
            with self.logging_wrapper.log_event(
                f"AnswerQuestionModule.answer_question ({hash(question)})"
            ):
                with dspy.settings.context(
                    lm=self.question_answering_lm, show_guidelines=False
                ):
                    answer = self.answer_question(
                        topic=topic, question=question, info=info_text, style=style
                    ).answer
                    # 🟢 应用中文后处理
                    answer = clean_chinese_output(answer)
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                    answer = trim_output_after_hint(
                        answer,
                        hint="Now give your response. (Try to use as many different sources as possible and do not hallucinate.)",
                    )
                    # enforce single citation index bracket. [1, 2] -> [1][2]
                    answer = separate_citations(answer)
                    if callback_handler is not None:
                        callback_handler.on_expert_utterance_generation_end()
        # construct cited search result
        cited_searched_results = extract_cited_storm_info(
            response=answer, index_to_storm_info=index_to_information_mapping
        )

        return dspy.Prediction(
            question=question,
            queries=queries,
            raw_retrieved_info=searched_results,
            cited_info=cited_searched_results,
            response=answer,
        )
