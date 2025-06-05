import dspy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set, Union, List
import re

from .callback import BaseCallbackHandler
from .collaborative_storm_utils import trim_output_after_hint
from ...dataclass import KnowledgeBase, KnowledgeNode
from ...logging_wrapper import LoggingWrapper
from ...utils import ArticleTextProcessing
from ...interface import Information

class ArticleGenerationModule(dspy.Module):
    """Use the information collected from the information-seeking conversation to write a section."""

    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
    ):
        super().__init__()
        self.write_section = dspy.Predict(WriteSection)
        self.engine = engine

    def _get_cited_information_string(
        self,
        all_citation_index: Set[int],
        knowledge_base: KnowledgeBase,
        max_words: int = 4000,
    ):
        information = []
        cur_word_count = 0
        for index in sorted(list(all_citation_index)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            snippet = info.snippets[0]
            info_text = f"[{index}]: {snippet} (Question: {info.meta['question']}. Query: {info.meta['query']})"
            cur_snippet_length = len(info_text.split())
            if cur_snippet_length + cur_word_count > max_words:
                break
            cur_word_count += cur_snippet_length
            information.append(info_text)
        return "\n".join(information)

    def gen_section(
        self, topic: str, node: KnowledgeNode, knowledge_base: KnowledgeBase
    ):
        if node is None or len(node.content) == 0:
            return ""
        if (
            node.synthesize_output is not None
            and node.synthesize_output
            and not node.need_regenerate_synthesize_output
        ):
            return node.synthesize_output
        all_citation_index = node.collect_all_content()
        information = self._get_cited_information_string(
            all_citation_index=all_citation_index, knowledge_base=knowledge_base
        )
        with dspy.settings.context(lm=self.engine):
            synthesize_output = self.write_section(
                topic=topic, info=information, section=node.name
            ).output
            # 使用中文文本清理函数
            text = re.sub(r'<think>.*?</think>', '', synthesize_output, flags=re.DOTALL)
            text = re.sub(r'<think>', '', text)
            synthesize_output = re.sub(r'</think>', '', text)
            synthesize_output = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(synthesize_output)
        node.synthesize_output = synthesize_output
        node.need_regenerate_synthesize_output = False
        return node.synthesize_output

    def forward(self, knowledge_base: KnowledgeBase):
        all_nodes = knowledge_base.collect_all_nodes()
        node_to_paragraph = {}

        # Define a function to generate paragraphs for nodes
        def _node_generate_paragraph(node):
            node_gen_paragraph = self.gen_section(
                topic=knowledge_base.topic, node=node, knowledge_base=knowledge_base
            )
            lines = node_gen_paragraph.split("\n")
            if lines[0].strip().replace("*", "").replace("#", "") == node.name:
                lines = lines[1:]
            node_gen_paragraph = "\n".join(lines)
            path = " -> ".join(node.get_path_from_root())
            return path, node_gen_paragraph

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_node = {
                executor.submit(_node_generate_paragraph, node): node
                for node in all_nodes
            }

            # Collect the results as they complete
            for future in as_completed(future_to_node):
                path, node_gen_paragraph = future.result()
                node_to_paragraph[path] = node_gen_paragraph

        def helper(cur_root, level):
            to_return = []
            if cur_root is not None:
                hash_tag = "#" * level + " "
                cur_path = " -> ".join(cur_root.get_path_from_root())
                node_gen_paragraph = node_to_paragraph[cur_path]
                to_return.append(f"{hash_tag}{cur_root.name}\n{node_gen_paragraph}")
                for child in cur_root.children:
                    to_return.extend(helper(child, level + 1))
            return to_return

        to_return = []
        for child in knowledge_base.root.children:
            to_return.extend(helper(child, level=1))

        return "\n".join(to_return)


class WriteSection(dspy.Signature):
    """基于收集的信息撰写维基百科风格的中文章节内容。您将得到主题、要撰写的章节名称和相关信息。
    每条信息都会提供原始内容以及导致该信息的问题和查询。
    
    重要格式要求：
    1. 请直接撰写章节的正文内容，不要包含章节标题（系统会自动添加标题）
    2. 使用[1]、[2]、...、[n]进行行内引用（例如："美国的首都是华盛顿特区[1][3]。"）
    3. 请用简体中文撰写，内容要准确、流畅、符合中文表达习惯
    4. 不要在文章末尾包含参考文献或来源部分
    5. 不要以"#"开头写标题，直接写正文内容即可
    """

    info = dspy.InputField(prefix="收集的信息：\n", format=str)
    topic = dspy.InputField(prefix="页面主题：", format=str)
    section = dspy.InputField(prefix="需要撰写的章节：", format=str)
    output = dspy.OutputField(
        prefix="请用简体中文撰写这个章节的正文内容，包含适当的行内引用（注意：不要包含标题，直接写正文内容即可）：\n",
        format=str,
    )