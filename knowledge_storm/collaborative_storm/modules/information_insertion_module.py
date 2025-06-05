import dspy
import numpy as np
import re
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Union, Dict, Optional

from .collaborative_storm_utils import trim_output_after_hint
from ...dataclass import KnowledgeNode, KnowledgeBase
from ...encoder import Encoder
from ...interface import Information
from .chinese_utils import clean_chinese_output


class InsertInformation(dspy.Signature):
    """您的任务是将给定信息插入到知识库中。知识库是一个基于树形结构的数据结构，用于组织收集的信息。每个知识节点包含来自主题相似问题或意图的信息。
    为了决定信息的最佳放置位置，您将在这个基于树的数据结构中逐层导航。
    您将看到导致此信息的问题和查询，以及树形结构。

    输出应严格遵循下面提供的选项之一，不包含其他信息。
    - 'insert'：将信息放置在当前节点下。
    - 'step: [子节点名称]'：进入指定的子节点。
    - 'create: [新子节点名称]'：创建新的子节点并在其下插入信息。

    示例输出：
    - insert
    - step: 节点2
    - create: 节点3
    """

    intent = dspy.InputField(
        prefix="导致此信息的问题和查询：", format=str
    )
    structure = dspy.InputField(prefix="树形结构：\n", format=str)
    choice = dspy.OutputField(prefix="选择：\n", format=str)


class InsertInformationCandidateChoice(dspy.Signature):
    """您的任务是将给定信息插入到知识库中。知识库是一个基于树形结构的数据结构，用于组织收集的信息。每个知识节点包含来自主题相似问题或意图的信息。
    您将看到导致此信息的问题和查询，以及候选放置选项。在这些选项中，-> 表示父子关系。请注意，合理的选择可能不在这些选项中。

    如果存在合理的选择，输出"最佳放置：[选择索引]"；否则，输出"无合理选择"。
    """

    intent = dspy.InputField(
        prefix="导致此信息的问题和查询：", format=str
    )
    choices = dspy.InputField(prefix="候选放置位置：\n", format=str)
    decision = dspy.OutputField(prefix="决定：\n", format=str)

class InsertInformationModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], encoder: Encoder):
        self.engine = engine
        self.encoder = encoder
        self.insert_info = dspy.ChainOfThought(InsertInformation)
        self.candidate_choosing = dspy.Predict(InsertInformationCandidateChoice)

    def _construct_intent(self, question: str, query: str):
        intent = ""
        if query == "Not applicable":
            return question
        if question:
            intent += f"Question: {question}\n"
        if query:
            intent += f"Query: {query}\n"
        if not intent:
            intent = "Not available."
        return intent

    def _get_navigation_choice(
        self, knowledge_node: KnowledgeNode, question: str, query: str
    ):
        # 构建信息意图
        intent = self._construct_intent(question, query)
        # 构建当前知识库结构
        structure = f"当前节点：{knowledge_node.name}\n"
        child_names = ", ".join(knowledge_node.get_children_names())
        if child_names:
            structure += f"子节点：{child_names}"
        navigated_path = " -> ".join(knowledge_node.get_path_from_root())
        structure += f"您已导航的路径：{navigated_path}"

        # 获取预测动作
        with dspy.settings.context(lm=self.engine):
            predicted_action = self.insert_info(
                intent=intent, structure=structure
            ).choice

        predicted_action = re.sub(r'Reasoning:.*', '', predicted_action, flags=re.DOTALL)
        if predicted_action.startswith("insert") or predicted_action.startswith("step") or predicted_action.startswith("create"):
            cleaned_predicted_action = predicted_action
        else:
            text = re.sub(r'<think>.*?</think>', '', predicted_action, flags=re.DOTALL)
            text = re.sub(r'<think>', '', text)
            cleaned_predicted_action = re.sub(r'</think>', '', text)
        
        if cleaned_predicted_action.startswith("选择"):
            cleaned_predicted_action = re.sub(r'选择：', '', cleaned_predicted_action)
        cleaned_predicted_action = cleaned_predicted_action.strip("-").strip()
        
        if cleaned_predicted_action.startswith("insert") or cleaned_predicted_action.startswith("- insert"):
            return "insert", ""
        elif cleaned_predicted_action.startswith("step") or cleaned_predicted_action.startswith("- step"):
            node_name = trim_output_after_hint(cleaned_predicted_action, "step:")
            return "step", node_name
        elif cleaned_predicted_action.startswith("create") or cleaned_predicted_action.startswith("- create"):
            node_name = trim_output_after_hint(cleaned_predicted_action, "create:")
            return "create", node_name
        
        raise Exception(
            f"知识导航中的未定义预测动作。{predicted_action}/{cleaned_predicted_action}"
        )


    def layer_by_layer_navigation_placement(
        self,
        knowledge_base: KnowledgeBase,
        question: str,
        query: str,
        allow_create_new_node: bool = False,
        root: Optional[KnowledgeNode] = None,
    ):
        current_node: KnowledgeNode = knowledge_base.root if root is None else root

        while True:
            action_type, node_name = self._get_navigation_choice(
                knowledge_node=current_node, question=question, query=query
            )
            if action_type == "insert":
                return dspy.Prediction(
                    information_placement=" -> ".join(
                        current_node.get_path_from_root(root)
                    ),
                    note="None",
                )
            elif action_type == "step":
                for child in current_node.children:
                    if child.name == node_name:
                        current_node = child
                        break
                else:
                    raise ValueError(f"Child node with name {node_name} not found.")
            elif action_type == "create":
                placement_path = current_node.get_path_from_root(root)
                if allow_create_new_node:
                    placement_path.append(node_name)
                    note = f"create new node: {{{node_name}}} under {{{current_node.name}}}"
                else:
                    note = f"attempt to create new node: {{{node_name}}} under {{{current_node.name}}}"
                return dspy.Prediction(
                    information_placement=" -> ".join(placement_path), note=note
                )
            else:
                raise ValueError(f"Unknown action type: {action_type}")

    def _get_sorted_embed_sim_section(
        self,
        encoded_outline: np.ndarray,
        outlines: List[str],
        question: str,
        query: str,
    ):
        if encoded_outline is not None and encoded_outline.size > 0:
            encoded_query = self.encoder.encode(f"{question}, {query}")
            sim = cosine_similarity([encoded_query], encoded_outline)[0]
            sorted_indices = np.argsort(sim)
            sorted_outlines = np.array(outlines)[sorted_indices[::-1]]
            return sorted_outlines
        else:
            return outlines

    def _parse_selected_index(self, string: str):
        # 支持中文输出解析
        patterns = [
            r"最佳放置[：:].*?(\d+)",
            r"Best placement[：:].*?\[(\d+)\]",
            r"\[(\d+)\]",
            r"(\d+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, string)
            if match:
                return int(match.group(1))
        
        # 处理"无合理选择"的情况
        if any(keyword in string for keyword in ["无合理选择", "No reasonable choice"]):
            return None
            
        try:
            return int(string.strip())
        except:
            pass
        return None

    def choose_candidate_from_embedding_ranking(
        self,
        question: str,
        query: str,
        encoded_outlines: np.ndarray,
        outlines: List[str],
        top_N_candidates: int = 5,
    ):
        sorted_candidates = self._get_sorted_embed_sim_section(
            encoded_outlines, outlines, question, query
        )
        considered_candidates = sorted_candidates[
            : min(len(sorted_candidates), top_N_candidates)
        ]
        choices_string = "\n".join(
            [
                f"{idx + 1}: {candidate}"
                for idx, candidate in enumerate(considered_candidates)
            ]
        )
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            decision = self.candidate_choosing(
                intent=self._construct_intent(question=question, query=query),
                choices=choices_string,
            ).decision
            text = re.sub(r'<think>.*?</think>', '', decision, flags=re.DOTALL)
            text = re.sub(r'<think>', '', text)
            decision = re.sub(r'</think>', '', text)
            if "最佳放置：" in decision:
                decision = trim_output_after_hint(decision, hint="最佳放置：")
                selected_index = self._parse_selected_index(decision)
                if selected_index is not None:
                    selected_index = selected_index - 1
                    if selected_index < len(sorted_candidates) and selected_index >= 0:
                        return dspy.Prediction(
                            information_placement=sorted_candidates[selected_index],
                            note=f"Choosing from:\n{considered_candidates}",
                        )
            return None

    def _info_list_to_intent_mapping(self, information_list: List[Information]):
        intent_to_placement_dict = {}
        for info in information_list:
            intent = (info.meta.get("question", ""), info.meta.get("query", ""))
            if intent not in intent_to_placement_dict:
                intent_to_placement_dict[intent] = None
        return intent_to_placement_dict

    def forward(
        self,
        knowledge_base: KnowledgeBase,
        information: Union[Information, List[Information]],
        allow_create_new_node: bool = False,
        max_thread: int = 5,
        insert_root: Optional[KnowledgeNode] = None,
        skip_candidate_from_embedding: bool = False,
    ):
        if not isinstance(information, List):
            information = [information]
        intent_to_placement_dict: Dict = self._info_list_to_intent_mapping(
            information_list=information
        )

        # process one intent
        def process_intent(question: str, query: str):
            candidate_placement = None
            try:
                if not skip_candidate_from_embedding:
                    candidate_placement = self.choose_candidate_from_embedding_ranking(
                        question=question,
                        query=query,
                        encoded_outlines=encoded_outlines,
                        outlines=outlines,
                        top_N_candidates=8,
                    )
                if candidate_placement is None:
                    candidate_placement = self.layer_by_layer_navigation_placement(
                        knowledge_base=knowledge_base,
                        question=question,
                        query=query,
                        allow_create_new_node=allow_create_new_node,
                        root=insert_root,
                    )
                return (question, query), candidate_placement
            except Exception as e:
                print(traceback.format_exc())
                return (question, query), None

        def insert_info_to_kb(info, placement_prediction):
            if placement_prediction is not None:
                missing_node_handling = (
                    "raise error" if not allow_create_new_node else "create"
                )
                knowledge_base.insert_information(
                    path=placement_prediction.information_placement,
                    information=info,
                    missing_node_handling=missing_node_handling,
                    root=insert_root,
                )

        (
            encoded_outlines,
            outlines,
        ) = knowledge_base.get_knowledge_base_structure_embedding(root=insert_root)
        to_return = []
        if not allow_create_new_node:
            # use multi thread as knowledge base structure does not change
            with ThreadPoolExecutor(max_workers=max_thread) as executor:
                futures = {
                    executor.submit(process_intent, question, query): (question, query)
                    for (question, query) in intent_to_placement_dict
                }

                for future in as_completed(futures):
                    (question, query), candidate_placement = future.result()
                    intent_to_placement_dict[(question, query)] = candidate_placement
            # back mapping placement to each information
            for info in information:
                intent = (info.meta.get("question", ""), info.meta.get("query", ""))
                placement_prediction = intent_to_placement_dict.get(intent, None)
                insert_info_to_kb(info, placement_prediction)
                to_return.append((info, placement_prediction))
            return to_return
        else:
            # use sequential insert as knowledge base structure might change
            for question, query in intent_to_placement_dict:
                (
                    encoded_outlines,
                    outlines,
                ) = knowledge_base.get_knowledge_base_structure_embedding(
                    root=insert_root
                )
                _, placement_prediction = process_intent(question=question, query=query)
                intent_to_placement_dict[(question, query)] = placement_prediction

            for info in information:
                intent = (info.meta.get("question", ""), info.meta.get("query", ""))
                placement_prediction = intent_to_placement_dict.get(intent, None)
                insert_info_to_kb(info, placement_prediction)
                to_return.append((info, placement_prediction))
            return to_return


class ExpandSection(dspy.Signature):
    """您的任务是通过在给定章节下创建新的子章节来扩展思维导图中的章节。
    您将得到一系列用于收集信息的问题和查询。
    输出应该是子章节名称，每个章节都应该作为信息和相应引用编号的连贯且主题性的组织。这些子章节名称最好简洁准确。
    输出格式如下：
    子章节 1
    子章节 2  
    子章节 3
    【注】：输出格式严格遵循上述格式，不要输出其他信息如“子章节 1：”, 直接输出名称。
    """

    section = dspy.InputField(prefix="您需要扩展的章节：", format=str)
    info = dspy.InputField(prefix="收集的信息：\n", format=str)
    output = dspy.OutputField(
        prefix="请用简体中文列出新的子章节名称：\n", format=str
    )


class ExpandNodeModule(dspy.Module):
    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        information_insert_module: dspy.Module,
        node_expansion_trigger_count: int,
    ):
        self.engine = engine
        self.expand_section = dspy.Predict(ExpandSection)
        self.information_insert_module = information_insert_module
        self.node_expansion_trigger_count = node_expansion_trigger_count

    def _get_cited_info_meta_string(self, node, knowledge_base):
        meta_string = set()
        for index in sorted(list(node.content)):
            info = knowledge_base.info_uuid_to_info_dict[index]
            intent = f"Question: {info.meta['question']}\nQuery: {info.meta['query']}"
            meta_string.add(intent)

        return "\n\n".join(meta_string)

    def _get_expand_subnode_names(self, node, knowledge_base):
        information = self._get_cited_info_meta_string(node, knowledge_base)
        node_path = node.get_path_from_root()
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            output = self.expand_section(section=node_path, info=information).output
        
        output = clean_chinese_output(output)
        subsections = []
        if "\n" in output and output != "None":
            subsections = output.split("\n")
            # remove any integer followed by a dot and a space, a leading dashline,
            # or a specific hint at the start of the string
            subsections = [
                re.sub(r"^\d+\.\s|-|" + re.escape(node.name), "", text)
                .replace("*", "")
                .strip()
                for text in subsections
            ]
        return subsections

    def _find_first_node_to_expand(
        self, root: KnowledgeNode, expanded_nodes: List[KnowledgeNode]
    ):
        if root is None:
            return None
        if (
            root not in expanded_nodes
            and len(root.content) >= self.node_expansion_trigger_count
        ):
            return root
        for child in root.children:
            to_return = self._find_first_node_to_expand(
                root=child, expanded_nodes=expanded_nodes
            )
            if to_return is not None:
                return to_return
        return None

    def _expand_node(self, node: KnowledgeNode, knowledge_base: KnowledgeBase):
        subsection_names = self._get_expand_subnode_names(node, knowledge_base)
        if len(subsection_names) <= 1:
            return
        # create new nodes
        for subsection_name in subsection_names:
            # remove citation bracket in the subsection name
            subsection_name = re.sub(r"\[.*?\]", "", subsection_name)
            knowledge_base.insert_node(new_node_name=subsection_name, parent_node=node)
        # reset original information placement
        original_cited_index = node.content
        original_cited_information = [
            knowledge_base.info_uuid_to_info_dict[index]
            for index in original_cited_index
        ]
        node.content = set()
        # re-insert under expanded section
        self.information_insert_module(
            knowledge_base=knowledge_base,
            information=original_cited_information,
            allow_create_new_node=False,
            insert_root=node,
        )

    def forward(self, knowledge_base: KnowledgeBase):
        expanded_nodes = []
        while True:
            node_to_expand = self._find_first_node_to_expand(
                root=knowledge_base.root, expanded_nodes=expanded_nodes
            )
            if node_to_expand is None:
                break
            self._expand_node(node=node_to_expand, knowledge_base=knowledge_base)
            expanded_nodes.append(node_to_expand)
