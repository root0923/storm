import dspy
from typing import Union
from ...dataclass import KnowledgeBase


class KnowledgeBaseSummmary(dspy.Signature):
    """您的任务是对圆桌对话中已讨论的内容给出简要总结。内容按主题分层组织成层次化章节。
    您将看到这些章节，其中"#"表示章节级别。
    请用简体中文给出简洁准确的总结。
    """

    topic = dspy.InputField(prefix="主题：", format=str)
    structure = dspy.InputField(prefix="树形结构：\n", format=str)
    output = dspy.OutputField(prefix="现在用简体中文给出简要总结：\n", format=str)

class KnowledgeBaseSummaryModule(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.engine = engine
        self.gen_summary = dspy.Predict(KnowledgeBaseSummmary)

    def forward(self, knowledge_base: KnowledgeBase):
        structure = knowledge_base.get_node_hierarchy_string(
            include_indent=False,
            include_full_path=False,
            include_hash_tag=True,
            include_node_content_count=False,
        )
        with dspy.settings.context(lm=self.engine, show_guidelines=False):
            summary = self.gen_summary(
                topic=knowledge_base.topic, structure=structure
            ).output
        return summary
