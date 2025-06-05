"""
中文处理工具集
包含中文输出清理、文本标准化、结构解析等功能
合并了 chinese_output_processor.py 和 chinese_text_utils.py 的所有功能
"""

import re
from typing import Any, Dict, List, Tuple


# ============================================================================
# 中文输出清理功能 (原 chinese_output_processor.py)
# ============================================================================

def clean_chinese_output(text: str, role_context: str = "") -> str:
    """
    清理模型输出，强制转换为纯中文内容
    
    Args:
        text (str): 原始模型输出
        role_context (str): 角色上下文，如"Moderator"、"Expert"等，用于调整清理策略
        
    Returns:
        str: 清理后的中文输出
    """
    if not text:
        return text
        
    # 1. 移除 <think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>', '', text)
    text = re.sub(r'</think>', '', text)
    
    # 2. 移除【注：...】格式的注释内容
    text = re.sub(r'【注：.*?】', '', text, flags=re.DOTALL)
    text = re.sub(r'\[注：.*?\]', '', text, flags=re.DOTALL)
    
    # 3. 移除常见的英文思考开头
    english_thinking_patterns = [
        r'^(Okay|Alright|Let me|I need to|First|So|Well),.*?(?=\n\n|\n[A-Z]|\n\d+\.)',
        r'^The (user|question|topic).*?(?=\n\n|\n[A-Z]|\n\d+\.)',
        r'^Looking at.*?(?=\n\n|\n[A-Z]|\n\d+\.)',
        r'^Based on.*?(?=\n\n|\n[A-Z]|\n\d+\.)',
    ]
    
    for pattern in english_thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
    
    # 4. 🔴 检测并移除元认知暴露（AI内部思考过程）
    # 🟢 为Moderator角色提供更宽松的清理策略
    if role_context.lower() == "moderator":
        # Moderator的元认知模式更保守，只删除明显的思考过程
        metacognitive_patterns = [
            r'好的，我(现在)?需要处理用户(提供的)?.*?(?=\n\n|$)',
            r'首先，我(需要)?仔细阅读用户.*?(?=\n\n|$)', 
            r'接下来，我(需要)?确定.*?(?=\n\n|$)',
            r'我(需要)?检查是否.*?(?=\n\n|$)',
            r'现在，我(需要)?将这些思考转化为.*?(?=\n\n|$)',
            r'根据用户.*?(?=\n\n|$)',
        ]
    else:
        # 普通专家的完整元认知清理
        metacognitive_patterns = [
            r'好的，我(现在)?需要处理用户(提供的)?.*?(?=\n\n|$)',
            r'首先，我(需要)?仔细阅读用户.*?(?=\n\n|$)', 
            r'接下来，我(需要)?确定.*?(?=\n\n|$)',
            r'然后，我(需要)?考虑.*?(?=\n\n|$)',
            r'最后，我?(需要)?确保.*?(?=\n\n|$)',
            r'我(需要)?检查是否.*?(?=\n\n|$)',
            r'现在，我(需要)?将这些思考转化为.*?(?=\n\n|$)',
            r'用户(可能希望|要求|提到).*?(?=\n\n|$)',
            r'根据用户.*?(?=\n\n|$)',
            r'这些信息需要整合.*?(?=\n\n|$)',
            r'可能的结构是.*?(?=\n\n|$)',
            r'需要确认用户.*?(?=\n\n|$)',
            r'但用户.*?(?=\n\n|$)',
            r'不过用户.*?(?=\n\n|$)',
            r'此外，必须使用正确的行内引用.*?(?=\n\n|$)',
            r'我需要检查是否有遗漏.*?(?=\n\n|$)',
            r'需要将这些信息浓缩成.*?(?=\n\n|$)',
            r'检查是否有遗漏的信息点.*?(?=\n\n|$)',
            r'确保每个引用都正确对应.*?(?=\n\n|$)',
            r'最终，组织语言.*?(?=\n\n|$)',
            r'可能分为几个主要.*?(?=\n\n|$)',
            r'我需要先确定.*?(?=\n\n|$)',
            r'然后，注意行内引用.*?(?=\n\n|$)',
            r'需要确保语言流畅.*?(?=\n\n|$)',
        ]
    
    for pattern in metacognitive_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.DOTALL)
    
    # 5. 智能处理中文思考过程：保留有实质内容和引用的句子
    lines = text.split('\n')
    useful_lines = []
    
    # 更精确的思考指示器
    thinking_indicators = [
        '好的，我现在需要处理',
        '首先，我需要仔细阅读',
        '接下来，我需要确定',
        '然后，需要组织这些信息',
        '最后，要确保语言流畅',
        '现在，我需要将这些思考转化为',
        '可能的结构是',
        '需要确保引用正确',
        '在组织语言时',
        '最后，确保内容流畅',
        '需要注意不要重复之前的内容',
        '现在，我需要将这些思考',
        # 🔴 新增：处理测试用例中的思考模式
        '需要将这些信息浓缩成',
        '检查是否有遗漏的信息点',
        '确保每个引用都正确对应',
        '最终，组织语言',
        '可能分为几个主要',
        '我需要先确定',
        '然后，注意行内引用',
        '需要确保语言流畅',
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 🔴 检查是否是思考过程
        is_pure_thinking = any(indicator in line for indicator in thinking_indicators)
        is_meta_instruction = line.startswith(('用户', '根据用户', '信息1', '信息2', '信息3', '信息4', '信息5', '信息6', '引用1', '引用2', '引用3'))
        is_task_analysis = '维基百科' in line and ('撰写' in line or '章节' in line or '格式' in line)
        
        # 🔴 跳过明显的思考过程、元指令和任务分析
        if is_pure_thinking or is_meta_instruction or is_task_analysis:
            continue
            
        useful_lines.append(line)
    
    # 6. 🔴 如果清理后没有内容，但原文包含引用，采用更保守的清理策略
    if not useful_lines and re.search(r'\[\d+\]', text):
        # 按段落重新处理
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if re.search(r'\[\d+\]', paragraph):
                # 只移除明显的思考开头句
                sentences = paragraph.split('。')
                keep_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    # 跳过明显的思考句
                    skip_sentence = any(bad_start in sentence for bad_start in [
                        '好的，我现在需要', '首先，我需要', '接下来，我需要', '然后，需要', '最后，要确保',
                        '现在，我需要', '可能的结构是', '需要确保', '在组织语言时', '最后，确保'
                    ])
                    if not skip_sentence:
                        keep_sentences.append(sentence)
                
                if keep_sentences:
                    reconstructed = '。'.join(keep_sentences)
                    if reconstructed and not reconstructed.endswith('。'):
                        reconstructed += '。'
                    useful_lines.append(reconstructed)
    
    text = '\n'.join(useful_lines)
    
    # 7. 清理多余的空行
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text


def process_dspy_output(prediction: Any) -> Any:
    """
    处理DSPy预测输出，清理所有字符串字段
    
    Args:
        prediction: DSPy预测对象
        
    Returns:
        处理后的预测对象
    """
    if hasattr(prediction, '__dict__'):
        for key, value in prediction.__dict__.items():
            if isinstance(value, str):
                setattr(prediction, key, clean_chinese_output(value))
            elif isinstance(value, list):
                # 处理列表中的字符串
                cleaned_list = []
                for item in value:
                    if isinstance(item, str):
                        cleaned_list.append(clean_chinese_output(item))
                    else:
                        cleaned_list.append(item)
                setattr(prediction, key, cleaned_list)
            elif isinstance(value, dict):
                # 处理字典中的字符串值
                cleaned_dict = {}
                for k, v in value.items():
                    if isinstance(v, str):
                        cleaned_dict[k] = clean_chinese_output(v)
                    else:
                        cleaned_dict[k] = v
                setattr(prediction, key, cleaned_dict)
    
    return prediction


def wrap_chinese_predict(predict_class):
    """
    包装DSPy Predict类，自动处理中文输出
    
    Args:
        predict_class: DSPy Predict类
        
    Returns:
        包装后的Predict类
    """
    original_forward = predict_class.forward
    
    def chinese_forward(self, **kwargs):
        # 调用原始forward方法
        result = original_forward(self, **kwargs)
        
        # 处理输出
        return process_dspy_output(result)
    
    predict_class.forward = chinese_forward
    return predict_class


class ChinesePredict:
    """
    中文化的DSPy Predict包装器
    """
    def __init__(self, signature, **kwargs):
        import dspy
        self.predict = dspy.Predict(signature, **kwargs)
    
    def __call__(self, **kwargs):
        result = self.predict(**kwargs)
        return process_dspy_output(result)
    
    def forward(self, **kwargs):
        result = self.predict.forward(**kwargs)
        return process_dspy_output(result)


# ============================================================================
# 中文文本处理工具 (原 chinese_text_utils.py)
# ============================================================================

def normalize_chinese_action_keywords(text: str) -> str:
    """标准化中文动作关键词"""
    # 动作关键词映射
    action_mappings = {
        "插入": "insert",
        "进入": "step",
        "创建": "create",
        "最佳放置": "Best placement",
        "无合理选择": "No reasonable choice",
    }
    
    normalized_text = text
    for chinese, english in action_mappings.items():
        # 保持原文，但确保解析逻辑能识别
        pass
    
    return normalized_text


def parse_chinese_node_expansion_output(output: str) -> List[str]:
    """解析中文节点扩展输出"""
    subsections = []
    for line in output.split("\n"):
        line = line.strip()
        if line:
            # 移除可能的编号和特殊字符
            cleaned_line = re.sub(r'^[\d\.\s\-\*\+]*', '', line)
            cleaned_line = cleaned_line.strip()
            if cleaned_line:
                subsections.append(cleaned_line)
    
    return subsections


def clean_chinese_knowledge_structure(structure: str) -> str:
    """清理中文知识结构显示"""
    # 标准化层级标记
    structure = re.sub(r'#+\s*', lambda m: '#' * len(m.group(0).strip()) + ' ', structure)
    return structure


def normalize_chinese_punctuation(text: str) -> str:
    """标准化中英文标点符号"""
    # 将英文冒号替换为中文冒号（在中文语境中）
    text = re.sub(r'(\w+):\s*([^\w])', r'\1：\2', text)
    
    # 处理其他标点符号
    replacements = {
        '，。': '，。',  # 保持中文标点
        '；！': '；！',  # 保持中文标点
        '？': '？',      # 保持中文问号
    }
    
    return text


def clean_chinese_expert_output(output: str) -> List[str]:
    """清理中文专家输出，提取专家角色列表"""
    # 移除特殊字符
    output = output.replace("*", "").replace("[", "").replace("]", "")
    
    expert_list = []
    for line in output.split("\n"):
        line = line.strip()
        # 匹配中文编号格式：1. 2. 或 1、2、
        patterns = [
            r"\d+\.\s*(.*)",  # 英文句号
            r"\d+、\s*(.*)",  # 中文顿号
            r"\d+：\s*(.*)",  # 中文冒号
            r"\d+:\s*(.*)",   # 英文冒号
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                expert_info = match.group(1).strip()
                if expert_info:
                    expert_list.append(expert_info)
                break
    
    return expert_list


def clean_chinese_section_text(text: str) -> str:
    """清理中文章节文本，包括think标签、元认知内容和截断处理"""
    if not text:
        return text
    
    # 🔴 首先使用通用的中文输出清理函数
    text = clean_chinese_output(text)
    
    # 移除不完整的句子（通常由于输出token限制）
    paragraphs = text.split("\n")
    
    # 清理逻辑：如果段落不以中文标点结尾，可能是不完整的
    chinese_end_punctuation = r'[。！？]$'
    
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # 如果是最后一个段落且没有以标点结尾，可能是不完整的
            if paragraph == paragraphs[-1] and not re.search(chinese_end_punctuation, paragraph):
                # 检查是否包含引用，如果有引用说明可能是完整的
                if re.search(r'\[\d+\]', paragraph):
                    cleaned_paragraphs.append(paragraph)
                # 否则跳过这个可能不完整的段落
            else:
                cleaned_paragraphs.append(paragraph)
    
    return "\n".join(cleaned_paragraphs)


# ============================================================================
# 通用中文处理工具函数
# ============================================================================

def is_chinese_text(text: str) -> bool:
    """判断文本是否主要为中文"""
    if not text:
        return False
    
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    total_chars = len(re.findall(r'[^\s]', text))
    
    if total_chars == 0:
        return False
    
    return chinese_chars / total_chars > 0.5


def extract_chinese_sentences(text: str) -> List[str]:
    """提取中文句子"""
    # 按中文标点符号分割句子
    sentences = re.split(r'[。！？；]', text)
    
    # 清理空句子和过短的句子
    chinese_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 3 and is_chinese_text(sentence):
            chinese_sentences.append(sentence)
    
    return chinese_sentences


def format_chinese_list(items: List[str], style: str = "numbered") -> str:
    """格式化中文列表
    
    Args:
        items: 列表项
        style: 格式样式 ("numbered", "bullet", "chinese_number")
    """
    if not items:
        return ""
    
    formatted_items = []
    
    for i, item in enumerate(items, 1):
        if style == "numbered":
            formatted_items.append(f"{i}. {item}")
        elif style == "bullet":
            formatted_items.append(f"• {item}")
        elif style == "chinese_number":
            chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
            if i <= len(chinese_numbers):
                formatted_items.append(f"{chinese_numbers[i-1]}、{item}")
            else:
                formatted_items.append(f"{i}、{item}")
        else:
            formatted_items.append(item)
    
    return "\n".join(formatted_items)


def clean_mixed_language_text(text: str, prefer_chinese: bool = True) -> str:
    """清理混合语言文本，优先保留指定语言
    
    Args:
        text: 输入文本
        prefer_chinese: 是否优先保留中文
    """
    if not text:
        return text
    
    # 先进行基本清理
    text = clean_chinese_output(text)
    
    if prefer_chinese:
        # 提取中文句子
        chinese_sentences = extract_chinese_sentences(text)
        if chinese_sentences:
            return "。".join(chinese_sentences) + "。"
    
    return text


# ============================================================================
# 便捷函数
# ============================================================================

def process_all_chinese_text(text_dict: Dict[str, str]) -> Dict[str, str]:
    """批量处理字典中的所有中文文本"""
    processed = {}
    for key, value in text_dict.items():
        if isinstance(value, str):
            processed[key] = clean_chinese_output(value)
        else:
            processed[key] = value
    return processed


def get_text_statistics(text: str) -> Dict[str, int]:
    """获取文本统计信息"""
    return {
        "总字符数": len(text),
        "中文字符数": len(re.findall(r'[\u4e00-\u9fff]', text)),
        "英文单词数": len(re.findall(r'\b[a-zA-Z]+\b', text)),
        "数字数量": len(re.findall(r'\d+', text)),
        "标点符号数": len(re.findall(r'[，。！？；：""''（）【】《》]', text)),
        "句子数": len(extract_chinese_sentences(text)),
    } 