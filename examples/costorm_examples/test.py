import sys
import os

# 添加storm项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
storm_root = os.path.dirname(os.path.dirname(current_dir))  # 回到storm根目录
sys.path.insert(0, storm_root)

from knowledge_storm.collaborative_storm.engine import CollaborativeStormLMConfigs, RunnerArgument, CoStormRunner
from knowledge_storm.lm import DeepSeekModel
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.rm import SerperRM
from knowledge_storm.collaborative_storm.modules.callback import (
    LocalConsolePrintCallBackHandler,
)
from knowledge_storm.encoder import Encoder
from config import MODEL_CONFIG, SEARCH_CONFIG, EMBEDDING_CONFIG
import json
import re
import logging
from datetime import datetime
import requests
import time

# 数据导入相关配置
API_BASE_URL = "http://localhost:8500"

def load_data_to_deepsearcher():
    """将data文件夹中的数据加载到DeepSearcher向量数据库"""
    
    # # 数据文件路径
    # data_dir = os.path.join(current_dir, "data")
        
    # # 例子: 导入milvus相关文档
    # load_config = {
    #     "paths": data_dir,
    #     "collection_name": "milvus_docs",
    #     "collection_description": "Milvus向量数据库文档",
    #     "batch_size": 10  # 阿里云API限制batch_size不能超过10
    # }
    
    # try:
    #     response = requests.post(f"{API_BASE_URL}/load-files/", 
    #                             json=load_config, 
    #                             timeout=300)
    #     if response.status_code == 200:
    #         print("✅ 本地文件加载成功")
    #     else:
    #         print(f"❌ 本地文件加载失败: {response.text}")
    # except Exception as e:
    #     print(f"加载本地文件时出错: {e}")

    # # 加载网页
    # websites_to_load = [
    #         "https://milvus.io/docs/overview.md",
    #         "https://milvus.io/docs/install_standalone-docker.md",
    #         "https://zilliz.com/what-is-milvus"
    #     ]
    
    # load_config = {
    #     "urls": websites_to_load,
    #     "collection_name": "milvus_docs_url",
    #     "collection_description": "Milvus向量数据库文档",
    #     "batch_size": 10  # 阿里云API限制batch_size不能超过10
    # }

    # try:
    #     response = requests.post(f"{API_BASE_URL}/load-website/", 
    #                             json=load_config, 
    #                             timeout=300)
    #     if response.status_code == 200:
    #         print("✅ 网页加载成功")
    #     else:  
    #         print(f"❌ 网页加载失败: {response.text}")
    # except Exception as e:
    #     print(f"加载网页时出错: {e}")
    

    return True

class RealTimeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)
        
    def emit(self, record):
        try:
            super().emit(record)
            self.flush() 
        except Exception as e:
            print(f"日志写入错误: {e}")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def dual_print(message):
    """同时打印到控制台和记录到日志"""
    original_print(message)
    logging.info(message)
    for handler in logging.getLogger().handlers:
        handler.flush()

original_print = print
print = dual_print

# ============ encoder configurations ============ 
os.environ["ENCODER_API_TYPE"] = "openai"

# Co-STORM adopts the same multi LM system paradigm as STORM 
lm_config: CollaborativeStormLMConfigs = CollaborativeStormLMConfigs()
kwargs = {
    "api_key": MODEL_CONFIG["api_key"],
    "api_base": MODEL_CONFIG["api_base"],
    "temperature": 1.0,
    "top_p": 0.9,
}
model_name = MODEL_CONFIG["model_name"]

question_answering_lm = DeepSeekModel(model=model_name, max_tokens=3000, **kwargs )
discourse_manage_lm = DeepSeekModel(model=model_name, max_tokens=1500, **kwargs )
utterance_polishing_lm = DeepSeekModel(model=model_name, max_tokens=3000, **kwargs )
warmstart_outline_gen_lm = DeepSeekModel(model=model_name, max_tokens=500, **kwargs )
question_asking_lm = DeepSeekModel(model=model_name, max_tokens=1000, **kwargs )
knowledge_base_lm = DeepSeekModel(model=model_name, max_tokens=1000, **kwargs )

lm_config.set_question_answering_lm(question_answering_lm)
lm_config.set_discourse_manage_lm(discourse_manage_lm)
lm_config.set_utterance_polishing_lm(utterance_polishing_lm)
lm_config.set_warmstart_outline_gen_lm(warmstart_outline_gen_lm)
lm_config.set_question_asking_lm(question_asking_lm)
lm_config.set_knowledge_base_lm(knowledge_base_lm)

# Check out the Co-STORM's RunnerArguments class for more configurations.
# 首先加载数据到DeepSearcher
print("="*60)
print("开始初始化DeepSearcher数据...")
print("="*60)

data_loaded = load_data_to_deepsearcher()
if not data_loaded:
    print("数据加载失败，但程序将继续运行（仅使用网络查询）")
    print("如需使用本地数据，请先启动DeepSearcher服务并重新运行")

print("="*60)
topic = input("请输入话题: ")

# 创建日志目录（如果不存在）
log_dir = os.path.join(storm_root, 'log')
os.makedirs(log_dir, exist_ok=True)

# 重新配置完整的日志记录系统
log_file_path = os.path.join(log_dir, f'{topic}.log')
file_handler = RealTimeFileHandler(log_file_path, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.root.setLevel(logging.INFO)
logging.root.addHandler(file_handler)
logging.root.addHandler(console_handler)

logging.info(f"用户输入话题: {topic}")

runner_argument = RunnerArgument(
    topic=topic,
    warmstart_max_num_experts=3,  # warm start阶段的专家数量（相当于视角个数）
    max_num_round_table_experts=3,  # 圆桌讨论中的活跃专家数量
    warmstart_max_turn_per_experts=1,  # 每个专家在warm start阶段的最大轮数
    moderator_override_N_consecutive_answering_turn=3,  # 主持人干预前的连续回答轮数
)
logging_wrapper = LoggingWrapper(lm_config)
callback_handler = LocalConsolePrintCallBackHandler()
serper_rm = SerperRM(
                serper_search_api_key=SEARCH_CONFIG["serper_api_key"],
                query_params={"autocorrect": True, "num": 10, "page": 1},
            )

# 创建带有embedding配置的encoder
encoder = Encoder(embedding_config=EMBEDDING_CONFIG)

costorm_runner = CoStormRunner(lm_config=lm_config,
                               runner_argument=runner_argument,
                               logging_wrapper=logging_wrapper,
                               rm=serper_rm,
                               callback_handler=callback_handler,
                               encoder=encoder
                               )

def print_kb_structure(costorm_runner, step_info=""):
    """打印知识库的树形结构和当前专家列表"""
    print(f"\n{'='*60}")
    print(f"Knowledge Base 结构 {step_info}")
    print(f"{'='*60}")
    
    print(f"\n知识库结构:")
    kb_structure = costorm_runner.knowledge_base.get_node_hierarchy_string(
        include_indent=True,
        include_hash_tag=True,
        include_node_content_count=True
    )
    
    if kb_structure.strip():
        print(kb_structure)
    else:
        print("知识库目前为空")
    
    print(f"{'='*60}\n")

print("🔥 warm start...")
# Warm start the system to build shared conceptual space between Co-STORM and users
costorm_runner.warm_start()

# 打印warm start阶段的conversation内容
print("\n Warm Start阶段的对话内容:")
for turn in costorm_runner.warmstart_conv_archive:
    print(f"**{turn.role}**\n{turn.utterance_type}:{turn.utterance}\n")

# 打印warm start后的知识库结构
print_kb_structure(costorm_runner, "(Warm Start后)")

print("💬 start conversation...")
# Step through the collaborative discourse 
# Run either of the code snippets below in any order, as many times as you'd like
# To observe the conversation:
successful_turns = 0
step_count = 0

while True:
    print("请选择下一步操作：")
    print("1: 系统生成话语；2: 用户输入话语；3: 终止生成话语，生成report")
    choice = input("请输入选项（1/2/3）: ")
    logging.info(f"用户选择: {choice}")
    
    if choice == "1":
        step_count += 1
        conv_turn = costorm_runner.step()
        while conv_turn is None:
            conv_turn = costorm_runner.step()
        print(f"**{conv_turn.role}**\n{conv_turn.utterance_type}:{conv_turn.utterance}\n")
        if conv_turn.role == "Moderator":
            print_kb_structure(costorm_runner, f"(Step {step_count}后)")
        elif conv_turn.role == "General Knowledge Provider":
            print("当前专家列表:")
            if hasattr(costorm_runner, 'discourse_manager') and hasattr(costorm_runner.discourse_manager, 'experts'):
                experts = costorm_runner.discourse_manager.experts
                if experts:
                    for expert in experts:
                        print(f"   {expert.role_name}")
                else:
                    print("  暂无专家")
            else:
                print("  专家信息不可用")
        
    elif choice == "2":
        step_count += 1
        user_utterance = input("请输入用户话语: ")
        logging.info(f"用户输入话语: {user_utterance}")
        costorm_runner.step(user_utterance=user_utterance)
        # 用户输入后自动执行一次系统生成话语
        step_count += 1
        conv_turn = costorm_runner.step()
        while conv_turn is None:
            print("语句为空，重新生成")
            conv_turn = costorm_runner.step()
        print(f"**{conv_turn.role}**\n{conv_turn.utterance_type}:{conv_turn.utterance}\n")
        if conv_turn.role == "Moderator":
            print_kb_structure(costorm_runner, f"(Step {step_count}后)")
        elif conv_turn.role == "General Knowledge Provider":
            print("当前专家列表:")
            if hasattr(costorm_runner, 'discourse_manager') and hasattr(costorm_runner.discourse_manager, 'experts'):
                experts = costorm_runner.discourse_manager.experts
                if experts:
                    for expert in experts:
                        print(f"   {expert.role_name}")
                else:
                    print("  暂无专家")
            else:
                print("  专家信息不可用")
        
    elif choice == "3":
        break
    else:
        print("无效选项，请重新输入。")


# 在生成报告前打印重组前的结构
print("📄 generate report...")
print_kb_structure(costorm_runner, "(重组前)")

costorm_runner.knowledge_base.reorganize()

# 打印重组后的结构
print_kb_structure(costorm_runner, "(重组后)")

article = costorm_runner.generate_report()

print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
print(article)
print(' >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ')
print("✅ report has been generated")