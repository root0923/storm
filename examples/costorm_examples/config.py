# API 配置文件

# 大模型配置
MODEL_CONFIG = {
    "api_key": "your_api_key_here",
    "api_base": "your_api_base_url_here", 
    "model_name": "your_model_name_here",
    "temperature": 1.0,
    "top_p": 0.9,
}

# 搜索引擎配置
SEARCH_CONFIG = {
    "serper_api_key": "your_serper_api_key_here",
}

# embedding配置
EMBEDDING_CONFIG = {
    "api_key": "your_embedding_api_key_here",
    "api_base": "your_embedding_api_base_here",
    "model": "text-embedding-v3",
    "encoding_format": "float"
}