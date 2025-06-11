MODEL_CONFIG = {
    "api_key": "YOUR_API_KEY",  
    "api_base": "https://api.deepseek.com",  
    "model_name": "deepseek-chat",  
    "temperature": 1.0,
    "top_p": 0.9,
}

# 搜索引擎配置
SEARCH_CONFIG = {
    "serper_api_key": "YOUR_SERPER_API_KEY", 
}

# embedding配置
EMBEDDING_CONFIG = {
    "api_key": "YOUR_EMBEDDING_API_KEY",  
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",  
    "model": "text-embedding-v4",  
    "encoding_format": "float" 
} 

# 网页爬虫配置
FIRECRAWL_CONFIG = {
    "api_key": "YOUR_FIRECRAWL_API_KEY",
} 