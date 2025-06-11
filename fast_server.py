import argparse
from typing import Dict, List, Union
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from deepsearcher.configuration import Configuration, init_config
from deepsearcher.offline_loading import load_from_local_files, load_from_website
from deepsearcher.online_query import query, retrieve
from hashlib import md5

# 导入配置参数
from examples.costorm_examples.config import MODEL_CONFIG, EMBEDDING_CONFIG, FIRECRAWL_CONFIG

# 设置FireCrawl API密钥环境变量
os.environ["FIRECRAWL_API_KEY"] = FIRECRAWL_CONFIG["api_key"]

app = FastAPI()

# 设置Milvus数据库路径
current_dir = os.path.dirname(os.path.abspath(__file__))
milvus_db_path = os.path.join(current_dir, "examples", "costorm_examples", "new.db")

config = Configuration()
# 使用config.py中的MODEL_CONFIG参数
config.set_provider_config("llm", "OpenAI", {
    "model": MODEL_CONFIG["model_name"],
    "api_key": MODEL_CONFIG["api_key"],
    "base_url": MODEL_CONFIG["api_base"] + "/v1"
})

# 使用config.py中的EMBEDDING_CONFIG参数
config.set_provider_config("embedding", "OpenAIEmbedding", {
    "model": EMBEDDING_CONFIG["model"],
    "api_key": EMBEDDING_CONFIG["api_key"],
    "base_url": EMBEDDING_CONFIG["api_base"],
    "dimension": 1024
})
# 设置Milvus向量数据库配置
config.set_provider_config("vector_db", "Milvus", {
    "default_collection": "deepsearcher",
    "uri": milvus_db_path,
    "token": "root:Milvus",
    "db": "default"
})

init_config(config = config)


class ProviderConfigRequest(BaseModel):
    """
    Request model for setting provider configuration.

    Attributes:
        feature (str): The feature to configure (e.g., 'embedding', 'llm').
        provider (str): The provider name (e.g., 'openai', 'azure').
        config (Dict): Configuration parameters for the provider.
    """

    feature: str
    provider: str
    config: Dict


@app.post("/set-provider-config/")
def set_provider_config(request: ProviderConfigRequest):
    """
    Set configuration for a specific provider.

    Args:
        request (ProviderConfigRequest): The request containing provider configuration.

    Returns:
        dict: A dictionary containing a success message and the updated configuration.

    Raises:
        HTTPException: If setting the provider config fails.
    """
    try:
        config.set_provider_config(request.feature, request.provider, request.config)
        init_config(config)
        return {
            "message": "Provider config set successfully",
            "provider": request.provider,
            "config": request.config,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set provider config: {str(e)}")


@app.post("/load-files/")
def load_files(
    paths: Union[str, List[str]] = Body(
        ...,
        description="A list of file paths to be loaded.",
        examples=["/path/to/file1", "/path/to/file2", "/path/to/dir1"],
    ),
    collection_name: str = Body(
        None,
        description="Optional name for the collection.",
        examples=["my_collection"],
    ),
    collection_description: str = Body(
        None,
        description="Optional description for the collection.",
        examples=["This is a test collection."],
    ),
    batch_size: int = Body(
        None,
        description="Optional batch size for the collection.",
        examples=[256],
    ),
):
    """
    Load files into the vector database.

    Args:
        paths (Union[str, List[str]]): File paths or directories to load.
        collection_name (str, optional): Name for the collection. Defaults to None.
        collection_description (str, optional): Description for the collection. Defaults to None.
        batch_size (int, optional): Batch size for processing. Defaults to None.

    Returns:
        dict: A dictionary containing a success message.

    Raises:
        HTTPException: If loading files fails.
    """
    try:
        load_from_local_files(
            paths_or_directory=paths,
            collection_name=collection_name,
            collection_description=collection_description,
            batch_size=batch_size,
        )
        return {"message": "Files loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-website/")
def load_website(
    urls: Union[str, List[str]] = Body(
        ...,
        description="A list of URLs of websites to be loaded.",
        examples=["https://milvus.io/docs/overview.md"],
    ),
    collection_name: str = Body(
        None,
        description="Optional name for the collection.",
        examples=["my_collection"],
    ),
    collection_description: str = Body(
        None,
        description="Optional description for the collection.",
        examples=["This is a test collection."],
    ),
    batch_size: int = Body(
        None,
        description="Optional batch size for the collection.",
        examples=[256],
    ),
):
    """
    Load website content into the vector database.

    Args:
        urls (Union[str, List[str]]): URLs of websites to load.
        collection_name (str, optional): Name for the collection. Defaults to None.
        collection_description (str, optional): Description for the collection. Defaults to None.
        batch_size (int, optional): Batch size for processing. Defaults to None.

    Returns:
        dict: A dictionary containing a success message.

    Raises:
        HTTPException: If loading website content fails.
    """
    try:
        load_from_website(
            urls=urls,
            collection_name=collection_name,
            collection_description=collection_description,
            batch_size=batch_size,
        )
        return {"message": "Website loaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/")
def perform_query(
    original_query: str = Query(
        ...,
        description="Your question here.",
        examples=["Write a report about Milvus."],
    ),
    max_iter: int = Query(
        3,
        description="The maximum number of iterations for reflection.",
        ge=1,
        examples=[3],
    ),
):
    """
    Perform a query against the loaded data.

    Args:
        original_query (str): The user's question or query.
        max_iter (int, optional): Maximum number of iterations for reflection. Defaults to 3.

    Returns:
        dict: A dictionary containing the query result and token consumption.

    Raises:
        HTTPException: If the query fails.
    """
    try:
        result_text, _, consume_token = query(original_query, max_iter)
        return {"result": result_text, "consume_token": consume_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/retrieve/")
def perform_retrieve(query: str=  Query(
    ...,
    description="Your question here.",
    example="Write a report about Milvus.",
)):
    
    if not query:
        return {"error": "Query parameter is required"}
    try:
        # 获取向量知识库结果，转为storm格式。
        vec_results, _, consume_token = retrieve(query, max_iter=1)
        out = []
        for rr in vec_results:
            clean = rr.metadata["wider_text"]
            title = clean[:25]
            desc = rr.reference
            url = md5(f'{desc}{title}'.encode()).hexdigest()
            query = rr.query
            ele = {
                "url": f'vec://{url}', "title": title,
                "desc": desc,       "snippets": clean,
                "query": query
            }
            out.append(ele)
        return out
    except Exception as ex:
        return {"perform_retrieve error": str(ex)}




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Server")
    parser.add_argument("--enable-cors", type=bool, default=True, help="Enable CORS support")
    args = parser.parse_args()
    if args.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("CORS is enabled.")
    else:
        print("CORS is disabled.")
    uvicorn.run(app, host="0.0.0.0", port=8500)
