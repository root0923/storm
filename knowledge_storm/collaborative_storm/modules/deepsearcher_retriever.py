import logging
import requests
from typing import List, Union
from ...interface import Information


class DeepSearcherRetriever:
    """
    DeepSearcher检索器，用于调用deep-searcher API进行向量搜索
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8500"):
        """
        初始化DeepSearcher检索器
        
        Args:
            api_base_url: DeepSearcher API的基础URL
        """
        self.api_base_url = api_base_url
    
    def retrieve(self, question: str, exclude_urls: List[str] = []) -> List[Information]:
        """
        使用DeepSearcher API进行检索
        
        Args:
            question: 查询的问题
            exclude_urls: 要排除的URL列表
            
        Returns:
            List[Information]: Information对象列表
        """
        information_list = []
        
        try:
            # 调用deep searcher的/retrieve/端点
            response = requests.get(
                f"{self.api_base_url}/retrieve/",
                params={"query": question}
            )
            
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list):
                    query_results = []
                    for result in results:
                        url = result.get("url", "")
                        if url and url not in exclude_urls:
                            # 将API结果转换为Information对象
                            info = Information(
                                url=url,
                                title=result.get("title", ""),
                                description=result.get("desc", ""),
                                snippets=[result.get("snippets", "")],
                                meta={"query": result.get("query", ""), "question": question}
                            )
                            query_results.append(info)
                    
                    information_list.extend(query_results)
                    logging.info(f"DeepSearcher查询 '{question}' 返回 {len(query_results)} 个结果")
                else:
                    logging.warning(f"DeepSearcher API返回非预期格式: {results}")
            else:
                logging.error(f"DeepSearcher API错误 {response.status_code}: {response.text}")
                
        except Exception as e:
            logging.error(f"调用DeepSearcher API时出错 '{question}': {e}")
    
        return information_list
    