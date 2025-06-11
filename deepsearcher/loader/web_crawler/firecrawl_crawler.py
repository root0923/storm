import os
from typing import List, Optional

from firecrawl import FirecrawlApp
from langchain_core.documents import Document

from deepsearcher.loader.web_crawler.base import BaseCrawler


class FireCrawlCrawler(BaseCrawler):
    """
    Web crawler using the FireCrawl service.

    This crawler uses the FireCrawl service to crawl web pages and convert them
    into markdown format for further processing. It supports both single-page scraping
    and recursive crawling of multiple pages.
    """

    def __init__(self, **kwargs):
        """
        Initialize the FireCrawlCrawler.

        Args:
            **kwargs: Optional keyword arguments.
        """
        super().__init__(**kwargs)
        self.app = None

    def crawl_url(
        self,
        url: str,
        max_depth: Optional[int] = None,
        limit: Optional[int] = None,
        allow_backward_links: Optional[bool] = None,
    ) -> List[Document]:
        """
        Crawl a URL using FireCrawl API v1.

        Args:
            url (str): The URL to crawl.
            max_depth (Optional[int]): Not used in single URL scraping.
            limit (Optional[int]): Not used in single URL scraping.
            allow_backward_links (Optional[bool]): Not used in single URL scraping.

        Returns:
            List[Document]: List of Document objects with page content and metadata.
        """

        # Lazy init
        if self.app is None:
            self.app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

        try:
            # 使用新的FireCrawl v1 API格式进行单页面抓取
            scrape_result = self.app.scrape_url(url, formats=["markdown"])
            
            # 检查响应格式 - v1 API直接返回data字典
            if isinstance(scrape_result, dict):
                markdown_content = scrape_result.get("markdown", "")
                metadata = scrape_result.get("metadata", {})
            else:
                # 如果是其他格式，尝试获取属性
                markdown_content = getattr(scrape_result, 'markdown', "")
                metadata = getattr(scrape_result, 'metadata', {})
            
            # 确保有reference字段
            if 'reference' not in metadata:
                metadata["reference"] = url
                
            return [Document(page_content=markdown_content, metadata=metadata)]
            
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            # 返回空文档而不是抛出异常
            return [Document(page_content="", metadata={"reference": url, "error": str(e)})]
