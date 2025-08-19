# Reference: https://github.com/bytedance/deer-flow/blob/main/src/tools/crawl.py
from ..crawler.crawler import Crawler
from .tools import Tool
from typing import Union, Dict
import logging

logger = logging.getLogger(__name__)

class CrawlerTool(Tool):
    def __init__(self):
        super().__init__("crawl", "Use this to crawl a url and get a readable content in markdown format.")

    def __call__(self, url: str) -> str:
        """Use this to crawl a url and get a readable content in markdown format."""
        try:
            crawler = Crawler()
            article = crawler.crawl(url)
            return {"url": url, "crawled_content": article.to_markdown()[:1000]}
        except BaseException as e:
            error_msg = f"Failed to crawl. Error: {repr(e)}"
            logger.error(error_msg)
            return error_msg
    
    # According to our tool definition
    def run(self, input: Union[str, Dict[str, str]]) -> Dict[str, str]:
        if isinstance(input, dict):
            url = input.get("url", "")
        else:
            url = input
        
        try:
            crawler = Crawler()
            article = crawler.crawl(url)
            return {"url": url, "crawled_content": article.to_markdown()[:1000]}
        except BaseException as e:
            error_msg = f"Failed to crawl. Error: {repr(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "url": url}
