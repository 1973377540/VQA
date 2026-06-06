"""DashScope Embeddings 封装 - 用于 RAG 向量嵌入"""

import dashscope
from dashscope import TextEmbedding
from typing import List


class DashScopeEmbeddings:
    """使用通义千问 embedding 模型的 LangChain 兼容封装"""

    def __init__(self, api_key: str = None, model: str = "text-embedding-v3"):
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档文本"""
        results = []
        # DashScope 每次最多支持 25 条文本
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = TextEmbedding.call(
                api_key=self.api_key,
                model=self.model,
                input=batch
            )
            if response.status_code == 200:
                for item in response.output['embeddings']:
                    results.append(item['embedding'])
            else:
                raise Exception(f"Embedding failed: {response.code} - {response.message}")
        return results

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询文本"""
        return self.embed_documents([text])[0]
