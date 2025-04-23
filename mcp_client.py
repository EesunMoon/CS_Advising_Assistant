import aiohttp
import json
import asyncio
import redis
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from langchain.docstore.document import Document

def document_to_dict(doc: Document) -> Dict[str, Any]:
    """Convert Document object to dictionary"""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

def dict_to_document(data: Dict[str, Any]) -> Document:
    """Convert dictionary to Document object"""
    return Document(
        page_content=data["page_content"],
        metadata=data["metadata"]
    )

@dataclass
class MCPConfig:
    base_url: str = "http://localhost:8000"
    api_prefix: str = "/api/v1"
    timeout: int = 30
    max_retries: int = 3
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

class MCPClient:
    def __init__(self, config: Optional[MCPConfig] = None, session: Optional[aiohttp.ClientSession] = None):
        self.config = config or MCPConfig()
        self.session = session
        self.redis_client = redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db
        )
    
    async def get_cache(self, key: str) -> Optional[List[Document]]:
        """Get cached documents from Redis"""
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                docs_dict = json.loads(cached_data)
                return [dict_to_document(doc) for doc in docs_dict]
        except Exception as e:
            print(f"Cache retrieval error: {e}")
        return None
    
    async def set_cache(self, key: str, docs: List[Document], expire: int = 3600):
        """Set cache in Redis with expiration time"""
        try:
            docs_dict = [document_to_dict(doc) for doc in docs]
            serialized = json.dumps(docs_dict)
            self.redis_client.setex(key, expire, serialized)
        except Exception as e:
            print(f"Cache setting error: {e}")
    
    async def __aenter__(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_requirements(self, program: str, level: str) -> List[Document]:
        """Get program requirements from MCP server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context manager.")
        
        for attempt in range(self.config.max_retries):
            try:
                url = f"{self.config.base_url}{self.config.api_prefix}/requirements"
                async with self.session.get(
                    url,
                    params={
                        "program": program.lower(),
                        "level": level.lower()
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [Document(
                            page_content=doc["content"],
                            metadata=doc["metadata"]
                        ) for doc in data.get("documents", [])]
                    elif response.status == 404:
                        print(f"No requirements found for {program} {level}")
                        return []
                    else:
                        print(f"Server returned status {response.status} for URL: {url}")
                        return []
            except Exception as e:
                print(f"Error getting requirements: {e}")
                if attempt == self.config.max_retries - 1:
                    return []
                await asyncio.sleep(1)  # 재시도 전 잠시 대기
                continue
        
        return []
    
    async def search(self, query: str, program: Optional[str] = None, level: Optional[str] = None) -> List[Document]:
        """Search for relevant content using MCP server"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async with context manager.")
        
        # 캐시 키 생성
        cache_key = f"search:{query}:{program}:{level}"
        
        # 캐시 확인
        cached_results = await self.get_cache(cache_key)
        if cached_results:
            return cached_results
        
        for attempt in range(self.config.max_retries):
            try:
                url = f"{self.config.base_url}{self.config.api_prefix}/search"
                params = {"query": query}
                if program:
                    params["program"] = program.lower()
                if level:
                    params["level"] = level.lower()
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        documents = [
                            Document(
                                page_content=item["content"],
                                metadata=item["metadata"]
                            )
                            for item in data.get("documents", [])
                        ]
                        
                        # 결과 캐시 저장
                        await self.set_cache(cache_key, documents)
                        return documents
                    elif response.status == 404:
                        print(f"No search results found for query: {query}")
                        return []
                    else:
                        print(f"Server returned status {response.status} for URL: {url}")
                        return []
            except Exception as e:
                print(f"Error searching: {e}")
                if attempt == self.config.max_retries - 1:
                    return []
                await asyncio.sleep(1)  # 재시도 전 잠시 대기
                continue
        
        return [] 