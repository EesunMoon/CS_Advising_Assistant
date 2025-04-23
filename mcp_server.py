import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from redis import Redis
import pickle
from datetime import datetime, timedelta
import re
import uvicorn
import logging
import sys
from contextlib import asynccontextmanager

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mcp_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 전역 변수
mcp_server = None
requirements_data = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global mcp_server
    logger.info("Starting up MCP server...")
    mcp_server = MCPServer()
    
    try:
        # 요구사항 데이터 로드
        requirements_file = "data/requirements.json"
        if os.path.exists(requirements_file):
            with open(requirements_file, "r") as f:
                requirements_data.update(json.load(f))
        logger.info("Server started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # shutdown
    try:
        if mcp_server and mcp_server.session:
            await mcp_server.session.close()
        if mcp_server and mcp_server.connector:
            await mcp_server.connector.close()
        logger.info("Server shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="Modal Context Protocol Server",
    description="MCP server for CS Advising Assistant",
    version="1.0.0",
    debug=True,
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = Redis(host='localhost', port=6379, db=0)

class WebRequest(BaseModel):
    url: str
    program: str
    level: str

class SearchRequest(BaseModel):
    query: str
    program: Optional[str] = None
    level: Optional[str] = None

class DocumentResponse(BaseModel):
    content: str
    metadata: Dict
    source: str
    url: str

class RequirementRequest(BaseModel):
    url: str
    program: str
    level: str

class SearchResponse(BaseModel):
    documents: List[Document]
    total: int
    query_time: float

class RequirementsResponse(BaseModel):
    documents: List[Document]
    program: str
    level: str
    last_updated: str

class MCPServer:
    def __init__(self):
        self.base_urls = {
            "ms": "https://www.cs.columbia.edu/education/ms",
            "phd": "https://www.cs.columbia.edu/education/phd",
            "undergraduate": "https://www.cs.columbia.edu/education/undergraduate"
        }
        self.session = None
        self.connector = None
        self.logger = logging.getLogger(__name__)
    
    async def ensure_session(self):
        """세션이 없으면 새로 생성"""
        if not self.session:
            self.connector = aiohttp.TCPConnector(verify_ssl=False)
            self.session = aiohttp.ClientSession(connector=self.connector)
    
    async def fetch_page(self, url: str) -> str:
        """Fetch page content from URL"""
        await self.ensure_session()
        
        try:
            self.logger.info(f"Fetching URL: {url}")
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    self.logger.info(f"Successfully fetched {url}")
                    return content
                else:
                    self.logger.error(f"Failed to fetch {url}: Status {response.status}")
                    return ""
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return ""
    
    async def extract_requirements(self, html: str, program: str) -> Document:
        """Extract program requirements from HTML"""
        if not html:
            return Document(
                page_content="No requirements data available",
                metadata={"program": program, "error": "Failed to fetch data"}
            )
            
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find requirements section
        requirements = []
        for section in soup.find_all(['div', 'section'], class_=re.compile(r'.*requirements.*', re.I)):
            text = section.get_text(strip=True)
            if text:  # 빈 텍스트가 아닌 경우만 추가
                requirements.append(text)
        
        if not requirements:
            # 요구사항을 찾지 못한 경우 다른 섹션도 검색
            for section in soup.find_all(['div', 'section']):
                text = section.get_text(strip=True)
                if 'requirement' in text.lower() or 'course' in text.lower():
                    requirements.append(text)
        
        content = "\n\n".join(requirements) if requirements else "No requirements found in the page"
        metadata = {
            "program": program,
            "source": "Columbia CS Website",
            "type": "program_requirements",
            "timestamp": datetime.now().isoformat()
        }
        
        return Document(page_content=content, metadata=metadata)
    
    async def search_content(self, query: str, program: Optional[str] = None) -> List[Document]:
        """Search for relevant content across CS department pages"""
        if program:
            urls = [self.base_urls.get(program.lower())]
        else:
            urls = list(self.base_urls.values())
        
        documents = []
        for url in urls:
            try:
                html = await self.fetch_page(url)
                soup = BeautifulSoup(html, 'html.parser')
                
                # 관련 섹션 찾기
                relevant_sections = []
                
                # 1. 트랙 관련 섹션 찾기
                track_keywords = ['machine learning', 'ml', 'artificial intelligence', 'ai', 
                                'track', 'concentration', 'specialization']
                
                # 2. 과목 관련 섹션 찾기
                course_keywords = ['course', 'requirement', 'curriculum', 'program']
                
                for section in soup.find_all(['div', 'section']):
                    text = section.get_text(strip=True)
                    text_lower = text.lower()
                    
                    # 트랙 또는 과목 관련 키워드가 있는지 확인
                    is_track_related = any(keyword in text_lower for keyword in track_keywords)
                    is_course_related = any(keyword in text_lower for keyword in course_keywords)
                    
                    # 쿼리와 관련된 내용이 있는지 확인
                    is_query_related = query.lower() in text_lower
                    
                    if is_query_related or (is_track_related and is_course_related):
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": url,
                                "program": program,
                                "type": "course_info" if is_course_related else "general_info",
                                "is_track_related": is_track_related
                            }
                        )
                        documents.append(doc)
                        
                self.logger.info(f"Found {len(documents)} relevant documents from {url}")
                
            except Exception as e:
                self.logger.error(f"Error searching {url}: {e}")
                continue
        
        return documents

@app.get("/api/v1/search")
async def search(
    query: str = Query(..., description="Search query"),
    program: Optional[str] = Query(None, description="Program type (e.g., ms, phd)"),
    level: Optional[str] = Query(None, description="Education level")
):
    start_time = datetime.now()
    
    try:
        if not mcp_server:
            raise HTTPException(status_code=500, detail="Server not initialized")
            
        docs = await mcp_server.search_content(query, program)
        
        # Document 객체를 JSON 직렬화 가능한 형식으로 변환
        serializable_docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]
        
        return {
            "documents": serializable_docs,
            "total": len(docs),
            "query_time": (datetime.now() - start_time).total_seconds()
        }
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/requirements")
async def get_requirements(
    program: str = Query(..., description="Program type (e.g., ms, phd)"),
    level: str = Query(..., description="Education level")
):
    try:
        if not mcp_server:
            raise HTTPException(status_code=500, detail="Server not initialized")
            
        html = await mcp_server.fetch_page(mcp_server.base_urls[program.lower()])
        doc = await mcp_server.extract_requirements(html, program)
        return {
            "documents": [doc],
            "program": program,
            "level": level,
            "last_updated": datetime.now().isoformat()
        }
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Program '{program}' not found")
    except Exception as e:
        logger.error(f"Requirements error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import socket
    
    def find_free_port(start_port: int = 8000, max_port: int = 8100) -> int:
        """사용 가능한 포트 찾기"""
        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No free ports found")
    
    port = find_free_port()
    logger.info(f"Starting server on port {port}")
    
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="debug"
    ) 