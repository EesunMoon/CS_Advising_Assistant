import os
import json
import re
import pickle
import asyncio
import time
import torch
import aiohttp
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import conversational_retrieval
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.messages import HumanMessage, AIMessage
from utils import time_model_call, TimingStats, log_model_stats
from mcp_client import MCPClient
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from cs_web_retriever import CSProgramWebRetriever
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType


"""
User Query
   ↓
[chat()]
   ├──> _get_relevant_context(query)
   │       ├──> FAISS (vector store)
   │       ├──> MCP API (structured data)
   │       ├──> DuckDuckGo (real-time, general site)
   │       └──> CSProgramWebRetriever(cs.columbia.edu, crawl+chunking)
   ↓
_build_prompt(query, context_docs)
   ↓
ChatOllama (deepseek-r1:1.5b)
   ↓
LLM response + confidence score

--

1. Convert Chat() dialogue multiturn
2. Introduce Chat History
3. Clarification Question logic if the input is insufficient
4. Expand RunnableSequence & Agent frame

"""

class EnhancedCSAssistant:
    def __init__(self, 
                 model_name="deepseek-r1:1.5b",
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 dirname="Dataset/",
                 faq_filename="FAQ_modified.json",
                 vectorstore_name="faq_modified"):
        
        print(f"--Loading Enhanced CS Assistant with {model_name}--")
        
        # Store model name
        self.model_name = model_name
        
        # Initialize paths
        self.faq_filename = os.path.join(dirname, faq_filename)
        self.vectorstore_dir = dirname
        self.vectorstore_name = vectorstore_name
        
        # Model Context Protocol settings
        self.context_window = 4096
        self.max_input_tokens = 3072 
        
        # Initialize MCP client
        self.mcp_client = None
        
        # Initialize models
        self.llm = ChatOllama(
            model=model_name,
            context_window=self.context_window
        )
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Initialize Web Retriever
        # self.web_retriever = DuckDuckGoSearchResults(k=3)  # top-3 web search

        # Initialize CS web Retriever
        self.cs_retriever = CSProgramWebRetriever(k=5, use_chunking=True)

        # Initialize vector store
        if os.path.exists(os.path.join(self.vectorstore_dir, f"{self.vectorstore_name}.faiss")):
            self.load_vector_store()
        else:
            self.build_vector_store()

            
        # Initialize timing stats
        self.timing_stats = TimingStats()
        
        # course information cache
        self.course_cache = {}
        
        # General Response Template
        self.default_responses = {
            "greeting": "Hello! I'm your CS department academic advisor. How can I help you today?",
            "farewell": "Thank you!",
            "unknown": "I understand you're asking about {}, but I need more specific information. Could you please clarify your question about courses, requirements, or specific programs?",
            "error": "I apologize, but I'm having trouble accessing some information right now. Could you please rephrase your question or be more specific about what you'd like to know?"
        }

        # [add] memory retrieval
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def process_documents(self, json_data: List[Dict[str, Any]]) -> List[Document]:
        documents = []
        for item in json_data:
            metadata = item.get("metadata", {})
            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return text_splitter.split_documents(documents)

    
    def build_vector_store(self):
        """Build and save FAISS vector store (LangChain Structure)"""
        with open(self.faq_filename, 'r') as f:
            json_data = json.load(f)
        
        documents = self.process_documents(json_data)
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        
        # save to local directory
        self.vector_store.save_local(folder_path=self.vectorstore_dir, index_name=self.vectorstore_name)
    
    def load_vector_store(self):
        """Load vector store using LangChain load_local"""
        self.vector_store = FAISS.load_local(folder_path=self.vectorstore_dir, 
                                            index_name=self.vectorstore_name, 
                                            embeddings=self.embedding_model,
                                            allow_dangerous_deserialization=True)

    async def _get_relevant_context(self, query: str) -> List[Document]:
        """Get relevant context using both vector store and MCP"""
        try:
            query_keywords = self.normalize_keywords(self.extract_keywords(query))
            print("\nQuery Keywords:", query_keywords)

            # 1. Search in the Vector store
            try:
                vector_docs = self.vector_store.similarity_search(query, k=10)
                print(f"Found {len(vector_docs)} documents from vector store")

                vector_docs = [
                    doc for doc in vector_docs
                    if doc.metadata and self.fuzzy_metadata_match(doc, query_keywords)
                ]
                print(f"Filtered vector store docs to {len(vector_docs)} using metadata keywords")

            except Exception as e:
                print(f"Vector store search failed: {e}")

            # 2. Search in the MCP server
            mcp_docs = []
            try:
                if not self.mcp_client:
                    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
                    self.mcp_client = MCPClient(session=session)
                
                mcp_docs = await self.mcp_client.search(query, program="ms", level="graduate")
                if not mcp_docs:  # If search fails, try requirements
                    mcp_docs = await self.mcp_client.get_requirements(program="ms", level="graduate")
                print(f"Found {len(mcp_docs)} documents from MCP")
            except Exception as e:
                print(f"MCP search failed: {e}")

            # 3. DuckDuckGo Web Research - consider as Deep Research Function
            """
            web_docs = []
            try:
                web_docs = self.web_retriever.get_relevant_documents(query)
                print(f"Found {len(web_docs)} documents from DuckDuckGo")
            except Exception as e:
                print(f"Web search failed: {e}")
            """

            # 4. CS web retriever
            try:
                scope_url = self.cs_retriever.infer_scope(query)
                if not scope_url.startswith("http"):
                    scope_url = "https://" + scope_url
                cs_docs = self.cs_retriever.crawl_and_chunk(scope_url)
                print(f"Found {len(cs_docs)} documents from CS Web Retriever")
            except Exception as e:
                print(f"CS site web retriever failed: {e}")


            # 4. Integrate result and Remove duplicates
            # all_docs = vector_docs + mcp_docs + web_docs + cs_docs
            all_docs = vector_docs + mcp_docs + cs_docs
            
            if not all_docs:
                print("No documents found from either source")
                return [Document(page_content=self.default_responses["unknown"].format(query))]

            # check
            # print("All Retrieval Docs")
            # self._check_retrival_docs(all_docs)

            unique_docs = self._remove_duplicates(all_docs)
            print(f"Total unique documents: {len(unique_docs)}")

            unique_docs = self.rerank_documents(query, query_keywords, unique_docs)
            # self._check_retrival_docs(unique_docs)
            
            return unique_docs

        except Exception as e:
            print(f"Error in _get_relevant_context: {e}")
            return []
    from rapidfuzz import fuzz

    def fuzzy_metadata_match(self, doc: Document, keywords: List[str], threshold: int = 80) -> bool:
        """Returns True if any keyword fuzzy-matches metadata value above a similarity threshold."""
        for key, value in (doc.metadata or {}).items():
            if isinstance(value, str):
                for kw in keywords:
                    similarity = fuzz.partial_ratio(kw.lower(), value.lower())
                    if similarity >= threshold:
                        return True
        return False

    def normalize_keywords(self, keywords: List[str]) -> List[str]:
        replacements = {
            "masters": "ms",
            "master": "ms",
            "undergraduate": "undergrad",
            "ugrad": "undergrad",
            "nlp": "natural language processing",
            "ml": "machine learning",
            "foundation": "foundations of computer science",
            "foundations": "foundations of computer science",
            "security": "computer security",
            "cv": "vision graphics",
            "vision": "vision graphics",
            "graphics": "vision graphics",
            "software": "software systems",
            "network": "network systems",
            "thesis": "ms thesis",
            "personalized": "ms personalized",
            "sw": "software systems"
        }

        normalized = []
        for kw in keywords:
            kw_clean = kw.replace(" ", "").lower()
            normalized.append(replacements.get(kw_clean, kw_clean))
        return list(set(normalized))  # remove duplicates

    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        text = re.sub(r'[^\w\s]', '', text.lower())

        vec = TfidfVectorizer(stop_words='english')
        X = vec.fit_transform([text])
        keywords = sorted(
            zip(vec.get_feature_names_out(), X.toarray()[0]),
            key=lambda x: x[1],
            reverse=True
        )
        filtered = [k for k, _ in keywords if len(k) > 2]

        must_include = []
        if re.search(r"\bms\b", text): must_include.append("ms")
        if re.search(r"\bphd\b", text): must_include.append("phd")
        if re.search(r"\bundergrad\b|\bundergraduate\b", text): must_include.append("undergrad")

        final = list(dict.fromkeys(must_include + filtered))  # preserves order, avoids duplicates
        return final[:top_k]

    
    def rerank_documents(self, query: str, keywords: List[str], docs: List[Document], top_k: int = 5):
        query_emb = self.embedding_model.embed_query(query)
        doc_embs = [self.embedding_model.embed_documents([doc.page_content])[0] for doc in docs]

        doc_scores = []
        for doc, emb in zip(docs, doc_embs):
            sim = cosine_similarity([query_emb], [emb])[0][0]

            # Metadata matching
            metadata = doc.metadata or {}
            meta_text = ' '.join([str(v).lower() for v in metadata.values()])

            # Page content keyword matching
            page_text = doc.page_content.lower()

            # calculate Fuzz match
            fuzzy_meta_score = max(
                [fuzz.partial_ratio(kw, meta_text) for kw in keywords] or [0]
            )
            fuzzy_page_score = max(
                [fuzz.partial_ratio(kw, page_text) for kw in keywords] or [0]
            )

            # Normalize fuzzy scores to 0~1 and weight
            fuzzy_meta_bonus = (fuzzy_meta_score / 100) * 0.05
            fuzzy_page_bonus = (fuzzy_page_score / 100) * 0.1

            final_score = sim + fuzzy_meta_bonus + fuzzy_page_bonus
            doc_scores.append((doc, final_score))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in doc_scores[:top_k]]
    
    def build_conversational_chain(self):
        return RunnableSequence(
            steps=[
                self.memory,  # Memory injection
                self._get_relevant_context,  # Retrieval step
                lambda inputs: self._build_prompt(inputs["query"], inputs["context"]),  # Prompt builder
                self.llm,  # LLM call (ChatOllama)
                StrOutputParser(),  # Parse to string if needed
            ]
        )

    def _build_prompt(self, query: str, context_docs: List[Document], history_str) -> str:
        """Build prompt with clear guidelines and structure"""
        context_str = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt_template = f"""
        You are a Computer Science Advising assistant for Columbia University's CS Advising team.
        Your task is to help students understand course requirements and make informed decisions about their academic path.

        CONTEXT:
        {context_str}

        CHAT HISTORY:
        {history_str}


        USER QUESTION:
        {query}

        GUIDELINES FOR YOUR RESPONSE:
        1. Be concise and clear (3-5 sentences per section)
        2. Only mention course codes that appear in the context
        3. Structure your response as follows:
        - Core Requirements (if any)
        - Track-Specific Requirements (if any)
        - Electives or Additional Courses (if any)
        4. If information is incomplete or unclear, recommend contacting:
        - For Undergraduate: ug-advising@cs.columbia.edu
        - For MS/Bridge: ms-advising@cs.columbia.edu
        - For PhD: phd-advising@cs.columbia.edu
        - For Career: career@cs.columbia.edu

        Remember: Only provide information that is directly supported by the context. Do not make assumptions or add information that isn't explicitly mentioned."""

        return prompt_template

    @time_model_call
    async def chat(self, query: str) -> str:
        try:
            # [add] update memory with the user's question
            self.memory.chat_memory.add_user_message(query)

            # Get relevant context
            context_docs = await self._get_relevant_context(query)

            chat_history = self.memory.chat_memory.messages
            history_str = "\n".join([f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in chat_history])

            
            # Build prompt with context
            prompt = self._build_prompt(query, context_docs, history_str)
            
            # Get response from model using LangChain
            response = ""
            async for chunk in self.llm.astream([{"role": "user", "content": prompt}]):
                if chunk.content:
                    response += chunk.content
            
            if not response.strip():
                return self.default_responses["error"]
            
            # Clean and verify response
            cleaned_response = self._clean_response(response)
            # confidence = self._verify_response(cleaned_response, context_docs)
            self.memory.chat_memory.add_ai_message(cleaned_response)

            # Format final response
            # final_response = f"{cleaned_response}\n\nResponse Confidence: {confidence:.1f}%"
            final_response = f"{cleaned_response}\n\n"
            return final_response
        
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return f"Error processing request: {str(e)}"

    def _clean_response(self, response: str) -> str:
        # Remove think tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Clean up whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Remove any "Assistant:" or similar prefixes
        response = re.sub(r'^(Assistant:|Bot:|AI:)\s*', '', response, flags=re.IGNORECASE)
        
        return response

    def _remove_duplicates(self, docs: List[Document], threshold: float = 0.9) -> List[Document]:
        """Remove semantically duplicate documents based on similarity threshold"""
        unique_docs = []
        seen_embeddings = []

        for doc in docs:
            doc_embedding = self.embedding_model.embed_documents([doc.page_content])[0]

            is_duplicate = False
            for seen_emb in seen_embeddings:
                sim = cosine_similarity([doc_embedding], [seen_emb])[0][0]
                if sim > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_embeddings.append(doc_embedding)

        return unique_docs

    def _check_retrival_docs(self, docs, show_meta=True, max_len=1000):
        print("\n🧾 Context Documents to be used:")
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i+1} ---")
            source = doc.metadata.get("source", "unknown").upper()
            print(f"📌 Source: {source}")
            
            if show_meta:
                meta_str = ", ".join(f"{k}: {v}" for k, v in doc.metadata.items() if k != "source")
                print(f"🧷 Metadata: {meta_str}")
            
            print("📄 Content Preview:")
            print(doc.page_content[:max_len])


    def _verify_response(self, response: str, context_docs: List[Document]) -> float:
        # 1. Course code-based verification
        course_pattern = r'(?:CS|COMS|MATH|STAT)[\s\-]*\d{4}'
        response_courses = set(re.findall(course_pattern, response, re.IGNORECASE))
        context_text = ' '.join(doc.page_content for doc in context_docs)
        context_courses = set(re.findall(course_pattern, context_text, re.IGNORECASE))

        course_score = 0.0
        if response_courses:
            verified_courses = response_courses.intersection(context_courses)
            course_score = (len(verified_courses) / len(response_courses)) * 100

        # 2. Keyword-based verification
        keywords = [
            "requirement", "track", "elective", "thesis",
            "credit", "advisor", "registration", "course", "project"
        ]
        matched_keywords = [kw for kw in keywords if kw in response.lower()]
        keyword_score = (len(matched_keywords) / len(keywords)) * 100

        # 3. weighted average
        final_score = (course_score * 0.6) + (keyword_score * 0.4)

        return final_score


    async def close(self):
        """Clean up resources"""
        if self.mcp_client and hasattr(self.mcp_client, 'session'):
            await self.mcp_client.session.close()

if __name__ == "__main__":
    import asyncio
    import time
    
    async def main():
        assistant = EnhancedCSAssistant()
        print(assistant.default_responses["greeting"])

        try:
            while True:
                user_input = input("\n👉 Question: ")
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(assistant.default_responses["farewell"])
                    break
                
                print("\n🤖 Processing...")
                start_time = time.time()
                response = await assistant.chat(user_input)
                elapsed_time = time.time() - start_time
                
                print(f"\n⏱️ Response time: {elapsed_time:.2f} seconds\n")
                print(f"🤖 Response: {response}\n")
                print("-" * 80)
        finally:
            if hasattr(assistant, 'close'):
                await assistant.close()
    
    asyncio.run(main()) 