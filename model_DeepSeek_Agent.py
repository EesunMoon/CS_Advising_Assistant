import os
import json
import re
import asyncio
import torch
import aiohttp
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from utils import time_model_call, TimingStats
from mcp_client import MCPClient
from cs_web_retriever import CSProgramWebRetriever
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

class CSAssistantAgent:
    def __init__(self, 
                 model_name="deepseek-r1:1.5b",
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 dirname="Dataset/",
                 faq_filename="FAQ_modified.json",
                 vectorstore_name="faq_modified"):

        self.model_name = model_name
        self.faq_filename = os.path.join(dirname, faq_filename)
        self.vectorstore_dir = dirname
        self.vectorstore_name = vectorstore_name
        self.context_window = 4096

        self.llm = ChatOllama(model=model_name, context_window=self.context_window)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.mcp_client = None
        self.cs_retriever = CSProgramWebRetriever(k=5, use_chunking=True)
        self.timing_stats = TimingStats()

        self.default_responses = {
            "greeting": "Hello! I'm your CS department academic advisor. How can I help you today?",
            "farewell": "Thank you!",
            "unknown": "I understand you're asking about {}, but I need more specific information. Could you please clarify your question?",
            "error": "Sorry, I'm having trouble processing that request. Could you rephrase?"
        }

        if os.path.exists(os.path.join(self.vectorstore_dir, f"{self.vectorstore_name}.faiss")):
            self.load_vector_store()
        else:
            self.build_vector_store()

    def process_documents(self, json_data: List[Dict[str, Any]]) -> List[Document]:
        docs = []
        for item in json_data:
            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            docs.append(Document(page_content=content, metadata=item.get("metadata", {})))
        return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

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
        try:
            keywords = self.normalize_keywords(self.extract_keywords(query))
            vector_docs = self.vector_store.similarity_search(query, k=10)
            vector_docs = [doc for doc in vector_docs if self.fuzzy_metadata_match(doc, keywords)]

            mcp_docs = []
            if not self.mcp_client:
                session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False))
                self.mcp_client = MCPClient(session=session)
            mcp_docs = await self.mcp_client.search(query, program="ms", level="graduate")
            if not mcp_docs:
                mcp_docs = await self.mcp_client.get_requirements(program="ms", level="graduate")

            cs_docs = []
            try:
                scope_url = self.cs_retriever.infer_scope(query)
                if not scope_url.startswith("http"):
                    scope_url = "https://" + scope_url
                cs_docs = self.cs_retriever.crawl_and_chunk(scope_url)
            except:
                pass

            all_docs = vector_docs + mcp_docs + cs_docs
            return self._remove_duplicates(self.rerank_documents(query, keywords, all_docs))
        except:
            return []

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
    
    # Clarification Logic
    def _needs_clarification(self, query: str) -> bool:
        non_question_patterns = [
            r"^(hi|hello|hey)\b",
            r"^(thank you|thanks|bye)\b",
            r"^(i'?m a|i am a)\s+(student|phd|ms|undergrad)",
            r"^(can you help|who are you)",
        ]
        return any(re.match(p, query.lower().strip()) for p in non_question_patterns)

    def needs_clarification(self, query: str, context_docs: List[Document], threshold: int = 1) -> bool:
        """Return True if context is too sparse or irrelevant"""
        if not context_docs:
            return True

        match_count = sum([
            any(kw.lower() in doc.page_content.lower() for kw in self.extract_keywords(query))
            for doc in context_docs
        ])
        return match_count < threshold

    def generate_clarification_question(self, query: str) -> str:
        if "track" in query.lower():
            return "Are you asking about MS tracks (like Machine Learning, Systems, etc.) or PhD specialization areas?"
        elif "requirement" in query.lower():
            return "Do you mean course requirements for a specific track or general graduation requirements?"
        else:
            return "Could you clarify your question so I can give you the most relevant information?"


    @time_model_call
    async def chat(self, query: str) -> str:
        """ memory -> context -> prompt -> model -> response """
        try:
            # 1. memory update
            self.memory.chat_memory.add_user_message(query)

            # 2. clarification logic
            if self._needs_clarification(query):
                clarification = (
                    "Hi! Could you tell me more about what you're looking for? "
                    "For example, are you asking about course requirements, track options, or application info?"
                )
                self.memory.chat_memory.add_ai_message(clarification)
                return clarification

            # 3. get context
            context_docs = await self._get_relevant_context(query)

            # 4. history string
            chat_history = self.memory.chat_memory.messages
            history_str = "\n".join([
                f"User: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}"
                for m in chat_history
            ])

            # 5. build prompt
            prompt = self._build_prompt(query, context_docs, history_str)

            # 6. run LLM
            response = ""
            async for chunk in self.llm.astream([{"role": "user", "content": prompt}]):
                if chunk.content:
                    response += chunk.content

            cleaned = self._clean_response(response)
            self.memory.chat_memory.add_ai_message(cleaned)

            return cleaned + "\n\n"
        
        except Exception as e:
            print(f"Agentic chat error: {e}")
            return f"Error processing request: {str(e)}"


    def extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        vec = TfidfVectorizer(stop_words='english')
        X = vec.fit_transform([re.sub(r'[^\w\s]', '', text.lower())])
        return [k for k, _ in sorted(zip(vec.get_feature_names_out(), X.toarray()[0]), key=lambda x: x[1], reverse=True) if len(k) > 2][:top_k]

    def normalize_keywords(self, keywords: List[str]) -> List[str]:
        replace = {"masters": "ms", "foundation": "foundations of computer science", "sw": "software systems"}
        return list(set([replace.get(k.replace(" ", "").lower(), k.lower()) for k in keywords]))

    def fuzzy_metadata_match(self, doc: Document, keywords: List[str], threshold: int = 80) -> bool:
        for val in doc.metadata.values():
            if isinstance(val, str):
                if any(fuzz.partial_ratio(kw, val.lower()) >= threshold for kw in keywords):
                    return True
        return False

    def rerank_documents(self, query: str, keywords: List[str], docs: List[Document], top_k: int = 5):
        query_emb = self.embedding_model.embed_query(query)
        doc_embs = [self.embedding_model.embed_documents([doc.page_content])[0] for doc in docs]
        scores = []
        for doc, emb in zip(docs, doc_embs):
            sim = cosine_similarity([query_emb], [emb])[0][0]
            meta_text = ' '.join(str(v).lower() for v in doc.metadata.values())
            page_text = doc.page_content.lower()
            fuzzy_meta = max([fuzz.partial_ratio(k, meta_text) for k in keywords] or [0]) / 100 * 0.05
            fuzzy_page = max([fuzz.partial_ratio(k, page_text) for k in keywords] or [0]) / 100 * 0.1
            scores.append((doc, sim + fuzzy_meta + fuzzy_page))
        return [doc for doc, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]]

    def _remove_duplicates(self, docs: List[Document], threshold: float = 0.9) -> List[Document]:
        unique, seen = [], []
        for doc in docs:
            emb = self.embedding_model.embed_documents([doc.page_content])[0]
            if all(cosine_similarity([emb], [e])[0][0] <= threshold for e in seen):
                unique.append(doc)
                seen.append(emb)
        return unique

    def _clean_response(self, response: str) -> str:
        return re.sub(r'<[^>]+>', '', re.sub(r'\s+', ' ', response)).strip()

    async def close(self):
        if self.mcp_client and hasattr(self.mcp_client, 'session'):
            await self.mcp_client.session.close()

if __name__ == "__main__":
    import asyncio
    import time
    
    async def main():
        assistant = CSAssistantAgent()
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