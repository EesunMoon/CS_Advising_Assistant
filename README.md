# 🎓 CS Advising Assistant (LangChain + DeepSeek + RAG)

## 🚀 Project Overview
This project replaces **OpenAI API** with **Ollama-powered DeepSeek models**, integrating **LangChain** to build an AI-driven **CS Advising Assistant** using **RAG (Retrieval-Augmented Generation)**.

- **No OpenAI API Calls** → Runs entirely on **Ollama-powered DeepSeek models**
- **LangChain Integration** → Context retrieval and RAG-based response generation
- **FAISS for Efficient Search** → Vector embedding-based document retrieval
- **Conversation Memory** → Maintains context for better responses
- **Prompt Tuning** → Enhances response quality


## 🛠 Tech Stack
### **LLM Execution**
- **Ollama** → Runs **DeepSeek models** locally  
- **DeepSeek-Chat (via Ollama)** → Generates responses without API costs  
- **LangChain ConversationalRetrievalChain** → Enables context-aware responses  

### **Vector Embedding & Retrieval**
- **FAISS** → Efficient similarity search for FAQ retrieval  
- **Hugging Face Sentence Embeddings** → Replaces OpenAI’s embedding API  

### **RAG Implementation**
- **Retrieves relevant FAQ entries**  
- **Augments context** before generating responses  
- **Optimized prompt tuning for response generation**  

### **Out-of-Scope Query Handling**
- **Vector embedding clustering** to detect irrelevant queries  

## 🔍 How It Works
1. Retrieves the most relevant FAQ entries using FAISS & DeepSeek Embeddings
2. Retrieves additional context using LangChain Retriever
3. Optimizes prompts for better responses using Prompt Tuning
4. Generates responses using DeepSeek LLM via Ollama
5. Detects out-of-scope queries using vector embedding clustering
