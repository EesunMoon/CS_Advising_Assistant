# CS Advising Assistant
A chatbot system using LLMs to assist in Computer Science advising.

## Workflow

1. **User Data Collection**

Collect user inputs.

2. **Knowledge Base Retrieval**

Convert a CSV file into a Knowledge Base.

3. **Use Vector Embedding (DeepSeek).**

Apply FAISS Similarity Search to retrieve the top-k most relevant entries.

4. **Prompt Construction**

Construct prompts using user data, the user’s question, and relevant FAQ entries.

5. **Response Generation**

Send the prompt to Ollama LLM.

Process the generated response with Out-of-Scope Query Detection.


## Technologies Stack

### Chatbot Deployment

- Cloud infrastructure: GCP, Azure, AWS

- Performance Logging: Caching to reduce API call costs

### Embedding

- DeepSeek Vector Embedding (previously OpenAI GPT-4o)

- FAISS: Similarity-based search

### LLM

DeepSeek LLM, LLaMA

### Out-of-Scope Query Handling

Vector embedding Clustering
