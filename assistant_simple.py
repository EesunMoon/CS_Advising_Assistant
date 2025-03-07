import ollama
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


"""
    [Model]
        - Not Enough GPU: DeepSeek-R1-Distill-Qwen-1.5B (VRAM 4GB+)
        - balanced performance: DeepSeek-R1-Distill-Qwen-7B (VRAM 12GB+)
"""

class CS_Assistant_DeepSeek:
    def __init__(self, model_name="deepseek-r1:1.5b", 
                 embedding_model_name = "all-MiniLM-L6-v2", 
                 dirname = "Dataset/", faq_filename="FAQ.json",
                 index_filename="faq_index.faiss", embeddings_filename="faq_embeddings.npy"):
        print(f"--Load Ollama-based {model_name}--")

        # set filename
        self.model_name = model_name
        self.faq_filename = dirname+faq_filename
        self.index_filename = dirname+index_filename
        self.embedding_filename = dirname+embeddings_filename

        # Load model
        self.llm = ollama.chat
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.faq_data = self.load_FAQ()

        # save messages
        self.messages = []
        self.faq_mappings = {}
        
        # Load Vector DB
        if os.path.exists(self.index_filename) and os.path.exists(self.embedding_filename):
            self.load_faiss_index()
        else:
            self.build_FAQ_index()
    
    def load_FAQ(self):
        if os.path.exists(self.faq_filename):
            with open(self.faq_filename, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}
    
    def build_FAQ_index(self):
        """
            OpenAI vector DB structure:
                {
                    "id": "file_abc123",
                    "vector": [0.12, 0.43, ..., -0.31],  # ebedding
                    "metadata": {
                        "Level": "Undergraduate",
                        "School": "SEAS/Engineering",
                        "Program Code": "EICOMS",
                        "Program": "COMS Minor",
                        "Question": "What are my Degree Requirements?",
                        "Response": "Minor in Computer Science consists of 6 courses as follows:..."
                    }
                }
        """
        questions = [] 
        answers = []
        for data in self.faq_data:
            metadata = f"{data['Level']} - {data['School']} - {data['Program Code']} - {data['Program']}"
            question_text = f"{metadata} - {data['Question']}"
            
            questions.append(question_text)
            answers.append(data['Response'])
            
            self.faq_mappings[question_text] = {
                "metadata":metadata,
                "response": data["Response"]
            }
        
        self.embeddings = self.embed_model.encode(questions, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        faiss.write_index(self.index, self.index_filename)
        np.save(self.embedding_filename, self.embeddings)

    def load_faiss_index(self):
        self.index = faiss.read_index(self.index_filename)
        self.embeddings = np.load(self.embedding_filename)

        for data in self.faq_data:
            metadata = f"{data['Level']} - {data['School']} - {data['Program Code']} - {data['Program']}"
            question_text = f"{metadata} - {data['Question']}"
            
            self.faq_mappings[question_text] = {
                "metadata":metadata,
                "response": data["Response"]
            }

    def search_FAQ(self, query):
        # Simple Search

        if not self.faq_data:
            return None
        
        for ques, ans in self.faq_data.items():
            if query.lower() in ques.lower():
                return ans
    
    def retrieve_FAQ(self, query, top_k=1):
        # Vector Search
        if self.index is None:
            return None, None
        
        # search in FAISS
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        _, indeces = self.index.search(query_embedding, top_k)

        best_match_index = indeces[0][0]
        best_match_question = list(self.faq_mappings.keys())[best_match_index]

        retrieved_data = self.faq_mappings[best_match_question]
        return best_match_question, retrieved_data["metadata"], retrieved_data["response"]
    
    def chat(self, user_input):

        # 1) retreive data in vector db
        # faq_ans = self.search_FAQ(user_input)
        retrieved_question, metadata, retrieved_answer = self.retrieve_FAQ(user_input)
        
        prompt = f"""
                You are a Computer Science Advising assistant for Columbia University's CS Advising team.
                Your response should ONLY be based on the <Retrieved Information> provided below. 
                Do NOT generate information outside of what is retrieved. If the retrieved information does not fully answer, 
                direct the user to the relevant email contact.

                <Retrieved Information>:
                - Matched Question: {retrieved_question}
                - Related Information: {metadata}
                - FAQ Answer: {retrieved_answer}

                <User Question>:
                {user_input}

                **Guidelines:**
                - Keep the answer concise (Max: 3-5 sentences)
                - Do NOT make assumptions or add extra details
                - If the answer is unclear, recommend the relevant email for follow-up:
                    - Undergraduate Student Services: ug-advising@cs.columbia.edu
                    - MS & CS@CU MS Bridge Student Services: ms-advising@cs.columbia.edu
                    - PhD Student Services: phd-advising@cs.columbia.edu
                    - Career Placement Officers: career@cs.columbia.edu
                    """
        

        # 2) Use Deepseek model
        self.messages.append({"role": "user", "content": prompt})

        response = ollama.chat(model=self.model_name, messages=self.messages)
        generated_message = response["message"]["content"]

        # store conversation context history
        self.messages.append({"role": "assistant", "content": generated_message})

        return generated_message

# model load
model_name = "deepseek-r1:1.5b"
def test(model_name):
    response = ollama.chat(model=model_name, 
                        messages=[
                            {"role": "user",
                                "content": "Explain reinforcement learning in simple terms."}
                        ])
    print("DeepSeek Assistant:")
    print(response["message"])

if __name__ == "__main__":
    assistant = CS_Assistant_DeepSeek()

    while True:
        user_input = input("\n 👉🏻Ask a Questions (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        
        response = assistant.chat(user_input)
        print(f"\n 🧐DeepSeek Assistant: \n{response}")