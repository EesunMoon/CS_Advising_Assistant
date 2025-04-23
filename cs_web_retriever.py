from bs4 import BeautifulSoup
import requests
from markdownify import markdownify as md
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from duckduckgo_search import DDGS
from langchain.docstore.document import Document


class CSProgramWebRetriever:
    def __init__(self, k=5, use_chunking=True):
        self.k = k
        self.use_chunking = use_chunking
        self.program_sites = {
            "undergraduate": "cs.columbia.edu/education/undergraduate",
            "phd": "cs.columbia.edu/education/phd",
            "ms": "cs.columbia.edu/education/ms",
        }
        self.ms_track_sites = {
            "computational biology": "cs.columbia.edu/education/ms/computationalBiology/",
            "computer security": "cs.columbia.edu/education/ms/newComputerSecurity/",
            "foundations of computer science": "cs.columbia.edu/education/ms/foundationsOfCS/",
            "machine learning": "cs.columbia.edu/education/ms/machineLearning/",
            "natural language processing": "cs.columbia.edu/education/ms/nlp/",
            "network systems": "cs.columbia.edu/education/ms/networkSystems/",
            "software systems": "cs.columbia.edu/education/ms/softwareSystems/",
            "vision graphics": "cs.columbia.edu/education/ms/visionAndGraphics/",
            "ms personalized": "cs.columbia.edu/education/ms/MSpersonalized/",
            "ms thesis": "cs.columbia.edu/education/ms/MSThesis/",
        }
        self.track_aliases = {
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

    def infer_scope(self, query: str) -> str:
        q = query.lower()
        if "phd" in q:
            return self.program_sites["phd"]
        if any(w in q for w in ["undergrad", "undergraduate", "ba"]):
            return self.program_sites["undergraduate"]
        
        # Check aliases first
        for alias, full_track in self.track_aliases.items():
            if alias in q:
                return self.ms_track_sites.get(full_track, self.program_sites["ms"])

        # Then check full track names
        for track, url in self.ms_track_sites.items():
            if track in q:
                return url
        return self.program_sites["ms"]

    
    def get_relevant_documents(self, query: str):
        scope = self.infer_scope(query)
        full_query = f"{query} site:{scope}"
        print(f"[DuckDuckGoRetriever] Query: {full_query}")

        documents = []

        with DDGS() as ddgs:
            results = ddgs.text(full_query, max_results=self.k)

            for r in results or []:
                if self.use_chunking:
                    chunks = self.crawl_and_chunk(r['href'])
                    documents.extend(chunks)
                else:
                    content = f"{r['title']}\n{r['body']}\n{r['href']}"
                    documents.append(Document(page_content=content, metadata={"source": r["href"], "scope": scope}))

        return documents


    def crawl_and_chunk(self, url: str):
        try:
            res = requests.get(url, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")

            # 🎯 STEP 1: Main content only (columbia CS uses <div class="entry-content">)
            main = soup.find("div", class_="entry-content")
            if not main:
                main = soup.body  # fallback

            # 🎯 STEP 2: Clean and markdown
            markdown = md(str(main))

            # 🎯 STEP 3: Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
            chunks = splitter.split_text(markdown)

            return [Document(page_content=c, metadata={"source": url}) for c in chunks]

        except Exception as e:
            print(f"[crawl_and_chunk] Failed for {url}: {e}")
            return []


if __name__ == "__main__":
    retriever = CSProgramWebRetriever()
    url = "https://www.cs.columbia.edu/education/ms/machineLearning/"
    
    print(f"🕸️ Crawling and chunking: {url}")
    chunks = retriever.crawl_and_chunk(url)
    
    print(f"\n✅ Total chunks extracted: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:5]): 
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content[:1000])
        print("\n")
