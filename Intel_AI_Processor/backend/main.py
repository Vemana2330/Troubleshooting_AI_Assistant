from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient  
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper 

# Load environment variables
load_dotenv()

# Retrieve API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")  # SerpAPI API Key

# Initialize FastAPI app
app = FastAPI()

# Initialize OpenAI LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone Client
pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

# Ensure Index Exists
if PINECONE_INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(PINECONE_INDEX_NAME, dimension=1536, metric="cosine")

# Load Existing Pinecone Index
index = Pinecone.from_existing_index(PINECONE_INDEX_NAME, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

# Define the PDF document path
PDF_FILE_PATH = "/Users/vemana/Documents/Intel_AI_Processor/backend/document.pdf"

def load_and_index_pdf():
    """Loads the Intel Xeon 6 document, converts to embeddings, and stores in Pinecone."""
    try:
        loader = PyPDFLoader(PDF_FILE_PATH)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        print(f"🔍 Number of Docs to Index: {len(docs)}")
        if not docs:
            return "⚠ No documents found in PDF."

        # Extract text content
        texts = [doc.page_content for doc in docs]

        # Store indexed data into Pinecone
        index.add_texts(texts)

        print(f"✅ Indexed {len(texts)} documents into Pinecone.")
        return "✅ PDF indexed successfully."
    except Exception as e:
        print(f"❌ Error in load_and_index_pdf: {str(e)}")
        return f"Error: {str(e)}"

@app.post("/train_llm")
def train_llm():
    """Trains the LLM by loading and indexing the Intel Xeon 6 PDF."""
    try:
        message = load_and_index_pdf()
        print(f"Training Message: {message}")
        return {"message": message}
    except Exception as e:
        print(f"Error in train_llm: {str(e)}")
        return {"error": str(e)}

class QueryRequest(BaseModel):
    query: str

@app.post("/query_document")
def query_document(request: QueryRequest):
    """Retrieves answers based on the uploaded document."""
    try:
        print(f"🔍 Received Query: {request.query}")  

        vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        retriever = vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.run(request.query)

        print(f"✅ AI Response: {response}")  
        return {"response": response}
    except Exception as e:
        print(f"❌ Query Error: {str(e)}")  
        return {"error": str(e)}

class IssueRequest(BaseModel):
    issue: str

@app.post("/analyze")
def analyze_issue(request: IssueRequest):
    """Uses AI to analyze and troubleshoot Intel product issues using SerpAPI and GPT."""
    try:
        print(f"🔍 Received Issue: {request.issue}")  

        # Fixed Timeout Validation Issue
        search_tool = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)

        # SerpAPI for fetching troubleshooting solutions
        tools = [
            Tool(
                name="Web Search",
                func=search_tool.run,
                description="Use this tool to fetch Intel troubleshooting solutions from the web."
            )
        ]

        # Create LangChain Agent (LLM + SerpAPI)
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # Structured Query for SerpAPI
        query = f"Intel {request.issue} troubleshooting solution site:intel.com OR site:reddit.com/r/intel"
        web_results = search_tool.run(query)

        if not web_results or web_results.strip() == "":
            print("⚠ No web solutions found.")
            return {"solution": "No solution found. Try rephrasing the issue or checking Intel's official support pages: [Intel Support](https://www.intel.com/content/www/us/en/support.html)"}

        # Generate troubleshooting solution using AI
        prompt = f"""
        Based on the following troubleshooting web results, summarize the best solution for the given Intel product issue.
        If no exact solution is found, suggest the best possible alternative.

        **Issue:** {request.issue}

        **Web Results:**
        {web_results}

        **Format your response with:**
        1. 🔧 **Possible Causes**
        2. 🛠 **Step-by-Step Fixes**
        3. ⚠️ **Precautions**
        4. 📌 **Additional Resources**
        """
        ai_solution = agent.run(prompt)

        print(f"✅ AI Solution: {ai_solution}")  # Debugging
        return {"solution": ai_solution}
    except Exception as e:
        print(f"❌ Troubleshooting Error: {str(e)}")  # Debugging
        return {"error": str(e)}
