import warnings
warnings.simplefilter("ignore")  # Ignore all warnings

# Existing imports
import os
import pdfplumber
import pytesseract
import concurrent.futures
from dotenv import load_dotenv
from pdf2image import convert_from_path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document
from huggingface_hub.utils import _deprecation

# Suppress specific deprecation warnings if needed
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="huggingface_hub.utils._deprecation")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Load environment variables
load_dotenv('.env')

# Set Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"  # Update path if needed
OCR_CONFIG = "--psm 6"

def extract_text_from_pdf(pdf):
    """Extract text from PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf)
    return loader.load()

def extract_tables_from_pdf(pdf):
    """Extract tables from PDF using pdfplumber."""
    structured_tables = []
    with pdfplumber.open(pdf) as pdf_doc:
        for page in pdf_doc.pages:
            tables = page.extract_tables()
            for table in tables:
                structured_tables.append({
                    "page": page.page_number,
                    "table": table
                })
    return structured_tables

def extract_text_from_images(pdf):
    """Extract text from images in PDF using OCR."""
    images = convert_from_path(pdf)
    return [
        Document(page_content=pytesseract.image_to_string(img, config=OCR_CONFIG), metadata={"source": pdf})
        for img in images if pytesseract.image_to_string(img, config=OCR_CONFIG).strip()
    ]

def load_pdfs_parallel(pdf_files):
    """Load and process PDFs in parallel."""
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(extract_text_from_pdf, pdf): 'text' for pdf in pdf_files
        }
        futures.update({
            executor.submit(extract_tables_from_pdf, pdf): 'table' for pdf in pdf_files
        })
        futures.update({
            executor.submit(extract_text_from_images, pdf): 'image' for pdf in pdf_files
        })

        for future in concurrent.futures.as_completed(futures):
            result_type = futures[future]
            try:
                result = future.result()
                if result_type == 'text':
                    documents.extend(result)
                elif result_type == 'table':
                    documents.extend([
                        Document(page_content="\n".join([" | ".join(filter(None, row)) for row in table["table"] if row]),
                                 metadata={"source": table["page"]})
                        for table in result
                    ])
                elif result_type == 'image':
                    documents.extend(result)
            except Exception as e:
                print(f"Error processing {result_type}: {e}")
    return documents

def chunk_and_embed(docs):
    """Chunk documents and create embeddings."""
    token_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = token_splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("faiss_index_embed")
    return vectorstore

def llm_model(repo_id, task):
    """Initialize LLM model."""
    return HuggingFaceEndpoint(repo_id=repo_id, max_length=2000, temperature=0.1, task=task, token=os.getenv('API_KEY'))

def sim_search_rag(query, llm, retriever):
    """Perform similarity search and answer queries using RAG."""
    docs_for_query = retriever.get_relevant_documents(query, k=12)

    context = "\n".join([doc.page_content for doc in docs_for_query])

    final_prompt_temp="""You are an AI assistant. Below is the provided context that should guide your response. Please ensure that your answer is based entirely on the information within the context. Avoid including any external information or assumptions.
    Context: {context}
    Question: {question}
    Answer:"""

    prompt_template = PromptTemplate(input_variables=["context", "question"], template=final_prompt_temp)
    chain = LLMChain(llm=llm, prompt=prompt_template)
    return chain.run({"context": context, "question": query})

# %%
def chatbot_session(llm, retriever):
    """Start chatbot session."""
    while True:
        query = input("Enter your query (or type 'quit' to exit): ").strip()
        if query.lower() == "quit":
            print("Chatbot session ended.")
            break
        try:  
            print(sim_search_rag(query, llm, retriever)+"\n")
        except Exception as e:
            print(f"Error processing query: {e}")

# %%
pdf_files = [
    "VISA_D_Payments.pdf",
    "HDFC_Credit_Cards_B_MITC.pdf",
    "Deloitte_C_Financial_Outlook.pdf",
    "Credit_Cards_A_Sectoral_Analysis.pdf"
]

# %%
index_folder = "faiss_index_embed"
if not os.path.exists(index_folder):
    print(f"Folder '{index_folder}' not found. Running embedding pipeline...")
    docs = load_pdfs_parallel(pdf_files)
    vectorstore = chunk_and_embed(docs)
else:
    print(f"Folder '{index_folder}' already exists. Skipping embedding pipeline.")


# %%
llm = llm_model("mistralai/Mistral-7B-Instruct-v0.2", "text-generation")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local("faiss_index_embed", embedding_model, allow_dangerous_deserialization=True).as_retriever()

# %%
def main():
    chatbot_session(llm, retriever)

if __name__ == "__main__":
    main()
