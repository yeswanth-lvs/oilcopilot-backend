from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.docstore.document import Document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: add a test root endpoint
@app.get("/")
def home():
    return {"status": "Backend is running"}

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    return {"chunks": chunks[:3]}  # limit for demo/testing

@app.post("/summary")
async def summarize(chunks: list[str]):
    prompt = (
        "You are an oilfield report assistant. Summarize the following chunks with key failures, NPTs, and lessons:\n\n"
        + "\n".join(chunks)
    )
    chat = ChatOpenAI(model="gpt-4", temperature=0)
    response = chat([HumanMessage(content=prompt)])
    return {"summary": response.content}

@app.post("/ask")
async def ask_question(query: str = Form(...), chunks: list[str] = Form(...)):
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    retriever = vectordb.as_retriever()
    qa = ChatOpenAI(model="gpt-4", temperature=0)
    context = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(query)])
    result = qa([HumanMessage(content=f"Use this context to answer: {query}\n\n{context}")])
    return {"answer": result.content}
