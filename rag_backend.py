import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_DIR = "chroma_db"
DATA_PATH = "data/knowledge_base.txt"

# In-memory dictionary to store session-specific chat history logs
sessions_db = {}

def get_session_history(session_id: str):
    if session_id not in sessions_db:
        sessions_db[session_id] = ChatMessageHistory()
    return sessions_db[session_id]

def initialize_vector_db():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Please create the {DATA_PATH} file first.")
        
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vector_store

def get_rag_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not os.path.exists(DB_DIR):
        vector_store = initialize_vector_db()
    else:
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Contextual Prompt Layout
    contextual_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional customer support agent for ACME Corp.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say exactly: "I am sorry, I do not have that information in my knowledge base."
Do not make things up. Keep your response concise.

         CRITICAL FORMATTING RULE: Output your answer as plain text only. Never wrap prices, numbers, or phrases inside backticks (`) or code blocks.
Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # FIX: We use itemgetter or a lambda to explicitly isolate the string 'question' 
    # before sending it to the vector retriever.
    core_rag_chain = RunnableParallel({
        "sources": (lambda x: x["question"]) | retriever,
        "answer": (
            {
                "context": (lambda x: x["question"]) | retriever | format_docs, 
                "question": lambda x: x["question"], 
                "chat_history": lambda x: x["chat_history"]
            }
            | contextual_prompt
            | llm
            | StrOutputParser()
        )
    })
    
    # Wrap the core RAG chain with stateful message history tracking
    conversational_rag_chain = RunnableWithMessageHistory(
        core_rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_rag_chain