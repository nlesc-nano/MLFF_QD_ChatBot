import chainlit as cl
import os
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

# NOTE: Ensure you have the 'langchain_classic' package or folder in your project
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever 
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory


from config import config

@cl.cache
def setup_retriever():
    """Builds a Hybrid Retriever (Vector + Keyword) + Re-Ranker."""
    documents = []
    
    data_paths = config.get_data_paths

    # Check if data directory exists
    if not os.path.exists(config.DATA_DIR):
         os.makedirs(config.DATA_DIR)
         print(f"Created missing directory: {config.DATA_DIR}. Please place files there.")
         return None

    # Load text files
    for file_path in data_paths:
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
            
        try:
            if file_path.endswith(".txt"):
                documents.extend(TextLoader(file_path, encoding="utf-8").load())
            elif file_path.endswith(".docx"):
                documents.extend(Docx2txtLoader(file_path).load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Check if DB exists to avoid re-ingesting if possible (though Chroma handles persistence)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=config.DB_PATH)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": config.SEARCH_K}) 

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = config.SEARCH_K

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.5, 0.5]
    )

    compressor = FlashrankRerank(top_n=config.RERANK_TOP_N)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    return compression_retriever

@cl.on_chat_start
async def on_chat_start():
    try:
        retriever = setup_retriever()
        if not retriever:
            await cl.Message("❌ No documents found! Please check your 'data' folder.").send()
            return

        if not config.GROQ_API_KEY:
             await cl.Message("❌ GROQ_API_KEY not found in environment variables.").send()
             return

        llm = ChatGroq(
            api_key=config.GROQ_API_KEY, 
            model=config.MODEL_NAME,
            temperature=0.3
        )
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are the official MLFF_QD Expert Guide.
        Your knowledge comes from three sources provided in the Context below:
        1. **Technical Manual:** Contains raw code, file paths, and CLI commands.
        2. **Scientific Paper:** Contains theory, motivation, and benefits of the platform.
        3. **Q&A Database:** Contains verified answers to common questions.

        ### INSTRUCTIONS:

        **1. FOR CONFIGURATION & FILES:**
           - If the user asks to "show" or "see" a file (like `schnet.yaml`), output the **RAW YAML/Code block** found in the context.
           - If asked for a **Location**, look for the line `**Exact Location:**` in the manual and provide the full path.
           - Use exact variable names (e.g., `n_layers`, `r_cut`) as they appear in the code.

        **2. FOR SCIENTIFIC & THEORY QUESTIONS:**
           - If the user asks "Why" or about "Benefits" (e.g., "Why use MLFF?", "Theory of Quantum Dots"), use the **Scientific Paper** content to explain the reasoning.

        **3. FOR "HOW-TO" & COMMANDS:**
           - Provide clear, numbered steps.
           - Always format commands as code blocks (e.g., `python -m ...`).

        **4. SAFETY:**
           - If the answer is NOT in the context, explicitly say: "I cannot find that information in the manual or paper."
           - Do not hallucinate flags or commands.

        Context:
        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        cl.user_session.set("chain", rag_chain)
        cl.user_session.set("history", ChatMessageHistory())

        await cl.Message("👋 **Expert Bot Ready!** ").send()
    except Exception as e:
        await cl.Message(f"❌ Error starting chat: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    history = cl.user_session.get("history")
    
    if not chain:
        await cl.Message("❌ Chat chain not initialized.").send()
        return

    try:
        res = await chain.ainvoke({
            "input": message.content, 
            "chat_history": history.messages
        })

        history.add_user_message(message.content)
        history.add_ai_message(res["answer"])

        source_elements = []
        seen_sources = set()
        
        if "context" in res:
            for doc in res["context"]:
                source_name = os.path.basename(doc.metadata.get("source", "Unknown"))
                if source_name not in seen_sources:
                    seen_sources.add(source_name)
                    source_elements.append(
                        cl.Text(content=doc.page_content, name=source_name, display="side")
                    )

        await cl.Message(content=res["answer"], elements=source_elements).send()
    except Exception as e:
        await cl.Message(f"❌ Error processing message: {str(e)}").send()