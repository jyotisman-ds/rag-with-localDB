import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


def main():
    """main function to execute"""
    print("Hello from rag-with-localdb!")
    # llm = ChatOpenAI()
    path_to_pdf = "2210.03629v3.pdf"
    pdf_loader = PyPDFLoader(file_path=path_to_pdf)
    pdf_text = pdf_loader.load()  # Load the PDF file (does not do any chunking)

    # Split the text into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = splitter.split_documents(documents=pdf_text)

    embeddings = None
    if os.getenv("USE_OLLAMA") == "YES":
        embeddings = OllamaEmbeddings(model="deepseek-r1:14b")
    else:
        embeddings = OpenAIEmbeddings()
    # vectorize and store the vectors in-memory
    vectorstore = None
    if not os.path.exists("faiss_index_react") and embeddings is not None:
        vectorstore = FAISS.from_documents(docs, embeddings)
    if vectorstore is not None:
        # persist for later use
        vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )

    res = retrieval_chain.invoke(
        {"input": "Describe the principles of ReAct in 3 sentences"}
    )
    print(res["answer"])


if __name__ == "__main__":
    main()
