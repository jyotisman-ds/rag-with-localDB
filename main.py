from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
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


if __name__ == "__main__":
    main()
