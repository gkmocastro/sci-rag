import os
import typer
import uuid
import chromadb
from dotenv import load_dotenv

from embed_function import embed_documents


load_dotenv()

client = chromadb.HttpClient(host=os.getenv("CHROMA_DB_URL"))


def load_documents(documents_path: str) -> list[str]:
    """
    Loads the contents of Markdown and text files from a specified directory.

    Returns:
        A list of strings, where each string is the content of a document.

    Notes:
        Only files with .md and .txt extensions are loaded.
    """
    documents = list()

    for filename in os.listdir(documents_path): #exercicio: fazer o codigo funcionar para pdf
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            documents.append(content)

    return documents


def split_documents_in_chunks(
    documents: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> list[str]:
    """
    Splits a list of documents into chunks.

    Returns:
        A list of document chunks.
    """
    chunks = []
    for doc in documents:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunks.append(doc[start:end])
            start += chunk_size - chunk_overlap
    return chunks


def main(cn: str = "colection",
         docs_path: str = "./data/txt") -> None:
    """
    Main function to ingest documents into a chroma database.

    Notes:
        - This function creates a chroma collection if it does not exist,
          loads documents,
        - splits them into chunks, generates embeddings, and adds them to the chroma
          collection.
    """
    collection = client.get_or_create_collection(name=cn)
    documents = load_documents(docs_path)

    chunks = split_documents_in_chunks(
        documents,
        chunk_size=1000,
        chunk_overlap=150
    )
    uuids = [str(uuid.uuid4()) for chunk in chunks]
    embeddings = embed_documents(documents=chunks, batch_size=32, verbose=True)

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=uuids
    )

    print("Done!")


if __name__ == "__main__":
    typer.run(main)
