import sys
import tqdm
import requests

Document   = str
Documents  = list[Document]
Embedding  = list[float]
Embeddings = list[Embedding]


def embed_documents(
    documents: Documents,
    batch_size: int = 64,
    verbose: bool = False
) -> Embeddings:
    """
    Embeds a list of documents into a dense vector space using a specified model.

    Returns:
        Embeddings: A List of embeddings for the input documents.

    Raises:
        requests.RequestException: If the request to the API endpoint fails.
    """

    if batch_size > 128:
        return "Batch size must be less than or equal to 128."

    embeddings = []

    # Split documents into batches
    batch_iter = range(0, len(documents), batch_size)

    if verbose:
        batch_iter = tqdm.tqdm(batch_iter, desc="Embedding documents")

    for i in batch_iter:
        batch = documents[i:i + batch_size]

        # Embed batch of documents
        response = requests.post(
            url=f"http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": batch
            }
        )
        # Append embeddings to list
        embeddings.extend(response.json()["embeddings"])
        
    return embeddings

if __name__ == "__main__":
    print(embed_documents(sys.argv[1]))