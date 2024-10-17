import os
import typer
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

from embed_function import embed_documents


load_dotenv()

def get_dnd_documents(query: str, cn: str, n_results: int = 3) -> list[str]:
    """
    Retrieves relevant documents from the Chroma database based on the input query.

    Args:
        query: The input query to search for relevant documents.
        cn: The name of the collection to query.
        n_results: The number of results to return (default: 3).

    Returns:
        A list of relevant documents retrieved from the database.

    Notes:
        Embeds the query using the embed_documents function to generate query embeddings.
        Uses the query embeddings to query the database collection (created if it does not exist).
    """
    chromadb_client = chromadb.HttpClient(host=os.getenv("CHROMA_DB_URL"))
    collection = chromadb_client.get_or_create_collection(name=cn)

    query_embeddings = embed_documents(documents=[query])

    result = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results
    )

    return str(result['documents'])


def query_llm(messages: list[dict[str,str]]) -> dict[str,str]:
    """
    Queries the LLM model using the provided messages and options.

    Args:
        messages: A list of messages to send to the LLM model.

    Returns:
        The response from the LLM model.

    Notes:
        Uses the OpenAI client to query the LLM model.
        Retrieves model options (temperature, max_tokens) from environment variables.
    """
    openai_client = OpenAI(
        base_url='http://localhost:11434/v1',
        #base_url='https://chat.balero.net/v1',
        api_key='-'
    )
    fns = {
        "get_dnd_documents": get_dnd_documents
    }

    tools=[
        {
            'type': 'function',
            'function': {
                'name': 'get_dnd_documents',
                'description': 'A tool to retrieve information from Dungeons n Dragons players handbook.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'The query to retrieve documents from vector database.'
                        },
                        'cn': {
                            'type': 'string',
                            'description': 'The collection name to retrieve documents from vector database',
                            'enum': ["dnd"]
                        }
                    },
                    'required': ['query']
                }
            }
        },
    ]

    response = openai_client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        temperature=0.4,
        max_tokens=1024,
        tools=tools
    )

    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.to_dict()

    # Process function calls made by the model
    messages.append(response.choices[0].message.to_dict())
    for tool_call in response.choices[0].message.tool_calls:
        fn_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        fn_result = fns[fn_name](**arguments)

        messages.append({
            'role': 'tool',
            'content': fn_result,
            "tool_call_id": tool_call.id
        })

    # Second API call: Get final response from the model
    final_response = openai_client.chat.completions.create(
        model="qwen2.5",
        messages=messages,
        temperature=0.4,
        max_tokens=1024,
    )
    return final_response.choices[0].message.to_dict()


def main(query: str) -> None:
    """
    The main function that queries the database and LLM model to answer a question.

    Notes:
    - The query is first used to retrieve relevant documents from the database
      using the query_documents function.
    - The retrieved documents are then used to create a prompt for the LLM model.
    - The LLM model is queried using the query_llm function.
    """

    prompt = """You are a helpful assistant that chats with a user.
    Use the given tools to retrieve information from a database with information from Dungeons and Dragons 5e.
    Only use the tool if the user make a question about Dungeons and Dragons.
    If the user asks about something else, just answer the question.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    query_llm(messages)


if __name__=="__main__":
    typer.run(main)