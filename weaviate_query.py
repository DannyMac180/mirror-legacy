import weaviate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_cluster_metadata():
    # Initialize Weaviate client
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
        additional_headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
        }
    )

    result = client.query.aggregate("ObsidianNotes").with_meta_count().do()
    object_count = result['data']['Aggregate']['ObsidianNotes'][0]['meta']['count']

    print(f"Number of objects in ObsidianNotes: {object_count}")

    # Get information about the vectors
    collection_info = client.schema.get("ObsidianNotes")
    vector_dimensions = collection_info['vectorIndexConfig']['dimensions']

    print(f"Vector dimensions for ObsidianNotes: {vector_dimensions}")

def query_weaviate(query, k=5):
    try:
        # Initialize Weaviate client
        client = weaviate.Client(
            url=os.getenv("WEAVIATE_CLUSTER_URL"),
            auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY")),
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
            }
        )

        # Print the query for debugging
        print(f"Executing query: {query}")

        # Perform semantic search
        result = (
            client.query
            .get("ObsidianNotes", ["content", "title"])
            .with_near_text({"concepts": [query]})
            .with_limit(k)
            .do()
        )

        # Print the raw result for debugging
        print(f"Raw result: {result}")

        # Extract and print results
        if "data" in result and "Get" in result["data"] and "ObsidianNotes" in result["data"]["Get"]:
            documents = result["data"]["Get"]["ObsidianNotes"]
            for i, doc in enumerate(documents, 1):
                print(f"\nDocument {i}:")
                print(f"Title: {doc.get('title', 'N/A')}")
                print(f"Path: {doc.get('path', 'N/A')}")
                print(f"Content: {doc.get('content', 'N/A')[:200]}...")  # Print first 200 characters of content
                print("-" * 50)
        else:
            print("No results found or there was an error in the query structure.")
            print("Result structure:", result)

    except weaviate.exceptions.AuthenticationFailedError:
        print("Authentication failed. Please check your Weaviate API key.")
    except weaviate.exceptions.WeaviateQueryError as e:
        print(f"Error in Weaviate query: {str(e)}")
    except weaviate.exceptions.WeaviateConnectionError:
        print("Failed to connect to Weaviate. Please check your Weaviate URL and network connection.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")

# Example usage
if __name__ == "__main__":
    # get_cluster_metadata()
    query = input("Enter your search query: ")
    k = int(input("Enter the number of documents to retrieve: "))
    query_weaviate(query, k)
