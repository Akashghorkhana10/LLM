from fastapi import FastAPI
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# Create an instance of FastAPI
app = FastAPI()

def get_overlapped_chunks(textin, chunksize, overlapsize):  
    return [textin[a:a+chunksize] for a in range(0,len(textin), chunksize-overlapsize)]

dataset = open('dataset\\text-format\without-index\\Constitution.txt').read()

chunks = get_overlapped_chunks(dataset, 1000, 100)
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder = '/models/sentence-transformers/all-mpnet-base-v2')
chunk_embeddings = embedding_model.encode(chunks)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="llm_rag")
max_batch_size = 166  # Maximum batch size allowed

# Calculate the number of batches needed
num_batches = (len(chunk_embeddings) + max_batch_size - 1) // max_batch_size

# Split the data into smaller batches
for i in range(num_batches):
    start_idx = i * max_batch_size
    end_idx = (i + 1) * max_batch_size
    batch_embeddings = chunk_embeddings[start_idx:end_idx]
    batch_chunks = chunks[start_idx:end_idx]
    batch_ids = [str(j) for j in range(start_idx, min(end_idx, len(chunk_embeddings)))]
    
    # Process each batch
    collection.add(
        embeddings=batch_embeddings,
        documents=batch_chunks,
        ids=batch_ids
    )

class QueryItem(BaseModel):
    query: str

@app.post("/RAG")
async def retrieve_vector_db(query_item: QueryItem):
    query = query_item.query  # Extract the query string from QueryItem object
    results = collection.query(
        query_embeddings=embedding_model.encode([query]).tolist(),  # Encode the query string
        n_results=1
    )
    return {"message": results['documents']}


# Define a route for the root endpoint "/"
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
