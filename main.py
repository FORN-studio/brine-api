import json
import uuid
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from contextlib import asynccontextmanager

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# config
CONFIG = {
    "model_path": "FORNstudio/brine", # model on huggingface
    "companies_file": "companies.json",
    "batch_size": 32,
    "upload_batch_size": 500,
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "max_workers": 2
}


# pydantic models for api
class Company(BaseModel):
    id: str
    website_url: str
    company_description: str
    company_name: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SimilarityRequest(BaseModel):
    company_id: str
    limit: int = 10


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """lifespan context manager for startup and shutdown events."""
    # startup
    global model, qdrant_client
    
    # initialize model and qdrant
    model = initialize_model()
    qdrant_client = initialize_qdrant()
    
    # load and index companies
    companies_file = CONFIG["companies_file"]
    if Path(companies_file).exists():
        companies = load_companies(companies_file)
        if companies:
            index_companies(companies)
            print(f"API ready! Indexed {len(companies)} companies.")
        else:
            print("Warning: No valid companies found in the file.")
    else:
        print(f"Warning: {companies_file} not found. Please update CONFIG['companies_file'].")
        print("API is running but no data is indexed.")
    
    yield
    
    # shutdown - no cleanup needed for in-memory qdrant


# initialize fastapi with lifespan
app = FastAPI(
    title="Company Similarity API", 
    version="0.1.0",
    lifespan=lifespan
)

# add cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # allows all methods
    allow_headers=["*"],  # allows all headers
)

# global variables
model = None
qdrant_client = None
companies_dict = {}
COLLECTION_NAME = "companies"
EMBEDDING_DIM = None


def initialize_model():
    """initialize the sentence transformer model with gpu support if available."""
    global model, EMBEDDING_DIM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # load your fine-tuned model
    model_path = CONFIG["model_path"]
    try:
        model = SentenceTransformer(model_path)
        print(f"Loaded fine-tuned model from {model_path}")
    except Exception as e:
        print(f"Error loading fine-tuned model from {model_path}: {e}")
        print("Falling back to base model...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    model = model.to(device)
    
    # get embedding dimension dynamically
    EMBEDDING_DIM = model.get_sentence_embedding_dimension()
    print(f"Model embedding dimension: {EMBEDDING_DIM}")
    
    return model


def initialize_qdrant():
    """initialize qdrant client with in-memory storage."""
    global qdrant_client
    # using in-memory storage for the prototype
    qdrant_client = QdrantClient(":memory:")
    
    # create collection
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    return qdrant_client


def load_companies(file_path: str) -> List[Company]:
    """load companies from json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # handle both list and dict formats
    if isinstance(data, dict):
        # if it's a dict, assume companies are in a 'companies' key or values
        if 'companies' in data:
            company_list = data['companies']
        else:
            company_list = list(data.values())
    else:
        company_list = data
    
    companies = []
    for item in company_list:
        try:
            # filter out companies with no description or empty description
            if not item.get("company_description"):
                continue

            company = Company(**item)
            companies.append(company)
            companies_dict[company.id] = company
        except Exception as e:
            print(f"Warning: Skipping invalid company entry: {e}")
    
    return companies


def create_embeddings_batch(texts: List[str], batch_size: int = CONFIG["batch_size"]) -> torch.Tensor:
    """create embeddings for texts in batches for optimal gpu utilization."""
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # progress indicator
            print(f"\r  Creating embeddings... batch {batch_num}/{total_batches} "
                  f"({(batch_num/total_batches)*100:.1f}%)", end='', flush=True)
            
            embeddings = model.encode(
                batch, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                normalize_embeddings=True  # normalize for cosine similarity
            )
            all_embeddings.append(embeddings)
    
    return torch.cat(all_embeddings, dim=0)


def index_companies(companies: List[Company]):
    """index all companies in qdrant."""
    print(f"Indexing {len(companies)} companies...")
    
    # extract descriptions for embedding
    descriptions = [company.company_description for company in companies]
    
    # create embeddings in batches
    print("Creating embeddings...")
    embeddings = create_embeddings_batch(descriptions)
    
    # prepare points for qdrant
    points = []
    for i, company in enumerate(companies):
        point = PointStruct(
            id=company.id,
            vector=embeddings[i].cpu().numpy().tolist(),
            payload={
                "company_name": company.company_name,
                "website_url": company.website_url,
                "description": company.company_description
            }
        )
        points.append(point)
    
    # upload to qdrant in batches
    batch_size = CONFIG["upload_batch_size"]
    print("Uploading to vector database...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True  # ensure write completes
        )
        print(f"  Uploaded {min(i + batch_size, len(points))}/{len(points)} companies")
    
    print("Indexing complete!")


@app.get("/health")
async def health_check():
    """health check endpoint for railway."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "qdrant_ready": qdrant_client is not None,
        "companies_indexed": len(companies_dict)
    }


@app.get("/")
async def root():
    """root endpoint with api info."""
    return {
        "name": "Company Similarity API",
        "version": "0.1.0",
        "status": "ready" if len(companies_dict) > 0 else "no data indexed",
        "companies_indexed": len(companies_dict),
        "endpoints": {
            "/search": "Semantic search for companies",
            "/similar": "Find similar companies by ID",
            "/company/{company_id}": "Get company details",
            "/company/by-index/{index}": "Get company details by internal list index",
            "/stats": "Database statistics",
            "/docs": "Interactive API documentation"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search_companies(request: SearchRequest):
    """semantic search for companies based on query text."""
    try:
        # encode query with normalization
        query_embedding = model.encode(
            request.query, 
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        # search in qdrant
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding.cpu().numpy().tolist(),
            limit=request.limit,
            score_threshold=0.0  # return all results, let client filter
        )
        
        # format results
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": float(hit.score),  # ensure json serializable
                "company_name": hit.payload["company_name"],
                "website_url": hit.payload["website_url"],
                "description": hit.payload["description"]
            }
            results.append(result)
        
        return SearchResponse(results=results, query=request.query)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/similar")
async def find_similar_companies(request: SimilarityRequest):
    """find companies similar to a given company id."""
    try:
        # check if company exists
        if request.company_id not in companies_dict:
            raise HTTPException(status_code=404, detail="Company not found")
        
        # get the company's embedding from qdrant
        retrieved_points = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[request.company_id],
            with_vectors=True  # explicitly request vectors
        )
        
        if not retrieved_points:
            raise HTTPException(status_code=404, detail=f"Company ID {request.company_id} not found in the vector index.")
        
        company_point = retrieved_points[0]
        
        if company_point.vector is None:
            raise HTTPException(
                status_code=500, 
                detail=f"Company ID {request.company_id} found, but has no associated vector in the index. Cannot perform similarity search."
            )
        
        # use the retrieved vector to find similar companies
        # we add 1 to limit because the query company itself will be in results
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=company_point.vector, # now guaranteed not to be none
            limit=request.limit + 1
        )
        
        # filter out the query company itself and format results
        results = []
        for hit in search_result:
            if hit.id != request.company_id:
                result = {
                    "id": hit.id,
                    "score": hit.score,
                    "company_name": hit.payload["company_name"],
                    "website_url": hit.payload["website_url"],
                    "description": hit.payload["description"]
                }
                results.append(result)
        
        # ensure we don't exceed requested limit
        results = results[:request.limit]
        
        return {
            "query_company": companies_dict[request.company_id].dict(),
            "similar_companies": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/company/{company_id:str}")
async def get_company(company_id: str):
    """get details for a specific company."""
    if company_id not in companies_dict:
        raise HTTPException(status_code=404, detail="Company not found")
    
    return companies_dict[company_id].dict()

@app.get("/company/{index:int}")
async def get_company_by_index(index: int):
    """get details for a specific company by index."""
    if index >= len(companies_dict):
        raise HTTPException(status_code=404, detail="Company not found")
    
    return companies_dict[list(companies_dict.keys())[index]].dict()



@app.get("/stats")
async def get_stats():
    """get statistics about the indexed data."""
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    
    return {
        "total_companies": collection_info.points_count,
        "vector_dimension": collection_info.config.params.vectors.size,
        "distance_metric": collection_info.config.params.vectors.distance.value
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=CONFIG["host"], 
        port=CONFIG["port"],
        log_level="info",
        reload=True
    )