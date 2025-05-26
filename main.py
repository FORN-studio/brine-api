import json
from typing import List, Dict, Any
from pathlib import Path
import asyncio
import os
import requests
import aiohttp
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "model_endpoint": os.getenv("MODEL_ENDPOINT"),
    "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
    "embedding_dim": 768,
    "companies_file": "companies.json",
    "batch_size": 32,
    "upload_batch_size": 500,
    "host": "0.0.0.0",
    "port": int(os.getenv("PORT", 8000)),
    "max_workers": 2,
    "max_concurrent_requests": 500
}


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
    global qdrant_client, HUGGINGFACE_API_URL, HUGGINGFACE_HEADERS, EMBEDDING_DIM
    
    qdrant_client = initialize_qdrant()
    
    HUGGINGFACE_API_URL = CONFIG["model_endpoint"]
    HUGGINGFACE_HEADERS = {
        "Accept": "application/json",
        "Authorization": f"Bearer {CONFIG['huggingface_token']}",
        "Content-Type": "application/json" 
    }
    EMBEDDING_DIM = CONFIG["embedding_dim"]

    companies_file = CONFIG["companies_file"]
    companies = load_companies(companies_file)
    await index_companies(companies)
    
    yield


app = FastAPI(
    title="Brine", 
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client = None
companies_dict = {}
COLLECTION_NAME = "companies"
EMBEDDING_DIM = None
HUGGINGFACE_API_URL = None
HUGGINGFACE_HEADERS = None


def initialize_qdrant():
    global qdrant_client
    qdrant_client = QdrantClient(":memory:")
    
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=CONFIG.get("embedding_dim", 768), distance=Distance.COSINE),
    )
    
    return qdrant_client


def load_companies(file_path: str) -> List[Company]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    company_list = data
    
    companies = []
    for item in company_list:
        try:
            if not item.get("company_description") or item.get("company_description") == "":
                continue

            company = Company(**item)
            companies.append(company)
            companies_dict[company.id] = company
        except Exception:
            continue
    
    return companies


async def create_single_embedding(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, text: str, index: int, total: int) -> tuple[int, List[float]]:
    async with semaphore:
        print(f"\r  Creating embedding {index + 1}/{total} ({((index + 1)/total)*100:.1f}%)", end='', flush=True)
        
        payload = {"inputs": text}
        
        try:
            async with session.post(HUGGINGFACE_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                response.raise_for_status()
                embedding_response = await response.json()
                
                if isinstance(embedding_response, dict) and 'embeddings' in embedding_response:
                    actual_embedding = embedding_response['embeddings']
                    if not (isinstance(actual_embedding, list) and all(isinstance(val, (float, int)) for val in actual_embedding)):
                        raise ValueError(f"Embeddings are not a list of numbers. Got: {type(actual_embedding)}")
                else:
                    raise ValueError(f"Expected response format {{'embeddings': [...]}} but got: {embedding_response}")

                return index, actual_embedding

        except Exception as e:
            print(f"\nBrine caught on fire ({index + 1}: {e})")
            raise


async def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    semaphore = asyncio.Semaphore(CONFIG["max_concurrent_requests"])
    
    async with aiohttp.ClientSession(headers=HUGGINGFACE_HEADERS) as session:
        tasks = [
            create_single_embedding(session, semaphore, text, i, len(texts))
            for i, text in enumerate(texts)
        ]
        
        results = await asyncio.gather(*tasks)
    
    results.sort(key=lambda x: x[0])
    embeddings = [result[1] for result in results]
    
    print("\nembeddings done")
    return embeddings


async def index_companies(companies: List[Company]):
    
    descriptions = [company.company_description for company in companies]
    
    print("creating embeddings...")
    embeddings = await create_embeddings_batch(descriptions)
    
    points = []
    for i, company in enumerate(companies):
        point = PointStruct(
            id=company.id,
            vector=embeddings[i],
            payload={
                "company_name": company.company_name,
                "website_url": company.website_url,
                "description": company.company_description
            }
        )
        points.append(point)
    
    batch_size = CONFIG["upload_batch_size"]
    print("uploading to skynet...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch,
            wait=True
        )
        print(f"uploaded {min(i + batch_size, len(points))}/{len(points)} companies")
    
    print("we made it!")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "qdrant_ready": qdrant_client is not None,
        "companies_indexed": len(companies_dict)
    }


@app.get("/")
async def root():
    return {
        "name": "Brine",
        "version": "1.0.0",
        "status": "ready" if len(companies_dict) > 0 else "no data indexed",
        "companies_indexed": len(companies_dict),
        "endpoints": {
            "/search": "Semantic search for companies",
            "/similar": "Find similar companies by ID",
            "/company/{company_id}": "Get company details",
            "/stats": "Basic stats",
            "/docs": "Documentation"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search_companies(request: SearchRequest):
    if not HUGGINGFACE_API_URL or not HUGGINGFACE_HEADERS:
        raise HTTPException(status_code=503, detail="Embedding service not configured.")
    try:
        payload = {"inputs": request.query}
        
        response = requests.post(HUGGINGFACE_API_URL, headers=HUGGINGFACE_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        
        query_embedding_response = response.json()

        if isinstance(query_embedding_response, dict) and 'embeddings' in query_embedding_response:
            query_embedding = query_embedding_response['embeddings']
            if not (isinstance(query_embedding, list) and all(isinstance(val, (float, int)) for val in query_embedding)):
                raise ValueError(f"Query embeddings are not a list of numbers. Got: {type(query_embedding)}")
        else:
            raise ValueError(f"Expected response format {{'embeddings': [...]}} but got: {query_embedding_response}")
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=0.0
        )
        
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": float(hit.score),
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
    try:
        if request.company_id not in companies_dict:
            raise HTTPException(status_code=404, detail="Company not found")
        
        retrieved_points = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[request.company_id],
            with_vectors=True
        )
        
        if not retrieved_points:
            raise HTTPException(status_code=404, detail=f"Company ID {request.company_id} not found in the vector index.")
        
        company_point = retrieved_points[0]
        
        if company_point.vector is None:
            raise HTTPException(
                status_code=500, 
                detail=f"Company ID {request.company_id} found, but has no associated vector in the index. Cannot perform similarity search."
            )
        
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=company_point.vector,
            limit=request.limit + 1
        )
        
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
    if company_id not in companies_dict:
        raise HTTPException(status_code=404, detail="Company not found")
    
    return companies_dict[company_id].dict()


@app.get("/stats")
async def get_stats():
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
    )