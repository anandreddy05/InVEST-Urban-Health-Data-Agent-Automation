from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

from agents.data_agent import EarthEngineDataAgentComplete
from agents.geollm_agent import GeoLLMAgent, GeoRequest


# ------------------------------------------------------------------------------
# 🌆 FastAPI Configuration
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Urban Data Automation Agent",
    description="Automated data acquisition, preprocessing, and visualization using LLM + Earth Engine",
    version="1.0.0"
)

# ------------------------------------------------------------------------------
# 🧠 Initialize Agents
# ------------------------------------------------------------------------------
data_agent = EarthEngineDataAgentComplete() 
geollm_agent = GeoLLMAgent()

# ------------------------------------------------------------------------------
# 📦 Pydantic Models for Requests / Responses
# ------------------------------------------------------------------------------
class DataRequest(BaseModel):
    city: str
    data_types: List[str] = ["land_cover", "tree_cover", "ndvi", "population"]
    year: Optional[int] = 2020


class DataResponse(BaseModel):
    job_id: str
    location: str
    timestamp: str
    datasets: Dict
    outputs: Dict
    status: str



class NaturalLanguageRequest(BaseModel):
    prompt: str
    year: Optional[int] = 2020
    context: Optional[Dict] = None


class ConversationalResponse(BaseModel):
    response: str
    structured_data: Dict
    clarification_needed: List[str]


# ------------------------------------------------------------------------------
# 🏠 Root Endpoint
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "🌆 Urban Data Automation API with LLM + Earth Engine Integration"}


# ------------------------------------------------------------------------------
# 🧩 1️⃣ Structured Request (Direct Parameters)
# ------------------------------------------------------------------------------
@app.post("/agents/data/fetch", response_model=DataResponse)
async def fetch_data(request: DataRequest):
    """
    Fetch data directly using structured city, data types, and year.
    """
    try:
        result = data_agent.process_city_data(
            city_name=request.city,
            data_types=request.data_types,
            year=request.year
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 💬 2️⃣ Natural Language → LLM → Data Fetch
# ------------------------------------------------------------------------------
@app.post("/agents/data/natural-language", response_model=DataResponse)
async def fetch_data_natural_language(request: NaturalLanguageRequest):
    """
    Fetch data using a natural language prompt parsed by the GeoLLM agent.
    """
    try:
        # Step 1: Parse user prompt via GeoLLM
        geo_request = GeoRequest(prompt=request.prompt)
        geo_response = geollm_agent.process_request(geo_request)

        # Step 2: Fetch actual data via Earth Engine
        result = data_agent.process_city_data(
            city_name=geo_response.city,
            data_types=geo_response.data_types,
            year=request.year
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 🗣️ 3️⃣ Conversational LLM Interface
# ------------------------------------------------------------------------------
@app.post("/agents/data/conversational", response_model=ConversationalResponse)
async def conversational_data_request(request: NaturalLanguageRequest):
    """
    Conversational LLM interface for interactive data discovery.
    """
    try:
        result = geollm_agent.conversational_retrieval(
            prompt=request.prompt,
            context=request.context
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return ConversationalResponse(
            response=result["conversational_response"],
            structured_data=result["structured_data"],
            clarification_needed=result.get("needs_clarification", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 🔍 4️⃣ Parse Prompt Only (Debugging Endpoint)
# ------------------------------------------------------------------------------
@app.post("/agents/data/parse-prompt")
async def parse_prompt_only(request: NaturalLanguageRequest):
    """
    Debug endpoint — just parse the user prompt without fetching any data.
    """
    try:
        extraction = geollm_agent.parse_prompt_with_llm(request.prompt)
        return extraction.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------------------
# 📊 5️⃣ Job Status Endpoint (Future Extension)
# ------------------------------------------------------------------------------
@app.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Retrieve the processing status of a previous job.
    """
    # Placeholder for future DB integration
    return {"job_id": job_id, "status": "completed"}


# ------------------------------------------------------------------------------
# 🚀 Run the Server (Development)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
