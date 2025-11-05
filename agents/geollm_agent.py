from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import os
from utils.geocoding import Geocoder
from dotenv import load_dotenv

load_dotenv(override=True)

open_api = os.getenv("OPENAI_API_KEY") or "Enter API Key"
if not open_api or open_api == "Enter API Key":
    raise ValueError("⚠️ OpenAI API key missing. Please set OPENAI_API_KEY in .env file.")


class GeoExtraction(BaseModel):
    city: str = Field(description="The city or region name extracted from the prompt")
    data_types: List[str] = Field(description="List of requested data types")


class GeoRequest(BaseModel):
    prompt: str
    data_types: List[str] = Field(default_factory=lambda: ["land_cover", "tree_cover", "ndvi", "population", "basemap"])


class GeoResponse(BaseModel):
    city: str
    aoi: Dict
    data_types: List[str]
    confidence: float

    
class GeoLLMAgent:
    def __init__(self):
        self.geocoder = Geocoder()
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=open_api,
            timeout=30  # Add timeout to prevent hanging
        )
        self.setup_prompt_template()
    
    def setup_prompt_template(self):
        """Setup LangChain prompt template for geographic parsing"""
        self.parser = PydanticOutputParser(pydantic_object=GeoExtraction)
        
        self.prompt_template = PromptTemplate(
            template="""You are a geographic data extraction assistant. 
Parse the user's natural language request to extract the location and requested data types.

User Request: {prompt}

Available Data Types:
- land_cover: Land Use/Land Cover data
- tree_cover: Tree Canopy Cover data  
- ndvi: Normalized Difference Vegetation Index
- population: Population density data
- basemap: Satellite imagery (true-color RGB)

Extract the city/region name and list the requested data types. 
If no specific types are mentioned, include all available types.

Return the result **strictly** as a JSON object:
{{
  "city": "<city name>",
  "data_types": ["list", "of", "data types"]
}}

{format_instructions}
""",
            input_variables=["prompt"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def parse_prompt_with_llm(self, prompt: str) -> GeoExtraction:
        """Use OpenAI LLM to parse natural language prompt"""
        try:
            chain = self.prompt_template | self.llm | self.parser
            result = chain.invoke({"prompt": prompt})
            return result
            
        except Exception as e:
            print(f"⚠️ LLM parsing error: {e}")
            return self.fallback_parsing(prompt)
    
    def fallback_parsing(self, prompt: str) -> GeoExtraction:
        """Fallback method if LLM fails"""
        prompt_lower = prompt.lower()
        
        # Simple location extraction (last 2-3 words typically)
        words = prompt.split()
        location = " ".join(words[-3:])
        
        # Keyword-based data type extraction
        data_types = []
        keyword_mapping = {
            "land_cover": ["land cover", "land use", "lulc", "nlcd", "landcover"],
            "tree_cover": ["tree cover", "tree canopy", "canopy", "forest", "trees"],
            "ndvi": ["ndvi", "vegetation", "greenness", "vegetation index"],
            "population": ["population", "pop", "people", "demographic", "census"],
            "basemap": ["basemap", "satellite", "imagery", "map", "rgb", "true color", "aerial"]
        }
        
        for data_type, keywords in keyword_mapping.items():
            if any(keyword in prompt_lower for keyword in keywords):
                data_types.append(data_type)
        
        if not data_types:
            data_types = list(keyword_mapping.keys())
        
        return GeoExtraction(city=location, data_types=data_types)
    
    def process_request(self, request: GeoRequest) -> GeoResponse:
        """Process natural language request using LLM"""
        extraction = self.parse_prompt_with_llm(request.prompt)
        bbox = self.geocoder.get_bounding_box(extraction.city)
        
        if not bbox:
            raise ValueError(f"Could not geocode location: {extraction.city}")
        
        return GeoResponse(
            city=bbox["name"],
            aoi=bbox,
            data_types=extraction.data_types,
            confidence=0.95
        )

    def conversational_retrieval(self, prompt: str, context: Dict = None) -> Dict:
        """
        ⚡ OPTIMIZED: Single LLM call instead of two
        """
        try:
            extraction = self.parse_prompt_with_llm(prompt)
            
            # Generate conversational response from extraction
            response_text = (
                f"I'll fetch {', '.join(extraction.data_types)} data for {extraction.city}. "
                f"Processing your request now..."
            )
            
            return {
                "conversational_response": response_text,
                "structured_data": extraction.dict(),
                "needs_clarification": self.check_clarification_needed(extraction)
            }
            
        except Exception as e:
            print(f"❌ Conversational retrieval error: {e}")
            return {"error": "Failed to process conversational request"}
    
    def check_clarification_needed(self, extraction: GeoExtraction) -> List[str]:
        """Check if any clarification is needed for the request"""
        clarifications = []
        
        if not extraction.city or len(extraction.city.split()) < 2:
            clarifications.append("Please specify which city or region you need data for.")
        
        if len(extraction.data_types) == 4:
            clarifications.append("Would you like all data types or specific ones?")
        
        return clarifications