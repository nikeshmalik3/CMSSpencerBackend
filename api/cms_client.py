import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import logging
from urllib.parse import urljoin, urlparse

from config.settings import config
from utils.siren_parser import SirenParser, SirenResponse

logger = logging.getLogger(__name__)

class CMSAPIClient:
    """Async client for interacting with CMS REST APIs"""
    
    def __init__(self):
        self.base_url = config.API_BASE_URL
        self.headers = config.get_api_headers()
        self.timeout = aiohttp.ClientTimeout(total=config.REQUEST_TIMEOUT)
        self.session: Optional[aiohttp.ClientSession] = None
        self.retry_count = config.MAX_RETRIES
        
    async def __aenter__(self):
        """Context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()
        
    async def initialize(self):
        """Initialize the HTTP session"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=self.timeout,
                connector=connector
            )
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        retry_count: int = None
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic"""
        if retry_count is None:
            retry_count = self.retry_count
            
        url = urljoin(self.base_url, endpoint)
        
        for attempt in range(retry_count):
            try:
                logger.debug(f"Making {method} request to {url}")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params
                ) as response:
                    response_text = await response.text()
                    
                    # Log response status
                    logger.debug(f"Response status: {response.status}")
                    
                    if response.status == 401:
                        raise Exception("Authentication failed - token may be expired")
                    
                    if response.status == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited, waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status >= 500:
                        # Server error - retry with backoff
                        logger.warning(f"Server error {response.status}, attempt {attempt + 1}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    
                    # Parse JSON response
                    try:
                        return json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON: {response_text[:200]}...")
                        raise
                        
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}")
                if attempt == retry_count - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {retry_count} attempts")
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a GET request"""
        return await self._make_request("GET", endpoint, params=params)
    
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request"""
        return await self._make_request("POST", endpoint, data=data)
    
    async def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a PUT request"""
        return await self._make_request("PUT", endpoint, data=data)
    
    async def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make a DELETE request"""
        return await self._make_request("DELETE", endpoint)
    
    async def get_siren(self, endpoint: str, params: Optional[Dict] = None) -> SirenResponse:
        """Get and parse a Siren response"""
        response = await self.get(endpoint, params)
        return SirenParser.parse_response(response)
    
    async def execute_siren_action(
        self, 
        action_name: str, 
        siren_response: SirenResponse,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Execute a Siren action from a response"""
        action = SirenParser.find_action(siren_response, action_name)
        if not action:
            raise ValueError(f"Action '{action_name}' not found in response")
        
        # Make request based on action method
        if action.method.upper() == "GET":
            return await self.get(action.href)
        elif action.method.upper() == "POST":
            return await self.post(action.href, data or {})
        elif action.method.upper() == "PUT":
            return await self.put(action.href, data or {})
        elif action.method.upper() == "DELETE":
            return await self.delete(action.href)
        else:
            raise ValueError(f"Unsupported method: {action.method}")
    
    async def paginate(
        self, 
        endpoint: str, 
        params: Optional[Dict] = None,
        max_pages: int = 10
    ) -> List[Dict[str, Any]]:
        """Handle paginated responses automatically"""
        all_results = []
        current_endpoint = endpoint
        page_count = 0
        
        while current_endpoint and page_count < max_pages:
            response = await self.get_siren(current_endpoint, params)
            data = SirenParser.extract_data(response)
            
            # Add current page results
            if "entities" in data:
                all_results.extend(data["entities"])
            else:
                all_results.append(data["data"])
            
            # Get next page link
            next_link = SirenParser.get_next_link(response)
            if not next_link:
                break
                
            # Update endpoint for next iteration
            current_endpoint = next_link
            params = None  # Params are usually included in the link
            page_count += 1
        
        return all_results

# Singleton instance
cms_client = CMSAPIClient()