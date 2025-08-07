import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os

from agents.base_agent import BaseAgent
from api.cms_client import cms_client
from utils.siren_parser import SirenParser
from storage import redis_connector
from config.settings import config

logger = logging.getLogger(__name__)

class APIExecutorAgent(BaseAgent):
    """
    The Gateway to all CMS data - handles all communication with REST APIs
    """
    
    def __init__(self):
        super().__init__(
            name="api_executor",
            description="Executes API calls to CMS endpoints and manages responses"
        )
        self.api_catalog = self._load_api_catalog()
        self.endpoint_cache = {}
        
    def _load_api_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Load endpoints from JSON file - easily updatable without code changes"""
        try:
            # Load from endpoints.json file
            endpoints_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config', 
                'endpoints.json'
            )
            
            if os.path.exists(endpoints_file):
                with open(endpoints_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {data['metadata']['total_endpoints']} endpoints from endpoints.json")
                    return data['endpoints']
            else:
                logger.warning("endpoints.json not found, using fallback endpoints")
                # Fallback to minimal set if file not found
                return {
                    "orders": {"list": "/api/orders"},
                    "projects": {"list": "/api/projects"},
                    "users": {"list": "/api/users"}
                }
        except Exception as e:
            logger.error(f"Error loading endpoints: {e}")
            # Return minimal fallback
            return {
                "orders": {"list": "/api/orders"},
                "projects": {"list": "/api/projects"},
                "users": {"list": "/api/users"}
            }
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate API executor request"""
        return "query" in request or "endpoint" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process API request - either direct endpoint call or natural language query
        """
        try:
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Direct endpoint call
            if "endpoint" in request:
                return await self._execute_direct_api_call(
                    request["endpoint"],
                    request.get("method", "GET"),
                    request.get("data"),
                    request.get("params")
                )
            
            # Natural language query processing
            query = request.get("query", "")
            
            # Detect what user is asking for
            api_intent = await self._detect_api_intent(query)
            
            # Build and execute API calls
            api_results = await self._execute_api_sequence(
                api_intent,
                context,
                parameters
            )
            
            # Format response
            formatted_response = await self._format_api_response(
                api_results,
                api_intent,
                query
            )
            
            return {
                "data": formatted_response,
                "metadata": {
                    "api_calls_made": len(api_results),
                    "endpoints_used": [r["endpoint"] for r in api_results],
                    "cache_hits": sum(1 for r in api_results if r.get("cached", False))
                }
            }
            
        except Exception as e:
            logger.error(f"API Executor error: {e}", exc_info=True)
            raise
    
    async def _detect_api_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect what API endpoints to call based on query
        """
        query_lower = query.lower()
        intent = {
            "resource": None,
            "action": None,
            "filters": {},
            "fields": []
        }
        
        # Detect resource type
        resource_keywords = {
            "orders": ["order", "orders", "purchase", "po"],
            "callouts": ["call-out", "callout", "call out"],
            "projects": ["project", "projects", "job", "site"],
            "users": ["user", "users", "people", "team"],
            "financial": ["invoice", "payment", "quote", "budget", "cost"],
            "inventory": ["inventory", "stock", "material", "item"]
        }
        
        for resource, keywords in resource_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent["resource"] = resource
                break
        
        # Detect action
        action_keywords = {
            "list": ["show", "list", "all", "get all", "find", "search"],
            "get": ["get", "details", "information", "info"],
            "create": ["create", "add", "new", "make"],
            "update": ["update", "edit", "modify", "change"],
            "approve": ["approve", "authorization", "sign off"],
            "count": ["how many", "count", "number of", "total"]
        }
        
        for action, keywords in action_keywords.items():
            if any(kw in query_lower for kw in keywords):
                intent["action"] = action
                break
        
        # Detect filters
        if "pending" in query_lower:
            intent["filters"]["status"] = "pending"
        if "approved" in query_lower:
            intent["filters"]["status"] = "approved"
        if "today" in query_lower:
            intent["filters"]["date"] = "today"
        if "this week" in query_lower:
            intent["filters"]["date"] = "this_week"
        if "urgent" in query_lower:
            intent["filters"]["priority"] = "urgent"
        
        # Extract IDs if present
        import re
        id_match = re.search(r'#?(\d+)', query)
        if id_match:
            intent["filters"]["id"] = id_match.group(1)
        
        # Default to list action if not specified
        if not intent["action"]:
            intent["action"] = "list"
        
        # Default to orders if no resource detected
        if not intent["resource"]:
            intent["resource"] = "orders"
        
        return intent
    
    async def _execute_api_sequence(
        self,
        api_intent: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute sequence of API calls based on intent
        """
        results = []
        resource = api_intent["resource"]
        action = api_intent["action"]
        filters = api_intent["filters"]
        
        # Get endpoint from catalog
        endpoint_template = self.api_catalog.get(resource, {}).get(action)
        if not endpoint_template:
            logger.warning(f"No endpoint found for {resource}.{action}")
            return results
        
        # Check cache first
        cache_key = f"{resource}:{action}:{json.dumps(filters, sort_keys=True)}"
        cached_response = await redis_connector.get_cached_api_response(
            endpoint_template,
            filters
        )
        
        if cached_response and parameters.get("mode") != "data_fetch":
            results.append({
                "endpoint": endpoint_template,
                "response": cached_response,
                "cached": True
            })
            return results
        
        async with cms_client as client:
            # Handle ID-based endpoints
            if "{id}" in endpoint_template and "id" in filters:
                endpoint = endpoint_template.replace("{id}", str(filters["id"]))
                response = await client.get_siren(endpoint)
            else:
                # List endpoints with filters
                params = self._build_query_params(filters)
                
                # Handle pagination for list requests
                if action == "list":
                    response_data = await client.paginate(
                        endpoint_template,
                        params,
                        max_pages=5
                    )
                    
                    results.append({
                        "endpoint": endpoint_template,
                        "response": {"items": response_data, "count": len(response_data)},
                        "cached": False
                    })
                    
                    # Cache the response
                    await redis_connector.cache_api_response(
                        endpoint_template,
                        params,
                        {"items": response_data}
                    )
                    
                    return results
                else:
                    response = await client.get_siren(endpoint_template, params)
            
            # Parse Siren response
            parsed_data = SirenParser.extract_data(response)
            
            results.append({
                "endpoint": endpoint_template,
                "response": parsed_data,
                "cached": False,
                "available_actions": parsed_data.get("available_actions", [])
            })
            
            # Cache the response
            await redis_connector.cache_api_response(
                endpoint_template,
                filters,
                parsed_data
            )
            
            # Execute related calls if needed
            if parameters.get("mode") == "data_fetch" and parsed_data.get("entities"):
                # Fetch additional details for analytics
                for entity in parsed_data["entities"][:10]:  # Limit to 10
                    if entity.get("href"):
                        detail_response = await client.get_siren(entity["href"])
                        detail_data = SirenParser.extract_data(detail_response)
                        
                        results.append({
                            "endpoint": entity["href"],
                            "response": detail_data,
                            "cached": False,
                            "parent": endpoint_template
                        })
        
        return results
    
    def _build_query_params(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build query parameters from filters"""
        params = {}
        
        # Status filter
        if "status" in filters:
            params["status"] = filters["status"]
        
        # Date filters
        if filters.get("date") == "today":
            params["created_at_from"] = datetime.utcnow().date().isoformat()
            params["created_at_to"] = datetime.utcnow().date().isoformat()
        elif filters.get("date") == "this_week":
            # Calculate week start/end
            today = datetime.utcnow().date()
            week_start = today - timedelta(days=today.weekday())
            params["created_at_from"] = week_start.isoformat()
            params["created_at_to"] = today.isoformat()
        
        # Priority filter
        if "priority" in filters:
            params["priority"] = filters["priority"]
        
        # Pagination
        params["per_page"] = 50
        params["page"] = 1
        
        return params
    
    async def _format_api_response(
        self,
        api_results: List[Dict[str, Any]],
        api_intent: Dict[str, Any],
        original_query: str
    ) -> Dict[str, Any]:
        """
        Format API results into user-friendly response
        """
        if not api_results:
            return {
                "message": "I couldn't find any data matching your request.",
                "items": [],
                "count": 0
            }
        
        primary_result = api_results[0]
        response_data = primary_result["response"]
        
        # Format based on resource type and action
        resource = api_intent["resource"]
        action = api_intent["action"]
        
        if action == "list":
            items = response_data.get("items", [])
            count = len(items)
            
            # Create summary message
            if resource == "orders":
                message = f"I found {count} orders"
                if api_intent["filters"].get("status"):
                    message += f" with status '{api_intent['filters']['status']}'"
                
                # Add summary statistics
                if items:
                    total_value = sum(
                        item.get("data", {}).get("total", 0) 
                        for item in items
                    )
                    message += f". Total value: £{total_value:,.2f}"
            
            elif resource == "projects":
                message = f"I found {count} projects"
                
            elif resource == "users":
                message = f"I found {count} users"
            
            else:
                message = f"I found {count} {resource}"
            
            return {
                "message": message,
                "items": items[:20],  # Limit items in response
                "count": count,
                "total_count": response_data.get("count", count),
                "actions": primary_result.get("available_actions", [])
            }
        
        elif action == "get":
            data = response_data.get("data", {})
            
            # Format single item response
            if resource == "orders":
                message = f"Order #{data.get('id', 'N/A')}: {data.get('description', 'No description')}"
                message += f"\nStatus: {data.get('status', 'Unknown')}"
                message += f"\nTotal: £{data.get('total', 0):,.2f}"
            else:
                message = f"Here are the details for {resource} #{api_intent['filters'].get('id', '')}"
            
            return {
                "message": message,
                "data": data,
                "entities": response_data.get("entities", []),
                "actions": primary_result.get("available_actions", [])
            }
        
        else:
            # Generic response
            return {
                "message": f"API request completed successfully",
                "data": response_data,
                "actions": primary_result.get("available_actions", [])
            }
    
    async def _execute_direct_api_call(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute direct API call when endpoint is specified"""
        async with cms_client as client:
            if method == "GET":
                response = await client.get(endpoint, params)
            elif method == "POST":
                response = await client.post(endpoint, data or {})
            elif method == "PUT":
                response = await client.put(endpoint, data or {})
            elif method == "DELETE":
                response = await client.delete(endpoint)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            return {
                "data": {
                    "response": response,
                    "endpoint": endpoint,
                    "method": method
                }
            }

# Create singleton instance
api_executor = APIExecutorAgent()