import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

from agents.base_agent import BaseAgent, AgentStatus
from storage import redis_connector, mongodb_connector
from models import Conversation, Message, MessageRole, ConversationContext, AgentExecution
from config.settings import config

logger = logging.getLogger(__name__)

class MasterOrchestratorAgent(BaseAgent):
    """
    The Brain of Spencer AI - receives all requests and orchestrates other agents
    """
    
    def __init__(self):
        super().__init__(
            name="master_orchestrator",
            description="Master orchestrator that routes requests to appropriate agents"
        )
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.intent_patterns = self._load_intent_patterns()
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agent_registry[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent patterns for routing"""
        return {
            "api_query": [
                "show", "get", "list", "find", "search", "fetch",
                "orders", "projects", "users", "call-outs", "quotes"
            ],
            "document_upload": [
                "upload", "analyze", "process", "read",
                "pdf", "excel", "autocad", "image", "blueprint"
            ],
            "workflow": [
                "create", "approve", "workflow", "automate",
                "process", "sequence", "chain"
            ],
            "analytics": [
                "report", "analytics", "dashboard", "kpi",
                "metrics", "statistics", "trend", "analysis"
            ],
            "voice": [
                "voice", "speech", "audio", "transcribe", "speak"
            ],
            "semantic_search": [
                "similar", "related", "context", "meaning",
                "understand", "knowledge"
            ]
        }
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate orchestrator request"""
        return "query" in request and "user_id" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method - analyzes intent and routes to agents
        """
        try:
            query = request.get("query", "")
            user_id = request.get("user_id")
            session_id = request.get("session_id")
            context = request.get("context", {})
            
            # Load or create conversation
            conversation = await self._get_or_create_conversation(
                user_id, session_id, context
            )
            
            # Add user message to conversation
            user_message = Message(
                role=MessageRole.USER,
                content=query,
                metadata={"source": "api"}
            )
            conversation.messages.append(user_message.dict())
            
            # Analyze intent
            intent, confidence = await self._analyze_intent(query, conversation)
            logger.info(f"Detected intent: {intent} (confidence: {confidence})")
            
            # Select agents based on intent
            selected_agents = await self._select_agents(intent, query, context)
            logger.info(f"Selected agents: {[a for a, _ in selected_agents]}")
            
            # Execute agents
            agent_results = await self._execute_agents(
                selected_agents, 
                query, 
                conversation,
                context
            )
            
            # Synthesize response
            final_response = await self._synthesize_response(
                agent_results,
                query,
                intent
            )
            
            # Add assistant response to conversation
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=final_response["message"],
                metadata={
                    "intent": intent,
                    "agents_used": [r["agent"] for r in agent_results]
                }
            )
            conversation.messages.append(assistant_message.dict())
            
            # Save conversation
            await self._save_conversation(conversation)
            
            return {
                "data": {
                    "response": final_response["message"],
                    "intent": intent,
                    "confidence": confidence,
                    "data": final_response.get("data", {}),
                    "actions": final_response.get("actions", []),
                    "session_id": conversation.session_id
                },
                "metadata": {
                    "agents_executed": len(agent_results),
                    "processing_details": agent_results
                }
            }
            
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            raise
    
    async def _analyze_intent(
        self, 
        query: str, 
        conversation: Conversation
    ) -> Tuple[str, float]:
        """
        Analyze user intent from query and context
        """
        query_lower = query.lower()
        
        # Score each intent based on keyword matches
        intent_scores = {}
        
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Consider conversation history
        if len(conversation.messages) > 1:
            # Boost score if continuing same type of conversation
            last_intents = [
                msg.get("metadata", {}).get("intent") 
                for msg in conversation.messages[-3:]
                if msg.get("metadata", {}).get("intent")
            ]
            
            for prev_intent in last_intents:
                if prev_intent in intent_scores:
                    intent_scores[prev_intent] += 0.5
        
        # Get highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            total_score = sum(intent_scores.values())
            confidence = best_intent[1] / total_score if total_score > 0 else 0
            
            return best_intent[0], confidence
        
        # Default intent if no match
        return "api_query", 0.5
    
    async def _select_agents(
        self, 
        intent: str, 
        query: str,
        context: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select agents to execute based on intent
        Returns list of (agent_name, parameters) tuples
        """
        selected = []
        
        # Always include compliance agent for validation
        selected.append(("compliance", {"validate_only": True}))
        
        # Select primary agent based on intent
        if intent == "api_query":
            selected.append(("api_executor", {}))
            # Also search for similar queries
            selected.append(("semantic_search", {"search_type": "queries"}))
            
        elif intent == "document_upload":
            selected.append(("document_processor", {}))
            # Prepare for semantic indexing
            selected.append(("semantic_search", {"mode": "index"}))
            
        elif intent == "workflow":
            selected.append(("workflow_automation", {}))
            # May need API calls
            selected.append(("api_executor", {"mode": "support"}))
            
        elif intent == "analytics":
            selected.append(("analytics", {}))
            # Will need data from APIs
            selected.append(("api_executor", {"mode": "data_fetch"}))
            
        elif intent == "voice":
            selected.append(("voice_nlu", {}))
            
        elif intent == "semantic_search":
            selected.append(("semantic_search", {"search_type": "knowledge"}))
        
        return selected
    
    async def _execute_agents(
        self,
        selected_agents: List[Tuple[str, Dict[str, Any]]],
        query: str,
        conversation: Conversation,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute selected agents in parallel where possible
        """
        results = []
        
        # Group agents by dependency
        # Compliance runs first, then parallel execution of others
        
        for agent_name, params in selected_agents:
            if agent_name not in self.agent_registry:
                logger.warning(f"Agent {agent_name} not registered")
                continue
            
            agent = self.agent_registry[agent_name]
            
            # Skip if agent is disabled
            if agent.status == AgentStatus.DISABLED:
                continue
            
            # Prepare agent request
            agent_request = {
                "query": query,
                "context": {
                    **context,
                    "conversation_id": conversation.id,
                    "user_id": conversation.user_id,
                    "session_id": conversation.session_id,
                    "intent": params
                },
                "parameters": params
            }
            
            # Execute agent
            execution_start = datetime.utcnow()
            
            try:
                if agent_name == "compliance" and params.get("validate_only"):
                    # Run compliance first
                    result = await agent.execute(agent_request)
                    
                    if not result["success"]:
                        # Validation failed, stop execution
                        return [result]
                else:
                    # Run other agents
                    result = await agent.execute(agent_request)
                
                # Record execution
                execution = AgentExecution(
                    agent_name=agent_name,
                    start_time=execution_start,
                    end_time=datetime.utcnow(),
                    success=result["success"],
                    result_data=result.get("data"),
                    error=result.get("error")
                )
                
                conversation.agent_executions.append(execution.dict())
                results.append(result)
                
            except Exception as e:
                logger.error(f"Agent {agent_name} execution failed: {e}")
                results.append({
                    "success": False,
                    "agent": agent_name,
                    "error": str(e)
                })
        
        return results
    
    async def _synthesize_response(
        self,
        agent_results: List[Dict[str, Any]],
        query: str,
        intent: str
    ) -> Dict[str, Any]:
        """
        Synthesize final response from agent results
        """
        # Check if any agent failed critically
        critical_failures = [
            r for r in agent_results 
            if not r["success"] and r["agent"] == "compliance"
        ]
        
        if critical_failures:
            return {
                "message": f"I cannot process this request: {critical_failures[0]['error']}",
                "data": {},
                "actions": []
            }
        
        # Combine successful results
        combined_data = {}
        all_actions = []
        messages = []
        
        for result in agent_results:
            if result["success"]:
                agent_data = result.get("data", {})
                
                # Extract data based on agent type
                if result["agent"] == "api_executor":
                    combined_data["api_results"] = agent_data
                    if "message" in agent_data:
                        messages.append(agent_data["message"])
                        
                elif result["agent"] == "semantic_search":
                    combined_data["related_items"] = agent_data.get("results", [])
                    
                elif result["agent"] == "analytics":
                    combined_data["analytics"] = agent_data
                    if "summary" in agent_data:
                        messages.append(agent_data["summary"])
                        
                elif result["agent"] == "document_processor":
                    combined_data["document"] = agent_data
                    messages.append(f"Document processed successfully: {agent_data.get('summary', '')}")
                
                # Collect available actions
                if "actions" in agent_data:
                    all_actions.extend(agent_data["actions"])
        
        # Create natural language response
        if messages:
            final_message = " ".join(messages)
        else:
            final_message = self._generate_fallback_response(intent, query)
        
        return {
            "message": final_message,
            "data": combined_data,
            "actions": all_actions
        }
    
    def _generate_fallback_response(self, intent: str, query: str) -> str:
        """Generate fallback response when no specific message available"""
        responses = {
            "api_query": f"I'll help you find that information. Let me search for: {query}",
            "document_upload": "I'm ready to process your document. Please provide the file.",
            "workflow": f"I'll help you set up that workflow for: {query}",
            "analytics": f"I'll generate that analytics report for: {query}",
            "voice": "I'm ready to process voice input.",
            "semantic_search": f"I'll search for relevant information about: {query}"
        }
        
        return responses.get(intent, f"I'll help you with: {query}")
    
    async def _get_or_create_conversation(
        self,
        user_id: int,
        session_id: str,
        context: Dict[str, Any]
    ) -> Conversation:
        """Get existing conversation or create new one"""
        # Try cache first
        cached = await redis_connector.get_conversation(session_id)
        if cached:
            return Conversation(**cached)
        
        # Try database
        db_conv = await mongodb_connector.get_conversation(session_id)
        if db_conv:
            conv = Conversation(**db_conv)
            # Cache it
            await redis_connector.cache_conversation(session_id, conv.dict())
            return conv
        
        # Create new conversation
        conv_context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            company_id=context.get("company_id", config.CLIENT_ID),
            project_id=context.get("project_id"),
            user_role=context.get("user_role", "user"),
            preferences=context.get("preferences", {})
        )
        
        conversation = Conversation(
            id="",  # Will be set by MongoDB
            session_id=session_id,
            user_id=user_id,
            messages=[],
            context=conv_context
        )
        
        return conversation
    
    async def _save_conversation(self, conversation: Conversation):
        """Save conversation to storage"""
        try:
            # Save to MongoDB
            conv_dict = conversation.dict()
            conv_id = await mongodb_connector.save_conversation(conv_dict)
            
            if not conversation.id:
                conversation.id = conv_id
            
            # Update cache
            await redis_connector.cache_conversation(
                conversation.session_id,
                conversation.dict()
            )
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")

# Create singleton instance
master_orchestrator = MasterOrchestratorAgent()