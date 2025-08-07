import logging
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator
import asyncio
from datetime import datetime, timedelta
import base64
import io
import json
import re

from agents.base_agent import BaseAgent
from storage import mongodb_connector, redis_connector
from config.settings import config
from utils.groq_whisper import groq_whisper_client

logger = logging.getLogger(__name__)

class VoiceNLUAgent(BaseAgent):
    """
    The Ears and Mouth - handles all voice interactions and natural language understanding
    Uses Groq Whisper API for ultra-fast speech-to-text (216x real-time)
    """
    
    def __init__(self):
        super().__init__(
            name="voice_nlu",
            description="Handles voice transcription with Groq Whisper and natural language understanding"
        )
        self.construction_vocabulary = self._load_construction_vocabulary()
        self.intent_patterns = self._load_intent_patterns()
        self.entity_extractors = self._initialize_entity_extractors()
    
    def _load_construction_vocabulary(self) -> Dict[str, List[str]]:
        """Load construction-specific vocabulary"""
        return {
            "materials": [
                "rebar", "concrete", "steel", "lumber", "drywall", 
                "insulation", "plywood", "cement", "aggregate", "asphalt"
            ],
            "equipment": [
                "excavator", "crane", "bulldozer", "backhoe", "forklift",
                "scaffolding", "generator", "compressor", "jackhammer"
            ],
            "actions": [
                "pour", "excavate", "frame", "install", "inspect",
                "measure", "level", "plumb", "weld", "fasten"
            ],
            "measurements": [
                "square feet", "cubic yards", "linear feet", "tons",
                "psi", "gauge", "grade", "elevation", "slope"
            ],
            "roles": [
                "foreman", "superintendent", "contractor", "subcontractor",
                "inspector", "engineer", "architect", "surveyor"
            ],
            "documents": [
                "blueprint", "permit", "inspection", "change order",
                "RFI", "submittal", "shop drawing", "as-built"
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load NLU intent patterns"""
        return {
            "create_order": {
                "patterns": [
                    r"(?:create|make|place|new)\s+(?:an?\s+)?order",
                    r"order\s+(?:some|the)?\s*(\w+)",
                    r"need\s+(?:to\s+)?(?:order|purchase|buy)"
                ],
                "required_entities": ["item_type"],
                "optional_entities": ["quantity", "urgency", "supplier"]
            },
            "check_status": {
                "patterns": [
                    r"(?:check|what\'s|show)\s+(?:the\s+)?status",
                    r"where\s+is\s+(?:my|the|our)\s+(\w+)",
                    r"status\s+(?:of|on)\s+(?:order|project|delivery)"
                ],
                "required_entities": ["entity_type"],
                "optional_entities": ["entity_id", "time_frame"]
            },
            "schedule_inspection": {
                "patterns": [
                    r"schedule\s+(?:an?\s+)?inspection",
                    r"need\s+(?:an?\s+)?inspector",
                    r"(?:book|arrange)\s+(?:a\s+)?(?:safety|quality)?\s*inspection"
                ],
                "required_entities": ["inspection_type"],
                "optional_entities": ["date", "time", "location"]
            },
            "report_issue": {
                "patterns": [
                    r"(?:report|found|there\'s)\s+(?:a\s+)?(?:problem|issue)",
                    r"(?:safety|quality)\s+(?:concern|violation)",
                    r"(?:defect|damage)\s+(?:in|at|on)"
                ],
                "required_entities": ["issue_type"],
                "optional_entities": ["location", "severity", "description"]
            },
            "request_materials": {
                "patterns": [
                    r"need\s+(?:more\s+)?(\w+)",
                    r"running\s+(?:low|out)\s+(?:of|on)",
                    r"deliver\s+(?:some|more)?\s*(\w+)"
                ],
                "required_entities": ["material_type"],
                "optional_entities": ["quantity", "delivery_date", "location"]
            }
        }
    
    def _initialize_entity_extractors(self) -> Dict[str, Any]:
        """Initialize entity extraction patterns"""
        return {
            "quantity": {
                "pattern": r"(\d+(?:\.\d+)?)\s*(?:units?|pieces?|tons?|yards?|feet|meters?|kg|lbs?)",
                "type": "number_with_unit"
            },
            "date": {
                "patterns": [
                    r"(?:today|tomorrow|yesterday)",
                    r"(?:next|this|last)\s+(?:week|month)",
                    r"(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                    r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})"
                ],
                "type": "temporal"
            },
            "location": {
                "patterns": [
                    r"(?:at|in|on)\s+(?:the\s+)?(\w+(?:\s+\w+)*)",
                    r"(?:site|area|zone|section)\s+([A-Z0-9]+)",
                    r"(?:building|floor|level)\s+(\w+)"
                ],
                "type": "spatial"
            },
            "urgency": {
                "keywords": ["urgent", "asap", "emergency", "priority", "rush", "immediately"],
                "type": "priority_level"
            },
            "entity_id": {
                "pattern": r"(?:#|number|id)?\s*(\d+)",
                "type": "identifier"
            }
        }
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate voice/NLU request"""
        return "audio_data" in request or "text" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process voice or text input for natural language understanding
        """
        try:
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Determine input type
            if "audio_data" in request:
                # Process voice input with Groq Whisper
                transcription = await self._transcribe_audio_with_groq(
                    request["audio_data"],
                    parameters
                )
                text_input = transcription["text"]
                confidence = transcription["confidence"]
                
                # Add transcription metadata
                metadata = {
                    "input_type": "voice",
                    "transcription_confidence": confidence,
                    "duration": transcription.get("duration", 0),
                    "processing_time": transcription.get("processing_time", 0),
                    "real_time_factor": transcription.get("real_time_factor", 0),
                    "estimated_cost": transcription.get("estimated_cost", 0),
                    "stt_engine": "groq_whisper"
                }
            else:
                # Process text input
                text_input = request.get("text", "")
                metadata = {"input_type": "text"}
            
            # Clean and normalize input
            normalized_text = self._normalize_text(text_input)
            
            # Extract intent and entities
            nlu_result = await self._analyze_text(normalized_text, context)
            
            # Enhance with context
            enhanced_result = await self._enhance_with_context(
                nlu_result,
                context,
                metadata
            )
            
            # Format response
            response = self._format_nlu_response(enhanced_result, text_input)
            
            return {
                "data": response,
                "metadata": {
                    **metadata,
                    "processing_time": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Voice/NLU processing error: {e}", exc_info=True)
            raise
    
    async def _transcribe_audio_with_groq(
        self,
        audio_data: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transcribe audio to text using Groq Whisper API"""
        try:
            # Determine audio format and noise level
            audio_format = parameters.get("format", "wav")
            noise_level = parameters.get("noise_level", "normal")
            
            # Use Groq Whisper with construction context
            result = await groq_whisper_client.transcribe_with_construction_context(
                audio_data=audio_data,
                audio_format=audio_format,
                noise_level=noise_level
            )
            
            # Log performance metrics
            if result.get("real_time_factor"):
                logger.info(
                    f"Groq Whisper transcription completed: "
                    f"{result['real_time_factor']:.1f}x real-time, "
                    f"cost: ${result.get('estimated_cost', 0):.4f}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Groq transcription error: {e}")
            # Fallback response
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        normalized = text.lower()
        
        # Expand contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        return normalized
    
    async def _analyze_text(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze text for intent and entities"""
        # Detect intent
        detected_intent = None
        intent_confidence = 0.0
        matched_pattern = None
        
        for intent_name, intent_config in self.intent_patterns.items():
            for pattern in intent_config["patterns"]:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    detected_intent = intent_name
                    intent_confidence = 0.85  # Base confidence
                    matched_pattern = pattern
                    break
            
            if detected_intent:
                break
        
        # Extract entities
        entities = await self._extract_entities(text, detected_intent)
        
        # Calculate final confidence based on entity completeness
        if detected_intent:
            required_entities = self.intent_patterns[detected_intent]["required_entities"]
            found_required = all(
                entity in entities 
                for entity in required_entities
            )
            
            if found_required:
                intent_confidence = min(intent_confidence + 0.1, 1.0)
            else:
                intent_confidence = max(intent_confidence - 0.2, 0.3)
        
        return {
            "intent": detected_intent or "unknown",
            "confidence": intent_confidence,
            "entities": entities,
            "matched_pattern": matched_pattern,
            "original_text": text
        }
    
    async def _extract_entities(
        self,
        text: str,
        intent: Optional[str]
    ) -> Dict[str, Any]:
        """Extract entities from text"""
        entities = {}
        
        # Extract using patterns
        for entity_type, extractor in self.entity_extractors.items():
            if "pattern" in extractor:
                match = re.search(extractor["pattern"], text, re.IGNORECASE)
                if match:
                    entities[entity_type] = {
                        "value": match.group(1),
                        "type": extractor["type"],
                        "confidence": 0.8
                    }
            
            elif "patterns" in extractor:
                for pattern in extractor["patterns"]:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        entities[entity_type] = {
                            "value": match.group(0),
                            "type": extractor["type"],
                            "confidence": 0.8
                        }
                        break
            
            elif "keywords" in extractor:
                for keyword in extractor["keywords"]:
                    if keyword in text.lower():
                        entities[entity_type] = {
                            "value": keyword,
                            "type": extractor["type"],
                            "confidence": 0.9
                        }
                        break
        
        # Extract construction-specific entities
        entities.update(await self._extract_construction_entities(text))
        
        # Parse dates and times
        if "date" in entities:
            entities["date"]["parsed"] = self._parse_date(entities["date"]["value"])
        
        return entities
    
    async def _extract_construction_entities(self, text: str) -> Dict[str, Any]:
        """Extract construction-specific entities"""
        entities = {}
        
        # Materials
        for material in self.construction_vocabulary["materials"]:
            if material in text.lower():
                entities["material_type"] = {
                    "value": material,
                    "type": "construction_material",
                    "confidence": 0.95
                }
                break
        
        # Equipment
        for equipment in self.construction_vocabulary["equipment"]:
            if equipment in text.lower():
                entities["equipment_type"] = {
                    "value": equipment,
                    "type": "construction_equipment",
                    "confidence": 0.95
                }
                break
        
        # Extract quantities with units
        quantity_pattern = r"(\d+(?:\.\d+)?)\s*(tons?|yards?|feet|meters?|units?|pieces?)"
        quantity_match = re.search(quantity_pattern, text, re.IGNORECASE)
        if quantity_match:
            entities["quantity"] = {
                "value": float(quantity_match.group(1)),
                "unit": quantity_match.group(2),
                "type": "measurement",
                "confidence": 0.9
            }
        
        return entities
    
    async def _enhance_with_context(
        self,
        nlu_result: Dict[str, Any],
        context: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance NLU result with contextual information"""
        enhanced = nlu_result.copy()
        
        # Add user context
        enhanced["user_context"] = {
            "user_id": context.get("user_id"),
            "role": context.get("user_role"),
            "project_id": context.get("project_id"),
            "location": context.get("location")
        }
        
        # Check conversation history for entity resolution
        if context.get("conversation_id"):
            # Get recent conversation
            conversation = await redis_connector.get_conversation(
                context["conversation_id"]
            )
            
            if conversation:
                # Resolve pronouns and references
                enhanced = self._resolve_references(enhanced, conversation)
        
        # Add confidence adjustments based on context
        if metadata.get("input_type") == "voice":
            # Adjust confidence based on transcription quality
            trans_confidence = metadata.get("transcription_confidence", 1.0)
            enhanced["confidence"] *= trans_confidence
        
        # Add domain-specific enhancements
        if enhanced["intent"] == "create_order" and "project_id" in context:
            enhanced["entities"]["project_id"] = {
                "value": context["project_id"],
                "type": "context",
                "confidence": 1.0
            }
        
        return enhanced
    
    def _resolve_references(
        self,
        nlu_result: Dict[str, Any],
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve pronouns and references from conversation history"""
        # Look for pronouns in original text
        text = nlu_result.get("original_text", "").lower()
        
        # Simple pronoun resolution
        if "it" in text or "that" in text:
            # Find last mentioned entity
            for message in reversed(conversation.get("messages", [])):
                if message.get("role") == "assistant":
                    # Look for entity mentions in assistant responses
                    content = message.get("content", "")
                    
                    # Extract order numbers
                    order_match = re.search(r"order #(\d+)", content, re.IGNORECASE)
                    if order_match and "entity_id" not in nlu_result["entities"]:
                        nlu_result["entities"]["entity_id"] = {
                            "value": order_match.group(1),
                            "type": "reference",
                            "confidence": 0.7
                        }
                        nlu_result["entities"]["entity_type"] = {
                            "value": "order",
                            "type": "reference",
                            "confidence": 0.7
                        }
                        break
        
        return nlu_result
    
    def _format_nlu_response(
        self,
        enhanced_result: Dict[str, Any],
        original_text: str
    ) -> Dict[str, Any]:
        """Format NLU response for use by other agents"""
        # Create structured query for orchestrator
        structured_query = self._build_structured_query(enhanced_result)
        
        # Generate natural language interpretation
        interpretation = self._generate_interpretation(enhanced_result)
        
        # Suggest actions based on intent
        suggested_actions = self._suggest_actions(enhanced_result)
        
        return {
            "original_text": original_text,
            "intent": enhanced_result["intent"],
            "confidence": enhanced_result["confidence"],
            "entities": enhanced_result["entities"],
            "structured_query": structured_query,
            "interpretation": interpretation,
            "suggested_actions": suggested_actions,
            "requires_clarification": self._needs_clarification(enhanced_result)
        }
    
    def _build_structured_query(self, result: Dict[str, Any]) -> str:
        """Build structured query from NLU result"""
        intent = result["intent"]
        entities = result["entities"]
        
        if intent == "create_order":
            material = entities.get("material_type", {}).get("value", "materials")
            quantity = entities.get("quantity", {})
            
            query = f"Create order for {quantity.get('value', '')} {quantity.get('unit', '')} of {material}"
            
            if entities.get("urgency", {}).get("value") == "urgent":
                query += " (URGENT)"
                
        elif intent == "check_status":
            entity_type = entities.get("entity_type", {}).get("value", "item")
            entity_id = entities.get("entity_id", {}).get("value", "")
            
            query = f"Check status of {entity_type}"
            if entity_id:
                query += f" #{entity_id}"
                
        elif intent == "schedule_inspection":
            inspection_type = entities.get("inspection_type", {}).get("value", "general")
            date = entities.get("date", {}).get("parsed", "")
            
            query = f"Schedule {inspection_type} inspection"
            if date:
                query += f" for {date}"
                
        else:
            # Fallback to original text
            query = result.get("original_text", "")
        
        return query
    
    def _generate_interpretation(self, result: Dict[str, Any]) -> str:
        """Generate natural language interpretation"""
        intent = result["intent"]
        entities = result["entities"]
        confidence = result["confidence"]
        
        if confidence < 0.5:
            return "I'm not quite sure what you're asking for. Could you please rephrase?"
        
        interpretations = {
            "create_order": "You want to create a new order",
            "check_status": "You're checking on the status",
            "schedule_inspection": "You need to schedule an inspection",
            "report_issue": "You're reporting an issue",
            "request_materials": "You need materials delivered"
        }
        
        base_interpretation = interpretations.get(intent, "I understand you need help")
        
        # Add entity details
        if "material_type" in entities:
            base_interpretation += f" for {entities['material_type']['value']}"
        
        if "quantity" in entities:
            q = entities["quantity"]
            base_interpretation += f" ({q['value']} {q.get('unit', 'units')})"
        
        if "date" in entities:
            base_interpretation += f" by {entities['date']['value']}"
        
        return base_interpretation
    
    def _suggest_actions(self, result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest possible actions based on intent"""
        intent = result["intent"]
        suggestions = []
        
        if intent == "create_order":
            suggestions = [
                {"action": "create_order", "label": "Create Order"},
                {"action": "check_inventory", "label": "Check Inventory First"},
                {"action": "get_quotes", "label": "Get Supplier Quotes"}
            ]
            
        elif intent == "check_status":
            suggestions = [
                {"action": "view_details", "label": "View Details"},
                {"action": "track_delivery", "label": "Track Delivery"},
                {"action": "contact_supplier", "label": "Contact Supplier"}
            ]
            
        elif intent == "schedule_inspection":
            suggestions = [
                {"action": "schedule_now", "label": "Schedule Now"},
                {"action": "view_calendar", "label": "View Calendar"},
                {"action": "check_requirements", "label": "Check Requirements"}
            ]
        
        return suggestions
    
    def _needs_clarification(self, result: Dict[str, Any]) -> bool:
        """Check if clarification is needed"""
        # Low confidence
        if result["confidence"] < 0.6:
            return True
        
        # Missing required entities
        if result["intent"] in self.intent_patterns:
            required = self.intent_patterns[result["intent"]]["required_entities"]
            found_entities = set(result["entities"].keys())
            
            if not all(req in found_entities for req in required):
                return True
        
        return False
    
    def _parse_date(self, date_str: str) -> str:
        """Parse date string to ISO format"""
        date_lower = date_str.lower()
        today = datetime.utcnow().date()
        
        if date_lower == "today":
            return today.isoformat()
        elif date_lower == "tomorrow":
            return (today + timedelta(days=1)).isoformat()
        elif date_lower == "yesterday":
            return (today - timedelta(days=1)).isoformat()
        elif "next week" in date_lower:
            return (today + timedelta(days=7)).isoformat()
        elif "next month" in date_lower:
            return (today + timedelta(days=30)).isoformat()
        else:
            # Try to parse specific date formats
            try:
                # Handle MM/DD/YYYY or MM-DD-YYYY
                for fmt in ["%m/%d/%Y", "%m-%d-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
                    try:
                        parsed = datetime.strptime(date_str, fmt)
                        return parsed.date().isoformat()
                    except ValueError:
                        continue
            except:
                pass
        
        # Return original if can't parse
        return date_str
    
    async def process_streaming_audio(
        self,
        audio_stream: AsyncIterator[bytes],
        context: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process streaming audio for real-time transcription with Groq"""
        # Use Groq's streaming capability
        async for result in groq_whisper_client.process_streaming_audio(
            audio_stream,
            context
        ):
            # Add NLU processing to each transcription result
            if result.get("text"):
                nlu_result = await self._analyze_text(result["text"], context)
                
                yield {
                    "type": result["type"],
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "timestamp": result["timestamp"],
                    "nlu": {
                        "intent": nlu_result["intent"],
                        "entities": nlu_result["entities"],
                        "confidence": nlu_result["confidence"]
                    }
                }

# Create singleton instance
voice_nlu = VoiceNLUAgent()