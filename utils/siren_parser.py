from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class SirenEntity:
    """Represents a Siren sub-entity"""
    class_: List[str]
    rel: List[str]
    href: Optional[str]
    properties: Dict[str, Any]
    links: List['SirenLink']
    actions: List['SirenAction']

@dataclass
class SirenLink:
    """Represents a Siren link"""
    rel: List[str]
    href: str
    title: Optional[str] = None
    type_: Optional[str] = None

@dataclass
class SirenField:
    """Represents a field in a Siren action"""
    name: str
    type_: str = "text"
    value: Optional[Any] = None
    title: Optional[str] = None

@dataclass
class SirenAction:
    """Represents a Siren action"""
    name: str
    href: str
    method: str = "GET"
    title: Optional[str] = None
    type_: Optional[str] = None
    fields: List[SirenField] = None

@dataclass
class SirenResponse:
    """Represents a complete Siren response"""
    class_: List[str]
    properties: Dict[str, Any]
    entities: List[SirenEntity]
    actions: List[SirenAction]
    links: List[SirenLink]
    title: Optional[str] = None

class SirenParser:
    """Parser for Siren hypermedia responses"""
    
    @staticmethod
    def parse_response(data: Dict[str, Any]) -> SirenResponse:
        """Parse a Siren response into structured objects"""
        return SirenResponse(
            class_=data.get("class", []),
            properties=data.get("properties", {}),
            entities=SirenParser._parse_entities(data.get("entities", [])),
            actions=SirenParser._parse_actions(data.get("actions", [])),
            links=SirenParser._parse_links(data.get("links", [])),
            title=data.get("title")
        )
    
    @staticmethod
    def _parse_entities(entities_data: List[Dict]) -> List[SirenEntity]:
        """Parse Siren entities"""
        entities = []
        for entity_data in entities_data:
            entity = SirenEntity(
                class_=entity_data.get("class", []),
                rel=entity_data.get("rel", []),
                href=entity_data.get("href"),
                properties=entity_data.get("properties", {}),
                links=SirenParser._parse_links(entity_data.get("links", [])),
                actions=SirenParser._parse_actions(entity_data.get("actions", []))
            )
            entities.append(entity)
        return entities
    
    @staticmethod
    def _parse_links(links_data: List[Dict]) -> List[SirenLink]:
        """Parse Siren links"""
        links = []
        for link_data in links_data:
            link = SirenLink(
                rel=link_data.get("rel", []),
                href=link_data.get("href", ""),
                title=link_data.get("title"),
                type_=link_data.get("type")
            )
            links.append(link)
        return links
    
    @staticmethod
    def _parse_actions(actions_data: List[Dict]) -> List[SirenAction]:
        """Parse Siren actions"""
        actions = []
        for action_data in actions_data:
            fields = []
            for field_data in action_data.get("fields", []):
                field = SirenField(
                    name=field_data.get("name", ""),
                    type_=field_data.get("type", "text"),
                    value=field_data.get("value"),
                    title=field_data.get("title")
                )
                fields.append(field)
            
            action = SirenAction(
                name=action_data.get("name", ""),
                href=action_data.get("href", ""),
                method=action_data.get("method", "GET"),
                title=action_data.get("title"),
                type_=action_data.get("type"),
                fields=fields
            )
            actions.append(action)
        return actions
    
    @staticmethod
    def extract_data(siren_response: SirenResponse) -> Dict[str, Any]:
        """Extract data from Siren response in a simplified format"""
        result = {
            "data": siren_response.properties,
            "entities": [],
            "available_actions": []
        }
        
        # Extract entities data
        for entity in siren_response.entities:
            entity_data = {
                "type": entity.class_[0] if entity.class_ else "unknown",
                "rel": entity.rel,
                "data": entity.properties,
                "href": entity.href
            }
            result["entities"].append(entity_data)
        
        # Extract available actions
        for action in siren_response.actions:
            action_data = {
                "name": action.name,
                "method": action.method,
                "href": action.href,
                "fields": [{"name": f.name, "type": f.type_} for f in (action.fields or [])]
            }
            result["available_actions"].append(action_data)
        
        return result
    
    @staticmethod
    def find_action(siren_response: SirenResponse, action_name: str) -> Optional[SirenAction]:
        """Find a specific action by name"""
        for action in siren_response.actions:
            if action.name == action_name:
                return action
        return None
    
    @staticmethod
    def get_next_link(siren_response: SirenResponse) -> Optional[str]:
        """Get the 'next' pagination link if available"""
        for link in siren_response.links:
            if "next" in link.rel:
                return link.href
        return None