import motor.motor_asyncio
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime
import pymongo
from bson import ObjectId

from config.settings import config

logger = logging.getLogger(__name__)

class MongoDBConnector:
    """Async MongoDB connector for persistent storage"""
    
    def __init__(self):
        self.client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self.db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            # Configure connection options
            connection_kwargs = {
                "maxPoolSize": config.MONGODB_MAX_POOL_SIZE,
                "serverSelectionTimeoutMS": 10000,
                "directConnection": True
            }
            
            # Add TLS configuration only if SSL is enabled
            if config.MONGODB_USE_SSL:
                connection_kwargs["tls"] = True
                connection_kwargs["tlsAllowInvalidCertificates"] = True  # For self-signed certs
                connection_kwargs["tlsAllowInvalidHostnames"] = True
            
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                config.MONGODB_URI,
                **connection_kwargs
            )
            
            self.db = self.client[config.MONGODB_DATABASE]
            
            # Test connection
            await self.client.server_info()
            logger.info(f"MongoDB connection established successfully (SSL: {config.MONGODB_USE_SSL})")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def _create_indexes(self):
        """Create necessary indexes for collections"""
        try:
            # Conversations indexes
            conversations = self.db.conversations
            await conversations.create_index("session_id")
            await conversations.create_index("user_id")
            await conversations.create_index([("created_at", pymongo.DESCENDING)])
            
            # Users indexes
            users = self.db.users
            await users.create_index("email", unique=True)
            await users.create_index("id", unique=True)
            
            # Documents indexes
            documents = self.db.documents
            await documents.create_index("user_id")
            await documents.create_index("project_id")
            await documents.create_index("tags")
            await documents.create_index([("created_at", pymongo.DESCENDING)])
            
            # Workflows indexes
            workflows = self.db.workflows
            await workflows.create_index("created_by")
            await workflows.create_index("category")
            
            # Analytics indexes
            analytics = self.db.analytics_reports
            await analytics.create_index("user_id")
            await analytics.create_index([("created_at", pymongo.DESCENDING)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    # Conversation Methods
    async def save_conversation(self, conversation: Dict[str, Any]) -> str:
        """Save conversation to database"""
        try:
            # Convert id to ObjectId if it exists
            if "id" in conversation:
                conversation["_id"] = ObjectId(conversation.pop("id")) if conversation["id"] else ObjectId()
            
            conversation["updated_at"] = datetime.utcnow()
            
            result = await self.db.conversations.replace_one(
                {"_id": conversation.get("_id", ObjectId())},
                conversation,
                upsert=True
            )
            
            return str(conversation.get("_id", result.upserted_id))
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation by ID"""
        try:
            conversation = await self.db.conversations.find_one(
                {"_id": ObjectId(conversation_id)}
            )
            
            if conversation:
                conversation["id"] = str(conversation.pop("_id"))
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            return None
    
    async def get_user_conversations(self, user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get user's recent conversations"""
        try:
            cursor = self.db.conversations.find(
                {"user_id": user_id, "active": True}
            ).sort("updated_at", pymongo.DESCENDING).limit(limit)
            
            conversations = []
            async for conv in cursor:
                conv["id"] = str(conv.pop("_id"))
                conversations.append(conv)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    # User Methods
    async def save_user(self, user: Dict[str, Any]) -> bool:
        """Save or update user"""
        try:
            user["updated_at"] = datetime.utcnow()
            
            await self.db.users.replace_one(
                {"id": user["id"]},
                user,
                upsert=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving user: {e}")
            return False
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = await self.db.users.find_one({"id": user_id})
            return user
            
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None
    
    # Document Methods
    async def save_document(self, document: Dict[str, Any]) -> str:
        """Save document metadata"""
        try:
            if "id" in document:
                document["_id"] = ObjectId(document.pop("id")) if document["id"] else ObjectId()
            
            result = await self.db.documents.replace_one(
                {"_id": document.get("_id", ObjectId())},
                document,
                upsert=True
            )
            
            return str(document.get("_id", result.upserted_id))
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        try:
            document = await self.db.documents.find_one(
                {"_id": ObjectId(document_id)}
            )
            
            if document:
                document["id"] = str(document.pop("_id"))
            
            return document
            
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            return None
    
    async def search_documents(self, filters: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Search documents with filters"""
        try:
            cursor = self.db.documents.find(filters).limit(limit)
            
            documents = []
            async for doc in cursor:
                doc["id"] = str(doc.pop("_id"))
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    # Workflow Methods
    async def save_workflow(self, workflow: Dict[str, Any]) -> str:
        """Save workflow template"""
        try:
            if "id" in workflow:
                workflow["_id"] = ObjectId(workflow.pop("id")) if workflow["id"] else ObjectId()
            
            workflow["updated_at"] = datetime.utcnow()
            
            result = await self.db.workflows.replace_one(
                {"_id": workflow.get("_id", ObjectId())},
                workflow,
                upsert=True
            )
            
            return str(workflow.get("_id", result.upserted_id))
            
        except Exception as e:
            logger.error(f"Error saving workflow: {e}")
            raise
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow by ID"""
        try:
            workflow = await self.db.workflows.find_one(
                {"_id": ObjectId(workflow_id)}
            )
            
            if workflow:
                workflow["id"] = str(workflow.pop("_id"))
            
            return workflow
            
        except Exception as e:
            logger.error(f"Error getting workflow {workflow_id}: {e}")
            return None
    
    async def get_workflows_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get workflows by category"""
        try:
            cursor = self.db.workflows.find({"category": category})
            
            workflows = []
            async for wf in cursor:
                wf["id"] = str(wf.pop("_id"))
                workflows.append(wf)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Error getting workflows by category: {e}")
            return []
    
    # Analytics Methods
    async def save_analytics_report(self, report: Dict[str, Any]) -> str:
        """Save analytics report"""
        try:
            if "id" in report:
                report["_id"] = ObjectId(report.pop("id")) if report["id"] else ObjectId()
            
            result = await self.db.analytics_reports.insert_one(report)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving analytics report: {e}")
            raise
    
    async def get_user_reports(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's analytics reports"""
        try:
            cursor = self.db.analytics_reports.find(
                {"user_id": user_id}
            ).sort("created_at", pymongo.DESCENDING).limit(limit)
            
            reports = []
            async for report in cursor:
                report["id"] = str(report.pop("_id"))
                reports.append(report)
            
            return reports
            
        except Exception as e:
            logger.error(f"Error getting user reports: {e}")
            return []
    
    # Generic Methods
    async def find_one(self, collection: str, filter: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generic find one document"""
        try:
            document = await self.db[collection].find_one(filter)
            
            if document and "_id" in document:
                document["id"] = str(document.pop("_id"))
            
            return document
            
        except Exception as e:
            logger.error(f"Error finding document in {collection}: {e}")
            return None
    
    async def find_many(self, collection: str, filter: Dict[str, Any], limit: int = 50) -> List[Dict[str, Any]]:
        """Generic find many documents"""
        try:
            cursor = self.db[collection].find(filter).limit(limit)
            
            documents = []
            async for doc in cursor:
                if "_id" in doc:
                    doc["id"] = str(doc.pop("_id"))
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error finding documents in {collection}: {e}")
            return []

# Singleton instance
mongodb_connector = MongoDBConnector()