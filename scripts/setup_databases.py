#!/usr/bin/env python3
"""
Database Setup Script for CMS AI Backend
Creates MongoDB collections and Redis structure - NO DUMMY DATA
"""

import asyncio
import logging
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.mongodb_connector import MongoDBConnector
from storage.redis_connector import RedisConnector
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    def __init__(self):
        self.mongo = MongoDBConnector()
        self.redis = RedisConnector()
        
    async def setup_mongodb_collections(self):
        """Create MongoDB collections - STRUCTURE ONLY, NO DATA"""
        logger.info("Creating MongoDB collections...")
        
        await self.mongo.initialize()
        db = self.mongo.db
        
        # Collections needed for the system
        collections = [
            "conversations",      # Store chat conversations
            "users",             # User profiles
            "documents",         # Uploaded documents
            "workflows",         # Automation workflows
            "analytics_reports", # Analytics data
            "api_cache",        # Cached API responses
            "agent_logs",       # Agent execution logs
            "error_logs"        # Error tracking
        ]
        
        # Create collections (empty)
        for collection_name in collections:
            try:
                if collection_name not in await db.list_collection_names():
                    # Create empty collection
                    await db.create_collection(collection_name)
                    logger.info(f"✓ Created collection: {collection_name}")
                else:
                    logger.info(f"✓ Collection exists: {collection_name}")
                    
            except Exception as e:
                logger.error(f"✗ Error creating collection {collection_name}: {e}")
        
        # Create indexes for performance (skip if already exists)
        logger.info("\nCreating indexes...")
        
        indexes_to_create = [
            ("conversations", "session_id", {}),
            ("conversations", "user_id", {}),
            ("conversations", [("created_at", -1)], {}),
            ("users", "email", {"unique": True, "sparse": True}),
            ("users", "id", {"unique": True, "sparse": True}),
            ("documents", "user_id", {}),
            ("documents", "project_id", {}),
            ("documents", [("created_at", -1)], {}),
            ("workflows", "created_by", {}),
            ("workflows", "category", {}),
            ("analytics_reports", "user_id", {}),
            ("analytics_reports", [("created_at", -1)], {}),
            ("api_cache", "endpoint", {}),
            ("api_cache", [("cached_at", -1)], {}),
            ("agent_logs", "agent_name", {}),
            ("agent_logs", [("timestamp", -1)], {}),
            ("error_logs", "error_type", {}),
            ("error_logs", [("timestamp", -1)], {}),
        ]
        
        for collection_name, index_field, options in indexes_to_create:
            try:
                collection = db[collection_name]
                await collection.create_index(index_field, **options)
                logger.info(f"  ✓ Index created: {collection_name}.{index_field}")
            except Exception as e:
                if "IndexKeySpecsConflict" in str(e) or "already exists" in str(e).lower():
                    logger.info(f"  ✓ Index exists: {collection_name}.{index_field}")
                else:
                    logger.warning(f"  ⚠ Index error on {collection_name}.{index_field}: {e}")
        
        logger.info("✓ All indexes ready")
        
    async def setup_redis_structure(self):
        """Setup Redis key namespaces - NO DATA"""
        logger.info("\nSetting up Redis namespaces...")
        
        await self.redis.initialize()
        
        # Just define the key patterns we'll use (documentation)
        key_patterns = [
            "rate_limit:*",      # Rate limiting per user/IP
            "session:*",         # Active sessions
            "cache:api:*",       # API response cache
            "metrics:*",         # System metrics
            "feature:*",         # Feature flags
            "status:*",          # Service status
            "queue:*",           # Job queues
            "lock:*"            # Distributed locks
        ]
        
        logger.info("Redis namespaces defined:")
        for pattern in key_patterns:
            logger.info(f"  - {pattern}")
            
        # Clear any test data from previous runs
        test_keys = await self.redis.redis_client.keys("test:*")
        if test_keys:
            for key in test_keys:
                await self.redis.redis_client.delete(key)
            logger.info(f"✓ Cleaned {len(test_keys)} test keys")
            
    async def verify_setup(self):
        """Verify database setup"""
        logger.info("\n" + "="*50)
        logger.info("VERIFYING DATABASE SETUP")
        logger.info("="*50)
        
        # Check MongoDB
        try:
            await self.mongo.initialize()
            info = await self.mongo.client.server_info()
            logger.info(f"✓ MongoDB: Connected (Version: {info.get('version', 'unknown')})")
            
            collections = await self.mongo.db.list_collection_names()
            logger.info(f"  Collections created: {len(collections)}")
            for col in collections:
                # Get index info instead of document count
                indexes = await self.mongo.db[col].index_information()
                logger.info(f"    - {col}: {len(indexes)} indexes")
                
        except Exception as e:
            logger.error(f"✗ MongoDB: {e}")
        
        # Check Redis
        try:
            await self.redis.initialize()
            
            # Test basic operations
            await self.redis.redis_client.set("test:connection", "ok", ex=1)
            test_value = await self.redis.redis_client.get("test:connection")
            
            if test_value:
                logger.info(f"✓ Redis: Connected and operational")
            
            # Check memory info
            info = await self.redis.redis_client.info("memory")
            used_memory = info.get("used_memory_human", "unknown")
            logger.info(f"  Memory used: {used_memory}")
                
        except Exception as e:
            logger.error(f"✗ Redis: {e}")
            
    async def main(self):
        """Main setup process"""
        logger.info("="*50)
        logger.info("CMS AI DATABASE SETUP (STRUCTURE ONLY)")
        logger.info("="*50)
        logger.info(f"MongoDB: {config.MONGODB_URI.split('@')[1]}")  # Hide password
        logger.info(f"Redis: {config.REDIS_HOST}:{config.REDIS_PORT}")
        logger.info("="*50 + "\n")
        
        try:
            # Setup MongoDB collections and indexes
            await self.setup_mongodb_collections()
            logger.info("✓ MongoDB structure ready\n")
            
            # Setup Redis namespaces
            await self.setup_redis_structure()
            logger.info("✓ Redis structure ready\n")
            
            # Verify everything
            await self.verify_setup()
            
            logger.info("\n" + "="*50)
            logger.info("✓ DATABASE SETUP COMPLETE - NO DATA ADDED")
            logger.info("  Ready for production use!")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"\n✗ Setup failed: {e}")
            raise
        finally:
            # Cleanup
            await self.mongo.close()
            await self.redis.close()

if __name__ == "__main__":
    setup = DatabaseSetup()
    asyncio.run(setup.main())