#!/usr/bin/env python3
"""
Test if backend can run and connect to CMS API
"""
import asyncio
import sys
sys.path.append('/mnt/d/cms assitant/CmsAI - Copy/final code/backend')

async def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        from config.settings import config
        print("✓ Config loaded")
        
        from api.cms_client import cms_client
        print("✓ CMS client imported")
        
        from agents.api_executor import APIExecutorAgent
        print("✓ API Executor imported")
        
        from agents.master_orchestrator import MasterOrchestratorAgent
        print("✓ Master Orchestrator imported")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_cms_connection():
    """Test CMS API connection"""
    print("\nTesting CMS API connection...")
    try:
        from api.cms_client import cms_client
        from config.settings import config
        
        print(f"Bearer Token: {config.BEARER_TOKEN[:50]}...")
        print(f"API URL: {config.API_BASE_URL}")
        
        await cms_client.initialize()
        
        # Try a simple API call
        response = await cms_client.get("/api/users/profile")
        print(f"✓ CMS API responded: {response.get('data', {}).get('name', 'Unknown')}")
        
        await cms_client.close()
        return True
        
    except Exception as e:
        print(f"✗ CMS connection failed: {e}")
        return False

async def test_agent_execution():
    """Test if agent can execute"""
    print("\nTesting agent execution...")
    try:
        from agents.api_executor import APIExecutorAgent
        
        agent = APIExecutorAgent()
        
        # Test intent detection
        intent = await agent._detect_api_intent("Show me all pending orders")
        print(f"✓ Intent detected: {intent}")
        
        # Note: Can't test full execution without Redis/MongoDB
        print("⚠ Full execution requires Redis/MongoDB (not initialized)")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

async def check_storage_status():
    """Check if storage can be initialized"""
    print("\nChecking storage status...")
    
    results = {
        "redis": False,
        "mongodb": False,
        "faiss": False
    }
    
    # Test Redis
    try:
        from storage.redis_connector import redis_connector
        await redis_connector.initialize()
        await redis_connector.redis_client.ping()
        print("✓ Redis connection successful")
        results["redis"] = True
        await redis_connector.close()
    except Exception as e:
        print(f"✗ Redis failed: {e}")
    
    # Test MongoDB
    try:
        from storage.mongodb_connector import mongodb_connector
        await mongodb_connector.initialize()
        await mongodb_connector.client.server_info()
        print("✓ MongoDB connection successful")
        results["mongodb"] = True
        await mongodb_connector.close()
    except Exception as e:
        print(f"✗ MongoDB failed: {e}")
    
    return results

async def main():
    print("="*60)
    print("BACKEND STATUS CHECK")
    print("="*60)
    
    # Test imports
    imports_ok = await test_imports()
    
    # Test CMS connection
    cms_ok = await test_cms_connection()
    
    # Test agent
    agent_ok = await test_agent_execution()
    
    # Test storage
    storage = await check_storage_status()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"CMS API: {'✓' if cms_ok else '✗'}")
    print(f"Agents: {'✓' if agent_ok else '✗'}")
    print(f"Redis: {'✓' if storage['redis'] else '✗'}")
    print(f"MongoDB: {'✓' if storage['mongodb'] else '✗'}")
    
    print("\nBACKEND STATUS:")
    if imports_ok and cms_ok and agent_ok:
        if storage["redis"] and storage["mongodb"]:
            print("✅ FULLY OPERATIONAL - All systems working")
        else:
            print("⚠️ PARTIALLY OPERATIONAL - Core works but storage not initialized in main.py")
    else:
        print("❌ NOT OPERATIONAL - Critical components failing")

if __name__ == "__main__":
    asyncio.run(main())