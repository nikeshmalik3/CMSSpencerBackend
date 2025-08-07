#!/usr/bin/env python3
"""Test all imports to find missing dependencies"""

import sys
import traceback

errors = []

# Test each module
modules_to_test = [
    "fastapi",
    "uvicorn", 
    "aiohttp",
    "aiofiles",
    "motor",
    "pymongo",
    "redis",
    "openai",
    "groq",
    "numpy",
    "PyMuPDF",
    "pymupdf4llm",
    "openpyxl",
    "docx",
    "PIL",
    "jwt",
    "jose",
    "pydantic",
    "email_validator",
    "dotenv",
    "orjson",
    "structlog",
    "psutil"
]

print("Testing imports...")
for module in modules_to_test:
    try:
        if module == "PyMuPDF":
            import fitz
        elif module == "PIL":
            from PIL import Image
        elif module == "jose":
            from jose import jwt
        elif module == "dotenv":
            from dotenv import load_dotenv
        elif module == "docx":
            import docx
        else:
            __import__(module)
        print(f"✓ {module}")
    except ImportError as e:
        errors.append(f"✗ {module}: {e}")
        print(f"✗ {module}: {e}")
    except Exception as e:
        errors.append(f"✗ {module}: Unexpected error: {e}")
        print(f"✗ {module}: Unexpected error: {e}")

if errors:
    print("\n=== MISSING MODULES ===")
    for error in errors:
        print(error)
    sys.exit(1)
else:
    print("\n✓ All imports successful!")
    sys.exit(0)