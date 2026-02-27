import os
import re
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

from fastmcp import Client as MCPClient

load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
MCP_HOST = "localhost"
MCP_PORT = 8000
MCP_SSE_URL = f"http://{MCP_HOST}:{MCP_PORT}/sse"

CHUNK_SIZE = 6
CHUNK_OVERLAP = 2

# ─── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0
)

# ─── MCP Client ───────────────────────────────────────────────────────────────
mcp_client = MCPClient(MCP_SSE_URL)

# ═══════════════════════════════════════════════════════════════════════════════
# Normalization + Chunking
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_code(code: str):
    code = re.sub(r"\r\n|\r", "\n", code).replace("\t", "    ")
    lines = code.split("\n")
    return [(i+1, line.rstrip()) for i, line in enumerate(lines)]

def chunk_code(lines):
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []

    for i in range(0, len(lines), step):
        block = lines[i:i+CHUNK_SIZE]
        chunks.append(block)
        if i + CHUNK_SIZE >= len(lines):
            break

    return chunks

def format_chunk(chunk):
    return "\n".join(f"{ln}: {txt}" for ln, txt in chunk)

# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def mcp_search(query: str) -> str:
    """Search MCP documentation for bug related context."""

    async def _run():
        async with mcp_client:
            result = await mcp_client.call_tool("search_documents", {"query": query})
        
        docs = []
        # Handle CallToolResult - extract the content
        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, list):
                docs = [item.get("text", "") if isinstance(item, dict) else (item.text if hasattr(item, "text") else str(item)) for item in content]
            elif isinstance(content, dict):
                text = content.get("text", "")
                if text:
                    docs = [text]
        elif isinstance(result, list):
            docs = [item.get("text", "") if isinstance(item, dict) else (item.text if hasattr(item, "text") else str(item)) for item in result]
        
        return "\n".join(docs[:6]) if docs else "No documentation found."

    return asyncio.run(_run())


@tool
def detect_bugs(payload: str) -> str:
    """
    Detect bugs from given payload JSON.
    Payload JSON:
    {
      "chunk": "...",
      "context": "..."
    }
    """

    try:
        # Try to parse the payload as JSON
        data = json.loads(payload)
    except json.JSONDecodeError:
        # If it fails, try to extract the JSON more carefully
        try:
            # Remove any trailing/leading whitespace and control characters
            cleaned = payload.encode('utf-8', 'ignore').decode('utf-8')
            data = json.loads(cleaned)
        except:
            return "[]"

    chunk_text = data.get('chunk', '')
    context = data.get('context', '')

    prompt = f"""
You are a C++ debugging expert.

Context:
{context}

Code Chunk:
{chunk_text}

Return only JSON (no markdown, no extra text):
[{{"line": <line_number>, "explanation": "<bug_description>"}}]
"""

    try:
        response = llm.invoke(prompt).content.strip()
        response = re.sub(r"```json|```", "", response).strip()
        
        json.loads(response)
        return response
    except:
        return "[]"


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

system_prompt = """
You are an autonomous C++ AI debugging agent.

You can:
- Search documentation using mcp_search
- Analyze code using detect_bugs

For each chunk:
1. Search docs
2. Detect bugs

Return only valid JSON:
[{"line": <line>, "explanation": "<bug>"}]
"""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT CREATION
# ═══════════════════════════════════════════════════════════════════════════════

tools = [mcp_search, detect_bugs]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_single_code(code_id, faulty_code, context):
    lines = normalize_code(faulty_code)
    chunks = chunk_code(lines)

    all_bugs = []

    for chunk in chunks:
        chunk_text = format_chunk(chunk)

        task = json.dumps({
            "chunk": chunk_text,
            "context": context
        })

        result = agent.invoke({
            "messages": [{"role": "user", "content": task}]
        })

        try:
            output = result.get("messages", [{}])[-1].get("content", "[]")
            bugs = json.loads(output)
            for b in bugs:
                all_bugs.append({
                    "ID": code_id,
                    "BugLine": b["line"],
                    "Explanation": b["explanation"]
                })
        except:
            pass

    return all_bugs

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    df = pd.read_csv("samples.csv")
    results = []

    for _, row in df.iterrows():
        results.extend(
            process_single_code(
                str(row["ID"]),
                str(row["Code"]),
                str(row["Context"])
            )
        )

    out = pd.DataFrame(results).drop_duplicates()
    out.to_csv("output.csv", index=False)
    print("\n Debugging completed")
    print(out)

if __name__ == "__main__":
    main()