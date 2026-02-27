"""
Single Agent AI Debugging System
=================================
Reads buggy C++ code from a CSV, queries MCP server for relevant bug documentation,
uses an LLM (via OpenRouter) to detect faulty lines, and writes results to output.csv.
"""

import os
import re
import json
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from fastmcp import Client as MCPClient

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.0-flash-001"
MCP_SSE_URL = "http://localhost:8000/sse"

CHUNK_SIZE = 5        # lines per chunk
CHUNK_OVERLAP = 2     # overlap between consecutive chunks

# ─── OpenRouter LLM Client ───────────────────────────────────────────────────
llm_client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CODE NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def normalize_code(raw_code: str) -> list[str]:
    """
    Clean up formatting noise and return a list of numbered source lines.
    Returns list of tuples: [(global_line_no, cleaned_line), ...]
    """
    # Normalize line endings
    code = raw_code.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs with 4 spaces
    code = code.replace("\t", "    ")
    # Collapse multiple blank lines into one
    code = re.sub(r"\n{3,}", "\n\n", code)

    lines = code.split("\n")
    numbered_lines = []
    for i, line in enumerate(lines, start=1):
        # Strip trailing whitespace but preserve leading indentation
        cleaned = line.rstrip()
        # Collapse multiple spaces (but keep leading indent intact)
        # Only collapse interior multiple spaces
        cleaned = re.sub(r"(?<=\S)  +", " ", cleaned)
        numbered_lines.append((i, cleaned))

    # Remove trailing empty lines
    while numbered_lines and numbered_lines[-1][1] == "":
        numbered_lines.pop()

    return numbered_lines


# ═══════════════════════════════════════════════════════════════════════════════
#  2. CODE CHUNKING (overlapping sliding window)
# ═══════════════════════════════════════════════════════════════════════════════
def chunk_code(
    numbered_lines: list[tuple[int, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split numbered lines into overlapping chunks.
    Returns list of dicts: {start_line, end_line, lines: [(line_no, text), ...]}
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    total = len(numbered_lines)

    if total == 0:
        return chunks

    for start_idx in range(0, total, step):
        end_idx = min(start_idx + chunk_size, total)
        chunk_lines = numbered_lines[start_idx:end_idx]
        chunks.append(
            {
                "start_line": chunk_lines[0][0],
                "end_line": chunk_lines[-1][0],
                "lines": chunk_lines,
            }
        )
        if end_idx >= total:
            break

    return chunks


def format_chunk(chunk: dict) -> str:
    """Format a chunk into numbered code text."""
    return "\n".join(f"{ln}: {text}" for ln, text in chunk["lines"])


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MCP CLIENT — call search_documents via FastMCP Client (async)
# ═══════════════════════════════════════════════════════════════════════════════

# Global MCP client — connects to the SSE endpoint
mcp_client = MCPClient(MCP_SSE_URL)


async def call_mcp_search_async(query: str, max_retries: int = 3) -> list[dict]:
    """
    Call MCP search_documents tool via the FastMCP async Client.
    Returns list of {text, score} dicts.
    """
    for attempt in range(max_retries):
        try:
            async with mcp_client:
                result = await mcp_client.call_tool("search_documents", {"query": query})

            # result is a list of content items (TextContent, etc.)
            # Each item has a .text attribute or similar
            if result and isinstance(result, list):
                for item in result:
                    text_data = None
                    # Handle TextContent objects
                    if hasattr(item, "text"):
                        text_data = item.text
                    elif isinstance(item, dict) and "text" in item:
                        text_data = item["text"]

                    if text_data:
                        try:
                            docs = json.loads(text_data)
                            if isinstance(docs, list):
                                return docs
                        except json.JSONDecodeError:
                            # Not JSON, try treating the raw text as useful
                            continue

            # If result is a single string
            if isinstance(result, str):
                try:
                    docs = json.loads(result)
                    if isinstance(docs, list):
                        return docs
                except json.JSONDecodeError:
                    pass

            return []
        except Exception as e:
            print(f"  [MCP retry {attempt+1}/{max_retries}] {e}")
            time.sleep(2)

    print("  [MCP] All retries exhausted.")
    return []


def call_mcp_search(query: str) -> list[dict]:
    """Synchronous wrapper around the async MCP call."""
    return asyncio.run(call_mcp_search_async(query))


# ═══════════════════════════════════════════════════════════════════════════════
#  4. LLM REASONING — via OpenRouter
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert C++ debugging assistant specializing in embedded test systems,
specifically Infineon/Advantest SmartRDI APIs. You analyze code snippets to find bugs.

You will receive:
1. A chunk of C++ code with line numbers
2. The context/description of what the code should do
3. Relevant bug documentation retrieved from a knowledge base

Your task:
- Carefully compare the code against the bug documentation and the stated context.
- Identify ANY lines that contain bugs, which could include:
  • Incorrect function names or API calls
  • Wrong argument values or order
  • Wrong method call order / lifecycle violations
  • Pin name mismatches or typos
  • Value out of allowed range
  • Missing or extra parameters
  • Using wrong units
  • Logical errors
- For each bug found, return the EXACT global line number from the code chunk.

RESPOND ONLY WITH VALID JSON. No markdown, no explanations outside the JSON.

If bugs are found, respond with:
[{"line": <global_line_number>, "explanation": "<concise explanation of the bug>"}]

If NO bugs are found in this chunk, respond with:
[]
"""


def analyze_chunk(
    chunk_text: str,
    context: str,
    mcp_docs: list[dict],
    code_id: str,
) -> list[dict]:
    """
    Send chunk + context + MCP docs to LLM for bug detection.
    Returns list of {line, explanation} dicts.
    """
    # Build the knowledge base context from MCP results
    kb_text = ""
    for i, doc in enumerate(mcp_docs[:10], 1):  # limit to top 10
        score = doc.get("score", 0)
        text = doc.get("text", "")
        kb_text += f"\n--- Document {i} (relevance: {score:.2f}) ---\n{text}\n"

    if not kb_text.strip():
        kb_text = "(No relevant documentation found)"

    user_prompt = f"""## Code ID: {code_id}

## Context
{context}

## Code Chunk (with global line numbers)
```
{chunk_text}
```

## Retrieved Bug Documentation
{kb_text}

Analyze the code chunk above. Identify ALL buggy lines. Return JSON array only.
"""

    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            raw = raw.strip()

            bugs = json.loads(raw)
            if isinstance(bugs, list):
                return [
                    b for b in bugs
                    if isinstance(b, dict) and "line" in b and "explanation" in b
                ]
            return []
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [LLM retry {attempt+1}] Parse error: {e}")
            time.sleep(1)

    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  5. DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def deduplicate_bugs(bugs: list[dict]) -> list[dict]:
    """Remove duplicate (ID, BugLine) pairs, keeping the first explanation."""
    seen = set()
    unique = []
    for bug in bugs:
        key = (bug["ID"], bug["BugLine"])
        if key not in seen:
            seen.add(key)
            unique.append(bug)
    return unique


# ═══════════════════════════════════════════════════════════════════════════════
#  6. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
def process_single_code(code_id: str, faulty_code: str, context: str) -> list[dict]:
    """Full pipeline for a single code entry."""
    print(f"\n{'='*60}")
    print(f"Processing Code ID: {code_id}")
    print(f"{'='*60}")

    # Step 1: Normalize
    numbered_lines = normalize_code(faulty_code)
    print(f"  Normalized: {len(numbered_lines)} lines")

    # Step 2: Chunk
    chunks = chunk_code(numbered_lines)
    print(f"  Chunked into {len(chunks)} overlapping windows")

    all_bugs = []

    for ci, chunk in enumerate(chunks):
        chunk_text = format_chunk(chunk)
        print(f"\n  Chunk {ci+1}/{len(chunks)} (lines {chunk['start_line']}-{chunk['end_line']})")

        # Step 3: Build MCP query
        query = f"C++ bug detection: {context}\nCode:\n{chunk_text}"

        # Step 4: Call MCP search_documents
        print(f"    Querying MCP...")
        mcp_docs = call_mcp_search(query)
        print(f"    Retrieved {len(mcp_docs)} documents")

        # Step 5: LLM Reasoning
        print(f"    Running LLM analysis...")
        bugs = analyze_chunk(chunk_text, context, mcp_docs, code_id)
        print(f"    Found {len(bugs)} bug(s) in this chunk")

        for bug in bugs:
            all_bugs.append(
                {
                    "ID": code_id,
                    "BugLine": bug["line"],
                    "Explanation": bug["explanation"],
                }
            )

    return all_bugs


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Single Agent AI Debugging System")
    print("=" * 60)

    # Validate API key
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        print("\n[ERROR] Please set your OPENROUTER_API_KEY in the .env file.")
        return

    # Load dataset
    csv_path = os.path.join(os.path.dirname(__file__), "samples.csv")
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Dataset not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} code entries from samples.csv")

    # Ensure required columns exist
    required_cols = {"ID", "Code", "Context"}
    if not required_cols.issubset(set(df.columns)):
        print(f"\n[ERROR] CSV must have columns: {required_cols}")
        print(f"  Found: {set(df.columns)}")
        return

    all_results = []

    for idx, row in df.iterrows():
        code_id = str(row["ID"])
        faulty_code = str(row["Code"])
        context = str(row["Context"])

        bugs = process_single_code(code_id, faulty_code, context)
        all_results.extend(bugs)

    # Deduplicate
    all_results = deduplicate_bugs(all_results)
    print(f"\n{'='*60}")
    print(f"Total unique bugs detected: {len(all_results)}")
    print(f"{'='*60}")

    # Write output CSV
    output_path = os.path.join(os.path.dirname(__file__), "output.csv")
    result_df = pd.DataFrame(all_results, columns=["ID", "BugLine", "Explanation"])
    result_df.to_csv(output_path, index=False)
    print(f"\nResults written to: {output_path}")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
