
import os
import json
import re
import time
import pandas as pd
import httpx

from dotenv import load_dotenv
from tqdm import tqdm

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()


GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
MCP_SERVER_URL  = os.getenv("MCP_SERVER_URL", "http://localhost:8003")
DATASET_PATH    = "samples.csv"
OUTPUT_CSV      = os.getenv("OUTPUT_CSV",     "./output.csv")
AUDIT_JSON      = os.getenv("AUDIT_JSON",     "./audit.json")
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE", "0.60"))


_dataset: pd.DataFrame = None

def load_dataset() -> pd.DataFrame:
    global _dataset
    if _dataset is None:
        print(f"[Dataset] Loading from: {DATASET_PATH}")
        _dataset = pd.read_csv(DATASET_PATH)
        print(f"[Dataset] Loaded {len(_dataset)} rows")
    return _dataset


class MCPClient:
    def __init__(self):
        self.base_url = MCP_SERVER_URL
        self.client   = httpx.Client(timeout=30.0)

    def search_documents(self, query: str) -> list:
        print(f"[MCP] Searching: '{query}'")
        try:
            url      = f"{self.base_url}/tools/search_documents"
            response = self.client.post(url, json={"arguments": {"query": query}})
            response.raise_for_status()
            docs = response.json()
            docs = docs if isinstance(docs, list) else docs.get("result", [])
            filtered = [d for d in docs if d.get("score", 0) > 0.5]
            filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
            print(f"[MCP] Got {len(filtered)} relevant docs")
            return filtered
        except Exception as e:
            print(f"[MCP] Error: {e}")
            return []

mcp = MCPClient()


def add_line_numbers(code: str) -> str:
    """Add line numbers to every line of code."""
    return "\n".join(
        f"Line {i+1}: {line}"
        for i, line in enumerate(code.splitlines())
    )


def diff_code(buggy: str, correct: str) -> tuple[str, list]:
    """
    Compare buggy vs correct code line by line.
    Returns (diff_string, list_of_differing_line_numbers)
    """
    b_lines = buggy.splitlines()
    c_lines = correct.splitlines()
    max_len = max(len(b_lines), len(c_lines))
    b_lines += [""] * (max_len - len(b_lines))
    c_lines += [""] * (max_len - len(c_lines))

    diff_out  = []
    diff_nums = []

    for i, (b, c) in enumerate(zip(b_lines, c_lines)):
        n = i + 1
        if b.strip() != c.strip():
            diff_out.append(f"   Line {n}: {b}   →     {c}")
            diff_nums.append(n)
        else:
            diff_out.append(f"   Line {n}: {b}")

    return "\n".join(diff_out), diff_nums


def sanitize_code(code: str) -> str:
    """Basic prompt injection guard — wrap code in strict delimiters."""
    # Remove suspicious instruction patterns in comments
    code = re.sub(r"(//|#).*?(ignore|forget|disregard|instruction).*", "", code, flags=re.IGNORECASE)
    return f"[CODE_START]\n{code}\n[CODE_END]"


def parse_agent_output(raw: str) -> dict:
    """
    Extract bug_line and explanation from agent final answer.
    Tries JSON parse first, then regex fallback.
    """
    # Try JSON block
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "bug_line" in data and "explanation" in data:
                return data
        except Exception:
            pass

    # Regex fallback
    line_match = re.search(r"bug.?line[\"']?\s*[:=]\s*(\d+)", raw, re.IGNORECASE)
    expl_match = re.search(r"explanation[\"']?\s*[:=]\s*[\"']?(.+?)(?:[\"']|$|\n)", raw, re.IGNORECASE)

    return {
        "bug_line":    int(line_match.group(1)) if line_match else -1,
        "explanation": expl_match.group(1).strip() if expl_match else raw[:300]
    }


def compute_confidence(
    parsed:   dict,
    diff_nums: list,
    mcp_docs:  list,
    buggy_code: str
) -> tuple[float, dict]:
    """
    Computes confidence score 0.0–1.0 based on 4 checks.
    Returns (score, checks_dict)
    """
    checks = {}

    # CHECK 1: Bug line detected (not -1)
    checks["bug_line_found"] = parsed.get("bug_line", -1) != -1

    # CHECK 2: Bug line matches diff between buggy and correct code
    checks["line_matches_diff"] = (
        parsed.get("bug_line") in diff_nums if diff_nums else False
    )

    # CHECK 3: MCP returned relevant docs with high score
    checks["mcp_grounded"] = bool(mcp_docs and mcp_docs[0].get("score", 0) > 0.65)

    # CHECK 4: Explanation mentions something from the buggy line
    if checks["bug_line_found"]:
        bug_line_idx  = parsed["bug_line"] - 1
        code_lines    = buggy_code.splitlines()
        if 0 <= bug_line_idx < len(code_lines):
            buggy_line_words = set(re.findall(r"\w+", code_lines[bug_line_idx]))
            expl_words       = set(re.findall(r"\w+", parsed.get("explanation", "")))
            overlap          = buggy_line_words & expl_words
            checks["explanation_relevant"] = len(overlap) >= 1
        else:
            checks["explanation_relevant"] = False
    else:
        checks["explanation_relevant"] = False

    score = sum(checks.values()) / len(checks)
    return round(score, 3), checks



@tool
def read_code(code_id: str) -> str:
    """
    Reads the buggy C++ code snippet from the dataset by ID.
    Returns line-numbered code so you can pinpoint exact bug line.
    Always call this FIRST for any code ID.

    Args:
        code_id: The ID of the code snippet (e.g. '0', '1')
    """
    print(f"\n[Tool1] read_code(id={code_id})")
    try:
        df  = load_dataset()
        row = df[df["ID"] == int(code_id)]
        if row.empty:
            return f"ERROR: No code found with ID={code_id}"
        row         = row.iloc[0]
        buggy_code  = str(row.get("Code", ""))
        context     = str(row.get("Context", "No context"))
        numbered    = add_line_numbers(buggy_code)
        safe_code   = sanitize_code(numbered)
        return (
            f"CODE SNIPPET (ID={code_id})\n"
            f"CONTEXT: {context}\n"
            f"TOTAL LINES: {len(buggy_code.splitlines())}\n\n"
            f"BUGGY CODE:\n{safe_code}"
        )
    except Exception as e:
        return f"ERROR in read_code: {e}"


@tool
def get_context(code_id: str) -> str:
    """
    Returns the context, correct code, and a line-by-line diff
    showing exactly which lines differ between buggy and correct code.
    Call this SECOND to understand the code purpose and validate bug line.

    Args:
        code_id: The ID of the code snippet (e.g. '0', '1')
    """
    print(f"\n[Tool2] get_context(id={code_id})")
    try:
        df  = load_dataset()
        row = df[df["ID"] == int(code_id)]
        if row.empty:
            return f"ERROR: No entry found with ID={code_id}"
        row          = row.iloc[0]
        context      = str(row.get("Context",      "No context"))
        correct_code = str(row.get("Correct Code", ""))
        explanation  = str(row.get("Explanation",  "No explanation"))
        buggy_code   = str(row.get("Code",         ""))
        diff_str, diff_nums = diff_code(buggy_code, correct_code)
        return (
            f"CONTEXT INFO (ID={code_id})\n"
            f"CONTEXT     : {context}\n"
            f"EXPLANATION : {explanation}\n\n"
            f"CORRECT CODE:\n{correct_code}\n\n"
            f"LINE-BY-LINE DIFF:\n{diff_str}\n\n"
            f"DIFF SUMMARY: Bug likely on line(s): {diff_nums}"
        )
    except Exception as e:
        return f"ERROR in get_context: {e}"


@tool
def lookup_bug_manual(query: str) -> str:
    """
    Searches the MCP Server's bug manual/documentation using
    semantic vector search. Use the code context as your query.
    Call this THIRD to ground your explanation in official documentation.

    Args:
        query: Search string describing the bug type (e.g. 'RDI method naming error')
    """
    print(f"\n[Tool3] lookup_bug_manual(query='{query}')")
    docs = mcp.search_documents(query)
    if not docs:
        return "No relevant documentation found in MCP server."
    results = []
    for i, doc in enumerate(docs[:5]):  # top 5 only
        results.append(
            f"[Doc {i+1}] Score={doc.get('score', 0):.3f}\n"
            f"{doc.get('text', '')[:400]}"
        )
    return "MCP DOCUMENTATION RESULTS:\n\n" + "\n\n---\n\n".join(results)


@tool
def validate_answer(code_id: str, bug_line: str, explanation: str) -> str:
    """
    Validates your detected bug_line and explanation before writing output.
    Checks: line exists, matches diff, explanation is relevant.
    Returns validation result + confidence score.
    Call this FOURTH before writing output.

    Args:
        code_id    : The ID of the code snippet
        bug_line   : Your detected bug line number (as string)
        explanation: Your generated explanation
    """
    print(f"\n[Tool4] validate_answer(id={code_id}, line={bug_line})")
    try:
        df  = load_dataset()
        row = df[df["ID"] == int(code_id)]
        if row.empty:
            return f"ERROR: No entry with ID={code_id}"

        row          = row.iloc[0]
        buggy_code   = str(row.get("Code", ""))
        correct_code = str(row.get("Correct Code", ""))
        total_lines  = len(buggy_code.splitlines())
        bug_line_int = int(bug_line)

        _, diff_nums = diff_code(buggy_code, correct_code)

        checks = {}

        # Check 1: line is within valid range
        checks["line_in_range"] = 1 <= bug_line_int <= total_lines

        # Check 2: line matches diff
        checks["line_matches_diff"] = bug_line_int in diff_nums if diff_nums else False

        # Check 3: explanation is not empty
        checks["explanation_not_empty"] = len(explanation.strip()) > 10

        # Check 4: explanation mentions something from the buggy line
        if checks["line_in_range"]:
            bline_words = set(re.findall(r"\w+", buggy_code.splitlines()[bug_line_int - 1]))
            expl_words  = set(re.findall(r"\w+", explanation))
            checks["explanation_relevant"] = len(bline_words & expl_words) >= 1
        else:
            checks["explanation_relevant"] = False

        confidence = round(sum(checks.values()) / len(checks), 3)
        passed     = confidence >= MIN_CONFIDENCE

        return (
            f"VALIDATION RESULT (ID={code_id})\n"
            f"Bug Line   : {bug_line_int} / {total_lines} total lines\n"
            f"Diff Lines : {diff_nums}\n"
            f"Checks     : {checks}\n"
            f"Confidence : {confidence}\n"
            f"Status     : {'  PASSED' if passed else '   FAILED — retry with better reasoning'}"
        )
    except Exception as e:
        return f"ERROR in validate_answer: {e}"


@tool
def write_output(code_id: str, bug_line: str, explanation: str) -> str:
    """
    Writes the final validated answer to output.csv.
    Only call this AFTER validate_answer passes.

    Args:
        code_id    : The ID of the code snippet
        bug_line   : Final detected bug line number
        explanation: Final explanation of the bug
    """
    print(f"\n[Tool5] write_output(id={code_id}, line={bug_line})")
    try:
        row = {
            "ID":          int(code_id),
            "Bug Line":    int(bug_line),
            "Explanation": explanation.strip()
        }
        file_exists = os.path.exists(OUTPUT_CSV)
        df_new      = pd.DataFrame([row])

        if file_exists:
            df_existing = pd.read_csv(OUTPUT_CSV)
            # avoid duplicate ID entries
            df_existing = df_existing[df_existing["ID"] != int(code_id)]
            df_final    = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final = df_final.sort_values("ID").reset_index(drop=True)
        df_final.to_csv(OUTPUT_CSV, index=False)

        print(f"[Tool5]   Written to {OUTPUT_CSV}")
        return f"  Successfully written: ID={code_id}, Bug Line={bug_line}"

    except Exception as e:
        return f"ERROR in write_output: {e}"



TOOLS = [read_code, get_context, lookup_bug_manual, validate_answer, write_output]



SYSTEM_PROMPT = """You are an expert C++ bug detection AI agent working for Infineon Technologies.
Your job is to analyze C++ code snippets, detect bugs with LINE-LEVEL PRECISION,
and provide clear explanations grounded in official documentation.

STRICT WORKFLOW — follow this order EVERY TIME:
1. Use read_code       → get the buggy code with line numbers
2. Use get_context     → understand purpose + see diff vs correct code
3. Use lookup_bug_manual → search MCP documentation for relevant bug patterns
4. Use validate_answer → verify your detected line and explanation
   - If validation FAILS → retry steps 1-3 with better reasoning
   - If validation PASSES → proceed to step 5
5. Use write_output    → write final answer to CSV

## Example Transformation
*Input:*

ID,BugLine,Explanation
4,3,The unit 'V' should be 'Volt'
4,4,The unit 'V' should be 'Volt'


*Output:*

ID,BugLine,Explanation
4,3,4,The unit 'V' should be 'Volt'

OUTPUT FORMAT RULES:
- bug_line must be an EXACT integer line number
- explanation must reference the MCP documentation if possible
- explanation must mention what specifically is wrong on that line
- Always respond in JSON at the end: {"bug_line": <int>, "explanation": "<string>"}"""


# ─────────────────────────────────────────────
#  AGENT BUILDER
# ─────────────────────────────────────────────
def build_agent():
    """Build and return the LangChain Agent."""

    print("[Agent] Initializing Groq LLM (free API)...")
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        api_key=GROQ_API_KEY,
        temperature=0.1,       # low temp for deterministic bug detection
        max_tokens=2048,
    )

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    print("[Agent]   Agent ready")
    return agent


# ─────────────────────────────────────────────
#  PROCESS SINGLE CODE ID  (with retry)
# ─────────────────────────────────────────────
def process_single(agent: AgentExecutor, code_id: int) -> dict:
    """
    Run the agent on a single code ID with retry mechanism.
    Returns audit record dict.
    """
    audit_record = {
        "id":               code_id,
        "attempts":         0,
        "confidence":       0.0,
        "validation_passed": False,
        "final_bug_line":   -1,
        "final_explanation": "",
        "mcp_used":         False,
        "error":            None
    }

    question = (
        f"Analyze code snippet with ID={code_id}. "
        f"Find the exact bug line and explain the bug clearly, "
        f"referencing the MCP documentation. "
        f"Write the result to output CSV when done."
    )

    for attempt in range(1, MAX_RETRIES + 1):
        audit_record["attempts"] = attempt
        print(f"\n{'='*60}")
        print(f"[Pipeline] Processing ID={code_id} | Attempt {attempt}/{MAX_RETRIES}")
        print(f"{'='*60}")

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            
            # Extract output from messages
            if isinstance(result, dict) and "messages" in result:
                messages = result.get("messages", [])
                raw_output = messages[-1].get("content", "") if messages else ""
            else:
                raw_output = result.get("output", str(result))

            print(f"\n[Pipeline] Agent final output:\n{raw_output}")

            # Parse structured answer
            parsed = parse_agent_output(raw_output)

            # Get diff for confidence
            df  = load_dataset()
            row = df[df["ID"] == code_id].iloc[0]
            buggy_code   = str(row.get("Code", ""))
            correct_code = str(row.get("Correct Code", ""))
            _, diff_nums = diff_code(buggy_code, correct_code)

            # MCP was used if lookup_bug_manual appears in output or steps
            mcp_used = "lookup_bug_manual" in raw_output
            audit_record["mcp_used"] = mcp_used

            # Compute confidence
            confidence, checks = compute_confidence(
                parsed, diff_nums, [], buggy_code
            )

            audit_record["confidence"]    = confidence
            audit_record["final_bug_line"]   = parsed.get("bug_line", -1)
            audit_record["final_explanation"] = parsed.get("explanation", "")

            print(f"[Pipeline] Confidence: {confidence} | Checks: {checks}")

            if confidence >= MIN_CONFIDENCE:
                audit_record["validation_passed"] = True
                print(f"[Pipeline]   ID={code_id} passed with confidence={confidence}")
                break
            else:
                print(f"[Pipeline]      Low confidence ({confidence}), retrying...")
                time.sleep(1)

        except Exception as e:
            print(f"[Pipeline]    Error on attempt {attempt}: {e}")
            audit_record["error"] = str(e)
            time.sleep(2)

    return audit_record


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline():
    """
    Main pipeline:
    1. Load dataset
    2. Build agent
    3. Process each code ID
    4. Save output.csv + audit.json
    """
    print("\n" + "="*60)
    print("  INFINEON BUG DETECTION — AGENTIC AI PIPELINE")
    print("="*60 + "\n")

    # Validate config
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set in .env file!")

    # Load dataset
    df      = load_dataset()
    all_ids = df["ID"].tolist()
    print(f"[Pipeline] Found {len(all_ids)} code snippets to process: {all_ids}")

    # Build agent
    agent = build_agent()

    # Clear output CSV if exists (fresh run)
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        print(f"[Pipeline] Cleared old {OUTPUT_CSV}")

    # Process all IDs
    audit_log = []

    for code_id in tqdm(all_ids, desc="Processing snippets"):
        record = process_single(agent, code_id)
        audit_log.append(record)

        # Small delay between IDs to avoid rate limits
        time.sleep(1)

    # Save audit log
    with open(AUDIT_JSON, "w") as f:
        json.dump(audit_log, f, indent=2)
    print(f"\n[Pipeline]   Audit log saved to {AUDIT_JSON}")

    # Print final results
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)

    if os.path.exists(OUTPUT_CSV):
        df_out = pd.read_csv(OUTPUT_CSV)
        print(df_out.to_string(index=False))
    else:
        print("     output.csv was not created — check errors above")

    # Summary stats
    passed  = sum(1 for r in audit_log if r["validation_passed"])
    total   = len(audit_log)
    avg_conf = round(sum(r["confidence"] for r in audit_log) / total, 3) if total else 0
    mcp_used = sum(1 for r in audit_log if r["mcp_used"])

    print(f"\n SUMMARY:")
    print(f"   Total processed : {total}")
    print(f"   Passed validation: {passed}/{total}")
    print(f"   Avg confidence  : {avg_conf}")
    print(f"   MCP used        : {mcp_used}/{total} snippets")
    print(f"\n  Done! Output saved to: {OUTPUT_CSV}")
    print(f"  Audit saved to       : {AUDIT_JSON}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_pipeline()