"""
test_hallucinations.py

Runs 14 targeted hallucination probes against the live ChatEngine / llama3.2:3b.
Each test has:
  - a prompt
  - a list of REQUIRED substrings (case-insensitive) in the reply
  - a list of FORBIDDEN substrings that indicate hallucination
  - the tool names expected to be called (empty = no tool call expected)

Scoring: PASS / FAIL / WARN printed per test; summary at the end.
"""

import sys
import re
import time
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, ".")

from database.schema_manager import SchemaManager
from database.dynamic_crud import DynamicCRUD
from ai.query_parser import ChatEngine
from ai.dynamic_tools import ToolResult

# ── colour helpers ─────────────────────────────────────────────────────────────

RESET = "\033[0m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

def ok(s):  return f"{GREEN}{s}{RESET}"
def err(s): return f"{RED}{s}{RESET}"
def warn(s):return f"{YELLOW}{s}{RESET}"
def bold(s):return f"{BOLD}{s}{RESET}"


# ── test definition ────────────────────────────────────────────────────────────

@dataclass
class HallucinationTest:
    name:          str
    prompt:        str
    require:       List[str] = field(default_factory=list)   # any match = required present
    forbid:        List[str] = field(default_factory=list)   # any match = hallucination
    expect_tools:  List[str] = field(default_factory=list)   # tool names that MUST be called
    forbid_tools:  List[str] = field(default_factory=list)   # tool names that must NOT be called
    history:       List[dict] = field(default_factory=list)  # prior turns to simulate memory
    description:   str = ""


# ── fixture setup ──────────────────────────────────────────────────────────────

def build_db():
    """
    Create an isolated in-memory-ish DB with:
      books  (title, author, price:FLOAT, genre)
      3 rows: Dune $14.99, Foundation $12.50, Neuromancer $11.00
    """
    db_dir = tempfile.mkdtemp()
    sm     = SchemaManager(db_dir)
    crud   = DynamicCRUD(sm)

    user_id = sm.create_user("tester", "test1234")
    sm.create_dynamic_table(user_id, "books", [
        {"name": "title",  "type": "TEXT"},
        {"name": "author", "type": "TEXT"},
        {"name": "price",  "type": "FLOAT"},
        {"name": "genre",  "type": "TEXT"},
    ])
    sm.create_dynamic_table(user_id, "empty_table", [
        {"name": "value", "type": "TEXT"},
    ])
    crud.insert_record(user_id, "books",
        {"title": "Dune", "author": "Frank Herbert", "price": 14.99, "genre": "sci-fi"})
    crud.insert_record(user_id, "books",
        {"title": "Foundation", "author": "Isaac Asimov", "price": 12.50, "genre": "sci-fi"})
    crud.insert_record(user_id, "books",
        {"title": "Neuromancer", "author": "William Gibson", "price": 11.00, "genre": "cyberpunk"})

    return sm, crud, user_id


# ── test cases ─────────────────────────────────────────────────────────────────

def build_tests() -> List[HallucinationTest]:
    return [

        HallucinationTest(
            name="T01: empty table → no records",
            description="Query a table with zero rows. Model must NOT invent rows.",
            prompt="Show me everything in empty_table.",
            require=["no records", "no rows", "0 record", "empty", "nothing",
                     "does not contain", "not contain", "no data", "not found",
                     "contains no", "no results", "zero", "0 result"],
            forbid=["value=", "row 1", "here are the records"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T02: nonexistent table",
            description="Ask about a nonexistent table. Model must say it doesn't exist (any phrasing).",
            prompt="Show me all records in the employees table.",
            require=["doesn't exist", "does not exist", "no table", "no such table",
                     "employees", "create"],  # any one of these signals correct behavior
            forbid=["name=", "salary=", "department=",
                    "here are the records", "here are all the records"],
        ),

        HallucinationTest(
            name="T03: off-topic question",
            description="Completely off-topic. Model must refuse and mention only the real tables.",
            prompt="What is the capital of France?",
            require=["can only", "books", "empty_table"],
            forbid=["paris", "france", "capital"],
        ),

        HallucinationTest(
            name="T04: count with zero matches",
            description="Filter that matches nothing. Model must say zero/none and must NOT invent HP records.",
            prompt="How many Harry Potter books do I have?",
            require=["no records", "0", "don't have", "do not have", "none",
                     "not harry potter", "are not", "they are not",
                     "do not match", "not match", "no harry", "no titles",
                     "no books", "no results"],
            forbid=["1 harry potter", "2 harry potter", "3 harry potter",
                    "harry potter and", "sorcerer's stone", "philosopher's stone"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T05: specific field lookup",
            description="Ask for one field from one row. Model must use real tool data.",
            prompt="What is the price of Dune?",
            require=["14.99", "14"],
            forbid=["12.50", "11.00", "i don't know", "not sure"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T06: insert into nonexistent table",
            description="Ask to add to a table that doesn't exist. Must not claim Alice was added.",
            prompt="Add a new employee named Alice with salary 50000.",
            require=["doesn't exist", "does not exist", "no table", "no such table",
                     "employees", "create"],
            forbid=["alice has been added", "alice was inserted", "successfully added alice",
                    "employee alice has been", "new employee alice"],
        ),

        HallucinationTest(
            name="T07: update without knowing ID",
            description="Update by value match. Model must query first to get ID, not guess.",
            prompt="Change the price of Foundation to 15.99.",
            require=["15.99", "updated", "foundation"],
            forbid=["id=0", "id=9", "id=99"],
            expect_tools=["query_data", "update_data"],
        ),

        HallucinationTest(
            name="T08: update nonexistent record by ID",
            description="Explicitly give a wrong ID. Model must report the error.",
            prompt="Update the price of book id=999 to 5.00.",
            require=["999", "no record", "not found", "error", "failed",
                     "does not exist", "update was not applied"],
            forbid=["updated successfully", "record 999 updated", "price is now 5"],
            expect_tools=["update_data"],
        ),

        HallucinationTest(
            name="T09: failed insert bad value",
            description="Ask to insert into a nonexistent table. Model must say table doesn't exist.",
            prompt="Add a book titled 'Ghost' to the readers table.",
            require=["doesn't exist", "does not exist", "readers", "no table",
                     "no such table", "error", "create"],
            forbid=["ghost has been added", "inserted successfully",
                    "ghost is now in"],
        ),

        HallucinationTest(
            name="T10: range query (no exact filter)",
            description="Range filters can't be expressed as equality. Model must fetch all and filter.",
            prompt="Which books cost less than $13?",
            require=["foundation", "neuromancer", "12", "11"],
            forbid=["dune", "14.99"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T11: count total rows",
            description="Count all rows in a table. Must return correct number (3).",
            prompt="How many books do I have in total?",
            require=["3"],
            forbid=["0 books", "1 book", "2 books", "4 books", "5 books"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T12: multi-step with history context",
            description="Second turn relies on first-turn data. Model must not mix up tables.",
            prompt="What genre is it?",
            history=[
                {"role": "user",      "content": "Show me the record for Dune."},
                {"role": "assistant", "content": "Dune (id=1) is by Frank Herbert, priced at $14.99, genre: sci-fi."},
            ],
            require=["sci-fi", "sci"],
            forbid=["cyberpunk", "fantasy", "horror", "no records"],
        ),

        HallucinationTest(
            name="T13: author lookup",
            description="Ask who wrote a specific book. Answer must come from real data.",
            prompt="Who wrote Neuromancer?",
            require=["william gibson", "gibson"],
            forbid=["frank herbert", "isaac asimov", "i don't know", "not sure"],
            expect_tools=["query_data"],
        ),

        HallucinationTest(
            name="T14: aggregate — most expensive book",
            description="Find the highest-priced row. Must use real data, not fabricate.",
            prompt="Which book is the most expensive?",
            require=["dune", "14.99"],
            forbid=["foundation is the most", "neuromancer is the most",
                    "i don't know", "not sure"],
            expect_tools=["query_data"],
        ),
    ]


# ── runner ─────────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    return text.lower()


def run_test(
    test: HallucinationTest,
    model: str = "llama3.2:3b",
) -> dict:
    # Fresh DB per test — prevents write tests (T07, T09…) from corrupting later reads.
    sm, crud, user_id = build_db()
    engine = ChatEngine(sm, crud, model=model, rag=None)

    t0 = time.time()
    try:
        reply, tool_results = engine.chat(
            user_id=user_id,
            user_message=test.prompt,
            history=list(test.history),
        )
    except Exception as exc:
        return {
            "status": "ERROR",
            "reply":  f"EXCEPTION: {exc}",
            "tools":  [],
            "ms":     int((time.time() - t0) * 1000),
            "issues": [f"ChatEngine raised: {exc}"],
        }

    ms      = int((time.time() - t0) * 1000)
    low     = normalise(reply)
    tools_used = [tr.name for tr in tool_results]
    issues  = []

    # Required substrings — at least ONE must appear
    if test.require and not any(req.lower() in low for req in test.require):
        issues.append(
            f"MISSING — none of the required phrases found: "
            + ", ".join(f"'{r}'" for r in test.require)
        )

    # Forbidden substrings
    for fbd in test.forbid:
        if fbd.lower() in low:
            issues.append(f"HALLUCINATION detected: '{fbd}' found in reply")

    # Expected tools called
    for et in test.expect_tools:
        if et not in tools_used:
            issues.append(f"Expected tool '{et}' was NOT called")

    # Forbidden tools called
    for ft in test.forbid_tools:
        if ft in tools_used:
            issues.append(f"Forbidden tool '{ft}' WAS called (should not be)")

    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "reply":  reply,
        "tools":  tools_used,
        "ms":     ms,
        "issues": issues,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(bold("\n════════════════════════════════════════════════════"))
    print(bold("  HALLUCINATION TEST SUITE  —  llama3.2:3b  "))
    print(bold("════════════════════════════════════════════════════\n"))

    tests  = build_tests()

    results = []
    for test in tests:
        sys.stdout.write(f"  {test.name} … ")
        sys.stdout.flush()
        r = run_test(test, model="llama3.2:3b")
        results.append((test, r))

        status_str = ok("PASS") if r["status"] == "PASS" else err("FAIL")
        print(f"{status_str}  ({r['ms']} ms)  tools={r['tools']}")

        if r["status"] != "PASS":
            # show reply excerpt
            excerpt = textwrap.fill(r["reply"], width=72,
                                    initial_indent="    Reply: ",
                                    subsequent_indent="           ")
            print(f"{DIM}{excerpt}{RESET}")
            for issue in r["issues"]:
                print(f"    {warn('→')} {issue}")
        print()

    # ── summary ───────────────────────────────────────────────────────────────
    passed = sum(1 for _, r in results if r["status"] == "PASS")
    failed = sum(1 for _, r in results if r["status"] == "FAIL")
    errors = sum(1 for _, r in results if r["status"] == "ERROR")
    total  = len(results)

    print(bold("════════════════════════════════════════════════════"))
    print(f"  {ok(f'{passed} PASSED')}   {err(f'{failed} FAILED')}   {warn(f'{errors} ERROR')}   / {total} total")
    print(bold("════════════════════════════════════════════════════\n"))

    if failed or errors:
        print("Failed tests:")
        for test, r in results:
            if r["status"] != "PASS":
                print(f"  • {test.name}")
                for issue in r["issues"]:
                    print(f"      {issue}")
        print()

    return failed + errors


if __name__ == "__main__":
    sys.exit(main())
