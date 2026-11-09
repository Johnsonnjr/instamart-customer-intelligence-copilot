import sqlite3
import json
import os
import mock_llm

def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
    """, (order_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


def search_policy(query):
    from src.policy_rag import (
        chunk_documents,
        build_vectorizer,
        retrieve
    )

    chunks = chunk_documents()
    vectorizer, matrix = build_vectorizer(chunks)

    results = retrieve(
        query, chunks, vectorizer, matrix, k=3
    )

    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {"refund_inr": refund}
def write_refund_record(order_id, refund_inr, tool_name):
    os.makedirs("outputs", exist_ok=True)

    record = {
        "order_id": order_id,
        "refund_inr": refund_inr,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "tool_name": tool_name
    }

    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {"refund_inr": refund_inr}

def issue_refund(order_id, refund_inr):
    return write_refund_record(
        order_id, refund_inr, "issue_refund"
    )


def escalate_to_human(order_id, refund_inr):
    return write_refund_record(
        order_id, refund_inr, "escalate_to_human"
    )


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Get order details using order ID.",
        "parameters": {
            "order_id": {"type": "string"}
        }
    },
    {
        "name": "search_policy",
        "description": "Search relevant refund policy documents.",
        "parameters": {
            "query": {"type": "string"}
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund based on order amount and delivery time.",
        "parameters": {
            "order_amount": {"type": "number"},
            "delivery_minutes": {"type": "number"}
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue an eligible refund.",
        "parameters": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Send a large refund request to a human lead.",
        "parameters": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
import time
import uuid



def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None
):
    start_time = time.monotonic()

    run_id = str(uuid.uuid4())

    messages = [
        {
            "role": "system",
            "content": "You are an AGENT. Help customers with refund requests using available tools."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    steps = []
    llm_calls = 0
    tokens_used = 0
    final_answer = None
    status = None

    while llm_calls < max_iterations:

        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = "FALLBACK_TIMEOUT: Request sent to human support."
            break

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1

        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = "FALLBACK_TOKEN_LIMIT: Request sent to human support."
            break


        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            final_answer = message.content
            status = final_answer.split(":")[0]
            break
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments

            observation = TOOL_MAP[tool_name](**arguments)

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            os.makedirs("state", exist_ok=True)

            checkpoint = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "messages": messages,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "step_number": len(steps),
                "steps": steps
            }

            with open(f"state/{run_id}.json", "w") as file:
                json.dump(checkpoint, file, indent=2)


        if crash_after_tools is not None and len(steps) >= crash_after_tools:
            raise RuntimeError(f"SIMULATED_CRASH:{run_id}")

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "FALLBACK_MAX_ITERATIONS: Request sent to human support."

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }

def resume_agent(run_id):
    checkpoint_path = f"state/{run_id}.json"

    with open(checkpoint_path, "r") as file:
        checkpoint = json.load(file)

    print("Checkpoint loaded successfully!")
    print("Scenario:", checkpoint["scenario_id"])
    print("Completed tools:", checkpoint["step_number"])
    messages = checkpoint["messages"]
    steps = checkpoint["steps"]
    tokens_used = checkpoint["tokens_used"]
    llm_calls = checkpoint["llm_calls"]
    start_time = time.monotonic()

    while llm_calls < 6:

        if time.monotonic() - start_time >= 10:
            return {
                "status": "FALLBACK_TIMEOUT",
                "final_answer": "FALLBACK_TIMEOUT: Request sent to human support."
            }

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        if tokens_used > 6000:
            return {
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": "FALLBACK_TOKEN_LIMIT: Request sent to human support."
            }

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            final_answer = message.content
            status = final_answer.split(":")[0]
            break
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })

        for call in tool_calls:
            tool_name = call.name
            arguments = call.arguments

            observation = TOOL_MAP[tool_name](**arguments)

            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })

            checkpoint = {
                "run_id": run_id,
                "scenario_id": checkpoint["scenario_id"],
                "messages": messages,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "step_number": len(steps),
                "steps": steps
            }

            with open(checkpoint_path, "w") as file:
                json.dump(checkpoint, file, indent=2)

    else:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = "FALLBACK_MAX_ITERATIONS: Request sent to human support."

    return {
        "run_id": run_id,
        "scenario_id": checkpoint["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used
    }
