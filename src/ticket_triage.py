import json
import re
import random
import time
import os

import pandas as pd
import mock_llm


ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

BASE_DELAY = 0.05
MAX_RETRIES = 3

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_ID_PATTERN = r"ORD-\d{6}"

def redact_pii(text):
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[PHONE]", text)
    text = re.sub(ORDER_ID_PATTERN, "[ORDER_ID]", text)

    return text

def validate_no_pii(text):
    patterns = [
        EMAIL_PATTERN,
        PHONE_PATTERN,
        ORDER_ID_PATTERN
    ]

    for pattern in patterns:
        assert re.search(pattern, text) is None, "PII detected!"

    return True

def check_input_guardrails(text):
    if not text.strip():
        return "skipped_empty"

    if mock_llm.est_tokens(text) > 400:
        return "flagged_long"

    return None

ZERO_SHOT_PROMPT = """
You are an Instamart customer support ticket classifier.

CLASSIFY the customer ticket into exactly one category:
billing, delivery, technical, fraud.

Return JSON only in this format:
{"category":"...","reason":"..."}

Do not include markdown or extra text.
"""

FEW_SHOT_PROMPT = """
You are an Instamart customer support ticket classifier.

CLASSIFY the customer ticket into exactly one category:
billing, delivery, technical, fraud.

EXAMPLES:
Ticket: I was charged twice for my order.
{"category":"billing","reason":"Duplicate charge"}

Ticket: My package was stolen and someone accessed my account.
{"category":"fraud","reason":"Unauthorized activity"}

Ticket: The app crashes when I open it.
{"category":"technical","reason":"App issue"}

Return JSON only in this format:
{"category":"...","reason":"..."}

Do not include markdown or extra text.
"""


def validate_model_response(response_text):
    try:
        result = json.loads(response_text)

        if result.get("category") in ALLOWED_CATEGORIES:
            return result["category"], "classified_json"

        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError, TypeError):
        words = re.findall(r"[a-z]+", (response_text or "").lower())
        categories = [
            word for word in words
            if word in ALLOWED_CATEGORIES
        ]

        if len(categories) == 1:
            return categories[0], "classified_keyword"

        return None, "flagged_invalid_output"


def call_with_retry(ticket_id, messages):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                raise

            retry_number = attempt
            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retry_number} wait={wait_time:.4f}s"
            )

            time.sleep(wait_time)



def classify_ticket(ticket_id, text, system_prompt,
                    tokens_so_far=0, token_budget=None):

    guardrail_status = check_input_guardrails(text)

    if guardrail_status:
        return None, guardrail_status, 0

    redacted_text = redact_pii(text)
    validate_no_pii(redacted_text)

    if token_budget is not None and tokens_so_far >= token_budget:
        return None, "budget_exhausted", 0

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]

    response = call_with_retry(ticket_id, messages)

    response_text = response.choices[0].message.content

    category, status = validate_model_response(response_text)

    tokens_used = response.usage.total_tokens

    return category, status, tokens_used


def run_triage(system_prompt, token_budget=None):

    mock_llm.reset()

    df = pd.read_csv(
        "data/raw/tickets.csv",
        keep_default_na=False
    )

    results = []
    total_tokens = 0

    for _, row in df.iterrows():

        category, status, tokens_used = classify_ticket(
            ticket_id=row["ticket_id"],
            text=row["text"],
            system_prompt=system_prompt,
            tokens_so_far=total_tokens,
            token_budget=token_budget
        )

        total_tokens += tokens_used

        results.append({
            "ticket_id": row["ticket_id"],
            "customer_id": row["customer_id"],
            "true_category": row["true_category"],
            "category": category,
            "status": status,
            "tokens_used": tokens_used
        })

    results_df = pd.DataFrame(results)

    return results_df, total_tokens


def route_ticket(status, category, is_premium):

    if status in {
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    }:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def create_final_triage():

    results_df, total_tokens = run_triage(
        FEW_SHOT_PROMPT,
        token_budget=100000
    )

    segments = pd.read_csv("outputs/customer_segments.csv")

    premium_map = segments.set_index("customer_id")["is_premium"].to_dict()
    segment_map = segments.set_index("customer_id")["segment_name"].to_dict()

    final_rows = []

    for _, row in results_df.iterrows():

        customer_id = row["customer_id"]

        is_premium = bool(premium_map.get(customer_id, False))
        segment_name = segment_map.get(customer_id, "")

        route, priority = route_ticket(
            row["status"],
            row["category"],
            is_premium
        )

        final_rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "status": row["status"],
            "category": row["category"],
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"]
        })

    final_df = pd.DataFrame(final_rows)

    os.makedirs("outputs", exist_ok=True)

    final_df.to_csv(
        "outputs/triage_results.csv",
        index=False
    )

    return final_df, total_tokens