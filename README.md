# Instamart Customer Intelligence & Support Copilot


## Overview

Instamart Customer Intelligence & Support Copilot is a Python-based data analytics and AI support project built using synthetic customer, product, order, and ticket data.

The project combines four components:

- **Part A – Data Foundation:** Data cleaning, SQLite database creation, SQL analysis, and exploratory data analysis.
- **Part B – Customer Intelligence:** RFM-based customer segmentation and machine learning-based lapse prediction.
- **Part C – Support Ticket Triage:** Guardrailed ticket classification, PII redaction, retry handling, and customer-aware routing.
- **Part D – Policy RAG and Support Agent:** TF-IDF policy retrieval and a tool-using refund support agent with safety limits and crash recovery.

All data is synthetic. The project uses the supplied `mock_llm.py` and does not require paid LLM APIs.

## Reproduce
Run all commands from the project root directory.

1. Create a virtual environment:

   `python3 -m venv .venv`

2. Activate it:

   `source .venv/bin/activate`

3. Install dependencies:

   `pip install -r requirements.txt`

4. Generate synthetic data:

   `python3 generate_data.py`

5. Run Part A – Data Pipeline:

   `python3 -c "from src.data_pipeline import audit, clean_data, validate_data; audit(); clean_data(); validate_data()"`

6. Run Part B – Customer Intelligence:

   `python3 -c "from src.customer_models import create_rfm, create_lapse_dataset; create_rfm(); create_lapse_dataset()"`

7. Run Part C – Ticket Triage:

   `python3 -c "from src.ticket_triage import create_final_triage; create_final_triage()"`

8. Run Part D – Policy RAG Evaluation:

   `python3 -c "from src.policy_rag import evaluate_policy, evaluate_zero_overlap; print(evaluate_policy()); print(evaluate_zero_overlap())"`

9. Run Part D – Seven Agent Scenarios:

   `python3 run_scenarios.py`

## Part A – Data Foundation
### EDA Findings

The Instamart dataset contains customer, product, and order information. The analysis focuses on completed orders across six cities and five product categories.

The dataset contains 11,110 completed orders with total revenue of ₹29,98,668.80. The average order value is ₹269.91, while the median is ₹135. The positive skewness of 4.61 indicates a right-skewed distribution, where a small number of high-value orders increase the average.

Using the IQR method, 897 orders are identified as upper outliers, with values above ₹645.30. No lower outliers are detected. These high-value orders should be investigated rather than automatically removed.

Electronics contributes the highest category revenue at ₹12,19,248.10, while Personal Care contributes the lowest at ₹1,68,433.95. Hyderabad has the highest city revenue at ₹5,28,589.90.

The strongest observed correlation is between order amount and discount percentage (-0.095), but this relationship is weak. Correlation does not imply causation.

### Data Cleaning Rules

**Customer Data**
- Standardized city names by removing extra spaces and applying title case.
- Standardized gender values into male, female, and unknown.
- Converted age values to numeric format.
- Treated ages outside 10–100 as missing.
- Filled missing ages using the median age of the corresponding city.
- Converted signup dates into datetime format.

**Order Data**
- Converted order timestamps into datetime format.
- Removed 119 exact duplicate rows.
- Kept the latest timestamp record for each order ID.
- Converted order amounts into numeric format.
- Calculated missing order amounts using product price, quantity, and discount percentage.
- Replaced negative delivery times with missing values.
- Filled missing delivery times for completed orders using category-wise median values.
- Kept missing delivery times for cancelled orders.

**Cleaning Results**
- Raw order rows: 13,084
- Rows after exact duplicate removal: 12,965
- Final unique orders: 12,309
- Completed orders: 11,110
- Cancelled orders: 1,199
- Pending orders: 0
- Missing customer ages after cleaning: 0
- Missing order amounts after cleaning: 0
- Missing delivery times for completed orders: 0
- Missing delivery times for cancelled orders: 1,199

### SQL Query Results

| Query | Analysis | Result |
|---|---|---|
| Q1 | Customers with completed order spending above ₹8,000 | 30 customers |
| Q2 | Revenue by city and product category | 30 city-category combinations |
| Q3 | Products priced above average catalogue price | 11 products |
| Q4 | Customers with no completed orders | 25 customers |
| Q5 | Customers spending above average | 550 customers |
| Q6 | Customer spending rankings within each city | RANK, DENSE_RANK and ROW_NUMBER applied |
| Q7 | Daily revenue and cumulative running total | 183 days |

**Additional SQL Findings**
- Total completed order revenue: ₹29,98,668.80
- Average completed customer spending: ₹2,552.06
- Highest city-category revenue: Bengaluru – Electronics, ₹2,33,909.45.
- Highest spending customer in each city was identified using window functions.
- Q7 cumulative revenue matches total completed order revenue.

## Part B – Customer Segments and Lapse Prediction

### RFM Analysis

RFM analysis was performed using completed orders as of 2025-10-01.

- Total customers: 1,175
- Median recency: 19 days
- Minimum recency: 0 days
- Maximum recency: 180 days
- Average frequency: 9.46 orders
- Maximum frequency: 33 orders
- Average monetary value: ₹2,552.06
- Maximum monetary value: ₹14,766.55

### KMeans Clustering

RFM features were standardized using StandardScaler. KMeans models were evaluated for K values from 2 to 8 using inertia and silhouette score.

| K | Inertia | Silhouette Score |
|---|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected based on its strong silhouette score, elbow pattern, and business interpretability. Lower inertia alone does not guarantee better clustering because inertia generally decreases as the number of clusters increases.

### Customer Segments

| Segment | Customers | Mean Recency | Mean Frequency | Mean Monetary | Business Action |
|---|---:|---:|---:|---:|---|
| Occasional Customers | 436 | 28.86 | 3.77 | ₹1,057.62 | Encourage repeat purchases |
| Premium Customers | 138 | 24.12 | 6.77 | ₹7,129.70 | Provide VIP rewards |
| Loyal Customers | 419 | 8.77 | 19.40 | ₹3,410.02 | Offer exclusive benefits |
| At-Risk Customers | 182 | 109.04 | 2.20 | ₹685.99 | Run win-back campaigns |

PCA explained variance:
- Component 1: 0.6343
- Component 2: 0.2180
- Total explained variance: 0.8523

The output file `outputs/customer_segments.csv` contains customer-level RFM values, segment IDs, segment names, and premium flags.

### Lapse Prediction

The lapse prediction cohort contains 1,156 customers.

- Lapsed customers: 443 (38.32%)
- Not-lapsed customers: 713
- Training rows: 924
- Testing rows: 232

The lapse label identifies customers with no completed orders during September 2025.

Features were created using orders before 2025-09-01. Segment labels were not used as model features.

### Preprocessing and Model Tuning

- Numeric features were standardized using StandardScaler.
- Nominal features were one-hot encoded using OneHotEncoder(drop="first").
- Preprocessing was fitted on training data only.
- GridSearchCV used 5-fold cross-validation and F1 scoring.

Logistic Regression:
- Best C: 0.1
- Best CV F1: 0.6392

Random Forest:
- Best max_depth: 8
- Best min_samples_leaf: 20
- Best CV F1: 0.6740

### Test Set Evaluation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.6545 | 0.6837 |
| Recall | 0.8090 | 0.7528 |
| F1 Score | 0.7236 | 0.7166 |

Logistic Regression was selected because it achieved a slightly higher test F1 score and higher recall. Missing a customer who is about to lapse can mean lost revenue, while incorrectly targeting a loyal customer can waste retention campaign resources.

### Class Balance Experiment

The training data was oversampled by duplicating lapsed customers with replacement. The test data remained unchanged.

| Metric | Original Model | Oversampled Model |
|---|---:|---:|
| Precision | 0.6545 | 0.5915 |
| Recall | 0.8090 | 0.9438 |
| F1 Score | 0.7236 | 0.7273 |

Oversampling improved recall but reduced precision.

### Feature Interpretation

Top Random Forest features:
1. Frequency
2. Monetary
3. Recency
4. Cancellation rate
5. Favourite category: Electronics

Top Logistic Regression coefficients:
- Frequency: -1.2811
- Favourite category Electronics: -0.3782
- Favourite category Snacks: -0.2026

Negative coefficients indicate a lower predicted lapse probability relative to the relevant reference values.

### Model Persistence and Retention List

The selected Logistic Regression model was saved to `outputs/lapse_model.joblib`. Reloading the saved model produced identical predictions.

The top 10 premium test customers were ranked by predicted lapse probability and saved to `outputs/retention_list.csv`.

- Top 10 customers selected.
- Actually lapsed customers in the list: 2.

**Limitation:** The premium flag comes from the full-window RFM segmentation, which includes the September label period. Therefore, the retention list is illustrative and is not a leakage-free model evaluation.

## Part C – Support Ticket Triage with Guardrails

### 1. Dataset Overview

The support ticket dataset contains 164 synthetic tickets with five columns:

- `ticket_id`
- `customer_id`
- `order_id`
- `text`
- `true_category`

Of the 164 tickets:
- 4 contained empty or whitespace-only text.
- 4 exceeded the 400-token input limit.
- 156 tickets were eligible for model classification.

### 2. Input Guardrails and PII Redaction

Empty tickets were assigned `skipped_empty`. Tickets exceeding 400 estimated tokens were assigned `flagged_long`. Neither group was sent to the model.

PII was redacted before every model call using these patterns, in order:

| PII type | Replacement | Redactions |
|---|---|---:|
| Email | `[EMAIL]` | 55 |
| Phone number | `[PHONE]` | 91 |
| Order ID | `[ORDER_ID]` | 132 |
| **Total** | | **278** |

A validation check confirmed zero remaining email addresses, phone numbers, or order IDs in redacted ticket text.

The model's user message contains only the redacted ticket text.

### 3. Zero-shot and Few-shot Classification

Both system prompts request one of four categories: billing, delivery, technical, or fraud. Both require JSON-only output.

- Zero-shot uses classification instructions without examples.
- Few-shot includes labelled examples, including a fraud example.

Results across 156 eligible tickets:

| Status | Zero-shot | Few-shot |
|---|---:|---:|
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |
| **Total classified** | **147** | **147** |

Accuracy was calculated only among classified tickets:

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Correct predictions | 126 | 147 |
| Accuracy | 85.71% | 100.00% |
| Total tokens | 11,428 | 18,136 |
| Tokens per classified ticket | 77.74 | 123.37 |

The zero-shot confusion matrix:

| Actual category | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

Few-shot improved accuracy from 85.71% to 100%, using 6,708 additional tokens (approximately 58.7% more). In this synthetic experiment, the accuracy improvement justifies the additional token usage. Real-world performance and cost would require separate evaluation.

### 4. Retry and Response Validation

Every model call uses a maximum of three retries after the initial attempt. Exponential backoff with jitter uses a base delay of 0.05 seconds.

The full runs produced 19 transient failures, and all 19 were recovered on the first retry. No permanent model-call failures occurred.

Responses were handled as follows:

- Valid JSON with an allowed category: `classified_json`.
- Non-JSON text containing exactly one allowed category word: `classified_keyword`.
- Invalid JSON or a disallowed category: `flagged_invalid_output`.

Invalid responses were not guessed or automatically classified.

### 5. Token Budget Demonstration

A zero-shot run used a configured token budget of 6,000.

| Metric | Result |
|---|---:|
| Configured budget | 6,000 |
| Actual tokens used | 6,021 |
| Classified JSON | 72 |
| Classified keyword | 6 |
| Invalid output | 4 |
| Empty skipped | 4 |
| Long flagged | 4 |
| Budget exhausted | 74 |

The budget is checked before each model call. The final successful call can take usage slightly above the configured limit; subsequent calls are stopped.

The demonstration is saved as `outputs/triage_budget_demo.csv`.

### 6. Customer-aware Routing

The final run uses the Few-shot prompt with a token budget of 100,000 and joins customer segments from Part B.

Routing rules:

| Condition | Route | Priority |
|---|---|---|
| Empty, long, invalid, or budget-exhausted ticket | `human_review` | NORMAL |
| Fraud category | `security_team` | HIGH |
| Premium customer | `premium_desk` | HIGH |
| Other classified tickets | `support_queue` | NORMAL |

Final routing results:

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

Priority results:

| Priority | Tickets |
|---|---:|
| HIGH | 45 |
| NORMAL | 119 |

The final output is `outputs/triage_results.csv`, containing exactly 164 rows and these columns:

`ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`

Final validation confirmed correct routing for human-review and fraud tickets, no duplicate ticket IDs, and 17 missing categories corresponding to the 17 tickets requiring human review.



## Part D – Policy RAG and Support Agent

### 1. Policy Document Chunking

Five policy documents are stored in `data/policies/`. Each document is split into chunks of 40 words with an overlap of 10 words. The sliding window moves 30 words at a time, preserving context between consecutive chunks.

The chunking process records the source filename for every chunk.

**Chunking results:**

| Policy document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

The overlap helps preserve information that might otherwise be split across chunk boundaries.


### 2. TF-IDF Retrieval

The policy retrieval system uses `TfidfVectorizer()` to convert policy chunks into numerical vectors.

TF-IDF (Term Frequency–Inverse Document Frequency) gives higher importance to terms that are relevant to a document and less common across the collection.

For each user query:
1. Transform the query using the fitted TF-IDF vectorizer.
2. Calculate cosine similarity between the query vector and all policy chunk vectors.
3. Rank chunks by similarity score.
4. Return the top 3 chunks with their source filename, similarity score, and text.

**Retrieval test:**

Query: `refund policy late delivery`

The top 3 retrieved chunks all came from `refund_policy.txt`.

**Limitation:** TF-IDF relies on word overlap. It may miss semantically related information when the query uses different words from the policy document.

### 3. Policy Assistant Evaluation

The policy assistant retrieves the top 3 relevant chunks and passes them to `mock_llm.chat()` as context.

The system prompt instructs the assistant to:
- Answer using only the supplied policy context.
- Avoid inventing policy information.
- Return a fallback response when the answer is not available in the retrieved context.

The assistant was evaluated using 14 questions from `data/raw/policy_eval.csv`.

**Evaluation with 10-word overlap:**

| Metric | Result |
|---|---:|
| Total questions | 14 |
| Answerable questions | 12 |
| Correct expected documents in top 3 | 12/12 |
| Expected facts found | 12/12 |
| Unanswerable questions correctly handled with fallback | 2/2 |

**Zero-overlap experiment:**

The same evaluation was repeated with zero chunk overlap.

| Metric | Result |
|---|---:|
| Correct expected documents in top 3 | 12/12 |
| Expected facts found | 10/12 |
| Correct fallback responses | 2/2 |

Two questions failed the expected-fact check in the zero-overlap experiment:
1. What delivery time makes a completed order eligible for a full refund?
2. How soon is money returned when a payment fails but is deducted?

This experiment demonstrates that chunk overlap helps preserve context across chunk boundaries.


### 4. Agent Tools and Refund Rules

The support agent uses five tools through a tool registry (`TOOL_MAP`).

| Tool | Description |
|---|---|
| `get_order` | Retrieves order details from SQLite using the order ID. |
| `search_policy` | Retrieves the top 3 relevant policy chunks. |
| `calculate_refund` | Calculates the eligible refund amount. |
| `issue_refund` | Records an approved refund in the refund ledger. |
| `escalate_to_human` | Records refunds requiring human approval. |

**Refund policy rules:**

1. Only completed orders are eligible for refunds.
2. Delivery taking more than 45 minutes qualifies for a 100% refund.
3. Delivery within 45 minutes receives no late-delivery refund.
4. Refunds above ₹1,000 require human approval.
5. Refund requests are subject to the 7-day request window.

Refund and escalation actions are recorded in `outputs/refund_ledger.jsonl`, including the order ID, refund amount, timestamp, and tool name.

### 5. ReAct Agent Execution

The support agent follows the ReAct (Reasoning and Acting) pattern.

The execution flow is:

1. Receive the customer message.
2. Send the conversation and available tool specifications to `mock_llm.chat()`.
3. Inspect the model response for tool calls.
4. Execute the requested tool using `TOOL_MAP`.
5. Append the tool observation to the conversation.
6. Repeat until the model returns a final answer or a safety limit is reached.

The agent records each tool name, arguments, and observation.

All seven predefined scenarios were executed. Results are saved in `outputs/agent_runs.jsonl`, including scenario ID, status, final answer, LLM calls, token usage, and tool execution steps.

### 6. Safety and Guardrails

The agent implements three execution safety limits.

**1. Maximum iterations**

The agent allows a maximum of 6 LLM calls. If no final answer is produced within this limit, it returns `FALLBACK_MAX_ITERATIONS`.

**2. Token limit**

The agent tracks cumulative token usage using `response.usage.total_tokens`. If usage exceeds the configured limit of 6,000 tokens, it stops without executing the pending tool call and returns `FALLBACK_TOKEN_LIMIT`.

**3. Time limit**

The agent uses `time.monotonic()` to track execution time. If execution reaches the configured 10-second limit, it returns `FALLBACK_TIMEOUT`.

**Safety test results:**

| Test | Configuration | Result |
|---|---|---|
| Maximum iterations | 6 calls | FALLBACK_MAX_ITERATIONS |
| Token limit | 200 tokens | FALLBACK_TOKEN_LIMIT |
| Timeout | 0 seconds | FALLBACK_TIMEOUT |

All three safety conditions were tested successfully.

### 7. Crash Recovery and Resume

The agent saves a checkpoint in the `state/` directory after every tool observation.

Each checkpoint contains:
- Run ID and scenario ID.
- Complete conversation messages.
- Cumulative token usage.
- Number of LLM calls.
- Completed step number.
- Executed tools, arguments, and observations.

The `resume_agent(run_id)` function loads the checkpoint and continues execution using the saved conversation state.

**Crash-resume demonstration – Scenario S1:**

A simulated crash was triggered immediately after the second tool execution.

| Stage | Tool |
|---|---|
| Before crash | `get_order` |
| Before crash | `search_policy` |
| After resume | `calculate_refund` |
| After resume | `issue_refund` |

**Final results:**

| Metric | Result |
|---|---|
| Scenario | S1 |
| Final status | REFUND |
| Refund amount | ₹128.25 |
| Total LLM calls | 5 |
| Total tokens | 912 |
| Repeated tool executions | 0 |

The resumed agent continued from the saved checkpoint without repeating the first two tools and successfully completed the refund.
### 8. Scenario Results
All seven predefined scenarios were executed successfully. The actual results are recorded in `outputs/agent_runs.jsonl`.

| Scenario | Expected outcome | Actual status | LLM calls | Tokens |
|---|---|---|---:|---:|
| S1 | REFUND ₹128.25 | REFUND | 5 | 912 |
| S2 | REFUND ₹135.00 | REFUND | 5 | 912 |
| S3 | ESCALATE ₹1,999.00 | ESCALATE | 5 | 899 |
| S4 | DENIED | DENIED | 4 | 617 |
| S5 | DENIED | DENIED | 3 | 346 |
| S6 | NOT_FOUND | NOT_FOUND | 2 | 78 |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 6 | 3,018 |

All seven actual statuses match their expected outcomes.

### 9. ReAct vs Plan-Execute

**ReAct** alternates between model decisions and tool execution. It observes each tool result before deciding the next action. In this project, the agent uses 5 LLM calls for the normal S1 refund flow, with 912 total tokens.

**Plan-Execute** first creates a complete plan and then executes the planned steps. This can reduce repeated planning calls for predictable workflows, but it may be less flexible when tool observations differ from expectations.

For example, if `get_order` returns `NOT_FOUND` or an order is cancelled, ReAct can use that observation to stop or change its next action. A fixed plan may attempt unnecessary later steps unless additional validation is implemented.

ReAct is preferred for this support copilot because order status, delivery time, refund eligibility, and human approval requirements determine the next action dynamically. Checkpointing also allows the workflow to resume after a crash.



## Limitations and Next Steps

### Limitations

- The project uses synthetic data and may not represent real customer behaviour.
- The supplied `mock_llm.py` simulates LLM responses; results may differ from a production LLM.
- TF-IDF retrieval depends on keyword overlap and may miss semantically related policy information.
- The lapse prediction model is trained on a fixed historical dataset and requires retraining when new customer data becomes available.
- Refund actions are simulated and do not connect to a real payment gateway.
- The agent uses fixed execution limits and simplified business rules.

### Next Steps

- Evaluate the system using real, anonymized customer data.
- Explore semantic embeddings to improve policy retrieval.
- Monitor model performance and customer lapse predictions over time.
- Add production authentication, authorization, and audit controls.
- Integrate approved refund workflows with secure payment systems.
- Add human feedback to improve ticket classification and support responses.
