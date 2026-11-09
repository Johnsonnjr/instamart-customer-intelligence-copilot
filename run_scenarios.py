import pandas as pd
import json
import os

from src.refund_agent import run_agent

# Load all 7 scenarios
df = pd.read_csv("data/raw/agent_scenarios.csv")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Clear previous test results
open("outputs/agent_runs.jsonl", "w").close()
open("outputs/refund_ledger.jsonl", "w").close()

# Run each scenario
with open("outputs/agent_runs.jsonl", "a") as file:

    for _, row in df.iterrows():

        result = run_agent(
            row["message"],
            row["scenario_id"]
        )

        record = {
            "scenario_id": result["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        file.write(json.dumps(record) + "\n")

        print(
            row["scenario_id"],
            result["status"],
            result["final_answer"]
        )

print("\nAll 7 scenarios completed!")
