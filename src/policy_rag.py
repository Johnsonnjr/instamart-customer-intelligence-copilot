import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_documents(overlap=10):
    chunks = []

    for filename in sorted(os.listdir("data/policies")):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join("data/policies", filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        words = text.split()
        step = 40 - overlap

        for start in range(0, len(words), step):
            chunk_words = words[start:start + 40]

            chunks.append({
                "source": filename,
                "text": " ".join(chunk_words)
            })

            if start + 40 >= len(words):
                break

    return chunks

def build_vectorizer(chunks):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix

def retrieve(query, chunks, vectorizer, matrix, k=3):
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, matrix)[0]

    top_indices = scores.argsort()[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })

    return results

import mock_llm
import json


def answer(question):
    chunks = chunk_documents()
    vectorizer, matrix = build_vectorizer(chunks)

    results = retrieve(question, chunks, vectorizer, matrix, k=3)

    context = "\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in results
    )

    system_message = (
        "You are a POLICY ASSISTANT. "
        "Answer only using the supplied context. "
        "If the information is not available in the context, "
        "say: I could not find the information."
    )

    user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"

    response = mock_llm.chat([
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ])

    return response.choices[0].message.content, results
def evaluate_policy():
    import pandas as pd

    evaluation = pd.read_csv("data/raw/policy_eval.csv")
    results = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = "" if pd.isna(row["expected_fact"]) else str(row["expected_fact"])

        response, sources = answer(question)

        source_names = [item["source"] for item in sources]

        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_correct_doc": (
                expected_doc in source_names if expected_doc != "NONE" else None
            ),
            "expected_fact_found": (
                expected_fact in response if expected_fact else None
            ),
            "fallback": response.startswith("I could not find")
        })

    return pd.DataFrame(results)

def evaluate_zero_overlap():
    import pandas as pd

    chunks = chunk_documents(overlap=0)
    vectorizer, matrix = build_vectorizer(chunks)

    evaluation = pd.read_csv("data/raw/policy_eval.csv")
    results = []

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = "" if pd.isna(row["expected_fact"]) else str(row["expected_fact"])

        sources = retrieve(question, chunks, vectorizer, matrix, k=3)
        source_names = [item["source"] for item in sources]

        context = "\n".join(
            f"[{item['source']}]\n{item['text']}" for item in sources
        )

        user_message = f"CONTEXT:\n{context}\nQUESTION:\n{question}"

        response = mock_llm.chat([
            {"role": "system", "content": "You are a POLICY ASSISTANT. Answer only from context. If unavailable, say I could not find the information."},
            {"role": "user", "content": user_message}
        ])

        answer_text = response.choices[0].message.content

        results.append({
            "question": question,
            "retrieved_correct_doc": expected_doc in source_names if expected_doc != "NONE" else None,
            "expected_fact_found": expected_fact in answer_text if expected_fact else None,
            "fallback": answer_text.startswith("I could not find")
        })

    return pd.DataFrame(results)
