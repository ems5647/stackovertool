from app.schemas import AskRequest, AskResponse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

from sentence_transformers import SentenceTransformer
from pymilvus import Collection

from app.milvus_client import MilvusClient
from app.stackoverflow_client import StackOverflowClient
from app.openai_client import OpenAIClient

app = FastAPI()

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Initialize clients (ensure appropriate config like embedding model etc.)
so_client = StackOverflowClient()
milvus_client = MilvusClient(collection_name="stackoverflow_qa", embedding_dim=1536)  # example dim
openai_client = OpenAIClient(...)  # assuming you have an OpenAI client for generating answers

@app.post("/ask")
def ask(query: str):
    # Step 1: Embed the user query for vector search
    query_vector = openai_client.embed_text(query)  # or however you obtain embeddings

    # Step 2: Search Milvus for similar questions/answers
    results = milvus_client.search_vectors(query_vector, top_k=5)
    # (Optional) If using a similarity score threshold, you could check `results[0]['score']` etc. here.

    # Step 3: If no relevant stored Q/A found, fetch new from StackOverflow
    if not results:  # or if results[0]['score'] is below a certain threshold of relevance
        # Search StackOverflow for relevant question(s)
        so_questions = so_client.search_questions(query)  # assume this returns a list of question dicts
        for q in so_questions:
            qid = q["question_id"]  # or however the search returns the ID
            # Avoid duplicates: skip if this question is already in Milvus
            if milvus_client.entry_exists(qid, is_question=True):
                # If question exists but maybe answers were not stored (old data), ensure answers are stored
                if not milvus_client.has_answers_for_question(qid):
                    answers = so_client.fetch_answers(qid)
                    for ans in answers:
                        # Clean/prepare answer text
                        ans_text = ans["body"]  # consider stripping HTML if not done already
                        ans_vector = openai_client.embed_text(ans_text)
                        milvus_client.insert_entry(
                            source_id=ans["answer_id"],
                            parent_id=qid,
                            is_question=False,
                            title="",               # no title for answers
                            body=ans_text,
                            embedding=ans_vector
                        )
                continue  # skip inserting the question again
            # **New question found**: fetch its answers and insert everything
            answers = so_client.fetch_answers(qid)
            # Prepare and insert the question
            q_title = q.get("title", "")
            q_body = q.get("body", "")
            # Compute embedding using title + body for best results
            q_text_for_embedding = f"{q_title}\n{q_body}"
            q_vector = openai_client.embed_text(q_text_for_embedding)
            milvus_client.insert_entry(
                source_id=qid,
                parent_id=0,
                is_question=True,
                title=q_title,
                body=q_body,
                embedding=q_vector
            )
            # Insert all answers for this question
            for ans in answers:
                ans_text = ans["body"]
                ans_vector = openai_client.embed_text(ans_text)
                milvus_client.insert_entry(
                    source_id=ans["answer_id"],
                    parent_id=qid,
                    is_question=False,
                    title="",  # answer has no title
                    body=ans_text,
                    embedding=ans_vector
                )
        # After inserting new data, search again to get combined results
        results = milvus_client.search_vectors(query_vector, top_k=5)

    else:
        # Step 4: If we found a question but its answers are not in DB (legacy case), fetch and store them.
        # (This ensures we have answers even if the question was stored by older version of the system)
        for res in results:
            if res["is_question"]:
                qid = res["source_id"]
                if not milvus_client.has_answers_for_question(qid):
                    answers = so_client.fetch_answers(qid)
                    for ans in answers:
                        ans_text = ans["body"]
                        ans_vector = openai_client.embed_text(ans_text)
                        milvus_client.insert_entry(
                            source_id=ans["answer_id"],
                            parent_id=qid,
                            is_question=False,
                            title="",
                            body=ans_text,
                            embedding=ans_vector
                        )
        # Now that answers are stored, you could re-run the search to include newly added answers:
        results = milvus_client.search_vectors(query_vector, top_k=5)

    # Step 5: Use the top results as context to generate an answer with OpenAI
    # Extract the content of top results for the prompt
    top_contexts = []
    for hit in results[:3]:  # use top 3 matches as context, for example
        if hit["is_question"]:
            # Combine title and body for context if it's a question
            context_text = f"Question: {hit['title']}\n{hit['body']}"
        else:
            # For answers, you might include the parent question title for clarity
            context_text = f"Answer (to question {hit['parent_id']}): {hit['body']}"
        top_contexts.append(context_text)
    # Generate answer (this could be a prompt engineering step; using a placeholder function here)
    final_answer = openai_client.generate_answer(query, top_contexts)

    # Step 6: Prepare sources for the response
    sources = []
    for hit in results[:3]:
        if hit["is_question"]:
            sources.append({
                "question_id": hit["source_id"],
                "question_title": hit["title"],
                "question_body": hit["body"],
                "url": f"https://stackoverflow.com/questions/{hit['source_id']}"
            })
        else:
            sources.append({
                "answer_id": hit["source_id"],
                "parent_question_id": hit["parent_id"],
                "answer_body": hit["body"],
                "url": f"https://stackoverflow.com/a/{hit['source_id']}"
            })

    # Return the answer and sources as a JSON response
    return {
        "answer": final_answer,
        "sources": sources
    }