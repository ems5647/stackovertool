import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Load your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

class OpenAIClient:
    def __init__(self, embedding_model="text-embedding-ada-002", chat_model="gpt-3.5-turbo"):
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def embed_text(self, text: str) -> list:
        """
        Get embedding vector for a given text using OpenAI embedding model.
        """
        response = openai.Embedding.create(
            input=[text],
            model=self.embedding_model
        )
        embedding = response['data'][0]['embedding']
        return embedding

    def generate_answer(self, query: str, contexts: list[str]) -> str:
        """
        Generate an answer given a query and related context passages.
        """
        # Build the prompt
        context_text = "\n\n".join(contexts)

        prompt = f"""
You are a helpful assistant that answers StackOverflow-style programming questions.

Here are some related posts:
{context_text}

Based on the information above, provide a clear and helpful answer to the following user question:

User Question: {query}
Answer:
"""

        # Call OpenAI Chat API
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful programming assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        answer = response['choices'][0]['message']['content']
        return answer