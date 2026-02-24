import requests
from django.conf import settings
import openai
import numpy as np
openai.api_key = settings.OPENAI_API_KEY

def get_embedding(text):

    if settings.AI_MODE == "local":
        """
        Call Ollama local API to get embeddings
        """
        url = settings.OLLAMA_URL
        model = settings.OLLAMA_MODEL

        payload = {
            "model": model,
            "input": text
        }

        response = requests.post(url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.text}")

        data = response.json()
        # Ollama returns embeddings in 'embedding' key (check API docs)
        return data.get("embedding")
    else:

        """
        Generate OpenAI embedding for a given text.
        Returns a float32 numpy array.
        """
        if not text:
            return []

        try:
            response = openai.Embedding.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=text
            )
            embedding = response["data"][0]["embedding"]
            return np.array(embedding, dtype="float32")

        except Exception as e:
            print("OpenAI embedding error:", e)
            return []
