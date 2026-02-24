import requests
from django.conf import settings
import openai

openai.api_key = settings.OPENAI_API_KEY

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
# Answer only from the HR policies below.
# Answer questions based ONLY on the provided documents
def generate_answer(question, context, system_prompt=None):
    if settings.AI_MODE == "local":
        prompt = f"""
You are Liberty HR Assistant.
Answer questions based ONLY on the provided documents
If the answer is not in the documents, respond, say you do not know.
            
Policies:
{context}

Question:
{question}
    """

        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
    else:
        """
        Generate answer from OpenAI Chat model using system prompt and context.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
            response = openai.ChatCompletion.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=0,   # deterministic
                max_tokens=500
            )
            answer = response["choices"][0]["message"]["content"]
            return answer.strip()

        except Exception as e:
            print("OpenAI chat error:", e)
            return "I’m not sure. Please send your question to hr@libertyng.com"

