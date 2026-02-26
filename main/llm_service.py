import requests
from django.conf import settings
import openai

openai.api_key = settings.OPENAI_API_KEY

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
# Answer only from the HR policies below.
# Answer questions based ONLY on the provided documents
def generate_answer(question, context, system_prompt=None):

    prompt = f"""
You are Liberty HR Assistant.
STRICT RULES:
1. Answer ONLY using the provided context.
2. If the answer is not explicitly in the context, say:
   "I'm not sure. Please send your question to hr@libertyng.com"
3. Do NOT make assumptions.
4. Cite the book title in your answer like: (Source: employee handbook)
            
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
    """
    if settings.AI_MODE == "local":
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": "llama3:8b",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
        # payload = {
        #     "model": settings.OLLAMA_CHAT_MODEL,
        #     "prompt": prompt,
        #     "stream": False
        # }

        # try:
        #     print(payload)
        #     response = requests.post(OLLAMA_GENERATE_URL, json=payload)

        #     print("STATUS:", response.status_code)
        #     print("RAW RESPONSE:", response.text)

        #     response.raise_for_status()

        #     data = response.json()

        #     return data.get("response", "No response field found")

        # except Exception as e:
        #     print("FULL ERROR:", repr(e))
        #     return None
    else:
        """
        Generate answer from OpenAI Chat model using system prompt and context.
        """
        try:
            messages = [
                {"role": "system", "content": prompt},
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
            return "I'm not sure. Please send your question to hr@libertyng.com"
