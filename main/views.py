from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.mail import send_mail
from .embedding_service import get_embedding
from .vector_store import search
from .llm_service import generate_answer

class ChatAPIView(APIView):
    """
    Chat API for HR bot
    """

    def post(self, request):
        question = request.data.get("question")

        if not question:
            return Response(
                {"error": "Question is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 1️⃣ Get embedding
            query_embedding = get_embedding(question)

            # 2️⃣ Search FAISS
            results = search(query_embedding, k=5)

            # 3️⃣ If no results, fallback
            if not results:
                return Response({
                    "answer": "I’m not sure. Please send your question to hr@libertyng.com"
                })

            # 4️⃣ Combine context
            context_text = "\n".join(results)

            # 5️⃣ Generate answer from LLM (optional)
            answer = generate_answer(question, context_text)

            return Response({"answer": answer})

        except ValueError as ve:
            # Catch dimension mismatch or other known FAISS errors
            return Response({
                "answer": "I’m not sure. Please send your question to hr@libertyng.com"
            })

        except Exception as e:
            # Catch all unexpected errors
            print(f"Chat API Error: {e}")  # optional: log to Sentry or file
            return Response({
                "answer": "I’m not sure. Please send your question to hr@libertyng.com"
            })
