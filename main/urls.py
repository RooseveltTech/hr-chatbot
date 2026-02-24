from main import views
from django.urls import path
from django.views.generic import TemplateView
from django.conf import settings
from django.conf.urls.static import static

# urlpatterns = [
#     path('api/vi/fetch/', views.ChatAPIView.as_view(), name='fetch'),
# ]

urlpatterns = [
    path("", TemplateView.as_view(template_name="chat.html"), name="chat-home"),
    path("api/chat/", views.ChatAPIView.as_view(), name="chat-api"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)