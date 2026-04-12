from fastapi import APIRouter
from ..models.schemas import ChatRequest, ChatResponse
from ..services.chatbot_service import ChatbotService

router = APIRouter()
# One shared service instance.
service = ChatbotService()

@router.get("/health")
def health():
    # A check that the API process is up.
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # The service returns the reply parts and the route packs them into the API schema.
    reply, citations, actions, quick, conf, needs, q, opts, vstat, esc = service.respond(req.session_id, req.message)
    return ChatResponse(
        reply=reply,
        citations=citations,
        actions=actions,
        quick_replies=quick,
        confidence=conf,
        needs_clarification=needs,
        clarification_question=q or None,
        clarification_options=opts,
        verification_status=vstat,
        evidence_score=esc
    )
