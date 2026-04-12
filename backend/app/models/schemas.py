from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# These schema classes are the contract between the widget and the API.
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session id")
    message: str = Field(..., min_length=1, description="User message text")


class Citation(BaseModel):
    title: str
    url: str


class Action(BaseModel):
    type: str
    payload: Dict[str, Any] = {}


class ClarificationOption(BaseModel):
    label: str
   
    payload: Dict[str, Any] = {}


class ChatResponse(BaseModel):
    reply: str
    citations: List[Citation] = []
    actions: List[Action] = []
    quick_replies: List[str] = []
    confidence: Optional[float] = None

  
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: List[ClarificationOption] = []


    verification_status: Optional[str] = None 
    evidence_score: Optional[float] = None
