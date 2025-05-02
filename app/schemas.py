from pydantic import BaseModel
from typing import List

class Source(BaseModel):
    question_id: int
    url: str
    excerpt: str

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    sources: List[Source]