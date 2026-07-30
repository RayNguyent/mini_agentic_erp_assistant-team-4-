from datetime import datetime

from pydantic import BaseModel

from app.state import AgentState


class PendingApproval(BaseModel):
    id: str
    state: AgentState
    created_at: datetime
