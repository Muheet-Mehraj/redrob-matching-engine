from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class SkillRecord(BaseModel):
    name: str
    proficiency: str = "beginner"
    duration_months: int = 0
    endorsements: int = 0

class CareerTenure(BaseModel):
    title: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: int = 0
    description: Optional[str] = ""
    industry: Optional[str] = ""

class BehavioralSignals(BaseModel):
    last_active_date: Optional[str] = None
    open_to_work_flag: bool = False
    recruiter_response_rate: float = 0.5
    interview_completion_rate: float = 0.5
    notice_period_days: int = 30
    github_activity_score: int = -1
    profile_completeness_score: int = 50

class CandidateSubmissionRow(BaseModel):
    candidate_id: str = Field(..., pattern=r"^CAND_\d+$")
    rank: int = Field(..., ge=1, le=100)
    score: float
    reasoning: str = Field(..., min_length=10, max_length=1000)
