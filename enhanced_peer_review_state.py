"""
Enhanced PeerReviewState for tracking iterative code reviews.
This enhances the original PeerReviewState with additional fields for
tracking iterations, review history, and targeted guidance.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ReviewIteration(BaseModel):
    """Model for storing a single review iteration."""
    
    iteration_number: int
    student_review: str
    review_analysis: Dict[str, Any]
    targeted_guidance: Optional[str] = None
    timestamp: Optional[str] = None

class EnhancedPeerReviewState(BaseModel):
    """Enhanced state for the peer code review agent workflow with iteration support."""
    
    # Input parameters
    programming_language: str = Field(default="java")  # Default to Java
    problem_areas: List[str] = Field(default_factory=lambda: ["style", "logical", "performance"])
    difficulty_level: str = Field(default="medium")
    code_length: str = Field(default="medium")
    
    # Generated during execution
    code_snippet: Optional[str] = None
    known_problems: Optional[List[str]] = None
    student_review: Optional[str] = None
    review_analysis: Optional[Dict] = None
    review_summary: Optional[str] = None
    comparison_report: Optional[str] = None
    
    # Iterative review tracking
    iteration_count: int = Field(default=1)
    max_iterations: int = Field(default=3)
    targeted_guidance: Optional[str] = None
    review_history: List[ReviewIteration] = Field(default_factory=list)
    review_sufficient: bool = Field(default=False)
    
    # Original raw error data for reference
    raw_errors: Optional[List[Dict[str, Any]]] = None
    
    # Flow control
    current_step: str = Field(default="generate_problem")
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True