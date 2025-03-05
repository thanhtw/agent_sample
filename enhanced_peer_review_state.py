"""
Enhanced PeerReviewState for tracking iterative code reviews.
This enhances the original PeerReviewState with additional fields for
tracking iterations, review history, targeted guidance, and specific error categories.
"""

from typing import List, Dict, Any, Optional
import copy
import datetime

class ReviewIteration:
    """Class for storing a single review iteration."""
    
    def __init__(self, iteration_number, student_review, review_analysis, targeted_guidance=None, timestamp=None):
        self.iteration_number = iteration_number
        self.student_review = student_review
        self.review_analysis = review_analysis
        self.targeted_guidance = targeted_guidance
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "student_review": self.student_review,
            "review_analysis": self.review_analysis,
            "targeted_guidance": self.targeted_guidance,
            "timestamp": self.timestamp
        }

class EnhancedPeerReviewState:
    """Enhanced state for the peer code review agent workflow with iteration support."""
    
    def __init__(self, **kwargs):
        # Input parameters
        self.programming_language = kwargs.get("programming_language", "java")
        self.problem_areas = kwargs.get("problem_areas", ["style", "logical", "performance"])
        self.difficulty_level = kwargs.get("difficulty_level", "medium")
        self.code_length = kwargs.get("code_length", "medium")
        
        # Specific error categories (new parameter)
        self.specific_error_categories = kwargs.get("specific_error_categories")
        
        # Generated during execution
        self.code_snippet = kwargs.get("code_snippet")
        self.known_problems = kwargs.get("known_problems", [])
        self.student_review = kwargs.get("student_review")
        self.review_analysis = kwargs.get("review_analysis")
        self.review_summary = kwargs.get("review_summary")
        self.comparison_report = kwargs.get("comparison_report")
        
        # Iterative review tracking
        self.iteration_count = kwargs.get("iteration_count", 1)
        self.max_iterations = kwargs.get("max_iterations", 3)
        self.targeted_guidance = kwargs.get("targeted_guidance")
        
        # Handle review_history specially
        self.review_history = []
        if "review_history" in kwargs and kwargs["review_history"]:
            if isinstance(kwargs["review_history"][0], dict):
                # Convert dictionaries to ReviewIteration objects
                for item_dict in kwargs["review_history"]:
                    self.review_history.append(ReviewIteration(
                        iteration_number=item_dict.get("iteration_number"),
                        student_review=item_dict.get("student_review"),
                        review_analysis=item_dict.get("review_analysis"),
                        targeted_guidance=item_dict.get("targeted_guidance"),
                        timestamp=item_dict.get("timestamp")
                    ))
            else:
                # Already ReviewIteration objects
                self.review_history = kwargs["review_history"]
            
        self.review_sufficient = kwargs.get("review_sufficient", False)
        
        # Original raw error data for reference
        self.raw_errors = kwargs.get("raw_errors")
        
        # Flow control
        self.current_step = kwargs.get("current_step", "generate_problem")
        self.error = kwargs.get("error")
    
    def to_dict(self):
        """Convert state to dictionary."""
        return {
            "programming_language": self.programming_language,
            "problem_areas": self.problem_areas,
            "difficulty_level": self.difficulty_level,
            "code_length": self.code_length,
            "specific_error_categories": self.specific_error_categories,
            "code_snippet": self.code_snippet,
            "known_problems": self.known_problems,
            "student_review": self.student_review,
            "review_analysis": self.review_analysis,
            "review_summary": self.review_summary,
            "comparison_report": self.comparison_report,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "targeted_guidance": self.targeted_guidance,
            "review_history": [item.to_dict() for item in self.review_history],
            "review_sufficient": self.review_sufficient,
            "raw_errors": self.raw_errors,
            "current_step": self.current_step,
            "error": self.error
        }
    
    def dict(self):
        """For compatibility with previous pydantic's dict() method."""
        return self.to_dict()
    
    def copy(self):
        """Create a deep copy of the state."""
        return copy.deepcopy(self)