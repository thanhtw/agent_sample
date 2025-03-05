"""
Peer Code Review Training Agent

This module implements a LangGraph-based workflow for training students in code review
skills with the following stages:
1. Generate code with intentional problems
2. Analyze student reviews
3. Summarize review feedback
4. Compare student review with expected issues and explain differences
"""

import os
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Import custom prompts and tools
from agent_prompts import (
    GENERATIVE_AGENT_PROMPT,
    REVIEW_AGENT_PROMPT,
    SUMMARY_AGENT_PROMPT,
    COMPARE_EXPLAIN_AGENT_PROMPT
)
from agent_tools import (
    generate_code_problem,
    analyze_student_review,
    summarize_review_comments,
    compare_and_explain
)

# Import the LLM Manager
from llm_manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM Manager
llm_manager = LLMManager()

# Define state for the graph
class PeerReviewState(BaseModel):
    """State for the peer code review agent workflow."""
    
    # Input parameters
    programming_language: str = Field(default="python")
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
    
    # Flow control
    current_step: str = Field(default="generate_problem")
    error: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

# Initialize all models
def initialize_llm_models():
    """Initialize all LLM models needed for the peer review process."""
    models = {}
    
    # Try to initialize models with LLM Manager
    models["generative"] = llm_manager.initialize_model_from_env("GENERATIVE_MODEL", "GENERATIVE_TEMPERATURE")
    models["review"] = llm_manager.initialize_model_from_env("REVIEW_MODEL", "REVIEW_TEMPERATURE")
    models["summary"] = llm_manager.initialize_model_from_env("SUMMARY_MODEL", "SUMMARY_TEMPERATURE")
    models["compare"] = llm_manager.initialize_model_from_env("COMPARE_MODEL", "COMPARE_TEMPERATURE")
    
    # Fall back to defaults for any model that failed to initialize
    if not models["generative"]:
        logger.info("Falling back to default model for generative tasks")
        default_temp = float(os.getenv("GENERATIVE_TEMPERATURE", "0.7"))
        models["generative"] = llm_manager.initialize_model(
            os.getenv("DEFAULT_MODEL", "llama3:1b"),
            {"temperature": default_temp}
        )
    
    if not models["review"]:
        logger.info("Falling back to default model for review tasks")
        default_temp = float(os.getenv("REVIEW_TEMPERATURE", "0.2"))
        models["review"] = llm_manager.initialize_model(
            os.getenv("DEFAULT_MODEL", "llama3:1b"),
            {"temperature": default_temp}
        )
        
    if not models["summary"]:
        logger.info("Falling back to default model for summary tasks")
        default_temp = float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))
        models["summary"] = llm_manager.initialize_model(
            os.getenv("DEFAULT_MODEL", "llama3:1b"),
            {"temperature": default_temp}
        )
        
    if not models["compare"]:
        logger.info("Falling back to default model for comparison tasks")
        default_temp = float(os.getenv("COMPARE_TEMPERATURE", "0.2"))
        models["compare"] = llm_manager.initialize_model(
            os.getenv("DEFAULT_MODEL", "llama3:1b"),
            {"temperature": default_temp}
        )
    
    return models

# Initialize all LLM models
llm_models = initialize_llm_models()

# Define the nodes for our graph
def generate_problem_node(state: PeerReviewState) -> PeerReviewState:
    """Generate code problems with intentional issues."""
    try:
        # Test Ollama connection first
        connection_status, message = llm_manager.check_ollama_connection()
        
        if not connection_status:
            return PeerReviewState(
                programming_language=state.programming_language,
                problem_areas=state.problem_areas,
                difficulty_level=state.difficulty_level,
                code_length=state.code_length,
                error=f"Cannot connect to Ollama: {message}",
                current_step="error"
            )
        
        # Check if default model exists
        default_model = llm_manager.default_model
        if not llm_manager.check_model_availability(default_model):
            return PeerReviewState(
                programming_language=state.programming_language,
                problem_areas=state.problem_areas,
                difficulty_level=state.difficulty_level,
                code_length=state.code_length,
                error=f"Model '{default_model}' not found in Ollama. Please run 'ollama pull {default_model}' first.",
                current_step="error"
            )
        
        # Continue with code generation if Ollama is available
        code_snippet, known_problems = generate_code_problem(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            llm=llm_models["generative"]
        )
        
        # Create a new state with updated values
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=code_snippet,
            known_problems=known_problems,
            current_step="wait_for_review"
        )
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error generating code problem: {str(e)}\n{error_traceback}")
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            error=f"Error generating code problem: {str(e)}",
            current_step="error"
        )

def analyze_review_node(state: PeerReviewState) -> PeerReviewState:
    """Analyze a student's review of code problems."""
    try:
        if not state.student_review:
            return PeerReviewState(
                programming_language=state.programming_language,
                problem_areas=state.problem_areas,
                difficulty_level=state.difficulty_level,
                code_length=state.code_length,
                code_snippet=state.code_snippet,
                known_problems=state.known_problems,
                error="No student review provided for analysis.",
                current_step="error"
            )
        
        review_analysis = analyze_student_review(
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            llm=llm_models["review"]
        )
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=review_analysis,
            current_step="summarize_review"
        )
    
    except Exception as e:
        logger.error(f"Error analyzing student review: {str(e)}")
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            error=f"Error analyzing student review: {str(e)}",
            current_step="error"
        )

def summarize_review_node(state: PeerReviewState) -> PeerReviewState:
    """Summarize review comments."""
    try:
        review_summary = summarize_review_comments(
            review_comments=state.student_review,
            llm=llm_models["summary"]
        )
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            review_summary=review_summary,
            current_step="compare_explain"
        )
    
    except Exception as e:
        logger.error(f"Error summarizing review: {str(e)}")
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            error=f"Error summarizing review: {str(e)}",
            current_step="error"
        )

def compare_explain_node(state: PeerReviewState) -> PeerReviewState:
    """Compare student review with known problems and explain differences."""
    try:
        comparison_report = compare_and_explain(
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            review_summary=state.review_summary,
            llm=llm_models["compare"]
        )
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            review_summary=state.review_summary,
            comparison_report=comparison_report,
            current_step="complete"
        )
    
    except Exception as e:
        logger.error(f"Error comparing and explaining: {str(e)}")
        
        return PeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            review_summary=state.review_summary,
            error=f"Error comparing and explaining: {str(e)}",
            current_step="error"
        )

# Router function to determine next step
def router(state: PeerReviewState) -> Literal["generate_problem", "analyze_review", "summarize_review", "compare_explain", "complete", "error"]:
    """Determine the next step in the workflow."""
    if state.error:
        return "error"
    return state.current_step

# Create the peer review agent graph
def create_peer_review_agent():
    """Create and compile the peer review agent workflow."""
    
    # Initialize the graph
    graph = StateGraph(PeerReviewState)
    
    # Add nodes
    graph.add_node("generate_problem", generate_problem_node)
    graph.add_node("analyze_review", analyze_review_node)
    graph.add_node("summarize_review", summarize_review_node)
    graph.add_node("compare_explain", compare_explain_node)
    
    # Add conditional edges
    graph.add_conditional_edges(
        "generate_problem",
        router,
        {
            "wait_for_review": "analyze_review",
            "error": END
        }
    )
    
    graph.add_conditional_edges(
        "analyze_review",
        router,
        {
            "summarize_review": "summarize_review",
            "error": END
        }
    )
    
    graph.add_conditional_edges(
        "summarize_review",
        router,
        {
            "compare_explain": "compare_explain",
            "error": END
        }
    )
    
    graph.add_conditional_edges(
        "compare_explain",
        router,
        {
            "complete": END,
            "error": END
        }
    )
    
    # Set entry point
    graph.set_entry_point("generate_problem")
    
    # Compile the graph
    return graph.compile()

# User-facing function to run the agent
def run_peer_review_agent(
    programming_language: str = "python",
    problem_areas: List[str] = None,
    difficulty_level: str = "medium",
    code_length: str = "medium",
    student_review: Optional[str] = None
) -> Dict:
    """
    Run the peer code review agent workflow.
    
    Args:
        programming_language: The programming language for the code
        problem_areas: Areas to include problems from (style, logical, performance, security, design)
        difficulty_level: How difficult the problems should be (easy, medium, hard)
        code_length: Approximate length of code (short, medium, long)
        student_review: The student's review of the code (optional for initial generation)
        
    Returns:
        Dictionary with the results of the workflow
    """
    # Set default problem areas if none provided
    if problem_areas is None:
        problem_areas = ["style", "logical", "performance"]
        
    # Create the initial state
    initial_state = PeerReviewState(
        programming_language=programming_language,
        problem_areas=problem_areas,
        difficulty_level=difficulty_level,
        code_length=code_length,
        student_review=student_review
    )
    
    # Create the agent
    agent = create_peer_review_agent()
    
    # Run the agent
    logger.info(f"Running peer review agent for {programming_language}")
    result = agent.invoke(initial_state)
    
    # Return the final state as a dictionary
    # Using manual conversion to avoid compatibility issues
    return {
        "programming_language": result.programming_language,
        "problem_areas": result.problem_areas,
        "difficulty_level": result.difficulty_level,
        "code_length": result.code_length,
        "code_snippet": result.code_snippet,
        "known_problems": result.known_problems,
        "student_review": result.student_review,
        "review_analysis": result.review_analysis,
        "review_summary": result.review_summary,
        "comparison_report": result.comparison_report,
        "current_step": result.current_step,
        "error": result.error
    }

if __name__ == "__main__":
    # This code will only run when executing this file directly
    # It's a simple example demonstrating agent functionality
    
    # Generate a code problem
    result = run_peer_review_agent()
    
    print("\n=== GENERATED CODE PROBLEM ===")
    print(result["code_snippet"])
    print("\n=== KNOWN PROBLEMS ===")
    for i, problem in enumerate(result["known_problems"], 1):
        print(f"{i}. {problem}")