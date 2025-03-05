"""
Enhanced Peer Code Review Training Agent

This module implements a LangGraph-based workflow for training students in Java code review
skills with the following stages:
1. Generate Java code with intentional problems from build_errors.json and checkstyle_error.json
2. Analyze student reviews with iterative feedback
3. Summarize review feedback
4. Compare student review with expected issues and explain differences
"""

import os
import logging
import traceback
import datetime
from typing import List, Dict, Any, Optional, Tuple, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Import custom prompts and tools
from agent_prompts import (
    JAVA_GENERATIVE_AGENT_PROMPT,
    JAVA_REVIEW_AGENT_PROMPT,
    JAVA_SUMMARY_AGENT_PROMPT,
    JAVA_COMPARE_EXPLAIN_AGENT_PROMPT
)

# Import enhanced agent tools
from agent_tools import (
    generate_code_problem,
    analyze_student_review,
    summarize_review_comments,
    compare_and_explain,
    generate_targeted_guidance
)

# Import the enhanced state model
from enhanced_peer_review_state import EnhancedPeerReviewState, ReviewIteration

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

# Define the nodes for our enhanced graph
def generate_problem_node(state: EnhancedPeerReviewState) -> EnhancedPeerReviewState:
    """Generate code problems with intentional issues."""
    try:
        # Test Ollama connection first
        connection_status, message = llm_manager.check_ollama_connection()
        
        if not connection_status:
            return EnhancedPeerReviewState(
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
            return EnhancedPeerReviewState(
                programming_language=state.programming_language,
                problem_areas=state.problem_areas,
                difficulty_level=state.difficulty_level,
                code_length=state.code_length,
                error=f"Model '{default_model}' not found in Ollama. Please run 'ollama pull {default_model}' first.",
                current_step="error"
            )
        
        # Continue with code generation if Ollama is available
        code_snippet, known_problems, raw_errors = generate_code_problem(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            llm=llm_models["generative"]
        )
        
        # Create a new state with updated values
        return EnhancedPeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            code_snippet=code_snippet,
            known_problems=known_problems,
            raw_errors=raw_errors,
            current_step="wait_for_review",
            iteration_count=1,
            max_iterations=3  # Default to 3 attempts
        )
    
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error generating code problem: {str(e)}\n{error_traceback}")
        
        return EnhancedPeerReviewState(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            error=f"Error generating code problem: {str(e)}",
            current_step="error"
        )

def analyze_review_node(state: EnhancedPeerReviewState) -> EnhancedPeerReviewState:
    """Analyze a student's review of code problems with iteration support."""
    try:
        if not state.student_review:
            return EnhancedPeerReviewState(
                **state.dict(),
                error="No student review provided for analysis.",
                current_step="error"
            )
        
        # Analyze the student review
        review_analysis = analyze_student_review(
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            llm=llm_models["review"]
        )
        
        # Create a review iteration record
        current_iteration = ReviewIteration(
            iteration_number=state.iteration_count,
            student_review=state.student_review,
            review_analysis=review_analysis,
            timestamp=datetime.datetime.now().isoformat()
        )
        
        # Update review history
        review_history = state.review_history.copy()
        review_history.append(current_iteration)
        
        # Check if the review is sufficient
        if review_analysis.get("review_sufficient", False):
            # If sufficient, proceed to summary
            return EnhancedPeerReviewState(
                **state.dict(),
                review_analysis=review_analysis,
                review_history=review_history,
                review_sufficient=True,
                current_step="summarize_review"
            )
        else:
            # If not sufficient and we have more iterations available
            if state.iteration_count < state.max_iterations:
                # Generate targeted guidance
                targeted_guidance = generate_targeted_guidance(
                    code_snippet=state.code_snippet,
                    known_problems=state.known_problems,
                    student_review=state.student_review,
                    review_analysis=review_analysis,
                    iteration_count=state.iteration_count,
                    max_iterations=state.max_iterations,
                    llm=llm_models["review"]
                )
                
                # Update state for next iteration
                return EnhancedPeerReviewState(
                    **state.dict(),
                    review_analysis=review_analysis,
                    review_history=review_history,
                    targeted_guidance=targeted_guidance,
                    iteration_count=state.iteration_count + 1,
                    review_sufficient=False,
                    current_step="wait_for_review"  # Go back to waiting for next review
                )
            else:
                # If we've run out of iterations, proceed anyway
                logger.info("Max iterations reached, proceeding to summary despite insufficient review")
                return EnhancedPeerReviewState(
                    **state.dict(),
                    review_analysis=review_analysis,
                    review_history=review_history,
                    review_sufficient=False,
                    current_step="summarize_review"
                )
    
    except Exception as e:
        logger.error(f"Error analyzing student review: {str(e)}")
        
        return EnhancedPeerReviewState(
            **state.dict(),
            error=f"Error analyzing student review: {str(e)}",
            current_step="error"
        )

def summarize_review_node(state: EnhancedPeerReviewState) -> EnhancedPeerReviewState:
    """Summarize review comments from all iterations."""
    try:
        # Get the latest/best review
        latest_review = state.student_review
        
        # Determine if code is Java
        is_java = state.programming_language.lower() == "java"
        
        review_summary = summarize_review_comments(
            review_comments=latest_review,
            llm=llm_models["summary"],
            is_java=is_java
        )
        
        return EnhancedPeerReviewState(
            **state.dict(),
            review_summary=review_summary,
            current_step="compare_explain"
        )
    
    except Exception as e:
        logger.error(f"Error summarizing review: {str(e)}")
        
        return EnhancedPeerReviewState(
            **state.dict(),
            error=f"Error summarizing review: {str(e)}",
            current_step="error"
        )

def compare_explain_node(state: EnhancedPeerReviewState) -> EnhancedPeerReviewState:
    """Compare student review with known problems and explain differences."""
    try:
        # Convert review history to a format usable by the compare function
        review_history_dicts = []
        for review in state.review_history:
            review_history_dicts.append({
                "iteration": review.iteration_number,
                "student_review": review.student_review,
                "review_analysis": review.review_analysis,
                "timestamp": review.timestamp
            })
        
        # Determine if code is Java
        is_java = state.programming_language.lower() == "java"
        
        comparison_report = compare_and_explain(
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            review_analysis=state.review_analysis,
            review_summary=state.review_summary,
            review_history=review_history_dicts if len(review_history_dicts) > 1 else None,
            llm=llm_models["compare"],
            is_java=is_java
        )
        
        return EnhancedPeerReviewState(
            **state.dict(),
            comparison_report=comparison_report,
            current_step="complete"
        )
    
    except Exception as e:
        logger.error(f"Error comparing and explaining: {str(e)}")
        
        return EnhancedPeerReviewState(
            **state.dict(),
            error=f"Error comparing and explaining: {str(e)}",
            current_step="error"
        )

# Router function to determine next step
def router(state: EnhancedPeerReviewState) -> Literal["generate_problem", "analyze_review", "summarize_review", "compare_explain", "wait_for_review", "complete", "error"]:
    """Determine the next step in the workflow."""
    if state.error:
        return "error"
    return state.current_step

# Create the enhanced peer review agent graph
def create_peer_review_agent():
    """Create and compile the enhanced peer review agent workflow."""
    
    # Initialize the graph with the enhanced state
    graph = StateGraph(EnhancedPeerReviewState)
    
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
            "wait_for_review": "analyze_review",  # Loop back for more review iterations
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
    programming_language: str = "java",  # Default to Java now
    problem_areas: List[str] = None,
    difficulty_level: str = "medium",
    code_length: str = "medium",
    student_review: Optional[str] = None,
    iteration_count: int = None,
    max_iterations: int = 3
) -> Dict:
    """
    Run the peer code review agent workflow.
    
    Args:
        programming_language: The programming language for the code
        problem_areas: Areas to include problems from (style, logical, performance, security, design)
        difficulty_level: How difficult the problems should be (easy, medium, hard)
        code_length: Approximate length of code (short, medium, long)
        student_review: The student's review of the code (optional for initial generation)
        iteration_count: Current review iteration (optional)
        max_iterations: Maximum number of review iterations allowed
        
    Returns:
        Dictionary with the results of the workflow
    """
    # Set default problem areas if none provided
    if problem_areas is None:
        problem_areas = ["style", "logical", "performance"]
        
    # Create the initial state
    initial_state = EnhancedPeerReviewState(
        programming_language=programming_language,
        problem_areas=problem_areas,
        difficulty_level=difficulty_level,
        code_length=code_length,
        student_review=student_review,
        iteration_count=iteration_count or 1,
        max_iterations=max_iterations
    )
    
    # Create the agent
    agent = create_peer_review_agent()
    
    # Run the agent
    logger.info(f"Running peer review agent for {programming_language}")
    try:
        result = agent.invoke(initial_state)
        
        # Convert the result to a dictionary
        try:
            return result.dict()
        except Exception as e:
            logger.error(f"Error converting result to dict: {str(e)}")
            return {
                "programming_language": programming_language,
                "problem_areas": problem_areas,
                "difficulty_level": difficulty_level,
                "code_length": code_length,
                "code_snippet": getattr(result, "code_snippet", None),
                "known_problems": getattr(result, "known_problems", []),
                "error": f"Error converting result to dict: {str(e)}"
            }
        
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return {
            "programming_language": programming_language,
            "problem_areas": problem_areas,
            "difficulty_level": difficulty_level,
            "code_length": code_length,
            "error": f"Error running agent: {str(e)}"
        }

if __name__ == "__main__":
    # This code will only run when executing this file directly
    # It's a simple example demonstrating agent functionality
    
    # Generate a Java code problem
    result = run_peer_review_agent()
    
    print("\n=== GENERATED JAVA CODE PROBLEM ===")
    print(result["code_snippet"])
    print("\n=== KNOWN PROBLEMS ===")
    for i, problem in enumerate(result["known_problems"], 1):
        print(f"{i}. {problem}")