import os
from typing import List, Dict, Any, Optional, Tuple, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
#from langgraph.prebuilt import ToolExecutor
from langchain_community.llms import Ollama
from langchain_core.messages import AIMessage, HumanMessage
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

# Load environment variables
load_dotenv()

# Initialize LLM models with Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2:1b")

generative_model = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=os.getenv("GENERATIVE_MODEL", DEFAULT_MODEL),
    temperature=float(os.getenv("GENERATIVE_TEMPERATURE", "0.7"))
)

review_model = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=os.getenv("REVIEW_MODEL", DEFAULT_MODEL),
    temperature=float(os.getenv("REVIEW_TEMPERATURE", "0.2"))
)

summary_model = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=os.getenv("SUMMARY_MODEL", DEFAULT_MODEL),
    temperature=float(os.getenv("SUMMARY_TEMPERATURE", "0.3"))
)

compare_model = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=os.getenv("COMPARE_MODEL", DEFAULT_MODEL),
    temperature=float(os.getenv("COMPARE_TEMPERATURE", "0.2"))
)

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

# Define the nodes for our graph
def generate_problem_node(state: PeerReviewState) -> PeerReviewState:
    """Generate code problems with intentional issues."""
    try:
        # Test Ollama connection first
        import requests
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code != 200:
                return PeerReviewState(
                    **state.dict(),
                    error=f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Status code: {response.status_code}",
                    current_step="error"
                )
            
            # Check if model exists
            models = response.json().get("models", [])
            model_exists = any(m.get("name") == DEFAULT_MODEL for m in models)
            if not model_exists:
                return PeerReviewState(
                    **state.dict(),
                    error=f"Model '{DEFAULT_MODEL}' not found in Ollama. Please run 'ollama pull {DEFAULT_MODEL}' first.",
                    current_step="error"
                )
        except requests.RequestException as re:
            return PeerReviewState(
                **state.dict(),
                error=f"Failed to connect to Ollama at {OLLAMA_BASE_URL}. Error: {str(re)}",
                current_step="error"
            )
        
        # Continue with code generation if Ollama is available
        code_snippet, known_problems = generate_code_problem(
            programming_language=state.programming_language,
            problem_areas=state.problem_areas,
            difficulty_level=state.difficulty_level,
            code_length=state.code_length,
            llm=generative_model
        )
        
        return PeerReviewState(
            **state.dict(),
            code_snippet=code_snippet,
            known_problems=known_problems,
            current_step="wait_for_review"
        )
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return PeerReviewState(
            **state.dict(),
            error=f"Error generating code problem: {str(e)}\n\nDetails: {error_traceback}",
            current_step="error"
        )

def analyze_review_node(state: PeerReviewState) -> PeerReviewState:
    """Analyze a student's review of code problems."""
    try:
        if not state.student_review:
            return PeerReviewState(
                **state.dict(),
                error="No student review provided for analysis.",
                current_step="error"
            )
        
        review_analysis = analyze_student_review(
            code_snippet=state.code_snippet,
            known_problems=state.known_problems,
            student_review=state.student_review,
            llm=review_model
        )
        
        return PeerReviewState(
            **state.dict(),
            review_analysis=review_analysis,
            current_step="summarize_review"
        )
    except Exception as e:
        return PeerReviewState(
            **state.dict(),
            error=f"Error analyzing student review: {str(e)}",
            current_step="error"
        )

def summarize_review_node(state: PeerReviewState) -> PeerReviewState:
    """Summarize review comments."""
    try:
        review_summary = summarize_review_comments(
            review_comments=state.student_review,
            llm=summary_model
        )
        
        return PeerReviewState(
            **state.dict(),
            review_summary=review_summary,
            current_step="compare_explain"
        )
    except Exception as e:
        return PeerReviewState(
            **state.dict(),
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
            llm=compare_model
        )
        
        return PeerReviewState(
            **state.dict(),
            comparison_report=comparison_report,
            current_step="complete"
        )
    except Exception as e:
        return PeerReviewState(
            **state.dict(),
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
    result = agent.invoke(initial_state)
    
    # Return the final state as a dictionary
    return result.dict()

if __name__ == "__main__":
    # This code will only run when executing this file directly
    # It's a simple example demonstrating agent functionality
    # For the UI, refer to app.py
    
    # Generate a code problem
    result = run_peer_review_agent()
    
    print("\n=== GENERATED CODE PROBLEM ===")
    print(result["code_snippet"])
    print("\n=== KNOWN PROBLEMS ===")
    for i, problem in enumerate(result["known_problems"], 1):
        print(f"{i}. {problem}")