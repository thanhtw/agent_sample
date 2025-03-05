"""
Enhanced Streamlit app for the Java Peer Code Review Training System.

This module provides a web interface with iterative review support,
allowing users to generate Java code problems, submit reviews,
receive feedback, and iterate on their reviews when needed.
Enhanced with specific error category selection.
"""

import streamlit as st
import os
import logging
import time
import datetime
import json
from typing import Dict, List, Any
from agent import run_peer_review_agent
from dotenv import load_dotenv

# Import LLM Manager and UI components
from llm_manager import LLMManager
from model_setup_ui import ModelSetupUI

# Import Java error handler to get available error categories
from java_error_handler import get_all_error_categories, get_all_error_types

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
model_setup_ui = ModelSetupUI(llm_manager)

# Set page config
st.set_page_config(
    page_title="Java Code Review Training System",
    page_icon="☕",  # Java coffee cup icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            margin-bottom: 0.5rem;
        }
        .stTextArea textarea {
            font-family: monospace;
        }
        .code-block {
            background-color: #f0f0f0;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .problem-list {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
            padding: 10px;
            margin: 10px 0;
        }
        .student-review {
            background-color: #f8f9fa;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 10px 0;
        }
        .review-analysis {
            background-color: #f8f9fa;
            border-left: 4px solid #FF9800;
            padding: 10px;
            margin: 10px 0;
        }
        .comparison-report {
            background-color: #f8f9fa;
            border-left: 4px solid #9C27B0;
            padding: 10px;
            margin: 10px 0;
        }
        .model-container {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            background-color: #f8f9fa;
        }
        .status-ok {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-warning {
            color: #FF9800;
            font-weight: bold;
        }
        .status-error {
            color: #F44336;
            font-weight: bold;
        }
        .guidance-box {
            background-color: #e8f4f8;
            border-left: 4px solid #03A9F4;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .iteration-badge {
            background-color: #E1F5FE;
            color: #0288D1;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            display: inline-block;
            margin-left: 10px;
        }
        .feedback-box {
            background-color: #E8F5E9;
            border-left: 4px solid #4CAF50;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #FFF8E1;
            border-left: 4px solid #FFC107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        .review-history-item {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin: 10px 0;
            background-color: #fafafa;
        }
        .tab-content {
            padding: 1rem;
            border: 1px solid #ddd;
            border-top: none;
            border-radius: 0 0 5px 5px;
        }
        .java-keyword {
            color: #0033B3;
            font-weight: bold;
        }
        .java-comment {
            color: #808080;
            font-style: italic;
        }
        .error-type-header {
            background-color: #f1f8ff;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-weight: bold;
        }
        .error-category {
            border-left: 3px solid #2196F3;
            padding-left: 10px;
            margin: 5px 0;
        }
        .subcategory-container {
            margin-left: 20px;
            border-left: 2px solid #e6e6e6;
            padding-left: 10px;
            margin-top: 5px;
            margin-bottom: 15px;
        }
        .error-item {
            margin: 5px 0;
            padding: 3px 5px;
            font-size: 0.9em;
            background-color: #f9f9f9;
            border-radius: 3px;
        }
        .show-hide-btn {
            background-color: #f1f8ff;
            border: 1px solid #ddedff;
            color: #0366d6;
            padding: 2px 8px;
            font-size: 0.8em;
            border-radius: 4px;
            margin-left: 10px;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

def add_line_numbers(code: str) -> str:
    """
    Add line numbers to code snippet.
    
    Args:
        code (str): The code snippet to add line numbers to
        
    Returns:
        str: Code with line numbers
    """
    lines = code.splitlines()
    max_line_num = len(lines)
    padding = len(str(max_line_num))
    
    # Create a list of lines with line numbers
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        # Format line number with consistent padding
        line_num = str(i).rjust(padding)
        numbered_lines.append(f"{line_num} | {line}")
    
    return "\n".join(numbered_lines)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'code_snippet' not in st.session_state:
        st.session_state.code_snippet = ""
    if 'known_problems' not in st.session_state:
        st.session_state.known_problems = []
    if 'student_review' not in st.session_state:
        st.session_state.student_review = ""
    if 'review_analysis' not in st.session_state:
        st.session_state.review_analysis = None
    if 'review_summary' not in st.session_state:
        st.session_state.review_summary = ""
    if 'comparison_report' not in st.session_state:
        st.session_state.comparison_report = ""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "generate"
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'models_configured' not in st.session_state:
        st.session_state.models_configured = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # New fields for iterative review
    if 'iteration_count' not in st.session_state:
        st.session_state.iteration_count = 1
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 3
    if 'review_sufficient' not in st.session_state:
        st.session_state.review_sufficient = False
    if 'targeted_guidance' not in st.session_state:
        st.session_state.targeted_guidance = None
    if 'review_history' not in st.session_state:
        st.session_state.review_history = []
    if 'raw_errors' not in st.session_state:
        st.session_state.raw_errors = None
        
    # New fields for error category selection
    if 'selected_error_categories' not in st.session_state:
        st.session_state.selected_error_categories = {"build": [], "checkstyle": []}
    if 'error_selection_mode' not in st.session_state:
        st.session_state.error_selection_mode = "standard"  # "standard" or "advanced"
    if 'expanded_categories' not in st.session_state:
        st.session_state.expanded_categories = {}

def check_model_status():
    """Check the status of all required models."""
    # Check Ollama connection
    status = {
        "ollama_running": False,
        "default_model_available": False,
        "all_models_configured": False
    }
    
    # Check if Ollama is running
    connection_status, _ = llm_manager.check_ollama_connection()
    status["ollama_running"] = connection_status
    
    # Check if default model is available
    if connection_status:
        default_model = llm_manager.default_model
        status["default_model_available"] = llm_manager.check_model_availability(default_model)
    
    # Check if all role-specific models are configured in environment
    required_models = ["GENERATIVE_MODEL", "REVIEW_MODEL", "SUMMARY_MODEL", "COMPARE_MODEL"]
    status["all_models_configured"] = all(os.getenv(model) for model in required_models)
    
    return status

def generate_problem(programming_language, problem_areas, difficulty_level, code_length, selected_error_categories=None):
    """Generate a code problem with a progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Show progress during generation
        status_text.text("Initializing models...")
        progress_bar.progress(10)
        time.sleep(0.5)
        
        # Log the selected error categories for debugging
        if selected_error_categories:
            logger.info(f"Using selected error categories: {json.dumps(selected_error_categories)}")
        
        status_text.text("Generating Java code problem...")
        progress_bar.progress(30)
        
        # Convert inputs to lowercase for the backend
        # Include selected error categories if in advanced mode
        if selected_error_categories and st.session_state.error_selection_mode == "advanced":
            # Ensure we only pass non-empty categories
            filtered_categories = {
                "build": [c for c in selected_error_categories["build"] if c],
                "checkstyle": [c for c in selected_error_categories["checkstyle"] if c]
            }
            
            # Only pass if we have at least one category selected
            if filtered_categories["build"] or filtered_categories["checkstyle"]:
                logger.info(f"Using filtered error categories: {json.dumps(filtered_categories)}")
                result = run_peer_review_agent(
                    programming_language=programming_language.lower(),
                    problem_areas=[area.lower() for area in problem_areas],
                    difficulty_level=difficulty_level.lower(),
                    code_length=code_length.lower(),
                    specific_error_categories=filtered_categories
                )
            else:
                # Fall back to standard mode if no categories selected
                logger.info("No error categories selected, falling back to standard mode")
                result = run_peer_review_agent(
                    programming_language=programming_language.lower(),
                    problem_areas=[area.lower() for area in problem_areas],
                    difficulty_level=difficulty_level.lower(),
                    code_length=code_length.lower()
                )
        else:
            result = run_peer_review_agent(
                programming_language=programming_language.lower(),
                problem_areas=[area.lower() for area in problem_areas],
                difficulty_level=difficulty_level.lower(),
                code_length=code_length.lower()
            )
        
        progress_bar.progress(90)
        status_text.text("Finalizing results...")
        time.sleep(0.5)
        
        # Update session state
        st.session_state.code_snippet = result.get("code_snippet", "")
        st.session_state.known_problems = result.get("known_problems", [])
        st.session_state.raw_errors = result.get("raw_errors", [])
        st.session_state.current_step = "review"
        st.session_state.error = None
        st.session_state.active_tab = 1  # Move to the review tab
        st.session_state.iteration_count = 1
        st.session_state.review_history = []
        st.session_state.targeted_guidance = None
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating code problem: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = str(e)
        return False

def analyze_review():
    """Analyze the student review with a progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Show progress during analysis
        status_text.text("Processing student review...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        # Run the analysis step
        programming_language = st.session_state.get("programming_language", "java")
        problem_areas = st.session_state.get("problem_areas", ["style", "logical", "performance"])
        difficulty_level = st.session_state.get("difficulty_level", "medium")
        code_length = st.session_state.get("code_length", "medium")
        iteration_count = st.session_state.get("iteration_count", 1)
        max_iterations = st.session_state.get("max_iterations", 3)
        
        status_text.text("Analyzing your review...")
        progress_bar.progress(40)
        
        # Include selected error categories if in advanced mode
        if st.session_state.error_selection_mode == "advanced":
            result = run_peer_review_agent(
                student_review=st.session_state.student_review,
                # Use the same parameters as before
                programming_language=programming_language.lower(),
                problem_areas=[area.lower() for area in problem_areas],
                difficulty_level=difficulty_level.lower(),
                code_length=code_length.lower(),
                iteration_count=iteration_count,
                max_iterations=max_iterations,
                specific_error_categories=st.session_state.selected_error_categories
            )
        else:
            result = run_peer_review_agent(
                student_review=st.session_state.student_review,
                # Use the same parameters as before
                programming_language=programming_language.lower(),
                problem_areas=[area.lower() for area in problem_areas],
                difficulty_level=difficulty_level.lower(),
                code_length=code_length.lower(),
                iteration_count=iteration_count,
                max_iterations=max_iterations
            )
        
        status_text.text("Generating feedback...")
        progress_bar.progress(70)
        
        # Update session state
        st.session_state.review_analysis = result.get("review_analysis", {})
        st.session_state.review_summary = result.get("review_summary", "")
        st.session_state.comparison_report = result.get("comparison_report", "")
        st.session_state.review_sufficient = result.get("review_sufficient", False)
        st.session_state.targeted_guidance = result.get("targeted_guidance", "")
        st.session_state.review_history = result.get("review_history", [])
        st.session_state.error = None
        
        # Decide which tab to display next
        if st.session_state.review_sufficient or iteration_count >= max_iterations:
            st.session_state.active_tab = 2  # Move to the analysis tab
        else:
            # Update iteration count for next review
            st.session_state.iteration_count = iteration_count + 1
            st.session_state.active_tab = 1  # Stay on review tab for another attempt
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return True
    
    except Exception as e:
        logger.error(f"Error analyzing review: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.session_state.error = str(e)
        return False

def display_error_category_selection():
    """Display advanced error category selection interface with subitems."""
    # Get all error categories
    all_categories = get_all_error_categories()
    all_error_types = get_all_error_types()
    
    st.subheader("Select Specific Error Categories")
    st.info("Choose specific error categories to include in the generated Java code.")
    
    # Add CSS for nested subcategory display
    st.markdown("""
    <style>
        .subcategory-container {
            margin-left: 20px;
            border-left: 2px solid #e6e6e6;
            padding-left: 10px;
        }
        .error-item {
            margin: 5px 0;
            padding: 3px 0;
            font-size: 0.9em;
        }
        .category-header {
            font-weight: bold;
            margin-top: 10px;
            background-color: #f1f1f1;
            padding: 5px;
            border-radius: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize state for expanded categories if not exists
    if 'expanded_categories' not in st.session_state:
        st.session_state.expanded_categories = {}
    
    # Build errors
    st.markdown("<div class='error-type-header'>Build Errors</div>", unsafe_allow_html=True)
    build_categories = all_categories["build"]
    
    # Create a multi-column layout for build errors
    build_cols = st.columns(2)
    
    # Split the build categories into two columns
    half_length = len(build_categories) // 2
    for i, col in enumerate(build_cols):
        start_idx = i * half_length
        end_idx = start_idx + half_length if i == 0 else len(build_categories)
        
        with col:
            for category in build_categories[start_idx:end_idx]:
                # Show how many errors in this category
                error_count = len(all_error_types["build"].get(category, []))
                category_label = f"{category} ({error_count} errors)"
                
                # Create a unique key for this category
                category_key = f"build_{category}"
                
                # Check if category is selected
                is_selected = st.checkbox(category_label, key=category_key)
                
                # Update selection state
                if is_selected:
                    if category not in st.session_state.selected_error_categories["build"]:
                        st.session_state.selected_error_categories["build"].append(category)
                else:
                    if category in st.session_state.selected_error_categories["build"]:
                        st.session_state.selected_error_categories["build"].remove(category)
                
                # Show expand/collapse control if there are errors
                if error_count > 0:
                    # Initialize expanded state for this category if it doesn't exist
                    if category_key not in st.session_state.expanded_categories:
                        st.session_state.expanded_categories[category_key] = False
                    
                    # Show expand/collapse button
                    if st.button(f"{'▼ Show' if not st.session_state.expanded_categories[category_key] else '▲ Hide'} specific errors", 
                               key=f"toggle_{category_key}"):
                        st.session_state.expanded_categories[category_key] = not st.session_state.expanded_categories[category_key]
                    
                    # Display subitems if expanded
                    if st.session_state.expanded_categories[category_key]:
                        st.markdown("<div class='subcategory-container'>", unsafe_allow_html=True)
                        # Display error subitems
                        for error in all_error_types["build"].get(category, []):
                            error_name = error.get("error_name", "Unknown error")
                            error_desc = error.get("description", "")
                            st.markdown(f"<div class='error-item'><b>{error_name}</b>: {error_desc}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Checkstyle errors
    st.markdown("<div class='error-type-header'>Checkstyle Errors</div>", unsafe_allow_html=True)
    checkstyle_categories = all_categories["checkstyle"]
    
    # Create a multi-column layout for checkstyle errors
    checkstyle_cols = st.columns(2)
    
    # Split the checkstyle categories into two columns
    half_length = len(checkstyle_categories) // 2
    for i, col in enumerate(checkstyle_cols):
        start_idx = i * half_length
        end_idx = start_idx + half_length if i == 0 else len(checkstyle_categories)
        
        with col:
            for category in checkstyle_categories[start_idx:end_idx]:
                # Show how many errors in this category
                error_count = len(all_error_types["checkstyle"].get(category, []))
                category_label = f"{category} ({error_count} errors)"
                
                # Create a unique key for this category
                category_key = f"checkstyle_{category}"
                
                # Check if category is selected
                is_selected = st.checkbox(category_label, key=category_key)
                
                # Update selection state
                if is_selected:
                    if category not in st.session_state.selected_error_categories["checkstyle"]:
                        st.session_state.selected_error_categories["checkstyle"].append(category)
                else:
                    if category in st.session_state.selected_error_categories["checkstyle"]:
                        st.session_state.selected_error_categories["checkstyle"].remove(category)
                
                # Show expand/collapse control if there are errors
                if error_count > 0:
                    # Initialize expanded state for this category if it doesn't exist
                    if category_key not in st.session_state.expanded_categories:
                        st.session_state.expanded_categories[category_key] = False
                    
                    # Show expand/collapse button
                    if st.button(f"{'▼ Show' if not st.session_state.expanded_categories[category_key] else '▲ Hide'} specific errors", 
                               key=f"toggle_{category_key}"):
                        st.session_state.expanded_categories[category_key] = not st.session_state.expanded_categories[category_key]
                    
                    # Display subitems if expanded
                    if st.session_state.expanded_categories[category_key]:
                        st.markdown("<div class='subcategory-container'>", unsafe_allow_html=True)
                        # Display error subitems
                        for error in all_error_types["checkstyle"].get(category, []):
                            error_name = error.get("check_name", "Unknown error")
                            error_desc = error.get("description", "")
                            st.markdown(f"<div class='error-item'><b>{error_name}</b>: {error_desc}</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    # Show selected categories
    build_selected = st.session_state.selected_error_categories["build"]
    checkstyle_selected = st.session_state.selected_error_categories["checkstyle"]
    
    st.write("### Selected Categories")
    if not build_selected and not checkstyle_selected:
        st.warning("No categories selected. Default categories will be used.")
    else:
        if build_selected:
            st.write("Build Error Categories:")
            for category in build_selected:
                st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)
        
        if checkstyle_selected:
            st.write("Checkstyle Error Categories:")
            for category in checkstyle_selected:
                st.markdown(f"<div class='error-category'>{category}</div>", unsafe_allow_html=True)

def display_results():
    """Display the analysis results and feedback."""
    # Display the comparison report
    st.subheader("Educational Feedback:")
    st.markdown(
        f'<div class="comparison-report">{st.session_state.comparison_report}</div>',
        unsafe_allow_html=True
    )
    
    # Show review history in an expander if there are multiple iterations
    if len(st.session_state.review_history) > 1:
        with st.expander("Review History", expanded=False):
            st.write("Your review attempts:")
            
            for i, review in enumerate(st.session_state.review_history):
                review_analysis = review.get("review_analysis", {})
                
                st.markdown(
                    f'<div class="review-history-item">'
                    f'<h4>Attempt {i+1}</h4>'
                    f'<p>Found {review_analysis.get("identified_count", 0)} of '
                    f'{review_analysis.get("total_problems", len(st.session_state.known_problems))} issues '
                    f'({review_analysis.get("accuracy_percentage", 0):.1f}% accuracy)</p>'
                    f'<details>'
                    f'<summary>View this review</summary>'
                    f'<pre>{review.get("student_review", "")}</pre>'
                    f'</details>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    # Display analysis details in an expander
    with st.expander("Detailed Analysis", expanded=False):
        # Display review summary
        st.subheader("Review Summary:")
        st.write(st.session_state.review_summary)
        
        # Display review analysis
        if st.session_state.review_analysis:
            st.subheader("Review Analysis:")
            accuracy = st.session_state.review_analysis.get("accuracy_percentage", 0)
            identified_percentage = st.session_state.review_analysis.get("identified_percentage", 0)
            
            st.write(f"**Accuracy:** {accuracy:.1f}%")
            st.write(f"**Problems Identified:** {identified_percentage:.1f}% of all issues")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Correctly Identified Issues:**")
                for issue in st.session_state.review_analysis.get("identified_problems", []):
                    st.write(f"✓ {issue}")
            
            with col2:
                st.write("**Missed Issues:**")
                for issue in st.session_state.review_analysis.get("missed_problems", []):
                    st.write(f"✗ {issue}")
            
            # Display false positives if any
            if st.session_state.review_analysis.get("false_positives"):
                st.write("**False Positives:**")
                for issue in st.session_state.review_analysis.get("false_positives"):
                    st.write(f"⚠ {issue}")
    
    # Start over button
    if st.button("Start New Review", type="primary"):
        # Reset session state
        for key in list(st.session_state.keys()):
            # Keep model configuration
            if key not in ["models_configured", "show_model_setup", "error_selection_mode"]:
                del st.session_state[key]
        
        # Initialize session state again
        initialize_session_state()
        
        # Rerun the app
        st.rerun()

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Java Code Review Training System")
    st.markdown("### Train your Java code review skills with AI-generated exercises")
    
    # Sidebar for model setup
    with st.sidebar:
        st.header("Model Settings")
        
        # Show status
        status = check_model_status()
        st.subheader("System Status")
        
        if status["ollama_running"]:
            st.markdown(f"- Ollama: <span class='status-ok'>Running</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- Ollama: <span class='status-error'>Not Running</span>", unsafe_allow_html=True)
            st.error("Ollama is not running. Please start it first.")
            
            # Troubleshooting information
            with st.expander("Troubleshooting"):
                st.markdown("""
                1. **Check if Ollama is running:**
                   ```bash
                   curl http://localhost:11434/api/tags
                   ```
                   
                2. **Make sure the model is downloaded:**
                   ```bash
                   ollama pull llama3:1b
                   ```
                   
                3. **Start Ollama:**
                   - On Linux/Mac: `ollama serve`
                   - On Windows: Start the Ollama application
                """)
        
        if status["default_model_available"]:
            st.markdown(f"- Default model: <span class='status-ok'>Available</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"- Default model: <span class='status-warning'>Not Found</span>", unsafe_allow_html=True)
            if status["ollama_running"]:
                st.warning(f"Default model '{llm_manager.default_model}' not found. You need to pull it.")
                if st.button("Pull Default Model"):
                    with st.spinner(f"Pulling {llm_manager.default_model}..."):
                        if llm_manager.download_ollama_model(llm_manager.default_model):
                            st.success("Default model pulled successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to pull default model.")
        
        if status["all_models_configured"]:
            st.markdown(f"- Model configuration: <span class='status-ok'>Complete</span>", unsafe_allow_html=True)
            st.session_state.models_configured = True
        else:
            st.markdown(f"- Model configuration: <span class='status-warning'>Incomplete</span>", unsafe_allow_html=True)
            
        # Show advanced model setup button
        if st.button("Advanced Model Setup"):
            st.session_state.show_model_setup = True
        
        # Sidebar section for iterative review settings
        st.markdown("---")
        st.header("Review Settings")
        
        max_iterations = st.slider(
            "Maximum Review Attempts:",
            min_value=1,
            max_value=5,
            value=st.session_state.max_iterations,
            help="Maximum number of review attempts allowed before final evaluation"
        )
        
        # Update max iterations in session state
        if max_iterations != st.session_state.max_iterations:
            st.session_state.max_iterations = max_iterations
    
    # Show model setup screen if needed
    if st.session_state.get("show_model_setup", False):
        model_configs = model_setup_ui.render_model_config_tabs()
        if st.button("Continue to Main Application"):
            st.session_state.show_model_setup = False
            st.rerun()
        return
    
    # Display error message if there's an error
    if st.session_state.error:
        st.error(f"Error: {st.session_state.error}")
        if st.button("Clear Error"):
            st.session_state.error = None
            st.rerun()
    
    # Create tabs for different steps of the workflow
    tabs = st.tabs(["1. Generate Code Problem", "2. Submit Review", "3. Analysis & Feedback"])
    
    # Set the active tab based on session state
    active_tab = st.session_state.active_tab
    
    # Update parameters in session state
    if "programming_language" not in st.session_state:
        st.session_state.programming_language = "Java"
    if "problem_areas" not in st.session_state:
        st.session_state.problem_areas = ["Style", "Logical", "Performance"]
    if "difficulty_level" not in st.session_state:
        st.session_state.difficulty_level = "Medium"
    if "code_length" not in st.session_state:
        st.session_state.code_length = "Medium"
    
    with tabs[0]:
        st.header("Generate Java Code Problem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Fixed to Java now
            st.info("This system is specialized for Java code review training.")
            st.session_state.programming_language = "Java"
            
            difficulty_level = st.select_slider(
                "Difficulty Level",
                options=["Easy", "Medium", "Hard"],
                value=st.session_state.difficulty_level,
                key="difficulty_level_select"
            )
            
            # Error selection mode toggle
            error_mode = st.radio(
                "Error Selection Mode",
                options=["Standard (by problem areas)", "Advanced (by specific error categories)"],
                index=0 if st.session_state.error_selection_mode == "standard" else 1,
                key="error_mode_select"
            )
            
            # Update error selection mode
            if "Standard" in error_mode and st.session_state.error_selection_mode != "standard":
                st.session_state.error_selection_mode = "standard"
                # Reset selected categories
                st.session_state.selected_error_categories = {"build": [], "checkstyle": []}
            elif "Advanced" in error_mode and st.session_state.error_selection_mode != "advanced":
                st.session_state.error_selection_mode = "advanced"
            
            # Update session state when widgets change
            st.session_state.difficulty_level = difficulty_level
        
        with col2:
            # Show standard problem area selection if in standard mode
            if st.session_state.error_selection_mode == "standard":
                problem_areas = st.multiselect(
                    "Problem Areas",
                    ["Style", "Logical", "Performance", "Security", "Design"],
                    default=st.session_state.problem_areas,
                    key="problem_areas_select"
                )
                
                code_length = st.select_slider(
                    "Code Length",
                    options=["Short", "Medium", "Long"],
                    value=st.session_state.code_length,
                    key="code_length_select"
                )
                
                # Update session state when widgets change
                st.session_state.problem_areas = problem_areas
                st.session_state.code_length = code_length
            else:
                # In advanced mode, code length is still needed
                code_length = st.select_slider(
                    "Code Length",
                    options=["Short", "Medium", "Long"],
                    value=st.session_state.code_length,
                    key="code_length_select"
                )
                
                # Update session state
                st.session_state.code_length = code_length
                # Use empty list for problem areas as we'll be using specific categories
                st.session_state.problem_areas = []
        
        # Show advanced error category selection if in advanced mode
        if st.session_state.error_selection_mode == "advanced":
            display_error_category_selection()
        
        generate_button = st.button(
            "Generate Java Code Problem", 
            type="primary", 
            disabled=not st.session_state.models_configured
        )
        
        if not st.session_state.models_configured:
            st.warning("Model configuration is incomplete. Use Advanced Model Setup in the sidebar.")
        
        if generate_button:
            with st.spinner("Generating Java code problem with intentional issues..."):
                # Pass selected error categories if in advanced mode
                if st.session_state.error_selection_mode == "advanced":
                    success = generate_problem(
                        "Java",  # Now fixed to Java
                        st.session_state.problem_areas,
                        difficulty_level,
                        code_length,
                        st.session_state.selected_error_categories
                    )
                else:
                    success = generate_problem(
                        "Java",  # Now fixed to Java
                        problem_areas,
                        difficulty_level,
                        code_length
                    )
                
                if success:
                    st.rerun()
        
        # Display existing code if available
        if st.session_state.code_snippet:
            st.subheader("Generated Java Code with Intentional Problems:")
            # Add line numbers to the code snippet
            numbered_code = add_line_numbers(st.session_state.code_snippet)
            st.code(numbered_code, language="java")
            
            # IMPORTANT: In a real application, we would NOT show the known problems
            # We're showing them here just for demonstration purposes
            if st.checkbox("Show Known Problems (Instructor View)", value=False):
                st.subheader("Known Problems:")
                for i, problem in enumerate(st.session_state.known_problems, 1):
                    st.markdown(f"{i}. {problem}")
    
    with tabs[1]:
        # Show iteration badge if not the first iteration
        if st.session_state.iteration_count > 1:
            st.header(
                f"Submit Your Code Review "
                f"<span class='iteration-badge'>Attempt {st.session_state.iteration_count} of "
                f"{st.session_state.max_iterations}</span>", 
                unsafe_allow_html=True
            )
        else:
            st.header("Submit Your Code Review")
        
        if not st.session_state.code_snippet:
            st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        else:
            # Display targeted guidance if available (for iterations after the first)
            if st.session_state.targeted_guidance and st.session_state.iteration_count > 1:
                st.markdown(
                    f'<div class="guidance-box">'
                    f'<h4>Review Guidance</h4>'
                    f'{st.session_state.targeted_guidance}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f'<div class="warning-box">'
                    f'<h4>Previous Attempt Results</h4>'
                    f'You identified {st.session_state.review_analysis.get("identified_count", 0)} of '
                    f'{st.session_state.review_analysis.get("total_problems", 0)} issues '
                    f'({st.session_state.review_analysis.get("identified_percentage", 0):.1f}%). '
                    f'Can you find more issues in this attempt?'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Display the code for review
            st.subheader("Java Code to Review:")
            # Add line numbers to the code snippet
            numbered_code = add_line_numbers(st.session_state.code_snippet)
            st.code(numbered_code, language="java")
            
            # Student review input
            st.subheader("Your Review:")
            st.write("Please review the code above and identify any issues or problems:")
            
            # If not the first iteration, clear the previous review text
            if st.session_state.iteration_count > 1 and st.session_state.student_review:
                # Save the previous review but clear the input for a new review
                prev_review = st.session_state.student_review
                if 'previous_reviews' not in st.session_state:
                    st.session_state.previous_reviews = []
                st.session_state.previous_reviews.append(prev_review)
                st.session_state.student_review = ""
            
            # Get or update the student review
            student_review = st.text_area(
                "Enter your review comments here",
                value=st.session_state.student_review,
                height=200,
                key="student_review_input"
            )
            
            # Update session state when the text area changes
            if student_review != st.session_state.student_review:
                st.session_state.student_review = student_review
            
            # Show previous reviews if available
            if hasattr(st.session_state, 'previous_reviews') and st.session_state.previous_reviews:
                with st.expander("View your previous review attempts", expanded=False):
                    for i, prev_review in enumerate(st.session_state.previous_reviews):
                        st.markdown(f"**Attempt {i+1}:**")
                        st.text(prev_review)
            
            # Submit button
            submit_text = "Submit Review" if st.session_state.iteration_count == 1 else f"Submit Review (Attempt {st.session_state.iteration_count} of {st.session_state.max_iterations})"
            
            if st.button(submit_text, type="primary"):
                if not st.session_state.student_review.strip():
                    st.warning("Please enter your review before submitting.")
                else:
                    st.session_state.current_step = "analyze"
                    # Analyze the review
                    with st.spinner("Analyzing your review..."):
                        success = analyze_review()
                        if success:
                            st.rerun()
    
    with tabs[2]:
        st.header("Analysis & Feedback")
        
        if not st.session_state.comparison_report:
            st.info("Please submit your review in the 'Submit Review' tab first.")
        else:
            # Display results
            display_results()

if __name__ == "__main__":
    main()