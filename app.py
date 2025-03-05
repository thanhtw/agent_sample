import streamlit as st
import os
import json
from agent import run_peer_review_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Peer Code Review Training System",
    page_icon="👨‍💻",
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
    </style>
""", unsafe_allow_html=True)

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

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("Peer Code Review Training System")
    st.markdown("### Train your code review skills with AI-generated exercises")
    
    # Check Ollama connection
    with st.sidebar:
        st.header("Ollama Settings")
        ollama_url = st.text_input("Ollama Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        model_name = st.text_input("Model Name", value=os.getenv("DEFAULT_MODEL", "llama3.2:1b"))
        
        if st.button("Test Ollama Connection"):
            import requests
            try:
                response = requests.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    st.success(f"Successfully connected to Ollama at {ollama_url}")
                    models = response.json().get("models", [])
                    if models:
                        st.write("Available models:")
                        for model in models:
                            st.write(f"- {model.get('name')}")
                    else:
                        st.warning("No models found. You may need to pull the llama3:21b model.")
                else:
                    st.error(f"Failed to connect to Ollama. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to Ollama: {str(e)}")
                st.info("Make sure Ollama is running and accessible at the specified URL.")
    
    # Create tabs for different steps of the workflow
    tab1, tab2, tab3 = st.tabs(["1. Generate Code Problem", "2. Submit Review", "3. Analysis & Feedback"])
    
    with tab1:
        st.header("Generate Code Problem")
        
        col1, col2 = st.columns(2)
        
        with col1:
            programming_language = st.selectbox(
                "Programming Language",
                ["Python", "JavaScript", "Java"],
                index=0
            )
            
            difficulty_level = st.select_slider(
                "Difficulty Level",
                options=["Easy", "Medium", "Hard"],
                value="Medium"
            )
        
        with col2:
            problem_areas = st.multiselect(
                "Problem Areas",
                ["Style", "Logical", "Performance", "Security", "Design"],
                default=["Style", "Logical", "Performance"]
            )
            
            code_length = st.select_slider(
                "Code Length",
                options=["Short", "Medium", "Long"],
                value="Medium"
            )
        
        if st.button("Generate Code Problem", type="primary"):
            with st.spinner("Generating code problem with intentional issues..."):
                try:
                    # Convert inputs to lowercase for the backend
                    result = run_peer_review_agent(
                        programming_language=programming_language.lower(),
                        problem_areas=[area.lower() for area in problem_areas],
                        difficulty_level=difficulty_level.lower(),
                        code_length=code_length.lower()
                    )
                    
                    # Update session state
                    st.session_state.code_snippet = result["code_snippet"]
                    st.session_state.known_problems = result["known_problems"]
                    st.session_state.current_step = "review"
                    st.session_state.error = None
                    
                    # Jump to the review tab
                    st.rerun()
                except Exception as e:
                    st.session_state.error = str(e)
                    st.error(f"Error generating code problem: {str(e)}")
                    
                    # Display troubleshooting instructions
                    st.markdown("""
                    ### Troubleshooting
                    
                    1. **Check if Ollama is running:**
                       ```bash
                       curl http://localhost:11434/api/tags
                       ```
                       
                    2. **Make sure the model is downloaded:**
                       ```bash
                       ollama pull llama3:21b
                       ```
                       
                    3. **Check Ollama logs for errors:**
                       - On Linux/Mac: `journalctl -u ollama`
                       - On Windows: Check the terminal where Ollama is running
                    """)
        
        # Display existing code if available
        if st.session_state.code_snippet:
            st.subheader("Generated Code with Intentional Problems:")
            st.code(st.session_state.code_snippet, language=programming_language.lower())
            
            # IMPORTANT: In a real application, we would NOT show the known problems
            # We're showing them here just for demonstration purposes
            if st.checkbox("Show Known Problems (Instructor View)", value=False):
                st.subheader("Known Problems:")
                for i, problem in enumerate(st.session_state.known_problems, 1):
                    st.markdown(f"{i}. {problem}")
    
    with tab2:
        st.header("Submit Your Code Review")
        
        if not st.session_state.code_snippet:
            st.info("Please generate a code problem first in the 'Generate Code Problem' tab.")
        else:
            # Display the code for review
            st.subheader("Code to Review:")
            st.code(st.session_state.code_snippet, language=programming_language.lower() if 'programming_language' in locals() else "python")
            
            # Student review input
            st.subheader("Your Review:")
            st.write("Please review the code above and identify any issues or problems:")
            
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
            
            # Submit button
            if st.button("Submit Review", type="primary"):
                if not st.session_state.student_review.strip():
                    st.warning("Please enter your review before submitting.")
                else:
                    st.session_state.current_step = "analyze"
                    # Jump to the analysis tab
                    st.rerun()
    
    with tab3:
        st.header("Analysis & Feedback")
        
        if st.session_state.current_step != "analyze":
            st.info("Please submit your review in the 'Submit Review' tab first.")
        else:
            if st.session_state.comparison_report:
                # Display existing results
                display_results()
            else:
                # Perform analysis
                with st.spinner("Analyzing your review..."):
                    try:
                        # Run the complete workflow with the student review
                        result = run_peer_review_agent(
                            student_review=st.session_state.student_review,
                            # Use the same parameters as before
                            programming_language=programming_language.lower() if 'programming_language' in locals() else "python",
                            problem_areas=[area.lower() for area in problem_areas] if 'problem_areas' in locals() else ["style", "logical", "performance"],
                            difficulty_level=difficulty_level.lower() if 'difficulty_level' in locals() else "medium",
                            code_length=code_length.lower() if 'code_length' in locals() else "medium"
                        )
                        
                        # Update session state
                        st.session_state.review_analysis = result["review_analysis"]
                        st.session_state.review_summary = result["review_summary"]
                        st.session_state.comparison_report = result["comparison_report"]
                        st.session_state.error = None
                        
                        # Display results
                        display_results()
                    except Exception as e:
                        st.session_state.error = str(e)
                        st.error(f"Error analyzing review: {str(e)}")

def display_results():
    """Display the analysis results and feedback."""
    # Display the comparison report
    st.subheader("Educational Feedback:")
    st.markdown(
        f'<div class="comparison-report">{st.session_state.comparison_report}</div>',
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
            st.write(f"**Accuracy:** {accuracy:.1f}%")
            
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
    if st.button("Start Over", type="primary"):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Initialize session state again
        initialize_session_state()
        
        # Rerun the app
        st.rerun()

if __name__ == "__main__":
    main()