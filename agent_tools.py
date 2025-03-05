"""
Agent tools for peer code review system.

This module provides the core tools for generating code problems, analyzing student 
reviews, summarizing review comments, and providing educational feedback.
"""

import random
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

# Import the templates from prompts
from agent_prompts import (
    GENERATE_CODE_PROBLEM_TEMPLATE,
    ANALYZE_STUDENT_REVIEW_TEMPLATE,
    SUMMARIZE_REVIEW_COMMENTS_TEMPLATE,
    COMPARE_AND_EXPLAIN_TEMPLATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common programming problems by category
PROGRAMMING_PROBLEMS = {
    "style": [
        "Inconsistent naming conventions",
        "Poor indentation or formatting",
        "Lack of comments or documentation",
        "Overly complex expressions",
        "Code duplication",
        "Unused imports or variables",
        "Inconsistent spacing around operators"
    ],
    "logical": [
        "Off-by-one errors",
        "Incorrect boundary conditions",
        "Missing edge case handling",
        "Incorrect boolean logic",
        "Infinite loops",
        "Race conditions",
        "Incorrect algorithm implementation"
    ],
    "performance": [
        "Inefficient algorithms (e.g., O(n²) when O(n) is possible)",
        "Unnecessary computation",
        "Memory leaks",
        "Inefficient data structure choices",
        "Redundant operations",
        "Excessive memory usage"
    ],
    "security": [
        "SQL injection vulnerabilities",
        "Unsanitized user input",
        "Hardcoded credentials",
        "Cross-site scripting (XSS) vulnerabilities",
        "Improper error handling exposing sensitive information",
        "Weak encryption algorithms"
    ],
    "design": [
        "Poor separation of concerns",
        "Violation of SOLID principles",
        "Excessive coupling between components",
        "Lack of abstraction",
        "Inappropriate use of design patterns",
        "Monolithic function design"
    ]
}

# Example code templates by language
CODE_TEMPLATES = {
    "python": {
        "data_processing": """
def process_data(data_list):
    \"\"\"
    Process a list of data items and return the results.
    
    Args:
        data_list (list): A list of data items to process
        
    Returns:
        list: Processed data items
    \"\"\"
    results = []
    for item in data_list:
        # Process each item
        processed_item = item * 2  # Example processing
        results.append(processed_item)
    return results
        """,
        "search_algorithm": """
def search_item(item_list, target):
    \"\"\"
    Search for a target item in a list.
    
    Args:
        item_list (list): List of items to search through
        target: The item to find
        
    Returns:
        int: Index of the target item, or -1 if not found
    \"\"\"
    for i in range(len(item_list)):
        if item_list[i] == target:
            return i
    return -1
        """,
        "web_service": """
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/users', methods=['GET'])
def get_users():
    # Simulate database retrieval
    users = [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"}
    ]
    return jsonify(users)

if __name__ == '__main__':
    app.run(debug=True)
        """
    },
    "javascript": {
        "async_function": """
async function fetchUserData(userId) {
  try {
    const response = await fetch(`https://api.example.com/users/${userId}`);
    if (!response.ok) {
      throw new Error('Failed to fetch user data');
    }
    const userData = await response.json();
    return userData;
  } catch (error) {
    console.error('Error fetching user data:', error);
    return null;
  }
}
        """,
        "event_handler": """
function setupEventListeners() {
  const button = document.getElementById('submit-button');
  const form = document.getElementById('user-form');
  
  button.addEventListener('click', (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const userData = Object.fromEntries(formData.entries());
    console.log('Submitting user data:', userData);
    // Submit the form data
  });
}
        """
    },
    "java": {
        "class_definition": """
public class UserManager {
    private List<User> users;
    
    public UserManager() {
        this.users = new ArrayList<>();
    }
    
    public void addUser(User user) {
        this.users.add(user);
    }
    
    public User findUserById(int id) {
        for (User user : users) {
            if (user.getId() == id) {
                return user;
            }
        }
        return null;
    }
}
        """
    }
}

class CodeProblemResult(BaseModel):
    """Output model for code problem generation."""
    code_snippet: str = Field(description="The generated code snippet with intentional problems")
    problems: List[str] = Field(description="List of intentional problems in the code")

class ReviewAnalysisResult(BaseModel):
    """Output model for review analysis."""
    identified_problems: List[str] = Field(description="Problems correctly identified by student")
    missed_problems: List[str] = Field(description="Problems missed by student")
    false_positives: List[str] = Field(description="Issues incorrectly identified as problems")
    accuracy_percentage: float = Field(description="Percentage of actual problems correctly identified")
    feedback: str = Field(description="General feedback on the review quality")

def extract_json_from_text(text: str) -> Dict:
    """
    Extract JSON data from a text that may contain other content.
    
    Args:
        text (str): Text containing JSON data
        
    Returns:
        Dict: Extracted JSON data or error dictionary
    """
    # Try to find JSON block with regex
    json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find any JSON-like structure
        json_match = re.search(r'({.*})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Last resort: assume the whole text might be JSON
            json_str = text
    
    # Try to parse the JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Could not parse JSON response: {e}")
        return {"error": "Could not parse JSON response"}

def extract_code_and_problems(text: str) -> Tuple[str, List[str]]:
    """
    Extract code and problems from the LLM response.
    
    Args:
        text (str): LLM response text
        
    Returns:
        Tuple[str, List[str]]: Extracted code snippet and list of problems
    """
    # Try to extract as JSON first
    try:
        json_data = extract_json_from_text(text)
        if "code_snippet" in json_data and "problems" in json_data:
            return json_data["code_snippet"], json_data["problems"]
    except Exception as e:
        logger.warning(f"Error extracting JSON: {e}")
    
    # Fallback: extract code block and try to find problem list
    code_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', text, re.DOTALL)
    code_snippet = code_match.group(1) if code_match else ""
    
    # Try to find problems section
    problems = []
    # Fix: Simplified regex pattern and corrected the search function call
    problems_section = re.search(r'problems?:?\s*(?:\n|:)(.*?)(?:\n\n|\n#|\Z)', text, re.DOTALL)
    
    if problems_section:
        problems_text = problems_section.group(1)
        # Look for numbered or bulleted list items
        problem_items = re.findall(r'(?:^|\n)\s*(?:\d+[.)]|-)\s*(.*?)(?=\n\s*(?:\d+[.)]|-)|$)', problems_text, re.DOTALL)
        if problem_items:
            problems = [p.strip() for p in problem_items if p.strip()]
    
    return code_snippet, problems

def generate_code_problem(
    programming_language: str,
    problem_areas: List[str], 
    difficulty_level: str,
    code_length: str,
    llm: BaseLanguageModel
) -> Tuple[str, List[str]]:
    """
    Generate a code snippet with intentional problems for review.
    
    Args:
        programming_language: The programming language to use
        problem_areas: Areas to include problems from
        difficulty_level: Difficulty level of the problems
        code_length: Approximate length of code
        llm: Language model to use
        
    Returns:
        Tuple of (code_snippet, list_of_problems)
    """
    # Select a few random problem types from each requested area
    selected_problems = []
    for area in problem_areas:
        if area in PROGRAMMING_PROBLEMS:
            # Select 1-3 problems from this area based on difficulty
            num_problems = {
                "easy": 1,
                "medium": 2,
                "hard": 3
            }.get(difficulty_level, 2)
            
            area_problems = random.sample(
                PROGRAMMING_PROBLEMS[area], 
                min(num_problems, len(PROGRAMMING_PROBLEMS[area]))
            )
            selected_problems.extend(area_problems)
    
    # Ensure we have at least some problems
    if not selected_problems:
        selected_problems = ["Missing input validation", "Inefficient algorithm"]
    
    # Choose an appropriate code template based on language
    template_options = CODE_TEMPLATES.get(programming_language.lower(), CODE_TEMPLATES["python"])
    template_key = random.choice(list(template_options.keys()))
    code_template = template_options[template_key]
    
    # Prepare the variables for the prompt
    problem_areas_str = ", ".join(problem_areas)
    problems_to_introduce = "\n".join([f"- {p}" for p in selected_problems])
    
    # Create and format the prompt
    prompt = PromptTemplate.from_template(GENERATE_CODE_PROBLEM_TEMPLATE)
    
    prompt_text = prompt.format(
        programming_language=programming_language,
        problem_areas=problem_areas_str,
        difficulty_level=difficulty_level,
        code_length=code_length,
        template=code_template,
        problems_to_introduce=problems_to_introduce
    )
    
    # Run the LLM
    logger.info(f"Generating code problem in {programming_language} with {len(selected_problems)} problems")
    response = llm.invoke(prompt_text)
    
    # Extract code and problems from the response
    code_snippet, problems = extract_code_and_problems(response)
    
    # If we couldn't extract problems, use the selected ones
    if not problems:
        logger.warning("Could not extract problems from LLM response, using selected problems")
        problems = selected_problems
        
    # If we couldn't extract code, use the template
    if not code_snippet:
        logger.warning("Could not extract code from LLM response, using template")
        code_snippet = code_template
    
    logger.info(f"Generated code problem with {len(problems)} problems")
    return code_snippet, problems

def analyze_student_review(
    code_snippet: str,
    known_problems: List[str],
    student_review: str,
    llm: BaseLanguageModel
) -> Dict:
    """
    Analyze how well a student's review identified known problems.
    
    Args:
        code_snippet: The original code snippet
        known_problems: List of known problems in the code
        student_review: The student's review comments
        llm: Language model to use
        
    Returns:
        Dictionary with analysis results
    """
    # Create and format the prompt
    prompt = PromptTemplate.from_template(ANALYZE_STUDENT_REVIEW_TEMPLATE)
    
    prompt_text = prompt.format(
        code_snippet=code_snippet,
        known_problems="\n".join([f"- {p}" for p in known_problems]),
        student_review=student_review
    )
    
    # Run the LLM
    logger.info("Analyzing student review")
    response = llm.invoke(prompt_text)
    
    # Try to extract JSON data
    try:
        analysis_data = extract_json_from_text(response)
        
        # Extract required fields with fallbacks
        return {
            "identified_problems": analysis_data.get("identified_problems", []),
            "missed_problems": analysis_data.get("missed_problems", []),
            "false_positives": analysis_data.get("false_positives", []),
            "accuracy_percentage": float(analysis_data.get("accuracy_percentage", 50.0)),
            "feedback": analysis_data.get("feedback", "The analysis was partially completed.")
        }
    except Exception as e:
        logger.warning(f"Error parsing analysis JSON: {e}, falling back to regex extraction")
        # If parsing fails, return a simplified analysis
        identified = []
        missed = []
        
        # Try to extract identified and missed problems with regex
        identified_match = re.search(r'identified(?:\s+problems)?(?:\s*:|\s*-)(.*?)(?=missed|false|$)', response, re.IGNORECASE | re.DOTALL)
        if identified_match:
            identified_text = identified_match.group(1)
            identified = re.findall(r'(?:^|\n)\s*(?:-|\d+[.)])\s*(.*?)(?=\n\s*(?:-|\d+[.)])|$)', identified_text, re.DOTALL)
            identified = [i.strip() for i in identified if i.strip()]
        
        missed_match = re.search(r'missed(?:\s+problems)?(?:\s*:|\s*-)(.*?)(?=identified|false|$)', response, re.IGNORECASE | re.DOTALL)
        if missed_match:
            missed_text = missed_match.group(1)
            missed = re.findall(r'(?:^|\n)\s*(?:-|\d+[.)])\s*(.*?)(?=\n\s*(?:-|\d+[.)])|$)', missed_text, re.DOTALL)
            missed = [m.strip() for m in missed if m.strip()]
        
        # Calculate a basic accuracy
        if not identified and not missed:
            accuracy = 50.0
        else:
            accuracy = len(identified) / max(1, len(identified) + len(missed)) * 100
                
        return {
            "identified_problems": identified or ["Some problems were identified"],
            "missed_problems": missed or ["Some problems may have been missed"],
            "false_positives": [],
            "accuracy_percentage": accuracy,
            "feedback": "The student has partially identified the issues in the code."
        }

def summarize_review_comments(
    review_comments: str,
    llm: BaseLanguageModel
) -> str:
    """
    Summarize review comments into a concise summary.
    
    Args:
        review_comments: The full review comments
        llm: Language model to use
        
    Returns:
        Summarized review comments
    """
    # Create and format the prompt
    prompt = PromptTemplate.from_template(SUMMARIZE_REVIEW_COMMENTS_TEMPLATE)
    
    prompt_text = prompt.format(
        review_comments=review_comments
    )
    
    # Run the LLM
    logger.info("Generating review summary")
    summary = llm.invoke(prompt_text)
    
    return summary

def compare_and_explain(
    code_snippet: str,
    known_problems: List[str],
    student_review: str,
    review_analysis: Dict,
    review_summary: str,
    llm: BaseLanguageModel
) -> str:
    """
    Compare student review with known problems and provide educational feedback.
    
    Args:
        code_snippet: The original code snippet
        known_problems: List of known problems in the code
        student_review: The student's review comments
        review_analysis: Analysis of the student review
        review_summary: Summary of the review comments
        llm: Language model to use
        
    Returns:
        Educational feedback explaining the comparison
    """
    # Create and format the prompt
    prompt = PromptTemplate.from_template(COMPARE_AND_EXPLAIN_TEMPLATE)
    
    prompt_text = prompt.format(
        code_snippet=code_snippet,
        known_problems="\n".join([f"- {p}" for p in known_problems]),
        student_review=student_review,
        identified_problems="\n".join([f"- {p}" for p in review_analysis.get("identified_problems", [])]),
        missed_problems="\n".join([f"- {p}" for p in review_analysis.get("missed_problems", [])]),
        false_positives="\n".join([f"- {p}" for p in review_analysis.get("false_positives", [])]),
        accuracy_percentage=review_analysis.get("accuracy_percentage", 0),
        review_summary=review_summary
    )
    
    # Run the LLM
    logger.info("Generating comparison report")
    comparison = llm.invoke(prompt_text)
    
    return comparison