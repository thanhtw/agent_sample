"""
Enhanced agent tools for Java peer code review system.

This module provides the core tools for generating Java code problems, 
analyzing student reviews, summarizing review comments, and providing 
educational feedback with iterative review support.
"""

import random
import re
import json
import logging
import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

# Import the templates from updated prompts
from agent_prompts import (
    JAVA_GENERATIVE_AGENT_PROMPT,
    JAVA_REVIEW_AGENT_PROMPT,
    JAVA_SUMMARY_AGENT_PROMPT,
    JAVA_COMPARE_EXPLAIN_AGENT_PROMPT,
    JAVA_GENERATE_CODE_PROBLEM_TEMPLATE,
    TARGETED_GUIDANCE_TEMPLATE,
    ITERATIVE_REVIEW_FEEDBACK_TEMPLATE
)

# Import Java error handler
from java_error_handler import (
    select_java_template,
    select_java_errors_by_categories,
    format_java_error_description,
    get_error_injection_instructions,
    get_all_error_categories
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Original code templates (keeping for backwards compatibility)
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

# Common programming problems by category (keeping for backwards compatibility)
PROGRAMMING_PROBLEMS = {
    "style": [
        "Inconsistent naming conventions",
        "Poor indentation or formatting",
        "Lack of comments or documentation",
        "Overly complex expressions",
        "Code duplication",
        "Unused imports or variables"
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

class CodeProblemResult:
    """Output model for code problem generation."""
    
    def __init__(self, code_snippet, problems, raw_errors=None):
        self.code_snippet = code_snippet  # The generated code snippet with intentional problems
        self.problems = problems  # List of intentional problems in the code
        self.raw_errors = raw_errors  # Original error data used for generation
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "code_snippet": self.code_snippet,
            "problems": self.problems,
            "raw_errors": self.raw_errors
        }

class ReviewAnalysisResult:
    """Output model for review analysis."""
    
    def __init__(self, identified_problems, missed_problems, false_positives, 
                 accuracy_percentage, review_sufficient, feedback):
        self.identified_problems = identified_problems  # Problems correctly identified by student
        self.missed_problems = missed_problems  # Problems missed by student
        self.false_positives = false_positives  # Issues incorrectly identified as problems
        self.accuracy_percentage = accuracy_percentage  # Percentage of actual problems correctly identified
        self.review_sufficient = review_sufficient  # Whether the review found enough problems
        self.feedback = feedback  # General feedback on the review quality
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "identified_problems": self.identified_problems,
            "missed_problems": self.missed_problems,
            "false_positives": self.false_positives,
            "accuracy_percentage": self.accuracy_percentage,
            "review_sufficient": self.review_sufficient,
            "feedback": self.feedback
        }

def extract_json_from_text(text: str) -> Dict:
    """
    Extract JSON data from a text that may contain other content.
    
    Args:
        text (str): Text containing JSON data
        
    Returns:
        Dict: Extracted JSON data or error dictionary
    """
    # Try to find JSON block with regex - handling different markdown code block styles
    patterns = [
        r'```json\s*([\s\S]*?)```',  # JSON code block
        r'```\s*({[\s\S]*?})\s*```',  # Any JSON object in code block
        r'({[\s\S]*"code_snippet"[\s\S]*"problems"[\s\S]*})',  # Look for our expected fields
        r'({[\s\S]*})',  # Any JSON-like structure
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                json_str = match.strip()
                # Try to parse as JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # If standard methods fail, try to manually construct a JSON object
    try:
        # Look for code block
        code_block_match = re.search(r'```(?:\w+)?\s*(.*?)\s*```', text, re.DOTALL)
        code_snippet = code_block_match.group(1) if code_block_match else ""
        
        # Look for problems section - different patterns
        problems = []
        # Try numbered list pattern
        problems_list_match = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', text, re.DOTALL)
        if problems_list_match:
            problems = [p.strip() for p in problems_list_match if p.strip()]
        
        # Try bullet list pattern if numbered list didn't work
        if not problems:
            problems_list_match = re.findall(r'[-*•]\s+(.*?)(?=[-*•]|$)', text, re.DOTALL)
            if problems_list_match:
                problems = [p.strip() for p in problems_list_match if p.strip()]
        
        # Construct a valid JSON object
        if code_snippet:
            return {
                "code_snippet": code_snippet,
                "problems": problems
            }
    except Exception as e:
        logger.warning(f"Manual JSON construction failed: {e}")
    
    # Last resort - return an error object with the raw text
    logger.warning("Could not extract JSON, returning error object")
    return {
        "error": "Could not parse JSON response",
        "raw_text": text[:500] + ("..." if len(text) > 500 else "")  # Include truncated raw text for debugging
    }

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
        if "code_snippet" in json_data and "problems" in json_data and not "error" in json_data:
            return json_data["code_snippet"], json_data["problems"]
    except Exception as e:
        logger.warning(f"Error extracting JSON: {e}")
    
    # Fallback: extract code block and try to find problem list
    code_snippet = ""
    problems = []
    
    # Extract code block
    code_block_matches = re.findall(r'```(?:java)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_matches:
        # Use the largest code block
        code_snippet = max(code_block_matches, key=len)
    
    # Try to find problems section in various formats
    problem_patterns = [
        r'(?:Problems|Issues|Errors):\s*((?:\d+\.\s*.*?(?:\n|$))+)',  # Numbered list with header
        r'(?:problems|issues|errors):\s*((?:[-*•]\s*.*?(?:\n|$))+)',  # Bullet list with header
        r'(?:^|\n)(\d+\.\s*.*?)(?=\n\d+\.|\n\n|\Z)',  # Numbered list items
        r'(?:^|\n)([-*•]\s*.*?)(?=\n[-*•]|\n\n|\Z)',  # Bullet list items
    ]
    
    for pattern in problem_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                # For patterns with headers, extract individual items
                if ":" in pattern:
                    if r'\d+' in pattern:  # Numbered list
                        items = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|\Z)', match, re.DOTALL)
                    else:  # Bullet list
                        items = re.findall(r'[-*•]\s*(.*?)(?=\n[-*•]|\n\n|\Z)', match, re.DOTALL)
                    
                    for item in items:
                        if item.strip():
                            problems.append(item.strip())
                else:
                    # Remove the numbering/bullet
                    problem = re.sub(r'^\d+\.\s*|^[-*•]\s*', '', match)
                    if problem.strip():
                        problems.append(problem.strip())
            if problems:
                break
    
    return code_snippet, problems

def generate_code_problem(
    programming_language: str,
    problem_areas: List[str], 
    difficulty_level: str,
    code_length: str,
    llm: BaseLanguageModel,
    specific_error_categories: Dict[str, List[str]] = None
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Generate a code snippet with intentional problems for review.
    Now with enhanced Java support and specific error category selection.
    
    Args:
        programming_language: The programming language to use
        problem_areas: Areas to include problems from
        difficulty_level: Difficulty level of the problems
        code_length: Approximate length of code
        llm: Language model to use
        specific_error_categories: Optional dictionary with 'build' and 'checkstyle' keys containing lists of specific error categories
        
    Returns:
        Tuple of (code_snippet, list_of_problems, raw_errors)
    """
    # Special handling for Java
    if programming_language.lower() == "java":
        return generate_java_code_problem(
            problem_areas, 
            difficulty_level, 
            code_length, 
            llm, 
            specific_error_categories
        )
    
    # For other languages, use the original implementation
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
    
    # Create and format the prompt for non-Java languages
    prompt = PromptTemplate.from_template("""
Please create a realistic code snippet in {programming_language} that contains intentional problems for a code review exercise.

The code should include problems related to: {problem_areas}
The difficulty level should be: {difficulty_level} 
The approximate length should be: {code_length}

You can use this template as a starting point:
```
{template}
```

Please introduce the following problems into the code:
{problems_to_introduce}

Return your response in this format:
```json
{{
  "code_snippet": "Your code here",
  "problems": [
    "Description of problem 1",
    "Description of problem 2",
    ...
  ]
}}
```

Remember to maintain the basic functionality while introducing these problems in a subtle way that's educational for students to identify.
""")
    
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
    # For non-Java languages, we don't have raw error data
    return code_snippet, problems, []

def generate_java_code_problem(
    problem_areas: List[str],
    difficulty_level: str,
    code_length: str,
    llm: BaseLanguageModel,
    specific_error_categories: Dict[str, List[str]] = None
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    """
    Generate a Java code snippet with intentional problems using errors from
    build_errors.json and checkstyle_error.json.
    
    Args:
        problem_areas: Areas to include problems from
        difficulty_level: Difficulty level of the problems
        code_length: Approximate length of code
        llm: Language model to use
        specific_error_categories: Optional dictionary with 'build' and 'checkstyle' keys containing lists of specific error categories
        
    Returns:
        Tuple of (code_snippet, list_of_problems, raw_errors)
    """
    # Get a Java template
    java_template = select_java_template(code_length)
    
    # Select Java-specific errors based on specific categories or problem areas
    if specific_error_categories:
        selected_errors = select_java_errors_by_categories(specific_error_categories, difficulty_level)
    else:
        # Create category mapping from problem areas
        selected_categories = {"build": [], "checkstyle": []}
        
        # Map problem areas to Java error categories
        area_to_build_errors = {
            "logical": ["LogicalErrors"],
            "performance": ["RuntimeErrors"],
            "security": ["RuntimeErrors", "LogicalErrors"],
            "design": ["LogicalErrors"]
        }
        
        area_to_checkstyle_errors = {
            "style": ["NamingConventionChecks", "WhitespaceAndFormattingChecks", "JavadocChecks"],
            "logical": [],
            "performance": ["MetricsChecks"],
            "security": ["CodeQualityChecks"],
            "design": ["MiscellaneousChecks", "FileStructureChecks", "BlockChecks"]
        }
        
        # Populate selected categories
        for area in problem_areas:
            area = area.lower()
            if area in area_to_build_errors:
                selected_categories["build"].extend(area_to_build_errors[area])
            if area in area_to_checkstyle_errors:
                selected_categories["checkstyle"].extend(area_to_checkstyle_errors[area])
        
        # Remove duplicates
        selected_categories["build"] = list(set(selected_categories["build"]))
        selected_categories["checkstyle"] = list(set(selected_categories["checkstyle"]))
        
        # Select errors based on categories
        selected_errors = select_java_errors_by_categories(selected_categories, difficulty_level)
    
    # Generate error injection instructions
    error_injection_instructions = get_error_injection_instructions(selected_errors)
    
    # Format the problem descriptions
    problem_descriptions = [format_java_error_description(error) for error in selected_errors]
    
    # Create and format the prompt
    system_prompt = """You are an expert Java programming educator who creates code review exercises. You will:
1. Follow instructions exactly
2. Create realistic Java code with intentional problems
3. Return your response in valid JSON format with "code_snippet" and "problems" fields
4. Ensure the JSON is properly formatted with proper escaping of quotes and special characters
5. Include clear problem descriptions with their locations in the code"""
    
    prompt = """
Please create a realistic Java code snippet that contains intentional problems for a code review exercise.

Use this Java class template as a starting point:
```java
{template}
```

{error_injection_instructions}

Return your response in this specific JSON format (ensure proper JSON formatting):
```json
{{
  "code_snippet": "// Your Java code with intentional problems here",
  "problems": [
    "Description of problem 1 with location",
    "Description of problem 2 with location",
    "Description of problem 3 with location"
  ]
}}
```

Remember to maintain the basic functionality while introducing these problems in a subtle way that's educational for students to identify.
"""
    
    formatted_prompt = prompt.format(
        template=java_template,
        error_injection_instructions=error_injection_instructions
    )
    
    # Run the LLM with the Java-specific prompt
    logger.info(f"Generating Java code problem with {len(selected_errors)} specific errors")
    response = llm.invoke(system_prompt + "\n\n" + formatted_prompt)
    
    # Improved extraction of code and problems 
    json_data = extract_json_from_text(response)
    
    if "code_snippet" in json_data and "problems" in json_data and not "error" in json_data:
        code_snippet = json_data["code_snippet"]
        # Check if the code snippet is just the placeholder text and not real code
        if code_snippet.strip() == "// Your Java code with intentional problems here":
            # Get real code from regex extraction as fallback
            extracted_code, _ = extract_code_and_problems(response)
            if extracted_code:
                code_snippet = extracted_code
        problems = json_data["problems"]
    else:
        # Fallback to regex extraction
        code_snippet, problems = extract_code_and_problems(response)
    
    # If we couldn't extract enough problems from LLM response, use our predefined ones
    if not problems or len(problems) < len(selected_errors) // 2:
        logger.warning("Couldn't extract enough problems from LLM response, using predefined ones")
        problems = problem_descriptions
    
    # If we couldn't extract code or it's just the placeholder, use the template with injected errors
    if not code_snippet or code_snippet.strip() == "// Your Java code with intentional problems here":
        logger.warning("Could not extract valid code from LLM response, using template")
        code_snippet = java_template
    
    logger.info(f"Generated Java code problem with {len(problems)} problems")
    return code_snippet, problems, selected_errors

def analyze_student_review(
    code_snippet: str,
    known_problems: List[str],
    student_review: str,
    llm: BaseLanguageModel,
    min_identified_percentage: float = 60.0  # Students should identify at least 60% of problems
) -> Dict:
    """
    Analyze how well a student's review identified known problems.
    Enhanced with better error identification and review sufficiency check.
    
    Args:
        code_snippet: The original code snippet
        known_problems: List of known problems in the code
        student_review: The student's review comments
        llm: Language model to use
        min_identified_percentage: Minimum percentage of problems that should be identified
        
    Returns:
        Dictionary with analysis results
    """
    # Create and format the prompt
    system_prompt = """You are an expert code review analyzer. When analyzing student reviews:
1. Be thorough and accurate in your assessment
2. Return your analysis in valid JSON format with proper escaping
3. Provide constructive feedback that helps students improve
4. Be precise in identifying which problems were found and which were missed
5. Format your response as proper JSON"""
    
    prompt = """
Please analyze how well the student's review identifies the known problems in the code.

ORIGINAL CODE:
```java
{code_snippet}
```

KNOWN PROBLEMS IN THE CODE:
{known_problems}

STUDENT'S REVIEW:
```
{student_review}
```

Carefully analyze how thoroughly and accurately the student identified the known problems.

For each known problem, determine if the student correctly identified it, partially identified it, or missed it completely.
Consider semantic matches - students may use different wording but correctly identify the same issue.

Return your analysis in this exact JSON format:
```json
{{
  "identified_problems": ["Problem 1 they identified correctly", "Problem 2 they identified correctly"],
  "missed_problems": ["Problem 1 they missed", "Problem 2 they missed"],
  "false_positives": ["Non-issue 1 they incorrectly flagged", "Non-issue 2 they incorrectly flagged"],
  "accuracy_percentage": 75.0,
  "review_sufficient": true,
  "feedback": "Your general assessment of the review quality and advice for improvement"
}}
```

A review is considered "sufficient" if the student correctly identified at least {min_percentage}% of the known problems.
Be specific in your feedback about what types of issues they missed and how they can improve their code review skills.
"""
    
    formatted_prompt = prompt.format(
        code_snippet=code_snippet,
        known_problems="\n".join([f"- {p}" for p in known_problems]),
        student_review=student_review,
        min_percentage=min_identified_percentage
    )
    
    # Run the LLM
    logger.info("Analyzing student review")
    response = llm.invoke(system_prompt + "\n\n" + formatted_prompt)
    
    # Try to extract JSON data using our improved function
    analysis_data = extract_json_from_text(response)
    
    # Extract required fields with fallbacks
    identified_problems = analysis_data.get("identified_problems", [])
    missed_problems = analysis_data.get("missed_problems", [])
    false_positives = analysis_data.get("false_positives", [])
    
    try:
        accuracy_percentage = float(analysis_data.get("accuracy_percentage", 50.0))
    except (TypeError, ValueError):
        accuracy_percentage = 50.0
        
    feedback = analysis_data.get("feedback", "The analysis was partially completed.")
    
    # Determine if review is sufficient based on identified percentage
    identified_count = len(identified_problems)
    total_problems = len(known_problems)
    
    if total_problems > 0:
        identified_percentage = (identified_count / total_problems) * 100
    else:
        identified_percentage = 100.0
        
    # Check if model didn't provide review_sufficient field
    if "review_sufficient" not in analysis_data:
        review_sufficient = identified_percentage >= min_identified_percentage
    else:
        review_sufficient = analysis_data["review_sufficient"]
    
    # Provide more detailed feedback for insufficient reviews
    if not review_sufficient and feedback == "The analysis was partially completed.":
        if identified_percentage < 30:
            feedback = ("Your review missed most of the critical issues in the code. "
                        "Try to look more carefully for logic errors, style violations, "
                        "and potential runtime exceptions.")
        else:
            feedback = ("Your review found some issues but missed important problems. "
                        f"You identified {identified_percentage:.1f}% of the known issues. "
                        "Try to be more thorough in your next review.")
    
    return {
        "identified_problems": identified_problems,
        "missed_problems": missed_problems,
        "false_positives": false_positives,
        "accuracy_percentage": accuracy_percentage,
        "identified_percentage": identified_percentage,
        "identified_count": identified_count,
        "total_problems": total_problems,
        "review_sufficient": review_sufficient,
        "feedback": feedback
    }

def generate_targeted_guidance(
    code_snippet: str,
    known_problems: List[str],
    student_review: str,
    review_analysis: Dict,
    iteration_count: int,
    max_iterations: int,
    llm: BaseLanguageModel
) -> str:
    """
    Generate targeted guidance for missed errors to help with next review attempt.
    Enhanced with more detailed feedback.
    
    Args:
        code_snippet: The original code snippet
        known_problems: List of known problems in the code
        student_review: The student's review comments
        review_analysis: Analysis of the student review
        iteration_count: Current iteration number
        max_iterations: Maximum number of iterations
        llm: Language model to use
        
    Returns:
        Guidance text
    """
    # Prepare the prompt
    system_prompt = """You are an expert Java programming mentor who provides constructive feedback to students.
Your guidance is:
1. Encouraging and supportive
2. Specific and actionable
3. Educational - teaching students how to find issues rather than just telling them what to find
4. Focused on developing their review skills
5. Balanced - acknowledging what they did well while guiding them to improve"""
    
    prompt = """
Please create targeted guidance for a student who has reviewed Java code but missed some important errors.

ORIGINAL JAVA CODE:
```java
{code_snippet}
```

KNOWN PROBLEMS IN THE CODE:
{known_problems}

STUDENT'S REVIEW ATTEMPT #{iteration_count} of {max_iterations}:
```
{student_review}
```

PROBLEMS CORRECTLY IDENTIFIED BY THE STUDENT:
{identified_problems}

PROBLEMS MISSED BY THE STUDENT:
{missed_problems}

The student has identified {identified_count} out of {total_problems} issues ({identified_percentage:.1f}%).

Create constructive guidance that:
1. Acknowledges what the student found correctly with specific praise
2. Provides hints about the types of errors they missed (without directly listing them all)
3. Suggests specific areas of the code to examine more carefully
4. Encourages them to look for particular Java error patterns they may have overlooked
5. If there are false positives, gently explain why those are not actually issues
6. End with specific questions that might help the student find the missed problems

The guidance should be educational and help the student improve their Java code review skills.
Focus on teaching them how to identify the types of issues they missed.

Be encouraging but specific. Help the student develop a more comprehensive approach to code review.
"""
    
    formatted_prompt = prompt.format(
        code_snippet=code_snippet,
        known_problems="\n".join([f"- {p}" for p in known_problems]),
        student_review=student_review,
        identified_problems="\n".join([f"- {p}" for p in review_analysis.get("identified_problems", [])]),
        missed_problems="\n".join([f"- {p}" for p in review_analysis.get("missed_problems", [])]),
        identified_count=review_analysis.get("identified_count", 0),
        total_problems=review_analysis.get("total_problems", len(known_problems)),
        identified_percentage=review_analysis.get("identified_percentage", 0),
        iteration_count=iteration_count,
        max_iterations=max_iterations
    )
    
    # Run the LLM
    logger.info("Generating targeted guidance for review iteration")
    guidance = llm.invoke(system_prompt + "\n\n" + formatted_prompt)
    
    return guidance

def summarize_review_comments(
    review_comments: str,
    llm: BaseLanguageModel,
    is_java: bool = True
) -> str:
    """
    Summarize review comments into a concise summary.
    
    Args:
        review_comments: The full review comments
        llm: Language model to use
        is_java: Whether the code is Java (for using Java-specific prompts)
        
    Returns:
        Summarized review comments
    """
    # Create and format the prompt
    system_prompt = """You are an expert in synthesizing technical information about Java code. Your summaries are:
1. Clear and concise
2. Well-structured and organized
3. Focused on the most important issues
4. Educational and actionable
5. Prioritized by severity and importance"""
    
    prompt = """
Please create a clear, concise summary of the following code review comments:

```
{review_comments}
```

Create a well-structured summary that:
1. Groups related issues together (e.g., style issues, logical errors, etc.)
2. Prioritizes issues by importance/severity
3. Is easy to understand for a Java programming student
4. Highlights patterns or common themes in the review

Your summary should be educational and actionable.
"""
    
    formatted_prompt = prompt.format(
        review_comments=review_comments
    )
    
    # Run the LLM
    logger.info("Generating review summary")
    if is_java:
        summary = llm.invoke(system_prompt + "\n\n" + formatted_prompt)
    else:
        summary = llm.invoke(formatted_prompt)
    
    return summary

def compare_and_explain(
    code_snippet: str,
    known_problems: List[str],
    student_review: str,
    review_analysis: Dict,
    review_summary: str,
    review_history: List[Dict] = None,
    llm: BaseLanguageModel = None,
    is_java: bool = True
) -> str:
    """
    Compare student review with known problems and provide educational feedback.
    Enhanced with review history for iterative feedback.
    
    Args:
        code_snippet: The original code snippet
        known_problems: List of known problems in the code
        student_review: The student's review comments
        review_analysis: Analysis of the student review
        review_summary: Summary of the review comments
        review_history: History of previous review attempts
        llm: Language model to use
        is_java: Whether the code is Java (for using Java-specific prompts)
        
    Returns:
        Educational feedback explaining the comparison
    """
    # Create and format the prompt
    system_prompt = """You are an expert Java programming educator specializing in teaching code review skills. Your educational reports are:
1. Encouraging and constructive
2. Specific about both strengths and areas for improvement
3. Educational, explaining why issues matter in Java
4. Practical, with tips for better identifying similar issues in the future
5. Supportive of the student's learning journey"""
    
    base_prompt = """
Please create an educational report comparing the student's code review with the known problems in the code.

Original code snippet:
```java
{code_snippet}
```

Known problems in the code:
{known_problems}

Student's review:
```
{student_review}
```

Analysis of the student's review:
- Problems correctly identified:
{identified_problems}

- Problems missed:
{missed_problems}

- False positives (things incorrectly identified as problems):
{false_positives}

- Overall accuracy: {accuracy_percentage}%

Review summary:
{review_summary}
"""

    # Add review history information if available
    if review_history and len(review_history) > 1:
        history_text = "\n\nReview iteration history:\n"
        for i, review in enumerate(review_history, 1):
            review_analysis = review.get("review_analysis", {})
            history_text += f"Attempt {i}: Found {review_analysis.get('identified_count', 0)} of "
            history_text += f"{review_analysis.get('total_problems', len(known_problems))} issues "
            history_text += f"({review_analysis.get('accuracy_percentage', 0):.1f}% accuracy)\n"
        base_prompt += history_text

    # Format the prompt
    formatted_prompt = base_prompt.format(
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
    if is_java:
        comparison = llm.invoke(system_prompt + "\n\n" + formatted_prompt)
    else:
        comparison = llm.invoke(formatted_prompt)
    
    return comparison