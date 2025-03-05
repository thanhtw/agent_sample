"""
Simple test script for Java code review agent.

Run this script to test the basic functionality of the agent.
"""

import sys
import logging
from agent import run_peer_review_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_script")

def test_code_generation():
    """Test the generation of Java code with problems."""
    logger.info("Testing code generation...")
    
    # Run the agent to generate code
    result = run_peer_review_agent(
        programming_language="java",
        problem_areas=["style", "logical"],
        difficulty_level="easy",
        code_length="short"
    )
    
    if "error" in result and result["error"]:
        logger.error(f"Error: {result['error']}")
        return False
    
    if "code_snippet" not in result or not result["code_snippet"]:
        logger.error("No code snippet generated")
        return False
    
    if "known_problems" not in result or not result["known_problems"]:
        logger.error("No problems generated")
        return False
    
    # Print the generated code and problems
    print("\n=== GENERATED JAVA CODE ===")
    print(result["code_snippet"])
    
    print("\n=== KNOWN PROBLEMS ===")
    for i, problem in enumerate(result["known_problems"], 1):
        print(f"{i}. {problem}")
    
    return True

def test_review_analysis():
    """Test the review analysis with a mock student review."""
    logger.info("Testing review analysis...")
    
    # First, generate code
    gen_result = run_peer_review_agent(
        programming_language="java",
        problem_areas=["style", "logical"],
        difficulty_level="easy",
        code_length="short"
    )
    
    if "error" in gen_result and gen_result["error"]:
        logger.error(f"Error in code generation: {gen_result['error']}")
        return False
    
    code_snippet = gen_result["code_snippet"]
    known_problems = gen_result["known_problems"]
    
    # Create a mock student review
    # In a real scenario, this would come from the student
    mock_review = "I've reviewed the code and found these issues:\n"
    mock_review += "1. The naming conventions are inconsistent\n"
    mock_review += "2. There's missing error handling\n"
    
    # Run the agent with the mock review
    review_result = run_peer_review_agent(
        programming_language="java",
        problem_areas=["style", "logical"],
        difficulty_level="easy",
        code_length="short",
        student_review=mock_review,
        iteration_count=1,
        max_iterations=3
    )
    
    if "error" in review_result and review_result["error"]:
        logger.error(f"Error in review analysis: {review_result['error']}")
        return False
    
    if "review_analysis" not in review_result:
        logger.error("No review analysis generated")
        return False
    
    # Print the review analysis
    print("\n=== REVIEW ANALYSIS ===")
    analysis = review_result["review_analysis"]
    
    print(f"Identified problems: {len(analysis.get('identified_problems', []))}")
    for i, problem in enumerate(analysis.get('identified_problems', []), 1):
        print(f"✓ {problem}")
    
    print(f"\nMissed problems: {len(analysis.get('missed_problems', []))}")
    for i, problem in enumerate(analysis.get('missed_problems', []), 1):
        print(f"✗ {problem}")
    
    print(f"\nReview sufficient: {analysis.get('review_sufficient', False)}")
    print(f"Accuracy: {analysis.get('accuracy_percentage', 0):.1f}%")
    
    # If the review is insufficient and targeted guidance is available, print it
    if not analysis.get('review_sufficient', False) and "targeted_guidance" in review_result:
        print("\n=== TARGETED GUIDANCE ===")
        print(review_result["targeted_guidance"])
    
    return True

def run_tests():
    """Run all tests."""
    try:
        print("=== TESTING JAVA CODE REVIEW AGENT ===")
        
        # Test code generation
        print("\n1. Testing code generation...")
        if not test_code_generation():
            print("❌ Code generation test failed")
            return False
        print("✅ Code generation test passed")
        
        # Test review analysis
        print("\n2. Testing review analysis...")
        if not test_review_analysis():
            print("❌ Review analysis test failed")
            return False
        print("✅ Review analysis test passed")
        
        print("\n=== ALL TESTS PASSED ===")
        return True
        
    except Exception as e:
        logger.exception("Error running tests")
        print(f"❌ Tests failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)