"""
Debugging script for Java code review agent workflow.

This script provides step-by-step testing of each component of the workflow
to identify and fix any issues.
"""

import os
import logging
import json
import time
from dotenv import load_dotenv

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug_agent.log")
    ]
)
logger = logging.getLogger("debug_agent")

# Load environment variables
load_dotenv()

def test_llm_manager():
    """Test LLM Manager initialization and connection."""
    logger.info("Testing LLM Manager...")
    
    try:
        from llm_manager import LLMManager
        
        logger.info("Initializing LLM Manager")
        llm_manager = LLMManager()
        
        # Test Ollama connection
        logger.info("Testing Ollama connection")
        connection_status, message = llm_manager.check_ollama_connection()
        
        logger.info(f"Connection status: {connection_status}")
        logger.info(f"Connection message: {message}")
        
        if not connection_status:
            logger.error("Ollama connection failed - aborting further tests")
            return False
        
        # Get available models
        logger.info("Getting available models")
        models = llm_manager.get_available_models()
        logger.info(f"Found {len(models)} models")
        
        # Check if default model exists
        default_model = llm_manager.default_model
        logger.info(f"Default model: {default_model}")
        
        model_available = llm_manager.check_model_availability(default_model)
        logger.info(f"Default model available: {model_available}")
        
        if not model_available:
            logger.warning(f"Default model {default_model} not available - trying to download")
            llm_manager.download_ollama_model(default_model)
            
            # Check again
            model_available = llm_manager.check_model_availability(default_model)
            logger.info(f"Default model available after download attempt: {model_available}")
            
            if not model_available:
                logger.error(f"Could not download default model {default_model}")
                return False
        
        # Initialize model
        logger.info(f"Initializing default model {default_model}")
        llm = llm_manager.initialize_model(default_model)
        
        if llm:
            logger.info("Model initialized successfully")
            
            # Test basic generation
            logger.info("Testing basic text generation")
            try:
                prompt = "Complete this Java statement: public class HelloWorld {"
                response = llm.invoke(prompt)
                logger.info(f"Model response: {response[:100]}...")
                
                return True
            except Exception as e:
                logger.error(f"Error during generation test: {e}")
                return False
        else:
            logger.error("Failed to initialize model")
            return False
    
    except Exception as e:
        logger.error(f"Error in LLM Manager test: {e}")
        return False

def test_code_generation():
    """Test the code generation component."""
    logger.info("Testing code generation...")
    
    try:
        from agent_tools import generate_code_problem
        from llm_manager import LLMManager
        
        # Initialize LLM
        llm_manager = LLMManager()
        default_model = llm_manager.default_model
        llm = llm_manager.initialize_model(default_model)
        
        if not llm:
            logger.error("Failed to initialize LLM for code generation test")
            return False
        
        # Test parameters
        language = "java"
        problem_areas = ["style", "logical"]
        difficulty_level = "medium"
        code_length = "short"
        
        logger.info(f"Generating {language} code with {problem_areas} problems")
        start_time = time.time()
        
        code_snippet, problems, raw_errors = generate_code_problem(
            programming_language=language,
            problem_areas=problem_areas,
            difficulty_level=difficulty_level,
            code_length=code_length,
            llm=llm
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Code generation completed in {generation_time:.2f} seconds")
        
        # Log results
        logger.info(f"Generated {len(problems)} problems")
        for i, problem in enumerate(problems, 1):
            logger.info(f"Problem {i}: {problem}")
        
        # Save to file for inspection
        with open("debug_generated_code.java", "w") as f:
            f.write(code_snippet)
        
        logger.info("Code saved to debug_generated_code.java")
        
        return bool(code_snippet and problems)
    
    except Exception as e:
        logger.error(f"Error in code generation test: {e}")
        return False

def test_review_analysis():
    """Test the student review analysis component."""
    logger.info("Testing review analysis...")
    
    try:
        from agent_tools import analyze_student_review
        from llm_manager import LLMManager
        
        # Initialize LLM
        llm_manager = LLMManager()
        default_model = llm_manager.initialize_model(default_model=llm_manager.default_model)
        
        if not default_model:
            logger.error("Failed to initialize LLM for review analysis test")
            return False
        
        # Use a sample code snippet and known problems
        code_snippet = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public double divide(int a, int b) {
        return a / b; // Integer division, should be (double)a / b
    }
}
"""
        
        known_problems = [
            "Integer division in divide() method without casting to double",
            "No null checks or validation in methods",
            "No documentation or comments explaining the class purpose"
        ]
        
        # Mock student review that finds some issues
        student_review = """
In the Calculator class:
1. The divide method has an issue with integer division. It should cast to double.
2. There's no JavaDoc or comments explaining the class.
"""
        
        logger.info("Analyzing student review")
        start_time = time.time()
        
        analysis = analyze_student_review(
            code_snippet=code_snippet,
            known_problems=known_problems,
            student_review=student_review,
            llm=default_model
        )
        
        analysis_time = time.time() - start_time
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        
        # Log results
        logger.info(f"Analysis results: {json.dumps(analysis, indent=2)}")
        
        # Check if we got expected fields
        if all(k in analysis for k in ["identified_problems", "missed_problems", "review_sufficient"]):
            logger.info("Analysis returned expected fields")
            return True
        else:
            logger.error("Analysis missing expected fields")
            return False
    
    except Exception as e:
        logger.error(f"Error in review analysis test: {e}")
        return False

def test_targeted_guidance():
    """Test the targeted guidance generation component."""
    logger.info("Testing targeted guidance generation...")
    
    try:
        from agent_tools import generate_targeted_guidance
        from llm_manager import LLMManager
        
        # Initialize LLM
        llm_manager = LLMManager()
        default_model = llm_manager.initialize_model(default_model=llm_manager.default_model)
        
        if not default_model:
            logger.error("Failed to initialize LLM for targeted guidance test")
            return False
        
        # Use sample code and analysis from previous test
        code_snippet = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public double divide(int a, int b) {
        return a / b; // Integer division, should be (double)a / b
    }
}
"""
        
        known_problems = [
            "Integer division in divide() method without casting to double",
            "No null checks or validation in methods",
            "No documentation or comments explaining the class purpose"
        ]
        
        student_review = """
In the Calculator class:
1. The divide method has an issue with integer division. It should cast to double.
2. There's no JavaDoc or comments explaining the class.
"""
        
        review_analysis = {
            "identified_problems": [
                "Integer division in divide() method",
                "No documentation or comments"
            ],
            "missed_problems": [
                "No null checks or validation in methods"
            ],
            "identified_percentage": 66.7,
            "identified_count": 2,
            "total_problems": 3,
            "review_sufficient": True,
            "feedback": "Good job identifying most issues"
        }
        
        logger.info("Generating targeted guidance")
        start_time = time.time()
        
        guidance = generate_targeted_guidance(
            code_snippet=code_snippet,
            known_problems=known_problems,
            student_review=student_review,
            review_analysis=review_analysis,
            iteration_count=1,
            max_iterations=3,
            llm=default_model
        )
        
        guidance_time = time.time() - start_time
        logger.info(f"Guidance generation completed in {guidance_time:.2f} seconds")
        
        # Log results
        logger.info(f"Guidance: {guidance[:200]}...")
        
        # Save guidance to file
        with open("debug_guidance.txt", "w") as f:
            f.write(guidance)
        
        logger.info("Guidance saved to debug_guidance.txt")
        
        return bool(guidance)
    
    except Exception as e:
        logger.error(f"Error in targeted guidance test: {e}")
        return False

def test_end_to_end_workflow():
    """Test the entire agent workflow."""
    logger.info("Testing end-to-end workflow...")
    
    try:
        from agent import run_peer_review_agent
        
        # Run with minimal parameters
        logger.info("Running peer review agent")
        start_time = time.time()
        
        result = run_peer_review_agent(
            programming_language="java",
            problem_areas=["style", "logical"],
            difficulty_level="easy",
            code_length="short"
        )
        
        workflow_time = time.time() - start_time
        logger.info(f"Initial workflow completed in {workflow_time:.2f} seconds")
        
        # Check for error
        if "error" in result and result["error"]:
            logger.error(f"Workflow returned error: {result['error']}")
            return False
        
        # Check for expected fields
        expected_fields = ["code_snippet", "known_problems", "current_step"]
        if all(field in result for field in expected_fields):
            logger.info("Workflow returned expected fields")
            
            # Log code snippet and problems
            logger.info(f"Generated code with {len(result['known_problems'])} problems")
            
            # Save to file
            with open("debug_workflow_result.json", "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info("Workflow result saved to debug_workflow_result.json")
            
            # Test a mock student review
            if "code_snippet" in result and result["code_snippet"]:
                mock_review = "After reviewing the code, I found a few issues:\n"
                mock_review += "1. The code has poor naming conventions\n"
                mock_review += "2. There's missing error handling\n"
                
                logger.info("Running workflow with mock student review")
                start_time = time.time()
                
                review_result = run_peer_review_agent(
                    programming_language="java",
                    problem_areas=["style", "logical"],
                    difficulty_level="easy",
                    code_length="short",
                    student_review=mock_review,
                    iteration_count=1
                )
                
                review_time = time.time() - start_time
                logger.info(f"Review workflow completed in {review_time:.2f} seconds")
                
                # Save review result
                with open("debug_review_result.json", "w") as f:
                    json.dump(review_result, f, indent=2)
                
                logger.info("Review result saved to debug_review_result.json")
                
                if "error" in review_result and review_result["error"]:
                    logger.error(f"Review workflow returned error: {review_result['error']}")
                    return False
                
                return True
            
            return True
        else:
            logger.error(f"Workflow missing expected fields. Got: {list(result.keys())}")
            return False
    
    except Exception as e:
        logger.error(f"Error in end-to-end workflow test: {e}")
        return False

def run_all_tests():
    """Run all tests in sequence."""
    tests = [
        ("LLM Manager", test_llm_manager),
        ("Code Generation", test_code_generation),
        ("Review Analysis", test_review_analysis),
        ("Targeted Guidance", test_targeted_guidance),
        ("End-to-End Workflow", test_end_to_end_workflow)
    ]
    
    results = {}
    
    logger.info("Starting agent workflow tests")
    logger.info("=" * 50)
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name} test...")
        try:
            success = test_func()
            results[test_name] = success
            logger.info(f"{test_name} test {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            logger.error(f"Exception in {test_name} test: {e}")
            results[test_name] = False
        
        logger.info("-" * 50)
    
    # Summary
    logger.info("Test Results Summary:")
    all_passed = True
    for test_name, result in results.items():
        logger.info(f"{test_name}: {'PASS' if result else 'FAIL'}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("All tests PASSED!")
    else:
        logger.info("Some tests FAILED. Check logs for details.")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()