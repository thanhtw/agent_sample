"""
Java error handler for peer code review system.

This module provides functionality for handling Java-specific errors,
loading error data from JSON files, and selecting errors for injection.
Enhanced with direct category selection and improved error injection.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to load Java error data
JAVA_BUILD_ERRORS = {}
JAVA_CHECKSTYLE_ERRORS = {}

try:
    # Get the directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Load build errors
    with open(os.path.join(dir_path, "build_errors.json"), "r") as f:
        JAVA_BUILD_ERRORS = json.load(f)
    logger.info(f"Loaded {sum(len(errors) for errors in JAVA_BUILD_ERRORS.values())} Java build errors")
    
    # Load checkstyle errors
    with open(os.path.join(dir_path, "checkstyle_error.json"), "r") as f:
        JAVA_CHECKSTYLE_ERRORS = json.load(f)
    logger.info(f"Loaded {sum(len(checks) for checks in JAVA_CHECKSTYLE_ERRORS.values())} Java checkstyle errors")
except Exception as e:
    logger.error(f"Error loading Java error data: {str(e)}")

# Java code templates of increasing complexity
JAVA_TEMPLATES = {
    "short": """
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
        return a / b; // Potential error: Integer division
    }
}
""",
    "medium": """
import java.util.ArrayList;
import java.util.List;

public class TaskManager {
    private List<Task> tasks;
    
    public TaskManager() {
        this.tasks = new ArrayList<>();
    }
    
    public void addTask(Task task) {
        tasks.add(task);
    }
    
    public Task findTask(String name) {
        for (Task task : tasks) {
            if (task.getName().equals(name)) {
                return task;
            }
        }
        return null;
    }
    
    public List<Task> getCompletedTasks() {
        List<Task> completedTasks = new ArrayList<>();
        for (Task task : tasks) {
            if (task.isCompleted()) {
                completedTasks.add(task);
            }
        }
        return completedTasks;
    }
    
    public class Task {
        private String name;
        private boolean completed;
        
        public Task(String name) {
            this.name = name;
            this.completed = false;
        }
        
        public String getName() {
            return name;
        }
        
        public boolean isCompleted() {
            return completed;
        }
        
        public void setCompleted(boolean completed) {
            this.completed = completed;
        }
    }
}
""",
    "long": """
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A student management system for tracking enrollments and grades.
 */
public class StudentManagementSystem {
    private Map<String, Student> students;
    private Map<String, Course> courses;
    
    public StudentManagementSystem() {
        this.students = new HashMap<>();
        this.courses = new HashMap<>();
    }
    
    public void addStudent(String id, String name, int age) {
        Student student = new Student(id, name, age);
        students.put(id, student);
    }
    
    public void addCourse(String code, String title, int maxCapacity) {
        Course course = new Course(code, title, maxCapacity);
        courses.put(code, course);
    }
    
    public boolean enrollStudent(String studentId, String courseCode) {
        Student student = students.get(studentId);
        Course course = courses.get(courseCode);
        
        if (student == null || course == null) {
            return false;
        }
        
        if (course.getEnrolledStudents().size() >= course.getMaxCapacity()) {
            return false; // Course is full
        }
        
        course.addStudent(student);
        student.enrollInCourse(course);
        return true;
    }
    
    public void assignGrade(String studentId, String courseCode, double grade) {
        Student student = students.get(studentId);
        Course course = courses.get(courseCode);
        
        if (student != null && course != null) {
            student.setGrade(courseCode, grade);
        }
    }
    
    public double getStudentAverageGrade(String studentId) {
        Student student = students.get(studentId);
        if (student == null) {
            return 0.0;
        }
        return student.calculateAverageGrade();
    }
    
    public List<Student> getTopStudents(int n) {
        List<Student> allStudents = new ArrayList<>(students.values());
        allStudents.sort((s1, s2) -> 
            Double.compare(s2.calculateAverageGrade(), s1.calculateAverageGrade()));
        
        List<Student> topStudents = new ArrayList<>();
        for (int i = 0; i < Math.min(n, allStudents.size()); i++) {
            topStudents.add(allStudents.get(i));
        }
        
        return topStudents;
    }
    
    public class Student {
        private String id;
        private String name;
        private int age;
        private List<Course> enrolledCourses;
        private Map<String, Double> grades;
        
        public Student(String id, String name, int age) {
            this.id = id;
            this.name = name;
            this.age = age;
            this.enrolledCourses = new ArrayList<>();
            this.grades = new HashMap<>();
        }
        
        public String getId() {
            return id;
        }
        
        public String getName() {
            return name;
        }
        
        public int getAge() {
            return age;
        }
        
        public List<Course> getEnrolledCourses() {
            return enrolledCourses;
        }
        
        public void enrollInCourse(Course course) {
            enrolledCourses.add(course);
        }
        
        public void setGrade(String courseCode, double grade) {
            grades.put(courseCode, grade);
        }
        
        public double getGrade(String courseCode) {
            return grades.getOrDefault(courseCode, 0.0);
        }
        
        public double calculateAverageGrade() {
            if (grades.isEmpty()) {
                return 0.0;
            }
            
            double total = 0.0;
            for (double grade : grades.values()) {
                total += grade;
            }
            
            return total / grades.size();
        }
    }
    
    public class Course {
        private String code;
        private String title;
        private int maxCapacity;
        private List<Student> enrolledStudents;
        
        public Course(String code, String title, int maxCapacity) {
            this.code = code;
            this.title = title;
            this.maxCapacity = maxCapacity;
            this.enrolledStudents = new ArrayList<>();
        }
        
        public String getCode() {
            return code;
        }
        
        public String getTitle() {
            return title;
        }
        
        public int getMaxCapacity() {
            return maxCapacity;
        }
        
        public List<Student> getEnrolledStudents() {
            return enrolledStudents;
        }
        
        public void addStudent(Student student) {
            enrolledStudents.add(student);
        }
    }
}
"""
}

def get_all_error_categories() -> Dict[str, List[str]]:
    """
    Get all available error categories from both build errors and checkstyle errors.
    
    Returns:
        Dict[str, List[str]]: Dictionary with 'build' and 'checkstyle' categories
    """
    build_categories = list(JAVA_BUILD_ERRORS.keys())
    checkstyle_categories = list(JAVA_CHECKSTYLE_ERRORS.keys())
    
    return {
        "build": build_categories,
        "checkstyle": checkstyle_categories
    }

def get_all_error_types() -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """
    Get all error types organized by category.
    
    Returns:
        Dict: Hierarchical dictionary of all error types
    """
    error_types = {
        "build": {},
        "checkstyle": {}
    }
    
    # Build errors
    for category, errors in JAVA_BUILD_ERRORS.items():
        error_types["build"][category] = errors
    
    # Checkstyle errors
    for category, errors in JAVA_CHECKSTYLE_ERRORS.items():
        error_types["checkstyle"][category] = errors
    
    return error_types

def select_java_template(code_length: str) -> str:
    """
    Select a Java code template based on code length.
    
    Args:
        code_length: Desired code length (short, medium, long)
        
    Returns:
        Template code as a string
    """
    return JAVA_TEMPLATES.get(code_length.lower(), JAVA_TEMPLATES["medium"])

def select_java_errors_by_categories(selected_categories: Dict[str, List[str]], 
                                    difficulty_level: str,
                                    error_counts: Dict[str, int] = None) -> List[Dict[str, str]]:
    """
    Select Java-specific errors based on selected categories.
    
    Args:
        selected_categories: Dictionary with 'build' and 'checkstyle' keys, each containing a list of selected categories
        difficulty_level: Difficulty level (easy, medium, hard)
        error_counts: Optional dictionary with 'build' and 'checkstyle' keys, each containing the number of errors to select
        
    Returns:
        List of selected error dictionaries with name and description
    """
    selected_errors = []
    
    # Determine number of errors based on difficulty if not specified
    if error_counts is None:
        total_errors = {
            "easy": 2,
            "medium": 4,
            "hard": 6
        }.get(difficulty_level.lower(), 3)
        
        # Calculate number of errors for each type
        build_categories = selected_categories.get("build", [])
        checkstyle_categories = selected_categories.get("checkstyle", [])
        
        total_categories = len(build_categories) + len(checkstyle_categories)
        
        if total_categories == 0:
            # If no categories selected, default to some standard categories
            build_categories = ["CompileTimeErrors", "RuntimeErrors"]
            checkstyle_categories = ["NamingConventionChecks", "WhitespaceAndFormattingChecks"]
            total_categories = len(build_categories) + len(checkstyle_categories)
        
        # Distribute errors between build and checkstyle
        if total_categories > 0:
            build_proportion = len(build_categories) / total_categories
            build_count = max(1, round(total_errors * build_proportion)) if build_categories else 0
            checkstyle_count = max(1, total_errors - build_count) if checkstyle_categories else 0
        else:
            build_count = total_errors // 2
            checkstyle_count = total_errors - build_count
        
        error_counts = {
            "build": build_count,
            "checkstyle": checkstyle_count
        }
    
    # Select build errors
    if "build" in selected_categories and selected_categories["build"]:
        build_categories = selected_categories["build"]
        build_count = error_counts["build"]
        
        if build_count > 0 and build_categories:
            # Distribute errors across selected categories
            errors_per_category = max(1, build_count // len(build_categories))
            remaining_errors = build_count - (errors_per_category * len(build_categories))
            
            for category in build_categories:
                if category in JAVA_BUILD_ERRORS:
                    # Get all errors in this category
                    category_errors = JAVA_BUILD_ERRORS[category]
                    if category_errors:
                        # Calculate how many errors to select from this category
                        category_count = errors_per_category
                        if remaining_errors > 0:
                            category_count += 1
                            remaining_errors -= 1
                        
                        # Select random errors from this category
                        sample_size = min(category_count, len(category_errors))
                        if sample_size > 0:
                            sampled_errors = random.sample(category_errors, sample_size)
                            for error in sampled_errors:
                                selected_errors.append({
                                    "type": "build",
                                    "category": category,
                                    "name": error["error_name"],
                                    "description": error["description"]
                                })
    
    # Select checkstyle errors
    if "checkstyle" in selected_categories and selected_categories["checkstyle"]:
        checkstyle_categories = selected_categories["checkstyle"]
        checkstyle_count = error_counts["checkstyle"]
        
        if checkstyle_count > 0 and checkstyle_categories:
            # Distribute errors across selected categories
            errors_per_category = max(1, checkstyle_count // len(checkstyle_categories))
            remaining_errors = checkstyle_count - (errors_per_category * len(checkstyle_categories))
            
            for category in checkstyle_categories:
                if category in JAVA_CHECKSTYLE_ERRORS:
                    # Get all errors in this category
                    category_errors = JAVA_CHECKSTYLE_ERRORS[category]
                    if category_errors:
                        # Calculate how many errors to select from this category
                        category_count = errors_per_category
                        if remaining_errors > 0:
                            category_count += 1
                            remaining_errors -= 1
                        
                        # Select random errors from this category
                        sample_size = min(category_count, len(category_errors))
                        if sample_size > 0:
                            sampled_errors = random.sample(category_errors, sample_size)
                            for error in sampled_errors:
                                selected_errors.append({
                                    "type": "checkstyle",
                                    "category": category,
                                    "name": error["check_name"],
                                    "description": error["description"]
                                })
    
    # Ensure we have at least some errors if nothing was selected
    if not selected_errors:
        # Default to one build error and one checkstyle error
        if "CompileTimeErrors" in JAVA_BUILD_ERRORS and JAVA_BUILD_ERRORS["CompileTimeErrors"]:
            build_error = random.choice(JAVA_BUILD_ERRORS["CompileTimeErrors"])
            selected_errors.append({
                "type": "build",
                "category": "CompileTimeErrors",
                "name": build_error["error_name"],
                "description": build_error["description"]
            })
            
        if "NamingConventionChecks" in JAVA_CHECKSTYLE_ERRORS and JAVA_CHECKSTYLE_ERRORS["NamingConventionChecks"]:
            style_error = random.choice(JAVA_CHECKSTYLE_ERRORS["NamingConventionChecks"])
            selected_errors.append({
                "type": "checkstyle",
                "category": "NamingConventionChecks",
                "name": style_error["check_name"],
                "description": style_error["description"]
            })
    
    # Legacy function for backward compatibility
    def select_java_errors(problem_areas: List[str], difficulty_level: str) -> List[Dict[str, str]]:
        """
        Select Java-specific errors based on problem areas and difficulty.
        
        Args:
            problem_areas: List of problem areas to select from
            difficulty_level: Difficulty level (easy, medium, hard)
            
        Returns:
            List of selected error dictionaries with name and description
        """
        # Map problem areas to categories
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
        
        # Use the new function
        return select_java_errors_by_categories(selected_categories, difficulty_level)
    
    return selected_errors

def format_java_error_description(error: Dict[str, str]) -> str:
    """
    Format a Java error dictionary into a readable description.
    
    Args:
        error: Error dictionary with type, category, name, and description
        
    Returns:
        Formatted error description
    """
    error_type = error["type"].title()
    category = error["category"]
    name = error["name"]
    description = error["description"]
    
    return f"{error_type} Error - {name}: {description} (Category: {category})"

def get_error_injection_instructions(errors: List[Dict[str, str]]) -> str:
    """
    Generate instructions on how to inject the selected errors into Java code.
    
    Args:
        errors: List of error dictionaries
        
    Returns:
        Instructions for error injection
    """
    instructions = "Please introduce the following specific Java errors into the code:\n\n"
    
    for idx, error in enumerate(errors, 1):
        error_type = error["type"]
        name = error["name"]
        description = error["description"]
        
        instructions += f"{idx}. {error_type.upper()} ERROR - {name}\n"
        instructions += f"   Description: {description}\n"
        
        # Add specific implementation suggestion based on error type
        if error_type == "build":
            if "NullPointer" in name:
                instructions += f"   Suggestion: Create a scenario where a null object is accessed\n"
            elif "Missing" in name:
                instructions += f"   Suggestion: Omit a required element\n"
            elif "Type" in name:
                instructions += f"   Suggestion: Use incompatible types in an assignment or operation\n"
            elif "Index" in name or "Bounds" in name:
                instructions += f"   Suggestion: Create an array access with an invalid index\n"
            else:
                instructions += f"   Suggestion: Implement this error in a realistic way\n"
        else:  # checkstyle
            if "Naming" in name or "Name" in name:
                instructions += f"   Suggestion: Use inappropriate naming conventions\n"
            elif "Whitespace" in name:
                instructions += f"   Suggestion: Use inconsistent whitespace\n"
            elif "Javadoc" in name:
                instructions += f"   Suggestion: Create incorrect or missing Javadoc\n"
            elif "Braces" in name or "Curly" in name:
                instructions += f"   Suggestion: Use inconsistent brace placement\n"
            else:
                instructions += f"   Suggestion: Implement this style violation naturally\n"
                
        instructions += "\n"
    
    return instructions