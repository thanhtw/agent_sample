"""
Java error handler for peer code review system.

This module provides functionality for handling Java-specific errors,
loading error data from JSON files, and selecting errors for injection.
"""

import os
import json
import random
import logging
from typing import List, Dict, Any, Optional, Tuple

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

def select_java_template(code_length: str) -> str:
    """
    Select a Java code template based on code length.
    
    Args:
        code_length: Desired code length (short, medium, long)
        
    Returns:
        Template code as a string
    """
    return JAVA_TEMPLATES.get(code_length.lower(), JAVA_TEMPLATES["medium"])

def select_java_errors(problem_areas: List[str], difficulty_level: str) -> List[Dict[str, str]]:
    """
    Select Java-specific errors based on problem areas and difficulty.
    
    Args:
        problem_areas: List of problem areas to select from
        difficulty_level: Difficulty level (easy, medium, hard)
        
    Returns:
        List of selected error dictionaries with name and description
    """
    selected_errors = []
    
    # Number of errors to select based on difficulty
    num_errors = {
        "easy": 2,
        "medium": 4,
        "hard": 6
    }.get(difficulty_level.lower(), 3)
    
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
    
    # For each problem area, select errors
    for area in problem_areas:
        area = area.lower()
        area_error_count = max(1, num_errors // len(problem_areas))
        
        # Select build errors
        build_categories = area_to_build_errors.get(area, [])
        for category in build_categories:
            if category in JAVA_BUILD_ERRORS:
                errors = JAVA_BUILD_ERRORS[category]
                sample_size = min(area_error_count // max(1, len(build_categories)), len(errors))
                if sample_size > 0:
                    sampled_errors = random.sample(errors, sample_size)
                    for error in sampled_errors:
                        selected_errors.append({
                            "type": "build",
                            "category": category,
                            "name": error["error_name"],
                            "description": error["description"]
                        })
        
        # Select checkstyle errors
        checkstyle_categories = area_to_checkstyle_errors.get(area, [])
        for category in checkstyle_categories:
            if category in JAVA_CHECKSTYLE_ERRORS:
                checks = JAVA_CHECKSTYLE_ERRORS[category]
                sample_size = min(area_error_count // max(1, len(checkstyle_categories)), len(checks))
                if sample_size > 0:
                    sampled_checks = random.sample(checks, sample_size)
                    for check in sampled_checks:
                        selected_errors.append({
                            "type": "checkstyle",
                            "category": category,
                            "name": check["check_name"],
                            "description": check["description"]
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
    
    # Limit to the specified number of errors
    if len(selected_errors) > num_errors:
        selected_errors = random.sample(selected_errors, num_errors)
    
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