#!/usr/bin/env python3
"""
Static analysis script for Dragonlight Coding Standards v2.
Analyzes Python source files for violations of the coding standards.
"""
import ast
import sys
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Violation:
    line: int
    column: int
    rule: str
    description: str

def analyze_file(filepath: str) -> List[Violation]:
    """Analyze a single Python file for coding standard violations."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        return [Violation(e.lineno, e.offset, "SYNTAX", f"Syntax error: {e.msg}")]
    
    violations = []
    
    # Check for module-level mutable state
    violations.extend(check_module_level_mutable(tree, filepath))
    
    # Check functions
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(check_function_standards(node, filepath))
        elif isinstance(node, ast.ClassDef):
            violations.extend(check_class_standards(node, filepath))
    
    return violations

def check_module_level_mutable(tree: ast.AST, filepath: str) -> List[Violation]:
    """Check for module-level mutable state."""
    violations = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Check if assignment is at module level (not inside function/class)
            # We'll consider any top-level assignment as potentially mutable
            # For simplicity, flag all top-level assignments that are not constants
            # (UPPER_CASE with underscores)
            for target in ast.walk(node):
                if isinstance(target, ast.Name):
                    # Check if it looks like a constant
                    if not target.id.isupper() or '_' not in target.id:
                        violations.append(Violation(
                            node.lineno, node.col_offset,
                            "MUTABLE_STATE",
                            f"Module-level mutable variable: {target.id}"
                        ))
                        break
    return violations

def check_function_standards(node: ast.FunctionDef, filepath: str) -> List[Violation]:
    """Check function-specific standards."""
    violations = []
    
    # Function length (approximate by counting lines in the function body)
    # We'll use the end_lineno if available (Python 3.8+)
    if hasattr(node, 'end_lineno'):
        length = node.end_lineno - node.lineno + 1
    else:
        # Fallback: estimate by counting lines in the source
        length = 0  # We'll skip exact length for now
    
    if length > 40:
        violations.append(Violation(
            node.lineno, node.col_offset,
            "FUNC_LENGTH",
            f"Function too long: {length} lines (max 40)"
        ))
    
    # Parameter count
    param_count = len(node.args.args) + len(node.args.kwonlyargs)
    if node.args.vararg:
        param_count += 1
    if node.args.kwarg:
        param_count += 1
    if param_count > 4:
        violations.append(Violation(
            node.lineno, node.col_offset,
            "PARAM_COUNT",
            f"Too many parameters: {param_count} (max 4)"
        ))
    
    # Nesting depth (we'll approximate by checking maximum nesting of control structures)
    max_depth = calculate_nesting_depth(node)
    if max_depth > 3:
        violations.append(Violation(
            node.lineno, node.col_offset,
            "NESTING_DEPTH",
            f"Nesting depth too deep: {max_depth} (max 3)"
        ))
    
    # Type annotation coverage (for public functions)
    if not node.name.startswith('_'):
        # Check return type
        if node.returns is None:
            violations.append(Violation(
                node.lineno, node.col_offset,
                "MISSING_RETURN_TYPE",
                f"Missing return type annotation for public function: {node.name}"
            ))
        # Check parameter types
        for arg in node.args.args:
            if arg.annotation is None:
                violations.append(Violation(
                    arg.lineno, arg.col_offset,
                    "MISSING_PARAM_TYPE",
                    f"Missing type annotation for parameter: {arg.arg}"
                ))
    
    # Assertion density (advisory: 2 per function)
    assertion_count = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            assertion_count += 1
    if assertion_count < 2:
        violations.append(Violation(
            node.lineno, node.col_offset,
            "LOW_ASSERTION_DENSITY",
            f"Low assertion density: {assertion_count} (advisory: 2 per function)"
        ))
    
    # Check for bare except or except Exception
    for n in ast.walk(node):
        if isinstance(n, ast.ExceptHandler):
            if n.type is None:
                violations.append(Violation(
                    n.lineno, n.col_offset,
                    "BARE_EXCEPT",
                    "Bare except clause"
                ))
            elif isinstance(n.type, ast.Name) and n.type.id == 'Exception':
                violations.append(Violation(
                    n.lineno, n.col_offset,
                    "EXCEPT_EXCEPTION",
                    "except Exception clause"
                ))
    
    # Check for pass in error handlers
    for n in ast.walk(node):
        if isinstance(n, ast.ExceptHandler):
            # Check if the body is just a pass statement
            if len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
                violations.append(Violation(
                    n.lineno, n.col_offset,
                    "PASS_IN_ERROR_HANDLER",
                    "pass in error handler"
                ))
    
    return violations

def check_class_standards(node: ast.ClassDef, filepath: str) -> List[Violation]:
    """Check class-specific standards."""
    violations = []
    
    # Check if it's a dataclass and if it's frozen
    is_dataclass = False
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
            is_dataclass = True
            # Check for frozen=True
            frozen_found = False
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if keyword.arg == 'frozen' and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                        frozen_found = True
            if not frozen_found:
                violations.append(Violation(
                    node.lineno, node.col_offset,
                    "NON_FROZEN_DATASET",
                    f"Dataclass {node.name} is not frozen"
                ))
    
    # If it's an error type (inherits from Exception), check if it's a frozen dataclass
    # We'll check base classes
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == 'Exception':
            if not is_dataclass:
                violations.append(Violation(
                    node.lineno, node.col_offset,
                    "ERROR_NOT_DATASET",
                    f"Error class {node.name} should be a frozen dataclass"
                ))
            else:
                # Already checked for frozen above
                pass
    
    return violations

def calculate_nesting_depth(node: ast.AST) -> int:
    """Calculate the maximum nesting depth of control structures in a node."""
    max_depth = 0
    
    def _visit(n, current_depth):
        nonlocal max_depth
        if isinstance(n, (ast.If, ast.For, ast.While, ast.With, ast.Try, 
                         ast.AsyncFor, ast.AsyncWith)):
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        for child in ast.iter_child_nodes(n):
            _visit(child, current_depth)
    
    _visit(node, 0)
    return max_depth

def main():
    """Main analysis function."""
    src_dir = os.path.join(os.path.dirname(__file__), 'src', 'dragonlight_router')
    if not os.path.exists(src_dir):
        print(f"Error: Source directory not found: {src_dir}")
        sys.exit(1)
    
    all_violations = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                violations = analyze_file(filepath)
                if violations:
                    all_violations.append((filepath, violations))
    
    # Print report
    for filepath, violations in all_violations:
        print(f"\n=== {filepath} ===")
        for v in violations:
            print(f"  Line {v.line}:{v.column} - {v.rule}: {v.description}")
    
    # Summary
    total_violations = sum(len(v) for _, v in all_violations)
    print(f"\nTotal files with violations: {len(all_violations)}")
    print(f"Total violations: {total_violations}")

if __name__ == '__main__':
    main()