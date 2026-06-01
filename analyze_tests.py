import ast
import os
import sys

def analyze_test_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    try:
        tree = ast.parse(content)
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []
    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            test_functions.append(node)
        elif isinstance(node, ast.AsyncFunctionDef) and node.name.startswith('test_'):
            test_functions.append(node)
    return test_functions

def function_body_empty_or_only_pass_or_docstring(node):
    # Check if the function body is empty or only contains pass, return, or docstring
    if not node.body:
        return True
    # If there's only one statement and it's a pass or a return or a docstring (string literal)
    if len(node.body) == 1:
        stmt = node.body[0]
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, ast.Return):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            return True
    return False

def function_body_only_assert_true(node):
    # Check if the function body has only one assert statement that asserts True
    # We'll look for an assert statement that compares to True
    # This is a simple check: we look for exactly one assert statement and its test is a constant True
    # But note: there could be a docstring first.
    # Let's collect all non-docstring statements.
    stmts = []
    for stmt in node.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue  # skip docstring
        stmts.append(stmt)
    if len(stmts) == 1 and isinstance(stmts[0], ast.Assert):
        # Check if the test is a constant True
        test = stmts[0].test
        if isinstance(test, ast.Constant) and test.value is True:
            return True
        # For older Python versions, NameConstant was used (but we are on 3.14, so we can ignore)
        # However, to be safe, we can check for ast.NameConstant if it exists (for compatibility)
        if hasattr(ast, 'NameConstant'):
            if isinstance(test, ast.NameConstant) and test.value is True:
                return True
    return False

def function_missing_assertions(node):
    # Check if there is no assert statement and no pytest.raises context manager
    # We'll look for any Assert or any With item that is pytest.raises
    has_assert = False
    has_pytest_raises = False
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            has_assert = True
        if isinstance(n, ast.With):
            for item in n.items:
                # Check if the context manager is pytest.raises
                if isinstance(item.context_expr, ast.Call):
                    if hasattr(item.context_expr.func, 'attr') and item.context_expr.func.attr == 'raises':
                        # Check if it's from pytest
                        if isinstance(item.context_expr.func.value, ast.Name) and item.context_expr.func.value.id == 'pytest':
                            has_pytest_raises = True
                    # Also check for `raises` from pytest.imported as `raises` but unlikely
                # Also check for `with pytest.raises(...) as ...`
    return not (has_assert or has_pytest_raises)

def main():
    test_dir = 'tests'
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    total_test_functions = 0
    empty_body_functions = []
    only_assert_true_functions = []
    missing_assertion_functions = []
    for file in test_files:
        functions = analyze_test_file(file)
        total_test_functions += len(functions)
        for node in functions:
            if function_body_empty_or_only_pass_or_docstring(node):
                empty_body_functions.append((file, node.name))
            if function_body_only_assert_true(node):
                only_assert_true_functions.append((file, node.name))
            if function_missing_assertions(node):
                missing_assertion_functions.append((file, node.name))
    print(f"Total test functions: {total_test_functions}")
    print(f"\nEmpty body functions (only pass/return/docstring): {len(empty_body_functions)}")
    for file, name in empty_body_functions:
        print(f"  {file}:{name}")
    print(f"\nFunctions with only `assert True`: {len(only_assert_true_functions)}")
    for file, name in only_assert_true_functions:
        print(f"  {file}:{name}")
    print(f"\nFunctions missing assertions (no assert or pytest.raises): {len(missing_assertion_functions)}")
    for file, name in missing_assertion_functions:
        print(f"  {file}:{name}")

    # Check for Hypothesis tests
    print("\n=== Checking for Hypothesis (property-based) tests ===")
    hypothesis_tests = []
    for file in test_files:
        with open(file, 'r') as f:
            content = f.read()
        if '@given' in content or 'from hypothesis' in content or 'import hypothesis' in content:
            hypothesis_tests.append(file)
    if hypothesis_tests:
        print(f"Found Hypothesis tests in {len(hypothesis_tests)} file(s):")
        for f in hypothesis_tests:
            print(f"  {f}")
    else:
        print("No Hypothesis tests found.")

    # Check for test files and missing test files for source modules
    print("\n=== Checking source modules for corresponding test files ===")
    src_dir = 'src'
    source_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                source_files.append(os.path.join(root, file))
    # We'll consider a test file exists if there is a test file in tests/unit with name test_<module_name>.py
    # where module_name is the relative path from src/dragonlight_router to the file without .py, with / replaced by _
    missing_test_files = []
    existing_test_files = []
    for sf in source_files:
        # Skip __init__.py files? We'll include them for now.
        rel_path = os.path.relpath(sf, src_dir)
        # Remove the .py extension
        if rel_path.endswith('.py'):
            rel_path = rel_path[:-3]
        # Convert path to test name: replace / with _
        test_name = rel_path.replace('/', '_')
        # The test file should be tests/unit/test_<test_name>.py
        expected_test_file = f"tests/unit/test_{test_name}.py"
        if not os.path.exists(expected_test_file):
            missing_test_files.append((sf, expected_test_file))
        else:
            existing_test_files.append((sf, expected_test_file))
    print(f"Source files: {len(source_files)}")
    print(f"Source files with corresponding test files: {len(existing_test_files)}")
    print(f"Source files MISSING corresponding test files: {len(missing_test_files)}")
    if missing_test_files:
        print("Missing test files for:")
        for sf, expected in missing_test_files:
            print(f"  {sf} -> expected {expected}")

    # Also check for any test files that don't correspond to a source file (optional)
    print("\n=== Checking for test files without corresponding source ===")
    # We'll skip this for brevity, but we can do it if needed.

if __name__ == '__main__':
    main()