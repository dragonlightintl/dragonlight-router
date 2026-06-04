# Dragonlight Router Codebase Audit Summary
## Dragonlight Coding Standards v2 Compliance Check

### Overview
- **Date**: Saturday, May 30, 2026
- **Repository**: /Users/coryflanigan/dragonlight-ops/dragonlight-router
- **Source Directory**: src/dragonlight_router/
- **Files Analyzed**: 34 Python files
- **Tools Used**: 
  - Custom static analysis script (function length, complexity, parameters, nesting, type annotations, assertions, dataclasses, error handling, mutable state)
  - Ruff (linting)
  - MyPy (type checking)

### Ruff Results (16 errors)
- **SIM105**: Use `contextlib.suppress(OSError)` instead of `try`-`except`-`pass` (2 instances)
- **SIM114**: Combine `if` branches using logical `or` operator (3 instances)
- **UP017**: Use `datetime.UTC` alias (2 instances)
- **B905**: `zip()` without explicit `strict=` parameter (1 instance)
- **F401**: Unused imports (4 instances)
- **UP035**: Import from `collections.abc` instead: `AsyncIterator` (1 instance)
- **E501**: Line too long (112 > 100) (1 instance)
- **C401**: Unnecessary generator (rewrite as set comprehension) (1 instance)
- **I001**: Import block un-sorted/un-formatted (1 instance)
- **F401**: Unused import (Router module) (1 instance)

### MyPy Results (13 errors)
- **type-arg**: Missing type arguments for generic type "dict" (8 instances)
- **no-any-return**: Returning Any from function declared to return specific type (3 instances)
- **import-untyped**: Library stubs not installed for "yaml" (1 instance)
- **assignment**: Incompatible types in assignment (expression has type "list[CatalogEntry] | BaseException", target has type "list[CatalogEntry]") (1 instance)
- **no-untyped-def**: Function missing type annotation for one or more parameters (1 instance)

### Custom Standards Analysis Results
**Total Files with Violations**: 21/34 (61.8%)
**Total Violations**: 173

#### Violation Breakdown by Category:

1. **Module-Level Mutable State**: 9 files
   - Variables: `__all__`, `logger`, `_router_lock`, `_router_instance`, `frozenset`

2. **Dataclass Standards**: 3 files
   - Non-frozen dataclasses: `BackendRegistry`, `BackendState`
   - Error classes not using frozen dataclasses

3. **Type Annotation Coverage**: 68 violations
   - Missing parameter type annotations (especially `self`): 42 instances
   - Missing return type annotations: 14 instances
   - Missing type arguments for generics: 12 instances

4. **Assertion Density (Advisory: 2/function)**: 41 violations
   - Functions with 0 assertions: 31 instances
   - Functions with 1 assertion: 10 instances

5. **Function Length (Hard Limit: 40 lines)**: 2 violations
   - `src/dragonlight_router/router.py`: 71 lines (function starting at line 82)
   - `src/dragonlight_router/selection/complexity.py`: 50 lines (function starting at line 34)

6. **Parameter Count (Hard Limit: 4)**: 4 violations
   - `src/dragonlight_router/router.py`: 6 parameters (function starting at line 154)
   - `src/dragonlight_router/caching/semantic.py`: 5 parameters (function starting at line 19)
   - `src/dragonlight_router/caching/simple.py`: 5 parameters (function starting at line 50)
   - `src/dragonlight_router/core/types.py`: 5 parameters (function starting at line 191)

7. **Nesting Depth (Hard Limit: 3)**: 1 violation
   - `src/dragonlight_router/roles/matrix.py`: Depth 4 (line 55)

8. **Error Handling Patterns**: 22 violations
   - `except Exception` clauses: 10 instances
   - Bare `except`: 2 instances
   - `pass` in error handlers: 6 instances
   - Try-except-pass patterns: 4 instances (SIM105)

9. **Other Standards**: 24 violations
   - Missing frozen dataclass decorator: 3 instances
   - Combined if branches (SIM114): 3 instances
   - Unnecessary generator (C401): 1 instance
   - Import sorting issues (I001): 1 instance
   - Line length issues (E501): 1 instance
   - Unused imports (F401): 6 instances
   - Datetime.UTC usage (UP017): 2 instances
   - Collections.abc import (UP035): 1 instance
   - Zip strict parameter (B905): 1 instance
   - Unused import in router.py: 1 instance

### Key Findings
1. **Most Common Issues**: Missing type annotations (especially for `self` parameters) and low assertion density
2. **Critical Violations**: Function length exceeding 40 lines in router.py and complexity.py
3. **Error Handling**: Widespread use of `except Exception` and `pass` in error handlers
4. **Immutability**: Numerous module-level mutable variables and non-frozen dataclasses
5. **Type Safety**: Multiple MyPy errors regarding missing type arguments and Any returns

### Recommendations
1. **Immediate Actions**:
   - Fix function length violations by breaking down large functions
   - Add missing type annotations for all public signatures
   - Replace `except Exception` with specific exception types
   - Eliminate `pass` in error handlers
   - Convert dataclasses to frozen where appropriate
   - Remove module-level mutable state or justify its use

2. **Medium Term**:
   - Increase assertion density to advisory minimum of 2 per function
   - Reduce parameter counts to maximum of 4
   - Decrease nesting depth to maximum of 3
   - Address all Ruff and MyPy errors

3. **Process Improvements**:
   - Add pre-commit hooks for Ruff and MyPy
   - Implement automated checks for function length and complexity
   - Create templates for standard error handling patterns
   - Establish code review checklist based on these standards

### Conclusion
The codebase shows significant deviations from Dragonlight Coding Standards v2, particularly in type safety, error handling, and function complexity. Addressing these issues will improve maintainability, reduce bugs, and align the codebase with established best practices.