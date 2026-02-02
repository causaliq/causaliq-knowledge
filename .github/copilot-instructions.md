# CausalIQ Knowledge - Copilot Instructions

This file provides project-specific instructions for GitHub Copilot when
working on the causaliq-knowledge repository.

For ecosystem-wide development standards, see:
https://github.com/causaliq/causaliq/blob/main/LLM_DEVELOPMENT_GUIDE.md

## 🎯 Project Overview

causaliq-knowledge provides LLM and human knowledge integration for causal
discovery. It includes LLM clients for multiple providers, response caching,
and tools for graph generation from variable specifications.

## 📋 Critical Standards

### Line Length
- **79 characters maximum** - This is a CRITICAL CausalIQ standard
- Write to 79 chars from the start, never retrospectively fix long lines
- Use Black with `line-length = 79` configuration

### Code Style
- All code must pass: `black`, `isort`, `flake8`, `mypy`
- Complete type hints for all function parameters and return values
- Google-style docstrings with examples
- British English spelling in all documentation and comments

### Testing Standards
- **Use pytest exclusively** - no unittest classes or methods
- **Prefer individual test functions** over test classes
- **One-line comment before each test function** for VS Code outline:
  ```python
  # Test BIC score calculation returns correct float type.
  def test_bic_score_returns_float() -> None:
      pass
  ```
- Target 100% test coverage
- Test categories:
  - **Unit tests**: Pure logic, no filesystem/external dependencies
  - **Functional tests**: Filesystem access, local resources
  - **Integration tests**: Remote services, network dependencies

### Import Organisation
- Group imports: stdlib, third-party, local
- Alphabetical within groups
- Single line imports preferred

## 🏗️ Environment Management

**CRITICAL**: Never use `pip install` directly.

```powershell
# Setup environment
.\scripts\setup-env.ps1

# Activate environment
.\scripts\activate.ps1

# Run CI checks
.\scripts\check_ci.ps1
```

For sequential terminal commands, use semicolon:
```powershell
.\scripts\activate.ps1; python -m pytest tests/
```

## 📚 Documentation

### Single Source of Truth
- `docs/roadmap.md` - All feature specifications and progress tracking
- `README.md` - Project gateway, brief overview
- Update roadmap.md when completing features

### Docstring Format
```python
def optimise_graph_structure(data: pd.DataFrame) -> Dict[str, List[str]]:
    """Optimise causal graph structure using statistical methods.

    This function analyses the input data to discover optimal causal
    relationships between variables.

    Args:
        data: Input dataset for analysis.

    Returns:
        Dictionary representing optimised graph structure.

    Raises:
        ValueError: If data contains invalid values.
    """
    pass
```

## 🔄 Development Workflow

### Commit Standards
- Small, focused commits (50-100 lines ideally)
- Each commit should pass all CI checks
- **100% test coverage required before each commit**
- Update roadmap.md in same commit as feature implementation

### Change Communication
For every proposed change, explain:
- **What**: Exactly what code is being modified
- **Why**: The problem being solved
- **Impact**: What other functionality might be affected

## ⚠️ Common Violations to Avoid

### Testing
- ❌ Using test classes when individual functions suffice
- ❌ Missing one-line test comments
- ❌ Putting filesystem tests in unit tests (belongs in functional)

### Code Quality
- ❌ Writing long lines and fixing retrospectively
- ❌ Missing type hints or incomplete docstrings
- ❌ Using American spelling (use British: optimise, colour, analyse)
- ❌ Unused imports or variables

### Environment
- ❌ Using `pip install` directly
- ❌ Not running CI checks before claiming completion
- ❌ Starting new terminal sessions without venv activation

## 📁 Project Structure

```
src/causaliq_knowledge/
├── __init__.py
├── base.py              # KnowledgeProvider abstract interface
├── models.py            # EdgeKnowledge, EdgeDirection models
├── cli.py               # Command-line interface
├── cache/               # Response caching (TokenCache)
├── graph/               # Graph generation from specs
│   ├── models.py        # ModelSpec, VariableSpec
│   └── loader.py        # ModelLoader
└── llm/                 # LLM client implementations
    ├── base_client.py
    ├── prompts.py
    ├── provider.py
    └── *_client.py      # Provider-specific clients
```

## 🧪 Running Tests

```powershell
# Run all tests
.\scripts\activate.ps1; python -m pytest tests/ -v

# Run specific test file
.\scripts\activate.ps1; python -m pytest tests/unit/graph/test_models.py -v

# Run with coverage
.\scripts\activate.ps1; python -m pytest tests/ --cov=src/causaliq_knowledge
```
