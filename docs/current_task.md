# Refactor causaliq-knowledge codebase to meet project coding policies

Please provide a plan to refactor the causaliq-knowledge code base to
ensure coding policies are adhered to, in particular that 
@scripts/check_code_rules.py reports no breaches.

Please stage the work into separate commits if you consider that desirable.
## Constraints

- **Behaviour must stay byte-identical**
- **Preserve import binding sites**
- **No behaviour-free bulk test rewrites**: add focused tests
  only where coverage gaps appear

## Quality Gates (as in `scripts/check_ci.ps1`)

- `black --check src tests`
- `isort --check-only src tests`
- `flake8 src tests`
- `mypy src/`
- `python scripts/check_code_rules.py --strict src/causaliq_knowledge`