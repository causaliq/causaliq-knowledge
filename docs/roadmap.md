# CausalIQ Knowledge - Development Roadmap

**Last updated**: February 20, 2026  

This project roadmap fits into the [overall ecosystem roadmap](https://causaliq.org/projects/ecosystem_roadmap/)

## 🚧  Under development

*No releases currently under active development.*

---
   
## ✅ Previous Releases

- **v0.1.0 - Foundation LLM** [January 2026]: Foundation release establishing LLM client infrastructure for causal graph
generation.

- **v0.2.0 - Additional LLMs** [January 2026]: Expanded LLM provider support from 2 to 7 providers.

- **v0.3.0 - LLM Caching** [January 2026]: SQLite-based response caching with CLI tools for cache management.

- **v0.4.0 - Graph Generation** [February 2026]: CLI tools for LLM-generated causal graphs.

- **v0.5.0 - Workflow Integration** [February 2026]: Integration into CausalIQ Workflows including writing results to cache.

## 🛣️ Upcoming Releases (speculative)

### Release v0.6.0 - Statistical Fusion

Support knowledge requirements for fusing LLM knowledge and statistical graph averaging.

**Scope**:

- graph generation returns separate existence and orientation probabilities
- integrate with latest infrastructure to record graph ede probabilities


### Release v0.7.0 - LLM Provider Cost Tracking

Query LLM provider APIs for usage and cost statistics.

**Scope:**

- Provider usage API clients
- CLI `llm costs` command
- Cache savings analysis

### Release v0.8.0 - Enhanced LLM Context

Background literature supplied to LLMs

### Release v0.9.0 - Structure Learning

Integration with structure learning algorithms


### Release v0.10.0 - Legacy Reference

Support for deriving knowledge from reference networks and migration of functionality from legacy discovery repo


