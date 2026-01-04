# causaliq-knowledge

[![Python Support](https://img.shields.io/pypi/pyversions/zenodo-sync.svg)](https://pypi.org/project/zenodo-sync/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The CausalIQ Knowledge project represents a novel approach to causal discovery by combining the traditional statistical structure learning algorithms with the contextual understanding and reasoning capabilities of Large Language Models. This integration enables more interpretable, domain-aware, and human-friendly causal discovery workflows. It is part of the [CausalIQ ecosystem](https://causaliq.org/) for intelligent causal discovery.

## Status

🚧 **Active Development** - this repository is currently in active development, which involves:

- adding new knowledge features, in particular knowledge from LLMs
- migrating functionality which provides knowledge based on standard reference networks from the legacy monolithic discovery repo
- ensure CausalIQ development standards are met


## Features

Under development:

- **Release v0.1.0 - Foundation LLM**: simple LLM queries to 1 or 2 LLMs about edge existence and orientation to support graph averaging

Currently implemented releases:

- none

Planned 

- **Release v0.2.0 - Additional LLMs**: support for more LLMs
- **Release v0.3.0 - LLM Caching**: caching of LLM query and responses
- **Release v0.4.0 - LLM Context**: variable/role/literature etc context
- **Release v0.5.0 - Algorithm integration**: integration into structure learning algorithms
- **Release v0.6.0 - Legacy Reference**: support for legacy approaches of deriving knowledge from reference networks

## Upcoming Key Innovations

### 🧠 LLMs support Causal Discovery and Inference
- initially LLM will work with **graph averaging** to resolve uncertain edges (use entropy to decide edges with uncertain existence or direction)
- integration into **structure learning** algorithms to provide knowledge for "uncertain" areas of the graph
- specification of domain constraints using **natural language**
- LLMs analyse learning process and errors to **suggest improved algorithms**
- LLMs used to preprocess **text and visual data** so they can be used as inputs to structure learning
- LLMs convert natural language to **causal queries**

### 🪟 Transparency and interpretability
- LLMs **interpret structure learning process** and outputs, including their uncertainties
- LLMs **interpret causal inference** results including uncertainties

### 🔒 Stability and reproducibility
- **cache queries and responses** so that experiments are stable and repeatable even if LLMs themselves are not
- **stable randomisation** of e.g. data sub-sampling

### 💰 Efficient use of LLM resources (important as an independent researcher)
- **cache queries and results** so that knowledge can be re-used
- evaluation and development of **simple context-adapted LLMs**

## Upcoming Integration with CausalIQ Ecosystem

- 🔍 CausalIQ Discovery makes use of this package to learn more accurate graphs.
- 🧪 CausalIQ Analysis uses this package to explain the learning process, intelligently combine end explain results.
- 🔮 CausalIq Predict uses this package to explain predictions made by learnt models.


---

**Supported Python Versions**: 3.9, 3.10, 3.11, 3.12  
**Default Python Version**: 3.11
