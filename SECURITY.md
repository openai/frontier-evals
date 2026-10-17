# Security Policy

For a more in-depth look at our security policy, please check out our [Coordinated Vulnerability Disclosure Policy](https://openai.com/security/disclosure/).

Our PGP key is located [at this address](https://cdn.openai.com/security.txt).

## Scope Note for `frontier-evals`

This repository contains intentionally adversarial content used to measure frontier-model capability and safety:

- **EVMBench** — smart contract exploit replication tasks, with audit reports, vulnerability hints, and patch references.
- **SWE-Lancer** — freelance software-engineering tasks, some of which include security-relevant code paths.
- **PaperBench** — paper-replication tasks that may invoke offensive-security or red-teaming workflows.

Model behaviour on these inputs is the eval signal, not a vulnerability. Please do not file reports for adversarial content that the evals are designed to elicit.

The following ARE in scope and we welcome reports for them:

- Vulnerabilities in the eval harness code itself (the orchestration scripts under `project/*/runtime_*/`, shared utilities under `project/common/`, or the top-level `pyproject.toml` tooling chain).
- Supply-chain issues in pinned dependencies (`uv.lock`, `pyproject.toml`).
- Sandbox-escape paths in EVMBench's task runners or PaperBench's replication environment that would allow eval code to affect the host beyond its intended boundary.
- Accidental disclosure of PII, secrets, or non-public OpenAI material in committed task data.

Please follow the [Coordinated Vulnerability Disclosure Policy](https://openai.com/security/disclosure/) for all reports.
