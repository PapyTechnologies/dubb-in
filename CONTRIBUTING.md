# Contributing to Dubb-in AI Studio

Thank you for your interest in contributing to **Dubb-in AI Studio**! 🎙️

Dubb-in provides a **100% free, local, private, and sovereign solution** for AI-powered video dubbing, neural voice cloning, and lip synchronization without cloud subscriptions. Contributions from the open-source community are essential to maintaining and advancing this software.

---

## 📜 Code of Conduct

We are committed to providing a welcoming, safe, and collaborative environment.
- **Be respectful and constructive** in discussions, issues, and code reviews.
- **Support diversity and inclusion**: Help developers and creators of all skill levels.
- **Zero tolerance** for harassment, insults, discriminatory language, or unprofessional conduct.

---

## ⚖️ Ethical Voice Cloning Policy

Dubb-in utilizes biometric voice embeddings (`.dvf`) and neural synthesis models. By contributing to or using Dubb-in, you agree to uphold our ethical standards:
1. **Explicit Consent**: Cloning a human voice requires explicit, verifiable consent from the voice owner.
2. **Prohibited Applications**: Impersonation, fraud, deceptive deepfakes, harassment, non-consensual biometric profiling, or political disinformation are strictly forbidden.
3. **Traceability**: Generated biometric files (`.dvf`) embed neural watermarks and integrity metadata for auditability.

---

## 🐛 How to Report a Bug

Encountered an unexpected error or an audio artifact?
1. **Check Existing Issues**: Search the [Issue Tracker](https://github.com/PapyTechnologies/dubb-in/issues) first.
2. **Submit a Detailed Report**:
   - **OS & Environment**: Windows 10/11 x64, Python version (`3.10.x`).
   - **Hardware Specs**: GPU model, VRAM (`nvidia-smi`), CPU, RAM.
   - **Steps to Reproduce**: Clear minimal reproduction scenario.
   - **Logs**: Copy relevant terminal output or traceback.

---

## 💡 Proposing Features & Improvements

Have ideas for optimizing VRAM consumption, adding language pairs, or refining audio DSP?
1. Open a **Feature Request** on GitHub.
2. Detail the motivation, expected user experience, and technical approach.
3. Discuss architectural feasibility before writing extensive pull requests.

---

## 🛠️ Development Setup & Workflow

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/PapyTechnologies/dubb-in.git
   cd dubb-in
   ```

2. **Virtual Environment**:
   ```bash
   py -3.10 -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip "setuptools<70" wheel
   pip install -r requirements.txt
   ```

3. **Branching Discipline**:
   Always branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```

4. **Conventional Commits**:
   All commits must adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:
   - `feat:` New features or UI capabilities
   - `fix:` Bug fixes and corrections
   - `perf:` Performance and VRAM optimizations
   - `docs:` Documentation improvements
   - `refactor:` Code refactoring without behavioral changes
   - `chore:` Dependency bumps, CI updates, configuration

5. **Submitting a Pull Request**:
   - Ensure code passes Python syntax checks (`python -m py_compile`).
   - Fill out the PR template completely.
   - Link any related issue numbers (`Fixes #12`).

---

Thank you for helping keep Dubb-in autonomous, free, and open to all creators!
