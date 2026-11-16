# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

---

## Reporting a Vulnerability

We take the security and integrity of Dubb-in AI Studio very seriously, particularly regarding local execution safety, biometric voice security, and dependency hygiene.

If you discover a security vulnerability, please **DO NOT open a public issue**. Instead, follow these steps:

1. **Email Advisory**: Send a description of the vulnerability directly to the project maintainers via `contact@papytechnologies.com` or through a [Private Vulnerability Advisory](https://github.com/PapyTechnologies/dubb-in/security/advisories/new) on GitHub.
2. **Include Technical Details**:
   - Component affected (`engine.py`, `lipsync/`, `window.py`, etc.).
   - Proof-of-concept or reproducible steps.
   - Potential impact (e.g., path traversal, remote execution vector via malicious inputs, unhandled sub-process execution).
3. **Response Timeline**:
   - Initial acknowledgement: **Within 48 hours**.
   - Assessment and patch verification: **Within 7 days**.
   - Public CVE advisory / patched release: Following coordinated disclosure.

---

## Biometric Data & Voice Security

- **100% Offline Processing**: Dubb-in performs all speech recognition, synthesis, and facial inference locally on your hardware. No voice samples, video files, or biometric embeddings (`.dvf`) are transmitted over the internet or collected on external servers.
- **Biometric Integrity**: `.dvf` files contain mathematical embeddings (tensors) of voice acoustics rather than raw uncompressed biometric audio. They are encrypted at rest when user password protection is configured.
