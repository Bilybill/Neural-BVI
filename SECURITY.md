# Security

Research code is not designed as a network-facing service. Never load an untrusted PyTorch checkpoint or prepared artifact: legacy reproduction paths use pickle-backed `torch.load(..., weights_only=False)` and may execute code. Prefer plain arrays or weights-only state dictionaries for new interfaces.

Report vulnerabilities privately to yonghao2025@hnu.edu.cn with a minimal reproduction and affected version. Do not include credentials, private measured data or executable payloads in a public issue. No guaranteed response time or security support period is promised.

The source packer excludes raw data, weights, local environments and credentials by an allowlist. Its local scan is not a comprehensive security audit. Inspect the release manifest before publishing and enable GitHub secret scanning where available.
