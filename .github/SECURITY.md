# Security

Please do not post credentials or private dataset material in public issues.
Use GitHub's private vulnerability reporting if it is enabled on this
repository: **Security → Report a vulnerability**. Otherwise, open an issue
requesting a private contact channel without including sensitive details.

Only load checkpoints you trust. This project uses PyTorch's
`weights_only=True` loader, but does not treat arbitrary model files as safe.

The repository does not distribute third-party training images or model
binaries. Dataset licensing and review status are recorded in
`results/dataset/source_review.json`.
