# quickgrok

Can grokking be made to happen sooner? This project trains small networks on modular arithmetic (a + b mod p), where they memorise the training set quickly but only generalise much later ("grokking"). During training, hidden neurons are periodically fit with a regression. Neurons that fit well are replaced by trainable parametric modules, a jump towards the structure the network is converging to anyway.

## Setup

```bash
uv sync
```

Torch is installed from the CUDA 13.2 wheel index (see `pyproject.toml`). Run the tests with `uv run pytest`.

## Usage

See [docs/usage.md](docs/usage.md) for the command-line entry points, the Python API and how to plug the training hooks into your own PyTorch loop. Results are in `reports/`; a student-level walkthrough is in [intuition/README.md](intuition/README.md).
