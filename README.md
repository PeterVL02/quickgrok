# quickgrok

Can grokking be made to happen sooner? This project trains small networks on modular arithmetic (a + b mod p), where they memorise the training set quickly but only generalise much later ("grokking"). During training, hidden neurons are periodically fit with a regression. Neurons that fit well are replaced by trainable parametric modules, a jump towards the structure the network is converging to anyway.

## Setup

```bash
uv sync
```

Torch is installed from the CUDA 13.2 wheel index (see `pyproject.toml`). Run the tests with `uv run pytest`.

## Usage

See [docs/usage.md](docs/usage.md) for the command-line entry points, the Python API and how to plug the training hooks into your own PyTorch loop. Results are in `reports/`; an intuition-first walkthrough for ML practitioners is in [intuition/README.md](intuition/README.md).

## Findings

The project is concluded. A pre-registered kill test failed, so what remains is a blog post and repo cleanup. The full summary, with links to each report, is in [reports/README.md](reports/README.md). In short:

- **Hidden neurons become structured early.** Measured from training data only, they turn into simple waves long before test accuracy moves.
- **Swapping neurons for trainable wave modules speeds grokking,** but mostly because waves are the right prior for this task. Task-blind replacement shapes stall generalisation.
- **A tuned learning rate and weight decay shrink the plateau far more than any method** (up to ~18×; this is known in the literature). On a tuned baseline, step savings remain (nowcasting 37%, wave modules 45%), but they shrink at the fastest settings, and only nowcasting saves wall-clock time.
- **Train-only monitors don't forecast *when* grokking happens better than cheap baselines.** They do flag *whether* a run will grok, with no false alarms on non-grokking controls.

## License

MIT, see [LICENSE](LICENSE).
