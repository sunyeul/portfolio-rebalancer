# Evaluation Framework v2

IPS Pilot v2 evaluates layers and assets with the same inspection frame. A layer is a first-class portfolio unit such as `core`, `satellite`, or `experiment`; an asset is a ticker inside one layer. The engine does not emit automatic order instructions.

## Common Frame

Each evaluation unit is inspected with:

- Weight: current weight, target gap, and layer-internal weight for assets.
- Return: period return, CAGR, benchmark return, and excess return.
- Risk: MDD, volatility, concentration, and risk contribution.
- Efficiency: return/MDD, CAGR/MDD, Sharpe, and Sortino.
- Thesis: valid, watch, broken, or unknown.
- Operations: low, medium, or high management burden.
- Status: `OK`, `Watch`, `Review`, or `Action`.

Layer and asset differences are expressed through benchmark, target, threshold, and limit parameters. The output vocabulary is an inspection vocabulary, not a trading vocabulary.

## Layers

- `core`: long-term market exposure. Normal drawdown or weak short-term efficiency is not, by itself, a reason to reduce exposure.
- `satellite`: return-seeking exposure. Weak data, thesis uncertainty, overlap, and management burden are stricter review inputs.
- `experiment`: strategy validation exposure. Rule clarity, MDD limits, and position-size limits matter before return chasing.

## Status Semantics

- `OK`: no material threshold, data, thesis, or burden warning.
- `Watch`: soft warning such as thesis `watch`/`unknown`, target gap outside tolerance, low efficiency, or elevated burden.
- `Review`: hard threshold breach, risk overage, broken thesis, or insufficient data that blocks judgment.
- `Action`: thesis is broken and at least one hard limit is breached.

`Action` means "inspect possible intervention"; it does not mean buy or sell.

## Guardrails

- Prefer regular-purchase adjustment over immediate trades.
- Treat immediate buying and selling as exceptional.
- Treat missing or low-quality data as a reason to observe or review.
- For satellites and experiments, require stronger thesis and burden checks before increasing exposure.
- Never interpret the CLI output as a market order instruction.
