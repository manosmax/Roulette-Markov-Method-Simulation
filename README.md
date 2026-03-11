# M/M/3 Queue Simulation – Roulette Wheel Method

A discrete-event simulation of an **M/M/3 Erlang loss system** using a multiplicative congruential pseudo-random number generator (LCG) and the roulette wheel selection method.

---

## Overview

This simulation models a telephone trunk system with **3 servers (trunks)** and **no waiting room** — calls that arrive when all servers are busy are lost. It estimates the **call blocking probability B** and compares it against the theoretical **Erlang B formula** value.

---

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ALPHA`   | 2.2   | Traffic load (Erlangs) |
| `S`       | 3     | Number of servers |
| `K`       | 3     | LCG multiplier |
| `M`       | 128   | LCG modulus (2⁷) |
| `STEPS`   | 32    | Steps per simulation run (= M/4) |
| `ERLANG_B`| 0.24  | Theoretical E₃(2.2) blocking probability |

---

## How It Works

### 1. Pseudo-Random Number Generator
A **multiplicative LCG** generates the random sequence:

```
xₙ = k · x₍ₙ₋₁₎  mod  M
```

With `k = 3`, `M = 128`, the generator period is `M/4 = 32`, which matches the number of simulation steps.

### 2. Roulette Wheel Event Selection
At each step, a uniform random number `Y = xₙ / M ∈ (0, 1)` determines whether the next event is an **arrival** or a **termination**:

- **Arrival** if `Y < α / (α + n)`
- **Termination** if `Y ≥ α / (α + n)`

where `n` is the current number of active calls.

### 3. Blocking Logic
- If an arrival occurs and `n < S` → call is **served**, a server is marked busy.
- If an arrival occurs and `n = S` → call is **lost** (blocked).
- Blocking probability: `B = lost_calls / total_arrivals`

---

## Usage

### Requirements

```bash
pip install matplotlib numpy
```

### Run

```bash
python mm3_simulation.py
```

---

## Output

The script runs in two parts:

**Part A – 4 measurements** (seeds: 3, 5, 7, 11)
- Verbose step-by-step table printed to console
- Bar chart saved as `blocking_4measurements.png`

**Part B – 8 measurements** (seeds: 3, 5, 7, 11, 13, 17, 19, 23)
- Summary table printed to console
- Bar chart saved as `blocking_8measurements.png`

**Comparison chart** saved as `blocking_comparison_4vs8.png`

### Example Console Summary

```
════════════════════════════════════════════════════════════
  SUMMARY  (8 measurements)
════════════════════════════════════════════════════════════
  Measurement   Seed     Arrivals   Lost       B
  ────────────────────────────────────────────────────────
  B1            3        ...        ...        xx.xx%
  ...
  ────────────────────────────────────────────────────────
  Mean value                                  xx.xx%
  Erlang B  E₃(2.2)                           24.00%
  Deviation                                    x.xx%
════════════════════════════════════════════════════════════
```

---

## Project Structure

```
mm3_simulation.py          # Main simulation script
blocking_4measurements.png # Output chart – 4 runs
blocking_8measurements.png # Output chart – 8 runs
blocking_comparison_4vs8.png # Side-by-side comparison chart
```

---

## Theory

This simulation verifies the **Erlang B (M/M/s/s) formula**:

$$E_s(\alpha) = \frac{\alpha^s / s!}{\sum_{k=0}^{s} \alpha^k / k!}$$

For `s = 3`, `α = 2.2 erl`: **E₃(2.2) ≈ 24%**

---

## Seeds Used

All seeds are odd prime numbers to ensure good LCG behavior:

`3, 5, 7, 11, 13, 17, 19, 23`
