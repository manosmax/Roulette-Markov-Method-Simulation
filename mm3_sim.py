import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
# ΠΑΡΑΜΕΤΡΟΙ
# ──────────────────────────────────────────────
ALPHA   = 2.2        # φορτίο κίνησης (erl)
S       = 3          # αριθμός servers
K       = 3          # πολλαπλασιαστής LCG
M       = 128
STEPS   = 32         # βήματα ανά προσομοίωση
ERLANG_B = 0.24      # E3(2.2) θεωρητική τιμή

# Seeds για 8 τρεξίματα (όλοι πρώτοι)
SEEDS = [3, 5, 7, 11, 13, 17, 19, 23]


# ──────────────────────────────────────────────
# 1. ΓΕΝΝΗΤΡΙΑ ΤΥΧΑΙΩΝ ΑΡΙΘΜΩΝ
#    Πολλαπλασιαστική μέθοδος υπολοίπου
#    xn = k * x(n-1)  mod  M
# ──────────────────────────────────────────────
def generate_rng(seed: int, count: int = STEPS) -> list[int]:
    nums, x = [], seed
    for _ in range(count):
        x = (K * x) % M
        nums.append(x)
    return nums


# ──────────────────────────────────────────────
# 2. ΠΥΡΗΝΑΣ ΠΡΟΣΟΜΟΙΩΣΗΣ
# ──────────────────────────────────────────────
def simulate(seed: int, verbose: bool = False) -> dict:

    rng_ints = generate_rng(seed)
    n = 0               # τρέχουσα κατάσταση (αριθμός κλήσεων)
    trunks = [0] * S    # κατάσταση κάθε server (0=ελεύθερος, 1=κατειλημμένος)

    log = []            # αρχείο καταγραφής βημάτων
    total_arrivals = 0
    lost_calls     = 0

    if verbose:
        print(f"\n{'═'*120}")
        print(f"  ΠΡΟΣΟΜΟΙΩΣΗ  |  Seed x₀ = {seed}  |  M = {M}  |  k = {K}  |  α = {ALPHA}  |  s = {S}")
        print(f"{'═'*120}")
        print(f"{'Α/Α':>6} {'Random':>10} {'Random(0,1)':>16} {'P_άφ':>12} {'P_τερμ':>9} {'Άφιξη/Τερμ':>18} "
              f"{'Trunks(0 0 0)':>15} {'Service(NAI/OXI)':>15}")
        print(f"{'─'*120}")

    for step, xi in enumerate(rng_ints, 1):
        Y      = xi / M
        P_arr  = ALPHA / (ALPHA + n)
        P_term = n     / (ALPHA + n)

        if Y < P_arr:                       # ── ΑΦΙΞΗ ──
            total_arrivals += 1
            if n < S:
                n += 1
                trunks[trunks.index(0)] = 1
                event, service = "ΑΦΙΞΗ", "ΝΑΙ"
            else:                           # χαμένη κλήση
                lost_calls += 1
                event, service = "ΑΦΙΞΗ", "ΟΧΙ"
        else:                               # ── ΤΕΡΜΑΤΙΣΜΟΣ ──
            if n > 0:
                n -= 1
                # ελευθερώνουμε τον τελευταίο κατειλημμένο server
                idx = len(trunks) - 1 - trunks[::-1].index(1)
                trunks[idx] = 0
            event, service = "ΤΕΡΜ.", "—"

        trunk_str = f"({' '.join(map(str, trunks))})"
        log.append({
            "step": step, "xi": xi, "Y": round(Y, 4),
            "P_arr": round(P_arr, 4), "P_term": round(P_term, 4),
            "event": event, "trunks": trunk_str, "service": service,
            "state": n,
        })

        if verbose:
            marker = "" if service == "ΟΧΙ" else ""
            print(f"{step:>4} {xi:>10} {Y:>13.4f} {P_arr:>18.4f} {P_term:>8.4f} "
                  f"{event:>15} {trunk_str:>14} {service:>15}{marker}")

    B = lost_calls / total_arrivals if total_arrivals > 0 else 0.0

    if verbose:
        print(f"{'─'*120}")
        print(f"  Αφίξεις: {total_arrivals}  |  Χαμένες: {lost_calls}  |  "
              f"B = {lost_calls}/{total_arrivals} = {B*100:.2f}%  "
              f"(Θεωρητικό: {ERLANG_B*100:.0f}%)\n")

    return {
        "seed": seed,
        "log": log,
        "total_arrivals": total_arrivals,
        "lost_calls": lost_calls,
        "B": B,
    }


# ──────────────────────────────────────────────
# 3. ΠΟΛΛΑΠΛΕΣ ΜΕΤΡΗΣΕΙΣ
# ──────────────────────────────────────────────
def run_all(seeds: list[int], verbose: bool = True) -> list[dict]:
    return [simulate(s, verbose=verbose) for s in seeds]


# ──────────────────────────────────────────────
# 4. ΓΡΑΦΗΜΑΤΑ  (95% Διάστημα Εμπιστοσύνης)
# ──────────────────────────────────────────────
# Πίνακας t_{0.05/2}^{n-1}  (df = n-1, από 1 έως 12)
# Τιμές από τον δοθέντα πίνακα κρίσιμων τιμών t
_T_TABLE = {
    1:  12.706,
    2:   4.303,
    3:   3.182,
    4:   2.776,
    5:   2.571,
    6:   2.447,
    7:   2.365,
    8:   2.306,
    9:   2.262,
    10:  2.228,
    11:  2.201,
    12:  2.179,
}


def _ci95(values: list) -> tuple:
    """Επιστρέφει (mean, lower_bound, upper_bound) για 95% CI.
    Χρησιμοποιεί τον πίνακα t_{0.05/2}^{n-1}."""
    n      = len(values)
    mean   = np.mean(values)
    se     = np.std(values, ddof=1) / np.sqrt(n)
    df     = n - 1
    if df not in _T_TABLE:
        raise ValueError(f"df={df} εκτός πίνακα (1–12). Προσθέστε την τιμή.")
    t_crit = _T_TABLE[df]
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def _draw_ci_axes(ax, results: list[dict], subtitle: str):
    """Κοινή λογική σχεδίασης CI για έναν άξονα."""
    b_vals         = [r["B"] * 100 for r in results]
    mean_B, lo, hi = _ci95(b_vals)
    n              = len(results)

    # σημείο + error bar
    ax.errorbar(
        x=ALPHA, y=mean_B,
        yerr=[[mean_B - lo], [hi - mean_B]],
        fmt="o", color="black",
        markersize=8, linewidth=1.5,
        capsize=6, capthick=1.5,
        zorder=5,
    )

    # ετικέτες τιμών δεξιά
    offset_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02 if ax.get_xlim()[0] != 0 else 0.3
    for val, label in [(hi, f"{hi:.6f}"),
                       (mean_B, f"{mean_B:.6f}"),
                       (lo, f"{lo:.6f}")]:
        ax.text(ALPHA + 0.3, val, label, va="center", ha="left", fontsize=9.5)

    # θεωρητική γραμμή Erlang B
    ax.axhline(ERLANG_B * 100, color="red", linestyle="--",
               linewidth=1.4, label=f"Erlang B  E₃(2.2) = {ERLANG_B*100:.0f}%", zorder=3)

    # άξονες
    ax.set_xlim(ALPHA - 5, ALPHA + 5)
    pad = max(5, (hi - lo) * 0.4)
    ax.set_ylim(max(0, lo - pad), hi + pad)
    ax.set_xlabel("Φορτίο Κίνησης α (erl)", fontsize=10)
    ax.set_ylabel("Πιθανότητα Απώλειας B (%)", fontsize=10)
    ax.set_title(
        f"{subtitle}\n"
        f"95% Διάστημα Εμπιστοσύνης  ({n} μετρήσεις)  |  M={M}, k={K}",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color="gray")
    ax.xaxis.grid(True, linestyle="--", alpha=0.45, color="gray")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.85)


def plot_results(results: list[dict], title_suffix: str = ""):
    """Ένα διάγραμμα 95% CI για n μετρήσεις."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(ALPHA - 5, ALPHA + 5)   # ορισμός πρώτα ώστε το offset να είναι σωστό
    _draw_ci_axes(ax, results,
                  f"Προσομοίωση M/M/3 – Μέθοδος Ρουλέτας{title_suffix}")
    plt.tight_layout()
    plt.show()


def plot_comparison(results4: list[dict], results8: list[dict]):
    """Δύο διαγράμματα 95% CI δίπλα-δίπλα (4 vs 8 μετρήσεις)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Σύγκριση 4 vs 8 Μετρήσεων – M/M/3  |  α = {ALPHA} erl  |  "
        f"E₃(2.2) = {ERLANG_B*100:.0f}%",
        fontsize=12, fontweight="bold",
    )
    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xlim(ALPHA - 5, ALPHA + 5)

    _draw_ci_axes(axes[0], results4, "4 Μετρήσεις")
    _draw_ci_axes(axes[1], results8, "8 Μετρήσεις")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# 5. ΕΚΤΥΠΩΣΗ ΣΥΝΟΨΗΣ
# ──────────────────────────────────────────────
def print_summary(results: list[dict]):
    n = len(results)
    b_vals = [r["B"] for r in results]
    mean_B = sum(b_vals) / n

    print(f"\n{'═'*60}")
    print(f"  ΣΥΝΟΨΗ  ({n} μετρήσεις)")
    print(f"{'═'*60}")
    print(f"  {'Μέτρηση':<12} {'Seed':<8} {'Αφίξεις':<10} {'Χαμένες':<10} {'B':>8}")
    print(f"  {'─'*56}")
    for i, r in enumerate(results):
        print(f"  B{i+1:<11} {r['seed']:<8} {r['total_arrivals']:<10} "
              f"{r['lost_calls']:<10} {r['B']*100:>7.2f}%")
    print(f"  {'─'*56}")
    print(f"  {'Μέση τιμή':<30} {mean_B*100:>7.2f}%")
    print(f"  {'Erlang B  E₃(2.2)':<30} {ERLANG_B*100:>7.2f}%")
    print(f"  {'Απόκλιση':<30} {abs(mean_B - ERLANG_B)*100:>7.2f}%")
    print(f"{'═'*60}\n")


# ──────────────────────────────────────────────
# ΚΥΡΙΟ ΠΡΟΓΡΑΜΜΑ
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  Προσομοίωση M/M/3  –  Μέθοδος Ρουλέτας")
    print(f"  α = {ALPHA} erl  |  s = {S}  |  M = {M}  |  k = {K}")
    print("█"*60)



    # ── 4 Μετρήσεις (verbose) ──
    print("\n" + "─"*60)
    print("  ΜΕΡΟΣ Α: 4 Μετρήσεις  (B1–B4)")
    print("─"*60)
    results4 = run_all(SEEDS[:4], verbose=True)
    print_summary(results4)
    plot_results(results4, " – 4 Μετρήσεις")

    # ── 8 Μετρήσεις ──
    print("\n" + "─"*60)
    print("  ΜΕΡΟΣ Β: 8 Μετρήσεις  (B1–B8)")
    print("─"*60)
    results8 = run_all(SEEDS[:8], verbose=False)
    print_summary(results8)
    plot_results(results8, " – 8 Μετρήσεις")

    # ── Σύγκριση ──
    print("\n  4 vs 8 Μετρήσεις")
    plot_comparison(results4, results8)


