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
              f"{'Trunks(0 0 0)':>18} {'Service(NAI/OXI)':>15}")
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
# 4. ΓΡΑΦΗΜΑΤΑ
# ──────────────────────────────────────────────
def plot_results(results: list[dict], title_suffix: str = ""):
    n     = len(results)
    labels = [f"B{i+1}\n(x₀={r['seed']})" for i, r in enumerate(results)]
    b_vals = [r["B"] * 100 for r in results]
    mean_B = np.mean(b_vals)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.4), 5.5))
    fig.patch.set_facecolor("#F5F8FC")
    ax.set_facecolor("#F5F8FC")

    colors = ["#2E75B6" if b <= ERLANG_B * 100 + 5 else "#C00000" for b in b_vals]
    bars = ax.bar(labels, b_vals, color=colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=1.2)

    # θεωρητική γραμμή
    ax.axhline(ERLANG_B * 100, color="#FF0000", linestyle="--", linewidth=1.8,
               label=f"Erlang B  E₃(2.2) = {ERLANG_B*100:.0f}%", zorder=4)
    # μέση τιμή
    ax.axhline(mean_B, color="#70AD47", linestyle=":", linewidth=1.8,
               label=f"Μέση τιμή B̄ = {mean_B:.2f}%", zorder=4)

    # ετικέτες τιμών
    for bar, val in zip(bars, b_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#1F3864")

    ax.set_ylim(0, max(b_vals + [ERLANG_B * 100]) * 1.35)
    ax.set_ylabel("Πιθανότητα Απώλειας B (%)", fontsize=10)
    ax.set_title(
        f"Προσομοίωση M/M/3 – Μέθοδος Ρουλέτας{title_suffix}\n"
        f"α = {ALPHA} erl  |  {STEPS} βήματα/μέτρηση  |  "
        f"M = {M}, k = {K}",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_comparison(results4: list[dict], results8: list[dict]):
    """Σύγκριση 4 vs 8 μετρήσεων σε ένα διάγραμμα."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.patch.set_facecolor("#F5F8FC")
    fig.suptitle(
        f"\nΣύγκριση 4 vs 8 Μετρήσεων  |  M/M/3  |  α = {ALPHA} erl  |  "
        f"E₃(2.2) = {ERLANG_B*100:.0f}%",
        fontsize=12, fontweight="bold", y=1.01
    ) 

    for ax, results, subtitle in zip(
        axes,
        [results4, results8],
        ["4 Μετρήσεις", "8 Μετρήσεις"]
    ):
        ax.set_facecolor("#F5F8FC")
        labels = [f"B{i+1}\n(x₀={r['seed']})" for i, r in enumerate(results)]
        b_vals = [r["B"] * 100 for r in results]
        mean_B = np.mean(b_vals)

        colors = ["#2E75B6" if b <= ERLANG_B * 100 + 6 else "#C00000" for b in b_vals]
        bars = ax.bar(labels, b_vals, color=colors, width=0.55,
                      edgecolor="white", linewidth=1.2, zorder=3)

        ax.axhline(ERLANG_B * 100, color="#FF0000", linestyle="--",
                   linewidth=1.8, label=f"E₃(2.2) = {ERLANG_B*100:.0f}%", zorder=4)
        ax.axhline(mean_B, color="#70AD47", linestyle=":",
                   linewidth=1.8, label=f"B̄ = {mean_B:.2f}%", zorder=4)

        for bar, val in zip(bars, b_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#1F3864")

        ax.set_title(subtitle, fontsize=11, fontweight="bold")
        ax.set_ylabel("B (%)", fontsize=10)
        ax.set_ylim(0, 55)
        ax.legend(fontsize=8.5, framealpha=0.85)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

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


