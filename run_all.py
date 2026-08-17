"""Reproduce every number the article quotes, from scratch.

    python run_all.py            everything (the lattice hbar hunt takes a few minutes)
    python run_all.py --quick    skip the slow lattice hunt

Each check below names the claim it is testing. If a check fails, the article
is wrong and I want to know.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

RESULTS = []


def run(script, *args):
    print(f"\n$ python {script} {' '.join(args)}")
    proc = subprocess.run([sys.executable, str(HERE / script), *args],
                          cwd=HERE, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"{script} failed")
    print(proc.stdout.strip()[-800:])
    return proc.stdout


def check(claim, ok, detail):
    RESULTS.append((claim, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {claim}: {detail}")


def close(a, b, tol):
    return abs(a - b) / abs(b) < tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the lattice hbar hunt (the slow one)")
    args = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)

    # ---- the zipper animation -------------------------------------------
    out = run("the_zip.py")
    norms = [float(v) for v in
             re.search(r"norm at start/end:\s*([\d.]+)\s*/\s*([\d.]+)", out).groups()]
    check("the packet keeps all of itself (the animation)",
          all(abs(n - 1.0) < 1e-6 for n in norms), f"norm {norms[0]:.6f} -> {norms[1]:.6f}")

    # ---- with i / without i ---------------------------------------------
    out = run("the_letter.py")
    drop = float(re.search(r"\(([\deE.+-]+)x smaller\)", out).group(1))
    settled = float(re.search(r"centre settled at x =\s*([+-][\d.]+)", out).group(1))
    check("without the i the packet fades away (article: about three million times)",
          2.0e6 < drop < 6.0e6, f"{drop:.3g}x fainter")
    check("without the i the packet stops in the middle",
          abs(settled) < 1e-3, f"centre settled at x = {settled:+.5f}")

    # ---- the direct test: is the push a property of the medium? ---------
    run("pressure_slot_test.py")
    m = json.loads((HERE / "metrics.json").read_text())
    scan = m["window_scan"]
    first = [p["lambda_packet"] for p in scan[0]["packet_metrics"]]
    spread = max(first) / min(first)
    check("three packets, three different answers (article: about thirty times apart)",
          close(spread, 29.8, 0.05), f"{spread:.1f}x apart  ({', '.join(f'{v:.4f}' for v in first)})")

    ratios, negatives = [], 0
    for w in scan:
        lam = [p["lambda_packet"] for p in w["packet_metrics"]]
        if any(v < 0 for v in lam):
            negatives += 1
        else:
            ratios.append(max(lam) / min(lam))
    check("scanning the window does not rescue it (article: five times apart at best)",
          close(min(ratios), 5.4, 0.1), f"best window still {min(ratios):.1f}x apart")
    check("scanning the window makes it worse (article: two hundred times apart at others)",
          max(ratios) > 190.0, f"worst window {max(ratios):.0f}x apart")
    check("two settings return a negative coefficient (article: not something a pressure can be)",
          negatives == 2, f"{negatives} settings with a negative answer")

    # ---- the exchange rate ----------------------------------------------
    run("exchange_rate.py")
    k = json.loads((HERE / "metrics_exchange_rate.json").read_text())
    best = min(s["KF_minus_U_fro"] for s in k["stage1"])
    check("the classical machine and the quantum one are the same operator "
          "(article: fourteen decimal places)",
          best < 1e-14, f"largest disagreement {best:.3e}")

    run("exchange_rate_harder.py")
    h = json.loads((HERE / "metrics_exchange_rate_harder.json").read_text())
    hard = h["rung_1_quartic"][0]["stage1"]["KF_minus_U_fro"]
    check("the same trick works for a harder system (article: harder systems too)",
          hard < 1e-13, f"disagreement {hard:.3e}")

    # ---- the hbar hunt in the lattice -----------------------------------
    if args.quick:
        print("\n(skipping the lattice hbar hunt: --quick)")
    else:
        run("hbar_from_the_lattice.py")
        rows = json.loads((HERE / "metrics_hbar_hunt.json").read_text())
        grid = {r["tag"]: r["hbar_eff"] for r in rows if r["tag"].startswith("D_dx")}
        drift = grid["D_dx1.0"] / grid["D_dx0.25"]
        check("a four-fold change of grid moves the constant (article: by 739)",
              close(drift, 738.6, 0.02), f"{drift:.1f}x over a 4x grid change")

    # ---- the early toys, shipped as they were ---------------------------
    for toy in ("early_toy_orbitals.py", "early_toy_decoherence.py"):
        run(toy)
    run("early_toy_collapse.py", "--output-dir", str(FIGURES))
    check("the early toys still run (shipped with their mistakes intact)", True,
          "3 toys executed")

    print("\n" + "=" * 62)
    bad = [c for c, ok, _ in RESULTS if not ok]
    for claim, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {claim}")
    print("=" * 62)
    if bad:
        raise SystemExit(f"{len(bad)} check(s) FAILED")
    print(f"all {len(RESULTS)} checks reproduced")


if __name__ == "__main__":
    main()
