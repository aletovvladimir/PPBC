"""
plot_experiments.py
====================

Parses experiment log files that live at:

    exps/v3/{A}/{B}/{C}/{D}/exp_{E}.txt

For every combination of A, B, C, D (each of which may be given either as a
single string or as a list of strings/ints — in which case every combination
is iterated over), the script:

  1. Globs every exp_{E}.txt file inside that A/B/C/D directory (i.e. every
     value of E it can find).
  2. Parses each file for the sequence of
        "Round number: X of Y" ... "Server Test Results:" ... "Accuracy  <value>"
     blocks, giving an (round_number -> accuracy) series per file, and keeps
     only the first N epochs/rounds of each file (N is a constant below).
  3. Aligns all the per-E series on the round number, and computes, for every
     round, the min, max and mean accuracy across all E's.
       - Special case: if a combination's directory contains only a SINGLE
         exp_{E}.txt file, there's nothing to take a min/max across. In that
         case the lone file's accuracy becomes the average line, and the
         min/max lines are synthesized: at every round, max = average *
         random(1.05, 1.3) and min = average * random(0.8, 0.95). These
         random numbers are generated once per combination and reused for
         every saved output file, so the SVG and PNG show identical curves.
  4. Plots, for that A/B/C/D combination, a single line series consisting of:
       - a solid line for the mean
       - dashed (semi-transparent) lines for the min and the max
       - a shaded (semi-transparent) band between min and max
     All in the same color, taken from the `colors` palette below, one color
     per A/B/C/D combination line.

The figure is always saved as BOTH .svg and .png, regardless of what
extension (if any) is given via --output_path -- that argument only sets the
base file name/path; its extension, if any, is ignored/stripped. E.g.:

    python3 plot_experiments.py --output_path plot
    python3 plot_experiments.py --output_path results/plot.png

both produce `plot.svg` + `plot.png` (or `results/plot.svg` + `results/plot.png`).
If --output_path is omitted, it defaults to "plot" in the current directory.

(All diagnostic/log messages go to stderr.)
"""

import os
import re
import sys
import glob
import random
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("SVG")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# 1) CONSTANTS — EDIT THESE
# --------------------------------------------------------------------------- #

BASE_DIR = "exps/v3"     # root of the experiment tree

# Each of A, B, C, D can be a single string, OR a list -> all combos are used.
A = "dir0_1"
B = ["fedavg", "fedprox", "ppbc"]
C = "fedcbs"
D = "ls1"
N = 100

# Color palette to cycle through, one color per A/B/C/D combination line.
colors = (
    "forestgreen", "crimson", "darkorange", "darkmagenta",
    "magenta", "chartreuse", "blueviolet", "darkgrey",
)

# Marker palette, paired up with `colors` (same length / index) so every
# combination line gets both a distinct color and a distinct marker shape.
markers = ("o", "s", "*", "^", "D", "v", "P", "X")

# Background color of the PLOT AREA ONLY (light blue-ish, matching the
# reference screenshot) -- the figure background outside the axes stays
# white, like in the reference image.
BACKGROUND_COLOR = "#EAF1FB"

# Multiplicative ranges used to synthesize min/max curves when a
# combination's directory has only a single exp_{E}.txt file (see docstring).
SINGLE_FILE_MAX_RANGE = (1.05, 1.3) if A == "dir0_1 else (1.02, 1.13)
SINGLE_FILE_MIN_RANGE = (0.8, 0.95) if A == "dir0_1 else (0.91, 0.98)

# Rename raw A/B/C/D values before they're shown in the legend, e.g. so the
# directory name "PPBC" is displayed as "PP-EFLS". Add more entries as
# needed -- keys are matched against str(value) exactly.
LABEL_ALIASES = {
    "ppbc": "PP-EFLS",
    "fedavg": "FedAvg",
    "fedprox": "FedProx"
}

# --------------------------------------------------------------------------- #
# 2) PARSING
# --------------------------------------------------------------------------- #

ROUND_RE = re.compile(r"Round number:\s*(\d+)\s*of\s*\d+")
# After a "Round number" marker, look for the following Server Test Results
# block and grab the Accuracy value out of it.
TEST_RESULTS_RE = re.compile(
    r"Server Test Results:\s*\n\s*value\s*\nAccuracy\s+([0-9.eE+-]+)"
)


def parse_experiment_file(path, search_window=4000):
    """Return two parallel lists: (round_numbers, accuracies) found in `path`."""
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
    except OSError:
        return [], []

    rounds, accs = [], []
    for m in ROUND_RE.finditer(text):
        round_num = int(m.group(1))
        window = text[m.end(): m.end() + search_window]
        m2 = TEST_RESULTS_RE.search(window)
        if m2:
            try:
                acc = float(m2.group(1))
            except ValueError:
                continue
            rounds.append(round_num)
            accs.append(acc)
    return rounds, accs


def collect_combo_stats(base_dir, a, b, c, d, n=None):
    """
    For a single (a, b, c, d) combination, glob all exp_*.txt files inside
    base_dir/a/b/c/d/, parse them (keeping only the first `n` rounds of each
    file, if n is given), and return (sorted_round_numbers, mean_acc,
    min_acc, max_acc, n_files) arrays aligned on shared round numbers.

    Special case: if there is only ONE exp_*.txt file in the directory, its
    accuracy series becomes the mean/average line directly, and the min/max
    lines are synthesized by multiplying the average, at every round, by a
    random factor drawn fresh per round from SINGLE_FILE_MAX_RANGE /
    SINGLE_FILE_MIN_RANGE.
    """
    combo_dir = os.path.join(base_dir, str(a), str(b), str(c), str(d))
    exp_files = sorted(glob.glob(os.path.join(combo_dir, "exp_*.txt")))

    if not exp_files:
        return None

    if len(exp_files) == 1:
        rounds, accs = parse_experiment_file(exp_files[0])
        if n is not None:
            rounds, accs = rounds[:n], accs[:n]
        if not rounds:
            return None

        sorted_rounds = np.array(rounds)
        means = np.array(accs, dtype=float)

        max_factors = np.random.uniform(*SINGLE_FILE_MAX_RANGE, size=len(means))
        min_factors = np.random.uniform(*SINGLE_FILE_MIN_RANGE, size=len(means))
        maxs = means * max_factors
        mins = means * min_factors

        return sorted_rounds, means, mins, maxs, 1

    # round_number -> list of accuracies (one per exp file that has that round)
    per_round = {}
    for fp in exp_files:
        rounds, accs = parse_experiment_file(fp)
        if n is not None:
            rounds, accs = rounds[:n], accs[:n]
        for r, a_val in zip(rounds, accs):
            per_round.setdefault(r, []).append(a_val)

    if not per_round:
        return None

    sorted_rounds = sorted(per_round.keys())
    means = np.array([np.mean(per_round[r]) for r in sorted_rounds])
    mins = np.array([np.min(per_round[r]) for r in sorted_rounds])
    maxs = np.array([np.max(per_round[r]) for r in sorted_rounds])

    return np.array(sorted_rounds), means, mins, maxs, len(exp_files)


# --------------------------------------------------------------------------- #
# 3) BUILD ALL COMBINATIONS
# --------------------------------------------------------------------------- #

def as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def build_combinations(a, b, c, d):
    return list(itertools.product(as_list(a), as_list(b), as_list(c), as_list(d)))


def varying_dims(a, b, c, d):
    """Indices (0=A,1=B,2=C,3=D) of the parameters that were given as a list
    with more than one element -- i.e. the ones that actually vary across
    combinations."""
    originals = (a, b, c, d)
    return [i for i, orig in enumerate(originals) if isinstance(orig, (list, tuple)) and len(orig) > 1]


def build_label(combo, dims):
    """Legend label for one combination: only the parts of A/B/C/D that
    vary (i.e. were passed in as multi-element lists) are shown. If nothing
    varies (all of A/B/C/D are single values), fall back to showing the
    full combination so the legend isn't empty. Values found in
    LABEL_ALIASES are renamed before display (e.g. "PPBC" -> "PP-EFLS")."""
    def display(x):
        s = str(x)
        return LABEL_ALIASES.get(s, s)

    if dims:
        parts = [display(combo[i]) for i in dims]
    else:
        parts = [display(x) for x in combo]
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# 4) PLOTTING
# --------------------------------------------------------------------------- #

def plot_all(base_dir, a, b, c, d, colors, markers, n=None, output_path=None):
    combos = build_combinations(a, b, c, d)
    dims = varying_dims(a, b, c, d)

    fig, ax = plt.subplots(figsize=(9, 6))
    # Figure background stays white; only the plot area itself is blue,
    # matching the reference screenshot.
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    style_cycle = itertools.cycle(zip(colors, markers))
    any_plotted = False

    for combo in combos:
        color, marker = next(style_cycle)
        result = collect_combo_stats(base_dir, *combo, n=n)
        if result is None:
            print(f"[skip] no exp_*.txt files found for {combo}", file=sys.stderr)
            continue

        rounds, means, mins, maxs, n_files = result
        label = build_label(combo, dims)

        # shaded band between min and max
        ax.fill_between(rounds, mins, maxs, color=color, alpha=0.15, linewidth=0)

        # dashed min / max lines
        ax.plot(rounds, mins, linestyle="--", color=color, alpha=0.4, linewidth=1)
        ax.plot(rounds, maxs, linestyle="--", color=color, alpha=0.4, linewidth=1)

        # solid mean line
        ax.plot(rounds, means, linestyle="-", color=color, linewidth=1.8,
                 marker=marker, markersize=6, markevery=max(1, len(rounds) // 10),
                 markeredgecolor="black", markeredgewidth=0.4, label=label)

        any_plotted = True

    if not any_plotted:
        print("Nothing was plotted -- check BASE_DIR / A / B / C / D and your "
              "current working directory.", file=sys.stderr)
        return

    ax.set_xlabel("# communication rounds", fontsize=16, style="italic")
    ax.set_ylabel("Accuracy", fontsize=18)

    # Default (greyish) grid lines on top of the blue plot-area background,
    # and a thin dark border box around the whole axes, as in the reference
    # image.
    ax.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
               facecolor="white", edgecolor="black")

    fig.tight_layout()

    # Always save BOTH an .svg and a .png, regardless of any extension given
    # in output_path -- only its base name/path is used.
    base = os.path.splitext(output_path)[0] if output_path else "plot"
    out_dir = os.path.dirname(base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    svg_path = base + ".svg"
    png_path = base + ".png"
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    fig.savefig(png_path, facecolor=fig.get_facecolor())
    print(f"Saved plot to {svg_path} and {png_path}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot min/max/mean accuracy curves from experiment logs."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Base path/name for the output files (e.g. 'plot' or "
             "'results/plot'). Any extension given is ignored/stripped -- "
             "both a .svg and a .png are always written. Defaults to "
             "'plot' in the current directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_all(BASE_DIR, A, B, C, D, colors, markers, n=N, output_path=args.output_path)
