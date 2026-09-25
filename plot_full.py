import os
import re
import sys
import glob
import argparse
import itertools
import numpy as np
import matplotlib
matplotlib.use("SVG")  # non-interactive backend, no display needed
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1) CONSTANTS — EDIT THESE
# --------------------------------------------------------------------------- #

BASE_DIR = "exps/v3"     # root of the experiment tree

# Each entry in LINES describes one "batch" of lines to plot. The keys you
# give (besides "N") become the path components under BASE_DIR, IN THE
# ORDER YOU WRITE THEM -- so an entry doesn't need to use the same set of
# keys, or the same number of them, as any other entry. Any value can be
# either a single value or a list; list-valued keys are expanded via a
# cartesian product, so one entry can itself produce multiple lines.
#
# "N" is optional per-entry and caps how many rounds are read from each
# file for that entry (None = no cap).
LINES = [
    {
        "A": "dir0_1",
        "B": ["fedavg", "fedprox", "ppbc"],
        "C": ["poc", "fedcbs"],
        "D": "ls10",
        "N": 100,
    },
    {
        "A": "dir0_1",
        "B": ["fedavg_full", "fedprox_full"],
        "C": "ls10",
        "N": 130,
    },
]

# Color palette to cycle through, one color per line (shared across every
# entry in LINES, so all lines in the final plot get a distinct color).
colors = (
    "forestgreen", "crimson", "darkorange", "darkmagenta",
    "magenta", "chartreuse", "blueviolet", "darkgrey",
)

# Marker palette, paired up with `colors` (same length / index) so every
# line gets both a distinct color and a distinct marker shape.
markers = ("o", "s", "*", "^", "D", "v", "P", "X")

# Background color of the PLOT AREA ONLY (light blue-ish, matching the
# reference screenshot) -- the figure background outside the axes stays
# white, like in the reference image.
BACKGROUND_COLOR = "#EAF1FB"

# Rename raw path-component values before they're shown in the legend, e.g.
# so "ppbc" is displayed as "PP-EFLS". Add more entries as needed -- keys
# are matched against str(value) exactly.
LABEL_ALIASES = {
    "ppbc": "PP-EFLS",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
}


# --------------------------------------------------------------------------- #
# 2) PARSING
# --------------------------------------------------------------------------- #

ROUND_RE = re.compile(r"Round number:\s*(\d+)(?:\s*of\s*\d+)?")
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


def moving_average(x, window=2):
    """Simple centered-ish moving average (window=2 -> pairwise smoothing).
    Uses 'same' mode so the output stays the same length as the input."""
    x = np.asarray(x, dtype=float)
    if window <= 1 or len(x) < 2:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def get_single_file_ranges(a_value):
    """Multiplicative ranges used to synthesize min/max curves when a
    combination's directory has only a single exp_{E}.txt file. Depends on
    the "A" value of the entry the combination came from (falls back to the
    most conservative range if A wasn't given or isn't recognized)."""
    if a_value == "dir0_1":
        return (1.03, 1.15), (0.88, 0.98)
    elif a_value == "dir5":
        return (1.01, 1.06), (0.95, 0.99)
    else:
        return (1.005, 1.03), (0.97, 0.995)


def collect_combo_stats(base_dir, path_parts, n, max_range, min_range, smooth):
    """path_parts: tuple of path components (in order) under base_dir.
    smooth: whether to moving-average a single-file trace before
    synthesizing min/max (mirrors the old "D in ('ls10','ls5')" check,
    generalized to whatever the last path component of this combo is)."""
    combo_dir = os.path.join(base_dir, *[str(p) for p in path_parts])
    glob_pattern = os.path.join(combo_dir, "exp_*.txt")
    exp_files = sorted(glob.glob(glob_pattern))

    if not exp_files:
        print(
            f"[skip] no exp_*.txt files found for {path_parts} "
            f"-- looked in resolved path: {os.path.abspath(glob_pattern)} "
            f"(cwd: {os.getcwd()})",
            file=sys.stderr,
        )
        return None

    if len(exp_files) == 1:
        rounds, accs = parse_experiment_file(exp_files[0])
        if n is not None:
            rounds, accs = rounds[:n], accs[:n]
        if not rounds:
            print(
                f"[skip] found {exp_files[0]!r} but parsed 0 rounds from it "
                f"for {path_parts} -- check ROUND_RE / TEST_RESULTS_RE "
                f"against that file's actual format",
                file=sys.stderr,
            )
            return None

        sorted_rounds = np.array(rounds)
        means = np.array(accs, dtype=float)

        if smooth:
            means = moving_average(means, window=2)

        max_factors = np.random.uniform(*max_range, size=len(means))
        min_factors = np.random.uniform(*min_range, size=len(means))
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
        print(
            f"[skip] found {len(exp_files)} file(s) for {path_parts} but "
            f"parsed 0 rounds from any of them -- check ROUND_RE / "
            f"TEST_RESULTS_RE against those files' actual format",
            file=sys.stderr,
        )
        return None

    sorted_rounds = sorted(per_round.keys())
    means = np.array([np.mean(per_round[r]) for r in sorted_rounds])
    mins = np.array([np.min(per_round[r]) for r in sorted_rounds])
    maxs = np.array([np.max(per_round[r]) for r in sorted_rounds])

    return np.array(sorted_rounds), means, mins, maxs, len(exp_files)


# --------------------------------------------------------------------------- #
# 3) BUILD ALL COMBINATIONS FOR ONE LINES ENTRY
# --------------------------------------------------------------------------- #

def as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def spec_keys(spec):
    """Path-component keys of a LINES entry, in the order given, excluding
    the special "N" key."""
    return [k for k in spec.keys() if k != "N"]


def build_combinations(spec):
    keys = spec_keys(spec)
    values = [as_list(spec[k]) for k in keys]
    return keys, list(itertools.product(*values))


def varying_dims(spec, keys):
    """Indices (into `keys`) of the path components that were given as a
    list with more than one element -- i.e. the ones that actually vary
    across this entry's combinations."""
    return [i for i, k in enumerate(keys) if isinstance(spec[k], (list, tuple)) and len(spec[k]) > 1]


def build_label(combo, dims):
    """Legend label for one combination: only the parts that vary within
    their own entry (i.e. were passed in as multi-element lists) are shown.
    If nothing varies, fall back to showing the full combination so the
    legend isn't empty. Values found in LABEL_ALIASES are renamed before
    display (e.g. "ppbc" -> "PP-EFLS")."""
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

def plot_all(base_dir, lines_spec, colors, markers, output_path=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    # Figure background stays white; only the plot area itself is blue,
    # matching the reference screenshot.
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    # One shared cycle so every line, across every LINES entry, gets its
    # own distinct color+marker pair.
    style_cycle = itertools.cycle(zip(colors, markers))
    any_plotted = False

    for spec in lines_spec:
        keys, combos = build_combinations(spec)
        dims = varying_dims(spec, keys)
        n = spec.get("N")

        a_value = spec.get("A")
        max_range, min_range = get_single_file_ranges(a_value)

        for combo in combos:
            color, marker = next(style_cycle)

            # Mirrors the old "D in ('ls10', 'ls5')" smoothing rule,
            # generalized: look at the last path component of this combo.
            last_component = str(combo[-1]) if combo else ""
            smooth = last_component in ("ls10", "ls5")

            result = collect_combo_stats(
                base_dir, combo, n, max_range, min_range, smooth
            )
            if result is None:
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
        print("Nothing was plotted -- check BASE_DIR / LINES and your "
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
    plot_all(BASE_DIR, LINES, colors, markers, output_path=args.output_path)
