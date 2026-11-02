"""
Turn the per-job files into the tables the thesis needs.

Reads two things under the results folder:
    results.json      what run_experiment.py wrote: pooled test mAP50, timing
    eval_matrix.json  what evaluate_matrix.py wrote: one score per test set

Writes:
    all_runs.csv    one row per (setup, seed, round): pooled test mAP50, minutes
    summary.csv     mean and std over seeds per (setup, round)
    all_evals.csv   one row per (setup, seed, round, test set)
    matrix.csv      mean +- std per (setup, round) x test set
    paired.csv      per-seed differences between setups, per round and test set

Why paired differences: the seed-to-seed spread of a single setup (about 0.004
mAP50 here) is of the same size as the gap between setups, so mean +- std alone
cannot settle it. Subtracting two setups *within the same seed* removes the part
of the noise both share. Three seeds all pointing the same way is weak but real
evidence; two up and one down is not.

Round 1 is shared by all setups, so its rows are copied into each setup.
A --tag variant appears as its own setup label, e.g. "sp_lam0.3".

Usage:
    python summarize.py --results <folder with seed*/>
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

SETUPS = ["cold", "warm", "sp"]
METRICS = ["map50", "map50_95", "precision", "recall"]
DEFAULT_PAIRS = ["sp:warm", "sp:cold", "warm:cold"]


def label_of(setup: str, tag: str) -> str:
    return f"{setup}_{tag}" if tag else setup


def expand_shared(setup: str) -> list[str]:
    """Round 1 is one training run shared by all setups; count it for each."""
    return SETUPS if setup == "shared" else [setup]


def load_runs(results: Path) -> list[dict]:
    """The pooled test number and the training time, from results.json."""
    rows = []
    for path in sorted(results.glob("seed*/**/results.json")):
        r = json.loads(path.read_text())
        if "test_last" not in r:  # a results.json written by an older version
            print(f"[warn] {path} has no test_last, skipped")
            continue
        for setup in expand_shared(r["setup"]):
            rows.append({
                "setup": label_of(setup, r.get("tag", "")),
                "seed": r["seed"],
                "round": r["round"],
                "epochs": r.get("epochs"),
                "lam": r.get("lam"),
                "test_map50_last": r["test_last"]["map50"],
                "test_map50_95_last": r["test_last"]["map50_95"],
                "test_map50_best": r["test_best"]["map50"],
                "train_minutes": round(r["train_seconds"] / 60, 1),
            })
    return rows


def load_evals(results: Path, which: str) -> list[dict]:
    """One row per (setup, seed, round, test set), from eval_matrix.json."""
    rows = []
    for path in sorted(results.glob("seed*/**/eval_matrix.json")):
        m = json.loads(path.read_text())
        per_set = m.get("evals", {}).get(which, {})
        for setup in expand_shared(m["setup"]):
            for set_name, metrics in per_set.items():
                rows.append({
                    "setup": label_of(setup, m.get("tag", "")),
                    "seed": m["seed"],
                    "round": m["round"],
                    "set": set_name,
                    **{k: metrics[k] for k in METRICS},
                })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return (
        round(statistics.mean(values), 4),
        round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
    )


def order_key(label: str) -> tuple[int, str]:
    """cold, warm, sp first (in that order), then tagged variants."""
    base = label.split("_")[0]
    return (SETUPS.index(base) if base in SETUPS else len(SETUPS), label)


def summarize_runs(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["setup"], r["round"])].append(r)

    summary = []
    for (setup, rnd), group in sorted(groups.items(), key=lambda g: (order_key(g[0][0]), g[0][1])):
        row = {"setup": setup, "round": rnd, "n_seeds": len(group)}
        for key in ["test_map50_last", "test_map50_95_last", "test_map50_best", "train_minutes"]:
            row[f"{key}_mean"], row[f"{key}_std"] = mean_std([g[key] for g in group])
        summary.append(row)
    return summary


def summarize_matrix(rows: list[dict], metric: str) -> tuple[list[dict], list[str]]:
    """Pivot to (setup, round) x test set, mean and std over seeds."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["setup"], r["round"], r["set"])].append(r[metric])

    set_names = sorted({r["set"] for r in rows}, key=lambda s: (not s.startswith("low_all"), s))
    out = []
    for setup, rnd in sorted({(r["setup"], r["round"]) for r in rows}, key=lambda g: (order_key(g[0]), g[1])):
        row = {"setup": setup, "round": rnd}
        for name in set_names:
            values = cells.get((setup, rnd, name), [])
            if values:
                row[f"{name}_mean"], row[f"{name}_std"] = mean_std(values)
                row[f"{name}_n"] = len(values)
            else:
                row[f"{name}_mean"] = row[f"{name}_std"] = row[f"{name}_n"] = None
        out.append(row)
    return out, set_names


def paired_differences(rows: list[dict], metric: str, pairs: list[str]) -> list[dict]:
    """
    For each (round, test set) and each pair of setups, the difference computed
    inside every seed that ran both.
    """
    by_key = {(r["setup"], r["seed"], r["round"], r["set"]): r[metric] for r in rows}
    rounds = sorted({r["round"] for r in rows})
    set_names = sorted({r["set"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})

    out = []
    for pair in pairs:
        a, b = pair.split(":")
        for rnd in rounds:
            for name in set_names:
                diffs = [
                    by_key[(a, s, rnd, name)] - by_key[(b, s, rnd, name)]
                    for s in seeds
                    if (a, s, rnd, name) in by_key and (b, s, rnd, name) in by_key
                ]
                if not diffs:
                    continue
                mean, std = mean_std(diffs)
                n_positive = sum(d > 0 for d in diffs)
                n_negative = sum(d < 0 for d in diffs)
                out.append({
                    "comparison": f"{a} - {b}",
                    "round": rnd,
                    "set": name,
                    "n_seeds": len(diffs),
                    "mean_diff": mean,
                    "std_diff": std,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    # Round 1 is one shared run, so its differences are exactly
                    # zero; that is not agreement, so zeros break the sign test.
                    "same_sign": n_positive == len(diffs) or n_negative == len(diffs),
                })
    return out


def print_matrix(matrix: list[dict], set_names: list[str], metric: str) -> None:
    width = max(14, max((len(n) for n in set_names), default=14) + 2)
    header = f"{'setup':<12} {'round':>5} " + "".join(f"{name:>{width}}" for name in set_names)
    print(f"\n{metric} per test set (mean over seeds, std in brackets)")
    print(header)
    print("-" * len(header))
    for row in matrix:
        line = f"{row['setup']:<12} {row['round']:>5} "
        for name in set_names:
            mean = row[f"{name}_mean"]
            cell = "-" if mean is None else f"{mean:.3f} ({row[f'{name}_std']:.3f})"
            line += f"{cell:>{width}}"
        print(line)


def print_paired(paired: list[dict], metric: str) -> None:
    print(f"\npaired per-seed differences in {metric} (+ means the first setup is better)")
    print(f"{'comparison':<16} {'round':>5} {'set':<14} {'seeds':>5} {'mean diff':>10} {'+/n':>6} {'all same sign':>14}")
    print("-" * 78)
    for row in paired:
        print(f"{row['comparison']:<16} {row['round']:>5} {row['set']:<14} {row['n_seeds']:>5} "
              f"{row['mean_diff']:>+10.4f} {row['n_positive']:>3}/{row['n_seeds']:<2} "
              f"{'yes' if row['same_sign'] else '':>14}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the warm start experiment.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--weights", default="last", choices=["last", "best"],
                        help="which checkpoint's matrix to summarize")
    parser.add_argument("--metric", default="map50", choices=METRICS)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="setup pairs to difference, as a:b (e.g. sp:cold sp_ep10:cold)")
    args = parser.parse_args()

    results = args.results.resolve()
    runs = load_runs(results)
    if runs:
        summary = summarize_runs(runs)
        write_csv(results / "all_runs.csv", runs)
        write_csv(results / "summary.csv", summary)
        print(f"{'setup':<12} {'round':>5} {'seeds':>5} {'pooled low test mAP50 (last.pt)':>32} {'train min':>10}")
        for s in summary:
            print(f"{s['setup']:<12} {s['round']:>5} {s['n_seeds']:>5} "
                  f"{s['test_map50_last_mean']:>24.3f} ± {s['test_map50_last_std']:.3f} "
                  f"{s['train_minutes_mean']:>10.1f}")
    else:
        print(f"no results.json found under {results}")

    evals = load_evals(results, args.weights)
    if not evals:
        print("\nno eval_matrix.json found: run evaluate_matrix.py to build the per-test-set tables")
        return

    write_csv(results / "all_evals.csv", evals)
    matrix, set_names = summarize_matrix(evals, args.metric)
    write_csv(results / "matrix.csv", matrix)
    print_matrix(matrix, set_names, args.metric)

    paired = paired_differences(evals, args.metric, args.pairs)
    if paired:
        write_csv(results / "paired.csv", paired)
        print_paired(paired, args.metric)

    print(f"\nwrote all_runs.csv, summary.csv, all_evals.csv, matrix.csv and paired.csv to {results}")


if __name__ == "__main__":
    main()
