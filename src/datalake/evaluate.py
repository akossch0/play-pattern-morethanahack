"""Score predictions on harmonised external matches with the official scorer.

    python -m datalake.evaluate runs/my_model                 # the test split (default)
    python -m datalake.evaluate runs/my_model --split train
    python -m datalake.evaluate runs/my_model --matches dfl_J03WMX
    python -m datalake.evaluate runs/a runs/b                 # compare several runs

A run is a folder of `<match_key>.csv` files, each `frame,label` with one row per harmonised frame
and labels H / A / D / X: the competition's submission format, per match. A missing or malformed
file is reported and counts as a failure for that run, never silently skipped.

`--labels` picks the label view the truth is read through (datalake/remap.py). The default is the
provider's own labels; a score is only comparable with scores made under the same view.

Two adaptations apply the official scorer to external matches:

  - The two halves of a match are scored as separate runs. The official scorer splits runs where
    the ground truth is X; a harmonised match has no gap at half-time, so one X frame is placed
    between the periods (in truth and prediction alike) before scoring.
  - Matches are pooled per label source, never across them. `flags` (DFL) is the ranked
    comparison. `provider_model` matches with H / A only have no dead-ball truth, so macro F1 is
    not reported for them.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

from evaluation import possession_metrics as pm

from . import core, remap
from .matches import BY_KEY, CONTRACT_VERSION, MATCHES

assert (pm.H, pm.A, pm.D, pm.X) == (core.H, core.A, core.D, core.X), "label codes differ"

GROUPS = (
    ("dfl", "DFL (flags, H/A/D): the ranked comparison", lambda m: m["source"] == "dfl"),
    ("clasicos", "Clasicos (SkillCorner 2019/20, H/A only)", lambda m: m["key"] in ("sc2019_2841", "sc2019_3442")),
    ("skillcorner_2019", "SkillCorner 2019/20 (H/A only)", lambda m: m["source"] == "skillcorner_2019"
     and m["key"] not in ("sc2019_2841", "sc2019_3442")),
    ("skillcorner_aleague", "SkillCorner A-League (weak inferred D)", lambda m: m["source"] == "skillcorner_aleague"),
)


def with_period_gap(codes, period):
    """Insert one X where the period changes, so the scorer never joins the two halves."""
    cut = np.flatnonzero(np.diff(period.astype(int)) != 0) + 1
    return np.insert(codes, cut, core.X)


def read_prediction(path, n_frames):
    """Label codes from a `frame,label` CSV, or a string saying what is wrong with it."""
    if not os.path.isfile(path):
        return "no file"
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # the official reader's own message is the useful part
        return "unreadable: {}".format(error)
    problem = pm.validate_submission(table, n_frames)
    if problem:
        return problem
    return pm.label_codes(table)


def score_group(truths, preds, bootstrap_n=0):
    """One official evaluate() call over several matches, separated by X so nothing joins."""
    gap = np.array([core.X], dtype=truths[0].dtype)
    gt = np.concatenate([part for t in truths for part in (t, gap)])
    pred = np.concatenate([part for p in preds for part in (p.astype(gt.dtype), gap)])
    return pm.evaluate(gt, pred, bootstrap_n=bootstrap_n)


def headline(result, has_dead):
    return {
        "spell_f1": round(result["spell_f1_mean"], 4),
        "macro_f1": round(result["macro_f1"], 4) if has_dead else None,
        "team_acc": round(result["team_acc_both_alive"], 4),
        "alive_dead_f1": round(result["alive_dead_f1"], 4) if has_dead else None,
        "gt_spells": result["n_gt_spells"],
        "pred_spells": result["n_pred_spells"],
        "gt_spell_s": round(result["gt_mean_spell_duration_s"], 1),
        "pred_spell_s": round(result["mean_spell_duration_s"], 1),
        "graded_frames": result["n_graded"],
        "spell_f1_ci": [round(result["spell_bootstrap_low"], 4), round(result["spell_bootstrap_high"], 4)]
        if result["bootstrap_n"] else None,
    }


def evaluate_run(run_dir, keys, harmonised=None, bootstrap_n=0, labels="provider"):
    """{"matches": {key: headline or {"error": ...}}, "groups": {name: headline}}."""
    matches, loaded = {}, {}
    for key in keys:
        try:
            truth = core.load(key, harmonised, labels=labels)
        except FileNotFoundError:
            matches[key] = {"error": "not harmonised yet: run python -m datalake.harmonise " + key}
            continue
        pred = read_prediction(os.path.join(run_dir, key + ".csv"), len(truth["label"]))
        if isinstance(pred, str):
            matches[key] = {"error": pred}
            continue
        gt = with_period_gap(truth["label"].astype(np.int8), truth["period"])
        pr = with_period_gap(np.asarray(pred).astype(np.int8), truth["period"])
        has_dead = "D" in BY_KEY[key]["labels"]
        matches[key] = headline(pm.evaluate(gt, pr), has_dead)
        loaded[key] = (gt, pr)

    groups = {}
    for name, _, belongs in GROUPS:
        members = [k for k in keys if belongs(BY_KEY[k])]
        if not members:
            continue
        scored = [k for k in members if k in loaded]
        if len(scored) != len(members):
            groups[name] = {"error": "{} of {} matches could not be scored".format(
                len(members) - len(scored), len(members))}
            continue
        has_dead = all("D" in BY_KEY[k]["labels"] for k in members)
        result = score_group([loaded[k][0] for k in scored], [loaded[k][1] for k in scored], bootstrap_n)
        groups[name] = dict(headline(result, has_dead), n_matches=len(scored))
    return {"matches": matches, "groups": groups}


def _cell(value):
    return "   -  " if value is None else "{:.4f}".format(value)


def print_comparison(reports, keys, labels="provider"):
    print("contract v{}  labels {}  scorer evaluation/possession_metrics.py  spell IoU {}".format(
        CONTRACT_VERSION, labels, ", ".join(str(i) for i in pm.SPELL_IOUS)))
    for name, title, belongs in GROUPS:
        if not any(belongs(BY_KEY[k]) for k in keys):
            continue
        print("\n{}".format(title))
        print("  {:<28} {:>7} {:>7} {:>8} {:>10}  {:>11}  {:>11}".format(
            "run", "spellF1", "macroF1", "teamAcc", "aliveDead", "spells p/gt", "dur s p/gt"))
        rows = []
        for run, report in reports:
            g = report["groups"].get(name)
            rows.append((run, g))
        rows.sort(key=lambda r: -(r[1].get("spell_f1", -1) if r[1] and "error" not in r[1] else -1))
        for run, g in rows:
            label = os.path.basename(os.path.normpath(run))[:28]
            if g is None or "error" in g:
                print("  {:<28} {}".format(label, (g or {}).get("error", "no matches")))
                continue
            print("  {:<28} {:>7} {:>7} {:>8} {:>10}  {:>5}/{:<5}  {:>5}/{:<5}{}".format(
                label, _cell(g["spell_f1"]), _cell(g["macro_f1"]), _cell(g["team_acc"]),
                _cell(g["alive_dead_f1"]), g["pred_spells"], g["gt_spells"],
                g["pred_spell_s"], g["gt_spell_s"],
                "  CI {}..{}".format(*g["spell_f1_ci"]) if g.get("spell_f1_ci") else ""))
    for run, report in reports:
        errors = [(k, v["error"]) for k, v in report["matches"].items() if "error" in v]
        if errors:
            print("\n{}: {} match(es) not scored".format(run, len(errors)))
            for key, reason in errors:
                print("  {:<16} {}".format(key, reason))


def run_context(labels, split, keys, harmonised=None):
    """What a score depends on besides the predictions: recorded with every report."""
    scorer = os.path.join(core.ROOT, "evaluation", "possession_metrics.py")
    with open(scorer, "rb") as handle:
        scorer_sha = hashlib.sha256(handle.read()).hexdigest()
    snapshot = None
    try:
        with open(os.path.join(harmonised or core.HARMONISED, "PROVENANCE.json")) as handle:
            snapshot = json.load(handle).get("version")
    except (OSError, ValueError):
        pass
    return {"contract_version": CONTRACT_VERSION, "data_snapshot": snapshot, "labels": labels,
            "split": split, "matches": list(keys), "scorer_sha256": scorer_sha}


def markdown_report(reports, context):
    """Render the evaluation as Markdown."""
    lines = ["**Evaluation** | data snapshot `{}` | contract v{} | labels `{}` | split `{}` | scorer `{}`".format(
        context["data_snapshot"] or "unrecorded", context["contract_version"], context["labels"],
        context["split"], context["scorer_sha256"][:8]), ""]
    for name, title, belongs in GROUPS:
        if not any(belongs(BY_KEY[k]) for k in context["matches"]):
            continue
        lines += ["{}".format(title), "",
                  "| run | spell F1 | macro F1 | team acc | alive/dead F1 | spells pred / true | 5-95% spell F1 |",
                  "|---|---|---|---|---|---|---|"]
        for run, report in reports:
            g = report["groups"].get(name)
            label = os.path.basename(os.path.normpath(run))
            if g is None or "error" in g:
                lines.append("| {} | not scored: {} | | | | | |".format(label, (g or {}).get("error", "no matches")))
                continue
            ci = "{} to {}".format(*g["spell_f1_ci"]) if g.get("spell_f1_ci") else "not run"
            lines.append("| {} | {} | {} | {} | {} | {} / {} | {} |".format(
                label, _cell(g["spell_f1"]).strip(), _cell(g["macro_f1"]).strip(),
                _cell(g["team_acc"]).strip(), _cell(g["alive_dead_f1"]).strip(),
                g["pred_spells"], g["gt_spells"], ci))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score runs on harmonised external matches.")
    parser.add_argument("runs", nargs="+", help="folders of <match_key>.csv predictions")
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--matches", nargs="*", help="score only these match keys")
    parser.add_argument("--harmonised", help="harmonised data folder (default src/data/team-dataset)")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws for the group CI")
    parser.add_argument("--labels", default="provider",
                        help="label view to score against (datalake/remap.py); default: provider")
    parser.add_argument("--json", help="also write the full report to this file")
    parser.add_argument("--report", help="also write a Markdown summary to this file")
    args = parser.parse_args(argv)

    if args.matches:
        unknown = [k for k in args.matches if k not in BY_KEY]
        if unknown:
            sys.exit("unknown match key(s): {}".format(", ".join(unknown)))
        keys = args.matches
    else:
        keys = [m["key"] for m in MATCHES if args.split == "all" or m["split"] == args.split]

    if args.labels not in remap.NAMES:
        sys.exit("unknown label view {!r}; have {}".format(args.labels, ", ".join(remap.NAMES)))
    reports = [(run, evaluate_run(run, keys, args.harmonised, args.bootstrap, args.labels))
               for run in args.runs]
    print_comparison(reports, keys, args.labels)
    context = run_context(args.labels, "custom" if args.matches else args.split, keys, args.harmonised)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(dict(context, runs={run: report for run, report in reports}), handle, indent=1)
            handle.write("\n")
    if args.report:
        with open(args.report, "w") as handle:
            handle.write(markdown_report(reports, context) + "\n")
    failed = any("error" in v for _, r in reports for v in r["matches"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
