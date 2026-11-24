"""Summarize matched human-source progress for two contact-QP trials."""
import json
import sys
from pathlib import Path


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(trial_dir):
    trial = Path(trial_dir)
    exp = json.loads((trial / "experiment.json").read_text(encoding="utf-8"))
    run = Path(exp["run"])
    rows = read_jsonl(run / "contact_qp_gait.jsonl")
    active = [r for r in rows if r.get("motion_start_sim_time") is not None]
    start_sim = float(active[0]["motion_start_sim_time"])
    motion_rows = [r for r in active if float(r["sim_time"]) >= start_sim - 1e-8]
    initial = motion_rows[0]["root_xy_m"]

    def drift(row):
        return ((row["root_xy_m"][0] - initial[0]) ** 2 +
                (row["root_xy_m"][1] - initial[1]) ** 2) ** .5

    summary = {
        "trial": trial.name,
        "qp_root_xy_weight": motion_rows[0].get("root_xy_task_weight"),
        "motion_start_sim_s": start_sim,
        "source_elapsed_max_s": max(float(r["source_elapsed_s"]) for r in motion_rows),
        "root_drift_max_qp_m": max(drift(r) for r in motion_rows),
        "root_error_mean_m": sum(float(r["root_xy_error_m"]) for r in motion_rows) / len(motion_rows),
        "root_error_p95_m": sorted(float(r["root_xy_error_m"]) for r in motion_rows)[int(.95 * (len(motion_rows) - 1))],
        "actual_liftoff_samples": sum(not all(r.get("actual_contact", [True, True])) for r in motion_rows),
        "points": [],
    }
    for target_elapsed in (0, 5, 10, 15, 20, 25, 30, 35):
        available = [r for r in motion_rows if float(r["source_elapsed_s"]) <= target_elapsed + 1e-8]
        if not available:
            continue
        row = available[-1]
        summary["points"].append({
            "source_elapsed_s": target_elapsed,
            "observed_elapsed_s": float(row["source_elapsed_s"]),
            "sim_time_s": float(row["sim_time"]),
            "root_xy_m": row["root_xy_m"],
            "target_xy_m": row["root_target_xy_m"],
            "root_error_m": float(row["root_xy_error_m"]),
            "drift_from_motion_start_m": drift(row),
        })
    review_path = trial / "review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        summary["review_passed"] = review.get("passed")
        summary["review_root_drift_m"] = review.get("metrics", {}).get("actual_root_xy_drift_m")
        summary["double_support_fraction"] = review.get("metrics", {}).get("double_support_fraction")
        summary["root_min_z_m"] = review.get("metrics", {}).get("actual_root_min_z_m")
    return summary


if __name__ == "__main__":
    print(json.dumps([summarize(p) for p in sys.argv[1:]], ensure_ascii=False, indent=2))
