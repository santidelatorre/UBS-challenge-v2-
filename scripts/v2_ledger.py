"""Append V2 evidence while preserving the exact frozen V1 ledger prefix."""
import v2_bootstrap
import json
import subprocess
from ubs_recurrence.data import ROOT


def main():
    rows=[json.loads(p.read_text()) for p in (ROOT/"outputs/experiments").glob("v2_*/metrics.json")]
    rows.sort(key=lambda r:(r["timestamp"],r["experiment_id"]))
    assert len({r["experiment_id"] for r in rows})==len(rows)
    original_json=subprocess.check_output(["git","show","baseline-v1-0.619493:reports/experiment_results.jsonl"],cwd=ROOT)
    original_md=subprocess.check_output(["git","show","baseline-v1-0.619493:reports/experiment_log.md"],cwd=ROOT)
    lines=[]
    for r in rows:
        name=r["experiment_id"]
        if "control" in name or "stress_baseline" in name:status="reference_verification"
        elif "stress_candidate" in name:status="rejected_four_event_gate"
        elif "pair_" in name:status="rejected_identity_transfer"
        elif "gate_only" in name:status="rejected_indicator_ablation"
        elif "sparse_" in name:status="train_gain_not_accepted_after_removal_gate"
        else:status="frozen_diagnostic"
        r["v2_disposition"]=status
        lines.append(f"| {name} | {r['evaluation_split']} | {r['macro_f1']:.6f} | {r['accuracy']:.6f} | {r['runtime_seconds']:.1f} | {status} | {r['hypothesis']} |\n")
    auxiliary=json.loads((ROOT/"reports/v2_pair_learning.json").read_text())
    auxiliary.update(experiment_id="v2_weak_pair_auxiliary",evaluation_split="auxiliary_weak_pair_calibration_NOT_challenge",
                     hypothesis="Conservative positive streams and within-client hard negatives can teach compatibility beyond anchor rules under corruption",
                     baseline_comparison="Weak anchor rule; challenge transfer evaluated separately",macro_f1=None,accuracy=None,
                     candidate_generation="Calibrated pair edges and constrained maximum-spanning unions",stream_identity="LightGBM weak pair compatibility",
                     continuation_model=None,challenge_oof_macro_f1=None,challenge_stress_macro_f1=None,sparse_cohort_metrics=None,
                     candidate_recall="Downstream proxy recalls in v2_learned_candidate_recall.csv",per_class_f1=None,
                     reproducible_source_commit="af39cd3",v2_disposition="rejected_downstream_transfer",
                     metric_note="Auxiliary precision/recall/AUC/AP are not challenge macro-F1")
    rows.append(auxiliary)
    lines.append(f"| v2_weak_pair_auxiliary | auxiliary_weak_pair_calibration_NOT_challenge | n/a | n/a | {auxiliary['runtime_seconds']:.1f} | rejected_downstream_transfer | Conservative weak pair membership; downstream transfer failed |\n")
    with (ROOT/"reports/experiment_results.jsonl").open("wb") as f:
        f.write(original_json)
        if not original_json.endswith(b"\n"):f.write(b"\n")
        f.write("".join(json.dumps(r)+"\n" for r in rows).encode())
    with (ROOT/"reports/experiment_log.md").open("wb") as f:
        f.write(original_md)
        if not original_md.endswith(b"\n"):f.write(b"\n")
        f.write("".join(lines).encode())
    assert (ROOT/"reports/experiment_results.jsonl").read_bytes().startswith(original_json)
    assert (ROOT/"reports/experiment_log.md").read_bytes().startswith(original_md)
    print("Preserved V1 ledger prefix; V2 records",len(rows),"including one auxiliary calibration")


if __name__=="__main__":main()
