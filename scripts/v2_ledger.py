"""Append V2 evidence while preserving the exact frozen V1 ledger prefix."""
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
        elif "pair_" in name:status="rejected_identity_transfer"
        elif "sparse_" in name:status="accepted_component_or_ablation"
        else:status="frozen_diagnostic"
        r["v2_disposition"]=status
        lines.append(f"| {name} | {r['evaluation_split']} | {r['macro_f1']:.6f} | {r['accuracy']:.6f} | {r['runtime_seconds']:.1f} | {status} | {r['hypothesis']} |\n")
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
    print("Preserved V1 ledger prefix; V2 executed rows",len(rows))


if __name__=="__main__":main()
