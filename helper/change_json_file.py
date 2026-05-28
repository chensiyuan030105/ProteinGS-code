from pathlib import Path
import json
import re

def fix_aggregate_input_paths(
    root="/home/mhg/ForSiyuan/proteinstudio/release/evaluation/aggregate_input",
    dry_run=False,
):
    root = Path(root)


    pat = re.compile(r'(?<!release/)evaluation/')

    def normalize_path(s: str) -> str:

        while "release/release/evaluation/" in s:
            s = s.replace("release/release/evaluation/", "release/evaluation/")

        return pat.sub("release/evaluation/", s)

    for fp in root.rglob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[SKIP] {fp} (parse error: {e})")
            continue

        changed = False
        # data: { model_name: { "dataset_path": {...}, ... }, ... }
        for _, cfg in data.items():
            dp = cfg.get("dataset_path")
            if isinstance(dp, dict):
                for k, v in list(dp.items()):
                    if isinstance(v, str):
                        nv = normalize_path(v)
                        if nv != v:
                            dp[k] = nv
                            changed = True

        if changed:
            if dry_run:
                print(f"[DRY] would update: {fp}")
            else:
                fp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"[OK] updated: {fp}")

fix_aggregate_input_paths(dry_run=False)

