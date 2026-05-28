from pathlib import Path

DIR = Path("./evaluation/eval_results/alphafold3/TEST")

for p in DIR.iterdir():
    if not p.is_dir():
        continue

    name = p.name
    chars = list(name)
    new = name

    for i, ch in enumerate(chars):
        if ch.isalpha():
            chars[i] = ch.lower()
            new = "".join(chars)
            break

    if new == name:
        continue

    target = p.with_name(new)
    if target.exists():
        print(f"[SKIP] collision: {name} -> {new} (target exists)")
        continue

    p.rename(target)
    print(f"[RENAMED] {name} -> {new}")
