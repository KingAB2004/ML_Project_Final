"""Fail loudly if supporter private reasoning reached any dialogue in a run dir."""
import glob, json, re, sys

PAT = re.compile(r"(?im)^\s*(analysis|strategy)\s*:|</?(analysis|strategy|response)>|what you say to them")
bad = 0
for f in sorted(glob.glob(f"{sys.argv[1]}/dialogues/*.jsonl")):
    sup = [t["text"] for l in open(f) for t in json.loads(l)["turns"] if t["role"] == "supporter"]
    n = sum(bool(PAT.search(t)) for t in sup)
    bad += n
    print(f"{f.split('/')[-1]}: {len(sup)} supporter turns, {n} leaking")
sys.exit(1 if bad else 0)
