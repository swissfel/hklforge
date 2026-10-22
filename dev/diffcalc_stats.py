"""Tally of the planner-vs-diffcalc comparison (tests/test_solve_diffcalc.py)
over many seeds: per mode family, cases with solutions, branch counts, skips."""
import collections, sys, time
import numpy as np, pytest
sys.path[:0] = ["src", "tests"]
import test_solve_diffcalc as T

def family(names):
    if "bisect" in names:
        return f"{names[0]}+bisect+{names[1]}"
    kinds = [n if n in ("qaz", "naz", "psi") else "det" if n in T.DETECTOR else "ref" if n in T.REFS
             else "samp" for n in names]
    return "+".join(f"{kinds.count(k)}{k}" for k in dict.fromkeys(kinds))

stats = collections.defaultdict(lambda: collections.Counter())
skips = collections.Counter()
t0 = time.time()
for names in T.MODES:
    for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 30):
        f = family(names)
        try:
            n, m = T.compare(names, seed)
        except pytest.skip.Exception as e:
            skips[str(e)[:45]] += 1; stats[f]["skipped"] += 1; continue
        if m is None:
            stats[f]["diffcalc failed, hklforge verified by diffcalc get_hkl"] += 1; continue
        stats[f]["cases"] += 1; stats[f]["with solutions"] += n > 0
        stats[f][f"{n} branches"] += 1
print(f"{time.time() - t0:.1f} s, every non-skipped case: identical branch sets")
for f, c in stats.items():
    print(f, dict(sorted(c.items())))
print("skips:", dict(skips))
print("diffcalc bisect-mode positions violating bisect/omega:", sum(T.BISECT_BAD),
      "of", len(T.BISECT_BAD), "bisect comparisons")
print("diffcalc crashes / self-check failures:", collections.Counter(c[1] for c in T.CRASHES))
