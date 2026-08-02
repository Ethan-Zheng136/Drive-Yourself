import glob, csv, sys

tag, nshard = sys.argv[1], int(sys.argv[2])
root = "/mnt/pfs/zhengguantian/autovla/exp"
rows = []
for i in range(nshard):
    cs = sorted(glob.glob(f"{root}/persona_{tag}_shard{i}/*/*.csv"))
    if cs:
        rows += list(csv.DictReader(open(cs[-1])))

sc = [float(r["score"]) for r in rows if r.get("score") not in (None, "")]
if sc:
    print(f"\n===== persona[{tag}] PDMS = {sum(sc)/len(sc):.4f}  (n={len(sc)}) =====")
else:
    print("no scores")


def m(col):
    v = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return sum(v) / len(v) if v else float("nan")


for col in [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
    "driving_direction_compliance",
]:
    print(f"    {col:34s} {m(col):.4f}")
