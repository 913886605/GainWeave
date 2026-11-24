"""Verify paper split membership, camera/EV keys, exclusions, and frozen file hashes."""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"Digest mismatch: {name}")
    views = read("laval_views.csv")
    lookup = {r["view_id"]: r for r in views}
    require(len(lookup) == len(views) == 8060, "Laval physical-view count or duplicate")
    usable = {k for k, row in lookup.items() if row["eligible"] == "1"}
    excluded = read("laval_excluded_views.csv")
    ex_ids = {r["view_id"] for r in excluded}
    require(len(ex_ids) == len(excluded) == 60 and ex_ids == lookup.keys() - usable, "Exclusion membership")
    require(Counter(r["split"] for r in excluded) == {"dev": 5, "train": 55}, "Exclusion split")
    require(len({r["source_id"] for r in excluded}) == 58, "Exclusion source count")
    for r in excluded:
        require(all(lookup[r["view_id"]][k] == v for k, v in r.items()), "Exclusion record mismatch")
    source_sets = {}
    for split, source_count, view_count in [("train", 1797, 7133), ("dev", 218, 867)]:
        group = [lookup[v] for v in usable if lookup[v]["split"] == split]
        sources = {r["source_id"] for r in group}
        require(len(sources) == source_count and len(group) == view_count, f"{split} membership")
        require(sources == {r["source_id"] for r in views if r["split"] == split}, "A source was lost")
        source_sets[split] = sources
    require(source_sets["dev"].isdisjoint(source_sets["train"]), "Panorama overlap")
    types = {r["view_type"] for r in views}
    require(types == {"hash_0","hash_1","hash_2","hotspot"}, "Laval view types")
    members = defaultdict(set)
    for r in views:
        members[r["source_id"]].add((r["source_sha256"],r["path_sha256"],r["source_bytes"]))
        require(bool(int(r["path_sha256"],16) % 10) == (r["split"] == "train"), "Laval hash split")
    require(all(len(values)==1 for values in members.values()), "Inconsistent panorama identity")
    require(len({next(iter(v))[0] for v in members.values()})==len(members), "Duplicate panorama content")
    cases = read("laval_dev_cases.csv")
    require(len(cases) == len({r["case_id"] for r in cases}) == 8670, "Case count")
    dev = {v for v in usable if lookup[v]["split"] == "dev"}
    expected = {(v,e,c) for v in dev for e in range(-2,3) for c in ("hard_clip","soft_compression")}
    registered = set()
    for r in cases:
        v = lookup[r["view_id"]]
        e = int(r["ev"])
        registered.add((r["view_id"],e,r["camera"]))
        require(r["case_id"] == f"V{int(v['allowlist_index']):04d}_EV{e+2}_{r['camera']}", "Case key")
        require(r["source_id"] == v["source_id"], "Case/source mismatch")
    require(expected == registered, "Missing view, EV or camera")
    q = [r for r in cases if r["q80_id"]]
    require({r["q80_id"] for r in q} == {f"Q_{i:03d}" for i in range(81,161)}, "Q80 case IDs")
    require(len(q)==80 and len({r["source_id"] for r in q})==len({r["view_id"] for r in q})==16,
            "Q80 membership")
    require(all(r["camera"]=="hard_clip" for r in q), "Q80 camera")
    vds = read("vds_scenes.csv")
    require(Counter(r["split"] for r in vds)=={"CAL":20,"TEST":48}, "VDS membership")
    require(len({r["scene"] for r in vds})==68, "VDS CAL/TEST overlap")
    for r in vds:
        split = "train" if r["split"]=="CAL" else "test"
        name = r["official_id"]
        require(r["scene"] == "VDS_"+name and r["official_split"]==split, "VDS ID")
        require(r["input_ev"]=="0" and r["sdr_member"]==f"{split}_set/{name}/{name}_0EV_true.jpg.png"
                and r["reference_member"]==f"{split}_hdr_set/{name}/target_hdr.hdr", "VDS members")
    print("Verified 8,060 Laval views (60 excluded), 8,670 camera/EV cases, 80 nested Q cases, "
          "and 20 CAL / 48 TEST VDS pairs.")


if __name__ == "__main__":
    main()
