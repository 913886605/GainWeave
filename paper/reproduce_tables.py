"""Regenerate GainWeave paper Tables II–IV from supported case-level measurements."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

PAPER = Path(__file__).resolve().parent
STRENGTHS = (-2, -1.5, *[i / 8 for i in range(-8, 9)], 1.5, 2)
DOMAINS = ("full", "lost", "bright")
DIGITAL = ("full", "digital255", "bright")
Q80 = ("fixed_native", "learned_native", "fixed_copy", "learned_copy")
VDS = ("linear_sdr", "anchor", "one", "rgb", "five", "gmnet", "vid2hdrimg", "x2hdr_ldr",
       "lediff", "fixed_copy", "learned_copy")
ARCH = tuple(f"{a}_{h}" for h in ("H0", "H") for a in ("five", "one", "rgb"))


def strength_id(a):
    return f"fixed_{a:+.3f}_copy"


def mean(values):
    return math.fsum(values) / len(values)


def read_rows(directory, name):
    compressed = directory / f"{name}.csv.gz"
    if compressed.exists():
        fp = gzip.open(compressed, "rt", encoding="utf-8", newline="")
    else:
        fp = (directory / f"{name}.csv").open(newline="", encoding="utf-8")
    with fp:
        yield from csv.DictReader(fp)


def aggregate(rows, methods, domains, expected_count, is_vds=False, expected_sources=None):
    """Within each supported case, MAE is per RGB component. Then weight view, EV, source."""
    cases = {m: set() for m in methods}
    bindings = {}
    cells = defaultdict(list)
    for row in rows:
        m, case, source, ev = row["method"], row["case"], row["source"], row["ev"]
        if m not in cases or case in cases[m]:
            raise ValueError(f"Unexpected method or duplicate case: {m} {case}")
        if not source or (is_vds and (case != source or ev != "")):
            raise ValueError(f"Invalid scene binding: {case}")
        if not is_vds and int(ev) not in (-2, -1, 0, 1, 2):
            raise ValueError(f"Invalid EV: {ev}")
        cases[m].add(case)
        counts = tuple(int(row[f"n_{d}"]) for d in domains)
        key = (source, ev, *counts)
        if bindings.setdefault(case, key) != key or counts[0] <= 0:
            raise ValueError(f"Support or source mismatch: {case}")
        for d, count in zip(domains, counts):
            value = row[d]
            if count < 0 or (count == 0) != (value == ""):
                raise ValueError(f"Missing or invalid support: {case}/{d}")
            if count:
                val = float(value)
                if not math.isfinite(val) or val < 0:
                    raise ValueError(f"Invalid MAE: {case}/{d}")
                cells[m, d, source, ev].append(val)
    ids = set(bindings)
    if len(ids) != expected_count or any(c != ids for c in cases.values()):
        raise ValueError(f"Expected {expected_count} paired cases per arm, found {len(ids)}")
    sources = {key[0] for key in bindings.values()}
    if expected_sources is not None and len(sources) != expected_sources:
        raise ValueError(f"Expected {expected_sources} sources, found {len(sources)}")
    by_source = defaultdict(list)
    for (m, d, source, ev), views in cells.items():
        by_source[m, d, source].append(mean(views))
    source_means = defaultdict(list)
    for (m, d, _), exposures in by_source.items():
        source_means[m, d].append(mean(exposures))
    return {(m, d): mean(values) for (m, d), values in source_means.items()}


def write_table(out, name, fields, rows):
    with (out / f"{name}.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        w.writerows(rows)


def table_ii(scores, out):
    q = aggregate(read_rows(scores, "laval_q80"), Q80, DOMAINS, 80, expected_sources=16)
    gains = [strength_id(a) for a in STRENGTHS]
    dev = aggregate(read_rows(scores, "laval_dev"), ["learned_copy", *gains], DOMAINS, 4335,
                    expected_sources=218)
    selected = min(STRENGTHS, key=lambda a: (dev[strength_id(a), "lost"], abs(a), a))
    rows = [["Q80", m, *[q[m, d] for d in DOMAINS]] for m in Q80]
    for m in [strength_id(selected), "learned_copy"]:
        rows.append(["full_dev", m, *[dev[m, d] for d in DOMAINS]])
    write_table(out, "table_II", ["split", "method", *DOMAINS], rows)
    write_table(out, "gain_scan", ["gain_ev", *DOMAINS],
                [[a, *[dev[strength_id(a), d] for d in DOMAINS]] for a in STRENGTHS])
    return selected


def table_iii(scores, out):
    vds = aggregate(read_rows(scores, "vds"), VDS, DIGITAL, 48, is_vds=True, expected_sources=48)
    write_table(out, "table_III", ["calibration", "method", *DIGITAL],
                [["anchor_shared" if m.endswith("_copy") else "per_method", m,
                  *[vds[m, d] for d in DIGITAL]] for m in VDS])


def table_iv(scores, out):
    arch_rows = []
    arch = {}
    for camera, domains in (("hard_clip", DOMAINS), ("soft_compression", ("full", "bright"))):
        arch[camera] = aggregate(read_rows(scores, f"features_{camera}"), ARCH, domains, 4335,
                                 expected_sources=218)
    for m in ARCH:
        hard, soft = arch["hard_clip"], arch["soft_compression"]
        arch_rows.append([*m.split("_"), hard[m, "lost"], hard[m, "full"], hard[m, "bright"],
                          soft[m, "full"], soft[m, "bright"]])
    write_table(out, "table_IV", ["arm", "output", "hard_lost", "hard_full", "hard_bright",
                                  "soft_full", "soft_bright"], arch_rows)
    reductions = {}
    for output in ("H0", "H"):
        five = hard[f"five_{output}", "lost"]
        rgb = hard[f"rgb_{output}", "lost"]
        reductions[output] = 100 * (rgb - five) / rgb
    return reductions


def tables(scores, out, only=("II", "III", "IV")):
    out.mkdir(exist_ok=True, parents=True)
    result = {}
    if "II" in only:
        result["selected_gain_ev"] = table_ii(scores, out)
    if "III" in only:
        table_iii(scores, out)
    if "IV" in only:
        result["five_reduction_vs_rgb_percent"] = table_iv(scores, out)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def verify_archived_scores(directory):
    for name, expected in json.loads((directory / "sha256.json").read_text()).items():
        with (directory / name).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--scores", type=Path, help="Alternate directory of per-case scoring CSVs")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--only", nargs="+", choices=["II", "III", "IV"], default=["II", "III", "IV"])
    args = cli.parse_args()
    directory = args.scores or PAPER / "scores"
    if args.scores is None:
        verify_archived_scores(directory)
    print(json.dumps(tables(directory, args.output, args.only), indent=2))
