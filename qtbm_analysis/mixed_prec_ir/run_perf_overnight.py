#!/usr/bin/env python3
"""
One overnight batch of the performance study: eight energy indices for every
(material, n_rhs) group that has any, run sequentially and plotted immediately
after.

    n_rhs = 1   carbon-nanotube, carbon-chain, si-bulk
    n_rhs = 2   carbon-nanotube, carbon-chain, graphene
    n_rhs = 3   carbon-nanotube, si-bulk
    n_rhs = 4   carbon-nanotube, carbon-chain, si-bulk, graphene

12 mpperf.py invocations, 96 indices, 4 solvers and 2 variants each. Groups
with no indices at that n_rhs are absent from SELECTION and are not run. This
is the cost counterpart of run_overnight.py, which does the same for accuracy.

One experiment per group rather than one per material: the summary figure
draws a group of bars per index, and a material's four n_rhs groups together
would be 32 of them side by side, which is past the width at which a pair of
bars can still be compared. The four stay comparable through the pooled
speedup-against-n_rhs figure drawn per material at the end.

Why the indices are what they are
---------------------------------
Eight per group, spaced across the log10(kappa_inf) range that group covers
and including both ends. Cost is flat in kappa_inf and monotone in n_rhs, so
the eight are there to show that flatness holds over the whole range, while
n_rhs is what the groups vary.

Unstable runs
-------------
A row whose slowest repeat exceeds its fastest by more than the stability
limit is not a measurement of the solver -- it is a measurement of whatever
else the node was doing (see mpperf.py's check_stability). Such a run is
repeated, up to MAX_ATTEMPTS times, and the attempt with the fewest unstable
rows is the one plotted and indexed. Every attempt is kept in the analysis
file, since numbered experiments are never overwritten, and the index page
marks the superseded ones.

That is the honest way to remove the red outlines from a figure: measure
again on a quiet node until the repeats agree, rather than raising the limit
until nothing trips it.

Usage
-----
    nohup python3 run_perf_overnight.py >&! /scratch/yimili/mpperf_overnight_logs/driver.log &

(tcsh: nohup ... & detaches; >&! forces the redirect past noclobber.)

    python3 run_perf_overnight.py --preflight

checks every index in SELECTION against the material files -- that it exists,
that its n_rhs is what the group claims, and that its kappa_inf is valid --
and runs nothing. A stale selection costs a whole night, so this runs
automatically before the batch as well.

    python3 run_perf_overnight.py --replot

redraws every experiment this batch wrote and runs no solves.

    python3 run_perf_overnight.py --threads 2

runs the batch at a different thread cap. See the comment on THREADS: the
count changes which precision wins, not only how long the batch takes.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path

# Read, never set from inside a subprocess's own code: fixed here, in the
# parent, so every child inherits it. mpperf.py deliberately never sets these
# -- it reports what it was given -- and an unpinned OpenBLAS on this node
# costs up to 68x on a Block Thomas factorization.
#
# The count is not a tuning detail. Threading speeds up the factorization,
# which is what u_f halves, but not the complex128 residual, which is memory
# bound, so it moves the answer rather than just the runtime. Measured on
# si-bulk idx 1864, complex128 Block Thomas and the LU-IR speedup beside it:
#
#     threads     1      2      4      8     16     32
#     c128    224.3  150.4  103.0   81.4   77.4   83.7  ms
#     speedup  1.06   1.07   0.97   0.79   0.83   0.87
#
# Refinement wins at one and two threads and loses from four up. On
# carbon-nanotube the whole curve is flat (9.2-9.6 ms from 1 to 16): its
# blocks are 33x33, below the size at which OpenBLAS splits the work at all.
# 8 is the default because it is near the floor for the large materials,
# harmless for the small ones, and one fixed value keeps the four materials
# comparable. Whatever is chosen, mpperf.py records it in every experiment.
#
# Parsed before argparse because the environment must be set before numpy or
# any solver library is imported; too late once a pool exists.
def _threads_from_argv(default="8"):
    """
    The values of --threads, as a list of strings, without argparse.

    More than one means a sweep: every group is measured once at each count.
    The parent process does no arithmetic itself, so its own cap only matters
    for the single-count case, where it is what the children inherit.
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads":
            rest = []
            for b in argv[i + 1:]:
                if b.startswith("-"):
                    break
                rest.append(b)
            return rest or [default]
        if a.startswith("--threads="):
            return a.split("=", 1)[1].split(",")
    return [default]


THREAD_LIST = _threads_from_argv()
for _t in THREAD_LIST:
    if not _t.isdigit() or int(_t) < 1:
        sys.exit(f"--threads takes positive integers, got {_t!r}")
THREADS = THREAD_LIST[0]
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS

def _value_from_argv(flag, default):
    """
    One option's value, read before argparse for the same reason as --threads:
    the output paths below depend on it, and they are module constants. The
    string is kept as typed, since it also names a directory.
    """
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


INNER = _value_from_argv("--inner", "direct")
if INNER not in ("direct", "gmres", "both"):
    sys.exit(f"--inner takes direct, gmres or both, got {INNER!r}")

# The inner GMRES tolerance decides how many inner iterations a correction
# costs and is therefore the main term in the runtime of GMRES-IR. Two
# tolerances are two different methods as far as a timing is concerned, so
# each writes to its own tree.
GMRES_TOL = _value_from_argv("--gmres-tol", "1e-8")
try:
    float(GMRES_TOL)
except ValueError:
    sys.exit(f"--gmres-tol takes a number, got {GMRES_TOL!r}")

import h5py  # noqa: E402  (after the thread cap, before anything numeric)

HERE = Path(__file__).resolve().parent
MPPERF = HERE / "mpperf.py"
PLOT = HERE.parent / "plotting" / "mixed_prec_ir" / "plot_mpperf.py"
THREAD_PLOT = HERE.parent / "plotting" / "mixed_prec_ir" / "plot_mpthreads.py"
HDF5_DIR = Path("/scratch/yimili/matrices2/hdf5")
ANALYSIS_DIR = Path("/scratch/yimili/mixed-precision-IR")
# LU-IR keeps the original tree. Any other inner solve writes to its own tree
# beside it. The thread figure pools every experiment it finds at a thread
# count, so two inner solves sharing one file would be drawn as one curve.
if INNER == "direct":
    BASE_DIR = ANALYSIS_DIR
    _TAG = ""
else:
    _TAG = INNER if INNER == "both" else f"gmres/tol{GMRES_TOL}"
    BASE_DIR = ANALYSIS_DIR / _TAG
LOG_DIR = Path("/scratch/yimili/mpperf_overnight_logs"
               + ("" if not _TAG else "_" + _TAG.replace("/", "_")))
INDEX_HTML = BASE_DIR / "PERF_EXPERIMENT_INDEX.html"

# Eight indices per group, spread over that group's kappa_inf range. Keyed by
# (material, n_rhs); a group with no indices at that n_rhs is simply absent.
SELECTION = {
    ("carbon-nanotube", 1): [926, 994, 995, 996, 1600, 1601, 1602, 1603],
    ("carbon-chain", 1): [805, 929, 903, 895, 894, 891, 893, 892],
    ("si-bulk", 1): [910, 944, 962, 960, 952, 956, 954, 958],

    ("carbon-nanotube", 2): [1726, 1677, 1646, 1625, 1614, 1608, 1605, 1604],
    ("carbon-chain", 2): [8, 470, 602, 706, 762, 787, 796, 800],
    ("graphene", 2): [461, 1630, 1634, 1640, 1636, 1639, 1637, 1638],

    ("carbon-nanotube", 3): [1820, 1774, 777, 764, 788, 734, 727, 724],
    ("si-bulk", 3): [1800, 826, 796, 874, 804, 806, 884, 808],

    ("carbon-nanotube", 4): [358, 1851, 450, 484, 709, 527, 591, 575],
    ("carbon-chain", 4): [5855, 5327, 5504, 4917, 4900, 5397, 5394, 4885],
    ("si-bulk", 4): [746, 1794, 1852, 666, 718, 1866, 870, 1864],
    ("graphene", 4): [15, 187, 315, 358, 1995, 1845, 1767, 1715],
}

# Materials in the order their groups are run, so one material's file is
# written contiguously and its pooled n_rhs figure can be drawn as soon as its
# last group finishes.
MATERIAL_ORDER = ["carbon-nanotube", "carbon-chain", "si-bulk", "graphene"]

# Where one solver is so much slower than the rest that it sets the y axis and
# flattens them, the time panel is clipped at this many milliseconds and the
# bars that run past it are cut and marked with a caret. On si-bulk and
# graphene SuperLU is 4 to 6 times the next slowest solver -- it is
# single-threaded, and these are the two largest matrices -- which leaves the
# other three in the bottom fifth of the panel, where a pair of bars cannot be
# compared. One value per material rather than per experiment, so a material's
# figures stay comparable with each other. None means the axis is sized to the
# tallest bar. Kept here rather than passed by hand so that --replot draws the
# same figures the batch did.
YMAX = {"si-bulk": 200.0, "graphene": 130.0,
        "carbon-nanotube": None, "carbon-chain": None}

SOLVERS = ["block-thomas", "mumps", "cudss", "superlu"]
REPEATS = "9"
REDUCE = "min"
MAX_ATTEMPTS = 3

_EXPERIMENT_RE = re.compile(r"wrote .+:/experiments/(\d+)")


def groups():
    """(material, n_rhs, indices) in run order: material outer, n_rhs inner."""
    out = []
    for material in MATERIAL_ORDER:
        for (mat, rhs), indices in sorted(SELECTION.items(),
                                          key=lambda kv: kv[0][1]):
            if mat == material:
                out.append((material, rhs, indices))
    return out


def out_root(threads):
    """
    Where one thread count's results go.

    A single count writes to ANALYSIS_DIR, which is where the 8-thread batch
    already is. A sweep writes each count to its own t<N>/ below it, so that
    two counts never share a file: a pooled figure drawn over one file would
    otherwise put measurements from different machines on the same line.
    """
    if INNER == "direct" and len(THREAD_LIST) == 1:
        return BASE_DIR
    return BASE_DIR / f"t{threads}"


def perf_h5(material, threads=None):
    """Where mpperf.py writes this material's performance file."""
    return (out_root(threads if threads is not None else THREADS)
            / material / f"{material}_perf.h5")


def banner(msg):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"\n{'=' * 78}\n{ts}  {msg}\n{'=' * 78}", flush=True)


def preflight():
    """
    Check every index in SELECTION against the material file it names.

    Three things go wrong with a hand-picked index list, and all three are
    silent at run time: an index that is not in the file at all, one whose
    n_rhs is not the group's (the listing it was copied from was for another
    material, or the file was regenerated), and one whose kappa_inf never
    converged, which the summary figure drops. Each costs a group's worth of
    the night, so they are found here instead.

    Returns the number of problems.
    """
    banner("preflight: checking SELECTION against the material files")
    problems = 0
    for material in MATERIAL_ORDER:
        path = HDF5_DIR / f"{material}.h5"
        if not path.exists():
            print(f"  MISSING material file {path}", flush=True)
            problems += sum(len(v) for (m, _), v in SELECTION.items()
                            if m == material)
            continue

        cond = {}
        cond_path = Path("/scratch/yimili/condition-est") / f"{material}.h5"
        if cond_path.exists():
            try:
                with h5py.File(cond_path, "r") as cf:
                    g = cf["condition"]
                    cond = dict(zip(g["indices"][()].tolist(),
                                    zip(g["cond_inf"][()].tolist(),
                                        g["valid"][()].tolist())))
            except Exception as exc:                       # noqa: BLE001
                print(f"  [note] {material}: condition file unreadable "
                      f"({exc}); kappa is not checked", flush=True)

        with h5py.File(path, "r") as f:
            for (mat, rhs), indices in sorted(SELECTION.items(),
                                              key=lambda kv: kv[0][1]):
                if mat != material:
                    continue
                bad, kappas = [], []
                for i in indices:
                    key = f"E_{i}"
                    if key not in f:
                        bad.append(f"idx {i}: not in the material file")
                        continue
                    rhs_ds = f[key]["rhs"]
                    n = rhs_ds.shape[1] if rhs_ds.ndim > 1 else 1
                    if n != rhs:
                        bad.append(f"idx {i}: n_rhs is {n}, group claims {rhs}")
                    k, valid = cond.get(i, (float("nan"), None))
                    if valid is False:
                        bad.append(f"idx {i}: kappa_inf did not converge")
                    elif k == k:
                        kappas.append(k)
                span = (f"{min(kappas):.1e}..{max(kappas):.1e}"
                        if kappas else "unknown")
                mark = "ok " if not bad else "BAD"
                print(f"  {mark} {material:<17} n_rhs={rhs}  "
                      f"{len(indices)} indices  kappa_inf {span}", flush=True)
                for line in bad:
                    print(f"        {line}", flush=True)
                problems += len(bad)

    banner(f"preflight: {problems} problem(s)")
    return problems


def experiment_attrs(material, name, threads):
    """
    n_unstable and the row count of one written experiment.

    Read back from the file rather than scraped from stdout: mpperf.py stores
    the stability verdict as an attribute, so this cannot drift from what the
    run actually decided.
    """
    with h5py.File(perf_h5(material, threads), "r") as f:
        g = f["experiments"][name]
        return (int(g.attrs.get("n_unstable", -1)),
                int(g["runs"]["idx"].shape[0]))


def run_group(material, rhs, indices, attempt, log_dir, threads):
    """
    One mpperf.py invocation. Returns (experiment_name, n_unstable, n_rows,
    log_path), with experiment_name None if the run failed or wrote nothing.
    """
    log = log_dir / f"{material}__rhs{rhs}__try{attempt}.log"
    argv = [sys.executable, str(MPPERF), str(HDF5_DIR / f"{material}.h5"),
            "--material", material,
            "--outdir", str(out_root(threads)),
            "--idx", *[str(i) for i in indices],
            "--solvers", *SOLVERS,
            "--inner", INNER,
            "--gmres-tol", GMRES_TOL,
            "--repeats", REPEATS,
            "--reduce", REDUCE]

    # The cap is read when numpy is imported, so it can only be set by the
    # process that is about to do the work, never changed inside one.
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)

    with open(log, "w") as f:
        f.write(f"OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
                + " ".join(argv) + "\n\n")
        f.flush()
        result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                                env=env)

    if result.returncode != 0:
        print(f"    FAILED (exit {result.returncode}) -- see {log}", flush=True)
        return None, -1, 0, log

    match = _EXPERIMENT_RE.search(log.read_text())
    if match is None:
        print(f"    ran but wrote no experiment -- see {log}", flush=True)
        return None, -1, 0, log

    name = f"{int(match.group(1)):04d}"
    n_unstable, n_rows = experiment_attrs(material, name, threads)
    return name, n_unstable, n_rows, log


def draw(material, name, log_dir, label, threads):
    """Draw one experiment's summary figure and report."""
    plog = log_dir / f"{label}__plot.log"
    argv = [sys.executable, str(PLOT), str(perf_h5(material, threads)),
            "--experiment", name]
    ymax = YMAX.get(material)
    if ymax is not None:
        argv += ["--ymax", str(ymax)]
    with open(plog, "w") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print(f"    (plot failed, exit {r.returncode} -- see {plog}; the "
              f"experiment itself is recorded)", flush=True)
    return r.returncode == 0


def draw_nrhs(material, names, log_dir, threads):
    """
    The pooled speedup-against-n_rhs figure for one material.

    Restricted to this batch's experiments: the file also holds older runs
    made with other reducers and repeat counts, and pooling those into one
    line would compare measurements that are not comparable.
    """
    if len(names) < 2:
        print(f"  [skip] {material}: only {len(names)} group(s), the n_rhs "
              f"figure needs at least two", flush=True)
        return
    plog = log_dir / f"{material}__nrhs__plot.log"
    argv = [sys.executable, str(PLOT), str(perf_h5(material, threads)),
            "--nrhs", "--experiments", *names]
    with open(plog, "w") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        r = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
    print(f"  n_rhs figure over experiments {', '.join(names)}: "
          f"{'ok' if r.returncode == 0 else f'FAILED, see {plog}'}", flush=True)


def write_index_html(records, failures):
    """
    (Re)write PERF_EXPERIMENT_INDEX.html from scratch after every group, so a
    batch interrupted at 3am still leaves a correct index for what finished.
    """
    by_material = {}
    for r in records:
        by_material.setdefault(r["material"], []).append(r)

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>mpperf overnight batch -- experiment index</title>",
             "<style>",
             "body{font-family:system-ui,sans-serif;margin:2em;color:#222}",
             "h1{font-size:1.3em}h2{margin-top:2em;border-bottom:2px solid #ccc}",
             "table{border-collapse:collapse;width:100%;margin-top:0.5em}",
             "th,td{border:1px solid #ccc;padding:4px 10px;text-align:left;"
             "font-size:0.92em}",
             "th{background:#eee}tr:nth-child(even){background:#f8f8f8}",
             "code{background:#eee;padding:1px 4px;border-radius:3px}",
             ".fail{color:#a00}.old{color:#888}.warn{color:#b45309}",
             "</style></head><body>",
             "<h1>mpperf.py overnight batch -- experiment index</h1>",
             f"<p>Generated "
             f"{datetime.datetime.now().isoformat(timespec='seconds')} by "
             f"run_perf_overnight.py. {REPEATS} repeats, reduced by "
             f"{REDUCE}, {THREADS} threads, solvers "
             f"{', '.join(SOLVERS)}. Experiment numbers continue from what "
             f"was already in these files, so they do not start at 1.</p>",
             "<p>A row marked <span class='old'>superseded</span> is an "
             "attempt whose repeats disagreed; it was re-run and is kept only "
             "because numbered experiments are never overwritten.</p>"]

    for material, rows in by_material.items():
        parts.append(f"<h2>{material}</h2>")
        parts.append(f"<p><code>{perf_h5(material)}</code></p>")
        parts.append("<table><tr><th>experiment</th><th>n_rhs</th>"
                     "<th>indices</th><th>rows</th><th>unstable</th>"
                     "<th>threads</th><th>attempt</th><th>y-max</th>"
                     "<th>started</th><th>log</th></tr>")
        for r in sorted(rows, key=lambda r: (int(r.get("threads", 8)),
                                            r["rhs"],
                                            int(r["experiment"]))):
            cls = " class='old'" if r["superseded"] else (
                " class='warn'" if r["n_unstable"] else "")
            state = ("superseded" if r["superseded"]
                     else f"{r['n_unstable']} of {r['n_rows']}")
            parts.append(f"<tr{cls}><td>{r['experiment']}</td>"
                         f"<td>{r['rhs']}</td>"
                         f"<td>{' '.join(str(i) for i in r['indices'])}</td>"
                         f"<td>{r['n_rows']}</td>"
                         f"<td>{state}</td>"
                         f"<td>{r.get('threads', THREADS)}</td>"
                         f"<td>{r['attempt']} of {MAX_ATTEMPTS}</td>"
                         f"<td>{YMAX.get(material) or 'auto'}</td>"
                         f"<td>{r['timestamp']}</td>"
                         f"<td><code>{r['log'].name}</code></td></tr>")
        parts.append("</table>")

    if failures:
        parts.append("<h2 class='fail'>failed (no experiment written)</h2>")
        parts.append("<table><tr><th>material</th><th>n_rhs</th>"
                     "<th>started</th><th>log</th></tr>")
        for f in failures:
            parts.append(f"<tr class='fail'><td>{f['material']}</td>"
                         f"<td>{f['rhs']}</td><td>{f['timestamp']}</td>"
                         f"<td><code>{f['log'].name}</code></td></tr>")
        parts.append("</table>")

    parts.append("</body></html>")
    INDEX_HTML.write_text("\n".join(parts))


def replot():
    """
    Redraw every experiment this batch wrote, and nothing else.

    The figures are a function of the stored experiment, so a change to
    plot_mpperf.py reaches them by running it again; the solves cost a night
    and do not need repeating. Which experiments those are is read from the
    index page, so an experiment made outside this batch is left alone.
    """
    if not INDEX_HTML.exists():
        print(f"no {INDEX_HTML} -- nothing to replot; run the batch first")
        return
    text = INDEX_HTML.read_text()
    drawn = 0
    for material in MATERIAL_ORDER:
        # The index lists each material's experiments in its own section.
        section = text.split(f"<h2>{material}</h2>")
        if len(section) < 2:
            continue
        body = section[1].split("<h2>")[0]
        names = [m for m in re.findall(r"<tr[^>]*><td>(\d{4})</td>", body)]
        kept = [n for n, row in zip(names, body.split("<tr")[1:])
                if "superseded" not in row]
        banner(f"replot: {material} ({len(kept)} experiments)")
        for name in kept:
            argv = [sys.executable, str(PLOT), str(perf_h5(material)),
                    "--experiment", name]
            ymax = YMAX.get(material)
            if ymax is not None:
                argv += ["--ymax", str(ymax)]
            r = subprocess.run(argv, capture_output=True, text=True)
            print(f"  {name}: {'ok' if r.returncode == 0 else 'FAILED'}",
                  flush=True)
            if r.returncode != 0:
                print(r.stdout + r.stderr, flush=True)
            drawn += r.returncode == 0
        if kept:
            draw_nrhs(material, kept, LOG_DIR)
    banner(f"replot done: {drawn} experiments redrawn")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preflight", action="store_true",
                    help="check SELECTION against the material files and exit")
    ap.add_argument("--replot", action="store_true",
                    help="redraw this batch's experiments and run no solves")
    ap.add_argument("--force", action="store_true",
                    help="run the batch even if the preflight found problems")
    ap.add_argument("--threads", nargs="+", default=THREAD_LIST, metavar="N",
                    help=f"OMP_NUM_THREADS and OPENBLAS_NUM_THREADS (default "
                         f"{THREADS}). Several values measure every group "
                         f"once at each count, into t<N>/ directories. Read "
                         f"before argparse, since the cap must be set before "
                         f"numpy is imported; listed here so --help shows it")
    ap.add_argument("--inner", default=INNER,
                    choices=("direct", "gmres", "both"),
                    help=f"inner correction solve (default {INNER}). direct "
                         f"is LU-IR, gmres is GMRES-IR, both times the two "
                         f"side by side. Anything but direct writes to its "
                         f"own tree under the analysis directory, so the two "
                         f"are never pooled into one figure. Read before "
                         f"argparse, since the output paths depend on it")
    ap.add_argument("--gmres-tol", default=GMRES_TOL, metavar="TOL",
                    help=f"inner GMRES tolerance (default {GMRES_TOL}). Each "
                         f"tolerance writes to gmres/tol<TOL>/, because the "
                         f"tolerance sets the inner iteration count and so "
                         f"two tolerances are two different costs. Read "
                         f"before argparse, since the output paths depend "
                         f"on it")
    args = ap.parse_args()

    if args.replot:
        replot()
        return

    if args.preflight:
        sys.exit(1 if preflight() else 0)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if preflight() and not args.force:
        print("\nrefusing to start: fix SELECTION, or pass --force to run "
              "anyway.\nA stale index list costs the whole night, which is "
              "why this is checked first.", flush=True)
        sys.exit(1)

    todo = [(t, m, r, i) for t in THREAD_LIST for m, r, i in groups()]
    records, failures = [], []
    kept = {}                      # (threads, material) -> [experiment names]
    done = 0

    for threads, material, rhs, indices in todo:
        done += 1
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        banner(f"[{done}/{len(todo)}] {material}  n_rhs={rhs}  "
               f"{len(indices)} indices  threads={threads}")

        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"  attempt {attempt} of {MAX_ATTEMPTS}", flush=True)
            name, n_unstable, n_rows, log = run_group(
                material, rhs, indices, attempt, LOG_DIR, threads)
            if name is None:
                failures.append(dict(material=material, rhs=rhs,
                                     threads=threads, timestamp=ts, log=log))
                write_index_html(records, failures)
                break
            attempts.append(dict(material=material, rhs=rhs, indices=indices,
                                 threads=threads,
                                 experiment=name, n_unstable=n_unstable,
                                 n_rows=n_rows, attempt=attempt,
                                 timestamp=ts, log=log, superseded=False))
            print(f"    -> experiment {name}: {n_unstable} of {n_rows} rows "
                  f"unstable", flush=True)
            if n_unstable == 0:
                break
            if attempt < MAX_ATTEMPTS:
                print(f"    repeats disagreed; measuring again", flush=True)

        if not attempts:
            continue

        # Keep the attempt that measured the machine least.
        best = min(attempts, key=lambda a: a["n_unstable"])
        for a in attempts:
            a["superseded"] = a is not best
        records.extend(attempts)

        if best["n_unstable"]:
            print(f"  still {best['n_unstable']} unstable row(s) after "
                  f"{len(attempts)} attempt(s); drawing experiment "
                  f"{best['experiment']} with them outlined", flush=True)
        draw(material, best["experiment"], LOG_DIR,
             f"t{threads}__{material}__rhs{rhs}", threads)
        kept.setdefault((threads, material), []).append(best["experiment"])
        write_index_html(records, failures)

    for (threads, material), names in kept.items():
        banner(f"pooled n_rhs figure: {material}  threads={threads}")
        draw_nrhs(material, names, LOG_DIR, threads)

    if len(THREAD_LIST) > 1:
        draw_threads()

    good = [r for r in records if not r["superseded"]]
    unstable = sum(r["n_unstable"] for r in good)
    banner(f"batch done: {len(good)}/{len(todo)} groups measured, "
           f"{len(failures)} failed, {unstable} unstable row(s) remaining "
           f"-- see {INDEX_HTML}")


def thread_roots():
    """
    Every t<N>/ directory that already holds results, ordered by N.

    Read off the filesystem rather than from THREAD_LIST, because the thread
    figure is shared: a run that measures only two counts must extend the
    curves with those two, not replace a six-count figure with a two-point
    one.
    """
    found = {}
    for path in BASE_DIR.glob("t*"):
        if path.is_dir() and path.name[1:].isdigit():
            found[int(path.name[1:])] = path
    return [found[n] for n in sorted(found)]


def draw_threads():
    """
    The speedup-against-thread-count figure, one per n_rhs.

    One figure per n_rhs rather than one for everything: speedup falls with
    n_rhs as well as with the thread count, so a curve pooled over several
    n_rhs would mix the two effects and could not be read as either.
    """
    every_rhs = sorted({rhs for (_m, rhs) in SELECTION})
    for rhs in every_rhs:
        materials = [m for (m, r) in SELECTION if r == rhs]
        files = []
        for root in thread_roots():
            for m in materials:
                path = root / m / f"{m}_perf.h5"
                if path.exists():
                    files.append(str(path))
        if not files:
            continue
        banner(f"thread figure: n_rhs={rhs}")
        argv = [sys.executable, str(THREAD_PLOT), *files,
                "--n-rhs", str(rhs), "--outdir", str(BASE_DIR / "threads")]
        if INNER == "gmres":
            argv += ["--refined", "c64_gmres"]
        r = subprocess.run(argv, capture_output=True, text=True)
        print((r.stdout + r.stderr).strip(), flush=True)


if __name__ == "__main__":
    main()
