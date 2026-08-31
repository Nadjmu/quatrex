# Running one material end to end

What to type, in what order, and what to check after each stage. The pipeline
itself is described in [`README.md`](README.md); this file is the operational
account of running it, written after taking graphene through from a bare
packed file and si-bulk through from a partial one on 2026-08-30/31.

Every number below was measured on the machines named, not estimated.

---

## 1. The two nodes

Each node has its own `/scratch/yimili`. Nothing is shared between them.

| Node | Interpreter | Has | Used for |
|---|---|---|---|
| `mont-fort1` | `/scratch/yimili/envs/nla_lab_cpu/bin/python` | SuperLU, UMFPACK, MUMPS. Intel Xeon E5-2680 v4, 28 cores | stages 1–4 |
| `attelas1` | `/home/msc26f15/.conda/envs/nla_lab_gpu/bin/python` | CuPy, nvmath (cuDSS), MUMPS, **no UMFPACK**. GH200 Grace, ARM, 72 cores | cuDSS, stages 5–7 |

**`attelas1` is canonical.** Material files, analysis files and figures are
read from there. `mont-fort1` is a compute node that feeds it; its copies are
stale by design and safe to delete when space is needed. There is no sync back.

The two are different architectures, so **wall times are not comparable across
them**. Error quantities are: `omega` for si-bulk index 0 agreed to one
significant figure between an Intel and an ARM run of the same solver.

### tcsh

The login shell on both nodes is tcsh, not bash.

- No `VAR=val cmd` — use `setenv` or `env VAR=val cmd`.
- No `2>/dev/null` — that is `Ambiguous output redirect`. Use `>&`.
- No `<<'EOF'` heredocs; they hang.
- `if`/`then` in a one-liner will not parse.

Put anything non-trivial in a `#!/bin/bash` script and run that. Every long job
below is launched as

```
ssh -f <node> 'chmod +x /tmp/job.sh; nohup /tmp/job.sh >& /scratch/yimili/job.log < /dev/null &'
```

which detaches it fully: the process is reparented to init with no controlling
terminal, so it survives the laptop being shut down.

### Cross-node transfers

`mont-fort1` has **no ssh key of its own**. Node-to-node ssh works only inside
a forwarded-agent session, because `~/.ssh/config` on the laptop sets
`ForwardAgent yes`. A detached `nohup` job has no `SSH_AUTH_SOCK` and fails
with `Permission denied (publickey,password)`.

**Never put an `rsync` to another node inside an unattended job.** Run it from
the laptop, holding the agent open:

```bash
ssh -A mont-fort1 'rsync -a --partial --inplace --info=progress2 \
    /scratch/yimili/matrices2/hdf5/<material>.h5 \
    attelas1:/scratch/yimili/matrices2/hdf5/'
```

No `-z`: the link runs at 112 MB/s and gzip on HDF5 float payload is slower
than that. 185 GB took 26 minutes.

Validating connectivity from an interactive `ssh` tests the forwarded agent,
not the job's environment. On 2026-08-30 that mistake cost a full day: the
overnight rsync failed at 06:05 and the dependent chain waited on a sentinel
that never appeared.

---

## 2. The stages

Ordering is forced by three dependencies and nothing else:

- `forward_error.py` reads the condition file, so `condition_est.py` must have run.
- Every stage-4 script reads stored factors or stored solutions, so the
  benchmarks must have run.
- cuDSS exists only on the GPU node, so the material file must be there before
  `forward_error.py` can see a cuDSS row.

### Stage 1–2 — export and pack

Only if the material is not already packed.

```bash
python main3.py                            # writes matrices2/<material>/
python make_hdf5.py --material <name> [--stride N]
```

`--stride N` keeps every Nth already-exported index **by literal index value**,
so `energy(i) = grid_energy_min + resolution * i` still holds and nothing
downstream needs to know a stride was used.

**Check:**

```bash
python solvers/verify.py material <name>
```

Confirm the index count, and that the number with a non-empty right-hand side
is what you expect. Only those indices cost anything: every driver skips
`rhs.shape[-1] == 0` before doing work.

| material | indices | non-empty rhs |
|---|---|---|
| carbon-nanotube | 2601 | 2194 |
| carbon-chain | 5881 | 2197 |
| si-bulk | 1386 (packed at stride 2) | 1003 |
| graphene | 2001 | 919 |

### Stage 2b — band edges

**Do this before any analysis runs**, because every stage-4 script copies the
band-edge attrs out of the material file at write time.

The registry is the authority, but `make_hdf5.py` prefers the values the export
carries and writes them once at pack time. A registry edit after packing
therefore does not reach the file. Restamp it:

```bash
python solvers/verify.py band-edges <name>
```

**Check the edges against the data, not just against the registry.** The
boundaries of the transmitting region are visible in the right-hand sides: the
last index with open channels below the gap and the first above it. For
carbon-nanotube, carbon-chain and si-bulk the registry agreed with those to
within 0.2 eV. For graphene it did not — the registry said valence 0.499 /
conduction 0.500, while the RHS is empty from +0.032 to +1.115 eV, a gap of
1.08 eV. Both markers would have been drawn in the middle of the dead region in
every graphene figure.

Those values were not a typo: `bandstructure.py` reproduces them, but only
because `find_band_edges` is handed `mid_gap_energy = 0.5` and returns the
nearest gap to it. The registry now carries 0.032 / 1.115.

### Stage 3 — CPU benchmark

```bash
cd run_bench
python run_benchmarks.py --material <name>
```

One material per process. This is the stage that stores the factors, the
solutions and the backward errors; everything downstream reads what it writes.

**Do not use `--exclude-solvers superlu`.** SuperLU is the a-priori-bounded
GEPP reference the growth figures are read against and the solver Block Thomas
is claimed to be indistinguishable from. si-bulk was swept without it in August
and needed a second pass to fill it in.

Measured, graphene (n = 2080, 6 blocks 26…416, 21% dense):
**10 s/index, 172 MB/index**, 919 indices → 2.5 h and 185 GB.

**Check:**

```bash
python solvers/verify.py material <name>
```

Every (solver, dtype) should read `N / N`. UMFPACK is complex128 only by
design; the driver prints a skip line for it at complex64.

### Stage 3b — half precision

Separate because `block-thomas-fp16` is not in the default solver set.

```bash
python run_benchmarks.py --material <name> --solvers block-thomas-fp16
```

Measured: **11.4 s/index on graphene, 12.0 s on si-bulk**. Predict the cost for
a new material by timing `BlockThomasFP16` on a synthetic system with the real
block partition — that predicted si-bulk within 1% and graphene within 2%.

fp16 fails outright on some indices with `FloatingPointError: cannot scale a
block with max |entry| = nan`; those simply get no complex32 row. The rate is
not a function of block size: graphene has the largest blocks and lost 3 of
919, si-bulk lost ~27%.

### Stage 4 — conditioning

`forward_error.py` needs this, so it is on the critical path.

```bash
cd condition-est
python condition_est.py /scratch/yimili/matrices2/hdf5/<name>.h5 --overwrite
python condition_est.py /scratch/yimili/matrices2/hdf5/<name>.h5 --only-skeel
python exact_condition.py /scratch/yimili/matrices2/hdf5/<name>.h5 --all
```

`--only-skeel` fills `cond_skeel` and `cond_skeel_x` on rows already valid, with
no singular values, so it is far cheaper than a full rerun. `exact_condition.py`
is the estimator verification: dense inverse plus full SVD, giving the reference
points the condition figure scatters the estimate against.

Measured at n = 768: `condition_est` 0.1 s/index, `exact_condition` 0.2 s/index.
The dense work is O(n³), so scale by `(n/768)³` — about 4 s at n = 2080 and
25 s at n = 3840.

`--resume` refuses when the existing group was written for a **different index
selection**, which is what happens when a spot check of a handful of indices is
followed by a strided sweep. Use `--overwrite`.

**Check:**

```bash
python solvers/verify.py analysis /scratch/yimili/condition-est/<name>.h5
```

`cond_skeel_x` is NaN exactly on the indices with no right-hand side, since it
needs `x`. That fraction should equal `1 - nonzero/total` for the material.

### Stage 5 — to the GPU node

```bash
# from the laptop, agent forwarded
ssh -A mont-fort1 'rsync -a --partial --inplace \
    /scratch/yimili/matrices2/hdf5/<name>.h5 attelas1:/scratch/yimili/matrices2/hdf5/'
ssh -A mont-fort1 'rsync -a --partial \
    /scratch/yimili/condition-est/<name>.h5 attelas1:/scratch/yimili/condition-est/'
```

Then, on `attelas1`:

```bash
cd run_bench && python gpu_run_benchmarks.py --material <name>
```

Measured: 919 indices in 7 minutes. Must run before `forward_error.py`, or
there will be no cuDSS rows.

### Stage 6 — error analysis

```bash
cd block-thomas
python forward_error.py /scratch/yimili/matrices2/hdf5/<name>.h5
python growth_factor.py /scratch/yimili/matrices2/hdf5/<name>.h5
python extract_lu.py    /scratch/yimili/matrices2/hdf5/<name>.h5 --idx A B C D
```

Measured per working index, graphene at n = 2080:

| | cost |
|---|---|
| `forward_error` | 2.5 s |
| `growth_factor`, complex128 + complex64 | 16 s |
| `growth_factor`, with complex32 as well | 24 s |

`growth_factor` is the slow one because it reassembles the global L and U from
the stored factors, and for fp16 it does that in the real embedding at twice
the dimension.

**Both scripts rewrite their group wholesale** — `save_table` replaces, it does
not append. Adding complex32 to a file that already has complex128/64 means
recomputing all of it. Run stage 3b **before** stage 6, not after; doing it the
other way cost about four extra hours on graphene.

`extract_lu.py` stores full L, U and A_eff per combination. Give it four or
five indices, never a sweep.

**Check:**

```bash
python solvers/verify.py analysis /scratch/yimili/error-analysis-block-thomas/<name>.h5
```

Expected non-finite fractions, which are structural rather than faults:

| column | NaN where |
|---|---|
| `inv_resid_max` | every row that is not `block-thomas-inv` |
| `schur_growth`, `schur_norm_max`, `schur_cond_max` | every row that is not a Block Thomas variant |
| everything in `forward_error` | nothing — this group should read `none` |

### Stage 7 — figures

```bash
cd plotting/block-thomas
python plot_growth_factor.py  /scratch/yimili/error-analysis-block-thomas/<name>.h5
python plot_backward_error.py /scratch/yimili/error-analysis-block-thomas/<name>.h5
python plot_forward_error.py  /scratch/yimili/error-analysis-block-thomas/<name>.h5
python plot_lu_factors.py     /scratch/yimili/error-analysis-block-thomas/<name>.h5 --idx A
cd ../condition-est
python plot_condition.py --materials <name>
```

Seven error-analysis figures, one or more LU-structure figures, two condition
figures.

**The solver set is a plot-time choice.** Every figure draws SuperLU and Block
Thomas and nothing else; UMFPACK, MUMPS, cuDSS and `block-thomas-inv` are
excluded by default and `--solvers` brings any of them back with no
recomputation. `block-thomas-fp16` must stay **in** `DEFAULT_SOLVERS` of
`plot_growth_factor.py`: `growth_factor.py` stores complex32 rows under that
name, so dropping it silently empties the complex32 column of every growth
figure rather than raising.

---

## 3. Stopping a running job

**A driver that writes an HDF5 file must be stopped with SIGINT only.** SIGTERM
kills Python without unwinding, leaving the file unflushed; on a file of this
size that means repacking from source. SIGINT raises `KeyboardInterrupt`, which
lets the `with h5py.File(...)` block close cleanly.

`run_benchmarks.py` was observed **not** to honour SIGINT across three attempts
on 2026-08-30. There was no safe way to stop it, so it ran to completion. Plan
around that: get the stride right before launching, not after.

A reader — `forward_error.py`, `growth_factor.py`, the plot scripts, all of
which open the material file `"r"` — stops safely with SIGTERM.

To stop a chain but let the current step finish, kill the wrapper shell and
leave the Python child; it reparents to init and completes, and nothing queued
after it starts.

## 4. Running things concurrently

Two readers on one HDF5 file are fine; only reader-versus-writer conflicts,
raising `BlockingIOError: [Errno 11]`. So the mixed-precision IR scripts can run
against a material file while `growth_factor.py` is reading it — both open `"r"`,
and they write to different files.

The exception is anything that measures time. `mpperf.py` reports wall time and
memory, and is only meaningful on a quiet node. `mpir.py` reports iteration
counts and forward error, which contention cannot change.

Since no figure plots `time_fact` or `time_solve` any more, benchmark timings no
longer constrain what else may run.

## 5. Reading a file another process has open

```
env HDF5_USE_FILE_LOCKING=FALSE python <script> ...
```

Safe only for indices well behind wherever a writer currently is.
