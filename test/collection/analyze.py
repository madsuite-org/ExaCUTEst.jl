#!/usr/bin/env python3
"""Census of a collection verification run.

    python3 -I analyze.py RUN [ULP_PASS]

Reads RUN/results/{fortran,cpu,gpu,gpu_retry,deaths}.tsv and
RUN/population.tsv; writes RUN/results/summary_problem.tsv (one row per
problem x configuration, nothing omitted; CV_SUMMARY overrides the path) and
prints the report. CV_WATCH, a space-separated list of problems, adds a
per-problem detail section.

ulp: |test - ref|_inf / (eps(Tw) * S), where S is the largest finite |ref|
that quantity reaches over BOTH measured points (x0, xp), not at the point.
Ratio (per problem and callback) = ulp(Float32 configuration) / ulp(Float64
configuration): ~1 is conditioning, ~eps64/eps32 = 1.86e-9 is a fixed
absolute error.
"""
import collections, math, os, sys

F = sys.argv[1] if len(sys.argv) > 1 else os.environ['CV_RUN']
SUMMARY = os.environ.get('CV_SUMMARY') or F + '/results/summary_problem.tsv'
EPS = {'F64': 2.220446049250313e-16, 'F32': 1.1920928955078125e-07}
CFG = {'gpu64_vs_cpu64': 'F64', 'gpu32_vs_cpu32': 'F32', 'cpu32_vs_cpu64': 'F32'}
VALUE_CBS = ('obj', 'grad', 'cons', 'jac_coord', 'jac_asm', 'hess_coord', 'hess_asm', 'hprod')
ULP_PASS = float(sys.argv[2]) if len(sys.argv) > 2 else 1e3

# Verdicts, for every (problem, configuration):
#   PASS      every reading computed, agrees within the thresholds, and at least
#             one reading is a non-trivial numeric comparison
#   FAIL      a reading disagrees, a callback raised, a build failed, or the
#             process crashed
#   TIMEOUT   the per-problem wall-clock alarm killed the process
#   RESOURCE  GPU out of memory, or the process was SIGKILLed (host OOM killer)
#   SKIPPED   computed, but nothing non-trivial to compare (reference unavailable,
#             or every reading empty or identically zero on both sides)
#   NOT_RUN   no worker reached the problem: the run is incomplete there
#   PASS_CONDITIONED  as PASS, except that some reading exceeds ULP_PASS while its
#             precision ratio lies in [RATIO_LO, RATIO_HI]: the error scales
#             with each precision's own rounding, i.e. it is conditioning
#   FIXED_ERROR  some reading's precision ratio is < FIXED_RATIO with a
#             Float64 error >= FIXED_MIN_ULP64 ulp: an error that does not
#             scale with the precision, i.e. a wrong value (a failure)
#   ZERO_REF  every other reading passes, but a reading's reference is
#             identically zero at a point, so no relative error exists there;
#             only problems listed in ZERO_REF_EXCEPTIONS (unlisted: FAIL)
CATS = ['PASS', 'PASS_CONDITIONED', 'FAIL', 'FIXED_ERROR', 'ZERO_REF', 'TIMEOUT',
        'RESOURCE', 'SKIPPED', 'NOT_RUN']

# Precision ratio of a (problem, callback) reading: ulp(Float32 comparison) /
# ulp(Float64 comparison), each in its own eps units. Error that tracks each
# precision's rounding (conditioning) gives O(1); a fixed absolute error, the
# same in both precisions, gives eps64/eps32 = 1.86e-9.
#   gpu64_vs_cpu64, gpu32_vs_cpu32: ulp(gpu32_vs_cpu32) / ulp(gpu64_vs_cpu64)
#   cpu32_vs_cpu64:                 ulp(cpu32_vs_cpu64) / ulp(gpu64_vs_cpu64)
RATIO_PAIR = {'gpu64_vs_cpu64': ('gpu32_vs_cpu32', 'gpu64_vs_cpu64'),
              'gpu32_vs_cpu32': ('gpu32_vs_cpu32', 'gpu64_vs_cpu64'),
              'cpu32_vs_cpu64': ('cpu32_vs_cpu64', 'gpu64_vs_cpu64')}
# The conditioned window applies to the GPU configurations only. The
# cpu32_vs_cpu64 ratio is reported but does not decide: that comparison also
# rounds x to Float32, an input perturbation gpu64_vs_cpu64 never sees, and its
# ratios over 1000-ulp readings run continuously past any upper bound.
WINDOW_CFGS = ('gpu64_vs_cpu64', 'gpu32_vs_cpu32')
# Readings with a Float64 error >= FIXED_MIN_ULP64 have ratios either at
# 1.86e-9 (fixed error) or >= ~1e-4 (DALLASM hprod, the lowest conditioned
# reading); RATIO_LO sits a decade below the latter.
RATIO_LO, RATIO_HI = 1e-5, 1e4
# Fixed-error test, applied to every reading of the GPU configurations
# whatever its ulp. The Float64 floor keeps last-bit roundoff (a Float64
# difference of ~1 ulp against a near-exact Float32 comparison) from reading
# as a fixed error.
FIXED_RATIO, FIXED_MIN_ULP64 = 1e-6, 10.0
FIXED_CFGS = ('gpu64_vs_cpu64', 'gpu32_vs_cpu32')

# Readings whose reference is identically zero at a point (status ZERO_REF).
# Each entry: problem -> (callbacks, reason). The cross-point ulp test still
# applies to these readings; only the per-point ZERO_REF status is excepted.
ZERO_REF_EXCEPTIONS = {
    'TAX13322':  ({'cons'}, 'x0 satisfies every constraint exactly in the Float64 reference; the test side leaves roundoff'),
    'TAXR13322': ({'cons'}, 'x0 satisfies every constraint exactly in the Float64 reference; the test side leaves roundoff'),
    'TAX53322':  ({'cons'}, 'x0 satisfies every constraint exactly in the Float64 reference; the test side leaves roundoff'),
    'TAXR53322': ({'cons'}, 'x0 satisfies every constraint exactly in the Float64 reference; the test side leaves roundoff'),
    'GOULDQP2':  ({'cons'}, 'cons(x0) is exactly zero in Float64; the Float32 model rounds x0 and leaves a residual'),
    'GOULDQP3':  ({'cons'}, 'cons(x0) is exactly zero in Float64; the Float32 model rounds x0 and leaves a residual'),
}
DEATH_CAT = lambda st: 'TIMEOUT' if st == 'TIMEOUT' else ('RESOURCE' if st == 'KILLED_137' else 'FAIL')
import glob
INFLIGHT = {open(f).read().split('\t')[0] for f in glob.glob(F + '/state/inflight.*')} - {''}
CLAIMED_CPU = ({l.split('\t')[0] for l in open(F + '/state/claimed_cpu.txt')}
               if os.path.exists(F + '/state/claimed_cpu.txt') else set())

pop = [l.split('\t') for l in open(F + '/population.tsv').read().splitlines()[1:]]
tier = dict(pop)
P = [p for p, _ in pop]

def rows(path, ncol):
    out = []
    try:
        for l in open(path):
            f = l.rstrip('\n').split('\t')
            if len(f) >= ncol:
                out.append(f)
    except FileNotFoundError:
        pass
    return out

A = rows(F + '/results/fortran.tsv', 14)
# Problems retried after a GPU OOM (finish.sh) are read from gpu_retry.tsv
# only: their first-attempt GPU rows and deaths are superseded.
GR = rows(F + '/results/gpu_retry.tsv', 20)
RETRIED = {f[0] for f in GR}
C = rows(F + '/results/cpu.tsv', 20) + \
    [f for f in rows(F + '/results/gpu.tsv', 20) if f[0] not in RETRIED] + GR
D = [d for d in rows(F + '/results/deaths.tsv', 7)
     if not (d[0] in RETRIED and d[1] == 'gpu' and d[6] != 'gpur')]
fl = lambda s: float(s) if s not in ('', 'NaN', 'nan') else float('nan')

deaths = collections.defaultdict(list)          # (prob, mode) -> [(status, phase)]
for p, mode, st, phase, secs, host, wid in (d[:7] for d in D):
    deaths[(p, mode)].append((st, phase, secs, host))

# which configuration a death phase belongs to (phases come from worker.jl's inflight())
PHASE_OF = {'gpu64_vs_cpu64': ('f64',), 'gpu32_vs_cpu32': ('f32',),
            'cpu32_vs_cpu64': ('cpu_f64', 'cpu_f32'), 'cpu64_vs_fortran': ('fortran',)}
def death_for(p, mode, cmp):
    L_ = [d for d in deaths.get((p, mode), []) if any(k in d[1] for k in PHASE_OF[cmp])]
    return L_[-1] if L_ else None

# ---------------- A: Fortran ----------------
Aprob = {}
Abad = collections.defaultdict(list)
Aadj = collections.defaultdict(list)
Aulp = []                                        # decoded readings, eps64 units, at-point scale
for f in A:
    p, t, cmp, cb, pt, st, val, tol, ad, sc, mode, host, wid, note = f[:14]
    if cb == 'PROBLEM':
        Aprob[p] = st
    elif st == 'BAD':
        Abad[p].append(f"{cb}@{pt}={val} tol={tol} {note[:120]}")
    elif st == 'REF_ADJUDICATED':
        Aadj[p].append(cb)
    if t == 'decoded' and cb in ('obj', 'grad', 'cons', 'jac', 'hess', 'hprod'):
        a, s = fl(ad), fl(sc)
        if s > 0 and not math.isnan(a) and st != 'REF_ADJUDICATED':
            Aulp.append(a / (EPS['F64'] * s))

# ---------------- B, C: configuration rows ----------------
builds = {}                                      # (prob, label) -> (status, note)
reads = collections.defaultdict(dict)            # (prob, cmp, cb) -> {pt: (status, scale, absdiff, note)}
evalerr = collections.defaultdict(list)          # (prob, cmp) -> notes
selfchk = collections.defaultdict(dict)
done_marker = collections.defaultdict(set)       # prob -> groups that wrote worker/PROBLEM DONE
for f in C:
    p, t, grp, cmp, cb, pt, st, n, sc, ad, ulp, nfr, nft, same, nzr, nzt, secs, host, wid, note = f[:20]
    if cmp == 'build':
        builds[(p, grp, cb)] = (st, note)
    elif cmp == 'worker' and cb == 'PROBLEM':
        done_marker[p].add(grp)
    elif cmp == 'worker':
        evalerr[(p, 'worker-' + grp)].append(note)
    elif cmp == 'ref_selfcheck':
        selfchk[p][pt] = (st, fl(sc), fl(ad))
    elif cmp == 'ref':
        evalerr[(p, 'ref')].append(f"{cb}@{pt}: {note[:150]}")
    elif cmp in CFG:
        if cb == 'eval':
            evalerr[(p, cmp)].append(f"{st}@{pt}: {note[:200]}")
        elif cb != 'STAGE':
            reads[(p, cmp, cb)][pt] = (st, fl(sc), fl(ad), note, nfr, nft, same)

byprob = collections.defaultdict(list)
for (q, c, cb) in reads:
    byprob[(q, c)].append(cb)

def xulp(p, cmp, cb):
    """cross-point ulp, or None when the reading has no numeric value"""
    r = reads.get((p, cmp, cb), {})
    S = max([v[1] for v in r.values() if not math.isnan(v[1])] or [float('nan')])
    A_ = max([v[2] for v in r.values() if not math.isnan(v[2])] or [float('nan')])
    if math.isnan(S) or math.isnan(A_):
        return None
    if S == 0:
        return 0.0 if A_ == 0 else float('inf')
    return A_ / (EPS[CFG[cmp]] * S)

def ratio(p, cmp, cb):
    """(precision ratio, None) for reading (p, cb) of configuration cmp, or
    (None, why) when it cannot be formed: either side missing, zero or non-finite."""
    num_c, den_c = RATIO_PAIR[cmp]
    num, den = xulp(p, num_c, cb), xulp(p, den_c, cb)
    for c, u in ((num_c, num), (den_c, den)):
        if u is None:
            return None, f"no {c} reading"
        if u == 0:
            return None, f"{c} difference exactly zero"
        if not math.isfinite(u):
            return None, f"{c} ulp non-finite"
    return num / den, None

def fixed_error(p, cmp, cb):
    """the precision ratio when reading (p, cb) carries the fixed-error signature, else None"""
    if cmp not in FIXED_CFGS:
        return None
    r, _ = ratio(p, cmp, cb)
    u64 = xulp(p, 'gpu64_vs_cpu64', cb)
    if r is not None and r < FIXED_RATIO and u64 >= FIXED_MIN_ULP64:
        return r
    return None

BUILD_LABEL = {'gpu64_vs_cpu64': ['cpu_f64', 'gpu_f64'], 'gpu32_vs_cpu32': ['cpu_f32', 'gpu_f32'],
               'cpu32_vs_cpu64': ['cpu_f64', 'cpu_f32']}
GROUP = {'gpu64_vs_cpu64': 'gpu', 'gpu32_vs_cpu32': 'gpu', 'cpu32_vs_cpu64': 'cpu'}

def label(cb, pt, st, note):
    """A reading's non-pass status, named for what it is."""
    if st.startswith('NONBITS_'):
        return f"{cb}@{pt} GPU_KERNEL_NONBITS[{st[8:]}]: device kernel not compilable (non-isbits argument)"
    if 'DivideError' in note:
        return f"{cb}@{pt} EMPTY_RANGE_LAUNCH: DivideError from a zero-length kernel launch"
    if st == 'NONFINITE_PART' or st == 'NONFINITE':
        return f"{cb}@{pt} {st} at DIFFERENT indices (ref {note})"
    return f"{cb}@{pt} {st} {note[:150]}"

def verdict_cfg(p, cmp):
    """(verdict, reasons, special) for one problem and configuration"""
    grp = GROUP[cmp]
    reasons, special = [], set()
    blab = BUILD_LABEL[cmp]
    cbs = byprob.get((p, cmp), [])
    death = death_for(p, grp, cmp)
    anydeath = deaths.get((p, grp))
    bf = [(l, builds.get((p, grp, l))) for l in blab]
    bad = [(l, b) for l, b in bf if b and b[0] != 'BUILD_OK']
    if not cbs:
        if bad:
            l, b = bad[0]
            return ('RESOURCE' if b[0] == 'OOM_GPU' else 'FAIL', [f"{b[0]} {l}: {b[1][:200]}"], special)
        if anydeath:
            st, phase, secs, host = anydeath[-1]
            return (DEATH_CAT(st), [f"{st}: died in phase '{phase}' after {secs}s on {host}"], special)
        errs = evalerr.get((p, cmp), []) + evalerr.get((p, 'ref'), [])
        if errs:
            oom = all('OOM_GPU' in e for e in evalerr.get((p, cmp), [])) and evalerr.get((p, cmp))
            return ('RESOURCE' if oom else 'FAIL', errs[:2], special)
        if p in INFLIGHT:
            return ('NOT_RUN', ['in flight when this analysis ran'], special)
        if grp not in done_marker[p] and not any(b for l, b in bf):
            return ('NOT_RUN', ['no worker reached this problem'], special)
        if grp not in done_marker[p]:
            return ('FAIL', ['built but no readings, no DONE marker and no death record'], special)
        return ('SKIPPED', ['no comparable readings (reference unavailable)'], special)
    fail = False
    resource = False
    nontrivial = False
    fixed = False
    conditioned = False
    zeroref = False
    for cb in sorted(set(cbs)):
        r = reads[(p, cmp, cb)]
        for pt, (st, sc, ad, note, nfr, nft, same) in sorted(r.items()):
            if st == 'OK' and cb in VALUE_CBS and sc > 0:
                nontrivial = True
            if st in ('OK', 'DEGENERATE', 'SKIPPED', 'EMPTY'):
                continue                          # EMPTY: zero-length on both sides
            if st == 'NNZ_DIFFER':
                special.add('NNZ_DIFFER:' + cb)
                continue                          # adjudicated by the assembled row
            if st in ('NONFINITE_PART', 'NONFINITE') and same == 'yes' and nfr == nft:
                special.add('NONFINITE_SAME')
                if st == 'NONFINITE_PART' and sc > 0:
                    nontrivial = True
                continue
            if st == 'ZERO_REF' and cb in ZERO_REF_EXCEPTIONS.get(p, ((), ''))[0]:
                zeroref = True
                reasons.append(f"{cb}@{pt} ZERO_REF (exception: {ZERO_REF_EXCEPTIONS[p][1]})")
                continue
            if st == 'OOM_GPU':
                resource = True
                reasons.append(f"{cb}@{pt} OOM_GPU")
                continue
            if st.startswith('NONBITS_') or st in ('HPROD_NO_PRODHELPER', 'HPROD_KSEG_PRODC'):
                special.add(st)
            reasons.append(label(cb, pt, st, note if st not in ('NONFINITE_PART', 'NONFINITE')
                                 else f"{nfr} vs test {nft}"))
            fail = True
        if cb in VALUE_CBS:
            if cb == 'hess_coord' and any(v[0] == 'NNZ_DIFFER' for k, v in reads.get((p, cmp, 'hess_structure'), {}).items()):
                continue
            if cb == 'jac_coord' and any(v[0] == 'NNZ_DIFFER' for k, v in reads.get((p, cmp, 'jac_structure'), {}).items()):
                continue
            u = xulp(p, cmp, cb)
            fr = fixed_error(p, cmp, cb)
            if fr is not None:
                reasons.append(f"{cb} FIXED_ERROR: ratio {fr:.3g} < {FIXED_RATIO:g}, "
                               f"Float64 error {xulp(p, 'gpu64_vs_cpu64', cb):.3g} ulp")
                fixed = True
            if u is not None and u > ULP_PASS and cmp not in WINDOW_CFGS:
                r, why = ratio(p, cmp, cb)
                reasons.append(f"{cb} {u:.3g} ulp > {ULP_PASS:g} (" +
                               (f"ratio {r:.3g}" if r is not None else f"no ratio: {why}") +
                               ", informational)")
                fail = True
            elif u is not None and u > ULP_PASS:
                r, why = ratio(p, cmp, cb)
                if r is not None and RATIO_LO <= r <= RATIO_HI:
                    reasons.append(f"{cb} {u:.3g} ulp > {ULP_PASS:g}, conditioned (ratio {r:.3g})")
                    conditioned = True
                else:
                    reasons.append(f"{cb} {u:.3g} ulp > {ULP_PASS:g}, " +
                                   (f"ratio {r:.3g} outside [{RATIO_LO:g}, {RATIO_HI:g}]"
                                    if r is not None else f"no ratio ({why})"))
                    fail = True
    errs = evalerr.get((p, cmp), [])
    for e in errs[:2]:
        if 'OOM_GPU' in e:
            resource = True
        else:
            fail = True
        reasons.append(e)
    if bad:
        for l, b in bad:
            reasons.append(f"{b[0]} {l}")
            resource |= b[0] == 'OOM_GPU'
            fail |= b[0] != 'OOM_GPU'
    if fixed:
        return ('FIXED_ERROR', reasons, special)
    if death:
        st, phase, secs, host = death
        reasons.append(f"process died ({st}) in phase '{phase}'")
        c = DEATH_CAT(st)
        if c == 'FAIL':
            fail = True
        elif not fail:
            return (c, reasons, special)
    if fail:
        return ('FAIL', reasons, special)
    if resource:
        return ('RESOURCE', reasons, special)
    if zeroref:
        return ('ZERO_REF', reasons, special)
    if not nontrivial:
        return ('SKIPPED', reasons + ['no non-trivial reading: every value reading empty or identically zero on both sides'], special)
    return ('PASS_CONDITIONED' if conditioned else 'PASS', reasons, special)

out = open(SUMMARY, 'w')
out.write('problem\ttier\tconfiguration\tverdict\treasons\n')
tot = collections.defaultdict(collections.Counter)
fails = collections.defaultdict(list)
specials = collections.defaultdict(lambda: collections.defaultdict(set))

for p in P:
    # A
    if p in Aprob:
        v = 'PASS' if Aprob[p] == 'PASS' else 'FAIL'
        why = ([f"BUILD_FAILED"] if Aprob[p] == 'BUILD_FAILED' else []) + Abad.get(p, [])
        if v == 'PASS' and Aadj.get(p):
            why = [f"pass with reference adjudication on {','.join(sorted(set(Aadj[p])))}"]
    elif deaths.get((p, 'cpu')):
        st, phase, secs, host = deaths[(p, 'cpu')][-1]
        v, why = DEATH_CAT(st), [f"{st}: died in phase '{phase}' after {secs}s on {host}"]
    elif p in INFLIGHT:
        v, why = 'NOT_RUN', ['in flight when this analysis ran']
    elif p in CLAIMED_CPU:
        v, why = 'FAIL', ['claimed by a CPU worker but no PROBLEM row and no death record']
    else:
        v, why = 'NOT_RUN', ['never claimed by a CPU worker']
    tot['cpu64_vs_fortran'][v] += 1
    if v != 'PASS':
        fails['cpu64_vs_fortran'].append((p, v, why))
    out.write(f"{p}\t{tier[p]}\tcpu64_vs_fortran\t{v}\t{' ; '.join(why)}\n")
    for cmp in CFG:
        v, why, sp = verdict_cfg(p, cmp)
        tot[cmp][v] += 1
        for s in sp:
            specials[cmp][s].add(p)
        if v != 'PASS':
            fails[cmp].append((p, v, why))
        out.write(f"{p}\t{tier[p]}\t{cmp}\t{v}\t{' ; '.join(why)}\n")
out.close()

def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float('nan')

def decades(xs):
    c = collections.Counter()
    for x in xs:
        c['0' if x == 0 else ('<1' if x < 1 else f"1e{int(math.floor(math.log10(x)))}")] += 1
    order = ['0', '<1'] + [f"1e{k}" for k in range(0, 30)]
    return ' '.join(f"{k}:{c[k]}" for k in order if c[k])

print(f"POPULATION {len(P)} ({sum(1 for p in P if tier[p]=='native')} native, "
      f"{sum(1 for p in P if tier[p]=='decoded')} decoded)")
print(f"""
THRESHOLDS
  A cpu64_vs_fortran: ExaModelsSIF pilot.jl -- relative error (|a-b|/(1+max|.|)) <= 1e-6,
    REF_NOISE problems <= REF_NOISE_TOL (default 1e-4); natives: native_verify.jl TOL table.
    Readings of problems in HESS_REF_BUGS / DERIV_REF_BUGS are REF_ADJUDICATED, not failures.
  B, C (gpu64_vs_cpu64, gpu32_vs_cpu32, cpu32_vs_cpu64): every value callback
    {', '.join(VALUE_CBS)} must satisfy
      max|test-ref| / (eps(T) * S) <= {ULP_PASS:g} ulp,
    S = largest finite |ref| of that quantity over both points (x0, xp), T the lower
    precision of the pair; structures must match exactly (a differing Hessian/Jacobian
    pattern is accepted only through the assembled-matrix row); non-finite entries must
    sit at the same indices on both sides; zero-length readings on both sides pass.
    gpu64_vs_cpu64 and gpu32_vs_cpu32: a value reading above {ULP_PASS:g} ulp passes as
    CONDITIONED when its precision ratio r = ulp(gpu32_vs_cpu32) / ulp(gpu64_vs_cpu64)
    lies in [{RATIO_LO:g}, {RATIO_HI:g}], and fails otherwise, including when r cannot
    be formed (either side missing, zero or non-finite).
    cpu32_vs_cpu64: plain threshold; r = ulp(cpu32_vs_cpu64) / ulp(gpu64_vs_cpu64) is
    shown for each failure but does not decide.
    Whatever its ulp, a GPU reading with r < {FIXED_RATIO:g} and a Float64 error
    >= {FIXED_MIN_ULP64:g} ulp is FIXED_ERROR (eps64/eps32 = 1.86e-9 is the signature of
    an error that does not scale with the precision).
    ZERO_REF exceptions (reference identically zero at a point):""")
for p_, (cbs_, why_) in sorted(ZERO_REF_EXCEPTIONS.items()):
    print(f"      {p_:10s} {','.join(sorted(cbs_))}: {why_}")
for cfg in ['cpu64_vs_fortran'] + list(CFG):
    print(f"\n=== {cfg}: " + '  '.join(f"{k} {tot[cfg][k]}" for k in CATS) +
          f"  (sum {sum(tot[cfg].values())})" +
          ('' if set(tot[cfg]) <= set(CATS) else f"  OTHER {dict((k, v) for k, v in tot[cfg].items() if k not in CATS)}"))
    for p, v, why in fails[cfg]:
        print(f"   {p:12s} {tier[p]:7s} {v:14s} {' ; '.join(why)[:300]}")

print("\n=== ulp distributions (cross-point scale) ===")
print(f"cpu64_vs_fortran decoded readings (at-point scale, eps64): n={len(Aulp)} "
      f"median {pct(Aulp,.5):.3g} p99 {pct(Aulp,.99):.3g} max {max(Aulp or [float('nan')]):.3g}")
print(f"   decades: {decades(Aulp)}")
for cmp in CFG:
    allu = []
    bycb = collections.defaultdict(list)
    for (p, c, cb) in reads:
        if c != cmp or cb not in VALUE_CBS:
            continue
        u = xulp(p, c, cb)
        if u is None or math.isinf(u):
            continue
        allu.append(u); bycb[cb].append(u)
    print(f"{cmp}: n={len(allu)} median {pct(allu,.5):.3g} p90 {pct(allu,.9):.3g} "
          f"p99 {pct(allu,.99):.3g} max {max(allu or [float('nan')]):.3g}")
    print(f"   decades: {decades(allu)}")
    for cb in VALUE_CBS:
        if bycb[cb]:
            print(f"   {cb:11s} n={len(bycb[cb]):5d} median {pct(bycb[cb],.5):.3g} "
                  f"p99 {pct(bycb[cb],.99):.3g} max {max(bycb[cb]):.3g}")
    worst = sorted(((xulp(p, c, cb), p, cb) for (p, c, cb) in reads
                    if c == cmp and cb in VALUE_CBS and xulp(p, c, cb) is not None
                    and xulp(p, c, cb) > ULP_PASS), reverse=True)
    print(f"   readings > {ULP_PASS:g} ulp: {len(worst)}: " +
          ', '.join(f"{p}/{cb}={u:.3g}" for u, p, cb in worst[:40]))

for cmp in CFG:
    num_c, den_c = RATIO_PAIR[cmp]
    print(f"\n=== precision ratio for {cmp}: ulp({num_c}) / ulp({den_c})" +
          ("" if cmp in WINDOW_CFGS else " (informational: not part of the verdict)") + " ===")
    rat, big, noratio, fixed_l = [], [], [], []
    for (p, c, cb) in list(reads):
        if c != cmp or cb not in VALUE_CBS:
            continue
        u = xulp(p, cmp, cb)
        r, why = ratio(p, cmp, cb)
        if r is not None:
            rat.append(r)
        fr = fixed_error(p, cmp, cb)
        if fr is not None:
            fixed_l.append((p, cb, xulp(p, 'gpu64_vs_cpu64', cb), fr))
        if u is not None and u > ULP_PASS:
            if r is None:
                noratio.append((p, cb, u, why))
            else:
                big.append((p, cb, u, r))
    print(f"all readings with a ratio: n={len(rat)} median {pct(rat,.5):.3g} p1 {pct(rat,.01):.3g} "
          f"p99 {pct(rat,.99):.3g}; ratio < {FIXED_RATIO:g}: {sum(1 for r in rat if r < FIXED_RATIO)}")
    print(f"   decades: {decades(rat)}")
    br = [r for _, _, _, r in big]
    print(f"readings > {ULP_PASS:g} ulp with a ratio: n={len(big)} min {min(br or [float('nan')]):.3g} "
          f"max {max(br or [float('nan')]):.3g}; inside [{RATIO_LO:g}, {RATIO_HI:g}]: "
          f"{sum(1 for r in br if RATIO_LO <= r <= RATIO_HI)}")
    print(f"   decades: {decades(br)}")
    for p, cb, u, r in sorted(big, key=lambda t: t[3]):
        inside = RATIO_LO <= r <= RATIO_HI
        print(f"   {p:12s} {cb:11s} ulp {u:9.3g}  ratio {r:9.3g}  {'inside' if inside else 'OUTSIDE'}")
    print(f"readings > {ULP_PASS:g} ulp without a ratio: {len(noratio)}")
    for p, cb, u, why in sorted(noratio):
        print(f"   {p:12s} {cb:11s} ulp {u:9.3g}  {why}")
    if cmp in FIXED_CFGS:
        print(f"fixed-error signature (ratio < {FIXED_RATIO:g}, Float64 >= {FIXED_MIN_ULP64:g} ulp): {len(fixed_l)}")
        for p, cb, u64, r in sorted(fixed_l):
            print(f"   {p:12s} {cb:11s} ulp64 {u64:9.3g}  ratio {r:9.3g}")
        low = sorted((ratio(p, cmp, cb)[0], p, cb, xulp(p, 'gpu64_vs_cpu64', cb), xulp(p, cmp, cb))
                     for (p, c, cb) in reads if c == cmp and cb in VALUE_CBS
                     and ratio(p, cmp, cb)[0] is not None and ratio(p, cmp, cb)[0] < 1e-2)
        print("every reading with ratio < 0.01 (ratio, ulp64, ulp of this configuration):")
        for r, p, cb, u64, u in low:
            print(f"   {p:12s} {cb:11s} ratio {r:9.3g}  ulp64 {u64:9.3g}  ulp {u:9.3g}")

print("\n=== GPU hprod over the collection ===")
for cmp in ('gpu64_vs_cpu64', 'gpu32_vs_cpu32'):
    st = collections.Counter()
    probs_ok = set(); probs_bad = collections.defaultdict(set)
    for (p, c, cb), r in reads.items():
        if c != cmp or cb != 'hprod':
            continue
        sts = {v[0] for v in r.values()}
        u = xulp(p, c, cb)
        if sts <= {'OK', 'DEGENERATE'} and (u is None or u <= ULP_PASS):
            probs_ok.add(p)
        else:
            key = ','.join(sorted(sts)) + ('' if u is None or u <= ULP_PASS else ' >ulp')
            probs_bad[key].add(p)
    print(f"{cmp}: hprod readable+agreeing on {len(probs_ok)} problems; others: " +
          '; '.join(f"{k}: {len(v)}" for k, v in probs_bad.items()))
    for k, v in probs_bad.items():
        print(f"     {k}: {' '.join(sorted(v))[:1500]}")
    for s, ps in specials[cmp].items():
        print(f"   special {s}: {len(ps)} problems")

print("\n=== zero hprod adjudication: host selfcheck (hess_coord*v vs hprod) ===")
sc_bad = [(p, d) for p, d in selfchk.items() if any(v[0] not in ('OK', 'DEGENERATE') or
          (v[1] > 0 and v[2] / (EPS['F64'] * v[1]) > ULP_PASS) for v in d.values())]
print(f"selfcheck rows for {len(selfchk)} problems; failing {len(sc_bad)}: " +
      ', '.join(f"{p}:{ {k: (v[0], v[2]/(EPS['F64']*v[1]) if v[1] > 0 else v[2]) for k, v in d.items()} }" for p, d in sc_bad[:30]))
deg = [p for p, d in selfchk.items() if all(v[0] == 'DEGENERATE' for v in d.values())]
print(f"problems whose host hprod and assembled Hessian are both identically zero at both points: {len(deg)}: {' '.join(sorted(deg))}")

WATCH = [p for p in os.environ.get('CV_WATCH', '').split() if p in tier]
if WATCH:
    print("\n=== watched problems (CV_WATCH) ===")
for p in WATCH:
    line = [f"A={next((v for q, v, w in fails['cpu64_vs_fortran'] if q == p), 'PASS')}"]
    for cmp in CFG:
        v, why, sp = verdict_cfg(p, cmp)
        bl = ','.join(f"{l}:{builds.get((p, GROUP[cmp], l), ('-',))[0]}" for l in BUILD_LABEL[cmp])
        hu = {cb: xulp(p, cmp, cb) for cb in ('hess_asm', 'hprod', 'grad')}
        line.append(f"{cmp}={v} [{bl}] " + ' '.join(f"{k}:{'-' if u is None else f'{u:.3g}'}" for k, u in hu.items())
                    + (f" ({' ; '.join(why)[:200]})" if why else ''))
    print(f"{p:11s} " + '\n            '.join(line))

print("\n=== A: reference adjudication ===")
print(f"problems with REF_ADJUDICATED readings: {len(Aadj)}: " + ' '.join(sorted(Aadj)))

VER = {}
for p in P:
    for cmp in CFG:
        VER[(p, cmp)] = verdict_cfg(p, cmp)

print("\n=== non-finite readings (same indices on both sides = accepted) ===")
for cmp in CFG:
    same, diff = set(), set()
    for (p, c, cb), r in reads.items():
        if c != cmp:
            continue
        for pt, v in r.items():
            if v[0] in ('NONFINITE_PART', 'NONFINITE'):
                (same if v[6] == 'yes' else diff).add(p)
    print(f"{cmp}: same indices {len(same)}: {' '.join(sorted(same))}")
    print(f"{cmp}: DIFFERENT indices {len(diff)}: {' '.join(sorted(diff))}")

print("\n=== differing structures (adjudicated through the assembled matrix) ===")
for cmp in CFG:
    d = collections.defaultdict(set)
    for p in P:
        for sp in VER[(p, cmp)][2]:
            if sp.startswith('NNZ_DIFFER'):
                d[sp].add(p)
    for k, v in d.items():
        print(f"{cmp} {k}: {len(v)}: {' '.join(sorted(v))} -> verdicts " +
              str(dict(collections.Counter(VER[(p, cmp)][0] for p in v))))

print(f"\n=== GPU OOM retries (read from gpu_retry.tsv): {len(RETRIED)}: {' '.join(sorted(RETRIED))}")
print("\n=== deaths ===")
for (p, mode), L in sorted(deaths.items()):
    print(f"   {p:12s} {mode} {L}")
