#!/usr/bin/env python3
"""Plan a run: GPU shards by kernel-signature affinity, and the CPU queue.

    python3 -I plan.py RUN NSHARDS [--prev PREVRUN] [--big-shards 3,2] [--bias 0]
                       [--big-nnz 2000000] [--top 10] [--slow-secs 1200]

Reads RUN/population.tsv and RUN/sigs.tsv (sigscan.sh); writes
RUN/shards/shard_gpu<i>.txt and RUN/state/queue_cpu.txt.

GPU cost is dominated by compiling each distinct block signature once per
process, so a shard's cost is roughly the number of DISTINCT signatures in it.
Greedy: problems in decreasing signature count; each goes to the shard where it
adds the fewest new signatures, ties and balance handled by adding the shard's
current cost. Native problems carry no decoded signatures and are spread
round-robin over the shards outside --big-shards. Within a shard, problems run
smallest first, so a crash or timeout late in a shard (the large problems)
costs the least compiled state.

Memory routing, from a previous run's results (--prev): BIG problems (total
nvar+ncon+nnzj+nnzh above --big-nnz in its cpu_f64 build rows, any OOM_GPU,
and any problem whose GPU pass took longer than --slow-secs) may only go to
the shards listed in --big-shards, which should be the devices with the most
free memory; the --top largest of them and the OOM problems go to the first
of those. Without --prev, nothing is routed.

The CPU queue is ordered longest first by the previous run's CPU time, then
largest first by nvar+ncon; one queue serves every CPU worker on every host.
"""
import argparse, collections, os

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('nshards', type=int)
ap.add_argument('--prev')
ap.add_argument('--big-shards', default='')
ap.add_argument('--bias', type=float, default=0.0,
                help='extra cost of a non-BIG problem on a BIG shard')
ap.add_argument('--big-nnz', type=int, default=2_000_000)
ap.add_argument('--top', type=int, default=10)
ap.add_argument('--slow-secs', type=float, default=1200.0)
a = ap.parse_args()

tier = dict(l.split('\t') for l in open(f'{a.run}/population.tsv').read().splitlines()[1:])
sigs, dims = collections.defaultdict(set), {}
for l in open(f'{a.run}/sigs.tsv'):
    f = l.rstrip('\n').split('\t')
    if f[0] not in tier:
        continue
    dims[f[0]] = int(f[1]) + int(f[2]) if f[1].isdigit() and f[2].isdigit() else 0
    if f[3] in ('obj', 'con'):
        sigs[f[0]].add(f[4])

def rows(path):
    try:
        return [l.rstrip('\n').split('\t') for l in open(path)]
    except FileNotFoundError:
        return []

nnz, oom, slow, cpu_secs = {}, set(), set(), {}
if a.prev:
    for f in rows(f'{a.prev}/results/cpu.tsv'):
        if len(f) < 20:
            continue
        if f[3] == 'build' and f[4] == 'cpu_f64' and f[6] == 'BUILD_OK':
            ncon = int(f[19].split('=')[1]) if f[19].startswith('ncon=') else 0
            nnz[f[0]] = int(f[7]) + ncon + max(int(f[14]), 0) + max(int(f[15]), 0)
        elif f[3] == 'worker' and f[4] == 'PROBLEM':
            cpu_secs[f[0]] = float(f[16])
    for f in rows(f'{a.prev}/results/gpu.tsv'):
        if len(f) < 20:
            continue
        if f[6] == 'OOM_GPU':
            oom.add(f[0])
        elif f[3] == 'worker' and f[4] == 'PROBLEM' and float(f[16]) > a.slow_secs:
            slow.add(f[0])

BIG_SHARDS = [int(s) for s in a.big_shards.split(',') if s]
if not BIG_SHARDS:
    oom, slow = set(), set()
ranked = sorted((p for p in tier if nnz.get(p, 0) > a.big_nnz), key=lambda p: -nnz[p])
BIG = (set(ranked) | (oom | slow)) & set(tier) if BIG_SHARDS else set()
TOP = (set(ranked[:a.top]) | oom) & set(tier) if BIG_SHARDS else set()

shards = [dict(sigs=set(), probs=[]) for _ in range(a.nshards)]
def place(p, allowed, bias=0.0):
    def cost(i):
        s = shards[i]
        return (len(s['sigs']) + len(sigs[p] - s['sigs']) + 0.02 * len(s['probs'])
                + (bias if i in BIG_SHARDS else 0.0))
    s = shards[min(allowed, key=cost)]
    s['sigs'] |= sigs[p]
    s['probs'].append(p)
for p in sorted(TOP):
    place(p, BIG_SHARDS[:1])
for p in sorted(BIG - TOP, key=lambda p: (-len(sigs[p]), p)):
    place(p, BIG_SHARDS)
for p in sorted((p for p in tier if tier[p] == 'decoded' and p not in BIG),
                key=lambda p: (-len(sigs[p]), p)):
    place(p, range(a.nshards), a.bias)
small = [i for i in range(a.nshards) if i not in BIG_SHARDS] or list(range(a.nshards))
for i, p in enumerate(sorted(p for p in tier if tier[p] == 'native' and p not in BIG)):
    shards[small[i % len(small)]]['probs'].append(p)
if BIG:
    print(f'BIG {len(BIG)} (TOP {len(TOP)} on shard {BIG_SHARDS[0]}): ' + ' '.join(sorted(BIG)))

os.makedirs(f'{a.run}/shards', exist_ok=True)
allsigs = set().union(*(s['sigs'] for s in shards))
for i, s in enumerate(shards):
    s['probs'].sort(key=lambda p: (nnz.get(p, dims.get(p, 0)), p))
    with open(f'{a.run}/shards/shard_gpu{i}.txt', 'w') as io:
        io.write('\n'.join(s['probs']) + '\n')
    print(f'shard {i}: {len(s["probs"])} problems, {len(s["sigs"])} distinct sigs')
print(f'sum of per-shard distinct sigs {sum(len(s["sigs"]) for s in shards)} '
      f'vs {len(allsigs)} total; problems assigned {sum(len(s["probs"]) for s in shards)} '
      f'of {len(tier)}')

os.makedirs(f'{a.run}/state', exist_ok=True)
queue = sorted(tier, key=lambda p: (-cpu_secs.get(p, -1.0), -dims.get(p, 0), p))
with open(f'{a.run}/state/queue_cpu.txt', 'w') as io:
    io.write('\n'.join(queue) + '\n')
print(f'CPU queue: {len(queue)} problems, {sum(1 for p in queue if p in cpu_secs)} timed by --prev')
