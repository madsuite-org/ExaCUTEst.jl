# Solve ONE oracle-tier problem. Run per-process so a single Hessian that
# takes longer than the budget cannot stall the sweep: MadNLP's max_wall_time
# is checked between iterations, and on these problems one Hessian evaluation
# can exceed the whole budget (the objective block is rebuilt one column at a
# time through Fortran hprod calls).
using Pkg; Pkg.activate(expanduser("~/git/scratch/solveenv"))
using ExaCUTEst, MadNLP, NLPModels
name = ARGS[1]
OUT = expanduser("~/git/scratch/solve34.tsv")
row = [name, "", "", "", "", "", "", ""]
try
    tb = @elapsed m = ExaCUTEstModel(name)
    row[2] = string(m.meta.nvar); row[3] = string(m.meta.ncon)
    row[7] = string(round(tb, sigdigits = 3))
    ts = @elapsed res = madnlp(m; max_wall_time = 60.0, print_level = MadNLP.ERROR,
                               max_iter = 3000)
    row[4] = string(res.status); row[6] = string(res.iter)
    row[5] = string(round(res.objective, sigdigits = 10))
    row[8] = string(round(ts, sigdigits = 3))
catch e
    row[4] = "ERROR"
    row[5] = replace(first(sprint(showerror, e), 80), '\t' => ' ', '\n' => ' ')
end
open(OUT, "a") do io; println(io, join(row, '\t')); end
