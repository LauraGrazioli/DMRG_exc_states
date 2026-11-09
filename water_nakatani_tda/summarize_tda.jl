# Table IV from the SUMMARY lines of tda_water_julia.jl logs, against FCI, the paper and the recorded run.
#   julia summarize_tda.jl [logs...]        (default: logs/*.log)
# The recorded M = 500 run (reference_logs/) is read for the last column; its logs use the older
# `SUMMARY <irrep> <j> E=…` form, the new driver prints `SUMMARY <state> <irrep> <j> E=…`: both are read.
using Printf

function read_summaries(files)
  E = Dict{Tuple{String,Int},Float64}()
  for f in files, l in eachline(f)
    m = match(r"SUMMARY (?:\S+ )?([AB]\d) (\d+) E=(-?[\d.]+)", l)
    m === nothing || (E[(String(m[1]), parse(Int, m[2]))] = parse(Float64, m[3]))
  end
  return E
end
here(p...) = joinpath(@__DIR__, p...)
files = isempty(ARGS) ? filter(endswith(".log"), readdir(here("logs"); join=true)) : ARGS
ours = read_summaries(files)
recorded = read_summaries(filter(endswith(".log"), readdir(here("reference_logs"); join=true)))
fci = Dict{Tuple{String,Int},Float64}()
for l in eachline(here("fci_vdz.txt"))
  startswith(l, "#") && continue
  f = split(l); fci[(String(f[1]), parse(Int, f[2]) + 1)] = parse(Float64, f[3])
end
# (state, irrep, index in irrep, paper DMRG-TDA M = 500 error E − E_conv / mHa)
const ROWS = [("X¹A₁", "A1", 1, 0.01), ("1¹B₁", "B1", 1, 0.02), ("1¹A₂", "A2", 1, 0.02), ("2¹A₁", "A1", 2, 0.12),
              ("1¹B₂", "B2", 1, 0.02), ("2¹B₂", "B2", 2, 0.45), ("3¹A₁", "A1", 3, 0.10), ("4¹A₁", "A1", 4, 2.69),
              ("2¹A₂", "A2", 2, 0.04), ("3¹A₂", "A2", 3, 0.27), ("2¹B₁", "B1", 2, 0.25), ("3¹B₁", "B1", 3, 0.07),
              ("3¹B₂", "B2", 3, 0.07)]
println("state   E (this run) / Ha   E − E_FCI / mHa   paper TDA / mHa   this run − recorded / µHa")
for (s, ir, j, paper) in ROWS
  haskey(ours, (ir, j)) || continue
  e = ours[(ir, j)]
  r = get(recorded, (ir, j), NaN)
  @printf("%-6s  %.9f       %8.3f          %6.2f            %+9.3f\n", s, e, 1e3 * (e - fci[(ir, j)]), paper, 1e6 * (e - r))
end
