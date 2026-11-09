# Water DMRG-TDA end to end in Julia (QNTensorTrains, SU(2) layer): Nakatani, Wouters, Van Neck & Chan,
# J. Chem. Phys. 140, 024108 (2014), Table IV. Every reference state is computed here.
#
#   per irrep Γ: aufbau CSF (+ one single excitation for Γ ≠ A1)
#     → stage A: in-irrep MALS ramp [16,32,64,128,128], 6 iterations, perturbative noise 1e-3 → 0
#     → stage B: noise-free in-irrep MALS at the pooled cap M until |ΔE| < 1e-8 (≤ 30 sweeps)
#     → typed Riemannian CG polish (per-irrep rank grid frozen) to ‖grad‖ ≤ POLISH_GTOL
#     → DMRG-TDA (tdaLowestDavidson) inside Γ, tolerance 1e-6
#   then E against per-irrep FCI (fci_vdz.txt) and the paper's Table IV.
#
# Usage (from this folder; QNTensorTrains as a sibling checkout of DMRG_exc_states, or QNTENSORTRAINS_PATH):
#   julia -t 32 tda_water_julia.jl [M] [irreps]
#     M       pooled SU(2) bond cap (Block's spin-adapted M counts the same reduced multiplets); default 500
#     irreps  comma list among A1,B1,A2,B2; default all four (run_tda.sh runs them one at a time, one log each)
# Environment: POLISH_GTOL (default 1e-8), POLISH_ITERS (800 CG iterations per call), SAVE_DIR (if set,
# each polished reference is saved there as <irrep>.jls with Serialization).
# Set OPENBLAS_NUM_THREADS=1 (QNTensorTrains' default too); threads come from `julia -t`.
#
# FCIDUMP is already in Fiedler order (fiedler_reorder.py), so the orbitals are used in file order.
# Recorded run: QNTensorTrains main 64182ca, AMD EPYC 9454, 32 threads, M = 500: logs in reference_logs/.
# Per irrep it took 2.4-3.8 h and 17-21 GB peak memory, with about 2.5 cores busy on average.
using Pkg
Pkg.activate(get(ENV, "QNTENSORTRAINS_PATH", joinpath(@__DIR__, "..", "..", "QNTensorTrains")); io=devnull)
using LinearAlgebra, Printf, Random, Serialization
using QNTensorTrains

M      = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 500
irreps = length(ARGS) ≥ 2 ? String.(split(ARGS[2], ",")) : ["A1", "B1", "A2", "B2"]

const IRREP_ID = Dict("A1" => 0, "A2" => 1, "B1" => 2, "B2" => 3)   # pyscf XOR irrep ids
const NSTATES  = Dict("A1" => 4, "A2" => 3, "B1" => 3, "B2" => 3)   # Table IV states per irrep (reference first)
# Table IV: (state, irrep, index in irrep, E_conv (4SA, M = 2000) / Ha, then E − E_conv / mHa for
# 4SA M = 500, TDA M = 500 and EOM-CCSD).
const PAPER = [
  ("X¹A₁", "A1", 1, -76.241697, 0.11, 0.01, 3.68),
  ("1¹B₁", "B1", 1, -75.939176, 0.20, 0.02, 1.50),
  ("1¹A₂", "A2", 1, -75.864445, 0.20, 0.02, 2.07),
  ("2¹A₁", "A1", 2, -75.842487, 0.18, 0.12, 2.08),
  ("1¹B₂", "B2", 1, -75.765964, 0.21, 0.02, 2.55),
  ("2¹B₂", "B2", 2, -75.696018, 0.20, 0.45, 4.14),
  ("3¹A₁", "A1", 3, -75.584080, 0.16, 0.10, 5.13),
  ("4¹A₁", "A1", 4, -75.462977, 0.26, 2.69, NaN),
  ("2¹A₂", "A2", 2, -75.448180, 0.23, 0.04, 4.53),
  ("3¹A₂", "A2", 3, -75.403286, 0.33, 0.27, NaN),
  ("2¹B₁", "B1", 2, -75.401515, 0.28, 0.25, 24.20),
  ("3¹B₁", "B1", 3, -75.381977, 0.26, 0.07, NaN),
  ("3¹B₂", "B2", 3, -75.322655, 0.26, 0.07, 5.75),
]

fcidump_data = read_fcidump(joinpath(@__DIR__, "FCIDUMP"))
d, N, e_core = fcidump_data.norb, fcidump_data.nelec, fcidump_data.E0
t, v = fcidump_data.t, fcidump_data.v
orbsym = molpro_to_xor(fcidump_data.orbsym_molpro, :C2v)
fci = Dict{Tuple{String,Int},Float64}()              # (irrep, 1-based index in irrep) => E
for l in eachline(joinpath(@__DIR__, "fci_vdz.txt"))
  startswith(l, "#") && continue
  f = split(l)
  fci[(String(f[1]), parse(Int, f[2]) + 1)] = parse(Float64, f[3])
end

println("== water cc-pVDZ: d=$d, N=$N, singlet, pooled SU(2) cap M=$M, irreps $(join(irreps, ","))")
println("   QNTensorTrains $(pkgversion(QNTensorTrains)) at $(pkgdir(QNTensorTrains)); orbital irreps $orbsym; ",
        "threads $(Threads.nthreads()), BLAS $(BLAS.get_num_threads())")
flush(stdout)
H = SU2.SparseHamiltonian(t, v, N, 0, d)

const STAGEB_MAX   = 30
const POLISH_ITERS = parse(Int, get(ENV, "POLISH_ITERS", "800"))
const POLISH_GTOL  = parse(Float64, get(ENV, "POLISH_GTOL", "1e-8"))

# A random SU(2) singlet of rank ≤ 2 per sector, projected onto irrep Γ: the TDA seeds.
function irrep_start(Γ; seed)
  Random.seed!(seed)
  r = [zeros(Int, N+1, N+1) for k = 1:d+1]
  for k = 1:d+1, (nl, sl) in SU2.state_qn(N, 0, d, k)
    r[k][nl, sl] = (k in (1, d+1) ? 1 : rand(1:2))
  end
  return first(irrep_symmetrize(round!(SU2.tt_randn(Val(d), Val(N), Val(0), r, Float64)), orbsym, Γ; tol=1e-12))
end

# Aufbau occupation: the N/2 lowest iterated Fock diagonals.
function aufbau_sites(t, v, nocc)
  occ = sortperm(diag(t))[1:nocc]
  for _ = 1:50
    F = [t[p,p] + sum(2v[p,p,i,i] - v[p,i,i,p] for i in occ) for p = 1:size(t, 1)]
    new = sortperm(F)[1:nocc]
    Set(new) == Set(occ) && break
    occ = new
  end
  return sort(occ)
end
occ = aufbau_sites(t, v, N ÷ 2)
# Start CSF: closed-shell aufbau for A1; otherwise one singlet excitation from the highest occupied site
# to the first virtual whose irrep reaches Γ.
function csf_start(Γ)
  codes = ones(Int, d); codes[occ] .= 4
  Γ == 0 && return SU2.tt_csf(codes)
  for o in reverse(occ)
    vir = findfirst(p -> !(p in occ) && xor(orbsym[p], orbsym[o]) == Γ, 1:d)
    vir === nothing && continue
    lo, hi = minmax(o, vir); codes[lo] = 2; codes[hi] = 3
    return SU2.tt_csf(codes)
  end
  error("no single excitation from the aufbau CSF reaches irrep $Γ")
end

# Typed CG polish (the per-irrep rank grid of the stage-B state is frozen), re-entered while the last
# call cut ‖grad‖ by at least 10 %.
function typed_polish(x, Γ; maxIter=POLISH_ITERS, gtol=POLISH_GTOL, restarts=20)
  λ0 = real(RayleighQuotient(H, x))
  g = QNTensorTrains._exact_gradnorm(H, x)
  for _ = 1:restarts
    _, x = riemannianCG(H, x; ε=gtol / abs(λ0), maxIter=maxIter, typed=orbsym, preserve_grid=true, verbosity=true)
    gn = QNTensorTrains._exact_gradnorm(H, x)
    @printf("   polish call: ‖g‖ %.3e\n", gn); flush(stdout)
    (gn ≤ gtol || gn ≥ 0.9g) && break
    g = gn
  end
  if 1 - irrep_weights(x, orbsym)[Γ+1] > 1e-12      # re-project only on leakage, onto x's own ranks
    x = round!(irrep_project(x, orbsym, Γ), [copy(rank(x, k)) for k = 1:d+1]); rmul!(x, inv(norm(x)))
  end
  return x
end

const RAMP = [16, 32, 64, 128, 128]
const FLUSHER = Timer(_ -> flush(stdout), 2; interval=2)   # keeps a batch log live
println("   aufbau sites $occ; stage-A ramp $RAMP; stage-B cap $M; polish gate $POLISH_GTOL")

function reference(name)
  Γ = IRREP_ID[name]
  sym = (orbsym, Γ)
  t0 = time()
  x = csf_start(Γ)
  _, x = QNTensorTrains.MALS!(H, x, 1e-8, RAMP, 6, false; noise=[1e-3, 1e-4, 1e-5, 1e-6, 0.0], symmetry=sym)
  Eprev = real(RayleighQuotient(H, x)); nB = 0
  @printf("   stage A  E = %.9f  %.1f s\n", Eprev + e_core, time() - t0); flush(stdout)
  tB = time()
  for s = 1:STAGEB_MAX
    nB = s
    _, x = QNTensorTrains.MALS!(H, x, 1e-10, [M], 1, false; symmetry=sym)
    En = real(RayleighQuotient(H, x))
    @printf("   stage B sweep %d  E = %.9f  ΔE %.1e  max bond %d  %.1f s\n", s, En + e_core, En - Eprev,
            maximum(sum(rank(x, k)) for k = 1:d+1), time() - tB); flush(stdout)
    abs(En - Eprev) < 1e-8 && break; Eprev = En
  end
  @printf("   stage B (cap %d)  E = %.9f  %d sweeps  %.1f s\n", M, real(RayleighQuotient(H, x)) + e_core, nB, time() - tB)
  flush(stdout)
  tP = time()
  x = typed_polish(first(irrep_symmetrize(x, orbsym, Γ; rmax=M)), Γ)
  λ = real(RayleighQuotient(H, x))
  g = QNTensorTrains._exact_gradnorm(H, x)
  leak = 1 - irrep_weights(x, orbsym)[Γ+1]
  @printf("   reference %s: E = %.9f (FCI %.9f, %+.3f mEh)  ‖grad_H‖ = %.1e  off-irrep %.1e  typed polish %.0f s (%.0f s)\n",
          name, λ + e_core, fci[(name, 1)], 1e3 * (λ + e_core - fci[(name, 1)]), g, leak, time() - tP, time() - t0)
  flush(stdout)
  haskey(ENV, "SAVE_DIR") && serialize(joinpath(mkpath(ENV["SAVE_DIR"]), "$name.jls"), x)
  return x, λ, g, leak
end

results = Dict{Tuple{String,Int},NamedTuple}()
for name in irreps
  Γ = IRREP_ID[name]
  k = NSTATES[name] - 1
  println("\n---- irrep $name (id $Γ): reference + $k TDA roots ----")
  x, λ, g, leak = reference(name)
  results[(name, 1)] = (; E=λ + e_core, res=g, offirrep=leak)

  t1 = time()
  ctx = SU2.WeingartenContext(H, x)
  baseL, baseR = bases(ctx)
  proj! = irrep_tangent_projector(ctx, orbsym, Γ)
  Random.seed!(200 + Γ)
  seeds = [project_tangent(baseL, baseR, irrep_start(Γ; seed=300 + 10Γ + j)) for j = 1:k+2]
  θ, ξs, info = tdaLowestDavidson(ctx, seeds, k; tol=1e-6, project=proj!, verbosity=0)
  @printf("TDA: converged=%s  iterations %d  A applies %d  space_dim %d  precond %s  (%.0f s)\n",
          info.converged, info.iterations, info.applies, info.space_dim, info.precond, time() - t1)
  for j = 1:k
    wj = irrep_weights(SU2.TTvector(ξs[j]), orbsym)
    results[(name, j+1)] = (; E=λ + θ[j] + e_core, res=info.residuals[j], offirrep=1 - wj[Γ+1])
    @printf("   root %d: θ = %.9f  E = %.9f  fresh ‖r‖ = %.1e  off-irrep %.1e\n",
            j + 1, θ[j], λ + θ[j] + e_core, info.residuals[j], 1 - wj[Γ+1])
  end
  flush(stdout)
end

println("\n==== this run vs FCI vs Nakatani et al. Table IV (mEh) ====")
println("state   E(TDA, ours)    E(FCI)          ours−FCI  paper TDA−conv")
for (s, name, j, econv, _, tda, _) in PAPER
  haskey(results, (name, j)) || continue
  E, ef = results[(name, j)].E, fci[(name, j)]
  @printf("%-6s  %.9f  %.9f  %7.3f   %7.2f\n", s, E, ef, 1e3 * (E - ef), tda)
  @printf("SUMMARY %s %s %d E=%.9f fci=%.9f conv=%.6f err=%.3e paper_err=%.2e res=%.1e offirrep=%.1e\n",
          s, name, j, E, ef, econv, E - ef, 1e-3 * tda, results[(name, j)].res, results[(name, j)].offirrep)
end
