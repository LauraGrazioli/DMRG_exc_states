# Water DMRG-TDA with QNTensorTrains-rs (Julia bindings `QNTT`, SU(2) layer): the QNTensorTrains-rs counterpart
# of tda_water_julia.jl — Nakatani, Wouters, Van Neck & Chan, J. Chem. Phys. 140, 024108 (2014),
# Table IV, DMRG-TDA column.
#
#   per irrep Γ: the lowest singlet of Γ at pooled cap M (reference_rs.jl: aufbau CSF → MALS ramp →
#     MALS at the cap → Riemannian CG polish to ‖grad‖ ≤ POLISH_GTOL), in the point-group-typed space
#     of Γ → QNTT.tda: the lowest TDA roots on that state's tangent space, inside Γ
#   then E against per-irrep FCI (fci_vdz.txt) and the paper's Table IV.
#
# QNTT.tda returns θ, an energy relative to λ = ⟨x,Hx⟩ (E = λ + θ + E_core), on the radial-deflated
# tangent space of the typed manifold: inside Γ and the singlet sector by construction.
#
# Usage (from anywhere; setup as in reference_rs.jl; needs QNTensorTrains-rs ≥ 0.1.1 for QNTT.tda):
#   julia tda_water_rs.jl [M] [irreps]
#     M       pooled SU(2) bond cap (Block's spin-adapted M counts the same reduced multiplets); default 500
#     irreps  comma list among A1,B1,A2,B2; default all four (run_tda_rs.sh runs them one at a time)
# Environment: QNTT_THREADS, POLISH_GTOL (1e-8), POLISH_ITERS (800), SAVE_DIR (cache the polished
# references), TDA_TOL (absolute residual tolerance per root, hartree; default 1e-6).
include(joinpath(@__DIR__, "reference_rs.jl"))

M = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 500
names = length(ARGS) ≥ 2 ? String.(split(ARGS[2], ",")) : ["A1", "B1", "A2", "B2"]
const TDA_TOL = parse(Float64, get(ENV, "TDA_TOL", "1e-6"))

const NSTATES = Dict("A1" => 4, "A2" => 3, "B1" => 3, "B2" => 3)   # Table IV states per irrep (reference first)
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

println("== water cc-pVDZ DMRG-TDA: d = $D, N = $NEL, singlet, pooled SU(2) cap M = $M, irreps $(join(names, ",")); ",
        "QNTT $(QNTT.version()), threads $(QNTT.nthreads())")
results = Dict{Tuple{String,Int},NamedTuple}()
for name in names
    k = NSTATES[name] - 1
    println("\n---- irrep $name: reference + $k TDA roots ----")
    H, x, λ = reference(name, M)
    results[(name, 1)] = (; E = λ + E_CORE, res = QNTT.grad_norm(H, x))
    t1 = time()
    θ, info = QNTT.tda(H, x; k, tol = TDA_TOL, seed = 300 + IRREPS[name])
    @printf("TDA: converged %s  iterations %d  A applies %d  dim %d  precond %s  negatives certified %s  (%.0f s)\n",
            info.converged, info.iterations, info.applies, info.space_dim, info.precond,
            info.negatives_certified, time() - t1)
    for j in eachindex(θ)
        E = λ + θ[j] + E_CORE
        results[(name, j + 1)] = (; E, res = info.residuals[j])
        @printf("   root %d: θ = %.9f  E = %.9f  (FCI %.9f, %+.3f mEh)  fresh ‖r‖ = %.1e\n", j + 1, θ[j], E,
                FCI[(name, j + 1)], 1e3 * (E - FCI[(name, j + 1)]), info.residuals[j])
    end
end

println("\n==== this run vs FCI vs Nakatani et al. Table IV (mEh) ====")
println("state   E(TDA, rs)      E(FCI)          rs−FCI    paper TDA−conv")
for (s, name, j, econv, _, tda, _) in PAPER
    haskey(results, (name, j)) || continue
    E, ef = results[(name, j)].E, FCI[(name, j)]
    @printf("%-6s  %.9f  %.9f  %7.3f   %7.2f\n", s, E, ef, 1e3 * (E - ef), tda)
    @printf("SUMMARY %s %s %d E=%.9f fci=%.9f conv=%.6f err=%.3e paper_err=%.2e res=%.1e\n",
            s, name, j, E, ef, econv, E - ef, 1e-3 * tda, results[(name, j)].res)
end
