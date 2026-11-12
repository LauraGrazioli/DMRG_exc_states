# Gentlest-ascent dynamics (GAD) on this folder's water cc-pVDZ 8e/23o FCIDUMP with
# QNTensorTrains-rs (Julia bindings `QNTT`): the counterpart of run_gad.sh. GAD climbs from a
# ground state to an index-1 saddle of the Rayleigh quotient on the fixed-rank SU(2) manifold, i.e. an
# excited state. Per irrep Γ:
#   reference: the lowest singlet of Γ at pooled cap M (reference_rs.jl: aufbau CSF → MALS ramp →
#              MALS at the cap → Riemannian CG polish), in the point-group-typed space of Γ
#   launch:    QNTT.saddle_seed — a kick along the softest mode of the real Hessian at the reference
#   GAD:       QNTT.gad on the typed manifold — PINNED to Γ, so it targets Γ's second singlet
#              (FCI level (Γ, 2) at full rank)
#
# Differences from run_gad.sh, all on the library side: QNTensorTrains-rs's GAD has the Rust step controller only
# (run_gad.sh's GAD_STEP=rust, no Newton tail) and seeds the tracked mode with the lowest term-A
# mode (GAD_V0=0); there is no unpinned run with a symmetry monitor. With those settings run_gad.sh
# measured, at M = 25: A1 refused at the manifold border, B1 at the 1000-step cap. Every result is a
# RANK-CONSTRAINED saddle at the reference's grid — quote M with it; at M = 25 the saddles sat 40-64 mEh
# above the FCI levels.
#
# Usage (from anywhere; setup as in reference_rs.jl):
#   julia gad_water_rs.jl [M] [irreps] [max_steps]
#     M          pooled SU(2) bond cap of the reference (default 25)
#     irreps     comma list among A1,B1,A2,B2 (default all four)
#     max_steps  GAD accepted-step budget per irrep (default 1000)
# Environment: QNTT_THREADS (library threads, default all cores), SAVE_DIR (cache the polished
# references there and reuse them), POLISH_GTOL (1e-8).
# Measured (M = 25, A2, 16 threads, Xeon): reference 72 s (+20.4 mEh); GAD 504 steps, 1283 s, a
# certified index-1 saddle 35.2 mEh above FCI A2 #2.
include(joinpath(@__DIR__, "reference_rs.jl"))

M = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 25
names = length(ARGS) ≥ 2 ? String.(split(ARGS[2], ",")) : ["A1", "B1", "A2", "B2"]
max_steps = length(ARGS) ≥ 3 ? parse(Int, ARGS[3]) : 1000

println("== water cc-pVDZ GAD: d = $D, N = $NEL, singlet, pooled SU(2) cap M = $M, irreps $(join(names, ",")), ",
        "≤ $max_steps steps; QNTT $(QNTT.version()), threads $(QNTT.nthreads())")
rows = []
for name in names
    println("\n---- irrep $name: reference, launch, pinned GAD ----")
    H, x, λ = reference(name, M)
    t0 = time()
    xk, λw = QNTT.saddle_seed(H, x)
    @printf("   launch: softest Hessian mode %.3e  E(kick) = %.9f\n", λw, QNTT.rayleigh_quotient(H, xk) + E_CORE)
    xe, info = QNTT.gad(H, xk; max_steps)
    E = info.lambda + E_CORE
    fci = FCI[(name, 2)]
    @printf("   GAD: converged %s  index-1 certified %s  index estimate %s  border %s  steps %d  applies %d  ‖g‖ %.1e  %.0f s\n",
            info.converged, info.index1_certified, something(info.index_estimate, "-"), info.border_verdict,
            info.steps, info.applies, info.gnorm, time() - t0)
    @printf("   saddle E = %.9f  (FCI %s #2 %.9f, %+.3f mEh; reference %+.3f mEh)\n", E, name, fci,
            1e3 * (E - fci), 1e3 * (λ + E_CORE - FCI[(name, 1)]))
    push!(rows, (name, info.converged, info.index1_certified, info.steps, E, E - fci))
end

println("\nirrep  converged  index-1  steps  E(saddle) / Ha     E − E_FCI(#2) / mEh")
for (name, conv, cert, steps, E, err) in rows
    @printf("%-5s  %-9s  %-7s  %5d  %.9f    %8.3f\n", name, conv, cert, steps, E, 1e3 * err)
end
