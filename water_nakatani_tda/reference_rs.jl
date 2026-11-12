# Shared by tda_water_rs.jl and gad_water_rs.jl: the water cc-pVDZ 8e/23o FCIDUMP (C2v, already in
# Fiedler order), the per-irrep FCI levels of fci_vdz.txt, and the per-irrep SU(2) singlet reference
# at a pooled bond cap M, computed with QNTensorTrains-rs (Julia bindings `QNTT`):
#
#   aufbau CSF (+ one singlet single excitation for Γ ≠ A1), in the point-group-typed space of Γ
#     → stage A: MALS ramp [16, 32, 64, 128, 128], perturbative noise 1e-3 → 0
#     → stage B: noise-free MALS at the cap M until |ΔE| < 1e-8 (≤ 30 sweeps)
#     → Riemannian CG polish on the fixed-rank manifold of Γ to ‖grad‖ ≤ POLISH_GTOL
#
# The space carries the point group, so every stage stays inside Γ by construction (no projection,
# no leakage to monitor). Setup: a checkout of https://github.com/paulcazeaux/QNTensorTrains-rs as a
# sibling of DMRG_exc_states (or QNTT_RS_PATH), with `cargo build --release -p qntt-capi` run once
# inside it; QNTT_LIB overrides the library file.
using Pkg
Pkg.activate(joinpath(get(ENV, "QNTT_RS_PATH", joinpath(@__DIR__, "..", "..", "QNTensorTrains-rs")), "julia"); io = devnull)
using LinearAlgebra, Printf, Serialization
using QNTT
# The library has its own thread pool (Julia's `-t` does not size it): QNTT_THREADS, default all
# cores. On a hyperthreaded machine, the physical core count is the better choice.
QNTT.init(parse(Int, get(ENV, "QNTT_THREADS", "0")))

const DATA = QNTT.read_fcidump(joinpath(@__DIR__, "FCIDUMP"); point_group = :C2v)
QNTT.screen_forbidden!(DATA)          # symmetry-forbidden integrals at roundoff level → exact zeros
const D, NEL, E_CORE = QNTT.norb(DATA), QNTT.nelec(DATA), QNTT.e0(DATA)
const ORBSYM = QNTT.orbsym(DATA)      # 1-based Molpro ids, one per orbital (FCIDUMP order = site order)
const IRREPS = Dict(QNTT.irrep_name(:C2v, g) => g for g in 0:3)   # name => 0-based XOR index
const ORB_IRREP = [QNTT.irrep_from_molpro(:C2v, s) for s in ORBSYM]

# Per-irrep singlet FCI levels: (irrep name, 1-based index in irrep) => E / Ha.
const FCI = let fci = Dict{Tuple{String,Int},Float64}()
    for l in eachline(joinpath(@__DIR__, "fci_vdz.txt"))
        startswith(l, "#") && continue
        f = split(l)
        fci[(String(f[1]), parse(Int, f[2]) + 1)] = parse(Float64, f[3])
    end
    fci
end

const POLISH_ITERS = parse(Int, get(ENV, "POLISH_ITERS", "800"))
const POLISH_GTOL = parse(Float64, get(ENV, "POLISH_GTOL", "1e-8"))
const RAMP = [16, 32, 64, 128, 128]
const STAGEB_MAX = 30

singlet_space(Γ) = QNTT.Su2Space(D, NEL, 0; point_group = :C2v, orbsym = ORBSYM, target = Γ)

# Aufbau occupation: the N/2 lowest iterated Fock diagonals.
function aufbau_sites(t, v, nocc)
    occ = sortperm(diag(t))[1:nocc]
    for _ = 1:50
        F = [t[p, p] + sum(2v[p, p, i, i] - v[p, i, i, p] for i in occ) for p = 1:size(t, 1)]
        new = sortperm(F)[1:nocc]
        Set(new) == Set(occ) && break
        occ = new
    end
    return sort(occ)
end
const OCC = aufbau_sites(QNTT.onebody(DATA), QNTT.twobody(DATA), NEL ÷ 2)

# Start CSF: closed-shell aufbau for A1; otherwise one open-shell singlet excitation from the highest
# occupied site to the first virtual whose irrep reaches Γ.
function csf_start(Γ, space)
    codes = fill(:empty, D)
    codes[OCC] .= :occ
    Γ == 0 && return QNTT.tt_csf(codes, space)
    for o in reverse(OCC)
        vir = findfirst(p -> !(p in OCC) && xor(ORB_IRREP[p], ORB_IRREP[o]) == Γ, 1:D)
        vir === nothing && continue
        lo, hi = minmax(o, vir)
        codes[lo] = :up
        codes[hi] = :dn
        return QNTT.tt_csf(codes, space)
    end
    error("no single excitation from the aufbau CSF reaches irrep $Γ")
end

maxbond(x) = maximum(sum, QNTT.ranks(x))

"""
    reference(name, M; save_dir = get(ENV, "SAVE_DIR", nothing)) -> (H, x, λ)

The lowest singlet of irrep `name` at pooled SU(2) cap `M` (reduced multiplets, Block's spin-adapted
M), with its Hamiltonian over the typed space. `λ` is ELECTRONIC (add `E_CORE`). With `save_dir`,
the polished state is saved there as `<name>_m<M>.tt` (`QNTT.save`), and reloaded on the next call.
"""
function reference(name, M; save_dir = get(ENV, "SAVE_DIR", nothing))
    Γ = IRREPS[name]
    space = singlet_space(Γ)
    H = QNTT.Hamiltonian(space, DATA)
    file = save_dir === nothing ? nothing : joinpath(mkpath(save_dir), "$(name)_m$M.tt")
    if file !== nothing && isfile(file)
        x = QNTT.load(QNTT.TTVector{Float64,QNTT.SU2}, file)
        λ = QNTT.rayleigh_quotient(H, x)
        @printf("   reference %s loaded from %s: E = %.9f  ‖grad‖ = %.1e\n", name, file, λ + E_CORE,
                QNTT.grad_norm(H, x))
        return H, x, λ
    end
    t0 = time()
    # The ramp runs to 128 whatever M is; stage B then truncates to the cap.
    _, x, _ = QNTT.mals(H, csf_start(Γ, space); bond_dims = RAMP, noise = [1e-3, 1e-4, 1e-5, 1e-6, 0.0],
                        eps = 1e-8, max_iter = 6)
    Eprev = QNTT.rayleigh_quotient(H, x)
    @printf("   stage A  E = %.9f  max bond %d  %.1f s\n", Eprev + E_CORE, maxbond(x), time() - t0)
    tB = time()
    nB = 0
    for s = 1:STAGEB_MAX
        nB = s
        En, x, _ = QNTT.mals(H, x; bond_dims = [M], eps = 1e-10, max_iter = 1)
        @printf("   stage B sweep %d  E = %.9f  ΔE %.1e  max bond %d  %.1f s\n", s, En + E_CORE, En - Eprev,
                maxbond(x), time() - tB)
        abs(En - Eprev) < 1e-8 && break
        Eprev = En
    end
    tP = time()
    g = QNTT.grad_norm(H, x)
    for _ = 1:20                        # re-enter while a call cuts ‖grad‖ by at least 10 %
        # `eps` is the stopping ratio ‖grad‖/|λ|: the absolute gate POLISH_GTOL, rescaled.
        λ0 = QNTT.rayleigh_quotient(H, x)
        _, x = QNTT.riemannian(H, x; eps = POLISH_GTOL / abs(λ0), max_iter = POLISH_ITERS)
        gn = QNTT.grad_norm(H, x)
        @printf("   polish call: ‖grad‖ %.3e\n", gn)
        (gn ≤ POLISH_GTOL || gn ≥ 0.9g) && break
        g = gn
    end
    λ = QNTT.rayleigh_quotient(H, x)
    @printf("   reference %s: E = %.9f (FCI %.9f, %+.3f mEh)  ‖grad‖ = %.1e  polish %.0f s (total %.0f s)\n",
            name, λ + E_CORE, FCI[(name, 1)], 1e3 * (λ + E_CORE - FCI[(name, 1)]), QNTT.grad_norm(H, x),
            time() - tP, time() - t0)
    file === nothing || QNTT.save(x, file)
    return H, x, λ
end
