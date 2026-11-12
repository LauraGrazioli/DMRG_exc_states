# Water CAS(8e,6o): state-average DMRG with QNTensorTrains-rs (Julia bindings `QNTT`).
#
# The same calculation as sa_dmrg_qntensortrains.ipynb, on the public Rust library: the two block2
# state-average runs in this folder, 3 roots with equal weights, bond dimension M = 50, on the
# active-space integrals in FCIDUMP (water/6-31G, AVAS: 6 orbitals, 8 electrons):
#
#   dmrg_sa_SSS.out (spin 0 0 0): three singlets            -> SU(2) layer, total spin S = 0
#   dmrg_sa_STS.out (spin 0 2 0): singlet, triplet, singlet  -> determinant layer, S_z = 0
#
# An SU(2) tensor train carries a single total spin, so the spin-mixed STS average runs in the
# S_z = 0 determinant basis, where the three lowest levels are S0, T1, S1. Both runs are then
# compared with block2 (energies, occupations, full 1-RDMs of the STS run) and with a dense FCI
# built here from the integrals.
#
# Usage (from anywhere):
#   julia sa_dmrg_rs.jl
# Setup: a checkout of https://github.com/paulcazeaux/QNTensorTrains-rs as a sibling of
# DMRG_exc_states (or set QNTT_RS_PATH), with the C library built once:
#   cargo build --release -p qntt-capi          (inside the checkout)
# QNTT finds the library in the checkout's target directory; set QNTT_LIB to the library file
# (libqntt_capi.dylib / .so) if it lives elsewhere. Runs in a few seconds.
using Pkg
Pkg.activate(joinpath(get(ENV, "QNTT_RS_PATH", joinpath(@__DIR__, "..", "..", "QNTensorTrains-rs")), "julia"); io = devnull)
using LinearAlgebra, Printf
using QNTT

# ---- Integrals --------------------------------------------------------------------------------
# t[i,j] = h_ij, v[i,j,k,l] = (ij|kl), E0 = nuclear repulsion + frozen core. The orbitals are put
# in block2's order (its Fiedler reordering [4 5 1 0 2 3], printed in the .out files), so the
# tensor train has the same site order as block2's MPS: site k holds orbital perm[k].
fd = QNTT.read_fcidump(joinpath(@__DIR__, "FCIDUMP"))
E0, d, N = QNTT.e0(fd), QNTT.norb(fd), QNTT.nelec(fd)
t_fd, v_fd = QNTT.onebody(fd), QNTT.twobody(fd)

perm = [5, 6, 2, 1, 3, 4]
t = t_fd[perm, perm]
v = v_fd[perm, perm, perm, perm]
occupied = [perm[k] ≤ N ÷ 2 for k in 1:d]   # reference: orbitals 1-4 doubly occupied
println("d = $d, N = $N, E0 = $E0, QNTT $(QNTT.version()), threads $(QNTT.nthreads())")

# ---- State-average DMRG -----------------------------------------------------------------------
# Two-site DMRG on nroots states sharing every core except the orthogonality centre; each bond is
# truncated against the weighted density matrix Σᵢ wᵢ ρᵢ, as in block2. Settings follow block2's:
# M = 50, equal weights, perturbative density-matrix noise 1e-4 then 1e-3, then 0.
#
# Start: the closed-shell reference plus a random perturbation of relative size 1e-2 with rank 1 in
# every sector. The noise only adds directions H couples to the current states, so from the bare
# reference every root stays in the reference's spatial symmetry and the lowest excitations (which
# empty orbital 4) are never found.
const NOISE = [1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 0.0]

function perturbed(x0, space, ε; seed = 1234)
    p = QNTT.tt_randn(space, QNTT.uniform_ranks(space, 1); seed)
    return x0 + (ε * norm(x0) / norm(p)) * p
end

function sa_dmrg(H, x0, space; nroots = 3, M = 50)
    λ, ψ, history = QNTT.state_average_mals(H, perturbed(x0, space, 1e-2), nroots;
                                            weights = fill(1 / nroots, nroots), bond_dims = [M],
                                            noise = NOISE, eps = 1e-8, max_iter = 30)
    for (it, h) in enumerate(history)
        @printf("iter %2d  noise %.0e  Σ wᵢEᵢ = %.10f  max bond %d  discarded %.1e\n", it, h.noise,
                h.energy + E0, h.max_bond_dim, h.discarded_weight)
    end
    return λ .+ E0, ψ
end

println("\nSSS: three singlets (SU(2) layer)")
space_sss = QNTT.Su2Space(d, N, 0)
H_sss = QNTT.Hamiltonian(space_sss, t, v)
csf = QNTT.tt_csf([occ ? :occ : :empty for occ in occupied])
E_sss, ψ_sss = sa_dmrg(H_sss, csf, space_sss)

println("\nSTS: singlet, triplet, singlet (S_z = 0 determinant layer)")
space_sts = QNTT.QnSpace(d, N ÷ 2, N ÷ 2)
H_sts = QNTT.Hamiltonian(space_sts, t, v)
hf = QNTT.tt_state(occupied, occupied)
E_sts, ψ_sts = sa_dmrg(H_sts, hf, space_sts)

# ---- Spin-summed 1-RDMs -----------------------------------------------------------------------
# γ_pq = Σσ ⟨a†_pσ a_qσ⟩ in site order, read as expectation values of one-body operators: the
# diagonal from n_p, the off-diagonal from the Hermitian (E_pq + E_qp)/2. On the SU(2) layer these
# are spin-free singlet operators, so the same construction works on both layers.
function one_rdm(ψ, space)
    z = zeros(d, d, d, d)
    γ = zeros(d, d)
    nψ = real(dot(ψ, ψ))
    for p in 1:d, q in p:d
        e = zeros(d, d)
        e[p, q] = e[q, p] = p == q ? 1.0 : 0.5
        γ[p, q] = γ[q, p] = real(QNTT.expectation(QNTT.Hamiltonian(space, e, z), ψ)) / nψ
    end
    return γ
end
to_fcidump_order(γ) = (g = zeros(d, d); g[perm, perm] = γ; g)
γ_sss = [to_fcidump_order(one_rdm(ψ, space_sss)) for ψ in ψ_sss]
γ_sts = [to_fcidump_order(one_rdm(ψ, space_sts)) for ψ in ψ_sts]

# ---- Comparison with block2 -------------------------------------------------------------------
# Final energies and the DMRG OCC lines (1-RDM diagonal, FCIDUMP orbital order) of the two .out files.
block2 = (sss = (E = [-75.994272293973580, -75.6831488437, -75.6025875550],
                 occ = [[1.998, 1.989, 1.994, 2.000, 0.009, 0.010],
                        [1.994, 1.972, 1.988, 1.000, 1.015, 0.031],
                        [1.994, 1.976, 1.985, 1.000, 0.027, 1.018]]),
          sts = (E = [-75.994272555924653, -75.7018502158, -75.6831275493],
                 occ = [[1.998, 1.989, 1.994, 2.000, 0.009, 0.010],
                        [1.998, 1.973, 1.989, 1.000, 1.010, 0.030],
                        [1.994, 1.972, 1.988, 1.000, 1.015, 0.031]]))

function compare(name, E, γs, ref)
    @printf("\n%-6s  %-17s  %-14s  %-8s     %s\n", name, "E(rs)", "E(block2)", "diff",
            "occupations rs / block2")
    for i in eachindex(E)
        @printf("root %d  %-17.10f  %-14.10f  %+.1e  %s / %s\n", i - 1, E[i], ref.E[i], E[i] - ref.E[i],
                join([@sprintf("%.3f", o) for o in diag(γs[i])], " "),
                join([@sprintf("%.3f", o) for o in ref.occ[i]], " "))
    end
end
compare("SSS", E_sss, γ_sss, block2.sss)
compare("STS", E_sts, γ_sts, block2.sts)

# Full spin-resolved 1-RDMs of the STS run, saved by block2 as .npy files in its run folder node0/
# (removed from the working tree after commit 416be13, so read from git), in FCIDUMP orbital order.
function read_npy(bytes::Vector{UInt8})   # format 1.x, little-endian Float64, no extra package
    bytes[7] == 0x01 || error("only .npy format version 1.x is handled")
    hlen = Int(bytes[9]) + 256 * Int(bytes[10])
    header = String(bytes[11:10+hlen])
    occursin("'descr': '<f8'", header) && occursin("'fortran_order': False", header) ||
        error("unsupported .npy header: $header")
    shape = Tuple(parse.(Int, split(match(r"'shape': \(([^)]*)\)", header)[1], ",", keepempty = false)))
    data = collect(reinterpret(Float64, bytes[11+hlen:end]))
    isempty(shape) && return data[1]
    return permutedims(reshape(data, reverse(shape)), length(shape):-1:1)   # C order → Julia order
end
node0(file) = read_npy(read(`git -C $(@__DIR__) show 416be13:water_block2/node0/$file`))

others = [1, 2, 3, 5, 6]
println("\nSTS 1-RDMs   max|γ_block2 − γ_rs|                  coupling of orbital 4 to the others")
println("root         all       diagonal  without orbital 4  rs        block2")
for i in 1:3
    γb = dropdims(sum(node0("1pdm-$(i-1)-$(i-1).npy"), dims = 1), dims = 1)   # spin-summed
    Δ = γb - γ_sts[i]
    @printf("%d            %.1e   %.1e   %.1e            %.1e   %.1e\n", i - 1, maximum(abs, Δ),
            maximum(abs, diag(Δ)), maximum(abs, Δ[others, others]),
            maximum(abs, γ_sts[i][4, others]), maximum(abs, γb[4, others]))
end

# ---- Check against exact FCI ------------------------------------------------------------------
# 225 determinants at S_z = 0, built directly from the integrals by second quantization on bit
# strings, independently of the QNTensorTrains-rs Hamiltonian (so it also checks how the FCIDUMP convention is
# read): H = E0 + Σ t_ij a†_iσ a_jσ + ½ Σ (ij|kl) a†_iσ a†_kτ a_lτ a_jσ. A level is a singlet when
# it does not also appear in the S_z = 1 spectrum.
function apply(b, ops)   # creation (true) / annihilation (false) string, rightmost first
    s = 1
    for (q, create) in Iterators.reverse(ops)
        isodd(b >> q) == create && return (b, 0)
        isodd(count_ones(b & ((one(b) << q) - 1))) && (s = -s)
        b ⊻= one(b) << q
    end
    return (b, s)
end

function fci_levels(t, v, E0, nup, ndn)
    d = size(t, 1)
    so(p, σ) = 2(p - 1) + σ
    upmask = sum(UInt(1) << so(p, 0) for p in 1:d)
    dets = [b for b in UInt(0):UInt(4^d - 1) if count_ones(b & upmask) == nup && count_ones(b & ~upmask) == ndn]
    index = Dict(b => i for (i, b) in enumerate(dets))
    Hm = Matrix(E0 * I, length(dets), length(dets))
    for (j, b) in enumerate(dets), σ in 0:1, p in 1:d, q in 1:d
        b1, s1 = apply(b, ((so(p, σ), true), (so(q, σ), false)))
        s1 == 0 || (Hm[index[b1], j] += s1 * t[p, q])
        for τ in 0:1, r in 1:d, u in 1:d
            b2, s2 = apply(b, ((so(p, σ), true), (so(r, τ), true), (so(u, τ), false), (so(q, σ), false)))
            s2 == 0 || (Hm[index[b2], j] += s2 * v[p, q, r, u] / 2)
        end
    end
    return eigvals(Symmetric(Hm))
end

E_sz0 = fci_levels(t_fd, v_fd, E0, N ÷ 2, N ÷ 2)
E_sz1 = fci_levels(t_fd, v_fd, E0, N ÷ 2 + 1, N ÷ 2 - 1)
spin(e) = any(abs.(E_sz1 .- e) .< 1e-8) ? "S≥1" : "S=0"
singlets = filter(e -> spin(e) == "S=0", E_sz0)
println()
for (name, E, exact, ref) in (("SSS", E_sss, singlets[1:3], block2.sss), ("STS", E_sts, E_sz0[1:3], block2.sts))
    println(name)
    for i in 1:3
        @printf("  root %d  FCI %.10f (%s)   rs − FCI %+.1e   block2 − FCI %+.1e\n", i - 1, exact[i],
                spin(exact[i]), E[i] - exact[i], ref.E[i] - exact[i])
    end
end
