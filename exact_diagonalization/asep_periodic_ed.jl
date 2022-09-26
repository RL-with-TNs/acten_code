include("operators.jl")
include("number_conservation.jl")

using LinearAlgebra
using SparseArrays

function bond_asep_model(N, p, q, s, T=identity)
    master_op = two_site_operator((p * exp(-s)) .* T(c⁻), N, T(c⁺), 1, N)
    master_op += two_site_operator((q * exp(s)) .* T(c⁺), N, T(c⁻), 1, N)
    master_op -= two_site_operator(p .* T(n), N, T(𝐈 - n), 1, N)
    master_op -= two_site_operator(q .* T(𝐈 - n), N, T(n), 1, N)
    for i = 1:N-1
        master_op += two_site_operator((p * exp(-s)) .* T(c⁻), i, T(c⁺), i + 1, N)
        master_op += two_site_operator((q * exp(s)) .* T(c⁺), i, T(c⁻), i + 1, N)
        master_op -= two_site_operator(p .* T(n), i, T(𝐈 - n), i + 1, N)
        master_op -= two_site_operator(q .* T(𝐈 - n), i, T(n), i + 1, N)
    end
    return (master_op ./ N) + I(2^N)
end

function particle_asep_model(L, N, p, s, T=identity)
    master_op = two_site_operator((p * exp(-s)) .* T(c⁻), L, T(c⁺), 1, L)
    master_op += two_site_operator(((1 - p) * exp(s)) .* T(c⁺), L, T(c⁻), 1, L)
    master_op -= two_site_operator(p .* T(n), L, T(𝐈 - n), 1, L)
    master_op -= two_site_operator((1 - p) .* T(𝐈 - n), L, T(n), 1, L)
    for i = 1:L-1
        master_op += two_site_operator((p * exp(-s)) .* T(c⁻), i, T(c⁺), i + 1, L)
        master_op += two_site_operator(((1 - p) * exp(s)) .* T(c⁺), i, T(c⁻), i + 1, L)
        master_op -= two_site_operator(p .* T(n), i, T(𝐈 - n), i + 1, L)
        master_op -= two_site_operator((1 - p) .* T(𝐈 - n), i, T(n), i + 1, L)
    end
    proj_mast_op = project_block(master_op, L, N + 1) ./ N
    return proj_mast_op + I(size(proj_mast_op)[1])
end

sizes = [2 * i for i = 2:7]
biases = -3.0:0.05:3.0
rate_diff = [0.1 * i for i = 5:10]
params = Iterators.product(enumerate(sizes), enumerate(biases), enumerate(rate_diff))

bond_scgfs = zeros(length(sizes), length(biases), length(rate_diff))
particle_scgfs = zeros(length(sizes), length(biases), length(rate_diff))
for param in params
    (i, N), (j, s), (k, p) = param
    println(getindex.(param, 1), " ", getindex.(param, 2))
    bond_mastop = Matrix(
        project_block(bond_asep_model(N, 1.0 - p, p, s, SparseMatrixCSC), N, N ÷ 2 + 1))
    particle_mastop = Matrix(particle_asep_model(N, N ÷ 2, p, s, SparseMatrixCSC))
    @time bond_scgfs[i, j, k] = log(real(last(eigvals(bond_mastop))))
    @time particle_scgfs[i, j, k] = log(real(last(eigvals(particle_mastop))))
end

using HDF5
h5open("periodic_bond_asep_ED_2.05_to_3.0.hdf5", "w") do file
    write(file, "data", bond_scgfs)
    write(file, "sizes", sizes)
    write(file, "biases", collect(biases))
    write(file, "transition_rates", rate_diff)
end
h5open("periodic_particle_asep_ED_2.05_to_3.0.hdf5", "w") do file
    write(file, "data", particle_scgfs)
    write(file, "sizes", sizes)
    write(file, "biases", collect(biases))
    write(file, "transition_rates", rate_diff)
end

using GLMakie

# Correction
invert_p = -1

# Max system size curves for different p
fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel=L"s", ylabel=L"\theta(s)")
for i = 1:length(rate_diff)
    lines!(ax1, invert_p .* biases, bond_scgfs[6, :, i], label="p=$(round(rate_diff[i], digits=1))")
end
axislegend(ax1, position=:ct)
ax2 = fig[1, 2] = Axis(fig, xlabel=L"s", ylabel=L"J(s)")
for i = 1:length(rate_diff)
    lines!(ax2, invert_p .* biases[1:end-1] .+ 0.1, invert_p .* diff(bond_scgfs[6, :, i]),
        label="p=$(round(rate_diff[i], digits=2))")
end
axislegend(ax2, position=:lt)
ax3 = fig[2, 1] = Axis(fig, xlabel=L"s", ylabel=L"\theta(s)")
for i = 1:length(rate_diff)
    lines!(ax3, biases, particle_scgfs[6, :, i], label="p=$(round(rate_diff[i], digits=1))")
end
axislegend(ax3, position=:ct)
ax4 = fig[2, 2] = Axis(fig, xlabel=L"s", ylabel=L"J(s)")
for i = 1:length(rate_diff)
    lines!(ax4, biases[1:end-1] .+ 0.1, diff(particle_scgfs[6, :, i]),
        label="p=$(round(rate_diff[i], digits=2))")
end
axislegend(ax4, position=:lt)

# p=0.9 curves for different system sizes
fig = Figure()
ax1 = fig[1, 1] = Axis(fig, xlabel=L"s", ylabel=L"\theta(s)")
for i = 1:length(sizes)
    lines!(ax1, invert_p .* biases, bond_scgfs[i, :, 5], label="L=$(sizes[i])")
end
axislegend(ax1, position=:ct)
ax2 = fig[1, 2] = Axis(fig, xlabel=L"s", ylabel=L"J(s)")
for i = 1:length(sizes)
    lines!(ax2, invert_p .* biases[1:end-1] .+ 0.1, invert_p .* diff(bond_scgfs[i, :, 5]),
        label="L=$(sizes[i])")
end
axislegend(ax2, position=:lt)
ax3 = fig[2, 1] = Axis(fig, xlabel=L"s", ylabel=L"\theta(s)")
for i = 1:length(sizes)
    lines!(ax3, biases, particle_scgfs[i, :, 5], label="L=$(sizes[i])")
end
axislegend(ax3, position=:ct)
ax4 = fig[2, 2] = Axis(fig, xlabel=L"s", ylabel=L"J(s)")
for i = 1:length(sizes)
    lines!(ax4, biases[1:end-1] .+ 0.1, diff(particle_scgfs[i, :, 5]),
        label="L=$(sizes[i])")
end
axislegend(ax4, position=:lt)

save("corrected_p0.9_finite_size_scaling.png", fig)