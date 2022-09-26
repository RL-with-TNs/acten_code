include("operators.jl")
include("number_conservation.jl")

using LinearAlgebra
using SparseArrays
using HDF5

function asep_model(L, N, p, s, T=identity)
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

sizes = 2 * (2:7)
biases = -3.0:0.05:3.0
rate_diff = 0.1 * (5:10)
params = Iterators.product(enumerate(sizes), enumerate(biases), enumerate(rate_diff))

scgfs = zeros(length(sizes), length(biases), length(rate_diff))
for param in params
    (i, N), (j, s), (k, p) = param
    println(getindex.(param, 1), " ", getindex.(param, 2))
    mastop = Matrix(asep_model(N, N ÷ 2, p, s, SparseMatrixCSC))
    @time scgfs[i, j, k] = log(real(last(eigvals(mastop))))
end

h5open("periodic_asep_ED_-3.0_to_3.0.hdf5", "w") do file
    write(file, "data", scgfs)
    write(file, "sizes", collect(sizes))
    write(file, "biases", collect(biases))
    write(file, "transition_rates", collect(rate_diff))
end
