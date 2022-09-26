using ITensors
using LinearAlgebra
using HDF5

import ITensors.op
op(::OpName"I", ::SiteType"S=1/2") = [1.0 0.0; 0.0 1.0]

svals = collect(0.0:0.0005:0.015)
Ns = vcat([4, 7], collect(10:5:30), [40, 50])
Ns = [30, 40, 50]
periodicdata = zeros(length(Ns), length(svals))
periodic = true

@time for (i, N) in enumerate(Ns)
    sites = siteinds("S=1/2", N)
    psi = randomMPS(sites, 10)
    for (k, s) in enumerate(svals)
        println("Sites: ", N, ", Bias: ", s)

        ampo = OpSum()
        for j = 1:N-1
            ampo += 1.0 / N, "ProjUp", j
        end
        if periodic
            ampo += -exp(-s) / N, "ProjUp", N, "X", 1
            ampo += 1.0 / N, "ProjUp", N
        else
            ampo += -exp(-s) / N, "X", 1
        end
        for j = 1:N-1
            ampo += -exp(-s) / N, "ProjUp", j, "X", j + 1
        end
        if periodic
            ampo += -1.0, "I", 1
        else
            ampo += -(1.0 - 1.0 / N), "I", 1
        end
        H = MPO(ampo, sites)
        darkstate = MPS(sites, ["Dn" for n = 1:N])


        sweeps = Sweeps(100)
        setmaxdim!(sweeps, 100, 200, 400, 600, 800, 1000)
        setcutoff!(sweeps, 1E-10)

        if periodic
            energy, psi = dmrg(H, [darkstate], psi, sweeps)
        else
            energy, psi = dmrg(H, psi0, sweeps)
        end
        periodicdata[i, k] = log(-energy)
    end
end
if periodic
    h5open("periodic_dmrg_data_zoom.hdf5", "cw") do file
        write(file, "data", periodicdata)
        write(file, "biases", svals)
        write(file, "sizes", Ns)
    end
end

