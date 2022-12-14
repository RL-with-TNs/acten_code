const cβΊ = [0.0 1.0; 0.0 0.0]
const cβ» = [0.0 0.0; 1.0 0.0]
const n = [1.0 0.0; 0.0 0.0]
const π = [1.0 0.0; 0.0 1.0]
const ΟΛ£ = [0.0 1.0; 1.0 0.0]
const ΟΚΈ = [0.0 -1.0im; 1.0im 0.0]
const ΟαΆ» = [1.0 0.0; 0.0 -1.0]

β(a, b) = kron(a, b)

function single_site_operator(O, i, N)
    if i > N
        error("The site i = $i the operator is located at must"
              *
              " be less than the number of sites N = $N.")
    end
    out = 1.0
    for site = 1:i-1
        out = out β π
    end
    out = out β O
    for site = i+1:N
        out = out β π
    end
    return out
end

function two_site_operator(O1, i1, O2, i2, N)
    if i1 > N
        error("The site i1 = $i1 the operator is located at must"
              *
              " be less than the number of sites N = $N.")
    end
    if i2 > N
        error("The site i2 = $i2 the operator is located at must"
              *
              " be less than the number of sites N = $N.")
    end
    if i1 == i2
        error("The sites i1 = $i1 and i2 = $i2 must be different.")
    end
    out = 1.0
    for i = 1:N
        if i1 == i
            out = out β O1
        elseif i2 == i
            out = out β O2
        else
            out = out β π
        end
    end
    return out
end