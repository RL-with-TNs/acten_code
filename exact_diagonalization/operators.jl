const c⁺ = [0.0 1.0; 0.0 0.0]
const c⁻ = [0.0 0.0; 1.0 0.0]
const n = [1.0 0.0; 0.0 0.0]
const 𝐈 = [1.0 0.0; 0.0 1.0]
const σˣ = [0.0 1.0; 1.0 0.0]
const σʸ = [0.0 -1.0im; 1.0im 0.0]
const σᶻ = [1.0 0.0; 0.0 -1.0]

⊗(a, b) = kron(a, b)

function single_site_operator(O, i, N)
    if i > N
        error("The site i = $i the operator is located at must"
              *
              " be less than the number of sites N = $N.")
    end
    out = 1.0
    for site = 1:i-1
        out = out ⊗ 𝐈
    end
    out = out ⊗ O
    for site = i+1:N
        out = out ⊗ 𝐈
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
            out = out ⊗ O1
        elseif i2 == i
            out = out ⊗ O2
        else
            out = out ⊗ 𝐈
        end
    end
    return out
end