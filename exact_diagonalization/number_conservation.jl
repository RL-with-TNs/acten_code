eigenspace_indices(N, m) = (0:2^N-1)[eigenspace_bools(N, m)] .+ 1
eigenspace_indices(N) = [eigenspace_indices(N, m) for m = 1:N+1]
eigenspace_bools(N, m) = count_ones.(0:2^N-1) .== m-1
eigenspace_bools(N) = [eigenspace_bools(N, m) for m = 1:N+1]
project_space(op, space) = op[space, space]
project_space(op, space1, space2) = op[space1, space2]
project_diagonal(op, N) = [project_space(op, space) for space in eigenspace_indices(N)]
project_block(op, N, m) = project_space(op, eigenspace_indices(N, m))
function project_diagonal(op, N, shift)
    spaces = eigenspace_indices(N)
    from_indices = (1:N+1)[1 .<= (1:N+1) .- shift .<= N + 1]
    [project_space(op, spaces[i-shift], spaces[i]) for i in from_indices]
end
function project_blocks(op, N)
    spaces = eigenspace_indices(N)
    temp = [[project_space(op, space1, space2) for space1 in spaces] for space2 in spaces]
    reduce(hcat, temp)
end
function embedding(N, m)
    inds = eigenspace_indices(N, m)
    E = zeros(2^N, length(inds))
    for i=1:length(inds)
        E[inds[i], i] = 1
    end
    return E
end
half_filling_sector(N) = N÷2 + 1