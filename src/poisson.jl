grid = Array(1:120)
permittivity = fill(10, size(grid))
phi = rand(120)
function poisson(grid, permittivity, phi)
    dgrid = diff(grid)
    avg_grid = dgrid[1:end-1] + dgrid[2:end]
    avg_perm = permittivity[1:end-1] + permittivity[2:end]
    ((avg_perm[1:end-1] .* diff(phi[1:end-1]) ./ dgrid[1:end-1]) - (avg_perm[2:end] .* diff(phi[2:end]) ./ dgrid[2:end])) ./ avg_grid
end


poisson(grid, permittivity, phi)