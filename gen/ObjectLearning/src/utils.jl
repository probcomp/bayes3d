normalize(v::Vector{Float64}) = isempty(v) ? v : v ./ sum(v)
