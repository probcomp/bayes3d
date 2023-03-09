using Test

@testset "ObjectLearning.jl" begin
    include("distrs_tests.jl")
    include("shape_models_tests.jl")
end
