@testset "Shape Models" begin

    import ObjectLearning: is_valid_crp_state, get_crp_choices

    
    @testset "CRP Marginalization" begin
        @test is_valid_crp_state([])
        @test is_valid_crp_state([1])
        @test !is_valid_crp_state([2])
        @test is_valid_crp_state([1, 1, 2, 2])
        @test !is_valid_crp_state([1, 1, 2, 4])
        @test is_valid_crp_state(1:5)

        # Bell numbers
        @test length(get_crp_choices(3, [], 1)) == 5
        @test length(get_crp_choices(4, [], 1)) == 15
        @test length(get_crp_choices(5, [], 1)) == 52
        @test length(get_crp_choices(6, [], 1)) == 203

        θ = 2
        crp3 = get_crp_choices(3, [], θ)
        log_prob3 = Dict([1, 1, 1] => log(1 * 1/(1+θ) * 2/(2 + θ)),
                         [1, 1, 2] => log(1 * 1/(1+θ) * θ/(2 + θ)),
                         [1, 2, 1] => log(1 * θ/(1+θ) * 1/(2 + θ)),
                         [1, 2, 2] => log(1 * θ/(1+θ) * 1/(2 + θ)),
                         [1, 2, 3] => log(1 * θ/(1+θ) * θ/(2 + θ)))
        @test all(s ≈ log_prob3[c] for (c, s) in crp3)

        θ = 2
        crp5 = get_crp_choices(5, 1:3, θ)
        function log_prob5(i, j)
            res = 0
            counts = [1, 1, 1, 0, 0]
            for (seat, t) in ((i, 3), (j, 4))
                res += log((counts[seat] == 0 ? θ : counts[seat])/(t + θ))
                counts[seat] += 1
            end
            res
        end
        @test length(crp5) == 17
        @test all(s ≈ log_prob5(c[4], c[5]) for (c, s) in crp5)

        # XXX make sure the weights are correct
    end
end
