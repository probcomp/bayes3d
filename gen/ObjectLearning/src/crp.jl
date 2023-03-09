
"""
the partition map encodes the state of the Chinese restaurant at a given point
in time. If at time `t ∈ ℕ` there are `m ∈ ℕ` occupied tables at the restaurant,
the partition map is a surjective mapping `f:{1, ..., t} → {1, ..., m}`, such
that `f(i)` denotes the table of the `i`-th customer.
"""
struct CRPState
    partition_map::Vector{Int}
    num_tables::Int
end

empty_crp_state = CRPState([], 0)

table(s::CRPState, custidx::Int) = s.partition_map[custidx]

num_tables(s::CRPState) = s.num_tables

num_customers(s::CRPState) = length(s.partition_map)

"""new customer arrives and sits at table i"""
update(s::CRPState, i::Int) = 
    let nt = num_tables(s);
        δ = i <= nt ? 0 : i == nt + 1 ? 1 : error("invalid table number")
        CRPState([s.partition_map; i], nt+δ)
    end

function table_counts(s::CRPState)
    counts = zeros(Int, num_tables(s))
    for i=1:num_customers(s)
        counts[table(s, i)] += 1
    end
    counts
end

function isvalid(s::CRPState)
    num_tables_so_far = 0
    for i=1:num_customers(s)
        if !(1 <= table(s, i) <= num_tables_so_far + 1)
            return false
        end
        num_tables_so_far += table(s, i) == num_tables_so_far + 1 ? 1 : 0
    end
    true
end

"""state of the restaurant after `t` customers have arrived"""
substate(s::CRPState, t::Int) =
    t <= num_customers(s) ? CRPState(s.partition_map, maximum(s.partition_map)) :
    error("invalid input")

struct ConditionalCRP <: Distribution{CRPState} end
cond_crp = ConditionalCRP()

struct CRP <: Distribution{CRPState} end
crp = CRP()

function random(::ConditionalCRP, s0::CRPState, num_custs::Int, θ::Float64)
    s = isvalid(s0) ? s0 : error("invialid initial state")
    num_new_custs = num_custs - num_customers(s0)
    for i=1:num_new_custs
        s = update(s, categorical(normalize([table_counts(s); θ])))
    end
    s
end

random(::CRP, num_custs::Int, θ::Float64) =
    random(cond_crp, empty(CRPState), num_custs, θ)

function logpdf(::CRP, s::CRPState, num_custs::Int, θ::Float64)
    table_cnts_so_far = []
    num_tables_so_far, res = 0, 0
    if !isvalid(s) || num_customers(s) != num_custs 
        return -Inf
    end
    for i=1:num_custs
        if table(s, i) == num_tables_so_far + 1
            num_tables_so_far += 1
            push!(table_cnts_so_far, 1)
            res += log(θ/(i - 1 + θ))
        else
            res += log(table_cnts_so_far[i]/(i - 1 + θ))
            table_cnts_so_far[i] += 1
        end
    end
    res
end

logpdf(::ConditionalCRP, s::CRPState, s0::CRPState, num_custs::Int, θ::Float64) = 
    substate(s, num_customers(s0)) != s0 ?  -Inf : 
    (logpdf(crp, s, num_custs, θ) - logpdf(crp, s0, num_customers(s0), θ))


# XXX  XXX  XXX  XXX  XXX  XXX 


"""
θ: the theta parameter of the CRP
cur_part: the state of the restaurant at the current moment: `cur_part[i] == j`
means that the `i`-th customer is sitting at the `j`-th table
n: the target number of customers

return a list of [(state, weight)], where each `state` has length `n`, and if `p`
is the pmf of the CRP with parameter `θ`, then
log p(state) = log p(cur_part) + w
"""
function get_crp_choices(n, cur_part, θ)
    if length(cur_part) == n
        return [(cur_part, 0)]
    end
    #@assert is_valid_crp_state(cur_part)
    num_tables = maximum(cur_part; init=0)
    num_people = length(cur_part)
    p = [count(==(i), cur_part)/(num_people + θ) for i=1:num_tables]
    q = θ/(num_people + θ) # XXX
    head = [(s, w + log(q))
            for (s, w) in get_crp_choices(n, [cur_part; num_tables + 1], θ)]
    tail = [(s, log(p[i]) + w)
            for i=1:num_tables for (s, w) in get_crp_choices(n, [cur_part; i], θ)]
    [head; tail]
end


