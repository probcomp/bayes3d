###############
# Shape Model #
###############

# shape = random(shape_model, shape_model_params) where
# shape::VoxelizedMesh
# shape_model::ShapeModel <: Distribution{VoxelizedMesh}
# shape_model_params::ShapeModelParams

struct VoxelizedMesh end

struct ShapeModelParams 
    size::Tuple{Int,Int,Int} # w, l, h
    scale::Real
    thetas::Array{Real,3}
end

struct ShapeModel <: Distribution{VoxelizedMesh} end
shape_model = ShapeModel()

function random(::ShapeModel, params::ShapeModelParams)
    VoxelizedMesh() # XXX TODO
end

function logpdf(::ShapeModel, shape::VoxelizedMesh, params::ShapeModelParams)
    0 # XXX TODO
end

#######################
# Unknown Shape Prior #
#######################

struct UnknownShapePrior <: Distribution{ShapeModelParams} end
unknown_shape_prior = UnknownShapePrior()

function random(::UnknownShapePrior) 
    thetas = [beta(UNKNOWN_SHAPE_ALPHA, UNKNOWN_SHAPE_BETA)
              for _=1:prod(VOXEL_GRID_SIZE)]
    ShapeModelParams(VOXEL_GRID_SIZE, SCALE, reshape(thetas, VOXEL_GRID_SIZE))
end

logpdf(::UnknownShapePrior, shape_params::ShapeModelParams) =
    sum(θ->logpdf(beta, θ, UNKNOWN_SHAPE_ALPHA, UNKNOWN_SHAPE_BETA), shape_params.thetas)

#########################
# Shapes in Scene Prior #
#########################

struct ShapesPrior <: Distribution{Vector{ShapeModelParams}} end
shapes_prior = ShapesPrior()

function random(::ShapesPrior, num_objs::Int, assignments::CRPState)
    m = length(KNOWN_SHAPE_PARAMS)
    map(i->(i <= m ? KNOWN_SHAPE_PARAMS[i] : random(unknown_shape_prior)),
        map(i->table(assignments, i), m+1:m+num_objs))
end

function logpdf(::ShapesPrior, shape_params::Vector{ShapeModelParams},
        num_objs::Int, assignments::CRPState)
    m = length(KNOWN_SHAPE_PARAMS)
    sum(map(i->logpdf(unknown_shape_prior, shape_params[i-m]),
            filter(i->(table(assignments, i) > m), m+1:m+num_objs)); init=0.0)
end

