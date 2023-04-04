module ObjectLearning

using Gen
using PyCall
import Gen: random, logpdf
using PoseComposition: IDENTITY_POSE, Pose, IDENTITY_ORN
using Rotations: RotX, RotMatrix

const j = PyNULL()
const jax = PyNULL()
const jnp = PyNULL()
const np = PyNULL()
const trimesh = PyNULL()

function __init__()
    copy!(j, pyimport("jax3dp3"))
    copy!(jax, pyimport("jax"))
    copy!(jnp, pyimport("jax.numpy"))
    copy!(np, pyimport("numpy"))
    copy!(trimesh, pyimport("trimesh"))
end

#############
# Constants #
#############

# XXX make these agree with the real model
P_OUTLIER_SHAPE = 1.0
P_OUTLIER_SCALE = 0.01
NOISE_SHAPE = 1.0
NOISE_SCALE = 0.01

VOXEL_GRID_SIZE = (50, 50, 50)
SCALE = 1.0

UNKNOWN_SHAPE_ALPHA = 0.1
UNKNOWN_SHAPE_BETA = 0.1

OBS_CLOUD_SIZE = 3000
OUTLIER_ΔX, OUTLIER_ΔY, OUTLIER_ΔZ = 10, 10, 10

include("utils.jl")
include("crp.jl")
include("meshes.jl")
include("shape_models.jl")
include("sg.jl")
include("distrs.jl")
include("jax_renderer.jl")
include("stoch_renderer.jl")

KNOWN_SHAPE_PARAMS = [ShapeModelParams(Array{Real,3}(undef, 1, 1, 1)),
                      ShapeModelParams(Array{Real,3}(undef, 1, 1, 1)),
                      ShapeModelParams(Array{Real,3}(undef, 1, 1, 1))]

TABLE = ShapeModelParams(Array{Real,3}(undef, 1, 1, 1))
TABLE_POSE = IDENTITY_POSE

Δx = 0.15
Δy = 0.15


include("model.jl")

end # module ObjectLearning
