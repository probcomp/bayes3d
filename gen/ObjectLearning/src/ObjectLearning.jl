module ObjectLearning

using Gen
import Gen: random, logpdf
import PoseComposition: IDENTITY_POSE, Pose, IDENTITY_ORN
using Rotations: RotX

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

include("utils.jl")
include("crp.jl")
include("shape_models.jl")
include("sg.jl")
include("distrs.jl")
include("stoch_renderer.jl")

KNOWN_SHAPE_PARAMS = [ShapeModelParams((1, 1, 1), 1, Array{Real,3}(undef, 0, 0, 0)),
                      ShapeModelParams((1, 1, 1), 2, Array{Real,3}(undef, 0, 0, 0)),
                      ShapeModelParams((1, 1, 1), 3, Array{Real,3}(undef, 0, 0, 0))]

TABLE = ShapeModelParams((1, 1, 1), 0, Array{Real,3}(undef, 0, 0, 0))
TABLE_POSE = IDENTITY_POSE

Δx = 0.15
Δy = 0.15


include("scene_model.jl")

end # module ObjectLearning
