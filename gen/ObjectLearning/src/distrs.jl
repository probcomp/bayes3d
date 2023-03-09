abstract type PointMass{T} <: Distribution{T} end

random(::PointMass{T}, support::T) where {T} = support
logpdf(::PointMass{T}, x::T, support::T) where {T} = x == support ? 0 : -Inf

delta(x::T) where {T} = random(PointMass{T}(), x)

struct UniformContactPlane <: Distribution{ContactPlane} end
uniform_contact_plane = UniformContactPlane()

random(::UniformContactPlane) = ContactPlane(uniform_discrete(0, 5))
logpdf(::UniformContactPlane, p::ContactPlane) = -log(6)

struct RelativePosePrior <: Distribution{Pose} end
relative_pose_prior = RelativePosePrior()

random(::RelativePosePrior, cluster_center, dx, dy) =
    Pose(uniform(0, dx), uniform(0, dy), 0, IDENTITY_ORN)

logpdf(::RelativePosePrior, p::Pose, cluster_center, dx, dy) =
    p.pos[3] != 0 || p.orientation != IDENTITY_ORN ? -Inf : 1/(dx * dy)

# XXX stubbed
struct CamPosePrior <: PointMass{Pose} end
cam_pose_prior = CamPosePrior()

random(::CamPosePrior) =
    Pose(0, 0, uniform(1, 4), RotX(π))

logpdf(::CamPosePrior, _::Pose) = 1/3
