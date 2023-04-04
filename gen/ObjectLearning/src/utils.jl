normalize(v::Vector{Float64}) = isempty(v) ? v : v ./ sum(v)

homcoords(p::Pose) = [p.orientation p.pos; 0 0 0 1]

homcoords_to_pose(h::Matrix{<:Real}) = Pose(h[1:3, 4], RotMatrix{3}(h[1:3, 1:3]))
    
cam_pose_from_pos_target_up(pos, target, up) = 
    homcoords_to_pose(np.array(j.t3d.transform_from_pos_target_up(pos, target, up)));