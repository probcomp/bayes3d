import jax
import jax.numpy as jnp

def separating_axis_test(axis, box1, box2):
    """
    Projects both boxes onto the given axis and checks for overlap.
    """
    min1, max1 = project_box(axis, box1)
    min2, max2 = project_box(axis, box2)

    return jax.lax.cond(jnp.logical_or(max1 < min2, max2 < min1), lambda: False, lambda: True)

    # if max1 < min2 or max2 < min1:
    #     return False
    # return True

def project_box(axis, box):
    """
    Projects a box onto an axis and returns the min and max projection values.
    """
    corners = get_transformed_box_corners(box)
    projections = jnp.array([jnp.dot(corner, axis) for corner in corners])
    return jnp.min(projections), jnp.max(projections)

def get_transformed_box_corners(box):
    """
    Returns the 8 corners of the box based on its dimensions and pose.
    """
    dim, pose = box
    corners = []
    for dx in [-dim[0]/2, dim[0]/2]:
        for dy in [-dim[1]/2, dim[1]/2]:
            for dz in [-dim[2]/2, dim[2]/2]:
                corner = jnp.array([dx, dy, dz, 1])
                transformed_corner = pose @ corner
                corners.append(transformed_corner[:3])
    return corners

def are_bboxes_intersecting(dim1, dim2, pose1, pose2):
    """
    Checks if two oriented bounding boxes (OBBs), which are AABBs with poses, are intersecting using the Separating 
    Axis Theorem (SAT).

    Args:
        dim1 (jnp.ndarray): Bounding box dimensions of first object. Shape (3,)
        dim2 (jnp.ndarray): Bounding box dimensions of second object. Shape (3,)
        pose1 (jnp.ndarray): Pose of first object. Shape (4,4)
        pose2 (jnp.ndarray): Pose of second object. Shape (4,4)
    Output:
        Bool: Returns true if bboxes intersect
    """
    box1 = (dim1, pose1)
    box2 = (dim2, pose2)

    # Axes to test - the face normals of each box
    axes_to_test = []
    for i in range(3):  # Add the face normals of box1
        axes_to_test.append(pose1[:3, i])
    for i in range(3):  # Add the face normals of box2
        axes_to_test.append(pose2[:3, i])

    # Perform SAT on each axis
    count_ = 0
    for axis in axes_to_test:
        count_+= jax.lax.cond(separating_axis_test(axis, box1, box2), lambda:0,lambda:-1)

    return jax.lax.cond(count_ < 0, lambda:False,lambda:True)

# For one reference pose (object 1) and many possible poses for the second object
are_bboxes_intersecting_many = jax.vmap(are_bboxes_intersecting, in_axes = (None, None, None, 0))
