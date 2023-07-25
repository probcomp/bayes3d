# TODO: still needs to be integrated into the actual tracking pipeline


import jax.numpy as jnp
from bayes3d.transforms_3d import transform_from_pos


class BoundingBox:
    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def intersects(self, other_bbox, threshold=0.03):
        # check if this bounding box intersects with another bounding box
        return (
            self.xmin <= other_bbox.xmax
            and self.xmax >= other_bbox.xmin
            and self.ymin <= other_bbox.ymax
            and self.ymax >= other_bbox.ymin
            and self.zmin <= other_bbox.zmax
            and self.zmax >= other_bbox.zmin
        )

    def is_contained_in(self, other_bbox, threshold=0.03):
        # check if this bounding box is contained within another bounding box
        return jnp.all(other_bbox.maxs > (self.maxs - threshold)) and jnp.all(
            other_bbox.mins < (self.mins + threshold)
        )
    
    def move(self, vector):
        self.mins += vector
        self.maxs += vector

class SceneObject:
    def __init__(self, mesh_name, bbox, transform):
        self.name = mesh_name
        self.bbox = bbox
        self.transform = transform

    def translate(self, translation):
        translation_mat = jnp.eye(4)
        translation_mat[:3, 3] = translation
        self.transform = jnp.dot(translation_mat, self.transform)
        self.bbox.translate(translation)

    @classmethod
    def from_points(name: str, object_points: jnp.ndarray):
        maxs = jnp.max(object_points, axis=0)
        mins = jnp.min(object_points, axis=0)
        obj_center = (maxs + mins) / 2
        obj_transform = transform_from_pos(obj_center)
        obj_bbox = BoundingBox(mins, maxs)
        return SceneObject(name, obj_bbox, obj_transform)


class SceneGraph:
    def __init__(self):
        self.objects = []
        self.edges = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_edge(self, obj1_idx, obj2_idx, relationship):
        self.edges.append((obj1_idx, obj2_idx, relationship))

    def infer_relationships(self):
        for i, obj1 in enumerate(self.objects):
            for j, obj2 in enumerate(self.objects):
                if i == j:
                    continue
                if obj1.bbox.is_contained_in(obj2.bbox):
                    self.add_edge(i, j, "containment")

    def apply_transform(self, i, transform):
        self.objects[i].transform = jnp.dot(transform, self.objects[i].transform)

    def update_positions(self, i, transform):
        # Apply the given transformation to object i
        self.objects[i].transform = jnp.dot(transform, self.objects[i].transform)
        self.objects[i].update_position_and_orientation()

        # Recursively update the positions of objects connected to object i through an edge
        for edge in self.edges:
            if edge[0] == i:
                if edge[2] == "containment":
                    containing_obj = self.objects[edge[1]]
                    self.update_positions(
                        edge[1],
                        jnp.dot(
                            containing_obj.transform,
                            jnp.linalg.inv(self.objects[i].transform),
                        ),
                    )
