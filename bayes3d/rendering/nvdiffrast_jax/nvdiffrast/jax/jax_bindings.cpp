#include "torch_types.h"
#include "jax_binding_ops.h"
#include "jax_rasterize_gl.h"
#include "jax_interpolate.h"
#include <tuple>
#include <pybind11/pybind11.h>

//------------------------------------------------------------------------
// Op prototypes.

void jax_rasterize_fwd_gl(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_rasterize_bwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_interpolate_fwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

void jax_interpolate_bwd(cudaStream_t stream,
                      void **buffers,
                      const char *opaque, std::size_t opaque_len);

//---------------------------------------------------

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
  return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
  pybind11::dict dict;
  dict["jax_rasterize_fwd_gl"] = EncapsulateFunction(jax_rasterize_fwd_gl);
  dict["jax_interpolate_fwd"] = EncapsulateFunction(jax_interpolate_fwd);
  dict["jax_rasterize_bwd"] = EncapsulateFunction(jax_rasterize_bwd);
  dict["jax_interpolate_bwd"] = EncapsulateFunction(jax_interpolate_bwd);
  return dict;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // State classes.
    pybind11::class_<RasterizeGLStateWrapper>(m, "RasterizeGLStateWrapper", py::module_local()).def(pybind11::init<bool, bool, int>())
        .def("set_context",     &RasterizeGLStateWrapper::setContext)
        .def("release_context", &RasterizeGLStateWrapper::releaseContext);

    // Ops.
    m.def("registrations", &Registrations, "custom call registrations");
    m.def("build_diff_rasterize_fwd_descriptor",
            [](RasterizeGLStateWrapper& stateWrapper,
            std::vector<int> images_vertices_triangles) {
            DiffRasterizeCustomCallDescriptor d;
            d.gl_state_wrapper = &stateWrapper;
            d.num_images = images_vertices_triangles[0];
            d.num_vertices = images_vertices_triangles[1];
            d.num_triangles = images_vertices_triangles[2];
            return PackDescriptor(d);
        });
    m.def("build_diff_interpolate_descriptor",
            [](std::vector<int> attr_shape,
            std::vector<int> rast_shape,
            std::vector<int> tri_shape, 
            int num_diff_attrs
            ) {
            DiffInterpolateCustomCallDescriptor d;
            d.num_images = attr_shape[0], 
            d.num_vertices = attr_shape[1],
            d.num_attributes = attr_shape[2], 
            d.rast_height = rast_shape[1],
            d.rast_width = rast_shape[2],
            d.rast_depth = rast_shape[0],
            d.num_triangles = tri_shape[0];
            d.num_diff_attributes = num_diff_attrs;
            return PackDescriptor(d);
        });
    m.def("build_diff_rasterize_bwd_descriptor",
            [](std::vector<int> pos_shape, std::vector<int> tri_shape, std::vector<int> rast_shape) {
            DiffRasterizeBwdCustomCallDescriptor d;
            d.num_images = pos_shape[0];
            d.num_vertices = pos_shape[1];
            d.num_triangles = tri_shape[0];
            d.rast_height = rast_shape[1];
            d.rast_width = rast_shape[2];
            d.rast_depth = rast_shape[0];
            return PackDescriptor(d);
        });
}


//------------------------------------------------------------------------
