#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "razer_gemm.h"

namespace py = pybind11;

static int _parse_impl(py::object impl_obj) {
      if (py::isinstance<py::int_>(impl_obj)) {
            return impl_obj.cast<int>();
      }
      if (py::isinstance<py::str>(impl_obj)) {
            const std::string s = impl_obj.cast<std::string>();
            if (s == "auto") return (int)RazerImpl::Auto;
            if (s == "gemv") return (int)RazerImpl::Gemv;
            if (s == "small") return (int)RazerImpl::Small;
      }
      throw std::runtime_error("impl must be an int or one of: auto, gemv, small");
}

static const char* _impl_name(int impl) {
      switch ((RazerImpl)impl) {
            case RazerImpl::Auto: return "auto";
            case RazerImpl::Gemv: return "gemv";
            case RazerImpl::Small: return "small";
            default: return "unknown";
      }
}

PYBIND11_MODULE(razer_ext, m) {
      m.def(
            "razer_gemm",
            [](at::Tensor fA,
                   at::Tensor qB,
                   at::Tensor scaling_factors,
                   at::Tensor out,
                   int groupsize,
                   py::object impl,
                   int split_k,
                   int gemv_g) {
                  const int impl_i = _parse_impl(impl);
                  const RazerGemmLaunchConfig cfg = razer_gemm(
                        fA, qB, scaling_factors, out,
                        groupsize, impl_i, split_k, gemv_g
                  );
                  py::dict d;
                  d["impl"] = std::string(_impl_name(cfg.impl));
                  d["split_k"] = cfg.split_k;
                  d["gemv_g"] = cfg.gemv_g;
                  d["small_r"] = cfg.small_r;
                  return d;
            },
            py::arg("fA"),
            py::arg("qB"),
            py::arg("scaling_factors"),
            py::arg("out"),
            py::arg("groupsize") = 128,
            py::arg("impl") = py::str("auto"),
            py::arg("split_k") = -1,
            py::arg("gemv_g") = -1,
            "RaZeR GEMM with optional implementation/launch overrides. Returns chosen config as a dict."
      );

  m.def("razer_dequant", &razer_dequant,
        py::arg("qB"),
        py::arg("scaling_factors"),
        py::arg("K"),
        py::arg("groupsize") = 128,
        "RaZeR dequant: (qB, scale) -> dense B (fp16)");
}