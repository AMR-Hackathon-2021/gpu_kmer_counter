
#include <pybind11/stl.h>
namespace py = pybind11;

#include "countmin.hpp"

PYBIND11_MODULE(cuda_kmers, m) {
  m.doc() = "GPU k-mer counting";
//  m.attr("version") = VERSION_INFO;

  // Results class (need to define here to be able to return this type)
  py::class_<CountMin, std::shared_ptr<CountMin>>(
      m, "count_min_table")
      .def(py::init<const std::vector<std::string>, const size_t,
           const size_t, const size_t, const size_t,
           const size_t, const int, const int, const bool, const int,
           const int>())
      .def("histogram", &CountMin::histogram)
      .def("get_count", &CountMin::get_count);
}