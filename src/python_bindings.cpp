
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <countmin.hpp>

PYBIND11_MODULE(cuda_kmers, m) {
  m.doc() = "GPU k-mer counting";
  m.attr("version") = VERSION_INFO;

  // Results class (need to define here to be able to return this type)
  py::class_<kmer_counts, std::shared_ptr<kmer_counts>>(
      m, "kmer_result")
      .def(py::init<const std::vector<std::string>>())
      .def("histogram", &kmer_counts::histogram)
      .def("get_count", &kmer_counts::get_count, py::arg("kmer"));
}
