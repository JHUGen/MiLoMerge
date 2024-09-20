#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/eigen.h"
#include "bin_splitter.hpp"
namespace py = pybind11;


PYBIND11_MAKE_OPAQUE(bin_splitter)
PYBIND11_MODULE(bin_splitter, m){
    py::class_<bin_splitter>(m, "bin_splitter")
        .def(py::init<std::vector<std::vector<std::vector<double>>>&>())
        .def(py::init<std::vector<std::vector<std::vector<double>>>&, std::vector<double>&>())
        .def(py::init<std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&>())
        .def("split", &bin_splitter::split)
        .def("getData", &bin_splitter::getData)
        .def("getFinalBinCounts", &bin_splitter::getFinalBinCounts)
        .def("getEncodedStrings", &bin_splitter::getEncodedStrings)
        .def("getMinimaAndMaxima", &bin_splitter::getMinimaAndMaxima)
        .def("getMinima", &bin_splitter::getMinima)
        .def("getMaxima", &bin_splitter::getMaxima)
        .def("reset", &bin_splitter::reset);
}
