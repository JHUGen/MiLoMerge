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
        .def(
            "split", &bin_splitter::split,
            py::arg("n_bins_desired"), py::arg("granularity"), py::arg("stat_limit"), py::arg("log")=true
        )
        .def_property_readonly("data", &bin_splitter::getData)
        .def_property_readonly("finalBinCounts", &bin_splitter::getFinalBinCounts)
        .def_property_readonly("encodedStrings", &bin_splitter::getEncodedStrings)
        .def_property_readonly("minimaAndMaxima", &bin_splitter::getMinimaAndMaxima)
        .def_property_readonly("minima", &bin_splitter::getMinima)
        .def_property_readonly("maxima", &bin_splitter::getMaxima)
        .def_property_readonly("nObservables", &bin_splitter::getNObservables)
        .def_property_readonly("nPoints", &bin_splitter::getNPoints)
        .def_property_readonly("nHypotheses", &bin_splitter::getNHypotheses)
        .def("get_decoded_cuts", &bin_splitter::decodeCuts)
        .def("reset", &bin_splitter::reset)
        .def(
            "__repr__", [](const bin_splitter &b){
                return "Splitter of dimension " + std::to_string(b.getNHypotheses()) \
                    + " x " + std::to_string(b.getNObservables()) \
                    + " x " + std::to_string(b.getNPoints());
            }
        );
    m.def("decode", &bin_splitter::decode);
}
