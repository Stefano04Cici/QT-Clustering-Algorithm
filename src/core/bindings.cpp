#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "qt_miner.cpp"

namespace py = pybind11;

namespace {

qt::Data make_data(const py::object& source, const py::object& feature_names) {
    if (py::isinstance<py::str>(source) || py::hasattr(source, "__fspath__")) {
        const std::string path =
            py::str(py::module_::import("os").attr("fspath")(source))
                .cast<std::string>();
        return qt::Data::from_csv(path);
    }

    using DoubleArray =
        py::array_t<double, py::array::c_style | py::array::forcecast>;

    DoubleArray array = DoubleArray::ensure(source);
    if (!array) {
        throw py::type_error("data must be a CSV path or a 2-D numeric array");
    }
    if (array.ndim() != 2) {
        throw py::value_error("data must be 2-dimensional");
    }

    const auto info = array.request();
    const auto rows = static_cast<std::size_t>(info.shape[0]);
    const auto cols = static_cast<std::size_t>(info.shape[1]);
    const auto* values = static_cast<const double*>(info.ptr);

    std::vector<std::vector<double>> matrix(
        rows, std::vector<double>(cols, 0.0));
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            matrix[r][c] = values[r * cols + c];
        }
    }

    std::vector<std::string> names;
    if (!feature_names.is_none()) {
        names = feature_names.cast<std::vector<std::string>>();
    }
    return qt::Data::from_matrix(matrix, names);
}

py::array_t<int> labels_array(const std::vector<int>& labels) {
    py::array_t<int> result(static_cast<py::ssize_t>(labels.size()));
    int* out = result.mutable_data();
    std::copy(labels.begin(), labels.end(), out);
    return result;
}

class PyQTMiner {
public:
    explicit PyQTMiner(double radius) : miner_(radius) {}

    double radius() const noexcept { return miner_.radius(); }

    void set_radius(double radius) { miner_.set_radius(radius); }

    std::size_t cluster_count() const noexcept {
        return miner_.cluster_set().get_number_of_clusters();
    }

    double elapsed_ms() const { return miner_.elapsed_ms(); }

    std::vector<std::vector<int>> clusters() const {
        std::vector<std::vector<int>> result;
        for (const auto& cluster : miner_.cluster_set()) {
            result.emplace_back(cluster.begin(), cluster.end());
        }
        return result;
    }

    std::size_t compute(const py::object& source,
                        const py::object& feature_names) {
        datasets_.push_back(
            std::make_shared<qt::Data>(make_data(source, feature_names)));
        py::gil_scoped_release release;
        return miner_.compute(*datasets_.back());
    }

    py::array_t<int> labels(std::size_t n_samples) const {
        return labels_array(miner_.cluster_set().get_labels(n_samples));
    }

    double outlier_percentage() const {
        return miner_.cluster_set().outlier_percentage();
    }

    std::string str() const { return miner_.cluster_set().str(); }

private:
    std::vector<std::shared_ptr<qt::Data>> datasets_;
    qt::QTMiner miner_;
};

}

PYBIND11_MODULE(_qt_miner, m) {
    py::class_<PyQTMiner>(m, "QTMiner")
        .def(py::init<double>(), py::arg("radius"))
        .def_property("radius", &PyQTMiner::radius, &PyQTMiner::set_radius)
        .def_property_readonly("cluster_count", &PyQTMiner::cluster_count)
        .def_property_readonly("clusters", &PyQTMiner::clusters)
        .def_property_readonly("elapsed_ms", &PyQTMiner::elapsed_ms)
        .def("compute", &PyQTMiner::compute, py::arg("source"),
             py::arg("feature_names") = py::none())
        .def("labels", &PyQTMiner::labels, py::arg("n_samples"))
        .def("get_labels", &PyQTMiner::labels, py::arg("n_samples"))
        .def("get_number_of_clusters", &PyQTMiner::cluster_count)
        .def("outlier_percentage", &PyQTMiner::outlier_percentage)
        .def("__len__", &PyQTMiner::cluster_count)
        .def("__str__", &PyQTMiner::str)
        .def("__repr__", [](const PyQTMiner& self) {
            return "QTMiner(radius=" + std::to_string(self.radius()) +
                   ", clusters=" + std::to_string(self.cluster_count()) + ")";
        });
}
