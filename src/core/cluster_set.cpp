#ifndef QT_CLUSTERING_CORE_CLUSTER_SET_CPP
#define QT_CLUSTERING_CORE_CLUSTER_SET_CPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__has_include)
#  if __has_include("cluster.hpp")
#    include "cluster.hpp"
#  elif __has_include("cluster.h")
#    include "cluster.h"
#  elif __has_include("cluster.cpp")
#    include "cluster.cpp"
#  endif
#else
#  include "cluster.cpp"
#endif

namespace qt {

class ClusterSet {
public:
    using const_iterator = std::vector<Cluster>::const_iterator;

    ClusterSet() = default;

    explicit ClusterSet(std::vector<Cluster> clusters)
        : clusters_(std::move(clusters)) {}

    const std::vector<Cluster>& clusters() const noexcept { return clusters_; }

    std::vector<Cluster>& clusters() noexcept { return clusters_; }

    void add(Cluster cluster) {
        clusters_.push_back(std::move(cluster));
        std::sort(clusters_.begin(), clusters_.end());
    }

    const_iterator begin() const noexcept { return clusters_.begin(); }

    const_iterator end() const noexcept { return clusters_.end(); }

    std::size_t size() const noexcept { return clusters_.size(); }

    bool empty() const noexcept { return clusters_.empty(); }

    const Cluster& operator[](std::size_t index) const {
        return clusters_.at(index);
    }

    Cluster& operator[](std::size_t index) { return clusters_.at(index); }

    std::size_t get_number_of_clusters() const noexcept {
        return clusters_.size();
    }

    std::vector<int> get_labels(std::size_t n_samples) const {
        std::vector<int> labels(n_samples, -1);
        for (std::size_t cluster_idx = 0; cluster_idx < clusters_.size();
             ++cluster_idx) {
            for (const int sample_idx : clusters_[cluster_idx]) {
                labels.at(static_cast<std::size_t>(sample_idx)) =
                    static_cast<int>(cluster_idx);
            }
        }
        return labels;
    }

    double outlier_percentage() const {
        std::size_t total = 0;
        for (const auto& cluster : clusters_) {
            total += cluster.get_size();
        }
        if (total == 0) {
            return 0.0;
        }
        std::size_t outliers = 0;
        for (const auto& cluster : clusters_) {
            if (cluster.get_size() == 1) {
                ++outliers;
            }
        }
        return (static_cast<double>(outliers) / static_cast<double>(total)) *
               100.0;
    }

    std::string str() const {
        std::string output =
            "ClusterSet(" + std::to_string(get_number_of_clusters()) +
            " clusters):";
        for (std::size_t i = 0; i < clusters_.size(); ++i) {
            output += "\n  " + std::to_string(i + 1) + ". " + clusters_[i].str();
        }
        return output;
    }

    bool equals(const ClusterSet& other) const {
        return clusters_ == other.clusters_;
    }

private:
    std::vector<Cluster> clusters_;
};

inline bool operator==(const ClusterSet& a, const ClusterSet& b) {
    return a.equals(b);
}

inline bool operator!=(const ClusterSet& a, const ClusterSet& b) {
    return !(a == b);
}

}

#endif
