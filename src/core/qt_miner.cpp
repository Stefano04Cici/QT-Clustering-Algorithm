#ifndef QT_CLUSTERING_CORE_QT_MINER_CPP
#define QT_CLUSTERING_CORE_QT_MINER_CPP

#include <chrono>
#include <cstddef>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cluster_set.cpp"

namespace qt {

class EmptyDatasetException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ClusteringRadiusException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class QTMiner {
public:
    explicit QTMiner(double radius) : radius_(radius) {
        if (radius <= 0.0) {
            throw std::invalid_argument("radius must be > 0");
        }
    }

    double radius() const noexcept { return radius_; }

    void set_radius(double radius) {
        if (radius <= 0.0) {
            throw std::invalid_argument("radius must be > 0");
        }
        radius_ = radius;
    }

    const ClusterSet& cluster_set() const noexcept { return cluster_set_; }

    ClusterSet& cluster_set() noexcept { return cluster_set_; }

    double elapsed_ms() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return elapsed_ms_;
    }

    std::size_t compute(const Data& data) {
        if (data.get_number_of_examples() == 0) {
            throw EmptyDatasetException("Dataset is empty");
        }

        const auto start = std::chrono::steady_clock::now();

        const std::size_t n = data.get_number_of_examples();
        std::vector<QTuple> item_sets;
        item_sets.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            item_sets.push_back(data.get_item_set(i));
        }

        std::vector<bool> is_clustered(n, false);
        std::size_t clustered_count = 0;

        while (clustered_count < n) {
            Cluster candidate = build_candidate_cluster(item_sets, is_clustered);
            cluster_set_.add(candidate);
            for (const int idx : candidate) {
                const auto position = static_cast<std::size_t>(idx);
                if (!is_clustered.at(position)) {
                    is_clustered[position] = true;
                    ++clustered_count;
                }
            }
        }

        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            elapsed_ms_ = elapsed_ms;
        }

        return cluster_set_.get_number_of_clusters();
    }

private:
    Cluster build_candidate_cluster(
        const std::vector<QTuple>& item_sets,
        const std::vector<bool>& is_clustered) const {
        const std::size_t n = item_sets.size();
        std::optional<Cluster> best_cluster;

        for (std::size_t i = 0; i < n; ++i) {
            if (is_clustered[i]) {
                continue;
            }

            const QTuple& centroid = item_sets[i];
            Cluster candidate{centroid};

            for (std::size_t j = 0; j < n; ++j) {
                if (is_clustered[j] || i == j) {
                    continue;
                }
                if (centroid.get_distance(item_sets[j]) <= radius_) {
                    candidate.add_data(static_cast<int>(j));
                }
            }

            candidate.add_data(static_cast<int>(i));

            if (!best_cluster.has_value() ||
                candidate.get_size() > best_cluster->get_size()) {
                best_cluster = candidate;
            }
        }

        if (!best_cluster.has_value()) {
            throw std::logic_error("no candidate cluster found");
        }
        return std::move(*best_cluster);
    }

    double radius_;
    ClusterSet cluster_set_;
    mutable std::mutex mutex_;
    double elapsed_ms_ = 0.0;
};

}

#endif
