#ifndef QT_CLUSTERING_CORE_CLUSTER_CPP
#define QT_CLUSTERING_CORE_CLUSTER_CPP

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "data.cpp"

namespace qt {

class Cluster {
public:
    using IdSet = std::set<int>;
    using const_iterator = IdSet::const_iterator;

    explicit Cluster(QTuple centroid, IdSet clustered_data = {})
        : centroid_(std::move(centroid)),
          clustered_data_(std::move(clustered_data)) {}

    const QTuple& centroid() const noexcept { return centroid_; }

    QTuple& centroid() noexcept { return centroid_; }

    const IdSet& clustered_data() const noexcept { return clustered_data_; }

    IdSet& clustered_data() noexcept { return clustered_data_; }

    bool add_data(int id) {
        return clustered_data_.insert(id).second;
    }

    bool contains(int id) const {
        return clustered_data_.count(id) != 0;
    }

    std::size_t get_size() const noexcept { return clustered_data_.size(); }

    const_iterator begin() const noexcept { return clustered_data_.begin(); }

    const_iterator end() const noexcept { return clustered_data_.end(); }

    bool operator<(const Cluster& other) const {
        if (get_size() != other.get_size()) {
            return get_size() < other.get_size();
        }
        const std::size_t length =
            std::min(centroid_.get_length(), other.centroid_.get_length());
        for (std::size_t i = 0; i < length; ++i) {
            const Item::Value& v1 = centroid_.get(i).value();
            const Item::Value& v2 = other.centroid_.get(i).value();
            if (v1 == v2) {
                continue;
            }
            const std::string s1 = value_to_string(v1);
            const std::string s2 = value_to_string(v2);
            double d1 = 0.0;
            double d2 = 0.0;
            if (try_parse_double(s1, d1) && try_parse_double(s2, d2)) {
                return d1 < d2;
            }
            return s1 < s2;
        }
        return centroid_.get_length() < other.centroid_.get_length();
    }

    std::string str() const {
        std::string centroid_values;
        for (std::size_t i = 0; i < centroid_.get_length(); ++i) {
            if (i != 0) {
                centroid_values += ", ";
            }
            centroid_values += centroid_.get(i).attribute().name() + "=" +
                               value_to_string(centroid_.get(i).value());
        }
        return "Cluster(size=" + std::to_string(get_size()) +
               ", centroid={" + centroid_values + "})";
    }

    std::string to_string(const Data& data) const {
        std::string output = str();

        for (const int idx : clustered_data_) {
            const auto& example =
                data.examples().at(static_cast<std::size_t>(idx));
            const double distance = centroid_.get_distance(
                data.get_item_set(static_cast<std::size_t>(idx)));
            output += "\n  Example " + std::to_string(idx) + ": {";
            for (std::size_t i = 0; i < data.attributes().size(); ++i) {
                if (i != 0) {
                    output += ", ";
                }
                output += data.attributes()[i]->name() + "=" +
                          value_to_string(example.at(i));
            }
            output += "} distance=" + format_fixed(distance);
        }

        const double average = centroid_.avg_distance(data, clustered_data_);
        output += "\n  Average distance: " + format_fixed(average);
        return output;
    }

    bool equals(const Cluster& other) const {
        return centroid_ == other.centroid_ &&
               clustered_data_ == other.clustered_data_;
    }

private:
    static std::string value_to_string(const Item::Value& value) {
        return std::visit(
            [](const auto& cell) -> std::string {
                using T = std::decay_t<decltype(cell)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return cell;
                } else {
                    std::ostringstream out;
                    out << cell;
                    std::string text = out.str();
                    if (text.find('.') == std::string::npos &&
                        text.find('e') == std::string::npos &&
                        text.find("inf") == std::string::npos &&
                        text.find("nan") == std::string::npos) {
                        text += ".0";
                    }
                    return text;
                }
            },
            value);
    }

    static bool try_parse_double(const std::string& text, double& out) {
        const std::size_t begin = text.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            return false;
        }
        const std::size_t end = text.find_last_not_of(" \t\r\n");
        const std::string trimmed = text.substr(begin, end - begin + 1);
        try {
            std::size_t consumed = 0;
            out = std::stod(trimmed, &consumed);
            return consumed == trimmed.size();
        } catch (const std::exception&) {
            return false;
        }
    }

    static std::string format_fixed(double value) {
        std::ostringstream out;
        out << std::fixed << std::setprecision(4) << value;
        return out.str();
    }

    QTuple centroid_;
    IdSet clustered_data_;
};

inline bool operator==(const Cluster& a, const Cluster& b) {
    return a.equals(b);
}

inline bool operator!=(const Cluster& a, const Cluster& b) {
    return !(a == b);
}

}

#endif
