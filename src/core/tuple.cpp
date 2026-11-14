#ifndef QT_CLUSTERING_CORE_TUPLE_CPP
#define QT_CLUSTERING_CORE_TUPLE_CPP

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "items.cpp"

namespace qt {

class QTuple {
public:
    using ItemHolder = std::variant<ContinuousItem, DiscreteItem>;

    QTuple() = default;

    explicit QTuple(std::vector<ItemHolder> items)
        : items_(std::move(items)) {}

    QTuple(std::initializer_list<ItemHolder> items) : items_(items) {}

    std::size_t get_length() const noexcept { return items_.size(); }

    Item& get(std::size_t index) {
        return std::visit([](auto& item) -> Item& { return item; },
                          items_.at(index));
    }

    const Item& get(std::size_t index) const {
        return std::visit([](const auto& item) -> const Item& { return item; },
                          items_.at(index));
    }

    Item& operator[](std::size_t index) { return get(index); }

    const Item& operator[](std::size_t index) const { return get(index); }

    const std::vector<ItemHolder>& items() const noexcept { return items_; }

    void add(ItemHolder item) { items_.push_back(std::move(item)); }

    double get_distance(const QTuple& other) const {
        if (get_length() != other.get_length()) {
            throw std::invalid_argument(
                "Cannot compute distance between tuples of different lengths: " +
                std::to_string(get_length()) + " vs " +
                std::to_string(other.get_length()));
        }
        double total = 0.0;
        for (std::size_t i = 0; i < items_.size(); ++i) {
            total += get(i).distance(other.get(i).value());
        }
        return total;
    }

    template <class DataType, class IdContainer>
    double avg_distance(const DataType& data,
                        const IdContainer& clustered_ids) const {
        if (clustered_ids.empty()) {
            return 0.0;
        }
        double total = 0.0;
        for (const auto& idx : clustered_ids) {
            total += get_distance(data.get_item_set(static_cast<int>(idx)));
        }
        return total / static_cast<double>(clustered_ids.size());
    }

    bool equals(const QTuple& other) const {
        if (get_length() != other.get_length()) {
            return false;
        }
        for (std::size_t i = 0; i < items_.size(); ++i) {
            if (!get(i).equals(other.get(i))) {
                return false;
            }
        }
        return true;
    }

private:
    std::vector<ItemHolder> items_;
};

inline bool operator==(const QTuple& a, const QTuple& b) { return a.equals(b); }

inline bool operator!=(const QTuple& a, const QTuple& b) { return !(a == b); }

}

#endif
