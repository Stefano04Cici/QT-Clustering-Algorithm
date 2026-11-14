#ifndef QT_CLUSTERING_CORE_ITEMS_CPP
#define QT_CLUSTERING_CORE_ITEMS_CPP

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include "attributes.cpp"

namespace qt {

class Item {
public:
    using Value = std::variant<double, std::string>;

    Item(const Attribute& attribute, Value value)
        : attribute_(&attribute), value_(std::move(value)) {}

    virtual ~Item() = default;

    Item(const Item&) = default;
    Item(Item&&) noexcept = default;
    Item& operator=(const Item&) = default;
    Item& operator=(Item&&) noexcept = default;

    virtual const Attribute& attribute() const noexcept { return *attribute_; }

    const Value& value() const noexcept { return value_; }

    virtual bool is_continuous() const noexcept = 0;

    virtual double distance(const Value& other_value) const = 0;

    virtual bool equals(const Item& other) const = 0;

protected:
    const Attribute* attribute_;
    Value value_;
};

inline bool operator==(const Item& a, const Item& b) { return a.equals(b); }

inline bool operator!=(const Item& a, const Item& b) { return !(a == b); }

class ContinuousItem final : public Item {
public:
    ContinuousItem(const ContinuousAttribute& attribute, double value = 0.0)
        : Item(attribute, value) {}

    const ContinuousAttribute& attribute() const noexcept override {
        return static_cast<const ContinuousAttribute&>(Item::attribute());
    }

    double value() const noexcept { return std::get<double>(value_); }

    double distance(const Value& other_value) const override {
        const auto* other = std::get_if<double>(&other_value);
        if (other == nullptr) {
            throw std::invalid_argument(
                "ContinuousItem::distance() expects a numeric value");
        }
        const auto& attr = attribute();
        return std::abs(attr.get_scaled_value(value()) -
                        attr.get_scaled_value(*other));
    }

    bool is_continuous() const noexcept override { return true; }

    bool equals(const Item& other) const override {
        const auto* o = dynamic_cast<const ContinuousItem*>(&other);
        return o != nullptr && attribute() == o->attribute() &&
               value() == o->value();
    }
};

class DiscreteItem final : public Item {
public:
    DiscreteItem(const DiscreteAttribute& attribute, std::string value = {})
        : Item(attribute, std::move(value)) {}

    const DiscreteAttribute& attribute() const noexcept override {
        return static_cast<const DiscreteAttribute&>(Item::attribute());
    }

    const std::string& value() const noexcept {
        return std::get<std::string>(value_);
    }

    double distance(const Value& other_value) const override {
        const auto* other = std::get_if<std::string>(&other_value);
        return (other != nullptr && value() == *other) ? 0.0 : 1.0;
    }

    bool is_continuous() const noexcept override { return false; }

    bool equals(const Item& other) const override {
        const auto* o = dynamic_cast<const DiscreteItem*>(&other);
        return o != nullptr && attribute() == o->attribute() &&
               value() == o->value();
    }
};

}

#endif
