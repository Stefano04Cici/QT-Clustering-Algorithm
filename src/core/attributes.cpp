#ifndef QT_CLUSTERING_CORE_ATTRIBUTES_CPP
#define QT_CLUSTERING_CORE_ATTRIBUTES_CPP

#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace qt {

class Attribute {
public:
    Attribute(std::string name, int index)
        : name_(std::move(name)), index_(index) {}

    virtual ~Attribute() = default;

    Attribute(const Attribute&) = default;
    Attribute(Attribute&&) noexcept = default;
    Attribute& operator=(const Attribute&) = default;
    Attribute& operator=(Attribute&&) noexcept = default;

    const std::string& name() const noexcept { return name_; }
    int index() const noexcept { return index_; }

    virtual bool is_continuous() const noexcept = 0;

    virtual std::size_t num_values() const noexcept { return 0; }

    virtual double get_scaled_value(double) const {
        throw std::logic_error(
            "get_scaled_value() is only defined for ContinuousAttribute");
    }

    virtual bool equals(const Attribute& other) const noexcept = 0;

protected:
    std::string name_;
    int index_;
};

inline bool operator==(const Attribute& a, const Attribute& b) noexcept {
    return a.equals(b);
}

inline bool operator!=(const Attribute& a, const Attribute& b) noexcept {
    return !(a == b);
}

class ContinuousAttribute final : public Attribute {
public:
    ContinuousAttribute(std::string name, int index, double min_val = 0.0,
                        double max_val = 0.0)
        : Attribute(std::move(name), index),
          min_val_(min_val),
          max_val_(max_val) {}

    double min_val() const noexcept { return min_val_; }
    double max_val() const noexcept { return max_val_; }

    bool is_continuous() const noexcept override { return true; }

    double get_scaled_value(double v) const override {
        const double r = max_val_ - min_val_;
        if (r == 0.0) {
            return 0.0;
        }
        return (v - min_val_) / r;
    }

    bool equals(const Attribute& other) const noexcept override {
        const auto* o = dynamic_cast<const ContinuousAttribute*>(&other);
        return o != nullptr && name_ == o->name_ && index_ == o->index_ &&
               min_val_ == o->min_val_ && max_val_ == o->max_val_;
    }

private:
    double min_val_;
    double max_val_;
};

class DiscreteAttribute final : public Attribute {
public:
    using ValueSet = std::set<std::string>;

    explicit DiscreteAttribute(std::string name, int index, ValueSet values = {})
        : Attribute(std::move(name), index), values_(std::move(values)) {}

    const ValueSet& values() const noexcept { return values_; }

    bool is_continuous() const noexcept override { return false; }

    std::size_t num_values() const noexcept override { return values_.size(); }

    bool equals(const Attribute& other) const noexcept override {
        const auto* o = dynamic_cast<const DiscreteAttribute*>(&other);
        return o != nullptr && name_ == o->name_ && index_ == o->index_ &&
               values_ == o->values_;
    }

private:
    ValueSet values_;
};

}

#endif
