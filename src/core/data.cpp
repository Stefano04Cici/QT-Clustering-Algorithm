#ifndef QT_CLUSTERING_CORE_DATA_CPP
#define QT_CLUSTERING_CORE_DATA_CPP

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "tuple.cpp"

namespace qt {

class Data {
public:
    using Cell = Item::Value;
    using AttributePtr = std::shared_ptr<const Attribute>;

    Data() = default;

    Data(std::vector<std::vector<Cell>> examples,
         std::vector<AttributePtr> attributes)
        : examples_(std::move(examples)),
          attributes_(std::move(attributes)) {}

    const std::vector<std::vector<Cell>>& examples() const noexcept {
        return examples_;
    }

    const std::vector<AttributePtr>& attributes() const noexcept {
        return attributes_;
    }

    static Data from_csv(const std::string& filepath) {
        std::ifstream input(filepath);
        if (!input) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }

        std::vector<std::vector<std::string>> rows;
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            rows.push_back(parse_csv_line(line));
        }

        if (rows.empty()) {
            return Data{};
        }

        const std::vector<std::string>& header = rows.front();
        std::vector<AttributePtr> attributes;
        attributes.reserve(header.size());

        for (std::size_t col = 0; col < header.size(); ++col) {
            std::vector<std::string> col_values;
            for (std::size_t r = 1; r < rows.size(); ++r) {
                if (col < rows[r].size() && !rows[r][col].empty()) {
                    col_values.push_back(rows[r][col]);
                }
            }

            std::vector<double> numeric_vals;
            bool all_numeric = true;
            for (const auto& value : col_values) {
                double parsed = 0.0;
                if (!try_parse_double(value, parsed)) {
                    all_numeric = false;
                    break;
                }
                numeric_vals.push_back(parsed);
            }

            if (all_numeric && !numeric_vals.empty()) {
                const auto [lo, hi] = std::minmax_element(numeric_vals.begin(),
                                                          numeric_vals.end());
                attributes.push_back(std::make_shared<ContinuousAttribute>(
                    header[col], static_cast<int>(col), *lo, *hi));
            } else {
                std::set<std::string> distinct(col_values.begin(),
                                               col_values.end());
                attributes.push_back(std::make_shared<DiscreteAttribute>(
                    header[col], static_cast<int>(col), std::move(distinct)));
            }
        }

        std::vector<std::vector<Cell>> examples;
        examples.reserve(rows.size() - 1);
        for (std::size_t r = 1; r < rows.size(); ++r) {
            std::vector<Cell> row;
            row.reserve(rows[r].size());
            for (const auto& value : rows[r]) {
                row.emplace_back(value);
            }
            examples.push_back(std::move(row));
        }

        return Data{std::move(examples), std::move(attributes)};
    }

    static Data from_matrix(const std::vector<std::vector<double>>& matrix,
                            const std::vector<std::string>& feature_names = {}) {
        const std::size_t n_rows = matrix.size();
        const std::size_t n_cols = matrix.empty() ? 0 : matrix.front().size();
        if (n_rows == 0 || n_cols == 0) {
            return Data{};
        }

        std::vector<std::string> names = feature_names;
        if (names.empty()) {
            names.reserve(n_cols);
            for (std::size_t i = 0; i < n_cols; ++i) {
                names.push_back("feat_" + std::to_string(i));
            }
        }

        std::vector<AttributePtr> attributes;
        attributes.reserve(n_cols);
        for (std::size_t col = 0; col < n_cols; ++col) {
            double lo = matrix[0][col];
            double hi = matrix[0][col];
            for (std::size_t r = 1; r < n_rows; ++r) {
                lo = std::min(lo, matrix[r][col]);
                hi = std::max(hi, matrix[r][col]);
            }
            attributes.push_back(std::make_shared<ContinuousAttribute>(
                names.at(col), static_cast<int>(col), lo, hi));
        }

        std::vector<std::vector<Cell>> examples;
        examples.reserve(n_rows);
        for (std::size_t r = 0; r < n_rows; ++r) {
            std::vector<Cell> row;
            row.reserve(n_cols);
            for (std::size_t c = 0; c < n_cols; ++c) {
                row.emplace_back(matrix[r][c]);
            }
            examples.push_back(std::move(row));
        }

        return Data{std::move(examples), std::move(attributes)};
    }

    std::size_t get_number_of_examples() const noexcept {
        return examples_.size();
    }

    std::size_t get_number_of_attributes() const noexcept {
        return attributes_.size();
    }

    const Attribute& get_attribute(std::size_t index) const {
        if (index >= attributes_.size()) {
            throw std::out_of_range(
                "index " + std::to_string(index) + " out of range [0, " +
                std::to_string(attributes_.size()) + ")");
        }
        return *attributes_[index];
    }

    const Cell& get_value(std::size_t example_idx,
                          std::size_t attribute_idx) const {
        if (example_idx >= examples_.size()) {
            throw std::out_of_range(
                "example_idx " + std::to_string(example_idx) +
                " out of range [0, " + std::to_string(examples_.size()) + ")");
        }
        if (attribute_idx >= attributes_.size()) {
            throw std::out_of_range(
                "attribute_idx " + std::to_string(attribute_idx) +
                " out of range [0, " + std::to_string(attributes_.size()) +
                ")");
        }
        return examples_[example_idx].at(attribute_idx);
    }

    QTuple get_item_set(std::size_t index) const {
        if (index >= examples_.size()) {
            throw std::out_of_range(
                "index " + std::to_string(index) + " out of range [0, " +
                std::to_string(examples_.size()) + ")");
        }

        std::vector<QTuple::ItemHolder> items;
        items.reserve(attributes_.size());
        for (const auto& attribute : attributes_) {
            const Cell& raw =
                examples_[index].at(static_cast<std::size_t>(attribute->index()));
            if (const auto* continuous =
                    dynamic_cast<const ContinuousAttribute*>(attribute.get())) {
                const double value = std::visit(
                    [](const auto& cell) -> double {
                        using T = std::decay_t<decltype(cell)>;
                        if constexpr (std::is_same_v<T, double>) {
                            return cell;
                        } else {
                            return std::stod(cell);
                        }
                    },
                    raw);
                items.emplace_back(ContinuousItem{*continuous, value});
            } else if (const auto* discrete =
                           dynamic_cast<const DiscreteAttribute*>(attribute.get())) {
                std::string value = value_to_string(raw);
                items.emplace_back(DiscreteItem{*discrete, std::move(value)});
            }
        }
        return QTuple{std::move(items)};
    }

    std::string to_string() const {
        std::ostringstream out;

        std::string header;
        for (std::size_t i = 0; i < attributes_.size(); ++i) {
            if (i != 0) {
                header += " | ";
            }
            header += attributes_[i]->name();
        }
        out << header << '\n';
        out << std::string(header.size(), '-') << '\n';

        const std::size_t limit = std::min<std::size_t>(examples_.size(), 10);
        for (std::size_t r = 0; r < limit; ++r) {
            for (std::size_t c = 0; c < examples_[r].size(); ++c) {
                if (c != 0) {
                    out << " | ";
                }
                out << value_to_string(examples_[r][c]);
            }
            out << '\n';
        }
        if (examples_.size() > 10) {
            out << "... (" << examples_.size() << " rows total)\n";
        }

        std::string result = out.str();
        while (!result.empty() && result.back() == '\n') {
            result.pop_back();
        }
        return result;
    }

    bool equals(const Data& other) const {
        if (examples_ != other.examples_) {
            return false;
        }
        if (attributes_.size() != other.attributes_.size()) {
            return false;
        }
        for (std::size_t i = 0; i < attributes_.size(); ++i) {
            if (!(*attributes_[i] == *other.attributes_[i])) {
                return false;
            }
        }
        return true;
    }

private:
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

    static std::vector<std::string> parse_csv_line(const std::string& line) {
        std::vector<std::string> fields;
        std::string field;
        bool in_quotes = false;
        for (std::size_t i = 0; i < line.size(); ++i) {
            const char c = line[i];
            if (in_quotes) {
                if (c == '"') {
                    if (i + 1 < line.size() && line[i + 1] == '"') {
                        field.push_back('"');
                        ++i;
                    } else {
                        in_quotes = false;
                    }
                } else {
                    field.push_back(c);
                }
            } else if (c == '"') {
                in_quotes = true;
            } else if (c == ',') {
                fields.push_back(field);
                field.clear();
            } else {
                field.push_back(c);
            }
        }
        fields.push_back(field);
        return fields;
    }

    static std::string value_to_string(const Cell& cell) {
        return std::visit(
            [](const auto& value) -> std::string {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return value;
                } else {
                    std::ostringstream out;
                    out << value;
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
            cell);
    }

    std::vector<std::vector<Cell>> examples_;
    std::vector<AttributePtr> attributes_;
};

inline bool operator==(const Data& a, const Data& b) { return a.equals(b); }

inline bool operator!=(const Data& a, const Data& b) { return !(a == b); }

}

#endif
