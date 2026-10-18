#pragma once

// Sınıf sırası (Python `baykus.classes`): cls_id = sıra; COCO category_id = cls_id + 1.
#include <array>
#include <string_view>

namespace baykus {

constexpr std::array<std::string_view, 5> kClassNames = {"car", "truck", "bus", "van",
                                                         "freight_car"};

}  // namespace baykus
