#pragma once

namespace {
    template<typename T, typename... AllowedTs>
    concept has_t = (std::is_same_v<T, AllowedTs> || ...);

}

namespace cup {

    // T has to be one of the other types
    // taken from cuda docs
    template<typename T>
    concept native_warp_syncable = has_t<T,
                                         int,
                                         unsigned int,
                                         long,
                                         unsigned long,
                                         long long,
                                         unsigned long long,
                                         float,
                                         double>;

    // clang-format off
    template <typename T>
    concept warp_syncable =
    native_warp_syncable<T> || requires(T a, uint32_t b, int c) {
        a.warp_sync(b, c);
    };

    // clang-format on
}  // namespace cup
