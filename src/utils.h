#ifndef UTILS_H
#define UTILS_H

// C includes:
#include <cstdint>

namespace NNet {

namespace Utils {

/**
 * Vector2
 * Description:
 * 2D Vector to hold dimensions.
 *
 * x - width
 * y - height
 */
template<typename T>
struct Vector2 {
 public:
  T x;
  T y;

  Vector2() : x(0), y(0) {}
  Vector2(T x, T y) : x(x), y(y) {}
};

typedef Vector2<uint32_t> Vector2u32;

/**
 * Vector3
 * Description:
 * 3D Vector to hold dimensions.
 *
 * z - depth
 */
template<typename T>
struct Vector3 : public Vector2<T> {
 public:
  T z;

  Vector3() : Vector2<T>(), z(0) {}
  Vector3(T x, T y, T z) : Vector2<T>(x, y), z(z) {}

};

typedef Vector3<uint32_t> Vector3u32;

} // namespace

} // namespace

#endif // UTILS_H
