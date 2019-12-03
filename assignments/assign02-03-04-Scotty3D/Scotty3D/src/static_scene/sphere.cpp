#include "sphere.h"

#include <cmath>

#include "../bsdf.h"
#include "../misc/sphere_drawing.h"

namespace CMU462 {
namespace StaticScene {

bool Sphere::test(const Ray& r, double& t1, double& t2) const {
  // Implement ray - sphere intersection test.
  // Return true if there are intersections and writing the
  // smaller of the two intersection times in t1 and the larger in t2.

  Vector3D oc = r.o - o;
  double a = r.d.norm2();
  double b = dot(oc, r.d);
  double c = oc.norm2() - r2;
  double disc = b * b - a * c;
  if (disc > 0) {
    double sqrtd = sqrt(disc);
    t1 = (-b - sqrtd) / a;
    t2 = (-b + sqrtd) / a;
    return true;
  }
  return false;
}

bool Sphere::intersect(const Ray& r) const {
  // Implement ray - sphere intersection.
  // Note that you might want to use the the Sphere::test helper here.
  double t1, t2;
  if (test(r, t1, t2)) {
    return !(t1 > r.max_t || t2 < r.min_t);
  }

  return false;
}

bool Sphere::intersect(const Ray& r, Intersection* isect) const {
  // Implement ray - sphere intersection.
  // Note again that you might want to use the the Sphere::test helper here.
  // When an intersection takes place, the Intersection data should be updated
  // correspondingly.

  double t1, t2;
  if (test(r, t1, t2)) {
    if (t1 > r.max_t || t2 < r.min_t) return false;
    isect->primitive = this;
    isect->bsdf = get_bsdf();
    if (t1 > r.min_t) {
      r.max_t = isect->t = t1;
      isect->n = (r.o + t1 * r.d - o);
      isect->n.normalize();
    } else {
      r.max_t = isect->t = t2;
      isect->n = (r.o + t2 * r.d - o);
      isect->n.normalize();
    }
    return true;
  }

  return false;
}

void Sphere::draw(const Color& c) const { Misc::draw_sphere_opengl(o, r, c); }

void Sphere::drawOutline(const Color& c) const {
  // Misc::draw_sphere_opengl(o, r, c);
}

}  // namespace StaticScene
}  // namespace CMU462
