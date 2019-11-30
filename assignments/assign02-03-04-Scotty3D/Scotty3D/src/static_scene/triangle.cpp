#include "triangle.h"

#include "CMU462/CMU462.h"
#include "GL/glew.h"

namespace CMU462 {
namespace StaticScene {

Triangle::Triangle(const Mesh* mesh, vector<size_t>& v) : mesh(mesh), v(v) {}
Triangle::Triangle(const Mesh* mesh, size_t v1, size_t v2, size_t v3)
    : mesh(mesh), v1(v1), v2(v2), v3(v3) {}

BBox Triangle::get_bbox() const {
  // compute the bounding box of the triangle
  Vector3D p1 = mesh->positions[v1];
  Vector3D p2 = mesh->positions[v2];
  Vector3D p3 = mesh->positions[v3];
  double minX = std::min({p1.x, p2.x, p3.x});
  double maxX = std::max({p1.x, p2.x, p3.x});
  double minY = std::min({p1.y, p2.y, p3.y});
  double maxY = std::max({p1.y, p2.y, p3.y});
  double minZ = std::min({p1.z, p2.z, p3.z});
  double maxZ = std::max({p1.z, p2.z, p3.z});

  return BBox(minX, minY, minZ, maxX, maxY, maxZ);
}

static bool doesRayIntersectSegment(const Ray& r, const Vector3D& v0, const Vector3D& v1, double &t) {
    Vector3D U = v1 - v0;
    Vector3D V = cross(U, r.d);
    if (V.norm() == 0) return false;
    Vector3D W = cross(U, V);
    double c = dot(W, v0);
    double temp = dot(W, r.d);
    if (temp == 0) return false;
    t = (c - dot(W, r.o)) / temp;
    Vector3D p = r.o + t * r.d;
    double u = dot(p - v0, U) / U.norm();
    return u >= 0 && u <= 1;
}

bool Triangle::intersect(const Ray& r) const {
  Vector3D p1 = mesh->positions[v1];
  Vector3D p2 = mesh->positions[v2];
  Vector3D p3 = mesh->positions[v3];

  Vector3D e1 = p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s = r.o - p1;

  double det = dot(cross(e1, r.d), e2);
  if (det != 0) {
    double du = -dot(cross(s, e2), r.d);
    double dv = dot(cross(e1, r.d), s);
    double dt = -dot(cross(s, e2), e1);
    double u = du / det;
    double v = dv / det;
    double t = dt / det;
    if (u < 0 || v < 0 || 1 - u - v < 0) return false;
    return !(t < r.min_t || t > r.max_t);
  } else {
    double t;
    if (doesRayIntersectSegment(r, p1, p2, t)) return !(t < r.min_t || t > r.max_t);
    if (doesRayIntersectSegment(r, p1, p3, t)) return !(t < r.min_t || t > r.max_t);
    if (doesRayIntersectSegment(r, p2, p3, t)) return !(t < r.min_t || t > r.max_t);
  }

  return false;
}

bool Triangle::intersect(const Ray& r, Intersection* isect) const {
  // implement ray-triangle intersection. When an intersection takes
  // place, the Intersection data should be updated accordingly

  Vector3D p1 = mesh->positions[v1];
  Vector3D p2 = mesh->positions[v2];
  Vector3D p3 = mesh->positions[v3];
  Vector3D n1 = mesh->normals[v1];
  Vector3D n2 = mesh->normals[v2];
  Vector3D n3 = mesh->normals[v3];

  Vector3D e1 = p2 - p1;
  Vector3D e2 = p3 - p1;
  Vector3D s = r.o - p1;

  double det = dot(cross(e1, r.d), e2);
  if (det != 0) {
    double du = -dot(cross(s, e2), r.d);
    double dv = dot(cross(e1, r.d), s);
    double dt = -dot(cross(s, e2), e1);
    double u = du / det;
    double v = dv / det;
    double t = dt / det;
    if (u < 0 || v < 0 || 1 - u - v < 0) return false;
    if (t < r.min_t || t > r.max_t) return false;
    isect->n = u * n1 + v * n2 + (1 - u - v) * n3;
    if (dot(isect->n, r.d) > 0) isect->n = -isect->n;
    r.max_t = isect->t = t;
    isect->primitive = this;
    isect->bsdf = get_bsdf();
    return true;
  } else {
    Matrix3x3 inv = Matrix3x3();
    inv.column(0) = p1;
    inv.column(1) = p2;
    inv.column(2) = p3;
    inv = inv.inv();
    double t, u, v;
    bool inter = false;
    if (doesRayIntersectSegment(r, v1, v2, t)) {
      if (t < r.min_t || t > r.max_t) return false;
      Vector3D ret = inv * (r.o + t * r.d);
      u = ret.x;
      v = ret.y;
      inter = true;
    }
    if (doesRayIntersectSegment(r, v1, v3, t)) {
      if (t < r.min_t || t > r.max_t) return false;
      Vector3D ret = inv * (r.o + t * r.d);
      u = ret.x;
      v = ret.y;
      inter = true;
    }
    if (doesRayIntersectSegment(r, v2, v3, t)) {
      if (t < r.min_t || t > r.max_t) return false;
      Vector3D ret = inv * (r.o + t * r.d);
      u = ret.x;
      v = ret.y;
      inter = true;
    }
    if (!inter) return false;
    isect->n = u * n1 + v * n2 + (1 - u - v) * n3;
    if (dot(isect->n, r.d) > 0) isect->n = -isect->n;
    r.max_t = isect->t = t;
    isect->primitive = this;
    isect->bsdf = get_bsdf();
    return true;
  }
}

void Triangle::draw(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_TRIANGLES);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

void Triangle::drawOutline(const Color& c) const {
  glColor4f(c.r, c.g, c.b, c.a);
  glBegin(GL_LINE_LOOP);
  glVertex3d(mesh->positions[v1].x, mesh->positions[v1].y,
             mesh->positions[v1].z);
  glVertex3d(mesh->positions[v2].x, mesh->positions[v2].y,
             mesh->positions[v2].z);
  glVertex3d(mesh->positions[v3].x, mesh->positions[v3].y,
             mesh->positions[v3].z);
  glEnd();
}

}  // namespace StaticScene
}  // namespace CMU462
