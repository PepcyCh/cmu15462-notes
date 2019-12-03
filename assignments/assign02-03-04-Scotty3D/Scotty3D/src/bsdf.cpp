#include "bsdf.h"

#include <algorithm>
#include <iostream>
#include <utility>


using std::min;
using std::max;
using std::swap;

namespace CMU462 {

void make_coord_space(Matrix3x3& o2w, const Vector3D& n) {
  Vector3D z = Vector3D(n.x, n.y, n.z);
  Vector3D h = z;
  if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z))
    h.x = 1.0;
  else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z))
    h.y = 1.0;
  else
    h.z = 1.0;

  z.normalize();
  Vector3D y = cross(h, z);
  y.normalize();
  Vector3D x = cross(z, y);
  x.normalize();

  o2w[0] = x;
  o2w[1] = y;
  o2w[2] = z;
}

// Diffuse BSDF //

Spectrum DiffuseBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return albedo * (1.0 / PI);
}

Spectrum DiffuseBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // Implement DiffuseBSDF
  *wi = sampler.get_sample(pdf);
  if (wo.z < 0) wi->z *= -1.;
  return albedo * (1.0 / PI);
}

// Mirror BSDF //

Spectrum MirrorBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return 1.0 / fabs(wi.z) * reflectance;
}

Spectrum MirrorBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // Implement MirrorBSDF
  reflect(wo, wi);
  *pdf = 1.0;
  return 1.0 / fabs(wi->z) * reflectance;
}

// Glossy BSDF //

/*
Spectrum GlossyBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum GlossyBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *pdf = 1.0f;
  return reflect(wo, wi, reflectance);
}
*/

// Refraction BSDF //

Spectrum RefractionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return transmittance;
}

Spectrum RefractionBSDF::sample_f(const Vector3D& wo, Vector3D* wi,
                                  float* pdf) {
  // Implement RefractionBSDF
  if (!refract(wo, wi, ior)) {
    return Spectrum();
  }
  *pdf = 1.0;
  return 1.0 / fabs(wi->z) * transmittance;
}

// Glass BSDF //
double schlick(double cosine, double ior) {
  double r0 = (1 - ior) / (1 + ior);
  r0 = r0 * r0;
  return r0 + (1 - r0) * pow(1 - cosine, 5);
}

Spectrum GlassBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  Vector3D temp;
  bool canRefract = refract(wo, &temp, ior);
  double fresnel = canRefract ? schlick(fabs(wo.z), ior) : 1.0;
  double k = wo.z >= 0 ? 1.0 / ior : ior;
  if (wo.z * wi.z >= 0) return fresnel / fabs(wi.z) * reflectance;
  else return k * k * (1 - fresnel) / fabs(wi.z) * transmittance;
}

Spectrum GlassBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  // Compute Fresnel coefficient and either reflect or refract based on it.
  Vector3D refle, refra;
  bool canRefract = refract(wo, &refra, ior);
  reflect(wo, &refle);
  double fresnel = canRefract ? schlick(fabs(wo.z), ior) : 1.0;
  double rnd = double(rand()) / RAND_MAX;

  Spectrum retf;
  if (rnd <= fresnel) {
    *wi = refle;
    retf = fresnel / fabs(wi->z) * reflectance;
    *pdf = fresnel;
  } else {
    double k = wo.z >= 0 ? 1.0 / ior : ior;
    *wi = refra;
    retf = k * k * (1 - fresnel) / fabs(wi->z) * transmittance;
    *pdf = 1.0 - fresnel;
  }

  return retf;
}

void BSDF::reflect(const Vector3D& wo, Vector3D* wi) {
  // Implement reflection of wo about normal (0,0,1) and store result in wi.
  wi->x = -wo.x;
  wi->y = -wo.y;
  wi->z = wo.z;
}

bool BSDF::refract(const Vector3D& wo, Vector3D* wi, float ior) {
  // Use Snell's Law to refract wo surface and store result ray in wi.
  // Return false if refraction does not occur due to total internal reflection
  // and true otherwise. When dot(wo,n) is positive, then wo corresponds to a
  // ray entering the surface through vacuum.

  double k = wo.z >= 0 ? 1.0 / ior : ior;
  double d = 1 - (1 - wo.z * wo.z) * k * k;
  if (d < 0) return false;
  wi->x = -wo.x * k;
  wi->y = -wo.y * k;
  wi->z = sqrt(d);
  if (wo.z >= 0) wi->z *= -1.;

  return true;
}

// Emission BSDF //

Spectrum EmissionBSDF::f(const Vector3D& wo, const Vector3D& wi) {
  return Spectrum();
}

Spectrum EmissionBSDF::sample_f(const Vector3D& wo, Vector3D* wi, float* pdf) {
  *wi = sampler.get_sample(pdf);
  return Spectrum();
}

}  // namespace CMU462
