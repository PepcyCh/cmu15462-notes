#include "sampler.h"
#include <vector>

namespace CMU462 {

// Uniform Sampler2D Implementation //

Vector2D UniformGridSampler2D::get_sample() const {
  // Implement uniform 2D grid sampler

  double x = double(std::rand()) / RAND_MAX;
  double y = double(std::rand()) / RAND_MAX;

  return Vector2D(x, y);
}

// Uniform Hemisphere Sampler3D Implementation //

Vector3D UniformHemisphereSampler3D::get_sample() const {
  double Xi1 = (double)(std::rand()) / RAND_MAX;
  double Xi2 = (double)(std::rand()) / RAND_MAX;

  double theta = acos(Xi1);
  double phi = 2.0 * PI * Xi2;

  double xs = sinf(theta) * cosf(phi);
  double ys = sinf(theta) * sinf(phi);
  double zs = cosf(theta);

  return Vector3D(xs, ys, zs);
}

Vector3D CosineWeightedHemisphereSampler3D::get_sample() const {
  float f;
  return get_sample(&f);
}

Vector3D CosineWeightedHemisphereSampler3D::get_sample(float *pdf) const {
  // You may implement this, but don't have to.
  return Vector3D(0, 0, 1);
}

Vector2D JitteredSampler::get_sample() const {
  static std::vector<Vector2D> samples(ns_aa);
  static int curr = ns_aa;
  if (curr == ns_aa) {
    int w = std::sqrt(ns_aa);
    curr = 0;
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < w; j++) {
        double x = (i + double(std::rand()) / RAND_MAX) / w;
        double y = (j + double(std::rand()) / RAND_MAX) / w;
        samples[curr++] = Vector2D(x, y);
      }
    }
    curr = 0;
  }
  return samples[curr++];
}

Vector2D MultiJitteredSampler::get_sample() const {
    static std::vector<Vector2D> samples(ns_aa);
    static int curr = ns_aa;
    if (curr == ns_aa) {
        int w = std::sqrt(ns_aa);
        curr = 0;
        int u = 0, v = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < w; j++) {
                double x = (u + double(std::rand()) / RAND_MAX) / w;
                double y = (v + double(std::rand()) / RAND_MAX) / w;
                samples[curr++] = Vector2D(x, y);
                if (v == w - 1) {
                    ++u;
                    v = 0;
                } else {
                    ++v;
                }
            }
        }
        std::random_shuffle(samples.begin(), samples.begin() + curr);
        curr = 0;
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < w; j++) {
                double x = (i + samples[curr].x) / w;
                double y = (j + samples[curr].y) / w;
                samples[curr++] = Vector2D(x, y);
            }
        }
        curr = 0;
    }
    return samples[curr++];
}

Vector2D NRooksSampler::get_sample() const {
    static std::vector<Vector2D> samples(ns_aa);
    static int curr = ns_aa;
    if (curr == ns_aa) {
        curr = 0;
        std::vector<double> X(ns_aa), Y(ns_aa);
        for (int i = 0; i < ns_aa; i++) {
            X[i] = (i + double(std::rand()) / RAND_MAX) / ns_aa;
            Y[i] = (i + double(std::rand()) / RAND_MAX) / ns_aa;
        }
        std::random_shuffle(Y.begin(), Y.end());
        for (int i = 0; i < ns_aa; i++) {
            samples[i] = Vector2D(X[i], Y[i]);
        }
        curr = 0;
    }
    return samples[curr++];
}

static double radicalInverse3(int i) {
    double b = 1. / 3.;
    double t = b;
    double res = 0;
    while (i) {
        res += t * (i % 3);
        t *= b;
        i /= 3;
    }
    return res;
}
static double radicalInverse2(int i) {
    double t = 1. / 2.;
    double res = 0;
    while (i) {
        res += t * (i & 1);
        t /= 2.;
        i >>= 1;
    }
    return res;
}

Vector2D SobolSampler::get_sample() const {
    static std::vector<Vector2D> samples(ns_aa);
    static bool init = false;
    static int curr = ns_aa;
    if (!init) {
        std::vector<unsigned> C(ns_aa), V(32);
        for (int i = 0; i < ns_aa; i++) {
            int w = i;
            C[i] = 1;
            while (w & 1) {
                ++C[i];
                w >>= 1;
            }
        }
        for (int i = 1; i <= 31; i++) V[i] = 1 << (32 - i);

        std::vector<unsigned> X(ns_aa);
        X[0] = 0;
        for (int i = 1; i < ns_aa; i++) {
            X[i] = X[i - 1] ^ V[C[i - 1]];
            samples[i].x = X[i] / std::pow(2., 32.);
        }

        V[1] = 1 << 31;
        for (int i = 2; i <= 31; i++) V[i] = V[i - 1] ^ (V[i - 1] >> 1);
        X[0] = 0;
        for (int i = 1; i < ns_aa; i++) {
            X[i] = X[i - 1] ^ V[C[i - 1]];
            samples[i].y = X[i] / std::pow(2., 32.);
        }
        init = true;
    }
    if (curr == ns_aa) curr = 0;
    return samples[curr++];
}

Vector2D HaltonSampler::get_sample() const {
    static std::vector<Vector2D> samples(ns_aa);
    static bool init = false;
    static int curr = ns_aa;
    if (!init) {
        for (int i = 0; i < ns_aa; i++) {
            double x = radicalInverse2(i);
            double y = radicalInverse3(i);
            samples[i] = Vector2D(x, y);
        }
        init = true;
    }
    if (curr == ns_aa) curr = 0;
    return samples[curr++];
}

Vector2D HammersleySampler::get_sample() const {
    static std::vector<Vector2D> samples(ns_aa);
    static bool init = false;
    static int curr = ns_aa;
    if (!init) {
        for (int i = 0; i < ns_aa; i++) {
            double x = double(i) / ns_aa;
            double y = radicalInverse2(i);
            samples[i] = Vector2D(x, y);
        }
        init = true;
    }
    if (curr == ns_aa) curr = 0;
    return samples[curr++];
}

}  // namespace CMU462
