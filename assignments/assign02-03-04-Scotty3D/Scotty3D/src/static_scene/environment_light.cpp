#include <iostream>
#include "environment_light.h"

namespace CMU462 {
namespace StaticScene {

void EnvironmentLight::AliasTable::init(const std::vector<double>& vec) {
  int N = vec.size();
  items.resize(N);
  for (int i = 0; i < N; i++) {
    items[i].id0 = i;
    items[i].id1 = -1;
    items[i].ratio = vec[i];
  }

  double mid = 1.0 / N;
  int rich = -1, poor = -1;
  for (int i = 0; i < N; i++) if (items[i].ratio < mid) {
    poor = i;
    break;
  }
  for (int i = 0; i < N; i++) if (items[i].ratio > mid) {
    rich = i;
    break;
  }

  while (rich != -1 && poor != -1) {
    double diff = mid - items[poor].ratio;
    items[poor].id1 = rich;
    items[poor].ratio = mid;
    items[rich].ratio -= diff;

    int temp_poor = -1;
    if (items[rich].ratio < mid && rich < poor) {
      temp_poor = rich;
    } else {
      for (int i = poor; i < N; i++) if (items[i].ratio < mid) {
        temp_poor = i;
        break;
      }
    }
    poor = temp_poor;

    int temp_rich = -1;
    for (int i = rich; i < N; i++) if (items[i].ratio > mid) {
      temp_rich = i;
      break;
    }
    rich = temp_rich;
  }
}

int EnvironmentLight::AliasTable::sample(double p) const {
  int id = p;
  double left = p - id;
  return left <= items[id].ratio ? items[id].id0 : items[id].id1;
}

EnvironmentLight::EnvironmentLight(const HDRImageBuffer* envMap)
    : envMap(envMap) {
  int w = envMap->w, h = envMap->h;

  probs.resize(w * h);
  double sum = 0;
  for (int j = 0; j < h; j++) {
    double theta = (j + 0.5) / h * PI;
    double sintheta = sin(theta);
    for (int i = 0; i < w; i++) {
      int id = i + j * w;
      probs[id] = envMap->data[id].illum() * sintheta;
      sum += probs[id];
    }
  }
  for (double& p : probs) p /= sum;
  table.init(probs);
}

Spectrum EnvironmentLight::sample_L(const Vector3D& p, Vector3D* wi,
                                    float* distToLight, float* pdf) const {
  *distToLight = INF_F;

  double rnd = double(rand()) / RAND_MAX * probs.size();
  int id = table.sample(rnd);
  *pdf = probs[id];

  int x = id % envMap->w;
  int y = id / envMap->w;
  double theta = PI * (y + double(rand()) / RAND_MAX) / envMap->h;
  double phi = 2 * PI * (x + double(rand() / RAND_MAX)) / envMap->w;

  wi->x = sin(theta) * cos(phi);
  wi->z = sin(theta) * sin(phi);
  wi->y = cos(theta);

  return sample_dir(*wi);
}

Spectrum EnvironmentLight::sample_dir(const Vector3D& r) const {
  int w = envMap->w, h = envMap->h;
  double theta = acos(r.y);
  double phi = atan2(r.z, r.x) + PI;
  double tx = phi / 2 / PI * w;
  double ty = theta / PI * h;

  int x[2], y[2];
  x[0] = round(tx) - 1;
  x[1] = x[0] + 1;
  x[0] = clamp(x[0], 0, w - 1);
  x[1] = clamp(x[1], 0, w - 1);
  double dx = tx - x[0] - 0.5;
  y[0] = round(ty) - 1;
  y[1] = y[0] + 1;
  y[0] = clamp(y[0], 0, h - 1);
  y[1] = clamp(y[1], 0, h - 1);
  double dy = ty - y[0] - 0.5;

  Spectrum mix(0, 0, 0);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      int id = x[i] + y[j] * w;
      mix += envMap->data[id] * (i * dx + (1 - i) * (1 - dx)) * (j * dy + (1 - j) * (1 - dy));
    }
  }

  return mix;
}

}  // namespace StaticScene
}  // namespace CMU462
