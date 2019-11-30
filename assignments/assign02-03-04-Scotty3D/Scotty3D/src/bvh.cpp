#include "bvh.h"

#include "CMU462/CMU462.h"
#include "static_scene/triangle.h"

#include <iostream>
#include <stack>
#include <queue>
#include <cfloat>

using namespace std;

namespace CMU462 {
namespace StaticScene {


BVHAccel::BVHAccel(const std::vector<Primitive *> &_primitives,
                   size_t max_leaf_size) {
    static const int B = 16;

    this->primitives = _primitives;

    // Construct a BVH from the given vector of primitives and maximum leaf
    // size configuration. The starter code build a BVH aggregate with a
    // single leaf node (which is also the root) that encloses all the
    // primitives.

    BBox bb;
    for (int i = 0; i < primitives.size(); i++) {
        bb.expand(primitives[i]->get_bbox());
    }
    root = new BVHNode(bb, 0, primitives.size());

    std::queue<BVHNode *> q;
    q.push(root);
    while (!q.empty()) {
        auto u = q.front();
        q.pop();
        if (u->range <= max_leaf_size) continue;

        int bestD = -1, bestI = -1;
        double SN = u->bb.surface_area(), bestC = DBL_MAX;

        for (int d = 0; d < 3; d++) {
            std::vector<BBox> boxes(B);
            std::vector<std::vector<Primitive*>> prims(B);
            double min = u->bb.min[d], max = u->bb.max[d];
            double length = (max - min) / B;
            if (length == 0) continue;

            for (size_t i = 0; i < u->range; i++) {
                BBox cb = primitives[u->start + i]->get_bbox();
                double p = cb.centroid()[d];
                int buc = clamp<int>((p - min) / length, 0, B - 1);
                prims[buc].push_back(primitives[u->start + i]);
                boxes[buc].expand(cb);
            }

            for (int i = 1; i < B; i++) {
                BBox lb, rb;
                int ln = 0, rn = 0;
                for (int j = 0; j < i; j++) {
                    lb.expand(boxes[j]);
                    ln += prims[j].size();
                }
                for (int j = i; j < B; j++) {
                    rb.expand(boxes[j]);
                    rn += prims[j].size();
                }
                double SA = lb.surface_area(), SB = rb.surface_area();
                double C = SA / SN * ln + SB / SN * rn;
                if (C < bestC) {
                    bestD = d;
                    bestI = i;
                    bestC = C;
                }
            }
        }

        double min = u->bb.min[bestD], max = u->bb.max[bestD];
        double length = (max - min) / B;
        BBox lb, rb;
        std::vector<Primitive *> lp, rp;
        for (size_t i = 0; i < u->range; i++) {
            BBox cb = primitives[u->start + i]->get_bbox();
            double p = cb.centroid()[bestD];
            int buc = clamp<int>((p - min) / length, 0, B - 1);
            if (buc < bestI) {
                lb.expand(cb);
                lp.push_back(primitives[u->start + i]);
            } else {
                rb.expand(cb);
                rp.push_back(primitives[u->start + i]);
            }
        }

        if (lp.size() == 0 || lp.size() == u->range) {
            lb = BBox(), rb = BBox();
            int hn = u->range / 2;
            for (int i = 0; i < hn; i++) lb.expand(primitives[u->start + i]->get_bbox());
            for (int i = hn; i < u->range; i++) rb.expand(primitives[u->start + i]->get_bbox());
            u->l = new BVHNode(lb, u->start, hn);
            u->r = new BVHNode(rb, u->start + hn, u->range - hn);
            q.push(u->l);
            q.push(u->r);
        } else {
            int p = 0;
            for (auto prim : lp) {
                primitives[u->start + p] = prim;
                ++p;
            }
            int ln = p;
            for (auto prim : rp) {
                primitives[u->start + p] = prim;
                ++p;
            }
            u->l = new BVHNode(lb, u->start, ln);
            u->r = new BVHNode(rb, u->start + ln, u->range - ln);
            q.push(u->l);
            q.push(u->r);
        }
    }
}


BVHAccel::~BVHAccel() {
  // Implement a proper destructor for your BVH accelerator aggregate

  std::queue<BVHNode *> q;
  q.push(root);
  while (!q.empty()) {
    auto u = q.front();
    q.pop();
    if (u->l) q.push(u->l);
    if (u->r) q.push(u->r);
    delete u;
  }
}

BBox BVHAccel::get_bbox() const { return root->bb; }

bool BVHAccel::intersect(const Ray &ray) const {
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate.

  bool hit = false;

  std::stack<BVHNode *> s;
  s.push(root);
  while (!s.empty()) {
    auto u = s.top();
    s.pop();

    double t0, t1;
    if (u->bb.intersect(ray, t0, t1)) {
      if (u->isLeaf()) {
        for (size_t i = 0; i < u->range; i++) {
          if (primitives[u->start + i]->intersect(ray)) {
            hit = true;
            break;
          }
        }
        if (hit) break;
      } else {
        if (u->l) s.push(u->l);
        if (u->r) s.push(u->r);
      }
    }
  }

  return hit;
}

bool BVHAccel::intersect(const Ray &ray, Intersection *isect) const {
  // Implement ray - bvh aggregate intersection test. A ray intersects
  // with a BVH aggregate if and only if it intersects a primitive in
  // the BVH that is not an aggregate. When an intersection does happen.
  // You should store the non-aggregate primitive in the intersection data
  // and not the BVH aggregate itself.

  bool hit = false;
  isect->t = ray.max_t;

  std::stack<BVHNode *> s;
  s.push(root);
  while (!s.empty()) {
    auto u = s.top();
    s.pop();

    double t0, t1;
    if (u->bb.intersect(ray, t0, t1)) {
      if (t0 <= isect->t) {
        if (u->isLeaf()) {
          for (size_t i = 0; i < u->range; i++) {
            if (primitives[u->start + i]->intersect(ray, isect)) {
              hit = true;
              ray.max_t = isect->t;
            }
          }
        } else {
          double lt0, lt1;
          bool lhit = false;
          if (u->l) lhit = u->l->bb.intersect(ray, lt0, lt1);
          double rt0, rt1;
          bool rhit = false;
          if (u->r) rhit = u->r->bb.intersect(ray, rt0, rt1);

          if (lhit && rhit) {
            if (lt0 < rt0) {
              s.push(u->r);
              s.push(u->l);
            } else {
              s.push(u->l);
              s.push(u->r);
            }
          } else if (lhit) {
            s.push(u->l);
          } else if (rhit) {
            s.push(u->r);
          }
        }
      }
    }
  }

  return hit;
}

}  // namespace StaticScene
}  // namespace CMU462
