// Given a time between 0 and 1, evaluates a cubic polynomial with
// the given endpoint and tangent values at the beginning (0) and
// end (1) of the interval.  Optionally, one can request a derivative
// of the spline (0=no derivative, 1=first derivative, 2=2nd derivative).
template <class T>
inline T Spline<T>::cubicSplineUnitInterval(
    const T& position0, const T& position1, const T& tangent0,
    const T& tangent1, double normalizedTime, int derivative) {
  // p_0 h_00 = 2t^3 - 3t^2 + 1
  //    h_00' = 6t^2 - 6t
  //   h_00'' = 12t - 6
  // m_0 h_10 = t^3 - 2t^2 + t
  //    h_10' = 3t^2 - 4t + 1
  //   h_10'' = 6t - 4
  // p_1 h_01 = -2t^3 + 3t^2
  //    h_01' = -6t^2 + 6t
  //   h_01'' = -12t + 6
  // m_1 h_11 = t^3 - t^2
  //    h_11' = 3t^2 - 2t
  //   h_11'' = 6t - 2

  double tl = normalizedTime;
  double ts = normalizedTime * tl;
  double tc = normalizedTime * ts;
  if (derivative == 0) {
    T h00 = (2 * tc - 3 * ts + 1) * position0;
    T h10 = (tc - 2 * ts + tl) * tangent0;
    T h01 = (-2 * tc + 3 * ts) * position1;
    T h11 = (tc - ts) * tangent1;
    return h00 + h10 + h01 + h11;
  } else if (derivative == 1) {
    T h00 = (6 * ts - 6 * tl) * position0;
    T h10 = (3 * ts - 4 * tl + 1) * tangent0;
    T h01 = (-6 * ts + 6 * tl) * position1;
    T h11 = (3 * ts - 2 * tl) * tangent1;
    return h00 + h10 + h01 + h11;
  } else if (derivative == 2) {
    T h00 = (12 * tl - 6) * position0;
    T h10 = (6 * tl - 4) * tangent0;
    T h01 = (-12 * tl + 6) * position1;
    T h11 = (6 * tl - 2) * tangent1;
    return h00 + h10 + h01 + h11;
  }
  return T();
}

// Returns a state interpolated between the values directly before and after the
// given time.
template <class T>
inline T Spline<T>::evaluate(double time, int derivative) {
  if (knots.size() < 1) {
    return T();
  } else if (knots.size() == 1) {
    if (derivative == 0) return knots.begin()->second;
    else return T();
  } else {
    double start = knots.begin()->first, end = knots.rbegin()->first;
    time = clamp(time, start, end);

    auto it = knots.upper_bound(time);
    if (it == knots.end()) --it;

    double k2 = it->first;
    const T& p2 = it->second;

    --it;
    double k1 = it->first;
    const T& p1 = it->second;

    auto it0 = it;
    bool isStart = it0 == knots.begin();
    double k0 = isStart ? k1 - (k2 - k1) : (--it0)->first;
    const T& p0 = isStart ? p1 - (p2 - p1) : it0->second;

    auto it1 = it;
    ++it1; ++it1;
    bool isEnd = it1 == knots.end();
    double k3 = isEnd ? k2 + (k2 - k1) : it1->first;
    const T& p3 = isEnd ? p2 + (p2 - p1) : it1->second;

    double len = k2 - k1;
    const T& m1 = (p2 - p0) / (k2 - k0) * len;
    const T& m2 = (p3 - p1) / (k3 - k1) * len;
    return cubicSplineUnitInterval(p1, p2, m1, m2, (time - k1) / len, derivative);
  }
}

// Removes the knot closest to the given time,
//    within the given tolerance..
// returns true iff a knot was removed.
template <class T>
inline bool Spline<T>::removeKnot(double time, double tolerance) {
  // Empty maps have no knots.
  if (knots.size() < 1) {
    return false;
  }

  // Look up the first element > or = to time.
  typename std::map<double, T>::iterator t2_iter = knots.lower_bound(time);
  typename std::map<double, T>::iterator t1_iter;
  t1_iter = t2_iter;
  t1_iter--;

  if (t2_iter == knots.end()) {
    t2_iter = t1_iter;
  }

  // Handle tolerance bounds,
  // because we are working with floating point numbers.
  double t1 = (*t1_iter).first;
  double t2 = (*t2_iter).first;

  double d1 = fabs(t1 - time);
  double d2 = fabs(t2 - time);

  if (d1 < tolerance && d1 < d2) {
    knots.erase(t1_iter);
    return true;
  }

  if (d2 < tolerance && d2 < d1) {
    knots.erase(t2_iter);
    return t2;
  }

  return false;
}

// Sets the value of the spline at a given time (i.e., knot),
// creating a new knot at this time if necessary.
template <class T>
inline void Spline<T>::setValue(double time, T value) {
  knots[time] = value;
}

template <class T>
inline T Spline<T>::operator()(double time) {
  return evaluate(time);
}
