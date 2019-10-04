#include "viewport.h"

#include "CMU462.h"

namespace CMU462 {

void ViewportImp::set_viewbox( float centerX, float centerY, float vspan ) {

    // Task 5 (part 2):
    // Set svg coordinate to normalized device coordinate transformation. Your input
    // arguments are defined as normalized SVG canvas coordinates.
    this->centerX = centerX;
    this->centerY = centerY;
    this->vspan = vspan;

    // Matrix3x3:
    // 00 10 20
    // 01 11 21
    // 02 12 22
    svg_2_norm = Matrix3x3::identity();
    svg_2_norm[0][0] = 0.5 / vspan;
    svg_2_norm[1][1] = 0.5 / vspan;
    svg_2_norm[2][0] = 0.5 - 0.5 * centerX / vspan;
    svg_2_norm[2][1] = 0.5 - 0.5 * centerY / vspan;
}

void ViewportImp::update_viewbox( float dx, float dy, float scale ) { 
  
  this->centerX -= dx;
  this->centerY -= dy;
  this->vspan *= scale;
  set_viewbox( centerX, centerY, vspan );
}

} // namespace CMU462
