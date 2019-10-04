#include "software_renderer.h"

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>

#include "triangulation.h"

using namespace std;

namespace CMU462 {


// Implements SoftwareRenderer //

void SoftwareRendererImp::draw_svg( SVG& svg ) {

  // set top level transformation
  transformation = svg_2_screen;

  // draw all elements
  for ( size_t i = 0; i < svg.elements.size(); ++i ) {
    draw_element(svg.elements[i]);
  }

  // draw canvas outline
  Vector2D a = transform(Vector2D(    0    ,     0    )); a.x--; a.y--;
  Vector2D b = transform(Vector2D(svg.width,     0    )); b.x++; b.y--;
  Vector2D c = transform(Vector2D(    0    ,svg.height)); c.x--; c.y++;
  Vector2D d = transform(Vector2D(svg.width,svg.height)); d.x++; d.y++;

  rasterize_line(a.x, a.y, b.x, b.y, Color::Black);
  rasterize_line(a.x, a.y, c.x, c.y, Color::Black);
  rasterize_line(d.x, d.y, b.x, b.y, Color::Black);
  rasterize_line(d.x, d.y, c.x, c.y, Color::Black);

  // resolve and send to render target
  resolve();

}

void SoftwareRendererImp::set_sample_rate( size_t sample_rate ) {

    // Task 4:
    // You may want to modify this for supersampling support
    this->sample_rate = sample_rate;
    supersampling = sample_rate > 1;
    if (supersampling) {
        this->supersample_target.resize(4 * target_w * target_h * sample_rate * sample_rate);
    }
}

void SoftwareRendererImp::set_render_target( unsigned char* render_target,
                                             size_t width, size_t height ) {

    // Task 4:
    // You may want to modify this for supersampling support
    this->render_target = render_target;
    this->target_w = width;
    this->target_h = height;

    if (supersampling) {
        this->supersample_target.resize(4 * width * height * sample_rate * sample_rate);
    }
}

void SoftwareRendererImp::draw_element( SVGElement* element ) {

  // Task 5 (part 1):
  // Modify this to implement the transformation stack

  Matrix3x3 temp_matrix = transformation;
  transformation = transformation * element->transform;

  switch(element->type) {
    case POINT:
      draw_point(static_cast<Point&>(*element));
      break;
    case LINE:
      draw_line(static_cast<Line&>(*element));
      break;
    case POLYLINE:
      draw_polyline(static_cast<Polyline&>(*element));
      break;
    case RECT:
      draw_rect(static_cast<Rect&>(*element));
      break;
    case POLYGON:
      draw_polygon(static_cast<Polygon&>(*element));
      break;
    case ELLIPSE:
      draw_ellipse(static_cast<Ellipse&>(*element));
      break;
    case IMAGE:
      draw_image(static_cast<Image&>(*element));
      break;
    case GROUP:
      draw_group(static_cast<Group&>(*element));
      break;
    default:
      break;
  }

  transformation = temp_matrix;
}


// Primitive Drawing //

void SoftwareRendererImp::draw_point( Point& point ) {

  Vector2D p = transform(point.position);
  rasterize_point( p.x, p.y, point.style.fillColor, true );

}

void SoftwareRendererImp::draw_line( Line& line ) { 

  Vector2D p0 = transform(line.from);
  Vector2D p1 = transform(line.to);
  rasterize_line( p0.x, p0.y, p1.x, p1.y, line.style.strokeColor );

}

void SoftwareRendererImp::draw_polyline( Polyline& polyline ) {

  Color c = polyline.style.strokeColor;

  if( c.a != 0 ) {
    int nPoints = polyline.points.size();
    for( int i = 0; i < nPoints - 1; i++ ) {
      Vector2D p0 = transform(polyline.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polyline.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_rect( Rect& rect ) {

  Color c;
  
  // draw as two triangles
  float x = rect.position.x;
  float y = rect.position.y;
  float w = rect.dimension.x;
  float h = rect.dimension.y;

  Vector2D p0 = transform(Vector2D(   x   ,   y   ));
  Vector2D p1 = transform(Vector2D( x + w ,   y   ));
  Vector2D p2 = transform(Vector2D(   x   , y + h ));
  Vector2D p3 = transform(Vector2D( x + w , y + h ));
  
  // draw fill
  c = rect.style.fillColor;
  if (c.a != 0 ) {
    rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    rasterize_triangle( p2.x, p2.y, p1.x, p1.y, p3.x, p3.y, c );
  }

  // draw outline
  c = rect.style.strokeColor;
  if( c.a != 0 ) {
    rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    rasterize_line( p1.x, p1.y, p3.x, p3.y, c );
    rasterize_line( p3.x, p3.y, p2.x, p2.y, c );
    rasterize_line( p2.x, p2.y, p0.x, p0.y, c );
  }

}

void SoftwareRendererImp::draw_polygon( Polygon& polygon ) {

  Color c;

  // draw fill
  c = polygon.style.fillColor;
  if( c.a != 0 ) {

    // triangulate
    vector<Vector2D> triangles;
    triangulate( polygon, triangles );

    // draw as triangles
    for (size_t i = 0; i < triangles.size(); i += 3) {
      Vector2D p0 = transform(triangles[i + 0]);
      Vector2D p1 = transform(triangles[i + 1]);
      Vector2D p2 = transform(triangles[i + 2]);
      rasterize_triangle( p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, c );
    }
  }

  // draw outline
  c = polygon.style.strokeColor;
  if( c.a != 0 ) {
    int nPoints = polygon.points.size();
    for( int i = 0; i < nPoints; i++ ) {
      Vector2D p0 = transform(polygon.points[(i+0) % nPoints]);
      Vector2D p1 = transform(polygon.points[(i+1) % nPoints]);
      rasterize_line( p0.x, p0.y, p1.x, p1.y, c );
    }
  }
}

void SoftwareRendererImp::draw_ellipse( Ellipse& ellipse ) {

  // Extra credit 

}

void SoftwareRendererImp::draw_image( Image& image ) {

  Vector2D p0 = transform(image.position);
  Vector2D p1 = transform(image.position + image.dimension);

  rasterize_image( p0.x, p0.y, p1.x, p1.y, image.tex );
}

void SoftwareRendererImp::draw_group( Group& group ) {

  for ( size_t i = 0; i < group.elements.size(); ++i ) {
    draw_element(group.elements[i]);
  }

}

// Rasterization //

// The input arguments in the rasterization functions 
// below are all defined in screen space coordinates

void SoftwareRendererImp::rasterize_point( float x, float y, Color color, bool point_or_line = false ) {
    // fill in the nearest pixel
    int sx = (int) floor(x);
    int sy = (int) floor(y);

    // check bounds
    if ( sx < 0 || sx >= target_w ) return;
    if ( sy < 0 || sy >= target_h ) return;

    // fill sample - NOT doing alpha blending!
    if (!supersampling) {
        render_target[4 * (sx + sy * target_w)] = (uint8_t) (color.r * 255);
        render_target[4 * (sx + sy * target_w) + 1] = (uint8_t) (color.g * 255);
        render_target[4 * (sx + sy * target_w) + 2] = (uint8_t) (color.b * 255);
        render_target[4 * (sx + sy * target_w) + 3] = (uint8_t) (color.a * 255);
    } else if (point_or_line) {
        sx *= sample_rate;
        sy *= sample_rate;
        for (int i = 0; i < sample_rate; i++) {
            for (int j = 0; j < sample_rate; j++) {
                supersample_target[4 * (sx + j + (sy + i) * target_w * sample_rate)] = (uint8_t) (color.r * 255);
                supersample_target[4 * (sx + j + (sy + i) * target_w * sample_rate) + 1] = (uint8_t) (color.g * 255);
                supersample_target[4 * (sx + j + (sy + i) * target_w * sample_rate) + 2] = (uint8_t) (color.b * 255);
                supersample_target[4 * (sx + j + (sy + i) * target_w * sample_rate) + 3] = (uint8_t) (color.a * 255);
            }
        }
    } else {
        sx = (int) floor(x * sample_rate);
        sy = (int) floor(y * sample_rate);
        supersample_target[4 * (sx + sy * target_w * sample_rate)] = (uint8_t) (color.r * 255);
        supersample_target[4 * (sx + sy * target_w * sample_rate) + 1] = (uint8_t) (color.g * 255);
        supersample_target[4 * (sx + sy * target_w * sample_rate) + 2] = (uint8_t) (color.b * 255);
        supersample_target[4 * (sx + sy * target_w * sample_rate) + 3] = (uint8_t) (color.a * 255);
    }
}

void SoftwareRendererImp::rasterize_line( float x0, float y0,
                                          float x1, float y1,
                                          Color color) {
    // Task 2:
    // Implement line rasterization
    float dx = abs(x1 - x0), dy = abs(y1 - y0);
    if (dy <= dx) {
        if (x0 == x1) return;
        if (x0 > x1) {
            swap(x0, x1);
            swap(y0, y1);
        }
        float k = (y1 - y0) / (x1 - x0);
        float x = floor(x0) + 0.5;
        float y = y0 + k * (x - x0);
        for (; x <= x1; x += 1) {
            float y2 = floor(y) == round(y) ? y - 1 : y + 1;
            float d = abs(y + 0.5 - round(y + 0.5));
            rasterize_point(x, y, color * (1 - d), true);
            rasterize_point(x, y2, color * d, true);
            y += k;
        }
    } else {
        if (y0 == y1) return;
        if (y0 > y1) {
            swap(y0, y1);
            swap(x0, x1);
        }
        float k = (x1 - x0) / (y1 - y0);
        float y = floor(y0) + 0.5;
        float x = x0 + k * (y - y0);
        for (; y <= y1; y += 1) {
            float x2 = floor(x) == round(x) ? x - 1 : x + 1;
            float d = abs(x + 0.5 - round(x + 0.5));
            rasterize_point(x, y, color * (1 - d), true);
            rasterize_point(x2, y, color * d, true);
            x += k;
        }
    }
}

void SoftwareRendererImp::rasterize_triangle( float x0, float y0,
                                              float x1, float y1,
                                              float x2, float y2,
                                              Color color ) {
    // Task 3:
    // Implement triangle rasterization
    int minx = (int) floor(min({x0, x1, x2}));
    int maxx = (int) floor(max({x0, x1, x2}));
    int miny = (int) floor(min({y0, y1, y2}));
    int maxy = (int) floor(max({y0, y1, y2}));

    float dx0 = x1 - x0, dy0 = y1 - y0;
    float dx1 = x2 - x1, dy1 = y2 - y1;
    float dx2 = x0 - x2, dy2 = y0 - y2;
    float rot = dx0 * dy1 - dy0 * dx1;

    float pd = 1.0f / sample_rate;
    for (int x = minx; x <= maxx; x++) {
        for (int y = miny; y <= maxy; y++) {
            for (int i = 0; i < sample_rate; i++) {
                for (int j = 0; j < sample_rate; j++) {
                    float px = (i + 0.5f) * pd, py = (j + 0.5f) * pd;
                    float e0 = (y + py - y0) * dx0 - (x + px - x0) * dy0;
                    float e1 = (y + py - y1) * dx1 - (x + px - x1) * dy1;
                    float e2 = (y + py - y2) * dx2 - (x + px - x2) * dy2;
                    if (e0 * rot >= 0 && e1 * rot >= 0 && e2 * rot >= 0)
                        rasterize_point(x + px, y + py, color);
                }
            }
        }
    }
}

void SoftwareRendererImp::rasterize_image( float x0, float y0,
                                           float x1, float y1,
                                           Texture& tex ) {
    // Task 6:
    // Implement image rasterization
    float dx = x1 - x0;
    float dy = y1 - y0;

    float pd = 1.0f / sample_rate;
    for (int x = (int) floor(x0 * sample_rate); x <= (int) floor(x1 * sample_rate); x++) {
        for (int y = (int) floor(y0 * sample_rate); y <= (int) floor(y1 * sample_rate); y++) {
            float u = ((x + 0.5f) * pd - x0) / dx;
            float v = ((y + 0.5f) * pd - y0) / dy;
            // Color c = sampler->sample_nearest(tex, u, v, 0);
            Color c = sampler->sample_bilinear(tex, u, v, 0);
            rasterize_point((x + 0.5f) * pd, (y + 0.5f) * pd, c);
        }
    }
    /*
    for (float x = floor(x0 + 0.5f * pd) + 0.5f * pd; x <= x1; x += pd) {
        for (float y = floor(y0 + 0.5f * pd) + 0.5f * pd; y <= y1; y += pd) {
            float u = (x - x0) / dx;
            float v = (y - y0) / dy;
            // Color c = sampler->sample_nearest(tex, u, v, 0);
            Color c = sampler->sample_bilinear(tex, u, v, 0);
            rasterize_point(x, y, c);
        }
    }
     */
}

// resolve samples to render target
void SoftwareRendererImp::resolve( void ) {

    // Task 4:
    // Implement supersampling
    // You may also need to modify other functions marked with "Task 4".
    if (!supersampling) return;

    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            float sumr = 0;
            float sumg = 0;
            float sumb = 0;
            float suma = 0;
            for (int i = 0; i < sample_rate; i++) {
                for (int j = 0; j < sample_rate; j++) {
                    sumr += supersample_target[4 * (x * sample_rate + j + (y * sample_rate + i) * sample_rate * target_w)];
                    sumg += supersample_target[4 * (x * sample_rate + j + (y * sample_rate + i) * sample_rate * target_w) + 1];
                    sumb += supersample_target[4 * (x * sample_rate + j + (y * sample_rate + i) * sample_rate * target_w) + 2];
                    suma += supersample_target[4 * (x * sample_rate + j + (y * sample_rate + i) * sample_rate * target_w) + 3];
                }
            }
            render_target[4 * (x + y * target_w)] = (uint8_t) (sumr / sample_rate / sample_rate);
            render_target[4 * (x + y * target_w) + 1] = (uint8_t) (sumg / sample_rate / sample_rate);
            render_target[4 * (x + y * target_w) + 2] = (uint8_t) (sumb / sample_rate / sample_rate);
            render_target[4 * (x + y * target_w) + 3] = (uint8_t) (suma / sample_rate / sample_rate);
        }
    }
}


} // namespace CMU462
