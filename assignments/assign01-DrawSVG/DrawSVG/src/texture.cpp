#include "texture.h"
#include "color.h"

#include <assert.h>
#include <iostream>
#include <algorithm>

using namespace std;

namespace CMU462 {

inline void uint8_to_float( float dst[4], unsigned char* src ) {
  uint8_t* src_uint8 = (uint8_t *)src;
  dst[0] = src_uint8[0] / 255.f;
  dst[1] = src_uint8[1] / 255.f;
  dst[2] = src_uint8[2] / 255.f;
  dst[3] = src_uint8[3] / 255.f;
}

inline void float_to_uint8( unsigned char* dst, float src[4] ) {
  uint8_t* dst_uint8 = (uint8_t *)dst;
  dst_uint8[0] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[0])));
  dst_uint8[1] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[1])));
  dst_uint8[2] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[2])));
  dst_uint8[3] = (uint8_t) ( 255.f * max( 0.0f, min( 1.0f, src[3])));
}

void Sampler2DImp::generate_mips(Texture& tex, int startLevel) {

  // NOTE: 
  // This starter code allocates the mip levels and generates a level 
  // map by filling each level with placeholder data in the form of a 
  // color that differs from its neighbours'. You should instead fill
  // with the correct data!

  // Task 7: Implement this

  // check start level
  if ( startLevel >= tex.mipmap.size() ) {
    std::cerr << "Invalid start level"; 
  }

  // allocate sublevels
  int baseWidth  = tex.mipmap[startLevel].width;
  int baseHeight = tex.mipmap[startLevel].height;
  int numSubLevels = (int)(log2f( (float)max(baseWidth, baseHeight)));

  numSubLevels = min(numSubLevels, kMaxMipLevels - startLevel - 1);
  tex.mipmap.resize(startLevel + numSubLevels + 1);

  int width  = baseWidth;
  int height = baseHeight;
  for (int i = 1; i <= numSubLevels; i++) {

    MipLevel& level = tex.mipmap[startLevel + i];

    // handle odd size texture by rounding down
    width  = max( 1, width  / 2); assert(width  > 0);
    height = max( 1, height / 2); assert(height > 0);

    level.width = width;
    level.height = height;
    level.texels = vector<unsigned char>(4 * width * height);

  }

  // fill all 0 sub levels with interchanging colors
  Color colors[3] = { Color(1,0,0,1), Color(0,1,0,1), Color(0,0,1,1) };
  for(size_t i = 1; i < tex.mipmap.size(); ++i) {

    Color c = colors[i % 3];
    MipLevel& mip = tex.mipmap[i];

    for(size_t i = 0; i < 4 * mip.width * mip.height; i += 4) {
      float_to_uint8( &mip.texels[i], &c.r );
    }
  }

}

Color Sampler2DImp::sample_nearest(Texture& tex, 
                                   float u, float v, 
                                   int level) {
    // Task 6: Implement nearest neighbour interpolation

    // return magenta for invalid level
    if (level >= tex.mipmap.size())
        return Color(1,0,1,1);

    int su = (int) floor(clamp(u, 0.0f, 0.99999f) * tex.mipmap[level].width);
    int sv = (int) floor(clamp(v, 0.0f, 0.99999f) * tex.mipmap[level].height);

    float r = tex.mipmap[level].texels[4 * (su + sv * tex.mipmap[level].width)] / 255.0f;
    float g = tex.mipmap[level].texels[4 * (su + sv * tex.mipmap[level].width) + 1] / 255.0f;
    float b = tex.mipmap[level].texels[4 * (su + sv * tex.mipmap[level].width) + 2] / 255.0f;
    float a = tex.mipmap[level].texels[4 * (su + sv * tex.mipmap[level].width) + 3] / 255.0f;

    return Color(r, g, b, a);
}

Color Sampler2DImp::sample_bilinear(Texture& tex, 
                                    float u, float v, 
                                    int level) {
    // Task 6: Implement bilinear filtering

    // return magenta for invalid level
    if (level >= tex.mipmap.size())
        return Color(1,0,1,1);

    float tu = clamp(u, 0.0f, 0.99999f) * tex.mipmap[level].width;
    float tv = clamp(v, 0.0f, 0.99999f) * tex.mipmap[level].height;

    int su[2];
    su[0] = clamp<int>(round(tu) - 1, 0, tex.mipmap[level].width - 1);
    su[1] = clamp<int>(su[0] + 1, 0, tex.mipmap[level].width - 1);
    float du = tu - 0.5f - su[0];

    int sv[2];
    sv[0] = clamp<int>(round(tv) - 1, 0, tex.mipmap[level].height - 1);
    sv[1] = clamp<int>(sv[0] + 1, 0, tex.mipmap[level].height - 1);
    float dv = tv - 0.5f - sv[0];

    Color mix = Color(0, 0, 0, 0);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            float r = tex.mipmap[level].texels[4 * (su[i] + sv[j] * tex.mipmap[level].width)] / 255.0f;
            float g = tex.mipmap[level].texels[4 * (su[i] + sv[j] * tex.mipmap[level].width) + 1] / 255.0f;
            float b = tex.mipmap[level].texels[4 * (su[i] + sv[j] * tex.mipmap[level].width) + 2] / 255.0f;
            float a = tex.mipmap[level].texels[4 * (su[i] + sv[j] * tex.mipmap[level].width) + 3] / 255.0f;
            Color c = Color(r * a, g * a, b * a, a);
            mix += (i * du + (1 - i) * (1 - du)) * (j * dv + (1 - j) * (1 - dv)) * c;
        }
    }

    if (mix.a != 0) {
        mix.r /= mix.a;
        mix.g /= mix.a;
        mix.b /= mix.a;
    }
    return mix;
}

Color Sampler2DImp::sample_trilinear(Texture& tex, 
                                     float u, float v, 
                                     float u_scale, float v_scale) {

  // Task 7: Implement trilinear filtering

  // return magenta for invalid level
  return Color(1,0,1,1);

}

} // namespace CMU462
