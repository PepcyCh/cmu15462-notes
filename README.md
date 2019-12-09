# Notes for CMU 15-462/662

Site: [Computer Graphics (cmu 15-462) fall2018](http://15462.courses.cs.cmu.edu/fall2018/)

## Todo

### Notes

* [x] 01 Course Introduction
* [x] 02 Math Review Part I: Linear Algebra
* [x] 03 Math Review Part II: (Vector Calculus)
* [x] 04 Drawing a Triangle
* [x] 05 Coordinate Spaces and Transformations
* [x] 06 3D Rotations and Complex Representations
* [x] 07 Perspective Projection and Texture Mapping
* [x] 08 Depth and Transparency
* [x] 09 Introduction to Geometry
* [x] 10 Meshes and Manifolds
* [x] 11 Digital Geometry Processing
* [x] 12 Geometric Queries
* [x] 13 Spatial Data Structures
* [x] 14 Color
* [x] 15 Radiometry
* [x] 16 The Rendering Equation
* [x] 17 Numerical Integration
* [x] 18 Monte Carlo Ray Tracing
* [x] 19 Variance Reduction
* [x] 20 Introduction to Animation
* [ ] 21 Dynamics and Time Integration
* [ ] 22 Introduction to Optimization
* [ ] 23 Physically-Based Animation and PDEs

### Assignments

* [x] 1.0 draw svg
  * [x] 1 Hardware Renderer
  * [x] 2 Warm Up: Drawing Lines
  * [x] 3 Drawing Triangles
  * [x] 4 Anti-Aliasing Using Supersampling
  * [x] 5 Implementing Modeling and Viewing Transforms
  * [x] 6 Drawing Scaled Images
  * [x] 7 Anti-Aliasing Image Elements Using Trilinear Filtering
  * [x] 8 Alpha Compositing
  * [x] 9 Draw Something!!!
* [x] 2.0 MeshEdit (But lots of bugs remain unfixed)
  * [x] Local Operations
    * [x] `VertexBevel`
    * [x] `EdgeBevel`
    * [x] `FaceBevel`
    * [x] `EraseVertex`
    * [x] `EraseEdge`
    * [x] `EdgeCollapse`
    * [x] `FaceCollapse`
    * [x] `EdgeFlip`
    * [x] `EdgeSplit`
  * [x] Global Operations
    * [x] `Triangulation`
    * [x] `LinearSubdivision`
    * [x] `CatmullClarkSubdivision`
    * [x] `LoopSubdivision` - depends on `EdgeSplit` and `EdgeFlip`
    * [x] `IsotropicRemeshing` - depends on `EdgeSplit`, `EdgeFlip`, and `EdgeCollapse`
    * [x] `Simplification` - depends on `EdgeCollapse`
* [x] 3.0 PathTracer
  * [x] 1 - Generating Camera Rays
  * [x] 2 - Intersecting Triangles and Spheres
  * [x] 3 - Implementing a Bounding Volume Hierarchy (BVH)
  * [x] 4 - Implementing Shadow Rays
  * [x] 5 - Adding Path Tracing
  * [x] 6 - Adding New Materials
  * [x] 7 - Infinite Environment Lighting
* [ ] 4.0 Animation
  * [x] 1 - Spline Interpolation
  * [ ] 2 - Skeleton Kinematics
  * [ ] 3 - Linear Blend Skinning
  * [ ] 4 - Phisycal Simulation
