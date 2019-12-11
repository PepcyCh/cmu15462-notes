# Notes for CMU 15-462

> site: [COMPUTER GRAPHICS (CMU 15-462/662) Fall2018](http://15462.courses.cs.cmu.edu/fall2018/)

## Lect 1 Course Introduction

> site: [Course Introduction](http://15462.courses.cs.cmu.edu/fall2018/lecture/intro)

### Foundations of Computer Graphics

* **Theory**
  * basic representations (how do you digitally encode shape, motion?)
  * sampling & aliasing (how do you acquire & reproduce a signal?)
  * numerical methods (how do you manipulate signals numerically?)
  * radiometry & light transport (how does light behave?)
  * perception (how does this all relate to humans?)
  * ...
* **Systems**
  * parallel, heterogeneous processing
  * graphics-specific programming languages
  * ...

### Draw a Cube

Perspective projection

### Draw a Line on Computer

Rasterization

Diamond rule: light up if line passes through associated diamond

[Direct 3D :: Rasterization Rules](https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules)

## Lect 2 Math Review Part I: Linear Algebra

> site: [Math Review Part I: Linear Algebra](http://15462.courses.cs.cmu.edu/fall2018/lecture/linearalgebra)

### Linear & Affine

Linear: $f(x + y) = f(x) + f(y)$

Affine functions can be turned to linear functions through homogeneous coordinates.

## Lect 3 Math Review Part II: (Vector Calculus)

> site: [Math Review Part II: (Vector Calculus)](http://15462.courses.cs.cmu.edu/fall2018/lecture/vectorcalc)

### Norm

### Inner Product / Dot Product

### Cross Product

Matrix form:
$$
\mathbf{u} \times \mathbf{v} = \hat{\mathbf{u}} \mathbf{v} \\
\hat{\mathbf{u}} = \begin{bmatrix}
0 & -u_3 & u_2 \\
u_3 & 0 & -u_1 \\
-u_2 & u_1 & 0
\end{bmatrix}
$$

### Determinant & Triple Product

$\det(\mathbf{u}, \mathbf{v}, \mathbf{w}) = (\mathbf{u} \times \mathbf{v}) \cdot \mathbf{w}$

#### Determinant of Linear Function

Changes in volume, whether the oritation is reversed.

#### Other Triple Product

**Jacobi's Identity: ** $\mathbf{u} \times (\mathbf{v} \times \mathbf{w}) + \mathbf{v} \times (\mathbf{w} \times \mathbf{u}) + \mathbf{w} \times (\mathbf{u} \times \mathbf{v}) = 0$

**Lagrange's Identity: ** $\mathbf{u} \times (\mathbf{v} \times \mathbf{w}) = \mathbf{v} (\mathbf{u} \cdot \mathbf{w}) - \mathbf{w} (\mathbf{u} \cdot \mathbf{v})$

### Derivative

Many functions in graphics are not differentable.

Best linear approximation (derivative & gradient).

#### Gradient of Matrix-Valued Expressions

$$
\nabla_{\mathbf{x}} (\mathbf{x}^T A \mathbf{y}) = A\mathbf{y} \\
\nabla_{\mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = 2A\mathbf{x} \\
\dots \\
(\mathbf{x}, \mathbf{y} \in \mathbb{R}^n, A \in \mathbb{R}^{n \times n} \text{ and is symmetric})
$$

### Divergence & Curl

#### Divergence

（散度）How much is field shrinking / expanding.
$$
\text{div } X = \nabla \cdot X
$$

#### Curl

（旋度）How much is field spinning.
$$
\text{curl } X = \nabla \times X
$$

#### Divergence & Curl

$$
\nabla \cdot X = \nabla \times X^{\bot}
$$

（$X^{\bot}$ is the $90$-degree rotation of $X$）

### Laplacian

Laplacian measures "curvature" of a function.

It maps a scalar function to scalar function linearly.
$$
\Delta f = \nabla \cdot \nabla f = \text{div}(\text{grad } f)
$$
By analogy, graph(picture) lapalcian: $\Delta f(x, y) = f(x + 1, y) + f(x - 1, y) + f(x, y + 1) + f(x, y - 1) - 4f(x, y)$

### Hessian

$$
(\nabla^2 f) \mathbf{u} = D_u (\nabla f) \\
\nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1 \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \dots & \frac{\partial^2 f}{\partial x_n \partial x_n} \\
\end{bmatrix}
$$

## Lect 4 Drawing a Triangle

> site: [Drawing a Triangle](http://15462.courses.cs.cmu.edu/fall2018/lecture/drawingatriangle)

Two major techniques for "getting stuff on the screen": Rasterization & Ray Tracing

### Triangle

#### Why

* can approximate any shape
* always planar
* easy to interpolate (barycentric coordinates)

**Key reason**: once everything is reduced to triangles, can focus on making an **extremely well-optimized pipeline** for drawing them.

### Rasterization Pipeline

1. Transform/position objects in the world
2. Project objects onto the screen
3. Sample triangle coverage
4. Interpolate triangle attributes at covered samples
5. Sample texture maps / evaluate shaders
6. Combine samples into final image (depth, alpha)

### Sampling

**Sampling**: measurement of a signal

**Reconstruction**: generating signal from a discrete set of samples

### Sampling for Coverage

"Top edge" and "left edge"

**Top edge**: horizontal edge that is above all other edges

**Left edge**: an edge that is not horizontak and is on the left side of triangle

### Aliasing

High frequencies in the original signal masquerade as low frequencies after reconstruction (due to undersampling).

Jaggies, roping or shimmering, moire pattern

### Resampling

### Point-in-triangle Test

If the three points $P_0, P_1, P_2$ are arranged counterclockwisely.
$$
P_i = (X_i, Y_i) \\
dX_i = X_{i + 1} - X_i, dY_i = Y_{i + 1} - Y_i \\
E_i(x, y) = (x - X_i)dY_i - (y - Y_i)dX_i \\
E_i \begin{cases}
= 0 ,& \text{point on edge} \\
> 0 ,& \text{outside edge} \\
< 0 ,& \text{inside edge}
\end{cases}
$$
So, we have
$$
\text{inside}(sx, sy) = E_0(sx, sy) < 0 \and E_0(sx, sy) < 0 \and E_0(sx, sy) < 0
$$
Actual implememtations of $\text{inside}(sx, sy)$ involves $\leq$ checks based on the coverage rules (top edge & left edge).

### Triangle Traversal

#### Incremental Triangle Traversal

$$
E_i(x, y) = (x - X_i)dY_i - (y - Y_i)dX_i = A_ix + B_iy + C_i \\
E_i(x + 1, y) = E_i(x, y) + A_i \\
E_i(x, y + 1) = E_i(x, y) + B_i
$$

#### Modern Approach: Tiled Triangle Traversal

Traverse triangle in blocks. Test all samples in block against triangle in parallel.

"early in" and "early out"

## Lect 5 Coordinate Spaces and Transformations

> site: [Coordinate Spaces and Transformations](http://15462.courses.cs.cmu.edu/fall2018/lecture/transformations)

### Transformations

#### Linear

Scale, rotation, reflection, shear

#### Affine

Translation

#### Euclidean (Isometries)

Translation, rotation, reflection

### Representing Transformations with Matrices

### Homogeneous Coordinates

Homogeneous coordinates let us encode translations as linear transformations (2D translation as 3D shear).

### Screen Transformation

(reflect +) translate + scale

## Lect 6 3D Rotations and Complex Representations

> site: [3D Rotations and Complex Representations](http://15462.courses.cs.cmu.edu/fall2018/lecture/3drotations)

### Euler Angles

#### Gimbal Lock

When using Euler angles $\theta_x, \theta_y, \theta_z$, may reach a situation where there is no way to rotate around one of the three axes:
$$
R_x R_y R_z = \begin{bmatrix}
\cos \theta_y \cos \theta_z & -\cos \theta_y \sin \theta_z & \sin \theta_y \\
\cos \theta_z \sin \theta_x \sin \theta_y + \cos \theta_x \sin \theta_z & \cos \theta_x \cos \theta_z - \sin \theta_x \sin \theta_y \sin \theta_z & -\cos \theta_y \sin \theta_x \\
-\cos \theta_x \cos \theta_z \sin \theta_y + \sin \theta_x \sin \theta_z & \cos \theta_z \sin \theta_x + \cos \theta_x \sin \theta_y \sin \theta_z & \cos \theta_x \cos \theta_y
\end{bmatrix}
$$
When $\theta_y = \pi / 2$:
$$
R_x R_y R_z = \begin{bmatrix}
0 & 0 & 1 \\
\cos \theta_z \sin \theta_x + \cos \theta_x \sin \theta_z & \cos \theta_x \cos \theta_z - \sin \theta_x \sin \theta_z & 0 \\
-\cos \theta_x \cos \theta_z + \sin \theta_x \sin \theta_z & \cos \theta_z \sin \theta_x + \cos \theta_x \sin \theta_z & 0\\
\end{bmatrix}
$$

### Rotations around a Given Axis $u$ by a Given Angle $\theta$

### Complex Representation for 2D Rotations

### Quaternions

$$
a + x\imath + y\jmath + zk = (a, \mathbf{v}), \mathbf{v} = (x, y, z) \\
(a, \mathbf{u}) (b, \mathbf{v}) = (ab - \mathbf{u} \cdot \mathbf{v}, a\mathbf{u} + b\mathbf{v} + \mathbf{u} \times \mathbf{v}) \\
\mathbf{u}\mathbf{v} = (-\mathbf{u} \cdot \mathbf{v}, \mathbf{u} \times \mathbf{v}) = \mathbf{u} \times \mathbf{v} - \mathbf{u} \cdot \mathbf{v}
$$

Let $q$ be a unit quaternion and $\mathbf{x}$ be a vector (pure imaginary), then $\bar{q}\mathbf{x}q$ always expresses some 3D rotation.

More precisely, if $q = \cos(\theta / 2) + \sin(\theta / 2) \mathbf{u}$, then $\bar{q}\mathbf{x}q$ is the vector after $\mathbf{x}$ is rotated around axis $\mathbf{u}$ by $\theta$.

### Interpolating Rotations

SLERP (Spherical linear interpolation):
$$
\text{Slerp}(q_0, q_1, t) = q_0(q_0^{-1}q_1)^t, ~ t \in [0, 1]
$$

## Lect 7 Perspective Projection and Texture Mapping

> site: [Perspective Projection and Texture Mapping](http://15462.courses.cs.cmu.edu/fall2018/lecture/texture)

### From Objects to the Screen

1. **World coordinates**: original desciption of objects
2. **View coordinates**: all positions now expressed relative to camera; camera is sitting at origin looking down -z direction
3. **Clip coordinates**: everything visible to the camera is mapped to unit cube for easy clipping
4. **Nomalized coordinates**: unit cube mapped to unit square via perspective divide
5. **Window coordinates**: screen transformation

### Perspective Projection

View frustum

Clipping

Z-fighting

Matrix for perspective projection

### Triangle Interpolation

#### Barycentric Coordinates

$$
\phi_i(x) = \frac{\text{area}(x, x_j, x_k)}{\text{area}(x_i, x_j, x_k)} \\
f(x) = f(x_i) \phi_i(x) + f(x_j) \phi_j(x) + f(x_k) \phi_k(x)
$$

#### Perspective Correct Interpolation

Perspective transformation is not affine.

Divide interpolated $f(x) / z$ by interpolated $1/z$.

### Texture Mapping

#### Texture Coordinates

#### Mipmap

Mipmap level:
$$
\frac{\text{d}u}{\text{d}x} = u_{10} - u_{00}, \frac{\text{d}v}{\text{d}x} = v_{10} - v_{00} \\
\frac{\text{d}u}{\text{d}y} = u_{01} - u_{00}, \frac{\text{d}v}{\text{d}y} = v_{01} - v_{00} \\
L = \max\left( \sqrt{ \left(\frac{\text{d}u}{\text{d}x}\right)^2 + \left(\frac{\text{d}v}{\text{d}x}\right)^2 }, \sqrt{ \left(\frac{\text{d}u}{\text{d}y}\right)^2 + \left(\frac{\text{d}v}{\text{d}y}\right)^2 } \right) \\
\text{mipmap d} = \log_2 L
$$
Tri-linear filtering

## Lect 8 Depth and Transparency

> site: [Depth and Transparency](http://15462.courses.cs.cmu.edu/fall2018/lecture/depthtransparency)

### Depth-buffer (Z-buffer)

### Transparency

#### Over operator

"Over" is not commutative.
$$
A = \begin{bmatrix}A_r & A_g & A_b\end{bmatrix}^T \\
B = \begin{bmatrix}B_r & B_g & B_b\end{bmatrix}^T \\
B \text{ over } A = C = \alpha_B B + (1 - \alpha_B) \alpha_A A
$$

#### Premultiplied alpha

**Non-premultiplied alpha**:
$$
A = \begin{bmatrix}A_r & A_g & A_b\end{bmatrix}^T \\
B = \begin{bmatrix}B_r & B_g & B_b\end{bmatrix}^T \\
B \text{ over } A = C = \alpha_B B + (1 - \alpha_B) \alpha_A A
$$
**Premultiplied alpha**:
$$
A' = \begin{bmatrix}\alpha_A A_r & \alpha_A A_g & \alpha_A A_b & \alpha_A\end{bmatrix}^T \\
B' = \begin{bmatrix}\alpha_B B_r & \alpha_B B_g & \alpha_B B_b & \alpha_B\end{bmatrix}^T \\
C' = B' + (1 - \alpha_B) A'
$$
**Composite alpha**:
$$
\alpha_C = \alpha_B + (1 - \alpha_B)\alpha_A
$$
Premultiplied alpha composites alpha just like how it composies rgb. Non-premutiplied alpha composites alpha differently than rgb.

#### Problems Caused by Non-premultiplied Alpha

Use premultiplied alpha when up- and downsampling.

Non-premultipled alpha is not closed under composition. (The result will be premultipled alpha.)

### Review on Rasterization Pipeline

1. Transform triangle vertices into camera space.
2. Apply perspective projection transform to transform vertices into normalized coordinate space.
3. Clipping.
4. Transform to screen coordinates.
5. Setup triangle. (Before rasterizing triangle, can compute a bunch of data that will be used by all fragments, such as triangl edge equations, triangle attribute equations etc.)
6. Sample coverage.
7. Compute triangle color at sample point.
8. Perform depth test.
9. Update color buffer.

### Graphics Pipeline

* Structures rendering computation as a sequence of operations performed on vertices, primitives (e.g., triangles), fragments, and screen samples.
* Behavior of parts of the pipeline is application-defined using shader programs.
* Pipeline operations implemented by highly, optimized parallel processors and fixed-function hardware (GPUs).

## Lect 9 Introduction to Geometry

> site: [Introduction to Geometry](http://15462.courses.cs.cmu.edu/fall2018/lecture/introgeometry)

### Representation of Geometry

#### Implicit

$f(x, y, z) = 0$

It makes some tasks hard (like sampling), but makes other tasks easy (like inside/outside tests).

#### Explicit

Traingle meshes, polygon meshes, subdivision surfaces, NURBS, point clouds, ...

It makes some tasks easy (like sampling), but makes other tasks hard (like inside/outside tests).

### Some Implicit Representations

#### Algebraic Surfaces

Surface is zero set of a polynomial in $x, y, z$.

$f(x, y, z) = 0$

#### Constructive Solid Geometry

Build more complicated shapes via Boolean operations (union, intersection, difference).

#### Blobby Surfaces

Instead of Booleans, gradually blend surfaces together.

#### Blending Distance Functions

A *distance function* gives distance to closest point on object.

Can blend any two distance functions $d_1, d_2$.

$D_1 \cup D_2 : f(x) = \min(d_1(x), d_2(x))$

#### Level Set Methods

Implicit surfaces have some nice features (e,g, merging/splitting). But, hard to describle complex shapes in closed form.

Alternative: store a grid of values approximating function. Surface is found where interpolated values equal zero.

Provides much more explicit control over shape (like a texture).

Often demands sophisticated filtering (trilinear, tricubic...).

**Level set storage**: Storage for 2D now is $O(n^3)$. Can reduce cost by storing only a narrow band around surface.

#### Fractal

#### Pros & Cons 

**Pros:** 

- description can be very compact (e.g., a polynomial) 
- easy to determine if a point is in our shape (just plug it in!) 
- other queries may also be easy (e.g., distance to surface) 
- for simple shapes, exact description/no sampling error 
- easy to handle changes in topology (e.g., fluid) 

**Cons:** 

- expensive to find all points in the shape (e.g., for drawing) 
- very difficult to model complex shapes 

### Some Explicit Representations

#### Point Cloud

Easiest representation.

Easily represent any kind of geometry.

Hard to interpolate undersample regions. Hard to do processing/simulation...

#### Polygon Mesh

Easier to do processing/simulation, adaptive sampling.

#### Triangle Mesh

 Store vertices and indices.

#### Bernstein Basis

$$
B_{n, k} (x) = \binom{n}{k} x^k (1 - x)^{n - k} \\
(0 \leq x \leq 1, ~ k = 0, 1, \dots, n, ~ 0^0 = 1)
$$

#### Bézier Curves

$$
\gamma(s) = \sum_{i = 0}^{n} B_{n, k} (s) \mathbf{P_k}
$$

$n = 1$: a line segment

$n = 3$: cubic Bézier

Important feartues:

* interpolates endpoints
* tangent to end segments
* contained in convex hull (nice to rasterization)

High-degree Bernstein polynomials don't interpolate well.

#### Piecewise Bézier Curves

Alternative idea: piece together many Bézier curves.
$$
\gamma(u) = \gamma_i(u) \left( \frac{u - u_i}{u_{i + 1} - u_i} \right) ~ (u_i \leq u < u_{i + 1})
$$
**Tangent continuity**

To get "seamless" curves, need points and tangents to line up.

#### Tensor Product

Can use a pair of curves to get a surface.
$$
(f \otimes g) (u, v) = f(u)g(v)
$$

#### Bézier Patches

Bézier patch is sum of tensor products of Berstein bases.
$$
B_{3, i, j} (u, v) = B_{3, i}(u) B_{3, j}(v) \\
S(u, v) = \sum_{i = 0}^{3} \sum_{j = 0}^{3} B_{3, i, j}(u, v) \mathbf{P_{ij}}
$$

#### Bézier Surfaces

#### Spline Patch Schemes

Tradeoffs:

* degrees of freedom
* continuity
* difficulty of editing
* cost of evaluation
* generality
* ...

### Subdivisions

1. Start with control curve. 
2. Insert new vertex at each edge midpoint. 
3. Update vertex positions according to fixed rule.
4. For careful choice of averaging rule, yields smooth curve.

#### Subdivision Surfaces (Explicit)

Start with coarse polygon mesh (control cage) and subdivide each element.

Many possible rules: Catmull-Clark (quads), Loop (triangles)...

Easier than splines for modeling, harder to evaluate pointwise.

## Lect 10 Meshes and Manifolds

> site: [Meshes and Manifolds](http://15462.courses.cs.cmu.edu/fall2018/lecture/meshes)

### Manifolds

(Manifolds：流形)

A surface is the boundary of an object.

Surfaces are manifolds. (If you zoom in far enough at any point look like a plane. / Can easily be flattened into a plane.)

#### Manifold Polygon Mesh:

Every edge is contained in (at most) only two polygons.

The polygons containing each vertex make a single "fan".

#### Boundary

Globally, each boundary forms a loop.

For polygon mesh, one polygon per boundary edge.

### Storage

#### Polygon Soup

Just array of vertices and indices.

#### Incidence Matrices

Incidence matrices between vertices and edges, edges and faces.

#### Halfedge

Linked list.

Halfedges act as "glu" between elements (vertices, edges, faces).

Each vertex, edge or face just points to one of its halfedge.

Traversal is easy.

Halfedge meshes are always manifolds.

**Operations on Halfedge**: flip, split, collapse

#### Other Data Structrues

* Winged edge
* Corner table
* Quadedge
* ...

### Subdivision Modeling

Common modeling paradigm in modern 3D tooIs:
- Coarse "control cage"
- Perform local operations to control/edit shape
- Global subdivision process determines final surface

#### Local Operations

* Edge bevel
* Face bevel
* Vertex bevel
* Edge collapse
* Face collapse
* Edge flip
* Edge split
* Erase edge
* Erase vertex

## Lect 11 Digital Geometry Processing

> site: [Digital Geometry Processing](http://15462.courses.cs.cmu.edu/fall2018/lecture/geometryprocessing)

### Geometry Processing

#### Reconstruction

Given samples of geometry, reconstruct surface.

Samples:

* points, points & normals...
* image pair / set (multi-view video)
* line density integrals (MRI/CT scans)

How to get a surface:

* silhouette-based (visual hull)
* Voronoi-based (e.g. power crust)
* PDE-based (e.g. Poisson reconstruction)
* Radon transform / isosurfacing (marching cubes)

#### Upsampling

Increase resolution via interpolation.

Images: bilinear, bicubic...

Polygon meshes:

* subdivision
* bilateral upsampling

#### Downsampling

Decrease resolution, try to preserve shape/appearence

Image: nearest, bilinear, bicubic

Point clouds: subsampling

Polygon meshes:

* iterative decimation
* variational shape appoximation

#### Resampling

Modify sample distribution to improve quality

Image: not an issue

Polygon meshes: shape of polygons is extremely important

* different notion of quality depending on task
* visualization & solving equation

#### Filtering

Remove noise or emphasize important features (e.g. edges)

Image: blurring, bilateral filtering, edge detection

Polygon meshes:

* curvature flow
* bilateral filter
* spectral filter

#### Compression

Image: run-length, Huffman; cosine, wavelet

Polygon meshes:

* compress geometry and connectivity
* lossy & lossless

#### Shape Analysis

Identity / understand important semantic features

Image: CV, segmentation, face detection

Polygon meshes:

* segmentation
* correspondence
* symmetry detection

### What makes a "good" mesh

#### Good appoximation of original shape

Keep only elements that contribute information about shape.

The vertices of a mesh are very close to the surface it approximates doesn't mean it's a good approximation. (e.g. normals)

#### Triangle shape

Delaunay / All angles close to 60 degree

Regular vertex degree

### Upsampling

#### Upsampling via Subdivision

Repeatedly split each element into smaller pieces.

Replace vertex positions with weighted average of neighbors.

Main considerations:

* interpolation vs. approximation
* limit surface continuity
* behavior at irregular vertex

Quad: Catmull-Clark

Triangle: Loop, Butterfly, Sqrt(3)

#### Catmull-Clark Subdivision

**Step 0:** split every polygon indo quadrilaterals

New vertex position are weighted combination of old ones.

**Step 1:** Face coords: average of vertices

**Step 2:** Edge coords: average of the two neighbouring face points and its two original endpoints.

**Step 3:** Vertex coords: $\frac{F + 2R + (n - 3)P}{n}$, $F$: average of face points, $R$: average of edge mid-points, $P$: original points, $n$: degree

Good for quad meshes but bad for triangle meshes.

#### Loop Subdivision

Split each triangle into four, and assign new vertex positions according to weights.

Can be implemented by edge-split and edge-flip

### Simplification

#### Simplification via Edge Collapse

Greedy:

* assign each edge a cost
* collapse edge with least cost
* repeat until target number of elements is reached

Particularly effective cost function: quadric error metric

#### Quadric Error Metric

Distance to a collection of triangles: sum of point-plane distances, $\sum N_i \cdot (x - p)$

A query point $(x, y, z)$, a normal $(a, b, c)$, an offset $d := -(p, q, r) \cdot (a, b, c)$

In homogeneous coordinates, let $u = (x, y, z, 1), v = (a, b, c, d)$, then

Signed distance: $u \cdot v = ax + by + cz + d$

Squared distance: $(u \cdot v)^2 = u^T(v v^T)u \overset{\triangle}= u^T K u$

Key idea: matrix $K$ encodes distance to plane.

Measure cost of edge-collapse:

* compute edge midpoint, measure quadric error
* use point that minimizes quadric error as a new point

### Remeshing

#### "More Delaunay"

Flip edge when $\alpha + \beta > \pi$

#### Improve degree

Edge flip.

(average valance of any triangle mesh is (about) 6)

#### "More round"

Can often improve shape by centering vertices.

Move only in tangent direction (direction orthogonal to surface normal)

#### Isotropic Remeshing Algorithm

Try to make triangles uniform shape & size

Repeat four steps:

- Split any edge over 4/3 mean edge legth
- Collapse any edge less than 4/5 mean edge length
- Flip edges to improve vertex degree
- Centervertices tangentially

## Lect 12 Geometric Queries

> site: [Geometric Queries](http://15462.courses.cs.cmu.edu/fall2018/lecture/geometricqueries)

### Closest point

#### To a point

#### To a line

$$
\text{line: } \mathbf{N}^T\mathbf{x} = c \\
\mathbf{N}^T(\mathbf{p} + t\mathbf{N}) = c \\
t = c - \mathbf{N}^T \mathbf{p}
$$

#### To a segment

#### To a triangle

#### To a plane

the same to "to a line"

#### To a mesh

### Ray-Mesh Intersection

$$
\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}
$$

#### Implicit Surface

$f(\mathbf{r}(t)) = 0$

#### Plane

$$
\mathbf{N}^T(\mathbf{o} + t \mathbf{d}) = c \\
t = \frac{c - \mathbf{N}^T \mathbf{o}}{\mathbf{N}^T \mathbf{d}}
$$

#### Triangle

Just calculate the intersection of ray-plane and check if the point is inside the triangle by barycentric coordinate.

Faster algorithm exists.

### Mesh-Mesh Intersection

#### Point-point intersection

#### Point-line intersection

#### Line-line intersection

What if lines are almost parallel?

#### Triangle-triangle intersection

Reduce to edge-triangle intersection.

What if triangle is moving?

* Consider it as a prism in time
* Turn to 4-D intertsection problem

## Lect 13 Spatial Data Structures

> site: [Spatial Data Structures](http://15462.courses.cs.cmu.edu/fall2018/lecture/spatialdatastructures)

### Ray-Mesh Intersection

#### Triangle

Triangle can be represented by $\mathbf{p_0} + u(\mathbf{p_1} - \mathbf{p_0}) + v(\mathbf{p_2} - \mathbf{p_0})$ .
$$
\mathbf{p_0} + u(\mathbf{p_1} - \mathbf{p_0}) + v(\mathbf{p_2} - \mathbf{p_0}) = \mathbf{o} + t\mathbf{d} \\
\begin{bmatrix}
\mathbf{p_1} - \mathbf{p_0} & \mathbf{p_2} - \mathbf{p_0} & -\mathbf{d}
\end{bmatrix}
\begin{bmatrix}
u \\
v \\
t
\end{bmatrix} = \mathbf{o} - \mathbf{p_0}
$$

#### Axis-aligned box

Calculate $t_{\text{min}}$ and $t_{\text{max}}$ for each direction, then $[\max(t_{\text{min}}), \min(t_{\text{max}})]$ is the intersection interval.

Can also work if ray misses the box, $\max(t_{\text{min}})$ will be greater than $\min(t_{\text{max}})$.

#### Scene

Check each primitive takes $O(n)$ time.

Can be done faster?

Check ray-bounding-box intersection first. But still need to check all the primitives if ray hits the bounding box.

Apply this strategy hierarchically.

### Bounding Volume Hierarchy (BVH)

BVH partitions each node's primitives into disjoisnt sets. But these sets may overlap in space.

"front-to-back" traversal: only traverse the farther child if the calculated $t$ in the nearer child is greater than that calculated with bounding box of the farther child.

#### Build a high-quality BVH

want small bounding boxes (minimize overlap between children, avoid empty space)

for a leaf node: $C = \sum_{i = 1}^{N} C_{\text{isect}(i)} = NC_{\text{isect}}$ ($C_{\text{isect}}(i)$ is  the cost of ray-primitive intersection for primitive $i$)

for an interior node: $C = C_{\text{trav}} + p_AC_A + p_BC_B$  ($p_A$ is the probablity a ray intersects with the bbox of child A)

Tp get hitting probablity: for convex object A inside convex object B, we have:
$$
P(\text{hit A} \mid \text{hit B}) = \frac{S_A}{S_B}
$$
, where $S_A$ is the surface areas of A. Therefore, (surface area heuristic - SAH)
$$
C = C_{\text{trav}} + \frac{S_A}{S_N}N_AC_{\text{isect}} + \frac{S_B}{S_N}N_BC_{\text{isect}}
$$

#### Implenting Partitions

Basic ideas:

- Choose an axis, choose a split plane on that axis
- Partition primitives by the side of splitting plane their centroid lies
- SAH changes only when split plane moves past triangle boundary
- Have to consider rather large number of possible split planes...

A more efficent way: split spatial extent of primitives into $B$ buckets by the position of their centroids ($B$ is typically small: $B < 32$).

need to consider only $3(B - 1)$ possible partitioning.

#### Troublesome cases

* All primitives with the same centroid
* All primitives with the same bbox

### Space-partitioning structures

#### Primitive-partitioning accelertion structures vs. Space-partitioning structures

* Primitive partitioning (bounding volume hierarchy) partitions node's primitives into disjoint sets (but sets may overlap in space)
* Space-partitioning (grid, K-D tree) partitions space into disjoint regions (primitives may be contained in multiple regions of space)

#### K-D Tree

Recursively partition space via axis-aligned partitioning planes
- Interior nodes correspond to spatial splits
- Node traversal can proceed in front-to-back order
- Unlike BVH, can terminate search after first hit is found

If a primitive overlaps multiple nodes, early break may leads to incorrect result.

Solution: require primitive intersection point to be within current leaf.

#### Unifrom grid

* Partition space into equal sized volumes (volume-elements or "voxels")
* Each grid cell contains primitives that overlap voxel.
* Walk ray through volume in order
  * Very efficient implementation possible (just like 3D line rasterization)
  * Only consider intersection with primitives in voxels the ray intersects

Grid resolution: Choose number of voxels ~ total number of primitives (constant prims per voxel if uniform distribution of primitives holds)

Time complexity: $O(\sqrt[3]{N})$ on average.

Uniform grid cannot adapt to non-uniform distribution of geometry in scene. ("teapot in a stadium")

#### Quad-tree / Octree

Easy to build and has greater ability to adapt to location of scene geometry than uniform grid.

But lower intersection performance than K-D tree.

### Basic Rasterization vs. Ray Casting

* **Rasterization**
  * Proceeds in triangle order
  * Store depth buffer (random access to regular structure of fixed size)
  * Don't have to store entire scene in memory, naturally supports unbounded size scenes
* **Ray casting**
  * Proceeds in screen sample order
    * Don't have to store closest depth so far for the entire screen (just current ray)
    * Natural order for rendering transparent surfaces (process surfaces in the order the are encountered along the ray: front-to-back or back-to-front)
  * Must store entire scene
  * Performance more strongly depends on distribution of primitives in scene
* **Modern high-performance implementations of rasterization and ray-casting embody very similar techniques**
  * Hierarchies of rays/samples
  * Hierarchies of geometry
  * Deferred shading

## Lect 14 Color

> site: [Color](http://15462.courses.cs.cmu.edu/fall2018/lecture/color)

Light is EM Radiation. Color is Frequency.

### Additive vs. Subtractive Models of Lights

emission spectrum & absorb spectrum

#### Emission Spectrum

Describes light intensity as a function of frequency.

#### Absorbed Spectrum

Fraction absorbed as function of frequency.

### Interaction of Emission and Reflection

Light source emission spectrum: $f(\nu)$

Surface reflection spectrum: $g(\nu)$

Result: $f(\nu) g(\nu)$

### EM Radiance to Color

#### Photosensor

EM power distribution over wavelength: $\Phi(\lambda)$

Sensity of sensor: $f(\lambda)$

Responce: $R = \int_{\lambda} f(\lambda) \Phi(\lambda) \mathrm{d}\lambda$

#### Human

Rods and cones.

Three types of cones: S, M, L
$$
S = \int_{\lambda} \Phi(\lambda) S(\lambda) \text{d} \lambda \\
M = \int_{\lambda} \Phi(\lambda) M(\lambda) \text{d} \lambda \\
L = \int_{\lambda} \Phi(\lambda) L(\lambda) \text{d} \lambda \\
$$

#### Metamers

(metamer: 条件等色体) Two different spectra that integral to the same $(S, M, L)$ response.

We don't have to reproduce the exact same spectrum that was present in a real world scene in order to reproduce the perceived color on a monitor (or piece of paper, or paint on a wall).

A usage: counterfeit detection, yeilds different appearence under UV light.

### Color Spaces and Color Models

#### Additive vs. Subtractive Color Models

Additive:

* use for combining colored lights
* RGB

subtractive (multiplicative)

* use for combining paint color
* CMYK

#### Other color spaces

HSV (Hue, Saturation, Value)

* more intuitive than RGB/CMYK

SML

* physiological model
* not practical for most color work

XYZ

* preceptually-driven model
* related to but different from SML

Lab

YCbCr

### CIE Color Space

Standard reference color space.

#### Chromaticity Diagram

#### sRGB

#### Color Acuity (MacAdam Ellipse)

Each ellipse corresponds to a region of "just noticeable differences" of color (chromaticity).

Try to avoid overlapping ellipses.

### Gamma Correction

Non-linear correction for old CRT display.

Doesn't apply to modern LCD displays, whose luminance output is linearly proportional to
input.

DOES still apply to other devices, like sensors, etc.

## Lect 15 Radiometry

> site: [Radiometry](http://15462.courses.cs.cmu.edu/fall2018/lecture/radiometry)

* System of units and measures for measuring EM radiation (light)
* Geometric optics model of light
  * Photons travel in straight lines
  * Represented by rays
  * Wavelength << size of objects
  * No diffraction, interference, …

### Concepts

Names don’t constitute knowledge!

Energy of photons hitting an object ~ “brightness

#### Radiant energy

total of hits

#### Radiant flux

hits per second

#### Irradiance

hits per second, per unit area

Image generation as irradiance estimation.

#### Color

irradiance per unit wavelength

### Measuring illumination

#### Radiant energy

$$
Q = h \nu = \frac{hc}{\lambda}
$$

#### Radiant flux

$$
\Phi = \frac{\mathrm{d}Q}{\mathrm{d}t}
$$

#### Irradiance

$$
E(p) = \frac{\mathrm{d}\Phi(p)}{\mathrm{d}A}
$$

#### Lambert's law

$$
E = \frac{\Phi}{A'} = \frac{\Phi \cos \theta}{A}
$$

**"N-dot-L" lighting**

#### Isotropic point source

$$
I = \frac{\Phi}{4\pi}
$$

##### Irradiance falloff with distance

$\Phi$ keeps unchanged.
$$
E_1 = \frac{\Phi}{4 \pi r_1^2}, \Phi = 4 \pi r_1^2 E_1 \\
E_2 = \frac{\Phi}{4 \pi r_2^2}, \Phi = 4 \pi r_2^2 E_2 \\
\frac{E_1}{E_2} = \left( \frac{r_2}{r_1} \right)^2
$$

#### Solid Angle

$$
\Omega = \frac{A}{r^2} \\
\mathrm{d} \omega = \frac{\mathrm{d}A}{r^2} = \sin \theta \mathrm{d} \theta \mathrm{d} \phi
$$

#### Radiance

Solid angle density of irradiance.
$$
L(p, \omega) = \frac{\mathrm{d}E_{\omega}(p)}{\mathrm{d}\omega} = \frac{\mathrm{d} E(p)}{\cos \theta \mathrm{d} \omega} = \frac{\mathrm{d}^2\Phi(p)}{\mathrm{d}A \mathrm{d}\omega \cos \theta}
$$
radiance is constant along rays.

#### Spectral radiance

In general, $L_i(\mathbf{p_i}, \omega) \neq L_o(\mathbf{p_o}, \omega)$.

### Properties of radiance

* Radiance is a fundamental field quantity that characterizes the distribution of light in an environment
  * Radiance is the quantity associated with a ray
  * Rendering is all about computing radiance
* Radiance is constant along a ray (in a vacuum)
* A pinhole camera measures radiance

### Ambient occlusion

### Irradiance calculation

#### Irradiance from the enviroment

$$
E(p) = \int_{H^2} L_i(p_i, \omega) \cos \theta \mathrm{d} \omega
$$

If $L$ is constant:
$$
E(p) = \int_{H^2} L \cos \theta \mathrm{d}\omega = L \int_{0}^{2 \pi} \int_{0}^{\pi / 2}\cos \theta \sin \theta \mathrm{d} \theta \mathrm{d} \phi = L \pi
$$

#### Irradiance from a uniform area source

$$
E(p) = \int_{\Omega} L \cos \theta \mathrm{d} \omega = L \Omega^{\bot}
$$

#### Uniform disk source (oriented perpendicular to plane)

$$
\Omega^{\bot} = \pi \sin^2 \alpha
$$

## Lect 16 The Rendering Equation

> site: [The Rendering Equation](http://15462.courses.cs.cmu.edu/fall2018/lecture/renderingequation)

Core functionality of photorealistic renderer is to estimate radiance at a given point $p$, in a given direction $\omega_o$.
$$
L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_{\mathcal{H}^2}f_r(\mathbf{p}, \omega_i \to \omega_o) L_i(\mathbf{p}, \omega_i) \cos \theta \mathrm{d}\omega_i
$$

* $\mathbf{p}$: point of interest
* $\omega_o$: direction of interest
* $L_o$: outgoing / observed radiance
* $L_e$: emitted radiance
* $L_i$: incoming radiance
* $f_r$: scattering function
* $\omega_i$: incoming direction
* $\theta$: angle between incoming direction and normal

Key challenge: to evaluate incoming radiance, we have to compute yet another integral. I.e., rendering equation is recursive.

### Reflection

Reflection is the process by which light incident on a surface interacts with the surface such that it leaves on the incident (same) side without change in frequency.

Choice of reflection function determines surface appearance.

### Scattering

In general, can talk about probability a particle arriving from a given direction is scattered in another direction.

At any point on any surface in the scene, there's an incident radiance field that gives the directional distribution of illumination at the point.

#### Scattering off a surface: BRDF

* BidirectionaI reflectance distribution function
* Encodes behavior of light that "bounces off" surface 
* Given incoming direction $\omega_i$, how much light gets scattered in any given outgoing direction $\omega_o$
* Describe as distribution $f_r(\omega_i \to \omega_o)$.

$$
f_r(\omega_i \to \omega_o) \geq 0 \\
\int_{\mathcal{H}^2} f_r(\omega_i \to \omega_o) \cos \theta \mathrm{d}\omega_i \leq 1 \\
f_r(\omega_i \to \omega_o) = f_r(\omega_o \to \omega_i)
$$

#### Radiometric description of BRDF:

$$
f_r(\omega_i \to \omega_o) = \frac{\mathrm{d}L_o(\omega_o)}{\mathrm{d}E_i(\omega_i)} = \frac{\mathrm{d}L_o(\omega_o)}{\mathrm{d}L_i(\omega_i) \cos \theta_i}
$$

For a given change in the incident irradiance, how much does the exitant radiance change.

#### Examples

##### Lambertian

$$
L_o(\omega_o) = \int_{H^2} f_r L_i(\omega_i) \cos \theta_i \mathrm{d}\omega_i = f_r E \\
f_r = \frac{\rho}{\pi} \; (0 \leq \rho \leq 1)
$$

$\rho$: albedo

##### Specular

$$
\omega_o = -\omega_i + 2(\omega_i \cdot \vec{n}) \vec{n} \\
f_r (\theta_i, \phi_i; \theta_o, \phi_o) = \frac{\delta(\cos \theta_i - \cos \theta_o)}{\cos \theta_i} \delta(\phi_i - \phi_o \pm \pi)
$$

Strictly speaking, $f_r$ is a distribution, not a function 

In practice, no hope of finding reflected direction via random sampling. Simply pick the reflected direction!

### Refraction

#### Snell's law

$\eta_i \sin \theta_i = \eta_t \sin \theta_t$

#### Fresnel reflection

Reflectance increases with viewing angle.

### Anisotropic reflection

Reflection depends on azimuthal angle $\phi$.

### Subsurcaface scattering

Visual characteristics of many surfaces caused by light entering at different points than it exits.

* Violates a fundamental assumption of the BRDF 
* Need to generalize scattering model (BSSRDF) 

Generalization of BRDF.

Describes exitant radiance at one point due to incident differential irradiance at another point.
$$
S(x_i, \omega_i, x_o, \omega_o)
$$
Generalization of reflection equation integrates over all points on the surface and all directions.
$$
L(x_o, \omega_o) = \int_A \int_{H^2} S(x_i, \omega_i, x_o, \omega_o) L_i(x_i, \omega_i) \mathrm{d}\omega_i \mathrm{d}A
$$

### Approximate integral

Approximate integral via Monte Carlo integration.

## Lect 17 Numerical Integration

> site: [Numerical Integration](http://15462.courses.cs.cmu.edu/fall2018/lecture/numericalintegration)

### Gauss Quadrature

For any polynomial of degree $n$, we can always obtain the exact integral by sampling at a special set of $n$ points and taking a special weighted combination.

### Trapezoid rule

$$
I = \frac{h}{2} (f(a) + f(b) + 2\sum_{i = 1}^{n - 1}f(x_i)) + O(h^2) \\
= \sum_{i = 0}^{n} A_i f(x_i) + O(h^2)
$$

2D:
$$
\begin{align}
I &= \int_{a_y}^{b_y} \int_{a_x}^{b_x} f(x, y) \mathrm{d} x \mathrm{d} y \\
&= \int_{a_y}^{b_y} (\sum_{i = 0}^{n}A_i f(x_i, y) + O(h^2)) \mathrm{d}y \\
&= O(h^2) + \sum_{i = 0}^{n} A_i \int_{a_y}^{b_y} f(x_i, y) \mathrm{d}y \\
&= O(h^2) + \sum_{i = 0}^{n} A_i \sum_{j = 0}^{n} (A_j f(x_i, y_j) + O(h^2)) \\
&= \sum_{i = 0}^n\sum_{j = 0}^n A_i A_j f(x_i, y_j) + O(h^2)
\end{align}
$$
still $O(h^2)$ but $O(1 / n)$, and $O(n^2)$ work.

### Monte Carlo Integration

* Estimate value of integral using random sampling of function
  * Value of estimate depends on random samples used 
  * But algorithm gives the correct value of integral "on average" 
* Only requires function to be evaluated at random points on its domain
  * Applicable to functions with discontinuities, functions that are impossible to integrate directly 
* Error of estimate is independent of the dimensionality ofthe integrand 
  * Depends on the number of random samples used

### Random variables

#### PDF and CDF

#### Generate samples with expect to a PDF

If $\xi \sim U(0, 1)$, then

Discrete:

Select $x_i$ if $P_{i - 1} < \xi \leq P_i$

Continuous:

$x = P^{-1}(\xi)$

#### Uniformly sample a unit circle

$$
A = \int_{0}^{2\pi} \int_{0}^{1} r \mathrm{d}\theta\mathrm{d}r = \pi \\
\int_{0}^{2\pi} \int_{0}^{1} p(\theta, r) \mathrm{d}\theta\mathrm{d}r = 1 \\
p(\theta, r) = \frac{r}{\pi} \\
p(\theta) = \frac{1}{2\pi}, p(r) = 2r \\
\theta = 2\pi \xi_1, r = \sqrt{\xi_2}
$$

Another way (rejection method): just generate uniform samples in $[-1, 1]^2$ and reject if not in unit circle.

## Lect 18 Monte Carlo Ray Tracing

> site: [Monte Carlo Ray Tracing](http://15462.courses.cs.cmu.edu/fall2018/lecture/montecarloraytracing)

* Basic idea: take average of random samples 
* Will need to flesh this idea out with some key concepts: 
  * EXPECTED VALUE - what value do we get on average? 
  * VARIANCE - what's the expected deviation from the average? 
  * IMPORTANCE SAMPLING - how do we (correctly) take more samples in more important regions? 

$$
\int_{\Omega} f(x) \mathrm d{x} \approx \frac{\vert\Omega\vert}{N}\sum_{i = 1}^{N} f(X_i)
$$

### Importance Sampling

What if $X_i \sim p(x)$ instead of $X_i \sim U(\Omega)$:
$$
\int_{\Omega} f(x) \mathrm{d}x \approx \frac{\vert \Omega \vert}{N} \sum_{i = 1}^{N} \frac{f(X_i)}{p(X_i)}
$$
basic idea: put more samples where integrand is large.

#### Direct light

Uniform sampled scattered light leads to noises: Incident lighting estimator uses different random directions in each pixel. Some of those directions point towards the light, others do not.

Don't need to integrate over entire hemisphere of directions (incoming radiance is 0 from most directions). Just integrate over the area of the light (directions where incoming radiance is non-zero) and weight appropriately.

Area integral:
$$
E(p) = \int_{A'} L_o(p', p - p') V(p, p') \frac{\cos \theta \cos \theta'}{\vert p - p' \vert^2} \mathrm{d} A' \\
(\mathrm{d} \omega = \frac{\mathrm{d}A}{\vert p - p' \vert^2} = \frac{\mathrm{d} A'\cos \theta'}{\vert p - p' \vert^2})
$$

### Cosine-weighted Sampling

$$
p(\omega) = \frac{\cos \theta}{\pi} \\
\int_{\Omega} L_i(\omega) \cos \theta \mathrm{d} \omega \approx \frac{1}{N} \sum_{i = 1}^{N} \frac{L(X_i) \cos \theta}{p(X_i)} = \frac{\pi}{N} \sum_{i = 1}^{N} L(X_i)
$$

### Rossian roulette

Want to avoid spending time evaluating function for samples that make a small contribution to the final result.

Ignoring low-contribution samples introduces systematic error: no longer converges to correct value!

Instead, randomly discard low-contribution samples in a way that leaves estimator unbiased: evaluate original estimator with probability $p_{rr}$, reweight. Otherwise ignore. 

## Lect 19 Variance Reduction

> site: [Variance Reduction](http://15462.courses.cs.cmu.edu/fall2018/lecture/variancereduction)

You can't reduce variance of the integrand! Can only reduce variance of an estimator. (An "estimator" is a formula used to approximate an integral)

### Consistency and Bias

consistency:
$$
\lim_{n \to \infty} P(\vert I - \hat{I_n} \vert > \varepsilon) = 0
$$
bias:
$$
\lim_{n \to \infty} E(\vert I - \hat{I_n} \vert) = 0
$$

### Path Space Integral

$$
I = \int_{\Omega} f(\bar{x}) \mathrm{d} \mu(\bar{x})
$$

* $\Omega$: all possible paths
* $f$: how much "light" is carried by this path
* $\mu$: how much of space does this path "cover"
* $\bar{x}$: one particular path

### Bidirectional Path Tracing

Idea: connect paths from light

### Metropoils-Hastings Algorithm

Take random walk of dependent samples ("mutations") 

Basic idea: prefer to take steps that increase sample value 

Want to take samples proportional to density $f$.

Start at random point; take steps in (normal) random direction.

Occasionally jump to random point (ergodicity)

Transition probability is "relative darkness": $f(x') / f(x_i)$

### Multiple Importance Sampling

### Sampling Patterns and Variance Reduction

Sampling patterns will affect variance.

#### Stratified Sampling

#### Low-discrepancy Sampling

Discrepancy measures deviation from its ideal.
$$
d_S(X) = \vert A(S) - \frac{n(S)}{\vert X \vert} \vert \\
D(X) = \max_\limits{S \in F} d_S(X)
$$
ideally, $D(X) = 0$.

Replace truly random samples with low-discrepancy samples. Koksma's theory:
$$
\vert \frac{1}{N}\sum_{i = 1}^{N} f(x_i) - \int_{0}^{1} f(x) \mathrm{d}x \vert \leq \nu(f) D(X)
$$
$\nu(f)$: total variation of $f$, integration of $f'$.

##### Hammersley and Halton Points

$n$ Halton points in $k$-dimensions:
$$
x_i = (\phi_{P_1}(i), \phi_{P_2}(i), \cdots, \phi_{P_k}(i))
$$
$n$ Hammersley points in $k$-dimensions:
$$
x_i = (\frac{i}{n}, \phi_{P_1}(i), \cdots, \phi_{P_{k - 1}}(i))
$$
$\phi_r(i)$: radical inverse

$P_k$: $k$-th prime number

#### Regular grid ?

Even low-discrepancy patterns can exhibit poor behavior. (e.g. sampling a regular black-white check pattern)

Want pattern to be anisotropic (no preferred direction)

Also want to avoid any preferred frequency (see above!)

#### Blue Noise

#### Poisson Disk Sampling

Iteratively add random non-overlapping disks until no space left.

#### Lloyd Relaxation

Iteratively move each disk to the center of its neighbors.

#### Voronoi-Based Methods

Natural evolution of Lloyd

Optimize qualities of this Voronoi diagram

#### Adaptive Blue Noise

### Sampling from CDF

Sample ways: $O(n \log n)$

#### Alias Table

rob the rich, give to the poor - cost $O(n)$

each column contains only 1 or 2 identities, ratio of heights per column is stored

pick uniform number between 1 and n, then biasd coin flip to pick one of the two identities - cost $O(1)$

### Other Techniques

#### Photon Mapping

Trace particles from light, deposit "photons" in kd-tree

Especially useful for, e.g., caustics, participating media (fog)

#### Finite Element Radiosity

Very different approach: transport between patches in scene

Solve large linear system for equilibrium distribution

Good for diffuse lighting; hard to capture other light paths

## Lect 20 Intro to Animation

> site: [Intro to Animation](http://15462.courses.cs.cmu.edu/fall2018/lecture/introanimation)

### Keyframing

Basic idea:

* specify important events only

* computer fills in the rest via interpolation/approximation

"Events" don't have to be position. Could be color, light intensity, camera zoom...

### Spline Interpolation

#### Splines

Piecewise polynomial functions.

#### Cubic Spline Interpolation

among all interpolated curves, minimizes norm of the second derivative

runge phenomenon

#### Spline Desiderata

In general, what are some properties of a "good" spline?
- INTERPOLATION: spline passes exactlythrough data points
- CONTINUITY: at least twice differentiable everywhere
- LOCALITY: moving one control point doesn't affect whole curve

Natural cubic spline:

- INTERPOLATION: yes
- CONTINUITY: $C^2$ everywhere
- LOCALITY: no, coefficients depend on global linear system

#### Hermite / Bézier Spline:

- INTERPOLATION: yes
- CONTINUITY: no
- LOCALITY: yes

#### Catmull-Rom Spline

Basic idea: use difference of neighbors to define tagent
$$
f'(x_i) = \frac{y_{i + 1} - y_{i - 1}}{x_{i + 1} - x_{i - 1}}
$$
All the same properties as any other Hermite spline (interpolation & locality)

commonly used to interpolate motion in computer animation

#### B-Splines

Get better continuity and local control by sacrificing interpolation

B-spline basis:
$$
B_{i, 1}(t) = \begin{cases}
1, &t_i \leq t < t_{i + 1} \\
0, &\text{otherwise}
\end{cases} \\
B_{i, k}(t) = \frac{t - t_i}{t_{i + k - 1} - t_i}B_{i, k - 1}(t) + \frac{t_{i + k} - t}{t_{i + k} - t_{i + 1}}B_{i + 1, k - 1}(t)
$$
B-spline: $f(t) = \sum_{i} a_i B_{i, d}(t)$

### What are we interpolating

#### Camera Path

each path is a function $f(t) =  (x(t), y(t), z(t))$, each component is a spline

#### Charater Animation

Scene graph / Kinematic chain: scene as tree of transformations

animate by interpolate transformations

#### Inverse Kinematics

Important technique in animation & robotics

Rather than adjust individual transformations, set ugoal" and use algorithm to come up with plausible motion

#### Skeletal Animation

Often use "skeleton" to drive deformation of continuous surface

Influence of each bone determined by, e.g., weighting function

#### Blend Shapes

Instead of skeleton, interpolate directly between surfaces

Simplest scheme: take linear combination of vertex positions

SpIine used to control choice of weights over time

## Lect 21 Dynamics and Time Integration

> site: [Dynamics and Time Integration](http://15462.courses.cs.cmu.edu/fall2018/lecture/dynamicstimeintegration)

### Dynamic Description of Motion

#### Animation Equations

#### Generalized coordinates

$q = (x_0, x_1, \dots, x_n)$

Similarly, generalized velocity: $v = \dot{q} = (\dot{x_0}, \dot{x_1}, \dots, \dot{x_n})$

### ODE

Many dynamical systems can be described via an ordinary differential equation (ODE) in generalized coordinates:
$$
\frac{\mathrm{d}}{\mathrm{d} t} q = f(q, \dot{q}, t)
$$

#### Lagrangian Mechanics

kinetic energy: $K$

potential energy: $U$

Lagrangian: $\mathcal{L} = K - U$

Euler-Lagrangian equations:
$$
\frac{\mathrm{d}}{\mathrm{d}t} \frac{\partial \mathcal{L}}{\partial \dot{q}} = \frac{\partial \mathcal{L}}{\partial q}
$$
Why is this useful?
- often easier to come up with (scalar) energies than forces
- very general, works in any kind of generalized coordinates
- helps develop nice class of numerical integrators (symplectic)

#### Example: pendulum

$$
q = \theta \\
K = \frac{1}{2} I \dot{\theta}^2 = \frac{1}{2} m L^2 \dot{\theta}^2 \\
U = -mg L \cos \theta \\
\mathcal{L} = K - U = m \left( \frac{1}{2}L^2 \dot{\theta}^2 + gL \cos \theta \right) \\
\frac{\partial \mathcal{L}}{\partial \theta} = - mgL \sin \theta \\
\frac{\partial \mathcal{L}}{\partial \dot{\theta}} = mL^2 \dot{\theta} \\
\frac{\mathrm{d}}{\mathrm{d} t} \frac{\partial \mathcal{L}}{\partial \dot{\theta}} = mL^2 \ddot{\theta} \\
\ddot{\theta} = - \frac{g}{L} \sin \theta
$$

For small angle, $\sin \theta \approx \theta$, we have $\theta(t) = a \cos (t \sqrt{g / L} + b)$.

* In general, there is no closed form solution!
* Hence, we must use a numerical approximation
* And this was (almost) the simplest system we can think of!

### Simulation via ODE

#### Simulated Flocking as an ODE

Subject to very simple forces:
- attraction to center of neighbors
- repulsion from individual neighbors
- alignment toward average trajectory of neighbors

Solve large system of ODEs (numerically!)

Emergent complex behavior

#### Particle Systems

More generally, model phenomena as large collection of particles

Each article has a behavior described by (pkysical or non-physical) forces

Extremely common in graphics/games

- easy to understand
- simple equation for each particle
- easy to scale up/down

May need many particles to capture certain phenomena (e.g., fluids)

- may require fast hierarchical data structure (kd-tree, BVH, ...)
- often better to use continuum model

#### Mass-Spring Systems

Connect particles $x_0$, $x_1$ by a spring of length $L_0$

Potential energy: $U = \frac{1}{2}k (|x_0 - x_1| - L_0)^2$

Connect up many springs to describe interesting phenomena

Extremely common in graphics/games

- easy to understand
- simple equation for each particle

Often good reasons for using continuum model (PDE)

### Numerical Integration

Key idea: replace derivatives with differences

In ODE, only need to worry about derivative in time

Replace time-continuous function $q(t)$ with samples $q_k$ in time

#### Forward Euler

$$
q_{k + 1} = q_k + \tau f(q_k)
$$

explicit

not very stable

##### Stability Analysis

ODE: $\dot{u} = -au, ~ a > 0$

exact solution: $u = \exp(-at)$, $u$ should decay
$$
u_{k + 1} = u_k - \tau a u_k = (1 - \tau a) u_k \\
u_{k} = (1 - \tau a)^n u_0
$$
$u$ decays only if $|1 - \tau a| < 1$, or say $\tau < 2 / a$

In practice: need very small time steps if $a$ is large ("stiff system")

#### Backward Euler

$$
q_{k + 1} = q_k + \tau f(q_{k + 1})
$$

harder to solve (implicit)

stable (even to stable...)

##### Stability Analysis

ODE: $\dot{u} = -au, ~ a > 0$

exact solution: $u = \exp(-at)$, $u$ should decay
$$
u_{k + 1} = u_k - \tau a u_{k + 1} \\
u_{k + 1} = \frac{1}{1 + \tau a} u_k \\
u_{k} = \left( \frac{1}{1 + \tau a} \right)^n u_0
$$
$u$ always decays

Backward Euler is unconditionally stable for linear ODEs

#### Symplectic Euler

Backward Euler was stable, but we also saw (empirically) that it exibits numerical damping (damping not found in original eqn.)

Nice alternative is symplectic Euler

- update velocity using current configuration
- update configuration using new velocity

Easy to implement; used often in practice (or leapfrog, Verlet...)

### Computational Differentiation

Very often in simulation, need to differentiate extremely complicated functions (e.g., potential energy, to get forces)

Several different techniques:

- keep doing it by hand! (laborious & error prone, but potentially fast)
- numerical differentiation (simpleto code, but usually poor accuracy)
- automatic differentiation (bigger code investment, better accuracy)
- symbolic differentiation (can help w/ "by-hand", often messy results)
- geometric differentiation (sometimes simplifies "by hand" expressions)

#### Numerical Differantiation

Idea: replace derivatives with differences

#### Automatic Differentiation

Completely different idea: do arithmetic simultaneously on a function and its derivative.

I.e., rather than work with values $f$, work with tuples $(f, f')$.

Use chain rule to determine rules for manipulating tuples

* Pros: good accuracy, reasonably fast
* Cons: have to redefine all our arithmetic operators!

#### Symbolic Differentiation

Build explicit tree representing expression

Apply transformations to obtain derivative

* Pros: only needs to happen once!
* Cons: serious development investment

But, can often use existing tools (Mathematica, Maple, etc.)

Current systems not great with vectors, 3D

Often produce unnecessarily complex formulae...

#### Geometric Differentiation

Sometimes symbolic differentiation misses the "big picture"

E.g., gradient of triangle area w.r.t. vertex position p

## Lect 22 Introduction to Optimization

> site: [Introduction to Optimization](http://15462.courses.cs.cmu.edu/fall2018/lecture/optimization)

### Optimization Problem

#### Standard form

$$
f(x^{\ast}) = \min_\limits{x \in \mathbb{R}^n} f_0(x) \\
\text{subject to } f_i(x) \leq b_i, i = 1, 2, \dots, m
$$

* Q: What if we want to maximize something instead?
* A: Just flip the sign of the objective!
* Q: What if we want equality constraints, rather than inequalities?
* A: Include two constraints: $g(x) \leq c$ and $-g(x) \leq -c$

#### Local & Global Minima

#### Existance & Uniqueness of Minimizers

#### Feasibility

#### Characterization of Local Minimizer

##### Unconstrained

$$
\nabla f_0(x^{\ast}) = 0 \\
\nabla^2 f_0(x^{\ast}) \succeq 0
$$

($\succeq$ means positive semidefinite here)

##### Constrained

The above two conditions may not be satisfied

Karush-Kuhn-Tucker(KKT) conditions:
$$
\begin{align}
\exist \lambda_i \text{ s.t. } \nabla f_0(x^{\ast}) = -\sum_{i = 1}^{m} \lambda_i \nabla f_i(x^{\ast}) &~~~~ \text{stationarity} \\
f_i(x^{\ast}) \leq 0, i = 1, \dots, m &~~~~ \text{primal feasibility} \\
\lambda_i \geq 0, i = 1, \dots, m &~~~~ \text{dual feasibility} \\
\lambda_i f_i(x^{\ast}) = 0, i = 1, \dots, m &~~~~ \text{complementary slackness}
\end{align}
$$

#### Convex Optimization

Special class of problems that are almost always "easy" to solve (polynomial-time!)

Problem convex if it has a convex domain and convex objective

Why care about convex problems in graphics?
- can make guarantees about solution (always the best)
- doesn't depend on initialization (strong convexity)
- often quite efficient, but not always

##### Convex Quadratic Objectives

$$
f_0(x) = \frac{1}{2} x^T A x - x^T b, A \succeq 0 \\
\nabla f_0(x) = Ax - b = 0, Ax = b \\
\nabla^2 f_0(x) = A \succeq 0
$$

### Descent Method

#### Gradient Descent

Basic idea: follow the gradient "downhill" until it's zero

Gradient descent equation is an ODE:
$$
\frac{\mathrm{d}}{\mathrm{d}t} x(t) = -f'_0(x(t))
$$

##### How do we pick the time step?

If we're not careful, we'll go zipping all over the place; won't make any progress.

Basic idea: use "step control" to determine step size based on value of objective & derivatives.

A careful strategy (e.g. Armijo-Wolfe) can guarantee convergence at least to a local minimum.

#### Higher Order Descent

General idea: apply a coordinate transformation so that the local energy landscape looks more like a "round bowl"

Gradient now points directly toward nearby minimizer

Most basic strategy: Newton's method:
$$
x_{k + 1} = x_k - \tau ( \nabla^2 f_0(x_k) )^{-1} \nabla f_0(x_k)
$$
Great for convex problems.

For nonconvex problems, need to be more careful

In general, nonconvex optimization is a BLACK ART

Meta-strategy: try lots of solvers, see what works! (e.g. quasi-Newton, trust region, L-BFGS...)

### Forward Kinematics

Many systems well-described by a kinematic chain
- collection of rigid bodies, connected by joints
- joints have various behaviors (ball, piston, ...)
- also have constraints (e.g., range of angles)
- hierarchical structure (body -> leg -> foot)

In animation, often called a rig

How do we specify the configuration of a "rig"?

- One way: artist sets each joint individually
- Another way: ...optimization!

#### Simple Kinematic Chain

In 2D:
$$
p_1 = p_0 + \begin{bmatrix}
\cos \theta_0 & \sin \theta_0 \\
- \sin \theta_0 & \cos \theta_0 
\end{bmatrix} u_0
$$
Or use complex numbers:
$$
p_1 = p_0 + e^{\imath \theta_0}u_0
$$
and
$$
p_2 = p_0 + e^{\imath \theta_0} u_0 + e^{\imath \theta_1} u_1
$$

#### Simple IK Algorithm

Basic idea behind our IK algorithm:
- write down distance between final point and "target"
- compute gradient with respect to angles
- apply gradient descent

Objective:
$$
f_0(\theta) = \frac{1}{2} \vert \tilde{p_n} - p_n \vert^2
$$
Constraints:
- None! Thejoint angle can take any value.
- Though we could limit joint angles (for instance)

