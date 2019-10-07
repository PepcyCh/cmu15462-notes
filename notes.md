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

