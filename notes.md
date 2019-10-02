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

