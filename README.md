# OpenGL GPU Fluid Solver
---
This project implements a 2D fluid solver on the CUDA and the OpenGL 3.3.
The solver features a marker-and-cell grid,  3rd order Runge-Kutta advection, a conjugate gradient solver.
# Videos
---
rendered with this code: https://www.youtube.com/watch?v=UFZW0CrogCk
# Compilation
---
Compilation requirements
- glfw3
- glad
- glm
- nvcc
 
The makefile in the repository should work on Linux.

# Refrence
---
I refered to following repositories
- https://github.com/tunabrain/incremental-fluids
- https://github.com/opengl-tutorials/ogl
