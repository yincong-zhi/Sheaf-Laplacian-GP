# Sheaf Laplacian Gaussian Processes

This repository implements learning a Sheaf Laplacian within a kernel function of a Gaussian process.

The main implementation using the same splits as [Transductive Kernels for Gaussian Processes on Graphs](https://arxiv.org/abs/2211.15322) can be found in <code>sheaf_gp.py</code>.

Split from [Neural Sheaf Diffusion](https://openreview.net/forum?id=vbPsD-BhOZ) are implemented in <code>sheaf_gp_nsd.py</code>, the raw files can be found in the <code>splits</code> folder.

Lastly, various kernels can be found in <code>kernels.py</code>.