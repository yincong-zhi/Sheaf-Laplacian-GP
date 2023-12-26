# Sheaf Laplacian Gaussian Processes

This repository implements learning a Sheaf Laplacian within a kernel function of a Gaussian process, giving us a Sheaf GP.

The main implementation using the same splits as [Transductive Kernels for Gaussian Processes on Graphs](https://arxiv.org/abs/2211.15322) can be found in <code>sheaf_gp.py</code>.

Implementation using the same splits as [Neural Sheaf Diffusion](https://openreview.net/forum?id=vbPsD-BhOZ) can be found in <code>sheaf_gp_nsd.py</code>, the raw files are stored in the <code>splits</code> folder.

Lastly, various kernels can be found in <code>kernels.py</code>.