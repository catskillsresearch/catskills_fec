## Appendix: 3rd party software

Packages below are those installed in the project `.venv` that are fusion-specific, or that are used to compute, differentiate, discretize, or plot physics equations. Versions are from `pip list` on this machine. HTTP, test, and editor packages are omitted.

## Fusion related

| Package | Version | Role |
|---|---|---|
| FreeGS | 0.8.3.dev54+g9fbec2056 | Free-boundary Grad–Shafranov solver for tokamak equilibria. |
| freeqdsk | 0.5.2 | Reader/writer for GEQDSK and AEQDSK equilibrium files. |
| omas | 0.95.2 | Ordered Multidimensional Array Structures — IMAS-style fusion data model (coil, probe, and equilibrium mappings). |
| omfit_classes | 3.2026.37.2 | OMFIT class library (One Modeling Framework for Integrated Tasks) for tokamak analysis. |
| fortranformat | 2.0.3 | Fortran textual I/O used by GEQDSK-style equilibrium files. |

## Scientific Computation

| Package | Version | Role |
|---|---|---|
| numpy | 2.5.1 | N-dimensional arrays; the substrate for $\psi(R,Z)$, coil Green functions, and finite differences. |
| scipy | 1.18.1 | Special functions (elliptic integrals for vacuum $\psi$), interpolation, linear algebra, and PDE-adjacent solvers. |
| sympy | 1.14.0 | Computer algebra — symbolic differentiation and rearrangement of equilibrium identities. |
| mpmath | 1.3.0 | Arbitrary-precision floating point (SymPy numeric backend). |
| torch | 2.13.0+cu132 | GPU tensors and autodiff for residual / operator models of $\Delta^*\psi$. |
| triton | 3.7.1 | Compiler for custom GPU kernels (finite-difference / Green stencils). |
| cuda-toolkit | 13.2.1 | CUDA 13.2 meta-package backing the Torch wheel. |
| cuda-bindings | 13.3.1 | Python bindings into the CUDA runtime. |
| cuda-pathfinder | 1.6.0 | Locates the installed CUDA libraries. |
| nvidia-cublas | 13.4.0.1 | Dense linear algebra. |
| nvidia-cusolver | 12.2.0.1 | Dense and sparse linear solvers. |
| nvidia-cusparse | 12.7.10.1 | Sparse matrix kernels. |
| nvidia-cusparselt-cu13 | 0.8.1 | Structured-sparse matrix multiply. |
| nvidia-cudnn-cu13 | 9.25.1.1 | Convolution and related tensor primitives. |
| nvidia-cufft | 12.2.0.46 | Fast Fourier transforms. |
| nvidia-curand | 10.4.2.55 | Random number generation on device. |
| nvidia-cuda-runtime | 13.2.75 | CUDA runtime. |
| nvidia-cuda-nvrtc | 13.2.78 | Runtime compilation of device code. |
| nvidia-cuda-cupti | 13.2.75 | Profiling / tracing of GPU kernels. |
| nvidia-cufile | 1.7.1.22 | GPU-direct storage I/O. |
| nvidia-nccl-cu13 | 2.29.7 | Multi-GPU collectives. |
| nvidia-nvjitlink | 13.3.33 | JIT linking of CUDA objects. |
| nvidia-nvshmem-cu13 | 3.4.5 | GPU shared-memory communications. |
| nvidia-nvtx | 13.2.75 | Kernel annotation for profilers. |
| Pint | 0.26.1 | Physical units and dimensional analysis ($I$, $B$, $\psi$, $\mathrm{eV}$, $\mathrm{m}^{-3}$). |
| uncertainties | 3.2.3 | Values with error bars and first-order error propagation. |
| pandas | 3.0.5 | Tabular / time-series handling of coil and Thomson clocks. |
| pyarrow | 25.0.0 | Columnar storage for the shot parquet files. |
| xarray | 2026.7.0 | Labeled N-D arrays (shot, time, $R$, $Z$) for equilibrium fields. |
| h5py | 3.16.0 | HDF5 scientific file I/O. |
| netCDF4 | 1.7.4 | netCDF scientific file I/O. |
| cftime | 1.6.5 | CF-convention time coordinates for netCDF. |
| zarr | 2.18.7 | Chunked compressed N-D arrays (flux pools, caches). |
| numcodecs | 0.15.1 | Compression codecs used by Zarr. |
| datasets | 5.0.1 | Hugging Face loader for the challenge parquet configs. |
| scikit-learn | 1.9.0 | Linear / kernel / MLP baselines in flux PCA space. |
| scikit-image | 0.26.0 | Image operators on the $65\times 65$ $\psi$ map. |
| shapely | 2.1.2 | Computational geometry on LCFS polygons. |
| networkx | 3.6.1 | Graphs (field-line / mesh connectivity; Torch dependency). |
| Cython | 3.3.0 | Compile numerical extensions. |
| joblib | 1.5.3 | Parallel map over shots / frames. |
| threadpoolctl | 3.6.0 | BLAS / LAPACK thread-pool control. |

## Visualization

| Package | Version | Role |
|---|---|---|
| matplotlib | 3.11.1 | Static $(R,Z)$ flux maps, coil geometry, and residual plots. |
| contourpy | 1.3.3 | Contours of 2-D quadrilateral grids — $\psi$ surfaces and LCFS. |
| plotly | 6.9.0 | Interactive shot / diagnostic figures (dFL-style). |
| tensorboard | 2.21.0 | Training curves for residual and scalar heads. |
| tensorboard-data-server | 0.7.2 | Fast event loading for TensorBoard. |
| pillow | 12.3.0 | Raster image encode/decode for figure export. |
| ImageIO | 2.37.4 | Scientific / volumetric image I/O. |
| tifffile | 2026.7.31 | TIFF I/O for 2-D field dumps. |
| cycler | 0.12.1 | Matplotlib style cycles. |
| kiwisolver | 1.5.0 | Matplotlib constraint solver for layouts. |
| fonttools | 4.63.0 | Font handling for publication figures. |
