"""Storm-trajectory sensitivity toolkit for the Bellinge 1D/2D SWMM model.

Modules:

    config      shared paths and constants
    radar       X-band radar cube parsing, QC, and event selection
    advection   motion estimation, frozen-field resynthesis, realizations
    mesh        DEM -> triangular 2D mesh, boundary conditions, node coupling
    model       .inp assembly (rain gages, [2D_*] sections) per realization
    run         parallel ensemble driver and metric extraction
    figures     box plots and centroid scatter plots
"""
