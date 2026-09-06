"""
generate_stl.py

V2 Biomimetic Multiscale Superhydrophobic Ship-Hull Surface
------------------------------------------------------------

Purpose
-------
Generate a visualization-oriented STL representation of the
multiscale biomimetic surface using the geometry-related design
variables from the V2 predictive framework:

    - riblet_spacing
    - riblet_height
    - lotus_intensity

Important scientific note
-------------------------
This STL is a visualization/model-representation tool.

It is NOT:
    - an experimentally validated hull surface,
    - a dimensionally faithful full-scale ship-hull CAD model,
    - a CFD mesh,
    - or a direct representation of manufacturing dimensions.

The physical inputs are supplied in millimetres, but the generated
STL uses a controlled visualization scale so that the micro/riblet
features remain numerically resolvable at the selected mesh
resolution.

Material and coating are deliberately NOT used to modify the STL
geometry. They remain ML/design variables in the predictive model
rather than arbitrary geometric scaling factors.

Dependencies
------------
    numpy
    numpy-stl
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from stl import mesh


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_STATE = 42


# ---------------------------------------------------------------------
# Visualization settings
# ---------------------------------------------------------------------

DEFAULT_RESOLUTION = 150

# The STL is a visualization patch, not a full ship hull.
VISUAL_DOMAIN_LENGTH = 5.0

# Physical riblet spacing range in the V2 synthetic dataset.
MIN_RIBLET_SPACING_MM = 0.05
MAX_RIBLET_SPACING_MM = 1.00

# Map the physical riblet-spacing range into a visually resolvable
# wavelength range on the 5 x 5 visualization domain.
#
# This avoids generating wavelengths smaller than the mesh can resolve.
VISUAL_SPACING_MIN = 0.25
VISUAL_SPACING_MAX = 1.25

# Visual amplitude factors.
#
# These are visualization scales, NOT physical material constants.
RIBLET_HEIGHT_VISUAL_SCALE = 0.75
LOTUS_AMPLITUDE_SCALE = 0.05
LOTUS_NOISE_SCALE = 0.01

# Hierarchical nano/microtexture frequency in visualization space.
LOTUS_FREQUENCY = 50.0


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _validate_inputs(
    riblet_spacing: float,
    riblet_height: float,
    lotus_intensity: float,
    resolution: int,
) -> None:
    """Validate geometry inputs before mesh generation."""

    if not np.isfinite(riblet_spacing):
        raise ValueError("riblet_spacing must be a finite number.")

    if not np.isfinite(riblet_height):
        raise ValueError("riblet_height must be a finite number.")

    if not np.isfinite(lotus_intensity):
        raise ValueError("lotus_intensity must be a finite number.")

    if riblet_spacing <= 0:
        raise ValueError("riblet_spacing must be greater than zero.")

    if riblet_height <= 0:
        raise ValueError("riblet_height must be greater than zero.")

    if not 0.0 <= lotus_intensity <= 1.0:
        raise ValueError("lotus_intensity must be between 0 and 1.")

    if not isinstance(resolution, (int, np.integer)):
        raise TypeError("resolution must be an integer.")

    if resolution < 20:
        raise ValueError("resolution must be at least 20.")


def _map_spacing_to_visual_scale(riblet_spacing: float) -> float:
    """
    Map physical riblet spacing in mm to a numerically resolvable
    visualization wavelength.

    The mapping is monotonic and bounded.

    Values outside the nominal V2 dataset range are clipped to the
    visualization range rather than producing unstable geometry.
    """

    spacing_clipped = np.clip(
        riblet_spacing,
        MIN_RIBLET_SPACING_MM,
        MAX_RIBLET_SPACING_MM,
    )

    normalized = (
        spacing_clipped - MIN_RIBLET_SPACING_MM
    ) / (
        MAX_RIBLET_SPACING_MM - MIN_RIBLET_SPACING_MM
    )

    visual_spacing = (
        VISUAL_SPACING_MIN
        + normalized
        * (VISUAL_SPACING_MAX - VISUAL_SPACING_MIN)
    )

    return float(visual_spacing)


def _build_surface(
    riblet_spacing: float,
    riblet_height: float,
    lotus_intensity: float,
    resolution: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the visualization surface.

    Returns
    -------
    Xg, Yg, Z : numpy.ndarray
        Surface coordinate grids.
    """

    # -------------------------------------------------------------
    # Visualization domain
    # -------------------------------------------------------------

    x = np.linspace(
        0.0,
        VISUAL_DOMAIN_LENGTH,
        resolution,
    )

    y = np.linspace(
        0.0,
        VISUAL_DOMAIN_LENGTH,
        resolution,
    )

    Xg, Yg = np.meshgrid(x, y)

    # -------------------------------------------------------------
    # Base curved surface
    # -------------------------------------------------------------

    # Smooth curved patch representing the underlying hull surface.
    #
    # This is a visualization shape and is not intended to represent
    # the actual sectional geometry of a specific ship hull.
    hull_base = np.clip(
        1.0 - (Yg ** 2) / (1.5 ** 2),
        0.0,
        1.0,
    )

    # -------------------------------------------------------------
    # Riblet geometry
    # -------------------------------------------------------------

    visual_spacing = _map_spacing_to_visual_scale(
        riblet_spacing
    )

    # Preserve the physical height-to-spacing relationship as much
    # as practical while applying a visualization scale.
    aspect_ratio = riblet_height / riblet_spacing

    height_visual = (
        visual_spacing
        * aspect_ratio
        * RIBLET_HEIGHT_VISUAL_SCALE
    )

    # Prevent extremely large visualization amplitudes when a user
    # supplies a value outside the normal synthetic-data range.
    height_visual = float(
        np.clip(
            height_visual,
            0.005,
            0.75,
        )
    )

    phase = (
        2.0
        * np.pi
        / visual_spacing
        * Xg
    )

    # Triangular riblet profile.
    riblet = (
        2.0
        * height_visual
        / np.pi
        * np.arcsin(np.sin(phase))
    )

    # -------------------------------------------------------------
    # Lotus-inspired hierarchical texture
    # -------------------------------------------------------------

    nano_amplitude = (
        LOTUS_AMPLITUDE_SCALE
        * lotus_intensity
    )

    rng = np.random.default_rng(RANDOM_STATE)

    noise = (
        LOTUS_NOISE_SCALE
        * rng.standard_normal(Xg.shape)
        * lotus_intensity
    )

    lotus = (
        nano_amplitude
        * np.cos(LOTUS_FREQUENCY * Xg)
        * np.cos(LOTUS_FREQUENCY * Yg)
        + noise
    )

    # -------------------------------------------------------------
    # Combined multiscale surface
    # -------------------------------------------------------------

    Z = hull_base + riblet + lotus

    return Xg, Yg, Z


def _create_stl_mesh(
    Xg: np.ndarray,
    Yg: np.ndarray,
    Z: np.ndarray,
) -> mesh.Mesh:
    """
    Convert the surface grids into a triangular STL mesh.

    The generated object is an open surface patch. It is intended for
    visualization and surface representation, not for solid-volume
    manufacturing or closed-volume analysis.
    """

    resolution = Xg.shape[0]

    # Number of grid cells:
    # (resolution - 1) x (resolution - 1)
    #
    # Two triangles per cell.
    number_of_triangles = (
        2 * (resolution - 1) ** 2
    )

    hull_mesh = mesh.Mesh(
        np.zeros(
            number_of_triangles,
            dtype=mesh.Mesh.dtype,
        )
    )

    triangle_index = 0

    for i in range(resolution - 1):

        for j in range(resolution - 1):

            # Four corners of the current grid cell.
            v0 = np.array(
                [
                    Xg[i, j],
                    Yg[i, j],
                    Z[i, j],
                ],
                dtype=float,
            )

            v1 = np.array(
                [
                    Xg[i + 1, j],
                    Yg[i + 1, j],
                    Z[i + 1, j],
                ],
                dtype=float,
            )

            v2 = np.array(
                [
                    Xg[i, j + 1],
                    Yg[i, j + 1],
                    Z[i, j + 1],
                ],
                dtype=float,
            )

            v3 = np.array(
                [
                    Xg[i + 1, j + 1],
                    Yg[i + 1, j + 1],
                    Z[i + 1, j + 1],
                ],
                dtype=float,
            )

            # First triangle.
            hull_mesh.vectors[
                triangle_index
            ] = np.array(
                [v0, v1, v2],
                dtype=float,
            )

            triangle_index += 1

            # Second triangle.
            hull_mesh.vectors[
                triangle_index
            ] = np.array(
                [v1, v3, v2],
                dtype=float,
            )

            triangle_index += 1

    return hull_mesh


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def generate_stl(
    riblet_spacing: float,
    riblet_height: float,
    lotus_intensity: float,
    resolution: int = DEFAULT_RESOLUTION,
    output_path: str = "biomimetic_hull_v2.stl",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Generate a visualization STL for the V2 biomimetic multiscale
    superhydrophobic ship-hull surface.

    Parameters
    ----------
    riblet_spacing : float
        Riblet spacing in mm.

    riblet_height : float
        Riblet height in mm.

    lotus_intensity : float
        Lotus-inspired hierarchical texture intensity.
        Expected V2 dataset range: 0.10-0.90.

    resolution : int, default=150
        Number of grid points along each surface direction.

    output_path : str, default="biomimetic_hull_v2.stl"
        Path where the STL file will be saved.

    Returns
    -------
    Xg : numpy.ndarray
        X-coordinate surface grid.

    Yg : numpy.ndarray
        Y-coordinate surface grid.

    Z : numpy.ndarray
        Generated surface elevation.

    output_path : str
        Path to the saved STL file.

    Notes
    -----
    Material and coating are intentionally excluded from this
    function.

    In the V2 architecture:

        Material/coating
            -> ML prediction inputs

        Riblet spacing/height/lotus intensity
            -> STL surface geometry

    This separation prevents arbitrary material/coating scaling
    factors from being interpreted as experimentally established
    geometric relationships.

    The STL coordinates use a visualization scale rather than direct
    physical ship-scale dimensions. Therefore, the generated STL
    should not be described as a dimensionally faithful CAD model
    or CFD-ready ship-hull geometry.
    """

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    _validate_inputs(
        riblet_spacing=riblet_spacing,
        riblet_height=riblet_height,
        lotus_intensity=lotus_intensity,
        resolution=resolution,
    )

    # -------------------------------------------------------------
    # Generate surface
    # -------------------------------------------------------------

    Xg, Yg, Z = _build_surface(
        riblet_spacing=riblet_spacing,
        riblet_height=riblet_height,
        lotus_intensity=lotus_intensity,
        resolution=resolution,
    )

    # -------------------------------------------------------------
    # Build triangular mesh
    # -------------------------------------------------------------

    hull_mesh = _create_stl_mesh(
        Xg=Xg,
        Yg=Yg,
        Z=Z,
    )

    # -------------------------------------------------------------
    # Prepare output directory
    # -------------------------------------------------------------

    output_path = os.fspath(output_path)

    output_directory = os.path.dirname(
        os.path.abspath(output_path)
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Save STL
    # -------------------------------------------------------------

    # numpy-stl recalculates normals during save by default.
    hull_mesh.save(
        output_path,
        update_normals=True,
    )

    return Xg, Yg, Z, output_path


# ---------------------------------------------------------------------
# Optional standalone test
# ---------------------------------------------------------------------

if __name__ == "__main__":

    Xg, Yg, Z, file_path = generate_stl(
        riblet_spacing=0.50,
        riblet_height=0.10,
        lotus_intensity=0.60,
        resolution=DEFAULT_RESOLUTION,
        output_path="biomimetic_hull_v2.stl",
    )

    print("STL generation completed successfully.")
    print(f"Output file: {file_path}")
    print(f"Surface resolution: {Xg.shape[0]} x {Xg.shape[1]}")
    print(f"Surface Z-range: {Z.min():.6f} to {Z.max():.6f}")
