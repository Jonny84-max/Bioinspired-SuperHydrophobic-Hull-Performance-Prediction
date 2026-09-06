import numpy as np
import pandas as pd

# ============================================================
# PHYSICS-INFORMED SYNTHETIC DATASET GENERATOR - VERSION 2.1
#
# Project:
# Simulated Multi-Scale Engineered Superhydrophobic Surface
# (SHS) for Advanced Ship Hull Performance
#
# Purpose:
# Synthetic physics-informed dataset for predictive modelling of:
#   1. Drag reduction
#   2. Biofouling accumulation
#   3. Surface durability
#
# IMPORTANT:
# This dataset is SYNTHETIC.
# It is not experimental data and does not constitute experimental
# or CFD validation.
#
# Physical relationships constrain the synthetic data.
# Phenomenological coefficients are modelling assumptions.
# ============================================================


# ============================================================
# 0. CONFIGURATION
# ============================================================

N_SAMPLES = 5000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# 1. SURFACE ENGINEERING VARIABLES
# ============================================================

riblet_h = rng.uniform(
    0.01, 0.30, N_SAMPLES
)                                      # mm

riblet_s = rng.uniform(
    0.05, 1.00, N_SAMPLES
)                                      # mm

lotus_intensity = rng.uniform(
    0.10, 0.90, N_SAMPLES
)

material = rng.integers(
    0, 3, N_SAMPLES
)

coating = rng.integers(
    0, 5, N_SAMPLES
)


# ============================================================
# 2. GEOMETRIC / MULTI-SCALE FEATURES
# ============================================================

aspect_ratio = (
    riblet_h /
    (riblet_s + 1e-9)
)

ridge_width = 0.05  # mm

f_riblet = (
    ridge_width /
    (ridge_width + riblet_s + 1e-9)
)

f_lotus_mod = (
    1.0 -
    0.40 * lotus_intensity
)

solid_fraction_f = np.clip(
    f_riblet * f_lotus_mod,
    0.01,
    0.99
)


# ============================================================
# 3. MARINE OPERATING ENVIRONMENT
# ============================================================

velocity = rng.uniform(
    0.5, 25.0, N_SAMPLES
)                                      # m/s

temperature = rng.uniform(
    2.0, 35.0, N_SAMPLES
)                                      # °C

salinity = rng.uniform(
    15.0, 40.0, N_SAMPLES
)                                      # PSU

time_days = rng.uniform(
    1.0, 1095.0, N_SAMPLES
)                                      # days


# ============================================================
# 4. APPROXIMATE SEAWATER PROPERTIES
# ============================================================

mu = (
    1.55e-3
    - 2.4e-5 * temperature
    + 1.0e-7 * temperature**2
)

mu = np.clip(
    mu,
    0.75e-3,
    1.55e-3
)

rho = (
    1027.0
    + 0.75 * (salinity - 35.0)
    - 0.20 * (temperature - 15.0)
)

rho = np.clip(
    rho,
    1015.0,
    1045.0
)

nu = mu / rho

surface_tension = (
    0.075
    - 1.5e-4 * (temperature - 15.0)
)

surface_tension = np.clip(
    surface_tension,
    0.070,
    0.076
)


# ============================================================
# 5. CHARACTERISTIC LENGTH SCALES
# ============================================================

# Explicit modelling scale for synthetic macro Reynolds number.
L_macro = 1.0  # m

riblet_h_m = riblet_h * 1e-3
riblet_s_m = riblet_s * 1e-3


# ============================================================
# 6. DIMENSIONLESS FLOW INDICES
# ============================================================

Re_macro = (
    velocity * L_macro / nu
)

Re_micro = (
    velocity * riblet_s_m / nu
)

Ca = (
    mu * velocity /
    surface_tension
)

We = (
    rho *
    velocity**2 *
    riblet_s_m /
    surface_tension
)


# ============================================================
# 7. SURFACE WETTING PHYSICS
# ============================================================

intrinsic_angle = 110.0

theta_y = np.radians(
    intrinsic_angle
)


# ------------------------------------------------------------
# Cassie-Baxter contact angle
# ------------------------------------------------------------

cos_theta_cassie = (
    solid_fraction_f *
    (np.cos(theta_y) + 1.0)
    - 1.0
)

cos_theta_cassie = np.clip(
    cos_theta_cassie,
    -1.0,
    1.0
)

cassie_contact_angle = np.degrees(
    np.arccos(cos_theta_cassie)
)


# ------------------------------------------------------------
# Wenzel contact angle
# ------------------------------------------------------------

roughness_ratio = (
    1.0 +
    2.0 *
    np.clip(
        aspect_ratio,
        0.0,
        1.0
    )
)

cos_theta_wenzel = (
    roughness_ratio *
    np.cos(theta_y)
)

cos_theta_wenzel = np.clip(
    cos_theta_wenzel,
    -1.0,
    1.0
)

wenzel_contact_angle = np.degrees(
    np.arccos(cos_theta_wenzel)
)


# ============================================================
# 8. PLASTRON STABILITY
# ============================================================
#
# Revised so that a meaningful fraction of engineered surfaces
# can remain in Cassie/plastron or transition states.
#
# This remains a phenomenological synthetic index.
# ============================================================

contact_angle_score = np.clip(
    (
        cassie_contact_angle - 130.0
    ) / 40.0,
    0.0,
    1.0
)

geometry_score = np.exp(
    -(
        np.log(
            (riblet_s_m + 1e-9) /
            0.00025
        ) ** 2
    )
)

velocity_penalty = np.clip(
    (velocity - 5.0) / 20.0,
    0.0,
    1.0
)

time_penalty = np.clip(
    time_days / 1095.0,
    0.0,
    1.0
)

plastron_stability = (
    0.40 * lotus_intensity
    +
    0.25 * contact_angle_score
    +
    0.20 * geometry_score
    +
    0.15 * (1.0 - velocity_penalty)
    -
    0.15 * time_penalty
)

plastron_stability = np.clip(
    plastron_stability,
    0.0,
    1.0
)


# ============================================================
# 9. WETTING STATE
# ============================================================

# 2 = Cassie/plastron-dominated
# 1 = transition
# 0 = Wenzel/wetted

wetting_state = np.select(
    [
        plastron_stability >= 0.55,
        plastron_stability >= 0.30
    ],
    [
        2,
        1
    ],
    default=0
)


# ============================================================
# 10. EFFECTIVE CONTACT ANGLE
# ============================================================

effective_contact_angle = np.where(
    wetting_state == 2,
    cassie_contact_angle,
    np.where(
        wetting_state == 1,
        (
            0.60 * cassie_contact_angle
            +
            0.40 * wenzel_contact_angle
        ),
        wenzel_contact_angle
    )
)

effective_contact_angle = np.clip(
    effective_contact_angle,
    0.0,
    180.0
)


# ============================================================
# 11. HYDRODYNAMIC / VISCOUS SCALE
# ============================================================

Cf = (
    0.0592 /
    (Re_macro ** 0.2)
)

Cf = np.clip(
    Cf,
    0.001,
    0.01
)

friction_velocity = (
    velocity *
    np.sqrt(Cf / 2.0)
)

viscous_length_scale = (
    nu /
    (friction_velocity + 1e-9)
)

viscous_sublayer_thickness = (
    5.0 *
    viscous_length_scale
)


# ============================================================
# 12. RIBLET / VISCOUS-SCALE PARAMETERS
# ============================================================

riblet_spacing_plus = (
    riblet_s_m /
    (viscous_length_scale + 1e-9)
)

riblet_height_plus = (
    riblet_h_m /
    (viscous_length_scale + 1e-9)
)


# ============================================================
# 13. MULTISCALE SURFACE FEATURES
# ============================================================

multiscale_index = (
    lotus_intensity *
    (
        1.0 +
        0.50 *
        np.clip(
            aspect_ratio,
            0.0,
            1.0
        )
    )
)

estimated_slip_length = (
    riblet_s_m /
    (
        np.sqrt(
            solid_fraction_f
        ) + 1e-9
    )
)

velocity_riblet_interact = (
    velocity *
    aspect_ratio
)


# ============================================================
# 14. MATERIAL / COATING EFFECTS
# ============================================================

material_factor = np.array(
    [
        0.95,
        1.00,
        1.05
    ]
)[material]

coating_factor = np.array(
    [
        0.90,
        0.95,
        1.00,
        1.05,
        1.10
    ]
)[coating]


# ============================================================
# 15. DRAG REDUCTION MODEL
# ============================================================
#
# Revised formulation.
#
# Instead of multiplying many small factors together, the model
# uses a bounded additive effectiveness structure. This prevents
# excessive artificial zero values while retaining physically
# meaningful dependencies.
# ============================================================


# ------------------------------------------------------------
# 15.1 Riblet-scale effectiveness
# ------------------------------------------------------------

optimal_riblet_plus = 15.0

riblet_scale_effect = np.exp(
    -0.55 *
    (
        np.log(
            (riblet_spacing_plus + 1e-6) /
            optimal_riblet_plus
        ) ** 2
    )
)

riblet_scale_effect = np.clip(
    riblet_scale_effect,
    0.0,
    1.0
)


# ------------------------------------------------------------
# 15.2 Geometry contribution
# ------------------------------------------------------------

geometry_score_drag = (
    0.55 * riblet_scale_effect
    +
    0.45 *
    np.clip(
        aspect_ratio / 0.50,
        0.0,
        1.0
    )
)

geometry_score_drag = np.clip(
    geometry_score_drag,
    0.0,
    1.0
)


# ------------------------------------------------------------
# 15.3 Hydrophobicity contribution
# ------------------------------------------------------------

contact_angle_effect = np.clip(
    (
        effective_contact_angle - 110.0
    ) / 70.0,
    0.0,
    1.0
)

hydrophobic_effect = (
    0.60 * lotus_intensity
    +
    0.40 * contact_angle_effect
)

hydrophobic_effect = np.clip(
    hydrophobic_effect,
    0.0,
    1.0
)


# ------------------------------------------------------------
# 15.4 Wetting contribution
# ------------------------------------------------------------

wetting_state_factor = np.select(
    [
        wetting_state == 2,
        wetting_state == 1
    ],
    [
        1.00,
        0.72
    ],
    default=0.45
)


# ------------------------------------------------------------
# 15.5 Reynolds-number contribution
# ------------------------------------------------------------

flow_effect_drag = (
    0.85
    +
    0.15 *
    np.tanh(
        Re_macro /
        2.0e7
    )
)


# ------------------------------------------------------------
# 15.6 Material/coating contribution
# ------------------------------------------------------------

surface_quality_factor = (
    0.90
    +
    0.05 * material
    +
    0.04 * coating
)

surface_quality_factor = np.clip(
    surface_quality_factor,
    0.90,
    1.30
)


# ------------------------------------------------------------
# 15.7 Combined SHS effectiveness
# ------------------------------------------------------------

surface_effectiveness = (
    0.50 * geometry_score_drag
    +
    0.30 * hydrophobic_effect
    +
    0.20 * plastron_stability
)

surface_effectiveness = np.clip(
    surface_effectiveness,
    0.0,
    1.0
)


# ------------------------------------------------------------
# 15.8 Drag reduction target
# ------------------------------------------------------------

drag_reduction_base = (
    2.0
    +
    11.0 *
    surface_effectiveness
    *
    wetting_state_factor
    *
    flow_effect_drag
    *
    surface_quality_factor
)


# ------------------------------------------------------------
# 15.9 Environmental degradation
# ------------------------------------------------------------

drag_degradation = (
    1.0
    -
    0.12 *
    np.clip(
        time_days / 1095.0,
        0.0,
        1.0
    )
)

drag_degradation = np.clip(
    drag_degradation,
    0.80,
    1.00
)


drag_reduction_base *= drag_degradation


# ------------------------------------------------------------
# 15.10 Heteroscedastic uncertainty
# ------------------------------------------------------------

drag_noise_sd = (
    0.20
    +
    0.015 * velocity
    +
    0.35 *
    np.abs(
        wetting_state_factor - 0.70
    )
)

drag_noise = rng.normal(
    0.0,
    drag_noise_sd,
    N_SAMPLES
)


# ------------------------------------------------------------
# 15.11 Final drag reduction
# ------------------------------------------------------------

drag_reduction = np.clip(
    drag_reduction_base
    +
    drag_noise,
    0.0,
    25.0
)


# ============================================================
# 16. BIOFOULING ACCUMULATION MODEL
# ============================================================

temperature_effect = (
    0.80
    +
    0.35 *
    np.clip(
        (temperature - 2.0) / 33.0,
        0.0,
        1.0
    )
)

salinity_effect = (
    0.90
    +
    0.20 *
    np.clip(
        (salinity - 15.0) / 25.0,
        0.0,
        1.0
    )
)

flow_effect_fouling = (
    1.15
    -
    0.30 *
    np.tanh(
        velocity / 8.0
    )
)


# ------------------------------------------------------------
# Surface protection
# ------------------------------------------------------------

surface_protection = (
    1.0
    -
    0.55 * lotus_intensity
    -
    0.15 *
    np.clip(
        (
            effective_contact_angle - 110.0
        ) / 70.0,
        0.0,
        1.0
    )
)

surface_protection = np.clip(
    surface_protection,
    0.25,
    1.0
)


# ------------------------------------------------------------
# Coating protection
# ------------------------------------------------------------

coating_protection = (
    1.05
    -
    0.12 *
    (coating / 4.0)
)


# ------------------------------------------------------------
# Growth rate
# ------------------------------------------------------------

growth_rate = (
    0.003
    *
    temperature_effect
    *
    salinity_effect
    *
    flow_effect_fouling
    *
    surface_protection
    *
    coating_protection
)


# ------------------------------------------------------------
# Lag phase
# ------------------------------------------------------------

lag_days = (
    45.0
    +
    70.0 * lotus_intensity
)

effective_time = np.maximum(
    time_days - lag_days,
    0.0
)


# ------------------------------------------------------------
# Carrying capacity
# ------------------------------------------------------------

carrying_capacity = np.clip(
    0.95
    -
    0.35 * lotus_intensity
    -
    0.10 * (coating / 4.0),
    0.35,
    0.95
)


# ------------------------------------------------------------
# Logistic accumulation
# ------------------------------------------------------------

biofouling_base = (
    carrying_capacity
    /
    (
        1.0
        +
        np.exp(
            -growth_rate *
            (
                effective_time - 300.0
            )
        )
    )
)


# ------------------------------------------------------------
# Surface degradation
# ------------------------------------------------------------

degradation_factor = (
    1.0
    +
    0.25 *
    np.clip(
        time_days / 1095.0,
        0.0,
        1.0
    )
)

biofouling = np.clip(
    biofouling_base
    *
    degradation_factor,
    0.001,
    1.0
)


# ============================================================
# 17. DURABILITY MODEL
# ============================================================

durability_base = 74.0

coating_contribution = (
    12.0 *
    (coating / 4.0)
)

material_contribution = (
    6.0 *
    (material / 2.0)
)

time_penalty = (
    10.0 *
    (time_days / 1095.0)
)

velocity_penalty = (
    5.0 *
    np.clip(
        velocity / 25.0,
        0.0,
        1.0
    )
)

temperature_penalty = (
    3.0 *
    np.clip(
        (temperature - 2.0) / 33.0,
        0.0,
        1.0
    )
)

salinity_penalty = (
    4.0 *
    np.clip(
        (salinity - 15.0) / 25.0,
        0.0,
        1.0
    )
)

velocity_salinity_interaction = (
    3.0 *
    (velocity / 25.0)
    *
    (salinity / 40.0)
)

temperature_coating_interaction = (
    2.0 *
    (temperature / 35.0)
    *
    (coating / 4.0)
)

geometry_penalty = (
    4.0 *
    np.clip(
        aspect_ratio,
        0.0,
        1.0
    )
)

wetting_degradation_penalty = np.select(
    [
        wetting_state == 1,
        wetting_state == 0
    ],
    [
        2.0,
        5.0
    ],
    default=0.0
)

durability_noise_sd = (
    1.0
    +
    0.025 * velocity
    +
    0.50 *
    np.clip(
        salinity / 40.0,
        0.0,
        1.0
    )
)

durability_noise = rng.normal(
    0.0,
    durability_noise_sd,
    N_SAMPLES
)

durability = np.clip(
    durability_base
    +
    coating_contribution
    +
    material_contribution
    -
    time_penalty
    -
    velocity_penalty
    -
    temperature_penalty
    -
    salinity_penalty
    -
    velocity_salinity_interaction
    -
    temperature_coating_interaction
    -
    geometry_penalty
    -
    wetting_degradation_penalty
    +
    durability_noise,
    0.0,
    100.0
)


# ============================================================
# 18. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame({

    "riblet_height": riblet_h,
    "riblet_spacing": riblet_s,
    "lotus_intensity": lotus_intensity,
    "material": material,
    "coating": coating,

    "velocity": velocity,
    "temperature": temperature,
    "salinity": salinity,
    "time": time_days,

    "aspect_ratio": aspect_ratio,
    "solid_fraction_f": solid_fraction_f,

    "cassie_contact_angle": cassie_contact_angle,
    "wenzel_contact_angle": wenzel_contact_angle,
    "effective_contact_angle": effective_contact_angle,
    "plastron_stability": plastron_stability,
    "wetting_state": wetting_state,

    "dynamic_viscosity": mu,
    "water_density": rho,
    "kinematic_viscosity": nu,
    "surface_tension": surface_tension,

    "Re_macro": Re_macro,
    "Re_micro": Re_micro,
    "Capillary_number": Ca,
    "Weber_number": We,

    "skin_friction_coefficient": Cf,
    "friction_velocity": friction_velocity,
    "viscous_length_scale": viscous_length_scale,
    "viscous_sublayer_thickness": viscous_sublayer_thickness,
    "riblet_spacing_plus": riblet_spacing_plus,
    "riblet_height_plus": riblet_height_plus,

    "multiscale_index": multiscale_index,
    "estimated_slip_length": estimated_slip_length,
    "velocity_riblet_interact": velocity_riblet_interact,

    "durability": durability,
    "drag_reduction": drag_reduction,
    "biofouling": biofouling
})


# ============================================================
# 19. DATA QUALITY CHECKS
# ============================================================

assert len(df) == N_SAMPLES

assert df.shape[1] == 36

assert df.isnull().sum().sum() == 0

assert df["drag_reduction"].between(
    0.0,
    25.0
).all()

assert df["biofouling"].between(
    0.001,
    1.0
).all()

assert df["durability"].between(
    0.0,
    100.0
).all()

assert df["effective_contact_angle"].between(
    0.0,
    180.0
).all()

assert df["wetting_state"].isin(
    [0, 1, 2]
).all()

assert (df["Re_macro"] > 0).all()
assert (df["Re_micro"] > 0).all()
assert (df["Capillary_number"] > 0).all()
assert (df["Weber_number"] > 0).all()

assert (df["kinematic_viscosity"] > 0).all()
assert (df["surface_tension"] > 0).all()


# ============================================================
# 20. DUPLICATE CHECK
# ============================================================

duplicate_rows = df.duplicated().sum()

assert duplicate_rows == 0


# ============================================================
# 21. CORRELATION / SENSITIVITY SCREENING
# ============================================================

target_columns = [
    "drag_reduction",
    "biofouling",
    "durability"
]

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns

correlation_matrix = df[
    numeric_columns
].corr()

target_correlations = (
    correlation_matrix[target_columns]
    .abs()
    .sort_values(
        by="drag_reduction",
        ascending=False
    )
)


# ============================================================
# 22. SAVE DATASET
# ============================================================

# IMPORTANT:
# Keep the filename agreed for the new V2 project.

output_file = (
    "biomimetic_opsimml_dataset.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 23. SAVE CORRELATION MATRIX
# ============================================================

correlation_file = (
    "biomimetic_opsimml_dataset_correlations.csv"
)

correlation_matrix.to_csv(
    correlation_file
)


# ============================================================
# 24. SCIENTIFIC INDICES SUMMARY
# ============================================================

print()
print("=" * 75)
print(
    "PHYSICS-INFORMED SYNTHETIC DATASET "
    "GENERATED - VERSION 2.1"
)
print("=" * 75)

print(
    f"Rows:                         "
    f"{df.shape[0]}"
)

print(
    f"Columns:                      "
    f"{df.shape[1]}"
)

print(
    f"Dataset:                      "
    f"{output_file}"
)

print()


print("TARGET VARIABLES")
print("-" * 75)

print(
    f"Drag reduction (%):           "
    f"{df['drag_reduction'].min():.2f} - "
    f"{df['drag_reduction'].max():.2f}"
)

print(
    f"Drag reduction mean:          "
    f"{df['drag_reduction'].mean():.2f}%"
)

print(
    f"Drag reduction median:        "
    f"{df['drag_reduction'].median():.2f}%"
)

print(
    f"Zero drag-reduction rows:     "
    f"{(df['drag_reduction'] == 0).sum()}"
)

print(
    f"Biofouling index:             "
    f"{df['biofouling'].min():.3f} - "
    f"{df['biofouling'].max():.3f}"
)

print(
    f"Durability score:             "
    f"{df['durability'].min():.2f} - "
    f"{df['durability'].max():.2f}"
)

print()


print("DIMENSIONLESS FLOW INDICES")
print("-" * 75)

print(
    f"Re_macro:                     "
    f"{df['Re_macro'].min():.3e} - "
    f"{df['Re_macro'].max():.3e}"
)

print(
    f"Re_micro:                     "
    f"{df['Re_micro'].min():.3e} - "
    f"{df['Re_micro'].max():.3e}"
)

print(
    f"Capillary number:             "
    f"{df['Capillary_number'].min():.3e} - "
    f"{df['Capillary_number'].max():.3e}"
)

print(
    f"Weber number:                 "
    f"{df['Weber_number'].min():.3e} - "
    f"{df['Weber_number'].max():.3e}"
)

print()


print("SURFACE PHYSICS")
print("-" * 75)

print(
    f"Cassie contact angle:         "
    f"{df['cassie_contact_angle'].min():.2f} - "
    f"{df['cassie_contact_angle'].max():.2f} deg"
)

print(
    f"Wenzel contact angle:         "
    f"{df['wenzel_contact_angle'].min():.2f} - "
    f"{df['wenzel_contact_angle'].max():.2f} deg"
)

print(
    f"Effective contact angle:      "
    f"{df['effective_contact_angle'].min():.2f} - "
    f"{df['effective_contact_angle'].max():.2f} deg"
)

print(
    f"Plastron stability:           "
    f"{df['plastron_stability'].min():.3f} - "
    f"{df['plastron_stability'].max():.3f}"
)

print()


print("WETTING STATES")
print("-" * 75)

cassie_count = (
    df["wetting_state"] == 2
).sum()

transition_count = (
    df["wetting_state"] == 1
).sum()

wenzel_count = (
    df["wetting_state"] == 0
).sum()

print(
    f"Cassie / plastron state:      "
    f"{cassie_count}"
)

print(
    f"Transition state:             "
    f"{transition_count}"
)

print(
    f"Wenzel / wetted state:        "
    f"{wenzel_count}"
)

print()


print("RIBLET / VISCOUS-SCALE INDICES")
print("-" * 75)

print(
    f"Riblet spacing+ :             "
    f"{df['riblet_spacing_plus'].min():.3f} - "
    f"{df['riblet_spacing_plus'].max():.3f}"
)

print(
    f"Riblet height+ :              "
    f"{df['riblet_height_plus'].min():.3f} - "
    f"{df['riblet_height_plus'].max():.3f}"
)

print(
    f"Viscous length scale (m):     "
    f"{df['viscous_length_scale'].min():.3e} - "
    f"{df['viscous_length_scale'].max():.3e}"
)

print()


print("QUALITY CONTROL")
print("-" * 75)

print(
    f"Missing values:               "
    f"{df.isnull().sum().sum()}"
)

print(
    f"Duplicate rows:               "
    f"{duplicate_rows}"
)

print(
    "Column-count validation:      PASSED"
)

print(
    "Target bounds:                PASSED"
)

print(
    "Dimensionless indices:        PASSED"
)

print(
    "Wetting-state validation:     PASSED"
)

print(
    "Physical positivity checks:   PASSED"
)

print()


print("SCIENTIFIC STATUS")
print("-" * 75)

print(
    "Synthetic physics-informed dataset."
)

print(
    "Not experimental or CFD-validated data."
)

print(
    "Phenomenological coefficients must be "
    "reported as modelling assumptions."
)

print("=" * 75)
print("DATASET GENERATION COMPLETE")
print("=" * 75)
