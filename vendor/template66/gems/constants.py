"""Official constants, each with its source.

Values marked OFFICIAL are printed on the DrivenData problem page or the official
rules PDF. Values marked REFERENCE come from the official reference notebook
(github.com/drivendataorg/gems-prize-reference-solution). Nothing here is a score.
"""

# OFFICIAL - problem page, "Submission format": EPSG 32611, 100 m, float32 in [0, 1],
# same bounds as training data, data outside bounds null or NaN, single layer.
EPSG = "EPSG:32611"
PIXEL_M = 100.0

# REFERENCE - reference notebook output: Width 3292, Height 3730 (rows x cols = 3730 x 3292).
SHAPE_HW = (3730, 3292)

# OFFICIAL - problem page, "Performance metric": alpha=0.2, beta=0.8, triangular kernel, R=300 m.
ALPHA = 0.2
BETA = 0.8
KERNEL_RANGE_M = 300.0

# REFERENCE - reference notebook: feature nodata sentinel is -3.4028234663852886e+38
# (cells below -1e38 are treated as missing); label nodata is -1.
FEATURE_NODATA_SENTINEL = -1e38
LABEL_NODATA = -1
