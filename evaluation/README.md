# Evaluation

Scripts for Tests 2 and 3 of the thesis evaluation. `common.py` holds the file loaders and
the Rand Index shared by both scripts.

## Test 2: BSP clustering vs. reference (`test2_bsp_vs_reference.py`)

Feeds a real, human-made CRUD matrix directly into the BSP algorithm, skipping the LLM
extraction step, so that only the clustering is evaluated. The matrix is clustered twice:
(a) structurally, with no LLM involvement, and (b) after one round of EA principles, where
the LLM turns the principles into bias weights. Both results are compared with a reference
clustering defined by the case's authors, using the Rand Index at process and entity level
and the ISA quality metrics.

## Test 3: density-threshold sensitivity (`test3_threshold_sensitivity.py`)

Runs the structural BSP clustering on the same matrix across a fine sweep of density
thresholds, from 0 to 1 in steps of 0.01, including the default 0.5 from Lee (1999) and the
adaptive threshold. For each threshold it records the number of clusters, the ISA quality
metrics and the Rand Index against the reference. This checks whether 0.5 is also the
threshold closest to the human reference and the one that gives the best ISA metrics. The
test is fully deterministic and makes no LLM calls.
