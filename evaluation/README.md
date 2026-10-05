# Evaluation

Scripts for Tests 2 and 3 of the thesis evaluation, and the APQC pipeline demonstration.
`common.py` holds the file loaders and the Rand Index shared by both scripts.

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

## APQC pipeline demonstration

Runs the full EA-GPT pipeline in the web app on processes from the APQC Process
Classification Framework (PCF) for Education, with no context document (process list only).

The APQC PCF is copyrighted by APQC, so the framework, the process lists taken from it and
the run logs are not included in this repository. The PCF for Education is available from
[apqc.org](https://www.apqc.org/). To reproduce a run, upload the process names below as a
CSV with one process name per line.

| Run | Input (from the APQC PCF for Education) | Processes | Context | Final systems |
|---|---|---|---|---|
| 1 | Level 2, all 12 categories | 79 | none | 38 |
| 2 | Level 3 of category 3.0, *Design and Deliver Student Support Services* | 35 | none | 15 |
| 3 | Level 1, the 12 categories | 12 | none | 3 |
| 4 | Same as run 2, plus 3 architect constraints | 35 | none | 4 |

Without context, the APQC level largely determines the result: level 2 and level 3 give
almost one entity per process and a fragmented architecture, while level 1 gives three
coherent but coarse systems (the only run where the EA principles changed the clustering,
from 5 to 3). In run 4 the architect's constraints merged the fragmented clusters from 8 to
4 systems, while the ISA metrics stayed the same or dropped slightly, showing that these
metrics favour fragmented architectures.
