# Data Release Notes

## First Release

QM9 data URL:

- https://huggingface.co/datasets/divelab/OrbEvo/tree/main/QM9_tddft

During internal validation, we found that **7.86%** of generated QM9 trajectories show abnormal behavior for dipole where the dipoles reproduced from the saved coefficients do not match excatly with the dipoles saved by ABACUS during data generation. We estimate that this will not significantly affect training or evaluation since we do not use dipole information in training, and we always re-caculate dipoles from coefficients during evaluation (although we did observe increased errors on abnormal samples using a trained model).
To avoid blocking use of the dataset, we release all QM9 data and provide a good-index file that marks the trajectories with good dipoles, Current code can use that good-index list to keep only the validated QM9 samples after the data split is formed.

Relevant good-index file:

- `orbevo/datasets/QM9_abs_max_lt_1e-03.txt`

## Recommendation

We recommend using the provided good-index filter when training or evaluating models.

## Planned Second Release

We are re-generating the abnormal samples and plan to provide a second data release with corrected trajectories soon.
