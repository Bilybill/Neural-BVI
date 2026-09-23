# Method and implementation map

## Computational sequence

1. Train inverse networks on paired permittivity models and noisy B-scans.
2. Construct a low-dimensional basis from training errors; encode residuals around a neural estimate.
3. Fit sequential Gaussian components using a forward-data likelihood, a latent prior, and a model-space trust-region penalty.
4. Generate samples and compute reconstruction, uncertainty and event summaries.

`e2e_bvi_gpr.run_bvi` implements the residual BVI kernel. `deepwave_physics_surrogate.DeepwavePhysicsSurrogate` implements the differentiable scalar-wave forward solver required by the paper wrapper.

## Final paper configuration

The final Neural-BVI method uses a five-network mean with development-fitted affine correction as its center. Centered ensemble and MC-dropout deviations are added to BVI samples with scales 1.8 and 7.5. The candidate mean set uses residual multipliers `[0, 0.025, 0.05, 0.1, 0.2]`.

The implementation selects the **smallest-magnitude eligible correction** whose reforwarded-data RMSE improves on the reference ensemble mean by at least 0.0001, breaking ties by data RMSE. If none qualifies, it uses the reference ensemble mean. This is not an unrestricted global RMSE minimization. Samples are projected/recentered to the chosen bounded mean. A development-error standard-deviation prior with scale 0.75 contributes to reported uncertainty, and each method's interval scale is frozen before the final holdout evaluation.

The extra dispersion and standard-deviation prior are part of the released final method; its uncertainty should not be described as solely the raw variational mixture. `sample_std` and the reported `std` can differ. The zero bias map and structural standard-deviation prior in `configs/paper/` are small synthetic-derived assets, not field measurements.

## Six reported methods

| Method | Implementation |
| --- | --- |
| Neural inverse | One fixed U-Net prediction |
| MC dropout | 30 stochastic test-time network passes |
| Deep ensemble | Five independently trained inverse networks |
| Residual MAP | One optimized residual-space estimate |
| Residual Laplace | Full latent covariance around the residual MAP solution |
| Neural-BVI | Two residual Gaussian components plus the final augmentation/mean-selection procedure above |

`scripts/run_paper.py` evaluates these six methods using the released configuration.

## Metrics and scope

Models use `m = (epsilon_r - 2) / 8`; image NRMSE is RMSE in that normalized model space. Data RMSE is the waveform residual RMS in the preprocessed observation space. Empirical CRPS uses posterior samples. Coverage and interval width use frozen method-specific scales. Spearman correlation compares pixelwise uncertainty with absolute reconstruction error.

Three noise views of one model are repeated observations. Statistical comparisons cluster by model. Development data include previously evaluated model blocks; the final evaluation uses a separate 20-model subset, defined in `configs/paper/test_set.json` (prepared-test local indices 64–83). Parameters and calibration are fixed before this final evaluation.

The Deepwave operator solves the scalar wave equation, not Maxwell's equations with antenna response and conductivity. The procedural demo deliberately uses a smaller grid/network and Gaussian noise; do not compare its numbers directly to the paper table.
