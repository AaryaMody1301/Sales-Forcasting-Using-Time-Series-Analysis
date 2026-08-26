# Model Inclusion Policy

A model is not promoted because it is newer or more complex.

## Required for benchmark promotion

A candidate must:

1. implement the canonical `ForecastModel` contract;
2. use no future target information in fit, feature construction, tuning, or weighting;
3. run on the same chronological folds as the baseline;
4. persist enough metadata to reproduce its configuration;
5. surface failures instead of silently substituting another algorithm;
6. be compared against the last-value baseline and simpler accepted models;
7. have enough data/folds for the performance statement being made.

The public API can contain a model even when it does not beat the baseline. In that case the project describes it as supported, not superior.

## v1 LSTM decision: no-go

The reviewed release benchmark contains only 32 contiguous weekly observations, with 24 observations in the earliest outer training fold and two outer holdout folds. That is enough for an acceptance smoke benchmark but not enough evidence to justify a new deep-learning dependency and model family.

The legacy repository had LSTM code, but it used incompatible execution paths and evaluation logic. It is not carried into v1.

LSTM can be reconsidered after a larger genuine target series is available. A future implementation should include chronological validation, `shuffle=False` where sequence ordering matters, explicit seeds/repeated runs, save/load verification, and a direct benchmark against naive/ETS/ARIMA and the strongest non-neural challenger.
