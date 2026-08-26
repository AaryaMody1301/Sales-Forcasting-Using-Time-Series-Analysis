# Security Policy

## Supported code

Security fixes are applied to the current `main` branch and the active v1 release line. Use the latest available patch release whenever possible.

## Reporting a vulnerability

Please avoid posting secrets, credentials, private data, or working exploit details in a public issue. Use GitHub's private vulnerability reporting / Security Advisory flow when it is available for this repository. If private reporting is unavailable, contact the repository owner through GitHub before disclosing sensitive details publicly.

Ordinary correctness bugs that do not contain sensitive security details can use the normal issue tracker.

## Persisted model trust boundary

Some adapters persist fitted Python objects with `pickle`:

- ARIMA and ETS adapters serialize fitted statsmodels/Python objects;
- Random Forest, Gradient Boosting, and XGBoost adapters serialize fitted Python estimator objects.

**Only load model artifacts that you created yourself or obtained from a trusted, verified source.** Python pickle deserialization can execute arbitrary code. A checksum can prove that an artifact has not changed since it was recorded, but it does not make an untrusted pickle safe.

Persisted Python estimators should also be restored with a compatible Python and dependency environment. Experiment manifests record package/dependency metadata to make that environment auditable.

The Prophet adapter is persisted using Prophet's JSON serialization rather than Python pickle.

## Data and secrets

Raw third-party datasets are not committed to the release tree. Do not commit API keys, service credentials, personal data, or proprietary datasets. Local/generated artifacts and raw-data locations should remain covered by `.gitignore`.

## CI supply-chain policy

Repository workflows should:

- use the minimum `GITHUB_TOKEN` permissions required by each job;
- pin external Actions to full commit SHAs;
- avoid persisting checkout credentials when subsequent steps do not need them;
- keep benchmark dependency versions explicit and auditable.
