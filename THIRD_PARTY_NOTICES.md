# Third-Party Notices

This repository contains code and a small derived benchmark that depend on third-party projects/data. Their own licenses and terms remain authoritative.

## Vehicle Sales Data

The reviewed benchmark in `data/benchmarks/car_prices_weekly_median.csv` is derived from the public **Vehicle Sales Data** dataset maintained on Kaggle:

- Source: https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data
- Upstream raw file: `car_prices.csv`
- Repository use: weekly median `sellingprice` benchmark derived through `scripts/prepare_vehicle_sales.py`

The raw upstream dataset is **not redistributed** in this repository. Only the small reviewed aggregate benchmark and provenance metadata are versioned. Consult the upstream dataset page for the current license, terms, and attribution requirements before redistributing or using the raw data independently.

## Python dependencies

Runtime, optional, and development dependencies are listed in `pyproject.toml`. The exact release-benchmark environment is recorded separately in `requirements-benchmark.txt`. Each dependency is distributed under its own upstream license; this repository's MIT license does not replace those dependency licenses.

## Prior repository attribution

The root `LICENSE` retains the original 2023 copyright notice and adds the current project's copyright notice. Retaining the earlier notice is intentional and avoids removing historical attribution while recognizing later work on this repository.
