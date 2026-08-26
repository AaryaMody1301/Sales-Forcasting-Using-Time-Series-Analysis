from importlib.metadata import version

import sales_forecasting


def test_runtime_version_matches_distribution_metadata():
    assert sales_forecasting.__version__ == version("sales-forecasting-time-series")
