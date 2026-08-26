import pandas as pd
import pytest

from sales_forecasting.data import DatasetContractError, clean_vehicle_sales_source


def test_vehicle_sales_cleaner_reports_invalid_and_out_of_range_rows():
    frame = pd.DataFrame(
        {
            "saledate": [
                "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)",
                "Wed Jul 08 2015 09:30:00 GMT-0700 (PDT)",
                "not-a-date",
                "Fri Jan 01 9899 09:00:00 GMT-0800 (PST)",
            ],
            "sellingprice": [10000, 12000, 13000, 14000],
        }
    )

    cleaned, report = clean_vehicle_sales_source(frame, max_excluded_fraction=0.75)

    assert len(cleaned) == 2
    assert report.raw_rows == 4
    assert report.invalid_timestamp_rows == 1
    assert report.out_of_range_timestamp_rows == 1
    assert report.excluded_rows == 2
    assert str(cleaned["saledate"].dt.tz) == "America/Los_Angeles"


def test_vehicle_sales_cleaner_rejects_large_invalid_fraction():
    frame = pd.DataFrame(
        {
            "saledate": ["not-a-date", "Tue Dec 16 2014 12:30:00 GMT-0800 (PST)"],
            "sellingprice": [10000, 12000],
        }
    )

    with pytest.raises(DatasetContractError, match="invalid-row fraction"):
        clean_vehicle_sales_source(frame)
