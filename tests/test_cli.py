import json

import pandas as pd

from sales_forecasting.cli import build_parser, main


def test_cli_has_canonical_subcommands():
    parser = build_parser()
    args = parser.parse_args(
        [
            "inspect",
            "--csv",
            "example.csv",
            "--timestamp-col",
            "date",
            "--target-col",
            "sales",
            "--frequency",
            "D",
        ]
    )
    assert args.command == "inspect"


def test_cli_inspect_validates_and_summarizes_custom_csv(tmp_path, capsys):
    path = tmp_path / "sales.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "sales": [1, 2, 3, 4, 5],
        }
    ).to_csv(path, index=False)

    code = main(
        [
            "inspect",
            "--csv",
            str(path),
            "--timestamp-col",
            "date",
            "--target-col",
            "sales",
            "--frequency",
            "D",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["observations"] == 5
    assert output["missing_periods"] == 0
    assert output["target"] == "sales"
