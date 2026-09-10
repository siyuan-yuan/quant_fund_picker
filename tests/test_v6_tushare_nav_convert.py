import pandas as pd
import pytest

from v6.tushare_nav_convert import convert_fund_nav, convert_nav_file


def test_convert_nav_uses_adjusted_nav_return_and_announcement_known_at():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000001.OF", "000001.OF"],
            "ann_date": ["20200102", "20200103", "20200106"],
            "nav_date": ["20200101", "20200102", "20200103"],
            "unit_nav": [1.0, 0.9, 0.99],
            "adj_nav": [1.0, 1.1, 1.21],
        }
    )

    out = convert_fund_nav(raw)

    assert out.columns.tolist() == ["date", "known_at", "nav", "adj_nav", "ret"]
    assert out.loc[0, "known_at"] == pd.Timestamp("2020-01-02")
    assert out.loc[1, "ret"] == pytest.approx(0.10)
    assert out.loc[2, "ret"] == pytest.approx(0.10)


def test_duplicate_nav_date_keeps_earliest_announcement():
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000001.OF"],
            "ann_date": ["20200105", "20200103"],
            "nav_date": ["20200102", "20200102"],
            "unit_nav": [1.1, 1.1],
            "adj_nav": [1.2, 1.2],
        }
    )

    out = convert_fund_nav(raw)

    assert len(out) == 1
    assert out.loc[0, "known_at"] == pd.Timestamp("2020-01-03")


def test_convert_nav_file_writes_one_numeric_code_file(tmp_path):
    source = tmp_path / "fund_nav.csv"
    pd.DataFrame(
        {
            "ts_code": ["000001.OF", "000001.OF"],
            "ann_date": ["20200102", "20200103"],
            "nav_date": ["20200101", "20200102"],
            "unit_nav": [1.0, 1.1],
            "adj_nav": [1.0, 1.1],
        }
    ).to_csv(source, index=False)

    report = convert_nav_file(source, tmp_path / "nav")

    saved = pd.read_csv(tmp_path / "nav" / "nav_000001.csv")
    assert len(saved) == 2
    assert report.input_codes == 1
    assert report.output_codes == 1
    assert report.duplicate_rows == 0
