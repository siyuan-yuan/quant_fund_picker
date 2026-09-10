from pathlib import Path

import pandas as pd
import pytest

from pit_universe import PITUniverseError, PITUniverseStore, TARGET_TYPES
from v6.full_pit_validate import validate_full_pit
from v6.pit_fund_master import (
    build_monthly_snapshots,
    from_tushare_fund_basic,
    normalize_fund_master,
)


def _raw_master(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "share_code",
            "name",
            "fund_type",
            "status",
            "inception_date",
            "end_date",
            "known_at",
            "source",
            "family_id",
        ],
    )


def test_share_class_merge_uses_earliest_eligible_share():
    raw = _raw_master(
        [
            ("000001", "某基金A", "股票型", "存续", "2010-01-01", None, "2010-01-01", "fixture", "F1"),
            ("000002", "某基金C", "股票型", "存续", "2015-01-01", None, "2015-01-01", "fixture", "F1"),
        ]
    )

    out = normalize_fund_master(raw)

    assert out["canonical_fund_id"].nunique() == 1
    assert out.loc[out.share_code.eq("000001"), "is_representative"].item()
    assert not out.loc[out.share_code.eq("000002"), "is_representative"].item()


def test_delisted_fund_is_present_before_end_and_absent_after():
    master = normalize_fund_master(
        _raw_master(
            [
                ("D00001", "退市基金A", "股票型", "清盘", "2010-01-01", "2018-06-15", "2010-01-01", "fixture", "F1"),
            ]
        )
    )

    snapshots = build_monthly_snapshots(
        master, pd.to_datetime(["2018-05-31", "2018-06-30"])
    )

    assert "D00001" in set(snapshots[pd.Timestamp("2018-05-31")].share_code)
    assert "D00001" not in set(snapshots[pd.Timestamp("2018-06-30")].share_code)


def test_store_rejects_row_known_after_snapshot(tmp_path):
    pd.DataFrame(
        {
            "code": ["000001", "000002"],
            "canonical_fund_id": ["000001", "000002"],
            "name": ["测试基金A", "测试基金B"],
            "fund_type": ["股票型", "股票型"],
            "inception_date": ["2010-01-01", "2010-01-01"],
            "known_at": ["2020-01-31", "2020-02-01"],
            "as_of": ["2020-01-31", "2020-01-31"],
        }
    ).to_csv(tmp_path / "2020-01-31.csv", index=False)

    store = PITUniverseStore(tmp_path)
    with pytest.raises(PITUniverseError, match="known_at"):
        store.universe("2020-01-31")


def test_validate_full_pit_passes_complete_fixture(tmp_path):
    master = normalize_fund_master(
        _raw_master(
            [
                ("D00001", "退市基金A", "股票型", "清盘", "2010-01-01", "2018-06-15", "2010-01-01", "fixture", "F1"),
            ]
        )
    )
    snapshot_root = tmp_path / "snapshots"
    nav_root = tmp_path / "nav"
    snapshot_root.mkdir()
    nav_root.mkdir()
    build_monthly_snapshots(
        master, pd.to_datetime(["2018-05-31"]), output_dir=snapshot_root
    )
    pd.DataFrame(
        {"date": ["2018-05-30", "2018-06-15"], "nav": [1.0, 1.01], "ret": [0.0, 0.01]}
    ).to_csv(nav_root / "nav_D00001.csv", index=False)

    report = validate_full_pit(
        master,
        nav_root=nav_root,
        snapshot_root=snapshot_root,
        expected_dates=pd.to_datetime(["2018-05-31"]),
    )

    assert report.passed
    assert report.delisted_nav_coverage == 1.0
    assert report.causality_violations == 0
    assert report.missing_months == ()


def test_validate_full_pit_fails_when_inputs_are_missing(tmp_path):
    report = validate_full_pit(
        pd.DataFrame(),
        nav_root=tmp_path / "missing-nav",
        snapshot_root=tmp_path / "missing-snapshots",
        expected_dates=pd.to_datetime(["2018-05-31"]),
    )

    assert not report.passed
    assert report.delisted_nav_coverage == 0.0
    assert report.missing_months == ("2018-05-31",)


def test_tushare_basic_maps_status_dates_and_investment_types():
    raw = pd.DataFrame(
        {
            "ts_code": ["510001.SH", "160001.OF", "160002.OF"],
            "name": ["境内ETF", "灵活基金", "保本基金"],
            "management": ["公司甲", "公司乙", "公司丙"],
            "fund_type": ["股票型", "混合型", "混合型"],
            "invest_type": ["被动指数型", "灵活配置型", "保本混合型"],
            "found_date": ["20100101", "20120201", "20130301"],
            "issue_date": [None, None, None],
            "list_date": ["20100105", None, None],
            "due_date": [None, "20180615", None],
            "delist_date": [None, None, None],
            "status": ["L", "D", "L"],
            "benchmark": ["沪深300指数收益率", "沪深300×60%+债券×40%", "三年定存"],
            "market": ["E", "O", "O"],
        }
    )

    out = from_tushare_fund_basic(raw)

    assert out.share_code.tolist() == ["160001", "510001"]
    assert out.fund_type.tolist() == ["混合型-灵活", "指数型-股票"]
    dead = out.loc[out.share_code.eq("160001")].iloc[0]
    assert dead.status == "清盘"
    assert dead.end_date == pd.Timestamp("2018-06-15")
    assert dead.end_known_at == pd.Timestamp("2018-06-15")


def test_tushare_basic_excludes_records_without_a_historical_inception_date():
    raw = pd.DataFrame(
        {
            "ts_code": ["160003.OF"],
            "name": ["日期缺失基金"],
            "management": ["公司甲"],
            "fund_type": ["股票型"],
            "invest_type": ["股票型"],
            "found_date": [None],
            "issue_date": [None],
            "list_date": [None],
            "due_date": [None],
            "delist_date": [None],
            "status": ["L"],
            "benchmark": ["沪深300指数收益率"],
            "market": ["O"],
        }
    )

    assert from_tushare_fund_basic(raw).empty


def test_full_pit_target_types_include_overseas_equity_index_funds():
    assert "指数型-海外股票" in TARGET_TYPES


def test_tushare_duplicate_market_rows_prefer_off_exchange_metadata():
    raw = pd.DataFrame(
        {
            "ts_code": ["150239.SZ", "150239.OF"],
            "name": ["医药指数分级-A", "医药指数A"],
            "management": ["公司甲", "公司甲"],
            "fund_type": ["股票型", "股票型"],
            "invest_type": [None, "被动指数型"],
            "found_date": ["20150817", "20150817"],
            "issue_date": [None, None],
            "list_date": ["20150820", None],
            "due_date": ["20201231", "20201231"],
            "delist_date": ["20201231", None],
            "status": ["D", "D"],
            "benchmark": ["医药指数收益率", "医药指数收益率"],
            "market": ["E", "O"],
        }
    )

    out = from_tushare_fund_basic(raw)

    assert len(out) == 1
    assert out.iloc[0].share_code == "150239"
    assert out.iloc[0].source_ts_code == "150239.OF"
    assert out.iloc[0].fund_type == "指数型-股票"


def test_validation_does_not_count_live_fund_with_future_due_date_as_delisted(tmp_path):
    master = normalize_fund_master(
        _raw_master(
            [
                ("000001", "存续基金A", "股票型", "存续", "2010-01-01", "2030-01-01", "2010-01-01", "fixture", "F1"),
            ]
        )
    )
    snapshot_root = tmp_path / "snapshots"
    build_monthly_snapshots(master, pd.to_datetime(["2020-01-31"]), snapshot_root)

    report = validate_full_pit(
        master,
        nav_root=tmp_path / "nav",
        snapshot_root=snapshot_root,
        expected_dates=pd.to_datetime(["2020-01-31"]),
    )

    assert report.delisted_count == 0


def test_validation_restores_leading_zero_fund_codes(tmp_path):
    master = pd.DataFrame(
        {
            "share_code": [49],
            "status": ["清盘"],
            "end_date": [pd.Timestamp("2020-01-31")],
        }
    )
    nav_root = tmp_path / "nav"
    snapshot_root = tmp_path / "snapshots"
    nav_root.mkdir()
    snapshot_root.mkdir()
    (nav_root / "nav_000049.csv").write_text("date,nav,ret\n2020-01-31,1.0,\n")
    pd.DataFrame({"code": ["000049"], "as_of": ["2020-01-31"]}).to_csv(
        snapshot_root / "2020-01-31.csv", index=False
    )

    report = validate_full_pit(
        master,
        nav_root=nav_root,
        snapshot_root=snapshot_root,
        expected_dates=pd.to_datetime(["2020-01-31"]),
    )

    assert report.delisted_nav_count == 1
