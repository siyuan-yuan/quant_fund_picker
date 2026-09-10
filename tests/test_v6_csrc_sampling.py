from __future__ import annotations

import pandas as pd

from v6.csrc_sampling import (
    build_stratified_mixed_sample,
    expand_sample_family_aliases,
)


def _fixture() -> pd.DataFrame:
    rows = []
    dates = {"pre2013": "20100101", "2013_2019": "20160101", "2020_2026": "20220101"}
    status_codes = {"存续": "L", "清盘": "D"}
    code = 1
    for status_name, status_code in status_codes.items():
        for era, found_date in dates.items():
            for family_no in range(4):
                base = f"{status_name}{era}基金{family_no}"
                for suffix in ("A", "C"):
                    rows.append(
                        {
                            "ts_code": f"{code:06d}.OF",
                            "name": base + "-" + suffix,
                            "management": "测试管理人",
                            "fund_type": "混合型",
                            "found_date": found_date,
                            "status": status_code,
                            "market": "O",
                        }
                    )
                    code += 1
    rows.append(
        {
            "ts_code": "999999.OF",
            "name": "非目标股票基金A",
            "management": "测试管理人",
            "fund_type": "股票型",
            "found_date": "20220101",
            "status": "L",
            "market": "O",
        }
    )
    return pd.DataFrame(rows)


def test_builds_deterministic_balanced_family_sample() -> None:
    raw = _fixture()

    first = build_stratified_mixed_sample(raw, per_stratum=2, seed=20260904)
    second = build_stratified_mixed_sample(raw, per_stratum=2, seed=20260904)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 12
    assert first.groupby(["status", "inception_era"]).size().eq(2).all()
    assert first["family_key"].is_unique
    assert first["share_code"].str.fullmatch(r"\d{6}").all()
    assert set(first["fund_type"]) == {"混合型"}


def test_uses_all_members_when_a_stratum_is_smaller_than_quota() -> None:
    raw = _fixture()
    raw = raw.drop(raw[(raw["status"] == "D") & (raw["found_date"] == "20100101")].index[2:])

    sample = build_stratified_mixed_sample(raw, per_stratum=3, seed=7)

    small = sample[(sample["status"] == "清盘") & (sample["inception_era"] == "pre2013")]
    assert len(small) == 1


def test_expands_frozen_family_to_every_share_alias_without_changing_families() -> None:
    raw = _fixture()
    sample = build_stratified_mixed_sample(raw, per_stratum=1, seed=20260904)
    selected_code = sample.iloc[0]["share_code"]
    matching = raw[raw["ts_code"].str.startswith(selected_code)].iloc[0].copy()
    raw.loc[matching.name, "market"] = "E"
    unrelated = matching.copy()
    unrelated["name"] = "同代码但不同基金-A"
    unrelated["management"] = "另一管理人"
    unrelated["market"] = "O"
    raw = pd.concat([raw, unrelated.to_frame().T], ignore_index=True)

    aliases = expand_sample_family_aliases(raw, sample)

    assert aliases["family_key"].nunique() == 6
    assert len(aliases) == 12
    assert aliases.groupby("family_key").size().eq(2).all()
    assert aliases["query_code"].str.fullmatch(r"\d{6}").all()
    assert not aliases["query_code"].eq("999999").any()
    assert aliases.groupby("family_key")["sample_share_code"].nunique().eq(1).all()


def test_treats_priority_and_aggressive_suffixes_as_share_aliases() -> None:
    raw = pd.DataFrame(
        [
            {"ts_code": "150010.SZ", "name": "测试混合(LOF)-优先", "management": "管理人"},
            {"ts_code": "150011.SZ", "name": "测试混合(LOF)-进取", "management": "管理人"},
        ]
    )
    sample = pd.DataFrame(
        [
            {
                "share_code": "150011",
                "name": "测试混合(LOF)-进取",
                "management": "管理人",
                "status": "清盘",
                "inception_era": "pre2013",
            }
        ]
    )

    aliases = expand_sample_family_aliases(raw, sample)

    assert aliases["query_code"].tolist() == ["150010", "150011"]
