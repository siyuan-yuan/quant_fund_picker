"""Resumable, manifest-driven Tushare collector for the V6 fund research layer.

Raw calls are written as independent CSV shards.  A job is checkpointed only
after its shard has been atomically replaced, so rerunning is safe after an
interruption.  The token is never stored in source code.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "cache" / "tushare" / "bulk"
DEFAULT_START = "20000101"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _replace_with_retry(temporary, path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    _replace_with_retry(temporary, path)


def _replace_with_retry(source: Path, target: Path, attempts: int = 8) -> None:
    """Tolerate short-lived Windows scanner/indexer locks on the target file."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (2**attempt))


def date_windows(start: str, end: str, years: int = 1) -> list[tuple[str, str]]:
    """Split inclusive YYYYMMDD bounds at calendar-year boundaries."""
    left = datetime.strptime(start, "%Y%m%d").date()
    right = datetime.strptime(end, "%Y%m%d").date()
    if left > right:
        raise ValueError("start must not be after end")
    result: list[tuple[str, str]] = []
    cursor = left
    while cursor <= right:
        window_end = date(cursor.year + years - 1, 12, 31)
        window_end = min(window_end, right)
        result.append((cursor.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        cursor = date(window_end.year + 1, 1, 1)
    return result


def fetch_offset_pages(
    api: Callable[..., pd.DataFrame], page_size: int, **kwargs
) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    offset = 0
    while True:
        page = api(offset=offset, limit=page_size, **kwargs)
        if page is None or page.empty:
            break
        pages.append(page)
        if len(page) < page_size:
            break
        offset += page_size
    return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()


def consolidate_shards(
    shard_dir: Path, target: Path, keys: Iterable[str] = ()
) -> int:
    paths = sorted(shard_dir.glob("*.csv"))
    frames = []
    for path in paths:
        if not path.stat().st_size:
            continue
        try:
            frames.append(pd.read_csv(path, dtype=str))
        except pd.errors.EmptyDataError:
            continue
    frames = [frame for frame in frames if len(frame.columns)]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    usable_keys = [key for key in keys if key in combined.columns]
    if usable_keys:
        combined = combined.drop_duplicates(usable_keys, keep="last")
    atomic_write_csv(combined, target)
    return len(combined)


DATASETS = {
    # endpoint: strategy, primary keys, page/window limit
    "fund_basic": ("fund_basic", ["ts_code"], None),
    "fund_company": ("single", ["name"], None),
    "fund_manager": ("offset", ["ts_code", "name", "begin_date"], 5000),
    "fund_nav": ("fund_dates", ["ts_code", "nav_date", "ann_date"], None),
    "fund_div": ("fund", ["ts_code", "ann_date", "ex_date"], None),
    "fund_portfolio": (
        "fund_dates",
        ["ts_code", "ann_date", "end_date", "symbol"],
        None,
    ),
    "fund_share": ("etf_year", ["ts_code", "trade_date"], None),
    "etf_basic": ("etf_basic", ["ts_code"], None),
    "etf_index": ("single", ["ts_code"], None),
    "fund_daily": ("etf_year", ["ts_code", "trade_date"], None),
    "fund_adj": ("etf_year", ["ts_code", "trade_date"], None),
    "etf_share_size": ("etf_year", ["ts_code", "trade_date"], None),
    "trade_cal": ("single_dates", ["exchange", "cal_date"], None),
    "yc_cb": ("year", ["ts_code", "curve_type", "trade_date", "curve_term"], None),
    "shibor": ("year", ["date"], None),
    "index_basic": ("index_basic", ["ts_code"], None),
    "index_daily": ("index_year", ["ts_code", "trade_date"], None),
    "index_dailybasic": ("index_year", ["ts_code", "trade_date"], None),
    "index_global": ("global_year", ["ts_code", "trade_date"], None),
}

RBSA_INDEX_CODES = [
    "000016.SH", "000300.SH", "000905.SH", "000852.SH", "399673.SZ",
    "000015.SH", "000987.CSI", "000988.CSI", "000990.CSI", "000991.CSI",
    "000992.CSI", "000993.CSI", "H30090.CSI", "H30533.CSI",
]
GLOBAL_INDEX_CODES = ["HSI", "HKTECH", "SPX", "IXIC"]


class Collector:
    def __init__(self, pro, out: Path, start: str, end: str, sleep: float = 0.12):
        self.pro = pro
        self.out = out
        self.start = start
        self.end = end
        self.sleep = sleep
        self.state_path = out / "checkpoint.json"
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"version": 1, "completed": {}, "failures": {}}

    def _done(self, dataset: str, job: str) -> bool:
        return job in self.state["completed"].get(dataset, [])

    def _save_result(self, dataset: str, job: str, frame: pd.DataFrame) -> None:
        safe = job.replace("/", "_").replace(":", "_")
        atomic_write_csv(frame, self.out / "shards" / dataset / f"{safe}.csv")
        completed = self.state["completed"].setdefault(dataset, [])
        if job not in completed:
            completed.append(job)
        self.state["failures"].get(dataset, {}).pop(job, None)
        atomic_write_json(self.state_path, self.state)

    def _record_failure(self, dataset: str, job: str, exc: Exception) -> None:
        failures = self.state["failures"].setdefault(dataset, {})
        failures[job] = {"error": str(exc), "at": datetime.now().isoformat()}
        atomic_write_json(self.state_path, self.state)
        print(f"[fail] {dataset} {job}: {exc}", flush=True)

    def _run(self, dataset: str, job: str, call: Callable[[], pd.DataFrame]) -> None:
        if self._done(dataset, job):
            return
        try:
            frame = call()
        except Exception as exc:  # remote errors are data-quality artifacts
            self._record_failure(dataset, job, exc)
        else:
            if frame is None:
                frame = pd.DataFrame()
            self._save_result(dataset, job, frame)
        time.sleep(self.sleep)

    def collect(self, dataset: str, fund_codes: list[str], etf_codes: list[str]) -> None:
        strategy, keys, page_size = DATASETS[dataset]
        api = getattr(self.pro, dataset)
        started_completed = len(self.state["completed"].get(dataset, []))
        attempted = 0

        def run(job: str, call: Callable[[], pd.DataFrame]) -> None:
            nonlocal attempted
            was_done = self._done(dataset, job)
            self._run(dataset, job, call)
            if not was_done:
                attempted += 1
                if attempted % 200 == 0:
                    done = len(self.state["completed"].get(dataset, []))
                    failures = len(self.state["failures"].get(dataset, {}))
                    print(
                        f"[progress] {dataset}: completed={done:,} "
                        f"(+{done-started_completed:,}), failures={failures:,}",
                        flush=True,
                    )

        if strategy == "single":
            run("all", lambda: api())
        elif strategy == "fund_basic":
            for market in ("E", "O"):
                run(market, lambda m=market: api(market=m))
        elif strategy == "etf_basic":
            for status in ("L", "D", "P"):
                run(status, lambda s=status: api(list_status=s))
        elif strategy == "index_basic":
            for market in ("MSCI", "CSI", "SSE", "SZSE", "CICC", "SW", "OTH"):
                run(market, lambda m=market: api(market=m))
        elif strategy == "offset":
            run(
                "all",
                lambda: fetch_offset_pages(api, page_size=page_size),
            )
        elif strategy == "single_dates":
            run(
                f"{self.start}_{self.end}",
                lambda: api(exchange="SSE", start_date=self.start, end_date=self.end),
            )
        elif strategy == "fund":
            for code in fund_codes:
                run(code, lambda c=code: api(ts_code=c))
        elif strategy == "fund_dates":
            for code in fund_codes:
                run(
                    code,
                    lambda c=code: api(
                        ts_code=c, start_date=self.start, end_date=self.end
                    ),
                )
        elif strategy in {"fund_year", "etf_year"}:
            codes = fund_codes if strategy == "fund_year" else etf_codes
            for code in codes:
                # Daily endpoints cap at 2,000/5,000 rows; five calendar years
                # remain below 2,000 mainland trading days.
                for left, right in date_windows(self.start, self.end, years=5):
                    job = f"{code}_{left}_{right}"
                    run(
                        job,
                        lambda c=code, a=left, b=right: api(
                            ts_code=c, start_date=a, end_date=b
                        ),
                    )
        elif strategy == "year":
            for left, right in date_windows(self.start, self.end):
                job = f"{left}_{right}"
                kwargs = {"start_date": left, "end_date": right}
                if dataset == "yc_cb":
                    kwargs.update(ts_code="1001.CB", curve_type="0")
                run(job, lambda kw=kwargs: api(**kw))
        elif strategy in {"index_year", "global_year"}:
            codes = RBSA_INDEX_CODES if strategy == "index_year" else GLOBAL_INDEX_CODES
            for code in codes:
                for left, right in date_windows(self.start, self.end):
                    job = f"{code}_{left}_{right}"
                    run(
                        job,
                        lambda c=code, a=left, b=right: api(
                            ts_code=c, start_date=a, end_date=b
                        ),
                    )
        rows = consolidate_shards(
            self.out / "shards" / dataset, self.out / f"{dataset}.csv", keys
        )
        print(f"[ok] {dataset}: {rows:,} rows", flush=True)


def _load_codes(path: Path) -> list[str]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype=str)
    return list(dict.fromkeys(frame["ts_code"].dropna().tolist()))


def get_pro(token: str, http_url: str | None):
    import tushare as ts

    pro = ts.pro_api(token)
    if http_url:
        pro._DataApi__http_url = http_url
    return pro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", default=None)
    parser.add_argument("--http-url", default=os.getenv("TUSHARE_HTTP_URL"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--only", default=",".join(DATASETS))
    args = parser.parse_args()

    token = args.token or os.getenv("TUSHARE_TOKEN")
    token_path = ROOT / "cache" / "tushare_token.txt"
    if not token and token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("Missing token: set TUSHARE_TOKEN or cache/tushare_token.txt")

    fund_basic = ROOT / "cache" / "tushare" / "fund_basic.csv"
    fund_codes = _load_codes(fund_basic)
    basic = pd.read_csv(fund_basic, dtype=str)
    etf_basic = args.out / "etf_basic.csv"
    if etf_basic.exists():
        etfs = pd.read_csv(etf_basic, dtype=str)
        etf_codes = etfs["ts_code"].dropna().drop_duplicates().tolist()
    else:
        etf_codes = basic.loc[basic["market"].eq("E"), "ts_code"].dropna().tolist()
    collector = Collector(get_pro(token, args.http_url), args.out, args.start, args.end, args.sleep)
    selected = [name.strip() for name in args.only.split(",") if name.strip()]
    unknown = sorted(set(selected) - set(DATASETS))
    if unknown:
        raise SystemExit(f"Unknown datasets: {unknown}")
    for dataset in selected:
        collector.collect(dataset, fund_codes, etf_codes)
    atomic_write_json(
        args.out / "manifest.json",
        {
            "generated_at": datetime.now().isoformat(),
            "start": args.start,
            "end": args.end,
            "datasets": selected,
            "fund_codes": len(fund_codes),
            "etf_codes": len(etf_codes),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
