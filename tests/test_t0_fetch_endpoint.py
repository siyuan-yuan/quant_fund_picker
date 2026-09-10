from types import SimpleNamespace

from t0_fetch import configure_http_url, load_requested_codes


def test_configure_http_url_sets_tushare_private_endpoint():
    pro = SimpleNamespace(_DataApi__http_url="https://api.tushare.pro")

    configure_http_url(pro, "https://tuaremax.top")

    assert pro._DataApi__http_url == "https://tuaremax.top"


def test_configure_http_url_leaves_default_when_not_provided():
    pro = SimpleNamespace(_DataApi__http_url="https://api.tushare.pro")

    configure_http_url(pro, None)

    assert pro._DataApi__http_url == "https://api.tushare.pro"


def test_load_requested_codes_deduplicates_and_preserves_ts_suffix(tmp_path):
    path = tmp_path / "codes.txt"
    path.write_text("150239.OF\n510001.SH\n150239.OF\n\n", encoding="utf-8")

    assert load_requested_codes(path) == ["150239.OF", "510001.SH"]
