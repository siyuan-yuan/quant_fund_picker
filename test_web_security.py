# -*- coding: utf-8 -*-
"""Web 边界回归：同源写入、请求体上限、输入规范化与安全响应头。"""

import webapp


def test_security_headers_and_no_wildcard_cors():
    client = webapp.app.test_client()
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "same-origin"
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cross_origin_writes_are_rejected_before_handler():
    client = webapp.app.test_client()
    response = client.post(
        "/api/fund/110011",
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403
    assert response.get_json()["message"] == "拒绝跨站写入请求"


def test_same_origin_and_non_browser_requests_remain_compatible():
    client = webapp.app.test_client()

    # 同源浏览器请求会进入路由，并由路由本身拒绝非法代码，而不是被同源校验误伤。
    same_origin = client.post(
        "/api/fund/not-a-code",
        headers={"Origin": "http://localhost"},
        base_url="http://localhost",
    )
    assert same_origin.status_code == 400
    assert same_origin.get_json()["message"] == "基金代码无效"

    # curl/脚本通常不带 Origin，也继续支持。
    no_origin = client.post("/api/watchlist", json={"codes": ["bad"]})
    assert no_origin.status_code == 400
    assert "有效" in no_origin.get_json()["message"]


def test_request_size_limit():
    client = webapp.app.test_client()
    response = client.post(
        "/api/ledger",
        data="x" * (2 * 1024 * 1024 + 1),
        content_type="application/json",
    )
    assert response.status_code == 413


def test_ledger_id_and_amount_are_normalized():
    txn = webapp._norm_txn({
        "id": "x');alert(1);//",
        "code": "110011",
        "date": "2026-08-20",
        "side": "buy",
        "amount": "1万",
    })
    assert txn is not None
    assert txn["id"].startswith("l")
    assert txn["amount"] == 10000.0

    assert webapp._parse_yuan("inf") is None
    assert webapp._parse_yuan("nan") is None
