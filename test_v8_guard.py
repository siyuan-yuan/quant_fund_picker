# -*- coding: utf-8 -*-
"""V8 护栏 + 东财 JSON 解析（不依赖 akshare / 网络）。"""
import json
import unittest

import em_fetch
import v8_guard


class TestV8Guard(unittest.TestCase):
    def test_install_idempotent(self):
        a = v8_guard.install()
        b = v8_guard.install()
        self.assertIn("installed", a)
        self.assertTrue(a["installed"])
        self.assertEqual(a["installed"], b["installed"])
        self.assertEqual(a["available"], b["available"])

    def test_miniracer_singleton(self):
        import sys
        import types

        class Fake:
            n = 0

            def __init__(self):
                type(self).n += 1
                self.n = type(self).n

            def eval(self, src):
                return src

        fake = types.ModuleType("py_mini_racer")
        fake.MiniRacer = Fake
        prev = sys.modules.get("py_mini_racer")
        sys.modules["py_mini_racer"] = fake
        try:
            v8_guard._reset_for_tests()
            st = v8_guard.install()
            self.assertTrue(st["singleton"])
            a = fake.MiniRacer()
            b = fake.MiniRacer()
            self.assertIs(a, b)
            self.assertEqual(Fake.n, 1)
            self.assertEqual(a.eval("1+1"), "1+1")
        finally:
            if prev is None:
                sys.modules.pop("py_mini_racer", None)
            else:
                sys.modules["py_mini_racer"] = prev
            v8_guard._reset_for_tests()
            v8_guard.install()

    def test_call_ak_serializes(self):
        box = []

        def fn(x):
            box.append(x)
            return x * 2

        self.assertEqual(v8_guard.call_ak(fn, 3), 6)
        self.assertEqual(box, [3])

    def test_status_copy(self):
        st = v8_guard.status()
        st["installed"] = "tampered"
        self.assertIsInstance(v8_guard.status()["installed"], bool)


class TestEmFetchParse(unittest.TestCase):
    def test_parse_jsonp(self):
        obj = em_fetch.parse_jsonp('jQuery351({"ErrCode":0,"Data":{"TotalCount":1}})')
        self.assertEqual(obj["ErrCode"], 0)
        self.assertEqual(obj["Data"]["TotalCount"], 1)

    def test_parse_jsonp_rejects_garbage(self):
        with self.assertRaises(RuntimeError):
            em_fetch.parse_jsonp("not-json")

    def test_lsjz_records(self):
        payload = {
            "Data": {
                "TotalCount": 2,
                "LSJZList": [
                    {"FSRQ": "2024-01-02", "DWJZ": "1.2300", "JZZZL": "1.50"},
                    {"FSRQ": "2024-01-03", "DWJZ": "1.2400", "JZZZL": "-0.80"},
                    "bad",
                ],
            }
        }
        rows, total = em_fetch.lsjz_records(payload)
        self.assertEqual(total, 2)
        self.assertEqual(rows[0], ("2024-01-02", "1.2300", "1.50"))
        self.assertEqual(len(rows), 2)

    def test_lsjz_empty_payload(self):
        rows, total = em_fetch.lsjz_records({})
        self.assertEqual(rows, [])
        self.assertEqual(total, 0)

    def test_parse_fundcode_search(self):
        text = 'var r = [["000001","HXCZHH","华夏成长混合","混合型-灵活","HUAXIA"],' \
               '["161725","ZSZZYL","招商中证白酒","股票型","ZHAOSHANG"]];'
        rows = em_fetch.parse_fundcode_search(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["基金代码"], "000001")
        self.assertEqual(rows[0]["基金简称"], "华夏成长混合")
        self.assertEqual(rows[1]["基金类型"], "股票型")

    def test_parse_fundcode_search_rejects_empty(self):
        with self.assertRaises(RuntimeError):
            em_fetch.parse_fundcode_search("var r = [];")

    def test_nav_roundtrip_schema(self):
        """解析结果能被 provider 同构的列转换吃下去（不 import provider）。"""
        payload = json.loads(
            '{"Data":{"TotalCount":1,"LSJZList":'
            '[{"FSRQ":"2024-06-01","DWJZ":"2.5","JZZZL":"10"}]}}'
        )
        rows, total = em_fetch.lsjz_records(payload)
        self.assertEqual(total, 1)
        date, nav, ret_pct = rows[0]
        self.assertEqual(date, "2024-06-01")
        self.assertAlmostEqual(float(nav), 2.5)
        self.assertAlmostEqual(float(ret_pct) / 100.0, 0.10)


if __name__ == "__main__":
    unittest.main()
