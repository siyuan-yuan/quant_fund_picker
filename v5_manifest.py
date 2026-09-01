#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V5 预登记 #19（M1.4）：产物内容哈希 manifest

每个研究产物目录追加 manifest.jsonl 一行：
  {artifact 内每个文件的 sha256 前16, 代码 commit, 参数签名, 运行时刻, 输入文件哈希}
用法:
    python v5_manifest.py output/v5 --inputs cache/pe_沪深300.csv --params '{"stage":"D0.2"}'
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys


def _sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def current_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def write_manifest(artifact_dir: str, inputs: list[str], params: dict, note: str = "") -> dict:
    files = {}
    for f in sorted(glob.glob(os.path.join(artifact_dir, "*"))):
        if os.path.isfile(f) and os.path.basename(f) != "manifest.jsonl":
            files[os.path.basename(f)] = _sha16(f)
    rec = dict(ts=dt.datetime.now().isoformat(timespec="seconds"),
               commit=current_commit(), params=params, note=note,
               inputs={os.path.basename(i): _sha16(i) for i in inputs if os.path.exists(i)},
               artifacts=files)
    with open(os.path.join(artifact_dir, "manifest.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir")
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--params", default="{}")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rec = write_manifest(a.artifact_dir, a.inputs, json.loads(a.params), a.note)
    print(json.dumps(rec, ensure_ascii=False, indent=1, default=str)[:800])


if __name__ == "__main__":
    main()
