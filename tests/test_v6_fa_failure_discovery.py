from tools.build_v6_fa_failure_discovery import candidate_contexts


def test_candidate_contexts_keeps_allocation_language():
    text = "本基金股票等权益类资产占基金资产的比例不超过30%，债券不低于70%。"
    contexts = candidate_contexts(text)
    assert len(contexts) == 1
    assert "权益类资产" in contexts[0]


def test_candidate_contexts_ignores_unrelated_percentage():
    assert candidate_contexts("持有人出席比例不低于50%。") == []
