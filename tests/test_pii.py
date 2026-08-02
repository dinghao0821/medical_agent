from services.pii import mask_pii


def test_mask_email():
    assert "[EMAIL]" in mask_pii("contact me at john.doe@example.com please")
    assert "john.doe@example.com" not in mask_pii("john.doe@example.com")


def test_mask_cn_phone():
    assert "[PHONE]" in mask_pii("my number is 13800138000")


def test_mask_id_like():
    assert "[ID]" in mask_pii("id 110101199003074477")


def test_empty_and_non_string():
    assert mask_pii("") == ""
    assert mask_pii(None) is None
    assert isinstance(mask_pii(12345), str)
