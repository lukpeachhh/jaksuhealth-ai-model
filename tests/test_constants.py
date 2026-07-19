from jaksuhealth_ai.constants import CLASS_NAMES_BY_ID


def test_canonical_amd_sd_label_mapping() -> None:
    assert CLASS_NAMES_BY_ID == {
        0: "Background",
        1: "SRF",
        2: "IRF",
        3: "PED",
        4: "SHRM",
        5: "IS/OS",
    }
