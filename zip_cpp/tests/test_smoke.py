def test_imports():
    import dimred_ccp
    from dimred_ccp import core
    assert hasattr(core, "to_downloadable_csv")
