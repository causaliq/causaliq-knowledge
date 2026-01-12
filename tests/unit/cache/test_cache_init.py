"""Unit tests for cache module initialisation."""


# Verify cache module can be imported
def test_cache_module_imports() -> None:
    """Cache module should import without errors."""
    import causaliq_knowledge.cache

    assert hasattr(causaliq_knowledge.cache, "__all__")


# Verify encoders submodule can be imported
def test_encoders_module_imports() -> None:
    """Encoders submodule should import without errors."""
    import causaliq_knowledge.cache.encoders

    assert hasattr(causaliq_knowledge.cache.encoders, "__all__")
