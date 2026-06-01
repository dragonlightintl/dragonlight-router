#!/usr/bin/env python3
"""Test script to verify our changes work correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_result_type():
    """Test that the Result type works correctly."""
    from dragonlight_router.result import Ok, Err, Result
    
    # Test Ok
    ok_result = Ok(42)
    assert ok_result.is_ok()
    assert not ok_result.is_err()
    assert ok_result.unwrap() == 42
    
    # Test Err
    err_result = Err("error message")
    assert not err_result.is_ok()
    assert err_result.is_err()
    assert err_result.unwrap_err() == "error message"
    
    print("✓ Result type tests passed")

def test_backend_tier():
    """Test that BackendTier enum has been updated."""
    from dragonlight_router.core.types import BackendTier
    
    # Check the values
    assert BackendTier.LOCAL.value == "local"
    assert BackendTier.SIMPLE.value == "simple"
    assert BackendTier.MODERATE.value == "moderate"
    assert BackendTier.COMPLEX.value == "complex"
    
    # Check that old values are gone
    try:
        _ = BackendTier.HAIKU
        assert False, "HAIKU should not exist"
    except AttributeError:
        pass
        
    try:
        _ = BackendTier.SONNET
        assert False, "SONNET should not exist"
    except AttributeError:
        pass
        
    try:
        _ = BackendTier.OPUS
        assert False, "OPUS should not exist"
    except AttributeError:
        pass
    
    print("✓ BackendTier tests passed")

def test_trust_tier():
    """Test that TrustTier enum has been updated."""
    from dragonlight_router.selection.context_filter import TrustTier, filter_by_trust_tier
    
    # Check the values
    assert TrustTier.LOCAL.value == 1
    assert TrustTier.SIMPLE.value == 2
    assert TrustTier.MODERATE.value == 3
    assert TrustTier.COMPLEX.value == 4
    
    # Check that old values are gone
    try:
        _ = TrustTier.HAIKU
        assert False, "HAIKU should not exist"
    except AttributeError:
        pass
        
    try:
        _ = TrustTier.SONNET
        assert False, "SONNET should not exist"
    except AttributeError:
        pass
        
    try:
        _ = TrustTier.OPUS
        assert False, "OPUS should not exist"
    except AttributeError:
        pass
    
    # Test filter function
    candidates = [TrustTier.LOCAL, TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX]
    
    # LOCAL should allow all
    assert filter_by_trust_tier(candidates, TrustTier.LOCAL) == candidates
    
    # SIMPLE should allow SIMPLE and above
    simple_result = filter_by_trust_tier(candidates, TrustTier.SIMPLE)
    assert set(simple_result) == {TrustTier.SIMPLE, TrustTier.MODERATE, TrustTier.COMPLEX}
    
    # MODERATE should allow MODERATE and above
    moderate_result = filter_by_trust_tier(candidates, TrustTier.MODERATE)
    assert set(moderate_result) == {TrustTier.MODERATE, TrustTier.COMPLEX}
    
    # COMPLEX should allow only COMPLEX
    complex_result = filter_by_trust_tier(candidates, TrustTier.COMPLEX)
    assert complex_result == [TrustTier.COMPLEX]
    
    print("✓ TrustTier tests passed")

def test_refresher_import():
    """Test that we can import the refresher with Result type."""
    from dragonlight_router.catalog.refresher import CatalogRefresher
    import inspect
    
    # Check the signature
    sig = inspect.signature(CatalogRefresher.refresh)
    return_annotation = sig.return_annotation
    # Should be Result[dict, CatalogRefreshError]
    assert "Result" in str(return_annotation)
    
    print("✓ Refresher import tests passed")

if __name__ == "__main__":
    test_result_type()
    test_backend_tier()
    test_trust_tier()
    test_refresher_import()
    print("\n🎉 All tests passed!")