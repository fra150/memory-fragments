"""Tests for the memory fragments model."""

import pytest
from memory_fragments.model import MemoryFragment


class TestMemoryFragment:
    """Test cases for MemoryFragment class."""
    
    def test_init_empty(self):
        """Test initialization with no data."""
        fragment = MemoryFragment()
        assert fragment.data == {}
        assert fragment.metadata == {}
        assert fragment.timestamp is None
    
    def test_init_with_data(self):
        """Test initialization with data."""
        test_data = {"key": "value", "number": 42}
        fragment = MemoryFragment(data=test_data)
        assert fragment.data == test_data
        assert fragment.metadata == {}
    
    def test_process(self):
        """Test processing a fragment."""
        test_data = {"original": "data"}
        fragment = MemoryFragment(data=test_data)
        result = fragment.process()
        
        assert result["original"] == "data"
        assert result["processed"] is True
        # Original data should remain unchanged
        assert fragment.data == test_data
    
    def test_add_metadata(self):
        """Test adding metadata."""
        fragment = MemoryFragment()
        fragment.add_metadata("author", "test_user")
        fragment.add_metadata("priority", 1)
        
        assert fragment.metadata["author"] == "test_user"
        assert fragment.metadata["priority"] == 1
    
    def test_get_metadata(self):
        """Test getting metadata."""
        fragment = MemoryFragment()
        fragment.add_metadata("test_key", "test_value")
        
        assert fragment.get_metadata("test_key") == "test_value"
        assert fragment.get_metadata("nonexistent") is None
    
    def test_repr(self):
        """Test string representation."""
        fragment = MemoryFragment(data={"test": "data"})
        fragment.add_metadata("meta", "value")
        
        repr_str = repr(fragment)
        assert "MemoryFragment" in repr_str
        assert "test" in repr_str
        assert "meta" in repr_str