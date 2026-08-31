"""Core model classes for memory fragments."""

from typing import Any, Dict, List, Optional


class MemoryFragment:
    """A class representing a memory fragment."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize a memory fragment.
        
        Args:
            data: Optional dictionary containing fragment data
        """
        self.data = data or {}
        self.timestamp = None
        self.metadata = {}
    
    def process(self) -> Dict[str, Any]:
        """Process the memory fragment.
        
        Returns:
            Processed fragment data
        """
        # Placeholder implementation
        processed_data = self.data.copy()
        processed_data['processed'] = True
        return processed_data
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the fragment.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str) -> Any:
        """Get metadata value by key.
        
        Args:
            key: Metadata key
            
        Returns:
            Metadata value or None if not found
        """
        return self.metadata.get(key)
    
    def __repr__(self) -> str:
        """String representation of the memory fragment."""
        return f"MemoryFragment(data={self.data}, metadata={self.metadata})"