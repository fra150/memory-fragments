from memory_fragments import MemoryFragment


def main():
    """Run the memory fragments demo."""
    print("Memory Fragments Demo")
    print("=" * 20)
    
    # Create a memory fragment with some sample data
    sample_data = {
        "content": "This is a sample memory fragment",
        "type": "text",
        "importance": 0.8
    }
    
    print("\n1. Creating a memory fragment...")
    fragment = MemoryFragment(data=sample_data)
    print(f"Created: {fragment}")
    
    # Add some metadata
    print("\n2. Adding metadata...")
    fragment.add_metadata("created_by", "demo_script")
    fragment.add_metadata("version", "1.0")
    print(f"Metadata added: {fragment.metadata}")
    
    # Process the fragment
    print("\n3. Processing the fragment...")
    result = fragment.process()
    print(f"Processing result: {result}")
    
    # Retrieve metadata
    print("\n4. Retrieving metadata...")
    creator = fragment.get_metadata("created_by")
    version = fragment.get_metadata("version")
    print(f"Creator: {creator}")
    print(f"Version: {version}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()