import pandas as pd

def load_data(file_path):
    """Load dataset from the given file path."""
    # Display CSV structure before loading
    print(f"Reading CSV file: {file_path}")
    print("Displaying first 5 lines to understand structure...")
    
    try:
        # Read only the first few lines to inspect structure
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 6:  # Show header + 5 rows
                    break
                print(line.strip())
                
        # Read the full CSV
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded {len(df)} rows from CSV")
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
