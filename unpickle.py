# pickle_to_json.py
import pickle
import json
import sys
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

def pickle_to_json(pickle_file, json_file=None):
    if json_file is None:
        json_file = pickle_file.replace('.p', '.json')
    
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    print(f"✓ Converted: {pickle_file} -> {json_file}")
    return data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python unpickle.py <pickle_file> [output_json]")
        sys.exit(1)
    
    pickle_file = sys.argv[1]
    json_file = sys.argv[2] if len(sys.argv) > 2 else None
    pickle_to_json(pickle_file, json_file)

