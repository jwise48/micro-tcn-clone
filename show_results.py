import json
import pandas as pd
import numpy as np

# === TABLE 1: Model Comparison ===
def generate_table1(test_results_path):
    with open(test_results_path) as f:
        results = json.load(f)
    
    table_data = []
    for model_name, configs in results.items():
        # Extract model properties
        is_causal = '-C__causal' in model_name
        rf = int(model_name.split('-')[2].split('-')[0])
        
        # Aggregate metrics
        all_metrics = {'L1': [], 'STFT': [], 'LUFS': []}
        for config_metrics in configs.values():
            for key in all_metrics:
                all_metrics[key].extend(config_metrics[key])
        
        table_data.append({
            'Model': model_name.split('__')[0],
            'Causal': is_causal,
            'RF (ms)': rf,
            'MAE': f"{np.mean(all_metrics['L1']):.2e}",
            'STFT': f"{np.mean(all_metrics['STFT']):.3f}",
            'LUFS': f"{np.mean(all_metrics['LUFS']):.3f}"
        })
    
    return pd.DataFrame(table_data)

# === TABLE 2: Speed Benchmarks ===
def generate_speed_table(speed_csv_path, frame_size=2048):
    df = pd.read_csv(speed_csv_path)
    
    # Filter for specific frame size
    df_filtered = df[df['N'] == frame_size]
    
    # Select your models
    models = ['TCN-100-C', 'TCN-300-C', 'TCN-1000-C', 
              'TCN-100-N', 'TCN-300-N', 'TCN-1000-N']
    
    return df_filtered[df_filtered['model_id'].isin(models)][
        ['model_id', 'rf', 'rtf']
    ]

# Generate tables
table1 = generate_table1('test_results_val.json')
table_speed = generate_speed_table('speed_gpu.csv')

print("=== TABLE 1: Model Performance ===")
print(table1.to_string(index=False))

print("\n=== Speed at Frame Size 2048 ===")
print(table_speed.to_string(index=False))

