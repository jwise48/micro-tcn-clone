#!/bin/bash
#
# SignalTrain LA-2A Dataset Inspector
# Analyzes dataset properties and split methodology
#

# Dataset path (can be overridden with argument)
DATASET_PATH="${1:-/home/hice1/jwise48/scratch/data/micro-tcn/SignalTrain_LA2A_Dataset_1.1}"

echo "======================================================"
echo "SignalTrain LA-2A Dataset Inspector"
echo "======================================================"
echo ""
echo "Dataset Path: $DATASET_PATH"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Function to analyze a subset
analyze_subset() {
    local subset_name=$1
    local subset_dir="$DATASET_PATH/$subset_name"
    
    echo "-----------------------------------------------------"
    echo "$subset_name Subset"
    echo "-----------------------------------------------------"
    
    if [ ! -d "$subset_dir" ]; then
        echo "  Directory not found: $subset_dir"
        return
    fi
    
    # Count files
    local input_files=$(find "$subset_dir" -name "input_*.wav" -type f | wc -l)
    local target_files=$(find "$subset_dir" -name "target_*.wav" -type f | wc -l)
    local total_files=$((input_files + target_files))
    
    echo "  Audio Files:"
    echo "    Input files:  $input_files"
    echo "    Target files: $target_files"
    echo "    Total files:  $total_files"
    
    # Get size
    local size=$(du -sh "$subset_dir" 2>/dev/null | awk '{print $1}')
    echo "  Directory Size: $size"
    
    # Analyze first file for audio properties
    local first_file=$(find "$subset_dir" -name "target_*.wav" -type f | head -1)
    
    if [ -n "$first_file" ]; then
        echo ""
        echo "  Audio Properties (from first file):"
        
        # Try using soxi (from sox package)
        if command -v soxi &> /dev/null; then
            local sample_rate=$(soxi -r "$first_file" 2>/dev/null)
            local channels=$(soxi -c "$first_file" 2>/dev/null)
            local duration=$(soxi -d "$first_file" 2>/dev/null)
            local precision=$(soxi -b "$first_file" 2>/dev/null)
            
            echo "    Sample Rate: ${sample_rate} Hz"
            echo "    Channels: $channels ($([ $channels -eq 1 ] && echo 'Mono' || echo 'Stereo'))"
            echo "    Bit Depth: ${precision} bits"
            echo "    Duration: ${duration} seconds"
            
        # Fallback to ffprobe
        elif command -v ffprobe &> /dev/null; then
            echo "    Using ffprobe for analysis..."
            ffprobe -v quiet -print_format json -show_streams "$first_file" | \
                python3 -c "import sys, json; d=json.load(sys.stdin)['streams'][0]; \
                print(f\"    Sample Rate: {d['sample_rate']} Hz\"); \
                print(f\"    Channels: {d['channels']} ({'Mono' if d['channels']==1 else 'Stereo'})\"); \
                print(f\"    Duration: {float(d.get('duration', 0)):.2f} seconds\")"
        else
            echo "    (Install 'sox' or 'ffmpeg' for audio analysis)"
        fi
        
        # Calculate total duration estimate
        if [ $target_files -gt 0 ] && [ -n "$duration" ]; then
            total_minutes=$(echo "$duration * $target_files / 60" | bc -l)
            printf "    Estimated Total: %.2f minutes\n" $total_minutes
        fi
    fi
    
    # Extract unique parameter configurations
    echo ""
    echo "  Parameter Configurations:"
    local param_configs=$(find "$subset_dir" -name "target_*.wav" -type f -exec basename {} \; | \
                         sed 's/target_[0-9]*__//; s/.wav//' | \
                         sort -u | head -20)
    
    local num_configs=$(echo "$param_configs" | wc -l)
    echo "    Unique configs: $num_configs"
    echo "    Sample configs (format: param1__param2):"
    echo "$param_configs" | head -10 | sed 's/^/      /'
    
    if [ $num_configs -gt 10 ]; then
        echo "      ... (showing first 10 of $num_configs)"
    fi
    
    echo ""
}

# Analyze each subset
analyze_subset "Train"
analyze_subset "Val"
analyze_subset "Test"

# Overall summary
echo "======================================================"
echo "Dataset Summary"
echo "======================================================"

total_size=$(du -sh "$DATASET_PATH" 2>/dev/null | awk '{print $1}')
echo "Total Dataset Size: $total_size"
echo ""

# Count total files across all subsets
total_train=$(find "$DATASET_PATH/Train" -name "*.wav" 2>/dev/null | wc -l)
total_val=$(find "$DATASET_PATH/Val" -name "*.wav" 2>/dev/null | wc -l)
total_test=$(find "$DATASET_PATH/Test" -name "*.wav" 2>/dev/null | wc -l)
total_all=$((total_train + total_val + total_test))

echo "Split Distribution:"
echo "  Train: $total_train files ($(echo "scale=1; $total_train * 100 / $total_all" | bc)%)"
echo "  Val:   $total_val files ($(echo "scale=1; $total_val * 100 / $total_all" | bc)%)"
echo "  Test:  $total_test files ($(echo "scale=1; $total_test * 100 / $total_all" | bc)%)"
echo ""

echo "Split Methodology:"
echo "  - Pre-defined splits in separate directories"
echo "  - Each subset contains input/target pairs"
echo "  - Parameters encoded in filenames"
echo "  - LA-2A compressor with 2 parameters"
echo ""

echo "======================================================"

