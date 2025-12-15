#!/bin/bash
# Wrapper script to suppress objc duplicate library warnings on macOS
# These warnings occur when both PyAV's bundled libraries and Homebrew's ffmpeg are present
# They are harmless but annoying.

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Function to filter objc warnings from stderr
filter_objc_warnings() {
    while IFS= read -r line; do
        # Skip lines containing objc duplicate class warnings
        if [[ ! "$line" =~ objc\[.*\]:\ Class.*is\ implemented\ in\ both ]]; then
            echo "$line" >&2
        fi
    done
}

# Run the Python script and filter stderr
python3 "$SCRIPT_DIR/generate_plots_from_trained_models.py" "$@" 2> >(filter_objc_warnings)

