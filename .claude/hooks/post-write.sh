#!/bin/bash

# Read JSON from stdin and extract file path
FILE_PATH=$(jq -r '.tool_input.file_path')

# Only run pre-commit if we got a valid file path
if [ -n "$FILE_PATH" ] && [ "$FILE_PATH" != "null" ]; then
    cd "$CLAUDE_PROJECT_DIR"
    # Run pre-commit and capture output
    OUTPUT=$(pre-commit run --files "$FILE_PATH" 2>&1)
    RESULT=$?
    
    if [ $RESULT -ne 0 ]; then
        # Show the full pre-commit output on error
        echo "$OUTPUT" >&2
        # Exit code 2 = blocking error (prevents the edit)
        exit 2
    else
        echo "Pre-commit checks passed" >&2
        # Exit code 0 = success
        exit 0
    fi
fi

exit 0