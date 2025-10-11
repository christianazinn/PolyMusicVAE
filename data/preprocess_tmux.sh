#!/bin/bash

# set params here
NUM_BARS=1
MAX_SEQ_LEN=256  # seems reasonable based on limited testing?

subfolders=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "a" "b" "c" "d" "e" "f")
SESSION_NAME="midi_processing"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' already exists: please kill first"
    exit 1
fi

echo "Creating tmux session $SESSION_NAME"
echo ""

tmux new-session -d -s "$SESSION_NAME" -n "subfolder-${subfolders[0]}"
tmux send-keys -t "$SESSION_NAME:subfolder-${subfolders[0]}" "source ../../.venv/bin/activate && python preprocess_subfolder.py ${subfolders[0]} $NUM_BARS $MAX_SEQ_LEN && exit" Enter

for i in "${!subfolders[@]}"; do
    if [ $i -eq 0 ]; then
        continue  # already did the first one
    fi
    
    subfolder="${subfolders[$i]}"
    window_name="subfolder-$subfolder"
    
    echo "Creating window: $window_name"
    
    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "$SESSION_NAME:$window_name" "source ../../.venv/bin/activate && python preprocess_subfolder.py $subfolder $NUM_BARS $MAX_SEQ_LEN && exit" Enter
done

echo ""
echo "All windows created and processing started!"
echo ""
echo "To monitor progress:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "Tmux navigation:"
echo "  SUPER+n          - Next window"
echo "  SUPER+p          - Previous window" 
echo "  SUPER+0-9        - Go to window 0-9"
echo "  SUPER+w          - List all windows"
echo "  SUPER+d          - Detach (leave running)"
echo ""
echo "To kill all processing:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""

echo "Attaching to session..."
sleep 2
tmux attach-session -t "$SESSION_NAME"