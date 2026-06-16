# Legibility Evaluation Pipeline

Clean, organized structure for evaluating robot motion legibility using Vision-Language Models.

## Directory Structure

```
prismatic-vlms/
├── src/
│   └── legibility_eval/           # Core evaluation modules
│       ├── horizontal_montage.py   # NEW: Horizontal layout with red dots + arrows
│       ├── checkpoint_evaluator.py # Checkpoint-based evaluation
│       ├── task_agnostic_prompt.py # Task-agnostic prompting
│       └── trajectory_encoding.py  # Trajectory extraction utilities
│
├── tests/                          # Test scripts (organized)
│   ├── test_horizontal_montage.py  # NEW: Test horizontal montage approach
│   ├── test_checkpoint_evaluation.py
│   └── test_color_markers.py
│
├── outputs/
│   ├── images/                     # All generated images go here
│   └── *.jsonl                     # Model outputs
│
├── results/                        # Evaluation results (JSON)
│   └── *.json
│
├── videos/                         # Test videos
│   └── *.mp4
│
├── data/
│   └── manifest.jsonl             # Video metadata
│
├── evaluate.py                     # Main evaluation script
└── README.md                       # This file
```

## Current Approach: Horizontal Montage

**Visual Design:**
- 6 frames arranged horizontally (left to right = time progression)
- **Red dots** mark the end effector position in each frame
- **Green arrows** connect frames to show temporal flow
- White background for clean presentation

**Prompt Strategy:**
- Task-agnostic: asks VLM to infer which spatial region is the probable goal
- Explicitly explains visual cues (red dots, green arrows)
- Requests numerical scores 0-100 for each possible goal region
- JSON output format for easy parsing

**Example Usage:**

```python
from src.legibility_eval.horizontal_montage import (
    create_horizontal_montage_checkpoint,
    build_horizontal_montage_prompt
)

# Generate montage at t=5s with 6 frames
img = create_horizontal_montage_checkpoint(
    "videos/le r block.mp4", 
    checkpoint_time=5.0,
    fps=30.0,
    num_frames=6
)

# Save image
img.save("outputs/images/my_montage.png")

# Get prompt
prompt = build_horizontal_montage_prompt(n_goal_regions=2)

# Query VLM
response = vlm.generate_text(image=img, prompt_text=prompt)
```

## Running Tests

```bash
# Test horizontal montage approach
cd tests
python test_horizontal_montage.py

# View generated images
ls ../outputs/images/

# View results
cat ../results/horizontal_montage_test.json
```

## Key Files

- **horizontal_montage.py** - Main implementation for the new approach
- **test_horizontal_montage.py** - Test script with full evaluation pipeline
- **outputs/images/horizontal_montage_t5s.png** - Example output image

## Previous Approaches (for reference)

All previous experiments are documented in `results/`:
- `layout_benchmark_*.json` - Frame montage layouts (horizontal/grid/vertical)
- `color_marker_test.json` - Color bounding boxes
- `trajectory_annotation_test.json` - Trajectory overlays on montage

These approaches achieved 37-50% accuracy due to spatial grounding issues.

## Next Steps

1. Test horizontal montage on full dataset (8 videos)
2. Compare accuracy against baseline (50%)
3. If successful (>70%), consider production deployment
4. If unsuccessful, explore hybrid CV+VLM or video-native models (GPT-4V, Video-LLaMA)
