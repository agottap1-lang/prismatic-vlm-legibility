# Comprehensive Analysis: VLM-Based Legibility Evaluation

## Executive Summary

**Problem**: Use Vision-Language Model (Prismatic) to evaluate robot motion legibility - can the VLM infer which goal a robot is moving toward?

**Dataset**: 8 videos (block picking, drawer closing tasks) from LIBERO benchmark

**Ground Truth Performance Target**: Humans achieve >90% accuracy in inferring robot intent from motion

## All Experiments Conducted

### 1. Baseline: Frame Montages (No Annotations)
**Approach**: Show VLM 6 frames in horizontal/grid/vertical layout, ask "which goal?"

**Results**:
- Horizontal layout: ~50% accuracy
- Grid layout: ~50% accuracy  
- Vertical layout: ~50% accuracy

**Failure Mode**: Systematic LEFT bias - VLM predicts "Goal A (left)" regardless of actual motion
- le_r_block (ground truth: RIGHT) → VLM predicts LEFT
- le_l_block (ground truth: LEFT) → VLM predicts LEFT (correct by chance)

**Conclusion**: Layout doesn't matter. VLM cannot infer motion from frame sequences.

---

### 2. Color Markers (RED/BLUE Bounding Boxes)
**Approach**: Add RED box on left block, BLUE box on right block. Change prompt to use "RED region" and "BLUE region" instead of "left/right"

**Hypothesis**: Spatial grounding issue - maybe VLM can't understand "left" and "right"

**Results**: 50.0% accuracy (4/8 correct)
- VLM can SEE the colors ("robot approaching RED region")
- But still predicts wrong goal (says "approaching RED" when actually moving to BLUE)

**Conclusion**: NOT a spatial grounding issue. VLM has positional bias unrelated to color perception.

---

### 3. Trajectory Annotation (Green Path + Arrows + Goal Markers)
**Approach**: Draw explicit trajectory visualization on montage:
- Green line connecting gripper positions across frames
- Yellow arrows showing direction
- "START" and "CURRENT" markers
- Large arrow from start to current position
- RED/BLUE goal region boxes

**Results**: 37.5% accuracy (3/8 correct) - **WORSE than baseline!**

**VLM Behavior**:
- For le_r_block (ground truth: RIGHT): "The arrow points toward RED" → predicts LEFT ❌
- For le_l_block (ground truth: LEFT): "The arrow points toward RED" → predicts LEFT ✓
- VLM says "arrow points toward RED" for ALL videos regardless of actual arrow direction

**Conclusion**: VLM sees the arrow but MISINTERPRETS its direction. Adding visual cues made it worse.

---

### 4. Horizontal Montage with Dots + Trace
**Approach**: 
- 12 frames in horizontal layout
- Red/yellow/green dots on gripper in each frame
- Orange polyline connecting dots
- Arrowheads on trajectory

**Results**: Not yet tested (just created)

**Expected Issue**: Similar to trajectory annotation - VLM will see visual features but may misinterpret them.

---

## Root Cause Analysis

### Prismatic VLM Architecture
```python
# From prismatic/models/vlms/prismatic.py
def forward(self, pixel_values):
    # Process image through vision backbone
    patch_features = self.vision_backbone(pixel_values)  # → [B, patches, dim]
    
    # Project to LLM space
    projected = self.projector(patch_features)  # → [B, patches, llm_dim]
    
    # LLM processes as static context
    return self.llm.generate(projected, prompt_tokens)
```

**Key Limitation**: 
- Vision backbone (DinoSigLIP) processes image as **SINGLE STATIC IMAGE**
- No temporal modeling, no frame-to-frame comparison
- No motion detection, no optical flow
- Montage is just a wide image with frames side-by-side

### Why Visual Annotations Fail
- **Trajectory overlays**: VLM sees lines/arrows as part of the scene, can't reliably determine their semantic meaning
- **Spatial reasoning**: VLM has poor spatial grounding ("arrow pointing left" vs "arrow pointing right")
- **Positional bias**: Some systematic bias in training data causes LEFT predictions

---

## What Doesn't Work

❌ **Frame montages** - No temporal understanding  
❌ **Color markers** - Sees colors but has positional bias  
❌ **Trajectory overlays** - Misinterprets visual annotations  
❌ **Arrows and markers** - Cannot reliably interpret spatial directions  

---

## What We Haven't Tried: RAW FRAMES + SEQUENTIAL REASONING

### Core Idea
**Don't modify images. Use prompting to guide frame-by-frame analysis.**

### Approach
1. **Show VLM multiple RAW frames** (no overlays, no annotations)
2. **Use chain-of-thought prompting** to guide sequential reasoning:
   - "Describe frame 1"
   - "Describe frame 2"
   - "What changed between frame 1 and 2?"
   - "Based on these changes, which direction is the robot moving?"
   - "Which goal region aligns with this direction?"

### Why This Might Work
- **Leverages VLM's strong vision capabilities** without requiring temporal architecture
- **Explicit reasoning chain** forces systematic analysis vs immediate prediction
- **Differential analysis** (frame N vs frame N+1) is simpler than full trajectory inference
- **Memory/context** builds up across frames through conversation

### Implementation Plan (Below in Todo)

---

## Alternative Solutions if RAW Frames Fail

### Option 1: Hybrid CV + VLM
- Use computer vision (optical flow, gripper tracking) to extract trajectory
- Ask VLM only to interpret trajectory direction given clear motion vector
- Example: "The motion vector points 45° to the right. Goal A is left, Goal B is right. Which goal?"

### Option 2: Video-Native Models
- **GPT-4V** (supports video input natively)
- **Video-LLaMA** (trained on video understanding)
- **Gemini 1.5 Pro** (native video understanding)
- These models have temporal convolutions, frame differencing, optical flow

### Option 3: Task-Specific Fine-Tuning
- Collect dataset of (video, trajectory, goal) triples
- Fine-tune Prismatic on legibility prediction task
- Requires significant data collection effort

---

## Next Steps: RAW FRAMES APPROACH

See TODO list below for implementation plan.

**Success Criteria**: >70% accuracy on block videos, >60% on drawer videos

**If successful**: This becomes the production pipeline

**If unsuccessful**: Move to Option 1 (Hybrid CV+VLM) or Option 2 (Video-Native Models)
