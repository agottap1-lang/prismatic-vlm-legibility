"""
Sequential reasoning prompts for raw frames approach.

Key principle: NO visual annotations. Motion understanding comes from
explicit frame-by-frame reasoning guided by prompts.
"""

from typing import List, Dict, Any


def build_sequential_reasoning_prompt(
    n_frames: int = 6,
    scene_description: str = "robotic manipulation",
    goal_descriptions: Dict[str, str] = None
) -> str:
    """
    Build chain-of-thought prompt for sequential frame analysis.
    
    This prompt guides the VLM to:
    1. Analyze each frame independently
    2. Compare consecutive frames to detect changes
    3. Infer motion direction from cumulative changes
    4. Predict which goal region aligns with the motion
    
    Args:
        n_frames: Number of frames shown
        scene_description: Brief scene context
        goal_descriptions: Dict mapping goal IDs to descriptions
    
    Returns:
        Prompt string
    """
    if goal_descriptions is None:
        goal_descriptions = {
            "A": "one possible goal location in the scene",
            "B": "another possible goal location in the scene"
        }
    
    prompt = f"""You are analyzing a sequence of {n_frames} frames from a {scene_description} task. The frames are shown in CHRONOLOGICAL ORDER (left to right or top to bottom).

Your goal is to determine which of two possible goal regions the robot is moving toward.

**AVAILABLE GOALS:**
"""
    
    for goal_id, goal_desc in goal_descriptions.items():
        prompt += f"- **Goal {goal_id}**: {goal_desc}\n"
    
    prompt += f"""
**INSTRUCTIONS:**

Follow this reasoning process step by step:

1. **FRAME-BY-FRAME DESCRIPTION**:
   - For each frame, describe the robot's configuration (arm position, gripper location, orientation)
   - Note the positions of objects in the scene
   - Identify any visible goal regions

2. **MOTION ANALYSIS**:
   - Compare consecutive frames: What changed?
   - Which direction did the robot arm move? (left, right, up, down, toward, away)
   - What is the overall trajectory pattern across all frames?

3. **GOAL INFERENCE**:
   - Based on the motion direction, which goal region is the robot moving toward?
   - What is your confidence level? (0-100)

4. **PROVIDE YOUR ANSWER** in this JSON format:
```json
{{
    "frame_descriptions": [
        "Frame 1: <description>",
        "Frame 2: <description>",
        ...
    ],
    "motion_observations": [
        "Frame 1→2: <what changed>",
        "Frame 2→3: <what changed>",
        ...
    ],
    "trajectory_summary": "<overall motion pattern>",
    "predicted_goal": "A or B",
    "confidence": <0-100>,
    "reasoning": "<explanation of why this goal was chosen>"
}}
```

**CRITICAL**: Your answer must be based ONLY on what you observe in the frames - the robot's position and movement. Do NOT make assumptions about task labels or names.
"""
    
    return prompt


def build_differential_analysis_prompt(
    frame_pair_index: int,
    total_pairs: int
) -> str:
    """
    Build prompt for analyzing a single pair of consecutive frames.
    
    Used for multi-turn conversation approach where each frame pair
    is analyzed separately, then results are aggregated.
    
    Args:
        frame_pair_index: Index of current pair (0-based)
        total_pairs: Total number of frame pairs
    
    Returns:
        Prompt string
    """
    prompt = f"""You are analyzing frame pair {frame_pair_index + 1} of {total_pairs} from a robot manipulation task.

**TASK**: Compare these two consecutive frames and describe what changed.

Focus on:
1. **Robot arm position**: Did it move? In which direction?
2. **Gripper location**: Where was it in frame 1? Where is it in frame 2?
3. **Movement direction**: Describe the motion (left, right, up, down, toward object, away)
4. **Distance traveled**: Approximately how much did the robot move?

Provide your answer in this format:
```json
{{
    "frame1_robot_position": "<description>",
    "frame2_robot_position": "<description>",
    "motion_direction": "<direction>",
    "motion_magnitude": "<small/medium/large>",
    "observations": "<any other notable changes>"
}}
```
"""
    return prompt


def build_aggregation_prompt(
    motion_summaries: List[Dict[str, Any]],
    goal_descriptions: Dict[str, str]
) -> str:
    """
    Build prompt to aggregate motion observations and infer final goal.
    
    Used after analyzing all frame pairs to synthesize overall trajectory.
    
    Args:
        motion_summaries: List of motion analysis results from frame pairs
        goal_descriptions: Dict mapping goal IDs to descriptions
    
    Returns:
        Prompt string
    """
    prompt = """Based on the frame-by-frame motion analysis below, determine which goal the robot is moving toward.

**MOTION OBSERVATIONS:**
"""
    
    for i, summary in enumerate(motion_summaries):
        prompt += f"\nFrame pair {i+1}:\n"
        prompt += f"- Direction: {summary.get('motion_direction', 'unknown')}\n"
        prompt += f"- Magnitude: {summary.get('motion_magnitude', 'unknown')}\n"
        prompt += f"- Notes: {summary.get('observations', '')}\n"
    
    prompt += "\n**AVAILABLE GOALS:**\n"
    for goal_id, goal_desc in goal_descriptions.items():
        prompt += f"- **Goal {goal_id}**: {goal_desc}\n"
    
    prompt += """
**YOUR TASK**: 
1. Synthesize the motion observations into an overall trajectory pattern
2. Determine which goal region this trajectory is heading toward
3. Provide confidence level (0-100)

Respond in JSON format:
```json
{
    "trajectory_pattern": "<description of overall motion>",
    "predicted_goal": "A or B",
    "confidence": <0-100>,
    "reasoning": "<why this goal based on the motion pattern>"
}
```
"""
    
    return prompt


def build_simple_spatial_prompt(
    goal_descriptions: Dict[str, str]
) -> str:
    """
    Simplified prompt for single-pass analysis (no chain-of-thought).
    
    Tests if VLM can directly infer goal from raw frames without
    explicit reasoning steps.
    
    Args:
        goal_descriptions: Dict mapping goal IDs to descriptions
    
    Returns:
        Prompt string
    """
    prompt = """You are viewing a sequence of frames from a robot manipulation task shown in chronological order.

**TASK**: Determine which goal the robot is moving toward.

**AVAILABLE GOALS:**
"""
    
    for goal_id, goal_desc in goal_descriptions.items():
        prompt += f"- **Goal {goal_id}**: {goal_desc}\n"
    
    prompt += """
Look at the robot's position across the frames. Which direction is it moving? Which goal does this motion lead toward?

Respond in JSON format:
```json
{
    "predicted_goal": "A or B",
    "confidence": <0-100>,
    "reasoning": "<brief explanation>"
}
```
"""
    
    return prompt


# Test each prompt strategy
if __name__ == "__main__":
    print("=== STRATEGY 1: FULL CHAIN-OF-THOUGHT ===")
    print(build_sequential_reasoning_prompt(
        n_frames=6,
        scene_description="block picking",
        goal_descriptions={
            "A": "the left red block",
            "B": "the right blue block"
        }
    ))
    
    print("\n\n=== STRATEGY 2: DIFFERENTIAL ANALYSIS (Frame Pair) ===")
    print(build_differential_analysis_prompt(frame_pair_index=0, total_pairs=5))
    
    print("\n\n=== STRATEGY 3: SIMPLE SPATIAL ===")
    print(build_simple_spatial_prompt(goal_descriptions={
        "A": "the left region",
        "B": "the right region"
    }))
