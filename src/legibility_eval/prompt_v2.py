"""
Improved prompting strategy - descriptive goals instead of spatial terms.

Key insight: Avoid spatial terms like "left/right" which create ambiguity.
Instead, use descriptive terms from the visual cues themselves.
"""

import re
import json
from typing import Tuple, Dict, Optional


def build_chain_of_thought_prompt(
    goal_A_desc: str,
    goal_B_desc: str,
    t_sec: int,
    video_id: str,
    n_frames: int = 6,
    mode: str = "prefix_frames"
) -> str:
    """
    Build chain-of-thought prompt that asks VLM to reason through motion direction first.
    
    Strategy:
    1. Ask what the gripper is moving toward (descriptive, visual)
    2. Then ask which goal matches that direction
    3. Then estimate probabilities
    """
    
    if mode == "prefix_frames":
        context_text = f"""You are shown {n_frames} frames from t=0 to t={t_sec}s from a robot manipulation task.
Frames are ordered chronologically (earliest to latest).

Task: Infer the robot's goal from its motion trajectory."""
    else:
        context_text = f"""You are shown 1 frame at t={t_sec}s from a robot manipulation task.

Task: Infer the robot's goal from what you see."""
    
    return f"""{context_text}

Step 1: Describe the motion
- What direction is the gripper moving?
- What object or location is the gripper approaching?

Step 2: Match to goals
There are two candidate goals:
- Goal A: {goal_A_desc}
- Goal B: {goal_B_desc}

Which goal better matches the observed motion direction?

Step 3: Estimate probabilities
Based on the motion trajectory:
- pA = probability of Goal A
- pB = probability of Goal B
(constraints: 0 ≤ pA, pB ≤ 1 and pA + pB ≈ 1)

Step 4: Legibility
Is the goal clearly inferable now ("legible_now") or still ambiguous ("not_legible_yet")?

Respond with JSON only:
{{"motion_description": "...", "pA": 0.X, "pB": 0.X, "legible": "legible_now"}}
"""


def build_descriptive_goals_prompt(
    video_id: str,
    t_sec: int,
    n_frames: int = 6,
    mode: str = "prefix_frames"
) -> Tuple[str, str]:
    """
    Build goal-agnostic prompt - first ask VLM to describe what it sees,
    then we'll interpret manually.
    
    Returns:
        (initial_prompt, followup_prompt) - two-stage questioning
    """
    
    if mode == "prefix_frames":
        context = f"You see {n_frames} frames (t=0 to t={t_sec}s) from a robot manipulation task."
    else:
        context = f"You see 1 frame at t={t_sec}s from a robot manipulation task."
    
    initial_prompt = f"""{context}

Describe:
1. What objects do you see? (blocks, drawer, etc.)
2. Where are they located in the image? (left side, right side, center, etc.)
3. Where is the robot gripper?
4. What direction is the gripper moving toward?

Be specific about spatial locations."""
    
    followup_prompt = """Based on your description, if there are two candidate goals:
- Goal A: pick the left block
- Goal B: pick the right block

Which goal is the gripper moving toward? Provide probabilities pA and pB.
Respond with JSON: {"pA": 0.X, "pB": 0.X, "cue": "...", "legible": "legible_now"}"""
    
    return initial_prompt, followup_prompt


def build_zero_indexed_goals_prompt(
    goal_descriptions: list,
    t_sec: int,
    video_id: str,
    n_frames: int = 6,
    mode: str = "prefix_frames"
) -> str:
    """
    Build prompt with numbered goals (avoid spatial terms entirely).
    
    Args:
        goal_descriptions: List like ["pick the red block", "pick the blue block"]
    """
    
    if mode == "prefix_frames":
        context = f"Frames from t=0 to t={t_sec}s ({n_frames} frames total)."
    else:
        context = f"Frame at t={t_sec}s."
    
    goals_text = "\n".join([f"Goal {i}: {desc}" for i, desc in enumerate(goal_descriptions)])
    
    return f"""{context}

Robot manipulation task with {len(goal_descriptions)} candidate goals:
{goals_text}

Based on the gripper's motion trajectory in the frame(s):
- Estimate p(Goal 0) and p(Goal 1)
- Describe the visual cue supporting your estimate
- Is the goal clearly inferable? ("legible_now" or "not_legible_yet")

Respond with JSON only:
{{"p0": 0.X, "p1": 0.X, "cue": "...", "legible": "legible_now"}}
"""


def build_spatial_explicit_prompt(
    goal_A_desc: str,
    goal_B_desc: str,
    t_sec: int,
    video_id: str,
    n_frames: int = 6,
    mode: str = "prefix_frames"
) -> str:
    """
    Build prompt with EXPLICIT spatial instructions to help VLM understand frame orientation.
    
    Key addition: Tell VLM how to interpret "left" and "right" in the image.
    """
    
    if mode == "prefix_frames":
        context = f"{n_frames} frames from t=0 to t={t_sec}s (ordered earliest to latest)."
    else:
        context = f"1 frame at t={t_sec}s."
    
    return f"""{context}

SPATIAL ORIENTATION:
- LEFT means: left side of the image (negative X direction if viewing from camera)
- RIGHT means: right side of the image (positive X direction if viewing from camera)
- When looking at the image, use YOUR left and YOUR right as the observer

CANDIDATE GOALS:
- Goal A: {goal_A_desc}
- Goal B: {goal_B_desc}

TASK:
1. Identify where objects are in the image (left side vs right side)
2. Identify where the gripper is moving toward
3. Match the motion to Goal A or Goal B
4. Estimate pA = P(Goal A | motion) and pB = P(Goal B | motion)
5. Determine if goal is clearly inferable ("legible_now") or ambiguous ("not_legible_yet")

Respond with JSON only:
{{"pA": 0.X, "pB": 0.X, "cue": "gripper moving toward [description]", "legible": "legible_now"}}

CRITICAL: "left" and "right" refer to positions in the IMAGE, from YOUR perspective as the viewer.
"""


def parse_probability_response(response: str) -> Dict[str, any]:
    """Parse VLM probability response (handles multiple formats)."""
    pA, pB = 0.5, 0.5
    cue = ""
    legible = "not_legible_yet"
    
    # Try JSON parsing
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join([l for l in lines if not l.strip().startswith("```")])
        
        data = json.loads(cleaned)
        
        # Handle different formats
        if "pA" in data and "pB" in data:
            pA = float(data.get("pA", 0.5))
            pB = float(data.get("pB", 0.5))
        elif "p0" in data and "p1" in data:
            pA = float(data.get("p0", 0.5))
            pB = float(data.get("p1", 0.5))
        
        cue = data.get("cue", data.get("motion_description", ""))
        legible = data.get("legible", "not_legible_yet")
        
    except (json.JSONDecodeError, ValueError):
        # Fallback: regex parsing
        pA_match = re.search(r'["\']?pA["\']?\s*:\s*([\d.]+)', response, re.IGNORECASE)
        pB_match = re.search(r'["\']?pB["\']?\s*:\s*([\d.]+)', response, re.IGNORECASE)
        
        if pA_match:
            pA = float(pA_match.group(1))
        if pB_match:
            pB = float(pB_match.group(1))
        
        # Extract cue
        cue_match = re.search(r'["\']?cue["\']?\s*:\s*["\']([^"\']+)["\']', response, re.IGNORECASE)
        if cue_match:
            cue = cue_match.group(1)
    
    return {
        "pA": pA,
        "pB": pB,
        "cue": cue,
        "legible": legible,
        "raw": response,
    }


def calculate_decision_metrics(pA: float, pB: float, threshold: float = 0.6) -> Tuple[str, float]:
    """
    Calculate choice and confidence from probabilities.
    
    Args:
        pA: Probability of Goal A
        pB: Probability of Goal B
        threshold: Minimum probability to make a decision
    
    Returns:
        (choice, confidence) where choice is "A", "B", or "None"
    """
    if pA > threshold and pA > pB:
        return "A", pA
    elif pB > threshold and pB > pA:
        return "B", pB
    else:
        return "None", max(pA, pB)
