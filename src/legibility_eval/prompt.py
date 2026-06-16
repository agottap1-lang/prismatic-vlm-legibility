"""
Probability-based prompting for legibility evaluation.

VLM provides pA and pB directly, code calculates choice and confidence.
"""

import re
import json
from typing import Tuple, Dict


def build_probability_prompt(goal_A_desc: str, goal_B_desc: str, t_sec: int, video_id: str, n_frames: int = 6, mode: str = "prefix_frames", use_color_markers: bool = False, use_trajectory: bool = False) -> str:
    """
    Build prompt asking VLM to provide probabilities for each goal.
    
    Args:
        goal_A_desc: Description of Goal A
        goal_B_desc: Description of Goal B
        t_sec: Current timestamp in seconds
        video_id: Video identifier
        n_frames: Number of frames shown (default 6)
        mode: Evaluation mode - "single_frame" or "prefix_frames"
        use_color_markers: If True, use color-based references instead of spatial terms
        use_trajectory: If True, use trajectory annotation visual cues
    
    Returns:
        Prompt string
    """
    
    # Context depends on evaluation mode
    if mode == "prefix_frames":
        context_text = f"""You are evaluating LEGIBILITY: how easily a typical human observer can infer the actor's intended goal from what they have observed.

You are given MULTIPLE images showing frames from t=0 to t={t_sec} seconds from video_id = "{video_id}".
Frames are ordered from earliest to latest; you have observed the motion up to time t={t_sec}s.
Use ALL frames provided to estimate the goal probabilities."""
    else:  # single_frame
        context_text = f"""You are evaluating LEGIBILITY: how easily a typical human observer can infer the actor's intended goal from what is visible NOW.

You are given ONLY ONE image: a single video frame captured at time t = {t_sec} seconds from video_id = "{video_id}".
Use ONLY this frame. Do NOT assume you saw earlier or later frames."""
    
    if use_trajectory:
        # Trajectory-annotated prompt (RECOMMENDED - explicit visual cues)
        return f"""{context_text}

The image shows a robot arm with TRAJECTORY VISUALIZATION:
- YELLOW DOTS mark the robot gripper position in each frame
- GREEN LINE shows the motion trail (trajectory path)
- THICK ARROW shows the direction of motion
- RED BOX marks Goal A region: {goal_A_desc}
- BLUE BOX marks Goal B region: {goal_B_desc}

The trajectory visualization explicitly shows WHERE the robot is moving.

Based on the ARROW DIRECTION and motion trail:
- If the arrow points toward the RED region, the robot is moving toward Goal A
- If the arrow points toward the BLUE region, the robot is moving toward Goal B

Estimate probabilities:
- pA = P(robot moving toward RED/Goal A | trajectory visualization)
- pB = P(robot moving toward BLUE/Goal B | trajectory visualization)

Constraints:
- 0 <= pA,pB <= 1
- pA + pB = 1 (within rounding)

Provide EXACTLY ONE short visual cue describing the arrow direction and motion pattern.
Also output legibility:
- "legible_now" if the arrow clearly points toward one goal
- "not_legible_yet" if the motion direction is still ambiguous

Output ONLY valid JSON with keys: pA, pB, cue, legible.
No markdown. No extra text. No code fences.
Example format:
{{"pA": 0.15, "pB": 0.85, "cue": "thick arrow points toward BLUE goal region", "legible": "legible_now"}}
"""
    
    elif use_color_markers:
        # Color-based prompt (no spatial language)
        return f"""{context_text}

In the images, you will see colored bounding boxes marking two possible goal regions:
- RED box marks one goal region
- BLUE box marks another goal region

The robot's task involves interacting with one of these marked regions.

Goal A: {goal_A_desc} (marked with RED box)
Goal B: {goal_B_desc} (marked with BLUE box)

Based on the robot gripper's position, trajectory, and motion pattern in the provided frame(s):

Estimate probabilities:
- pA = P(robot is moving toward RED region | observed frames)
- pB = P(robot is moving toward BLUE region | observed frames)

Constraints:
- 0 <= pA,pB <= 1
- pA + pB = 1 (within rounding)

Provide EXACTLY ONE short visual cue from the frame(s) that supports your probabilities.
Also output legibility:
- "legible_now" if a typical human could confidently infer the goal now
- "not_legible_yet" if the motion is still ambiguous

Output ONLY valid JSON with keys: pA, pB, cue, legible.
No markdown. No extra text. No code fences.
Example format:
{{"pA": 0.75, "pB": 0.25, "cue": "gripper trajectory approaching RED region", "legible": "legible_now"}}
"""
    else:
        # Original spatial-based prompt
        return f"""{context_text}

There are exactly two candidate goals:
Goal A: {goal_A_desc}
Goal B: {goal_B_desc}

Estimate probabilities using the provided frame(s):
- pA = P(Goal A | frames)
- pB = P(Goal B | frames)
Constraints:
- 0 <= pA,pB <= 1
- pA + pB = 1 (within rounding)

Provide EXACTLY ONE short visual cue from the frame(s) that supports your probabilities.
Also output legibility:
- "legible_now" if a typical human could infer the goal now, else "not_legible_yet".

Output ONLY valid JSON with keys: pA, pB, cue, legible.
No markdown. No extra text. No code fences.
Example format:
{{"pA": 0.62, "pB": 0.38, "cue": "gripper aligned with left block", "legible": "legible_now"}}
"""


def parse_probability_response(response: str) -> Dict[str, any]:
    """
    Parse VLM probability response (tries JSON first, then fallback parsing).
    
    Args:
        response: Raw VLM response
    
    Returns:
        dict with pA, pB, cue, legible, raw
    """
    pA = 0.5  # default
    pB = 0.5  # default
    cue = ""
    legible = "not_legible_yet"
    
    # Try JSON parsing first
    try:
        # Clean up response - remove markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove code fence markers
            lines = cleaned.split("\n")
            cleaned = "\n".join([l for l in lines if not l.strip().startswith("```")])
        
        data = json.loads(cleaned)
        pA = float(data.get("pA", 0.5))
        pB = float(data.get("pB", 0.5))
        cue = data.get("cue", "")
        legible = data.get("legible", "not_legible_yet")
        
        # Clamp probabilities
        pA = max(0.0, min(1.0, pA))
        pB = max(0.0, min(1.0, pB))
        
    except (json.JSONDecodeError, ValueError, KeyError):
        # Fallback: line-by-line parsing
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("pA:") or '"pA"' in line:
                try:
                    pA = float(re.findall(r"[0-9.]+", line)[0])
                    pA = max(0.0, min(1.0, pA))
                except (IndexError, ValueError):
                    pA = 0.5
            elif line.startswith("pB:") or '"pB"' in line:
                try:
                    pB = float(re.findall(r"[0-9.]+", line)[0])
                    pB = max(0.0, min(1.0, pB))
                except (IndexError, ValueError):
                    pB = 0.5
            elif line.startswith("CUE:") or '"cue"' in line:
                cue = re.sub(r'^(CUE:|"cue":)\s*', '', line).strip(' "')
            elif '"legible"' in line or line.startswith("legible:"):
                if "legible_now" in line:
                    legible = "legible_now"
                else:
                    legible = "not_legible_yet"
    
    return {
        "pA": pA,
        "pB": pB,
        "cue": cue,
        "legible": legible,
        "raw": response,
    }


def calculate_decision_metrics(pA: float, pB: float) -> Tuple[str, float]:
    """
    Calculate decision and confidence from VLM-provided probabilities.
    
    Args:
        pA: Probability for Goal A (from VLM)
        pB: Probability for Goal B (from VLM)
    
    Returns:
        (choice, confidence) where choice is 'A', 'B', or None
    """
    threshold = 0.6
    
    if pA > threshold and pA > pB:
        return ("A", pA)
    elif pB > threshold and pB > pA:
        return ("B", pB)
    else:
        return (None, max(pA, pB))
