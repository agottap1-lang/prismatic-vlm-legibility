"""
task_agnostic_prompt.py

Task-agnostic prompting system for trajectory intent recognition.

Does NOT assume specific task types (block picking, drawer closing, etc.).
Uses generic spatial reasoning about trajectory direction and goal regions.
"""

from typing import Dict, List, Optional


def build_trajectory_intent_prompt(
    n_goal_regions: int = 2,
    include_confidence: bool = True
) -> str:
    """
    Build task-agnostic prompt for trajectory intent recognition.
    
    This prompt:
    - Does NOT mention specific tasks (blocks, drawers, etc.)
    - Asks about SPATIAL trajectory direction
    - Requests NUMERICAL values (not just text descriptions)
    - Works for any goal-directed motion task
    
    Args:
        n_goal_regions: Number of possible goal regions (default 2)
        include_confidence: Whether to ask for confidence scores
    
    Returns:
        Prompt string
    """
    
    if n_goal_regions == 2:
        return """You are analyzing a robot trajectory image.

The image shows:
- A GREEN DOT marking where motion started (START)
- A RED DOT marking the current position (CURRENT)  
- A GREEN PATH showing the trajectory taken from START to CURRENT
- ARROWS showing the direction of motion

There are TWO possible goal regions where the robot might be heading.

Your task: Analyze the trajectory geometry and determine which goal region the robot is moving toward.

Consider:
1. The DIRECTION of the trajectory path (which way is it curving?)
2. The POSITION of the current point relative to potential goals
3. The MOMENTUM indicated by the motion arrows

Respond with scores between 0 and 100 for each region:
- score_region_1: How strongly the trajectory points toward region 1 (0-100)
- score_region_2: How strongly the trajectory points toward region 2 (0-100)

The scores should sum to approximately 100.

Also provide:
- trajectory_direction: One word describing the trajectory ("leftward", "rightward", "upward", "downward", "central", "curved-left", "curved-right")
- reasoning: One sentence explaining your score assignment

Output ONLY valid JSON with these exact keys: score_region_1, score_region_2, trajectory_direction, reasoning.
No markdown. No code fences. No extra text.

Example:
{"score_region_1": 75, "score_region_2": 25, "trajectory_direction": "leftward", "reasoning": "Trajectory curves strongly toward left side of frame"}
"""
    
    else:
        # Multi-goal version (3+ regions)
        region_scores = ", ".join([f'"score_region_{i+1}": 0-100' for i in range(n_goal_regions)])
        
        return f"""You are analyzing a robot trajectory image.

The image shows:
- A GREEN DOT marking where motion started (START)
- A RED DOT marking the current position (CURRENT)
- A GREEN PATH showing the trajectory taken from START to CURRENT
- ARROWS showing the direction of motion

There are {n_goal_regions} possible goal regions where the robot might be heading.

Your task: Analyze the trajectory geometry and determine which goal region the robot is moving toward.

Consider:
1. The DIRECTION of the trajectory path
2. The POSITION of the current point relative to potential goals  
3. The MOMENTUM indicated by the motion arrows

Respond with scores between 0 and 100 for each region:
{region_scores}

The scores should sum to approximately 100.

Also provide:
- trajectory_direction: Brief description of trajectory direction
- reasoning: One sentence explaining your score assignment

Output ONLY valid JSON. No markdown. No code fences.
"""


def parse_trajectory_response(response: str) -> Dict:
    """
    Parse VLM response for trajectory intent.
    
    Args:
        response: Raw VLM response string
    
    Returns:
        Dictionary with parsed values
    """
    import json
    import re
    
    # Try direct JSON parse first
    try:
        result = json.loads(response.strip())
        return {
            "score_region_1": result.get("score_region_1", 50),
            "score_region_2": result.get("score_region_2", 50),
            "trajectory_direction": result.get("trajectory_direction", "unknown"),
            "reasoning": result.get("reasoning", ""),
            "raw": response
        }
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code fence
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            return {
                "score_region_1": result.get("score_region_1", 50),
                "score_region_2": result.get("score_region_2", 50),
                "trajectory_direction": result.get("trajectory_direction", "unknown"),
                "reasoning": result.get("reasoning", ""),
                "raw": response
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: extract values with regex
    score1_match = re.search(r'score_region_1["\']?\s*:\s*(\d+)', response)
    score2_match = re.search(r'score_region_2["\']?\s*:\s*(\d+)', response)
    direction_match = re.search(r'trajectory_direction["\']?\s*:\s*["\']([^"\']+)["\']', response)
    reasoning_match = re.search(r'reasoning["\']?\s*:\s*["\']([^"\']+)["\']', response)
    
    score1 = int(score1_match.group(1)) if score1_match else 50
    score2 = int(score2_match.group(1)) if score2_match else 50
    direction = direction_match.group(1) if direction_match else "unknown"
    reasoning = reasoning_match.group(1) if reasoning_match else ""
    
    return {
        "score_region_1": score1,
        "score_region_2": score2,
        "trajectory_direction": direction,
        "reasoning": reasoning,
        "raw": response
    }


def compute_intent_metrics(score_region_1: int, score_region_2: int, threshold: int = 60) -> Dict:
    """
    Compute intent metrics from region scores.
    
    Args:
        score_region_1: Score for region 1 (0-100)
        score_region_2: Score for region 2 (0-100)
        threshold: Threshold for confident decision (default 60)
    
    Returns:
        Dictionary with computed metrics
    """
    # Normalize scores to probabilities
    total = score_region_1 + score_region_2
    if total == 0:
        p1, p2 = 0.5, 0.5
    else:
        p1 = score_region_1 / total
        p2 = score_region_2 / total
    
    # Determine choice
    if score_region_1 >= threshold and score_region_1 > score_region_2:
        choice = "A"
        confidence = score_region_1 / 100.0
    elif score_region_2 >= threshold and score_region_2 > score_region_1:
        choice = "B"
        confidence = score_region_2 / 100.0
    else:
        choice = "uncertain"
        confidence = max(score_region_1, score_region_2) / 100.0
    
    # Legibility assessment
    score_diff = abs(score_region_1 - score_region_2)
    if score_diff >= 40:  # Clear separation
        legible = "legible_now"
    elif score_diff >= 20:  # Some separation
        legible = "becoming_legible"
    else:  # Ambiguous
        legible = "not_legible_yet"
    
    return {
        "choice": choice,
        "confidence": confidence,
        "legible": legible,
        "probability_A": p1,
        "probability_B": p2,
        "score_difference": score_diff
    }


def map_choice_to_ground_truth(
    choice: str,
    goal_A_is_region_1: bool
) -> str:
    """
    Map VLM choice (based on spatial position) to ground truth goal labels.
    
    Args:
        choice: VLM choice ("A", "B", "uncertain")
        goal_A_is_region_1: Whether ground truth Goal A corresponds to spatial region 1
    
    Returns:
        Mapped choice
    """
    if choice == "uncertain":
        return "uncertain"
    
    # If Goal A is spatial region 1, mapping is direct
    if goal_A_is_region_1:
        return choice
    else:
        # Goal A is spatial region 2, so flip
        return "B" if choice == "A" else "A"


if __name__ == "__main__":
    # Test prompt generation
    print("=" * 60)
    print("TASK-AGNOSTIC TRAJECTORY INTENT PROMPT")
    print("=" * 60)
    print(build_trajectory_intent_prompt())
    
    print("\n" + "=" * 60)
    print("TEST RESPONSE PARSING")
    print("=" * 60)
    
    # Test parsing
    test_response_1 = '{"score_region_1": 75, "score_region_2": 25, "trajectory_direction": "leftward", "reasoning": "Path curves left"}'
    result_1 = parse_trajectory_response(test_response_1)
    print(f"\nClean JSON: {result_1}")
    
    test_response_2 = '''```json
{
    "score_region_1": 30,
    "score_region_2": 70,
    "trajectory_direction": "rightward",
    "reasoning": "Motion arrows point right"
}
```'''
    result_2 = parse_trajectory_response(test_response_2)
    print(f"\nMarkdown JSON: {result_2}")
    
    # Test metrics
    print("\n" + "=" * 60)
    print("TEST METRICS COMPUTATION")
    print("=" * 60)
    
    metrics_1 = compute_intent_metrics(75, 25)
    print(f"\nScore 75-25: {metrics_1}")
    
    metrics_2 = compute_intent_metrics(55, 45)
    print(f"\nScore 55-45: {metrics_2}")
    
    print("\n✓ Task-agnostic prompting system ready!")
