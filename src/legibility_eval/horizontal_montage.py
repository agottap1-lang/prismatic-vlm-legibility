"""
Horizontal montage with red dots and connecting arrows.

This approach shows multiple frames in a horizontal layout with:
- Red dots marking the end effector position in each frame
- Green arrows connecting frames to show temporal progression
- Task-agnostic prompt asking VLM to infer probable goal
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple


def detect_gripper_simple(frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Optional[Tuple[int, int]]:
    """
    Detect gripper using motion detection or edge-based fallback.
    
    Args:
        frame: RGB image as numpy array
        prev_frame: Previous frame for motion detection (optional)
    
    Returns:
        (x, y) position or None
    """
    h, w = frame.shape[:2]
    
    # STRATEGY 1: Motion-based detection
    if prev_frame is not None:
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        
        diff = cv2.absdiff(gray_curr, gray_prev)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 200:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy)
    
    # STRATEGY 2: Edge-based detection (fallback)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Focus on central region
    center_region = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 10000:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if center_region[0] < cx < center_region[2] and center_region[1] < cy < center_region[3]:
                    valid_contours.append((cnt, cx, cy, area))
    
    if not valid_contours:
        return None
    
    _, cx, cy, _ = max(valid_contours, key=lambda x: x[3])
    return (cx, cy)


def render_horizontal_montage(
    frames_with_positions: List[Tuple[np.ndarray, Optional[Tuple[int, int]]]],
    frame_width: int = 256,
    frame_height: int = 256,
) -> np.ndarray:
    """
    Create horizontal montage with red dots on gripper and arrows between frames.
    
    Args:
        frames_with_positions: List of (frame, (x,y) position) tuples
        frame_width: Target width for each frame
        frame_height: Target height for each frame
    
    Returns:
        Montage image with visual cues
    """
    num_frames = len(frames_with_positions)
    if num_frames == 0:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Arrow spacing between frames
    arrow_width = 60
    total_width = num_frames * frame_width + (num_frames - 1) * arrow_width
    
    # Create canvas (white background)
    montage = np.ones((frame_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place frames with connecting arrows (no individual red dots)
    for i, (frame, pos) in enumerate(frames_with_positions):
        # Resize frame
        resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Place frame in montage
        x_offset = i * (frame_width + arrow_width)
        montage[:, x_offset:x_offset + frame_width] = resized
        
        # Draw arrow to next frame
        if i < num_frames - 1:
            arrow_start_x = x_offset + frame_width
            arrow_end_x = arrow_start_x + arrow_width
            arrow_y = frame_height // 2
            
            # Draw thick green arrow showing temporal progression
            cv2.arrowedLine(
                montage,
                (arrow_start_x + 10, arrow_y),
                (arrow_end_x - 10, arrow_y),
                (0, 200, 0),  # Green arrow
                5,
                cv2.LINE_AA,
                tipLength=0.3
            )
    
    return montage


def render_horizontal_montage_with_trajectory(
    frames_with_positions: List[Tuple[np.ndarray, Optional[Tuple[int, int]]]],
    frame_width: int = 256,
    frame_height: int = 256,
) -> np.ndarray:
    """
    Create horizontal montage with trajectory overlay:
    - Polyline trace connecting all gripper positions
    - Arrowheads on the trace
    - Green dot at start position
    - Red dot at current position
    
    Args:
        frames_with_positions: List of (frame, (x,y) position) tuples
        frame_width: Target width for each frame
        frame_height: Target height for each frame
    
    Returns:
        Montage image with trajectory visualization
    """
    num_frames = len(frames_with_positions)
    if num_frames == 0:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Arrow spacing between frames
    arrow_width = 60
    total_width = num_frames * frame_width + (num_frames - 1) * arrow_width
    
    # Create canvas (white background)
    montage = np.ones((frame_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place frames and collect scaled positions
    scaled_positions = []
    
    for i, (frame, pos) in enumerate(frames_with_positions):
        # Resize frame
        resized = cv2.resize(frame, (frame_width, frame_height))
        
        # Place frame in montage
        x_offset = i * (frame_width + arrow_width)
        montage[:, x_offset:x_offset + frame_width] = resized
        
        # Scale and store position for trajectory
        if pos is not None:
            orig_h, orig_w = frame.shape[:2]
            scale_x = frame_width / orig_w
            scale_y = frame_height / orig_h
            scaled_x = int(pos[0] * scale_x) + x_offset
            scaled_y = int(pos[1] * scale_y)
            scaled_positions.append((scaled_x, scaled_y))
        
        # Draw arrow to next frame
        if i < num_frames - 1:
            arrow_start_x = x_offset + frame_width
            arrow_end_x = arrow_start_x + arrow_width
            arrow_y = frame_height // 2
            
            # Draw green arrow showing temporal progression
            cv2.arrowedLine(
                montage,
                (arrow_start_x + 10, arrow_y),
                (arrow_end_x - 10, arrow_y),
                (0, 200, 0),  # Green arrow
                5,
                cv2.LINE_AA,
                tipLength=0.3
            )
    
    # Draw trajectory overlay if we have positions
    if len(scaled_positions) >= 2:
        # 1. Draw polyline trace connecting all positions
        for i in range(len(scaled_positions) - 1):
            cv2.line(
                montage,
                scaled_positions[i],
                scaled_positions[i + 1],
                (0, 150, 255),  # Orange trace
                4,
                cv2.LINE_AA
            )
        
        # 2. Draw arrowheads along the trace
        # Place arrows every other segment to avoid clutter
        for i in range(1, len(scaled_positions) - 1, 2):
            cv2.arrowedLine(
                montage,
                scaled_positions[i],
                scaled_positions[i + 1],
                (0, 100, 255),  # Darker orange for arrows
                3,
                cv2.LINE_AA,
                tipLength=0.4
            )
        
        # 3. Draw dots at ALL gripper positions
        for i, pos in enumerate(scaled_positions):
            if i == 0:
                # GREEN dot at START position
                cv2.circle(montage, pos, 10, (0, 255, 0), -1)  # Green fill
                cv2.circle(montage, pos, 12, (255, 255, 255), 2)  # White border
            elif i == len(scaled_positions) - 1:
                # RED dot at END position
                cv2.circle(montage, pos, 12, (255, 0, 0), -1)  # Red fill
                cv2.circle(montage, pos, 14, (255, 255, 255), 2)  # White border
            else:
                # YELLOW dots at intermediate positions
                cv2.circle(montage, pos, 8, (0, 255, 255), -1)  # Yellow fill
                cv2.circle(montage, pos, 10, (255, 255, 255), 2)  # White border
    
    return montage


def create_horizontal_montage_checkpoint(
    video_path: str,
    checkpoint_time: float,
    fps: float,
    num_frames: int = 6,
) -> Image.Image:
    """
    Create horizontal montage with red dots and arrows at checkpoint time.
    
    Args:
        video_path: Path to video
        checkpoint_time: Time point to evaluate (e.g., 5.0s)
        fps: Video FPS
        num_frames: Number of frames to show in montage (default 6)
    
    Returns:
        PIL Image with horizontal montage
    """
    # Sample evenly spaced frames from start to checkpoint
    times = np.linspace(0.0, checkpoint_time, num_frames)
    
    frames_with_positions = []
    prev_frame = None
    
    cap = cv2.VideoCapture(video_path)
    
    for t in times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect gripper position
        pos = detect_gripper_simple(frame_rgb, prev_frame)
        
        frames_with_positions.append((frame_rgb, pos))
        prev_frame = frame_rgb
    
    cap.release()
    
    # Create horizontal montage with visual cues
    montage = render_horizontal_montage(frames_with_positions)
    
    # Convert to PIL Image
    return Image.fromarray(montage)


def create_horizontal_montage_with_trajectory(
    video_path: str,
    checkpoint_time: float,
    fps: float,
    num_frames: int = 6,
) -> Image.Image:
    """
    Create horizontal montage with trajectory overlay at checkpoint time.
    
    Args:
        video_path: Path to video
        checkpoint_time: Time point to evaluate (e.g., 5.0s)
        fps: Video FPS
        num_frames: Number of frames to show in montage (default 6)
    
    Returns:
        PIL Image with horizontal montage and trajectory overlay
    """
    # Sample evenly spaced frames from start to checkpoint
    times = np.linspace(0.0, checkpoint_time, num_frames)
    
    # First pass: collect all frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    for t in times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    # Second pass: detect positions with motion context
    frames_with_positions = []
    positions = []
    
    for i, frame in enumerate(frames):
        if i == 0:
            # For first frame, use next frame for motion detection (backward)
            next_frame = frames[1] if len(frames) > 1 else None
            pos = detect_gripper_simple(frame, next_frame)
        else:
            # For other frames, use previous frame
            prev_frame = frames[i - 1]
            pos = detect_gripper_simple(frame, prev_frame)
        
        # If detection failed but we have previous positions, interpolate
        if pos is None and len(positions) >= 2:
            # Use linear extrapolation from last two positions
            dx = positions[-1][0] - positions[-2][0]
            dy = positions[-1][1] - positions[-2][1]
            pos = (positions[-1][0] + dx, positions[-1][1] + dy)
        elif pos is None and len(positions) == 1:
            # Use last known position
            pos = positions[-1]
        
        if pos is not None:
            positions.append(pos)
        
        frames_with_positions.append((frame, pos))
    
    # Create horizontal montage with trajectory overlay
    montage = render_horizontal_montage_with_trajectory(frames_with_positions)
    
    # Convert to PIL Image
    return Image.fromarray(montage)


def build_horizontal_montage_prompt(n_goal_regions: int = 2) -> str:
    """
    Build task-agnostic prompt for horizontal montage with red dots and arrows.
    
    Args:
        n_goal_regions: Number of possible goal regions (default 2)
    
    Returns:
        Prompt string
    """
    prompt = f"""You are analyzing a robot manipulation task. You will see a sequence of {6} frames showing the robot's motion over time, displayed horizontally from left to right.

KEY VISUAL CUES:
- RED DOTS: Mark the end effector (gripper) position in each frame
- GREEN ARROWS: Connect frames to show temporal progression (left → right)
- The frames are ordered chronologically from LEFT (start) to RIGHT (most recent)

TASK:
Based on the sequence of red dot positions and the trajectory you observe, determine which of the {n_goal_regions} spatial regions is the MOST PROBABLE goal that the robot is moving toward.

SPATIAL REGIONS:
"""
    
    for i in range(1, n_goal_regions + 1):
        prompt += f"- Region {i}: One of the possible goal locations in the scene\n"
    
    # Build score fields
    score_fields = '"score_region_1": <integer 0-100>,\n    "score_region_2": <integer 0-100>'
    if n_goal_regions > 2:
        for i in range(3, n_goal_regions + 1):
            score_fields += f',\n    "score_region_{i}": <integer 0-100>'
    
    prompt += f"""
REQUIRED OUTPUT (JSON only):
{{
    "trajectory_direction": "brief description of the observed motion direction",
    {score_fields},
    "reasoning": "brief explanation of which region is most probable based on the red dot trajectory"
}}

SCORING GUIDELINES:
- 0-30: Very unlikely to be the goal
- 31-60: Possible but uncertain
- 61-85: Likely goal based on trajectory
- 86-100: Very confident this is the goal

Focus on the SPATIAL TRAJECTORY shown by the red dots across frames. Higher scores should go to regions that align with the observed motion direction."""
    
    return prompt
