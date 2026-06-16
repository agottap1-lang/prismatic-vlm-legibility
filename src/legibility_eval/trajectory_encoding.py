"""
trajectory_encoding.py

Encode robot trajectory as STATIC GEOMETRY in a single image.
This converts temporal reasoning → spatial reasoning.

Key principle: Motion is represented as visible geometric features (lines, arrows, dots)
that the VLM can reason about spatially.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


def extract_trajectory_points(
    video_path: str,
    t_start: float,
    t_end: float,
    fps: float,
    sample_rate: float = 0.2
) -> List[Tuple[int, int]]:
    """
    Extract gripper positions from video between t_start and t_end.
    
    Args:
        video_path: Path to video file
        t_start: Start time (usually 0.0)
        t_end: End time (checkpoint time)
        fps: Video FPS
        sample_rate: Sample every N seconds (default 0.2s = 5 samples/sec)
    
    Returns:
        List of (x, y) positions
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    # Sample times from t_start to t_end
    times = np.arange(t_start, t_end + sample_rate, sample_rate)
    times = times[times <= t_end]
    
    positions = []
    prev_frame = None
    
    for t in times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect gripper position (using motion if prev_frame available)
        pos = detect_gripper_simple(frame_rgb, prev_frame)
        if pos is not None:
            positions.append(pos)
        
        prev_frame = frame_rgb
    
    cap.release()
    return positions


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


def render_trajectory_overlay(
    base_frame: np.ndarray,
    trajectory_points: List[Tuple[int, int]],
    checkpoint_progress: float,
) -> np.ndarray:
    """
    Render trajectory as static geometry on a single frame.
    
    This is the KEY function that encodes motion as visible geometric features.
    
    Args:
        base_frame: Base image (current frame at checkpoint time)
        trajectory_points: List of (x, y) positions from start to checkpoint
        checkpoint_progress: Progress through trajectory (0.0 to 1.0)
    
    Returns:
        Annotated frame with trajectory overlay
    """
    # Work on copy
    overlay = base_frame.copy()
    
    if len(trajectory_points) < 2:
        return overlay
    
    # 1. Draw TRAJECTORY TRACE (smooth path line)
    # Use thick green line to show the path taken
    for i in range(len(trajectory_points) - 1):
        # Gradient thickness: thinner at start, thicker at end
        thickness = 2 + int(4 * (i + 1) / len(trajectory_points))
        
        cv2.line(
            overlay,
            trajectory_points[i],
            trajectory_points[i + 1],
            (0, 255, 100),  # Green trajectory
            thickness,
            cv2.LINE_AA
        )
    
    # 2. Draw DIRECTIONAL ARROWHEADS along path
    # Place arrows every ~20% of trajectory to show direction
    num_arrows = min(4, len(trajectory_points) // 3)
    arrow_indices = np.linspace(len(trajectory_points) // 4, len(trajectory_points) - 1, num_arrows, dtype=int)
    
    for idx in arrow_indices:
        if idx > 0:
            start_pt = trajectory_points[idx - 1]
            end_pt = trajectory_points[idx]
            
            # Draw small arrow
            cv2.arrowedLine(
                overlay,
                start_pt,
                end_pt,
                (255, 200, 0),  # Yellow-orange arrows
                2,
                tipLength=0.4,
                line_type=cv2.LINE_AA
            )
    
    # 3. Draw START MARKER (green dot)
    start_pos = trajectory_points[0]
    cv2.circle(overlay, start_pos, 8, (0, 255, 0), -1)  # Green fill
    cv2.circle(overlay, start_pos, 8, (0, 0, 0), 2)     # Black outline
    
    # 4. Draw CURRENT POSITION MARKER (red dot)
    current_pos = trajectory_points[-1]
    cv2.circle(overlay, current_pos, 10, (255, 0, 0), -1)  # Red fill
    cv2.circle(overlay, current_pos, 10, (0, 0, 0), 2)     # Black outline
    
    # 5. Add TEXT LABELS (small, unobtrusive)
    cv2.putText(
        overlay,
        "START",
        (start_pos[0] - 25, start_pos[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )
    
    cv2.putText(
        overlay,
        "CURRENT",
        (current_pos[0] - 30, current_pos[1] + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )
    
    # 6. Draw MOTION VECTOR (large arrow from start to current)
    # This is the primary visual cue for trajectory direction
    if len(trajectory_points) >= 3:
        # Use smoothed vector (not just first to last)
        mid_idx = len(trajectory_points) // 3
        reference_pt = trajectory_points[mid_idx]
        
        cv2.arrowedLine(
            overlay,
            reference_pt,
            current_pos,
            (255, 100, 0),  # Orange motion vector
            6,
            tipLength=0.25,
            line_type=cv2.LINE_AA
        )
    
    return overlay


def create_checkpoint_image(
    video_path: str,
    checkpoint_time: float,
    fps: float,
    total_duration: float
) -> Image.Image:
    """
    Create a single image encoding trajectory up to checkpoint time.
    
    This is the main interface function.
    
    Args:
        video_path: Path to video
        checkpoint_time: Time of checkpoint (e.g., 2.0s for 20% of 10s video)
        fps: Video FPS
        total_duration: Total video duration
    
    Returns:
        PIL Image with trajectory overlay
    """
    import cv2
    
    # 1. Get base frame at checkpoint time
    cap = cv2.VideoCapture(video_path)
    frame_num = int(checkpoint_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, base_frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read frame at t={checkpoint_time}s")
    
    # Convert BGR to RGB
    base_frame_rgb = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
    
    # 2. Extract trajectory points from 0 to checkpoint_time
    trajectory_points = extract_trajectory_points(
        video_path=video_path,
        t_start=0.0,
        t_end=checkpoint_time,
        fps=fps,
        sample_rate=0.1  # Sample every 0.1 seconds
    )
    
    if not trajectory_points:
        # Return base frame if no trajectory detected
        return Image.fromarray(base_frame_rgb)
    
    # 3. Render trajectory overlay
    checkpoint_progress = checkpoint_time / total_duration
    overlay = render_trajectory_overlay(
        base_frame=base_frame_rgb,
        trajectory_points=trajectory_points,
        checkpoint_progress=checkpoint_progress
    )
    
    # 4. Convert to PIL Image
    return Image.fromarray(overlay)


def get_checkpoint_times(total_duration: float, checkpoints: List[float] = [0.1, 0.2, 0.4, 0.6, 0.8]) -> List[float]:
    """
    Get checkpoint times based on percentage of total duration.
    
    Args:
        total_duration: Total video duration in seconds
        checkpoints: List of progress percentages (0.0 to 1.0)
    
    Returns:
        List of checkpoint times in seconds
    """
    return [total_duration * cp for cp in checkpoints]


if __name__ == "__main__":
    # Test trajectory encoding
    print("Testing trajectory encoding...")
    
    video_path = "videos/le r block.mp4"
    
    # Get video info
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Duration: {duration:.2f}s")
    
    # Create checkpoint images at 20%, 40%, 60%, 80%
    checkpoints = [0.2, 0.4, 0.6, 0.8]
    checkpoint_times = get_checkpoint_times(duration, checkpoints)
    
    for i, (cp, t) in enumerate(zip(checkpoints, checkpoint_times)):
        print(f"\nCheckpoint {int(cp*100)}% (t={t:.2f}s)...")
        
        image = create_checkpoint_image(
            video_path=video_path,
            checkpoint_time=t,
            fps=fps,
            total_duration=duration
        )
        
        output_path = f"test_checkpoint_{int(cp*100)}.png"
        image.save(output_path)
        print(f"Saved: {output_path}")
    
    print("\n✓ Trajectory encoding test complete!")
