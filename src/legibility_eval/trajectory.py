"""
trajectory.py

Trajectory detection and visualization for robot legibility evaluation.

Key insight: Prismatic VLM cannot infer motion from frame sequences.
Solution: Explicitly visualize trajectory with arrows, motion trails, and markers.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image


def detect_gripper_position(frame: np.ndarray, scene_type: str = "block_scene") -> Optional[Tuple[int, int]]:
    """
    Detect robot gripper position in a frame.
    
    Args:
        frame: RGB image as numpy array
        scene_type: Scene identifier for detection strategy
    
    Returns:
        (x, y) gripper center position, or None if not detected
    """
    # Convert to HSV for color-based detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    
    if scene_type == "block_scene":
        # Gripper is typically dark/metallic - look for gray regions
        # Define gray color range (low saturation, medium value)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        
        # Create mask for gripper regions
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour in upper portion of frame (gripper is usually above blocks)
        valid_contours = [c for c in contours if cv2.moments(c)['m00'] > 100]  # Min size filter
        
        if not valid_contours:
            return None
        
        # Get contour in upper half of image (y < height/2)
        height = frame.shape[0]
        upper_contours = []
        for c in valid_contours:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if cy < height * 0.7:  # Upper 70% of frame
                    upper_contours.append(c)
        
        if not upper_contours:
            # Fallback: use largest contour anywhere
            largest_contour = max(valid_contours, key=cv2.contourArea)
        else:
            largest_contour = max(upper_contours, key=cv2.contourArea)
        
        # Get centroid
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        return (cx, cy)
    
    return None


def compute_trajectory(gripper_positions: List[Optional[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    """
    Compute smooth trajectory from detected gripper positions.
    
    Args:
        gripper_positions: List of (x, y) positions, may contain None for missed detections
    
    Returns:
        Smoothed trajectory positions (interpolates missing points)
    """
    # Filter out None values
    valid_positions = [(i, pos) for i, pos in enumerate(gripper_positions) if pos is not None]
    
    if len(valid_positions) < 2:
        return [pos for pos in gripper_positions if pos is not None]
    
    # Interpolate missing positions
    trajectory = []
    for i in range(len(gripper_positions)):
        if gripper_positions[i] is not None:
            trajectory.append(gripper_positions[i])
        else:
            # Find nearest valid positions before and after
            before = [p for j, p in valid_positions if j < i]
            after = [p for j, p in valid_positions if j > i]
            
            if before and after:
                # Linear interpolation
                t = (i - valid_positions[len(before)-1][0]) / (valid_positions[len(before)][0] - valid_positions[len(before)-1][0])
                x = int(before[-1][0] + t * (after[0][0] - before[-1][0]))
                y = int(before[-1][1] + t * (after[0][1] - before[-1][1]))
                trajectory.append((x, y))
            elif before:
                trajectory.append(before[-1])
            else:
                trajectory.append(after[0])
    
    return trajectory


def detect_goal_regions(frame: np.ndarray, scene_type: str = "block_scene") -> Dict[str, Tuple[int, int, int, int]]:
    """
    Detect goal region positions (same as block_detection.detect_blocks).
    
    Args:
        frame: RGB image as numpy array
        scene_type: Scene identifier
    
    Returns:
        Dictionary with goal region bounding boxes
    """
    if scene_type == "block_scene":
        # Fixed positions for block scene (based on empirical measurements)
        height, width = frame.shape[:2]
        
        # Blocks are in lower portion, left and right thirds
        block_y_center = int(height * 0.65)
        block_height = int(height * 0.2)
        block_width = int(width * 0.15)
        
        left_x_center = int(width * 0.25)
        right_x_center = int(width * 0.75)
        
        return {
            'goal_A': (
                left_x_center - block_width // 2,
                block_y_center - block_height // 2,
                left_x_center + block_width // 2,
                block_y_center + block_height // 2
            ),
            'goal_B': (
                right_x_center - block_width // 2,
                block_y_center - block_height // 2,
                right_x_center + block_width // 2,
                block_y_center + block_height // 2
            )
        }
    
    return {}


def annotate_trajectory_on_montage(
    montage: Image.Image,
    frames: List[Image.Image],
    timestamps: List[float],
    scene_type: str = "block_scene",
    layout: str = "horizontal",
    grid_cols: int = 3,
    frame_size: Tuple[int, int] = (320, 320)
) -> Image.Image:
    """
    Annotate trajectory visualization on a frame montage.
    
    This is the KEY function that converts temporal reasoning → spatial reasoning.
    
    Args:
        montage: The frame montage image
        frames: Original frames
        timestamps: Frame timestamps
        scene_type: Scene identifier for detection
        layout: Montage layout ("horizontal", "grid", "vertical")
        grid_cols: Columns for grid layout
        frame_size: Size of each frame in montage
    
    Returns:
        Annotated montage image with trajectory visualization
    """
    # Convert montage to numpy for drawing
    montage_array = np.array(montage)
    
    # Detect gripper in each original frame
    gripper_positions = []
    for frame in frames:
        frame_array = np.array(frame)
        pos = detect_gripper_position(frame_array, scene_type)
        gripper_positions.append(pos)
    
    print(f"[Trajectory] Detected gripper in {sum(1 for p in gripper_positions if p is not None)}/{len(frames)} frames")
    
    # Compute smooth trajectory
    trajectory = compute_trajectory(gripper_positions)
    
    if len(trajectory) < 2:
        print("[Trajectory] Warning: Could not detect sufficient trajectory points")
        return montage
    
    # Map trajectory positions to montage coordinates
    montage_trajectory = []
    for i, pos in enumerate(trajectory):
        if pos is None:
            continue
        
        # Calculate frame position in montage
        if layout == "horizontal":
            frame_x_offset = i * frame_size[0]
            frame_y_offset = 0
        elif layout == "vertical":
            frame_x_offset = 0
            frame_y_offset = i * frame_size[1]
        elif layout == "grid":
            row = i // grid_cols
            col = i % grid_cols
            frame_x_offset = col * frame_size[0]
            frame_y_offset = row * frame_size[1]
        else:
            frame_x_offset = 0
            frame_y_offset = 0
        
        # Add frame offset to local position
        montage_x = frame_x_offset + pos[0]
        montage_y = frame_y_offset + pos[1]
        montage_trajectory.append((montage_x, montage_y))
    
    # === DRAW TRAJECTORY VISUALIZATION ===
    
    # 1. Draw motion trail (gradient from transparent to opaque)
    for i in range(len(montage_trajectory) - 1):
        alpha = int(255 * (i + 1) / len(montage_trajectory))  # Fade in effect
        thickness = 3 + int(5 * (i + 1) / len(montage_trajectory))  # Thicker toward end
        
        cv2.line(
            montage_array,
            montage_trajectory[i],
            montage_trajectory[i + 1],
            (0, 255, 0),  # GREEN trail
            thickness,
            cv2.LINE_AA
        )
    
    # 2. Draw gripper position dots
    for i, pos in enumerate(montage_trajectory):
        radius = 6 if i == len(montage_trajectory) - 1 else 4  # Larger for final position
        cv2.circle(montage_array, pos, radius, (255, 255, 0), -1)  # YELLOW dots
        cv2.circle(montage_array, pos, radius + 1, (0, 0, 0), 2)  # Black outline
    
    # 3. Detect goal regions in LAST frame
    last_frame_array = np.array(frames[-1])
    goal_regions = detect_goal_regions(last_frame_array, scene_type)
    
    if goal_regions:
        # Calculate goal positions in montage
        last_frame_idx = len(frames) - 1
        
        if layout == "horizontal":
            last_frame_x_offset = last_frame_idx * frame_size[0]
            last_frame_y_offset = 0
        elif layout == "vertical":
            last_frame_x_offset = 0
            last_frame_y_offset = last_frame_idx * frame_size[1]
        elif layout == "grid":
            row = last_frame_idx // grid_cols
            col = last_frame_idx % grid_cols
            last_frame_x_offset = col * frame_size[0]
            last_frame_y_offset = row * frame_size[1]
        else:
            last_frame_x_offset = 0
            last_frame_y_offset = 0
        
        # 4. Draw goal region markers
        if 'goal_A' in goal_regions:
            x1, y1, x2, y2 = goal_regions['goal_A']
            montage_x1 = last_frame_x_offset + x1
            montage_y1 = last_frame_y_offset + y1
            montage_x2 = last_frame_x_offset + x2
            montage_y2 = last_frame_y_offset + y2
            
            # RED box for Goal A
            cv2.rectangle(montage_array, (montage_x1, montage_y1), (montage_x2, montage_y2), (255, 0, 0), 6)
            cv2.putText(montage_array, "GOAL A (RED)", (montage_x1, max(montage_y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        if 'goal_B' in goal_regions:
            x1, y1, x2, y2 = goal_regions['goal_B']
            montage_x1 = last_frame_x_offset + x1
            montage_y1 = last_frame_y_offset + y1
            montage_x2 = last_frame_x_offset + x2
            montage_y2 = last_frame_y_offset + y2
            
            # BLUE box for Goal B
            cv2.rectangle(montage_array, (montage_x1, montage_y1), (montage_x2, montage_y2), (0, 0, 255), 6)
            cv2.putText(montage_array, "GOAL B (BLUE)", (montage_x1, max(montage_y1 - 10, 20)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 5. Draw DIRECTIONAL ARROW from final gripper position to nearest goal
        if montage_trajectory:
            final_pos = montage_trajectory[-1]
            
            # Calculate distances to each goal
            goal_A_center = (
                last_frame_x_offset + (goal_regions['goal_A'][0] + goal_regions['goal_A'][2]) // 2,
                last_frame_y_offset + (goal_regions['goal_A'][1] + goal_regions['goal_A'][3]) // 2
            )
            goal_B_center = (
                last_frame_x_offset + (goal_regions['goal_B'][0] + goal_regions['goal_B'][2]) // 2,
                last_frame_y_offset + (goal_regions['goal_B'][1] + goal_regions['goal_B'][3]) // 2
            )
            
            dist_A = np.sqrt((final_pos[0] - goal_A_center[0])**2 + (final_pos[1] - goal_A_center[1])**2)
            dist_B = np.sqrt((final_pos[0] - goal_B_center[0])**2 + (final_pos[1] - goal_B_center[1])**2)
            
            # Draw arrow toward closer goal
            target = goal_A_center if dist_A < dist_B else goal_B_center
            color = (255, 0, 0) if dist_A < dist_B else (0, 0, 255)  # RED or BLUE
            
            # Draw thick arrow
            cv2.arrowedLine(
                montage_array,
                final_pos,
                target,
                color,
                thickness=10,
                tipLength=0.3,
                line_type=cv2.LINE_AA
            )
            
            # Add "MOTION DIRECTION" label
            mid_point = ((final_pos[0] + target[0]) // 2, (final_pos[1] + target[1]) // 2)
            cv2.putText(
                montage_array,
                "MOTION →",
                (mid_point[0] - 60, mid_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3
            )
            cv2.putText(
                montage_array,
                "MOTION →",
                (mid_point[0] - 60, mid_point[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )
    
    # Convert back to PIL Image
    annotated_montage = Image.fromarray(montage_array)
    return annotated_montage


if __name__ == "__main__":
    # Test trajectory detection
    from src.legibility_eval.frames import extract_frame_sequence, create_frame_montage
    
    print("Testing trajectory detection and annotation...")
    
    # Extract frames
    frames, timestamps = extract_frame_sequence(
        video_path="videos/le r block.mp4",
        t_current=5.0,
        fps=30.0,
        n_frames=6
    )
    
    print(f"Extracted {len(frames)} frames")
    
    # Create base montage
    montage = create_frame_montage(frames, timestamps, layout="horizontal")
    montage.save("test_trajectory_base.png")
    print("Base montage saved to test_trajectory_base.png")
    
    # Annotate trajectory
    annotated = annotate_trajectory_on_montage(
        montage, frames, timestamps,
        scene_type="block_scene",
        layout="horizontal"
    )
    annotated.save("test_trajectory_annotated.png")
    print("Annotated montage saved to test_trajectory_annotated.png")
    print("\n✓ Trajectory annotation test complete!")
