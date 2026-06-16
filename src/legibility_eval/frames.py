"""
Frame utilities for temporal sequence visualization.

Key: Show multiple frames as a single montage image so Prismatic can see temporal progression.
"""

from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def create_frame_montage(
    frames: List[Image.Image],
    timestamps: List[float],
    grid_cols: int = 6,  # Changed default to 6 for horizontal strip
    frame_size: tuple = (320, 320),  # Larger frames
    add_labels: bool = True,
    layout: str = "horizontal",  # "horizontal", "grid", or "vertical"
) -> Image.Image:
    """
    Create a grid montage of frames showing temporal progression.
    
    Args:
        frames: List of PIL Images in chronological order
        timestamps: List of timestamps for each frame
        grid_cols: Number of columns in grid (used for grid layout)
        frame_size: Size to resize each frame to (width, height)
        add_labels: Whether to add timestamp labels
        layout: Layout type - "horizontal" (1xN), "grid" (MxN), or "vertical" (Nx1)
    
    Returns:
        Single PIL Image containing frame montage
    """
    n_frames = len(frames)
    
    # Determine grid dimensions based on layout
    if layout == "horizontal":
        grid_cols = n_frames
        grid_rows = 1
    elif layout == "vertical":
        grid_cols = 1
        grid_rows = n_frames
    else:  # grid layout
        grid_rows = (n_frames + grid_cols - 1) // grid_cols
    
    # Resize frames
    resized_frames = [frame.resize(frame_size, Image.Resampling.LANCZOS) for frame in frames]
    
    # Create montage canvas
    label_height = 40 if add_labels else 0
    montage_width = frame_size[0] * grid_cols
    montage_height = (frame_size[1] + label_height) * grid_rows
    montage = Image.new('RGB', (montage_width, montage_height), color=(255, 255, 255))
    
    # Place frames in grid
    for idx, (frame, timestamp) in enumerate(zip(resized_frames, timestamps)):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * frame_size[0]
        y = row * (frame_size[1] + label_height)
        
        montage.paste(frame, (x, y))
        
        # Add timestamp label
        if add_labels:
            draw = ImageDraw.Draw(montage)
            label = f"t={timestamp:.1f}s"
            # Use default font (PIL's built-in bitmap font)
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (frame_size[0] - text_width) // 2
            text_y = y + frame_size[1] + 5
            draw.text((text_x, text_y), label, fill=(0, 0, 0))
    
    return montage


def extract_frame_sequence(
    video_path: str,
    t_current: float,
    fps: float,
    n_frames: int = None,
    sample_rate: float = 1.0,
    add_color_markers: bool = False,
    scene_id: str = None,
) -> tuple:
    """
    Extract sequence of frames from t=0 to t=current.
    
    Args:
        video_path: Path to video file
        t_current: Current timestamp in seconds
        fps: Video FPS
        n_frames: Number of frames to extract (if None, use sample_rate)
        sample_rate: Sample every N seconds (default 1.0s)
        add_color_markers: Whether to add colored bounding boxes to mark goals
        scene_id: Scene identifier for block detection (e.g., "block_scene")
    
    Returns:
        (frames, timestamps) tuple
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    if n_frames is not None:
        # Extract exactly n_frames evenly spaced from 0 to t_current
        timestamps = np.linspace(0, t_current, n_frames).tolist()
    else:
        # Sample at regular intervals
        timestamps = np.arange(0, t_current + sample_rate, sample_rate)
        timestamps = timestamps[timestamps <= t_current].tolist()
    
    frames = []
    for t in timestamps:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # Add color markers if requested
            if add_color_markers and scene_id == "block_scene":
                from .block_detection import detect_blocks, add_color_markers as add_markers
                block_positions = detect_blocks(frame, scene_type=scene_id)
                frame = add_markers(frame, block_positions)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            frames.append(pil_frame)
        else:
            # If frame not available, skip
            continue
    
    cap.release()
    
    return frames, timestamps[:len(frames)]
