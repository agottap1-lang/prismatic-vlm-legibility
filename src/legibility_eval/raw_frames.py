"""
Raw frame extraction for sequential reasoning approach.

Principle: Extract UNMODIFIED frames from videos. No overlays, no annotations,
no visual modifications. The VLM must understand motion purely from observing
consecutive raw frames.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
from pathlib import Path


def extract_raw_frames(
    video_path: str,
    checkpoint_time: float,
    fps: float,
    n_frames: int = 6,
    start_time: float = 0.0,
    target_size: Tuple[int, int] = (512, 512)
) -> List[Image.Image]:
    """
    Extract N evenly-spaced RAW frames from video.
    
    NO processing, NO annotations - just resize for VLM input.
    
    Args:
        video_path: Path to video file
        checkpoint_time: End time for frame extraction
        fps: Video FPS
        n_frames: Number of frames to extract
        start_time: Start time (default 0.0)
        target_size: Resize frames to this size (w, h)
    
    Returns:
        List of PIL Images (raw frames, resized)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Sample times evenly from start_time to checkpoint_time
    times = np.linspace(start_time, checkpoint_time, n_frames)
    
    frames = []
    for t in times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame at t={t:.2f}s")
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame_resized = cv2.resize(frame_rgb, target_size)
        
        # Convert to PIL Image
        img = Image.fromarray(frame_resized)
        frames.append(img)
    
    cap.release()
    
    return frames


def create_frame_grid(
    frames: List[Image.Image],
    grid_cols: int = 3,
    add_frame_numbers: bool = True,
    frame_number_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Arrange frames in a grid layout.
    
    Optional: Add small frame numbers (1, 2, 3...) in corner for reference.
    These numbers don't explain anything - just help VLM track frame order.
    
    Args:
        frames: List of PIL Images
        grid_cols: Number of columns
        add_frame_numbers: Whether to add small numbers (default True)
        frame_number_color: RGB color for frame numbers
    
    Returns:
        Grid montage as PIL Image
    """
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames provided")
    
    # Calculate grid dimensions
    grid_rows = (n_frames + grid_cols - 1) // grid_cols
    
    frame_width, frame_height = frames[0].size
    grid_width = frame_width * grid_cols
    grid_height = frame_height * grid_rows
    
    # Create white canvas
    grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
    
    for i, frame in enumerate(frames):
        row = i // grid_cols
        col = i % grid_cols
        
        y_offset = row * frame_height
        x_offset = col * frame_width
        
        # Place frame
        frame_array = np.array(frame)
        grid[y_offset:y_offset+frame_height, x_offset:x_offset+frame_width] = frame_array
        
        # Add small frame number in top-left corner
        if add_frame_numbers:
            cv2.putText(
                grid,
                str(i + 1),
                (x_offset + 10, y_offset + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                frame_number_color,
                2,
                cv2.LINE_AA
            )
    
    return Image.fromarray(grid)


def create_horizontal_strip(
    frames: List[Image.Image],
    spacing: int = 10,
    add_frame_numbers: bool = True
) -> Image.Image:
    """
    Arrange frames in a horizontal strip with small spacing.
    
    Args:
        frames: List of PIL Images
        spacing: Pixels between frames
        add_frame_numbers: Whether to add small numbers
    
    Returns:
        Horizontal strip as PIL Image
    """
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames provided")
    
    frame_width, frame_height = frames[0].size
    total_width = frame_width * n_frames + spacing * (n_frames - 1)
    
    # Create white canvas
    strip = np.ones((frame_height, total_width, 3), dtype=np.uint8) * 255
    
    for i, frame in enumerate(frames):
        x_offset = i * (frame_width + spacing)
        
        # Place frame
        frame_array = np.array(frame)
        strip[:, x_offset:x_offset+frame_width] = frame_array
        
        # Add small frame number
        if add_frame_numbers:
            cv2.putText(
                strip,
                str(i + 1),
                (x_offset + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
    
    return Image.fromarray(strip)


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        video_path: Path to video
    
    Returns:
        Dict with fps, duration, frame_count, width, height
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# Test the module
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    video_path = "videos/le r block.mp4"
    
    # Get video info
    info = get_video_info(video_path)
    print(f"Video info:")
    print(f"  Duration: {info['duration']:.2f}s")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  Resolution: {info['width']}x{info['height']}")
    
    # Extract raw frames at t=5s
    print(f"\nExtracting 6 raw frames from 0s to 5s...")
    frames = extract_raw_frames(
        video_path=video_path,
        checkpoint_time=5.0,
        fps=info['fps'],
        n_frames=6,
        target_size=(384, 384)
    )
    print(f"Extracted {len(frames)} frames")
    
    # Create grid montage
    grid = create_frame_grid(frames, grid_cols=3, add_frame_numbers=True)
    grid.save("outputs/images/raw_frames_grid.png")
    print("✓ Saved grid: outputs/images/raw_frames_grid.png")
    
    # Create horizontal strip
    strip = create_horizontal_strip(frames, spacing=10, add_frame_numbers=True)
    strip.save("outputs/images/raw_frames_horizontal.png")
    print("✓ Saved strip: outputs/images/raw_frames_horizontal.png")
