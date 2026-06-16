#!/usr/bin/env python3
"""Detect block positions in frames using simple color/contour detection."""

import cv2
import numpy as np
from pathlib import Path

def detect_blocks(frame, scene_type="block_scene"):
    """Detect block positions in frame.
    
    Returns dict with 'left_block' and 'right_block' bounding boxes as (x1, y1, x2, y2).
    """
    if scene_type != "block_scene":
        return {}
    
    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to find objects
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (blocks should be reasonable size)
    block_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area < 10000:  # Adjust based on actual block sizes
            x, y, w, h = cv2.boundingRect(cnt)
            block_contours.append((x, y, x+w, y+h, area))
    
    if len(block_contours) < 2:
        # Fallback: use fixed positions based on typical block scene layout
        height, width = frame.shape[:2]
        # Left block typically in left third
        left_block = (width//6, height//2, width//3, 2*height//3)
        # Right block typically in right third
        right_block = (2*width//3, height//2, 5*width//6, 2*height//3)
        return {
            'left_block': left_block,
            'right_block': right_block,
            'method': 'fixed'
        }
    
    # Sort by x-coordinate (left to right)
    block_contours = sorted(block_contours, key=lambda b: b[0])
    
    # Take leftmost and rightmost as left and right blocks
    left_block = block_contours[0][:4]
    right_block = block_contours[-1][:4]
    
    return {
        'left_block': left_block,
        'right_block': right_block,
        'method': 'detected'
    }


def add_color_markers(frame, block_positions):
    """Add colored bounding boxes and labels to mark blocks.
    
    Args:
        frame: BGR image
        block_positions: dict with 'left_block' and 'right_block' as (x1, y1, x2, y2)
    
    Returns:
        frame with markers overlaid
    """
    frame = frame.copy()
    
    if 'left_block' in block_positions:
        x1, y1, x2, y2 = block_positions['left_block']
        # RED box for left block
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv2.putText(frame, "RED GOAL", (x1, max(y1-10, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    if 'right_block' in block_positions:
        x1, y1, x2, y2 = block_positions['right_block']
        # BLUE box for right block
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(frame, "BLUE GOAL", (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 3)
    
    return frame


if __name__ == "__main__":
    # Test on a debug frame
    debug_dir = Path("debug_frames_output")
    if not debug_dir.exists():
        print("No debug frames found")
        exit(1)
    
    test_frame_path = debug_dir / "frame_5_t5.00.png"
    if not test_frame_path.exists():
        test_frame_path = list(debug_dir.glob("*.png"))[0]
    
    print(f"Testing on: {test_frame_path}")
    frame = cv2.imread(str(test_frame_path))
    
    # Detect blocks
    blocks = detect_blocks(frame)
    print(f"Block positions (method={blocks.get('method', 'unknown')}):")
    print(f"  Left block: {blocks.get('left_block', 'N/A')}")
    print(f"  Right block: {blocks.get('right_block', 'N/A')}")
    
    # Add markers
    marked_frame = add_color_markers(frame, blocks)
    
    # Save result
    output_path = "test_block_detection.png"
    cv2.imwrite(output_path, marked_frame)
    print(f"\nMarked frame saved to: {output_path}")
