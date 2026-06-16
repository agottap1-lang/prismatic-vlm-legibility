#!/usr/bin/env python3
"""
Test color-based spatial markers on block videos.

This tests Solution 1 from SPATIAL_GROUNDING_ISSUE.md - using colored bounding boxes
(RED for left block, BLUE for right block) to provide explicit visual reference
instead of relying on spatial language like "left" and "right".
"""

import json
from pathlib import Path
from src.legibility_eval.client import PrismaticTemporalClient

def test_color_markers():
    """Test color marker approach on block videos."""
    
    # Initialize client
    print("Loading Prismatic VLM...")
    client = PrismaticTemporalClient(
        model_path="prism-dinosiglip+7b",
        device=None,  # Auto-detect
    )
    print("✓ Model loaded\n")
    
    # Test configuration
    test_videos = [
        {
            "video_id": "le_r_block",
            "video_path": "videos/le r block.mp4",
            "goal_A_desc": "Pick up the block marked with RED box",
            "goal_B_desc": "Pick up the block marked with BLUE box",
            "ground_truth": "B",  # Robot moves toward right (BLUE) block
        },
        {
            "video_id": "le_l_block",
            "video_path": "videos/le l block.mp4",
            "goal_A_desc": "Pick up the block marked with RED box",
            "goal_B_desc": "Pick up the block marked with BLUE box",
            "ground_truth": "A",  # Robot moves toward left (RED) block
        },
    ]
    
    # Test at multiple timestamps
    timestamps = [3, 5, 7, 9]
    
    results = []
    
    for video_info in test_videos:
        video_id = video_info["video_id"]
        print(f"Testing {video_id}...")
        print(f"Ground truth: Goal {video_info['ground_truth']}")
        print("-" * 60)
        
        for t in timestamps:
            print(f"  t={t}s: ", end="", flush=True)
            
            try:
                # Test WITH color markers
                result_with_markers = client.evaluate_video_at_time(
                    video_path=video_info["video_path"],
                    t_current=t,
                    goal_A_desc=video_info["goal_A_desc"],
                    goal_B_desc=video_info["goal_B_desc"],
                    video_id=video_id,
                    n_frames=6,
                    fps=30.0,
                    mode="prefix_frames",
                    layout="horizontal",
                    use_color_markers=True,
                    scene_id="block_scene",
                )
                
                # Check accuracy
                predicted = result_with_markers["choice"]
                correct = (predicted == video_info["ground_truth"])
                
                print(f"pA={result_with_markers['pA']:.2f}, pB={result_with_markers['pB']:.2f}, "
                      f"choice={predicted}, {'✓' if correct else '✗'}")
                print(f"           cue: \"{result_with_markers['cue']}\"")
                
                results.append({
                    "video_id": video_id,
                    "timestamp": t,
                    "ground_truth": video_info["ground_truth"],
                    "predicted": predicted,
                    "correct": correct,
                    "pA": result_with_markers["pA"],
                    "pB": result_with_markers["pB"],
                    "cue": result_with_markers["cue"],
                    "legible": result_with_markers["legible"],
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({
                    "video_id": video_id,
                    "timestamp": t,
                    "error": str(e),
                })
        
        print()
    
    # Calculate accuracy
    correct_count = sum(1 for r in results if r.get("correct", False))
    total_count = len([r for r in results if "correct" in r])
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    
    print("=" * 60)
    print(f"Overall Accuracy with Color Markers: {accuracy:.1%} ({correct_count}/{total_count})")
    print("=" * 60)
    
    # Save results
    output_path = Path("results/color_marker_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "test_config": {
                "method": "color_markers",
                "layout": "horizontal",
                "n_frames": 6,
                "timestamps": timestamps,
            },
            "results": results,
            "summary": {
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total_count,
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    
    if accuracy > 0.75:
        print("✓ Color markers SOLVED the spatial grounding problem!")
        print("  The VLM can successfully use color references (RED/BLUE)")
        print("  to identify which region the robot is approaching.")
    elif accuracy > 0.5:
        print("⚠ Color markers PARTIALLY helped but not sufficient.")
        print("  Consider implementing Solution 3 (relative motion description)")
        print("  or Solution 5 (try different VLM like GPT-4V).")
    else:
        print("✗ Color markers did NOT solve the problem.")
        print("  The issue may be deeper than spatial grounding.")
        print("  Recommend trying Solution 5 (different VLM architecture).")
    
    return results, accuracy

if __name__ == "__main__":
    test_color_markers()
