#!/usr/bin/env python3
"""
Test trajectory annotation approach on block videos.

This tests Tier 1 solution from DEEP_ANALYSIS_AND_SOLUTION.md:
- Explicitly visualizes robot gripper trajectory with arrows and motion trails
- Converts temporal reasoning problem → spatial reasoning problem
- Leverages DINOv2's strength in detecting visual cues (arrows, lines, colors)
"""

import json
from pathlib import Path
from src.legibility_eval.client import PrismaticTemporalClient

def test_trajectory_approach():
    """Test trajectory annotation approach on block videos."""
    
    # Initialize client
    print("Loading Prismatic VLM...")
    client = PrismaticTemporalClient(
        model_path="prism-dinosiglip+7b",
        device=None,  # Auto-detect
    )
    print("✓ Model loaded\n")
    
    # Test configuration - now with TRAJECTORY VISUALIZATION
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
                # Test WITH trajectory annotation
                result = client.evaluate_video_at_time(
                    video_path=video_info["video_path"],
                    t_current=t,
                    goal_A_desc=video_info["goal_A_desc"],
                    goal_B_desc=video_info["goal_B_desc"],
                    video_id=video_id,
                    n_frames=6,
                    fps=30.0,
                    mode="prefix_frames",
                    layout="horizontal",
                    use_trajectory=True,  # KEY: Enable trajectory annotation
                    scene_id="block_scene",
                )
                
                # Check accuracy
                predicted = result["choice"]
                correct = (predicted == video_info["ground_truth"])
                
                print(f"pA={result['pA']:.2f}, pB={result['pB']:.2f}, "
                      f"choice={predicted}, {'✓' if correct else '✗'}")
                print(f"           cue: \"{result['cue']}\"")
                
                results.append({
                    "video_id": video_id,
                    "timestamp": t,
                    "ground_truth": video_info["ground_truth"],
                    "predicted": predicted,
                    "correct": correct,
                    "pA": result["pA"],
                    "pB": result["pB"],
                    "cue": result["cue"],
                    "legible": result["legible"],
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
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
    print(f"Overall Accuracy with Trajectory Annotation: {accuracy:.1%} ({correct_count}/{total_count})")
    print("=" * 60)
    
    # Save results
    output_path = Path("results/trajectory_annotation_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "test_config": {
                "method": "trajectory_annotation",
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
    
    if accuracy >= 0.9:
        print("✓✓✓ TRAJECTORY ANNOTATION SOLVED THE PROBLEM!")
        print("    The VLM successfully uses explicit visual trajectory cues.")
        print("    Accuracy ≥90% demonstrates the approach works.")
        print("\n    Next steps:")
        print("    1. Test on all 8 videos across all timestamps")
        print("    2. Measure legibility emergence time (when accuracy reaches 80%)")
        print("    3. Deploy to full evaluation suite")
    elif accuracy >= 0.75:
        print("✓✓ TRAJECTORY ANNOTATION MOSTLY WORKS!")
        print("   Accuracy ≥75% is promising. The approach helps significantly.")
        print("\n   Potential improvements:")
        print("   - Fine-tune gripper detection for edge cases")
        print("   - Adjust arrow size/color for better visibility")
        print("   - Add Tier 2 (differential analysis) for ambiguous cases")
    elif accuracy > 0.5:
        print("✓ TRAJECTORY ANNOTATION HELPS SOMEWHAT")
        print("  Accuracy >50% shows improvement over baseline (50%).")
        print("\n  Recommended actions:")
        print("  - Analyze failure cases to identify patterns")
        print("  - Implement multi-pass ensemble (Tier 3)")
        print("  - Consider hybrid approach with motion analysis")
    else:
        print("✗ TRAJECTORY ANNOTATION INSUFFICIENT")
        print("  Accuracy ≤50% suggests deeper architectural issues.")
        print("\n  Alternative approaches:")
        print("  - Test with Video-LLaMA or GPT-4V (video-native models)")
        print("  - Implement hybrid pipeline with specialized motion models")
        print("  - Consider fine-tuning Prismatic on robot trajectory data")
    
    # Comparison with baseline
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON:")
    print("=" * 60)
    print(f"Color markers only:       50.0% accuracy")
    print(f"Trajectory annotation:    {accuracy:.1%} accuracy")
    
    if accuracy > 0.5:
        improvement = (accuracy - 0.5) / 0.5 * 100
        print(f"\n✓ Improvement: +{improvement:.1f}% relative to baseline")
    
    return results, accuracy

if __name__ == "__main__":
    test_trajectory_approach()
