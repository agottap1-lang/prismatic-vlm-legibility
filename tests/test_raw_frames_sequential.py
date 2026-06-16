"""
Test RAW frames + sequential reasoning approach.

This is the FINAL TEST to determine if VLM can infer legibility from
unmodified frames using explicit reasoning prompts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legibility_eval.sequential_evaluator import run_single_video_test


def test_raw_frames_approach():
    """
    Test all 3 strategies on le_r_block video.
    """
    
    video_path = "../videos/le r block.mp4"
    checkpoints = [3.0, 5.0, 7.0]
    ground_truth = "B"  # Right block
    
    goal_descriptions = {
        "A": "the left block (appears on the left side of the scene)",
        "B": "the right block (appears on the right side of the scene)"
    }
    
    print("="*70)
    print("RAW FRAMES + SEQUENTIAL REASONING TEST")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Ground truth: Goal {ground_truth} (right block)")
    print(f"Checkpoints: {checkpoints}")
    print("\nTesting 3 prompting strategies:")
    print("  1. Chain-of-thought (full reasoning in single pass)")
    print("  2. Differential (frame-pair analysis + aggregation)")
    print("  3. Simple (direct spatial reasoning)")
    print("="*70)
    
    results_summary = {}
    
    # Test each strategy
    for strategy in ["simple", "chain_of_thought", "differential"]:
        print(f"\n\n{'='*70}")
        print(f"STRATEGY: {strategy.upper().replace('_', ' ')}")
        print(f"{'='*70}")
        
        try:
            _, accuracy = run_single_video_test(
                video_path=video_path,
                checkpoints=checkpoints,
                ground_truth_goal=ground_truth,
                goal_descriptions=goal_descriptions,
                strategy=strategy
            )
            
            results_summary[strategy] = accuracy
            
        except Exception as e:
            print(f"✗ Error with {strategy}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[strategy] = 0.0
    
    # Final comparison
    print("\n\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    print("\n| Approach | Accuracy | vs Baseline |")
    print("|----------|----------|-------------|")
    
    baseline_acc = 0.50  # Historical baseline
    color_marker_acc = 0.50
    trajectory_acc = 0.375
    
    print(f"| Baseline (no markers) | {baseline_acc:.1%} | - |")
    print(f"| Color markers | {color_marker_acc:.1%} | {(color_marker_acc-baseline_acc):.1%} |")
    print(f"| Trajectory overlay | {trajectory_acc:.1%} | {(trajectory_acc-baseline_acc):.1%} |")
    print(f"| **Simple spatial** | **{results_summary.get('simple', 0):.1%}** | **{(results_summary.get('simple', 0)-baseline_acc):.1%}** |")
    print(f"| **Chain-of-thought** | **{results_summary.get('chain_of_thought', 0):.1%}** | **{(results_summary.get('chain_of_thought', 0)-baseline_acc):.1%}** |")
    print(f"| **Differential** | **{results_summary.get('differential', 0):.1%}** | **{(results_summary.get('differential', 0)-baseline_acc):.1%}** |")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    best_strategy = max(results_summary.items(), key=lambda x: x[1])
    best_acc = best_strategy[1]
    
    if best_acc >= 0.75:
        print("\n✓✓✓ SUCCESS! RAW FRAMES + SEQUENTIAL REASONING WORKS!")
        print(f"    Best strategy: {best_strategy[0]} ({best_acc:.1%} accuracy)")
        print("\n    Next steps:")
        print("    1. Scale to all 8 videos")
        print("    2. Test at more checkpoints (10%, 20%, 40%, 60%, 80%)")
        print("    3. Analyze legibility emergence time")
        print("    4. Deploy to production")
        
    elif best_acc > 0.55:
        print("\n✓ PARTIAL SUCCESS - Shows promise but needs refinement")
        print(f"   Best strategy: {best_strategy[0]} ({best_acc:.1%} accuracy)")
        print("\n   Potential improvements:")
        print("   - Refine prompts (add more explicit spatial cues)")
        print("   - Increase frame count (12 instead of 6)")
        print("   - Try different layouts (horizontal vs grid)")
        print("   - Add goal region descriptions in prompt")
        
    else:
        print("\n✗ RAW FRAMES APPROACH DID NOT SOLVE THE PROBLEM")
        print(f"   Best accuracy: {best_acc:.1%} (not better than baseline)")
        print("\n   ROOT CAUSE: Prismatic VLM cannot perform spatial-temporal reasoning")
        print("   even with explicit prompting. The model lacks the architectural")
        print("   components needed for motion understanding.")
        print("\n   RECOMMENDED SOLUTIONS:")
        print("   1. Switch to video-native model (GPT-4V, Video-LLaMA, Gemini)")
        print("   2. Hybrid CV+VLM (extract trajectory with CV, ask VLM to interpret)")
        print("   3. Fine-tune Prismatic on legibility prediction task")
    
    print("="*70)


if __name__ == "__main__":
    test_raw_frames_approach()
