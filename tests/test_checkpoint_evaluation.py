#!/usr/bin/env python3
"""
Test checkpoint-based evaluation with single-image trajectory encoding.

This is the CORRECT approach:
- ONE image per checkpoint with trajectory overlay
- NO frame montages, NO temporal ordering assumptions
- Task-agnostic prompting
- Numerical scores (0-100 per region)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.legibility_eval.client import PrismaticTemporalClient
from src.legibility_eval.checkpoint_evaluator import CheckpointEvaluator

def test_checkpoint_evaluation():
    """Test checkpoint-based evaluation on block videos."""
    
    print("=" * 70)
    print("CHECKPOINT-BASED LEGIBILITY EVALUATION")
    print("Single-image trajectory encoding + Task-agnostic prompting")
    print("=" * 70)
    
    # Initialize VLM client
    print("\nLoading Prismatic VLM...")
    vlm_client = PrismaticTemporalClient(
        model_path="prism-dinosiglip+7b",
        device=None
    )
    print("✓ Model loaded")
    
    # Initialize checkpoint evaluator
    checkpoints = [0.2, 0.4, 0.6, 0.8]  # 20%, 40%, 60%, 80%
    evaluator = CheckpointEvaluator(vlm_client, checkpoints=checkpoints)
    
    # Test videos
    test_videos = [
        {
            "video_id": "le_r_block",
            "video_path": "videos/le r block.mp4",
            "goal_A": "pick the left block",
            "goal_B": "pick the right block",
            "goal_gt": "B",
            "notes": "Legible trajectory toward right block"
        },
        {
            "video_id": "le_l_block",
            "video_path": "videos/le l block.mp4",
            "goal_A": "pick the left block",
            "goal_B": "pick the right block",
            "goal_gt": "A",
            "notes": "Legible trajectory toward left block"
        },
    ]
    
    all_results = []
    
    for video_info in test_videos:
        print("\n" + "=" * 70)
        
        result = evaluator.evaluate_video(
            video_path=video_info["video_path"],
            video_id=video_info["video_id"],
            goal_A_desc=video_info["goal_A"],
            goal_B_desc=video_info["goal_B"],
            ground_truth=video_info["goal_gt"],
            fps=30.0
        )
        
        all_results.append(result)
        
        # Print statistics
        stats = result["statistics"]
        print(f"\n  Overall accuracy: {stats['accuracy']:.1%} ({stats['n_correct']}/{stats['n_total']})")
        if stats["first_correct_checkpoint"]:
            print(f"  First correct at: {int(stats['first_correct_checkpoint']*100)}%")
        if stats["first_legible_checkpoint"]:
            print(f"  First legible at: {int(stats['first_legible_checkpoint']*100)}%")
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    # Per checkpoint accuracy
    for cp in checkpoints:
        checkpoint_results = []
        for video_result in all_results:
            cp_result = next((r for r in video_result["checkpoints"] if r["checkpoint_pct"] == cp), None)
            if cp_result and "correct" in cp_result and cp_result["correct"] is not None:
                checkpoint_results.append(cp_result["correct"])
        
        if checkpoint_results:
            accuracy = sum(checkpoint_results) / len(checkpoint_results)
            n_correct = sum(checkpoint_results)
            n_total = len(checkpoint_results)
            print(f"  Checkpoint {int(cp*100):3d}%: {accuracy:5.1%} ({n_correct}/{n_total})")
    
    # Overall
    all_correct = []
    for video_result in all_results:
        for cp_result in video_result["checkpoints"]:
            if "correct" in cp_result and cp_result["correct"] is not None:
                all_correct.append(cp_result["correct"])
    
    if all_correct:
        overall_accuracy = sum(all_correct) / len(all_correct)
        print(f"\n  Overall accuracy: {overall_accuracy:.1%} ({sum(all_correct)}/{len(all_correct)})")
    
    # Save results
    import json
    from pathlib import Path
    
    output_path = Path("results/checkpoint_test.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            "test_videos": test_videos,
            "results": all_results,
            "checkpoints": checkpoints
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if all_correct:
        if overall_accuracy >= 0.8:
            print("✓✓✓ SUCCESS! Checkpoint-based approach works!")
            print("    The single-image trajectory encoding allows the VLM")
            print("    to reason about motion as spatial geometry.")
        elif overall_accuracy >= 0.6:
            print("✓✓ PROMISING! Significant improvement over baseline (50%).")
            print("   Fine-tuning gripper detection may improve further.")
        elif overall_accuracy > 0.5:
            print("✓ IMPROVEMENT! Better than baseline (50%).")
            print("  May need better trajectory visualization or prompting.")
        else:
            print("✗ Still struggling. Consider:")
            print("  - GPT-4V for comparison")
            print("  - Hybrid approach with classical CV")
            print("  - Different trajectory visualization style")
    
    return all_results


if __name__ == "__main__":
    test_checkpoint_evaluation()
