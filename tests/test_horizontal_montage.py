"""
Test horizontal montage approach with red dots and arrows.

This script evaluates the horizontal montage approach on the video dataset.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.legibility_eval.horizontal_montage import (
    create_horizontal_montage_checkpoint,
    build_horizontal_montage_prompt
)
from src.legibility_eval.task_agnostic_prompt import (
    parse_trajectory_response,
    compute_intent_metrics
)
from prismatic import load


def test_single_video():
    """Test horizontal montage on a single video."""
    
    # Load model
    print("Loading Prismatic VLM...")
    vlm = load("prism-dinosiglip+7b")
    
    # Test video
    video_path = "videos/le r block.mp4"
    video_name = "le_r_block"
    fps = 30.0
    
    # Ground truth: right block is the goal
    ground_truth_goal = "right"
    
    # Test at multiple checkpoints
    checkpoints = [3.0, 5.0, 7.0]
    
    results = []
    
    for checkpoint_time in checkpoints:
        print(f"\n{'='*60}")
        print(f"Testing {video_name} at t={checkpoint_time}s")
        print(f"{'='*60}")
        
        # Generate horizontal montage
        img = create_horizontal_montage_checkpoint(
            video_path, 
            checkpoint_time, 
            fps, 
            num_frames=6
        )
        
        # Save image
        output_path = f"../outputs/images/test_{video_name}_t{int(checkpoint_time)}.png"
        img.save(output_path)
        print(f"✓ Saved montage: {output_path}")
        
        # Get prompt
        prompt = build_horizontal_montage_prompt(n_goal_regions=2)
        prompt_with_context = f"{prompt}\n\nNote: In this scene, Region 1 is the LEFT side and Region 2 is the RIGHT side."
        
        # Query VLM
        print("\nQuerying VLM...")
        response = vlm.generate_text(
            image=img,
            prompt_text=prompt_with_context,
            temperature=0.2,
            max_new_tokens=512
        )
        
        print(f"\nVLM Response:\n{response}")
        
        # Parse response
        parsed = parse_trajectory_response(response)
        
        if parsed:
            metrics = compute_intent_metrics(
                parsed['score_region_1'],
                parsed['score_region_2']
            )
            
            # Determine if correct
            predicted_goal = "left" if metrics['choice'] == 'A' else "right" if metrics['choice'] == 'B' else "uncertain"
            correct = (predicted_goal == ground_truth_goal)
            
            result = {
                'video': video_name,
                'checkpoint': checkpoint_time,
                'ground_truth': ground_truth_goal,
                'predicted': predicted_goal,
                'correct': correct,
                'scores': {
                    'region_1_left': parsed['score_region_1'],
                    'region_2_right': parsed['score_region_2']
                },
                'confidence': metrics['confidence'],
                'reasoning': parsed.get('reasoning', '')
            }
            
            results.append(result)
            
            print(f"\n{'='*60}")
            print(f"RESULT:")
            print(f"  Ground Truth: {ground_truth_goal.upper()}")
            print(f"  Predicted: {predicted_goal.upper()}")
            print(f"  Correct: {'✓' if correct else '✗'}")
            print(f"  Scores: Left={parsed['score_region_1']}, Right={parsed['score_region_2']}")
            print(f"  Confidence: {metrics['confidence']:.1f}%")
            print(f"{'='*60}")
    
    # Save results
    output_file = "../results/horizontal_montage_test.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(results) * 100
    
    print(f"\nSUMMARY:")
    print(f"  Total tests: {len(results)}")
    print(f"  Correct: {correct_count}")
    print(f"  Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    test_single_video()
