"""
checkpoint_evaluator.py

Checkpoint-based legibility evaluation using single-image trajectory encoding.

Key principles:
1. Each checkpoint = ONE image with trajectory overlay
2. No frame montages, no temporal ordering assumptions
3. Task-agnostic prompting
4. Numerical scores for quantitative analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image

from .trajectory_encoding import create_checkpoint_image, get_checkpoint_times
from .task_agnostic_prompt import (
    build_trajectory_intent_prompt,
    parse_trajectory_response,
    compute_intent_metrics,
    map_choice_to_ground_truth
)


class CheckpointEvaluator:
    """
    Evaluates legibility at multiple checkpoints using trajectory-encoded images.
    """
    
    def __init__(self, vlm_client, checkpoints: List[float] = [0.1, 0.2, 0.4, 0.6, 0.8]):
        """
        Initialize evaluator.
        
        Args:
            vlm_client: Prismatic VLM client instance
            checkpoints: List of progress percentages (0.0 to 1.0)
        """
        self.vlm = vlm_client
        self.checkpoints = checkpoints
        self.prompt = build_trajectory_intent_prompt(n_goal_regions=2)
    
    def evaluate_video(
        self,
        video_path: str,
        video_id: str,
        goal_A_desc: str,
        goal_B_desc: str,
        ground_truth: str,
        fps: Optional[float] = None
    ) -> Dict:
        """
        Evaluate legibility at multiple checkpoints.
        
        Args:
            video_path: Path to video file
            video_id: Video identifier
            goal_A_desc: Description of Goal A (for metadata only)
            goal_B_desc: Description of Goal B (for metadata only)
            ground_truth: Ground truth goal ("A" or "B")
            fps: Video FPS (auto-detected if None)
        
        Returns:
            Dictionary with checkpoint results
        """
        import cv2
        
        # Get video info
        if fps is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
        else:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
        
        # Get checkpoint times
        checkpoint_times = get_checkpoint_times(duration, self.checkpoints)
        
        results = {
            "video_id": video_id,
            "video_path": video_path,
            "duration": duration,
            "ground_truth": ground_truth,
            "goal_A_desc": goal_A_desc,
            "goal_B_desc": goal_B_desc,
            "checkpoints": []
        }
        
        # Determine spatial mapping (which goal is which region)
        # For block tasks: left block = region 1, right block = region 2
        # Goal A = left block, so region 1 = Goal A
        goal_A_is_region_1 = self._infer_spatial_mapping(video_id, goal_A_desc, goal_B_desc)
        
        print(f"\nEvaluating {video_id} (duration={duration:.2f}s)...")
        print(f"Ground truth: Goal {ground_truth}")
        print(f"Spatial mapping: Goal A -> Region {'1' if goal_A_is_region_1 else '2'}")
        
        # Evaluate each checkpoint
        for i, (checkpoint_pct, checkpoint_time) in enumerate(zip(self.checkpoints, checkpoint_times)):
            print(f"  Checkpoint {int(checkpoint_pct*100)}% (t={checkpoint_time:.2f}s)... ", end="", flush=True)
            
            try:
                # 1. Create trajectory-encoded image
                trajectory_image = create_checkpoint_image(
                    video_path=video_path,
                    checkpoint_time=checkpoint_time,
                    fps=fps,
                    total_duration=duration
                )
                
                # 2. Query VLM with task-agnostic prompt
                response = self.vlm.generate(
                    trajectory_image,
                    self.prompt,
                    do_sample=False,
                    max_new_tokens=200
                )
                
                # 3. Parse response
                parsed = parse_trajectory_response(response)
                
                # 4. Compute metrics
                metrics = compute_intent_metrics(
                    parsed["score_region_1"],
                    parsed["score_region_2"],
                    threshold=60
                )
                
                # 5. Map to ground truth labels
                predicted_choice = map_choice_to_ground_truth(
                    metrics["choice"],
                    goal_A_is_region_1
                )
                
                # 6. Check accuracy
                correct = (predicted_choice == ground_truth) if predicted_choice != "uncertain" else None
                
                checkpoint_result = {
                    "checkpoint_pct": checkpoint_pct,
                    "checkpoint_time": checkpoint_time,
                    "score_region_1": parsed["score_region_1"],
                    "score_region_2": parsed["score_region_2"],
                    "trajectory_direction": parsed["trajectory_direction"],
                    "reasoning": parsed["reasoning"],
                    "predicted_choice": predicted_choice,
                    "correct": correct,
                    "legible": metrics["legible"],
                    "confidence": metrics["confidence"],
                    "probability_A": metrics["probability_A"],
                    "probability_B": metrics["probability_B"],
                    "vlm_raw_response": parsed["raw"]
                }
                
                results["checkpoints"].append(checkpoint_result)
                
                # Print result
                status = "✓" if correct else ("✗" if correct is not None else "?")
                print(f"scores=[{parsed['score_region_1']}, {parsed['score_region_2']}], "
                      f"predicted={predicted_choice}, {status}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                results["checkpoints"].append({
                    "checkpoint_pct": checkpoint_pct,
                    "checkpoint_time": checkpoint_time,
                    "error": str(e)
                })
        
        # Compute overall statistics
        results["statistics"] = self._compute_statistics(results["checkpoints"], ground_truth)
        
        return results
    
    def _infer_spatial_mapping(self, video_id: str, goal_A_desc: str, goal_B_desc: str) -> bool:
        """
        Infer whether Goal A corresponds to spatial region 1 (left/top).
        
        Args:
            video_id: Video identifier
            goal_A_desc: Goal A description
            goal_B_desc: Goal B description
        
        Returns:
            True if Goal A is region 1 (left/top), False otherwise
        """
        # For block videos: left block = region 1
        if "left" in goal_A_desc.lower() or "l_block" in video_id:
            return True
        elif "right" in goal_A_desc.lower() or "r_block" in video_id:
            return False
        
        # Default assumption: Goal A = region 1
        return True
    
    def _compute_statistics(self, checkpoint_results: List[Dict], ground_truth: str) -> Dict:
        """Compute aggregate statistics across checkpoints."""
        valid_results = [r for r in checkpoint_results if "error" not in r and r.get("correct") is not None]
        
        if not valid_results:
            return {
                "accuracy": 0.0,
                "n_correct": 0,
                "n_total": 0,
                "first_legible_checkpoint": None,
                "first_correct_checkpoint": None
            }
        
        n_correct = sum(1 for r in valid_results if r["correct"])
        n_total = len(valid_results)
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        
        # Find first legible checkpoint
        legible_checkpoints = [r for r in valid_results if r["legible"] == "legible_now"]
        first_legible = legible_checkpoints[0]["checkpoint_pct"] if legible_checkpoints else None
        
        # Find first correct checkpoint
        correct_checkpoints = [r for r in valid_results if r["correct"]]
        first_correct = correct_checkpoints[0]["checkpoint_pct"] if correct_checkpoints else None
        
        return {
            "accuracy": accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "first_legible_checkpoint": first_legible,
            "first_correct_checkpoint": first_correct,
            "legibility_emergence_time": first_legible * 100 if first_legible else None  # as percentage
        }


def evaluate_dataset(
    vlm_client,
    manifest_path: str,
    output_dir: str = "results/checkpoint_evaluation",
    checkpoints: List[float] = [0.1, 0.2, 0.4, 0.6, 0.8]
) -> Dict:
    """
    Evaluate entire dataset using checkpoint-based approach.
    
    Args:
        vlm_client: Prismatic VLM client
        manifest_path: Path to manifest.jsonl
        output_dir: Output directory for results
        checkpoints: List of checkpoint percentages
    
    Returns:
        Aggregate results dictionary
    """
    import json
    from pathlib import Path
    
    # Load manifest
    videos = []
    with open(manifest_path, 'r') as f:
        for line in f:
            videos.append(json.loads(line))
    
    print(f"Loaded {len(videos)} videos from {manifest_path}")
    
    # Initialize evaluator
    evaluator = CheckpointEvaluator(vlm_client, checkpoints)
    
    # Evaluate each video
    all_results = []
    
    for video_info in videos:
        result = evaluator.evaluate_video(
            video_path=video_info["video_path"],
            video_id=video_info["video_id"],
            goal_A_desc=video_info["goal_A"],
            goal_B_desc=video_info["goal_B"],
            ground_truth=video_info["goal_gt"],
            fps=30.0  # Assuming 30 FPS
        )
        all_results.append(result)
    
    # Compute aggregate statistics
    aggregate = {
        "n_videos": len(all_results),
        "checkpoints": checkpoints,
        "per_video_results": all_results,
        "aggregate_accuracy_per_checkpoint": {},
        "overall_accuracy": 0.0
    }
    
    # Compute accuracy per checkpoint
    for cp in checkpoints:
        checkpoint_results = []
        for video_result in all_results:
            cp_result = next((r for r in video_result["checkpoints"] if r["checkpoint_pct"] == cp), None)
            if cp_result and "correct" in cp_result and cp_result["correct"] is not None:
                checkpoint_results.append(cp_result["correct"])
        
        if checkpoint_results:
            accuracy = sum(checkpoint_results) / len(checkpoint_results)
            aggregate["aggregate_accuracy_per_checkpoint"][f"{int(cp*100)}%"] = {
                "accuracy": accuracy,
                "n_correct": sum(checkpoint_results),
                "n_total": len(checkpoint_results)
            }
    
    # Overall accuracy (across all checkpoints)
    all_correct = []
    for video_result in all_results:
        for cp_result in video_result["checkpoints"]:
            if "correct" in cp_result and cp_result["correct"] is not None:
                all_correct.append(cp_result["correct"])
    
    if all_correct:
        aggregate["overall_accuracy"] = sum(all_correct) / len(all_correct)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "checkpoint_evaluation_results.json", 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path / 'checkpoint_evaluation_results.json'}")
    
    return aggregate


if __name__ == "__main__":
    print("Checkpoint evaluator module ready.")
    print("\nUsage:")
    print("  from src.legibility_eval.checkpoint_evaluator import CheckpointEvaluator")
    print("  evaluator = CheckpointEvaluator(vlm_client)")
    print("  results = evaluator.evaluate_video(...)")
