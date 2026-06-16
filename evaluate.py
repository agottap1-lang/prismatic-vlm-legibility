#!/usr/bin/env python3
"""Legibility evaluation - observation-based approach."""

import json
import argparse
from pathlib import Path
from src.legibility_eval.client import PrismaticTemporalClient


def evaluate_video(client, video_entry, times_to_eval):
    """Evaluate single video at specified timestamps."""
    video_id = video_entry["video_id"]
    gt_goal = video_entry['goal_gt']
    gt_desc = video_entry[f'goal_{gt_goal}']
    
    print(f"\n{'='*70}")
    print(f"{video_id}: Ground truth = Goal {gt_goal} ({gt_desc})")
    print(f"{'='*70}")
    
    results = []
    
    for t in times_to_eval:
        result = client.evaluate_video_at_time(
            video_path=video_entry["video_path"],
            t_current=t,
            goal_A_desc=video_entry["goal_A"],
            goal_B_desc=video_entry["goal_B"],
            video_id=video_id,
            n_frames=6,
        )
        
        # Check correctness
        correct = (result['choice'] == gt_goal) if result['choice'] else None
        status = "✓" if correct else ("✗" if correct is False else "?")
        
        # Print
        print(f"t={t}: pA={result['pA']:.2f} pB={result['pB']:.2f} → {result['choice']} {status}")
        print(f"     VLM Cue: {result['cue']}")
        print(f"     VLM Legible: {result['legible']}")
        print(f"     Code Confidence: {result['confidence']:.2f}")
        
        # Store
        results.append({
            "video_id": video_id,
            "t": t,
            "pA": result['pA'],
            "pB": result['pB'],
            "choice": result['choice'],
            "confidence": result['confidence'],
            "cue": result['cue'],
            "legible": result['legible'],
            "ground_truth": gt_goal,
            "correct": correct,
            "vlm_raw": result['vlm_raw_response'],
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--times", nargs="+", type=int, default=[0, 5])
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    
    # Load manifest
    with open("data/manifest.jsonl") as f:
        manifest = [json.loads(line) for line in f]
    
    # Initialize client once
    client = PrismaticTemporalClient(model_path="prism-dinosiglip+7b")
    
    # Evaluate
    all_results = []
    for video_id in args.videos:
        video_entry = next((v for v in manifest if v["video_id"] == video_id), None)
        if not video_entry:
            print(f"[SKIP] {video_id} not found in manifest")
            continue
        
        results = evaluate_video(client, video_entry, args.times)
        all_results.extend(results)
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    total = len(all_results)
    correct = sum(1 for r in all_results if r['correct'] is True)
    incorrect = sum(1 for r in all_results if r['correct'] is False)
    uncertain = sum(1 for r in all_results if r['correct'] is None)
    
    print(f"\n{'='*70}")
    print(f"Total: {total} | Correct: {correct} ({100*correct/total:.0f}%) | Incorrect: {incorrect} | Uncertain: {uncertain}")
    print(f"Saved: {args.output}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
