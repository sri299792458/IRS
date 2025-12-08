#!/usr/bin/env python3
"""
Ablation Study: Performance by Referring Expression Type

Replicates CROG paper Section 5.2, Figure 3 analysis.

Expression Types (from OCID-VLG):
- Name: Object name/category only (e.g., "chocolate corn flakes")
- Attribute: Color or property (e.g., "brown cereal box")
- Relation: Spatial relation to another object (e.g., "bowl left of mug")
- Location: Absolute position (e.g., "leftmost bowl", "closest cereal")
- Mixed: Combination of above (e.g., "red apple behind the blue mug")

Key Finding from CROG:
- SSG+CLIP baseline: <30% on relations/locations (loses spatial info in segment-then-rank)
- CROG: Robust across all types (~70-80%)

For our VLM approach, we expect:
- Uniform performance across types (VLM processes full context)
- Potentially better on relations/locations (language understanding)
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# Expression Type Classification
# =============================================================================

# Keywords for classification
LOCATION_KEYWORDS = [
    'leftmost', 'rightmost', 'furthest', 'closest', 'nearest',
    'left side', 'right side', 'far', 'near',
    'top', 'bottom', 'middle', 'center'
]

RELATION_KEYWORDS = [
    'left of', 'right of', 'behind', 'in front of', 'front of',
    'rear left', 'rear right', 'front left', 'front right',
    'next to', 'beside', 'on top of', 'below', 'above',
    'that is left', 'that is right', 'that is behind', 'that is in front',
    'to the left', 'to the right', 'to the rear', 'to the front',
    'on the left', 'on the right'
]

COLOR_KEYWORDS = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
    'brown', 'black', 'white', 'gray', 'grey', 'beige', 'dark', 'light',
    'colored', 'coloured'
]

ATTRIBUTE_KEYWORDS = COLOR_KEYWORDS + [
    'small', 'large', 'big', 'tall', 'short', 'round', 'square',
    'shiny', 'matte', 'spotted', 'striped', 'plain',
    'empty', 'full', 'open', 'closed'
]


def classify_expression(sentence: str) -> str:
    """
    Classify a referring expression into one of 5 types.
    
    Priority order (for mixed cases):
    1. If has relation keywords AND something else → Mixed
    2. If has relation keywords only → Relation  
    3. If has location keywords only → Location
    4. If has attribute/color keywords only → Attribute
    5. Otherwise → Name
    """
    sentence_lower = sentence.lower()
    
    has_relation = any(kw in sentence_lower for kw in RELATION_KEYWORDS)
    has_location = any(kw in sentence_lower for kw in LOCATION_KEYWORDS)
    has_attribute = any(kw in sentence_lower for kw in ATTRIBUTE_KEYWORDS)
    
    # Count how many types are present
    type_count = sum([has_relation, has_location, has_attribute])
    
    if type_count >= 2:
        return 'Mixed'
    elif has_relation:
        return 'Relation'
    elif has_location:
        return 'Location'
    elif has_attribute:
        return 'Attribute'
    else:
        return 'Name'


def classify_expression_v2(sentence: str, metadata: dict = None) -> str:
    """
    Alternative classification using template patterns from OCID-VLG.
    
    Template structure: [prefix] ([LOC1] [ATT1]) [OBJ1] ((that is) [REL] the ([LOC2] [ATT2]) [OBJ2])
    """
    sentence_lower = sentence.lower()
    
    # Check for relation patterns (references another object)
    relation_patterns = [
        r'(left|right|behind|front|rear) of the',
        r'(left|right|behind|front|rear) from the',
        r'that is (left|right|behind|front|rear)',
        r'to the (left|right|rear|front) of',
        r'on the (left|right) side of',
        r'next to the',
        r'beside the',
        r'near the',
    ]
    has_relation = any(re.search(p, sentence_lower) for p in relation_patterns)
    
    # Check for absolute location
    location_patterns = [
        r'\b(leftmost|rightmost|furthest|closest|nearest)\b',
        r'\bthe (left|right|far|near|closest) ',
        r'\b(most left|most right)\b',
    ]
    has_location = any(re.search(p, sentence_lower) for p in location_patterns)
    
    # Check for attributes
    has_attribute = any(kw in sentence_lower for kw in ATTRIBUTE_KEYWORDS)
    
    # Classification logic
    if has_relation and (has_location or has_attribute):
        return 'Mixed'
    elif has_location and has_attribute:
        return 'Mixed'
    elif has_relation:
        return 'Relation'
    elif has_location:
        return 'Location'
    elif has_attribute:
        return 'Attribute'
    else:
        return 'Name'


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results_by_type(results: dict, use_v2: bool = True) -> Dict[str, dict]:
    """
    Break down evaluation results by expression type.
    
    Args:
        results: Dict with 'sentences', 'successes', 'ious', 'targets'
        use_v2: Use pattern-based classification
        
    Returns:
        Dict mapping type → metrics
    """
    classify_fn = classify_expression_v2 if use_v2 else classify_expression
    
    # Group by type
    by_type = defaultdict(lambda: {'successes': [], 'ious': [], 'sentences': []})
    
    for sent, success, iou in zip(results['sentences'], results['successes'], results['ious']):
        expr_type = classify_fn(sent)
        by_type[expr_type]['successes'].append(success)
        by_type[expr_type]['ious'].append(iou)
        by_type[expr_type]['sentences'].append(sent)
    
    # Compute metrics per type
    metrics_by_type = {}
    for expr_type, data in by_type.items():
        metrics_by_type[expr_type] = {
            'num_samples': len(data['successes']),
            'success_rate': np.mean(data['successes']) if data['successes'] else 0,
            'mean_iou': np.mean(data['ious']) if data['ious'] else 0,
            'std_success': np.std(data['successes']) if data['successes'] else 0,
            'example_sentences': data['sentences'][:3],  # First 3 examples
        }
    
    return metrics_by_type


def print_type_analysis(metrics_by_type: dict, title: str = "Expression Type Analysis"):
    """Pretty print the type breakdown."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    
    # Define order
    type_order = ['Name', 'Attribute', 'Relation', 'Location', 'Mixed']
    
    # Header
    print(f"\n{'Type':<12} {'Samples':>8} {'Success%':>10} {'Mean IoU':>10} {'Std':>8}")
    print("-" * 50)
    
    total_samples = 0
    total_success = 0
    
    for expr_type in type_order:
        if expr_type in metrics_by_type:
            m = metrics_by_type[expr_type]
            sr = m['success_rate'] * 100
            total_samples += m['num_samples']
            total_success += m['success_rate'] * m['num_samples']
            
            print(f"{expr_type:<12} {m['num_samples']:>8} {sr:>9.1f}% {m['mean_iou']:>10.3f} {m['std_success']:>8.3f}")
    
    # Overall
    print("-" * 50)
    overall_sr = (total_success / total_samples * 100) if total_samples > 0 else 0
    print(f"{'Overall':<12} {total_samples:>8} {overall_sr:>9.1f}%")
    
    print("=" * 70)
    
    # Example sentences
    print("\nExample Sentences per Type:")
    print("-" * 50)
    for expr_type in type_order:
        if expr_type in metrics_by_type:
            examples = metrics_by_type[expr_type].get('example_sentences', [])
            if examples:
                print(f"\n{expr_type}:")
                for ex in examples[:2]:
                    # Truncate long sentences
                    ex_short = ex[:60] + "..." if len(ex) > 60 else ex
                    print(f"  - \"{ex_short}\"")


def create_comparison_chart(ours: dict, baseline: dict = None, output_path: str = None):
    """
    Create a bar chart comparing performance by expression type.
    Similar to CROG Figure 3.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return
    
    type_order = ['Name', 'Attribute', 'Relation', 'Location', 'Mixed']
    
    # Get data
    ours_sr = [ours.get(t, {}).get('success_rate', 0) * 100 for t in type_order]
    
    x = np.arange(len(type_order))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if baseline:
        baseline_sr = [baseline.get(t, {}).get('success_rate', 0) * 100 for t in type_order]
        bars1 = ax.bar(x - width/2, baseline_sr, width, label='Baseline (SSG+CLIP)', color='#4A90D9', alpha=0.8)
        bars2 = ax.bar(x + width/2, ours_sr, width, label='Ours (VLM)', color='#E94B4B', alpha=0.8)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    else:
        bars = ax.bar(x, ours_sr, width, label='Ours (VLM)', color='#E94B4B', alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Grasp Success Rate (%)', fontsize=12)
    ax.set_xlabel('Referring Expression Type', fontsize=12)
    ax.set_title('Performance by Expression Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_order, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {output_path}")
    else:
        plt.show()


# =============================================================================
# CROG Baseline Numbers (from paper Figure 3)
# =============================================================================

# These are approximate values read from CROG paper Figure 3 and Table 6
# Test split: Name=5809, Attribute=781, Relation=5769, Location=2672, Mixed=2718
CROG_BASELINE_SSG_CLIP = {
    'Name': {'num_samples': 5809, 'success_rate': 0.60, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': ["Pick the chocolate corn flakes", "Grasp the soda can"]},
    'Attribute': {'num_samples': 781, 'success_rate': 0.55, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': ["Grab the blue ball", "Pick the red apple"]},
    'Relation': {'num_samples': 5769, 'success_rate': 0.25, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': ["Pick the bowl left of the mug", "Grasp the cereal behind the apple"]},
    'Location': {'num_samples': 2672, 'success_rate': 0.28, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': ["Grasp the leftmost marker", "Pick the closest shampoo"]},
    'Mixed': {'num_samples': 2718, 'success_rate': 0.15, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': ["Grab the red apple behind the blue mug", "Pick the leftmost brown cereal box"]},
}

CROG_OURS = {
    'Name': {'num_samples': 5809, 'success_rate': 0.82, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': []},
    'Attribute': {'num_samples': 781, 'success_rate': 0.75, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': []},
    'Relation': {'num_samples': 5769, 'success_rate': 0.72, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': []},
    'Location': {'num_samples': 2672, 'success_rate': 0.75, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': []},
    'Mixed': {'num_samples': 2718, 'success_rate': 0.68, 'mean_iou': 0.0, 'std_success': 0.0, 'example_sentences': []},
}


# =============================================================================
# Main Analysis
# =============================================================================

def main(args):
    print("=" * 70)
    print("ABLATION STUDY: Performance by Expression Type")
    print("=" * 70)
    
    # Load results
    if args.results_file and os.path.exists(args.results_file):
        print(f"\nLoading results from: {args.results_file}")
        with open(args.results_file) as f:
            results = json.load(f)
        
        # Analyze by type
        metrics_by_type = analyze_results_by_type(results, use_v2=True)
        
        # Print analysis
        print_type_analysis(metrics_by_type, title="Our VLM Results by Expression Type")
        
        # Create chart
        if args.output_chart:
            create_comparison_chart(
                ours=metrics_by_type, 
                baseline=CROG_BASELINE_SSG_CLIP if args.show_baseline else None,
                output_path=args.output_chart
            )
        
        # Save detailed results
        if args.output_json:
            # Convert to serializable format
            save_data = {}
            for k, v in metrics_by_type.items():
                save_data[k] = {
                    'num_samples': v['num_samples'],
                    'success_rate': float(v['success_rate']),
                    'mean_iou': float(v['mean_iou']),
                    'std_success': float(v['std_success']),
                }
            
            with open(args.output_json, 'w') as f:
                json.dump(save_data, f, indent=2)
            print(f"\nDetailed results saved to: {args.output_json}")
    
    else:
        # Demo mode with CROG baseline numbers
        print("\n[DEMO MODE - Using CROG paper baseline numbers]")
        print("\nCROG Baseline (SSG+CLIP):")
        print_type_analysis(CROG_BASELINE_SSG_CLIP, "SSG+CLIP Baseline (from CROG paper)")
        
        print("\nCROG Results:")
        print_type_analysis(CROG_OURS, "CROG (from paper)")
        
        print("\n" + "=" * 70)
        print("KEY INSIGHT FROM CROG PAPER:")
        print("=" * 70)
        print("""
SSG+CLIP struggles with spatial concepts (Relation, Location, Mixed)
because the segment-then-rank pipeline loses spatial context:
  - Relation: 25% (needs to know where OTHER objects are)
  - Location: 28% (needs to know absolute position)
  - Mixed:    15% (combines multiple spatial concepts)

CROG solves this with dense pixel-text alignment (cross-attention decoder).

For our VLM approach, we expect:
  - Uniform performance across types (processes full image + text)
  - Especially strong on Relation/Location (language understanding)
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze performance by expression type")
    parser.add_argument('--results_file', type=str, default=None,
                       help='Path to evaluation results JSON with sentences and successes')
    parser.add_argument('--output_chart', type=str, default=None,
                       help='Path to save comparison chart')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save detailed metrics JSON')
    parser.add_argument('--show_baseline', action='store_true',
                       help='Include CROG baseline in chart')
    
    args = parser.parse_args()
    main(args)