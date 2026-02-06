"""
Run Interpretability Analysis for All Mamba Variants
====================================================

This script runs interpretability analysis separately and saves results to JSON.
The results can then be loaded by steering_s6_types_jamba.py to avoid re-running
interpretability each time.
"""

import torch
import logging
import json
from pathlib import Path
from typing import Dict
import sys
import os

# Import from steering script
from steering_s6_types_jamba import (
    MAMBA_VARIANTS,
    MambaVariantAnalyzer,
    run_interpretability_analysis,
    load_model_variant,
    CLUSTER_9_NEURONS
)

# Configure logging to show all output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Also configure the steering script's logger
steering_logger = logging.getLogger('steering_s6_types_jamba')
steering_logger.setLevel(logging.INFO)


def run_all_interpretability(output_file: str = "experiment_logs/interpretability_results.json"):
    """
    Run interpretability analysis on all Mamba variants and save to JSON.
    
    Args:
        output_file: Path to save the JSON results
    """
    logger.info("="*80)
    logger.info("🔬 INTERPRETABILITY ANALYSIS FOR ALL MAMBA VARIANTS")
    logger.info("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    all_results = {}
    
    # Process each variant
    for variant_name in MAMBA_VARIANTS.keys():
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing: {variant_name.upper()}")
        logger.info(f"{'='*80}")
        
        # Load model
        model, tokenizer = load_model_variant(variant_name, device)
        if model is None:
            logger.warning(f"⚠️ Skipping {variant_name} (model not available)")
            all_results[variant_name] = {
                "error": "Model not available",
                "skipped": True
            }
            continue
        
        try:
            # Run interpretability analysis
            logger.info(f"\n🔬 Running interpretability analysis for {variant_name}...")
            logger.info(f"   This may take a few minutes...")
            logger.info(f"   Model type: {type(model).__name__}")
            logger.info(f"   Device: {device}")
            
            try:
                interpretability_results = run_interpretability_analysis(model, tokenizer, variant_name)
                
                # Store results
                all_results[variant_name] = interpretability_results
                
                logger.info(f"\n{'='*80}")
                logger.info(f"✅ Completed interpretability for {variant_name}")
                logger.info(f"{'='*80}")
                logger.info(f"   📊 Results Summary:")
                logger.info(f"      - Bottleneck layer: {interpretability_results['bottleneck_layer']} ({interpretability_results['bottleneck_pct']:.1f}% depth)")
                logger.info(f"      - Critical layer: {interpretability_results['critical_layer']} ({interpretability_results['critical_layer_pct']:.1f}% depth)")
                logger.info(f"      - Cluster neurons: {len(interpretability_results['cluster_neurons'])} identified")
                logger.info(f"      - Cluster 9 overlap: {interpretability_results['cluster_9_overlap']}/{len(CLUSTER_9_NEURONS)}")
                logger.info(f"      - Total layers: {interpretability_results['num_layers']}")
                if interpretability_results['cluster_neurons']:
                    logger.info(f"      - Sample neurons: {interpretability_results['cluster_neurons'][:5]}...")
                logger.info(f"{'='*80}\n")
                
            except Exception as e:
                logger.error(f"\n❌ Interpretability analysis failed for {variant_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[variant_name] = {
                    "error": str(e),
                    "skipped": True,
                    "traceback": traceback.format_exc()
                }
            
            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            logger.error(f"\n❌ Error analyzing {variant_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_results[variant_name] = {
                "error": str(e),
                "skipped": True,
                "traceback": traceback.format_exc()
            }
            continue
    
    # Save results to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"💾 Results saved to: {output_path}")
    logger.info(f"{'='*80}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 SUMMARY:")
    logger.info(f"{'='*80}")
    
    if not all_results:
        logger.warning("⚠️  No results generated! Check for errors above.")
    else:
        for variant_name, results in all_results.items():
            if "error" in results or results.get("skipped"):
                logger.info(f"   {variant_name}: ❌ Skipped or Error")
                if "error" in results:
                    logger.info(f"      Error: {results['error']}")
            else:
                logger.info(f"   {variant_name}: ✅ Analyzed")
                logger.info(f"      Bottleneck: Layer {results.get('bottleneck_layer', 'N/A')} ({results.get('bottleneck_pct', 0):.1f}%)")
                logger.info(f"      Critical: Layer {results.get('critical_layer', 'N/A')} ({results.get('critical_layer_pct', 0):.1f}%)")
                logger.info(f"      Neurons: {len(results.get('cluster_neurons', []))} identified")
                logger.info(f"      Cluster 9 overlap: {results.get('cluster_9_overlap', 0)}")
    
    logger.info(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run interpretability analysis for Mamba variants")
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_logs/interpretability_results.json",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    results = run_all_interpretability(args.output)
    
    print("\n" + "="*80)
    print("✅ INTERPRETABILITY ANALYSIS COMPLETE")
    print("="*80)

