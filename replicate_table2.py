#!/usr/bin/env python3
"""
Replicate Table 2 from the paper: Multilingual Generation Control Results

This script replicates the full Table 2 results for:
- Models: Llama2-7B, Llama2-13B, Llama3.1-8B, Aya23-8B
- Languages: French, German, Chinese, Japanese, Spanish
- Settings: Monolingual and Parallel
- 3 runs with different random seeds
- Evaluation on FLORES-200

Key Process:
1. IDENTIFICATION: Use 50 samples to identify language-specific dimensions
   (NO training! Just inference to extract hidden states)
2. EVALUATION: Test intervention on separate 100 samples
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os
import gc
from typing import Dict, List, Tuple

# Import functional implementations
from functional_implementation import (
    identify_dimensions_monolingual,
    identify_dimensions_parallel,
    compute_steering_vector,
    apply_intervention,
    compute_bleu,
)

from fasttext_acc import (
    load_fasttext_model,
    compute_accuracy_fasttext,
    get_fasttext_lang_code
)

from flores200_loader import Flores200Loader
from config import MODEL_CONFIGS, LANGUAGE_NAMES, LANGUAGE_SHORT

prompt_func = lambda eng: f"Translate an English sentence into a target language. English: {eng} Target language:"


def setup_model(model_name: str, device: str = "cuda", hf_token: str = None):
    """Load model and tokenizer with optimized device placement"""
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True
    )
    
    if device == "cuda":
        # Try to load entire model on GPU first (optimal for A100 80GB)
        try:
            print("  Attempting to load entire model on GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16,
                device_map={"": 0},  # Force all layers on GPU 0
                low_cpu_mem_usage=True
            )
            print("  ✓ Model fully loaded on GPU")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ✗ GPU OOM, falling back to device_map='auto'")
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=hf_token,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "75GB"}  # Use most of A100's 80GB
                )
            else:
                raise e
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float32,
            device_map=None
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    
    # Print device distribution
    if hasattr(model, 'hf_device_map'):
        devices = set(model.hf_device_map.values())
        print(f"  Model distributed across devices: {devices}")
        if 'cpu' in devices or 'disk' in devices:
            print(f"  ⚠️  Warning: Some layers on CPU/disk (will be slow!)")
    
    return model, tokenizer


def sample_identification_data(
    flores: Flores200Loader,
    language: str,
    num_samples: int,
    seed: int
) -> Tuple[List[str], List[str]]:
    """
    Sample sentences for dimension identification (NOT training!)
    
    These samples are used to:
    - Extract hidden states (inference only, no training!)
    - Compute mean representations
    - Identify top-K dimensions
    
    NO training, NO gradient updates occur!
    
    Args:
        flores: FLORES-200 loader
        language: Target language code
        num_samples: Number of samples to draw
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (english_sentences, target_sentences)
    """
    np.random.seed(seed)
    df = flores.load_language_pair("eng_Latn", language, split="devtest")
    
    # Sample from first 800 sentences (reserve last 200+ for testing)
    available_indices = list(range(min(800, len(df))))
    sampled_indices = np.random.choice(available_indices, size=num_samples, replace=False)
    sampled_indices = sorted(sampled_indices)
    
    english_sentences = df.iloc[sampled_indices]["sentence_eng_Latn"].tolist()
    target_sentences = df.iloc[sampled_indices][f"sentence_{language}"].tolist()
    
    return english_sentences, target_sentences


def load_test_data(
    flores: Flores200Loader,
    language: str,
    num_samples: int,
    dataset_name: str = "FLORES-200"
) -> Tuple[List[str], List[str]]:
    """
    Load test data from specified dataset
    
    Args:
        flores: FLORES-200 loader
        language: Target language code
        num_samples: Number of test samples
        dataset_name: Name of dataset (FLORES-200, IWSLT2017, WMT)
    
    Returns:
        Tuple of (english_sentences, target_sentences) or (None, None) if not available
    """
    if dataset_name == "FLORES-200":
        df = flores.load_language_pair("eng_Latn", language, split="devtest")
        # Use last num_samples for testing
        test_df = df.tail(num_samples)
        english_sentences = test_df["sentence_eng_Latn"].tolist()
        target_sentences = test_df[f"sentence_{language}"].tolist()
        return english_sentences, target_sentences


def grid_search_hyperparams(
    model,
    tokenizer,
    flores: Flores200Loader,
    language: str,
    setting: str,
    model_config: dict,
    top_k: int = 400,
    device: str = "cuda"
) -> Tuple[int, float]:
    """
    Grid search for best intervention layer and alpha
    
    Uses all 50 samples for both identification and validation.
    
    Returns:
        Tuple of (best_intervention_layer, best_alpha)
    """
    print(f"  Grid searching hyperparameters...")
    
    # Sample 50 sentences for identification (use ALL 50 for validation too)
    id_english, id_target = sample_identification_data(flores, language, num_samples=50, seed=9999)
    
    # Use same samples for dimension identification
    # (In grid search, we reuse identification samples for quick validation)
    
    # Note: outputs.hidden_states[0] = embeddings, hidden_states[num_hidden_layers] = final layer
    final_layer = model.config.num_hidden_layers
    
    # Identify dimensions using all 50 samples
    if setting == "monolingual":
        dim_indices, _, mu_inter, mu_final = identify_dimensions_monolingual(
            model=model,
            tokenizer=tokenizer,
            target_sentences=id_target,
            intermediate_layer=model_config[setting]["intermediate_layer"],
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        steering_vec = mu_final - mu_inter
    else:
        dim_indices, _, mu_en, mu_target = identify_dimensions_parallel(
            model=model,
            tokenizer=tokenizer,
            english_sentences=id_english,
            target_sentences=id_target,
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        steering_vec = mu_target - mu_en
    
    # Load fastText
    fasttext_model = load_fasttext_model()
    target_lang_code = get_fasttext_lang_code(language)
    lang_name = LANGUAGE_NAMES[language]
    
    # Grid search
    best_composite = 0
    best_config = (20, 0.4)  # Default
    
    # Test different layers and alphas
    test_layers = [16, 20, 24, 28] if model.config.num_hidden_layers >= 32 else [16, 20, 24]
    test_alphas = [0.3, 0.4, 0.5, 0.6]
    
    for intervention_layer in test_layers:
        for alpha in test_alphas:
            # Generate with intervention on all 50 identification samples
            generated_texts = []
            prompts = []
            
            for eng in id_english:  # Use ALL 50 samples
                prompt = prompt_func(eng)
                prompts.append(prompt)
                
                # Apply scaled steering vector
                scaled_steering = alpha * steering_vec
                
                output = apply_intervention(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    dimension_indices=dim_indices,
                    steering_vector=scaled_steering,
                    intervention_layer=intervention_layer,
                    max_new_tokens=50,
                    device=device,
                    do_sample=False
                )
                generated_texts.append(output)
            
            # Evaluate on all 50 samples
            acc, successes, _ = compute_accuracy_fasttext(
                generated_texts, prompts, target_lang_code, fasttext_model, threshold=0.5
            )
            
            # Calculate BLEU only on successful samples
            if sum(successes) > 0:
                successful_generated = [gen for gen, success in zip(generated_texts, successes) if success]
                successful_references = [ref for ref, success in zip(id_target, successes) if success]
                bleu, _ = compute_bleu(successful_generated, [], successful_references)
            else:
                bleu = 0.0
            
            composite = acc * bleu
            
            if composite > best_composite:
                best_composite = composite
                best_config = (intervention_layer, alpha)
    
    print(f"  Best config: layer={best_config[0]}, alpha={best_config[1]:.1f}, composite={best_composite:.2f}")
    return best_config


def run_single_experiment(
    model,
    tokenizer,
    flores: Flores200Loader,
    language: str,
    setting: str,
    run_idx: int,
    intervention_layer: int,
    alpha: float,
    model_config: dict,
    top_k: int = 400,
    num_identification: int = 50,
    num_test: int = 100,
    device: str = "cuda"
) -> List[Dict]:
    """
    Run a single experiment (one run, all datasets)
    
    Uses all 50 samples for dimension identification.
    Evaluates on separate 100 test samples.
    
    Returns:
        List of results for each dataset
    """
    print(f"\n  Run {run_idx + 1}/3")
    
    # Step 1: Sample data for dimension identification (NOT training!)
    seed = 42 + run_idx
    id_english, id_target = sample_identification_data(
        flores, language, num_identification, seed
    )
    
    # Step 2: Identify dimensions using ALL 50 samples
    print(f"    Identifying dimensions on {num_identification} samples...")
    # Note: outputs.hidden_states includes embedding at index 0, so final layer is at index num_hidden_layers
    final_layer = model.config.num_hidden_layers  # For 32 layers: hidden_states[32] = final layer output
    
    if setting == "monolingual":
        # Use all 50 identification samples
        # Compare intermediate layer with final layer to find language-specific dimensions
        dim_indices, _, mu_inter, mu_final = identify_dimensions_monolingual(
            model=model,
            tokenizer=tokenizer,
            target_sentences=id_target,  # All 50 samples
            intermediate_layer=model_config[setting]["intermediate_layer"],
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        steering_vec = compute_steering_vector(mu_inter, mu_final, alpha)
    else:
        # Use all 50 identification samples
        dim_indices, _, mu_en, mu_target = identify_dimensions_parallel(
            model=model,
            tokenizer=tokenizer,
            english_sentences=id_english,  # All 50 samples
            target_sentences=id_target,    # All 50 samples
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        steering_vec = compute_steering_vector(mu_en, mu_target, alpha)
    
    # Step 3: Evaluate on each dataset
    datasets = ["FLORES-200"]  # Can add "IWSLT2017", "WMT" when implemented
    results = []
    
    # Load fastText
    fasttext_model = load_fasttext_model()
    target_lang_code = get_fasttext_lang_code(language)
    lang_name = LANGUAGE_NAMES[language]
    
    for dataset_name in datasets:
        test_english, test_target = load_test_data(flores, language, num_test, dataset_name)
        
        if test_english is None:
            continue
        
        print(f"    Evaluating on {dataset_name}...")
        
        # Generate with intervention
        generated_texts = []
        prompts = []
        
        for eng in tqdm(test_english, desc=f"      Generating", leave=False):
            prompt = prompt_func(eng)
            prompts.append(prompt)
            
            output = apply_intervention(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                dimension_indices=dim_indices,
                steering_vector=steering_vec,
                intervention_layer=intervention_layer,
                max_new_tokens=50,
                device=device,
                do_sample=False
            )
            generated_texts.append(output)
        
        # Compute metrics
        acc, successes, confidences = compute_accuracy_fasttext(
            generated_texts, prompts, target_lang_code, fasttext_model, threshold=0.5, verbose=False
        )
        
        # Calculate BLEU only on successful samples (where language was correctly detected)
        if sum(successes) > 0:
            successful_generated = [gen for gen, success in zip(generated_texts, successes) if success]
            successful_references = [ref for ref, success in zip(test_target, successes) if success]
            bleu, _ = compute_bleu(successful_generated, [], successful_references)
        else:
            bleu = 0.0
        
        composite = acc * bleu
        
        results.append({
            "run": run_idx,
            "dataset": dataset_name,
            "acc": float(acc),
            "bleu": float(bleu),
            "composite": float(composite),
            "num_successful": int(sum(successes)),
            "num_total": len(successes)
        })
        
        print(f"      ACC={acc:.1f}, BLEU={bleu:.1f}, A*B={composite:.2f} (BLEU on {sum(successes)}/{len(successes)} successful samples)")
    
    return results


def replicate_table2_entry(
    model_name: str,
    language: str,
    setting: str,
    output_dir: Path,
    num_runs: int = 3,
    do_grid_search: bool = False,
    device: str = "cuda",
    hf_token: str = None
) -> Dict:
    """
    Replicate one entry in Table 2
    
    Args:
        model_name: Model identifier
        language: Target language code
        setting: "monolingual" or "parallel"
        output_dir: Output directory
        num_runs: Number of runs with different seeds
        do_grid_search: Whether to perform grid search (slow)
        device: Device to use
        hf_token: Hugging Face token
    
    Returns:
        Dict with final averaged results
    """
    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"Language: {language} ({LANGUAGE_SHORT[language]})")
    print(f"Setting: {setting}")
    print(f"{'='*80}")
    
    # Load FLORES-200 (only once, doesn't use GPU memory)
    print("Loading FLORES-200...")
    flores = Flores200Loader(cache_dir="./data/flores200")
    
    # Configuration (from paper)
    top_k = 400              # Top-400 dimensions
    num_identification = 50   # 50 samples for dimension identification
    num_test = 100           # 100 samples for evaluation
    
    # Get model config
    model_config = MODEL_CONFIGS[model_name]
    
    # Grid search or use defaults (do once before runs)
    if do_grid_search:
        # Load model for grid search
        model, tokenizer = setup_model(model_name, device, hf_token)
        intervention_layer, alpha = grid_search_hyperparams(
            model, tokenizer, flores, language, setting,
            model_config, top_k, device
        )
        # Free model after grid search
        print("\nFreeing model after grid search...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print("  ✓ GPU memory freed")
    else:
        # Use defaults from config for this model and setting
        intervention_layer = model_config[setting]["intervention_layer"]
        alpha = model_config[setting]["alpha"]
        print(f"  Using config: layer={intervention_layer}, alpha={alpha}")
    
    # Run multiple experiments with fresh model for each seed
    all_results = []
    
    for run_idx in range(num_runs):
        print(f"\n  {'='*70}")
        print(f"  Run {run_idx + 1}/{num_runs} - Loading fresh model for reproducibility")
        print(f"  {'='*70}")
        
        # Load fresh model for this seed
        model, tokenizer = setup_model(model_name, device, hf_token)
        
        # Run experiment with this seed
        run_results = run_single_experiment(
            model, tokenizer, flores, language, setting,
            run_idx, intervention_layer, alpha, model_config,
            top_k, num_identification, num_test, device
        )
        all_results.extend(run_results)
        
        # Free model after this seed to ensure clean state for next seed
        print(f"\n  Freeing model after seed {run_idx + 1}...")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ✓ GPU memory freed")
    
    # Average across all runs and datasets
    final_acc = np.mean([r["acc"] for r in all_results])
    final_bleu = np.mean([r["bleu"] for r in all_results])
    final_composite = np.mean([r["composite"] for r in all_results])
    
    print(f"\n  {'='*70}")
    print(f"  FINAL (averaged over {num_runs} seeds): ACC={final_acc:.1f}, BLEU={final_bleu:.1f}, A*B={final_composite:.2f}")
    print(f"  {'='*70}")
    
    result = {
        "model": model_name,
        "language": language,
        "language_short": LANGUAGE_SHORT[language],
        "setting": setting,
        "acc": float(final_acc),
        "bleu": float(final_bleu),
        "composite": float(final_composite),
        "intervention_layer": intervention_layer,
        "alpha": alpha,
        "num_runs": num_runs,
        "all_run_results": all_results
    }
    
    # Save result
    result_file = output_dir / f"{model_name.replace('/', '_')}_{language}_{setting}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"  Results saved to: {result_file.name}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Replicate Table 2 Results")
    parser.add_argument("--models", nargs='+',
                       default=["meta-llama/Llama-2-7b-hf"],
                       help="Models to test")
    parser.add_argument("--languages", nargs='+',
                       default=["fra_Latn", "deu_Latn", "zho_Hans", "jpn_Jpan", "spa_Latn"],
                       help="Languages to test")
    parser.add_argument("--settings", nargs='+',
                       choices=["monolingual", "parallel"],
                       default=["monolingual", "parallel"],
                       help="Settings to test")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of runs with different seeds")
    parser.add_argument("--grid_search", action="store_true",
                       help="Perform grid search for hyperparameters (slow)")
    parser.add_argument("--output_dir", type=str,
                       default="results/table2_replication",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    
    # Get HF token
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    all_results = {}
    
    for model_name in args.models:
        model_short = model_name.split('/')[-1]
        all_results[model_short] = {}
        
        for language in args.languages:
            lang_short = LANGUAGE_SHORT[language]
            all_results[model_short][lang_short] = {}
            
            for setting in args.settings:
                result = replicate_table2_entry(
                    model_name=model_name,
                    language=language,
                    setting=setting,
                    output_dir=output_dir,
                    num_runs=args.num_runs,
                    do_grid_search=args.grid_search,
                    device=args.device,
                    hf_token=hf_token
                )
                
                all_results[model_short][lang_short][setting] = {
                    "acc": result["acc"],
                    "bleu": result["bleu"],
                    "composite": result["composite"]
                }
    
    # Save summary
    with open(output_dir / "table2_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print table
    print("\n" + "="*80)
    print("TABLE 2 REPLICATION RESULTS")
    print("="*80)
    
    for model_short, model_results in all_results.items():
        print(f"\n{model_short}:")
        print(f"{'Lang':<6} {'Setting':<12} {'ACC':>6} {'BLEU':>6} {'A*B':>6}")
        print("-"*50)
        
        for lang_short, lang_results in model_results.items():
            for setting, metrics in lang_results.items():
                print(f"{lang_short:<6} {setting:<12} "
                      f"{metrics['acc']:>6.1f} {metrics['bleu']:>6.1f} {metrics['composite']:>6.2f}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

