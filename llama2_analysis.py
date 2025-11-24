#!/usr/bin/env python3
"""
Replicate Paper Results: Table 1, Figures 3, 4, 5

This script replicates the key results from the paper for Llama-2-7B:
- Table 1: Overlap matrix of language-specific dimensions
- Figure 3: Top-K threshold selection (parallel setting)
- Figure 4: Anchor layer selection (monolingual setting)
- Figure 5: Monolingual vs Parallel overlap rate

Each experiment is saved in a separate folder.
"""

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import os

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

prompt_func = lambda eng: f"Translate an English sentence into a target language. English: {eng} Target language:"

def setup_model(model_name: str, device: str = "cuda", hf_token: str = None):
    """Load model and tokenizer"""
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers")
    return model, tokenizer


def experiment_table1_dimension_overlap(
    model, tokenizer, flores, languages, output_dir, device="cuda"
):
    """
    Replicate Table 1: Overlap matrix of top-400 language-specific dimensions
    
    Args:
        languages: List of language codes (e.g., ['zho_Hans', 'jpn_Jpan', 'fra_Latn', 'spa_Latn', 'deu_Latn'])
    """
    print("\n" + "="*80)
    print("TABLE 1: Dimension Overlap Matrix (Monolingual Setting)")
    print("="*80)
    
    output_dir = Path(output_dir) / "table1_overlap_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Settings
    top_k = 400
    intermediate_layer = 19
    final_layer = model.config.num_hidden_layers - 1
    num_samples = 50
    
    # Store all dimension sets
    all_dimensions = {}
    
    # Identify dimensions for each language
    for lang_code in languages:
        print(f"\n--- Processing {lang_code} ---")
        
        # Load data
        df = flores.load_language_pair("eng_Latn", lang_code, split="devtest")
        target_sentences = df.head(num_samples)[f"sentence_{lang_code}"].tolist()
        
        # Identify dimensions
        dim_indices, _, _, _ = identify_dimensions_monolingual(
            model=model,
            tokenizer=tokenizer,
            target_sentences=target_sentences,
            intermediate_layer=intermediate_layer,
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        
        all_dimensions[lang_code] = set(dim_indices.tolist())
        
        # Save dimensions
        with open(output_dir / f"{lang_code}_dimensions.json", 'w') as f:
            json.dump({"dimensions": dim_indices.tolist()}, f)
    
    # Compute overlap matrix
    overlap_matrix = np.zeros((len(languages), len(languages)), dtype=int)
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i == j:
                overlap_matrix[i, j] = top_k
            else:
                overlap = len(all_dimensions[lang1] & all_dimensions[lang2])
                overlap_matrix[i, j] = overlap
    
    # Save matrix
    np.save(output_dir / "overlap_matrix.npy", overlap_matrix)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Short language names for display
    lang_names = [l.split('_')[0] for l in languages]
    
    sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=lang_names,
        yticklabels=lang_names,
        ax=ax,
        cbar_kws={'label': 'Number of Overlapping Dimensions'}
    )
    
    ax.set_title(f'Overlap Matrix of Top-{top_k} Language-Specific Dimensions\n(Monolingual Setting, Llama2-7B)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "overlap_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print table
    print(f"\n{'='*80}")
    print("Overlap Matrix (Table 1 Format)")
    print(f"{'='*80}")
    print(f"{'':8s}", end='')
    for lang in lang_names:
        print(f"{lang:8s}", end='')
    print()
    
    for i, lang1 in enumerate(lang_names):
        print(f"{lang1:8s}", end='')
        for j in range(len(languages)):
            val = overlap_matrix[i, j]
            if val > 130:  # Bold threshold (like in paper)
                print(f"{val:8d}", end='')
            else:
                print(f"{val:8d}", end='')
        print()
    
    # Save results
    results = {
        "languages": languages,
        "top_k": top_k,
        "overlap_matrix": overlap_matrix.tolist(),
        "num_samples": num_samples
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    return overlap_matrix


def experiment_figure3_topk_selection(
    model, tokenizer, flores, language, output_dir, device="cuda"
):
    """
    Replicate Figure 3: Selecting top-K threshold (parallel setting)
    
    Test different K values: [50, 100, 200, 300, 400, 500, 600]
    """
    print("\n" + "="*80)
    print("FIGURE 3: Top-K Threshold Selection (Parallel Setting)")
    print("="*80)
    
    output_dir = Path(output_dir) / "figure3_topk_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Settings (matching paper Section 5.1)
    k_values = [50, 100, 200, 300, 400, 500, 600]
    final_layer = model.config.num_hidden_layers - 1
    intervention_layer = 19
    alpha = 0.4
    num_train = 50
    num_eval = 100
    
    # Load fastText
    fasttext_model = load_fasttext_model()
    target_lang_code = get_fasttext_lang_code(language)
    
    # Language name
    lang_names = {
        "fra_Latn": "Français", "deu_Latn": "Deutsch", "spa_Latn": "Español",
        "jpn_Jpan": "日本語", "zho_Hans": "中文"
    }
    lang_name = lang_names.get(language, language)
    
    # Load data
    df = flores.load_language_pair("eng_Latn", language, split="devtest")
    english_train = df.head(num_train)[f"sentence_eng_Latn"].tolist()
    target_train = df.head(num_train)[f"sentence_{language}"].tolist()
    
    df_eval = df.iloc[num_train:num_train + num_eval]
    english_eval = df_eval[f"sentence_eng_Latn"].tolist()
    target_eval = df_eval[f"sentence_{language}"].tolist()
    
    results = {"k_values": k_values, "acc": [], "bleu": [], "composite": []}
    
    for k in k_values:
        print(f"\n--- Testing K={k} ---")
        
        # Identify dimensions with this K
        dim_indices, _, mu_en, mu_target = identify_dimensions_parallel(
            model=model,
            tokenizer=tokenizer,
            english_sentences=english_train,
            target_sentences=target_train,
            final_layer=final_layer,
            top_k=k,
            batch_size=4,
            device=device
        )
        
        # Compute steering vector
        steering_vec = compute_steering_vector(mu_en, mu_target, alpha)
        
        # Generate
        generated_texts = []
        prompts = []
        
        for eng in tqdm(english_eval, desc=f"K={k}"):
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
        
        # Evaluate (Figure 3: all 100 samples)
        acc, _, _ = compute_accuracy_fasttext(
            generated_texts, prompts, target_lang_code, fasttext_model, threshold=0.5
        )
        bleu, _ = compute_bleu(generated_texts, prompts, target_eval)
        composite = acc * bleu
        
        results["acc"].append(float(acc))
        results["bleu"].append(float(bleu))
        results["composite"].append(float(composite))
        
        print(f"ACC: {acc:.3f}, BLEU: {bleu:.2f}, Composite: {composite:.2f}")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plot (Figure 3 style)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(results["k_values"], results["acc"], 'o-', color='steelblue', linewidth=2, 
             markersize=8, label='ACC')
    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('ACC', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(results["k_values"], results["bleu"], 's--', color='coral', linewidth=2, 
             markersize=8, label='BLEU')
    ax2.plot(results["k_values"], results["composite"], '^-', color='orange', linewidth=2, 
             markersize=8, label='ACC*BLEU')
    ax2.set_ylabel('BLEU / ACC*BLEU', fontsize=12, color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Add vertical line at K=400 (if in range)
    if 400 in results["k_values"]:
        ax1.axvline(x=400, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Selecting Top-K Threshold ({language}, Parallel Setting, Llama2-7B)', 
              fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / "topk_selection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to {output_dir}")
    return results


def experiment_figure4_layer_selection(
    model, tokenizer, flores, language, output_dir, device="cuda"
):
    """
    Replicate Figure 4: Choosing anchor layer (monolingual setting)
    
    Test different intermediate layers: [8, 12, 16, 20, 24]
    """
    print("\n" + "="*80)
    print("FIGURE 4: Anchor Layer Selection (Monolingual Setting)")
    print("="*80)
    
    output_dir = Path(output_dir) / "figure4_layer_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Settings (matching paper)
    test_layers = [8, 12, 16, 20, 24]
    final_layer = model.config.num_hidden_layers - 1
    intervention_layer = 19
    alpha = 0.4
    top_k = 400
    num_train = 50
    num_eval = 100
    
    # Load fastText
    fasttext_model = load_fasttext_model()
    target_lang_code = get_fasttext_lang_code(language)
    
    # Language name
    lang_names = {
        "fra_Latn": "Français", "deu_Latn": "Deutsch", "spa_Latn": "Español",
        "jpn_Jpan": "日本語", "zho_Hans": "中文"
    }
    lang_name = lang_names.get(language, language)
    
    # Load data
    df = flores.load_language_pair("eng_Latn", language, split="devtest")
    target_train = df.head(num_train)[f"sentence_{language}"].tolist()
    
    df_eval = df.iloc[num_train:num_train + num_eval]
    english_eval = df_eval[f"sentence_eng_Latn"].tolist()
    target_eval = df_eval[f"sentence_{language}"].tolist()
    
    results = {"layers": test_layers, "acc": [], "bleu": [], "composite": []}
    
    for layer in test_layers:
        print(f"\n--- Testing Intermediate Layer={layer} ---")
        
        # Identify dimensions with this layer
        dim_indices, _, mu_inter, mu_final = identify_dimensions_monolingual(
            model=model,
            tokenizer=tokenizer,
            target_sentences=target_train,
            intermediate_layer=layer,
            final_layer=final_layer,
            top_k=top_k,
            batch_size=4,
            device=device
        )
        
        # Compute steering vector
        steering_vec = compute_steering_vector(mu_inter, mu_final, alpha)
        
        # Generate
        generated_texts = []
        prompts = []
        
        for eng in tqdm(english_eval, desc=f"Layer {layer}"):
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
        
        # Evaluate (Figure 4: all 100 samples)
        acc, _, _ = compute_accuracy_fasttext(
            generated_texts, prompts, target_lang_code, fasttext_model, threshold=0.5
        )
        bleu, _ = compute_bleu(generated_texts, prompts, target_eval)
        composite = acc * bleu
        
        results["acc"].append(float(acc))
        results["bleu"].append(float(bleu))
        results["composite"].append(float(composite))
        
        print(f"ACC: {acc:.3f}, BLEU: {bleu:.2f}, Composite: {composite:.2f}")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plot (Figure 4 style)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(results["layers"], results["acc"], 'o-', color='steelblue', linewidth=2, 
             markersize=8, label='ACC')
    ax1.set_xlabel('Layer l', fontsize=12)
    ax1.set_ylabel('ACC', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(results["layers"], results["bleu"], 's--', color='coral', linewidth=2, 
             markersize=8, label='BLEU')
    ax2.plot(results["layers"], results["composite"], '^-', color='orange', linewidth=2, 
             markersize=8, label='ACC*BLEU')
    ax2.set_ylabel('BLEU / ACC*BLEU', fontsize=12, color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Add vertical line at optimal layer
    optimal_idx = np.argmax(results["composite"])
    optimal_layer = results["layers"][optimal_idx]
    ax1.axvline(x=optimal_layer, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Choosing Anchor Layer ({language}, Monolingual Setting, Llama2-7B)', 
              fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_dir / "layer_selection.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to {output_dir}")
    return results


def experiment_figure5_overlap_rate(
    model, tokenizer, flores, languages, output_dir, device="cuda"
):
    """
    Replicate Figure 5: Overlap rate between monolingual and parallel settings
    
    Test different K values: [50, 100, 200, 400]
    """
    print("\n" + "="*80)
    print("FIGURE 5: Monolingual vs Parallel Overlap Rate")
    print("="*80)
    
    output_dir = Path(output_dir) / "figure5_overlap_rate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Settings
    k_values = [50, 100, 200, 400]
    intermediate_layer = 19
    final_layer = model.config.num_hidden_layers - 1
    num_samples = 50
    
    results = {"k_values": k_values, "overlap_rates": []}
    
    for k in k_values:
        print(f"\n--- Testing K={k} ---")
        
        overlaps = []
        
        for lang_code in languages:
            print(f"Processing {lang_code}...")
            
            # Load data
            df = flores.load_language_pair("eng_Latn", lang_code, split="devtest")
            english_sentences = df.head(num_samples)[f"sentence_eng_Latn"].tolist()
            target_sentences = df.head(num_samples)[f"sentence_{lang_code}"].tolist()
            
            # Monolingual dimensions
            dim_mono, _, _, _ = identify_dimensions_monolingual(
                model=model,
                tokenizer=tokenizer,
                target_sentences=target_sentences,
                intermediate_layer=intermediate_layer,
                final_layer=final_layer,
                top_k=k,
                batch_size=4,
                device=device
            )
            
            # Parallel dimensions
            dim_para, _, _, _ = identify_dimensions_parallel(
                model=model,
                tokenizer=tokenizer,
                english_sentences=english_sentences,
                target_sentences=target_sentences,
                final_layer=final_layer,
                top_k=k,
                batch_size=4,
                device=device
            )
            
            # Compute overlap
            overlap = len(set(dim_mono.tolist()) & set(dim_para.tolist()))
            overlap_rate = overlap / k
            overlaps.append(overlap_rate)
            
            print(f"  Overlap: {overlap}/{k} = {overlap_rate:.1%}")
        
        avg_overlap = np.mean(overlaps)
        results["overlap_rates"].append(float(avg_overlap))
        print(f"Average overlap rate for K={k}: {avg_overlap:.1%}")
    
    # Save results
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plot (Figure 5 style)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    overlap_percentages = [r * 100 for r in results["overlap_rates"]]
    
    ax.plot(results["k_values"], overlap_percentages, 'o-', color='steelblue', 
            linewidth=2, markersize=10)
    
    # Annotate points
    for k, rate in zip(results["k_values"], overlap_percentages):
        ax.annotate(f'{rate:.1f}%', 
                   xy=(k, rate), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=10)
    
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Overlap rate', fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    
    plt.title('Overlap Rate of Language-Specific Dimensions\n(Monolingual vs Parallel Settings, Llama2-7B)', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "overlap_rate.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Replicate Paper Results (Table 1, Figures 3-5)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--experiments", nargs='+', 
                       choices=["table1", "figure3", "figure4", "figure5", "all"],
                       default=["all"])
    parser.add_argument("--languages", nargs='+',
                       default=["zho_Hans", "jpn_Jpan", "fra_Latn", "spa_Latn", "deu_Latn"],
                       help="Languages for Table 1")
    parser.add_argument("--test_language", type=str, default="fra_Latn",
                       help="Language for Figures 3 and 4")
    parser.add_argument("--output_dir", type=str, default="results/llama-2-7b-analysis")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hf_token", type=str, default=None)
    
    args = parser.parse_args()
    
    # Get HF token
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Setup
    print("="*80)
    print("PAPER REPLICATION: Table 1, Figures 3, 4, 5")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Experiments: {args.experiments}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    model, tokenizer = setup_model(args.model, args.device, hf_token)
    
    # Load FLORES-200
    print("\nLoading FLORES-200...")
    flores = Flores200Loader(cache_dir="./data/flores200")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    experiments = args.experiments if "all" not in args.experiments else ["table1", "figure3", "figure4", "figure5"]
    
    
    if "table1" in experiments:
        experiment_table1_dimension_overlap(
            model, tokenizer, flores, args.languages, output_dir, args.device
        )
    
    if "figure3" in experiments:
        experiment_figure3_topk_selection(
            model, tokenizer, flores, args.test_language, output_dir, args.device
        )
    
    if "figure4" in experiments:
        experiment_figure4_layer_selection(
            model, tokenizer, flores, args.test_language, output_dir, args.device
        )
    
    if "figure5" in experiments:
        experiment_figure5_overlap_rate(
            model, tokenizer, flores, args.languages, output_dir, args.device
        )
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()