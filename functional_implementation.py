"""
Functional Implementation of Language-Specific Dimensions
For Notebook Exploration and Analysis

Based on paper: "Language Lives in Sparse Dimensions" (Zhong et al., 2024)
https://arxiv.org/pdf/2510.07213

This module provides functional (non-class) implementations of:
- Section 3.2.1: Identifying Language-Specific Dimensions (Monolingual)
- Section 3.2.2: Identifying Language-Specific Dimensions (Parallel)
- Intervention mechanism
- Visualization and analysis tools
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json
from tqdm.auto import tqdm
import sacrebleu

# ==============================================================================
# Section 3.2.1: Identifying Language-Specific Dimensions (Monolingual Setting)
# ==============================================================================

def extract_hidden_states(
    model,
    tokenizer,
    sentences: List[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Extract hidden states from a specific layer for a list of sentences.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sentences: List of sentences
        layer_idx: Layer index to extract from
        batch_size: Batch size for processing
        device: Device to run on
    
    Returns:
        Tensor of shape [num_sentences, hidden_dim] containing mean-pooled hidden states
    """
    all_hidden_states = []
    
    model.eval()
    # Only move model if it's not already dispatched with accelerate
    if not hasattr(model, 'hf_device_map'):
        model.to(device)
    
    # Get the actual device for inputs (first device if using accelerate)
    if hasattr(model, 'hf_device_map'):
        # When using accelerate, get the device of the first layer
        input_device = next(model.parameters()).device
    else:
        input_device = device
    
    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Extracting layer {layer_idx}"):
        batch = sentences[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(input_device)
        
        # Extract hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_idx]  # [batch_size, seq_len, hidden_dim]
            
            # Mean pool over sequence length (excluding padding)
            attention_mask = inputs["attention_mask"].unsqueeze(-1)  # [batch_size, seq_len, 1]
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)  # [batch_size, hidden_dim]
            num_tokens = attention_mask.sum(dim=1)  # [batch_size, 1]
            mean_hidden = sum_hidden / num_tokens  # [batch_size, hidden_dim]
            
            # Keep on GPU for faster downstream operations (mean, topk, etc.)
            all_hidden_states.append(mean_hidden)
    
    # Concatenate all batches (stays on GPU for dimension identification)
    return torch.cat(all_hidden_states, dim=0)


def compute_corpus_mean(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute mean representation across all sentences.
    
    Args:
        hidden_states: Tensor of shape [num_sentences, hidden_dim]
    
    Returns:
        Mean tensor of shape [hidden_dim]
    """
    return hidden_states.mean(dim=0)


def identify_dimensions_monolingual(
    model,
    tokenizer,
    target_sentences: List[str],
    intermediate_layer: int,
    final_layer: int,
    top_k: int = 512,
    batch_size: int = 8,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Identify language-specific dimensions using monolingual setting.
    
    Implementation of Section 3.2.1 from the paper:
    1. Extract representations at intermediate layer ℓ and final layer L
    2. Compute mean: μ_ℓ and μ_L
    3. Calculate: δ_diff = |μ_L - μ_ℓ|
    4. Select top-K dimensions with largest differences
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        target_sentences: List of target language sentences (e.g., 50 sentences)
        intermediate_layer: Intermediate layer index ℓ
        final_layer: Final layer index L
        top_k: Number of dimensions to select (default: 512)
        batch_size: Batch size for extraction
        device: Device to run on
    
    Returns:
        Tuple of:
        - dimension_indices: Indices of top-K dimensions [top_k]
        - delta_diff: Difference vector [hidden_dim]
        - mu_intermediate: Mean at intermediate layer [hidden_dim]
        - mu_final: Mean at final layer [hidden_dim]
    """
    print(f"\n{'='*70}")
    print(f"MONOLINGUAL DIMENSION IDENTIFICATION")
    print(f"{'='*70}")
    print(f"Number of sentences: {len(target_sentences)}")
    print(f"Intermediate layer: {intermediate_layer}")
    print(f"Final layer: {final_layer}")
    print(f"Top-K dimensions: {top_k}")
    
    # Step 1: Extract representations at intermediate layer ℓ
    print(f"\nStep 1: Extracting representations at intermediate layer {intermediate_layer}...")
    hidden_intermediate = extract_hidden_states(
        model, tokenizer, target_sentences, intermediate_layer, batch_size, device
    )
    
    # Step 2: Extract representations at final layer L
    print(f"\nStep 2: Extracting representations at final layer {final_layer}...")
    hidden_final = extract_hidden_states(
        model, tokenizer, target_sentences, final_layer, batch_size, device
    )
    
    # Step 3: Compute mean representations
    print(f"\nStep 3: Computing mean representations...")
    mu_intermediate = compute_corpus_mean(hidden_intermediate)  # μ_ℓ
    mu_final = compute_corpus_mean(hidden_final)  # μ_L
    
    # Step 4: Calculate absolute difference (Equation 1 from paper)
    print(f"\nStep 4: Computing δ_diff = |μ_L - μ_ℓ|...")
    delta_diff = torch.abs(mu_final - mu_intermediate)
    
    # Step 5: Select top-K dimensions (Equation 2 from paper)
    print(f"\nStep 5: Selecting top-{top_k} dimensions...")
    top_k_values, top_k_indices = torch.topk(delta_diff, k=top_k)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Hidden dimension size: {len(delta_diff)}")
    print(f"Top-K selected: {top_k}")
    print(f"Top 10 dimension indices: {top_k_indices[:10].tolist()}")
    print(f"Top 10 difference values: {[f'{v:.4f}' for v in top_k_values[:10].tolist()]}")
    print(f"Mean difference (top-K): {top_k_values.mean().item():.6f}")
    print(f"Min difference (top-K): {top_k_values.min().item():.6f}")
    print(f"Max difference (top-K): {top_k_values.max().item():.6f}")
    print(f"{'='*70}\n")
    
    return top_k_indices, delta_diff, mu_intermediate, mu_final


# ==============================================================================
# Section 3.2.2: Identifying Language-Specific Dimensions (Parallel Setting)
# ==============================================================================

def identify_dimensions_parallel(
    model,
    tokenizer,
    english_sentences: List[str],
    target_sentences: List[str],
    final_layer: int,
    top_k: int = 512,
    batch_size: int = 8,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Identify language-specific dimensions using parallel setting.
    
    Implementation of Section 3.2.2 from the paper:
    1. Extract final layer representations for English and target language
    2. Compute means: μ_L^(en) and μ_L^(target)
    3. Calculate: δ_diff = |μ_L^(target) - μ_L^(en)|
    4. Select top-K dimensions with largest differences
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        english_sentences: List of English sentences
        target_sentences: List of target language sentences (parallel to English)
        final_layer: Final layer index L
        top_k: Number of dimensions to select (default: 512)
        batch_size: Batch size for extraction
        device: Device to run on
    
    Returns:
        Tuple of:
        - dimension_indices: Indices of top-K dimensions [top_k]
        - delta_diff: Difference vector [hidden_dim]
        - mu_english: Mean English representation [hidden_dim]
        - mu_target: Mean target representation [hidden_dim]
    """
    print(f"\n{'='*70}")
    print(f"PARALLEL DIMENSION IDENTIFICATION")
    print(f"{'='*70}")
    print(f"Number of parallel pairs: {len(english_sentences)}")
    print(f"Final layer: {final_layer}")
    print(f"Top-K dimensions: {top_k}")
    
    # Step 1: Extract English representations at final layer
    print(f"\nStep 1: Extracting English representations at layer {final_layer}...")
    hidden_english = extract_hidden_states(
        model, tokenizer, english_sentences, final_layer, batch_size, device
    )
    
    # Step 2: Extract target language representations at final layer
    print(f"\nStep 2: Extracting target language representations at layer {final_layer}...")
    hidden_target = extract_hidden_states(
        model, tokenizer, target_sentences, final_layer, batch_size, device
    )
    
    # Step 3: Compute mean representations
    print(f"\nStep 3: Computing mean representations...")
    mu_english = compute_corpus_mean(hidden_english)  # μ_L^(en)
    mu_target = compute_corpus_mean(hidden_target)  # μ_L^(target)
    
    # Step 4: Calculate absolute difference
    print(f"\nStep 4: Computing δ_diff = |μ_L^(target) - μ_L^(en)|...")
    delta_diff = torch.abs(mu_target - mu_english)
    
    # Step 5: Select top-K dimensions
    print(f"\nStep 5: Selecting top-{top_k} dimensions...")
    top_k_values, top_k_indices = torch.topk(delta_diff, k=top_k)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Hidden dimension size: {len(delta_diff)}")
    print(f"Top-K selected: {top_k}")
    print(f"Top 10 dimension indices: {top_k_indices[:10].tolist()}")
    print(f"Top 10 difference values: {[f'{v:.4f}' for v in top_k_values[:10].tolist()]}")
    print(f"Mean difference (top-K): {top_k_values.mean().item():.6f}")
    print(f"Min difference (top-K): {top_k_values.min().item():.6f}")
    print(f"Max difference (top-K): {top_k_values.max().item():.6f}")
    print(f"{'='*70}\n")
    
    return top_k_indices, delta_diff, mu_english, mu_target


# ==============================================================================
# Steering Vector Computation
# ==============================================================================

def compute_steering_vector(
    mu_source: torch.Tensor,
    mu_target: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Compute steering vector for intervention.
    
    Paper's Equation 3: h'_j[i] = α × μ^(lang)_L[i] for i ∈ I^(lang)_K
    
    steering_vector = α × μ_target
    
    Args:
        mu_source: Source representation (English or intermediate layer) - not used in intervention
        mu_target: Target representation (target language or final layer)
        alpha: Scaling coefficient
    
    Returns:
        Steering vector [hidden_dim]
    """
    return alpha * mu_target


# ==============================================================================
# Intervention Mechanism
# ==============================================================================

def apply_intervention(
    model,
    tokenizer,
    prompt: str,
    dimension_indices: torch.Tensor,
    steering_vector: torch.Tensor,
    intervention_layer: int,
    max_new_tokens: int = 50,
    device: str = "cuda",
    **generation_kwargs
) -> str:
    """
    Generate text with intervention on language-specific dimensions.
    
    Paper's exact prompt format: English:"[sentence]"-[Language]:"[prefix]"
    Example: English:"Today is hot."-日本語:"今日は"
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt (use paper's format!)
        dimension_indices: Indices of language-specific dimensions
        steering_vector: Steering vector to apply
        intervention_layer: Layer to apply intervention
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        **generation_kwargs: Additional generation parameters
    
    Returns:
        Generated text
    """
    # Only move model if it's not already dispatched with accelerate
    if not hasattr(model, 'hf_device_map'):
        model.to(device)
    
    # Get model's actual device (more robust than using device parameter)
    model_device = next(model.parameters()).device
    
    # Move tensors to model's device (if not already there)
    if steering_vector.device != model_device:
        steering_vector = steering_vector.to(model_device)
    if dimension_indices.device != model_device:
        dimension_indices = dimension_indices.to(model_device)
    
    # Create intervention hook
    hook_handle = None
    
    def intervention_hook(module, input, output):
        """Hook function to modify hidden states at the intervention layer"""
        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = None
        
        # Apply intervention to last token position at this layer
        last_token_hidden = hidden_states[:, -1, :].clone()
        
        # Overwrite language-specific dimensions with steering vector
        # (as per paper: "overwrite these language-specific dimensions")
        last_token_hidden[:, dimension_indices] = steering_vector[dimension_indices]
        hidden_states[:, -1, :] = last_token_hidden
        
        if other_outputs is not None:
            return (hidden_states,) + other_outputs
        else:
            return hidden_states
    
    # Register hook
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[intervention_layer]
    elif hasattr(model, 'layers'):
        target_layer = model.layers[intervention_layer]
    else:
        raise ValueError("Could not find model layers")
    
    hook_handle = target_layer.register_forward_hook(intervention_hook)
    
    try:
        # Tokenize and move to model's device
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
        
        # Decode ONLY the newly generated tokens (not the prompt!)
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up: stop at first newline to avoid continuation
        # The model sometimes continues generating beyond the translation
        if '\n' in generated_text:
            generated_text = generated_text.split('\n')[0].strip()
        
        # Also stop if model starts repeating the prompt pattern
        stop_phrases = ['Translate an English', 'English:', 'Target language:', 'Français:', 'Deutsch:', 'Español:', '中文:', '日本語:']
        for phrase in stop_phrases:
            if phrase in generated_text:
                generated_text = generated_text.split(phrase)[0].strip()
                break
        
    finally:
        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()
    
    return generated_text


def generate_without_intervention(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cuda",
    **generation_kwargs
) -> str:
    """Generate text without intervention (baseline)."""
    # Only move model if it's not already dispatched with accelerate
    if not hasattr(model, 'hf_device_map'):
        model.to(device)
    
    # Get model's actual device
    model_device = next(model.parameters()).device
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **generation_kwargs
        )
    
    # Decode ONLY the newly generated tokens (not the prompt!)
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up: stop at first newline to avoid continuation
    if '\n' in generated_text:
        generated_text = generated_text.split('\n')[0].strip()
    
    # Also stop if model starts repeating the prompt pattern
    stop_phrases = ['Translate an English', 'English:', 'Target language:', 'Français:', 'Deutsch:', 'Español:', '中文:', '日本語:']
    for phrase in stop_phrases:
        if phrase in generated_text:
            generated_text = generated_text.split(phrase)[0].strip()
            break
    
    return generated_text


# ==============================================================================
# Evaluation Functions
# ==============================================================================

def detect_language(text: str) -> str:
    """Detect language of text using langdetect."""
    try:
        return detect(text)
    except:
        return "unknown"


def compute_accuracy(
    generated_texts: List[str],
    prompts: List[str],
    target_lang_code: str
) -> Tuple[float, List[bool]]:
    """
    Compute language-specific accuracy (ACC).
    
    ACC = percentage of outputs in target language
    
    Args:
        generated_texts: List of generated texts
        prompts: List of prompts (to extract generated part)
        target_lang_code: Target language code (e.g., 'fr', 'de', 'ja')
    
    Returns:
        Tuple of (accuracy, list of success indicators)
    """
    successes = []
    
    for full_text, prompt in zip(generated_texts, prompts):
        # Extract generated part (remove prompt)
        if full_text.startswith(prompt):
            generated = full_text[len(prompt):]
        else:
            generated = full_text
        
        generated = generated.strip().split("'")[0] if "'" in generated else generated.strip()
        
        # Detect language
        detected = detect_language(generated)
        is_correct = (detected == target_lang_code)
        successes.append(is_correct)
    
    accuracy = sum(successes) / len(successes) if successes else 0.0
    return accuracy, successes


def compute_bleu(
    generated_texts: List[str],
    prompts: List[str],
    references: List[str]
) -> Tuple[float, List[float]]:
    """
    Compute BLEU scores.
    
    Args:
        generated_texts: List of generated texts (WITHOUT prompts - only generated part!)
        prompts: List of prompts (NOT USED anymore - kept for API compatibility)
        references: List of reference translations
    
    Returns:
        Tuple of (average BLEU, list of individual BLEU scores)
    """
    individual_bleus = []
    
    for generated, reference in zip(generated_texts, references):
        # The generated_texts are now ALREADY extracted (no prompt included!)
        # Just clean up the text
        generated = generated.strip()
        
        # Remove quotes if present
        if generated.startswith('"') or generated.startswith("'"):
            quote_char = generated[0]
            end_quote = generated.find(quote_char, 1)
            if end_quote > 0:
                generated = generated[1:end_quote]
        
        # Take only the first sentence/line
        lines = generated.split('\n')
        generated = lines[0].strip()
        
        # Compute BLEU
        bleu = sacrebleu.sentence_bleu(
            generated,
            [reference],
            smooth_method='exp',
            smooth_value=0.1
        )
        individual_bleus.append(bleu.score)
    
    avg_bleu = sum(individual_bleus) / len(individual_bleus) if individual_bleus else 0.0
    return avg_bleu, individual_bleus


# ==============================================================================
# Visualization Functions (for Figures 3, 4, 5)
# ==============================================================================

def plot_dimension_distribution(
    delta_diff: torch.Tensor,
    dimension_indices: torch.Tensor,
    title: str = "Language-Specific Dimension Distribution",
    save_path: Optional[str] = None
):
    """
    Plot distribution of differences across dimensions (Figure 3 style).
    
    Args:
        delta_diff: Difference vector for all dimensions
        dimension_indices: Indices of selected dimensions
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Histogram of all differences
    ax1.hist(delta_diff.numpy(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(delta_diff[dimension_indices].min().item(), color='red', linestyle='--', 
                label=f'Top-K threshold ({len(dimension_indices)} dims)')
    ax1.set_xlabel('|Δ| (Absolute Difference)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Dimension Differences', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Top-K values
    top_k_values = delta_diff[dimension_indices].numpy()
    ax2.plot(range(len(top_k_values)), sorted(top_k_values, reverse=True), 
             linewidth=2, color='coral')
    ax2.set_xlabel('Dimension Rank', fontsize=12)
    ax2.set_ylabel('|Δ| Value', fontsize=12)
    ax2.set_title(f'Top-{len(dimension_indices)} Dimension Values', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_layer_comparison(
    results_by_layer: Dict[int, Dict],
    metric: str = "accuracy",
    title: str = "Performance vs Intervention Layer",
    save_path: Optional[str] = None
):
    """
    Plot performance across different intervention layers (Figure 4 style).
    
    Args:
        results_by_layer: Dict mapping layer_idx -> results dict
        metric: Metric to plot ('accuracy', 'bleu', 'composite')
        title: Plot title
        save_path: Path to save figure
    """
    layers = sorted(results_by_layer.keys())
    values = [results_by_layer[layer][metric] for layer in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, values, marker='o', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Intervention Layer', fontsize=12)
    plt.ylabel(metric.upper() if metric != "composite" else "ACC × BLEU", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_alpha_sweep(
    results_by_alpha: Dict[float, Dict],
    metric: str = "accuracy",
    title: str = "Performance vs Alpha (α)",
    save_path: Optional[str] = None
):
    """
    Plot performance across different alpha values (Figure 5 style).
    
    Args:
        results_by_alpha: Dict mapping alpha -> results dict
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
    """
    alphas = sorted(results_by_alpha.keys())
    values = [results_by_alpha[alpha][metric] for alpha in alphas]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, values, marker='s', linewidth=2, markersize=8, color='coral')
    plt.xlabel('Alpha (α)', fontsize=12)
    plt.ylabel(metric.upper() if metric != "composite" else "ACC × BLEU", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def create_results_table(
    results: Dict[str, Dict[str, Dict]],
    save_path: Optional[str] = None
) -> str:
    """
    Create Table 1 style results table.
    
    Args:
        results: Nested dict: {model: {language: {setting: metrics}}}
        save_path: Path to save table
    
    Returns:
        Formatted table string
    """
    import pandas as pd
    
    rows = []
    for model, lang_results in results.items():
        for language, setting_results in lang_results.items():
            row = {"Model": model, "Language": language}
            
            for setting in ["monolingual", "parallel"]:
                if setting in setting_results:
                    metrics = setting_results[setting]
                    prefix = "Mono" if setting == "monolingual" else "Para"
                    row[f"{prefix} ACC"] = f"{metrics['accuracy']:.3f}"
                    row[f"{prefix} BLEU"] = f"{metrics['bleu']:.1f}"
                    row[f"{prefix} ACC×BLEU"] = f"{metrics['composite']:.1f}"
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    table_str = df.to_markdown(index=False)
    
    print(table_str)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table_str)
        print(f"\nTable saved to {save_path}")
    
    return table_str


# ==============================================================================
# Helper Functions
# ==============================================================================

def format_prompt_paper_style(
    english_sentence: str,
    target_language_name: str
) -> str:
    """
    Format prompt according to paper's Section 4.3 (Multilingual Generation Control).
    
    Paper Section 4.3:
    "Translate an English sentence into a target language. English: {source text} Target language:."
    
    Format: "Translate an English sentence into a target language. English: {source} {TargetLang}:"
    
    Args:
        english_sentence: Source English sentence
        target_language_name: Target language name (e.g., "Français", "日本語", "Deutsch")
    
    Returns:
        Formatted prompt string
 
    """
    return f"Translate an English sentence into a target language. English: {english_sentence} {target_language_name}:"


def save_dimensions(
    dimension_indices: torch.Tensor,
    steering_vector: torch.Tensor,
    save_path: str,
    metadata: Dict = None
):
    """Save identified dimensions and steering vector."""
    save_dict = {
        "dimension_indices": dimension_indices.cpu().tolist(),
        "steering_vector": steering_vector.cpu().tolist(),
    }
    
    if metadata:
        save_dict.update(metadata)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)
    
    print(f"Dimensions saved to {save_path}")


def load_dimensions(load_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Load previously identified dimensions."""
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    dimension_indices = torch.tensor(data["dimension_indices"])
    steering_vector = torch.tensor(data["steering_vector"])
    
    metadata = {k: v for k, v in data.items() 
               if k not in ["dimension_indices", "steering_vector"]}
    
    return dimension_indices, steering_vector, metadata


if __name__ == "__main__":
    print("Language-Specific Dimensions - Functional Implementation")
    print("="*70)
    print("\nThis module provides functional implementations for:")
    print("- Section 3.2.1: Monolingual dimension identification")
    print("- Section 3.2.2: Parallel dimension identification")
    print("- Intervention mechanism")
    print("- Evaluation (ACC, BLEU)")
    print("- Visualization (Figures 3, 4, 5)")
    print("- Results table (Table 1)")
    print("\nImport this module in a notebook for interactive analysis!")

