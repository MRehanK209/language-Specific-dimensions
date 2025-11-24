"""
Configuration file for replicating the Language-Specific Dimensions paper.
Contains model parameters, layer indices, and alpha values from Appendix A.
"""

# Model configurations from the paper
MODEL_CONFIGS = {
    "meta-llama/Llama-2-7b-hf": {
        "name": "Llama2-7B",
        "num_layers": 32,
        "hidden_size": 4096,
        "monolingual": {
            "intervention_layer": 19,
            "alpha": 0.4,
            "intermediate_layer": 19,  # For comparison with final layer
        },
        "parallel": {
            "intervention_layer": 19,
            "alpha": 0.4,
        }
    },
    "meta-llama/Llama-2-13b-hf": {
        "name": "Llama2-13B",
        "num_layers": 40,
        "hidden_size": 5120,
        "monolingual": {
            "intervention_layer": 19,
            "alpha": 0.4,
            "intermediate_layer": 22,  # Anchor layer from paper (Fig 8)
        },
        "parallel": {
            "intervention_layer": 19,
            "alpha": 0.3,
        }
    },
    "meta-llama/Meta-Llama-3.1-8B": {
        "name": "Llama3.1-8B",
        "num_layers": 32,
        "hidden_size": 4096,
        "monolingual": {
            "intervention_layer": 21,
            "alpha": 0.4,
            "intermediate_layer": 21,
        },
        "parallel": {
            "intervention_layer": 27,
            "alpha": 0.8,
        }
    }
}

# Language configurations for FLORES-200
LANGUAGE_CONFIGS = {
    "french": {
        "flores_code": "fra_Latn",
        "language_name": "French",
        "prompt_prefix": "Français"
    },
    "german": {
        "flores_code": "deu_Latn",
        "language_name": "German",
        "prompt_prefix": "Deutsch"
    },
    "spanish": {
        "flores_code": "spa_Latn",
        "language_name": "Spanish",
        "prompt_prefix": "Español"
    },
    "chinese": {
        "flores_code": "zho_Hans",
        "language_name": "Chinese",
        "prompt_prefix": "中文"
    },
    "japanese": {
        "flores_code": "jpn_Jpan",
        "language_name": "Japanese",
        "prompt_prefix": "日本語"
    }
}

# Language names by FLORES code (for prompt generation)
LANGUAGE_NAMES = {
    "fra_Latn": "Français",
    "deu_Latn": "Deutsch",
    "spa_Latn": "Español",
    "zho_Hans": "中文",
    "jpn_Jpan": "日本語"
}

# Short language codes for display
LANGUAGE_SHORT = {
    "fra_Latn": "Fr",
    "deu_Latn": "De",
    "spa_Latn": "Es",
    "zho_Hans": "Zh",
    "jpn_Jpan": "Ja"
}

# English configuration
ENGLISH_FLORES_CODE = "eng_Latn"

# Data configuration
DATA_CONFIG = {
    "dataset_name": "openlanguagedata/flores_plus",
    "dataset_subset": "all",
    "split": "dev",  # Use dev split for dimension identification
    "num_samples": 50,  # Number of sentences for dimension identification
    "seed": 42
}

# Dimension identification configuration
DIMENSION_CONFIG = {
    "top_k": 512,  # Number of dimensions to select
}

# Generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 50,
    "do_sample": False,  # Greedy decoding
    "temperature": 1.0,
    "top_p": 1.0,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "test_split": "devtest",  # Use devtest for evaluation
    "num_test_samples": None,  # Use all available samples
}

# Output directories
OUTPUT_DIRS = {
    "results": "results",
    "dimensions": "dimensions",
    "logs": "logs",
    "checkpoints": "checkpoints"
}

