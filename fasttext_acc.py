"""
Accurate ACC calculation using fastText as specified in the paper.

Paper quote: "For calculating accuracy, we use the fastText (Joulin et al., 2017) 
language identification classifier to detect the output language. A sample is 
counted as successful if the classifier's score exceeds a threshold of 0.5"
"""

import fasttext
from typing import List, Tuple
import os
from pathlib import Path
import urllib.request


def download_fasttext_model(model_path: str = "./models/lid.176.bin") -> str:
    """
    Download fastText language identification model.
    
    Args:
        model_path: Path to save the model
    
    Returns:
        Path to downloaded model
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_path.exists():
        print(f"fastText model already exists at {model_path}")
        return str(model_path)
    
    print("Downloading fastText language identification model...")
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    
    urllib.request.urlretrieve(url, model_path)
    print(f"Model downloaded to {model_path}")
    
    return str(model_path)


def load_fasttext_model(model_path: str = "./models/lid.176.bin"):
    """
    Load fastText language identification model.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        fastText model
    """
    if not os.path.exists(model_path):
        model_path = download_fasttext_model(model_path)
    
    # Suppress fastText warnings
    model = fasttext.load_model(model_path)
    return model


def detect_language_fasttext(
    text: str,
    model,
    threshold: float = 0.5
) -> Tuple[str, float, bool]:
    """
    Detect language using fastText with confidence threshold.
    
    As per paper: "A sample is counted as successful if the classifier's 
    score exceeds a threshold of 0.5"
    
    Args:
        text: Input text to detect language
        model: fastText model
        threshold: Confidence threshold (default: 0.5 from paper)
    
    Returns:
        Tuple of (language_code, confidence_score, is_confident)
    """
    # Preprocess: fastText expects text without newlines
    text = text.replace('\n', ' ').strip()
    
    if not text:
        return "unknown", 0.0, False
    
    try:
        # Predict language
        predictions = model.predict(text, k=1)
        
        # Extract language code and confidence
        lang_label = predictions[0][0]  # e.g., '__label__en'
        # Handle both numpy array and scalar
        if hasattr(predictions[1], '__getitem__'):
            confidence = float(predictions[1][0])
        else:
            confidence = float(predictions[1])
        
        # Remove '__label__' prefix
        lang_code = lang_label.replace('__label__', '')
        
        # Check if confidence exceeds threshold
        is_confident = confidence >= threshold
        
        return lang_code, confidence, is_confident
    
    except Exception as e:
        print(f"Warning: fasttext prediction failed: {e}")
        return "unknown", 0.0, False


def compute_accuracy_fasttext(
    generated_texts: List[str],
    prompts: List[str],
    target_lang_code: str,
    model,
    threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[float, List[bool], List[float]]:
    """
    Compute ACC (accuracy) using fastText as specified in the paper.
    
    Implementation of paper's ACC metric:
    "A sample is counted as successful if the classifier's score exceeds 
    a threshold of 0.5"
    
    Args:
        generated_texts: List of generated texts (WITHOUT prompts - only generated part!)
        prompts: List of input prompts (NOT USED anymore - kept for API compatibility)
        target_lang_code: Target language code (e.g., 'fr', 'de', 'ja')
        model: fastText model
        threshold: Confidence threshold (default: 0.5)
        verbose: If True, print debug information
    
    Returns:
        Tuple of (accuracy, list of success indicators, list of confidence scores)
    """
    successes = []
    confidences = []
    
    for idx, generated in enumerate(generated_texts):
        # The generated_texts are now ALREADY extracted (no prompt included!)
        # Just clean up the text
        generated = generated.strip()
        
        # Remove quotes if present
        if generated.startswith('"') or generated.startswith("'"):
            # Find the closing quote
            quote_char = generated[0]
            end_quote = generated.find(quote_char, 1)
            if end_quote > 0:
                generated = generated[1:end_quote]
        
        # Take only the first sentence/line (stop at newline or period followed by space)
        # This prevents detecting text from prompt that might be repeated
        lines = generated.split('\n')
        generated = lines[0].strip()
        
        # If still empty or very short, it's a failure
        if len(generated.strip()) < 3:
            if verbose and idx < 3:
                print(f"\n[Sample {idx}] TOO SHORT")
                print(f"  Generated: '{generated}'")
            successes.append(False)
            confidences.append(0.0)
            continue
        
        # Detect language with fastText
        detected_lang, confidence, is_confident = detect_language_fasttext(
            generated, model, threshold
        )
        
        # Check if detected language matches target AND confidence > threshold
        is_correct = (detected_lang == target_lang_code) and is_confident
        
        if verbose and idx < 3:  # Show first 3 samples
            print(f"\n[Sample {idx}]")
            print(f"  Generated: '{generated[:100]}...'")
            print(f"  Detected: {detected_lang} (conf: {confidence:.3f})")
            print(f"  Target: {target_lang_code}")
            print(f"  Match: {is_correct}")
        
        successes.append(is_correct)
        confidences.append(confidence)
    
    # Calculate accuracy
    accuracy = sum(successes) / len(successes) if successes else 0.0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Total samples: {len(successes)}")
        print(f"Successes: {sum(successes)}")
        print(f"ACC: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"Mean confidence: {sum(confidences)/len(confidences):.3f}")
        print(f"{'='*60}")
    
    return accuracy, successes, confidences


# Language code mappings (FLORES-200 to fastText)
FLORES_TO_FASTTEXT = {
    "fra_Latn": "fr",
    "deu_Latn": "de", 
    "spa_Latn": "es",
    "jpn_Jpan": "ja",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
    "eng_Latn": "en",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "ara_Arab": "ar",
    "hin_Deva": "hi",
    "kor_Hang": "ko",
    "tur_Latn": "tr",
    "vie_Latn": "vi",
    "pol_Latn": "pl",
    "ukr_Cyrl": "uk",
    "nld_Latn": "nl",
    "ell_Grek": "el",
    "ces_Latn": "cs",
    "swe_Latn": "sv",
    "ron_Latn": "ro",
    "dan_Latn": "da",
    "fin_Latn": "fi",
    "nor_Latn": "no",
    "ind_Latn": "id",
    "tha_Thai": "th",
    "cat_Latn": "ca",
    "hrv_Latn": "hr",
    "heb_Hebr": "he",
}


def get_fasttext_lang_code(flores_code: str) -> str:
    """
    Convert FLORES-200 language code to fastText language code.
    
    Args:
        flores_code: FLORES-200 code (e.g., 'fra_Latn')
    
    Returns:
        fastText language code (e.g., 'fr')
    """
    return FLORES_TO_FASTTEXT.get(flores_code, flores_code.split('_')[0][:2])


if __name__ == "__main__":
    print("fastText-based ACC Calculator")
    print("="*70)
    print("\nThis matches the paper's methodology:")
    print("'For calculating accuracy, we use the fastText (Joulin et al., 2017)")
    print("language identification classifier to detect the output language.")
    print("A sample is counted as successful if the classifier's score exceeds")
    print("a threshold of 0.5'")
    print("\n" + "="*70)
    
    # Test
    print("\nDownloading/loading fastText model...")
    model = load_fasttext_model()
    print("✓ Model loaded!\n")
    
    # Test language detection
    test_texts = {
        "fr": "Bonjour, comment allez-vous aujourd'hui?",
        "de": "Guten Tag, wie geht es Ihnen heute?",
        "es": "Hola, ¿cómo estás hoy?",
        "ja": "こんにちは、今日はどうですか？",
        "zh": "你好，你今天怎么样？",
        "en": "Hello, how are you today?"
    }
    
    print("Testing language detection:")
    print("-" * 70)
    for expected_lang, text in test_texts.items():
        detected, confidence, is_confident = detect_language_fasttext(
            text, model, threshold=0.5
        )
        status = "✓" if (detected == expected_lang and is_confident) else "✗"
        print(f"{status} Expected: {expected_lang}, Detected: {detected}, "
              f"Confidence: {confidence:.3f}, Passes threshold: {is_confident}")
    
    print("\n" + "="*70)
    print("✓ fastText ACC calculator ready to use!")

