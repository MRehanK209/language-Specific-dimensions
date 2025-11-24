#!/usr/bin/env python3
"""Direct FLORES-200 dataset loader (no HF datasets script required)"""

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


_LANGUAGES = [
    "ace_Arab", "bam_Latn", "dzo_Tibt", "hin_Deva", "khm_Khmr", "mag_Deva", "pap_Latn", "sot_Latn", "tur_Latn",
    "ace_Latn", "ban_Latn", "ell_Grek", "hne_Deva", "kik_Latn", "mai_Deva", "pbt_Arab", "spa_Latn", "twi_Latn",
    "acm_Arab", "bel_Cyrl", "eng_Latn", "hrv_Latn", "kin_Latn", "mal_Mlym", "pes_Arab", "srd_Latn", "tzm_Tfng",
    "acq_Arab", "bem_Latn", "epo_Latn", "hun_Latn", "kir_Cyrl", "mar_Deva", "plt_Latn", "srp_Cyrl", "uig_Arab",
    "aeb_Arab", "ben_Beng", "est_Latn", "hye_Armn", "kmb_Latn", "min_Arab", "pol_Latn", "ssw_Latn", "ukr_Cyrl",
    "afr_Latn", "bho_Deva", "eus_Latn", "ibo_Latn", "kmr_Latn", "min_Latn", "por_Latn", "sun_Latn", "umb_Latn",
    "ajp_Arab", "bjn_Arab", "ewe_Latn", "ilo_Latn", "knc_Arab", "mkd_Cyrl", "prs_Arab", "swe_Latn", "urd_Arab",
    "aka_Latn", "bjn_Latn", "fao_Latn", "ind_Latn", "knc_Latn", "mlt_Latn", "quy_Latn", "swh_Latn", "uzn_Latn",
    "als_Latn", "bod_Tibt", "fij_Latn", "isl_Latn", "kon_Latn", "mni_Beng", "ron_Latn", "szl_Latn", "vec_Latn",
    "amh_Ethi", "bos_Latn", "fin_Latn", "ita_Latn", "kor_Hang", "mos_Latn", "run_Latn", "tam_Taml", "vie_Latn",
    "apc_Arab", "bug_Latn", "fon_Latn", "jav_Latn", "lao_Laoo", "mri_Latn", "rus_Cyrl", "taq_Latn", "war_Latn",
    "arb_Arab", "bul_Cyrl", "fra_Latn", "jpn_Jpan", "lij_Latn", "mya_Mymr", "sag_Latn", "taq_Tfng", "wol_Latn",
    "arb_Latn", "cat_Latn", "fur_Latn", "kab_Latn", "lim_Latn", "nld_Latn", "san_Deva", "tat_Cyrl", "xho_Latn",
    "ars_Arab", "ceb_Latn", "fuv_Latn", "kac_Latn", "lin_Latn", "nno_Latn", "sat_Olck", "tel_Telu", "ydd_Hebr",
    "ary_Arab", "ces_Latn", "gaz_Latn", "kam_Latn", "lit_Latn", "nob_Latn", "scn_Latn", "tgk_Cyrl", "yor_Latn",
    "arz_Arab", "cjk_Latn", "gla_Latn", "kan_Knda", "lmo_Latn", "npi_Deva", "shn_Mymr", "tgl_Latn", "yue_Hant",
    "asm_Beng", "ckb_Arab", "gle_Latn", "kas_Arab", "ltg_Latn", "nso_Latn", "sin_Sinh", "tha_Thai", "zho_Hans",
    "ast_Latn", "crh_Latn", "glg_Latn", "kas_Deva", "ltz_Latn", "nus_Latn", "slk_Latn", "tir_Ethi", "zho_Hant",
    "awa_Deva", "cym_Latn", "grn_Latn", "kat_Geor", "lua_Latn", "nya_Latn", "slv_Latn", "tpi_Latn", "zsm_Latn",
    "ayr_Latn", "dan_Latn", "guj_Gujr", "kaz_Cyrl", "lug_Latn", "oci_Latn", "smo_Latn", "tsn_Latn", "zul_Latn",
    "azb_Arab", "deu_Latn", "hat_Latn", "kbp_Latn", "luo_Latn", "ory_Orya", "sna_Latn", "tso_Latn",
    "azj_Latn", "dik_Latn", "hau_Latn", "kea_Latn", "lus_Latn", "pag_Latn", "snd_Arab", "tuk_Latn",
    "bak_Cyrl", "dyu_Latn", "heb_Hebr", "khk_Cyrl", "lvs_Latn", "pan_Guru", "som_Latn", "tum_Latn"
]

_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
_SPLITS = ["dev", "devtest"]


class Flores200Loader:
    """Simple FLORES-200 dataset loader"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the loader
        
        Args:
            cache_dir: Directory to cache the downloaded dataset. 
                      Defaults to ~/.cache/flores200
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/flores200")
        self.cache_dir = Path(cache_dir)
        self.data_dir = self.cache_dir / "flores200_dataset"
        
    def download_and_extract(self, force: bool = False):
        """Download and extract the FLORES-200 dataset
        
        Args:
            force: If True, re-download even if already cached
        """
        if self.data_dir.exists() and not force:
            print(f"Dataset already exists at {self.data_dir}")
            return
            
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.cache_dir / "flores200_dataset.tar.gz"
        
        if not tar_path.exists() or force:
            print(f"Downloading FLORES-200 from {_URL}...")
            urllib.request.urlretrieve(_URL, tar_path)
            print(f"Downloaded to {tar_path}")
        
        print(f"Extracting to {self.cache_dir}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.cache_dir)
        print("Extraction complete!")
        
    def load_language(self, lang: str, split: str = "dev") -> pd.DataFrame:
        """Load a single language split
        
        Args:
            lang: Language code (e.g., 'eng_Latn', 'fra_Latn')
            split: Split to load ('dev' or 'devtest')
            
        Returns:
            DataFrame with columns: id, sentence, URL, domain, topic, has_image, has_hyperlink
        """
        if lang not in _LANGUAGES:
            raise ValueError(f"Language {lang} not in FLORES-200. Available: {_LANGUAGES}")
        if split not in _SPLITS:
            raise ValueError(f"Split {split} not available. Choose from: {_SPLITS}")
            
        # Ensure data is downloaded
        self.download_and_extract()
        
        # Load sentences
        sentence_file = self.data_dir / split / f"{lang}.{split}"
        with open(sentence_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f]
        
        # Load metadata
        metadata_file = self.data_dir / f"metadata_{split}.tsv"
        metadata_df = pd.read_csv(metadata_file, sep="\t")
        
        # Combine
        df = pd.DataFrame({
            "id": range(1, len(sentences) + 1),
            "sentence": sentences,
            "URL": metadata_df["URL"],
            "domain": metadata_df["domain"],
            "topic": metadata_df["topic"],
            "has_image": (metadata_df["has_image"] == "yes").astype(int),
            "has_hyperlink": (metadata_df["has_hyperlink"] == "yes").astype(int)
        })
        
        return df
    
    def load_language_pair(self, lang1: str, lang2: str, split: str = "dev") -> pd.DataFrame:
        """Load a parallel corpus for two languages
        
        Args:
            lang1: First language code
            lang2: Second language code
            split: Split to load ('dev' or 'devtest')
            
        Returns:
            DataFrame with columns: id, sentence_lang1, sentence_lang2, URL, domain, topic, has_image, has_hyperlink
        """
        df1 = self.load_language(lang1, split)
        df2 = self.load_language(lang2, split)
        
        # Merge on ID and metadata
        df = pd.DataFrame({
            "id": df1["id"],
            f"sentence_{lang1}": df1["sentence"],
            f"sentence_{lang2}": df2["sentence"],
            "URL": df1["URL"],
            "domain": df1["domain"],
            "topic": df1["topic"],
            "has_image": df1["has_image"],
            "has_hyperlink": df1["has_hyperlink"]
        })
        
        return df
    
    def load_all_languages(self, split: str = "dev") -> pd.DataFrame:
        """Load all languages for a split
        
        Args:
            split: Split to load ('dev' or 'devtest')
            
        Returns:
            DataFrame with columns: id, sentence_<lang> for each language, URL, domain, topic, has_image, has_hyperlink
        """
        print(f"Loading all {len(_LANGUAGES)} languages for split '{split}'...")
        
        # Ensure data is downloaded
        self.download_and_extract()
        
        # Load metadata
        metadata_file = self.data_dir / f"metadata_{split}.tsv"
        metadata_df = pd.read_csv(metadata_file, sep="\t")
        
        # Start with metadata
        data = {
            "id": range(1, len(metadata_df) + 1),
            "URL": metadata_df["URL"],
            "domain": metadata_df["domain"],
            "topic": metadata_df["topic"],
            "has_image": (metadata_df["has_image"] == "yes").astype(int),
            "has_hyperlink": (metadata_df["has_hyperlink"] == "yes").astype(int)
        }
        
        # Add each language
        for i, lang in enumerate(_LANGUAGES):
            sentence_file = self.data_dir / split / f"{lang}.{split}"
            with open(sentence_file, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f]
            data[f"sentence_{lang}"] = sentences
            
            if (i + 1) % 50 == 0:
                print(f"Loaded {i + 1}/{len(_LANGUAGES)} languages...")
        
        print("All languages loaded!")
        return pd.DataFrame(data)
    
    def get_available_languages(self) -> List[str]:
        """Get list of all available language codes"""
        return _LANGUAGES.copy()


def main():
    """Example usage"""
    loader = Flores200Loader()
    
    # Example 1: Load English dev set
    print("=" * 60)
    print("Example 1: Load English dev set")
    print("=" * 60)
    df_en = loader.load_language("eng_Latn", split="dev")
    print(f"Loaded {len(df_en)} sentences")
    print(df_en.head())
    print()
    
    # Example 2: Load English-French parallel corpus
    print("=" * 60)
    print("Example 2: Load English-French parallel corpus")
    print("=" * 60)
    df_en_fr = loader.load_language_pair("eng_Latn", "fra_Latn", split="dev")
    print(f"Loaded {len(df_en_fr)} sentence pairs")
    print(df_en_fr.head())
    print()
    
    # Example 3: Show available languages
    print("=" * 60)
    print(f"Total languages available: {len(loader.get_available_languages())}")
    print("First 10 languages:", loader.get_available_languages()[:10])


if __name__ == "__main__":
    main()

