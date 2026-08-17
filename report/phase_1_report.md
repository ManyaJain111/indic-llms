# Phase 1: Data Collection and Tokenizer Construction

## 1.1 Language Selection
* **Model H (Higher-resource): Hindi**
  * **Justification:** Hindi is a widely spoken Indian language with abundant public textual resources, including extensive Wikipedia articles, open-source Indian datasets (Indic corpora), and government press releases (PIB). This allows for a robust foundation for a higher-resource language model. This aligns perfectly with the requirement for a non-English Indian language that has a relatively abundant public text corpus.
* **Model L (Lower-resource): Nepali**
  * **Justification:** Nepali is explicitly listed as an allowed lower-resource language in the Phase 1 rubric. It presents a unique tokenization challenge due to its script and morphology. Despite being lower-resource, we have a feasible amount of text available through a combination of targeted web scraping (news portals) and specific open-source corpora (e.g., IRIIS).

---

## 1.2 Dataset Requirements

Both models use completely independent monolingual corpora with dedicated train, validation, and test splits (90/5/5 ratio, seed 42). No documents or vocabulary are shared between the two models.

### Model H: Hindi Dataset Statistics
* **Corpus Sources & Token Breakdown:** We gathered the dataset using a mix of public dataset downloads and manual collection (which includes custom web scraping, subtitles, Wikipedia, and PIB).
  * *Downloaded (Public Corpora):*
    * `sangraha_corpus`: 194,198,603 words
    * `indic_corpus`: 176,448,929 words
  * *Manual (Scraped/Extracted):*
    * `hindi_wikipedia`: 42,801,209 words
    * `new_hindi_news_scrape`: 42,383,013 words
    * `hindi_subtitles`: 5,234,233 words
    * `hindi_news_scrape`: 360,205 words
    * `pib_releases`: 5,885 words
* **Cleaning Steps:** The dataset was rigorously cleaned. We applied Unicode NFC normalization to ensure character consistency, exact-content deduplication to remove duplicate articles/sentences, and removed all English characters to maintain absolute Devanagari purity.
* **Requirement Compliance (Tokens & Split):**
  * **Dataset Splits (90/5/5):**
    * *Train:* 410,362,761 words
    * *Validation:* 22,804,967 words
    * *Test:* 22,698,979 words
  * **Total Post-Cleaning Words:** 455,866,707 words *(Note: The raw source collection was 461,432,077 words. Approximately 5.5M words were removed during exact-content deduplication and cleaning).*
  * **Estimated Tokenizer Tokens:** ~549,410,555 tokens *(using fertility rate of 1.2052)*
  * **Status:** The dataset successfully surpasses the 500M token mark. With Wikipedia, Subtitles, and PIB classified as manual collection, the manual fraction is **19.67%** (~90.8M manual words out of ~461.4M raw total words). This strictly falls short of the 20% manual collection requirement.

### Model L: Nepali Dataset Statistics
* **Corpus Sources & Token Breakdown:** The Nepali corpus was constructed to meet the strict manual collection requirement while supplementing with high-quality academic data.
  * *Downloaded (Public Corpora):*
    * `iriis_nepali`: 400,000,000 words
  * *Manual (Scraped/Extracted):*
    * `nepali_news_scrape`: 107,100,815 words (scraped from 43 high-traffic Nepali news portals, adhering to robots.txt).
    * `nepali_wikipedia`: 7,479,341 words
* **Cleaning Steps:** We applied cross-split exact deduplication, whitespace normalization to remove excess gaps, and Unicode NFC normalization. A strict Devanagari purity filter was also applied.
* **Requirement Compliance (Tokens & Split):**
  * **Dataset Splits (90/5/5):**
    * *Train:* 462,910,635 words
    * *Validation:* 25,881,021 words
    * *Test:* 25,788,500 words
  * **Total Post-Cleaning Words:** 514,580,156 words *(Note: The raw source collection was 514,580,156 words).*
  * **Estimated Tokenizer Tokens:** ~618,599,016 tokens *(using fertility rate of 1.2058)*
  * **Status:** The Nepali dataset successfully meets both the 500M total token requirement and the ≥20% manual collection requirement. Specifically, **22.27%** of the raw dataset (114.5M words) was sourced through manual collection, safely exceeding the 20% minimum.

---

## 1.3 Tokenizer Training

Separate tokenizers were trained from scratch for each language. Pretrained tokenizers and shared vocabularies were strictly avoided. The vocabulary sizes (32,000 for Hindi and 37,000 for Nepali) were selected specifically because they achieved an optimal balance: a low fertility rate (~1.205 tokens per word) while ensuring a near-zero unknown-token rate on the held-out test texts.

### Tokenizer Architecture Differences
To better handle the unique properties of our two corpora, we deliberately chose two different BPE architectures:
* **Model H (Hindi) - Byte-Level BPE:** Hindi, our higher-resource language, was scraped from a vast array of diverse, noisy sources (subtitles, news, Wikipedia). Byte-Level BPE was chosen because it breaks characters down to raw UTF-8 bytes. This completely eliminates `<UNK>` tokens (0.00% rate) and prevents vocabulary explosion when encountering rare symbols or emojis in massive datasets.
* **Model L (Nepali) - Unicode Character-Level BPE:** For Nepali, our lower-resource language, we utilized a strict Unicode Character-Level BPE. Because the Nepali corpus is more tightly controlled, operating strictly on valid Devanagari characters ensures that the tokenizer natively preserves the structural integrity of the script. This completely prevents the language model from hallucinating invalid or broken Unicode byte sequences during generation.

### Model H: Hindi Tokenizer Statistics
* **Vocabulary Size:** 32,000
* **Average Characters per Token:** 4.22
* **Average Tokens per Word (Fertility):** 1.205
* **Unknown Tokens / Rate:** 0 (0.00%)
* **Top 5 Token Frequencies:** 
  1. `।` (26,578)
  2. ` के` (26,357)
  3. ` में` (20,000)
  4. ` है` (18,975)
  5. `,` (18,000)
* **Tokenization Example:**
  * *Text:* मैं बंदूक नकलनेवल हूँ।
  * *Token IDs:* `[1458, 8944, 9138, 313, 1016, 2094, 284]`

### Model L: Nepali Tokenizer Statistics
* **Vocabulary Size:** 37,000
* **Average Characters per Token:** 5.38
* **Average Tokens per Word (Fertility):** 1.205
* **Unknown Tokens / Rate:** 7 (~0.00011%)
* **Top 5 Token Frequencies:** 
  1. ` ।` (222,735)
  2. `,` (143,279)
  3. `।` (99,363)
  4. ` छ` (94,568)
  5. ` र` (89,894)
* **Tokenization Example:**
  * *Text:* कश्मीरी का अर्थ हो सकता हैः
  * *Token IDs:* `[273, 1986, 9232, 334, 1430, 347, 732, 6632]`
