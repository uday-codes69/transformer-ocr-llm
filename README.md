# Transformer-based OCR for Historical Sources

## Architecture Overview

### Why Transformer Architecture?
- **VisionEncoderDecoder (TrOCR)**: Unlike traditional CNN-RNN (CRNN) models, TrOCR uses a Vision Transformer (ViT) as the encoder and a language model as the decoder. This allows the model to understand both visual features and linguistic context simultaneously, which is critical for historical sources where text might be degraded or use archaic spelling.
- **End-to-End**: It simplifies the pipeline by removing the need for complex character-level segmentation.

### Why LLM is Late-Stage?
- **Post-hoc Error Correction**: The OCR model might make locally consistent but globally incorrect errors (e.g., swapping 'f' and 's' in old English).
- **Contextual Awareness**: LLMs like GPT or Llama have vast knowledge of language patterns, allowing them to fix spelling and spacing while preserving the semantic meaning. It is "late-stage" to avoid interfering with the visual recognition process and only act on the textual output.

### Evaluation Metrics
- **CER (Character Error Rate)**: Measures distances at the character level. $CER = (S+D+I)/N$ where S is substitutions, D deletions, I insertions, and N total characters in the reference.
- **WER (Word Error Rate)**: Same calculation as CER but at the word level.

## Project Structure
```
src/
  preprocessing/      # Image cleaning and isolation
  models/             # TrOCR model wrapper
  evaluation/         # CER/WER metrics
  llm_postprocessing/ # LLM refinement logic
  pipeline/           # End-to-end orchestration
inference.py          # CLI for inference
evaluate.py           # Evaluation script
train.py              # Training boilerplate
demo.ipynb            # Interactive demo
requirements.txt      # Dependencies
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run inference: `python inference.py --image path/to/image.jpg --llm`
3. Check evaluation: `python evaluate.py`
