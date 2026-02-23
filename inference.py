import argparse
from src.pipeline.ocr_pipeline import OCRPipeline

def main():
    parser = argparse.ArgumentParser(description="Run OCR Inference on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM post-processing.")
    args = parser.parse_args()

    pipeline = OCRPipeline(llm_enabled=args.llm)
    results = pipeline.run(args.image)

    print("\n" + "="*20)
    print("OCR RESULTS")
    print("="*20)
    print(f"Raw OCR: {results['raw_text']}")
    if args.llm:
        print(f"Refined OCR: {results['refined_text']}")

if __name__ == "__main__":
    main()
