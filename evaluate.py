from src.evaluation.metrics import Metrics

def main():
    # Example ground truth and predictions
    ground_truth = "In the year 1745, historical records show..."
    raw_ocr = "In the year 1745, historica records show..."
    refined_ocr = "In the year 1745, historical records show..."

    metrics = Metrics()
    
    # Raw stats
    raw_cer = metrics.calculate_cer(ground_truth, raw_ocr)
    raw_wer = metrics.calculate_wer(ground_truth, raw_ocr)
    
    # Refined stats
    ref_cer = metrics.calculate_cer(ground_truth, refined_ocr)
    ref_wer = metrics.calculate_wer(ground_truth, refined_ocr)

    print("Evaluation Results:")
    print(f"{'Metric':<10} | {'Raw OCR':<10} | {'Refined OCR':<10}")
    print("-" * 35)
    print(f"{'CER':<10} | {raw_cer:<10.4f} | {ref_cer:<10.4f}")
    print(f"{'WER':<10} | {raw_wer:<10.4f} | {ref_wer:<10.4f}")

if __name__ == "__main__":
    main()
