import editdistance
from jiwer import cer, wer

class Metrics:
    """
    Calculates OCR evaluation metrics.
    """
    
    @staticmethod
    def calculate_cer(reference, hypothesis):
        """
        Character Error Rate: (Substitutions + Deletions + Insertions) / Total Characters in Reference
        """
        if len(reference) == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        return cer(reference, hypothesis)

    @staticmethod
    def calculate_wer(reference, hypothesis):
        """
        Word Error Rate: (Substitutions + Deletions + Insertions) / Total Words in Reference
        """
        if len(reference.split()) == 0:
            return 1.0 if len(hypothesis.split()) > 0 else 0.0
        return wer(reference, hypothesis)

if __name__ == "__main__":
    # Test
    ref = "Historically OCR was hard."
    hyp = "Historical OCR was hard."
    print(f"CER: {Metrics.calculate_cer(ref, hyp):.4f}")
    print(f"WER: {Metrics.calculate_wer(ref, hyp):.4f}")
