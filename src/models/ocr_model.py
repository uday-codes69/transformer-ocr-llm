from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

class TRocrModel:
    """
    Wrapper for HuggingFace TrOCR Model.
    """
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        print(f"Model loaded on {self.device}")

    def recognize(self, image: Image.Image):
        """
        Performs OCR on the given image.
        """
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
        
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text

if __name__ == "__main__":
    print("TRocrModel module loaded.")
