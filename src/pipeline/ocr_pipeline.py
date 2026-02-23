from src.preprocessing.image_processor import ImagePreprocessor
from src.models.ocr_model import TRocrModel
from src.llm_postprocessing.refiner import LLMRefiner

class OCRPipeline:
    """
    End-to-end OCR Pipeline.
    """
    def __init__(self, llm_enabled=True):
        self.preprocessor = ImagePreprocessor()
        self.ocr_model = TRocrModel()
        self.llm_refiner = LLMRefiner() if llm_enabled else None

    def run(self, image_path):
        # 1. Preprocess
        processed_image = self.preprocessor.preprocess(image_path)
        
        # 2. OCR Inference
        raw_text = self.ocr_model.recognize(processed_image)
        
        # 3. LLM Refinement
        refined_text = raw_text
        if self.llm_refiner:
            refined_text = self.llm_refiner.refine(raw_text)
            
        return {
            "raw_text": raw_text,
            "refined_text": refined_text
        }

if __name__ == "__main__":
    print("OCRPipeline module loaded.")
