from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs
from pathlib import Path
import time


class ImageProcessor:
    def __init__(self, options: VlmPipelineOptions = None):
        pipeline_options = options if options else VlmPipelineOptions(
            vlm_options=vlm_model_specs.GRANITE_VISION_OLLAMA,  # <-- change the model here
            enable_remote_services=True
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        self.path: str = '../pdf_as_images'

    def process_image(self, image_name: str):
        source = f"{self.path}/{image_name}"
        doc = self.converter.convert(source=source).document
        return doc.export_to_markdown()

def img2markdown_main():
    # Main execution
    processor = ImageProcessor()

    # Get all subfolders
    main_folder = Path("../pdf_as_images")
    subfolders = sorted([d for d in main_folder.iterdir() if d.is_dir()])

    # Concatenate all text
    all_text = ""

    for subfolder in subfolders:
        print(f"\n{'='*80}")
        print(f"Processing subfolder: {subfolder.name}")
        print(f"{'='*80}\n")

        # Get all images from subfolder
        images = sorted([f.name for f in subfolder.iterdir() if f.suffix in ['.png', '.jpg', '.jpeg']])

        start_time = time.time()
        for image_name in images:
            print(f"Processing: {subfolder.name}/{image_name}")

            # Update processor path to current subfolder
            processor.path = str(subfolder)

            text = processor.process_image(image_name)
            all_text += text + "\n\n"

            # Delete image after processing
            image_path = subfolder / image_name
            image_path.unlink()
            print(f"Deleted: {image_name}\n")

        end_time = time.time()
        print(f"Total time to process {subfolder.name}: {end_time - start_time}")
        print(f"All text: {all_text}\n")

        # Save concatenated text - Optional
        with open(f"./{subfolder.name}.md", "w") as f:
            f.write(all_text)

    print("All images processed and concatenated!")

# if __name__ == "__main__":
#     img2markdown_main()