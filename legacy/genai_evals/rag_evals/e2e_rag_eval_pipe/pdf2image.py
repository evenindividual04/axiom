import glob, pymupdf
from dotenv import load_dotenv, find_dotenv
import logging
import asyncio
import os

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pdf2ImageGenerator:
    def __init__(self):
        self.dest_path = "../pdf_as_images"
    async def get_images_from_pdf(self, path: str = '../../datasets/'):
        # To get better resolution
        zoom_x = 2.0  # horizontal zoom
        zoom_y = 2.0  # vertical zoom
        mat = pymupdf.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

        all_files = glob.glob(path + "*.pdf")
        file_pointer = 0
        for filename in all_files:
            doc = pymupdf.open(filename)  # open document
            os.makedirs(f"{self.dest_path}/file_{file_pointer}", exist_ok=True)
            logger.info(f"saving to {self.dest_path}/file_{file_pointer}")
            for page in doc:  # iterate through the pages
                pix = page.get_pixmap(matrix=mat)  # render page to an image
                pix.save(f"{self.dest_path}/file_{file_pointer}/page-%i.png" % page.number)  # store image as a PNG
            file_pointer += 1

    async def process_document(self, path: str = '../../datasets/'):
        await self.get_images_from_pdf(path=path)
        logger.info(f"Parsed docs in {path} successfully")


def pdf2image_main():
    """Main function to run document parser."""
    parser = Pdf2ImageGenerator()
    # Create destination directory if it doesn't exist
    os.makedirs(parser.dest_path, exist_ok=True)
    # Run the async process_document method
    asyncio.run(parser.process_document())


# if __name__ == "__main__":
#     pdf2image_main()