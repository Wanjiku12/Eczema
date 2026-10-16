
from PIL import Image
from pathlib import Path

# Change this to the path of an image you are allowed to use
image_path = Path("sample_skin_image.jpg")

if not image_path.exists():
    print("Image not found.")
    print("Place a permitted sample image in this folder.")
else:
    image = Image.open(image_path)

    print("Image format:", image.format)
    print("Image dimensions:", image.size)
    print("Image mode:", image.mode)

    image.show()
