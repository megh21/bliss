import os
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import PIL.Image
from typing import List

TEMP_DIR = "./tmp/solution_videos"
GOOGLE_API_KEY = "AIzaSyChnsHn3RFTImjAzqFb7OZAw74Hl7HywLk"


def generate_instruction_images(instructions: str, image_path: str) -> List[str]:
    """
    Generate a series of images highlighting different steps from instructions

    Args:
        instructions (str): The text instructions to visualize
        image_path (str): Path to the original image

    Returns:
        List[str]: List of paths to generated images
    """
    # Initialize Google AI client
    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Create prompt
    prompt = f"Hi, can you create a separate picture that is same as I uploaded with a highlighting for each of the following suggested steps in this instruction? The instructions: {instructions}"

    # Load image
    try:
        image = PIL.Image.open(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return []

    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Generate content
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, image],
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
        )
    except Exception as e:
        print(f"Error generating content: {e}")
        return []

    # Process and save generated images
    generated_images = []
    i = 0

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None:
            try:
                image = Image.open(BytesIO(part.inline_data.data))
                output_path = os.path.join(TEMP_DIR, f"answer{i:02d}.png")
                image.save(output_path)
                generated_images.append(output_path)
                i += 1
            except Exception as e:
                print(f"Error saving image {i}: {e}")

    return generated_images
