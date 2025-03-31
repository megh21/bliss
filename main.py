from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import logging
import asyncio
from typing import List
import uuid
from pydantic import BaseModel
import uvicorn

# Custom modules for different services
from src.pdf_service import PDFVectorStore
from src.summarise_image import summarize_image
from src.answer_service import get_problem_solution
from src.image_service import generate_instruction_images
from src.video_service import create_video

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Solution Video Generator")

# Configuration
TEMP_DIR = "tmp/solution_videos"

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


class SolutionRequest(BaseModel):
    query_text: str
    pdf_url: str
    steps: List[str]


@app.post("/generate-solution")
async def generate_solution(
    problem_text: str = Form(...),
    problem_image: UploadFile = File(...),
):
    """
    Generate a solution video based on a problem description and image.

    1. Receive problem text and image
    2. Query PDF service to find relevant PDF
    3. Generate solution steps using answer service
    4. Generate images for each step
    5. Create a video from the images
    6. Return the video file
    """
    # Create a unique session ID for this request
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    vector_store = PDFVectorStore()
    vector_store.load()
    try:
        # Save the uploaded image
        image_path = os.path.join(session_dir, "problem_image.jpg")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(problem_image.file, buffer)

        logger.info(f"Saved problem image to {image_path}")

        image_description = summarize_image(
            image_path="problem_image.png",
            user_query="",
        )
        # Step 1: Find relevant PDF
        updated_problem_text = (
            f"{problem_text} the image description is as below {image_description}"
        )
        pdf_url = vector_store.query(text_query=updated_problem_text)

        # Step 2: Generate solution steps
        solution_steps = get_problem_solution(
            image_description=image_description, pdf_url=pdf_url[0]
        )
        if not solution_steps:
            raise HTTPException(
                status_code=500, detail="Failed to generate solution steps"
            )

        steps = solution_steps["steps"]
        logger.info(f"Generated {len(steps)} solution steps")

        # Step 3: Generate images for each step
        image_paths = generate_instruction_images(steps, image_path, session_dir)
        if not image_paths or len(image_paths) == 0:
            raise HTTPException(
                status_code=500, detail="Failed to generate step images"
            )

        logger.info(f"Generated {len(image_paths)} step images")

        # Step 4: Create video from images
        video_path = create_video(image_paths, session_dir)
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(
                status_code=500, detail="Failed to create solution video"
            )

        logger.info(f"Created solution video: {video_path}")

        # Return the video file
        return FileResponse(
            path=video_path, filename="solution.mp4", media_type="video/mp4"
        )

    except Exception as e:
        logger.error(f"Error generating solution: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating solution: {str(e)}"
        )

    finally:
        # Clean up will be handled by a background task
        asyncio.create_task(cleanup_session_files(session_dir))


# def generate_step_images(
#     steps: List[str], original_image_path: str, output_dir: str
# ) -> List[str]:
#     """
#     Generate images for each solution step using the image_generator module.
#     """
#     image_paths = []

#     for i, step in enumerate(steps):
#         # Call the image generator function
#         image_data = generate_instruction_images(step, original_image_path, i + 1)

#         if not image_data:
#             logger.error(f"Image generation failed for step {i + 1}")
#             continue

#         # Save the generated image
#         image_filename = f"step_{i + 1}.jpg"
#         image_path = os.path.join(output_dir, image_filename)

#         with open(image_path, "wb") as f:
#             f.write(image_data)

#         image_paths.append(image_path)

#     return image_paths


# # def create_video(image_paths: List[str], output_dir: str) -> str:
#     """
#     Create a video from the solution step images using FFmpeg.
#     """
#     output_path = os.path.join(output_dir, "solution.mp4")

#     # Create a temporary file with the list of images
#     list_file = os.path.join(output_dir, "images.txt")
#     with open(list_file, "w") as f:
#         for path in image_paths:
#             f.write(f"file '{path}'\n")
#             f.write("duration 3\n")  # Each image stays for 3 seconds

#     # Use FFmpeg to create the video
#     cmd = [
#         "ffmpeg",
#         "-y",  # Overwrite output file if it exists
#         "-f",
#         "concat",
#         "-safe",
#         "0",
#         "-i",
#         list_file,
#         "-c:v",
#         "libx264",
#         "-pix_fmt",
#         "yuv420p",
#         "-r",
#         "30",  # 30 fps
#         output_path,
#     ]

#     try:
#         process = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         return output_path

#     except subprocess.CalledProcessError as e:
#         logger.error(f"FFmpeg error: {e.stderr}")
#         return None

#     except Exception as e:
#         logger.error(f"Error creating video: {str(e)}")
#         return None


async def cleanup_session_files(session_dir: str):
    """
    Clean up temporary files after a delay.
    """
    try:
        # Wait for 1 hour before cleaning up
        await asyncio.sleep(3600)

        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned up session directory: {session_dir}")

    except Exception as e:
        logger.error(f"Error cleaning up session files: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
