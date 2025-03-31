# Bliss Hackathon 2025 

## Overview

This project aims to generate solution videos based on user-provided problem descriptions and images. It leverages various services, including:

- **PDF Service:**  Indexes and retrieves information from PDF documents.
- **Image Summarization Service:** Summarizes the key elements of an image.
- **Answer Service:** Generates step-by-step solutions to problems.
- **Image Generation Service:** Creates images illustrating the solution steps.
- **Video Service:** Combines the generated images into a video.

## Setup

1.  **Create a conda environment:**

    ```bash
    conda env create -f env.yaml
    conda activate bliss
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the FastAPI application:**

    ```bash
    python main.py
    ```

2.  **Access the API endpoints:**

    -   `/generate-solution`:  Generates a solution video based on a problem description and image.
    -   `/health`:  Performs a health check.

## API Details

### `/generate-solution`

**Method:** POST

**Description:** Generates a solution video.

**Request Body (Form Data):**

-   `problem_text` (string):  Description of the problem.
-   `problem_image` (file):  Image related to the problem.

**Response:**

-   Returns a video file (`solution.mp4`) with the solution.

## Modules

### `src/pdf_service.py`

-   Provides functionality to process PDF documents, create embeddings, and perform similarity searches.
-   Uses `faiss` for efficient vector storage and retrieval.

### `src/summarise_image.py`

-   Summarizes images using the Gemini API.
-   Extracts key elements and information from the image.

### `src/answer_service.py`

-   Generates step-by-step solutions to problems based on a PDF document and image description.
-   Uses the Gemini API to generate the solution steps.

### `src/image_service.py`

-   Generates images for each solution step, highlighting the relevant parts of the original image.
-   Uses the Gemini API for image generation.

### `src/video_service.py`

-   Creates a video from a sequence of images.
-   Currently a placeholder; implementation needs to be added.

### `main.py`

-   Defines the FastAPI application and API endpoints.
-   Orchestrates the different services to generate the solution video.

## Configuration

-   `TEMP_DIR`:  Temporary directory for storing intermediate files (images, etc.).  Defaults to `tmp/solution_videos`.
-   Google Cloud Project ID and Region are hardcoded in the source files.

## Notes

-   The `create_video` function in `src/video_service.py` is currently a placeholder and needs to be implemented.
-   Error handling and logging are implemented throughout the application.
-   Temporary files are cleaned up after a delay using an asynchronous background task.