from google import genai
from google.genai import types

# from google.cloud import documentai_v1beta3 as documentai
import base64


def summarize_image(
    image_path, user_query="", project_id="bliss-hack25fra-9578", location="us-central1"
):
    """
    Summarizes an image using Gemini.

    Args:
      image_path: Path to the image file.
      user_query: Text prompt to guide the summarization.
      project_id: Your Google Cloud project ID.
      location: The location of the Vertex AI endpoint.

    Returns:
      The summarized text from Gemini.
    """

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    pre_prompt = "Summarize the key elements and information conveyed in this image. "  # Modified prompt

    # Read file content - changed for images
    with open(image_path, "rb") as f:
        image_data = f.read()

    text_query = types.Part.from_text(text=pre_prompt + user_query)

    # Encode Image to base64 - changed for images
    base64_image = base64.b64encode(image_data).decode("utf-8")

    # Determine image MIME type (e.g., image/jpeg, image/png)
    import imghdr

    mime_type = f"image/{imghdr.what(image_path)}"  # Determine mime type dynamically
    if not mime_type:
        raise ValueError(
            "Could not determine image type.  Ensure it's a valid image format."
        )

    # Create the content with inlineData
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data={
                        "mime_type": mime_type,  # e.g., "image/jpeg"
                        "data": base64_image,
                    }
                ),
                text_query,
            ],
        )
    ]

    model = "gemini-2.5-pro-exp-03-25"  # Or whichever model you're using
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        top_p=1,
        seed=0,
        max_output_tokens=8192,
        response_modalities=["TEXT"],
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
            ),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
        ),
    )

    output = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        output += chunk.text
    return output
