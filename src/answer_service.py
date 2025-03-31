from google import genai
from google.genai import types
from src.summarise_image import summarize_image
import base64


def get_problem_solution(
    image_description,
    user_query="My technivolt 1100 is not working. Why is my car not charging?",
    pdf_path="files/BA_DE_TECHNIVOLT_11+22_20220714_2238001000100_web.pdf",
):
    client = genai.Client(
        vertexai=True,
        project="bliss-hack25fra-9578",
        location="us-central1",
    )

    # pdf_path = "/home/seb/hackathon/ProduktAssets/ProduktAssets/TechniVolt/Dokumente/BDA/BA_DE_TECHNIVOLT_11+22_20220714_2238001000100_web.pdf"
    pre_prompt = "Can you give step by step solution to the following problem based on the provided document? This is the user problem: "
    image_preprompt = " And a description of their device: "

    # Read file content
    with open(pdf_path, "rb") as f:
        document_data = f.read()

    prompt = f"{pre_prompt}{user_query}{image_preprompt}{image_description} Please separate each step by a newline character and dont use them anywhere else."

    text_query = types.Part.from_text(text=prompt)

    # Encode PDF to base64
    base64_pdf = base64.b64encode(document_data).decode("utf-8")

    # Create the content with inlineData
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    inline_data={"mime_type": "application/pdf", "data": base64_pdf}
                ),
                text_query,
            ],
        )
    ]

    model = "gemini-2.5-pro-exp-03-25"
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
            # enable_thinking=,
            include_thoughts=True,
            # thinking_budget=,
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


image_description = summarize_image(
    image_path="problem_image.png",
    user_query="",
)
output = get_problem_solution(image_description=image_description)
print(output)
