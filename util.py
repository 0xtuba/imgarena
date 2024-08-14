import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import adapter
import db

load_dotenv()


def get_all_models(prompt: str):
    sdxl = adapter.SDXLLightning(prompt)
    sd1 = adapter.StableDiffusion(prompt)
    sd3 = adapter.StableDiffusion3(prompt)
    fp = adapter.FluxPro(prompt)
    fs = adapter.FluxSchnell(prompt)

    return [sdxl, sd1, sd3, fp, fs]


def generate_prompts(num_prompts: int):
    class ModelResponse(BaseModel):
        results: list[str]

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": f"Generate {num_prompts} different descriptive prompts for my image generational model. Just respond with the prompt description itself, not any titles or prefixes. The prompt should be between 30 to 50 words each.",
            },
        ],
        model="gpt-4o-2024-08-06",
        response_format=ModelResponse,
    )

    message = completion.choices[0].message
    if message.parsed:
        return message.parsed.results


def save_prompts():
    prompts = generate_prompts(10)
    for prompt in prompts:
        print(prompt)
        db.write_prompt(prompt)


def generate_images():
    prompts = db.read_prompts()

    for prompt in prompts:
        text = prompt["text"]
        models = get_all_models(text)
        for model in models:
            image_url = model.generate_image()
            filename, res = db.upload_image_to_bucket(image_url=image_url)
            print("Uploaded file to bucket:", res)
            img_bucket_url = f"https://yycelpiurkvyijumsxcw.supabase.co/storage/v1/object/public/images_bucket/{filename}.webp"
            r = db.write_image(
                prompt_id=prompt["id"],
                model_id=model.id,
                image_url=img_bucket_url,
                image_id=filename,
            )
            print("Write image to db", r.data)
