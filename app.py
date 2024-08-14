import random

from fastapi import FastAPI

import db

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/prompt")
async def choose_prompt():
    prompts = db.read_prompts()
    prompt = random.choice(prompts)

    images = db.read_images(prompt_id=prompt.id)
    return {"prompt": prompt.text, "images": [image.image_url for image in images]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
