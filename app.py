import random

from fastapi import FastAPI, HTTPException
from pydantic import UUID4, BaseModel

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
    return {
        "prompt_id": str(prompt.id),
        "prompt": prompt.text,
        "images": [
            {
                "image_id": str(image.id),
                "image_url": image.image_url,
            }
            for image in images
        ],
    }


class FavoriteSelection(BaseModel):
    prompt_id: UUID4
    image1_id: UUID4
    image2_id: UUID4
    image3_id: UUID4
    image4_id: UUID4
    winner_id: UUID4


@app.post("/select-favorite")
async def select_favorite(selection: FavoriteSelection):
    try:
        db.write_comparison(
            str(selection.prompt_id),
            str(selection.image1_id),
            str(selection.image2_id),
            str(selection.image3_id),
            str(selection.image4_id),
            str(selection.winner_id),
        )
        return {"message": "Comparison recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
