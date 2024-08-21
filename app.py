import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import UUID4, BaseModel

import db
import util

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://imgarena-fe.vercel.app",
        "https://www.imgenarena.ai",
        "https://imgenarena.ai",
    ],  # Allows CORS for localhost:3000
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/prompt")
async def choose_prompt():
    prompts = db.read_prompts()
    prompt = random.choice(prompts)

    images = db.read_images(prompt_id=prompt.id)

    # Select 4 random images or all if there are fewer than 4
    selected_images = random.sample(images, min(4, len(images)))
    return {
        "prompt_id": str(prompt.id),
        "prompt": prompt.text,
        "images": [
            {
                "image_id": str(image.id),
                "image_url": util.transform_img_url(str(image.image_url)),
            }
            for image in selected_images
        ],
    }


@app.get("/leaderboard")
async def leaderboard():
    leaderboard = db.get_leaderboard()
    return leaderboard


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

        choices = [
            selection.image1_id,
            selection.image2_id,
            selection.image3_id,
            selection.image4_id,
        ]
        winner_index = choices.index(selection.winner_id)
        ratings = db.get_model_ratings_from_image_ids(choices)
        updated_ratings = util.update_ratings(winner_index, ratings)
        db.bulk_write_rankings(updated_ratings)

        winner_model = ratings[winner_index][1]

        return {
            "winner_model": winner_model,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
