import logging
import random

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import UUID4, BaseModel

import db
import util

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@app.get("/categories")
async def categories():
    return db.get_prompt_categories()


@app.get("/prompt")
async def choose_prompt(
    category: str = Query(default="random", description="Category of the prompt"),
):
    prompts = db.read_prompts(category=category)
    if not prompts:
        raise HTTPException(
            status_code=404, detail=f"No prompts found for category: {category}"
        )

    prompt = random.choice(prompts)
    images = db.read_images(prompt_id=prompt.id)

    # Select 4 random images or all if there are fewer than 4
    selected_images = random.sample(images, min(4, len(images)))
    res = {
        "prompt_id": str(prompt.id),
        "prompt_category": prompt.category,
        "prompt": prompt.text,
        "images": [
            {
                "image_id": str(image.id),
                "image_url": image.image_url,
            }
            for image in selected_images
        ],
    }
    return res


@app.get("/leaderboard")
async def leaderboard():
    leaderboard = db.get_leaderboard("random")
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
        image_ids = [
            str(selection.image1_id),
            str(selection.image2_id),
            str(selection.image3_id),
            str(selection.image4_id),
        ]

        # Get model ratings from image IDs
        ratings = db.get_model_ratings_from_image_ids(image_ids)

        # Find the winner index
        winner_index = image_ids.index(str(selection.winner_id))

        updated_ratings = util.update_ratings(winner_index, ratings)
        logger.info(
            f"Updated ratings: winner_index={winner_index}, ratings={ratings}, updated_ratings={updated_ratings}"
        )
        prompt = db.read_prompts(id=selection.prompt_id)
        db.bulk_write_rankings(updated_ratings, prompt.category)

        return {
            "winner_model": ratings[winner_index][1],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
