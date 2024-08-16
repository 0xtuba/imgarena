import concurrent.futures
import logging
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
    fd = adapter.FluxDev(prompt)
    k22 = adapter.Kandinsky22(prompt)
    d3 = adapter.DALLE3(prompt)

    return [sdxl, sd1, sd3, fp, fs, fd, k22, d3]


def save_models():
    models = get_all_models("")
    for model in models:
        db.write_model(model.id, model.name, model.description)


def generate_prompts(num_prompts: int):
    class ModelResponse(BaseModel):
        results: list[str]

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=OPENAI_API_KEY)

    completion = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": f"Generate {num_prompts} different descriptive prompts for my image generational model. Just respond with the prompt description itself, not any titles or prefixes. The prompt should be between 30 to 50 words each. Use a varation of fictional and non-fictional prompts, including scenery, portraits, real-life scenes, humans, objects, and animals. Do not number them, just give me the prompts.",
            },
        ],
        model="gpt-4o-2024-08-06",
        response_format=ModelResponse,
    )

    message = completion.choices[0].message
    if message.parsed:
        return message.parsed.results


def save_prompts():
    prompts = generate_prompts(100)
    for prompt in prompts:
        print(prompt)
        db.write_prompt(prompt)


def process_model(
    prompt, model, prompt_index, model_index, total_prompts, total_models
):
    try:
        logging.info(
            f"Processing prompt {prompt_index + 1}/{total_prompts}, "
            f"model {model_index + 1}/{total_models}: {model.__class__.__name__}"
        )

        image_url = model.generate_image()
        filename, res = db.upload_image_to_bucket(image_url=image_url)
        logging.info(f"Uploaded file to bucket: {res}")

        img_bucket_url = f"https://yycelpiurkvyijumsxcw.supabase.co/storage/v1/object/public/images_bucket/{filename}.webp"
        r = db.write_image(
            prompt_id=str(prompt.id),
            model_id=model.id,
            image_url=img_bucket_url,
            image_id=filename,
        )
        logging.info(f"Write image to db: {r}")
    except Exception as e:
        logging.error(
            f"Error processing model {model.__class__.__name__} for prompt {prompt_index + 1}: {str(e)}"
        )


def generate_images():
    logging.basicConfig(
        filename="image_generation_2.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # prompts = db.read_prompts()
    prompts = db.fetch_unprocessed_prompts()

    max_workers = 8

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, prompt in enumerate(prompts):
            logging.info(
                f"Processing prompt {i + 1}/{len(prompts)}: {prompt.text[:50]}..."
            )

            models = get_all_models(prompt.text)
            for j, model in enumerate(models):
                futures.append(
                    executor.submit(
                        process_model, prompt, model, i, j, len(prompts), len(models)
                    )
                )

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    logging.info("Image generation completed.")


def update_elo(winner_rating, loser_rating, k_factor=32, win_loss_multiplier=1):
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    winner_new_rating = (
        winner_rating + k_factor * (1 - expected_winner) * win_loss_multiplier
    )
    loser_new_rating = (
        loser_rating + k_factor * (0 - (1 - expected_winner)) / win_loss_multiplier
    )
    return winner_new_rating, loser_new_rating


def update_ratings(selected_index, model_ratings, k_factor=32):
    """
    Update ratings based on the selected winner.

    Args:
    selected_index (int): Index of the winning model in the model_ratings list.
    model_ratings (List[Tuple[str, float]]): List of tuples (model_id, rating).
    k_factor (int): The K-factor for Elo rating updates.

    Returns:
    List[Tuple[str, float]]: Updated list of tuples (model_id, new_rating).
    """
    winner_model, winner_name, winner_rating = model_ratings[selected_index]
    updated_ratings = []

    for i, (model, _, rating) in enumerate(model_ratings):
        if i != selected_index:
            new_winner_rating, loser_rating = update_elo(
                winner_rating, rating, k_factor
            )
            updated_ratings.append((model, loser_rating))
            winner_rating = new_winner_rating
        else:
            updated_ratings.append((model, winner_rating))

    # Update the winner's rating in the result
    updated_ratings[selected_index] = (winner_model, winner_rating)

    return updated_ratings


def initialize_ratings():
    models = db.read_models()
    for model in models:
        db.write_ranking(model_id=model.id, elo_score=1000)


def fix():
    import re

    pattern = r"^\d+\.\s"
    prompts = db.read_prompts()
    counter = 0
    problems = []
    for prompt in prompts:
        if re.match(pattern, prompt.text):
            counter += 1
            problems.append(prompt)

    print(counter)
