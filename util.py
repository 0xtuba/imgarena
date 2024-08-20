import concurrent.futures
import csv
import logging
import os
from collections import defaultdict
from datetime import datetime
from uuid import UUID

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

import adapter
import db
from db import Prompt

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
    mj = adapter.Midjourney(prompt)

    return [sdxl, sd1, sd3, fp, fs, fd, k22, d3, mj]


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


def update_elo(winner_rating, loser_rating, k_factor=32, win_loss_multiplier=1.1):
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


def start_mj_jobs():
    model = adapter.Midjourney("")
    logging.basicConfig(
        filename="image_generation.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Read the CSV file
    unprocessed_prompts = []
    with open("output.csv", "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not any(row[f"img{i}"] for i in range(1, 5)):
                unprocessed_prompts.append({"id": row["ID"], "text": row["Text"]})

    total_prompts = len(unprocessed_prompts)
    responses = []

    for i, prompt in enumerate(unprocessed_prompts):
        logging.info(
            f"Processing prompt {i + 1}/{total_prompts}: {prompt['text'][:50]}..."
        )
        model.prompt = prompt["text"]
        try:
            r = process_model(prompt, model, i, 0, total_prompts, 1)
            if r.status_code == 200:
                responses.append(r["data"])
            else:
                logging.error(f"Error processing prompt {i + 1}: {str(r)}")
        except Exception as e:
            logging.error(f"Error processing prompt {i + 1}: {str(e)}")


def write_prompts_to_csv():
    import csv

    prompts = db.read_prompts()

    with open("prompts.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Text"])  # Header row

        for prompt in prompts:
            writer.writerow([prompt.id, prompt.text])

    print(f"Successfully wrote {len(prompts)} prompts to prompts.csv")


def check_missing(model_id, filename):
    with open(filename, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    total_prompts = len(rows)
    column_name = f"{model_id}"
    missing_prompts = [
        Prompt(
            id=UUID(row["prompt_id"]),
            text=row["prompt_text"],
            created_at=datetime.now(),  # Note: Using current time as creation time
        )
        for row in rows
        if not row.get(column_name)
    ]

    print(f"Total prompts: {total_prompts}")
    print(f"Prompts with missing {model_id} images: {len(missing_prompts)}")

    return missing_prompts


def fill_missing_columns(missing_prompts, model_class):
    logging.basicConfig(
        filename="fill_missing_columns.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    total_prompts = len(missing_prompts)
    logging.info(f"Total prompts to process: {total_prompts}")
    logging.info(f"Model: {model_class.__name__}")

    # Process each prompt
    for i, prompt in enumerate(missing_prompts):
        model = model_class(prompt.text)
        process_model(
            prompt=prompt,
            model=model,
            prompt_index=i,
            model_index=0,
            total_prompts=total_prompts,
            total_models=1,
        )

    logging.info("Image generation complete.")


def generate_image_comparison_csv(output_csv):
    from collections import defaultdict

    # Fetch all prompts
    prompts = {row.id: row.text for row in db.read_prompts()}

    # Fetch all models
    models = {row.id: row.id for row in db.read_models()}

    # Fetch all images
    images_response = db.read_images()
    images_by_prompt = defaultdict(dict)
    for row in images_response:
        images_by_prompt[row.prompt_id][row.model_id] = row.image_url

    # Prepare CSV headers
    headers = ["prompt_id", "prompt_text"] + [f"{id}" for id in models.values()]

    # Write to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for prompt_id, prompt_text in prompts.items():
            row = [prompt_id, prompt_text]
            for model_id in models:
                row.append(images_by_prompt[prompt_id].get(model_id, ""))
            writer.writerow(row)

    print(f"CSV file '{output_csv}' has been generated successfully.")


def delete_duplicate_images():
    # Use db.read_images() to fetch all images
    images_response = db.read_images()

    # Group images by prompt_id and model_id
    image_groups = defaultdict(list)
    for row in images_response:
        image_groups[(row.prompt_id, row.model_id)].append(row)

    total_duplicates = 0
    duplicates_by_model = defaultdict(int)

    for (prompt_id, model_id), images in image_groups.items():
        if len(images) > 1:
            # Sort images by created_at timestamp in descending order
            sorted_images = sorted(images, key=lambda x: x.created_at, reverse=True)

            # Keep the first (latest) image, delete the rest
            for image_to_delete in sorted_images[1:]:
                try:
                    # First, delete any comparisons referencing this image
                    db.delete_comparisons_by_image_id(image_to_delete.id)

                    # Then delete the image
                    db.delete_image(image_to_delete.id)

                    total_duplicates += 1
                    duplicates_by_model[model_id] += 1
                    print(
                        f"Deleted duplicate: Prompt ID: {prompt_id}, Model ID: {model_id}, Image ID: {image_to_delete.id}"
                    )
                except Exception as e:
                    print(f"Error deleting image {image_to_delete.id}: {str(e)}")

    print(f"\nTotal number of deleted duplicate images: {total_duplicates}")
    print("\nDeleted duplicates by model:")
    for model_id, count in duplicates_by_model.items():
        print(f"Model ID: {model_id}, Deleted duplicate count: {count}")

    # Calculate total number of images after deletion
    total_images = sum(len(images) for images in image_groups.values())

    print(f"\nTotal number of images after deletion: {total_images}")
    print(
        f"Percentage of duplicates deleted: {(total_duplicates / (total_images + total_duplicates)) * 100:.2f}%"
    )


def count_duplicate_images():
    # Fetch all images
    images_response = db.read_images()

    # Count images per prompt_id and model_id combination
    image_counts = defaultdict(lambda: defaultdict(int))
    total_images = 0

    for row in images_response:
        prompt_id = row.prompt_id
        model_id = row.model_id
        image_counts[prompt_id][model_id] += 1
        total_images += 1

    # Count and print duplicates
    total_duplicates = 0
    duplicates_by_model = defaultdict(int)

    for prompt_id, models in image_counts.items():
        for model_id, count in models.items():
            if count > 1:
                duplicates = count - 1
                total_duplicates += duplicates
                duplicates_by_model[model_id] += duplicates
                print(
                    f"Prompt ID: {prompt_id}, Model ID: {model_id}, Duplicate count: {duplicates}"
                )

    print(f"\nTotal number of duplicate images: {total_duplicates}")
    print("\nDuplicates by model:")
    for model_id, count in duplicates_by_model.items():
        print(f"Model ID: {model_id}, Duplicate count: {count}")

    print(f"\nTotal number of images: {total_images}")
    if total_images > 0:
        print(
            f"Percentage of duplicates: {(total_duplicates / total_images) * 100:.2f}%"
        )
    else:
        print("No images found.")
