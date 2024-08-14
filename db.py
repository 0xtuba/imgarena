import os
from datetime import UTC, datetime
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image
from supabase import Client, create_client

from adapter import ImageGen

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
# key: str = os.environ.get("SUPABASE_KEY")
key: str = os.environ.get("SUPABASE_ADMIN_KEY")
supabase: Client = create_client(url, key)


def get_utc_timestamp():
    return datetime.now(UTC).isoformat()


# Comparisons table operations
def write_comparison(prompt_id, image1_id, image2_id, image3_id, image4_id, winner_id):
    data = (
        supabase.table("comparisons")
        .insert(
            {
                "prompt_id": prompt_id,
                "image1_id": image1_id,
                "image2_id": image2_id,
                "image3_id": image3_id,
                "image4_id": image4_id,
                "winner_id": winner_id,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def read_comparisons(id=None):
    if id:
        return supabase.table("comparisons").select("*").eq("id", id).execute()
    return supabase.table("comparisons").select("*").execute()


# Images table operations
def write_image(prompt_id, model_id, image_url):
    data = (
        supabase.table("images")
        .insert(
            {
                "prompt_id": prompt_id,
                "model_id": model_id,
                "image_url": image_url,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def read_images(id=None):
    if id:
        return supabase.table("images").select("*").eq("id", id).execute()
    return supabase.table("images").select("*").execute()


# Prompts table operations
def write_prompt(text):
    data = (
        supabase.table("prompts")
        .insert({"text": text, "created_at": get_utc_timestamp()})
        .execute()
    )
    return data


def read_prompts(id=None):
    if id:
        return supabase.table("prompts").select("*").eq("id", id).execute()
    return supabase.table("prompts").select("*").execute()


# Rankings table operations
def write_ranking(model_id, elo_score):
    data = (
        supabase.table("rankings")
        .insert(
            {
                "model_id": model_id,
                "elo_score": elo_score,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def read_rankings(id=None):
    if id:
        return supabase.table("rankings").select("*").eq("id", id).execute()
    return supabase.table("rankings").select("*").execute()


# Models table operations
def write_model(name, description):
    data = (
        supabase.table("models")
        .insert(
            {
                "name": name,
                "description": description,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def read_models(id=None):
    if id:
        return supabase.table("models").select("*").eq("id", id).execute()
    return supabase.table("models").select("*").execute()


def upload_image_to_bucket(image_url: str, metadata: dict):
    if "model_id" not in metadata or "prompt_id" not in metadata:
        raise ValueError("metadata must contain 'model_id' and 'prompt_id'")

    if not isinstance(image_url, str):
        raise ValueError("image_url must be a string")

    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    image_gen = ImageGen(
        image=image,
        model_id=metadata.get("model_id"),
        prompt_id=metadata.get("prompt_id"),
    )
    filename = image_gen.filename()

    webp_image_path = f"/tmp/{filename}.webp"
    image.save(webp_image_path, "WEBP")

    with open(webp_image_path, "rb") as f:
        r = supabase.storage.from_("images_bucket").upload(
            file=f,
            path=f"{filename}.webp",
            file_options={"content-type": "image/webp"},
        )

        return r
