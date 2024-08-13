import os
from datetime import UTC, datetime

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
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


prompt_data = write_prompt(
    "An underwater scene of a vibrant coral reef teeming with life. Schools of tropical fish dart between the coral, while a curious octopus changes colors to blend with its surroundings. Shafts of sunlight penetrate the crystal-clear water."
)
model_data = write_model("StableDiffusion3", "Stable Diffusion 3")
image_data = write_image(
    prompt_data.data[0]["id"],
    model_data.data[0]["id"],
    "https://yycelpiurkvyijumsxcw.supabase.co/storage/v1/object/public/images_bucket/R8_SD3_00001.png",
)

# Read examples
print("All prompts:", read_prompts().data)
print("Single prompt:", read_prompts(prompt_data.data[0]["id"]).data)

print("All models:", read_models().data)
print("Single model:", read_models(model_data.data[0]["id"]).data)

print("All images:", read_images().data)
print("Single image:", read_images(image_data.data[0]["id"]).data)
