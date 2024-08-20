import os
from datetime import UTC, datetime
from io import BytesIO
from typing import List, Optional, Tuple, Union

import requests
from dotenv import load_dotenv
from PIL import Image as Img
from pydantic import UUID4, BaseModel, HttpUrl
from supabase import Client, create_client

from adapter import ImageGen

# Load environment variables
load_dotenv()

# Initialize Supabase client
url: str = os.environ.get("SUPABASE_URL")
# key: str = os.environ.get("SUPABASE_KEY")
key: str = os.environ.get("SUPABASE_ADMIN_KEY")
supabase: Client = create_client(url, key)


class Prompt(BaseModel):
    id: UUID4
    text: str
    created_at: datetime


class Image(BaseModel):
    id: UUID4
    created_at: datetime
    prompt_id: UUID4
    model_id: str
    image_url: HttpUrl


class Comparison(BaseModel):
    id: UUID4
    created_at: datetime
    prompt_id: UUID4
    image1_id: UUID4
    image2_id: UUID4
    image3_id: UUID4
    image4_id: UUID4
    winner_id: UUID4


class Ranking(BaseModel):
    id: UUID4
    created_at: datetime
    model_id: str
    elo_score: float


class Model(BaseModel):
    id: str
    created_at: datetime
    name: str
    description: str


def get_utc_timestamp():
    return datetime.now(UTC).isoformat()


def write_comparison(prompt_id, image1_id, image2_id, image3_id, image4_id, winner_id):
    required_fields = {
        "prompt_id": prompt_id,
        "image1_id": image1_id,
        "image2_id": image2_id,
        "image3_id": image3_id,
        "image4_id": image4_id,
        "winner_id": winner_id,
    }

    missing_fields = [
        field for field, value in required_fields.items() if value is None
    ]

    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

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


def read_comparisons(
    id: Optional[UUID4] = None,
) -> Union[Comparison, List[Comparison], None]:
    if id:
        result = supabase.table("comparisons").select("*").eq("id", id).execute()
        if result.data:
            return Comparison(**result.data[0])
        return None
    else:
        result = supabase.table("comparisons").select("*").execute()
        return [Comparison(**item) for item in result.data]


def delete_comparisons_by_image_id(image_id):
    supabase.table("comparisons").delete().or_(
        f"image1_id.eq.{image_id},"
        f"image2_id.eq.{image_id},"
        f"image3_id.eq.{image_id},"
        f"image4_id.eq.{image_id}"
    ).execute()


# Images table operations
def write_image(prompt_id, model_id, image_url, image_id):
    data = (
        supabase.table("images")
        .insert(
            {
                "prompt_id": prompt_id,
                "model_id": model_id,
                "image_url": image_url,
                "created_at": get_utc_timestamp(),
                "id": image_id,
            }
        )
        .execute()
    )
    return data


def delete_image(image_id: UUID4):
    try:
        response = supabase.table("images").delete().eq("id", str(image_id)).execute()
        if response.data:
            return response.data
        else:
            raise Exception(f"No image found with id: {image_id}")
    except Exception as e:
        print(f"An error occurred while deleting the image: {e}")
        raise


def read_images(
    id: Optional[UUID4] = None, prompt_id: Optional[UUID4] = None
) -> Union[Image, List[Image], None]:
    query = supabase.table("images").select("*")

    if id:
        result = query.eq("id", id).execute()
        if result.data:
            return Image(**result.data[0])
        return None
    elif prompt_id:
        result = query.eq("prompt_id", prompt_id).execute()
        return [Image(**item) for item in result.data]
    else:
        all_images = []
        page = 1
        page_size = 1000  # Maximum allowed by Supabase

        while True:
            result = query.range((page - 1) * page_size, page * page_size - 1).execute()
            images = [Image(**item) for item in result.data]
            all_images.extend(images)

            if len(images) < page_size:
                break

            page += 1

        return all_images


# Prompts table operations
def write_prompt(text):
    data = (
        supabase.table("prompts")
        .insert({"text": text, "created_at": get_utc_timestamp()})
        .execute()
    )
    return data


def get_prompt_by_text(text: str) -> Optional[Prompt]:
    result = supabase.table("prompts").select("*").eq("text", text).limit(1).execute()
    if result.data:
        return Prompt(**result.data[0])
    return None


def read_prompts(id: Optional[UUID4] = None) -> Union[Prompt, List[Prompt], None]:
    if id:
        result = supabase.table("prompts").select("*").eq("id", id).execute()
        if result.data:
            return Prompt(**result.data[0])
        return None
    else:
        result = supabase.table("prompts").select("*").execute()
        return [Prompt(**item) for item in result.data]


def fetch_unprocessed_prompts() -> List[Prompt]:
    response = supabase.rpc("fetch_unprocessed_prompts", {}).execute()
    if hasattr(response, "error") and response.error is not None:
        raise Exception(f"Error fetching unprocessed prompts: {response.error}")

    return [Prompt(**item) for item in response.data]


# Rankings table operations
def write_ranking(model_id, elo_score):
    data = (
        supabase.table("rankings")
        .upsert(
            {
                "model_id": model_id,
                "elo_score": elo_score,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def bulk_write_rankings(rankings: List[Tuple[str, float]]):
    current_time = get_utc_timestamp()
    data_to_update = [
        {"model_id": model_id, "elo_score": elo_score, "created_at": current_time}
        for model_id, elo_score in rankings
    ]

    try:
        response = (
            supabase.table("rankings")
            .upsert(data_to_update, on_conflict="model_id")
            .execute()
        )

        return response
    except Exception as e:
        print(f"An error occurred during bulk insert: {e}")
        raise


def read_rankings(id: Optional[UUID4] = None) -> Union[Ranking, List[Ranking], None]:
    if id:
        result = supabase.table("rankings").select("*").eq("id", id).execute()
        if result.data:
            return Ranking(**result.data[0])
        return None
    else:
        result = supabase.table("rankings").select("*").execute()
        return [Ranking(**item) for item in result.data]


# Models table operations
def write_model(id, name, description):
    data = (
        supabase.table("models")
        .upsert(
            {
                "id": id,
                "name": name,
                "description": description,
                "created_at": get_utc_timestamp(),
            }
        )
        .execute()
    )
    return data


def read_models(id: Optional[str] = None) -> Union[Model, List[Model], None]:
    if id:
        result = supabase.table("models").select("*").eq("id", id).execute()
        if result.data:
            return Model(**result.data[0])
        return None
    else:
        result = supabase.table("models").select("*").execute()
        return [Model(**item) for item in result.data]


def upload_image_to_bucket(image_url: str):
    if not isinstance(image_url, str):
        raise ValueError("image_url must be a string")

    response = requests.get(image_url)
    image = Img.open(BytesIO(response.content))

    image_gen = ImageGen(image=image)
    filename = image_gen.img_id

    webp_image_path = f"/tmp/{filename}.webp"
    image.save(webp_image_path, "WEBP")

    with open(webp_image_path, "rb") as f:
        r = supabase.storage.from_("images_bucket").upload(
            file=f,
            path=f"{filename}.webp",
            file_options={"content-type": "image/webp"},
        )

        return filename, r.json()


def get_model_ratings_from_image_ids(
    image_ids: List[str],
) -> List[Tuple[str, str, float]]:
    if len(image_ids) != 4:
        raise ValueError("Exactly 4 image IDs must be provided")

    try:
        images_response = (
            supabase.table("images")
            .select("id, model_id")
            .in_("id", image_ids)
            .execute()
        )

        if hasattr(images_response, "error") and images_response.error is not None:
            raise Exception(f"Supabase query error: {images_response.error}")

        images_data = images_response.data

        if len(images_data) != 4:
            raise ValueError(f"Expected 4 results, but got {len(images_data)}")

        # Extract model_ids
        model_ids = [img["model_id"] for img in images_data]

        # Query rankings table
        rankings_response = (
            supabase.table("rankings")
            .select("models(name), model_id, elo_score")
            .in_("model_id", model_ids)
            .execute()
        )

        if hasattr(rankings_response, "error") and rankings_response.error is not None:
            raise Exception(f"Supabase query error: {rankings_response.error}")

        rankings_data = {
            r["model_id"]: {"elo_score": r["elo_score"], "name": r["models"]["name"]}
            for r in rankings_response.data
        }
        # Combine the results
        result = []
        for img in images_data:
            model_id = img["model_id"]
            res = rankings_data.get(model_id, None)
            elo_score = res["elo_score"]
            name = res["name"]

            result.append(
                (
                    model_id,
                    name,
                    elo_score,
                )
            )

        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def get_all_rankings_with_names() -> List[Tuple[str, str, float]]:
    """
    Retrieve rankings for all models, joined with model names,
    sorted by elo_score in descending order.

    Returns:
    List[Tuple[str, str, float]]: A list of tuples containing (model_id, model_name, elo_score),
                                  sorted by elo_score in descending order.
    """
    try:
        # Fetch all rankings joined with model names
        response = (
            supabase.table("rankings")
            .select("model_id, elo_score, models(name)")
            .order("elo_score", desc=True)
            .execute()
        )

        if hasattr(response, "error") and response.error is not None:
            raise Exception(f"Supabase query error: {response.error}")

        rankings = [
            {
                "model_id": item["model_id"],
                "model_name": item["models"]["name"],
                "elo": item["elo_score"],
            }
            for item in response.data
        ]

        return rankings

    except Exception as e:
        print(f"An error occurred while fetching rankings: {e}")
        raise


def get_leaderboard():
    try:
        response = supabase.rpc("get_rankings_with_win_counts", {}).execute()

        if hasattr(response, "error") and response.error is not None:
            raise Exception(f"Supabase query error: {response.error}")

        return response.data

    except Exception as e:
        print(f"An error occurred while fetching and processing win counts: {e}")
        raise
