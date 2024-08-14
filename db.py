import os
from datetime import UTC, datetime
from io import BytesIO
from typing import List, Optional, Union

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

    print("i got here")
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
        result = query.execute()
        return [Image(**item) for item in result.data]


# Prompts table operations
def write_prompt(text):
    data = (
        supabase.table("prompts")
        .insert({"text": text, "created_at": get_utc_timestamp()})
        .execute()
    )
    return data


def read_prompts(id: Optional[UUID4] = None) -> Union[Prompt, List[Prompt], None]:
    if id:
        result = supabase.table("prompts").select("*").eq("id", id).execute()
        if result.data:
            return Prompt(**result.data[0])
        return None
    else:
        result = supabase.table("prompts").select("*").execute()
        return [Prompt(**item) for item in result.data]


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
        .insert(
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
