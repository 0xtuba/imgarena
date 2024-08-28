import csv

import db
import util


def extract_info_from_csv(file_path):
    data = []
    with open(file_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def match_prompts(missing_prompts, imagine_csv_path):
    # Read the imagine.csv file
    with open(imagine_csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        imagine_data = list(reader)

    matched_data = []

    # Iterate through missing prompts and find matches in imagine_data
    for missing_prompt in missing_prompts:
        # Assuming the Prompt object has a 'text' attribute
        prompt_text = missing_prompt.text.strip().lower()
        for row in imagine_data:
            if prompt_text == row["prompt"].strip().lower():
                matched_data.append(row)
                break

    return matched_data


def image_url(img_id):
    return f"https://cl.imagineapi.dev/assets/{img_id}.png"


# Get missing prompts
missing_prompts = util.check_missing("midjourney", "master.csv")

# Match prompts
matched_data = match_prompts(missing_prompts, "imagine.csv")

# Process each matched prompt
for row in matched_data:
    prompt = next(
        p
        for p in missing_prompts
        if p.text.strip().lower() == row["prompt"].strip().lower()
    )
    print(prompt)
    import json

    # Parse the upscaled field and get the first upscaled image ID
    upscaled_ids = json.loads(row["upscaled"])
    if not upscaled_ids:
        print(f"No upscaled images for prompt: {row['prompt']}")
        continue

    img_id = upscaled_ids[0]

    # Get the image URL
    img_url = image_url(img_id)

    # Upload image to bucket
    filename, res = db.upload_image_to_bucket(image_url=img_url)

    # Construct the bucket URL
    img_bucket_url = f"https://yycelpiurkvyijumsxcw.supabase.co/storage/v1/object/public/images_bucket/{filename}.webp"

    # Write image to database
    r = db.write_image(
        prompt_id=str(prompt.id),
        model_id="midjourney",
        image_url=img_bucket_url,
        image_id=filename,
    )

    print(f"Processed prompt: {row['prompt']}")
    print(f"Image uploaded and written to database: {img_bucket_url}")
    print("---")
