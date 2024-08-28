import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Initialize Supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_ADMIN_KEY")
supabase = create_client(url, key)

# Cloudflare API configuration
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_R2_BUCKET = "imgenarena"

# Cloudflare API endpoint
API_ENDPOINT = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/r2/buckets/{CLOUDFLARE_R2_BUCKET}/objects"

# Headers for Cloudflare API requests
headers = {
    "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
}

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def upload_to_r2(file_name, file_content):
    upload_url = f"{API_ENDPOINT}/{file_name}"
    response = requests.put(upload_url, headers=headers, data=file_content)
    if response.status_code == 200:
        logging.info(f"Successfully uploaded {file_name} to R2")
    else:
        logging.error(
            f"Failed to upload {file_name}: {response.status_code} - {response.text}"
        )


def list_all_objects(bucket_name, limit_to_100=False):
    all_objects = []
    offset = 0
    limit = 100  # You can adjust this value if needed

    while True:
        res = supabase.storage.from_(bucket_name).list(
            path="", options={"limit": limit, "offset": offset}
        )

        if not res:
            break

        all_objects.extend(res)

        # Check if we've reached the desired limit or if there are no more items
        if limit_to_100 and len(all_objects) >= 100:
            all_objects = all_objects[:100]  # Trim to exactly 100 items
            break

        if len(res) < limit:
            break

        # Update the offset for the next page
        offset += limit

        # Break if we're limiting to 100 and have reached or exceeded that number
        if limit_to_100 and len(all_objects) >= 100:
            break

    return all_objects


# List all objects in Supabase storage
all_files = list_all_objects("images_bucket")

if isinstance(all_files, list):
    total_files = len(all_files)
    logging.info(f"Found {total_files} files to transfer")

    def process_file(item):
        file_name = item["name"]
        try:
            # Download file from Supabase
            result = supabase.storage.from_("images_bucket").download(file_name)

            # Check if result is a tuple and unpack accordingly
            if isinstance(result, tuple):
                file_data = result[0]
            else:
                file_data = result

            # Upload file to Cloudflare R2
            upload_to_r2(file_name, file_data)
            return True
        except Exception as e:
            logging.error(f"Error processing {file_name}: {str(e)}")
            logging.error(f"Result type: {type(result)}, Content: {result}")
            return False

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file, item) for item in all_files]

        for _ in tqdm(
            as_completed(futures),
            total=total_files,
            desc="Transferring files",
            unit="file",
        ):
            pass

    successful_transfers = sum(future.result() for future in futures)
    logging.info(
        f"Transfer complete. Successfully transferred {successful_transfers} out of {total_files} files."
    )
else:
    logging.error("Failed to list objects from Supabase storage")

print("Transfer complete")
