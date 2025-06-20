# To grab training videos from Blink
import asyncio
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load
from aiohttp import ClientSession 
from datetime import datetime, timedelta
import os

async def main():
    """Main function to run blinkpy by loading credentials from file."""
    credentials_file = "blink_config.json"

    # Check if the credentials file exists
    if not os.path.exists(credentials_file):
        print(f"Error: Credentials file '{credentials_file}' not found.")
        print("Please run 'setup.py' first to save credentials.")
        return

    session = ClientSession()
    blink = Blink(session=session)

    try:
        # Load credentials from the JSON file
        auth = Auth(await json_load(credentials_file), no_prompt=True)
        blink.auth = auth


        result = await blink.start()

        if result:
            download_dir = r"C:\Users\glori_7afg9d\Videos\blink-cat"
            os.makedirs(download_dir, exist_ok=True)
            print(f"Ensured download directory exists: {download_dir}")
            
            print("BlinkPy login successful from saved credentials! No errors observed.")
            print(f"Cat camera is named: {list(blink.cameras.keys())[9]}")
            camera = blink.cameras[list(blink.cameras.keys())[9]] # 44 cat bowl
            print(camera.attributes)

            # Download clips from the last 90 days
            ninety_days_ago = datetime.now() - timedelta(days=90)
            since_date_str = ninety_days_ago.strftime("%Y/%m/%d %H:%M")

            print(f"Starting download of clips for '{camera.name}' since {since_date_str}...")

            # Download the videos
            await blink.download_videos(
                path=download_dir,
                since=since_date_str,
                camera=camera.name,
                delay=2, # Recommended 2-second delay between API calls
                stop=50 # Increased pages to ensure more clips. 50 pages * ~25 videos/page = ~1250 videos
            )
            print(f"Finished downloading clips to '{download_dir}'.")

        else:
            print("BlinkPy login failed from saved credentials. Tokens might be expired.")
            print("Running 'setup.py' to refresh credentials.")

    finally:
        if session and not session.closed:
            await session.close()
            # print("aiohttp client session closed.")

if __name__ == "__main__":
    asyncio.run(main())
    # print("Done.")