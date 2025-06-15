import asyncio
from blinkpy.blinkpy import Blink
from blinkpy.auth import Auth
from blinkpy.helpers.util import json_load
from aiohttp import ClientSession 
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
            print("BlinkPy login successful from saved credentials! No errors observed.")
            print(f"Cat camera is named: {list(blink.cameras.keys())[9]}")
            camera = blink.cameras[list(blink.cameras.keys())[9]] # 44 cat bowl
            print(camera.attributes)

        else:
            print("BlinkPy login failed from saved credentials. Tokens might be expired.")
            print("Consider running 'setup.py' again to refresh credentials.")

    finally:
        if session and not session.closed:
            await session.close()
            # print("aiohttp client session closed.")

if __name__ == "__main__":
    asyncio.run(main())
    # print("Done.")