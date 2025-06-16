import asyncio
from blinkpy.blinkpy import Blink
from aiohttp import ClientSession
from blinkpy.auth import Auth
import os

async def main():
    """Main function to run blinkpy."""
    session = ClientSession()
    blink = Blink(session=session) # Pass the session to Blink

    try:
        # This will prompt for username/password/2FA
        result = await blink.start()

        if result:
            print("BlinkPy login successful! No errors observed.")
            print(f"You have {len(blink.cameras)} cameras.")

            # iterating throguh cameras
            # for name, camera in blink.cameras.items():
            #     print(f"Camera is named: {name}")
            #     print(camera.attributes) # Print all attributes for one camera

            # Save credentials to file
            credentials_file = "blink_config.json"
            await blink.save(credentials_file)
            print(f"\nCredentials saved to: {credentials_file}")
        else:
            print("BlinkPy login failed.")

    finally:
        # Ensure the session is closed
        if session and not session.closed:
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())