import asyncio
from blinkpy.blinkpy import Blink
from aiohttp import ClientSession
from blinkpy.auth import Auth

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

            print(f"Cat camera is named: {list(blink.cameras.keys())[9]}")
            camera = blink.cameras["44 cat bowl"]
            print(camera.attributes)

        else:
            print("BlinkPy login failed.")

    finally:
        # --- IMPORTANT: Ensure the session is closed ---
        # This 'finally' block ensures the session.close() is called
        # whether the try block succeeds or fails.
        if session and not session.closed:
            await session.close()

if __name__ == "__main__":
    asyncio.run(main())