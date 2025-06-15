import asyncio
from blinkpy.blinkpy import Blink
from aiohttp import ClientSession

async def main():
    """Main function to run blinkpy."""
    blink = Blink(session=ClientSession())

    # This will prompt for username/password/2FA
    result = await blink.start()

    if result:
        print("BlinkPy login successful! No errors observed.")
        # print(f"You have {len(blink.cameras)} cameras.")
        # for camera in blink.cameras:
        #     print(f"Camera {camera}: {blink.cameras[camera].name}")
    else:
        print("BlinkPy login failed.")
    
    print(f"Cat camera is named: {list(blink.cameras.keys())[9]}")
    camera = blink.cameras["44 cat bowl"]
    print(camera.attributes)

if __name__ == "__main__":
    asyncio.run(main())