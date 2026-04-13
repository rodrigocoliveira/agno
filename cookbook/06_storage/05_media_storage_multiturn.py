"""
Test: Multi-turn conversation with media storage + store_media=False.

This script reproduces the bug where turn 2 loses the image when
store_media=False + media_storage is configured.

With the fix, the agent should remember the image on turn 2 because
the MediaReference pointer is preserved in the DB and reconstructed
via URL refresh when loading history.

Usage:
    .venvs/demo/bin/python cookbook/06_storage/test_media_storage_multiturn.py
"""

import shutil
import time
from pathlib import Path

import httpx
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.media import Image
from agno.media_storage.local import LocalMediaStorage
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clean slate
# ---------------------------------------------------------------------------
MEDIA_DIR = "./tmp/test_multiturn_media"
DB_FILE = "tmp/test_multiturn.db"

shutil.rmtree(MEDIA_DIR, ignore_errors=True)
Path(DB_FILE).unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Setup: store_media=False + LocalMediaStorage
# ---------------------------------------------------------------------------
storage = LocalMediaStorage(base_path=MEDIA_DIR)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    media_storage=storage,
    store_media=False,  # Don't store raw bytes in DB
    db=SqliteDb(db_file=DB_FILE),
    session_id="test-multiturn-session",
    add_history_to_context=True,
)

# Download image
image_url = "https://picsum.photos/id/15/800/600.jpg"
print("Downloading test image...")
image_bytes = httpx.get(image_url, follow_redirects=True).content
print(f"Downloaded {len(image_bytes)} bytes\n")

# ---------------------------------------------------------------------------
# Turn 1: Send image + ask about it
# ---------------------------------------------------------------------------
print("=" * 70)
print(" TURN 1: Sending image + asking 'What do you see in this image?'")
print("=" * 70)

t0 = time.time()
response1 = agent.run(
    "What do you see in this image?",
    images=[Image(content=image_bytes, format="jpeg")],
)
t1 = time.time()

print(f"\n--- Response ({t1 - t0:.1f}s) ---")
print(response1.content)
print()

# ---------------------------------------------------------------------------
# Turn 2: Ask about the image WITHOUT re-sending it
# ---------------------------------------------------------------------------
print("=" * 70)
print(" TURN 2: Asking 'What was the image about?' (no image attached)")
print("=" * 70)

t0 = time.time()
response2 = agent.run("What was the image about?")
t1 = time.time()

print(f"\n--- Response ({t1 - t0:.1f}s) ---")
print(response2.content)
print()

# ---------------------------------------------------------------------------
# Verification: Did the image ACTUALLY get sent to the model on turn 2?
# ---------------------------------------------------------------------------
print("=" * 70)
print(" VERIFICATION")
print("=" * 70)

# Check 1: History messages in turn 2 should contain image data
history_images_found = 0
history_images_with_content = 0
for msg in response2.messages or []:
    if msg.from_history and msg.images:
        for img in msg.images:
            history_images_found += 1
            has_content = img.content is not None
            has_url = img.url is not None
            has_ref = img.media_reference is not None
            if has_content or has_url:
                history_images_with_content += 1
            print(
                f"  History image: id={img.id}, "
                f"has_content={has_content}, "
                f"has_url={has_url}, "
                f"has_media_reference={has_ref}"
            )

print(f"\n  History images found in turn 2 messages: {history_images_found}")
print(
    f"  History images with content/url (sent to model): {history_images_with_content}"
)

# Check 2: Media files exist on disk
media_files = list(Path(MEDIA_DIR).glob("*"))
media_files = [f for f in media_files if not f.name.endswith(".meta.json")]
print(f"  Media files on disk: {len(media_files)}")
for f in media_files:
    print(f"    {f.name} ({f.stat().st_size} bytes)")

# Check 3: DB should have MediaReference but NOT raw bytes
print()
db_size = Path(DB_FILE).stat().st_size if Path(DB_FILE).exists() else 0
print(f"  DB file size: {db_size:,} bytes")
print(f"  Image size:   {len(image_bytes):,} bytes")
if db_size < len(image_bytes):
    print("  DB is smaller than the image -> raw bytes NOT in DB (good!)")
else:
    print("  WARNING: DB is larger than image -> raw bytes may be in DB")

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------
print()
print("=" * 70)

if history_images_with_content > 0:
    print(" PASS: Image was reconstructed from storage and sent to model on turn 2")
elif history_images_found > 0:
    print(" PARTIAL: Image reference found but no content/url to send to model")
else:
    # Fallback: check model response text
    keywords = [
        "landscape",
        "nature",
        "green",
        "mountain",
        "tree",
        "forest",
        "hill",
        "road",
        "scenic",
        "waterfall",
        "rock",
        "stream",
    ]
    content_lower = (response2.content or "").lower()
    remembered = any(kw in content_lower for kw in keywords)
    if remembered:
        print(" PASS (text-based): Turn 2 response references the image content")
    else:
        cant_see = any(
            phrase in content_lower
            for phrase in [
                "can't see",
                "cannot see",
                "don't have access",
                "don't have the image",
            ]
        )
        if cant_see:
            print(" FAIL: Turn 2 lost the image (model says it can't see images)")
        else:
            print(f" UNCLEAR: Check manually. Response: {content_lower[:200]}")

print("=" * 70)
