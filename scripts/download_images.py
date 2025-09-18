#!/usr/bin/env python3
"""
Image Download Script

Downloads verified helmet images to local repository:
- Downloads only curl-confirmed working URLs
- Saves images with proper filenames
- Updates database with local file paths
- Ensures data permanence in repository
"""

import requests
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

# Base directories
ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets')
IMAGES_DIR = os.path.join(ASSETS_DIR, 'helmet_images')
DOWNLOADS_DIR = os.path.join(IMAGES_DIR, 'downloads')

def download_verified_images():
    """Download all verified helmet images to local storage"""

    print("📥 Downloading Verified Helmet Images to Repository...")

    # Create downloads directory
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)

    # Load verified image database
    db_file = os.path.join(IMAGES_DIR, 'helmet_images_database.json')

    try:
        with open(db_file, 'r') as f:
            image_database = json.load(f)
    except FileNotFoundError:
        print("❌ Image database not found")
        return False

    print(f"🔍 Found {len(image_database)} verified images to download\n")

    downloaded_count = 0
    updated_database = {}

    for image_id, image_data in image_database.items():
        try:
            result = download_single_image(image_id, image_data)
            if result:
                updated_database[image_id] = result
                downloaded_count += 1
            else:
                print(f"  ❌ Failed to download {image_id}")
        except Exception as e:
            print(f"  ❌ Error downloading {image_id}: {e}")

    # Save updated database with local file paths
    if updated_database:
        save_updated_database(updated_database)
        print(f"\n✅ Successfully downloaded {downloaded_count}/{len(image_database)} images")
        generate_download_report(updated_database)
    else:
        print(f"\n❌ No images could be downloaded")

    return downloaded_count > 0

def download_single_image(image_id, image_data):
    """Download a single image and update its metadata"""

    image_url = image_data.get('image_url')
    if not image_url:
        print(f"  ❌ No URL for {image_id}")
        return None

    print(f"  📥 Downloading {image_id}...")
    print(f"      URL: {image_url}")

    try:
        # Get file extension from URL or headers
        parsed_url = urlparse(image_url)
        original_filename = os.path.basename(parsed_url.path)
        file_ext = get_file_extension(original_filename, image_url)

        # Generate local filename
        local_filename = f"{image_id.lower()}{file_ext}"
        local_filepath = os.path.join(DOWNLOADS_DIR, local_filename)

        # Download image
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()

        # Verify it's actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            print(f"      ⚠️ Not an image: {content_type}")
            return None

        # Save image file
        with open(local_filepath, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content)
        print(f"      ✅ Saved: {local_filename} ({file_size:,} bytes)")

        # Update metadata with local file info
        updated_data = image_data.copy()
        updated_data.update({
            'local_file_path': f'assets/helmet_images/downloads/{local_filename}',
            'local_filename': local_filename,
            'downloaded_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size_bytes': file_size,
            'content_type': content_type,
            'download_status': 'success',
            'original_url': image_url,  # Keep original URL as backup
            'data_permanence': True
        })

        # Add a small delay to be respectful
        time.sleep(1)

        return updated_data

    except requests.RequestException as e:
        print(f"      ❌ Download failed: {e}")
        return None
    except Exception as e:
        print(f"      ❌ Unexpected error: {e}")
        return None

def get_file_extension(filename, url):
    """Get appropriate file extension"""

    # Try to get extension from filename
    if filename and '.' in filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
            return ext

    # Default to .jpg for most helmet product images
    if 'jpg' in url.lower() or 'jpeg' in url.lower():
        return '.jpg'
    elif 'png' in url.lower():
        return '.png'
    elif 'webp' in url.lower():
        return '.webp'
    else:
        return '.jpg'  # Default fallback

def save_updated_database(updated_database):
    """Save updated database with local file paths"""

    print(f"💾 Updating database with local file paths...")

    # Save updated database
    db_file = os.path.join(IMAGES_DIR, 'helmet_images_database.json')
    with open(db_file, 'w', encoding='utf-8') as f:
        json.dump(updated_database, f, indent=2, ensure_ascii=False)

    # Also save a backup of URLs-only version
    urls_only = {}
    for image_id, data in updated_database.items():
        urls_only[image_id] = {
            'id': data.get('id'),
            'title': data.get('title'),
            'original_url': data.get('original_url'),
            'verification_status': data.get('verification_status')
        }

    backup_file = os.path.join(IMAGES_DIR, 'original_urls_backup.json')
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(urls_only, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Database updated with local file paths")
    print(f"  ✅ Original URLs backed up to: original_urls_backup.json")

def generate_download_report(database):
    """Generate download summary report"""

    print(f"\n📊 Download Summary Report:")
    print("=" * 50)

    total_size = 0
    formats = {}

    for image_id, data in database.items():
        file_size = data.get('file_size_bytes', 0)
        content_type = data.get('content_type', 'unknown')
        local_file = data.get('local_filename', 'unknown')

        total_size += file_size

        # Count formats
        if content_type not in formats:
            formats[content_type] = 0
        formats[content_type] += 1

        print(f"✅ {image_id}:")
        print(f"   File: {local_file}")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Type: {content_type}")

    print(f"\n📈 Summary:")
    print(f"  • Total Images: {len(database)}")
    print(f"  • Total Size: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"  • File Formats:")
    for fmt, count in formats.items():
        print(f"    - {fmt}: {count} files")

    print(f"\n📁 Storage Location: {DOWNLOADS_DIR}")
    print(f"💡 Images are now permanently stored in the repository!")

def create_readme():
    """Create README for images directory"""

    readme_content = f"""# Helmet Images Database

This directory contains verified helmet product images downloaded from official sources.

## Contents

- `helmet_images_database.json` - Complete image database with local file paths
- `original_urls_backup.json` - Backup of original URLs for reference
- `downloads/` - Downloaded image files

## Image Sources

All images are from verified, curl-confirmed sources:
- Gentex Corporation product catalog
- Professional product photography suitable for QC analysis

## Usage

Images are ready for use in:
- Computer vision QC systems
- AI model training
- Defect detection demonstrations
- Quality control procedures

## Data Permanence

Images are stored locally to ensure:
- No broken links
- Permanent data availability
- Offline access capability
- Repository self-containment

Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    readme_path = os.path.join(IMAGES_DIR, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"📝 Created README.md for images directory")

def main():
    """Main download function"""

    print("🚀 Starting Helmet Image Download Process\n")

    try:
        # Download verified images
        success = download_verified_images()

        if success:
            # Create documentation
            create_readme()
            print(f"\n🎯 Image download completed successfully!")
            print(f"📁 Images saved to: {DOWNLOADS_DIR}")
            print(f"💾 Database updated with local file paths")
            print(f"📖 Documentation created")
        else:
            print(f"\n❌ Image download failed")

    except Exception as e:
        print(f"\n❌ Download process failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()