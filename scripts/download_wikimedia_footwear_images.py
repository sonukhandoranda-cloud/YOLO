from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"


def safe_name(text: str) -> str:
    text = re.sub(r"^File:", "", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:120]


def api_get(params: dict[str, str]) -> dict:
    params = {
        "format": "json",
        "formatversion": "2",
        **params,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Attendance-System/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_images(query: str, limit: int, thumb_width: int) -> list[str]:
    try:
        data = api_get(
            {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": "6",
                "gsrlimit": str(limit),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": str(thumb_width),
            }
        )
    except Exception as exc:
        print(f"Search skipped for {query}: {exc}")
        return []
    pages = data.get("query", {}).get("pages", [])
    urls: list[str] = []
    for page in pages:
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        info = imageinfo[0]
        mime = info.get("mime", "")
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        url = info.get("thumburl") or info.get("url")
        if url and mime in {"image/jpeg", "image/png"} and width >= 200 and height >= 200:
            urls.append(url)
    return urls


def download_file(url: str, output_dir: Path) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    filename = safe_name(Path(urllib.parse.unquote(parsed.path)).name)
    if not filename:
        return None
    output_path = output_dir / filename
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Attendance-System/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                return None
            output_path.write_bytes(response.read())
        return output_path
    except Exception as exc:
        print(f"Skipped {url}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sample footwear images from Wikimedia Commons.")
    parser.add_argument("--output-dir", default="data/raw/footwear_samples")
    parser.add_argument("--per-query", type=int, default=15)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--thumb-width", type=int, default=900)
    parser.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args()

    queries = [
        "shoe footwear",
        "sneaker footwear",
        "sandal footwear",
        "slipper footwear",
        "feet shoes",
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    seen_urls: set[str] = set()
    for query in queries:
        print(f"Searching: {query}")
        for url in search_images(query, args.per_query, args.thumb_width):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            path = download_file(url, output_dir)
            if path is not None:
                downloaded.append(path)
                print(f"Downloaded {len(downloaded):02d}: {path.name}")
                time.sleep(args.delay)
            if len(downloaded) >= args.max_images:
                break
        if len(downloaded) >= args.max_images:
            break

    print()
    print(f"Downloaded {len(downloaded)} images to {output_dir}")


if __name__ == "__main__":
    main()
