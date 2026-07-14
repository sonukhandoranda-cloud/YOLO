from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


API_URL = "https://commons.wikimedia.org/w/api.php"

QUERIES = {
    "shoe": [
        "sports shoe footwear",
        "sneaker footwear",
        "formal shoe footwear",
    ],
    "sandal": [
        "sandal footwear",
        "leather sandal footwear",
        "sandals feet",
    ],
    "other_footwear": [
        "slipper footwear",
        "flip flop footwear",
        "bare feet",
    ],
}


def safe_name(text: str) -> str:
    text = re.sub(r"^File:", "", text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text[:140]


def api_get(params: dict[str, str]) -> dict:
    params = {"format": "json", "formatversion": "2", **params}
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "AI-Attendance-System/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def search_images(query: str, limit: int, thumb_width: int) -> list[str]:
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
    urls: list[str] = []
    for page in data.get("query", {}).get("pages", []):
        imageinfo = page.get("imageinfo") or []
        if not imageinfo:
            continue
        info = imageinfo[0]
        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        url = info.get("thumburl") or info.get("url")
        if url and info.get("mime") in {"image/jpeg", "image/png"} and width >= 200 and height >= 200:
            urls.append(url)
    return urls


def download_file(url: str, output_dir: Path, prefix: str) -> Path | None:
    parsed = urllib.parse.urlparse(url)
    filename = f"{prefix}_{safe_name(Path(urllib.parse.unquote(parsed.path)).name)}"
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
    parser = argparse.ArgumentParser(description="Download public footwear images into class folders.")
    parser.add_argument("--output-dir", default="data/raw/footwear_classes")
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument("--thumb-width", type=int, default=900)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    seen_urls: set[str] = set()
    for class_name, queries in QUERIES.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        downloaded = len([p for p in class_dir.iterdir() if p.is_file()])
        for query_index, query in enumerate(queries, start=1):
            if downloaded >= args.max_per_class:
                break
            print(f"Searching {class_name}: {query}")
            try:
                urls = search_images(query, args.per_query, args.thumb_width)
            except Exception as exc:
                print(f"Search skipped for {query}: {exc}")
                continue
            for url in urls:
                if url in seen_urls or downloaded >= args.max_per_class:
                    continue
                seen_urls.add(url)
                path = download_file(url, class_dir, f"{class_name}_{query_index:02d}")
                if path is None:
                    continue
                downloaded += 1
                print(f"{class_name}: {downloaded:03d} {path.name}")
                time.sleep(args.delay)
        print(f"{class_name}: {downloaded} images")

    print(f"Downloaded class images to {root}")


if __name__ == "__main__":
    main()
