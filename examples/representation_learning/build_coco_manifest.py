"""Convert a COCO captions annotation JSON into the retrieval JSONL manifest format."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--image-prefix",
        default="",
        help="Optional path prepended to each COCO file_name, e.g. val2014.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = json.loads(Path(args.captions_json).read_text())
    captions = defaultdict(list)
    for item in annotation["annotations"]:
        captions[item["image_id"]].append(item["caption"])
    prefix = Path(args.image_prefix)
    records = []
    for image in annotation["images"]:
        image_captions = captions.get(image["id"], [])
        if image_captions:
            records.append(
                {
                    "image": str(prefix / image["file_name"]),
                    "captions": image_captions,
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} image-caption records to {output}")


if __name__ == "__main__":
    main()
