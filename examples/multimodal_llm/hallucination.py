"""CHAIR hallucination metrics for image captioning.

CHAIR_s  = fraction of mentioned objects that are not present in the image.
CHAIR_i  = fraction of captions that mention at least one hallucinated object.
Ground truth object presence comes from COCO instances annotations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set


# Condensed CHAIR synonym table for the 80 COCO categories.  The official
# CHAIR implementation ships a longer list; this subset covers the common
# surface forms used by COCO captions.
SYNONYMS: Dict[str, List[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "guy", "guys", "lady", "ladies", "baby", "babies", "adult", "adults", "human", "humans", "someone", "somebody", "everyone", "pedestrian", "pedestrians", "rider", "riders", "skater", "skaters", "skier", "skiers", "surfer", "surfers"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cycle", "cycles", "cyclist", "cyclists"],
    "car": ["car", "cars", "automobile", "automobiles", "vehicle", "vehicles", "auto", "autos", "cab", "cabs", "taxi", "taxis", "sedan", "sedans", "convertible", "convertibles", "suv", "suvs", "jeep", "jeeps", "wagon", "wagons"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "moped", "mopeds", "scooter", "scooters", "biker", "bikers"],
    "airplane": ["airplane", "airplanes", "aeroplane", "aeroplanes", "plane", "planes", "jet", "jets", "aircraft", "airliner", "airliners", "aircrafts"],
    "bus": ["bus", "buses", "busses", "coach", "coaches"],
    "train": ["train", "trains", "railway", "railways", "railroad", "railroads", "locomotive", "locomotives", "carriage", "carriages"],
    "truck": ["truck", "trucks", "lorry", "lorries", "pickup", "pickups", "semi", "semis", "tractor", "tractors"],
    "boat": ["boat", "boats", "ship", "ships", "sailboat", "sailboats", "canoe", "canoes", "kayak", "kayaks", "yacht", "yachts", "vessel", "vessels", "ferry", "ferries", "cruise", "cruises", "sail", "sails", "dinghy", "dinghies"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights", "signal", "signals"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters", "meter", "meters"],
    "bench": ["bench", "benches", "seat", "seats", "stool", "stools"],
    "bird": ["bird", "birds", "pigeon", "pigeons", "seagull", "seagulls", "gull", "gulls", "crow", "crows", "sparrow", "sparrows", "goose", "geese", "duck", "ducks", "swan", "swans", "owl", "owls", "parrot", "parrots", "chicken", "chickens", "rooster", "roosters", "hen", "hens", "eagle", "eagles", "hawk", "hawks", "turkey", "turkeys", "penguin", "penguins", "flamingo", "flamingos"],
    "cat": ["cat", "cats", "kitten", "kittens", "kitty", "kitties", "feline", "felines"],
    "dog": ["dog", "dogs", "puppy", "puppies", "pup", "pups", "canine", "canines", "pooch", "pooches"],
    "horse": ["horse", "horses", "pony", "ponies", "foal", "foals", "stallion", "stallions", "mare", "mares", "colt", "colts", "equestrian", "equestrians"],
    "sheep": ["sheep", "lamb", "lambs", "ram", "rams", "ewe", "ewes", "flock", "flocks"],
    "cow": ["cow", "cows", "calf", "calves", "bull", "bulls", "cattle", "ox", "oxen", "heifer", "heifers", "bovine", "bovines"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears", "grizzly", "grizzlies", "polar bear", "polar bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "knapsack", "knapsacks", "rucksack", "rucksacks", "pack", "packs", "bag", "bags", "satchel", "satchels"],
    "umbrella": ["umbrella", "umbrellas", "parasol", "parasols", "brolly"],
    "handbag": ["handbag", "handbags", "purse", "purses", "pocketbook", "pocketbooks", "clutch", "clutches"],
    "tie": ["tie", "ties", "necktie", "neckties", "bow tie", "bow ties"],
    "suitcase": ["suitcase", "suitcases", "luggage", "baggage", "trunk", "trunks", "briefcase", "briefcases"],
    "frisbee": ["frisbee", "frisbees", "disc", "discs", "disk", "disks"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "sports balls", "ball", "balls", "football", "footballs", "soccer ball", "soccer balls", "basketball", "basketballs", "baseball", "baseballs", "tennis ball", "tennis balls", "volleyball", "volleyballs", "beach ball", "beach balls", "golf ball", "golf balls"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves", "mitt", "mitts"],
    "skateboard": ["skateboard", "skateboards", "skateboarder", "skateboarders"],
    "surfboard": ["surfboard", "surfboards", "surfboarder", "surfboarders"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottle", "bottles", "jar", "jars", "flask", "flasks", "container", "containers", "vase", "vases"],
    "wine glass": ["wine glass", "wine glasses", "glass", "glasses", "goblet", "goblets"],
    "cup": ["cup", "cups", "mug", "mugs", "teacup", "teacups", "coffee cup", "coffee cups"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "frankfurter", "frankfurters", "sausage", "sausages"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes", "pastry", "pastries", "dessert", "desserts", "cupcake", "cupcakes"],
    "chair": ["chair", "chairs", "seat", "seats", "stool", "stools", "armchair", "armchairs", "sofa", "sofas", "couch", "couches", "bench", "benches", "recliner", "recliners"],
    "couch": ["couch", "couches", "sofa", "sofas", "settee", "settees"],
    "potted plant": ["potted plant", "potted plants", "plant", "plants", "flowerpot", "flowerpots", "pot", "pots", "houseplant", "houseplants"],
    "bed": ["bed", "beds", "mattress", "mattresses", "bunk", "bunks", "cot", "cots"],
    "dining table": ["dining table", "dining tables", "table", "tables", "desk", "desks", "counter", "counters", "coffee table", "coffee tables"],
    "toilet": ["toilet", "toilets", "bathroom", "bathrooms", "lavatory", "lavatories", "urinal", "urinals", "restroom", "restrooms"],
    "tv": ["tv", "tvs", "television", "televisions", "screen", "screens", "monitor", "monitors", "television set", "television sets", "flat screen", "flat screens"],
    "laptop": ["laptop", "laptops", "notebook", "notebooks", "computer", "computers", "macbook", "macbooks"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes", "remote control", "remote controls", "clicker", "clickers"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "mobile phone", "mobile phones", "smartphone", "smartphones", "phone", "phones", "iphone", "iphones"],
    "microwave": ["microwave", "microwaves", "microwave oven", "microwave ovens"],
    "oven": ["oven", "ovens", "stove", "stoves", "range", "ranges", "cooker", "cookers"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks", "basin", "basins"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges", "freezer", "freezers"],
    "book": ["book", "books", "notebook", "notebooks", "novel", "novels", "magazine", "magazines", "textbook", "textbooks", "paperback", "paperbacks", "volume", "volumes"],
    "clock": ["clock", "clocks", "watch", "watches", "alarm clock", "alarm clocks", "timepiece", "timepieces"],
    "vase": ["vase", "vases", "urn", "urns"],
    "scissors": ["scissors", "scissor"],
    "teddy bear": ["teddy bear", "teddy bears", "stuffed animal", "stuffed animals", "teddy", "teddies", "plush", "plushies"],
    "hair drier": ["hair drier", "hair dryers", "hairdryer", "hairdryers", "blow dryer", "blow dryers", "blowdryer", "blowdryers"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}


def _image_id_from_path(path: str) -> int:
    return int(Path(path).stem)


def load_gt_categories(instances_json: str) -> Dict[int, Set[str]]:
    """Map image id -> set of COCO category names annotated as present."""
    data = json.loads(Path(instances_json).read_text())
    category_id_to_name = {
        item["id"]: item["name"] for item in data["categories"]
    }
    result: Dict[int, Set[str]] = {}
    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        result.setdefault(image_id, set()).add(
            category_id_to_name[annotation["category_id"]]
        )
    return result


def detect_mentioned_categories(caption: str) -> Set[str]:
    """Return COCO categories mentioned by any synonym in the caption."""
    text = " " + caption.lower() + " "
    mentioned = set()
    for category, synonyms in SYNONYMS.items():
        for synonym in synonyms:
            pattern = r"\b" + re.escape(synonym) + r"s?\b"
            if re.search(pattern, text):
                mentioned.add(category)
                break
    return mentioned


def compute_chair(
    generated_captions: Dict[int, str],
    gt_categories: Dict[int, Set[str]],
) -> Dict[str, float]:
    """CHAIR_s and CHAIR_i over images with at least one mentioned object."""
    mentioned_total = 0
    hallucinated_total = 0
    captions_with_mention = 0
    captions_with_hallucination = 0
    for image_id, caption in generated_captions.items():
        mentioned = detect_mentioned_categories(caption)
        if not mentioned:
            continue
        captions_with_mention += 1
        present = gt_categories.get(image_id, set())
        hallucinated = mentioned - present
        mentioned_total += len(mentioned)
        hallucinated_total += len(hallucinated)
        if hallucinated:
            captions_with_hallucination += 1
    return {
        "chair_s": hallucinated_total / mentioned_total if mentioned_total else float("nan"),
        "chair_i": (
            captions_with_hallucination / captions_with_mention
            if captions_with_mention
            else float("nan")
        ),
    }
