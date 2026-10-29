"""Keep every annotated instance of an image in the same partition."""

import random
from settings import SPLIT_SEED


def split_annotation_ids(coco, seed=SPLIT_SEED):
    image_ids = sorted({ann['image_id'] for ann in coco.anns.values()})
    random.Random(seed).shuffle(image_ids)
    train_end = int(0.70 * len(image_ids))
    val_end = train_end + int(0.15 * len(image_ids))
    image_groups = {
        'train': set(image_ids[:train_end]),
        'val': set(image_ids[train_end:val_end]),
        'test': set(image_ids[val_end:]),
    }
    if any(not group for group in image_groups.values()):
        raise ValueError('The 70/15/15 split requires at least seven annotated images.')
    return {
        name: [ann_id for ann_id in sorted(coco.anns)
               if coco.anns[ann_id]['image_id'] in group]
        for name, group in image_groups.items()
    }


def class_names(coco):
    return [coco.cats[category_id]['name'] for category_id in sorted(coco.cats)]
