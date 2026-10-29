import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
import os

from PIL import Image

# import torchvision.transforms as T

class MyCOCODataset(data.Dataset):
    def __init__(self, data_root, annofile, ids=None, output_size=(192, 192)):
        """
        data_root: Directory that contains subfolders like "JPEGImages"
        annofile: Path to the annotations.json file
        ids: list of annotation IDs to include (for train, val, or test)
        output_size: (height, width) to resize the cropped instances
        """
        self.data_root = data_root
        self.annofile = annofile
        self.coco = COCO(str(annofile))
        self.category_ids = sorted(self.coco.cats)
        self.category_to_label = {cat_id: label for label, cat_id in enumerate(self.category_ids)}
        self.class_names = [self.coco.cats[cat_id]['name'] for cat_id in self.category_ids]
        if not self.category_ids:
            raise ValueError('The annotations file must contain categories.')

        if ids is None:
            # If no subset is provided, use all annotation IDs
            self.instance_ids = sorted(self.coco.anns)
        else:
            self.instance_ids = ids

        self.instances = self.coco.anns
        self.output_size = output_size

        # (Optional) Define transforms
        # self.transform = T.Compose([
        #     T.Resize(self.output_size),  # or T.RandomResizedCrop(self.output_size)
        #     T.RandomHorizontalFlip(p=0.5),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])
        # ])

    def __getitem__(self, index):
        # Retrieve annotation
        ann_id = self.instance_ids[index]
        ann = self.instances[ann_id]

        # Get image ID and bounding box
        imgid = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]

        # Load image file name
        img_file = self.coco.loadImgs([imgid])[0]["file_name"]
        # Replace any Windows-style backslashes with forward slashes
        img_file = img_file.replace("\\", "/")

        # Build the absolute path
        img_path = os.path.join(self.data_root, img_file)

        # Open the image
        with Image.open(img_path) as image:
            img = image.convert("RGB")

        # Crop the instance from the bounding box
        _bbox_int = list(map(int, bbox))
        img = img.crop((_bbox_int[0],
                        _bbox_int[1],
                        _bbox_int[0] + _bbox_int[2],
                        _bbox_int[1] + _bbox_int[3]))

        # Resize the image
        img = img.resize((self.output_size[1], self.output_size[0]), Image.Resampling.BILINEAR)

        # Ensure RGB
        if img.mode == "L":
            img = img.convert("RGB")

        img = np.array(img, dtype=np.uint8)
        category = np.int64(self.category_to_label[ann["category_id"]])

        return img, category

    def __len__(self):
        return len(self.instance_ids)

    # Utility function to print all categories
    def print_all_categories(self):
        for cat_id, cat_info in self.coco.cats.items():
            print(f'ID: {cat_id}, Label: {cat_info["name"]}')


if __name__ == "__main__":
    from settings import DATA_ROOT, ANNOTATIONS
    dataset = MyCOCODataset(DATA_ROOT, ANNOTATIONS)
    dataset.print_all_categories()
    print("Dataset length:", len(dataset))
