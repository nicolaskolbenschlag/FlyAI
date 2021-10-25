import os
import collections
import torch
import torchvision.datasets
from xml.etree.ElementTree import Element as ET_Element
from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List
import numpy as np
import sklearn.preprocessing


class StanfordDroneDataset(torchvision.datasets.VisionDataset):
    """
    Adopts this code https://pytorch.org/vision/stable/_modules/torchvision/datasets/voc.html for the SDD dataset in Pascal VOC like style.
    """
    def __init__(
        self,
        root: str,
        image_set: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, transforms, transform, target_transform)

        self.class_label_encoder = sklearn.preprocessing.LabelEncoder().fit(["_", "Pedestrian", "Biker", "Cart", "Skater", "Car", "Bus"])

        valid_image_sets = ["train", "trainval", "val", "test"]# NOTE trainval = train + val
        if image_set not in valid_image_sets:
            raise ValueError(f"Image set {image_set} not in supported image sets {valid_image_sets}.")

        splits_dir = os.path.join(self.root, "ImageSets", "Main")
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
            
        image_dir = os.path.join(self.root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(self.root, "Annotations")
        self.targets = [os.path.join(target_dir, x + ".xml") for x in file_names]

        assert len(self.images) == len(self.targets)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert("RGB")
        image = np.array(image)

        labels_file = self.parse_voc_xml(ET_parse(self.targets[index]).getroot())

        objects = labels_file["annotation"]["object"]
        bboxes, class_labels = [], []
        
        for object in objects:
            
            # NOTE sort out if out of border (causes trouble in augmentations elsewise)
            if int(object["bndbox"]["xmin"]) < 0\
                or int(object["bndbox"]["ymin"]) < 0\
                or int(object["bndbox"]["xmax"]) > image.shape[1]\
                or int(object["bndbox"]["ymax"]) > image.shape[0]:
                continue

            bboxes += [
                [int(num) for num in list(object["bndbox"].values())]
            ]
            class_labels += [object["name"]]
                
        if self.transforms is not None:
            augmentations = self.transforms(image=image, bboxes=bboxes, class_labels=class_labels)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            class_labels = augmentations["class_labels"]

        target = {
            "boxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(self.class_label_encoder.transform(class_labels), dtype=torch.int64)
        }
        return image, target, class_labels

    def parse_voc_xml(self, node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


if __name__ == "__main__":
    import utils

    dataset = StanfordDroneDataset(
        root="data/sdd_voc",
        image_set="test",#"trainval",
        transforms=utils.get_transforms(train=True)
    )    

    image, target, labels = dataset[-1]
    image = image.numpy().transpose(1, 2, 0)

    utils.plot_bboxes(image, {"boxes": target["boxes"]})