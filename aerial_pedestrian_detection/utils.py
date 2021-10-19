import matplotlib.patches
import matplotlib.pyplot as plt
import torch.nn
import torchvision.models.detection
import torchvision.models.detection.faster_rcnn
import torchvision.transforms.transforms
import albumentations
import albumentations.pytorch.transforms

def plot_bboxes(image, target) -> None:
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(image)
    for box in target["boxes"]:
        x, y, width, height  = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="r", facecolor="none")
        a.add_patch(rect)
    plt.show()

def get_model(num_classes: int) -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def get_transforms(train):
    """
    Use this: https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    """
    transforms = []
    if train:
        transforms += [
            albumentations.HorizontalFlip()
        ]

    transforms += [
        albumentations.Resize(480, 480),
        albumentations.transforms.Normalize(),
        albumentations.pytorch.transforms.ToTensorV2()
    ]

    return albumentations.Compose(
        transforms=transforms,
        bbox_params=albumentations.BboxParams(format="pascal_voc", label_fields=["class_labels"])
    )

def collate_fn(batch):
    return tuple(zip(*batch))