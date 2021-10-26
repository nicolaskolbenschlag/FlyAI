import torch
import os

import utils
import dataset

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = utils.get_model(num_classes=7, pretrained=False)
    model.load_state_dict(torch.load(os.path.join("aerial_pedestrian_detection", "models", "model.pth"), map_location=device))
    model.eval()

    ds = dataset.StanfordDroneDataset(
        root="data/sdd_voc",
        image_set="test",
        transforms=utils.get_transforms(train=False)
    )

    idx = 0
    image, target, labels = ds[idx]
    image.to(device)

    with torch.no_grad():
        prediction = model(image.unsqueeze(0))[0]
    