import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches

import utils
import dataset

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = utils.get_model(num_classes=7)
    model.load_state_dict(torch.load(os.path.join("aerial_pedestrian_detection", "models", "model.pth"), map_location=device))
    model.eval()

    ds = dataset.StanfordDroneDataset(
        root="data/sdd_voc",
        image_set="test",
        transforms=utils.get_transforms(train=False)
    )

    idx = 10
    image, target, labels = ds[idx]
    image.to(device)

    with torch.no_grad():
        prediction = model(image.unsqueeze(0))[0]
            
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(10,10)
    a.imshow(image.numpy().transpose(1,2,0))
    
    for box in target["boxes"]:
        x, y, width, height  = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="orange", facecolor="none")
        a.add_patch(rect)
    
    for box in prediction["boxes"]:
        x, y, width, height  = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), width, height, linewidth=2, edgecolor="r", facecolor="none")
        a.add_patch(rect)

    plt.show()