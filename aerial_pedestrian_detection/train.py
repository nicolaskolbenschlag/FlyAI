import torch.utils.data
import torch.nn
import torch.cuda
import torch
import argparse

import dataset
import utils

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)

    params = parser.parse_args()
    return params


def train(args: argparse.Namespace, model: torch.nn.Module = None) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    dataset_train = dataset.StanfordDroneDataset(
        root="data/sdd_voc",
        image_set="trainval",
        transforms=utils.get_transforms(train=True)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    
    if model is None:
        model = utils.get_model(num_classes=7)
    model.to(device)
    
    for i, (images, targets, _) in enumerate(dataloader):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss = model(images, targets)
        
        print(loss)
        exit()

if __name__ == "__main__":
    args = parse_args()
    train(args)