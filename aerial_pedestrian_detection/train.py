import torch.utils.data
import torch.nn
import torch.cuda
import torch
import argparse
import logging
import tqdm
import os

import dataset
import utils

log = logging.getLogger(__name__)
log.setLevel(logging.NOTSET)
log.addHandler(utils.TqdmLoggingHandler())

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)

    params = parser.parse_args()
    return params


def train(args: argparse.Namespace, model: torch.nn.Module = None) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    log.debug(f"device: {device}")

    dataset_train = dataset.StanfordDroneDataset(
        root="data/sdd_voc",
        image_set="trainval",
        transforms=utils.get_transforms(train=True)
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=utils.collate_fn
    )
    
    if model is None:
        model = utils.get_model(num_classes=7)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=.005, momentum=.9, weight_decay=.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=.1)
    
    for epoch in range(args.epochs):
        for i_batch, (images, targets, _) in enumerate(tqdm.tqdm(data_loader, total=len(data_loader))):
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            # TODO reduce dict (https://github.com/pytorch/vision/blob/master/references/detection/utils.py) to log single losses
            log.info(f"Loss: {loss.item()}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()
    
    torch.save(model.state_dict(), os.join("models", "model-01"))


if __name__ == "__main__":
    args = parse_args()
    train(args)