# train.py
import torch
import torchvision
from engine import train_one_epoch, evaluate
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

def train_model(model, data_loader, data_loader_valid, device, num_epochs,
                lr=0.005, momentum=0.9, weight_decay=0.0005, step_size=3, gamma=0.1):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_valid, device=device)

    #torch.save(model.state_dict(), 'results/models/model_weights.pth')
