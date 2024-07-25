from loguru import logger
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torchvision.models as models
from tqdm import tqdm

from model.kidney_classifier import ImageClassifier


def final_test(classifier_net, device, dataset, compute_cm: bool = True, verbose: bool = False):
    num_correct = 0
    y_preds = []
    y_gt = []
    with torch.no_grad():
        for x, y in tqdm(iter(dataset), total=len(dataset)):
            y_gt += [y]
            y_pred = classifier_net(x.to(device).unsqueeze(0))
            y_pred = torch.nn.Softmax(dim=0)(y_pred.squeeze())
            y_pred = np.argmax(y_pred.to('cpu').numpy())
            y_preds += [y_pred]
            num_correct += y_pred == y

    if verbose:
        print(y_preds)
        print(f'Accuracy = {num_correct / len(dataset) * 100:1.2f}%')
        if compute_cm:
            cm = confusion_matrix(y_gt, y_preds)
            print('Confusion matrix\n', cm)

    return num_correct / len(dataset)


def train_epoch(classifier_net, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    classifier_net.train()
    train_loss = 0.0
    num_imgs = 0
    for x, y in dataloader:
        # Move tensor to the proper device
        x = x.to(device)
        y_pred = classifier_net(x)
        # Evaluate loss
        loss = torch.nn.CrossEntropyLoss()(y_pred, y.to(device).squeeze())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        logger.debug(f'\t partial train loss (single batch): {loss.item():%1.4f}')
        train_loss += loss.item()
        num_imgs += len(y)

    return train_loss / num_imgs


def test_epoch(classifier_net, device, dataloader):
    # Set evaluation mode for encoder and decoder
    classifier_net.eval()
    val_loss = 0.0
    num_correct = 0
    num_imgs = 0
    with torch.no_grad():
        for x, y in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            y_pred = classifier_net(x)
            loss = torch.nn.CrossEntropyLoss()(y_pred, y.to(device).squeeze())
            val_loss += loss.item()

            y_pred = torch.nn.Softmax(dim=0)(y_pred.squeeze())
            y_pred = np.argmax(y_pred.to('cpu').numpy(), axis=1)
            num_correct += np.count_nonzero(y_pred == y.numpy())
            num_imgs += len(y)

    return val_loss / num_imgs, num_correct / num_imgs


def create_network(variant: str, num_cls: int) -> torch.nn.Module:
    if variant == 'resnet50':
        network = models.resnet50(weights=None, num_classes=num_cls)
    elif variant == 'resnet50-pretrained':
        network = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in network.parameters():
            param.requires_grad = False
        network.fc = torch.nn.Linear(2048, num_cls)
        network.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=2048, out_features=1024, bias=True),
            torch.nn.Linear(in_features=1024, out_features=num_cls, bias=True)
        )
        network.fc.requires_grad = True
    elif variant == 'resnet101-pretrained':
        network = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        network.fc = torch.nn.Linear(2048, num_cls)
    elif variant == 'resnet101':
        network = models.resnet101(weights=None, num_classes=num_cls)
    elif variant == 'efficient_net':
        network = models.efficientnet_v2_m(weights=None, num_classes=num_cls)
    elif variant == 'vit-pretrained':
        network = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        network.heads = torch.nn.Sequential(torch.nn.Linear(in_features=768, out_features=num_cls, bias=True))
    elif variant == 'efficient_net-pretrained':
        network = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        network.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.3, inplace=True),
                                                 torch.nn.Linear(in_features=1280, out_features=num_cls))
    elif variant == 'alex_net':
        network = models.alexnet(pretrained=False, num_classes=num_cls)
    else:
        network = ImageClassifier(num_classes=num_cls)

    return network
