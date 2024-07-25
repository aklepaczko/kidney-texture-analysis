"""Train classification model.

Usage:
  train.py [options] <patient-id> <train-dir>
  train.py -h | --help
  train.py --version

Arguments:
  <patient-id>         Patient ID folder with train, validation, test split files.
                       Note, we do not check if train/valid/test overlap.
  <train-dir>          Location where train images are stored.

Options:
  --optimizer <value>  Name of the optimization algorithm [default: Adam].
  --log-dir <value>    Location to store monitoring logs and final checkpoint [default: logs].
  --test-dir <value>   Location where test image are stored (<train-dir> is used if not provided).
  --weights <value>    Path to checkpoint with initial weights to start from. Note, you have to match the chosen
                       architecture.
  -h --help            Show this screen.
  --version            Show version.
"""
from pathlib import Path
from typing import Any, Optional

from docopt import docopt
from loguru import logger
import torch
from torch.optim.lr_scheduler import MultiStepLR, MultiplicativeLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dixon_vibe_dataset import DixonVibeDataset
from utils.train_utils import create_network, final_test, train_epoch, test_epoch


def main(args: dict[str, Optional[Any]]):
    logger.debug(args)

    train_dir = Path(args['<train-dir>'])

    patient_id = Path(args['<patient-id>'])

    train_patients_filepath = patient_id / f'{patient_id.stem}_train.txt'
    with open(train_patients_filepath, 'r') as f:
        train_ids = f.readlines()
    train_ids = [int(x.rstrip()) for x in train_ids]
    categories = [2, 3]
    train_dataset = DixonVibeDataset(root_path=train_dir, patients=train_ids, categories=categories, augment=True)

    valid_patients_filepath = patient_id / f'{patient_id.stem}_valid.txt'
    with open(valid_patients_filepath, 'r') as f:
        valid_ids = f.readlines()
    valid_ids = [int(x.rstrip()) for x in valid_ids]
    valid_dataset = DixonVibeDataset(root_path=train_dir, patients=valid_ids, categories=categories, augment=False)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    test_patients_filepath = patient_id / f'{patient_id.stem}_test.txt'
    with open(test_patients_filepath, 'r') as f:
        test_ids = f.readlines()
    test_ids = [int(x.rstrip()) for x in test_ids]
    if args['--test-dir']:
        test_dir = Path(args['--test-dir'])
    else:
        test_dir = train_dir
    test_dataset = DixonVibeDataset(root_path=test_dir, patients=test_ids, categories=categories, augment=False)

    # torch.manual_seed(0)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    variant = 'resnet50-pretrained'
    num_cls = len(categories)
    network = create_network(variant, num_cls)

    lr_initial = 1e-7

    opt_str = args['--optimizer']

    if opt_str == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(), lr=lr_initial, weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=lr_initial, weight_decay=1e-5)

    lr_scheduler = MultiStepLR(optimizer, milestones=[40, 60, 80], gamma=0.1)  # lrsched=1
    # lr_scheduler = MultiStepLR(optimizer, milestones=[10, 50, 70], gamma=0.5)  # lrsched=2
    # lrlmbd = lambda epoch_no: 1.2 if epoch_no < 10 else 0.98
    # lr_scheduler = MultiplicativeLR(optimizer, lr_lambda=lrlmbd)  # lrsched=3

    # lrlmbd = lambda epoch_no: 1.5 if epoch_no < 10 else 0.995
    # lr_scheduler = MultiplicativeLR(optimizer, lr_lambda=lrlmbd)  # lrsched=4

    logger.info(f'Selected device: {device}')

    network.to(device)

    num_epochs = 100
    min_loss = 1e6

    logdir = train_dir.parent / args['--log-dir']
    runs = list(logdir.glob(f'{Path(train_patients_filepath).stem}*-run*'))
    if runs:
        runs = [str(x) for x in runs]
        run_ids = [int(x[int(x.rfind('_'))+1:]) for x in runs]
        run = max(run_ids) + 1
    else:
        run = 0

    logdir = logdir / f'{Path(train_patients_filepath).stem}-{variant}-{opt_str}-run_{run}'
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logdir))

    for epoch in range(num_epochs):
        train_loss = train_epoch(network, device, train_loader, optimizer)
        val_loss, val_acc = test_epoch(network, device, valid_loader)
        test_acc = final_test(network, device, test_dataset, compute_cm=False)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', val_loss, epoch)
        writer.add_scalar('Accuracy/valid', val_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('Learning rate', lr_scheduler.get_lr()[0], epoch)
        logger.debug('\n EPOCH {}/{} \t train loss {:.4f} \t val loss {:.4f}'.format(epoch + 1,
                                                                                     num_epochs,
                                                                                     train_loss,
                                                                                     val_loss))
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(network.state_dict(), logdir / f'best_model-run_{run}.pth')
            # best_weights = network.state_dict()

        lr_scheduler.step()
        writer.flush()

    writer.close()

    network.load_state_dict(torch.load(logdir / f'best_model-run_{run}.pth'))
    network.eval()

    final_test(network, device, test_dataset, compute_cm=True, verbose=True)


if __name__ == '__main__':
    main(docopt(__doc__, version='train.py 0.1.0'))
