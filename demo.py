import argparse

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pathlib import Path

from utils.utils import *
from utils.models import *

from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--content_dir',
        type=str,
        default='C:/Users/DELL/OneDrive/Desktop/NST_Code/content_data',
        help='Location of content dataset'
    )

    parser.add_argument(
        '--style_dir',
        type=str,
        default='C:/Users/DELL/OneDrive/Desktop/NST_Code/style_data',
        help='Location of style dataset'
    )

    parser.add_argument(
        '--vgg',
        type=str,
        default='C:/Users/DELL/OneDrive/Desktop/NST_Code/vgg_normalised.pth',
        help='Location of pre-trained VGG'
    )

    parser.add_argument(
        '--experiment',
        type=str,
        default='experiment1',
        help='Name of experiment'
    )

    parser.add_argument(
        '--final_size',
        type=int,
        default=256,
        help='Size of final image'
    )

    parser.add_argument(
        '--content_size',
        type=int,
        default=512,
        help='Size of content image'
    )

    parser.add_argument(
        '--style_size',
        type=int,
        default=512,
        help='Size of style image'
    )

    parser.add_argument(
        '--crop',
        action='store_true',
        default=True,
        help='Crop image'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )

    parser.add_argument(
        '--lr_decay',
        type=float,
        default=5e-5,
        help='Learning rate decay'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Number of additional epochs to train'
    )

    parser.add_argument(
        '--content_weight',
        type=float,
        default=1.0,
        help='Content weight'
    )

    parser.add_argument(
        '--style_weight',
        type=float,
        default=5,
        help='Style weight'
    )

    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='Log interval'
    )

    parser.add_argument(
        '--save_interval',
        type=int,
        default=2,
        help='Save interval'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume training'
    )

    parser.add_argument(
        '--decoder_path',
        type=str,
        default=None,
        help='Path to decoder checkpoint'
    )

    parser.add_argument(
        '--optimizer_path',
        type=str,
        default=None,
        help='Path to optimizer checkpoint'
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Experiment directory
    # --------------------------------------------------

    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:

        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    # --------------------------------------------------
    # Transforms
    # --------------------------------------------------

    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    print(
        'Number of batches in content dataset: ',
        len(content_dataloader)
    )

    print(
        'Number of batches in style dataset: ',
        len(style_dataloader)
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    encoder = VGGEncoder(args.vgg).to(device)

    decoder = Decoder().to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    # --------------------------------------------------
    # Resume information
    # --------------------------------------------------

    start_epoch = 0

    if args.resume:

        if args.decoder_path is None:
            raise ValueError(
                "Please provide --decoder_path when using --resume"
            )

        if args.optimizer_path is None:
            raise ValueError(
                "Please provide --optimizer_path when using --resume"
            )

        # Load decoder weights
        decoder.load_state_dict(
            torch.load(
                args.decoder_path,
                map_location=device
            )
        )

        # Load optimizer state
        optimizer.load_state_dict(
            torch.load(
                args.optimizer_path,
                map_location=device
            )
        )

        # Extract epoch number from filename
        #
        # Example:
        # decoder_160.pth
        #
        # becomes:
        # 160

        start_epoch = int(
            Path(args.decoder_path).stem.split('_')[-1]
        )

        print()
        print("=" * 50)
        print("RESUMING TRAINING")
        print("=" * 50)
        print(f"Checkpoint: {args.decoder_path}")
        print(f"Starting from epoch: {start_epoch}")
        print(f"Next epoch: {start_epoch + 1}")
        print("=" * 50)
        print()

    else:

        print()
        print("=" * 50)
        print("STARTING TRAINING FROM SCRATCH")
        print("=" * 50)
        print()

    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            1.0 / (
                1.0 +
                args.lr_decay *
                (start_epoch + epoch)
            )
        )
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    mse_loss = torch.nn.MSELoss()

    encoder.eval()

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("Training...")

    for epoch in range(
        start_epoch,
        start_epoch + args.epochs
    ):

        progress_bar = tqdm(
            zip(
                content_dataloader,
                style_dataloader
            ),
            total=min(
                len(content_dataloader),
                len(style_dataloader)
            )
        )

        running_loss = 0
        running_closs = 0
        running_sloss = 0

        # --------------------------------------------------
        # Batches
        # --------------------------------------------------

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)

            style_batch = style_batch.to(device)

            # --------------------------------------------------
            # Extract VGG features
            # --------------------------------------------------

            c_feats = encoder(content_batch)

            s_feats = encoder(style_batch)

            # --------------------------------------------------
            # Adaptive Instance Normalization
            # --------------------------------------------------

            t = adaptive_instance_normalization(
                c_feats[-1],
                s_feats[-1]
            )

            # --------------------------------------------------
            # Decoder
            # --------------------------------------------------

            g = decoder(t)

            # --------------------------------------------------
            # Generated image features
            # --------------------------------------------------

            g_feats = encoder(g)

            # --------------------------------------------------
            # Content loss
            # --------------------------------------------------

            loss_c = (
                mse_loss(
                    g_feats[-1],
                    t
                )
                * args.content_weight
            )

            # --------------------------------------------------
            # Style loss
            # --------------------------------------------------

            loss_s = 0

            for g_f, s_f in zip(
                g_feats,
                s_feats
            ):

                g_mean, g_std = calc_mean_std(g_f)

                s_mean, s_std = calc_mean_std(s_f)

                loss_s += (
                    mse_loss(
                        g_mean,
                        s_mean
                    )
                    +
                    mse_loss(
                        g_std,
                        s_std
                    )
                )

            loss_s = (
                loss_s *
                args.style_weight
            )

            # --------------------------------------------------
            # Total loss
            # --------------------------------------------------

            loss = loss_c + loss_s

            # --------------------------------------------------
            # Backpropagation
            # --------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # --------------------------------------------------
            # Progress bar
            # --------------------------------------------------

            progress_bar.set_description(
                f'Loss: {loss.item():.4f}, '
                f'Content Loss: {loss_c.item():.4f}, '
                f'Style Loss: {loss_s.item():.4f}'
            )

            # --------------------------------------------------
            # Running losses
            # --------------------------------------------------

            running_loss += loss.item()

            running_closs += loss_c.item()

            running_sloss += loss_s.item()

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        scheduler.step()

        # --------------------------------------------------
        # Average losses
        # --------------------------------------------------

        running_loss /= len(content_dataloader)

        running_closs /= len(content_dataloader)

        running_sloss /= len(content_dataloader)

        # --------------------------------------------------
        # Logging
        # --------------------------------------------------

        if (epoch + 1) % args.log_interval == 0:

            tqdm.write(
                f'Iter {epoch + 1}: '
                f'Loss: {running_loss:.4f}, '
                f'Content Loss: {running_closs:.4f}, '
                f'Style Loss: {running_sloss:.4f}'
            )

        # --------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------

        if (epoch + 1) % args.save_interval == 0:

            decoder_path = (
                save_dir /
                f'decoder_{epoch + 1}.pth'
            )

            optimizer_path = (
                save_dir /
                f'optimizer_{epoch + 1}.pth'
            )

            output_path = (
                save_dir /
                f'output_{epoch + 1}.png'
            )

            # Save decoder
            torch.save(
                decoder.state_dict(),
                decoder_path
            )

            # Save optimizer
            torch.save(
                optimizer.state_dict(),
                optimizer_path
            )

            # Save sample output
            with torch.no_grad():

                output = torch.cat(
                    [
                        content_batch,
                        style_batch,
                        g
                    ],
                    dim=0
                )

                save_image(
                    output,
                    output_path,
                    nrow=args.batch_size
                )

            print()
            print(
                f'Checkpoint saved at epoch {epoch + 1}'
            )
            print(
                f'Decoder: {decoder_path}'
            )
            print(
                f'Optimizer: {optimizer_path}'
            )
            print()


if __name__ == '__main__':
    main()