import argparse
from run import train, test


def main():

    parser = argparse.ArgumentParser()

    """
    out_dir; dataset; train or test; network;
    """
    parser.add_argument("--output_dir", type=str, default="outputs",  # don't change it
                        help="The output path.")
    parser.add_argument("--dataroot", type=str, default='/media/userdisk0/Dataset/DiscRegion',
                        help="The path of the dataset")
    parser.add_argument("--dataset", choices=["RIGA"], default="RIGA",
                        help="Which downstream task.")
    parser.add_argument("--rater_num", type=int, default=6,
                        help="number of rater.")
    parser.add_argument("--phase", choices=["train", "test"], default="train",
                        help="phase: train or only test?")
    parser.add_argument("--net_arch",
                        choices=["PADL"],
                        default="PADL",
                        help="Which network to use.")
    parser.add_argument("--loss_func", choices=["dice", "ce", "bce"], default="bce",
                        help="which loss function to use.")

    """
    pretrained params
    """
    parser.add_argument("--pretrained", type=int, default=0,
                        help="whether to load pretrained models.")
    parser.add_argument("--pretrained_dir", type=str, default="none",
                        help="the path of pretrained models.")

    """
    img for network input
    """
    parser.add_argument("--img_width", default=256, type=int,
                        help="Resolution size")
    parser.add_argument("--img_height", default=256, type=int,
                        help="Resolution size")
    parser.add_argument("--img_channel", default=3, type=int,
                        help="channel size")

    """
    training settings: classes; bs; lr; EPOCH; device_id
    """
    parser.add_argument("--num_classes", default=2, type=int,
                        help="the number of classes for pixel-wise classification")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Total batch size for training, validation and testing.")
    parser.add_argument("--learning_rate", default=7e-4, type=float,
                        help="The initial learning rate of optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="The weight_decay of Adam optimizer.")
    parser.add_argument("--power", default=0.9, type=float,
                        help="the hyper-parameter of poly learning rate adjust")
    parser.add_argument("--num_epoch",  default=60, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--device_id", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"],
                        help="gpu ID.")

    """
    dataset setting: k_fold(loop;fold); random(ratio) --> to find .csv file under the dataset dir.
    """
    parser.add_argument("--data_split", default="official", choices=["k_fold", "official"],
                        help="k_fold: needs to set --fold; official: needs to set nothing")
    parser.add_argument("--loop", default=0, type=int,
                        help="this is the {loop}-th run.")

    args = parser.parse_args()
    print(args)

    if args.phase == "train":
        train(args)
    else:  # test
        test(args)


if __name__ == "__main__":
    main()
