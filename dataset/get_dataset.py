from dataset.DiscRegion import Disc_Cup


def getDataset(args):

    if args.dataset == "RIGA":
        train_set = Disc_Cup(args.dataroot, args.batch_size, DF=['BinRushed', 'MESSIDOR'], transform=True)
        test_set = Disc_Cup(args.dataroot, args.batch_size, DF=['Magrabia'])
        return train_set, test_set
