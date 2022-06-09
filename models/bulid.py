from models.res_unet.unet_model import ResUNet, PADL


def build_model(args):
    """
    return models
    """
    if args.net_arch == "res_unet":
        model = ResUNet(resnet='resnet34', num_classes=args.num_classes, pretrained=True)
    elif args.net_arch == "PADL":
        model = PADL(resnet='resnet34', num_classes=args.num_classes, rater_num=args.rater_num, pretrained=True)
    return model
