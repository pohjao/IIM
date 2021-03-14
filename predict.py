import argparse
from typing import Iterable, List
import torchvision
import torchvision.transforms as standard_transforms

import torch 

from model.locator import Crowd_locator

mean_std = ([0.446139603853, 0.409515678883, 0.395083993673], [0.288205742836, 0.278144598007, 0.283502370119])
img_transform = standard_transforms.Compose(standard_transforms.Normalize(*mean_std))

def iterable_data(filepath:str) -> Iterable:
    if filepath.lower().endswith(".mp4"):
        return torchvision.io.VideoReader(filepath)
    else:
        raise NotImplementedError("Input file format not currently supported.")

def load_model(weights_path: str, net_name: str, gpu_id) -> Crowd_locator:
    locator = Crowd_locator(net_name,gpu_id,pretrained=True)
    locator.cuda()

    return locator

def get_boxInfo_from_Binar_map(Binar_numpy, min_area=3):
    Binar_numpy = Binar_numpy.squeeze().astype(np.uint8)
    assert Binar_numpy.ndim == 2
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

    boxes = stats[1:, :]
    points = centroids[1:, :]
    index = (boxes[:, 4] >= min_area)
    boxes = boxes[index]
    points = points[index]
    pre_data = {'num': len(points), 'points': points}
    return pre_data, boxes

def main(input_file:str, model_weights_pth: str, gpu_id: str, net_name: str):
    locator = load_model(model_weights_pth, net_name, gpu_id)
    for input in iterable_data(input_file):
        with torch.no_grad():
            img = img_transform(input)
            [pred_threshold, pred_map, __] = [i.cpu() for i in locator(img, mask_gt=None, mode='val')]
            
            a = torch.ones_like(pred_map)
            b = torch.zeros_like(pred_map)
            binar_map = torch.where(pred_map >= pred_threshold, a, b)
            pred_data, boxes = get_boxInfo_from_Binar_map(binar_map.cpu().numpy())

if __name__ == 'main':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", help="Input file. Image or video.", Required=True)
    argparser.add_argument("--gpu",default=0, help="Id of the gpu used.")
    argparser.add_argument("--net", default="HR_Net", help="'HR_Net' or 'VGG16_FPN'")
    argparser.add_argument("-w", "--weights", help="Path to model weights file.", Required=True)
    args = argparser.parse_args()
    main(args.input, args.weigths, args.gpu, args.net)