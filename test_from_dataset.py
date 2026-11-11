import os
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import cv2
import clip.clip
import torch
from torchvision.transforms import functional as F

from model.TEDFusion_model import TEDFusion1 as create_model1


import argparse

import random
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def set_seed_thread(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    # set_seed_thread(200)

    root_path = args.dataset_path
    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    # text_line = args.input_text

    visible_root = os.path.join(root_path, "Visible")
    infrared_root = os.path.join(root_path, "Infrared")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model1(model_clip).to(device)

        model_weight_path = args.weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]

        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        height, width = vi.size

        new_width = (width // 16) * 16

        new_height = (height // 16) * 16

        # new_width = (width * 16.0) / 16.0
        #
        # new_height = (height * 16.0) / 16.0


        ir = ir.resize((new_height, new_width))
        vi = vi.resize((new_height, new_width))

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)



        ir = ir.unsqueeze(0).cuda()
        vi = vi.unsqueeze(0).cuda()

        _check_rgb_batch(vi)
        _check_rgb_batch(ir)
        if vi.shape != ir.shape:
            raise ValueError("Paired image shapes must match, got {} and {}".format(
                tuple(vi.shape), tuple(ir.shape)
            ))

        # Match training and model(vi, ir, feature): visible first, infrared second.
        h1, s1, v1 = mergy_RGB_to_HSV(vi)
        s_avg1 = s1.mean(axis=(1, 2))
        v_avg1 = v1.mean(axis=(1, 2))
        texture_complexity1 = _texture_complexity(vi, (0.7, 0.3))
        contrast1 = calculate_contrast(vi.detach()).cpu().numpy()

        h2, s2, v2 = mergy_RGB_to_HSV(ir)
        s_avg2 = s2.mean(axis=(1, 2))
        v_avg2 = v2.mean(axis=(1, 2))
        texture_complexity2 = _texture_complexity(ir, (0.6, 0.4))
        contrast2 = calculate_contrast(ir.detach()).cpu().numpy()

        feature = torch.as_tensor(
            np.stack([
                s_avg1, v_avg1, texture_complexity1, contrast1,
                s_avg2, v_avg2, texture_complexity2, contrast2,
            ], axis=1).astype(np.float32),
            dtype=torch.float32, device=vi.device,
        )

        with torch.no_grad():
            # text = clip.tokenize(text_line).to(device)
            # text1 = clip.tokenize(text_line[0]).to(device)
            # text2 = clip.tokenize(text_line[1]).to(device)
            # text3 = clip.tokenize(text_line[2]).to(device)
            # text4 = clip.tokenize(text_line[3]).to(device)
            i = model(vi, ir,feature)

            text1 = torch.tensor(0)
            text2 = torch.tensor(0)
            # i = model(vi, ir, feature,text1,text2)
            fused_img_Y = tensor2numpy(i)

            save_pic(fused_img_Y, save_path, img_name)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor[0].squeeze(0).cpu().detach().numpy()
    # img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    cv2.imwrite(save_path, outputpic)


def _check_rgb_batch(images):
    if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
        raise ValueError("Expected a nonempty RGB batch [B,3,H,W], got {}".format(
            tuple(images.shape)
        ))


def mergy_RGB_to_HSV(img):
    """Return H, S and V arrays, each with shape [B,H,W]."""
    _check_rgb_batch(img)
    images = img.detach().cpu().float().numpy()
    hsv = np.stack([
        cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2HSV)
        for image in images
    ], axis=0)
    return hsv[..., 0], hsv[..., 1], hsv[..., 2]


def rgb_to_gray(rgb_tensor):
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1)


def calculate_contrast(tensor):
    """Return the mean channel-wise spatial standard deviation per image [B]."""
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    var = ((tensor - mean) ** 2).mean(dim=(2, 3))
    return torch.sqrt(var).mean(dim=1)


def calculate_edge(tensor):
    """Return mean spatial-gradient magnitude per image [B], for H,W >= 2."""
    if tensor.ndim != 4 or min(tensor.shape[-2:]) < 2:
        raise ValueError("Expected [B,C,H,W] with H,W >= 2, got {}".format(
            tuple(tensor.shape)
        ))
    images = tensor.detach().cpu().float().numpy()
    grad_y, grad_x = np.gradient(images, axis=(-2, -1))
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return magnitude.mean(axis=(1, 2, 3))


def _texture_complexity(images, weights):
    """Apply the existing single-image GLCM calculation independently."""
    values = []
    for image in images.detach():
        # Preserve the original B=1 spatial reduction and quantization. Only
        # remove aggregation across different batch samples in this repair.
        gray = rgb_to_gray(image.unsqueeze(0)).sum(dim=0).sum(dim=1)
        gray = (gray.cpu().numpy() * 255).astype(np.uint8)
        glcm = graycomatrix(
            gray, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            256, symmetric=True, normed=True,
        ).astype(np.float32)
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        values.append(weights[0] * energy + weights[1] * homogeneity)
    return np.asarray(values, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')
    # parser.add_argument('--input_text', type=str, required=True, help='text control input')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)
