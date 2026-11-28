import os
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix
import cv2
import clip.clip
import torch
from torchvision.transforms import functional as F

from model.TEDFusion_model import TEDFusion1 as create_model1
from scripts.image_output import existing_image_names, reserve_image_stem, write_image
from scripts.inference import crop_to_original, pad_image_pair


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
    os.makedirs(save_path, exist_ok=True)

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
    output_names = _output_names(visible_path, save_path)

    print("Begin to run!")
    with torch.no_grad():
        checkpoint = torch.load(args.weights_path, map_location="cpu", weights_only=False)
        curvature = getattr(args, "curvature", None)
        if curvature is None:
            curvature = checkpoint.get("curvature", 1.0)
        model_weights = checkpoint["model"]
        model_weights.pop("conv48to384.weight", None)
        model_weights.pop("conv48to384.bias", None)
        del checkpoint

        #model_clip, _ = clip.load("ViT-B/32", device=device)
        model_clip, _ = clip.load("ViT-B/32", device=device, download_root=os.path.dirname(os.path.realpath(__file__)))

        model = create_model1(model_clip, curvature=curvature).to(device)
        model.load_state_dict(model_weights)
        del model_weights
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = output_names[vi_path]

        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        with Image.open(ir_path) as source:
            ir = source.convert(mode="RGB")
        with Image.open(vi_path) as source:
            vi = source.convert(mode="RGB")
        if vi.size != ir.size:
            raise ValueError("Original paired image sizes must match: {} {} and {} {}".format(
                vi_path, vi.size, ir_path, ir.size
            ))

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)



        ir = ir.unsqueeze(0).to(device)
        vi = vi.unsqueeze(0).to(device)

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
        texture_complexity1 = _texture_complexity(vi, (0.6, 0.4))
        contrast1 = calculate_contrast(vi.detach()).cpu().numpy()

        h2, s2, v2 = mergy_RGB_to_HSV(ir)
        s_avg2 = s2.mean(axis=(1, 2))
        v_avg2 = v2.mean(axis=(1, 2))
        texture_complexity2 = _texture_complexity(ir, (0.7, 0.3))
        contrast2 = calculate_contrast(ir.detach()).cpu().numpy()

        feature = torch.as_tensor(
            np.stack([
                s_avg1, v_avg1, texture_complexity1, contrast1,
                s_avg2, v_avg2, texture_complexity2, contrast2,
            ], axis=1).astype(np.float32),
            dtype=torch.float32, device=vi.device,
        )

        # Measure attributes on the original pixels; padding is only a model
        # shape requirement and must not change the exported field of view.
        vi, ir, original_size = pad_image_pair(vi, ir)
        with torch.no_grad():
            fused = model(vi, ir, feature)
            fused_img_Y = tensor2numpy(crop_to_original(fused, original_size))

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

def _output_names(visible_paths, save_path):
    used_names = existing_image_names(save_path)
    names = {}
    collisions = []
    # Reserve preferred names before allocating aliases. A native PNG takes priority.
    for path in sorted(visible_paths, key=lambda name: Path(name).suffix == ".jpg"):
        source = Path(path)
        suffix = ".png" if source.suffix == ".jpg" else source.suffix
        preferred_name = source.stem + suffix
        key = preferred_name.casefold()
        if key in used_names:
            collisions.append((path, source.stem, suffix))
        else:
            names[path] = preferred_name
            used_names.add(key)

    for path, stem, suffix in collisions:
        stem = reserve_image_stem(stem, (suffix,), used_names)
        names[path] = stem + suffix
    return names


def save_pic(outputpic, path, index: str):
    # Use the same absolute-intensity quantization as validation. Per-image
    # min-max stretching changes the experiment's brightness and contrast.
    outputpic = np.rint(np.clip(outputpic, 0.0, 1.0) * 255.0).astype(np.uint8)
    outputpic = outputpic[:, :, ::-1]
    write_image(os.path.join(path, index), outputpic)


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
    """Compute per-image texture complexity using Appendix A, Eqs. (10)-(13)."""
    levels = np.arange(256)
    homogeneity_weights = 1.0 / (1 + np.abs(levels[:, None] - levels[None, :]))
    values = []
    for gray in rgb_to_gray(images.detach())[:, 0].cpu().numpy():
        # Keep both spatial dimensions for the four GLCM directions.
        gray = np.rint(np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
        glcm = graycomatrix(
            gray, [2, 4, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
            256, symmetric=True, normed=True,
        )
        energy = np.square(glcm).sum(axis=(0, 1)).mean()
        homogeneity = np.einsum('ijda,ij->da', glcm, homogeneity_weights).mean()
        values.append(weights[0] * energy + weights[1] * homogeneity)
    return np.asarray(values, dtype=np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='trusted local model checkpoint')
    parser.add_argument('--curvature', type=float, default=None, help='override the checkpoint curvature')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')
    # parser.add_argument('--input_text', type=str, required=True, help='text control input')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)
