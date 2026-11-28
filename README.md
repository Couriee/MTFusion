# TEDFusion
Text-Driven Fusion for Infrared and Visible Images: Achieving Image Scene Adaptation on Hyperbolic Space
====
Accetped by [ICML 2026] 🔗"https://arxiv.org/pdf/2606.15104"


⬇️Model Download
----
The model can be found in https://pan.baidu.com/s/1Daz11onyLza_jLuXcBy1mA?pwd=h5b6 and the password is: h5b6

🖥️Environment
----
python==3.12.7

pytorch==2.5.1

pytorch-cuda==12.6

scipy==1.13.1

numpy==1.26.4


⚙️Training
----
```bash
python train_fusion.py `
  --dataset_path "./dataset/train_MSRS" `
  --samples_per_epoch 800 `
  --epochs 15 `
  --lr 1e-4 `
  --weight_decay 1e-4 `
  --grad_clip_norm 1.0
```

Validation now uses every full-size image pair by default (`--max_val_samples 0`)
and runs the same attribute-only path as standalone inference. A positive cap
selects deterministic samples across tasks rather than a filename prefix.
Inference pads and crops to preserve source size, and exports absolute intensity.

New checkpoints record experiment protocol version 2. Continue these experiments
with `--resume`; omitted epoch, batch-size, sampling and optimizer settings are
restored and conflicting settings are rejected. To initialize a new experiment
from an older checkpoint, use `--weights` instead. The hyperbolic objective and
inference protocol changed; previous metrics require a new evaluation.
See [the before/after repair record](修复日志.md) for implementation and validation.


✔️Testing
----
For inputs with standard image dimensions, please use the following command for testing:
```bash
python test_from_dataset.py  --weights_path "pretrained_weights/train/TEDFusion.pth" --dataset_path "./dataset/test_LLVIP" --save_path "./results/LLVIP"
```

📖Citation
----
If you are interested in our work, please cite it in the following format:
```bash
@inproceedings{pmlr-v306-kang26k,
  title     = {Text-Driven Fusion for Infrared and Visible Images: Achieving Image Scene Adaptation on Hyperbolic Space},
  author    = {Kang, Huan and Li, Hui and Xu, Tianyang and Zhou, Tao and Wu, Xiaojun and Kittler, Josef},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {55658--55671},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/kang26k.html}
}
```
