# SUL

Simple but useful layers

Whatever.

Now updated to SUL3.0 for tf2

Examples in './example/'

### Projects

- ArcFace (Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018))

- Enforced Softmax (Cheng, Yu, et al. "Know you at one glance: A compact vector representation for low-shot learning." Proceedings of the IEEE International Conference on Computer Vision. 2017)

- Cycle GAN (Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE International Conference on Computer Vision. 2017)

- PSM Net (Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018)

- Stacked Hourglass (Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." European Conference on Computer Vision. Springer, Cham, 2016)

- Aging face (Zhao, Jian, et al. "Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition." arXiv preprint arXiv:1809.00338 (2018))

- Graph Convolution Net (Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016))

- Single stage object detection framework (Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017)

- Semantic segmentation (I cannot find reference. It's just softmax cross entropy on output.)

- Object edge segmentation (No reference)

- MNIST (Everthing begins here)

- Model conversion (from torch to tf)

- Multi-GPU training

##### Note

To install, run
```
pip install tf-nightly-gpu-2.0-preview
```

Remember to change registry if using Windows.

1. Start the registry editor (regedit.exe)

2. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem

3. Double click LongPathsEnabled, set to 1 and click OK

4. Reboot
