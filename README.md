# SUL

Simple but useful layers

Whatever.

Now updated to SUL3.0 for tf2

Examples in './example/'

## Documentation

[The documentation can be viewed here.](https://sul.readthedocs.io/en/latest/)

### Projects

- ArcFace (Deng, Jiankang, et al. "Arcface: Additive angular margin loss for deep face recognition." arXiv preprint arXiv:1801.07698 (2018))

- Enforced Softmax (Cheng, Yu, et al. "Know you at one glance: A compact vector representation for low-shot learning." Proceedings of the IEEE International Conference on Computer Vision. 2017)

- MobileFaceNet (Chen, Sheng, et al. "Mobilefacenets: Efficient cnns for accurate real-time face verification on mobile devices." Chinese Conference on Biometric Recognition. Springer, Cham, 2018)

- Cycle GAN (Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE International Conference on Computer Vision. 2017)

- PSM Net (Chang, Jia-Ren, and Yong-Sheng Chen. "Pyramid stereo matching network." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018)

- Stacked Hourglass (Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." European Conference on Computer Vision. Springer, Cham, 2016)

- Aging face (Zhao, Jian, et al. "Look Across Elapse: Disentangled Representation Learning and Photorealistic Cross-Age Face Synthesis for Age-Invariant Face Recognition." arXiv preprint arXiv:1809.00338 (2018))

- Graph Convolution Net (Kipf, Thomas N., and Max Welling. "Semi-supervised classification with graph convolutional networks." arXiv preprint arXiv:1609.02907 (2016))

- Single stage object detection framework (Redmon, Joseph, and Ali Farhadi. "YOLO9000: better, faster, stronger." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017)

- OctConv (Chen, Yunpeng, et al. "Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution." arXiv preprint arXiv:1904.05049 (2019))

- 2D Human Pose Detector (Xiao, Bin, Haiping Wu, and Yichen Wei. "Simple baselines for human pose estimation and tracking." Proceedings of the European Conference on Computer Vision (ECCV). 2018)

- HR Net (Sun, Ke, et al. "Deep High-Resolution Representation Learning for Human Pose Estimation." arXiv preprint arXiv:1902.09212 (2019))

- Semantic segmentation (I cannot find reference. It's just softmax cross entropy on output.)

- FaceSwap

- MNIST (Everthing begins here)

- Model conversion (from torch to tf)


#### Note

To install, run
```
pip install tf-nightly-gpu-2.0-preview
```

Remember to change registry if using Windows.

1. Start the registry editor (regedit.exe)

2. Navigate to HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem

3. Double click LongPathsEnabled, set to 1 and click OK

4. Reboot

#### Legacy 

SUL1 for Tensorflow 1.x

SUL2 for Tensorflow >= 1.12
