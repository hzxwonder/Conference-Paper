## [800] Light Field Neural Rendering

**Authors**: *Mohammed Suhail, Carlos Esteves, Leonid Sigal, Ameesh Makadia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00809](https://doi.org/10.1109/CVPR52688.2022.00809)

**Abstract**:

Classical light field rendering for novel view synthesis can accurately reproduce view-dependent effects such as reflection, refraction, and translucency, but requires a dense view sampling of the scene. Methods based on geometric reconstruction need only sparse views, but cannot accurately model non-Lambertian effects. We introduce a model that combines the strengths and mitigates the limitations of these two directions. By operating on a four-dimensional representation of the light field, our model learns to represent view-dependent effects accurately. By enforcing geometric constraints during training and inference, the scene geometry is implicitly learned from a sparse set of views. Concretely, we introduce a two-stage transformer-based model that first aggregates features along epipolar lines, then aggregates features along reference views to produce the color of a target ray. Our model outperforms the state-of-the-art on multiple forward-facing and 360° datasets, with larger margins on scenes with severe view-dependent variations. Code and results can be found at light-field-neural-rendering.github. io.

----

## [801] Extracting Triangular 3D Models, Materials, and Lighting From Images

**Authors**: *Jacob Munkberg, Wenzheng Chen, Jon Hasselgren, Alex Evans, Tianchang Shen, Thomas Müller, Jun Gao, Sanja Fidler*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00810](https://doi.org/10.1109/CVPR52688.2022.00810)

**Abstract**:

We present an efficient method for joint optimization of topology, materials and lighting from multi-view image observations. Unlike recent multi-view reconstruction approaches, which typically produce entangled 3D representations encoded in neural networks, we output triangle meshes with spatially-varying materials and environment lighting that can be deployed in any traditional graphics engine unmodified. We leverage recent work in differentiable rendering, coordinate-based networks to compactly represent volumetric texturing, alongside differentiable marching tetrahedrons to enable gradient-based optimization directly on the surface mesh. Finally, we introduce a differentiable formulation of the split sum approximation of environment lighting to efficiently recover all-frequency lighting. Experiments show our extracted models used in advanced scene editing, material decomposition, and high quality view interpolation, all running at interactive rates in triangle-based renderers (rasterizers and path tracers).

----

## [802] Super-Fibonacci Spirals: Fast, Low-Discrepancy Sampling of SO(3)

**Authors**: *Marc Alexa*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00811](https://doi.org/10.1109/CVPR52688.2022.00811)

**Abstract**:

Super-Fibonacci spirals are an extension of Fibonacci spirals, enabling fast generation of an arbitrary but fixed number of 3D orientations. The algorithm is simple and fast. A comprehensive evaluation comparing to other meth-ods shows that the generated sets of orientations have low discrepancy, minimal spurious components in the power spectrum, and almost identical Voronoi volumes. This makes them useful for a variety of applications, in partic-ular Monte Carlo sampling.

----

## [803] Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models

**Authors**: *Feng Cheng, Mingze Xu, Yuanjun Xiong, Hao Chen, Xinyu Li, Wei Li, Wei Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00812](https://doi.org/10.1109/CVPR52688.2022.00812)

**Abstract**:

We propose a memory efficient method, named Stochastic Backpropagation (SBP), for training deep neural networks on videos. It is based on the finding that gradients from incomplete execution for backpropagation can still effectively train the models with minimal accuracy loss, which attributes to the high redundancy of video. SBP keeps all forward paths but randomly and independently removes the backward paths for each network layer in each training step. It reduces the GPU memory cost by eliminating the need to cache activation values corresponding to the dropped backward paths, whose amount can be controlled by an adjustable keep-ratio. Experiments show that SBP can be applied to a wide range of models for video tasks, leading to up to 80.0% GPU memory saving and 10% training speedup with less than 1% accuracy drop on action recognition and temporal action detection.

----

## [804] It's All In the Teacher: Zero-Shot Quantization Brought Closer to the Teacher

**Authors**: *Kanghyun Choi, Hyeyoon Lee, Deokki Hong, Joonsang Yu, Noseong Park, Youngsok Kim, Jinho Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00813](https://doi.org/10.1109/CVPR52688.2022.00813)

**Abstract**:

Model quantization is considered as a promising method to greatly reduce the resource requirements of deep neural networks. To deal with the performance drop induced by quantization errors, a popular method is to use training data to fine-tune quantized networks. In real-world environments, however, such a method is frequently infeasible because training data is unavailable due to security, privacy, or confidentiality concerns. Zero-shot quantization addresses such problems, usually by taking information from the weights of a full-precision teacher network to compensate the performance drop of the quantized networks. In this paper, we first analyze the loss surface of state-of-the-art zero-shot quantization techniques and provide several findings. In contrast to usual knowledge distillation problems, zero-shot quantization often suffers from 1) the difficulty of optimizing multiple loss terms together, and 2) the poor generalization capability due to the use of synthetic samples. Furthermore, we observe that many weights fail to cross the rounding threshold during training the quantized networks even when it is necessary to do so for better performance. Based on the observations, we propose AIT, a simple yet powerful technique for zero-shot quantization, which addresses the aforementioned two problems in the following way: AIT i) uses a KL distance loss only without a cross-entropy loss, and ii) manipulates gradients to guarantee that a certain portion of weights are properly updated after crossing the rounding thresholds. Experiments show that AIT outperforms the performance of many existing methods by a great margin, taking over the overall state-of-the-art position in the field.

----

## [805] NLX-GPT: A Model for Natural Language Explanations in Vision and Vision-Language Tasks

**Authors**: *Fawaz Sammani, Tanmoy Mukherjee, Nikos Deligiannis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00814](https://doi.org/10.1109/CVPR52688.2022.00814)

**Abstract**:

Natural language explanation (NLE) models aim at explaining the decision-making process of a black box system via generating natural language sentences which are human-friendly, high-level and fine-grained. Current NLE models
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Throughout this paper, we refer to NLE models as Natural Language Explanation models aimed for vision and vision-language tasks. explain the decision-making process of a vision or vision-language model (a.k.a., task model), e.g., a VQA model, via a language model (a.k.a., explanation model), e.g., GPT. Other than the additional memory resources and inference time required by the task model, the task and explanation models are completely independent, which disassociates the explanation from the reasoning process made to predict the answer. We introduce NLX-GPT, a general, compact and faithful language model that can simultaneously predict an answer and explain it. We first conduct pre-training on large scale data of image-caption pairs for general understanding of images, and then formulate the answer as a text prediction task along with the explanation. Without region proposals nor a task model, our resulting overall framework attains better evaluation scores, contains much less parameters and is 15× faster than the current SoA model. We then address the problem of evaluating the explanations which can be in many times generic, data-biased and can come in several forms. We therefore design 2 new evaluation measures: (1) explain-predict and (2) retrieval-based attack, a selfevaluation framework that requires no labels. Code is at: https://github.com/fawazsammani/nlxgpt.

----

## [806] Explaining Deep Convolutional Neural Networks via Latent Visual-Semantic Filter Attention

**Authors**: *Yu Yang, Seungbae Kim, Jungseock Joo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00815](https://doi.org/10.1109/CVPR52688.2022.00815)

**Abstract**:

Interpretability is an important property for visual mod-els as it helps researchers and users understand the in-ternal mechanism of a complex model. However, gener-ating semantic explanations about the learned representation is challenging without direct supervision to produce such explanations. We propose a general framework, La-tent Visual Semantic Explainer (LaViSE), to teach any ex-isting convolutional neural network to generate text de-scriptions about its own latent representations at the filter level. Our method constructs a mapping between the vi-sual and semantic spaces using generic image datasets, using images and category names. It then transfers the map-ping to the target domain which does not have semantic la-bels. The proposedframework employs a modular structure and enables to analyze any trained network whether or not its original training data is available. We show that our method can generate novel descriptions for learned filters beyond the set of categories defined in the training dataset and perform an extensive evaluation on multiple datasets. We also demonstrate a novel application of our method for unsupervised dataset bias analysis which allows us to auto-matically discover hidden biases in datasets or compare dif-ferent subsets without using additional labels. The dataset and code are made public to facilitate further research.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/YuYang0901/LaViSE

----

## [807] Parameter-free Online Test-time Adaptation

**Authors**: *Malik Boudiaf, Romain Müller, Ismail Ben Ayed, Luca Bertinetto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00816](https://doi.org/10.1109/CVPR52688.2022.00816)

**Abstract**:

Training state-of-the-art vision models has become prohibitively expensive for researchers and practitioners. For the sake of accessibility and resource reuse, it is important to focus on adapting these models to a variety of down-stream scenarios. An interesting and practical paradigm is online test-time adaptation, according to which training data is inaccessible, no labelled data from the test distribution is available, and adaptation can only happen at test time and on a handful of samples. In this paper, we investigate how test-time adaptation methods fare for a number of pre-trained models on a variety of real-world scenarios, significantly extending the way they have been originally evaluated. We show that they perform well only in narrowly-defined experimental setups and sometimes fail catastrophically when their hyperparameters are not selected for the same scenario in which they are being tested. Motivated by the inherent uncertainty around the conditions that will ultimately be encountered at test time, we propose a particularly “conservative” approach, which addresses the problem with a Laplacian Adjusted Maximum-likelihood Estimation (LAME) objective. By adapting the model's output (not its parameters), and solving our objective with an efficient concave-convex procedure, our approach exhibits a much higher average accuracy across scenarios than existing methods, while being notably faster and have a much lower memory footprint. The code is available at https://github.com/fiveai/LAME.

----

## [808] Patch-level Representation Learning for Self-supervised Vision Transformers

**Authors**: *Sukmin Yun, Hankook Lee, Jaehyung Kim, Jinwoo Shin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00817](https://doi.org/10.1109/CVPR52688.2022.00817)

**Abstract**:

Recent self-supervised learning (SSL) methods have shown impressive results in learning visual representations from unlabeled images. This paper aims to improve their performance further by utilizing the architectural advan-tages of the underlying neural network, as the current state-of-the-art visual pretext tasks for SSL do not enjoy the ben-efit, i.e., they are architecture-agnostic. In particular, we fo-cus on Vision Transformers (ViTs), which have gained much attention recently as a better architectural choice, often out-performing convolutional networks for various visual tasks. The unique characteristic of ViT is that it takes a sequence of disjoint patches from an image and processes patch-level representations internally. Inspired by this, we design a simple yet effective visual pretext task, coined Self Patch, for learning better patch-level representations. To be specific, we enforce invariance against each patch and its neigh-bors, i.e., each patch treats similar neighboring patches as positive samples. Consequently, training ViTs with Self-Patch learns more semantically meaningful relations among patches (without using human-annotated labels), which can be beneficial, in particular, to downstream tasks of a dense prediction type. Despite its simplicity, we demonstrate that it can significantly improve the performance of existing SSL methods for various visual tasks, including object detection and semantic segmentation. Specifically, Self Patch signif-icantly improves the recent self-supervised ViT, DINO, by achieving +1.3 AP on COCO object detection, +1.2 AP on COCO instance segmentation, and +2.9 mIoU on ADE20K semantic segmentation.

----

## [809] Deep Spectral Methods: A Surprisingly Strong Baseline for Unsupervised Semantic Segmentation and Localization

**Authors**: *Luke Melas-Kyriazi, Christian Rupprecht, Iro Laina, Andrea Vedaldi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00818](https://doi.org/10.1109/CVPR52688.2022.00818)

**Abstract**:

Unsupervised localization and segmentation are long-standing computer vision challenges that involve decom-posing an image into semantically meaningful segments without any labeled data. These tasks are particularly interesting in an unsupervised setting due to the difficulty and cost of obtaining dense image annotations, but existing un-supervised approaches struggle with complex scenes containing multiple objects. Differently from existing methods, which are purely based on deep learning, we take inspiration from traditional spectral segmentation methods by re-framing image decomposition as a graph partitioning problem. Specifically, we examine the eigenvectors of the Laplacian of a feature affinity matrix from self-supervised networks. We find that these eigenvectors already decompose an image into meaningful segments, and can be readily used to localize objects in a scene. Furthermore, by clustering the features associated with these segments across a dataset, we can obtain well-delineated, nameable regions, i.e. semantic segmentations. Experiments on complex datasets (PASCAL VOC, MS-COCO) demonstrate that our simple spectral method outperforms the state-of-the-art in unsupervised localization and segmentation by a significant margin. Furthermore, our method can be readily usedfor a variety of complex image editing tasks, such as background removal and compositing.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project Page: https://lukemelas.github.io/deep-spectral-segmentation/

----

## [810] Mixed Differential Privacy in Computer Vision

**Authors**: *Aditya Golatkar, Alessandro Achille, Yu-Xiang Wang, Aaron Roth, Michael Kearns, Stefano Soatto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00819](https://doi.org/10.1109/CVPR52688.2022.00819)

**Abstract**:

We introduce AdaMix, an adaptive differentially private algorithm for training deep neural network classifiers using both private and public image data. While pre-training language models on large public datasets has enabled strong differential privacy (DP) guarantees with minor loss of accuracy, a similar practice yields punishing trade-offs in vision tasks. A few-shot or even zero-shot learning baseline that ignores private data can outperform fine-tuning on a large private dataset. AdaMix incorporates few-shot training, or cross-modal zero-shot learning, on public data prior to private fine-tuning, to improve the trade-off. AdaMix reduces the error increase from the non-private upper bound from the 167–311% of the baseline, on average across 6 datasets, to 68-92% depending on the desired privacy level selected by the user. AdaMix tackles the trade-off arising in visual classification, whereby the most privacy sensitive data, corresponding to isolated points in representation space, are also critical for high classification accuracy. In addition, AdaMix comes with strong theoretical privacy guarantees and convergence analysis.

----

## [811] DPGEN: Differentially Private Generative Energy-Guided Network for Natural Image Synthesis

**Authors**: *Jia-Wei Chen, Chia-Mu Yu, Ching-Chia Kao, Tzai-Wei Pang, Chun-Shien Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00820](https://doi.org/10.1109/CVPR52688.2022.00820)

**Abstract**:

Despite an increased demand for valuable data, the privacy concerns associated with sensitive datasets present a barrier to data sharing. One may use differentially private generative models to generate synthetic data. Unfortunately, generators are typically restricted to generating images of low-resolutions due to the limitation of noisy gradients. Here, we propose DPGEN, a network model designed to synthesize high-resolution natural images while satisfying differential privacy. In particular, we propose an energy-guided network trained on sanitized data to indicate the direction of the true data distribution via Langevin Markov chain Monte Carlo (MCMC) sampling method. In contrast to the state-of-the-art methods that can process only low-resolution images (e.g., MNIST and Fashion-MNIST), DPGEN can generate differentially private synthetic images with resolutions up to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$128\times 128$</tex>
 with superior visual quality and data utility. Our code is available at https://github.com/chiamuyu/DPGEN

----

## [812] Local Learning Matters: Rethinking Data Heterogeneity in Federated Learning

**Authors**: *Matías Mendieta, Taojiannan Yang, Pu Wang, Minwoo Lee, Zhengming Ding, Chen Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00821](https://doi.org/10.1109/CVPR52688.2022.00821)

**Abstract**:

Federated learning (FL) is a promising strategy for performing privacy-preserving, distributed learning with a network of clients (i.e., edge devices). However, the data distribution among clients is often non-IID in nature, making efficient optimization difficult. To alleviate this issue, many FL algorithms focus on mitigating the effects of data heterogeneity across clients by introducing a variety of proximal terms, some incurring considerable compute and/or memory overheads, to restrain local updates with respect to the global model. Instead, we consider rethinking solutions to data heterogeneity in FL with a focus on local learning generality rather than proximal restriction. To this end, we first present a systematic study informed by second-order indicators to better understand algorithm effectiveness in FL. Interestingly, we find that standard regularization methods are surprisingly strong performers in mitigating data heterogeneity effects. Based on our findings, we further propose a simple and effective method, FedAlign, to overcome data heterogeneity and the pitfalls of previous methods. FedAlign achieves competitive accuracy with state-of-the-art FL methods across a variety of settings while minimizing computation and memory overhead. Code is available at https://github.com/mmendiet/FedAlign.

----

## [813] AirObject: A Temporally Evolving Graph Embedding for Object Identification

**Authors**: *Nikhil Varma Keetha, Chen Wang, Yuheng Qiu, Kuan Xu, Sebastian A. Scherer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00822](https://doi.org/10.1109/CVPR52688.2022.00822)

**Abstract**:

Object encoding and identification are vital for robotic tasks such as autonomous exploration, semantic scene understanding, and relocalization. Previous approaches have attempted to either track objects or generate descriptors for object identification. However, such systems are limited to a “fixed” partial object representation from a single viewpoint. In a robot exploration setup, there is a requirement for a temporally “evolving” global object representation built as the robot observes the object from multiple viewpoints. Furthermore, given the vast distribution of unknown novel objects in the real world, the object identification process must be class-agnostic. In this context, we propose a novel temporal 3D object encoding approach, dubbed AirObject, to obtain global keypoint graph-based embeddings of objects. Specifically, the global 3D object embeddings are generated using a temporal convolutional network across structural information of multiple frames obtained from a graph attention-based encoding method. We demonstrate that AirObject achieves the state-of-the-art performance for video object identification and is robust to severe occlusion, perceptual aliasing, viewpoint shift, deformation, and scale transform, outperforming the state-of-the-art single-frame and sequential descriptors. To the best of our knowledge, AirObject is one of the first temporal object encoding methods. Source code is available at https://github.com/Nik-v9/AirObject.

----

## [814] Voxel Set Transformer: A Set-to-Set Approach to 3D Object Detection from Point Clouds

**Authors**: *Chenhang He, Ruihuang Li, Shuai Li, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00823](https://doi.org/10.1109/CVPR52688.2022.00823)

**Abstract**:

Transformer has demonstrated promising performance in many 2D vision tasks. However, it is cumbersome to compute the self-attention on large-scale point cloud data because point cloud is a long sequence and unevenly distributed in 3D space. To solve this issue, existing methods usually compute self-attention locally by grouping the points into clusters of the same size, or perform convolutional self-attention on a discretized representation. However, the former results in stochastic point dropout, while the latter typically has narrow attention fields. In this paper, we propose a novel voxel-based architecture, namely Voxel Set Transformer (VoxSeT), to detect 3D objects from point clouds by means of set-to-set translation. VoxSeT is built upon a voxel-based set attention (VSA) module, which reduces the self-attention in each voxel by two cross-attentions and models features in a hidden space induced by a group of latent codes. With the VSA module, VoxSeT can manage voxelized point clusters with arbitrary size in a wide range, and process them in parallel with linear complexity. The proposed VoxSeT integrates the high performance of transformer with the efficiency of voxel-based model, which can be used as a good alternative to the convolutional and point-based backbones. VoxSeT reports competitive results on the KITTI and Waymo detection benchmarks. The source codes can be found at https://github.com/skyhehe123/VoxSeT.

----

## [815] SS3D: Sparsely-Supervised 3D Object Detection from Point Cloud

**Authors**: *Chuandong Liu, Chenqiang Gao, Fangcen Liu, Jiang Liu, Deyu Meng, Xinbo Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00824](https://doi.org/10.1109/CVPR52688.2022.00824)

**Abstract**:

Conventional deep learning based methods for 3D object detection require a large amount of 3D bounding box annotations for training, which is expensive to obtain in practice. Sparsely annotated object detection, which can largely reduce the annotations, is very challenging since the missing-annotated instances would be regarded as the background during training. In this paper, we propose a sparsely-supervised 3D object detection method, named SS3D. Aiming to eliminate the negative supervision caused by the missing annotations, we design a missing-annotated instance mining module with strict filtering strategies to mine positive instances. In the meantime, we design a reliable background mining module and a point cloud filling data augmentation strategy to generate the confident data for iteratively learning with reliable supervision. The proposed SS3D is a general framework that can be used to learn any modern 3D object detector. Extensive experiments on the KITTI dataset reveal that on different 3D detectors, the proposed SS3D framework with only 20% annotations required can achieve onpar performance comparing to fully-supervised methods. Comparing with the state-of-the-art semi-supervised 3D objection detection on KITTI, our SS3D improves the benchmarks by significant margins under the same annotation workload. Moreover, our SS3D also out-performs the state-of-the-art weakly-supervised method by remarkable margins, highlighting its effectiveness.

----

## [816] Back to Reality: Weakly-supervised 3D Object Detection with Shape-guided Label Enhancement

**Authors**: *Xiuwei Xu, Yifan Wang, Yu Zheng, Yongming Rao, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00825](https://doi.org/10.1109/CVPR52688.2022.00825)

**Abstract**:

In this paper, we propose a weakly-supervised approach for 3D object detection, which makes it possible to train a strong 3D detector with position-level annotations (i.e. annotations of object centers). In order to remedy the information loss from box annotations to centers, our method, namely Back to Reality (BR), makes use of synthetic 3D shapes to convert the weak labels into fully-annotated virtual scenes as stronger supervision, and in turn utilizes the perfect virtual labels to complement and refine the real labels. Specifically, we first assemble 3D shapes into physically reasonable virtual scenes according to the coarse scene layout extracted from position-level annotations. Then we go back to reality by applying a virtual-to-real domain adaptation method, which refine the weak labels and additionally supervise the training of detector with the virtual scenes. Furthermore, we propose a more challenging benckmark for indoor 3D object detection with more diversity in object sizes for better evaluation. With less than 5% of the labeling labor, we achieve comparable detection performance with some popular fully-supervised approaches on the widely used ScanNet dataset. Code is available at: https://github.com/wyf-ACCEPT/BackToReality.

----

## [817] VISTA: Boosting 3D Object Detection via Dual Cross-VIew SpaTial Attention

**Authors**: *Shengheng Deng, Zhihao Liang, Lin Sun, Kui Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00826](https://doi.org/10.1109/CVPR52688.2022.00826)

**Abstract**:

Detecting objects from LiDAR point clouds is of tremendous significance in autonomous driving. In spite of good progress, accurate and reliable 3D detection is yet to be achieved due to the sparsity and irregularity of LiDAR point clouds. Among existing strategies, multi-view methods have shown great promise by leveraging the more comprehensive information from both bird's eye view (BEV) and range view (RV). These multi-view methods either refine the proposals predicted from single view via fused features, or fuse the features without considering the global spatial con-text; their performance is limited consequently. In this paper, we propose to adaptively fuse multi-view features in a global spatial context via Dual Cross- VIew SpaTial Attention (VISTA). The proposed VISTA is a novel plug-and-play fusion module, wherein the multi-layer perceptron widely adopted in standard attention modules is replaced with a convolutional one. Thanks to the learned attention mechanism, VISTA can produce fused features of high quality for prediction of proposals. We decouple the classification and regression tasks in VISTA, and an additional constraint of attention variance is applied that enables the attention module to focus on specific targets instead of generic points. We conduct thorough experiments on the benchmarks of nuScenes and Waymo; results confirm the efficacy of our designs. At the time of submission, our method achieves 63.0% in overall mAP and 69.8% in NDS on the nuScenes benchmark, outperforming all published methods by up to 24% in safety-crucial categories such as cyclist.

----

## [818] Embracing Single Stride 3D Object Detector with Sparse Transformer

**Authors**: *Lue Fan, Ziqi Pang, Tianyuan Zhang, Yu-Xiong Wang, Hang Zhao, Feng Wang, Naiyan Wang, Zhaoxiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00827](https://doi.org/10.1109/CVPR52688.2022.00827)

**Abstract**:

In LiDAR-based 3D object detection for autonomous driving, the ratio of the object size to input scene size is significantly smaller compared to 2D detection cases. Over-looking this difference, many 3D detectors directly follow the common practice of 2D detectors, which downsample the feature maps even after quantizing the point clouds. In this paper, we start by rethinking how such multi-stride stereotype affects the LiDAR-based 3D object detectors. Our experiments point out that the downsampling operations bring few advantages, and lead to inevitable information loss. To remedy this issue, we propose Single-stride Sparse Transformer (SST) to maintain the original resolution from the beginning to the end of the network. Armed with transformers, our method addresses the problem of insufficient receptive field in single-stride architectures. It also cooperates well with the sparsity of point clouds and naturally avoids expensive computation. Eventually, our SST achieves state-of-the-art results on the large-scale Waymo Open Dataset. It is worth mentioning that our method can achieve exciting performance (83.8 LEVEL_1 AP on validation split) on small object (pedestrian) detection due to the characteristic of single stride. Our codes will be public soon.

----

## [819] Point Density-Aware Voxels for LiDAR 3D Object Detection

**Authors**: *Jordan S. K. Hu, Tianshu Kuai, Steven L. Waslander*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00828](https://doi.org/10.1109/CVPR52688.2022.00828)

**Abstract**:

LiDAR has become one of the primary 3D object detection sensors in autonomous driving. However, LiDAR's diverging point pattern with increasing distance results in a non-uniform sampled point cloud ill-suited to discretized volumetric feature extraction. Current methods either rely on voxelized point clouds or use inefficient farthest point sampling to mitigate detrimental effects caused by density variation but largely ignore point density as a feature and its predictable relationship with distance from the LiDAR sensor. Our proposed solution, Point Density-Aware Voxel network (PDV), is an end-to-end two stage LiDAR 3D object detection architecture that is designed to account for these point density variations. PDV efficiently localizes voxel features from the 3D sparse convolution backbone through voxel point centroids. The spatially localized voxel features are then aggregated through a density-aware RoI grid pooling module using kernel density estimation (KDE) and self attention with point density positional encoding. Finally, we exploit LiDAR's point density to distance relationship to refine our final bounding box confidences. PDV outperforms all state-of-the-art methods on the Waymo Open Dataset and achieves competitive results on the KITTI dataset.

----

## [820] Point-to-Voxel Knowledge Distillation for LiDAR Semantic Segmentation

**Authors**: *Yuenan Hou, Xinge Zhu, Yuexin Ma, Chen Change Loy, Yikang Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00829](https://doi.org/10.1109/CVPR52688.2022.00829)

**Abstract**:

This article addresses the problem of distilling knowledge from a large teacher model to a slim student network for LiDAR semantic segmentation. Directly employing previous distillation approaches yields inferior results due to the intrinsic challenges of point cloud, i.e., sparsity, randomness and varying density. To tackle the aforementioned problems, we propose the Point-to-Voxel Knowledge Distillation (PVD), which transfers the hidden knowledge from both point level and voxel level. Specifically, we first leverage both the pointwise and voxelwise output distillation to complement the sparse supervision signals. Then, to better exploit the structural information, we divide the whole point cloud into several supervoxels and design a difficulty-aware sampling strategy to more frequently sample supervoxels containing less-frequent classes and faraway objects. On these supervoxels, we propose inter-point and inter-voxel affinity distillation, where the similarity information between points and voxels can help the student model better capture the structural information of the surrounding environment. We conduct extensive experiments on two popular LiDAR segmentation benchmarks, i.e., nuScenes [3] and SemanticKITTI [1]. On both benchmarks, our PVD-consistently outperforms previous distillation approaches by a large margin on three representative backbones, i.e., Cylinder3D [36], [37], SPVNAS [25] and MinkowskiNet [5]. Notably, on the challenging nuScenes and SemanticKITTI datasets, our method can achieve roughly 75% MACs reduction and 2× speedup on the competitive Cylinder3D model and rank 1st on the SemanticKITTI leaderboard among all published algorithms
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://competitions.codalab.org/competitions/20331#results (single-scan competition) till 2021-11-18 04:00 Pacific Time, and our method is termed Point-Voxel-KD. Our method (PV-KD) ranks 3rd on the multi-scan challenge till 2021-12-1 00:00 Pacific Time.. Our code is available at https://github.com/cardwing/Codes-for-PVKD.

----

## [821] Contrastive Boundary Learning for Point Cloud Segmentation

**Authors**: *Liyao Tang, Yibing Zhan, Zhe Chen, Baosheng Yu, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00830](https://doi.org/10.1109/CVPR52688.2022.00830)

**Abstract**:

Point cloud segmentation is fundamental in understanding 3D environments. However, current 3D point cloud segmentation methods usually perform poorly on scene boundaries, which degenerates the overall segmentation performance. In this paper, we focus on the segmentation of scene boundaries. Accordingly, we first explore metrics to evaluate the segmentation performance on scene boundaries. To address the unsatisfactory performance on boundaries, we then propose a novel contrastive boundary learning (CBL) framework for point cloud segmentation. Specifically, the proposed CBL enhances feature discrimination between points across boundaries by contrasting their representations with the assistance of scene contexts at multiple scales. By applying CBL on three different baseline methods, we experimentally show that CBL consistently improves different baselines and assists them to achieve compelling performance on boundaries, as well as the overall performance, e.g. in mIoU. The experimental results demonstrate the effectiveness of our method and the importance of boundaries for 3D point cloud segmentation. Code and model will be made publicly available at https://github.com/LiyaoTang/contrastBoundary.

----

## [822] Stratified Transformer for 3D Point Cloud Segmentation

**Authors**: *Xin Lai, Jianhui Liu, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00831](https://doi.org/10.1109/CVPR52688.2022.00831)

**Abstract**:

3D point cloud segmentation has made tremendous progress in recent years. Most current methods focus on aggregating local features, but fail to directly model long-range dependencies. In this paper, we propose Stratified Transformer that is able to capture long-range contexts and demonstrates strong generalization ability and high performance. Specifically, we first put forward a novel key sampling strategy. For each query point, we sample nearby points densely and distant points sparsely as its keys in a stratified way, which enables the model to enlarge the effective receptive field and enjoy long-range contexts at a low computational cost. Also, to combat the challenges posed by irregular point arrangements, we propose first-layer point embedding to aggregate local information, which facilitates convergence and boosts performance. Besides, we adopt contextual relative position encoding to adaptively capture position information. Finally, a memory-efficient implementation is introduced to overcome the issue of varying point numbers in each window. Extensive experiments demonstrate the effectiveness and superiority of our method on S3DIS, ScanNetv2 and ShapeNetPart datasets. Code is available at https://github.com/dvlab-research/Stratified-Transformer.

----

## [823] No Pain, Big Gain: Classify Dynamic Point Cloud Sequences with Static Models by Fitting Feature-level Space-time Surfaces

**Authors**: *Jia-Xing Zhong, Kaichen Zhou, Qingyong Hu, Bing Wang, Niki Trigoni, Andrew Markham*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00832](https://doi.org/10.1109/CVPR52688.2022.00832)

**Abstract**:

Scene flow is a powerful tool for capturing the motion field of 3D point clouds. However, it is difficult to directly apply flow-based models to dynamic point cloud classification since the unstructured points make it hard or even impossible to efficiently and effectively trace point-wise correspondences. To capture 3D motions without explicitly tracking correspondences, we propose a kinematics-inspired neural network (Kinet) by generalizing the kinematic concept of ST-surfaces to the feature space. By unrolling the normal solver of ST-surfaces in the feature space, Kinet implicitly encodes feature-level dynamics and gains advantages from the use of mature back-bones for static point cloud processing. With only minor changes in network structures and low computing overhead, it is painless to jointly train and deploy our framework with a given static model. Experiments on NvGesture, SHREC'17, MSRAction-3D, and NTU-RGBD demonstrate its efficacy in performance, efficiency in both the number of parameters and computational complexity, as well as its versatility to various static backbones. Noticeably, Kinet achieves the accuracy of 93.27% on MSRAction-3D with only 3.20M parameters and 10.35G FLOPS. The code is available at https://github.com/jx-zhong-for-academic-purpose/Kinet.

----

## [824] Point2Seq: Detecting 3D Objects as Sequences

**Authors**: *Yujing Xue, Jiageng Mao, Minzhe Niu, Hang Xu, Michael Bi Mi, Wei Zhang, Xiaogang Wang, Xinchao Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00833](https://doi.org/10.1109/CVPR52688.2022.00833)

**Abstract**:

We present a simple and effective framework, named Point2Seq, for 3D object detection from point clouds. In contrast to previous methods that normally predict attributes of 3D objects all at once, we expressively model the interdependencies between attributes of 3D objects, which in turn enables a better detection accuracy. Specifically, we view each 3D object as a sequence of words and reformulate the 3D object detection task as decoding words from 3D scenes in an auto-regressive manner. We further propose a lightweight scene-to-sequence decoder that can auto-regressively generate words conditioned on features from a 3D scene as well as cues from the preceding words. The predicted words eventually constitute a set of sequences that completely describe the 3D objects in the scene, and all the predicted sequences are then automatically assigned to the respective ground truths through similarity-based sequence matching. Our approach is conceptually intuitive and can be readily plugged upon most existing 3D-detection backbones without adding too much computational over-head; the sequential decoding paradigm we proposed, on the other hand, can better exploit information from complex 3D scenes with the aid of preceding predicted words. Without bells and whistles, our method significantly out-performs previous anchor- and center-based 3D object detection frameworks, yielding the new state of the art on the challenging ONCE dataset as well as the Waymo Open Dataset. Code is available at https://github.com/ocNflag/point2seq.

----

## [825] PTTR: Relational 3D Point Cloud Object Tracking with Transformer

**Authors**: *Changqing Zhou, Zhipeng Luo, Yueru Luo, Tianrui Liu, Liang Pan, Zhongang Cai, Haiyu Zhao, Shijian Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00834](https://doi.org/10.1109/CVPR52688.2022.00834)

**Abstract**:

In a point cloud sequence, 3D object tracking aims to predict the location and orientation of an object in the current search point cloud given a template point cloud. Motivated by the success of transformers, we propose Point Tracking TRansformer (PTTR), which efficiently predicts high-quality 3D tracking results in a coarse-to-fine manner with the help of transformer operations. PTTR consists of three novel designs. 1) Instead of random sampling, we design Relation-Aware Sampling to preserve relevant points to given templates during subsampling. 2) Furthermore, we propose a Point Relation Transformer (PRT) consisting of a self-attention and a cross-attention module. The global self-attention operation captures long-range dependencies to enhance encoded point features for the search area and the template, respectively. Subsequently, we generate the coarse tracking results by matching the two sets of point features via cross-attention. 3) Based on the coarse tracking results, we employ a novel Prediction Refinement Module to obtain the final refined prediction. In addition, we create a large-scale point cloud single object tracking benchmark based on the Waymo Open Dataset. Extensive experiments show that PTTR achieves superior point cloud tracking in both accuracy and efficiency. Our code is available at https://github.com/Jasonkks/PTTR.

----

## [826] A Unified Query-based Paradigm for Point Cloud Understanding

**Authors**: *Zetong Yang, Li Jiang, Yanan Sun, Bernt Schiele, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00835](https://doi.org/10.1109/CVPR52688.2022.00835)

**Abstract**:

3D point cloud understanding is an important component in autonomous driving and robotics. In this paper, we present a novel Embedding-Querying paradigm (EQ-Paradigm) for 3D understanding tasks including detection, segmentation and classification. EQ-Paradigm is a unified paradigm that enables combination of existing 3D back-bone architectures with different task heads. Under the EQ-Paradigm, the input is first encoded in the embedding stage with an arbitrary feature extraction architecture, which is independent of tasks and heads. Then, the querying stage enables the encoded features for diverse task heads. This is achieved by introducing an intermediate representation, i.e., Q-representation, in the querying stage to bridge the embedding stage and task heads. We design a novel Q-Net as the querying stage network. Extensive experimental results on various 3D tasks show that EQ-Paradigm in tandem with Q-Net is a general and effective pipeline, which enables flexible collaboration of backbones and heads. It further boosts performance of state-of-the-art methods.

----

## [827] PointCLIP: Point Cloud Understanding by CLIP

**Authors**: *Renrui Zhang, Ziyu Guo, Wei Zhang, Kunchang Li, Xupeng Miao, Bin Cui, Yu Qiao, Peng Gao, Hongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00836](https://doi.org/10.1109/CVPR52688.2022.00836)

**Abstract**:

Recently, zero-shot and few-shot learning via Contrastive Vision-Language Pre-training (CLIP) have shown inspirational performance on 2D visual recognition, which learns to match images with their corresponding texts in open-vocabulary settings. However, it remains under explored that whether CLIP, pre-trained by large-scale image-text pairs in 2D, can be generalized to 3D recognition. In this paper, we identify such a setting is feasible by proposing PointCLIP, which conducts alignment between CLIP-encoded point clouds and 3D category texts. Specifically, we encode a point cloud by projecting it onto multi-view depth maps and aggregate the view-wise zero-shot prediction in an end-to-end manner, which achieves efficient knowledge transfer from 2D to 3D. We further design an inter-view adapter to better extract the global feature and adaptively fuse the 3D few-shot knowledge into CLIP pre-trained in 2D. By just fine-tuning the adapter under few-shot settings, the performance of PointCLIP could be largely improved. In addition, we observe the knowledge complementary property between PointCLIP and classical 3D-supervised networks. Via simple ensemble during inference, PointCLIP contributes to favorable performance enhancement over state-of-the-art 3D networks. Therefore, PointCLIP is a promising alternative for effective 3D point cloud understanding under low data regime with marginal resource cost. We conduct thorough experiments on Model-NetlO, ModelNet40 and ScanObjectNN to demonstrate the effectiveness of PointCLIP. Code is available at https://github.com/ZrrSkywalker/PointCLIP.

----

## [828] X -Trans2Cap: Cross-Modal Knowledge Transfer using Transformer for 3D Dense Captioning

**Authors**: *Zhihao Yuan, Xu Yan, Yinghong Liao, Yao Guo, Guanbin Li, Shuguang Cui, Zhen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00837](https://doi.org/10.1109/CVPR52688.2022.00837)

**Abstract**:

3D dense captioning aims to describe individual objects in 3D scenes by natural language, where 3D scenes are usually represented as RGB-D scans or point clouds. However, only exploiting single modal information, e.g., point cloud, previous approaches fail to produce faithful descriptions. Though aggregating 2D features into point clouds may be beneficial, it introduces an extra computational burden, especially in the inference phase. In this study, we investigate a cross-modal knowledge transfer using Transformer for 3D dense captioning, namely X-Trans2Cap. Our proposed X-Trans2Cap effectively boost the performance of single-modal 3D captioning through the knowledge distillation enabled by a teacher-student framework. In practice, during the training phase, the teacher network exploits auxiliary 2D modality and guides the student network that only takes point clouds as input through the feature consistency constraints. Owing to the well-designed cross-modal feature fusion module and the feature alignment in the training phase, X-Trans2Cap acquires rich appearance information embedded in 2D images with ease. Thus, a more faithful caption can be generated only using point clouds during the inference. Qualitative and quantitative results confirm that X-Trans2Cap outperforms previous state-of-the-art by a large margin, i.e., about +21 and +16 CIDEr points on ScanRefer and Nr3D datasets, respectively.

----

## [829] MVS2D: Efficient Multiview Stereo via Attention-Driven 2D Convolutions

**Authors**: *Zhenpei Yang, Zhile Ren, Qi Shan, Qixing Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00838](https://doi.org/10.1109/CVPR52688.2022.00838)

**Abstract**:

Deep learning has made significant impacts on multiview stereo systems. State-of-the-art approaches typically involve building a cost volume, followed by multiple 3D convolution operations to recover the input image's pixel-wise depth. While such end-to-end learning of plane-sweeping stereo advances public benchmarks' accuracy, they are typically very slow to compute. We present MVS2D, a highly efficient multi-view stereo algorithm that seamlessly integrates multi-view constraints into single-view net-works via an attention mechanism. Since MVS2D only builds on 2D convolutions, it is at least 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$2\times faster$</tex>
 than all the notable counterparts. Moreover, our algorithm produces precise depth estimations and 3D reconstructions, achieving state-of-the-art results on challenging benchmarks ScanNet, SUN3D, RGBD, and the classical DTU dataset. our algorithm also outperforms all other algorithms in the setting of inexact camera poses. Our code is released at https://github.com/zhenpeiyang/MVS2D

----

## [830] TransMVSNet: Global Context-aware Multi-view Stereo Network with Transformers

**Authors**: *Yikang Ding, Wentao Yuan, Qingtian Zhu, Haotian Zhang, Xiangyue Liu, Yuanjiang Wang, Xiao Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00839](https://doi.org/10.1109/CVPR52688.2022.00839)

**Abstract**:

In this paper, we present TransMVSNet, based on our exploration of feature matching in multi-view stereo (MVS). We analogize MVS back to its nature of a feature matching task and therefore propose a powerful Feature Matching Transformer (FMT) to leverage intra- (self-) and inter-(cross-) attention to aggregate long-range context information within and across images. To facilitate a better adaptation of the FMT, we leverage an Adaptive Receptive Field (ARF) module to ensure a smooth transit in scopes of features and bridge different stages with a feature pathway to pass transformed features and gradients across different scales. In addition, we apply pair-wise feature correlation to measure similarity between features, and adopt ambiguity-reducing focal loss to strengthen the supervision. To the best of our knowledge, TransMVSNet is the first attempt to leverage Transformer into the task of MVS. As a result, our method achieves state-of-the-art performance on DTU dataset, Tanks and Temples benchmark, and BlendedMVS dataset. Code is available at https://github.com/MegviiRobot/TransMVSNet.

----

## [831] RayMVSNet: Learning Ray-based 1D Implicit Fields for Accurate Multi-View Stereo

**Authors**: *Junhua Xi, Yifei Shi, Yijie Wang, Yulan Guo, Kai Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00840](https://doi.org/10.1109/CVPR52688.2022.00840)

**Abstract**:

Learning-based multi-view stereo (MVS) has by far cen-tered around 3D convolution on cost volumes. Due to the high computation and memory consumption of 3D CNN, the resolution of output depth is often considerably limited. Differentfrom most existing works dedicated to adaptive re-finement of cost volumes, we opt to directly optimize the depth value along each camera ray, mimicking the range (depth) finding of a laser scanner. This reduces the MVS problem to ray-based depth optimization which is much more light-weight than full cost volume optimization. In particular, we propose RayMVSNet which learns sequen-tial prediction of aID implicit field along each camera ray with the zero-crossing point indicating scene depth. This sequential modeling, conducted based on transformer features, essentially learns the epipolar line search in traditional multi-view stereo. We also devise a multi-task learning for better optimization convergence and depth accuracy. Our method ranks top on both the DTU and the Tanks & Temples datasets over all previous learning-based methods, achieving overall reconstruction score of 0.33mm on DTU andf-score of59.48% on Tanks & Temples.

----

## [832] IterMVS: Iterative Probability Estimation for Efficient Multi-View Stereo

**Authors**: *Fangjinhua Wang, Silvano Galliani, Christoph Vogel, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00841](https://doi.org/10.1109/CVPR52688.2022.00841)

**Abstract**:

We present IterMVS, a new data-driven method for high-resolution multi-view stereo. We propose a novel GRU-based estimator that encodes pixel-wise probability distributions of depth in its hidden state. Ingesting multi-scale matching information, our model refines these distributions over multiple iterations and infers depth and confidence. To extract the depth maps, we combine traditional classification and regression in a novel manner. We verify the efficiency and effectiveness of our method on DTU, Tanks&Temples and ETH3D. While being the most efficient method in both memory and run-time, our model achieves competitive performance on DTU and better generalization ability on Tanks&Temples as well as ETH3D than most state-of-the-art methods. Code is available at https://github.com/FangjinhuaWang/IterMVS.

----

## [833] PSMNet: Position-aware Stereo Merging Network for Room Layout Estimation

**Authors**: *Haiyan Wang, Will Hutchcroft, Yuguang Li, Zhiqiang Wan, Ivaylo Boyadzhiev, Yingli Tian, Sing Bing Kang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00842](https://doi.org/10.1109/CVPR52688.2022.00842)

**Abstract**:

In this paper, we propose a new deep learning-based method for estimating room layout given a pair of 360째 panoramas. Our system, called Position-aware Stereo Merging Network or PSMNet, is an end-to-end joint layout-pose estimator. PSMNet consists of a Stereo Pano Pose (SP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) transformer and a novel Cross-Perspective Projection (CP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) layer. The stereo-view SP2 transformer is used to implicitly infer correspondences between views, and can handle noisy poses. The pose-aware CP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 layer is designed to render features from the adjacent view to the anchor (reference) view, in order to perform view fusion and estimate the visible layout. Our experiments and analysis validate our method, which significantly outperforms the state-of-the-art layout estimators, especially for large and complex room spaces.

----

## [834] Non-parametric Depth Distribution Modelling based Depth Inference for Multi-view Stereo

**Authors**: *Jiayu Yang, José M. Álvarez, Miaomiao Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00843](https://doi.org/10.1109/CVPR52688.2022.00843)

**Abstract**:

Recent cost volume pyramid based deep neural networks have unlocked the potential of efficiently leveraging high-resolution images for depth inference from multi-view stereo. In general, those approaches assume that the depth of each pixel follows a unimodal distribution. Boundary pixels usually follow a multi-modal distribution as they represent different depths; Therefore, the assumption results in an erroneous depth prediction at the coarser level of the cost volume pyramid and can not be corrected in the refinement levels leading to wrong depth predictions. In contrast, we propose constructing the cost volume by non-parametric depth distribution modeling to handle pixels with unimodal and multi-modal distributions. Our approach outputs multiple depth hypotheses at the coarser level to avoid errors in the early stage. As we perform local search around these multiple hypotheses in subsequent levels, our approach does not maintain the rigid depth spatial ordering and, therefore, we introduce a sparse cost aggregation network to derive information within each volume. We evaluate our approach extensively on two benchmark datasets: DTU and Tanks & Temples. Our experimental results show that our model outperforms existing methods by a large margin and achieves superior performance on boundary regions. Code is available at https://github.com/NVlabs/NP-CVP-MVSNet

----

## [835] Differentiable Stereopsis: Meshes from multiple views using differentiable rendering

**Authors**: *Shubham Goel, Georgia Gkioxari, Jitendra Malik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00844](https://doi.org/10.1109/CVPR52688.2022.00844)

**Abstract**:

We propose Differentiable Stereopsis, a multi-view stereo approach that reconstructs shape and texture from few input views and noisy cameras. We pair traditional stereopsis and modern differentiable rendering to build an end-to-end model which predicts textured 3D meshes of objects with varying topologies and shape. We frame stereopsis as an optimization problem and simultaneously update shape and cameras via simple gradient descent. We run an extensive quantitative analysis and compare to traditional multi-view stereo techniques and state-of-the-art learning based methods. We show compelling reconstructions on challenging real-world scenes and for an abundance of object types with complex shape, topology and texture.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project webpage: https://shubham-goel.github.io/ds/

----

## [836] Rethinking Depth Estimation for Multi-View Stereo: A Unified Representation

**Authors**: *Rui Peng, Rongjie Wang, Zhenyu Wang, Yawen Lai, Ronggang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00845](https://doi.org/10.1109/CVPR52688.2022.00845)

**Abstract**:

Depth estimation is solved as a regression or classification problem in existing learning-based multi-view stereo methods. Although these two representations have recently demonstrated their excellent performance, they still have apparent shortcomings, e.g., regression methods tend to overfit due to the indirect learning cost volume, and classification methods cannot directly infer the exact depth due to its discrete prediction. In this paper, we propose a novel representation, termed Unification, to unify the advantages of regression and classification. It can directly constrain the cost volume like classification methods, but also realize the sub-pixel depth prediction like regression methods. To excavate the potential of unification, we design a new loss function named Unified Focal Loss, which is more uniform and reasonable to combat the challenge of sample imbalance. Combining these two unburdened modules, we present a coarse-to-fine framework, that we call UniMVSNet. The results of ranking first on both DTU and Tanks and Temples benchmarks verify that our model not only performs the best but also has the best generalization ability.

----

## [837] Efficient Multi-view Stereo by Iterative Dynamic Cost Volume

**Authors**: *Shaoqian Wang, Bo Li, Yuchao Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00846](https://doi.org/10.1109/CVPR52688.2022.00846)

**Abstract**:

In this paper, we propose a novel iterative dynamic cost volume for multi-view stereo. Compared with other works, our cost volume is much lighter, thus could be processed with 2D convolution based GRU. Notably, the every-step output of the GRU could be further used to generate new cost volume. In this way, an iterative GRU-based optimizer is constructed. Furthermore, we present a cascade and hierarchical refinement architecture to utilize the multiscale information and speed up the convergence. Specifically, a lightweight 3D CNN is utilized to generate the coarsest initial depth map which is essential to launch the GRU and guarantee a fast convergence. Then the depth map is refined by multi-stage GRUs which work on the pyramid feature maps. Extensive experiments on the DTU and Tanks & Temples benchmarks demonstrate that our method could achieve state-of-the-art results in terms of accuracy, speed and memory usage. Code will be released at https://github.com/bdwsq1996/Effi-MVS.

----

## [838] PlaneMVS: 3D Plane Reconstruction from Multi-View Stereo

**Authors**: *Jiachen Liu, Pan Ji, Nitin Bansal, Changjiang Cai, Qingan Yan, Xiaolei Huang, Yi Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00847](https://doi.org/10.1109/CVPR52688.2022.00847)

**Abstract**:

We present a novel framework named PlaneMVS for 3D plane reconstruction from multiple input views with known camera poses. Most previous learning-based plane reconstruction methods reconstruct 3D planes from single images, which highly rely on single-view regression and suffer from depth scale ambiguity. In contrast, we reconstruct 3D planes with a multi-view-stereo (MVS) pipeline that takes advantage of multi-view geometry. We decouple plane reconstruction into a semantic plane detection branch and a plane MVS branch. The semantic plane detection branch is based on a single-view plane detection framework but with differences. The plane MVS branch adopts a set of slanted plane hypotheses to replace conventional depth hypotheses to perform plane sweeping strategy and finally learns pixel-level plane parameters and its planar depth map. We present how the two branches are learned in a balanced way, and propose a soft-pooling loss to associate the outputs of the two branches and make them benefit from each other. Extensive experiments on various indoor datasets show that PlaneMVS significantly outperforms state-of-the-art (SOTA) single-view plane reconstruction methods on both plane detection and 3D geometry metrics. Our method even outperforms a set of SOTA learning-based MVS methods thanks to the learned plane priors. To the best of our knowledge, this is the first work on 3D plane reconstruction within an end-to-end MVS framework.

----

## [839] Discrete time convolution for fast event-based stereo

**Authors**: *Kaixuan Zhang, Kaiwei Che, Jianguo Zhang, Jie Cheng, Ziyang Zhang, Qinghai Guo, Luziwei Leng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00848](https://doi.org/10.1109/CVPR52688.2022.00848)

**Abstract**:

Inspired by biological retina, dynamical vision sensor transmits events of instantaneous changes of pixel intensity, giving it a series of advantages over traditional frame-based camera, such as high dynamical range, high temporal resolution and low power consumption. However, extracting information from highly asynchronous event data is a challenging task. Inspired by continuous dynamics of biological neuron models, we propose a novel encoding method for sparse events-continuous time convolution (CTC)-which learns to model the spatial feature of the data with intrinsic dynamics. Adopting channel-wise parameterization, temporal dynamics of the model is synchronized on the same feature map and diverges across different ones, enabling it to embed data in a variety of temporal scales. Abstracted from CTC, we further develop discrete time convolution (DTC) which accelerates the process with lower computational cost. We apply these methods to event-based multi- view stereo matching where they surpass state-of-the-art methods on benchmark criteria of the MVSEC dataset. Spatially sparse event data often leads to inaccurate estimation of edges and local contours. To address this problem, we propose a dual-path architecture in which the feature map is complemented by underlying edge information from original events extracted with spatially-adaptive denormal-ization. We demonstrate the superiority of our model in terms of speed (up to 110 FPS), accuracy and robustness, showing a great potential for real-time fast depth estimation. Finally, we perform experiments on the recent DSEC dataset to demonstrate the general usage of our model.

----

## [840] Stereo Magnification with Multi-Layer Images

**Authors**: *Taras Khakhulin, Denis Korzhenkov, Pavel Solovev, Gleb Sterkin, Andrei-Timotei Ardelean, Victor Lempitsky*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00849](https://doi.org/10.1109/CVPR52688.2022.00849)

**Abstract**:

Representing scenes with multiple semitransparent colored layers has been a popular and successful choice for real-time novel view synthesis. Existing approaches infer colors and transparency values over regularly spaced layers of planar or spherical shape. In this work, we introduce a new view synthesis approach based on multiple semitransparent layers with scene-adapted geometry. Our approach infers such representations from stereo pairs in two stages. The first stage produces the geometry of a small number of data-adaptive layers from a given pair of views. The second stage infers the color and transparency values for these layers, producing the final representation for novel view synthesis. Importantly, both stages are connected through a differentiable renderer and are trained end-to-end. In the experiments, we demonstrate the advantage of the proposed approach over the use of regularly spaced layers without adaptation to scene geometry. Despite being orders of magnitude faster during rendering, our approach also outperforms the recently proposed IBRNet system based on implicit geometry representation.

----

## [841] TransforMatcher: Match-to-Match Attention for Semantic Correspondence

**Authors**: *Seungwook Kim, Juhong Min, Minsu Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00850](https://doi.org/10.1109/CVPR52688.2022.00850)

**Abstract**:

Establishing correspondences between images remains a challenging task, especially under large appearance changes due to different viewpoints or intra-class variations. In this work, we introduce a strong semantic image matching learner, dubbed TransforMatcher, which builds on the success of transformer networks in vision domains. Un-like existing convolution- or attention-based schemes for correspondence, TransforMatcher performs global match-to-match attention for precise match localization and dynamic refinement. To handle a large number of matches in a dense correlation map, we develop a light-weight attention architecture to consider the global match-to-match interactions. We also propose to utilize a multi-channel correlation map for refinement, treating the multi-level scores as features instead of a single score to fully exploit the richer layer-wise semantics. In experiments, TransforMatcher sets a new state of the art on SPair-71k while performing on par with existing SOTA methods on the PF-PASCAL dataset.

----

## [842] Probabilistic Warp Consistency for Weakly-Supervised Semantic Correspondences

**Authors**: *Prune Truong, Martin Danelljan, Fisher Yu, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00851](https://doi.org/10.1109/CVPR52688.2022.00851)

**Abstract**:

We propose Probabilistic Warp Consistency, a weakly-supervised learning objective for semantic matching. Our approach directly supervises the dense matching scores predicted by the network, encoded as a conditional probability distribution. We first construct an image triplet by applying a known warp to one of the images in a pair depicting different instances of the same object class. Our probabilistic learning objectives are then derived using the constraints arising from the resulting image triplet. We further account for occlusion and background clutter present in real image pairs by extending our probabilistic output space with a learnable unmatched state. To supervise it, we design an objective between image pairs depicting different object classes. We validate our method by applying it to four recent semantic matching architectures. Our weakly-supervised approach sets a new state-of-the-art on four challenging semantic matching benchmarks. Lastly, we demonstrate that our objective also brings substantial improvements in the strongly-supervised regime, when combined with keypoint annotations.

----

## [843] Locality-Aware Inter-and Intra-Video Reconstruction for Self-Supervised Correspondence Learning

**Authors**: *Liulei Li, Tianfei Zhou, Wenguan Wang, Lu Yang, Jianwu Li, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00852](https://doi.org/10.1109/CVPR52688.2022.00852)

**Abstract**:

Our target is to learn visual correspondence from unlabeled videos. We develop Liir, a locality-aware inter-and intra-video reconstruction method that fills in three missing pieces, i.e., instance discrimination, location awareness, and spatial compactness, of self-supervised correspondence learning puzzle. First, instead of most existing efforts focusing on intra-video self-supervision only, we exploit cross-video affinities as extra negative samples within a unified, inter-and intra-video reconstruction scheme. This enables instance discriminative representation learning by contrasting desired intra-video pixel association against negative inter-video correspondence. Second, we merge position information into correspondence matching, and design a position shifting strategy to remove the side-effect of position encoding during inter-video affinity computation, making our Liir location-sensitive. Third, to make full use of the spatial continuity nature of video data, we impose a compactness-based constraint on correspondence matching, yielding more sparse and reliable solutions. The learned representation surpasses self-supervised state-of-the-arts on label propagation tasks including objects, semantic parts, and keypoints.

----

## [844] Transforming Model Prediction for Tracking

**Authors**: *Christoph Mayer, Martin Danelljan, Goutam Bhat, Matthieu Paul, Danda Pani Paudel, Fisher Yu, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00853](https://doi.org/10.1109/CVPR52688.2022.00853)

**Abstract**:

Optimization based tracking methods have been widely successful by integrating a target model prediction module, providing effective global reasoning by minimizing an objective function. While this inductive bias integrates valuable domain knowledge, it limits the expressivity of the tracking network. In this work, we therefore propose a tracker architecture employing a Transformer-based model prediction module. Transformers capture global relations with little inductive bias, allowing it to learn the prediction of more powerful target models. We further extend the model predictor to estimate a second set of weights that are applied for accurate bounding box regression. The resulting tracker ToMP relies on training and on test frame information in order to predict all weights transductively. We train the proposed tracker end-to-end and validate its performance by conducting comprehensive experiments on multiple tracking datasets. ToMP sets a new state of the art on three benchmarks, achieving an AUC of 68.5% on the challenging LaSOT [14] dataset. The code and trained models are available at https://github.com/visionml/pytracking

----

## [845] Ranking-Based Siamese Visual Tracking

**Authors**: *Feng Tang, Qiang Ling*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00854](https://doi.org/10.1109/CVPR52688.2022.00854)

**Abstract**:

Current Siamese-based trackers mainly formulate the visual tracking into two independent subtasks, including classification and localization. They learn the classification subnetwork by processing each sample separately and neglect the relationship among positive and negative samples. Moreover, such tracking paradigm takes only the classification confidence of proposals for the final prediction, which may yield the misalignment between classification and localization. To resolve these issues, this paper proposes a ranking-based optimization algorithm to explore the relationship among different proposals. To this end, we introduce two ranking losses, including the classification one and the IoU-guided one, as optimization constraints. The classification ranking loss can ensure that positive samples rank higher than hard negative ones, i.e., distractors, so that the trackers can select the foreground samples successfully without being fooled by the distractors. The IoU-guided ranking loss aims to align classification confidence scores with the Intersection over Union(IoU) of the corresponding localization prediction for positive samples, enabling the well-localized prediction to be represented by high classification confidence. Specifically, the proposed two ranking losses are compatible with most Siamese trackers and incur no additional computation for inference. Extensive experiments on seven tracking benchmarks, including OTB100, UAV123, TC128, VOT2016, NFS30, GOT-10k and LaSOT, demonstrate the effectiveness of the proposed ranking-based optimization algorithm. The code and raw results are available at https://github.com/sansanfree/RBO.

----

## [846] Correlation-Aware Deep Tracking

**Authors**: *Fei Xie, Chunyu Wang, Guangting Wang, Yue Cao, Wankou Yang, Wenjun Zeng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00855](https://doi.org/10.1109/CVPR52688.2022.00855)

**Abstract**:

Robustness and discrimination power are two fundamental requirements in visual object tracking. In most tracking paradigms, we find that the features extracted by the popular Siamese-like networks cannot fully discriminatively model the tracked targets and distractor objects, hindering them from simultaneously meeting these two requirements. While most methods focus on designing robust correlation operations, we propose a novel target-dependent feature network inspired by the self-/cross-attention scheme. In contrast to the Siamese-like feature extraction, our network deeply embeds cross-image feature correlation in multiple layers of the feature network. By extensively matching the features of the two images through multiple layers, it is able to suppress non-target features, resulting in instance-varying feature extraction. The output features of the search image can be directly used for predicting target locations without extra correlation step. Moreover, our model can be flexibly pre-trained on abundant unpaired images, leading to notably faster convergence than the existing methods. Extensive experiments show our method achieves the state-of-the-art results while running at real-time. Our feature networks also can be applied to existing tracking pipelines seamlessly to raise the tracking performance.

----

## [847] Global Tracking via Ensemble of Local Trackers

**Authors**: *Zikun Zhou, Jianqiu Chen, Wenjie Pei, Kaige Mao, Hongpeng Wang, Zhenyu He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00856](https://doi.org/10.1109/CVPR52688.2022.00856)

**Abstract**:

The crux of long-term tracking lies in the difficulty of tracking the target with discontinuous moving caused by out-of-view or occlusion. Existing long-term tracking methods follow two typical strategies. The first strategy employs a local tracker to perform smooth tracking and uses another re-detector to detect the target when the target is lost. While it can exploit the temporal context like historical appearances and locations of the target, a potential limitation of such strategy is that the local tracker tends to misidentify a nearby distractor as the target instead of activating the re-detector when the real target is out of view. The other long-term tracking strategy tracks the target in the entire image globally instead of local tracking based on the previous tracking results. Unfortunately, such global tracking strategy cannot leverage the temporal context effectively. In this work, we combine the advantages of both strategies: tracking the target in a global view while exploiting the temporal context. Specifically, we perform global tracking via ensemble of local trackers spreading the full image. The smooth moving of the target can be handled steadily by one local tracker. When the local tracker accidentally loses the target due to suddenly discontinuous moving, another local tracker close to the target is then activated and can readily take over the tracking to locate the target. While the activated local tracker performs tracking locally by leveraging the temporal context, the ensemble of local trackers renders our model the global view for tracking. Extensive experiments on six datasets demonstrate that our method performs favorably against state-of-the-art algorithms.

----

## [848] Global Tracking Transformers

**Authors**: *Xingyi Zhou, Tianwei Yin, Vladlen Koltun, Philipp Krähenbühl*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00857](https://doi.org/10.1109/CVPR52688.2022.00857)

**Abstract**:

We present a novel transformer-based architecture for global multi-object tracking. Our network takes a short sequence of frames as input and produces global trajectories for all objects. The core component is a global tracking transformer that operates on objects from all frames in the sequence. The transformer encodes object features from all frames, and uses trajectory queries to group them into trajectories. The trajectory queries are object features from a single frame and naturally produce unique trajectories. Our global tracking transformer does not require intermediate pairwise grouping or combinatorial association, and can be jointly trained with an object detector. It achieves competitive performance on the popular MOT17 benchmark, with 75.3 MOTA and 59.1 HOTA. More importantly, our framework seamlessly integrates into state-of-the-art large-vocabulary detectors to track any objects. Experiments on the challenging TAO dataset show that our framework consistently improves upon baselines that are based on pairwise association, outperforming published work by a significant 7.7 tracking mAP. Code is available at https://github.com/xingyizhou/GTR.

----

## [849] Unified Transformer Tracker for Object Tracking

**Authors**: *Fan Ma, Mike Zheng Shou, Linchao Zhu, Haoqi Fan, Yilei Xu, Yi Yang, Zhicheng Yan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00858](https://doi.org/10.1109/CVPR52688.2022.00858)

**Abstract**:

As an important area in computer vision, object tracking has formed two separate communities that respectively study Single Object Tracking (SOT) and Multiple Object Tracking (MOT). However, current methods in one tracking scenario are not easily adapted to the other due to the divergent training datasets and tracking objects of both tasks. Although UniTrack [45] demonstrates that a shared appearance model with multiple heads can be used to tackle individual tracking tasks, it fails to exploit the large-scale tracking datasets for training and performs poorly on the single object tracking. In this work, we present the Unified Transformer Tracker (UTT) to address tracking problems in different scenarios with one paradigm. A track transformer is developed in our UTT to track the target in both SOT and MOT where the correlation between the target feature and the tracking frame feature is exploited to localize the target. We demonstrate that both SOT and MOT tasks can be solved within this framework, and the model can be simultaneously end-to-end trained by alternatively optimizing the SOT and MOT objectives on the datasets of individual tasks. Extensive experiments are conducted on several benchmarks with a unified model trained on both SOT and MOT datasets.

----

## [850] Transformer Tracking with Cyclic Shifting Window Attention

**Authors**: *Zikai Song, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00859](https://doi.org/10.1109/CVPR52688.2022.00859)

**Abstract**:

Transformer architecture has been showing its great strength in visual object tracking, for its effective attention mechanism. Existing transformer-based approaches adopt the pixel-to-pixel attention strategy on flattened image features and unavoidably ignore the integrity of ob-jects. In this paper, we propose a new transformer ar-chitecture with multi-scale cyclic shifting window attention for visual object tracking, elevating the attention from pixel to window level. The cross-window multi-scale at-tention has the advantage of aggregating attention at dif-ferent scales and generates the best fine-scale match for the target object. Furthermore, the cyclic shifting strat-egy brings greater accuracy by expanding the window sam-ples with positional information, and at the same time saves huge amounts of computational power by removing redun-dant calculations. Extensive experiments demonstrate the superior performance of our method, which also sets the new state-of-the-art records on five challenging datasets, along with the VOT2020, UAV123, LaSOT, TrackingNet, and GOT-lOk benchmarks. Our project is available at https://github.com/SkyeSong38/CSWinTT.

----

## [851] Spiking Transformers for Event-based Single Object Tracking

**Authors**: *Jiqing Zhang, Bo Dong, Haiwei Zhang, Jianchuan Ding, Felix Heide, Baocai Yin, Xin Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00860](https://doi.org/10.1109/CVPR52688.2022.00860)

**Abstract**:

Event-based cameras bring a unique capability to tracking, being able to function in challenging real-world conditions as a direct result of their high temporal resolution and high dynamic range. These imagers capture events asynchronously that encode rich temporal and spatial information. However, effectively extracting this information from events remains an open challenge. In this work, we propose a spiking transformer network, STNet, for single object tracking. STNet dynamically extracts and fuses information from both temporal and spatial domains. In particular, the proposed architecture features a transformer module to provide global spatial information and a spiking neural network (SNN) module for extracting temporal cues. The spiking threshold of the SNN module is dynamically adjusted based on the statistical cues of the spatial information, which we find essential in providing robust SNN features. We fuse both feature branches dynamically with a novel cross-domain attention fusion algorithm. Extensive experiments on three event-based datasets, FE240hz, EED and VisEvent validate that the proposed STNet outperforms existing state-of-the-art methods in both tracking accuracy and speed with a significant margin. The code and pretrained models are at https://github.com/Jee-King/CVPR2022_STNet.

----

## [852] Adiabatic Quantum Computing for Multi Object Tracking

**Authors**: *Jan-Nico Zaech, Alexander Liniger, Martin Danelljan, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00861](https://doi.org/10.1109/CVPR52688.2022.00861)

**Abstract**:

Multi-Object Tracking (MOT) is most often approached in the tracking-by-detection paradigm, where object detections are associated through time. The association step naturally leads to discrete optimization problems. As these optimization problems are often NP-hard, they can only be solved exactly for small instances on current hardware. Adiabatic quantum computing (AQC) offers a solution for this, as it has the potential to provide a considerable speedup on a range of NP-hard optimization problems in the near future. However, current MOT formulations are unsuitable for quantum computing due to their scaling properties. In this work, we therefore propose the first MOT formulation designed to be solved with AQC. We employ an Ising model that represents the quantum mechanical system implemented on the AQC. We show that our approach is competitive compared with state-of-the-art optimization-based approaches, even when using of-the-shelf integer programming solvers. Finally, we demonstrate that our MOT problem is already solvable on the current generation of real quantum computers for small examples, and analyze the properties of the measured solutions.

----

## [853] HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction

**Authors**: *Zikang Zhou, Luyao Ye, Jianping Wang, Kui Wu, Kejie Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00862](https://doi.org/10.1109/CVPR52688.2022.00862)

**Abstract**:

Accurately predicting the future motions of surrounding traffic agents is critical for the safety of autonomous ve-hicles. Recently, vectorized approaches have dominated the motion prediction community due to their capability of capturing complex interactions in traffic scenes. How-ever, existing methods neglect the symmetries of the prob-lem and suffer from the expensive computational cost, facing the challenge of making real-time multi-agent motion prediction without sacrificing the prediction performance. To tackle this challenge, we propose Hierarchical Vector Transformer (HiVT) for fast and accurate multi-agent motion prediction. By decomposing the problem into local con-text extraction and global interaction modeling, our method can effectively and efficiently model a large number of agents in the scene. Meanwhile, we propose a translation-invariant scene representation and rotation-invariant spa-tial learning modules, which extract features robust to the geometric transformations of the scene and enable the model to make accurate predictions for multiple agents in a single forward pass. Experiments show that HiVT achieves the state-of-the-art performance on the Argoverse motion forecasting benchmark with a small model size and can make fast multi-agent motion prediction.

----

## [854] Towards Discriminative Representation: Multi-view Trajectory Contrastive Learning for Online Multi-object Tracking

**Authors**: *En Yu, Zhuoling Li, Shoudong Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00863](https://doi.org/10.1109/CVPR52688.2022.00863)

**Abstract**:

Discriminative representation is crucial for the association step in multi-object tracking. Recent work mainly utilizes features in single or neighboring frames for constructing metric loss and empowering networks to extract representation of targets. Although this strategy is effective, it fails to fully exploit the information contained in a whole trajectory. To this end, we propose a strategy, namely multi-view trajectory contrastive learning, in which each trajectory is represented as a center vector. By maintaining all the vectors in a dynamically updated memory bank, a trajectory-level contrastive loss is devised to explore the inter-frame information in the whole trajectories. Besides, in this strategy, each target is represented as multiple adaptively selected keypoints rather than a pre-defined anchor or center. This design allows the network to generate richer representation from multiple views of the same target, which can better characterize occluded objects. Additionally, in the inference stage, a similarity-guided feature fusion strategy is developed for further boosting the quality of the trajectory representation. Extensive experiments have been conducted on MOTChallenge to verify the effectiveness of the proposed techniques. The experimental results indicate that our method has surpassed preceding trackers and established new state-of-the-art performance.

----

## [855] TrackFormer: Multi-Object Tracking with Transformers

**Authors**: *Tim Meinhardt, Alexander Kirillov, Laura Leal-Taixé, Christoph Feichtenhofer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00864](https://doi.org/10.1109/CVPR52688.2022.00864)

**Abstract**:

The challenging task of multi-object tracking (MOT) requires simultaneous reasoning about track initialization, identity, and spatio-temporal trajectories. We formulate this task as a frame-to-frame set prediction problem and introduce TrackFormer, an end-to-end trainable MOT approach based on an encoder-decoder Transformer architecture. Our model achieves data association between frames via attention by evolving a set of track predictions through a video sequence. The Transformer decoder initializes new tracks from static object queries and autoregressively follows existing tracks in space and time with the conceptually new and identity preserving track queries. Both query types benefit from self- and encoder-decoder attention on global frame-level features, thereby omitting any additional graph optimization or modeling of motion and/or appearance. TrackFormer introduces a new tracking-by-attention paradigm and while simple in its design is able to achieve state-of-the-art performance on the task of multi-object tracking (MOT17) and segmentation (MOTS20). The code is available at https://github.com/timmeinhardt/TrackFormer

----

## [856] Learning of Global Objective for Network Flow in Multi-Object Tracking

**Authors**: *Shuai Li, Yu Kong, Hamid Rezatofighi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00865](https://doi.org/10.1109/CVPR52688.2022.00865)

**Abstract**:

This paper concerns the problem of multi-object tracking based on the min-cost flow (MCF) formulation, which is conventionally studied as an instance of linear program. Given its computationally tractable inference, the success of MCF tracking largely relies on the learned cost function of underlying linear program. Most previous studies focus on learning the cost function by only taking into account two frames during training, therefore the learned cost function is sub-optimal for MCF where a multi-frame data association must be considered during inference. In order to address this problem, in this paper we propose a novel differentiable framework that ties training and inference to-gether during learning by solving a bi-level optimization problem, where the lower-level solves a linear program and the upper-level contains a loss function that incorpo-rates global tracking result. By back-propagating the loss through differentiable layers via gradient descent, the glob-ally parameterized cost function is explicitly learned and regularized. With this approach, we are able to learn a better objective for global MCF tracking. As a result, we achieve competitive performances compared to the current state-of-the-art methods on the popular multi-object tracking benchmarks such as MOT16, MOT17 and MOT20.

----

## [857] LMGP: Lifted Multicut Meets Geometry Projections for Multi-Camera Multi-Object Tracking

**Authors**: *Duy M. H. Nguyen, Roberto Henschel, Bodo Rosenhahn, Daniel Sonntag, Paul Swoboda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00866](https://doi.org/10.1109/CVPR52688.2022.00866)

**Abstract**:

Multi-Camera Multi-Object Tracking is currently drawing attention in the computer vision field due to its superior performance in real-world applications such as video surveillance with crowded scenes or in wide spaces. In this work, we propose a mathematically elegant multi-camera multiple object tracking approach based on a spatial-temporal lifted multicut formulation. Our model utilizes state-of-the-art tracklets produced by single-camera trackers as proposals. As these tracklets may contain ID-Switch errors, we refine them through a novel pre-clustering obtained from 3D geometry projections. As a result, we derive a better tracking graph without ID switches and more precise affinity costs for the data association phase. Tracklets are then matched to multi-camera trajectories by solving a global lifted multicut formulation that incorporates short and long-range temporal interactions on tracklets located in the same camera as well as inter-camera ones. Experimental results on the WildTrack dataset yield near-perfect performance, outperforming state-of-the-art trackers on Campus while being on par on the PETS-09 dataset. We will release our implementations at this link https://github.com/nhmduy/LMGP.

----

## [858] Multi-Object Tracking Meets Moving UAV

**Authors**: *Shuai Liu, Xin Li, Huchuan Lu, You He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00867](https://doi.org/10.1109/CVPR52688.2022.00867)

**Abstract**:

Multi-object tracking in unmanned aerial vehicle (UAV) videos is an important vision task and can be applied in a wide range of applications. However, conventional multi-object trackers do not work well on UAV videos due to the challenging factors of irregular motion caused by moving camera and view change in 3D directions. In this paper, we propose a UAVMOT network specially for multi-object tracking in UAV views. The UAVMOT introduces an ID feature update module to enhance the object's feature association. To better handle the complex motions under UAV views, we develop an adaptive motion filter module. In addition, a gradient balanced focal loss is used to tackle the imbalance categories and small objects detection problem. Experimental results on the VisDrone2019 and UAVDT datasets demonstrate that the proposed UAVMOT achieves considerable improvement against the state-of-the-art tracking methods on UAV videos.

----

## [859] Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline

**Authors**: *Pengyu Zhang, Jie Zhao, Dong Wang, Huchuan Lu, Xiang Ruan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00868](https://doi.org/10.1109/CVPR52688.2022.00868)

**Abstract**:

With the popularity of multi-modal sensors, visible-thermal (RGB-T) object tracking is to achieve robust performance and wider application scenarios with the guidance of objects' temperature information. However, the lack of paired training samples is the main bottleneck for unlocking the power of RGB-T tracking. Since it is laborious to collect high-quality RGB-T sequences, recent benchmarks only provide test sequences. In this paper, we construct a large-scale benchmark with high diversity for visible-thermal UAV tracking (VTUAV), including 500 sequences with 1.7 million high-resolution (1920* 1080 pixels) frame pairs. In addition, comprehensive applications (short-term tracking, long-term tracking and segmentation mask prediction) with diverse categories and scenes are considered for exhaustive evaluation. Moreover, we provide a coarse-to-fine attribute annotation, where frame-level attributes are provided to exploit the potential of challenge-specific trackers. In addition, we design a new RGB-T baseline, named Hierarchical Multi-modal Fusion Tracker (HMFT), which fuses RGB-T data in various levels. Numerous experiments on several datasets are conducted to reveal the effectiveness of HMFT and the complement of different fusion types. The project is available at here.

----

## [860] Unsupervised Domain Adaptation for Nighttime Aerial Tracking

**Authors**: *Junjie Ye, Changhong Fu, Guangze Zheng, Danda Pani Paudel, Guang Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00869](https://doi.org/10.1109/CVPR52688.2022.00869)

**Abstract**:

Previous advances in object tracking mostly reported on favorable illumination circumstances while neglecting performance at nighttime, which significantly impeded the development of related aerial robot applications. This work instead develops a novel unsupervised domain adaptation framework for nighttime aerial tracking (named UDAT). Specifically, a unique object discovery approach is provided to generate training patches from raw nighttime tracking videos. To tackle the domain discrepancy, we employ a Transformer-based bridging layer post to the feature extractor to align image features from both domains. With a Transformer day/night feature discriminator, the day-time tracking model is adversarially trained to track at night. Moreover, we construct a pioneering benchmark namely NAT2021 for unsupervised domain adaptive night-time tracking, which comprises a test set of 180 manually annotated tracking sequences and a train set of over 276k unlabelled nighttime tracking frames. Exhaustive experiments demonstrate the robustness and domain adaptability of the proposed framework in nighttime aerial tracking. The code and benchmark are available at https://github.com/vision4robotics/UDAT.

----

## [861] Learning Optical Flow with Kernel Patch Attention

**Authors**: *Ao Luo, Fan Yang, Xin Li, Shuaicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00870](https://doi.org/10.1109/CVPR52688.2022.00870)

**Abstract**:

Optical flow is a fundamental method used for quantitative motion estimation on the image plane. In the deep learning era, most works treat it as a task of ‘matching of features’, learning to pull matched pixels as close as possible in feature space and vice versa. However, spatial affinity (smoothness constraint), another important component for motion understanding, has been largely overlooked. In this paper, we introduce a novel approach, called kernel patch attention (KPA), to better resolve the ambiguity in dense matching by explicitly taking the local context relations into consideration. Our KPA operates on each local patch, and learns to mine the context affinities for better inferring the flow fields. It can be plugged into contemporary optical flow architecture and empower the model to conduct comprehensive motion analysis with both feature similarities and spatial relations. On Sintel dataset, the proposed KPA-Flow achieves the best performance with EPE of 1.35 on clean pass and 2.36 on final pass, and it sets a new record of 4.60% in F1-all on KITTI-15 benchmark. Code is publicly available at https://github.com/megvii-research/KPAFlow.

----

## [862] Towards Understanding Adversarial Robustness of Optical Flow Networks

**Authors**: *Simon Schrodi, Tonmoy Saikia, Thomas Brox*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00871](https://doi.org/10.1109/CVPR52688.2022.00871)

**Abstract**:

Recent work demonstrated the lack of robustness of optical flow networks to physical patch-based adversarial attacks. The possibility to physically attack a basic component of automotive systems is a reason for serious concerns. In this paper, we analyze the cause of the problem and show that the lack of robustness is rooted in the classical aperture problem of optical flow estimation in combination with bad choices in the details of the network architecture. We show how these mistakes can be rectified in order to make optical flow networks robust to physical patch-based attacks. Additionally, we take a look at global white-box attacks in the scope of optical flow. We find that targeted white-box attacks can be crafted to bias flow estimation models towards any desired output, but this requires access to the input images and model weights. However, in the case of universal attacks, we find that optical flow networks are robust. Code is available at https://github.com/lmb-freiburg/understanding_flow_robustness.

----

## [863] DIP: Deep Inverse Patchmatch for High-Resolution Optical Flow

**Authors**: *Zihua Zheng, Ni Nie, Zhi Ling, Pengfei Xiong, Jiangyu Liu, Hao Wang, Jiankun Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00872](https://doi.org/10.1109/CVPR52688.2022.00872)

**Abstract**:

Recently, the dense correlation volume method achieves state-of-the-art performance in optical flow. However, the correlation volume computation requires a lot of memory, which makes prediction difficult on high-resolution images. In this paper, we propose a novel Patchmatch-based framework to work on high-resolution optical flow estimation. Specifically, we introduce the first end-to-end Patchmatch based deep learning optical flow. It can get high-precision results with lower memory benefiting from propagation and local search of Patchmatch. Furthermore, a new inverse propagation is proposed to decouple the complex operations of propagation, which can significantly reduce calculations in multiple iterations. At the time of submission, our method ranks 1st on all the metrics on the popular KITTI2015 [28] benchmark, and ranks 2
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">nd</sup>
 on EPE on the Sintel [7] clean benchmark among published optical flow methods. Experiment shows our method has a strong cross-dataset generalization ability that the F1-all achieves 13.73%, reducing 21% from the best published result 17.4% on KITTI2015. What's more, our method shows a good details preserving result on the high-resolution dataset DAVIS [1] and consumes 2× less memory than RAFT [36]. Code will be available at github.com/zihuarheng/DIP

----

## [864] On the Instability of Relative Pose Estimation and RANSAC's Role

**Authors**: *Hongyi Fan, Joe Kileel, Benjamin B. Kimia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00873](https://doi.org/10.1109/CVPR52688.2022.00873)

**Abstract**:

Relative pose estimation using the 5-point or 7-point Random Sample Consensus (RANSAC) algorithms can fail even when no outliers are present and there are enough inliers to support a hypothesis. These cases arise due to numerical instability of the 5- and 7-point minimal problems. This paper characterizes these instabilities, both in terms of minimal world scene configurations that lead to infinite condition number in epipolar estimation, and also in terms of the related minimal image feature pair correspondence configurations. The instability is studied in the context of a novel framework for analyzing the conditioning of minimal problems in multiview geometry, based on Riemannian manifolds. Experiments with synthetic and real-world data reveal that RANSAC does not only serve to filter out outliers, but RANSAC also selects for well-conditioned image data, sufficiently separated from the ill-posed locus that our theory predicts. These findings suggest that, in future work, one could try to accelerate and increase the success of RANSAC by testing only well-conditioned image data.

----

## [865] Bootstrapping ViTs: Towards Liberating Vision Transformers from Pre-training

**Authors**: *Haofei Zhang, Jiarui Duan, Mengqi Xue, Jie Song, Li Sun, Mingli Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00874](https://doi.org/10.1109/CVPR52688.2022.00874)

**Abstract**:

Recently, vision Transformers (ViTs) are developing rapidly and starting to challenge the domination of con-volutional neural networks (CNNs) in the realm of computer vision (CV). With the general-purpose Transformer architecture replacing the hard-coded inductive biases of convolution, ViTs have surpassed CNNs, especially in data-sufficient circumstances. However, ViTs are prone to over-fit on small datasets and thus rely on large-scale pre-training, which expends enormous time. In this paper, we strive to liberate ViTs from pre-training by introducing CNNs' in- ductive biases back to ViTs while preserving their network architectures for higher upper bound and setting up more suitable optimization objectives. To begin with, an agent CNN is designed based on the given ViT with inductive bi-ases. Then a bootstrapping training algorithm is proposed to jointly optimize the agent and ViT with weight sharing, during which the ViT learns inductive biases from the intermediate features of the agent. Extensive experiments on CIFAR-10/100 and ImageNet-1k with limited training data have shown encouraging results that the inductive biases help ViTs converge significantly faster and outperform conventional CNNs with even fewer parameters. Our code is publicly available at https://github.com/zhfeing/Bootstrapping-ViTs-pytorch.

----

## [866] Global Sensing and Measurements Reuse for Image Compressed Sensing

**Authors**: *Zi-En Fan, Feng Lian, Jia-Ni Quan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00875](https://doi.org/10.1109/CVPR52688.2022.00875)

**Abstract**:

Recently, deep network-based image compressed sensing methods achieved high reconstruction quality and reduced computational overhead compared with traditional methods. However, existing methods obtain measurements only from partial features in the network and use it only once for image reconstruction. They ignore there are low, mid, and high-level features in the network [38] and all of them are essential for high-quality reconstruction. Moreover, using measurements only once may not be enough for extracting richer information from measurements. To address these issues, we propose a novel Measurements Reuse Convolutional Compressed Sensing Network (MR-CCSNet) which employs Global Sensing Module (GSM) to collect all level features for achieving an efficient sensing and Measurements Reuse Block (MRB) to reuse measurements multiple times on multi-scale. Finally, we conduct a series of experiments on three benchmark datasets to show that our model can significantly outperform state-of-the-art methods. Code is available at: https://github.com/fze0012/MR-CCSNet.

----

## [867] Maximum Consensus by Weighted Influences of Monotone Boolean Functions

**Authors**: *Erchuan Zhang, David Suter, Ruwan B. Tennakoon, Tat-Jun Chin, Alireza Bab-Hadiashar, Giang Truong, Syed Zulqarnain Gilani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00876](https://doi.org/10.1109/CVPR52688.2022.00876)

**Abstract**:

Maximisation of Consensus (MaxCon) is one of the most widely used robust criteria in computer vision. Tennakoon et al. (CVPR2021), made a connection between MaxCon and estimation of influences of a Monotone Boolean function. In such, there are two distributions involved: the distribution defining the influence measure; and the distribution used for sampling to estimate the influence measure. This paper studies the concept of weighted influences for solving MaxCon. In particular, we study the Bernoulli measures. Theoretically, we prove the weighted influences, under this measure, of points belonging to larger structures are smaller than those of points belonging to smaller structures in general. We also consider another “natural” family of weighting strategies: sampling with uniform measure concentrated on a particular (Hamming) level of the cube. One can choose to have matching distributions: the same for defining the measure as for implementing the sampling. This has the advantage that the sampler is an unbiased estimator of the measure. Based on weighted sampling, we modify the algorithm of Tennakoon et al., and test on both synthetic and real datasets. We show some modest gains of Bernoulli sampling, and we illuminate some of the interactions between structure in data and weighted measures and weighted sampling.

----

## [868] MS2DG-Net: Progressive Correspondence Learning via Multiple Sparse Semantics Dynamic Graph

**Authors**: *Luanyuan Dai, Yizhang Liu, Jiayi Ma, Lifang Wei, Taotao Lai, Changcai Yang, Riqing Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00877](https://doi.org/10.1109/CVPR52688.2022.00877)

**Abstract**:

Establishing superior-quality correspondences in an image pair is pivotal to many subsequent computer vision tasks. Using Euclidean distance between correspondences to find neighbors and extract local information is a common strategy in previous works. However, most such works ignore similar sparse semantics information between two given images and cannot capture local topology among correspondences well. Therefore, to deal with the above problems, Multiple Sparse Semantics Dynamic Graph Network (MS
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 DG-Net) is proposed, in this paper, to predict probabilities of correspondences as inliers and recover camera poses. MS2 DG-Net dynamically builds sparse semantics graphs based on sparse semantics similarity between two given images, to capture local topology among correspondences, while maintaining permutation-equivariant. Extensive experiments prove that MS2 DG-Net outperforms state-of-the-art methods in outlier removal and camera pose estimation tasks on the public datasets with heavy outliers. Source code:https://github.com/changcaiyang/MS2DG-Net

----

## [869] Styleformer: Transformer based Generative Adversarial Networks with Style Vector

**Authors**: *Jeeseung Park, Younggeun Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00878](https://doi.org/10.1109/CVPR52688.2022.00878)

**Abstract**:

We propose Styleformer, a generator that synthesizes image using style vectors based on the Transformer structure. In this paper, we effectively apply the modified Transformer structure (e.g., Increased multi-head attention and Prelayer normalization) and introduce novel Attention Style Injection module which is style modulation and demodulation method for self-attention operation. The new generator components have strengths in CNN's shortcomings, handling long-range dependency and understanding global structure of objects. We present two methods to generate high-resolution images using Styleformer. First, we apply Linformer in the field of visual synthesis (Styleformer-L), enabling Styleformer to generate higher resolution images and result in improvements in terms of computation cost and performance. This is the first case using Linformer to image generation. Second, we combine Styleformer and Style-GAN2 (Styleformer-C) to generate high-resolution compositional scene efficiently, which Styleformer captures long-range dependencies between components. With these adaptations, Styleformer achieves comparable performances to state-of-the-art in both single and multi-object datasets. Furthermore, groundbreaking results from style mixing and attention map visualization demonstrate the advantages and efficiency of our model.

----

## [870] Scanline Homographies for Rolling-Shutter Plane Absolute Pose

**Authors**: *Fang Bai, Agniva Sengupta, Adrien Bartoli*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00879](https://doi.org/10.1109/CVPR52688.2022.00879)

**Abstract**:

Cameras on portable devices are manufactured with a rolling-shutter (RS) mechanism, where the image rows (aka. scanlines) are read out sequentially. The unknown camera motions during the imaging process cause the so-called RS effects which are solved by motion assumptions in the literature. In this work, we give a solution to the absolute pose problem free of motion assumptions. We categorically demonstrate that the only requirement is motion smoothness instead of stronger constraints on the camera motion. To this end, we propose a novel mathematical abstraction for RS cameras observing a planar scene, called the scanline-homography, a 3 × 2 matrix with 5 DOFs. We establish the relationship between a scanline-homography and the corresponding plane-homography, a 3 × 3 matrix with 6 DOFs assuming the camera is calibrated. We estimate the scanline-homographies of an RS frame using a smooth image warp powered by B-Splines, and recover the plane-homographies afterwards to obtain the scanline-poses based on motion smoothness. We back our claims with various experiments. Code and new datasets: https://bitbucket.org/clermontferrand/planarscanlinehomography/src/master/.

----

## [871] Generating Representative Samples for Few-Shot Classification

**Authors**: *Jingyi Xu, Hieu Le*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00880](https://doi.org/10.1109/CVPR52688.2022.00880)

**Abstract**:

Few-shot learning (FSL) aims to learn new categories with a few visual samples per class. Few-shot class representations are often biased due to data scarcity. To mitigate this issue, we propose to generate visual samples based on semantic embeddings using a conditional variational autoencoder (CVAE) model. We train this CVAE model on base classes and use it to generate features for novel classes. More importantly, we guide this VAE to strictly generate representative samples by removing non-representative samples from the base training set when training the CVAE model. We show that this training scheme enhances the representativeness of the generated samples and therefore, improves the few-shot classification results. Experimental results show that our method improves three FSL baseline methods by substantial margins, achieving state-of-the-art few-shot classification performance on miniImageNet and tieredImageNet datasets for both 1-shot and 5-shot settings. Code is available at: https://github.com/cvlab-stonybrook/fsl-rsvae.

----

## [872] Matching Feature Sets for Few-Shot Image Classification

**Authors**: *Arman Afrasiyabi, Hugo Larochelle, Jean-François Lalonde, Christian Gagné*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00881](https://doi.org/10.1109/CVPR52688.2022.00881)

**Abstract**:

In image classification, it is common practice to train deep networks to extract a single feature vector per input image. Few-shot classification methods also mostly follow this trend. In this work, we depart from this established direction and instead propose to extract sets of feature vectors for each image. We argue that a set-based representation intrinsically builds a richer representation of images from the base classes, which can subsequently better transfer to the few-shot classes. To do so, we propose to adapt existing feature extractors to instead produce sets of feature vectors from images. Our approach, dubbed SetFeat, embeds shallow self-attention mechanisms inside existing encoder architectures. The attention modules are lightweight, and as such our method results in encoders that have approximately the same number of parameters as their original versions. During training and inference, a set-to-set matching metric is used to perform image classification. The effectiveness of our proposed architecture and metrics is demonstrated via thorough experiments on standard few-shot datasets-namely miniImageNet, tieredImageNet, and CUB-in both the 1- and 5-shot scenarios. In all cases but one, our method outperforms the state-of-the-art.

----

## [873] Improving Adversarially Robust Few-shot Image Classification with Generalizable Representations

**Authors**: *Junhao Dong, Yuan Wang, Jianhuang Lai, Xiaohua Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00882](https://doi.org/10.1109/CVPR52688.2022.00882)

**Abstract**:

Few-Shot Image Classification (FSIC) aims to recognize novel image classes with limited data, which is significant in practice. In this paper, we consider the FSIC problem in the case of adversarial examples. This is an extremely challenging issue because current deep learning methods are still vulnerable when handling adversarial examples, even with massive labeled training samples. For this problem, existing works focus on training a network in the meta-learning fashion that depends on numerous sampled few-shot tasks. In comparison, we propose a simple but effective baseline through directly learning generalizable representations without tedious task sampling, which is robust to unforeseen adversarial FSIC tasks. Specifically, we introduce an adversarial-aware mechanism to establish auxiliary supervision via feature-level differences between legitimate and adversarial examples. Furthermore, we design a novel adversarial-reweighted training manner to alleviate the imbalance among adversarial examples. The feature purifier is also employed as post-processing for adversarial features. Moreover, our method can obtain generalizable representations to remain superior transferability, even facing cross-domain adversarial examples. Extensive experiments show that our method can significantly outperform state-of-the-art adversarially robust FSIC methods on two standard benchmarks.

----

## [874] Sylph: A Hypernetwork Framework for Incremental Few-shot Object Detection

**Authors**: *Li Yin, Juan M. Perez-Rua, Kevin J. Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00883](https://doi.org/10.1109/CVPR52688.2022.00883)

**Abstract**:

We study the challenging incremental few-shot object de-tection (iFSD) setting. Recently, hypernetwork-based approaches have been studied in the context of continuous and finetune-free iFSD with limited success. We take a closer look at important design choices of such methods, leading to several key improvements and resulting in a more accurate and flexible framework, which we call Sylph. In particular, we demonstrate the effectiveness of decou-pling object classification from localization by leveraging a base detector that is pretrained for class-agnostic local-ization on large-scale dataset. Contrary to what previous results have suggested, we show that with a carefully de-signed class-conditional hypernetwork, finetune-free iFSD can be highly effective, especially when a large number of base categories with abundant data are available for meta-training, almost approaching alternatives that undergo test-time-training. This result is even more significant considering its many practical advantages: (1) incrementally learning new classes in sequence without additional training, (2) detecting both novel and seen classes in a single pass, and (3) no forgetting of previously seen classes. We benchmark our model on both COCO and LVIS, reporting as high as 17% AP on the long-tail rare classes on LVIS, indicating the promise of hypernetwork-based iFSD.

----

## [875] Forward Compatible Few-Shot Class-Incremental Learning

**Authors**: *Da-Wei Zhou, Fu-Yun Wang, Han-Jia Ye, Liang Ma, Shiliang Pu, De-Chuan Zhan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00884](https://doi.org/10.1109/CVPR52688.2022.00884)

**Abstract**:

Novel classes frequently arise in our dynamically changing world, e.g., new users in the authentication system, and a machine learning model should recognize new classes without forgetting old ones. This scenario becomes more challenging when new class instances are insufficient, which is called few-shot class-incremental learning (FSCIL). Cur-rent methods handle incremental learning retrospectively by making the updated model similar to the old one. By contrast, we suggest learning prospectively to prepare for future updates, and propose ForwArd Compatible Training (FACT) for FSCIL. Forward compatibility requires future new classes to be easily incorporated into the current model based on the current stage data, and we seek to realize it by reserving embedding space for future new classes. In detail, we assign virtual prototypes to squeeze the embedding of known classes and reserve for new ones. Besides, we forecast possible new classes and prepare for the updating process. The virtual prototypes allow the model to accept possible updates in the future, which act as proxies scattered among embedding space to build a stronger classifier during inference. Fact efficiently incorporates new classes with forward compatibility and meanwhile resists for-getting of old ones. Extensive experiments validate FACT's state-of-the-art performance. Code is available at: https://github.com/zhoudw-zdw/CVPR22-Fact

----

## [876] Constrained Few-shot Class-incremental Learning

**Authors**: *Michael Hersche, Geethan Karunaratne, Giovanni Cherubini, Luca Benini, Abu Sebastian, Abbas Rahimi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00885](https://doi.org/10.1109/CVPR52688.2022.00885)

**Abstract**:

Continually learning new classes from fresh data without forgetting previous knowledge of old classes is a very challenging research problem. Moreover, it is imperative that such learning must respect certain memory and computational constraints such as (i) training samples are limited to only a few per class, (ii) the computational cost of learning a novel class remains constant, and (iii) the memory footprint of the model grows at most linearly with the number of classes observed. To meet the above constraints, we propose C-FSCIL, which is architecturally composed of a frozen meta-learned feature extractor, a trainable fixed-size fully connected layer, and a rewritable dynamically growing memory that stores as many vectors as the number of encountered classes. C-FSCIL provides three update modes that offer a trade-off between accuracy and compute-memory cost of learning novel classes. C-FSCIL exploits hyperdimensional embedding that allows to continually express many more classes than the fixed dimensions in the vector space, with minimal interference. The quality of class vector representations is further improved by aligning them quasi-orthogonally to each other by means of novel loss functions. Experiments on the CIFAR100, mini-ImageNet, and Omniglot datasets show that C-FSCIL outperforms the baselines with remarkable accuracy and compression. It also scales up to the largest problem size ever tried in this few-shot setting by learning 423 novel classes on top of 1200 base classes with less than 1.6% accuracy drop. Our code is available at https://github.com/IBM/constrained-FSCIL.

----

## [877] Pushing the Limits of Simple Pipelines for Few-Shot Learning: External Data and Fine-Tuning Make a Difference

**Authors**: *Shell Xu Hu, Da Li, Jan Stühmer, Minyoung Kim, Timothy M. Hospedales*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00886](https://doi.org/10.1109/CVPR52688.2022.00886)

**Abstract**:

Few-shot learning (FSL) is an important and topical problem in computer vision that has motivated extensive research into numerous methods spanning from sophisticated metalearning methods to simple transfer learning baselines. We seek to push the limits of a simple-but-effective pipeline for real-worldfew-shot image classification in practice. To this end, we explore few-shot learning from the perspective of neural architecture, as well as a three stage pipeline of pre-training on external data, meta-training with labelled few-shot tasks, and task-specific fine-tuning on unseen tasks. We investigate questions such as: ① How pre-training on external data benefits FSL? ② How state of the art transformer architectures can be exploited? and ③ How to best exploit finetuning? Ultimately, we show that a simple transformer-based pipeline yields surprisingly good performance on standard benchmarks such as Mini-ImageNet, CIFAR-FS, CDFSL and Meta-Dataset. Our code is available at https://hushell.github.io/pmf.

----

## [878] EASE: Unsupervised Discriminant Subspace Learning for Transductive Few-Shot Learning

**Authors**: *Hao Zhu, Piotr Koniusz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00887](https://doi.org/10.1109/CVPR52688.2022.00887)

**Abstract**:

Few-shot learning (FSL) has received a lot of attention due to its remarkable ability to adapt to novel classes. Although many techniques have been proposed for FSL, they mostly focus on improving FSL backbones. Some works also focus on learning on top of the features generated by these backbones to adapt them to novel classes. We present an unsupErvised discriminAnt Subspace lEarning (EASE) that improves transductive few-shot learning performance by learning a linear projection onto a subspace built from features of the support set and the unlabeled query set in the test time. Specifically, based on the support set and the unlabeled query set, we generate the similarity matrix and the dissimilarity matrix based on the structure prior for the proposed EASE method, which is efficiently solved with SVD. We also introduce conStraIned wAsserstein MEan Shift clustEring (SIAMESE) which extends Sinkhorn K-means by incorporating labeled support samples. SIAMESE works on the features obtained from EASE to estimate class centers and query predictions. On the miniImageNet, tiered-ImageNet, CIFAR-FS, CUB and OpenMIC benchmarks, both steps significantly boost the performance in transductive FSL and semi-supervised FSL.

----

## [879] Few-shot Learning with Noisy Labels

**Authors**: *Kevin J. Liang, Samrudhdhi B. Rangrej, Vladan Petrovic, Tal Hassner*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00888](https://doi.org/10.1109/CVPR52688.2022.00888)

**Abstract**:

Few-shot learning (FSL) methods typically assume clean support sets with accurately labeled samples when training on novel classes. This assumption can often be unrealistic: support sets, no matter how small, can still include mislabeled samples. Robustness to label noise is therefore essential for FSL methods to be practical, but this problem surprisingly remains largely unexplored. To address mislabeled samples in FSL settings, we make several technical contributions. (1) We offer simple, yet effective, feature aggregation methods, improving the prototypes used by ProtoNet, a popular FSL technique. (2) We describe a novel Transformer model for Noisy Few-Shot Learning (TraNFS). TraNFS leverages a transformer's attention mechanism to weigh mislabeled versus correct samples. (3) Finally, we extensively test these methods on noisy versions of MinilmageNet and TieredImageNet. Our results show that TraNFS is on-par with leading FSL methods on clean support sets, yet outperforms them, by far, in the presence of label noise.

----

## [880] Ranking Distance Calibration for Cross-Domain Few-Shot Learning

**Authors**: *Pan Li, Shaogang Gong, Chengjie Wang, Yanwei Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00889](https://doi.org/10.1109/CVPR52688.2022.00889)

**Abstract**:

Recent progress in few-shot learning promotes a more realistic cross-domain setting, where the source and target datasets are in different domains. Due to the domain gap and disjoint label spaces between source and target datasets, their shared knowledge is extremely limited. This encourages us to explore more information in the target domain rather than to overly elaborate training strategies on the source domain as in many existing methods. Hence, we start from a generic representation pre-trained by a cross-entropy loss and a conventional distance-based classifier, along with an image retrieval view, to employ a re-ranking process to calibrate a target distance matrix by discovering the k-reciprocal neighbours within the task. Assuming the pre-trained representation is biased towards the source, we construct a non-linear subspace to minimise task-irrelevant features therewithin while keep more transferrable discriminative information by a hyperbolic tangent transformation. The calibrated distance in this target-aware non-linear sub-space is complementary to that in the pre-trained representation. To impose such distance calibration information onto the pre-trained representation, a Kullback-Leibler divergence loss is employed to gradually guide the model towards the calibrated distance-based distribution. Extensive evaluations on eight target domains show that this target ranking calibration process can improve conventional distance-based classifiers in few-shot learning.

----

## [881] Revisiting Learnable Affines for Batch Norm in Few-Shot Transfer Learning

**Authors**: *Moslem Yazdanpanah, Aamer Abdul Rahman, Muawiz Chaudhary, Christian Desrosiers, Mohammad Havaei, Eugene Belilovsky, Samira Ebrahimi Kahou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00890](https://doi.org/10.1109/CVPR52688.2022.00890)

**Abstract**:

Batch normalization is a staple of computer vision models, including those employed in few-shot learning. Batch nor-malization layers in convolutional neural networks are composed of a normalization step, followed by a shift and scale of these normalized features applied via the per-channel trainable affine parameters 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\gamma$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\beta$</tex>
. These affine param-eters were introduced to maintain the expressive powers of the model following normalization. While this hypothesis holds true for classification within the same domain, this work illustrates that these parameters are detrimen-tal to downstream performance on common few-shot trans-fer tasks. This effect is studied with multiple methods on well-known benchmarks such as few-shot classification on minilmageNet, cross-domain few-shot learning (CD-FSL) and META-DATASET. Experiments reveal consistent performance improvements on CNNs with affine unaccompanied batch normalization layers; particularly in large domain-shift few-shot transfer settings. As opposed to common practices in few-shot transfer learning where the affine pa-rameters are fixed during the adaptation phase, we show fine-tuning them can lead to improved performance.

----

## [882] Attribute Surrogates Learning and Spectral Tokens Pooling in Transformers for Few-shot Learning

**Authors**: *Yangji He, Weihan Liang, Dongyang Zhao, Hong-Yu Zhou, Weifeng Ge, Yizhou Yu, Wenqiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00891](https://doi.org/10.1109/CVPR52688.2022.00891)

**Abstract**:

This paper presents new hierarchically cascaded transformers that can improve data efficiency through attribute surrogates learning and spectral tokens pooling. Vision transformers have recently been thought of as a promising alternative to convolutional neural networks for visual recognition. But when there is no sufficient data, it gets stuck in overfitting and shows inferior performance. To improve data efficiency, we propose hierarchically cascaded transformers that exploit intrinsic image structures through spectral tokens pooling and optimize the learnable parameters through latent attribute surrogates. The intrinsic image structure is utilized to reduce the ambiguity between foreground content and background noise by spectral tokens pooling. And the attribute surrogate learning scheme is designed to benefit from the rich visual information in image-label pairs instead of simple visual concepts assigned by their labels. Our Hierarchically Cascaded Transformers, called HCTransformers, is built upon a self-supervised learning framework DINO and is tested on several popular few-shot learning benchmarks. In the inductive setting, HCTransformers surpass the DINO baseline by a large margin of 9.7% 5-way 1-shot accuracy and 9.17% 5-way 5-shot accuracy on miniImageNet, which demonstrates HCTransformers are efficient to extract discriminative features. Also, HCTransformers show clear advantages over SOTA few-shot classification methods in both 5-way 1-shot and 5-way 5-shot settings on four popular benchmark datasets, including miniImageNet, tieredImageNet, FC100, and CIFAR-FS. The trained weights and codes are available at https://github.com/StomachCold/HCTransformers.

----

## [883] Learning to Memorize Feature Hallucination for One-Shot Image Generation

**Authors**: *Yu Xie, Yanwei Fu, Ying Tai, Yun Cao, Junwei Zhu, Chengjie Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00892](https://doi.org/10.1109/CVPR52688.2022.00892)

**Abstract**:

This paper studies the task of One-Shot image Generation (OSG), where generation network learned on base dataset should be generalizable to synthesize images of novel categories with only one available sample per novel category. Most existing methods for feature transfer in oneshot image generation only learn reusable features implicitly on pre-training tasks. Such methods would be likely to overfit pre-training tasks. In this paper, we propose a novel model to explicitly learn and memorize reusable features that can help hallucinate novel category images. To be specific, our algorithm learns to decompose image features into the Category-Related (CR) and Category-Independent(CI) features. Our model learning to memorize class-independent CI features which are further utilized by our feature hallucination component to generate target novel category images. We validate our model on several benchmarks. Extensive experiments demonstrate that our model effectively boosts the OSG performance and can generate compelling and diverse samples.

----

## [884] A Closer Look at Few-shot Image Generation

**Authors**: *Yunqing Zhao, Henghui Ding, Houjing Huang, Ngai-Man Cheung*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00893](https://doi.org/10.1109/CVPR52688.2022.00893)

**Abstract**:

Modern GANs excel at generating high quality and diverse images. However, when transferring the pretrained GANs on small target data (e.g., 10-shot), the generator tends to replicate the training samples. Several methods have been proposed to address this few-shot image generation task, but there is a lack of effort to analyze them under a unified framework. As our first contribution, we propose a framework to analyze existing methods during the adaptation. Our analysis discovers that while some methods have disproportionate focus on diversity preserving which impede quality improvement, all methods achieve similar quality after convergence. Therefore, the better methods are those that can slow down diversity degradation. Furthermore, our analysis reveals that there is still plenty of room to further slow down diversity degradation. Informed by our analysis and to slow down the diversity degradation of the target generator during adaptation, our second contribution proposes to apply mutual information (MI) maximization to retain the source domain's rich multi-level diversity information in the target domain generator. We propose to perform MI maximization by contrastive loss (CL), leverage the generator and discriminator as two feature encoders to extract different multi-level features for computing CL. We refer to our method as Dual Contrastive Learning (DCL). Extensive experiments on several public datasets show that, while leading to a slower diversity-degrading generator during adaptation, our proposed DCL brings visually pleasant quality and state-of-the-art quantitative performance.

----

## [885] Motion-modulated Temporal Fragment Alignment Network For Few-Shot Action Recognition

**Authors**: *Jiamin Wu, Tianzhu Zhang, Zhe Zhang, Feng Wu, Yongdong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00894](https://doi.org/10.1109/CVPR52688.2022.00894)

**Abstract**:

While the majority of FSL models focus on image classification, the extension to action recognition is rather challenging due to the additional temporal dimension in videos. To address this issue, we propose an end-to-end Motion-modulated Temporal Fragment Alignment Network (MT-FAN) by jointly exploring the task-specific motion modulation and the multi-level temporal fragment alignment for Few-Shot Action Recognition (FSAR). The proposed MT-FAN model enjoys several merits. First, we design a motion modulator conditioned on the learned task-specific motion embeddings, which can activate the channels related to the task-shared motion patterns for each frame. Second, a segment attention mechanism is proposed to automatically discover the higher-level segments for multi-level temporal fragment alignment, which encompasses the frame-to-frame, segment-to-segment, and segment-to-frame alignments. To the best of our knowledge, this is the first work to exploit task-specific motion modulation for FSAR. Extensive experimental results on four standard benchmarks demonstrate that the proposed model performs favorably against the state-of-the-art FSAR methods.

----

## [886] Knowledge Distillation as Efficient Pre-training: Faster Convergence, Higher Data-efficiency, and Better Transferability

**Authors**: *Ruifei He, Shuyang Sun, Jihan Yang, Song Bai, Xiaojuan Qi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00895](https://doi.org/10.1109/CVPR52688.2022.00895)

**Abstract**:

Large-scale pre-training has been proven to be crucial for various computer vision tasks. However, with the increase of pre-training data amount, model architecture amount, and the private/inaccessible data, it is not very efficient or possible to pre-train all the model architectures on large-scale datasets. In this work, we investigate an alternative strategy for pre-training, namely Knowledge Distillation as Efficient Pre-training (KDEP), aiming to efficiently transfer the learned feature representation from existing pre-trained models to new student models for future downstream tasks. We observe that existing Knowledge Distillation (KD) methods are unsuitable towards pre-training since they normally distill the logits that are going to be discarded when transferred to downstream tasks. To resolve this problem, we propose a feature-based KD method with non-parametric feature dimension aligning. Notably, our method performs comparably with supervised pre-training counterparts in 3 downstream tasks and 9 downstream datasets requiring 10× less data and 5× less pre-training time. Code is available at https://github.com/CVMI-Lab/KDEP.

----

## [887] Transferability Estimation using Bhattacharyya Class Separability

**Authors**: *Michal Pándy, Andrea Agostinelli, Jasper R. R. Uijlings, Vittorio Ferrari, Thomas Mensink*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00896](https://doi.org/10.1109/CVPR52688.2022.00896)

**Abstract**:

Transfer learning has become a popular method for leveraging pre-trained models in computer vision. However, without performing computationally expensive fine-tuning, it is difficult to quantify which pre-trained source models are suitable for a specific target task, or, conversely, to which tasks a pre-trained source model can be easily adapted to. In this work, we propose Gaussian Bhattacharyya Coefficient (GBC), a novel method for quantifying transferability between a source model and a target dataset. In a first step we embed all target images in the feature space defined by the source model, and represent them with per-class Gaussians. Then, we estimate their pairwise class separability using the Bhattacharyya coefficient, yielding a simple and effective measure of how well the source model transfers to the target task. We evaluate GBC on image classification tasks in the context of dataset and architecture selection. Further, we also perform experiments on the more complex semantic segmentation transferability estimation task. We demonstrate that GBC outperforms state-of-the-art transferability metrics on most evaluation criteria in the semantic segmentation settings, matches the performance of top methods for dataset transferability in image classification, and performs best on architecture selection problems for image classification.

----

## [888] Revisiting the Transferability of Supervised Pretraining: an MLP Perspective

**Authors**: *Yizhou Wang, Shixiang Tang, Feng Zhu, Lei Bai, Rui Zhao, Donglian Qi, Wanli Ouyang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00897](https://doi.org/10.1109/CVPR52688.2022.00897)

**Abstract**:

The pretrain-finetune paradigm is a classical pipeline in visual learning. Recent progress on unsupervised pretraining methods shows superior transfer performance to their supervised counterparts. This paper revisits this phenomenon and sheds new light on understanding the transferability gap between unsupervised and supervised pretraining from a multilayer perceptron (MLP) perspective. While previous works [6], [8], [17] focus on the effectiveness of MLP on unsupervised image classification where pretraining and evaluation are conducted on the same dataset, we reveal that the MLP projector is also the key factor to better transferability of unsupervised pretraining methods than supervised pretraining methods. Based on this observation, we attempt to close the transferability gap between supervised and unsupervised pretraining by adding an MLP projector before the classifier in supervised pretraining. Our analysis indicates that the MLP projector can help retain intra-class variation of visual features, decrease the feature distribution distance between pretraining and evaluation datasets, and reduce feature redundancy. Extensive experiments on public benchmarks demonstrate that the added MLP projector significantly boosts the transferability of supervised pretraining, e.g. +7.2% top-1 accuracy on the concept generalization task, +5.8% top-1 accuracy for linear evaluation on 12 -domain classification tasks, and +0.8% AP on COCO object detection task, making supervised pretraining comparable or even better than unsupervised pretraining.

----

## [889] Task2Sim: Towards Effective Pre-training and Transfer from Synthetic Data

**Authors**: *Samarth Mishra, Rameswar Panda, Cheng Perng Phoo, Chun-Fu Richard Chen, Leonid Karlinsky, Kate Saenko, Venkatesh Saligrama, Rogério Schmidt Feris*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00898](https://doi.org/10.1109/CVPR52688.2022.00898)

**Abstract**:

Pre-training models on Imagenet or other massive datasets of real images has led to major advances in Computer vision, albeit accompanied with shortcomings related to curation cost, privacy, usage rights, and ethical issues. In this paper, for the first time, we study the transferability of pre-trained models based on synthetic data generated by graphics simulators to downstream tasks from very different domains. In using such synthetic data for pre-training, we find that downstream performance on different tasks are fa-vored by different configurations of simulation parameters (e.g. lighting, object pose, backgrounds, etc.), and that there is no one-size-fits-all solution. It is thus better to tailor syn-thetic pre-training data to a specific downstream task, for best performance. We introduce Task2Sim, a unified model mapping downstream task representations to optimal sim-ulation parameters to generate synthetic pre-training data for them. Task2Sim learns this mapping by training to find the set of best parameters on a set of “seen” tasks. Once trained, it can then be used to predict best simulation pa-rameters for novel “unseen” tasks in one shot, without re-quiring additional training. Given a budget in number of images per class, our extensive experiments with 20 di-verse downstream tasks show Task2Sim's task-adaptive pre-training data results in significantly better downstream per-formance than non-adaptively choosing simulation param-eters on both seen and unseen tasks. It is even competitive with pre-training on real images from Imagenet.

----

## [890] Which Model to Transfer? Finding the Needle in the Growing Haystack

**Authors**: *Cédric Renggli, André Susano Pinto, Luka Rimanic, Joan Puigcerver, Carlos Riquelme, Ce Zhang, Mario Lucic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00899](https://doi.org/10.1109/CVPR52688.2022.00899)

**Abstract**:

Transfer learning has been recently popularized as a data-efficient alternative to training models from scratch, in particular for computer vision tasks where it provides a remarkably solid baseline. The emergence of rich model repositories, such as TensorFlow Hub, enables the practitioners and researchers to unleash the potential of these models across a wide range of downstream tasks. As these repositories keep growing exponentially, efficiently selecting a good model for the task at hand becomes paramount. We provide a formalization of this problem through afamiliar notion of regret and introduce the predominant strategies, namely task-agnostic (e.g. ranking models by their ImageNet performance) and task-aware search strategies (such as linear or kNN evaluation). We conduct a large-scale empirical study and show that both task-agnostic and task-aware methods can yield high regret. We then propose a simple and computationally efficient hybrid search strategy which outperforms the existing approaches. We highlight the practical benefits of the proposed solution on a set of 19 diverse vision tasks.

----

## [891] Does Robustness on ImageNet Transfer to Downstream Tasks?

**Authors**: *Yutaro Yamada, Mayu Otani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00900](https://doi.org/10.1109/CVPR52688.2022.00900)

**Abstract**:

As clean ImageNet accuracy nears its ceiling, the re-search community is increasingly more concerned about ro-bust accuracy under distributional shifts. While a variety of methods have been proposed to robustify neural networks, these techniques often target models trained on ImageNet classification. At the same time, it is a common practice to use ImageNet pretrained backbones for downstream tasks such as object detection, semantic segmentation, and image classification from different domains. This raises a question: Can these robust image classifiers transfer robustness to downstream tasks? For object detection and semantic segmentation, we find that a vanilla Swin Transformer, a variant of Vision Transformer tailored for dense prediction tasks, transfers robustness better than Convolutional Neu-ral Networks that are trained to be robust to the corrupted version of ImageNet. For CIFAR10 classification, we find that models that are robustified for ImageNet do not re-tain robustness when fully fine-tuned. These findings sug-gest that current robustification techniques tend to empha-size ImageNet evaluations. Moreover, network architecture is a strong source of robustness when we consider transfer learning.

----

## [892] What Makes Transfer Learning Work for Medical Images: Feature Reuse & Other Factors

**Authors**: *Christos Matsoukas, Johan Fredin Haslum, Moein Sorkhei, Magnus Söderberg, Kevin Smith*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00901](https://doi.org/10.1109/CVPR52688.2022.00901)

**Abstract**:

Transfer learning is a standard technique to transfer knowledge from one domain to another. For applications in medical imaging, transfer from ImageNet has become the de-facto approach, despite differences in the tasks and im-age characteristics between the domains. However, it is un-clear what factors determine whether - and to what extent- transfer learning to the medical domain is useful. The long- standing assumption that features from the source domain get reused has recently been called into question. Through a series of experiments on several medical image bench-mark datasets, we explore the relationship between transfer learning, data size, the capacity and inductive bias of the model, as well as the distance between the source and tar-get domain. Our findings suggest that transfer learning is beneficial in most cases, and we characterize the important role feature reuse plays in its success.

----

## [893] OW-DETR: Open-world Detection Transformer

**Authors**: *Akshita Gupta, Sanath Narayan, K. J. Joseph, Salman Khan, Fahad Shahbaz Khan, Mubarak Shah*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00902](https://doi.org/10.1109/CVPR52688.2022.00902)

**Abstract**:

Open-world object detection (OWOD) is a challenging computer vision problem, where the task is to detect a known set of object categories while simultaneously identifying unknown objects. Additionally, the model must incrementally learn new classes that become known in the next training episodes. Distinct from standard object detection, the OWOD setting poses significant challenges for generating quality candidate proposals on potentially unknown objects, separating the unknown objects from the background and detecting diverse unknown objects. Here, we introduce a novel end-to-end transformer-based framework, OW-DETR, for open-world object detection. The proposed OW-DETR comprises three dedicated components namely, attention-driven pseudo-labeling, novelty classification and objectness scoring to explicitly address the aforementioned OWOD challenges. Our OW-DETR explicitly encodes multi-scale contextual information, possesses less inductive bias, enables knowledge transfer from known classes to the unknown class and can better discriminate between unknown objects and background. Comprehensive experiments are performed on two benchmarks: MS-COCO and PASCAL VOC. The extensive ablations reveal the merits of our proposed contributions. Further, our model out-performs the recently introduced OWOD approach, ORE, with absolute gains ranging from 1.8% to 3.3% in terms of unknown recall on MS-COCO. In the case of incremental object detection, OW-DETR outperforms the state-of-the-art for all settings on PASCAL VOC. Our code is available at https://github.com/akshitac8/OW-DEtr.

----

## [894] Unseen Classes at a Later Time? No Problem

**Authors**: *Hari Chandana Kuchibhotla, Sumitra S. Malagi, Shivam Chandhok, Vineeth N. Balasubramanian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00903](https://doi.org/10.1109/CVPR52688.2022.00903)

**Abstract**:

Recent progress towards learning from limited supervision has encouraged efforts towards designing models that can recognize novel classes at test time (generalized zero-shot learning or GZSL). GZSL approaches assume knowledge of all classes, with or without labeled data, beforehand. However, practical scenarios demand models that are adaptable and can handle dynamic addition of new seen and unseen classes on the fly (i.e continual generalized zero-shot learning or CGZSL). One solution is to sequentially retrain and reuse conventional GZSL methods, however, such an approach suffers from catastrophic forgetting leading to suboptimal generalization performance. A few recent efforts towards tackling CGZSL have been limited by difference in settings, practicality, data splits and protocols followed - inhibiting fair comparison and a clear direction forward. Motivated from these observations, in this work, we firstly consolidate the different CGZSL setting variants and propose a new Online-CGZSL setting which is more practical and flexible. Secondly, we introduce a unified feature-generative framework for CGZSL that leverages bi-directional incremental alignment to dynamically adapt to addition of new classes, with or without labeled data, that arrive over time in any of these CGZSL settings. Our comprehensive experiments and analysis on five benchmark datasets and comparison with baselines show that our approach consistently outperforms existing methods, especially on the more practical Online setting.

----

## [895] Continual Object Detection via Prototypical Task Correlation Guided Gating Mechanism

**Authors**: *Binbin Yang, Xinchi Deng, Han Shi, Changlin Li, Gengwei Zhang, Hang Xu, Shen Zhao, Liang Lin, Xiaodan Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00904](https://doi.org/10.1109/CVPR52688.2022.00904)

**Abstract**:

Continual learning is a challenging real-world problem for constructing a mature AI system when data are provided in a streaming fashion. Despite recent progress in continual classification, the researches of continual object detection are impeded by the diverse sizes and numbers of objects in each image. Different from previous works that tune the whole network for all tasks, in this work, we present a simple and flexible framework for continual object detection via pRotOtypical taSk corrElaTion guided gaTing mechAnism (ROSETTA). Concretely, a unified framework is shared by all tasks while task-aware gates are introduced to automatically select sub-models for specific tasks. In this way, various knowledge can be successively memorized by storing their corresponding sub-model weights in this system. To make ROSETTA automatically determine which experience is available and useful, a prototypical task correlation guided Gating Diversity Controller (GDC) is introduced to adaptively adjust the diversity of gates for the new task based on class-specific prototypes. GDC module computes class-to-class correlation matrix to depict the cross-task correlation, and hereby activates more exclusive gates for the new task if a significant domain gap is observed. Comprehensive experiments on COCO-VOC, KITTI-Kitchen, class-incremental detection on VOC and sequential learning of four tasks show that ROSETTA yields state-of-the-art performance on both task-based and class-based continual object detection.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Codes are available at: https://github.com/dkxocl/ROSSETA.

----

## [896] On Generalizing Beyond Domains in Cross-Domain Continual Learning

**Authors**: *Christian Simon, Masoud Faraki, Yi-Hsuan Tsai, Xiang Yu, Samuel Schulter, Yumin Suh, Mehrtash Harandi, Manmohan Chandraker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00905](https://doi.org/10.1109/CVPR52688.2022.00905)

**Abstract**:

Humans have the ability to accumulate knowledge of new tasks in varying conditions, but deep neural networks of-ten suffer from catastrophic forgetting of previously learned knowledge after learning a new task. Many recent methods focus on preventing catastrophic forgetting under the assumption of train and test data following similar distributions. In this work, we consider a more realistic scenario of continual learning under domain shifts where the model must generalize its inference to an unseen domain. To this end, we encourage learning semantically meaningful features by equipping the classifier with class similarity metrics as learning parameters which are obtained through Mahalanobis similarity computations. Learning of the backbone representation along with these extra parameters is done seamlessly in an end-to-end manner. In addition, we propose an approach based on the exponential moving average of the parameters for better knowledge distillation. We demonstrate that, to a great extent, existing continual learning algorithms fail to handle the forgetting issue under multiple distributions, while our proposed approach learns new tasks under domain shift with accuracy boosts up to 10% on challenging datasets such as DomainNet and OfficeHome.

----

## [897] Online Continual Learning on a Contaminated Data Stream with Blurry Task Boundaries

**Authors**: *Jihwan Bang, Hyunseo Koh, Seulki Park, Hwanjun Song, Jung-Woo Ha, Jonghyun Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00906](https://doi.org/10.1109/CVPR52688.2022.00906)

**Abstract**:

Learning under a continuously changing data distribution with incorrect labels is a desirable real-world problem yet challenging. A large body of continual learning (CL) methods, however, assumes data streams with clean labels, and online learning scenarios under noisy data streams are yet underexplored. We consider a more practical CL task setup of an online learning from blurry data stream with corrupted labels, where existing CL methods struggle. To address the task, we first argue the importance of both diversity and purity of examples in the episodic memory of continual learning models. To balance diversity and purity in the episodic memory, we propose a novel strategy to manage and use the memory by a unified approach of label noise aware diverse sampling and robust learning with semi-supervised learning. Our empirical validations on four real-world or synthetic noise datasets (CI-FAR10 and 100, mini-WebVision, and Food-101N) exhibit that our method significantly outperforms prior arts in this realistic and challenging continual learning scenario. Code and data splits are available in https://github.com/clovaai/puridiver.

----

## [898] DyTox: Transformers for Continual Learning with DYnamic TOken eXpansion

**Authors**: *Arthur Douillard, Alexandre Ramé, Guillaume Couairon, Matthieu Cord*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00907](https://doi.org/10.1109/CVPR52688.2022.00907)

**Abstract**:

Deep network architectures struggle to continually learn new tasks without forgetting the previous tasks. A recent trend indicates that dynamic architectures based on an ex-pansion of the parameters can reduce catastrophic forget-ting efficiently in continual learning. However, existing approaches often require a task identifier at test-time, need complex tuning to balance the growing number of parameters, and barely share any information across tasks. As a result, they struggle to scale to a large number of tasks without significant overhead. In this paper, we propose a transformer architecture based on a dedicated encoder/decoder framework. Critically, the encoder and decoder are shared among all tasks. Through a dynamic expansion of special tokens, we specialize each forward of our decoder network on a task distribution. Our strategy scales to a large number of tasks while having neg-ligible memory and time overheads due to strict control of the expansion of the parameters. Moreover, this efficient strategy doesn't need any hyperparameter tuning to control the network's expansion. Our model reaches excellent results on CIFAR100 and state-of-the-art performances on the large-scale ImageNet100 and ImageNet100 while having fewer parameters than concurrent dynamic frameworks.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is released at https://github.com/arthurdouillard/dytox.

----

## [899] Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning

**Authors**: *Kai Zhu, Wei Zhai, Yang Cao, Jiebo Luo, Zhengjun Zha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00908](https://doi.org/10.1109/CVPR52688.2022.00908)

**Abstract**:

Non-exemplar class-incremental learning is to recognize both the old and new classes when old class samples cannot be saved. It is a challenging task since representation optimization and feature retention can only be achieved under supervision from new classes. To address this problem, we propose a novel self-sustaining representation expansion scheme. Our scheme consists of a structure reorganization strategy that fuses main-branch expansion and side-branch updating to maintain the old features, and a main-branch distillation scheme to transfer the invariant knowledge. Furthermore, a prototype selection mechanism is proposed to enhance the discrimination between the old and new classes by selectively incorporating new samples into the distillation process. Extensive experiments on three benchmarks demonstrate significant incremental performance, outperforming the state-of-the-art methods by a margin of 3%, 3% and 6%, respectively.

----

## [900] En-Compactness: Self-Distillation Embedding & Contrastive Generation for Generalized Zero-Shot Learning

**Authors**: *Xia Kong, Zuodong Gao, Xiaofan Li, Ming Hong, Jun Liu, Chengjie Wang, Yuan Xie, Yanyun Qu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00909](https://doi.org/10.1109/CVPR52688.2022.00909)

**Abstract**:

Generalized zero-shot learning (GZSL) requires a classifier trained on seen classes that can recognize objects from both seen and unseen classes. Due to the absence of unseen training samples, the classifier tends to bias towards seen classes. To mitigate this problem, feature generation based models are proposed to synthesize visual features for unseen classes. However, these features are generated in the visual feature space which lacks of discriminative ability. Therefore, some methods turn to find a better embedding space for the classifier training. They emphasize the inter-class relationships of seen classes, leading the embedding space overfitted to seen classes and unfriendly to unseen classes. Instead, in this paper, we propose an Intra-Class Compactness Enhancement method (ICCE) for GZSL. Our ICCE promotes intra-class compactness with inter-class separability on both seen and unseen classes in the embedding space and visual feature space. By promoting the intra-class relationships but the inter-class structures, we can distinguish different classes with better generalization. Specifically, we propose a Self-Distillation Embedding (SDE) module and a Semantic-Visual Contrastive Generation (SVCG) module. The former promotes intra-class compactness in the embedding space, while the latter accomplishes it in the visual feature space. The experiments demonstrate that our ICCE outperforms the state-of-the-art methods on four datasets and achieves competitive results on the remaining dataset.

----

## [901] VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning

**Authors**: *Wenjia Xu, Yongqin Xian, Jiuniu Wang, Bernt Schiele, Zeynep Akata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00910](https://doi.org/10.1109/CVPR52688.2022.00910)

**Abstract**:

Human-annotated attributes serve as powerful semantic embeddings in zero-shot learning. However, their annotation process is labor-intensive and needs expert supervision. Current unsupervised semantic embeddings, i.e., word embeddings, enable knowledge transfer between classes. However, word embeddings do not always reflect visual similarities and result in inferior zero-shot performance. We propose to discover semantic embeddings containing discriminative visual properties for zero-shot learning, without requiring any human annotation. Our model visually divides a set of images from seen classes into clusters of local image regions according to their visual similarity, and further imposes their class discrimination and semantic relatedness. To associate these clusters with previously unseen classes, we use external knowledge, e.g., word embeddings and propose a novel class relation discovery module. Through quantitative and qualitative evaluation, we demonstrate that our model discovers semantic embeddings that model the visual properties of both seen and unseen classes. Furthermore, we demonstrate on three benchmarks that our visually-grounded semantic embeddings further improve performance over word embeddings across various ZSL models by a large margin. Code is available at https://github.com/wenjiaXu/VGSE

----

## [902] Siamese Contrastive Embedding Network for Compositional Zero-Shot Learning

**Authors**: *Xiangyu Li, Xu Yang, Kun Wei, Cheng Deng, Muli Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00911](https://doi.org/10.1109/CVPR52688.2022.00911)

**Abstract**:

Compositional Zero-Shot Learning (CZSL) aims to recognize unseen compositions formed from seen state and object during training. Since the same state may be various in the visual appearance while entangled with different objects, CZSL is still a challenging task. Some methods recognize state and object with two trained classifiers, ignoring the impact of the interaction between object and state; the other methods try to learn the joint representation of the state-object compositions, leading to the domain gap between seen and unseen composition sets. In this paper, we propose a novel Siamese Contrastive Embedding Network (SCEN)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/XDUxyLi/SCEN-master for unseen composition recognition. Considering the entanglement between state and object, we embed the visual feature into a Siamese Contrastive Space to capture prototypes of them separately, alleviating the interaction between state and object. In addition, we design a State Transition Module (STM) to increase the diversity of training compositions, improving the robustness of the recognition model. Extensive experiments indicate that our method significantly outperforms the state-of-the-art approaches on three challenging benchmark datasets, including the recent proposed C-QGA dataset.

----

## [903] KG-SP: Knowledge Guided Simple Primitives for Open World Compositional Zero-Shot Learning

**Authors**: *Shyamgopal Karthik, Massimiliano Mancini, Zeynep Akata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00912](https://doi.org/10.1109/CVPR52688.2022.00912)

**Abstract**:

The goal of open-world compositional zero-shot learning (OW-CZSL) is to recognize compositions of state and objects in images, given only a subset of them during training and no prior on the unseen compositions. In this setting, models operate on a huge output space, containing all possible state-object compositions. While previous works tackle the problem by learning embeddings for the compositions jointly, here we revisit a simple CZSL baseline and predict the primitives, i.e. states and objects, independently. To ensure that the model develops primitive-specific features, we equip the state and object classifiers with separate, non-linear feature extractors. Moreover, we estimate the feasibility of each composition through external knowledge, using this prior to remove unfeasible compositions from the output space. Finally, we propose a new setting, i.e. CZSL under partial supervision (pCZSL), where either only objects or state labels are available during training, and we can use our prior to estimate the missing labels. Our model, Knowledge-Guided Simple Primitives (KG-SP), achieves state of the art in both OW-CZSL and pCZSL, surpassing most recent competitors even when coupled with semi-supervised learning techniques. Code available at: https://github.com/ExplainableML/KG-SP.

----

## [904] Non-generative Generalized Zero-shot Learning via Task-correlated Disentanglement and Controllable Samples Synthesis

**Authors**: *Yaogong Feng, Xiaowen Huang, Pengbo Yang, Jian Yu, Jitao Sang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00913](https://doi.org/10.1109/CVPR52688.2022.00913)

**Abstract**:

Synthesizing pseudo samples is currently the most effective way to solve the Generalized Zero Shot Learning (GZSL) problem. Most models achieve competitive performance but still suffer from two problems: (1) Feature confounding, the overall representations confound task-correlated and task-independent features, and existing models disentangle them in a generative way, but they are unreasonable to synthesize reliable pseudo samples with limited samples; (2) Distribution uncertainty, that massive data is needed when existing models synthesize samples from the uncertain distribution, which causes poor performance in limited samples of seen classes. In this paper, we propose a non-generative model to address these problems correspondingly in two modules: (1) Task-correlated feature disentanglement, to exclude the task-correlated features from task-independent ones by adversarial learning of domain adaption towards reasonable synthesis; (2) Controllable pseudo sample synthesis, to synthesize edge-pseudo and center-pseudo samples with certain characteristics towards more diversity generated and intuitive transfer. In addation, to describe the new scene that is the limit seen class samples in the training process, we further formulate a new ZSL task named the ‘Few-shot Seen class and Zero-shot Unseen class learning’ (FSZU). Extensive experiments on four benchmarks verify that the proposed method is competitive in the GZSL and the FSZU tasks.

----

## [905] WALT: Watch And Learn 2D amodal representation from Time-lapse imagery

**Authors**: *N. Dinesh Reddy, Robert Tamburo, Srinivasa G. Narasimhan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00914](https://doi.org/10.1109/CVPR52688.2022.00914)

**Abstract**:

Current methods for object detection, segmentation, and tracking fail in the presence of severe occlusions in busy urban environments. Labeled real data of occlusions is scarce (even in large datasets) and synthetic data leaves a domain gap, making it hard to explicitly model and learn occlusions. In this work, we present the best of both the real and synthetic worlds for automatic occlusion supervision using a large readily available source of data: time-lapse imagery from stationary webcams observing street intersections over weeks, months, or even years. We introduce a new dataset, Watch and Learn Time-lapse (WALT), consisting of 12 (4K and 1080p) cameras capturing urban environments over a year. We exploit this real data in a novel way to automatically mine a large set of unoccluded objects and then composite them in the same views to generate occlusions. This longitudinal self-supervision is strong enough for an amodal network to learn object-occluder-occluded layer representations. We show how to speed up the discovery of unoccluded objects and relate the confidence in this discovery to the rate and accuracy of training occluded objects. After watching and automatically learning for several days, this approach shows significant performance improvement in detecting and segmenting occluded people and vehicles, over human-supervised amodal approaches.

----

## [906] Omni-DETR: Omni-Supervised Object Detection with Transformers

**Authors**: *Pei Wang, Zhaowei Cai, Hao Yang, Gurumurthy Swaminathan, Nuno Vasconcelos, Bernt Schiele, Stefano Soatto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00915](https://doi.org/10.1109/CVPR52688.2022.00915)

**Abstract**:

We consider the problem of omni-supervised object detection, which can use unlabeled, fully labeled and weakly labeled annotations, such as image tags, counts, points, etc., for object detection. This is enabled by a unified architecture, Omni-DETR, based on the recent progress on student-teacher framework and end-to-end transformer based object detection. Under this unified architecture, different types of weak labels can be leveraged to generate accurate pseudo labels, by a bipartite matching based filtering mechanism, for the model to learn. In the experiments, Omni-DETR has achieved state-of-the-art results on multiple datasets and settings. And we have found that weak annotations can help to improve detection performance and a mixture of them can achieve a better trade-off between annotation cost and accuracy than the standard complete annotation. These findings could encourage larger object detection datasets with mixture annotations. The code is available at https://github.com/amazon-research/omni-detr.

----

## [907] DESTR: Object Detection with Split Transformer

**Authors**: *Liqiang He, Sinisa Todorovic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00916](https://doi.org/10.1109/CVPR52688.2022.00916)

**Abstract**:

Self- and cross-attention in Transformers provide for high model capacity, making them viable models for object detection. However, Transformers still lag in performance behind CNN-based detectors. This is, we believe, because: (a) Cross-attention is used for both classification and bounding-box regression tasks; (b) Transformer's decoder poorly initializes content queries; and (c) Self-attention poorly accounts for certain prior knowledge which could help improve inductive bias. These limitations are addressed with the corresponding three contributions. First, we propose a new Detection Split Transformer (DESTR) that separates estimation of cross-attention into two independent branches — one tailored for classification and the other for box regression. Second, we use a mini-detector to initialize the content queries in the decoder with classification and regression embeddings of the respective heads in the mini-detector. Third, we augment self-attention in the decoder to additionally account for pairs of adjacent object queries. Our experiments on the MS-COCO dataset show that DESTR outperforms DETR and its successors.

----

## [908] A Dual Weighting Label Assignment Scheme for Object Detection

**Authors**: *Shuai Li, Chenhang He, Ruihuang Li, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00917](https://doi.org/10.1109/CVPR52688.2022.00917)

**Abstract**:

Label assignment (LA), which aims to assign each training sample a positive (pos) and a negative (neg) loss weight, plays an important role in object detection. Existing LA methods mostly focus on the design of pos weighting function, while the neg weight is directly derived from the pos weight. Such a mechanism limits the learning capacity of detectors. In this paper, we explore a new weighting paradigm, termed dual weighting (DW), to specify pos and neg weights separately. We first identify the key influential factors of pos/neg weights by analyzing the evaluation metrics in object detection, and then design the pos and neg weighting functions based on them. Specifically, the pos weight of a sample is determined by the consistency degree between its classification and localization scores, while the neg weight is decomposed into two terms: the probability that it is a neg sample and its importance conditioned on being a neg sample. Such a weighting strategy offers greater flexibility to distinguish between important and less important samples, resulting in a more effective object detector. Equipped with the proposed DW method, a single FCOS-ResNet-50 detector can reach 41.5% mAP on COCO under 1× schedule, outperforming other existing LA methods. It consistently improves the baselines on COCO by a large margin under various backbones without bells and whistles. Code is available at https://github.com/strongwolf/DW.

----

## [909] Entropy-based Active Learning for Object Detection with Progressive Diversity Constraint

**Authors**: *Jiaxi Wu, Jiaxin Chen, Di Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00918](https://doi.org/10.1109/CVPR52688.2022.00918)

**Abstract**:

Active learning is a promising alternative to alleviate the issue of high annotation cost in the computer vision tasks by consciously selecting more informative samples to label. Active learning for object detection is more challenging and existing efforts on it are relatively rare. In this paper, we propose a novel hybrid approach to address this problem, where the instance-level uncertainty and diversity are jointly considered in a bottom-up manner. To balance the computational complexity, the proposed approach is designed as a two-stage procedure. At the first stage, an Entropy-based Non-Maximum Suppression (ENMS) is presented to estimate the uncertainty of every image, which performs NMS according to the entropy in the feature space to remove predictions with redundant information gains. At the second stage, a diverse prototype (DivProto) strategy is explored to ensure the diversity across images by progressively converting it into the intra-class and inter-class diversities of the entropy-based class-specific prototypes. Extensive experiments are conducted on MS COCO and Pascal VOC, and the proposed approach achieves state of the art results and significantly outperforms the other counter-parts, highlighting its superiority.

----

## [910] Localization Distillation for Dense Object Detection

**Authors**: *Zhaohui Zheng, Rongguang Ye, Ping Wang, Dongwei Ren, Wangmeng Zuo, Qibin Hou, Ming-Ming Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00919](https://doi.org/10.1109/CVPR52688.2022.00919)

**Abstract**:

Knowledge distillation (KD) has witnessed its powerful capability in learning compact models in object detection. Previous KD methods for object detection mostly focus on imitating deep features within the imitation regions instead of mimicking classification logit due to its inefficiency in distilling localization information and trivial improvement. In this paper, by reformulating the knowledge distillation process on localization, we present a novel localization distillation (LD) method which can efficiently transfer the localization knowledge from the teacher to the student. Moreover, we also heuristically introduce the concept of valuable localization region that can aid to selectively distill the semantic and localization knowledge for a certain region. Combining these two new components, for the first time, we show that logit mimicking can outperform feature imitation and localization knowledge distillation is more important and efficient than semantic knowledge for distilling object detectors. Our distillation scheme is simple as well as effective and can be easily applied to different dense object detectors. Experiments show that our LD can boost the AP score of GFocal-ResNet-50 with a single-scale 1 x training schedule from 40.1 to 42.1 on the COCO benchmark without any sacrifice on the inference speed. Our source code and pretrained models are publicly available at https://github.com/HikariTJU/LD.

----

## [911] Group R-CNN for Weakly Semi-supervised Object Detection with Points

**Authors**: *Shilong Zhang, Zhuoran Yu, Liyang Liu, Xinjiang Wang, Aojun Zhou, Kai Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00920](https://doi.org/10.1109/CVPR52688.2022.00920)

**Abstract**:

We study the problem of weakly semi-supervised object detection with points (WSSOD-P), where the training data is combined by a small set of fully annotated images with bounding boxes and a large set of weakly-labeled images with only a single point annotated for each instance. The core of this task is to train a point-to-box regressor on well-labeled images that can be used to predict credible bounding boxes for each point annotation. We challenge the prior belief that existing CNN-based detectors are not compatible with this task. Based on the classic R-CNN architecture, we propose an effective point-to-box regressor: Group R-CNN. Group R-CNN first uses instance-level proposal grouping to generate a group of proposals for each point annotation and thus can obtain a high recall rate. To better distinguish different instances and improve precision, we propose instance-level proposal assignment to replace the vanilla assignment strategy adopted in original R-CNN methods. As naive instance-level assignment brings converging difficulty, we propose instance aware representation learning which consists of instance aware feature enhancement and instance-aware parameter generation to overcome this issue. Comprehensive experiments on the MS-COCO benchmark demonstrate the effectiveness of our method. Specifically, Group R-CNN significantly outperforms the prior method Point DETR by 3.9 mAP with 5% well-labeled images, which is the most challenging scenario. The source code can be found at https://github.com/jshilong/GroupRCNN.

----

## [912] Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation

**Authors**: *Tao Feng, Mang Wang, Hangjie Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00921](https://doi.org/10.1109/CVPR52688.2022.00921)

**Abstract**:

Traditional object detectors are ill-equipped for incremental learning. However, fine-tuning directly on a well-trained detection model with only new data will lead to catastrophic forgetting. Knowledge distillation is a flexible way to mitigate catastrophic forgetting. In Incremental Object Detection (IOD), previous work mainly focuses on distilling for the combination of features and responses. However, they under-explore the information that contains in responses. In this paper, we propose a response-based incremental distillation method, dubbed Elastic Response Distillation (ERD), which focuses on elastically learning responses from the classification head and the regression head. Firstly, our method transfers category knowledge while equipping student detector with the ability to retain localization information during incremental learning. In addition, we further evaluate the quality of all locations and provide valuable responses by the Elastic Response Selection (ERS) strategy. Finally, we elucidate that the knowledge from different responses should be assigned with different importance during incremental distillation. Extensive experiments conducted on MS COCO demonstrate our method achieves state-of-the-art result, which substantially narrows the performance gap towards full training. Code is available at https://github.com/Hi-FT/ERD.

----

## [913] CREAM: Weakly Supervised Object Localization via Class RE-Activation Mapping

**Authors**: *Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Rui-Wei Zhao, Tao Zhang, Xuequan Lu, Shang Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00922](https://doi.org/10.1109/CVPR52688.2022.00922)

**Abstract**:

Weakly Supervised Object Localization (WSOL) aims to localize objects with image-level supervision. Existing works mainly rely on Class Activation Mapping (CAM) de-rived from a classification model. However, CAM-based methods usually focus on the most discriminative parts of an object (i.e., incomplete localization problem). In this paper, we empirically prove that this problem is associated with the mixup of the activation values between less discrimi-native foreground regions and the background. To address it, we propose Class RE-Activation Mapping (CREAM), a novel clustering-based approach to boost the activation values of the integral object regions. To this end, we in-troduce class-specific foreground and background context embeddings as cluster centroids. A CAM-guided momen-tum preservation strategy is developed to learn the context embeddings during training. At the inference stage, the re-activation mapping is formulated as a parameter es-timation problem under Gaussian Mixture Model, which can be solved by deriving an unsupervised Expectation- Maximization based soft-clustering algorithm. By simply integrating CREAM into various WSOL approaches, our method significantly improves their performance. CREAM achieves the state-of-the-art performance on CUB, ILSVRC and OpenImages benchmark datasets. Code will be avail-able at https://github.com/lazzcharles/CREAM.

----

## [914] One Loss for Quantization: Deep Hashing with Discrete Wasserstein Distributional Matching

**Authors**: *Khoa D. Doan, Peng Yang, Ping Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00923](https://doi.org/10.1109/CVPR52688.2022.00923)

**Abstract**:

Image hashing is a principled approximate nearest neighbor approach to find similar items to a query in a large collection of images. Hashing aims to learn a binary-output function that maps an image to a binary vector. For optimal retrieval performance, producing balanced hash codes with low-quantization error to bridge the gap between the learning stage's continuous relaxation and the inference stage's discrete quantization is important. However, in the existing deep supervised hashing methods, coding balance and low-quantization error are difficult to achieve and involve several losses. We argue that this is because the existing quantization approaches in these methods are heuristically constructed and not effective to achieve these objectives. This paper considers an alternative approach to learning the quantization constraints. The task of learning balanced codes with low quantization error is re-formulated as matching the learned distribution of the continuous codes to a pre-defined discrete, uniform distribution. This is equivalent to minimizing the distance between two distributions. We then propose a computationally efficient distributional distance by leveraging the discrete property of the hash functions. This distributional distance is a valid distance and enjoys lower time and sample complexities. The proposed single-loss quantization objective can be integrated into any existing supervised hashing method to improve code balance and quantization error. Experiments confirm that the proposed approach substantially improves the performance of several representative hashing methods.

----

## [915] PSTR: End-to-End One-Step Person Search With Transformers

**Authors**: *Jiale Cao, Yanwei Pang, Rao Muhammad Anwer, Hisham Cholakkal, Jin Xie, Mubarak Shah, Fahad Shahbaz Khan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00924](https://doi.org/10.1109/CVPR52688.2022.00924)

**Abstract**:

We propose a novel one-step transformer-based person search framework, PSTR, that jointly performs person detection and re-identification (re-id) in a single architecture. PSTR comprises a person search-specialized (PSS) module that contains a detection encoder-decoder for person detection along with a discriminative re-id decoder for person re-id. The discriminative re-id decoder utilizes a multi-level supervision scheme with a shared decoder for discriminative re-id feature learning and also comprises a part attention block to encode relationship between different parts of a person. We further introduce a simple multi-scale scheme to support re-id across person instances at different scales. PSTR jointly achieves the diverse objectives of object-level recognition (detection) and instance-level matching (re-id). To the best of our knowledge, we are the first to propose an end-to-end one-step transformer-based person search framework. Experiments are performed on two popular benchmarks: CUHK-SYSU and PRW. Our extensive ablations reveal the merits of the proposed contributions. Further, the proposed PSTR sets a new state-of-the-art on both benchmarks. On the challenging PRW benchmark, PSTR achieves a mean average precision (mAP) score of 56.5%. The source code is available at https://github.com/JialeCao001/PSTR.

----

## [916] Protecting Celebrities from DeepFake with Identity Consistency Transformer

**Authors**: *Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Ting Zhang, Weiming Zhang, Nenghai Yu, Dong Chen, Fang Wen, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00925](https://doi.org/10.1109/CVPR52688.2022.00925)

**Abstract**:

In this work we propose Identity Consistency Transformer, a novel face forgery detection method that focuses on high-level semantics, specifically identity information, and detecting a suspect face by finding identity inconsistency in inner and outer face regions. The Identity Consistency Transformer incorporates a consistency loss for identity consistency determination. We show that Identity Consistency Transformer exhibits superior generalization ability not only across different datasets but also across various types of image degradation forms found in real-world applications including deepfake videos. The Identity Consistency Transformer can be easily enhanced with additional identity information when such information is available, and for this reason it is especially well-suited for detecting face forgeries involving celebrities.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code will be released at https://github.com/LightDXY/ICT_DeepFake

----

## [917] MDAN: Multi-level Dependent Attention Network for Visual Emotion Analysis

**Authors**: *Liwen Xu, Zhengtao Wang, Bin Wu, Simon Lui*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00926](https://doi.org/10.1109/CVPR52688.2022.00926)

**Abstract**:

Visual Emotion Analysis (VEA) is attracting increasing attention. One of the biggest challenges of VEA is to bridge the affective gap between visual clues in a picture and the emotion expressed by the picture. As the granularity of emotions increases, the affective gap increases as well. Existing deep approaches try to bridge the gap by directly learning discrimination among emotions globally in one shot. They ignore the hierarchical relationship among emotions at different affective levels, and the variation in the affective level of emotions to be classified. In this paper, we present the multi-level dependent attention network (MDAN) with two branches to leverage the emotion hierarchy and the correlation between different affective levels and semantic levels. The bottom-up branch directly learns emotions at the highest affective level and largely prevents hierarchy violation by explicitly following the emotion hierarchy while predicting emotions at lower affective levels. In contrast, the top-down branch aims to disentangle the affective gap by one-to-one mapping between semantic levels and affective levels, namely, Affective Semantic Mapping. A local classifier is appended at each semantic level to learn discrimination among emotions at the corresponding affective level. Then, we integrate global learning and local learning into a unified deep framework and optimize it simultaneously. Moreover, to properly model channel dependencies and spatial attention while disentangling the affective gap, we carefully designed two attention modules: the Multi-head Cross Channel Attention module and the Level-dependent Class Activation Map module. Finally, the proposed deep framework obtains new state-of-the-art performance on six VEA benchmarks, where it outperforms existing state-of-the-art methods by a large margin, e.g., +3.85% on the WEBEmo dataset at 25 classes classification accuracy.

----

## [918] Contextual Similarity Distillation for Asymmetric Image Retrieval

**Authors**: *Hui Wu, Min Wang, Wengang Zhou, Houqiang Li, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00927](https://doi.org/10.1109/CVPR52688.2022.00927)

**Abstract**:

Asymmetric image retrieval, which typically uses small model for query side and large model for database server, is an effective solution for resource-constrained scenarios. However, existing approaches either fail to achieve feature coherence or make strong assumptions, e.g., requiring labeled datasets or classifiers from large model, etc., which limits their practical application. To this end, we propose a flexible contextual similarity distillation framework to enhance the small query model and keep its output feature compatible with that of the large gallery model, which is crucial with asymmetric retrieval. In our approach, we learn the small model with a new contextual similarity consistency constraint without any data label. During the small model learning, it preserves the contextual similarity among each training image and its neighbors with the features extracted by the large model. Note that this simple constraint is consistent with simultaneous first-order feature vector preserving and second-order ranking list preserving. Extensive experiments show that the proposed method outperforms the state-of-the-art methods on the Revisited Oxford and Paris datasets.

----

## [919] Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning

**Authors**: *Li Yang, Yan Xu, Chunfeng Yuan, Wei Liu, Bing Li, Weiming Hu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00928](https://doi.org/10.1109/CVPR52688.2022.00928)

**Abstract**:

Visual grounding is a task to locate the target indicated by a natural language expression. Existing methods extend the generic object detection framework to this problem. They base the visual grounding on the features from pre-generated proposals or anchors, and fuse these features with the text embeddings to locate the target mentioned by the text. However, modeling the visual features from these predefined locations may fail to fully exploit the visual context and attribute information in the text query, which limits their performance. In this paper, we propose a transformer-based framework for accurate visual grounding by establishing text-conditioned discriminative features and performing multi-stage cross-modal reasoning. Specifically, we develop a visual-linguistic verification module to focus the visual features on regions relevant to the textual descriptions while suppressing the unrelated areas. A language-guided feature encoder is also devised to aggregate the visual contexts of the target object to improve the object's distinctiveness. To retrieve the target from the encoded visual features, we further propose a multi-stage cross-modal decoder to iteratively speculate on the correlations between the image and text for accurate target localization. Extensive experiments on five widely used datasets validate the efficacy of our proposed components and demonstrate state-of-the-art performance.

----

## [920] MPC: Multi-view Probabilistic Clustering

**Authors**: *Junjie Liu, Junlong Liu, Shaotian Yan, Rongxin Jiang, Xiang Tian, Boxuan Gu, Yaowu Chen, Chen Shen, Jianqiang Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00929](https://doi.org/10.1109/CVPR52688.2022.00929)

**Abstract**:

Despite the promising progress having been made, the two challenges of multi-view clustering (MVC) are still waiting for better solutions: i) Most existing methods are either not qualified or require additional steps for incomplete multi-view clustering and ii) noise or outliers might significantly degrade the overall clustering performance. In this paper, we propose a novel unified framework for incomplete and complete MVC named multi-view probabilistic clustering (MPC). MPC equivalently transforms multi-view pairwise posterior matching probability into composition of each view's individual distribution, which tolerates data missing and might extend to any number of views. Then graph-context-aware refinement with path propagation and co-neighbor propagation is used to refine pairwise probability, which alleviates the impact of noise and outliers. Finally, MPC also equivalently transforms probabilistic clustering's objective to avoid complete pairwise computation and adjusts clustering assignments by maximizing joint probability iteratively. Extensive experiments on multiple benchmarks for incomplete and complete MVC show that MPC significantly outperforms previous state-of-the-art methods in both effectiveness and efficiency.

----

## [921] Text Spotting Transformers

**Authors**: *Xiang Zhang, Yongwen Su, Subarna Tripathi, Zhuowen Tu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00930](https://doi.org/10.1109/CVPR52688.2022.00930)

**Abstract**:

In this paper, we present TExt Spotting TRansformers (TESTR), a generic end-to-end text spotting framework using Transformers for text detection and recognition in the wild. TESTR builds upon a single encoder and dual decoders for the joint text-box control point regression and character recognition. Other than most existing literature, our method is free from Region-of-Interest operations and heuristics-driven post-processing procedures; TESTR is particularly effective when dealing with curved text-boxes where special cares are needed for the adaptation of the tra-ditional bounding-box representations. We show our canonical representation of control points suitable for text in-stances in both Bezier curve and polygon annotations. In addition, we design a bounding-box guided polygon detection (box-to-polygon) process. Experiments on curved and arbitrarily shaped datasets demonstrate state-of-the-art performances of the proposed TESTR algorithm.

----

## [922] Represent, Compare, and Learn: A Similarity-Aware Framework for Class-Agnostic Counting

**Authors**: *Min Shi, Hao Lu, Chen Feng, Chengxin Liu, Zhiguo Cao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00931](https://doi.org/10.1109/CVPR52688.2022.00931)

**Abstract**:

Class-agnostic counting (CAC) aims to count all instances in a query image given few exemplars. A standard pipeline is to extract visual features from exemplars and match them with query images to infer object counts. Two essential components in this pipeline are feature representation and similarity metric. Existing methods either adopt a pretrained network to represent features or learn a new one, while applying a naive similarity metric with fixed inner product. We find this paradigm leads to noisy similarity matching and hence harms counting performance. In this work, we propose a similarity-aware CAC framework that jointly learns representation and similarity metric. We first instantiate our framework with a naive baseline called Bilinear Matching Network (BMNet), whose key component is a learnable bilinear similarity metric. To further embody the core of our framework, we extend BMNet to BMNet+ that models similarity from three aspects: 1) representing the instances via their self-similarity to enhance feature robustness against intra-class variations; 2) comparing the similarity dynamically to focus on the key patterns of each exemplar; 3) learning from a supervision signal to impose explicit constraints on matching results. Extensive experiments on a recent CAC dataset FSC147 show that our models significantly outperform state-of-the-art CAC approaches. In addition, we also validate the cross-dataset generality of BMNet and BMNet+ on a car counting dataset CARPK. Code is at tiny.one/BMNet

----

## [923] Reflection and Rotation Symmetry Detection via Equivariant Learning

**Authors**: *Ahyun Seo, Byungjin Kim, Suha Kwak, Minsu Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00932](https://doi.org/10.1109/CVPR52688.2022.00932)

**Abstract**:

The inherent challenge of detecting symmetries stems from arbitrary orientations of symmetry patterns; a reflection symmetry mirrors itself against an axis with a specific orientation while a rotation symmetry matches its rotated copy with a specific orientation. Discovering such symmetry patterns from an image thus benefits from an equivariant feature representation, which varies consistently with reflection and rotation of the image. In this work, we introduce a group-equivariant convolutional network for symmetry detection, dubbed EquiSym, which leverages equivariant feature maps with respect to a dihedral group of reflection and rotation. The proposed network is built end-to-end with dihedrally-equivariant layers and trained to output a spatial map for reflection axes or rotation centers. We also present a new dataset, DENse and DIverse symmetry (DENDI), which mitigates limitations of existing benchmarks for reflection and rotation symmetry detection. Experiments show that our method achieves the state of the arts in symmetry detection on LDRS and DENDI datasets.

----

## [924] Learning to Imagine: Diversify Memory for Incremental Learning using Unlabeled Data

**Authors**: *Yu-Ming Tang, Yi-Xing Peng, Wei-Shi Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00933](https://doi.org/10.1109/CVPR52688.2022.00933)

**Abstract**:

Deep neural network (DNN) suffers from catastrophic forgetting when learning incrementally, which greatly limits its applications. Although maintaining a handful of samples (called “exemplars”) of each task could alleviate forgetting to some extent, existing methods are still limited by the small number of exemplars since these exemplars are too few to carry enough task-specific knowledge, and therefore the forgetting remains. To overcome this problem, we propose to “imagine” diverse counterparts of given exemplars referring to the abundant semantic-irrelevant information from unlabeled data. Specifically, we develop a learnable feature generator to diversify exemplars by adaptively generating diverse counterparts of exemplars based on semantic information from exemplars and semantically-irrelevant information from unlabeled data. We introduce semantic contrastive learning to enforce the generated samples to be semantic consistent with exemplars and perform semantic-decoupling contrastive learning to encourage diversity of generated samples. The diverse generated samples could effectively prevent DNN from forgetting when learning new tasks. Our method does not bring any extra inference cost and outperforms state-of-the-art methods on two benchmarks CIFAR-100 and ImageNet-Subset by a clear margin.

----

## [925] A Simple Episodic Linear Probe Improves Visual Recognition in the Wild

**Authors**: *Yuanzhi Liang, Linchao Zhu, Xiaohan Wang, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00934](https://doi.org/10.1109/CVPR52688.2022.00934)

**Abstract**:

Understanding network generalization and feature discrimination is an open research problem in visual recognition. Many studies have been conducted to assess the quality of feature representations. One of the simple strategies is to utilize a linear probing classifier to quantitatively evaluate the class accuracy under the obtained features. The typical linear probe is only applied as a proxy at the inference time, but its efficacy in measuring features' suitability for linear classification is largely neglected in training. In this paper, we propose an episodic linear probing (ELP) classifier to reflect the generalization of visual rep-resentations in an online manner. ELP is trained with detached features from the network and re-initialized episodically. It demonstrates the discriminability of the visual representations in training. Then, an ELP-suitable Regularization term (ELP-SR) is introduced to reflect the distances of probability distributions between the ELP classifier and the main classifier. ELP-SR leverages are-scaling factor to regularize each sample in training, which modulates the loss function adaptively and encourages the features to be discriminative and generalized. We observe significant improvements in three real-world visual recognition tasks: fine-grained visual classification, long-tailed visual recognition, and generic object recognition. The performance gains show the effectiveness of our method in im-proving network generalization and feature discrimination.

----

## [926] Cross Domain Object Detection by Target-Perceived Dual Branch Distillation

**Authors**: *Mengzhe He, Yali Wang, Jiaxi Wu, Yiru Wang, Hanqing Li, Bo Li, Weihao Gan, Wei Wu, Yu Qiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00935](https://doi.org/10.1109/CVPR52688.2022.00935)

**Abstract**:

Cross domain object detection is a realistic and challenging task in the wild. It suffers from performance degradation due to large shift of data distributions and lack of instance-level annotations in the target domain. Existing approaches mainly focus on either of these two difficulties, even though they are closely coupled in cross domain object detection. To solve this problem, we propose a novel Target-perceived Dual-branch Distillation (TDD) framework. By integrating detection branches of both source and target domains in a unified teacher-student learning scheme, it can reduce domain shift and generate reliable supervision effectively. In particular, we first introduce a distinct Target Proposal Perceiver between two domains. It can adaptively enhance source detector to perceive objects in a target image, by leveraging target proposal contexts from iterative cross-attention. Afterwards, we design a concise Dual Branch Self Distillation strategy for model training, which can progressively integrate complementary object knowledge from different domains via self-distillation in two branches. Finally, we conduct extensive experiments on a number of widely-used scenarios in cross domain object detection. The results show that our TDD significantly outperforms the state-of-the-art methods on all the benchmarks. The codes and models will be released afterwards.

----

## [927] Multi-Granularity Alignment Domain Adaptation for Object Detection

**Authors**: *Wenzhang Zhou, Dawei Du, Libo Zhang, Tiejian Luo, Yanjun Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00936](https://doi.org/10.1109/CVPR52688.2022.00936)

**Abstract**:

Domain adaptive object detection is challenging due to distinctive data distribution between source domain and target domain. In this paper, we propose a unified multi-granularity alignment based object detection framework towards domain-invariant feature learning. To this end, we encode the dependencies across different granularity perspectives including pixel-, instance-, and category-levels simultaneously to align two domains. Based on pixel-level feature maps from the backbone network, we first develop the omniscale gated fusion module to aggregate discriminative representations of instances by scale-aware convolutions, leading to robust multi-scale object detection. Meanwhile, the multi-granularity discriminators are proposed to identify which domain different granularities of samples (i.e., pixels, instances, and categories) come from. Notably, we leverage not only the instance discriminability in different categories but also the category consistency between two domains. Extensive experiments are carried out on multiple domain adaptation scenarios, demonstrating the effectiveness of our framework over state-of-the-art algorithms on top of anchor-free FCOS and anchor-based Faster R-CNN detectors with different backbones.

----

## [928] Expanding Low-Density Latent Regions for Open-Set Object Detection

**Authors**: *Jiaming Han, Yuqiang Ren, Jian Ding, Xingjia Pan, Ke Yan, Gui-Song Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00937](https://doi.org/10.1109/CVPR52688.2022.00937)

**Abstract**:

Modern object detectors have achieved impressive progress under the close-set setup. However, open-set object detection (OSOD) remains challenging since objects of unknown categories are often misclassified to existing known classes. In this work, we propose to identify unknown objects by separating high/low-density regions in the latent space, based on the consensus that unknown objects are usually distributed in low-density latent regions. As traditional threshold-based methods only maintain limited low-density regions, which cannot cover all unknown objects, we present a novel Openset Detector (OpenDet) with expanded low-density regions. To this aim, we equip Open-Det with two learners, Contrastive Feature Learner (CFL) and Unknown Probability Learner (UPL). CFL performs instance-level contrastive learning to encourage compact features of known classes, leaving more low-density regions for unknown classes; UPL optimizes unknown probability based on the uncertainty of predictions, which further divides more low-density regions around the cluster of known classes. Thus, unknown objects in low-density regions can be easily identified with the learned unknown probability. Extensive experiments demonstrate that our method can significantly improve the OSOD performance, e.g., OpenDet reduces the Absolute Open-Set Errors by 25%-35% on six OSOD benchmarks. Code is available at: https://github.com/csuhan/opendet2.

----

## [929] Class-Incremental Learning with Strong Pre-trained Models

**Authors**: *Tz-Ying Wu, Gurumurthy Swaminathan, Zhizhong Li, Avinash Ravichandran, Nuno Vasconcelos, Rahul Bhotika, Stefano Soatto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00938](https://doi.org/10.1109/CVPR52688.2022.00938)

**Abstract**:

Class-incremental learning (CIL) has been widely stud-ied under the setting of starting from a small number of classes (base classes). Instead, we explore an understud-ied real-world setting of CIL that starts with a strong model pre-trained on a large number of base classes. We hypoth-esize that a strong base model can provide a good repre-sentation for novel classes and incremental learning can be done with small adaptations. We propose a 2-stage training scheme, i) feature augmentation - cloning part of the backbone and fine-tuning it on the novel data, and ii) fusion - combining the base and novel classifiers into a unified classifier. Experiments show that the proposed method sig-nificantly outperforms state-of-the-art CIL methods on the large-scale ImageNet dataset (e.g. + 10% overall accuracy than the best). We also propose and analyze understudied practical CIL scenarios, such as base-novel overlap with distribution shift. Our proposed method is robust and gen-eralizes to all analyzed CIL settings.

----

## [930] ProposalCLIP: Unsupervised Open-Category Object Proposal Generation via Exploiting CLIP Cues

**Authors**: *Hengcan Shi, Munawar Hayat, Yicheng Wu, Jianfei Cai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00939](https://doi.org/10.1109/CVPR52688.2022.00939)

**Abstract**:

Object proposal generation is an important and fundamental task in computer vision. In this paper, we propose ProposalCLIP, a method towards unsupervised open-category object proposal generation. Unlike previous works which require a large number of bounding box annotations and/or can only generate proposals for limited object categories, our ProposalCLIP is able to predict proposals for a large variety of object categories without annotations, by exploiting CLIP (contrastive language-image pre-training) cues. Firstly, we analyze CLIP for unsupervised open-category proposal generation and design an objectness score based on our empirical analysis on proposal selection. Secondly, a graph-based merging module is proposed to solve the limitations of CLIP cues and merge fragmented proposals. Finally, we present a proposal regression module that extracts pseudo labels based on CLIP cues and trains a lightweight network to further refine proposals. Extensive experiments on PASCAL VOC, COCO and Visual Genome datasets show that our ProposalCLIP can better generate proposals than previous state-of-the-art methods. Our ProposalCLIP also shows benefits for downstream tasks, such as unsupervised object detection.

----

## [931] Self-Supervised Models are Continual Learners

**Authors**: *Enrico Fini, Victor G. Turrisi da Costa, Xavier Alameda-Pineda, Elisa Ricci, Karteek Alahari, Julien Mairal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00940](https://doi.org/10.1109/CVPR52688.2022.00940)

**Abstract**:

Self-supervised models have been shown to produce comparable or better visual representations than their su-pervised counterparts when trained offline on unlabeled data at scale. However, their efficacy is catastrophically reduced in a Continual Learning (CL) scenario where data is presented to the model sequentially. In this paper, we show that self-supervised loss functions can be seamlessly converted into distillation mechanisms for CL by adding a predictor network that maps the current state of the repre-sentations to their past state. This enables us to devise a framework for Continual self-supervised visual representation Learning that (i) significantly improves the quality of the learned representations, (ii) is compatible with several state-of-the-art self-supervised objectives, and (iii) needs little to no hyperparameter tuning. We demonstrate the ef-fectiveness of our approach empirically by training six pop-ular self-supervised models in various CL settings. Code: github.com/DonkeyShot21/cassle.

----

## [932] The Two Dimensions of Worst-case Training and Their Integrated Effect for Out-of-domain Generalization

**Authors**: *Zeyi Huang, Haohan Wang, Dong Huang, Yong Jae Lee, Eric P. Xing*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00941](https://doi.org/10.1109/CVPR52688.2022.00941)

**Abstract**:

Training with an emphasis on “hard-to-learn” components of the data has been proven as an effective method to improve the generalization of machine learning models, especially in the settings where robustness (e.g., generalization across distributions) is valued. Existing literature discussing this “hard-to-learn” concept are mainly expanded either along the dimension of the samples or the dimension of the features. In this paper, we aim to introduce a simple view merging these two dimensions, leading to a new, simple yet effective, heuristic to train machine learning models by emphasizing the worst-cases on both the sample and the feature dimensions. We name our method W2D following the concept of “Worst-case along Two Dimensions”. We validate the idea and demonstrate its empirical strength over standard benchmarks.

----

## [933] Beyond Supervised vs Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning

**Authors**: *Matthew Gwilliam, Abhinav Shrivastava*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00942](https://doi.org/10.1109/CVPR52688.2022.00942)

**Abstract**:

By leveraging contrastive learning, clustering, and other pretext tasks, unsupervised methods for learning image representations have reached impressive results on standard benchmarks. The result has been a crowded field - many methods with substantially different implementations yield results that seem nearly identical on popular benchmarks, such as linear evaluation on ImageNet. However, a single result does not tell the whole story. In this paper, we compare methods using performance-based benchmarks such as linear evaluation, nearest neighbor classification, and clustering for several different datasets, demonstrating the lack of a clear front-runner within the current state-of-the-art. In contrast to prior work that performs only supervised vs. unsupervised comparison, we compare several different unsupervised methods against each other. To enrich this comparison, we analyze embeddings with measurements such as uniformity, tolerance, and centered kernel alignment (CKA), and propose two new metrics of our own: nearest neighbor graph similarity and linear prediction overlap. We reveal through our analysis that in isolation, single popular methods should not be treated as though they represent the field as a whole, and that future work ought to consider how to leverage the complimentary nature of these methods. We also leverage CKA to provide a framework to robustly quantify augmentation invariance, and provide a reminder that certain types of invariance will be undesirable for downstream tasks.

----

## [934] SimMIM: a Simple Framework for Masked Image Modeling

**Authors**: *Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, Han Hu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00943](https://doi.org/10.1109/CVPR52688.2022.00943)

**Abstract**:

This paper presents SimMIM, a simple framework for masked image modeling. We have simplified recently proposed relevant approaches, without the need for special designs, such as block-wise masking and tokenization via discrete VAE or clustering. To investigate what makes a masked image modeling task learn good representations, we systematically study the major components in our framework, and find that the simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a powerful pre-text task; 2) predicting RGB values of raw pixels by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied to a larger model with about 650 million parameters, SwinV2-H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to address the data-hungry issue faced by large-scale model training, that a 3B model (Swin V2-G) is successfully trained to achieve state-of-the-art accuracy on four representative vision benchmarks using 40× less labelled data than that in previous practice (JFT-3B). The code is available at https://github.com/microsoft/SimMIM.

----

## [935] Semantic-Aware Auto-Encoders for Self-supervised Representation Learning

**Authors**: *Guangrun Wang, Yansong Tang, Liang Lin, Philip H. S. Torr*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00944](https://doi.org/10.1109/CVPR52688.2022.00944)

**Abstract**:

The resurgence of unsupervised learning can be attributed to the remarkable progress of self-supervised learning, which includes generative 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathcal{G})$</tex>
 and discriminative 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathcal{D})$</tex>
 models. In computer vision, the mainstream self-supervised learning algorithms are 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{D}$</tex>
 models. However, designing a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{D}$</tex>
 model could be over-complicated; also, some studies hinted that a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{D}$</tex>
 model might not be as general and interpretable as a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{G}$</tex>
 model. In this paper, we switch from 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{D}$</tex>
 models to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{G}$</tex>
 models using the classical auto-encoder 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(AE)$</tex>
. Note that a vanilla 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{G}$</tex>
 model was far less efficient than a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{D}$</tex>
 model in self-supervised computer vision tasks, as it wastes model capability on overfitting semantic-agnostic high-frequency details. Inspired by perceptual learning that could use cross-view learning to perceive concepts and semantics
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Following [26], we refer to semantics as visual concepts, e.g., a semantic-ware model indicates the model can perceive visual concepts, and the learned features are efficient in object recognition, detection, etc., we propose a novel 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AE$</tex>
 that could learn semantic-aware representation via cross-view image reconstruction. We use one view of an image as the input and another view of the same image as the reconstruction target. This kind of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AE$</tex>
 has rarely been studied before, and the optimization is very difficult. To enhance learning ability and find a feasible solution, we propose a semantic aligner that uses geometric transformation knowledge to align the hidden code of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AE$</tex>
 to help optimization. These techniques significantly improve the representation learning ability of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AE$</tex>
 and make selfsupervised learning with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{G}$</tex>
 models possible. Extensive experiments on many large-scale benchmarks (e.g., ImageNet, COCO 2017, and SYSU-30k) demonstrate the effectiveness of our methods. Code is available at https://github.com/wanggrun/Semantic-Aware-AE.

----

## [936] UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning

**Authors**: *Nazmul Karim, Mamshad Nayeem Rizve, Nazanin Rahnavard, Ajmal Mian, Mubarak Shah*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00945](https://doi.org/10.1109/CVPR52688.2022.00945)

**Abstract**:

Supervised deep learning methods require a large repository of annotated data; hence, label noise is inevitable. Training with such noisy data negatively impacts the generalization performance of deep neural networks. To combat label noise, recent state-of-the-art methods employ some sort of sample selection mechanism to select a possibly clean subset of data. Next, an off-the-shelf semi-supervised learning method is used for training where rejected samples are treated as unlabeled data. Our comprehensive analysis shows that current selection methods disproportionately select samples from easy (fast learnable) classes while rejecting those from relatively harder ones. This creates class imbalance in the selected clean set and in turn, deteriorates performance under high label noise. In this work, we propose UNICON, a simple yet effective sample selection method which is robust to high label noise. To address the disproportionate selection of easy and hard samples, we introduce a Jensen-Shannon divergence based uniform selection mechanism which does not require any probabilistic modeling and hyperparameter tuning. We complement our selection method with contrastive learning to further combat the memorization of noisy labels. Extensive experimentation on multiple benchmark datasets demonstrates the effectiveness of UNICON; we obtain an 11.4% improvement over the current state-of-the-art on CIFAR100 dataset with a 90% noise rate. Our code is publicly available.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/nazmul-karim170/UNICON-Noisy-Label

----

## [937] Contrastive Conditional Neural Processes

**Authors**: *Zesheng Ye, Lina Yao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00946](https://doi.org/10.1109/CVPR52688.2022.00946)

**Abstract**:

Conditional Neural Processes (CNPs) bridge neural net-works with probabilistic inference to approximate functions of Stochastic Processes under meta-learning settings. Given a batch of non-i.i.d function instantiations, CNPs are jointly optimized for in-instantiation observation prediction and cross-instantiation meta-representation adaptation within a generative reconstruction pipeline. There can be a challenge in tying together such two targets when the distri-bution of function observations scales to high-dimensional and noisy spaces. Instead, noise contrastive estimation might be able to provide more robust representations by learning distributional matching objectives to combat such inherent limitation of generative models. In light of this, we propose to equip CNPs by 1) aligning prediction with en-coded ground-truth observation, and 2) decoupling meta-representation adaptation from generative reconstruction. Specifically, two auxiliary contrastive branches are set up hierarchically, namely in-instantiation temporal contrastive learning (TCL) and cross-instantiation function contrastive learning (FCL), to facilitate local predictive alignment and global function consistency, respectively. We empirically show that TCL captures high-level abstraction of obser-vations, whereas FCL helps identify underlying functions, which in turn provides more efficient representations. Our model outperforms other CNPs variants when evaluating function distribution reconstruction and parameter identifi-cation across 1D, 2D and high-dimensional time-series.

----

## [938] One-bit Active Query with Contrastive Pairs

**Authors**: *Yuhang Zhang, Xiaopeng Zhang, Lingxi Xie, Jie Li, Robert C. Qiu, Hengtong Hu, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00947](https://doi.org/10.1109/CVPR52688.2022.00947)

**Abstract**:

How to achieve better results with fewer labeling costs remains a challenging task. In this paper, we present a new active learning framework, which for the first time incorporates contrastive learning into recently proposed one-bit supervision. Here one-bit supervision denotes a simple Yes or No query about the correctness of the model's prediction, and is more efficient than previous active learning methods requiring assigning accurate labels to the queried samples. We claim that such one-bit information is intrinsically in accordance with the goal of contrastive loss that pulls positive pairs together and pushes negative samples away. Towards this goal, we design an uncertainty metric to actively select samples for query. These samples are then fed into different branches according to the queried results. The Yes query is treated as positive pairs of the queried category for contrastive pulling, while the No query is treated as hard negative pairs for contrastive repelling. Additionally, we design a negative loss that penalizes the negative samples away from the incorrect predicted class, which can be treated as optimizing hard negatives for the corresponding category. Our method, termed as ObCP, produces a more powerful active learning framework, and experiments on several benchmarks demonstrate its superiority.

----

## [939] HCSC: Hierarchical Contrastive Selective Coding

**Authors**: *Yuanfan Guo, Minghao Xu, Jiawen Li, Bingbing Ni, Xuanyu Zhu, Zhenbang Sun, Yi Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00948](https://doi.org/10.1109/CVPR52688.2022.00948)

**Abstract**:

Hierarchical semantic structures naturally exist in an image dataset, in which several semantically relevant image clusters can be further integrated into a larger cluster with coarser-grained semantics. Capturing such structures with image representations can greatly benefit the semantic understanding on various downstream tasks. Existing contrastive representation learning methods lack such an important model capability. In addition, the negative pairs used in these methods are not guaranteed to be semantically distinct, which could further hamper the structural correctness of learned image representations. To tackle these limitations, we propose a novel contrastive learning framework called Hierarchical Contrastive Selective Coding (HCSC). In this framework, a set of hierarchical prototypes are constructed and also dynamically updated to represent the hierarchical semantic structures underlying the data in the latent space. To make image representations better fit such semantic structures, we employ and further improve conventional instance-wise and prototypical contrastive learning via an elaborate pair selection scheme. This scheme seeks to select more diverse positive pairs with similar semantics and more precise negative pairs with truly distinct semantics. On extensive downstream tasks, we verify the state-of-the-art performance of HCSC and also the effectiveness of major model components. We are continually building a comprehensive model zoo (see supplementary material). Our source code and model weights are available at https://github.com/gyfastas/HCSC.

----

## [940] Motion-aware Contrastive Video Representation Learning via Foreground-background Merging

**Authors**: *Shuangrui Ding, Maomao Li, Tianyu Yang, Rui Qian, Haohang Xu, Qingyi Chen, Jue Wang, Hongkai Xiong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00949](https://doi.org/10.1109/CVPR52688.2022.00949)

**Abstract**:

In light of the success of contrastive learning in the image domain, current self-supervised video representation learning methods usually employ contrastive loss to facilitate video representation learning. When naively pulling two augmented views of a video closer, the model however tends to learn the common static background as a shortcut but fails to capture the motion information, a phenomenon dubbed as background bias. Such bias makes the model suffer from weak generalization ability, leading to worse performance on downstream tasks such as action recognition. To alleviate such bias, we propose Foreground-background Merging (FAME) to deliberately compose the moving foreground region of the selected video onto the static background of others. Specifically, without any off-the-shelf detector, we extract the moving fore-ground out of background regions via the frame difference and color statistics, and shuffle the background regions among the videos. By leveraging the semantic consistency between the original clips and the fused ones, the model focuses more on the motion patterns and is debiased from the background shortcut. Extensive experiments demonstrate that FAME can effectively resist background cheating and thus achieve the state-of-the-art performance on downstream tasks across UCF101, HMDB51, and Diving48 datasets. The code and configurations are released at https://github.com/Mark12Ding/FAME.

----

## [941] Hierarchical Self-supervised Representation Learning for Movie Understanding

**Authors**: *Fanyi Xiao, Kaustav Kundu, Joseph Tighe, Davide Modolo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00950](https://doi.org/10.1109/CVPR52688.2022.00950)

**Abstract**:

Most self-supervised video representation learning approaches focus on action recognition. In contrast, in this paper we focus on self-supervised video learning for movie understanding and propose a novel hierarchical self-supervised pretraining strategy that separately pretrains each level of our hierarchical movie understanding model (based on [37]). Specifically, we propose to pretrain the low-level video backbone using a contrastive learning objective, while pretrain the higher-level video contextualizer using an event mask prediction task, which enables the usage of different data sources for pretraining different levels of the hierarchy. We first show that our self-supervised pre-training strategies are effective and lead to improved performance on all tasks and metrics on VidSitu benchmark [37] (e.g., improving on semantic role prediction from 47% to 61% CIDEr scores). We further demonstrate the effectiveness of our contextualized event features on LVU tasks [54], both when used alone and when combined with instance features, showing their complementarity.

----

## [942] Anomaly Detection via Reverse Distillation from One-Class Embedding

**Authors**: *Hanqiu Deng, Xingyu Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00951](https://doi.org/10.1109/CVPR52688.2022.00951)

**Abstract**:

Knowledge distillation (KD) achieves promising results on the challenging problem of unsupervised anomaly detection (AD). The representation discrepancy of anomalies in the teacher-student (T-S) model provides essential evidence for AD. However, using similar or identical architectures to build the teacher and student models in previous studies hinders the diversity of anomalous representations. To tackle this problem, we propose a novel T-S model consisting of a teacher encoder and a student decoder and introduce a simple yet effective “reverse distillation” paradigm accordingly. Instead of receiving raw images directly, the student network takes teacher model's one-class embedding as input and targets to restore the teacher's multi-scale representations. Inherently, knowledge distillation in this study starts from abstract, high-level presentations to low-level features. In addition, we introduce a trainable one-class bottleneck embedding (OCBE) module in our T-S model. The obtained compact embedding effectively preserves essential information on normal patterns, but aban-dons anomaly perturbations. Extensive experimentation on AD and one-class novelty detection benchmarks shows that our method surpasses SOTA performance, demonstrating our proposed approach's effectiveness and generalizability.

----

## [943] Unsupervised Representation Learning for Binary Networks by Joint Classifier Learning

**Authors**: *Dahyun Kim, Jonghyun Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00952](https://doi.org/10.1109/CVPR52688.2022.00952)

**Abstract**:

Self-supervised learning is a promising unsupervised learning framework that has achieved success with large floating point networks. But such networks are not readily deployable to edge devices. To accelerate deployment of models with the benefit of unsupervised representation learning to such resource limited devices for various downstream tasks, we propose a self-supervised learning method for binary networks that uses a moving target network. In particular, we propose to Jointly train a randomly initialized classifier, attached to a pretrained floating point feature extractor, with a binary network. Additionally, we propose a feature similarity loss, a dynamic loss balancing and modified multi-stage training to further improve the accuracy, and call our method BURN. Our empirical validations over five downstream tasks using seven datasets show that BURN outperforms self-supervised baselines for binary networks and sometimes outperforms supervised pretraining. Code is availabe at https://github.com/naver-ai/burn.

----

## [944] DC-SSL: Addressing Mismatched Class Distribution in Semi-supervised Learning

**Authors**: *Zhen Zhao, Luping Zhou, Yue Duan, Lei Wang, Lei Qi, Yinghuan Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00953](https://doi.org/10.1109/CVPR52688.2022.00953)

**Abstract**:

Consistency-based Semi-supervised learning (SSL) has achieved promising performance recently. However, the success largely depends on the assumption that the labeled and unlabeled data share an identical class distribution, which is hard to meet in real practice. The distribution mismatch between the labeled and unlabeled sets can cause severe bias in the pseudo-labels of SSL, resulting in significant performance degradation. To bridge this gap, we put forward a new SSL learning framework, named Distribution Consistency SSL (DC-SSL), which rectifies the pseudolabels from a distribution perspective. The basic idea is to directly estimate a reference class distribution (RCD), which is regarded as a surrogate of the ground truth class distribution about the unlabeled data, and then improve the pseudo-labels by encouraging the predicted class distribution (PCD) of the unlabeled data to approach RCD gradually. To this end, this paper revisits the Exponentially Moving Average (EMA) model and utilizes it to estimate RCD in an iteratively improved manner, which is achieved with a momentum-update scheme throughout the training procedure. On top of this, two strategies are proposed for RCD to rectify the pseudo-label prediction, respectively. They correspond to an efficient training-free scheme and a training-based alternative that generates more accurate and reliable predictions. DC-SSL is evaluated on multiple SSL benchmarks and demonstrates remarkable performance improvement over competitive methods under matched- and mismatched-distribution scenarios.

----

## [945] Learning to Collaborate in Decentralized Learning of Personalized Models

**Authors**: *Shuangtong Li, Tianyi Zhou, Xinmei Tian, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00954](https://doi.org/10.1109/CVPR52688.2022.00954)

**Abstract**:

Learning personalized models for user-customized computer-vision tasks is challenging due to the limited private-data and computation available on each edge device. Decentralized learning (DL) can exploit the images distributed over devices on a network topology to train a global model but is not designed to train personalized models for different tasks or optimize the topology. Moreover, the mixing weights used to aggregate neighbors' gradient messages in DL can be suboptimal for personalization since they are not adaptive to different nodes/tasks and learning stages. In this paper, we dynamically update the mixing-weights to improve the personalized model for each node's task and meanwhile learn a sparse topology to reduce communication costs. Our first approach, “learning to collaborate (L2C) ”, directly optimizes the mixing weights to minimize the local validation loss per node for a predefined set of nodes/tasks. In order to produce mixing weights for new nodes or tasks, we further develop “meta-L2C‘, which learns an attention mechanism to automatically assign mixing weights by comparing two nodes' model updates. We evaluate both methods on diverse benchmarks and experimental settings for image classification. Thorough comparisons to both classical and recent methods for IID/non-IID decentralized and federated learning demonstrate our method's advantages in identifying collaborators among nodes, learning sparse topology, and producing better personalized models with low communication and computational cost.

----

## [946] Highly-efficient Incomplete Largescale Multiview Clustering with Consensus Bipartite Graph

**Authors**: *Siwei Wang, Xinwang Liu, Li Liu, Wenxuan Tu, Xinzhong Zhu, Jiyuan Liu, Sihang Zhou, En Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00955](https://doi.org/10.1109/CVPR52688.2022.00955)

**Abstract**:

Multiview clustering has received increasing attention due to its effectiveness in fusing complementary information without manual annotations. Most previous methods hold the assumption that each instance appears in all views. However, it is not uncommon to see that some views may contain some missing instances, which gives rise to incomplete multi-view clustering (IMVC) in literature. Although many IMVC methods have been recently proposed, they always encounter high complexity and expensive time expenditure from being applied into large-scale tasks. In this paper, we present a flexible highly-efficient incomplete large-scale multi-view clustering approach based on bipartite graph framework to solve these issues. Specifically, we formalize multi-view anchor learning and incomplete bipartite graph into a unified framework, which coordinates with each other to boost cluster performance. By introducing the flexible bipartite graph framework to handle IMVC for the first practice, our proposed method enjoys linear complexity respecting to instance numbers, which is more applicable for large-scale IMVC tasks. Comprehensive experimental results on various benchmark datasets demonstrate the effectiveness and efficiency of our proposed algorithm against other IMVC competitors. The code is available at
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/wangsiwei2010/CVPR22-IMVC-CBG.

----

## [947] DASO: Distribution-Aware Semantics-Oriented Pseudo-label for Imbalanced Semi-Supervised Learning

**Authors**: *Youngtaek Oh, Dong-Jin Kim, In So Kweon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00956](https://doi.org/10.1109/CVPR52688.2022.00956)

**Abstract**:

The capability of the traditional semi-supervised learning (SSL) methods is far from real-world application due to severely biased pseudo-labels caused by (1) class imbalance and (2) class distribution mismatch between labeled and unlabeled data. This paper addresses such a relatively under-explored problem. First, we propose a general pseudo-labeling framework that class-adaptively blends the semantic pseudo-label from a similarity-based classifier to the linear one from the linear classifier, after making the observation that both types of pseudo-labels have complementary properties in terms of bias. We further introduce a novel semantic alignment loss to establish balanced feature representation to reduce the biased predictions from the classifier. We term the whole framework as Distribution-Aware Semantics-Oriented (DASO) Pseudo-label. We conduct extensive experiments in a wide range of imbalanced benchmarks: CIFAR10/100-LT, STL10-LT, and large-scale long-tailed Semi-Aves with open-set class, and demonstrate that, the proposed DASO framework reliably improves SSL learners with unlabeled data especially when both (1) class imbalance and (2) distribution mismatch dominate.

----

## [948] Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning

**Authors**: *Haoxiang Wang, Yite Wang, Ruoyu Sun, Bo Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00957](https://doi.org/10.1109/CVPR52688.2022.00957)

**Abstract**:

Model-agnostic meta-learning (MAML) and its variants have become popular approaches for few-shot learning. However, due to the non-convexity of deep neural nets (DNNs) and the bi-level formulation of MAML, the theoretical properties of MAML with DNNs remain largely unknown. In this paper, we first prove that MAML with over-parameterized DNNs is guaranteed to converge to global optima at a linear rate. Our convergence analysis indicates that MAML with over-parameterized DNNs is equivalent to kernel regression with a novel class of kernels, which we name as Meta Neural Tangent Kernels (MetaNTK). Then, we propose MetaNTK-NAS, a new training-free neural architecture search (NAS) method for few-shot learning that uses MetaNTK to rank and select architectures. Empirically, we compare our MetaNTK-NAS with previous NAS methods on two popular few-shot learning benchmarks, miniImageNet, and tieredImageNet. We show that the performance of MetaNTK-NAS is comparable or better than the state-of-the-art NAS method designed for few-shot learning while enjoying more than 100x speedup. We believe the efficiency of MetaNTK-NAS makes itself more practical for many real-world tasks. Our code is released at github.com/YiteWang/MetaNTK-NAS.

----

## [949] Semi-Supervised Object Detection via Multi-instance Alignment with Global Class Prototypes

**Authors**: *Aoxue Li, Peng Yuan, Zhenguo Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00958](https://doi.org/10.1109/CVPR52688.2022.00958)

**Abstract**:

Semi-Supervised object detection (SSOD) aims to improve the generalization ability of object detectors with large-scale unlabeled images. Current pseudo-labeling-based SSOD methods individually learn from labeled data and unlabeled data, without considering the relation be-tween them. To make full use of labeled data, we pro-pose a Multi-instance Alignment model which enhances the prediction consistency based on Global Class Proto-types (MA-GCP). Specifically, we impose the consistency between pseudo ground-truths and their high-IoU candi-dates by minimizing the cross-entropy loss of their class distributions computed based on global class prototypes. These global class prototypes are estimated with the whole labeled dataset via the exponential moving average algorithm. To evaluate the proposed MA-GCP model, we inte-grate it into the state-of-the-art SSOD framework and ex-periments on two benchmark datasets demonstrate the ef-fectiveness of our MA-GCP approach.

----

## [950] Unbiased Teacher v2: Semi-supervised Object Detection for Anchor-free and Anchor-based Detectors

**Authors**: *Yen-Cheng Liu, Chih-Yao Ma, Zsolt Kira*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00959](https://doi.org/10.1109/CVPR52688.2022.00959)

**Abstract**:

With the recent development of Semi-Supervised Object Detection (SS-OD) techniques, object detectors can be improved by using a limited amount of labeled data and abundant unlabeled data. However, there are still two challenges that are not addressed: (1) there is no prior SS-OD work on anchor-free detectors, and (2) prior works are ineffective when pseudo-labeling bounding box regression. In this paper, we present Unbiased Teacher v2, which shows the generalization of SS-OD method to anchor-free detectors and also introduces Listen2Student mechanism for the unsupervised regression loss. Specifically, we first present a study examining the effectiveness of existing SS-OD methods on anchor-free detectors and find that they achieve much lower performance improvements under the semi-supervised setting. We also observe that box selection with centerness and the localization-based labeling used in anchor-free detectors cannot work well under the semi-supervised setting. On the other hand, our Listen2Student mechanism explicitly prevents misleading pseudo-labels in the training of bounding box regression; we specifically develop a novel pseudo-labeling selection mechanism based on the Teacher and Student's relative uncertainties. This idea contributes to favorable improvement in the regression branch in the semi-supervised setting. Our method, which works for both anchor-free and anchor-based methods, consistently performs favorably against the state-of-the-art methods in VOC, COCO-standard, and COCO-additional.

----

## [951] Spectral Unsupervised Domain Adaptation for Visual Recognition

**Authors**: *Jingyi Zhang, Jiaxing Huang, Zichen Tian, Shijian Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00960](https://doi.org/10.1109/CVPR52688.2022.00960)

**Abstract**:

Though unsupervised domain adaptation (UDA) has achieved very impressive progress recently, it remains a great challenge due to missing target annotations and the rich discrepancy between source and target distributions. We propose Spectral UDA (SUDA), an effective and efficient UDA technique that works in the spectral space and can generalize across different visual recognition tasks. SUDA addresses the UDA challenges from two perspectives. First, it introduces a spectrum transformer (ST) that mitigates inter-domain discrepancies by enhancing domain-invariant spectra while suppressing domain-variant spectra of source and target samples simultaneously. Second, it introduces multi-view spectral learning that learns useful unsupervised representations by maximizing mutual information among multiple ST-generated spectral views of each target sample. Extensive experiments show that SUDA achieves superior accuracy consistently across different visual tasks in object detection, semantic segmentation and image classification. Additionally, SUDA also works with the transformer-based network and achieves state-of-the-art performance on object detection.

----

## [952] DATA: Domain-Aware and Task-Aware Self-supervised Learning

**Authors**: *Qing Chang, Junran Peng, Lingxi Xie, Jiajun Sun, Haoran Yin, Qi Tian, Zhaoxiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00961](https://doi.org/10.1109/CVPR52688.2022.00961)

**Abstract**:

The paradigm of training models on massive data without label through self-supervised learning (SSL) and fine-tuning on many downstream tasks has become a trend recently. However, due to the high training costs and the un-consciousness of downstream usages, most self-supervised learning methods lack the capability to correspond to the diversities of downstream scenarios, as there are various data domains, different vision tasks and latency constraints on models. Neural architecture search (NAS) is one universally acknowledged fashion to conquer the issues above, but applying NAS on SSL seems impossible as there is no label or metric provided for judging model selection. In this paper, we present DATA, a simple yet effective NAS approach specialized for SSL that provides Domain-Aware and Task-Aware pre-training. Specifically, we (i) train a supernet which could be deemed as a set of millions of networks covering a wide range of model scales without any label, (ii) propose a flexible searching mechanism compatible with SSL that enables finding networks of different computation costs, for various downstream vision tasks and data domains without explicit metric provided. Instantiated With MoCo v2, our method achieves promising results across a wide range of computation costs on down-stream tasks, including image classification, object detection and semantic segmentation. DATA is orthogonal to most existing SSL methods and endows them the ability of customization on downstream needs. Extensive experiments on other SSL methods demonstrate the generalizability of the proposed method. Code is released at https://github.com/GAIA-vision/GAIA-ssl.

----

## [953] Dynamic Kernel Selection for Improved Generalization and Memory Efficiency in Meta-learning

**Authors**: *Arnav Chavan, Rishabh Tiwari, Udbhav Bamba, Deepak K. Gupta*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00962](https://doi.org/10.1109/CVPR52688.2022.00962)

**Abstract**:

Gradient based meta-learning methods are prone to overfit on the meta-training set, and this behaviour is more prominent with large and complex networks. Moreover, large networks restrict the application of meta-learning models on low-power edge devices. While choosing smaller networks avoid these issues to a certain extent, it affects the overall generalization leading to reduced performance. Clearly, there is an approximately optimal choice of network architecture that is best suited for every meta-learning problem, however, identifying it beforehand is not straight-forward. In this paper, we present Metadock, a task-specific dynamic kernel selection strategy for designing compressed CNN models that generalize well on unseen tasks in meta-learning. Our method is based on the hypothesis that for a given set of similar tasks, not all kernels of the network are needed by each individual task. Rather, each task uses only a fraction of the kernels, and the selection of the kernels per task can be learnt dynamically as a part of the inner update steps. Metadockcompresses the meta-model as well as the task-specific inner models, thus providing significant reduction in model size for each task, and through constraining the number of active kernels for every task, it implicitly mitigates the issue of meta-overfitting. We show that for the same inference budget, pruned versions of large CNN models obtained using our approach consistently outperform the conventional choices of CNN models. Metadock couples well with popular meta-learning approaches such as iMAML [22]. The efficacy of our method is validated on CIFAR-fs [1] and mini-ImageNet [28] datasets, and we have observed that our approach can provide improvements in model accuracy of up to 2% on standard meta-learning benchmark, while reducing the model size by more than 75%. Our code is available at https://github.com/transmuteAI/MetaDOCK.

----

## [954] DeepDPM: Deep Clustering With an Unknown Number of Clusters

**Authors**: *Meitar Ronen, Shahaf E. Finder, Oren Freifeld*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00963](https://doi.org/10.1109/CVPR52688.2022.00963)

**Abstract**:

Deep Learning (DL) has shown great promise in the unsupervised task of clustering. That said, while in classical (i.e., non-deep) clustering the benefits of the nonparametric approach are well known, most deep-clustering methods are parametric: namely, they require a predefined and fixed number of clusters, denoted by K. When 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$K$</tex>
 is unknown, however, using model-selection criteria to choose its optimal value might become computationally expensive, especially in DL as the training process would have to be repeated numerous times. In this work, we bridge this gap by introducing an effective deep-clustering method that does not require knowing the value of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$K$</tex>
 as it infers it during the learning. Using a split/merge framework, a dynamic architecture that adapts to the changing 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$K$</tex>
, and a novel loss, our proposed method outperforms existing nonparametric methods (both classical and deep ones). While the very few existing deep nonparametric methods lack scalability, we demonstrate ours by being the first to report the performance of such a method on ImageNet. We also demonstrate the importance of inferring 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$K$</tex>
 by showing how methods that fix it deteriorate in performance when their assumed 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$K$</tex>
 value gets further from the ground-truth one, especially on imbalanced datasets. Our code is available at https://github.com/BGU-CS-VIL/DeepDPM.

----

## [955] PLAD: Learning to Infer Shape Programs with Pseudo-Labels and Approximate Distributions

**Authors**: *R. Kenny Jones, Homer Walke, Daniel Ritchie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00964](https://doi.org/10.1109/CVPR52688.2022.00964)

**Abstract**:

Inferring programs which generate 2D and 3D shapes is important for reverse engineering, editing, and more. Training models to perform this task is complicated because paired (shape, program) data is not readily available for many domains, making exact supervised learning infeasible. However, it is possible to get paired data by compromising the accuracy of either the assigned program labels or the shape distribution. Wake-sleep methods use samples from a generative model of shape programs to approximate the distribution of real shapes. In self-training, shapes are passed through a recognition model, which predicts programs that are treated as ‘pseudo-labels’ for those shapes. Related to these approaches, we introduce a novel self-training variant unique to program inference, where program pseudo-labels are paired with their executed output shapes, avoiding label mismatch at the cost of an approximate shape distribution. We propose to group these regimes under a single conceptual framework, where training is performed with maximum likelihood updates sourced from either Pseudo-Labels or an Approximate Distribution (PLAD). We evaluate these techniques on multiple 2D and 3D shape program inference domains. Compared with policy gradient reinforcement learning, we show that PLAD techniques infer more accurate shape programs and converge significantly faster. Finally, we propose to combine updates from different PLAD methods within the training of a single model, and find that this approach outperforms any individual technique.

----

## [956] Robust outlier detection by de-biasing VAE likelihoods

**Authors**: *Kushal Chauhan, Barath Mohan Umapathi, Pradeep Shenoy, Manish Gupta, Devarajan Sridharan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00965](https://doi.org/10.1109/CVPR52688.2022.00965)

**Abstract**:

Deep networks often make confident, yet, incorrect, predictions when tested with outlier data that is far removed from their training distributions. Likelihoods computed by deep generative models (DGMs) are a candidate metric for outlier detection with unlabeled data. Yet, previous studies have shown that DGM likelihoods are unreliable and can be easily biased by simple transformations to input data. Here, we examine outlier detection with variational autoencoders (VAEs), among the simplest of DGMs. We propose novel analytical and algorithmic approaches to ameliorate key biases with VAE likelihoods. Our bias corrections are sample-specific, computationally inexpensive, and readily computed for various decoder visible distributions. Next, we show that a well-known image pre-processing technique – contrast stretching – extends the effectiveness of bias correction to further improve outlier detection. Our approach achieves state-of-the-art accuracies with nine grayscale and natural image datasets, and demonstrates significant advantages – both with speed and performance – over four recent, competing approaches. In summary, lightweight remedies suffice to achieve robust outlier detection with VAEs.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/google-research/google-research/tree/master/vae_ood.

----

## [957] Image-to-Lidar Self-Supervised Distillation for Autonomous Driving Data

**Authors**: *Corentin Sautier, Gilles Puy, Spyros Gidaris, Alexandre Boulch, Andrei Bursuc, Renaud Marlet*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00966](https://doi.org/10.1109/CVPR52688.2022.00966)

**Abstract**:

Segmenting or detecting objects in sparse Lidar point clouds are two important tasks in autonomous driving to allow a vehicle to act safely in its 3D environment. The best performing methods in 3D semantic segmentation or object detection rely on a large amount of annotated data. Yet annotating 3D Lidar data for these tasks is tedious and costly. In this context, we propose a self-supervised pretraining method for 3D perception models that is tailored to autonomous driving data. Specifically, we leverage the availability of synchronized and calibrated image and Lidar sensors in autonomous driving setups for distilling self-supervised pre-trained image representations into 3D models. Hence, our method does not require any point cloud nor image annotations. The keyingredient of our method is the use of superpixels which are used to pool 3D point features and 2D pixel features in visually similar regions. We then train a 3D network on the self-supervised task of matching these pooled point features with the corresponding pooled image pixel features. The advantages of contrasting regions obtained by superpixels are that: (1) grouping together pixels and points of visually coherent regions leads to a more meaningful contrastive task that produces features well adapted to 3D semantic segmentation and 3D object detection; (2) all the different regions have the same weight in the contrastive loss regardless of the number of 3D points sampled in these regions; (3) it mitigates the noise produced by incorrect matching of points and pixels due to occlusions between the different sensors. Extensive experiments on autonomous driving datasets demonstrate the ability of our image-to-Lidar distillation strategy to produce 3D representations that transfer well on semantic segmentation and object detection tasks.

----

## [958] CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding

**Authors**: *Mohamed Afham, Isuru Dissanayake, Dinithi Dissanayake, Amaya Dharmasiri, Kanchana Thilakarathna, Ranga Rodrigo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00967](https://doi.org/10.1109/CVPR52688.2022.00967)

**Abstract**:

Manual annotation of large-scale point cloud dataset for varying tasks such as 3D object classification, segmentation and detection is often laborious owing to the irregular structure of point clouds. Self-supervised learning, which operates without any human labeling, is a promising approach to address this issue. We observe in the real world that humans are capable of mapping the visual concepts learnt from 2D images to understand the 3D world. Encouraged by this insight, we propose CrossPoint, a simple cross-modal contrastive learning approach to learn transferable 3D point cloud representations. It enables a 3D-2D correspondence of objects by maximizing agreement between point clouds and the corresponding rendered 2D image in the invariant space, while encouraging invariance to transformations in the point cloud modality. Our joint training objective combines the feature correspondences within and across modalities, thus ensembles a rich learning signal from both 3D point cloud and 2D image modalities in a self-supervised fashion. Experimental results show that our approach outperforms the previous unsupervised learning methods on a diverse range of downstream tasks including 3D object classification and segmentation. Further, the ablation studies validate the potency of our approach for a better point cloud understanding. Code and pretrained models are available at https://github.com/MohamedAfham/CrossPoint.

----

## [959] Cross-Domain Correlation Distillation for Unsupervised Domain Adaptation in Nighttime Semantic Segmentation

**Authors**: *Huan Gao, Jichang Guo, Guoli Wang, Qian Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00968](https://doi.org/10.1109/CVPR52688.2022.00968)

**Abstract**:

The performance of nighttime semantic segmentation is restricted by the poor illumination and a lack of pixel-wise annotation, which severely limit its application in autonomous driving. Existing works, e.g., using the twilight as the intermediate target domain to perform the adaptation from daytime to nighttime, may fail to cope with the inherent difference between datasets caused by the camera equipment and the urban style. Faced with these two types of domain shifts, i.e., the illumination and the inherent difference of the datasets, we propose a novel domain adaptation framework via cross-domain correlation distillation, called CCDistill. The invariance of illumination or inherent difference between two images is fully explored so as to make up for the lack of labels for nighttime images. Specifically, we extract the content and style knowledge contained in features, calculate the degree of inherent or illumination difference between two images. The domain adaptation is achieved using the invariance of the same kind of difference. Extensive experiments on Dark Zurich and ACDC demon-strate that CCDistill achieves the state-of-the-art performance for nighttime semantic segmentation. Notably, our method is a one-stage domain adaptation network which can avoid affecting the inference time. Our implementation is available at https://github.com/ghuan99/CCDistill.

----

## [960] DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation

**Authors**: *Lukas Hoyer, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00969](https://doi.org/10.1109/CVPR52688.2022.00969)

**Abstract**:

As acquiring pixel-wise annotations of real-world images for semantic segmentation is a costly process, a model can instead be trained with more accessible synthetic data and adapted to real images without requiring their annotations. This process is studied in unsupervised domain adaptation (UDA). Even though a large number of methods propose new adaptation strategies, they are mostly based on outdated network architectures. As the influence of recent network architectures has not been systematically studied, we first benchmark different network architectures for UDA and newly reveal the potential of Transformers for UDA semantic segmentation. Based on the findings, we propose a novel UDA method, DAFormer. The network architecture of DAFormer consists of a Transformer encoder and a multi-level context-aware feature fusion decoder. It is enabled by three simple but crucial training strategies to stabilize the training and to avoid overfitting to the source domain: While (1) Rare Class Sampling on the source domain improves the quality of the pseudo-labels by mitigating the confirmation bias of self-training toward common classes, (2) a Thing-Class ImageNet Feature Distance and (3) a learning rate warmup promote feature transfer from ImageNet pretraining. DAFormer represents a major advance in UDA. It improves the state of the art by 10.8 mIoU for GTA→Cityscapes and 5.4 mIoU for Syruhia→Cityscapes and enables learning even difficult classes such as train, bus, and truck well. The implementation is available at https://github.com/lhoyer/DAFormer.

----

## [961] WildNet: Learning Domain Generalized Semantic Segmentation from the Wild

**Authors**: *Suhyeon Lee, Hongje Seong, Seongwon Lee, Euntai Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00970](https://doi.org/10.1109/CVPR52688.2022.00970)

**Abstract**:

We present a new domain generalized semantic segmentation network named WildNet, which learns domain-generalized features by leveraging a variety of contents and styles from the wild. In domain generalization, the low generalization ability for unseen target domains is clearly due to overfitting to the source domain. To address this problem, previous works have focused on generalizing the domain by removing or diversifying the styles of the source domain. These alleviated overfitting to the source-style but overlooked overfitting to the source-content. In this paper, we propose to diversify both the content and style of the source domain with the help of the wild. Our main idea is for networks to naturally learn domain-generalized semantic information from the wild. To this end, we diversify styles by augmenting source features to resemble wild styles and enable networks to adapt to a variety of styles. Further-more, we encourage networks to learn class-discriminant features by providing semantic variations borrowed from the wild to source contents in the feature space. Finally, we regularize networks to capture consistent semantic information even when both the content and style of the source domain are extended to the wild. Extensive experiments on five different datasets validate the effectiveness of our WildNet, and we significantly outperform state-of-the-art methods. The source code and model are available online: https://github.com/suhyeonlee/WildNet.

----

## [962] UCC: Uncertainty guided Cross-head Cotraining for Semi-Supervised Semantic Segmentation

**Authors**: *Jiashuo Fan, Bin Gao, Huan Jin, Lihui Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00971](https://doi.org/10.1109/CVPR52688.2022.00971)

**Abstract**:

Deep neural networks (DNNs) have witnessed great successes in semantic segmentation, which requires a large number of labeled data for training. We present a novel learning framework called Uncertainty guided Cross-head Cotraining (UCC) for semi-supervised semantic segmentation. Our framework introduces weak and strong augmentations within a shared encoder to achieve cotraining, which naturally combines the benefits of consistency and self-training. Every segmentation head interacts with its peers and, the weak augmentation result is used for supervising the strong. The consistency training samples' diversity can be boosted by Dynamic Cross-Set Copy-Paste (DCSCP), which also alleviates the distribution mismatch and class imbalance problems. Moreover, our proposed Uncertainty Guided Re-weight Module (UGRM) enhances the self-training pseudo labels by suppressing the effect of the low-quality pseudo labels from its peer via modeling uncertainty. Extensive experiments on Cityscapes and PASCAL VOC 2012 demonstrate the effectiveness of our UCC. Our approach significantly outperforms other state-of-the-art semi-supervised semantic segmentation methods. It achieves 77.17%, 76.49% mIoU on Cityscapes and PASCAL VOC 2012 datasets respectively under 1/16 protocols, which are + 10.1%, + 7.91% better than the supervised baseline.

----

## [963] Semi-supervised Semantic Segmentation with Error Localization Network

**Authors**: *Donghyeon Kwon, Suha Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00972](https://doi.org/10.1109/CVPR52688.2022.00972)

**Abstract**:

This paper studies semi-supervised learning of semantic segmentation, which assumes that only a small portion of training images are labeled and the others remain unlabeled. The unlabeled images are usually assigned pseudo labels to be used in training, which however often causes the risk of performance degradation due to the confirmation bias towards errors on the pseudo labels. We present a novel method that resolves this chronic issue of pseudo labeling. At the heart of our method lies error localization network (ELN), an auxiliary module that takes an image and its segmentation prediction as input and identifies pixels whose pseudo labels are likely to be wrong. ELN enables semi-supervised learning to be robust against inaccurate pseudo labels by disregarding label noises during training and can be naturally integrated with self-training and contrastive learning. Moreover, we introduce a new learning strategy for ELN that simulates plausible and diverse segmentation errors during training of ELN to enhance its generalization. Our method is evaluated on PASCAL VOC 2012 and Cityscapes, where it outperforms all existing methods in every evaluation setting.

----

## [964] Unbiased Subclass Regularization for Semi-Supervised Semantic Segmentation

**Authors**: *Dayan Guan, Jiaxing Huang, Aoran Xiao, Shijian Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00973](https://doi.org/10.1109/CVPR52688.2022.00973)

**Abstract**:

Semi-supervised semantic segmentation learns from small amounts of labelled images and large amounts of unlabelled images, which has witnessed impressive progress with the recent advance of deep neural networks. However, it often suffers from severe class-bias problem while exploring the unlabelled images, largely due to the clear pixel-wise class imbalance in the labelled images. This paper presents an unbiased subclass regularization network (USRN) that alleviates the class imbalance issue by learning class-unbiased segmentation from balanced subclass distributions. We build the balanced subclass distributions by clustering pixels of each original class into multiple subclasses of similar sizes, which provide class-balanced pseudo supervision to regularize the class-biased segmentation. In addition, we design an entropy-based gate mechanism to coordinate learning between the original classes and the clustered subclasses which facilitates subclass regularization effectively by suppressing unconfident subclass predictions. Extensive experiments over multiple public benchmarks show that USRN achieves superior performance as compared with the state-of-the-art.

----

## [965] Integrative Few-Shot Learning for Classification and Segmentation

**Authors**: *Dahyun Kang, Minsu Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00974](https://doi.org/10.1109/CVPR52688.2022.00974)

**Abstract**:

We introduce the integrative task of few-shot classification and segmentation (FS-CS) that aims to both classify and segment target objects in a query image when the target classes are given with a few examples. This task combines two conventional few-shot learning problems, few-shot classification and segmentation. FS-CS generalizes them to more realistic episodes with arbitrary image pairs, where each target class may or may not be present in the query. To address the task, we propose the integrative few-shot learning (iFSL) framework for FS-CS, which trains a learner to construct class-wise foreground maps for multi-label classification and pixel-wise segmentation. We also develop an effective iFSL model, attentive squeeze network (ASNet), that leverages deep semantic correlation and global self-attention to produce reliable foreground maps. In experiments, the proposed method shows promising performance on the FS-CS task and also achieves the state of the art on standard few-shot segmentation benchmarks.

----

## [966] GANORCON: Are Generative Models Useful for Few-shot Segmentation?

**Authors**: *Oindrila Saha, Zezhou Cheng, Subhransu Maji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00975](https://doi.org/10.1109/CVPR52688.2022.00975)

**Abstract**:

Advances in generative modeling based on GANs has motivated the community to find their use beyond image generation and editing tasks. In particular, several re-cent works have shown that GAN representations can be re-purposed for discriminative tasks such as part segmen-tation, especially when training data is limited. But how do these improvements stack-up against recent advances in self-supervised learning? Motivated by this we present an alternative approach based on contrastive learning and compare their performance on standard few-shot part seg-mentation benchmarks. Our experiments reveal that not only do the GAN-based approach offer no significant per-formance advantage, their multi-step training is complex, nearly an order-of-magnitude slower, and can introduce ad-ditional bias. These experiments suggest that the inductive biases of generative models, such as their ability to dis-entangle shape and texture, are well captured by standard feed-forward networks trained using contrastive learning.

----

## [967] SphericGAN: Semi-supervised Hyper-spherical Generative Adversarial Networks for Fine-grained Image Synthesis

**Authors**: *Tianyi Chen, Yunfei Zhang, Xiaoyang Huo, Si Wu, Yong Xu, Hau-San Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00976](https://doi.org/10.1109/CVPR52688.2022.00976)

**Abstract**:

Generative Adversarial Network (GAN)-based models have greatly facilitated image synthesis. However, the model performance may be degraded when applied to finegrained data, due to limited training samples and subtle distinction among categories. Different from generic GAN-s, we address the issue from a new perspective of discovering and utilizing the underlying structure of real data to explicitly regularize the spatial organization of latent space. To reduce the dependence of generative models on labeled data, we propose a semi-supervised hyper-spherical GAN for class-conditional fine-grained image generation, and our model is referred to as SphericGAN. By projecting random vectors drawn from a prior distribution onto a hyper-sphere, we can model more complex distributions, while at the same time the similarity between the resulting latent vectors depends only on the angle, but not on their magnitudes. On the other hand, we also incorporate a mapping network to map real images onto the hyper-sphere, and match latent vectors with the underlying structure of real data via real-fake cluster alignment. As a result, we obtain a spatially organized latent space, which is useful for capturing class-independent variation factors. The experi-mental results suggest that our SphericGAN achieves state-of-the-art performance in synthesizing high-fidelity images with precise class semantics.

----

## [968] CoordGAN: Self-Supervised Dense Correspondences Emerge from GANs

**Authors**: *Jiteng Mu, Shalini De Mello, Zhiding Yu, Nuno Vasconcelos, Xiaolong Wang, Jan Kautz, Sifei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00977](https://doi.org/10.1109/CVPR52688.2022.00977)

**Abstract**:

Recent advances show that Generative Adversarial Networks (GANs) can synthesize images with smooth variations along semantically meaningful latent directions, such as pose, expression, layout, etc. While this indicates that GANs implicitly learn pixel-level correspondences across images, few studies explored how to extract them explicitly. In this work, we introduce Coordinate GAN (CoordGAN), a structure-texture disentangled GAN that learns a dense correspondence map for each generated image. We represent the correspondence maps of different images as warped coordinate frames transformed from a canonical coordinate frame, i.e., the correspondence map, which describes the structure (e.g., the shape of a face), is controlled via a transformation. Hence, finding correspondences boils down to locating the same coordinate in different correspondence maps. In CoordGAN, we sample a transformation to represent the structure of a synthesized instance, while an independent texture branch is responsible for rendering appearance details orthogonal to the structure. Our approach can also extract dense correspondence maps for real images by adding an encoder on top of the generator. We quantitatively demonstrate the quality of the learned dense correspondences through segmentation mask transfer on multiple datasets. We also show that the proposed generator achieves better structure and texture disentanglement compared to existing approaches. Project page: https://jitengmu.github.io/CoordGAN/

----

## [969] GradViT: Gradient Inversion of Vision Transformers

**Authors**: *Ali Hatamizadeh, Hongxu Yin, Holger Roth, Wenqi Li, Jan Kautz, Daguang Xu, Pavlo Molchanov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00978](https://doi.org/10.1109/CVPR52688.2022.00978)

**Abstract**:

In this work we demonstrate the vulnerability of vision transformers (ViTs) to gradient-based inversion attacks. During this attack, the original data batch is reconstructed given model weights and the corresponding gradients. We introduce a method, named GradViT, that optimizes random noise into naturally looking images via an iterative process. The optimization objective consists of (i) a loss on matching the gradients, (ii) image prior in the form of distance to batch-normalization statistics of a pretrained CNN model, and (iii) a total variation regularization on patches to guide correct recovery locations. We propose a unique loss scheduling function to overcome local minima during optimization. We evaluate GadViT on ImageNet1K and MS-Celeb-1M datasets, and observe unprecedentedly high fidelity and closeness to the original (hidden) data. During the analysis we find that vision transformers are significantly more vulnerable than previously studied CNNs due to the presence of the attention mechanism. Our method demonstrates new state-of-the-art results for gradient inversion in both qualitative and quantitative metrics. Project page at https://gradvit.github.io/.

----

## [970] Deep 3D-to-2D Watermarking: Embedding Messages in 3D Meshes and Extracting Them from 2D Renderings

**Authors**: *Innfarn Yoo, Huiwen Chang, Xiyang Luo, Ondrej Stava, Ce Liu, Peyman Milanfar, Feng Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00979](https://doi.org/10.1109/CVPR52688.2022.00979)

**Abstract**:

Digital watermarking is widely used for copyright protection. Traditional 3D watermarking approaches or commercial software are typically designed to embed messages into 3D meshes, and later retrieve the messages directly from distorted/undistorted watermarked 3D meshes. However, in many cases, users only have access to rendered 2D images instead of 3D meshes. Unfortunately, retrieving messages from 2D renderings of 3D meshes is still challenging and underexplored. We introduce a novel end-to-end learning framework to solve this problem through: 1) an encoder to covertly embed messages in both mesh geometry and textures; 2) a differentiable renderer to render watermarked 3D objects from different camera angles and under varied lighting conditions; 3) a decoder to recover the messages from 2D rendered images. From our experiments, we show that our model can learn to embed information visually imperceptible to humans, and to retrieve the embedded information from 2D renderings that undergo 3D distortions. In addition, we demonstrate that our method can also work with other renderers, such as ray tracers and real-time renderers with and without fine-tuning.

----

## [971] CD2-pFed: Cyclic Distillation-guided Channel Decoupling for Model Personalization in Federated Learning

**Authors**: *Yiqing Shen, Yuyin Zhou, Lequan Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00980](https://doi.org/10.1109/CVPR52688.2022.00980)

**Abstract**:

Federated learning (FL) is a distributed learning paradigm that enables multiple clients to collaboratively learn a shared global model. Despite the recent progress, it remains challenging to deal with heterogeneous data clients, as the discrepant data distributions usually prevent the global model from delivering good generalization ability on each participating client. In this paper, we propose CD
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-pFed, a novel Cyclic Distillation-guided Channel Decoupling framework, to personalize the global model in FL, under various settings of data heterogeneity. Different from previous works which establish layer-wise personalization to overcome the non-IID data across different clients, we make the first attempt at channel-wise assignment for model personalization, referred to as channel decoupling. To further facilitate the collaboration between private and shared weights, we propose a novel cyclic distillation scheme to impose a consistent regularization between the local and global model representations during the federation. Guided by the cyclical distillation, our channel decoupling framework can deliver more accurate and generalized results for different kinds of heterogeneity, such as feature skew, label distribution skew, and concept shift. Comprehensive experiments on four benchmarks, including natural image and medical image analysis tasks, demonstrate the consistent effectiveness of our method on both local and external validations.

----

## [972] APRIL: Finding the Achilles' Heel on Privacy for Vision Transformers

**Authors**: *Jiahao Lu, Xi Sheryl Zhang, Tianli Zhao, Xiangyu He, Jian Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00981](https://doi.org/10.1109/CVPR52688.2022.00981)

**Abstract**:

Federated learning frameworks typically require collaborators to share their local gradient updates of a common model instead of sharing training data to preserve privacy. However, prior works on Gradient Leakage Attacks showed that private training data can be revealed from gradients. So far almost all relevant works base their attacks on fully-connected or convolutional neural networks. Given the recent overwhelmingly rising trend of adapting Transformers to solve multifarious vision tasks, it is highly valuable to investigate the privacy risk of vision transformers. In this paper, we analyse the gradient leakage risk of self-attention based mechanism in both theoretical and practical manners. Particularly, we propose APRIL - Attention PRIvacy Leakage, which poses a strong threat to self-attention inspired models such as ViT. Showing how vision Transformers are at the risk of privacy leakage via gradients, we urge the significance of designing privacy-safer Transformer models and defending schemes.

----

## [973] Rethinking Architecture Design for Tackling Data Heterogeneity in Federated Learning

**Authors**: *Liangqiong Qu, Yuyin Zhou, Paul Pu Liang, Yingda Xia, Feifei Wang, Ehsan Adeli, Li Fei-Fei, Daniel L. Rubin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00982](https://doi.org/10.1109/CVPR52688.2022.00982)

**Abstract**:

Federated learning is an emerging research paradigm enabling collaborative training of machine learning models among different organizations while keeping data private at each institution. Despite recent progress, there remain fundamental challenges such as the lack of convergence and the potential for catastrophic forgetting across real-world heterogeneous devices. In this paper, we demonstrate that self-attention-based architectures (e.g., Transformers) are more robust to distribution shifts and hence improve federated learning over heterogeneous data. Concretely, we conduct the first rigorous empirical investigation of different neural architectures across a range of federated algorithms, real-world benchmarks, and heterogeneous data splits. Our experiments show that simply replacing convolutional networks with Transformers can greatly reduce catastrophic forgetting of previous devices, accelerate convergence, and reach a better global model, especially when dealing with heterogeneous data. We release our code and pretrained models to encourage future exploration in robust architectures as an alternative to current research efforts on the optimization front.

----

## [974] Robust Federated Learning with Noisy and Heterogeneous Clients

**Authors**: *Xiuwen Fang, Mang Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00983](https://doi.org/10.1109/CVPR52688.2022.00983)

**Abstract**:

Model heterogeneous federated learning is a challenging task since each client independently designs its own model. Due to the annotation difficulty and free-riding par-ticipant issue, the local client usually contains unavoidable and varying noises, which cannot be effectively addressed by existing algorithms. This paper starts the first attempt to study a new and challenging robust federated learning problem with noisy and heterogeneous clients. We present a novel solution RHFL (Robust Heterogeneous Federated Learning), which simultaneously handles the label noise and performs federated learning in a single framework. It is featured in three aspects: (1) For the communication be-tween heterogeneous models, we directly align the models feedback by utilizing public data, which does not require additional shared global models for collaboration. (2) For internal label noise, we apply a robust noise-tolerant loss function to reduce the negative effects. (3) For challenging noisy feedback from other participants, we design a novel client confidence re-weighting scheme, which adaptively as-signs corresponding weights to each client in the collabo-rative learning stage. Extensive experiments validate the effectiveness of our approach in reducing the negative ef-fects of different noise rates/types under both model ho-mogeneous and heterogeneous federated learning settings, consistently outperforming existing methods.

----

## [975] Federated Learning with Position-Aware Neurons

**Authors**: *Xin-Chun Li, Yichu Xu, Shaoming Song, Bingshuai Li, Yinchuan Li, Yunfeng Shao, De-Chuan Zhan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00984](https://doi.org/10.1109/CVPR52688.2022.00984)

**Abstract**:

Federated Learning (FL) fuses collaborative models from local nodes without centralizing users' data. The permutation invariance property of neural networks and the non-i.i.d. data across clients make the locally updated parameters imprecisely aligned, disabling the coordinate-based parameter averaging. Traditional neurons do not explicitly consider position information. Hence, we propose Position-Aware Neurons (PANs) as an alternative, fusing position-related values (i.e., position encodings) into neuron outputs. PANs couple themselves to their positions and minimize the possibility of dislocation, even updating on heterogeneous data. We turn on/off PANs to disable/enable the permutation invariance property of neural networks. PANs are tightly coupled with positions when applied to FL, making parameters across clients pre-aligned and facilitating coordinate-based parameter averaging. PANs are algorithm-agnostic and could universally improve existing FL algorithms. Furthermore, “FL with PANs” is simple to implement and computationally friendly.

----

## [976] Layer-wised Model Aggregation for Personalized Federated Learning

**Authors**: *Xiaosong Ma, Jie Zhang, Song Guo, Wenchao Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00985](https://doi.org/10.1109/CVPR52688.2022.00985)

**Abstract**:

Personalized Federated Learning (pFL) not only can capture the common priors from broad range of distributed data, but also support customized models for heterogeneous clients. Researches over the past few years have applied the weighted aggregation manner to produce personalized models, where the weights are determined by calibrating the distance of the entire model parameters or loss values, and have yet to consider the layer-level impacts to the aggregation process, leading to lagged model convergence and inadequate personalization over non-IID datasets. In this paper, we propose a novel pFL training framework dubbed Layer-wised Personalized Federated learning (pFedLA) that can discern the importance of each layer from different clients, and thus is able to optimize the personalized model aggregation for clients with heterogeneous data. Specifically, we employ a dedicated hyper-network per client on the server side, which is trained to identify the mutual contribution factors at layer granularity. Meanwhile, a parameterized mechanism is introduced to update the layer-wised aggregation weights to progressively exploit the inter-user similarity and realize accurate model personalization. Extensive experiments are conducted over different models and learning tasks, and we show that the proposed methods achieve significantly higher performance than state-of-the-art pFL methods.

----

## [977] FedCor: Correlation-Based Active Client Selection Strategy for Heterogeneous Federated Learning

**Authors**: *Minxue Tang, Xuefei Ning, Yitu Wang, Jingwei Sun, Yu Wang, Hai Helen Li, Yiran Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00986](https://doi.org/10.1109/CVPR52688.2022.00986)

**Abstract**:

Client-wise data heterogeneity is one of the major issues that hinder effective training in federated learning (FL). Since the data distribution on each client may vary dramatically, the client selection strategy can significantly influence the convergence rate of the FL process. Active client selection strategies are popularly proposed in recent studies. However, they neglect the loss correlations between the clients and achieve only marginal improvement compared to the uniform selection strategy. In this work, we propose FedCoran FLframework built on a correlation-based client selection strategy, to boost the convergence rate of FL. Specifically, we first model the loss correlations between the clients with a Gaussian Process (GP). Based on the GP model, we derive a client selection strategy with a significant reduction of expected global loss in each round. Besides, we develop an efficient GP training method with a low communication overhead in the FL scenario by utilizing the covariance stationarity. Our experimental results show that compared to the state-of-the-art method, FedCorr can improve the convergence rates by 34% ~ 99% and 26% ~ 51% on FMNIST and CIFAR-10, respectively.

----

## [978] FedDC: Federated Learning with Non-IID Data via Local Drift Decoupling and Correction

**Authors**: *Liang Gao, Huazhu Fu, Li Li, Yingwen Chen, Ming Xu, Cheng-Zhong Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00987](https://doi.org/10.1109/CVPR52688.2022.00987)

**Abstract**:

Federated learning (FL) allows multiple clients to collectively train a high-performance global model without sharing their private data. However, the key challenge in federated learning is that the clients have significant statistical heterogeneity among their local data distributions, which would cause inconsistent optimized local models on the clientside. To address this fundamental dilemma, we propose a novel federated learning algorithm with local drift decoupling and correction (FedDC). Our FedDC only introduces lightweight modifications in the local training phase, in which each client utilizes an auxiliary local drift variable to track the gap between the local model parameter and the global model parameters. The key idea of FedDC is to utilize this learned local drift variable to bridge the gap, i.e., conducting consistency in parameter-level. The experiment results and analysis demonstrate that FedDC yields expediting convergence and better performance on various image classification tasks, robust in partial participation settings, non-iid data, and heterogeneous clients.

----

## [979] Differentially Private Federated Learning with Local Regularization and Sparsification

**Authors**: *Anda Cheng, Peisong Wang, Xi Sheryl Zhang, Jian Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00988](https://doi.org/10.1109/CVPR52688.2022.00988)

**Abstract**:

User-level differential privacy (DP) provides certifiable privacy guarantees to the information that is specific to any user's data in federated learning. Existing methods that ensure user-level DP come at the cost of severe accuracy decrease. In this paper, we study the cause of model performance degradation in federated learning with user-level DP guarantee. We find the key to solving this issue is to naturally restrict the norm of local updates before ex-ecuting operations that guarantee DP. To this end, we propose two techniques, Bounded Local Update Regularization and Local Update Sparsification, to increase model quality without sacrificing privacy. We provide theoretical analysis on the convergence of our framework and give rigorous privacy guarantees. Extensive experiments show that our framework significantly improves the privacy-utility trade-off over the state-of-the-arts for federated learning with user-level DP guarantee.

----

## [980] Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage

**Authors**: *Zhuohang Li, Jiaxin Zhang, Luyang Liu, Jian Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00989](https://doi.org/10.1109/CVPR52688.2022.00989)

**Abstract**:

Federated Learning (FL) framework brings privacy benefits to distributed learning systems by allowing multiple clients to participate in a learning task under the coordination of a central server without exchanging their private data. However, recent studies have revealed that private information can still be leaked through shared gradient information. To further protect user's privacy, several defense mechanisms have been proposed to prevent privacy leakage via gradient information degradation methods, such as using additive noise or gradient compression before sharing it with the server. In this work, we validate that the private training data can still be leaked under certain defense settings with a new type of leakage, i.e., Generative Gradient Leakage (GGL). Unlike existing methods that only rely on gradient information to reconstruct data, our method leverages the latent space of generative adversarial networks (GAN) learned from public image datasets as a prior to compensate for the informational loss during gradient degradation. To address the nonlinearity caused by the gradient operator and the GAN model, we explore various gradient-free optimization methods (e.g., evolution strategies and Bayesian optimization) and empirically show their superiority in reconstructing high-quality images from gradients compared to gradient-based optimizers. We hope the proposed method can serve as a tool for empirically measuring the amount of privacy leakage to facilitate the design of more robust defense mechanisms.

----

## [981] Learn from Others and Be Yourself in Heterogeneous Federated Learning

**Authors**: *Wenke Huang, Mang Ye, Bo Du*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00990](https://doi.org/10.1109/CVPR52688.2022.00990)

**Abstract**:

Federated learning has emerged as an important distributed learning paradigm, which normally involves collaborative updating with others and local updating on private data. However, heterogeneity problem and catastrophic forgetting bring distinctive challenges. First, due to non-i.i.d (identically and independently distributed) data and heterogeneous architectures, models suffer performance degradation on other domains and communication barrier with participants models. Second, in local updating, model is separately optimized on private data, which is prone to overfit current data distribution and forgets previously acquired knowledge, resulting in catastrophic forgetting. In this work, we propose FCCL (Federated CrossCorrelation and Continual Learning). For heterogeneity problem, FCCL leverages unlabeled public data for communication and construct cross-correlation matrix to learn a generalizable representation under domain shift. Mean- while, for catastrophic forgetting, FCCL utilizes knowledge distillation in local updating, providing inter and intra domain information without leaking privacy. Empirical results on various image classification tasks demonstrate the effectiveness of our method and the efficiency of modules.

----

## [982] RSCFed: Random Sampling Consensus Federated Semi-supervised Learning

**Authors**: *Xiaoxiao Liang, Yiqun Lin, Huazhu Fu, Lei Zhu, Xiaomeng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00991](https://doi.org/10.1109/CVPR52688.2022.00991)

**Abstract**:

Federated semi-supervised learning (FSSL) aims to derive a global model by training fully-labeled and fully-unlabeled clients or training partially labeled clients. The existing approaches work well when local clients have in-dependent and identically distributed (IID) data but fail to generalize to a more practical FSSL setting, i.e., Non-IID setting. In this paper, we present a Random Sampling Consensus Federated learning, namely RSCFed, by con-sidering the uneven reliability among models from fully-labeled clients, fully-unlabeled clients or partially labeled clients. Our key motivation is that given models with large deviations from either labeled clients or unlabeled clients, the consensus could be reached by performing random sub-sampling over clients. To achieve it, instead of di-rectly aggregating local models, we first distill several sub-consensus models by random sub-sampling over clients and then aggregating the sub-consensus models to the global model. To enhance the robustness of sub-consensus models, we also develop a novel distance-reweighted model aggre-gation method. Experimental results show that our method outperforms state-of-the-art methods on three benchmarked datasets, including both natural and medical images. The code is available at https://github.com/XMed-Lab/RSCFed.

----

## [983] Federated Class-Incremental Learning

**Authors**: *Jiahua Dong, Lixu Wang, Zhen Fang, Gan Sun, Shichao Xu, Xiao Wang, Qi Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00992](https://doi.org/10.1109/CVPR52688.2022.00992)

**Abstract**:

Federated learning (FL) has attracted growing attentions via data-private collaborative training on decentralized clients. However, most existing methods unrealistically assume object classes of the overall framework are fixed over time. It makes the global model suffer from significant catastrophic forgetting on old classes in real-world scenarios, where local clients often collect new classes continuously and have very limited storage memory to store old classes. Moreover, new clients with unseen new classes may participate in the FL training, further aggravating the catastrophic forgetting of global model. To address these challenges, we develop a novel Global-Local Forgetting Compensation (GLFC) model, to learn a global class-incremental model for alleviating the catastrophic forgetting from both local and global perspectives. Specifically, to address local forgetting caused by class imbalance at the local clients, we design a class-aware gradient compensation loss and a class-semantic relation distillation loss to balance the forgetting of old classes and distill consistent inter-class relations across tasks. To tackle the global forgetting brought by the non-i.i.d class imbalance across clients, we propose a proxy server that selects the best old global model to assist the local relation distillation. Moreover, a prototype gradient-based communication mechanism is developed to protect the privacy. Our model outperforms state-of-the-art methods by 4.4%~15.1% in terms of average accuracy on representative benchmark datasets. The code is available at https://github.com/conditionWang/FCIL.

----

## [984] Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning

**Authors**: *Lin Zhang, Li Shen, Liang Ding, Dacheng Tao, Ling-Yu Duan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00993](https://doi.org/10.1109/CVPR52688.2022.00993)

**Abstract**:

Federated Learning (FL) is an emerging distributed learning paradigm under privacy constraint. Data heterogeneity is one of the main challenges in FL, which results in slow convergence and degraded performance. Most existing approaches only tackle the heterogeneity challenge by restricting the local model update in client, ignoring the performance drop caused by direct global model aggregation. Instead, we propose a data-free knowledge distillation method to fine-tune the global model in the server (FedFTG), which relieves the issue of direct model aggregation. Concretely, FedFTG explores the input space of local models through a generator, and uses it to transfer the knowledge from local models to the global model. Besides, we propose a hard sample mining scheme to achieve effective knowledge distillation throughout the training. In addition, we develop customized label sampling and class-level ensemble to derive maximum utilization of knowledge, which implicitly mitigates the distribution discrepancy across clients. Extensive experiments show that our FedFTG significantly outperforms the state-of-the-art (SOTA) FL algorithms and can serve as a strong plugin for enhancing FedAvg, FedProx, FedDyn, and SCAFFOLD.

----

## [985] FedCorr: Multi-Stage Federated Learning for Label Noise Correction

**Authors**: *Jingyi Xu, Zihan Chen, Tony Q. S. Quek, Kai Fong Ernest Chong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00994](https://doi.org/10.1109/CVPR52688.2022.00994)

**Abstract**:

Federated learning (FL) is a privacy-preserving distributed learning paradigm that enables clients to jointly train a global model. In real-world FL implementations, client data could have label noise, and different clients could have vastly different label noise levels. Although there exist methods in centralized learning for tackling label noise, such methods do not perform well on heterogeneous label noise in FL settings, due to the typically smaller sizes of client datasets and data privacy requirements in FL. In this paper, we propose FedCorr, a general multi-stage framework to tackle heterogeneous label noise in FL, without making any assumptions on the noise models of local clients, while still maintaining client data privacy. In particular, (1) FedCorr dynamically identifies noisy clients by exploiting the dimensionalities of the model prediction subspaces independently measured on all clients, and then identifies incorrect labels on noisy clients based on persample losses. To deal with data heterogeneity and to increase training stability, we propose an adaptive local proximal regularization term that is based on estimated local noise levels. (2) We further finetune the global model on identified clean clients and correct the noisy labels for the remaining noisy clients after finetuning. (3) Finally, we apply the usual training on all clients to make full use of all local data. Experiments conducted on CIFAR-10/100 with federated synthetic label noise, and on a real-world noisy dataset, Clothing1M, demonstrate that FedCorr is robust to label noise and substantially outperforms the state-of-the-art methods at multiple noise levels.

----

## [986] ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning

**Authors**: *Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, Chaitali Chakrabarti*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00995](https://doi.org/10.1109/CVPR52688.2022.00995)

**Abstract**:

This work aims to tackle Model Inversion (MI) attack on Split Federated Learning (SFL). SFL is a recent distributed training scheme where multiple clients send intermediate activations (i. e., feature map), instead of raw data, to a central server. While such a scheme helps reduce the computational load at the client end, it opens itself to reconstruction of raw data from intermediate activation by the server. Existing works on protecting SFL only consider inference and do not handle attacks during training. So we propose ResSFL, a Split Federated Learning Framework that is designed to be MI-resistant during training. It is based on deriving a resistant feature extractor via attacker-aware training, and using this extractor to initialize the client-side model prior to standard SFL training. Such a method helps in reducing the computational complexity due to use of strong inversion model in client-side adversarial training as well as vulnerability of attacks launched in early training epochs. On CIFAR-100 dataset, our proposed framework successfully mitigates MI attack on a VGG-11 model with a high reconstruction Mean-Square-Error of 0.050 compared to 0.005 obtained by the baseline system. The frame-work achieves 67.5% accuracy (only 1 % accuracy drop) with very low computation overhead. Code is released at: https://github.com/zlijingtao/ResSFL.

----

## [987] Cycle-Consistent Counterfactuals by Latent Transformations

**Authors**: *Saeed Khorram, Fuxin Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00996](https://doi.org/10.1109/CVPR52688.2022.00996)

**Abstract**:

CounterFactual (CF) visual explanations try to find images similar to the query image that change the decision of a vision system to a specified outcome. Existing methods either require inference-time optimization or joint training with a generative adversarial model which makes them time-consuming and difficult to use in practice. We propose a novel approach, Cycle-Consistent Counterfactuals by Latent Transformations (C3LT), which learns a latent transformation that automatically generates visual CFs by steering in the latent space of generative models. Our method uses cycle consistency between the query and CF latent representations which helps our training to find better solutions. C3LT can be easily plugged into any state-of-the-art pretrained generative network. This enables our method to generate high-quality and interpretable CF images at high resolution such as those in ImageNet. In addition to several established metrics for evaluating CF explanations, we introduce a novel metric tailored to assess the quality of the generated CF examples and validate the effectiveness of our method on an extensive set of experiments.

----

## [988] Consistent Explanations by Contrastive Learning

**Authors**: *Vipin Pillai, Soroush Abbasi Koohpayegani, Ashley Ouligian, Dennis Fong, Hamed Pirsiavash*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00997](https://doi.org/10.1109/CVPR52688.2022.00997)

**Abstract**:

Post-hoc explanation methods, e.g., Grad-CAM, enable humans to inspect the spatial regions responsible for a particular network decision. However, it is shown that such explanations are not always consistent with human priors, such as consistency across image transformations. Given an interpretation algorithm, e.g., Grad-CAM, we introduce a novel training method to train the model to produce more consistent explanations. Since obtaining the ground truth for a desired model interpretation is not a well-defined task, we adopt ideas from contrastive self-supervised learning, and apply them to the interpretations of the model rather than its embeddings. We show that our method, Contrastive Grad-CAM Consistency (CGC), results in Grad-CAM interpretation heatmaps that are more consistent with human annotations while still achieving comparable classification accuracy. Moreover, our method acts as a regularizer and improves the accuracy on limited-data, fine-grained classification settings. In addition, because our method does not rely on annotations, it allows for the incorporation of unlabeled data into training, which enables better generalization of the model. The code is available here: https://github.com/UCDvision/CGC

----

## [989] Towards Better Understanding Attribution Methods

**Authors**: *Sukrut Rao, Moritz Böhle, Bernt Schiele*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00998](https://doi.org/10.1109/CVPR52688.2022.00998)

**Abstract**:

Deep neural networks are very successful on many vision tasks, but hard to interpret due to their black box nature. To overcome this, various post-hoc attribution methods have been proposed to identify image regions most influential to the models' decisions. Evaluating such methods is challenging since no ground truth attributions exist. We thus propose three novel evaluation schemes to more reliably measure the faithfulness of those methods, to make comparisons between them more fair, and to make visual inspection more systematic. To address faithfulness, we propose a novel evaluation setting (DiFull) in which we carefully control which parts of the input can influence the output in order to distinguish possible from impossible attributions. To address fairness, we note that different methods are applied at different layers, which skews any comparison, and so evaluate all methods on the same layers (ML-Att) and discuss how this impacts their performance on quantitative metrics. For more systematic visualizations, we propose a scheme (AggAtt) to qualitatively evaluate the methods on complete datasets. We use these evaluation schemes to study strengths and shortcomings of some widely used attribution methods. Finally, we propose a post-processing smoothing step that significantly improves the performance of some attribution methods, and discuss its applicability.

----

## [990] Proto2Proto: Can you recognize the car, the way I do?

**Authors**: *Monish Keswani, Sriranjani Ramakrishnan, Nishant Reddy, Vineeth N. Balasubramanian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00999](https://doi.org/10.1109/CVPR52688.2022.00999)

**Abstract**:

Prototypical methods have recently gained a lot of attention due to their intrinsic interpretable nature, which is obtained through the prototypes. With growing use cases of model reuse and distillation, there is a need to also study transfer of interpretability from one model to another. We present Proto2Proto, a novel method to transfer interpretability of one prototypical part network to another via knowledge distillation. Our approach aims to add interpretability to the “dark” knowledge transferred from the teacher to the shallower student model. We propose two novel losses: “Global Explanation” loss and “Patch-Prototype Correspondence” loss to facilitate such a transfer. Global Explanation loss forces the student prototypes to be close to teacher prototypes, and Patch-Prototype Cor-respondence loss enforces the local representations of the student to be similar to that of the teacher. Further, we propose three novel metrics to evaluate the student's proximity to the teacher as measures of interpretability transfer in our settings. We qualitatively and quantitatively demon-strate the effectiveness of our method on CUB-200-2011 and Stanford Cars datasets. Our experiments show that the proposed method indeed achieves interpretability transfer from teacher to student while simultaneously exhibiting competitive performance. The code is available at h t tps: //github.com/archmaester/proto2proto

----

## [991] Do Explanations Explain? Model Knows Best

**Authors**: *Ashkan Khakzar, Pedram Khorsandi, Rozhin Nobahari, Nassir Navab*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01000](https://doi.org/10.1109/CVPR52688.2022.01000)

**Abstract**:

It is a mystery which input features contribute to a neural network's output. Various explanation (feature attribution) methods are proposed in the literature to shed light on the problem. One peculiar observation is that these explanations (attributions) point to different features as being important. The phenomenon raises the question, which explanation to trust? We propose a framework for evaluating the explanations using the neural network model itself. The framework leverages the network to generate input features that impose a particular behavior on the output. Using the generated features, we devise controlled experimental setups to evaluate whether an explanation method conforms to an axiom. Thus we propose an empirical framework for axiomatic evaluation of explanation methods. We evaluate well-known and promising explanation solutions using the proposed framework. The framework provides a toolset to reveal properties and drawbacks within existing and future explanation solutions.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/CAMP-eXplain-AI/Do-Explanations-Explain

----

## [992] HINT: Hierarchical Neuron Concept Explainer

**Authors**: *Andong Wang, Wei-Ning Lee, Xiaojuan Qi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01001](https://doi.org/10.1109/CVPR52688.2022.01001)

**Abstract**:

To interpret deep networks, one main approach is to associate neurons with human-understandable concepts. However, existing methods often ignore the inherent connections of different concepts (e.g., dog and cat both belong to animals), and thus lose the chance to explain neurons responsible for higher-level concepts (e.g., animal). In this paper, we study hierarchical concepts inspired by the hierarchical cognition process of human beings. To this end, we propose HIerarchical Neuron concepT explainer (HINT) to effectively build bidirectional associations between neurons and hierarchical concepts in a low-cost and scalable manner. HINT enables us to systematically and quantitatively study whether and how the implicit hierarchical relationships of concepts are embedded into neurons. Specifically, HINT identifies collaborative neurons responsible for one concept and multimodal neurons pertinent to different concepts, at different semantic levels from concrete concepts (e.g., dog) to more abstract ones (e.g., animal). Finally, we verify the faithfulness of the associations using Weakly Supervised Object Localization, and demonstrate its applicability in various tasks, such as discovering saliency regions and explaining adversarial attacks. Code is available on https://github.com/AntonotnaWang/HINT.

----

## [993] Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes

**Authors**: *Jon Donnelly, Alina Jade Barnett, Chaofan Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01002](https://doi.org/10.1109/CVPR52688.2022.01002)

**Abstract**:

We present a deformable prototypical part network (Deformable ProtoPNet), an interpretable image classifier that integrates the power of deep learning and the interpretability of case-based reasoning. This model classifies input images by comparing them with prototypes learned during training, yielding explanations in the form of “this looks like that.” However, while previous methods use spatially rigid prototypes, we address this shortcoming by proposing spatially flexible prototypes. Each prototype is made up of several prototypical parts that adaptively change their relative spatial positions depending on the input image. Consequently, a Deformable ProtoPNet can explicitly capture pose variations and context, improving both model accuracy and the richness of explanations provided. Compared to other case-based interpretable models using prototypes, our approach achieves state-of-the-art accuracy and gives an explanation with greater context. The code is available at https://github.com/jdonnelly36IDeformable-ProtoPNet.

----

## [994] What do navigation agents learn about their environment?

**Authors**: *Kshitij Dwivedi, Gemma Roig, Aniruddha Kembhavi, Roozbeh Mottaghi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01003](https://doi.org/10.1109/CVPR52688.2022.01003)

**Abstract**:

Today's state of the art visual navigation agents typically consist of large deep learning models trained end to end. Such models offer little to no interpretability about the learned skills or the actions of the agent taken in response to its environment. While past works have explored interpreting deep learning models, little attention has been devoted to interpreting embodied AI systems, which often involve reasoning about the structure of the environment, target characteristics and the outcome of one's actions. In this paper, we introduce the Interpretability System for Embodied agEnts (iSEE) for Point Goal and Object Goal navigation agents. We use iS
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">EE</inf>
 to probe the dynamic representations produced by these agents for the presence of information about the agent as well as the environment. We demonstrate interesting insights about navigation agents using iSEE, including the ability to encode reachable locations (to avoid obstacles), visibility of the target, progress from the initial spawn location as well as the dramatic effect on the behaviors of agents when we mask out critical individual neurons.

----

## [995] A Framework for Learning Ante-hoc Explainable Models via Concepts

**Authors**: *Anirban Sarkar, Deepak Vijaykeerthy, Anindya Sarkar, Vineeth N. Balasubramanian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01004](https://doi.org/10.1109/CVPR52688.2022.01004)

**Abstract**:

Self-explaining deep models are designed to learn the latent concept-based explanations implicitly during training, which eliminates the requirement of any post-hoc explanation generation technique. In this work, we propose one such model that appends an explanation generation module on top of any basic network and jointly trains the whole module that shows high predictive performance and generates meaningful explanations in terms of concepts. Our training strategy is suitable for unsupervised concept learning with much lesser parameter space requirements compared to baseline methods. Our proposed model also has provision for leveraging self-supervision on concepts to extract better explanations. However, with full concept supervision, we achieve the best predictive performance compared to recently proposed concept-based explainable models. We report both qualitative and quantitative results with our method, which shows better performance than recently proposed concept-based explainability methods. We reported exhaustive results with two datasets without ground truth concepts, i.e., CIFAR10, ImageNet, and two datasets with ground truth concepts, i.e., AwA2, CUB-200, to show the effectiveness of our method for both cases. To the best of our knowledge, we are the first ante-hoc explanation generation method to show results with a large-scale dataset such as ImageNet.

----

## [996] Exploiting Explainable Metrics for Augmented SGD

**Authors**: *Mahdi S. Hosseini, Mathieu Tuli, Konstantinos N. Plataniotis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01005](https://doi.org/10.1109/CVPR52688.2022.01005)

**Abstract**:

Explaining the generalization characteristics of deep learning is an emerging topic in advanced machine learning. There are several unanswered questions about how learning under stochastic optimization really works and why certain strategies are better than others. In this paper, we address the following question: can we probe intermediate layers of a deep neural network to identify and quantify the learning quality of each layer? With this question in mind, we propose new explainability metrics that measure the redundant information in a network's layers using a low-rank factorization framework and quantify a complexity measure that is highly correlated with the generalization performance of a given optimizer, network, and dataset. We subsequently exploit these metrics to augment the Stochastic Gradient Descent (SGD) optimizer by adaptively adjusting the learning rate in each layer to improve in generalization performance. Our augmented SGD - dubbed RMSGD - introduces minimal computational overhead compared to SOTA methods and outperforms them by exhibiting strong generalization characteristics across application, architecture, and dataset.

----

## [997] FAM: Visual Explanations for the Feature Representations from Deep Convolutional Networks

**Authors**: *Yuxi Wu, Changhuai Chen, Jun Che, Shiliang Pu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01006](https://doi.org/10.1109/CVPR52688.2022.01006)

**Abstract**:

In recent years, increasing attention has been drawn to the internal mechanisms of representation models. Traditional methods are inapplicable to fully explain the feature representations, especially if the images do not fit into any category. In this case, employing an existing class or the similarity with other image is unable to provide a complete and reliable visual explanation. To handle this task, we propose a novel visual explanation paradigm called Fea-ture Activation Mapping (FAM) in this paper. Under this paradigm, Grad-FAM and Score-FAM are designed for vi-sualizing feature representations. Unlike the previous approaches, FAM locates the regions of images that contribute most to the feature vector itself. Extensive experiments and evaluations, both subjective and objective, showed that Score-FAM provided most promising interpretable vi-sual explanations for feature representations in Person Re-Identification. Furthermore, FAM also can be employed to analyze other vision tasks, such as self-supervised represen-tation learning.

----

## [998] Interactive Disentanglement: Learning Concepts by Interacting with their Prototype Representations

**Authors**: *Wolfgang Stammer, Marius Memmel, Patrick Schramowski, Kristian Kersting*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01007](https://doi.org/10.1109/CVPR52688.2022.01007)

**Abstract**:

Learning visual concepts from raw images without strong supervision is a challenging task. In this work, we show the advantages of prototype representations for understanding and revising the latent space of neural concept learners. For this purpose, we introduce interactive Concept Swapping Networks (iCSNs), a novel framework for learning concept-grounded representations via weak supervision and implicit prototype representations. iCSNs learn to bind conceptual information to specific prototype slots by swapping the latent representations of paired images. This semantically grounded and discrete latent space facilitates human understanding and human-machine interaction. We support this claim by conducting experiments on our novel data set “Elementary Concept Reasoning” (ECR), focusing on visual concepts shared by geometric objects.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at: https://github.com/ml-research/ XIConceptLearning

----

## [999] B-cos Networks: Alignment is All We Need for Interpretability

**Authors**: *Moritz Böhle, Mario Fritz, Bernt Schiele*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01008](https://doi.org/10.1109/CVPR52688.2022.01008)

**Abstract**:

We present a new direction for increasing the interpretability of deep neural networks (DNNs) by promoting weight-input alignment during training. For this, we propose to replace the linear transforms in DNNs by our B-cos transform. As we show, a sequence (network) of such transforms induces a single linear transform that faithfully summarises the full model computations. Moreover, the B-cos transform introduces alignment pressure on the weights during optimisation. As a result, those induced linear transforms become highly interpretable and align with task-relevant features. Importantly, the B-cos transform is designed to be compatible with existing architectures and we show that it can easily be integrated into common models such as VGGs, ResNets, InceptionNets, and DenseNets, whilst maintaining similar performance on ImageNet. The resulting explanations are of high visual quality and perform well under quantitative metrics for interpretability. Code available at github.com/moboehle/B-cos.

----



[Go to the previous page](CVPR-2022-list04.md)

[Go to the next page](CVPR-2022-list06.md)

[Go to the catalog section](README.md)