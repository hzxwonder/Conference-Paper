## [200] Visual Sound Localization in the Wild by Cross-Modal Interference Erasing

**Authors**: *Xian Liu, Rui Qian, Hang Zhou, Di Hu, Weiyao Lin, Ziwei Liu, Bolei Zhou, Xiaowei Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20073](https://doi.org/10.1609/aaai.v36i2.20073)

**Abstract**:

The task of audiovisual sound source localization has been well studied under constrained scenes, where the audio recordings are clean. However, in real world scenarios, audios are usually contaminated by off screen sound and background noise. They will interfere with the procedure of identifying desired sources and building visual sound connections, making previous studies nonapplicable. In this work, we propose the Interference Eraser (IEr) framework, which tackles the problem of audiovisual sound source localization in the wild. The key idea is to eliminate the interference by redefining and carving discriminative audio representations. Specifically, we observe that the previous practice of learning only a single audio representation is insufficient due to the additive nature of audio signals. We thus extend the audio representation with our Audio Instance Identifier module, which clearly distinguishes sounding instances when audio signals of different volumes are unevenly mixed. Then we erase the influence of the audible but off screen sounds and the silent but visible objects by a Cross modal Referrer module with cross modality distillation. Quantitative and qualitative evaluations demonstrate that our framework achieves superior results on sound localization tasks, especially under real world scenarios.

----

## [201] Learning Auxiliary Monocular Contexts Helps Monocular 3D Object Detection

**Authors**: *Xianpeng Liu, Nan Xue, Tianfu Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20074](https://doi.org/10.1609/aaai.v36i2.20074)

**Abstract**:

Monocular 3D object detection aims to localize 3D bounding boxes in an input single 2D image. It is a highly challenging problem and remains open, especially when no extra information (e.g., depth, lidar and/or multi-frames) can be leveraged in training and/or inference.  This paper proposes a simple yet effective formulation for monocular 3D object detection without exploiting any extra information. It presents the MonoCon method which learns Monocular Contexts, as auxiliary tasks in training, to help monocular 3D object detection. The key idea is that with the annotated 3D bounding boxes of objects in an image, there is a rich set of well-posed projected 2D supervision signals available in training, such as the projected corner keypoints and their associated offset vectors with respect to the center of 2D bounding box, which should be exploited as auxiliary tasks in training. 
The proposed MonoCon is motivated by the Cramer–Wold theorem in measure theory at a high level.  In implementation, it utilizes a very simple end-to-end design to justify the effectiveness of learning auxiliary monocular contexts, which  consists of three components: a Deep Neural Network (DNN) based feature backbone, a number of regression head branches for learning the essential parameters used in the 3D bounding box prediction, and a number of regression head branches for learning auxiliary contexts. After training, the auxiliary context regression branches are discarded for better inference efficiency. In experiments, the proposed MonoCon is tested in the KITTI benchmark (car, pedestrian and cyclist). It outperforms all prior arts in the leaderboard on the car category and obtains comparable performance on pedestrian and cyclist in terms of accuracy. Thanks to the simple design, the proposed MonoCon method obtains the fastest inference speed with 38.7 fps in comparisons. Our code is released at https://git.io/MonoCon.

----

## [202] Highlighting Object Category Immunity for the Generalization of Human-Object Interaction Detection

**Authors**: *Xinpeng Liu, Yong-Lu Li, Cewu Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20075](https://doi.org/10.1609/aaai.v36i2.20075)

**Abstract**:

Human-Object Interaction (HOI) detection plays a core role in activity understanding. As a compositional learning problem (human-verb-object), studying its generalization matters. However, widely-used metric mean average precision (mAP) fails to model the compositional generalization well. Thus, we propose a novel metric, mPD (mean Performance Degradation), as a complementary of mAP to evaluate the performance gap among compositions of different objects and the same verb. Surprisingly, mPD reveals that previous methods usually generalize poorly. With mPD as a cue, we propose Object Category (OC) Immunity to boost HOI generalization. The idea is to prevent model from learning spurious object-verb correlations as a short-cut to over-fit the train set. To achieve OC-immunity, we propose an OC-immune network that decouples the inputs from OC, extracts OC-immune representations, and leverages uncertainty quantification to generalize to unseen objects. In both conventional and zero-shot experiments, our method achieves decent improvements. To fully evaluate the generalization, we design a new and more difficult benchmark, on which we present significant advantage. The code is available at https://github.com/Foruck/OC-Immunity.

----

## [203] DMN4: Few-Shot Learning via Discriminative Mutual Nearest Neighbor Neural Network

**Authors**: *Yang Liu, Tu Zheng, Jie Song, Deng Cai, Xiaofei He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20076](https://doi.org/10.1609/aaai.v36i2.20076)

**Abstract**:

Few-shot learning (FSL) aims to classify images under low-data regimes, where the conventional pooled global feature is likely to lose useful local characteristics. Recent work has achieved promising performances by using deep descriptors. They generally take all deep descriptors from neural networks into consideration while ignoring that some of them are useless in classification due to their limited receptive field, e.g., task-irrelevant descriptors could be misleading and multiple aggregative descriptors from background clutter could even overwhelm the object's presence. In this paper, we argue that a Mutual Nearest Neighbor (MNN) relation should be established to explicitly select the query descriptors that are most relevant to each task and discard less relevant ones from aggregative clutters in FSL. Specifically, we propose Discriminative Mutual Nearest Neighbor Neural Network (DMN4) for FSL. Extensive experiments demonstrate that our method outperforms the existing state-of-the-arts on both fine-grained and generalized datasets.

----

## [204] Multi-Knowledge Aggregation and Transfer for Semantic Segmentation

**Authors**: *Yuang Liu, Wei Zhang, Jun Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20077](https://doi.org/10.1609/aaai.v36i2.20077)

**Abstract**:

As a popular deep neural networks (DNN) compression technique, knowledge distillation (KD) has attracted increasing attentions recently. Existing KD methods usually utilize one kind of knowledge in an intermediate layer of DNN for classification tasks to transfer useful information from cumbersome teacher networks to compact student networks. However, this paradigm is not very suitable for semantic segmentation, a comprehensive vision task based on both pixel-level and contextual information, since it cannot provide rich information for distillation. In this paper, we propose a novel multi-knowledge aggregation and transfer (MKAT) framework to comprehensively distill knowledge within an intermediate layer for semantic segmentation. Specifically, the proposed framework consists of three parts: Independent Transformers and Encoders module (ITE), Auxiliary Prediction Branch (APB), and Mutual Label Calibration (MLC) mechanism, which can take advantage of abundant knowledge from intermediate features. To demonstrate the effectiveness of our proposed approach, we conduct extensive experiments on three segmentation datasets: Pascal VOC, Cityscapes, and CamVid, showing that MKAT outperforms the other KD methods.

----

## [205] Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency

**Authors**: *Zhenhuan Liu, Liang Li, Huajie Jiang, Xin Jin, Dandan Tu, Shuhui Wang, Zheng-Jun Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20078](https://doi.org/10.1609/aaai.v36i2.20078)

**Abstract**:

In recent years, creative content generations like style transfer and neural photo editing have attracted more and more attention. Among these, cartoonization of real-world scenes has promising applications in entertainment and industry. Different from image translations focusing on improving the style effect of generated images, video cartoonization has additional requirements on the temporal consistency. In this paper, we propose a spatially-adaptive semantic alignment framework with perceptual motion consistency for coherent video cartoonization in an unsupervised manner. The semantic alignment module is designed to restore deformation of semantic structure caused by spatial information lost in the encoder-decoder architecture. Furthermore, we introduce the spatio-temporal correlative map as a style-independent, global-aware regularization on perceptual motion consistency. Deriving from similarity measurement of high-level features in photo and cartoon frames, it captures global semantic information beyond raw pixel-value of optical flow. Besides, the similarity measurement disentangles temporal relationship from domain-specific style properties, which helps regularize the temporal consistency without hurting style effects of cartoon images. Qualitative and quantitative experiments demonstrate our method is able to generate highly stylistic and temporal consistent cartoon videos.

----

## [206] Task-Customized Self-Supervised Pre-training with Scalable Dynamic Routing

**Authors**: *Zhili Liu, Jianhua Han, Lanqing Hong, Hang Xu, Kai Chen, Chunjing Xu, Zhenguo Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20079](https://doi.org/10.1609/aaai.v36i2.20079)

**Abstract**:

Self-supervised learning (SSL), especially contrastive methods, has raised attraction recently as it learns effective transferable representations without semantic annotations. A common practice for self-supervised pre-training is to use as much data as possible. For a specific downstream task, however, 
 involving irrelevant data in pre-training may degenerate the downstream performance, observed from our extensive experiments. On the other hand, for existing SSL methods, it is burdensome and infeasible to use different downstream-task-customized datasets in pre-training for different tasks.
 To address this issue, we propose a novel SSL paradigm called Scalable Dynamic Routing (SDR), which can be trained once and deployed efficiently to different downstream tasks with task-customized pre-trained models. Specifically, we construct the SDRnet with various sub-nets and train each sub-net with only one subset of the data by data-aware progressive training. When a downstream task arrives, we route among all the pre-trained sub-nets to get the best along with its corresponding weights. Experiment results show that our SDR can train 256 sub-nets on ImageNet simultaneously, which provides better transfer performance than a unified model trained on the full ImageNet, achieving state-of-the-art (SOTA) averaged accuracy over 11 downstream classification tasks and AP on PASCAL VOC detection task.

----

## [207] Pose Guided Image Generation from Misaligned Sources via Residual Flow Based Correction

**Authors**: *Jiawei Lu, He Wang, Tianjia Shao, Yin Yang, Kun Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20080](https://doi.org/10.1609/aaai.v36i2.20080)

**Abstract**:

Generating new images with desired properties (e.g. new view/poses) from source images has been enthusiastically pursued recently, due to its wide range of potential applications. One way to ensure high-quality generation is to use multiple sources with complementary information such as different views of the same object. However, as source images are often misaligned due to the large disparities among the camera settings, strong assumptions have been made in the past with respect to the camera(s) or/and the object in interest, limiting the application of such techniques. Therefore, we propose a new general approach which models multiple types of variations among sources, such as view angles, poses, facial expressions, in a unified framework, so that it can be employed on datasets of vastly different nature. We verify our approach on a variety of data including humans bodies, faces, city scenes and 3D objects. Both the qualitative and quantitative results demonstrate the better performance of our method than the state of the art.

----

## [208] PMAL: Open Set Recognition via Robust Prototype Mining

**Authors**: *Jing Lu, Yunlu Xu, Hao Li, Zhanzhan Cheng, Yi Niu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20081](https://doi.org/10.1609/aaai.v36i2.20081)

**Abstract**:

Open Set Recognition (OSR) has been an emerging topic. Besides recognizing predefined classes, the system needs to reject the unknowns. Prototype learning is a potential manner to handle the problem, as its ability to improve intra-class compactness of representations is much needed in discrimination between the known and the unknowns. In this work, we propose a novel Prototype Mining And Learning (PMAL) framework. It has a prototype mining mechanism before the phase of optimizing embedding space, explicitly considering two crucial properties, namely high-quality and diversity of the prototype set. Concretely, a set of high-quality candidates are firstly extracted from training samples based on data uncertainty learning, avoiding the interference from unexpected noise. Considering the multifarious appearance of objects even in a single category, a diversity-based strategy for prototype set filtering is proposed. Accordingly, the embedding space can be better optimized to discriminate therein the predefined classes and between known and unknowns. Extensive experiments verify the two good characteristics (i.e., high-quality and diversity) embraced in prototype mining, and show the remarkable performance of the proposed framework compared to state-of-the-arts.

----

## [209] Barely-Supervised Learning: Semi-supervised Learning with Very Few Labeled Images

**Authors**: *Thomas Lucas, Philippe Weinzaepfel, Grégory Rogez*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20082](https://doi.org/10.1609/aaai.v36i2.20082)

**Abstract**:

This paper tackles the problem of semi-supervised learning when the set of labeled samples is limited to a small number of images per class, typically less than 10, problem that we refer to as barely-supervised learning. We analyze in depth the behavior of a state-of-the-art semi-supervised method, FixMatch, which relies on a weakly-augmented version of an image to obtain supervision signal for a more strongly-augmented version. We show that it frequently fails in barely-supervised scenarios, due to a lack of training signal when no pseudo-label can be predicted with high confidence. We propose a method to leverage self-supervised methods that provides training signal in the absence of confident pseudo-labels. We then propose two methods to refine the pseudo-label selection process which lead to further improvements.The first one relies on a per-sample history of the model predictions, akin to a voting scheme. The second iteratively up-dates class-dependent confidence thresholds to better explore classes that are under-represented in the pseudo-labels. Our experiments show that our approach performs significantly better on STL-10 in the barely-supervised regime,e.g. with 4 or 8 labeled images per class.

----

## [210] Learning Optical Flow with Adaptive Graph Reasoning

**Authors**: *Ao Luo, Fan Yang, Kunming Luo, Xin Li, Haoqiang Fan, Shuaicheng Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20083](https://doi.org/10.1609/aaai.v36i2.20083)

**Abstract**:

Estimating per-pixel motion between video frames, known as optical flow, is a long-standing problem in video understanding and analysis. Most contemporary optical flow techniques largely focus on addressing the cross-image matching with feature similarity, with few methods considering how to explicitly reason over the given scene for achieving a holistic motion understanding. In this work, taking a fresh perspective, we introduce a novel graph-based approach, called adaptive graph reasoning for optical flow (AGFlow), to emphasize the value of scene/context information in optical flow. Our key idea is to decouple the context reasoning from the matching procedure, and exploit scene information to effectively assist motion estimation by learning to reason over the adaptive graph. The proposed AGFlow can effectively exploit the context information and incorporate it within the matching procedure, producing more robust and accurate results. On both Sintel clean and final passes, our AGFlow achieves the best accuracy with EPE of 1.43 and 2.47 pixels, outperforming state-of-the-art approaches by 11.2% and 13.6%, respectively. Code is publicly available at https://github.com/megvii-research/AGFlow.

----

## [211] A Fusion-Denoising Attack on InstaHide with Data Augmentation

**Authors**: *Xinjian Luo, Xiaokui Xiao, Yuncheng Wu, Juncheng Liu, Beng Chin Ooi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20084](https://doi.org/10.1609/aaai.v36i2.20084)

**Abstract**:

InstaHide is a state-of-the-art mechanism for protecting private training images, by mixing multiple private images and modifying them such that their visual features are indistinguishable to the naked eye. In recent work, however, Carlini et al. show that it is possible to reconstruct private images from the encrypted dataset generated by InstaHide. Nevertheless, we demonstrate that Carlini et al.’s attack can be easily defeated by incorporating data augmentation into InstaHide. This leads to a natural question: is InstaHide with data augmentation secure? In this paper, we provide a negative answer to this question, by devising an attack for recovering private images from the outputs of InstaHide even when data augmentation is present. The basic idea is to use a comparative network to identify encrypted images that are likely to correspond to the same private image, and then employ a fusion-denoising network for restoring the private image from the encrypted ones, taking into account the effects of data augmentation. Extensive experiments demonstrate the effectiveness of the proposed attack in comparison to Carlini et al.’s attack.

----

## [212] Deep Neural Networks Learn Meta-Structures from Noisy Labels in Semantic Segmentation

**Authors**: *Yaoru Luo, Guole Liu, Yuanhao Guo, Ge Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20085](https://doi.org/10.1609/aaai.v36i2.20085)

**Abstract**:

How deep neural networks (DNNs) learn from noisy labels has been studied extensively in image classification but much less in image segmentation. So far, our understanding of the learning behavior of DNNs trained by noisy segmentation labels remains limited. In this study, we address this deficiency in both binary segmentation of biological microscopy images and multi-class segmentation of natural images. We generate extremely noisy labels by randomly sampling a small fraction (e.g., 10%) or flipping a large fraction (e.g., 90%) of the ground truth labels. When trained with these noisy labels, DNNs provide largely the same segmentation performance as trained by the original ground truth. This indicates that DNNs learn structures hidden in labels rather than pixel-level labels per se in their supervised training for semantic segmentation. We refer to these hidden structures in labels as meta-structures. When DNNs are trained by labels with different perturbations to the meta-structure, we find consistent degradation in their segmentation performance. In contrast, incorporation of meta-structure information substantially improves performance of an unsupervised segmentation model developed for binary semantic segmentation. We define meta-structures mathematically as spatial density distributions and show both theoretically and experimentally how this formulation explains key observed learning behavior of DNNs.

----

## [213] Stochastic Planner-Actor-Critic for Unsupervised Deformable Image Registration

**Authors**: *Ziwei Luo, Jing Hu, Xin Wang, Shu Hu, Bin Kong, Youbing Yin, Qi Song, Xi Wu, Siwei Lyu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20086](https://doi.org/10.1609/aaai.v36i2.20086)

**Abstract**:

Large deformations of organs, caused by diverse shapes and nonlinear shape changes, pose a significant challenge for medical image registration. Traditional registration methods need to iteratively optimize an objective function via a specific deformation model along with meticulous parameter tuning, but which have limited capabilities in registering images with large deformations. While deep learning-based methods can learn the complex mapping from input images to their respective deformation field, it is regression-based and is prone to be stuck at local minima, particularly when large deformations are involved. To this end, we present Stochastic Planner-Actor-Critic (spac), a novel reinforcement learning-based framework that performs step-wise registration. The key notion is warping a moving image successively by each time step to finally align to a fixed image. Considering that it is challenging to handle high dimensional continuous action and state spaces in the conventional reinforcement learning (RL) framework, we introduce a new concept `Plan' to the standard Actor-Critic model, which is of low dimension and can facilitate the actor to generate a tractable high dimensional action. The entire framework is based on unsupervised training and operates in an end-to-end manner. We evaluate our method on several 2D and 3D medical image datasets, some of which contain large deformations. Our empirical results highlight that our work achieves consistent, significant gains and outperforms state-of-the-art methods.

----

## [214] Adaptive Poincaré Point to Set Distance for Few-Shot Classification

**Authors**: *Rongkai Ma, Pengfei Fang, Tom Drummond, Mehrtash Harandi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20087](https://doi.org/10.1609/aaai.v36i2.20087)

**Abstract**:

Learning and generalizing from limited examples, i.e., few-shot learning, is of core importance to many real-world vision applications. A principal way of achieving few-shot learning is to realize an embedding where samples from different classes are distinctive. Recent studies suggest that embedding via hyperbolic geometry enjoys low distortion for hierarchical and structured data, making it suitable for few-shot learning. In this paper, we propose to learn a context-aware hyperbolic metric to characterize the distance between a point and a set associated with a learned set to set distance. To this end, we formulate the metric as a weighted sum on the tangent bundle of the hyperbolic space and develop a mechanism to obtain the weights adaptively, based on the constellation of the points. This not only makes the metric local but also dependent on the task in hand, meaning that the metric will adapt depending on the samples that it compares. We empirically show that such metric yields robustness in the presence of outliers and achieves a tangible improvement over baseline models. This includes the state-of-the-art results on five popular few-shot classification benchmarks, namely mini-ImageNet, tiered-ImageNet, Caltech-UCSD Birds-200-2011(CUB), CIFAR-FS, and FC100.

----

## [215] Generative Adaptive Convolutions for Real-World Noisy Image Denoising

**Authors**: *Ruijun Ma, Shuyi Li, Bob Zhang, Zhengming Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20088](https://doi.org/10.1609/aaai.v36i2.20088)

**Abstract**:

Recently, deep learning techniques are soaring and have shown dramatic improvements in real-world noisy image denoising. However, the statistics of real noise generally vary with different camera sensors and in-camera signal processing pipelines. This will induce problems of most deep denoisers for the overfitting or degrading performance due to the noise discrepancy between the training and test sets. To remedy this issue, we propose a novel flexible and adaptive denoising network, coined as FADNet. Our FADNet is equipped with a plane dynamic filter module, which generates weight filters with flexibility that can adapt to the specific input and thereby impedes the FADNet from overfitting to the training data. Specifically, we exploit the advantage of the spatial and channel attention, and utilize this to devise a decoupling filter generation scheme. The generated filters are conditioned on the input and collaboratively applied to the decoded features for representation capability enhancement. We additionally introduce the Fourier transform and its inverse to guide the predicted weight filters to adapt to the noisy input with respect to the image contents. Experimental results demonstrate the superior denoising performances of the proposed FADNet versus the state-of-the-art. In contrast to the existing deep denoisers, our FADNet is not only flexible and efficient, but also exhibits a compelling generalization capability, enjoying tremendous potential for practical usage.

----

## [216] REMOTE: Reinforced Motion Transformation Network for Semi-supervised 2D Pose Estimation in Videos

**Authors**: *Xianzheng Ma, Hossein Rahmani, Zhipeng Fan, Bin Yang, Jun Chen, Jun Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20089](https://doi.org/10.1609/aaai.v36i2.20089)

**Abstract**:

Existing approaches for 2D pose estimation in videos often require a large number of dense annotations, which are costly and labor intensive to acquire. In this paper, we propose a semi-supervised REinforced MOtion Transformation nEtwork (REMOTE) to leverage a few labeled frames and temporal pose variations in videos, which enables effective learning of 2D pose estimation in sparsely annotated videos. Specifically, we introduce a Motion Transformer (MT) module to perform cross frame reconstruction, aiming to learn motion dynamic knowledge in videos. Besides, a novel reinforcement learning-based Frame Selection Agent (FSA) is designed within our framework, which is able to harness informative frame pairs on the fly to enhance the pose estimator under our cross reconstruction mechanism. We conduct extensive experiments that show the efficacy of our proposed REMOTE framework.

----

## [217] Learning from the Target: Dual Prototype Network for Few Shot Semantic Segmentation

**Authors**: *Binjie Mao, Xinbang Zhang, Lingfeng Wang, Qian Zhang, Shiming Xiang, Chunhong Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20090](https://doi.org/10.1609/aaai.v36i2.20090)

**Abstract**:

Due to the scarcity of annotated samples, the diversity between support set and query set becomes the main obstacle for few shot semantic segmentation. Most existing prototype-based approaches only exploit the prototype from the support feature and ignore the information from the query sample, failing to remove this obstacle.In this paper, we proposes a dual prototype network (DPNet) to dispose of few shot semantic segmentation from a new perspective. Along with the prototype extracted from the support set, we propose to build the pseudo-prototype based on foreground features in the query image. To achieve this goal, the cycle comparison module is developed to select reliable foreground features and generate the pseudo-prototype with them. Then, a prototype interaction module is utilized to integrate the information of the prototype and the pseudo-prototype based on their underlying correlation. Finally, a multi-scale fusion module is introduced to capture contextual information during the dense comparison between prototype (pseudo-prototype) and query feature. Extensive experiments conducted on two benchmarks demonstrate that our method exceeds previous state-of-the-arts with a sizable margin, verifying the effectiveness of the proposed method.

----

## [218] MOST-GAN: 3D Morphable StyleGAN for Disentangled Face Image Manipulation

**Authors**: *Safa C. Medin, Bernhard Egger, Anoop Cherian, Ye Wang, Joshua B. Tenenbaum, Xiaoming Liu, Tim K. Marks*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20091](https://doi.org/10.1609/aaai.v36i2.20091)

**Abstract**:

Recent advances in generative adversarial networks (GANs) have led to remarkable achievements in face image synthesis. While methods that use style-based GANs can generate strikingly photorealistic face images, it is often difficult to control the characteristics of the generated faces in a meaningful and disentangled way. Prior approaches aim to achieve such semantic control and disentanglement within the latent space of a previously trained GAN. In contrast, we propose a framework that a priori models physical attributes of the face such as 3D shape, albedo, pose, and lighting explicitly, thus providing disentanglement by design. Our method, MOST-GAN, integrates the expressive power and photorealism of style-based GANs with the physical disentanglement and flexibility of nonlinear 3D morphable models, which we couple with a state-of-the-art 2D hair manipulation network. MOST-GAN achieves photorealistic manipulation of portrait images with fully disentangled 3D control over their physical attributes, enabling extreme manipulation of lighting, facial expression, and pose variations up to full profile view.

----

## [219] Towards Bridging Sample Complexity and Model Capacity

**Authors**: *Shibin Mei, Chenglong Zhao, Shengchao Yuan, Bingbing Ni*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20092](https://doi.org/10.1609/aaai.v36i2.20092)

**Abstract**:

In this paper, we give a new definition for sample complexity, and further develop a theoretical analysis to bridge the gap between sample complexity and model capacity. In contrast to previous works which study on some toy samples, we conduct our analysis on more general data space, and build a qualitative relationship from sample complexity to model capacity required to achieve comparable performance. Besides, we introduce a simple indicator to evaluate the sample complexity based on continuous mapping. Moreover, we further analysis the relationship between sample complexity and data distribution, which paves the way to understand the present representation learning. Extensive experiments on several datasets well demonstrate the effectiveness of our evaluation method.

----

## [220] Towards Accurate Facial Motion Retargeting with Identity-Consistent and Expression-Exclusive Constraints

**Authors**: *Langyuan Mo, Haokun Li, Chaoyang Zou, Yubing Zhang, Ming Yang, Yihong Yang, Mingkui Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20093](https://doi.org/10.1609/aaai.v36i2.20093)

**Abstract**:

We address the problem of facial motion retargeting that aims to transfer facial motion from a 2D face image to 3D characters. Existing methods often formulate this problem as a 3D face reconstruction problem, which estimates the face attributes such as face identity and expression from face images. However, due to the lack of ground-truth labels for both identity and expression, most 3D-face reconstruction-based methods fail to capture the facial identity and expression accurately. As a result, these methods may not achieve promising performance. To address this, we propose an identity-consistent constraint to learn accurate identities by encouraging consistent identity prediction across multiple frames. Based on a more accurate identity, we are able to obtain a more accurate facial expression. Moreover, we further propose an expression-exclusive constraint to improve performance by avoiding the co-occurrence of contradictory expression units (e.g., ``brow lower'' vs. ``brow raise''). Extensive experiments on facial motion retargeting and 3D face reconstruction tasks demonstrate the superiority of the proposed method over existing methods. Our code and  supplementary materials are available at https://github.com/deepmo24/CPEM.

----

## [221] Can Vision Transformers Learn without Natural Images?

**Authors**: *Kodai Nakashima, Hirokatsu Kataoka, Asato Matsumoto, Kenji Iwata, Nakamasa Inoue, Yutaka Satoh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20094](https://doi.org/10.1609/aaai.v36i2.20094)

**Abstract**:

Is it possible to complete Vision Transformer (ViT) pre-training without natural images and human-annotated labels? This question has become increasingly relevant in recent months because while current ViT pre-training tends to rely heavily on a large number of natural images and human-annotated labels, the recent use of natural images has resulted in problems related to privacy violation, inadequate fairness protection, and the need for labor-intensive annotations. In this paper, we experimentally verify that the results of formula-driven supervised learning (FDSL) framework are comparable with, and can even partially outperform, sophisticated self-supervised learning (SSL) methods like SimCLRv2 and MoCov2 without using any natural images in the pre-training phase. We also consider ways to reorganize FractalDB generation based on our tentative conclusion that there is room for configuration improvements in the iterated function system (IFS) parameter settings of such databases. Moreover, we show that while ViTs pre-trained without natural images produce visualizations that are somewhat different from ImageNet pre-trained ViTs, they can still interpret natural image datasets to a large extent. Finally, in experiments using the CIFAR-10 dataset, we show that our model achieved a performance rate of 97.8, which is comparable to the rate of 97.4 achieved with SimCLRv2 and 98.0 achieved with ImageNet.

----

## [222] Federated Learning for Face Recognition with Gradient Correction

**Authors**: *Yifan Niu, Weihong Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20095](https://doi.org/10.1609/aaai.v36i2.20095)

**Abstract**:

With increasing appealing to privacy issues in face recognition, federated learning has emerged as one of the most prevalent approaches to study the unconstrained face recognition problem with private decentralized data. However, conventional decentralized federated algorithm sharing whole parameters of networks among clients suffers from privacy leakage in face recognition scene. In this work, we introduce a framework, FedGC, to tackle federated learning for face recognition and guarantees higher privacy. We explore a novel idea of correcting gradients from the perspective of backward propagation and propose a softmax-based regularizer to correct gradients of class embeddings by precisely injecting a cross-client gradient term. Theoretically, we show that FedGC constitutes a valid loss function similar to standard softmax. Extensive experiments have been conducted to validate the superiority of FedGC which can match the performance of conventional centralized methods utilizing full training dataset on several popular benchmark datasets.

----

## [223] Restorable Image Operators with Quasi-Invertible Networks

**Authors**: *Hao Ouyang, Tengfei Wang, Qifeng Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20096](https://doi.org/10.1609/aaai.v36i2.20096)

**Abstract**:

Image operators have been extensively applied to create visually attractive photos for users to share processed images on social media. However, most image operators often smooth out details or generate textures after the processing, which removes the original content and raises challenges for restoring the original image. To resolve this issue, we propose a quasi-invertible model that learns common image processing operators in a restorable fashion: the learned image operators can generate visually pleasing results with the original content embedded. Our model is trained on input-output pairs that represent an image processing operator's behavior and uses a network that consists of an invertible branch and a non-invertible branch to increase our model's approximation capability. We evaluate the proposed model on ten image operators, including detail enhancement, abstraction, blur,  photographic style, and non-photorealistic style. Extensive experiments show that our approach outperforms relevant baselines in the restoration quality, and the learned restorable operator is fast in inference and robust to compression. Furthermore, we demonstrate that the invertible operator can be easily applied to practical applications such as restorable human face retouching and highlight preserved exposure adjustment.

----

## [224] TEACh: Task-Driven Embodied Agents That Chat

**Authors**: *Aishwarya Padmakumar, Jesse Thomason, Ayush Shrivastava, Patrick Lange, Anjali Narayan-Chen, Spandana Gella, Robinson Piramuthu, Gökhan Tür, Dilek Hakkani-Tür*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20097](https://doi.org/10.1609/aaai.v36i2.20097)

**Abstract**:

Robots operating in human spaces must be able to engage in natural language interaction, both understanding and executing instructions, and using conversation to resolve ambiguity and correct mistakes. To study this, we introduce TEACh, a dataset of over 3,000 human-human, interactive dialogues to complete household tasks in simulation. A Commander with access to oracle information about a task communicates in natural language with a Follower. The Follower navigates through and interacts with the environment to complete tasks varying in complexity from "Make Coffee" to "Prepare Breakfast", asking questions and getting additional information from the Commander. We propose three benchmarks using TEACh to study embodied intelligence challenges, and we evaluate initial models' abilities in dialogue understanding, language grounding, and task execution.

----

## [225] Label-Efficient Hybrid-Supervised Learning for Medical Image Segmentation

**Authors**: *Junwen Pan, Qi Bi, Yanzhan Yang, Pengfei Zhu, Cheng Bian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20098](https://doi.org/10.1609/aaai.v36i2.20098)

**Abstract**:

Due to the lack of expertise for medical image annotation, the investigation of label-efficient methodology for medical image segmentation becomes a heated topic. Recent progresses focus on the efficient utilization of weak annotations together with few strongly-annotated labels so as to achieve comparable segmentation performance in many unprofessional scenarios. However, these approaches only concentrate on the supervision inconsistency between strongly- and weakly-annotated instances but ignore the instance inconsistency inside the weakly-annotated instances, which inevitably leads to performance degradation.
 To address this problem, we propose a novel label-efficient hybrid-supervised framework, which considers each weakly-annotated instance individually and learns its weight guided by the gradient direction of the strongly-annotated instances, so that the high-quality prior in the strongly-annotated instances is better exploited and the weakly-annotated instances are depicted more precisely. Specially, our designed dynamic instance indicator (DII) realizes the above objectives, and is adapted to our dynamic co-regularization (DCR) framework further to alleviate the erroneous accumulation from distortions of weak annotations. Extensive experiments on two hybrid-supervised medical segmentation datasets demonstrate that with only 10% strong labels, the proposed framework can leverage the weak labels efficiently and achieve competitive performance against the 100% strong-label supervised scenario.

----

## [226] Less Is More: Pay Less Attention in Vision Transformers

**Authors**: *Zizheng Pan, Bohan Zhuang, Haoyu He, Jing Liu, Jianfei Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20099](https://doi.org/10.1609/aaai.v36i2.20099)

**Abstract**:

Transformers have become one of the dominant architectures in deep learning, particularly as a powerful alternative to convolutional neural networks (CNNs) in computer vision. However, Transformer training and inference in previous works can be prohibitively expensive due to the quadratic complexity of self-attention over a long sequence of representations, especially for high-resolution dense prediction tasks. To this end, we present a novel Less attention vIsion Transformer (LIT), building upon the fact that the early self-attention layers in Transformers still focus on local patterns and bring minor benefits in recent hierarchical vision Transformers. Specifically, we propose a hierarchical Transformer where we use pure multi-layer perceptrons (MLPs) to encode rich local patterns in the early stages while applying self-attention modules to capture longer dependencies in deeper layers. Moreover, we further propose a learned deformable token merging module to adaptively fuse informative patches in a non-uniform manner. The proposed LIT achieves promising performance on image recognition tasks, including image classification, object detection and instance segmentation, serving as a strong backbone for many vision tasks. Code is available at https://github.com/zip-group/LIT.

----

## [227] Unsupervised Representation for Semantic Segmentation by Implicit Cycle-Attention Contrastive Learning

**Authors**: *Bo Pang, Yizhuo Li, Yifan Zhang, Gao Peng, Jiajun Tang, Kaiwen Zha, Jiefeng Li, Cewu Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20100](https://doi.org/10.1609/aaai.v36i2.20100)

**Abstract**:

We study the unsupervised representation learning for the semantic segmentation task. Different from previous works that aim at providing unsupervised pre-trained backbones for segmentation models which need further supervised fine-tune, here, we focus on providing representation that is only trained by unsupervised methods. This means models need to directly generate pixel-level, linearly separable semantic results.  We first explore and present two factors that have significant effects on segmentation under the contrastive learning framework: 1) the difficulty and diversity of the positive contrastive pairs, 2) the balance of global and local features. With the intention of optimizing these factors, we propose the cycle-attention contrastive learning (CACL). CACL makes use of semantic continuity of video frames, adopting unsupervised cycle-consistent attention mechanism to implicitly conduct contrastive learning with difficult, global-local-balanced positive pixel pairs. Compared with baseline model MoCo-v2 and other unsupervised methods, CACL demonstrates consistently superior performance on PASCAL VOC (+4.5 mIoU) and Cityscapes (+4.5 mIoU) datasets.

----

## [228] Graph-Based Point Tracker for 3D Object Tracking in Point Clouds

**Authors**: *Minseong Park, Hongje Seong, Wonje Jang, Euntai Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20101](https://doi.org/10.1609/aaai.v36i2.20101)

**Abstract**:

In this paper, a new deep learning network named as graph-based point tracker (GPT) is proposed for 3D object tracking in point clouds. GPT is not based on Siamese network applied to template and search area, but it is based on the transfer of target clue from the template to the search area. GPT is end-to-end trainable. GPT has two new modules: graph feature augmentation (GFA) and improved target clue (ITC) module. The key idea of GFA is to exploit one-to-many relationship between template and search area points using a bipartite graph. In GFA, edge features of the bipartite graph are generated by transferring the target clues of template points to search area points through edge convolution. It captures the relationship between template and search area points effectively from the perspective of geometry and shape of two point clouds. The second module is ITC. The key idea of ITC is to embed the information of the center of the target into the edges of the bipartite graph via Hough voting, strengthening the discriminative power of GFA. Both modules significantly contribute to the improvement of GPT by transferring geometric and shape information including target center from target template to search area effectively. Experiments on the KITTI tracking dataset show that GPT achieves state-of-the-art performance and can run in real-time.

----

## [229] SyncTalkFace: Talking Face Generation with Precise Lip-Syncing via Audio-Lip Memory

**Authors**: *Se Jin Park, Minsu Kim, Joanna Hong, Jeongsoo Choi, Yong Man Ro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20102](https://doi.org/10.1609/aaai.v36i2.20102)

**Abstract**:

The challenge of talking face generation from speech lies in aligning two different modal information, audio and video, such that the mouth region corresponds to input audio. Previous methods either exploit audio-visual representation learning or leverage intermediate structural information such as landmarks and 3D models. However, they struggle to synthesize fine details of the lips varying at the phoneme level as they do not sufficiently provide visual information of the lips at the video synthesis step. To overcome this limitation, our work proposes Audio-Lip Memory that brings in visual information of the mouth region corresponding to input audio and enforces fine-grained audio-visual coherence. It stores lip motion features from sequential ground truth images in the value memory and aligns them with corresponding audio features so that they can be retrieved using audio input at inference time. Therefore, using the retrieved lip motion features as visual hints, it can easily correlate audio with visual dynamics in the synthesis step. By analyzing the memory, we demonstrate that unique lip features are stored in each memory slot at the phoneme level, capturing subtle lip motion based on memory addressing. In addition, we introduce visual-visual synchronization loss which can enhance lip-syncing performance when used along with audio-visual synchronization loss in our model. Extensive experiments are performed to verify that our method generates high-quality video with mouth shapes that best align with the input audio, outperforming previous state-of-the-art methods.

----

## [230] Vision Transformers Are Robust Learners

**Authors**: *Sayak Paul, Pin-Yu Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20103](https://doi.org/10.1609/aaai.v36i2.20103)

**Abstract**:

Transformers, composed of multiple self-attention layers, hold strong promises toward a generic learning primitive applicable to different data modalities, including the recent breakthroughs in computer vision achieving state-of-the-art (SOTA) standard accuracy. What remains largely unexplored is their robustness evaluation and attribution. In this work, we study the robustness of the Vision Transformer (ViT) (Dosovitskiy et al. 2021) against common corruptions and perturbations, distribution shifts, and natural adversarial examples. We use six different diverse ImageNet datasets concerning robust classification to conduct a comprehensive performance comparison of ViT(Dosovitskiy et al. 2021) models and SOTA convolutional neural networks (CNNs), Big-Transfer (Kolesnikov et al. 2020). Through a series of six systematically designed experiments, we then present analyses that provide both quantitative andqualitative indications to explain why ViTs are indeed more robust learners. For example, with fewer parameters and similar dataset and pre-training combinations, ViT gives a top-1accuracy of 28.10% on ImageNet-A which is 4.3x higher than a comparable variant of BiT. Our analyses on image masking, Fourier spectrum sensitivity, and spread on discrete cosine energy spectrum reveal intriguing properties of ViT attributing to improved robustness. Code for reproducing our experiments is available at https://git.io/J3VO0.

----

## [231] Self-Supervised Category-Level 6D Object Pose Estimation with Deep Implicit Shape Representation

**Authors**: *Wanli Peng, Jianhang Yan, Hongtao Wen, Yi Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20104](https://doi.org/10.1609/aaai.v36i2.20104)

**Abstract**:

Category-level 6D pose estimation can be better generalized to unseen objects in a category compared with instance-level 6D pose estimation. However, existing category-level 6D pose estimation methods usually require supervised training with a sufficient number of 6D pose annotations of objects which makes them difficult to be applied in real scenarios. To address this problem, we propose a self-supervised framework for category-level 6D pose estimation in this paper. We leverage DeepSDF as a 3D object representation and design several novel loss functions based on DeepSDF to help the self-supervised model predict unseen object poses without any 6D object pose labels and explicit 3D models in real scenarios. Experiments demonstrate that our method achieves comparable performance with the state-of-the-art fully supervised methods on the category-level NOCS benchmark.

----

## [232] Semantic-Aware Representation Blending for Multi-Label Image Recognition with Partial Labels

**Authors**: *Tao Pu, Tianshui Chen, Hefeng Wu, Liang Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20105](https://doi.org/10.1609/aaai.v36i2.20105)

**Abstract**:

Training the multi-label image recognition models with partial labels, in which merely some labels are known while others are unknown for each image, is a considerably challenging and practical task. To address this task, current algorithms mainly depend on pre-training classification or similarity models to generate pseudo labels for the unknown labels. However, these algorithms depend on sufficient multi-label annotations to train the models, leading to poor performance especially with low known label proportion. In this work, we propose to blend category-specific representation across different images to transfer information of known labels to complement unknown labels, which can get rid of pre-training models and thus does not depend on sufficient annotations. To this end, we design a unified semantic-aware representation blending (SARB) framework that exploits instance-level and prototype-level semantic representation to complement unknown labels by two complementary modules: 1) an instance-level representation blending (ILRB) module blends the representations of the known labels in an image to the representations of the unknown labels in another image to complement these unknown labels. 2) a prototype-level representation blending (PLRB) module learns more stable representation prototypes for each category and blends the representation of unknown labels with the prototypes of corresponding labels to complement these labels. Extensive experiments on the MS-COCO, Visual Genome, Pascal VOC 2007 datasets show that the proposed SARB framework obtains superior performance over current leading competitors on all known label proportion settings, i.e., with the mAP improvement of 4.6%, 4.6%, 2.2% on these three datasets when the known label proportion is 10%. Codes are available at https://github.com/HCPLab-SYSU/HCP-MLR-PL.

----

## [233] ReX: An Efficient Approach to Reducing Memory Cost in Image Classification

**Authors**: *Xuwei Qian, Renlong Hang, Qingshan Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20106](https://doi.org/10.1609/aaai.v36i2.20106)

**Abstract**:

Exiting simple samples in adaptive multi-exit networks through early modules is an effective way to achieve high computational efficiency. One can observe that deployments of multi-exit architectures on resource-constrained devices are easily limited by high memory footprint of early modules. In this paper, we propose a novel approach named recurrent aggregation operator (ReX), which uses recurrent neural networks (RNNs) to effectively aggregate intra-patch features within a large receptive field to get delicate local representations, while bypassing large early activations. The resulting model, named ReXNet, can be easily extended to dynamic inference by introducing a novel consistency-based early exit criteria, which is based on the consistency of classification decisions over several modules, rather than the entropy of the prediction distribution. Extensive experiments on two benchmark datasets, i.e., Visual Wake Words, ImageNet-1k, demonstrate that our method consistently reduces the peak RAM and average latency of a wide variety of adaptive models on low-power devices.

----

## [234] CPRAL: Collaborative Panoptic-Regional Active Learning for Semantic Segmentation

**Authors**: *Yu Qiao, Jincheng Zhu, Chengjiang Long, Zeyao Zhang, Yuxin Wang, Zhenjun Du, Xin Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20107](https://doi.org/10.1609/aaai.v36i2.20107)

**Abstract**:

Acquiring the most representative examples via active learning (AL) can benefit many data-dependent computer vision tasks by minimizing efforts of image-level or pixel-wise annotations. In this paper, we propose a novel Collaborative Panoptic-Regional Active Learning framework (CPRAL) to address the semantic segmentation task. For a small batch of images initially sampled with pixel-wise annotations, we employ panoptic information to initially select unlabeled samples. Considering the class imbalance in the segmentation dataset, we import a Regional Gaussian Attention module (RGA) to achieve semantics-biased selection. The subset is highlighted by vote entropy and then attended by Gaussian kernels to maximize the biased regions. We also propose a Contextual Labels Extension (CLE) to boost regional annotations with contextual attention guidance. With the collaboration of semantics-agnostic panoptic matching and region-biased selection and extension, our CPRAL can strike a balance between labeling efforts and performance and compromise the semantics distribution. We perform extensive experiments on Cityscapes and BDD10K datasets and show that CPRAL outperforms the cutting-edge methods with impressive results and less labeling proportion.

----

## [235] Activation Modulation and Recalibration Scheme for Weakly Supervised Semantic Segmentation

**Authors**: *Jie Qin, Jie Wu, Xuefeng Xiao, Lujun Li, Xingang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20108](https://doi.org/10.1609/aaai.v36i2.20108)

**Abstract**:

Image-level weakly supervised semantic segmentation (WSSS) is a fundamental yet challenging computer vision task facilitating scene understanding and automatic driving. Most existing methods resort to classification-based Class Activation Maps (CAMs) to play as the initial pseudo labels, which tend to focus on the discriminative image regions and lack customized characteristics for the segmentation task. To alleviate this issue, we propose a novel activation modulation and recalibration (AMR) scheme, which leverages a spotlight branch and a compensation branch to obtain weighted CAMs that can provide recalibration supervision and task-specific concepts. Specifically, an attention modulation module (AMM) is employed to rearrange the distribution of feature importance from the channel-spatial sequential perspective, which helps to explicitly model channel-wise interdependencies and spatial encodings to adaptively modulate segmentation-oriented activation responses. Furthermore, we introduce a cross pseudo supervision for dual branches, which can be regarded as a semantic similar regularization to mutually refine two branches. Extensive experiments show that AMR establishes a new state-of-the-art performance on the PASCAL VOC 2012 dataset, surpassing not only current methods trained with the image-level of supervision but also some methods relying on stronger supervision, such as saliency label. Experiments also reveal that our scheme is plug-and-play and can be incorporated with other approaches to boost their performance. Our code is available at: https://github.com/jieqin-ai/AMR.

----

## [236] TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework Using Self-Supervised Multi-Task Learning

**Authors**: *Linhao Qu, Shaolei Liu, Manning Wang, Zhijian Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20109](https://doi.org/10.1609/aaai.v36i2.20109)

**Abstract**:

In this paper, we propose TransMEF, a transformer-based multi-exposure image fusion framework that uses self-supervised multi-task learning. The framework is based on an encoder-decoder network, which can be trained on large natural image datasets and does not require ground truth fusion images. We design three self-supervised reconstruction tasks according to the characteristics of multi-exposure images and conduct these tasks simultaneously using multi-task learning; through this process, the network can learn the characteristics of multi-exposure images and extract more generalized features. In addition, to compensate for the defect in establishing long-range dependencies in CNN-based architectures, we design an encoder that combines a CNN module with a transformer module. This combination enables the network to focus on both local and global information. We evaluated our method and compared it to 11 competitive traditional and deep learning-based methods on the latest released multi-exposure image fusion benchmark dataset, and our method achieved the best performance in both subjective and objective evaluations. Code will be available at https://github.com/miccaiif/TransMEF.

----

## [237] Deep Implicit Statistical Shape Models for 3D Medical Image Delineation

**Authors**: *Ashwin Raju, Shun Miao, Dakai Jin, Le Lu, Junzhou Huang, Adam P. Harrison*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20110](https://doi.org/10.1609/aaai.v36i2.20110)

**Abstract**:

3D delineation of anatomical structures is a cardinal goal in medical imaging analysis. Prior to deep learning, statistical shape models (SSMs) that imposed anatomical constraints and produced high quality surfaces were a core technology. Today’s fully-convolutional networks (FCNs), while dominant, do not offer these capabilities. We present deep implicit statistical shape models (DISSMs), a new approach that marries the representation power of deep networks with the benefits of SSMs. DISSMs use an implicit representation to produce compact and descriptive deep surface embeddings that permit statistical models of anatomical variance. To reliably fit anatomically plausible shapes to an image, we introduce a novel rigid and non-rigid pose estimation pipeline that is modelled as a Markov decision process (MDP). Intra-dataset experiments on the task of pathological liver segmentation demonstrate that DISSMs can perform more robustly than four leading FCN models, including nnU-Net + an adversarial prior: reducing the mean Hausdorff distance (HD) by 7.5-14.3 mm and improving the worst case Dice-Sørensen coefficient (DSC) by 1.2-2.3%. More critically, cross-dataset experiments on an external and highly challenging clinical dataset demonstrate that DISSMs improve the mean DSC and HD by 2.1-5.9% and 9.9-24.5 mm, respectively, and the worst-case DSC by 5.4-7.3%. Supplemental validation on a highly challenging and low-contrast larynx dataset further demonstrate DISSM’s improvements. These improvements are over and above any benefits from representing delineations with high-quality surfaces.

----

## [238] Decompose the Sounds and Pixels, Recompose the Events

**Authors**: *Varshanth R. Rao, Md Ibrahim Khalil, Haoda Li, Peng Dai, Juwei Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20111](https://doi.org/10.1609/aaai.v36i2.20111)

**Abstract**:

In this paper, we propose a framework centering around a novel architecture called the Event Decomposition Recomposition Network (EDRNet) to tackle the Audio-Visual Event (AVE) localization problem in the supervised and weakly supervised settings. AVEs in the real world exhibit common unraveling patterns (termed as Event Progress Checkpoints(EPC)), which humans can perceive through the cooperation of their auditory and visual senses. Unlike earlier methods which attempt to recognize entire event sequences, the EDRNet models EPCs and inter-EPC relationships using stacked temporal convolutions. Based on the postulation that EPC representations are theoretically consistent for an event category, we introduce the State Machine Based Video Fusion, a novel augmentation technique that blends source videos using different EPC template sequences. Additionally, we design a new loss function called the Land-Shore-Sea loss to compactify continuous foreground and background representations. Lastly, to alleviate the issue of confusing events during weak supervision, we propose a prediction stabilization method called Bag to Instance Label Correction. Experiments on the AVE dataset show that our collective framework outperforms the state-of-the-art by a sizable margin.

----

## [239] Learning from Label Proportions with Prototypical Contrastive Clustering

**Authors**: *Laura Elena Cué La Rosa, Dário Augusto Borges Oliveira*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20112](https://doi.org/10.1609/aaai.v36i2.20112)

**Abstract**:

The use of priors to avoid manual labeling for training machine learning methods has received much attention in the last few years. One of the critical subthemes in this regard is Learning from Label Proportions (LLP), where only the information about class proportions is available for training the models. While various LLP training settings verse in the literature, most approaches focus on bag-level label proportions errors, often leading to suboptimal solutions. This paper proposes a new model that jointly uses prototypical contrastive learning and bag-level cluster proportions to implement efficient LLP classification. Our proposal explicitly relaxes the equipartition constraint commonly used in prototypical contrastive learning methods and incorporates the exact cluster proportions into the optimal transport algorithm used for cluster assignments. At inference time, we compute the clusters' assignment, delivering instance-level classification. We experimented with our method on two widely used image classification benchmarks and report a new state-of-art LLP performance, achieving results close to fully supervised methods.

----

## [240] Beyond Learning Features: Training a Fully-Functional Classifier with ZERO Instance-Level Labels

**Authors**: *Deepak Babu Sam, Abhinav Agarwalla, Venkatesh Babu Radhakrishnan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20113](https://doi.org/10.1609/aaai.v36i2.20113)

**Abstract**:

We attempt to train deep neural networks for classification without using any labeled data. Existing unsupervised methods, though mine useful clusters or features, require some annotated samples to facilitate the final task-specific predictions. This defeats the true purpose of unsupervised learning and hence we envisage a paradigm of `true' self-supervision, where absolutely no annotated instances are used for training a classifier. The proposed method first pretrains a deep network through self-supervision and performs clustering on the learned features. A classifier layer is then appended to the self-supervised network and is trained by matching the distribution of the predictions to that of a predefined prior. This approach leverages the distribution of labels for supervisory signals and consequently, no image-label pair is needed. Experiments reveal that the method works on major nominal as well as ordinal classification datasets and delivers significant performance.

----

## [241] Reference-Guided Pseudo-Label Generation for Medical Semantic Segmentation

**Authors**: *Constantin Marc Seibold, Simon Reiß, Jens Kleesiek, Rainer Stiefelhagen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20114](https://doi.org/10.1609/aaai.v36i2.20114)

**Abstract**:

Producing densely annotated data is a difficult and tedious task for medical imaging applications. 
To address this problem, we propose a novel approach to generate supervision for semi-supervised semantic segmentation. 
We argue that visually similar regions between labeled and unlabeled images likely contain the same semantics and therefore should share their label.
Following this thought, we use a small number of labeled images as reference material and match pixels in an unlabeled image to the semantic of the best fitting pixel in a reference set.
This way, we avoid pitfalls such as confirmation bias, common in purely prediction-based pseudo-labeling.
Since our method does not require any architectural changes or accompanying networks, one can easily insert it into existing frameworks.
We achieve the same performance as a standard fully supervised model on X-ray anatomy segmentation, albeit using 95% fewer labeled images.
Aside from an in-depth analysis of different aspects of our proposed method, we further demonstrate the effectiveness of our reference-guided learning paradigm by comparing our approach against existing methods for retinal fluid segmentation with competitive performance as we improve upon recent work by up to 15% mean IoU.

----

## [242] Information-Theoretic Bias Reduction via Causal View of Spurious Correlation

**Authors**: *Seonguk Seo, Joon-Young Lee, Bohyung Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20115](https://doi.org/10.1609/aaai.v36i2.20115)

**Abstract**:

We propose an information-theoretic bias measurement technique through a causal interpretation of spurious correlation, which is effective to identify the feature-level algorithmic bias by taking advantage of conditional mutual information. Although several bias measurement methods have been proposed and widely investigated to achieve algorithmic fairness in various tasks such as face recognition, their accuracy- or logit-based metrics are susceptible to leading to trivial prediction score adjustment rather than fundamental bias reduction. Hence, we design a novel debiasing framework against the algorithmic bias, which incorporates a bias regularization loss derived by the proposed information-theoretic bias measurement approach. In addition, we present a simple yet effective unsupervised debiasing technique based on stochastic label noise, which does not require the explicit supervision of bias information. The proposed bias measurement and debiasing approaches are validated in diverse realistic scenarios through extensive experiments on multiple standard benchmarks.

----

## [243] Improving Scene Graph Classification by Exploiting Knowledge from Texts

**Authors**: *Sahand Sharifzadeh, Sina Moayed Baharlou, Martin Schmitt, Hinrich Schütze, Volker Tresp*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20116](https://doi.org/10.1609/aaai.v36i2.20116)

**Abstract**:

Training scene graph classification models requires a large amount of annotated image data. Meanwhile, scene graphs represent relational knowledge that can be modeled with symbolic data from texts or knowledge graphs. While image annotation demands extensive labor, collecting textual descriptions of natural scenes requires less effort. In this work, we investigate whether textual scene descriptions can substitute for annotated image data. To this end, we employ a scene graph classification framework that is trained not only from annotated images but also from symbolic data. In our architecture, the symbolic entities are first mapped to their correspondent image-grounded representations and then fed into the relational reasoning pipeline. Even though a structured form of knowledge, such as the form in knowledge graphs, is not always available, we can generate it from unstructured texts using a transformer-based language model. We show that by fine-tuning the classification pipeline with the extracted knowledge from texts, we can achieve ~8x more accurate results in scene graph classification, ~3x in object classification, and ~1.5x in predicate classification, compared to the supervised baselines with only 1% of the annotated images.

----

## [244] Reliable Inlier Evaluation for Unsupervised Point Cloud Registration

**Authors**: *Yaqi Shen, Le Hui, Haobo Jiang, Jin Xie, Jian Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20117](https://doi.org/10.1609/aaai.v36i2.20117)

**Abstract**:

Unsupervised point cloud registration algorithm usually suffers from the unsatisfied registration precision in the partially overlapping problem due to the lack of effective inlier evaluation. In this paper, we propose a neighborhood consensus based reliable inlier evaluation method for robust unsupervised point cloud registration. It is expected to capture the discriminative geometric difference between the source neighborhood and the corresponding pseudo target neighborhood for effective inlier distinction. Specifically, our model consists of a matching map refinement module and an inlier evaluation module. In our matching map refinement module, we improve the point-wise matching map estimation by integrating the matching scores of neighbors into it. The aggregated neighborhood information potentially facilitates the discriminative map construction so that high-quality correspondences can be provided for generating the pseudo target point cloud. Based on the observation that the outlier has the significant structure-wise difference between its source neighborhood and corresponding pseudo target neighborhood while this difference for inlier is small, the inlier evaluation module exploits this difference to score the inlier confidence for each estimated correspondence. In particular, we construct an effective graph representation for capturing this geometric difference between the neighborhoods. Finally, with the learned correspondences and the corresponding inlier confidence, we use the weighted SVD algorithm for transformation estimation.Under the unsupervised setting, we exploit the Huber function based global alignment loss, the local neighborhood consensus loss and spatial consistency loss for model optimization. The experimental results on extensive datasets demonstrate that our unsupervised point cloud registration method can yield comparable performance.

----

## [245] Explainable Survival Analysis with Convolution-Involved Vision Transformer

**Authors**: *Yifan Shen, Li Liu, Zhihao Tang, Zongyi Chen, Guixiang Ma, Jiyan Dong, Xi Zhang, Lin Yang, Qingfeng Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20118](https://doi.org/10.1609/aaai.v36i2.20118)

**Abstract**:

Image-based survival prediction models can facilitate doctors in diagnosing and treating cancer patients. With the advance of digital pathology technologies, the big whole slide images (WSIs) provide increasing resolution and more details for diagnosis. However, the gigabyte-size WSIs would make most models computationally infeasible. To this end, instead of using the complete WSIs, most of existing models only use a pre-selected subset of key patches or patch clusters as input, which might fail to completely capture the patient's tumor morphology. In this work, we aim to develop a novel survival analysis model to fully utilize the complete WSI information. We show that the use of a Vision Transformer (ViT) backbone, together with convolution operations involved in it, is an effective framework to improve the prediction performance. Additionally, we present a post-hoc explainable method to identify the most salient patches and distinct morphology features, making the model more faithful and the results easier to comprehend by human users. Evaluations on two large cancer datasets show that our proposed model is more effective and has better interpretability for survival prediction.

----

## [246] Un-mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning

**Authors**: *Zhiqiang Shen, Zechun Liu, Zhuang Liu, Marios Savvides, Trevor Darrell, Eric Poe Xing*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20119](https://doi.org/10.1609/aaai.v36i2.20119)

**Abstract**:

The recently advanced unsupervised learning approaches use the siamese-like framework to compare two "views" from the same image for learning representations. Making the two views distinctive is a core to guarantee that unsupervised methods can learn meaningful information. However, such frameworks are sometimes fragile on overfitting if the augmentations used for generating two views are not strong enough, causing the over-confident issue on the training data. This drawback hinders the model from learning subtle variance and fine-grained information. To address this, in this work we aim to involve the soft distance concept on label space in the contrastive-based unsupervised learning task and let the model be aware of the soft degree of similarity between positive or negative pairs through mixing the input data space, to further work collaboratively for the input and loss spaces. Despite its conceptual simplicity, we show empirically that with the solution -- Unsupervised image mixtures (Un-Mix), we can learn subtler, more robust and generalized representations from the transformed input and corresponding new label space. Extensive experiments are conducted on CIFAR-10, CIFAR-100, STL-10, Tiny ImageNet and standard ImageNet-1K with popular unsupervised methods SimCLR, BYOL, MoCo V1&V2, SwAV, etc. Our proposed image mixture and label assignment strategy can obtain consistent improvement by 1~3% following exactly the same hyperparameters and training procedures of the base methods. Code is publicly available at https://github.com/szq0214/Un-Mix.

----

## [247] On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals

**Authors**: *Haizhou Shi, Youcai Zhang, Siliang Tang, Wenjie Zhu, Yaqian Li, Yandong Guo, Yueting Zhuang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20120](https://doi.org/10.1609/aaai.v36i2.20120)

**Abstract**:

It is a consensus that small models perform quite poorly under the paradigm of self-supervised contrastive learning. Existing methods usually adopt a large off-the-shelf model to transfer knowledge to the small one via distillation. Despite their effectiveness, distillation-based methods may not be suitable for some resource-restricted scenarios due to the huge computational expenses of deploying a large model. In this paper, we study the issue of training self-supervised small models without distillation signals. We first evaluate the representation spaces of the small models and make two non-negligible observations: (i) the small models can complete the pretext task without overfitting despite their limited capacity and (ii) they universally suffer the problem of over clustering. Then we verify multiple assumptions that are considered to alleviate the over-clustering phenomenon. Finally, we combine the validated techniques and improve the baseline performances of five small architectures with considerable margins, which indicates that training small self-supervised contrastive models is feasible even without distillation signals. The code is available at https://github.com/WOWNICE/ssl-small.

----

## [248] Social Interpretable Tree for Pedestrian Trajectory Prediction

**Authors**: *Liushuai Shi, Le Wang, Chengjiang Long, Sanping Zhou, Fang Zheng, Nanning Zheng, Gang Hua*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20121](https://doi.org/10.1609/aaai.v36i2.20121)

**Abstract**:

Understanding the multiple socially-acceptable future behaviors is an essential task for many vision applications. In this paper, we propose a tree-based method, termed as Social Interpretable Tree (SIT), to address this multi-modal prediction task, where a hand-crafted tree is built depending on the prior information of observed trajectory to model multiple future trajectories. Specifically, a path in the tree from the root to leaf represents an individual possible future trajectory. SIT employs a coarse-to-fine optimization strategy, in which the tree is first built by high-order velocity to balance the complexity and coverage of the tree and then optimized greedily to encourage multimodality. Finally, a teacher-forcing refining operation is used to predict the final fine trajectory. Compared with prior methods which leverage implicit latent variables to represent possible future trajectories, the path in the tree can explicitly explain the rough moving behaviors (e.g., go straight and then turn right), and thus provides better interpretability. Despite the hand-crafted tree, the experimental results on ETH-UCY and Stanford Drone datasets demonstrate that our method is capable of matching or exceeding the performance of state-of-the-art methods. Interestingly, the experiments show that the raw built tree without training outperforms many prior deep neural network based approaches. Meanwhile, our method presents sufficient flexibility in long-term prediction and different best-of-K predictions.

----

## [249] P^3-Net: Part Mobility Parsing from Point Cloud Sequences via Learning Explicit Point Correspondence

**Authors**: *Yahao Shi, Xinyu Cao, Feixiang Lu, Bin Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20122](https://doi.org/10.1609/aaai.v36i2.20122)

**Abstract**:

Understanding an articulated 3D object with its movable parts is an essential skill for an intelligent agent. This paper presents a novel approach to parse 3D part mobility from point cloud sequences. The key innovation is learning explicit point correspondence from a raw unordered point cloud sequence. We propose a novel deep network called P^3-Net to parallelize trajectory feature extraction and point correspondence establishment, performing joint optimization between them. Specifically, we design a Match-LSTM module to reaggregate point features among different frames by a point correspondence matrix, a.k.a. the matching matrix. To obtain this matrix, an attention module is proposed to calculate the point correspondence. Moreover, we implement a Gumbel-Sinkhorn module to reduce the many-to-one relationship for better point correspondence. We conduct comprehensive evaluations on public benchmarks, including the motion dataset and the PartNet dataset. Results demonstrate that our approach outperforms SOTA methods on various 3D parsing tasks of part mobility, including motion flow prediction, motion part segmentation, and motion attribute (i.e. axis & range) estimation. Moreover, we integrate our approach into a robot perception module to validate its robustness.

----

## [250] Improving Zero-Shot Phrase Grounding via Reasoning on External Knowledge and Spatial Relations

**Authors**: *Zhan Shi, Yilin Shen, Hongxia Jin, Xiaodan Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20123](https://doi.org/10.1609/aaai.v36i2.20123)

**Abstract**:

Phrase grounding is a multi-modal problem that localizes a particular noun phrase in an image referred to by a text query. In the challenging zero-shot phrase grounding setting, the existing state-of-the-art grounding models have limited capacity in handling the unseen phrases. Humans, however, can ground novel types of objects in images with little effort, significantly benefiting from reasoning with commonsense. In this paper, we design a novel phrase grounding architecture that builds multi-modal knowledge graphs using external knowledge and then performs graph reasoning and spatial relation reasoning to localize the referred nouns phrases. We perform extensive experiments on different zero-shot grounding splits sub-sampled from the Flickr30K Entity and Visual Genome dataset, demonstrating that the proposed framework is orthogonal to backbone image encoders and outperforms the baselines by 2~3% in accuracy, resulting in a significant improvement under the standard evaluation metrics.

----

## [251] Iterative Contrast-Classify for Semi-supervised Temporal Action Segmentation

**Authors**: *Dipika Singhania, Rahul Rahaman, Angela Yao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20124](https://doi.org/10.1609/aaai.v36i2.20124)

**Abstract**:

Temporal action segmentation classifies the action of each frame in (long) video sequences. Due to the high cost of frame-wise labeling, we propose the first semi-supervised method for temporal action segmentation. Our method hinges on unsupervised representation learning, which, for temporal action segmentation, poses unique challenges. Actions in untrimmed videos vary in length and have unknown labels and start/end times. Ordering of actions across videos may also vary. We propose a novel way to learn frame-wise representations from temporal convolutional networks (TCNs) by clustering input features with added time-proximity conditions and multi-resolution similarity. By merging representation learning with conventional supervised learning, we develop an "Iterative Contrast-Classify (ICC)'' semi-supervised learning scheme. With more labelled data, ICC progressively improves in performance; ICC semi-supervised learning, with 40% labelled videos, performs similarly to fully-supervised counterparts. Our ICC improves MoF by {+1.8, +5.6, +2.5}% on Breakfast, 50Salads, and GTEA respectively for 100% labelled videos.

----

## [252] JPV-Net: Joint Point-Voxel Representations for Accurate 3D Object Detection

**Authors**: *Nan Song, Tianyuan Jiang, Jian Yao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20125](https://doi.org/10.1609/aaai.v36i2.20125)

**Abstract**:

Voxel and point representations are widely applied in recent 3D object detection tasks from LiDAR point clouds. Voxel representations contribute to efficiently and rapidly locating objects, whereas point representations are capable of describing intra-object spatial relationship for detection refinement. In this work, we aim to exploit the strengths of both two representations, and present a novel two-stage detector, named Joint Point-Voxel Network (JPV-Net). Specifically, our framework is equipped with a Dual Encoders-Fusion Decoder, which consists of the dual encoders to extract voxel features of sketchy 3D scenes and point features rich in geometric context, respectively, and the Feature Propagation Fusion (FP-Fusion) decoder to attentively fuse them from coarse to fine. By making use of the advantages of these features, the refinement network can effectively eliminate false detection and provide better accuracy. Besides, to further develop the perception characteristics of voxel CNN and point backbone, we design two novel intersection-over-union (IoU) estimation modules for proposal generation and refinement, both of which can alleviate the misalignment between the localization and the classification confidence. Extensive experiments on the KITTI dataset and ONCE dataset demonstrate that our proposed JPV-Net outperforms other state-of-the-art methods with remarkable margins.

----

## [253] Fully Attentional Network for Semantic Segmentation

**Authors**: *Qi Song, Jie Li, Chenghong Li, Hao Guo, Rui Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20126](https://doi.org/10.1609/aaai.v36i2.20126)

**Abstract**:

Recent non-local self-attention methods have proven to be effective in capturing long-range dependencies for semantic segmentation. These methods usually form a similarity map of R^(CxC) (by compressing spatial dimensions) or R^(HWxHW) (by compressing channels) to describe the feature relations along either channel or spatial dimensions, where C is the number of channels, H and W are the spatial dimensions of the input feature map. However, such practices tend to condense feature dependencies along the other dimensions, hence causing attention missing, which might lead to inferior results for small/thin categories or inconsistent segmentation inside large objects. To address this problem, we propose a new approach, namely Fully Attentional Network (FLANet), to encode both spatial and channel attentions in a single similarity map while maintaining high computational efficiency. Specifically, for each channel map, our FLANet can harvest feature responses from all other channel maps, and the associated spatial positions as well, through a novel fully attentional module. Our new method has achieved state-of-the-art performance on three challenging semantic segmentation datasets, i.e., 83.6%, 46.99%, and 88.5% on the Cityscapes test set, the ADE20K validation set, and the PASCAL VOC test set, respectively.

----

## [254] Self-Supervised Object Localization with Joint Graph Partition

**Authors**: *Yukun Su, Guosheng Lin, Yun Hao, Yiwen Cao, Wenjun Wang, Qingyao Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20127](https://doi.org/10.1609/aaai.v36i2.20127)

**Abstract**:

Object localization aims to generate a tight bounding box for the target object, which is a challenging problem that has been deeply studied in recent years. Since collecting bounding-box labels is time-consuming and laborious, many researchers focus on weakly supervised object localization (WSOL). As the recent appealing self-supervised learning technique shows its powerful function in visual tasks, in this paper, we take the early attempt to explore unsupervised object localization by self-supervision. Specifically, we adopt different geometric transformations to image and utilize their parameters as pseudo labels for self-supervised learning. Then, the class-agnostic activation map (CAAM) is used to highlight the target object potential regions. However, such attention maps merely focus on the most discriminative part of the objects, which will affect the quality of the predicted bounding box. Based on the motivation that the activation maps of different transformations of the same image should be equivariant, we further design a siamese network that encodes the paired images and propose a joint graph cluster partition mechanism in an unsupervised manner to enhance the object co-occurrent regions. To validate the effectiveness of the proposed method, extensive experiments are conducted on CUB-200-2011, Stanford Cars and FGVC-Aircraft datasets. Experimental results show that our method outperforms state-of-the-art methods using the same level of supervision, even outperforms some weakly-supervised methods.

----

## [255] Correlation Field for Boosting 3D Object Detection in Structured Scenes

**Authors**: *Jianhua Sun, Haoshu Fang, Xianghui Zhu, Jiefeng Li, Cewu Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20128](https://doi.org/10.1609/aaai.v36i2.20128)

**Abstract**:

Data augmentation is an efficient way to elevate 3D object detection performance. In this paper, we propose a simple but effective online crop-and-paste data augmentation pipeline for structured 3D point cloud scenes, named CorrelaBoost. Observing that 3D objects should have reasonable relative positions in a structured scene because of the objects' functionalities and natural relationships, we express this correlation as a kind of interactive force. An energy field called Correlation Field can be calculated correspondingly across the whole 3D space. According to the Correlation Field, we propose two data augmentation strategies to explore highly congruent positions that a designated object may be pasted to: 1) Category Consistent Exchanging and 2) Energy Optimized Transformation. We conduct exhaustive experiments on various popular benchmarks with different detection frameworks and the results illustrate that our method brings huge free-lunch improvement and significantly outperforms state-of-the-art approaches in terms of data augmentation. It is worth noting that the performance of VoteNet with mAP@0.5 is improved by 7.7 on ScanNetV2 dataset and 5.0 on SUN RGB-D dataset. Our method is simple to implement and increases few computational overhead.

----

## [256] Boost Supervised Pretraining for Visual Transfer Learning: Implications of Self-Supervised Contrastive Representation Learning

**Authors**: *Jinghan Sun, Dong Wei, Kai Ma, Liansheng Wang, Yefeng Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20129](https://doi.org/10.1609/aaai.v36i2.20129)

**Abstract**:

Unsupervised pretraining based on contrastive learning has made significant progress recently and showed comparable or even superior transfer learning performance to traditional supervised pretraining on various tasks. In this work, we first empirically investigate when and why unsupervised pretraining surpasses supervised counterparts for image classification tasks with a series of control experiments. Besides the commonly used accuracy, we further analyze the results qualitatively with the class activation maps and assess the learned representations quantitatively with the representation entropy and uniformity. Our core finding is that it is the amount of information effectively perceived by the learning model that is crucial to transfer learning, instead of absolute size of the dataset. Based on this finding, we propose Classification Activation Map guided contrastive (CAMtrast) learning which better utilizes the label supervsion to strengthen supervised pretraining, by making the networks perceive more information from the training images. CAMtrast is evaluated with three fundamental visual learning tasks: image recognition, object detection, and semantic segmentation, on various public datasets. Experimental results show that our CAMtrast effectively improves the performance of supervised pretraining, and that its performance is superior to both unsupervised counterparts and a recent related work which similarly attempted improving supervised pretraining.

----

## [257] Dual Contrastive Learning for General Face Forgery Detection

**Authors**: *Ke Sun, Taiping Yao, Shen Chen, Shouhong Ding, Jilin Li, Rongrong Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20130](https://doi.org/10.1609/aaai.v36i2.20130)

**Abstract**:

With various facial manipulation techniques arising, face forgery detection has drawn growing attention due to security concerns. Previous works always formulate face forgery detection as a classiﬁcation problem based on cross-entropy loss, which emphasizes category-level differences rather than the essential discrepancies between real and fake faces, limiting model generalization in unseen domains. To address this issue, we propose a novel face forgery detection framework, named Dual Contrastive Learning (DCL), which specially constructs positive and negative paired data and performs designed contrastive learning at different granularities to learn generalized feature representation. Concretely, combined with the hard sample selection strategy, Inter-Instance Contrastive Learning (Inter-ICL) is ﬁrst proposed to promote task-related discriminative features learning by especially constructing instance pairs. Moreover, to further explore the essential discrepancies, Intra-Instance Contrastive Learning (Intra-ICL) is introduced to focus on the local content inconsistencies prevalent in the forged faces by constructing local region pairs inside instances. Extensive experiments and visualizations on several datasets demonstrate the generalization of our method against the state-of-the-art competitors. Our Code is available at https://github.com/Tencent/TFace.git.

----

## [258] SSAT: A Symmetric Semantic-Aware Transformer Network for Makeup Transfer and Removal

**Authors**: *Zhaoyang Sun, Yaxiong Chen, Shengwu Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20131](https://doi.org/10.1609/aaai.v36i2.20131)

**Abstract**:

Makeup transfer is not only to extract the makeup style of the reference image, but also to render the makeup style to the semantic corresponding position of the target image. However, most existing methods focus on the former and ignore the latter, resulting in a failure to achieve desired results. To solve the above problems, we propose a unified Symmetric Semantic-Aware Transformer (SSAT) network, which incorporates semantic correspondence learning to realize makeup transfer and removal simultaneously. In SSAT, a novel Symmetric Semantic Corresponding Feature Transfer (SSCFT) module and a weakly supervised semantic loss are proposed to model and facilitate the establishment of accurate semantic correspondence. In the generation process, the extracted makeup features are spatially distorted by SSCFT to achieve semantic alignment with the target image, then the distorted makeup features are combined with unmodified makeup irrelevant features to produce the final result. Experiments show that our method obtains more visually accurate makeup transfer results, and user study in comparison with other state-of-the-art makeup transfer methods reflects the superiority of our method. Besides, we verify the robustness of the proposed method in the difference of expression and pose, object occlusion scenes, and extend it to video makeup transfer.

----

## [259] Adversarial Bone Length Attack on Action Recognition

**Authors**: *Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20132](https://doi.org/10.1609/aaai.v36i2.20132)

**Abstract**:

Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.

----

## [260] Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?

**Authors**: *Chuanxin Tang, Yucheng Zhao, Guangting Wang, Chong Luo, Wenxuan Xie, Wenjun Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20133](https://doi.org/10.1609/aaai.v36i2.20133)

**Abstract**:

Transformers have sprung up in the field of computer vision. In this work, we explore whether the core self-attention module in Transformer is the key to achieving excellent performance in image recognition. To this end, we build an attention-free network called sMLPNet based on the existing MLP-based vision models. Specifically, we replace the MLP module in the token-mixing step with a novel sparse MLP (sMLP) module. For 2D image tokens, sMLP applies 1D MLP along the axial directions and the parameters are shared among rows or columns. By sparse connection and weight sharing, sMLP module significantly reduces the number of model parameters and computational complexity, avoiding the common over-fitting problem that plagues the performance of MLP-like models. When only trained on the ImageNet-1K dataset, the proposed sMLPNet achieves 81.9% top-1 accuracy with only 24M parameters, which is much better than most CNNs and vision Transformers under the same model size constraint. When scaling up to 66M parameters, sMLPNet achieves 83.4% top-1 accuracy, which is on par with the state-of-the-art Swin Transformer. The success of sMLPNet suggests that the self-attention mechanism is not necessarily a silver bullet in computer vision. The code and models are publicly available at https://github.com/microsoft/SPACH.

----

## [261] Not All Voxels Are Equal: Semantic Scene Completion from the Point-Voxel Perspective

**Authors**: *Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, Gang Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20134](https://doi.org/10.1609/aaai.v36i2.20134)

**Abstract**:

We revisit Semantic Scene Completion (SSC), a useful task to predict the semantic and occupancy representation of 3D scenes, in this paper. A number of methods for this task are always based on voxelized scene representations. Although voxel representations keep local structures of the scene, these methods suffer from heavy computation redundancy due to the existence of visible empty voxels when the network goes deeper. To address this dilemma, we propose our novel point-voxel aggregation network for this task. We first transfer the voxelized scenes to point clouds by removing these visible empty voxels and adopt a deep point stream to capture semantic information from the scene efficiently. Meanwhile, a light-weight voxel stream containing only two 3D convolution layers preserves local structures of the voxelized scenes. Furthermore, we design an anisotropic voxel aggregation operator to fuse the structure details from the voxel stream into the point stream, and a semantic-aware propagation module to enhance the up-sampling process in the point stream by semantic labels. We demonstrate that our model surpasses state-of-the-arts on two benchmarks by a large margin, with only the depth images as input.

----

## [262] Transfer Learning for Color Constancy via Statistic Perspective

**Authors**: *Yuxiang Tang, Xuejing Kang, Chunxiao Li, Zhaowen Lin, Anlong Ming*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20135](https://doi.org/10.1609/aaai.v36i2.20135)

**Abstract**:

Color Constancy aims to correct image color casts caused by scene illumination. Recently, although the deep learning approaches have remarkably improved on single-camera data, these models still suffer from the seriously insufficient data problem, resulting in shallow model capacity and degradation in multi-camera settings. In this paper, to alleviate this problem, we present a Transfer Learning Color Constancy (TLCC) method that leverages cross-camera RAW data and massive unlabeled sRGB data to support training. Specifically, TLCC consists of the Statistic Estimation Scheme (SE-Scheme) and Color-Guided Adaption Branch (CGA-Branch). SE-Scheme builds a statistic perspective to map the camera-related illumination labels into camera-agnostic form and produce pseudo labels for sRGB data, which greatly expands data for joint training. Then, CGA-Branch further promotes efficient transfer learning from sRGB to RAW data by extracting color information to regularize the backbone's features adaptively. Experimental results show the TLCC has overcome the data limitation and model degradation, outperforming the state-of-the-art performance on popular benchmarks. Moreover, the experiments also prove the TLCC is capable of learning new scenes information from sRGB data to improve accuracy on the RAW images with similar scenes.

----

## [263] TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval

**Authors**: *Jialin Tian, Xing Xu, Fumin Shen, Yang Yang, Heng Tao Shen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20136](https://doi.org/10.1609/aaai.v36i2.20136)

**Abstract**:

In this paper, we study the zero-shot sketch-based image retrieval (ZS-SBIR) task, which retrieves natural images related to sketch queries from unseen categories. In the literature, convolutional neural networks (CNNs) have become the de-facto standard and they are either trained end-to-end or used to extract pre-trained features for images and sketches. However, CNNs are limited in modeling the global structural information of objects due to the intrinsic locality of convolution operations. To this end, we propose a Transformer-based approach called Three-Way Vision Transformer (TVT) to leverage the ability of Vision Transformer (ViT) to model global contexts due to the global self-attention mechanism. Going beyond simply applying ViT to this task, we propose a token-based strategy of adding fusion and distillation tokens and making them complementary to each other. Specifically, we integrate three ViTs, which are pre-trained on data of each modality, into a three-way pipeline through the processes of distillation and multi-modal hypersphere learning. The distillation process is proposed to supervise fusion ViT (ViT with an extra fusion token) with soft targets from modality-specific ViTs, which prevents fusion ViT from catastrophic forgetting. Furthermore, our method learns a multi-modal hypersphere by performing inter- and intra-modal alignment without loss of uniformity, which aims to bridge the modal gap between modalities of sketch and image and avoid the collapse in dimensions. Extensive experiments on three benchmark datasets, i.e., Sketchy, TU-Berlin, and QuickDraw, demonstrate the superiority of our TVT method over the state-of-the-art ZS-SBIR methods.

----

## [264] GuidedMix-Net: Semi-supervised Semantic Segmentation by Using Labeled Images as Reference

**Authors**: *Peng Tu, Yawen Huang, Feng Zheng, Zhenyu He, Liujuan Cao, Ling Shao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20137](https://doi.org/10.1609/aaai.v36i2.20137)

**Abstract**:

Semi-supervised learning is a challenging problem which aims to construct a model by learning from limited labeled examples.
Numerous methods for this task focus on utilizing the predictions of unlabeled instances consistency alone to regularize networks.
However, treating labeled and unlabeled data separately often leads to the discarding of mass prior knowledge learned from the labeled examples. 
In this paper, we propose a novel method for semi-supervised semantic segmentation named GuidedMix-Net, by leveraging labeled information to guide the learning of unlabeled instances.
Specifically, GuidedMix-Net employs three operations: 1) interpolation of similar labeled-unlabeled image pairs; 2) transfer of mutual information; 3) generalization of pseudo masks.
It enables segmentation models can learning the higher-quality pseudo masks of unlabeled data by transfer the knowledge from labeled samples to unlabeled data.
Along with supervised learning for labeled data, the prediction of unlabeled data is jointly learned with the generated pseudo masks from the mixed data. 
Extensive experiments on PASCAL VOC 2012, and Cityscapes demonstrate the effectiveness of our GuidedMix-Net, which achieves competitive segmentation accuracy and significantly improves the mIoU over 7$\%$ compared to previous approaches.

----

## [265] MTLDesc: Looking Wider to Describe Better

**Authors**: *Changwei Wang, Rongtao Xu, Yuyang Zhang, Shibiao Xu, Weiliang Meng, Bin Fan, Xiaopeng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20138](https://doi.org/10.1609/aaai.v36i2.20138)

**Abstract**:

Limited by the locality of convolutional neural networks, most existing local features description methods only learn local descriptors with local information and lack awareness of global and surrounding spatial context. In this work, we focus on making local descriptors ``look wider to describe better'' by learning local Descriptors with More Than Local information (MTLDesc). Specifically, we resort to context augmentation and spatial attention mechanism to make the descriptors obtain non-local awareness. First, Adaptive Global Context Augmented Module and Diverse Local Context Augmented Module are proposed to construct robust local descriptors with context information from global to local. Second, we propose the Consistent Attention Weighted Triplet Loss to leverage spatial attention awareness in both optimization and matching of local descriptors. Third, Local Features Detection with Feature Pyramid is proposed to obtain more stable and accurate keypoints localization. With the above innovations, the performance of the proposed MTLDesc significantly surpasses the current state-of-the-art local descriptors on HPatches, Aachen Day-Night localization and InLoc indoor localization benchmarks. Our code is available at https://github.com/vignywang/MTLDesc.

----

## [266] Active Boundary Loss for Semantic Segmentation

**Authors**: *Chi Wang, Yunke Zhang, Miaomiao Cui, Peiran Ren, Yin Yang, Xuansong Xie, Xian-Sheng Hua, Hujun Bao, Weiwei Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20139](https://doi.org/10.1609/aaai.v36i2.20139)

**Abstract**:

This paper proposes a novel active boundary loss for semantic segmentation. It can progressively encourage the alignment between predicted boundaries and ground-truth boundaries during end-to-end training, which is not explicitly enforced in commonly used cross-entropy loss. Based on the predicted boundaries detected from the segmentation results using current network parameters, we formulate the boundary alignment problem as a differentiable direction vector prediction problem to guide the movement of predicted boundaries in each iteration. Our loss is model-agnostic and can be plugged in to the training of segmentation networks to improve the boundary details. Experimental results show that training with the active boundary loss can effectively improve the boundary F-score and mean Intersection-over-Union on challenging image and video object segmentation datasets.

----

## [267] Online-Updated High-Order Collaborative Networks for Single Image Deraining

**Authors**: *Cong Wang, Jinshan Pan, Xiao-Ming Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20140](https://doi.org/10.1609/aaai.v36i2.20140)

**Abstract**:

Single image deraining is an important and challenging task for some downstream artificial intelligence applications such as video surveillance and self-driving systems.
Most of the existing deep-learning-based methods constrain the network to generate derained images but few of them explore features from intermediate layers, different levels, and different modules which are beneficial for rain streaks removal.
In this paper, we propose a high-order collaborative network with multi-scale compact constraints and a bidirectional scale-content similarity mining module to exploit features from deep networks externally and internally for rain streaks removal.
Externally, we design a deraining framework with three sub-networks trained in a collaborative manner, where the bottom network transmits intermediate features to the middle network which also receives shallower rainy features from the top network and sends back features to the bottom network.
Internally, we enforce multi-scale compact constraints on the intermediate layers of deep networks to learn useful features via a Laplacian pyramid.
Further, we develop a bidirectional scale-content similarity mining module to explore features at different scales in a down-to-up and up-to-down manner.
To improve the model performance on real-world images, we propose an online-update learning approach, which uses real-world rainy images to fine-tune the network and update the deraining results in a self-supervised manner.
Extensive experiments demonstrate that our proposed method performs favorably against eleven state-of-the-art methods on five public synthetic datasets and one real-world dataset.

----

## [268] FCA: Learning a 3D Full-Coverage Vehicle Camouflage for Multi-View Physical Adversarial Attack

**Authors**: *Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Zhiqiang Gong, Xiaoya Zhang, Wen Yao, Xiaoqian Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20141](https://doi.org/10.1609/aaai.v36i2.20141)

**Abstract**:

Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle’s surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo-realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors.

----

## [269] When Shift Operation Meets Vision Transformer: An Extremely Simple Alternative to Attention Mechanism

**Authors**: *Guangting Wang, Yucheng Zhao, Chuanxin Tang, Chong Luo, Wenjun Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20142](https://doi.org/10.1609/aaai.v36i2.20142)

**Abstract**:

Attention mechanism has been widely believed as the key to success of vision transformers (ViTs), since it provides a flexible and powerful way to model spatial relationships. However, is the attention mechanism truly an indispensable part of ViT? Can it be replaced by some other alternatives? To demystify the role of attention mechanism, we simplify it into an extremely simple case: ZERO FLOP and ZERO parameter. Concretely, we revisit the shift operation. It does not contain any parameter or arithmetic calculation. The only operation is to exchange a small portion of the channels between neighboring features. Based on this simple operation, we construct a new backbone network, namely ShiftViT, where the attention layers in ViT are substituted by shift operations. Surprisingly, ShiftViT works quite well in several mainstream tasks, e.g., classification, detection, and segmentation. The performance is on par with or even better than the strong baseline Swin Transformer. These results suggest that the attention mechanism might not be the vital factor that makes ViT successful. It can be even replaced by a zero-parameter operation. We should pay more attentions to the remaining parts of ViT in the future work. Code is available at github.com/microsoft/SPACH.

----

## [270] Self-Supervised Representation Learning Framework for Remote Physiological Measurement Using Spatiotemporal Augmentation Loss

**Authors**: *Hao Wang, Euijoon Ahn, Jinman Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i2.20143](https://doi.org/10.1609/aaai.v36i2.20143)

**Abstract**:

Recent advances in supervised deep learning methods are enabling remote measurements of photoplethysmography-based physiological signals using facial videos. The performance of these supervised methods, however, are dependent on the availability of large labelled data. Contrastive learning as a self-supervised method has recently achieved state-of-the-art performances in learning representative data features by maximising mutual information between different augmented views. However, existing data augmentation techniques for contrastive learning are not designed to learn physiological signals from videos and often fail when there are complicated noise and subtle and periodic colour/shape variations between video frames. To address these problems, we present a novel self-supervised spatiotemporal learning framework for remote physiological signal representation learning, where there is a lack of labelled training data. Firstly, we propose a landmark-based spatial augmentation that splits the face into several informative parts based on the Shafer’s dichromatic reﬂection model to characterise subtle skin colour fluctuations. We also formulate a sparsity-based temporal augmentation exploiting Nyquist–Shannon sampling theorem to effectively capture periodic temporal changes by modelling physiological signal features. Furthermore, we introduce a constrained spatiotemporal loss which generates pseudo-labels for augmented video clips. It is used to regulate the training process and handle complicated noise. We evaluated our framework on 3 public datasets and demonstrated superior performances than other self-supervised methods and achieved competitive accuracy compared to the state-of-the-art supervised methods. Code is available at https://github.com/Dylan-H-Wang/SLF-RPM.

----

## [271] UCTransNet: Rethinking the Skip Connections in U-Net from a Channel-Wise Perspective with Transformer

**Authors**: *Haonan Wang, Peng Cao, Jiaqi Wang, Osmar R. Zaïane*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20144](https://doi.org/10.1609/aaai.v36i3.20144)

**Abstract**:

Most recent semantic segmentation methods adopt a U-Net framework with an encoder-decoder architecture. It is still challenging for U-Net with a simple skip connection scheme to model the global multi-scale context: 1) Not each skip connection setting is effective due to the issue of incompatible feature sets of encoder and decoder stage, even some skip connection negatively influence the segmentation performance; 2) The original U-Net is worse than the one without any skip connection on some datasets. Based on our findings, we propose a new segmentation framework, named UCTransNet (with a proposed CTrans module in U-Net), from the channel perspective with attention mechanism. Specifically, the CTrans (Channel Transformer) module is an alternate of the U-Net skip connections, which consists of a sub-module to conduct the multi-scale Channel Cross fusion with Transformer (named CCT) and a sub-module Channel-wise Cross-Attention (named CCA) to guide the fused multi-scale channel-wise information to effectively connect to the decoder features for eliminating the ambiguity. Hence, the proposed connection consisting of the CCT and CCA is able to replace the original skip connection to solve the semantic gaps for an accurate automatic medical image segmentation. The experimental results suggest that our UCTransNet produces more precise segmentation performance and achieves consistent improvements over the state-of-the-art for semantic segmentation across different datasets and conventional architectures involving transformer or U-shaped framework. Code: https://github.com/McGregorWwww/UCTransNet.

----

## [272] Renovate Yourself: Calibrating Feature Representation of Misclassified Pixels for Semantic Segmentation

**Authors**: *Hualiang Wang, Huanpeng Chu, Siming Fu, Zuozhu Liu, Haoji Hu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20145](https://doi.org/10.1609/aaai.v36i3.20145)

**Abstract**:

Existing image semantic segmentation methods favor learning consistent representations by extracting long-range contextual features with the attention, multi-scale, or graph aggregation strategies. These methods usually treat the misclassified and correctly classified pixels equally, hence misleading the optimization process and causing inconsistent intra-class pixel feature representations in the embedding space during learning. In this paper, we propose the auxiliary representation calibration head (RCH), which consists of the image decoupling, prototype clustering, error calibration modules and a metric loss function, to calibrate these error-prone feature representations for better intra-class consistency and segmentation performance. RCH could be incorporated into the hidden layers, trained together with the segmentation networks, and decoupled in the inference stage without additional parameters. Experimental results show that our method could significantly boost the performance of current segmentation methods on multiple datasets (e.g., we outperform the original HRNet and OCRNet by 1.1% and 0.9% mIoU on the Cityscapes test set). Codes are available at https://github.com/VipaiLab/RCH.

----

## [273] Separated Contrastive Learning for Organ-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotation

**Authors**: *Jiacheng Wang, Xiaomeng Li, Yiming Han, Jing Qin, Liansheng Wang, Qichao Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20146](https://doi.org/10.1609/aaai.v36i3.20146)

**Abstract**:

Automatic delineation of organ-at-risk (OAR) and gross-tumor-volume (GTV) is of great significance for radiotherapy planning. However, it is a challenging task to learn powerful representations for accurate delineation under limited pixel (voxel)-wise annotations. Contrastive learning at pixel-level can alleviate the dependency on annotations by learning dense representations from unlabeled data. Recent studies in this direction design various contrastive losses on the feature maps, to yield discriminative features for each pixel in the map. However, pixels in the same map inevitably share semantics to be closer than they actually are, which may affect the discrimination of pixels in the same map and lead to the unfair comparison to pixels in other maps. To address these issues, we propose a separated region-level contrastive learning scheme, namely SepaReg, the core of which is to separate each image into regions and encode each region separately. Specifically, SepaReg comprises two components: a structure-aware image separation (SIS) module and an intra- and inter-organ distillation (IID) module. The SIS is proposed to operate on the image set to rebuild a region set under the guidance of structural information. The inter-organ representation will be learned from this set via typical contrastive losses cross regions. On the other hand, the IID is proposed to tackle the quantity imbalance in the region set as tiny organs may produce fewer regions, by exploiting intra-organ representations. We conducted extensive experiments to evaluate the proposed model on a public dataset and two private datasets. The experimental results demonstrate the effectiveness of the proposed model, consistently achieving better performance than state-of-the-art approaches. Code is available at https://github.com/jcwang123/Separate_CL.

----

## [274] Contrastive Quantization with Code Memory for Unsupervised Image Retrieval

**Authors**: *Jinpeng Wang, Ziyun Zeng, Bin Chen, Tao Dai, Shu-Tao Xia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20147](https://doi.org/10.1609/aaai.v36i3.20147)

**Abstract**:

The high efficiency in computation and storage makes hashing (including binary hashing and quantization) a common strategy in large-scale retrieval systems. To alleviate the reliance on expensive annotations, unsupervised deep hashing becomes an important research problem. This paper provides a novel solution to unsupervised deep quantization, namely Contrastive Quantization with Code Memory (MeCoQ). Different from existing reconstruction-based strategies, we learn unsupervised binary descriptors by contrastive learning, which can better capture discriminative visual semantics. Besides, we uncover that codeword diversity regularization is critical to prevent contrastive learning-based quantization from model degeneration. Moreover, we introduce a novel quantization code memory module that boosts contrastive learning with lower feature drift than conventional feature memories. Extensive experiments on benchmark datasets show that MeCoQ outperforms state-of-the-art methods. Code and configurations are publicly released.

----

## [275] Learning Temporally and Semantically Consistent Unpaired Video-to-Video Translation through Pseudo-Supervision from Synthetic Optical Flow

**Authors**: *Kaihong Wang, Kumar Akash, Teruhisa Misu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20148](https://doi.org/10.1609/aaai.v36i3.20148)

**Abstract**:

Unpaired video-to-video translation aims to translate videos between a source and a target domain without the need of paired training data, making it more feasible for real applications. Unfortunately, the translated videos generally suffer from temporal and semantic inconsistency. To address this, many existing works adopt spatiotemporal consistency constraints incorporating temporal information based on motion estimation. However, the inaccuracies in the estimation of motion deteriorate the quality of the guidance towards spatiotemporal consistency, which leads to unstable translation. In this work, we propose a novel paradigm that regularizes the spatiotemporal consistency by synthesizing motions in input videos with the generated optical flow instead of estimating them. Therefore, the synthetic motion can be applied in the regularization paradigm to keep motions consistent across domains without the risk of errors in motion estimation. Thereafter, we utilize our unsupervised recycle and unsupervised spatial loss, guided by the pseudo-supervision provided by the synthetic optical flow, to accurately enforce spatiotemporal consistency in both domains. Experiments show that our method is versatile in various scenarios and achieves state-of-the-art performance in generating temporally and semantically consistent videos. Code is available at: https://github.com/wangkaihong/Unsup_Recycle_GAN/.

----

## [276] Cross-Dataset Collaborative Learning for Semantic Segmentation in Autonomous Driving

**Authors**: *Li Wang, Dong Li, Han Liu, Jinzhang Peng, Lu Tian, Yi Shan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20149](https://doi.org/10.1609/aaai.v36i3.20149)

**Abstract**:

Semantic segmentation is an important task for scene understanding in self-driving cars and robotics, which aims to assign dense labels for all pixels in the image. Existing work typically improves semantic segmentation performance by exploring different network architectures on a target dataset. Little attention has been paid to build a unified system by simultaneously learning from multiple datasets due to the inherent distribution shift across different datasets. In this paper, we propose a simple, flexible, and general method for semantic segmentation, termed Cross-Dataset Collaborative Learning (CDCL). Our goal is to train a unified model for improving the performance in each dataset by leveraging information from all the datasets. Specifically, we first introduce a family of Dataset-Aware Blocks (DAB) as the fundamental computing units of the network, which help capture homogeneous convolutional representations and heterogeneous statistics across different datasets. Second, we present a Dataset Alternation Training (DAT) mechanism to facilitate the collaborative optimization procedure. We conduct extensive evaluations on diverse semantic segmentation datasets for autonomous driving. Experiments demonstrate that our method consistently achieves notable improvements over prior single-dataset and cross-dataset training methods without introducing extra FLOPs. Particularly, with the same architecture of PSPNet (ResNet-18), our method outperforms the single-dataset baseline by 5.65\%, 6.57\%, and 5.79\% mIoU on the validation sets of Cityscapes, BDD100K, CamVid, respectively. We also apply CDCL for point cloud 3D semantic segmentation and achieve improved performance, which further validates the superiority and generality of our method. Code and models will be released.

----

## [277] Scaled ReLU Matters for Training Vision Transformers

**Authors**: *Pichao Wang, Xue Wang, Hao Luo, Jingkai Zhou, Zhipeng Zhou, Fan Wang, Hao Li, Rong Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20150](https://doi.org/10.1609/aaai.v36i3.20150)

**Abstract**:

Vision transformers (ViTs) have been an alternative design paradigm to convolutional neural networks (CNNs). However, the training of ViTs is much harder than CNNs, as it is sensitive to the training parameters, such as learning rate, optimizer and warmup epoch. The reasons for training difficulty are empirically analysed in the paper Early Convolutions Help Transformers See Better, and the authors conjecture that the issue lies with the patchify-stem of ViT models. In this paper, we further investigate this problem and extend the above conclusion: only early convolutions do not help for stable training, but the scaled ReLU operation in the convolutional stem (conv-stem) matters. We verify, both theoretically and empirically, that scaled ReLU in conv-stem not only improves training stabilization, but also increases the diversity of patch tokens, thus boosting peak performance with a large margin via adding few parameters and flops. In addition, extensive experiments are conducted to demonstrate that previous ViTs are far from being well trained, further showing that ViTs have great potential to be a better substitute of CNNs.

----

## [278] CQA-Face: Contrastive Quality-Aware Attentions for Face Recognition

**Authors**: *Qiangchang Wang, Guodong Guo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20151](https://doi.org/10.1609/aaai.v36i3.20151)

**Abstract**:

Few existing face recognition (FR) models take local representations into account. Although some works achieved this by extracting features on cropped parts around face landmarks, landmark detection may be inaccurate or even fail in some extreme cases. Recently, without relying on landmarks, attention-based networks can focus on useful parts automatically. However, there are two issues: 1) It is noticed that these approaches focus on few facial parts, while missing other potentially discriminative regions. This can cause performance drops when emphasized facial parts are invisible under heavy occlusions (e.g. face masks) or large pose variations; 2) Different facial parts may appear at various quality caused by occlusion, blur, or illumination changes. In this paper, we propose contrastive quality-aware attentions, called CQA-Face, to address these two issues. First, a Contrastive Attention Learning (CAL) module is proposed, pushing models to explore comprehensive facial parts. Consequently, more useful parts can help identification if some facial parts are invisible. Second, a Quality-Aware Network (QAN) is developed to emphasize important regions and suppress noisy parts in a global scope. Thus, our CQA-Face model is developed by integrating the CAL with QAN, which extracts diverse quality-aware local representations. It outperforms the state-of-the-art methods on several benchmarks, demonstrating its effectiveness and usefulness.

----

## [279] Category-Specific Nuance Exploration Network for Fine-Grained Object Retrieval

**Authors**: *Shijie Wang, Zhihui Wang, Haojie Li, Wanli Ouyang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20152](https://doi.org/10.1609/aaai.v36i3.20152)

**Abstract**:

Employing additional prior knowledge to model local features as a final fine-grained object representation has become a trend for fine-grained object retrieval (FGOR). A potential limitation of these methods is that they only focus on common parts across the dataset (e.g. head, body or even leg) by introducing additional prior knowledge, but the retrieval of a fine-grained object may rely on category-specific nuances that contribute to category prediction. To handle this limitation, we propose an end-to-end Category-specific Nuance Exploration Network (CNENet) that elaborately discovers category-specific nuances that contribute to category prediction, and semantically aligns these nuances grouped by subcategory without any additional prior knowledge, to directly emphasize the discrepancy among subcategories. Specifically, we design a Nuance Modelling Module that adaptively predicts a group of category-specific response (CARE) maps via implicitly digging into category-specific nuances, specifying the locations and scales for category-specific nuances. Upon this, two nuance regularizations are proposed: 1) semantic discrete loss that forces each CARE map to attend to different spatial regions to capture diverse nuances; 2) semantic alignment loss that constructs a consistent semantic correspondence for each CARE map of the same order with the same subcategory via guaranteeing each instance and its transformed counterpart to be spatially aligned. Moreover, we propose a Nuance Expansion Module, which exploits context appearance information of discovered nuances and refines the prediction of current nuance by its similar neighbors, leading to further improvement on nuance consistency and completeness. Extensive experiments validate that our CNENet consistently yields the best performance under the same settings against most competitive approaches on CUB Birds, Stanford Cars, and FGVC Aircraft datasets.

----

## [280] Detail-Preserving Transformer for Light Field Image Super-resolution

**Authors**: *Shunzhou Wang, Tianfei Zhou, Yao Lu, Huijun Di*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20153](https://doi.org/10.1609/aaai.v36i3.20153)

**Abstract**:

Recently, numerous algorithms have been developed to tackle the problem of light field super-resolution (LFSR), i.e., super-resolving low-resolution light fields to gain high-resolution views. Despite delivering encouraging results, these approaches are all convolution-based, and are naturally weak in global relation modeling of sub-aperture images necessarily to characterize the inherent structure of light fields. In this paper, we put forth a novel formulation built upon Transformers, by treating LFSR as a sequence-to-sequence reconstruction task. In particular, our model regards sub-aperture images of each vertical or horizontal angular view as a sequence, and establishes long-range geometric dependencies within each sequence via a spatial-angular locally-enhanced self-attention layer, which maintains the locality of each sub-aperture image as well. Additionally, to better recover image details, we propose a detail-preserving Transformer (termed as DPT), by leveraging gradient maps of light field to guide the sequence learning. DPT consists of two branches, with each associated with a Transformer for learning from an original or gradient image sequence. The two branches are finally fused to obtain comprehensive feature representations for reconstruction. Evaluations are conducted on a number of light field datasets, including real-world scenes and synthetic data. The proposed method achieves superior performance comparing with other state-of-the-art schemes. Our code is publicly available at: https://github.com/BITszwang/DPT.

----

## [281] One-Shot Talking Face Generation from Single-Speaker Audio-Visual Correlation Learning

**Authors**: *Suzhen Wang, Lincheng Li, Yu Ding, Xin Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20154](https://doi.org/10.1609/aaai.v36i3.20154)

**Abstract**:

Audio-driven one-shot talking face generation methods are usually trained on video resources of various persons. However, their created videos often suffer unnatural mouth shapes and asynchronous lips because those methods struggle to learn a consistent speech style from different speakers. We observe that it would be much easier to learn a consistent speech style from a specific speaker, which leads to authentic mouth movements. Hence, we propose a novel one-shot talking face generation framework by exploring consistent correlations between audio and visual motions from a specific speaker and then transferring audio-driven motion fields to a reference image. Specifically, we develop an Audio-Visual Correlation Transformer (AVCT) that aims to infer talking motions represented by keypoint based dense motion fields from an input audio. In particular, considering audio may come from different identities in deployment, we incorporate phonemes to represent audio signals. In this manner, our AVCT can inherently generalize to audio spoken by other identities. Moreover, as face keypoints are used to represent speakers, AVCT is agnostic against appearances of the training speaker, and thus allows us to manipulate face images of different identities readily. Considering different face shapes lead to different motions, a motion field transfer module is exploited to reduce the audio-driven dense motion field gap between the training identity and the one-shot reference. Once we obtained the dense motion field of the reference image, we employ an image renderer to generate its talking face videos from an audio clip. Thanks to our learned consistent speaking style, our method generates authentic mouth shapes and vivid movements. Extensive experiments demonstrate that our synthesized videos outperform the state-of-the-art in terms of visual quality and lip-sync.

----

## [282] Pose-Guided Feature Disentangling for Occluded Person Re-identification Based on Transformer

**Authors**: *Tao Wang, Hong Liu, Pinhao Song, Tianyu Guo, Wei Shi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20155](https://doi.org/10.1609/aaai.v36i3.20155)

**Abstract**:

Occluded person re-identification is a challenging task as human body parts could be occluded by some obstacles (e.g. trees, cars, and pedestrians) in certain scenes. Some existing pose-guided methods solve this problem by aligning body parts according to graph matching, but these graph-based methods are not intuitive and complicated. Therefore, we propose a transformer-based Pose-guided Feature Disentangling (PFD) method by utilizing pose information to clearly disentangle semantic components (e.g. human body or joint parts) and selectively match non-occluded parts correspondingly. First, Vision Transformer (ViT) is used to extract the patch features with its strong capability. Second, to preliminarily disentangle the pose information from patch information, the matching and distributing mechanism is leveraged in Pose-guided Feature Aggregation (PFA) module. Third, a set of learnable semantic views are introduced in transformer decoder to implicitly enhance the disentangled body part features. However, those semantic views are not guaranteed to be related to the body without additional supervision. Therefore, Pose-View Matching (PVM) module is proposed to explicitly match visible body parts and automatically separate occlusion features. Fourth, to better prevent the interference of occlusions, we design a Pose-guided Push Loss to emphasize the features of visible body parts. Extensive experiments over five challenging datasets for two tasks (occluded and holistic Re-ID) demonstrate that our proposed PFD is superior promising, which performs favorably against state-of-the-art methods. Code is available at https://github.com/WangTaoAs/PFD_Net

----

## [283] FFNet: Frequency Fusion Network for Semantic Scene Completion

**Authors**: *Xuzhi Wang, Di Lin, Liang Wan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20156](https://doi.org/10.1609/aaai.v36i3.20156)

**Abstract**:

Semantic scene completion (SSC) requires the estimation of the 3D geometric occupancies of objects in the scene, along with the object categories. Currently, many methods employ RGB-D images to capture the geometric and semantic information of objects. These methods use simple but popular spatial- and channel-wise operations, which fuse the information of RGB and depth data. Yet, they ignore the large discrepancy of RGB-D data and the uncertainty measurements of depth data. To solve this problem, we propose the Frequency Fusion Network (FFNet), a novel method for boosting semantic scene completion by better utilizing RGB-D data. FFNet explicitly correlates the RGB-D data in the frequency domain, different from the features directly extracted by the convolution operation. Then, the network uses the correlated information to guide the feature learning from the RG- B and depth images, respectively. Moreover, FFNet accounts for the properties of different frequency components of RGB- D features. It has a learnable elliptical mask to decompose the features learned from the RGB and depth images, attending to various frequencies to facilitate the correlation process of RGB-D data. We evaluate FFNet intensively on the public SSC benchmarks, where FFNet surpasses the state-of- the-art methods. The code package of FFNet is available at https://github.com/alanWXZ/FFNet.

----

## [284] Privacy-Preserving Face Recognition in the Frequency Domain

**Authors**: *Yinggui Wang, Jian Liu, Man Luo, Le Yang, Li Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20157](https://doi.org/10.1609/aaai.v36i3.20157)

**Abstract**:

Some applications may require performing face recognition (FR) on third-party servers, which could be accessed by attackers 
 with malicious intents to compromise the privacy of users’ face information. This paper advocates a practical privacy-preserving FR scheme without key management realized in the frequency domain. The new scheme first collects the components of the same frequency from different blocks of a face image to form component channels. Only part of the channels are retained and fed into the analysis network that performs an interpretable privacy-accuracy trade-off analysis to identify channels important for face image visualization but not crucial for maintaining high FR accuracy. For this purpose, the loss function of the analysis network consists of the empirical FR error loss and a face visualization penalty term, and the network is trained in an end-to-end manner. We find that with the developed analysis network, more than 94% of the image energy can be dropped while the face recognition accuracy stays almost undegraded. In order to further protect the remaining frequency components, we propose a fast masking method. Effectiveness of the new scheme in removing the visual information of face images while maintaining their distinguishability is validated over several large face datasets. Results show that the proposed scheme achieves a recognition performance and inference time comparable to ArcFace operating on original face images directly.

----

## [285] Anchor DETR: Query Design for Transformer-Based Detector

**Authors**: *Yingming Wang, Xiangyu Zhang, Tong Yang, Jian Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20158](https://doi.org/10.1609/aaai.v36i3.20158)

**Abstract**:

In this paper, we propose a novel query design for the transformer-based object detection. 
In previous transformer-based detectors, the object queries are a set of learned embeddings.
However, each learned embedding does not have an explicit physical meaning and we cannot explain where it will focus on.
It is difficult to optimize as the prediction slot of each object query does not have a specific mode. 
In other words, each object query will not focus on a specific region.
To solve these problems, in our query design, object queries are based on anchor points, which are widely used in CNN-based detectors. 
So each object query focuses on the objects near the anchor point. 
Moreover, our query design can predict multiple objects at one position to solve the difficulty: ``one region, multiple objects''.
In addition, we design an attention variant, which can reduce the memory cost while achieving similar or better performance than the standard attention in DETR.
Thanks to the query design and the attention variant, the proposed detector that we called Anchor DETR, can achieve better performance and run faster than the DETR with 10x fewer training epochs.
For example, it achieves 44.2 AP with 19 FPS on the MSCOCO dataset when using the ResNet50-DC5 feature for training 50 epochs.
Extensive experiments on the MSCOCO benchmark prove the effectiveness of the proposed methods.
Code is available at https://github.com/megvii-research/AnchorDETR.

----

## [286] Panini-Net: GAN Prior Based Degradation-Aware Feature Interpolation for Face Restoration

**Authors**: *Yinhuai Wang, Yujie Hu, Jian Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20159](https://doi.org/10.1609/aaai.v36i3.20159)

**Abstract**:

Emerging high-quality face restoration (FR) methods often utilize pre-trained GAN models (i.e., StyleGAN2) as GAN Prior. However, these methods usually struggle to balance realness and fidelity when facing various degradation levels. Besides, there is still a noticeable visual quality gap compared with pre-trained GAN models. In this paper, we propose a novel GAN Prior based degradation-aware feature interpolation network, dubbed Panini-Net, for FR tasks by explicitly learning the abstract representations to distinguish various degradations. Specifically, an unsupervised degradation representation learning (UDRL) strategy is first developed to extract degradation representations (DR) of the input degraded images. Then, a degradation-aware feature interpolation (DAFI) module is proposed to dynamically fuse the two types of informative features (i.e., features from input images and features from GAN Prior) with flexible adaption to various degradations based on DR. Ablation studies reveal the working mechanism of DAFI and its potential for editable FR. Extensive experiments demonstrate that our Panini-Net achieves state-of-the-art performance for multi-degradation face restoration and face super-resolution. The source code is available at https://github.com/jianzhangcs/panini.

----

## [287] End-to-End Transformer Based Model for Image Captioning

**Authors**: *Yiyu Wang, Jungang Xu, Yingfei Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20160](https://doi.org/10.1609/aaai.v36i3.20160)

**Abstract**:

CNN-LSTM based architectures have played an important role in image captioning, but limited by the training efficiency and expression ability, researchers began to explore the CNN-Transformer based models and achieved great success. Meanwhile, almost all recent works adopt Faster R-CNN as the backbone encoder to extract region-level features from given images. However, Faster R-CNN needs a pre-training on an additional dataset, which divides the image captioning task into two stages and limits its potential applications. In this paper, we build a pure Transformer-based model, which integrates image captioning into one stage and realizes end-to-end training. Firstly, we adopt SwinTransformer to replace Faster R-CNN as the backbone encoder to extract grid-level features from given images; Then, referring to Transformer, we build a refining encoder and a decoder. The refining encoder refines the grid features by capturing the intra-relationship between them, and the decoder decodes the refined features into captions word by word. Furthermore, in order to increase the interaction between multi-modal (vision and language) features to enhance the modeling capability, we calculate the mean pooling of grid features as the global feature, then introduce it into refining encoder to refine with grid features together, and add a pre-fusion process of refined global feature and generated words in decoder. To validate the effectiveness of our proposed model, we conduct experiments on MSCOCO dataset. The experimental results compared to existing published works demonstrate that our model achieves new state-of-the-art performances of 138.2% (single model) and 141.0% (ensemble of 4 models) CIDEr scores on 'Karpathy' offline test split and 136.0% (c5) and 138.3% (c40) CIDEr scores on the official online test server. Trained models and source code will be released.

----

## [288] Learning to Detect 3D Facial Landmarks via Heatmap Regression with Graph Convolutional Network

**Authors**: *Yuan Wang, Min Cao, Zhenfeng Fan, Silong Peng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20161](https://doi.org/10.1609/aaai.v36i3.20161)

**Abstract**:

3D facial landmark detection is extensively used in many research fields such as face registration, facial shape analysis, and face recognition. Most existing methods involve traditional features and 3D face models for the detection of landmarks, and their performances are limited by the hand-crafted intermediate process. In this paper, we propose a novel 3D facial landmark detection method, which directly locates the coordinates of landmarks from 3D point cloud with a well-customized graph convolutional network. The graph convolutional network learns geometric features adaptively for 3D facial landmark detection with the assistance of constructed 3D heatmaps, which are Gaussian functions of distances to each landmark on a 3D face. On this basis, we further develop a local surface unfolding and registration module to predict 3D landmarks from the heatmaps. The proposed method forms the first baseline of deep point cloud learning method for 3D facial landmark detection. We demonstrate experimentally that the proposed method exceeds the existing approaches by a clear margin on BU-3DFE and FRGC datasets for landmark localization accuracy and stability, and also achieves high-precision results on a recent large-scale dataset.

----

## [289] Low-Light Image Enhancement with Normalizing Flow

**Authors**: *Yufei Wang, Renjie Wan, Wenhan Yang, Haoliang Li, Lap-Pui Chau, Alex C. Kot*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20162](https://doi.org/10.1609/aaai.v36i3.20162)

**Abstract**:

To enhance low-light images to normally-exposed ones is highly ill-posed, namely that the mapping relationship between them is one-to-many. Previous works based on the pixel-wise reconstruction losses and deterministic processes fail to capture the complex conditional distribution of normally exposed images, which results in improper brightness, residual noise, and artifacts. In this paper, we investigate to model this one-to-many relationship via a proposed normalizing flow model. An invertible network that takes the low-light images/features as the condition and learns to map the distribution of normally exposed images into a Gaussian distribution. In this way, the conditional distribution of the normally exposed images can be well modeled, and the enhancement process, i.e., the other inference direction of the invertible network, is equivalent to being constrained by a loss function that better describes the manifold structure of natural images during the training. The experimental results on the existing benchmark datasets show our method achieves better quantitative and qualitative results, obtaining better-exposed illumination, less noise and artifact, and richer colors.

----

## [290] Negative Sample Matters: A Renaissance of Metric Learning for Temporal Grounding

**Authors**: *Zhenzhi Wang, Limin Wang, Tao Wu, Tianhao Li, Gangshan Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20163](https://doi.org/10.1609/aaai.v36i3.20163)

**Abstract**:

Temporal grounding aims to localize a video moment which is semantically aligned with a given natural language query. Existing methods typically apply a detection or regression pipeline on the fused representation with the research focus on designing complicated prediction heads or fusion strategies. Instead, from a perspective on temporal grounding as a metric-learning problem, we present a Mutual Matching Network (MMN), to directly model the similarity between language queries and video moments in a joint embedding space. This new metric-learning framework enables fully exploiting negative samples from two new aspects: constructing negative cross-modal pairs in a mutual matching scheme and mining negative pairs across different videos. These new negative samples could enhance the joint representation learning of two modalities via cross-modal mutual matching to maximize their mutual information. Experiments show that our MMN achieves highly competitive performance compared with the state-of-the-art methods on four video grounding benchmarks. Based on MMN, we present a winner solution for the HC-STVG challenge of the 3rd PIC workshop. This suggests that metric learning is still a promising method for temporal grounding via capturing the essential cross-modal correlation in a joint embedding space. Code is available at https://github.com/MCG-NJU/MMN.

----

## [291] Texture Reformer: Towards Fast and Universal Interactive Texture Transfer

**Authors**: *Zhizhong Wang, Lei Zhao, Haibo Chen, Ailin Li, Zhiwen Zuo, Wei Xing, Dongming Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20164](https://doi.org/10.1609/aaai.v36i3.20164)

**Abstract**:

In this paper, we present the texture reformer, a fast and universal neural-based framework for interactive texture transfer with user-specified guidance. The challenges lie in three aspects: 1) the diversity of tasks, 2) the simplicity of guidance maps, and 3) the execution efficiency. To address these challenges, our key idea is to use a novel feed-forward multi-view and multi-stage synthesis procedure consisting of I) a global view structure alignment stage, II) a local view texture refinement stage, and III) a holistic effect enhancement stage to synthesize high-quality results with coherent structures and fine texture details in a coarse-to-fine fashion. In addition, we also introduce a novel learning-free view-specific texture reformation (VSTR) operation with a new semantic map guidance strategy to achieve more accurate semantic-guided and structure-preserved texture transfer. The experimental results on a variety of application scenarios demonstrate the effectiveness and superiority of our framework. And compared with the state-of-the-art interactive texture transfer algorithms, it not only achieves higher quality results but, more remarkably, also is 2-5 orders of magnitude faster.

----

## [292] Interact, Embed, and EnlargE: Boosting Modality-Specific Representations for Multi-Modal Person Re-identification

**Authors**: *Zi Wang, Chenglong Li, Aihua Zheng, Ran He, Jin Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20165](https://doi.org/10.1609/aaai.v36i3.20165)

**Abstract**:

Multi-modal person Re-ID introduces more complementary information to assist the traditional Re-ID task. Existing multi-modal methods ignore the importance of modality-specific information in the feature fusion stage. To this end, we propose a novel method to boost modality-specific representations for multi-modal person Re-ID: Interact, Embed, and EnlargE (IEEE). First, we propose a cross-modal interacting module to exchange useful information between different modalities in the feature extraction phase. Second, we propose a relation-based embedding module to enhance the richness of feature descriptors by embedding the global feature into the fine-grained local information. Finally, we propose multi-modal margin loss to force the network to learn modality-specific information for each modality by enlarging the intra-class discrepancy. Superior performance on multi-modal Re-ID dataset RGBNT201 and three constructed Re-ID datasets validate the effectiveness of the proposed method compared with the state-of-the-art approaches.

----

## [293] Can Semantic Labels Assist Self-Supervised Visual Representation Learning?

**Authors**: *Longhui Wei, Lingxi Xie, Jianzhong He, Xiaopeng Zhang, Qi Tian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20166](https://doi.org/10.1609/aaai.v36i3.20166)

**Abstract**:

Recently, contrastive learning has largely advanced the progress of unsupervised visual representation learning. Pre-trained on ImageNet, some self-supervised algorithms reported higher transfer learning performance compared to fully-supervised methods, seeming to deliver the message that human labels hardly contribute to learning transferrable visual features. In this paper, we defend the usefulness of semantic labels but point out that fully-supervised and self-supervised methods are pursuing different kinds of features. To alleviate this issue, we present a new algorithm named Supervised Contrastive Adjustment in Neighborhood (SCAN) that maximally prevents the semantic guidance from damaging the appearance feature embedding. In a series of downstream tasks, SCAN achieves superior performance compared to previous fully-supervised and self-supervised methods, and sometimes the gain is significant. More importantly, our study reveals that semantic labels are useful in assisting self-supervised methods, opening a new direction for the community.

----

## [294] Rethinking the Two-Stage Framework for Grounded Situation Recognition

**Authors**: *Meng Wei, Long Chen, Wei Ji, Xiaoyu Yue, Tat-Seng Chua*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20167](https://doi.org/10.1609/aaai.v36i3.20167)

**Abstract**:

Grounded Situation Recognition (GSR), i.e., recognizing the salient activity (or verb) category in an image (e.g.,buying) and detecting all corresponding semantic roles (e.g.,agent and goods), is an essential step towards “human-like” event understanding. Since each verb is associated with a specific set of semantic roles, all existing GSR methods resort to a two-stage framework: predicting the verb in the first stage and detecting the semantic roles in the second stage. However, there are obvious drawbacks in both stages: 1) The widely-used cross-entropy (XE) loss for object recognition is insufficient in verb classification due to the large intra-class variation and high inter-class similarity among daily activities. 2) All semantic roles are detected in an autoregressive manner, which fails to model the complex semantic relations between different roles. To this end, we propose a novel SituFormerfor GSR which consists of a Coarse-to-Fine Verb Model (CFVM) and a Transformer-based Noun Model (TNM). CFVM is a two-step verb prediction model: a coarse-grained model trained with XE loss first proposes a set of verb candidates, and then a fine-grained model trained with triplet loss re-ranks these candidates with enhanced verb features (not only separable but also discriminative). TNM is a transformer-based semantic role detection model, which detects all roles parallelly. Owing to the global relation modeling ability and flexibility of the transformer decoder, TNM can fully explore the statistical dependency of the roles. Extensive validations on the challenging SWiG benchmark show that SituFormer achieves a new state-of-the-art performance with significant gains under various metrics. Code is available at https://github.com/kellyiss/SituFormer.

----

## [295] Boosting the Transferability of Video Adversarial Examples via Temporal Translation

**Authors**: *Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20168](https://doi.org/10.1609/aaai.v36i3.20168)

**Abstract**:

Although deep-learning based video recognition models have achieved remarkable success, they are vulnerable to adversarial examples that are generated by adding human-imperceptible perturbations on clean video samples. As indicated in recent studies, adversarial examples are transferable, which makes it feasible for black-box attacks in real-world applications. Nevertheless, most existing adversarial attack methods have poor transferability when attacking other video models and transfer-based attacks on video models are still unexplored. To this end, we propose to boost the transferability of video adversarial examples for black-box attacks on video recognition models. Through extensive analysis, we discover that different video recognition models rely on different discriminative temporal patterns, leading to the poor transferability of video adversarial examples. This motivates us to introduce a temporal translation attack method, which optimizes the adversarial perturbations over a set of temporal translated video clips. By generating adversarial examples over translated videos, the resulting adversarial examples are less sensitive to temporal patterns existed in the white-box model being attacked and thus can be better transferred. Extensive experiments on the Kinetics-400 dataset and the UCF-101 dataset demonstrate that our method can significantly boost the transferability of video adversarial examples. For transfer-based attack against video recognition models, it achieves a 61.56% average attack success rate on the Kinetics-400 and 48.60% on the UCF-101.

----

## [296] Towards Transferable Adversarial Attacks on Vision Transformers

**Authors**: *Zhipeng Wei, Jingjing Chen, Micah Goldblum, Zuxuan Wu, Tom Goldstein, Yu-Gang Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20169](https://doi.org/10.1609/aaai.v36i3.20169)

**Abstract**:

Vision transformers (ViTs) have demonstrated impressive performance on a series of computer vision tasks, yet they still suffer from adversarial examples. In this paper, we posit that adversarial attacks on transformers should be specially tailored for their architecture, jointly considering both patches and self-attention, in order to achieve high transferability. More specifically, we introduce a dual attack framework, which contains a Pay No Attention (PNA) attack and a PatchOut attack, to improve the transferability of adversarial samples across different ViTs. We show that skipping the gradients of attention during backpropagation can generate adversarial examples with high transferability. In addition, adversarial perturbations generated by optimizing randomly sampled subsets of patches at each iteration achieve higher attack success rates than attacks using all patches. We evaluate the transferability of attacks on state-of-the-art ViTs, CNNs and robustly trained CNNs. The results of these experiments demonstrate that the proposed dual attack can greatly boost transferability between ViTs and from ViTs to CNNs. In addition, the proposed method can easily be combined with existing transfer methods to boost performance.

----

## [297] L-CoDe: Language-Based Colorization Using Color-Object Decoupled Conditions

**Authors**: *Shuchen Weng, Hao Wu, Zheng Chang, Jiajun Tang, Si Li, Boxin Shi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20170](https://doi.org/10.1609/aaai.v36i3.20170)

**Abstract**:

Colorizing a grayscale image is inherently an ill-posed problem with multi-modal uncertainty. Language-based colorization offers a natural way of interaction to reduce such uncertainty via a user-provided caption. However, the color-object coupling and mismatch issues make the mapping from word to color difficult. In this paper, we propose L-CoDe, a Language-based Colorization network using color-object Decoupled conditions. A predictor for object-color corresponding matrix (OCCM) and a novel attention transfer module (ATM) are introduced to solve the color-object coupling problem. To deal with color-object mismatch that results in incorrect color-object correspondence, we adopt a soft-gated injection module (SIM). We further present a new dataset containing annotated color-object pairs to provide supervisory signals for resolving the coupling problem. Experimental results show that our approach outperforms state-of-the-art methods conditioned on captions.

----

## [298] Neural Interferometry: Image Reconstruction from Astronomical Interferometers Using Transformer-Conditioned Neural Fields

**Authors**: *Benjamin Wu, Chao Liu, Benjamin Eckart, Jan Kautz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20171](https://doi.org/10.1609/aaai.v36i3.20171)

**Abstract**:

Astronomical interferometry enables a collection of telescopes to achieve angular resolutions comparable to that of a single, much larger telescope. This is achieved by combining simultaneous observations from pairs of telescopes such that the signal is mathematically equivalent to sampling the Fourier domain of the object. However, reconstructing images from such sparse sampling is a challenging and ill-posed problem, with current methods requiring precise tuning of parameters and manual, iterative cleaning by experts. We present a novel deep learning approach in which the representation in the Fourier domain of an astronomical source is learned implicitly using a neural field representation. Data-driven priors can be added through a transformer encoder. Results on synthetically observed galaxies show that transformer-conditioned neural fields can successfully reconstruct astronomical observations even when the number of visibilities is very sparse.

----

## [299] TDv2: A Novel Tree-Structured Decoder for Offline Mathematical Expression Recognition

**Authors**: *Changjie Wu, Jun Du, Yunqing Li, Jianshu Zhang, Chen Yang, Bo Ren, Yiqing Hu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20172](https://doi.org/10.1609/aaai.v36i3.20172)

**Abstract**:

In recent years, tree decoders become more popular than LaTeX string decoders in the field of handwritten mathematical expression recognition (HMER) as they can capture the hierarchical tree structure of mathematical expressions. However previous tree decoders converted the tree structure labels into a fixed and ordered sequence, which could not make full use of the diversified expression of tree labels. In this study, we propose a novel tree decoder (TDv2) to fully utilize the tree structure labels. Compared with previous tree decoders, this new model does not require a fixed priority for different branches of a node during training and inference, which can effectively improve the model generalization capability. The input and output of the model make full use of the tree structure label, so that there is no need to find the parent node in the decoding process, which simplifies the decoding process and adds a prior information to help predict the node. We verified the effectiveness of each part of the model through comprehensive ablation experiments and attention visualization analysis. On the authoritative CROHME 14/16/19 datasets, our method achieves the state-of-the-art results.

----

## [300] Learning Token-Based Representation for Image Retrieval

**Authors**: *Hui Wu, Min Wang, Wengang Zhou, Yang Hu, Houqiang Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20173](https://doi.org/10.1609/aaai.v36i3.20173)

**Abstract**:

In image retrieval, deep local features learned in a data-driven manner have been demonstrated effective to improve retrieval performance. To realize efficient retrieval on large image database, some approaches quantize deep local features with a large codebook and match images with aggregated match kernel. However, the complexity of these approaches is non-trivial with large memory footprint, which limits their capability to jointly perform feature learning and aggregation. To generate compact global representations while maintaining regional matching capability, we propose a unified framework to jointly learn local feature representation and aggregation. In our framework, we first extract local features using CNNs. Then, we design a tokenizer module to aggregate them into a few visual tokens, each corresponding to a specific visual pattern. This helps to remove background noise, and capture more discriminative regions in the image. Next, a refinement block is introduced to enhance the visual tokens with self-attention and cross-attention. Finally, different visual tokens are concatenated to generate a compact global representation. The whole framework is trained end-to-end with image-level labels. Extensive experiments are conducted to evaluate our approach, which outperforms the state-of-the-art methods on the Revisited Oxford and Paris datasets.

----

## [301] Multi-Modal Answer Validation for Knowledge-Based VQA

**Authors**: *Jialin Wu, Jiasen Lu, Ashish Sabharwal, Roozbeh Mottaghi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20174](https://doi.org/10.1609/aaai.v36i3.20174)

**Abstract**:

The problem of knowledge-based visual question answering involves answering questions that require external knowledge in addition to the content of the image. Such knowledge typically comes in various forms, including visual, textual, and commonsense knowledge. Using more knowledge sources increases the chance of retrieving more irrelevant or noisy facts, making it challenging to comprehend the facts and find the answer. To address this challenge, we propose Multi-modal Answer Validation using External knowledge (MAVEx), where the idea is to validate a set of promising answer candidates based on answer-specific knowledge retrieval. Instead of searching for the answer in a vast collection of often irrelevant facts as most existing approaches do, MAVEx aims to learn how to extract relevant knowledge from noisy sources, which knowledge source to trust for each answer candidate, and how to validate the candidate using that source.
Our multi-modal setting is the first to leverage external visual knowledge (images searched using Google), in addition to textual knowledge in the form of Wikipedia sentences and ConceptNet concepts. Our experiments with OK-VQA, a challenging knowledge-based VQA dataset, demonstrate that MAVEx achieves new state-of-the-art results. Our code is available at https://github.com/jialinwu17/MAVEX

----

## [302] Neighborhood Consensus Contrastive Learning for Backward-Compatible Representation

**Authors**: *Shengsen Wu, Liang Chen, Yihang Lou, Yan Bai, Tao Bai, Minghua Deng, Ling-Yu Duan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20175](https://doi.org/10.1609/aaai.v36i3.20175)

**Abstract**:

In object re-identification (ReID), the development of deep learning techniques often involves model updates and deployment. It is unbearable to re-embedding and re-index with the system suspended when deploying new models. Therefore, backward-compatible representation is proposed to enable ``new'' features to be compared with ``old'' features directly, which means that the database is active when there are both ``new'' and ``old'' features in it. Thus we can scroll-refresh the database or even do nothing on the database to update. 

The existing backward-compatible methods either require a strong overlap between old and new training data or simply conduct constraints at the instance level. Thus they are difficult in handling complicated cluster structures and are limited in eliminating the impact of outliers in old embeddings, resulting in a risk of damaging the discriminative capability of new features.  In this work, we propose a Neighborhood Consensus Contrastive Learning (NCCL) method. With no assumptions about the new training data, we estimate the sub-cluster structures of old embeddings. A new embedding is constrained with multiple old embeddings in both embedding space and discrimination space at the sub-class level. The effect of outliers diminished, as the multiple samples serve as ``mean teachers''. Besides, we propose a scheme to filter the old embeddings with low credibility, further improving the compatibility robustness. Our method ensures the compatibility without impairing the accuracy of the new model. It can even improve the new model's accuracy in most scenarios.

----

## [303] Pale Transformer: A General Vision Transformer Backbone with Pale-Shaped Attention

**Authors**: *Sitong Wu, Tianyi Wu, Haoru Tan, Guodong Guo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20176](https://doi.org/10.1609/aaai.v36i3.20176)

**Abstract**:

Recently, Transformers have shown promising performance in various vision tasks. To reduce the quadratic computation complexity caused by the global self-attention, various methods constrain the range of attention within a local region to improve its efficiency. Consequently, their receptive fields in a single attention layer are not large enough, resulting in insufficient context modeling. To address this issue, we propose a Pale-Shaped self-Attention (PS-Attention), which performs self-attention within a pale-shaped region. Compared to the global self-attention, PS-Attention can reduce the computation and memory costs significantly. Meanwhile, it can capture richer contextual information under the similar computation complexity with previous local self-attention mechanisms. Based on the PS-Attention, we develop a general Vision Transformer backbone with a hierarchical architecture, named Pale Transformer, which achieves 83.4%, 84.3%, and 84.9% Top-1 accuracy with the model size of 22M, 48M, and 85M respectively for 224x224 ImageNet-1K classification, outperforming the previous Vision Transformer backbones. For downstream tasks, our Pale Transformer backbone performs better than the recent state-of-the-art CSWin Transformer by a large margin on ADE20K semantic segmentation and COCO object detection & instance segmentation. The code will be released on https://github.com/BR-IDL/PaddleViT.

----

## [304] Style Mixing and Patchwise Prototypical Matching for One-Shot Unsupervised Domain Adaptive Semantic Segmentation

**Authors**: *Xinyi Wu, Zhenyao Wu, Yuhang Lu, Lili Ju, Song Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20177](https://doi.org/10.1609/aaai.v36i3.20177)

**Abstract**:

In this paper, we tackle the problem of one-shot unsupervised domain adaptation (OSUDA) for semantic segmentation where the segmentors only see one unlabeled target image during training. In this case, traditional unsupervised domain adaptation models usually fail since they cannot adapt to the target domain with over-fitting to one (or few) target samples. To address this problem, existing OSUDA methods usually integrate a style-transfer module to perform domain randomization based on the unlabeled target sample, with which multiple domains around the target sample can be explored during training. However, such a style-transfer module relies on an additional set of images as style reference for pre-training and also increases the memory demand for domain adaptation. Here we propose a new OSUDA method that can effectively relieve such computational burden. Specifically, we integrate several style-mixing layers into the segmentor which play the role of style-transfer module to stylize the source images without introducing any learned parameters. Moreover, we propose a patchwise prototypical matching (PPM) method to weighted consider the importance of source pixels during the supervised training to relieve the negative adaptation. Experimental results show that our method achieves new state-of-the-art performance on two commonly used benchmarks for domain adaptive semantic segmentation under the one-shot setting and is more efficient than all comparison approaches.

----

## [305] Multi-Centroid Representation Network for Domain Adaptive Person Re-ID

**Authors**: *Yuhang Wu, Tengteng Huang, Haotian Yao, Chi Zhang, Yuanjie Shao, Chuchu Han, Changxin Gao, Nong Sang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20178](https://doi.org/10.1609/aaai.v36i3.20178)

**Abstract**:

Recently, many approaches tackle the Unsupervised Domain Adaptive person re-identification (UDA re-ID) problem through pseudo-label-based contrastive learning. During training, a uni-centroid representation is obtained by simply averaging all the instance features from a cluster with the same pseudo label. However, a cluster may contain images with different identities (label noises) due to the imperfect clustering results, which makes the uni-centroid representation inappropriate. In this paper, we present a novel Multi-Centroid Memory (MCM) to adaptively capture different identity information within the cluster. MCM can effectively alleviate the issue of label noises by selecting proper positive/negative centroids for the query image. Moreover, we further propose two strategies to improve the contrastive learning process. First, we present a Domain-Specific Contrastive Learning (DSCL) mechanism to fully explore intra-domain information by comparing samples only from the same domain. Second, we propose Second-Order Nearest Interpolation (SONI) to obtain abundant and informative negative samples. We integrate MCM, DSCL, and SONI into a unified framework named Multi-Centroid Representation Network (MCRN). Extensive experiments demonstrate the superiority of MCRN over state-of-the-art approaches on multiple UDA re-ID tasks and fully unsupervised re-ID tasks.

----

## [306] Efficient Non-local Contrastive Attention for Image Super-resolution

**Authors**: *Bin Xia, Yucheng Hang, Yapeng Tian, Wenming Yang, Qingmin Liao, Jie Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20179](https://doi.org/10.1609/aaai.v36i3.20179)

**Abstract**:

Non-Local Attention (NLA) brings significant improvement for Single Image Super-Resolution (SISR) by leveraging intrinsic feature correlation in natural images. However, NLA gives noisy information large weights and consumes quadratic computation resources with respect to the input size, limiting its performance and application. In this paper, we propose a novel Efficient Non-Local Contrastive Attention (ENLCA) to perform long-range visual modeling and leverage more relevant non-local features. Specifically, ENLCA consists of two parts, Efficient Non-Local Attention (ENLA) and Sparse Aggregation. ENLA adopts the kernel method to approximate exponential function and obtains linear computation complexity. For Sparse Aggregation, we multiply inputs by an amplification factor to focus on informative features, yet the variance of approximation increases exponentially. Therefore, contrastive learning is applied to further separate relevant and irrelevant features. To demonstrate the effectiveness of ENLCA, we build an architecture called Efficient Non-Local Contrastive Network (ENLCN) by adding a few of our modules in a simple backbone. Extensive experimental results show that ENLCN reaches superior performance over state-of-the-art approaches on both quantitative and qualitative evaluations.

----

## [307] Coarse-to-Fine Embedded PatchMatch and Multi-Scale Dynamic Aggregation for Reference-Based Super-resolution

**Authors**: *Bin Xia, Yapeng Tian, Yucheng Hang, Wenming Yang, Qingmin Liao, Jie Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20180](https://doi.org/10.1609/aaai.v36i3.20180)

**Abstract**:

Reference-based super-resolution (RefSR) has made significant progress in producing realistic textures using an external reference (Ref) image. However, existing RefSR methods obtain high-quality correspondence matchings consuming quadratic computation resources with respect to the input size, limiting its application. Moreover, these approaches usually suffer from scale misalignments between the low-resolution (LR) image and Ref image. In this paper, we propose an Accelerated Multi-Scale Aggregation network (AMSA) for Reference-based Super-Resolution, including Coarse-to-Fine Embedded PatchMatch (CFE-PatchMatch) and Multi-Scale Dynamic Aggregation (MSDA) module. To improve matching efficiency, we design a novel Embedded PatchMacth scheme with random samples propagation, which involves end-to-end training with asymptotic linear computational cost to the input size. To further reduce computational cost and speed up convergence, we apply the coarse-to-fine strategy on Embedded PatchMacth constituting CFE-PatchMatch. To fully leverage reference information across multiple scales and enhance robustness to scale misalignment, we develop the MSDA module consisting of Dynamic Aggregation and Multi-Scale Aggregation. The Dynamic Aggregation corrects minor scale misalignment by dynamically aggregating features, and the Multi-Scale Aggregation brings robustness to large scale misalignment by fusing multi-scale information. Experimental results show that the proposed AMSA achieves superior performance over state-of-the-art approaches on both quantitative and qualitative evaluations.

----

## [308] Cross-Domain Collaborative Normalization via Structural Knowledge

**Authors**: *Haifeng Xia, Zhengming Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20181](https://doi.org/10.1609/aaai.v36i3.20181)

**Abstract**:

Batch Normalization (BN) as an important component assists Deep Neural Networks in achieving promising performance for extensive learning tasks by scaling distribution of feature representations within mini-batches. However, the application of BN suffers from performance degradation under the scenario of Unsupervised Domain Adaptation (UDA), since the estimated statistics fail to concurrently describe two different domains. In this paper, we develop a novel normalization technique, named Collaborative Normalization (CoN), for eliminating domain discrepancy and accelerating the model training of neural networks for UDA. Unlike typical strategies only exploiting domain-specific statistics during normalization, our CoN excavates cross-domain knowledge and simultaneously scales features from various domains by mimicking the merits of collaborative representation. Our CoN can be easily plugged into popular neural network backbones for cross-domain learning. On the one hand, theoretical analysis guarantees that models with CoN promote discriminability of feature representations and accelerate convergence rate; on the other hand, empirical study verifies that replacing BN with CoN in popular network backbones effectively improves classification accuracy in most learning tasks across three cross-domain visual benchmarks.

----

## [309] ReMoNet: Recurrent Multi-Output Network for Efficient Video Denoising

**Authors**: *Liuyu Xiang, Jundong Zhou, Jirui Liu, Zerun Wang, Haidong Huang, Jie Hu, Jungong Han, Yuchen Guo, Guiguang Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20182](https://doi.org/10.1609/aaai.v36i3.20182)

**Abstract**:

While deep neural network-based video denoising methods have achieved promising results, it is still hard to deploy them on mobile devices due to their high computational cost and memory demands. This paper aims to develop a lightweight deep video denoising method that is friendly to resource-constrained mobile devices. Inspired by the facts that 1) consecutive video frames usually contain redundant temporal coherency, and 2) neural networks are usually over-parameterized, we propose a multi-input multi-output (MIMO) paradigm to process consecutive video frames within one-forward-pass. The basic idea is concretized to a novel architecture termed Recurrent Multi-output Network (ReMoNet), which consists of recurrent temporal fusion and temporal aggregation blocks and is further reinforced by similarity-based mutual distillation. We conduct extensive experiments on NVIDIA GPU and Qualcomm Snapdragon 888 mobile platform with Gaussian noise and simulated Image-Signal-Processor (ISP) noise. The experimental results show that ReMoNet is both effective and efficient on video denoising. Moreover, we show that ReMoNet is more robust under higher noise level scenarios.

----

## [310] Transfer Learning from Synthetic to Real LiDAR Point Cloud for Semantic Segmentation

**Authors**: *Aoran Xiao, Jiaxing Huang, Dayan Guan, Fangneng Zhan, Shijian Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20183](https://doi.org/10.1609/aaai.v36i3.20183)

**Abstract**:

Knowledge transfer from synthetic to real data has been widely studied to mitigate data annotation constraints in various computer vision tasks such as semantic segmentation. However, the study focused on 2D images and its counterpart in 3D point clouds segmentation lags far behind due to the lack of large-scale synthetic datasets and effective transfer methods. We address this issue by collecting SynLiDAR, a large-scale synthetic LiDAR dataset that contains point-wise annotated point clouds with accurate geometric shapes and comprehensive semantic classes. SynLiDAR was collected from multiple virtual environments with rich scenes and layouts which consists of over 19 billion points of 32 semantic classes. In addition, we design PCT, a novel point cloud translator that effectively mitigates the gap between synthetic and real point clouds. Specifically, we decompose the synthetic-to-real gap into an appearance component and a sparsity component and handle them separately which improves the point cloud translation greatly. We conducted extensive experiments over three transfer learning setups including data augmentation, semi-supervised domain adaptation and unsupervised domain adaptation. Extensive experiments show that SynLiDAR provides a high-quality data source for studying 3D transfer and the proposed PCT achieves superior point cloud translation consistently across the three setups.  The dataset is available at https://github.com/xiaoaoran/SynLiDAR.

----

## [311] Video as Conditional Graph Hierarchy for Multi-Granular Question Answering

**Authors**: *Junbin Xiao, Angela Yao, Zhiyuan Liu, Yicong Li, Wei Ji, Tat-Seng Chua*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20184](https://doi.org/10.1609/aaai.v36i3.20184)

**Abstract**:

Video question answering requires the models to understand and reason about both the complex video and language data to correctly derive the answers. Existing efforts have been focused on designing sophisticated cross-modal interactions to fuse the information from two modalities, while encoding the video and question holistically as frame and word sequences. Despite their success, these methods are essentially revolving around the sequential nature of video- and question-contents, providing little insight to the problem of question-answering and lacking interpretability as well. In this work, we argue that while video is presented in frame sequence, the visual elements (e.g., objects, actions, activities and events) are not sequential but rather hierarchical in semantic space. To align with the multi-granular essence of linguistic concepts in language queries, we propose to model video as a conditional graph hierarchy which weaves together visual facts of different granularity in a level-wise manner, with the guidance of corresponding textual cues. Despite the simplicity, our extensive experiments demonstrate the superiority of such conditional hierarchical graph architecture, with clear performance improvements over prior methods and also better generalization across different type of questions. Further analyses also demonstrate the model's reliability as it shows meaningful visual-textual evidences for the predicted answers.

----

## [312] AdaptivePose: Human Parts as Adaptive Points

**Authors**: *Yabo Xiao, Xiaojuan Wang, Dongdong Yu, Guoli Wang, Qian Zhang, Mingshu He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20185](https://doi.org/10.1609/aaai.v36i3.20185)

**Abstract**:

Multi-person pose estimation methods generally follow top-down and bottom-up paradigms, both of which can be considered as two-stage approaches thus leading to the high computation cost and low efficiency. Towards a compact and efficient pipeline for multi-person pose estimation task, in this paper, we propose to represent the human parts as points and present a novel body representation, which leverages an adaptive point set including the human center and seven human-part related points to represent the human instance in a more fine-grained manner. The novel representation is more capable of capturing the various pose deformation and adaptively factorizes the long-range center-to-joint displacement thus delivers a single-stage differentiable network to more precisely regress multi-person pose, termed as AdaptivePose. For inference, our proposed network eliminates the grouping as well as refinements and only needs a single-step disentangling process to form multi-person pose. Without any bells and whistles, we achieve the best speed-accuracy trade-offs of 67.4% AP / 29.4 fps with DLA-34 and 71.3% AP / 9.1 fps with HRNet-W48 on COCO test-dev dataset.

----

## [313] Learning Quality-Aware Representation for Multi-Person Pose Regression

**Authors**: *Yabo Xiao, Dongdong Yu, Xiaojuan Wang, Lei Jin, Guoli Wang, Qian Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20186](https://doi.org/10.1609/aaai.v36i3.20186)

**Abstract**:

Off-the-shelf single-stage multi-person pose regression methods generally leverage the instance score (i.e., confidence of the instance localization) to indicate the pose quality for selecting the pose candidates. We consider that there are two gaps involved in existing paradigm: 1) The instance score is not well interrelated with the pose regression quality. 2) The instance feature representation, which is used for predicting the instance score, does not explicitly encode the structural pose information to predict the reasonable score that represents pose regression quality. To address the aforementioned issues, we propose to learn the pose regression quality-aware representation. Concretely, for the first gap, instead of using the previous instance confidence label (e.g., discrete {1,0} or Gaussian representation) to denote the position and confidence for person instance, we firstly introduce the Consistent Instance Representation (CIR) that unifies the pose regression quality score of instance and the confidence of background into a pixel-wise score map to calibrates the inconsistency between instance score and pose regression quality. To fill the second gap, we further present the Query Encoding Module (QEM) including the Keypoint Query Encoding (KQE) to encode the positional and semantic information for each keypoint and the Pose Query Encoding (PQE) which explicitly encodes the predicted structural pose information to better fit the Consistent Instance Representation (CIR). By using the proposed components, we significantly alleviate the above gaps. Our method outperforms previous single-stage regression-based even bottom-up methods and achieves the state-of-the-art result of 71.7 AP on MS COCO test-dev set.

----

## [314] Attribute-Based Progressive Fusion Network for RGBT Tracking

**Authors**: *Yun Xiao, Mengmeng Yang, Chenglong Li, Lei Liu, Jin Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20187](https://doi.org/10.1609/aaai.v36i3.20187)

**Abstract**:

RGBT tracking usually suffers from various challenge factors, such as fast motion, scale variation, illumination variation, thermal crossover and occlusion, to name a few. Existing works often study fusion models to solve all challenges simultaneously, and it requires fusion models complex enough and training data large enough, which are usually difficult to be constructed in real-world scenarios. In this work, we disentangle the fusion process via the challenge attributes, and thus propose a novel Attribute-based Progressive Fusion Network (APFNet) to increase the fusion capacity with a small number of parameters while reducing the dependence on large-scale training data. In particular, we design five attribute-specific fusion branches to integrate RGB and thermal features under the challenges of thermal crossover, illumination variation, scale variation, occlusion and fast motion respectively. By disentangling the fusion process, we can use a small number of parameters for each branch to achieve robust fusion of different modalities and train each branch using the small training subset with the corresponding attribute annotation. Then, to adaptive fuse features of all branches, we design an aggregation fusion module based on SKNet. Finally, we also design an enhancement fusion transformer to strengthen the aggregated feature and modality-specific features. Experimental results on benchmark datasets demonstrate the effectiveness of our APFNet against other state-of-the-art methods.

----

## [315] Detailed Facial Geometry Recovery from Multi-View Images by Learning an Implicit Function

**Authors**: *Yunze Xiao, Hao Zhu, Haotian Yang, Zhengyu Diao, Xiangju Lu, Xun Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20188](https://doi.org/10.1609/aaai.v36i3.20188)

**Abstract**:

Recovering detailed facial geometry from a set of calibrated multi-view images is valuable for its wide range of applications. Traditional multi-view stereo (MVS) methods adopt an optimization-based scheme to regularize the matching cost. Recently, learning-based methods integrate all these into an end-to-end neural network and show superiority of efficiency. In this paper, we propose a novel architecture to recover extremely detailed 3D faces within dozens of seconds.  Unlike previous learning-based methods that regularize the cost volume via 3D CNN, we propose to learn an implicit function for regressing the matching cost.  By fitting a 3D morphable model from multi-view images, the features of multiple images are extracted and aggregated in the mesh-attached UV space, which makes the implicit function more effective in recovering detailed facial shape. Our method outperforms SOTA learning-based MVS in accuracy by a large margin on the FaceScape dataset. The code and data are released in https://github.com/zhuhao-nju/mvfr.

----

## [316] FINet: Dual Branches Feature Interaction for Partial-to-Partial Point Cloud Registration

**Authors**: *Hao Xu, Nianjin Ye, Guanghui Liu, Bing Zeng, Shuaicheng Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20189](https://doi.org/10.1609/aaai.v36i3.20189)

**Abstract**:

Data association is important in the point cloud registration. In this work, we propose to solve the partial-to-partial registration from a new perspective, by introducing multi-level feature interactions between the source and the reference clouds at the feature extraction stage, such that the registration can be realized without the attentions or explicit mask estimation for the overlapping detection as adopted previously. Specifically, we present FINet, a feature interactionbased structure with the capability to enable and strengthen the information associating between the inputs at multiple stages. To achieve this, we first split the features into two components, one for rotation and one for translation, based on the fact that they belong to different solution spaces, yielding a dual branches structure. Second, we insert several interaction modules at the feature extractor for the data association. Third, we propose a transformation sensitivity loss to obtain rotation-attentive and translation-attentive features. Experiments demonstrate that our method performs higher precision and robustness compared to the state-of-the-art traditional and learning-based methods. Code is available at https://github.com/megvii-research/FINet.

----

## [317] Rendering-Aware HDR Environment Map Prediction from a Single Image

**Authors**: *Jun-Peng Xu, Chenyu Zuo, Fang-Lue Zhang, Miao Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20190](https://doi.org/10.1609/aaai.v36i3.20190)

**Abstract**:

High dynamic range (HDR) illumination estimation from a single low dynamic range (LDR) image is a significant task in computer vision, graphics, and augmented reality. We present a two-stage deep learning-based method to predict an HDR environment map from a single narrow field-of-view LDR image. We first learn a hybrid parametric representation that sufficiently covers high- and low-frequency illumination components in the environment. Taking the estimated illuminations as guidance, we build a generative adversarial network to synthesize an HDR environment map that enables realistic rendering effects. We specifically consider the rendering effect by supervising the networks using rendering losses in both stages, on the predicted environment map as well as the hybrid illumination representation. Quantitative and qualitative experiments demonstrate that our approach achieves lower relighting errors for virtual object insertion and is preferred by users compared to state-of-the-art methods.

----

## [318] Topology-Aware Convolutional Neural Network for Efficient Skeleton-Based Action Recognition

**Authors**: *Kailin Xu, Fanfan Ye, Qiaoyong Zhong, Di Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20191](https://doi.org/10.1609/aaai.v36i3.20191)

**Abstract**:

In the context of skeleton-based action recognition, graph convolutional networks (GCNs) have been rapidly developed, whereas convolutional neural networks (CNNs) have received less attention. One reason is that CNNs are considered poor in modeling the irregular skeleton topology. To alleviate this limitation, we propose a pure CNN architecture named Topology-aware CNN (Ta-CNN) in this paper. In particular, we develop a novel cross-channel feature augmentation module, which is a combo of map-attend-group-map operations. By applying the module to the coordinate level and the joint level subsequently, the topology feature is effectively enhanced. Notably, we theoretically prove that graph convolution is a special case of normal convolution when the joint dimension is treated as channels. This confirms that the topology modeling power of GCNs can also be implemented by using a CNN. Moreover, we creatively design a SkeletonMix strategy which mixes two persons in a unique manner and further boosts the performance. Extensive experiments are conducted on four widely used datasets, i.e. N-UCLA, SBU, NTU RGB+D and NTU RGB+D 120 to verify the effectiveness of Ta-CNN. We surpass existing CNN-based methods significantly. Compared with leading GCN-based methods, we achieve comparable performance with much less complexity in terms of the required GFLOPs and parameters.

----

## [319] Transcoded Video Restoration by Temporal Spatial Auxiliary Network

**Authors**: *Li Xu, Gang He, Jinjia Zhou, Jie Lei, Weiying Xie, Yunsong Li, Yu-Wing Tai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20192](https://doi.org/10.1609/aaai.v36i3.20192)

**Abstract**:

In most video platforms, such as Youtube, Kwai, and TikTok, the played videos usually have undergone multiple video encodings such as hardware encoding by recording devices, software encoding by video editing apps, and single/multiple video transcoding by video application servers. Previous works in compressed video restoration typically assume the compression artifacts are caused by one-time encoding. Thus, the derived solution usually does not work very well in practice. In this paper, we propose a new method, temporal spatial auxiliary network (TSAN), for transcoded video restoration. Our method considers the unique traits between video encoding and transcoding, and we consider the initial shallow encoded videos as the intermediate labels to assist the network to conduct self-supervised attention training. In addition, we employ adjacent multi-frame information and propose the temporal deformable alignment and pyramidal spatial fusion for transcoded video restoration. The experimental results demonstrate that the performance of the proposed method is superior to that of the previous techniques. The code is available at https://github.com/icecherylXuli/TSAN.

----

## [320] DIRL: Domain-Invariant Representation Learning for Generalizable Semantic Segmentation

**Authors**: *Qi Xu, Liang Yao, Zhengkai Jiang, Guannan Jiang, Wenqing Chu, Wenhui Han, Wei Zhang, Chengjie Wang, Ying Tai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20193](https://doi.org/10.1609/aaai.v36i3.20193)

**Abstract**:

Model generalization to the unseen scenes is crucial to real-world applications, such as autonomous driving, which requires robust vision systems. To enhance the model generalization, domain generalization through learning the domain-invariant representation has been widely studied. However, most existing works learn the shared feature space within multi-source domains but ignore the characteristic of the feature itself (e.g., the feature sensitivity to the domain-specific style). Therefore, we propose the Domain-invariant Representation Learning (DIRL) for domain generalization which utilizes the feature sensitivity as the feature prior to guide the enhancement of the model generalization capability. The guidance reflects in two folds: 1) Feature re-calibration that introduces the Prior Guided Attention Module (PGAM) to emphasize the insensitive features and suppress the sensitive features. 2): Feature whiting that proposes the Guided Feature Whiting (GFW) to remove the feature correlations which are sensitive to the domain-specific style. We construct the domain-invariant representation which suppresses the effect of the domain-specific style on the quality and correlation of the features. As a result, our method is simple yet effective, and can enhance the robustness of various backbone networks with little computational cost. Extensive experiments over multiple domains generalizable segmentation tasks show the superiority of our approach to other methods.

----

## [321] Behind the Curtain: Learning Occluded Shapes for 3D Object Detection

**Authors**: *Qiangeng Xu, Yiqi Zhong, Ulrich Neumann*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20194](https://doi.org/10.1609/aaai.v36i3.20194)

**Abstract**:

Advances in LiDAR sensors provide rich 3D data that supports 3D scene understanding. However, due to occlusion and signal miss, LiDAR point clouds are in practice 2.5D as they cover only partial underlying shapes, which poses a fundamental challenge to 3D perception. To tackle the challenge, we present a novel LiDAR-based 3D object detection model, dubbed Behind the Curtain Detector (BtcDet), which learns the object shape priors and estimates the complete object shapes that are partially occluded (curtained) in point clouds. BtcDet first identifies the regions that are affected by occlusion and signal miss. In these regions, our model predicts the probability of occupancy that indicates if a region contains object shapes and integrates this probability map with detection features and generates high-quality 3D proposals. Finally, the occupancy estimation is integrated into the proposal refinement module to generate accurate bounding boxes. Extensive experiments on the KITTI Dataset and the Waymo Open Dataset demonstrate the effectiveness of BtcDet. Particularly for the 3D detection of both cars and cyclists on the KITTI benchmark, BtcDet surpasses all of the published state-of-the-art methods by remarkable margins. Code is released.

----

## [322] Domain Disentangled Generative Adversarial Network for Zero-Shot Sketch-Based 3D Shape Retrieval

**Authors**: *Rui Xu, Zongyan Han, Le Hui, Jianjun Qian, Jin Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20195](https://doi.org/10.1609/aaai.v36i3.20195)

**Abstract**:

Sketch-based 3D shape retrieval is a challenging task due to the large domain discrepancy between sketches and 3D shapes. Since existing methods are trained and evaluated on the same categories, they cannot effectively recognize the categories that have not been used during training. In this paper, we propose a novel domain disentangled generative adversarial network (DD-GAN) for zero-shot sketch-based 3D retrieval, which can retrieve the unseen categories that are not accessed during training. Specifically, we first generate domain-invariant features and domain-specific features by disentangling the learned features of sketches and 3D shapes, where the domain-invariant features are used to align with the corresponding word embeddings. Then, we develop a generative adversarial network that combines the domain-specific features of the seen categories with the aligned domain-invariant features to synthesize samples, where the synthesized samples of the unseen categories are generated by using the corresponding word embeddings. Finally, we use the synthesized samples of the unseen categories combined with the real samples of the seen categories to train the network for retrieval, so that the unseen categories can be recognized. In order to reduce the domain shift problem, we utilize unlabeled unseen samples to enhance the discrimination ability of the discriminator. With the discriminator distinguishing the generated samples from the unlabeled unseen samples, the generator can generate more realistic unseen samples. Extensive experiments on the SHREC'13 and SHREC'14 datasets show that our method significantly improves the retrieval performance of the unseen categories.

----

## [323] Dual Attention Networks for Few-Shot Fine-Grained Recognition

**Authors**: *Shu-Lin Xu, Faen Zhang, Xiu-Shen Wei, Jianhua Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20196](https://doi.org/10.1609/aaai.v36i3.20196)

**Abstract**:

The task of few-shot fine-grained recognition is to classify images belonging to subordinate categories merely depending on few examples. Due to the fine-grained nature, it is desirable to capture subtle but discriminative part-level patterns from limited training data, which makes it a challenging problem. In this paper, to generate fine-grained tailored representations for few-shot recognition, we propose a Dual Attention Network (Dual Att-Net) consisting of two dual branches of both hard- and soft-attentions. Specifically, by producing attention guidance from deep activations of input images, our hard-attention is realized by keeping a few useful deep descriptors and forming them as a bag of multi-instance learning. Since these deep descriptors could correspond to objects' parts, the advantage of modeling as a multi-instance bag is able to exploit inherent correlation of these fine-grained parts. On the other side, a soft attended activation representation can be obtained by applying attention guidance upon original activations, which brings comprehensive attention information as the counterpart of hard-attention. After that, both outputs of dual branches are aggregated as a holistic image embedding w.r.t. input images. By performing meta-learning, we can learn a powerful image embedding in such a metric space to generalize to novel classes. Experiments on three popular fine-grained benchmark datasets show that our Dual Att-Net obviously outperforms other existing state-of-the-art methods.

----

## [324] Sparse Cross-Scale Attention Network for Efficient LiDAR Panoptic Segmentation

**Authors**: *Shuangjie Xu, Rui Wan, Maosheng Ye, Xiaoyi Zou, Tongyi Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20197](https://doi.org/10.1609/aaai.v36i3.20197)

**Abstract**:

Two major challenges of 3D LiDAR Panoptic Segmentation (PS) are that point clouds of an object are surface-aggregated and thus hard to model the long-range dependency especially for large instances, and that objects are too close to separate each other. Recent literature addresses these problems by time-consuming grouping processes such as dual-clustering, mean-shift offsets and etc., or by bird-eye-view (BEV) dense centroid representation that downplays geometry. However, the long-range geometry relationship has not been sufficiently modeled by local feature learning from the above methods. To this end, we present SCAN, a novel sparse cross-scale attention network to first align multi-scale sparse features with global voxel-encoded attention to capture the long-range relationship of instance context, which is able to boost the regression accuracy of the over-segmented large objects. For the surface-aggregated points, SCAN adopts a novel sparse class-agnostic representation of instance centroids, which can not only maintain the sparsity of aligned features to solve the under-segmentation on small objects, but also reduce the computation amount of the network through sparse convolution. Our method outperforms previous methods by a large margin in the SemanticKITTI dataset for the challenging 3D PS task, achieving 1st place with a real-time inference speed.

----

## [325] Towards Fully Sparse Training: Information Restoration with Spatial Similarity

**Authors**: *Weixiang Xu, Xiangyu He, Ke Cheng, Peisong Wang, Jian Cheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20198](https://doi.org/10.1609/aaai.v36i3.20198)

**Abstract**:

The 2:4 structured sparsity pattern released by NVIDIA Ampere architecture, requiring four consecutive values containing at least two zeros, enables doubling math throughput for matrix multiplications. Recent works mainly focus on inference speedup via 2:4 sparsity while training acceleration has been largely overwhelmed where backpropagation consumes around 70% of the training time. However, unlike inference, training speedup with structured pruning is nontrivial due to the need to maintain the fidelity of gradients and reduce the additional overhead of performing 2:4 sparsity online. For the first time, this article proposes fully sparse training (FST) where `fully' indicates that ALL matrix multiplications in forward/backward propagation are structurally pruned while maintaining accuracy. To this end, we begin with saliency analysis, investigating the sensitivity of different sparse objects to structured pruning. Based on the observation of spatial similarity among activations, we propose pruning activations with fixed 2:4 masks. Moreover, an Information Restoration block is proposed to retrieve the lost information, which can be implemented by efficient gradient-shift operation. Evaluation of accuracy and efficiency shows that we can achieve 2× training acceleration with negligible accuracy degradation on challenging large-scale classification and detection tasks.

----

## [326] Hierarchical Image Generation via Transformer-Based Sequential Patch Selection

**Authors**: *Xiaogang Xu, Ning Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20199](https://doi.org/10.1609/aaai.v36i3.20199)

**Abstract**:

To synthesize images with preferred objects and interactions, a controllable way is to generate the image from a scene graph and a large pool of object crops, where the spatial arrangements of the objects in the image are defined by the scene graph while their appearances are determined by the retrieved crops from the pool. In this paper, we propose a novel framework with such a semi-parametric generation strategy. First, to encourage the retrieval of mutually compatible crops, we design a sequential selection strategy where the crop selection for each object is determined by the contents and locations of all object crops that have been chosen previously. Such process is implemented via a transformer trained with contrastive losses. Second, to generate the final image, our hierarchical generation strategy leverages hierarchical gated convolutions which are employed to synthesize areas not covered by any image crops, and a patch guided spatially adaptive normalization module which is proposed to guarantee the final generated images complying with the crop appearance and the scene graph. Evaluated on the challenging Visual Genome and COCO-Stuff dataset, our experimental results demonstrate the superiority of our proposed method over existing state-of-the-art methods.

----

## [327] Reliable Propagation-Correction Modulation for Video Object Segmentation

**Authors**: *Xiaohao Xu, Jinglu Wang, Xiao Li, Yan Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20200](https://doi.org/10.1609/aaai.v36i3.20200)

**Abstract**:

Error propagation is a general but crucial problem in online semi-supervised video object segmentation.
 We aim to suppress error propagation through a correction mechanism with high reliability.
 The key insight is to disentangle the correction from the conventional mask propagation process with reliable cues.
 We introduce two modulators, propagation and correction modulators, to separately perform channel-wise recalibration on the target frame embeddings according to local temporal correlations and reliable references respectively.
 Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 
 Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We augment the reference cues by supplementing reliable feature patches to a maintained pool, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames.
 Our model achieves state-of-the-art performance on YouTube-VOS18, YouTube-VOS19 and DAVIS17-Val/Test benchmarks.
 Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

----

## [328] Adaptive Hypergraph Neural Network for Multi-Person Pose Estimation

**Authors**: *Xixia Xu, Qi Zou, Xue Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20201](https://doi.org/10.1609/aaai.v36i3.20201)

**Abstract**:

This paper proposes a novel two-stage hypergraph-based framework, dubbed ADaptive Hypergraph Neural Network (AD-HNN) to estimate multiple human poses from a single image, with a keypoint localization network and an Adaptive-Pose Hypergraph Neural Network (AP-HNN) added onto the former network. For providing better guided representations of AP-HNN, we employ a Semantic Interaction Convolution (SIC) module within the initial localization network to acquire more explicit predictions. Build upon this, we design a novel adaptive hypergraph to represent a human body for capturing high-order semantic relations among different joints. Notably, it can adaptively adjust the relations between joints and seek the most reasonable structure for the variable poses to benefit the keypoint localization. These two stages are combined to be trained in an end-to-end fashion. Unlike traditional Graph Convolutional Networks (GCNs) that are based on a fixed tree structure, AP-HNN can deal with ambiguity in human pose estimation. Experimental results demonstrate that the AD-HNN achieves state-of-the-art performance both on the MS-COCO, MPII and CrowdPose datasets.

----

## [329] Evo-ViT: Slow-Fast Token Evolution for Dynamic Vision Transformer

**Authors**: *Yifan Xu, Zhijie Zhang, Mengdan Zhang, Kekai Sheng, Ke Li, Weiming Dong, Liqing Zhang, Changsheng Xu, Xing Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20202](https://doi.org/10.1609/aaai.v36i3.20202)

**Abstract**:

Vision transformers (ViTs) have recently received explosive popularity, but the huge computational cost is still a severe issue. Since the computation complexity of ViT is quadratic with respect to the input sequence length, a mainstream paradigm for computation reduction is to reduce the number of tokens. Existing designs include structured spatial compression that uses a progressive shrinking pyramid to reduce the computations of large feature maps, and unstructured token pruning that dynamically drops redundant tokens. However, the limitation of existing token pruning lies in two folds: 1) the incomplete spatial structure caused by pruning is not compatible with structured spatial compression that is commonly used in modern deep-narrow transformers; 2) it usually requires a time-consuming pre-training procedure. To tackle the limitations and expand the applicable scenario of token pruning, we present Evo-ViT, a self-motivated slow-fast token evolution approach for vision transformers. Specifically, we conduct unstructured instance-wise token selection by taking advantage of the simple and effective global class attention that is native to vision transformers. Then, we propose to update the selected informative tokens and uninformative tokens with different computation paths, namely, slow-fast updating. Since slow-fast updating mechanism maintains the spatial structure and information flow, Evo-ViT can accelerate vanilla transformers of both flat and deep-narrow structures from the very beginning of the training process. Experimental results demonstrate that our method significantly reduces the computational cost of vision transformers while maintaining comparable performance on image classification. For example, our method accelerates DeiT-S by over 60% throughput while only sacrificing 0.4% top-1 accuracy on ImageNet-1K, outperforming current token pruning methods on both accuracy and efficiency.

----

## [330] MobileFaceSwap: A Lightweight Framework for Video Face Swapping

**Authors**: *Zhiliang Xu, Zhibin Hong, Changxing Ding, Zhen Zhu, Junyu Han, Jingtuo Liu, Errui Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20203](https://doi.org/10.1609/aaai.v36i3.20203)

**Abstract**:

Advanced face swapping methods have achieved appealing results. However, most of these methods have many parameters and computations, which makes it challenging to apply them in real-time applications or deploy them on edge devices like mobile phones. In this work, we propose a lightweight Identity-aware Dynamic Network (IDN) for subject-agnostic face swapping by dynamically adjusting the model parameters according to the identity information. In particular, we design an efficient Identity Injection Module (IIM) by introducing two dynamic neural network techniques, including the weights prediction and weights modulation. Once the IDN is updated, it can be applied to swap faces given any target image or video. The presented IDN contains only 0.50M parameters and needs 0.33G FLOPs per frame, making it capable for real-time video face swapping on mobile phones. In addition, we introduce a knowledge distillation-based method for stable training, and a loss reweighting module is employed to obtain better synthesized results. Finally, our method achieves comparable results with the teacher models and other state-of-the-art methods.

----

## [331] Clinical-BERT: Vision-Language Pre-training for Radiograph Diagnosis and Reports Generation

**Authors**: *Bin Yan, Mingtao Pei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20204](https://doi.org/10.1609/aaai.v36i3.20204)

**Abstract**:

In this paper, we propose a vision-language pre-training model, Clinical-BERT, for the medical domain, and devise three domain-specific tasks: Clinical Diagnosis (CD), Masked MeSH Modeling (MMM), Image-MeSH Matching (IMM), together with one general pre-training task: Masked Language Modeling (MLM), to pre-train the model. The CD task helps the model to learn medical domain knowledge by predicting disease from radiographs. Medical Subject Headings (MeSH) words are important semantic components in radiograph reports, and the MMM task helps the model focus on the prediction of MeSH words. The IMM task helps the model learn the alignment of MeSH words with radiographs by matching scores obtained by a two-level sparse attention: region sparse attention and word sparse attention. Region sparse attention generates corresponding visual features for each word, and word sparse attention enhances the contribution of images-MeSH matching to the matching scores. To the best of our knowledge, this is the first attempt to learn domain knowledge during pre-training for the medical domain. We evaluate the pre-training model on Radiograph Diagnosis and Reports Generation tasks across four challenging datasets: MIMIC-CXR, IU X-Ray, COV-CTR, and NIH, and achieve state-of-the-art results for all the tasks, which demonstrates the effectiveness of our pre-training model.

----

## [332] Inferring Prototypes for Multi-Label Few-Shot Image Classification with Word Vector Guided Attention

**Authors**: *Kun Yan, Chenbin Zhang, Jun Hou, Ping Wang, Zied Bouraoui, Shoaib Jameel, Steven Schockaert*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20205](https://doi.org/10.1609/aaai.v36i3.20205)

**Abstract**:

Multi-label few-shot image classification (ML-FSIC) is the task of assigning descriptive labels to previously unseen images, based on a small number of training examples. A key feature of the multi-label setting is that images often have multiple labels, which typically refer to different regions of the image. When estimating prototypes, in a metric-based setting, it is thus important to determine which regions are relevant for which labels, but the limited amount of training data makes this highly challenging. As a solution, in this paper we propose to use word embeddings as a form of prior knowledge about the meaning of the labels. In particular, visual prototypes are obtained by aggregating the local feature maps of the support images, using an attention mechanism that relies on the label embeddings. As an important advantage, our model can infer prototypes for unseen labels without the need for fine-tuning any model parameters, which demonstrates its strong generalization abilities. Experiments on COCO and PASCAL VOC furthermore show that our model substantially improves the current state-of-the-art.

----

## [333] Unsupervised Domain Adaptive Salient Object Detection through Uncertainty-Aware Pseudo-Label Learning

**Authors**: *Pengxiang Yan, Ziyi Wu, Mengmeng Liu, Kun Zeng, Liang Lin, Guanbin Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20206](https://doi.org/10.1609/aaai.v36i3.20206)

**Abstract**:

Recent advances in deep learning significantly boost the performance of salient object detection (SOD) at the expense of labeling larger-scale per-pixel annotations. To relieve the burden of labor-intensive labeling, deep unsupervised SOD methods have been proposed to exploit noisy labels generated by handcrafted saliency methods. However, it is still difficult to learn accurate saliency details from rough noisy labels. In this paper, we propose to learn saliency from synthetic but clean labels, which naturally has higher pixel-labeling quality without the effort of manual annotations. Specifically, we first construct a novel synthetic SOD dataset by a simple copy-paste strategy. Considering the large appearance differences between the synthetic and real-world scenarios, directly training with synthetic data will lead to performance degradation on real-world scenarios. To mitigate this problem, we propose a novel unsupervised domain adaptive SOD method to adapt between these two domains by uncertainty-aware self-training. Experimental results show that our proposed method outperforms the existing state-of-the-art deep unsupervised SOD methods on several benchmark datasets, and is even comparable to fully-supervised ones.

----

## [334] Transmission-Guided Bayesian Generative Model for Smoke Segmentation

**Authors**: *Siyuan Yan, Jing Zhang, Nick Barnes*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20207](https://doi.org/10.1609/aaai.v36i3.20207)

**Abstract**:

Smoke segmentation is essential to precisely localize wildﬁre so that it can be extinguished in an early phase. Although deep neural networks have achieved promising results on image segmentation tasks, they are prone to be overconﬁdent for smoke segmentation due to its non-rigid shape and transparent appearance. This is caused by both knowledge level uncertainty due to limited training data for accurate smoke segmentation and labeling level uncertainty representing the difﬁculty in labeling ground-truth. To effectively model the two types of uncertainty, we introduce a Bayesian generative model to simultaneously estimate the posterior distribution of model parameters and its predictions. Further, smoke images suffer from low contrast and ambiguity, inspired by physics-based image dehazing methods, we design a transmission-guided local coherence loss to guide the network to learn pair-wise relationships based on pixel distance and the transmission feature. To promote the development of this ﬁeld, we also contribute a high-quality smoke segmentation dataset, SMOKE5K, consisting of 1,400 real and 4,000 synthetic images with pixel-wise annotation. Experimental results on benchmark testing datasets illustrate that our model achieves both accurate predictions and reliable uncertainty maps representing model ignorance about its prediction. Our code and dataset are publicly available at: https://github.com/redlessme/Transmission-BVM.

----

## [335] Cross-Species 3D Face Morphing via Alignment-Aware Controller

**Authors**: *Xirui Yan, Zhenbo Yu, Bingbing Ni, Hang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20208](https://doi.org/10.1609/aaai.v36i3.20208)

**Abstract**:

We address cross-species 3D face morphing (i.e., 3D face morphing from human to animal), a novel problem with promising applications in social media and movie industry. It remains challenging how to preserve target structural information and source ﬁne-grained facial details simultaneously. To this end, we propose an Alignment-aware 3D Face Morphing (AFM) framework, which builds semantic-adaptive correspondence between source and target faces across species, via an alignment-aware controller mesh (Explicit Controller, EC) with explicit source/target mesh binding. Based on EC, we introduce Controller-Based Mapping (CBM), which builds semantic consistency between source and target faces according to the semantic importance of different face regions. Additionally, an inference-stage coarse-to-ﬁne strategy is exploited to produce ﬁne-grained meshes with rich facial details from rough meshes. Extensive experimental results in multiple people and animals demonstrate that our method produces high-quality deformation results.

----

## [336] Exploring Visual Context for Weakly Supervised Person Search

**Authors**: *Yichao Yan, Jinpeng Li, Shengcai Liao, Jie Qin, Bingbing Ni, Ke Lu, Xiaokang Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20209](https://doi.org/10.1609/aaai.v36i3.20209)

**Abstract**:

Person search has recently emerged as a challenging task that jointly addresses pedestrian detection and person re-identification. Existing approaches follow a fully supervised setting where both bounding box and identity annotations are available. However, annotating identities is labor-intensive, limiting the practicability and scalability of current frameworks. This paper inventively considers weakly supervised person search with only bounding box annotations. We propose to address this novel task by investigating three levels of context clues (i.e., detection, memory and scene) in unconstrained natural images. The first two are employed to promote local and global discriminative capabilities, while the latter enhances clustering accuracy. Despite its simple design, our CGPS boosts the baseline model by 8.8% in mAP on CUHK-SYSU. Surprisingly, it even achieves comparable performance with several supervised person search models. Our code is available at https://github. com/ljpadam/CGPS.

----

## [337] Cross-Modal Mutual Learning for Audio-Visual Speech Recognition and Manipulation

**Authors**: *Chih-Chun Yang, Wan-Cyuan Fan, Cheng-Fu Yang, Yu-Chiang Frank Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20210](https://doi.org/10.1609/aaai.v36i3.20210)

**Abstract**:

As a key characteristic in audio-visual speech recognition (AVSR), relating linguistic information observed across visual and audio data has been a challenge, benefiting not only audio/visual speech recognition (ASR/VSR) but also for manipulating data within/across modalities. In this paper, we present a feature disentanglement-based framework for jointly addressing the above tasks. By advancing cross-modal mutual learning strategies, our model is able to convert visual or audio-based linguistic features into modality-agnostic representations. Such derived linguistic representations not only allow one to perform ASR, VSR, and AVSR, but also to manipulate audio and visual data output based on the desirable subject identity and linguistic content information. We perform extensive experiments on different recognition and synthesis tasks to show that our model performs favorably against state-of-the-art approaches on each individual task, while ours is a unified solution that is able to jointly tackle the aforementioned audio-visual learning tasks.

----

## [338] Mutual Contrastive Learning for Visual Representation Learning

**Authors**: *Chuanguang Yang, Zhulin An, Linhang Cai, Yongjun Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20211](https://doi.org/10.1609/aaai.v36i3.20211)

**Abstract**:

We present a collaborative learning method called Mutual Contrastive Learning (MCL) for general visual representation learning. The core idea of MCL is to perform mutual interaction and transfer of contrastive distributions among a cohort of networks. A crucial component of MCL is Interactive Contrastive Learning (ICL). Compared with vanilla contrastive learning, ICL can aggregate cross-network embedding information and maximize the lower bound to the mutual information between two networks. This enables each network to learn extra contrastive knowledge from others, leading to better feature representations for visual recognition tasks. We emphasize that the resulting MCL is conceptually simple yet empirically powerful. It is a generic framework that can be applied to both supervised and self-supervised representation learning. Experimental results on image classification and transfer learning to object detection show that MCL can lead to consistent performance gains, demonstrating that MCL can guide the network to generate better feature representations. Code is available at https://github.com/winycg/MCL.

----

## [339] Temporal Action Proposal Generation with Background Constraint

**Authors**: *Haosen Yang, Wenhao Wu, Lining Wang, Sheng Jin, Boyang Xia, Hongxun Yao, Hujie Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20212](https://doi.org/10.1609/aaai.v36i3.20212)

**Abstract**:

Temporal action proposal generation (TAPG) is a challenging task that aims to locate action instances in untrimmed videos with temporal boundaries.
 To evaluate the confidence of proposals, the existing works typically predict action score of proposals that are supervised by the temporal Intersection-over-Union (tIoU) between proposal and the ground-truth.
 In this paper, we innovatively propose a general auxiliary Background Constraint idea to further suppress low-quality proposals, by utilizing the background prediction score to restrict the confidence of proposals. In this way, the Background Constraint concept can be easily plug-and-played into existing TAPG methods (BMN, GTAD). 
 From this perspective, we propose the Background Constraint Network (BCNet) to further take advantage of the rich information of action and background. Specifically, we introduce an Action-Background Interaction module for reliable confidence evaluation, which models the inconsistency between action and background by attention mechanisms at the frame and clip levels.
 Extensive experiments are conducted on two popular benchmarks, ActivityNet-1.3 and THUMOS14. The results demonstrate that our method outperforms state-of-the-art methods. Equipped with the existing action classifier, our method also achieves remarkable performance on the temporal action localization task.

----

## [340] Cross-Modal Federated Human Activity Recognition via Modality-Agnostic and Modality-Specific Representation Learning

**Authors**: *Xiaoshan Yang, Baochen Xiong, Yi Huang, Changsheng Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20213](https://doi.org/10.1609/aaai.v36i3.20213)

**Abstract**:

In this paper, we propose a new task of cross-modal federated human activity recognition (CMF-HAR), which is conducive to promote the large-scale use of the HAR model on more local devices. To address the new task, we propose a feature-disentangled activity recognition network (FDARN), which has five important modules of altruistic encoder, egocentric encoder, shared activity classifier, private activity classifier and modality discriminator. The altruistic encoder aims to collaboratively embed local instances on different clients into a modality-agnostic feature subspace. The egocentric encoder aims to produce modality-specific features that cannot be shared across clients with different modalities. The modality discriminator is used to adversarially guide the parameter learning of the altruistic and egocentric encoders. Through decentralized optimization with a spherical modality discriminative loss, our model can not only generalize well across different clients by leveraging the modality-agnostic features but also capture the modality-specific discriminative characteristics of each client. Extensive experiment results on four datasets demonstrate the effectiveness of our method.

----

## [341] Polygon-to-Polygon Distance Loss for Rotated Object Detection

**Authors**: *Yang Yang, Jifeng Chen, Xiaopin Zhong, Yuanlong Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20214](https://doi.org/10.1609/aaai.v36i3.20214)

**Abstract**:

There are two key issues that limit further improvements in the performance of existing rotational detectors: 1) Periodic sudden change of the parameters in the rotating bounding box (RBBox) definition causes a numerical discontinuity in the loss (such as smoothL1 loss). 2) There is a gap of optimization asynchrony between the loss in the RBBox regression and evaluation metrics. In this paper, we define a new distance formulation between two convex polygons describing the overlapping degree and non-overlapping degree. Based on this smooth distance, we propose a loss called Polygon-to-Polygon distance loss (P2P Loss). The distance is derived from the area sum of triangles specified by the vertexes of one polygon and the edges of the other. Therefore, the P2P Loss is continuous, differentiable, and inherently free from any RBBox definition. Our P2P Loss is not only consistent with the detection metrics but also able to measure how far, as well as how similar, a RBBox is from another one even when they are completely non-overlapping. These features allow the RetinaNet using the P2P Loss to achieve 79.15% mAP on the DOTA dataset, which is quite competitive compared with many state-of-the-art rotated object detectors.

----

## [342] An Empirical Study of GPT-3 for Few-Shot Knowledge-Based VQA

**Authors**: *Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, Lijuan Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20215](https://doi.org/10.1609/aaai.v36i3.20215)

**Abstract**:

Knowledge-based visual question answering (VQA) involves answering questions that require external knowledge not present in the image. Existing methods first retrieve knowledge from external resources, then reason over the selected knowledge, the input image, and question for answer prediction. However, this two-step approach could lead to mismatches that potentially limit the VQA performance. For example, the retrieved knowledge might be noisy and irrelevant to the question, and the re-embedded knowledge features during reasoning might deviate from their original meanings in the knowledge base (KB). To address this challenge, we propose PICa, a simple yet effective method that Prompts GPT3 via the use of Image Captions, for knowledge-based VQA. Inspired by GPT-3’s power in knowledge retrieval and question answering, instead of using structured KBs as in previous work, we treat GPT-3 as an implicit and unstructured KB that can jointly acquire and process relevant knowledge. Specifically, we first convert the image into captions (or tags) that GPT-3 can understand, then adapt GPT-3 to solve the VQA task in a few-shot manner by just providing a few in-context VQA examples. We further boost performance by carefully investigating: (i) what text formats best describe the image content, and (ii) how in-context examples can be better selected and used. PICa unlocks the first use of GPT-3 for multimodal tasks. By using only 16 examples, PICa surpasses the supervised state of the art by an absolute +8.6 points on the OK-VQA dataset. We also benchmark PICa on VQAv2, where PICa also shows a decent few-shot performance.

----

## [343] ACGNet: Action Complement Graph Network for Weakly-Supervised Temporal Action Localization

**Authors**: *Zichen Yang, Jie Qin, Di Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20216](https://doi.org/10.1609/aaai.v36i3.20216)

**Abstract**:

Weakly-supervised temporal action localization (WTAL) in untrimmed videos has emerged as a practical but challenging task since only video-level labels are available. Existing approaches typically leverage off-the-shelf segment-level features, which suffer from spatial incompleteness and temporal incoherence, thus limiting their performance. In this paper, we tackle this problem from a new perspective by enhancing segment-level representations with a simple yet effective graph convolutional network, namely action complement graph network (ACGNet). It facilitates the current video segment to perceive spatial-temporal dependencies from others that potentially convey complementary clues, implicitly mitigating the negative effects caused by the two issues above. By this means, the segment-level features are more discriminative and robust to spatial-temporal variations, contributing to higher localization accuracies. More importantly, the proposed ACGNet works as a universal module that can be flexibly plugged into different WTAL frameworks, while maintaining the end-to-end training fashion. Extensive experiments are conducted on the THUMOS'14 and ActivityNet1.2 benchmarks, where the state-of-the-art results clearly demonstrate the superiority of the proposed approach.

----

## [344] Enhancing Pseudo Label Quality for Semi-supervised Domain-Generalized Medical Image Segmentation

**Authors**: *Huifeng Yao, Xiaowei Hu, Xiaomeng Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20217](https://doi.org/10.1609/aaai.v36i3.20217)

**Abstract**:

Generalizing the medical image segmentation algorithms to unseen domains is an important research topic for computer-aided diagnosis and surgery. Most existing methods require a fully labeled dataset in each source domain. Although some researchers developed a semi-supervised domain generalized method, it still requires the domain labels. This paper presents a novel confidence-aware cross pseudo supervision algorithm for semi-supervised domain generalized medical image segmentation. The main goal is to enhance the pseudo label quality for unlabeled images from unknown distributions. To achieve it, we perform the Fourier transformation to learn low-level statistic information across domains and augment the images to incorporate cross-domain information. With these augmentations as perturbations, we feed the input to a confidence-aware cross pseudo supervision network to measure the variance of pseudo labels and regularize the network to learn with more confident pseudo labels. Our method sets new records on public datasets, i.e., M&Ms and SCGM. Notably, without using domain labels, our method surpasses the prior art that even uses domain labels by 11.67% on Dice on M&Ms dataset with 2% labeled data. Code is available at https://github.com/XMed-Lab/EPL SemiDG.

----

## [345] Image Difference Captioning with Pre-training and Contrastive Learning

**Authors**: *Linli Yao, Weiying Wang, Qin Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20218](https://doi.org/10.1609/aaai.v36i3.20218)

**Abstract**:

The Image Difference Captioning (IDC) task aims to describe the visual differences between two similar images with natural language. The major challenges of this task lie in two aspects: 1) fine-grained visual differences that require learning stronger vision and language association and 2) high-cost of manual annotations that leads to limited supervised data. To address these challenges, we propose a new modeling framework following the pre-training-finetuning paradigm. Specifically, we design three self-supervised tasks and contrastive learning strategies to align visual differences and text descriptions at a fine-grained level. Moreover, we propose a data expansion strategy to utilize extra cross-task supervision information, such as data for fine-grained image classification, to alleviate the limitation of available supervised IDC data. Extensive experiments on two IDC benchmark datasets, CLEVR-Change and Birds-to-Words, demonstrate the effectiveness of the proposed modeling framework. The codes and models will be released at https://github.com/yaolinli/IDC.

----

## [346] Safe Distillation Box

**Authors**: *Jingwen Ye, Yining Mao, Jie Song, Xinchao Wang, Cheng Jin, Mingli Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20219](https://doi.org/10.1609/aaai.v36i3.20219)

**Abstract**:

Knowledge distillation (KD) has recently emerged as a powerful strategy to transfer knowledge from a pre-trained teacher model to a lightweight student, and has demonstrated its unprecedented success over a wide spectrum of applications. 
 In spite of the encouraging results, the KD process \emph{per se} poses a potential threat to network ownership protection, since the knowledge contained in network can be effortlessly distilled and hence exposed to a malicious user.
 In this paper, we propose a novel framework, termed as Safe Distillation Box~(SDB), that allows us to wrap a pre-trained model in a virtual box for intellectual property protection. Specifically, SDB preserves the inference capability of the wrapped model to all users, but precludes KD from unauthorized users. For authorized users, on the other hand, SDB carries out a knowledge augmentation scheme to strengthen the KD performances and the results of the student model. In other words, all users may employ a model in SDB for inference, but only authorized users get access to KD from the model. The proposed SDB imposes no constraints over the model architecture, and may readily serve as a plug-and-play solution to protect the ownership of a pre-trained network. Experiments across various datasets and architectures demonstrate that, with SDB, the performance of an unauthorized KD drops significantly while that of an authorized gets enhanced, demonstrating the effectiveness of SDB.

----

## [347] Joint Deep Multi-Graph Matching and 3D Geometry Learning from Inhomogeneous 2D Image Collections

**Authors**: *Zhenzhang Ye, Tarun Yenamandra, Florian Bernard, Daniel Cremers*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20220](https://doi.org/10.1609/aaai.v36i3.20220)

**Abstract**:

Graph matching aims to establish correspondences between vertices of graphs such that both the node and edge attributes agree. Various learning-based methods were recently proposed for finding correspondences between image key points based on deep graph matching formulations. While these approaches mainly focus on learning node and edge attributes, they completely ignore the 3D geometry of the underlying 3D objects depicted in the 2D images. We fill this gap by proposing a trainable framework that takes advantage of graph neural networks for learning a deformable 3D geometry model from inhomogeneous image collections, i.e. a set of images that depict different instances of objects from the same category. Experimentally we demonstrate that our method outperforms recent learning-based approaches for graph matching considering both accuracy and cycle-consistency error, while we in addition obtain the underlying 3D geometry of the objects depicted in the 2D images.

----

## [348] Content-Variant Reference Image Quality Assessment via Knowledge Distillation

**Authors**: *Guanghao Yin, Wei Wang, Zehuan Yuan, Chuchu Han, Wei Ji, Shouqian Sun, Changhu Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20221](https://doi.org/10.1609/aaai.v36i3.20221)

**Abstract**:

Generally, humans are more skilled at perceiving differences between high-quality (HQ) and low-quality (LQ) images than directly judging the quality of a single LQ image. This situation also applies to image quality assessment (IQA). Although recent no-reference (NR-IQA) methods have made great progress to predict image quality free from the reference image, they still have the potential to achieve better performance since HQ image information is not fully exploited. In contrast, full-reference (FR-IQA) methods tend to provide more reliable quality evaluation, but its practicability is affected by the requirement for pixel-level aligned reference images. To address this, we firstly propose the content-variant reference method via knowledge distillation (CVRKD-IQA). Specifically, we use non-aligned reference (NAR) images to introduce various prior distributions of high-quality images. The comparisons of distribution differences between HQ and LQ images can help our model better assess the image quality. Further, the knowledge distillation transfers more HQ-LQ distribution difference information from the FR-teacher to the NAR-student and stabilizing CVRKD-IQA performance. Moreover, to fully mine the local-global combined information, while achieving faster inference speed, our model directly processes multiple image patches from the input with the MLP-mixer. Cross-dataset experiments verify that our model can outperform all NAR/NR-IQA SOTAs, even reach comparable performance than FR-IQA methods on some occasions. Since the content-variant and non-aligned reference HQ images are easy to obtain, our model can support more IQA applications with its robustness to content variations. Our code is available: https://github.com/guanghaoyin/CVRKD-IQA.

----

## [349] Width & Depth Pruning for Vision Transformers

**Authors**: *Fang Yu, Kun Huang, Meng Wang, Yuan Cheng, Wei Chu, Li Cui*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20222](https://doi.org/10.1609/aaai.v36i3.20222)

**Abstract**:

Transformer models have demonstrated their promising potential and achieved excellent performance on a series of computer vision tasks. However, the huge computational cost of vision transformers hinders their deployment and application to edge devices. Recent works have proposed to ﬁnd and remove the unimportant units of vision transformers. Despite achieving remarkable results, these methods take one dimension of network width into consideration and ignore network depth, which is another important dimension for pruning vision transformers. Therefore, we propose a Width & Depth Pruning (WDPruning) framework that reduces both width and depth dimensions simultaneously. Speciﬁcally, for width pruning, a set of learnable pruning-related parameters is used to adaptively adjust the width of transformer. For depth pruning, we introduce several shallow classiﬁers by using the intermediate information of the transformer blocks, which allows images to be classiﬁed by shallow classiﬁers instead of the deeper classiﬁers. In the inference period, all of the blocks after shallow classiﬁers can be dropped so they don’t bring additional parameters and computation. Experimental results on benchmark datasets demonstrate that the proposed method can signiﬁcantly reduce the computational costs of mainstream vision transformers such as DeiT and Swin Transformer with a minor accuracy drop. In particular, on ILSVRC-12, we achieve over 22% pruning ratio of FLOPs by compressing DeiT-Base, even with an increase of 0.14% Top-1 accuracy.

----

## [350] Anisotropic Fourier Features for Neural Image-Based Rendering and Relighting

**Authors**: *Huangjie Yu, Anpei Chen, Xin Chen, Lan Xu, Ziyu Shao, Jingyi Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20223](https://doi.org/10.1609/aaai.v36i3.20223)

**Abstract**:

Recent neural rendering techniques have greatly benefited image-based modeling and relighting tasks. They provide a continuous, compact, and parallelable representation by modeling the plenoptic function as multilayer perceptrons (MLPs). However, vanilla MLPs suffer from spectral biases on multidimensional datasets. Recent rescues based on isotropic Fourier features mapping mitigate the problem but still fall short of handling heterogeneity across different dimensions, causing imbalanced regression and visual artifacts such as excessive blurs. We present an anisotropic random Fourier features (RFF) mapping scheme to tackle spectral biases. We first analyze the influence of bandwidth from a different perspective: we show that the optimal bandwidth exhibits strong correlations with the frequency spectrum of the training data across various dimensions. We then introduce an anisotropic feature mapping scheme with multiple bandwidths to model the multidimensional signal characteristics. We further propose an efficient bandwidth searching scheme through iterative golden-section search that can significantly reduce the training overload from polynomial time to logarithm. Our anisotropic scheme directly applies to neural surface light-field rendering and image-based relighting. Comprehensive experiments show that our scheme can more faithfully model lighting conditions and object features as well as preserve fine texture details and smooth view transitions even when angular and spatial samples are highly imbalanced.

----

## [351] Self-Labeling Framework for Novel Category Discovery over Domains

**Authors**: *Qing Yu, Daiki Ikami, Go Irie, Kiyoharu Aizawa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20224](https://doi.org/10.1609/aaai.v36i3.20224)

**Abstract**:

Unsupervised domain adaptation (UDA) has been highly successful in transferring knowledge acquired from a label-rich source domain to a label-scarce target domain. Open-set domain adaptation (open-set DA) and universal domain adaptation (UniDA) have been proposed as solutions to the problem concerning the presence of additional novel categories in the target domain. Existing open-set DA and UniDA approaches treat all novel categories as one unified unknown class and attempt to detect this unknown class during the training process. However, the features of the novel categories learned by these methods are not discriminative. This limits the applicability of UDA in the further classification of these novel categories into their original categories, rather than assigning them to a single unified class. In this paper, we propose a self-labeling framework to cluster all target samples, including those in the ''unknown'' categories. We train the network to learn the representations of target samples via self-supervised learning (SSL) and to identify the seen and unseen (novel) target-sample categories simultaneously by maximizing the mutual information between labels and input data. We evaluated our approach under different DA settings and concluded that our method generally outperformed existing ones by a wide margin.

----

## [352] Efficient Compact Bilinear Pooling via Kronecker Product

**Authors**: *Tan Yu, Yunfeng Cai, Ping Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20225](https://doi.org/10.1609/aaai.v36i3.20225)

**Abstract**:

Bilinear pooling has achieved excellent performance in fine-grained recognition tasks. Nevertheless, high-dimensional bilinear features suffer from over-fitting and inefficiency. To alleviate these issues, compact bilinear pooling (CBP) methods were developed to generate low-dimensional features. Although the low-dimensional features from existing CBP methods enable high efficiency in subsequent classification, CBP methods themselves are inefficient. Thus, the inefficiency issue of the bilinear pooling is still unsolved. In this work, we propose an efficient compact bilinear pooling method to solve the inefficiency problem inherited in bilinear pooling thoroughly. It decomposes the huge-scale projection matrix into a two-level Kronecker product of several small-scale matrices. By exploiting the ``vec trick'' and the tensor modal product, we can obtain the compact bilinear feature through the decomposed projection matrices in a speedy manner. Systematic experiments on four public benchmarks using two backbones demonstrate the efficiency and effectiveness of the proposed method in fine-grained recognition.

----

## [353] Hybrid Graph Neural Networks for Few-Shot Learning

**Authors**: *Tianyuan Yu, Sen He, Yi-Zhe Song, Tao Xiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20226](https://doi.org/10.1609/aaai.v36i3.20226)

**Abstract**:

Graph neural networks (GNNs) have been used to tackle the few-shot learning (FSL) problem and shown great potentials under the transductive setting. However under the inductive setting, existing GNN based methods are less competitive. This is because they use an instance GNN as a label propagation/classification module, which is jointly meta-learned with a feature embedding network. This design is problematic because the classifier needs to adapt quickly to new tasks while the embedding does not. To overcome this problem, in this paper we propose a novel hybrid GNN (HGNN) model consisting of two GNNs, an instance GNN and a prototype GNN. Instead of label propagation, they act as feature embedding adaptation modules for quick adaptation of the meta-learned feature embedding to new tasks. Importantly they are designed to deal with a fundamental yet often neglected challenge in FSL, that is, with only a handful of shots per class, any few-shot classifier would be sensitive to badly sampled shots which are either outliers or can cause inter-class distribution overlapping. Extensive experiments show that our HGNN obtains new state-of-the-art on three FSL benchmarks. The code and models are available at https://github.com/TianyuanYu/HGNN.

----

## [354] SOIT: Segmenting Objects with Instance-Aware Transformers

**Authors**: *Xiaodong Yu, Dahu Shi, Xing Wei, Ye Ren, Tingqun Ye, Wenming Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20227](https://doi.org/10.1609/aaai.v36i3.20227)

**Abstract**:

This paper presents an end-to-end instance segmentation framework, termed SOIT, that Segments Objects with Instance-aware Transformers. Inspired by DETR, our method views instance segmentation as a direct set prediction problem and effectively removes the need for many hand-crafted components like RoI cropping, one-to-many label assignment, and non-maximum suppression (NMS). In SOIT, multiple queries are learned to directly reason a set of object embeddings of semantic category, bounding-box location, and pixel-wise mask in parallel under the global image context. The class and bounding-box can be easily embedded by a fixed-length vector. The pixel-wise mask, especially, is embedded by a group of parameters to construct a lightweight instance-aware transformer. Afterward, a full-resolution mask is produced by the instance-aware transformer without involving any RoI-based operation. Overall, SOIT introduces a simple single-stage instance segmentation framework that is both RoI- and NMS-free. Experimental results on the MS COCO dataset demonstrate that SOIT outperforms state-of-the-art instance segmentation approaches significantly. Moreover, the joint learning of multiple tasks in a unified query embedding can also substantially improve the detection performance. Code is available at https://github.com/yuxiaodongHRI/SOIT.

----

## [355] MSML: Enhancing Occlusion-Robustness by Multi-Scale Segmentation-Based Mask Learning for Face Recognition

**Authors**: *Ge Yuan, Huicheng Zheng, Jiayu Dong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20228](https://doi.org/10.1609/aaai.v36i3.20228)

**Abstract**:

In unconstrained scenarios, face recognition remains challenging, particularly when faces are occluded. Existing methods generalize poorly due to the distribution distortion induced by unpredictable occlusions. To tackle this problem, we propose a hierarchical segmentation-based mask learning strategy for face recognition, enhancing occlusion-robustness by integrating segmentation representations of occlusion into face recognition in the latent space. We present a novel multi-scale segmentation-based mask learning (MSML) network, which consists of a face recognition branch (FRB), an occlusion segmentation branch (OSB), and hierarchical elaborate feature masking (FM) operators. With the guidance of hierarchical segmentation representations of occlusion learned by the OSB, the FM operators can generate multi-scale latent masks to eliminate mistaken responses introduced by occlusions and purify the contaminated facial features at multiple layers. In this way, the proposed MSML network can effectively identify and remove the occlusions from feature representations at multiple levels and aggregate features from visible facial areas. Experiments on face verification and recognition under synthetic or realistic occlusions demonstrate the effectiveness of our method compared to state-of-the-art methods.

----

## [356] Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics

**Authors**: *Hangjie Yuan, Mang Wang, Dong Ni, Liangpeng Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20229](https://doi.org/10.1609/aaai.v36i3.20229)

**Abstract**:

Human-Object Interaction (HOI) detection is an essential task to understand human-centric images from a fine-grained perspective. Although end-to-end HOI detection models thrive, their paradigm of parallel human/object detection and verb class prediction loses two-stage methods' merit: object-guided hierarchy. The object in one HOI triplet gives direct clues to the verb to be predicted. In this paper, we aim to boost end-to-end models with object-guided statistical priors. Specifically, We propose to utilize a Verb Semantic Model (VSM) and use semantic aggregation to profit from this object-guided hierarchy. Similarity KL (SKL) loss is proposed to optimize VSM to align with the HOI dataset's priors. To overcome the static semantic embedding problem, we propose to generate cross-modality-aware visual and semantic features by Cross-Modal Calibration (CMC). The above modules combined composes Object-guided Cross-modal Calibration Network (OCN). Experiments conducted on two popular HOI detection benchmarks demonstrate the significance of incorporating the statistical prior knowledge and produce state-of-the-art performances. More detailed analysis indicates proposed modules serve as a stronger verb predictor and a more superior method of utilizing prior knowledge. The codes are available at https://github.com/JacobYuan7/OCN-HOI-Benchmark.

----

## [357] Task-Level Self-Supervision for Cross-Domain Few-Shot Learning

**Authors**: *Wang Yuan, Zhizhong Zhang, Cong Wang, Haichuan Song, Yuan Xie, Lizhuang Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20230](https://doi.org/10.1609/aaai.v36i3.20230)

**Abstract**:

Learning with limited labeled data is a long-standing problem. Among various solutions, episodic training progres-sively classifies a series of few-shot tasks and thereby is as-sumed to be beneficial for improving the model’s generalization ability. However, recent studies show that it is eveninferior to the baseline model when facing domain shift between base and novel classes. To tackle this problem, we pro-pose a domain-independent task-level self-supervised (TL-SS) method for cross-domain few-shot learning.TL-SS strategy promotes the general idea of label-based instance-levelsupervision to task-level self-supervision by augmenting mul-tiple views of tasks. Two regularizations on task consistencyand correlation metric are introduced to remarkably stabi-lize the training process and endow the generalization ability into the prediction model. We also propose a high-order associated encoder (HAE) being adaptive to various tasks.By utilizing 3D convolution module, HAE is able to generate proper parameters and enables the encoder to flexibly toany unseen tasks. Two modules complement each other andshow great promotion against state-of-the-art methods experimentally. Finally, we design a generalized task-agnostic test,where our intriguing findings highlight the need to re-think the generalization ability of existing few-shot approaches.

----

## [358] Improving 360 Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-Supervised Learning

**Authors**: *Ilwi Yun, Hyuk-Jae Lee, Chae-Eun Rhee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20231](https://doi.org/10.1609/aaai.v36i3.20231)

**Abstract**:

Due to difficulties in acquiring ground truth depth of equirectangular (360) images, the quality and quantity of equirectangular depth data today is insufficient to represent the various scenes in the world. Therefore, 360 depth estimation studies, which relied solely on supervised learning, are destined to produce unsatisfactory results. Although self-supervised learning methods focusing on equirectangular images (EIs) are introduced, they often have incorrect or non-unique solutions, causing unstable performance. In this paper, we propose 360 monocular depth estimation methods which improve on the areas that limited previous studies. First, we introduce a self-supervised 360 depth learning method that only utilizes gravity-aligned videos, which has the potential to eliminate the needs for depth data during the training procedure. Second, we propose a joint learning scheme realized by combining supervised and self-supervised learning. The weakness of each learning is compensated, thus leading to more accurate depth estimation. Third, we propose a non-local fusion block, which can further retain the global information encoded by vision transformer when reconstructing the depths. With the proposed methods, we successfully apply the transformer to 360 depth estimations, to the best of our knowledge, which has not been tried before. On several benchmarks, our approach achieves significant improvements over previous works and establishes a state of the art.

----

## [359] Homography Decomposition Networks for Planar Object Tracking

**Authors**: *Xinrui Zhan, Yueran Liu, Jianke Zhu, Yang Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20232](https://doi.org/10.1609/aaai.v36i3.20232)

**Abstract**:

Planar object tracking plays an important role in AI applications, such as robotics, visual servoing, and visual SLAM. Although the previous planar trackers work well in most scenarios, it is still a challenging task due to the rapid motion and large transformation between two consecutive frames. The essential reason behind this problem is that the condition number of such a non-linear system changes unstably when the searching range of the homography parameter space becomes larger. To this end, we propose a novel Homography Decomposition Networks~(HDN) approach that drastically reduces and stabilizes the condition number by decomposing the homography transformation into two groups. Specifically, a similarity transformation estimator is designed to predict the first group robustly by a deep convolution equivariant network. By taking advantage of the scale and rotation estimation with high confidence, a residual transformation is estimated by a simple regression model. Furthermore, the proposed end-to-end network is trained in a semi-supervised fashion. Extensive experiments show that our proposed approach outperforms the state-of-the-art planar tracking methods at a large margin on the challenging POT, UCSB and POIC datasets. Codes and models are available at https://github.com/zhanxinrui/HDN.

----

## [360] Patch Diffusion: A General Module for Face Manipulation Detection

**Authors**: *Baogen Zhang, Sheng Li, Guorui Feng, Zhenxing Qian, Xinpeng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20233](https://doi.org/10.1609/aaai.v36i3.20233)

**Abstract**:

Detection of manipulated face images has attracted a lot of interest recently. Various schemes have been proposed to tackle this challenging problem, where the patch-based approaches are shown to be promising. However, the existing patch-based approaches tend to treat different patches equally, which do not fully exploit the patch discrepancy for effective feature learning. In this paper, we propose a Patch Diffusion (PD) module which can be integrated into the existing face manipulation detection networks to boost the performance. The PD consists of Discrepancy Patch Feature Learning (DPFL) and Attention-Aware Message Passing (AMP). The DPFL effectively learns the patch features by a newly designed Pairwise Patch Loss (PPLoss), which takes both the patch importance and correlations into consideration. The AMP diffuses the patches through attention-aware message passing in a graph network, where the attentions are explicitly computed based on the patch features learnt in DPFL. We integrate our PD module into four recent face manipulation detection networks, and carry out the experiments on four popular datasets. The results demonstrate that our PD module is able to boost the performance of the existing networks for face manipulation detection.

----

## [361] Semi-supervised Object Detection with Adaptive Class-Rebalancing Self-Training

**Authors**: *Fangyuan Zhang, Tianxiang Pan, Bin Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20234](https://doi.org/10.1609/aaai.v36i3.20234)

**Abstract**:

While self-training achieves state-of-the-art results in semi-supervised object detection (SSOD), it severely suffers from foreground-background and foreground-foreground imbalances in SSOD. In this paper, we propose an Adaptive Class-Rebalancing Self-Training (ACRST) with a novel memory module called CropBank to alleviate these imbalances and generate unbiased pseudo-labels. Besides, we observe that both self-training and data-rebalancing procedures suffer from noisy pseudo-labels in SSOD. Therefore, we contribute a simple yet effective two-stage pseudo-label filtering scheme to obtain accurate supervision. Our method achieves competitive performance on MS-COCO and VOC benchmarks. When using only 1% labeled data of MS-COCO, our method achieves 17.02 mAP improvement over the supervised method and 5.32 mAP gains compared with state-of-the-arts.

----

## [362] Show Your Faith: Cross-Modal Confidence-Aware Network for Image-Text Matching

**Authors**: *Huatian Zhang, Zhendong Mao, Kun Zhang, Yongdong Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20235](https://doi.org/10.1609/aaai.v36i3.20235)

**Abstract**:

Image-text matching bridges vision and language, which is a crucial task in the field of multi-modal intelligence. The key challenge lies in how to measure image-text relevance accurately as matching evidence. Most existing works aggregate the local semantic similarities of matched region-word pairs as the overall relevance, and they typically assume that the matched pairs are equally reliable. However, although a region-word pair is locally matched across modalities, it may be inconsistent/unreliable from the global perspective of image-text, resulting in inaccurate relevance measurement. In this paper, we propose a novel Cross-Modal Confidence-Aware Network to infer the matching confidence that indicates the reliability of matched region-word pairs, which is combined with the local semantic similarities to refine the relevance measurement. Specifically, we first calculate the matching confidence via the relevance between the semantic of image regions and the complete described semantic in the image, with the text as a bridge. Further, to richly express the region semantics, we extend the region to its visual context in the image. Then, local semantic similarities are weighted with the inferred confidence to filter out unreliable matched pairs in aggregating. Comprehensive experiments show that our method achieves state-of-the-art performance on benchmarks Flickr30K and MSCOCO.

----

## [363] SCSNet: An Efficient Paradigm for Learning Simultaneously Image Colorization and Super-resolution

**Authors**: *Jiangning Zhang, Chao Xu, Jian Li, Yue Han, Yabiao Wang, Ying Tai, Yong Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20236](https://doi.org/10.1609/aaai.v36i3.20236)

**Abstract**:

In the practical application of restoring low-resolution gray-scale images, we generally need to run three separate processes of image colorization, super-resolution, and dows-sampling operation for the target device. However, this pipeline is redundant and inefficient for the independent processes, and some inner features could have been shared. Therefore, we present an efficient paradigm to perform Simultaneously Image Colorization and Super-resolution (SCS) and propose an end-to-end SCSNet to achieve this goal. The proposed method consists of two parts: colorization branch for learning color information that employs the proposed plug-and-play Pyramid Valve Cross Attention (PVCAttn) module to aggregate feature maps between source and reference images; and super-resolution branch for integrating color and texture information to predict target images, which uses the designed Continuous Pixel Mapping (CPM) module to predict high-resolution images at continuous magnification. Furthermore, our SCSNet supports both automatic and referential modes that is more flexible for practical application. Abundant experiments demonstrate the superiority of our method for generating authentic images over state-of-the-art methods, e.g., averagely decreasing FID by 1.8 and 5.1 compared with current best scores for automatic and referential modes, respectively, while owning fewer parameters (more than x2) and faster running speed (more than x3).

----

## [364] Energy-Based Generative Cooperative Saliency Prediction

**Authors**: *Jing Zhang, Jianwen Xie, Zilong Zheng, Nick Barnes*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20237](https://doi.org/10.1609/aaai.v36i3.20237)

**Abstract**:

Conventional saliency prediction models typically learn a deterministic mapping from an image to its saliency map, and thus fail to explain the subjective nature of human attention.  In this paper, to model the uncertainty of visual saliency, we study the saliency prediction problem from the perspective of generative models by learning a conditional probability distribution over the saliency map given an input image, and treating the saliency prediction as a sampling process from the learned distribution. Specifically, we propose a generative cooperative saliency prediction framework, where a conditional latent variable model~(LVM) and a conditional energy-based model~(EBM) are jointly trained to predict salient objects in a cooperative manner. The LVM serves as a fast but coarse predictor to efficiently produce an initial saliency map, which is then refined by the iterative Langevin revision of the EBM that serves as a slow but fine predictor. Such a coarse-to-fine cooperative saliency prediction strategy offers the best of both worlds. Moreover, we propose a ``cooperative learning while recovering" strategy and apply it to weakly supervised saliency prediction, where saliency annotations of training images are partially observed. Lastly, we find that the learned energy function in the EBM can serve as a refinement module that can refine the results of other pre-trained saliency prediction models. Experimental results show that our model can produce a set of diverse and plausible saliency maps of an image, and obtain state-of-the-art performance in both fully supervised and weakly supervised saliency prediction tasks.

----

## [365] Attention-Based Transformation from Latent Features to Point Clouds

**Authors**: *Kaiyi Zhang, Ximing Yang, Yuan Wu, Cheng Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20238](https://doi.org/10.1609/aaai.v36i3.20238)

**Abstract**:

In point cloud generation and completion, previous methods for transforming latent features to point clouds are generally based on fully connected layers (FC-based) or folding operations (Folding-based). However, point clouds generated by FC-based methods are usually troubled by outliers and rough surfaces. For folding-based methods, their data flow is large, convergence speed is slow, and they are also hard to handle the generation of non-smooth surfaces. In this work, we propose AXform, an attention-based method to transform latent features to point clouds. AXform first generates points in an interim space, using a fully connected layer. These interim points are then aggregated to generate the target point cloud. AXform takes both parameter sharing and data flow into account, which makes it has fewer outliers, fewer network parameters, and a faster convergence speed. The points generated by AXform do not have the strong 2-manifold constraint, which improves the generation of non-smooth surfaces. When AXform is expanded to multiple branches for local generations, the centripetal constraint makes it has properties of self-clustering and space consistency, which further enables unsupervised semantic segmentation. We also adopt this scheme and design AXformNet for point cloud completion. Considerable experiments on different datasets show that our methods achieve state-of-the-art results.

----

## [366] Suppressing Static Visual Cues via Normalizing Flows for Self-Supervised Video Representation Learning

**Authors**: *Manlin Zhang, Jinpeng Wang, Andy J. Ma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20239](https://doi.org/10.1609/aaai.v36i3.20239)

**Abstract**:

Despite the great progress in video understanding made by deep convolutional neural networks, feature representation learned by existing methods may be biased to static visual cues. To address this issue, we propose a novel method to suppress static visual cues (SSVC) based on probabilistic analysis for self-supervised video representation learning. In our method, video frames are first encoded to obtain latent variables under standard normal distribution via normalizing flows. By modelling static factors in a video as a random variable, the conditional distribution of each latent variable becomes shifted and scaled normal. Then, the less-varying latent variables along time are selected as static cues and suppressed to generate motion-preserved videos. Finally, positive pairs are constructed by motion-preserved videos for contrastive learning to alleviate the problem of representation bias to static cues. The less-biased video representation can be better generalized to various downstream tasks. Extensive experiments on publicly available benchmarks demonstrate that the proposed method outperforms the state of the art when only single RGB modality is used for pre-training.

----

## [367] LGD: Label-Guided Self-Distillation for Object Detection

**Authors**: *Peizhen Zhang, Zijian Kang, Tong Yang, Xiangyu Zhang, Nanning Zheng, Jian Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20240](https://doi.org/10.1609/aaai.v36i3.20240)

**Abstract**:

In this paper, we propose the first self-distillation framework for general object detection, termed LGD (Label-Guided self-Distillation). Previous studies rely on a strong pretrained teacher to provide instructive knowledge that could be unavailable in real-world scenarios. Instead, we generate an instructive knowledge by inter-and-intra relation modeling among objects, requiring only student representations and regular labels. Concretely, our framework involves sparse label-appearance encoding, inter-object relation adaptation and intra-object knowledge mapping to obtain the instructive knowledge. They jointly form an implicit teacher at training phase, dynamically dependent on labels and evolving student representations. Modules in LGD are trained end-to-end with student detector and are discarded in inference. Experimentally, LGD obtains decent results on various detectors, datasets, and extensive tasks like instance segmentation. For example in MS-COCO dataset, LGD improves RetinaNet with ResNet-50 under 2x single-scale training from 36.2% to 39.0% mAP (+ 2.8%). It boosts much stronger detectors like FCOS with ResNeXt-101 DCN v2 under 2x multi-scale training from 46.1% to 47.9% (+ 1.8%).
Compared with a classical teacher-based method FGFI, LGD not only performs better without requiring pretrained teacher but also reduces 51% training cost beyond inherent student learning.

----

## [368] Uncertainty Modeling with Second-Order Transformer for Group Re-identification

**Authors**: *Quan Zhang, Jian-Huang Lai, Zhan-Xiang Feng, Xiaohua Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20241](https://doi.org/10.1609/aaai.v36i3.20241)

**Abstract**:

Group re-identification (G-ReID) focuses on associating the group images containing the same persons under different cameras. The key challenge of G-ReID is that all the cases of the intra-group member and layout variations are hard to exhaust. To this end, we propose a novel uncertainty modeling, which treats each image as a distribution depending on the current member and layout, then digs out potential group features by random samplings. Based on potential and original group features, uncertainty modeling can learn better decision boundaries, which is implemented by two modules, member variation module (MVM) and layout variation module (LVM). Furthermore, we propose a novel second-order transformer framework (SOT), which is inspired by the fact that the position modeling in the transformer is coped with the G-ReID task. SOT is composed of the intra-member module and inter-member module. Specifically, the intra-member module extracts the first-order token for each member, and then the inter-member module learns a second-order token as a group feature by the above first-order tokens, which can be regarded as the token of tokens. A large number of experiments have been conducted on three available datasets, including CSG, DukeGroup and RoadGroup. The experimental results show that the proposed SOT outperforms all previous state-of-the-art methods.

----

## [369] Deep Spatial Adaptive Network for Real Image Demosaicing

**Authors**: *Tao Zhang, Ying Fu, Cheng Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20242](https://doi.org/10.1609/aaai.v36i3.20242)

**Abstract**:

Demosaicing is the crucial step in the image processing pipeline and is a highly ill-posed inverse problem. Recently, various deep learning based demosaicing methods have achieved promising performance, but they often design the same nonlinear mapping function for different spatial location and are not well consider the difference of mosaic pattern for each color. In this paper, we propose a deep spatial adaptive network (SANet) for real image demosaicing, which can adaptively learn the nonlinear mapping function for different locations. The weights of spatial adaptive convolution layer are generated by the pattern information in the receptive filed. Besides, we collect a paired real demosaicing dataset to train and evaluate the deep network, which can make the learned demosaicing network more practical in the real world. The experimental results show that our SANet outperforms the state-of-the-art methods under both comprehensive quantitative metrics and perceptive quality in both noiseless and noisy cases.

----

## [370] MAGIC: Multimodal relAtional Graph adversarIal inferenCe for Diverse and Unpaired Text-Based Image Captioning

**Authors**: *Wenqiao Zhang, Haochen Shi, Jiannan Guo, Shengyu Zhang, Qingpeng Cai, Juncheng Li, Sihui Luo, Yueting Zhuang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20243](https://doi.org/10.1609/aaai.v36i3.20243)

**Abstract**:

Text-based image captioning (TextCap) requires simultaneous comprehension of visual content and reading the text of images to generate a natural language description. Although a task can teach machines to understand the complex human environment further given that text is omnipresent in our daily surroundings, it poses additional challenges in normal captioning. A text-based image intuitively contains abundant and complex multimodal relational content, that is, image details can be described diversely from multiview rather than a single caption. Certainly, we can introduce additional paired training data to show the diversity of images' descriptions, this process is labor-intensive and time-consuming for TextCap pair annotations with extra texts. Based on the insight mentioned above, we investigate how to generate diverse captions that focus on different image parts using an unpaired training paradigm. We propose the Multimodal relAtional Graph adversarIal InferenCe (MAGIC) framework for diverse and unpaired TextCap. This framework can adaptively construct multiple multimodal relational graphs of images and model complex relationships among graphs to represent descriptive diversity. Moreover, a cascaded generative adversarial network is developed from modeled graphs to infer the unpaired caption generation in image–sentence feature alignment and linguistic coherence levels. We validate the effectiveness of MAGIC in generating diverse captions from different relational information items of an image. Experimental results show that MAGIC can generate very promising outcomes without using any image–caption training pairs.

----

## [371] Class Guided Channel Weighting Network for Fine-Grained Semantic Segmentation

**Authors**: *Xiang Zhang, Wanqing Zhao, Hangzai Luo, Jinye Peng, Jianping Fan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20244](https://doi.org/10.1609/aaai.v36i3.20244)

**Abstract**:

Deep learning has achieved promising performance on semantic segmentation, but few works focus on semantic segmentation at the fine-grained level. Fine-grained semantic segmentation requires recognizing and distinguishing hundreds of sub-categories. Due to the high similarity of different sub-categories and large variations in poses, scales, rotations, and color of the same sub-category in the fine-grained image set, the performance of traditional semantic segmentation methods will decline sharply. To alleviate these dilemmas, a new approach, named Class Guided Channel Weighting Network (CGCWNet), is developed in this paper to enable fine-grained semantic segmentation. For the large intra-class variations, we propose a Class Guided Weighting (CGW) module, which learns the image-level fine-grained category probabilities by exploiting second-order feature statistics, and use them as global information to guide semantic segmentation. For the high similarity between different sub-categories, we specially build a Channel Relationship Attention (CRA) module to amplify the distinction of features. Furthermore, a Detail Enhanced Guided Filter (DEGF) module is proposed to refine the boundaries of object masks by using an edge contour cue extracted from the enhanced original image. Experimental results on PASCAL VOC 2012 and six fine-grained image sets show that our proposed CGCWNet has achieved state-of-the-art results.

----

## [372] Context-Based Contrastive Learning for Scene Text Recognition

**Authors**: *Xinyun Zhang, Binwu Zhu, Xufeng Yao, Qi Sun, Ruiyu Li, Bei Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20245](https://doi.org/10.1609/aaai.v36i3.20245)

**Abstract**:

Pursuing accurate and robust recognizers has been a long-lasting goal for scene text recognition (STR) researchers. Recently, attention-based methods have demonstrated their effectiveness and achieved impressive results on public benchmarks. The attention mechanism enables models to recognize scene text with severe visual distortions by leveraging contextual information. However, recent studies revealed that the implicit over-reliance of context leads to catastrophic out-of-vocabulary performance. On the contrary to the superior accuracy of the seen text, models are prone to misrecognize unseen text even with good image quality. We propose a novel framework, Context-based contrastive learning (ConCLR), to alleviate this issue. Our proposed method first generates characters with different contexts via simple image concatenation operations and then optimizes contrastive loss on their embeddings. By pulling together clusters of identical characters within various contexts and pushing apart clusters of different characters in embedding space, ConCLR suppresses the side-effect of overfitting to specific contexts and learns a more robust representation. Experiments show that ConCLR significantly improves out-of-vocabulary generalization and achieves state-of-the-art performance on public benchmarks together with attention-based recognizers.

----

## [373] Learning Network Architecture for Open-Set Recognition

**Authors**: *Xuelin Zhang, Xuelian Cheng, Donghao Zhang, C. Paul Bonnington, Zongyuan Ge*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20246](https://doi.org/10.1609/aaai.v36i3.20246)

**Abstract**:

Given the incomplete knowledge of classes that exist in the world, Open-set Recognition (OSR) enables networks to identify and reject the unseen classes after training. This problem of breaking the common closed-set assumption is far from being solved. Recent studies focus on designing new losses, neural network encoding structures, and calibration methods to optimize a feature space for OSR relevant tasks. In this work, we make the first attempt to tackle OSR by searching the architecture of a Neural Network (NN) under the open-set assumption. In contrast to the prior arts, we develop a mechanism to both search the architecture of the network and train a network suitable for tackling OSR. Inspired by the compact abating probability (CAP) model, which is theoretically proven to reduce the open space risk, we regularize the searching space by VAE contrastive learning. To discover a more robust structure for OSR, we propose Pseudo Auxiliary Searching (PAS), in which we split a pretended set of know-unknown classes from the original training set in the searching phase, hence enabling the super-net to explore an effective architecture that can handle unseen classes in advance. We demonstrate the benefits of this learning pipeline on 5 OSR datasets, including MNIST, SVHN, CIFAR10, CIFARAdd10, and CIFARAdd50, where our approach outperforms prior state-of-the-art networks designed by humans. To spark research in this field, our code is available at https://github.com/zxl101/NAS OSR.

----

## [374] An Adversarial Framework for Generating Unseen Images by Activation Maximization

**Authors**: *Yang Zhang, Wang Zhou, Gaoyuan Zhang, David D. Cox, Shiyu Chang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20247](https://doi.org/10.1609/aaai.v36i3.20247)

**Abstract**:

Activation maximization (AM) refers to the task of generating input examples that maximize the activation of a target class of a classifier, which can be used for class-conditional image generation and model interpretation. A popular class of AM method, GAN-based AM, introduces a GAN pre-trained on a large image set, and performs AM over its input random seed or style embeddings, so that the generated images are natural and adversarial attacks are prevented. Most of these methods would require the image set to contain some images of the target class to be visualized. Otherwise they tend to generate other seen class images that most maximizes the target class activation. In this paper, we aim to tackle the case where information about the target class is completely removed from the image set. This would ensure that the generated images truly reflect the target class information residing in the classifier, not the target class information in the image set, which contributes to a more faithful interpretation technique. To this end, we propose PROBEGAN, a GAN-based AM algorithm capable of generating image classes unseen in the image set. Rather than using a pre-trained GAN, PROBEGAN trains a new GAN with AM explicitly included in its training objective. PROBEGAN consists of a class-conditional generator, a seen-class discriminator, and an all-class unconditional discriminator. It can be shown that such a framework can generate images with the features of the unseen target class, while retaining the naturalness as depicted in the image set. Experiments have shown that PROBEGAN can generate unseen-class images with much higher quality than the baselines. We also explore using PROBEGAN as a model interpretation tool. Our code is at https://github.com/csmiler/ProbeGAN/.

----

## [375] Contrastive Spatio-Temporal Pretext Learning for Self-Supervised Video Representation

**Authors**: *Yujia Zhang, Lai-Man Po, Xuyuan Xu, Mengyang Liu, Yexin Wang, Weifeng Ou, Yuzhi Zhao, Wing Yin Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20248](https://doi.org/10.1609/aaai.v36i3.20248)

**Abstract**:

Spatio-temporal representation learning is critical for video self-supervised representation. Recent approaches mainly use contrastive learning and pretext tasks. However, these approaches learn representation by discriminating sampled instances via feature similarity in the latent space while ignoring the intermediate state of the learned representations, which limits the overall performance. In this work, taking into account the degree of similarity of sampled instances as the intermediate state, we propose a novel pretext task - spatio-temporal overlap rate (STOR) prediction. It stems from the observation that humans are capable of discriminating the overlap rates of videos in space and time. This task encourages the model to discriminate the STOR of two generated samples to learn the representations. Moreover, we employ a joint optimization combining pretext tasks with contrastive learning to further enhance the spatio-temporal representation learning. We also study the mutual influence of each component in the proposed scheme. Extensive experiments demonstrate that our proposed STOR task can favor both contrastive learning and pretext tasks and the joint optimization scheme can significantly improve the spatio-temporal representation in video understanding. The code is available at https://github.com/Katou2/CSTP.

----

## [376] Pose-Invariant Face Recognition via Adaptive Angular Distillation

**Authors**: *Zhenduo Zhang, Yongru Chen, Wenming Yang, Guijin Wang, Qingmin Liao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20249](https://doi.org/10.1609/aaai.v36i3.20249)

**Abstract**:

Pose-invariant face recognition is a practically useful but challenging task. This paper introduces a novel method to learn pose-invariant feature representation without normalizing profile faces to frontal ones or learning disentangled features. We first design a novel strategy to learn pose-invariant feature embeddings by distilling the angular knowledge of frontal faces extracted by teacher network to student network, which enables the handling of faces with large pose variations. In this way, the features of faces across variant poses can cluster compactly for the same person to create a pose-invariant face representation. Secondly, we propose a Pose-Adaptive Angular Distillation loss to mitigate the negative effect of uneven distribution of face poses in the training dataset to pay more attention to the samples with large pose variations. Extensive experiments on two challenging benchmarks (IJB-A and CFP-FP) show that our approach consistently outperforms the existing methods.

----

## [377] End-to-End Learning the Partial Permutation Matrix for Robust 3D Point Cloud Registration

**Authors**: *Zhiyuan Zhang, Jiadai Sun, Yuchao Dai, Dingfu Zhou, Xibin Song, Mingyi He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20250](https://doi.org/10.1609/aaai.v36i3.20250)

**Abstract**:

Even though considerable progress has been made in deep learning-based 3D point cloud processing, how to obtain accurate correspondences for robust registration remains a major challenge because existing hard assignment methods cannot deal with outliers naturally. Alternatively, the soft matching-based methods have been proposed to learn the matching probability rather than hard assignment. However, in this paper, we prove that these methods have an inherent ambiguity causing many deceptive correspondences. To address the above challenges, we propose to learn a partial permutation matching matrix, which does not assign corresponding points to outliers, and implements hard assignment to prevent ambiguity. However, this proposal poses two new problems, i.e. existing hard assignment algorithms can only solve a full rank permutation matrix rather than a partial permutation matrix, and this desired matrix is defined in the discrete space, which is non-differentiable. In response, we design a dedicated soft-to-hard (S2H) matching procedure within the registration pipeline consisting of two steps: solving the soft matching matrix (S-step) and projecting this soft matrix to the partial permutation matrix (H-step). Specifically, we augment the profit matrix before the hard assignment to solve an augmented permutation matrix, which is cropped to achieve the final partial permutation matrix. Moreover, to guarantee end-to-end learning, we supervise the learned partial permutation matrix but propagate the gradient to the soft matrix instead. Our S2H matching procedure can be easily integrated with existing registration frameworks, which has been verified in representative frameworks including DCP, RPMNet, and DGR. Extensive experiments have validated our method, which creates a new state-of-the-art performance.

----

## [378] PetsGAN: Rethinking Priors for Single Image Generation

**Authors**: *Zicheng Zhang, Yinglu Liu, Congying Han, Hailin Shi, Tiande Guo, Bowen Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20251](https://doi.org/10.1609/aaai.v36i3.20251)

**Abstract**:

Single image generation (SIG), described as generating diverse samples that have the same visual content as the given natural image, is first introduced by SinGAN, which builds a pyramid of GANs to progressively learn the internal patch distribution of the single image. It shows excellent performance in a wide range of image manipulation tasks. However, SinGAN has some limitations. Firstly, due to lack of semantic information, SinGAN cannot handle the object images well as it does on the scene and texture images. Secondly, the independent progressive training scheme is time-consuming and easy to cause artifacts accumulation. To tackle these problems, in this paper, we dig into the single image generation problem and improve SinGAN by fully-utilization of internal and external priors. The main contributions of this paper include: 1) We interpret single image generation from the perspective of the general generative task, that is, to learn a diverse distribution from the Dirac distribution composed of a single image. In order to solve this non-trivial problem, we construct a regularized latent variable model to formulate SIG. To the best of our knowledge, it is the first time to give a clear formulation and optimization goal of SIG, and all the existing methods for SIG can be regarded as special cases of this model. 2) We design a novel Prior-based end-to-end training GAN (PetsGAN), which is infused with internal prior and external prior to overcome the problems of SinGAN. For one thing, we employ the pre-trained GAN model to inject external prior for image generation, which can alleviate the problem of lack of semantic information and generate natural, reasonable and diverse samples, even for the object image. For another, we fully-utilize the internal prior by a differential Patch Matching module and an effective reconstruction network to generate consistent and realistic texture. 3) We construct abundant of qualitative and quantitative experiments on three datasets. The experimental results show our method surpasses other methods on both generated image quality, diversity, and training speed. Moreover, we apply our method to other image manipulation tasks (e.g., style transfer, harmonization) and the results further prove the effectiveness and efficiency of our method.

----

## [379] Nested Hierarchical Transformer: Towards Accurate, Data-Efficient and Interpretable Visual Understanding

**Authors**: *Zizhao Zhang, Han Zhang, Long Zhao, Ting Chen, Sercan Ö. Arik, Tomas Pfister*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20252](https://doi.org/10.1609/aaai.v36i3.20252)

**Abstract**:

Hierarchical structures are popular in recent vision transformers, however, they require sophisticated designs and massive datasets to work well. In this paper, we explore the idea of nesting basic local transformers on non-overlapping image blocks and aggregating them in a hierarchical way. We find that the block aggregation function plays a critical role in enabling cross-block non-local information communication. This observation leads us to design a simplified architecture that requires minor code changes upon the original vision transformer. The benefits of the proposed judiciously-selected design are threefold:  (1) NesT converges faster and requires much less training data to achieve good generalization on both ImageNet and small datasets like CIFAR; (2) when extending our key ideas to image generation, NesT leads to a strong decoder that is 8 times faster than previous transformer-based generators; and (3) we show that decoupling the feature learning and abstraction processes via this nested hierarchy in our design enables constructing a novel method (named GradCAT) for visually interpreting the learned model. Source code is available https://github.com/google-research/nested-transformer.

----

## [380] OA-FSUI2IT: A Novel Few-Shot Cross Domain Object Detection Framework with Object-Aware Few-Shot Unsupervised Image-to-Image Translation

**Authors**: *Lifan Zhao, Yunlong Meng, Lin Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20253](https://doi.org/10.1609/aaai.v36i3.20253)

**Abstract**:

Unsupervised image-to-image (UI2I) translation methods aim to learn a mapping between different visual domains with well-preserved content and consistent structure. It has been proven that the generated images are quite useful for enhancing the performance of computer vision tasks like object detection in a different domain with distribution discrepancies. Current methods require large amounts of images in both source and target domains for successful translation. However, data collection and annotations in many scenarios are infeasible or even impossible. In this paper, we propose an Object-Aware Few-Shot UI2I Translation (OA-FSUI2IT) framework to address the few-shot cross domain (FSCD) object detection task with limited unlabeled images in the target domain. To this end, we first introduce a discriminator augmentation (DA) module into the OA-FSUI2IT framework for successful few-shot UI2I translation. Then, we present a patch pyramid contrastive learning (PPCL) strategy to further improve the quality of the generated images. Last, we propose a self-supervised content-consistency (SSCC) loss to enforce the content-consistency in the translation. We implement extensive experiments to demonstrate the effectiveness of our OA-FSUI2IT framework for FSCD object detection and achieve state-of-the-art performance on the benchmarks of Normal-to-Foggy, Day-to-Night, and Cross-scene adaptation. The source code of our proposed method is also available at https://github.com/emdata-ailab/FSCD-Det.

----

## [381] Static-Dynamic Co-teaching for Class-Incremental 3D Object Detection

**Authors**: *Na Zhao, Gim Hee Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20254](https://doi.org/10.1609/aaai.v36i3.20254)

**Abstract**:

Deep learning-based approaches have shown remarkable performance in the 3D object detection task. However, they suffer from a catastrophic performance drop on the originally trained classes when incrementally learning new classes without revisiting the old data. This "catastrophic forgetting" phenomenon impedes the deployment of 3D object detection approaches in real-world scenarios, where continuous learning systems are needed. In this paper, we study the unexplored yet important class-incremental 3D object detection problem and present the first solution - SDCoT, a novel static-dynamic co-teaching method. Our SDCoT alleviates the catastrophic forgetting of old classes via a static teacher, which provides pseudo annotations for old classes in the new samples and regularizes the current model by extracting previous knowledge with a distillation loss. At the same time, SDCoT consistently learns the underlying knowledge from new data via a dynamic teacher. We conduct extensive experiments on two benchmark datasets and demonstrate the superior performance of our SDCoT over baseline approaches in several incremental learning scenarios. Our code is available at https://github.com/Na-Z/SDCoT.

----

## [382] Local Surface Descriptor for Geometry and Feature Preserved Mesh Denoising

**Authors**: *Wenbo Zhao, Xianming Liu, Junjun Jiang, Debin Zhao, Ge Li, Xiangyang Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20255](https://doi.org/10.1609/aaai.v36i3.20255)

**Abstract**:

3D meshes are widely employed to represent geometry structure of 3D shapes. Due to limitation of scanning sensor precision and other issues, meshes are inevitably affected by noise, which hampers the subsequent applications. Convolultional neural networks (CNNs) achieve great success in image processing tasks, including 2D image denoising, and have been proven to own the capacity of modeling complex features at different scales, which is also particularly useful for mesh denoising. However, due to the nature of irregular structure, CNNs-based denosing strategies cannot be trivially applied for meshes. To circumvent this limitation, in the paper, we propose the local surface descriptor (LSD), which is able to transform the local deformable surface around a face into 2D grid representation and thus facilitates the deployment of CNNs to generate denoised face normals. To verify the superiority of LSD, we directly feed LSD into the classical Resnet without any complicated network design. The extensive experimental results show that, compared to the state-of-the-arts, our method achieves encouraging performance with respect to both objective and subjective evaluations.

----

## [383] Boosting Generative Zero-Shot Learning by Synthesizing Diverse Features with Attribute Augmentation

**Authors**: *Xiaojie Zhao, Yuming Shen, Shidong Wang, Haofeng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20256](https://doi.org/10.1609/aaai.v36i3.20256)

**Abstract**:

The recent advance in deep generative models outlines a promising perspective in the realm of Zero-Shot Learning (ZSL).  Most generative ZSL methods use category semantic attributes plus a Gaussian noise to generate visual features. After generating unseen samples, this family of approaches effectively transforms the ZSL problem into a supervised classification scheme. However, the existing models use a single semantic attribute, which contains the complete attribute information of the category. The generated data also carry the complete attribute information, but in reality, visual samples usually have limited attributes. Therefore, the generated data from attribute could have incomplete semantics. Based on this fact, we propose a novel framework to boost ZSL by synthesizing diverse features. This method uses augmented semantic attributes to train the generative model, so as to simulate the real distribution of visual features. We evaluate the proposed model on four benchmark datasets, observing significant performance improvement against the state-of-the-art.

----

## [384] Self-Supervised Pretraining for RGB-D Salient Object Detection

**Authors**: *Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu, Xiang Ruan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20257](https://doi.org/10.1609/aaai.v36i3.20257)

**Abstract**:

Existing CNNs-Based RGB-D salient object detection (SOD) networks are all required to be pretrained on the ImageNet to learn the hierarchy features which helps  provide a good initialization. However, the collection and annotation of large-scale datasets are time-consuming and expensive. In this paper, we utilize self-supervised representation learning (SSL) to design two pretext tasks: the cross-modal auto-encoder and the depth-contour estimation. Our pretext tasks require only a few and unlabeled RGB-D datasets to perform pretraining, which makes the network capture rich semantic contexts and reduce the gap between two modalities, thereby providing an effective initialization for the downstream task. In addition, for the inherent problem of cross-modal fusion in RGB-D SOD, we propose a consistency-difference aggregation (CDA) module that splits a single feature fusion into multi-path fusion to achieve an adequate perception of consistent and differential information. The CDA module is general and suitable for cross-modal and cross-level feature fusion.  Extensive experiments on six benchmark datasets show that our self-supervised pretrained model performs favorably against most state-of-the-art methods pretrained on ImageNet.  The source code will be publicly available at  https://github.com/Xiaoqi-Zhao-DLUT/SSLSOD.

----

## [385] Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition

**Authors**: *Yan Zhao, Weicong Chen, Xu Tan, Kai Huang, Jihong Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20258](https://doi.org/10.1609/aaai.v36i3.20258)

**Abstract**:

Data in the real world tends to exhibit a long-tailed label distribution, which poses great challenges for the training of neural networks in visual recognition. Existing methods tackle this problem mainly from the perspective of data quantity, i.e., the number of samples in each class. To be specific, they pay more attention to tail classes, like applying larger adjustments to the logit. However, in the training process, the quantity and difficulty of data are two intertwined and equally crucial problems. For some tail classes, the features of their instances are distinct and discriminative, which can also bring satisfactory accuracy; for some head classes, although with sufficient samples, the high semantic similarity with other classes and lack of discriminative features will bring bad accuracy. Based on these observations, we propose Adaptive Logit Adjustment Loss (ALA Loss) to apply an adaptive adjusting term to the logit. The adaptive adjusting term is composed of two complementary factors: 1) quantity factor, which pays more attention to tail classes, and 2) difficulty factor, which adaptively pays more attention to hard instances in the training process. The difficulty factor can alleviate the over-optimization on tail yet easy instances and under-optimization on head yet hard instances. The synergy of the two factors can not only advance the performance on tail classes even further, but also promote the accuracy on head classes. Unlike previous logit adjusting methods that only concerned about data quantity, ALA Loss tackles the long-tailed problem from a more comprehensive, fine-grained and adaptive perspective. Extensive experimental results show that our method achieves the state-of-the-art performance on challenging recognition benchmarks, including ImageNet-LT, iNaturalist 2018, and Places-LT.

----

## [386] CADRE: A Cascade Deep Reinforcement Learning Framework for Vision-Based Autonomous Urban Driving

**Authors**: *Yinuo Zhao, Kun Wu, Zhiyuan Xu, Zhengping Che, Qi Lu, Jian Tang, Chi Harold Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20259](https://doi.org/10.1609/aaai.v36i3.20259)

**Abstract**:

Vision-based autonomous urban driving in dense traffic is quite challenging due to the complicated urban environment and the dynamics of the driving behaviors. Widely-applied methods either heavily rely on hand-crafted rules or learn from limited human experience, which makes them hard to generalize to rare but critical scenarios. In this paper, we present a novel CAscade Deep REinforcement learning framework, CADRE, to achieve model-free vision-based autonomous urban driving. In CADRE, to derive representative latent features from raw observations, we first offline train a Co-attention Perception Module (CoPM) that leverages the co-attention mechanism to learn the inter-relationships between the visual and control information from a pre-collected driving dataset. Cascaded by the frozen CoPM, we then present an efficient distributed proximal policy optimization framework to online learn the driving policy under the guidance of particularly designed reward functions. We perform a comprehensive empirical study with the CARLA NoCrash benchmark as well as specific obstacle avoidance scenarios in autonomous urban driving tasks. The experimental results well justify the effectiveness of CADRE and its superiority over the state-of-the-art by a wide margin.

----

## [387] Learning from the Tangram to Solve Mini Visual Tasks

**Authors**: *Yizhou Zhao, Liang Qiu, Pan Lu, Feng Shi, Tian Han, Song-Chun Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20260](https://doi.org/10.1609/aaai.v36i3.20260)

**Abstract**:

Current pre-training methods in computer vision focus on natural images in the daily-life context. However, abstract diagrams such as icons and symbols are common and important in the real world. We are inspired by Tangram, a game that requires replicating an abstract pattern from seven dissected shapes. By recording human experience in solving tangram puzzles, we present the Tangram dataset and show that a pre-trained neural model on the Tangram helps solve some mini visual tasks based on low-resolution vision. Extensive experiments demonstrate that our proposed method generates intelligent solutions for aesthetic tasks such as folding clothes and evaluating room layouts. The pre-trained feature extractor can facilitate the convergence of few-shot learning tasks on human handwriting and improve the accuracy in identifying icons by their contours. The Tangram dataset is available at https://github.com/yizhouzhao/Tangram.

----

## [388] Handling Slice Permutations Variability in Tensor Recovery

**Authors**: *Jingjing Zheng, Xiaoqin Zhang, Wenzhe Wang, Xianta Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20261](https://doi.org/10.1609/aaai.v36i3.20261)

**Abstract**:

This work studies the influence of slice permutations on tensor recovery, which is derived from a reasonable assumption about algorithm, i.e. changing data order should not affect the effectiveness of the algorithm. However, as we will discussed in this paper, this assumption is not satisfied by tensor recovery under some cases. We call this interesting problem as Slice Permutations Variability (SPV) in tensor recovery. In this paper,  we discuss SPV of several key tensor recovery problems  theoretically and experimentally. The obtained results show that there is a huge gap between results by tensor recovery using tensor with different slices sequences. To overcome   SPV  in tensor recovery, we develop a novel tensor recovery algorithm  by Minimum Hamiltonian Circle for SPV (TRSPV)  which  exploits a low dimensional subspace structures within data tensor  more exactly. To the best of our knowledge, this is the first work to discuss  and  effectively solve the SPV problem in tensor recovery. The experimental results demonstrate the effectiveness of the proposed algorithm in eliminating SPV in tensor recovery.

----

## [389] Boosting Contrastive Learning with Relation Knowledge Distillation

**Authors**: *Kai Zheng, Yuanjiang Wang, Ye Yuan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20262](https://doi.org/10.1609/aaai.v36i3.20262)

**Abstract**:

While self-supervised representation learning (SSL) has proved to be effective in the large model, there is still a huge gap between the SSL and supervised method in the lightweight model when following the same solution. We delve into this problem and find that the lightweight model is prone to collapse in semantic space when simply performing instance-wise contrast. To address this issue, we propose a relation-wise contrastive paradigm with Relation Knowledge Distillation (ReKD). We introduce a heterogeneous teacher to explicitly mine the semantic information and transferring a novel relation knowledge to the student (lightweight model). The theoretical analysis supports our main concern about instance-wise contrast and verify the effectiveness of our relation-wise contrastive learning. Extensive experimental results also demonstrate that our method achieves significant improvements on multiple lightweight models. Particularly, the linear evaluation on AlexNet obviously improves the current state-of-art from 44.7% to 50.1% , which is the first work to get close to the supervised (50.5%). Code will be made available.

----

## [390] Weakly Supervised Video Moment Localization with Contrastive Negative Sample Mining

**Authors**: *Minghang Zheng, Yanjie Huang, Qingchao Chen, Yang Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20263](https://doi.org/10.1609/aaai.v36i3.20263)

**Abstract**:

Video moment localization aims at localizing the video segments which are most related to the given free-form natural language query. The weakly supervised setting, where only video level description is available during training, is getting more and more attention due to its lower annotation cost. Prior weakly supervised methods mainly use sliding windows to generate temporal proposals, which are independent of video content and low quality, and train the model to distinguish matched video-query pairs and unmatched ones collected from different videos, while neglecting what the model needs is to distinguish the unaligned segments within the video. In this work, we propose a novel weakly supervised solution by introducing Contrastive Negative sample Mining (CNM). Specifically, we use a learnable Gaussian mask to generate positive samples, highlighting the video frames most related to the query, and consider other frames of the video and the whole video as easy and hard negative samples respectively. We then train our network with the Intra-Video Contrastive loss to make our positive and negative samples more discriminative. Our method has two advantages: (1) Our proposal generation process with a learnable Gaussian mask is more efficient and makes our positive sample higher quality. (2) The more difficult intra-video negative samples enable our model to distinguish highly confusing scenes. Experiments on two datasets show the effectiveness of our method. Code can be found at https://github.com/minghangz/cnm.

----

## [391] Dual Decoupling Training for Semi-supervised Object Detection with Noise-Bypass Head

**Authors**: *Shida Zheng, Chenshu Chen, Xiaowei Cai, Tingqun Ye, Wenming Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20264](https://doi.org/10.1609/aaai.v36i3.20264)

**Abstract**:

Pseudo bounding boxes from the self-training paradigm are inevitably noisy for semi-supervised object detection. To cope with that, a dual decoupling training framework is proposed in the present study, i.e. clean and noisy data decoupling, and classification and localization task decoupling. In the first decoupling, two-level thresholds are used to categorize pseudo boxes into three groups, i.e. clean backgrounds, noisy foregrounds and clean foregrounds. With a specially designed noise-bypass head focusing on noisy data, backbone networks can extract coarse but diverse information; and meanwhile, an original head learns from clean samples for more precise predictions. In the second decoupling, we take advantage of the two-head structure for better evaluation of localization quality, thus the category label and location of a pseudo box can remain independent of each other during training. The approach of two-level thresholds is also applied to group pseudo boxes into three sections of different location accuracy. We outperform existing works by a large margin on VOC datasets, reaching 54.8 mAP(+1.8), and even up to 55.9 mAP(+1.5) by leveraging MS-COCO train2017 as extra unlabeled data. On MS-COCO benchmark, our method also achieves about 1.0 mAP improvements averaging across protocols compared with the prior state-of-the-art.

----

## [392] SCALoss: Side and Corner Aligned Loss for Bounding Box Regression

**Authors**: *Tu Zheng, Shuai Zhao, Yang Liu, Zili Liu, Deng Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20265](https://doi.org/10.1609/aaai.v36i3.20265)

**Abstract**:

Bounding box regression is an important component in object detection. Recent work achieves promising performance by optimizing the Intersection over Union (IoU). However, IoU-based loss has the gradient vanish problem in the case of low overlapping bounding boxes, and the model could easily ignore these simple cases. In this paper, we propose Side Overlap (SO) loss by maximizing the side overlap of two bounding boxes, which puts more penalty for low overlapping bounding box cases. Besides, to speed up the convergence, the Corner Distance (CD) is added into the objective function. Combining the Side Overlap and Corner Distance, we get a new regression objective function, Side and Corner Align Loss (SCALoss). The SCALoss is well-correlated with IoU loss, which also benefits the evaluation metric but produces more penalty for low-overlapping cases. It can serve as a comprehensive similarity measure, leading to better localization performance and faster convergence speed. Experiments on COCO, PASCAL VOC, and LVIS benchmarks show that SCALoss can bring consistent improvement and outperform ln loss and IoU based loss with popular object detectors such as YOLOV3, SSD, Faster-RCNN. Code is available at: https://github.com/Turoad/SCALoss.

----

## [393] SepFusion: Finding Optimal Fusion Structures for Visual Sound Separation

**Authors**: *Dongzhan Zhou, Xinchi Zhou, Di Hu, Hang Zhou, Lei Bai, Ziwei Liu, Wanli Ouyang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20266](https://doi.org/10.1609/aaai.v36i3.20266)

**Abstract**:

Multiple modalities can provide rich semantic information; and exploiting such information will normally lead to better performance compared with the single-modality counterpart.
 However, it is not easy to devise an effective cross-modal fusion structure due to the variations of feature dimensions and semantics, especially when the inputs even come from different sensors, as in the field of audio-visual learning. In this work, we propose SepFusion, a novel framework that can smoothly produce optimal fusion structures for visual-sound separation. The framework is composed of two components, namely the model generator and the evaluator. To construct the generator, we devise a lightweight architecture space that can adapt to different input modalities. In this way, we can easily obtain audio-visual fusion structures according to our demands. For the evaluator, we adopt the idea of neural architecture search to select superior networks effectively. This automatic process can significantly save human efforts while achieving competitive performances. Moreover, since our SepFusion provides a series of strong models, we can utilize the model family for broader applications, such as further promoting performance via model assembly, or providing suitable architectures for the separation of certain instrument classes. These potential applications further enhance the competitiveness of our approach.

----

## [394] Pan-Sharpening with Customized Transformer and Invertible Neural Network

**Authors**: *Man Zhou, Jie Huang, Yanchi Fang, Xueyang Fu, Aiping Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20267](https://doi.org/10.1609/aaai.v36i3.20267)

**Abstract**:

In remote sensing imaging systems, pan-sharpening is an important technique to obtain high-resolution multispectral images from a high-resolution panchromatic image and its corresponding low-resolution multispectral image. Owing to the powerful learning capability of convolution neural network (CNN), CNN-based methods have dominated this field. However, due to the limitation of the convolution operator, long-range spatial features are often not accurately obtained, thus limiting the overall performance. To this end, we propose a novel and effective method by exploiting a customized transformer architecture and information-lossless invertible neural module for long-range dependencies modeling and effective feature fusion in this paper. Specifically, the customized transformer formulates the PAN and MS features as queries and keys to encourage joint feature learning across two modalities while the designed invertible neural module enables effective feature fusion to generate the expected pan-sharpened results. To the best of our knowledge, this is the first attempt to introduce transformer and invertible neural network into pan-sharpening field. Extensive experiments over different kinds of satellite datasets demonstrate that our method outperforms state-of-the-art algorithms both visually and quantitatively with fewer parameters and flops. Further, the ablation experiments also prove the effectiveness of the proposed customized long-range transformer and effective invertible neural feature fusion module for pan-sharpening.

----

## [395] Promoting Single-Modal Optical Flow Network for Diverse Cross-Modal Flow Estimation

**Authors**: *Shili Zhou, Weimin Tan, Bo Yan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20268](https://doi.org/10.1609/aaai.v36i3.20268)

**Abstract**:

In recent years, optical flow methods develop rapidly, achieving unprecedented high performance. Most of the methods only consider single-modal optical flow under the well-known brightness-constancy assumption. However, in many application systems, images of different modalities need to be aligned, which demands to estimate cross-modal flow between the cross-modal image pairs. A lot of cross-modal matching methods are designed for some specific cross-modal scenarios. We argue that the prior knowledge of the advanced optical flow models can be transferred to the cross-modal flow estimation, which may be a simple but unified solution for diverse cross-modal matching tasks. To verify our hypothesis, we design a self-supervised framework to promote the single-modal optical flow networks for diverse corss-modal flow estimation. Moreover, we add a Cross-Modal-Adapter block as a plugin to  the state-of-the-art optical flow model RAFT for better performance in cross-modal scenarios. Our proposed Modality Promotion Framework and Cross-Modal Adapter have multiple advantages compared to the existing methods. The experiments demonstrate that our method is effective on multiple datasets of different cross-modal scenarios.

----

## [396] Edge-Aware Guidance Fusion Network for RGB-Thermal Scene Parsing

**Authors**: *Wujie Zhou, Shaohua Dong, Caie Xu, Yaguan Qian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20269](https://doi.org/10.1609/aaai.v36i3.20269)

**Abstract**:

RGB–thermal scene parsing has recently attracted increasing research interest in the field of computer vision. However, most existing methods fail to perform good boundary extraction for prediction maps and cannot fully use high-level features. In addition, these methods simply fuse the features from RGB and thermal modalities but are unable to obtain comprehensive fused features. To address these problems, we propose an edge-aware guidance fusion network (EGFNet) for RGB–thermal scene parsing. First, we introduce a prior edge map generated using the RGB and thermal images to capture detailed information in the prediction map and then embed the prior edge information in the feature maps. To effectively fuse the RGB and thermal information, we propose a multimodal fusion module that guarantees adequate cross-modal fusion. Considering the importance of high-level semantic information, we propose a global information module and a semantic information module to extract rich semantic information from the high-level features. For decoding, we use simple elementwise addition for cascaded feature fusion. Finally, to improve the parsing accuracy, we apply multitask deep supervision to the semantic and boundary maps. Extensive experiments were performed on benchmark datasets to demonstrate the effectiveness of the proposed EGFNet and its superior performance compared with state-of-the-art methods. The code and results can be found at https://github.com/ShaohuaDong2021/EGFNet.

----

## [397] TiGAN: Text-Based Interactive Image Generation and Manipulation

**Authors**: *Yufan Zhou, Ruiyi Zhang, Jiuxiang Gu, Chris Tensmeyer, Tong Yu, Changyou Chen, Jinhui Xu, Tong Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20270](https://doi.org/10.1609/aaai.v36i3.20270)

**Abstract**:

Using natural-language feedback to guide image generation and manipulation can greatly lower the required efforts and skills. This topic has received increased attention in recent years through refinement of Generative Adversarial Networks (GANs); however, most existing works are limited to single-round interaction, which is not reflective of real world interactive image editing workflows. Furthermore, previous works dealing with multi-round scenarios are limited to predefined feedback sequences, which is also impractical. In this paper, we propose a novel framework for Text-based Interactive image generation and manipulation (TiGAN) that responds to users' natural-language feedback. 
 TiGAN utilizes the powerful pre-trained CLIP model to understand users' natural-language feedback and exploits contrastive learning for a better text-to-image mapping. To maintain the image consistency during interactions, TiGAN generates intermediate feature vectors aligned with the feedback and selectively feeds these vectors to our proposed generative model. Empirical results on several datasets show that TiGAN improves both interaction efficiency and image quality while better avoids undesirable image manipulation during interactions.

----

## [398] Cross-Domain Empirical Risk Minimization for Unbiased Long-Tailed Classification

**Authors**: *Beier Zhu, Yulei Niu, Xian-Sheng Hua, Hanwang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20271](https://doi.org/10.1609/aaai.v36i3.20271)

**Abstract**:

We address the overlooked unbiasedness in existing long-tailed classification methods: we find that their overall improvement is mostly attributed to the biased preference of "tail" over "head", as the test distribution is assumed to be balanced; however, when the test is as imbalanced as the long-tailed training data---let the test respect Zipf's law of nature---the "tail" bias is no longer beneficial overall because it hurts the "head" majorities. In this paper, we propose Cross-Domain Empirical Risk Minimization (xERM) for training an unbiased test-agnostic model to achieve strong performances on both test distributions, which empirically demonstrates that xERM fundamentally improves the classification by learning better feature representation rather than the "head vs. tail" game. Based on causality, we further theoretically explain why xERM achieves unbiasedness: the bias caused by the domain selection is removed by adjusting the empirical risks on the imbalanced domain and the balanced but unseen domain.

----

## [399] Deep Recurrent Neural Network with Multi-Scale Bi-directional Propagation for Video Deblurring

**Authors**: *Chao Zhu, Hang Dong, Jinshan Pan, Boyang Liang, Yuhao Huang, Lean Fu, Fei Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i3.20272](https://doi.org/10.1609/aaai.v36i3.20272)

**Abstract**:

The success of the state-of-the-art video deblurring methods stems mainly from implicit or explicit estimation of alignment among the adjacent frames for latent video restoration. However, due to the influence of the blur effect, estimating the alignment information from the blurry adjacent frames is not a trivial task. Inaccurate estimations will interfere the following frame restoration. Instead of estimating alignment information, we propose a simple and effective deep Recurrent Neural Network with Multi-scale Bi-directional Propagation (RNN-MBP) to effectively propagate and gather the information from unaligned neighboring frames for better video deblurring. Specifically, we build a Multi-scale Bi-directional Propagation (MBP) module with two U-Net RNN cells which can directly exploit the inter-frame information from unaligned neighboring hidden states by integrating them in different scales. Moreover, to better evaluate the proposed algorithm and existing state-of-the-art methods on real-world blurry scenes, we also create a Real-World Blurry Video Dataset (RBVD) by a well-designed Digital Video Acquisition System (DVAS) and use it as the training and evaluation dataset. Extensive experimental results demonstrate that the proposed RBVD dataset effectively improve the performance of existing algorithms on real-world blurry videos, and the proposed algorithm performs favorably against the state-of-the-art methods on three typical benchmarks. The code is available at https://github.com/XJTU-CVLAB-LOWLEVEL/RNN-MBP.

----



[Go to the previous page](AAAI-2022-list01.md)

[Go to the next page](AAAI-2022-list03.md)

[Go to the catalog section](README.md)