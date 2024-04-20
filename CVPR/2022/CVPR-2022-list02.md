## [200] Noise Distribution Adaptive Self-Supervised Image Denoising using Tweedie Distribution and Score Matching

**Authors**: *Kwanyoung Kim, Taesung Kwon, Jong Chul Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00205](https://doi.org/10.1109/CVPR52688.2022.00205)

**Abstract**:

Tweedie distributions are a special case of exponential dispersion models, which are often used in classical statistics as distributions for generalized linear models. Here, we show that Tweedie distributions also play key roles in modern deep learning era, leading to a distribution adaptive self-supervised image denoising formula without clean reference images. Specifically, by combining with the recent Noise2Score self-supervised image denoising approach and the saddle point approximation of Tweedie distribution, we provide a general closed-form denoising formula that can be used for large classes of noise distributions without ever knowing the underlying noise distribution. Similar to the original Noise2Score, the new approach is composed of two successive steps: score matching using perturbed noisy images, followed by a closed form image denoising formula via distribution-independent Tweedie's formula. In addition, we reveal a systematic algorithm to estimate the noise model and noise parameters for a given noisy image data set. Through extensive experiments, we demonstrate that the proposed method can accurately estimate noise models and parameters, and provide the state-of-the-art self-supervised image denoising performance in the benchmark dataset and real-world dataset.

----

## [201] Unpaired Deep Image Deraining Using Dual Contrastive Learning

**Authors**: *Xiang Chen, Jinshan Pan, Kui Jiang, Yufeng Li, Yufeng Huang, Caihua Kong, Longgang Dai, Zhentao Fan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00206](https://doi.org/10.1109/CVPR52688.2022.00206)

**Abstract**:

Learning single image deraining (SID) networks from an unpaired set of clean and rainy images is practical and valuable as acquiring paired real-world data is almost infeasible. However, without the paired data as the supervision, learning a SID network is challenging. Moreover, simply using existing unpaired learning methods (e.g., unpaired adversarial learning and cycle-consistency constraints) in the SID task is insufficient to learn the underlying relationship from rainy inputs to clean outputs as there exists significant domain gap between the rainy and clean images. In this paper, we develop an effective unpaired SID adversarial framework which explores mutual properties of the unpaired exemplars by a dual contrastive learning manner in a deep feature space, named as DCD-GAN. The proposed method mainly consists of two cooperative branches: Bidirectional Translation Branch (BTB) and Contrastive Guidance Branch (CGB). Specifically, BTB exploits full advantage of the circulatory architecture of adversarial consistency to generate abundant exemplar pairs and excavates latent feature distributions between two domains by equipping it with bidirectional mapping. Simultaneously, CGB implicitly constrains the embeddings of different exemplars in the deep feature space by encouraging the similar feature distributions closer while pushing the dissimilar further away, in order to better facilitate rain removal and help image restoration. Extensive experiments demonstrate that our method performs favorably against existing unpaired deraining approaches on both synthetic and real-world datasets, and generates comparable results against several fully-supervised or semi-supervised models.

----

## [202] Blind2Unblind: Self-Supervised Image Denoising with Visible Blind Spots

**Authors**: *Zejin Wang, Jiazheng Liu, Guoqing Li, Hua Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00207](https://doi.org/10.1109/CVPR52688.2022.00207)

**Abstract**:

Real noisy-clean pairs on a large scale are costly and difficult to obtain. Meanwhile, supervised denoisers trained on synthetic data perform poorly in practice. Self-supervised denoisers, which learn only from single noisy images, solve the data collection problem. However, self-supervised denoising methods, especially blindspot-driven ones, suffer sizable information loss during input or network design. The absence of valuable information dramatically reduces the upper bound of denoising performance. In this paper, we propose a simple yet efficient approach called Blind2Unblind to overcome the information loss in blindspot-driven denoising methods. First, we introduce a global-aware mask mapper that enables global perception and accelerates training. The mask mapper samples all pixels at blind spots on denoised volumes and maps them to the same channel, allowing the loss function to optimize all blind spots at once. Second, we propose a revisible loss to train the denoising network and make blind spots visible. The denoiser can learn directly from raw noise images without losing information or being trapped in identity mapping. We also theoretically analyze the convergence of the revisible loss. Extensive experiments on synthetic and real-world datasets demonstrate the superior performance of our approach compared to previous work. Code is available at https://github.com/demonsjin/Blind2Unblind.

----

## [203] Self-augmented Unpaired Image Dehazing via Density and Depth Decomposition

**Authors**: *Yang Yang, Chaoyue Wang, Risheng Liu, Lin Zhang, Xiaojie Guo, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00208](https://doi.org/10.1109/CVPR52688.2022.00208)

**Abstract**:

To overcome the overfitting issue of dehazing models trained on synthetic hazy-clean image pairs, many recent methods attempted to improve models' generalization ability by training on unpaired data. Most of them simply formulate dehazing and rehazing cycles, yet ignore the physical properties of the real-world hazy environment, i.e. the haze varies with density and depth. In this paper, we propose a self-augmented image dehazing framework, termed D4 (Dehazing via Decomposing transmission map into Density and Depth) for haze generation and removal. Instead of merely estimating transmission maps or clean content, the proposed framework focuses on exploring scattering coefficient and depth information contained in hazy and clean images. With estimated scene depth, our method is capable of re-rendering hazy images with different thick-nesses which further benefits the training of the dehazing network. It is worth noting that the whole training process needs only unpaired hazy and clean images, yet succeeded in recovering the scattering coefficient, depth map and clean content from a single hazy image. Comprehensive experiments demonstrate our method outperforms state-of-the-art unpaired dehazing methods with much fewer parameters and FLOPs. Our code is available at https://github.com/YaN9-Y/D4.

----

## [204] VideoINR: Learning Video Implicit Neural Representation for Continuous Space-Time Super-Resolution

**Authors**: *Zeyuan Chen, Yinbo Chen, Jingwen Liu, Xingqian Xu, Vidit Goel, Zhangyang Wang, Humphrey Shi, Xiaolong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00209](https://doi.org/10.1109/CVPR52688.2022.00209)

**Abstract**:

Videos typically record the streaming and continuous visual data as discrete consecutive frames. Since the storage cost is expensive for videos of high fidelity, most of them are stored in a relatively low resolution and frame rate. Recent works of Space-Time Video Super-Resolution (STVSR) are developed to incorporate temporal interpolation and spatial super-resolution in a unified framework. However, most of them only support a fixed up-sampling scale, which limits their flexibility and applications. In this work, instead of following the discrete representations, we propose Video Implicit Neural Representation (VideoINR), and we show its applications for STVSR. The learned implicit neural representation can be decoded to videos of arbitrary spatial resolution and frame rate. We show that VideoINR achieves competitive performances with state-of-the-art STVSR methods on common up-sampling scales and significantly outperforms prior works on continuous and out-of-training-distribution scales. Our project page is at here and code is available at https://github.com/Picsart-AI-Research/VideoINR-Continuous-Space-Time-Super-Resolution.

----

## [205] Fast Algorithm for Low-rank Tensor Completion in Delay-embedded Space

**Authors**: *Ryuki Yamamoto, Hidekata Hontani, Akira Imakura, Tatsuya Yokota*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00210](https://doi.org/10.1109/CVPR52688.2022.00210)

**Abstract**:

Tensor completion using multiway delay-embedding transform (MDT) (or Hankelization) suffers from the large memory requirement and high computational cost in spite of its high potentiality for the image modeling. Recent studies have shown high completion performance with a relatively small window size, but experiments with large window sizes require huge amount of memory and cannot be easily calculated. In this study, we address this serious computational issue, and propose its fast and efficient algorithm. Key techniques of the proposed method are based on two properties: (1) the signal after MDT can be diagonalized by Fourier transform, (2) an inverse MDT can be represented as a convolutional form. To use the properties, we modify MDT-Tucker [26], a method using Tucker decomposition with MDT, and introducing the fast and efficient algorithm. Our experiments show more than 100 times acceleration while maintaining high accuracy, and to realize the computation with large window size.

----

## [206] Exploring and Evaluating Image Restoration Potential in Dynamic Scenes

**Authors**: *Cheng Zhang, Shaolin Su, Yu Zhu, Qingsen Yan, Jinqiu Sun, Yanning Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00211](https://doi.org/10.1109/CVPR52688.2022.00211)

**Abstract**:

In dynamic scenes, images often suffer from dynamic blur due to superposition of motions or low signal-noise ratio resulted from quick shutter speed when avoiding motions. Recovering sharp and clean results from the captured images heavily depends on the ability of restoration methods and the quality of the input. Although existing research on image restoration focuses on developing models for obtaining better restored results, fewer have studied to evaluate how and which input image leads to superior restored quality. In this paper, to better study an image's potential value that can be explored for restoration, we propose a novel concept, referring to image restoration potential (IRP). Specifically, We first establish a dynamic scene imaging dataset containing composite distortions and applied image restoration processes to validate the rationality of the existence to IRP. Based on this dataset, we investigate several properties of IRP and propose a novel deep model to accurately predict IRP values. By gradually distilling and selective fusing the degradation features, the proposed model shows its superiority in IRP prediction. Thanks to the proposed model, we are then able to validate how various image restoration related applications are benefited from IRP prediction. We show the potential usages of IRP as a filtering principle to select valuable frames, an auxiliary guidance to improve restoration models, and also an indicator to optimize camera settings for capturing better images under dynamic scenarios.

----

## [207] GIQE: Generic Image Quality Enhancement via Nth Order Iterative Degradation

**Authors**: *Pranjay Shyam, Kyung-Soo Kim, Kuk-Jin Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00212](https://doi.org/10.1109/CVPR52688.2022.00212)

**Abstract**:

Visual degradations caused by motion blur, raindrop, rain, snow, illumination, and fog deteriorate image quality and, subsequently, the performance of perception algorithms deployed in outdoor conditions. While degradation-specific image restoration techniques have been extensively studied, such algorithms are domain sensitive and fail in real scenarios where multiple degradations exist simultaneously. This makes a case for blind image restoration and reconstruction algorithms as practically relevant. However, the absence of a dataset diverse enough to encapsulate all variations hinders development for such an algorithm. In this paper, we utilize a synthetic degradation model that recursively applies sets of random degradations to generate naturalistic degradation images of varying complexity, which are used as input. Furthermore, as the degradation intensity can vary across an image, the spatially invariant convolutional filter cannot be applied for all degradations. Hence to enable spatial variance during image restoration and reconstruction, we design a transformer-based architecture to benefit from the long-range dependencies. In addition, to reduce the computational cost of transformers, we propose a multi-branch structure coupled with modifications such as a complimentary feature selection mechanism and the replacement of a feed-forward network with lightweight multiscale convolutions. Finally, to improve restoration and reconstruction, we integrate an auxiliary decoder branch to predict the degradation mask to ensure the underlying network can localize the degradation information. From empirical analysis on 10 datasets covering rain drop removal, deraining, dehazing, image enhancement, and deblurring, we demonstrate the efficacy of the proposed approach while obtaining SoTA performance.

----

## [208] Does text attract attention on e-commerce images: A novel saliency prediction dataset and method

**Authors**: *Lai Jiang, Yifei Li, Shengxi Li, Mai Xu, Se Lei, Yichen Guo, Bo Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00213](https://doi.org/10.1109/CVPR52688.2022.00213)

**Abstract**:

E-commerce images are playing a central role in attracting people's attention when retailing and shopping online, and an accurate attention prediction is of significant importance for both customers and retailers, where its research is yet to start. In this paper, we establish the first dataset of saliency e-commerce images (SalECI), which allows for learning to predict saliency on the e-commerce images. We then provide specialized and thorough analysis by high-lighting the distinct features of e-commerce images, e.g., non-locality and correlation to text regions. Correspondingly, taking advantages of the non-local and self-attention mechanisms, we propose a salient SWin-Transformer back-bone, followed by a multi-task learning with saliency and text detection heads, where an information flow mechanism is proposed to further benefit both tasks. Experimental results have verified the state-of-the-art performances of our work in the e-commerce scenario.

----

## [209] IDR: Self-Supervised Image Denoising via Iterative Data Refinement

**Authors**: *Yi Zhang, Dasong Li, Ka Lung Law, Xiaogang Wang, Hongwei Qin, Hongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00214](https://doi.org/10.1109/CVPR52688.2022.00214)

**Abstract**:

The lack of large-scale noisy-clean image pairs restricts supervised denoising methods' deployment in actual applications. While existing unsupervised methods are able to learn image denoising without ground-truth clean images, they either show poor performance or work under impractical settings (e.g., paired noisy images). In this paper, we present a practical unsupervised image denoising method to achieve state-of-the-art denoising performance. Our method only requires single noisy images and a noise model, which is easily accessible in practical raw image denoising. It performs two steps iteratively: (1) Constructing a noisier-noisy dataset with random noise from the noise model; (2) training a model on the noisier-noisy dataset and using the trained model to refine noisy images to obtain the targets used in the next round. We further approximate our full iterative method with a fast algorithm for more efficient training while keeping its original high performance. Experiments on real-world, synthetic, and correlated noise show that our proposed unsupervised denoising approach has superior performances over existing unsupervised methods and competitive performance with supervised methods. In addition, we argue that existing denoising datasets are of low quality and contain only a small number of scenes. To evaluate raw image denoising performance in real-world applications, we build a high-quality raw image dataset SenseNoise-500 that contains 500 real-life scenes. The dataset can serve as a strong benchmark for better evaluating raw image denoising. Code and dataset will be released
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/zhangyi-3/IDR

----

## [210] ABPN: Adaptive Blend Pyramid Network for Real-Time Local Retouching of Ultra High-Resolution Photo

**Authors**: *Biwen Lei, Xiefan Guo, Hongyu Yang, Miaomiao Cui, Xuansong Xie, Di Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00215](https://doi.org/10.1109/CVPR52688.2022.00215)

**Abstract**:

Photo retouching finds many applications in various fields. However, most existing methods are designed for global retouching and seldom pay attention to the local region, while the latter is actually much more tedious and time-consuming in photography pipelines. In this paper, we propose a novel adaptive blend pyramid network, which aims to achieve fast local retouching on ultra high-resolution photos. The network is mainly composed of two components: a context-aware local retouching layer (LRL) and an adaptive blend pyramid layer (BPL). The LRL is designed to implement local retouching on low-resolution images, giving full consideration of the global context and local texture information, and the BPL is then developed to progressively expand the low-resolution results to the higher ones, with the help of the proposed adaptive blend module and refining module. Our method outperforms the existing methods by a large margin on two local photo retouching tasks and exhibits excellent performance in terms of running speed, achieving real-time inference on 4K images with a single NVIDIA Tesla P100 GPU. Moreover, we introduce the first high-definition cloth retouching dataset CRHD-3K to promote the research on local photo retouching. The dataset is available at https://github.com/youngLbw/crhd-3K.

----

## [211] Texture-based Error Analysis for Image Super-Resolution

**Authors**: *Salma Abdel Magid, Zudi Lin, Donglai Wei, Yulun Zhang, Jinjin Gu, Hanspeter Pfister*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00216](https://doi.org/10.1109/CVPR52688.2022.00216)

**Abstract**:

Evaluation practices for image super-resolution (SR) use a single-value metric, the PSNR or SSIM, to determine model performance. This provides little insight into the source of errors and model behavior. Therefore, it is beneficial to move beyond the conventional approach and reconceptualize evaluation with interpretability as our main priority. We focus on a thorough error analysis from a variety of perspectives. Our key contribution is to leverage a texture classifier, which enables us to assign patches with semantic labels, to identify the source of SR errors both globally and locally. We then use this to determine (a) the semantic alignment of SR datasets, (b) how SR models perform on each label, (c) to what extent high-resolution (HR) and SR patches semantically correspond, and more. Through these different angles, we are able to highlight potential pitfalls and blindspots. Our overall investigation highlights numerous unexpected insights. We hope this work serves as an initial step for debugging blackbox SR networks.

----

## [212] Blind Image Super-resolution with Elaborate Degradation Modeling on Noise and Kernel

**Authors**: *Zongsheng Yue, Qian Zhao, Jianwen Xie, Lei Zhang, Deyu Meng, Kwan-Yee K. Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00217](https://doi.org/10.1109/CVPR52688.2022.00217)

**Abstract**:

While researches on model-based blind single image super-resolution (SISR) have achieved tremendous successes recently, most of them do not consider the image degradation sufficiently. Firstly, they always assume image noise obeys an independent and identically distributed (i.i.d.) Gaussian or Laplacian distribution, which largely underestimates the complexity of real noise. Secondly, previous commonly-used kernel priors (e.g., normalization, sparsity) are not effective enough to guarantee a rational kernel solution, and thus degenerates the performance of subsequent SISR task. To address the above issues, this paper proposes a model-based blind SISR method under the probabilistic framework, which elaborately models image degradation from the perspectives of noise and blur kernel. Specifically, instead of the traditional i.i.d. noise assumption, a patch-based non-i.i.d. noise model is proposed to tackle the complicated real noise, expecting to increase the degrees of freedom of the model for noise representation. As for the blur kernel, we novelly construct a concise yet effective kernel generator, and plug it into the proposed blind SISR method as an explicit kernel prior (EKP). To solve the proposed model, a theoretically grounded Monte Carlo EM algorithm is specifically designed. Comprehensive experiments demonstrate the superiority of our method over current state-of-the-arts on synthetic and real datasets. The source code is available at https://github.com/zsyOAOA/BSRDM.

----

## [213] KNN Local Attention for Image Restoration

**Authors**: *Hunsang Lee, Hyesong Choi, Kwanghoon Sohn, Dongbo Min*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00218](https://doi.org/10.1109/CVPR52688.2022.00218)

**Abstract**:

Recent works attempt to integrate the non-local operation with CNNs or Transformer, achieving remarkable performance in image restoration tasks. The global similarity, however, has the problems of the lack of locality and the high computational complexity that is quadratic to an input resolution. The local attention mechanism alleviates these issues by introducing the inductive bias of the locality with convolution-like operators. However, by focusing only on adjacent positions, the local attention suffers from an insufficient receptive field for image restoration. In this paper, we propose a new attention mechanism for image restoration, called k-NN Image Transformer (KiT), that rectifies the above mentioned limitations. Specifically, the KiT groups k-nearest neighbor patches with locality sensitive hashing (LSH), and the grouped patches are aggregated into each query patch by performing a pair-wise local attention. In this way, the pair-wise operation establishes nonlocal connectivity while maintaining the desired properties of the local attention, i.e., inductive bias of locality and linear complexity to input resolution. The proposed method outperforms state-of-the-art restoration approaches on image denoising, deblurring and deraining benchmarks. The code will be available soon.

----

## [214] Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection

**Authors**: *Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00219](https://doi.org/10.1109/CVPR52688.2022.00219)

**Abstract**:

Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, i.e., highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first blackbox joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the Internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.

----

## [215] Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection

**Authors**: *Youwei Pang, Xiaoqi Zhao, Tian-Zhu Xiang, Lihe Zhang, Huchuan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00220](https://doi.org/10.1109/CVPR52688.2022.00220)

**Abstract**:

The recently proposed camouflaged object detection (COD) attempts to segment objects that are visually blended into their surroundings, which is extremely complex and difficult in real-world scenarios. Apart from high intrinsic similarity between the camouflaged objects and their background, the objects are usually diverse in scale, fuzzy in appearance, and even severely occluded. To deal with these problems, we propose a mixed-scale triplet network, Zoom- Net, which mimics the behavior of humans when observing vague images, i.e., zooming in and out. Specifically, our ZoomNet employs the zoom strategy to learn the discriminative mixed-scale semantics by the designed scale integration unit and hierarchical mixed-scale unit, which fully explores imperceptible clues between the candidate objects and background surroundings. Moreover, considering the uncertainty and ambiguity derived from indistinguishable textures, we construct a simple yet effective regularization constraint, uncertainty-aware loss, to promote the model to accurately produce predictions with higher confidence in candidate regions. Without bells and whistles, our proposed highly task-friendly model consistently surpasses the existing 23 state-of-the-art methods on four public datasets. Besides, the superior performance over the recent cutting-edge models on the SOD task also verifies the effectiveness and generality of our model. The code will be available at https://github.com/lartpang/ZoomNet.

----

## [216] Self-Supervised Keypoint Discovery in Behavioral Videos

**Authors**: *Jennifer J. Sun, Serim Ryou, Roni H. Goldshmid, Brandon Weissbourd, John O. Dabiri, David J. Anderson, Ann Kennedy, Yisong Yue, Pietro Perona*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00221](https://doi.org/10.1109/CVPR52688.2022.00221)

**Abstract**:

We propose a method for learning the posture and structure of agents from unlabelled behavioral videos. Starting from the observation that behaving agents are generally the main sources of movement in behavioral videos, our method, Behavioral Keypoint Discovery (B-KinD), uses an encoder-decoder architecture with a geometric bottleneck to reconstruct the spatiotemporal difference between video frames. By focusing only on regions of movement, our approach works directly on input videos without requiring manual annotations. Experiments on a variety of agent types (mouse, fly, human, jellyfish, and trees) demonstrate the generality of our approach and reveal that our discovered keypoints represent semantically meaningful body parts, which achieve state-of-the-art performance on key-point regression among self-supervised methods. Additionally, B-KinD achieve comparable performance to supervised keypoints on downstream tasks, such as behavior classification, suggesting that our method can dramatically reduce model training costs vis-a-vis supervised methods.

----

## [217] Learning to Align Sequential Actions in the Wild

**Authors**: *Weizhe Liu, Bugra Tekin, Huseyin Coskun, Vibhav Vineet, Pascal Fua, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00222](https://doi.org/10.1109/CVPR52688.2022.00222)

**Abstract**:

State-of-the-art methods for self-supervised sequential action alignment rely on deep networks that find correspondences across videos in time. They either learn frame-to-frame mapping across sequences, which does not leverage temporal information, or assume monotonic alignment between each video pair, which ignores variations in the order of actions. As such, these methods are not able to deal with common real-world scenarios that involve background frames or videos that contain non-monotonic sequence of actions. In this paper, we propose an approach to align sequential actions in the wild that involve diverse temporal variations. To this end, we propose an approach to enforce temporal priors on the optimal transport matrix, which leverages temporal consistency, while allowing for variations in the order of actions. Our model accounts for both monotonic and non-monotonic sequences and handles background frames that should not be aligned. We demonstrate that our approach consistently outperforms the state-of-the-art in self-supervised sequential action representation learning on four different benchmark datasets. Code is publicly available at https://github.com/weizheliu/VAVA.

----

## [218] Dynamic 3D Gaze from Afar: Deep Gaze Estimation from Temporal Eye-Head-Body Coordination

**Authors**: *Soma Nonaka, Shohei Nobuhara, Ko Nishino*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00223](https://doi.org/10.1109/CVPR52688.2022.00223)

**Abstract**:

We introduce a novel method and dataset for 3D gaze estimation of a freely moving person from a distance, typically in surveillance views. Eyes cannot be clearly seen in such cases due to occlusion and lacking resolution. Existing gaze estimation methods suffer or fall back to approximating gaze with head pose as they primarily rely on clear, close-up views of the eyes. Our key idea is to instead leverage the intrinsic gaze, head, and body coordination of people. Our method formulates gaze estimation as Bayesian prediction given temporal estimates of head and body orientations which can be reliably estimated from a far. We model the head and body orientation likelihoods and the conditional prior of gaze direction on those with separate neural networks which are then cascaded to output the 3D gaze direction. We introduce an extensive new dataset that consists of surveillance videos annotated with 3D gaze directions captured in 5 indoor and outdoor scenes. Experimental results on this and other datasets validate the accuracy of our method and demonstrate that gaze can be accurately estimated from a typical surveillance distance even when the person's face is not visible to the camera.

----

## [219] End-to-End Human-Gaze-Target Detection with Transformers

**Authors**: *Danyang Tu, Xiongkuo Min, Huiyu Duan, Guodong Guo, Guangtao Zhai, Wei Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00224](https://doi.org/10.1109/CVPR52688.2022.00224)

**Abstract**:

In this paper, we propose an effective and efficient method for Human-Gaze-Target (HGT) detection, i.e., gaze following. Current approaches decouple the HGT detection task into separate branches of salient object detection and human gaze prediction, employing a two-stage framework where human head locations must first be detected and then be fed into the next gaze target prediction sub-network. In contrast, we redefine the HGT detection task as detecting human head locations and their gaze targets, simultaneously. By this way, our method, named Human-Gaze-Target detection TRansformer or HGTTR, streamlines the HGT detection pipeline by eliminating all other additional components. HGTTR reasons about the relations of salient objects and human gaze from the global image context. Moreover, unlike existing two-stage methods that require human head locations as input and can predict only one human's gaze target at a time, HGTTR can directly predict the locations of all people and their gaze targets at one time in an end-to-end manner. The effectiveness and robustness of our proposed method are verified with extensive experiments on the two standard benchmark datasets, GazeFollowing and VideoAttentionTarget. Without bells and whistles, HGTTR outperforms existing state-of-the-art methods by large margins (6.4 mAP gain on GazeFollowing and 10.3 mAP gain on VideoAttentionTarget) with a much simpler architecture.

----

## [220] Automatic Synthesis of Diverse Weak Supervision Sources for Behavior Analysis

**Authors**: *Albert Tseng, Jennifer J. Sun, Yisong Yue*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00225](https://doi.org/10.1109/CVPR52688.2022.00225)

**Abstract**:

Obtaining annotations for large training sets is expensive, especially in settings where domain knowledge is required, such as behavior analysis. Weak supervision has been studied to reduce annotation costs by using weak labels from task-specific labeling functions (LFs) to augment ground truth labels. However, domain experts still need to hand-craft different LFs for different tasks, limiting scalability. To reduce expert effort, we present AutoSWAP: a framework for automatically synthesizing data-efficient task-level LFs. The key to our approach is to efficiently represent expert knowledge in a reusable domain-specific language and more general domain-level LFs, with which we use state-of-the-art program synthesis techniques and a small labeled dataset to generate task-level LFs. Additionally, we propose a novel structural diversity cost that allows for efficient synthesis of diverse sets of LFs, further improving AutoSWAP's performance. We evaluate AutoSWAP in three behavior analysis domains and demonstrate that AutoSWAP outperforms existing approaches using only afraction of the data. Our results suggest that AutoSWAP is an effective way to automatically generate LFs that can significantly reduce expert effort for behavior analysis.

----

## [221] MUSE-VAE: Multi-Scale VAE for Environment-Aware Long Term Trajectory Prediction

**Authors**: *Mihee Lee, Samuel S. Sohn, Seonghyeon Moon, Sejong Yoon, Mubbasir Kapadia, Vladimir Pavlovic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00226](https://doi.org/10.1109/CVPR52688.2022.00226)

**Abstract**:

Accurate long-term trajectory prediction in complex scenes, where multiple agents (e.g., pedestrians or vehicles) interact with each other and the environment while attempting to accomplish diverse and often unknown goals, is a challenging stochastic forecasting problem. In this work, we propose MUSEVAE, a new probabilistic modeling framework based on a cascade of Conditional VAEs, which tackles the long-term, uncertain trajectory prediction task using a coarse-to-fine multi-factor forecasting architecture. In its Macro stage, the model learns a joint pixel-space representation of two key factors, the underlying environment and the agent movements, to predict the long and short term motion goals. Conditioned on them, the Micro stage learns a fine-grained spatio-temporal representation for the prediction of individual agent trajectories. The VAE backbones across the two stages make it possible to naturally account for the joint uncertainty at both levels of granularity. As a result, MUSEVAE offers diverse and simultaneously more accurate predictions compared to the current state-of-the-art. We demonstrate these assertions through a comprehensive set of experiments on nuScenes and SDD benchmarks as well as PFSD, a new synthetic dataset, which challenges the forecasting ability of models on complex agent-environment interaction scenarios.

----

## [222] Graph-based Spatial Transformer with Memory Replay for Multi-future Pedestrian Trajectory Prediction

**Authors**: *Lihuan Li, Maurice Pagnucco, Yang Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00227](https://doi.org/10.1109/CVPR52688.2022.00227)

**Abstract**:

Pedestrian trajectory prediction is an essential and challenging task for a variety of real-life applications such as autonomous driving and robotic motion planning. Besides generating a single future path, predicting multiple plausible future paths is becoming popular in some recent work on trajectory prediction. However, existing methods typically emphasize spatial interactions between pedestrians and surrounding areas but ignore the smoothness and temporal consistency of predictions. Our model aims to forecast multiple paths based on a historical trajectory by modeling multi-scale graph-based spatial transformers combined with a trajectory smoothing algorithm named “Memory Replay” utilizing a memory graph. Our method can comprehensively exploit the spatial information as well as correct the temporally inconsistent trajectories (e.g., sharp turns). We also propose a new evaluation metric named “Percentage of Trajectory Usage” to evaluate the comprehensiveness of diverse multi-future predictions. Our extensive experiments show that the proposed model achieves state-of-the-art performance on multi-future prediction and competitive results for single-future prediction. Code released at https://github.com/Jacobieee/ST-MR.

----

## [223] End-to-End Trajectory Distribution Prediction Based on Occupancy Grid Maps

**Authors**: *Ke Guo, Wenxi Liu, Jia Pan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00228](https://doi.org/10.1109/CVPR52688.2022.00228)

**Abstract**:

In this paper, we aim to forecast a future trajectory distribution of a moving agent in the real world, given the social scene images and historical trajectories. Yet, it is a challenging task because the ground-truth distribution is unknown and unobservable, while only one of its samples can be applied for supervising model learning, which is prone to bias. Most recent works focus on predicting diverse trajectories in order to cover all modes of the real distribution, but they may despise the precision and thus give too much credit to unrealistic predictions. To address the issue, we learn the distribution with symmetric cross-entropy using occupancy grid maps as an explicit and scene-compliant approximation to the ground-truth distribution, which can effectively penalize unlikely predictions. In specific, we present an inverse reinforcement learning based multi-modal trajectory distribution forecasting framework that learns to plan by an approximate value iteration network in an end-to-end manner. Besides, based on the predicted distribution, we generate a small set of representative trajectories through a differentiable Transformer-based network, whose attention mechanism helps to model the relations of trajectories. In experiments, our method achieves state-of-the-art performance on the Stanford Drone Dataset and Intersection Drone Dataset.

----

## [224] Learning Affordance Grounding from Exocentric Images

**Authors**: *Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00229](https://doi.org/10.1109/CVPR52688.2022.00229)

**Abstract**:

Affordance grounding, a task to ground (i.e., localize) action possibility region in objects, which faces the challenge of establishing an explicit link with object parts due to the diversity of interactive affordance. Human has the ability that transform the various exocentric interactions to invariant egocentric affordance so as to counter the impact of interactive diversity. To empower an agent with such ability, this paper proposes a task of affordance grounding from exocentric view, i.e., given exocentric human-object interaction and egocentric object images, learning the affordance knowledge of the object and transferring it to the egocentric image using only the affordance label as supervision. To this end, we devise a cross-view knowledge transfer framework that extracts affordance-specific features from exocentric interactions and enhances the perception of affordance regions by preserving affordance correlation. Specifically, an Affordance Invariance Mining module is devised to extract specific clues by minimizing the intra-class differences originated from interaction habits in exocentric images. Besides, an Affordance Co-relation Preserving strategy is presented to perceive and localize affordance by aligning the co-relation matrix of predicted results between the two views. Particularly, an affordance grounding dataset named AGD20K is constructed by collecting and labeling over 20K images from 36 affordance categories. Experimental results demonstrate that our method outperforms the representative models in terms of objective metrics and visual quality. Code: github.com/lhc1224/Cross-View-AG.

----

## [225] 3D Scene Painting via Semantic Image Synthesis

**Authors**: *Jaebong Jeong, Janghun Jo, Sunghyun Cho, Jaesik Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00230](https://doi.org/10.1109/CVPR52688.2022.00230)

**Abstract**:

We propose a novel approach to 3D scene painting using a configurable 3D scene layout. Our approach takes a 3D scene with semantic class labels as input and trains a 3D scene painting network that synthesizes color values for the input 3D scene. We exploit an off-the-shelf 2D seman-tic image synthesis method to teach the 3D painting net-work without explicit color supervision. Experiments show that our approach produces images with geometrically cor-rect structures and supports scene manipulation, such as the change of viewpoint, object poses, and painting style. Our approach provides rich controllability to synthesized images in the aspect of 3D geometry.

----

## [226] Learning Invisible Markers for Hidden Codes in Offline-to-online Photography

**Authors**: *Jun Jia, Zhongpai Gao, Dandan Zhu, Xiongkuo Min, Guangtao Zhai, Xiaokang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00231](https://doi.org/10.1109/CVPR52688.2022.00231)

**Abstract**:

QR (quick response) codes are widely used as an offline-to-online channel to convey information (e.g., links) from publicity materials (e.g., display and print) to mobile devices. However, QR codes are not favorable for taking up valuable space of publicity materials. Recent works propose invisible codes/hyperlinks that can convey hidden information from offline to online. However, they require markers to locate invisible codes, which fails the purpose of invisible codes to be visible because of the markers. This paper proposes a novel invisible information hiding architecture for display/print-camera scenarios, consisting of hiding, locating, correcting, and recovery, where invisible markers are learned to make hidden codes truly invisible. We hide information in a sub-image rather than the entire image and include a localization module in the end-to-end framework. To achieve both high visual quality and high recovering robustness, an effective multi-stage training strategy is proposed. The experimental results show that the proposed method outperforms the state-of-the-art information hiding methods in both visual quality and robustness. In addition, the automatic localization of hidden codes significantly reduces the time of manually correcting geometric distortions for photos, which is a revolutionary innovation for information hiding in mobile applications.

----

## [227] ETHSeg: An Amodel Instance Segmentation Network and a Real-world Dataset for X-Ray Waste Inspection

**Authors**: *Lingteng Qiu, Zhangyang Xiong, Xuhao Wang, Kenkun Liu, Yihan Li, Guanying Chen, Xiaoguang Han, Shuguang Cui*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00232](https://doi.org/10.1109/CVPR52688.2022.00232)

**Abstract**:

Waste inspection for packaged waste is an important step in the pipeline of waste disposal. Previous methods either rely on manual visual checking or RGB image-based inspection algorithm, requiring costly preparation procedures (e.g., open the bag and spread the waste items). Moreover, occluded items are very likely to be left out. Inspired by the fact that X-ray has a strong penetrating power to see through the bag and overlapping objects, we propose to perform waste inspection efficiently using X-ray images without the need to open the bag. We introduce a novel problem of instance-level waste segmentation in X-ray image for intelligent waste inspection, and contribute a real dataset consisting of 5,038 X-ray images (totally 30,881 waste items) with high-quality annotations (i.e., waste categories, object boxes, and instance-level masks) as a benchmark for this problem. As existing segmentation methods are mainly designed for natural images and cannot take advantage of the characteristics of X-ray waste images (e.g., heavy occlusions and penetration effect), we propose a new instance segmentation method to explicitly take these image characteristics into account. Specifically, our method adopts an easy-to-hard disassembling strategy to use high confidence predictions to guide the segmentation of highly overlapped objects, and a global structure guidance module to better capture the complex contour information caused by the penetration effect. Extensive experiments demonstrate the effectiveness of the proposed method. Our dataset is released at WIXRayNet.

----

## [228] Doodle It Yourself: Class Incremental Learning by Drawing a Few Sketches

**Authors**: *Ayan Kumar Bhunia, Viswanatha Reddy Gajjala, Subhadeep Koley, Rohit Kundu, Aneeshan Sain, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00233](https://doi.org/10.1109/CVPR52688.2022.00233)

**Abstract**:

The human visual system is remarkable in learning new visual concepts from just a few examples. This is precisely the goal behind few-shot class incremental learning (FS-CIL), where the emphasis is additionally placed on ensuring the model does not suffer from “forgetting”. In this paper, we push the boundary further for FSCIL by addressing two key questions that bottleneck its ubiquitous application (i) can the model learn from diverse modalities other than just photo (as humans do), and (ii) what if photos are not readily accessible (due to ethical and privacy constraints). Our key innovation lies in advocating the use of sketches as a new modality for class support. The product is a “Doodle It Yourself” (DIY) FSCIL framework where the users can freely sketch a few examples of a novel class for the model to learn to recognise photos of that class. For that, we present a framework that infuses (i) gradient consensus for domain invariant learning, (ii) knowledge distillation for preserving old class information, and (iii) graph attention networks for message passing between old and novel classes. We experimentally show that sketches are better class support than text in the context of FSCIL, echoing findings elsewhere in the sketching literature.

----

## [229] Image Disentanglement Autoencoder for Steganography without Embedding

**Authors**: *Xiyao Liu, Ziping Ma, Junxing Ma, Jian Zhang, Gerald Schaefer, Hui Fang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00234](https://doi.org/10.1109/CVPR52688.2022.00234)

**Abstract**:

Conventional steganography approaches embed a secret message into a carrier for concealed communication but are prone to attack by recent advanced steganalysis tools. In this paper, we propose Image DisEntanglement Autoencoder for Steganography (IDEAS) as a novel steganography without embedding (SWE) technique. Instead of directly embedding the secret message into a carrier image, our approach hides it by transforming it into a synthesised image, and is thus fundamentally immune to typical steganalysis attacks. By disentangling an image into two representations for structure and texture, we exploit the stability of structure representation to improve secret message extraction while increasing synthesis diversity via randomising texture representations to enhance steganography security. In addition, we design an adaptive mapping mechanism to further enhance the diversity of synthesised images when ensuring different required extraction levels. Experimental results convincingly demonstrate IDEAS to achieve superior performance in terms of enhanced security, reliable secret message extraction and flexible adaptation for different extraction levels, compared to state-of-the-art SWE methods.

----

## [230] Adaptive Hierarchical Representation Learning for Long-Tailed Object Detection

**Authors**: *Banghuai Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00235](https://doi.org/10.1109/CVPR52688.2022.00235)

**Abstract**:

General object detectors are always evaluated on hand-designed datasets, e.g., MS COCO and Pascal VOC, which tend to maintain balanced data distribution over different classes. However, it goes against the practical applications in the real world which suffer from a heavy class imbalance problem, known as the long-tailed object detection. In this paper, we propose a novel method, named Adaptive Hierarchical Representation Learning (AHRL), from a metric learning perspective to address long-tailed object detection. We visualize each learned class representation in the feature space, and observe that some classes, especially under-represented scarce classes, are prone to cluster with analogous ones due to the lack of discriminative representation. Inspired by this, we propose to split the whole feature space into a hierarchical structure and eliminate the problem in a coarse-to-fine way. AHRL contains a two-stage training paradigm. First, we train a normal baseline model and construct the hierarchical structure under the unsupervised clustering method. Then, we design an AHR loss that consists of two optimization objectives. On the one hand, AHR loss retains the hierarchical structure and keeps representation clusters away from each other. On the other hand, AHR loss adopts adaptive margins according to specific class pairs in the same cluster to further optimize locally. We conduct extensive experiments on the challenging LVIS dataset and AHRL outperforms all the existing state-of-the-art methods, with 29.1% segmentation AP and 29.3% box AP on LVIS v0.5 and 27.6% segmentation AP and 28.7% box AP on LVIS v1.0 based on ResNet-101. We hope our simple yet effective approach will serve as a solid baseline to help stimulate future research in long-tailed object detection. Code will be released soon

----

## [231] Semiconductor Defect Detection by Hybrid Classical-Quantum Deep Learning

**Authors**: *YuanFu Yang, Min Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00236](https://doi.org/10.1109/CVPR52688.2022.00236)

**Abstract**:

With the rapid development of artificial intelligence and autonomous driving technology, the demand for semiconductors is projected to rise substantially. However, the massive expansion of semiconductor manufacturing and the development of new technology will bring many defect wafers. If these defect wafers have not been correctly inspected, the ineffective semiconductor processing on these defect wafers will cause additional impact to our environment, such as excessive carbon dioxide emission and energy consumption. In this paper, we utilize the information processing advantages of quantum computing to promote the defect learning defect review (DLDR). We propose a classical-quantum hybrid algorithm for deep learning on near-term quantum processors. By tuning parameters implemented on it, quantum circuit driven by our framework learns a given DLDR task, include of wafer defect map classification, defect pattern classification, and hotspot detection. In addition, we explore parametrized quantum circuits with different expressibility and entangling capacities. These results can be used to build a future roadmap to develop circuit-based quantum deep learning for semiconductor defect detection.

----

## [232] Density-preserving Deep Point Cloud Compression

**Authors**: *Yun He, Xinlin Ren, Danhang Tang, Yinda Zhang, Xiangyang Xue, Yanwei Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00237](https://doi.org/10.1109/CVPR52688.2022.00237)

**Abstract**:

Local density of point clouds is crucial for representing local details, but has been overlooked by existing point cloud compression methods. To address this, we propose a novel deep point cloud compression method that preserves local density information. Our method works in an auto-encoder fashion: the encoder downsamples the points and learns point-wise features, while the decoder upsamples the points using these features. Specifically, we propose to encode local geometry and density with three embeddings: density embedding, local position embedding and ancestor embedding. During the decoding, we explicitly predict the upsampling factor for each point, and the directions and scales of the upsampled points. To mitigate the clustered points issue in existing methods, we design a novel sub-point convolution layer, and an upsampling block with adaptive scale. Furthermore, our method can also compress point-wise attributes, such as normal. Extensive qualitative and quantitative results on SemanticKITTI and ShapeNet demonstrate that our method achieves the state-of-the-art rate-distortion trade-off.

----

## [233] Graph-context Attention Networks for Size-varied Deep Graph Matching

**Authors**: *Zheheng Jiang, Hossein Rahmani, Plamen P. Angelov, Sue Black, Bryan M. Williams*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00238](https://doi.org/10.1109/CVPR52688.2022.00238)

**Abstract**:

Deep learning for graph matching has received growing interest and developed rapidly in the past decade. Although recent deep graph matching methods have shown excellent performance on matching between graphs of equal size in the computer vision area, the size-varied graph matching problem, where the number of keypoints in the images of the same category may vary due to occlusion, is still an open and challenging problem. To tackle this, we firstly propose to formulate the combinatorial problem of graph matching as an Integer Linear Programming (ILP) problem, which is more flexible and efficient to facilitate comparing graphs of varied sizes. A novel Graph-context Attention Network (GCAN), which jointly capture intrinsic graph structure and cross-graph information for improving the discrimination of node features, is then proposed and trained to resolve this ILP problem with node correspondence supervision. We further show that the proposed GCAN model is efficient to resolve the graph-level matching problem and is able to automatically learn node-to-node similarity via graph-level matching. The proposed approach is evaluated on three public keypoint-matching datasets and one graph-matching dataset for blood vessel patterns, with experimental results showing its superior performance over existing state-of-the-art algorithms for keypoint and graph-level matching.

----

## [234] TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions

**Authors**: *Jeya Maria Jose Valanarasu, Rajeev Yasarla, Vishal M. Patel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00239](https://doi.org/10.1109/CVPR52688.2022.00239)

**Abstract**:

Removing adverse weather conditions like rain, fog, and snow from images is an important problem in many applications. Most methods proposed in the literature have been designed to deal with just removing one type of degradation. Recently, a CNN-based method using neural architecture search (All-in-One) was proposed to remove all the weather conditions at once. However, it has a large number of parameters as it uses multiple encoders to cater to each weather removal task and still has scope for improvement in its performance. In this work, we focus on developing an efficient solution for the all adverse weather removal problem. To this end, we propose TransWeather, a transformer-based end-to-end model with just a single encoder and a decoder that can restore an image degraded by any weather condition. Specifically, we utilize a novel transformer encoder using intra-patch transformer blocks to enhance attention inside the patches to effectively remove smaller weather degradations. We also introduce a transformer decoder with learnable weather type embeddings to adjust to the weather degradation at hand. Trans Weather achieves significant improvements across multiple test datasets over both All-in-One network as well as methods fine-tuned for specific tasks. TransWeather is also validated on real world test images and found to be more effective than previous methods. Implementation code can be found in the supplementary document. Code is available at https//github.com/jeya-maria-jose/TransWeather.

----

## [235] ObjectFormer for Image Manipulation Detection and Localization

**Authors**: *Junke Wang, Zuxuan Wu, Jingjing Chen, Xintong Han, Abhinav Shrivastava, Ser-Nam Lim, Yu-Gang Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00240](https://doi.org/10.1109/CVPR52688.2022.00240)

**Abstract**:

Recent advances in image editing techniques have posed serious challenges to the trustworthiness of multimedia data, which drives the research of image tampering detection. In this paper, we propose ObjectFormer to detect and localize image manipulations. To capture subtle manipulation traces that are no longer visible in the RGB domain, we extract high-frequency features of the images and combine them with RGB features as multimodal patch embeddings. Additionally, we use a set of learnable object prototypes as mid-level representations to model the object-level consistencies among different regions, which are further used to refine patch embeddings to capture the patch-level consistencies. We conduct extensive experiments on various datasets and the results verify the effectiveness of the proposed method, outperforming state-of-the-art tampering detection and localization methods.

----

## [236] Sequential Voting with Relational Box Fields for Active Object Detection

**Authors**: *Qichen Fu, Xingyu Liu, Kris M. Kitani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00241](https://doi.org/10.1109/CVPR52688.2022.00241)

**Abstract**:

A key component of understanding hand-object interactions is the ability to identify the active object-the object that is being manipulated by the human hand. In order to accurately localize the active object, any method must reason using information encoded by each image pixel, such as whether it belongs to the hand, the object, or the background. To leverage each pixel as evidence to determine the bounding box of the active object, we propose a pixel-wise voting function. Our pixel-wise voting function takes an initial bounding box as input and produces an improved bounding box of the active object as output. The voting function is designed so that each pixel inside of the input bounding box votes for an improved bounding box, and the box with the majority vote is selected as the output. We call the collection of bounding boxes generated inside of the voting function, the Relational Box Field, as it characterizes a field of bounding boxes defined in relationship to the current bounding box. While our voting function is able to improve the bounding box of the active object, one round of voting is typically not enough to accurately localize the active object. Therefore, we repeatedly apply the voting function to sequentially improve the location of the bounding box. However, since it is known that repeatedly applying a one-step predictor (i.e., auto-regressive processing with our voting function) can cause a data distribution shift, we mitigate this issue using reinforcement learning (RL). We adopt standard RL to learn the voting function parameters and show that it provides a meaningful improvement over a standard supervised learning approach. We perform experiments on two large-scale datasets: 100DOH and MECCANO, improving AP50 performance by 8% and 30%, respectively, over the state of the art. The project page with code and visualizations can be found at https://fuqichen1998.github.io/SequentialVotingDet/.

----

## [237] Efficient Classification of Very Large Images with Tiny Objects

**Authors**: *Fanjie Kong, Ricardo Henao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00242](https://doi.org/10.1109/CVPR52688.2022.00242)

**Abstract**:

An increasing number of applications in computer vision, specially, in medical imaging and remote sensing, become challenging when the goal is to classify very large images with tiny informative objects. Specifically, these classification tasks face two key challenges: i) the size of the input image is usually in the order of mega- or giga-pixels, however, existing deep architectures do not easily operate on such big images due to memory constraints, consequently, we seek a memory-efficient method to process these images; and ii) only a very small fraction of the input images are informative of the label of interest, resulting in low region of interest (ROI) to image ratio. However, most of the current convolutional neural networks (CNNs) are designed for image classification datasets that have relatively large ROIs and small image sizes (sub-megapixel). Existing approaches have addressed these two challenges in isolation. We present an end-to-end CNN model termed Zoom-In network that leverages hierarchical attention sampling for classification of large images with tiny objects using a single GPU. We evaluate our method on four large-image histopathology, road-scene and satellite imaging datasets, and one gigapixel pathology dataset. Experimental results show that our model achieves higher accuracy than existing methods while requiring less memory resources.

----

## [238] Partially Does It: Towards Scene-Level FG-SBIR with Partial Input

**Authors**: *Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Viswanatha Reddy Gajjala, Aneeshan Sain, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00243](https://doi.org/10.1109/CVPR52688.2022.00243)

**Abstract**:

We scrutinise an important observation plaguing scene-level sketch research - that a significant portion of scene sketches are “partial”. A quick pilot study reveals: (i) a scene sketch does not necessarily contain all objects in the corresponding photo, due to the subjective holistic interpretation of scenes, (ii) there exists significant empty (white) regions as a result of object-level abstraction, and as a result, (iii) existing scene-level fine-grained sketch-based image retrieval methods collapse as scene sketches become more partial. To solve this “partial” problem, we advocate for a simple set-based approach using optimal transport (OT) to model cross-modal region associativity in a partially-aware fashion. Importantly, we improve upon OT to further account for holistic partialness by comparing intra-modal adjacency matrices. Our proposed method is not only robust to partial scene-sketches but also yields state-of-the-art performance on existing datasets.

----

## [239] Long-term Visual Map Sparsification with Heterogeneous GNN

**Authors**: *Ming-Fang Chang, Yipu Zhao, Rajvi Shah, Jakob J. Engel, Michael Kaess, Simon Lucey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00244](https://doi.org/10.1109/CVPR52688.2022.00244)

**Abstract**:

We address the problem of map sparsification for long-term visual localization. For map sparsification, a commonly employed assumption is that the pre-build map and the later captured localization query are consistent. However, this assumption can be easily violated in the dynamic world. Additionally, the map size grows as new data accumulate through time, causing large data overhead in the long term. In this paper, we aim to overcome the environmental changes and reduce the map size at the same time by selecting points that are valuable to future localization. Inspired by the recent progress in Graph Neural Network (GNN), we propose the first work that models SfM maps as heterogeneous graphs and predicts 3D point importance scores with a GNN, which enables us to directly exploit the rich information in the SfM map graph. Two novel supervisions are proposed: 1) a data-fitting term for selecting valuable points to future localization based on training queries; 2) a K-Cover term for selecting sparse points with full-map coverage. The experiments show that our method selected map points on stable and widely visible structures and out-performed baselines in localization performance

----

## [240] Connecting the Complementary-view Videos: Joint Camera Identification and Subject Association

**Authors**: *Ruize Han, Yiyang Gan, Jiacheng Li, Feifan Wang, Wei Feng, Song Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00245](https://doi.org/10.1109/CVPR52688.2022.00245)

**Abstract**:

We attempt to connect the data from complementary views, i.e., top view from drone-mounted cameras in the air, and side view from wearable cameras on the ground. Collaborative analysis of such complementary-view data can facilitate to build the air-ground cooperative visual system for various kinds of applications. This is a very challenging problem due to the large view difference between top and side views. In this paper, we develop a new approach that can simultaneously handle three tasks: i) localizing the side-view camera in the top view; ii) estimating the view direction of the side-view camera; iii) detecting and associating the same subjects on the ground across the complementary views. Our main idea is to explore the spatial position layout of the subjects in two views. In particular, we propose a spatial-aware position representation method to embed the spatial-position distribution of the subjects in different views. We further design a cross-view video collaboration framework composed of a camera identification module and a subject association module to simultaneously perform the above three tasks. We collect a new synthetic dataset consisting of top-view and side-view video sequence pairs for performance evaluation and the experimental results show the effectiveness of the proposed method.

----

## [241] DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation

**Authors**: *Gwanghyun Kim, Taesung Kwon, Jong Chul Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00246](https://doi.org/10.1109/CVPR52688.2022.00246)

**Abstract**:

Recently, GAN inversion methods combined with Contrastive Language-Image Pretraining (CLIP) enables zeroshot image manipulation guided by text prompts. However, their applications to diverse real images are still difficult due to the limited GAN inversion capability. Specifically, these approaches often have difficulties in reconstructing images with novel poses, views, and highly variable contents compared to the training data, altering object identity, or producing unwanted image artifacts. To mitigate these problems and enable faithful manipulation of real images, we propose a novel method, dubbed DiffusionCLIP, that performs textdriven image manipulation using diffusion models. Based on full inversion capability and high-quality image generation power of recent diffusion models, our method performs zeroshot image manipulation successfully even between unseen domains and takes another step towards general application by manipulating images from a widely varying ImageNet dataset. Furthermore, we propose a novel noise combination method that allows straightforward multi-attribute manipulation. Extensive experiments and human evaluation confirmed robust and superior manipulation performance of our methods compared to the existing baselines. Code is available at https://github.com/gwang-kim/DiffusionCLIP.git

----

## [242] Aesthetic Text Logo Synthesis via Content-aware Layout Inferring

**Authors**: *Yizhi Wang, Guo Pu, Wenhan Luo, Yexin Wang, Pengfei Xiong, Hongwen Kang, Zhouhui Lian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00247](https://doi.org/10.1109/CVPR52688.2022.00247)

**Abstract**:

Text logo design heavily relies on the creativity and expertise of professional designers, in which arranging element layouts is one of the most important procedures. However, few attention has been paid to this task which needs to take many factors (e.g., fonts, linguistics, topics, etc.) into consideration. In this paper, we propose a content-aware layout generation network which takes glyph images and their corresponding text as input and synthesizes aesthetic layouts for them automatically. Specifically, we develop a dual-discriminator module, including a sequence discriminator and an image discriminator, to evaluate both the character placing trajectories and rendered shapes of synthesized text logos, respectively. Furthermore, we fuse the information of linguistics from texts and visual semantics from glyphs to guide layout prediction, which both play important roles in professional layout design. To train and evaluate our approach, we construct a dataset named as TextLogo3K, consisting of about 3,500 text logo images and their pixel-level annotations. Experimental studies on this dataset demonstrate the effectiveness of our approach for synthesizing visually-pleasing text logos and verify its superiority against the state of the art.

----

## [243] Rethinking Image Cropping: Exploring Diverse Compositions from Global Views

**Authors**: *Gengyun Jia, Huaibo Huang, Chaoyou Fu, Ran He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00248](https://doi.org/10.1109/CVPR52688.2022.00248)

**Abstract**:

Existing image cropping works mainly use anchor evaluation methods or coordinate regression methods. However, it is difficult for pre-defined anchors to cover good crops globally, and the regression methods ignore the cropping diversity. In this paper, we regard image cropping as a set prediction problem. A set of crops regressed from multiple learnable anchors is matched with the labeled good crops, and a classifier is trained using the matching results to select a valid subset from all the predictions. This new perspective equips our model with globality and diversity, mitigating the shortcomings but inherit the strengthens of previous methods. Despite the advantages, the set prediction method causes inconsistency between the validity labels and the crops. To deal with this problem, we propose to smooth the validity labels with two different methods. The first method that uses crop qualities as direct guidance is designed for the datasets with nearly dense quality labels. The second method based on the self distillation can be used in sparsely labeled datasets. Experimental results on the public datasets show the merits of our approach over state-of-the-art counterparts.

----

## [244] Defensive Patches for Robust Recognition in the Physical World

**Authors**: *Jiakai Wang, Zixin Yin, Pengfei Hu, Aishan Liu, Renshuai Tao, Haotong Qin, Xianglong Liu, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00249](https://doi.org/10.1109/CVPR52688.2022.00249)

**Abstract**:

To operate in real-world high-stakes environments, deep learning systems have to endure noises that have been con-tinuously thwarting their robustness. Data-end defense, which improves robustness by operations on input data in-stead of modifying models, has attracted intensive attention due to its feasibility in practice. However, previous data-end defenses show low generalization against diverse noises and weak transferability across multiple models. Motivated by the fact that robust recognition depends on both local and global features, we propose a defensive patch generation framework to address these problems by helping mod-els better exploit these features. For the generalization against diverse noises, we inject class-specific identifiable patterns into a confined local patch prior, so that defensive patches could preserve more recognizable features towards specific classes, leading models for better recognition under noises. For the transferability across multiple models, we guide the defensive patches to capture more global fea-ture correlations within a class, so that they could activate model-shared global perceptions and transfer better among models. Our defensive patches show great potentials to im-prove application robustness in practice by simply sticking them around target objects. Extensive experiments show that we outperform others by large margins (improve 20+ % accuracy for both adversarial and corruption robustness on average in the digital and physical world).
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our codes are available at https://github.com/nlsde-safety-team/DefensivePatch.

----

## [245] Semi-supervised Video Paragraph Grounding with Contrastive Encoder

**Authors**: *Xun Jiang, Xing Xu, Jingran Zhang, Fumin Shen, Zuo Cao, Heng Tao Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00250](https://doi.org/10.1109/CVPR52688.2022.00250)

**Abstract**:

Video events grounding aims at retrieving the most relevant moments from an untrimmed video in terms of a given natural language query. Most previous works focus on Video Sentence Grounding (VSG), which localizes the moment with a sentence query. Recently, researchers extended this task to Video Paragraph Grounding (VPG) by retrieving multiple events with a paragraph. However, we find the existing VPG methods may not perform well on context modeling and highly rely on video-paragraph annotations. To tackle this problem, we propose a novel VPG method termed Semi-supervised Video-Paragraph TRansformer (SVPTR), which can more effectively exploit contextual information in paragraphs and significantly reduce the dependency on annotated data. Our SVPTR method consists of two key components: (1) a base model VPTR that learns the video-paragraph alignment with contrastive encoders and tackles the lack of sentence-level contextual interactions and (2) a semi-supervised learning framework with multimodal feature perturbations that reduces the requirements of annotated training data. We evaluate our model on three widely-used video grounding datasets, i.e., ActivityNet-Caption, Charades-CD-OOD, and TACoS. The experimental results show that our SVPTR method establishes the new state-of-the-art performance on all datasets. Even under the conditions of fewer annotations, it can also achieve competitive results compared with recent VPG methods.

----

## [246] Meta Distribution Alignment for Generalizable Person Re-Identification

**Authors**: *Hao Ni, Jingkuan Song, Xiaopeng Luo, Feng Zheng, Wen Li, Heng Tao Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00252](https://doi.org/10.1109/CVPR52688.2022.00252)

**Abstract**:

Domain Generalizable (DG) person ReID is a challenging task which trains a model on source domains yet generalizes well on target domains. Existing methods use source domains to learn domain-invariant features, and assume those features are also irrelevant with target domains. However, they do not consider the target domain information which is unavailable in the training phrase of DG. To address this issue, we propose a novel Meta Distribution Alignment (MDA) method to enable them to share similar distribution in a test-time-training fashion. Specifically, since high-dimensional features are difficult to constrain with a known simple distribution, we first introduce an intermediate latent space constrained to a known prior distribution. The source domain data is mapped to this latent space and then reconstructed back. A meta-learning strategy is introduced to facilitate generalization and support fast adaption. To reduce their discrepancy, we further propose a test-time adaptive updating strategy based on the latent space which efficiently adapts model to unseen domains with a few samples. Extensive experimental results show that our model outperforms the state-of-the-art methods by up to 5.2% R-1 on average on the large-scale and 4.7% R-1 on the single-source domain generalization ReID benchmark. Source code is publicly available at https://github.com/haoni0812/MDA.git.

----

## [247] FvOR: Robust Joint Shape and Pose Optimization for Few-view Object Reconstruction

**Authors**: *Zhenpei Yang, Zhile Ren, Miguel Ángel Bautista, Zaiwei Zhang, Qi Shan, Qixing Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00253](https://doi.org/10.1109/CVPR52688.2022.00253)

**Abstract**:

Reconstructing an accurate 3D object model from a few image observations remains a challenging problem in computer vision. State-of-the-art approaches typically assume accurate camera poses as input, which could be difficult to obtain in realistic settings. In this paper, we present FvOR, a learning-based object reconstruction method that predicts accurate 3D models given a few images with noisy input poses. The core of our approach is a fast and robust multi-view reconstruction algorithm to jointly refine 3D geometry and camera pose estimation using learnable neural network modules. We provide a thorough benchmark of state-of-the-art approaches for this problem on ShapeNet. Our approach achieves best-in-class results. It is also two orders of magnitude faster than the recent optimization-based approach IDR [67].

----

## [248] It's About Time: Analog Clock Reading in the Wild

**Authors**: *Charig Yang, Weidi Xie, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00254](https://doi.org/10.1109/CVPR52688.2022.00254)

**Abstract**:

In this paper, we present a framework for reading analog clocks in natural images or videos. Specifically, we make the following contributions: First, we create a scalable pipeline for generating synthetic clocks, significantly reducing the requirements for the labour-intensive annotations; Second, we introduce a clock recognition architecture based on spatial transformer networks (STN), which is trained end-to-end for clock alignment and recognition. We show that the model trained on the proposed synthetic dataset generalises towards real clocks with good accuracy, advocating a Sim2Real training regime; Third, to further reduce the gap between simulation and real data, we leverage the special property of “time”, i.e. uniformity, to generate reliable pseudo-labels on real unlabelled clock videos, and show that training on these videos offers further improvements while still requiring zero manual annotations. Lastly, we introduce three benchmark datasets based on COCO, Open Images, and The Clock movie, with full annotations for time, accurate to the minute.

----

## [249] Consistency driven Sequential Transformers Attention Model for Partially Observable Scenes

**Authors**: *Samrudhdhi B. Rangrej, Chetan L. Srinidhi, James J. Clark*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00255](https://doi.org/10.1109/CVPR52688.2022.00255)

**Abstract**:

Most hard attention models initially observe a complete scene to locate and sense informative glimpses, and predict class-label of a scene based on glimpses. However, in many applications (e.g., aerial imaging), observing an entire scene is not always feasible due to the limited time and resources available for acquisition. In this paper, we develop a Sequential Transformers Attention Model (STAM) that only partially observes a complete image and predicts informative glimpse locations solely based on past glimpses. We design our agent using DeiT-distilled [44] and train it with a one-step actorcritic algorithm. Furthermore, to improve classification performance, we introduce a novel training objective, which enforces consistency between the class distribution predicted by a teacher model from a complete image and the class distribution predicted by our agent using glimpses. When the agent senses only 4% of the total image area, the inclusion of the proposed consistency loss in our training objective yields 3% and 8% higher accuracy on ImageNet and fMoW datasets, respectively. Moreover, our agent outperforms previous state-of-the-art by observing nearly 27% and 42% fewer pixels in glimpses on ImageNet and fMoW.

----

## [250] Smartadapt: Multi-branch Object Detection Framework for Videos on Mobiles

**Authors**: *Ran Xu, Fangzhou Mu, Jayoung Lee, Preeti Mukherjee, Somali Chaterji, Saurabh Bagchi, Yin Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00256](https://doi.org/10.1109/CVPR52688.2022.00256)

**Abstract**:

Several recent works seek to create lightweight deep net-works for video object detection on mobiles. We observe that many existing detectors, previously deemed computationally costly for mobiles, intrinsically support adaptive inference, and offer a multi-branch object detection frame-work (MBODF). Here, an MBODF is referred to as a so-lution that has many execution branches and one can dy-namically choose from among them at inference time to sat-isfy varying latency requirements (e.g. by varying resolution of an input frame). In this paper, we ask, and answer, the wide-ranging question across all MBODFs: How to expose the right set of execution branches and then how to sched-ule the optimal one at inference time? In addition, we un-cover the importance of making a content-aware decision on which branch to run, as the optimal one is conditioned on the video content. Finally, we explore a content-aware scheduler, an Oracle one, and then a practical one, leveraging various lightweight feature extractors. Our evaluation shows that layered on Faster R-CNN-based MBODF, compared to 7 baselines, our Smartadapt achieves a higher Pareto optimal curve in the accuracy-vs-latency space for the ILSVRC VID dataset.

----

## [251] Generating 3D Bio-Printable Patches Using Wound Segmentation and Reconstruction to Treat Diabetic Foot Ulcers

**Authors**: *Han Joo Chae, Seunghwan Lee, Hyewon Son, Seungyeob Han, Taebin Lim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00257](https://doi.org/10.1109/CVPR52688.2022.00257)

**Abstract**:

We introduce AiD Regen, a novel system that generates 3D wound models combining 2D semantic segmentation with 3D reconstruction so that they can be printed via 3D bio-printers during the surgery to treat diabetic foot ulcers (DFUs). AiD Regen seamlessly binds the full pipeline, which includes RGB-D image capturing, semantic segmentation, boundary-guided point-cloud processing, 3D model reconstruction, and 3D printable G-code generation, into a single system that can be used out of the box. We developed a multi-stage data preprocessing method to handle small and unbalanced DFU image datasets. AiD Regen's human-in-the-loop machine learning interface enables clinicians to not only create 3D regenerative patches with just a few touch interactions but also customize and confirm wound boundaries. As evidenced by our experiments, our model outperforms prior wound segmentation models and our reconstruction algorithm is capable of generating 3D wound models with compelling accuracy. We further conducted a case study on a real DFU patient and demonstrated the effectiveness of AiD Regen in treating DFU wounds.

----

## [252] Investigating the Impact of Multi-LiDAR Placement on Object Detection for Autonomous Driving

**Authors**: *Hanjiang Hu, Zuxin Liu, Sharad Chitlangia, Akhil Agnihotri, Ding Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00258](https://doi.org/10.1109/CVPR52688.2022.00258)

**Abstract**:

The past few years have witnessed an increasing interest in improving the perception performance of LiDARs on au-tonomous vehicles. While most of the existing works focus on developing new deep learning algorithms or model ar-chitectures, we study the problem from the physical design perspective, i.e., how different placements of multiple Li-DARs influence the learning-based perception. To this end, we introduce an easy-to-compute information-theoretic sur-rogate metric to quantitatively and fast evaluate LiDAR placement for 3D detection of different types of objects. We also present a new data collection, detection model training and evaluation framework in the realistic CARLA simula-tor to evaluate disparate multi-LiDAR configurations. Using several prevalent placements inspired by the designs of self-driving companies, we show the correlation between our surrogate metric and object detection performance of different representative algorithms on KITTI through exten-sive experiments, validating the effectiveness of our LiDAR placement evaluation approach. Our results show that sen-sor placement is non-negligible in 3D point cloud-based ob-ject detection, which will contribute to 5% ~ 10% performance discrepancy in terms of average precision in chal-lenging 3D object detection settings. We believe that this is one of the first studies to quantitatively investigate the influence of LiDAR placement on perception performance.

----

## [253] CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation

**Authors**: *Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell D. Collins, Yukun Zhu, Hartwig Adam, Alan L. Yuille, Liang-Chieh Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00259](https://doi.org/10.1109/CVPR52688.2022.00259)

**Abstract**:

We propose Clustering Mask Transformer (CMT-DeepLab), a transformer-based framework for panoptic segmentation designed around clustering. It rethinks the existing transformer architectures used in segmentation and detection; CMT-DeepLab considers the object queries as cluster centers, which fill the role of grouping the pixels when applied to segmentation. The clustering is computed with an alternating procedure, by first assigning pixels to the clusters by their feature affinity, and then updating the cluster centers and pixel features. Together, these operations comprise the Clustering Mask Transformer (CMT) layer, which produces cross-attention that is denser and more consistent with the final segmentation task. CMT-DeepLab improves the performance over prior art significantly by 4.4% PQ, achieving a new state-of-the-art of 55.7% PQ on the COCO test-dev set.

----

## [254] Unsupervised Hierarchical Semantic Segmentation with Multiview Cosegmentation and Clustering Transformers

**Authors**: *Tsung-Wei Ke, Jyh-Jing Hwang, Yunhui Guo, Xudong Wang, Stella X. Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00260](https://doi.org/10.1109/CVPR52688.2022.00260)

**Abstract**:

Unsupervised semantic segmentation aims to discover groupings within and across images that capture object-and view-invariance of a category without external supervision. Grouping naturally has levels of granularity, creating ambiguity in unsupervised segmentation. Existing methods avoid this ambiguity and treat it as a factor outside modeling, whereas we embrace it and desire hierarchical grouping consistency for unsupervised segmentation. We approach unsupervised segmentation as a pixel-wise feature learning problem. Our idea is that a good representation shall reveal not just a particular level of grouping, but any level of grouping in a consistent and predictable manner. We enforce spatial consistency of grouping and bootstrap feature learning with co-segmentation among multiple views of the same image, and enforce semantic consistency across the grouping hierarchy with clustering transformers between coarse- and fine-grained features. We deliver the first data-driven unsupervised hierarchical semantic segmentation method called Hierarchical Segment Grouping (HSG). Capturing visual similarity and statistical co-occurrences, HSG also outperforms existing un-supervised segmentation methods by a large margin on five major object- and scene-centric benchmarks.

----

## [255] Rethinking Semantic Segmentation: A Prototype View

**Authors**: *Tianfei Zhou, Wenguan Wang, Ender Konukoglu, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00261](https://doi.org/10.1109/CVPR52688.2022.00261)

**Abstract**:

Prevalent semantic segmentation solutions, despite their different network designs (FCN based or attention based) and mask decoding strategies (parametric softmax based or pixel-query based), can be placed in one category, by considering the softmax weights or query vectors as learnable class prototypes. In light of this prototype view, this study uncovers several limitations of such parametric segmentation regime, and proposes a nonparametric alternative based on non-learnable prototypes. Instead of prior methods learning a single weight/query vector for each class in a fully parametric manner, our model represents each class as a set of non-learnable prototypes, relying solely on the mean fea-tures of several training pixels within that class. The dense prediction is thus achieved by nonparametric nearest prototype retrieving. This allows our model to directly shape the pixel embedding space, by optimizing the arrangement between embedded pixels and anchored prototypes. It is able to handle arbitrary number of classes with a constant amount of learnable parameters. We empirically show that, with FCN based and attention based segmentation models (i.e., HR-Net, Swin, SegFormer) and backbones (i.e., ResNet, HRNet, Swin, MiT), our nonparametric framework yields compel-ling results over several datasets (i.e., ADE20K, Cityscapes, COCO-Stuff), and performs well in the large-vocabulary situation. We expect this work will provoke a rethink of the current de facto semantic segmentation model design.

----

## [256] Semantic-Aware Domain Generalized Segmentation

**Authors**: *Duo Peng, Yinjie Lei, Munawar Hayat, Yulan Guo, Wen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00262](https://doi.org/10.1109/CVPR52688.2022.00262)

**Abstract**:

Deep models trained on source domain lack generalization when evaluated on unseen target domains with different data distributions. The problem becomes even more pro-nounced when we have no access to target domain samples for adaptation. In this paper, we address domain generalized semantic segmentation, where a segmentation model is trained to be domain-invariant without using any target domain data. Existing approaches to tackle this problem standardize data into a unified distribution. We argue that while such a standardization promotes global normalization, the resulting features are not discriminative enough to get clear segmentation boundaries. To enhance separation between categories while simultaneously promoting domain invariance, we propose a framework including two novel modules: Semantic-Aware Normalization (SAN) and Semantic-Aware Whitening (SAW). Specifically, SAN focuses on category-level center alignment between features from different image styles, while SAW enforces distributed alignment for the already center-aligned features. With the help of SAN and SAW, we encourage both intra-category compactness and inter-category separability. We validate our approach through extensive experiments on widely-used datasets (i.e. GTAV, SYNTHIA, Cityscapes, Mapillary and BDDS). Our approach shows significant improvements over existing state-of-the-art on various backbone networks. Code is available at https://github.com/leolyj/SAN-SAW

----

## [257] Adaptive Early-Learning Correction for Segmentation from Noisy Annotations

**Authors**: *Sheng Liu, Kangning Liu, Weicheng Zhu, Yiqiu Shen, Carlos Fernandez-Granda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00263](https://doi.org/10.1109/CVPR52688.2022.00263)

**Abstract**:

Deep learning in the presence of noisy annotations has been studied extensively in classification, but much less in segmentation tasks. In this work, we study the learning dynamics of deep segmentation networks trained on inaccurately annotated data. We observe a phenomenon that has been previously reported in the context of classification: the networks tend to first fit the clean pixel-level labels during an “early-learning” phase, before eventually memorizing the false annotations. However, in contrast to classification, memorization in segmentation does not arise simultaneously for all semantic categories. Inspired by these findings, we propose a new method for segmentation from noisy annotations with two key elements. First, we detect the beginning of the memorization phase separately for each category during training. This allows us to adaptively correct the noisy annotations in order to exploit early learning. Second, we incorporate a regularization term that enforces consistency across scales to boost robustness against annotation noise. Our method outperforms standard approaches on a medical-imaging segmentation task where noises are synthesized to mimic human annotation errors. It also provides robustness to realistic noisy annotations present in weakly-supervised semantic segmentation, achieving state-of-the-art results on PASCAL VOC 2012. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/Kangningthu/ADELE

----

## [258] Pointly-Supervised Instance Segmentation

**Authors**: *Bowen Cheng, Omkar Parkhi, Alexander Kirillov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00264](https://doi.org/10.1109/CVPR52688.2022.00264)

**Abstract**:

We propose an embarrassingly simple point annotation scheme to collect weak supervision for instance segmentation. In addition to bounding boxes, we collect binary labels for a set of points uniformly sampled inside each bounding box. We show that the existing instance segmentation models developed for full mask supervision can be seamlessly trained with point-based supervision collected via our scheme. Remarkably, Mask R-CNN trained on COCO, PASCAL VOC, Cityscapes, and LVIS with only 10 annotated random points per object achieves 94%−98% of its fully-supervised performance, setting a strong baseline for weakly-supervised instance segmentation. The new point annotation scheme is approximately 5 times faster than annotating full object masks, making high-quality instance segmentation more accessible in practice. Inspired by the point-based annotation form, we propose a modification to PointRend instance segmentation module. For each object, the new architecture, called Implicit PointRend, generates parameters for a function that makes the final point-level mask prediction. Implicit PointRend is more straightforward and uses a single point-level mask loss. Our experiments show that the new module is more suitable for the point-based supervision.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://bowenc0221.github.io/point-sup

----

## [259] Joint Forecasting of Panoptic Segmentations with Difference Attention

**Authors**: *Colin Graber, Cyril Jazra, Wenjie Luo, Liangyan Gui, Alexander G. Schwing*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00265](https://doi.org/10.1109/CVPR52688.2022.00265)

**Abstract**:

Forecasting of a representation is important for safe and effective autonomy. For this, panoptic segmentations have been studied as a compelling representation in recent work. However, recent state-of-the-art on panoptic segmentation forecasting suffers from two issues: first, individual object instances are treated independently of each other; second, individual object instance forecasts are merged in a heuristic manner. To address both issues, we study a new panoptic segmentation forecasting model that jointly forecasts all object instances in a scene using a transformer model based on ‘difference attention.’ It further refines the predictions by taking depth estimates into account. We evaluate the proposed model on the Cityscapes and AIODrive datasets. We find difference attention to be particularly suitable for forecasting because the difference of quantities like locations enables a model to explicitly reason about velocities and acceleration. Because of this, we attain state-of-the-art on panoptic segmentation forecasting metrics.

----

## [260] FocusCut: Diving into a Focus View in Interactive Segmentation

**Authors**: *Zheng Lin, Zheng-Peng Duan, Zhao Zhang, Chun-Le Guo, Ming-Ming Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00266](https://doi.org/10.1109/CVPR52688.2022.00266)

**Abstract**:

Interactive image segmentation is an essential tool in pixel-level annotation and image editing. To obtain a high-precision binary segmentation mask, users tend to add interaction clicks around the object details, such as edges and holes, for efficient refinement. Current methods regard these repair clicks as the guidance to jointly determine the global prediction. However, the global view makes the model lose focus from later clicks, and is not in line with user intentions. In this paper, we dive into the view of clicks' eyes to endow them with the decisive role in object details again. To verify the necessity of focus view, we design a simple yet effective pipeline, named FocusCut, which integrates the functions of object segmentation and local refinement. After obtaining the global prediction, it crops click-centered patches from the original image with adaptive scopes to refine the local predictions progressively. Without user perception and parameters increase, our method has achieved state-of-the-art results. Extensive experiments and visualized results demonstrate that FocusCut makes hyper-fine segmentation possible for interactive image segmentation.

----

## [261] Human Instance Matting via Mutual Guidance and Multi-Instance Refinement

**Authors**: *Yanan Sun, Chi-Keung Tang, Yu-Wing Tai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00267](https://doi.org/10.1109/CVPR52688.2022.00267)

**Abstract**:

This paper introduces a new matting task called human instance matting (HIM), which requires the pertinent model to automatically predict a precise alpha matte for each human instance. Straightforward combination of closely related techniques, namely, instance segmentation, soft segmentation and human/conventional matting, will easily fail in complex cases requiring disentangling mingled colors belonging to multiple instances along hairy and thin boundary structures. To tackle these technical challenges, we propose a human instance matting framework, called InstMatt, where a novel mutual guidance strategy working in tandem with a multi-instance refinement module is used, for delineating multi-instance relationship among humans with complex and overlapping boundaries if present. A new instance matting metric called instance matting quality (IMQ) is proposed, which addresses the absence of a unified and fair means of evaluation emphasizing both instance recognition and matting quality. Finally, we construct a HIM benchmark for evaluation, which comprises of both synthetic and natural benchmark images. In addition to thorough experimental results on complex cases with multiple and overlapping human instances each has intricate boundaries, preliminary results are presented on general instance matting. Code and benchmark are available in https://github.com/nowsyn/InstMatt.

----

## [262] Deformable Sprites for Unsupervised Video Decomposition

**Authors**: *Vickie Ye, Zhengqi Li, Richard Tucker, Angjoo Kanazawa, Noah Snavely*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00268](https://doi.org/10.1109/CVPR52688.2022.00268)

**Abstract**:

We describe a method to extract persistent elements of a dynamic scene from an input video. We represent each scene element as a Deformable Sprite consisting of three components: 1) a 2D texture image for the entire video, 2) per-frame masks for the element, and 3) non-rigid deformations that map the texture image into each video frame. The resulting decomposition allows for applications such as consistent video editing. Deformable Sprites are a type of video auto-encoder model that is optimized on individual videos, and does not require training on a large dataset, nor does it rely on pretrained models. Moreover, our method does not require object masks or other user input, and discovers moving objects of a wider variety than previous work. We evaluate our approach on standard video datasets and show qualitative results on a diverse array of Internet videos.

----

## [263] Eigencontours: Novel Contour Descriptors Based on Low-Rank Approximation

**Authors**: *Wonhui Park, Dongkwon Jin, Chang-Su Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00269](https://doi.org/10.1109/CVPR52688.2022.00269)

**Abstract**:

Novel contour descriptors, called eigencontours, based on low-rank approximation are proposed in this paper. First, we construct a contour matrix containing all object boundaries in a training set. Second, we decompose the contour matrix into eigencontours via the best rank-M approximation. Third, we represent an object boundary by a linear combination of the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$M$</tex>
 eigencontours. We also incorporate the eigencontours into an instance segmentation framework. Experimental results demonstrate that the proposed eigencontours can represent object boundaries more effectively and more efficiently than existing descriptors in a low-dimensional space. Furthermore, the proposed algorithm yields meaningful performances on instance segmentation datasets.

----

## [264] Robust and Accurate Superquadric Recovery: a Probabilistic Approach

**Authors**: *Weixiao Liu, Yuwei Wu, Sipu Ruan, Gregory S. Chirikjian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00270](https://doi.org/10.1109/CVPR52688.2022.00270)

**Abstract**:

Interpreting objects with basic geometric primitives has long been studied in computer vision. Among geometric primitives, superquadrics are well known for their ability to represent a wide range of shapes with few parameters. However, as the first and foremost step, recovering superquadrics accurately and robustly from 3D data still remains challenging. The existing methods are subject to local optima and sensitive to noise and outliers in real-world scenarios, resulting in frequent failure in capturing geometric shapes. In this paper, we propose the first probabilistic method to recover superquadrics from point clouds. Our method builds a Gaussian-uniform mixture model (GUM) on the parametric surface of a superquadric, which explicitly models the generation of outliers and noise. The superquadric recovery is formulated as a Maximum Likelihood Estimation (MLE) problem. We propose an algorithm, Expectation, Maximization, and Switching (EMS), to solve this problem, where: (1) outliers are predicted from the posterior perspective; (2) the superquadric parameter is optimized by the trust-region reflective algorithm; and (3) local optima are avoided by globally searching and switching among parameters encoding similar superquadrics. We show that our method can be extended to the multi-superquadrics recovery for complex objects. The proposed method outperforms the state-of-the-art in terms of accuracy, efficiency, and robustness on both synthetic and real-world datasets. The code is at http://github.com/bmlklwx/EMS-superquadric_fitting.git.

----

## [265] Medial Spectral Coordinates for 3D Shape Analysis

**Authors**: *Morteza Rezanejad, Mohammad Khodadad, Hamidreza Mahyar, Herve Lombaert, Michael Gruninger, Dirk B. Walther, Kaleem Siddiqi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00271](https://doi.org/10.1109/CVPR52688.2022.00271)

**Abstract**:

In recent years there has been a resurgence of interest in our community in the shape analysis of 3D objects repre-sented by surface meshes, their voxelized interiors, or surface point clouds. In part, this interest has been stimulated by the increased availability of RGBD cameras, and by applications of computer vision to autonomous driving, medical imaging, and robotics. In these settings, spectral co-ordinates have shown promise for shape representation due to their ability to incorporate both local and global shape properties in a manner that is qualitatively invariant to iso-metric transformations. Yet, surprisingly, such coordinates have thus far typically considered only local surface positional or derivative information. In the present article, we propose to equip spectral coordinates with medial (object width) information, so as to enrich them. The key idea is to couple surface points that share a medial ball, via the weights of the adjacency matrix. We develop a spectral feature using this idea, and the algorithms to compute it. The incorporation of object width and medial coupling has direct benefits, as illustrated by our experiments on object classification, object part segmentation, and surface point correspondence.

----

## [266] Scribble-Supervised LiDAR Semantic Segmentation

**Authors**: *Ozan Unal, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00272](https://doi.org/10.1109/CVPR52688.2022.00272)

**Abstract**:

Densely annotating LiDAR point clouds remains too expensive and time-consuming to keep up with the ever growing volume of data. While current literature focuses on fully-supervised performance, developing efficient methods that take advantage of realistic weak supervision have yet to be explored. In this paper, we propose using scribbles to annotate LiDAR point clouds and release ScribbleKITTI, the first scribble-annotated dataset for LiDAR semantic segmentation. Furthermore, we present a pipeline to reduce the performance gap that arises when using such weak annotations. Our pipeline comprises of three stand-alone contributions that can be combined with any LiDAR semantic segmentation model to achieve up to 95.7% of the fully-supervised performance while using only 8% labeled points. Our scribble annotations and code are available at github.com/ouenal/scribblekitti.

----

## [267] SoftGroup for 3D Instance Segmentation on Point Clouds

**Authors**: *Thang Vu, Kookhoi Kim, Tung Minh Luu, Thanh Xuan Nguyen, Chang D. Yoo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00273](https://doi.org/10.1109/CVPR52688.2022.00273)

**Abstract**:

Existing state-of-the-art 3D instance segmentation methods perform semantic segmentation followed by grouping. The hard predictions are made when performing semantic segmentation such that each point is associated with a single class. However, the errors stemming from hard decision propagate into grouping that results in (1) low overlaps between the predicted instance with the ground truth and (2) substantial false positives. To address the aforementioned problems, this paper proposes a 3D instance segmentation method referred to as SoftGroup by performing bottom-up soft grouping followed by top-down refinement. SoftGroup allows each point to be associated with multiple classes to mitigate the problems stemming from semantic prediction errors and suppresses false positive instances by learning to categorize them as background. Experimental results on different datasets and multiple evaluation metrics demonstrate the efficacy of SoftGroup. Its performance surpasses the strongest prior method by a significant margin of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$+6.2\%$</tex>
 on the ScanNet v2 hidden test set and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$+6.8\%$</tex>
 on S3DIS Area 5 in terms of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AP_{50}$</tex>
. Soft-Group is also fast, running at 345ms per scan with a sin-gle Titan X on ScanNet v2 dataset. The source code and trained models for both datasets are available at https://github.com/thangvubk/SoftGroup.git.

----

## [268] Accurate 3D Body Shape Regression using Metric and Semantic Attributes

**Authors**: *Vasileios Choutas, Lea Müller, Chun-Hao P. Huang, Siyu Tang, Dimitrios Tzionas, Michael J. Black*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00274](https://doi.org/10.1109/CVPR52688.2022.00274)

**Abstract**:

While methods that regress 3D human meshes from images have progressed rapidly, the estimated body shapes often do not capture the true human shape. This is problematic since, for many applications, accurate body shape is as important as pose. The key reason that body shape accuracy lags pose accuracy is the lack of data. While humans can label 2D joints, and these constrain 3D pose, it is not so easy to “label” 3D body shape. Since paired data with images and 3D body shape are rare, we exploit two sources of information: (1) we collect internet images of diverse “fashion” models together with a small set of anthropometric measurements; (2) we collect linguistic shape attributes for a wide range of 3D body meshes and the model images. Taken together, these datasets provide sufficient constraints to infer dense 3D shape. We exploit the anthropometric measurements and linguistic shape attributes in several novel ways to train a neural network, called SHAPY, that regresses 3D human pose and shape from an RGB image. We evaluate SHAPY on public benchmarks, but note that they either lack significant body shape variation, ground-truth shape, or clothing variation. Thus, we collect a new dataset for evaluating 3D human shape estimation, called HBW, containing photos of “Human Bodies in the Wild” for which we have ground-truth 3D body scans. On this new benchmark, SHAPY significantly outperforms state-of-the-art methods on the task of 3D body shape estimation. This is the first demonstration that 3D body shape regression from images can be trained from easy-to-obtain anthropometric measurements and linguistic shape attributes. Our model and data are available at: shapy.is.tue.mpg.de

----

## [269] JIFF: Jointly-aligned Implicit Face Function for High Quality Single View Clothed Human Reconstruction

**Authors**: *Yukang Cao, Guanying Chen, Kai Han, Wenqi Yang, Kwan-Yee K. Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00275](https://doi.org/10.1109/CVPR52688.2022.00275)

**Abstract**:

This paper addresses the problem of single view 3D human reconstruction. Recent implicit function based methods have shown impressive results, but they fail to recover fine face details in their reconstructions. This largely degrades user experience in applications like 3D telepresence. In this paper, we focus on improving the quality of face in the reconstruction and propose a novel Jointly-aligned Implicit Face Function (JIFF) that combines the merits of the implicit function based approach and model based approach. We employ a 3D morphable face model as our shape prior and compute space-aligned 3D features that capture detailed face geometry information. Such space-aligned 3D features are combined with pixel-aligned 2D features to jointly predict an implicit face function for high quality face reconstruction. We further extend our pipeline and introduce a coarse-to-fine architecture to predict high quality texture for our detailed face model. Extensive evaluations have been carried out on public datasets and our proposed JIFF has demonstrates superior performance (both quantitatively and qualitatively) over existing state-of-the-arts.

----

## [270] Tracking People by Predicting 3D Appearance, Location and Pose

**Authors**: *Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Jitendra Malik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00276](https://doi.org/10.1109/CVPR52688.2022.00276)

**Abstract**:

We present an approach for tracking people in monocular videos by predicting their future 3D representations. To achieve this, we first lift people to 3D from a single frame in a robust manner. This lifting includes information about the 3D pose of the person, their location in the 3D space, and the 3D appearance. As we track a person, we collect 3D observations over time in a tracklet representation. Given the 3D nature of our observations, we build temporal models for each one of the previous attributes. We use these models to predict the future state of the tracklet, including 3D appearance, 3D location, and 3D pose. For a future frame, we compute the similarity between the predicted state of a tracklet and the single frame observations in a probabilistic manner. Association is solved with simple Hungarian matching, and the matches are used to update the respective tracklets. We evaluate our approach on various benchmarks and report state-of-the-art results. Code and models are available at: https://brjathu.github.io/PHALP.

----

## [271] ArtiBoost: Boosting Articulated 3D Hand-Object Pose Estimation via Online Exploration and Synthesis

**Authors**: *Lixin Yang, Kailin Li, Xinyu Zhan, Jun Lv, Wenqiang Xu, Jiefeng Li, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00277](https://doi.org/10.1109/CVPR52688.2022.00277)

**Abstract**:

Estimating the articulated 3D hand-object pose from a single RGB image is a highly ambiguous and challenging problem, requiring large-scale datasets that contain diverse hand poses, object types, and camera viewpoints. Most real-world datasets lack these diversities. In contrast, data synthesis can easily ensure those diversities separately. However, constructing both valid and diverse hand-object interactions and efficiently learning from the vast synthetic data is still challenging. To address the above issues, we propose ArtiBoost, a lightweight online data enhancement method. ArtiBoost can cover diverse hand-object poses and camera viewpoints through sampling in a Composited hand-object Configuration and View-point space (CCV-space) and can adaptively enrich the current hard-discernable items by loss-feedback and sample re-weighting. ArtiBoost alternatively performs data exploration and synthesis within a learning pipeline, and those synthetic data are blended into real-world source data for training. We apply ArtiBoost on a simple learning baseline network and witness the performance boost on several hand-object benchmarks. Our models and code are available at https://github.com/lixiny/ArtiBoost.

----

## [272] Interacting Attention Graph for Single Image Two-Hand Reconstruction

**Authors**: *Mengcheng Li, Liang An, Hongwen Zhang, Lianpeng Wu, Feng Chen, Tao Yu, Yebin Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00278](https://doi.org/10.1109/CVPR52688.2022.00278)

**Abstract**:

Graph convolutional network (GCN) has achieved great success in single hand reconstruction task, while interacting two-hand reconstruction by GCN remains unexplored. In this paper, we present Interacting Attention Graph Hand (IntagHand), the first graph convolution based network that reconstructs two interacting hands from a single RGB image. To solve occlusion and interaction challenges of two-hand reconstruction, we introduce two novel attention based modules in each upsampling step of the original GCN. The first module is the pyramid image feature attention (PIFA) module, which utilizes multiresolution features to implicitly obtain vertex-to-image alignment. The second module is the cross hand attention (CHA) module that encodes the coherence of interacting hands by building dense cross-attention between two hand vertices. As a result, our model outperforms all existing two-hand re-construction methods by a large margin on InterHand2.6M benchmark. Moreover, ablation studies verify the effectiveness of both PIFA and CHA modules for improving the reconstruction accuracy. Results on in-the-wild images and live video streams further demonstrate the generalization ability of our network. Our code is available at https://github.com/Dw1010/IntagHand.

----

## [273] 3D human tongue reconstruction from single "in-the-wild" images

**Authors**: *Stylianos Ploumpis, Stylianos Moschoglou, Vasileios Triantafyllou, Stefanos Zafeiriou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00279](https://doi.org/10.1109/CVPR52688.2022.00279)

**Abstract**:

3D face reconstruction from a single image is a task that has garnered increased interest in the Computer Vision community, especially due to its broad use in a number of applications such as realistic 3D avatar creation, pose invariant face recognition and face hallucination. Since the introduction of the 3D Morphable Model in the late 90's, we witnessed an explosion of research aiming at particularly tackling this task. Nevertheless, despite the increasing level of detail in the 3D face reconstructions from single images mainly attributed to deep learning advances, finer and highly deformable components of the face such as the tongue are still absent from all 3D face models in the literature, although being very important for the realness of the 3D avatar representations. In this work we present the first, to the best of our knowledge, end-to-end trainable pipeline that accurately reconstructs the 3D face together with the tongue. Moreover, we make this pipeline robust in “in-the-wild” images by introducing a novel GAN method tailored for 3D tongue surface generation. Finally, we make publicly available to the community the first diverse tongue dataset, consisting of 1,800 raw scans of 700 individuals varying in gender, age, and ethnicity backgrounds**Project url: www.github.com/steliosploumpis/tongue. As we demonstrate in an extensive series of quantitative as well as qualitative experiments, our model proves to be robust and realistically captures the 3D tongue structure, even in adverse “in-the- wild” conditions.

----

## [274] EPro-PnP: Generalized End-to-End Probabilistic Perspective-n-Points for Monocular Object Pose Estimation

**Authors**: *Hansheng Chen, Pichao Wang, Fan Wang, Wei Tian, Lu Xiong, Hao Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00280](https://doi.org/10.1109/CVPR52688.2022.00280)

**Abstract**:

Locating 3D objects from a single RGB image via Perspective-n-Points (PnP) is a long-standing problem in computer vision. Driven by end-to-end deep learning, recent studies suggest interpreting PnP as a differentiable layer, so that 2D-3D point correspondences can be partly learned by backpropagating the gradient w.r.t. object pose. Yet, learning the entire set of unrestricted 2D-3D points from scratch fails to converge with existing approaches, since the deterministic pose is inherently non-differentiable. In this paper, we propose the EPro-PnP a probabilistic PnP layer for general end-to-end pose estimation, which outputs a distribution of pose on the SE(3) manifold, essentially bringing categorical Softmax to the continuous domain. The 2D-3D coordinates and corresponding weights are treated as intermediate variables learned by minimizing the KL divergence between the predicted and target pose distribution. The underlying principle unifies the existing approaches and resembles the attention mechanism. EPro-PnP significantly outperforms competitive baselines, closing the gap between PnP-based method and the task-specific leaders on the LineMOD 6DoF pose estimation and nuScenes 3D object detection benchmarks.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>

----

## [275] Diversity Matters: Fully Exploiting Depth Clues for Reliable Monocular 3D Object Detection

**Authors**: *Zhuoling Li, Zhan Qu, Yang Zhou, Jianzhuang Liu, Haoqian Wang, Lihui Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00281](https://doi.org/10.1109/CVPR52688.2022.00281)

**Abstract**:

As an inherently ill-posed problem, depth estimation from single images is the most challenging part of monocular 3D object detection (M3OD). Many existing methods rely on preconceived assumptions to bridge the missing spatial information in monocular images, and predict a sole depth value for every object of interest. However, these assumptions do not always hold in practical applications. To tackle this problem, we propose a depth solving system that fully explores the visual clues from the subtasks in M3OD and generates multiple estimations for the depth of each target. Since the depth estimations rely on different assumptions in essence, they present diverse distributions. Even if some assumptions collapse, the estimations established on the remaining assumptions are still reliable. In addition, we develop a depth selection and combination strategy. This strategy is able to remove abnormal estimations caused by collapsed assumptions, and adaptively combine the remaining estimations into a single one. In this way, our depth solving system becomes more precise and robust. Exploiting the clues from multiple subtasks of M3OD and without introducing any extra information, our method surpasses the current best method by more than 20% relatively on the Moderate level of test split in the KITTI 3D object detection benchmark, while still maintaining real-time efficiency.

----

## [276] OmniFusion: 360 Monocular Depth Estimation via Geometry-Aware Fusion

**Authors**: *Yuyan Li, Yuliang Guo, Zhixin Yan, Xinyu Huang, Ye Duan, Liu Ren*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00282](https://doi.org/10.1109/CVPR52688.2022.00282)

**Abstract**:

A well-known challenge in applying deep-learning methods to omnidirectional images is spherical distortion. In dense regression tasks such as depth estimation, where structural details are required, using a vanilla CNN layer on the distorted 360 image results in undesired information loss. In this paper, we propose a 360 monocular depth estimation pipeline, OmniFusion, to tackle the spherical distortion issue. Our pipeline transforms a 360 image into less-distorted perspective patches (i.e. tangent images) to obtain patch-wise predictions via CNN, and then merge the patch-wise results for final output. To handle the discrepancy between patch-wise predictions which is a major issue affecting the merging quality, we propose a new framework with the following key components. First, we propose a geometry-aware feature fusion mechanism that combines 3D geometric features with 2D image features to compensate for the patch-wise discrepancy. Second, we employ the self-attention-based transformer architecture to conduct a global aggregation of patch-wise information, which further improves the consistency. Last, we introduce an iterative depth refinement mechanism, to further refine the estimated depth based on the more accurate geometric features. Experiments show that our method greatly mitigates the distortion issue, and achieves state-of-the-art performances on several 360 monocular depth estimation benchmark datasets. Our code is available at https://github.com/yuyanli0831/OmniFusion.

----

## [277] Gated2Gated: Self-Supervised Depth Estimation from Gated Images

**Authors**: *Amanpreet Walia, Stefanie Walz, Mario Bijelic, Fahim Mannan, Frank D. Julca-Aguilar, Michael S. Langer, Werner Ritter, Felix Heide*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00283](https://doi.org/10.1109/CVPR52688.2022.00283)

**Abstract**:

Gated cameras hold promise as an alternative to scanning LiDAR sensors with high-resolution 3D depth that is robust to back-scatter in fog, snow, and rain. Instead of sequentially scanning a scene and directly recording depth via the photon time-of-flight, as in pulsed LiDAR sensors, gated imagers encode depth in the relative intensity of a handful of gated slices, captured at megapixel resolution. Although existing methods have shown that it is possible to decode high-resolution depth from such measurements, these methods require synchronized and calibrated LiDAR to supervise the gated depth decoder - prohibiting fast adoption across geographies, training on large unpaired datasets, and exploring alternative applications outside of automotive use cases. In this work, propose an entirely self-supervised depth estimation method that uses gated in-tensity profiles and temporal consistency as a training signal. The proposed model is trained end-to-end from gated video sequences, does not require LiDAR or RGB data, and learns to estimate absolute depth values. We take gated slices as input and disentangle the estimation of the scene albedo, depth, and ambient light, which are then used to learn to reconstruct the input slices through a cyclic loss. We rely on temporal consistency between a given frame and neighboring gated slices to estimate depth in regions with shadows and reflections. We experimentally validate that the proposed approach outperforms existing super-vised and self-supervised depth estimation methods based on monocular RGB and stereo images, as well as super-vised methods based on gated images. Code is available at https://github.com/princeton-computational-imaging/Gated2Gated.

----

## [278] IRISformer: Dense Vision Transformers for Single-Image Inverse Rendering in Indoor Scenes

**Authors**: *Rui Zhu, Zhengqin Li, Janarbek Matai, Fatih Porikli, Manmohan Chandraker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00284](https://doi.org/10.1109/CVPR52688.2022.00284)

**Abstract**:

Indoor scenes exhibit significant appearance variations due to myriad interactions between arbitrarily diverse object shapes, spatially-changing materials, and complex lighting. Shadows, highlights, and inter-reflections caused by visible and invisible light sources require reasoning about long-range interactions for inverse rendering, which seeks to recover the components of image formation, namely, shape, material, and lighting. In this work, our intuition is that the long-range attention learned by transformer architectures is ideally suited to solve longstanding challenges in single-image inverse rendering. We demonstrate with a specific instantiation of a dense vision transformer, IRISformer, that excels at both single-task and multi-task reasoning required for inverse rendering. Specifically, we propose a transformer architecture to simultaneously estimate depths, normals, spatially-varying albedo, roughness and lighting from a single image of an indoor scene. Our extensive evaluations on benchmark datasets demonstrate state-of-the-art results on each of the above tasks, enabling applications like object insertion and material editing in a single unconstrained real image, with greater photorealism than prior works. Code and data are publicly released.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/ViLab-UCSD/IRISformer

----

## [279] Egocentric Scene Understanding via Multimodal Spatial Rectifier

**Authors**: *Tien Do, Khiem Vuong, Hyun Soo Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00285](https://doi.org/10.1109/CVPR52688.2022.00285)

**Abstract**:

In this paper, we study a problem of egocentric scene understanding, i.e., predicting depths and surface normals from an egocentric image. Egocentric scene understanding poses unprecedented challenges: (1) due to large head movements, the images are taken from non-canonical viewpoints (i.e., tilted images) where existing models of geometry prediction do not apply; (2) dynamic foreground objects including hands constitute a large proportion of visual scenes. These challenges limit the performance of the existing models learned from large indoor datasets, such as ScanNet [6] and NYUv2 [36], which comprise predominantly upright images of static scenes. We present a multimodal spatial rectifier that stabilizes the egocentric images to a set of reference directions, which allows learning a coherent visual representation. Unlike unimodal spatial rectifier that often produces excessive perspective warp for egocentric images, the multimodal spatial rectifier learns from multiple directions that can minimize the impact of the perspective warp. To learn visual representations of the dynamic foreground objects, we present a new dataset called EDINA (Egocentric Depth on everyday INdoor Activities) that comprises more than 500K synchronized RGBD frames and gravity directions. Equipped with the multimodal spatial rectifier and the EDINA dataset, our proposed method on single-view depth and surface normal estimation significantly outperforms the baselines not only on our ED-INA dataset, but also on other popular egocentric datasets, such as First Person Hand Action (FPHA) [18] and EPIC-KITCHENS [7].

----

## [280] Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry

**Authors**: *Gwangbin Bae, Ignas Budvytis, Roberto Cipolla*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00286](https://doi.org/10.1109/CVPR52688.2022.00286)

**Abstract**:

Multi-view depth estimation methods typically require the computation of a multi-view cost-volume, which leads to huge memory consumption and slow inference. Furthermore, multi-view matching can fail for texture-less surfaces, reflective surfaces and moving objects. For such failure modes, single-view depth estimation methods are often more reliable. To this end, we propose MaGNet, a novel framework for fusing single-view depth probability with multi-view geometry, to improve the accuracy, robustness and efficiency of multi-view depth estimation. For each frame, MaGNet estimates a single-view depth probability distribution, parameterized as a pixel-wise Gaussian. The distribution estimated for the reference frame is then used to sample per-pixel depth candidates. Such probabilistic sampling enables the network to achieve higher accuracy while evaluating fewer depth candidates. We also propose depth consistency weighting for the multi-view matching score, to ensure that the multi-view depth is consistent with the single-view predictions. The proposed method achieves state-of-the-art performance on ScanNet [8], 7- Scenes [38] and KITTI [15]. Qualitative evaluation demonstrates that our method is more robust against challenging artifacts such as texture-less/reflective surfaces and moving objects. Our code and model weights are available at https://github.com/baegwangbin/MaGNet.

----

## [281] The Implicit Values of A Good Hand Shake: Handheld Multi-Frame Neural Depth Refinement

**Authors**: *Ilya Chugunov, Yuxuan Zhang, Zhihao Xia, Xuaner Zhang, Jiawen Chen, Felix Heide*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00287](https://doi.org/10.1109/CVPR52688.2022.00287)

**Abstract**:

Modern smartphones can continuously stream multi-megapixel RGB images at 60 Hz, synchronized with high-quality 3D pose information and low-resolution LiDAR-driven depth estimates. During a snapshot photograph, the natural unsteadiness of the photographer's hands offers millimeter-scale variation in camera pose, which we can capture along with RGB and depth in a circular buffer. In this work we explore how, from a bundle of these measurements acquired during viewfinding, we can combine dense micro-baseline parallax cues with kilopixel LiDAR depth to distill a high-fidelity depth map. We take a test-time optimization approach and train a coordinate MLP to output photometrically and geometrically consistent depth estimates at the continuous coordinates along the path traced by the photographer's natural hand shake. With no additional hardware, artificial hand motion, or user interaction beyond the press of a button, our proposed method brings high-resolution depth estimates to point-and-shoot “table-top” photography – textured objects at close range.

----

## [282] BANMo: Building Animatable 3D Neural Models from Many Casual Videos

**Authors**: *Gengshan Yang, Minh Vo, Natalia Neverova, Deva Ramanan, Andrea Vedaldi, Hanbyul Joo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00288](https://doi.org/10.1109/CVPR52688.2022.00288)

**Abstract**:

Prior work for articulated 3D shape reconstruction often relies on specialized multi-view and depth sensors or pre-built deformable 3D models. Such methods do not scale to diverse sets of objects in the wild. We present a method that requires neither of them. It aims to create high-fidelity, articulated 3D models from many casual RGB videos in a differentiable rendering framework. Our key in-sight is to merge three schools of thought: (1) classic deformable shape models that make use of articulated bones and blend skinning, (2) canonical embeddings that establish correspondences between pixels and a canonical 3D model, and (3) volumetric neural radiance fields (NeRFs) that are amenable to gradient-based optimization. We introduce neural blend skinning models that allow for differentiable and invertible articulated deformations. When combined with canonical embeddings, such models allow us to establish dense correspondences across videos that can be self-supervised with cycle consistency. On real and synthetic datasets, our method shows higher-fidelity 3D reconstructions than prior works for humans and animals, with the ability to render realistic images from novel viewpoints. Project page: https://banmo-www.github.io/.

----

## [283] Self-supervised Video Transformer

**Authors**: *Kanchana Ranasinghe, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan, Michael S. Ryoo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00289](https://doi.org/10.1109/CVPR52688.2022.00289)

**Abstract**:

In this paper, we propose self-supervised training for video transformers using unlabeled video data. From a given video, we create local and global spatiotemporal views with varying spatial sizes and frame rates. Our self-supervised objective seeks to match the features of these different views representing the same video, to be invariant to spatiotemporal variations in actions. To the best of our knowledge, the proposed approach is the first to alleviate the dependency on negative samples or dedicated memory banks in Self-supervised Video Transformer (SVT). Further, owing to the flexibility of Transformer models, SVT supports slow-fast video processing within a single architecture using dynamically adjusted positional encoding and supports longterm relationship modeling along spatiotemporal dimensions. Our approach performs well on four action recognition benchmarks (Kinetics-400, UCF-101, HMDB-51, and SSv2) and converges faster with small batch sizes. Code is available at: https://git.io/J1juJ.

----

## [284] Temporally Efficient Vision Transformer for Video Instance Segmentation

**Authors**: *Shusheng Yang, Xinggang Wang, Yu Li, Yuxin Fang, Jiemin Fang, Wenyu Liu, Xun Zhao, Ying Shan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00290](https://doi.org/10.1109/CVPR52688.2022.00290)

**Abstract**:

Recently vision transformer has achieved tremendous success on image-level visual recognition tasks. To effectively and efficiently model the crucial temporal information within a video clip, we propose a Temporally Efficient Vision Transformer (TeViT) for video instance segmentation (VIS). Different from previous transformer-based VIS methods, TeViT is nearly convolution-free, which contains a transformer backbone and a query-based video instance segmentation head. In the backbone stage, we propose a nearly parameter-free messenger shift mechanism for early temporal context fusion. In the head stages, we propose a parameter-shared spatiotemporal query interaction mechanism to build the one-to-one correspondence between video instances and queries. Thus, TeViT fully utilizes both frame-level and instance-level temporal context information and obtains strong temporal modeling capacity with negligible extra computational cost. On three widely adopted VIS benchmarks, i.e., YouTube-VIS-2019, YouTube-VIS-2021, and OVIS, TeViT obtains state-of-the-art results and maintains high inference speed, e.g., 46.6 AP with 68.9 FPS on YouTube-VIS-2019. Code is available at https://github.com/hustvl/TeViT.

----

## [285] VISOLO: Grid-Based Space-Time Aggregation for Efficient Online Video Instance Segmentation

**Authors**: *Su Ho Han, Sukjun Hwang, Seoung Wug Oh, Yeonchool Park, Hyunwoo Kim, Min-Jung Kim, Seon Joo Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00291](https://doi.org/10.1109/CVPR52688.2022.00291)

**Abstract**:

For online video instance segmentation (VIS), fully utilizing the information from previous frames in an efficient manner is essential for real-time applications. Most previous methods follow a two-stage approach requiring additional computations such as RPN and RoIAlign, and do not fully exploit the available information in the video for all subtasks in VIS. In this paper, we propose a novel single-stage framework for online VIS built based on the grid structured feature representation. The grid-based features allow us to employ fully convolutional networks for real-time processing, and also to easily reuse and share features within different components. We also introduce cooperatively operating modules that aggregate information from available frames, in order to enrich the features for all subtasks in VIS. Our design fully takes advantage of previous information in a grid form for all tasks in VIS in an efficient way, and we achieved the new state-of-the-art accuracy (38.6 AP and 36.9 AP) and speed (40.0 FPS) on YouTube-VIS 2019 and 2021 datasets among online VIS methods. The code is available at https://github.com/SuHoHan95/VISOLQ

----

## [286] Temporal Alignment Networks for Long-term Video

**Authors**: *Tengda Han, Weidi Xie, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00292](https://doi.org/10.1109/CVPR52688.2022.00292)

**Abstract**:

The objective of this paper is a temporal alignment network that ingests long term video sequences, and associated text sentences, in order to: (1) determine if a sentence is alignable with the video; and (2) if it is alignable, then determine its alignment. The challenge is to train such networks from large-scale datasets, such as HowTo100M, where the associated text sentences have significant noise, and are only weakly aligned when relevant. Apart from proposing the alignment network, we also make four contributions: (i) we describe a novel co-training method that enables to denoise and train on raw instructional videos without using manual annotation, de-spite the considerable noise; (ii) to benchmark the align-ment performance, we manually curate a 10-hour subset of HowTo100M, totalling 80 videos, with sparse temporal de-scriptions. Our proposed model, trained on HowTo100M, outperforms strong baselines (CLIP, MIL-NCE) on this alignment dataset by a significant margin; (iii) we ap-ply the trained model in the zero-shot settings to mul-tiple downstream video understanding tasks and achieve state-of-the-art results, including text-video retrieval on YouCook2, and weakly supervised video action segmentation on Breakfast-Action. (iv) we use the automatically-aligned HowTo100M annotations for end-to-end finetuning of the backbone model, and obtain improved performance on downstream action recognition tasks.

----

## [287] Revisiting the "Video" in Video-Language Understanding

**Authors**: *Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, Juan Carlos Niebles*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00293](https://doi.org/10.1109/CVPR52688.2022.00293)

**Abstract**:

What makes a video task uniquely suited for videos, beyond what can be understood from a single image? Building on recent progress in self-supervised image-language models, we revisit this question in the context of video and language tasks. We propose the atemporal probe (ATP), a new model for video-language analysis which provides a stronger bound on the baseline accuracy of multimodal models constrained by image-level understanding. By applying this model to standard discriminative video and language tasks, such as video question answering and text-to-video retrieval, we characterize the limitations and potential of current video-language benchmarks. We find that understanding of event temporality is often not necessary to achieve strong or state-of-the-art performance, even compared with recent large-scale video-language models and in contexts intended to benchmark deeper video-level understanding. We also demonstrate how ATP can improve both video-language dataset and model design. We describe a technique for leveraging ATP to better disentangle dataset subsets with a higher concentration of temporally challenging data, improving benchmarking efficacy for causal and temporal understanding. Further, we show that effectively integrating ATP into full video-level temporal models can improve efficiency and state-of-the-art accuracy. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project website: https://stanfordvl.github.io/atp-revisit-video-lang/

----

## [288] Invariant Grounding for Video Question Answering

**Authors**: *Yicong Li, Xiang Wang, Junbin Xiao, Wei Ji, Tat-Seng Chua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00294](https://doi.org/10.1109/CVPR52688.2022.00294)

**Abstract**:

Video Question Answering (VideoQA) is the task of an-swering questions about a video. At its core is understanding the alignments between visual scenes in video and linguistic semantics in question to yield the answer. In leading VideoQA models, the typical learning objective, empirical risk minimization (ERM), latches on superficial correlations between video-question pairs and answers as the alignments. However, ERM can be problematic, because it tends to over-exploit the spurious correlations between question-irrelevant scenes and answers, instead of inspecting the causal effect of question-critical scenes. As a result, the VideoQA models suffer from unreliable reasoning. In this work, we first take a causal look at VideoQA and argue that invariant grounding is the key to ruling out the spurious correlations. Towards this end, we propose a new learning framework, Invariant Grounding for VideoQA (IGV), to ground the question-critical scene, whose causal relations with answers are invariant across different interventions on the complement. With IGV, the VideoQA mod-els are forced to shield the answering process from the negative influence of spurious correlations, which significantly improves the reasoning ability. Experiments on three benchmark datasets validate the superiority of IGV in terms of accuracy, visual explainability, and generalization ability over the leading baselines. Our code is available at https://github.com/y13800/IGV.

----

## [289] P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision

**Authors**: *He Zhao, Isma Hadji, Nikita Dvornik, Konstantinos G. Derpanis, Richard P. Wildes, Allan D. Jepson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00295](https://doi.org/10.1109/CVPR52688.2022.00295)

**Abstract**:

In this paper, we study the problem of procedure planning in instructional videos. Here, an agent must produce a plausible sequence of actions that can transform the environment from a given start to a desired goal state. When learning procedure planning from instructional videos, most recent work leverages intermediate visual observations as supervision, which requires expensive annotation efforts to localize precisely all the instructional steps in training videos. In contrast, we remove the need for expensive temporal video annotations and propose a weakly supervised approach by learning from natural language instructions. Our model is based on a transformer equipped with a memory module, which maps the start and goal observations to a sequence of plausible actions. Furthermore, we augment our model with a probabilistic generative module to capture the uncertainty inherent to procedure planning, an aspect largely overlooked by previous work. We evaluate our model on three datasets and show our weakly-supervised approach outperforms previous fully supervised state-of-the-art models on multiple metrics.

----

## [290] FineDiving: A Fine-grained Dataset for Procedure-aware Action Quality Assessment

**Authors**: *Jinglin Xu, Yongming Rao, Xumin Yu, Guangyi Chen, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00296](https://doi.org/10.1109/CVPR52688.2022.00296)

**Abstract**:

Most existing action quality assessment methods rely on the deep features of an entire video to predict the score, which is less reliable due to the non-transparent inference process and poor interpretability. We argue that understanding both high-level semantics and internal temporal structures of actions in competitive sports videos is the key to making predictions accurate and interpretable. Towards this goal, we construct a new fine-grained dataset, called FineDiving, developed on diverse diving events with detailed annotations on action procedures. We also propose a procedure-aware approach for action quality assessment, learned by a new Temporal Segmentation Attention module. Specifically, we propose to parse pairwise query and exemplar action instances into consecutive steps with diverse semantic and temporal correspondences. The procedure-aware cross-attention is proposed to learn embeddings between query and exemplar steps to discover their semantic, spatial, and temporal correspondences, and further serve for fine-grained contrastive regression to derive a reliable scoring mechanism. Extensive experiments demonstrate that our approach achieves substantial improvements over the state-of-the-art methods with better interpretability. The dataset and code are available at https://github.com/xujinglin/FineDiving.

----

## [291] Cross-Model Pseudo-Labeling for Semi-Supervised Action Recognition

**Authors**: *Yinghao Xu, Fangyun Wei, Xiao Sun, Ceyuan Yang, Yujun Shen, Bo Dai, Bolei Zhou, Stephen Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00297](https://doi.org/10.1109/CVPR52688.2022.00297)

**Abstract**:

Semi-supervised action recognition is a challenging but important task due to the high cost of data annotation. A common approach to this problem is to assign unlabeled data with pseudo-labels, which are then used as additional supervision in training. Typically in recent work, the pseudo-labels are obtained by training a model on the labeled data, and then using confident predictions from the model to teach itself. In this work, we propose a more effective pseudo-labeling scheme, called Cross-Model Pseudo-Labeling (CMPL). Concretely, we introduce a lightweight auxiliary network in addition to the primary backbone, and ask them to predict pseudo-labels for each other. We observe that, due to their different structural biases, these two models tend to learn complementary representations from the same video clips. Each model can thus benefit from its counterpart by utilizing cross-model predictions as supervision. Experiments on different data partition protocols demonstrate the significant improvement of our framework over existing alternatives. For example, CMPL achieves 17.6% and 25.1% Top-1 accuracy on Kinetics-400 and UCF-101 using only the RGB modality and 1% labeled data, outperforming our baseline model, FixMatch [17], by 9.0% and 10.3%, respectively. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page is at https://justimyhxu.github.io/projects/cmpl/.

----

## [292] Revisiting Skeleton-based Action Recognition

**Authors**: *Haodong Duan, Yue Zhao, Kai Chen, Dahua Lin, Bo Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00298](https://doi.org/10.1109/CVPR52688.2022.00298)

**Abstract**:

Human skeleton, as a compact representation of human action, has received increasing attention in recent years. Many skeleton-based action recognition methods adopt GCNs to extract features on top of human skeletons. Despite the positive results shown in these attempts, GCN-based methods are subject to limitations in robustness, interoperability, and scalability. In this work, we propose PoseConv3D, a new approach to skeleton-based action recognition. PoseConv3D relies on a 3D heatmap volume instead of a graph sequence as the base representation of human skeletons. Compared to GCN-based methods, PoseConv3D is more effective in learning spatiotemporal features, more robust against pose estimation noises, and generalizes better in cross-dataset settings. Also, PoseConv3D can handle multiple-person scenarios without additional computation costs. The hierarchical features can be easily integrated with other modalities at early fusion stages, providing a great design space to boost the performance. PoseConv3D achieves the state-of-the-art on five of six standard skeleton-based action recognition benchmarks. Once fused with other modalities, it achieves the state-of-the-art on all eight multi-modality action recognition benchmarks. Code has been made available at: https://github.com/kennymckormick/pyskl.

----

## [293] OpenTAL: Towards Open Set Temporal Action Localization

**Authors**: *Wentao Bao, Qi Yu, Yu Kong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00299](https://doi.org/10.1109/CVPR52688.2022.00299)

**Abstract**:

Temporal Action Localization (TAL) has experienced remarkable success under the supervised learning paradigm. However, existing TAL methods are rooted in the closed set assumption, which cannot handle the inevitable unknown actions in open-world scenarios. In this paper, we, for the first time, step toward the Open Set TAL (OSTAL) problem and propose a general framework Open TAL based on Evidential Deep Learning (EDL). Specifically, the OpenTAL consists of uncertainty-aware action classification, actionness prediction, and temporal location regression. With the proposed importance-balanced EDL method, classification uncertainty is learned by collecting categorical evidence majorly from important samples. To distinguish the unknown actions from background video frames, the actionness is learned by the positive-unlabeled learning. The classification uncertainty is further calibrated by leveraging the guidance from the temporal localization quality. The OpenTAL is general to enable existing TAL models for open set scenarios, and experimental results on THUMOS14 and ActivityNet1.3 benchmarks show the effectiveness of our method. The code and pre-trained models are released at https://www.rit.edu/actionlab/opental.

----

## [294] Dual-AI: Dual-path Actor Interaction Learning for Group Activity Recognition

**Authors**: *Mingfei Han, David Junhao Zhang, Yali Wang, Rui Yan, Lina Yao, Xiaojun Chang, Yu Qiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00300](https://doi.org/10.1109/CVPR52688.2022.00300)

**Abstract**:

Learning spatial-temporal relation among multiple actors is crucial for group activity recognition. Different group activities often show the diversified interactions between actors in the video. Hence, it is often difficult to model complex group activities from a single view of spatial-temporal actor evolution. To tackle this problem, we propose a distinct Dual-path Actor Interaction (Dual-AI) framework, which flexibly arranges spatial and temporal transformers in two complementary orders, enhancing actor relations by integrating merits from different spatio-temporal paths. Moreover, we introduce a novel Multi-scale Actor Contrastive Loss (MAC-Loss) between two interactive paths of Dual-AI. Via self-supervised actor consistency in both frame and video levels, MAC-Loss can effectively distinguish individual actor representations to reduce action confusion among different actors. Consequently, our Dual-AI can boost group activity recognition by fusing such discriminative features of different actors. To evaluate the proposed approach, we conduct extensive experiments on the widely used benchmarks, including Volleyball [21], Collective Activity [II], and NBA datasets [49]. The proposed Dual-AI achieves state-of-the-art performance on all these datasets. It is worth noting the proposed Dual-AI with 50% training data outperforms a number of recent approaches with 100% training data. This confirms the generalization power of Dual-AI for group activity recognition, even under the challenging scenarios of limited supervision.

----

## [295] TransRank: Self-supervised Video Representation Learning via Ranking-based Transformation Recognition

**Authors**: *Haodong Duan, Nanxuan Zhao, Kai Chen, Dahua Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00301](https://doi.org/10.1109/CVPR52688.2022.00301)

**Abstract**:

Recognizing transformation types applied to a video clip (RecogTrans) is a long-established paradigm for selfsupervised video representation learning, which achieves much inferior performance compared to instance discrimination approaches (InstDisc) in recent works. However, based on a thorough comparison of representative Recog-Trans and InstDisc methods, we observe the great potential of RecogTrans on both semantic-related and temporalrelated downstream tasks. Based on hard-label classification, existing RecogTrans approaches suffer from noisy supervision signals in pre-training. To mitigate this problem, we developed TransRank, a unified framework for recognizing Transformations in a Ranking formulation. TransRank provides accurate supervision signals by recognizing transformations relatively, consistently outperforming the classification-based formulation. Meanwhile, the unified framework can be instantiated with an arbitrary set of temporal or spatial transformations, demonstrating good generality. With a ranking-based formulation and several empirical practices, we achieve competitive performance on video retrieval and action recognition. Under the same setting, TransRank surpasses the previous state-of-the-art method [28] by 6.4% on UCF101 and 8.3% on HMDB51 for action recognition (Topl Acc); improves video retrieval on UCF101 by 20.4% (R@1). The promising results validate that RecogTrans is still a worth exploring paradigm for video self-supervised learning. Codes will be released at https://github.com/kennymckormick/TransRank.

----

## [296] Revealing Occlusions with 4D Neural Fields

**Authors**: *Basile Van Hoorick, Purva Tendulkar, Dídac Surís, Dennis Park, Simon Stent, Carl Vondrick*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00302](https://doi.org/10.1109/CVPR52688.2022.00302)

**Abstract**:

For computer vision systems to operate in dynamic situations, they need to be able to represent and reason about object permanence. We introduce a framework for learning to estimate 4D visual representations from monocular RGB-D video, which is able to persist objects, even once they become obstructed by occlusions. Unlike traditional video representations, we encode point clouds into a continuous representation, which permits the model to attend across the spatiotemporal context to resolve occlusions. On two large video datasets that we release along with this paper, our experiments show that the representation is able to successfully reveal occlusions for several tasks, without any architectural changes. Visualizations show that the attention mechanism automatically learns to follow occluded objects. Since our approach can be trained end-to-end and is easily adaptable, we believe it will be useful for handling occlusions in many video understanding tasks. Data, code, and models are available at occ1usions. cs. co1umbia. edu.

----

## [297] HODOR: High-level Object Descriptors for Object Re-segmentation in Video Learned from Static Images

**Authors**: *Ali Athar, Jonathon Luiten, Alexander Hermans, Deva Ramanan, Bastian Leibe*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00303](https://doi.org/10.1109/CVPR52688.2022.00303)

**Abstract**:

Existing state-of-the-art methods for Video Object Segmentation (VOS) learn low-level pixel-to-pixel correspondences between frames to propagate object masks across video. This requires a large amount of densely annotated video data, which is costly to annotate, and largely redundant since frames within a video are highly correlated. In light of this, we propose HODOR: a novel method that tackles VOS by effectively leveraging annotated static images for understanding object appearance and scene context. We encode object instances and scene information from an image frame into robust high-level descriptors which can then be used to re-segment those objects in different frames. As a result, HODOR achieves state-of-the-art performance on the DAVIS and YouTube-VOS benchmarks compared to existing methods trained without video annotations. With out any architectural modification, HODOR can also learn from video context around single annotated video frames by utilizing cyclic consistency, whereas other methods rely on dense, temporally consistent annotations. Source code: https://github.com/Ali2500/HODOR.

----

## [298] Compositional Temporal Grounding with Structured Variational Cross-Graph Correspondence Learning

**Authors**: *Juncheng Li, Junlin Xie, Long Qian, Linchao Zhu, Siliang Tang, Fei Wu, Yi Yang, Yueting Zhuang, Xin Eric Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00304](https://doi.org/10.1109/CVPR52688.2022.00304)

**Abstract**:

Temporal grounding in videos aims to localize one target video segment that semantically corresponds to a given query sentence. Thanks to the semantic diversity of natural language descriptions, temporal grounding allows activity grounding beyond pre-defined classes and has received increasing attention in recent years. The semantic diversity is rooted in the principle of compositionality in linguistics, where novel semantics can be systematically described by combining known words in novel ways (compositional generalization). However, current temporal grounding datasets do not specifically test for the compositional generalizability. To systematically measure the compositional generalizability of temporal grounding models, we introduce a new Compositional Temporal Grounding task and construct two new dataset splits, i.e., Charades-CG and ActivityNet-CG. Evaluating the state-of-the-art methods on our new dataset splits, we empirically find that they fail to generalize to queries with novel combinations of seen words. To tackle this challenge, we propose a variational cross-graph reasoning framework that explicitly decomposes video and language into multiple structured hierarchies and learns fine-grained semantic correspondence among them. Experiments illustrate the superior compositional generalizability of our approach. The repository of this work is at ht tps: / / gi thub. com/YYJMJC/ Composi tional- Temporal-Grounding.

----

## [299] UMT: Unified Multi-modal Transformers for Joint Video Moment Retrieval and Highlight Detection

**Authors**: *Ye Liu, Siyuan Li, Yang Wu, Chang Wen Chen, Ying Shan, Xiaohu Qie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00305](https://doi.org/10.1109/CVPR52688.2022.00305)

**Abstract**:

Finding relevant moments and highlights in videos according to natural language queries is a natural and highly valuable common need in the current video content explosion era. Nevertheless, jointly conducting moment retrieval and highlight detection is an emerging research topic, even though its component problems and some related tasks have already been studied for a while. In this paper, we present the first unified framework, named Unified Multi-modal Transformers (UMT), capable of realizing such joint optimization while can also be easily degenerated for solving individual problems. As far as we are aware, this is the first scheme to integrate multi-modal (visual-audio) learning for either joint optimization or the individual moment retrieval task, and tackles moment retrieval as a keypoint detection problem using a novel query generator and query decoder. Extensive comparisons with existing methods and ablation studies on QVHighlights, Charades-STA, YouTube Highlights, and TVSum datasets demonstrate the effectiveness, superiority, and flexibility of the proposed method under various settings. Source code and pre-trained models are available at https://github.com/TencentARC/UMT.

----

## [300] Future Transformer for Long-term Action Anticipation

**Authors**: *Dayoung Gong, Joonseok Lee, Manjin Kim, Seong Jong Ha, Minsu Cho*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00306](https://doi.org/10.1109/CVPR52688.2022.00306)

**Abstract**:

The task of predicting future actions from a video is crucial for a real-world agent interacting with others. When anticipating actions in the distant future, we humans typically consider long-term relations over the whole sequence of actions, i.e., not only observed actions in the past but also potential actions in the future. In a similar spirit, we propose an end-to-end attention model for action anticipation, dubbed Future Transformer (FUTR), that leverages global attention over all input frames and output tokens to predict a minutes-long sequence of future actions. Unlike the previous autoregressive models, the proposed method learns to predict the whole sequence of future actions in parallel decoding, enabling more accurate and fast inference for long-term anticipation. We evaluate our method on two standard benchmarks for long-term action anticipation, Breakfast and 50 Salads, achieving state-of-the-art results.

----

## [301] MLP-3D: A MLP-like 3D Architecture with Grouped Time Mixing

**Authors**: *Zhaofan Qiu, Ting Yao, Chong-Wah Ngo, Tao Mei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00307](https://doi.org/10.1109/CVPR52688.2022.00307)

**Abstract**:

Convolutional Neural Networks (CNNs) have been re-garded as the go-to models for visual recognition. More re-cently, convolution-free networks, based on multi-head self-attention (MSA) or multi-layer perceptrons (MLPs), become more and more popular. Nevertheless, it is not trivial when utilizing these newly-minted networks for video recognition due to the large variations and complexities in video data. In this paper, we present MLP-3D networks, a novel MLP-like 3D architecture for video recognition. Specifically, the architecture consists of MLP-3D blocks, where each block contains one MLP applied across tokens (i.e., token-mixing MLP) and one MLP applied independently to each token (i.e., channel MLP). By deriving the novel grouped time mixing (GTM) operations, we equip the basic token-mixing MLP with the ability of temporal modeling. GTM divides the input tokens into several temporal groups and linearly maps the tokens in each group with the shared projection matrix. Furthermore, we devise several variants of GTM with different grouping strategies, and compose each vari-ant in different blocks of MLP-3D network by greedy ar-chitecture search. Without the dependence on convolutions or attention mechanisms, our MLP-3D networks achieves 68.5%/81.4% top-1 accuracy on Something-Something V2 and Kinetics-400 datasets, respectively. Despite with fewer computations, the results are comparable to state-of-the-art widely-used 3D CNNs and video transformers.

----

## [302] Learning Pixel-Level Distinctions for Video Highlight Detection

**Authors**: *Fanyue Wei, Biao Wang, Tiezheng Ge, Yuning Jiang, Wen Li, Lixin Duan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00308](https://doi.org/10.1109/CVPR52688.2022.00308)

**Abstract**:

The goal of video highlight detection is to select the most attractive segments from a long video to depict the most interesting parts of the video. Existing methods typically focus on modeling relationship between different video segments in order to learning a model that can assign highlight scores to these segments; however, these approaches do not explicitly consider the contextual dependency within individual segments. To this end, we propose to learn pixel-level distinctions to improve the video highlight detection. This pixel-level distinction indicates whether or not each pixel in one video belongs to an interesting section. The advantages of modeling such fine-level distinctions are two-fold. First, it allows us to exploit the temporal and spatial relations of the content in one video, since the distinction of a pixel in one frame is highly dependent on both the content before this frame and the content around this pixel in this frame. Second, learning the pixel-level distinction also gives a good explanation to the video highlight task regarding what contents in a highlight segment will be attractive to people. We design an encoder-decoder network to estimate the pixel-level distinction, in which we leverage the 3D convolutional neural networks to exploit the temporal context information, and further take advantage of the visual saliency to model the spatial distinction. State-of-the-art performance on three public benchmarks clearly validates the effectiveness of our framework for video highlight detection.

----

## [303] DRVIC: Decomposition and Reasoning for Video Individual Counting

**Authors**: *Tao Han, Lei Bai, Junyu Gao, Qi Wang, Wanli Ouyang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00309](https://doi.org/10.1109/CVPR52688.2022.00309)

**Abstract**:

Pedestrian counting is a fundamental tool for under-standing pedestrian patterns and crowd flow analysis. Existing works (e.g., image-level pedestrian counting, cross-line crowd counting et al.) either only focus on the image-level counting or are constrained to the manual annotation of lines. In this work, we propose to conduct the pedes-trian counting from a new perspective - Video Individual Counting (VIC), which counts the total number of individual pedestrians in the given video (a person is only counted once). Instead of relying on the Multiple Object Tracking (MOT) techniques, we propose to solve the problem by decomposing all pedestrians into the initial pedestrians who existed in the first frame and the new pedestrians with separate identities in each following frame. Then, an end-to-end Decomposition and Reasoning Network (DRNet) is designed to predict the initial pedestrian count with the density estimation method and reason the new pedestrian's count of each frame with the differentiable optimal transport. Extensive experiments are conducted on two datasets with congested pedestrians and diverse scenes, demonstrating the effectiveness of our method over baselines with great superiority in counting the individual pedestrians. Code: https://github.com/taohan10200/DRNet.

----

## [304] Slot-VPS: Object-centric Representation Learning for Video Panoptic Segmentation

**Authors**: *Yi Zhou, Hui Zhang, Hana Lee, Shuyang Sun, Pingjun Li, Yangguang Zhu, ByungIn Yoo, Xiaojuan Qi, Jae-Joon Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00310](https://doi.org/10.1109/CVPR52688.2022.00310)

**Abstract**:

Video Panoptic Segmentation (VPS) aims at assigning a class label to each pixel, uniquely segmenting and identifying all object instances consistently across all frames. Classic solutions usually decompose the VPS task into several subtasks and utilize multiple surrogates (e.g. boxes and masks, centers and offsets) to represent objects. However, this divide-and-conquer strategy requires complex post-processing in both spatial and temporal domains and is vulnerable to failures from surrogate tasks. In this paper, inspired by object-centric learning which learns compact and robust object representations, we present Slot- VPS, the first end-to-end framework for this task. We encode all panoptic entities in a video, including both foreground instances and background semantics, with a unified representation called panoptic slots. The coherent spatio-temporal object's information is retrieved and encoded into the panoptic slots by the proposed Video Panoptic Retriever, enabling to localize, segment, differentiate, and associate objects in a unified manner. Finally, the output panoptic slots can be directly converted into the class, mask, and object ID of panoptic objects in the video. We conduct extensive ablation studies and demonstrate the effectiveness of our approach on two benchmark datasets, Cityscapes- VP S (val and test sets) and VIPER (val set), achieving new state-of-the-art performance of 63.7, 63.3 and 56.2 VPQ, respectively.

----

## [305] Explore Spatio-temporal Aggregation for Insubstantial Object Detection: Benchmark Dataset and Baseline

**Authors**: *Kailai Zhou, Yibo Wang, Tao Lv, Yunqian Li, Linsen Chen, Qiu Shen, Xun Cao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00311](https://doi.org/10.1109/CVPR52688.2022.00311)

**Abstract**:

We endeavor on a rarely explored task named Insubstantial Object Detection (IOD), which aims to localize the object with following characteristics: (1) amorphous shape with indistinct boundary; (2) similarity to surroundings; (3) absence in color. Accordingly, it is far more challenging to distinguish insubstantial objects in a single static frame and the collaborative representation of spatial and temporal information is crucial. Thus, we construct an IOD-Video dataset comprised of 600 videos (141,017 frames) covering various distances, sizes, visibility, and scenes captured by different spectral ranges. In addition, we develop a spatio-temporal aggregation framework for IOD, in which different backbones are deployed and a spatio-temporal aggregation loss (STAloss) is elaborately designed to leverage the consistency along the time axis. Experiments conducted on IOD-Video dataset demonstrate that spatio-temporal aggregation can significantly improve the performance of IOD. We hope our work will attract further researches into this valuable yet challenging task. The code will be available at: https://github.com/CalayZhou/IOD-Video.

----

## [306] Video Shadow Detection via Spatio-Temporal Interpolation Consistency Training

**Authors**: *Xiao Lu, Yihong Cao, Sheng Liu, Chengjiang Long, Zipei Chen, Xuanyu Zhou, Yimin Yang, Chunxia Xiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00312](https://doi.org/10.1109/CVPR52688.2022.00312)

**Abstract**:

It is challenging to annotate large-scale datasets for supervised video shadow detection methods. Using a model trained on labeled images to the video frames directly may lead to high generalization error and temporal inconsistent results. In this paper, we address these challenges by proposing a Spatio-Temporal Interpolation Consistency Training (STICT) framework to rationally feed the unlabeled video frames together with the labeled images into an image shadow detection network training. Specifically, we propose the Spatial and Temporal ICT, in which we define two new interpolation schemes, i.e., the spatial interpolation and the temporal interpolation. We then derive the spatial and temporal interpolation consistency constraints accordingly for enhancing generalization in the pixel-wise classification task and for encouraging temporal consistent predictions, respectively. In addition, we design a Scale- Aware Network for multi-scale shadow knowledge learning in images, and propose a scale-consistency constraint to minimize the discrepancy among the predictions at different scales. Our proposed approach is extensively validated on the ViSha dataset and a self-annotated dataset. Experimental results show that, even without video labels, our approach is better than most state of the art supervised, semi-supervised or unsupervised image/video shadow detection methods and other methods in related tasks. Code and dataset are available at https://github.com/yihong-97/STICT.

----

## [307] Coarse-to-Fine Feature Mining for Video Semantic Segmentation

**Authors**: *Guolei Sun, Yun Liu, Henghui Ding, Thomas Probst, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00313](https://doi.org/10.1109/CVPR52688.2022.00313)

**Abstract**:

The contextual information plays a core role in semantic segmentation. As for video semantic segmentation, the contexts include static contexts and motional contexts, corresponding to static content and moving content in a video clip, respectively. The static contexts are well exploited in image semantic segmentation by learning multi-scale and global/long-range features. The motional contexts are studied in previous video semantic segmentation. However, there is no research about how to simultaneously learn static and motional contexts which are highly correlated and complementary to each other. To address this problem, we propose a Coarse-to-Fine Feature Mining (CFFM) technique to learn a unified presentation of static contexts and motional contexts. This technique consists of two parts: coarse-to-fine feature assembling and cross-frame feature mining. The former operation prepares data for further processing, enabling the subsequent joint learning of static and motional contexts. The latter operation mines useful information/contexts from the sequential frames to enhance the video contexts of the features of the target frame. The enhanced features can be directly applied for the final prediction. Experimental results on popular benchmarks demonstrate that the proposed CFFM performs favorably against state-of-the-art methods for video semantic segmentation. Our implementation is available at https://github.com/GuoleiSun/VSS-CFFM.

----

## [308] Tencent-MVSE: A Large-Scale Benchmark Dataset for Multi-Modal Video Similarity Evaluation

**Authors**: *Zhaoyang Zeng, Yongsheng Luo, Zhenhua Liu, Fengyun Rao, Dian Li, Weidong Guo, Zhen Wen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00314](https://doi.org/10.1109/CVPR52688.2022.00314)

**Abstract**:

Multi-modal video similarity evaluation is important for video recommendation systems such as video de-duplication, relevance matching, ranking, and diversity control. However, there still lacks a benchmark dataset that can support supervised training and accurate evaluation. In this paper, we propose the Tencent-MVSE dataset, which is the first benchmark dataset for the multi-modal video similarity evaluation task. The Tencent-MVSE dataset contains video pairs similarity annotations, and diverse metadata including Chinese title, automatic speech recognition (ASR) text, as well as human-annotated categories/tags. We provide a simple baseline with a multi-modal Transformer architecture to perform supervised multi-modal video similarity evaluation. We also explore pre-training strategies to make use of the unpaired data. The whole dataset as well as our baseline will be released to promote the development of the multi-modal video similarity evaluation. The dataset has been released in https://tencent-mvse.github.io/.

----

## [309] Object-Region Video Transformers

**Authors**: *Roei Herzig, Elad Ben-Avraham, Karttikeya Mangalam, Amir Bar, Gal Chechik, Anna Rohrbach, Trevor Darrell, Amir Globerson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00315](https://doi.org/10.1109/CVPR52688.2022.00315)

**Abstract**:

Recently, video transformers have shown great success in video understanding, exceeding CNN performance; yet existing video transformer models do not explicitly model objects, although objects can be essential for recognizing actions. In this work, we present Object-Region Video Transformers (ORViT), an object-centric approach that extends video transformer layers with a block that directly incorporates object representations. The key idea is to fuse object-centric representations starting from early layers and propagate them into the transformer-layers, thus affecting the spatio-temporal representations throughout the network. Our ORViT block consists of two object-level streams: appearance and dynamics. In the appearance stream, an “Object-Region Attention” module applies self-attention over the patches and object regions. In this way, visual object regions interact with uniform patch tokens and enrich them with contextualized object information. We further model object dynamics via a separate “Object-Dynamics Module”, which captures trajectory interactions, and show how to integrate the two streams. We evaluate our model on four tasks and five datasets: compositional and few-shot action recognition on SomethingElse, spatio-temporal action detection on AVA, and standard action recognition on Something-Something V2, Diving48 and Epic-Kitchen100. We show strong performance improvement across all tasks and datasets considered, demonstrating the value of a model that incorporates object representations into a transformer architecture. For code and pretrained models, visit the project page at https://roeiherz.github.io/ORViT/

----

## [310] Colar: Effective and Efficient Online Action Detection by Consulting Exemplars

**Authors**: *Le Yang, Junwei Han, Dingwen Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00316](https://doi.org/10.1109/CVPR52688.2022.00316)

**Abstract**:

Online action detection has attracted increasing research interests in recent years. Current works model historical dependencies and anticipate the future to perceive the action evolution within a video segment and improve the detection accuracy. However, the existing paradigm ignores category-level modeling and does not pay sufficient attention to efficiency. Considering a category, its representative frames exhibit various characteristics. Thus, the category-level modeling can provide complimentary guidance to the temporal dependencies modeling. This paper develops an effective exemplar-consultation mechanism that first measures the similarity between a frame and exemplary frames, and then aggregates exemplary features based on the similarity weights. This is also an efficient mechanism, as both similarity measurement and feature aggregation require limited computations. Based on the exemplar-consultation mechanism, the long-term dependencies can be captured by regarding historical frames as exemplars, while the category-level modeling can be achieved by regarding representative frames from a category as exemplars. Due to the complementarity from the categorylevel modeling, our method employs a lightweight architecture but achieves new high performance on three benchmarks. In addition, using a spatio-temporal network to tackle video frames, our method makes a good trade-off between effectiveness and efficiency. Code is available at https://github.com/VividLe/Online-Action-Detection.

----

## [311] SimVP: Simpler yet Better Video Prediction

**Authors**: *Zhangyang Gao, Cheng Tan, Lirong Wu, Stan Z. Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00317](https://doi.org/10.1109/CVPR52688.2022.00317)

**Abstract**:

From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVp, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. Without introducing any additional tricks and complicated strategies, we can achieve state-of-the-art performance on five benchmark datasets. Through extended experiments, we demonstrate that SimVP has strong generalization and extensibility on real-world datasets. The significant reduction of training cost makes it easier to scale to complex scenarios. We believe SimVP can serve as a solid baseline to stimulate the further development of video prediction.

----

## [312] Imposing Consistency for Optical Flow Estimation

**Authors**: *Jisoo Jeong, Jamie Menjay Lin, Fatih Porikli, Nojun Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00318](https://doi.org/10.1109/CVPR52688.2022.00318)

**Abstract**:

Imposing consistency through proxy tasks has been shown to enhance data-driven learning and enable self-supervision in various tasks. This paper introduces novel and effective consistency strategies for optical flow estimation, a problem where labels from real-world data are very challenging to derive. More specifically, we propose occlusion consistency and zero forcing in the forms of self-supervised learning and transformation consistency in the form of semi-supervised learning. We apply these consistency techniques in a way that the network model learns to describe pixel-level motions better while requiring no additional annotations. We demonstrate that our consistency strategies applied to a strong baseline network model using the original datasets and labels provide further improvements, attaining the state-of-the-art results on the KITTI-2015 scene flow benchmark in the non-stereo category. Our method achieves the best foreground accuracy (4.33% in Fl-all) over both the stereo and non-stereo categories, even though using only monocular image inputs.

----

## [313] Stand-Alone Inter-Frame Attention in Video Models

**Authors**: *Fuchen Long, Zhaofan Qiu, Yingwei Pan, Ting Yao, Jiebo Luo, Tao Mei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00319](https://doi.org/10.1109/CVPR52688.2022.00319)

**Abstract**:

Motion, as the uniqueness of a video, has been critical to the development of video understanding models. Modern deep learning models leverage motion by either executing spatio-temporal 3D convolutions, factorizing 3D convolutions into spatial and temporal convolutions separately, or computing self-attention along temporal dimension. The implicit assumption behind such successes is that the feature maps across consecutive frames can be nicely aggregated. Nevertheless, the assumption may not always hold especially for the regions with large deformation. In this paper, we present a new recipe of inter-frame attention block, namely Stand-alone Inter-Frame Attention (SIFA), that novelly delves into the deformation across frames to estimate local self-attention on each spatial location. Technically, SIFA remoulds the deformable design via re-scaling the offset predictions by the difference between two frames. Taking each spatial location in the current frame as the query, the locally deformable neighbors in the next frame are regarded as the keys/values. Then, SIFA measures the similarity between query and keys as stand-alone attention to weighted average the values for temporal aggregation. We further plug SIFA block into ConvNets and Vision Transformer, respectively, to devise SIFA-Net and SIFA-Transformer. Extensive experiments conducted on four video datasets demonstrate the superiority of SIFA-Net and SIFA-Transformer as stronger backbones. More remarkably, SIFA-Transformer achieves an accuracy of 83.1% on Kinetics-400 dataset. Source code is available at https://github.com/FuchenUSTC/SIFA.

----

## [314] Video Swin Transformer

**Authors**: *Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00320](https://doi.org/10.1109/CVPR52688.2022.00320)

**Abstract**:

The vision community is witnessing a modeling shift from CNNs to Transformers, where pure Transformer architectures have attained top accuracy on the major video recognition benchmarks. These video models are all built on Transformer layers that globally connect patches across the spatial and temporal dimensions. In this paper, we instead advocate an inductive bias of locality in video Transformers, which leads to a better speed-accuracy trade-off compared to previous approaches which compute self-attention globally even with spatial-temporal factorization. The locality of the proposed video architecture is realized by adapting the Swin Transformer designed for the image domain, while continuing to leverage the power of pre-trained image models. Our approach achieves state-of-the-art accuracy on a broad range of video recognition benchmarks, including on action recognition (84.9 top-l accuracy on Kinetics-400 and 85.9 top-l accuracy on Kinetics-600 with ~20× less pre-training data and ~3× smaller model size) and temporal modeling (69.6 top-l accuracy on Something-Something v2).

----

## [315] Bayesian Nonparametric Submodular Video Partition for Robust Anomaly Detection

**Authors**: *Hitesh Sapkota, Qi Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00321](https://doi.org/10.1109/CVPR52688.2022.00321)

**Abstract**:

Multiple-instance learning (MIL) provides an effective way to tackle the video anomaly detection problem by modeling it as a weakly supervised problem as the labels are usually only available at the video level while missing for frames due to expensive labeling cost. We propose to conduct novel Bayesian non-parametric submodular video partition (BN-SVP) to significantly improve MIL model training that can offer a highly reliable solution for robust anomaly detection in practical settings that include outlier segments or multiple types of abnormal events. BN-SVP essentially performs dynamic non-parametric hierarchical clustering with an enhanced self-transition that groups segments in a video into temporally consistent and semantically coherent hidden states that can be naturally interpreted as scenes. Each segment is assumed to be generated through a non-parametric mixture process that allows variations of segments within the same scenes to accommodate the dynamic and noisy nature of many real-world surveillance videos. The scene and mixture component assignment of BN-SVP also induces a pairwise similarity among segments, resulting in non-parametric construction of a submodular set function. Integrating this function with an MIL loss effectively exposes the model to a diverse set of potentially positive instances to improve its training. A greedy algorithm is developed to optimize the submodular function and support efficient model training. Our theoretical analysis ensures a strong performance guarantee of the proposed algorithm. The effectiveness of the proposed approach is demonstrated over multiple real-world anomaly video datasets with robust detection performance.

----

## [316] Likert Scoring with Grade Decoupling for Long-term Action Assessment

**Authors**: *Angchi Xu, Ling-An Zeng, Wei-Shi Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00323](https://doi.org/10.1109/CVPR52688.2022.00323)

**Abstract**:

Long-term action quality assessment is a task of evaluating how well an action is performed, namely, estimating a quality score from a long video. Intuitively, long-term actions generally involve parts exhibiting different levels of skill, and we call the levels of skill as performance grades. For example, technical highlights and faults may appear in the same long-term action. Hence, the final score should be determined by the comprehensive effect of different grades exhibited in the video. To explore this latent relationship, we design a novel Likert scoring paradigm in-spired by the Likert scale in psychometrics, in which we quantify the grades explicitly and generate the final quality score by combining the quantitative values and the corresponding responses estimated from the video, instead of performing direct regression. Moreover, we extract grade-specific features, which will be used to estimate the responses of each grade, through a Transformer decoder architecture with diverse learnable queries. The whole model is named as Grade-decoupling Likert Transformer (GDLT), and we achieve state-of-the-art results on two long-term action assessment datasets.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page https://isee-ai.cn/-angchi/CVPR22_GDLT.html

----

## [317] Complex Video Action Reasoning via Learnable Markov Logic Network

**Authors**: *Yang Jin, Linchao Zhu, Yadong Mu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00324](https://doi.org/10.1109/CVPR52688.2022.00324)

**Abstract**:

Profiting from the advance of deep convolutional networks, current state-of-the-art video action recognition models have achieved remarkable progress. Nevertheless, most of existing models suffer from low interpretability of the predicted actions. Inspired by the observation that temporally-configured human-object interactions often serve as a key indicator of many actions, this work crafts an action reasoning framework that performs Markov Logic Network (MLN) based probabilistic logical inference. Crucially, we propose to encode an action by first-order logical rules that correspond to the temporal changes of visual relationships in videos. The main contributions of this work are two-fold: 1) Different from existing black-box models, the proposed model simultaneously implements the localization of temporal boundaries and the recognition of action categories by grounding the logical rules of MLN in videos. The weight associated with each such rule further provides an estimate of confidence. These collectively make our model more explainable and robust. 2) Instead of using hand-crafted logical rules in conventional MLN, we develop a data-driven instantiation of the MLN. In specific, a hybrid learning scheme is proposed. It combines MLN's weight learning and reinforcement learning, using the former's results as a self-critic for guiding the latter's training. Additionally, by treating actions as logical predicates, the proposed framework can also be integrated with deep models for further performance boost. Comprehensive experiments on two complex video action datasets (Charades & CAD-120) clearly demonstrate the effectiveness and explainability of our proposed method.

----

## [318] Learning from Temporal Gradient for Semi-supervised Action Recognition

**Authors**: *Junfei Xiao, Longlong Jing, Lin Zhang, Ju He, Qi She, Zongwei Zhou, Alan L. Yuille, Yingwei Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00325](https://doi.org/10.1109/CVPR52688.2022.00325)

**Abstract**:

Semi-supervised video action recognition tends to enable deep neural networks to achieve remarkable performance even with very limited labeled data. However, existing methods are mainly transferred from current image-based methods (e.g., FixMatch). Without specifically utilizing the temporal dynamics and inherent multimodal attributes, their results could be suboptimal. To better leverage the encoded temporal information in videos, we introduce temporal gradient as an additional modality for more attentive feature extraction in this paper. To be specific, our method explicitly distills the fine-grained motion representations from temporal gradient (TG) and imposes consistency across different modalities (i.e., RGB and TG). The performance of semi-supervised action recognition is significantly improved without additional computation or parameters during inference. Our method achieves the state-of-the-art performance on three video action recognition benchmarks (i.e., Kinetics-400, UCF-101, and HMDB-51) under several typical semi-supervised settings (i.e., different ratios of labeled data). Code is made available at https://github.com/lambert-x/video-semisup.

----

## [319] Semi-Supervised Video Semantic Segmentation with Inter-Frame Feature Reconstruction

**Authors**: *Jiafan Zhuang, Zilei Wang, Yuan Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00326](https://doi.org/10.1109/CVPR52688.2022.00326)

**Abstract**:

One major challenge for semantic segmentation in realworld scenarios is only limited pixel-level labels available due to high expense of human labor though a vast volume of video data is provided. Existing semi-supervised methods attempt to exploit unlabeled data in model training, but they just regard video as a set of independent images. To better explore semi-supervised segmentation problem with video data, we formulate a semi-supervised video semantic segmentation task in this paper. For this task, we observe that the overfitting is surprisingly severe between labeled and unlabeled frames within a training video although they are very similar in style and contents. This is called inner-video overfitting, and it would actually lead to inferior performance. To tackle this issue, we propose a novel interframe feature reconstruction (IFR) technique to leverage the ground-truth labels to supervise the model training on unlabeled frames. IFR is essentially to utilize the internal relevance of different frames within a video. During training, IFR would enforce the feature distributions between labeled and unlabeled frames to be narrowed. Consequently, the inner-video overfitting issue can be effectively alleviated. We conduct extensive experiments on Cityscapes and CamVid, and the results demonstrate the superiority of our proposed method to previous state-of-the-art methods. The code is available at https://github.com/jfzhuang/IFR.

----

## [320] Weakly Supervised Temporal Action Localization via Representative Snippet Knowledge Propagation

**Authors**: *Linjiang Huang, Liang Wang, Hongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00327](https://doi.org/10.1109/CVPR52688.2022.00327)

**Abstract**:

Weakly supervised temporal action localization aims to localize temporal boundaries of actions and simultaneously identify their categories with only video-level category labels. Many existing methods seek to generate pseudo labels for bridging the discrepancy between classification and localization, but usually only make use of limited contextual information for pseudo label generation. To alleviate this problem, we propose a representative snippet summarization and propagation framework. Our method seeks to mine the representative snippets in each video for propagating information between video snippets to generate better pseudo labels. For each video, its own representative snippets and the representative snippets from a memory bank are propagated to update the input features in an intra and inter-video manner. The pseudo labels are generated from the temporal class activation maps of the updated features to rectify the predictions of the main branch. Our method obtains superior performance in comparison to the existing methods on two benchmarks, THUMOS14 and ActivityNet1.3, achieving gains as high as 1.2% in terms of average mAP on THUMOS14. Our code is available at https://github.com/LeonHLJ/RSKP.

----

## [321] Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos

**Authors**: *Shaowei Liu, Subarna Tripathi, Somdeb Majumdar, Xiaolong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00328](https://doi.org/10.1109/CVPR52688.2022.00328)

**Abstract**:

We propose to forecast future hand-object interactions given an egocentric video. Instead of predicting action labels or pixels, we directly predict the hand motion trajectory and the future contact points on the next active object (i.e., interaction hotspots). This relatively low-dimensional representation provides a con-crete description of future interactions. To tackle this task, we first provide an automatic way to collect trajectory and hotspots labels on large-scale data. We then use this data to train an Object-Centric Transformer (OCT) model for prediction. Our model performs hand and object interaction reasoning via the self-attention mechanism in Transformers. OCT also provides a probabilistic framework to sample the future trajectory and hotspots to handle uncertainty in prediction. We perform experi-ments on the Epic-Kitchens-55, Epic-Kitchens-100 and EGTEA Gaze+ datasets, and show that OCT significantly outperforms state-of the-art approaches by a large margin. Project page is available at https://stevenlsw.github.io/hoi-forecast.

----

## [322] Human Hands as Probes for Interactive Object Understanding

**Authors**: *Mohit Goyal, Sahil Modi, Rishabh Goyal, Saurabh Gupta*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00329](https://doi.org/10.1109/CVPR52688.2022.00329)

**Abstract**:

Interactive object understanding, or what we can do to objects and how is a long-standing goal of computer vision. In this paper, we tackle this problem through observation of human hands in in-the-wild egocentric videos. We demonstrate that observation of what human hands interact with and how can provide both the relevant data and the necessary supervision. Attending to hands, readily localizes and stabilizes active objects for learning and reveals places where interactions with objects occur. Analyzing the hands shows what we can do to objects and how. We apply these basic principles on the EPIC-KITCHENS dataset, and successfully learn state-sensitive features, and object affordances (regions of interaction and afforded grasps), purely by observing hands in egocentric videos.

----

## [323] LD-ConGR: A Large RGB-D Video Dataset for Long-Distance Continuous Gesture Recognition

**Authors**: *Dan Liu, Libo Zhang, Yanjun Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00330](https://doi.org/10.1109/CVPR52688.2022.00330)

**Abstract**:

Gesture recognition plays an important role in natural human-computer interaction and sign language recognition. Existing research on gesture recognition is limited to close-range interaction such as vehicle gesture control and face-to-face communication. To apply gesture recognition to long-distance interactive scenes such as meetings and smart homes, a large RGB-D video dataset LD-ConGR is established in this paper. LD-ConGR is distinguished from existing gesture datasets by its long-distance gesture collection, fine-grained annotations, and high video qual-ity. Specifically, 1) the farthest gesture provided by the LD-ConGR is captured 4m away from the camera while existing gesture datasets collect gestures within 1m from the camera; 2) besides the gesture category, the temporal segmentation of gestures and hand location are also anno-tated in LD-ConGR; 3) videos are captured at high reso-lution (1280 x 720 for color streams and 640 x 576 for depth streams) and high frame rate (30 fps). On top of the LD-ConGR, a series of experimental and studies are conducted, and the proposed gesture region estimation and key frame sampling strategies are demonstrated to be effective in dealing with long-distance gesture recognition and the uncertainty of gesture duration. The dataset and experimen-tal results presented in this paper are expected to boost the research of long-distance gesture recognition. The dataset is available at https://github.com/Diananini/LD-ConGR-CVPR2022.

----

## [324] Object-aware Video-language Pre-training for Retrieval

**Authors**: *Alex Jinpeng Wang, Yixiao Ge, Guanyu Cai, Rui Yan, Xudong Lin, Ying Shan, Xiaohu Qie, Mike Zheng Shou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00331](https://doi.org/10.1109/CVPR52688.2022.00331)

**Abstract**:

Recently, by introducing large-scale dataset and strong transformer network, video-language pre-training has shown great success especially for retrieval. Yet, existing video-language transformer models do not explicitly fine-grained semantic align. In this work, we present Object-aware Transformers, an object-centric approach that extends video-language transformer to incorporate object representations. The key idea is to leverage the bounding boxes and object tags to guide the training process. We evaluate our model on three standard sub-tasks of video-text matching on four widely used benchmarks. We also provide deep analysis and detailed ablation about the proposed method. We show clear improvement in performance across all tasks and datasets considered, demonstrating the value of a model that incorporates object representations into a video-language architecture. The code has been released in https://github.com/FingerRec/OA-Transformer.

----

## [325] Fast and Unsupervised Action Boundary Detection for Action Segmentation

**Authors**: *Zexing Du, Xue Wang, Guoqing Zhou, Qing Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00332](https://doi.org/10.1109/CVPR52688.2022.00332)

**Abstract**:

To deal with the great number of untrimmed videos produced every day, we propose an efficient unsupervised action segmentation method by detecting boundaries, named action boundary detection (ABD). In particular, the proposed method has the following advantages: no training stage and low-latency inference. To detect action boundaries, we estimate the similarities across smoothed frames, which inherently have the properties of internal consistency within actions and external discrepancy across actions. Under this circumstance, we successfully transfer the boundary detection task into the change point detection based on the similarity. Then, non-maximum suppression (NMS) is conducted in local windows to select the smallest points as candidate boundaries. In addition, a clustering algorithm is followed to refine the initial proposals. Moreover, we also extend ABD to the online setting, which enables real-time action segmentation in long untrimmed videos. By evaluating on four challenging datasets, our method achieves state-of-the-art performance. Moreover, thanks to the efficiency of ABD, we achieve the best trade-off between the accuracy and the inference time compared with existing unsupervised approaches.

----

## [326] Multiview Transformers for Video Recognition

**Authors**: *Shen Yan, Xuehan Xiong, Anurag Arnab, Zhichao Lu, Mi Zhang, Chen Sun, Cordelia Schmid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00333](https://doi.org/10.1109/CVPR52688.2022.00333)

**Abstract**:

Video understanding requires reasoning at multiple spatiotemporal resolutions – from short fine-grained motions to events taking place over longer durations. Although transformer architectures have recently advanced the state-of-the-art, they have not explicitly modelled different spatiotemporal resolutions. To this end, we present Multiview Transformers for Video Recognition (MTV). Our model consists of separate encoders to represent different views of the input video with lateral connections to fuse information across views. We present thorough ablation studies of our model and show that MTV consistently performs better than single-view counterparts in terms of accuracy and computational cost across a range of model sizes. Furthermore, we achieve state-of-the-art results on six standard datasets, and improve even further with large-scale pretraining. Code and checkpoints are available at: https://github.com/google-research/scenic.

----

## [327] Semi-Weakly-Supervised Learning of Complex Actions from Instructional Task Videos

**Authors**: *Yuhan Shen, Ehsan Elhamifar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00334](https://doi.org/10.1109/CVPR52688.2022.00334)

**Abstract**:

We address the problem of action segmentation in instructional task videos with a small number of weakly-labeled training videos and a large number of unlabeled videos, which we refer to as Semi-Weakly-Supervised Learning (SWSL) of actions. We propose a general SWSL framework that can efficiently learn from both types of videos and can leverage any of the existing weakly-supervised action segmentation methods. Our key observation is that the distance between the transcript of an unlabeled video and those of the weakly-labeled videos from the same task is small yet often nonzero. Therefore, we develop a Soft Restricted Edit (SRE) loss to encourage small variations between the predicted transcripts of unlabeled videos and ground-truth transcripts of the weakly-labeled videos of the same task. To compute the SRE loss, we develop a flexible transcript prediction (FTP) method that uses the output of the action classifier to find both the length of the transcript and the sequence of actions occurring in an unlabeled video. We propose an efficient learning scheme in which we alternate between minimizing our proposed loss and generating pseudo-transcripts for unlabeled videos. By experiments on two benchmark datasets, we demonstrate that our approach can significantly improve the performance by using unlabeled videos, especially when the number of weakly-labeled videos is small. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/Yuhan-Shen/SWSL..

----

## [328] Progressive Attention on Multi-Level Dense Difference Maps for Generic Event Boundary Detection

**Authors**: *Jiaqi Tang, Zhaoyang Liu, Chen Qian, Wayne Wu, Limin Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00335](https://doi.org/10.1109/CVPR52688.2022.00335)

**Abstract**:

Generic event boundary detection (GEBD) is an important yet challenging task in video understanding, which aims at detecting the moments where humans naturally perceive event boundaries. The main challenge of this task is perceiving various temporal variations of diverse event boundaries. To this end, this paper presents an effective and end-to-end learnable framework (DDM-Net). To tackle the diversity and complicated semantics of event boundaries, we make three notable improvements. First, we construct a feature bank to store multi-level features of space and time, prepared for difference calculation at multiple scales. Second, to alleviate inadequate temporal modeling of pre-vious methods, we present dense difference maps (DDM) to comprehensively characterize the motion pattern. Finally, we exploit progressive attention on multi-level DDM to jointly aggregate appearance and motion clues. As a result, DDM-Net respectively achieves a significant boost of 14% and 8% on Kinetics-GEBD and TAPOS benchmark, and outperforms the top-1 winner solution of LOVEU Challenge@CVPR 2021 without bells and whistles. The state-of-the-art result demonstrates the effectiveness of richer motion representation and more sophisticated aggregation, in handling the diversity of GEBD. The code is made available at https://github.com/MCG-NJU/DDM.

----

## [329] Comparing Correspondences: Video Prediction with Correspondence-wise Losses

**Authors**: *Daniel Geng, Max Hamilton, Andrew Owens*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00336](https://doi.org/10.1109/CVPR52688.2022.00336)

**Abstract**:

Image prediction methods often struggle on tasks that require changing the positions of objects, such as video prediction, producing blurry images that average over the many positions that objects might occupy. In this paper, we propose a simple change to existing image similarity metrics that makes them more robust to positional errors: we match the images using optical flow, then measure the visual similarity of corresponding pixels. This change leads to crisper and more perceptually accurate predictions, and does not require modifications to the image prediction network. We apply our method to a variety of video prediction tasks, where it obtains strong performance with simple network architectures, and to the closely related task of video interpolation. Code and results are available at our webpage: https://dangeng.github.io/CorrWiseLosses

----

## [330] Sound-Guided Semantic Image Manipulation

**Authors**: *Seung Hyun Lee, Wonseok Roh, Wonmin Byeon, Sang Ho Yoon, Chanyoung Kim, Jinkyu Kim, Sangpil Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00337](https://doi.org/10.1109/CVPR52688.2022.00337)

**Abstract**:

The recent success of the generative model shows that leveraging the multi-modal embedding space can manipu-late an image using text information. However, manipulating an image with other sources rather than text, such as sound, is not easy due to the dynamic characteristics of the sources. Especially, sound can convey vivid emotions and dynamic expressions of the real world. Here, we propose a framework that directly encodes sound into the multi-modal (image-text) embedding space and manipulates an image from the space. Our audio encoder is trained to pro-duce a latent representation from an audio input, which is forced to be aligned with image and text representations in the multi-modal embedding space. We use a direct latent op-timization method based on aligned embeddings for sound-guided image manipulation. We also show that our method can mix different modalities, i.e., text and audio, which en-rich the variety of the image modification. The experiments on zero-shot audio classification and semantic-level image classification show that our proposed model outperforms other text and sound-guided state-of-the-art methods.

----

## [331] Expressive Talking Head Generation with Granular Audio-Visual Control

**Authors**: *Borong Liang, Yan Pan, Zhizhi Guo, Hang Zhou, Zhibin Hong, Xiaoguang Han, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00338](https://doi.org/10.1109/CVPR52688.2022.00338)

**Abstract**:

Generating expressive talking heads is essential for creating virtual humans. However, existing one- or few-shot methods focus on lip-sync and head motion, ignoring the emotional expressions that make talking faces realistic. In this paper, we propose the Granularly Controlled Audio-Visual Talking Heads (GC-AVT), which controls lip movements, head poses, and facial expressions of a talking head in a granular manner. Our insight is to decouple the audio-visual driving sources through prior-based pre-processing designs. Detailedly, we disassemble the driving image into three complementary parts including: 1) a cropped mouth that facilitates lip-sync; 2) a masked head that implicitly learns pose; and 3) the upper face which works corporately and complementarily with a time-shifted mouth to contribute the expression. Interestingly, the encoded features from the three sources are integrally balanced through reconstruction training. Extensive experiments show that our method generates expressive faces with not only synced mouth shapes, controllable poses, but precisely animated emotional expressions as well.

----

## [332] Depth-Aware Generative Adversarial Network for Talking Head Video Generation

**Authors**: *Fa-Ting Hong, Longhao Zhang, Li Shen, Dan Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00339](https://doi.org/10.1109/CVPR52688.2022.00339)

**Abstract**:

Talking head video generation aims to produce a synthetic human face video that contains the identity and pose information respectively from a given source image and a driving video. Existing works for this task heavily rely on 2D representations (e.g. appearance and motion) learned from the input images. However, dense 3D facial geometry (e.g. pixel-wise depth) is extremely important for this task as it is particularly beneficial for us to essentially generate accurate 3D face structures and distinguish noisy information from the possibly cluttered background. Nevertheless, dense 3D geometry annotations are prohibitively costly for videos and are typically not available for this video generation task. In this paper, we introduce a self-supervised face-depth learning method to automatically recover dense 3D facial geometry (i.e. depth) from the face videos without the requirement of any expensive 3D annotation data. Based on the learned dense depth maps, we further propose to leverage them to estimate sparse facial keypoints that capture the critical movement of the human head. In a more dense way, the depth is also utilized to learn 3D-aware cross-modal (i.e. appearance and depth) attention to guide the generation of motion fields for warping source image representations. All these contributions compose a novel depth-aware generative adversarial network (DaGAN) for talking head generation. Extensive experiments conducted demonstrate that our proposed method can generate highly realistic faces, and achieve significant results on the unseen human faces. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/harlanhong/CVPR2022-DaGAN

----

## [333] Learning Motion-Dependent Appearance for High-Fidelity Rendering of Dynamic Humans from a Single Camera

**Authors**: *Jae Shin Yoon, Duygu Ceylan, Tuanfeng Y. Wang, Jingwan Lu, Jimei Yang, Zhixin Shu, Hyun Soo Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00340](https://doi.org/10.1109/CVPR52688.2022.00340)

**Abstract**:

Appearance of dressed humans undergoes a complex geometric transformation induced not only by the static pose but also by its dynamics, i.e., there exists a number of cloth geometric configurations given a pose depending on the way it has moved. Such appearance modeling conditioned on motion has been largely neglected in existing human rendering methods, resulting in rendering of physically implausible motion. A key challenge of learning the dynamics of the appearance lies in the requirement of a prohibitively large amount of observations. In this paper, we present a compact motion representation by enforcing equivariance—a representation is expected to be transformed in the way that the pose is transformed. We model an equivariant encoder that can generate the generalizable representation from the spatial and temporal derivatives of the 3D body surface. This learned representation is decoded by a compositional multi-task decoder that renders high fidelity time-varying appearance. Our experiments show that our method can generate a temporally coherent video of dynamic humans for unseen body poses and novel views given a single view video.

----

## [334] Audio-driven Neural Gesture Reenactment with Video Motion Graphs

**Authors**: *Yang Zhou, Jimei Yang, Dingzeyu Li, Jun Saito, Deepali Aneja, Evangelos Kalogerakis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00341](https://doi.org/10.1109/CVPR52688.2022.00341)

**Abstract**:

Human speech is often accompanied by body gestures including arm and hand gestures. We present a method that reenacts a high-quality video with gestures matching a target speech audio. The key idea of our method is to split and re-assemble clips from a reference video through a novel video motion graph encoding valid transitions between clips. To seamlessly connect different clips in the reenactment, we propose a pose-aware video blending network which synthesizes video frames around the stitched frames between two clips. Moreover, we developed an audio-based gesture searching algorithm to find the optimal order of the reenacted frames. Our system generates reen-actments that are consistent with both the audio rhythms and the speech content. We evaluate our synthesized video quality quantitatively, qualitatively, and with user studies, demonstrating that our method produces videos of much higher quality and consistency with the target audio compared to previous work and baselines. Our project page https://github.com/yzhou359/vid-reenact includes code and data.

----

## [335] Portrait Eyeglasses and Shadow Removal by Leveraging 3D Synthetic Data

**Authors**: *Junfeng Lyu, Zhibo Wang, Feng Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00342](https://doi.org/10.1109/CVPR52688.2022.00342)

**Abstract**:

In portraits, eyeglasses may occlude facial regions and generate cast shadows on faces, which degrades the performance of many techniques like face verification and expression recognition. Portrait eyeglasses removal is critical in handling these problems. However, completely removing the eyeglasses is challenging because the lighting effects (e.g., cast shadows) caused by them are often complex. In this paper, we propose a novel framework to remove eyeglasses as well as their cast shadows from face images. The method works in a detect-then-remove manner, in which eyeglasses and cast shadows are both detected and then removed from images. Due to the lack of paired data for supervised training, we present a new synthetic portrait dataset with both intermediate and final supervisions for both the detection and removal tasks. Furthermore, we apply a cross-domain technique to fill the gap between the synthetic and real data. To the best of our knowledge, the proposed technique is the first to remove eyeglasses and their cast shadows simultaneously. The code and synthetic dataset are available at https://gethub.com/StoryMY/take-off-eyeglasses.

----

## [336] Weakly Supervised High-Fidelity Clothing Model Generation

**Authors**: *Ruili Feng, Cheng Ma, Chengji Shen, Xin Gao, Zhenjiang Liu, Xiaobo Li, Kairi Ou, Deli Zhao, Zheng-Jun Zha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00343](https://doi.org/10.1109/CVPR52688.2022.00343)

**Abstract**:

The development of online economics arouses the demand of generating images of models on product clothes, to display new clothes and promote sales. However, the expensive proprietary model images challenge the existing image virtual try-on methods in this scenario, as most of them need to be trained on considerable amounts of model images accompanied with paired clothes images. In this paper, we propose a cheap yet scalable weakly-supervised method called Deep Generative Projection (DGP) to address this specific scenario. Lying in the heart of the proposed method is to imitate the process of human predicting the wearing effect, which is an unsupervised imagination based on life experience rather than computation rules learned from supervisions. Here a pretrained StyleGAN is used to capture the practical experience of wearing. Experiments show that projecting the rough alignment of clothing and body onto the StyleGAN space can yield photo-realistic wearing results. Experiments on real scene proprietary model images demonstrate the superiority of DGP over several state-of-the-art supervised methods when generating clothing model images.

----

## [337] TemporalUV: Capturing Loose Clothing with Temporally Coherent UV Coordinates

**Authors**: *You Xie, Huiqi Mao, Angela Yao, Nils Thuerey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00344](https://doi.org/10.1109/CVPR52688.2022.00344)

**Abstract**:

We propose a novel approach to generate temporally coherent UV coordinates for loose clothing. Our method is not constrained by human body outlines and can capture loose garments and hair. We implemented a differentiable pipeline to learn UV mapping between a sequence of RGB inputs and textures via UV coordinates. Instead of treating the UV coordinates of each frame separately, our data generation approach connects all UV coordinates via feature matching for temporal stability. Subsequently, a generative model is trained to balance the spatial quality and temporal stability. It is driven by supervised and unsupervised losses in both UV and image spaces. Our experiments show that the trained models output high-quality UV coordinates and generalize to new poses. Once a sequence of UV co-ordinates has been inferred by our model, it can be used to flexibly synthesize new looks and modified visual styles. Compared to existing methods, our approach reduces the computational workload to animate new outfits by several orders of magnitude.

----

## [338] Full-Range Virtual Try-On with Recurrent Tri-Level Transform

**Authors**: *Han Yang, Xinrui Yu, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00345](https://doi.org/10.1109/CVPR52688.2022.00345)

**Abstract**:

Virtual try-on aims to transfer a target clothing image onto a reference person. Though great progress has been achieved, the functioning zone of existing works is still limited to standard clothes (e.g., plain shirt without complex laces or ripped effect), while the vast complexity and variety of non-standard clothes (e.g., off-shoulder shirt, word-shoulder dress) are largely ignored. In this work, we propose a principled framework, Re-current Tri-Level Transform (RT-VTON), that performs full-range virtual try-on on both standard and non-standard clothes. We have two key insights towards the framework design: 1) Semantics transfer requires a gradual feature transform on three different levels of clothing representations, namely clothes code, pose code and parsing code. 2) Geometry transfer requires a regularized image deformation between rigidity and flexibility. Firstly, we predict the semantics of the “after-try-on” person by recurrently refining the tri-level feature codes using local gated attention and non-local correspondence learning. Next, we design a semi-rigid deformation to align the clothing image and the predicted semantics, which preserves local warping similarity. Finally, a canonical try-on synthesizer fuses all the processed information to generate the clothed person image. Extensive experiments on conventional benchmarks along with user studies demonstrate that our framework achieves state-of-the-art performance both quantitatively and qualitatively. Notably, RT-VTON shows compelling results on a wide range of non-standard clothes. Project page: https://lzqhardworker.github.io/RT-VTON/.

----

## [339] Style-Based Global Appearance Flow for Virtual Try-On

**Authors**: *Sen He, Yi-Zhe Song, Tao Xiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00346](https://doi.org/10.1109/CVPR52688.2022.00346)

**Abstract**:

Image-based virtual try-on aims to fit an in-shop garment into a clothed person image. To achieve this, a key step is garment warping which spatially aligns the target garment with the corresponding body parts in the person image. Prior methods typically adopt a local appearance flow estimation model. They are thus intrinsically susceptible to difficult body poses/occlusions and large mis-alignments between person and garment images (see Fig. 1). To overcome this limitation, a novel global appearance flow estimation model is proposed in this work. For the first time, a StyleGAN based architecture is adopted for appearance flow estimation. This enables us to take advantage of a global style vector to encode a whole-image context to cope with the aforementioned challenges. To guide the StyleGAN flow generator to pay more attention to local garment deformation, a flow refinement module is introduced to add local context. Experiment results on a popular virtual tryon benchmark show that our method achieves new state-of-the-art performance. It is particularly effective in a ‘in-the-wild’ application scenario where the reference image is full-body resulting in a large mis-alignment with the garment image (Fig. 1 Top). Code is available at: https://github.com/SenHe/Flow-Style-VTON.

----

## [340] Dressing in the Wild by Watching Dance Videos

**Authors**: *Xin Dong, Fuwei Zhao, Zhenyu Xie, Xijin Zhang, Daniel K. Du, Min Zheng, Xiang Long, Xiaodan Liang, Jianchao Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00347](https://doi.org/10.1109/CVPR52688.2022.00347)

**Abstract**:

While significant progress has been made in garment transfer, one of the most applicable directions of human-centric image generation, existing works overlook the in-the-wild imagery, presenting severe garment-person mis-alignment as well as noticeable degradation in fine texture details. This paper, therefore, attends to virtual try-on in real-world scenes and brings essential improvements in authenticity and naturalness especially for loose garment (e.g., skirts, formal dresses), challenging poses (e.g., cross arms, bent legs), and cluttered backgrounds. Specifically, we find that the pixel flow excels at handling loose gar-ments whereas the vertex flow is preferred for hard poses, and by combining their advantages we propose a novel generative network called wFlow that can effectively push up garment transfer to in-the-wild context. Moreover, former approaches require paired images for training. Instead, we cut down the laboriousness by working on a newly constructed large-scale video dataset named Dance50k with self-supervised cross-frame training and an online cycle op-timization. The proposed Dance50k can boost real-world virtual dressing by covering a wide variety of garments under dancing poses. Extensive experiments demonstrate the superiority of our w Flow in generating realistic garment transfer results for in-the-wild images without resorting to expensive paired datasets. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Xiaodan Liang is the corresponding author. The project page of wFlow is https://awesome-wflow.github.io.

----

## [341] A Brand New Dance Partner: Music-Conditioned Pluralistic Dancing Controlled by Multiple Dance Genres

**Authors**: *Jinwoo Kim, Heeseok Oh, Seongjean Kim, Hoseok Tong, Sanghoon Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00348](https://doi.org/10.1109/CVPR52688.2022.00348)

**Abstract**:

When coming up with phrases of movement, choreographers all have their habits as they are used to their skilled dance genres. Therefore, they tend to return certain patterns of the dance genres that they are familiar with. What if artificial intelligence could be used to help choreographers blend dance genres by suggesting various dances, and one that matches their choreographic style? Numerous task-specific variants of autoregressive networks have been developed for dance generation. Yet, a serious limitation remains that all existing algorithms can return repeated patterns for a given initial pose sequence, which may be inferior. To mitigate this issue, we propose MNET, a novel and scalable approach that can perform music-conditioned pluralistic dance generation synthesized by multiple dance genres using only a single model. Here, we learn a dancegenre aware latent representation by training a conditional generative adversarial network leveraging Transformer architecture. We conduct extensive experiments on AIST++ along with user studies. Compared to the state-of-the-art methods, our method synthesizes plausible and diverse outputs according to multiple dance genres as well as generates outperforming dance sequences qualitatively and quantitatively.

----

## [342] Unpaired Cartoon Image Synthesis via Gated Cycle Mapping

**Authors**: *Yifang Men, Yuan Yao, Miaomiao Cui, Zhouhui Lian, Xuansong Xie, Xian-Sheng Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00349](https://doi.org/10.1109/CVPR52688.2022.00349)

**Abstract**:

In this paper, we present a general-purpose solution to cartoon image synthesis with unpaired training data. In contrast to previous works learning pre-defined cartoon styles for specified usage scenarios (portrait or scene), we aim to train a common cartoon translator which can not only simultaneously render exaggerated anime faces and realistic cartoon scenes, but also provide flexible user controls for desired cartoon styles. It is challenging due to the complexity of the task and the absence of paired data. The core idea of the proposed method is to introduce gated cycle mapping, that utilizes a novel gated mapping unit to produce the category-specific style code and embeds this code into cycle networks to control the translation process. For the concept of category, we classify images into different categories (e.g., 4 types: photo/cartoon portrait/scene) and learn finer-grained category translations rather than overall mappings between two domains (e.g., photo and cartoon). Furthermore, the proposed method can be easily extended to cartoon video generation with an auxiliary dataset and a new adaptive style loss. Experimental results demonstrate the superiority of the proposed method over the state of the art and validate its effectiveness in the brand-new task of general cartoon image synthesis.

----

## [343] DLFormer: Discrete Latent Transformer for Video Inpainting

**Authors**: *Jingjing Ren, Qingqing Zheng, Yuanyuan Zhao, Xuemiao Xu, Chen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00350](https://doi.org/10.1109/CVPR52688.2022.00350)

**Abstract**:

Video inpainting remains a challenging problem to fill with plausible and coherent content in unknown areas in video frames despite the prevalence of data-driven methods. Although various transformer-based architectures yield promising result for this task, they still suffer from hallucinating blurry contents and long-term spatial-temporal inconsistency. While noticing the capability of discrete representation for complex reasoning and predictive learning, we propose a novel Discrete Latent Transformer (DLFormer) to reformulate video inpainting tasks into the discrete latent space rather the previous continuous feature space. Specifically, we first learn a unique compact discrete codebook and the corresponding autoencoder to represent the target video. Built upon these representative discrete codes obtained from the entire target video, the subsequent discrete latent transformer is capable to infer proper codes for unknown areas under a self-attention mechanism, and thus produces fine-grained content with long-term spatial-temporal consistency. Moreover, we further explicitly enforce the short-term consistency to relieve temporal visual jitters via a temporal aggregation block among adjacent frames. We conduct comprehensive quantitative and qualitative evaluations to demonstrate that our method significantly outperforms other state-of-the-art approaches in reconstructing visually-plausible and spatial-temporal coherent content with fine-grained details. Code is available at https://github.com/JingjingRenabc/dlformer.

----

## [344] ST-MFNet: A Spatio-Temporal Multi-Flow Network for Frame Interpolation

**Authors**: *Duolikun Danier, Fan Zhang, David R. Bull*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00351](https://doi.org/10.1109/CVPR52688.2022.00351)

**Abstract**:

Video frame interpolation (VFI) is currently a very active research topic, with applications spanning computer vision, post production and video encoding. VFI can be extremely challenging, particularly in sequences containing large motions, occlusions or dynamic textures, where existing approaches fail to offer perceptually robust inter-polation performance. In this context, we present a novel deep learning based VFI method, ST-MFNet, based on a Spatio-Temporal Multi-Flow architecture. ST-MFNet employs a new multi-scale multi-flow predictor to estimate many-to-one intermediate flows, which are combined with conventional one-to-one optical flows to capture both large and complex motions. In order to enhance interpolation performance for various textures, a 3D CNN is also employed to model the content dynamics over an extended temporal window. Moreover, ST-MFNet has been trained within an ST-GAN framework, which was originally developedfor texture synthesis, with the aim of further improving perceptual interpolation quality. Our approach has been comprehensively evaluated - compared with fourteen state-of-the-art VFI algorithms - clearly demonstrating that ST-MFNet consistently outperforms these benchmarks on var-ied and representative test datasets, with significant gains up to 1.09dB in PSNR for cases including large motions and dynamic textures. Our source code is available at https://github.com/danielism97/ST-MFNet.

----

## [345] Video Frame Interpolation with Transformer

**Authors**: *Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00352](https://doi.org/10.1109/CVPR52688.2022.00352)

**Abstract**:

Video frame interpolation (VFI), which aims to synthesize intermediate frames of a video, has made remarkable progress with development of deep convolutional networks over past years. Existing methods built upon convolutional networks generally face challenges of handling large motion due to the locality of convolution operations. To overcome this limitation, we introduce a novel framework, which takes advantage of Transformer to model long-range pixel correlation among video frames. Further, our network is equipped with a novel cross-scale window-based attention mechanism, where cross-scale windows interact with each other. This design effectively enlarges the receptive field and aggregates multi-scale information. Extensive quantitative and qualitative experiments demonstrate that our method achieves new state-of-the-art results on various benchmarks.

----

## [346] Long-term Video Frame Interpolation via Feature Propagation

**Authors**: *Dawit Mureja Argaw, In So Kweon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00353](https://doi.org/10.1109/CVPR52688.2022.00353)

**Abstract**:

Video frame interpolation (VFI) works generally predict intermediate frame(s) by first estimating the motion between inputs and then warping the inputs to the target time with the estimated motion. This approach, however, is not optimal when the temporal distance between the input sequence increases as existing motion estimation modules cannot effectively handle large motions. Hence, VFI works perform well for small frame gaps and perform poorly as the frame gap increases. In this work, we propose a novel framework to address this problem. We argue that when there is a large gap between inputs, instead of estimating imprecise motion that will eventually lead to inaccurate interpolation, we can safely propagate from one side of the input up to a reliable time frame using the other input as a reference. Then, the rest of the intermediate frames can be interpolated using standard approaches as the temporal gap is now narrowed. To this end, we propose a propagation network (PNet) by extending the classic feature-level forecasting with a novel motion-to-feature approach. To be thorough, we adopt a simple interpolation model along with PNet as our full model and design a simple procedure to train the full model in an end-to-end manner. Experimental results on several benchmark datasets confirm the effectiveness of our method for long-term VFI compared to state-of-the-art approaches.

----

## [347] Many-to-many Splatting for Efficient Video Frame Interpolation

**Authors**: *Ping Hu, Simon Niklaus, Stan Sclaroff, Kate Saenko*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00354](https://doi.org/10.1109/CVPR52688.2022.00354)

**Abstract**:

Motion-based video frame interpolation commonly relies on optical flow to warp pixels from the inputs to the desired interpolation instant. Yet due to the inherent challenges of motion estimation (e.g. occlusions and discontinuities), most state-of-the-art interpolation approaches require subsequent refinement of the warped result to generate satisfying outputs, which drastically decreases the efficiency for multi-frame interpolation. In this work, we propose a fully differentiable Many-to-Many (M2M) splatting framework to interpolate frames efficiently. Specifically, given a frame pair, we estimate multiple bidirectional flows to directly forward warp the pixels to the desired time step, and then fuse any overlapping pixels. In doing so, each source pixel renders multiple target pixels and each target pixel can be synthesized from a larger area of visual context. This establishes a many-to-many splatting scheme with robustness to artifacts like holes. Moreover, for each input frame pair, M2M only performs motion estimation once and has a minuscule computational overhead when interpolating an arbitrary number of in-between frames, hence achieving fast multi-frame interpolation. We conducted extensive experiments to analyze M2M, and found that it significantly improves the efficiency while maintaining high effectiveness.

----

## [348] Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image

**Authors**: *Xuanchi Ren, Xiaolong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00355](https://doi.org/10.1109/CVPR52688.2022.00355)

**Abstract**:

Novel view synthesis from a single image has recently attracted a lot of attention, and it has been primarily advanced by 3D deep learning and rendering techniques. However, most work is still limited by synthesizing new views within relatively small camera motions. In this paper, we propose a novel approach to synthesize a consistent long-term video given a single scene image and a trajectory of large camera motions. Our approach utilizes an autoregressive Transformer to perform sequential modeling of multiple frames, which reasons the relations between multiple frames and the corresponding cameras to predict the next frame. To facilitate learning and ensure consistency among generated frames, we introduce a locality constraint based on the input cameras to guide self-attention among a large number of patches across space and time. Our method outperforms state-of-the-art view synthesis approaches by a large margin, especially when synthesizing long-term future in indoor 3D scenes. Project page at https://xrenaa.github.io/look-outside-room/.

----

## [349] Spatial-Temporal Space Hand-in-Hand: Spatial-Temporal Video Super-Resolution via Cycle-Projected Mutual Learning

**Authors**: *Mengshun Hu, Kui Jiang, Liang Liao, Jing Xiao, Junjun Jiang, Zheng Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00356](https://doi.org/10.1109/CVPR52688.2022.00356)

**Abstract**:

Spatial-Temporal Video Super-Resolution (ST-VSR) aims to generate super-resolved videos with higher resolution (HR) and higher frame rate (HFR). Quite intuitively, pioneering two-stage based methods complete ST-VSR by directly combining two sub-tasks: Spatial Video Super-Resolution (S-VSR) and Temporal Video Super-Resolution (T-VSR) but ignore the reciprocal relations among them. Specifically, 1) T-VSR to S-VSR: temporal correlations help accurate spatial detail representation with more clues; 2) S-VSR to T-VSR: abundant spatial information contributes to the refinement of temporal prediction. To this end, we propose a one-stage based Cycle-projected Mutual learning network (CycMu-Net) for ST-VSR, which makes full use of spatial-temporal correlations via the mutual learning between S-VSR and T-VSR. Specifically, we propose to exploit the mutual information among them via iterative up-and-down projections, where the spatial and temporal features are fully fused and distilled, helping the high-quality video reconstruction. Besides extensive experiments on benchmark datasets, we also compare our proposed CycMu-Net with S-VSR and T-VSR tasks, demonstrating that our method significantly outperforms state-of-the-art methods. Codes are publicly available at: https://github.com/hhhhhumengshun/CycMuNet.

----

## [350] Playable Environments: Video Manipulation in Space and Time

**Authors**: *Willi Menapace, Stéphane Lathuilière, Aliaksandr Siarohin, Christian Theobalt, Sergey Tulyakov, Vladislav Golyanik, Elisa Ricci*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00357](https://doi.org/10.1109/CVPR52688.2022.00357)

**Abstract**:

We present Playable Environments-a new representation for interactive video generation and manipulation in space and time. With a single image at inference time, our novel framework allows the user to move objects in 3D while generating a video by providing a sequence of desired actions. The actions are learnt in an unsupervised manner. The camera can be controlled to get the desired viewpoint. Our method builds an environment state for each frame, which can be manipulated by our proposed action mod-ule and decoded back to the image space with volumetric rendering. To support diverse appearances of objects, we extend neural radiance fields with style-based modulation. Our method trains on a collection of various monocular videos requiring only the estimated camera parameters and 2D object locations. To set a challenging benchmark, we in-troduce two large scale video datasets with significant cam-era movements. As evidenced by our experiments, playable environments enable several creative applications not at-tainable by prior video synthesis works, including playable 3D video generation, stylization and manipulation
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
willi-menapace.github.io/playable-environments-website.

----

## [351] Event-based Video Reconstruction via Potential-assisted Spiking Neural Network

**Authors**: *Lin Zhu, Xiao Wang, Yi Chang, Jianing Li, Tiejun Huang, Yonghong Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00358](https://doi.org/10.1109/CVPR52688.2022.00358)

**Abstract**:

Neuromorphic vision sensor is a new bio-inspired imaging paradigm that reports asynchronous, continuously perpixel brightness changes called ‘events’ with high temporal resolution and high dynamic range. So far, the event-based image reconstruction methods are based on artificial neural networks (ANN) or hand-crafted spatiotemporal smoothing techniques. In this paper, we first implement the image reconstruction work via deep spiking neural network (SNN) architecture. As the bio-inspired neural networks, SNNs operating with asynchronous binary spikes distributed over time, can potentially lead to greater computational efficiency on event-driven hardware. We propose a novel Event-based Video reconstruction framework based on a fully Spiking Neural Network (EVSNN), which utilizes Leaky-Integrate-and-Fire (LIF) neuron and Membrane Potential (MP) neuron. We find that the spiking neurons have the potential to store useful temporal information (memory) to complete such time-dependent tasks. Further-more, to better utilize the temporal information, we propose a hybrid potential-assisted framework (PAEVSNN) using the membrane potential of spiking neuron. The proposed neuron is referred as Adaptive Membrane Potential (AMP) neuron, which adaptively updates the membrane potential according to the input spikes. The experimental results demonstrate that our models achieve comparable performance to ANN-based models on IJRR, MVSEC, and HQF datasets. The energy consumptions of EVSNN and PAEVSNN are 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$19.36\times$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$7.75\times$</tex>
 more computationally ef-ficient than their ANN architectures, respectively. The code and pretrained model are available at https://sites.google.com/view/evsnn.

----

## [352] Modular Action Concept Grounding in Semantic Video Prediction

**Authors**: *Wei Yu, Wenxin Chen, Songheng Yin, Steve Easterbrook, Animesh Garg*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00359](https://doi.org/10.1109/CVPR52688.2022.00359)

**Abstract**:

Recent works in video prediction have mainly focused on passive forecasting and low-level action-conditional pre-diction, which sidesteps the learning of interaction between agents and objects. We introduce the task of semantic action-conditional video prediction, which uses semantic action labels to describe those interactions and can be regarded as an inverse problem of action recognition. The challenge of this new task primarily lies in how to effectively inform the model of semantic action information. Inspired by the idea of Mixture of Experts, we embody each abstract label by a structured combination of various visual concept learn-ers and propose a novel video prediction model, Modular Action Concept Network (MAC). Our method is evaluated on two newly designed synthetic datasets, CLEVR-Building- Blocks and Sapien-Kitchen, and one real-world dataset called Tower-Creation. Extensive experiments demonstrate that MAC can correctly condition on given instructions and generate corresponding future frames without need of bounding boxes. We further show that the trained model can make out-of-distribution generalization, be quickly adapted to new object categories and exploit its learnt features for object detection, showing the progression towards higher-level cognitive abilities. More visualizations can be found at http://www.pair.toronto.edu/mac/.

----

## [353] Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning

**Authors**: *Ligong Han, Jian Ren, Hsin-Ying Lee, Francesco Barbieri, Kyle Olszewski, Shervin Minaee, Dimitris N. Metaxas, Sergey Tulyakov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00360](https://doi.org/10.1109/CVPR52688.2022.00360)

**Abstract**:

Most methods for conditional video synthesis use a single modality as the condition. This comes with major limitations. For example, it is problematic for a model conditioned on an image to generate a specific motion trajectory desired by the user since there is no means to provide motion information. Conversely, language information can describe the desired motion, while not precisely defining the content of the video. This work presents a multimodal video generation framework that benefits from text and images provided jointly or separately. We leverage the recent progress in quantized representations for videos and apply a bidirectional transformer with multiple modalities as inputs to predict a discrete video representation. To improve video quality and consistency, we propose a new video token trained with self-learning and an improved mask-prediction algorithm for sampling video tokens. We introduce text augmentation to improve the robustness of the textual representation and diversity of generated videos. Our framework can incorporate various visual modalities, such as segmentation masks, drawings, and partially occluded images. It can generate much longer sequences than the one used for training. In addition, our model can extract visual information as suggested by the text prompt, e.g., “an object in image one is moving northeast”, and generate corresponding videos. We run evaluations on three public datasets and a newly collected dataset labeled with facial attributes, achieving state-of-the-art generation results on all four
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/snap-research/MMVID and Webpage..

----

## [354] StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2

**Authors**: *Ivan Skorokhodov, Sergey Tulyakov, Mohamed Elhoseiny*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00361](https://doi.org/10.1109/CVPR52688.2022.00361)

**Abstract**:

Videos show continuous events, yet most - if not all - video synthesis frameworks treat them discretely in time. In this work, we think of videos of what they should be - time-continuous signals, and extend the paradigm of neural representations to build a continuous-time video generator. For this, we first design continuous motion representations through the lens of positional embeddings. Then, we explore the question of training on very sparse videos and demon-strate that a good generator can be learned by using as few as 2 frames per clip. After that, we rethink the traditional image + video discriminators pair and design a holistic dis-criminator that aggregates temporal information by simply concatenating frames' features. This decreases the training cost and provides richer learning signal to the generator, making it possible to train directly on 1024
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 videos for the first time. We build our model on top of StyleGAN2 and it is just ≈5% more expensive to train at the same resolution while achieving almost the same image quality. Moreover, our latent space features similar properties, enabling spa-tial manipulations that our method can propagate in time. We can generate arbitrarily long videos at arbitrary high frame rate, while prior work struggles to generate even 64 frames at a fixed rate. Our model is tested on four mod-ern 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 and one 1024
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 -resolution video synthesis bench-marks. In terms of sheer metrics, it performs on average ≈30% better than the closest runner-up. Project website: https://universome.github.io/stylegan-v.

----

## [355] Structure-Aware Motion Transfer with Deformable Anchor Model

**Authors**: *Jiale Tao, Biao Wang, Borun Xu, Tiezheng Ge, Yuning Jiang, Wen Li, Lixin Duan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00362](https://doi.org/10.1109/CVPR52688.2022.00362)

**Abstract**:

Given a source image and a driving video depicting the same object type, the motion transfer task aims to generate a video by learning the motion from the driving video while preserving the appearance from the source image. In this paper, we propose a novel structure-aware motion modeling approach, the deformable anchor model (DAM), which can automatically discover the motion structure of arbitrary objects without leveraging their prior structure information. Specifically, inspired by the known deformable part model (DPM), our DAM introduces two types of anchors or key-points: i) a number of motion anchors that capture both appearance and motion information from the source image and driving video; ii) a latent root anchor, which is linked to the motion anchors to facilitate better learning of the representations of the object structure information. More-over, DAM can be further extended to a hierarchical version through the introduction of additional latent anchors to model more complicated structures. By regularizing motion anchors with latent anchor(s), DAM enforces the corre-spondences between them to ensure the structural information is well captured and preserved. Moreover, DAM can be learned effectively in an unsupervised manner. We validate our proposed DAM for motion transfer on different bench-mark datasets. Extensive experiments clearly demonstrate that DAM achieves superior performance relative to existing state-of-the-art methods.

----

## [356] Image Animation with Perturbed Masks

**Authors**: *Yoav Shalev, Lior Wolf*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00363](https://doi.org/10.1109/CVPR52688.2022.00363)

**Abstract**:

We present a novel approach for image-animation of a source image by a driving video, both depicting the same type of object. We do not assume the existence of pose models and our method is able to animate arbitrary objects without the knowledge of the object's structure. Furthermore, both, the driving video and the source image are only seen during test-time. Our method is based on a shared mask generator, which separates the foreground object from its background, and captures the object's general pose and shape. To control the source of the identity of the output frame, we employ perturbations to interrupt the unwanted identity information on the driver's mask. A mask-refinement module then replaces the identity of the driver with the identity of the source. Conditioned on the source image, the transformed mask is then decoded by a multi-scale generator that renders a realistic image, in which the content of the source frame is animated by the pose in the driving video. Due to the lack of fully supervised data, we train on the task of reconstructing frames from the same video the source image is taken from. Our method is shown to greatly outperform the state-of-the-art methods on multiple benchmarks. Our code and samples are available at https://github.com/itsyoavshalevlImage-Animation-with-Perturbed-Masks.

----

## [357] Thin-Plate Spline Motion Model for Image Animation

**Authors**: *Jian Zhao, Hui Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00364](https://doi.org/10.1109/CVPR52688.2022.00364)

**Abstract**:

Image animation brings life to the static object in the source image according to the driving video. Recent works attempt to perform motion transfer on arbitrary objects through unsupervised methods without using a priori knowledge. However, it remains a significant challenge for current unsupervised methods when there is a large pose gap between the objects in the source and driving images. In this paper, a new end-to-end unsupervised motion transfer framework is proposed to overcome such issues. Firstly, we propose thin-plate spline motion estimation to produce a more flexible optical flow, which warps the feature maps of the source image to the feature domain of the driving image. Secondly, in order to restore the missing regions more realistically, we leverage multi-resolution occlusion masks to achieve more effective feature fusion. Finally, additional auxiliary loss functions are designed to ensure that there is a clear division of labor in the network modules, encouraging the network to generate high-quality images. Our method
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our source code is publicly available: https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model. can animate a variety of objects, including talking faces, human bodies, and pixel animations. Experiments demonstrate that our method performs better on most benchmarks than the state of the art with visible improvements in motion-related metrics.

----

## [358] Controllable Animation of Fluid Elements in Still Images

**Authors**: *Aniruddha Mahapatra, Kuldeep Kulkarni*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00365](https://doi.org/10.1109/CVPR52688.2022.00365)

**Abstract**:

We propose a method to interactively control the animation of fluid elements in still images to generate cinemagraphs. Specifically, we focus on the animation of fluid elements like water, smoke, fire, which have the properties of repeating textures and continuous fluid motion. Taking inspiration from prior works, we represent the motion of such fluid elements in the image in the form of a constant 2D optical flow map. To this end, we allow the user to provide any number of arrow directions and their associated speeds along with a mask of the regions the user wants to animate. The user-provided input arrow directions, their corresponding speed values, and the mask are then converted into a dense flow map representing a constant optical flow map (F
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">D</inf>
). We observe that F
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">D</inf>
, obtained using simple exponential operations can closely approximate the plausible motion of elements in the image. We further refine computed dense optical flow map F
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">D</inf>
 using a generative-adversarial network (GAN) to obtain a more realistic flow map. We devise a novel UNet based architecture to autoregressively generate future frames using the refined optical flow map by forward-warping the input image features at different resolutions. We conduct extensive experiments on a publicly available dataset and show that our method is superior to the baselines in terms of qualitative and quantitative metrics. In addition, we show the qualitative animations of the objects in directions that did not exist in the training set and provide a way to synthesize videos that otherwise would not exist in the real world. Project url: https://controllable-cinemagraphs.github.io/

----

## [359] Watch It Move: Unsupervised Discovery of 3D Joints for Re-Posing of Articulated Objects

**Authors**: *Atsuhiro Noguchi, Umar Iqbal, Jonathan Tremblay, Tatsuya Harada, Orazio Gallo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00366](https://doi.org/10.1109/CVPR52688.2022.00366)

**Abstract**:

Rendering articulated objects while controlling their poses is critical to applications such as virtual reality or animation for movies. Manipulating the pose of an object, however, requires the understanding of its underlying structure, that is, its joints and how they interact with each other. Unfortunately, assuming the structure to be known, as existing methods do, precludes the ability to work on new object categories. We propose to learn both the appearance and the structure of previously unseen articulated objects by ob-serving them move from multiple views, with no joints annotation supervision, or information about the structure. We observe that 3D points that are static relative to one another should belong to the same part, and that adjacent parts that move relative to each other must be connected by a joint. To leverage this insight, we model the object parts in 3D as ellipsoids, which allows us to identify joints. We combine this explicit representation with an implicit one that compensates for the approximation introduced. We show that our method works for different structures, from quadrupeds, to single-arm robots, to humans. The code is available at https://github.com/NVlabs/watch-it-move and a version of this manuscript that uses animations is at https://arxiv.org/abs/2112.11347

----

## [360] Geometric Structure Preserving Warp for Natural Image Stitching

**Authors**: *Peng Du, Jifeng Ning, Jiguang Cui, Shaoli Huang, Xinchao Wang, Jiaxin Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00367](https://doi.org/10.1109/CVPR52688.2022.00367)

**Abstract**:

Preserving geometric structures in the scene plays a vital role in image stitching. However, most of the existing methods ignore the large-scale layouts reflected by straight lines or curves, decreasing overall stitching quality. To address this issue, this work presents a structure-preserving stitching approach that produces images with natural visual effects and less distortion. Our method first employs deep learning-based edge detection to extract various types of large-scale edges. Then, the extracted edges are sampled to construct multiple groups of triangles to represent geometric structures. Meanwhile, a GEometric Structure preserving (GES) energy term is introduced to make these triangles undergo similarity transformation. Further, an optimized GES energy term is presented to reasonably determine the weights of the sampling points on the geometric structure, and the term is added into the Global Similarity Prior (GSP) stitching model called GES-GSP to achieve a smooth transition between local alignment and geometric structure preservation. The effectiveness of GES-GSP is validated through comprehensive experiments on a stitching dataset. The experimental results show that the proposed method outperforms several state-of-the-art methods in geometric structure preservation and obtains more natural stitching results. The code and dataset are available at https://github.com/flowerDuo/GES-GSP-Stitching.

----

## [361] Few-Shot Incremental Learning for Label-to-Image Translation

**Authors**: *Pei Chen, Yangkang Zhang, Zejian Li, Lingyun Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00368](https://doi.org/10.1109/CVPR52688.2022.00368)

**Abstract**:

Label-to-image translation models generate images from semantic label maps. Existing models depend on large volumes of pixel-level annotated samples. When given new training samples annotated with novel semantic classes, the models should be trained from scratch with both learned and new classes. This hinders their practical applications and motivates us to introduce an incremental learning strategy to the label-to-image translation scenario. In this paper, we introduce a few-shot incremental learning method for label-to-image translation. It learns new classes one by one from a few samples of each class. We propose to adopt semantically-adaptive convolution filters and normalization. When incrementally trained on a novel semantic class, the model only learns a few extra parameters of class-specific modulation. Such design avoids catastrophic forgetting of already-learned semantic classes and enables label-to-image translation of scenes with increasingly rich content. Furthermore, to facilitate few-shot learning, we propose a modulation transfer strategy for better initialization. Extensive experiments show that our method outperforms existing related methods in most cases and achieves zero forgetting.

----

## [362] Exemplar-based Pattern Synthesis with Implicit Periodic Field Network

**Authors**: *Haiwei Chen, Jiayi Liu, Weikai Chen, Shichen Liu, Yajie Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00369](https://doi.org/10.1109/CVPR52688.2022.00369)

**Abstract**:

Synthesis of ergodic, stationary visual patterns is widely applicable in texturing, shape modeling, and digital content creation. The wide applicability of this technique thus requires the pattern synthesis approaches to be scalable, diverse, and authentic. In this paper, we propose an exemplar-based visual pattern synthesis framework that aims to model the inner statistics of visual patterns and generate new, versatile patterns that meet the aforementioned requirements. To this end, we propose an implicit network based on generative adversarial network (GAN) and periodic encoding, thus calling our network the Implicit Periodic Field Network (IPFN). The design of IPFN ensures scalability: the implicit formulation directly maps the input coordinates to features, which enables synthesis of arbitrary size and is computationally efficient for 3D shape synthesis. Learning with a periodic encoding scheme encourages diversity: the network is constrained to model the inner statistics of the exemplar based on spatial latent codes in a periodic field. Coupled with continuously designed GAN training procedures, IPFN is shown to synthesize tileable patterns with smooth transitions and local variations. Last but not least, thanks to both the adversarial training technique and the encoded Fourier features, IPFN learns high-frequency functions that produce authentic, high-quality results. To validate our approach, we present novel experimental results on various applications in 2D texture synthesis and 3D shape synthesis.

----

## [363] SIMBAR: Single Image-Based Scene Relighting For Effective Data Augmentation For Automated Driving Vision Tasks

**Authors**: *Xianling Zhang, Nathan Tseng, Ameerah Syed, Rohan Bhasin, Nikita Jaipuria*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00370](https://doi.org/10.1109/CVPR52688.2022.00370)

**Abstract**:

Real-world autonomous driving datasets comprise of images aggregated from different drives on the road. The ability to relight captured scenes to unseen lighting conditions, in a controllable manner, presents an opportunity to augment datasets with a richer variety of lighting conditions, similar to what would be encountered in the real-world. This paper presents a novel image-based relighting pipeline, SIMBAR, that can work with a single image as input. To the best of our knowledge, there is no prior work on scene relighting leveraging explicit geometric representations from a single image. We present qualitative comparisons with prior multi-view scene relighting baselines. To further validate and effectively quantify the benefit of leveraging SIMBAR for data augmentation for automated driving vision tasks, object detection and tracking experiments are conducted with a state-of-the-art method, a Multiple Object Tracking Accuracy (MOTA) of 93.3% is achieved with CenterTrack on SIMBAR-augmented KITTI - an impressive 9.0% relative improvement over the baseline MOTA of 85.6% with CenterTrack on original KITTI, both models trained from scratch and tested on Virtual KITTI. For more details and sample relit datasets, please visit our project website (https://simbarv1.github.io).

----

## [364] SoftCollage: A Differentiable Probabilistic Tree Generator for Image Collage

**Authors**: *Jiahao Yu, Li Chen, Mingrui Zhang, Mading Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00371](https://doi.org/10.1109/CVPR52688.2022.00371)

**Abstract**:

Image collage task aims to create an informative and visual-aesthetic visual summarization for an image collection. While several recent works exploit tree-based algorithm to preserve image content better, all of them resort to hand-crafted adjustment rules to optimize the collage tree structure, leading to the failure of fully exploring the structure space of collage tree. Our key idea is to soften the discrete tree structure space into a continuous probability space. We propose SoftCollage, a novel method that employs a neural-based differentiable probabilistic tree generator to produce the probability distribution of correlation-preserving collage tree conditioned on deep image feature, aspect ratio and canvas size. The differentiable characteristic allows us to formulate the tree-based collage generation as a differentiable process and directly exploit gradient to optimize the collage layout in the level of probability space in an end-to-end manner. To facilitate image collage research, we propose AIC, a large-scale public-available annotated dataset for image collage evaluation. Extensive experiments on the introduced dataset demonstrate the superior performance of the proposed method. Data and codes are available at https://github.com/ChineseYjh/SoftCollage.

----

## [365] PILC: Practical Image Lossless Compression with an End-to-end GPU Oriented Neural Framework

**Authors**: *Ning Kang, Shanzhao Qiu, Shifeng Zhang, Zhenguo Li, Shutao Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00372](https://doi.org/10.1109/CVPR52688.2022.00372)

**Abstract**:

Generative model based image lossless compression algorithms have seen a great success in improving compression ratio. However, the throughput for most of them is less than 1 MB/s even with the most advanced AI accelerated chips, preventing them from most real-world applications, which often require 100 MB/s. In this paper, we propose PILC, an end-to-end image lossless compression framework that achieves 200 MB/s for both compression and decom-pression with a single NVIDIA Tesla V100 GPU, 10× faster than the most efficient one before. To obtain this result, we first develop an AI codec that combines auto-regressive model and VQ-VAE which performs well in lightweight setting, then we design a low complexity entropy coder that works well with our codec. Experiments show that our framework compresses better than PNG by a margin of 30% in multiple datasets. We believe this is an important step to bring AI compression forward to commercial use.

----

## [366] Kubric: A scalable dataset generator

**Authors**: *Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J. Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, Thomas Kipf, Abhijit Kundu, Dmitry Lagun, Issam H. Laradji, Hsueh-Ti Derek Liu, Henning Meyer, Yishu Miao, Derek Nowrouzezahrai, A. Cengiz Öztireli, Etienne Pot, Noha Radwan, Daniel Rebain, Sara Sabour, Mehdi S. M. Sajjadi, Matan Sela, Vincent Sitzmann, Austin Stone, Deqing Sun, Suhani Vora, Ziyu Wang, Tianhao Wu, Kwang Moo Yi, Fangcheng Zhong, Andrea Tagliasacchi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00373](https://doi.org/10.1109/CVPR52688.2022.00373)

**Abstract**:

Data is the driving force of machine learning, with the amount and quality of training data often being more important for the performance of a system than architecture and training details. But collecting, processing and annotating real data at scale is difficult, expensive, and frequently raises additional privacy, fairness and legal concerns. Synthetic data is a powerful tool with the potential to address these shortcomings: 1) it is cheap 2) supports rich ground-truth annotations 3) offers full control over data and 4) can circumvent or mitigate problems regarding bias, privacy and licensing. Unfortunately, software tools for effective data generation are less mature than those for architecture design and training, which leads to fragmented generation efforts. To address these problems we introduce Kubric, an open-source Python framework that interfaces with PyBullet and Blender to generate photo-realistic scenes, with rich annotations, and seamlessly scales to large jobs distributed over thousands of machines, and generating TBs of data. We demonstrate the effectiveness of Kubric by presenting a series of 13 different generated datasets for tasks ranging from studying 3D NeRF models to optical flow estimation. We release Kubric, the used assets, all of the generation code, as well as the rendered datasets for reuse and modification.

----

## [367] 360MonoDepth: High-Resolution 360° Monocular Depth Estimation

**Authors**: *Manuel Rey-Area, Mingze Yuan, Christian Richardt*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00374](https://doi.org/10.1109/CVPR52688.2022.00374)

**Abstract**:

360° cameras can capture complete environments in a single shot, which makes 360° imagery alluring in many computer vision tasks. However, monocular depth estimation remains a challenge for 360° data, particularly for high resolutions like 2K (2048 × 1 024) and beyond that are important for novel-view synthesis and virtual reality applications. Current CNN-based methods do not support such high resolutions due to limited GPU memory. In this work, we propose aflexible framework for monocular depth estimation from high-resolution 360° images using tangent images. We project the 360° input image onto a set of tangent planes that produce perspective views, which are suitable for the latest, most accurate state-of-the-art perspective monocular depth estimators. To achieve globally consistent disparity estimates, we recombine the individual depth estimates using deformable multi-scale alignment followed by gradient-domain blending. The result is a dense, high-resolution 360° depth map with a high level of detail, also for outdoor scenes which are not supported by existing methods. Our source code and data are available at https://manurare.github.io/360monodepth/.

----

## [368] Pretrain, Self-train, Distill: A simple recipe for Supersizing 3D Reconstruction

**Authors**: *Kalyan Vasudev Alwala, Abhinav Gupta, Shubham Tulsiani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00375](https://doi.org/10.1109/CVPR52688.2022.00375)

**Abstract**:

Our work learns a unified model for single-view 3D reconstruction of objects from hundreds of semantic categories. As a scalable alternative to direct 3D supervision, our work relies on segmented image collections for learning 3D of generic categories. Unlike prior works that use similar supervision but learn independent category-specific models from scratch, our approach of learning a unified model simplifies the training process while also allowing the model to benefit from the common structure across categories. Using image collections from standard recognition datasets, we show that our approach allows learning 3D inference for over 150 object categories. We evaluate using two datasets and qualitatively and quantitatively show that our unified reconstruction approach improves over prior category-specific reconstruction baselines. Our final 3D reconstruction model is also capable of zero-shot inference on images from unseen object categories and we empirically show that increasing the number of training categories improves the reconstruction quality.

----

## [369] DGECN: A Depth-Guided Edge Convolutional Network for End-to-End 6D Pose Estimation

**Authors**: *Tuo Cao, Fei Luo, Yanping Fu, Wenxiao Zhang, Shengjie Zheng, Chunxia Xiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00376](https://doi.org/10.1109/CVPR52688.2022.00376)

**Abstract**:

Monocular 6D pose estimation is a fundamental task in computer vision. Existing works often adopt a two-stage pipeline by establishing correspondences and utilizing a RANSAC algorithm to calculate 6 degrees-of-freedom (6DoF) pose. Recent works try to integrate differentiable RANSAC algorithms to achieve an end-to-end 6D pose estimation. However, most of them hardly consider the geometric features in 3D space, and ignore the topology cues when performing differentiable RANSAC algorithms. To this end, we proposed a Depth-Guided Edge Convolutional Network (DGECN) for 6D pose estimation task. We have made efforts from the following three aspects: 1) We take advantages of estimated depth information to guide both the correspondences-extraction process and the cascaded differentiable RANSAC algorithm with geometric information. 2) We leverage the uncertainty of the estimated depth map to improve accuracy and robustness of the output 6D pose. 3) We propose a differentiable Perspective-n-Point(PnP) algorithm via edge convolution to explore the topology relations between 2D-3D correspondences. Experiments demonstrate that our proposed network outperforms current works on both effectiveness and efficiency.

----

## [370] MonoGround: Detecting Monocular 3D Objects from the Ground

**Authors**: *Zequn Qin, Xi Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00377](https://doi.org/10.1109/CVPR52688.2022.00377)

**Abstract**:

Monocular 3D object detection has attracted great attention for its advantages in simplicity and cost. Due to the ill-posed 2D to 3D mapping essence from the monocular imaging process, monocular 3D object detection suffers from inaccurate depth estimation and thus has poor 3D detection results. To alleviate this problem, we propose to introduce the ground plane as a prior in the monocular 3d object detection. The ground plane prior serves as an additional geometric condition to the ill-posed mapping and an extra source in depth estimation. In this way, we can get a more accurate depth estimation from the ground. Meanwhile, to take full advantage of the ground plane prior, we propose a depth-align training strategy and a precise two-stage depth inference method tailored for the ground plane prior. It is worth noting that the introduced ground plane prior requires no extra data sources like LiDAR, stereo images, and depth information. Extensive experiments on the KITTI benchmark show that our method could achieve state-of-the-art results compared with other methods while maintaining a very fast speed. Our code, models, and training logs are available at https://github.com/cfzd/MonoGround.

----

## [371] 3D Shape Reconstruction from 2D Images with Disentangled Attribute Flow

**Authors**: *Xin Wen, Junsheng Zhou, Yu-Shen Liu, Hua Su, Zhen Dong, Zhizhong Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00378](https://doi.org/10.1109/CVPR52688.2022.00378)

**Abstract**:

Reconstructing 3D shape from a single 2D image is a challenging task, which needs to estimate the detailed 3D structures based on the semantic attributes from 2D image. So far, most of the previous methods still struggle to extract semantic attributes for 3D reconstruction task. Since the semantic attributes of a single image are usually implicit and entangled with each other, it is still challenging to reconstruct 3D shape with detailed semantic structures represented by the input image. To address this problem, we propose 3DAttriFlow to disentangle and extract semantic attributes through different semantic levels in the input images. These disentangled semantic attributes will be integrated into the 3D shape reconstruction process, which can provide definite guidance to the reconstruction of specific attribute on 3D shape. As a result, the 3D decoder can explicitly capture high-level semantic features at the bottom of the network, and utilize low-level features at the top of the network, which allows to reconstruct more accurate 3D shapes. Note that the explicit disentangling is learned without extra labels, where the only supervision used in our training is the input image and its corresponding 3D shape. Our comprehensive experiments on ShapeNet dataset demonstrate that 3DAttriFlow outperforms the state-of-the-art shape reconstruction methods, and we also validate its generalization ability on shape completion task. Code is available at https://github.com/junshengzhou/3DAttriFlow.

----

## [372] Toward Practical Monocular Indoor Depth Estimation

**Authors**: *Cho-Ying Wu, Jialiang Wang, Michael Hall, Ulrich Neumann, Shuochen Su*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00379](https://doi.org/10.1109/CVPR52688.2022.00379)

**Abstract**:

The majority of prior monocular depth estimation meth-ods without groundtruth depth guidance focus on driving scenarios. We show that such methods generalize poorly to unseen complex indoor scenes, where objects are cluttered and arbitrarily arranged in the near field. To obtain more robustness, we propose a structure distillation approach to learn knacks from an off-the-shelf relative depth estima-tor that produces structured but metric-agnostic depth. By combining structure distillation with a branch that learns metrics from left-right consistency, we attain structured and metric depth for generic indoor scenes and make inferences in real-time. To facilitate learning and evaluation, we col-lect SimSIN, a dataset from simulation with thousands of environments, and UniSIN, a dataset that contains about 500 real scan sequences of generic indoor environments. We experiment in both sim-to-real and real-to-real settings, and show improvements, as well as in downstream applications using our depth maps. This work provides a full study, covering methods, data, and applications aspects.

----

## [373] Focal Length and Object Pose Estimation via Render and Compare

**Authors**: *Georgy Ponimatkin, Yann Labbé, Bryan C. Russell, Mathieu Aubry, Josef Sivic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00380](https://doi.org/10.1109/CVPR52688.2022.00380)

**Abstract**:

We introduce FocalPose, a neural render-and-compare method for jointly estimating the camera-object 6D pose and camera focal length given a single RGB input image depicting a known object. The contributions of this work are twofold. First, we derive a focal length update rule that extends an existing state-of-the-art render-and-compare 6D pose estimator to address the joint estimation task. Second, we investigate several different loss functions for jointly estimating the object pose and focal length. We find that a combination of direct focal length regression with a reprojection loss disentangling the contribution of translation, rotation, and focal length leads to improved results. We show results on three challenging benchmark datasets that depict known 3D models in uncontrolled settings. We demonstrate that our focal length and 6D pose estimates have lower error than the existing state-of-the-art methods.

----

## [374] CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields

**Authors**: *Can Wang, Menglei Chai, Mingming He, Dongdong Chen, Jing Liao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00381](https://doi.org/10.1109/CVPR52688.2022.00381)

**Abstract**:

We present CLIP-NeRF, a multi-modal 3D object manipulation method for neural radiance fields (NeRF). By leveraging the joint language-image embedding space of the recent Contrastive Language-Image Pre-Training (CLIP) model, we propose a unified framework that allows manip-ulating NeRF in a user-friendly way, using either a short text prompt or an exemplar image. Specifically, to combine the novel view synthesis capability of NeRF and the controllable manipulation ability of latent representations from generative models, we introduce a disentangled conditional NeRF architecture that allows individual control over both shape and appearance. This is achieved by performing the shape conditioning via applying a learned deformation field to the positional encoding and deferring color conditioning to the volumetric rendering stage. To bridge this disentangled latent representation to the CLIP embedding, we design two code mappers that take a CLIP embedding as input and update the latent codes to reflect the targeted editing. The mappers are trained with a CLIP-based matching loss to ensure the manipulation accuracy. Furthermore, we propose an inverse optimization method that accurately projects an input image to the latent codes for manipulation to enable editing on real images. We evaluate our approach by extensive experiments on a variety of text prompts and exemplar images and also provide an intuitive interface for interactive editing.

----

## [375] Registering Explicit to Implicit: Towards High-Fidelity Garment mesh Reconstruction from Single Images

**Authors**: *Heming Zhu, Lingteng Qiu, Yuda Qiu, Xiaoguang Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00382](https://doi.org/10.1109/CVPR52688.2022.00382)

**Abstract**:

Fueled by the power of deep learning techniques and implicit shape learning, recent advances in single-image human digitalization have reached unprecedented accuracy and could recover fine-grained surface details such as garment wrinkles. However, a common problem for the implicit-based methods is that they cannot produce separated and topology-consistent mesh for each garment piece, which is crucial for the current 3D content creation pipeline. To address this issue, we proposed a novel geometry inference framework ReEF that reconstructs topology-consistent layered garment mesh by registering the explicit garment template to the whole-body implicit fields predicted from single images. Experiments demonstrate that our method notably outperforms the counterparts on single-image layered garment reconstruction and could bring high-quality digital assets for further content creation.

----

## [376] Layered Depth Refinement with Mask Guidance

**Authors**: *Soo Ye Kim, Jianming Zhang, Simon Niklaus, Yifei Fan, Simon Chen, Zhe Lin, Munchurl Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00383](https://doi.org/10.1109/CVPR52688.2022.00383)

**Abstract**:

Depth maps are used in a wide range of applications from 3D rendering to 2D image effects such as Bokeh. However, those predicted by single image depth estimation (SIDE) models often fail to capture isolated holes in objects and/or have inaccurate boundary regions. Meanwhile, high-quality masks are much easier to obtain, using commercial auto-masking tools or off-the-shelf methods of segmentation and matting or even by manual editing. Hence, in this paper, we formulate a novel problem of mask-guided depth refinement that utilizes a generic mask to refine the depth prediction of SIDE models. Our framework performs layered refinement and inpainting/outpainting, decomposing the depth map into two separate layers signified by the mask and the inverse mask. As datasets with both depth and mask annotations are scarce, we propose a self-supervised learning scheme that uses arbitrary masks and RGB-D datasets. We empirically show that our method is robust to different types of masks and initial depth predictions, accurately refining depth values in inner and outer mask boundary regions. We further analyze our model with an ablation study and demonstrate results on real applications. More information can be found on our project page.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://sooyekim.github.io/MaskDepth/

----

## [377] HEAT: Holistic Edge Attention Transformer for Structured Reconstruction

**Authors**: *Jiacheng Chen, Yiming Qian, Yasutaka Furukawa*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00384](https://doi.org/10.1109/CVPR52688.2022.00384)

**Abstract**:

This paper presents a novel attention-based neural net-workfor structured reconstruction, which takes a 2D raster image as an input and reconstructs a planar graph depicting an underlying geometric structure. The approach detects corners and classifies edge candidates between corners in an end-to-end manner. Our contribution is a holistic edge clas-sification architecture, which 1) initializes the feature of an edge candidate by a trigonometric positional encoding of its end-points; 2) fuses image feature to each edge candidate by deformable attention; 3) employs two weight-sharing Trans-former decoders to learn holistic structural patterns over the graph edge candidates; and 4) is trained with a masked learning strategy. The corner detector is a variant of the edge classification architecture, adapted to operate on pixels as corner candidates. We conduct experiments on two structured reconstruction tasks: outdoor building architecture and indoor fioorplan planar graph reconstruction. Exten-sive qualitative and quantitative evaluations demonstrate the superiority of our approach over the state of the art. Code and pre-trained models are available at https://heat-structured-reconstruction.github.io/

----

## [378] BARC: Learning to Regress 3D Dog Shape from Images by Exploiting Breed Information

**Authors**: *Nadine Rüegg, Silvia Zuffi, Konrad Schindler, Michael J. Black*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00385](https://doi.org/10.1109/CVPR52688.2022.00385)

**Abstract**:

Our goal is to recover the 3D shape and pose of dogs from a single image. This is a challenging task because dogs exhibit a wide range of shapes and appearances, and are highly articulated. Recent work has proposed to directly regress the SMAL animal model, with additional limb scale parameters, from images. Our method, called BARC (Breed-Augmented Regression using Classification), goes beyond prior work in several important ways. First, we modify the SMAL shape space to be more appropriate for representing dog shape. But, even with a better shape model, the problem of regressing dog shape from an image is still challenging because we lack paired images with 3D ground truth. To compensate for the lack of paired data, we formulate novel losses that exploit information about dog breeds. In particular, we exploit the fact that dogs of the same breed have similar body shapes. We formulate a novel breed similarity loss consisting of two parts: One term encourages the shape of dogs from the same breed to be more similar than dogs of different breeds. The second one, a breed classification loss, helps to produce recognizable breed-specific shapes. Through ablation studies, we find that our breed losses significantly improve shape accuracy over a baseline without them. We also compare BARC qualitatively to WLDO with a perceptual study and find that our approach produces dogs that are significantly more realistic. This work shows that a-priori information about genetic similarity can help to compensate for the lack of 3D training data. This concept may be applicable to other animal species or groups of species. Our code is publicly available for research purposes at https://barc.is.tue.mpg.de/.

----

## [379] Time3D: End-to-End Joint Monocular 3D Object Detection and Tracking for Autonomous Driving

**Authors**: *Peixuan Li, Jieyu Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00386](https://doi.org/10.1109/CVPR52688.2022.00386)

**Abstract**:

While separately leveraging monocular 3D object detection and 2D multi-object tracking can be straightforwardly applied to sequence images in a frame-by-frame fashion, stand-alone tracker cuts off the transmission of the uncertainty from the 3D detector to tracking while cannot pass tracking error differentials back to the 3D detector. In this work, we propose jointly training 3D detection and 3D tracking from only monocular videos in an end-to-end manner. The key component is a novel spatial-temporal information flow module that aggregates geometric and appearance features to predict robust similarity scores across all objects in current and past frames. Specifically, we leverage the attention mechanism of the transformer, in which self-attention aggregates the spatial information in a specific frame, and cross-attention exploits relation and affinities of all objects in the temporal domain of sequence frames. The affinities are then supervised to estimate the trajectory and guide the flow of information between corresponding 3D objects. In addition, we propose a temporal -consistency loss that explicitly involves 3D target motion modeling into the learning, making the 3D trajectory smooth in the world coordinate system. Time3D achieves 21.4% AMOTA, 13.6% AMOTP on the nuScenes 3D tracking benchmark, surpassing all published competitors, and running at 38 FPS, while Time3D achieves 31.2% mAP, 39.4% NDS on the nuScenes 3D detection benchmark.

----

## [380] What's in your hands? 3D Reconstruction of Generic Objects in Hands

**Authors**: *Yufei Ye, Abhinav Gupta, Shubham Tulsiani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00387](https://doi.org/10.1109/CVPR52688.2022.00387)

**Abstract**:

Our work aims to reconstruct hand-held objects given a single RGB image. In contrast to prior works that typically assume known 3D templates and reduce the problem to 3D pose estimation, our work reconstructs generic hand-held object without knowing their 3D templates. Our key insight is that hand articulation is highly predictive of the object shape, and we propose an approach that conditionally reconstructs the object based on the articulation and the visual input. Given an image depicting a hand-held object, we first use off-the-shelf systems to estimate the underlying hand pose and then infer the object shape in a normalized hand-centric coordinate frame. We parameterized the object by signed distance which are inferred by an implicit network which leverages the information from both visual feature and articulation-aware coordinates to process a query point. We perform experiments across three datasets and show that our method consistently outperforms baselines and is able to reconstruct a diverse set of objects. We analyze the benefits and robustness of explicit articulation conditioning and also show that this allows the hand pose estimation to further improve in test-time optimization.

----

## [381] 3D Moments from Near-Duplicate Photos

**Authors**: *Qianqian Wang, Zhengqi Li, David Salesin, Noah Snavely, Brian Curless, Janne Kontkanen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00388](https://doi.org/10.1109/CVPR52688.2022.00388)

**Abstract**:

We introduce 3D Moments, a new computational photography effect. As input we take a pair of near-duplicate photos, i.e., photos of moving subjects from similar viewpoints, common in people's photo collections. As output, we produce a video that smoothly interpolates the scene motion from the first photo to the second, while also producing camera motion with parallax that gives a heightened sense of 3D. To achieve this effect, we represent the scene as a pair of feature-based layered depth images augmented with scene flow. This representation enables motion interpolation along with independent control of the camera viewpoint. Our system produces photorealistic space-time videos with motion parallax and scene dynamics, while plausibly recovering regions occluded in the original views. We conduct extensive experiments demonstrating superior performance over baselines on public datasets and in-the-wild photos. Project page: https://3d-moments.github.io/.

----

## [382] Neural Window Fully-connected CRFs for Monocular Depth Estimation

**Authors**: *Weihao Yuan, Xiaodong Gu, Zuozhuo Dai, Siyu Zhu, Ping Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00389](https://doi.org/10.1109/CVPR52688.2022.00389)

**Abstract**:

Estimating the accurate depth from a single image is challenging since it is inherently ambiguous and ill-posed. While recent works design increasingly complicated and powerful networks to directly regress the depth map, we take the path of CRFs optimization. Due to the expensive computation, CRFs are usually performed between neighborhoods rather than the whole graph. To leverage the potential of fully-connected CRFs, we split the input into windows and perform the FC-CRFs optimization within each window, which reduces the computation complexity and makes FC-CRFs feasible. To better capture the relationships between nodes in the graph, we exploit the multi-head attention mechanism to compute a multi-head potential function, which is fed to the networks to output an optimized depth map. Then we build a bottom-up-top-down structure, where this neural window FC-CRFs module serves as the decoder, and a vision transformer serves as the encoder. The experiments demonstrate that our method significantly improves the performance across all metrics on both the KITTI and NYUv2 datasets, compared to previous methods. Furthermore, the proposed method can be directly applied to panorama images and outperforms all previous panorama methods on the MatterPort3D dataset.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://weihaosky.github.io/newcrfs

----

## [383] PUMP: Pyramidal and Uniqueness Matching Priors for Unsupervised Learning of Local Descriptors

**Authors**: *Jérôme Revaud, Vincent Leroy, Philippe Weinzaepfel, Boris Chidlovskii*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00390](https://doi.org/10.1109/CVPR52688.2022.00390)

**Abstract**:

Existing approaches for learning local image descriptors have shown remarkable achievements in a wide range of geometric tasks. However, most of them require perpixel correspondence-level supervision, which is difficult to acquire at scale and in high quality. In this paper, we propose to explicitly integrate two matching priors in a single loss in order to learn local descriptors without supervision. Given two images depicting the same scene, we extract pixel descriptors and build a correlation volume. The first prior enforces the local consistency of matches in this volume via a pyramidal structure iteratively constructed using a non-parametric module. The second prior exploits the fact that each descriptor should match with at most one descriptor from the other image. We combine our unsupervised loss with a standard self-supervised loss trained from synthetic image augmentations. Feature descriptors learned by the proposed approach outperform their fully- and self-supervised counterparts on various geometric benchmarks such as visual localization and image matching, achieving state-of-the-art performance. Project webpage: https://europe.naverlabs.com/research/3d-vision/pump.

----

## [384] CroMo: Cross-Modal Learning for Monocular Depth Estimation

**Authors**: *Yannick Verdié, Jifei Song, Barnabé Mas, Benjamin Busam, Ales Leonardis, Steven McDonagh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00391](https://doi.org/10.1109/CVPR52688.2022.00391)

**Abstract**:

Learning-based depth estimation has witnessed recent progress in multiple directions; from self-supervision using monocular video to supervised methods offering highest accuracy. Complementary to supervision, further boosts to performance and robustness are gained by combining information from multiple signals. In this paper we systematically investigate key trade-offs associated with sensor and modality design choices as well as related model training strategies. Our study leads us to a new method, capable of connecting modality-specific advantages from polarisation, Time-of-Flight and structured-light inputs. We propose a novel pipeline capable of estimating depth from monocular polarisation for which we evaluate various training signals. The inversion of differentiable analytic models thereby connects scene geometry with polarisation and ToF signals and enables self-supervised and cross-modal learning. In the absence of existing multimodal datasets, we examine our approach with a custom-made multi-modal camera rig and collect CroMo; the first dataset to consist of synchronized stereo polarisation, indirect ToF and structured-light depth, captured at video rates. Extensive experiments on challenging video scenes confirm both qualitative and quantitative pipeline advantages where we are able to outperform competitive monocular depth estimation methods.

----

## [385] $\phi$-SfT: Shape-from-Template with a Physics-Based Deformation Model

**Authors**: *Navami Kairanda, Edith Tretschk, Mohamed Elgharib, Christian Theobalt, Vladislav Golyanik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00392](https://doi.org/10.1109/CVPR52688.2022.00392)

**Abstract**:

Shape-from-Template (SfT) methods estimate 3D surface deformations from a single monocular RGB camera while assuming a 3D state known in advance (a template). This is an important yet challenging problem due to the under-constrained nature of the monocular setting. Existing SfT techniques predominantly use geometric and simplified deformation models, which often limits their reconstruction abilities. In contrast to previous works, this paper proposes a new SfT approach explaining 2D observations through physical simulations accounting for forces and material properties. Our differentiable physics simulator regularises the surface evolution and optimises the material elastic properties such as bending coefficients, stretching stiffness and density. We use a differentiable renderer to minimise the dense reprojection error between the estimated 3D states and the input images and recover the deformation parameters using an adaptive gradient-based optimisation. For the evaluation, we record with an RGB-D camera challenging real surfaces exposed to physical forces with various material properties and textures. Our approach significantly reduces the 3D reconstruction error compared to multiple competing methods. For the source code and data, see https://4dqv.mpi-inf.mpg.de/phi-SfT/.

----

## [386] Human-Aware Object Placement for Visual Environment Reconstruction

**Authors**: *Hongwei Yi, Chun-Hao P. Huang, Dimitrios Tzionas, Muhammed Kocabas, Mohamed Hassan, Siyu Tang, Justus Thies, Michael J. Black*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00393](https://doi.org/10.1109/CVPR52688.2022.00393)

**Abstract**:

Humans are in constant contact with the world as they move through it and interact with it. This contact is a vital source of information for understanding 3D humans, 3D scenes, and the interactions between them. In fact, we demonstrate that these human-scene interactions (HSIs) can be leveraged to improve the 3D reconstruction of a scene from a monocular RGB video. Our key idea is that, as a person moves through a scene and interacts with it, we accumulate HSIs across multiple input images, and use these in optimizing the 3D scene to reconstruct a consistent, physically plausible, 3D scene layout. Our optimization-based approach exploits three types of HSI constraints: (1) humans who move in a scene are occluded by, or occlude, objects, thus constraining the depth ordering of the objects, (2) humans move throughfree space and do not interpenetrate objects, (3) when humans and objects are in contact, the contact surfaces occupy the same place in space. Using these constraints in an optimization formulation across all observations, we significantly improve 3D scene layout reconstruction. Furthermore, we show that our scene reconstruction can be used to refine the initial 3D human pose and shape (HPS) estimation. We evaluate the 3D scene layout reconstruction and HPS estimates qualitatively and quantitatively using the PROX and PiGraphs datasets. The code and data are available for research purposes at https://mover.is.tue.mpg.de.

----

## [387] AutoRF: Learning 3D Object Radiance Fields from Single View Observations

**Authors**: *Norman Müller, Andrea Simonelli, Lorenzo Porzi, Samuel Rota Bulò, Matthias Nießner, Peter Kontschieder*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00394](https://doi.org/10.1109/CVPR52688.2022.00394)

**Abstract**:

We introduce AutoRF - a new approach for learning neural 3D object representations where each object in the training set is observed by only a single view. This setting is in stark contrast to the majority of existing works that leverage multiple views of the same object, employ explicit priors during training, or require pixel-perfect annotations. To address this challenging setting, we propose to learn a normalized, object-centric representation whose embedding describes and disentangles shape, appearance, and pose. Each encoding provides well-generalizable, compact information about the object of interest, which is decoded in a single-shot into a new target view, thus enabling novel view synthesis. We further improve the reconstruction quality by optimizing shape and appearance codes at test time by fitting the representation tightly to the input image. In a series of experiments, we show that our method generalizes well to unseen objects, even across different datasets of challenging real-world street scenes such as nuScenes, KITTI, and Mapillary Metropolis. Additional results can be found on our project page https://sirwyver.github.io/AutoRF/.

----

## [388] Pix2NeRF: Unsupervised Conditional $\pi$-GAN for Single Image to Neural Radiance Fields Translation

**Authors**: *Shengqu Cai, Anton Obukhov, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00395](https://doi.org/10.1109/CVPR52688.2022.00395)

**Abstract**:

We propose a pipeline to generate Neural Radiance Fields (NeRF) of an object or a scene of a specific class, conditioned on a single input image. This is a challenging task, as training NeRF requires multiple views of the same scene, coupled with corresponding poses, which are hard to obtain. Our method is based on 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\pi$</tex>
-GAN, a generative model for unconditional 3D-aware image synthesis, which maps random latent codes to radiance fields of a class of objects. We jointly optimize (1) the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\pi$</tex>
-GAN objective to utilize its high-fidelity 3D-aware generation and (2) a carefully designed reconstruction objective. The latter includes an encoder coupled with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\pi$</tex>
-GAN generator to form an autoencoder. Unlike previous few-shot NeRF approaches, our pipeline is unsupervised, capable of being trained with independent images without 3D, multi-view, or pose supervision. Applications of our pipeline include 3d avatar generation, object-centric novel view synthesis with a single input image, and 3d-aware super-resolution, to name a few.

----

## [389] MonoScene: Monocular 3D Semantic Scene Completion

**Authors**: *Anh-Quan Cao, Raoul de Charette*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00396](https://doi.org/10.1109/CVPR52688.2022.00396)

**Abstract**:

MonoScene proposes a 3D Semantic Scene Completion (SSC) framework, where the dense geometry and semantics of a scene are inferred from a single monocular RGB image. Different from the SSC literature, relying on 2.5 or 3D input, we solve the complex problem of 2D to 3D scene reconstruction while jointly inferring its semantics. Our framework relies on successive 2D and 3D UNets, bridged by a novel 2D-3D features projection inspired by optics, and introduces a 3D context relation prior to enforce spatio-semantic consistency. Along with architectural contributions, we introduce novel global scene and local frustums losses. Experiments show we outperform the literature on all metries and datasets while hallucinating plausible scenery even beyond the camera field of view. Our code and trained models are available at https://github.com/cv-rits/MonoScene.

----

## [390] GenDR: A Generalized Differentiable Renderer

**Authors**: *Felix Petersen, Bastian Goldluecke, Christian Borgelt, Oliver Deussen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00397](https://doi.org/10.1109/CVPR52688.2022.00397)

**Abstract**:

In this work, we present and study a generalized family of differentiable renderers. We discuss from scratch which components are necessary for differentiable rendering and formalize the requirements for each component. We instantiate our general differentiable renderer, which generalizes existing differentiable renderers like SoftRas and DIB-R, with an array of different smoothing distributions to cover a large spectrum of reasonable settings. We evaluate an array of differentiable renderer instantiations on the popular ShapeNet 3D reconstruction benchmark and analyze the implications of our results. Surprisingly, the simple uniform distribution yields the best overall results when averaged over 13 classes; in general, however, the optimal choice of distribution heavily depends on the task.

----

## [391] MonoDTR: Monocular 3D Object Detection with Depth-Aware Transformer

**Authors**: *Kuan-Chih Huang, Tsung-Han Wu, Hung-Ting Su, Winston H. Hsu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00398](https://doi.org/10.1109/CVPR52688.2022.00398)

**Abstract**:

Monocular 3D object detection is an important yet challenging task in autonomous driving. Some existing methods leverage depth information from an off-the-shelf depth estimator to assist 3D detection, but suffer from the additional computational burden and achieve limited performance caused by inaccurate depth priors. To alleviate this, we propose MonoDTR, a novel end-to-end depth-aware transformer network for monocular 3D object detection. It mainly consists of two components: (1) the Depth-Aware Feature Enhancement (DFE) module that implicitly learns depth-aware features with auxiliary supervision without requiring extra computation, and (2) the Depth-Aware Transformer (DTR) module that globally integrates context- and depth-aware features. Moreover, different from conventional pixel-wise positional encodings, we introduce a novel depth positional encoding (DPE) to inject depth positional hints into transformers. Our proposed depth-aware modules can be easily plugged into existing image-only monocular 3D object detectors to improve the performance. Extensive experiments on the KITTI dataset demonstrate that our approach outperforms previous state-of-the-art monocular-based methods and achieves real-time detection. Code is available at https://github.com/kuanchihhuang/.MonoDTR.

----

## [392] ROCA: Robust CAD Model Retrieval and Alignment from a Single Image

**Authors**: *Can Gümeli, Angela Dai, Matthias Nießner*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00399](https://doi.org/10.1109/CVPR52688.2022.00399)

**Abstract**:

We present ROCA 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code is made available at https://github.com/cangurneli/ROCA., a novel end-to-end approach that re-trieves and aligns 3D CAD models from a shape database to a single input image. This enables 3D perception of an ob-served scene from a 2D RGB observation, characterized as a lightweight, compact, clean CAD representation. Core to our approach is our differentiable alignment optimization based on dense 2D-3D object correspondences and Pro-crustes alignment. ROCA can thus provide a robust CAD alignment while simultaneously informing CAD retrieval by leveraging the 2D-3D correspondences to learn geometri-cally similar CAD models. Experiments on challenging, real-world imagery from ScanNet show that ROCA signif-icantly improves on state of the art, from 9.5% to 17.6% in retrieval-aware CAD alignment accuracy.

----

## [393] HP-Capsule: Unsupervised Face Part Discovery by Hierarchical Parsing Capsule Network

**Authors**: *Chang Yu, Xiangyu Zhu, Xiaomei Zhang, Zidu Wang, Zhaoxiang Zhang, Zhen Lei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00400](https://doi.org/10.1109/CVPR52688.2022.00400)

**Abstract**:

Capsule networks are designed to present the objects by a set of parts and their relationships, which provide an insight into the procedure of visual perception. Although recent works have shown the success of capsule networks on simple objects like digits, the human faces with homologous structures, which are suitable for capsules to describe, have not been explored. In this paper, we propose a Hierarchical Parsing Capsule Network (HP-Capsule) for unsupervised face subpart-part discovery. When browsing large-scale face images without labels, the network first encodes the frequently observed patterns with a set of explainable subpart capsules. Then, the subpart capsules are assembled into part-level capsules through a Transformer-based Parsing Module (TPM) to learn the compositional relations between them. During training as the face hierarchy is progressively built and refined, the part capsules adaptively encode the face parts with semantic consistency. HP-Capsule extends the application of capsule networks from digits to human faces and takes a step forward to show how the neural networks understand homologous objects without human intervention. Besides, HP-Capsule gives unsupervised face segmentation results by the covered regions of part capsules, enabling qualitative and quantitative evaluation. Experiments on BP4D and Multi-PIE datasets show the effectiveness of our method.

----

## [394] Killing Two Birds with One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC

**Authors**: *Xiang An, Jiankang Deng, Jia Guo, Ziyong Feng, Xuhan Zhu, Jing Yang, Tongliang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00401](https://doi.org/10.1109/CVPR52688.2022.00401)

**Abstract**:

Learning discriminative deep feature embeddings by using million-scale in-the-wild datasets and margin-based softmax loss is the current state-of-the-art approach for face recognition. However, the memory and computing cost of the Fully Connected (FC) layer linearly scales up to the number of identities in the training set. Besides, the largescale training data inevitably suffers from inter-class conflict and long-tailed distribution. In this paper, we propose a sparsely updating variant of the FC layer, named Partial FC (PFC). In each iteration, positive class centers and a random subset of negative class centers are selected to compute the margin-based softmax loss. All class centers are still maintained throughout the whole training process, but only a subset is selected and updated in each iteration. Therefore, the computing requirement, the probability of inter-class conflict, and the frequency of passive update on tail class centers, are dramatically reduced. Extensive experiments across different training data and backbones (e.g. CNN and ViT) confirm the effectiveness, robustness and efficiency of the proposed PFC. The source code is available at https://github.com/deepinsight/insightface/tree/master/recognition.

----

## [395] Sparse Local Patch Transformer for Robust Face Alignment and Landmarks Inherent Relation Learning

**Authors**: *Jiahao Xia, Weiwei Qu, Wenjian Huang, Jianguo Zhang, Xi Wang, Min Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00402](https://doi.org/10.1109/CVPR52688.2022.00402)

**Abstract**:

Heatmap regression methods have dominated face alignment area in recent years while they ignore the inherent relation between different landmarks. In this paper, we propose a Sparse Local Patch Transformer (SLPT) for learning the inherent relation. The SLPT generates the representation of each single landmark from a local patch and aggregates them by an adaptive inherent relation based on the attention mechanism. The subpixel coordinate of each landmark is predicted independently based on the aggregated feature. Moreover, a coarse-to-fine framework is further introduced to incorporate with the SLPT, which enables the initial landmarks to gradually converge to the target facial landmarks using fine-grained features from dynamically resized local patches. Extensive experiments carried out on three popular benchmarks, including WFLW, 300W and COFW, demonstrate that the proposed method works at the state-of-the-art level with much less computational complexity by learning the inherent relation between facial landmarks. The code is available at the project website
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/Jiahao-UTS/SLPT-master.

----

## [396] Enhancing Face Recognition with Self-Supervised 3D Reconstruction

**Authors**: *Mingjie He, Jie Zhang, Shiguang Shan, Xilin Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00403](https://doi.org/10.1109/CVPR52688.2022.00403)

**Abstract**:

Attributed to both the development of deep networks and abundant data, automatic face recognition (FR) has quickly reached human-level capacity in the past few years. However, the FR problem is not perfectly solved in case of uncontrolled illumination and pose. In this paper, we propose to enhance face recognition with a bypass of self-supervised 3D reconstruction, which enforces the neural backbone to focus on the identity-related depth and albedo information while neglects the identity-irrelevant pose and illumination information. Specifically, inspired by the physical model of image formation, we improve the backbone FR network by introducing a 3D face reconstruction loss with two auxiliary networks. The first one estimates the pose and illumination from the input face image while the second one decodes the canonical depth and albedo from the intermediate feature of the FR backbone network. The whole network is trained in end-to-end manner with both classic face identification loss and the loss of 3D face reconstruction with the physical parameters. In this way, the self-supervised reconstruction acts as a regularization that enables the recognition network to understand faces in 3D view, and the learnt features are forced to encode more information of canonical facial depth and albedo, which is more intrinsic and beneficial to face recognition. Extensive experimental results on various face recognition benchmarks show that, without any cost of extra annotations and computations, our method outperforms state-of-the-art ones. Moreover, the learnt representations can also well generalize to other face-related downstream tasks such as the facial attribute recognition with limited labeled data.

----

## [397] Learning to Learn across Diverse Data Biases in Deep Face Recognition

**Authors**: *Chang Liu, Xiang Yu, Yi-Hsuan Tsai, Masoud Faraki, Ramin Moslemi, Manmohan Chandraker, Yun Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00404](https://doi.org/10.1109/CVPR52688.2022.00404)

**Abstract**:

Convolutional Neural Networks have achieved remarkable success in face recognition, in part due to the abundant availability of data. However, the data used for training CNNs is often imbalanced. Prior works largely focus on the long-tailed nature of face datasets in data volume per identity, or focus on single bias variation. In this paper, we show that many bias variations such as ethnicity, head pose, occlusion and blur can jointly affect the accuracy significantly. We propose a sample level weighting approach termed Multi-variation Cosine Margin (MvCoM), to simultaneously consider the multiple variation factors, which orthogonally enhances the face recognition losses to incorporate the importance of training samples. Further, we leverage a learning to learn approach, guided by a held-out meta learning set and use an additive modeling to predict the MvCoM. Extensive experiments on challenging face recognition benchmarks demonstrate the advantages of our method in jointly handling imbalances due to multiple variations.

----

## [398] An Efficient Training Approach for Very Large Scale Face Recognition

**Authors**: *Kai Wang, Shuo Wang, Panpan Zhang, Zhipeng Zhou, Zheng Zhu, Xiaobo Wang, Xiaojiang Peng, Baigui Sun, Hao Li, Yang You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00405](https://doi.org/10.1109/CVPR52688.2022.00405)

**Abstract**:

Face recognition has achieved significant progress in deep learning era due to the ultra-large-scale and well- labeled datasets. However, training on the outsize datasets is time-consuming and takes up a lot of hardware resource. Therefore, designing an efficient training approach is in- dispensable. The heavy computational and memory costs mainly result from the million-level dimensionality of the fully connected (FC) layer. To this end, we propose a novel training approach, termed Faster Face Classification (F
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</inf>
C), to alleviate time and cost without sacrificing the performance. This method adopts Dynamic Class Pool (DCP) for storing and updating the identities' features dy-namically, which could be regarded as a substitute for the FC layer. DCP is efficiently time-saving and cost-saving, as its smaller size with the independence from the whole face identities together. We further validate the proposed F
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
C method across several face benchmarks and private datasets, and display comparable results, meanwhile the speed is faster than state-of-the-art FC-based methods in terms of recognition accuracy and hardware costs. More-over, our method is further improved by a well-designed dual data loader including indentity-based and instance- based loaders, which makes it more efficient for updating DCP parameters.

----

## [399] MogFace: Towards a Deeper Appreciation on Face Detection

**Authors**: *Yang Liu, Fei Wang, Jiankang Deng, Zhipeng Zhou, Baigui Sun, Hao Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00406](https://doi.org/10.1109/CVPR52688.2022.00406)

**Abstract**:

Benefiting from the pioneering design of generic object detectors, significant achievements have been made in the field of face detection. Typically, the architectures of the backbone, feature pyramid layer, and detection head module within the face detector all assimilate the excellent experience from general object detectors. However, several effective methods, including label assignment and scale-level data augmentation strategy, fail to maintain consistent superiority when applying on the face detector directly. Concretely, the former strategy involves a vast body of hyperparameters and the latter one suffers from the challenge of scale distribution bias between different detection tasks, which both limit their generalization abilities. Furthermore, in order to provide accurate face bounding boxes for facial down-stream tasks, the face detector imperatively requires the elimination of false alarms. As a result, practical solutions on label assignment, scale-level data augmentation, and reducing false alarms are necessary for advancing face detectors. In this paper, we focus on resolving three aforementioned challenges that exiting methods are difficult to finish off and present a novel face detector, termed MogFace. In our Mogface, three key components, Adaptive Online Incremental Anchor Mining Strategy, Selective Scale Enhancement Strategy and Hierarchical Context-Aware Module, are separately proposed to boost the performance of face detectors. Finally, to the best of our knowledge, our MogFace is the best face detector on the Wider Face leader-board, achieving all champions across different testing scenarios. The code is available at https://github.com/damo-cv/MogFace.

----



[Go to the previous page](CVPR-2022-list01.md)

[Go to the next page](CVPR-2022-list03.md)

[Go to the catalog section](README.md)