## [400] Grab What You Need: Rethinking Complex Table Structure Recognition with Flexible Components Deliberation

**Authors**: *Hao Liu, Xin Li, Mingming Gong, Bing Liu, Yunfei Wu, Deqiang Jiang, Yinsong Liu, Xing Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28149](https://doi.org/10.1609/aaai.v38i4.28149)

**Abstract**:

Recently, Table Structure Recognition (TSR) task, aiming at identifying table structure into machine readable formats, has received increasing interest in the community. While impressive success, most single table component-based methods can not perform well on unregularized table cases distracted by not only complicated inner structure but also exterior capture distortion. In this paper, we raise it as Complex TSR problem, where the performance degeneration of existing methods is attributable to their inefficient component usage and redundant post-processing. To mitigate it, we shift our perspective from table component extraction towards the efficient multiple components leverage, which awaits further exploration in the field. Specifically, we propose a seminal method, termed GrabTab, equipped with newly proposed Component Deliberator, to handle various types of tables in a unified framework. Thanks to its progressive deliberation mechanism, our GrabTab can flexibly accommodate to most complex tables with reasonable components selected but without complicated post-processing involved. Quantitative experimental results on public benchmarks demonstrate that our method significantly outperforms the state-of-the-arts, especially under more challenging scenes.

----

## [401] DiDA: Disambiguated Domain Alignment for Cross-Domain Retrieval with Partial Labels

**Authors**: *Haoran Liu, Ying Ma, Ming Yan, Yingke Chen, Dezhong Peng, Xu Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28150](https://doi.org/10.1609/aaai.v38i4.28150)

**Abstract**:

Driven by generative AI and the Internet, there is an increasing availability of a wide variety of images, leading to the significant and popular task of cross-domain image retrieval. To reduce annotation costs and increase performance, this paper focuses on an untouched but challenging problem, i.e., cross-domain image retrieval with partial labels (PCIR). Specifically, PCIR faces great challenges due to the ambiguous supervision signal and the domain gap. To address these challenges, we propose a novel method called disambiguated domain alignment (DiDA) for cross-domain retrieval with partial labels. In detail, DiDA elaborates a novel prototype-score unitization learning mechanism (PSUL) to extract common discriminative representations by simultaneously disambiguating the partial labels and narrowing the domain gap. Additionally, DiDA proposes a prototype-based domain alignment mechanism (PBDA) to further bridge the inherent cross-domain discrepancy. Attributed to PSUL and PBDA, our DiDA effectively excavates domain-invariant discrimination for cross-domain image retrieval. We demonstrate the effectiveness of DiDA through comprehensive experiments on three benchmarks, comparing it to existing state-of-the-art methods. Code available: https://github.com/lhrrrrrr/DiDA.

----

## [402] Test-Time Personalization with Meta Prompt for Gaze Estimation

**Authors**: *Huan Liu, Julia Qi, Zhenhao Li, Mohammad Hassanpour, Yang Wang, Konstantinos N. Plataniotis, Yuanhao Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28151](https://doi.org/10.1609/aaai.v38i4.28151)

**Abstract**:

Despite the recent remarkable achievement in gaze estimation, efficient and accurate personalization of gaze estimation without labels is a practical problem but rarely touched on in the literature. To achieve efficient  personalization, we take inspiration from the recent advances in Natural Language Processing (NLP) by updating a negligible number of parameters, "prompts", at the test time. Specifically, the prompt is additionally attached  without perturbing original network and can contain less than 1% of a ResNet-18's parameters. Our experiments show high efficiency of the prompt tuning approach. The proposed one can be 10 times faster in terms of adaptation speed than the methods compared. However, it is non-trivial to update the prompt for personalized gaze estimation without labels.  At the test time, it is essential to ensure that the minimizing of particular unsupervised loss leads to the goals of minimizing gaze estimation error. To address this difficulty, we propose to meta-learn the prompt to ensure that its updates align with the goal. Our experiments show that the meta-learned prompt can be effectively adapted even with a simple symmetry loss. In addition, we experiment on four cross-dataset validations to show the remarkable advantages of the proposed method.

----

## [403] M3SOT: Multi-Frame, Multi-Field, Multi-Space 3D Single Object Tracking

**Authors**: *Jiaming Liu, Yue Wu, Maoguo Gong, Qiguang Miao, Wenping Ma, Cai Xu, Can Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28152](https://doi.org/10.1609/aaai.v38i4.28152)

**Abstract**:

3D Single Object Tracking (SOT) stands a forefront task of computer vision, proving essential for applications like autonomous driving. Sparse and occluded data in scene point clouds introduce variations in the appearance of tracked objects, adding complexity to the task. In this research, we unveil M3SOT, a novel 3D SOT framework, which synergizes multiple input frames (template sets), multiple receptive fields (continuous contexts), and multiple solution spaces (distinct tasks) in ONE model. Remarkably, M3SOT pioneers in modeling temporality, contexts, and tasks directly from point clouds, revisiting a perspective on the key factors influencing SOT. To this end, we design a transformer-based network centered on point cloud targets in the search area, aggregating diverse contextual representations and propagating target cues by employing historical frames. As M3SOT spans varied processing perspectives, we've streamlined the network—trimming its depth and optimizing its structure—to ensure a lightweight and efficient deployment for SOT applications. We posit that, backed by practical construction, M3SOT sidesteps the need for complex frameworks and auxiliary components to deliver sterling results. Extensive experiments on benchmarks such as KITTI, nuScenes, and Waymo Open Dataset demonstrate that M3SOT achieves state-of-the-art performance at 38 FPS. Our code and models are available at https://github.com/ywu0912/TeamCode.git.

----

## [404] Unsupervised Continual Anomaly Detection with Contrastively-Learned Prompt

**Authors**: *Jiaqi Liu, Kai Wu, Qiang Nie, Ying Chen, Bin-Bin Gao, Yong Liu, Jinbao Wang, Chengjie Wang, Feng Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28153](https://doi.org/10.1609/aaai.v38i4.28153)

**Abstract**:

Unsupervised Anomaly Detection (UAD) with incremental training is crucial in industrial manufacturing, as unpredictable defects make obtaining sufficient labeled data infeasible. However, continual learning methods primarily rely on supervised annotations, while the application in UAD is limited due to the absence of supervision. Current UAD methods train separate models for different classes sequentially, leading to catastrophic forgetting and a heavy computational burden. To address this issue, we introduce a novel Unsupervised Continual Anomaly Detection framework called UCAD, which equips the UAD with continual learning capability through contrastively-learned prompts. In the proposed UCAD, we design a Continual Prompting Module (CPM) by utilizing a concise key-prompt-knowledge memory bank to guide task-invariant 'anomaly' model predictions using task-specific 'normal' knowledge. Moreover, Structure-based Contrastive Learning (SCL) is designed with the Segment Anything Model (SAM) to improve prompt learning and anomaly segmentation results. Specifically, by treating SAM's masks as structure, we draw features within the same mask closer and push others apart for general feature representations. We conduct comprehensive experiments and set the benchmark on unsupervised continual anomaly detection and segmentation, demonstrating that our method is significantly better than anomaly detection methods, even with rehearsal training. The code will be available at https://github.com/shirowalker/UCAD.

----

## [405] Region-Aware Exposure Consistency Network for Mixed Exposure Correction

**Authors**: *Jin Liu, Huiyuan Fu, Chuanming Wang, Huadong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28154](https://doi.org/10.1609/aaai.v38i4.28154)

**Abstract**:

Exposure correction aims to enhance images suffering from improper exposure to achieve satisfactory visual effects. Despite recent progress, existing methods generally mitigate either overexposure or underexposure in input images, and they still struggle to handle images with mixed exposure, i.e., one image incorporates both overexposed and underexposed regions. The mixed exposure distribution is non-uniform and leads to varying representation, which makes it challenging to address in a unified process. In this paper, we introduce an effective Region-aware Exposure Correction Network (RECNet) that can handle mixed exposure by adaptively learning and bridging different regional exposure representations. Specifically, to address the challenge posed by mixed exposure disparities, we develop a region-aware de-exposure module that effectively translates regional features of mixed exposure scenarios into an exposure-invariant feature space. Simultaneously, as de-exposure operation inevitably reduces discriminative information, we introduce a mixed-scale restoration unit that integrates exposure-invariant features and unprocessed features to recover local information. To further achieve a uniform exposure distribution in the global image, we propose an exposure contrastive regularization strategy under the constraints of intra-regional exposure consistency and inter-regional exposure continuity. Extensive experiments are conducted on various datasets, and the experimental results demonstrate the superiority and generalization of our proposed method. The code is released at: https://github.com/kravrolens/RECNet.

----

## [406] R3CD: Scene Graph to Image Generation with Relation-Aware Compositional Contrastive Control Diffusion

**Authors**: *Jinxiu Liu, Qi Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28155](https://doi.org/10.1609/aaai.v38i4.28155)

**Abstract**:

Image generation tasks have achieved remarkable performance using large-scale diffusion models. However, these models are limited to capturing the abstract relations (viz., interactions excluding positional relations) among multiple entities of complex scene graphs. Two main problems exist:  1) fail to depict more concise and accurate interactions via abstract relations; 2) fail to generate complete entities. To address that, we propose a novel Relation-aware Compositional Contrastive Control Diffusion method, dubbed as R3CD, that leverages large-scale diffusion models to learn abstract interactions from scene graphs. Herein, a scene graph transformer based on node and edge encoding is first designed to perceive both local and global information from input scene graphs, whose embeddings are initialized by a T5 model. Then a joint contrastive loss based on attention maps and denoising steps is developed to control the diffusion model to understand and further generate images, whose spatial structures and interaction features are consistent with a priori relation. Extensive experiments are conducted on two datasets: Visual Genome and COCO-Stuff, and demonstrate that the proposal outperforms existing models both in quantitative and qualitative metrics to generate more realistic and diverse images according to different scene graph specifications.

----

## [407] DifAttack: Query-Efficient Black-Box Adversarial Attack via Disentangled Feature Space

**Authors**: *Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28156](https://doi.org/10.1609/aaai.v38i4.28156)

**Abstract**:

This work investigates efficient score-based black-box adversarial attacks with high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a Disentangled Feature space, called DifAttack, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack firstly disentangles an image's latent feature into an adversarial feature and a visual feature, where the former dominates the adversarial capability of an image, while the latter largely determines its visual appearance. We train an autoencoder for the disentanglement by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, DifAttack iteratively optimizes the adversarial feature according to the query feedback from the victim model until a successful AE is generated, while keeping the visual feature unaltered. In addition, due to the avoidance of using surrogate models' gradient information when optimizing AEs for black-box models, our proposed DifAttack inherently possesses better attack capability in the open-set scenario, where the training dataset of the victim model is unknown. Extensive experimental results demonstrate that our method achieves significant improvements in ASR and query efficiency simultaneously, especially in the targeted attack and open-set scenarios. The code is available The code is available at https://github.com/csjunjun/DifAttack.git.

----

## [408] Frequency Shuffling and Enhancement for Open Set Recognition

**Authors**: *Lijun Liu, Rui Wang, Yuan Wang, Lihua Jing, Chuan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28157](https://doi.org/10.1609/aaai.v38i4.28157)

**Abstract**:

Open-Set Recognition (OSR) aims to accurately identify known classes while effectively rejecting unknown classes to guarantee reliability. Most existing OSR methods focus on learning in the spatial domain, where subtle texture and global structure are potentially intertwined. Empirical studies have shown that DNNs trained in the original spatial domain are inclined to over-perceive subtle texture. The biased semantic perception could lead to catastrophic over-confidence when predicting both known and unknown classes. To this end, we propose an innovative approach by decomposing the spatial domain to the frequency domain to separately consider global (low-frequency) and subtle (high-frequency) information, named Frequency Shuffling and Enhancement (FreSH). To alleviate the overfitting of subtle texture, we introduce the High-Frequency Shuffling (HFS) strategy that generates diverse high-frequency information and promotes the capture of low-frequency invariance. Moreover, to enhance the perception of global structure, we propose the Low-Frequency Residual (LFR) learning procedure that constructs a composite feature space, integrating low-frequency and original spatial features. Experiments on various benchmarks demonstrate that the proposed FreSH consistently trumps the state-of-the-arts by a considerable margin.

----

## [409] KPA-Tracker: Towards Robust and Real-Time Category-Level Articulated Object 6D Pose Tracking

**Authors**: *Liu Liu, Anran Huang, Qi Wu, Dan Guo, Xun Yang, Meng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28158](https://doi.org/10.1609/aaai.v38i4.28158)

**Abstract**:

Our life is populated with articulated objects. Current category-level articulation estimation works largely focus on predicting part-level 6D poses on static point cloud observations. In this paper, we tackle the problem of category-level online robust and real-time 6D pose tracking of articulated objects, where we propose KPA-Tracker, a novel 3D KeyPoint based Articulated object pose Tracker. Given an RGB-D image or a partial point cloud at the current frame as well as the estimated per-part 6D poses from the last frame, our KPA-Tracker can effectively update the poses with learned 3D keypoints between the adjacent frames. Specifically, we first canonicalize the input point cloud and formulate the pose tracking as an inter-frame pose increment estimation task. To learn consistent and separate 3D keypoints for every rigid part, we build KPA-Gen that outputs the high-quality ordered 3D keypoints in an unsupervised manner. During pose tracking on the whole video, we further propose a keypoint-based articulation tracking algorithm that mines keyframes as reference for accurate pose updating. We provide extensive experiments on validating our KPA-Tracker on various datasets ranging from synthetic point cloud observation to real-world scenarios, which demonstrates the superior performance and robustness of the KPA-Tracker. We believe that our work has the potential to be applied in many fields including robotics, embodied intelligence and augmented reality. All the datasets and codes are available at https://github.com/hhhhhar/KPA-Tracker.

----

## [410] UVAGaze: Unsupervised 1-to-2 Views Adaptation for Gaze Estimation

**Authors**: *Ruicong Liu, Feng Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28159](https://doi.org/10.1609/aaai.v38i4.28159)

**Abstract**:

Gaze estimation has become a subject of growing interest in recent research. Most of the current methods rely on single-view facial images as input. Yet, it is hard for these approaches to handle large head angles, leading to potential inaccuracies in the estimation. To address this issue, adding a second-view camera can help better capture eye appearance. However, existing multi-view methods have two limitations. 1) They require multi-view annotations for training, which are expensive. 2) More importantly, during testing, the exact positions of the multiple cameras must be known and match those used in training, which limits the application scenario. To address these challenges, we propose a novel 1-view-to-2-views (1-to-2 views) adaptation solution in this paper, the Unsupervised 1-to-2 Views Adaptation framework for Gaze estimation (UVAGaze). Our method adapts a traditional single-view gaze estimator for flexibly placed dual cameras. Here, the "flexibly" means we place the dual cameras in arbitrary places regardless of the training data, without knowing their extrinsic parameters. Specifically, the UVAGaze builds a dual-view mutual supervision adaptation strategy, which takes advantage of the intrinsic consistency of gaze directions between both views. In this way, our method can not only benefit from common single-view pre-training, but also achieve more advanced dual-view gaze estimation. The experimental results show that a single-view estimator, when adapted for dual views, can achieve much higher accuracy, especially in cross-dataset settings, with a substantial improvement of 47.0%. Project page: https://github.com/MickeyLLG/UVAGaze.

----

## [411] Compact HD Map Construction via Douglas-Peucker Point Transformer

**Authors**: *Ruixin Liu, Zejian Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28160](https://doi.org/10.1609/aaai.v38i4.28160)

**Abstract**:

High-definition (HD) map construction requires a comprehensive understanding of traffic environments, encompassing centimeter-level localization and rich semantic information. Previous works face challenges in redundant point representation or high-complexity curve modeling. In this paper, we present a flexible yet effective map element detector that synthesizes hierarchical information with a compact Douglas-Peucker (DP) point representation in a transformer architecture for robust and reliable predictions. Specifically, our proposed representation approximates class-agnostic map elements with DP points, which are sparsely located in crucial positions of structures and can get rid of redundancy and complexity. Besides, we design a position constraint with uncertainty to avoid potential ambiguities. Moreover, pairwise-point shape matching constraints are proposed to balance local structural information of different scales. Experiments on the public nuScenes dataset demonstrate that our method overwhelms current SOTAs. Extensive ablation studies validate each component of our methods. Codes will be released at https://github.com/sweety121/DPFormer.

----

## [412] Primitive-Based 3D Human-Object Interaction Modelling and Programming

**Authors**: *Siqi Liu, Yong-Lu Li, Zhou Fang, Xinpeng Liu, Yang You, Cewu Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28161](https://doi.org/10.1609/aaai.v38i4.28161)

**Abstract**:

Embedding Human and Articulated Object Interaction (HAOI) in 3D is an important direction for a deeper human activity understanding. Different from previous works that use parametric and CAD models to represent humans and objects, in this work, we propose a novel 3D geometric primitive-based language to encode both humans and objects. Given our new paradigm, humans and objects are all compositions of primitives instead of heterogeneous entities. Thus, mutual information learning may be achieved between the limited 3D data of humans and different object categories. Moreover, considering the simplicity of the expression and the richness of the information it contains, we choose the superquadric as the primitive representation.
To explore an effective embedding of HAOI for the machine, we build a new benchmark on 3D HAOI consisting of primitives together with their images and propose a task requiring machines to recover 3D HAOI using primitives from images.
Moreover, we propose a baseline of single-view 3D reconstruction on HAOI. We believe this primitive-based 3D HAOI representation would pave the way for 3D HAOI studies. Our code and data are available at https://mvig-rhos.com/p3haoi.

----

## [413] Fast Inter-frame Motion Prediction for Compressed Dynamic Point Cloud Attribute Enhancement

**Authors**: *Wang Liu, Wei Gao, Xingming Mu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28162](https://doi.org/10.1609/aaai.v38i4.28162)

**Abstract**:

Recent years have witnessed the success of deep learning methods in quality enhancement of compressed point cloud. However, existing methods focus on geometry and attribute enhancement of single-frame point cloud. This paper proposes a novel compressed quality enhancement method for dynamic point cloud (DAE-MP). Specifically, we propose a fast inter-frame motion prediction module (IFMP) to explicitly estimate motion displacement and achieve inter-frame feature alignment. To maintain motion continuity between consecutive frames, we propose a motion consistency loss for supervised learning. Furthermore, a frequency component separation and fusion module is designed to extract rich frequency features adaptively. To the best of our knowledge, the proposed method is the first deep learning-based work to enhance the quality for compressed dynamic point cloud. Experimental results show that the proposed method can greatly improve the quality of compressed dynamic point cloud and provide a fast and efficient motion prediction plug-in for large-scale point cloud. For dynamic point cloud attribute with severely compressed artifact, our proposed DAE-MP method achieves up to 0.52dB (PSNR) performance gain. Moreover, the proposed IFMP module has a certain real-time processing ability for calculating the motion offset between dynamic point cloud frame.

----

## [414] RWMS: Reliable Weighted Multi-Phase for Semi-supervised Segmentation

**Authors**: *Wensi Liu, Xiao-Yu Tang, Chong Yang, Chunjie Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28163](https://doi.org/10.1609/aaai.v38i4.28163)

**Abstract**:

Semantic segmentation is one of the tasks concerned in the field of computer vision. However, the cost of capturing large numbers of pixel-level annotations is expensive. Semi-supervised learning can utilize labeled and unlabeled data, providing new ideas for solving the problem of insufficient labeled data. In this work, we propose a data-reliability weighted multi-phase learning method for semi-supervised segmentation (RWMS). Under the framework of self-training, we train two different teacher models to evaluate the reliability of pseudo labels. By selecting reliable data at the image level and reweighting pseudo labels at the pixel level, multi-phase training is guided to focus on more reliable knowledge. Besides, we also inject strong data augmentations on unlabeled images while training. Through extensive experiments, we demonstrate that our method performs remarkably well compared to baseline methods and substantially outperforms them, more than 3% on VOC and Cityscapes.

----

## [415] Learning Real-World Image De-weathering with Imperfect Supervision

**Authors**: *Xiaohui Liu, Zhilu Zhang, Xiaohe Wu, Chaoyu Feng, Xiaotao Wang, Lei Lei, Wangmeng Zuo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28164](https://doi.org/10.1609/aaai.v38i4.28164)

**Abstract**:

Real-world image de-weathering aims at removing various undesirable weather-related artifacts. Owing to the impossibility of capturing image pairs concurrently, existing real-world de-weathering datasets often exhibit inconsistent illumination, position, and textures between the ground-truth images and the input degraded images, resulting in imperfect supervision. Such non-ideal supervision negatively affects the training process of learning-based de-weathering methods. In this work, we attempt to address the problem with a unified solution for various inconsistencies. Specifically, inspired by information bottleneck theory, we first develop a Consistent Label Constructor (CLC) to generate a pseudo-label as consistent as possible with the input degraded image while removing most weather-related degradation. In particular, multiple adjacent frames of the current input are also fed into CLC to enhance the pseudo-label. Then we combine the original imperfect labels and pseudo-labels to jointly supervise the de-weathering model by the proposed Information Allocation Strategy (IAS). During testing, only the de-weathering model is used for inference. Experiments on two real-world de-weathering datasets show that our method helps existing de-weathering models achieve better performance. Code is available at https://github.com/1180300419/imperfect-deweathering.

----

## [416] Differentiable Auxiliary Learning for Sketch Re-Identification

**Authors**: *Xingyu Liu, Xu Cheng, Haoyu Chen, Hao Yu, Guoying Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28165](https://doi.org/10.1609/aaai.v38i4.28165)

**Abstract**:

Sketch re-identification (Re-ID) seeks to match pedestrians' photos from surveillance videos with corresponding sketches. However, we observe that existing works still have two critical limitations: (i) cross- and intra-modality discrepancies hinder the extraction of modality-shared features, (ii) standard triplet loss fails to constrain latent feature distribution in each modality with inadequate samples. To overcome the above issues, we propose a differentiable auxiliary learning network (DALNet) to explore a robust auxiliary modality for Sketch Re-ID. Specifically, for (i) we construct an auxiliary modality by using a dynamic auxiliary generator (DAG) to bridge the gap between sketch and photo modalities. The auxiliary modality highlights the described person in photos to mitigate background clutter and learns sketch style through style refinement. Moreover, a modality interactive attention module (MIA) is presented to align the features and learn the invariant patterns of two modalities by auxiliary modality. To address (ii), we propose a multi-modality collaborative learning scheme (MMCL) to align the latent distribution of three modalities. An intra-modality circle loss in MMCL brings learned global and modality-shared features of the same identity closer in the case of insufficient samples within each modality. Extensive experiments verify the superior performance of our DALNet over the state-of-the-art methods for Sketch Re-ID, and the generalization in sketch-based image retrieval and sketch-photo face recognition tasks.

----

## [417] Keypoint Fusion for RGB-D Based 3D Hand Pose Estimation

**Authors**: *Xingyu Liu, Pengfei Ren, Yuanyuan Gao, Jingyu Wang, Haifeng Sun, Qi Qi, Zirui Zhuang, Jianxin Liao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28166](https://doi.org/10.1609/aaai.v38i4.28166)

**Abstract**:

Previous 3D hand pose estimation methods primarily rely on a single modality, either RGB or depth, and the comprehensive utilization of the dual modalities has not been extensively explored. RGB and depth data provide complementary information and thus can be fused to enhance the robustness of 3D hand pose estimation. However, there exist two problems for applying existing fusion methods in 3D hand pose estimation: redundancy of dense feature fusion and ambiguity of visual features. First, pixel-wise feature interactions introduce high computational costs and ineffective calculations of invalid pixels. Second, visual features suffer from ambiguity due to color and texture similarities, as well as depth holes and noise caused by frequent hand movements, which interferes with modeling cross-modal correlations. In this paper, we propose Keypoint-Fusion for RGB-D based 3D hand pose estimation, which leverages the unique advantages of dual modalities to mutually eliminate the feature ambiguity, and performs cross-modal feature fusion in a more efficient way. Specifically, we focus cross-modal fusion on sparse yet informative spatial regions (i.e. keypoints). Meanwhile, by explicitly extracting relatively more reliable information as disambiguation evidence, depth modality provides 3D geometric information for RGB feature pixels, and RGB modality complements the precise edge information lost due to the depth noise. Keypoint-Fusion achieves state-of-the-art performance on two challenging hand datasets, significantly decreasing the error compared with previous single-modal methods.

----

## [418] CAVEN: An Embodied Conversational Agent for Efficient Audio-Visual Navigation in Noisy Environments

**Authors**: *Xiulong Liu, Sudipta Paul, Moitreya Chatterjee, Anoop Cherian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28167](https://doi.org/10.1609/aaai.v38i4.28167)

**Abstract**:

Audio-visual navigation of an agent towards locating an audio goal is a challenging task especially when the audio is sporadic or the environment is noisy. In this paper, we present CAVEN,  a Conversation-based Audio-Visual Embodied Navigation framework in which the agent may interact with a human/oracle for solving the task of navigating to an audio goal. Specifically, CAVEN is modeled as a budget-aware partially observable semi-Markov decision process that implicitly learns the uncertainty in the audio-based navigation policy to decide when and how the agent may interact with the oracle. Our CAVEN agent can engage in fully-bidirectional natural language conversations by producing relevant questions and interpret free-form, potentially noisy responses from the oracle based on the audio-visual context. To enable such a capability, CAVEN is equipped with: i) a trajectory forecasting network that is grounded in audio-visual cues to produce a potential trajectory to the estimated goal, and (ii) a natural language based question generation and reasoning network to pose an interactive question to the oracle or interpret the oracle's response to produce navigation instructions. To train the interactive modules, we present a large scale dataset: AVN-Instruct, based on the Landmark-RxR dataset. To substantiate the usefulness of conversations, we present experiments on the benchmark audio-goal task using the SoundSpaces simulator under various noisy settings. Our results reveal that our fully-conversational approach leads to nearly an order-of-magnitude improvement in success rate, especially in localizing new sound sources and against methods that use only uni-directional interaction.

----

## [419] DeepCalliFont: Few-Shot Chinese Calligraphy Font Synthesis by Integrating Dual-Modality Generative Models

**Authors**: *Yitian Liu, Zhouhui Lian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28168](https://doi.org/10.1609/aaai.v38i4.28168)

**Abstract**:

Few-shot font generation, especially for Chinese calligraphy fonts, is a challenging and ongoing problem. With the help of prior knowledge that is mainly based on glyph consistency assumptions, some recently proposed methods can synthesize high-quality Chinese glyph images. However, glyphs in calligraphy font styles often do not meet these assumptions. To address this problem, we propose a novel model, DeepCalliFont, for few-shot Chinese calligraphy font synthesis by integrating dual-modality generative models. Specifically, the proposed model consists of image synthesis and sequence generation branches, generating consistent results via a dual-modality representation learning strategy. The two modalities (i.e., glyph images and writing sequences) are properly integrated using a feature recombination module and a rasterization loss function. Furthermore, a new pre-training strategy is adopted to improve the performance by exploiting large amounts of uni-modality data. Both qualitative and quantitative experiments have been conducted to demonstrate the superiority of our method to other state-of-the-art approaches in the task of few-shot Chinese calligraphy font synthesis. The source code can be found at https://github.com/lsflyt-pku/DeepCalliFont.

----

## [420] Stable Unlearnable Example: Enhancing the Robustness of Unlearnable Examples via Stable Error-Minimizing Noise

**Authors**: *Yixin Liu, Kaidi Xu, Xun Chen, Lichao Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28169](https://doi.org/10.1609/aaai.v38i4.28169)

**Abstract**:

The open sourcing of large amounts of image data promotes the development of deep learning techniques. Along with this comes the privacy risk of these image datasets being exploited by unauthorized third parties to train deep learning models for commercial or illegal purposes. To avoid the abuse of data, a poisoning-based technique, "unlearnable example", has been proposed to significantly degrade the generalization performance of models by adding imperceptible noise to the data. To further enhance its robustness against adversarial training, existing works leverage iterative adversarial training on both the defensive noise and the surrogate model. However, it still remains unknown whether the robustness of unlearnable examples primarily comes from the effect of enhancement in the surrogate model or the defensive noise. Observing that simply removing the adversarial perturbation on the training process of the defensive noise can improve the performance of robust unlearnable examples, we identify that solely the surrogate model's robustness contributes to the performance. Furthermore, we found a negative correlation exists between the robustness of defensive noise and the protection performance, indicating defensive noise's instability issue. Motivated by this, to further boost the robust unlearnable example, we introduce Stable Error-Minimizing noise (SEM), which trains the defensive noise against random perturbation instead of the time-consuming adversarial perturbation to improve the stability of defensive noise. Through comprehensive experiments, we demonstrate that SEM achieves a new state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet Subset regarding both effectiveness and efficiency.

----

## [421] Scaling and Masking: A New Paradigm of Data Sampling for Image and Video Quality Assessment

**Authors**: *Yongxu Liu, Yinghui Quan, Guoyao Xiao, Aobo Li, Jinjian Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28170](https://doi.org/10.1609/aaai.v38i4.28170)

**Abstract**:

Quality assessment of images and videos emphasizes both local details and global semantics, whereas general data sampling methods (e.g., resizing, cropping or grid-based fragment) fail to catch them simultaneously. To address the deficiency, current approaches have to adopt multi-branch models and take as input the multi-resolution data, which burdens the model complexity. In this work, instead of stacking up models, a more elegant data sampling method (named as SAMA, scaling and masking) is explored, which compacts both the local and global content in a regular input size. The basic idea is to scale the data into a pyramid first, and reduce the pyramid into a regular data dimension with a masking strategy. Benefiting from the spatial and temporal redundancy in images and videos, the processed data maintains the multi-scale characteristics with a regular input size, thus can be processed by a single-branch model. We verify the sampling method in image and video quality assessment. Experiments show that our sampling method can improve the performance of current single-branch models significantly, and achieves competitive performance to the multi-branch models without extra model complexity. The source code will be available at https://github.com/Sissuire/SAMA.

----

## [422] Implicit Modeling of Non-rigid Objects with Cross-Category Signals

**Authors**: *Yuchun Liu, Benjamin Planche, Meng Zheng, Zhongpai Gao, Pierre Sibut-Bourde, Fan Yang, Terrence Chen, Ziyan Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28171](https://doi.org/10.1609/aaai.v38i4.28171)

**Abstract**:

Deep implicit functions (DIFs) have emerged as a potent and articulate means of representing 3D shapes. However, methods modeling object categories or non-rigid entities have mainly focused on single-object scenarios. In this work, we propose MODIF, a multi-object deep implicit function that jointly learns the deformation fields and instance-specific latent codes for multiple objects at once. Our emphasis is on non-rigid, non-interpenetrating entities such as organs. To effectively capture the interrelation between these entities and ensure precise, collision-free representations, our approach facilitates signaling between category-specific fields to adequately rectify shapes. We also introduce novel inter-object supervision: an attraction-repulsion loss is formulated to refine contact regions between objects. Our approach is demonstrated on various medical benchmarks, involving modeling different groups of intricate anatomical entities. Experimental results illustrate that our model can proficiently learn the shape representation of each organ and their relations to others, to the point that shapes missing from unseen instances can be consistently recovered by our method. Finally, MODIF can also propagate semantic information throughout the population via accurate point correspondences.

----

## [423] Recasting Regional Lighting for Shadow Removal

**Authors**: *Yuhao Liu, Zhanghan Ke, Ke Xu, Fang Liu, Zhenwei Wang, Rynson W. H. Lau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28172](https://doi.org/10.1609/aaai.v38i4.28172)

**Abstract**:

Removing shadows requires an understanding of both lighting conditions and object textures in a scene. Existing methods typically learn pixel-level color mappings between
shadow and non-shadow images, in which the joint modeling of lighting and object textures is implicit and inadequate. We observe that in a shadow region, the degradation degree of object textures depends on the local illumination, while simply enhancing the local illumination cannot fully recover the attenuated textures. Based on this observation, we propose to condition the restoration of attenuated textures on the corrected local lighting in the shadow region. Specifically, We first design a shadow-aware decomposition network to estimate the illumination and reflectance layers of shadow regions explicitly. We then propose a novel bilateral correction network to recast the lighting of shadow regions in the illumination layer via a novel local lighting correction module, and to restore the textures conditioned on the corrected illumination layer via a novel illumination-guided texture restoration module. We further annotate pixel-wise shadow masks for the public SRD dataset, which originally contains only image pairs. Experiments on three benchmarks show that our method outperforms existing state-of-the-art shadow removal methods.  Project page in: yuhaoliu7456.github.io/RRL-Net.

----

## [424] Rolling-Unet: Revitalizing MLP's Ability to Efficiently Extract Long-Distance Dependencies for Medical Image Segmentation

**Authors**: *Yutong Liu, Haijiang Zhu, Mengting Liu, Huaiyuan Yu, Zihan Chen, Jie Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28173](https://doi.org/10.1609/aaai.v38i4.28173)

**Abstract**:

Medical image segmentation methods based on deep learning network are mainly divided into CNN and Transformer. However, CNN struggles to capture long-distance dependencies, while Transformer suffers from high computational complexity and poor local feature learning. To efficiently extract and fuse local features and long-range dependencies, this paper proposes Rolling-Unet, which is a CNN model combined with MLP. Specifically, we propose the core R-MLP module, which is responsible for learning the long-distance dependency in a single direction of the whole image. By controlling and combining R-MLP modules in different directions, OR-MLP and DOR-MLP modules are formed to capture long-distance dependencies in multiple directions. Further, Lo2 block is proposed to encode both local context information and long-distance dependencies without excessive computational burden. Lo2 block has the same parameter size and computational complexity as a 3×3 convolution. The experimental results on four public datasets show that Rolling-Unet achieves superior performance compared to the state-of-the-art methods.

----

## [425] Advancing Video Synchronization with Fractional Frame Analysis: Introducing a Novel Dataset and Model

**Authors**: *Yuxuan Liu, Haizhou Ai, Junliang Xing, Xuri Li, Xiaoyi Wang, Pin Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28174](https://doi.org/10.1609/aaai.v38i4.28174)

**Abstract**:

Multiple views play a vital role in 3D pose estimation tasks. Ideally, multi-view 3D pose estimation tasks should directly utilize naturally collected videos for pose estimation. However, due to the constraints of video synchronization, existing methods often use expensive hardware devices to synchronize the initiation of cameras, which restricts most 3D pose collection scenarios to indoor settings. Some recent works learn deep neural networks to align desynchronized datasets derived from synchronized cameras and can only produce frame-level accuracy. For fractional frame video synchronization, this work proposes an Inter-Frame and Intra-Frame Desynchronized Dataset (IFID), which labels fractional time intervals between two video clips. IFID is the first dataset that annotates inter-frame and intra-frame intervals, with a total of 382,500 video clips annotated, making it the largest dataset to date. We also develop a novel model based on the Transformer architecture, named InSynFormer, for synchronizing inter-frame and intra-frame. Extensive experimental evaluations demonstrate its promising performance. The dataset and source code of the model are available at https://github.com/yuxuan-cser/InSynFormer.

----

## [426] FedCD: Federated Semi-Supervised Learning with Class Awareness Balance via Dual Teachers

**Authors**: *Yuzhi Liu, Huisi Wu, Jing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28175](https://doi.org/10.1609/aaai.v38i4.28175)

**Abstract**:

Recent advancements in deep learning have greatly improved the efficiency of auxiliary medical diagnostics. However, concerns over patient privacy and data annotation costs restrict the viability of centralized training models. In response, federated semi-supervised learning has garnered substantial attention from medical institutions. However, it faces challenges arising from knowledge discrepancies among local clients and class imbalance in non-independent and identically distributed data. Existing methods like class balance adaptation for addressing class imbalance often overlook low-confidence yet valuable rare samples in unlabeled data and may compromise client privacy. To address these issues, we propose a novel framework with class awareness balance and dual teacher distillation called FedCD. FedCD introduces a global-local framework to balance and purify global and local knowledge. Additionally, we introduce a novel class awareness balance module to effectively explore potential rare classes and encourage balanced learning in unlabeled clients. Importantly, our approach prioritizes privacy protection by only exchanging network parameters during communication. Experimental results on two medical datasets under various settings demonstrate the effectiveness of FedCD. The code is available at https://github.com/YunzZ-Liu/FedCD.

----

## [427] BLADE: Box-Level Supervised Amodal Segmentation through Directed Expansion

**Authors**: *Zhaochen Liu, Zhixuan Li, Tingting Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28176](https://doi.org/10.1609/aaai.v38i4.28176)

**Abstract**:

Perceiving the complete shape of occluded objects is essential for human and machine intelligence. While the amodal segmentation task is to predict the complete mask of partially occluded objects, it is time-consuming and labor-intensive to annotate the pixel-level ground truth amodal masks. Box-level supervised amodal segmentation addresses this challenge by relying solely on ground truth bounding boxes and instance classes as supervision, thereby alleviating the need for exhaustive pixel-level annotations. Nevertheless, current box-level methodologies encounter limitations in generating low-resolution masks and imprecise boundaries, failing to meet the demands of practical real-world applications. We present a novel solution to tackle this problem by introducing a directed expansion approach from visible masks to corresponding amodal masks. Our approach involves a hybrid end-to-end network based on the overlapping region - the area where different instances intersect. Diverse segmentation strategies are applied for overlapping regions and non-overlapping regions according to distinct characteristics. To guide the expansion of visible masks, we introduce an elaborately-designed connectivity loss for overlapping regions, which leverages correlations with visible masks and facilitates accurate amodal segmentation. Experiments are conducted on several challenging datasets and the results show that our proposed method can outperform existing state-of-the-art methods with large margins.

----

## [428] Towards Balanced Alignment: Modal-Enhanced Semantic Modeling for Video Moment Retrieval

**Authors**: *Zhihang Liu, Jun Li, Hongtao Xie, Pandeng Li, Jiannan Ge, Sun'ao Liu, Guoqing Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28177](https://doi.org/10.1609/aaai.v38i4.28177)

**Abstract**:

Video Moment Retrieval (VMR) aims to retrieve temporal segments in untrimmed videos corresponding to a given language query by constructing cross-modal alignment strategies. However, these existing strategies are often sub-optimal since they ignore the modality imbalance problem, i.e., the semantic richness inherent in videos far exceeds that of a given limited-length sentence. Therefore, in pursuit of better alignment, a natural idea is enhancing the video modality to filter out query-irrelevant semantics, and enhancing the text modality to capture more segment-relevant knowledge. In this paper, we introduce Modal-Enhanced Semantic Modeling (MESM), a novel framework for more balanced alignment through enhancing features at two levels. First, we enhance the video modality at the frame-word level through word reconstruction. This strategy emphasizes the portions associated with query words in frame-level features while suppressing irrelevant parts. Therefore, the enhanced video contains less redundant semantics and is more balanced with the textual modality. Second, we enhance the textual modality at the segment-sentence level by learning complementary knowledge from context sentences and ground-truth segments. With the knowledge added to the query, the textual modality thus maintains more meaningful semantics and is more balanced with the video modality. By implementing two levels of MESM, the semantic information from both modalities is more balanced to align, thereby bridging the modality gap. Experiments on three widely used benchmarks, including the out-of-distribution settings, show that the proposed framework achieves a new start-of-the-art performance with notable generalization ability (e.g., 4.42% and 7.69% average gains of R1@0.7 on Charades-STA and Charades-CG). The code will be available at https://github.com/lntzm/MESM.

----

## [429] Improving Cross-Modal Alignment with Synthetic Pairs for Text-Only Image Captioning

**Authors**: *Zhiyue Liu, Jinyuan Liu, Fanrong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28178](https://doi.org/10.1609/aaai.v38i4.28178)

**Abstract**:

Although image captioning models have made significant advancements in recent years, the majority of them heavily depend on high-quality datasets containing paired images and texts which are costly to acquire. Previous works leverage the CLIP's cross-modal association ability for image captioning, relying solely on textual information under unsupervised settings. However, not only does a modality gap exist between CLIP text and image features, but a discrepancy also arises between training and inference due to the unavailability of real-world images, which hinders the cross-modal alignment in text-only captioning. This paper proposes a novel method to address these issues by incorporating synthetic image-text pairs. A pre-trained text-to-image model is deployed to obtain images that correspond to textual data, and the pseudo features of generated images are optimized toward the real ones in the CLIP embedding space. Furthermore, textual information is gathered to represent image features, resulting in the image features with various semantics and the bridged modality gap. To unify training and inference, synthetic image features would serve as the training prefix for the language decoder, while real images are used for inference. Additionally, salient objects in images are detected as assistance to enhance the learning of modality alignment. Experimental results demonstrate that our method obtains the state-of-the-art performance on benchmark datasets.

----

## [430] Cell Graph Transformer for Nuclei Classification

**Authors**: *Wei Lou, Guanbin Li, Xiang Wan, Haofeng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28179](https://doi.org/10.1609/aaai.v38i4.28179)

**Abstract**:

Nuclei classification is a critical step in computer-aided diagnosis with histopathology images. In the past, various methods have employed graph neural networks (GNN) to analyze cell graphs that model inter-cell relationships by considering nuclei as vertices. However, they are limited by the GNN mechanism that only passes messages among local nodes via fixed edges. To address the issue, we develop a cell graph transformer (CGT) that treats nodes and edges as input tokens to enable learnable adjacency and information exchange among all nodes. Nevertheless, training the transformer with a cell graph presents another challenge. Poorly initialized features can lead to noisy self-attention scores and inferior convergence, particularly when processing the cell graphs with numerous connections. Thus, we further propose a novel topology-aware pretraining method that leverages a graph convolutional network (GCN) to learn a feature extractor. The pre-trained features may suppress unreasonable correlations and hence ease the finetuning of CGT. Experimental results suggest that the proposed cell graph transformer with topology-aware pretraining significantly improves the nuclei classification results, and achieves the state-of-the-art performance. Code and models are available at https://github.com/lhaof/CGT

----

## [431] Detect Any Keypoints: An Efficient Light-Weight Few-Shot Keypoint Detector

**Authors**: *Changsheng Lu, Piotr Koniusz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28180](https://doi.org/10.1609/aaai.v38i4.28180)

**Abstract**:

Recently the prompt-based models have become popular across various language and vision tasks. Following that trend, we perform few-shot keypoint detection (FSKD) by detecting any keypoints in a query image, given the prompts formed by support images and keypoints. FSKD can be applied to detecting keypoints and poses of diverse animal species. In order to maintain flexibility of detecting varying number of keypoints, existing FSKD approaches modulate query feature map per support keypoint, then detect the corresponding keypoint from each modulated feature via a detection head. Such a separation of modulation-detection makes model heavy and slow when the number of keypoints increases. To overcome this issue, we design a novel light-weight detector which combines modulation and detection into one step, with the goal of reducing the computational cost without the drop of performance. Moreover, to bridge the large domain shift of keypoints between seen and unseen species, we further improve our model with mean feature based contrastive learning to align keypoint distributions, resulting in better keypoint representations for FSKD. Compared to the state of the art, our light-weight detector reduces the number of parameters by 50%, training/test time by 50%, and achieves 5.62% accuracy gain on 1-shot novel keypoint detection in the Animal pose dataset. Our model is also robust to the number of keypoints and saves memory when evaluating a large number of keypoints (e.g., 1000) per episode.

----

## [432] TCNet: Continuous Sign Language Recognition from Trajectories and Correlated Regions

**Authors**: *Hui Lu, Albert Ali Salah, Ronald Poppe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28181](https://doi.org/10.1609/aaai.v38i4.28181)

**Abstract**:

A key challenge in continuous sign language recognition (CSLR) is to efficiently capture long-range spatial interactions over time from the video input. To address this challenge, we propose TCNet, a hybrid network that effectively models spatio-temporal information from Trajectories and Correlated regions. TCNet's trajectory module transforms frames into aligned trajectories composed of continuous visual tokens. This facilitates extracting region trajectory patterns. In addition, for a query token, self-attention is learned along the trajectory. As such, our network can also focus on fine-grained spatio-temporal patterns, such as finger movement, of a region in motion. TCNet's correlation module utilizes a novel dynamic attention mechanism that filters out irrelevant frame regions. Additionally, it assigns dynamic key-value tokens from correlated regions to each query. Both innovations significantly reduce the computation cost and memory. We perform experiments on four large-scale datasets: PHOENIX14, PHOENIX14-T, CSL, and CSL-Daily. Our results demonstrate that TCNet consistently achieves state-of-the-art performance. For example, we improve over the previous state-of-the-art by 1.5\% and 1.0\% word error rate on PHOENIX14 and PHOENIX14-T, respectively. Code is available at https://github.com/hotfinda/TCNet

----

## [433] MLNet: Mutual Learning Network with Neighborhood Invariance for Universal Domain Adaptation

**Authors**: *Yanzuo Lu, Meng Shen, Andy J. Ma, Xiaohua Xie, Jian-Huang Lai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28182](https://doi.org/10.1609/aaai.v38i4.28182)

**Abstract**:

Universal domain adaptation (UniDA) is a practical but challenging problem, in which information about the relation between the source and the target domains is not given for knowledge transfer. Existing UniDA methods may suffer from the problems of overlooking intra-domain variations in the target domain and difficulty in separating between the similar known and unknown class. To address these issues, we propose a novel Mutual Learning Network (MLNet) with neighborhood invariance for UniDA. In our method, confidence-guided invariant feature learning with self-adaptive neighbor selection is designed to reduce the intra-domain variations for more generalizable feature representation. By using the cross-domain mixup scheme for better unknown-class identification, the proposed method compensates for the misidentified known-class errors by mutual learning between the closed-set and open-set classifiers. Extensive experiments on three publicly available benchmarks demonstrate that our method achieves the best results compared to the state-of-the-arts in most cases and significantly outperforms the baseline across all the four settings in UniDA. Code is available at https://github.com/YanzuoLu/MLNet.

----

## [434] Set Prediction Guided by Semantic Concepts for Diverse Video Captioning

**Authors**: *Yifan Lu, Ziqi Zhang, Chunfeng Yuan, Peng Li, Yan Wang, Bing Li, Weiming Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28183](https://doi.org/10.1609/aaai.v38i4.28183)

**Abstract**:

Diverse video captioning aims to generate a set of sentences to describe the given video in various aspects. Mainstream methods are trained with independent pairs of a video and a caption from its ground-truth set without exploiting the intra-set relationship, resulting in low diversity of generated captions. Different from them, we formulate diverse captioning into a semantic-concept-guided set prediction (SCG-SP) problem by fitting the predicted caption set to the ground-truth set, where the set-level relationship is fully captured. Specifically, our set prediction consists of two synergistic tasks, i.e., caption generation and an auxiliary task of concept combination prediction providing extra semantic supervision. Each caption in the set is attached to a concept combination indicating the primary semantic content of the caption and facilitating element alignment in set prediction. Furthermore, we apply a diversity regularization term on concepts to encourage the model to generate semantically diverse captions with various concept combinations. These two tasks share multiple semantics-specific encodings as input, which are obtained by iterative interaction between visual features and conceptual queries. The correspondence between the generated captions and specific concept combinations further guarantees the interpretability of our model. Extensive experiments on benchmark datasets show that the proposed SCG-SP achieves state-of-the-art (SOTA) performance under both relevance and diversity metrics.

----

## [435] Entropy Induced Pruning Framework for Convolutional Neural Networks

**Authors**: *Yiheng Lu, Ziyu Guan, Yaming Yang, Wei Zhao, Maoguo Gong, Cai Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28184](https://doi.org/10.1609/aaai.v38i4.28184)

**Abstract**:

Structured pruning techniques have achieved great compression performance on convolutional neural networks for image classification tasks. However, the majority of existing methods are sensitive with respect to the model parameters, and their pruning results may be unsatisfactory when the original model is trained poorly. That is, they need the original model to be fully trained, to obtain useful weight information. This is time-consuming, and makes the effectiveness of the pruning results dependent on the degree of model optimization. To address the above issue, we propose a novel metric named Average Filter Information Entropy (AFIE). It decomposes the weight matrix of each layer into a low-rank space, and quantifies the filter importance based on the distribution of the normalized eigenvalues. Intuitively, the eigenvalues capture the covariance among filters, and therefore could be a good guide for pruning. Since the distribution of eigenvalues is robust to the updating of parameters, AFIE can yield a stable evaluation for the importance of each filter no matter whether the original model is trained fully. We implement our AFIE-based pruning method for three popular CNN models of AlexNet, VGG-16, and ResNet-50, and test them on three widely-used image datasets MNIST, CIFAR-10, and ImageNet, respectively. The experimental results are encouraging. We surprisingly observe that for our methods, even when the original model is trained with only one epoch, the AFIE score of each filter keeps identical to the results when the model is fully-trained. This fully indicates the effectiveness of the proposed pruning method.

----

## [436] Pano-NeRF: Synthesizing High Dynamic Range Novel Views with Geometry from Sparse Low Dynamic Range Panoramic Images

**Authors**: *Zhan Lu, Qian Zheng, Boxin Shi, Xudong Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28185](https://doi.org/10.1609/aaai.v38i4.28185)

**Abstract**:

Panoramic imaging research on geometry recovery and High Dynamic Range (HDR) reconstruction becomes a trend with the development of Extended Reality (XR). Neural Radiance Fields (NeRF) provide a promising scene representation for both tasks without requiring extensive prior data. How- ever, in the case of inputting sparse Low Dynamic Range (LDR) panoramic images, NeRF often degrades with under-constrained geometry and is unable to reconstruct HDR radiance from LDR inputs. We observe that the radiance from each pixel in panoramic images can be modeled as both a signal to convey scene lighting information and a light source to illuminate other pixels. Hence, we propose the irradiance fields from sparse LDR panoramic images, which increases the observation counts for faithful geometry recovery and leverages the irradiance-radiance attenuation for HDR reconstruction. Extensive experiments demonstrate that the irradiance fields outperform state-of-the-art methods on both geometry recovery and HDR reconstruction and validate their effectiveness. Furthermore, we show a promising byproduct of spatially-varying lighting estimation. The code is available at https://github.com/Lu-Zhan/Pano-NeRF.

----

## [437] ScanERU: Interactive 3D Visual Grounding Based on Embodied Reference Understanding

**Authors**: *Ziyang Lu, Yunqiang Pei, Guoqing Wang, Peiwei Li, Yang Yang, Yinjie Lei, Heng Tao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28186](https://doi.org/10.1609/aaai.v38i4.28186)

**Abstract**:

Aiming to link natural language descriptions to specific regions in a 3D scene represented as 3D point clouds, 3D visual grounding is a very fundamental task for human-robot interaction. The recognition errors can significantly impact the overall accuracy and then degrade the operation of AI systems. Despite their effectiveness, existing methods suffer from the difficulty of low recognition accuracy in cases of multiple adjacent objects with similar appearance. To address this issue, this work intuitively introduces the human-robot interaction as a cue to facilitate the development of 3D visual grounding. Specifically, a new task termed Embodied Reference Understanding (ERU) is first designed for this concern. Then a new dataset called ScanERU is constructed to evaluate the effectiveness of this idea. Different from existing datasets, our ScanERU dataset is the first to cover semi-synthetic scene integration with textual, real-world visual, and synthetic gestural information. Additionally, this paper formulates a heuristic framework based on attention mechanisms and human body movements to enlighten the research of ERU. Experimental results demonstrate the superiority of the proposed method, especially in the recognition of multiple identical objects. Our codes and dataset are available in the ScanERU repository.

----

## [438] MGNet: Learning Correspondences via Multiple Graphs

**Authors**: *Luanyuan Dai, Xiaoyu Du, Hanwang Zhang, Jinhui Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28187](https://doi.org/10.1609/aaai.v38i4.28187)

**Abstract**:

Learning correspondences aims to find correct correspondences (inliers) from the initial correspondence set with an uneven correspondence distribution and a low inlier rate, which can be regarded as graph data. Recent advances usually use graph neural networks (GNNs) to build a single type of graph or simply stack local graphs into the global one to complete the task. But they ignore the complementary relationship between different types of graphs, which can effectively capture potential relationships among sparse correspondences. To address this problem, we propose MGNet to effectively combine multiple complementary graphs. To obtain information integrating implicit and explicit local graphs, we construct local graphs from implicit and explicit aspects and combine them effectively, which is used to build a global graph. Moreover, we propose Graph Soft Degree Attention (GSDA) to make full use of all sparse correspondence information at once in the global graph, which can capture and amplify discriminative features.  Extensive experiments demonstrate that MGNet outperforms state-of-the-art methods in different visual tasks. The code is provided in https://github.com/DAILUANYUAN/MGNet-2024AAAI.

----

## [439] SCP: Spherical-Coordinate-Based Learned Point Cloud Compression

**Authors**: *Ao Luo, Linxin Song, Keisuke Nonaka, Kyohei Unno, Heming Sun, Masayuki Goto, Jiro Katto*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28188](https://doi.org/10.1609/aaai.v38i4.28188)

**Abstract**:

In recent years, the task of learned point cloud compression has gained prominence. An important type of point cloud, LiDAR point cloud, is generated by spinning LiDAR on vehicles. This process results in numerous circular shapes and azimuthal angle invariance features within the point clouds. However, these two features have been largely overlooked by previous methodologies. In this paper, we introduce a model-agnostic method called Spherical-Coordinate-based learned Point cloud compression (SCP), designed to fully leverage the features of circular shapes and azimuthal angle invariance. Additionally, we propose a multi-level Octree for SCP to mitigate the reconstruction error for distant areas within the Spherical-coordinate-based Octree. SCP exhibits excellent universality, making it applicable to various learned point cloud compression techniques. Experimental results demonstrate that SCP surpasses previous state-of-the-art methods by up to 29.14% in point-to-point PSNR BD-Rate.

----

## [440] DLCA-Recon: Dynamic Loose Clothing Avatar Reconstruction from Monocular Videos

**Authors**: *Chunjie Luo, Fei Luo, Yusen Wang, Enxu Zhao, Chunxia Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28189](https://doi.org/10.1609/aaai.v38i4.28189)

**Abstract**:

Reconstructing a dynamic human with loose clothing is an important but difficult task. To address this challenge, we propose a method named DLCA-Recon to create human avatars from monocular videos. The distance from loose clothing to the underlying body rapidly changes in every frame when the human freely moves and acts. Previous methods lack effective geometric initialization and constraints for guiding the optimization of deformation to explain this dramatic change, resulting in the discontinuous and incomplete reconstruction surface.To model the deformation more accurately, we propose to initialize an estimated 3D clothed human in the canonical space, as it is easier for deformation fields to learn from the clothed human than from SMPL.With both representations of explicit mesh and implicit SDF, we utilize the physical connection information between consecutive frames and propose a dynamic deformation field (DDF) to optimize deformation fields. DDF accounts for contributive forces on loose clothing to enhance the interpretability of deformations and effectively capture the free movement of loose clothing. Moreover, we propagate SMPL skinning weights to each individual and refine pose and skinning weights during the optimization to improve skinning transformation. Based on more reasonable initialization and DDF, we can simulate real-world physics more accurately. Extensive experiments on public and our own datasets validate that our method can produce superior results for humans with loose clothing compared to the SOTA methods.

----

## [441] Dual-Window Multiscale Transformer for Hyperspectral Snapshot Compressive Imaging

**Authors**: *Fulin Luo, Xi Chen, Xiuwen Gong, Weiwen Wu, Tan Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28190](https://doi.org/10.1609/aaai.v38i4.28190)

**Abstract**:

Coded aperture snapshot spectral imaging (CASSI) system is an effective manner for hyperspectral snapshot compressive imaging. The core issue of CASSI is to solve the inverse problem for the reconstruction of hyperspectral image (HSI). In recent years, Transformer-based methods achieve promising performance in HSI reconstruction. However, capturing both long-range dependencies and local information while ensuring reasonable computational costs remains a challenging problem. In this paper, we propose a Transformer-based HSI reconstruction method called dual-window multiscale Transformer (DWMT), which is a coarse-to-fine process, reconstructing the global properties of HSI with the long-range dependencies. In our method, we propose a novel U-Net architecture using a dual-branch encoder to refine pixel information and full-scale skip connections to fuse different features, enhancing the extraction of fine-grained features. Meanwhile, we design a novel self-attention mechanism called dual-window multiscale multi-head self-attention (DWM-MSA), which utilizes two different-sized windows to compute self-attention, which can capture the long-range dependencies in a local region at different scales to improve the reconstruction performance. We also propose a novel position embedding method for Transformer, named con-abs position embedding (CAPE), which effectively enhances positional information of the HSIs. Extensive experiments on both the simulated and the real data are conducted to demonstrate the superior performance, stability, and generalization ability of our DWMT. Code of this project is at https://github.com/chenx2000/DWMT.

----

## [442] Electron Microscopy Images as Set of Fragments for Mitochondrial Segmentation

**Authors**: *Naisong Luo, Rui Sun, Yuwen Pan, Tianzhu Zhang, Feng Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28191](https://doi.org/10.1609/aaai.v38i4.28191)

**Abstract**:

Automatic mitochondrial segmentation enjoys great popularity with the development of deep learning. However, the coarse prediction raised by the presence of regular 3D grids in previous methods regardless of 3D CNN or the vision transformers suggest a possibly sub-optimal feature arrangement. To mitigate this limitation, we attempt to interpret the 3D EM image stacks as a set of interrelated 3D fragments for a better solution. However, it is non-trivial to model the 3D fragments without introducing excessive computational overhead. In this paper, we design a coherent fragment vision transformer (FragViT) combined with affinity learning to manipulate features on 3D fragments yet explore mutual relationships to model fragment-wise context, enjoying locality prior without sacrificing global reception. The proposed FragViT includes a fragment encoder and a hierarchical fragment aggregation module. The fragment encoder is equipped with affinity heads to transform the tokens into fragments with homogeneous semantics, and the multi-layer self-attention is used to explicitly learn inter-fragment relations with long-range dependencies. The hierarchical fragment aggregation module is responsible for hierarchically aggregating fragment-wise prediction back to the final voxel-wise prediction in a progressive manner. Extensive experimental results on the challenging MitoEM, Lucchi, and AC3/AC4 benchmarks demonstrate the effectiveness of the proposed method.

----

## [443] DiffusionTrack: Diffusion Model for Multi-Object Tracking

**Authors**: *Run Luo, Zikai Song, Lintao Ma, Jinlin Wei, Wei Yang, Min Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28192](https://doi.org/10.1609/aaai.v38i5.28192)

**Abstract**:

Multi-object tracking (MOT) is a challenging vision task that aims to detect individual objects within a single frame and associate them across multiple frames. Recent MOT approaches can be categorized into two-stage tracking-by-detection (TBD) methods and one-stage joint detection and tracking (JDT) methods. Despite the success of these approaches, they also suffer from common problems, such as harmful global or local inconsistency, poor trade-off between robustness and model complexity, and lack of flexibility in different scenes within the same video. In this paper we propose a simple but robust framework that formulates object detection and association jointly as a consistent denoising diffusion process from paired noise boxes to paired ground-truth boxes. This novel progressive denoising diffusion strategy substantially augments the tracker's effectiveness, enabling it to discriminate between various objects. During the training stage, paired object boxes diffuse from paired ground-truth boxes to random distribution, and the model learns detection and tracking simultaneously by reversing this noising process. In inference, the model refines a set of paired randomly generated boxes to the detection and tracking results in a flexible one-step or multi-step denoising diffusion process. Extensive experiments on three widely used MOT benchmarks, including MOT17, MOT20, and DanceTrack, demonstrate that our approach achieves competitive performance compared to the current state-of-the-art methods. Code is available at https://github.com/RainBowLuoCS/DiffusionTrack.

----

## [444] Devignet: High-Resolution Vignetting Removal via a Dual Aggregated Fusion Transformer with Adaptive Channel Expansion

**Authors**: *Shenghong Luo, Xuhang Chen, Weiwen Chen, Zinuo Li, Shuqiang Wang, Chi-Man Pun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28193](https://doi.org/10.1609/aaai.v38i5.28193)

**Abstract**:

Vignetting commonly occurs as a degradation in images resulting from factors such as lens design, improper lens hood usage, and limitations in camera sensors. This degradation affects image details, color accuracy, and presents challenges in computational photography. Existing vignetting removal algorithms predominantly rely on ideal physics assumptions and hand-crafted parameters, resulting in the ineffective removal of irregular vignetting and suboptimal results. Moreover, the substantial lack of real-world vignetting datasets hinders the objective and comprehensive evaluation of vignetting removal. To address these challenges, we present VigSet, a pioneering dataset for vignetting removal. VigSet includes 983 pairs of both vignetting and vignetting-free high-resolution (over 4k) real-world images under various conditions. In addition, We introduce DeVigNet, a novel frequency-aware Transformer architecture designed for vignetting removal. Through the Laplacian Pyramid decomposition, we propose the Dual Aggregated Fusion Transformer to handle global features and remove vignetting in the low-frequency domain. Additionally, we propose the Adaptive Channel Expansion Module to enhance details in the high-frequency domain. The experiments demonstrate that the proposed model outperforms existing state-of-the-art methods. The code, models, and dataset are available at https://github.com/CXH-Research/DeVigNet.

----

## [445] AdaFormer: Efficient Transformer with Adaptive Token Sparsification for Image Super-resolution

**Authors**: *Xiaotong Luo, Zekun Ai, Qiuyuan Liang, Ding Liu, Yuan Xie, Yanyun Qu, Yun Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28194](https://doi.org/10.1609/aaai.v38i5.28194)

**Abstract**:

Efficient transformer-based models have made remarkable progress in image super-resolution (SR). Most of these works mainly design elaborate structures to accelerate the inference of the transformer, where all feature tokens are propagated equally. However, they ignore the underlying characteristic of image content, i.e., various image regions have distinct restoration difficulties, especially for large images (2K-8K), failing to achieve adaptive inference. In this work, we propose an adaptive token sparsification transformer (AdaFormer) to speed up the model inference for image SR. Specifically, a texture-relevant sparse attention block with parallel global and local branches is introduced, aiming to integrate informative tokens from the global view instead of only in fixed local windows. Then, an early-exit strategy is designed to progressively halt tokens according to the token importance. To estimate the plausibility of each token, we adopt a lightweight confidence estimator, which is constrained by an uncertainty-guided loss to obtain a binary halting mask about the tokens. Experiments on large images have illustrated that our proposal reduces nearly 90% latency against SwinIR on Test8K, while maintaining a comparable performance.

----

## [446] SkipDiff: Adaptive Skip Diffusion Model for High-Fidelity Perceptual Image Super-resolution

**Authors**: *Xiaotong Luo, Yuan Xie, Yanyun Qu, Yun Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28195](https://doi.org/10.1609/aaai.v38i5.28195)

**Abstract**:

It is well-known that image quality assessment usually meets with the problem of perception-distortion (p-d) tradeoff. The existing deep image super-resolution (SR) methods either focus on high fidelity with pixel-level objectives or high perception with generative models. The emergence of diffusion model paves a fresh way for image restoration, which has the potential to offer a brand-new solution for p-d trade-off. We experimentally observed that the perceptual quality and distortion change in an opposite direction with the increase of sampling steps. In light of this property, we propose an adaptive skip diffusion model (SkipDiff), which aims to achieve
high-fidelity perceptual image SR with fewer sampling steps. Specifically, it decouples the sampling procedure into coarse skip approximation and fine skip refinement stages. A coarse-grained skip diffusion is first performed as a high-fidelity prior to obtaining a latent approximation of the full diffusion. Then, a fine-grained skip diffusion is followed to further refine the latent sample for promoting perception, where the fine time steps are adaptively learned by deep reinforcement learning. Meanwhile, this approach also enables faster sampling of diffusion model through skipping the intermediate denoising process to shorten the effective steps of the computation. Extensive experimental results show that our SkipDiff achieves superior perceptual quality with plausible reconstruction accuracy and a faster sampling speed.

----

## [447] Modeling Continuous Motion for 3D Point Cloud Object Tracking

**Authors**: *Zhipeng Luo, Gongjie Zhang, Changqing Zhou, Zhonghua Wu, Qingyi Tao, Lewei Lu, Shijian Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28196](https://doi.org/10.1609/aaai.v38i5.28196)

**Abstract**:

The task of 3D single object tracking (SOT) with LiDAR point clouds is crucial for various applications, such as autonomous driving and robotics. However, existing approaches have primarily relied on appearance matching or motion modeling within only two successive frames, thereby overlooking the long-range continuous motion property of objects in 3D space. To address this issue, this paper presents a novel approach that views each tracklet as a continuous stream: at each timestamp, only the current frame is fed into the network to interact with multi-frame historical features stored in a memory bank, enabling efficient exploitation of sequential information. To achieve effective cross-frame message passing, a hybrid attention mechanism is designed to account for both long-range relation modeling and local geometric feature extraction. Furthermore, to enhance the utilization of multi-frame features for robust tracking, a contrastive sequence enhancement strategy is proposed, which uses ground truth tracklets to augment training sequences and promote discrimination against false positives in a contrastive manner. Extensive experiments demonstrate that the proposed method outperforms the state-of-the-art method by significant margins on multiple benchmarks.

----

## [448] SGFormer: Semantic Graph Transformer for Point Cloud-Based 3D Scene Graph Generation

**Authors**: *Changsheng Lv, Mengshi Qi, Xia Li, Zhengyuan Yang, Huadong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28197](https://doi.org/10.1609/aaai.v38i5.28197)

**Abstract**:

In this paper, we propose a novel model called SGFormer, Semantic Graph TransFormer for point cloud-based 3D scene graph generation. The task aims to parse a point cloud-based scene into a semantic structural graph, with the core challenge of modeling the complex global structure. Existing methods based on graph convolutional networks (GCNs) suffer from the over-smoothing dilemma and can only propagate information from limited neighboring nodes. In contrast, SGFormer uses Transformer layers as the base building block to allow global information passing, with two types of newly-designed layers tailored for the 3D scene graph generation task. Specifically, we introduce the graph embedding layer to best utilize the global information in graph edges while maintaining comparable computation costs. Furthermore, we propose the semantic injection layer to leverage linguistic knowledge from large-scale language model (i.e., ChatGPT), to enhance objects' visual features. We benchmark our SGFormer on the established 3DSSG dataset and achieve a 40.94% absolute improvement in relationship prediction's R@50 and an 88.36% boost on the subset with complex scenes over the state-of-the-art. Our analyses further show SGFormer's superiority in the long-tail and zero-shot scenarios. Our source code is available at https://github.com/Andy20178/SGFormer.

----

## [449] Privileged Prior Information Distillation for Image Matting

**Authors**: *Cheng Lyu, Jiake Xie, Bo Xu, Cheng Lu, Han Huang, Xin Huang, Ming Wu, Chuang Zhang, Yong Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28198](https://doi.org/10.1609/aaai.v38i5.28198)

**Abstract**:

Performance of trimap-free image matting methods is limited when trying to decouple the deterministic and undetermined regions, especially in the scenes where foregrounds are semantically ambiguous, chromaless, or high transmittance. In this paper, we propose a novel framework named Privileged Prior Information Distillation for Image Matting (PPID-IM) that can effectively transfer privileged prior environment-aware information to improve the performance of trimap-free students in solving hard foregrounds. The prior information of trimap regulates only the teacher model during the training stage, while not being fed into the student network during actual inference. To achieve effective privileged cross-modality (i.e. trimap and RGB) information distillation, we introduce a Cross-Level Semantic Distillation (CLSD) module that reinforces the students with more knowledgeable semantic representations and environment-aware information. We also propose an Attention-Guided Local Distillation module that efficiently transfers privileged local attributes from the trimap-based teacher to trimap-free students for the guidance of local-region optimization. Extensive experiments demonstrate the effectiveness and superiority of our PPID on image matting. The code will be released soon.

----

## [450] FedST: Federated Style Transfer Learning for Non-IID Image Segmentation

**Authors**: *Boyuan Ma, Xiang Yin, Jing Tan, Yongfeng Chen, Haiyou Huang, Hao Wang, Weihua Xue, Xiaojuan Ban*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28199](https://doi.org/10.1609/aaai.v38i5.28199)

**Abstract**:

Federated learning collaboratively trains machine learning models among different clients while keeping data privacy and has become the mainstream for breaking data silos. However, the non-independently and identically distribution (i.e., Non-IID) characteristic of different image domains among different clients reduces the benefits of federated learning and has become a bottleneck problem restricting the accuracy and generalization of federated models. In this work, we propose a novel federated image segmentation method based on style transfer, FedST, by using a denoising diffusion probabilistic model to achieve feature disentanglement and image synthesis of cross-domain image data between multiple clients. Thus it can share style features among clients while protecting structure features of image data, which effectively alleviates the influence of the Non-IID phenomenon. Experiments prove that our method achieves superior segmentation performance compared to state-of-art methods among four different Non-IID datasets in objective and subjective assessment. The code is available at https://github.com/YoferChen/FedST.

----

## [451] SlowTrack: Increasing the Latency of Camera-Based Perception in Autonomous Driving Using Adversarial Examples

**Authors**: *Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28200](https://doi.org/10.1609/aaai.v38i5.28200)

**Abstract**:

In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.

----

## [452] Uncertainty-Aware GAN for Single Image Super Resolution

**Authors**: *Chenxi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28201](https://doi.org/10.1609/aaai.v38i5.28201)

**Abstract**:

Generative adversarial network (GAN) has become a popular tool in the perceptual-oriented single image super-resolution (SISR) for its excellent capability to hallucinate details. However, the performance of most GAN-based SISR methods is impeded due to the limited discriminative ability of their discriminators. In specific, these discriminators only focus on the global image reconstruction quality and ignore the more fine-grained reconstruction quality for constraining the generator, as they predict the overall realness of an image instead of the pixel-level realness. Here, we first introduce the uncertainty into the GAN and propose an Uncertainty-aware GAN (UGAN) to regularize SISR solutions, where the challenging pixels with large reconstruction uncertainty and importance (e.g., texture and edge) are prioritized for optimization. The uncertainty-aware adversarial training strategy enables the discriminator to capture the pixel-level SR uncertainty, which constrains the generator to focus on image areas with high reconstruction difficulty, meanwhile, it improves the interpretability of the SR. To balance weights of multiple training losses, we introduce an uncertainty-aware loss weighting strategy to adaptively learn the optimal loss weights. Extensive experiments demonstrate the effectiveness of our approach in extracting the SR uncertainty and the superiority of the UGAN over the state-of-the-arts in terms of the reconstruction accuracy and perceptual quality.

----

## [453] Stitching Segments and Sentences towards Generalization in Video-Text Pre-training

**Authors**: *Fan Ma, Xiaojie Jin, Heng Wang, Jingjia Huang, Linchao Zhu, Yi Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28202](https://doi.org/10.1609/aaai.v38i5.28202)

**Abstract**:

Video-language pre-training models have recently achieved remarkable results on various multi-modal downstream tasks. However, most of these models rely on contrastive learning or masking modeling to align global features across modalities, neglecting the local associations between video frames and text tokens. This limits the model’s ability to perform fine-grained matching and generalization, especially for tasks that selecting segments in long videos based on query texts. To address this issue, we propose a novel stitching and matching pre-text task for video-language pre-training that encourages fine-grained interactions between modalities. Our task involves stitching video frames or sentences into longer sequences and predicting the positions of cross-model queries in the stitched sequences. The individual frame and sentence representations are thus aligned via the stitching and matching strategy, encouraging the fine-grained interactions between videos and texts. in the stitched sequences for the cross-modal query. We conduct extensive experiments on various benchmarks covering text-to-video retrieval, video question answering, video captioning, and moment retrieval. Our results demonstrate that the proposed method significantly improves the generalization capacity of the video-text pre-training models.

----

## [454] Image Captioning with Multi-Context Synthetic Data

**Authors**: *Feipeng Ma, Yizhou Zhou, Fengyun Rao, Yueyi Zhang, Xiaoyan Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28203](https://doi.org/10.1609/aaai.v38i5.28203)

**Abstract**:

Image captioning requires numerous annotated image-text pairs, resulting in substantial annotation costs. Recently, large models (e.g. diffusion models and large language models) have excelled in producing high-quality images and text. This potential can be harnessed to create synthetic image-text pairs for training captioning models. Synthetic data can improve cost and time efficiency in data collection, allow for customization to specific domains, bootstrap generalization capability for zero-shot performance, and circumvent privacy concerns associated with real-world data. However, existing methods struggle to attain satisfactory performance solely through synthetic data. We identify the issue as generated images from simple descriptions mostly capture a solitary perspective with limited context, failing to align with the intricate scenes prevalent in real-world imagery. To tackle this, we present an innovative pipeline that introduces multi-context data generation. Beginning with an initial text corpus, our approach employs a large language model to extract multiple sentences portraying the same scene from diverse viewpoints. These sentences are then condensed into a single sentence with multiple contexts. Subsequently, we generate intricate images using the condensed captions through diffusion models. Our model is exclusively trained on synthetic image-text pairs crafted through this process. The effectiveness of our pipeline is validated through experimental results in both the in-domain and cross-domain settings, where it achieves state-of-the-art performance on well-known datasets such as MSCOCO, Flickr30k, and NoCaps.

----

## [455] Directed Diffusion: Direct Control of Object Placement through Attention Guidance

**Authors**: *Wan-Duo Kurt Ma, Avisek Lahiri, John P. Lewis, Thomas Leung, W. Bastiaan Kleijn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28204](https://doi.org/10.1609/aaai.v38i5.28204)

**Abstract**:

Text-guided diffusion models such as DALLE-2, Imagen, and Stable Diffusion are able to generate an effectively endless variety of images given only a short text prompt describing the desired image content. In many cases the images are of very high quality. However, these models often struggle to compose scenes containing several key objects such as characters in specified positional relationships. The missing capability to ``direct'' the placement of characters and objects both within and across images is crucial in storytelling, as recognized in the literature on film and animation theory. In this work, we take a particularly straightforward approach to providing the needed direction. Drawing on the observation that the cross-attention maps for prompt words reflect the spatial layout of objects denoted by those words, we introduce an optimization objective that produces ``activation'' at desired positions in these cross-attention maps. The resulting approach is a step toward generalizing the applicability of text-guided diffusion models beyond single images to collections of related images, as in storybooks. Directed Diffusion provides easy high-level positional control over multiple objects, while making use of an existing pre-trained model and maintaining a coherent blend between the positioned objects and the background. Moreover, it requires only a few lines to implement.

----

## [456] Unifying Visual and Vision-Language Tracking via Contrastive Learning

**Authors**: *Yinchao Ma, Yuyang Tang, Wenfei Yang, Tianzhu Zhang, Jinpeng Zhang, Mengxue Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28205](https://doi.org/10.1609/aaai.v38i5.28205)

**Abstract**:

Single object tracking aims to locate the target object in a video sequence according to the state specified by different modal references, including the initial bounding box (BBOX), natural language (NL), or both (NL+BBOX). Due to the gap between different modalities, most existing trackers are designed for single or partial of these reference settings and overspecialize on the specific modality. Differently, we present a unified tracker called UVLTrack, which can simultaneously handle all three reference settings (BBOX, NL, NL+BBOX) with the same parameters. The proposed UVLTrack enjoys several merits. First, we design a modality-unified feature extractor for joint visual and language feature learning and propose a multi-modal contrastive loss to align the visual and language features into a unified semantic space. Second, a modality-adaptive box head is proposed, which makes full use of the target reference to mine ever-changing scenario features dynamically from video contexts and distinguish the target in a contrastive way, enabling robust performance in different reference settings. Extensive experimental results demonstrate that UVLTrack achieves promising performance on seven visual tracking datasets, three vision-language tracking datasets, and three visual grounding datasets. Codes and models will be open-sourced at https://github.com/OpenSpaceAI/UVLTrack.

----

## [457] Follow Your Pose: Pose-Guided Text-to-Video Generation Using Pose-Free Videos

**Authors**: *Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, Qifeng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28206](https://doi.org/10.1609/aaai.v38i5.28206)

**Abstract**:

Generating text-editable and pose-controllable character videos have an imperious demand in creating various digital human. Nevertheless, this task has been restricted by the absence of a comprehensive dataset featuring paired video-pose captions and the generative prior models for videos. In this work, we design a novel two-stage training scheme that can utilize easily obtained datasets (i.e., image pose pair and pose-free video) and the pre-trained text-to-image (T2I) model to obtain the pose-controllable character videos. Specifically, in the first stage, only the keypoint image pairs are used only for a controllable text-to-image generation. We learn a zero-initialized convolutional encoder to encode the pose information. In the second stage, we finetune the motion of the above network via a pose-free video dataset by adding the learnable temporal self-attention and reformed cross-frame self-attention blocks. Powered by our new designs, our method successfully generates continuously pose-controllable character videos while keeps the editing and concept composition ability of the pre-trained T2I model. The code and models are available on https://follow-your-pose.github.io/.

----

## [458] Let All Be Whitened: Multi-Teacher Distillation for Efficient Visual Retrieval

**Authors**: *Zhe Ma, Jianfeng Dong, Shouling Ji, Zhenguang Liu, Xuhong Zhang, Zonghui Wang, Sifeng He, Feng Qian, Xiaobo Zhang, Lei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28207](https://doi.org/10.1609/aaai.v38i5.28207)

**Abstract**:

Visual retrieval aims to search for the most relevant visual items, e.g., images and videos, from a candidate gallery with a given query item. Accuracy and efficiency are two competing objectives in retrieval tasks. Instead of crafting a new method pursuing further improvement on accuracy, in this paper we propose a multi-teacher distillation framework Whiten-MTD, which is able to transfer knowledge from off-the-shelf pre-trained retrieval models to a lightweight student model for efficient visual retrieval. Furthermore, we discover that the similarities obtained by different retrieval models are diversified and incommensurable, which makes it challenging to jointly distill knowledge from multiple models. Therefore, we propose to whiten the output of teacher models before fusion, which enables effective multi-teacher distillation for retrieval models. Whiten-MTD is conceptually simple and practically effective. Extensive experiments on two landmark image retrieval datasets and one video retrieval dataset demonstrate the effectiveness of our proposed method, and its good balance of retrieval performance and efficiency. Our source code is released at https://github.com/Maryeon/whiten_mtd.

----

## [459] Cross-Layer and Cross-Sample Feature Optimization Network for Few-Shot Fine-Grained Image Classification

**Authors**: *Zhen-Xiang Ma, Zhen-Duo Chen, Li-Jun Zhao, Zi-Chao Zhang, Xin Luo, Xin-Shun Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28208](https://doi.org/10.1609/aaai.v38i5.28208)

**Abstract**:

Recently, a number of Few-Shot Fine-Grained Image Classification (FS-FGIC) methods have been proposed, but they primarily focus on better fine-grained feature extraction while overlooking two important issues. The first one is how to extract discriminative features for Fine-Grained Image Classification tasks while reducing trivial and non-generalizable sample level noise introduced in this procedure, to overcome the over-fitting problem under the setting of Few-Shot Learning. The second one is how to achieve satisfying feature matching between limited support and query samples with variable spatial positions and angles. To address these issues, we propose a novel Cross-layer and Cross-sample feature optimization Network for FS-FGIC, C2-Net for short. The proposed method consists of two main modules: Cross-Layer Feature Refinement (CLFR) module and Cross-Sample Feature Adjustment (CSFA) module. The CLFR module further refines the extracted features while integrating outputs from multiple layers to suppress sample-level feature noise interference. Additionally, the CSFA module addresses the feature mismatch between query and support samples through both channel activation and position matching operations. Extensive experiments have been conducted on five fine-grained benchmark datasets, and the results show that the C2-Net outperforms other state-of-the-art methods by a significant margin in most cases. Our code is available at: https://github.com/zenith0923/C2-Net.

----

## [460] LMD: Faster Image Reconstruction with Latent Masking Diffusion

**Authors**: *Zhiyuan Ma, Zhihuan Yu, Jianjun Li, Bowen Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28209](https://doi.org/10.1609/aaai.v38i5.28209)

**Abstract**:

As a class of fruitful approaches, diffusion probabilistic models (DPMs) have shown excellent advantages in high-resolution image reconstruction. On the other hand, masked autoencoders (MAEs), as popular self-supervised vision learners, have demonstrated simpler and more effective image reconstruction and transfer capabilities on downstream tasks. However, they all require extremely high training costs, either due to inherent high temporal-dependence (i.e., excessively long diffusion steps) or due to artificially low spatial-dependence (i.e., human-formulated high mask ratio, such as 0.75). To the end, this paper presents LMD, a faster image reconstruction framework with Latent Masking Diffusion. First, we propose to project and reconstruct images in latent space through a pre-trained variational autoencoder, which is theoretically more efficient than in the pixel-based space. Then, we combine the advantages of MAEs and DPMs to design a progressive masking diffusion model, which gradually increases the masking proportion by three different schedulers and reconstructs the latent features from simple to difficult, without sequentially performing denoising diffusion as in DPMs or using fixed high masking ratio as in MAEs, so as to alleviate the high training time-consumption predicament. Our approach allows for learning high-capacity models and accelerate their training (by 3x or more) and barely reduces the original accuracy. Inference speed in downstream tasks also significantly outperforms the previous  approaches.

----

## [461] AdapEdit: Spatio-Temporal Guided Adaptive Editing Algorithm for Text-Based Continuity-Sensitive Image Editing

**Authors**: *Zhiyuan Ma, Guoli Jia, Bowen Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28210](https://doi.org/10.1609/aaai.v38i5.28210)

**Abstract**:

With the great success of text-conditioned diffusion models in creative text-to-image generation, various text-driven image editing approaches have attracted the attentions of many researchers. However, previous works mainly focus on discreteness-sensitive instructions such as adding, removing or replacing specific objects, background elements or global styles (i.e., “hard editing”), while generally ignoring subject-binding but semantically fine-changing continuity-sensitive instructions such as actions, poses or adjectives, and so on (i.e., “soft editing”), which hampers generative AI from generating user-customized visual contents. To mitigate this predicament, we propose a spatio-temporal guided adaptive editing algorithm AdapEdit, which realizes adaptive image editing by introducing a soft-attention strategy to dynamically vary the guiding degree from the editing conditions to visual pixels from both temporal and spatial perspectives. Note our approach has a significant advantage in preserving model priors and does not require model training, fine-tuning, extra data, or optimization. We present our results over a wide variety of raw images and editing instructions, demonstrating competitive performance and showing it significantly outperforms the previous approaches. Code is available: https://github.com/AnonymousPony/adap-edit.

----

## [462] Pay Attention to Target: Relation-Aware Temporal Consistency for Domain Adaptive Video Semantic Segmentation

**Authors**: *Huayu Mai, Rui Sun, Yuan Wang, Tianzhu Zhang, Feng Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28211](https://doi.org/10.1609/aaai.v38i5.28211)

**Abstract**:

Video semantic segmentation has achieved conspicuous achievements attributed to the development of deep learning, but suffers from labor-intensive annotated training data gathering. To alleviate the data-hunger issue, domain adaptation approaches are developed in the hope of adapting the model trained on the labeled synthetic videos to the real videos in the absence of annotations. By analyzing the dominant paradigm consistency regularization in the domain adaptation task, we find that the bottlenecks exist in previous methods from the perspective of pseudo-labels. To take full advantage of the information contained in the pseudo-labels and empower more effective supervision signals, we propose a coherent PAT network including a target domain focalizer and relation-aware temporal consistency. The proposed PAT network enjoys several merits. First, the target domain focalizer is responsible for paying attention to the target domain, and increasing the accessibility of pseudo-labels in consistency training. Second, the relation-aware temporal consistency aims at modeling the inter-class consistent relationship across frames to equip the model with effective supervision signals. Extensive experimental results on two challenging benchmarks demonstrate that our method performs favorably against state-of-the-art domain adaptive video semantic segmentation methods.

----

## [463] Improving Automatic VQA Evaluation Using Large Language Models

**Authors**: *Oscar Mañas, Benno Krojer, Aishwarya Agrawal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28212](https://doi.org/10.1609/aaai.v38i5.28212)

**Abstract**:

8 years after the visual question answering (VQA) task was proposed, accuracy remains the primary metric for automatic evaluation. VQA Accuracy has been effective so far in the IID evaluation setting. However, our community is undergoing a shift towards open-ended generative models and OOD evaluation. In this new paradigm, the existing VQA Accuracy metric is overly stringent and underestimates the performance of VQA systems. Thus, there is a need to develop more robust automatic VQA metrics that serve as a proxy for human judgment. In this work, we propose to leverage the in-context learning capabilities of instruction-tuned large language models (LLMs) to build a better VQA metric. We formulate VQA evaluation as an answer-rating task where the LLM is instructed to score the accuracy of a candidate answer given a set of reference answers. We demonstrate the proposed metric better correlates with human judgment compared to existing metrics across several VQA models and benchmarks. We hope wide adoption of our metric will contribute to better estimating the research progress on the VQA task. We plan to release the evaluation code and collected human judgments.

----

## [464] Inconsistency-Based Data-Centric Active Open-Set Annotation

**Authors**: *Ruiyu Mao, Ouyang Xu, Yunhui Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28213](https://doi.org/10.1609/aaai.v38i5.28213)

**Abstract**:

Active learning, a method to reduce labeling effort for training deep neural networks, is often limited by the assumption that all unlabeled data belong to known classes. This closed-world assumption fails in practical scenarios with unknown classes in the data, leading to active open-set annotation challenges. Existing methods struggle with this uncertainty. We introduce NEAT, a novel, computationally efficient, data-centric active learning approach for open-set data. NEAT differentiates and labels known classes from a mix of known and unknown classes, using a clusterability criterion and a consistency mea- sure that detects inconsistencies between model predictions and feature distribution. In contrast to recent learning-centric solutions, NEAT shows superior performance in active open- set annotation, as our experiments confirm. Additional details on the further evaluation metrics, implementation, and archi- tecture of our method can be found in the public document at https://arxiv.org/pdf/2401.04923.pdf.

----

## [465] Progressive High-Frequency Reconstruction for Pan-Sharpening with Implicit Neural Representation

**Authors**: *Ge Meng, Jingjia Huang, Yingying Wang, Zhenqi Fu, Xinghao Ding, Yue Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28214](https://doi.org/10.1609/aaai.v38i5.28214)

**Abstract**:

Pan-sharpening aims to leverage the high-frequency signal of the panchromatic (PAN) image to enhance the resolution of its corresponding multi-spectral (MS) image. However, deep neural networks (DNNs) tend to prioritize learning the low-frequency components during the training process, which limits the restoration of high-frequency edge details in MS images. To overcome this limitation, we treat pan-sharpening as a coarse-to-fine high-frequency restoration problem and propose a novel method for achieving high-quality restoration of edge information in MS images. Specifically, to effectively obtain fine-grained multi-scale contextual features, we design a Band-limited Multi-scale High-frequency Generator (BMHG) that generates high-frequency signals from the PAN image within different bandwidths. During training, higher-frequency signals are progressively injected into the MS image, and corresponding residual blocks are introduced into the network simultaneously. This design enables gradients to flow from later to earlier blocks smoothly, encouraging intermediate blocks to concentrate on missing details. Furthermore, to address the issue of pixel position misalignment arising from multi-scale features fusion, we propose a Spatial-spectral Implicit Image Function (SIIF) that employs implicit neural representation to effectively represent and fuse spatial and spectral features in the continuous domain. Extensive experiments on different datasets demonstrate that our method outperforms existing approaches in terms of quantitative and visual measurements for high-frequency detail recovery.

----

## [466] NaMa: Neighbor-Aware Multi-Modal Adaptive Learning for Prostate Tumor Segmentation on Anisotropic MR Images

**Authors**: *Runqi Meng, Xiao Zhang, Shijie Huang, Yuning Gu, Guiqin Liu, Guangyu Wu, Nizhuan Wang, Kaicong Sun, Dinggang Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28215](https://doi.org/10.1609/aaai.v38i5.28215)

**Abstract**:

Accurate segmentation of prostate tumors from multi-modal magnetic resonance (MR) images is crucial for diagnosis and treatment of prostate cancer. However, the robustness of existing segmentation methods is limited, mainly because these methods 1) fail to adaptively assess subject-specific information of each MR modality for accurate tumor delineation, and 2) lack effective utilization of inter-slice information across thick slices in MR images to segment tumor as a whole 3D volume. In this work, we propose a two-stage neighbor-aware multi-modal adaptive learning network (NaMa) for accurate prostate tumor segmentation from multi-modal anisotropic MR images. In particular, in the first stage, we apply subject-specific multi-modal fusion in each slice by developing a novel modality-informativeness adaptive learning (MIAL) module for selecting and adaptively fusing informative representation of each modality based on inter-modality correlations. In the second stage, we exploit inter-slice feature correlations to derive volumetric tumor segmentation. Specifically, we first use a Unet variant with sequence layers to coarsely capture slice relationship at a global scale, and further generate an activation map for each slice. Then, we introduce an activation mapping guidance (AMG) module to refine slice-wise representation (via information from adjacent slices) for consistent tumor segmentation across neighboring slices. Besides, during the network training, we further apply a random mask strategy to each MR modality to improve feature representation efficiency. Experiments on both in-house and public (PICAI) multi-modal prostate tumor datasets show that our proposed NaMa performs better than state-of-the-art methods.

----

## [467] ConVQG: Contrastive Visual Question Generation with Multimodal Guidance

**Authors**: *Li Mi, Syrielle Montariol, Javiera Castillo Navarro, Xianjie Dai, Antoine Bosselut, Devis Tuia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28216](https://doi.org/10.1609/aaai.v38i5.28216)

**Abstract**:

Asking questions about visual environments is a crucial way for intelligent agents to understand rich multi-faceted scenes, raising the importance of Visual Question Generation (VQG) systems. Apart from being grounded to the image, existing VQG systems can use textual constraints, such as expected answers or knowledge triplets, to generate focused questions. These constraints allow VQG systems to specify the question content or leverage external commonsense knowledge that can not be obtained from the image content only. However, generating focused questions using textual constraints while enforcing a high relevance to the image content remains a challenge, as VQG systems often ignore one or both forms of grounding. In this work, we propose Contrastive Visual Question Generation (ConVQG), a method using a dual contrastive objective to discriminate questions generated using both modalities from those based on a single one. Experiments on both knowledge-aware and standard VQG benchmarks demonstrate that ConVQG outperforms the state-of-the-art methods and generates image-grounded, text-guided, and knowledge-rich questions. Our human evaluation results also show preference for ConVQG questions compared to non-contrastive baselines.

----

## [468] Out-of-Distribution Detection in Long-Tailed Recognition with Calibrated Outlier Class Learning

**Authors**: *Wenjun Miao, Guansong Pang, Xiao Bai, Tianqi Li, Jin Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28217](https://doi.org/10.1609/aaai.v38i5.28217)

**Abstract**:

Existing out-of-distribution (OOD) methods have shown great success on balanced datasets but become ineffective in long-tailed recognition (LTR) scenarios where 1) OOD samples are often wrongly classified into head classes and/or 2) tail-class samples are treated as OOD samples. To address these issues, current studies fit a prior distribution of auxiliary/pseudo OOD data to the long-tailed in-distribution (ID) data. However, it is difficult to obtain such an accurate prior distribution given the unknowingness of real OOD samples and heavy class imbalance in LTR. A straightforward solution to avoid the requirement of this prior is to learn an outlier class to encapsulate the OOD samples. The main challenge is then to tackle the aforementioned confusion between OOD samples and head/tail-class samples when learning the outlier class. To this end, we introduce a novel calibrated outlier class learning (COCL) approach, in which 1) a debiased large margin learning method is introduced in the outlier class learning to distinguish OOD samples from both head and tail classes in the representation space and 2) an outlier-class-aware logit calibration method is defined to enhance the long-tailed classification confidence. Extensive empirical results on three popular benchmarks CIFAR10-LT, CIFAR100-LT, and ImageNet-LT demonstrate that COCL substantially outperforms existing state-of-the-art OOD detection methods in LTR while being able to improve the classification accuracy on ID data.  Code is available at https://github.com/mala-lab/COCL.

----

## [469] BCLNet: Bilateral Consensus Learning for Two-View Correspondence Pruning

**Authors**: *Xiangyang Miao, Guobao Xiao, Shiping Wang, Jun Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28218](https://doi.org/10.1609/aaai.v38i5.28218)

**Abstract**:

Correspondence pruning aims to establish reliable correspondences between two related images and recover relative camera motion. Existing approaches often employ a progressive strategy to handle the local and global contexts, with a prominent emphasis on transitioning from local to global, resulting in the neglect of interactions between different contexts. To tackle this issue, we propose a parallel context learning strategy that involves acquiring bilateral consensus for the two-view correspondence pruning task. In our approach, we design a distinctive self-attention block to capture global context and parallel process it with the established local context learning module, which enables us to simultaneously capture both local and global consensuses. By combining these local and global consensuses, we derive the required bilateral consensus. We also design a recalibration block, reducing the influence of erroneous consensus information and enhancing the robustness of the model. The culmination of our efforts is the Bilateral Consensus Learning Network (BCLNet), which efficiently estimates camera pose and identifies inliers (true correspondences). Extensive experiments results demonstrate that our network not only surpasses state-of-the-art methods on benchmark datasets but also showcases robust generalization abilities across various feature extraction techniques. Noteworthily, BCLNet obtains significant improvement gains over the second best method on unknown outdoor dataset, and obviously accelerates model training speed.

----

## [470] Understanding the Role of the Projector in Knowledge Distillation

**Authors**: *Roy Miles, Krystian Mikolajczyk*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28219](https://doi.org/10.1609/aaai.v38i5.28219)

**Abstract**:

In this paper we revisit the efficacy of knowledge distillation as a function matching and metric learning problem. In doing so we verify three important design decisions, namely the normalisation, soft maximum function, and projection layers as key ingredients. We theoretically show that the projector implicitly encodes information on past examples, enabling relational gradients for the student. We then show that the normalisation of representations is tightly coupled with the training dynamics of this projector, which can have a large impact on the students performance. Finally, we show that a simple soft maximum function can be used to address any significant capacity gap problems. Experimental results on various benchmark datasets demonstrate that using these insights can lead to superior or comparable performance to state-of-the-art knowledge distillation techniques, despite being much more computationally efficient. In particular, we obtain these results across image classification (CIFAR100 and ImageNet), object detection (COCO2017), and on more difficult distillation objectives, such as training data efficient transformers, whereby we attain a 77.2% top-1 accuracy with DeiT-Ti on ImageNet. Code and models are publicly available.

----

## [471] Robust Blind Text Image Deblurring via Maximum Consensus Framework

**Authors**: *Zijian Min, Gundu Mohamed Hassan, Geun-Sik Jo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28220](https://doi.org/10.1609/aaai.v38i5.28220)

**Abstract**:

The blind text image deblurring problem presents a formidable challenge, requiring the recovery of a clean and sharp text image from a blurry version with an unknown blur kernel. Sparsity-based strategies have demonstrated their efficacy by emphasizing the sparse priors of the latent image and kernel. However, these existing strategies have largely neglected the influence of additional noise, imposing limitations on their performance. To overcome this limitation, we propose a novel framework designed to effectively mitigate the impact of extensive noise prevalent in blurred images. Our approach centers around a robust Maximum Consensus Framework, wherein we optimize the quantity of interest from the noisy blurry image based on the maximum consensus criterion. Furthermore, we propose the integration of the Alternating Direction Method of Multipliers (ADMM) and the Half-Quadratic Splitting (HQS) method to address the computationally intractable L0 norm problem. This innovative strategy enables improvements in the deblurring performance of blurry text images with the additional synthetic noise. Experimental evaluations conducted on various noisy blurry text images demonstrate the superiority of the proposed approach over existing methods.

----

## [472] Knowledge Guided Semi-supervised Learning for Quality Assessment of User Generated Videos

**Authors**: *Shankhanil Mitra, Rajiv Soundararajan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28221](https://doi.org/10.1609/aaai.v38i5.28221)

**Abstract**:

Perceptual quality assessment of user generated content (UGC) videos is challenging due to the requirement of large scale human annotated videos for training. In this work, we address this challenge by first designing a self-supervised Spatio-Temporal Visual Quality Representation Learning (ST-VQRL) framework to generate robust quality aware features for videos. Then, we propose a dual-model based Semi Supervised Learning (SSL) method specifically designed for the Video Quality Assessment (SSL-VQA) task, through a novel knowledge transfer of quality predictions between the two models. Our SSL-VQA method uses the ST-VQRL backbone to produce robust performances across various VQA datasets including cross-database settings, despite being learned with limited human annotated videos. Our model improves the state-of-the-art performance when trained only with limited data by around 10%, and by around 15% when unlabelled data is also used in SSL. Source codes and checkpoints are available at https://github.com/Shankhanil006/SSL-VQA.

----

## [473] Bridging the Gap between 2D and 3D Visual Question Answering: A Fusion Approach for 3D VQA

**Authors**: *Wentao Mo, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28222](https://doi.org/10.1609/aaai.v38i5.28222)

**Abstract**:

In 3D Visual Question Answering (3D VQA), the scarcity of fully annotated data and limited visual content diversity hampers the generalization to novel scenes and 3D concepts (e.g., only around 800 scenes are utilized in ScanQA and SQA dataset). Current approaches resort supplement 3D reasoning with 2D information. However, these methods face challenges: either they use top-down 2D views that introduce overly complex and sometimes question-irrelevant visual clues, or they rely on globally aggregated scene/image-level representations from 2D VLMs, losing the fine-grained vision-language correlations. To overcome these limitations, our approach utilizes question-conditional 2D view selection procedure, pinpointing semantically relevant 2D inputs for crucial visual clues. We then integrate this 2D knowledge into the 3D-VQA system via a two-branch Transformer structure. This structure, featuring a Twin-Transformer design, compactly combines 2D and 3D modalities and captures fine-grained correlations between modalities, allowing them mutually augmenting each other. Integrating proposed mechanisms above, we present BridgeQA, that offers a fresh perspective on multi-modal transformer-based architectures for 3D-VQA. Experiments validate that BridgeQA achieves state-of-the-art on 3D-VQA datasets and significantly outperforms existing solutions. Code is available at https://github.com/matthewdm0816/BridgeQA.

----

## [474] Augmented Commonsense Knowledge for Remote Object Grounding

**Authors**: *Bahram Mohammadi, Yicong Hong, Yuankai Qi, Qi Wu, Shirui Pan, Javen Qinfeng Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28223](https://doi.org/10.1609/aaai.v38i5.28223)

**Abstract**:

The vision-and-language navigation (VLN) task necessitates an agent to perceive the surroundings, follow natural language instructions, and act in photo-realistic unseen environments. Most of the existing methods employ the entire image or object features to represent navigable viewpoints. However, these representations are insufficient for proper action prediction, especially for the REVERIE task, which uses concise high-level instructions, such as “Bring me the blue cushion in the master bedroom”. To address enhancing representation, we propose an augmented commonsense knowledge model (ACK) to leverage commonsense information as a spatio-temporal knowledge graph for improving agent navigation. Specifically, the proposed approach involves constructing a knowledge base by retrieving commonsense information from ConceptNet, followed by a refinement module to remove noisy and irrelevant knowledge. We further present ACK which consists of knowledge graph-aware cross-modal and concept aggregation modules to enhance visual representation and visual-textual data alignment by integrating visible objects, commonsense knowledge, and concept history, which includes object and knowledge temporal information. Moreover, we add a new pipeline for the commonsense-based decision-making process which leads to more accurate local action prediction. Experimental results demonstrate our proposed model noticeably outperforms the baseline and archives the state-of-the-art on the REVERIE benchmark. The source code is available at https://github.com/Bahram-Mohammadi/ACK.

----

## [475] Recurrent Partial Kernel Network for Efficient Optical Flow Estimation

**Authors**: *Henrique Morimitsu, Xiaobin Zhu, Xiangyang Ji, Xu-Cheng Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28224](https://doi.org/10.1609/aaai.v38i5.28224)

**Abstract**:

Optical flow estimation is a challenging task consisting of predicting per-pixel motion vectors between images. Recent methods have employed larger and more complex models to improve the estimation accuracy. However, this impacts the widespread adoption of optical flow methods and makes it harder to train more general models since the optical flow data is hard to obtain. This paper proposes a small and efficient model for optical flow estimation. We design a new spatial recurrent encoder that extracts discriminative features at a significantly reduced size. Unlike standard recurrent units, we utilize Partial Kernel Convolution (PKConv) layers to produce variable multi-scale features with a single shared block. We also design efficient Separable Large Kernels (SLK) to capture large context information with low computational cost. Experiments on public benchmarks show that we achieve state-of-the-art generalization performance while requiring significantly fewer parameters and memory than competing methods. Our model ranks first in the Spring benchmark without finetuning, improving the results by over 10% while requiring an order of magnitude fewer FLOPs and over four times less memory than the following published method without finetuning. The code is available at github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/rpknet.

----

## [476] TETRIS: Towards Exploring the Robustness of Interactive Segmentation

**Authors**: *Andrey Moskalenko, Vlad Shakhuro, Anna Vorontsova, Anton Konushin, Anton Antonov, Alexander Krapukhin, Denis Shepelev, Konstantin Soshin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28225](https://doi.org/10.1609/aaai.v38i5.28225)

**Abstract**:

Interactive segmentation methods rely on user inputs to iteratively update the selection mask. A click specifying the object of interest is arguably the most simple and intuitive interaction type, and thereby the most common choice for interactive segmentation. However, user clicking patterns in the interactive segmentation context remain unexplored. Accordingly, interactive segmentation evaluation strategies rely more on intuition and common sense rather than empirical studies (e.g., assuming that users tend to click in the center of the area with the largest error). In this work, we conduct a real-user study to investigate real user clicking patterns. This study reveals that the intuitive assumption made in the common evaluation strategy may not hold. As a result, interactive segmentation models may show high scores in the standard benchmarks, but it does not imply that they would perform well in a real world scenario. To assess the applicability of interactive segmentation methods, we propose a novel evaluation strategy providing a more comprehensive analysis of a model's performance. To this end, we propose a methodology for finding extreme user inputs by a direct optimization in a white-box adversarial attack on the interactive segmentation model. Based on the performance with such adversarial user inputs, we assess the robustness of interactive segmentation models w.r.t click positions. Besides, we introduce a novel benchmark for measuring the robustness of interactive segmentation, and report the results of an extensive evaluation of dozens of models.

----

## [477] T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models

**Authors**: *Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28226](https://doi.org/10.1609/aaai.v38i5.28226)

**Abstract**:

The incredible generative ability of large-scale text-to-image (T2I) models has demonstrated strong power of learning complex structures and meaningful semantics. However, relying solely on text prompts cannot fully take advantage of the knowledge learned by the model, especially when flexible and accurate controlling (e.g., structure and color) is needed. In this paper, we aim to ``dig out" the capabilities that T2I models have implicitly learned, and then explicitly use them to control the generation more granularly. Specifically, we propose to learn low-cost T2I-Adapters to align internal knowledge in T2I models with external control signals, while freezing the original large T2I models. In this way, we can train various adapters according to different conditions, achieving rich control and editing effects in the color and structure of the generation results. Further, the proposed T2I-Adapters have attractive properties of practical value, such as composability and generalization ability. Extensive experiments demonstrate that our T2I-Adapter has promising generation quality and a wide range of applications. Our code is available at https://github.com/TencentARC/T2I-Adapter.

----

## [478] Semi-supervised Open-World Object Detection

**Authors**: *Sahal Shaji Mullappilly, Abhishek Singh Gehlot, Rao Muhammad Anwer, Fahad Shahbaz Khan, Hisham Cholakkal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28227](https://doi.org/10.1609/aaai.v38i5.28227)

**Abstract**:

Conventional open-world object detection (OWOD) problem setting first distinguishes known and unknown classes and then later incrementally learns the unknown objects when introduced with labels in the subsequent tasks. However, the current OWOD formulation heavily relies on the external human oracle for knowledge input during the incremental learning stages. Such reliance on run-time makes this formulation less realistic in a real-world deployment. To address this, we introduce a more realistic formulation, named semi-supervised open-world detection (SS-OWOD), that reduces the annotation cost by casting the incremental learning stages of OWOD in a semi-supervised manner. We demonstrate that the performance of the state-of-the-art OWOD detector dramatically deteriorates in the proposed SS-OWOD setting. Therefore, we introduce a novel SS-OWOD detector, named SS-OWFormer, that utilizes a feature-alignment scheme to better align the object query representations between the original and augmented images to leverage the large unlabeled and few labeled data. We further introduce a pseudo-labeling scheme for unknown detection that exploits the inherent capability of decoder object queries to capture object-specific information. On the COCO dataset, our SS-OWFormer using only 50% of the labeled data achieves detection performance that is on par with the state-of-the-art (SOTA) OWOD detector using all the 100% of labeled data. Further, our SS-OWFormer achieves an absolute gain of 4.8% in unknown recall over the SOTA OWOD detector. Lastly, we demonstrate the effectiveness of our SS-OWOD problem setting and approach for remote sensing object detection, proposing carefully curated splits and baseline performance evaluations. Our experiments on 4 datasets including MS COCO, PASCAL, Objects365 and DOTA demonstrate the effectiveness of our approach. Our source code, models and splits are available here https://github.com/sahalshajim/SS-OWFormer

----

## [479] Adversarial Attacks on the Interpretation of Neuron Activation Maximization

**Authors**: *Géraldin Nanfack, Alexander Fulleringer, Jonathan Marty, Michael Eickenberg, Eugene Belilovsky*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28228](https://doi.org/10.1609/aaai.v38i5.28228)

**Abstract**:

Feature visualization is one of the most popular techniques used to interpret the internal behavior of individual units of trained deep neural networks. Based on activation maximization, they consist of finding synthetic or natural inputs that maximize neuron activations. This paper introduces an optimization framework that aims to deceive feature visualization through adversarial model manipulation. It consists of finetuning a pre-trained model with a specifically introduced loss that aims to maintain model performance, while also significantly changing feature visualization. We provide evidence of the success of this manipulation on several pre-trained models for the classification task with ImageNet.

----

## [480] ColNeRF: Collaboration for Generalizable Sparse Input Neural Radiance Field

**Authors**: *Zhangkai Ni, Peiqi Yang, Wenhan Yang, Hanli Wang, Lin Ma, Sam Kwong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28229](https://doi.org/10.1609/aaai.v38i5.28229)

**Abstract**:

Neural Radiance Fields (NeRF) have demonstrated impressive potential in synthesizing novel views from dense input, however, their effectiveness is challenged when dealing with sparse input. Existing approaches that incorporate additional depth or semantic supervision can alleviate this issue to an extent. However, the process of supervision collection is not only costly but also potentially inaccurate. In our work, we introduce a novel model: the Collaborative Neural Radiance Fields (ColNeRF) designed to work with sparse input. The collaboration in ColNeRF includes the cooperation among sparse input source images and the cooperation among the output of the NeRF. Through this, we construct a novel collaborative module that aligns information from various views and meanwhile imposes self-supervised constraints to ensure multi-view consistency in both geometry and appearance. A Collaborative Cross-View Volume Integration module (CCVI) is proposed to capture complex occlusions and implicitly infer the spatial location of objects. Moreover, we introduce self-supervision of target rays projected in multiple directions to ensure geometric and color consistency in adjacent regions. Benefiting from the collaboration at the input and output ends, ColNeRF is capable of capturing richer and more generalized scene representation, thereby facilitating higher-quality results of the novel view synthesis. Our extensive experimental results demonstrate that ColNeRF outperforms state-of-the-art sparse input generalizable NeRF methods. Furthermore, our approach exhibits superiority in fine-tuning towards adapting to new scenes, achieving competitive performance compared to per-scene optimized NeRF-based methods while significantly reducing computational costs. Our code is available at: https://github.com/eezkni/ColNeRF.

----

## [481] Wavelet-Driven Spatiotemporal Predictive Learning: Bridging Frequency and Time Variations

**Authors**: *Xuesong Nie, Yunfeng Yan, Siyuan Li, Cheng Tan, Xi Chen, Haoyuan Jin, Zhihang Zhu, Stan Z. Li, Donglian Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28230](https://doi.org/10.1609/aaai.v38i5.28230)

**Abstract**:

Spatiotemporal predictive learning is a paradigm that empowers models to learn spatial and temporal patterns by predicting future frames from past frames in an unsupervised manner. This method typically uses recurrent units to capture long-term dependencies, but these units often come with high computational costs and limited performance in real-world scenes. This paper presents an innovative Wavelet-based SpatioTemporal (WaST) framework, which extracts and adaptively controls both low and high-frequency components at image and feature levels via 3D discrete wavelet transform for faster processing while maintaining high-quality predictions. We propose a Time-Frequency Aware Translator uniquely crafted to efficiently learn short- and long-range spatiotemporal information by individually modeling spatial frequency and temporal variations. Meanwhile, we design a wavelet-domain High-Frequency Focal Loss that effectively supervises high-frequency variations. Extensive experiments across various real-world scenarios, such as driving scene prediction, traffic flow prediction, human motion capture, and weather forecasting, demonstrate that our proposed WaST achieves state-of-the-art performance over various spatiotemporal prediction methods.

----

## [482] Painterly Image Harmonization by Learning from Painterly Objects

**Authors**: *Li Niu, Junyan Cao, Yan Hong, Liqing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28231](https://doi.org/10.1609/aaai.v38i5.28231)

**Abstract**:

Given a composite image with photographic object and painterly background, painterly image harmonization targets at stylizing the composite object to be compatible with the background. Despite the competitive performance of existing painterly harmonization works, they did not fully leverage the painterly objects in artistic paintings. In this work, we explore learning from painterly objects for painterly image harmonization. In particular, we learn a mapping from background style and object information to object style based on painterly objects in artistic paintings. With the learnt mapping, we can hallucinate the target style of composite object, which is used to harmonize encoder feature maps to produce the harmonized image. Extensive experiments on the benchmark dataset demonstrate the effectiveness of our proposed method.

----

## [483] Progressive Painterly Image Harmonization from Low-Level Styles to High-Level Styles

**Authors**: *Li Niu, Yan Hong, Junyan Cao, Liqing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28232](https://doi.org/10.1609/aaai.v38i5.28232)

**Abstract**:

Painterly image harmonization aims to harmonize a photographic foreground object on the painterly background. Different from previous auto-encoder based harmonization networks, we develop a progressive multi-stage harmonization network, which harmonizes the composite foreground from low-level styles (e.g., color, simple texture) to high-level styles (e.g., complex texture). Our network has better interpretability and harmonization performance. Moreover, we design an early-exit strategy to automatically decide the proper stage to exit, which can skip the unnecessary and even harmful late stages. Extensive experiments on the benchmark dataset demonstrate the effectiveness of our progressive harmonization network.

----

## [484] Domain Generalizable Person Search Using Unreal Dataset

**Authors**: *Minyoung Oh, Duhyun Kim, Jae-Young Sim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28233](https://doi.org/10.1609/aaai.v38i5.28233)

**Abstract**:

Collecting and labeling real datasets to train the person search networks not only requires a lot of time and effort, but also accompanies privacy issues. 
The weakly-supervised and unsupervised domain adaptation methods have been proposed to alleviate the labeling burden for target datasets, however, their generalization capability is limited. 
We introduce a novel person search method based on the domain generalization framework, that uses an automatically labeled unreal dataset only for training but is applicable to arbitrary unseen real datasets. 
To alleviate the domain gaps when transferring the knowledge from the unreal source dataset to the real target datasets, we estimate the fidelity of person instances which is then used to train the end-to-end network adaptively. 
Moreover, we devise a domain-invariant feature learning scheme to encourage the network to suppress the domain-related features.
Experimental results demonstrate that the proposed method provides the competitive performance to existing person search methods even though it is applicable to arbitrary unseen datasets without any prior knowledge and re-training burdens.

----

## [485] OctOcc: High-Resolution 3D Occupancy Prediction with Octree

**Authors**: *Wenzhe Ouyang, Xiaolin Song, Bailan Feng, Zenglin Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28234](https://doi.org/10.1609/aaai.v38i5.28234)

**Abstract**:

3D semantic occupancy has garnered considerable attention due to its abundant structural information encompassing the entire scene in autonomous driving.
However, existing 3D occupancy prediction methods contend with the constraint of low-resolution 3D voxel features arising from the limitation of computational memory.
To address this limitation and achieve a more fine-grained representation of 3D scenes, we propose OctOcc, a novel octree-based approach for 3D semantic occupancy prediction. 
OctOcc is conceptually rooted in the observation that the vast majority of 3D space is left unoccupied. 
Capitalizing on this insight, we endeavor to cultivate memory-efficient high-resolution 3D occupancy predictions by mitigating superfluous cross-attentions. 
Specifically, we devise a hierarchical octree structure that selectively generates finer-grained cross-attentions solely in potentially occupied regions.
Extending our inquiry beyond 3D space, we identify analogous redundancies within another side of cross attentions, 2D images.
Consequently, a 2D image feature filtering network is conceived to expunge extraneous regions.
Experimental results demonstrate that the proposed OctOcc significantly outperforms existing methods on nuScenes and SemanticKITTI datasets with limited memory consumption.

----

## [486] NeSyFOLD: A Framework for Interpretable Image Classification

**Authors**: *Parth Padalkar, Huaduo Wang, Gopal Gupta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28235](https://doi.org/10.1609/aaai.v38i5.28235)

**Abstract**:

Deep learning models such as CNNs have surpassed human
performance in computer vision tasks such as image classi-
fication. However, despite their sophistication, these models
lack interpretability which can lead to biased outcomes re-
flecting existing prejudices in the data. We aim to make pre-
dictions made by a CNN interpretable. Hence, we present a
novel framework called NeSyFOLD to create a neurosym-
bolic (NeSy) model for image classification tasks. The model
is a CNN with all layers following the last convolutional layer
replaced by a stratified answer set program (ASP) derived
from the last layer kernels. The answer set program can be
viewed as a rule-set, wherein the truth value of each pred-
icate depends on the activation of the corresponding kernel
in the CNN. The rule-set serves as a global explanation for
the model and is interpretable. We also use our NeSyFOLD
framework with a CNN that is trained using a sparse kernel
learning technique called Elite BackProp (EBP). This leads to
a significant reduction in rule-set size without compromising
accuracy or fidelity thus improving scalability of the NeSy
model and interpretability of its rule-set. Evaluation is done
on datasets with varied complexity and sizes. We also pro-
pose a novel algorithm for labelling the predicates in the rule-
set with meaningful semantic concept(s) learnt by the CNN.
We evaluate the performance of our “semantic labelling algo-
rithm” to quantify the efficacy of the semantic labelling for
both the NeSy model and the NeSy-EBP model.

----

## [487] Semi-Supervised Blind Image Quality Assessment through Knowledge Distillation and Incremental Learning

**Authors**: *Wensheng Pan, Timin Gao, Yan Zhang, Xiawu Zheng, Yunhang Shen, Ke Li, Runze Hu, Yutao Liu, Pingyang Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28236](https://doi.org/10.1609/aaai.v38i5.28236)

**Abstract**:

Blind Image Quality Assessment (BIQA) aims to simulate human assessment of image quality. It has a great demand for labeled data, which is often insufficient in practice. Some researchers employ unsupervised methods to address this issue, which is challenging to emulate the human subjective system. To this end, we introduce a unified framework that combines semi-supervised and incremental learning to address the mentioned issue. Specifically, when training data is limited, semi-supervised learning is necessary to infer extensive unlabeled data. To facilitate semi-supervised learning, we use knowledge distillation to assign pseudo-labels to unlabeled data, preserving analytical capability. To gradually improve the quality of pseudo labels, we introduce incremental learning. However, incremental learning can lead to catastrophic forgetting. We employ Experience Replay by selecting representative samples during multiple rounds of semi-supervised learning, to alleviate forgetting and ensure model stability. Experimental results show that the proposed approach achieves state-of-the-art performance across various benchmark datasets. After being trained on the LIVE dataset, our method can be directly transferred to the CSIQ dataset. Compared with other methods, it significantly outperforms unsupervised methods on the CSIQ dataset with a marginal performance drop (-0.002) on the LIVE dataset. In conclusion, our proposed method demonstrates its potential to tackle the challenges in real-world production processes.

----

## [488] Less Is More: Label Recommendation for Weakly Supervised Point Cloud Semantic Segmentation

**Authors**: *Zhiyi Pan, Nan Zhang, Wei Gao, Shan Liu, Ge Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28237](https://doi.org/10.1609/aaai.v38i5.28237)

**Abstract**:

Weak supervision has proven to be an effective strategy for reducing the burden of annotating semantic segmentation tasks in 3D space. However, unconstrained or heuristic weakly supervised annotation forms may lead to suboptimal label efficiency. To address this issue, we propose a novel label recommendation framework for weakly supervised point cloud semantic segmentation. Distinct from pre-training and active learning, the label recommendation framework consists of three stages: inductive bias learning, recommendations for points to be labeled, and point cloud semantic segmentation learning. In practice, we first introduce the point cloud upsampling task to induct inductive bias from structural information. During the recommendation stage, we present a cross-scene clustering strategy to generate centers of clustering as recommended points. Then we introduce a recommended point positions attention module LabelAttention to model the long-range dependency under sparse annotations. Additionally, we employ position encoding to enhance the spatial awareness of semantic features. Throughout the framework, the useful information obtained from inductive bias learning is propagated to subsequent semantic segmentation networks in the form of label positions. Experimental results demonstrate that our framework outperforms weakly supervised point cloud semantic segmentation methods and other methods for labeling efficiency on S3DIS and ScanNetV2, even at an extremely low label rate.

----

## [489] patchDPCC: A Patchwise Deep Compression Framework for Dynamic Point Clouds

**Authors**: *Zirui Pan, Mengbai Xiao, Xu Han, Dongxiao Yu, Guanghui Zhang, Yao Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28238](https://doi.org/10.1609/aaai.v38i5.28238)

**Abstract**:

When compressing point clouds, point-based deep learning models operate points in a continuous space, which has a chance to minimize the geometric fidelity loss introduced by voxelization in preprocessing. But these methods could hardly scale to inputs with arbitrary points. Furthermore, the point cloud frames are individually compressed, failing the conventional wisdom of leveraging inter-frame similarity. In this work, we propose a patchwise compression framework called patchDPCC, which consists of a patch group generation module and a point-based compression model. Algorithms are developed to generate patches from different frames representing the same object, and more importantly, these patches are regulated to have the same number of points. We also incorporate a feature transfer module in the compression model, which refines the feature quality by exploiting the inter-frame similarity. Our model generates point-wise features for entropy coding, which guarantees the reconstruction speed. The evaluation on the MPEG 8i dataset shows that our method improves the compression ratio by 47.01% and 85.22% when compared to PCGCv2 and V-PCC with the same reconstruction quality, which is 9% and 16% better than that D-DPCC does. Our method also achieves the fastest decoding speed among the learning-based compression models.

----

## [490] LISR: Learning Linear 3D Implicit Surface Representation Using Compactly Supported Radial Basis Functions

**Authors**: *Atharva Pandey, Vishal Yadav, Rajendra Nagar, Santanu Chaudhury*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28239](https://doi.org/10.1609/aaai.v38i5.28239)

**Abstract**:

Implicit 3D surface reconstruction of an object from its partial and noisy 3D point cloud scan is the classical geometry processing and 3D computer vision problem. In the literature, various 3D shape representations have been developed, differing in memory efficiency and shape retrieval effectiveness, such as volumetric, parametric, and implicit surfaces. Radial basis functions provide memory-efficient parameterization of the implicit surface. However, we show that training a neural network using the mean squared error between the ground-truth implicit surface and the linear basis-based implicit surfaces does not converge to the global solution. In this work, we propose locally supported compact radial basis functions for a linear representation of the implicit surface. This representation enables us to generate  3D shapes with arbitrary topologies at any resolution due to their continuous nature. We then propose a neural network architecture for learning the linear implicit shape representation of the 3D surface of an object. We learn linear implicit shapes within a supervised learning framework using ground truth Signed-Distance Field (SDF) data for guidance. The classical strategies face difficulties in finding linear implicit shapes from a given 3D point cloud due to numerical issues (requires solving inverse of a large matrix) in basis and query point selection. The proposed approach achieves better Chamfer distance and comparable F-score than the state-of-the-art approach on the benchmark dataset. We also show the effectiveness of the proposed approach by using it for the 3D shape completion task.

----

## [491] RadarMOSEVE: A Spatial-Temporal Transformer Network for Radar-Only Moving Object Segmentation and Ego-Velocity Estimation

**Authors**: *Changsong Pang, Xieyuanli Chen, Yimin Liu, Huimin Lu, Yuwei Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28240](https://doi.org/10.1609/aaai.v38i5.28240)

**Abstract**:

Moving object segmentation (MOS) and Ego velocity estimation (EVE) are vital capabilities for mobile systems to achieve full autonomy. Several approaches have attempted to achieve MOSEVE using a LiDAR sensor. However, LiDAR sensors are typically expensive and susceptible to adverse weather conditions. Instead, millimeter-wave radar (MWR) has gained popularity in robotics and autonomous driving for real applications due to its cost-effectiveness and resilience to bad weather. Nonetheless, publicly available MOSEVE datasets and approaches using radar data are limited. Some existing methods adopt point convolutional networks from LiDAR-based approaches, ignoring the specific artifacts and the valuable radial velocity information of radar measurements, leading to suboptimal performance. In this paper, we propose a novel transformer network that effectively addresses the sparsity and noise issues and leverages the radial velocity measurements of radar points using our devised radar self- and cross-attention mechanisms. Based on that, our method achieves accurate EVE of the robot and performs MOS using only radar data simultaneously. To thoroughly evaluate the MOSEVE performance of our method, we annotated the radar points in the public View-of-Delft (VoD) dataset and additionally constructed a new radar dataset in various environments. The experimental results demonstrate the superiority of our approach over existing state-of-the-art methods. The code is available at https://github.com/ORCAUboat/RadarMOSEVE.

----

## [492] NeBLa: Neural Beer-Lambert for 3D Reconstruction of Oral Structures from Panoramic Radiographs

**Authors**: *Sihwa Park, Seongjun Kim, Doeyoung Kwon, Yohan Jang, In-Seok Song, Seung Jun Baek*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28241](https://doi.org/10.1609/aaai.v38i5.28241)

**Abstract**:

Panoramic radiography (Panoramic X-ray, PX) is a widely used imaging modality for dental examination. However, PX only provides a flattened 2D image, lacking in a 3D view of the oral structure. In this paper, we propose NeBLa (Neural Beer-Lambert) to estimate 3D oral structures from real-world PX. NeBLa tackles full 3D reconstruction for varying subjects (patients) where each reconstruction is based only on a single panoramic image. We create an intermediate representation called simulated PX (SimPX) from 3D Cone-beam computed tomography (CBCT) data based on the Beer-Lambert law of X-ray rendering and rotational principles of PX imaging. SimPX aims at not only truthfully simulating PX, but also facilitates the reverting process back to 3D data. We propose a novel neural model based on ray tracing which exploits both global and local input features to convert SimPX to 3D output. At inference, a real PX image is translated to a SimPX-style image with semantic regularization, and the translated image is processed by generation module to produce high-quality outputs. Experiments show that NeBLa outperforms prior state-of-the-art in reconstruction tasks both quantitatively and qualitatively. Unlike prior methods, NeBLa does not require any prior information such as the shape of dental arches, nor the matched PX-CBCT dataset for training, which is difficult to obtain in clinical practice. Our code is available at https://github.com/sihwa-park/nebla.

----

## [493] Task-Disruptive Background Suppression for Few-Shot Segmentation

**Authors**: *Suho Park, Su Been Lee, Sangeek Hyun, Hyun Seok Seong, Jae-Pil Heo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28242](https://doi.org/10.1609/aaai.v38i5.28242)

**Abstract**:

Few-shot segmentation aims to accurately segment novel target objects within query images using only a limited number of annotated support images. The recent works exploit support background as well as its foreground to precisely compute the dense correlations between query and support. However, they overlook the characteristics of the background that generally contains various types of objects. In this paper, we highlight this characteristic of background which can bring problematic cases as follows: (1) when the query and support backgrounds are dissimilar and (2) when objects in the support background are similar to the target object in the query. Without any consideration of the above cases, adopting the entire support background leads to a misprediction of the query foreground as background. To address this issue, we propose Task-disruptive Background Suppression(TBS), a module to suppress those disruptive support background features based on two spatial-wise scores: query-relevant and target-relevant scores. The former aims to mitigate the impact of unshared features solely existing in the support background, while the latter aims to reduce the influence of target-similar support background features. Based on these two scores, we define a query background relevant score that captures the similarity between the backgrounds of the query and the support, and utilize it to scale support background features to adaptively restrict the impact of disruptive support backgrounds. Our proposed method achieves state-of-the-art performance on standard few-shot segmentation benchmarks. Our official code is available at github.com/SuhoPark0706/TBSNet.

----

## [494] SA²VP: Spatially Aligned-and-Adapted Visual Prompt

**Authors**: *Wenjie Pei, Tongqi Xia, Fanglin Chen, Jinsong Li, Jiandong Tian, Guangming Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28243](https://doi.org/10.1609/aaai.v38i5.28243)

**Abstract**:

As a prominent parameter-efficient fine-tuning technique in NLP, prompt tuning is being explored its potential in computer vision. Typical methods for visual prompt tuning follow the sequential modeling paradigm stemming from NLP, which represents an input image as a flattened sequence of token embeddings and then learns a set of unordered parameterized tokens prefixed to the sequence representation as the visual prompts for task adaptation of large vision models. While such sequential modeling paradigm of visual prompt has shown great promise, there are two potential limitations. First, the learned visual prompts cannot model the underlying spatial relations in the input image, which is crucial for image encoding. Second, since all prompt tokens play the same role of prompting for all image tokens without distinction, it lacks the fine-grained prompting capability, i.e., individual prompting for different image tokens. In this work, we propose the Spatially Aligned-and-Adapted Visual Prompt model (SA^2VP), which learns a two-dimensional prompt token map with equal (or scaled) size to the image token map, thereby being able to spatially align with the image map. Each prompt token is designated to prompt knowledge only for the spatially corresponding image tokens. As a result, our model can conduct individual prompting for different image tokens in a fine-grained manner. Moreover, benefiting from the capability of preserving the spatial structure by the learned prompt token map, our SA^2VP is able to model the spatial relations in the input image, leading to more effective prompting. Extensive experiments on three challenging benchmarks for image classification demonstrate the superiority of our model over other state-of-the-art methods for visual prompt tuning. Code is available at https://github.com/tommy-xq/SA2VP.

----

## [495] ConditionVideo: Training-Free Condition-Guided Video Generation

**Authors**: *Bo Peng, Xinyuan Chen, Yaohui Wang, Chaochao Lu, Yu Qiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28244](https://doi.org/10.1609/aaai.v38i5.28244)

**Abstract**:

Recent works have successfully extended large-scale text-to-image models to the video domain, producing promising results but at a high computational cost and requiring a large amount of video data. In this work, we introduce ConditionVideo, a training-free approach to text-to-video generation based on the provided condition, video, and input text, by leveraging the power of off-the-shelf text-to-image generation methods (e.g., Stable Diffusion). ConditionVideo generates realistic dynamic videos from random noise or given scene videos. Our method explicitly disentangles the motion representation into condition-guided and scenery motion components. To this end, the ConditionVideo model is designed with a UNet branch and a control branch. To improve temporal coherence, we introduce sparse bi-directional spatial-temporal attention (sBiST-Attn). The 3D control network extends the conventional 2D controlnet model, aiming to strengthen conditional generation accuracy by additionally leveraging the bi-directional frames in the temporal domain. Our method exhibits superior performance in terms of frame consistency, clip score, and conditional accuracy, outperforming other compared methods.

----

## [496] ViTEraser: Harnessing the Power of Vision Transformers for Scene Text Removal with SegMIM Pretraining

**Authors**: *Dezhi Peng, Chongyu Liu, Yuliang Liu, Lianwen Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28245](https://doi.org/10.1609/aaai.v38i5.28245)

**Abstract**:

Scene text removal (STR) aims at replacing text strokes in natural scenes with visually coherent backgrounds. Recent STR approaches rely on iterative refinements or explicit text masks, resulting in high complexity and sensitivity to the accuracy of text localization. Moreover, most existing STR methods adopt convolutional architectures while the potential of vision Transformers (ViTs) remains largely unexplored. In this paper, we propose a simple-yet-effective ViT-based text eraser, dubbed ViTEraser. Following a concise encoder-decoder framework, ViTEraser can easily incorporate various ViTs to enhance long-range modeling. Specifically, the encoder hierarchically maps the input image into the hidden space through ViT blocks and patch embedding layers, while the decoder gradually upsamples the hidden features to the text-erased image with ViT blocks and patch splitting layers. As ViTEraser implicitly integrates text localization and inpainting, we propose a novel end-to-end pretraining method, termed SegMIM, which focuses the encoder and decoder on the text box segmentation and masked image modeling tasks, respectively. Experimental results demonstrate that ViTEraser with SegMIM achieves state-of-the-art performance on STR by a substantial margin and exhibits strong generalization ability when extended to other tasks, e.g., tampered scene text detection. Furthermore, we comprehensively explore the architecture, pretraining, and scalability of the ViT-based encoder-decoder for STR, which provides deep insights into the application of ViT to the STR field. Code is available at https://github.com/shannanyinxiang/ViTEraser.

----

## [497] FRIH: Fine-Grained Region-Aware Image Harmonization

**Authors**: *Jinlong Peng, Zekun Luo, Liang Liu, Boshen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28246](https://doi.org/10.1609/aaai.v38i5.28246)

**Abstract**:

Image harmonization aims to generate a more realistic appearance of foreground and background for a composite image. All the existing methods perform the same harmonization process for the whole foreground. However, the implanted foreground always contains different appearance patterns. Existing solutions ignore the difference of each color block and lose some specific details. Therefore, we propose a novel global-local two stages framework for Fine-grained Region-aware Image Harmonization (FRIH). In the first stage, the whole input foreground mask is used to make a global coarse-grained harmonization. In the second stage, we adaptively cluster the input foreground mask into several submasks. Each submask and the coarsely adjusted image are concatenated respectively and fed into a lightweight cascaded module, refining the global harmonization result. Moreover, we further design a fusion prediction module to generate the final result, utilizing the different degrees of harmonization results comprehensively. Without bells and whistles, our FRIH achieves a competitive performance on iHarmony4 dataset with a lightweight model.

----

## [498] Navigating Open Set Scenarios for Skeleton-Based Action Recognition

**Authors**: *Kunyu Peng, Cheng Yin, Junwei Zheng, Ruiping Liu, David Schneider, Jiaming Zhang, Kailun Yang, M. Saquib Sarfraz, Rainer Stiefelhagen, Alina Roitberg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28247](https://doi.org/10.1609/aaai.v38i5.28247)

**Abstract**:

In real-world scenarios, human actions often fall outside the distribution of training data, making it crucial for models to recognize known actions and reject unknown ones. However, using pure skeleton data in such open-set conditions poses challenges due to the lack of visual background cues and the distinct sparse structure of body pose sequences. In this paper, we tackle the unexplored Open-Set Skeleton-based Action Recognition (OS-SAR) task and formalize the benchmark on three skeleton-based datasets. We assess the performance of seven established open-set approaches on our task and identify their limits and critical generalization issues when dealing with skeleton information.To address these challenges, we propose a distance-based cross-modality ensemble method that leverages the cross-modal alignment of skeleton joints, bones, and velocities to achieve superior open-set recognition performance. We refer to the key idea as CrossMax - an approach that utilizes a novel cross-modality mean max discrepancy suppression mechanism to align latent spaces during training and a cross-modality distance-based logits refinement method during testing. CrossMax outperforms existing approaches and consistently yields state-of-the-art results across all datasets and backbones. We will release the benchmark, code, and models to the community.

----

## [499] LaneGraph2Seq: Lane Topology Extraction with Language Model via Vertex-Edge Encoding and Connectivity Enhancement

**Authors**: *Renyuan Peng, Xinyue Cai, Hang Xu, Jiachen Lu, Feng Wen, Wei Zhang, Li Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28248](https://doi.org/10.1609/aaai.v38i5.28248)

**Abstract**:

Understanding road structures is crucial for autonomous driving. Intricate road structures are often depicted using lane graphs, which include centerline curves and connections forming a Directed Acyclic Graph (DAG). Accurate extraction of lane graphs relies on precisely estimating vertex and edge information within the DAG.
Recent research highlights Transformer-based language models' impressive sequence prediction abilities, making them effective for learning graph representations when graph data are encoded as sequences. However, existing studies focus mainly on modeling vertices explicitly, leaving edge information simply embedded in the network. 
Consequently, these approaches fall short in the task of lane graph extraction. To address this, we introduce LaneGraph2Seq, a novel approach for lane graph extraction. It leverages a language model with vertex-edge encoding and connectivity enhancement. Our serialization strategy includes a vertex-centric depth-first traversal and a concise edge-based partition sequence. Additionally, we use classifier-free guidance combined with nucleus sampling to improve lane connectivity. We validate our method on prominent datasets, nuScenes and Argoverse 2, showcasing consistent and compelling results. Our LaneGraph2Seq approach demonstrates superior performance compared to state-of-the-art techniques in lane graph extraction.

----

## [500] Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification

**Authors**: *Wenshuo Peng, Kaipeng Zhang, Yue Yang, Hao Zhang, Yu Qiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28249](https://doi.org/10.1609/aaai.v38i5.28249)

**Abstract**:

Vision-language foundation models have been incredibly successful in a wide range of downstream computer vision tasks using adaptation methods. However, due to the high cost of obtaining pre-training datasets, pairs with weak image-text correlation in the data exist in large numbers. We call them weak-paired samples. Due to the limitations of these weak-paired samples, the pre-training model are unable to mine all the knowledge from pre-training data. The existing adaptation methods do not consider the missing knowledge, which may lead to crucial task-related knowledge for the downstream tasks being ignored. To address this issue, we propose a new adaptation framework called Data Adaptive Traceback (DAT). Specifically, we utilize a zero-shot-based method to extract the most downstream task-related subset of the pre-training data to enable the downstream tasks. Furthermore, we adopt a pseudo-label-based semi-supervised technique to reuse the pre-training images and a vision-language contrastive learning method to address the confirmation bias issue in semi-supervised learning. We conduct extensive experiments that show our proposed DAT approach meaningfully improves various benchmark datasets’ performance over traditional adaptation methods by simply.

----

## [501] SAM-PARSER: Fine-Tuning SAM Efficiently by Parameter Space Reconstruction

**Authors**: *Zelin Peng, Zhengqin Xu, Zhilin Zeng, Xiaokang Yang, Wei Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28250](https://doi.org/10.1609/aaai.v38i5.28250)

**Abstract**:

Segment Anything Model (SAM) has received remarkable attention as it offers a powerful and versatile solution for object segmentation in images. However, fine-tuning SAM for downstream segmentation tasks under different scenarios remains a challenge, as the varied characteristics of different scenarios naturally requires diverse model parameter spaces. Most existing fine-tuning methods attempt to bridge the gaps among different scenarios by introducing a set of new parameters to modify SAM's original parameter space. Unlike these works, in this paper, we propose fine-tuning SAM efficiently by parameter space reconstruction (SAM-PARSER), which introduce nearly zero trainable parameters during fine-tuning. In SAM-PARSER, we assume that SAM's original parameter space is relatively complete, so that its bases are able to reconstruct the parameter space of a new scenario. We obtain the bases by matrix decomposition, and fine-tuning the coefficients to reconstruct the parameter space tailored to the new scenario by an optimal linear combination of the bases. Experimental results show that SAM-PARSER exhibits superior segmentation performance across various scenarios, while reducing the number of trainable parameters by approximately 290 times compared with current parameter-efficient fine-tuning methods.

----

## [502] Relational Distant Supervision for Image Captioning without Image-Text Pairs

**Authors**: *Yayun Qi, Wentian Zhao, Xinxiao Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28251](https://doi.org/10.1609/aaai.v38i5.28251)

**Abstract**:

Unsupervised image captioning aims to generate descriptions of images without relying on any image-sentence pairs for training. Most existing works use detected visual objects or concepts as bridge to connect images and texts. Considering that the relationship between objects carries more information, we use the object relationship as a more accurate connection between images and texts. In this paper, we adapt the idea of distant supervision that extracts the knowledge about object relationships from an external corpus and imparts them to images to facilitate inferring visual object relationships, without introducing any extra pre-trained relationship detectors. Based on these learned informative relationships, we construct pseudo image-sentence pairs for captioning model training. Specifically, our method consists of three modules: (1) a relationship learning module that learns to infer relationships from images under the distant supervision; (2) a relationship-to-sentence module that transforms the inferred relationships into sentences to generate pseudo image-sentence pairs; (3) an image captioning module that is trained by using the generated image-sentence pairs. Promising results on three datasets show that our method outperforms the state-of-the-art methods of unsupervised image captioning.

----

## [503] Bias-Conflict Sample Synthesis and Adversarial Removal Debias Strategy for Temporal Sentence Grounding in Video

**Authors**: *Zhaobo Qi, Yibo Yuan, Xiaowen Ruan, Shuhui Wang, Weigang Zhang, Qingming Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28252](https://doi.org/10.1609/aaai.v38i5.28252)

**Abstract**:

Temporal Sentence Grounding in Video (TSGV) is troubled by dataset bias issue, which is caused by the uneven temporal distribution of the target moments for samples with similar semantic components in input videos or query texts. Existing methods resort to utilizing prior knowledge about bias to artificially break this uneven distribution, which only removes a limited amount of significant language biases. In this work, we propose the bias-conflict sample synthesis and adversarial removal debias strategy (BSSARD), which dynamically generates bias-conflict samples by explicitly leveraging potentially spurious correlations between single-modality features and the temporal position of the target moments. Through adversarial training, its bias generators continuously introduce biases and generate bias-conflict samples to deceive its grounding model. Meanwhile, the grounding model continuously eliminates the introduced biases, which requires it to model multi-modality alignment information. BSSARD will cover most kinds of coupling relationships and disrupt language and visual biases simultaneously. Extensive experiments on Charades-CD and ActivityNet-CD demonstrate the promising debiasing capability of BSSARD. Source codes are available at https://github.com/qzhb/BSSARD.

----

## [504] NuScenes-QA: A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario

**Authors**: *Tianwen Qian, Jingjing Chen, Linhai Zhuo, Yang Jiao, Yu-Gang Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28253](https://doi.org/10.1609/aaai.v38i5.28253)

**Abstract**:

We introduce a novel visual question answering (VQA) task in the context of autonomous driving, aiming to answer natural language questions based on street-view clues. Compared to traditional VQA tasks, VQA in autonomous driving scenario presents more challenges. Firstly, the raw visual data are multi-modal, including images and point clouds captured by camera and LiDAR, respectively. Secondly, the data are multi-frame due to the continuous, real-time acquisition. Thirdly, the outdoor scenes exhibit both moving foreground and static background. Existing VQA benchmarks fail to adequately address these complexities. To bridge this gap, we propose NuScenes-QA, the first benchmark for VQA in the autonomous driving scenario, encompassing 34K visual scenes and 460K question-answer pairs. Specifically, we leverage existing 3D detection annotations to generate scene graphs and design question templates manually. Subsequently, the question-answer pairs are generated programmatically based on these templates. Comprehensive statistics prove that our NuScenes-QA is a balanced large-scale benchmark with diverse question formats. Built upon it, we develop a series of baselines that employ advanced 3D detection and VQA techniques. Our extensive experiments highlight the challenges posed by this new task. Codes and dataset are available at https://github.com/qiantianwen/NuScenes-QA.

----

## [505] X-RefSeg3D: Enhancing Referring 3D Instance Segmentation via Structured Cross-Modal Graph Neural Networks

**Authors**: *Zhipeng Qian, Yiwei Ma, Jiayi Ji, Xiaoshuai Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28254](https://doi.org/10.1609/aaai.v38i5.28254)

**Abstract**:

Referring 3D instance segmentation is a challenging task aimed at accurately segmenting a target instance within a 3D scene based on a given referring expression. However, previous methods have overlooked the distinct roles played by different words in referring expressions. Additionally, they have failed to incorporate the positional relationship within referring expressions with the spatial correlations in 3D scenes. To alleviate these issues, we present a novel model called X-RefSeg3D, which constructs a cross-modal graph for the input 3D scene and unites textual and spatial relationships for reasoning via graph neural networks. Our approach begins by capturing object-specific text features, which are then fused with the instance features to construct a comprehensive cross-modal scene graph. Subsequently, we integrate the obtained cross-modal features into graph neural networks, leveraging the K-nearest algorithm to derive explicit instructions from expressions and factual relationships in scenes. This enables the effective capture of higher-order relationships among instances, thereby enhancing feature fusion and facilitating reasoning. Finally, the refined feature undergoes a matching module to compute the ultimate matching score. Experimental results on ScanRefer demonstrate the effectiveness of our method, surpassing previous approaches by a substantial margin of +3.67% in terms of mIOU.

----

## [506] BARET: Balanced Attention Based Real Image Editing Driven by Target-Text Inversion

**Authors**: *Yuming Qiao, Fanyi Wang, Jingwen Su, Yanhao Zhang, Yunjie Yu, Siyu Wu, Guo-Jun Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28255](https://doi.org/10.1609/aaai.v38i5.28255)

**Abstract**:

Image editing approaches with diffusion models have been rapidly developed, yet their applicability are subject to requirements such as specific editing types (e.g., foreground or background object editing, style transfer), multiple conditions (e.g., mask, sketch, caption), and time consuming fine-tuning of diffusion models. For alleviating these limitations and realizing efficient real image editing, we propose a novel editing technique that only requires an input image and target text for various editing types including non-rigid edits without fine-tuning diffusion model. Our method contains three novelties: (I) Target-text Inversion Schedule (TTIS) is designed to fine-tune the input target text embedding to achieve fast image reconstruction without image caption and acceleration of convergence. (II) Progressive Transition Scheme applies progressive linear interpolation between target text embedding and its fine-tuned version to generate transition embedding for maintaining non-rigid editing capability. (III) Balanced Attention Module (BAM) balances the tradeoff between textual description and image semantics. By the means of combining self-attention map from reconstruction process and cross-attention map from transition process, the guidance of target text embeddings in diffusion process is optimized. In order to demonstrate editing capability, effectiveness and efficiency of the proposed BARET, we have conducted extensive qualitative and quantitative experiments. Moreover, results derived from user study and ablation study further prove the superiority over other methods.

----

## [507] High-Fidelity 3D Head Avatars Reconstruction through Spatially-Varying Expression Conditioned Neural Radiance Field

**Authors**: *Minghan Qin, Yifan Liu, Yuelang Xu, Xiaochen Zhao, Yebin Liu, Haoqian Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28256](https://doi.org/10.1609/aaai.v38i5.28256)

**Abstract**:

One crucial aspect of 3D head avatar reconstruction lies in the details of facial expressions. Although recent NeRF-based photo-realistic 3D head avatar methods achieve high-quality avatar rendering, they still encounter challenges retaining intricate facial expression details because they overlook the potential of specific expression variations at different spatial positions when conditioning the radiance field. Motivated by this observation, we introduce a novel Spatially-Varying Expression (SVE) conditioning. The SVE can be obtained by a simple MLP-based generation network, encompassing both spatial positional features and global expression information. Benefiting from rich and diverse information of the SVE at different positions, the proposed SVE-conditioned NeRF can deal with intricate facial expressions and achieve realistic rendering and geometry details of high-fidelity 3D head avatars. Additionally, to further elevate the geometric and rendering quality, we introduce a new coarse-to-fine training strategy, including a geometry initialization strategy at the coarse stage and an adaptive importance sampling strategy at the fine stage. Extensive experiments indicate that our method outperforms other state-of-the-art (SOTA) methods in rendering and geometry quality on mobile phone-collected and public datasets.  Code and data can be found at https://github.com/minghanqin/AvatarSVE.

----

## [508] Text2City: One-Stage Text-Driven Urban Layout Regeneration

**Authors**: *Yiming Qin, Nanxuan Zhao, Bin Sheng, Rynson W. H. Lau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28257](https://doi.org/10.1609/aaai.v38i5.28257)

**Abstract**:

Regenerating urban layout is an essential process for urban regeneration. In this paper, we propose a new task called text-driven urban layout regeneration, which provides an intuitive input modal - text - for users to specify the regeneration, instead of designing complex rules. Given the target region to be regenerated, we propose a one-stage text-driven urban layout regeneration model, Text2City, to jointly and progressively regenerate the urban layout (i.e., road and building layouts) based on textual layout descriptions and surrounding context (i.e., urban layouts and functions of the surrounding regions). Text2City first extracts road and building attributes from the textual layout description to guide the regeneration. It includes a novel one-stage joint regenerator network based on the conditioned denoising diffusion probabilistic models (DDPMs) and prior knowledge exchange. To harmonize the regenerated layouts through joint optimization, we propose the interactive & enhanced guidance module for self-enhancement and prior knowledge exchange between road and building layouts during the regeneration. We also design a series of constraints from attribute-, geometry- and pixel-levels to ensure rational urban layout generation. To train our model, we build a large-scale dataset containing urban layouts and layout descriptions, covering 147K regions. Qualitative and quantitative evaluations show that our proposed method outperforms the baseline methods in regenerating desirable urban layouts that meet the textual descriptions.

----

## [509] Empowering CAM-Based Methods with Capability to Generate Fine-Grained and High-Faithfulness Explanations

**Authors**: *Changqing Qiu, Fusheng Jin, Yining Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28258](https://doi.org/10.1609/aaai.v38i5.28258)

**Abstract**:

Recently, the explanation of neural network models has garnered considerable research attention. In computer vision, CAM (Class Activation Map)-based methods and LRP (Layer-wise Relevance Propagation) method are two common explanation methods. However, since most CAM-based methods can only generate global weights, they can only generate coarse-grained explanations at a deep layer. LRP and its variants, on the other hand, can generate fine-grained explanations. But the faithfulness of the explanations is too low. To address these challenges, in this paper, we propose FG-CAM (Fine-Grained CAM), which extends CAM-based methods to enable generating fine-grained and high-faithfulness explanations. FG-CAM uses the relationship between two adjacent layers of feature maps with resolution differences to gradually increase the explanation resolution, while finding the contributing pixels and filtering out the pixels that do not contribute. Our method not only solves the shortcoming of CAM-based methods without changing their characteristics, but also generates fine-grained explanations that have higher faithfulness than LRP and its variants. We also present FG-CAM with denoising, which is a variant of FG-CAM and is able to generate less noisy explanations with almost no change in explanation faithfulness. Experimental results show that the performance of FG-CAM is almost unaffected by the explanation resolution. FG-CAM outperforms existing CAM-based methods significantly in both shallow and intermediate layers, and outperforms LRP and its variants significantly in the input layer. Our code is available at https://github.com/dongmo-qcq/FG-CAM.

----

## [510] High-Order Structure Based Middle-Feature Learning for Visible-Infrared Person Re-identification

**Authors**: *Liuxiang Qiu, Si Chen, Yan Yan, Jing-Hao Xue, Da-Han Wang, Shunzhi Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28259](https://doi.org/10.1609/aaai.v38i5.28259)

**Abstract**:

Visible-infrared person re-identification (VI-ReID) aims to retrieve images of the same persons captured by visible (VIS) and infrared (IR) cameras. Existing VI-ReID methods ignore high-order structure information of features while being relatively difficult to learn a reasonable common feature space due to the large modality discrepancy between VIS and IR images. To address the above problems, we propose a novel high-order structure based middle-feature learning network (HOS-Net) for effective VI-ReID. Specifically, we first leverage a short- and long-range feature extraction (SLE) module to effectively exploit both short-range and long-range features. Then, we propose a high-order structure learning (HSL) module to successfully model the high-order relationship across different local features of each person image based on a whitened hypergraph network. This greatly alleviates model collapse and enhances feature representations. Finally, we develop a common feature space learning (CFL) module to learn a discriminative and reasonable common feature space based on middle features generated by aligning features from different modalities and ranges. In particular, a modality-range identity-center contrastive (MRIC) loss is proposed to reduce the distances between the VIS, IR, and middle features, smoothing the training process. Extensive experiments on the SYSU-MM01, RegDB, and LLCM  datasets show that our HOS-Net achieves superior state-of-the-art performance. Our code is available at https://github.com/Jaulaucoeng/HOS-Net.

----

## [511] Mining Fine-Grained Image-Text Alignment for Zero-Shot Captioning via Text-Only Training

**Authors**: *Longtian Qiu, Shan Ning, Xuming He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28260](https://doi.org/10.1609/aaai.v38i5.28260)

**Abstract**:

Image captioning aims at generating descriptive and meaningful textual descriptions of images, enabling a broad range of vision-language applications. Prior works have demonstrated that harnessing the power of Contrastive Image Language Pre-training (CLIP) offers a promising approach to achieving zero-shot captioning, eliminating the need for expensive caption annotations. However, the widely observed modality gap in the latent space of CLIP harms the performance of zero-shot captioning by breaking the alignment between paired image-text features. To address this issue, we conduct an analysis on the CLIP latent space which leads to two findings. Firstly, we observe that the CLIP's visual feature of image subregions can achieve closer proximity to the paired caption due to the inherent information loss in text descriptions. In addition, we show that the modality gap between a paired image-text can be empirically modeled as a zero-mean Gaussian distribution. Motivated by the findings, we propose a novel zero-shot image captioning framework with text-only training to reduce the modality gap. In particular, we introduce a subregion feature aggregation to leverage local region information, which produces a compact visual representation for matching text representation. Moreover, we incorporate a noise injection and CLIP reranking strategy to boost captioning performance. We also extend our framework to build a zero-shot VQA pipeline, demonstrating its generality. Through extensive experiments on common captioning and VQA datasets such as MSCOCO, Flickr30k and VQAV2, we show that our method achieves remarkable performance improvements. Code is available at https://github.com/Artanic30/MacCap.

----

## [512] HiHPQ: Hierarchical Hyperbolic Product Quantization for Unsupervised Image Retrieval

**Authors**: *Zexuan Qiu, Jiahong Liu, Yankai Chen, Irwin King*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28261](https://doi.org/10.1609/aaai.v38i5.28261)

**Abstract**:

Existing unsupervised deep product quantization methods primarily aim for the increased similarity between different views of the identical image, whereas the delicate multi-level semantic similarities preserved between images are overlooked. Moreover, these methods predominantly focus on the Euclidean space for computational convenience, compromising their ability to map the multi-level semantic relationships between images effectively. To mitigate these shortcomings, we propose a novel unsupervised product quantization method dubbed Hierarchical Hyperbolic Product Quantization (HiHPQ),  which learns quantized representations by incorporating hierarchical semantic similarity within hyperbolic geometry. Specifically, we propose a hyperbolic product quantizer, where the hyperbolic codebook attention mechanism and the quantized contrastive learning on the hyperbolic product manifold are introduced to expedite quantization. Furthermore, we propose a hierarchical semantics learning module, designed to enhance the distinction between similar and non-matching images for a query by utilizing the extracted hierarchical semantics as an additional training supervision. Experiments on benchmark image datasets show that our proposed method outperforms state-of-the-art baselines.

----

## [513] S2CycleDiff: Spatial-Spectral-Bilateral Cycle-Diffusion Framework for Hyperspectral Image Super-resolution

**Authors**: *Jiahui Qu, Jie He, Wenqian Dong, Jingyu Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28262](https://doi.org/10.1609/aaai.v38i5.28262)

**Abstract**:

Hyperspectral image super-resolution (HISR) is a technique that can break through the limitation of imaging mechanism to obtain the hyperspectral image (HSI) with high spatial resolution. Although some progress has been achieved by existing methods, most of them directly learn the spatial-spectral joint mapping between the observed images and the target high-resolution HSI (HrHSI), failing to fully reserve the spectral distribution of low-resolution HSI (LrHSI) and the spatial distribution of high-resolution multispectral imagery (HrMSI). To this end, we propose a spatial-spectral-bilateral cycle-diffusion framework (S2CycleDiff) for HISR, which can step-wise generate the HrHSI with high spatial-spectral fidelity by learning the conditional distribution of spatial and spectral super-resolution processes bilaterally. Specifically, a customized conditional cycle-diffusion framework is designed as the backbone to achieve the spatial-spectral-bilateral super-resolution by repeated refinement, wherein the spatial/spectral guided pyramid denoising (SGPD) module seperately takes HrMSI and LrHSI as the guiding factors to achieve the spatial details injection and spectral correction. The outputs of the conditional cycle-diffusion framework are fed into a complementary fusion block to integrate the spatial and spectral details to generate the desired HrHSI. Experiments have been conducted on three widely used datasets to demonstrate the superiority of the proposed method over state-of-the-art HISR methods. The code is available at https://github.com/Jiahuiqu/S2CycleDiff.

----

## [514] E2HQV: High-Quality Video Generation from Event Camera via Theory-Inspired Model-Aided Deep Learning

**Authors**: *Qiang Qu, Yiran Shen, Xiaoming Chen, Yuk Ying Chung, Tongliang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28263](https://doi.org/10.1609/aaai.v38i5.28263)

**Abstract**:

The bio-inspired event cameras or dynamic vision sensors are capable of asynchronously capturing per-pixel brightness changes (called event-streams) in high temporal resolution and high dynamic range. However, the non-structural spatial-temporal event-streams make it challenging for providing intuitive visualization with rich semantic information for human vision. It calls for  events-to-video (E2V) solutions which take event-streams as input and generate high quality video frames for intuitive visualization. However, current solutions are predominantly data-driven without considering the prior knowledge of the underlying statistics relating event-streams and video frames. It highly relies on the non-linearity and generalization capability of the deep neural networks, thus, is struggling on reconstructing detailed textures when the scenes are complex.  In this work, we propose E2HQV, a novel E2V paradigm designed to produce high-quality video frames from events. This approach leverages a model-aided deep learning framework, underpinned by a theory-inspired E2V model, which is meticulously derived from the fundamental imaging principles of event cameras. To deal with the issue of state-reset in the recurrent components of E2HQV, we also design a temporal shift embedding module to further improve the quality of the video frames. Comprehensive evaluations on the real world event camera datasets validate our approach, with E2HQV, notably outperforming state-of-the-art approaches, e.g., surpassing the second best by over 40% for some evaluation metrics.

----

## [515] BLiRF: Bandlimited Radiance Fields for Dynamic Scene Modeling

**Authors**: *Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham, Anton van den Hengel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28264](https://doi.org/10.1609/aaai.v38i5.28264)

**Abstract**:

Inferring the 3D structure of a non-rigid dynamic scene from a single moving camera is an under-constrained problem. Inspired by the remarkable progress of  neural radiance fields (NeRFs) in photo-realistic novel view synthesis of static scenes, it has also been extended to dynamic settings. Such methods heavily rely on implicit neural priors to regularize the problem.  In this work, we take a step back and investigate how current implementations may entail deleterious effects including limited expressiveness, entanglement of light and density fields, and sub-optimal motion localization. Further, we devise a factorisation-based framework that represents the scene as a composition of bandlimited, high-dimensional  signals. We demonstrate  compelling results across complex dynamic scenes that involve changes in lighting, texture and long-range dynamics.

----

## [516] Cross-Sentence Gloss Consistency for Continuous Sign Language Recognition

**Authors**: *Qi Rao, Ke Sun, Xiaohan Wang, Qi Wang, Bang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28265](https://doi.org/10.1609/aaai.v38i5.28265)

**Abstract**:

Continuous sign language recognition (CSLR) aims to recognize gloss sequences from continuous sign videos. Recent works enhance the gloss representation consistency by mining correlations between visual and contextual modules within individual sentences. However, there still remain much richer correlations among glosses across different sentences. In this paper, we present a simple yet effective Cross-Sentence Gloss Consistency (CSGC), which enforces glosses belonging to a same category to be more consistent in representation than those belonging to different categories, across all training sentences. Specifically, in CSGC, a prototype is maintained for each gloss category and benefits the gloss discrimination in a contrastive way. Thanks to the well-distinguished gloss prototype, an auxiliary similarity classifier is devised to enhance the recognition clues, thus yielding more accurate results. Extensive experiments conducted on three CSLR datasets show that our proposed CSGC significantly boosts the performance of CSLR, surpassing existing state-of-the-art works by large margins (i.e., 1.6% on PHOENIX14, 2.4% on PHOENIX14-T, and 5.7% on CSL-Daily).

----

## [517] Forecasting Bimanual Object Manipulation Sequences from Unimanual Observations

**Authors**: *Haziq Razali, Yiannis Demiris*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28266](https://doi.org/10.1609/aaai.v38i5.28266)

**Abstract**:

Learning to forecast bimanual object manipulation sequences from unimanual observations has broad applications in assistive robots and augmented reality. This challenging task requires us to first infer motion from the missing arm and the object it would have been manipulating were the person bimanual, then forecast the human and object motion while maintaining hand-object contact during manipulation. Previous attempts model the hand-object interactions only implicitly, and thus tend to produce unrealistic motion where the objects float in air. We address this with a novel neural network that (i) identifies and forecasts the pose for only the objects undergoing motion through an object motion module and (ii) refines human pose predictions by encouraging hand-object contact during manipulation through an ensemble of human pose predictors. The components are also designed to be generic enough for use in both unimanual and bimanual contexts. Our approach outperforms the state-of-the-art pose forecasting methods on bimanual manipulation datasets.

----

## [518] Multi-Step Denoising Scheduled Sampling: Towards Alleviating Exposure Bias for Diffusion Models

**Authors**: *Zhiyao Ren, Yibing Zhan, Liang Ding, Gaoang Wang, Chaoyue Wang, Zhongyi Fan, Dacheng Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28267](https://doi.org/10.1609/aaai.v38i5.28267)

**Abstract**:

Denoising Diffusion Probabilistic Models (DDPMs) have achieved significant success in generation tasks. Nevertheless, the exposure bias issue, i.e., the natural discrepancy between the training (the output of each step is calculated individually by a given input) and inference (the output of each step is calculated based on the input iteratively obtained based on the model), harms the performance of DDPMs. To our knowledge, few works have tried to tackle this issue by modifying the training process for DDPMs, but they still perform unsatisfactorily due to 1) partially modeling the discrepancy and 2) ignoring the prediction error accumulation. To address the above issues, in this paper, we propose a multi-step denoising scheduled sampling (MDSS) strategy to alleviate the exposure bias for DDPMs. Analyzing the formulations of the training and inference of DDPMs, MDSS 1) comprehensively considers the discrepancy influence of prediction errors on the output of the model (the Gaussian noise) and the output of the step (the calculated input signal of the next step), and 2) efficiently models the prediction error accumulation by using multiple iterations of a mathematical formulation initialized from one-step prediction error obtained from the model. The experimental results, compared with previous works, demonstrate that our approach is more effective in mitigating exposure bias in DDPM, DDIM, and DPM-solver. In particular, MDSS achieves an FID score of 3.86 in 100 sample steps of DDIM on the CIFAR-10 dataset, whereas the second best obtains 4.78. The code will be available on GitHub.

----

## [519] CRA-PCN: Point Cloud Completion with Intra- and Inter-level Cross-Resolution Transformers

**Authors**: *Yi Rong, Haoran Zhou, Lixin Yuan, Cheng Mei, Jiahao Wang, Tong Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28268](https://doi.org/10.1609/aaai.v38i5.28268)

**Abstract**:

Point cloud completion is an indispensable task for recovering complete point clouds due to incompleteness caused by occlusion, limited sensor resolution, etc. The family of coarse-to-fine generation architectures has recently exhibited great success in point cloud completion and gradually became mainstream. In this work, we unveil one of the key ingredients behind these methods: meticulously devised feature extraction operations with explicit cross-resolution aggregation. We present Cross-Resolution Transformer that efficiently performs cross-resolution aggregation with local attention mechanisms. With the help of our recursive designs, the proposed operation can capture more scales of features than common aggregation operations, which is beneficial for capturing fine geometric characteristics. While prior methodologies have ventured into various manifestations of inter-level cross-resolution aggregation, the effectiveness of intra-level one and their combination has not been analyzed. With unified designs, Cross-Resolution Transformer can perform intra- or inter-level cross-resolution aggregation by switching inputs. We integrate two forms of Cross-Resolution Transformers into one up-sampling block for point generation, and following the coarse-to-fine manner, we construct CRA-PCN to incrementally predict complete shapes with stacked up-sampling blocks. Extensive experiments demonstrate that our method outperforms state-of-the-art methods by a large margin on several widely used benchmarks. Codes are available at https://github.com/EasyRy/CRA-PCN.

----

## [520] Entropic Open-Set Active Learning

**Authors**: *Bardia Safaei, Vibashan VS, Celso M. de Melo, Vishal M. Patel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28269](https://doi.org/10.1609/aaai.v38i5.28269)

**Abstract**:

Active Learning (AL) aims to enhance the performance of deep models by selecting the most informative samples for annotation from a pool of unlabeled data. Despite impressive performance in closed-set settings, most AL methods fail in real-world scenarios where the unlabeled data contains unknown categories. Recently, a few studies have attempted to tackle the AL problem for the open-set setting. However, these methods focus more on selecting known samples and do not efficiently utilize unknown samples obtained during AL rounds. In this work, we propose an Entropic Open-set AL (EOAL) framework which leverages both known and unknown distributions effectively to select informative samples during AL rounds. Specifically, our approach employs two different entropy scores. One measures the uncertainty of a sample with respect to the known-class distributions. The other measures the uncertainty of the sample with respect to the unknown-class distributions. By utilizing these two entropy scores we effectively separate the known and unknown samples from the unlabeled data resulting in better sampling. Through extensive experiments, we show that the proposed method outperforms existing state-of-the-art methods on CIFAR-10, CIFAR-100, and TinyImageNet datasets. Code is available at https://github.com/bardisafa/EOAL.

----

## [521] Generating Images of Rare Concepts Using Pre-trained Diffusion Models

**Authors**: *Dvir Samuel, Rami Ben-Ari, Simon Raviv, Nir Darshan, Gal Chechik*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28270](https://doi.org/10.1609/aaai.v38i5.28270)

**Abstract**:

Text-to-image diffusion models can synthesize high quality images, but they have various limitations. Here we highlight a common failure mode of these models, namely, generating uncommon concepts and structured concepts like hand palms. We show that their limitation is partly due to the long-tail nature of their training data: web-crawled data sets are strongly unbalanced, causing models to under-represent concepts from the tail of the distribution. We characterize the effect of unbalanced training data on text-to-image models and offer a remedy. We show that rare concepts can be correctly generated by carefully selecting suitable generation seeds in the noise space, using a small reference set of images, a technique that we call SeedSelect. SeedSelect does not require retraining or finetuning the diffusion model. We assess the faithfulness, quality and diversity of SeedSelect in creating rare objects and generating complex formations like hand images, and find it consistently achieves superior performance. We further show the advantage of SeedSelect in semantic data augmentation. Generating semantically appropriate images can successfully improve performance in few-shot recognition benchmarks, for classes from the head and from the tail of the training data of diffusion models.

----

## [522] RG-GAN: Dynamic Regenerative Pruning for Data-Efficient Generative Adversarial Networks

**Authors**: *Divya Saxena, Jiannong Cao, Jiahao Xu, Tarun Kulshrestha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28271](https://doi.org/10.1609/aaai.v38i5.28271)

**Abstract**:

Training Generative Adversarial Networks (GAN) to generate high-quality images typically requires large datasets. Network pruning during training has recently emerged as a significant advancement for data-efficient GAN. However, simple and straightforward pruning can lead to the risk of losing key information, resulting in suboptimal results due to GAN’s competitive dynamics between generator (G) and discriminator (D). Addressing this, we present RG-GAN, a novel approach that marks the first incorporation of dynamic weight regeneration and pruning in GAN training to improve the quality of the generated samples, even with limited data. Specifically, RG-GAN initiates layer-wise dynamic pruning by removing less important weights to the quality of the generated images. While pruning enhances efficiency, excessive sparsity within layers can pose a risk of model collapse. To mitigate this issue, RG-GAN applies a dynamic regeneration method to reintroduce specific weights when they become important, ensuring a balance between sparsity and image quality. Though effective, the sparse network achieved through this process might eliminate some weights important to the combined G and D performance, a crucial aspect for achieving stable and effective GAN training. RG-GAN addresses this loss of weights by integrating learned sparse network weights back into the dense network at the previous stage during a follow-up regeneration step. Our results consistently demonstrate RG-GAN’s robust performance across a variety of scenarios, including different GAN architectures, datasets, and degrees of data scarcity, reinforcing its value as a generic training methodology. Results also show that data augmentation exhibits improved performance in conjunction with RG-GAN. Furthermore, RG-GAN can achieve fewer parameters without compromising, and even enhancing, the quality of the generated samples. Code can be found at this link: https://github.com/IntellicentAI-Lab/RG-GAN

----

## [523] SeTformer Is What You Need for Vision and Language

**Authors**: *Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Michael Felsberg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28272](https://doi.org/10.1609/aaai.v38i5.28272)

**Abstract**:

The dot product self-attention (DPSA) is a fundamental component of transformers. However, scaling them to long sequences, like documents or high-resolution images, becomes prohibitively expensive due to the quadratic time and memory complexities arising from the softmax operation. Kernel methods are employed to simplify computations by approximating softmax but often lead to performance drops compared to softmax attention. We propose SeTformer, a novel transformer where DPSA is purely replaced by Self-optimal Transport (SeT) for achieving better performance and computational efficiency. SeT is based on two essential softmax properties: maintaining a non-negative attention matrix and using a nonlinear reweighting mechanism to emphasize important tokens in input sequences. By introducing a kernel cost function for optimal transport, SeTformer effectively satisfies these properties. In particular, with small and base-sized models, SeTformer achieves impressive top-1 accuracies of 84.7% and 86.2% on ImageNet-1K. In object detection, SeTformer-base outperforms the FocalNet counterpart by +2.2 mAP, using 38% fewer parameters and 29% fewer FLOPs. In semantic segmentation, our base-size model surpasses NAT by +3.5 mIoU with 33% fewer parameters. SeTformer also achieves state-of-the-art results in language modeling on the GLUE benchmark. These findings highlight SeTformer applicability for vision and language tasks.

----

## [524] Multi-Domain Multi-Scale Diffusion Model for Low-Light Image Enhancement

**Authors**: *Kai Shang, Mingwen Shao, Chao Wang, Yuanshuo Cheng, Shuigen Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28273](https://doi.org/10.1609/aaai.v38i5.28273)

**Abstract**:

Diffusion models have achieved remarkable progress in low-light image enhancement. However, there remain two practical limitations: (1) existing methods mainly focus on the spatial domain for the diffusion process, while neglecting the essential features in the frequency domain; (2) conventional patch-based sampling strategy inevitably leads to severe checkerboard artifacts due to the uneven overlapping. To address these limitations in one go, we propose a Multi-Domain Multi-Scale (MDMS) diffusion model for low-light image enhancement. In particular, we introduce a spatial-frequency fusion module to seamlessly integrates spatial and frequency information. By leveraging the Multi-Domain Learning (MDL) paradigm, our proposed model is endowed with the capability to adaptively facilitate noise distribution learning, thereby enhancing the quality of the generated images. Meanwhile, we propose a Multi-Scale Sampling (MSS) strategy that follows a divide-ensemble manner by merging the restored patches under different resolutions. Such a multi-scale learning paradigm explicitly derives patch information from different granularities, thus leading to smoother boundaries. Furthermore, we empirically adopt the Bright Channel Prior (BCP) which indicates natural statistical regularity as an additional restoration guidance. Experimental results on LOL and LOLv2 datasets demonstrate that our method achieves state-of-the-art performance for the low-light image enhancement task. Codes are available at https://github.com/Oliiveralien/MDMS.

----

## [525] Polyper: Boundary Sensitive Polyp Segmentation

**Authors**: *Hao Shao, Yang Zhang, Qibin Hou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28274](https://doi.org/10.1609/aaai.v38i5.28274)

**Abstract**:

We present a new boundary sensitive framework for polyp segmentation, termed Polyper.Our method is motivated by a clinical approach that seasoned medical practitioners often leverage the inherent features of interior polyp regions to tackle blurred boundaries.Inspired by this, we propose to explicitly leverages boundary regions to bolster the model's boundary discrimination capability while minimizing computational resource wastage. Our approach first extracts low-confidence boundary regions and high-confidence prediction regions from an initial segmentation map through differentiable morphological operators.Then, we design the boundary sensitive attention that concentrates on augmenting the features near the boundary regions using the high-confidence prediction region's characteristics to generate good segmentation results.Our proposed method can be seamlessly integrated with classical encoder networks, like ResNet-50, MiT-B1, and Swin Transformer.To evaludate the effectiveness of Polyper, we conduct experiments on five publicly available challenging datasets, and receive state-of-the-art performance on all of them. Code is available at https://github.com/haoshao-nku/medical_seg.git.

----

## [526] Collaborative Consortium of Foundation Models for Open-World Few-Shot Learning

**Authors**: *Shuai Shao, Yu Bai, Yan Wang, Baodi Liu, Bin Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28275](https://doi.org/10.1609/aaai.v38i5.28275)

**Abstract**:

Open-World Few-Shot Learning (OFSL) is a crucial research field dedicated to accurately identifying target samples in scenarios where data is limited and labels are unreliable. This research holds significant practical implications and is highly relevant to real-world applications. Recently, the advancements in foundation models like CLIP and DINO have showcased their robust representation capabilities even in resource-constrained settings with scarce data. This realization has brought about a transformative shift in focus, moving away from “building models from scratch” towards “effectively harnessing the potential of foundation models to extract pertinent prior knowledge suitable for OFSL and utilizing it sensibly”. Motivated by this perspective, we introduce the Collaborative Consortium of Foundation Models (CO3), which leverages CLIP, DINO, GPT-3, and DALL-E to collectively address the OFSL problem. CO3 comprises four key blocks: (1) the Label Correction Block (LC-Block) corrects unreliable labels, (2) the Data Augmentation Block (DA-Block) enhances available data, (3) the Feature Extraction Block (FE-Block) extracts multi-modal features, and (4) the Text-guided Fusion Adapter (TeFu-Adapter) integrates multiple features while mitigating the impact of noisy labels through semantic constraints. Only the adapter's parameters are adjustable, while the others remain frozen. Through collaboration among these foundation models, CO3 effectively unlocks their potential and unifies their capabilities to achieve state-of-the-art performance on multiple benchmark datasets. https://github.com/The-Shuai/CO3.

----

## [527] FaceCoresetNet: Differentiable Coresets for Face Set Recognition

**Authors**: *Gil Shapira, Yosi Keller*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28276](https://doi.org/10.1609/aaai.v38i5.28276)

**Abstract**:

In set-based face recognition, we aim to compute the most discriminative descriptor from an unbounded set of images and videos showing a single person. A discriminative descriptor balances two policies when aggregating information from a given set. The first is a quality-based policy: emphasizing high-quality and down-weighting low-quality images. The second is a diversity-based policy: emphasizing unique images in the set and down-weighting multiple occurrences of similar images as found in video clips which can overwhelm the set representation.
This work frames face-set representation as a differentiable coreset selection problem. Our model learns how to select a small coreset of the input set that balances quality and diversity policies using a learned metric parameterized by the face quality, optimized end-to-end. The selection process is a differentiable farthest-point sampling (FPS) realized by approximating the non-differentiable Argmax operation with differentiable sampling from the Gumbel-Softmax distribution of distances. The small coreset is later used as queries in a self and cross-attention architecture to enrich the descriptor with information from the whole set. Our model is order-invariant and linear in the input set size.
We set a new SOTA to set face verification on the IJB-B and IJB-C datasets. Our code is publicly available at https://github.com/ligaripash/FaceCoresetNet.

----

## [528] Decouple Content and Motion for Conditional Image-to-Video Generation

**Authors**: *Cuifeng Shen, Yulu Gan, Chen Chen, Xiongwei Zhu, Lele Cheng, Tingting Gao, Jinzhi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28277](https://doi.org/10.1609/aaai.v38i5.28277)

**Abstract**:

The goal of conditional image-to-video (cI2V) generation is to create a believable new video by beginning with the condition, i.e., one image and text. The previous cI2V generation methods conventionally perform in RGB pixel space, with limitations in modeling motion consistency and visual continuity. Additionally, the efficiency of generating videos in pixel space is quite low. In this paper, we propose a novel approach to address these challenges by disentangling the target RGB pixels into two distinct components: spatial content and temporal motions. Specifically, we predict temporal motions which include motion vector and residual based on a 3D-UNet diffusion model.  By explicitly modeling temporal motions and warping them to the starting image, we improve the temporal consistency of generated videos. This results in a reduction of spatial redundancy, emphasizing temporal details. Our proposed method achieves performance improvements by disentangling content and motion, all without introducing new structural complexities to the model. Extensive experiments on various datasets confirm our approach's superior performance over the majority of state-of-the-art methods in both effectiveness and efficiency.

----

## [529] GroundVLP: Harnessing Zero-Shot Visual Grounding from Vision-Language Pre-training and Open-Vocabulary Object Detection

**Authors**: *Haozhan Shen, Tiancheng Zhao, Mingwei Zhu, Jianwei Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28278](https://doi.org/10.1609/aaai.v38i5.28278)

**Abstract**:

Visual grounding, a crucial vision-language task involving the understanding of the visual context based on the query expression, necessitates the model to capture the interactions between objects, as well as various spatial and attribute information. However, the annotation data of visual grounding task is limited due to its time-consuming and labor-intensive annotation process, resulting in the trained models being constrained from generalizing its capability to a broader domain. To address this challenge, we propose GroundVLP, a simple yet effective zero-shot method that harnesses visual grounding ability from the existing models trained from image-text pairs and pure object detection data, both of which are more conveniently obtainable and offer a broader domain compared to visual grounding annotation data. GroundVLP proposes a fusion mechanism that combines the heatmap from GradCAM and the object proposals of open-vocabulary detectors. We demonstrate that the proposed method significantly outperforms other zero-shot methods on RefCOCO/+/g datasets, surpassing prior zero-shot state-of-the-art by approximately 28% on the test split of RefCOCO and RefCOCO+. Furthermore, GroundVLP performs comparably to or even better than some non-VLP-based supervised models on the Flickr30k entities dataset. Our code is available at https://github.com/om-ai-lab/GroundVLP.

----

## [530] Automatic Radiology Reports Generation via Memory Alignment Network

**Authors**: *Hongyu Shen, Mingtao Pei, Juncai Liu, Zhaoxing Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28279](https://doi.org/10.1609/aaai.v38i5.28279)

**Abstract**:

The automatic generation of radiology reports is of great significance, which can reduce the workload of doctors and improve the accuracy and reliability of medical diagnosis and treatment, and has attracted wide attention in recent years. Cross-modal mapping between images and text, a key component of generating high-quality reports, is challenging due to the lack of corresponding annotations. Despite its importance, previous studies have often overlooked it or lacked adequate designs for this crucial component. In this paper, we propose a method with memory alignment embedding to assist the model in aligning visual and textual features to generate a coherent and informative report. Specifically, we first get the memory alignment embedding by querying the memory matrix, where the query is derived from a combination of the visual features and their corresponding positional embeddings. Then the alignment between the visual and textual features can be guided by the memory alignment embedding during the generation process. The comparison experiments with other alignment methods show that the proposed alignment method is less costly and more effective. The proposed approach achieves better performance than state-of-the-art approaches on two public datasets IU X-Ray and MIMIC-CXR, which further demonstrates the effectiveness of the proposed alignment method.

----

## [531] CGMGM: A Cross-Gaussian Mixture Generative Model for Few-Shot Semantic Segmentation

**Authors**: *Junao Shen, Kun Kuang, Jiaheng Wang, Xinyu Wang, Tian Feng, Wei Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28280](https://doi.org/10.1609/aaai.v38i5.28280)

**Abstract**:

Few-shot semantic segmentation (FSS) aims to segment unseen objects in a query image using a few pixel-wise annotated support images, thus expanding the capabilities of semantic segmentation. The main challenge lies in extracting sufficient information from the limited support images to guide the segmentation process. Conventional methods typically address this problem by generating single or multiple prototypes from the support images and calculating their cosine similarity to the query image. However, these methods often fail to capture meaningful information for modeling the de facto joint distribution of pixel and category. Consequently, they result in incomplete segmentation of foreground objects and mis-segmentation of the complex background. To overcome this issue, we propose the Cross Gaussian Mixture Generative Model (CGMGM), a novel Gaussian Mixture Models~(GMMs)-based FSS method, which establishes the joint distribution of pixel and category in both the support and query images. Specifically, our method initially matches the feature representations of the query image with those of the support images to generate and refine an initial segmentation mask. It then employs GMMs to accurately model the joint distribution of foreground and background using the support masks and the initial segmentation mask. Subsequently, a parametric decoder utilizes the posterior probability of pixels in the query image, by applying the Bayesian theorem, to the joint distribution, to generate the final segmentation mask. Experimental results on PASCAL-5i and COCO-20i datasets demonstrate our CGMGM's effectiveness and superior performance compared to the state-of-the-art methods.

----

## [532] Learn How to See: Collaborative Embodied Learning for Object Detection and Camera Adjusting

**Authors**: *Lingdong Shen, Chunlei Huo, Nuo Xu, Chaowei Han, Zichen Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28281](https://doi.org/10.1609/aaai.v38i5.28281)

**Abstract**:

Passive object detectors, trained on large-scale static datasets, often overlook the feedback from object detection to image acquisition. Embodied vision and active detection mitigate this issue by interacting with the environment. Nevertheless, the materialization of activeness hinges on resource-intensive data collection and annotation. To tackle these challenges, we propose a collaborative student-teacher framework. Technically, a replay buffer is built based on the trajectory data to encapsulate the relationship of state, action, and reward. In addition, the student network diverges from reinforcement learning by redefining sequential decision pathways using a GPT structure enriched with causal self-attention. Moreover, the teacher network establishes a subtle state-reward mapping based on adjacent benefit differences, providing reliable rewards for student adaptively self-tuning with the vast unlabeled replay buffer data. Additionally, an innovative yet straightforward benefit reference value is proposed within the teacher network, adding to its effectiveness and simplicity. Leveraging a flexible replay buffer and embodied collaboration between teacher and student, the framework learns to see before detection with shallower features and shorter inference steps. Experiments highlight significant advantages of our algorithm over state-of-the-art detectors. The code is released at https://github.com/lydonShen/STF.

----

## [533] Distributed Manifold Hashing for Image Set Classification and Retrieval

**Authors**: *Xiaobo Shen, Peizhuo Song, Yun-Hao Yuan, Yuhui Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28282](https://doi.org/10.1609/aaai.v38i5.28282)

**Abstract**:

Conventional image set methods typically learn from image sets stored in one location. However, in real-world applications, image sets are often distributed or collected across different positions. Learning from such distributed image sets presents a challenge that has not been studied thus far. Moreover, efficiency is seldom addressed in large-scale image set applications. To fulfill these gaps, this paper proposes Distributed Manifold Hashing (DMH), which models distributed image sets as a connected graph. DMH employs Riemannian manifold to effectively represent each image set and further suggests learning hash code for each image set to achieve efficient computation and storage. DMH is formally formulated as a distributed learning problem with local consistency constraint on global variables among neighbor nodes, and can be optimized in parallel. Extensive experiments on three benchmark datasets demonstrate that DMH achieves highly competitive accuracies in a distributed setting and provides faster classification and retrieval than state-of-the-arts.

----

## [534] Controllable 3D Face Generation with Conditional Style Code Diffusion

**Authors**: *Xiaolong Shen, Jianxin Ma, Chang Zhou, Zongxin Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28283](https://doi.org/10.1609/aaai.v38i5.28283)

**Abstract**:

Generating photorealistic 3D faces from given conditions is a challenging task. Existing methods often rely on time-consuming one-by-one optimization approaches, which are not efficient for modeling the same distribution content, e.g., faces. Additionally, an ideal controllable 3D face generation model should consider both facial attributes and expressions.
Thus we propose a novel approach called TEx-Face(TExt & Expression-to-Face) that addresses these challenges by dividing the task into three components, i.e., 3D GAN Inversion, Conditional Style Code Diffusion, and 3D Face Decoding. For 3D GAN inversion, we introduce two methods, which aim to enhance the representation of style codes and alleviate 3D inconsistencies. Furthermore, we design a style code denoiser to incorporate multiple conditions into the style code and propose a data augmentation strategy to address the issue of insufficient paired visual-language data. Extensive experiments conducted on FFHQ, CelebA-HQ, and CelebA-Dialog demonstrate the promising performance of our TEx-Face in achieving the efficient and controllable generation of photorealistic 3D faces. The code will be publicly available.

----

## [535] Adaptive Integration of Partial Label Learning and Negative Learning for Enhanced Noisy Label Learning

**Authors**: *Mengmeng Sheng, Zeren Sun, Zhenhuang Cai, Tao Chen, Yichao Zhou, Yazhou Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28284](https://doi.org/10.1609/aaai.v38i5.28284)

**Abstract**:

There has been significant attention devoted to the effectiveness of various domains, such as semi-supervised learning, contrastive learning, and meta-learning, in enhancing the performance of methods for noisy label learning (NLL) tasks. However, most existing methods still depend on prior assumptions regarding clean samples amidst different sources of noise (e.g., a pre-defined drop rate or a small subset of clean samples). In this paper, we propose a simple yet powerful idea called NPN, which revolutionizes Noisy label learning by integrating Partial label learning (PLL) and Negative learning (NL). Toward this goal, we initially decompose the given label space adaptively into the candidate and complementary labels, thereby establishing the conditions for PLL and NL. We propose two adaptive data-driven paradigms of label disambiguation for PLL: hard disambiguation and soft disambiguation. Furthermore, we generate reliable complementary labels using all non-candidate labels for NL to enhance model robustness through indirect supervision. To maintain label reliability during the later stage of model training, we introduce a consistency regularization term that encourages agreement between the outputs of multiple augmentations. Experiments conducted on both synthetically corrupted and real-world noisy datasets demonstrate the superiority of NPN compared to other state-of-the-art (SOTA) methods. The source code has been made available at https://github.com/NUST-Machine-Intelligence-Laboratory/NPN.

----

## [536] Transformer-Based No-Reference Image Quality Assessment via Supervised Contrastive Learning

**Authors**: *Jinsong Shi, Pan Gao, Jie Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28285](https://doi.org/10.1609/aaai.v38i5.28285)

**Abstract**:

Image Quality Assessment (IQA) has long been a research hotspot in the field of image processing, especially No-Reference Image Quality Assessment (NR-IQA). Due to the powerful feature extraction ability, existing Convolution Neural Network (CNN) and Transformers based NR-IQA methods have achieved considerable progress. However, they still exhibit limited capability when facing unknown authentic distortion datasets. To further improve NR-IQA performance, in this paper, a novel supervised contrastive learning (SCL) and Transformer-based NR-IQA model SaTQA is proposed. We first train a model on a large-scale synthetic dataset by SCL (no image subjective score is required) to extract degradation features of images with various distortion types and levels. To further extract distortion information from images, we propose a backbone network incorporating the Multi-Stream Block (MSB) by combining the CNN inductive bias and Transformer long-term dependence modeling capability. Finally, we propose the Patch Attention Block (PAB) to obtain the final distorted image quality score by fusing the degradation features learned from contrastive learning with the perceptual distortion information extracted by the backbone network. Experimental results on six standard IQA datasets show that SaTQA outperforms the state-of-the-art methods for both synthetic and authentic datasets. Code is available at https://github.com/I2-Multimedia-Lab/SaTQA.

----

## [537] Explicit Visual Prompts for Visual Object Tracking

**Authors**: *Liangtao Shi, Bineng Zhong, Qihua Liang, Ning Li, Shengping Zhang, Xianxian Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28286](https://doi.org/10.1609/aaai.v38i5.28286)

**Abstract**:

How to effectively exploit spatio-temporal information is crucial to capture target appearance changes in visual tracking. However, most deep learning-based trackers mainly focus on designing a complicated appearance model or template updating strategy, while lacking the exploitation of context between consecutive frames and thus entailing the when-and-how-to-update dilemma. To address these issues, we propose a novel explicit visual prompts framework for visual tracking, dubbed EVPTrack. Specifically, we utilize spatio-temporal tokens to propagate information between consecutive frames without focusing on updating templates. As a result, we cannot only alleviate the challenge of when-to-update, but also avoid the hyper-parameters associated with updating strategies. Then, we utilize the spatio-temporal tokens to generate explicit visual prompts that facilitate inference in the current frame. The prompts are fed into a transformer encoder together with the image tokens without additional processing.
Consequently, the efficiency of our model is improved by avoiding how-to-update. In addition, we consider multi-scale information as explicit visual prompts, providing multiscale template features to enhance the EVPTrack's ability to handle target scale changes. Extensive experimental results on six benchmarks (i.e., LaSOT, LaSOText, GOT-10k, UAV123, TrackingNet, and TNL2K.) validate that our EVPTrack can achieve competitive performance at a real-time speed by effectively exploiting both spatio-temporal and multi-scale information. Code and models are available at
https://github.com/GXNU-ZhongLab/EVPTrack.

----

## [538] Evidential Uncertainty-Guided Mitochondria Segmentation for 3D EM Images

**Authors**: *Ruohua Shi, Lingyu Duan, Tiejun Huang, Tingting Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28287](https://doi.org/10.1609/aaai.v38i5.28287)

**Abstract**:

Recent advances in deep learning have greatly improved the segmentation of mitochondria from Electron Microscopy (EM) images. However, suffering from variations in mitochondrial morphology, imaging conditions, and image noise, existing methods still exhibit high uncertainty in their predictions. Moreover, in view of our findings, predictions with high levels of uncertainty are often accompanied by inaccuracies such as ambiguous boundaries and amount of false positive segments. To deal with the above problems, we propose a novel approach for mitochondria segmentation in 3D EM images that leverages evidential uncertainty estimation, which for the first time integrates evidential uncertainty to enhance the performance of segmentation. To be more specific, our proposed method not only provides accurate segmentation results, but also estimates associated uncertainty. Then, the estimated uncertainty is used to help improve the segmentation performance by an uncertainty rectification module, which leverages uncertainty maps and multi-scale information to refine the segmentation. Extensive experiments conducted on four challenging benchmarks demonstrate the superiority of our proposed method over existing approaches.

----

## [539] Towards Squeezing-Averse Virtual Try-On via Sequential Deformation

**Authors**: *Sang-Heon Shim, Jiwoo Chung, Jae-Pil Heo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28288](https://doi.org/10.1609/aaai.v38i5.28288)

**Abstract**:

In this paper, we first investigate a visual quality degradation problem observed in recent high-resolution virtual try-on approach. The tendency is empirically found that the textures of clothes are squeezed at the sleeve, as visualized in the upper row of Fig.1(a). A main reason for the issue arises from a gradient conflict between two popular losses, the Total Variation (TV) and adversarial losses. Specifically, the TV loss aims to disconnect boundaries between the sleeve and torso in a warped clothing mask, whereas the adversarial loss aims to combine between them. Such contrary objectives feedback the misaligned gradients to a cascaded appearance flow estimation, resulting in undesirable squeezing artifacts. To reduce this, we propose a Sequential Deformation (SD-VITON) that disentangles the appearance flow prediction layers into TV objective-dominant (TVOB) layers and a task-coexistence (TACO) layer. Specifically, we coarsely fit the clothes onto a human body via the TVOB layers, and then keep on refining via the TACO layer. In addition, the bottom row of Fig.1(a) shows a different type of squeezing artifacts around the waist. To address it, we further propose that we first warp the clothes into a tucked-out shirts style, and then partially erase the texture from the warped clothes without hurting the smoothness of the appearance flows. Experimental results show that our SD-VITON successfully resolves both types of artifacts and outperforms the baseline methods. Source code will be available at https://github.com/SHShim0513/SD-VITON.

----

## [540] DPA-P2PNet: Deformable Proposal-Aware P2PNet for Accurate Point-Based Cell Detection

**Authors**: *Zhongyi Shui, Sunyi Zheng, Chenglu Zhu, Shichuan Zhang, Xiaoxuan Yu, Honglin Li, Jingxiong Li, Pingyi Chen, Lin Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28289](https://doi.org/10.1609/aaai.v38i5.28289)

**Abstract**:

Point-based cell detection (PCD), which pursues high-performance cell sensing under low-cost data annotation, has garnered increased attention in computational pathology community. Unlike mainstream PCD methods that rely on intermediate density map representations, the Point-to-Point network (P2PNet) has recently emerged as an end-to-end solution for PCD, demonstrating impressive cell detection accuracy and efficiency. Nevertheless, P2PNet is limited to decoding from a single-level feature map due to the scale-agnostic property of point proposals, which is insufficient to leverage multi-scale information. Moreover, the spatial distribution of pre-set point proposals is biased from that of cells, leading to inaccurate cell localization. To lift these limitations, we present DPA-P2PNet in this work. The proposed method directly extracts multi-scale features for decoding according to the coordinates of point proposals on hierarchical feature maps. On this basis, we further devise deformable point proposals to mitigate the positional bias between proposals and potential cells to promote cell localization. Inspired by practical pathological diagnosis that usually combines high-level tissue structure and low-level cell morphology for accurate cell classification, we propose a multi-field-of-view (mFoV) variant of DPA-P2PNet to accommodate additional large FoV images with tissue information as model input. Finally, we execute the first self-supervised pre-training on immunohistochemistry histopathology image data and evaluate the suitability of four representative self-supervised methods on the PCD task. Experimental results on three benchmarks and a large-scale and real-world interval dataset demonstrate the superiority of our proposed models over the state-of-the-art counterparts. Codes and pre-trained weights are available at https://github.com/windygoo/DPA-P2PNet.

----

## [541] DVANet: Disentangling View and Action Features for Multi-View Action Recognition

**Authors**: *Nyle Siddiqui, Praveen Tirupattur, Mubarak Shah*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28290](https://doi.org/10.1609/aaai.v38i5.28290)

**Abstract**:

In this work, we present a novel approach to multi-view action recognition where we guide learned action representations to be separated from view-relevant information in a video. When trying to classify action instances captured from multiple viewpoints, there is a higher degree of difficulty due to the difference in background, occlusion, and visibility of the captured action from different camera angles. To tackle the various problems introduced in multi-view action recognition, we propose a novel configuration of learnable transformer decoder queries, in conjunction with two supervised contrastive losses, to enforce the learning of action features that are robust to shifts in viewpoints. Our disentangled feature learning occurs in two stages: the transformer decoder uses separate queries to separately learn action and view information, which are then further disentangled using our two contrastive losses. We show that our model and method of training significantly outperforms all other uni-modal models on four multi-view action recognition datasets: NTU RGB+D, NTU RGB+D 120, PKU-MMD, and N-UCLA. Compared to previous RGB works, we see maximal improvements of 1.5%, 4.8%, 2.2%, and 4.8% on each dataset, respectively. Our code can be found here: https://github.com/NyleSiddiqui/MultiView_Actions

----

## [542] Learning to Approximate Adaptive Kernel Convolution on Graphs

**Authors**: *Jaeyoon Sim, Sooyeon Jeon, Injun Choi, Guorong Wu, Won Hwa Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28291](https://doi.org/10.1609/aaai.v38i5.28291)

**Abstract**:

Various Graph Neural Networks (GNN) have been successful in analyzing data in non-Euclidean spaces, however, they have limitations such as oversmoothing, i.e., information becomes excessively averaged as the number of hidden layers increases. The issue stems from the intrinsic formulation of conventional graph convolution where the nodal features are aggregated from a direct neighborhood per layer across the entire nodes in the graph. As setting different number of hidden layers per node is infeasible, recent works leverage a diffusion kernel to redefine the graph structure and incorporate information from farther nodes. Unfortunately, such approaches suffer from heavy diagonalization of a graph Laplacian or learning a large transform matrix. In this regards, we propose a diffusion learning framework where the range of feature aggregation is controlled by the scale of a diffusion kernel. For efficient computation, we derive closed-form derivatives of approximations of the graph convolution with respect to the scale, so that node-wise range can be adaptively learned.With a downstream classifier, the entire framework is made trainable in an end-to-end manner. Our model is tested on various standard datasets for node-wise classification for the state-of-the-art performance, and it is also validated on a real-world brain network data for graph classifications to demonstrate its practicality for Alzheimer classification.

----

## [543] Semi-supervised Active Learning for Video Action Detection

**Authors**: *Ayush Singh, Aayush J. Rana, Akash Kumar, Shruti Vyas, Yogesh Singh Rawat*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28292](https://doi.org/10.1609/aaai.v38i5.28292)

**Abstract**:

In this work, we focus on label efficient learning for video
action detection. We develop a novel semi-supervised active
learning approach which utilizes both labeled as well as un-
labeled data along with informative sample selection for ac-
tion detection. Video action detection requires spatio-temporal
localization along with classification, which poses several
challenges for both active learning (informative sample se-
lection) as well as semi-supervised learning (pseudo label
generation). First, we propose NoiseAug, a simple augmenta-
tion strategy which effectively selects informative samples for
video action detection. Next, we propose fft-attention, a novel
technique based on high-pass filtering which enables effective
utilization of pseudo label for SSL in video action detection
by emphasizing on relevant activity region within a video.
We evaluate the proposed approach on three different bench-
mark datasets, UCF-101-24, JHMDB-21, and Youtube-VOS.
First, we demonstrate its effectiveness on video action detec-
tion where the proposed approach outperforms prior works in
semi-supervised and weakly-supervised learning along with
several baseline approaches in both UCF101-24 and JHMDB-
21. Next, we also show its effectiveness on Youtube-VOS for
video object segmentation demonstrating its generalization
capability for other dense prediction tasks in videos.

----

## [544] DeblurSR: Event-Based Motion Deblurring under the Spiking Representation

**Authors**: *Chen Song, Chandrajit Bajaj, Qixing Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28293](https://doi.org/10.1609/aaai.v38i5.28293)

**Abstract**:

We present DeblurSR, a novel motion deblurring approach that converts a blurry image into a sharp video. DeblurSR utilizes event data to compensate for motion ambiguities and exploits the spiking representation to parameterize the sharp output video as a mapping from time to intensity. Our key contribution, the Spiking Representation (SR), is inspired by the neuromorphic principles determining how biological neurons communicate with each other in living organisms. We discuss why the spikes can represent sharp edges and how the spiking parameters are interpreted from the neuromorphic perspective. DeblurSR has higher output quality and requires fewer computing resources than state-of-the-art event-based motion deblurring methods. We additionally show that our approach easily extends to video super-resolution when combined with recent advances in implicit neural representation.

----

## [545] Multi-Cross Sampling and Frequency-Division Reconstruction for Image Compressed Sensing

**Authors**: *Heping Song, Jingyao Gong, Hongying Meng, Yuping Lai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28294](https://doi.org/10.1609/aaai.v38i5.28294)

**Abstract**:

Deep Compressed Sensing (DCS) has attracted considerable interest due to its superior quality and speed compared to traditional CS algorithms. However, current approaches employ simplistic convolutional downsampling to acquire measurements, making it difficult to retain high-level features of the original signal for better image reconstruction. Furthermore, these approaches often overlook the presence of both high- and low-frequency information within the network, despite their critical role in achieving high-quality reconstruction. To address these challenges, we propose a novel Multi-Cross Sampling and Frequency Division Network (MCFD-Net) for image CS. The Dynamic Multi-Cross Sampling (DMCS) module, a sampling network of MCFD-Net, incorporates pyramid cross convolution and dual-branch sampling with multi-level pooling. Additionally, it introduces an attention mechanism between perception blocks to enhance adaptive learning effects. In the second deep reconstruction stage, we design a Frequency Division Reconstruction Module (FDRM). This module employs a discrete wavelet transform to extract high- and low-frequency information from images. It then applies multi-scale convolution and self-similarity attention compensation separately to both types of information before merging the output reconstruction results. The MCFD-Net integrates the DMCS and FDRM to construct an end-to-end learning network. Extensive CS experiments conducted on multiple benchmark datasets demonstrate that our MCFD-Net outperforms state-of-the-art approaches, while also exhibiting superior noise robustness.

----

## [546] Generalizable Fourier Augmentation for Unsupervised Video Object Segmentation

**Authors**: *Huihui Song, Tiankang Su, Yuhui Zheng, Kaihua Zhang, Bo Liu, Dong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28295](https://doi.org/10.1609/aaai.v38i5.28295)

**Abstract**:

The performance of existing unsupervised video object segmentation methods typically suffers from severe performance degradation on test videos when tested in out-of-distribution scenarios. The primary reason is that the test data in real-
world may not follow the independent and identically distribution (i.i.d.) assumption, leading to domain shift. In this paper, we propose a generalizable fourier augmentation method during training to improve the generalization ability of the model. To achieve this, we perform Fast Fourier Transform (FFT) over the intermediate spatial domain features in each layer to yield corresponding frequency representations, including amplitude components (encoding scene-aware styles such as texture, color, contrast of the scene) and phase components (encoding rich semantics). We produce a variety of style features via Gaussian sampling to augment the training data, thereby improving the generalization capability of the model. To further improve the cross-domain generalization
performance of the model, we design a phase feature update strategy via exponential moving average using phase features from past frames in an online update manner, which could help the model to learn cross-domain-invariant features. Extensive experiments show that our proposed method achieves
the state-of-the-art performance on popular benchmarks.

----

## [547] Semantic-Aware Autoregressive Image Modeling for Visual Representation Learning

**Authors**: *Kaiyou Song, Shan Zhang, Tong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28296](https://doi.org/10.1609/aaai.v38i5.28296)

**Abstract**:

The development of autoregressive modeling (AM) in computer vision lags behind natural language processing (NLP) in self-supervised pre-training. This is mainly caused by the challenge that images are not sequential signals and lack a natural order when applying autoregressive modeling. In this study, inspired by human beings’ way of grasping an image, i.e., focusing on the main object first, we present a semantic-aware autoregressive image modeling (SemAIM) method to tackle this challenge. The key insight of SemAIM is to autoregressively model images from the semantic patches to the less semantic patches. To this end, we first calculate a semantic-aware permutation of patches according to their feature similarities and then perform the autoregression procedure based on the permutation. In addition, considering that the raw pixels of patches are low-level signals and are not ideal prediction targets for learning high-level semantic representation, we also explore utilizing the patch features as the prediction targets. Extensive experiments are conducted on a broad range of downstream tasks, including image classification, object detection, and instance/semantic segmentation, to evaluate the performance of SemAIM. The results demonstrate SemAIM achieves state-of-the-art performance compared with other self-supervised methods. Specifically, with ViT-B, SemAIM achieves 84.1% top-1 accuracy for fine-tuning on ImageNet, 51.3% AP and 45.4% AP for object detection and instance segmentation on COCO, which outperforms the vanilla MAE by 0.5%, 1.0%, and 0.5%, respectively. Code is available at https://github.com/skyoux/SemAIM.

----

## [548] Self-Prompt Mechanism for Few-Shot Image Recognition

**Authors**: *Mingchen Song, Huiqiang Wang, Guoqiang Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28297](https://doi.org/10.1609/aaai.v38i5.28297)

**Abstract**:

Few-shot learning poses a formidable challenge as it necessitates effective recognition of novel classes based on a limited set of examples. Recent studies have sought to address the challenge of rare samples by tuning visual features through the utilization of external text prompts. However, the performance of these methods is constrained due to the inherent modality gap between the prompt text and image features. Instead of naively utilizing the external semantic information generated from text to guide the training of the image encoder, we propose a novel self-prompt mechanism (SPM) to adaptively adjust the neural network according to unseen data. Specifically, SPM involves a systematic selection of intrinsic semantic features generated by the image encoder across spatial and channel dimensions, thereby engendering self-prompt information. Subsequently, upon backpropagation of this self-prompt information to the deeper layers of the neural network, it effectively steers the network toward the learning and adaptation of new samples. Meanwhile, we propose a novel parameter-efficient tuning method that exclusively fine-tunes the parameters relevant to self-prompt (prompts are no more than 2% of the total parameters), and the incorporation of additional learnable parameters as self-prompt ensures the retention of prior knowledge through frozen encoder weights. Therefore, our method is highly suited for few-shot recognition tasks that require both information retention and adaptive adjustment of network parameters with limited labeling data constraints. Extensive experiments demonstrate the effectiveness of the proposed SPM in both 5-way 1-shot and 5-way 5-shot settings for standard single-domain and cross-domain few-shot recognition datasets, respectively. Our code is available at https://github.com/codeshop715/SPM.

----

## [549] Diverse Person: Customize Your Own Dataset for Text-Based Person Search

**Authors**: *Zifan Song, Guosheng Hu, Cairong Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28298](https://doi.org/10.1609/aaai.v38i5.28298)

**Abstract**:

Text-based person search is a challenging task aimed at locating specific target pedestrians through text descriptions. Recent advancements have been made in this field, but there remains a deficiency in datasets tailored for text-based person search. The creation of new, real-world datasets is hindered by concerns such as the risk of pedestrian privacy leakage and the substantial costs of annotation. In this paper, we introduce a framework, named Diverse Person (DP), to achieve efficient and high-quality text-based person search data generation without involving privacy concerns. Specifically, we propose to leverage available images of clothing and accessories as reference attribute images to edit the original dataset images through diffusion models. Additionally, we employ a Large Language Model (LLM) to produce annotations that are both high in quality and stylistically consistent with those found in real-world datasets. Extensive experimental results demonstrate that the baseline models trained with our DP can achieve new state-of-the-art results on three public datasets, with performance improvements up to 4.82%, 2.15%, and 2.28% on CUHK-PEDES, ICFG-PEDES, and RSTPReid in terms of Rank-1 accuracy, respectively.

----

## [550] V2Meow: Meowing to the Visual Beat via Video-to-Music Generation

**Authors**: *Kun Su, Judith Yue Li, Qingqing Huang, Dima Kuzmin, Joonseok Lee, Chris Donahue, Fei Sha, Aren Jansen, Yu Wang, Mauro Verzetti, Timo I. Denk*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28299](https://doi.org/10.1609/aaai.v38i5.28299)

**Abstract**:

Video-to-music generation demands both a temporally localized high-quality listening experience and globally aligned video-acoustic signatures. While recent music generation models excel at the former through advanced audio codecs, the exploration of video-acoustic signatures has been confined to specific visual scenarios. In contrast, our research confronts the challenge of learning globally aligned signatures between video and music directly from paired music and videos, without explicitly modeling domain-specific rhythmic or semantic relationships. We propose V2Meow, a video-to-music generation system capable of producing high-quality music audio for a diverse range of video input types using a multi-stage autoregressive model. Trained on 5k hours of music audio clips paired with video frames mined from in-the-wild music videos, V2Meow is competitive with previous domain-specific models when evaluated in a zero-shot manner. It synthesizes high-fidelity music audio waveforms solely by conditioning on pre-trained general-purpose visual features extracted from video frames, with optional style control via text prompts. Through both qualitative and quantitative evaluations, we demonstrate that our model outperforms various existing music generation systems in terms of visual-audio correspondence and audio quality. Music samples are available at tinyurl.com/v2meow.

----

## [551] F³-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis

**Authors**: *Sitong Su, Jianzhi Liu, Lianli Gao, Jingkuan Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28300](https://doi.org/10.1609/aaai.v38i5.28300)

**Abstract**:

Recently Text-to-Video (T2V) synthesis has undergone a breakthrough by training transformers or diffusion models on large-scale datasets. Nevertheless, inferring such large models incurs huge costs. Previous inference acceleration works either require costly retraining or are model-specific. To address this issue, instead of retraining we explore the inference process of two mainstream T2V models using transformers and diffusion models. The exploration reveals the redundancy in temporal attention modules of both models, which are commonly utilized to establish temporal relations among frames. Consequently, we propose a training-free and generalized pruning strategy called F3-Pruning to prune redundant temporal attention weights. Specifically, when aggregate temporal attention values are ranked below a certain ratio, corresponding weights will be pruned. Extensive experiments on three datasets using a classic transformer-based model CogVideo and a typical diffusion-based model Tune-A-Video verify the effectiveness of F3-Pruning in inference acceleration, quality assurance and broad applicability.

----

## [552] A Unified Environmental Network for Pedestrian Trajectory Prediction

**Authors**: *Yuchao Su, Yuanman Li, Wei Wang, Jiantao Zhou, Xia Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28301](https://doi.org/10.1609/aaai.v38i5.28301)

**Abstract**:

Accurately predicting pedestrian movements in complex environments is challenging due to social interactions, scene constraints, and pedestrians' multimodal behaviors. Sequential models like long short-term memory fail to effectively integrate scene features to make predicted trajectories comply with scene constraints due to disparate feature modalities of scene and trajectory. Though existing convolution neural network (CNN) models can extract scene features, they are ineffective in mapping these features into scene constraints for pedestrians and struggle to model pedestrian interactions due to the loss of target pedestrian information. To address these issues, we propose a unified environmental network based on CNN for pedestrian trajectory prediction. We introduce a polar-based method to reflect the distance and direction relationship between any position in the environment and the target pedestrian. This enables us to simultaneously model scene constraints and pedestrian social interactions in the form of feature maps. Additionally, we capture essential local features in the feature map, characterizing potential multimodal movements of pedestrians at each time step to prevent redundant predicted trajectories. We verify the performance of our proposed model on four trajectory prediction datasets, encompassing both short-term and long-term predictions. The experimental results demonstrate the superiority of our approach over existing methods.

----

## [553] LRANet: Towards Accurate and Efficient Scene Text Detection with Low-Rank Approximation Network

**Authors**: *Yuchen Su, Zhineng Chen, Zhiwen Shao, Yuning Du, Zhilong Ji, Jinfeng Bai, Yong Zhou, Yu-Gang Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28302](https://doi.org/10.1609/aaai.v38i5.28302)

**Abstract**:

Recently, regression-based methods, which predict parameterized text shapes for text localization, have gained popularity in scene text detection. However, the existing parameterized text shape methods still have limitations in modeling arbitrary-shaped texts due to ignoring the utilization of text-specific shape information. Moreover, the time consumption of the entire pipeline has been largely overlooked, leading to a suboptimal overall inference speed. To address these issues, we first propose a novel parameterized text shape method based on low-rank approximation. Unlike other shape representation methods that employ data-irrelevant parameterization, our approach utilizes singular value decomposition and reconstructs the text shape using a few eigenvectors learned from labeled text contours. By exploring the shape correlation among different text contours, our method achieves consistency, compactness, simplicity, and robustness in shape representation. Next, we propose a dual assignment scheme for speed acceleration. It adopts a sparse assignment branch to accelerate the inference speed, and meanwhile, provides ample supervised signals for training through a dense assignment branch. Building upon these designs, we implement an accurate and efficient arbitrary-shaped text detector named LRANet. Extensive experiments are conducted on several challenging benchmarks, demonstrating the superior accuracy and efficiency of LRANet compared to state-of-the-art methods. Code is available at: https://github.com/ychensu/LRANet.git

----

## [554] Spatial-Semantic Collaborative Cropping for User Generated Content

**Authors**: *Yukun Su, Yiwen Cao, Jingliang Deng, Fengyun Rao, Qingyao Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28303](https://doi.org/10.1609/aaai.v38i5.28303)

**Abstract**:

A large amount of User Generated Content (UGC) is uploaded to the Internet daily and displayed to people world-widely through the client side (mobile and PC). This requires the cropping algorithms to produce the aesthetic thumbnail within a specific aspect ratio on different devices. However, existing image cropping works mainly focus on landmark or landscape images, which fail to model the relations among the multi-objects with the complex background in UGC. Besides, previous methods merely consider the aesthetics of the cropped images while ignoring the content integrity, which is crucial for UGC cropping. In this paper, we propose a Spatial-Semantic Collaborative cropping network (S2CNet) for arbitrary user generated content accompanied by a new cropping benchmark. Specifically, we first mine the visual genes of the potential objects. Then, the suggested adaptive attention graph recasts this task as a procedure of information association over visual nodes. The underlying spatial and semantic relations are ultimately centralized to the crop candidate through differentiable message passing, which helps our network efficiently to preserve both the aesthetics and the content integrity. Extensive experiments on the proposed UGCrop5K and other public datasets demonstrate the superiority of our approach over state-of-the-art counterparts.

----

## [555] TR-DETR: Task-Reciprocal Transformer for Joint Moment Retrieval and Highlight Detection

**Authors**: *Hao Sun, Mingyao Zhou, Wenjing Chen, Wei Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28304](https://doi.org/10.1609/aaai.v38i5.28304)

**Abstract**:

Video moment retrieval (MR) and highlight detection (HD) based on natural language queries are two highly related tasks, which aim to obtain relevant moments within videos and highlight scores of each video clip. Recently, several methods have been devoted to building DETR-based networks to solve both MR and HD jointly. These methods simply add two separate task heads after multi-modal feature extraction and feature interaction, achieving good performance. Nevertheless, these approaches underutilize the reciprocal relationship between two tasks. In this paper, we propose a task-reciprocal transformer based on DETR (TR-DETR) that focuses on exploring the inherent reciprocity between MR and HD. Specifically, a local-global multi-modal alignment module is first built to align features from diverse modalities into a shared latent space. Subsequently, a visual feature refinement is designed to eliminate query-irrelevant information from visual features for modal interaction. Finally, a task cooperation module is constructed to refine the retrieval pipeline and the highlight score prediction process by utilizing the reciprocity between MR and HD. Comprehensive experiments on QVHighlights, Charades-STA and TVSum datasets demonstrate that TR-DETR outperforms existing state-of-the-art methods. Codes are available at https://github.com/mingyao1120/TR-DETR.

----

## [556] UniAP: Towards Universal Animal Perception in Vision via Few-Shot Learning

**Authors**: *Meiqi Sun, Zhonghan Zhao, Wenhao Chai, Hanjun Luo, Shidong Cao, Yanting Zhang, Jenq-Neng Hwang, Gaoang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28305](https://doi.org/10.1609/aaai.v38i5.28305)

**Abstract**:

Animal visual perception is an important technique for automatically monitoring animal health, understanding animal behaviors, and assisting animal-related research. However, it is challenging to design a deep learning-based perception model that can freely adapt to different animals across various perception tasks, due to the varying poses of a large diversity of animals, lacking data on rare species, and the semantic inconsistency of different tasks. We introduce UniAP, a novel Universal Animal Perception model that leverages few-shot learning to enable cross-species perception among various visual tasks. Our proposed model takes support images and labels as prompt guidance for a query image. Images and labels are processed through a Transformer-based encoder and a lightweight label encoder, respectively. Then a matching module is designed for aggregating information between prompt guidance and the query image, followed by a multi-head label decoder to generate outputs for various tasks. By capitalizing on the shared visual characteristics among different animals and tasks, UniAP enables the transfer of knowledge from well-studied species to those with limited labeled data or even unseen species. We demonstrate the effectiveness of UniAP through comprehensive experiments in pose estimation, segmentation, and classification tasks on diverse animal species, showcasing its ability to generalize and adapt to new classes with minimal labeled examples.

----

## [557] CFR-ICL: Cascade-Forward Refinement with Iterative Click Loss for Interactive Image Segmentation

**Authors**: *Shoukun Sun, Min Xian, Fei Xu, Luca Capriotti, Tiankai Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28306](https://doi.org/10.1609/aaai.v38i5.28306)

**Abstract**:

The click-based interactive segmentation aims to extract the object of interest from an image with the guidance of user clicks. Recent work has achieved great overall performance by employing feedback from the output. However, in most state-of-the-art approaches, 1) the inference stage involves inflexible heuristic rules and requires a separate refinement model, and 2) the number of user clicks and model performance cannot be balanced. To address the challenges, we propose a click-based and mask-guided interactive image segmentation framework containing three novel components: Cascade-Forward Refinement (CFR), Iterative Click Loss (ICL), and SUEM image augmentation. The CFR offers a unified inference framework to generate segmentation results in a coarse-to-fine manner. The proposed ICL allows model training to improve segmentation and reduce user interactions simultaneously. The proposed SUEM augmentation is a comprehensive way to create large and diverse training sets for interactive image segmentation. Extensive experiments demonstrate the state-of-the-art performance of the proposed approach on five public datasets. Remarkably, our model reduces by 33.2%, and 15.5% the number of clicks required to surpass an IoU of 0.95 in the previous state-of-the-art approach on the Berkeley and DAVIS sets, respectively.

----

## [558] RL-SeqISP: Reinforcement Learning-Based Sequential Optimization for Image Signal Processing

**Authors**: *Xinyu Sun, Zhikun Zhao, Lili Wei, Congyan Lang, Mingxuan Cai, Longfei Han, Juan Wang, Bing Li, Yuxuan Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28307](https://doi.org/10.1609/aaai.v38i5.28307)

**Abstract**:

Hardware image signal processing (ISP), aiming at converting RAW inputs to RGB images, consists of a series of processing blocks, each with multiple parameters. Traditionally, ISP parameters are manually tuned in isolation by imaging experts according to application-specific quality and performance metrics, which is time-consuming and biased towards human perception due to complex interaction with the output image. Since the relationship between any single parameter’s variation and the output performance metric is a complex, non-linear function, optimizing such a large number of ISP parameters is challenging. To address this challenge, we propose a novel Sequential ISP parameter optimization model, called the RL-SeqISP model, which utilizes deep reinforcement learning to jointly optimize all ISP parameters for a variety of imaging applications. Concretely, inspired by the sequential tuning process of human experts, the proposed model can progressively enhance image quality by seamlessly integrating information from both the image feature space and the parameter space. Furthermore, a dynamic parameter optimization module is introduced to avoid ISP parameters getting stuck into local optima, which is able to more effectively guarantee the optimal parameters resulting from the sequential learning strategy. These merits of the RL-SeqISP model as well as its high efficiency are substantiated by comprehensive experiments on a wide range of downstream tasks, including two visual analysis tasks (instance segmentation and object detection), and image quality assessment (IQA), as compared with representative methods both quantitatively and qualitatively. In particular, even using only 10% of the training data, our model outperforms other SOTA methods by an average of 7% mAP on two visual analysis tasks.

----

## [559] PathAsst: A Generative Foundation AI Assistant towards Artificial General Intelligence of Pathology

**Authors**: *Yuxuan Sun, Chenglu Zhu, Sunyi Zheng, Kai Zhang, Lin Sun, Zhongyi Shui, Yunlong Zhang, Honglin Li, Lin Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28308](https://doi.org/10.1609/aaai.v38i5.28308)

**Abstract**:

As advances in large language models (LLMs) and multimodal techniques continue to mature, the development of general-purpose multimodal large language models (MLLMs) has surged, offering significant applications in interpreting natural images. However, the field of pathology has largely remained untapped, particularly in gathering high-quality data and designing comprehensive model frameworks. To bridge the gap in pathology MLLMs, we present PathAsst, a multimodal generative foundation AI assistant to revolutionize diagnostic and predictive analytics in pathology. The development of PathAsst involves three pivotal steps:  data acquisition, CLIP model adaptation, and the training of PathAsst's multimodal generative capabilities. Firstly, we collect over 207K high-quality pathology image-text pairs from authoritative sources. Leveraging the advanced power of ChatGPT, we generate over 180K instruction-following samples. Furthermore, we devise additional instruction-following data specifically tailored for invoking eight pathology-specific sub-models we prepared, allowing the PathAsst to effectively collaborate with these models, enhancing its diagnostic ability. Secondly, by leveraging the collected data, we construct PathCLIP, a pathology-dedicated CLIP, to enhance PathAsst's capabilities in interpreting pathology images. Finally, we integrate PathCLIP with the Vicuna-13b and utilize pathology-specific instruction-tuning data to enhance the multimodal generation capacity of PathAsst and bolster its synergistic interactions with sub-models. The experimental results of PathAsst show the potential of harnessing AI-powered generative foundation model to improve pathology diagnosis and treatment processes. We open-source our dataset, as well as a comprehensive toolkit for extensive pathology data collection and preprocessing at https://github.com/superjamessyx/Generative-Foundation-AI-Assistant-for-Pathology.

----

## [560] FG-EmoTalk: Talking Head Video Generation with Fine-Grained Controllable Facial Expressions

**Authors**: *Zhaoxu Sun, Yuze Xuan, Fang Liu, Yang Xiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28309](https://doi.org/10.1609/aaai.v38i5.28309)

**Abstract**:

Although deep generative models have greatly improved one-shot video-driven talking head generation, few studies address fine-grained controllable facial expression editing, which is crucial for practical applications. Existing methods rely on a fixed set of predefined discrete emotion labels or simply copy expressions from input videos. This is limiting as expressions are complex, and methods using only emotion labels cannot generate fine-grained, accurate or mixed expressions. Generating talking head video with precise expressions is also difficult using 3D model-based approaches, as 3DMM only models facial movements and tends to produce deviations. In this paper, we propose a novel framework enabling fine-grained facial expression editing in talking face generation. Our goal is to achieve expression control by manipulating the intensities of individual facial Action Units (AUs) or groups. First, compared with existing methods which decouple the face into pose and expression, we propose a disentanglement scheme to isolates three components from the human face, namely, appearance, pose, and expression. Second, we propose to use input AUs to control muscle group intensities in the generated face, and integrate the AUs features with the disentangled expression latent code. Finally, we present a self-supervised training strategy with well-designed constraints. Experiments show our method achieves fine-grained expression control, produces high-quality talking head videos and outperforms baseline methods.

----

## [561] Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning

**Authors**: *Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28310](https://doi.org/10.1609/aaai.v38i5.28310)

**Abstract**:

This research addresses the challenge of developing a universal deepfake detector that can effectively identify unseen deepfake images despite limited training data.  Existing frequency-based paradigms have relied on frequency-level artifacts introduced during the up-sampling in GAN pipelines to detect forgeries. However, the rapid advancements in synthesis technology have led to specific artifacts for each generation model. Consequently, these detectors have exhibited a lack of proficiency in learning the frequency domain and tend to overfit to the artifacts present in the training data, leading to suboptimal performance on unseen sources. To address this issue, we introduce a novel frequency-aware approach called FreqNet, centered around frequency domain learning, specifically designed to enhance the generalizability of deepfake detectors. Our method forces the detector to continuously focus on high-frequency information, exploiting high-frequency representation of features across spatial and channel dimensions. Additionally, we incorporate a straightforward frequency domain learning module to learn source-agnostic features. It involves convolutional layers applied to both the phase spectrum and amplitude spectrum between the Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (iFFT). Extensive experimentation involving 17 GANs demonstrates the effectiveness of our proposed method, showcasing state-of-the-art performance (+9.8\%) while requiring fewer parameters. The code is available at https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection.

----

## [562] Compound Text-Guided Prompt Tuning via Image-Adaptive Cues

**Authors**: *Hao Tan, Jun Li, Yizhuang Zhou, Jun Wan, Zhen Lei, Xiangyu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28311](https://doi.org/10.1609/aaai.v38i5.28311)

**Abstract**:

Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable generalization capabilities to downstream tasks. However, existing prompt tuning based frameworks need to parallelize learnable textual inputs for all categories, suffering from massive GPU memory consumption when there is a large number of categories in the target dataset. Moreover, previous works require to include category names within prompts, exhibiting subpar performance when dealing with ambiguous category names. To address these shortcomings, we propose Compound Text-Guided Prompt Tuning (TGP-T) that significantly reduces resource demand while achieving superior performance. We introduce text supervision to the optimization of prompts, which enables two benefits: 1) releasing the model reliance on the pre-defined category names during inference, thereby enabling more flexible prompt generation; 2) reducing the number of inputs to the text encoder, which decreases GPU memory consumption significantly. Specifically, we found that compound text supervisions, i.e., category-wise and content-wise, is highly effective, since they provide inter-class separability and capture intra-class variations, respectively. Moreover, we condition the prompt generation on visual features through a module called Bonder, which facilitates the alignment between prompts and visual features. Extensive experiments on few-shot recognition and domain generalization demonstrate that TGP-T achieves superior performance with consistently lower training costs. It reduces GPU memory usage by 93% and attains a 2.5% performance gain on 16-shot ImageNet. The code is available at https://github.com/EricTan7/TGP-T.

----

## [563] Occluded Person Re-identification via Saliency-Guided Patch Transfer

**Authors**: *Lei Tan, Jiaer Xia, Wenfeng Liu, Pingyang Dai, Yongjian Wu, Liujuan Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28312](https://doi.org/10.1609/aaai.v38i5.28312)

**Abstract**:

While generic person re-identification has made remarkable improvement in recent years, these methods are designed under the assumption that the entire body of the person is available. This assumption brings about a significant performance degradation when suffering from occlusion caused by various obstacles in real-world applications. To address this issue, data-driven strategies have emerged to enhance the model's robustness to occlusion. Following the random erasing paradigm, these strategies typically employ randomly generated noise to supersede randomly selected image regions to simulate obstacles. However, the random strategy is not sensitive to location and content, meaning they cannot mimic real-world occlusion cases in application scenarios. To overcome this limitation and fully exploit the real scene information in datasets, this paper proposes a more intuitive and effective data-driven strategy named Saliency-Guided Patch Transfer (SPT). Combined with the vision transformer, SPT divides person instances and background obstacles using salient patch selection. By transferring person instances to different background obstacles, SPT can easily generate photo-realistic occluded samples. Furthermore, we propose an occlusion-aware Intersection over Union (OIoU) with mask-rolling to filter the more suitable combination and a class-ignoring strategy to achieve more stable processing. Extensive experimental evaluations conducted on occluded and holistic person re-identification benchmarks demonstrate that SPT provides a significant performance gain among different ViT-based ReID algorithms on occluded ReID.

----

## [564] Style2Talker: High-Resolution Talking Head Generation with Emotion Style and Art Style

**Authors**: *Shuai Tan, Bin Ji, Ye Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28313](https://doi.org/10.1609/aaai.v38i5.28313)

**Abstract**:

Although automatically animating audio-driven talking heads has recently received growing interest, previous efforts have mainly concentrated on achieving lip synchronization with the audio, neglecting two crucial elements for generating expressive videos: emotion style and art style. In this paper, we present an innovative audio-driven talking face generation method called Style2Talker. It involves two stylized stages, namely Style-E and Style-A, which integrate text-controlled emotion style and picture-controlled art style into the final output. In order to prepare the scarce emotional text descriptions corresponding to the videos, we propose a labor-free paradigm that employs large-scale pretrained models to automatically annotate emotional text labels for existing audio-visual datasets. Incorporating the synthetic emotion texts, the Style-E stage utilizes a large-scale CLIP model to extract emotion representations, which are combined with the audio, serving as the condition for an efficient latent diffusion model designed to produce emotional motion coefficients of a 3DMM model. Moving on to the Style-A stage, we develop a coefficient-driven motion generator and an art-specific style path embedded in the well-known StyleGAN. This allows us to synthesize high-resolution artistically stylized talking head videos using the generated emotional motion coefficients and an art style source picture. Moreover, to better preserve image details and avoid artifacts, we provide StyleGAN with the multi-scale content features extracted from the identity image and refine its intermediate feature maps by the designed content encoder and refinement network, respectively. Extensive experimental results demonstrate our method outperforms existing state-of-the-art methods in terms of audio-lip synchronization and performance of both emotion style and art style.

----

## [565] Say Anything with Any Style

**Authors**: *Shuai Tan, Bin Ji, Yu Ding, Ye Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i5.28314](https://doi.org/10.1609/aaai.v38i5.28314)

**Abstract**:

Generating stylized talking head with diverse head motions is crucial for achieving natural-looking videos but still remains challenging. Previous works either adopt a regressive method to capture the speaking style, resulting in a coarse style that is averaged across all training data, or employ a universal network to synthesize videos with different styles which causes suboptimal performance. To address these, we propose a novel dynamic-weight method, namely Say Anything with Any Style (SAAS), which queries the discrete style representation via a generative model with a learned style codebook. Specifically, we develop a multi-task VQ-VAE that incorporates three closely related tasks to learn a style codebook as a prior for style extraction. This discrete prior, along with the generative model, enhances the precision and robustness when extracting the speaking styles of the given style clips. By utilizing the extracted style, a residual architecture comprising a canonical branch and style-specific branch is employed to predict the mouth shapes conditioned on any driving audio while transferring the speaking style from the source to any desired one. To adapt to different speaking styles, we steer clear of employing a universal network by exploring an elaborate HyperStyle to produce the style-specific weights offset for the style branch. Furthermore, we construct a pose generator and a pose codebook to store the quantized pose representation, allowing us to sample diverse head motions aligned with the audio and the extracted style. Experiments demonstrate that our approach surpasses state-of-the-art methods in terms of both lip-synchronization and stylized expression. Besides, we extend our SAAS to video-driven style editing field and achieve satisfactory performance as well.

----

## [566] Semantic-Aware Data Augmentation for Text-to-Image Synthesis

**Authors**: *Zhaorui Tan, Xi Yang, Kaizhu Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28315](https://doi.org/10.1609/aaai.v38i6.28315)

**Abstract**:

Data augmentation has been recently leveraged as an effective regularizer in various vision-language deep neural networks. However, in text-to-image synthesis (T2Isyn), current augmentation wisdom still suffers from the semantic mismatch between augmented paired data. Even worse, semantic collapse may occur when generated images are less semantically constrained. In this paper, we develop a novel Semantic-aware Data Augmentation (SADA) framework dedicated to T2Isyn. In particular, we propose to augment texts in the semantic space via an Implicit Textual Semantic Preserving Augmentation, in conjunction with a specifically designed Image Semantic Regularization Loss as Generated Image Semantic Conservation, to cope well with semantic mismatch and collapse. As one major contribution, we theoretically show that  Implicit Textual Semantic Preserving Augmentation can certify better text-image consistency while Image Semantic Regularization Loss regularizing the semantics of generated images would avoid semantic collapse and enhance image quality. Extensive experiments validate that SADA enhances text-image consistency and improves image quality significantly in T2Isyn models across various backbones. Especially, incorporating SADA during the tuning process of Stable Diffusion models also yields performance improvements.

----

## [567] Data-Free Generalized Zero-Shot Learning

**Authors**: *Bowen Tang, Jing Zhang, Long Yan, Qian Yu, Lu Sheng, Dong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28316](https://doi.org/10.1609/aaai.v38i6.28316)

**Abstract**:

Deep learning models have the ability to extract rich knowledge from large-scale datasets. However, the sharing of data has become increasingly challenging due to concerns regarding data copyright and privacy. Consequently, this hampers the effective transfer of knowledge from existing data to novel downstream tasks and concepts. Zero-shot learning (ZSL) approaches aim to recognize new classes by transferring semantic knowledge learned from base classes. However, traditional generative ZSL methods often require access to real images from base classes and rely on manually annotated attributes, which presents challenges in terms of data restrictions and model scalability. To this end, this paper tackles a challenging and practical problem dubbed as data-free zero-shot learning (DFZSL), where only the CLIP-based base classes data pre-trained classifier is available for zero-shot classification. Specifically, we propose a generic framework for DFZSL, which consists of three main components. Firstly, to recover the virtual features of the base data, we model the CLIP features of base class images as samples from a von Mises-Fisher (vMF) distribution based on the pre-trained classifier. Secondly, we leverage the text features of CLIP as low-cost semantic information and propose a feature-language prompt tuning (FLPT) method to further align the virtual image features and textual features. Thirdly, we train a conditional generative model using the well-aligned virtual image features and corresponding semantic text features, enabling the generation of new classes features and achieve better zero-shot generalization. Our framework has been evaluated on five commonly used benchmarks for generalized ZSL, as well as 11 benchmarks for the base-to-new ZSL. The results demonstrate the superiority and effectiveness of our approach. Our code is available in https://github.com/ylong4/DFZSL.

----

## [568] Offline and Online Optical Flow Enhancement for Deep Video Compression

**Authors**: *Chuanbo Tang, Xihua Sheng, Zhuoyuan Li, Haotian Zhang, Li Li, Dong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28317](https://doi.org/10.1609/aaai.v38i6.28317)

**Abstract**:

Video compression relies heavily on exploiting the temporal redundancy between video frames, which is usually achieved by estimating and using the motion information. The motion information is represented as optical flows in most of the existing deep video compression networks. Indeed, these networks often adopt pre-trained optical flow estimation networks for motion estimation. The optical flows, however, may be less suitable for video compression due to the following two factors. First, the optical flow estimation networks were trained to perform inter-frame prediction as accurately as possible, but the optical flows themselves may cost too many bits to encode. Second, the optical flow estimation networks were trained on synthetic data, and may not generalize well enough to real-world videos. We address the twofold limitations by enhancing the optical flows in two stages: offline and online. In the offline stage, we fine-tune a trained optical flow estimation network with the motion information provided by a traditional (non-deep) video compression scheme, e.g. H.266/VVC, as we believe the motion information of H.266/VVC achieves a better rate-distortion trade-off. In the online stage, we further optimize the latent features of the optical flows with a gradient descent-based algorithm for the video to be compressed, so as to enhance the adaptivity of the optical flows. We conduct experiments on two state-of-the-art deep video compression schemes, DCVC and DCVC-DC. Experimental results demonstrate that the proposed offline and online enhancement together achieves on average 13.4% bitrate saving for DCVC and 4.1% bitrate saving for DCVC-DC on the tested videos, without increasing the model or computational complexity of the decoder side.

----

## [569] Manifold Constraints for Imperceptible Adversarial Attacks on Point Clouds

**Authors**: *Keke Tang, Xu He, Weilong Peng, Jianpeng Wu, Yawen Shi, Daizong Liu, Pan Zhou, Wenping Wang, Zhihong Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28318](https://doi.org/10.1609/aaai.v38i6.28318)

**Abstract**:

Adversarial attacks on 3D point clouds often exhibit unsatisfactory imperceptibility, which primarily stems from the disregard for manifold-aware distortion, i.e., distortion of the underlying 2-manifold surfaces. In this paper, we develop novel manifold constraints to reduce such distortion, aiming to enhance the imperceptibility of adversarial attacks on 3D point clouds. Specifically, we construct a bijective manifold mapping between point clouds and a simple parameter shape using an invertible auto-encoder. Consequently, manifold-aware distortion during attacks can be captured within the parameter space. By enforcing manifold constraints that preserve local properties of the parameter shape, manifold-aware distortion is effectively mitigated, ultimately leading to enhanced imperceptibility. Extensive experiments demonstrate that integrating manifold constraints into conventional adversarial attack solutions yields superior imperceptibility, outperforming the state-of-the-art methods.

----

## [570] Once and for All: Universal Transferable Adversarial Perturbation against Deep Hashing-Based Facial Image Retrieval

**Authors**: *Long Tang, Dengpan Ye, Yunna Lv, Chuanxi Chen, Yunming Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28319](https://doi.org/10.1609/aaai.v38i6.28319)

**Abstract**:

Deep Hashing (DH)-based image retrieval has been widely applied to face-matching systems due to its accuracy and efficiency. However, this convenience comes with an increased risk of privacy leakage. DH models inherit the vulnerability to adversarial attacks, which can be used to prevent the retrieval of private images. Existing adversarial attacks against DH typically target a single image or a specific class of images, lacking universal adversarial perturbation for the entire hash dataset. In this paper, we propose the first universal transferable adversarial perturbation against DH-based facial image retrieval, a single perturbation can protect all images. Specifically, we explore the relationship between clusters learned by different DH models and define the optimization objective of universal perturbation as leaving from the overall hash center. To mitigate the challenge of single-objective optimization, we randomly obtain sub-cluster centers and further propose sub-task-based meta-learning to aid in overall optimization. We test our method with popular facial datasets and DH models, indicating impressive cross-image, -identity, -model, and -scheme universal anti-retrieval performance. Compared to state-of-the-art methods, our performance is competitive in white-box settings and exhibits significant improvements of 10%-70% in transferability in all black-box settings.

----

## [571] Prior and Prediction Inverse Kernel Transformer for Single Image Defocus Deblurring

**Authors**: *Peng Tang, Zhiqiang Xu, Chunlai Zhou, Pengfei Wei, Peng Han, Xin Cao, Tobias Lasser*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28320](https://doi.org/10.1609/aaai.v38i6.28320)

**Abstract**:

Defocus blur, due to spatially-varying sizes and shapes, is hard to remove. Existing methods either are unable to effectively handle irregular defocus blur or fail to generalize well on other datasets.
In this work, we propose a divide-and-conquer approach to tackling this issue, which gives rise to a novel end-to-end deep learning method, called prior-and-prediction inverse kernel transformer (P2IKT), for single image defocus deblurring. Since most defocus blur can be approximated as Gaussian blur or its variants, we construct an inverse Gaussian kernel module in our method to enhance its generalization ability. At the same time, an inverse kernel prediction module is introduced in order to flexibly address the irregular blur that cannot be approximated by Gaussian blur. We further design a scale recurrent transformer, which estimates mixing coefficients for adaptively combining the results from the two modules and runs the scale recurrent ``coarse-to-fine" procedure for progressive defocus deblurring. Extensive experimental results demonstrate that our P2IKT outperforms previous methods in terms of PSNR on multiple defocus deblurring datasets.

----

## [572] Semantic Lens: Instance-Centric Semantic Alignment for Video Super-resolution

**Authors**: *Qi Tang, Yao Zhao, Meiqin Liu, Jian Jin, Chao Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28321](https://doi.org/10.1609/aaai.v38i6.28321)

**Abstract**:

As a critical clue of video super-resolution (VSR), inter-frame alignment significantly impacts overall performance. However, accurate pixel-level alignment is a challenging task due to the intricate motion interweaving in the video. In response to this issue, we introduce a novel paradigm for VSR named Semantic Lens, predicated on semantic priors drawn from degraded videos. Specifically, video is modeled as instances, events, and scenes via a Semantic Extractor. Those semantics assist the Pixel Enhancer in understanding the recovered contents and generating more realistic visual results. The distilled global semantics embody the scene information of each frame, while the instance-specific semantics assemble the spatial-temporal contexts related to each instance. Furthermore, we devise a Semantics-Powered Attention Cross-Embedding (SPACE) block to bridge the pixel-level features with semantic knowledge, composed of a Global Perspective Shifter (GPS) and an Instance-Specific Semantic Embedding Encoder (ISEE). Concretely, the GPS module generates pairs of affine transformation parameters for pixel-level feature modulation conditioned on global semantics. After that the ISEE module harnesses the attention mechanism to align the adjacent frames in the instance-centric semantic space. In addition, we incorporate a simple yet effective pre-alignment module to alleviate the difficulty of model training. Extensive experiments demonstrate the superiority of our model over existing state-of-the-art VSR methods.

----

## [573] Boosting Residual Networks with Group Knowledge

**Authors**: *Shengji Tang, Peng Ye, Baopu Li, Weihao Lin, Tao Chen, Tong He, Chong Yu, Wanli Ouyang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28322](https://doi.org/10.1609/aaai.v38i6.28322)

**Abstract**:

Recent research understands the residual networks from a new perspective of the implicit ensemble model. From this view, previous methods such as stochastic depth and stimulative training have further improved the performance of the residual network by sampling and training of its subnets. However, they both use the same supervision for all subnets of different capacities and neglect the valuable knowledge generated by subnets during training. In this manuscript, we mitigate the significant knowledge distillation gap caused by using the same kind of supervision and advocate leveraging the subnets to provide diverse knowledge. Based on this motivation, we propose a group knowledge based training framework for boosting the performance of residual networks. Specifically, we implicitly divide all subnets into hierarchical groups by subnet-in-subnet sampling, aggregate the knowledge of different subnets in each group during training, and exploit upper-level group knowledge to supervise lower-level subnet group. Meanwhile, we also develop a subnet sampling strategy that naturally samples larger subnets, which are found to be more helpful than smaller subnets in boosting performance for hierarchical groups. Compared with typical subnet training and other methods, our method achieves the best efficiency and performance trade-offs on multiple datasets and network structures. The code is at https://github.com/tsj-001/AAAI24-GKT.

----

## [574] Point-PEFT: Parameter-Efficient Fine-Tuning for 3D Pre-trained Models

**Authors**: *Yiwen Tang, Ray Zhang, Zoey Guo, Xianzheng Ma, Bin Zhao, Zhigang Wang, Dong Wang, Xuelong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28323](https://doi.org/10.1609/aaai.v38i6.28323)

**Abstract**:

The popularity of pre-trained large models has revolutionized downstream tasks across diverse fields, such as language, vision, and multi-modality. To minimize the adaption cost for downstream tasks, many Parameter-Efficient Fine-Tuning (PEFT) techniques are proposed for language and 2D image pre-trained models. However, the specialized PEFT method for 3D pre-trained models is still under-explored. To this end, we introduce Point-PEFT, a novel framework for adapting point cloud pre-trained models with minimal learnable parameters. Specifically, for a pre-trained 3D model, we freeze most of its parameters, and only tune the newly added PEFT modules on downstream tasks, which consist of a Point-prior Prompt and a Geometry-aware Adapter. The Point-prior Prompt adopts a set of learnable prompt tokens, for which we propose to construct a memory bank with domain-specific knowledge, and utilize a parameter-free attention to enhance the prompt tokens. The Geometry-aware Adapter aims to aggregate point cloud features within spatial neighborhoods to capture fine-grained geometric information through local interactions. Extensive experiments indicate that our Point-PEFT can achieve better performance than the full fine-tuning on various downstream tasks, while using only 5% of the trainable parameters, demonstrating the efficiency and effectiveness of our approach. Code is released at https://github.com/Ivan-Tang-3D/Point-PEFT.

----

## [575] Context-I2W: Mapping Images to Context-Dependent Words for Accurate Zero-Shot Composed Image Retrieval

**Authors**: *Yuanmin Tang, Jing Yu, Keke Gai, Jiamin Zhuang, Gang Xiong, Yue Hu, Qi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28324](https://doi.org/10.1609/aaai.v38i6.28324)

**Abstract**:

Different from the Composed Image Retrieval task that requires expensive labels for training task-specific models, Zero-Shot Composed Image Retrieval (ZS-CIR) involves diverse tasks with a broad range of visual content manipulation intent that could be related to domain, scene, object, and attribute. The key challenge for ZS-CIR tasks is to learn a more accurate image representation that has adaptive attention to the reference image for various manipulation descriptions. In this paper, we propose a novel context-dependent mapping network, named Context-I2W,  for adaptively converting description-relevant Image information into a pseudo-word token composed of the description for accurate ZS-CIR. Specifically, an Intent View Selector first dynamically learns a rotation rule to map the identical image to a task-specific manipulation view. Then a Visual Target Extractor further captures local information covering the main targets in ZS-CIR tasks under the guidance of multiple learnable queries. The two complementary modules work together to map an image to a context-dependent pseudo-word token without extra supervision. Our model shows strong generalization ability on four ZS-CIR tasks, including domain conversion, object composition, object manipulation, and attribute manipulation. It obtains consistent and significant performance boosts ranging from 1.88% to 3.60% over the best methods and achieves new state-of-the-art results on ZS-CIR. Our code is available at https://anonymous.4open.science/r/Context-I2W-4224/.

----

## [576] Generative-Based Fusion Mechanism for Multi-Modal Tracking

**Authors**: *Zhangyong Tang, Tianyang Xu, Xiaojun Wu, Xuefeng Zhu, Josef Kittler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28325](https://doi.org/10.1609/aaai.v38i6.28325)

**Abstract**:

Generative models (GMs) have received increasing research interest for their remarkable capacity to achieve comprehensive understanding. However, their potential application in the domain of multi-modal tracking has remained unexplored. In this context, we seek to uncover the potential of harnessing generative techniques to address the critical challenge, information fusion, in multi-modal tracking. In this paper, we delve into two prominent GM techniques, namely, Conditional Generative Adversarial Networks (CGANs) and Diffusion Models (DMs). Different from the standard fusion process where the features from each modality are directly fed into the fusion block, we combine these multi-modal features with random noise in the GM framework, effectively transforming the original training samples into harder instances. This design excels at extracting discriminative clues from the features, enhancing the ultimate tracking performance. Based on this, we conduct extensive experiments across two multi-modal tracking tasks, three baseline methods, and four challenging benchmarks. The experimental results demonstrate that the proposed generative-based fusion mechanism achieves state-of-the-art performance by setting new records on GTOT, LasHeR and RGBD1K. Code will be available at https://github.com/Zhangyong-Tang/GMMT.

----

## [577] Shadow Generation with Decomposed Mask Prediction and Attentive Shadow Filling

**Authors**: *Xinhao Tao, Junyan Cao, Yan Hong, Li Niu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28326](https://doi.org/10.1609/aaai.v38i6.28326)

**Abstract**:

Image composition refers to inserting a foreground object into a background image to obtain a composite image. In this work, we focus on generating plausible shadows for the inserted foreground object to make the composite image more realistic. To supplement the existing small-scale dataset, we create a large-scale dataset called RdSOBA with rendering techniques. Moreover, we design a two-stage network named DMASNet with decomposed mask prediction and attentive shadow filling. Specifically, in the first stage, we decompose shadow mask prediction into box prediction and shape prediction. In the second stage, we attend to reference background shadow pixels to fill the foreground shadow. Abundant experiments prove that our DMASNet achieves better visual effects and generalizes well to real composite images.

----

## [578] Towards Efficient and Effective Text-to-Video Retrieval with Coarse-to-Fine Visual Representation Learning

**Authors**: *Kaibin Tian, Yanhua Cheng, Yi Liu, Xinglin Hou, Quan Chen, Han Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28327](https://doi.org/10.1609/aaai.v38i6.28327)

**Abstract**:

In recent years, text-to-video retrieval methods based on CLIP have experienced rapid development. The primary direction of evolution is to exploit the much wider gamut of visual and textual cues to achieve alignment. Concretely, those methods with impressive performance often design a heavy fusion block for sentence (words)-video (frames) interaction, regardless of the prohibitive computation complexity. Nevertheless, these approaches are not optimal in terms of feature utilization and retrieval efficiency. To address this issue, we adopt multi-granularity visual feature learning, ensuring the model's comprehensiveness in capturing visual content features spanning from abstract to detailed levels during the training phase. To better leverage the multi-granularity features, we devise a two-stage retrieval architecture in the retrieval phase. This solution ingeniously balances the coarse and fine granularity of retrieval content. Moreover, it also strikes a harmonious equilibrium between retrieval effectiveness and efficiency. Specifically, in training phase, we design a parameter-free text-gated interaction block (TIB) for fine-grained video representation learning and embed an extra Pearson Constraint to optimize cross-modal representation learning. In retrieval phase, we use coarse-grained video representations for fast recall of top-k candidates, which are then reranked by fine-grained video representations. Extensive experiments on four benchmarks demonstrate the efficiency and effectiveness. Notably, our method achieves comparable performance with the current state-of-the-art methods while being nearly 50 times faster.

----

## [579] Open-Vocabulary Video Relation Extraction

**Authors**: *Wentao Tian, Zheng Wang, Yuqian Fu, Jingjing Chen, Lechao Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28328](https://doi.org/10.1609/aaai.v38i6.28328)

**Abstract**:

A comprehensive understanding of videos is inseparable from describing the action with its contextual action-object interactions. However, many current video understanding tasks prioritize general action classification and overlook the actors and relationships that shape the nature of the action, resulting in a superficial understanding of the action. 
Motivated by this, we introduce Open-vocabulary Video Relation Extraction (OVRE), a novel task that views action understanding through the lens of action-centric relation triplets. OVRE focuses on pairwise relations that take part in the action and describes these relation triplets with natural languages. Moreover, we curate the Moments-OVRE dataset, which comprises 180K videos with action-centric relation triplets, sourced from a multi-label action classification dataset. With Moments-OVRE, we further propose a cross-modal mapping model to generate relation triplets as a sequence. Finally, we benchmark existing cross-modal generation models on the new task of OVRE. Our code and dataset are available at https://github.com/Iriya99/OVRE.

----

## [580] Divide and Conquer: Hybrid Pre-training for Person Search

**Authors**: *Yanling Tian, Di Chen, Yunan Liu, Jian Yang, Shanshan Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28329](https://doi.org/10.1609/aaai.v38i6.28329)

**Abstract**:

Large-scale pre-training has proven to be an effective method for improving performance across different tasks. Current person search methods use ImageNet pre-trained models for feature extraction, yet it is not an optimal solution due to the gap between the pre-training task and person search task (as a downstream task). Therefore, in this paper, we focus on pre-training for person search, which involves detecting and re-identifying individuals simultaneously. Although labeled data for person search is scarce, datasets for two sub-tasks person detection and re-identification are relatively abundant. To this end, we propose a hybrid pre-training framework specifically designed for person search using sub-task data only. It consists of a hybrid learning paradigm that handles data with different kinds of supervisions, and an intra-task alignment module that alleviates domain discrepancy under limited resources. To the best of our knowledge, this is the first work that investigates how to support full-task pre-training using sub-task data. Extensive experiments demonstrate that our pre-trained model can achieve significant improvements across diverse protocols, such as person search method, fine-tuning data, pre-training data and model backbone. For example, our model improves ResNet50 based NAE by 10.3% relative improvement w.r.t. mAP. Our code and pre-trained models are released for plug-and-play usage to the person search community (https://github.com/personsearch/PretrainPS).

----

## [581] Taxonomy Driven Fast Adversarial Training

**Authors**: *Kun Tong, Chengze Jiang, Jie Gui, Yuan Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28330](https://doi.org/10.1609/aaai.v38i6.28330)

**Abstract**:

Adversarial training (AT) is an effective defense method against gradient-based attacks to enhance the robustness of neural networks. Among them, single-step AT has emerged as a hotspot topic due to its simplicity and efficiency, requiring only one gradient propagation in generating adversarial examples. Nonetheless, the problem of catastrophic overfitting (CO) that causes training collapse remains poorly understood, and there exists a gap between the robust accuracy achieved through single- and multi-step AT. In this paper, we present a surprising finding that the taxonomy of adversarial examples reveals the truth of CO. Based on this conclusion, we propose taxonomy driven fast adversarial training (TDAT) which jointly optimizes learning objective, loss function, and initialization method, thereby can be regarded as a new paradigm of single-step AT. Compared with other fast AT methods, TDAT can boost the robustness of neural networks, alleviate the influence of misclassified examples, and prevent CO during the training process while requiring almost no additional computational and memory resources. Our method achieves robust accuracy improvement of 1.59%, 1.62%, 0.71%, and 1.26% on CIFAR-10, CIFAR-100, Tiny ImageNet, and ImageNet-100 datasets, when against projected gradient descent PGD10 attack with perturbation budget 8/255. Furthermore, our proposed method also achieves state-of-the-art robust accuracy against other attacks. Code is available at https://github.com/bookman233/TDAT.

----

## [582] End-to-End Real-Time Vanishing Point Detection with Transformer

**Authors**: *Xin Tong, Shi Peng, Yufei Guo, Xuhui Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28331](https://doi.org/10.1609/aaai.v38i6.28331)

**Abstract**:

In this paper, we propose a novel transformer-based end-to-end real-time vanishing point detection method, which is named Vanishing Point TRansformer (VPTR). The proposed method can directly regress the locations of vanishing points from given images. To achieve this goal, we pose vanishing point detection as a point object detection task on the Gaussian hemisphere with region division. Considering low-level features always provide more geometric information which can contribute to accurate vanishing point prediction, we propose a clear architecture where vanishing point queries in the decoder can directly gather multi-level features from CNN backbone with deformable attention in VPTR. Our method does not rely on line detection or Manhattan world assumption, which makes it more flexible to use. VPTR runs at an inferring speed of 140 FPS on one NVIDIA 3090 card. Experimental results on synthetic and real-world datasets demonstrate that our method can be used in both natural and structural scenes, and is superior to other state-of-the-art methods on the balance of accuracy and efficiency.

----

## [583] Discrete Cycle-Consistency Based Unsupervised Deep Graph Matching

**Authors**: *Siddharth Tourani, Muhammad Haris Khan, Carsten Rother, Bogdan Savchynskyy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28332](https://doi.org/10.1609/aaai.v38i6.28332)

**Abstract**:

We contribute to the sparsely populated area of unsupervised deep graph matching with application to keypoint matching in images. Contrary to the standard supervised approach, our method does not require ground truth correspondences between keypoint pairs. Instead, it is self-supervised by enforcing consistency of matchings between images of the same object category. As the matching and the consistency loss are discrete, their derivatives cannot be straightforwardly used for learning. We address this issue in a principled way by building our method upon the recent results on black-box differentiation of combinatorial solvers. This makes our method exceptionally flexible, as it is compatible with arbitrary network architectures and combinatorial solvers. Our experimental evaluation suggests that our technique sets a new state-of-the-art for unsupervised graph matching.

----

## [584] A Unified Masked Autoencoder with Patchified Skeletons for Motion Synthesis

**Authors**: *Esteve Valls Mascaro, Hyemin Ahn, Dongheui Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28333](https://doi.org/10.1609/aaai.v38i6.28333)

**Abstract**:

The synthesis of human motion has traditionally been addressed through task-dependent models that focus on specific challenges, such as predicting future motions or filling in intermediate poses conditioned on known key-poses. In this paper, we present a novel task-independent model called UNIMASK-M, which can effectively address these challenges using a unified architecture. Our model obtains comparable or better performance than the state-of-the-art in each field. Inspired by Vision Transformers (ViTs), our UNIMASK-M model decomposes a human pose into body parts to leverage the spatio-temporal relationships existing in human motion. Moreover, we reformulate various pose-conditioned motion synthesis tasks as a reconstruction problem with different masking patterns given as input. By explicitly informing our model about the masked joints, our UNIMASK-M becomes more robust to occlusions. Experimental results show that our model successfully forecasts human motion on the Human3.6M dataset while achieving state-of-the-art results in motion inbetweening on the LaFAN1 dataset for long transition periods.

----

## [585] CoVR: Learning Composed Video Retrieval from Web Video Captions

**Authors**: *Lucas Ventura, Antoine Yang, Cordelia Schmid, Gül Varol*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28334](https://doi.org/10.1609/aaai.v38i6.28334)

**Abstract**:

Composed Image Retrieval (CoIR) has recently gained popularity as a task that considers both text and image queries together, to search for relevant images in a database. Most CoIR approaches require manually annotated datasets, comprising image-text-image triplets, where the text describes a modification from the query image to the target image. However, manual curation of CoIR triplets is expensive and prevents scalability. In this work, we instead propose a scalable automatic dataset creation methodology that generates triplets given video-caption pairs, while also expanding the scope of the task to include composed video retrieval (CoVR). To this end, we mine paired videos with a similar caption from a large database, and leverage a large language model to generate the corresponding modification text. Applying this methodology to the extensive WebVid2M collection, we automatically construct our WebVid-CoVR dataset, resulting in 1.6 million triplets. Moreover, we introduce a new benchmark for CoVR with a manually annotated evaluation set, along with baseline results. Our experiments further demonstrate that training a CoVR model on our dataset effectively transfers to CoIR, leading to improved state-of-the-art performance in the zero-shot setup on both the CIRR and FashionIQ benchmarks. Our code, datasets, and models are publicly available at https://imagine.enpc.fr/~ventural/covr.

----

## [586] Supervision Interpolation via LossMix: Generalizing Mixup for Object Detection and Beyond

**Authors**: *Thanh Vu, Baochen Sun, Bodi Yuan, Alex Ngai, Yueqi Li, Jan-Michael Frahm*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28335](https://doi.org/10.1609/aaai.v38i6.28335)

**Abstract**:

The success of data mixing augmentations in image classification tasks has been well-received. However, these techniques cannot be readily applied to object detection due to challenges such as spatial misalignment, foreground/background distinction, and plurality of instances. To tackle these issues, we first introduce a novel conceptual framework called Supervision Interpolation (SI), which offers a fresh perspective on interpolation-based augmentations by relaxing and generalizing Mixup. Based on SI, we propose LossMix, a simple yet versatile and effective regularization that enhances the performance and robustness of object detectors and more. Our key insight is that we can effectively regularize the training on mixed data by interpolating their loss errors instead of ground truth labels. Empirical results on the PASCAL VOC and MS COCO datasets demonstrate that LossMix can consistently outperform state-of-the-art methods widely adopted for detection. Furthermore, by jointly leveraging LossMix with unsupervised domain adaptation, we successfully improve existing approaches and set a new state of the art for cross-domain object detection.

----

## [587] Integrated Decision Gradients: Compute Your Attributions Where the Model Makes Its Decision

**Authors**: *Chase Walker, Sumit Jha, Kenny Chen, Rickard Ewetz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28336](https://doi.org/10.1609/aaai.v38i6.28336)

**Abstract**:

Attribution algorithms are frequently employed to explain the decisions of neural network models. Integrated Gradients (IG) is an influential attribution method due to its strong axiomatic foundation. The algorithm is based on integrating the gradients along a path from a reference image to the input image. Unfortunately, it can be observed that gradients computed from regions where the output logit changes minimally along the path provide poor explanations for the model decision, which is called the saturation effect problem. In this paper, we propose an attribution algorithm called integrated decision gradients (IDG). The algorithm focuses on integrating gradients from the region of the path where the model makes its decision, i.e., the portion of the path where the output logit rapidly transitions from zero to its final value. This is practically realized by scaling each gradient by the derivative of the output logit with respect to the path. The algorithm thereby provides a principled solution to the saturation problem. Additionally, we minimize the errors within the Riemann sum approximation of the path integral by utilizing non-uniform subdivisions determined by adaptive sampling. In the evaluation on ImageNet, it is demonstrated that IDG outperforms IG, Left-IG, Guided IG, and adversarial gradient integration both qualitatively and quantitatively using standard insertion and deletion metrics across three common models.

----

## [588] HISR: Hybrid Implicit Surface Representation for Photorealistic 3D Human Reconstruction

**Authors**: *Angtian Wang, Yuanlu Xu, Nikolaos Sarafianos, Robert Maier, Edmond Boyer, Alan L. Yuille, Tony Tung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28337](https://doi.org/10.1609/aaai.v38i6.28337)

**Abstract**:

Neural reconstruction and rendering strategies have demonstrated state-of-the-art performances due, in part, to their ability to preserve high level shape details. Existing approaches, however, either represent objects as implicit surface functions or neural volumes and still struggle to recover shapes with heterogeneous materials, in particular human skin, hair or clothes. To this aim, we present a new hybrid implicit surface representation to model human shapes. This representation is composed of two surface layers that represent opaque and translucent regions on the clothed human body. We segment different regions automatically using visual cues and learn to reconstruct two signed distance functions (SDFs). We perform surface-based rendering on opaque regions (e.g., body, face, clothes) to preserve high-fidelity surface normals and volume rendering on translucent regions (e.g., hair). Experiments demonstrate that our approach obtains state-of-the-art results on 3D human reconstructions, and also shows competitive performances on other objects.

----

## [589] VIGC: Visual Instruction Generation and Correction

**Authors**: *Bin Wang, Fan Wu, Xiao Han, Jiahui Peng, Huaping Zhong, Pan Zhang, Xiaoyi Dong, Weijia Li, Wei Li, Jiaqi Wang, Conghui He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28338](https://doi.org/10.1609/aaai.v38i6.28338)

**Abstract**:

The integration of visual encoders and large language models (LLMs) has driven recent progress in multimodal large language models (MLLMs). However, the scarcity of high-quality instruction-tuning data for vision-language tasks remains a challenge. The current leading paradigm, such as LLaVA, relies on language-only GPT-4 to generate data, which requires pre-annotated image captions and detection bounding boxes, suffering from understanding image details. A practical solution to this problem would be to utilize the available multimodal large language models to generate instruction data for vision-language tasks. However, it's worth noting that the currently accessible MLLMs are not as powerful as their LLM counterparts, as they tend to produce inadequate responses and generate false information. As a solution for addressing the current issue, this paper proposes the Visual Instruction Generation and Correction (VIGC) framework that enables multimodal large language models to generate instruction-tuning data and progressively enhance its quality on-the-fly. Specifically, Visual Instruction Generation (VIG) guides the vision-language model to generate diverse instruction-tuning data. To ensure generation quality, Visual Instruction Correction (VIC) adopts an iterative update mechanism to correct any inaccuracies in data produced by VIG, effectively reducing the risk of hallucination. Leveraging the diverse, high-quality data generated by VIGC, we finetune mainstream models and validate data quality based on various evaluations. Experimental results demonstrate that VIGC not only compensates for the shortcomings of language-only data generation methods, but also effectively enhances the benchmark performance. The models, datasets, and code are available at https://opendatalab.github.io/VIGC

----

## [590] Low-Light Face Super-resolution via Illumination, Structure, and Texture Associated Representation

**Authors**: *Chenyang Wang, Junjun Jiang, Kui Jiang, Xianming Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28339](https://doi.org/10.1609/aaai.v38i6.28339)

**Abstract**:

Human face captured at night or in dimly lit environments has become a common practice, accompanied by complex low-light and low-resolution degradations. However, the existing face super-resolution (FSR) technologies and derived cascaded schemes are inadequate to recover credible textures. In this paper, we propose a novel approach that decomposes the restoration task into face structural fidelity maintaining and texture consistency learning. The former aims to enhance the quality of face images while improving the structural fidelity, while the latter focuses on eliminating perturbations and artifacts caused by low-light degradation and reconstruction. Based on this, we develop a novel low-light low-resolution face super-resolution framework. Our method consists of two steps: an illumination correction face super-resolution network (IC-FSRNet) for lighting the face and recovering the structural information, and a detail enhancement model (DENet) for improving facial details, thus making them more visually appealing and easier to analyze. As the relighted regions could provide complementary information to boost face super-resolution and vice versa, we introduce the mutual learning to harness the informative components from relighted regions and reconstruction, and achieve the iterative refinement. In addition, DENet equipped with diffusion probabilistic model is built to further improve face image visual quality. Experiments demonstrate that the proposed joint optimization framework achieves significant improvements in reconstruction quality and perceptual quality over existing two-stage sequential solutions. Code is available at https://github.com/wcy-cs/IC-FSRDENet.

----

## [591] SelfPromer: Self-Prompt Dehazing Transformers with Depth-Consistency

**Authors**: *Cong Wang, Jinshan Pan, Wanyu Lin, Jiangxin Dong, Wei Wang, Xiao-Ming Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28340](https://doi.org/10.1609/aaai.v38i6.28340)

**Abstract**:

This work presents an effective depth-consistency Self-Prompt Transformer, terms as SelfPromer, for image dehazing. It is motivated by an observation that the estimated depths of an image with haze residuals and its clear counterpart vary. Enforcing the depth consistency of dehazed images with clear ones, therefore, is essential for dehazing. For this purpose, we develop a prompt based on the features of depth differences between the hazy input images and corresponding clear counterparts that can guide dehazing models for better restoration. Specifically, we first apply deep features extracted from the input images to the depth difference features for generating the prompt that contains the haze residual information in the input. Then we propose a prompt embedding module that is designed to perceive the haze residuals, by linearly adding the prompt to the deep features. Further, we develop an effective prompt attention module to pay more attention to haze residuals for better removal. By incorporating the prompt, prompt embedding, and prompt attention into an encoder-decoder network based on VQGAN, we can achieve better perception quality. As the depths of clear images are not available at inference, and the dehazed images with one-time feed-forward execution may still contain a portion of haze residuals, we propose a new continuous self-prompt inference that can iteratively correct the dehazing model towards better haze-free image generation. Extensive experiments show that our SelfPromer performs favorably against the state-of-the-art approaches on both synthetic and real-world datasets in terms of perception metrics including NIQE, PI, and PIQE. The source codes will be made available at https://github.com/supersupercong/SelfPromer.

----

## [592] Correlation Matching Transformation Transformers for UHD Image Restoration

**Authors**: *Cong Wang, Jinshan Pan, Wei Wang, Gang Fu, Siyuan Liang, Mengzhu Wang, Xiao-Ming Wu, Jun Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28341](https://doi.org/10.1609/aaai.v38i6.28341)

**Abstract**:

This paper proposes UHDformer, a general Transformer for Ultra-High-Definition (UHD) image restoration. UHDformer contains two learning spaces: (a) learning in high-resolution space and (b) learning in low-resolution space. The former learns multi-level high-resolution features and fuses low-high features and reconstructs the residual images, while the latter explores more representative features learning from the high-resolution ones to facilitate better restoration. To better improve feature representation in low-resolution space, we propose to build feature transformation from the high-resolution space to the low-resolution one. To that end, we propose two new modules: Dual-path Correlation Matching Transformation module (DualCMT) and Adaptive Channel Modulator (ACM). The DualCMT selects top C/r (r is greater or equal to 1 which controls the squeezing level) correlation channels from the max-pooling/mean-pooling high-resolution features to replace low-resolution ones in Transformers, which can effectively squeeze useless content to improve the feature representation in low-resolution space to facilitate better recovery. The ACM is exploited to adaptively modulate multi-level high-resolution features, enabling to provide more useful features to low-resolution space for better learning. Experimental results show that our UHDformer reduces about ninety-seven percent model sizes compared with most state-of-the-art methods while significantly improving performance under different training sets on 3 UHD image restoration tasks, including low-light image enhancement, image dehazing, and image deblurring. The source codes will be made available at https://github.com/supersupercong/UHDformer.

----

## [593] EulerMormer: Robust Eulerian Motion Magnification via Dynamic Filtering within Transformer

**Authors**: *Fei Wang, Dan Guo, Kun Li, Meng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28342](https://doi.org/10.1609/aaai.v38i6.28342)

**Abstract**:

Video Motion Magnification (VMM) aims to break the resolution limit of human visual perception capability and reveal the imperceptible minor motion that contains valuable information in the macroscopic domain. However, challenges arise in this task due to photon noise inevitably introduced by photographic devices and spatial inconsistency in amplification, leading to flickering artifacts in static fields and motion blur and distortion in dynamic fields in the video. Existing methods focus on explicit motion modeling without emphasizing prioritized denoising during the motion magnification process. This paper proposes a novel dynamic filtering strategy to achieve static-dynamic field adaptive denoising. Specifically, based on Eulerian theory, we separate texture and shape to extract motion representation through inter-frame shape differences, expecting to leverage these subdivided features to solve this task finely. Then, we introduce a novel dynamic filter that eliminates noise cues and preserves critical features in the motion magnification and amplification generation phases. Overall, our unified framework, EulerMormer, is a pioneering effort to first equip with Transformer in learning-based VMM. The core of the dynamic filter lies in a global dynamic sparse cross-covariance attention mechanism that explicitly removes noise while preserving vital information, coupled with a multi-scale dual-path gating mechanism that selectively regulates the dependence on different frequency features to reduce spatial attenuation and complement motion boundaries. We demonstrate extensive experiments that EulerMormer achieves more robust video motion magnification from the Eulerian perspective, significantly outperforming state-of-the-art methods. The source code is available at https://github.com/VUT-HFUT/EulerMormer.

----

## [594] Learning to Learn Better Visual Prompts

**Authors**: *Fengxiang Wang, Wanrong Huang, Shaowu Yang, Qi Fan, Long Lan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28343](https://doi.org/10.1609/aaai.v38i6.28343)

**Abstract**:

Prompt tuning provides a low-cost way of adapting vision-language models (VLMs) for various downstream vision tasks without requiring updating the huge pre-trained parameters. Dispensing with the conventional manual crafting of prompts, the recent prompt tuning method of Context Optimization (CoOp) introduces adaptable vectors as text prompts. Nevertheless, several previous works point out that the CoOp-based approaches are easy to overfit to the base classes and hard to generalize to novel classes. In this paper, we reckon that the prompt tuning works well only in the base classes because of the limited capacity of the adaptable vectors. The scale of the pre-trained model is hundreds times the scale of the adaptable vector, thus the learned vector has a very limited ability to absorb the knowledge of novel classes. To minimize this excessive overfitting of textual knowledge on the base class, we view prompt tuning as learning to learn (LoL) and learn the prompt in the way of meta-learning, the training manner of dividing the base classes into many different subclasses could fully exert the limited capacity of prompt tuning and thus transfer it power to recognize the novel classes.  To be specific, we initially perform fine-tuning on the base class based on the CoOp method for pre-trained CLIP. Subsequently, predicated on the fine-tuned CLIP model, we carry out further fine-tuning in an N-way K-shot manner from the perspective of meta-learning on the base classes. We finally apply the learned textual vector and VLM for unseen classes.Extensive experiments on benchmark datasets validate the efficacy of our meta-learning-informed prompt tuning, affirming its role as a robust optimization strategy for VLMs.

----

## [595] MuST: Robust Image Watermarking for Multi-Source Tracing

**Authors**: *Guanjie Wang, Zehua Ma, Chang Liu, Xi Yang, Han Fang, Weiming Zhang, Nenghai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28344](https://doi.org/10.1609/aaai.v38i6.28344)

**Abstract**:

In recent years, with the popularity of social media applications, massive digital images are available online, which brings great convenience to image recreation. However, the use of unauthorized image materials in multi-source composite images is still inadequately regulated, which may cause significant loss and discouragement to the copyright owners of the source image materials. Ideally, deep watermarking techniques could provide a solution for protecting these copyrights based on their encoder-noise-decoder training strategy. Yet existing image watermarking schemes, which are mostly designed for single images, cannot well address the copyright protection requirements in this scenario, since the multi-source image composing process commonly includes distortions that are not well investigated in previous methods, e.g., the extreme downsizing.
To meet such demands, we propose MuST, a multi-source tracing robust watermarking scheme, whose architecture includes a multi-source image detector and minimum external rectangle operation for multiple watermark resynchronization and extraction. Furthermore, we constructed an image material dataset covering common image categories and designed the simulation model of the multi-source image composing process as the noise layer. Experiments demonstrate the excellent performance of MuST in tracing sources of image materials from the composite images compared with SOTA watermarking methods, which could maintain the extraction accuracy above 98% to trace the sources of at least 3 different image materials while keeping the average PSNR of watermarked image materials higher than 42.51 dB. We released our code on https://github.com/MrCrims/MuST

----

## [596] LION: Implicit Vision Prompt Tuning

**Authors**: *Haixin Wang, Jianlong Chang, Yihang Zhai, Xiao Luo, Jinan Sun, Zhouchen Lin, Qi Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28345](https://doi.org/10.1609/aaai.v38i6.28345)

**Abstract**:

Despite recent promising performances across a range of vision tasks, vision Transformers still have an issue of high computational costs.
Recently, vision prompt learning has provided an economical solution to this problem without fine-tuning the whole large-scale model. 
However, the efficiency and effectiveness of existing models are still far from satisfactory due to the parameter cost of extensive prompt blocks and tricky prompt framework designs. 
In this paper, we propose a light-weight prompt framework named impLicit vIsion prOmpt tuNing (LION), which is motivated by deep implicit models with stable low memory costs for various complex tasks.
In particular, we merely insect two equilibrium implicit layers in two ends of the pre-trained backbone with parameters frozen. Moreover, according to the lottery hypothesis, we further prune the parameters to relieve the computation burden in implicit layers. Various experiments have validated that our LION obtains promising performances on a wide range of datasets. Most importantly, LION reduces up to 11.5 % of training parameter numbers while obtaining higher performance than the state-of-the-art VPT, especially under challenging scenes. Furthermore, we find that our proposed LION has an excellent generalization performance, making it an easy way to boost transfer learning in the future.

----

## [597] B-spine: Learning B-spline Curve Representation for Robust and Interpretable Spinal Curvature Estimation

**Authors**: *Hao Wang, Qiang Song, Ruofeng Yin, Rui Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28346](https://doi.org/10.1609/aaai.v38i6.28346)

**Abstract**:

Spinal curvature estimation is important to the diagnosis and treatment of the scoliosis. Existing methods face several issues such as the need of expensive annotations on the vertebral landmarks and being sensitive to the image quality. It is challenging to achieve robust estimation and obtain interpretable results, especially for low-quality images which are blurry and hazy. In this paper, we propose B-Spine, a novel deep learning pipeline to learn B-spline curve representation of the spine and estimate the Cobb angles for spinal curvature estimation from low-quality X-ray images. Given a low quality input, a novel SegRefine network which employs the unpaired image-to-image translation is proposed to generate a high quality spine mask from the initial segmentation result. Next, a novel mask-based B-spline prediction model is proposed to predict the B-spline curve for the spine centerline. Finally, the Cobb angles are estimated by a hybrid approach which combines the curve slope analysis and a curve based regression model. We conduct quantitative and qualitative comparisons with the representative and SOTA learning-based methods on the public AASCE2019 dataset and our new proposed JLU-CJUH dataset which contains more challenging low-quality images. The superior performance on both datasets shows our method can achieve both robustness and interpretability for spinal curvature estimation.

----

## [598] ViLT-CLIP: Video and Language Tuning CLIP with Multimodal Prompt Learning and Scenario-Guided Optimization

**Authors**: *Hao Wang, Fang Liu, Licheng Jiao, Jiahao Wang, Zehua Hao, Shuo Li, Lingling Li, Puhua Chen, Xu Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28347](https://doi.org/10.1609/aaai.v38i6.28347)

**Abstract**:

Pre-trained vision-language(V-L) models such as CLIP have demonstrated impressive Zero-Shot performance in many downstream tasks. Since adopting contrastive video-text pairs methods like CLIP to video tasks is limited by its high cost and scale, recent approaches focus on efficiently transferring the image-based CLIP to the video domain. A major finding is that fine-tuning the pre-trained model to achieve strong fully supervised performance leads to low zero shot, few shot, and base to novel generalization. Instead, freezing the backbone network to maintain generalization ability weakens fully supervised performance. Otherwise, no single prompt tuning branch consistently performs optimally. In this work, we proposed a multimodal prompt learning scheme that balances supervised and generalized performance. Our prompting approach contains three sections: 1) Independent prompt on both the vision and text branches to learn the language and visual contexts. 2) Inter-modal prompt mapping to ensure mutual synergy. 3) Reducing the discrepancy between the hand-crafted prompt (a video of a person doing [CLS]) and the learnable prompt, to alleviate the forgetting about essential video scenarios. Extensive validation of fully supervised, zero-shot, few-shot, base-to-novel generalization settings for video recognition indicates that the proposed approach achieves competitive performance with less commute cost.

----

## [599] Triple Feature Disentanglement for One-Stage Adaptive Object Detection

**Authors**: *Haoan Wang, Shilong Jia, Tieyong Zeng, Guixu Zhang, Zhi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28348](https://doi.org/10.1609/aaai.v38i6.28348)

**Abstract**:

In recent advancements concerning Domain Adaptive Object Detection (DAOD), unsupervised domain adaptation techniques have proven instrumental. These methods enable enhanced detection capabilities within unlabeled target domains by mitigating distribution differences between source and target domains. A subset of DAOD methods employs disentangled learning to segregate Domain-Specific Representations (DSR) and Domain-Invariant Representations (DIR), with ultimate predictions relying on the latter. Current practices in disentanglement, however, often lead to DIR containing residual domain-specific information. To address this, we introduce the Multi-level Disentanglement Module (MDM) that progressively disentangles DIR, enhancing comprehensive disentanglement. Additionally, our proposed Cyclic Disentanglement Module (CDM) facilitates DSR separation. To refine the process further, we employ the Categorical Features Disentanglement Module (CFDM) to isolate DIR and DSR, coupled with category alignment across scales for improved source-target domain alignment. Given its practical suitability, our model is constructed upon the foundational framework of the Single Shot MultiBox Detector (SSD), which is a one-stage object detection approach. Experimental validation highlights the effectiveness of our method, demonstrating its state-of-the-art performance across three benchmark datasets.

----



[Go to the previous page](AAAI-2024-list02.md)

[Go to the next page](AAAI-2024-list04.md)

[Go to the catalog section](README.md)