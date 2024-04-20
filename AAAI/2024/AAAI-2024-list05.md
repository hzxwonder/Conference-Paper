## [800] Deep Semantic Graph Transformer for Multi-View 3D Human Pose Estimation

**Authors**: *Lijun Zhang, Kangkang Zhou, Feng Lu, Xiang-Dong Zhou, Yu Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28549](https://doi.org/10.1609/aaai.v38i7.28549)

**Abstract**:

Most Graph Convolutional Networks based 3D human pose estimation (HPE) methods were involved in single-view 3D HPE and utilized certain spatial graphs, existing key problems such as depth ambiguity, insufficient feature representation, or limited receptive fields. To address these issues, we propose a multi-view 3D HPE framework based on deep semantic graph transformer, which adaptively learns and fuses multi-view significant semantic features of human nodes to improve 3D HPE performance. First, we propose a deep semantic graph transformer encoder to enrich spatial feature information. It deeply mines the position, spatial structure, and skeletal edge knowledge of joints and dynamically learns their correlations. Then, we build a progressive multi-view spatial-temporal feature fusion framework to mitigate joint depth uncertainty. To enhance the pose spatial representation, deep spatial semantic feature are interacted and fused across different viewpoints during monocular feature extraction. Furthermore, long-time relevant temporal dependencies are modeled and spatial-temporal information from all viewpoints is fused to intermediately supervise the depth. Extensive experiments on three 3D HPE benchmarks show that our method achieves state-of-the-art results. It can effectively enhance pose features, mitigate depth ambiguity in single-view 3D HPE, and improve 3D HPE performance without providing camera parameters. Codes and models are available at https://github.com/z0911k/SGraFormer.

----

## [801] Brush Your Text: Synthesize Any Scene Text on Images via Diffusion Model

**Authors**: *Lingjun Zhang, Xinyuan Chen, Yaohui Wang, Yue Lu, Yu Qiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28550](https://doi.org/10.1609/aaai.v38i7.28550)

**Abstract**:

Recently, diffusion-based image generation methods are credited for their remarkable text-to-image generation capabilities, while still facing challenges in accurately generating multilingual scene text images. To tackle this problem, we propose Diff-Text, which is a training-free scene text generation framework for any language. Our model outputs a photo-realistic image given a text of any language along with a textual description of a scene. The model leverages rendered sketch images as priors, thus arousing the potential multilingual-generation ability of the pre-trained Stable Diffusion. Based on the observation from the influence of the cross-attention map on object placement in generated images, we propose a localized attention constraint into the cross-attention layer to address the unreasonable positioning problem of scene text. Additionally, we introduce contrastive image-level prompts to further refine the position of the textual region and achieve more accurate scene text generation. Experiments demonstrate that our method outperforms the existing method in both the accuracy of text recognition and the naturalness of foreground-background blending.

----

## [802] IRPruneDet: Efficient Infrared Small Target Detection via Wavelet Structure-Regularized Soft Channel Pruning

**Authors**: *Mingjin Zhang, Handi Yang, Jie Guo, Yunsong Li, Xinbo Gao, Jing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28551](https://doi.org/10.1609/aaai.v38i7.28551)

**Abstract**:

Infrared Small Target Detection (IRSTD) refers to detecting faint targets in infrared images, which has achieved notable progress with the advent of deep learning. However, the drive for improved detection accuracy has led to larger, intricate models with redundant parameters, causing storage and computation inefficiencies. In this pioneering study, we introduce the concept of utilizing network pruning to enhance the efficiency of IRSTD. Due to the challenge posed by low signal-to-noise ratios and the absence of detailed semantic information in infrared images, directly applying existing pruning techniques yields suboptimal performance. To address this, we propose a novel wavelet structure-regularized soft channel pruning method, giving rise to the efficient IRPruneDet model. Our approach involves representing the weight matrix in the wavelet domain and formulating a wavelet channel pruning strategy. We incorporate wavelet regularization to induce structural sparsity without incurring extra memory usage. Moreover, we design a soft channel reconstruction method that preserves important target information against premature pruning, thereby ensuring an optimal sparse structure while maintaining overall sparsity. Through extensive experiments on two widely-used benchmarks, our IRPruneDet method surpasses established techniques in both model complexity and accuracy. Specifically, when employing U-net as the baseline network, IRPruneDet achieves a 64.13% reduction in parameters and a 51.19% decrease in FLOPS, while improving IoU from 73.31% to 75.12% and nIoU from 70.92% to 74.30%. The code is available at https://github.com/hd0013/IRPruneDet.

----

## [803] M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis

**Authors**: *Ning Zhang, Hiuyi Cheng, Jiayu Chen, Zongyuan Jiang, Jun Huang, Yang Xue, Lianwen Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28552](https://doi.org/10.1609/aaai.v38i7.28552)

**Abstract**:

Document layout analysis is a crucial step for intelligent document understanding. However, many existing methods primarily focus on the visual aspects and overlook the textual features of documents. Although document pre-trained models utilize multi-modal features during the pre-training phase, they tend to operate as a unimodal pipeline when it comes to layout analysis tasks. Furthermore, current multi-modal methods perform worse than unimodal detectors on complex layout analysis datasets. To address these limitations, we propose an effective and pluggable multi-modal fusion approach named M2Doc, which fuses visual and textual features for better layout detection. M2Doc contains two pluggable multi-modal fusion modules, early-fusion and late-fusion, which align and fuse visual and textual features at the pixel level and block level. Benefitting from the concision and effectiveness of M2Doc, it can be easily applied to various detectors for better layout detection, including two-stage and end-to-end object detectors. Our experimental results demonstrate significant performance improvements in detectors equipped with M2Doc on datasets such as DocLayNet (+11.3 mAP) and M6Doc (+1.9 mAP). Furthermore, through the integration of the DINO detector with M2Doc, we achieve state-of-the-art results on DocLayNet (89.0 mAP), M6Doc (69.9 mAP), and PubLayNet (95.5 mAP). The code will be publicly released at https://github.com/johnning2333/M2Doc.

----

## [804] Multi-View People Detection in Large Scenes via Supervised View-Wise Contribution Weighting

**Authors**: *Qi Zhang, Yunfei Gong, Daijie Chen, Antoni B. Chan, Hui Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28553](https://doi.org/10.1609/aaai.v38i7.28553)

**Abstract**:

Recent deep learning-based multi-view people detection (MVD) methods have shown promising results on existing datasets. However, current methods are mainly trained and evaluated on small, single scenes with a limited number of multi-view frames and fixed camera views. As a result, these methods may not be practical for detecting people in larger, more complex scenes with severe occlusions and camera calibration errors. This paper focuses on improving multi-view people detection by developing a supervised view-wise contribution weighting approach that better fuses multi-camera information under large scenes. Besides, a large synthetic dataset is adopted to enhance the model's generalization ability and enable more practical evaluation and comparison. The model's performance on new testing scenes is further improved with a simple domain adaptation technique. Experimental results demonstrate the effectiveness of our approach in achieving promising cross-scene multi-view people detection performance.

----

## [805] Aligning Geometric Spatial Layout in Cross-View Geo-Localization via Feature Recombination

**Authors**: *Qingwang Zhang, Yingying Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28554](https://doi.org/10.1609/aaai.v38i7.28554)

**Abstract**:

Cross-view geo-localization holds significant potential for various applications, but drastic differences in viewpoints and visual appearances between cross-view images make this task extremely challenging. Recent works have made notable progress in cross-view geo-localization. However, existing methods either ignore the correspondence between geometric spatial layout in cross-view images or require high costs or strict constraints to achieve such alignment. In response to these challenges, we propose a Feature Recombination Module (FRM) that explicitly establishes the geometric spatial layout correspondences between two views. Unlike existing methods, FRM aligns geometric spatial layout by directly recombining features, avoiding image preprocessing, and introducing no additional computational and parameter costs. This effectively reduces ambiguities caused by geometric misalignments between ground-level and aerial-level images. Furthermore, it is not sensitive to frameworks and applies to both CNN-based and Transformer-based architectures. Additionally, as part of the training procedure, we also introduce a novel weighted (B+1)-tuple loss (WBL) as optimization objective. Compared to the widely used weighted soft margin ranking loss, this innovative loss enhances convergence speed and final performance. Based on the two core components (FRM and WBL), we develop an end-to-end network architecture (FRGeo) to address these limitations from a different perspective. Extensive experiments show that our proposed FRGeo not only achieves state-of-the-art performance on cross-view geo-localization benchmarks, including CVUSA, CVACT, and VIGOR, but also is significantly superior or competitive in terms of computational complexity and trainable parameters. Our project homepage is at https://zqwlearning.github.io/FRGeo.

----

## [806] MobileInst: Video Instance Segmentation on the Mobile

**Authors**: *Renhong Zhang, Tianheng Cheng, Shusheng Yang, Haoyi Jiang, Shuai Zhang, Jiancheng Lyu, Xin Li, Xiaowen Ying, Dashan Gao, Wenyu Liu, Xinggang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28555](https://doi.org/10.1609/aaai.v38i7.28555)

**Abstract**:

Video instance segmentation on mobile devices is an important yet very challenging edge AI problem. It mainly suffers from (1) heavy computation and memory costs for frame-by-frame pixel-level instance perception and (2) complicated heuristics for tracking objects. To address these issues, we present MobileInst, a lightweight and mobile-friendly framework for video instance segmentation on mobile devices. Firstly, MobileInst adopts a mobile vision transformer to extract multi-level semantic features and presents an efficient query-based dual-transformer instance decoder for mask kernels and a semantic-enhanced mask decoder to generate instance segmentation per frame. Secondly, MobileInst exploits simple yet effective kernel reuse and kernel association to track objects for video instance segmentation. Further, we propose temporal query passing to enhance the tracking ability for kernels. We conduct experiments on COCO and YouTube-VIS datasets to demonstrate the superiority of MobileInst and evaluate the inference latency on one single CPU core of the Snapdragon 778G Mobile Platform, without other methods of acceleration. On the COCO dataset, MobileInst achieves 31.2 mask AP and 433 ms on the mobile CPU, which reduces the latency by 50% compared to the previous SOTA. For video instance segmentation, MobileInst achieves 35.0 AP and 30.1 AP on YouTube-VIS 2019 & 2021.

----

## [807] Scalable Geometric Fracture Assembly via Co-creation Space among Assemblers

**Authors**: *Ruiyuan Zhang, Jiaxiang Liu, Zexi Li, Hao Dong, Jie Fu, Chao Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28556](https://doi.org/10.1609/aaai.v38i7.28556)

**Abstract**:

Geometric fracture assembly presents a challenging practical task in archaeology and 3D computer vision. Previous methods have focused solely on assembling fragments based on semantic information, which has limited the quantity of objects that can be effectively assembled. Therefore, there is a need to develop a scalable framework for geometric fracture assembly without relying on semantic information. To improve the effectiveness of assembling geometric fractures without semantic information, we propose a co-creation space comprising several assemblers capable of gradually and unambiguously assembling fractures. Additionally, we introduce a novel loss function, i.e., the geometric-based collision loss, to address collision issues during the fracture assembly process and enhance the results. Our framework exhibits better performance on both PartNet and Breaking Bad datasets compared to existing state-of-the-art frameworks. Extensive experiments and quantitative comparisons demonstrate the effectiveness of our proposed framework, which features linear computational complexity, enhanced abstraction, and improved generalization. Our code is publicly available at https://github.com/Ruiyuan-Zhang/CCS.

----

## [808] S3A: Towards Realistic Zero-Shot Classification via Self Structural Semantic Alignment

**Authors**: *Sheng Zhang, Muzammal Naseer, Guangyi Chen, Zhiqiang Shen, Salman H. Khan, Kun Zhang, Fahad Shahbaz Khan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28557](https://doi.org/10.1609/aaai.v38i7.28557)

**Abstract**:

Large-scale pre-trained Vision Language Models (VLMs) have proven effective for zero-shot classification. Despite the success, most traditional VLMs-based methods are restricted by the assumption of partial source supervision or ideal target vocabularies, which rarely satisfy the open-world scenario. In this paper, we aim at a more challenging setting, Realistic Zero-Shot Classification, which assumes no annotation but instead a broad vocabulary. To address the new problem, we propose the Self Structural Semantic Alignment (S3A) framework, which extracts the structural semantic information from unlabeled data while simultaneously self-learning. Our S3A framework adopts a unique Cluster-Vote-Prompt-Realign (CVPR) algorithm, which iteratively groups unlabeled data to derive structural semantics for pseudo-supervision. Our CVPR algorithm includes iterative clustering on images, voting within each cluster to identify initial class candidates from the vocabulary, generating discriminative prompts with large language models to discern confusing candidates, and realigning images and the vocabulary as structural semantic alignment. Finally, we propose to self-train the CLIP image encoder with both individual and structural semantic alignment through a teacher-student learning strategy. Our comprehensive experiments across various generic and fine-grained benchmarks demonstrate that the S3A method substantially improves over existing VLMs-based approaches, achieving a more than 15% accuracy improvement over CLIP on average. Our codes, models, and prompts are publicly released at https://github.com/sheng-eatamath/S3A.

----

## [809] A Computation-Aware Shape Loss Function for Point Cloud Completion

**Authors**: *Shunran Zhang, Xiubo Zhang, Tsz Nam Chan, Shenghui Zhang, Leong Hou U*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28558](https://doi.org/10.1609/aaai.v38i7.28558)

**Abstract**:

Learning-based point cloud completion tasks have shown potential in various critical tasks, such as object detection, assignment, and registration. However, accurately and efficiently quantifying the shape error between the predicted point clouds generated by networks and the ground truth remains challenging. While EMD-based loss functions excel in shape detail and perceived density distribution, their approach can only yield results with significant discrepancies from the actual EMD within a tolerable training time.
To address these challenges, we first propose the initial price based on the auction algorithm, reducing the number of iterations required for the algorithm while ensuring the correctness of the assignment results. We then introduce an algorithm to compute the initial price through a successive shortest path and the Euclidean information between its nodes. Finally, we adopt a series of optimization strategies to speed up the algorithm and offer an EMD approximation scheme for point cloud problems that balances time loss and computational accuracy based on point cloud data characteristics.
Our experimental results confirm that our algorithm achieves the smallest gap with the real EMD within an acceptable time range and yields the best results in end-to-end training.

----

## [810] Vision-Language Pre-training with Object Contrastive Learning for 3D Scene Understanding

**Authors**: *Taolin Zhang, Sunan He, Tao Dai, Zhi Wang, Bin Chen, Shu-Tao Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28559](https://doi.org/10.1609/aaai.v38i7.28559)

**Abstract**:

In recent years, vision language pre-training frameworks have made significant progress in natural language processing and computer vision, achieving remarkable performance improvement on various downstream tasks. However, when extended to point cloud data, existing works mainly focus on building task-specific models, and fail to extract universal 3D vision-language embedding that generalize well. We carefully investigate three common tasks in semantic 3D scene understanding, and derive key insights into the development of a pre-training model. Motivated by these observations, we propose a vision-language pre-training framework 3DVLP (3D vision-language pre-training with object contrastive learning), which transfers flexibly on 3D vision-language downstream tasks. 3DVLP takes visual grounding as the proxy task and introduces Object-level IoU-guided Detection (OID) loss to obtain high-quality proposals in the scene. Moreover, we design Object-level Cross-Contrastive alignment (OCC) task and Object-level Self-Contrastive learning (OSC) task to align the objects with descriptions and distinguish different objects in the scene, respectively. Extensive experiments verify the excellent performance of 3DVLP on three 3D vision-language tasks, reflecting its superiority in semantic 3D scene understanding. Code is available at https://github.com/iridescentttt/3DVLP.

----

## [811] Transformer-Based Selective Super-resolution for Efficient Image Refinement

**Authors**: *Tianyi Zhang, Kishore Kasichainula, Yaoxin Zhuo, Baoxin Li, Jae-Sun Seo, Yu Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28560](https://doi.org/10.1609/aaai.v38i7.28560)

**Abstract**:

Conventional super-resolution methods suffer from two drawbacks: substantial computational cost in upscaling an entire large image, and the introduction of extraneous or potentially detrimental information for downstream computer vision tasks during the refinement of the background. To solve these issues, we propose a novel transformer-based algorithm, Selective Super-Resolution (SSR), which partitions images into non-overlapping tiles, selects tiles of interest at various scales with a pyramid architecture, and exclusively reconstructs these selected tiles with deep features. Experimental results on three datasets demonstrate the efficiency and robust performance of our approach for super-resolution. Compared to the state-of-the-art methods, the FID score is reduced from 26.78 to 10.41 with 40% reduction in computation cost for the BDD100K dataset.

----

## [812] Exploring Base-Class Suppression with Prior Guidance for Bias-Free One-Shot Object Detection

**Authors**: *Wenwen Zhang, Yun Hu, Hangguan Shan, Eryun Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28561](https://doi.org/10.1609/aaai.v38i7.28561)

**Abstract**:

One-shot object detection (OSOD) aims to detect all object instances towards the given category specified by a query image. Most existing studies in OSOD endeavor to establish effective cross-image correlation with limited query information, however, ignoring the problems of the model bias towards the base classes and the generalization degradation on the novel classes. Observing this, we propose a novel algorithm, namely Base-class Suppression with Prior Guidance (BSPG) network to achieve bias-free OSOD. Specifically, the objects of base categories can be detected by a base-class predictor and eliminated by a base-class suppression module (BcS). Moreover, a prior guidance module (PG) is designed to calculate the correlation of high-level features in a non-parametric manner, producing a class-agnostic prior map with unbiased semantic information to guide the subsequent detection process. Equipped with the proposed two modules, we endow the model with a strong discriminative ability to distinguish the target objects from distractors belonging to the base classes. Extensive experiments show that our method outperforms the previous techniques by a large margin and achieves new state-of-the-art performance under various evaluation settings.

----

## [813] HEAP: Unsupervised Object Discovery and Localization with Contrastive Grouping

**Authors**: *Xin Zhang, Jinheng Xie, Yuan Yuan, Michael Bi Mi, Robby T. Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28562](https://doi.org/10.1609/aaai.v38i7.28562)

**Abstract**:

Unsupervised object discovery and localization aims to detect or segment objects in an image without any supervision. Recent efforts have demonstrated a notable potential to identify salient foreground objects by utilizing self-supervised transformer features. However, their scopes only build upon patch-level features within an image, neglecting region/image-level and cross-image relationships at a broader scale. Moreover, these methods cannot differentiate various semantics from multiple instances. To address these problems, we introduce Hierarchical mErging framework via contrAstive grouPing (HEAP). Specifically, a novel lightweight head with cross-attention mechanism is designed to adaptively group intra-image patches into semantically coherent regions based on correlation among self-supervised features. Further, to ensure the distinguishability among various regions, we introduce a region-level contrastive clustering loss to pull closer similar regions across images. Also, an image-level contrastive loss is present to push foreground and background representations apart, with which foreground objects and background are accordingly discovered. HEAP facilitates efficient hierarchical image decomposition, which contributes to more accurate object discovery while also enabling differentiation among objects of various classes. Extensive experimental results on semantic segmentation retrieval, unsupervised object discovery, and saliency detection tasks demonstrate that HEAP achieves state-of-the-art performance.

----

## [814] Scribble Hides Class: Promoting Scribble-Based Weakly-Supervised Semantic Segmentation with Its Class Label

**Authors**: *Xinliang Zhang, Lei Zhu, Hangzhou He, Lujia Jin, Yanye Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28563](https://doi.org/10.1609/aaai.v38i7.28563)

**Abstract**:

Scribble-based weakly-supervised semantic segmentation using sparse scribble supervision is gaining traction as it reduces annotation costs when compared to fully annotated alternatives. Existing methods primarily generate pseudo-labels by diffusing labeled pixels to unlabeled ones with local cues for supervision. However, this diffusion process fails to exploit global semantics and class-specific cues, which are important for semantic segmentation. In this study, we propose a class-driven scribble promotion network, which utilizes both scribble annotations and pseudo-labels informed by image-level classes and global semantics for supervision. Directly adopting pseudo-labels might misguide the segmentation model, thus we design a localization rectification module to correct foreground representations in the feature space. To further combine the advantages of both supervisions, we also introduce a distance entropy loss for uncertainty reduction, which adapts per-pixel confidence weights according to the reliable region determined by the scribble and pseudo-label's boundary.  Experiments on the ScribbleSup dataset with different qualities of scribble annotations outperform all the previous methods, demonstrating the superiority and robustness of our method. The code is available at https://github.com/Zxl19990529/Class-driven-Scribble-Promotion-Network.

----

## [815] Negative Pre-aware for Noisy Cross-Modal Matching

**Authors**: *Xu Zhang, Hao Li, Mang Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28564](https://doi.org/10.1609/aaai.v38i7.28564)

**Abstract**:

Cross-modal noise-robust learning is a challenging task since noisy correspondence is hard to recognize and rectify. Due to the cumulative and unavoidable negative impact of unresolved noise, existing methods cannot maintain a stable performance when the noise increases. In this paper, we present a novel Negative Pre-aware Cross-modal (NPC) matching solution for large visual-language model fine-tuning on noisy downstream tasks. It is featured in two aspects: (1) For noise recognition and resistance, previous methods usually directly filter out a noise subset, we propose to estimate the negative impact of each sample. It does not need additional correction mechanisms that may predict unreliable correction results, leading to self-reinforcing error. We assign a confidence weight to each sample according to its negative impact in the training process. This adaptively adjusts the contribution of each sample to avoid noisy accumulation. (2) For maintaining stable performance with increasing noise, we utilize the memorization effect of DNNs by maintaining a memory bank. Specifically, we apply GMM to select high-confident clean samples as the memory entry, where the memory entry is used to estimate the negative impact of each sample. Since clean samples are easier distinguished by GMM with increasing noise, the memory bank can still maintain high quality at a high noise ratio. Compared to the correction mechanism focusing on noise samples, memory bank-based estimation is more robust, which makes the model performance stable on noisy datasets. Extensive experiments demonstrate that our method significantly improves matching accuracy and performance stability at increasing noise ratio. Our approach also surpasses the state-of-the-art methods by a large margin. The code is available at: https://github.com/ZhangXu0963/NPC.

----

## [816] Compositional Inversion for Stable Diffusion Models

**Authors**: *Xulu Zhang, Xiao-Yong Wei, Jinlin Wu, Tianyi Zhang, Zhaoxiang Zhang, Zhen Lei, Qing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28565](https://doi.org/10.1609/aaai.v38i7.28565)

**Abstract**:

Inversion methods, such as Textual Inversion, generate personalized images by incorporating concepts of interest provided by user images. However, existing methods often suffer from overfitting issues, where the dominant presence of inverted concepts leads to the absence of other desired concepts. It stems from the fact that during inversion, the irrelevant semantics in the user images are also encoded, forcing the inverted concepts to occupy locations far from the core distribution in the embedding space. To address this issue, we propose a method that guides the inversion process towards the core distribution for compositional embeddings. Additionally, we introduce a spatial regularization approach to balance the attention on the concepts being composed. Our method is designed as a post-training approach and can be seamlessly integrated with other inversion methods. Experimental results demonstrate the effectiveness of our proposed approach in mitigating the overfitting problem and generating more diverse and balanced compositions of concepts in the synthesized images. The source code is available at https://github.com/zhangxulu1996/Compositional-Inversion.

----

## [817] Cross-Modal Match for Language Conditioned 3D Object Grounding

**Authors**: *Yachao Zhang, Runze Hu, Ronghui Li, Yanyun Qu, Yuan Xie, Xiu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28566](https://doi.org/10.1609/aaai.v38i7.28566)

**Abstract**:

Language conditioned 3D object grounding aims to find the object within the 3D scene mentioned by natural language descriptions, which mainly depends on the matching between visual and natural language. Considerable improvement in grounding performance is achieved by improving the multimodal fusion mechanism or bridging the gap between detection and matching. However, several mismatches are ignored, i.e., mismatch in local visual representation and global sentence representation, and mismatch in visual space and corresponding label word space. In this paper, we propose crossmodal match for 3D grounding from mitigating these mismatches perspective. Specifically, to match local visual features with the global description sentence, we propose BEV (Bird’s-eye-view) based global information embedding module. It projects multiple object proposal features into the BEV and the relations of different objects are accessed by the visual transformer which can model both positions and features with long-range dependencies. To circumvent the mismatch in feature spaces of different modalities, we propose crossmodal consistency learning. It performs cross-modal consistency constraints to convert the visual feature space into the label word feature space resulting in easier matching. Besides, we introduce label distillation loss and global distillation loss to drive these matches learning in a distillation way. We evaluate our method in mainstream evaluation settings on three datasets, and the results demonstrate the effectiveness of the proposed method.

----

## [818] MotionGPT: Finetuned LLMs Are General-Purpose Motion Generators

**Authors**: *Yaqi Zhang, Di Huang, Bin Liu, Shixiang Tang, Yan Lu, Lu Chen, Lei Bai, Qi Chu, Nenghai Yu, Wanli Ouyang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28567](https://doi.org/10.1609/aaai.v38i7.28567)

**Abstract**:

Generating realistic human motion from given action descriptions has experienced significant advancements because of the emerging requirement of digital humans. While recent works have achieved impressive results in generating motion directly from textual action descriptions, they often support only a single modality of the control signal, which limits their application in the real digital human industry. This paper presents a Motion General-Purpose generaTor (MotionGPT) that can use multimodal control signals, e.g., text and single-frame poses, for generating consecutive human motions by treating multimodal signals as special input tokens in large language models (LLMs). Specifically, we first quantize multimodal control signals into discrete codes and then formulate them in a unified prompt instruction to ask the LLMs to generate the motion answer. Our MotionGPT demonstrates a unified human motion generation model with multimodal control signals by tuning a mere 0.4% of LLM parameters. To the best of our knowledge, MotionGPT is the first method to generate human motion by multimodal control signals, which we hope can shed light on this new direction. Visit our webpage at https://qiqiapink.github.io/MotionGPT/.

----

## [819] Concept-Guided Prompt Learning for Generalization in Vision-Language Models

**Authors**: *Yi Zhang, Ce Zhang, Ke Yu, Yushun Tang, Zhihai He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28568](https://doi.org/10.1609/aaai.v38i7.28568)

**Abstract**:

Contrastive Language-Image Pretraining (CLIP) model has exhibited remarkable efficacy in establishing cross-modal connections between texts and images, yielding impressive
performance across a broad spectrum of downstream applications through fine-tuning. However, for generalization tasks, the current fine-tuning methods for CLIP, such as CoOp and
CoCoOp, demonstrate relatively low performance on some fine-grained datasets. We recognize the underlying reason is that these previous methods only projected global features
into the prompt, neglecting the various visual concepts, such as colors, shapes, and sizes, which are naturally transferable
across domains and play a crucial role in generalization tasks. To address this issue, in this work, we propose
Concept-Guided Prompt Learning (CPL) for vision-language models. Specifically, we leverage the well-learned knowledge
of CLIP to create a visual concept cache to enable conceptguided prompting. In order to refine the text features, we further
develop a projector that transforms multi-level visual features into text features. We observe that this concept-guided
prompt learning approach is able to achieve enhanced consistency between visual and linguistic modalities. Extensive
experimental results demonstrate that our CPL method significantly improves generalization capabilities compared to
the current state-of-the-art methods.

----

## [820] ISP-Teacher: Image Signal Process with Disentanglement Regularization for Unsupervised Domain Adaptive Dark Object Detection

**Authors**: *Yin Zhang, Yongqiang Zhang, Zian Zhang, Man Zhang, Rui Tian, Mingli Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28569](https://doi.org/10.1609/aaai.v38i7.28569)

**Abstract**:

Object detection in dark conditions has always been a great challenge due to the complex formation process of low-light images. Currently, the mainstream methods usually adopt domain adaptation with Teacher-Student architecture to solve the dark object detection problem, and they imitate the dark conditions by using non-learnable data augmentation strategies on the annotated source daytime images. Note that these methods neglected to model the intrinsic imaging process, i.e. image signal processing (ISP), which is important for camera sensors to generate low-light images. To solve the above problems, in this paper, we propose a novel method named ISP-Teacher for dark object detection by exploring Teacher-Student architecture from a new perspective (i.e. self-supervised learning based ISP degradation). Specifically, we first design a day-to-night transformation module that consistent with the ISP pipeline of the camera sensors (ISP-DTM) to make the augmented images look more in line with the natural low-light images captured by cameras, and the ISP-related parameters are learned in a self-supervised manner. Moreover, to avoid the conflict between the ISP degradation and detection tasks in a shared encoder, we propose a disentanglement regularization (DR) that minimizes the absolute value of cosine similarity to disentangle two tasks and push two gradients vectors as orthogonal as possible. Extensive experiments conducted on two benchmarks show the effectiveness of our method in dark object detection. In particular, ISP-Teacher achieves an improvement of +2.4% AP and +3.3% AP over the SOTA method on BDD100k and SHIFT datasets, respectively. The code can be found at https://github.com/zhangyin1996/ISP-Teacher.

----

## [821] ArtBank: Artistic Style Transfer with Pre-trained Diffusion Model and Implicit Style Prompt Bank

**Authors**: *Zhanjie Zhang, Quanwei Zhang, Wei Xing, Guangyuan Li, Lei Zhao, Jiakai Sun, Zehua Lan, Junsheng Luan, Yiling Huang, Huaizhong Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28570](https://doi.org/10.1609/aaai.v38i7.28570)

**Abstract**:

Artistic style transfer aims to repaint the content image with the learned artistic style. Existing artistic style transfer methods can be divided into two categories: small model-based approaches and pre-trained large-scale model-based approaches. Small model-based approaches can preserve the content strucuture, but fail to produce highly realistic stylized images and introduce artifacts and disharmonious patterns; Pre-trained large-scale model-based approaches can generate highly realistic stylized images but struggle with preserving the content structure. To address the above issues, we propose ArtBank, a novel artistic style transfer framework, to generate highly realistic stylized images while preserving the content structure of the content images. Specifically, to sufficiently dig out the knowledge embedded in pre-trained large-scale models, an Implicit Style Prompt Bank (ISPB), a set of trainable parameter matrices, is designed to learn and store knowledge from the collection of artworks and behave as a visual prompt to guide pre-trained large-scale models to generate highly realistic stylized images while preserving content structure. Besides, to accelerate training the above ISPB, we propose a novel Spatial-Statistical-based self-Attention Module (SSAM). The qualitative and quantitative experiments demonstrate the superiority of our proposed method over state-of-the-art artistic style transfer methods. Code is available at https://github.com/Jamie-Cheung/ArtBank.

----

## [822] A New Benchmark and Model for Challenging Image Manipulation Detection

**Authors**: *Zhenfei Zhang, Mingyang Li, Ming-Ching Chang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28571](https://doi.org/10.1609/aaai.v38i7.28571)

**Abstract**:

The ability to detect manipulation in multimedia data is vital in digital forensics. Existing Image Manipulation Detection (IMD) methods are mainly based on detecting anomalous features arisen from image editing or double compression artifacts. All existing IMD techniques encounter challenges when it comes to detecting small tampered regions from a large image. Moreover, compression-based IMD approaches face difficulties in cases of double compression of identical quality factors. To investigate the State-of-The-Art (SoTA) IMD methods in those challenging conditions, we introduce a new Challenging Image Manipulation Detection (CIMD) benchmark dataset, which consists of two subsets, for evaluating editing-based and compression-based IMD methods, respectively. The dataset images were manually taken and tampered with high-quality annotations. In addition, we propose a new two-branch network model based on HRNet that can better detect both the image-editing and compression artifacts in those challenging conditions. Extensive experiments on the CIMD benchmark show that our model significantly outperforms SoTA IMD methods on CIMD. The dataset is available at: https://github.com/ZhenfeiZ/CIMD.

----

## [823] TMFormer: Token Merging Transformer for Brain Tumor Segmentation with Missing Modalities

**Authors**: *Zheyu Zhang, Gang Yang, Yueyi Zhang, Huanjing Yue, Aiping Liu, Yunwei Ou, Jian Gong, Xiaoyan Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28572](https://doi.org/10.1609/aaai.v38i7.28572)

**Abstract**:

Numerous techniques excel in brain tumor segmentation using multi-modal magnetic resonance imaging (MRI) sequences, delivering exceptional results. However, the prevalent absence of modalities in clinical scenarios hampers performance. Current approaches frequently resort to zero maps as substitutes for missing modalities, inadvertently introducing feature bias and redundant computations. To address these issues, we present the Token Merging transFormer (TMFormer) for robust brain tumor segmentation with missing modalities. TMFormer tackles these challenges by extracting and merging accessible modalities into more compact token sequences. The architecture comprises two core components: the Uni-modal Token Merging Block (UMB) and the Multi-modal Token Merging Block (MMB). The UMB enhances individual modality representation by adaptively consolidating spatially redundant tokens within and outside tumor-related regions, thereby refining token sequences for augmented representational capacity. Meanwhile, the MMB mitigates multi-modal feature fusion bias, exclusively leveraging tokens from present modalities and merging them into a unified multi-modal representation to accommodate varying modality combinations. Extensive experimental results on the BraTS 2018 and 2020 datasets demonstrate the superiority and efficacy of TMFormer compared to state-of-the-art methods when dealing with missing modalities.

----

## [824] FaceRSA: RSA-Aware Facial Identity Cryptography Framework

**Authors**: *Zhongyi Zhang, Tianyi Wei, Wenbo Zhou, Hanqing Zhao, Weiming Zhang, Nenghai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28573](https://doi.org/10.1609/aaai.v38i7.28573)

**Abstract**:

With the flourishing of the Internet, sharing one's photos or automated processing of faces using computer vision technology has become an everyday occurrence. While enjoying the convenience, the concern for identity privacy is also emerging. Therefore, some efforts introduced the concept of ``password'' from traditional cryptography such as RSA into the face anonymization and deanonymization task to protect the facial identity without compromising the usability of the face image. However, these methods either suffer from the poor visual quality of the synthesis results or do not possess the full cryptographic properties, resulting in compromised security. In this paper, we present the first facial identity cryptography framework with full properties analogous to RSA. Our framework leverages the powerful generative capabilities of StyleGAN to achieve megapixel-level facial identity anonymization and deanonymization. Thanks to the great semantic decoupling of StyleGAN's latent space, the identity encryption and decryption process are performed in latent space by a well-designed password mapper in the manner of editing latent code. Meanwhile, the password-related information is imperceptibly hidden in the edited latent code owing to the redundant nature of the latent space. To make our cryptographic framework possesses all the properties analogous to RSA, we propose three types of loss functions: single anonymization loss, sequential anonymization loss, and associated anonymization loss. Extensive experiments and ablation analyses demonstrate the superiority of our method in terms of the quality of synthesis results, identity-irrelevant attributes preservation, deanonymization accuracy, and completeness of properties analogous to RSA.

----

## [825] Spatial-Contextual Discrepancy Information Compensation for GAN Inversion

**Authors**: *Ziqiang Zhang, Yan Yan, Jing-Hao Xue, Hanzi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28574](https://doi.org/10.1609/aaai.v38i7.28574)

**Abstract**:

Most existing GAN inversion methods either achieve accurate reconstruction but lack editability or offer strong editability at the cost of fidelity. Hence, how to balance the distortion-editability trade-off is a significant challenge for GAN inversion.  To address this challenge, we introduce a novel spatial-contextual discrepancy information compensation-based GAN-inversion method (SDIC), which consists of a discrepancy information prediction network (DIPN) and a discrepancy information compensation network (DICN). SDIC follows a ``compensate-and-edit'' paradigm and successfully bridges the gap in image details between the original image and the reconstructed/edited image. On the one hand, DIPN encodes the multi-level spatial-contextual information of the original and initial reconstructed images and then predicts a spatial-contextual guided discrepancy map with two hourglass modules. In this way, a reliable discrepancy map that models the contextual relationship and captures fine-grained image details is learned. On the other hand, DICN incorporates the predicted discrepancy information into both the latent code and the GAN generator with different transformations, generating high-quality reconstructed/edited images. This effectively compensates for the loss of image details during GAN inversion. Both quantitative and qualitative experiments demonstrate that our proposed method achieves the excellent distortion-editability trade-off at a fast inference speed for both image inversion and editing tasks. Our code is available at https://github.com/ZzqLKED/SDIC.

----

## [826] Self-Distillation Regularized Connectionist Temporal Classification Loss for Text Recognition: A Simple Yet Effective Approach

**Authors**: *Ziyin Zhang, Ning Lu, Minghui Liao, Yongshuai Huang, Cheng Li, Min Wang, Wei Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28575](https://doi.org/10.1609/aaai.v38i7.28575)

**Abstract**:

Text recognition methods are gaining rapid development. Some advanced techniques, e.g., powerful modules, language models, and un- and semi-supervised learning schemes, consecutively push the performance on public benchmarks forward. However, the problem of how to better optimize a text recognition model from the perspective of loss functions is largely overlooked. CTC-based methods, widely used in practice due to their good balance between performance and inference speed, still grapple with accuracy degradation. This is because CTC loss emphasizes the optimization of the entire sequence target while neglecting to learn individual characters. We propose a self-distillation scheme for CTC-based model to address this issue. It incorporates a framewise regularization term in CTC loss to emphasize individual supervision, and leverages the maximizing-a-posteriori of latent alignment to solve the inconsistency problem that arises in distillation between CTC-based models. We refer to the regularized CTC loss as Distillation Connectionist Temporal Classification (DCTC) loss. DCTC loss is module-free, requiring no extra parameters, longer inference lag, or additional training data or phases. Extensive experiments on public benchmarks demonstrate that DCTC can boost text recognition model accuracy by up to 2.6%, without any of these drawbacks.

----

## [827] PNeRFLoc: Visual Localization with Point-Based Neural Radiance Fields

**Authors**: *Boming Zhao, Luwei Yang, Mao Mao, Hujun Bao, Zhaopeng Cui*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28576](https://doi.org/10.1609/aaai.v38i7.28576)

**Abstract**:

Due to the ability to synthesize high-quality novel views, Neural Radiance Fields (NeRF) has been recently exploited to improve visual localization in a known environment. However, the existing methods mostly utilize NeRF for data augmentation to improve the regression model training, and their performances on novel viewpoints and appearances are still limited due to the lack of geometric constraints. In this paper, we propose a novel visual localization framework, i.e., PNeRFLoc, based on a unified point-based representation. On one hand, PNeRFLoc supports the initial pose estimation by matching 2D and 3D feature points as traditional structure-based methods; on the other hand, it also enables pose refinement with novel view synthesis using rendering-based optimization. Specifically, we propose a novel feature adaption module to close the gaps between the features for visual localization and neural rendering. To improve the efficacy and efficiency of neural rendering-based optimization, we also developed an efficient rendering-based framework with a warping loss function. Extensive experiments demonstrate that PNeRFLoc performs the best on the synthetic dataset when the 3D NeRF model can be well learned, and significantly outperforms all the NeRF-boosted localization methods with on-par SOTA performance on the real-world benchmark localization datasets. Project webpage: https://zju3dv.github.io/PNeRFLoc/.

----

## [828] SimDistill: Simulated Multi-Modal Distillation for BEV 3D Object Detection

**Authors**: *Haimei Zhao, Qiming Zhang, Shanshan Zhao, Zhe Chen, Jing Zhang, Dacheng Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28577](https://doi.org/10.1609/aaai.v38i7.28577)

**Abstract**:

Multi-view camera-based 3D object detection has become popular due to its low cost, but accurately inferring 3D geometry solely from camera data remains challenging and may lead to inferior performance. Although distilling precise 3D geometry knowledge from LiDAR data could help tackle this challenge, the benefits of LiDAR information could be greatly hindered by the significant modality gap between different sensory modalities. To address this issue, we propose a Simulated multi-modal Distillation (SimDistill) method by carefully crafting the model architecture and distillation strategy. Specifically, we devise multi-modal architectures for both teacher and student models, including a LiDAR-camera fusion-based teacher and a simulated fusion-based student. Owing to the ``identical'' architecture design, the student can mimic the teacher to generate multi-modal features with merely multi-view images as input, where a geometry compensation module is introduced to bridge the modality gap. Furthermore, we propose a comprehensive multi-modal distillation scheme that supports intra-modal, cross-modal, and multi-modal fusion distillation simultaneously in the Bird's-eye-view space. Incorporating them together, our SimDistill can learn better feature representations for 3D object detection while maintaining a cost-effective camera-only deployment. Extensive experiments validate the effectiveness and superiority of SimDistill over state-of-the-art methods, achieving an improvement of 4.8% mAP and 4.1% NDS over the baseline detector. The source code will be released at https://github.com/ViTAE-Transformer/SimDistill.

----

## [829] Large Occluded Human Image Completion via Image-Prior Cooperating

**Authors**: *Hengrun Zhao, Yu Zeng, Huchuan Lu, Lijun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28578](https://doi.org/10.1609/aaai.v38i7.28578)

**Abstract**:

The completion of large occluded human body images poses a unique challenge for general image completion methods. The complex shape variations of human bodies make it difficult to establish a consistent understanding of their structures. Furthermore, as human vision is highly sensitive to human bodies, even slight artifacts can significantly compromise image fidelity. To address these challenges, we propose a large occluded human image completion (LOHC) model based on a novel image-prior cooperative completion strategy. Our model leverages human segmentation maps as a prior, and completes the image and prior simultaneously. Compared to the widely adopted prior-then-image completion strategy for object completion, this cooperative completion process fosters more effective interaction between the prior and image information. Our model consists of two stages. The first stage is a transformer-based auto-regressive network that predicts the overall structure of the missing area by generating a coarse completed image at a lower resolution. The second stage is a convolutional network that refines the coarse images. As the coarse result may not always be accurate, we propose a Dynamic Fusion Module (DFM) to selectively fuses the useful features from the coarse image with the original input at spatial and channel levels. Through extensive experiments, we demonstrate our method’s superior performance compared to state-of-the-art methods.

----

## [830] Recognizing Ultra-High-Speed Moving Objects with Bio-Inspired Spike Camera

**Authors**: *Junwei Zhao, Shiliang Zhang, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28579](https://doi.org/10.1609/aaai.v38i7.28579)

**Abstract**:

Bio-inspired spike camera mimics the sampling principle of primate fovea. It presents high temporal resolution and dynamic range, showing great promise in fast-moving object recognition. However, the physical limit of CMOS technology in spike cameras still hinders their capability of recognizing ultra-high-speed moving objects, e.g., extremely fast motions cause blur during the imaging process of spike cameras. This paper presents the first theoretical analysis for the causes of spiking motion blur and proposes a robust representation that addresses this issue through temporal-spatial context learning. The proposed method leverages multi-span feature aggregation to capture temporal cues and employs residual deformable convolution to model spatial correlation among neighbouring pixels. Additionally, this paper contributes an original real-captured spiking recognition dataset consisting of 12,000 ultra-high-speed (equivalent speed > 500 km/h) moving objects. Experimental results show that the proposed method achieves 73.2% accuracy in recognizing 10 classes of ultra-high-speed moving objects, outperforming all existing spike-based recognition methods. Resources will be available at https://github.com/Evin-X/UHSR.

----

## [831] Rethinking Two-Stage Referring Expression Comprehension: A Novel Grounding and Segmentation Method Modulated by Point

**Authors**: *Peizhi Zhao, Shiyi Zheng, Wenye Zhao, Dongsheng Xu, Pijian Li, Yi Cai, Qingbao Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28580](https://doi.org/10.1609/aaai.v38i7.28580)

**Abstract**:

As a fundamental and challenging task in the vision and language domain, Referring Expression Comprehension (REC) has shown impressive improvements recently. However, for a complex task that couples the comprehension of abstract concepts and the localization of concrete instances, one-stage approaches are bottlenecked by computing and data resources. To obtain a low-cost solution, the prevailing two-stage approaches decouple REC into localization (region proposal) and comprehension (region-expression matching) at region-level, but the solution based on isolated regions cannot sufficiently utilize the context and is usually limited by the quality of proposals. Therefore, it is necessary to rebuild an efficient two-stage solution system. In this paper, we propose a point-based two-stage framework for REC, in which the two stages are redefined as point-based cross-modal comprehension and point-based instance localization. Specifically, we reconstruct the raw bounding box and segmentation mask into center and mass scores as soft ground-truth for measuring point-level cross-modal correlations. With the soft ground-truth, REC can be approximated as a binary classification problem, which fundamentally avoids the impact of isolated regions on the optimization process. Remarkably, the consistent metrics between center and mass scores allow our system to directly optimize grounding and segmentation by utilizing the same architecture. Experiments on multiple benchmarks show the feasibility and potential of our point-based paradigm. Our code available at https://github.com/VILAN-Lab/PBREC-MT.

----

## [832] Optical Flow for Spike Camera with Hierarchical Spatial-Temporal Spike Fusion

**Authors**: *Rui Zhao, Ruiqin Xiong, Jian Zhang, Xinfeng Zhang, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28581](https://doi.org/10.1609/aaai.v38i7.28581)

**Abstract**:

As an emerging neuromorphic camera with an asynchronous working mechanism, spike camera shows good potential for high-speed vision tasks. Each pixel in spike camera accumulates photons persistently and fires a spike whenever the accumulation exceeds a threshold. Such high-frequency fine-granularity photon recording facilitates the analysis and recovery of dynamic scenes with high-speed motion. This paper considers the optical flow estimation problem for spike cameras. Due to the Poisson nature of incoming photons, the occurrence of spikes is random and fluctuating, making conventional image matching inefficient. We propose a Hierarchical Spatial-Temporal (HiST) fusion module for spike representation to pursue reliable feature matching and develop a robust optical flow network, dubbed as HiST-SFlow. The HiST extracts features at multiple moments and hierarchically fuses the spatial-temporal information. We also propose an intra-moment filtering module to further extract the feature and suppress the influence of randomness in spikes. A scene loss is proposed to ensure that this hierarchical representation recovers the essential visual information in the scene. Experimental results demonstrate that the proposed method achieves state-of-the-art performance compared with the existing methods. The source codes are available at https://github.com/ruizhao26/HiST-SFlow.

----

## [833] Towards Fine-Grained HBOE with Rendered Orientation Set and Laplace Smoothing

**Authors**: *Ruisi Zhao, Mingming Li, Zheng Yang, Binbin Lin, Xiaohui Zhong, Xiaobo Ren, Deng Cai, Boxi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28582](https://doi.org/10.1609/aaai.v38i7.28582)

**Abstract**:

Human body orientation estimation (HBOE) aims to estimate the orientation of a human body relative to the camera’s frontal view. Despite recent advancements in this field, there still exist limitations in achieving fine-grained results. We identify certain defects and propose corresponding approaches as follows: 1). Existing datasets suffer from non-uniform angle distributions, resulting in sparse image data for certain angles. To provide comprehensive and high-quality data, we introduce RMOS (Rendered Model Orientation Set), a rendered dataset comprising 150K accurately labeled human instances with a wide range of orientations. 2). Directly using one-hot vector as labels may overlook the similarity between angle labels, leading to poor supervision.  And converting the predictions from radians to degrees enlarges the regression error. To enhance supervision, we employ Laplace smoothing to vectorize the label, which contains more information. For fine-grained predictions, we adopt weighted Smooth-L1-loss to align predictions with the smoothed-label, thus providing robust supervision. 3). Previous works ignore body-part-specific information, resulting in coarse predictions. By employing local-window self-attention, our model could utilize different body part information for more precise orientation estimations. We validate the effectiveness of our method in the benchmarks with extensive experiments and show that our method outperforms state-of-the-art. Project is available at: https://github.com/Whalesong-zrs/Towards-Fine-grained-HBOE.

----

## [834] No Head Left Behind - Multi-Head Alignment Distillation for Transformers

**Authors**: *Tianyang Zhao, Kunwar Yashraj Singh, Srikar Appalaraju, Peng Tang, Vijay Mahadevan, R. Manmatha, Ying Nian Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28583](https://doi.org/10.1609/aaai.v38i7.28583)

**Abstract**:

Knowledge distillation aims at reducing model size without compromising much performance. Recent work has applied it to large vision-language (VL) Transformers, and has shown that attention maps in the multi-head attention modules of vision-language Transformers contain extensive intra-modal and cross-modal co-reference relations to be distilled. The standard approach is to apply a one-to-one attention map distillation loss, i.e. the Teacher's first attention head instructs the Student's first head, the second teaches the second, and so forth, but this only works when the numbers of attention heads in the Teacher and Student are the same. To remove this constraint, we propose a new Attention Map Alignment Distillation (AMAD) method for Transformers with multi-head attention, which works for a Teacher and a Student with different numbers of attention heads. Specifically, we soft-align different heads in Teacher and Student attention maps using a cosine similarity weighting. The Teacher head contributes more to the Student heads for which it has a higher similarity weight. Each Teacher head contributes to all the Student heads by minimizing the divergence between the attention activation distributions for the soft-aligned heads. No head is left behind. This distillation approach operates like cross-attention. We experiment on distilling VL-T5 and BLIP, and apply AMAD loss on their T5, BERT, and ViT sub-modules. We show, under vision-language setting, that AMAD outperforms conventional distillation methods on VQA-2.0, COCO captioning, and Multi30K translation datasets. We further show that even without VL pre-training, the distilled VL-T5 models outperform corresponding VL pre-trained VL-T5 models that are further fine-tuned by ground-truth signals, and that fine-tuning distillation can also compensate to some degree for the absence of VL pre-training for BLIP models.

----

## [835] SFC: Shared Feature Calibration in Weakly Supervised Semantic Segmentation

**Authors**: *Xinqiao Zhao, Feilong Tang, Xiaoyang Wang, Jimin Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28584](https://doi.org/10.1609/aaai.v38i7.28584)

**Abstract**:

Image-level weakly supervised semantic segmentation has received increasing attention due to its low annotation cost. Existing methods mainly rely on Class Activation Mapping (CAM) to obtain pseudo-labels for training semantic segmentation models. In this work, we are the first to demonstrate that long-tailed distribution in training data can cause the CAM calculated through classifier weights over-activated for head classes and under-activated for tail classes due to the shared features among head- and tail- classes. This degrades pseudo-label quality and further influences final semantic segmentation performance. To address this issue, we propose a Shared Feature Calibration (SFC) method for CAM generation. Specifically, we leverage the class prototypes which carry positive shared features and propose a Multi-Scaled Distribution-Weighted (MSDW) consistency loss for narrowing the gap between the CAMs generated through classifier weights and class prototypes during training. The MSDW loss counterbalances over-activation and under-activation by calibrating the shared features in head-/tail-class classifier weights. Experimental results show that our SFC significantly improves CAM boundaries and achieves new state-of-the-art performances. The project is available at https://github.com/Barrett-python/SFC.

----

## [836] Unifying Multi-Modal Uncertainty Modeling and Semantic Alignment for Text-to-Image Person Re-identification

**Authors**: *Zhiwei Zhao, Bin Liu, Yan Lu, Qi Chu, Nenghai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28585](https://doi.org/10.1609/aaai.v38i7.28585)

**Abstract**:

Text-to-Image person re-identification (TI-ReID) aims to retrieve the images of target identity according to the given textual description. The existing methods in TI-ReID focus on aligning the visual and textual modalities through contrastive feature alignment or reconstructive masked language modeling (MLM). However, these methods parameterize the image/text instances as deterministic embeddings and do not explicitly consider the inherent uncertainty in pedestrian images and their textual descriptions, leading to limited image-text relationship expression and semantic alignment. To address the above problem, in this paper, we propose a novel method that unifies multi-modal uncertainty modeling and semantic alignment for TI-ReID. Specifically, we model the image and textual feature vectors of pedestrian as Gaussian distributions, where the multi-granularity uncertainty of the distribution is estimated by incorporating batch-level and identity-level feature variances for each modality. The multi-modal uncertainty modeling acts as a feature augmentation and provides richer image-text semantic relationship. Then we present a bi-directional cross-modal circle loss to more effectively align the probabilistic features between image and text in a self-paced manner. To further promote more comprehensive image-text semantic alignment, we design a task that complements the masked language modeling, focusing on the cross-modality semantic recovery of global masked token after cross-modal interaction. Extensive experiments conducted on three TI-ReID datasets highlight the effectiveness and superiority of our method over state-of-the-arts.

----

## [837] Mining Gaze for Contrastive Learning toward Computer-Assisted Diagnosis

**Authors**: *Zihao Zhao, Sheng Wang, Qian Wang, Dinggang Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28586](https://doi.org/10.1609/aaai.v38i7.28586)

**Abstract**:

Obtaining large-scale radiology reports can be difficult for medical images due to ethical concerns, limiting the effectiveness of contrastive pre-training in the medical image domain and underscoring the need for alternative methods. In this paper, we propose eye-tracking as an alternative to text reports, as it allows for the passive collection of gaze signals without ethical issues. By tracking the gaze of radiologists as they read and diagnose medical images, we can understand their visual attention and clinical reasoning. When a radiologist has similar gazes for two medical images, it may indicate semantic similarity for diagnosis, and these images should be treated as positive pairs when pre-training a computer-assisted diagnosis (CAD) network through contrastive learning. Accordingly, we introduce the Medical contrastive Gaze Image Pre-training (McGIP) as a plug-and-play module for contrastive learning frameworks. McGIP uses radiologist gaze to guide contrastive pre-training. We evaluate our method using two representative types of medical images and two common types of gaze data. The experimental results demonstrate the practicality of McGIP, indicating its high potential for various clinical scenarios and applications.

----

## [838] Quad Bayer Joint Demosaicing and Denoising Based on Dual Encoder Network with Joint Residual Learning

**Authors**: *Bolun Zheng, Haoran Li, Quan Chen, Tingyu Wang, Xiaofei Zhou, Zhenghui Hu, Chenggang Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28587](https://doi.org/10.1609/aaai.v38i7.28587)

**Abstract**:

The recent imaging technology Quad Bayer CFA brings better imaging PSNR and higher visual quality compared to traditional Bayer CFA, but also serious challenges for demosaicing and denoising during the ISP pipeline. In this paper, we propose a novel dual encoder network, namely DRNet, to achieve joint demosaicing and denoising for Quad Bayer CFA. The dual encoders are carefully designed in that one is mainly constructed by a joint residual block to jointly estimate the residuals for demosaicing and denoising separately. In contrast, the other one is started with a pixel modulation block which is specially designed to match the characteristics of Quad Bayer pattern for better feature extraction. We demonstrate the effectiveness of each proposed component through detailed ablation investigations. The comparison results on public benchmarks illustrate that our DRNet achieves an apparent performance gain~(0.38dB to the 2nd best) from the state-of-the-art method and balances performance and efficiency well. The experiments on real-world images show that the proposed method could enhance the reconstruction quality from the native ISP algorithm.

----

## [839] End-to-End RGB-D Image Compression via Exploiting Channel-Modality Redundancy

**Authors**: *Huiming Zheng, Wei Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28588](https://doi.org/10.1609/aaai.v38i7.28588)

**Abstract**:

As a kind of 3D data, RGB-D images have been extensively used in object tracking, 3D reconstruction, remote sensing mapping, and other tasks. In the realm of computer vision, the significance of RGB-D images is progressively growing. However, the existing learning-based image compression methods usually process RGB images and depth images separately, which cannot entirely exploit the redundant information between the modalities, limiting the further improvement of the Rate-Distortion performance. With the goal of overcoming the defect, in this paper, we propose a learning-based dual-branch RGB-D image compression framework. Compared with traditional RGB domain compression scheme, a YUV domain compression scheme is presented for spatial redundancy removal. In addition, Intra-Modality Attention (IMA) and Cross-Modality Attention (CMA) are introduced for modal redundancy removal. For the sake of benefiting from cross-modal prior information, Context Prediction Module (CPM) and Context Fusion Module (CFM) are raised in the conditional entropy model which makes the context probability prediction more accurate. The experimental results demonstrate our method outperforms existing image compression methods in two RGB-D image datasets. Compared with BPG, our proposed framework can achieve up to 15% bit rate saving for RGB images.

----

## [840] Any-Size-Diffusion: Toward Efficient Text-Driven Synthesis for Any-Size HD Images

**Authors**: *Qingping Zheng, Yuanfan Guo, Jiankang Deng, Jianhua Han, Ying Li, Songcen Xu, Hang Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28589](https://doi.org/10.1609/aaai.v38i7.28589)

**Abstract**:

Stable diffusion, a generative model used in text-to-image synthesis, frequently encounters resolution-induced composition problems when generating images of varying sizes. This issue primarily stems from the model being trained on pairs of single-scale images and their corresponding text descriptions. Moreover, direct training on images of unlimited sizes is unfeasible, as it would require an immense number of text-image pairs and entail substantial computational expenses. To overcome these challenges, we propose a two-stage pipeline named Any-Size-Diffusion (ASD), designed to efficiently generate well-composed HD images of any size, while minimizing the need for high-memory GPU resources. Specifically, the initial stage, dubbed Any Ratio Adaptability Diffusion (ARAD), leverages a selected set of images with a restricted range of ratios to optimize the text-conditional diffusion model, thereby improving its ability to adjust composition to accommodate diverse image sizes. To support the creation of images at any desired size, we further introduce a technique called Fast Seamless Tiled Diffusion (FSTD) at the subsequent stage. This method allows for the rapid enlargement of the ASD output to any high-resolution size, avoiding seaming artifacts or memory overloads. Experimental results on the LAION-COCO and MM-CelebA-HQ benchmarks demonstrate that ASD can produce well-structured images of arbitrary sizes, cutting down the inference time by 2X compared to the traditional tiled algorithm. The source code is available at https://github.com/ProAirVerse/Any-Size-Diffusion.

----

## [841] Spatio-Temporal Fusion for Human Action Recognition via Joint Trajectory Graph

**Authors**: *Yaolin Zheng, Hongbo Huang, Xiuying Wang, Xiaoxu Yan, Longfei Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28590](https://doi.org/10.1609/aaai.v38i7.28590)

**Abstract**:

Graph Convolutional Networks (GCNs) and Transformers have been widely applied to skeleton-based human action recognition, with each offering unique advantages in capturing spatial relationships and long-range dependencies. However, for most GCN methods, the construction of topological structures relies solely on the spatial information of human joints, limiting their ability to directly capture richer spatio-temporal dependencies. Additionally, the self-attention modules of many Transformer methods lack topological structure information, restricting the robustness and generalization of the models. To address these issues, we propose a Joint Trajectory Graph (JTG) that integrates spatio-temporal information into a uniform graph structure. We also present a Joint Trajectory GraphFormer (JT-GraphFormer), which directly captures the spatio-temporal relationships among all joint trajectories for human action recognition. To better integrate topological information into spatio-temporal relationships, we introduce a Spatio-Temporal Dijkstra Attention (STDA) mechanism to calculate relationship scores for all the joints in JTG. Furthermore, we incorporate the Koopman operator into the classification stage to enhance the model's representation ability and classification performance. Experiments demonstrate that JT-GraphFormer achieves outstanding performance in human action recognition tasks, outperforming state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and N-UCLA datasets.

----

## [842] ODTrack: Online Dense Temporal Token Learning for Visual Tracking

**Authors**: *Yaozong Zheng, Bineng Zhong, Qihua Liang, Zhiyi Mo, Shengping Zhang, Xianxian Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28591](https://doi.org/10.1609/aaai.v38i7.28591)

**Abstract**:

Online contextual reasoning and association across consecutive video frames are critical to perceive instances in visual tracking. However, most current top-performing trackers persistently lean on sparse temporal relationships between reference and search frames via an offline mode. Consequently, they can only interact independently within each image-pair and establish limited temporal correlations. To alleviate the above problem, we propose a simple, flexible and effective video-level tracking pipeline, named ODTrack, which densely associates the contextual relationships of video frames in an online token propagation manner. ODTrack receives video frames of arbitrary length to capture the spatio-temporal trajectory relationships of an instance, and compresses the discrimination features (localization information) of a target into a token sequence to achieve frame-to-frame association. This new solution brings the following benefits: 1) the purified token sequences can serve as prompts for the inference in the next video frame, whereby past information is leveraged to guide future inference; 2) the complex online update strategies are effectively avoided by the iterative propagation of token sequences, and thus we can achieve more efficient model representation and computation. ODTrack achieves a new SOTA performance on seven benchmarks, while running at real-time speed. Code and models are available at https://github.com/GXNU-ZhongLab/ODTrack.

----

## [843] PVALane: Prior-Guided 3D Lane Detection with View-Agnostic Feature Alignment

**Authors**: *Zewen Zheng, Xuemin Zhang, Yongqiang Mou, Xiang Gao, Chengxin Li, Guoheng Huang, Chi-Man Pun, Xiaochen Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28592](https://doi.org/10.1609/aaai.v38i7.28592)

**Abstract**:

Monocular 3D lane detection is essential for a reliable autonomous driving system and has recently been rapidly developing. Existing popular methods mainly employ a predefined 3D anchor for lane detection based on front-viewed (FV) space, aiming to mitigate the effects of view transformations. However, the perspective geometric distortion between FV and 3D space in this FV-based approach introduces extremely dense anchor designs, which ultimately leads to confusing lane representations. In this paper, we introduce a novel prior-guided perspective on lane detection and propose an end-to-end framework named PVALane, which utilizes 2D prior knowledge to achieve precise and efficient 3D lane detection. Since 2D lane predictions can provide strong priors for lane existence, PVALane exploits FV features to generate sparse prior anchors with potential lanes in 2D space. These dynamic prior anchors help PVALane to achieve distinct lane representations and effectively improve the precision of PVALane due to the reduced lane search space. Additionally, by leveraging these prior anchors and representing lanes in both FV and bird-eye-viewed (BEV) spaces, we effectively align and merge semantic and geometric information from FV and BEV features. Extensive experiments conducted on the OpenLane and ONCE-3DLanes datasets demonstrate the superior performance of our method compared to existing state-of-the-art approaches and exhibit excellent robustness.

----

## [844] SpFormer: Spatio-Temporal Modeling for Scanpaths with Transformer

**Authors**: *Wenqi Zhong, Linzhi Yu, Chen Xia, Junwei Han, Dingwen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28593](https://doi.org/10.1609/aaai.v38i7.28593)

**Abstract**:

Saccadic scanpath, a data representation of human visual behavior, has received broad interest in multiple domains. Scanpath is a complex eye-tracking data modality that includes the sequences of fixation positions and fixation duration, coupled with image information. However, previous methods usually face the spatial misalignment problem of fixation features and loss of critical temporal data (including temporal correlation and fixation duration). In this study, we propose a Transformer-based scanpath model, SpFormer, to alleviate these problems. First, we propose a fixation-centric paradigm to extract the aligned spatial fixation features and tokenize the scanpaths. Then, according to the visual working memory mechanism, we design a local meta attention to reduce the semantic redundancy of fixations and guide the model to focus on the meta scanpath. Finally, we progressively integrate the duration information and fuse it with the fixation features to solve the problem of ambiguous location with the Transformer block increasing. We conduct extensive experiments on four databases under three tasks. The SpFormer establishes new state-of-the-art results in distinct settings, verifying its flexibility and versatility in practical applications. The code can be obtained from https://github.com/wenqizhong/SpFormer.

----

## [845] ExpCLIP: Bridging Text and Facial Expressions via Semantic Alignment

**Authors**: *Yicheng Zhong, Huawei Wei, Peiji Yang, Zhisheng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28594](https://doi.org/10.1609/aaai.v38i7.28594)

**Abstract**:

The objective of stylized speech-driven facial animation is to create animations that encapsulate specific emotional expressions. Existing methods often depend on pre-established emotional labels or facial expression templates, which may limit the necessary flexibility for accurately conveying user intent.
In this research, we introduce a technique that enables the control of arbitrary styles by leveraging natural language as emotion prompts. This technique presents benefits in terms of both flexibility and user-friendliness.
To realize this objective, we initially construct a Text-Expression Alignment Dataset (TEAD), wherein each facial expression is paired with several prompt-like descriptions. We propose an innovative automatic annotation method, supported by CahtGPT, to expedite the dataset construction, thereby eliminating the substantial expense of manual annotation.
Following this, we utilize TEAD to train a CLIP-based model, termed ExpCLIP, which encodes text and facial expressions into semantically aligned style embeddings. The embeddings are subsequently integrated into the facial animation generator to yield expressive and controllable facial animations. Given the limited diversity of facial emotions in existing speech-driven facial animation training data, we further introduce an effective Expression Prompt Augmentation (EPA) mechanism to enable the animation generator to support unprecedented richness in style control.
Comprehensive experiments illustrate that our method accomplishes expressive facial animation generation and offers enhanced flexibility in effectively conveying the desired style.

----

## [846] Learning Image Demoiréing from Unpaired Real Data

**Authors**: *Yunshan Zhong, Yuyao Zhou, Yuxin Zhang, Fei Chao, Rongrong Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28595](https://doi.org/10.1609/aaai.v38i7.28595)

**Abstract**:

This paper focuses on addressing the issue of image demoiréing. Unlike the large volume of existing studies that rely on learning from paired real data, we attempt to learn a demoiréing model from unpaired real data, i.e., moiré images associated with irrelevant clean images. The proposed method, referred to as Unpaired Demoiréing(UnDeM), synthesizes pseudo moiré images from unpaired datasets, generating pairs with clean images for training demoiréing models. To achieve this, we divide real moiré images into patches and group them in compliance with their moiré complexity. We introduce a novel moiré generation framework to synthesize moiré images with diverse moiré features, resembling real moiré patches, and details akin to real moiré-free images. Additionally, we introduce an adaptive denoise method to eliminate the low-quality pseudo moiré images that adversely impact the learning of demoiréing models. We conduct extensive experiments on the commonly-used FHDMi and UHDM datasets. Results manifest that our UnDeM performs better than existing methods when using existing demoiréing models such as MBCNN and ESDNet-L. Code: https://github.com/zysxmu/UnDeM.

----

## [847] Lifting by Image - Leveraging Image Cues for Accurate 3D Human Pose Estimation

**Authors**: *Feng Zhou, Jianqin Yin, Peiyang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28596](https://doi.org/10.1609/aaai.v38i7.28596)

**Abstract**:

The "lifting from 2D pose" method has been the dominant approach to 3D Human Pose Estimation (3DHPE) due to the powerful visual analysis ability of 2D pose estimators. Widely known, there exists a depth ambiguity problem when estimating solely from 2D pose, where one 2D pose can be mapped to multiple 3D poses. Intuitively, the rich semantic and texture information in images can contribute to a more accurate "lifting" procedure. Yet, existing research encounters two primary challenges. Firstly, the distribution of image data in 3D motion capture datasets is too narrow because of the laboratorial environment, which leads to poor generalization ability of methods trained with image information. Secondly, effective strategies for leveraging image information are lacking. In this paper, we give new insight into the cause of poor generalization problems and the effectiveness of image features. Based on that, we propose an advanced framework. Specifically, the framework consists of two stages. First, we enable the keypoints to query and select the beneficial features from all image patches. To reduce the keypoints attention to inconsequential background features, we design a novel Pose-guided Transformer Layer, which adaptively limits the updates to unimportant image patches. Then, through a designed Adaptive Feature Selection Module, we prune less significant image patches from the feature map. In the second stage, we allow the keypoints to further emphasize the retained critical image features. This progressive learning approach prevents further training on insignificant image features. Experimental results show that our model achieves state-of-the-art performance on both the Human3.6M dataset and the MPI-INF-3DHP dataset.

----

## [848] NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models

**Authors**: *Gengze Zhou, Yicong Hong, Qi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28597](https://doi.org/10.1609/aaai.v38i7.28597)

**Abstract**:

Trained with an unprecedented scale of data, large language models (LLMs) like ChatGPT and GPT-4 exhibit the emergence of significant reasoning abilities from model scaling. Such a trend underscored the potential of training LLMs with unlimited language data, advancing the development of a universal embodied agent. In this work, we introduce the NavGPT, a purely LLM-based instruction-following navigation agent, to reveal the reasoning capability of GPT models in complex embodied scenes by performing zero-shot sequential action prediction for vision-and-language navigation (VLN). At each step, NavGPT takes the textual descriptions of visual observations, navigation history, and future explorable directions as inputs to reason the agent's current status, and makes the decision to approach the target. Through comprehensive experiments, we demonstrate NavGPT can explicitly perform high-level planning for navigation, including decomposing instruction into sub-goals, integrating commonsense knowledge relevant to navigation task resolution, identifying landmarks from observed scenes, tracking navigation progress, and adapting to exceptions with plan adjustment. Furthermore, we show that LLMs is capable of generating high-quality navigational instructions from observations and actions along a path, as well as drawing accurate top-down metric trajectory given the agent's navigation history. Despite the performance of using NavGPT to zero-shot R2R tasks still falling short of trained models, we suggest adapting multi-modality inputs for LLMs to use as visual navigation agents and applying the explicit reasoning of LLMs to benefit learning-based models. Code is available at: https://github.com/GengzeZhou/NavGPT.

----

## [849] Novel Class Discovery in Chest X-rays via Paired Images and Text

**Authors**: *Jiaying Zhou, Yang Liu, Qingchao Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28598](https://doi.org/10.1609/aaai.v38i7.28598)

**Abstract**:

Novel class discover(NCD) aims to identify new classes undefined during model training phase with the help of knowledge of known classes. Many methods have been proposed and notably boosted performance of NCD in natural images. However, there has been no work done in discovering new classes based on medical images and disease categories, which is crucial for understanding and diagnosing specific diseases. Moreover, most of the existing methods only utilize information from image modality and use labels as the only supervisory information. In this paper, we propose a multi-modal novel class discovery method based on paired images and text, inspired by the low classification accuracy of chest X-ray images and the relatively higher accuracy of the paired text. Specifically, we first pretrain the image encoder and text encoder with multi-modal contrastive learning on the entire dataset and then we generate pseudo-labels separately on the image branch and text branch. We utilize intra-modal consistency to assess the quality of pseudo-labels and adjust the weights of the pseudo-labels from both branches to generate the ultimate pseudo-labels for training. Experiments on eight subset splits of MIMIC-CXR-JPG dataset show that our method improves the clustering performance of unlabeled classes by about 10% on average compared to state-of-the-art methods. Code is available at: https://github.com/zzzzzzzzjy/MMNCD-main.

----

## [850] AMSP-UOD: When Vortex Convolution and Stochastic Perturbation Meet Underwater Object Detection

**Authors**: *Jingchun Zhou, Zongxin He, Kin-Man Lam, Yudong Wang, Weishi Zhang, Chunle Guo, Chongyi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28599](https://doi.org/10.1609/aaai.v38i7.28599)

**Abstract**:

In this paper, we present a novel Amplitude-Modulated Stochastic Perturbation and Vortex Convolutional Network, AMSP-UOD, designed for underwater object detection. AMSP-UOD specifically addresses the impact of non-ideal imaging factors on detection accuracy in complex underwater environments. To mitigate the influence of noise on object detection performance, we propose AMSP Vortex Convolution (AMSP-VConv) to disrupt the noise distribution, enhance feature extraction capabilities, effectively reduce parameters, and improve network robustness. We design the Feature Association Decoupling Cross Stage Partial (FAD-CSP) module, which strengthens the association of long and short range features, improving the network performance in complex underwater environments. Additionally, our sophisticated post-processing method, based on non-maximum suppression with aspect-ratio similarity thresholds, optimizes detection in dense scenes, such as waterweed and schools of fish, improving object detection accuracy. Extensive experiments on the URPC and RUOD datasets demonstrate that our method outperforms existing state-of-the-art methods in terms of accuracy and noise immunity. AMSP-UOD proposes an innovative solution with the potential for real-world applications. Our code is available at https://github.com/zhoujingchun03/AMSP-UOD.

----

## [851] SOGDet: Semantic-Occupancy Guided Multi-View 3D Object Detection

**Authors**: *Qiu Zhou, Jinming Cao, Hanchao Leng, Yifang Yin, Yu Kun, Roger Zimmermann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28600](https://doi.org/10.1609/aaai.v38i7.28600)

**Abstract**:

In the field of autonomous driving, accurate and comprehensive perception of the 3D environment is crucial.
Bird's Eye View (BEV) based methods have emerged as a promising solution for 3D object detection using multi-view images as input.
However, existing 3D object detection methods often ignore the physical context in the environment, such as sidewalk and vegetation, resulting in sub-optimal performance. 
In this paper, we propose a novel approach called SOGDet (Semantic-Occupancy Guided Multi-view 3D Object Detection), that leverages a 3D semantic-occupancy branch to improve the accuracy of 3D object detection.  
In particular, the physical context modeled by semantic occupancy helps the detector to perceive the scenes in a more holistic view.
Our SOGDet is flexible to use and can be seamlessly integrated with most existing BEV-based methods.
To evaluate its effectiveness, we apply this approach to several state-of-the-art baselines and conduct extensive experiments on the exclusive nuScenes dataset.
Our results show that SOGDet consistently enhance the performance of three baseline methods in terms of nuScenes Detection Score (NDS) and mean Average Precision (mAP). 
This indicates that the combination of 3D object detection and 3D semantic occupancy leads to a more comprehensive perception of the 3D environment, thereby aiding build more robust autonomous driving systems.
The codes are available at: https://github.com/zhouqiu/SOGDet.

----

## [852] Test-Time Adaptation via Style and Structure Guidance for Histological Image Registration

**Authors**: *Shenglong Zhou, Zhiwei Xiong, Feng Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28601](https://doi.org/10.1609/aaai.v38i7.28601)

**Abstract**:

Image registration plays a crucial role in histological image analysis, encompassing tasks like multi-modality fusion and disease grading. 
Traditional registration methods optimize objective functions for each image pair, yielding reliable accuracy but demanding heavy inference burdens.
Recently, learning-based registration methods utilize networks to learn the optimization process during training and apply a one-step forward process during testing. 
While these methods offer promising registration performance with reduced inference time, they remain sensitive to appearance variances and local structure changes commonly encountered in histological image registration scenarios.
In this paper, for the first time, we propose a novel test-time adaptation method for histological image registration, aiming to improve the generalization ability of learning-based methods. 
Specifically, we design two operations, style guidance and shape guidance, for the test-time adaptation process. 
The former leverages style representations encoded by feature statistics to address the issue of appearance variances, while the latter incorporates shape representations encoded by HOG features to improve registration accuracy in regions with structural changes.
Furthermore, we consider the continuity of the model during the test-time adaptation process.
Different from the previous methods initialized by a given trained model, we introduce a smoothing strategy to leverage historical models for better generalization. 
We conduct experiments with several representative learning-based backbones on the public histological dataset, demonstrating the superior registration performance of our test-time adaptation method.

----

## [853] Reducing Spatial Fitting Error in Distillation of Denoising Diffusion Models

**Authors**: *Shengzhe Zhou, Zejian Li, Shengyuan Zhang, Lefan Hou, Changyuan Yang, Guang Yang, Zhiyuan Yang, Lingyun Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28602](https://doi.org/10.1609/aaai.v38i7.28602)

**Abstract**:

Denoising Diffusion models have exhibited remarkable capabilities in image generation. However, generating high-quality samples requires a large number of iterations. Knowledge distillation for diffusion models is an effective method to address this limitation with a shortened sampling process but causes degraded generative quality. Based on our analysis with bias-variance decomposition and experimental observations, we attribute the degradation to the spatial fitting error occurring in the training of both the teacher and student model in the distillation. Accordingly, we propose Spatial Fitting-Error Reduction Distillation model (SFERD). SFERD utilizes attention guidance from the teacher model and a designed semantic gradient predictor to reduce the student's fitting error. Empirically, our proposed model facilitates high-quality sample generation in a few function evaluations. We achieve an FID of 5.31 on CIFAR-10 and 9.39 on ImageNet 64x64 with only one step, outperforming existing diffusion methods. Our study provides a new perspective on diffusion distillation by highlighting the intrinsic denoising ability of models.

----

## [854] SAMFlow: Eliminating Any Fragmentation in Optical Flow with Segment Anything Model

**Authors**: *Shili Zhou, Ruian He, Weimin Tan, Bo Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28603](https://doi.org/10.1609/aaai.v38i7.28603)

**Abstract**:

Optical Flow Estimation aims to find the 2D dense motion field between two frames. Due to the limitation of model structures and training datasets, existing methods often rely too much on local clues and ignore the integrity of objects, resulting in fragmented motion estimation. Through theoretical analysis, we find the pre-trained large vision models are helpful in optical flow estimation, and we notice that the recently famous Segment Anything Model (SAM) demonstrates a strong ability to segment complete objects, which is suitable for solving the fragmentation problem. We thus propose a solution to embed the frozen SAM image encoder into FlowFormer to enhance object perception. To address the challenge of in-depth utilizing SAM in non-segmentation tasks like optical flow estimation, we propose an Optical Flow Task-Specific Adaption scheme, including a Context Fusion Module to fuse the SAM encoder with the optical flow context encoder, and a Context Adaption Module to adapt the SAM features for optical flow task with Learned Task-Specific Embedding. Our proposed SAMFlow model reaches 0.86/2.10 clean/final EPE and 3.55/12.32 EPE/F1-all on Sintel and KITTI-15 training set, surpassing Flowformer by 8.5%/9.9% and 13.2%/16.3%. Furthermore, our model achieves state-of-the-art performance on the Sintel and KITTI-15 benchmarks, ranking #1 among all two-frame methods on Sintel clean pass.

----

## [855] Efficient Lightweight Image Denoising with Triple Attention Transformer

**Authors**: *Yubo Zhou, Jin Lin, Fangchen Ye, Yanyun Qu, Yuan Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28604](https://doi.org/10.1609/aaai.v38i7.28604)

**Abstract**:

Transformer has shown outstanding performance on image denoising, but the existing Transformer methods for image denoising are with large model sizes and high computational complexity, which is unfriendly to resource-constrained devices. In this paper, we propose a Lightweight Image Denoising Transformer method (LIDFormer) based on Triple Multi-Dconv Head Transposed Attention (TMDTA) to boost computational efficiency. LIDFormer first implements Discrete Wavelet Transform (DWT), which transforms the input image into a low-frequency space, greatly reducing the computational complexity of image denoising. However, the low-frequency image lacks fine-feature information, which degrades the denoising performance. To handle this problem, we introduce the Complementary Periodic Feature Reusing (CPFR) scheme for aggregating the shallow-layer features and the deep-layer features. Furthermore, TMDTA is proposed to integrate global context along three dimensions, thereby enhancing the ability of global feature representation. Note that our method can be applied as a pipeline for both convolutional neural networks and Transformers. Extensive experiments on several benchmarks demonstrate that the proposed LIDFormer achieves a better trade-off between high performance and low computational complexity on real-world image denoising tasks.

----

## [856] Intentional Evolutionary Learning for Untrimmed Videos with Long Tail Distribution

**Authors**: *Yuxi Zhou, Xiujie Wang, Jianhua Zhang, Jiajia Wang, Jie Yu, Hao Zhou, Yi Gao, Shengyong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28605](https://doi.org/10.1609/aaai.v38i7.28605)

**Abstract**:

Human intention understanding in untrimmed videos aims to watch a natural video and predict what the person’s intention is. Currently, exploration of predicting human intentions in untrimmed videos is far from enough. On the one hand, untrimmed videos with mixed actions and backgrounds have a significant long-tail distribution with concept drift characteristics. On the other hand, most methods can only perceive instantaneous intentions, but cannot determine the evolution of intentions. To solve the above challenges, we propose a loss based on Instance Confidence and Class Accuracy (ICCA), which aims to alleviate the prediction bias caused by the long-tail distribution with concept drift characteristics in video streams. In addition, we propose an intention-oriented evolutionary learning method to determine the intention evolution pattern (from what action to what action) and the time of evolution (when the action evolves). We conducted extensive experiments on two untrimmed video datasets (THUMOS14 and ActivityNET v1.3), and our method has achieved excellent results compared to SOTA methods. The code and supplementary materials are available at https://github.com/Jennifer123www/UntrimmedVideo.

----

## [857] SasWOT: Real-Time Semantic Segmentation Architecture Search WithOut Training

**Authors**: *Chendi Zhu, Lujun Li, Yuli Wu, Zhengxing Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28606](https://doi.org/10.1609/aaai.v38i7.28606)

**Abstract**:

In this paper, we present SasWOT, the first training-free Semantic segmentation Architecture Search (SAS) framework via an auto-discovery proxy. Semantic segmentation is widely used in many real-time applications. For fast inference and memory efficiency, Previous SAS seeks the optimal segmenter by differentiable or RL Search. However, the significant computational costs of these training-based SAS limit their practical usage. To improve the search efficiency, we explore the training-free route but empirically observe that the existing zero-cost proxies designed on the classification task are sub-optimal on the segmentation benchmark. To address this challenge, we develop a customized proxy search framework for SAS tasks to augment its predictive capabilities. Specifically, we design the proxy search space based on the some observations: (1) different inputs of segmenter statistics can be well combined; (2) some basic operators can effectively improve the correlation. Thus, we build computational graphs with multiple statistics as inputs and different advanced basis arithmetic as the primary operations to represent candidate proxies. Then, we employ an evolutionary algorithm to crossover and mutate the superior candidates in the population based on correlation evaluation. Finally, based on the searched proxy, we perform the segmenter search without candidate training. In this way, SasWOT not only enables automated proxy optimization for SAS tasks but also achieves significant search acceleration before the retrain stage. Extensive experiments on Cityscapes and CamVid datasets demonstrate that SasWOT achieves superior trade-off between accuracy and speed over several state-of-the-art techniques. More remarkably, on Cityscapes dataset, SasWOT achieves the performance of 71.3% mIoU with the speed of 162 FPS.

----

## [858] Enhance Sketch Recognition's Explainability via Semantic Component-Level Parsing

**Authors**: *Guangming Zhu, Siyuan Wang, Tianci Wu, Liang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28607](https://doi.org/10.1609/aaai.v38i7.28607)

**Abstract**:

Free-hand sketches are appealing for humans as a universal tool to depict the visual world. Humans can recognize varied sketches of a category easily by identifying the concurrence and layout of the intrinsic semantic components of the category, since humans draw free-hand sketches based a common consensus that which types of semantic components constitute each sketch category. For example, an airplane should at least have a fuselage and wings. Based on this analysis, a semantic component-level memory module is constructed and embedded in the proposed structured sketch recognition network in this paper. The memory keys representing semantic components of each sketch category can be self-learned and enhance the recognition network's explainability. Our proposed networks can deal with different situations of sketch recognition, i.e., with or without semantic components labels of strokes. Experiments on the SPG and SketchIME datasets demonstrate the memory module's flexibility and the recognition network's explainability. The code and data are available at https://github.com/GuangmingZhu/SketchESC.

----

## [859] Learning Discriminative Noise Guidance for Image Forgery Detection and Localization

**Authors**: *Jiaying Zhu, Dong Li, Xueyang Fu, Gang Yang, Jie Huang, Aiping Liu, Zheng-Jun Zha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28608](https://doi.org/10.1609/aaai.v38i7.28608)

**Abstract**:

This study introduces a new method for detecting and localizing image forgery by focusing on manipulation traces within the noise domain. We posit that nearly invisible noise in RGB images carries tampering traces, useful for distinguishing and locating forgeries. However, the advancement of tampering technology complicates the direct application of noise for forgery detection, as the noise inconsistency between forged and authentic regions is not fully exploited. To tackle this, we develop a two-step discriminative noise-guided approach to explicitly enhance the representation and use of noise inconsistencies, thereby fully exploiting noise information to improve the accuracy and robustness of forgery detection. Specifically, we first enhance the noise discriminability of forged regions compared to authentic ones using a de-noising network and a statistics-based constraint. Then, we merge a model-driven guided filtering mechanism with a data-driven attention mechanism to create a learnable and differentiable noise-guided filter. This sophisticated filter allows us to maintain the edges of forged regions learned from the noise. Comprehensive experiments on multiple datasets demonstrate that our method can reliably detect and localize forgeries, surpassing existing state-of-the-art methods.

----

## [860] Video Frame Prediction from a Single Image and Events

**Authors**: *Juanjuan Zhu, Zhexiong Wan, Yuchao Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28609](https://doi.org/10.1609/aaai.v38i7.28609)

**Abstract**:

Recently, the task of Video Frame Prediction (VFP), which predicts future video frames from previous ones through extrapolation, has made remarkable progress. However, the performance of existing VFP methods is still far from satisfactory due to the fixed framerate video used: 1) they have difficulties in handling complex dynamic scenes; 2) they cannot predict future frames with flexible prediction time intervals. The event cameras can record the intensity changes asynchronously with a very high temporal resolution, which provides rich dynamic information about the observed scenes. In this paper, we propose to predict video frames from a single image and the following events, which can not only handle complex dynamic scenes but also predict future frames with flexible prediction time intervals. First, we introduce a symmetrical cross-modal attention augmentation module to enhance the complementary information between images and events. Second, we propose to jointly achieve optical flow estimation and frame generation by combining the motion information of events and the semantic information of the image, then inpainting the holes produced by forward warping to obtain an ideal prediction frame. Based on these, we propose a lightweight pyramidal coarse-to-fine model that can predict a 720P frame within 25 ms. Extensive experiments show that our proposed model significantly outperforms the state-of-the-art frame-based and event-based VFP methods and has the fastest runtime. Code is available at https://npucvr.github.io/VFPSIE/.

----

## [861] Finding Visual Saliency in Continuous Spike Stream

**Authors**: *Lin Zhu, Xianzhang Chen, Xiao Wang, Hua Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28610](https://doi.org/10.1609/aaai.v38i7.28610)

**Abstract**:

As a bio-inspired vision sensor, the spike camera emulates the operational principles of the fovea, a compact retinal region, by employing spike discharges to encode the accumulation of per-pixel luminance intensity. Leveraging its high temporal resolution and bio-inspired neuromorphic design, the spike camera holds significant promise for advancing computer vision applications. Saliency detection mimic the behavior of human beings and capture the most salient region from the scenes. In this paper, we investigate the visual saliency in the continuous spike stream for the first time. To effectively process the binary spike stream, we propose a Recurrent Spiking Transformer (RST) framework, which is based on a full spiking neural network. Our framework enables the extraction of spatio-temporal features from the continuous spatio-temporal spike stream while maintaining low power consumption. To facilitate the training and validation of our proposed model, we build a comprehensive real-world spike-based visual saliency dataset, enriched with numerous light conditions. Extensive experiments demonstrate the superior performance of our Recurrent Spiking Transformer framework in comparison to other spike neural network-based methods. Our framework exhibits a substantial margin of improvement in capturing and highlighting visual saliency in the spike stream, which not only provides a new perspective for spike-based saliency segmentation but also shows a new paradigm for full SNN-based transformer models. The code and dataset are available at https://github.com/BIT-Vision/SVS.

----

## [862] SEER: Backdoor Detection for Vision-Language Models through Searching Target Text and Image Trigger Jointly

**Authors**: *Liuwan Zhu, Rui Ning, Jiang Li, Chunsheng Xin, Hongyi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28611](https://doi.org/10.1609/aaai.v38i7.28611)

**Abstract**:

This paper proposes SEER, a novel backdoor detection algorithm for vision-language models, addressing the gap in the literature on multi-modal backdoor detection. While backdoor detection in single-modal models has been well studied, the investigation of such defenses in multi-modal models remains limited. Existing backdoor defense mechanisms cannot be directly applied to multi-modal settings due to their increased complexity and search space explosion. In this paper, we propose to detect backdoors in vision-language models by jointly searching image triggers and malicious target texts in feature space shared by vision and language modalities. Our extensive experiments demonstrate that SEER can achieve over 92% detection rate on backdoor detection in vision-language models in various settings without accessing training data or knowledge of downstream tasks.

----

## [863] Text Image Inpainting via Global Structure-Guided Diffusion Models

**Authors**: *Shipeng Zhu, Pengfei Fang, Chenjie Zhu, Zuoyan Zhao, Qiang Xu, Hui Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28612](https://doi.org/10.1609/aaai.v38i7.28612)

**Abstract**:

Real-world text can be damaged by corrosion issues caused by environmental or human factors, which hinder the preservation of the complete styles of texts, e.g., texture and structure. These corrosion issues, such as graffiti signs and incomplete signatures, bring difficulties in understanding the texts, thereby posing significant challenges to downstream applications, e.g., scene text recognition and signature identification. Notably, current inpainting techniques often fail to adequately address this problem and have difficulties restoring accurate text images along with reasonable and consistent styles. Formulating this as an open problem of text image inpainting, this paper aims to build a benchmark to facilitate its study. In doing so, we establish two specific text inpainting datasets which contain scene text images and handwritten text images, respectively. Each of them includes images revamped by real-life and synthetic datasets, featuring pairs of original images, corrupted images, and other assistant information. On top of the datasets, we further develop a novel neural framework, Global Structure-guided Diffusion Model (GSDM), as a potential solution. Leveraging the global structure of the text as a prior, the proposed GSDM develops an efficient diffusion model to recover clean texts. The efficacy of our approach is demonstrated by thorough empirical study, including a substantial boost in both recognition accuracy and image quality. These findings not only highlight the effectiveness of our method but also underscore its potential to enhance the broader field of text image understanding and processing. Code and datasets are available at: https://github.com/blackprotoss/GSDM.

----

## [864] Rethinking Mesh Watermark: Towards Highly Robust and Adaptable Deep 3D Mesh Watermarking

**Authors**: *Xingyu Zhu, Guanhui Ye, Xiapu Luo, Xuetao Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28613](https://doi.org/10.1609/aaai.v38i7.28613)

**Abstract**:

The goal of 3D mesh watermarking is to embed the message in 3D meshes that can withstand various attacks imperceptibly and reconstruct the message accurately from watermarked meshes. The watermarking algorithm is supposed to withstand multiple attacks, and the complexity should not grow significantly with the mesh size. Unfortunately, previous methods are less robust against attacks and lack of adaptability. In this paper, we propose a robust and adaptable deep 3D mesh watermarking Deep3DMark that leverages attention-based convolutions in watermarking tasks to embed binary messages in vertex distributions without texture assistance. Furthermore, our Deep3DMark exploits the property that simplified meshes inherit similar relations from the original ones, where the relation is the offset vector directed from one vertex to its neighbor. By doing so, our method can be trained on simplified meshes but remains effective on large size meshes (size adaptable) and unseen categories of meshes (geometry adaptable). Extensive experiments demonstrate our method remains efficient and effective even if the mesh size is 190× increased. Under mesh attacks, Deep3DMark achieves 10%∼50% higher accuracy than traditional methods, and 2× higher SNR and 8% higher accuracy than previous DNN-based methods.

----

## [865] Boosting Few-Shot Learning via Attentive Feature Regularization

**Authors**: *Xingyu Zhu, Shuo Wang, Jinda Lu, Yanbin Hao, Haifeng Liu, Xiangnan He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28614](https://doi.org/10.1609/aaai.v38i7.28614)

**Abstract**:

Few-shot learning (FSL) based on manifold regularization aims to improve the recognition capacity of novel objects with limited training samples by mixing two samples from different categories with a blending factor. However, this mixing operation weakens the feature representation due to the linear interpolation and the overlooking of the importance of specific channels. To solve these issues, this paper proposes attentive feature regularization (AFR) which aims to improve the feature representativeness and discriminability. In our approach, we first calculate the relations between different categories of semantic labels to pick out the related features used for regularization. Then, we design two attention-based calculations at both the instance and channel levels. These calculations enable the regularization procedure to focus on two crucial aspects: the feature complementarity through adaptive interpolation in related categories and the emphasis on specific feature channels. Finally, we combine these regularization strategies to significantly improve the classifier performance. Empirical studies on several popular FSL benchmarks demonstrate the effectiveness of AFR, which improves the recognition accuracy of novel categories without the need to retrain any feature extractor, especially in the 1-shot setting. Furthermore, the proposed AFR can seamlessly integrate into other FSL methods to improve classification performance.

----

## [866] Memory-Efficient Prompt Tuning for Incremental Histopathology Classification

**Authors**: *Yu Zhu, Kang Li, Lequan Yu, Pheng-Ann Heng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28615](https://doi.org/10.1609/aaai.v38i7.28615)

**Abstract**:

Recent studies have made remarkable progress in histopathology classification. Based on current successes, contemporary works proposed to further upgrade the model towards a more generalizable and robust direction through incrementally learning from the sequentially delivered domains. Unlike previous parameter isolation based approaches that usually demand massive computation resources during model updating, we present a memory-efficient prompt tuning framework to cultivate model generalization potential in economical memory cost. For each incoming domain, we reuse the existing parameters of the initial classification model and attach lightweight trainable prompts into it for customized tuning. Considering the domain heterogeneity, we perform decoupled prompt tuning, where we adopt a domain-specific prompt for each domain to independently investigate its distinctive characteristics, and one domain-invariant prompt shared across all domains to continually explore the common content embedding throughout time. All domain-specific prompts will be appended to the prompt bank and isolated from further changes to prevent forgetting the distinctive features of early-seen domains. While the domain-invariant prompt will be passed on and iteratively evolve by style-augmented prompt refining to improve model generalization capability over time. In specific, we construct a graph with existing prompts and build a style-augmented graph attention network to guide the domain-invariant prompt exploring the overlapped latent embedding among all delivered domains for more domain-generic representations. We have extensively evaluated our framework with two histopathology tasks, i.e., breast cancer metastasis classification and epithelium-stroma tissue classification, where our approach yielded superior performance and memory efficiency over the competing methods.

----

## [867] SPGroup3D: Superpoint Grouping Network for Indoor 3D Object Detection

**Authors**: *Yun Zhu, Le Hui, Yaqi Shen, Jin Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28616](https://doi.org/10.1609/aaai.v38i7.28616)

**Abstract**:

Current 3D object detection methods for indoor scenes mainly follow the voting-and-grouping strategy to generate proposals. However, most methods utilize instance-agnostic groupings, such as ball query, leading to inconsistent semantic information and inaccurate regression of the proposals. To this end, we propose a novel superpoint grouping network for indoor anchor-free one-stage 3D object detection. Specifically, we first adopt an unsupervised manner to partition raw point clouds into superpoints, areas with semantic consistency and spatial similarity. Then, we design a geometry-aware voting module that adapts to the centerness in anchor-free detection by constraining the spatial relationship between superpoints and object centers. Next, we present a superpoint-based grouping module to explore the consistent representation within proposals. This module includes a superpoint attention layer to learn feature interaction between neighboring superpoints, and a superpoint-voxel fusion layer to propagate the superpoint-level information to the voxel level. Finally, we employ effective multiple matching to capitalize on the dynamic receptive fields of proposals based on superpoints during the training.  Experimental results demonstrate our method achieves state-of-the-art performance on ScanNet V2, SUN RGB-D, and S3DIS datasets in the indoor one-stage 3D object detection. Source code is available at https://github.com/zyrant/SPGroup3D.

----

## [868] SEIT: Structural Enhancement for Unsupervised Image Translation in Frequency Domain

**Authors**: *Zhifeng Zhu, Yaochen Li, Yifan Li, Jinhuo Yang, Peijun Chen, Yuehu Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28617](https://doi.org/10.1609/aaai.v38i7.28617)

**Abstract**:

For the task of unsupervised image translation, transforming the image style while preserving its original structure remains challenging. In this paper, we propose an unsupervised image translation method with structural enhancement in frequency domain named SEIT. Specifically, a frequency dynamic adaptive (FDA) module is designed for image style transformation that can well transfer the image style while maintaining its overall structure by decoupling the image content and style in frequency domain. Moreover, a wavelet-based structure enhancement (WSE) module is proposed to improve the intermediate translation results by matching the high-frequency information, thus enriching the structural details. Furthermore, a multi-scale network architecture is designed to extract the domain-specific information using image-independent encoders for both the source and target domains. The extensive experimental results well demonstrate the effectiveness of the proposed method.

----

## [869] A Pre-convolved Representation for Plug-and-Play Neural Illumination Fields

**Authors**: *Yiyu Zhuang, Qi Zhang, Xuan Wang, Hao Zhu, Ying Feng, Xiaoyu Li, Ying Shan, Xun Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28618](https://doi.org/10.1609/aaai.v38i7.28618)

**Abstract**:

Recent advances in implicit neural representation have demonstrated the ability to recover detailed geometry and material from multi-view images. However, the use of simplified lighting models such as environment maps to represent non-distant illumination, or using a network to fit indirect light modeling without a solid basis, can lead to an undesirable decomposition between lighting and material. To address this, we propose a fully differentiable framework named Neural Illumination Fields (NeIF) that uses radiance fields as a lighting model to handle complex lighting in a physically based way. Together with integral lobe encoding for roughness-adaptive specular lobe and leveraging the pre-convolved background for accurate decomposition, the proposed method represents a significant step towards integrating physically based rendering into the NeRF representation. The experiments demonstrate the superior performance of novel-view rendering compared to previous works, and the capability to re-render objects under arbitrary NeRF-style environments opens up exciting possibilities for bridging the gap between virtual and real-world scenes.

----

## [870] IPRemover: A Generative Model Inversion Attack against Deep Neural Network Fingerprinting and Watermarking

**Authors**: *Wei Zong, Yang-Wai Chow, Willy Susilo, Joonsang Baek, Jongkil Kim, Seyit Camtepe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28619](https://doi.org/10.1609/aaai.v38i7.28619)

**Abstract**:

Training Deep Neural Networks (DNNs) can be expensive when data is difficult to obtain or labeling them requires significant domain expertise. Hence, it is crucial that the Intellectual Property (IP) of DNNs trained on valuable data be protected against IP infringement. DNN fingerprinting and watermarking are two lines of work in DNN IP protection. Recently proposed DNN fingerprinting techniques are able to detect IP infringement while preserving model performance by relying on the key assumption that the decision boundaries of independently trained models are intrinsically different from one another. In contrast, DNN watermarking embeds a watermark in a model and verifies IP infringement if an identical or similar watermark is extracted from a suspect model. The techniques deployed in fingerprinting and watermarking vary significantly because their underlying mechanisms are different. From an adversary's perspective, a successful IP removal attack should defeat both fingerprinting and watermarking. However, to the best of our knowledge, there is no work on such attacks in the literature yet. In this paper, we fill this gap by presenting an IP removal attack that can defeat both fingerprinting and watermarking. We consider the challenging data-free scenario whereby all data is inverted from the victim model. Under this setting, a stolen model only depends on the victim model. Experimental results demonstrate the success of our attack in defeating state-of-the-art DNN fingerprinting and watermarking techniques. This work reveals a novel attack surface that exploits generative model inversion attacks to bypass DNN IP defenses. This threat must be addressed by future defenses for reliable IP protection.

----

## [871] DiffBEV: Conditional Diffusion Model for Bird's Eye View Perception

**Authors**: *Jiayu Zou, Kun Tian, Zheng Zhu, Yun Ye, Xingang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28620](https://doi.org/10.1609/aaai.v38i7.28620)

**Abstract**:

BEV perception is of great importance in the field of autonomous driving, serving as the cornerstone of planning, controlling, and motion prediction. The quality of the BEV feature highly affects the performance of BEV perception. However, taking the noises in camera parameters and LiDAR scans into consideration, we usually obtain BEV representation with harmful noises. Diffusion models naturally have the ability to denoise noisy samples to the ideal data, which motivates us to utilize the diffusion model to get a better BEV representation. In this work, we propose an end-to-end framework, named DiffBEV, to exploit the potential of diffusion model to generate a more comprehensive BEV representation. To the best of our knowledge, we are the first to apply diffusion model to BEV perception. In practice, we design three types of conditions to guide the training of the diffusion model which denoises the coarse samples and refines the semantic feature in a progressive way. What's more, a cross-attention module is leveraged to fuse the context of BEV feature and the semantic content of conditional diffusion model. DiffBEV achieves a 25.9% mIoU on the nuScenes dataset, which is 6.2% higher than the best-performing existing approach. Quantitative and qualitative results on multiple benchmarks demonstrate the effectiveness of DiffBEV in BEV semantic segmentation and 3D object detection tasks.

----

## [872] Cross-Covariate Gait Recognition: A Benchmark

**Authors**: *Shinan Zou, Chao Fan, Jianbo Xiong, Chuanfu Shen, Shiqi Yu, Jin Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28621](https://doi.org/10.1609/aaai.v38i7.28621)

**Abstract**:

Gait datasets are essential for gait research. However, this paper observes that present benchmarks, whether conventional constrained or emerging real-world datasets, fall short regarding covariate diversity. To bridge this gap, we undertake an arduous 20-month effort to collect a cross-covariate gait recognition (CCGR) dataset. The CCGR dataset has 970 subjects and about 1.6 million sequences; almost every subject has 33 views and 53 different covariates. Compared to existing datasets, CCGR has both population and individual-level diversity. In addition, the views and covariates are well labeled, enabling the analysis of the effects of different factors. CCGR provides multiple types of gait data, including RGB, parsing, silhouette, and pose, offering researchers a comprehensive resource for exploration. In order to delve deeper into addressing cross-covariate gait recognition, we propose parsing-based gait recognition (ParsingGait) by utilizing the newly proposed parsing data. We have conducted extensive experiments. Our main results show: 1) Cross-covariate emerges as a pivotal challenge for practical applications of gait recognition. 2) ParsingGait demonstrates remarkable potential for further advancement. 3) Alarmingly, existing SOTA methods achieve less than 43% accuracy on the CCGR, highlighting the urgency of exploring cross-covariate gait recognition. Link: https://github.com/ShinanZou/CCGR.

----

## [873] Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks

**Authors**: *Siyu Zou, Jiji Tang, Yiyi Zhou, Jing He, Chaoyi Zhao, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28622](https://doi.org/10.1609/aaai.v38i7.28622)

**Abstract**:

Diffusion-based Image Editing (DIE) is an emerging research hot-spot, which often applies a semantic mask to control the target area for diffusion-based editing. However, most existing solutions obtain these masks via manual operations or off-line processing, greatly reducing their efficiency. In this paper, we propose a novel and efficient image editing method for Text-to-Image (T2I) diffusion models, termed Instant Diffusion Editing (InstDiffEdit). In particular, InstDiffEdit aims to employ the cross-modal attention ability of existing diffusion models to achieve instant mask guidance during the diffusion steps. To reduce the noise of attention maps and realize the full automatics, we equip InstDiffEdit with a training-free refinement scheme to adaptively aggregate the attention distributions for the automatic yet accurate mask generation. Meanwhile, to supplement the existing evaluations of DIE, we propose a new benchmark called Editing-Mask to examine the mask accuracy and local editing ability of existing methods. To validate InstDiffEdit, we also conduct extensive experiments on ImageNet and Imagen, and compare it with a bunch of the SOTA methods. The experimental results show that InstDiffEdit not only outperforms the SOTA methods in both image quality and editing results, but also has a much faster inference speed, i.e., +5 to +6 times. Our code available at https://anonymous.4open.science/r/InstDiffEdit-C306

----

## [874] VQCNIR: Clearer Night Image Restoration with Vector-Quantized Codebook

**Authors**: *Wenbin Zou, Hongxia Gao, Tian Ye, Liang Chen, Weipeng Yang, Shasha Huang, Hongsheng Chen, Sixiang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28623](https://doi.org/10.1609/aaai.v38i7.28623)

**Abstract**:

Night photography often struggles with challenges like low light and blurring, stemming from dark environments and prolonged exposures. Current methods either disregard priors and directly fitting end-to-end networks, leading to inconsistent illumination, or rely on unreliable handcrafted priors to constrain the network, thereby bringing the greater error to the final result. We believe in the strength of data-driven high-quality priors and strive to offer a reliable and consistent prior, circumventing the restrictions of manual priors.
In this paper, we propose Clearer Night Image Restoration with Vector-Quantized Codebook (VQCNIR) to achieve remarkable and consistent restoration outcomes on real-world and synthetic benchmarks. To ensure the faithful restoration of details and illumination, we propose the incorporation of two essential modules: the Adaptive Illumination Enhancement Module (AIEM) and the Deformable Bi-directional Cross-Attention (DBCA) module. The AIEM leverages the inter-channel correlation of features to dynamically maintain illumination consistency between degraded features and high-quality codebook features. Meanwhile, the DBCA module effectively integrates texture and structural information through bi-directional cross-attention and deformable convolution, resulting in enhanced fine-grained detail and structural fidelity across parallel decoders.
Extensive experiments validate the remarkable benefits of VQCNIR in enhancing image quality under low-light conditions, showcasing its state-of-the-art performance on both synthetic and real-world datasets. The code is available at https://github.com/AlexZou14/VQCNIR.

----

## [875] Enhancing Neural Radiance Fields with Adaptive Multi-Exposure Fusion: A Bilevel Optimization Approach for Novel View Synthesis

**Authors**: *Yang Zou, Xingyuan Li, Zhiying Jiang, Jinyuan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28624](https://doi.org/10.1609/aaai.v38i7.28624)

**Abstract**:

Neural Radiance Fields (NeRF) have made significant strides in the modeling and rendering of 3D scenes. However, due to the complexity of luminance information, existing NeRF methods often struggle to produce satisfactory renderings when dealing with high and low exposure images. To address this issue, we propose an innovative approach capable of effectively modeling and rendering images under multiple exposure conditions. Our method adaptively learns the characteristics of images under different exposure conditions through an unsupervised evaluator-simulator structure for HDR (High Dynamic Range) fusion. This approach enhances NeRF's comprehension and handling of light variations, leading to the generation of images with appropriate brightness. Simultaneously, we present a bilevel optimization method tailored for novel view synthesis, aiming to harmonize the luminance information of input images while preserving their structural and content consistency. This approach facilitates the concurrent optimization of multi-exposure correction and novel view synthesis, in an unsupervised manner. Through comprehensive experiments conducted on the LOM and LOL datasets, our approach surpasses existing methods, markedly enhancing the task of novel view synthesis for multi-exposure environments and attaining state-of-the-art results. The source code can be found at https://github.com/Archer-204/AME-NeRF.

----

## [876] Improved MLP Point Cloud Processing with High-Dimensional Positional Encoding

**Authors**: *Yanmei Zou, Hongshan Yu, Zhengeng Yang, Zechuan Li, Naveed Akhtar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28625](https://doi.org/10.1609/aaai.v38i7.28625)

**Abstract**:

Multi-Layer Perceptron (MLP) models are the bedrock of contemporary point cloud processing. However, their complex network architectures obscure the source of their strength. We first develop an “abstraction and refinement” (ABS-REF) view for the neural modeling of point clouds. This view elucidates that whereas the early models focused on the ABS stage, the more recent techniques devise sophisticated REF stages to attain performance advantage in point cloud processing. We then borrow the concept of “positional encoding” from  transformer literature, and propose a High-dimensional Positional Encoding (HPE) module, which can be  readily deployed to MLP based architectures. We leverage our module to develop a suite of HPENet, which are MLP networks that follow ABS-REF paradigm, albeit with a sophisticated HPE based REF stage. The developed technique is extensively evaluated for 3D object classification, object part segmentation, semantic segmentation and object detection. We establish new state-of-the-art results of 87.6 mAcc on ScanObjectNN for object classification, and 85.5 class mIoU on ShapeNetPart for object part segmentation, and 72.7 and 78.7 mIoU on Area-5 and 6-fold experiments with S3DIS for semantic segmentation. The source code for this work is available at https://github.com/zouyanmei/HPENet.

----

## [877] Sparse3D: Distilling Multiview-Consistent Diffusion for Object Reconstruction from Sparse Views

**Authors**: *Zixin Zou, Weihao Cheng, Yan-Pei Cao, Shi-Sheng Huang, Ying Shan, Song-Hai Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28626](https://doi.org/10.1609/aaai.v38i7.28626)

**Abstract**:

Reconstructing 3D objects from extremely sparse views is a long-standing and challenging problem. While recent techniques employ image diffusion models for generating plausible images at novel viewpoints or for distilling pre-trained diffusion priors into 3D representations using score distillation sampling (SDS), these methods often struggle to simultaneously achieve high-quality, consistent, and detailed results for both novel-view synthesis (NVS) and geometry. In this work, we present Sparse3D, a novel 3D reconstruction method tailored for sparse view inputs. Our approach distills robust priors from a multiview-consistent diffusion model to refine a neural radiance field. Specifically, we employ a controller that harnesses epipolar features from input views, guiding a pre-trained diffusion model, such as Stable Diffusion, to produce novel-view images that maintain 3D consistency with the input. By tapping into 2D priors from powerful image diffusion models, our integrated model consistently delivers high-quality results, even when faced with open-world objects. To address the blurriness introduced by conventional SDS, we introduce the category-score distillation sampling (C-SDS) to enhance detail. We conduct experiments on CO3DV2 which is a multi-view dataset of real-world objects. Both quantitative and qualitative evaluations demonstrate that our approach outperforms previous state-of-the-art works on the metrics regarding NVS and geometry reconstruction.

----

## [878] CEDFlow: Latent Contour Enhancement for Dark Optical Flow Estimation

**Authors**: *Fengyuan Zuo, Zhaolin Xiao, Haiyan Jin, Haonan Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28627](https://doi.org/10.1609/aaai.v38i7.28627)

**Abstract**:

Accurately computing optical flow in low-contrast and noisy dark images is challenging, especially when contour information is degraded or difficult to extract. This paper proposes CEDFlow, a latent space contour enhancement for estimating optical flow in dark environments. By leveraging spatial frequency feature decomposition, CEDFlow effectively encodes local and global motion features. Importantly, we introduce the 2nd-order Gaussian difference operation to select salient contour features in the latent space precisely. It is specifically designed for large-scale contour components essential in dark optical flow estimation. Experimental results on the FCDN and VBOF datasets demonstrate that CEDFlow outperforms state-of-the-art methods in terms of the EPE index and produces more accurate and robust flow estimation. Our code is available at: https://github.com/xautstuzfy.

----

## [879] Parameterization of (Partial) Maximum Satisfiability above Matching in a Variable-Clause Graph

**Authors**: *Vasily Alferov, Ivan Bliznets, Kirill Brilliantov*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28628](https://doi.org/10.1609/aaai.v38i8.28628)

**Abstract**:

In the paper, we study the Maximum Satisfiability and the Partial Maximum Satisfiability problems. Using Gallai–Edmonds decomposition, we significantly improve the upper bound for the Maximum Satisfiability problem parameterized above maximum matching in the variable-clause graph. Our algorithm operates with a runtime of O*(2.83^k'), a substantial improvement compared to the previous approach requiring O*(4^k' ), where k' denotes the relevant parameter. Moreover, this result immediately implies O*(1.14977^m) and O*(1.27895^m) time algorithms for the (n, 3)-MaxSAT and (n, 4)-MaxSAT where m is the overall number of clauses. These upper bounds improve prior-known upper bounds equal to O*(1.1554^m) and O*(1.2872^m). We also adapt the algorithm so that it can handle instances of Partial Maximum Satisfiability without losing performance in some cases. Note that this is somewhat surprising, as the existence of even one hard clause can significantly increase the hardness of a problem.

----

## [880] Approximation Scheme for Weighted Metric Clustering via Sherali-Adams

**Authors**: *Dmitrii Avdiukhin, Vaggos Chatziafratis, Konstantin Makarychev, Grigory Yaroslavtsev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28629](https://doi.org/10.1609/aaai.v38i8.28629)

**Abstract**:

Motivated by applications to classification problems on metric data, we study Weighted Metric Clustering problem: given a metric d over n points and a k x k symmetric matrix A with non-negative entries, the goal is to find a k-partition of these points into clusters C1,...,Ck, while minimizing the sum of A[i,j] * d(u,v) over all pairs of clusters Ci and Cj and all pairs of points u from Ci and v from Cj. Specific choices of A lead to Weighted Metric Clustering capturing well-studied graph partitioning problems in metric spaces, such as Min-Uncut, Min-k-Sum, Min-k-Cut, and more.

Our main result is that Weighted Metric Clustering admits a polynomial-time approximation scheme (PTAS). Our algorithm handles all the above problems using the Sherali-Adams linear programming relaxation. This subsumes several prior works, unifies many of the techniques for various metric clustering objectives, and yields a PTAS for several new problems, including metric clustering on manifolds and a new family of hierarchical clustering objectives. Our experiments on the hierarchical clustering objective show that it better captures the ground-truth structural information compared to the popular Dasgupta's objective.

----

## [881] Neural Time-Reversed Generalized Riccati Equation

**Authors**: *Alessandro Betti, Michele Casoni, Marco Gori, Simone Marullo, Stefano Melacci, Matteo Tiezzi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28630](https://doi.org/10.1609/aaai.v38i8.28630)

**Abstract**:

Optimal control deals with optimization problems in which variables steer a dynamical system, and its outcome contributes to the objective function. Two classical approaches to solving these problems are Dynamic Programming and the Pontryagin Maximum Principle. In both approaches, Hamiltonian equations offer an interpretation of optimality through auxiliary variables known as costates. However, Hamiltonian equations are rarely used due to their reliance on forward-backward algorithms across the entire temporal domain. This paper introduces a novel neural-based approach to optimal control. Neural networks are employed not only for implementing state dynamics but also for estimating costate variables. The parameters of the latter network are determined at each time step using a newly introduced local policy referred to as the time-reversed generalized Riccati equation. This policy is inspired by a result discussed in the Linear Quadratic (LQ) problem, which we conjecture stabilizes state dynamics. We support this conjecture by discussing experimental results from a range of optimal control case studies.

----

## [882] Runtime vs Extracted Proof Size: An Exponential Gap for CDCL on QBFs

**Authors**: *Olaf Beyersdorff, Benjamin Böhm, Meena Mahajan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28631](https://doi.org/10.1609/aaai.v38i8.28631)

**Abstract**:

In both SAT and QBF, proofs can be efficiently extracted from runs of (Q)CDCL solvers. While for CDCL, it is known that the proof size in the underlying proof system propositional resolution matches the CDCL runtime up to a polynomial factor, we show that in QBF there is an exponential gap between QCDCL runtime and the size of the extracted proofs in QBF resolution systems. We demonstrate that this is not just a gap between QCDCL runtime and the size of any QBF resolution proof, but even the extracted proofs are exponentially smaller for some instances. Hence searching for a small proof via QCDCL (even with non-deterministic decision policies) will provably incur an exponential overhead for some instances.

----

## [883] Testing Self-Reducible Samplers

**Authors**: *Rishiraj Bhattacharyya, Sourav Chakraborty, Yash Pote, Uddalok Sarkar, Sayantan Sen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28632](https://doi.org/10.1609/aaai.v38i8.28632)

**Abstract**:

Samplers are the backbone of the implementations of any randomized algorithm. Unfortunately, obtaining an efficient algorithm to test the correctness of samplers is very hard to find. Recently, in a series of works, testers like Barbarik, Teq, Flash for testing of some particular kinds of samplers, like CNF-samplers and Horn-samplers, were obtained. However, their techniques have a significant limitation because one can not expect to use their methods to test for other samplers, such as perfect matching samplers or samplers for sampling linear extensions in posets. 
In this paper, we present a new testing algorithm that works for such samplers and can estimate the distance of a new sampler from a known sampler (say, the uniform sampler).  

Testing the identity of distributions is the heart of testing the correctness of samplers. This paper's main technical contribution is developing a new distance estimation algorithm for distributions over high-dimensional cubes using the recently proposed subcube conditioning sampling model. Given subcube conditioning access to an unknown distribution P, and a known distribution Q defined over an n-dimensional Boolean hypercube, our algorithm CubeProbeEst estimates the variation distance between P and Q within additive error using subcube conditional samples from P.  Following the testing-via-learning paradigm, we also get a tester that distinguishes between the cases when P and Q are close or far in variation distance with high probability using subcube conditional samples.

This estimation algorithm CubeProbeEst in the subcube conditioning sampling model helps us to design the first tester for self-reducible samplers. The correctness of the tester is formally proved. Moreover, we implement CubeProbeEst to test the quality of three samplers for sampling linear extensions in posets.

----

## [884] Using Symmetries to Lift Satisfiability Checking

**Authors**: *Pierre Carbonnelle, Gottfried Schenner, Maurice Bruynooghe, Bart Bogaerts, Marc Denecker*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28633](https://doi.org/10.1609/aaai.v38i8.28633)

**Abstract**:

We analyze how symmetries can be used to compress structures (also known as interpretations) onto a smaller domain without loss of information. This analysis suggests the possibility to solve satisfiability problems in the compressed domain for better performance. Thus, we propose a 2-step novel method: (i) the sentence to be satisfied is automatically translated into an equisatisfiable sentence over a ``lifted'' vocabulary that allows domain compression; (ii) satisfiability of the lifted sentence is checked by growing the (initially unknown) compressed domain until a satisfying structure is found.
The key issue is to ensure that this satisfying structure can always be expanded into an uncompressed structure that satisfies the original sentence to be satisfied.

We present an adequate translation for sentences in typed first-order logic extended with aggregates. Our experimental evaluation shows large speedups for generative configuration problems.  The method also has applications in the verification of software operating on complex data structures.  Our results justify further research in automatic translation of sentences for symmetry reduction.

----

## [885] Robust Beamforming for Downlink Multi-Cell Systems: A Bilevel Optimization Perspective

**Authors**: *Xingdi Chen, Yu Xiong, Kai Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28634](https://doi.org/10.1609/aaai.v38i8.28634)

**Abstract**:

Utilization of inter-base station cooperation for information processing has shown great potential in enhancing the overall quality of communication services (QoS) in wireless communication networks. Nevertheless, such cooperations require the knowledge of channel state information (CSI) at base stations (BSs), which is assumed to be perfectly known. However, CSI errors are inevitable in practice which necessitates beamforming technique that can achieve robust performance in the presence of channel estimation errors. Existing approaches relax the robust beamforming design problems into semidefinite programming (SDP), which can only achieve a solution that is far from being optimal. To this end, this paper views robust beamforming design problems from a bilevel optimization perspective. In particular, we focus on maximizing the worst-case weighted sum-rate (WSR) in the downlink multi-cell multi-user multiple-input single-output (MISO) system considering bounded CSI errors. We first reformulate this problem into a bilevel optimization problem and then develop an efficient algorithm based on the cutting plane method. A distributed optimization algorithm has also been developed to facilitate the parallel processing in practical settings. Numerical results are provided to confirm the effectiveness of the proposed algorithm in terms of performance and complexity, particularly in the presence of CSI uncertainties.

----

## [886] Hardness of Random Reordered Encodings of Parity for Resolution and CDCL

**Authors**: *Leroy Chew, Alexis de Colnet, Friedrich Slivovsky, Stefan Szeider*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28635](https://doi.org/10.1609/aaai.v38i8.28635)

**Abstract**:

Parity reasoning is challenging for Conflict-Driven Clause Learning (CDCL) SAT solvers. This has been observed even for simple formulas encoding two contradictory parity constraints with different variable orders (Chew and Heule 2020). We provide an analytical explanation for their hardness by showing that they require exponential resolution refutations with high probability when the variable order is chosen at random. We obtain this result by proving that these formulas, which are known to be Tseitin formulas, have Tseitin graphs of linear treewidth with high probability. Since such Tseitin formulas require exponential resolution refutations, our result follows. We generalize this argument to a new class of formulas that capture a basic form of parity reasoning involving a sum of two random parity constraints with random orders. Even when the variable order for the sum is chosen favorably, these formulas remain hard for resolution. In contrast, we prove that they have short DRAT refutations. We show experimentally that the running time of CDCL SAT solvers on both classes of formulas grows exponentially with their treewidth.

----

## [887] Percentile Risk-Constrained Budget Pacing for Guaranteed Display Advertising in Online Optimization

**Authors**: *Liang Dai, Kejie Lyu, Chengcheng Zhang, Guangming Zhao, Zhonglin Zu, Liang Wang, Bo Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28636](https://doi.org/10.1609/aaai.v38i8.28636)

**Abstract**:

Guaranteed display (GD) advertising is a critical component of advertising since it provides publishers with stable revenue and enables advertisers to target specific audiences with guaranteed impressions. However, smooth pacing control for online ad delivery presents a challenge due to significant budget disparities, user arrival distribution drift, and dynamic change between supply and demand. This paper presents robust risk-constrained pacing (RCPacing) that utilizes Lagrangian dual multipliers to fine-tune probabilistic throttling through monotonic mapping functions within the percentile space of impression performance distribution. RCPacing combines distribution drift resilience and compatibility with guaranteed allocation mechanism, enabling us to provide near-optimal online services. We also show that RCPacing achieves O(sqrt(T)) dynamic regret where T is the length of the horizon. RCPacing's effectiveness is validated through offline evaluations and online A/B testing conducted on Taobao brand advertising platform.

----

## [888] Unifying Decision and Function Queries in Stochastic Boolean Satisfiability

**Authors**: *Yu-Wei Fan, Jie-Hong R. Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28637](https://doi.org/10.1609/aaai.v38i8.28637)

**Abstract**:

Stochastic Boolean satisfiability (SSAT) is a natural formalism for optimization under uncertainty. Its decision version implicitly imposes a final threshold quantification on an SSAT formula. However, the single threshold quantification restricts the expressive power of SSAT. In this work, we enrich SSAT with an additional threshold quantifier, resulting in a new formalism SSAT(θ). The increased expressiveness allows SSAT(θ), which remains in the PSPACE complexity class, to subsume and encode the languages in the counting hierarchy. An SSAT(θ) solver, ClauSSat(θ), is developed. Experiments show the applicability of the solver in uniquely solving complex SSAT(θ) instances of parameter synthesis and SSAT extension.

----

## [889] Parallel Empirical Evaluations: Resilience despite Concurrency

**Authors**: *Johannes Klaus Fichte, Tobias Geibinger, Markus Hecher, Matthias Schlögel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28638](https://doi.org/10.1609/aaai.v38i8.28638)

**Abstract**:

Computational evaluations are crucial in modern problem-solving when we surpass theoretical algorithms or bounds. These experiments frequently take much work, and the sheer amount of needed resources makes it impossible to execute them on a single personal computer or laptop. Cluster schedulers allow for automatizing these tasks and scale to many computers. But, when we evaluate implementations of combinatorial algorithms, we depend on stable runtime results. Common approaches either limit parallelism or suffer from unstable runtime measurements due to interference among jobs on modern hardware. The former is inefficient and not sustainable. The latter results in unreplicable experiments.
In this work, we address this issue and offer an acceptable balance between efficiency, software, hardware complexity, reliability, and replicability. We investigate effects towards replicability stability and illustrate how to efficiently use widely employed cluster resources for parallel evaluations. Furthermore, we present solutions which mitigate issues that emerge from the concurrent execution of benchmark jobs. Our experimental evaluation shows that – despite parallel execution – our approach reduces the runtime instability on the majority of instances to one second.

----

## [890] Locally Rainbow Paths

**Authors**: *Till Fluschnik, Leon Kellerhals, Malte Renken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28639](https://doi.org/10.1609/aaai.v38i8.28639)

**Abstract**:

We introduce the algorithmic problem of finding a locally rainbow path of length l connecting two distinguished vertices s and t in a vertex-colored directed graph. Herein, a path is locally rainbow if between any two visits of equally colored vertices, the path traverses consecutively at leaset r differently colored vertices. This problem generalizes the well-known problem of finding a rainbow path. It finds natural applications whenever there are different types of resources that must be protected from overuse, such as crop sequence optimization or production process scheduling. We show that the problem is computationally intractable even if r=2 or if one looks for a locally rainbow among the shortest paths. On the positive side, if one looks for a path that takes only a short detour (i.e., it is slightly longer than the shortest path) and if r is small, the problem can be solved efficiently. Indeed, the running time of the respective algorithm is near-optimal unless the ETH fails.

----

## [891] Approximate Integer Solution Counts over Linear Arithmetic Constraints

**Authors**: *Cunjing Ge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28640](https://doi.org/10.1609/aaai.v38i8.28640)

**Abstract**:

Counting integer solutions of linear constraints has found interesting applications in various fields. It is equivalent to the problem of counting lattice points inside a polytope. However, state-of-the-art algorithms for this problem become too slow for even a modest number of variables. In this paper, we propose a new framework to approximate the lattice counts inside a polytope with a new random-walk sampling method. The counts computed by our approach has been proved approximately bounded by a (epsilon, delta)-bound. Experiments on extensive benchmarks show that our algorithm could solve polytopes with dozens of dimensions, which significantly outperforms state-of-the-art counters.

----

## [892] Composing Biases by Using CP to Decompose Minimal Functional Dependencies for Acquiring Complex Formulae

**Authors**: *Ramiz Gindullin, Nicolas Beldiceanu, Jovial Cheukam-Ngouonou, Rémi Douence, Claude-Guy Quimper*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28641](https://doi.org/10.1609/aaai.v38i8.28641)

**Abstract**:

Given a table with a minimal set of input columns that functionally determines an output column, we introduce a method that tries to gradually decompose the corresponding minimal functional dependency (mfd) to acquire a formula expressing the output column in terms of the input columns. A first key element of the method is to create sub-problems that are easier to solve than the original formula acquisition problem, either because it learns formulae with fewer inputs parameters, or as it focuses on formulae of a particular class, such as Boolean formulae; as a result, the acquired formulae can mix different learning biases such as polynomials, conditionals or Boolean expressions. A second key feature of the method is that it can be applied recursively to find formulae that combine polynomial, conditional or Boolean sub-terms in a nested manner. The method was tested on data for eight families of combinatorial objects; new conjectures were found that were previously unattainable. The method often creates conjectures that combine several formulae into one with a limited number of automatically found Boolean terms.

----

## [893] End-to-End Verification for Subgraph Solving

**Authors**: *Stephan Gocht, Ciaran McCreesh, Magnus O. Myreen, Jakob Nordström, Andy Oertel, Yong Kiam Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28642](https://doi.org/10.1609/aaai.v38i8.28642)

**Abstract**:

Modern subgraph-finding algorithm implementations consist of thousands of lines of highly optimized code, and this complexity raises questions about their trustworthiness.  Recently, some state-of-the-art subgraph solvers have been enhanced to output machine-verifiable proofs that their results are correct.  While this significantly improves reliability, it is not a fully satisfactory solution,  since end-users have to trust both the proof checking algorithms and the translation of the high-level graph problem into a low-level 0-1 integer linear program (ILP) used for the proofs.
    
In this work, we present the first formally verified toolchain capable of full end-to-end verification for subgraph solving, which closes both of these trust gaps.  We have built encoder frontends for various graph problems together with a 0-1 ILP (a.k.a. pseudo-Boolean) proof checker, all implemented and formally verified in the CakeML ecosystem.  This toolchain is flexible and extensible, and we use it to build verified proof checkers for both decision and optimization graph problems, namely, subgraph isomorphism, maximum clique, and maximum common (connected) induced subgraph.  Our experimental evaluation shows that end-to-end formal verification is now feasible for a wide range of hard graph problems.

----

## [894] SAT-Based Techniques for Lexicographically Smallest Finite Models

**Authors**: *Mikolás Janota, Choiwah Chow, João Araújo, Michael Codish, Petr Vojtechovský*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28643](https://doi.org/10.1609/aaai.v38i8.28643)

**Abstract**:

This paper proposes SAT-based techniques to calculate a specific normal form of a given finite mathematical structure (model). The normal form is obtained by permuting the domain elements so that the representation of the structure is lexicographically smallest possible. Such a normal form is of interest to mathematicians as it enables easy cataloging of algebraic structures. In particular, two structures are isomorphic precisely when their normal forms are the same. This form is also natural to inspect as mathematicians have been using it routinely for many decades.

We develop a novel approach where a SAT solver is used in a black-box fashion to compute the smallest representative. The approach constructs the representative gradually and searches the space of possible isomorphisms, requiring a small number of variables. However, the approach may lead to a large number of SAT calls and therefore we devise propagation techniques to reduce this number. The paper focuses on finite structures with a single binary operation (encompassing groups, semigroups, etc.). However, the approach is generalizable to arbitrary finite structures. We provide an implementation of the proposed algorithm and evaluate it on a variety of algebraic structures.

----

## [895] Theoretical and Empirical Analysis of Cost-Function Merging for Implicit Hitting Set WCSP Solving

**Authors**: *Javier Larrosa, Conrado Martínez, Emma Rollon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28644](https://doi.org/10.1609/aaai.v38i8.28644)

**Abstract**:

The Implicit Hitting Set (HS) approach has shown very effective for MaxSAT solving. However, only preliminary promising results have been obtained for the very similar Weighted CSP framework. In this paper we contribute towards both a better theoretical understanding of the HS approach and a more effective HS-based solvers for WCSP. First, we bound the minimum number of iterations of HS thanks to what we call distinguished cores. Then, we show a source of inefficiency by
introducing two simple problems where HS is unfeasible. Next, we propose two reformulation methods that merge cost-functions to overcome the problem. We provide a theoretical analysis that quantifies the magnitude of the improvement of each method with respect to the number of iterations of the algorithm. In particular, we show that the reformulations can bring an exponential number of iterations down to a constant number in our working examples. Finally, we complement our theoretical analysis with two sets of experiments. First, we show that our results are aligned with real executions. Second, and most importantly, we conduct experiments on typical benchmark problems and show that cost-function merging may be heuristically applied and it may accelerate HS algorithms by several orders of magnitude. In some cases, it even outperforms state-of-the-art solvers.

----

## [896] Automatic Core-Guided Reformulation via Constraint Explanation and Condition Learning

**Authors**: *Kevin Leo, Graeme Gange, Maria Garcia de la Banda, Mark Wallace*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28645](https://doi.org/10.1609/aaai.v38i8.28645)

**Abstract**:

SAT and propagation solvers often underperform for optimisation models whose objective sums many single-variable terms.
MaxSAT solvers avoid this by detecting and exploiting cores: subsets of these terms that cannot collectively take their lower bounds.
Previous work has shown manual analysis of cores can help define model reformulations likely to speed up solving for many model instances.
This paper presents a method to automate this process.
For each selected core the method identifies the instance constraints that caused it;
infers the model constraints and parameters that explain how these instance constraints were formed;
and learns the conditions that made those model constraint instances generate cores, while others did not.
It then uses this information to reformulate the objective.
The empirical evaluation shows this method can produce useful reformulations.
Importantly, the method can be useful in many other situations that require explaining a set of constraints.

----

## [897] Learning to Pivot as a Smart Expert

**Authors**: *Tianhao Liu, Shanwen Pu, Dongdong Ge, Yinyu Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28646](https://doi.org/10.1609/aaai.v38i8.28646)

**Abstract**:

Linear programming has been practically solved mainly by simplex and interior point methods. Compared with the weakly polynomial complexity obtained by the interior point methods, the existence of strongly polynomial bounds for the length of the pivot path generated by the simplex methods remains a mystery. In this paper, we propose two novel pivot experts that leverage both global and local information of the linear programming instances for the primal simplex method and show their excellent performance numerically. The experts can be regarded as a benchmark to evaluate the performance of classical pivot rules, although they are hard to directly implement. To tackle this challenge, we employ a graph convolutional neural network model, trained via imitation learning, to mimic the behavior of the pivot expert. Our pivot rule, learned empirically, displays a significant advantage over conventional methods in various linear programming problems, as demonstrated through a series of rigorous experiments.

----

## [898] Using Clustering to Strengthen Decision Diagram Bounds for Discrete Optimization

**Authors**: *Mohsen Nafar, Michael Römer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28647](https://doi.org/10.1609/aaai.v38i8.28647)

**Abstract**:

Offering a generic approach to obtaining both upper and lower bounds, decision diagrams (DDs) are becoming an increasingly important tool for solving discrete optimization problems. In particular, they provide a powerful and often complementary alternative to other well-known generic bounding mechanisms such as the LP relaxation. A standard approach to employ DDs for discrete optimization is to formulate the problem as a Dynamic Program and use that formulation to compile a DD top-down in a layer-by-layer fashion. To limit the size of the resulting DD and to obtain bounds, one typically imposes a maximum width for each layer which is then enforced by either merging nodes (resulting in a so-called relaxed DD that provides a dual bound) or by dropping nodes (resulting in a so-called restricted DD that provides a primal bound). The quality of the DD bounds obtained from this top-down compilation process heavily depends on the heuristics used for the selection of the nodes to merge or drop. While it is sometimes possible to engineer problem-specific heuristics for this selection problem, the most generic approach relies on sorting the layer’s nodes based on objective function information. In this paper, we propose a generic and problem-agnostic approach that relies on clustering nodes based on the state information associated with each node. In a set of computational experiments with different knapsack and scheduling problems, we show that our approach generally outperforms the classical generic approach, and often achieves drastically better bounds both with respect to the size of the DD and the time used for compiling the DD.

----

## [899] On Partial Optimal Transport: Revising the Infeasibility of Sinkhorn and Efficient Gradient Methods

**Authors**: *Anh Duc Nguyen, Tuan Dung Nguyen, Quang Minh Nguyen, Hoang H. Nguyen, Lam M. Nguyen, Kim-Chuan Toh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28648](https://doi.org/10.1609/aaai.v38i8.28648)

**Abstract**:

This paper studies the Partial Optimal Transport (POT) problem between two unbalanced measures with at most n supports and its applications in various AI tasks such as color transfer or domain adaptation. There is hence a need for fast approximations of POT with increasingly large problem sizes in arising applications. We first theoretically and experimentally investigate the infeasibility of the state-of-the-art Sinkhorn algorithm for POT, which consequently degrades its qualitative performance in real world applications like point-cloud registration. To this end, we propose a novel rounding algorithm for POT, and then provide a feasible Sinkhorn procedure with a revised computation complexity of O(n^2/epsilon^4). Our rounding algorithm also permits the development of two first-order methods to approximate the POT problem. The first algorithm, Adaptive Primal-Dual Accelerated Gradient Descent (APDAGD), finds an epsilon-approximate solution to the POT problem in O(n^2.5/epsilon). The second method, Dual Extrapolation, achieves the computation complexity of O(n^2/epsilon), thereby being the best in the literature. We further demonstrate the flexibility of POT compared to standard OT as well as the practicality of our algorithms on real applications where two marginal distributions are unbalanced.

----

## [900] An Eager Satisfiability Modulo Theories Solver for Algebraic Datatypes

**Authors**: *Amar Shah, Federico Mora, Sanjit A. Seshia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28649](https://doi.org/10.1609/aaai.v38i8.28649)

**Abstract**:

Algebraic data types (ADTs) are a construct classically found in functional programming languages that capture data structures like enumerated types, lists, and trees. In recent years, interest in ADTs has increased. For example, popular programming languages, like Python, have added support for ADTs. Automated reasoning about ADTs can be done using satisfiability modulo theories (SMT) solving, an extension of the Boolean satisfiability problem with first-order logic and associated background theories. Unfortunately, SMT solvers that support ADTs do not scale as state-of-the-art approaches all use variations of the same lazy approach. In this paper, we present an SMT solver that takes a fundamentally different approach, an eager approach. Specifically, our solver reduces ADT queries to a simpler logical theory, uninterpreted functions (UF), and then uses an existing solver on the reduced query. We prove the soundness and completeness of our approach and demonstrate that it outperforms the state of the art on existing benchmarks, as well as a new, more challenging benchmark set from the planning domain.

----

## [901] An Approximate Skolem Function Counter

**Authors**: *Arijit Shaw, Brendan Juba, Kuldeep S. Meel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28650](https://doi.org/10.1609/aaai.v38i8.28650)

**Abstract**:

One approach to probabilistic inference involves counting the number of models of a given Boolean formula. Here, we are interested in inferences involving higher-order objects, i.e., functions. We study the following task: Given a Boolean specification between a set of inputs and outputs, count the number of functions of inputs such that the specification is met. Such functions are called Skolem functions.

We are motivated by the recent development of scalable approaches to Boolean function synthesis. This stands in relation to our problem analogously to the relationship between Boolean satisfiability and the model counting problem. Yet, counting Skolem functions poses considerable new challenges. From the complexity-theoretic standpoint, counting Skolem functions is not only #P-hard; it is quite unlikely to have an FPRAS (Fully Polynomial Randomized Approximation Scheme) as the problem of synthesizing a Skolem function remains challenging, even given access to an NP oracle.

The primary contribution of this work is the first algorithm, SkolemFC, that computes the number of Skolem functions. SkolemFC relies on technical connections between counting functions and propositional model counting: our algorithm makes a linear number of calls to an approximate model counter and computes an estimate of the number of Skolem functions with theoretical guarantees. Our prototype displays impressive scalability, handling benchmarks comparably to state-of-the-art Skolem function synthesis engines, even though counting all such functions ostensibly poses a greater challenge than synthesizing a single function.

----

## [902] Optimizing ADMM and Over-Relaxed ADMM Parameters for Linear Quadratic Problems

**Authors**: *Jintao Song, Wenqi Lu, Yunwen Lei, Yuchao Tang, Zhenkuan Pan, Jinming Duan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28651](https://doi.org/10.1609/aaai.v38i8.28651)

**Abstract**:

The Alternating Direction Method of Multipliers (ADMM) has gained significant attention across a broad spectrum of machine learning applications. Incorporating the over-relaxation technique shows potential for enhancing the convergence rate of ADMM. However, determining optimal algorithmic parameters, including both the associated penalty and relaxation parameters, often relies on empirical approaches tailored to specific problem domains and contextual scenarios. Incorrect parameter selection can significantly hinder ADMM's convergence rate. To address this challenge, in this paper we first propose a general approach to optimize the value of penalty parameter, followed by a novel closed-form formula to compute the optimal relaxation parameter in the context of linear quadratic problems (LQPs). We then experimentally validate our parameter selection methods through random instantiations and diverse imaging applications, encompassing diffeomorphic image registration, image deblurring, and MRI reconstruction.

----

## [903] Disjoint Partial Enumeration without Blocking Clauses

**Authors**: *Giuseppe Spallitta, Roberto Sebastiani, Armin Biere*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28652](https://doi.org/10.1609/aaai.v38i8.28652)

**Abstract**:

A basic algorithm for enumerating disjoint propositional models (disjoint AllSAT) is based on adding blocking clauses incrementally, ruling out previously found models. On the one hand, blocking clauses have the potential to reduce the number of generated models exponentially, as they can handle partial models. On the other hand, the introduction of a large number of blocking clauses affects memory consumption and drastically slows down unit propagation. 
 We propose a new approach that allows for enumerating disjoint partial models with no need for blocking clauses by integrating: Conflict-Driven Clause-Learning (CDCL), Chronological Backtracking (CB), and methods for shrinking models (Implicant Shrinking). Experiments clearly show the benefits of our novel approach.

----

## [904] SAT-Based Algorithms for Regular Graph Pattern Matching

**Authors**: *Miguel Terra-Neves, José Amaral, Alexandre Lemos, Rui Quintino, Pedro Resende, António Alegria*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28653](https://doi.org/10.1609/aaai.v38i8.28653)

**Abstract**:

Graph matching is a fundamental problem in pattern recognition, with many applications such as software analysis and computational biology. One well-known type of graph matching problem is graph isomorphism, which consists of deciding if two graphs are identical. Despite its usefulness, the properties that one may check using graph isomorphism are rather limited, since it only allows strict equality checks between two graphs. For example, it does not allow one to check complex structural properties such as if the target graph is an arbitrary length sequence followed by an arbitrary size loop.

We propose a generalization of graph isomorphism that allows one to check such properties through a declarative specification. This specification is given in the form of a Regular Graph Pattern (ReGaP), a special type of graph, inspired by regular expressions, that may contain wildcard nodes that represent arbitrary structures such as variable-sized sequences or subgraphs. We propose a SAT-based algorithm for checking if a target graph matches a given ReGaP. We also propose a preprocessing technique for improving the performance of the algorithm and evaluate it through an extensive experimental evaluation on benchmarks from the CodeSearchNet dataset.

----

## [905] CEGAR-Based Approach for Solving Combinatorial Optimization Modulo Quantified Linear Arithmetics Problems

**Authors**: *Kerian Thuillier, Anne Siegel, Loïc Paulevé*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28654](https://doi.org/10.1609/aaai.v38i8.28654)

**Abstract**:

Bioinformatics has always been a prolific domain for generating complex satisfiability and optimization problems. For instance, the synthesis of multi-scale models of biological networks has recently been associated with the resolution of optimization problems mixing Boolean logic and universally quantified linear constraints (OPT+qLP), which can be benchmarked on real-world models. In this paper, we introduce a Counter-Example-Guided Abstraction Refinement (CEGAR) to solve such problems efficiently. Our CEGAR exploits monotone properties inherent to linear optimization in order to generalize counter-examples of Boolean relaxations. We implemented our approach by extending Answer Set Programming (ASP) solver Clingo with a quantified linear constraints propagator. Our prototype enables exploiting independence of sub-formulas to further exploit the generalization of counter-examples. We evaluate the impact of refinement and partitioning on two sets of OPT+qLP problems inspired by system biology. Additionally, we conducted a comparison with the state-of-the-art ASP solver Clingo[lpx] that handles non-quantified linear constraints, showing the advantage of our CEGAR approach for solving large problems.

----

## [906] Learning to Learn in Interactive Constraint Acquisition

**Authors**: *Dimosthenis C. Tsouros, Senne Berden, Tias Guns*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28655](https://doi.org/10.1609/aaai.v38i8.28655)

**Abstract**:

Constraint Programming (CP) has been successfully used to model and solve complex combinatorial problems. However, modeling is often not trivial and requires expertise, which is a bottleneck to wider adoption. In Constraint Acquisition (CA), the goal is to assist the user by automatically learning the model.
In (inter)active CA, this is done by interactively posting queries to the user, e.g. does this partial solution satisfy your (unspecified) constraints or not.
While interactive CA methods learn the constraints, the learning is related to symbolic concept learning, as the goal is to learn an exact representation. 
However, a large number of queries is required to learn the model, which is a major limitation. In this paper, we aim to alleviate this limitation by tightening the connection of CA and Machine Learning (ML), by, for the first time in interactive CA, exploiting statistical ML methods. We propose to use probabilistic classification models to guide interactive CA queries to the most promising parts. We discuss how to train classifiers to predict whether a candidate expression from the bias is a constraint of the problem or not, using both relation-based and scope-based features. We then show how the predictions can be used in all layers of interactive CA: the query generation, the scope finding, and the lowest-level constraint finding. We experimentally evaluate our proposed methods using different classifiers and show that our methods greatly outperform the state of the art, decreasing the number of queries needed to converge by up to 72%.

----

## [907] GSO-Net: Grid Surface Optimization via Learning Geometric Constraints

**Authors**: *Chaoyun Wang, Jingmin Xin, Nanning Zheng, Caigui Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28656](https://doi.org/10.1609/aaai.v38i8.28656)

**Abstract**:

In the context of surface representations, we find a natural structural similarity between grid surface and image data. Motivated by this inspiration, we propose a novel approach: encoding grid surfaces as geometric images and using image processing methods to address surface optimization-related problems. As a result, we have created the first dataset for grid surface optimization and devised a learning-based grid surface optimization network specifically tailored to geometric images, addressing the surface optimization problem through a data-driven learning of geometric constraints paradigm. We conduct extensive experiments on developable surface optimization, surface flattening, and surface denoising tasks using the designed network and datasets. The results demonstrate that our proposed method not only addresses the surface optimization problem better than traditional numerical optimization methods, especially for complex surfaces, but also boosts the optimization speed by multiple orders of magnitude. This pioneering study successfully applies deep learning methods to the field of surface optimization and provides a new solution paradigm for similar tasks, which will provide inspiration and guidance for future developments in the field of discrete surface optimization. The code and dataset are available at https://github.com/chaoyunwang/GSO-Net.

----

## [908] Encoding Constraints as Binary Constraint Networks Satisfying BTP

**Authors**: *Ruiwei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28657](https://doi.org/10.1609/aaai.v38i8.28657)

**Abstract**:

Recently, the Binary Constraint Tree (BCT), a tree structured Binary Constraint Network (BCN), has been shown to be more succinct than various ad-hoc constraints. In this paper, we investigate the modelling power of a well-known tractable hybrid class generalizing BCT, i.e. the class of BCNs satisfying Broken Triangle Property (BTP) called BTP Networks (BTPNs). We show that the consistency checker of BTPN can be computed by polysize monotone circuit, thus, some global constraints cannot be encoded as polysize BTPN, such as the AllDifferent and Linear constraints. Then our study reveals that BTPN is strictly more succinct than the DNNF constraint and all 14 ad-hoc constraints discussed in (Wang and Yap 2023), such as the context-free grammar, BCT and smart table constraints. Furthermore, we also show that BTPN is as powerful as DNNF in terms of computing various operations and queries. In addition, we prove that it is NP-hard to determine the minimum sized BTPN encoding a constraint.

----

## [909] What Are the Rules? Discovering Constraints from Data

**Authors**: *Boris Wiegand, Dietrich Klakow, Jilles Vreeken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28658](https://doi.org/10.1609/aaai.v38i8.28658)

**Abstract**:

Constraint programming and AI planning are powerful tools for solving assignment, optimization, and scheduling problems. They require, however, the rarely available combination of domain knowledge and mathematical modeling expertise. Learning constraints from exemplary solutions can close this gap and alleviate the effort of modeling. Existing approaches either require extensive user interaction, need exemplary invalid solutions that must be generated by experts at great expense, or show high noise-sensitivity. 
We aim to find constraints from potentially noisy solutions, without the need of user interaction. To this end, we formalize the problem in terms of the Minimum Description Length (MDL) principle, by which we select the model with the best lossless compression of the data. Solving the problem involves model counting, which is #P-hard to approximate. We therefore propose the greedy URPILS algorithm to find high-quality constraints in practice. Extensive experiments on constraint programming and AI planning benchmark data show URPILS not only finds more accurate and succinct constraints, but also is more robust to noise, and has lower sample complexity than the state of the art.

----

## [910] SAT-Based Tree Decomposition with Iterative Cascading Policy Selection

**Authors**: *Hai Xia, Stefan Szeider*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28659](https://doi.org/10.1609/aaai.v38i8.28659)

**Abstract**:

Solvers for propositional satisfiability (SAT) effectively tackle hard optimization problems. However, translating to SAT can cause a significant size increase, restricting its use to smaller instances. To mitigate this, frameworks using multiple local SAT calls for gradually improving a heuristic solution have been proposed. The performance of such algorithmic frameworks heavily relies on critical parameters, including the size of selected local instances and the time allocated per SAT call.

This paper examines the automated configuration of the treewidth SAT-based local improvement method (TW-SLIM) framework, which uses multiple SAT calls for computing tree decompositions of small width, a fundamental problem in combinatorial optimization. We explore various TW-SLIM configuration methods, including offline learning and real-time adjustments, significantly outperforming default settings in multi-SAT scenarios with changing problems.

Building upon insights gained from offline training and real-time configurations for TW-SLIM, we propose the iterative cascading policy—a novel hybrid technique that uniquely combines both. The iterative cascading policy employs a pool of 30 configurations obtained through clustering-based offline methods, deploying them in dynamic cascades across multiple rounds. In each round, the 30 configurations are tested according to the cascading ordering, and the best tree decomposition is retained for further improvement, with the option to adjust the following ordering of cascades. This iterative approach significantly enhances the performance of TW-SLIM beyond baseline results, even within varying global timeouts. This highlights the effectiveness of the proposed iterative cascading policy in enhancing the efficiency and efficacy of complex algorithmic frameworks like TW-SLIM.

----

## [911] Engineering an Exact Pseudo-Boolean Model Counter

**Authors**: *Suwei Yang, Kuldeep S. Meel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28660](https://doi.org/10.1609/aaai.v38i8.28660)

**Abstract**:

Model counting, a fundamental task in computer science, involves determining the number of satisfying assignments to a Boolean formula, typically represented in conjunctive normal form (CNF). While model counting for CNF formulas has received extensive attention with a broad range of applications, the study of model counting for Pseudo-Boolean (PB) formulas has been relatively overlooked. Pseudo-Boolean formulas, being more succinct than propositional Boolean formulas, offer greater flexibility in representing real-world problems. Consequently, there is a crucial need to investigate efficient techniques for model counting for PB formulas.

In this work, we propose the first exact Pseudo-Boolean model counter, PBCount , that relies on knowledge compilation approach via algebraic decision diagrams. Our extensive empirical evaluation shows that PBCount  can compute counts for 1513 instances while the current state-of-the-art approach could only handle 1013 instances. Our work opens up several avenues for future work in the context of model counting for PB formulas, such as the development of preprocessing techniques and exploration of approaches other than knowledge compilation.

----

## [912] A Reinforcement-Learning-Based Multiple-Column Selection Strategy for Column Generation

**Authors**: *Haofeng Yuan, Lichang Fang, Shiji Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28661](https://doi.org/10.1609/aaai.v38i8.28661)

**Abstract**:

Column generation (CG) is one of the most successful approaches for solving large-scale linear programming (LP) problems. Given an LP with a prohibitively large number of variables (i.e., columns), the idea of CG is to explicitly consider only a subset of columns and iteratively add potential columns to improve the objective value. While adding the column with the most negative reduced cost can guarantee the convergence of CG, it has been shown that adding multiple columns per iteration rather than a single column can lead to faster convergence. However, it remains a challenge to design a multiple-column selection strategy to select the most promising columns from a large number of candidate columns. In this paper, we propose a novel reinforcement-learning-based (RL) multiple-column selection strategy. To the best of our knowledge, it is the first RL-based multiple-column selection strategy for CG. The effectiveness of our approach is evaluated on two sets of problems: the cutting stock problem and the graph coloring problem. Compared to several widely used single-column and multiple-column selection strategies, our RL-based multiple-column selection strategy leads to faster convergence and achieves remarkable reductions in the number of CG iterations and runtime.

----

## [913] Large-Scale Non-convex Stochastic Constrained Distributionally Robust Optimization

**Authors**: *Qi Zhang, Yi Zhou, Ashley Prater-Bennette, Lixin Shen, Shaofeng Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28662](https://doi.org/10.1609/aaai.v38i8.28662)

**Abstract**:

Distributionally robust optimization (DRO) is a powerful framework for training robust models against data distribution shifts. This paper focuses on constrained DRO, which has an explicit characterization of the robustness level.  Existing studies on constrained DRO mostly focus on convex loss function, and exclude the practical and challenging case with non-convex loss function, e.g., neural network. This paper develops a  stochastic algorithm and its performance analysis for non-convex constrained DRO. The computational complexity of our stochastic algorithm at each iteration is independent of the overall dataset size, and thus is suitable for large-scale applications. We focus on the general Cressie-Read family divergence defined uncertainty set which includes chi^2-divergences as a special case. We prove that our algorithm finds an epsilon-stationary point with an improved computational complexity than existing methods. Our method also applies to the smoothed conditional value at risk (CVaR) DRO.

----

## [914] Multimodal Graph Neural Architecture Search under Distribution Shifts

**Authors**: *Jie Cai, Xin Wang, Haoyang Li, Ziwei Zhang, Wenwu Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28663](https://doi.org/10.1609/aaai.v38i8.28663)

**Abstract**:

Multimodal graph neural architecture search (MGNAS) has shown great success for automatically designing the optimal multimodal graph neural network (MGNN) architecture by leveraging multimodal representation, crossmodal information and graph structure in one unified framework. However, existing MGNAS fails to handle distribution shifts that naturally exist in multimodal graph data, since the searched architectures inevitably capture spurious statistical correlations under distribution shifts. To solve this problem, we propose a novel Out-of-distribution Generalized Multimodal Graph Neural Architecture Search (OMG-NAS) method which optimizes the MGNN architecture with respect to its performance on decorrelated OOD data. Specifically, we propose a multimodal graph representation decorrelation strategy, which encourages the searched MGNN model to output representations that eliminate spurious correlations through iteratively optimizing the feature weights and controller. In addition, we propose a global sample weight estimator that facilitates the sharing of optimal sample weights learned from existing architectures. This design promotes the effective estimation of the sample weights for candidate MGNN architectures to generate decorrelated multimodal graph representations, concentrating more on the truly predictive relations between invariant features and ground-truth labels. Extensive experiments on real-world multimodal graph datasets demonstrate the superiority of our proposed method over SOTA baselines.

----

## [915] Make Lossy Compression Meaningful for Low-Light Images

**Authors**: *Shilv Cai, Liqun Chen, Sheng Zhong, Luxin Yan, Jiahuan Zhou, Xu Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28664](https://doi.org/10.1609/aaai.v38i8.28664)

**Abstract**:

Low-light images frequently occur due to unavoidable environmental influences or technical limitations, such as insufficient lighting or limited exposure time. To achieve better visibility for visual perception, low-light image enhancement is usually adopted. Besides, lossy image compression is vital for meeting the requirements of storage and transmission in computer vision applications. To touch the above two practical demands, current solutions can be categorized into two sequential manners: ``Compress before Enhance (CbE)'' or ``Enhance before Compress (EbC)''. However, both of them are not suitable since: (1) Error accumulation in the individual models plagues sequential solutions. Especially, once low-light images are compressed by existing general lossy image compression approaches, useful information (e.g., texture details) would be lost resulting in a dramatic performance decrease in low-light image enhancement. (2) Due to the intermediate process, the sequential solution introduces an additional burden resulting in low efficiency. We propose a novel joint solution to simultaneously achieve a high compression rate and good enhancement performance for low-light images with much lower computational cost and fewer model parameters. We design an end-to-end trainable architecture, which includes the main enhancement branch and the signal-to-noise ratio (SNR) aware branch. Experimental results show that our proposed joint solution achieves a significant improvement over different combinations of existing state-of-the-art sequential ``Compress before Enhance'' or ``Enhance before Compress'' solutions for low-light images, which would make lossy low-light image compression more meaningful. The project is publicly available at: https://github.com/CaiShilv/Joint-IC-LL.

----

## [916] RR-PU: A Synergistic Two-Stage Positive and Unlabeled Learning Framework for Robust Tax Evasion Detection

**Authors**: *Shuzhi Cao, Jianfei Ruan, Bo Dong, Bin Shi, Qinghua Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28665](https://doi.org/10.1609/aaai.v38i8.28665)

**Abstract**:

Tax evasion, an unlawful practice in which taxpayers deliberately conceal information to avoid paying tax liabilities, poses significant challenges for tax authorities. Effective tax evasion detection is critical for assisting tax authorities in mitigating tax revenue loss. Recently, machine-learning-based methods, particularly those employing positive and unlabeled (PU) learning, have been adopted for tax evasion detection, achieving notable success. However, these methods exhibit two major practical limitations. First, their success heavily relies on the strong assumption that the label frequency (the fraction of identified taxpayers among tax evaders) is known in advance. Second, although some methods attempt to estimate label frequency using approaches like Mixture Proportion Estimation (MPE) without making any assumptions, they subsequently construct a classifier based on the error-prone label frequency obtained from the previous estimation. This two-stage approach may not be optimal, as it neglects error accumulation in classifier training resulting from the estimation bias in the first stage. To address these limitations, we propose a novel PU learning-based tax evasion detection framework called RR-PU, which can revise the bias in a two-stage synergistic manner. Specifically, RR-PU refines the label frequency initialization by leveraging a regrouping technique to fortify the MPE perspective. Subsequently, we integrate a trainable slack variable to fine-tune the initial label frequency, concurrently optimizing this variable and the classifier to eliminate latent bias in the initial stage. Experimental results on three real-world tax datasets demonstrate that RR-PU outperforms state-of-the-art methods in tax evasion detection tasks.

----

## [917] Hierarchical and Incremental Structural Entropy Minimization for Unsupervised Social Event Detection

**Authors**: *Yuwei Cao, Hao Peng, Zhengtao Yu, Philip S. Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28666](https://doi.org/10.1609/aaai.v38i8.28666)

**Abstract**:

As a trending approach for social event detection, graph neural network (GNN)-based methods enable a fusion of natural language semantics and the complex social network structural information, thus showing SOTA performance. However, GNN-based methods can miss useful message correlations. Moreover, they require manual labeling for training and predetermining the number of events for prediction. In this work, we address social event detection via graph structural entropy (SE) minimization. While keeping the merits of the GNN-based methods, the proposed framework, HISEvent, constructs more informative message graphs, is unsupervised, and does not require the number of events given a priori. Specifically, we incrementally explore the graph neighborhoods using 1-dimensional (1D) SE minimization to supplement the existing message graph with edges between semantically related messages. We then detect events from the message graph by hierarchically minimizing 2-dimensional (2D) SE. Our proposed 1D and 2D SE minimization algorithms are customized for social event detection and effectively tackle the efficiency problem of the existing SE minimization algorithms. Extensive experiments show that HISEvent consistently outperforms GNN-based methods and achieves the new SOTA for social event detection under both closed- and open-set settings while being efficient and robust.

----

## [918] Distributional Off-Policy Evaluation for Slate Recommendations

**Authors**: *Shreyas Chaudhari, David Arbour, Georgios Theocharous, Nikos Vlassis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28667](https://doi.org/10.1609/aaai.v38i8.28667)

**Abstract**:

Recommendation strategies are typically evaluated by using previously logged data, employing off-policy evaluation methods to estimate their expected performance. However, for strategies that present users with slates of multiple items, the resulting combinatorial action space renders many of these methods impractical. Prior work has developed estimators that leverage the structure in slates to estimate the expected off-policy performance, but the estimation of the entire performance distribution remains elusive. Estimating the complete distribution allows for a more comprehensive evaluation of recommendation strategies, particularly along the axes of risk and fairness that employ metrics computable from the distribution. In this paper, we propose an estimator for the complete off-policy performance distribution for slates and establish conditions under which the estimator is unbiased and consistent. This builds upon prior work on off-policy evaluation for slates and off-policy distribution estimation in reinforcement learning. We validate the efficacy of our method empirically on synthetic data as well as on a slate recommendation simulator constructed from real-world data (MovieLens-20M). Our results show a significant reduction in estimation variance and improved sample efficiency over prior work across a range of slate structures.

----

## [919] Uncertainty-Aware Yield Prediction with Multimodal Molecular Features

**Authors**: *Jiayuan Chen, Kehan Guo, Zhen Liu, Olexandr Isayev, Xiangliang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28668](https://doi.org/10.1609/aaai.v38i8.28668)

**Abstract**:

Predicting chemical reaction yields is pivotal for efficient chemical synthesis, an area that focuses on the creation of novel compounds for diverse uses. 
Yield prediction demands accurate representations of reactions for forecasting practical transformation rates. Yet, the uncertainty issues broadcasting in real-world situations prohibit current models to excel in this task owing to the high sensitivity of yield activities and the uncertainty in yield measurements.  Existing models often utilize single-modal feature representations, such as molecular fingerprints, SMILES sequences, or molecular graphs, which is not sufficient to capture the complex interactions and dynamic behavior of molecules in reactions. In this paper, we present an advanced Uncertainty-Aware Multimodal model (UAM) to tackle these challenges. Our approach seamlessly integrates data sources from multiple modalities by encompassing sequence representations, molecular graphs, and expert-defined chemical reaction features for a comprehensive representation of reactions. Additionally, we address both the model and data-based uncertainty, refining the model's predictive capability. Extensive experiments on three datasets, including two high throughput experiment (HTE) datasets and one chemist-constructed Amide coupling reaction dataset, demonstrate that UAM outperforms the state-of-the-art methods. The code and used datasets are available at https://github.com/jychen229/Multimodal-reaction-yield-prediction.

----

## [920] Sparse Enhanced Network: An Adversarial Generation Method for Robust Augmentation in Sequential Recommendation

**Authors**: *Junyang Chen, Guoxuan Zou, Pan Zhou, Yirui Wu, Zhenghan Chen, Houcheng Su, Huan Wang, Zhiguo Gong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28669](https://doi.org/10.1609/aaai.v38i8.28669)

**Abstract**:

Sequential Recommendation plays a significant role in daily recommendation systems, such as e-commerce platforms like Amazon and Taobao. However, even with the advent of large models, these platforms often face sparse issues in the historical browsing records of individual users due to new users joining or the introduction of new products. As a result, existing sequence recommendation algorithms may not perform well. To address this, sequence-based data augmentation methods have garnered attention.

Existing sequence enhancement methods typically rely on augmenting existing data, employing techniques like cropping, masking prediction, random reordering, and random replacement of the original sequence. While these methods have shown improvements, they often overlook the exploration of the deep embedding space of the sequence. To tackle these challenges, we propose a Sparse Enhanced Network (SparseEnNet), which is a robust adversarial generation method. SparseEnNet aims to fully explore the hidden space in sequence recommendation, generating more robust enhanced items. Additionally, we adopt an adversarial generation method, allowing the model to differentiate between data augmentation categories and achieve better prediction performance for the next item in the sequence. Experiments have demonstrated that our method achieves a remarkable 4-14% improvement over existing methods when evaluated on the real-world datasets. (https://github.com/junyachen/SparseEnNet)

----

## [921] Signed Graph Neural Ordinary Differential Equation for Modeling Continuous-Time Dynamics

**Authors**: *Lanlan Chen, Kai Wu, Jian Lou, Jing Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28670](https://doi.org/10.1609/aaai.v38i8.28670)

**Abstract**:

Modeling continuous-time dynamics constitutes a foundational challenge, and uncovering inter-component correlations within complex systems holds promise for enhancing the efficacy of dynamic modeling. The prevailing approach of integrating graph neural networks with ordinary differential equations has demonstrated promising performance. However, they disregard the crucial signed information potential on graphs, impeding their capacity to accurately capture real-world phenomena and leading to subpar outcomes. In response, we introduce a novel approach: a signed graph neural ordinary differential equation, adeptly addressing the limitations of miscapturing signed information. Our proposed solution boasts both flexibility and efficiency. To substantiate its effectiveness, we seamlessly integrate our devised strategies into three preeminent graph-based dynamic modeling frameworks: graph neural ordinary differential equations, graph neural controlled differential equations, and graph recurrent neural networks. Rigorous assessments encompass three intricate dynamic scenarios from physics and biology, as well as scrutiny across four authentic real-world traffic datasets. Remarkably outperforming the trio of baselines, empirical results underscore the substantial performance enhancements facilitated by our proposed approach. Our code can be found at https://github.com/beautyonce/SGODE.

----

## [922] Deep Structural Knowledge Exploitation and Synergy for Estimating Node Importance Value on Heterogeneous Information Networks

**Authors**: *Yankai Chen, Yixiang Fang, Qiongyan Wang, Xin Cao, Irwin King*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28671](https://doi.org/10.1609/aaai.v38i8.28671)

**Abstract**:

The classic problem of node importance estimation has been conventionally studied with homogeneous network topology analysis. To deal with practical network heterogeneity, a few recent methods employ graph neural models to automatically learn diverse sources of information. However, the major concern revolves around that their fully adaptive learning process may lead to insufficient information exploration, thereby formulating the problem as the isolated node value prediction with underperformance and less interpretability. In this work, we propose a novel learning framework namely SKES. Different from previous automatic learning designs, SKES exploits heterogeneous structural knowledge to enrich the informativeness of node representations. Then based on a sufficiently uninformative reference, SKES estimates the importance value for any input node, by quantifying its informativeness disparity against the reference. This establishes an interpretable node importance computation paradigm. Furthermore, SKES dives deep into the understanding that "nodes with similar characteristics are prone to have similar importance values" whilst guaranteeing that such informativeness disparity between any different nodes is orderly reflected by the embedding distance of their associated latent features. Extensive experiments on three widely-evaluated benchmarks demonstrate the performance superiority of SKES over several recent competing methods.

----

## [923] KGTS: Contrastive Trajectory Similarity Learning over Prompt Knowledge Graph Embedding

**Authors**: *Zhen Chen, Dalin Zhang, Shanshan Feng, Kaixuan Chen, Lisi Chen, Peng Han, Shuo Shang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28672](https://doi.org/10.1609/aaai.v38i8.28672)

**Abstract**:

Trajectory similarity computation serves as a fundamental functionality of various spatial information applications. Although existing deep learning similarity computation methods offer better efficiency and accuracy than non-learning solutions, they are still immature in trajectory embedding and suffer from poor generality and heavy preprocessing for training. Targeting these limitations, we propose a novel framework named KGTS based on knowledge graph grid embedding, prompt trajectory embedding, and unsupervised contrastive learning for improved trajectory similarity computation. Specifically, we first embed map grids with a GRot embedding method to vigorously grasp the neighbouring relations of grids. Then, a prompt trajectory embedding network incorporates the resulting grid embedding and extracts trajectory structure and point order information. It is trained by unsupervised contrastive learning, which not only alleviates the heavy preprocessing burden but also provides exceptional generality with creatively designed strategies for positive sample generation. The prompt trajectory embedding adopts a customized prompt paradigm to mitigate the gap between the grid embedding and the trajectory embedding. Extensive experiments on two real-world trajectory datasets demonstrate the superior performance of KGTS over state-of-the-art methods.

----

## [924] Learning to Reweight for Generalizable Graph Neural Network

**Authors**: *Zhengyu Chen, Teng Xiao, Kun Kuang, Zheqi Lv, Min Zhang, Jinluan Yang, Chengqiang Lu, Hongxia Yang, Fei Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28673](https://doi.org/10.1609/aaai.v38i8.28673)

**Abstract**:

Graph Neural Networks (GNNs) show promising results for graph tasks. However, existing GNNs' generalization ability will degrade when there exist distribution shifts between testing and training graph data.
The fundamental reason for the severe degeneration is that most GNNs are designed based on the I.I.D hypothesis. In such a setting, GNNs tend to exploit subtle statistical correlations existing in the training set for predictions, even though it is a spurious correlation.
In this paper, we study the problem of the generalization ability of GNNs on Out-Of-Distribution (OOD) settings.
To solve this problem, we propose the Learning to Reweight for Generalizable Graph Neural Network (L2R-GNN) to enhance the generalization ability for achieving satisfactory performance on unseen testing graphs that have different distributions with training graphs.
We propose a novel nonlinear graph decorrelation method, which can substantially improve the out-of-distribution generalization ability and compares favorably to previous methods in restraining the over-reduced sample size.
The variables of graph representation are clustered based on the stability of their correlations, and graph decorrelation method learns weights to remove correlations between the variables of different clusters rather than any two variables.
Besides, we introduce an effective stochastic algorithm based on bi-level optimization for the L2R-GNN framework, which enables simultaneously learning the optimal weights and GNN parameters, and avoids the over-fitting issue.
Experiments show that L2R-GNN greatly outperforms baselines on various graph prediction benchmarks under distribution shifts.

----

## [925] Effective Comparative Prototype Hashing for Unsupervised Domain Adaptation

**Authors**: *Hui Cui, Lihai Zhao, Fengling Li, Lei Zhu, Xiaohui Han, Jingjing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28674](https://doi.org/10.1609/aaai.v38i8.28674)

**Abstract**:

Unsupervised domain adaptive hashing is a highly promising research direction within the field of retrieval. It aims to transfer valuable insights from the source domain to the target domain while maintaining high storage and retrieval efficiency. Despite its potential, this field remains relatively unexplored. Previous methods usually lead to unsatisfactory retrieval performance, as they frequently directly apply slightly modified domain adaptation algorithms to hash learning framework, or pursue domain alignment within the Hamming space characterized by limited semantic information. In this paper, we propose a simple yet effective approach named Comparative Prototype Hashing (CPH) for unsupervised domain adaptive image retrieval. We establish a domain-shared unit hypersphere space through prototype contrastive learning and then obtain the Hamming hypersphere space via mapping from the shared hypersphere. This strategy achieves a cohesive synergy between learning uniformly distributed and category conflict-averse feature representations, eliminating domain discrepancies, and facilitating hash code learning. Moreover, by leveraging dual-domain information to supervise the entire hashing model training process, we can generate hash codes that retain inter-sample similarity relationships within both domains. Experimental results validate that our CPH significantly outperforms the state-of-the-art counterparts across multiple cross-domain and single-domain retrieval tasks. Notably, on Office-Home and Office-31 datasets, CPH achieves an average performance improvement of 19.29% and 13.85% on cross-domain retrieval tasks compared to the second-best results, respectively. The source codes of our method are available at: https://github.com/christinecui/CPH.

----

## [926] Modeling Knowledge Graphs with Composite Reasoning

**Authors**: *Wanyun Cui, Linqiu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28675](https://doi.org/10.1609/aaai.v38i8.28675)

**Abstract**:

The ability to combine multiple pieces of existing knowledge to infer new knowledge is both crucial and challenging. In this paper, we explore how facts of various entities are combined in the context of knowledge graph completion (KGC). We use composite reasoning to unify the views from different KGC models, including translational models, tensor factorization (TF)-based models, instance-based learning models, and KGC regularizers.

Moreover, our comprehensive examination of composite reasoning revealed an unexpected phenomenon: certain TF-based models learn embeddings with erroneous composite reasoning, which ultimately violates their fundamental collaborative filtering assumption and reduces their effects. This motivates us to reduce their composition error. Empirical evaluations demonstrate that mitigating the composition risk not only enhances the performance of TF-based models across all tested settings, but also surpass or is competitive with the state-of-the-art performance on two out of four benchmarks.

----

## [927] Discovering Sequential Patterns with Predictable Inter-event Delays

**Authors**: *Joscha Cüppers, Paul Krieger, Jilles Vreeken*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28676](https://doi.org/10.1609/aaai.v38i8.28676)

**Abstract**:

Summarizing sequential data with serial episodes allows non-trivial insight into the data generating process. Existing methods penalize gaps in pattern occurrences equally, regardless of where in the pattern these occur. This results in a strong bias against patterns with long inter-event delays, and in addition that regularity in terms of delays is not rewarded or discovered---even though both aspects provide key insight.
In this paper we tackle both these problems by explicitly modeling inter-event delay distributions. That is, we are not only interested in discovering the patterns, but also in describing how many times steps typically occur between their individual events. We formalize the problem in terms of the Minimum Description Length principle, by which we say the best set of patterns is the one that compresses the data best. The resulting optimization problem does not lend itself to exact optimization, and hence we propose Hopper to heuristically mine high quality patterns. Extensive experiments show that Hopper efficiently recovers the ground truth, discovers meaningful patterns from real-world data, and outperforms existing methods in discovering long-delay patterns.

----

## [928] Unveiling Implicit Deceptive Patterns in Multi-Modal Fake News via Neuro-Symbolic Reasoning

**Authors**: *Yiqi Dong, Dongxiao He, Xiaobao Wang, Youzhu Jin, Meng Ge, Carl Yang, Di Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28677](https://doi.org/10.1609/aaai.v38i8.28677)

**Abstract**:

In the current Internet landscape, the rampant spread of fake news, particularly in the form of multi-modal content, poses a great social threat. While automatic multi-modal fake news detection methods have shown promising results, the lack of explainability remains a significant challenge. Existing approaches provide superficial explainability by displaying learned important components or views from well-trained networks, but they often fail to uncover the implicit deceptive patterns that reveal how fake news is fabricated.  To address this limitation, we begin by predefining three typical deceptive patterns, namely image manipulation, cross-modal inconsistency, and image repurposing, which shed light on the mechanisms underlying fake news fabrication. Then, we propose a novel Neuro-Symbolic Latent Model called NSLM, that not only derives accurate judgments on the veracity of news but also uncovers the implicit deceptive patterns as explanations. Specifically, the existence of each deceptive pattern is expressed as a two-valued learnable latent variable, which is acquired through amortized variational inference and weak supervision based on symbolic logic rules.  Additionally, we devise pseudo-siamese networks to capture distinct deceptive patterns effectively. Experimental results on two real-world datasets demonstrate that our NSLM achieves the best performance in fake news detection while providing insightful explanations of deceptive patterns.

----

## [929] Enhancing Job Recommendation through LLM-Based Generative Adversarial Networks

**Authors**: *Yingpeng Du, Di Luo, Rui Yan, Xiaopei Wang, Hongzhi Liu, Hengshu Zhu, Yang Song, Jie Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28678](https://doi.org/10.1609/aaai.v38i8.28678)

**Abstract**:

Recommending suitable jobs to users is a critical task in online recruitment platforms. While existing job recommendation methods encounter challenges such as the low quality of users' resumes, which hampers their accuracy and practical effectiveness.With the rapid development of large language models (LLMs), utilizing the rich external knowledge encapsulated within them, as well as their powerful reasoning capabilities, is a promising way to complete users' resumes for more accurate recommendations. However, directly leveraging LLMs to enhance recommendation results is not a one-size-fits-all solution, as LLMs may suffer from fabricated generation and few-shot problems, which degrade the quality of resume completion.

In this paper, we propose a novel LLM-based approach for job recommendation. To alleviate the limitation of fabricated generation for LLMs, we extract accurate and valuable information beyond users' self-description, which helps the LLMs better profile users for resume completion. Specifically,  we not only extract users' explicit properties (e.g., skills, interests) from their self-description but also infer users' implicit characteristics from their behaviors for more accurate and meaningful resume completion. Nevertheless, some users still suffer from few-shot problems, which arise due to scarce interaction records, leading to limited guidance for high-quality resume generation. To address this issue, we propose aligning unpaired low-quality with high-quality generated resumes by Generative Adversarial Networks (GANs), which can refine the resume representations for better recommendation results. Extensive experiments on three large real-world recruitment datasets demonstrate the effectiveness of our proposed method.

----

## [930] Structural Entropy Based Graph Structure Learning for Node Classification

**Authors**: *Liang Duan, Xiang Chen, Wenjie Liu, Daliang Liu, Kun Yue, Angsheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28679](https://doi.org/10.1609/aaai.v38i8.28679)

**Abstract**:

As one of the most common tasks in graph data analysis, node classification is frequently solved by using graph structure learning (GSL) techniques to optimize graph structures and learn suitable graph neural networks. Most of the existing GSL methods focus on fusing different structural features (basic views) extracted from the graph, but very little graph semantics, like hierarchical communities, has been incorporated. Thus, they might be insufficient when dealing with the graphs containing noises from real-world complex systems. To address this issue, we propose a novel and effective GSL framework for node classification based on the structural information theory. Specifically, we first prove that an encoding tree with the minimal structural entropy could contain sufficient information for node classification and eliminate redundant noise via the graph's hierarchical abstraction. Then, we provide an efficient algorithm for constructing the encoding tree to enhance the basic views. Combining the community influence deduced from the encoding tree and the prediction confidence of each view, we further fuse the enhanced views to generate the optimal structure. Finally, we conduct extensive experiments on a variety of datasets. The results demonstrate that our method outperforms the state-of-the-art competitors on effectiveness and robustness.

----

## [931] Progressive Distillation Based on Masked Generation Feature Method for Knowledge Graph Completion

**Authors**: *Cunhang Fan, Yujie Chen, Jun Xue, Yonghui Kong, Jianhua Tao, Zhao Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28680](https://doi.org/10.1609/aaai.v38i8.28680)

**Abstract**:

In recent years, knowledge graph completion (KGC) models based on pre-trained language model (PLM) have shown promising results. However, the large number of parameters and high computational cost of PLM models pose challenges for their application in downstream tasks. This paper proposes a progressive distillation method based on masked generation features for KGC task, aiming to significantly reduce the complexity of pre-trained models. Specifically, we perform pre-distillation on PLM to obtain high-quality teacher models, and compress the PLM network to obtain multi-grade student models. However, traditional feature distillation suffers from the limitation of having a single representation of information in teacher models. To solve this problem, we propose masked generation of teacher-student features, which contain richer representation information. Furthermore, there is a significant gap in representation ability between teacher and student. Therefore, we design a progressive distillation method to distill student models at each grade level, enabling efficient knowledge transfer from teachers to students. The experimental results demonstrate that the model in the pre-distillation stage surpasses the existing state-of-the-art methods. Furthermore, in the progressive distillation stage, the model significantly reduces the model parameters while maintaining a certain level of performance. Specifically, the model parameters of the lower-grade student model are reduced by 56.7\% compared to the baseline.

----

## [932] StockMixer: A Simple Yet Strong MLP-Based Architecture for Stock Price Forecasting

**Authors**: *Jinyong Fan, Yanyan Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28681](https://doi.org/10.1609/aaai.v38i8.28681)

**Abstract**:

Stock price forecasting is a fundamental yet challenging task in quantitative investment. Various researchers have developed a combination of neural network models (e.g., RNNs, GNNs, Transformers) for capturing complex indicator, temporal and stock correlations of the stock data.While complex architectures are highly expressive, they are often difficult to optimize and the performances are often compromised by the limited stock data. In this paper, we propose a simple MLP-based architecture named StockMixer which is easy to optimize and enjoys strong predictive performance. StockMixer performs indicator mixing, followed by time mixing, and finally stock mixing. Unlike the standard MLP-based mixing, we devise the time mixing to exchange multi-scale time patch information and realize the stock mixing by exploiting stock-to-market and market-to-stock influences explicitly. Extensive experiments on real stock benchmarks demonstrate our proposed StockMixer outperforms various state-of-the-art forecasting methods with a notable margin while reducing memory usage and runtime cost.Code is available at https://github.com/SJTU-Quant/StockMixer.

----

## [933] Dense Projection for Anomaly Detection

**Authors**: *Dazhi Fu, Zhao Zhang, Jicong Fan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28682](https://doi.org/10.1609/aaai.v38i8.28682)

**Abstract**:

This work presents a novel method called dense projection for unsupervised anomaly detection (DPAD). The main idea is maximizing the local density of (normal) training data and then determining whether a test data is anomalous or not by evaluating its density. Specifically, DPAD uses a deep neural network to learn locally dense representations of normal data. Since density estimation is computationally expensive, we minimize the local distances of the representations in an iteratively reweighting manner, where the weights are updated adaptively and the parameters are regularized to avoid model collapse (all representations collapse to a single point). Compared with many state-of-the-art methods of anomaly detection, our DPAD does not rely on any assumption about the distribution or spatial structure of the normal data and representations. Moreover, we provide theoretical guarantees for the effectiveness of DPAD. The experiments show that our method DPAD is effective not only in traditional one-class classification problems but also in scenarios with complex normal data composed of multiple classes.

----

## [934] Knowledge-Enhanced Historical Document Segmentation and Recognition

**Authors**: *En-Hao Gao, Yu-Xuan Huang, Wen-Chao Hu, Xin-Hao Zhu, Wang-Zhou Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28683](https://doi.org/10.1609/aaai.v38i8.28683)

**Abstract**:

Optical Character Recognition (OCR) of historical document images remains a challenging task because of the distorted input images, extensive number of uncommon characters, and the scarcity of labeled data, which impedes modern deep learning-based OCR techniques from achieving good recognition accuracy. Meanwhile, there exists a substantial amount of expert knowledge that can be utilized in this task. However, such knowledge is usually complicated and could only be accurately expressed with formal languages such as first-order logic (FOL), which is difficult to be directly integrated into deep learning models. This paper proposes KESAR, a novel Knowledge-Enhanced Document Segmentation And Recognition method for historical document images based on the Abductive Learning (ABL) framework. The segmentation and recognition models are enhanced by incorporating background knowledge for character extraction and prediction, followed by an efficient joint optimization of both models. We validate the effectiveness of KESAR on historical document datasets. The experimental results demonstrate that our method can simultaneously utilize knowledge-driven reasoning and data-driven learning, which outperforms the current state-of-the-art methods.

----

## [935] Zero-1-to-3: Domain-Level Zero-Shot Cognitive Diagnosis via One Batch of Early-Bird Students towards Three Diagnostic Objectives

**Authors**: *Weibo Gao, Qi Liu, Hao Wang, Linan Yue, Haoyang Bi, Yin Gu, Fangzhou Yao, Zheng Zhang, Xin Li, Yuanjing He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28684](https://doi.org/10.1609/aaai.v38i8.28684)

**Abstract**:

Cognitive diagnosis seeks to estimate the cognitive states of students by exploring their logged practice quiz data. It plays a pivotal role in personalized learning guidance within intelligent education systems. In this paper, we focus on an important, practical, yet often underexplored task: domain-level zero-shot cognitive diagnosis (DZCD), which arises due to the absence of student practice logs in newly launched domains. Recent cross-domain diagnostic models have been demonstrated to be a promising strategy for DZCD. These methods primarily focus on how to transfer student states across domains. However, they might inadvertently incorporate non-transferable information into student representations, thereby limiting the efficacy of knowledge transfer. To tackle this, we propose Zero-1-to-3, a domain-level zero-shot cognitive diagnosis framework via one batch of early-bird students towards three diagnostic objectives. Our approach initiates with pre-training a diagnosis model with dual regularizers, which decouples student states into domain-shared and domain-specific parts. The shared cognitive signals can be transferred to the target domain, enriching the cognitive priors for the new domain, which ensures the cognitive state propagation objective. Subsequently, we devise a strategy to generate simulated practice logs for cold-start students through analyzing the behavioral patterns from early-bird students, fulfilling the domain-adaption goal. Consequently, we refine the cognitive states of cold-start students as diagnostic outcomes via virtual data, aligning with the diagnosis-oriented goal. Finally, extensive experiments on six real-world datasets highlight the efficacy of our model for DZCD and its practical application in question recommendation. The code is publicly available at https://github.com/bigdata-ustc/Zero-1-to-3.

----

## [936] Your Career Path Matters in Person-Job Fit

**Authors**: *Zhuocheng Gong, Yang Song, Tao Zhang, Ji-Rong Wen, Dongyan Zhao, Rui Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28685](https://doi.org/10.1609/aaai.v38i8.28685)

**Abstract**:

We are again confronted with one of the most vexing aspects of the advancement of technology: automation and AI technology cause the devaluation of human labor, resulting in unemployment. With this background, automatic person-job fit systems are promising solutions to promote the employment rate. The purpose of person-job fit is to calculate a matching score between the job seeker's resume and the job posting, determining whether the job seeker is suitable for the position. In this paper, we propose a new approach to person-job fit that characterizes the hidden preference derived from the job seeker's career path. We categorize and utilize three types of preferences in the career path: consistency, likeness, and continuity. We prove that understanding the career path enables us to provide more appropriate career suggestions to job seekers. To demonstrate the practical value of our proposed model, we conduct extensive experiments on real-world data extracted from an online recruitment platform and then present detailed cases to show how the career path matter in person-job fit.

----

## [937] Efficient Representation Learning of Satellite Image Time Series and Their Fusion for Spatiotemporal Applications

**Authors**: *Poonam Goyal, Arshveer Kaur, Arvind Ram, Navneet Goyal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28686](https://doi.org/10.1609/aaai.v38i8.28686)

**Abstract**:

Satellite data bolstered by their increasing accessibility is leading to many endeavors of automated monitoring of the earth's surface for various applications. Such applications demand high spatial resolution images at a temporal resolution of a few days which entails the challenge of processing a huge volume of image time series data. To overcome this computing bottleneck, we present PatchNet, a bespoke adaptation of beam search and attention mechanism. PatchNet is an automated patch selection neural network that requires only a partial spatial traversal of an image time series and yet achieves impressive results. Satellite systems face a trade-off between spatial and temporal resolutions due to budget/technical constraints e.g., Landsat-8/9 or Sentinel-2 have high spatial resolution whereas, MODIS has high temporal resolution. To deal with the limitation of coarse temporal resolution, we propose FuSITSNet, a twofold feature-based generic fusion model with multimodal learning in a contrastive setting. It produces a learned representation after fusion of two satellite image time series leveraging finer spatial resolution of Landsat and finer temporal resolution of MODIS. The patch alignment module of FuSITSNet aligns the PatchNet processed patches of Landsat-8 with the corresponding MODIS regions to incorporate its finer resolution temporal features. The untraversed patches are handled by the cross-modality attention which highlights additional hot spot features from the two modalities. We conduct extensive experiments on more than 2000 counties of US for crop yield, snow cover, and solar energy prediction and show that even one-fourth spatial processing of image time series produces state-of-the-art results.  FuSITSNet outperforms the predictions of single modality and data obtained using existing generative fusion models and allows for monitoring of dynamic phenomena using freely accessible images, thereby unlocking new opportunities.

----

## [938] Rethinking Reverse Distillation for Multi-Modal Anomaly Detection

**Authors**: *Zhihao Gu, Jiangning Zhang, Liang Liu, Xu Chen, Jinlong Peng, Zhenye Gan, Guannan Jiang, Annan Shu, Yabiao Wang, Lizhuang Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28687](https://doi.org/10.1609/aaai.v38i8.28687)

**Abstract**:

In recent years, there has been significant progress in employing color images for anomaly detection in industrial scenarios, but it is insufficient for identifying anomalies that are invisible in RGB images alone. As a supplement, introducing extra modalities such as depth and surface normal maps can be helpful to detect these anomalies. To this end, we present a novel Multi-Modal Reverse Distillation (MMRD) paradigm that consists of a frozen multi-modal teacher encoder to generate distillation targets and a learnable student decoder targeting to restore multi-modal representations from the teacher. Specifically, the teacher extracts complementary visual features from different modalities via a siamese architecture and then parameter-freely fuses these information from multiple levels as the targets of distillation. For the student, it learns modality-related priors from the teacher representations of normal training data and performs interaction between them to form multi-modal representations for target reconstruction. Extensive experiments show that our MMRD outperforms recent state-of-the-art methods on both anomaly detection and localization on MVTec-3D AD and Eyecandies benchmarks. Codes will be available upon acceptance.

----

## [939] LGMRec: Local and Global Graph Learning for Multimodal Recommendation

**Authors**: *Zhiqiang Guo, Jianjun Li, Guohui Li, Chaoyang Wang, Si Shi, Bin Ruan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28688](https://doi.org/10.1609/aaai.v38i8.28688)

**Abstract**:

The multimodal recommendation has gradually become the infrastructure of online media platforms, enabling them to provide personalized service to users through a joint modeling of user historical behaviors (e.g., purchases, clicks) and item various modalities (e.g., visual and textual). The majority of existing studies typically focus on utilizing modal features or modal-related graph structure to learn user local interests. Nevertheless, these approaches encounter two limitations: (1) Shared updates of user ID embeddings result in the consequential coupling between collaboration and multimodal signals; (2) Lack of exploration into robust global user interests to alleviate the sparse interaction problems faced by local interest modeling. To address these issues, we propose a novel Local and Global Graph Learning-guided Multimodal Recommender (LGMRec), which jointly models local and global user interests. Specifically, we present a local graph embedding module to independently learn collaborative-related and modality-related embeddings of users and items with local topological relations. Moreover, a global hypergraph embedding module is designed to capture global user and item embeddings by modeling insightful global dependency relations. The global embeddings acquired within the hypergraph embedding space can then be combined with two decoupled local embeddings to improve the accuracy and robustness of recommendations. Extensive experiments conducted on three benchmark datasets demonstrate the superiority of our LGMRec over various state-of-the-art recommendation baselines, showcasing its effectiveness in modeling both local and global user interests.

----

## [940] Intra- and Inter-group Optimal Transport for User-Oriented Fairness in Recommender Systems

**Authors**: *Zhongxuan Han, Chaochao Chen, Xiaolin Zheng, Meng Li, Weiming Liu, Binhui Yao, Yuyuan Li, Jianwei Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28689](https://doi.org/10.1609/aaai.v38i8.28689)

**Abstract**:

Recommender systems are typically biased toward a small group of users, leading to severe unfairness in recommendation performance, i.e., User-Oriented Fairness (UOF) issue. Existing research on UOF exhibits notable limitations in two phases of recommendation models. In the training phase, current methods fail to tackle the root cause of the UOF issue, which lies in the unfair training process between advantaged and disadvantaged users. In the evaluation phase, the current UOF metric lacks the ability to comprehensively evaluate varying cases of unfairness. In this paper, we aim to address the aforementioned limitations and ensure recommendation models treat user groups of varying activity levels equally. In the training phase, we propose a novel Intra- and Inter-GrOup Optimal Transport framework (II-GOOT) to alleviate the data sparsity problem for disadvantaged users and narrow the training gap between advantaged and disadvantaged users. In the evaluation phase, we introduce a novel metric called ?-UOF, which enables the identification and assessment of various cases of UOF. This helps prevent recommendation models from leading to unfavorable fairness outcomes, where both advantaged and disadvantaged users experience subpar recommendation performance. We conduct extensive experiments on three real-world datasets based on four backbone recommendation models to prove the effectiveness of ?-UOF and the efficiency of our proposed II-GOOT.

----

## [941] A Diffusion-Based Framework for Multi-Class Anomaly Detection

**Authors**: *Haoyang He, Jiangning Zhang, Hongxu Chen, Xuhai Chen, Zhishan Li, Xu Chen, Yabiao Wang, Chengjie Wang, Lei Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28690](https://doi.org/10.1609/aaai.v38i8.28690)

**Abstract**:

Reconstruction-based approaches have achieved remarkable outcomes in anomaly detection. The exceptional image reconstruction capabilities of recently popular diffusion models have sparked research efforts to utilize them for enhanced reconstruction of anomalous images. Nonetheless, these methods might face challenges related to the preservation of image categories and pixel-wise structural integrity in the more practical multi-class setting. To solve the above problems, we propose a Difusion-based Anomaly Detection (DiAD) framework for multi-class anomaly detection, which consists of a pixel-space autoencoder, a latent-space Semantic-Guided (SG) network with a connection to the stable diffusion’s denoising network, and a feature-space pre-trained feature extractor. Firstly, The SG network is proposed for reconstructing anomalous regions while preserving the original image’s semantic information. Secondly, we introduce Spatial-aware Feature Fusion (SFF) block to maximize reconstruction accuracy when dealing with extensively reconstructed areas. Thirdly, the input and reconstructed images are processed by a pre-trained feature extractor to generate anomaly maps based on features extracted at different scales. Experiments on MVTec-AD and VisA datasets demonstrate the effectiveness of our approach which surpasses the state-of-the-art methods, e.g., achieving 96.8/52.6 and 97.2/99.0 (AUROC/AP) for localization and detection respectively on multi-class MVTec-AD dataset. Code will be available at https://lewandofskee.github.io/projects/diad.

----

## [942] ADA-GAD: Anomaly-Denoised Autoencoders for Graph Anomaly Detection

**Authors**: *Junwei He, Qianqian Xu, Yangbangyan Jiang, Zitai Wang, Qingming Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28691](https://doi.org/10.1609/aaai.v38i8.28691)

**Abstract**:

Graph anomaly detection is crucial for identifying nodes that deviate from regular behavior within graphs, benefiting various domains such as fraud detection and social network. Although existing reconstruction-based methods have achieved considerable success, they may face the Anomaly Overfitting and Homophily Trap problems caused by the abnormal patterns in the graph, breaking the assumption that normal nodes are often better reconstructed than abnormal ones. Our observations indicate that models trained on graphs with fewer anomalies exhibit higher detection performance. Based on this insight, we introduce a novel two-stage framework called Anomaly-Denoised Autoencoders for Graph Anomaly Detection (ADA-GAD). In the first stage, we design a learning-free anomaly-denoised augmentation method to generate graphs with reduced anomaly levels. We pretrain graph autoencoders on these augmented graphs at multiple levels, which enables the graph autoencoders to capture normal patterns. In the next stage, the decoders are retrained for detection on the original graph, benefiting from the multi-level representations learned in the previous stage. Meanwhile, we propose the node anomaly distribution regularization to further alleviate Anomaly Overfitting. We validate the effectiveness of our approach through extensive experiments on both synthetic and real-world datasets.

----

## [943] ViSTec: Video Modeling for Sports Technique Recognition and Tactical Analysis

**Authors**: *Yuchen He, Zeqing Yuan, Yihong Wu, Liqi Cheng, Dazhen Deng, Yingcai Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28692](https://doi.org/10.1609/aaai.v38i8.28692)

**Abstract**:

The immense popularity of racket sports has fueled substantial demand in tactical analysis with broadcast videos. However, existing manual methods require laborious annotation, and recent attempts leveraging video perception models are limited to low-level annotations like ball trajectories, overlooking tactics that necessitate an understanding of stroke techniques. State-of-the-art action segmentation models also struggle with technique recognition due to frequent occlusions and motion-induced blurring in racket sports videos. To address these challenges, We propose ViSTec, a Video-based Sports Technique recognition model inspired by human cognition that synergizes sparse visual data with rich contextual insights. Our approach integrates a graph to explicitly model strategic knowledge in stroke sequences and enhance technique recognition with contextual inductive bias. A two-stage action perception model is jointly trained to align with the contextual knowledge in the graph. Experiments demonstrate that our method outperforms existing models by a significant margin. Case studies with experts from the Chinese national table tennis team validate our model's capacity to automate analysis for technical actions and tactical strategies. More details are available at: https://ViSTec2024.github.io/.

----

## [944] Label Attentive Distillation for GNN-Based Graph Classification

**Authors**: *Xiaobin Hong, Wenzhong Li, Chaoqun Wang, Mingkai Lin, Sanglu Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28693](https://doi.org/10.1609/aaai.v38i8.28693)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as a powerful tool for modeling graph-structured data, exhibiting remarkable potential in applications such as social networks, recommendation systems, and molecular structures. However, the conventional GNNs perform node-level feature aggregation from neighbors without considering  graph-label information, which leads to the misaligned embedding problem that may cause a detrimental effect on graph-level tasks such as graph classification. In this paper, we propose a novel label-attentive distillation method called LAD-GNN for graph representation learning to solve this problem. It alternatively trains a teacher model and a student GNN with a distillation-based approach. In the teacher model, a label-attentive encoder is proposed to encode the label information fusing with the node features to generate ideal embedding. In the student model, the ideal embedding is used as intermediate supervision to urge the student GNN to learn class-friendly node embedding to facilitate graph-level tasks. Generally, LAD-GNN is an enhanced GNN training approach that can be incorporated with arbitrary GNN backbone to improve performance without significant increase of computational cost. Extensive experiments with 7 GNN backbones based on 10 benchmark datasets show that LAD-GNN improves the SOTA GNNs in graph classification accuracy. The source codes of LAD-GNN are publicly available on https://github.com/XiaobinHong/LAD-GNN.

----

## [945] DAG-Aware Variational Autoencoder for Social Propagation Graph Generation

**Authors**: *Dongpeng Hou, Chao Gao, Xuelong Li, Zhen Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28694](https://doi.org/10.1609/aaai.v38i8.28694)

**Abstract**:

Propagation models in social networks are critical, with extensive applications across various fields and downstream tasks. However, existing propagation models are often oversimplified, scenario-specific, and lack real-world user social attributes. These limitations detaching from real-world analysis lead to inaccurate representations of the propagation process in social networks. To address these issues, we propose a User Features Attention-based DAG-Aware Variational Autoencoder (DAVA) for propagation graph generation. First, nearly 1 million pieces of user attributes data are collected. Then DAVA can integrate the analysis of propagation graph topology and corresponding user attributes as prior knowledge. By leveraging a lightweight attention-based framework and a sliding window mechanism based on BFS permutations weighted by user influence, DAVA significantly enhances the ability to generate realistic, large-scale propagation data, yielding graph scales ten times greater than those produced by existing SOTA methods. Every module of DAVA has flexibility and extension that allows for easy substitution to suit other generation tasks. Additionally, we provide a comprehensive evaluation of DAVA, one focus is the effectiveness of generated data in improving the performance of downstream tasks. During the generation process, we discover the Credibility Erosion Effect by modifying the generation rules, revealing a social phenomenon in social network propagation.

----

## [946] Social-Aware Group Display Configuration in VR Conference

**Authors**: *Bay-Yuan Hsu, Chih-Ya Shen, Hao Shan Yuan, Wang-Chien Lee, De-Nian Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28695](https://doi.org/10.1609/aaai.v38i8.28695)

**Abstract**:

Virtual Reality (VR) has emerged due to advancements in hardware and computer graphics. During the pandemic, conferences and exhibitions leveraging VR have gained attention. However, large-scale VR conferences, face a significant problem not yet studied in the literature -- displaying too many irrelevant users on the screen which may negatively impact the user experience. To address this issue, we formulate a new research problem, Social-Aware VR Conference Group Display Configuration (SVGD). Accordingly, we design the Social Utility-Aware VR Conference Group Formation (SVC) algorithm, which is a 2-approximation algorithm to SVGD.  SVC iteratively selects either the P-Configuration or S-Configuration based on their effective ratios. This ensures that in each iteration, SVC identifies and chooses the solution with the highest current effectiveness. Experiments on real metaverse datasets show that the proposed SVC outperforms 11 baselines by 75% in terms of solution quality.

----

## [947] AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model

**Authors**: *Teng Hu, Jiangning Zhang, Ran Yi, Yuzhen Du, Xu Chen, Liang Liu, Yabiao Wang, Chengjie Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28696](https://doi.org/10.1609/aaai.v38i8.28696)

**Abstract**:

Anomaly inspection plays an important role in industrial manufacture. Existing anomaly inspection methods are limited in their performance due to insufficient anomaly data. Although anomaly generation methods have been proposed to augment the anomaly data, they either suffer from poor generation authenticity or inaccurate alignment between the generated anomalies and masks. To address the above problems, we propose AnomalyDiffusion, a novel diffusion-based few-shot anomaly generation model, which utilizes the strong prior information of latent diffusion model learned from large-scale dataset to enhance the generation authenticity under few-shot training data. Firstly, we propose Spatial Anomaly Embedding, which consists of a learnable anomaly embedding and a spatial embedding encoded from an anomaly mask, disentangling the anomaly information into anomaly appearance and location information. Moreover, to improve the alignment between the generated anomalies and the anomaly masks, we introduce a novel Adaptive Attention Re-weighting Mechanism. Based on the disparities between the generated anomaly image and normal sample, it dynamically guides the model to focus more on the areas with less noticeable generated anomalies, enabling generation of accurately-matched anomalous image-mask pairs. Extensive experiments demonstrate that our model significantly outperforms the state-of-the-art methods in generation authenticity and diversity, and effectively improves the performance of downstream anomaly inspection tasks. The code and data are available in https://github.com/sjtuplayer/anomalydiffusion.

----

## [948] Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation

**Authors**: *Tianhao Huang, Xuan Pan, Xiangrui Cai, Ying Zhang, Xiaojie Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28697](https://doi.org/10.1609/aaai.v38i8.28697)

**Abstract**:

Next Point-of-Interests (POIs) recommendation task aims to provide a dynamic ranking of POIs based on users' current check-in trajectories. The recommendation performance of this task is contingent upon a comprehensive understanding of users' personalized behavioral patterns through Location-based Social Networks (LBSNs) data. While prior studies have adeptly captured sequential patterns and transitional relationships within users' check-in trajectories, a noticeable gap persists in devising a mechanism for discerning specialized behavioral patterns during distinct time slots, such as noon, afternoon, or evening. In this paper, we introduce an innovative data structure termed the ``Mobility Tree'', tailored for hierarchically describing users' check-in records. The Mobility Tree encompasses multi-granularity time slot nodes to learn user preferences across varying temporal periods. Meanwhile, we propose the Mobility Tree Network (MTNet), a multitask framework for personalized preference learning based on Mobility Trees. We develop a four-step node interaction operation to propagate feature information from the leaf nodes to the root node. Additionally, we adopt a multitask training strategy to push the model towards learning a robust representation. The comprehensive experimental results demonstrate the superiority of MTNet over eleven state-of-the-art next POI recommendation models across three real-world LBSN datasets, substantiating the efficacy of time slot preference learning facilitated by Mobility Tree.

----

## [949] ReGCL: Rethinking Message Passing in Graph Contrastive Learning

**Authors**: *Cheng Ji, Zixuan Huang, Qingyun Sun, Hao Peng, Xingcheng Fu, Qian Li, Jianxin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28698](https://doi.org/10.1609/aaai.v38i8.28698)

**Abstract**:

Graph contrastive learning (GCL) has demonstrated remarkable efficacy in graph representation learning. However, previous studies have overlooked the inherent conflict that arises when employing graph neural networks (GNNs) as encoders for node-level contrastive learning. This conflict pertains to the partial incongruity between the feature aggregation mechanism of graph neural networks and the embedding distinction characteristic of contrastive learning. Theoretically, to investigate the location and extent of the conflict, we analyze the participation of message-passing from the gradient perspective of InfoNCE loss. Different from contrastive learning in other domains, the conflict in GCL arises due to the presence of certain samples that contribute to both the gradients of positive and negative simultaneously under the manner of message passing, which are opposite optimization directions. To further address the conflict issue, we propose a practical framework called ReGCL, which utilizes theoretical findings of GCL gradients to effectively improve graph contrastive learning. Specifically, two gradient-based strategies are devised in terms of both message passing and loss function to mitigate the conflict. Firstly, a gradient-guided structure learning method is proposed in order to acquire a structure that is adapted to contrastive learning principles. Secondly, a gradient-weighted InfoNCE loss function is designed to reduce the impact of false negative samples with high probabilities, specifically from the standpoint of the graph encoder. Extensive experiments demonstrate the superiority of the proposed method in comparison to state-of-the-art baselines across various node classification benchmarks.

----

## [950] D3: A Methodological Exploration of Domain Division, Modeling, and Balance in Multi-Domain Recommendations

**Authors**: *Pengyue Jia, Yichao Wang, Shanru Lin, Xiaopeng Li, Xiangyu Zhao, Huifeng Guo, Ruiming Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28699](https://doi.org/10.1609/aaai.v38i8.28699)

**Abstract**:

To enhance the efficacy of multi-scenario services in industrial recommendation systems, the emergence of multi-domain recommendation has become prominent, which entails simultaneous modeling of all domains through a unified model, effectively capturing commonalities and differences among them. However, current methods rely on manual domain partitioning, which overlook the intricate domain relationships and the heterogeneity of different domains during joint optimization, hindering the integration of domain commonalities and differences. To address these challenges, this paper proposes a universal and flexible framework D3 aimed at optimizing the multi-domain recommendation pipeline from three key aspects. Firstly, an attention-based domain adaptation module is introduced to automatically identify and incorporate domain-sensitive features during training. Secondly, we propose a fusion gate module that enables the seamless integration of commonalities and diversities among domains, allowing for implicit characterization of intricate domain relationships. Lastly, we tackle the issue of joint optimization by deriving loss weights from two complementary viewpoints: domain complexity and domain specificity, alleviating inconsistencies among different domains during the training phase. Experiments on three public datasets demonstrate the effectiveness and superiority of our proposed framework. In addition, D3 has been implemented on a real-life, high-traffic internet platform catering to millions of users daily.

----

## [951] Graph Invariant Learning with Subgraph Co-mixup for Out-of-Distribution Generalization

**Authors**: *Tianrui Jia, Haoyang Li, Cheng Yang, Tao Tao, Chuan Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28700](https://doi.org/10.1609/aaai.v38i8.28700)

**Abstract**:

Graph neural networks (GNNs) have been demonstrated to perform well in graph representation learning, but always lacking in generalization capability when tackling out-of-distribution (OOD) data. Graph invariant learning methods, backed by the invariance principle among defined multiple environments, have shown effectiveness in dealing with this issue. However, existing methods heavily rely on well-predefined or accurately generated environment partitions, which are hard to be obtained in practice, leading to sub-optimal OOD generalization performances.
In this paper, we propose a novel graph invariant learning method based on invariant and variant patterns co-mixup strategy, which is capable of jointly generating mixed multiple environments and capturing invariant patterns from the mixed graph data. Specifically, we first adopt a subgraph extractor to identify invariant subgraphs. Subsequently, we design one novel co-mixup strategy, i.e., jointly conducting environment mixup and invariant mixup. For the environment mixup, we mix the variant environment-related subgraphs so as to generate sufficiently diverse multiple environments, which is important to guarantee the quality of the graph invariant learning. For the invariant mixup, we mix the invariant subgraphs, further encouraging to capture invariant patterns behind graphs while getting rid of spurious correlations for OOD generalization. We demonstrate that the proposed environment mixup and invariant mixup can mutually promote each other.
Extensive experiments on both synthetic and real-world datasets demonstrate that our method significantly outperforms state-of-the-art under various distribution shifts.

----

## [952] Enhancing Multi-Scale Diffusion Prediction via Sequential Hypergraphs and Adversarial Learning

**Authors**: *Pengfei Jiao, Hongqian Chen, Qing Bao, Wang Zhang, Huaming Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28701](https://doi.org/10.1609/aaai.v38i8.28701)

**Abstract**:

Information diffusion prediction plays a crucial role in understanding the propagation of information in social networks, encompassing both macroscopic and microscopic prediction tasks. Macroscopic prediction estimates the overall impact of information diffusion, while microscopic prediction focuses on identifying the next user to be influenced. While prior research often concentrates on one of these aspects, a few tackle both concurrently. These two tasks provide complementary insights into the diffusion process at different levels, revealing common traits and unique attributes. The exploration of leveraging common features across these tasks to enhance information prediction remains an underexplored avenue. In this paper, we propose an intuitive and effective model that addresses both macroscopic and microscopic prediction tasks. Our approach considers the interactions and dynamics among cascades at the macro level and incorporates the social homophily of users in social networks at the micro level. Additionally, we introduce adversarial training and orthogonality constraints to ensure the integrity of shared features. Experimental results on four datasets demonstrate that our model significantly outperforms state-of-the-art methods.

----

## [953] Multi-Domain Recommendation to Attract Users via Domain Preference Modeling

**Authors**: *Hyunjun Ju, SeongKu Kang, Dongha Lee, Junyoung Hwang, Sanghwan Jang, Hwanjo Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28702](https://doi.org/10.1609/aaai.v38i8.28702)

**Abstract**:

Recently, web platforms are operating various service domains simultaneously. Targeting a platform that operates multiple service domains, we introduce a new task, Multi-Domain Recommendation to Attract Users (MDRAU), which recommends items from multiple ``unseen'' domains with which each user has not interacted yet, by using knowledge from the user's ``seen'' domains. In this paper, we point out two challenges of MDRAU task. First, there are numerous possible combinations of mappings from seen to unseen domains because users have usually interacted with a different subset of service domains. Second, a user might have different preference for each of the target unseen domains, which requires recommendations to reflect users' preference on domains as well as items. To tackle these challenges, we propose DRIP framework that models users' preference at two levels (i.e., domain and item) and learns various seen-unseen domain mappings in a unified way with masked domain modeling. Our extensive experiments demonstrate the effectiveness of DRIP in MDRAU task and its ability to capture users' domain-level preferences.

----

## [954] Few Shot Part Segmentation Reveals Compositional Logic for Industrial Anomaly Detection

**Authors**: *Soopil Kim, Sion An, Philip Chikontwe, Myeongkyun Kang, Ehsan Adeli, Kilian M. Pohl, Sanghyun Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28703](https://doi.org/10.1609/aaai.v38i8.28703)

**Abstract**:

Logical anomalies (LA) refer to data violating underlying logical constraints e.g., the quantity, arrangement, or composition of components within an image. Detecting accurately such anomalies requires models to reason about various component types through segmentation. However, curation of pixel-level annotations for semantic segmentation is both time-consuming and expensive. Although there are some prior few-shot or unsupervised co-part segmentation algorithms, they often fail on images with industrial object. These images have components with similar textures and shapes, and a precise differentiation proves challenging. In this study, we introduce a novel component segmentation model for LA detection that leverages a few labeled samples and unlabeled images sharing logical constraints. To ensure consistent segmentation across unlabeled images, we employ a histogram matching loss in conjunction with an entropy loss. As segmentation predictions play a crucial role, we propose to enhance both local and global sample validity detection by capturing key aspects from visual semantics via three memory banks: class histograms, component composition embeddings and patch-level representations. For effective LA detection, we propose an adaptive scaling strategy to standardize anomaly scores from different memory banks in inference. Extensive experiments on the public benchmark MVTec LOCO AD reveal our method achieves 98.1% AUROC in LA detection vs. 89.6% from competing methods.

----

## [955] VITA: 'Carefully Chosen and Weighted Less' Is Better in Medication Recommendation

**Authors**: *Taeri Kim, Jiho Heo, Hongil Kim, Kijung Shin, Sang-Wook Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28704](https://doi.org/10.1609/aaai.v38i8.28704)

**Abstract**:

We address the medication recommendation problem, which aims to recommend effective medications for a patient's current visit by utilizing information (e.g., diagnoses and procedures) given at the patient's current and past visits. While there exist a number of recommender systems designed for this problem, we point out that they are challenged in accurately capturing the relation (spec., the degree of relevance) between the current and each of the past visits for the patient when obtaining her current health status, which is the basis for recommending medications. To address this limitation, we propose a novel medication recommendation framework, named VITA, based on the following two novel ideas: (1) relevant-Visit selectIon; (2) Target-aware Attention. Through extensive experiments using real-world datasets, we demonstrate the superiority of VITA (spec., up to 5.67% higher accuracy, in terms of Jaccard, than the best competitor) and the effectiveness of its two core ideas. The code is available at https://github.com/jhheo0123/VITA.

----

## [956] Optimal Quasi-clique: Hardness, Equivalence with Densest-k-Subgraph, and Quasi-partitioned Community Mining

**Authors**: *Aritra Konar, Nicholas D. Sidiropoulos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28705](https://doi.org/10.1609/aaai.v38i8.28705)

**Abstract**:

Dense subgraph discovery (DSD) is a key primitive in graph mining that typically deals with extracting cliques and near-cliques. In this paper, we revisit the optimal quasi-clique (OQC) formulation for DSD and establish that it is NP--hard. In addition, we reveal the hitherto unknown property that OQC can be used to explore the entire spectrum of densest subgraphs of all distinct sizes by appropriately varying a single hyperparameter, thereby forging an intimate link with the classic densest-k-subgraph problem (DkS). We corroborate these findings on real-world graphs by applying the simple greedy algorithm for OQC with improved hyperparameter tuning, to quickly generate high-quality approximations of the size-density frontier. Our findings indicate that OQC not only extracts high quality (near)-cliques, but also large and loosely-connected  subgraphs that exhibit well defined local community structure. The latter discovery is particularly intriguing, since OQC is not explicitly geared towards community detection.

----

## [957] Learning Persistent Community Structures in Dynamic Networks via Topological Data Analysis

**Authors**: *Dexu Kong, Anping Zhang, Yang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28706](https://doi.org/10.1609/aaai.v38i8.28706)

**Abstract**:

Dynamic community detection methods often lack effective mechanisms to ensure temporal consistency, hindering the analysis of network evolution. In this paper, we propose a novel deep graph clustering framework with temporal consistency regularization on inter-community structures, inspired by the concept of minimal network topological changes within short intervals. Specifically, to address the representation collapse problem, we first introduce MFC, a matrix factorization-based deep graph clustering algorithm that preserves node embedding. Based on static clustering results, we construct probabilistic community networks and compute their persistence homology, a robust topological measure, to assess structural similarity between them. Moreover, a novel neural network regularization TopoReg is introduced to ensure the preservation of topological similarity between inter-community structures over time intervals. Our approach enhances temporal consistency and clustering accuracy on real-world datasets with both fixed and varying numbers of communities. It is also a pioneer application of TDA in temporally persistent community detection, offering an insightful contribution to field of network analysis. Code and data are available at the public git repository: https://github.com/kundtx/MFC-TopoReg.

----

## [958] Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting

**Authors**: *Weiyang Kong, Ziyu Guo, Yubao Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28707](https://doi.org/10.1609/aaai.v38i8.28707)

**Abstract**:

Traffic flow forecasting is a classical spatio-temporal data mining problem with many real-world applications. Recently, various methods based on Graph Neural Networks (GNN) have been proposed for the problem and achieved impressive prediction performance. However, we argue that the majority of existing methods disregarding the importance of certain nodes (referred to as pivotal nodes) that naturally exhibit extensive connections with multiple other nodes. Predicting on pivotal nodes poses a challenge due to their complex spatio-temporal dependencies compared to other nodes. In this paper, we propose a novel GNN-based method called Spatio-Temporal Pivotal Graph Neural Networks (STPGNN) to address the above limitation. We introduce a pivotal node identification module for identifying pivotal nodes. We propose a novel pivotal graph convolution module, enabling precise capture of spatio-temporal dependencies centered around pivotal nodes. Moreover, we propose a parallel framework capable of extracting spatio-temporal traffic features on both pivotal and non-pivotal nodes. Experiments on seven real-world traffic datasets verify our proposed method's effectiveness and efficiency compared to state-of-the-art baselines.

----

## [959] Knowledge-Aware Explainable Reciprocal Recommendation

**Authors**: *Kai-Huang Lai, Zhe-Rui Yang, Pei-Yuan Lai, Chang-Dong Wang, Mohsen Guizani, Min Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28708](https://doi.org/10.1609/aaai.v38i8.28708)

**Abstract**:

Reciprocal recommender systems (RRS) have been widely used in online platforms such as online dating and recruitment. They can simultaneously fulfill the needs of both parties involved in the recommendation process. Due to the inherent nature of the task, interaction data is relatively sparse compared to other recommendation tasks. Existing works mainly address this issue through content-based recommendation methods. However, these methods often implicitly model textual information from a unified perspective, making it challenging to capture the distinct intentions held by each party, which further leads to limited performance and the lack of interpretability. In this paper, we propose a Knowledge-Aware Explainable Reciprocal Recommender System (KAERR), which models metapaths between two parties independently, considering their respective perspectives and requirements. Various metapaths are fused using an attention-based mechanism, where the attention weights unveil dual-perspective preferences and provide recommendation explanations for both parties. Extensive experiments on two real-world datasets from diverse scenarios demonstrate that the proposed model outperforms state-of-the-art baselines, while also delivering compelling reasons for recommendations to both parties.

----

## [960] Adaptive Hardness Negative Sampling for Collaborative Filtering

**Authors**: *Riwei Lai, Rui Chen, Qilong Han, Chi Zhang, Li Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28709](https://doi.org/10.1609/aaai.v38i8.28709)

**Abstract**:

Negative sampling is essential for implicit collaborative filtering to provide proper negative training signals so as to achieve desirable performance. We experimentally unveil a common limitation of all existing negative sampling methods that they can only select negative samples of a fixed hardness level, leading to the false positive problem (FPP) and false negative problem (FNP). We then propose a new paradigm called adaptive hardness negative sampling (AHNS) and discuss its three key criteria. By adaptively selecting negative samples with appropriate hardnesses during the training process, AHNS can well mitigate the impacts of FPP and FNP. Next, we present a concrete instantiation of AHNS called AHNS_{p<0}, and theoretically demonstrate that AHNS_{p<0} can fit the three criteria of AHNS well and achieve a larger lower bound of normalized discounted cumulative gain. Besides, we note that existing negative sampling methods can be regarded as more relaxed cases of AHNS. Finally, we conduct comprehensive experiments, and the results show that AHNS_{p<0} can consistently and substantially outperform several state-of-the-art competitors on multiple datasets.

----

## [961] MDFL: Multi-Domain Diffusion-Driven Feature Learning

**Authors**: *Daixun Li, Weiying Xie, Jiaqing Zhang, Yunsong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28710](https://doi.org/10.1609/aaai.v38i8.28710)

**Abstract**:

High-dimensional images, known for their rich semantic information, are widely applied in remote sensing and other fields. The spatial information in these images reflects the object's texture features, while the spectral information reveals the potential spectral representations across different bands. Currently, the understanding of high-dimensional images remains limited to a single-domain perspective with performance degradation. Motivated by the masking texture effect observed in the human visual system, we present a multi-domain diffusion-driven feature learning network (MDFL) , a scheme to redefine the effective information domain that the model really focuses on. This method employs diffusion-based posterior sampling to explicitly consider joint information interactions between the high-dimensional manifold structures in the spectral, spatial, and frequency domains, thereby eliminating the influence of masking texture effects in visual models. Additionally, we introduce a feature reuse mechanism to gather deep and raw features of high-dimensional data. We demonstrate that MDFL significantly improves the feature extraction performance of high-dimensional data, thereby providing a powerful aid for revealing the intrinsic patterns and structures of such data. The experimental results on three multi-modal remote sensing datasets show that MDFL reaches an average overall accuracy of 98.25%, outperforming various state-of-the-art baseline schemes. Code available at  https://github.com/LDXDU/MDFL-AAAI-24.

----

## [962] CoreRec: A Counterfactual Correlation Inference for Next Set Recommendation

**Authors**: *Kexin Li, Chengjiang Long, Shengyu Zhang, Xudong Tang, Zhichao Zhai, Kun Kuang, Jun Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28711](https://doi.org/10.1609/aaai.v38i8.28711)

**Abstract**:

Next set recommendation aims to predict the items that are likely to be bought in the next purchase. Central to this endeavor is the task of capturing intra-set and cross-set correlations among items. However, the modeling of cross-set correlations poses challenges due to specific issues. Primarily, these correlations are often implicit, and the prevailing approach of establishing an indiscriminate link across the entire set of objects neglects factors like purchase frequency and correlations between purchased items. Such hastily formed connections across sets introduce substantial noise. Additionally, the preeminence of high-frequency items in numerous sets could potentially overshadow and distort correlation modeling with respect to low-frequency items. Thus, we devoted to mitigating misleading inter-set correlations. With a fresh perspective rooted in causality, we delve into the question of whether correlations between a particular item and items from other sets should be relied upon for item representation learning and set prediction. Technically, we introduce the Counterfactual Correlation Inference framework for next set recommendation, denoted as CoreRec. This framework establishes a counterfactual scenario in which the recommendation model impedes cross-set correlations to generate intervened predictions. By contrasting these intervened predictions with the original ones, we gauge the causal impact of inter-set neighbors on set prediction—essentially assessing whether they contribute to spurious correlations. During testing, we introduce a post-trained switch module that selects between set-aware item representations derived from either the original or the counterfactual scenarios. To validate our approach, we extensively experiment using three real-world datasets, affirming both the effectiveness of CoreRec and the cogency of our analytical approach.

----

## [963] Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations

**Authors**: *Lei Li, Jianxun Lian, Xiao Zhou, Xing Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28712](https://doi.org/10.1609/aaai.v38i8.28712)

**Abstract**:

Retrieval models aim at selecting a small set of item candidates which match the preference of a given user. They play a vital role in large-scale recommender systems since subsequent models such as rankers highly depend on the quality of item candidates. However, most existing retrieval models employ a single-round inference paradigm, which may not adequately capture the dynamic nature of user preferences and stuck in one area in the item space. In this paper, we propose Ada-Retrieval, an adaptive multi-round retrieval paradigm for recommender systems that iteratively refines user representations to better capture potential candidates in the full item space. Ada-Retrieval comprises two key modules: the item representation adapter and the user representation adapter, designed to inject context information into items' and users' representations. The framework maintains a model-agnostic design, allowing seamless integration with various backbone models such as RNNs or Transformers. We perform experiments on three widely used public datasets, incorporating five powerful sequential recommenders as backbone models. Our results demonstrate that Ada-Retrieval significantly enhances the performance of various base models, with consistent improvements observed across different datasets. Our code and data are publicly available at: https://github.com/ll0ruc/Ada-Retrieval.

----

## [964] CONSIDER: Commonalities and Specialties Driven Multilingual Code Retrieval Framework

**Authors**: *Rui Li, Liyang He, Qi Liu, Yuze Zhao, Zheng Zhang, Zhenya Huang, Yu Su, Shijin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28713](https://doi.org/10.1609/aaai.v38i8.28713)

**Abstract**:

Multilingual code retrieval aims to find code snippets relevant to a user's query from a multilingual codebase, which plays a crucial role in software development and expands their application scenarios compared to classical monolingual code retrieval. Despite the performance improvements achieved by previous studies, two crucial problems are overlooked in the multilingual scenario. First, certain programming languages face data scarcity in specific domains, resulting in limited representation capabilities within those domains. Second, different programming languages can be used interchangeably within the same domain, making it challenging for multilingual models to accurately identify the intended programming language of a user's query. To address these issues, we propose the CommONalities and SpecIalties Driven Multilingual CodE Retrieval Framework (CONSIDER), which includes two modules. The first module enhances the representation of various programming languages by modeling pairwise and global commonalities among them. The second module introduces a novel contrastive learning negative sampling algorithm that leverages language confusion to automatically extract specific language features. Through our experiments, we confirm the significant benefits of our model in real-world multilingual code retrieval scenarios in various aspects. Furthermore, an evaluation demonstrates the effectiveness of our proposed CONSIDER framework in monolingual scenarios as well. Our source code is available at https://github.com/smsquirrel/consider.

----

## [965] UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models

**Authors**: *Xiaoxi Li, Yujia Zhou, Zhicheng Dou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28714](https://doi.org/10.1609/aaai.v38i8.28714)

**Abstract**:

Generative information retrieval, encompassing two major tasks of Generative Document Retrieval (GDR) and Grounded Answer Generation (GAR), has gained significant attention in natural language processing. Existing methods for GDR and GAR rely on separate retrieval and reader modules, which hinder simultaneous optimization. To overcome this, we present UniGen, a Unified Generative framework for retrieval and question answering that integrates both tasks into a single generative model leveraging the capabilities of large language models. UniGen employs a shared encoder and two distinct decoders for generative retrieval and question answering. To facilitate the learning of both tasks, we introduce connectors, generated by large language models, to bridge the gaps between query inputs and generation targets, as well as between document identifiers and answers. Furthermore, we propose an iterative enhancement strategy that leverages generated answers and retrieved documents to iteratively improve both tasks. Through extensive experiments on the MS MARCO and NQ datasets, we demonstrate the effectiveness of UniGen, showcasing its superior performance in both retrieval and question answering tasks.

----

## [966] MESED: A Multi-Modal Entity Set Expansion Dataset with Fine-Grained Semantic Classes and Hard Negative Entities

**Authors**: *Yangning Li, Tingwei Lu, Hai-Tao Zheng, Yinghui Li, Shulin Huang, Tianyu Yu, Jun Yuan, Rui Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28715](https://doi.org/10.1609/aaai.v38i8.28715)

**Abstract**:

The Entity Set Expansion (ESE) task aims to expand a handful of seed entities with new entities belonging to the same semantic class. Conventional ESE methods are based on mono-modality (i.e., literal modality), which struggle to deal with complex entities in the real world such as (1) Negative entities with fine-grained semantic differences. (2) Synonymous entities. (3) Polysemous entities. (4) Long-tailed entities. These challenges prompt us to propose novel Multi-modal Entity Set Expansion (MESE), where models integrate information from multiple modalities to represent entities. Intuitively, the benefits of multi-modal information for ESE are threefold: (1) Different modalities can provide complementary information. (2) Multi-modal information provides a unified signal via common visual properties for the same semantic class or entity. (3) Multi-modal information offers robust alignment signals for synonymous entities. To assess model performance in MESE, we constructed the MESED dataset which is the first multi-modal dataset for ESE with large-scale and elaborate manual calibration. A powerful multi-modal model MultiExpan is proposed which is pre-trained on four multimodal pre-training tasks. The extensive experiments and analyses on MESED demonstrate the high quality of the dataset and the effectiveness of our MultiExpan, as well as pointing the direction for future research. The benchmark and code are public at https://github.com/THUKElab/MESED.

----

## [967] A Generalized Neural Diffusion Framework on Graphs

**Authors**: *Yibo Li, Xiao Wang, Hongrui Liu, Chuan Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28716](https://doi.org/10.1609/aaai.v38i8.28716)

**Abstract**:

Recent studies reveal the connection between GNNs and the diffusion process, which motivates many diffusion based GNNs to be proposed. However, since these two mechanisms are closely related, one fundamental question naturally arises: Is there a general diffusion framework that can formally unify these GNNs? The answer to this question can not only deepen our understanding of the learning process of GNNs, but also may open a new door to design a broad new class of GNNs. In this paper, we propose a general diffusion equation framework with the fidelity term, which formally establishes the relationship between the diffusion process with more GNNs. Meanwhile, with this framework, we identify one characteristic of graph diffusion networks, i.e., the current neural diffusion process only corresponds to the first-order diffusion equation. However, by an experimental investigation, we show that the labels of high-order neighbors actually appear monophily property, which induces the similarity based on labels among high-order neighbors without requiring the similarity among first-order neighbors. This discovery motives to design a new high-order neighbor-aware diffusion equation, and derive a new type of graph diffusion network (HiD-Net) based on the framework. With the high-order diffusion equation, HiD-Net is more robust against attacks and works on both homophily and heterophily graphs. We not only theoretically analyze the relation between HiD-Net with high-order random walk, but also provide a theoretical convergence guarantee. Extensive experimental results well demonstrate the effectiveness of HiD-Net over state-of-the-art graph diffusion networks.

----

## [968] Learning to Rank in Generative Retrieval

**Authors**: *Yongqi Li, Nan Yang, Liang Wang, Furu Wei, Wenjie Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28717](https://doi.org/10.1609/aaai.v38i8.28717)

**Abstract**:

Generative retrieval stands out as a promising new paradigm in text retrieval that aims to generate identifier strings of relevant passages as the retrieval target.  This generative paradigm taps into powerful generative language models, distinct from traditional sparse or dense retrieval methods. However, only learning to generate is insufficient for generative retrieval. Generative retrieval learns to generate identifiers of relevant passages as an intermediate goal and then converts predicted identifiers into the final passage rank list.  The disconnect between the learning objective of autoregressive models and the desired passage ranking target leads to a learning gap. To bridge this gap, we propose a learning-to-rank framework for generative retrieval, dubbed LTRGR. LTRGR enables generative retrieval to learn to rank passages directly, optimizing the autoregressive model toward the final passage ranking target via a rank loss. This framework only requires an additional learning-to-rank training phase to enhance current generative retrieval systems and does not add any burden to the inference stage. We conducted experiments on three public benchmarks, and the results demonstrate that LTRGR achieves state-of-the-art performance among generative retrieval methods. The code and checkpoints are released at https://github.com/liyongqi67/LTRGR.

----

## [969] Urban Region Embedding via Multi-View Contrastive Prediction

**Authors**: *Zechen Li, Weiming Huang, Kai Zhao, Min Yang, Yongshun Gong, Meng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28718](https://doi.org/10.1609/aaai.v38i8.28718)

**Abstract**:

Recently, learning urban region representations utilizing multi-modal data (information views) has become increasingly popular, for deep understanding of the distributions of various socioeconomic features in cities. However, previous methods usually blend multi-view information in a posteriors stage, falling short in learning coherent and consistent representations across different views. In this paper, we form a new pipeline to learn consistent representations across varying views, and propose the multi-view Contrastive Prediction model for urban Region embedding (ReCP), which leverages the multiple information views from point-of-interest (POI) and human mobility data. Specifically, ReCP comprises two major modules, namely an intra-view learning module utilizing contrastive learning and feature reconstruction to capture the unique information from each single view, and inter-view learning module that perceives the consistency between the two views using a contrastive prediction learning scheme. We conduct thorough experiments on two downstream tasks to assess the proposed model, i.e., land use clustering and region popularity prediction. The experimental results demonstrate that our model outperforms state-of-the-art baseline methods significantly in urban region representation learning.

----

## [970] Hawkes-Enhanced Spatial-Temporal Hypergraph Contrastive Learning Based on Criminal Correlations

**Authors**: *Ke Liang, Sihang Zhou, Meng Liu, Yue Liu, Wenxuan Tu, Yi Zhang, Liming Fang, Zhe Liu, Xinwang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28719](https://doi.org/10.1609/aaai.v38i8.28719)

**Abstract**:

Crime prediction is a crucial yet challenging task within urban computing, which benefits public safety and resource optimization. Over the years, various models have been proposed, and spatial-temporal hypergraph learning models have recently shown outstanding performances. However, three correlations underlying crime are ignored, thus hindering the performance of previous models. Specifically, there are two spatial correlations and one temporal correlation, i.e., (1) co-occurrence of different types of crimes (type spatial correlation), (2) the closer to the crime center, the more dangerous it is around the neighborhood area (neighbor spatial correlation), and (3) the closer between two timestamps, the more relevant events are (hawkes temporal correlation). To this end, we propose Hawkes-enhanced Spatial-Temporal Hypergraph Contrastive Learning framework (HCL), which mines the aforementioned correlations via two specific strategies. Concretely, contrastive learning strategies are designed for two spatial correlations, and hawkes process modeling is adopted for temporal correlations. Extensive experiments demonstrate the promising capacities of HCL from four aspects, i.e., superiority, transferability, effectiveness, and sensitivity.

----

## [971] A Comprehensive Augmentation Framework for Anomaly Detection

**Authors**: *Jiang Lin, Yaping Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28720](https://doi.org/10.1609/aaai.v38i8.28720)

**Abstract**:

Data augmentation methods are commonly integrated into the training of anomaly detection models.
Previous approaches have primarily focused on replicating real-world anomalies or enhancing diversity, without considering that the standard of anomaly varies across different classes, potentially leading to a biased training distribution. This paper analyzes crucial traits of simulated anomalies that contribute to the training of reconstructive networks and condenses them into several methods, thus creating a comprehensive framework by selectively utilizing appropriate combinations. Furthermore, we integrate this framework with a reconstruction-based approach and concurrently propose a split training strategy that alleviates the overfitting issue while avoiding introducing interference to the reconstruction process. The evaluations conducted on the MVTec anomaly detection dataset demonstrate that our method outperforms the previous state-of-the-art approach, particularly in terms of object classes. We also generate a simulated dataset comprising anomalies with diverse characteristics, and experimental results demonstrate that our approach exhibits promising potential for generalizing effectively to various unseen anomalies encountered in real-world scenarios.

----

## [972] Temporally and Distributionally Robust Optimization for Cold-Start Recommendation

**Authors**: *Xinyu Lin, Wenjie Wang, Jujia Zhao, Yongqi Li, Fuli Feng, Tat-Seng Chua*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28721](https://doi.org/10.1609/aaai.v38i8.28721)

**Abstract**:

Collaborative Filtering (CF) recommender models highly depend on user-item interactions to learn CF representations, thus falling short of recommending cold-start items. To address this issue, prior studies mainly introduce item features (e.g., thumbnails) for cold-start item recommendation. They learn a feature extractor on warm-start items to align feature representations with interactions, and then leverage the feature extractor to extract the feature representations of cold-start items for interaction prediction. Unfortunately, the features of cold-start items, especially the popular ones, tend to diverge from those of warm-start ones due to temporal feature shifts, preventing the feature extractor from accurately learning feature representations of cold-start items. 
To alleviate the impact of temporal feature shifts, we consider using Distributionally Robust Optimization (DRO) to enhance the generation ability of the feature extractor. Nonetheless, existing DRO methods face an inconsistency issue: the worse-case warm-start items emphasized during DRO training might not align well with the cold-start item distribution. To capture the temporal feature shifts and combat this inconsistency issue, we propose a novel temporal DRO with new optimization objectives, namely, 1) to integrate a worst-case factor to improve the worst-case performance, and 2) to devise a shifting factor to capture the shifting trend of item features and enhance the optimization of the potentially popular groups in cold-start items. Substantial experiments on three real-world datasets validate the superiority of our temporal DRO in enhancing the generalization ability of cold-start recommender models.

----

## [973] Towards Continual Knowledge Graph Embedding via Incremental Distillation

**Authors**: *Jiajun Liu, Wenjun Ke, Peng Wang, Ziyu Shang, Jinhua Gao, Guozheng Li, Ke Ji, Yanhe Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28722](https://doi.org/10.1609/aaai.v38i8.28722)

**Abstract**:

Traditional knowledge graph embedding (KGE) methods typically require preserving the entire knowledge graph (KG) with significant training costs when new knowledge emerges. To address this issue, the continual knowledge graph embedding (CKGE) task has been proposed to train the KGE model by learning emerging knowledge efficiently while simultaneously preserving decent old knowledge. However, the explicit graph structure in KGs, which is critical for the above goal, has been heavily ignored by existing CKGE methods. On the one hand, existing methods usually learn new triples in a random order, destroying the inner structure of new KGs. On the other hand, old triples are preserved with equal priority, failing to alleviate catastrophic forgetting effectively. In this paper, we propose a competitive method for CKGE based on incremental distillation (IncDE), which considers the full use of the explicit graph structure in KGs. First, to optimize the learning order, we introduce a hierarchical strategy, ranking new triples for layer-by-layer learning. By employing the inter- and intra-hierarchical orders together, new triples are grouped into layers based on the graph structure features. Secondly, to preserve the old knowledge effectively, we devise a novel incremental distillation mechanism, which facilitates the seamless transfer of entity representations from the previous layer to the next one, promoting old knowledge preservation. Finally, we adopt a two-stage training paradigm to avoid the over-corruption of old knowledge influenced by under-trained new knowledge. Experimental results demonstrate the superiority of IncDE over state-of-the-art baselines. Notably, the incremental distillation mechanism contributes to improvements of 0.2%-6.5% in the mean reciprocal rank (MRR) score. More exploratory experiments validate the effectiveness of IncDE in proficiently learning new knowledge while preserving old knowledge across all time steps.

----

## [974] Graph Disentangled Contrastive Learning with Personalized Transfer for Cross-Domain Recommendation

**Authors**: *Jing Liu, Lele Sun, Weizhi Nie, Peiguang Jing, Yuting Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28723](https://doi.org/10.1609/aaai.v38i8.28723)

**Abstract**:

Cross-Domain Recommendation (CDR) has been proven to effectively alleviate the data sparsity problem in Recommender System (RS). Recent CDR methods often disentangle user features into domain-invariant and domain-specific features for efficient cross-domain knowledge transfer. Despite showcasing robust performance, three crucial aspects remain unexplored for existing disentangled CDR approaches: i) The significance nuances of the interaction behaviors are ignored in generating disentangled features; ii) 
The user features are disentangled irrelevant to the individual items to be recommended; iii) The general knowledge transfer overlooks the user's personality when interacting with diverse items. To this end, we propose a Graph Disentangled Contrastive framework for CDR (GDCCDR) with personalized transfer by meta-networks. An adaptive parameter-free filter is proposed to gauge the significance of diverse interactions, thereby facilitating more refined disentangled representations. In sight of the success of Contrastive Learning (CL) in RS, we propose two CL-based constraints for item-aware disentanglement. Proximate CL ensures the coherence of domain-invariant features between domains, while eliminatory CL strives to disentangle features within each domains using mutual information between users and items. Finally, for domain-invariant features, we adopt meta-networks to achieve personalized transfer. Experimental results on four real-world datasets demonstrate the superiority of GDCCDR over state-of-the-art methods.

----

## [975] Multimodal Event Causality Reasoning with Scene Graph Enhanced Interaction Network

**Authors**: *Jintao Liu, Kaiwen Wei, Chenglong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28724](https://doi.org/10.1609/aaai.v38i8.28724)

**Abstract**:

Multimodal event causality reasoning aims to recognize the causal relations based on the given events and accompanying image pairs, requiring the model to have a comprehensive grasp of visual and textual information. However, existing studies fail to effectively model the relations of the objects within the image and capture the object interactions across the image pair, resulting in an insufficient understanding of visual information by the model. To address these issues, we propose a Scene Graph Enhanced Interaction Network (SEIN) in this paper, which can leverage the interactions of the generated scene graph for multimodal event causality reasoning. Specifically, the proposed method adopts a graph convolutional network to model the objects and their relations derived from the scene graph structure, empowering the model to exploit the rich structural and semantic information in the image adequately. To capture the object interactions between the two images, we design an optimal transport-based alignment strategy to match the objects across the images, which could help the model recognize changes in visual information and facilitate causality reasoning. In addition, we introduce a cross-modal fusion module to combine textual and visual features for causality prediction. Experimental results indicate that the proposed SEIN outperforms state-of-the-art methods on the Vis-Causal dataset.

----

## [976] AT4CTR: Auxiliary Match Tasks for Enhancing Click-Through Rate Prediction

**Authors**: *Qi Liu, Xuyang Hou, Defu Lian, Zhe Wang, Haoran Jin, Jia Cheng, Jun Lei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28725](https://doi.org/10.1609/aaai.v38i8.28725)

**Abstract**:

Click-through rate (CTR) prediction is a vital task in industrial recommendation systems. Most existing methods focus on the network architecture design of the CTR model for better accuracy and suffer from the data sparsity problem. Especially in industrial recommendation systems, the widely applied negative sample down-sampling technique due to resource limitation worsens the problem, resulting in a decline in performance. In this paper, we propose Auxiliary Match Tasks for enhancing Click-Through Rate (AT4CTR) prediction accuracy by alleviating the data sparsity problem. Specifically, we design two match tasks inspired by collaborative filtering to enhance the relevance modeling between user and item. As the "click" action is a strong signal which indicates the user's preference towards the item directly, we make the first match task aim at pulling closer the representation between the user and the item regarding the positive samples. Since the user's past click behaviors can also be treated as the user him/herself, we apply the next item prediction as the second match task. For both the match tasks, we choose the InfoNCE as their loss function. The two match tasks can provide meaningful training signals to speed up the model's convergence and alleviate the data sparsity. We conduct extensive experiments on one public dataset and one large-scale industrial recommendation dataset. The result demonstrates the effectiveness of the proposed auxiliary match tasks. AT4CTR has been deployed in the real industrial advertising system and has gained remarkable revenue.

----

## [977] Online Conversion Rate Prediction via Multi-Interval Screening and Synthesizing under Delayed Feedback

**Authors**: *Qiming Liu, Xiang Ao, Yuyao Guo, Qing He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28726](https://doi.org/10.1609/aaai.v38i8.28726)

**Abstract**:

Due to the widespread adoption of the cost-per-action(CPA) display strategy that demands a real-time conversion rate prediction(CVR), delayed feedback is becoming one of the major challenges in online advertising. As the true labels of a significant quantity of samples are only available after long delays, the observed training data are usually biased, harming the performance of models. Recent studies show integrating models with varying waiting windows to observe true labels is beneficial, but the aggregation framework remains far from reaching a consensus. In this work, we propose the Multi-Interval Screening and Synthesizing model (MISS for short) for online CVR prediction. We first design a multi-interval screening model with various output heads to produce accurate and distinctive estimates. Then a light-weight synthesizing model with an assembled training pipeline is applied to thoroughly exploit the knowledge and relationship among heads, obtaining reliable predictions. Extensive experiments on two real-world advertising datasets validate the effectiveness of our model.

----

## [978] KG-TREAT: Pre-training for Treatment Effect Estimation by Synergizing Patient Data with Knowledge Graphs

**Authors**: *Ruoqi Liu, Lingfei Wu, Ping Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28727](https://doi.org/10.1609/aaai.v38i8.28727)

**Abstract**:

Treatment effect estimation (TEE) is the task of determining the impact of various treatments on patient outcomes. Current TEE methods fall short due to reliance on limited labeled data and challenges posed by sparse and high-dimensional observational patient data. To address the challenges, we introduce a novel pre-training and fine-tuning framework, KG-TREAT, which synergizes large-scale observational patient data with biomedical knowledge graphs (KGs) to enhance TEE. Unlike previous approaches, KG-TREAT constructs dual-focus KGs and integrates a deep bi-level attention synergy method for in-depth information fusion, enabling distinct encoding of treatment-covariate and outcome-covariate relationships. KG-TREAT also incorporates two pre-training tasks to ensure a thorough grounding and contextualization of patient data and KGs. Evaluation on four downstream TEE tasks shows KG-TREAT’s superiority over existing methods, with an average improvement of 7% in Area under the ROC Curve (AUC) and 9% in Influence Function-based Precision of Estimating Heterogeneous Effects (IF-PEHE). The effectiveness of our estimated treatment effects is further affirmed by alignment with established randomized clinical trial findings.

----

## [979] Learning Accurate and Bidirectional Transformation via Dynamic Embedding Transportation for Cross-Domain Recommendation

**Authors**: *Weiming Liu, Chaochao Chen, Xinting Liao, Mengling Hu, Yanchao Tan, Fan Wang, Xiaolin Zheng, Yew Soon Ong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28728](https://doi.org/10.1609/aaai.v38i8.28728)

**Abstract**:

With the rapid development of Internet and Web techniques, Cross-Domain Recommendation (CDR) models have been widely explored for resolving the data-sparsity
and cold-start problem. Meanwhile, most CDR models should utilize explicit domain-shareable information (e.g., overlapped users or items) for knowledge transfer across domains. However, this assumption may not be always satisfied since users and items are always non-overlapped in real practice. The performance of many previous works will be severely impaired when these domain-shareable information are not available. To address the aforementioned issues, we propose the Joint Preference Exploration and Dynamic Embedding Transportation model (JPEDET) in this paper which is a novel framework for solving the CDR problem when users and items are non-overlapped. JPEDET includes two main modules, i.e., joint preference exploration module and dynamic embedding transportation module. The joint preference exploration module aims to fuse rating and review information for modelling user preferences. The dynamic embedding transportation module is set to share knowledge via neural ordinary equations for dual transformation across domains. Moreover, we innovatively propose the dynamic transport flow equipped with linear interpolation guidance on barycentric Wasserstein path for achieving accurate and bidirectional transformation. Our empirical study on Amazon datasets demonstrates that JPEDET significantly outperforms the state-of-the-art models under the CDR setting.

----

## [980] Knowledge Graph Error Detection with Contrastive Confidence Adaption

**Authors**: *Xiangyu Liu, Yang Liu, Wei Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28729](https://doi.org/10.1609/aaai.v38i8.28729)

**Abstract**:

Knowledge graphs (KGs) often contain various errors. Previous works on detecting errors in KGs mainly rely on triplet embedding from graph structure. We conduct an empirical study and find that these works struggle to discriminate noise from semantically-similar correct triplets. In this paper, we propose a KG error detection model CCA to integrate both textual and graph structural information from triplet reconstruction for better distinguishing semantics. We design interactive contrastive learning to capture the differences between textual and structural patterns. Furthermore, we construct realistic datasets with semantically-similar noise and adversarial noise. Experimental results demonstrate that CCA outperforms state-of-the-art baselines, especially on semantically-similar noise and adversarial noise.

----

## [981] Perturbation-Invariant Adversarial Training for Neural Ranking Models: Improving the Effectiveness-Robustness Trade-Off

**Authors**: *Yu-An Liu, Ruqing Zhang, Mingkun Zhang, Wei Chen, Maarten de Rijke, Jiafeng Guo, Xueqi Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28730](https://doi.org/10.1609/aaai.v38i8.28730)

**Abstract**:

Neural ranking models (NRMs) have shown great success in information retrieval (IR). But their predictions can easily be manipulated using adversarial examples, which are crafted by adding imperceptible perturbations to legitimate documents. This vulnerability raises significant concerns about their reliability and hinders the widespread deployment of NRMs. By incorporating adversarial examples into training data, adversarial training has become the de facto defense approach to adversarial attacks against NRMs. However, this defense mechanism is subject to a trade-off between effectiveness and adversarial robustness. In this study, we establish theoretical guarantees regarding the effectiveness-robustness trade-off in NRMs. We decompose the robust ranking error into two components, i.e., a natural ranking error for effectiveness evaluation and a boundary ranking error for assessing adversarial robustness. Then, we define the perturbation invariance of a ranking model and prove it to be a differentiable upper bound on the boundary ranking error for attainable computation. Informed by our theoretical analysis, we design a novel perturbation-invariant adversarial training (PIAT) method for ranking models to achieve a better effectiveness-robustness trade-off. We design a regularized surrogate loss, in which one term encourages the effectiveness to be maximized while the regularization term encourages the output to be smooth, so as to improve adversarial robustness. Experimental results on several ranking models demonstrate the superiority of PITA compared to existing adversarial defenses.

----

## [982] Full Bayesian Significance Testing for Neural Networks

**Authors**: *Zehua Liu, Zimeng Li, Jingyuan Wang, Yue He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28731](https://doi.org/10.1609/aaai.v38i8.28731)

**Abstract**:

Significance testing aims to determine whether a proposition about the population distribution is the truth or not given observations. However, traditional significance testing often needs to derive the distribution of the testing statistic, failing to deal with complex nonlinear relationships. In this paper, we propose to conduct Full Bayesian Significance Testing for neural networks, called nFBST, to overcome the limitation in relationship characterization of traditional approaches. A Bayesian neural network is utilized to fit the nonlinear and multi-dimensional relationships with small errors and avoid hard theoretical derivation by computing the evidence value. Besides, nFBST can test not only global significance but also local and instance-wise significance, which previous testing methods don't focus on. Moreover, nFBST is a general framework that can be extended based on the measures selected, such as Grad-nFBST, LRP-nFBST, DeepLIFT-nFBST, LIME-nFBST.  A range of experiments on both simulated and real data are conducted to show the advantages of our method.

----

## [983] KGDM: A Diffusion Model to Capture Multiple Relation Semantics for Knowledge Graph Embedding

**Authors**: *Xiao Long, Liansheng Zhuang, Aodi Li, Jiuchang Wei, Houqiang Li, Shafei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28732](https://doi.org/10.1609/aaai.v38i8.28732)

**Abstract**:

Knowledge graph embedding (KGE) is an efficient and scalable method for knowledge graph completion. However, most existing KGE methods suffer from the challenge of multiple relation semantics, which often degrades their performance. This is because most KGE methods learn fixed continuous vectors for entities (relations) and make deterministic entity predictions to complete the knowledge graph, which hardly captures multiple relation semantics. To tackle this issue, previous works try to learn complex probabilistic embeddings instead of fixed embeddings but suffer from heavy computational complexity. In contrast, this paper proposes a simple yet efficient framework namely the Knowledge Graph Diffusion Model (KGDM) to capture the multiple relation semantics in prediction. Its key idea is to cast the problem of entity prediction into conditional entity generation. Specifically, KGDM estimates the probabilistic distribution of target entities in prediction through Denoising Diffusion Probabilistic Models (DDPM). To bridge the gap between continuous diffusion models and discrete KGs, two learnable embedding functions are defined to map entities and relation to continuous vectors. To consider connectivity patterns of KGs, a Conditional Entity Denoiser model is introduced to generate target entities conditioned on given entities and relations. Extensive experiments demonstrate that KGDM significantly outperforms existing state-of-the-art methods in three benchmark datasets.

----

## [984] Deep Hierarchical Video Compression

**Authors**: *Ming Lu, Zhihao Duan, Fengqing Zhu, Zhan Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28733](https://doi.org/10.1609/aaai.v38i8.28733)

**Abstract**:

Recently, probabilistic predictive coding that directly models the conditional distribution of latent features across successive frames for temporal redundancy removal has yielded promising results.  Existing methods using a single-scale Variational AutoEncoder (VAE) must devise complex networks for conditional probability estimation in latent space, neglecting multiscale characteristics of video frames. Instead, this work proposes hierarchical probabilistic predictive coding, for which hierarchal VAEs are carefully designed to characterize multiscale latent features as a family of flexible priors and posteriors to predict the probabilities of future frames. Under such a hierarchical structure, lightweight networks are sufficient for prediction. The proposed method outperforms representative learned video compression models on common testing videos and demonstrates computational friendliness with much less memory footprint and faster encoding/decoding. Extensive experiments on adaptation to temporal patterns also indicate the better generalization of our hierarchical predictive mechanism. Furthermore, our solution is the first to enable progressive decoding that is favored in networked video applications with packet loss.

----

## [985] Spectral-Based Graph Neural Networks for Complementary Item Recommendation

**Authors**: *Haitong Luo, Xuying Meng, Suhang Wang, Hanyun Cao, Weiyao Zhang, Yequan Wang, Yujun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28734](https://doi.org/10.1609/aaai.v38i8.28734)

**Abstract**:

Modeling complementary relationships greatly helps recommender systems to accurately and promptly recommend the subsequent items when one item is purchased. Unlike traditional similar relationships, items with complementary relationships may be purchased successively (such as iPhone and Airpods Pro), and they not only share relevance but also exhibit dissimilarity. Since the two attributes are opposites, modeling complementary relationships is challenging. Previous attempts to exploit these relationships have either ignored or oversimplified the dissimilarity attribute, resulting in ineffective modeling and an inability to balance the two attributes. Since Graph Neural Networks (GNNs) can capture the relevance and dissimilarity between nodes in the spectral domain, we can leverage spectral-based GNNs to effectively understand and model complementary relationships. 
In this study, we present a novel approach called Spectral-based Complementary Graph Neural Networks (SComGNN) that utilizes the spectral properties of complementary item graphs. We make the first observation that complementary relationships consist of low-frequency and mid-frequency components, corresponding to the relevance and dissimilarity attributes, respectively. Based on this spectral observation, we design spectral graph convolutional networks with low-pass and mid-pass filters to capture the low-frequency and mid-frequency components. Additionally, we propose a two-stage attention mechanism to adaptively integrate and balance the two attributes. Experimental results on four e-commerce datasets demonstrate the effectiveness of our model, with SComGNN significantly outperforming existing baseline models.

----

## [986] Enhancing Cognitive Diagnosis Using Un-interacted Exercises: A Collaboration-Aware Mixed Sampling Approach

**Authors**: *Haiping Ma, Changqian Wang, Hengshu Zhu, Shangshang Yang, Xiaoming Zhang, Xingyi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28735](https://doi.org/10.1609/aaai.v38i8.28735)

**Abstract**:

Cognitive diagnosis is a crucial task in computer-aided education, aimed at evaluating students' proficiency levels across various knowledge concepts through exercises. Current models, however, primarily rely on students' answered exercises, neglecting the complex and rich information contained in un-interacted exercises. While recent research has attempted to leverage the data within un-interacted exercises linked to interacted knowledge concepts, aiming to address the long-tail issue, these studies fail to fully explore the informative, un-interacted exercises related to broader knowledge concepts. This oversight results in diminished performance when these models are applied to comprehensive datasets. In response to this gap, we present the Collaborative-aware Mixed Exercise Sampling (CMES) framework, which can effectively exploit the information present in un-interacted exercises linked to un-interacted knowledge concepts. Specifically, we introduce a novel universal sampling module where the training samples comprise not merely raw data slices, but enhanced samples generated by combining weight-enhanced attention mixture techniques. Given the necessity of real response labels in cognitive diagnosis, we also propose a ranking-based pseudo feedback module to regulate students' responses on generated exercises. The versatility of the CMES framework bolsters existing models and improves their adaptability. Finally, we demonstrate the effectiveness and interpretability of our framework through comprehensive experiments on real-world datasets.

----

## [987] Plug-In Diffusion Model for Sequential Recommendation

**Authors**: *Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin, Zhanhui Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28736](https://doi.org/10.1609/aaai.v38i8.28736)

**Abstract**:

Pioneering efforts have verified the effectiveness of the diffusion models in exploring the informative uncertainty for recommendation. Considering the difference between recommendation and image synthesis tasks, existing methods have undertaken tailored refinements to the diffusion and reverse process. However, these approaches typically use the highest-score item in corpus for user interest prediction, leading to the ignorance of the user's generalized preference contained within other items, thereby remaining constrained by the data sparsity issue. To address this issue, this paper presents a novel Plug-in Diffusion Model for Recommendation (PDRec) framework, which employs the diffusion model as a flexible plugin to jointly take full advantage of the diffusion-generating user preferences on all items. Specifically, PDRec first infers the users' dynamic preferences on all items via a time-interval diffusion model and proposes a Historical Behavior Reweighting (HBR) mechanism to identify the high-quality behaviors and suppress noisy behaviors. In addition to the observed items, PDRec proposes a Diffusion-based Positive Augmentation (DPA) strategy to leverage the top-ranked unobserved items as the potential positive samples, bringing in informative and diverse soft signals to alleviate data sparsity. To alleviate the false negative sampling issue, PDRec employs Noise-free Negative Sampling (NNS) to select stable negative samples for ensuring effective model optimization. Extensive experiments and analyses on four datasets have verified the superiority of the proposed PDRec over the state-of-the-art baselines and showcased the universality of PDRec as a flexible plugin for commonly-used sequential encoders in different recommendation scenarios. The code is available in https://github.com/hulkima/PDRec.

----

## [988] Tail-STEAK: Improve Friend Recommendation for Tail Users via Self-Training Enhanced Knowledge Distillation

**Authors**: *Yijun Ma, Chaozhuo Li, Xiao Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28737](https://doi.org/10.1609/aaai.v38i8.28737)

**Abstract**:

Graph neural networks (GNNs) are commonly employed in collaborative friend recommendation systems. Nevertheless, recent studies reveal a notable performance gap, particularly for users with limited connections, commonly known as tail users, in contrast to their counterparts with abundant connections (head users). Uniformly treating head and tail users poses two challenges for tail user preference learning: (C1) Label Sparsity, as tail users typically possess limited labels; and (C2) Neighborhood Sparsity, where tail users exhibit sparse observable friendships, leading to distinct preference distributions and performance degradation compared to head users. In response to these challenges, we introduce Tail-STEAK, an innovative framework that combines self-training with enhanced knowledge distillation for tail user representation learning. To address(C1), we present Tail-STEAK-base, a two-stage self-training framework. In the first stage, only head users and their accurate connections are utilized for training, while pseudo links are generated for tail users in the second stage. To tackle (C2), we propose two data augmentation-based self-knowledge distillation pretext tasks. These tasks are seamlessly integrated into different stages of Tail-STEAK-base, culminating in the comprehensive Tail-STEAK framework. Extensive experiments, conducted on state-of-the-art GNN-based friend recommendation models, substantiate the efficacy of Tail-STEAK in significantly improving tail user performance. Our code and data are publicly available at https://github.com/antman9914/Tail-STEAK.

----

## [989] Graph Contrastive Invariant Learning from the Causal Perspective

**Authors**: *Yanhu Mo, Xiao Wang, Shaohua Fan, Chuan Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28738](https://doi.org/10.1609/aaai.v38i8.28738)

**Abstract**:

Graph contrastive learning (GCL), learning the node representation by contrasting two augmented graphs in a self-supervised way, has attracted considerable attention. GCL is usually believed to learn the invariant representation. However, does this understanding always hold in practice? In this paper, we first study GCL from the perspective of causality. By analyzing GCL with the structural causal model (SCM), we discover that traditional GCL may not well learn the invariant representations due to the non-causal information contained in the graph. How can we fix it and encourage the current GCL to learn better invariant representations? The SCM offers two requirements and motives us to propose a novel GCL method. Particularly, we introduce the spectral graph augmentation to simulate the intervention upon non-causal factors. Then we design the invariance objective and independence objective to better capture the causal factors. Specifically, (i) the invariance objective encourages the encoder to capture the invariant information contained in causal variables, and (ii) the independence objective aims to reduce the influence of confounders on the causal variables. Experimental results demonstrate the effectiveness of our approach on node classification tasks.

----

## [990] HGE: Embedding Temporal Knowledge Graphs in a Product Space of Heterogeneous Geometric Subspaces

**Authors**: *Jiaxin Pan, Mojtaba Nayyeri, Yinan Li, Steffen Staab*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28739](https://doi.org/10.1609/aaai.v38i8.28739)

**Abstract**:

Temporal knowledge graphs represent temporal facts (s,p,o,?) relating a subject s and an object o via a relation label p at time ?, where ? could be a time point or time interval. Temporal knowledge graphs may exhibit static temporal patterns at distinct points in time and dynamic temporal patterns between different timestamps. In order to learn a rich set of static and dynamic temporal patterns and apply them for inference, several embedding approaches have been suggested in the literature. However, as most of them resort to single underlying embedding spaces, their capability to model all kinds of temporal patterns was severely limited by having to adhere to the geometric property of their one embedding space. We lift this limitation by an embedding approach that maps temporal facts into a product space of several heterogeneous geometric subspaces with distinct geometric properties, i.e.\  Complex,  Dual, and  Split-complex spaces. In addition, we propose a temporal-geometric attention mechanism to integrate information from different geometric subspaces conveniently according to the captured relational and temporal information. Experimental results on standard temporal benchmark datasets favorably evaluate our approach against state-of-the-art models.

----

## [991] Cross-Domain Contrastive Learning for Time Series Clustering

**Authors**: *Furong Peng, Jiachen Luo, Xuan Lu, Sheng Wang, Feijiang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28740](https://doi.org/10.1609/aaai.v38i8.28740)

**Abstract**:

Most deep learning-based time series clustering models concentrate on data representation in a separate process from clustering. This leads to that clustering loss cannot guide feature extraction. Moreover, most methods solely analyze data from the temporal domain, disregarding the potential within the frequency domain.

To address these challenges, we introduce a novel end-to-end Cross-Domain Contrastive learning model for time series Clustering (CDCC). Firstly, it integrates the clustering process and feature extraction using contrastive constraints at both cluster-level and instance-level. Secondly, the data is encoded simultaneously in both temporal and frequency domains, leveraging contrastive learning to enhance within-domain representation. Thirdly, cross-domain constraints are proposed to align the latent representations and category distribution across domains. With the above strategies, CDCC not only achieves end-to-end output but also effectively integrates frequency domains. Extensive experiments and visualization analysis are conducted on 40 time series datasets from UCR, demonstrating the superior performance of the proposed model.

----

## [992] Refining Latent Homophilic Structures over Heterophilic Graphs for Robust Graph Convolution Networks

**Authors**: *Chenyang Qiu, Guoshun Nan, Tianyu Xiong, Wendi Deng, Di Wang, Zhiyang Teng, Lijuan Sun, Qimei Cui, Xiaofeng Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28741](https://doi.org/10.1609/aaai.v38i8.28741)

**Abstract**:

Graph convolution networks (GCNs) are extensively utilized in various graph tasks to mine knowledge from spatial data. Our study marks the pioneering attempt to quantitatively investigate the GCN robustness over omnipresent heterophilic graphs for node classification. We uncover that the predominant vulnerability is caused by the structural out-of-distribution (OOD) issue. This finding motivates us to present a novel method that aims to harden GCNs by automatically learning Latent Homophilic Structures over heterophilic graphs. We term such a methodology as LHS. To elaborate, our initial step involves learning a latent structure by employing a novel self-expressive technique based on multi-node interactions. Subsequently, the structure is refined using a pairwisely constrained dual-view contrastive learning approach. We iteratively perform the above procedure, enabling a GCN model to aggregate information in a homophilic way on heterophilic graphs. Armed with such an adaptable structure, we can properly mitigate the structural OOD threats over heterophilic graphs. Experiments on various benchmarks show the effectiveness of the proposed LHS approach for robust GCNs.

----

## [993] Link Prediction in Multilayer Networks via Cross-Network Embedding

**Authors**: *Guojing Ren, Xiao Ding, Xiao-Ke Xu, Hai-Feng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28742](https://doi.org/10.1609/aaai.v38i8.28742)

**Abstract**:

Link prediction is a fundamental task in network analysis, with the objective of predicting missing or potential links. While existing studies have mainly concentrated on single networks, it is worth noting that numerous real-world networks exhibit interconnectedness. For example, individuals often register on various social media platforms to access diverse services, such as chatting, tweeting, blogging, and rating movies. These platforms share a subset of users and are termed multilayer networks. The interlayer links in such networks hold valuable information that provides more comprehensive insights into the network structure. To effectively exploit this complementary information and enhance link prediction in the target network, we propose a novel cross-network embedding method. This method aims to represent different networks in a shared latent space, preserving proximity within single networks as well as consistency across multilayer networks. Specifically, nodes can aggregate messages from aligned nodes in other layers. Extensive experiments conducted on real-world datasets demonstrate the superior performance of our proposed method for link prediction in multilayer networks.

----

## [994] Towards Diverse Perspective Learning with Selection over Multiple Temporal Poolings

**Authors**: *Jihyeon Seong, Jungmin Kim, Jaesik Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28743](https://doi.org/10.1609/aaai.v38i8.28743)

**Abstract**:

In Time Series Classification (TSC), temporal pooling methods that consider sequential information have been proposed. However, we found that each temporal pooling has a distinct mechanism, and can perform better or worse depending on time series data. We term this fixed pooling mechanism a single perspective of temporal poolings. In this paper, we propose a novel temporal pooling method with diverse perspective learning: Selection over Multiple Temporal Poolings (SoM-TP). SoM-TP dynamically selects the optimal temporal pooling among multiple methods for each data by attention. The dynamic pooling selection is motivated by the ensemble concept of Multiple Choice Learning (MCL), which selects the best among multiple outputs. The pooling selection by SoM-TP's attention enables a non-iterative pooling ensemble within a single classifier. Additionally, we define a perspective loss and Diverse Perspective Learning Network (DPLN). The loss works as a regularizer to reflect all the pooling perspectives from DPLN. Our perspective analysis using Layer-wise Relevance Propagation (LRP) reveals the limitation of a single perspective and ultimately demonstrates diverse perspective learning of SoM-TP. We also show that SoM-TP outperforms CNN models based on other temporal poolings and state-of-the-art models in TSC with extensive UCR/UEA repositories.

----

## [995] LAFA: Multimodal Knowledge Graph Completion with Link Aware Fusion and Aggregation

**Authors**: *Bin Shang, Yinliang Zhao, Jun Liu, Di Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28744](https://doi.org/10.1609/aaai.v38i8.28744)

**Abstract**:

Recently, an enormous amount of research has emerged on multimodal knowledge graph completion (MKGC), which seeks to extract knowledge from multimodal data and predict the most plausible missing facts to complete a given multimodal knowledge graph (MKG). However, existing MKGC approaches largely ignore that visual information may introduce noise and lead to uncertainty when adding them to the traditional KG embeddings due to the contribution of each associated image to entity is different in diverse link scenarios. Moreover, treating each triple independently when learning entity embeddings leads to local structural and the whole graph information missing. To address these challenges, we propose a novel link aware fusion and aggregation based multimodal knowledge graph completion model named LAFA, which is composed of link aware fusion module and link aware aggregation module. The link aware fusion module alleviates noise of irrelevant visual information by calculating the importance between an entity and its associated images in different link scenarios, and fuses the visual and structural embeddings according to the importance through our proposed modality embedding fusion mechanism. The link aware aggregation module assigns neighbor structural information to a given central entity by calculating the importance between the entity and its neighbors, and aggregating the fused embeddings through linear combination according to the importance. Extensive experiments on standard datasets validate that LAFA can obtain state-of-the-art performance.

----

## [996] Mixed Geometry Message and Trainable Convolutional Attention Network for Knowledge Graph Completion

**Authors**: *Bin Shang, Yinliang Zhao, Jun Liu, Di Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28745](https://doi.org/10.1609/aaai.v38i8.28745)

**Abstract**:

Knowledge graph completion (KGC) aims to study the embedding representation to solve the incompleteness of knowledge graphs (KGs). Recently, graph convolutional networks (GCNs) and graph attention networks (GATs) have been widely used in KGC tasks by capturing neighbor information of entities. However, Both GCNs and GATs based KGC models have their limitations, and the best method is to analyze the neighbors of each entity (pre-validating), while this process is prohibitively expensive. Furthermore, the representation quality of the embeddings can affect the aggregation of neighbor information (message passing). To address the above limitations, we propose a novel knowledge graph completion model with mixed geometry message and trainable convolutional attention network named MGTCA. Concretely, the mixed geometry message function generates rich neighbor message by integrating spatially information in the hyperbolic space, hypersphere space and Euclidean space jointly. To complete the autonomous switching of graph neural networks (GNNs) and eliminate the necessity of pre-validating the local structure of KGs, a trainable convolutional attention network is proposed by comprising three types of GNNs in one trainable formulation. Furthermore, a mixed geometry scoring function is proposed, which calculates scores of triples by novel prediction function and similarity function based on different geometric spaces. Extensive experiments on three standard datasets confirm the effectiveness of our innovations, and the performance of MGTCA is significantly improved compared to the state-of-the-art approaches.

----

## [997] ResDiff: Combining CNN and Diffusion Model for Image Super-resolution

**Authors**: *Shuyao Shang, Zhengyang Shan, Guangxing Liu, Lunqian Wang, Xinghua Wang, Zekai Zhang, Jinglin Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28746](https://doi.org/10.1609/aaai.v38i8.28746)

**Abstract**:

Adapting the Diffusion Probabilistic Model (DPM) for direct image super-resolution is wasteful, given that a simple Convolutional Neural Network (CNN) can recover the main low-frequency content. Therefore, we present ResDiff, a novel Diffusion Probabilistic Model based on Residual structure for Single Image Super-Resolution (SISR). ResDiff utilizes a combination of a CNN, which restores primary low-frequency components, and a DPM, which predicts the residual between the ground-truth image and the CNN predicted image. In contrast to the common diffusion-based methods that directly use LR space to guide the noise towards HR space, ResDiff utilizes the CNN’s initial prediction to direct the noise towards the residual space between HR space and CNN-predicted space, which not only accelerates the generation process but also acquires superior sample quality. Additionally, a frequency-domain-based loss function for CNN is introduced to facilitate its restoration, and a frequency-domain guided diffusion is designed for DPM on behalf of predicting high-frequency details. The extensive experiments on multiple benchmark datasets demonstrate that ResDiff outperforms previous diffusion based methods in terms of shorter model convergence time, superior generation quality, and more diverse samples.

----

## [998] An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention

**Authors**: *Yehjin Shin, Jeongwhan Choi, Hyowon Wi, Noseong Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28747](https://doi.org/10.1609/aaai.v38i8.28747)

**Abstract**:

Sequential recommendation (SR) models based on Transformers have achieved remarkable successes. The self-attention mechanism of Transformers for computer vision and natural language processing suffers from the oversmoothing problem, i.e., hidden representations becoming similar to tokens. In the SR domain, we, for the first time, show that the same problem occurs. We present pioneering investigations that reveal the low-pass filtering nature of self-attention in the SR, which causes oversmoothing. To this end, we propose a novel method called Beyond Self-Attention for Sequential Recommendation (BSARec), which leverages the Fourier transform to i) inject an inductive bias by considering fine-grained sequential patterns and ii) integrate low and high-frequency information to mitigate oversmoothing. Our discovery shows significant advancements in the SR domain and is expected to bridge the gap for existing Transformer-based SR models. We test our proposed approach through extensive experiments on 6 benchmark datasets. The experimental results demonstrate that our model outperforms 7 baseline methods in terms of recommendation performance. Our code is available at https://github.com/yehjin-shin/BSARec.

----

## [999] A Diffusion-Based Pre-training Framework for Crystal Property Prediction

**Authors**: *Zixing Song, Ziqiao Meng, Irwin King*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i8.28748](https://doi.org/10.1609/aaai.v38i8.28748)

**Abstract**:

Many significant problems involving crystal property prediction from 3D structures have limited labeled data due to expensive and time-consuming physical simulations or lab experiments. To overcome this challenge, we propose a pretrain-finetune framework for the crystal property prediction task named CrysDiff based on diffusion models. In the pre-training phase, CrysDiff learns the latent marginal distribution of crystal structures via the reconstruction task. Subsequently, CrysDiff can be fine-tuned under the guidance of the new sparse labeled data, fitting the conditional distribution of the target property given the crystal structures. To better model the crystal geometry, CrysDiff notably captures the full symmetric properties of the crystals, including the invariance of reflection, rotation, and periodic translation. Extensive experiments demonstrate that CrysDiff can significantly improve the performance of the downstream crystal property prediction task on multiple target properties, outperforming all the SOTA pre-training models for crystals with good margins on the popular JARVIS-DFT dataset.

----



[Go to the previous page](AAAI-2024-list04.md)

[Go to the next page](AAAI-2024-list06.md)

[Go to the catalog section](README.md)