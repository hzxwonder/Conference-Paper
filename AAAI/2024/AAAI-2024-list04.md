## [600] Neural Physical Simulation with Multi-Resolution Hash Grid Encoding

**Authors**: *Haoxiang Wang, Tao Yu, Tianwei Yang, Hui Qiao, Qionghai Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28349](https://doi.org/10.1609/aaai.v38i6.28349)

**Abstract**:

We explore the generalization of the implicit representation in the physical simulation task. Traditional time-dependent partial differential equations (PDEs) solvers for physical simulation often adopt the grid or mesh for spatial discretization, which is memory-consuming for high resolution and lack of adaptivity. Many implicit representations like local extreme machine or Siren are proposed but they are still too compact to suffer from limited accuracy in handling local details and a long time of convergence. We contribute a neural simulation framework based on multi-resolution hash grid representation to introduce hierarchical consideration of global and local information, simultaneously. Furthermore, we propose two key strategies:  1) a numerical gradient method for computing high-order derivatives with boundary conditions;  2) a range analysis sample method for fast neural geometry boundary sampling with dynamic topologies. Our method shows much higher accuracy and strong flexibility for various simulation problems: e.g., large elastic deformations, complex fluid dynamics, and multi-scale phenomena which remain challenging for existing neural physical solvers.

----

## [601] Deep Unfolded Network with Intrinsic Supervision for Pan-Sharpening

**Authors**: *Hebaixu Wang, Meiqi Gong, Xiaoguang Mei, Hao Zhang, Jiayi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28350](https://doi.org/10.1609/aaai.v38i6.28350)

**Abstract**:

Existing deep pan-sharpening methods lack the learning of complementary information between PAN and MS modalities in the intermediate layers, and exhibit low interpretability due to their black-box designs. To this end, an interpretable deep unfolded network with intrinsic supervision for pan-sharpening is proposed. Building upon the observation degradation process, it formulates the pan-sharpening task as a variational model minimization with spatial consistency prior and spectral projection prior. The former prior requires a joint component decomposition of PAN and MS images to extract intrinsic features. By being supervised in the intermediate layers, it can selectively provide high-frequency information for spatial enhancement. The latter prior constrains the intensity correlation between MS and PAN images derived from physical observations, so as to improve spectral fidelity. To further enhance the transparency of network design, we develop an iterative solution algorithm following the half-quadratic splitting to unfold the deep model. It rigorously adheres to the variational model, significantly enhancing the interpretability behind network design and efficiently alternating the optimization of the network. Extensive experiments demonstrate the advantages of our method compared to state-of-the-arts, showcasing its remarkable generalization capability to real-world scenes. Our code is publicly available at https://github.com/Baixuzx7/DISPNet.

----

## [602] Continuous Piecewise-Affine Based Motion Model for Image Animation

**Authors**: *Hexiang Wang, Fengqi Liu, Qianyu Zhou, Ran Yi, Xin Tan, Lizhuang Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28351](https://doi.org/10.1609/aaai.v38i6.28351)

**Abstract**:

Image animation aims to bring static images to life according to driving videos and create engaging visual content that can be used for various purposes such as animation, entertainment, and education. Recent unsupervised methods utilize affine and thin-plate spline transformations based on keypoints to transfer the motion in driving frames to the source image. However, limited by the expressive power of the transformations used, these methods always produce poor results when the gap between the motion in the driving frame and the source image is large. To address this issue, we propose to model motion from the source image to the driving frame in highly-expressive diffeomorphism spaces. Firstly, we introduce Continuous Piecewise-Affine based (CPAB) transformation to model the motion and present a well-designed inference algorithm to generate CPAB transformation from control keypoints. Secondly, we propose a SAM-guided keypoint semantic loss to further constrain the keypoint extraction process and improve the semantic consistency between the corresponding keypoints on the source and driving images. Finally, we design a structure alignment loss to align the structure-related features extracted from driving and generated images, thus helping the generator generate results that are more consistent with the driving action. Extensive experiments on four datasets demonstrate the effectiveness of our method against state-of-the-art competitors quantitatively and qualitatively. Code will be publicly available at: https://github.com/DevilPG/AAAI2024-CPABMM.

----

## [603] Temporal Adaptive RGBT Tracking with Modality Prompt

**Authors**: *Hongyu Wang, Xiaotao Liu, Yifan Li, Meng Sun, Dian Yuan, Jing Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28352](https://doi.org/10.1609/aaai.v38i6.28352)

**Abstract**:

RGBT tracking has been widely used in various fields such as robotics, surveillance processing, and autonomous driving. Existing RGBT trackers fully explore the spatial information between the template and the search region and locate the target based on the appearance matching results. However, these RGBT trackers have very limited exploitation of temporal information, either ignoring temporal information or exploiting it through online sampling and training. The former struggles to cope with the object state changes, while the latter neglects the correlation between spatial and temporal information. To alleviate these limitations, we propose a novel Temporal Adaptive RGBT Tracking framework, named as TATrack. TATrack has a spatio-temporal two-stream structure and captures temporal information by an online updated template, where the two-stream structure refers to the multi-modal feature extraction and cross-modal interaction for the initial template and the online update template respectively. TATrack contributes to comprehensively exploit spatio-temporal information and multi-modal information for target localization. In addition, we design a spatio-temporal interaction (STI) mechanism that bridges two branches and enables cross-modal interaction to span longer time scales. Extensive experiments on three popular RGBT tracking benchmarks show that our method achieves state-of-the-art performance, while running at real-time speed.

----

## [604] SAUI: Scale-Aware Unseen Imagineer for Zero-Shot Object Detection

**Authors**: *Jiahao Wang, Caixia Yan, Weizhan Zhang, Huan Liu, Hao Sun, Qinghua Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28353](https://doi.org/10.1609/aaai.v38i6.28353)

**Abstract**:

Zero-shot object detection (ZSD) aims to localize and classify unseen objects without access to their training annotations. As a prevailing solution to ZSD, generation-based methods synthesize unseen visual features by taking seen features as reference and class semantic embeddings as guideline. Although previous works continuously improve the synthesis quality, they fail to consider the scale-varying nature of unseen objects. The generation process is preformed over a single scale of object features and thus lacks scale-diversity among synthesized features. In this paper, we reveal the scale-varying challenge in ZSD and propose a Scale-Aware Unseen Imagineer (SAUI) to lead the way of a novel scale-aware ZSD paradigm. To obtain multi-scale features of seen-class objects, we design a specialized coarse-to-fine extractor to capture features through multiple scale-views. To generate unseen features scale by scale, we innovate a Series-GAN synthesizer along with three scale-aware contrastive components to imagine separable, diverse and robust scale-wise unseen features. Extensive experiments on PASCAL VOC, COCO and DIOR datasets demonstrate SAUI's better performance in different scenarios, especially for scale-varying and small objects. Notably, SAUI achieves the new state-of-the art performance on COCO and DIOR.

----

## [605] Omnidirectional Image Super-resolution via Bi-projection Fusion

**Authors**: *Jiangang Wang, Yuning Cui, Yawen Li, Wenqi Ren, Xiaochun Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28354](https://doi.org/10.1609/aaai.v38i6.28354)

**Abstract**:

With the rapid development of virtual reality, omnidirectional images (ODIs) have attracted much attention from both the industrial community and academia. However, due to storage and transmission limitations, the resolution of current ODIs is often insufficient to provide an immersive virtual reality experience. Previous approaches address this issue using conventional 2D super-resolution techniques on equirectangular projection without exploiting the unique geometric properties of ODIs. In particular, the equirectangular projection (ERP) provides a complete field-of-view but introduces significant distortion, while the cubemap projection (CMP) can reduce distortion yet has a limited field-of-view. In this paper, we present a novel Bi-Projection Omnidirectional Image Super-Resolution (BPOSR) network to take advantage of the geometric properties of the above two projections. Then, we design two tailored attention methods for these projections: Horizontal Striped Transformer Block (HSTB) for ERP and Perspective Shift Transformer Block (PSTB) for CMP. Furthermore, we propose a fusion module to make these projections complement each other. Extensive experiments demonstrate that BPOSR achieves state-of-the-art performance on omnidirectional image super-resolution. The code is available at https://github.com/W-JG/BPOSR.

----

## [606] Adaptive FSS: A Novel Few-Shot Segmentation Framework via Prototype Enhancement

**Authors**: *Jing Wang, Jiangyun Li, Chen Chen, Yisi Zhang, Haoran Shen, Tianxiang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28355](https://doi.org/10.1609/aaai.v38i6.28355)

**Abstract**:

The Few-Shot Segmentation (FSS) aims to accomplish the novel class segmentation task with a few annotated images. Current FSS research based on meta-learning focuses on designing a complex interaction mechanism between the query and support feature. However, unlike humans who can rapidly learn new things from limited samples, the existing approach relies solely on fixed feature matching to tackle new tasks, lacking adaptability. In this paper, we propose a novel framework based on the adapter mechanism, namely Adaptive FSS, which can efficiently adapt the existing FSS model to the novel classes. In detail, we design the Prototype Adaptive Module (PAM), which utilizes accurate category information provided by the support set to derive class prototypes, enhancing class-specific information in the multi-stage representation. In addition, our approach is compatible with diverse FSS methods with different backbones by simply inserting PAM between the layers of the encoder. Experiments demonstrate that our method effectively improves the performance of the FSS models (e.g., MSANet, HDMNet, FPTrans, and DCAMA) and achieves new state-of-the-art (SOTA) results (i.e., 72.4% and 79.1% mIoU on PASCAL-5i 1-shot and 5-shot settings, 52.7% and 60.0% mIoU on COCO-20i 1-shot and 5-shot settings). Our code is available at https://github.com/jingw193/AdaptiveFSS.

----

## [607] PointAttN: You Only Need Attention for Point Cloud Completion

**Authors**: *Jun Wang, Ying Cui, Dongyan Guo, Junxia Li, Qingshan Liu, Chunhua Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28356](https://doi.org/10.1609/aaai.v38i6.28356)

**Abstract**:

Point cloud completion referring to completing 3D shapes from partial 3D point clouds is a fundamental problem for 3D point cloud analysis tasks. Benefiting from the development of deep neural networks, researches on point cloud completion have made great progress in recent years. However, the explicit local region partition like kNNs involved in existing methods makes them sensitive to the density distribution of point clouds. Moreover, it serves limited receptive fields that prevent capturing features from long-range context information. To solve the problems, we leverage the cross-attention and self-attention mechanisms to design novel neural network for point cloud completion with implicit local region partition. Two basic units Geometric Details Perception (GDP) and Self-Feature Augment (SFA) are proposed to establish the structural relationships directly among points in a simple yet effective way via attention mechanism. Then based on GDP and SFA, we construct a new framework with popular encoder-decoder architecture for point cloud completion. The proposed framework, namely PointAttN, is simple, neat and effective, which can precisely capture the structural information of 3D shapes and predict complete point clouds with detailed geometry. Experimental results demonstrate that our PointAttN outperforms state-of-the-art methods on multiple challenging benchmarks. Code is available at: https://github.com/ohhhyeahhh/PointAttN

----

## [608] EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering

**Authors**: *Junjue Wang, Zhuo Zheng, Zihang Chen, Ailong Ma, Yanfei Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28357](https://doi.org/10.1609/aaai.v38i6.28357)

**Abstract**:

Earth vision research typically focuses on extracting geospatial object locations and categories but neglects the exploration of relations between objects and comprehensive reasoning. Based on city planning needs, we develop a multi-modal multi-task VQA dataset (EarthVQA) to advance relational reasoning-based judging, counting, and comprehensive analysis. The EarthVQA dataset contains 6000 images, corresponding semantic masks, and 208,593 QA pairs with urban and rural governance requirements embedded. As objects are the basis for complex relational reasoning, we propose a Semantic OBject Awareness framework (SOBA) to advance VQA in an object-centric way. To preserve refined spatial locations and semantics, SOBA leverages a segmentation network for object semantics generation. The object-guided attention aggregates object interior features via pseudo masks, and bidirectional cross-attention further models object external relations hierarchically. To optimize object counting, we propose a numerical difference loss that dynamically adds difference penalties, unifying the classification and regression tasks. Experimental results show that SOBA outperforms both advanced general and remote sensing methods. We believe this dataset and framework provide a strong benchmark for Earth vision's complex analysis. The project page is at https://Junjue-Wang.github.io/homepage/EarthVQA.

----

## [609] Semi-supervised Class-Agnostic Motion Prediction with Pseudo Label Regeneration and BEVMix

**Authors**: *Kewei Wang, Yizheng Wu, Zhiyu Pan, Xingyi Li, Ke Xian, Zhe Wang, Zhiguo Cao, Guosheng Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28358](https://doi.org/10.1609/aaai.v38i6.28358)

**Abstract**:

Class-agnostic motion prediction methods aim to comprehend motion within open-world scenarios, holding significance for autonomous driving systems. However, training a high-performance model in a fully-supervised manner always requires substantial amounts of manually annotated data, which can be both expensive and time-consuming to obtain. To address this challenge, our study explores the potential of semi-supervised learning (SSL) for class-agnostic motion prediction. Our SSL framework adopts a consistency-based self-training paradigm, enabling the model to learn from unlabeled data by generating pseudo labels through test-time inference. To improve the quality of pseudo labels, we propose a novel motion selection and re-generation module. This module effectively selects reliable pseudo labels and re-generates unreliable ones. Furthermore, we propose two data augmentation strategies: temporal sampling and BEVMix. These strategies facilitate consistency regularization in SSL. Experiments conducted on nuScenes demonstrate that our SSL method can surpass the self-supervised approach by a large margin by utilizing only a tiny fraction of labeled data. Furthermore, our method exhibits comparable performance to  weakly and some fully supervised methods. These results highlight the ability of our method to strike a favorable balance between annotation costs and performance. Code will be available at https://github.com/kwwcv/SSMP.

----

## [610] Multi-Domain Incremental Learning for Face Presentation Attack Detection

**Authors**: *Keyao Wang, Guosheng Zhang, Haixiao Yue, Ajian Liu, Gang Zhang, Haocheng Feng, Junyu Han, Errui Ding, Jingdong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28359](https://doi.org/10.1609/aaai.v38i6.28359)

**Abstract**:

Previous face Presentation Attack Detection (PAD) methods aim to improve the effectiveness of cross-domain tasks. However, in real-world scenarios, the original training data of the pre-trained model is not available due to data privacy or other reasons. Under these constraints, general methods for fine-tuning single-target domain data may lose previously learned knowledge, leading to a catastrophic forgetting problem. To address these issues, we propose a multi-domain incremental learning (MDIL) method for PAD, which not only learns knowledge well from the new domain but also maintains the performance of previous domains stably. Specifically, we propose an adaptive domain-specific experts (ADE) framework based on the vision transformer to preserve the discriminability of previous domains. Furthermore,  an asymmetric classifier is designed to keep the output distribution of different classifiers consistent, thereby improving the generalization ability. Extensive experiments show that our proposed method achieves state-of-the-art performance compared to prior methods of incremental learning. Excitingly, under more stringent setting conditions, our method approximates or even outperforms the DA/DG-based methods.

----

## [611] AltNeRF: Learning Robust Neural Radiance Field via Alternating Depth-Pose Optimization

**Authors**: *Kun Wang, Zhiqiang Yan, Huang Tian, Zhenyu Zhang, Xiang Li, Jun Li, Jian Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28360](https://doi.org/10.1609/aaai.v38i6.28360)

**Abstract**:

Neural Radiance Fields (NeRF) have shown promise in generating realistic novel views from sparse scene images. However, existing NeRF approaches often encounter challenges due to the lack of explicit 3D supervision and imprecise camera poses, resulting in suboptimal outcomes. To tackle these issues, we propose AltNeRF---a novel framework designed to create resilient NeRF representations using self-supervised monocular depth estimation (SMDE) from monocular videos, without relying on known camera poses. SMDE in AltNeRF masterfully learns depth and pose priors to regulate NeRF training. The depth prior enriches NeRF's capacity for precise scene geometry depiction, while the pose prior provides a robust starting point for subsequent pose refinement. Moreover, we introduce an alternating algorithm that harmoniously melds NeRF outputs into SMDE through a consistence-driven mechanism, thus enhancing the integrity of depth priors. This alternation empowers AltNeRF to progressively refine NeRF representations, yielding the synthesis of realistic novel views. Extensive experiments showcase the compelling capabilities of AltNeRF in generating high-fidelity and robust novel views that closely resemble reality.

----

## [612] A Multimodal, Multi-Task Adapting Framework for Video Action Recognition

**Authors**: *Mengmeng Wang, Jiazheng Xing, Boyuan Jiang, Jun Chen, Jianbiao Mei, Xingxing Zuo, Guang Dai, Jingdong Wang, Yong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28361](https://doi.org/10.1609/aaai.v38i6.28361)

**Abstract**:

Recently, the rise of large-scale vision-language pretrained models like CLIP, coupled with the technology of Parameter-Efficient FineTuning (PEFT), has captured substantial attraction in video action recognition. Nevertheless, prevailing approaches tend to prioritize strong supervised performance at the expense of compromising the models' generalization capabilities during transfer. In this paper, we introduce a novel Multimodal, Multi-task CLIP adapting framework named M2-CLIP to address these challenges, preserving both high supervised performance and robust transferability.
Firstly, to enhance the individual modality architectures, we introduce multimodal adapters to both the visual and text branches. Specifically, we design a novel visual TED-Adapter, that performs global Temporal Enhancement and local temporal Difference modeling to improve the temporal representation capabilities of the visual encoder. Moreover, we adopt text encoder adapters to strengthen the learning of semantic label information.
Secondly, we design a multi-task decoder with a rich set of supervisory signals, including the original contrastive learning head, a cross-modal classification head, a cross-modal masked language modeling head, and a visual classification head. This multi-task decoder adeptly satisfies the need for strong supervised performance within a multimodal framework.
Experimental results validate the efficacy of our approach, demonstrating exceptional performance in supervised learning while maintaining strong generalization in zero-shot scenarios.

----

## [613] msLPCC: A Multimodal-Driven Scalable Framework for Deep LiDAR Point Cloud Compression

**Authors**: *Miaohui Wang, Runnan Huang, Hengjin Dong, Di Lin, Yun Song, Wuyuan Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28362](https://doi.org/10.1609/aaai.v38i6.28362)

**Abstract**:

LiDAR sensors are widely used in autonomous driving, and the growing storage and transmission demands have made LiDAR point cloud compression (LPCC) a hot research topic. To address the challenges posed by the large-scale and uneven-distribution (spatial and categorical) of LiDAR point data, this paper presents a new multimodal-driven scalable LPCC framework. For the large-scale challenge, we decouple the original LiDAR data into multi-layer point subsets, compress and transmit each layer separately, so as to ensure the reconstruction quality requirement under different scenarios. For the uneven-distribution challenge, we extract, align, and fuse heterologous feature representations, including point modality with position information, depth modality with spatial distance information, and segmentation modality with category information. Extensive experimental results on the benchmark SemanticKITTI database validate that our method outperforms 14 recent representative LPCC methods.

----

## [614] Cycle-Consistency Learning for Captioning and Grounding

**Authors**: *Ning Wang, Jiajun Deng, Mingbo Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28363](https://doi.org/10.1609/aaai.v38i6.28363)

**Abstract**:

We present that visual grounding and image captioning, which perform as two mutually inverse processes, can be bridged together for collaborative training by careful designs. By consolidating this idea, we introduce CyCo, a cyclic-consistent learning framework to ameliorate the independent training pipelines of visual grounding and image captioning. The proposed framework (1) allows the semi-weakly supervised training of visual grounding; (2) improves the performance of fully supervised visual grounding; (3) yields a general captioning model that can describe arbitrary image regions. Extensive experiments show that our fully supervised grounding model achieves state-of-the-art performance, and the semi-weakly supervised one also exhibits competitive performance compared to the fully supervised counterparts. Our image captioning model has the capability to freely describe image regions and meanwhile shows impressive performance on prevalent captioning benchmarks.

----

## [615] Compositional Text-to-Image Synthesis with Attention Map Control of Diffusion Models

**Authors**: *Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, Xiaodong Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28364](https://doi.org/10.1609/aaai.v38i6.28364)

**Abstract**:

Recent text-to-image (T2I) diffusion models show outstanding performance in generating high-quality images conditioned on textual prompts. However, they fail to semantically align the generated images with the prompts due to their limited compositional capabilities, leading to attribute leakage, entity leakage, and missing entities. In this paper, we propose a novel attention mask control strategy based on predicted object boxes to address these issues. In particular, we first train a BoxNet to predict a box for each entity that possesses the attribute specified in the prompt. Then, depending on the predicted boxes, a unique mask control is applied to the cross- and self-attention maps. Our approach produces a more semantically accurate synthesis by constraining the attention regions of each token in the prompt to the image. In addition, the proposed method is straightforward and effective and can be readily integrated into existing cross-attention-based T2I generators. We compare our approach to competing methods and demonstrate that it can faithfully convey the semantics of the original text to the generated content and achieve high availability as a ready-to-use plugin. Please refer to  https://github.com/OPPO-Mente-Lab/attention-mask-control.

----

## [616] AGS: Affordable and Generalizable Substitute Training for Transferable Adversarial Attack

**Authors**: *Ruikui Wang, Yuanfang Guo, Yunhong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28365](https://doi.org/10.1609/aaai.v38i6.28365)

**Abstract**:

In practical black-box attack scenarios, most of the existing transfer-based attacks employ pretrained models (e.g. ResNet50) as the substitute models. Unfortunately, these substitute models are not always appropriate for transfer-based attacks. Firstly, these models are usually trained on a largescale annotated dataset, which is extremely expensive and time-consuming to construct. Secondly, the primary goal of these models is to perform a specific task, such as image classification, which is not developed for adversarial attacks. To tackle the above issues, i.e., high cost and over-fitting on taskspecific models, we propose an Affordable and Generalizable Substitute (AGS) training framework tailored for transferbased adversarial attack. Specifically, we train the substitute model from scratch by our proposed adversary-centric constrastive learning. This proposed learning mechanism introduces another sample with slight adversarial perturbations as an additional positive view of the input image, and then encourages the adversarial view and two benign views to interact comprehensively with each other. To further boost the generalizability of the substitute model, we propose adversarial invariant learning to maintain the representations of the adversarial example invariants under augmentations with various strengths. Our AGS model can be trained solely with unlabeled and out-of domain data and avoid overfitting to any task-specific models, because of its inherently self-supervised nature. Extensive experiments demonstrate that our AGS achieves comparable or superior performance compared to substitute models pretrained on the complete ImageNet training set, when executing attacks across a diverse range of target models, including ViTs, robustly trained models, object detection and segmentation models. Our source codes are available at https://github.com/lwmming/AGS.

----

## [617] DocNLC: A Document Image Enhancement Framework with Normalized and Latent Contrastive Representation for Multiple Degradations

**Authors**: *Ruilu Wang, Yang Xue, Lianwen Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28366](https://doi.org/10.1609/aaai.v38i6.28366)

**Abstract**:

Document Image Enhancement (DIE) remains challenging due to the prevalence of multiple degradations in document images captured by cameras. In this paper, we respond an interesting question: can the performance of pre-trained models and downstream DIE models be improved if they are bootstrapped using different degradation types of the same semantic samples and their high-dimensional features with ambiguous inter-class distance? To this end, we propose an effective contrastive learning paradigm for DIE — a Document image enhancement framework with Normalization and Latent Contrast (DocNLC). While existing DIE methods focus on eliminating one type of degradation, DocNLC considers the relationship between different types of degradation while utilizing both direct and latent contrasts to constrain content consistency, thus achieving a unified treatment of multiple types of degradation. Specifically, we devise a latent contrastive learning module to enforce explicit decorrelation of the normalized representations of different degradation types and to minimize the redundancy between them. Comprehensive experiments show that our method outperforms state-of-the-art DIE models in both pre-training and fine-tuning stages
on four publicly available independent datasets. In addition, we discuss the potential benefits of DocNLC for downstream tasks. Our code is released at https://github.com/RylonW/DocNLC

----

## [618] Towards Evidential and Class Separable Open Set Object Detection

**Authors**: *Ruofan Wang, Rui-Wei Zhao, Xiaobo Zhang, Rui Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28367](https://doi.org/10.1609/aaai.v38i6.28367)

**Abstract**:

Detecting in open-world scenarios poses a formidable challenge for models intended for real-world deployment. The advanced closed set object detectors achieve impressive performance under the closed set setting, but often produce overconfident misprediction on unknown objects due to the lack of supervision. In this paper, we propose a novel Evidential Object Detector (EOD) to formulate the Open Set Object Detection (OSOD) problem from the perspective of Evidential Deep Learning (EDL) theory, which quantifies classification uncertainty by placing the Dirichlet Prior over the categorical distribution parameters. The task-specific customized evidential framework, equipped with meticulously designed model architecture and loss function, effectively bridges the gap between EDL theory and detection tasks. Moreover, we utilize contrastive learning as an implicit means of evidential regularization and to encourage the class separation in the latent space. Alongside, we innovatively model the background uncertainty to further improve the unknown discovery ability. Extensive experiments on benchmark datasets demonstrate the outperformance of the proposed method over existing ones.

----

## [619] Suppressing Uncertainty in Gaze Estimation

**Authors**: *Shijing Wang, Yaping Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28368](https://doi.org/10.1609/aaai.v38i6.28368)

**Abstract**:

Uncertainty in gaze estimation manifests in two aspects: 1) low-quality images caused by occlusion, blurriness, inconsistent eye movements, or even non-face images; 2) uncorrected labels resulting from the misalignment between the labeled and actual gaze points during the annotation process. Allowing these uncertainties to participate in training hinders the improvement of gaze estimation.  To tackle these challenges, in this paper, we propose an effective solution, named Suppressing Uncertainty in Gaze Estimation (SUGE), which introduces a novel triplet-label consistency measurement to estimate and reduce the uncertainties. Specifically, for each training sample, we propose to estimate a novel ``neighboring label'' calculated by a linearly weighted projection from the neighbors to capture the similarity relationship between image features and their corresponding labels, which can be incorporated with the predicted pseudo label and ground-truth label for uncertainty estimation. By modeling such triplet-label consistency, we can largely reduce the negative effects of unqualified images and wrong labels through our designed sample weighting and label correction strategies. Experimental results on the gaze estimation benchmarks indicate that our proposed SUGE achieves state-of-the-art performance.

----

## [620] What Effects the Generalization in Visual Reinforcement Learning: Policy Consistency with Truncated Return Prediction

**Authors**: *Shuo Wang, Zhihao Wu, Xiaobo Hu, Jinwen Wang, Youfang Lin, Kai Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28369](https://doi.org/10.1609/aaai.v38i6.28369)

**Abstract**:

In visual Reinforcement Learning (RL), the challenge of generalization to new environments is paramount. This study pioneers a theoretical analysis of visual RL generalization, establishing an upper bound on the generalization objective, encompassing policy divergence and Bellman error components. Motivated by this analysis, we propose maintaining the cross-domain consistency for each policy in the policy space, which can reduce the divergence of the learned policy during the test. In practice, we introduce the Truncated Return Prediction (TRP) task, promoting cross-domain policy consistency by predicting truncated returns of historical trajectories. Moreover, we also propose a Transformer-based predictor for this auxiliary task. Extensive experiments on DeepMind Control Suite and Robotic Manipulation tasks demonstrate that TRP achieves state-of-the-art generalization performance. We further demonstrate that TRP outperforms previous methods in terms of sample efficiency during training.

----

## [621] DeepAccident: A Motion and Accident Prediction Benchmark for V2X Autonomous Driving

**Authors**: *Tianqi Wang, Sukmin Kim, Wenxuan Ji, Enze Xie, Chongjian Ge, Junsong Chen, Zhenguo Li, Ping Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28370](https://doi.org/10.1609/aaai.v38i6.28370)

**Abstract**:

Safety is the primary priority of autonomous driving. Nevertheless, no published dataset currently supports the direct and explainable safety evaluation for autonomous driving. In this work, we propose DeepAccident, a large-scale dataset generated via a realistic simulator containing diverse accident scenarios that frequently occur in real-world driving. The proposed DeepAccident dataset includes 57K annotated frames and 285K annotated samples, approximately 7 times more than the large-scale nuScenes dataset with 40k annotated samples. In addition, we propose a new task, end-to-end motion and accident prediction, which can be used to directly evaluate the accident prediction ability for different autonomous driving algorithms. Furthermore, for each scenario, we set four vehicles along with one infrastructure to record data, thus providing diverse viewpoints for accident scenarios and enabling V2X (vehicle-to-everything) research on perception and prediction tasks. Finally, we present a baseline V2X model named V2XFormer that demonstrates superior performance for motion and accident prediction and 3D object detection compared to the single-vehicle model.

----

## [622] Semantic-Guided Novel Category Discovery

**Authors**: *Weishuai Wang, Ting Lei, Qingchao Chen, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28371](https://doi.org/10.1609/aaai.v38i6.28371)

**Abstract**:

The Novel Category Discovery problem aims to cluster an unlabeled set with the help of a labeled set consisting of disjoint but related classes. However, existing models treat class names as discrete one-hot labels and ignore the semantic understanding of these classes. In this paper, we propose a new setting named Semantic-guided Novel Category Discovery (SNCD), which requires the model to not only cluster the unlabeled images but also semantically recognize these images based on a set of their class names. The first challenge we confront pertains to effectively leveraging the class names of unlabeled images, given the inherent gap between the visual and linguistic domains. To address this issue, we incorporate a semantic-aware recognition mechanism. This is achieved by constructing dynamic class-wise visual prototypes as well as a semantic similarity matrix that enables the projection of visual features into the semantic space. The second challenge originates from the granularity disparity between the classification and clustering tasks. To deal with this, we develop a semantic-aware clustering process to facilitate the exchange of knowledge between the two tasks. Through extensive experiments, we demonstrate the mutual benefits of the recognition and clustering tasks, which can be jointly optimized. Experimental results on multiple datasets confirm the effectiveness of our proposed method. Our code is available at https://github.com/wang-weishuai/Semantic-guided-NCD.

----

## [623] HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors

**Authors**: *Xiao Wang, Zongzhen Wu, Bo Jiang, Zhimin Bao, Lin Zhu, Guoqi Li, Yaowei Wang, Yonghong Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28372](https://doi.org/10.1609/aaai.v38i6.28372)

**Abstract**:

The main streams of human activity recognition (HAR) algorithms are developed based on RGB cameras which usually suffer from illumination, fast motion, privacy preservation, and large energy consumption. Meanwhile, the biologically inspired event cameras attracted great interest due to their unique features, such as high dynamic range, dense temporal but sparse spatial resolution, low latency, low power, etc. As it is a newly arising sensor, even there is no realistic large-scale dataset for HAR. Considering its great practical value, in this paper, we propose a large-scale benchmark dataset to bridge this gap, termed HARDVS, which contains 300 categories and more than 100K event sequences. We evaluate and report the performance of multiple popular HAR algorithms, which provide extensive baselines for future works to compare. More importantly, we propose a novel spatial-temporal feature learning and fusion framework, termed ESTF, for event stream based human activity recognition. It first projects the event streams into spatial and temporal embeddings using StemNet, then, encodes and fuses the dual-view representations using Transformer networks. Finally, the dual features are concatenated and fed into a classification head for activity prediction. Extensive experiments on multiple datasets fully validated the effectiveness of our model. Both the dataset and source code will be released at https://github.com/Event-AHU/HARDVS.

----

## [624] Structural Information Guided Multimodal Pre-training for Vehicle-Centric Perception

**Authors**: *Xiao Wang, Wentao Wu, Chenglong Li, Zhicheng Zhao, Zhe Chen, Yukai Shi, Jin Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28373](https://doi.org/10.1609/aaai.v38i6.28373)

**Abstract**:

Understanding vehicles in images is important for various applications such as intelligent transportation and self-driving system. Existing vehicle-centric works typically pre-train models on large-scale classification datasets and then fine-tune them for specific downstream tasks. However, they neglect the specific characteristics of vehicle perception in different tasks and might thus lead to sub-optimal performance. To address this issue, we propose a novel vehicle-centric pre-training framework called VehicleMAE, which incorporates the structural information including the spatial structure from vehicle profile information and the semantic structure from informative high-level natural language descriptions for effective masked vehicle appearance reconstruction. To be specific, we explicitly extract the sketch lines of vehicles as a form of the spatial structure to guide vehicle reconstruction. The more comprehensive knowledge distilled from the CLIP big model based on the similarity between the paired/unpaired vehicle image-text sample is further taken into consideration to help achieve a better understanding of vehicles. A large-scale dataset is built to pre-train our model, termed Autobot1M, which contains about 1M vehicle images and 12693 text information. Extensive experiments on four vehicle-based downstream tasks fully validated the effectiveness of our VehicleMAE. The source code and pre-trained models will be released at https://github.com/Event-AHU/VehicleMAE.

----

## [625] ICAR: Image-Based Complementary Auto Reasoning

**Authors**: *Xijun Wang, Anqi Liang, Junbang Liang, Ming C. Lin, Yu Lou, Shan Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28374](https://doi.org/10.1609/aaai.v38i6.28374)

**Abstract**:

Scene-aware Complementary Item Retrieval (CIR) is a challenging task which requires to generate a set of compatible items across domains. Due to the subjectivity, it is difficult to set up a rigorous standard for both data collection and learning objectives. To address this challenging task, we propose a visual compatibility concept, composed of similarity (resembling in color, geometry, texture, and etc.) and  complementarity (different items like table vs chair completing a group). Based on this notion, we propose a compatibility learning framework, a category-aware Flexible Bidirectional Transformer (FBT), for visual ``scene-based set compatibility reasoning'' with the cross-domain visual similarity input and auto-regressive complementary item generation. We introduce a ``Flexible Bidirectional Transformer (FBT),'' consisting of an encoder with flexible masking, a category prediction arm, and an auto-regressive visual embedding prediction arm. And the inputs for FBT are cross-domain visual similarity invariant embeddings, making this framework quite generalizable. Furthermore, our proposed FBT model learns the inter-object compatibility from a large set of scene images in a self-supervised way. Compared with the SOTA methods, this approach achieves up to 5.3% and 9.6% in FITB score and 22.3% and 31.8% SFID improvement on fashion and furniture, respectively.

----

## [626] GCNext: Towards the Unity of Graph Convolutions for Human Motion Prediction

**Authors**: *Xinshun Wang, Qiongjie Cui, Chen Chen, Mengyuan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28375](https://doi.org/10.1609/aaai.v38i6.28375)

**Abstract**:

The past few years has witnessed the dominance of Graph Convolutional Networks (GCNs) over human motion prediction. Various styles of graph convolutions have been proposed, with each one meticulously designed and incorporated into a carefully-crafted network architecture. This paper breaks the limits of existing knowledge by proposing Universal Graph Convolution (UniGC), a novel graph convolution concept that re-conceptualizes different graph convolutions as its special cases. Leveraging UniGC on network-level, we propose GCNext, a novel GCN-building paradigm that dynamically determines the best-fitting graph convolutions both sample-wise and layer-wise. GCNext offers multiple use cases, including training a new GCN from scratch or refining a preexisting GCN. Experiments on Human3.6M, AMASS, and 3DPW datasets show that, by incorporating unique module-to-network designs, GCNext yields up to 9x lower computational cost than existing GCN methods, on top of achieving state-of-the-art performance. Our code is available at https://github.com/BradleyWang0416/GCNext.

----

## [627] CL2CM: Improving Cross-Lingual Cross-Modal Retrieval via Cross-Lingual Knowledge Transfer

**Authors**: *Yabing Wang, Fan Wang, Jianfeng Dong, Hao Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28376](https://doi.org/10.1609/aaai.v38i6.28376)

**Abstract**:

Cross-lingual cross-modal retrieval has garnered increasing attention recently, which aims to achieve the alignment between vision and target language (V-T) without using any annotated V-T data pairs. Current methods employ machine translation (MT) to construct pseudo-parallel data pairs,  which are then used to learn a multi-lingual and multi-modal embedding space that aligns visual and target-language representations. However, the large heterogeneous gap between vision and text, along with the noise present in target language translations, poses significant challenges in effectively aligning their representations. To address these challenges, we propose a general framework, Cross-Lingual to Cross-Modal (CL2CM), which improves the alignment between vision and target language using cross-lingual transfer. This approach allows us to fully leverage the merits of multi-lingual pre-trained models (e.g., mBERT) and the benefits of the same modality structure, i.e., smaller gap, to provide reliable and comprehensive semantic correspondence (knowledge) for the cross-modal network. We evaluate our proposed approach on two multilingual image-text datasets, Multi30K and MSCOCO, and one video-text dataset, VATEX. The results clearly demonstrate the effectiveness of our proposed method and its high potential for large-scale retrieval.

----

## [628] OSFFNet: Omni-Stage Feature Fusion Network for Lightweight Image Super-Resolution

**Authors**: *Yang Wang, Tao Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28377](https://doi.org/10.1609/aaai.v38i6.28377)

**Abstract**:

Recently, several lightweight methods have been proposed to implement single-image super-resolution (SISR) on resource-constrained devices. However, these methods primarily focus on simplifying network structures without the full utilization of shallow features. The fact remains that shallow features encompass crucial details for the super-resolution task, including edges, textures, and colors. Therefore, developing a novel architecture that can effectively integrate features from different levels and capitalize on their mutual complementarity is necessary. We first analyze the relationship between multi-stage features and the restoration tasks in a classic lightweight SR method. Based on these observations, we propose an Omni-Stage Feature Fusion (OSFF) architecture, which incorporates Original Image Stacked Initialisation, Shallow Feature Global Connection, and Multi-Receptive Field Dynamic Fusion. An Attention-Enhanced Feature Distillation module is also designed to enhance the model performance. Finally, leveraging these contributions, we construct an Omni-Stage Feature Fusion Network (OSFFNet). Through extensive experiments on various benchmark datasets, the proposed model outperforms state-of-the-art methods. Notably, it achieves a 0.26dB PSNR improvement over the second-best method for x2 SR on the Urban100 dataset.

----

## [629] Prompting Segmentation with Sound Is Generalizable Audio-Visual Source Localizer

**Authors**: *Yaoting Wang, Weisong Liu, Guangyao Li, Jian Ding, Di Hu, Xi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28378](https://doi.org/10.1609/aaai.v38i6.28378)

**Abstract**:

Never having seen an object and heard its sound simultaneously, can the model still accurately localize its visual position from the input audio? In this work, we concentrate on the Audio-Visual Localization and Segmentation tasks but under the demanding zero-shot and few-shot scenarios. To achieve this goal, different from existing approaches that mostly employ the encoder-fusion-decoder paradigm to decode localization information from the fused audio-visual feature, we introduce the encoder-prompt-decoder paradigm, aiming to better fit the data scarcity and varying data distribution dilemmas with the help of abundant knowledge from pre-trained models. Specifically, we first propose to construct a Semantic-aware Audio Prompt (SAP) to help the visual foundation model focus on sounding objects, meanwhile, the semantic gap between the visual and audio modalities is also encouraged to shrink. Then, we develop a Correlation Adapter (ColA) to keep minimal training efforts as well as maintain adequate knowledge of the visual foundation model. By equipping with these means, extensive experiments demonstrate that this new paradigm outperforms other fusion-based methods in both the unseen class and cross-dataset settings. We hope that our work can further promote the generalization study of Audio-Visual Localization and Segmentation in practical application scenarios. Project page: https://github.com/GeWu-Lab/Generalizable-Audio-Visual-Segmentation

----

## [630] Mask-Homo: Pseudo Plane Mask-Guided Unsupervised Multi-Homography Estimation

**Authors**: *Yasi Wang, Hong Liu, Chao Zhang, Lu Xu, Qiang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28379](https://doi.org/10.1609/aaai.v38i6.28379)

**Abstract**:

Homography estimation is a fundamental problem in computer vision. Previous works mainly focus on estimating either a single homography, or multiple homographies based on mesh grid division of the image. In practical scenarios, single homography is inadequate and often leads to a compromised result for multiple planes; while mesh grid multi-homography damages the plane distribution of the scene, and does not fully address the restriction to use homography. 

In this work, we propose a novel semantics guided multi-homography estimation framework, Mask-Homo, to provide an explicit solution to the multi-plane depth disparity problem. First, a pseudo plane mask generation module is designed to obtain multiple correlated regions that follow the plane distribution of the scene. Then, multiple local homography transformations, each of which aligns a correlated region precisely, are predicted and corresponding warped images are fused to obtain the final result. Furthermore, a new metric, Mask-PSNR, is proposed for more comprehensive evaluation of alignment. Extensive experiments are conducted to verify the effectiveness of the proposed method. Our code is available at https://github.com/SAITPublic/MaskHomo.

----

## [631] PointPatchMix: Point Cloud Mixing with Patch Scoring

**Authors**: *Yi Wang, Jiaze Wang, Jinpeng Li, Zixu Zhao, Guangyong Chen, Anfeng Liu, Pheng-Ann Heng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28380](https://doi.org/10.1609/aaai.v38i6.28380)

**Abstract**:

Data augmentation is an effective regularization strategy for mitigating overfitting in deep neural networks, and it plays a crucial role in 3D vision tasks, where the point cloud data is relatively limited. While mixing-based augmentation has shown promise for point clouds, previous methods mix point clouds either on block level or point level, which has constrained their ability to strike a balance between generating diverse training samples and preserving the local characteristics of point clouds. The significance of each part component of the point clouds has not been fully considered, as not all parts contribute equally to the classification task, and some parts may contain unimportant or redundant information. To overcome these challenges, we propose PointPatchMix, a novel approach that mixes point clouds at the patch level and integrates a patch scoring module to generate content-based targets for mixed point clouds. Our approach preserves local features at the patch level, while the patch scoring module assigns targets based on the content-based significance score from a pre-trained teacher model. We evaluate PointPatchMix on two benchmark datasets including ModelNet40 and ScanObjectNN, and demonstrate significant improvements over various baselines in both synthetic and real-world datasets, as well as few-shot settings. With Point-MAE as our baseline, our model surpasses previous methods by a significant margin. Furthermore, our approach shows strong generalization across various point cloud methods and enhances the robustness of the baseline model. Code is available at https://jiazewang.com/projects/pointpatchmix.html.

----

## [632] Data Distribution Distilled Generative Model for Generalized Zero-Shot Recognition

**Authors**: *Yijie Wang, Mingjian Hong, Luwen Huangfu, Sheng Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28381](https://doi.org/10.1609/aaai.v38i6.28381)

**Abstract**:

In the realm of Zero-Shot Learning (ZSL), we address biases in Generalized Zero-Shot Learning (GZSL) models, which favor seen data. To counter this, we introduce an end-to-end generative GZSL framework called D3GZSL. This framework respects seen and synthesized unseen data as in-distribution and out-of-distribution data, respectively, for a more balanced model. D3GZSL comprises two core modules: in-distribution dual space distillation (ID2SD) and out-of-distribution batch distillation (O2DBD). ID2SD aligns teacher-student outcomes in embedding and label spaces, enhancing learning coherence. O2DBD introduces low-dimensional out-of-distribution representations per batch sample, capturing shared structures between seen and un seen categories. Our approach demonstrates its effectiveness across established GZSL benchmarks, seamlessly integrating into mainstream generative frameworks. Extensive experiments consistently showcase that D3GZSL elevates the performance of existing generative GZSL methods, under scoring its potential to refine zero-shot learning practices. The code is available at: https://github.com/PJBQ/D3GZSL.git

----

## [633] SiMA-Hand: Boosting 3D Hand-Mesh Reconstruction by Single-to-Multi-View Adaptation

**Authors**: *Yinqiao Wang, Hao Xu, Pheng-Ann Heng, Chi-Wing Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28382](https://doi.org/10.1609/aaai.v38i6.28382)

**Abstract**:

Estimating 3D hand mesh from RGB images is a longstanding track, in which occlusion is one of the most challenging problems. Existing attempts towards this task often fail when the occlusion dominates the image space. In this paper, we propose SiMA-Hand, aiming to boost the mesh reconstruction performance by Single-to-Multi-view Adaptation. First, we design a multi-view hand reconstructor to fuse information across multiple views by holistically adopting feature fusion at image, joint, and vertex levels. Then, we introduce a single-view hand reconstructor equipped with SiMA. Though taking only one view as input at inference, the shape and orientation features in the single-view reconstructor can be enriched by learning non-occluded knowledge from the extra views at training, enhancing the reconstruction precision on the occluded regions. We conduct experiments on the Dex-YCB and HanCo benchmarks with challenging object- and self-caused occlusion cases, manifesting that SiMA-Hand consistently achieves superior performance over the state of the arts. Code will be released on https://github.com/JoyboyWang/SiMA-Hand Pytorch.

----

## [634] SQLdepth: Generalizable Self-Supervised Fine-Structured Monocular Depth Estimation

**Authors**: *Youhong Wang, Yunji Liang, Hao Xu, Shaohui Jiao, Hongkai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28383](https://doi.org/10.1609/aaai.v38i6.28383)

**Abstract**:

Recently, self-supervised monocular depth estimation has gained popularity with numerous applications in autonomous driving and robotics. However, existing solutions primarily seek to estimate depth from immediate visual features, and struggle to recover fine-grained scene details. In this paper, we introduce SQLdepth, a novel approach that can effectively learn fine-grained scene structure priors from ego-motion. In SQLdepth, we propose a novel Self Query Layer (SQL) to build a self-cost volume and infer depth from it, rather than inferring depth from feature maps. We show that, the self-cost volume is an effective inductive bias for geometry learning, which implicitly models the single-frame scene geometry, with each slice of it indicating a relative distance map between points and objects in a latent space. Experimental results on KITTI and Cityscapes show that our method attains remarkable state-of-the-art performance, and showcases computational efficiency, reduced training complexity, and the ability to recover fine-grained scene details. Moreover, the self-matching-oriented relative distance querying in SQL improves the robustness and zero-shot generalization capability of SQLdepth. Code is available at https://github.com/hisfog/SfMNeXt-Impl.

----

## [635] H2GFormer: Horizontal-to-Global Voxel Transformer for 3D Semantic Scene Completion

**Authors**: *Yu Wang, Chao Tong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28384](https://doi.org/10.1609/aaai.v38i6.28384)

**Abstract**:

3D Semantic Scene Completion (SSC) has emerged as a novel task in vision-based holistic 3D scene understanding. Its objective is to densely predict the occupancy and category of each voxel in a 3D scene based on input from either LiDAR or images. Currently, many transformer-based semantic scene completion frameworks employ simple yet popular Cross-Attention and Self-Attention mechanisms to integrate and infer dense geometric and semantic information of voxels. However, they overlook the distinctions among voxels in the scene, especially in outdoor scenarios where the horizontal direction contains more variations. And voxels located at object boundaries and within the interior of objects exhibit varying levels of positional significance. To address this issue, we propose a transformer-based SSC framework called H2GFormer that incorporates a horizontal-to-global approach. This framework takes into full consideration the variations of voxels in the horizontal direction and the characteristics of voxels on object boundaries. We introduce a horizontal window-to-global attention (W2G) module that effectively fuses semantic information by first diffusing it horizontally from reliably visible voxels and then propagating the semantic understanding to global voxels, ensuring a more reliable fusion of semantic-aware features. Moreover, an Internal-External Position Awareness Loss (IoE-PALoss) is utilized during network training to emphasize the critical positions within the transition regions between objects. The experiments conducted on the SemanticKITTI dataset demonstrate that H2GFormer exhibits superior performance in both geometric and semantic completion tasks. Our code is available on https://github.com/Ryanwy1/H2GFormer.

----

## [636] Exploring Diverse Representations for Open Set Recognition

**Authors**: *Yu Wang, Junxian Mu, Pengfei Zhu, Qinghua Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28385](https://doi.org/10.1609/aaai.v38i6.28385)

**Abstract**:

Open set recognition (OSR) requires the model to classify samples that belong to closed sets while rejecting unknown samples during test. Currently, generative models often perform better than discriminative models in OSR, but recent studies show that generative models may be computationally infeasible or unstable on complex tasks. In this paper, we provide insights into OSR and find that learning supplementary representations can theoretically reduce the open space risk. Based on the analysis, we propose a new model, namely Multi-Expert Diverse Attention Fusion (MEDAF), that learns diverse representations in a discriminative way. MEDAF consists of multiple experts that are learned with an attention diversity regularization term to ensure the attention maps are mutually different. The logits learned by each expert are adaptively fused and used to identify the unknowns through the score function. We show that the differences in attention maps can lead to diverse representations so that the fused representations can well handle the open space. Extensive experiments are conducted on standard and OSR large-scale benchmarks. Results show that the proposed discriminative method can outperform existing generative models by up to 9.5% on AUROC and achieve new state-of-the-art performance with little computational cost. Our method can also seamlessly integrate existing classification models. Code is available at https://github.com/Vanixxz/MEDAF.

----

## [637] SMILEtrack: SiMIlarity LEarning for Occlusion-Aware Multiple Object Tracking

**Authors**: *Yu-Hsiang Wang, Jun-Wei Hsieh, Ping-Yang Chen, Ming-Ching Chang, Hung-Hin So, Xin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28386](https://doi.org/10.1609/aaai.v38i6.28386)

**Abstract**:

Despite recent progress in Multiple Object Tracking (MOT), several obstacles such as occlusions, similar objects, and complex scenes remain an open challenge. Meanwhile, a systematic study of the cost-performance tradeoff for the popular tracking-by-detection paradigm is still lacking. This paper introduces SMILEtrack, an innovative object tracker that effectively addresses these challenges by integrating an efficient object detector with a Siamese network-based Similarity Learning Module (SLM). The technical contributions of SMILETrack are twofold.  First, we propose an SLM that calculates the appearance similarity between two objects, overcoming the limitations of feature descriptors in Separate Detection and Embedding (SDE) models. The SLM incorporates a Patch Self-Attention (PSA) block inspired by the vision Transformer, which generates reliable features for accurate similarity matching. Second, we develop a Similarity Matching Cascade (SMC) module with a novel GATE function for robust object matching across consecutive video frames, further enhancing MOT performance. Together, these innovations help SMILETrack achieve an improved trade-off between the cost (e.g., running speed) and performance (e.g., tracking accuracy) over several existing state-of-the-art benchmarks, including the popular BYTETrack method. SMILETrack outperforms BYTETrack by 0.4-0.8 MOTA and 2.1-2.2 HOTA points on MOT17 and MOT20 datasets. Code is available at http://github.com/pingyang1117/SMILEtrack_official.

----

## [638] Learning Hierarchical Prompt with Structured Linguistic Knowledge for Vision-Language Models

**Authors**: *Yubin Wang, Xinyang Jiang, De Cheng, Dongsheng Li, Cairong Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28387](https://doi.org/10.1609/aaai.v38i6.28387)

**Abstract**:

Prompt learning has become a prevalent strategy for adapting vision-language foundation models to downstream tasks. As large language models (LLMs) have emerged, recent studies have explored the use of category-related descriptions as input to enhance prompt effectiveness. Nevertheless, conventional descriptions fall short of structured information that effectively represents the interconnections among entities or attributes linked to a particular category. To address this limitation and prioritize harnessing structured knowledge, this paper advocates for leveraging LLMs to build a graph for each description to model the entities and attributes describing the category, as well as their correlations. Preexisting prompt tuning methods exhibit inadequacies in managing this structured knowledge. Consequently, we propose a novel approach called Hierarchical Prompt Tuning (HPT), which enables simultaneous modeling of both structured and conventional linguistic knowledge. Specifically, we introduce a relationship-guided attention module to capture pair-wise associations among entities and attributes for low-level prompt learning. In addition, by incorporating high-level and global-level prompts modeling overall semantics, the proposed hierarchical structure forges cross-level interlinks and empowers the model to handle more complex and long-term relationships. Extensive experiments demonstrate that our HPT shows strong effectiveness and generalizes much better than existing SOTA methods. Our code is available at https://github.com/Vill-Lab/2024-AAAI-HPT.

----

## [639] TOP-ReID: Multi-Spectral Object Re-identification with Token Permutation

**Authors**: *Yuhao Wang, Xuehu Liu, Pingping Zhang, Hu Lu, Zhengzheng Tu, Huchuan Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28388](https://doi.org/10.1609/aaai.v38i6.28388)

**Abstract**:

Multi-spectral object Re-identification (ReID) aims to retrieve specific objects by leveraging complementary information from different image spectra. It delivers great advantages over traditional single-spectral ReID in complex visual environment. However, the significant distribution gap among different image spectra poses great challenges for effective multi-spectral feature representations. In addition, most of current Transformer-based ReID methods only utilize the global feature of class tokens to achieve the holistic retrieval, ignoring the local discriminative ones. To address the above issues, we step further to utilize all the tokens of Transformers and propose a cyclic token permutation framework for multi-spectral object ReID, dubbled TOP-ReID. More specifically, we first deploy a multi-stream deep network based on vision Transformers to preserve distinct information from different image spectra. Then, we propose a Token Permutation Module (TPM) for cyclic multi-spectral feature aggregation. It not only facilitates the spatial feature alignment across different image spectra, but also allows the class token of each spectrum to perceive the local details of other spectra. Meanwhile, we propose a Complementary Reconstruction Module (CRM), which introduces dense token-level reconstruction constraints to reduce the distribution gap across different image spectra. With the above modules, our proposed framework can generate more discriminative multi-spectral features for robust object ReID. Extensive experiments on three ReID benchmarks (i.e., RGBNT201, RGBNT100 and MSVR310) verify the effectiveness of our methods. The code is available at https://github.com/924973292/TOP-ReID.

----

## [640] GMMFormer: Gaussian-Mixture-Model Based Transformer for Efficient Partially Relevant Video Retrieval

**Authors**: *Yuting Wang, Jinpeng Wang, Bin Chen, Ziyun Zeng, Shu-Tao Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28389](https://doi.org/10.1609/aaai.v38i6.28389)

**Abstract**:

Given a text query, partially relevant video retrieval (PRVR) seeks to find untrimmed videos containing pertinent moments in a database. For PRVR, clip modeling is essential to capture the partial relationship between texts and videos. Current PRVR methods adopt scanning-based clip construction to achieve explicit clip modeling, which is information-redundant and requires a large storage overhead. To solve the efficiency problem of PRVR methods, this paper proposes GMMFormer, a Gaussian-Mixture-Model based Transformer which models clip representations implicitly. During frame interactions, we incorporate Gaussian-Mixture-Model constraints to focus each frame on its adjacent frames instead of the whole video. Then generated representations will contain multi-scale clip information, achieving implicit clip modeling. In addition, PRVR methods ignore semantic differences between text queries relevant to the same video, leading to a sparse embedding space. We propose a query diverse loss to distinguish these text queries, making the embedding space more intensive and contain more semantic information. Extensive experiments on three large-scale video datasets (i.e., TVR, ActivityNet Captions, and Charades-STA) demonstrate the superiority and efficiency of GMMFormer.

----

## [641] Out of Thin Air: Exploring Data-Free Adversarial Robustness Distillation

**Authors**: *Yuzheng Wang, Zhaoyu Chen, Dingkang Yang, Pinxue Guo, Kaixun Jiang, Wenqiang Zhang, Lizhe Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28390](https://doi.org/10.1609/aaai.v38i6.28390)

**Abstract**:

Adversarial Robustness Distillation (ARD) is a promising task to solve the issue of limited adversarial robustness of small capacity models while optimizing the expensive computational costs of Adversarial Training (AT). Despite the good robust performance, the existing ARD methods are still impractical to deploy in natural high-security scenes due to these methods rely entirely on original or publicly available data with a similar distribution. In fact, these data are almost always private, specific, and distinctive for scenes that require high robustness. To tackle these issues, we propose a challenging but significant task called Data-Free Adversarial Robustness Distillation (DFARD), which aims to train small, easily deployable, robust models without relying on data. We demonstrate that the challenge lies in the lower upper bound of knowledge transfer information, making it crucial to mining and transferring knowledge more efficiently. Inspired by human education, we design a plug-and-play Interactive Temperature Adjustment (ITA) strategy to improve the efficiency of knowledge transfer and propose an Adaptive Generator Balance (AGB) module to retain more data information. Our method uses adaptive hyperparameters to avoid a large number of parameter tuning, which significantly outperforms the combination of existing techniques. Meanwhile, our method achieves stable and reliable performance on multiple benchmarks.

----

## [642] QAGait: Revisit Gait Recognition from a Quality Perspective

**Authors**: *Zengbin Wang, Saihui Hou, Man Zhang, Xu Liu, Chunshui Cao, Yongzhen Huang, Peipei Li, Shibiao Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28391](https://doi.org/10.1609/aaai.v38i6.28391)

**Abstract**:

Gait recognition is a promising biometric method that aims to identify pedestrians from their unique walking patterns. Silhouette modality, renowned for its easy acquisition, simple structure, sparse representation, and convenient modeling, has been widely employed in controlled in-the-lab research. However, as gait recognition rapidly advances from in-the-lab to in-the-wild scenarios, various conditions raise significant challenges for silhouette modality, including 1) unidentifiable low-quality silhouettes (abnormal segmentation, severe occlusion, or even non-human shape), and 2) identifiable but challenging silhouettes (background noise, non-standard posture, slight occlusion). To address these challenges, we revisit gait recognition pipeline and approach gait recognition from a quality perspective, namely QAGait. Specifically, we propose a series of cost-effective quality assessment strategies, including Maxmial Connect Area and Template Match to eliminate background noises and unidentifiable silhouettes, Alignment strategy to handle non-standard postures. We also propose two quality-aware loss functions to integrate silhouette quality into optimization within the embedding space. Extensive experiments demonstrate our QAGait can guarantee both gait reliability and performance enhancement. Furthermore, our quality assessment strategies can seamlessly integrate with existing gait datasets, showcasing our superiority. Code is available at https://github.com/wzb-bupt/QAGait.

----

## [643] Enhancing Hyperspectral Images via Diffusion Model and Group-Autoencoder Super-resolution Network

**Authors**: *Zhaoyang Wang, Dongyang Li, Mingyang Zhang, Hao Luo, Maoguo Gong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28392](https://doi.org/10.1609/aaai.v38i6.28392)

**Abstract**:

Existing hyperspectral image (HSI) super-resolution (SR) methods struggle to effectively capture the complex spectral-spatial relationships and low-level details, while diffusion models represent a promising generative model known for their exceptional performance in modeling complex relations and learning high and low-level visual features. The direct application of diffusion models to HSI SR is hampered by challenges such as difficulties in model convergence and protracted inference time. In this work, we introduce a novel Group-Autoencoder (GAE) framework that synergistically combines with the diffusion model to construct a highly effective HSI SR model (DMGASR). Our proposed GAE framework encodes high-dimensional HSI data into low-dimensional latent space where the diffusion model works, thereby alleviating the difficulty of training the diffusion model while maintaining band correlation and considerably reducing inference time. Experimental results on both natural and remote sensing hyperspectral datasets demonstrate that the proposed method is superior to other state-of-the-art methods both visually and metrically.

----

## [644] SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing

**Authors**: *Zhecheng Wang, Rajanie Prabha, Tianyuan Huang, Jiajun Wu, Ram Rajagopal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28393](https://doi.org/10.1609/aaai.v38i6.28393)

**Abstract**:

Remote sensing imagery, despite its broad applications in helping achieve Sustainable Development Goals and tackle climate change, has not yet benefited from the recent advancements of versatile, task-agnostic vision language models (VLMs). A key reason is that the large-scale, semantically diverse image-text dataset required for developing VLMs is still absent for remote sensing images. Unlike natural images, remote sensing images and their associated text descriptions cannot be efficiently collected from the public Internet at scale. In this work, we bridge this gap by using geo-coordinates to automatically connect open, unlabeled remote sensing images with rich semantics covered in OpenStreetMap, and thus construct SkyScript, a comprehensive vision-language dataset for remote sensing images, comprising 2.6 million image-text pairs covering 29K distinct semantic tags. 
With continual pre-training on this dataset, we obtain a VLM that surpasses baseline models with a 6.2% average accuracy gain in zero-shot scene classification across seven benchmark datasets. It also demonstrates the ability of zero-shot transfer for fine-grained object attribute classification and cross-modal retrieval. We hope this dataset can support the advancement of VLMs for various multi-modal tasks in remote sensing, such as open-vocabulary classification, retrieval, captioning, and text-to-image synthesis.

----

## [645] DTMFormer: Dynamic Token Merging for Boosting Transformer-Based Medical Image Segmentation

**Authors**: *Zhehao Wang, Xian Lin, Nannan Wu, Li Yu, Kwang-Ting Cheng, Zengqiang Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28394](https://doi.org/10.1609/aaai.v38i6.28394)

**Abstract**:

Despite the great potential in capturing long-range dependency, one rarely-explored underlying issue of transformer in medical image segmentation is attention collapse, making it often degenerate into a bypass module in CNN-Transformer hybrid architectures. This is due to the high computational complexity of vision transformers requiring extensive training data while well-annotated medical image data is relatively limited, resulting in poor convergence. In this paper, we propose a plug-n-play transformer block with dynamic token merging, named DTMFormer, to avoid building long-range dependency on redundant and duplicated tokens and thus pursue better convergence. Specifically, DTMFormer consists of an attention-guided token merging (ATM) module to adaptively cluster tokens into fewer semantic tokens based on feature and dependency similarity and a light token reconstruction module to fuse ordinary and semantic tokens. In this way, as self-attention in ATM is calculated based on fewer tokens, DTMFormer is of lower complexity and more friendly to converge. Extensive experiments on publicly-available datasets demonstrate the effectiveness of DTMFormer working as a plug-n-play module for simultaneous complexity reduction and performance improvement. We believe it will inspire future work on rethinking transformers in medical image segmentation. Code: https://github.com/iam-nacl/DTMFormer.

----

## [646] SGNet: Structure Guided Network via Gradient-Frequency Awareness for Depth Map Super-resolution

**Authors**: *Zhengxue Wang, Zhiqiang Yan, Jian Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28395](https://doi.org/10.1609/aaai.v38i6.28395)

**Abstract**:

Depth super-resolution (DSR) aims to restore high-resolution (HR) depth from low-resolution (LR) one, where RGB image is often used to promote this task. Recent image guided DSR approaches mainly focus on spatial domain to rebuild depth structure. However, since the structure of LR depth is usually blurry, only considering spatial domain is not very sufficient to acquire satisfactory results. In this paper, we propose structure guided network (SGNet), a method that pays more attention to gradient and frequency domains, both of which have the inherent ability to capture high-frequency structure. Specifically, we first introduce the gradient calibration module (GCM), which employs the accurate gradient prior of RGB to sharpen the LR depth structure. Then we present the Frequency Awareness Module (FAM) that recursively conducts multiple spectrum differencing blocks (SDB), each of which propagates the precise high-frequency components of RGB into the LR depth. Extensive experimental results on both real and synthetic datasets demonstrate the superiority of our SGNet, reaching the state-of-the-art (see Fig. 1). Codes and pre-trained models are available at https://github.com/yanzq95/SGNet.

----

## [647] Vision Transformer Off-the-Shelf: A Surprising Baseline for Few-Shot Class-Agnostic Counting

**Authors**: *Zhicheng Wang, Liwen Xiao, Zhiguo Cao, Hao Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28396](https://doi.org/10.1609/aaai.v38i6.28396)

**Abstract**:

Class-agnostic counting (CAC) aims to count objects of interest from a query image given few exemplars. This task is typically addressed by extracting the features of query image and exemplars respectively and then matching their feature similarity, leading to an extract-then-match paradigm. In this work, we show that CAC can be simplified in an extract-and-match manner, particularly using a vision transformer (ViT) where feature extraction and similarity matching are executed simultaneously within the self-attention. We reveal the rationale of such simplification from a decoupled view of the self-attention.The resulting model, termed CACViT, simplifies the CAC pipeline into a single pretrained plain ViT. Further, to compensate the loss of the scale and the order-of-magnitude information due to resizing and normalization in plain ViT, we present two effective strategies for scale and magnitude embedding. Extensive experiments on the FSC147 and the CARPK datasets show that CACViT significantly outperforms state-of-the-art CAC approaches in both effectiveness (23.60% error reduction) and generalization, which suggests CACViT provides a concise and strong baseline for CAC. Code will be available.

----

## [648] Existence Is Chaos: Enhancing 3D Human Motion Prediction with Uncertainty Consideration

**Authors**: *Zhihao Wang, Yulin Zhou, Ningyu Zhang, Xiaosong Yang, Jun Xiao, Zhao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28397](https://doi.org/10.1609/aaai.v38i6.28397)

**Abstract**:

Human motion prediction is consisting in forecasting future body poses from historically observed sequences. It is a longstanding challenge due to motion's complex dynamics and uncertainty. Existing methods focus on building up complicated neural networks to model the motion dynamics. The predicted results are required to be strictly similar to the training samples with L2 loss in current training pipeline. However, little attention has been paid to the uncertainty property which is crucial to the prediction task. We argue that the recorded motion in training data could be an observation of possible future, rather than a predetermined result. In addition, existing works calculate the predicted error on each future frame equally during training, while recent work indicated that different frames could play different roles. In this work, a novel computationally efficient encoder-decoder model with uncertainty consideration is proposed, which could learn proper characteristics for future frames by a dynamic function. Experimental results on benchmark datasets demonstrate that our uncertainty consideration approach has obvious advantages both in quantity and quality. Moreover, the proposed method could produce motion sequences with much better quality that avoids the intractable shaking artefacts. We believe our work could provide a novel perspective to consider the uncertainty quality for the general motion prediction task and encourage the studies in this field. The code will be available in
https://github.com/Motionpre/Adaptive-Salient-Loss-SAGGB.

----

## [649] Heterogeneous Test-Time Training for Multi-Modal Person Re-identification

**Authors**: *Zi Wang, Huaibo Huang, Aihua Zheng, Ran He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28398](https://doi.org/10.1609/aaai.v38i6.28398)

**Abstract**:

Multi-modal person re-identification (ReID) seeks to mitigate challenging lighting conditions by incorporating diverse modalities. Most existing multi-modal ReID methods concentrate on leveraging complementary multi-modal information via fusion or interaction. However, the relationships among heterogeneous modalities and the domain traits of unlabeled test data are rarely explored. In this paper, we propose a Heterogeneous Test-time Training (HTT) framework for multi-modal person ReID. We first propose a Cross-identity Inter-modal Margin (CIM) loss to amplify the differentiation among distinct identity samples. Moreover, we design a Multi-modal Test-time Training (MTT) strategy to enhance the generalization of the model by leveraging the relationships in the heterogeneous modalities and the information existing in the test data. Specifically, in the training stage, we utilize the CIM loss to further enlarge the distance between anchor and negative by forcing the inter-modal distance to maintain the margin, resulting in an enhancement of the discriminative capacity of the ultimate descriptor. Subsequently, since the test data contains characteristics of the target domain, we adapt the MTT strategy to optimize the network before the inference by using self-supervised tasks designed based on relationships among modalities. Experimental results on benchmark multi-modal ReID datasets RGBNT201, Market1501-MM, RGBN300, and RGBNT100 validate the effectiveness of the proposed method. The codes can be found at https://github.com/ziwang1121/HTT.

----

## [650] Fine-Grained Prototypes Distillation for Few-Shot Object Detection

**Authors**: *Zichen Wang, Bo Yang, Haonan Yue, Zhenghao Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28399](https://doi.org/10.1609/aaai.v38i6.28399)

**Abstract**:

Few-shot object detection (FSOD) aims at extending a generic detector for novel object detection with only a few training examples. It attracts great concerns recently due to the practical meanings. Meta-learning has been demonstrated to be an effective paradigm for this task. In general, methods based on meta-learning employ an additional support branch to encode novel examples (a.k.a. support images) into class prototypes, which are then fused with query branch to facilitate the model prediction. However, the class-level prototypes are difficult to precisely generate, and they also lack detailed information, leading to instability in performance. New methods are required to capture the distinctive local context for more robust novel object detection. To this end, we propose to distill the most representative support features into fine-grained prototypes. These prototypes are then assigned into query feature maps based on the matching results, modeling the detailed feature relations between two branches. This process is realized by our Fine-Grained Feature Aggregation (FFA) module. Moreover, in terms of high-level feature fusion, we propose Balanced Class-Agnostic Sampling (B-CAS) strategy and Non-Linear Fusion (NLF) module from differenct perspectives. They are complementary to each other and depict the high-level feature relations more effectively. Extensive experiments on PASCAL VOC and MS COCO benchmarks show that our method sets a new state-of-the-art performance in most settings. Our code is available at https://github.com/wangchen1801/FPD.

----

## [651] Semantic Complete Scene Forecasting from a 4D Dynamic Point Cloud Sequence

**Authors**: *Zifan Wang, Zhuorui Ye, Haoran Wu, Junyu Chen, Li Yi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28400](https://doi.org/10.1609/aaai.v38i6.28400)

**Abstract**:

We study a new problem of semantic complete scene forecasting (SCSF) in this work. Given a 4D dynamic point cloud sequence, our goal is to forecast the complete scene corresponding to the future next frame along with its semantic labels. To tackle this challenging problem, we properly model the synergetic relationship between future forecasting and semantic scene completion through a novel network named SCSFNet. SCSFNet leverages a hybrid geometric representation for high-resolution complete scene forecasting. To leverage multi-frame observation as well as the understanding of scene dynamics to ease the completion task, SCSFNet introduces an attention-based skip connection scheme. To ease the need to model occlusion variations and to better focus on the occluded part, SCSFNet utilizes auxiliary visibility grids to guide the forecasting task. To evaluate the effectiveness of SCSFNet, we conduct experiments on various benchmarks including two large-scale indoor benchmarks we contributed and the outdoor SemanticKITTI benchmark. Extensive experiments show SCSFNet outperforms baseline methods on multiple metrics by a large margin, and also prove the synergy between future forecasting and semantic scene completion.The project page with code is available at scsfnet.github.io.

----

## [652] Enhanced Fine-Grained Motion Diffusion for Text-Driven Human Motion Synthesis

**Authors**: *Dong Wei, Xiaoning Sun, Huaijiang Sun, Shengxiang Hu, Bin Li, Weiqing Li, Jianfeng Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28401](https://doi.org/10.1609/aaai.v38i6.28401)

**Abstract**:

The emergence of text-driven motion synthesis technique provides animators with great potential to create efficiently. However, in most cases, textual expressions only contain general and qualitative motion descriptions, while lack fine depiction and sufficient intensity, leading to the synthesized motions that either (a) semantically compliant but uncontrollable over specific pose details, or (b) even deviates from the provided descriptions, bringing animators with undesired cases. In this paper, we propose DiffKFC, a conditional diffusion model for text-driven motion synthesis with KeyFrames Collaborated, enabling realistic generation with collaborative and efficient dual-level control: coarse guidance at semantic level, with only few keyframes for direct and fine-grained depiction down to body posture level. Unlike existing inference-editing diffusion models that incorporate conditions without training, our conditional diffusion model is explicitly trained and can fully exploit correlations among texts, keyframes and the diffused target frames. To preserve the control capability of discrete and sparse keyframes, we customize dilated mask attention modules where only partial valid tokens participate in local-to-global attention, indicated by the dilated keyframe mask. Additionally, we develop a simple yet effective smoothness prior, which steers the generated frames towards seamless keyframe transitions at inference. Extensive experiments show that our model not only achieves state-of-the-art performance in terms of semantic fidelity, but more importantly, is able to satisfy animator requirements through fine-grained guidance without tedious labor.

----

## [653] Image as a Language: Revisiting Scene Text Recognition via Balanced, Unified and Synchronized Vision-Language Reasoning Network

**Authors**: *Jiajun Wei, Hongjian Zhan, Yue Lu, Xiao Tu, Bing Yin, Cong Liu, Umapada Pal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28402](https://doi.org/10.1609/aaai.v38i6.28402)

**Abstract**:

Scene text recognition is inherently a vision-language task. However, previous works have predominantly focused either on extracting more robust visual features or designing better language modeling. How to effectively and jointly model vision and language to mitigate heavy reliance on a single modality remains a problem. In this paper, aiming to enhance vision-language reasoning in scene text recognition, we present a balanced, unified and synchronized vision-language reasoning network (BUSNet). Firstly, revisiting the image as a language by balanced concatenation along length dimension alleviates the issue of over-reliance on vision or language. Secondly, BUSNet learns an ensemble of unified external and internal vision-language model with shared weight by masked modality modeling (MMM). Thirdly, a novel vision-language reasoning module (VLRM) with synchronized vision-language decoding capacity is proposed. Additionally, BUSNet achieves improved performance through iterative reasoning, which utilizes the vision-language prediction as a new language input. Extensive experiments indicate that BUSNet achieves state-of-the-art performance on several mainstream benchmark datasets and more challenge datasets for both synthetic and real training data compared to recent outstanding methods. Code and dataset will be available at https://github.com/jjwei66/BUSNet.

----

## [654] WeakPCSOD: Overcoming the Bias of Box Annotations for Weakly Supervised Point Cloud Salient Object Detection

**Authors**: *Jun Wei, S. Kevin Zhou, Shuguang Cui, Zhen Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28403](https://doi.org/10.1609/aaai.v38i6.28403)

**Abstract**:

Point cloud salient object detection (PCSOD) is a newly proposed task in 3D dense segmentation. However, the acquisition of accurate 3D dense annotations comes at a high cost, severely limiting the progress of PCSOD. To address this issue, we propose the first weakly supervised PCSOD (named WeakPCSOD) model, which relies solely on cheap 3D bounding box annotations. In WeakPCSOD, we extract noise-free supervision from coarse 3D bounding boxes while mitigating shape biases inherent in box annotations. To achieve this, we introduce a novel mask-to-box (M2B) transformation and a color consistency (CC) loss. The M2B transformation, from a shape perspective, disentangles predictions from labels, enabling the extraction of noiseless supervision from labels while preserving object shapes independently of the box bias. From an appearance perspective, we further introduce the CC loss to provide dense supervision, which mitigates the non-unique predictions stemming from weak supervision and substantially reduces prediction variability. Furthermore, we employ a self-training (ST) strategy to enhance performance by utilizing high-confidence pseudo labels. Notably, the M2B transformation, CC loss, and ST strategy are seamlessly integrated into any model and incur no computational costs for inference. Extensive experiments demonstrate the effectiveness of our WeakPCSOD model, even comparable to fully supervised models utilizing dense annotations.

----

## [655] RetouchFormer: Semi-supervised High-Quality Face Retouching Transformer with Prior-Based Selective Self-Attention

**Authors**: *Xue Wen, Lianxin Xie, Le Jiang, Tianyi Chen, Si Wu, Cheng Liu, Hau-San Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28404](https://doi.org/10.1609/aaai.v38i6.28404)

**Abstract**:

Face retouching is to beautify a face image, while preserving the image content as much as possible. It is a promising yet challenging task to remove face imperfections and fill with normal skin. Generic image enhancement methods are hampered by the lack of imperfection localization, which often results in incomplete removal of blemishes at large scales. To address this issue, we propose a transformer-based approach, RetouchFormer, which simultaneously identify imperfections and synthesize realistic content in the corresponding regions. Specifically, we learn a latent dictionary to capture the clean face priors, and predict the imperfection regions via a reconstruction-oriented localization module. Also based on this, we can realize face retouching by explicitly suppressing imperfections in our selective self-attention computation, such that local content will be synthesized from normal skin. On the other hand, multi-scale feature tokens lead to increased flexibility in dealing with the imperfections at various scales. The design elements bring greater effectiveness and efficiency. RetouchFormer outperforms the advanced face retouching methods and synthesizes clean face images with high fidelity in our list of extensive experiments performed.

----

## [656] Mean Teacher DETR with Masked Feature Alignment: A Robust Domain Adaptive Detection Transformer Framework

**Authors**: *Weixi Weng, Chun Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28405](https://doi.org/10.1609/aaai.v38i6.28405)

**Abstract**:

Unsupervised domain adaptation object detection(UDAOD) research on Detection Transformer(DETR) mainly focuses on feature alignment and existing methods can be divided into two kinds, each of which has its unresolved issues. One-stage feature alignment methods can easily lead to performance fluctuation and training stagnation. Two-stage feature alignment method based on mean teacher comprises a pretraining stage followed by a self-training stage, each facing problems in obtaining reliable pretrained model and achieving consistent performance gains. Methods mentioned above have not yet explore how to utilize the third related domain such as target-like domain to assist adaptation. To address these issues, we propose a two-stage framework named MTM, i.e. Mean Teacher-DETR with Masked Feature Alignment. In the pretraining stage, we utilize labeled target-like images produced by image style transfer to avoid performance fluctuation. In the self-training stage, we leverage unlabeled target images by pseudo labels based on mean teacher and propose a module called Object Queries Knowledge Transfer(OQKT) to ensure consistent performance gains of the student model. Most importantly, we propose masked feature alignment methods including Masked Domain Query-based Feature Alignment(MDQFA) and Masked Token-wise Feature Alignment(MTWFA) to alleviate domain shift in a more robust way, which not only prevent training stagnation and lead to a robust pretrained model in the pretraining stage, but also enhance the model's target performance in the self-training stage. Experiments on three challenging scenarios and a theoretical analysis verify the effectiveness of MTM.

----

## [657] Keep the Faith: Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning

**Authors**: *Tom Nuno Wolf, Fabian Bongratz, Anne-Marie Rickmann, Sebastian Pölsterl, Christian Wachinger*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28406](https://doi.org/10.1609/aaai.v38i6.28406)

**Abstract**:

Explaining predictions of black-box neural networks is crucial when applied to decision-critical tasks. Thus, attribution maps are commonly used to identify important image regions, despite prior work showing that humans prefer explanations based on similar examples. To this end, ProtoPNet learns a set of class-representative feature vectors (prototypes) for case-based reasoning. During inference, similarities of latent features to prototypes are linearly classified to form predictions and attribution maps are provided to explain the similarity. In this work, we evaluate whether architectures for case-based reasoning fulfill established axioms required for faithful explanations using the example of ProtoPNet. We show that such architectures allow the extraction of faithful explanations. However, we prove that the attribution maps used to explain the similarities violate the axioms. We propose a new procedure to extract explanations for trained ProtoPNets, named ProtoPFaith. Conceptually, these explanations are Shapley values, calculated on the similarity scores of each prototype. They allow to faithfully answer which prototypes are present in an unseen image and quantify each pixel’s contribution to that presence, thereby complying with all axioms. The theoretical violations of ProtoPNet manifest in our experiments on three datasets (CUB-200-2011, Stanford Dogs, RSNA) and five architectures (ConvNet, ResNet, ResNet50, WideResNet50, ResNeXt50). Our experiments show a qualitative difference between the explanations given by ProtoPNet and ProtoPFaith. Additionally, we quantify the explanations with the Area Over the Perturbation Curve, on which ProtoPFaith outperforms ProtoPNet on all experiments by a factor >10^3.

----

## [658] Factorized Diffusion Autoencoder for Unsupervised Disentangled Representation Learning

**Authors**: *Ancong Wu, Wei-Shi Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28407](https://doi.org/10.1609/aaai.v38i6.28407)

**Abstract**:

Unsupervised disentangled representation learning aims to recover semantically meaningful factors from real-world data without supervision, which is significant for model generalization and interpretability. Current methods mainly rely on assumptions of independence or informativeness of factors, regardless of interpretability. Intuitively, visually interpretable concepts better align with human-defined factors. However, exploiting visual interpretability as inductive bias is still under-explored. Inspired by the observation that most explanatory image factors can be represented by ``content + mask'', we propose a content-mask factorization network (CMFNet) to decompose an image into different groups of content codes and masks, which are further combined as content masks to represent different visual concepts. To ensure informativeness of the representations, the CMFNet is jointly learned with a generator conditioned on the content masks for reconstructing the input image. The conditional generator employs a diffusion model to leverage its robust distribution modeling capability. Our model is called the Factorized Diffusion Autoencoder (FDAE). To enhance disentanglement of visual concepts, we propose a content decorrelation loss and a mask entropy loss to decorrelate content masks in latent space and spatial space, respectively. Experiments on Shapes3d, MPI3D and Cars3d show that our method achieves advanced performance and can generate visually interpretable concept-specific masks. Source code and supplementary materials are available at https://github.com/wuancong/FDAE.

----

## [659] 3D-STMN: Dependency-Driven Superpoint-Text Matching Network for End-to-End 3D Referring Expression Segmentation

**Authors**: *Changli Wu, Yiwei Ma, Qi Chen, Haowei Wang, Gen Luo, Jiayi Ji, Xiaoshuai Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28408](https://doi.org/10.1609/aaai.v38i6.28408)

**Abstract**:

In 3D Referring Expression Segmentation (3D-RES), the earlier approach adopts a two-stage paradigm, extracting segmentation proposals and then matching them with referring expressions. However, this conventional paradigm encounters significant challenges, most notably in terms of the generation of lackluster initial proposals and a pronounced deceleration in inference speed. Recognizing these limitations, we introduce an innovative end-to-end Superpoint-Text Matching Network (3D-STMN) that is enriched by dependency-driven insights. One of the keystones of our model is the Superpoint-Text Matching (STM) mechanism. Unlike traditional methods that navigate through instance proposals, STM directly correlates linguistic indications with their respective superpoints, clusters of semantically related points. This architectural decision empowers our model to efficiently harness cross-modal semantic relationships, primarily leveraging densely annotated superpoint-text pairs, as opposed to the more sparse instance-text pairs. In pursuit of enhancing the role of text in guiding the segmentation process, we further incorporate the Dependency-Driven Interaction (DDI) module to deepen the network's semantic comprehension of referring expressions. Using the dependency trees as a beacon, this module discerns the intricate relationships between primary terms and their associated descriptors in expressions, thereby elevating both the localization and segmentation capacities. Comprehensive experiments on the ScanRefer benchmark reveal that our model not only sets new performance standards, registering an mIoU gain of 11.7 points but also achieves a staggering enhancement in inference speed, surpassing traditional methods by 95.7 times. The code and models are available at https://github.com/sosppxo/3D-STMN.

----

## [660] SCD-Net: Spatiotemporal Clues Disentanglement Network for Self-Supervised Skeleton-Based Action Recognition

**Authors**: *Cong Wu, Xiao-Jun Wu, Josef Kittler, Tianyang Xu, Sara Ahmed, Muhammad Awais, Zhenhua Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28409](https://doi.org/10.1609/aaai.v38i6.28409)

**Abstract**:

Contrastive learning has achieved great success in skeleton-based action recognition. However, most existing approaches encode the skeleton sequences as entangled spatiotemporal representations and confine the contrasts to the same level of representation. Instead, this paper introduces a novel contrastive learning framework, namely Spatiotemporal Clues Disentanglement Network (SCD-Net). Specifically, we integrate the decoupling module with a feature extractor to derive explicit clues from spatial and temporal domains respectively. As for the training of SCD-Net, with a constructed global anchor, we encourage the interaction between the anchor and extracted clues. Further, we propose a new masking strategy with structural constraints to strengthen the contextual associations, leveraging the latest development from masked image modelling into the proposed SCD-Net. We conduct extensive evaluations on the NTU-RGB+D (60&120) and PKU-MMD (I&II) datasets, covering various downstream tasks such as action recognition, action retrieval, transfer learning, and semi-supervised learning. The experimental results demonstrate the effectiveness of our method, which outperforms the existing state-of-the-art (SOTA) approaches significantly. Our code and supplementary material can be found at https://github.com/cong-wu/SCD-Net.

----

## [661] G-NAS: Generalizable Neural Architecture Search for Single Domain Generalization Object Detection

**Authors**: *Fan Wu, Jinling Gao, Lanqing Hong, Xinbing Wang, Chenghu Zhou, Nanyang Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28410](https://doi.org/10.1609/aaai.v38i6.28410)

**Abstract**:

In this paper, we focus on a realistic yet challenging task, Single Domain Generalization Object Detection (S-DGOD), where only one source domain's data can be used for training object detectors, but have to generalize multiple distinct target domains. In S-DGOD, both high-capacity fitting and generalization abilities are needed due to the task's complexity. Differentiable Neural Architecture Search (NAS) is known for its high capacity for complex data fitting and we propose to leverage Differentiable NAS to solve S-DGOD. However, it may confront severe over-fitting issues due to the feature imbalance phenomenon, where parameters optimized by gradient descent are biased to learn from the easy-to-learn features, which are usually non-causal and spuriously correlated to ground truth labels, such as the features of background in object detection data. Consequently, this leads to serious performance degradation, especially in generalizing to unseen target domains with huge domain gaps between the source domain and target domains. To address this issue, we propose the Generalizable loss (G-loss), which is an OoD-aware objective, preventing NAS from over-fitting by using gradient descent to optimize parameters not only on a subset of easy-to-learn features but also the remaining predictive features for generalization, and the overall framework is named G-NAS. Experimental results on the S-DGOD urban-scene datasets demonstrate that the proposed G-NAS achieves SOTA performance compared to baseline methods. Codes are available at https://github.com/wufan-cse/G-NAS.

----

## [662] Multiscale Low-Frequency Memory Network for Improved Feature Extraction in Convolutional Neural Networks

**Authors**: *Fuzhi Wu, Jiasong Wu, Youyong Kong, Chunfeng Yang, Guanyu Yang, Huazhong Shu, Guy Carrault, Lotfi Senhadji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28411](https://doi.org/10.1609/aaai.v38i6.28411)

**Abstract**:

Deep learning and Convolutional Neural Networks (CNNs) have driven major transformations in diverse research areas. However, their limitations in handling low-frequency in-formation present obstacles in certain tasks like interpreting global structures or managing smooth transition images. Despite the promising performance of transformer struc-tures in numerous tasks, their intricate optimization com-plexities highlight the persistent need for refined CNN en-hancements using limited resources. Responding to these complexities, we introduce a novel framework, the Mul-tiscale Low-Frequency Memory (MLFM) Network, with the goal to harness the full potential of CNNs while keep-ing their complexity unchanged. The MLFM efficiently preserves low-frequency information, enhancing perfor-mance in targeted computer vision tasks. Central to our MLFM is the Low-Frequency Memory Unit (LFMU), which stores various low-frequency data and forms a parallel channel to the core network. A key advantage of MLFM is its seamless compatibility with various prevalent networks, requiring no alterations to their original core structure. Testing on ImageNet demonstrated substantial accuracy improvements in multiple 2D CNNs, including ResNet, MobileNet, EfficientNet, and ConvNeXt. Furthermore, we showcase MLFM's versatility beyond traditional image classification by successfully integrating it into image-to-image translation tasks, specifically in semantic segmenta-tion networks like FCN and U-Net. In conclusion, our work signifies a pivotal stride in the journey of optimizing the ef-ficacy and efficiency of CNNs with limited resources. This research builds upon the existing CNN foundations and paves the way for future advancements in computer vision. Our codes are available at https://github.com/AlphaWuSeu/MLFM.

----

## [663] Learning from History: Task-agnostic Model Contrastive Learning for Image Restoration

**Authors**: *Gang Wu, Junjun Jiang, Kui Jiang, Xianming Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28412](https://doi.org/10.1609/aaai.v38i6.28412)

**Abstract**:

Contrastive learning has emerged as a prevailing paradigm for high-level vision tasks, which, by introducing properly negative samples, has also been exploited for low-level vision tasks to achieve a compact optimization space to account for their ill-posed nature. However, existing methods rely on manually predefined and task-oriented negatives, which often exhibit pronounced task-specific biases. To address this challenge, our paper introduces an innovative method termed 'learning from history', which dynamically generates negative samples from the target model itself. Our approach, named Model Contrastive Learning for Image Restoration (MCLIR), rejuvenates latency models as negative models, making it compatible with diverse image restoration tasks. We propose the Self-Prior guided Negative loss (SPN) to enable it. This approach significantly enhances existing models when retrained with the proposed model contrastive paradigm. The results show significant improvements in image restoration across various tasks and architectures. For example, models retrained with SPN outperform the original FFANet and DehazeFormer by 3.41 and 0.57 dB on the RESIDE indoor dataset for image dehazing. Similarly, they achieve notable improvements of 0.47 dB on SPA-Data over IDT for image deraining and 0.12 dB on Manga109 for a 4x scale super-resolution over lightweight SwinIR, respectively. Code and retrained models are available at https://github.com/Aitical/MCLIR.

----

## [664] Hybrid-Supervised Dual-Search: Leveraging Automatic Learning for Loss-Free Multi-Exposure Image Fusion

**Authors**: *Guanyao Wu, Hongming Fu, Jinyuan Liu, Long Ma, Xin Fan, Risheng Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28413](https://doi.org/10.1609/aaai.v38i6.28413)

**Abstract**:

Multi-exposure image fusion (MEF) has emerged as a prominent solution to address the limitations of digital imaging in representing varied exposure levels. Despite its advancements, the field grapples with challenges, notably the reliance on manual designs for network structures and loss functions, and the constraints of utilizing simulated reference images as ground truths. Consequently, current methodologies often suffer from color distortions and exposure artifacts, further complicating the quest for authentic image representation. In addressing these challenges, this paper presents a Hybrid-Supervised Dual-Search approach for MEF, dubbed HSDS-MEF, which introduces a bi-level optimization search scheme for automatic design of both network structures and loss functions. More specifically, we harness a unique dual research mechanism rooted in a novel weighted structure refinement architecture search. Besides, a hybrid supervised contrast constraint seamlessly guides and integrates with searching process, facilitating a more adaptive and comprehensive search for optimal loss functions. We realize the state-of-the-art performance in comparison to various competitive schemes, yielding a 10.61% and 4.38% improvement in Visual Information Fidelity (VIF)
for general and no-reference scenarios, respectively, while providing results with high contrast, rich details and colors. The code is available at https://github.com/RollingPlain/HSDS_MEF.

----

## [665] When to Grow? A Fitting Risk-Aware Policy for Layer Growing in Deep Neural Networks

**Authors**: *Haihang Wu, Wei Wang, Tamasha Malepathirana, Damith A. Senanayake, Denny Oetomo, Saman K. Halgamuge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28414](https://doi.org/10.1609/aaai.v38i6.28414)

**Abstract**:

Neural growth is the process of growing a small neural network to a large network and has been utilized to accelerate the training of deep neural networks. One crucial aspect of neural growth is determining the optimal growth timing. However, few studies investigate this systematically. Our study reveals that neural growth inherently exhibits a regularization effect, whose intensity is influenced by the chosen policy for growth timing. While this regularization effect may mitigate the overfitting risk of the model, it may lead to a notable accuracy drop when the model underfits. Yet, current approaches have not addressed this issue due to their lack of consideration of the regularization effect from neural growth. Motivated by these findings, we propose an under/over fitting risk-aware growth timing policy, which automatically adjusts the growth timing informed by the level of potential under/overfitting risks to address both risks.  Comprehensive experiments conducted using CIFAR-10/100 and ImageNet datasets show that the proposed policy achieves accuracy improvements of up to 1.3% in models prone to underfitting while achieving similar accuracies in models suffering from overfitting compared to the existing methods.

----

## [666] p-Laplacian Adaptation for Generative Pre-trained Vision-Language Models

**Authors**: *Haoyuan Wu, Xinyun Zhang, Peng Xu, Peiyu Liao, Xufeng Yao, Bei Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28415](https://doi.org/10.1609/aaai.v38i6.28415)

**Abstract**:

Vision-Language models (VLMs) pre-trained on large corpora have demonstrated notable success across a range of downstream tasks. In light of the rapidly increasing size of pre-trained VLMs, parameter-efficient transfer learning (PETL) has garnered attention as a viable alternative to full fine-tuning. One such approach is the adapter, which introduces a few trainable parameters into the pre-trained models while preserving the original parameters during adaptation.
In this paper, we present a novel modeling framework that recasts adapter tuning after attention as a graph message passing process on attention graphs, where the projected query and value features and attention matrix constitute the node features and the graph adjacency matrix, respectively. Within this framework, tuning adapters in VLMs necessitates handling heterophilic graphs, owing to the disparity between the projected query and value space.
To address this challenge, we propose a new adapter architecture, p-adapter, which employs p-Laplacian message passing in Graph Neural Networks (GNNs). Specifically, the attention weights are re-normalized based on the features, and the features are then aggregated using the calibrated attention matrix, enabling the dynamic exploitation of information with varying frequencies in the heterophilic attention graphs.
We conduct extensive experiments on different pre-trained VLMs and multi-modal tasks, including visual question answering, visual entailment, and image captioning. The experimental results validate our method's significant superiority over other PETL methods. Our code is available at https://github.com/wuhy68/p-Adapter/.

----

## [667] Task-Adaptive Prompted Transformer for Cross-Domain Few-Shot Learning

**Authors**: *Jiamin Wu, Xin Liu, Xiaotian Yin, Tianzhu Zhang, Yongdong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28416](https://doi.org/10.1609/aaai.v38i6.28416)

**Abstract**:

Cross-Domain Few-Shot Learning (CD-FSL) aims at recognizing samples in novel classes from unseen domains that are vastly different from training classes, with few labeled samples. However, the large domain gap between training and novel classes makes previous FSL methods perform poorly. To address this issue, we propose MetaPrompt, a Task-adaptive Prompted Transformer model for CD-FSL, by jointly exploiting prompt learning and the parameter generation framework. The proposed MetaPrompt enjoys several merits. First, a task-conditioned prompt generator is established upon attention mechanisms. It can flexibly produce a task-adaptive prompt with arbitrary length for unseen tasks, by selectively gathering task characteristics from the contextualized support embeddings. Second, the task-adaptive prompt is attached to Vision Transformer to facilitate fast task adaptation, steering the task-agnostic representation to incorporate task knowledge. To our best knowledge, this is the first work to exploit a prompt-based parameter generation mechanism for CD-FSL. Extensive experimental results on the Meta-Dataset benchmark demonstrate that our method achieves superior results against state-of-the-art methods.

----

## [668] SyFormer: Structure-Guided Synergism Transformer for Large-Portion Image Inpainting

**Authors**: *Jie Wu, Yuchao Feng, Honghui Xu, Chuanmeng Zhu, Jianwei Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28417](https://doi.org/10.1609/aaai.v38i6.28417)

**Abstract**:

Image inpainting is in full bloom accompanied by the progress of convolutional neural networks (CNNs) and transformers, revolutionizing the practical management of abnormity disposal, image editing, etc. However, due to the ever-mounting image resolutions and missing areas, the challenges of distorted long-range dependencies from cluttered background distributions and reduced reference information in image domain inevitably rise, which further cause severe performance degradation. To address the challenges, we propose a novel large-portion image inpainting approach, namely the Structure-Guided Synergism Transformer (SyFormer), to rectify the discrepancies in feature representation and enrich the structural cues from limited reference. Specifically, we devise a dual-routing filtering module that employs a progressive filtering strategy to eliminate invalid noise interference and establish global-level texture correlations. Simultaneously, the structurally compact perception module maps an affinity matrix within the introduced structural priors from a structure-aware generator, assisting in matching and filling the corresponding patches of large-proportionally damaged images. Moreover, we carefully assemble the aforementioned modules to achieve feature complementarity. Finally, a feature decoding alignment scheme is introduced in the decoding process, which meticulously achieves texture amalgamation across hierarchical features. Extensive experiments are conducted on two publicly available datasets, i.e., CelebA-HQ and Places2, to qualitatively and quantitatively demonstrate the superiority of our model over state-of-the-arts.

----

## [669] MedSegDiff-V2: Diffusion-Based Medical Image Segmentation with Transformer

**Authors**: *Junde Wu, Wei Ji, Huazhu Fu, Min Xu, Yueming Jin, Yanwu Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28418](https://doi.org/10.1609/aaai.v38i6.28418)

**Abstract**:

The Diffusion Probabilistic Model (DPM) has recently gained popularity in the field of computer vision, thanks to its image generation applications, such as Imagen, Latent Diffusion Models, and Stable Diffusion, which have demonstrated impressive capabilities and sparked much discussion within the community. Recent investigations have further unveiled the utility of DPM in the domain of medical image analysis, as underscored by the commendable performance exhibited by the medical image segmentation model across various tasks. Although these models were originally underpinned by a UNet architecture, there exists a potential avenue for enhancing their performance through the integration of vision transformer mechanisms. However, we discovered that simply combining these two models resulted in subpar performance. To effectively integrate these two cutting-edge techniques for the Medical image segmentation, we propose a novel Transformer-based Diffusion framework, called MedSegDiff-V2. We verify its effectiveness on 20 medical image segmentation tasks with different image modalities. Through comprehensive evaluation, our approach demonstrates superiority over prior state-of-the-art (SOTA) methodologies. Code is released at https://github.com/KidsWithTokens/MedSegDiff.

----

## [670] Selective and Orthogonal Feature Activation for Pedestrian Attribute Recognition

**Authors**: *Junyi Wu, Yan Huang, Min Gao, Yuzhen Niu, Mingjing Yang, Zhipeng Gao, Jianqiang Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28419](https://doi.org/10.1609/aaai.v38i6.28419)

**Abstract**:

Pedestrian Attribute Recognition (PAR) involves identifying the attributes of individuals in person images. Existing PAR methods typically rely on CNNs as the backbone network to extract pedestrian features. However, CNNs process only one adjacent region at a time, leading to the loss of long-range inter-relations between different attribute-specific regions. To address this limitation, we leverage the Vision Transformer (ViT) instead of CNNs as the backbone for PAR, aiming to model long-range relations and extract more robust features. However, PAR suffers from an inherent attribute imbalance issue, causing ViT to naturally focus more on attributes that appear frequently in the training set and ignore some pedestrian attributes that appear less. The native features extracted by ViT are not able to tolerate the imbalance attribute distribution issue. To tackle this issue, we propose two novel components: the Selective Feature Activation Method (SFAM) and the Orthogonal Feature Activation Loss. SFAM smartly suppresses the more informative attribute-specific features, compelling the PAR model to capture discriminative features from regions that are easily overlooked. The proposed loss enforces an orthogonal constraint on the original feature extracted by ViT and the suppressed features from SFAM, promoting the complementarity of features in space. We conduct experiments on several benchmark PAR datasets, including PETA, PA100K, RAPv1, and RAPv2, demonstrating the effectiveness of our method. Specifically, our method outperforms existing state-of-the-art approaches by GRL, IAA-Caps, ALM, and SSC in terms of mA on the four datasets, respectively.

----

## [671] Swift-Mapping: Online Neural Implicit Dense Mapping in Urban Scenes

**Authors**: *Ke Wu, Kaizhao Zhang, Mingzhe Gao, Jieru Zhao, Zhongxue Gan, Wenchao Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28420](https://doi.org/10.1609/aaai.v38i6.28420)

**Abstract**:

Online dense mapping of urban scenes is of paramount importance for scene understanding of autonomous navigation. Traditional online dense mapping methods fuse sensor measurements (vision, lidar, etc.) across time and space via explicit geometric correspondence. Recently, NeRF-based methods have proved the superiority of neural implicit representations by high-fidelity reconstruction of large-scale city scenes. However, it remains an open problem how to integrate powerful neural implicit representations into online dense mapping. Existing methods are restricted to constrained indoor environments and are too computationally expensive to meet online requirements. To this end, we propose Swift-Mapping, an online neural implicit dense mapping framework in urban scenes. We introduce a novel neural implicit octomap (NIO) structure that provides efficient neural representation for large and dynamic urban scenes while retaining online update capability. Based on that, we propose an online neural dense mapping framework that effectively manages and updates neural octree voxel features. Our approach achieves SOTA reconstruction accuracy while being more than 10x faster in reconstruction speed, demonstrating the superior performance of our method in both accuracy and efficiency.

----

## [672] CPN: Complementary Proposal Network for Unconstrained Text Detection

**Authors**: *Longhuang Wu, Shangxuan Tian, Youxin Wang, Pengfei Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28421](https://doi.org/10.1609/aaai.v38i6.28421)

**Abstract**:

Existing methods for scene text detection can be divided into two paradigms: segmentation-based and anchor-based. While Segmentation-based methods are well-suited for irregular shapes, they struggle with compact or overlapping layouts. Conversely, anchor-based approaches excel for complex layouts but suffer from irregular shapes. To strengthen their merits and overcome their respective demerits, we propose a Complementary Proposal Network (CPN) that seamlessly and parallelly integrates semantic and geometric information for superior performance. The CPN comprises two efficient networks for proposal generation: the Deformable Morphology Semantic Network, which generates semantic proposals employing an innovative deformable morphological operator, and the Balanced Region Proposal Network, which produces geometric proposals with pre-defined anchors. To further enhance the complementarity, we introduce an Interleaved Feature Attention module that enables semantic and geometric features to interact deeply before proposal generation. By leveraging both complementary proposals and features, CPN outperforms state-of-the-art approaches with significant margins under comparable computation cost. Specifically, our approach achieves improvements of 3.6%, 1.3% and 1.0% on challenging benchmarks ICDAR19-ArT, IC15, and MSRA-TD500, respectively. Code for our method will be released.

----

## [673] Toward Open-Set Human Object Interaction Detection

**Authors**: *Mingrui Wu, Yuqi Liu, Jiayi Ji, Xiaoshuai Sun, Rongrong Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28422](https://doi.org/10.1609/aaai.v38i6.28422)

**Abstract**:

This work is oriented toward the task of open-set Human Object Interaction (HOI) detection. The challenge lies in identifying completely new, out-of-domain relationships, as opposed to in-domain ones which have seen improvements in zero-shot HOI detection. To address this challenge, we introduce a simple Disentangled HOI Detection (DHD) model for detecting novel relationships by integrating an open-set object detector with a Visual Language Model (VLM). We utilize a disentangled image-text contrastive learning metric for training and connect the bottom-up visual features to text embeddings through lightweight unary and pair-wise adapters. Our model can benefit from the open-set object detector and the VLM to detect novel action categories and combine actions with novel object categories. We further present the VG-HOI dataset, a comprehensive benchmark with over 17k HOI relationships for open-set scenarios. Experimental results show that our model can detect unknown action classes and combine unknown object classes. Furthermore, it can generalize to over 17k HOI classes while being trained on just 600 HOI classes.

----

## [674] VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection

**Authors**: *Peng Wu, Xuerong Zhou, Guansong Pang, Lingru Zhou, Qingsen Yan, Peng Wang, Yanning Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28423](https://doi.org/10.1609/aaai.v38i6.28423)

**Abstract**:

The recent contrastive language-image pre-training (CLIP) model has shown great success in a wide range of image-level tasks, revealing remarkable ability for learning powerful visual representations with rich semantics. An open and worthwhile problem is efficiently adapting such a strong model to the video domain and designing a robust video anomaly detector. In this work, we propose VadCLIP, a new paradigm for weakly supervised video anomaly detection (WSVAD) by leveraging the frozen CLIP model directly without any pre-training and fine-tuning process. Unlike current works that directly feed extracted features into the weakly supervised classifier for frame-level binary classification, VadCLIP makes full use of fine-grained associations between vision and language on the strength of CLIP and involves dual branch. One branch simply utilizes visual features for coarse-grained binary classification, while the other fully leverages the fine-grained language-image alignment. With the benefit of dual branch, VadCLIP achieves both coarse-grained and fine-grained video anomaly detection by transferring pre-trained knowledge from CLIP to WSVAD task. We conduct extensive experiments on two commonly-used benchmarks, demonstrating that VadCLIP achieves the best performance on both coarse-grained and fine-grained WSVAD, surpassing the state-of-the-art methods by a large margin. Specifically, VadCLIP achieves 84.51% AP and 88.02% AUC on XD-Violence and UCF-Crime, respectively. Code and features are released at https://github.com/nwpu-zxr/VadCLIP.

----

## [675] Temporal Correlation Vision Transformer for Video Person Re-Identification

**Authors**: *Pengfei Wu, Le Wang, Sanping Zhou, Gang Hua, Changyin Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28424](https://doi.org/10.1609/aaai.v38i6.28424)

**Abstract**:

Video Person Re-Identification (Re-ID) is a task of retrieving persons from multi-camera surveillance systems. Despite the progress made in leveraging spatio-temporal information in videos, occlusion in dense crowds still hinders further progress. To address this issue, we propose a Temporal Correlation Vision Transformer (TCViT) for video person Re-ID. TCViT consists of a Temporal Correlation Attention (TCA) module and a Learnable Temporal Aggregation (LTA) module. The TCA module is designed to reduce the impact of non-target persons by relative state, while the LTA module is used to aggregate frame-level features based on their completeness. Specifically, TCA is a parameter-free module that first aligns frame-level features to restore semantic coherence in videos and then enhances the features of the target person according to temporal correlation. Additionally, unlike previous methods that treat each frame equally with a pooling layer, LTA introduces a lightweight learnable module to weigh and aggregate frame-level features under the guidance of a classification score. Extensive experiments on four prevalent benchmarks demonstrate that our method achieves state-of-the-art performance in video Re-ID.

----

## [676] Point-to-Spike Residual Learning for Energy-Efficient 3D Point Cloud Classification

**Authors**: *Qiaoyun Wu, Quanxiao Zhang, Chunyu Tan, Yun Zhou, Changyin Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28425](https://doi.org/10.1609/aaai.v38i6.28425)

**Abstract**:

Spiking neural networks (SNNs) have revolutionized neural learning and are making remarkable strides in image analysis and robot control tasks with ultra-low power consumption advantages. Inspired by this success, we investigate the application of spiking neural networks to 3D point cloud processing. We present a point-to-spike residual learning network for point cloud classification, which operates on points with binary spikes rather than floating-point numbers. Specifically, we first design a spatial-aware kernel point spiking neuron to relate spiking generation to point position in 3D space. On this basis, we then design a 3D spiking residual block for effective feature learning based on spike sequences. By stacking the 3D spiking residual blocks, we build the point-to-spike residual classification network, which achieves low computation cost and low accuracy loss on two benchmark datasets, ModelNet40 and ScanObjectNN. Moreover, the classifier strikes a good balance between classification accuracy and biological characteristics, allowing us to explore the deployment of 3D processing to neuromorphic chips for developing energy-efficient 3D robotic perception systems.

----

## [677] Segment beyond View: Handling Partially Missing Modality for Audio-Visual Semantic Segmentation

**Authors**: *Renjie Wu, Hu Wang, Feras Dayoub, Hsiang-Ting Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28426](https://doi.org/10.1609/aaai.v38i6.28426)

**Abstract**:

Augmented Reality (AR) devices, emerging as prominent mobile interaction platforms, face challenges in user safety, particularly concerning oncoming vehicles. While some solutions leverage onboard camera arrays, these cameras often have limited field-of-view (FoV) with front or downward perspectives. Addressing this, we propose a new out-of-view semantic segmentation task and Segment Beyond View (SBV), a novel audio-visual semantic segmentation method. SBV supplements the visual modality, which miss the information beyond FoV, with the auditory information using a teacher-student distillation model (Omni2Ego). The model consists of a vision teacher utilising panoramic information, an auditory teacher with 8-channel audio, and an audio-visual student that takes views with limited FoV and binaural audio as input and produce semantic segmentation for objects outside FoV. SBV outperforms existing models in comparative evaluations and shows a consistent performance across varying FoV ranges and in monaural audio settings.

----

## [678] Towards Transferable Adversarial Attacks with Centralized Perturbation

**Authors**: *Shangbo Wu, Yu-an Tan, Yajie Wang, Ruinan Ma, Wencong Ma, Yuanzhang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28427](https://doi.org/10.1609/aaai.v38i6.28427)

**Abstract**:

Adversarial transferability enables black-box attacks on unknown victim deep neural networks (DNNs), rendering attacks viable in real-world scenarios. Current transferable attacks create adversarial perturbation over the entire image, resulting in excessive noise that overfit the source model. Concentrating perturbation to dominant image regions that are model-agnostic is crucial to improving adversarial efficacy. However, limiting perturbation to local regions in the spatial domain proves inadequate in augmenting transferability. To this end, we propose a transferable adversarial attack with fine-grained perturbation optimization in the frequency domain, creating centralized perturbation. We devise a systematic pipeline to dynamically constrain perturbation optimization to dominant frequency coefficients. The constraint is optimized in parallel at each iteration, ensuring the directional alignment of perturbation optimization with model prediction. Our approach allows us to centralize perturbation towards sample-specific important frequency features, which are shared by DNNs, effectively mitigating source model overfitting. Experiments demonstrate that by dynamically centralizing perturbation on dominating frequency coefficients, crafted adversarial examples exhibit stronger transferability, and allowing them to bypass various defenses.

----

## [679] CLIM: Contrastive Language-Image Mosaic for Region Representation

**Authors**: *Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Wentao Liu, Chen Change Loy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28428](https://doi.org/10.1609/aaai.v38i6.28428)

**Abstract**:

Detecting objects accurately from a large or open vocabulary necessitates the vision-language alignment on region representations. However, learning such a region-text alignment by obtaining high-quality box annotations with text labels or descriptions is expensive and infeasible. In contrast, collecting image-text pairs is simpler but lacks precise object location information to associate regions with texts. In this paper, we propose a novel approach called Contrastive Language-Image Mosaic (CLIM), which leverages large-scale image-text pairs effectively for aligning region and text representations. CLIM combines multiple images into a mosaicked image and treats each image as a ‘pseudo region’. The feature of each pseudo region is extracted and trained to be similar to the corresponding text embedding while dissimilar from others by a contrastive loss, enabling the model to learn the region-text alignment without costly box annotations. As a generally
applicable approach, CLIM consistently improves different open-vocabulary object detection methods that use caption supervision. Furthermore, CLIM can effectively enhance the region representation of vision-language models, thus providing stronger backbones for open-vocabulary object detectors. Our experimental results demonstrate that CLIM improves different baseline open-vocabulary object detectors by a large margin on both OV-COCO and OV-LVIS benchmarks. The code is available at https://github.com/wusize/CLIM.

----

## [680] SphereDiffusion: Spherical Geometry-Aware Distortion Resilient Diffusion Model

**Authors**: *Tao Wu, Xuewei Li, Zhongang Qi, Di Hu, Xintao Wang, Ying Shan, Xi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28429](https://doi.org/10.1609/aaai.v38i6.28429)

**Abstract**:

Controllable spherical panoramic image generation holds substantial applicative potential across a variety of domains. However, it remains a challenging task due to the inherent spherical distortion and geometry characteristics, resulting in low-quality content generation. In this paper, we introduce a novel framework of SphereDiffusion to address these unique challenges, for better generating high-quality and precisely controllable spherical panoramic images. For the spherical distortion characteristic, we embed the semantics of the distorted object with text encoding, then explicitly construct the relationship with text-object correspondence to better use the pre-trained knowledge of the planar images. Meanwhile, we employ a deformable technique to mitigate the semantic deviation in latent space caused by spherical distortion. For the spherical geometry characteristic, in virtue of spherical rotation invariance, we improve the data diversity and optimization objectives in the training process, enabling the model to better learn the spherical geometry characteristic. Furthermore, we enhance the denoising process of the diffusion model, enabling it to effectively use the learned geometric characteristic to ensure the boundary continuity of the generated images. With these specific techniques, experiments on Structured3D dataset show that SphereDiffusion significantly improves the quality of controllable spherical image generation and relatively reduces around 35% FID on average.

----

## [681] LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate

**Authors**: *Tao Wu, Tie Luo, Donald C. Wunsch II*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28430](https://doi.org/10.1609/aaai.v38i6.28430)

**Abstract**:

The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking given pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose Lipschitz Regularized Surrogate (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

----

## [682] CR-SAM: Curvature Regularized Sharpness-Aware Minimization

**Authors**: *Tao Wu, Tie Luo, Donald C. Wunsch II*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28431](https://doi.org/10.1609/aaai.v38i6.28431)

**Abstract**:

The capacity to generalize to future unseen data stands as one of the utmost crucial attributes of deep neural networks. Sharpness-Aware Minimization (SAM) aims to enhance the generalizability by minimizing worst-case loss using one-step gradient ascent as an approximation. However, as training progresses, the non-linearity of the loss landscape increases, rendering one-step gradient ascent less effective. On the other hand, multi-step gradient ascent will incur higher training cost. In this paper, we introduce a normalized Hessian trace to accurately measure the curvature of loss landscape on both training and test sets. In particular, to counter excessive non-linearity of loss landscape, we propose Curvature Regularized SAM (CR-SAM), integrating the normalized Hessian trace as a SAM regularizer. Additionally, we present an efficient way to compute the trace via finite differences with parallelism. Our theoretical analysis based on PAC-Bayes bounds establishes the regularizer's efficacy in reducing generalization error. Empirical evaluation on CIFAR and ImageNet datasets shows that CR-SAM consistently enhances classification performance for ResNet and Vision Transformer (ViT) models across various datasets. Our code is available at https://github.com/TrustAIoT/CR-SAM.

----

## [683] Semi-supervised 3D Object Detection with PatchTeacher and PillarMix

**Authors**: *Xiaopei Wu, Liang Peng, Liang Xie, Yuenan Hou, Binbin Lin, Xiaoshui Huang, Haifeng Liu, Deng Cai, Wanli Ouyang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28432](https://doi.org/10.1609/aaai.v38i6.28432)

**Abstract**:

Semi-supervised learning aims to leverage numerous unlabeled data to improve the model performance. Current semi-supervised 3D object detection methods typically use a teacher to generate pseudo labels for a student, and the quality of the pseudo labels is essential for the final performance. In this paper, we propose PatchTeacher, which focuses on partial scene 3D object detection to provide high-quality pseudo labels for the student. Specifically, we divide a complete scene into a series of patches and feed them to our PatchTeacher sequentially. PatchTeacher leverages the low memory consumption advantage of partial scene detection to process point clouds with a high-resolution voxelization, which can minimize the information loss of quantization and extract more fine-grained features. However, it is non-trivial to train a detector on fractions of the scene. Therefore, we introduce three key techniques, i.e., Patch Normalizer, Quadrant Align, and Fovea Selection, to improve the performance of PatchTeacher. Moreover, we devise PillarMix, a strong data augmentation strategy that mixes truncated pillars from different LiDAR scans to generate diverse training samples and thus help the model learn more general representation. Extensive experiments conducted on Waymo and ONCE datasets verify the effectiveness and superiority of our method and we achieve new state-of-the-art results, surpassing existing methods by a large margin. Codes are available at https://github.com/LittlePey/PTPM.

----

## [684] Text-Based Occluded Person Re-identification via Multi-Granularity Contrastive Consistency Learning

**Authors**: *Xinyi Wu, Wentao Ma, Dan Guo, Tongqing Zhou, Shan Zhao, Zhiping Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28433](https://doi.org/10.1609/aaai.v38i6.28433)

**Abstract**:

Text-based Person Re-identification (T-ReID), which aims at retrieving a specific pedestrian image from a collection of images via text-based information, has received significant attention. However, previous research has overlooked a challenging yet practical form of T-ReID: dealing with image galleries mixed with occluded and inconsistent personal visuals, instead of ideal visuals with a full-body and clear view. Its major challenges lay in the insufficiency of benchmark datasets and the enlarged semantic gap incurred by arbitrary occlusions and modality gap between text description and visual representation of the target person. To alleviate these issues, we first design an Occlusion Generator (OGor) for the automatic generation of artificial occluded images from generic surveillance images. Then, a fine-granularity token selection mechanism is proposed to minimize the negative impact of occlusion for robust feature learning, and a novel multi-granularity contrastive consistency alignment framework is designed to leverage intra-/inter-granularity of visual-text representations for semantic alignment of occluded visuals and query texts. Experimental results demonstrate that our method exhibits superior performance. We believe this work could inspire the community to investigate more dedicated designs for implementing T-ReID in real-world scenarios. The source code is available at https://github.com/littlexinyi/MGCC.

----

## [685] CMG-Net: Robust Normal Estimation for Point Clouds via Chamfer Normal Distance and Multi-Scale Geometry

**Authors**: *Yingrui Wu, Mingyang Zhao, Keqiang Li, Weize Quan, Tianqi Yu, Jianfeng Yang, Xiaohong Jia, Dong-Ming Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28434](https://doi.org/10.1609/aaai.v38i6.28434)

**Abstract**:

This work presents an accurate and robust method for estimating normals from point clouds. In contrast to predecessor approaches that minimize the deviations between the annotated and the predicted normals directly, leading to direction inconsistency, we first propose a new metric termed Chamfer Normal Distance to address this issue. This not only mitigates the challenge but also facilitates network training and substantially enhances the network robustness against noise. Subsequently, we devise an innovative architecture that encompasses Multi-scale Local Feature Aggregation and Hierarchical Geometric Information Fusion. This design empowers the network to capture intricate geometric details more effectively and alleviate the ambiguity in scale selection. Extensive experiments demonstrate that our method achieves the state-of-the-art performance on both synthetic and real-world datasets, particularly in scenarios contaminated by noise. Our implementation is available at https://github.com/YingruiWoo/CMG-Net_Pytorch.

----

## [686] WaveFormer: Wavelet Transformer for Noise-Robust Video Inpainting

**Authors**: *Zhiliang Wu, Changchang Sun, Hanyu Xuan, Gaowen Liu, Yan Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28435](https://doi.org/10.1609/aaai.v38i6.28435)

**Abstract**:

Video inpainting aims to fill in the missing regions of the video frames with plausible content. Benefiting from the outstanding long-range modeling capacity, the transformer-based models have achieved unprecedented performance regarding inpainting quality. Essentially, coherent contents from all the frames along both spatial and temporal dimensions are concerned by a patch-wise attention module, and then the missing contents are generated based on the attention-weighted summation. In this way, attention retrieval accuracy has become the main bottleneck to improve the video inpainting performance, where the factors affecting attention calculation should be explored to maximize the advantages of transformer. Towards this end, in this paper, we theoretically certificate that noise is the culprit that entangles the process of attention calculation. Meanwhile, we propose a novel wavelet transformer network with noise robustness for video inpainting, named WaveFormer. Unlike existing transformer-based methods that utilize the whole embeddings to calculate the attention, our WaveFormer first separates the noise existing in the embedding into high-frequency components by introducing the Discrete Wavelet Transform (DWT), and then adopts clean low-frequency components to calculate the attention. In this way, the impact of noise on attention computation can be greatly mitigated and the missing content regarding different frequencies can be generated by sharing the calculated attention. Extensive experiments validate the superior performance of our method over state-of-the-art baselines both qualitatively and quantitatively.

----

## [687] FD3D: Exploiting Foreground Depth Map for Feature-Supervised Monocular 3D Object Detection

**Authors**: *Zizhang Wu, Yuanzhu Gan, Yunzhe Wu, Ruihao Wang, Xiaoquan Wang, Jian Pu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28436](https://doi.org/10.1609/aaai.v38i6.28436)

**Abstract**:

Monocular 3D object detection usually adopts direct or hierarchical label supervision. Recently, the distillation supervision transfers the spatial knowledge from LiDAR- or stereo-based teacher networks to monocular detectors, but remaining the domain gap. To mitigate this issue and pursue adequate label manipulation, we exploit Foreground Depth map for feature-supervised monocular 3D object detection named FD3D, which develops the high-quality instructive intermediate features to conduct desirable auxiliary feature supervision with only the original image and annotation foreground object-wise depth map (AFOD) as input. Furthermore, we build up our instructive feature generation network to create instructive spatial features based on the sufficient correlation between image features and pre-processed AFOD, where AFOD provides the attention focus only on foreground objects to achieve clearer guidance in the detection task. Moreover, we apply the auxiliary feature supervision from the pixel and distribution level to achieve comprehensive spatial knowledge guidance. Extensive experiments demonstrate that our method achieves state-of-the-art performance on both the KITTI and nuScenes datasets, with no external data and no extra inference computational cost. We also conduct quantitative and qualitative studies to reveal the effectiveness of our designs.

----

## [688] Attention Disturbance and Dual-Path Constraint Network for Occluded Person Re-identification

**Authors**: *Jiaer Xia, Lei Tan, Pingyang Dai, Mingbo Zhao, Yongjian Wu, Liujuan Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28437](https://doi.org/10.1609/aaai.v38i6.28437)

**Abstract**:

Occluded person re-identification (Re-ID) aims to address the potential occlusion problem when matching occluded or holistic pedestrians from different camera views. Many methods use the background as artificial occlusion  and rely on attention networks to exclude noisy interference. However, the significant discrepancy between simple background occlusion and realistic occlusion can negatively impact the generalization of the network. To address  this issue, we propose a novel transformer-based Attention Disturbance and Dual-Path Constraint Network (ADP) to  enhance the generalization of attention networks. Firstly, to  imitate real-world obstacles, we introduce an Attention Disturbance Mask (ADM) module that generates an offensive  noise, which can distract attention like a realistic occluder,  as a more complex form of occlusion. Secondly, to fully  exploit these complex occluded images, we develop a DualPath Constraint Module (DPC) that can obtain preferable  supervision information from holistic images through dualpath interaction. With our proposed method, the network  can effectively circumvent a wide variety of occlusions using the basic ViT baseline. Comprehensive experimental  evaluations conducted on person re-ID benchmarks demonstrate the superiority of ADP over state-of-the-art methods.

----

## [689] Locality Preserving Refinement for Shape Matching with Functional Maps

**Authors**: *Yifan Xia, Yifan Lu, Yuan Gao, Jiayi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28438](https://doi.org/10.1609/aaai.v38i6.28438)

**Abstract**:

In this paper, we address the nonrigid shape matching with outliers by a novel and effective pointwise map refinement method, termed Locality Preserving Refinement. For accurate pointwise conversion from a given functional map, our method formulates a two-step procedure. Firstly, starting with noisy point-to-point correspondences, we identify inliers by leveraging the neighborhood support, which yields a closed-form solution with linear time complexity. After obtained the reliable correspondences of inliers, we refine the pointwise correspondences for outliers using local linear embedding, which operates in an adaptive spectral similarity space to further eliminate the ambiguities that are difficult to handle in the functional space. By refining pointwise correspondences with local consistency thus embedding geometric constraints into functional spaces, our method achieves considerable improvement in accuracy with linearithmic time and space cost. Extensive experiments on public benchmarks demonstrate the superiority of our method over the state-of-the-art methods. Our code is publicly available at https://github.com/XiaYifan1999/LOPR.

----

## [690] SocialCVAE: Predicting Pedestrian Trajectory via Interaction Conditioned Latents

**Authors**: *Wei Xiang, Haoteng Yin, He Wang, Xiaogang Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28439](https://doi.org/10.1609/aaai.v38i6.28439)

**Abstract**:

Pedestrian trajectory prediction is the key technology in many applications for providing insights into human behavior and anticipating human future motions. Most existing empirical models are explicitly formulated by observed human behaviors using explicable mathematical terms with deterministic nature, while recent work has focused on developing hybrid models combined with learning-based techniques for powerful expressiveness while maintaining explainability. However, the deterministic nature of the learned steering behaviors from the empirical models limits the models' practical performance. To address this issue, this work proposes the social conditional variational autoencoder (SocialCVAE) for predicting pedestrian trajectories, which employs a CVAE to explore behavioral uncertainty in human motion decisions. SocialCVAE learns socially reasonable motion randomness by utilizing a socially explainable interaction energy map as the CVAE's condition, which illustrates the future occupancy of each pedestrian's local neighborhood area. The energy map is generated using an energy-based interaction model, which anticipates the energy cost (i.e., repulsion intensity) of pedestrians' interactions with neighbors. Experimental results on two public benchmarks including 25 scenes demonstrate that SocialCVAE significantly improves prediction accuracy compared with the state-of-the-art methods, with up to 16.85% improvement in Average Displacement Error (ADE) and 69.18% improvement in Final Displacement Error (FDE). Code is available at: https://github.com/ViviXiang/SocialCVAE.

----

## [691] Dynamic Semantic-Based Spatial Graph Convolution Network for Skeleton-Based Human Action Recognition

**Authors**: *Jianyang Xie, Yanda Meng, Yitian Zhao, Anh Nguyen, Xiaoyun Yang, Yalin Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28440](https://doi.org/10.1609/aaai.v38i6.28440)

**Abstract**:

Graph convolutional networks (GCNs) have attracted great attention and achieved remarkable performance in skeleton-based action recognition. However, most of the previous works are designed to refine skeleton topology without considering the types of different joints and edges, making them infeasible to represent the semantic information. In this paper, we proposed a dynamic semantic-based graph convolution network (DS-GCN) for skeleton-based human action recognition, where the joints and edge types were encoded in the skeleton topology in an implicit way. Specifically, two semantic modules, the joints type-aware adaptive topology and the edge type-aware adaptive topology, were proposed. Combining proposed semantics modules with temporal convolution, a powerful framework named DS-GCN was developed for skeleton-based action recognition. Extensive experiments in two datasets, NTU-RGB+D and Kinetics-400 show that the proposed semantic modules were generalized enough to be utilized in various backbones for boosting recognition accuracy. Meanwhile, the proposed DS-GCN notably outperformed state-of-the-art methods. The code is released here https://github.com/davelailai/DS-GCN

----

## [692] G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model

**Authors**: *Pan Xie, Qipeng Zhang, Taiying Peng, Hao Tang, Yao Du, Zexian Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28441](https://doi.org/10.1609/aaai.v38i6.28441)

**Abstract**:

The Sign Language Production (SLP) project aims to automatically translate spoken languages into sign sequences. Our approach focuses on the transformation of sign gloss sequences into their corresponding sign pose sequences (G2P). In this paper, we present a novel solution for this task by converting the continuous pose space generation problem into a discrete sequence generation problem. We introduce the Pose-VQVAE framework, which combines Variational Autoencoders (VAEs) with vector quantization to produce a discrete latent representation for continuous pose sequences. Additionally, we propose the G2P-DDM model, a discrete denoising diffusion architecture for length-varied discrete sequence data, to model the latent prior. To further enhance the quality of pose sequence generation in the discrete space, we present the CodeUnet model to leverage spatial-temporal information. Lastly, we develop a heuristic sequential clustering method to predict variable lengths of pose sequences for corresponding gloss sequences. Our results show that our model outperforms state-of-the-art G2P models on the public SLP evaluation benchmark. For more generated results, please visit our project page: https://slpdiffusier.github.io/g2p-ddm.

----

## [693] Towards Understanding Future: Consistency Guided Probabilistic Modeling for Action Anticipation

**Authors**: *Zhao Xie, Yadong Shi, Kewei Wu, Yaru Cheng, Dan Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28442](https://doi.org/10.1609/aaai.v38i6.28442)

**Abstract**:

Action anticipation aims to infer the action in the unobserved segment (future segment) with the observed segment (past segment). 
Existing methods focus on learning key past semantics to predict the future, but they do not model the temporal continuity between the past and the future. However, past actions are always highly uncertain in anticipating the unobserved future. 
The absence of temporal continuity smoothing in the video's past-and-future segments may result in an inconsistent anticipation of future action. 
In this work, we aim to smooth the global semantics changes in the past and future segments. We propose a Consistency-guided Probabilistic Model (CPM), which focuses on learning the globally temporal probabilistic consistency to inhibit the unexpected temporal consistency. 
The CPM is deployed on the Transformer architecture, which includes three modules of future semantics estimation, global semantics estimation, and global distribution estimation involving the learning of past-to-future semantics, past-and-future semantics, and semantically probabilistic distributions. 
To achieve the smoothness of temporal continuity, we follow the principle of variational analysis and describe two probabilistic distributions, i.e., a past-aware distribution and a global-aware distribution, which help to estimate the evidence lower bound of future anticipation. 
In this study, we maximize the evidence lower bound of future semantics by reducing the distribution distance between the above two distributions for model optimization. Extensive experiments demonstrate that the effectiveness of our method and the CPM achieves state-of-the-art performance on Epic-Kitchen100, Epic-Kitchen55, and EGTEA-GAZE.

----

## [694] Towards Detailed Text-to-Motion Synthesis via Basic-to-Advanced Hierarchical Diffusion Model

**Authors**: *Zhenyu Xie, Yang Wu, Xuehao Gao, Zhongqian Sun, Wei Yang, Xiaodan Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28443](https://doi.org/10.1609/aaai.v38i6.28443)

**Abstract**:

Text-guided motion synthesis aims to generate 3D human motion that not only precisely reflects the textual description but reveals the motion details as much as possible. Pioneering methods explore the diffusion model for text-to-motion synthesis and obtain significant superiority. However, these methods conduct diffusion processes either on the raw data distribution or the low-dimensional latent space, which typically suffer from the problem of modality inconsistency or detail-scarce. To tackle this problem, we propose a novel Basic-to-Advanced Hierarchical Diffusion Model, named B2A-HDM, to collaboratively exploit low-dimensional and  high-dimensional diffusion models for high quality detailed motion synthesis. Specifically, the basic diffusion model in low-dimensional latent space provides the intermediate denoising result that to be consistent with the textual description, while the advanced diffusion model in high-dimensional latent space focuses on the following detail-enhancing denoising process. Besides, we introduce a multi-denoiser framework for the advanced diffusion model to ease the learning of high-dimensional model and fully explore the generative potential of the diffusion model. Quantitative and qualitative experiment results on two text-to-motion benchmarks (HumanML3D and KIT-ML) demonstrate that B2A-HDM can outperform existing state-of-the-art methods in terms of fidelity, modality consistency, and diversity.

----

## [695] Learning by Erasing: Conditional Entropy Based Transferable Out-of-Distribution Detection

**Authors**: *Meng Xing, Zhiyong Feng, Yong Su, Changjae Oh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28444](https://doi.org/10.1609/aaai.v38i6.28444)

**Abstract**:

Detecting OOD inputs is crucial to deploy machine learning models to the real world safely. However, existing OOD detection methods require an in-distribution (ID) dataset to retrain the models. In this paper, we propose a Deep Generative Models (DGMs) based transferable OOD detection that does not require retraining on the new ID dataset. We first establish and substantiate two hypotheses on DGMs: DGMs exhibit a predisposition towards acquiring low-level features, in preference to semantic information; the lower bound of DGM's log-likelihoods is tied to the conditional entropy between the model input and target output. Drawing on the aforementioned hypotheses, we present an innovative image-erasing strategy, which is designed to create distinct conditional entropy distributions for each individual ID dataset. By training a DGM on a complex dataset with the proposed image-erasing strategy, the DGM could capture the discrepancy of conditional entropy distribution for varying ID datasets, without re-training. We validate the proposed method on the five datasets and show that, without retraining, our method achieves comparable performance to the state-of-the-art group-based OOD detection methods. The project codes will be open-sourced on our project website.

----

## [696] Unsupervised Action Segmentation via Fast Learning of Semantically Consistent Actoms

**Authors**: *Zheng Xing, Weibing Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28445](https://doi.org/10.1609/aaai.v38i6.28445)

**Abstract**:

Action segmentation serves as a pivotal component in comprehending videos, encompassing the learning of a sequence of semantically consistent action units known as actoms. Conventional methodologies tend to require a significant consumption of time for both training and learning phases. This paper introduces an innovative unsupervised framework for action segmentation in video, characterized by its fast learning capability and absence of mandatory training. The core idea involves splitting the video into distinct actoms, which are then merging together based on shared actions. The key challenge here is to prevent the inadvertent creation of singular actoms that attempt to represent multiple actions during the splitting phase. Additionally, it is crucial to avoid situations where actoms associated with the same action are incorrectly grouped into multiple clusters during the merging phase. In this paper, we present a method for calculating the similarity between adjacent frames under a subspace assumption. Then, we employ a local minimum searching procedure, which effectively splits the video into coherent actoms aligned with their semantic meaning and provides us an action segmentation proposal. Subsequently, we calculate a spatio-temporal similarity between actoms, followed by developing a merging process to merge actoms representing identical actions within the action segmentation proposals. Our approach is evaluated on four benchmark datasets, and the results demonstrate that our method achieves state-of-the-art performance. Besides, our method also achieves the optimal balance between accuracy and learning time when compared to existing unsupervised techniques. Code is available at https://github.com/y66y/SaM.

----

## [697] SPEAL: Skeletal Prior Embedded Attention Learning for Cross-Source Point Cloud Registration

**Authors**: *Kezheng Xiong, Maoji Zheng, Qingshan Xu, Chenglu Wen, Siqi Shen, Cheng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28446](https://doi.org/10.1609/aaai.v38i6.28446)

**Abstract**:

Point cloud registration, a fundamental task in 3D computer vision, has remained largely unexplored in cross-source point clouds and unstructured scenes. The primary challenges arise from noise, outliers, and variations in scale and density. However, neglected geometric natures of point clouds restricts the performance of current methods. In this paper, we propose a novel method termed SPEAL to leverage skeletal representations for effective learning of intrinsic topologies of point clouds, facilitating robust capture of geometric intricacy. Specifically, we design the Skeleton Extraction Module to extract skeleton points and skeletal features in an unsupervised manner, which is inherently robust to noise and density variances. Then, we propose the Skeleton-Aware GeoTransformer to encode high-level skeleton-aware features. It explicitly captures the topological natures and inter-point-cloud skeletal correlations with the noise-robust and density-invariant skeletal representations. Next, we introduce the Correspondence Dual-Sampler to facilitate correspondences by augmenting the correspondence set with skeletal correspondences. Furthermore, we construct a challenging novel cross-source point cloud dataset named KITTI CrossSource for benchmarking cross-source point cloud registration methods. Extensive quantitative and qualitative experiments are conducted to demonstrate our approach’s superiority and robustness on both cross-source and same-source datasets. To the best of our knowledge, our approach is the first to facilitate point cloud registration with skeletal geometric priors.

----

## [698] Patched Line Segment Learning for Vector Road Mapping

**Authors**: *Jiakun Xu, Bowen Xu, Gui-Song Xia, Liang Dong, Nan Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28447](https://doi.org/10.1609/aaai.v38i6.28447)

**Abstract**:

This paper presents a novel approach to computing vector road maps from satellite remotely sensed images, building upon a well-defined Patched Line Segment (PaLiS) representation for road graphs that holds geometric significance. Unlike prevailing methods that derive road vector representations from satellite images using binary masks or keypoints, our method employs line segments. These segments not only convey road locations but also capture their orientations, making them a robust choice for representation. More precisely, given an input image, we divide it into non-overlapping patches and predict a suitable line segment within each patch. This strategy enables us to capture spatial and structural cues from these patch-based line segments, simplifying the process of constructing the road network graph without the necessity of additional neural networks for connectivity. In our experiments, we demonstrate how an effective representation of a road graph significantly enhances the performance of vector road mapping on established benchmarks, without requiring extensive modifications to the neural network architecture. Furthermore, our method achieves state-of-the-art performance with just 6 GPU hours of training, leading to a substantial 32-fold reduction in training costs in terms of GPU hours.

----

## [699] MuLTI: Efficient Video-and-Language Understanding with Text-Guided MultiWay-Sampler and Multiple Choice Modeling

**Authors**: *Jiaqi Xu, Bo Liu, Yunkuo Chen, Mengli Cheng, Xing Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28448](https://doi.org/10.1609/aaai.v38i6.28448)

**Abstract**:

Video-and-language understanding has a variety of applications in the industry, such as video question answering, text-video retrieval, and multi-label classification. Existing video-and-language understanding methods generally adopt heavy multi-modal encoders and feature fusion modules, which consume high computational costs. Specially, they have difficulty dealing with dense video frames or long text prevalent in industrial applications. 
This paper proposes MuLTI, a highly accurate and efficient video-and-language understanding model that achieves efficient and effective feature fusion and rapid adaptation to downstream tasks. Specifically, we design a Text-Guided MultiWay-Sampler based on adapt-pooling residual mapping and self-attention modules to sample long sequences and fuse multi-modal features, which reduces the computational costs and addresses performance degradation caused by previous samplers. Therefore, MuLTI can handle longer sequences with limited computational costs. Then, to further enhance the model's performance and fill in the lack of pretraining tasks in the video question answering, we propose a new pretraining task named Multiple Choice Modeling. This task bridges the gap between pretraining and downstream tasks and improves the model's ability to align video and text features. Benefiting from the efficient feature fusion module and the new pretraining task, MuLTI achieves state-of-the-art performance on multiple datasets. Implementation and pretrained models will be released.

----

## [700] Regulating Intermediate 3D Features for Vision-Centric Autonomous Driving

**Authors**: *Junkai Xu, Liang Peng, Haoran Cheng, Linxuan Xia, Qi Zhou, Dan Deng, Wei Qian, Wenxiao Wang, Deng Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28449](https://doi.org/10.1609/aaai.v38i6.28449)

**Abstract**:

Multi-camera perception tasks have gained significant attention in the field of autonomous driving. However, existing frameworks based on Lift-Splat-Shoot (LSS) in the multi-camera setting cannot produce suitable dense 3D features due to the projection nature and uncontrollable densification process. To resolve this problem, we propose to regulate intermediate dense 3D features with the help of volume rendering. Specifically, we employ volume rendering to process the dense 3D features to obtain corresponding 2D features (e.g., depth maps, semantic maps), which are supervised by associated labels in the training. This manner regulates the generation of dense 3D features on the feature level, providing appropriate dense and unified features for multiple perception tasks. Therefore, our approach is termed Vampire, stands for ``Volume rendering As Multi-camera Perception Intermediate feature REgulator''. Experimental results on the Occ3D and nuScenes datasets demonstrate that Vampire facilitates fine-grained and appropriate extraction of dense 3D features, and is competitive with existing SOTA methods across diverse downstream perception tasks like 3D occupancy prediction, LiDAR segmentation and 3D objection detection, while utilizing moderate GPU resources. We provide a video demonstration in the supplementary materials and Codes are available at github.com/cskkxjk/Vampire.

----

## [701] ZOOM: Learning Video Mirror Detection with Extremely-Weak Supervision

**Authors**: *Ke Xu, Tsun Wai Siu, Rynson W. H. Lau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28450](https://doi.org/10.1609/aaai.v38i6.28450)

**Abstract**:

Mirror detection is an active research topic in computer vision. However, all existing mirror detectors learn mirror representations from large-scale pixel-wise datasets, which are tedious and expensive to obtain. Although weakly-supervised learning has been widely explored in related topics, we note that popular weak supervision signals (e.g., bounding boxes, scribbles, points) still require some efforts from the user to locate the target objects, with a strong assumption that the images to annotate always contain the target objects. Such an assumption may result in the over-segmentation of mirrors. Our key idea of this work is that the existence of mirrors over a time period may serve as a weak supervision to train a mirror detector, for two reasons. First, if a network can predict the existence of mirrors, it can essentially locate the mirrors. Second, we observe that the reflected contents of a mirror tend to be similar to those in adjacent frames, but exhibit considerable contrast to regions in far-away frames (e.g., non-mirror frames). To this end, in this paper, we propose ZOOM, the first method to learn robust mirror representations from extremely-weak annotations of per-frame ZerO-One Mirror indicators in videos. The key insight of ZOOM is to model the similarity and contrast (between mirror and non-mirror regions) in temporal variations to locate and segment the mirrors. To this end, we propose a novel fusion strategy to leverage temporal consistency information for mirror localization, and a novel temporal similarity-contrast modeling module for mirror segmentation. We construct a new video mirror dataset for training and evaluation. Experimental results under new and standard metrics show that ZOOM performs favorably against existing fully-supervised mirror detection methods.

----

## [702] Weakly Supervised Multimodal Affordance Grounding for Egocentric Images

**Authors**: *Lingjing Xu, Yang Gao, Wenfeng Song, Aimin Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28451](https://doi.org/10.1609/aaai.v38i6.28451)

**Abstract**:

To enhance the interaction between intelligent systems and the environment, locating the affordance regions of objects is crucial. These regions correspond to specific areas that provide distinct functionalities. Humans often acquire the ability to identify these regions through action demonstrations and verbal instructions. In this paper, we present a novel multimodal framework that extracts affordance knowledge from exocentric images, which depict human-object interactions, as well as from accompanying textual descriptions that describe the performed actions. The extracted knowledge is then transferred to egocentric images.
To achieve this goal, we propose the HOI-Transfer Module, which utilizes local perception to disentangle individual actions within exocentric images. This module effectively captures localized features and correlations between actions, leading to valuable affordance knowledge. Additionally, we introduce the Pixel-Text Fusion Module, which fuses affordance knowledge by identifying regions in egocentric images that bear resemblances to the textual features defining affordances.
We employ a Weakly Supervised Multimodal Affordance (WSMA) learning approach, utilizing image-level labels for training. Through extensive experiments, we demonstrate the superiority of our proposed method in terms of evaluation metrics and visual results when compared to existing affordance grounding models. Furthermore, ablation experiments confirm the effectiveness of our approach. Code:https://github.com/xulingjing88/WSMA.

----

## [703] Gaze from Origin: Learning for Generalized Gaze Estimation by Embedding the Gaze Frontalization Process

**Authors**: *Mingjie Xu, Feng Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28452](https://doi.org/10.1609/aaai.v38i6.28452)

**Abstract**:

Gaze estimation aims to accurately estimate the direction or position at which a person is looking. With the development of deep learning techniques, a number of gaze estimation methods have been proposed and achieved state-of-the-art performance. However, these methods are limited to within-dataset settings, whose performance drops when tested on unseen datasets. We argue that this is caused by infinite and continuous gaze labels. To alleviate this problem, we propose using gaze frontalization as an auxiliary task to constrain gaze estimation. Based on this, we propose a novel gaze domain generalization framework named Gaze Frontalization-based Auxiliary Learning (GFAL) Framework which embeds the gaze frontalization process, i.e., guiding the feature so that the eyeball can rotate and look at the front (camera), without any target domain information during training. Experimental results show that our proposed framework is able to achieve state-of-the-art performance on gaze domain generalization task, which is competitive with or even superior to the SOTA gaze unsupervised domain adaptation methods.

----

## [704] HACDR-Net: Heterogeneous-Aware Convolutional Network for Diabetic Retinopathy Multi-Lesion Segmentation

**Authors**: *QiHao Xu, Xiaoling Luo, Chao Huang, Chengliang Liu, Jie Wen, Jialei Wang, Yong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28453](https://doi.org/10.1609/aaai.v38i6.28453)

**Abstract**:

Diabetic Retinopathy (DR), the leading cause of blindness in diabetic patients, is diagnosed by the condition of retinal multiple lesions. As a difficult task in medical image segmentation, DR multi-lesion segmentation faces the main concerns as follows. On the one hand, retinal lesions vary in location, shape, and size. On the other hand, because some lesions occupy only a very small part of the entire fundus image, the high proportion of background leads to difficulties in lesion segmentation. To solve the above problems, we propose a heterogeneous-aware convolutional network (HACDR-Net) that composes heterogeneous cross-convolution, heterogeneous modulated deformable convolution, and optional near-far-aware convolution. Our network introduces an adaptive aggregation module to summarize the heterogeneous feature maps and get diverse lesion areas in the heterogeneous receptive field along the channels and space. In addition, to solve the problem of the highly imbalanced proportion of focal areas, we design a new medical image segmentation loss function, Noise Adjusted Loss (NALoss). NALoss balances the predictive feature distribution of background and lesion by jointing Gaussian noise and hard example mining, thus enhancing awareness of lesions. We conduct the experiments on the public datasets IDRiD and DDR, and the experimental results show that the proposed method achieves better performance than other state-of-the-art methods. The code is open-sourced on github.com/xqh180110910537/HACDR-Net.

----

## [705] Learning Invariant Inter-pixel Correlations for Superpixel Generation

**Authors**: *Sen Xu, Shikui Wei, Tao Ruan, Lixin Liao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28454](https://doi.org/10.1609/aaai.v38i6.28454)

**Abstract**:

Deep superpixel algorithms have made remarkable strides by substituting hand-crafted features with learnable ones. Nevertheless, we observe that existing deep superpixel methods, serving as mid-level representation operations, remain sensitive to the statistical properties (e.g., color distribution, high-level semantics) embedded within the training dataset. Consequently, learnable features exhibit constrained discriminative capability, resulting in unsatisfactory pixel grouping performance, particularly in untrainable application scenarios. To address this issue, we propose the Content Disentangle Superpixel (CDS) algorithm to selectively separate the invariant inter-pixel correlations and statistical properties, i.e., style noise. Specifically, We first construct auxiliary modalities that are homologous to the original RGB image but have substantial stylistic variations. Then, driven by mutual information, we propose the local-grid correlation alignment across modalities to reduce the distribution discrepancy of adaptively selected features and learn invariant inter-pixel correlations. Afterwards, we perform global-style mutual information minimization to enforce the separation of invariant content and train data styles. The experimental results on four benchmark datasets demonstrate the superiority of our approach to existing state-of-the-art methods, regarding boundary adherence, generalization, and efficiency. Code and pre-trained model are available at https://github.com/rookiie/CDSpixel.

----

## [706] Direction-Aware Video Demoiréing with Temporal-Guided Bilateral Learning

**Authors**: *Shuning Xu, Binbin Song, Xiangyu Chen, Jiantao Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28455](https://doi.org/10.1609/aaai.v38i6.28455)

**Abstract**:

Moiré patterns occur when capturing images or videos on screens, severely degrading the quality of the captured images or videos. Despite the recent progresses, existing video demoiréing methods neglect the physical characteristics and formation process of moiré patterns, significantly limiting the effectiveness of video recovery. This paper presents a unified framework, DTNet, a direction-aware and temporal-guided bilateral learning network for video demoiréing. DTNet effectively incorporates the process of moiré pattern removal, alignment, color correction, and detail refinement. Our proposed DTNet comprises two primary stages: Frame-level Direction-aware Demoiréing and Alignment (FDDA) and Tone and Detail Refinement (TDR). In FDDA, we employ multiple directional DCT modes to perform the moiré pattern removal process in the frequency domain, effectively detecting the prominent moiré edges. Then, the coarse and fine-grained alignment is applied on the demoiréd features for facilitating the utilization of neighboring information. In TDR, we propose a temporal-guided bilateral learning pipeline to mitigate the degradation of color and details caused by the moiré patterns while preserving the restored frequency information in FDDA. Guided by the aligned temporal features from FDDA, the affine transformations for the recovery of the ultimate clean frames are learned in TDR. Extensive experiments demonstrate that our video demoiréing method outperforms state-of-the-art approaches by 2.3 dB in PSNR, and also delivers a superior visual experience.

----

## [707] Spectral Prompt Tuning: Unveiling Unseen Classes for Zero-Shot Semantic Segmentation

**Authors**: *Wenhao Xu, Rongtao Xu, Changwei Wang, Shibiao Xu, Li Guo, Man Zhang, Xiaopeng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28456](https://doi.org/10.1609/aaai.v38i6.28456)

**Abstract**:

Recently, CLIP has found practical utility in the domain of pixel-level zero-shot segmentation tasks. 
The present landscape features two-stage methodologies beset by issues such as intricate pipelines and elevated computational costs. While current one-stage approaches alleviate these concerns and incorporate Visual Prompt Training (VPT) to uphold CLIP's generalization capacity, they still fall short in fully harnessing CLIP's potential for pixel-level unseen class demarcation and precise pixel predictions.
To further stimulate CLIP's zero-shot dense prediction capability, we propose SPT-SEG, a one-stage approach that improves CLIP's adaptability from image to pixel.
Specifically, we initially introduce Spectral Prompt Tuning (SPT), incorporating spectral prompts into the CLIP visual encoder's shallow layers to capture structural intricacies of images, thereby enhancing comprehension of unseen classes.
Subsequently, we introduce the Spectral Guided Decoder (SGD), utilizing both high and low-frequency information to steer the network's spatial focus towards more prominent classification features, enabling precise pixel-level prediction outcomes.
Through extensive experiments on two public datasets, we demonstrate the superiority of our method over state-of-the-art approaches, performing well across all classes and particularly excelling in handling unseen classes.

----

## [708] SCTNet: Single-Branch CNN with Transformer Semantic Information for Real-Time Segmentation

**Authors**: *Zhengze Xu, Dongyue Wu, Changqian Yu, Xiangxiang Chu, Nong Sang, Changxin Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28457](https://doi.org/10.1609/aaai.v38i6.28457)

**Abstract**:

Recent real-time semantic segmentation methods usually adopt an additional semantic branch to pursue rich long-range context. However, the additional branch incurs undesirable computational overhead and slows inference speed. To eliminate this dilemma, we propose SCTNet, a single branch CNN with transformer semantic information for real-time segmentation. SCTNet enjoys the rich semantic representations of an inference-free semantic branch while retaining the high efficiency of lightweight single branch CNN. SCTNet utilizes a transformer as the training-only semantic branch considering its superb ability to extract long-range context. With the help of the proposed transformer-like CNN block CFBlock and the semantic information alignment module, SCTNet could capture the rich semantic information from the transformer branch in training. During the inference, only the single branch CNN needs to be deployed. We conduct extensive experiments on Cityscapes, ADE20K, and COCO-Stuff-10K, and the results show that our method achieves the new state-of-the-art performance. The code and model is available at https://github.com/xzz777/SCTNet.

----

## [709] Chain of Generation: Multi-Modal Gesture Synthesis via Cascaded Conditional Control

**Authors**: *Zunnan Xu, Yachao Zhang, Sicheng Yang, Ronghui Li, Xiu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28458](https://doi.org/10.1609/aaai.v38i6.28458)

**Abstract**:

This study aims to improve the generation of 3D gestures by utilizing multimodal information from human speech. Previous studies have focused on incorporating additional modalities to enhance the quality of generated gestures. However, these methods perform poorly when certain modalities are missing during inference. To address this problem, we suggest using speech-derived multimodal priors to improve gesture generation. We introduce a novel method that separates priors from speech and employs multimodal priors as constraints for generating gestures. Our approach utilizes a chain-like modeling method to generate facial blendshapes, body movements, and hand gestures sequentially. Specifically, we incorporate rhythm cues derived from facial deformation and stylization prior based on speech emotions, into the process of generating gestures. By incorporating multimodal priors, our method improves the quality of generated gestures and eliminate the need for expensive setup preparation during inference. Extensive experiments and user studies confirm that our proposed approach achieves state-of-the-art performance.

----

## [710] Decoupled Contrastive Learning for Long-Tailed Recognition

**Authors**: *Shiyu Xuan, Shiliang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28459](https://doi.org/10.1609/aaai.v38i6.28459)

**Abstract**:

Supervised Contrastive Loss (SCL) is popular in visual representation learning.
    Given an anchor image, SCL pulls two types of positive samples, i.e., its augmentation and other images from the same class together, while pushes negative images apart to optimize the learned embedding. In the scenario of long-tailed recognition, where the number of samples in each class is imbalanced, treating two types of positive samples equally leads to the biased optimization for intra-category distance. In addition, similarity relationship among negative samples, that are ignored by SCL, also presents meaningful semantic cues. To improve the performance on long-tailed recognition, this paper addresses those two issues of SCL by decoupling the training objective. Specifically, it decouples two types of positives in SCL and optimizes their relations toward different objectives to alleviate the influence of the imbalanced dataset. We further propose a patch-based self distillation to transfer knowledge from head to tail classes to relieve the under-representation of tail classes. It uses patch-based features to mine shared visual patterns among different instances and leverages a self distillation procedure to transfer such knowledge. Experiments on different long-tailed classification benchmarks demonstrate the superiority of our method. For instance, it achieves the 57.7% top-1 accuracy on the ImageNet-LT dataset. Combined with the ensemble-based method, the performance can be further boosted to 59.7%, which substantially outperforms many recent works. Our code will be released.

----

## [711] Revisiting Gradient Pruning: A Dual Realization for Defending against Gradient Attacks

**Authors**: *Lulu Xue, Shengshan Hu, Ruizhi Zhao, Leo Yu Zhang, Shengqing Hu, Lichao Sun, Dezhong Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28460](https://doi.org/10.1609/aaai.v38i6.28460)

**Abstract**:

Collaborative learning (CL) is a distributed learning framework that aims to protect user privacy by allowing users to jointly train a model by sharing their gradient updates only. However, gradient inversion attacks (GIAs), which recover users' training data from shared gradients, impose severe privacy threats to CL.  Existing defense methods adopt different techniques, e.g., differential privacy, cryptography, and perturbation defenses, to defend against the GIAs. Nevertheless, all current defense methods suffer from a poor trade-off between privacy, utility, and efficiency. To mitigate the weaknesses of existing solutions, we propose a novel defense method, Dual Gradient Pruning (DGP), based on gradient pruning, which can improve communication efficiency while preserving the utility and privacy of CL. Specifically, DGP slightly changes gradient pruning with a stronger privacy guarantee. And DGP can also significantly improve communication efficiency with a theoretical analysis of its convergence and generalization. Our extensive experiments show that DGP can effectively defend against the most powerful GIAs and reduce the communication cost without sacrificing the model's utility.

----

## [712] A Convolutional Neural Network Interpretable Framework for Human Ventral Visual Pathway Representation

**Authors**: *Mufan Xue, Xinyu Wu, Jinlong Li, Xuesong Li, Guoyuan Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28461](https://doi.org/10.1609/aaai.v38i6.28461)

**Abstract**:

Recently, convolutional neural networks (CNNs) have become the best quantitative encoding models for capturing neural activity and hierarchical structure in the ventral visual pathway. However, the weak interpretability of these black-box models hinders their ability to reveal visual representational encoding mechanisms. Here, we propose a convolutional neural network interpretable framework (CNN-IF) aimed at providing a transparent interpretable encoding model for the ventral visual pathway. First, we adapt the feature-weighted receptive field framework to train two high-performing ventral visual pathway encoding models using large-scale functional Magnetic Resonance Imaging (fMRI) in both goal-driven and data-driven approaches. We find that network layer-wise predictions align with the functional hierarchy of the ventral visual pathway. Then, we correspond feature units to voxel units in the brain and successfully quantify the alignment between voxel responses and visual concepts. Finally, we conduct Network Dissection along the ventral visual pathway including the fusiform face area (FFA), and discover variations related to the visual concept of `person'. Our results demonstrate the CNN-IF provides a new perspective for understanding encoding mechanisms in the human ventral visual pathway, and the combination of ante-hoc interpretable structure and post-hoc interpretable approaches can achieve fine-grained voxel-wise correspondence between model and brain. The source code is available at: https://github.com/BIT-YangLab/CNN-IF.

----

## [713] Self-Supervised 3D Human Mesh Recovery from a Single Image with Uncertainty-Aware Learning

**Authors**: *Guoli Yan, Zichun Zhong, Jing Hua*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28462](https://doi.org/10.1609/aaai.v38i6.28462)

**Abstract**:

Despite achieving impressive improvement in accuracy, most existing monocular 3D human mesh reconstruction methods require large-scale 2D/3D ground-truths for supervision, which limits their applications on unlabeled in-the-wild data that is ubiquitous. To alleviate the reliance on 2D/3D ground-truths, we present a self-supervised 3D human pose and shape reconstruction framework that relies only on self-consistency between intermediate representations of images and projected 2D predictions. Specifically, we extract 2D joints and depth maps from monocular images as proxy inputs, which provides complementary clues to infer accurate 3D human meshes. Furthermore, to reduce the impacts from noisy and ambiguous inputs while better concentrate on the high-quality information, we design an uncertainty-aware module to automatically learn the reliability of the inputs at body-joint level based on the consistency between 2D joints and depth map. Experiments on benchmark datasets show that our approach outperforms other state-of-the-art methods at similar supervision levels.

----

## [714] HORIZON: High-Resolution Semantically Controlled Panorama Synthesis

**Authors**: *Kun Yan, Lei Ji, Chenfei Wu, Jian Liang, Ming Zhou, Nan Duan, Shuai Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28463](https://doi.org/10.1609/aaai.v38i6.28463)

**Abstract**:

Panorama synthesis endeavors to craft captivating 360-degree visual landscapes, immersing users in the heart of virtual worlds. Nevertheless, contemporary panoramic synthesis techniques grapple with the challenge of semantically guiding the content generation process. Although recent breakthroughs in visual synthesis have unlocked the potential for semantic control in 2D flat images, a direct application of these methods to panorama synthesis yields distorted content. In this study, we unveil an innovative framework for generating high-resolution panoramas, adeptly addressing the issues of spherical distortion and edge discontinuity through sophisticated spherical modeling. Our pioneering approach empowers users with semantic control, harnessing both image and text inputs, while concurrently streamlining the generation of high-resolution panoramas using parallel decoding. We rigorously evaluate our methodology on a diverse array of indoor and outdoor datasets, establishing its superiority over recent related work, in terms of both quantitative and qualitative performance metrics. Our research elevates the controllability, efficiency, and fidelity of panorama synthesis to new levels.

----

## [715] CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental Learning

**Authors**: *Qingsong Yan, Qiang Wang, Kaiyong Zhao, Jie Chen, Bo Li, Xiaowen Chu, Fei Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28464](https://doi.org/10.1609/aaai.v38i6.28464)

**Abstract**:

Neural Radiance Fields have demonstrated impressive performance in novel view synthesis. However, NeRF and most of its variants still rely on traditional complex pipelines to provide extrinsic and intrinsic camera parameters, such as COLMAP. Recent works, like NeRFmm, BARF, and L2G-NeRF, directly treat camera parameters as learnable and estimate them through differential volume rendering. However, these methods work for forward-looking scenes with slight motions and fail to tackle the rotation scenario in practice. To overcome this limitation, we propose a novel camera parameter free neural radiance field (CF-NeRF), which incrementally reconstructs 3D representations and recovers the camera parameters inspired by incremental structure from motion. Given a sequence of images, CF-NeRF estimates camera parameters of images one by one and reconstructs the scene through initialization, implicit localization, and implicit optimization. To evaluate our method, we use a challenging real-world dataset, NeRFBuster, which provides 12 scenes under complex trajectories. Results demonstrate that CF-NeRF is robust to rotation and achieves state-of-the-art results without providing prior information and constraints.

----

## [716] Referred by Multi-Modality: A Unified Temporal Transformer for Video Object Segmentation

**Authors**: *Shilin Yan, Renrui Zhang, Ziyu Guo, Wenchao Chen, Wei Zhang, Hongyang Li, Yu Qiao, Hao Dong, Zhongjiang He, Peng Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28465](https://doi.org/10.1609/aaai.v38i6.28465)

**Abstract**:

Recently, video object segmentation (VOS) referred by multi-modal signals, e.g., language and audio, has evoked increasing attention in both industry and academia. It is challenging for exploring the semantic alignment within modalities and the visual correspondence across frames.
However, existing methods adopt separate network architectures for different modalities, and neglect the inter-frame temporal interaction with references. In this paper, we propose MUTR, a Multi-modal Unified Temporal transformer for Referring video object segmentation. With a unified framework for the first time, MUTR adopts a DETR-style transformer and is capable of segmenting video objects designated by either text or audio reference. Specifically, we introduce two strategies to fully explore the temporal relations between videos and multi-modal signals. 
Firstly, for low-level temporal aggregation before the transformer, we enable the multi-modal references to capture multi-scale visual cues from consecutive video frames. This effectively endows the text or audio signals with temporal knowledge and boosts the semantic alignment between modalities.
Secondly, for high-level temporal interaction after the transformer, we conduct inter-frame feature communication for different object embeddings, contributing to better object-wise correspondence for tracking along the video.
On Ref-YouTube-VOS and AVSBench datasets with respective text and audio references, MUTR achieves +4.2% and +8.7% J&F improvements to state-of-the-art methods, demonstrating our significance for unified multi-modal VOS. Code is released at https://github.com/OpenGVLab/MUTR.

----

## [717] Embracing Language Inclusivity and Diversity in CLIP through Continual Language Learning

**Authors**: *Bang Yang, Yong Dai, Xuxin Cheng, Yaowei Li, Asif Raza, Yuexian Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28466](https://doi.org/10.1609/aaai.v38i6.28466)

**Abstract**:

While vision-language pre-trained models (VL-PTMs) have advanced multimodal research in recent years, their mastery in a few languages like English restricts their applicability in broader communities. To this end, there is an increasing interest in developing multilingual VL models via a joint-learning setup, which, however, could be unrealistic due to expensive costs and data availability. In this work, we propose to extend VL-PTMs' language capacity by continual language learning (CLL), where a model needs to update its linguistic knowledge incrementally without suffering from catastrophic forgetting (CF). We begin our study by introducing a model dubbed CLL-CLIP, which builds upon CLIP, a prevailing VL-PTM that has acquired image-English text alignment. Specifically, CLL-CLIP contains an expandable token embedding layer to handle linguistic differences. It solely trains token embeddings to improve memory stability and is optimized under cross-modal and cross-lingual objectives to learn the alignment between images and multilingual texts. To alleviate CF raised by covariate shift and lexical overlap, we further propose a novel approach that ensures the identical distribution of all token embeddings during initialization and regularizes token embedding learning during training. We construct a CLL benchmark covering 36 languages based on MSCOCO and XM3600 datasets and then evaluate multilingual image-text retrieval performance. Extensive experiments verify the effectiveness of CLL-CLIP and show that our approach can boost CLL-CLIP, e.g., by 6.7% in text-to-image average Recall@1 on XM3600, and improve various state-of-the-art methods consistently. Our code and data are available at https://github.com/yangbang18/CLFM.

----

## [718] Geometry-Guided Domain Generalization for Monocular 3D Object Detection

**Authors**: *Fan Yang, Hui Chen, Yuwei He, Sicheng Zhao, Chenghao Zhang, Kai Ni, Guiguang Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28467](https://doi.org/10.1609/aaai.v38i6.28467)

**Abstract**:

Monocular 3D object detection (M3OD) is important for autonomous driving. However, existing deep learning-based methods easily suffer from performance degradation in real-world scenarios due to the substantial domain gap between training and testing. M3OD's domain gaps are complex, including camera intrinsic parameters, extrinsic parameters, image appearance, etc. Existing works primarily focus on the domain gaps of camera intrinsic parameters, ignoring other key factors. Moreover, at the feature level, conventional domain invariant learning methods generally cause the negative transfer issue, due to the ignorance of dependency between geometry tasks and domains. To tackle these issues, in this paper, we propose MonoGDG, a geometry-guided domain generalization framework for M3OD, which effectively addresses the domain gap at both camera and feature levels. Specifically, MonoGDG consists of two major components. One is geometry-based image reprojection, which mitigates the impact of camera discrepancy by unifying intrinsic parameters, randomizing camera orientations, and unifying the field of view range. The other is geometry-dependent feature disentanglement, which overcomes the negative transfer problems by incorporating domain-shared and domain-specific features. Additionally, we leverage a depth-disentangled domain discriminator and a domain-aware geometry regression attention mechanism to account for the geometry-domain dependency. Extensive experiments on multiple autonomous driving benchmarks demonstrate that our method achieves state-of-the-art performance in domain generalization for M3OD.

----

## [719] Diversity-Authenticity Co-constrained Stylization for Federated Domain Generalization in Person Re-identification

**Authors**: *Fengxiang Yang, Zhun Zhong, Zhiming Luo, Yifan He, Shaozi Li, Nicu Sebe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28468](https://doi.org/10.1609/aaai.v38i6.28468)

**Abstract**:

This paper tackles the problem of federated domain generalization in person re-identification (FedDG re-ID), aiming to learn a model generalizable to unseen domains with decentralized source domains. Previous methods mainly focus on preventing local overfitting. However, the direction of diversifying local data through stylization for model training is largely overlooked. This direction is popular in domain generalization but will encounter two issues under federated scenario: (1) Most stylization methods require the centralization of multiple domains to generate novel styles but this is not applicable under decentralized constraint. (2) The authenticity of generated data cannot be ensured especially given limited local data, which may impair the model optimization. To solve these two problems, we propose the Diversity-Authenticity Co-constrained Stylization (DACS), which can generate diverse and authentic data for learning robust local model. Specifically, we deploy a style transformation model on each domain to generate novel data with two constraints: (1) A diversity constraint is designed to increase data diversity, which enlarges the Wasserstein distance between the original and transformed data; (2) An authenticity constraint is proposed to ensure data authenticity, which enforces the transformed data to be easily/hardly recognized by the local-side global/local model. Extensive experiments demonstrate the effectiveness of the proposed DACS and show that DACS achieves state-of-the-art performance for FedDG re-ID.

----

## [720] Semantic-Aware Transformation-Invariant RoI Align

**Authors**: *Guo-Ye Yang, George Kiyohiro Nakayama, Zi-Kai Xiao, Tai-Jiang Mu, Xiaolei Huang, Shi-Min Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28469](https://doi.org/10.1609/aaai.v38i6.28469)

**Abstract**:

Great progress has been made in learning-based object detection methods in the last decade. Two-stage detectors often have higher detection accuracy than one-stage detectors, due to the use of region of interest (RoI) feature extractors which extract transformation-invariant RoI features for different RoI proposals, making refinement of bounding boxes and prediction of object categories more robust and accurate. However, previous RoI feature extractors can only extract invariant features under limited transformations. In this paper, we propose a novel RoI feature extractor, termed Semantic RoI Align (SRA), which is capable of extracting invariant RoI features under a variety of transformations for two-stage detectors. Specifically, we propose a semantic attention module to adaptively determine different sampling areas by leveraging the global and local semantic relationship within the RoI. We also propose a Dynamic Feature Sampler which dynamically samples features based on the RoI aspect ratio to enhance the efficiency of SRA, and a new position embedding, i.e., Area Embedding, to provide more accurate position information for SRA through an improved sampling area representation. Experiments show that our model significantly outperforms baseline models with slight computational overhead. In addition, it shows excellent generalization ability and can be used to improve performance with various state-of-the-art backbones and detection methods. The code is available at https://github.com/cxjyxxme/SemanticRoIAlign.

----

## [721] FACL-Attack: Frequency-Aware Contrastive Learning for Transferable Adversarial Attacks

**Authors**: *Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i6.28470](https://doi.org/10.1609/aaai.v38i6.28470)

**Abstract**:

Deep neural networks are known to be vulnerable to security risks due to the inherent transferable nature of adversarial examples. Despite the success of recent generative model-based attacks demonstrating strong transferability, it still remains a challenge to design an efficient attack strategy in a real-world strict black-box setting, where both the target domain and model architectures are unknown. In this paper, we seek to explore a feature contrastive approach in the frequency domain to generate adversarial examples that are robust in both cross-domain and cross-model settings. With that goal in mind, we propose two modules that are only employed during the training phase: a Frequency-Aware Domain Randomization (FADR) module to randomize domain-variant low- and high-range frequency components and a Frequency-Augmented Contrastive Learning (FACL) module to effectively separate domain-invariant mid-frequency features of clean and perturbed image. We demonstrate strong transferability of our generated adversarial perturbations through extensive cross-domain and cross-model experiments, while keeping the inference time complexity.

----

## [722] Hybrid-SORT: Weak Cues Matter for Online Multi-Object Tracking

**Authors**: *Mingzhan Yang, Guangxin Han, Bin Yan, Wenhua Zhang, Jinqing Qi, Huchuan Lu, Dong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28471](https://doi.org/10.1609/aaai.v38i7.28471)

**Abstract**:

Multi-Object Tracking (MOT) aims to detect and associate all desired objects across frames. Most methods accomplish the task by explicitly or implicitly leveraging strong cues (i.e., spatial and appearance information), which exhibit powerful instance-level discrimination. However, when object occlusion and clustering occur, spatial and appearance information will become ambiguous simultaneously due to the high overlap among objects. In this paper, we demonstrate this long-standing challenge in MOT can be efficiently and effectively resolved by incorporating weak cues to compensate for strong cues. Along with velocity direction, we introduce the confidence and height state as potential weak cues. With superior performance, our method still maintains Simple, Online and Real-Time (SORT) characteristics. Also, our method shows strong generalization for diverse trackers and scenarios in a plug-and-play and training-free manner. Significant and consistent improvements are observed when applying our method to 5 different representative trackers. Further, with both strong and weak cues, our method Hybrid-SORT achieves superior performance on diverse benchmarks, including MOT17, MOT20, and especially DanceTrack where interaction and severe occlusion frequently happen with complex motions. The code and models are available at https://github.com/ymzis69/HybridSORT.

----

## [723] Multi-Modal Prompting for Open-Vocabulary Video Visual Relationship Detection

**Authors**: *Shuo Yang, Yongqi Wang, Xiaofeng Ji, Xinxiao Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28472](https://doi.org/10.1609/aaai.v38i7.28472)

**Abstract**:

Open-vocabulary video visual relationship detection aims to extend video visual relationship detection beyond annotated categories by detecting unseen relationships between objects in videos.  Recent progresses in open-vocabulary perception, primarily driven by large-scale image-text pre-trained models like CLIP, have shown remarkable success in recognizing novel objects and semantic categories.  However, directly applying CLIP-like models to video visual relationship detection encounters significant challenges due to the substantial gap between images and video object relationships. To address this challenge, we propose a multi-modal prompting method that adapts CLIP well to open-vocabulary video visual relationship detection by prompt-tuning on both visual representation and language input. Specifically, we enhance the image encoder of CLIP by using spatio-temporal visual prompting to capture spatio-temporal contexts, thereby making it suitable for object-level relationship representation in videos. Furthermore, we propose visual-guided language prompting to leverage CLIP's comprehensive semantic knowledge for discovering unseen relationship categories, thus facilitating recognizing novel video relationships. Extensive experiments on two public datasets,  VidVRD and  VidOR, demonstrate the effectiveness of our method, especially achieving a significant gain of nearly 10% in mAP on novel relationship categories on the VidVRD dataset.

----

## [724] Learning Dense Correspondence for NeRF-Based Face Reenactment

**Authors**: *Songlin Yang, Wei Wang, Yushi Lan, Xiangyu Fan, Bo Peng, Lei Yang, Jing Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28473](https://doi.org/10.1609/aaai.v38i7.28473)

**Abstract**:

Face reenactment is challenging due to the need to establish dense correspondence between various face representations for motion transfer. Recent studies have utilized Neural Radiance Field (NeRF) as fundamental representation, which further enhanced the performance of multi-view face reenactment in photo-realism and 3D consistency. However, establishing dense correspondence between different face NeRFs is non-trivial, because implicit representations lack ground-truth correspondence annotations like mesh-based 3D parametric models (e.g., 3DMM) with index-aligned vertexes. Although aligning 3DMM space with NeRF-based face representations can realize motion control, it is sub-optimal for their limited face-only modeling and low identity fidelity. Therefore, we are inspired to ask: Can we learn the dense correspondence between different NeRF-based face representations without a 3D parametric model prior? To address this challenge, we propose a novel framework, which adopts tri-planes as fundamental NeRF representation and decomposes face tri-planes into three components: canonical tri-planes, identity deformations, and motion. In terms of motion control, our key contribution is proposing a Plane Dictionary (PlaneDict) module, which efficiently maps the motion conditions to a linear weighted addition of learnable orthogonal plane bases. To the best of our knowledge, our framework is the first method that achieves one-shot multi-view face reenactment without a 3D parametric model prior. Extensive experiments demonstrate that we produce better results in fine-grained motion control and identity preservation than previous methods.

----

## [725] Motion Deblurring via Spatial-Temporal Collaboration of Frames and Events

**Authors**: *Wen Yang, Jinjian Wu, Jupo Ma, Leida Li, Guangming Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28474](https://doi.org/10.1609/aaai.v38i7.28474)

**Abstract**:

Motion deblurring can be advanced by exploiting informative features from supplementary sensors such as event cameras, which can capture rich motion information asynchronously with high temporal resolution. Existing event-based motion deblurring methods neither consider the modality redundancy in spatial fusion nor temporal cooperation between events and frames. To tackle these limitations, a novel spatial-temporal collaboration network (STCNet) is proposed for event-based motion deblurring. Firstly, we propose a differential-modality based cross-modal calibration strategy to suppress redundancy for complementarity enhancement, and then bimodal spatial fusion is achieved with an elaborate cross-modal co-attention mechanism to weight the contributions of them for importance balance. Besides, we present a frame-event mutual spatio-temporal attention scheme to alleviate the errors of relying only on frames to compute cross-temporal similarities when the motion blur is significant, and then the spatio-temporal features from both frames and events are aggregated with the custom cross-temporal coordinate attention. Extensive experiments on both synthetic and real-world datasets demonstrate that our method achieves state-of-the-art performance. Project website: https://github.com/wyang-vis/STCNet.

----

## [726] DGL: Dynamic Global-Local Prompt Tuning for Text-Video Retrieval

**Authors**: *Xiangpeng Yang, Linchao Zhu, Xiaohan Wang, Yi Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28475](https://doi.org/10.1609/aaai.v38i7.28475)

**Abstract**:

Text-video retrieval is a critical multi-modal task to find the most relevant video for a text query. Although pretrained models like CLIP have demonstrated impressive potential in this area, the rising cost of fully finetuning these models due to increasing model size continues to pose a problem. To address this challenge, prompt tuning has emerged as an alternative. However, existing works still face two problems when adapting pretrained image-text models to downstream video-text tasks: (1) The visual encoder could only encode frame-level features and failed to extract global-level general video information. (2) Equipping the visual and text encoder with separated prompts failed to mitigate the visual-text modality gap. To this end, we propose DGL, a cross-modal Dynamic prompt tuning method with Global-Local video attention. In contrast to previous prompt tuning methods, we employ the shared latent space to generate local-level text and frame prompts that encourage inter-modal interaction. Furthermore, we propose modeling video in a global-local attention mechanism to capture global video information from the perspective of prompt tuning. Extensive experiments reveal that when only 0.67% parameters are tuned, our cross-modal prompt tuning strategy DGL outperforms or is comparable to fully finetuning methods on MSR-VTT, VATEX, LSMDC, and ActivityNet datasets. Code will be available at https://github.com/knightyxp/DGL.

----

## [727] Diverse and Stable 2D Diffusion Guided Text to 3D Generation with Noise Recalibration

**Authors**: *Xiaofeng Yang, Fayao Liu, Yi Xu, Hanjing Su, Qingyao Wu, Guosheng Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28476](https://doi.org/10.1609/aaai.v38i7.28476)

**Abstract**:

In recent years, following the success of text guided image generation, text guided 3D generation has gained increasing attention among researchers. Dreamfusion is a notable approach that enhances generation quality by utilizing 2D text guided diffusion models and introducing SDS loss, a technique for distilling 2D diffusion model information to train 3D models. However, the SDS loss has two major limitations that hinder its effectiveness. Firstly, when given a text prompt, the SDS loss struggles to produce diverse content. Secondly, during training, SDS loss may cause the generated content to overfit and collapse, limiting the model's ability to learn intricate texture details. To overcome these challenges, we propose a novel approach called Noise Recalibration algorithm. By incorporating this technique, we can generate 3D content with significantly greater diversity and stunning details. Our approach offers a promising solution to the limitations of SDS loss.

----

## [728] Semantic Segmentation in Multiple Adverse Weather Conditions with Domain Knowledge Retention

**Authors**: *Xin Yang, Wending Yan, Yuan Yuan, Michael Bi Mi, Robby T. Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28477](https://doi.org/10.1609/aaai.v38i7.28477)

**Abstract**:

Semantic segmentation's performance is often compromised when applied to unlabeled adverse weather conditions. Unsupervised domain adaptation is a potential approach to enhancing the model's adaptability and robustness to adverse weather. However, existing methods encounter difficulties when sequentially adapting the model to multiple unlabeled adverse weather conditions. They struggle to acquire new knowledge while also retaining previously learned knowledge. To address these problems, we propose a semantic segmentation method for multiple adverse weather conditions that incorporates adaptive knowledge acquisition, pseudo-label blending, and weather composition replay. Our adaptive knowledge acquisition enables the model to avoid learning from extreme images that could potentially cause the model to forget. In our approach of blending pseudo-labels, we not only utilize the current model but also integrate the previously learned model into the ongoing learning process. This collaboration between the current teacher and the previous model enhances the robustness of the pseudo-labels for the current target. Our weather composition replay mechanism allows the model to continuously refine its previously learned weather information while simultaneously learning from the new target domain. Our method consistently outperforms the state-of-the-art methods, and obtains the best performance with averaged mIoU (%) of 65.7 and the lowest forgetting (%) of 3.6 against 60.1 and 11.3, on the ACDC datsets for a four-target continual multi-target domain adaptation.

----

## [729] Hyperspectral Image Reconstruction via Combinatorial Embedding of Cross-Channel Spatio-Spectral Clues

**Authors**: *Xingxing Yang, Jie Chen, Zaifeng Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28478](https://doi.org/10.1609/aaai.v38i7.28478)

**Abstract**:

Existing learning-based hyperspectral reconstruction methods show limitations in fully exploiting the information among the hyperspectral bands. As such, we propose to investigate the chromatic inter-dependencies in their respective hyperspectral embedding space. These embedded features can be fully exploited by querying the inter-channel correlations in a combinatorial manner, with the unique and complementary information efficiently fused into the final prediction. We found such independent modeling and combinatorial excavation mechanisms are extremely beneficial to uncover marginal spectral features, especially in the long wavelength bands. In addition, we have proposed a spatio-spectral attention block and a spectrum-fusion attention module, which greatly facilitates the excavation and fusion of information at both semantically long-range levels and fine-grained pixel levels across all dimensions. Extensive quantitative and qualitative experiments show that our method (dubbed CESST) achieves SOTA performance. Code for this project is at: https://github.com/AlexYangxx/CESST.

----

## [730] Decomposing Semantic Shifts for Composed Image Retrieval

**Authors**: *Xingyu Yang, Daqing Liu, Heng Zhang, Yong Luo, Chaoyue Wang, Jing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28479](https://doi.org/10.1609/aaai.v38i7.28479)

**Abstract**:

Composed image retrieval is a type of image retrieval task where the user provides a reference image as a starting point and specifies a text on how to shift from the starting point to the desired target image. However, most existing methods focus on the composition learning of text and reference images and oversimplify the text as a description, neglecting the inherent structure and the user's shifting intention of the texts. As a result, these methods typically take shortcuts that disregard the visual cue of the reference images. To address this issue, we reconsider the text as instructions and propose a Semantic Shift Network (SSN) that explicitly decomposes the semantic shifts into two steps: from the reference image to the visual prototype and from the visual prototype to the target image. Specifically, SSN explicitly decomposes the instructions into two components: degradation and upgradation, where the degradation is used to picture the visual prototype from the reference image, while the upgradation is used to enrich the visual prototype into the final representations to retrieve the desired target image. The experimental results show that the proposed SSN demonstrates a significant improvement of 5.42% and 1.37% on the CIRR and FashionIQ datasets, respectively, and establishes a new state-of-the-art performance. The code is available at https://github.com/starxing-yuu/SSN.

----

## [731] Gaze Target Detection by Merging Human Attention and Activity Cues

**Authors**: *Yaokun Yang, Yihan Yin, Feng Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28480](https://doi.org/10.1609/aaai.v38i7.28480)

**Abstract**:

Despite achieving impressive performance, current methods for detecting gaze targets, which depend on visual saliency and spatial scene geometry, continue to face challenges when it comes to detecting gaze targets within intricate image backgrounds. One of the primary reasons for this lies in the oversight of the intricate connection between human attention and activity cues. In this study, we introduce an innovative approach that amalgamates the visual saliency detection with the body-part & object interaction both guided by the soft gaze attention. This fusion enables precise and dependable detection of gaze targets amidst intricate image backgrounds. Our approach attains state-of-the-art performance on both the Gazefollow benchmark and the GazeVideoAttn benchmark. In comparison to recent methods that rely on intricate 3D reconstruction of a single input image, our approach, which solely leverages 2D image information, still exhibits a substantial lead across all evaluation metrics, positioning it closer to human-level performance. These outcomes underscore the potent effectiveness of our proposed method in the gaze target detection task.

----

## [732] PM-INR: Prior-Rich Multi-Modal Implicit Large-Scale Scene Neural Representation

**Authors**: *Yiying Yang, Fukun Yin, Wen Liu, Jiayuan Fan, Xin Chen, Gang Yu, Tao Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28481](https://doi.org/10.1609/aaai.v38i7.28481)

**Abstract**:

Recent advancements in implicit neural representations have contributed to high-fidelity surface reconstruction and photorealistic novel view synthesis. However, with the expansion of the scene scale, such as block or city level, existing methods
will encounter challenges because traditional sampling cannot cope with the cubically growing sampling space. To alleviate the dependence on filling the sampling space, we explore using multi-modal priors to assist individual points to
obtain more global semantic information and propose a priorrich multi-modal implicit neural representation network, Pm-INR, for the outdoor unbounded large-scale scene. The core of our method is multi-modal prior extraction and crossmodal prior fusion modules. The former encodes codebooks from different modality inputs and extracts valuable priors, while the latter fuses priors to maintain view consistency and preserve unique features among multi-modal priors. Finally, feature-rich cross-modal priors are injected into the sampling
regions to allow each region to perceive global information without filling the sampling space. Extensive experiments have demonstrated the effectiveness and robustness of our method for outdoor unbounded large-scale scene novel
view synthesis, which outperforms state-of-the-art methods in terms of PSNR, SSIM, and LPIPS.

----

## [733] FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning

**Authors**: *Zhenhua Yang, Dezhi Peng, Yuxin Kong, Yuyi Zhang, Cong Yao, Lianwen Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28482](https://doi.org/10.1609/aaai.v38i7.28482)

**Abstract**:

Automatic font generation is an imitation task, which aims to create a font library that mimics the style of reference images while preserving the content from source images. Although existing font generation methods have achieved satisfactory performance, they still struggle with complex characters and large style variations. To address these issues, we propose FontDiffuser, a diffusion-based image-to-image one-shot font generation method, which innovatively models the font imitation task as a noise-to-denoise paradigm. In our method, we introduce a Multi-scale Content Aggregation (MCA) block, which effectively combines global and local content cues across different scales, leading to enhanced preservation of intricate strokes of complex characters. Moreover, to better manage the large variations in style transfer, we propose a Style Contrastive Refinement (SCR) module, which is a novel structure for style representation learning. It utilizes a style extractor to disentangle styles from images, subsequently supervising the diffusion model via a meticulously designed style contrastive loss. Extensive experiments demonstrate FontDiffuser's state-of-the-art performance in generating diverse characters and styles. It consistently excels on complex characters and large style changes compared to previous methods. The code is available at https://github.com/yeungchenwa/FontDiffuser.

----

## [734] Full-Body Motion Reconstruction with Sparse Sensing from Graph Perspective

**Authors**: *Feiyu Yao, Zongkai Wu, Li Yi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28483](https://doi.org/10.1609/aaai.v38i7.28483)

**Abstract**:

Estimating 3D full-body pose from sparse sensor data is a pivotal technique employed for the reconstruction of realistic human motions in Augmented Reality and Virtual Reality. However, translating sparse sensor signals into comprehensive human motion remains a challenge since the sparsely distributed sensors in common VR systems fail to capture the motion of full human body. In this paper, we use well-designed Body Pose Graph (BPG) to represent the human body and translate the challenge into a prediction problem of graph missing nodes. Then, we propose a novel full-body motion reconstruction framework based on BPG. To establish BPG, nodes are initially endowed with features extracted from sparse sensor signals. Features from identifiable joint nodes across diverse sensors are amalgamated and processed from both temporal and spatial perspectives. Temporal dynamics are captured using the Temporal Pyramid Structure, while spatial relations in joint movements inform the spatial attributes. The resultant features serve as the foundational elements of the BPG nodes. To further refine the BPG, node features are updated through a graph neural network that incorporates edge reflecting varying joint relations. Our method's effectiveness is evidenced by the attained state-of-the-art performance, particularly in lower body motion, outperforming other baseline methods. Additionally, an ablation study validates the efficacy of each module in our proposed framework.

----

## [735] FoSp: Focus and Separation Network for Early Smoke Segmentation

**Authors**: *Lujian Yao, Haitao Zhao, Jingchao Peng, Zhongze Wang, Kaijie Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28484](https://doi.org/10.1609/aaai.v38i7.28484)

**Abstract**:

Early smoke segmentation (ESS) enables the accurate identification of smoke sources, facilitating the prompt extinguishing of fires and preventing large-scale gas leaks. But ESS poses greater challenges than conventional object and regular smoke segmentation due to its small scale and transparent appearance, which can result in high miss detection rate and low precision. To address these issues, a Focus and Separation Network (FoSp) is proposed. We first introduce a Focus module employing bidirectional cascade which guides low-resolution and high-resolution features towards mid-resolution to locate and determine the scope of smoke, reducing the miss detection rate. Next, we propose a Separation module that separates smoke images into a pure smoke foreground and a smoke-free background, enhancing the contrast between smoke and background fundamentally, improving segmentation precision. Finally, a Domain Fusion module is developed to integrate the distinctive features of the two modules which can balance recall and precision to achieve high F_beta. Futhermore, to promote the development of ESS, we introduce a high-quality real-world dataset called SmokeSeg, which contains more small and transparent smoke than the existing datasets. Experimental results show that our model achieves the best performance on three available smoke segmentation datasets: SYN70K (mIoU: 83.00%), SMOKE5K (F_beta: 81.6%) and SmokeSeg (F_beta: 72.05%). The code can be found at https://github.com/LujianYao/FoSp.

----

## [736] How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection

**Authors**: *Yiyang Yao, Peng Liu, Tiancheng Zhao, Qianqian Zhang, Jiajia Liao, Chunxin Fang, Kyusong Lee, Qing Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28485](https://doi.org/10.1609/aaai.v38i7.28485)

**Abstract**:

Object detection (OD) in computer vision has made significant progress in recent years, transitioning from closed-set labels to open-vocabulary detection (OVD) based on large-scale vision-language pre-training (VLP). However, current evaluation methods and datasets are limited to testing generalization over object types and referral expressions, which do not provide a systematic, fine-grained, and accurate benchmark of OVD models' abilities. In this paper, we propose a new benchmark named OVDEval, which includes 9 sub-tasks and introduces evaluations on commonsense knowledge, attribute understanding, position understanding, object relation comprehension, and more. The dataset is meticulously created to provide hard negatives that challenge models' true understanding of visual and linguistic input. Additionally, we identify a problem with the popular Average Precision (AP) metric when benchmarking models on these fine-grained label datasets and propose a new metric called Non-Maximum Suppression Average Precision (NMS-AP) to address this issue. Extensive experimental results show that existing top OVD models all fail on the new tasks except for simple object types, demonstrating the value of the proposed dataset in pinpointing the weakness of current OVD models and guiding future research. Furthermore, the proposed NMS-AP metric is verified by experiments to provide a much more truthful evaluation of OVD models, whereas traditional AP metrics yield deceptive results. Data is available at https://github.com/om-ai-lab/OVDEval

----

## [737] Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation

**Authors**: *Guy Yariv, Itai Gat, Sagie Benaim, Lior Wolf, Idan Schwartz, Yossi Adi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28486](https://doi.org/10.1609/aaai.v38i7.28486)

**Abstract**:

We consider the task of generating diverse and realistic videos guided by natural audio samples from a wide variety of semantic classes. For this task, the videos are required to be aligned both globally and temporally with the input audio: globally, the input audio is semantically associated with the entire output video, and temporally, each segment of the input audio is associated with a corresponding segment of that video. We utilize an existing text-conditioned video generation model and a pre-trained audio encoder model. The proposed method is based on a lightweight adaptor network, which learns to map the audio-based representation to the input representation expected by the text-to-video generation model. As such, it also enables video generation conditioned on text, audio, and, for the first time as far as we can ascertain, on both text and audio. We validate our method extensively on three datasets demonstrating significant semantic diversity of audio-video samples and further propose a novel evaluation metric (AV-Align) to assess the alignment of generated videos with input audio samples. AV-Align is based on the detection and comparison of energy peaks in both modalities. In comparison to recent state-of-the-art approaches, our method generates videos that are better aligned with the input sound, both with respect to content and temporal axis. We also show that videos produced by our method present higher visual quality and are more diverse. Code and samples are available at: https://pages.cs.huji.ac.il/adiyoss-lab/TempoTokens/.

----

## [738] AltDiffusion: A Multilingual Text-to-Image Diffusion Model

**Authors**: *Fulong Ye, Guang Liu, Xinya Wu, Ledell Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28487](https://doi.org/10.1609/aaai.v38i7.28487)

**Abstract**:

Large Text-to-Image(T2I) diffusion models have shown a remarkable capability to produce photorealistic and diverse images based on text inputs. However, existing works only support limited language input, e.g., English, Chinese, and Japanese, leaving users beyond these languages underserved and blocking the global expansion of T2I models. Therefore, this paper presents AltDiffusion, a novel multilingual T2I diffusion model that supports eighteen different languages. Specifically, we first train a multilingual text encoder based on the knowledge distillation. Then we plug it into a pretrained English-only diffusion model and train the model with a two-stage schema to enhance the multilingual capability, including concept alignment and quality improvement stage on a large-scale multilingual dataset. Furthermore, we introduce a new benchmark, which includes Multilingual-General-18(MG-18) and Multilingual-Cultural-18(MC-18) datasets, to evaluate the capabilities of T2I diffusion models for generating high-quality images and capturing culture-specific concepts in different languages. Experimental results on both MG-18 and MC-18 demonstrate that AltDiffusion outperforms current state-of-the-art T2I models, e.g., Stable Diffusion in multilingual understanding, especially with respect to culture-specific concepts, while still having comparable capability for generating high-quality images. All source code and checkpoints could be found in https://github.com/superhero-7/AltDiffuson.

----

## [739] Mutual-Modality Adversarial Attack with Semantic Perturbation

**Authors**: *Jingwen Ye, Ruonan Yu, Songhua Liu, Xinchao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28488](https://doi.org/10.1609/aaai.v38i7.28488)

**Abstract**:

Adversarial attacks constitute a notable threat to machine learning systems, given their potential to induce erroneous predictions and classifications. However, within real-world contexts, the essential specifics of the deployed model are frequently treated as a black box, consequently mitigating the vulnerability to such attacks.
Thus, enhancing the transferability of the adversarial samples has become a crucial area of research, which heavily relies on selecting appropriate surrogate models.
To address this challenge, we propose a novel approach that generates adversarial attacks in a mutual-modality optimization scheme. Our approach is accomplished by leveraging the pre-trained CLIP model. Firstly, we conduct a visual attack on the clean image that causes semantic perturbations on the aligned embedding space with the other textual modality. 
Then, we apply the corresponding defense on the textual modality by updating the prompts, which forces the re-matching on the perturbed embedding space. 
Finally, to enhance the attack transferability, we utilize the iterative training strategy on the visual attack and the textual defense, where the two processes optimize from each other.
We evaluate our approach on several benchmark datasets and demonstrate that our mutual-modal attack strategy can effectively produce high-transferable attacks, which are stable regardless of the target networks. Our approach outperforms state-of-the-art attack methods and can be readily deployed as a plug-and-play solution.

----

## [740] STDiff: Spatio-Temporal Diffusion for Continuous Stochastic Video Prediction

**Authors**: *Xi Ye, Guillaume-Alexandre Bilodeau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28489](https://doi.org/10.1609/aaai.v38i7.28489)

**Abstract**:

Predicting future frames of a video is challenging because it is difficult to learn the uncertainty of the underlying factors influencing their contents. In this paper, we propose a novel video prediction model, which has infinite-dimensional latent variables over the spatio-temporal domain. Specifically, we first decompose the video motion and content information, then take a neural stochastic differential equation to predict the temporal motion information, and finally, an image diffusion model autoregressively generates the video frame by conditioning on the predicted motion feature and the previous frame. The better expressiveness and stronger stochasticity learning capability of our model lead to state-of-the-art video prediction performances. As well, our model is able to achieve temporal continuous prediction, i.e., predicting in an unsupervised way the future video frames with an arbitrarily high frame rate. Our code is available at https://github.com/XiYe20/STDiffProject.

----

## [741] DiffusionEdge: Diffusion Probabilistic Model for Crisp Edge Detection

**Authors**: *Yunfan Ye, Kai Xu, Yuhang Huang, Renjiao Yi, Zhiping Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28490](https://doi.org/10.1609/aaai.v38i7.28490)

**Abstract**:

Limited by the encoder-decoder architecture, learning-based edge detectors usually have difficulty predicting edge maps that satisfy both correctness and crispness. With the recent success of the diffusion probabilistic model (DPM), we found it is especially suitable for accurate and crisp edge detection since the denoising process is directly applied to the original image size. Therefore, we propose the first diffusion model for the task of general edge detection, which we call DiffusionEdge. To avoid expensive computational resources while retaining the final performance, we apply DPM in the latent space and enable the classic cross-entropy loss which is uncertainty-aware in pixel level to directly optimize the parameters in latent space in a distillation manner.  We also adopt a decoupled architecture to speed up the denoising process and propose a corresponding adaptive Fourier filter to adjust the latent features of specific frequencies. With all the technical designs, DiffusionEdge can be stably trained with limited resources, predicting crisp and accurate edge maps with much fewer augmentation strategies. Extensive experiments on four edge detection benchmarks demonstrate the superiority of DiffusionEdge both in correctness and crispness. On the NYUDv2 dataset, compared to the second best, we increase the ODS, OIS (without post-processing) and AC by 30.2%, 28.1% and 65.1%, respectively. Code: https://github.com/GuHuangAI/DiffusionEdge.

----

## [742] Dynamic Feature Pruning and Consolidation for Occluded Person Re-identification

**Authors**: *Yuteng Ye, Hang Zhou, Jiale Cai, Chenxing Gao, Youjia Zhang, Junle Wang, Qiang Hu, Junqing Yu, Wei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28491](https://doi.org/10.1609/aaai.v38i7.28491)

**Abstract**:

Occluded person re-identification (ReID) is a challenging problem due to contamination from occluders. Existing approaches address the issue with prior knowledge cues, such as human body key points and semantic segmentations, which easily fail in the presence of heavy occlusion and other humans as occluders. In this paper, we propose a feature pruning and consolidation (FPC) framework to circumvent explicit human structure parsing. The framework mainly consists of a sparse encoder, a multi-view feature mathcing module, and a feature consolidation decoder. Specifically, the sparse encoder drops less important image tokens, mostly related to background noise and occluders, solely based on correlation within the class token attention. Subsequently, the matching stage relies on the preserved tokens produced by the sparse encoder to identify k-nearest neighbors in the gallery by measuring the image and patch-level combined similarity. Finally, we use the feature consolidation module to compensate pruned features using identified neighbors for recovering essential information while disregarding disturbance from noise and occlusion. Experimental results demonstrate the effectiveness of our proposed framework on occluded, partial, and holistic Re-ID datasets. In particular, our method outperforms state-of-the-art results by at least 8.6% mAP and 6.0% Rank-1 accuracy on the challenging Occluded-Duke dataset.

----

## [743] Progressive Text-to-Image Diffusion with Soft Latent Direction

**Authors**: *Yuteng Ye, Jiale Cai, Hang Zhou, Guanwen Li, Youjia Zhang, Zikai Song, Chenxing Gao, Junqing Yu, Wei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28492](https://doi.org/10.1609/aaai.v38i7.28492)

**Abstract**:

In spite of the rapidly evolving landscape of text-to-image generation, the synthesis and manipulation of multiple entities while adhering to specific relational constraints pose enduring challenges. This paper introduces an innovative progressive synthesis and editing operation that systematically incorporates entities into the target image, ensuring their adherence to spatial and relational constraints at each sequential step. Our key insight stems from the observation that while a pre-trained text-to-image diffusion model adeptly handles one or two entities, it often falters when dealing with a greater number. To address this limitation, we propose harnessing the capabilities of a Large Language Model (LLM) to decompose intricate and protracted text descriptions into coherent directives adhering to stringent formats. To facilitate the execution of directives involving distinct semantic operations—namely insertion, editing, and erasing—we formulate the Stimulus, Response, and Fusion (SRF) framework. Within this framework, latent regions are gently stimulated in alignment with each operation, followed by the fusion of the responsive latent components to achieve cohesive entity manipulation. Our proposed framework yields notable advancements in object synthesis, particularly when confronted with intricate and lengthy textual inputs. Consequently, it establishes a new benchmark for text-to-image generation tasks, further elevating the field's performance standards.

----

## [744] UCMCTrack: Multi-Object Tracking with Uniform Camera Motion Compensation

**Authors**: *Kefu Yi, Kai Luo, Xiaolei Luo, Jiangui Huang, Hao Wu, Rongdong Hu, Wei Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28493](https://doi.org/10.1609/aaai.v38i7.28493)

**Abstract**:

Multi-object tracking (MOT) in video sequences remains a challenging task, especially in scenarios with significant camera movements. This is because targets can drift considerably on the image plane, leading to erroneous tracking outcomes. Addressing such challenges typically requires supplementary appearance cues or Camera Motion Compensation (CMC). While these strategies are effective, they also introduce a considerable computational burden, posing challenges for real-time MOT. In response to this, we introduce UCMCTrack, a novel motion model-based tracker robust to camera movements. Unlike conventional CMC that computes compensation parameters frame-by-frame, UCMCTrack consistently applies the same compensation parameters throughout a video sequence. It employs a Kalman filter on the ground plane and introduces the Mapped Mahalanobis Distance (MMD) as an alternative to the traditional Intersection over Union (IoU) distance measure. By leveraging projected probability distributions on the ground plane, our approach efficiently captures motion patterns and adeptly manages uncertainties introduced by homography projections. Remarkably, UCMCTrack, relying solely on motion cues, achieves state-of-the-art performance across a variety of challenging datasets, including MOT17, MOT20, DanceTrack and KITTI. More details and code are available at https://github.com/corfyi/UCMCTrack.

----

## [745] DiffRAW: Leveraging Diffusion Model to Generate DSLR-Comparable Perceptual Quality sRGB from Smartphone RAW Images

**Authors**: *Mingxin Yi, Kai Zhang, Pei Liu, Tanli Zuo, Jingduo Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28494](https://doi.org/10.1609/aaai.v38i7.28494)

**Abstract**:

Deriving DSLR-quality sRGB images from smartphone RAW images has become a compelling challenge due to discernible detail disparity, color mapping instability, and spatial misalignment in RAW-sRGB data pairs. We present DiffRAW, a novel method that incorporates the diffusion model for the first time in learning RAW-to-sRGB mappings. By leveraging the diffusion model, our approach effectively learns the high-quality detail distribution of DSLR images, thereby enhancing the details of output images. Simultaneously, we use the RAW image as a diffusion condition to maintain image structure information such as contours and textures. To mitigate the interference caused by the color and spatial misalignment in training data pairs, we embed a color-position preserving condition within DiffRAW, ensuring that the output images do not exhibit color biases and pixel shift issues. To accelerate the inference process of DiffRAW, we designed the Domain Transform Diffusion Method, an efficient diffusion process with its corresponding reverse process. The Domain Transform Diffusion Method can reduce the required inference steps for diffusion model-based image restoration/enhancement algorithms while enhancing the quality of the generated images. Through evaluations on the ZRR dataset, DiffRAW consistently demonstrates state-of-the-art performance across all perceptual quality metrics (e.g., LPIPS, FID, MUSIQ), while achieving comparable results in PSNR and SSIM.

----

## [746] Efficient Look-Up Table from Expanded Convolutional Network for Accelerating Image Super-resolution

**Authors**: *Kai Yin, Jie Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28495](https://doi.org/10.1609/aaai.v38i7.28495)

**Abstract**:

The look-up table (LUT) has recently shown its practicability and effectiveness in super-resolution (SR) tasks due to its low computational cost and hardware independence. However, most existing methods focus on improving the performance of SR, neglecting the demand for high-speed SR on low-computational edge devices. In this paper, we propose an efficient expanded convolution (EC) layer, which expands the output size of regular convolution to enlarge the receptive field (RF) indirectly. It can increase the size of the LUT corresponding to the network linearly with the increase of RF. Additionally, after introducing the EC, multiple LUTs are merged into one LUT, achieving faster running speed while maintaining SR performance. More specifically, we expand the coverage of the convolutional output so that the output at the current position covers the target position and its surroundings, forming an overlapping sliding window at the output end. We sum up the overlapping parts of the sliding window as the output, thereby achieving the effect of enlarging the RF size. Moreover, by expanding the numerical range of the accumulated results and rescaling them to [0,255], the method can mitigate the error caused by quantization output. Experiments indicate that the proposed method performs better than the baseline method and is faster than other LUT-based SR methods.

----

## [747] CLIP-Gaze: Towards General Gaze Estimation via Visual-Linguistic Model

**Authors**: *Pengwei Yin, Guanzhong Zeng, Jingjing Wang, Di Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28496](https://doi.org/10.1609/aaai.v38i7.28496)

**Abstract**:

Gaze estimation methods often experience significant performance degradation when evaluated across different domains, due to the domain gap between the testing and training data. Existing methods try to address this issue using various domain generalization approaches, but with little success because of the limited diversity of gaze datasets, such as appearance, wearable, and image quality. To overcome these limitations, we propose a novel framework called CLIP-Gaze that utilizes a pre-trained vision-language model to leverage its transferable knowledge. Our framework is the first to leverage the vision-and-language cross-modality approach for gaze estimation task. Specifically, we extract gaze-relevant feature by pushing it away from gaze-irrelevant features which can be flexibly constructed via language descriptions. To learn more suitable prompts, we propose a personalized context optimization method for text prompt tuning. Furthermore, we utilize the relationship among gaze samples to refine the distribution of gaze-relevant features, thereby improving the generalization capability of the gaze estimation model. Extensive experiments demonstrate the excellent performance of CLIP-Gaze over existing methods on four cross-domain evaluations.

----

## [748] Point Deformable Network with Enhanced Normal Embedding for Point Cloud Analysis

**Authors**: *Xingyilang Yin, Xi Yang, Liangchen Liu, Nannan Wang, Xinbo Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28497](https://doi.org/10.1609/aaai.v38i7.28497)

**Abstract**:

Recently MLP-based methods have shown strong performance in point cloud analysis. Simple MLP architectures are able to learn geometric features in local point groups yet fail to model long-range dependencies directly. In this paper, we propose Point Deformable Network (PDNet), a concise MLP-based network that can capture long-range relations with strong representation ability. Specifically, we put forward Point Deformable Aggregation Module (PDAM) to improve representation capability in both long-range dependency and adaptive aggregation among points. For each query point, PDAM aggregates information from deformable reference points rather than points in limited local areas. The deformable reference points are generated data-dependent, and we initialize them according to the input point positions. Additional offsets and modulation scalars are learned on the whole point features, which shift the deformable reference points to the regions of interest. We also suggest estimating the normal vector for point clouds and applying Enhanced Normal Embedding (ENE) to the geometric extractors to improve the representation ability of single-point. Extensive experiments and ablation studies on various benchmarks demonstrate the effectiveness and superiority of our PDNet.

----

## [749] Revisiting Open-Set Panoptic Segmentation

**Authors**: *Yufei Yin, Hao Chen, Wengang Zhou, Jiajun Deng, Haiming Xu, Houqiang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28498](https://doi.org/10.1609/aaai.v38i7.28498)

**Abstract**:

In this paper, we focus on the open-set panoptic segmentation (OPS) task to circumvent the data explosion problem. Different from the close-set setting, OPS targets to detect both known and unknown categories, where the latter is not annotated during training. Different from existing work that only selects a few common categories as unknown ones, we move forward to the real-world scenario by considering the various tail categories (~1k). To this end, we first build a new dataset with long-tail distribution for the OPS task. Based on this dataset, we additionally add a new class type for unknown classes and re-define the training annotations to make the OPS definition more complete and reasonable. Moreover, we analyze the influence of several significant factors in the OPS task and explore the upper bound of performance on unknown classes with different settings. Furthermore, based on the analyses, we design an effective two-phase framework for the OPS task, including thing-agnostic map generation and unknown segment mining. We further adopt semi-supervised learning to improve the OPS performance. Experimental results on different datasets validate the effectiveness of our method.

----

## [750] VQAttack: Transferable Adversarial Attacks on Visual Question Answering via Pre-trained Models

**Authors**: *Ziyi Yin, Muchao Ye, Tianrong Zhang, Jiaqi Wang, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28499](https://doi.org/10.1609/aaai.v38i7.28499)

**Abstract**:

Visual Question Answering (VQA) is a fundamental task in computer vision and natural language process fields. Although the “pre-training & finetuning” learning paradigm significantly improves the VQA performance, the adversarial robustness of such a learning paradigm has not been explored. In this paper, we delve into a new problem: using a pre-trained multimodal source model to create adversarial image-text pairs and then transferring them to attack the target VQA models. Correspondingly, we propose a novel VQATTACK model, which can iteratively generate both im- age and text perturbations with the designed modules: the large language model (LLM)-enhanced image attack and the cross-modal joint attack module. At each iteration, the LLM-enhanced image attack module first optimizes the latent representation-based loss to generate feature-level image perturbations. Then it incorporates an LLM to further enhance the image perturbations by optimizing the designed masked answer anti-recovery loss. The cross-modal joint attack module will be triggered at a specific iteration, which updates the image and text perturbations sequentially. Notably, the text perturbation updates are based on both the learned gradients in the word embedding space and word synonym-based substitution. Experimental results on two VQA datasets with five validated models demonstrate the effectiveness of the proposed VQATTACK in the transferable attack setting, compared with state-of-the-art baselines. This work reveals
a significant blind spot in the “pre-training & fine-tuning” paradigm on VQA tasks. The source code can be found in the link https://github.com/ericyinyzy/VQAttack.

----

## [751] TF-CLIP: Learning Text-Free CLIP for Video-Based Person Re-identification

**Authors**: *Chenyang Yu, Xuehu Liu, Yingquan Wang, Pingping Zhang, Huchuan Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28500](https://doi.org/10.1609/aaai.v38i7.28500)

**Abstract**:

Large-scale language-image pre-trained models (e.g., CLIP) have shown superior performances on many cross-modal retrieval tasks.
However, the problem of transferring the knowledge learned from such models to video-based person re-identification (ReID) has barely been explored.
In addition, there is a lack of decent text descriptions in current ReID benchmarks.
To address these issues, in this work, we propose a novel one-stage text-free CLIP-based learning framework named TF-CLIP for video-based person ReID.
More specifically, we extract the identity-specific sequence feature as the CLIP-Memory to replace the text feature.
Meanwhile, we design a Sequence-Specific Prompt (SSP) module to update the CLIP-Memory online.
To capture temporal information, we further propose a Temporal Memory Diffusion (TMD) module, which consists of two key components: Temporal Memory Construction (TMC) and Memory Diffusion (MD).
Technically, TMC allows the frame-level memories in a sequence to communicate with each other, and to extract temporal information based on the relations within the sequence.
MD further diffuses the temporal memories to each token in the original features to obtain more robust sequence features.
Extensive experiments demonstrate that our proposed method shows much better results than other state-of-the-art methods on MARS, LS-VID and iLIDS-VID.

----

## [752] MM-Point: Multi-View Information-Enhanced Multi-Modal Self-Supervised 3D Point Cloud Understanding

**Authors**: *Hai-Tao Yu, Mofei Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28501](https://doi.org/10.1609/aaai.v38i7.28501)

**Abstract**:

In perception, multiple sensory information is integrated to map visual information from 2D views onto 3D objects, which is beneficial for understanding in 3D environments. But in terms of a single 2D view rendered from different angles, only limited partial information can be provided. The richness and value of Multi-view 2D information can provide superior self-supervised signals for 3D objects. In this paper, we propose a novel self-supervised point cloud representation learning method, MM-Point, which is driven by intra-modal and inter-modal similarity objectives. The core of MM-Point lies in the Multi-modal interaction and transmission between 3D objects and multiple 2D views at the same time. In order to more effectively simultaneously perform the consistent cross-modal objective of 2D multi-view information based on contrastive learning, we further propose Multi-MLP and Multi-level Augmentation strategies. Through carefully designed transformation strategies, we further learn Multi-level invariance in 2D Multi-views. MM-Point demonstrates state-of-the-art (SOTA) performance in various downstream tasks. For instance, it achieves a peak accuracy of 92.4% on the synthetic dataset ModelNet40, and a top accuracy of 87.8% on the real-world dataset ScanObjectNN, comparable to fully supervised methods. Additionally, we demonstrate its effectiveness in tasks such as few-shot classification, 3D part segmentation and 3D semantic segmentation.

----

## [753] Spatial Transform Decoupling for Oriented Object Detection

**Authors**: *Hongtian Yu, Yunjie Tian, Qixiang Ye, Yunfan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28502](https://doi.org/10.1609/aaai.v38i7.28502)

**Abstract**:

Vision Transformers (ViTs) have achieved remarkable success in computer vision tasks. However, their potential in rotation-sensitive scenarios has not been fully explored, and this limitation may be inherently attributed to the lack of spatial invariance in the data-forwarding process. In this study, we present a novel approach, termed Spatial Transform Decoupling (STD), providing a simple-yet-effective solution for oriented object detection with ViTs. Built upon stacked ViT blocks, STD utilizes separate network branches to predict the position, size, and angle of bounding boxes, effectively harnessing the spatial transform potential of ViTs in a divide-and-conquer fashion. Moreover, by aggregating cascaded activation masks (CAMs) computed upon the regressed parameters, STD gradually enhances features within regions of interest (RoIs), which complements the self-attention mechanism. Without bells and whistles, STD achieves state-of-the-art performance on the benchmark datasets including DOTA-v1.0 (82.24% mAP) and HRSC2016 (98.55% mAP), which demonstrates the effectiveness of the proposed method. Source code is available at https://github.com/yuhongtian17/Spatial-Transform-Decoupling.

----

## [754] Step Vulnerability Guided Mean Fluctuation Adversarial Attack against Conditional Diffusion Models

**Authors**: *Hongwei Yu, Jiansheng Chen, Xinlong Ding, Yudong Zhang, Ting Tang, Huimin Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28503](https://doi.org/10.1609/aaai.v38i7.28503)

**Abstract**:

The high-quality generation results of conditional diffusion models have brought about concerns regarding privacy and copyright issues. As a possible technique for preventing the abuse of diffusion models, the adversarial attack against diffusion models has attracted academic attention recently. In this work, utilizing the phenomenon that diffusion models are highly sensitive to the mean value of the input noise, we propose the Mean Fluctuation Attack (MFA) to introduce mean fluctuations by shifting the mean values of the estimated noises during the reverse process. In addition, we reveal that the vulnerability of different reverse steps against adversarial attacks actually varies significantly. By modeling the step vulnerability and using it as guidance to sample the target steps for generating adversarial examples, the effectiveness of adversarial attacks can be substantially enhanced. Extensive experiments show that our algorithm can steadily cause the mean shift of the predicted noises so as to disrupt the entire reverse generation process and degrade the generation results significantly. We also demonstrate that the step vulnerability is intrinsic to the reverse process by verifying its effectiveness in an attack method other than MFA. Code and Supplementary is available at https://github.com/yuhongwei22/MFA

----

## [755] PaintHuman: Towards High-Fidelity Text-to-3D Human Texturing via Denoised Score Distillation

**Authors**: *Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Tom Weidong Cai, Wayne Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28504](https://doi.org/10.1609/aaai.v38i7.28504)

**Abstract**:

Recent advances in zero-shot text-to-3D human generation, which employ the human model prior (e.g., SMPL) or Score Distillation Sampling (SDS) with pre-trained text-to-image diffusion models, have been groundbreaking. However, SDS may provide inaccurate gradient directions under the weak diffusion guidance, as it tends to produce over-smoothed results and generate body textures that are inconsistent with the detailed mesh geometry. Therefore, directly leveraging existing strategies for high-fidelity text-to-3D human texturing is challenging. In this work, we propose a model called PaintHuman to addresses the challenges from two perspectives. We first propose a novel score function, Denoised Score Distillation (DSD), which directly modifies the SDS by introducing negative gradient components to iteratively correct the gradient direction and generate high-quality textures. In addition, we use the depth map as a geometric guide to ensure that the texture is semantically aligned to human mesh surfaces. To guarantee the quality of rendered results, we employ geometry-aware networks to predict surface materials and render realistic human textures. Extensive experiments, benchmarked against state-of-the-art (SoTA) methods, validate the efficacy of our approach.Project page: https://painthuman.github.io/.

----

## [756] CatFormer: Category-Level 6D Object Pose Estimation with Transformer

**Authors**: *Sheng Yu, Di-Hua Zhai, Yuanqing Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28505](https://doi.org/10.1609/aaai.v38i7.28505)

**Abstract**:

Although there has been significant progress in category-level object pose estimation in recent years, there is still considerable room for improvement. In this paper, we propose a novel transformer-based category-level 6D pose estimation method  called CatFormer to enhance the accuracy pose estimation. CatFormer comprises three main parts: a coarse deformation part, a fine deformation part, and a recurrent refinement part. In the coarse and fine deformation sections, we introduce a transformer-based deformation module that performs point cloud deformation and completion in the feature space. Additionally, after each deformation, we incorporate a transformer-based graph module to adjust fused features and establish geometric and topological relationships between points based on these features. Furthermore, we present an end-to-end recurrent refinement module that enables the prior point cloud to deform multiple times according to real scene features. We evaluate CatFormer's performance by training and testing it on CAMERA25 and REAL275 datasets. Experimental results demonstrate that CatFormer surpasses state-of-the-art methods. Moreover, we extend the usage of CatFormer to instance-level object pose estimation on the LINEMOD dataset, as well as object pose estimation in real-world scenarios. The experimental results validate the effectiveness and generalization capabilities of CatFormer. Our code and the supplemental materials are avaliable at https://github.com/BIT-robot-group/CatFormer.

----

## [757] DME: Unveiling the Bias for Better Generalized Monocular Depth Estimation

**Authors**: *Songsong Yu, Yifan Wang, Yunzhi Zhuge, Lijun Wang, Huchuan Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28506](https://doi.org/10.1609/aaai.v38i7.28506)

**Abstract**:

This paper aims to design monocular depth estimation models with better generalization abilities. To this end, we have conducted quantitative analysis and discovered two important insights. First, the Simulation Correlation phenomenon, commonly seen in long-tailed classification problems, also exists in monocular depth estimation, indicating that the imbalanced depth distribution in training data may be the cause of limited generalization ability. Second, the imbalanced and long-tail distribution of depth values extends beyond the dataset scale, and also manifests within each individual image, further exacerbating the challenge of monocular depth estimation. Motivated by the above findings, we propose the Distance-aware Multi-Expert (DME) depth estimation model. Unlike prior methods that handle different depth range indiscriminately, DME adopts a divide-and-conquer philosophy where each expert is responsible for depth estimation of regions within a specific depth range. As such, the depth distribution seen by each expert is more uniform and can be more easily predicted. A pixel-level routing module is further designed and learned to stitch the prediction of all experts into the final depth map. Experiments show that DME achieves state-of-the-art performance on both NYU-Depth v2 and KITTI, and also delivers favorable zero-shot generalization capability on unseen datasets.

----

## [758] DOCTR: Disentangled Object-Centric Transformer for Point Scene Understanding

**Authors**: *Xiaoxuan Yu, Hao Wang, Weiming Li, Qiang Wang, SoonYong Cho, Younghun Sung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28507](https://doi.org/10.1609/aaai.v38i7.28507)

**Abstract**:

Point scene understanding is a challenging task to process real-world scene point cloud, which aims at segmenting each object, estimating its pose, and reconstructing its mesh simultaneously. Recent state-of-the-art method first segments each object and then processes them independently with multiple stages for the different sub-tasks. This leads to a complex pipeline to optimize and makes it hard to leverage the relationship constraints between multiple objects. In this work, we propose a novel Disentangled Object-Centric TRansformer (DOCTR) that explores object-centric representation to facilitate learning with multiple objects for the multiple sub-tasks in a unified manner. Each object is represented as a query, and a Transformer decoder is adapted to iteratively optimize all the queries involving their relationship. In particular, we introduce a semantic-geometry disentangled query (SGDQ) design that enables the query features to attend separately to semantic information and geometric information relevant to the corresponding sub-tasks. A hybrid bipartite matching module is employed to well use the supervisions from all the sub-tasks during training. Qualitative and quantitative experimental results demonstrate that our method achieves state-of-the-art performance on the challenging ScanNet dataset. Code is available at https://github.com/SAITPublic/DOCTR.

----

## [759] Discretization-Induced Dirichlet Posterior for Robust Uncertainty Quantification on Regression

**Authors**: *Xuanlong Yu, Gianni Franchi, Jindong Gu, Emanuel Aldea*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28508](https://doi.org/10.1609/aaai.v38i7.28508)

**Abstract**:

Uncertainty quantification is critical for deploying deep neural networks (DNNs) in real-world applications. An Auxiliary Uncertainty Estimator (AuxUE) is one of the most effective means to estimate the uncertainty of the main task prediction without modifying the main task model. To be considered robust, an AuxUE must be capable of maintaining its performance and triggering higher uncertainties while encountering Out-of-Distribution (OOD) inputs, i.e., to provide robust aleatoric and epistemic uncertainty. However, for vision regression tasks, current AuxUE designs are mainly adopted for aleatoric uncertainty estimates, and AuxUE robustness has not been explored. In this work, we propose a generalized AuxUE scheme for more robust uncertainty quantification on regression tasks. Concretely, to achieve a more robust aleatoric uncertainty estimation, different distribution assumptions are considered for heteroscedastic noise, and Laplace distribution is finally chosen to approximate the prediction error. For epistemic uncertainty, we propose a novel solution named Discretization-Induced Dirichlet pOsterior (DIDO), which models the Dirichlet posterior on the discretized prediction error. Extensive experiments on age estimation, monocular depth estimation, and super-resolution tasks show that our proposed method can provide robust uncertainty estimates in the face of noisy inputs and that it can be scalable to both image-level and pixel-wise tasks.

----

## [760] Attacks on Continual Semantic Segmentation by Perturbing Incremental Samples

**Authors**: *Zhidong Yu, Wei Yang, Xike Xie, Zhenbo Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28509](https://doi.org/10.1609/aaai.v38i7.28509)

**Abstract**:

As an essential computer vision task, Continual Semantic Segmentation (CSS) has received a lot of attention. However, security issues regarding this task have not been fully studied. To bridge this gap, we study the problem of attacks in CSS in this paper. We first propose a new task, namely, attacks on incremental samples in CSS, and reveal that the attacks on incremental samples corrupt the performance of CSS in both old and new classes.  Moreover, we present an adversarial sample generation method based on class shift, namely Class Shift Attack (CS-Attack), which is an offline and easy-to-implement approach for CSS. CS-Attack is able to significantly degrade the performance of models on both old and new classes without knowledge of the incremental learning approach, which undermines the original purpose of the incremental learning, i.e., learning new classes while retaining old knowledge. Experiments show that on the popular datasets Pascal VOC, ADE20k, and Cityscapes, our approach easily degrades the performance of currently popular CSS methods, which reveals the importance of security in CSS.

----

## [761] Data-Free Hard-Label Robustness Stealing Attack

**Authors**: *Xiaojian Yuan, Kejiang Chen, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28510](https://doi.org/10.1609/aaai.v38i7.28510)

**Abstract**:

The popularity of Machine Learning as a Service (MLaaS) has led to increased concerns about Model Stealing Attacks (MSA), which aim to craft a clone model by querying MLaaS. Currently, most research on MSA assumes that MLaaS can provide soft labels and that the attacker has a proxy dataset with a similar distribution. However, this fails to encapsulate the more practical scenario where only hard labels are returned by MLaaS and the data distribution remains elusive. Furthermore, most existing work focuses solely on stealing the model accuracy, neglecting the model robustness, while robustness is essential in security-sensitive scenarios, e.g, face-scan payment. Notably, improving model robustness often necessitates the use of expensive techniques such as adversarial training, thereby further making stealing robustness a more lucrative prospect. In response to these identified gaps, we introduce a novel Data-Free Hard-Label Robustness Stealing (DFHL-RS) attack in this paper, which enables the stealing of both model accuracy and robustness by simply querying hard labels of the target model without the help of any natural data. Comprehensive experiments demonstrate the effectiveness of our method. The clone model achieves a clean accuracy of 77.86% and a robust accuracy of 39.51% against AutoAttack, which are only 4.71% and 8.40% lower than the target model on the CIFAR-10 dataset, significantly exceeding the baselines. Our code is available at: https://github.com/LetheSec/DFHL-RS-Attack.

----

## [762] Efficient Conditional Diffusion Model with Probability Flow Sampling for Image Super-resolution

**Authors**: *Yutao Yuan, Chun Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28511](https://doi.org/10.1609/aaai.v38i7.28511)

**Abstract**:

Image super-resolution is a fundamentally ill-posed problem because multiple valid high-resolution images exist for one low-resolution image. Super-resolution methods based on diffusion probabilistic models can deal with the ill-posed nature by learning the distribution of high-resolution images conditioned on low-resolution images, avoiding the problem of blurry images in PSNR-oriented methods. However, existing diffusion-based super-resolution methods have high time consumption with the use of iterative sampling, while the quality and consistency of generated images are less than ideal due to problems like color shifting. In this paper, we propose Efficient Conditional Diffusion Model with Probability Flow Sampling (ECDP) for image super-resolution. To reduce the time consumption, we design a continuous-time conditional diffusion model for image super-resolution, which enables the use of probability flow sampling for efficient generation. Additionally, to improve the consistency of generated images, we propose a hybrid parametrization for the denoiser network, which interpolates between the data-predicting parametrization and the noise-predicting parametrization for different noise scales. Moreover, we design an image quality loss as a complement to the score matching loss of diffusion models, further improving the consistency and quality of super-resolution. Extensive experiments on DIV2K, ImageNet, and CelebA demonstrate that our method achieves higher super-resolution quality than existing diffusion-based image super-resolution methods while having lower time consumption. Our code is available at https://github.com/Yuan-Yutao/ECDP.

----

## [763] SD-MVS: Segmentation-Driven Deformation Multi-View Stereo with Spherical Refinement and EM Optimization

**Authors**: *Zhenlong Yuan, Jiakai Cao, Zhaoxin Li, Hao Jiang, Zhaoqi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28512](https://doi.org/10.1609/aaai.v38i7.28512)

**Abstract**:

In this paper, we introduce Segmentation-Driven Deformation Multi-View Stereo (SD-MVS), a method that can effectively tackle challenges in 3D reconstruction of textureless areas. We are the first to adopt the Segment Anything Model (SAM) to distinguish semantic instances in scenes and further leverage these constraints for pixelwise patch deformation on both matching cost and propagation. Concurrently, we propose a unique refinement strategy that combines spherical coordinates and gradient descent on normals and pixelwise search interval on depths, significantly improving the completeness of reconstructed 3D model. Furthermore, we adopt the Expectation-Maximization (EM) algorithm to alternately optimize the aggregate matching cost and hyperparameters, effectively mitigating the problem of parameters being excessively dependent on empirical tuning. Evaluations on the ETH3D high-resolution multi-view stereo benchmark and the Tanks and Temples dataset demonstrate that our method can achieve state-of-the-art results with less time consumption.

----

## [764] KeDuSR: Real-World Dual-Lens Super-Resolution via Kernel-Free Matching

**Authors**: *Huanjing Yue, Zifan Cui, Kun Li, Jingyu Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28513](https://doi.org/10.1609/aaai.v38i7.28513)

**Abstract**:

Dual-lens super-resolution (SR) is a practical scenario for reference (Ref) based SR by utilizing the telephoto image (Ref) to assist the super-resolution of the low-resolution wide-angle image (LR input). Different from general RefSR, the Ref in dual-lens SR only covers the overlapped field of view (FoV) area. However, current dual-lens SR methods rarely utilize these specific characteristics and directly perform dense matching between the LR input and Ref. Due to the resolution gap between LR and Ref, the matching may miss the best-matched candidate and destroy the consistent structures in the overlapped FoV area. Different from them, we propose to first align the Ref with the center region (namely the overlapped FoV area) of the LR input by combining global warping and local warping to make the aligned Ref be sharp and consistent. Then, we formulate the aligned Ref and LR center as value-key pairs, and the corner region of the LR is formulated as queries. In this way, we propose a kernel-free matching strategy by matching between the LR-corner (query) and LR-center (key) regions, and the corresponding aligned Ref (value) can be warped to the corner region of the target. Our kernel-free matching strategy avoids the resolution gap between LR and Ref, which makes our network have better generalization ability. In addition, we construct a DuSR-Real dataset with (LR, Ref, HR) triples, where the LR and HR are well aligned. Experiments on three datasets demonstrate that our method outperforms the second-best method by a large margin. Our code and dataset are available at https://github.com/ZifanCui/KeDuSR.

----

## [765] SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation

**Authors**: *Wenxi Yue, Jing Zhang, Kun Hu, Yong Xia, Jiebo Luo, Zhiyong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28514](https://doi.org/10.1609/aaai.v38i7.28514)

**Abstract**:

The Segment Anything Model (SAM) is a powerful foundation model that has revolutionised image segmentation. To apply SAM to surgical instrument segmentation, a common approach is to locate precise points or boxes of instruments and then use them as prompts for SAM in a zero-shot manner. However, we observe two problems with this naive pipeline: (1) the domain gap between natural objects and surgical instruments leads to inferior generalisation of SAM; and (2) SAM relies on precise point or box locations for accurate segmentation, requiring either extensive manual guidance or a well-performing specialist detector for prompt preparation, which leads to a complex multi-stage pipeline. To address these problems, we introduce SurgicalSAM, a novel end-to-end efficient-tuning approach for SAM to effectively integrate surgical-specific information with SAM’s pre-trained knowledge for improved generalisation. Specifically, we propose a lightweight prototype-based class prompt encoder for tuning, which directly generates prompt embeddings from class prototypes and eliminates the use of explicit prompts for improved robustness and a simpler pipeline. In addition, to address the low inter-class variance among surgical instrument categories, we propose contrastive prototype learning, further enhancing the discrimination of the class prototypes for more accurate class prompting. The results of extensive experiments on both EndoVis2018 and EndoVis2017 datasets demonstrate that SurgicalSAM achieves state-of-the-art performance while only requiring a small number of tunable parameters. The source code is available at https://github.com/wenxi-yue/SurgicalSAM.

----

## [766] Unveiling Details in the Dark: Simultaneous Brightening and Zooming for Low-Light Image Enhancement

**Authors**: *Ziyu Yue, Jiaxin Gao, Zhixun Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28515](https://doi.org/10.1609/aaai.v38i7.28515)

**Abstract**:

Existing super-resolution methods exhibit limitations when applied to nighttime scenes, primarily due to their lack of adaptation to low-pair dynamic range and noise-heavy dark-light images. In response, this research introduces an innovative customized framework to simultaneously Brighten and Zoom in low-resolution images captured in low-light conditions, dubbed BrZoNet. The core method begins by feeding low-light, low-resolution images, and their corresponding ground truths into the Retinex-induced siamese decoupling network. This process yields distinct reflectance maps and illuminance maps, guided by supervision from the ground truth’s decomposition maps. Subsequently, these reflectance and illuminance maps transition into an intricate super-resolution sub-network. This sub-network employs a meticulously designed cross-layer content-aware interactor - Illumination-aware Interaction Unit(IaIU), elegantly endowed with a gating mechanism. The IaIU facilitates meaningful feature interaction between illuminance and reflectance features while effectively reducing unwanted noise. An intricate super-resolution cage is also constructed to comprehensively integrate information, ultimately resulting in the generation of high-resolution images featuring intricate details. Thorough and diverse experiments validate the superiority of the proposed BrZoNet, surpassing contemporary cutting-edge technologies by proficiently augmenting brightness and intricately recovering complex details, showcasing advancements of 7.1% in PSNR, 2.4% in SSIM, and an impressive 36.8% in LPIPS metrics.

----

## [767] Weakly-Supervised Temporal Action Localization by Inferring Salient Snippet-Feature

**Authors**: *Wulian Yun, Mengshi Qi, Chuanming Wang, Huadong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28516](https://doi.org/10.1609/aaai.v38i7.28516)

**Abstract**:

Weakly-supervised temporal action localization aims to locate action regions and identify action categories in untrimmed videos simultaneously by taking only video-level labels as the supervision. Pseudo label generation is a promising strategy to solve the challenging problem, but the current methods ignore the natural temporal structure of the video that can provide rich information to assist such a generation process. In this paper, we propose a novel weakly-supervised temporal action localization method by inferring salient snippet-feature. First, we design a saliency inference module that exploits the variation relationship between temporal neighbor snippets to discover salient snippet-features, which can reflect the significant dynamic change in the video. Secondly, we introduce a boundary refinement module that enhances salient snippet-features through the information interaction unit. Then, a discrimination enhancement module is introduced to enhance the discriminative nature of snippet-features. Finally, we adopt the refined snippet-features to produce high-fidelity pseudo labels, which could be used to supervise the training of the action localization network. Extensive experiments on two publicly available datasets, i.e., THUMOS14 and ActivityNet v1.3, demonstrate our proposed method achieves significant improvements compared to the state-of-the-art methods. Our source code is available at https://github.com/wuli55555/ISSF.

----

## [768] Behavioral Recognition of Skeletal Data Based on Targeted Dual Fusion Strategy

**Authors**: *Xiao Yun, Chenglong Xu, Kévin Riou, Kaiwen Dong, Yanjing Sun, Song Li, Kévin Subrin, Patrick Le Callet*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28517](https://doi.org/10.1609/aaai.v38i7.28517)

**Abstract**:

The deployment of multi-stream fusion strategy on behavioral recognition from skeletal data can extract complementary features from different information streams and improve the recognition accuracy, but suffers from high model complexity and a large number of parameters. Besides, existing multi-stream methods using a fixed adjacency matrix homogenizes the model’s discrimination process across diverse actions, causing reduction of the actual lift for the multi-stream model. Finally, attention mechanisms are commonly applied to the multi-dimensional features, including spatial, temporal and channel dimensions. But their attention scores are typically fused in a concatenated manner, leading to the ignorance of the interrelation between joints in complex actions. To alleviate these issues, the Front-Rear dual Fusion Graph Convolutional Network (FRF-GCN) is proposed to provide a lightweight model based on skeletal data. Targeted adjacency matrices are also designed for different front fusion streams, allowing the model to focus on actions of varying magnitudes. Simultaneously, the mechanism of Spatial-Temporal-Channel Parallel Attention (STC-P), which processes attention in parallel and places greater emphasis on useful information, is proposed to further improve model’s performance. FRF-GCN demonstrates significant competitiveness compared to the current state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120 and Kinetics-Skeleton 400 datasets. Our code is available at: https://github.com/sunbeam-kkt/FRF-GCN-master.

----

## [769] Zero-Shot Aerial Object Detection with Visual Description Regularization

**Authors**: *Zhengqing Zang, Chenyu Lin, Chenwei Tang, Tao Wang, Jiancheng Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28518](https://doi.org/10.1609/aaai.v38i7.28518)

**Abstract**:

Existing object detection models are mainly trained on large-scale labeled datasets. However, annotating data for novel aerial object classes is expensive since it is time-consuming and may require expert knowledge. Thus, it is desirable to study label-efficient object detection methods on aerial images. In this work, we propose a zero-shot method for aerial object detection named visual Description Regularization, or DescReg. 
Concretely, we identify the weak semantic-visual correlation of the aerial objects and aim to address the challenge with prior descriptions of their visual appearance. Instead of directly encoding the descriptions into class embedding space which suffers from the representation gap problem, we propose to infuse the prior inter-class visual similarity conveyed in the descriptions into the embedding learning. The infusion process is accomplished with a newly designed similarity-aware triplet loss which incorporates structured regularization on the representation space. We conduct extensive experiments with three challenging aerial object detection datasets, including DIOR, xView, and DOTA. The results demonstrate that DescReg significantly outperforms the state-of-the-art ZSD methods with complex projection designs and generative frameworks, e.g., DescReg outperforms 
best reported ZSD method on DIOR by 4.5 mAP on unseen classes and 8.1 in HM. We further show the generalizability of DescReg by integrating it into generative ZSD methods as well as varying the detection architecture.
Codes will be released at https://github.com/zq-zang/DescReg.

----

## [770] Controllable Mind Visual Diffusion Model

**Authors**: *Bohan Zeng, Shanglin Li, Xuhui Liu, Sicheng Gao, Xiaolong Jiang, Xu Tang, Yao Hu, Jianzhuang Liu, Baochang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28519](https://doi.org/10.1609/aaai.v38i7.28519)

**Abstract**:

Brain signal visualization has emerged as an active research area, serving as a critical interface between the human visual system and computer vision models. Diffusion-based methods have recently shown promise in analyzing functional magnetic resonance imaging (fMRI) data, including the reconstruction of high-quality images consistent with original visual stimuli. Nonetheless, it remains a critical challenge to effectively harness the semantic and silhouette information extracted from brain signals. In this paper, we propose a novel approach, termed as Controllable Mind Visual Diffusion Model (CMVDM). Specifically, CMVDM first extracts semantic and silhouette information from fMRI data using attribute alignment and assistant networks. Then, a control model is introduced in conjunction with a residual block to fully exploit the extracted information for image synthesis, generating high-quality images that closely resemble the original visual stimuli in both semantic content and silhouette characteristics. Through extensive experimentation, we demonstrate that CMVDM outperforms existing state-of-the-art methods both qualitatively and quantitatively. Our code is available at https://github.com/zengbohan0217/CMVDM.

----

## [771] MGQFormer: Mask-Guided Query-Based Transformer for Image Manipulation Localization

**Authors**: *Kunlun Zeng, Ri Cheng, Weimin Tan, Bo Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28520](https://doi.org/10.1609/aaai.v38i7.28520)

**Abstract**:

Deep learning-based models have made great progress in image tampering localization, which aims to distinguish between manipulated and authentic regions. However, these models suffer from inefficient training. This is because they use ground-truth mask labels mainly through the cross-entropy loss, which prioritizes per-pixel precision but disregards the spatial location and shape details of manipulated regions. To address this problem, we propose a Mask-Guided Query-based Transformer Framework (MGQFormer), which uses ground-truth masks to guide the learnable query token (LQT) in identifying the forged regions. Specifically, we extract feature embeddings of ground-truth masks as the guiding query token (GQT) and feed GQT and LQT into MGQFormer to estimate fake regions, respectively. Then we make MGQFormer learn the position and shape information in ground-truth mask labels by proposing a mask-guided loss to reduce the feature distance between GQT and LQT.  We also observe that such mask-guided training strategy has a significant impact on the convergence speed of MGQFormer training. Extensive experiments on multiple benchmarks show that our method significantly improves over state-of-the-art methods.

----

## [772] Weakly-Supervised Mirror Detection via Scribble Annotations

**Authors**: *Mingfeng Zha, Yunqiang Pei, Guoqing Wang, Tianyu Li, Yang Yang, Wenbin Qian, Heng Tao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28521](https://doi.org/10.1609/aaai.v38i7.28521)

**Abstract**:

Mirror detection is of great significance for avoiding false recognition of reflected objects in computer vision tasks. Existing mirror detection frameworks usually follow a supervised setting, which relies heavily on high quality labels and suffers from poor generalization. To resolve this, we instead propose the first weakly-supervised mirror detection framework and also provide the first scribble-based mirror dataset. Specifically, we relabel 10,158 images, most of which have a labeled pixel ratio of less than 0.01 and take only about 8 seconds to label. Considering that the mirror regions usually show great scale variation, and also irregular and occluded, thus leading to issues of incomplete or over detection, we propose a local-global feature enhancement (LGFE) module to fully capture the context and details. Moreover, it is difficult to obtain basic mirror structure using scribble annotation, and the distinction between foreground (mirror) and background (non-mirror) features is not emphasized caused by mirror reflections. Therefore, we propose a foreground-aware mask attention (FAMA), integrating mirror edges and semantic features to complete mirror regions and suppressing the influence of backgrounds. Finally, to improve the robustness of the network, we propose a prototype contrast loss (PCL) to learn more general foreground features across images. Extensive experiments show that our network outperforms relevant state-of-the-art weakly supervised methods, and even some fully supervised methods. The dataset and codes are available at https://github.com/winter-flow/WSMD.

----

## [773] Towards Compact 3D Representations via Point Feature Enhancement Masked Autoencoders

**Authors**: *Yaohua Zha, Huizhen Ji, Jinmin Li, Rongsheng Li, Tao Dai, Bin Chen, Zhi Wang, Shu-Tao Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28522](https://doi.org/10.1609/aaai.v38i7.28522)

**Abstract**:

Learning 3D  representation plays a critical role in masked autoencoder (MAE) based pre-training methods for point cloud, including single-modal and cross-modal based MAE. Specifically, although cross-modal MAE methods learn strong 3D representations via the auxiliary of other modal knowledge, they often suffer from heavy computational burdens and heavily rely on massive cross-modal data pairs that are often unavailable, which hinders their applications in practice. Instead, single-modal methods with solely point clouds as input are preferred in real applications due to their simplicity and efficiency. However, such methods easily suffer from limited 3D representations with global random mask input. To learn compact 3D representations, we propose a simple yet effective Point Feature Enhancement Masked Autoencoders (Point-FEMAE), which mainly consists of a global branch and a local branch to capture latent semantic features. Specifically, to learn more compact features, a share-parameter Transformer encoder is introduced to extract point features from the global and local unmasked patches obtained by global random and local block mask strategies, followed by a specific decoder to reconstruct. Meanwhile, to further enhance features in the local branch, we propose a Local Enhancement Module with local patch convolution to perceive fine-grained local context at larger scales. Our method significantly improves the pre-training efficiency compared to cross-modal alternatives, and extensive downstream experiments underscore the state-of-the-art effectiveness, particularly outperforming our baseline (Point-MAE) by 5.16%, 5.00%, and 5.04% in three variants of ScanObjectNN, respectively. Code is available at https://github.com/zyh16143998882/AAAI24-PointFEMAE.

----

## [774] Fine-Grained Knowledge Selection and Restoration for Non-exemplar Class Incremental Learning

**Authors**: *Jiang-Tian Zhai, Xialei Liu, Lu Yu, Ming-Ming Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28523](https://doi.org/10.1609/aaai.v38i7.28523)

**Abstract**:

Non-exemplar class incremental learning aims to learn both the new and old tasks without accessing any training data from the past. This strict restriction enlarges the difficulty of alleviating catastrophic forgetting since all techniques can only be applied to current task data. Considering this challenge, we propose a novel framework of fine-grained knowledge selection and restoration. The conventional knowledge distillation-based methods place too strict constraints on the network parameters and features to prevent forgetting, which limits the training of new tasks. To loose this constraint, we proposed a novel fine-grained selective patch-level distillation to adaptively balance plasticity and stability. Some task-agnostic patches can be used to preserve the decision boundary of the old task. While some patches containing the important foreground are favorable for learning the new task.
   Moreover, we employ a task-agnostic mechanism to generate more realistic prototypes of old tasks with the current task sample for reducing classifier bias for fine-grained knowledge restoration.  Extensive experiments on CIFAR100, TinyImageNet and ImageNet-Subset demonstrate the effectiveness of our method. Code is available at https://github.com/scok30/vit-cil.

----

## [775] Multi-Prompts Learning with Cross-Modal Alignment for Attribute-Based Person Re-identification

**Authors**: *Yajing Zhai, Yawen Zeng, Zhiyong Huang, Zheng Qin, Xin Jin, Da Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28524](https://doi.org/10.1609/aaai.v38i7.28524)

**Abstract**:

The fine-grained attribute descriptions can significantly supplement the valuable semantic information for person image, which is vital to the success of person re-identification (ReID)
task. However, current ReID algorithms typically failed to effectively leverage the rich contextual information available, primarily due to their reliance on simplistic and coarse utilization of image attributes. Recent advances in artificial intelligence generated content have made it possible to automatically generate plentiful fine-grained attribute descriptions and make full use of them. Thereby, this paper explores the potential of using the generated multiple person attributes as prompts in ReID tasks with off-the-shelf (large) models for more accurate retrieval results. To this end, we present a new framework called Multi-Prompts ReID (MP-ReID), based on prompt learning and language models, to fully dip fine attributes to assist ReID task. Specifically, MP-ReID first learns to hallucinate diverse, informative, and promptable sentences for describing the query images. This procedure includes (i) explicit prompts of which attributes a person has and furthermore (ii) implicit learnable prompts for adjusting/conditioning the criteria used towards this person identity matching. Explicit prompts are obtained by ensembling generation models, such as ChatGPT and VQA models. Moreover, an alignment module is designed to fuse multi-prompts (i.e., explicit and implicit ones) progressively and mitigate the cross-modal gap. Extensive experiments on the existing attribute-involved ReID datasets, namely, Market1501 and DukeMTMC-reID, demonstrate the effectiveness and rationality of the proposed MP-ReID solution.

----

## [776] Mono3DVG: 3D Visual Grounding in Monocular Images

**Authors**: *Yang Zhan, Yuan Yuan, Zhitong Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28525](https://doi.org/10.1609/aaai.v38i7.28525)

**Abstract**:

We introduce a novel task of 3D visual grounding in monocular RGB images using language descriptions with both appearance and geometry information. Specifically, we build a large-scale dataset, Mono3DRefer, which contains 3D object targets with their corresponding geometric text descriptions, generated by ChatGPT and refined manually. To foster this task, we propose Mono3DVG-TR, an end-to-end transformer-based network, which takes advantage of both the appearance and geometry information in text embeddings for multi-modal learning and 3D object localization. Depth predictor is designed to explicitly learn geometry features. The dual text-guided adapter is proposed to refine multiscale visual and geometry features of the referred object. Based on depth-text-visual stacking attention, the decoder fuses object-level geometric cues and visual appearance into a learnable query. Comprehensive benchmarks and some insightful analyses are provided for Mono3DVG. Extensive comparisons and ablation studies show that our method significantly outperforms all baselines. The dataset and code will be released.

----

## [777] Amodal Scene Analysis via Holistic Occlusion Relation Inference and Generative Mask Completion

**Authors**: *Bowen Zhang, Qing Liu, Jianming Zhang, Yilin Wang, Liyang Liu, Zhe Lin, Yifan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28526](https://doi.org/10.1609/aaai.v38i7.28526)

**Abstract**:

Amodal scene analysis entails interpreting the occlusion relationship among scene elements and inferring the possible shapes of the invisible parts. Existing methods typically frame this task as an extended instance segmentation or a pair-wise object de-occlusion problem. In this work, we propose a new framework, which comprises a Holistic Occlusion Relation Inference (HORI) module followed by an instance-level Generative Mask Completion (GMC) module. 
   Unlike previous approaches, which rely on mask completion results for occlusion reasoning, our HORI module directly predicts an occlusion relation matrix in a single pass. This approach is much more efficient than the pair-wise de-occlusion process and it naturally handles mutual occlusion, a common but often neglected situation.
   Moreover, we formulate the mask completion task as a generative process and use a diffusion-based GMC module for instance-level mask completion. This improves mask completion quality and provides multiple plausible solutions.
   We further introduce a large-scale amodal segmentation dataset with high-quality human annotations, including mutual occlusions. Experiments on our dataset and two public benchmarks demonstrate the advantages of our method. code public available at https://github.com/zbwxp/Amodal-AAAI.

----

## [778] High-Quality Real-Time Rendering Using Subpixel Sampling Reconstruction

**Authors**: *Boyu Zhang, Hongliang Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28527](https://doi.org/10.1609/aaai.v38i7.28527)

**Abstract**:

Generating high-quality, realistic rendering images for real-time applications generally requires tracing a few samples-per-pixel (spp) and using deep learning-based approaches to denoise the resulting low-spp images. Existing denoising methods necessitate a substantial time expenditure when rendering at high resolutions due to the physically-based sampling and network inference time burdens. In this paper, we propose a novel Monte Carlo sampling strategy to accelerate the sampling process and a corresponding denoiser, subpixel sampling reconstruction (SSR), to obtain high-quality images. Extensive experiments demonstrate that our method significantly outperforms previous approaches in denoising quality and reduces overall time costs, enabling real-time rendering capabilities at 2K resolution.

----

## [779] Weakly Supervised Few-Shot Object Detection with DETR

**Authors**: *Chenbo Zhang, Yinglu Zhang, Lu Zhang, Jiajia Zhao, Jihong Guan, Shuigeng Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28528](https://doi.org/10.1609/aaai.v38i7.28528)

**Abstract**:

In recent years, Few-shot Object Detection (FSOD) has become an increasingly important research topic in computer vision. However, existing FSOD methods require strong annotations including category labels and bounding boxes, and their performance is heavily dependent on the quality of box annotations. However, acquiring strong annotations is both expensive and time-consuming. This inspires the study on weakly supervised FSOD (WS-FSOD in short), which realizes FSOD with only image-level annotations, i.e., category labels. In this paper, we propose a new and effective weakly supervised FSOD method named WFS-DETR. By a well-designed pretraining process, WFS-DETR first acquires general object localization and integrity judgment capabilities on large-scale pretraining data. Then, it introduces object integrity into multiple-instance learning to solve the common local optimum problem by comprehensively exploiting both semantic and visual information. Finally, with simple fine-tuning, it transfers the knowledge learned from the base classes to the novel classes, which enables accurate detection of novel objects. Benefiting from this ``pretraining-refinement'' mechanism, WSF-DETR can achieve good generalization on different datasets. Extensive experiments also show that the proposed method clearly outperforms the existing counterparts in the WS-FSOD task.

----

## [780] S2WAT: Image Style Transfer via Hierarchical Vision Transformer Using Strips Window Attention

**Authors**: *Chiyu Zhang, Xiaogang Xu, Lei Wang, Zaiyan Dai, Jun Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28529](https://doi.org/10.1609/aaai.v38i7.28529)

**Abstract**:

Transformer's recent integration into style transfer leverages its proficiency in establishing long-range dependencies, albeit at the expense of attenuated local modeling. This paper introduces Strips Window Attention Transformer (S2WAT), a novel hierarchical vision transformer designed for style transfer. S2WAT employs attention computation in diverse window shapes to capture both short- and long-range dependencies. The merged dependencies utilize the "Attn Merge" strategy, which adaptively determines spatial weights based on their relevance to the target. Extensive experiments on representative datasets show the proposed method's effectiveness compared to state-of-the-art (SOTA) transformer-based and other approaches. The code and pre-trained models are available at https://github.com/AlienZhang1996/S2WAT.

----

## [781] Synergistic Multiscale Detail Refinement via Intrinsic Supervision for Underwater Image Enhancement

**Authors**: *Dehuan Zhang, Jingchun Zhou, Chunle Guo, Weishi Zhang, Chongyi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28530](https://doi.org/10.1609/aaai.v38i7.28530)

**Abstract**:

Visually restoring underwater scenes primarily involves mitigating interference from underwater media. Existing methods ignore the inherent scale-related characteristics in underwater scenes. Therefore, we present the synergistic multi-scale detail refinement via intrinsic supervision (SMDR-IS) for enhancing underwater scene details, which contain multi-stages. The low-degradation stage from the original images furnishes the original stage with multi-scale details, achieved through feature propagation using the Adaptive Selective Intrinsic Supervised Feature (ASISF) module. By using intrinsic supervision, the ASISF module can precisely control and guide feature transmission across multi-degradation stages, enhancing multi-scale detail refinement and minimizing the interference from irrelevant information in the low-degradation stage. In multi-degradation encoder-decoder framework of SMDR-IS, we introduce the Bifocal Intrinsic-Context Attention Module (BICA). Based on the intrinsic supervision principles, BICA efficiently exploits multi-scale scene information in images. BICA directs higher-resolution spaces by tapping into the insights of lower-resolution ones, underscoring the pivotal role of spatial contextual relationships in underwater image restoration. Throughout training, the inclusion of a multi-degradation loss function can enhance the network, allowing it to adeptly extract information across diverse scales. When benchmarked against state-of-the-art methods, SMDR-IS consistently showcases superior performance. Our code is available at https://github.com/zhoujingchun03/SMDR-IS

----

## [782] W2P: Switching from Weak Supervision to Partial Supervision for Semantic Segmentation

**Authors**: *Fangyuan Zhang, Tianxiang Pan, Jun-Hai Yong, Bin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28531](https://doi.org/10.1609/aaai.v38i7.28531)

**Abstract**:

Current weakly-supervised semantic segmentation (WSSS) techniques concentrate on enhancing class activation maps (CAMs) with image-level annotations. Yet, the emphasis on producing these pseudo-labels often overshadows the pivotal role of training the segmentation model itself. This paper underscores the significant influence of noisy pseudo-labels on segmentation network performance, particularly in boundary region. To address above issues, we introduce a novel paradigm: Weak to Partial Supervision (W2P). At its core, W2P categorizes the pseudo-labels from WSSS into two unique supervisions: trustworthy clean labels and uncertain noisy labels. Next, our proposed partially-supervised framework adeptly employs these clean labels to rectify the noisy ones, thereby promoting the continuous enhancement of the segmentation model. To further optimize boundary segmentation, we incorporate a noise detection mechanism that specifically preserves boundary regions while eliminating noise. During the noise refinement phase, we adopt a boundary-conscious noise correction technique to extract comprehensive boundaries from noisy areas. Furthermore, we devise a boundary generation approach that assists in predicting intricate boundary zones. Evaluations on the PASCAL VOC 2012 and MS COCO 2014 datasets confirm our method's impressive segmentation capabilities across various pseudo-labels.

----

## [783] HyperEditor: Achieving Both Authenticity and Cross-Domain Capability in Image Editing via Hypernetworks

**Authors**: *Hai Zhang, Chunwei Wu, Guitao Cao, Hailing Wang, Wenming Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28532](https://doi.org/10.1609/aaai.v38i7.28532)

**Abstract**:

Editing real images authentically while also achieving cross-domain editing remains a challenge. Recent studies have focused on converting real images into latent codes and accomplishing image editing by manipulating these codes. However, merely manipulating the latent codes would constrain the edited images to the generator's image domain, hindering the attainment of diverse editing goals. In response, we propose an innovative image editing method called HyperEditor, which utilizes weight factors generated by hypernetworks to reassign the weights of the pre-trained StyleGAN2's generator. Guided by CLIP's cross-modal image-text semantic alignment, this innovative approach enables us to simultaneously accomplish authentic attribute editing and cross-domain style transfer, a capability not realized in previous methods. Additionally, we ascertain that modifying only the weights of specific layers in the generator can yield an equivalent editing result. Therefore, we introduce an adaptive layer selector, enabling our hypernetworks to autonomously identify the layers requiring output weight factors, which can further improve our hypernetworks' efficiency. Extensive experiments on abundant challenging datasets demonstrate the effectiveness of our method.

----

## [784] RadOcc: Learning Cross-Modality Occupancy Knowledge through Rendering Assisted Distillation

**Authors**: *Haiming Zhang, Xu Yan, Dongfeng Bai, Jiantao Gao, Pan Wang, Bingbing Liu, Shuguang Cui, Zhen Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28533](https://doi.org/10.1609/aaai.v38i7.28533)

**Abstract**:

3D occupancy prediction is an emerging task that aims to estimate the occupancy states and semantics of 3D scenes using multi-view images. However, image-based scene perception encounters significant challenges in achieving accurate prediction due to the absence of geometric priors. In this paper, we address this issue by exploring cross-modal knowledge distillation in this task, i.e., we leverage a stronger multi-modal model to guide the visual model during training. In practice, we observe that directly applying features or logits alignment, proposed and widely used in bird's-eye-view (BEV) perception, does not yield satisfactory results. To overcome this problem, we introduce RadOcc, a Rendering assisted distillation paradigm for 3D Occupancy prediction. By employing differentiable volume rendering, we generate depth and semantic maps in perspective views and propose two novel consistency criteria between the rendered outputs of teacher and student models. Specifically, the depth consistency loss aligns the termination distributions of the rendered rays, while the semantic consistency loss mimics the intra-segment similarity guided by vision foundation models (VLMs). Experimental results on the nuScenes dataset demonstrate the effectiveness of our proposed method in improving various 3D occupancy prediction approaches, e.g., our proposed methodology enhances our baseline by 2.2% in the metric of mIoU and achieves 50% in Occ3D benchmark.

----

## [785] GSDD: Generative Space Dataset Distillation for Image Super-resolution

**Authors**: *Haiyu Zhang, Shaolin Su, Yu Zhu, Jinqiu Sun, Yanning Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28534](https://doi.org/10.1609/aaai.v38i7.28534)

**Abstract**:

Single image super-resolution (SISR), especially in the real world, usually builds a large amount of LR-HR image pairs to learn representations that contain rich textural and structural information. However, relying on massive data for model training not only reduces training efficiency, but also causes heavy data storage burdens. In this paper, we attempt a pioneering study on dataset distillation (DD) for SISR problems to explore how data could be slimmed and compressed for the task. Unlike previous coreset selection methods which select a few typical examples directly from the original data, we remove the limitation that the selected data cannot be further edited, and propose to synthesize and optimize samples to preserve more task-useful representations. Concretely, by utilizing pre-trained GANs as a suitable approximation of realistic data distribution, we propose GSDD, which distills data in a latent generative space based on GAN-inversion techniques. By optimizing them to match with the practical data distribution in an informative feature space, the distilled data could then be synthesized. Experimental results demonstrate that when trained with our distilled data, GSDD can achieve comparable performance to the state-of-the-art (SOTA) SISR algorithms, while a nearly ×8 increase in training efficiency and a saving of almost 93.2% data storage space can be realized. Further experiments on challenging real-world data also demonstrate the promising generalization ability of GSDD.

----

## [786] CSL: Class-Agnostic Structure-Constrained Learning for Segmentation Including the Unseen

**Authors**: *Hao Zhang, Fang Li, Lu Qi, Ming-Hsuan Yang, Narendra Ahuja*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28535](https://doi.org/10.1609/aaai.v38i7.28535)

**Abstract**:

Addressing Out-Of-Distribution (OOD) Segmentation and Zero-Shot Semantic Segmentation (ZS3) is challenging, necessitating segmenting unseen classes. Existing strategies adapt the class-agnostic Mask2Former (CA-M2F) tailored to specific tasks. However, these methods cater to singular tasks, demand training from scratch, and we demonstrate certain deficiencies in CA-M2F, which affect performance. We propose the Class-Agnostic Structure-Constrained Learning (CSL), a plug-in framework that can integrate with existing methods, thereby embedding structural constraints and achieving performance gain, including the unseen, specifically OOD, ZS3, and domain adaptation (DA) tasks. There are two schemes for CSL to integrate with existing methods (1) by distilling knowledge from a base teacher network, enforcing constraints across training and inference phrases, or (2) by leveraging established models to obtain per-pixel distributions without retraining, appending constraints during the inference phase. Our soft assignment and mask split methodologies enhance OOD object segmentation. Empirical evaluations demonstrate CSL's prowess in boosting the performance of existing algorithms spanning OOD segmentation, ZS3, and DA segmentation, consistently transcending the state-of-art across all three tasks.

----

## [787] A Robust Mutual-Reinforcing Framework for 3D Multi-Modal Medical Image Fusion Based on Visual-Semantic Consistency

**Authors**: *Hao Zhang, Xuhui Zuo, Huabing Zhou, Tao Lu, Jiayi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28536](https://doi.org/10.1609/aaai.v38i7.28536)

**Abstract**:

This work proposes a robust 3D medical image fusion framework to establish a mutual-reinforcing mechanism between visual fusion and lesion segmentation, achieving their double improvement. Specifically, we explore the consistency between vision and semantics by sharing feature fusion modules. Through the coupled optimization of the visual fusion loss and the lesion segmentation loss, visual-related and semantic-related features will be pulled into the same domain, effectively promoting accuracy improvement in a mutual-reinforcing manner. Further, we establish the robustness guarantees by constructing a two-level refinement constraint in the process of feature extraction and reconstruction. Benefiting from full consideration for common degradations in medical images, our framework can not only provide clear visual fusion results for doctor's observation, but also enhance the defense ability of lesion segmentation against these negatives. Extensive evaluations of visual fusion and lesion segmentation scenarios demonstrate the advantages of our method in terms of accuracy and robustness. Moreover, our proposed framework is generic, which can be well-compatible with existing lesion segmentation algorithms and improve their performance. The code is publicly available at https://github.com/HaoZhang1018/RMR-Fusion.

----

## [788] Learning Task-Aware Language-Image Representation for Class-Incremental Object Detection

**Authors**: *Hongquan Zhang, Bin-Bin Gao, Yi Zeng, Xudong Tian, Xin Tan, Zhizhong Zhang, Yanyun Qu, Jun Liu, Yuan Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28537](https://doi.org/10.1609/aaai.v38i7.28537)

**Abstract**:

Class-incremental object detection (CIOD) is a real-world desired capability, requiring an object detector to continuously adapt to new tasks without forgetting learned ones, with the main challenge being catastrophic forgetting. Many methods based on distillation and replay have been proposed to alleviate this problem. However, they typically learn on a pure visual backbone, neglecting the powerful representation capabilities of textual cues, which to some extent limits their performance. In this paper, we propose task-aware language-image representation to mitigate catastrophic forgetting, introducing a new paradigm for language-image-based CIOD. First of all, we demonstrate the significant advantage of language-image detectors in mitigating catastrophic forgetting. Secondly, we propose a learning task-aware language-image representation method that overcomes the existing drawback of directly utilizing the language-image detector for CIOD. More specifically, we learn the language-image representation of different tasks through an insulating approach in the training stage, while using the alignment scores produced by task-specific language-image representation in the inference stage. Through our proposed method, language-image detectors can be more practical for CIOD. We conduct extensive experiments on COCO 2017 and Pascal VOC 2007 and demonstrate that the proposed method achieves state-of-the-art results under the various CIOD settings.

----

## [789] Identification of Necessary Semantic Undertakers in the Causal View for Image-Text Matching

**Authors**: *Huatian Zhang, Lei Zhang, Kun Zhang, Zhendong Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28538](https://doi.org/10.1609/aaai.v38i7.28538)

**Abstract**:

Image-text matching bridges vision and language, which is a fundamental task in multimodal intelligence. Its key challenge lies in how to capture visual-semantic relevance. Fine-grained semantic interactions come from fragment alignments between image regions and text words. However, not all fragments contribute to image-text relevance, and many existing methods are devoted to mining the vital ones to measure the relevance accurately. How well image and text relate depends on the degree of semantic sharing between them. Treating the degree as an effect and fragments as its possible causes, we define those indispensable causes for the generation of the degree as necessary undertakers, i.e., if any of them did not occur, the relevance would be no longer valid. In this paper, we revisit image-text matching in the causal view and uncover inherent causal properties of relevance generation. Then we propose a novel theoretical prototype for estimating the probability-of-necessity of fragments, PN_f, for the degree of semantic sharing by means of causal inference, and further design a Necessary Undertaker Identification Framework (NUIF) for image-text matching, which explicitly formalizes the fragment's contribution to image-text relevance by modeling PN_f in two ways. Extensive experiments show our method achieves state-of-the-art on benchmarks Flickr30K and MSCOCO.

----

## [790] HR-Pro: Point-Supervised Temporal Action Localization via Hierarchical Reliability Propagation

**Authors**: *Huaxin Zhang, Xiang Wang, Xiaohao Xu, Zhiwu Qing, Changxin Gao, Nong Sang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28539](https://doi.org/10.1609/aaai.v38i7.28539)

**Abstract**:

Point-supervised Temporal Action Localization (PSTAL) is an emerging research direction for label-efficient learning. However, current methods mainly focus on optimizing the network either at the snippet-level or the instance-level, neglecting the inherent reliability of point annotations at both levels. In this paper, we propose a Hierarchical Reliability Propagation (HR-Pro) framework, which consists of two reliability-aware stages: Snippet-level Discrimination Learning and Instance-level Completeness Learning, both stages explore the efficient propagation of high-confidence cues in point annotations. For snippet-level learning, we introduce an online-updated memory to store reliable snippet prototypes for each class. We then employ a Reliability-aware Attention Block to capture both intra-video and inter-video dependencies of snippets, resulting in more discriminative and robust snippet representation. For instance-level learning, we propose a point-based proposal generation approach as a means of connecting snippets and instances, which produces high-confidence proposals for further optimization at the instance level. Through multi-level reliability-aware learning, we obtain more reliable confidence scores and more accurate temporal boundaries of predicted proposals. Our HR-Pro achieves state-of-the-art performance on multiple challenging benchmarks, including an impressive average mAP of 60.3% on THUMOS14. Notably, our HR-Pro largely surpasses all previous point-supervised methods, and even outperforms several competitive fully-supervised methods. Code will be available at https://github.com/pipixin321/HR-Pro.

----

## [791] AvatarVerse: High-Quality & Stable 3D Avatar Creation from Text and Pose

**Authors**: *Huichao Zhang, Bowen Chen, Hao Yang, Liao Qu, Xu Wang, Li Chen, Chao Long, Feida Zhu, Daniel K. Du, Min Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28540](https://doi.org/10.1609/aaai.v38i7.28540)

**Abstract**:

Creating expressive, diverse and high-quality 3D avatars from highly customized text descriptions and pose guidance is a challenging task, due to the intricacy of modeling and texturing in 3D that ensure details and various styles (realistic, fictional, etc). We present AvatarVerse, a stable pipeline for generating expressive high-quality 3D avatars from nothing but text descriptions and pose guidance. In specific, we introduce a 2D diffusion model conditioned on DensePose signal to establish 3D pose control of avatars through 2D images, which enhances view consistency from partially observed scenarios. It addresses the infamous Janus Problem and significantly stablizes the generation process. Moreover, we propose a progressive high-resolution 3D synthesis strategy, which obtains substantial improvement over the quality of the created 3D avatars. To this end, the proposed AvatarVerse pipeline achieves zero-shot 3D modeling of 3D avatars that are not only more expressive, but also in higher quality and fidelity than previous works. Rigorous qualitative evaluations and user studies showcase AvatarVerse's superiority in synthesizing high-fidelity 3D avatars, leading to a new standard in high-quality and stable 3D avatar creation. Our project page is: https://avatarverse3d.github.io/ .

----

## [792] Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection

**Authors**: *Jianping Zhang, Yizhan Huang, Zhuoer Xu, Weibin Wu, Michael R. Lyu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28541](https://doi.org/10.1609/aaai.v38i7.28541)

**Abstract**:

With the great achievement of vision transformers (ViTs), transformer-based approaches have become the new paradigm for solving various computer vision tasks. However, recent research shows that similar to convolutional neural networks (CNNs), ViTs are still vulnerable to adversarial attacks. To explore the shared deficiency of models with different structures, researchers begin to analyze the cross-structure adversarial transferability, which is still under-explored. Therefore, in this work, we focus on the ViT attacks to improve the cross-structure transferability between the transformer-based and convolution-based models. Previous studies fail to thoroughly investigate the influence of the components inside the ViT models on adversarial transferability, leading to inferior performance. To overcome the drawback, we launch a motivating study by linearly down-scaling the gradients of components inside the ViT models to analyze their influence on adversarial transferability. Based on the motivating study, we find that the gradient of the skip connection most influences transferability and believe that back-propagating gradients from deeper blocks can enhance transferability. Therefore, we propose the Virtual Dense Connection method (VDC). Specifically, without changing the forward pass, we first recompose the original network  to add virtual dense connections. Then we back-propagate gradients of deeper Attention maps and Multi-layer Perceptron (MLP)  blocks via virtual dense connections when generating adversarial samples. Extensive experiments confirm the superiority of our proposed method over the state-of-the-art baselines, with an 8.2% improvement in transferability between ViT models and a 7.2% improvement in cross-structure transferability from ViTs to CNNs.

----

## [793] Curvature-Invariant Adversarial Attacks for 3D Point Clouds

**Authors**: *Jianping Zhang, Wenwei Gu, Yizhan Huang, Zhihan Jiang, Weibin Wu, Michael R. Lyu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28542](https://doi.org/10.1609/aaai.v38i7.28542)

**Abstract**:

Imperceptibility is one of the crucial requirements for adversarial examples. Previous adversarial attacks on 3D point cloud recognition suffer from noticeable outliers, resulting in low imperceptibility. We think that the drawbacks can be alleviated via taking the local curvature of the point cloud into consideration. Existing approaches introduce the local geometry distance into the attack objective function. However, their definition of the local geometry distance neglects different perceptibility of distortions along different directions.  In this paper, we aim to enhance the imperceptibility of adversarial attacks on 3D point cloud recognition by better preserving the local curvature of the original 3D point clouds. To this end, we propose the Curvature-Invariant Method (CIM), which directly regularizes the back-propagated gradient during the generation of adversarial point clouds based on two assumptions. Specifically, we first decompose the back-propagated gradients into the tangent plane and the normal direction. Then we directly reduce the gradient along the large curvature direction on the tangent plane and only keep the gradient along the negative normal direction. Comprehensive experimental comparisons confirm the superiority of our approach. Notably, our strategy can achieve 7.2% and 14.5% improvements in Hausdorff distance and Gaussian curvature measurements of the imperceptibility.

----

## [794] Cross-Modal Feature Distribution Calibration for Few-Shot Visual Question Answering

**Authors**: *Jing Zhang, Xiaoqiang Liu, Mingzhe Chen, Zhe Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28543](https://doi.org/10.1609/aaai.v38i7.28543)

**Abstract**:

Few-shot Visual Question Answering (VQA) realizes few-shot cross-modal learning,  which is an emerging and challenging task in computer vision. Currently, most of the few-shot VQA methods are confined to simply extending few-shot classification methods to cross-modal tasks while ignoring the spatial distribution properties of multimodal features and cross-modal information interaction. To address this problem, we propose a novel Cross-modal feature Distribution Calibration Inference Network (CDCIN) in this paper, where a new concept named visual information entropy is proposed to realize multimodal features distribution calibration by cross-modal information interaction for more effective few-shot VQA. Visual information entropy is a statistical variable that represents the spatial distribution of visual features guided by the question, which is aligned before and after the reasoning process to mitigate redundant information and improve multi-modal features by our proposed visual information entropy calibration module. To further enhance the inference ability of cross-modal features, we additionally propose a novel pre-training method, where the reasoning sub-network of CDCIN is pretrained on the base class in a VQA classification paradigm and fine-tuned on the few-shot VQA datasets. Extensive experiments demonstrate that our proposed CDCIN achieves excellent performance on few-shot VQA and outperforms state-of-the-art methods on three widely used benchmark datasets.

----

## [795] Robust 3D Tracking with Quality-Aware Shape Completion

**Authors**: *Jingwen Zhang, Zikun Zhou, Guangming Lu, Jiandong Tian, Wenjie Pei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28544](https://doi.org/10.1609/aaai.v38i7.28544)

**Abstract**:

3D single object tracking remains a challenging problem due to the sparsity and incompleteness of the point clouds. Existing algorithms attempt to address the challenges in two strategies. The first strategy is to learn dense geometric features based on the captured sparse point cloud. Nevertheless, it is quite a formidable task since the learned dense geometric features are with high uncertainty for depicting the shape of the target object. The other strategy is to aggregate the sparse geometric features of multiple templates to enrich the shape information, which is a routine solution in 2D tracking. However, aggregating the coarse shape representations can hardly yield a precise shape representation. Different from 2D pixels, 3D points of different frames can be directly fused by coordinate transform, i.e., shape completion. Considering that, we propose to construct a synthetic target representation composed of dense and complete point clouds depicting the target shape precisely by shape completion for robust 3D tracking. Specifically, we design a voxelized 3D tracking framework with shape completion, in which we propose a quality-aware shape completion mechanism to alleviate the adverse effect of noisy historical predictions. It enables us to effectively construct and leverage the synthetic target representation. Besides, we also develop a voxelized relation modeling module and box refinement module to improve tracking performance. Favorable performance against state-of-the-art algorithms on three benchmarks demonstrates the effectiveness and generalization ability of our method.

----

## [796] Neighborhood-Enhanced 3D Human Pose Estimation with Monocular LiDAR in Long-Range Outdoor Scenes

**Authors**: *Jingyi Zhang, Qihong Mao, Guosheng Hu, Siqi Shen, Cheng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28545](https://doi.org/10.1609/aaai.v38i7.28545)

**Abstract**:

3D human pose estimation (3HPE) in large-scale outdoor scenes using commercial LiDAR has attracted significant attention due to its potential for real-life applications. However, existing LiDAR-based methods for 3HPE primarily rely on recovering 3D human poses from individual point clouds, and the coherence cues present in the neighborhood are not sufficiently harnessed. In this work, we explore spatial and contexture coherence cues contained in the neighborhood that lead to great performance improvements in 3HPE. Specifically, firstly, we deeply investigate the 3D neighbor in the background (3BN) which serves as a spatial coherence cue for inferring reliable motion since it provides physical laws to limit motion targets. Secondly, we introduce a novel 3D scanning neighbor (3SN) generated during the data collection and 3SN implies structural edge coherence cues. We use 3SN to overcome the degradation of performance and data quality caused by the sparsity-varying properties of LiDAR point clouds. In order to effectively model the complementation between these distinct cues and build consistent temporal relationships across human motions, we propose a new transformer-based module called the CoherenceFuse module. Extensive experiments were conducted on publicly available datasets, namely LidarHuman26M, CIMI4D, SLOPER4D and Waymo Open Dataset v2.0, showcase the superiority and effectiveness of our proposed method. In particular, when compared with LidarCap on the LidarHuman26M dataset, our method demonstrates a reduction of 7.08mm in the average MPJPE metric, along with a decrease of 16.55mm in the MPJPE metric for distances exceeding 25 meters. The code and models are available at https://github.com/jingyi-zhang/Neighborhood-enhanced-LidarCap.

----

## [797] NeRF-LiDAR: Generating Realistic LiDAR Point Clouds with Neural Radiance Fields

**Authors**: *Junge Zhang, Feihu Zhang, Shaochen Kuang, Li Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28546](https://doi.org/10.1609/aaai.v38i7.28546)

**Abstract**:

Labelling LiDAR point clouds for training autonomous driving is extremely expensive and difficult. LiDAR simulation aims at generating realistic LiDAR data with labels for training and verifying self-driving algorithms more efficiently. Recently, Neural Radiance Fields (NeRF) have been proposed for novel view synthesis  using implicit reconstruction of 3D scenes. Inspired by this, we present NeRF-LIDAR, a novel LiDAR simulation method that leverages real-world information to generate realistic LIDAR point clouds. Different from existing LiDAR simulators, we use real images and point cloud data collected by self-driving cars to learn the 3D scene representation, point cloud generation and label rendering. We verify the effectiveness of our NeRF-LiDAR  by training different 3D segmentation models on the generated LiDAR point clouds. 
It reveals that the trained models are able to achieve similar accuracy when compared with the same model trained on the real LiDAR data.  Besides, the generated data is capable of  boosting the accuracy through pre-training which helps reduce the requirements of the real labeled data. Code is available at https://github.com/fudan-zvg/NeRF-LiDAR

----

## [798] Point Cloud Part Editing: Segmentation, Generation, Assembly, and Selection

**Authors**: *Kaiyi Zhang, Yang Chen, Ximing Yang, Weizhong Zhang, Cheng Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28547](https://doi.org/10.1609/aaai.v38i7.28547)

**Abstract**:

Ideal part editing should guarantee the diversity of edited parts, the fidelity to the remaining parts, and the quality of the results. However, previous methods do not disentangle each part completely, which means the edited parts will affect the others, resulting in poor diversity and fidelity. In addition, some methods lack constraints between parts, which need manual selections of edited results to ensure quality. Therefore, we propose a four-stage process for point cloud part editing: Segmentation, Generation, Assembly, and Selection. Based on this process, we introduce SGAS, a model for part editing that employs two strategies: feature disentanglement and constraint. By independently fitting part-level feature distributions, we realize the feature disentanglement. By explicitly modeling the transformation from object-level distribution to part-level distributions, we realize the feature constraint. Considerable experiments on different datasets demonstrate the efficiency and effectiveness of SGAS on point cloud part editing. In addition, SGAS can be pruned to realize unsupervised part-aware point cloud generation and achieves state-of-the-art results.

----

## [799] CatmullRom Splines-Based Regression for Image Forgery Localization

**Authors**: *Li Zhang, Mingliang Xu, Dong Li, Jianming Du, Rujing Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i7.28548](https://doi.org/10.1609/aaai.v38i7.28548)

**Abstract**:

IFL (Image Forgery Location) helps secure digital media forensics. However, many methods suffer from false detections (i.e., FPs) and inaccurate boundaries. In this paper, we proposed the CatmullRom Splines-based Regression Network (CSR-Net), which first rethinks the IFL task from the perspective of regression to deal with this problem. Specifically speaking, we propose an adaptive CutmullRom splines fitting scheme for coarse localization of the tampered regions. Then, for false positive cases, we first develop a novel re-scoring mechanism, which aims to filter out samples that cannot have responses on both the classification branch and the instance branch. Later on, to further restrict the boundaries, we design a learnable texture extraction module, which refines and enhances the contour representation by decoupling the horizontal and vertical forgery features to extract a more robust contour representation, thus suppressing FPs. Compared to segmentation-based methods, our method is simple but effective due to the unnecessity of post-processing. Extensive experiments show the superiority of CSR-Net to existing state-of-the-art methods, not only on standard natural image datasets but also on social media datasets.

----



[Go to the previous page](AAAI-2024-list03.md)

[Go to the next page](AAAI-2024-list05.md)

[Go to the catalog section](README.md)