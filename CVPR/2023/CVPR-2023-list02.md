## [200] N-Gram in Swin Transformers for Efficient Lightweight Image Super-Resolution

        **Authors**: *Haram Choi, Jeongmin Lee, Jihoon Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00206](https://doi.org/10.1109/CVPR52729.2023.00206)

        **Abstract**:

        While some studies have proven that Swin Transformer (Swin) with window self-attention (WSA) is suitable for single image super-resolution (SR), the plain WSA ignores the broad regions when reconstructing high-resolution images due to a limited receptive field. In addition, many deep learning SR methods suffer from intensive computations. To address these problems, we introduce the N-Gram context to the low-level vision with Transformers for the first time. We define N-Gram as neighboring local windows in Swin, which differs from text analysis that views N-Gram as consecutive characters or words. N-Grams interact with each other by sliding-WSA, expanding the regions seen to restore degraded pixels. Using the N-Gram context, we propose NGswin, an efficient SR network with SCDP bottleneck taking multi-scale outputs of the hierarchical encoder. Experimental results show that NGswin achieves competitive performance while maintaining an efficient structure when compared with previous leading methods. Moreover, we also improve other Swin-based SR methods with the N-Gram context, thereby building an enhanced model: SwinIR-NG. Our improved SwinIR-NG out-performs the current best lightweight SR approaches and establishes state-of-the-art results. Codes are available at https://github.com/rami0205/NGramSwin.

        ----

        ## [201] Slide-Transformer: Hierarchical Vision Transformer with Local Self-Attention

        **Authors**: *Xuran Pan, Tianzhu Ye, Zhuofan Xia, Shiji Song, Gao Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00207](https://doi.org/10.1109/CVPR52729.2023.00207)

        **Abstract**:

        Self-attention mechanism has been a key factor in the recent progress of Vision Transformer (ViT), which enables adaptive feature extraction from global contexts. However, existing self-attention methods either adopt sparse global attention or window attention to reduce the computation complexity, which may compromise the local feature learning or subject to some handcrafted designs. In contrast, local attention, which restricts the receptive field of each query to its own neighboring pixels, enjoys the benefits of both convolution and self-attention, namely local inductive bias and dynamic feature selection. Nevertheless, current local attention modules either use inefficient Im2Col function or rely on specific CUDA kernels that are hard to generalize to devices without CUDA support. In this paper, we propose a novel local attention module, Slide Attention, which leverages common convolution operations to achieve high efficiency, flexibility and generalizability. Specifically, we first re-interpret the column-based Im2Col function from a new row-based perspective and use Depthwise Convolution as an efficient substitution. On this basis, we propose a deformed shifting module based on the re-parameterization technique, which further relaxes the fixed key/value positions to deformed features in the local region. In this way, our module realizes the local attention paradigm in both efficient and flexible manner. Extensive experiments show that our slide attention module is applicable to a variety of advanced Vision Transformer models and compatible with various hardware devices, and achieves consistently improved performances on comprehensive benchmarks.

        ----

        ## [202] Joint Token Pruning and Squeezing Towards More Aggressive Compression of Vision Transformers

        **Authors**: *Siyuan Wei, Tianzhu Ye, Shen Zhang, Yao Tang, Jiajun Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00208](https://doi.org/10.1109/CVPR52729.2023.00208)

        **Abstract**:

        Although vision transformers (ViTs) have shown promising results in various computer vision tasks recently, their high computational cost limits their practical applications. Previous approaches that prune redundant tokens have demonstrated a good trade-off between performance and computation costs. Nevertheless, errors caused by pruning strategies can lead to significant information loss. Our quantitative experiments reveal that the impact of pruned tokens on performance should be noticeable. To address this issue, we propose a novel joint Token Pruning & Squeezing module (TPS) for compressing vision transformers with higher efficiency. Firstly, TPS adopts pruning to get the reserved and pruned subsets. Secondly, TPS squeezes the information of pruned tokens into partial reserved tokens via the unidirectional nearest-neighbor matching and similarity-based fusing steps. Compared to state-of-the-art methods, our approach outperforms them under all token pruning intensities. Especially while shrinking DeiT-tiny&small computational budgets to 35%, it improves the accuracy by 1%-6% compared with baselines on ImageNet classification. The proposed method can accelerate the throughput of DeiT-small beyond DeiT-tiny, while its accuracy surpasses DeiT-tiny by 4.78%. Experiments on various transformers demonstrate the effectiveness of our method, while analysis experiments prove our higher robustness to the errors of the token pruning policy. Code is available at https://github.com/megvii-research/TPS-CVPR2023.

        ----

        ## [203] Top-Down Visual Attention from Analysis by Synthesis

        **Authors**: *Baifeng Shi, Trevor Darrell, Xin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00209](https://doi.org/10.1109/CVPR52729.2023.00209)

        **Abstract**:

        Current attention algorithms (e.g., self-attention) are stimulus-driven and highlight all the salient objects in an image. However, intelligent agents like humans often guide their attention based on the high-level task at hand, focusing only on task-related objects. This ability of task-guided top-down attention provides task-adaptive representation and helps the model generalize to various tasks. In this paper, we consider top-down attention from a classic Analysis-by-Synthesis (AbS) perspective of vision. Prior work indicates a functional equivalence between visual attention and sparse reconstruction; we show that an AbS visual system that optimizes a similar sparse reconstruction objective modulated by a goal-directed top-down signal naturally simulates top-down attention. We further propose Analysis-by-Synthesis Vision Transformer (AbSViT), which is a top-down modulated ViT model that variationally approximates AbS, and achieves controllable top-down attention. For real-world applications, AbSViT consistently improves over baselines on Vision-Language tasks such as VQA and zero-shot retrieval where language guides the top-down attention. AbSViT can also serve as a general backbone, improving performance on classification, semantic segmentation, and model robustness. Project page: https://sites.google.com/view/absvit.

        ----

        ## [204] Probing Neural Representations of Scene Perception in a Hippocampally Dependent Task Using Artificial Neural Networks

        **Authors**: *Markus Frey, Christian F. Doeller, Caswell Barry*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00210](https://doi.org/10.1109/CVPR52729.2023.00210)

        **Abstract**:

        Deep artificial neural networks (DNNs) trained through back propagation provide effective models of the mammalian visual system, accurately capturing the hierarchy of neural responses through primary visual cortex to inferior temporal cortex (IT) [41, 43]. However, the ability of these networks to explain representations in higher cortical areas is relatively lacking and considerably less well researched. For example, DNNs have been less successful as a model of the egocentric to allocentric transformation embodied by circuits in retrosplenial and posterior parietal cortex. We describe a novel scene perception benchmark inspired by a hippocampal dependent task, designed to probe the ability of DNNs to transform scenes viewed from different egocentric perspectives. Using a network architecture inspired by the connectivity between temporal lobe structures and the hippocampus, we demonstrate that DNNs trained using a triplet loss can learn this task. Moreover, by enforcing a factorized latent space, we can split information propagation into “what” and “wdere” pathways, which we use to reconstruct the input. This allows us to beat the state-of-the-art for unsupervised object segmentation on the CATER and MOVi-A, B, C benchmarks.

        ----

        ## [205] Masked Image Modeling with Local Multi-Scale Reconstruction

        **Authors**: *Haoqing Wang, Yehui Tang, Yunhe Wang, Jianyuan Guo, Zhi-Hong Deng, Kai Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00211](https://doi.org/10.1109/CVPR52729.2023.00211)

        **Abstract**:

        Masked Image Modeling (MIM) achieves outstanding success in self-supervised representation learning. Unfortunately, MIM models typically have huge computational burden and slow learning process, which is an inevitable obstacle for their industrial applications. Although the lower layers play the key role in MIM, existing MIM models conduct reconstruction task only at the top layer of encoder. The lower layers are not explicitly guided and the interaction among their patches is only used for calculating new activations. Considering the reconstruction task requires non-trivial inter-patch interactions to reason target signals, we apply it to multiple local layers including lower and upper layers. Further, since the multiple layers expect to learn the information of different scales, we design local multi-scale reconstruction, where the lower and upper layers reconstruct fine-scale and coarse-scale supervision signals respectively. This design not only accelerates the representation learning process by explicitly guiding multiple layers, but also facilitates multi-scale semantical understanding to the input. Extensive experiments show that with significantly less pre-training burden, our model achieves comparable or better performance on classification, detection and segmentation tasks than existing MIM models. Code is available with both MindSpore and PyTorch.

        ----

        ## [206] Siamese Image Modeling for Self-Supervised Vision Representation Learning

        **Authors**: *Chenxin Tao, Xizhou Zhu, Weijie Su, Gao Huang, Bin Li, Jie Zhou, Yu Qiao, Xiaogang Wang, Jifeng Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00212](https://doi.org/10.1109/CVPR52729.2023.00212)

        **Abstract**:

        Self-supervised learning (SSL) has delivered superior performance on a variety of downstream vision tasks. Two main-stream SSL frameworks have been proposed, i.e., Instance Discrimination (ID) and Masked Image Modeling (MIM). ID pulls together representations from different views of the same image, while avoiding feature collapse. It lacks spatial sensitivity, which requires modeling the local structure within each image. On the other hand, MIM reconstructs the original content given a masked image. It instead does not have good semantic alignment, which requires projecting semantically similar views into nearby representations. To address this dilemma, we observe that (1) semantic alignment can be achieved by matching different image views with strong augmentations; (2) spatial sensitivity can benefit from predicting dense representations with masked images. Driven by these analysis, we propose Siamese Image Modeling (SiameseIM), which predicts the dense representations of an augmented view, based on another masked view from the same image but with different augmentations. SiameseIM uses a Siamese network with two branches. The online branch encodes the first view, and predicts the second view's representation according to the relative positions between these two views. The target branch produces the target by encoding the second view. SiameseIM can surpass both ID and MIM on a wide range of downstream tasks, including ImageNet finetuning and linear probing, COCO and LVIS detection, and ADE20k semantic segmentation. The improvement is more significant in few-shot, long-tail and robustness-concerned scenarios. Code shall be released.

        ----

        ## [207] MAGE: MAsked Generative Encoder to Unify Representation Learning and Image Synthesis

        **Authors**: *Tianhong Li, Huiwen Chang, Shlok Kumar Mishra, Han Zhang, Dina Katabi, Dilip Krishnan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00213](https://doi.org/10.1109/CVPR52729.2023.00213)

        **Abstract**:

        Generative modeling and representation learning are two key tasks in computer vision. However, these models are typically trained independently, which ignores the potential for each task to help the other, and leads to training and model maintenance overheads. In this work, we propose MAsked Generative Encoder (MAGE), the first framework to unify SOTA image generation and self-supervised representation learning. Our key insight is that using variable masking ratios in masked image modeling pre-training can allow generative training (very high masking ratio) and representation learning (lower masking ratio) under the same training framework. Inspired by previous generative models, MAGE uses semantic tokens learned by a vector-quantized GAN at inputs and outputs, combining this with masking. We can further improve the representation by adding a contrastive loss to the encoder output. We extensively evaluate the generation and representation learning capabilities of MAGE. On ImageNet-1K, a single MAGE ViT-L model obtains 9.10 FID in the task of class-unconditional image generation and 78.9% top-1 accuracy for linear probing, achieving state-of-the-art performance in both image generation and representation learning. Code is available at https://github.com/LTHl4/mage.

        ----

        ## [208] Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-identification

        **Authors**: *Yukang Zhang, Hanzi Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00214](https://doi.org/10.1109/CVPR52729.2023.00214)

        **Abstract**:

        For the visible-infrared person re-identification (VIReID) task, one of the major challenges is the modality gaps between visible (VIS) and infrared (IR) images. However, the training samples are usually limited, while the modality gaps are too large, which leads that the existing methods cannot effectively mine diverse cross-modality clues. To handle this limitation, we propose a novel augmentation network in the embedding space, called diverse embedding expansion network (DEEN). The proposed DEEN can effectively generate diverse embeddings to learn the informative feature representations and reduce the modality discrepancy between the VIS and IR images. Moreover, the VIReID model may be seriously affected by drastic illumination changes, while all the existing VIReID datasets are captured under sufficient illumination without significant light changes. Thus, we provide a low-light cross-modality (LLCM) dataset, which contains 46,767 bounding boxes of 1,064 identities captured by 9 RGB/IR cameras. Extensive experiments on the SYSU-MM01, RegDB and LLCM datasets show the superiority of the proposed DEEN over several other state-of-the-art methods. The code and dataset are released at: https://github.com/ZYK100/LLCM

        ----

        ## [209] DistilPose: Tokenized Pose Regression with Heatmap Distillation

        **Authors**: *Suhang Ye, Yingyi Zhang, Jie Hu, Liujuan Cao, Shengchuan Zhang, Lei Shen, Jun Wang, Shouhong Ding, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00215](https://doi.org/10.1109/CVPR52729.2023.00215)

        **Abstract**:

        In the field of human pose estimation, regression-based methods have been dominated in terms of speed, while heatmap-based methods are far ahead in terms of performance. How to take advantage of both schemes remains a challenging problem. In this paper, we propose a novel human pose estimation framework termed DistilPose, which bridges the gaps between heatmap-based and regression-based methods. Specifically, DistilPose maximizes the transfer of knowledge from the teacher model (heatmap-based) to the student model (regression-based) through Token-distilling Encoder (TDE) and Simulated Heatmaps. TDE aligns the feature spaces of heatmap-based and regression-based models by introducing tokenization, while Simulated Heatmaps transfer explicit guidance (distribution and confidence) from teacher heatmaps into student models. Extensive experiments show that the proposed DistilPose can significantly improve the performance of the regression-based models while maintaining efficiency. Specifically, on the MSCOCO validation dataset, DistilPose-S obtains 71.6% mAP with 5.36M parameters, 2.38 GFLOPs, and 40.2 FPS, which saves 12.95×, 7.16× computational cost and is 4.9× faster than its teacher model with only 0.9 points performance drop. Furthermore, DistilPose-L obtains 74.4% mAP on MSCOCO validation dataset, achieving a new state-of-the-art among predominant regression-based models. Code will be available at https://github.com/yshMars/DistilPose.

        ----

        ## [210] Graph Transformer GANs for Graph-Constrained House Generation

        **Authors**: *Hao Tang, Zhenyu Zhang, Humphrey Shi, Bo Li, Ling Shao, Nicu Sebe, Radu Timofte, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00216](https://doi.org/10.1109/CVPR52729.2023.00216)

        **Abstract**:

        We present a novel graph Transformer generative adversarial network (GTGAN) to learn effective graph node relations in an end-to-end fashion for the challenging graph-constrained house generation task. The proposed graph-Transformer-based generator includes a novel graph Transformer encoder that combines graph convolutions and self-attentions in a Transformer to model both local and global interactions across connected and non-connected graph nodes. Specifically, the proposed connected node attention (CNA) and non-connected node attention (NNA) aim to capture the global relations across connected nodes and non-connected nodes in the input graph, respectively. The proposed graph modeling block (GMB) aims to exploit local vertex interactions based on a house layout topology. More-over, we propose a new node classification-based discriminator to preserve the high-level semantic and discriminative node features for different house components. Finally, we propose a novel graph-based cycle-consistency loss that aims at maintaining the relative spatial relationships between ground truth and predicted graphs. Experiments on two challenging graph-constrained house generation tasks (i.e., house layout and roof generation) with two public datasets demonstrate the effectiveness of GTGAN in terms of objective quantitative scores and subjective visual realism. New state-of-the-art results are established by large margins on both tasks.

        ----

        ## [211] Automatic High Resolution Wire Segmentation and Removal

        **Authors**: *Mang Tik Chiu, Xuaner Zhang, Zijun Wei, Yuqian Zhou, Eli Shechtman, Connelly Barnes, Zhe Lin, Florian Kainz, Sohrab Amirghodsi, Humphrey Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00217](https://doi.org/10.1109/CVPR52729.2023.00217)

        **Abstract**:

        Wires and powerlines are common visual distractions that often undermine the aesthetics of photographs. The manual process of precisely segmenting and removing them is extremely tedious and may take up hours, especially on high-resolution photos where wires may span the entire space. In this paper, we present an automatic wire clean-up system that eases the process of wire segmentation and removal/inpainting to within a few seconds. We observe several unique challenges: wires are thin, lengthy, and sparse. These are rare properties of subjects that common segmentation tasks cannot handle, especially in high-resolution images. We thus propose a two-stage method that leverages both global and local contexts to accurately segment wires in high-resolution images efficiently, and a tile-based inpainting strategy to remove the wires given our predicted segmentation masks. We also introduce the first wire segmentation benchmark dataset, WireSegHR. Finally, we demonstrate quantitatively and qualitatively that our wire clean-up system enables fully automated wire removal with great generalization to various wire appearances.

        ----

        ## [212] Tree Instance Segmentation with Temporal Contour Graph

        **Authors**: *Adnan Firoze, Cameron Wingren, Raymond A. Yeh, Bedrich Benes, Daniel G. Aliaga*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00218](https://doi.org/10.1109/CVPR52729.2023.00218)

        **Abstract**:

        We present a novel approach to perform instance segmentation and counting for densely packed self-similar trees using a top-view RGB image sequence. We propose a solution that leverages pixel content, shape, and self-occlusion. First, we perform an initial over-segmentation of the image sequence and aggregate structural characteristics into a contour graph with temporal information incorporated. Second, using a graph convolutional network and its inherent local messaging passing abilities, we merge adjacent tree crown patches into a final set of tree crowns. Per various studies and comparisons, our method is superior to all prior methods and results in high-accuracy instance segmentation and counting despite the trees being tightly packed. Finally, we provide various forest image sequence datasets suitable for subsequent benchmarking and evaluation captured at different altitudes and leaf conditions.

        ----

        ## [213] Dual-Path Adaptation from Image to Video Transformers

        **Authors**: *Jungin Park, Jiyoung Lee, Kwanghoon Sohn*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00219](https://doi.org/10.1109/CVPR52729.2023.00219)

        **Abstract**:

        In this paper, we efficiently transfer the surpassing representation power of the vision foundation models, such as ViT and Swin, for video understanding with only a few trainable parameters. Previous adaptation methods have simultaneously considered spatial and temporal modeling with a unified learnable module but still suffered from fully leveraging the representative capabilities of image transformers. We argue that the popular dual-path (two-stream) architecture in video models can mitigate this problem. We propose a novel DualPath adaptation separated into spatial and temporal adaptation paths, where a lightweight bottleneck adapter is employed in each transformer block. Especially for temporal dynamic modeling, we incorporate consecutive frames into a grid-like frameset to precisely imitate vision transformers' capability that extrapolates relationships between tokens. In addition, we extensively investigate the multiple baselines from a unified perspective in video understanding and compare them with DualPath. Experimental results on four action recognition benchmarks prove that pretrained image transformers with DualPath can be effectively generalized beyond the data domain.

        ----

        ## [214] Rethinking Video ViTs: Sparse Video Tubes for Joint Image and Video Learning

        **Authors**: *A. J. Piergiovanni, Weicheng Kuo, Anelia Angelova*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00220](https://doi.org/10.1109/CVPR52729.2023.00220)

        **Abstract**:

        We present a simple approach which can turn a ViT en-coder into an efficient video model, which can seamlessly work with both image and video inputs. By sparsely sam-pling the inputs, the model is able to do training and in-ference from both input modalities. The model is easily scalable and can be adapted to large-scale pre-trained ViTs without requiring full finetuning. The model achieves SOTA results
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://sites.google.com/view/tubevit.

        ----

        ## [215] Modeling Video as Stochastic Processes for Fine-Grained Video Representation Learning

        **Authors**: *Heng Zhang, Daqing Liu, Qi Zheng, Bing Su*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00221](https://doi.org/10.1109/CVPR52729.2023.00221)

        **Abstract**:

        A meaningful video is semantically coherent and changes smoothly. However, most existing fine-grained video representation learning methods learn frame-wise features by aligning frames across videos or exploring relevance between multiple views, neglecting the inherent dynamic process of each video. In this paper, we propose to learn video representations by modeling Video as Stochastic Processes (VSP) via a novel process-based contrastive learning framework, which aims to discriminate between video processes and simultaneously capture the temporal dynamics in the processes. Specifically, we enforce the embeddings of the frame sequence of interest to approximate a goal-oriented stochastic process, i.e., Brownian bridge, in the latent space via a process-based contrastive loss. To construct the Brownian bridge, we adapt specialized sampling strategies under different annotations for both self-supervised and weakly-supervised learning. Experimental results on four datasets show that VSP stands as a state-of-the-art method for various video understanding tasks, including phase progression, phase classification, and frame retrieval. Code is available at https://github.com/hengRUC/VSP.

        ----

        ## [216] Masked Motion Encoding for Self-Supervised Video Representation Learning

        **Authors**: *Xinyu Sun, Peihao Chen, Liangwei Chen, Changhao Li, Thomas H. Li, Mingkui Tan, Chuang Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00222](https://doi.org/10.1109/CVPR52729.2023.00222)

        **Abstract**:

        How to learn discriminative video representation from unlabeled videos is challenging but crucial for video analysis. The latest attempts seek to learn a representation model by predicting the appearance contents in the masked regions. However, simply masking and recovering appearance contents may not be sufficient to model temporal clues as the appearance contents can be easily reconstructed from a single frame. To overcome this limitation, we present Masked Motion Encoding (MME), a new pretraining paradigm that reconstructs both appearance and motion information to explore temporal clues. In MME, we focus on addressing two critical challenges to improve the representation performance: 1) how to well represent the possible long-term motion across multiple frames; and 2) how to obtain fine-grained temporal clues from sparsely sampled videos. Motivated by the fact that human is able to recognize an action by tracking objects' position changes and shape changes, we propose to reconstruct a motion trajectory that represents these two kinds of change in the masked regions. Besides, given the sparse video input, we enforce the model to reconstruct dense motion trajectories in both spatial and temporal dimensions. Pre-trained with our MME paradigm, the model is able to anticipate long-term and fine-grained motion details. Code is available at https://github.com/XinyuSun/MME.

        ----

        ## [217] Boosting Video Object Segmentation via Space-Time Correspondence Learning

        **Authors**: *Yurong Zhang, Liulei Li, Wenguan Wang, Rong Xie, Li Song, Wenjun Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00223](https://doi.org/10.1109/CVPR52729.2023.00223)

        **Abstract**:

        Current top-leading solutions for video object segmentation (VOS) typically follow a matching-based regime: for each query frame, the segmentation mask is inferred according to its correspondence to previously processed and the first annotated frames. They simply exploit the supervisory signals from the groundtruth masks for learning mask prediction only, without posing any constraint on the space-time correspondence matching, which, however, is the fundamental building block of such regime. To alleviate this crucial yet commonly ignored issue, we devise a correspondence-aware training framework, which boosts matching-based VOS solutions by explicitly encouraging robust correspondence matching during network learning. Through comprehensively exploring the intrinsic coherence in videos on pixel and object levels, our algorithm reinforces the standard, fully supervised training of mask segmentation with label-free, contrastive correspondence learning. Without neither requiring extra annotation cost during training, nor causing speed delay during deployment, nor incurring architectural modification, our algorithm provides solid performance gains on four widely used benchmarks, i.e., DAVIS2016&2017, and YouTube-VOS2018&2019, on the top of famous matching-based VOS solutions.

        ----

        ## [218] Two-shot Video Object Segmentation

        **Authors**: *Kun Yan, Xiao Li, Fangyun Wei, Jinglu Wang, Chenbin Zhang, Ping Wang, Yan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00224](https://doi.org/10.1109/CVPR52729.2023.00224)

        **Abstract**:

        Previous works on video object segmentation (VOS) are trained on densely annotated videos. Nevertheless, acquiring annotations in pixel level is expensive and time-consuming. In this work, we demonstrate the feasibility of training a satisfactory VOS model on sparsely annotated videos—we merely require two labeled frames per training video while the performance is sustained. We term this novel training paradigm as two-shot video object segmentation, or two-shot VOS for short. The underlying idea is to generate pseudo labels for unlabeled frames during training and to optimize the model on the combination of labeled and pseudo-labeled data. Our approach is extremely simple and can be applied to a majority of existing frameworks. We first pre-train a VOS model on sparsely annotated videos in a semi-supervised manner, with the first frame always being a labeled one. Then, we adopt the pretrained VOS model to generate pseudo labels for all unlabeled frames, which are subsequently stored in a pseudo-label bank. Finally, we retrain a VOS model on both labeled and pseudo-labeled data without any restrictions on the first frame. For the first time, we present a general way to train VOS models on two-shot VOS datasets. By using 7.3% and 2.9% labeled data of YouTube-VOS and DAVIS benchmarks, our approach achieves comparable results in contrast to the counterparts trained on fully labeled set. Code and models are available at https://github.com/yk-pku/Two-shot-Video-Object-Segmentation.

        ----

        ## [219] Look Before You Match: Instance Understanding Matters in Video Object Segmentation

        **Authors**: *Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Chuanxin Tang, Xiyang Dai, Yucheng Zhao, Yujia Xie, Lu Yuan, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00225](https://doi.org/10.1109/CVPR52729.2023.00225)

        **Abstract**:

        Exploring dense matching between the current frame and past frames for long-range context modeling, memory-based methods have demonstrated impressive results in video object segmentation (VOS) recently. Nevertheless, due to the lack of instance understanding ability, the above approaches are oftentimes brittle to large appearance variations or viewpoint changes resulted from the movement of objects and cameras. In this paper, we argue that instance understanding matters in VOS, and integrating it with memory-based matching can enjoy the synergy, which is intuitively sensible from the definition of VOS task, i.e., identifying and segmenting object instances within the video. Towards this goal, we present a two-branch network for VOS, where the query-based instance segmentation (IS) branch delves into the instance details of the current frame and the VOS branch performs spatial-temporal matching with the memory bank. We employ the well-learned object queries from IS branch to inject instance-specific information into the query key, with which the instance-augmented matching is further performed. In addition, we introduce a multi-path fusion block to effectively combine the memory readout with multi-scale features from the instance segmentation decoder, which incorporates high-resolution instance-aware features to produce final segmentation results. Our method achieves state-of-the-art performance on DAVIS 2016/2017 val (92.6% and 87.1%), DAVIS 2017 test-dev (82.8%), and YouTube-VOS 2018/2019 val (86.3% and 86.3%), outperforming alternative methods by clear margins.

        ----

        ## [220] Spatial-then-Temporal Self-Supervised Learning for Video Correspondence

        **Authors**: *Rui Li, Dong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00226](https://doi.org/10.1109/CVPR52729.2023.00226)

        **Abstract**:

        In low-level video analyses, effective representations are important to derive the correspondences between video frames. These representations have been learned in a selfsupervised fashion from unlabeled images or videos, using carefully designed pretext tasks in some recent studies. However, the previous work concentrates on either spatial-discriminative features or temporal-repetitive features, with little attention to the synergy between spatial and temporal cues. To address this issue, we propose a spatial-then-temporal self-supervised learning method. Specifically, we firstly extract spatial features from unlabeled images via contrastive learning, and secondly enhance the features by exploiting the temporal cues in unlabeled videos via reconstructive learning. In the second step, we design a global correlation distillation loss to ensure the learning not to forget the spatial cues, and a local correlation distillation loss to combat the temporal discontinuity that harms the reconstruction. The proposed method outperforms the state-of-the-art self-supervised methods, as established by the experimental results on a series of correspondence-based video analysis tasks. Also, we performed ablation studies to verify the effectiveness of the two-step design as well as the distillation losses.

        ----

        ## [221] Few-Shot Referring Relationships in Videos

        **Authors**: *Yogesh Kumar, Anand Mishra*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00227](https://doi.org/10.1109/CVPR52729.2023.00227)

        **Abstract**:

        Interpreting visual relationships is a core aspect of comprehensive video understanding. Given a query visual relationship as <subject, predicate, object> and a test video, our objective is to localize the subject and object that are connected via the predicate. Given modern visio-lingual understanding capabilities, solving this problem is achievable, provided that there are large-scale annotated training examples available. However, annotating for every combination of subject, object, and predicate is cumbersome, expensive, and possibly infeasible. Therefore, there is a need for models that can learn to spatially and temporally localize subjects and objects that are connected via an unseen predicate using only a few support set videos sharing the common predicate. We address this challenging problem, referred to as few-shot referring relationships in videos for the first time. To this end, we pose the problem as a minimization of an objective function defined over a T-partite random field. Here, the vertices of the random field correspond to candidate bounding boxes for the subject and object, and T represents the number of frames in the test video. This objective function is composed of frame-level and visual relationship similarity potentials. To learn these potentials, we use a relation network that takes query-conditioned translational relationship embedding as inputs and is meta-trained using support set videos in an episodic manner. Further, the objective function is minimized using a belief propagation-based message passing on the random field to obtain the spatiotemporal localization or subject and object trajectories. We perform extensive experiments using two public benchmarks, namely ImageNet-VidVRD and VidOR, and compare the proposed approach with competitive baselines to assess its efficacy.

        ----

        ## [222] Vision Transformers are Parameter-Efficient Audio-Visual Learners

        **Authors**: *Yan-Bo Lin, Yi-Lin Sung, Jie Lei, Mohit Bansal, Gedas Bertasius*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00228](https://doi.org/10.1109/CVPR52729.2023.00228)

        **Abstract**:

        Vision transformers (ViTs) have achieved impressive results on various computer vision tasks in the last several years. In this work, we study the capability of frozen ViTs, pretrained only on visual data, to generalize to audio-visual data without finetuning any of its original parameters. To do so, we propose a latent audio-visual hybrid (LAVISH) adapter that adapts pretrained ViTs to audio-visual tasks by injecting a small number of trainable parameters into every layer of a frozen ViT. To efficiently fuse visual and audio cues, our LAVISH adapter uses a small set of latent tokens, which form an attention bottleneck, thus, eliminating the quadratic cost of standard cross-attention. Compared to the existing modality-specific audio-visual methods, our approach achieves competitive or even better performance on various audio-visual tasks while using fewer tunable parameters and without relying on costly audio pretraining or external audio encoders. Our code is available at https://genjib.github.io/project_page/LAVISH/

        ----

        ## [223] Egocentric Video Task Translation

        **Authors**: *Zihui Xue, Yale Song, Kristen Grauman, Lorenzo Torresani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00229](https://doi.org/10.1109/CVPR52729.2023.00229)

        **Abstract**:

        Different video understanding tasks are typically treated in isolation, and even with distinct types of curated data (e.g., classifying sports in one dataset, tracking animals in another). However, in wearable cameras, the immersive egocentric perspective of a person engaging with the world around them presents an interconnected web of video understanding tasks—hand-object manipulations, navigation in the space, or human-human interactions—that unfold continuously, driven by the person's goals. We argue that this calls for a much more unified approach. We propose EgoTask Translation (EgoT2), which takes a collection of models optimized on separate tasks and learns to translate their outputs for improved performance on any or all of them at once. Unlike traditional transfer or multi-task learning, EgoT2's “flipped design” entails separate task-specific backbones and a task translator shared across all tasks, which captures synergies between even heterogeneous tasks and mitigates task competition. Demonstrating our model on a wide array of video tasks from Ego4D, we show its advantages over existing transfer paradigms and achieve top-ranked results on four of the Ego4D 2022 benchmark challenges.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project webpage: https://vision.cs.utexas.edu/projects/egot2/.

        ----

        ## [224] QPGesture: Quantization-Based and Phase-Guided Motion Matching for Natural Speech-Driven Gesture Generation

        **Authors**: *Sicheng Yang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Lei Hao, Weihong Bao, Haolin Zhuang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00230](https://doi.org/10.1109/CVPR52729.2023.00230)

        **Abstract**:

        Speech-driven gesture generation is highly challenging due to the random jitters of human motion. In addition, there is an inherent asynchronous relationship between human speech and gestures. To tackle these challenges, we introduce a novel quantization-based and phase-guided motion matching framework. Specifically, we first present a gesture VQ-VAE module to learn a codebook to summarize meaningful gesture units. With each code representing a unique gesture, random jittering problems are alleviated effectively. We then use Levenshtein distance to align diverse gestures with different speech. Levenshtein distance based on audio quantization as a similarity metric of corresponding speech of gestures helps match more appropriate gestures with speech, and solves the alignment problem of speech and gestures well. Moreover, we introduce phase to guide the optimal gesture matching based on the semantics of context or rhythm of audio. Phase guides when text-based or speech-based gestures should be performed to make the generated gestures more natural. Extensive experiments show that our method outperforms recent approaches on speech-driven gesture generation. Our code, database, pre-trained models and demos are available at https://github.com/YoungSeng/QPGesture.

        ----

        ## [225] Co-speech Gesture Synthesis by Reinforcement Learning with Contrastive Pretrained Rewards

        **Authors**: *Mingyang Sun, Mengchen Zhao, Yaqing Hou, Minglei Li, Huang Xu, Songcen Xu, Jianye Hao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00231](https://doi.org/10.1109/CVPR52729.2023.00231)

        **Abstract**:

        There is a growing demand of automatically synthesizing co-speech gestures for virtual characters. However, it remains a challenge due to the complex relationship between input speeches and target gestures. Most existing works focus on predicting the next gesture that fits the data best, however, such methods are myopic and lack the ability to plan for future gestures. In this paper, we propose a novel reinforcement learning (RL) framework called RACER to generate sequences of gestures that maximize the overall satisfactory. RACER employs a vector quantized variational autoencoder to learn compact representations of gestures and a GPT-based policy architecture to generate coherent sequence of gestures autoregressively. In particular, we propose a contrastive pre-training approach to calculate the rewards, which integrates contextual information into action evaluation and successfully captures the complex relationships between multi-modal speech-gesture data. Experimental results show that our method significantly outperforms existing baselines in terms of both objective metrics and subjective human judgements. Demos can be found at https://github.com/RLracer/RACER.git.

        ----

        ## [226] TimeBalance: Temporally-Invariant and Temporally-Distinctive Video Representations for Semi-Supervised Action Recognition

        **Authors**: *Ishan Rajendrakumar Dave, Mamshad Nayeem Rizve, Chen Chen, Mubarak Shah*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00232](https://doi.org/10.1109/CVPR52729.2023.00232)

        **Abstract**:

        Semi-Supervised Learning can be more beneficial for the video domain compared to images because of its higher an-notation cost and dimensionality. Besides, any video understanding task requires reasoning over both spatial and temporal dimensions. In order to learn both the static and motion related features for the semi-supervised action recognition task, existing methods rely on hard in-put inductive biases like using two-modalities (RGB and Optical-flow) or two-stream of different playback rates. Instead of utilizing unlabeled videos through diverse in-put streams, we rely on self-supervised video represen-tations, particularly, we utilize temporally-invariant and temporally-distinctive representations. We observe that these representations complement each other depending on the nature of the action. Based on this observation, we propose a student-teacher semi-supervised learning frame-work, TimeBalance, where we distill the knowledge from a temporally-invariant and a temporally-distinctive teacher. Depending on the nature of the unlabeled video, we dy-namically combine the knowledge of these two teach-ers based on a novel temporal similarity-based reweighting scheme. Our method achieves state-of-the-art performance on three action recognition benchmarks: UCF101, HMDB51, and Kinetics400. Code: https://github.com/DAVEISHAN/TimeBalance.

        ----

        ## [227] How can objects help action recognition?

        **Authors**: *Xingyi Zhou, Anurag Arnab, Chen Sun, Cordelia Schmid*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00233](https://doi.org/10.1109/CVPR52729.2023.00233)

        **Abstract**:

        Current state-of-the-art video models process a video clip as a long sequence of spatio-temporal tokens. However, they do not explicitly model objects, their interactions across the video, and instead process all the tokens in the video. In this paper, we investigate how we can use knowledge of objects to design better video models, namely to process fewer tokens and to improve recognition accuracy. This is in contrast to prior works which either drop tokens at the cost of accuracy, or increase accuracy whilst also increasing the computation required. First, we propose an object-guided token sampling strategy that enables us to retain a small fraction of the input tokens with minimal impact on accuracy. And second, we propose an object-aware attention module that enriches our feature representation with object information and improves overall accuracy. Our resulting model, ObjectViViT, achieves better performance when using fewer tokens than strong baselines. In particular, we match our baseline with 30%, 40%, and 60% of the input tokens on SomethingElse, Something-something v2, and Epic-Kitchens, respectively. When we use Object-ViViT to process the same number of tokens as our baseline, we improve by 0.6 to 4.2 points on these datasets.

        ----

        ## [228] Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition

        **Authors**: *Lilang Lin, Jiahang Zhang, Jiaying Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00234](https://doi.org/10.1109/CVPR52729.2023.00234)

        **Abstract**:

        The self-supervised pretraining paradigm has achieved great success in skeleton-based action recognition. However, these methods treat the motion and static parts equally, and lack an adaptive design for different parts, which has a negative impact on the accuracy of action recognition. To realize the adaptive action modeling of both parts, we propose an Actionlet-Dependent Contrastive Learning method (ActCLR). The actionlet, defined as the discriminative subset of the human skeleton, effectively decomposes motion regions for better action modeling. In detail, by contrasting with the static anchor without motion, we extract the motion region of the skeleton data, which serves as the actionlet, in an unsupervised manner. Then, centering on actionlet, a motion-adaptive data transformation method is built. Different data transformations are applied to action let and non-actionlet regions to introduce more diversity while maintaining their own characteristics. Meanwhile, we propose a semantic-aware feature pooling method to build feature representations among motion and static regions in a distinguished manner. Extensive experiments on NTU RGB+D and PKUMMD show that the proposed method achieves remarkable action recognition performance. More visualization and quantitative experiments demonstrate the effectiveness of our method. Our project website is available at https://langlandslin.github.io/projects/ActCLR/

        ----

        ## [229] Decomposed Cross-Modal Distillation for RGB-based Temporal Action Detection

        **Authors**: *Pilhyeon Lee, Taeoh Kim, Minho Shim, Dongyoon Wee, Hyeran Byun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00235](https://doi.org/10.1109/CVPR52729.2023.00235)

        **Abstract**:

        Temporal action detection aims to predict the time intervals and the classes of action instances in the video. Despite the promising performance, existing two-stream models exhibit slow inference speed due to their reliance on computationally expensive optical flow. In this paper, we introduce a decomposed cross-modal distillation framework to build a strong RGB-based detector by transferring knowledge of the motion modality. Specifically, instead of direct distillation, we propose to separately learn RGB and motion representations, which are in turn combined to perform action localization. The dual-branch design and the asymmetric training objectives enable effective motion knowledge transfer while preserving RGB information intact. In addition, we introduce a local attentive fusion to better exploit the multimodal complementarity. It is designed to preserve the local discriminability of the features that is important for action localization. Extensive experiments on the benchmarks verify the effectiveness of the proposed method in enhancing RGB-based action detectors. Notably, our framework is agnostic to backbones and detection heads, bringing consistent gains across different model combinations.

        ----

        ## [230] ASPnet: Action Segmentation with Shared-Private Representation of Multiple Data Sources

        **Authors**: *Beatrice van Amsterdam, Abdolrahim Kadkhodamohammadi, Imanol Luengo, Danail Stoyanov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00236](https://doi.org/10.1109/CVPR52729.2023.00236)

        **Abstract**:

        Most state-of-the-art methods for action segmentation are based on single input modalities or na誰ve fusion of multiple data sources. However, effective fusion of complementary information can potentially strengthen segmentation models and make them more robust to sensor noise and more accurate with smaller training datasets. In order to improve multimodal representation learning for action segmentation, we propose to disentangle hidden features of a multi-stream segmentation model into modality-shared components, containing common information across data sources, and private components; we then use an attention bottleneck to capture long-range temporal dependencies in the data while preserving disentanglement in consecutive processing layers. Evaluation on 50salads, Breakfast and RARP45 datasets shows that our multimodal approach outperforms different data fusion baselines on both multiview and multimodal data sources, obtaining competitive or better results compared with the state-of-the-art. Our model is also more robust to additive sensor noise and can achieve performance on par with strong video baselines even with less training data.

        ----

        ## [231] Proposal-Based Multiple Instance Learning for Weakly-Supervised Temporal Action Localization

        **Authors**: *Huan Ren, Wenfei Yang, Tianzhu Zhang, Yongdong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00237](https://doi.org/10.1109/CVPR52729.2023.00237)

        **Abstract**:

        Weakly-supervised temporal action localization aims to localize and recognize actions in untrimmed videos with only video-level category labels during training. Without instance-level annotations, most existing methods follow the Segment-based Multiple Instance Learning (S-MIL) framework, where the predictions of segments are supervised by the labels of videos. However, the objective for acquiring segment-level scores during training is not consistent with the target for acquiring proposal-level scores during testing, leading to suboptimal results. To deal with this problem, we propose a novel Proposal-based Multiple Instance Learning (P-MIL) framework that directly classifies the candidate proposals in both the training and testing stages, which includes three key designs: 1) a surrounding contrastive feature extraction module to suppress the discriminative short proposals by considering the surrounding contrastive information, 2) a proposal completeness evaluation module to inhibit the low-quality proposals with the guidance of the completeness pseudo labels, and 3) an instance-level rank consistency loss to achieve robust detection by leveraging the complementarity of RGB and FLOW modalities. Extensive experimental results on two challenging benchmarks including THUMOS14 and ActivityNet demonstrate the superior performance of our method. Our code is available at github.com/RenHuan1999/CVPR2023_P-MIL.

        ----

        ## [232] LOGO: A Long-Form Video Dataset for Group Action Quality Assessment

        **Authors**: *Shiyi Zhang, Wenxun Dai, Sujia Wang, Xiangwei Shen, Jiwen Lu, Jie Zhou, Yansong Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00238](https://doi.org/10.1109/CVPR52729.2023.00238)

        **Abstract**:

        Action quality assessment (AQA) has become an emerging topic since it can be extensively applied in numerous scenarios. However, most existing methods and datasets focus on single-person short-sequence scenes, hindering the application of AQA in more complex situations. To address this issue, we construct a new multi-person long-form video dataset for action quality assessment named LOGO. Distinguished in scenario complexity, our dataset contains 200 videos from 26 artistic swimming events with 8 athletes in each sample along with an average duration of 204.2 seconds. As for richness in annotations, LOGO includes formation labels to depict group information of multiple athletes and detailed annotations on action procedures. Furthermore, we propose a simple yet effective method to model relations among athletes and reason about the potential temporal logic in long-form videos. Specifically, we design a group-aware attention module, which can be easily plugged into existing AQA methods, to enrich the clip-wise representations based on contextual group information. To benchmark LOGO, we systematically conduct investigations on the performance of several popular methods in AQA and action segmentation. The results reveal the challenges our dataset brings. Extensive experiments also show that our approach achieves state-of-the-art on the LOGO dataset. The dataset and code will be released at https://github.com/shiyi-zh0408/LOGO.

        ----

        ## [233] Use Your Head: Improving Long-Tail Video Recognition

        **Authors**: *Toby Perrett, Saptarshi Sinha, Tilo Burghardt, Majid Mirmehdi, Dima Damen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00239](https://doi.org/10.1109/CVPR52729.2023.00239)

        **Abstract**:

        This paper presents an investigation into long-tail video recognition. We demonstrate that, unlike naturally-collected video datasets and existing long-tail image benchmarks, current video benchmarks fall short on multiple long-tailed properties. Most critically, they lack few-shot classes in their tails. In response, we propose new video benchmarks that better assess long-tail recognition, by sampling subsets from two datasets: SSv2 and VideoLT. We then propose a method, Long-Tail Mixed Reconstruction (LMR), which reduces overfitting to instances from few-shot classes by reconstructing them as weighted combinations of samples from head classes. LMR then employs label mixing to learn robust decision boundaries. It achieves state-of-the-art average class accuracy on EPIC-KITCHENS and the proposed SSv2-LT and VideoLT-LT. Benchmarks and code at: github.com/tobyperrett/lmr

        ----

        ## [234] Conditional Generation of Audio from Video via Foley Analogies

        **Authors**: *Yuexi Du, Ziyang Chen, Justin Salamon, Bryan Russell, Andrew Owens*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00240](https://doi.org/10.1109/CVPR52729.2023.00240)

        **Abstract**:

        The sound effects that designers add to videos are designed to convey a particular artistic effect and, thus, may be quite different from a scene's true sound. Inspired by the challenges of creating a soundtrack for a video that differs from its true sound, but that nonetheless matches the actions occurring on screen, we propose the problem of conditional Foley. We present the following contributions to address this problem. First, we propose a pretext task for training our model to predict sound for an input video clip using a conditional audio-visual clip sampled from another time within the same source video. Second, we propose a model for generating a soundtrack for a silent input video, given a user-supplied example that specifies what the video should “sound like”. We show through human studies and automated evaluation metrics that our model successfully generates sound from videos, while varying its output according to the content of a supplied example. Project site: https://xypb.github.io/CondFoleyGen.

        ----

        ## [235] Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos

        **Authors**: *Sixun Dong, Huazhang Hu, Dongze Lian, Weixin Luo, Yicheng Qian, Shenghua Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00241](https://doi.org/10.1109/CVPR52729.2023.00241)

        **Abstract**:

        Sequential video understanding, as an emerging video understanding task, has driven lots of researchers' attention because of its goal-oriented nature. This paper studies weakly supervised sequential video understanding where the accurate time-stamp level text-video alignment is not provided. We solve this task by borrowing ideas from CLIP. Specifically, we use a transformer to aggregate frame-level features for video representation and use a pre-trained text encoder to encode the texts corresponding to each action and the whole video, respectively. To model the correspondence between text and video, we propose a multiple granularity loss, where the video-paragraph contrastive loss enforces matching between the whole video and the complete script, and a fine-grained frame-sentence contrastive loss enforces the matching between each action and its description. As the frame-sentence correspondence is not available, we propose to use the fact that video actions happen sequentially in the temporal domain to generate pseudo frame-sentence correspondence and supervise the network training with the pseudo labels. Extensive experiments on video sequence verification and text-to-video matching show that our method outperforms baselines by a large margin, which validates the effectiveness of our proposed approach. Code is available at https://github.com/svip-lab/WeakSVR.

        ----

        ## [236] You Can Ground Earlier than See: An Effective and Efficient Pipeline for Temporal Sentence Grounding in Compressed Videos

        **Authors**: *Xiang Fang, Daizong Liu, Pan Zhou, Guoshun Nan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00242](https://doi.org/10.1109/CVPR52729.2023.00242)

        **Abstract**:

        Given an untrimmed video, temporal sentence grounding (TSG) aims to locate a target moment semantically according to a sentence query. Although previous respectable works have made decent success, they only focus on high-level visual features extracted from the consecutive decoded frames and fail to handle the compressed videos for query modelling, suffering from insufficient representation capability and significant computational complexity during training and testing. In this paper, we pose a new setting, compressed-domain TSG, which directly utilizes compressed videos rather than fully-decompressed frames as the visual input. To handle the raw video bit-stream input, we propose a novel Three-branch Compressed-domain Spatial-temporal Fusion (TCSF) framework, which extracts and aggregates three kinds of low-level visual features (I-frame, motion vector and residual features) for effective and efficient grounding. Particularly, instead of encoding the whole decoded frames like previous works, we capture the appearance representation by only learning the I-frame feature to reduce delay or latency. Besides, we explore the motion information not only by learning the motion vector feature, but also by exploring the relations of neighboring frames via the residual feature. In this way, a three-branch spatial-temporal attention layer with an adaptive motion-appearance fusion module is further designed to extract and aggregate both appearance and motion information for the final grounding. Experiments on three challenging datasets shows that our TCSF achieves better performance than other state-of-the-art methods with lower complexity.

        ----

        ## [237] Connecting Vision and Language with Video Localized Narratives

        **Authors**: *Paul Voigtlaender, Soravit Changpinyo, Jordi Pont-Tuset, Radu Soricut, Vittorio Ferrari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00243](https://doi.org/10.1109/CVPR52729.2023.00243)

        **Abstract**:

        We propose Video Localized Narratives, a new form of multimodal video annotations connecting vision and language. In the original Localized Narratives [36], annotators speak and move their mouse simultaneously on an image, thus grounding each word with a mouse trace segment. However, this is challenging on a video. Our new protocol empowers annotators to tell the story of a video with Localized Narratives, capturing even complex events involving multiple actors interacting with each other and with several passive objects. We annotated 20k videos of the OVIS, UVO, and Oops datasets, totalling 1.7M words. Based on this data, we also construct new benchmarks for the video narrative grounding and video question answering tasks, and provide reference results from strong baseline models. Our annotations are available at https://google.github.io/video-localized-narratives/

        ----

        ## [238] Video-Text as Game Players: Hierarchical Banzhaf Interaction for Cross-Modal Representation Learning

        **Authors**: *Peng Jin, Jinfa Huang, Pengfei Xiong, Shangxuan Tian, Chang Liu, Xiangyang Ji, Li Yuan, Jie Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00244](https://doi.org/10.1109/CVPR52729.2023.00244)

        **Abstract**:

        Contrastive learning-based video-language representation learning approaches, e.g., CLIP, have achieved outstanding performance, which pursue semantic interaction upon pre-defined video-text pairs. To clarify this coarse-grained global interaction and move a step further, we have to encounter challenging shell-breaking interactions for fine-grained cross-modal learning. In this paper, we creatively model video-text as game players with multivariate cooperative game theory to wisely handle the uncertainty during fine-grained semantic interaction with diverse granularity, flexible combination, and vague intensity. Concretely, we propose Hierarchical Banzhaf Interaction (HBI) to value possible correspondence between video frames and text words for sensitive and explainable cross-modal contrast. To efficiently realize the cooperative game of multiple video frames and multiple text words, the proposed method clusters the original video frames (text words) and computes the Banzhaf Interaction between the merged tokens. By stacking token merge modules, we achieve cooperative games at different semantic levels. Extensive experiments on commonly used text-video retrieval and video-question answering bench-marks with superior performances justify the efficacy of our HBI. More encouragingly, it can also serve as a visualization tool to promote the understanding of cross-modal interaction, which have a far-reaching impact on the community. Project page is available at https://jpthu17.github.io/HBI/.

        ----

        ## [239] Aligning Step-by-Step Instructional Diagrams to Video Demonstrations

        **Authors**: *Jiahao Zhang, Anoop Cherian, Yanbin Liu, Yizhak Ben-Shabat, Cristian Rodriguez Opazo, Stephen Gould*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00245](https://doi.org/10.1109/CVPR52729.2023.00245)

        **Abstract**:

        Multimodal alignment facilitates the retrieval of instances from one modality when queried using another. In this paper, we consider a novel setting where such an alignment is between (i) instruction steps that are depicted as assembly diagrams (commonly seen in Ikea assembly manuals) and (ii) segments from in-the-wild videos; these videos comprising an enactment of the assembly actions in the real world. We introduce a supervised contrastive learning approach that learns to align videos with the subtle details of assembly diagrams, guided by a set of novel losses. To study this problem and evaluate the effectiveness of our method, we introduce a new dataset: IAW-for Ikea assembly in the wild-consisting of 183 hours of videos from diverse furniture assembly collections and nearly 8,300 illustrations from their associated instruction manuals and annotated for their ground truth alignments. We define two tasks on this dataset: First, nearest neighbor retrieval between video segments and illustrations, and, second, alignment of instruction steps and the segments for each video. Extensive experiments on IAW demonstrate superior performance of our approach against alternatives.

        ----

        ## [240] Make-A-Story: Visual Memory Conditioned Consistent Story Generation

        **Authors**: *Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, Leonid Sigal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00246](https://doi.org/10.1109/CVPR52729.2023.00246)

        **Abstract**:

        There has been a recent explosion of impressive generative models that can produce high quality images (or videos) conditioned on text descriptions. However, all such approaches rely on conditional sentences that contain un-ambiguous descriptions of scenes and main actors in them. Therefore employing such models for more complex task of story visualization, where naturally references and co-references exist, and one requires to reason about when to maintain consistency of actors and backgrounds across frames/scenes, and when not to, based on story progression, remains a challenge. In this work, we address the aforementioned challenges and propose a novel autoregressive diffusion-based framework with a visual memory module that implicitly captures the actor and background context across the generated frames. Sentence-conditioned soft attention over the memories enables effective reference resolution and learns to maintain scene and actor consistency when needed. To validate the effectiveness of our approach, we extend the MUGEN dataset [19] and introduce additional characters, backgrounds and referencing in multi-sentence storylines. Our experiments for story generation on the MUGEN, the PororoSV [30] and the FlintstonesSV [16] dataset show that our method not only outperforms prior state-of-the-art in generating frames with high visual quality, which are consistent with the story, but also models appropriate correspondences between the characters and the background.

        ----

        ## [241] Test of Time: Instilling Video-Language Models with a Sense of Time

        **Authors**: *Piyush Bagad, Makarand Tapaswi, Cees G. M. Snoek*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00247](https://doi.org/10.1109/CVPR52729.2023.00247)

        **Abstract**:

        Modelling and understanding time remains a challenge in contemporary video understanding models. With language emerging as a key driver towards powerful generalization, it is imperative for foundational video-language models to have a sense of time. In this paper, we consider a specific aspect of temporal understanding: consistency of time order as elicited by before/after relations. We establish that seven existing video-language models struggle to understand even such simple temporal relations. We then question whether it is feasible to equip these foundational models with temporal awareness without re-training them from scratch. Towards this, we propose a temporal adaptation recipe on top of one such model, VideoCLIp, based on post-pretraining on a small amount of video-text data. We conduct a zero-shot evaluation of the adapted models on six datasets for three downstream tasks which require varying degrees of time awareness. We observe encouraging performance gains especially when the task needs higher time awareness. Our work serves as a first step towards probing and instilling a sense of time in existing video-language models without the need for data and compute-intense training from scratch.

        ----

        ## [242] How You Feelin'? Learning Emotions and Mental States in Movie Scenes

        **Authors**: *Dhruv Srivastava, Aditya Kumar Singh, Makarand Tapaswi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00248](https://doi.org/10.1109/CVPR52729.2023.00248)

        **Abstract**:

        Movie story analysis requires understanding characters' emotions and mental states. Towards this goal, we formulate emotion understanding as predicting a diverse and multi-label set of emotions at the level of a movie scene and for each character. We propose EmoTx, a multimodal Transformer-based architecture that ingests videos, multiple characters, and dialog utterances to make joint predictions. By leveraging annotations from the MovieGraphs dataset [72], we aim to predict classic emotions (e.g. happy, angry) and other mental states (e.g. honest, helpful). We conduct experiments on the most frequently occurring 10 and 25 labels, and a mapping that clusters 181 labels to 26. Ablation studies and comparison against adapted state-of-the-art emotion recognition approaches shows the effectiveness of EmoTx. Analyzing EmoTx's self-attention scores reveals that expressive emotions often look at character tokens while other mental states rely on video and dialog cues.

        ----

        ## [243] Continuous Sign Language Recognition with Correlation Network

        **Authors**: *Lianyu Hu, Liqing Gao, Zekang Liu, Wei Feng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00249](https://doi.org/10.1109/CVPR52729.2023.00249)

        **Abstract**:

        Human body trajectories are a salient cue to identify actions in the video. Such body trajectories are mainly conveyed by hands and face across consecutive frames in sign language. However, current methods in continuous sign language recognition (CSLR) usually process frames independently, thus failing to capture cross-frame trajectories to effectively identify a sign. To handle this limitation, we propose correlation network (CorrNet) to explicitly capture and leverage body trajectories across frames to identify signs. In specific, a correlation module is first proposed to dynamically compute correlation maps between the current frame and adjacent frames to identify trajectories of all spatial patches. An identification module is then presented to dynamically emphasize the body trajectories within these correlation maps. As a result, the generated features are able to gain an overview of local temporal movements to identify a sign. Thanks to its special attention on body trajectories, CorrNet achieves new state-of-the-art accuracy on four largescale datasets, i.e., PHOENIX14, PHOENIX14-T, CSL-Daily, and CSL. A comprehensive comparison with previous spatial-temporal reasoning methods verifies the effectiveness of CorrNet. Visualizations demonstrate the effects of CorrNet on emphasizing human body trajectories across adjacent frames.

        ----

        ## [244] DIP: Dual Incongruity Perceiving Network for Sarcasm Detection

        **Authors**: *Changsong Wen, Guoli Jia, Jufeng Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00250](https://doi.org/10.1109/CVPR52729.2023.00250)

        **Abstract**:

        Sarcasm indicates the literal meaning is contrary to the real attitude. Considering the popularity and complementarity of image-text data, we investigate the task of multi-modal sarcasm detection. Different from other multi-modal tasks, for the sarcastic data, there exists intrinsic incongruity between a pair of image and text as demonstrated in psychological theories. To tackle this issue, we propose a Dual Incongruity Perceiving (DIP) network consisting of two branches to mine the sarcastic information from factual and affective levels. For the factual aspect, we introduce a channel-wise reweighting strategy to obtain semantically discriminative embeddings, and leverage gaussian distribution to model the uncertain correlation caused by the incongruity. The distribution is generated from the latest data stored in the memory bank, which can adaptively model the difference of semantic similarity between sarcastic and non-sarcastic data. For the affective aspect, we utilize siamese layers with shared parameters to learn cross-modal sentiment information. Furthermore, we use the polarity value to construct a relation graph for the mini-batch, which forms the continuous contrastive loss to acquire affective embeddings. Extensive experiments demonstrate that our proposed method performs favorably against state-of-the-art approaches. Our code is released on https://github.com/downdric/MSD.

        ----

        ## [245] Gloss Attention for Gloss-free Sign Language Translation

        **Authors**: *Aoxiong Yin, Tianyun Zhong, Li Tang, Weike Jin, Tao Jin, Zhou Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00251](https://doi.org/10.1109/CVPR52729.2023.00251)

        **Abstract**:

        Most sign language translation (SLT) methods to date require the use of gloss annotations to provide additional supervision information, however, the acquisition of gloss is not easy. To solve this problem, we first perform an analysis of existing models to confirm how gloss annotations make SLT easier. We find that it can provide two aspects of information for the model, 1) it can help the model implicitly learn the location of semantic boundaries in continuous sign language videos, 2) it can help the model understand the sign language video globally. We then propose gloss attention, which enables the model to keep its attention within video segments that have the same semantics locally, just as gloss helps existing models do. Furthermore, we transfer the knowledge of sentence-to-sentence similarity from the natural language model to our gloss attention SLT network (GASLT) to help it understand sign language videos at the sentence level. Experimental results on multiple large-scale sign language datasets show that our proposed GASLT model significantly outperforms existing methods. Our code is provided in https://github.com/YinAoXiong/GASLT.

        ----

        ## [246] Object-Goal Visual Navigation via Effective Exploration of Relations Among Historical Navigation States

        **Authors**: *Heming Du, Lincheng Li, Zi Huang, Xin Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00252](https://doi.org/10.1109/CVPR52729.2023.00252)

        **Abstract**:

        Object-goal visual navigation aims at steering an agent toward an object via a series of moving steps. Previous works mainly focus on learning informative visual representations for navigation, but overlook the impacts of navigation states on the effectiveness and efficiency of navigation. We observe that high relevance among navigation states will cause navigation inefficiency or failure for existing methods. In this paper, we present a History-inspired Navigation Policy Learning (HiNL) framework to estimate navigation states effectively by exploring relationships among historical navigation states. In HiNL, we propose a History-aware State Estimation (HaSE) module to alleviate the impacts of dominant historical states on the current state estimation. Meanwhile, HaSE also encourages an agent to be alert to the current observation changes, thus enabling the agent to make valid actions. Furthermore, we design a History-based State Regularization (HbSR) to explicitly suppress the correlation among navigation states in training. As a result, our agent can update states more effectively while reducing the correlations among navigation states. Experiments on the artificial platform AI2-THOR (i.e., iTHOR and RoboTHOR) demonstrate that HiNL significantly outperforms state-of-the-art methods on both Success Rate and SPL in unseen testing environments.

        ----

        ## [247] Behavioral Analysis of Vision-and-Language Navigation Agents

        **Authors**: *Zijiao Yang, Arjun Majumdar, Stefan Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00253](https://doi.org/10.1109/CVPR52729.2023.00253)

        **Abstract**:

        To be successful, Vision-and-Language Navigation (VLN) agents must be able to ground instructions to actions based on their surroundings. In this work, we develop a methodology to study agent behavior on a skill-specific basis - examining how well existing agents ground instructions about stopping, turning, and moving towards specified objects or rooms. Our approach is based on generating skill-specific interventions and measuring changes in agent predictions. We present a detailed case study analyzing the behavior of a recent agent and then compare multiple agents in terms of skill-specific competency scores. This analysis suggests that biases from training have lasting effects on agent behavior and that existing models are able to ground simple referring expressions. Our comparisons between models show that skill-specific scores correlate with improvements in overall VLN task performance.

        ----

        ## [248] KERM: Knowledge Enhanced Reasoning for Vision-and-Language Navigation

        **Authors**: *Xiangyang Li, Zihan Wang, Jiahao Yang, Yaowei Wang, Shuqiang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00254](https://doi.org/10.1109/CVPR52729.2023.00254)

        **Abstract**:

        Vision-and-language navigation (VLN) is the task to enable an embodied agent to navigate to a remote location following the natural language instruction in real scenes. Most of the previous approaches utilize the entire features or object-centric features to represent navigable candidates. However, these representations are not efficient enough for an agent to perform actions to arrive the target location. As knowledge provides crucial information which is complementary to visible content, in this paper, we propose a Knowledge Enhanced Reasoning Model (KERM) to leverage knowledge to improve agent navigation ability. Specifically, we first retrieve facts (i.e., knowledge described by language descriptions) for the navigation views based on local regions from the constructed knowledge base. The re-trieved facts range from properties of a single object (e.g., color, shape) to relationships between objects (e.g., action, spatial position), providing crucial information for VLN. We further present the KERM which contains the purification, fact-aware interaction, and instruction-guided aggregation modules to integrate visual, history, instruction, and fact features. The proposed KERM can automatically select and gather crucial and relevant cues, obtaining more accurate action prediction. Experimental results on the REVERIE, R2R, and SOON datasets demonstrate the effectiveness of the proposed method. The source code is available at https://github.com/XiangyangLi20/KERM.

        ----

        ## [249] Where is my Wallet? Modeling Object Proposal Sets for Egocentric Visual Query Localization

        **Authors**: *Mengmeng Xu, Yanghao Li, Cheng-Yang Fu, Bernard Ghanem, Tao Xiang, Juan-Manuel Pérez-Rúa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00255](https://doi.org/10.1109/CVPR52729.2023.00255)

        **Abstract**:

        This paper deals with the problem of localizing objects in image and video datasets from visual exemplars. In particular, we focus on the challenging problem of egocentric visual query localization. We first identify grave implicit biases in current query-conditioned model design and visual query datasets. Then, we directly tackle such biases at both frame and object set levels. Concretely, our method solves these issues by expanding limited annotations and dynamically dropping object proposals during training. Additionally, we propose a novel transformer-based module that allows for object-proposal set context to be considered while incorporating query information. We name our module Conditioned Contextual Transformer or CocoFormer. Our experiments show the proposed adaptations improve egocentric query detection, leading to a better visual query localization system in both 2D and 3D configurations. Thus, we can improve frame-level detection performance from 26.28% to 31.26% in AP, which correspondingly improves the VQ2D and VQ3D localization scores by significant margins. Our improved context-aware query object detector ranked first and second respectively in the VQ2D and VQ3D tasks in the 2nd Ego4D challenge. In addition to this, we showcase the relevance of our proposed model in the Few-Shot Detection (FSD) task, where we also achieve SOTA results. Our code is available at https://github.com/facebookresearch/vq2d_cvpr.

        ----

        ## [250] Efficient Multimodal Fusion via Interactive Prompting

        **Authors**: *Yaowei Li, Ruijie Quan, Linchao Zhu, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00256](https://doi.org/10.1109/CVPR52729.2023.00256)

        **Abstract**:

        Large-scale pre-training has brought unimodal fields such as computer vision and natural language processing to a new era. Following this trend, the size of multimodal learning models constantly increases, leading to an urgent need to reduce the massive computational cost of finetuning these models for downstream tasks. In this paper, we propose an efficient and flexible multimodal fusion method, namely PMF, tailored for fusing unimodally pretrained transformers. Specifically, we first present a modular multimodal fusion framework that exhibits high flexibility and facilitates mutual interactions among different modalities. In addition, we disentangle vanilla prompts into three types in order to learn different optimizing objectives for multimodal learning. It is also worth noting that we propose to add prompt vectors only on the deep layers of the unimodal transformers, thus significantly reducing the training memory usage. Experiment results show that our proposed method achieves comparable performance to several other multimodal finetuning methods with less than 3% trainable parameters and up to 66% saving of training memory usage.

        ----

        ## [251] NS3D: Neuro-Symbolic Grounding of 3D Objects and Relations

        **Authors**: *Joy Hsu, Jiayuan Mao, Jiajun Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00257](https://doi.org/10.1109/CVPR52729.2023.00257)

        **Abstract**:

        Grounding object properties and relations in 3D scenes is a prerequisite for a wide range of artificial intelligence tasks, such as visually grounded dialogues and embodied manipulation. However, the variability of the 3D domain induces two fundamental challenges: 1) the expense of labeling and 2) the complexity of 3D grounded language. Hence, essential desiderata for models are to be data-efficient, generalize to different data distributions and tasks with unseen semantic forms, as well as ground complex language semantics (e.g., view-point anchoring and multi-object reference). To address these challenges, we propose NS3D, a neuro-symbolic framework for 3D grounding. NS3D translates language into programs with hierarchical structures by leveraging large language-to-code models. Different functional modules in the programs are implemented as neural networks. Notably, NS3D extends prior neuro-symbolic visual reasoning methods by introducing functional modules that effectively reason about high-arity relations (i.e., relations among more than two objects), key in disambiguating objects in complex 3D scenes. Modular and compositional architecture enables NS3D to achieve state-of-the-art results on the ReferIt3D view-dependence task, a 3D referring expression compre-hension benchmark. Importantly, NS3D shows significantly improved performance on settings of data-efficiency and generalization, and demonstrate zero-shot transfer to an unseen 3D question-answering task.

        ----

        ## [252] Dynamic Inference with Grounding Based Vision and Language Models

        **Authors**: *Burak Uzkent, Amanmeet Garg, Wentao Zhu, Keval Doshi, Jingru Yi, Xiaolong Wang, Mohamed Omar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00258](https://doi.org/10.1109/CVPR52729.2023.00258)

        **Abstract**:

        Transformers have been recently utilized for vision and language tasks successfully. For example, recent image and language models with more than 200M parameters have been proposed to learn visual grounding in the pre-training step and show impressive results on downstream vision and language tasks. On the other hand, there exists a large amount of computational redundancy in these large models which skips their run-time efficiency. To address this problem, we propose dynamic inference for grounding based vision and language models conditioned on the input image-text pair. We first design an approach to dynamically skip multihead self-attention and feed forward network layers across two backbones and multimodal network. Additionally, we propose dynamic token pruning and fusion for two backbones. In particular, we remove redundant tokens at different levels of the backbones and fuse the image tokens with the language tokens in an adaptive manner. To learn policies for dynamic inference, we train agents using reinforcement learning. In this direction, we replace the CNN backbone in a recent grounding-based vision and language model, MDETR, with a vision transformer and call it ViTMDETR. Then, we apply our dynamic inference method to ViTMDETR, called D-ViTDMETR, and perform experiments on image-language tasks. Our results show that we can improve the run-time efficiency of the state-of-the-art models MDETR and GLIP by up to ~ 50% on Referring Expression Comprehension and Segmentation, and VQA with only maximum ~ 0.3% accuracy drop.

        ----

        ## [253] Improving Commonsense in Vision-Language Models via Knowledge Graph Riddles

        **Authors**: *Shuquan Ye, Yujia Xie, Dongdong Chen, Yichong Xu, Lu Yuan, Chenguang Zhu, Jing Liao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00259](https://doi.org/10.1109/CVPR52729.2023.00259)

        **Abstract**:

        This paper focuses on analyzing and improving the commonsense ability of recent popular vision-language (VL) models. Despite the great success, we observe that existing VL-models still lack commonsense knowledge/reasoning ability (e.g., “Lemons are sour”), which is a vital component towards artificial general intelligence. Through our analysis, we find one important reason is that existing large-scale VL datasets do not contain much commonsense knowledge, which motivates us to improve the commonsense of VL-models from the data perspective. Rather than collecting a new VL training dataset, we propose a more scalable strategy, i.e., “Data Augmentation with kNowledge graph linearization for CommonsensE capability” (DANCE). It can be viewed as one type of data augmentation technique, which can inject commonsense knowledge into existing VL datasets on the fly during training. More specifically, we leverage the commonsense knowledge graph (e.g., ConceptNet) and create variants of text description in VL datasets via bidirectional sub-graph sequentialization. For better commonsense evaluation, we further propose the first retrieval-based commonsense diagnostic benchmark. By conducting extensive experiments on some representative VL-models, we demonstrate that our DANCE technique is able to significantly improve the commonsense ability while maintaining the performance on vanilla retrieval tasks. The code and data are available at https://github.com/pleaseconnectwifi/DANCE.

        ----

        ## [254] S3C: Semi-Supervised VQA Natural Language Explanation via Self-Critical Learning

        **Authors**: *Wei Suo, Mengyang Sun, Weisong Liu, Yiqi Gao, Peng Wang, Yanning Zhang, Qi Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00260](https://doi.org/10.1109/CVPR52729.2023.00260)

        **Abstract**:

        VQA Natural Language Explanation (VQA-NLE) task aims to explain the decision-making process of VQA models in natural language. Unlike traditional attention or gradient analysis, free-text rationales can be easier to understand and gain users' trust. Existing methods mostly use post-hoc or selfrationalization models to obtain a plausible explanation. However, these frameworks are bottle-necked by the following challenges: 1) the reasoning process cannot be faithfully responded to and suffer from the problem of logical inconsistency. 2) Human-annotated explanations are expensive and time-consuming to collect. In this paper, we propose a new Semi-Supervised VQA-NLE via Self-Critical Learning (S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
C), which evaluates the candidate explanations by answering rewards to improve the logical consistency between answers and rationales. With a semi-supervised learning framework, the S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
C can benefit from a tremendous amount of samples without human-annotated explanations. A large number of automatic measures and human evaluations all show the effectiveness of our method. Meanwhile, the framework achieves a new state-of-the-art performance on the two VQA-NLE datasets.

        ----

        ## [255] Teaching Structured Vision & Language Concepts to Vision & Language Models

        **Authors**: *Sivan Doveh, Assaf Arbelle, Sivan Harary, Eli Schwartz, Roei Herzig, Raja Giryes, Rogério Feris, Rameswar Panda, Shimon Ullman, Leonid Karlinsky*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00261](https://doi.org/10.1109/CVPR52729.2023.00261)

        **Abstract**:

        Vision and Language (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
) models have demonstrated remarkable zero-shot performance in a variety of tasks. However, some aspects of complex language understanding still remain a challenge. We introduce the collective notion of Structured Vision & Language Concepts (SVLC) which includes object attributes, relations, and states which are present in the text and visible in the image. Recent studies have shown that even the best 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 models struggle with SVLC. A possible way of fixing this issue is by collecting dedicated datasets for teaching each SVLC type, yet this might be expensive and time-consuming. Instead, we propose a more elegant data-driven approach for enhancing 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 models' understanding of SVLCs that makes more effective use of existing 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 pre-training datasets and does not require any additional data. While automatic understanding of image structure still remains largely unsolved, language structure is much better modeled and understood, allowing for its effective utilization in teaching 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 models. In this paper, we propose various techniques based on language structure understanding that can be used to manipulate the textual part of off-the-shelf paired 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 datasets. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$VL$</tex>
 models trained with the updated data exhibit a significant improvement of up to 15% in their SVLC understanding with only a mild degradation in their zero-shot capabilities both when training from scratch or fine-tuning a pre-trained model. Our code and pretrained models are available at: https://github.com/SivanDoveh/TSVLC

        ----

        ## [256] FAME-ViL: Multi-Tasking Vision-Language Model for Heterogeneous Fashion Tasks

        **Authors**: *Xiao Han, Xiatian Zhu, Licheng Yu, Li Zhang, Yi-Zhe Song, Tao Xiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00262](https://doi.org/10.1109/CVPR52729.2023.00262)

        **Abstract**:

        In the fashion domain, there exists a variety of vision-and-language (V+L) tasks, including cross-modal retrieval, text-guided image retrieval, multi-modal classification, and image captioning. They differ drastically in each individual input/output format and dataset size. It has been common to design a task-specific model and fine-tune it independently from a pre-trained V+l model (e.g., CLIP). This results in parameter inefficiency and inability to exploit inter-task relatedness. To address such issues, we propose a novel FAshion-focused Multi-task Efficient learning method for Vision-and-Language tasks (FAME-ViL) in this work. Compared with existing approaches, FAME-ViL applies a single model for multiple heterogeneous fashion tasks, therefore being much more parameter-efficient. It is enabled by two novel components: (1) a task-versatile architecture with cross-attention adapters and task-specific adapters integrated into a unified V+L model, and (2) a stable and effective multi-task training strategy that supports learning from heterogeneous data and prevents negative transfer. Extensive experiments on four fashion tasks show that our FAME-ViL can save 61.5% of parameters over alternatives, while significantly outperforming the conventional independently trained single-task models. Code is available at https://github.com/BrandonHanx/FAME-ViL.

        ----

        ## [257] Uni-Perceiver v2: A Generalist Model for Large-Scale Vision and Vision-Language Tasks

        **Authors**: *Hao Li, Jinguo Zhu, Xiaohu Jiang, Xizhou Zhu, Hongsheng Li, Chun Yuan, Xiaohua Wang, Yu Qiao, Xiaogang Wang, Wenhai Wang, Jifeng Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00264](https://doi.org/10.1109/CVPR52729.2023.00264)

        **Abstract**:

        Despite the remarkable success of foundation models, their task-specific fine-tuning paradigm makes them inconsistent with the goal of general perception modeling. The key to eliminating this inconsistency is to use generalist models for general task modeling. However, existing attempts at generalist models are inadequate in both versatility and performance. In this paper, we propose Uni-Perceiver v2, which is the first generalist model capable of handling major large-scale vision and vision-language tasks with competitive performance. Specifically, images are encoded as general region proposals, while texts are encoded via a Transformer-based language model. The encoded representations are transformed by a task-agnostic decoder. Different tasks are formulated as a unified maximum likelihood estimation problem. We further propose an effective optimization technique named Task-Balanced Gradient Normalization to ensure stable multi-task learning with an unmixed sampling strategy, which is helpful for tasks requiring large batch-size training. After being jointly trained on various tasks, Uni-Perceiver v2 is capable of directly handling downstream tasks without any task-specific adaptation. Results show that Uni-Perceiver v2 outperforms all existing generalist models in both versatility and performance. Meanwhile, compared with the commonly-recognized strong baselines that require tasks-specific fine-tuning, Uni-Perceiver v2 achieves competitive performance on a broad range of vision and vision-language tasks.

        ----

        ## [258] Learning from Unique Perspectives: User-aware Saliency Modeling

        **Authors**: *Shi Chen, Nachiappan Valliappan, Shaolei Shen, Xinyu Ye, Kai Kohlhoff, Junfeng He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00265](https://doi.org/10.1109/CVPR52729.2023.00265)

        **Abstract**:

        Everyone is unique. Given the same visual stimuli, people's attention is driven by both salient visual cues and their own inherent preferences. Knowledge of visual preferences not only facilitates understanding of fine-grained attention patterns of diverse users, but also has the potential of benefiting the development of customized applications. Nevertheless, existing saliency models typically limit their scope to attention as it applies to the general population and ignore the variability between users' behaviors. In this paper, we identify the critical roles of visual preferences in attention modeling, and for the first time study the problem of user-aware saliency modeling. Our work aims to advance attention research from three distinct perspectives: (1) We present a new model with the flexibility to capture attention patterns of various combinations of users, so that we can adaptively predict personalized attention, user group attention, and general saliency at the same time with one single model; (2) To augment models with knowledge about the composition of attention from different users, we further propose a principled learning method to understand visual attention in a progressive manner; and (3) We carry out extensive analyses on publicly available saliency datasets to shed light on the roles of visual preferences. Experimental results on diverse stimuli, including naturalistic images and web pages, demonstrate the advantages of our method in capturing the distinct visual behaviors of different users and the general saliency of visual stimuli.

        ----

        ## [259] CRAFT: Concept Recursive Activation FacTorization for Explainability

        **Authors**: *Thomas Fel, Agustin Martin Picard, Louis Béthune, Thibaut Boissin, David Vigouroux, Julien Colin, Rémi Cadène, Thomas Serre*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00266](https://doi.org/10.1109/CVPR52729.2023.00266)

        **Abstract**:

        Attribution methods, which employ heatmaps to identify the most influential regions of an image that impact model decisions, have gained widespread popularity as a type of explainability method. However, recent research has exposed the limited practical value of these methods, attributed in part to their narrow focus on the most prominent regions of an image - revealing “where” the model looks, but failing to elucidate “what” the model sees in those areas. In this work, we try to fill in this gap with CRAFT - a novel approach to identify both “what” and “where” by generating concept-based explanations. We introduce 3 new ingredients to the automatic concept extraction literature: (i) a recursive strategy to detect and decompose concepts across layers, (ii) a novel method for a more faithful estimation of concept importance using Sobol indices, and (iii) the use of implicit differentiation to unlock Concept Attribution Maps. We conduct both human and computer vision experiments to demonstrate the benefits of the proposed approach. We show that the proposed concept importance estimation technique is more faithful to the model than previous methods. When evaluating the usefulness of the method for human experimenters on a human-centered utility benchmark, we find that our approach significantly improves on two of the three test scenarios.

        ----

        ## [260] Doubly Right Object Recognition: A Why Prompt for Visual Rationales

        **Authors**: *Chengzhi Mao, Revant Teotia, Amrutha Sundar, Sachit Menon, Junfeng Yang, Xin Wang, Carl Vondrick*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00267](https://doi.org/10.1109/CVPR52729.2023.00267)

        **Abstract**:

        Many visual recognition models are evaluated only on their classification accuracy, a metric for which they obtain strong performance. In this paper, we investigate whether computer vision models can also provide correct rationales for their predictions. We propose a “doubly right” object recognition benchmark, where the metric requires the model to simultaneously produce both the right labels as well as the right rationales. We find that state-of-the-art visual models, such as CLIP, often provide incorrect rationales for their categorical predictions. However, by transferring the rationales from language models into visual representations through a tailored dataset, we show that we can learn a “why prompt,” which adapts large visual representations to produce correct rationales. Visualizations and empirical experiments show that our prompts significantly improve performance on doubly right object recognition, in addition to zero-shot transfer to unseen tasks and datasets.

        ----

        ## [261] Sketch2Saliency: Learning to Detect Salient Objects from Human Drawings

        **Authors**: *Ayan Kumar Bhunia, Subhadeep Koley, Amandeep Kumar, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00268](https://doi.org/10.1109/CVPR52729.2023.00268)

        **Abstract**:

        Human sketch has already proved its worth in various visual understanding tasks (e.g., retrieval, segmentation, image-captioning, etc). In this paper, we reveal a new trait of sketches – that they are also salient. This is intuitive as sketching is a natural attentive process at its core. More specifically, we aim to study how sketches can be used as a weak label to detect salient objects present in an image. To this end, we propose a novel method that emphasises on how “salient object” could be explained by hand-drawn sketches. To accomplish this, we introduce a photo-to-sketch generation model that aims to generate sequential sketch coordinates corresponding to a given visual photo through a 2D attention mechanism. Attention maps accumulated across the time steps give rise to salient regions in the process. Extensive quantitative and qualitative experiments prove our hypothesis and delineate how our sketch-based saliency detection model gives a competitive performance compared to the state-of-the-art.

        ----

        ## [262] PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification

        **Authors**: *Meike Nauta, Jörg Schlötterer, Maurice van Keulen, Christin Seifert*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00269](https://doi.org/10.1109/CVPR52729.2023.00269)

        **Abstract**:

        Interpretable methods based on prototypical patches recognize various components in an image in order to explain their reasoning to humans. However, existing prototype-based methods can learn prototypes that are not in line with human visual perception, i.e., the same prototype can refer to different concepts in the real world, making interpretation not intuitive. Driven by the principle of explainability-by-design, we introduce PIP-Net (Patch-based Intuitive Prototypes Network): an interpretable image classification model that learns prototypical parts in a self-supervised fashion which correlate better with human vision. PIP-Net can be interpreted as a sparse scoring sheet where the presence of a prototypical part in an image adds evidence for a class. The model can also abstain from a decision for out-of-distribution data by saying “I haven't seen this before”. We only use image-level labels and do not rely on any part annotations. PIP-Net is globally interpretable since the set of learned prototypes shows the entire reasoning of the model. A smaller local explanation locates the relevant prototypes in one image. We show that our prototypes correlate with ground-truth object parts, indicating that PIP-Net closes the “semantic gap” between latent space and pixel space. Hence, our PIP-Net with interpretable prototypes enables users to interpret the decision making process in an intuitive, faithful and semantically meaningful way. Code is available at https://github.com/M-Nauta/PIPNet.

        ----

        ## [263] CLIP for All Things Zero-Shot Sketch-Based Image Retrieval, Fine-Grained or Not

        **Authors**: *Aneeshan Sain, Ayan Kumar Bhunia, Pinaki Nath Chowdhury, Subhadeep Koley, Tao Xiang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00271](https://doi.org/10.1109/CVPR52729.2023.00271)

        **Abstract**:

        In this paper, we leverage CLIP for zero-shot sketch based image retrieval (ZS-SBIR). We are largely inspired by recent advances on foundation models and the unparalleled generalisation ability they seem to offer, but for the first time tailor it to benefit the sketch community. We put forward novel designs on how best to achieve this synergy, for both the category setting and the fine-grained setting ('all”}. At the very core of our solution is a prompt learning setup. First we show just via factoring in sketch-specific prompts, we already have a category-level ZS-SBIR system that over-shoots all prior arts, by a large margin (24.8%) - a great testimony on studying the CLIP and ZS-SBIR synergy. Moving onto the fine-grained setup is however trickier, and re-quires a deeper dive into this synergy. For that, we come up with two specific designs to tackle the fine-grained matching nature of the problem: (i) an additional regularisation loss to ensure the relative separation between sketches and photos is uniform across categories, which is not the case for the gold standard standalone triplet loss, and (ii) a clever patch shuffling technique to help establishing instance-level structural correspondences between sketch-photo pairs. With these designs, we again observe signifi-cant performance gains in the region of 26.9% over previ-ous state-of-the-art. The take-home message, if any, is the proposed CLIP and prompt learning paradigm carries great promise in tackling other sketch-related tasks (not limited to ZS-SBIR) where data scarcity remains a great challenge. Project page: https://aneeshan95.github.ioISketchLVM/

        ----

        ## [264] iCLIP: Bridging Image Classification and Contrastive Language-Image Pre-training for Visual Recognition

        **Authors**: *Yixuan Wei, Yue Cao, Zheng Zhang, Houwen Peng, Zhuliang Yao, Zhenda Xie, Han Hu, Baining Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00272](https://doi.org/10.1109/CVPR52729.2023.00272)

        **Abstract**:

        This paper presents a method that effectively combines two prevalent visual recognition methods, i.e., image classification and contrastive language-image pre-training, dubbed iCLIP. Instead of naïve multi-task learning that use two separate heads for each task, we fuse the two tasks in a deep fashion that adapts the image classification to share the same formula and the same model weights with the language-image pre-training. To further bridge these two tasks, we propose to enhance the category names in image classification tasks using external knowledge, such as their descriptions in dictionaries. Extensive experiments show that the proposed method combines the advantages of two tasks well: the strong discrimination ability in image classification tasks due to the clean category labels, and the good zero-shot ability in CLIP tasks ascribed to the richer semantics in the text descriptions. In particular, it reaches 82.9% top-1 accuracy on IN-1K, and mean-while surpasses CLIP by 1.8%, with similar model size, on zero-shot recognition of Kornblith 12-dataset benchmark. The code and models are publicly available at https://github.com/weiyx16/iCLIP.

        ----

        ## [265] Cross-Modal Implicit Relation Reasoning and Aligning for Text-to-Image Person Retrieval

        **Authors**: *Ding Jiang, Mang Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00273](https://doi.org/10.1109/CVPR52729.2023.00273)

        **Abstract**:

        Text-to-image person retrieval aims to identify the target person based on a given textual description query. The primary challenge is to learn the mapping of visual and textual modalities into a common latent space. Prior works have attempted to address this challenge by leveraging separately pre-trained unimodal models to extract visual and textual features. However, these approaches lack the necessary underlying alignment capabilities required to match multimodal data effectively. Besides, these works use prior information to explore explicit part alignments, which may lead to the distortion of intra-modality information. To alleviate these issues, we present IRRA: a cross-modal Implicit Relation Reasoning and Aligning framework that learns relations between local visual-textual tokens and enhances global image-text matching without requiring additional prior supervision. Specifically, we first design an Implicit Relation Reasoning module in a masked language modeling paradigm. This achieves cross-modal interaction by integrating the visual cues into the textual tokens with a cross-modal multimodal interaction encoder. Secondly, to globally align the visual and textual embeddings, Similarity Distribution Matching is proposed to minimize the KL divergence between image-text similarity distributions and the normalized label matching distributions. The proposed method achieves new state-of-the-art results on all three public datasets, with a notable margin of about 3%-9% for Rank-1 accuracy compared to prior methods.

        ----

        ## [266] Multi-Modal Representation Learning with Text-Driven Soft Masks

        **Authors**: *Jaeyoo Park, Bohyung Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00274](https://doi.org/10.1109/CVPR52729.2023.00274)

        **Abstract**:

        We propose a visual-linguistic representation learning approach within a self-supervised learning framework by introducing a new operation, loss, and data augmentation strategy. First, we generate diverse features for the image-text matching (ITM) task via soft-masking the regions in an image, which are most relevant to a certain word in the cor-responding caption, instead of completely removing them. Since our framework relies only on image-caption pairs with no fine-grained annotations, we identify the relevant regions to each word by computing the word-conditional vi-sual attention using multi-modal encoder. Second, we encourage the model to focus more on hard but diverse examples by proposing a focal loss for the image-text contrastive learning (ITC) objective, which alleviates the inherent limitations of overfitting and bias issues. Last, we perform multi-modal data augmentations for self-supervised learning via mining various examples by masking texts and rendering distortions on images. We show that the combination of these three innovations is effective for learning a pretrained model, leading to outstanding performance on multiple vision-language downstream tasks.

        ----

        ## [267] Texts as Images in Prompt Tuning for Multi-Label Image Recognition

        **Authors**: *Zixian Guo, Bowen Dong, Zhilong Ji, Jinfeng Bai, Yiwen Guo, Wangmeng Zuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00275](https://doi.org/10.1109/CVPR52729.2023.00275)

        **Abstract**:

        Prompt tuning has been employed as an efficient way to adapt large vision-language pre-trained models (e.g. CLIP) to various downstream tasks in data-limited or label-limited settings. Nonetheless, visual data (e.g., images) is by default prerequisite for learning prompts in existing methods. In this work, we advocate that the effectiveness of image-text contrastive learning in aligning the two modalities (for training CLIP) further makes it feasible to treat texts as images for prompt tuning and introduce TaI prompting. In contrast to the visual data, text descriptions are easy to collect, and their class labels can be directly derived. Particularly, we apply TaI prompting to multi-label image recognition, where sentences in the wild serve as alternatives to images for prompt tuning. Moreover, with TaI, double-grained prompt tuning (TaI-DPT) is further presented to extract both coarse-grained and fine-grained embeddings for enhancing the multi-label recognition performance. Experimental results show that our proposed TaI-DPT outperforms zero-shot CLIP by a large margin on multiple benchmarks, e.g., MS-COCO, VOC2007, and NUS-WIDE, while it can be combined with existing methods of prompting from images to improve recognition performance further. The code is released at https://github.com/guozix/TaI-DPT.

        ----

        ## [268] Reproducible Scaling Laws for Contrastive Language-Image Learning

        **Authors**: *Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, Jenia Jitsev*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00276](https://doi.org/10.1109/CVPR52729.2023.00276)

        **Abstract**:

        Scaling up neural networks has led to remarkable performance across a wide range of tasks. Moreover, performance often follows reliable scaling laws as a function of training set size, model size, and compute, which offers valuable guidance as large-scale experiments are becoming increasingly expensive. However, previous work on scaling laws has primarily used private data & models or focused on uni-modal language or vision learning. To address these limitations, we investigate scaling laws for contrastive language-image pre-training (CLIP) with the public LAION dataset and the open-source OpenCLIP repository. Our large-scale experiments involve models trained on up to two billion image-text pairs and identify power law scaling for multiple downstream tasks including zero-shot classification, retrieval, linear probing, and end-to-end fine-tuning. We find that the training distribution plays a key role in scaling laws as the OpenAI and OpenCLIP models exhibit different scaling behavior despite identical model architectures and similar training recipes. We open-source our evaluation workflow and all models, including the largest public CLIP models, to ensure reproducibility and make scaling laws research more accessible. Source code and instructions to reproduce this study is available at https://github.eom/LAION-AI/sealing-laws-openelip.

        ----

        ## [269] Multilateral Semantic Relations Modeling for Image Text Retrieval

        **Authors**: *Zheng Wang, Zhenwei Gao, Kangshuai Guo, Yang Yang, Xiaoming Wang, Heng Tao Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00277](https://doi.org/10.1109/CVPR52729.2023.00277)

        **Abstract**:

        Image-text retrieval is a fundamental task to bridge vision and language by exploiting various strategies to finegrained alignment between regions and words. This is still tough mainly because of one-to-many correspondence, where a set of matches from another modality can be accessed by a random query. While existing solutions to this problem including multi-point mapping, probabilistic distribution, and geometric embedding have made promising progress, one-to-many correspondence is still under-explored. In this work, we develop a Multilateral Semantic Relations Modeling (termed MSRM) for image-text retrieval to capture the one-to-many correspondence between multiple samples and a given query via hypergraph modeling. Specifically, a given query is first mapped as a probabilistic embedding to learn its true semantic distribution based on Mahalanobis distance. Then each candidate instance in a mini-batch is regarded as a hypergraph node with its mean semantics while a Gaussian query is modeled as a hyperedge to capture the semantic correlations beyond the pair between candidate points and the query. Comprehensive experimental results on two widely used datasets demonstrate that our MSRM method can outper-form state-of-the-art methods in the settlement of multiple matches while still maintaining the comparable performance of instance-level matching.

        ----

        ## [270] Smallcap: Lightweight Image Captioning Prompted with Retrieval Augmentation

        **Authors**: *Rita Ramos, Bruno Martins, Desmond Elliott, Yova Kementchedjhieva*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00278](https://doi.org/10.1109/CVPR52729.2023.00278)

        **Abstract**:

        Recent advances in image captioning have focused on scaling the data and model size, substantially increasing the cost of pretraining and finetuning. As an alternative to large models, we present Smallcap, which generates a caption conditioned on an input image and related captions retrieved from a datastore. Our model is lightweight and fast to train, as the only learned parameters are in newly introduced cross-attention layers between a pre-trained CLIP encoder and GPT-2 decoder. Smallcap can transfer to new domains without additional finetuning and can exploit large-scale data in a training-free fashion since the contents of the datastore can be readily replaced. Our experiments show that Smallcap, trained only on COCO, has competitive performance on this benchmark, and also transfers to other domains without retraining, solely through retrieval from target-domain data. Further improvement is achieved through the training-free exploitation of diverse human-labeled and web data, which proves to be effective for a range of domains, including the nocaps benchmark, designed to test generalization to unseen visual concepts.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/RitaRamo/smallcap.

        ----

        ## [271] Probing Sentiment-Oriented PreTraining Inspired by Human Sentiment Perception Mechanism

        **Authors**: *Tinglei Feng, Jiaxuan Liu, Jufeng Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00279](https://doi.org/10.1109/CVPR52729.2023.00279)

        **Abstract**:

        Pre-training of deep convolutional neural networks (DC-NNs) plays a crucial role in the field of visual sentiment analysis (VSA). Most proposed methods employ the off-the-shelf backbones pre-trained on large-scale object classification datasets (i.e., ImageNet). While it boosts performance for a big margin against initializing model states from random, we argue that DCNNs simply pre-trained on ImageNet may excessively focus on recognizing objects, but failed to provide high-level concepts in terms of sentiment. To address this long-term overlooked problem, we propose a sentiment-oriented pre-training method that is built upon human visual sentiment perception (VSP) mechanism. Specifically, we factorize the process of VSP into three steps, namely stimuli taking, holistic organizing, and high-level perceiving. From imitating each VSP step, a total of three models are separately pre-trained via our devised sentiment-aware tasks that contribute to excavating sentiment-discriminated representations. Moreover, along with our elaborated multi-model amalgamation strategy, the prior knowledge learned from each perception step can be effectively transferred into a single target model, yielding substantial performance gains. Finally, we verify the superiorities of our proposed method over extensive experiments, covering mainstream VSA tasks from single-label learning (SLL), multi-label learning (MLL), to label distribution learning (LDL). Experiment results demonstrate that our proposed method leads to unanimous improvements in these downstream tasks. Our code is released on https://github.com/tinglyfeng/sentiment_pretraining.

        ----

        ## [272] Prefix Conditioning Unifies Language and Label Supervision

        **Authors**: *Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, Tomas Pfister*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00280](https://doi.org/10.1109/CVPR52729.2023.00280)

        **Abstract**:

        Pretraining visual models on web-scale image-caption datasets has recently emerged as a powerful alternative to traditional pretraining on image classification data. Image-caption datasets are more “opendomain ”, containing broader scene types and vocabulary words, and result in models that have strong performance in fewand zero-shot recognition tasks. However large-scale classification datasets can provide fine-grained categories with a balanced label distribution. In this work, we study a pretraining strategy that uses both classification and caption datasets to unite their complementary benefits. First, we show that naively unifying the datasets results in sub-optimal performance in downstream zero-shot recognition tasks, as the model is affected by dataset bias: the coverage of image domains and vocabulary words is different in each dataset. We address this problem with novel Prefix Conditioning, a simple yet effective method that helps disentangle dataset biases from visual concepts. This is done by intro-ducing prefix tokens that inform the language encoder of the input data type (e.g., classification vs caption) at training time. Our approach allows the language encoder to learn from both datasets while also tailoring feature extraction to each dataset. Prefix conditioning is generic and can be easily integrated into existing VL pretraining objectives, such as CLIP or UniCL. In experiments, we show that it improves zero-shot image recognition and robustness to image-level distribution shift.

        ----

        ## [273] Crossing the Gap: Domain Generalization for Image Captioning

        **Authors**: *Yuchen Ren, Zhendong Mao, Shancheng Fang, Yan Lu, Tong He, Hao Du, Yongdong Zhang, Wanli Ouyang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00281](https://doi.org/10.1109/CVPR52729.2023.00281)

        **Abstract**:

        Existing image captioning methods are under the assumption that the training and testing data are from the same domain or that the data from the target domain (i.e., the domain that testing data lie in) are accessible. However, this assumption is invalid in real-world applications where the data from the target domain is inaccessible. In this paper, we introduce a new setting called Domain Generalization for Image Captioning (DGIC), where the data from the target domain is unseen in the learning process. We first construct a benchmark dataset for DGIC, which helps us to investigate models' domain generalization (DG) ability on unseen domains. With the support of the new benchmark, we further propose a new framework called language-guided semantic metric learning (LSML) for the DGIC setting. Experiments on multiple datasets demonstrate the challenge of the task and the effectiveness of our newly proposed benchmark and LSML framework.

        ----

        ## [274] A Bag-of-Prototypes Representation for Dataset-Level Applications

        **Authors**: *Weijie Tu, Weijian Deng, Tom Gedeon, Liang Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00282](https://doi.org/10.1109/CVPR52729.2023.00282)

        **Abstract**:

        This work investigates dataset vectorization for two dataset-level tasks: assessing training set suitability and test set difficulty. The former measures how suitable a training set is for a target domain, while the latter studies how challenging a test set is for a learned model. Central to the two tasks is measuring the underlying relationship between datasets. This needs a desirable dataset vectorization scheme, which should preserve as much discriminative dataset information as possible so that the distance between the resulting dataset vectors can reflect dataset-to-dataset similarity. To this end, we propose a bag-of-prototypes (BoP) dataset representation that extends the image-level bag consisting of patch descriptors to dataset-level bag consisting of semantic prototypes. Specifically, we develop a codebook consisting of K prototypes clustered from a ref-erence dataset. Given a dataset to be encoded, we quan-tize each of its image features to a certain prototype in the codebook and obtain a K -dimensional histogram. Without assuming access to dataset labels, the BoP representation provides rich characterization of the dataset semantic distribution. Furthermore, BoP representations cooperate well with Jensen-Shannon divergence for measuring dataset-to-dataset similarity. Although very simple, BoP consistently shows its advantage over existing representations on a series of benchmarks for two dataset-level tasks.

        ----

        ## [275] CrowdCLIP: Unsupervised Crowd Counting via Vision-Language Model

        **Authors**: *Dingkang Liang, Jiahao Xie, Zhikang Zou, Xiaoqing Ye, Wei Xu, Xiang Bai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00283](https://doi.org/10.1109/CVPR52729.2023.00283)

        **Abstract**:

        Supervised crowd counting relies heavily on costly manual labeling, which is difficult and expensive, especially in dense scenes. To alleviate the problem, we propose a novel unsupervised framework for crowd counting, named CrowdCLIP. The core idea is built on two observations: 1) the recent contrastive pre-trained vision-language model (CLIP) has presented impressive performance on various downstream tasks; 2) there is a natural mapping between crowd patches and count text. To the best of our knowledge, CrowdCLIP is the first to investigate the vision-language knowledge to solve the counting problem. Specifically, in the training stage, we exploit the multi-modal ranking loss by constructing ranking text prompts to match the size-sorted crowd patches to guide the image encoder learning. In the testing stage, to deal with the diversity of image patches, we propose a simple yet effective progressive filtering strategy to first select the highly potential crowd patches and then map them into the language space with various counting intervals. Extensive experiments on five challenging datasets demonstrate that the proposed CrowdCLIP achieves superior performance compared to previous unsupervised state-of-the-art counting methods. Notably, CrowdCLIP even surpasses some pop-ular fully-supervised methods under the cross-dataset setting. The source code will be available at https://github.com/dk-liang/CrowdCLIP.

        ----

        ## [276] D2Former: Jointly Learning Hierarchical Detectors and Contextual Descriptors via Agent-Based Transformers

        **Authors**: *Jianfeng He, Yuan Gao, Tianzhu Zhang, Zhe Zhang, Feng Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00284](https://doi.org/10.1109/CVPR52729.2023.00284)

        **Abstract**:

        Establishing pixel-level matches between image pairs is vital for a variety of computer vision applications. How-ever, achieving robust image matching remains challenging because CNN extracted descriptors usually lack discrim-inative ability in texture-less regions and keypoint detec-tors are only good at identifying keypoints with a specific level of structure. To deal with these issues, a novel im-age matching method is proposed by Jointly Learning Hier-archical Detectors and Contextual Descriptors via Agent-based Transformers (D
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
Former), including a contextual feature descriptor learning (CFDL) module and a hierar-chical keypoint detector learning (HKDL) module. The proposed D
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 Former enjoys several merits. First, the pro-posed CFDL module can model long-range contexts effi-ciently and effectively with the aid of designed descriptor agents. Second, the HKDL module can generate keypoint detectors in a hierarchical way, which is helpful for detecting keypoints with diverse levels of structures. Extensive experimental results on four challenging benchmarks show that our proposed method significantly outperforms state-of-the-art image matching methods.

        ----

        ## [277] Learning to Generate Language-Supervised and Open-Vocabulary Scene Graph Using Pre-Trained Visual-Semantic Space

        **Authors**: *Yong Zhang, Yingwei Pan, Ting Yao, Rui Huang, Tao Mei, Chang Wen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00285](https://doi.org/10.1109/CVPR52729.2023.00285)

        **Abstract**:

        Scene graph generation (SGG) aims to abstract an image into a graph structure, by representing objects as graph nodes and their relations as labeled edges. However, two knotty obstacles limit the practicability of current SGG methods in real-world scenarios: 1) training SGG models requires time-consuming ground-truth annotations, and 2) the closed-set object categories make the SGG models limited in their ability to recognize novel objects outside of training corpora. To address these issues, we novelly exploit a powerful pre-trained visual-semantic space (VSS) to trigger language-supervised and open-vocabulary SGG in a simple yet effective manner. Specifically, cheap scene graph supervision data can be easily obtained by parsing image language descriptions into semantic graphs. Next, the noun phrases on such semantic graphs are directly grounded over image regions through region-word alignment in the pre-trained VSS. In this way, we enable open-vocabulary object detection by performing object category name grounding with a text prompt in this VSS. On the basis of visually-grounded objects, the relation representations are naturally built for relation recognition, pursuing open-vocabulary SGG. We validate our proposed approach with extensive experiments on the Visual Genome benchmark across various SGG scenarios (i.e., supervised / language-supervised, closed-set / open-vocabulary). Consistent superior performances are achieved compared with existing methods, demonstrating the potential of exploiting pre-trained VSS for SGG in more practical scenarios.

        ----

        ## [278] Relational Context Learning for Human-Object Interaction Detection

        **Authors**: *Sanghyun Kim, Deunsol Jung, Minsu Cho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00286](https://doi.org/10.1109/CVPR52729.2023.00286)

        **Abstract**:

        Recent state-of-the-art methods for HOI detection typically build on transformer architectures with two decoder branches, one for human-object pair detection and the other for interaction classification. Such disentangled transformers, however, may suffer from insufficient context exchange between the branches and lead to a lack of context information for relational reasoning, which is critical in discovering HOI instances. In this work, we propose the multiplex relation network (MUREN) that performs rich context exchange between three decoder branches using unary, pair-wise, and ternary relations of human, object, and interaction tokens. The proposed method learns comprehensive relational contexts for discovering HOI instances, achieving state-of-the-art performance on two standard benchmarks for HOI detection, HICO-DET and V-COCO.

        ----

        ## [279] Learning Open-Vocabulary Semantic Segmentation Models From Natural Language Supervision

        **Authors**: *Jilan Xu, Junlin Hou, Yuejie Zhang, Rui Feng, Yi Wang, Yu Qiao, Weidi Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00287](https://doi.org/10.1109/CVPR52729.2023.00287)

        **Abstract**:

        This paper considers the problem of open-vocabulary semantic segmentation (OVS), that aims to segment objects of arbitrary classes beyond a pre-defined, closed-set categories. The main contributions are as follows: First, we propose a transformer-based model for OVS, termed as OVSegmentor, which only exploits web-crawled imagetext pairs for pre-training without using any mask annotations. OVSegmentor assembles the image pixels into a set of learnable group tokens via a slotattention based binding module, then aligns the group tokens to corresponding caption embeddings. Second, we propose two proxy tasks for training, namely masked entity completion and cross-image mask consistency. The former aims to infer all masked entities in the caption given group tokens, that enables the model to learn fine-grained alignment between visual groups and text entities. The latter enforces consistent mask predictions between images that contain shared entities, encouraging the model to learn visual invariance. Third, we construct CC4M dataset for pre-training by filtering CC12M with frequently appeared entities, which significantly improves training efficiency. Fourth, we perform zero-shot transfer on four benchmark datasets, PASCAL VOC, PASCAL Context, COCO Object, and ADE20K. OVSegmentor achieves superior results over state-of-the-art approaches on PASCAL VOC using only 3% data (4M vs 134M) for pre-training.

        ----

        ## [280] Side Adapter Network for Open-Vocabulary Semantic Segmentation

        **Authors**: *Mengde Xu, Zheng Zhang, Fangyun Wei, Han Hu, Xiang Bai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00288](https://doi.org/10.1109/CVPR52729.2023.00288)

        **Abstract**:

        This paper presents a new framework for open-vocabulary semantic segmentation with the pre-trained vision-language model, named Side Adapter Network (SAN). Our approach models the semantic segmentation task as a region recognition problem. A side network is attached to a frozen CLIP model with two branches: one for predicting mask proposals, and the other for predicting attention bias which is applied in the CLIP model to recognize the class of masks. This decoupled design has the benefit CLIP in recognizing the class of mask proposals. Since the attached side network can reuse CLIP features, it can be very light. In addition, the entire network can be trained end-to-end, allowing the side network to be adapted to the frozen CLIP model, which makes the predicted mask proposals CLIP-aware. Our approach is fast, accurate, and only adds a few additional trainable parameters. We evaluate our approach on multiple semantic segmentation benchmarks. Our method significantly outperforms other counterparts, with up to 18 times fewer trainable parameters and 19 times faster inference speed. Fig. 1 shows some visualization results on ImageNet. We hope our approach will serve as a solid baseline and help ease future research in open-vocabulary semantic segmentation.

        ----

        ## [281] Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models

        **Authors**: *Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, Shalini De Mello*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00289](https://doi.org/10.1109/CVPR52729.2023.00289)

        **Abstract**:

        We present ODISE: Open-vocabulary DIffusion-based panoptic SEgmentation, which unifies pre-trained text-image diffusion and discriminative models to perform open-vocabulary panoptic segmentation. Text-to-image diffusion models have the remarkable ability to generate high-quality images with diverse open-vocabulary language descriptions. This demonstrates that their internal representation space is highly correlated with open concepts in the real world. Text-image discriminative models like CLIP, on the other hand, are good at classifying images into open-vocabulary labels. We leverage the frozen internal representations of both these models to perform panoptic segmentation of any category in the wild. Our approach outperforms the previous state of the art by significant margins on both open-vocabulary panoptic and semantic segmentation tasks. In particular, with COCO training only, our method achieves 23.4 PQ and 30.0 mIoU on the ADE20K dataset, with 8.3 PQ and 7.9 mIoU absolute improvement over the previous state of the art. We open-source our code and models at https://github.com/NVlabs/ODISE.

        ----

        ## [282] IFSeg: Image-free Semantic Segmentation via Vision-Language Model

        **Authors**: *Sukmin Yun, Seong Hyeon Park, Paul Hongsuck Seo, Jinwoo Shin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00290](https://doi.org/10.1109/CVPR52729.2023.00290)

        **Abstract**:

        Vision-language (VL) pre-training has recently gained much attention for its transferability and flexibility in novel concepts (e.g., cross-modality transfer) across various visual tasks. However, VL-driven segmentation has been under-explored, and the existing approaches still have the burden of acquiring additional training images or even segmentation annotations to adapt a VL model to downstream segmentation tasks. In this paper, we introduce a novel image-free segmentation task where the goal is to perform semantic segmentation given only a set of the target semantic categories, but without any task-specific images and annotations. To tackle this challenging task, our proposed method, coined IFSeg, generates VL-driven artificial image-segmentation pairs and updates a pretrained VL model to a segmentation task. We construct this artificial training data by creating a 2D map of random semantic categories and another map of their corresponding word tokens. Given that a pretrained VL model projects visual and text tokens into a common space where tokens that share the semantics are located closely, this artificially generated word map can replace the real image inputs for such a VL model. Through an extensive set of experiments, our model not only establishes an effective baseline for this novel task but also demonstrates strong performances compared to existing methods that rely on stronger supervision, such as task-specific images and segmentation masks. Code is available at https://github.com/alinlab/ifseg.

        ----

        ## [283] PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations

        **Authors**: *Haoran Geng, Ziming Li, Yiran Geng, Jiayi Chen, Hao Dong, He Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00291](https://doi.org/10.1109/CVPR52729.2023.00291)

        **Abstract**:

        Learning a generalizable object manipulation policy is vital for an embodied agent to work in complex real-world scenes. Parts, as the shared components in different object categories, have the potential to increase the generalization ability of the manipulation policy and achieve cross-category object manipulation. In this work, we build the first large-scale, part-based cross-category object manipulation benchmark, PartManip, which is composed of 11 object categories, 494 objects, and 1432 tasks in 6 task classes. Compared to previous work, our benchmark is also more diverse and realistic, i.e., having more objects and using sparse-view point cloud as input without oracle information like part segmentation. To tackle the difficulties of vision-based policy learning, we first train a statebased expert with our proposed part-based canonicalization and part-aware rewards, and then distill the knowledge to a vision-based student. We also find an expressive backbone is essential to overcome the large diversity of different objects. For cross-category generalization, we introduce domain adversarial learning for domain-invariant feature extraction. Extensive experiments in simulation show that our learned policy can outperform other methods by a large margin, especially on unseen object categories. We also demonstrate our method can successfully manipulate novel objects in the real world. Our benchmark has been released in https://pku-epic.github.io/PartManip.

        ----

        ## [284] OneFormer: One Transformer to Rule Universal Image Segmentation

        **Authors**: *Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, Humphrey Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00292](https://doi.org/10.1109/CVPR52729.2023.00292)

        **Abstract**:

        Universal Image Segmentation is not a new concept. Past attempts to unify image segmentation include scene parsing, panoptic segmentation, and, more recently, new panoptic architectures. However, such panoptic architectures do not truly unify image segmentation because they need to be trained individually on the semantic, instance, or panoptic segmentation to achieve the best performance. Ideally, a truly universal framework should be trained only once and achieve SOTA performance across all three image segmentation tasks. To that end, we propose OneFormer, a universal image segmentation framework that unifies segmentation with a multi-task train-once design. We first propose a task-conditioned joint training strategy that enables training on ground truths of each domain (semantic, instance, and panoptic segmentation) within a single multi-task training process. Secondly, we introduce a task token to condition our model on the task at hand, making our model task-dynamic to support multi-task training and inference. Thirdly, we propose using a query-text contrastive loss during training to establish better intertask and inter-class distinctions. Notably, our single OneFormer model outperforms specialized Mask2Former models across all three segmentation tasks on ADE20k, Cityscapes, and COCO, despite the latter being trained on each task individually. We believe OneFormer is a significant step towards making image segmentation more universal and accessible.

        ----

        ## [285] Delving into Shape-aware Zero-shot Semantic Segmentation

        **Authors**: *Xinyu Liu, Beiwen Tian, Zhen Wang, Rui Wang, Kehua Sheng, Bo Zhang, Hao Zhao, Guyue Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00293](https://doi.org/10.1109/CVPR52729.2023.00293)

        **Abstract**:

        Thanks to the impressive progress of large-scale vision-language pretraining, recent recognition models can classify arbitrary objects in a zero-shot and open-set manner, with a surprisingly high accuracy. However, translating this success to semantic segmentation is not trivial, because this dense prediction task requires not only accurate semantic understanding but also fine shape delineation and existing vision-language models are trained with image-level language descriptions. To bridge this gap, we pursue shape-aware zero-shot semantic segmentation in this study. Inspired by classical spectral methods in the image segmentation literature, we propose to leverage the eigen vectors of Laplacian matrices constructed with self-supervised pixel-wise features to promote shape-awareness. Despite that this simple and effective technique does not make use of the masks of seen classes at all, we demonstrate that it out-performs a state-of-the-art shape-aware formulation that aligns ground truth and predicted edges during training. We also delve into the performance gains achieved on different datasets using different backbones and draw several interesting and conclusive observations: the benefits of promoting shape-awareness highly relates to mask compactness and language embedding locality. Finally, our method sets new state-of-the-art performance for zero-shot semantic segmentation on both Pascal and COCO, with significant margins. Code and models will be accessed at SAZS.

        ----

        ## [286] CoMFormer: Continual Learning in Semantic and Panoptic Segmentation

        **Authors**: *Fabio Cermelli, Matthieu Cord, Arthur Douillard*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00294](https://doi.org/10.1109/CVPR52729.2023.00294)

        **Abstract**:

        Continual learning for segmentation has recently seen increasing interest. However, all previous works focus on narrow semantic segmentation and disregard panoptic segmentation, an important task with real-world impacts. In this paper, we present the first continual learning model capable of operating on both semantic and panoptic segmentation. Inspired by recent transformer approaches that consider segmentation as a mask-classification problem, we design CoMFormer. Our method carefully exploits the properties of transformer architectures to learn new classes over time. Specifically, we propose a novel adaptive distillation loss along with a mask-based pseudo-labeling technique to effectively prevent forgetting. To evaluate our approach, we introduce a novel continual panoptic segmentation benchmark on the challenging ADE20K dataset. Our CoMFormer outperforms all the existing baselines by forgetting less old classes but also learning more effectively new classes. In addition, we also report an extensive evaluation in the large-scale continual semantic segmentation scenario showing that CoMFormer also significantly outperforms state-of-the-art methods. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/fcdl94/CoMFormer

        ----

        ## [287] Learning to Segment Every Referring Object Point by Point

        **Authors**: *Mengxue Qu, Yu Wu, Yunchao Wei, Wu Liu, Xiaodan Liang, Yao Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00295](https://doi.org/10.1109/CVPR52729.2023.00295)

        **Abstract**:

        Referring Expression Segmentation (RES) can facilitate pixel-level semantic alignment between vision and language. Most of the existing RES approaches require massive pixel-level annotations, which are expensive and exhaustive. In this paper, we propose a new partially supervised training paradigm for RES, i.e., training using abundant referring bounding boxes and only a few (e.g., 1%) pixel-level referring masks. To maximize the transferability from the REC model, we construct our model based on the point-based sequence prediction model. We propose the co-content teacher-forcing to make the model explicitly associate the point coordinates (scale values) with the referred spatial features, which alleviates the exposure bias caused by the limited segmentation masks. To make the most of referring bounding box annotations, we further propose the resampling pseudo points strategy to select more accurate pseudo-points as supervision. Extensive experiments show that our model achieves 52.06% in terms of accuracy (versus 58.93% in fully supervised setting) on Re-fCOCO+@testA, when only using 1% of the mask annotations. Code is available at https://github.com/qumengxue/Partial-RES.git.

        ----

        ## [288] Unsupervised Continual Semantic Adaptation Through Neural Rendering

        **Authors**: *Zhizheng Liu, Francesco Milano, Jonas Frey, Roland Siegwart, Hermann Blum, Cesar Cadena*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00296](https://doi.org/10.1109/CVPR52729.2023.00296)

        **Abstract**:

        An increasing amount of applications rely on data-driven models that are deployed for perception tasks across a sequence of scenes. Due to the mismatch between training and deployment data, adapting the model on the new scenes is often crucial to obtain good performance. In this work, we study continual multi-scene adaptation for the task of semantic segmentation, assuming that no ground-truth labels are available during deployment and that performance on the previous scenes should be maintained. We propose training a Semantic-NeRF network for each scene by fusing the predictions of a segmentation model and then using the view-consistent rendered semantic labels as pseudo-labels to adapt the model. Through joint training with the segmentation model, the Semantic-NeRF model effectively enables 2D-3D knowledge transfer. Furthermore, due to its compact size, it can be stored in a long-term memory and subsequently used to render data from arbitrary viewpoints to reduce forgetting. We evaluate our approach on Scan-Net, where we outperform both a voxel-based baseline and a state-of-the-art unsupervised domain adaptation method.

        ----

        ## [289] Mask DINO: Towards A Unified Transformer-based Framework for Object Detection and Segmentation

        **Authors**: *Feng Li, Hao Zhang, Huaizhe Xu, Shilong Liu, Lei Zhang, Lionel M. Ni, Heung-Yeung Shum*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00297](https://doi.org/10.1109/CVPR52729.2023.00297)

        **Abstract**:

        In this paper we present Mask DINO, a unified object detection and segmentation framework. Mask DINO extends DINO (DETR with Improved Denoising Anchor Boxes) by adding a mask prediction branch which supports all image segmentation tasks (instance, panoptic, and semantic). It makes use of the query embeddings from DINO to dot-product a high-resolution pixel embedding map to predict a set of binary masks. Some key components in DINO are extended for segmentation through a shared architecture and training process. Mask DINO is simple, efficient, and scalable, and it can benefit from joint large-scale detection and segmentation datasets. Our experiments show that Mask DINO significantly outperforms all existing specialized segmentation methods, both on a ResNet-50 backbone and a pre-trained model with SwinL backbone. Notably, Mask DINO establishes the best results to date on instance segmentation (54.5 AP on COCO), panoptic segmentation (59.4 PQ on COCO), and semantic segmentation (60.8 mIoU on ADE20K) among models under one billion parameters. Code is available at https://github.com/IDEA-Research/MaskDINO.

        ----

        ## [290] Transformer Scale Gate for Semantic Segmentation

        **Authors**: *Hengcan Shi, Munawar Hayat, Jianfei Cai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00298](https://doi.org/10.1109/CVPR52729.2023.00298)

        **Abstract**:

        Effectively encoding multi-scale contextual information is crucial for accurate semantic segmentation. Most of the existing transformer-based segmentation models combine features across scales without any selection, where features on sub-optimal scales may degrade segmentation outcomes. Leveraging from the inherent properties of Vision Transformers, we propose a simple yet effective module, Transformer Scale Gate (TSG), to optimally combine multi-scale features. TSG exploits cues in self and cross attentions in Vision Transformers for the scale selection. TSG is a highly flexible plug-and-play module, and can easily be incorporated with any encoder-decoder-based hierarchical vision Transformer. Extensive experiments on the Pascal Context, ADE20K and Cityscapes datasets demonstrate that the proposed feature selection strategy achieves consistent gains.

        ----

        ## [291] Style Projected Clustering for Domain Generalized Semantic Segmentation

        **Authors**: *Wei Huang, Chang Chen, Yong Li, Jiacheng Li, Cheng Li, Fenglong Song, Youliang Yan, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00299](https://doi.org/10.1109/CVPR52729.2023.00299)

        **Abstract**:

        Existing semantic segmentation methods improve generalization capability, by regularizing various images to a canonical feature space. While this process contributes to generalization, it weakens the representation inevitably. In contrast to existing methods, we instead utilize the difference between images to build a better representation space, where the distinct style features are extracted and stored as the bases of representation. Then, the generalization to unseen image styles is achieved by projecting features to this known space. Specifically, we realize the style projection as a weighted combination of stored bases, where the similarity distances are adopted as the weighting factors. Based on the same concept, we extend this process to the decision part of model and promote the generalization of semantic prediction. By measuring the similarity distances to semantic bases (i.e., prototypes), we replace the common deterministic prediction with semantic clustering. Comprehensive experiments demonstrate the advantage of proposed method to the state of the art, up to 3.6% mIoU improvement in average on unseen scenarios. Code and models are available at https://gitee.com/mindspore/models/tree/master/research/cv/SPC-Net.

        ----

        ## [292] Rethinking Few-Shot Medical Segmentation: A Vector Quantization View

        **Authors**: *Shiqi Huang, Tingfa Xu, Ning Shen, Feng Mu, Jianan Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00300](https://doi.org/10.1109/CVPR52729.2023.00300)

        **Abstract**:

        The existing few-shot medical segmentation networks share the same practice that the more prototypes, the better performance. This phenomenon can be theoretically interpreted in Vector Quantization (VQ) view: the more prototypes, the more clusters are separated from pixel-wise feature points distributed over the full space. However, as we further think about few-shot segmentation with this perspective, it is found that the clusterization of feature points and the adaptation to unseen tasks have not received enough attention. Motivated by the observation, we propose a learning VQ mechanism consisting of grid-format VQ (GFVQ), self-organized VQ (SOVQ) and residual oriented VQ (ROVQ). To be specific, GFVQ generates the prototype matrix by averaging square grids over the spatial extent, which uniformly quantizes the local details; SOVQ adaptively assigns the feature points to different local classes and creates a new representation space where the learnable local prototypes are updated with a global view; ROVQ introduces residual information to fine-tune the aforementioned learned local prototypes without retraining, which benefits the generalization performance for the irrelevance to the training task. We empirically show that our VQ framework yields the state-of-the-art performance over abdomen, cardiac and prostate MRI datasets and expect this work will provoke a rethink of the current few-shot medical segmentation model design. Our code will soon be publicly available.

        ----

        ## [293] Continual Semantic Segmentation with Automatic Memory Sample Selection

        **Authors**: *Lanyun Zhu, Tianrun Chen, Jianxiong Yin, Simon See, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00301](https://doi.org/10.1109/CVPR52729.2023.00301)

        **Abstract**:

        Continual Semantic Segmentation (CSS) extends static semantic segmentation by incrementally introducing new classes for training. To alleviate the catastrophic forgetting issue in CSS, a memory buffer that stores a small number of samples from the previous classes is constructed for replay. However, existing methods select the memory samples either randomly or based on a single-factor-driven handcrafted strategy, which has no guarantee to be optimal. In this work, we propose a novel memory sample selection mechanism that selects informative samples for effective replay in a fully automatic way by considering comprehensive factors including sample diversity and class performance. Our mechanism regards the selection operation as a decision-making process and learns an optimal selection policy that directly maximizes the validation performance on a reward set. To facilitate the selection decision, we design a novel state representation and a dual-stage action space. Our extensive experiments on Pascal-VOC 2012 and ADE 20K datasets demonstrate the effectiveness of our approach with state-of-the-art (SOTA) performance achieved, outperforming the second-place one by 12.54% for the 6-stage setting on Pascal-VOC 2012.

        ----

        ## [294] Token Contrast for Weakly-Supervised Semantic Segmentation

        **Authors**: *Lixiang Ru, Heliang Zheng, Yibing Zhan, Bo Du*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00302](https://doi.org/10.1109/CVPR52729.2023.00302)

        **Abstract**:

        Weakly-Supervised Semantic Segmentation (WSSS) using image-level labels typically utilizes Class Activation Map (CAM) to generate the pseudo labels. Limited by the local structure perception of CNN, CAM usually cannot identify the integral object regions. Though the recent Vision Transformer (ViT) can remedy this flaw, we observe it also brings the over-smoothing issue, i.e., the final patch tokens incline to be uniform. In this work, we propose Token Contrast (ToCo) to address this issue and further explore the virtue of ViT for WSSS. Firstly, motivated by the observation that intermediate layers in ViT can still retain semantic diversity, we designed a Patch Token Contrast module (PTC). PTC supervises the final patch tokens with the pseudo token relations derived from intermediate layers, allowing them to align the semantic regions and thus yield more accurate CAM. Secondly, to further differentiate the low-confidence regions in CAM, we devised a Class Token Contrast module (CTC) inspired by the fact that class tokens in ViT can capture high-level semantics. CTC facilitates the representation consistency between uncertain local regions and global objects by contrasting their class tokens. Experiments on the PASCAL VOC and MS COCO datasets show the proposed ToCo can remarkably surpass other single-stage competitors and achieve comparable performance with state-of-the-art multi-stage methods. Code is available at https://github.com/rulixiang/ToCo.

        ----

        ## [295] Multi-Granularity Archaeological Dating of Chinese Bronze Dings Based on a Knowledge-Guided Relation Graph

        **Authors**: *Rixin Zhou, Jiafu Wei, Qian Zhang, Ruihua Qi, Xi Yang, Chuntao Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00303](https://doi.org/10.1109/CVPR52729.2023.00303)

        **Abstract**:

        The archaeological dating of bronze dings has played a critical role in the study of ancient Chinese history. Current archaeology depends on trained experts to carry out bronze dating, which is time-consuming and labor-intensive. For such dating, in this study, we propose a learning-based approach to integrate advanced deep learning techniques and archaeological knowledge. To achieve this, we first collect a large-scale image dataset of bronze dings, which contains richer attribute information than other existing fine-grained datasets. Second, we introduce a multihead classifier and a knowledge-guided relation graph to mine the relationship between attributes and the ding era. Third, we conduct comparison experiments with various existing methods, the results of which show that our dating method achieves a state-of-the-art performance. We hope that our data and applied networks will enrich fine-grained classification re-search relevant to other interdisciplinary areas of expertise. The dataset and source code used are included in our supplementary materials, and will be open after submission owing to the anonymity policy. Source codes and data are available at: https://github.com/zhourixin/bronze-Ding

        ----

        ## [296] Hunting Sparsity: Density-Guided Contrastive Learning for Semi-Supervised Semantic Segmentation

        **Authors**: *Xiaoyang Wang, Bingfeng Zhang, Limin Yu, Jimin Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00304](https://doi.org/10.1109/CVPR52729.2023.00304)

        **Abstract**:

        Recent semi-supervised semantic segmentation methods combine pseudo labeling and consistency regularization to enhance model generalization from perturbation-invariant training. In this work, we argue that adequate supervision can be extracted directly from the geometry of feature space. Inspired by density-based unsupervised clustering, we propose to leverage feature density to locate sparse regions within feature clusters defined by label and pseudo labels. The hypothesis is that lower-density features tend to be under-trained compared with those densely gathered. Therefore, we propose to apply regularization on the structure of the cluster by tackling the sparsity to increase intra-class compactness in feature space. With this goal, we present a Density-Guided Contrastive Learning (DGCL) strategy to push anchor features in sparse regions toward cluster centers approximated by high-density positive keys. The heart of our method is to estimate feature density which is defined as neighbor compactness. We design a multi-scale density estimation module to obtain the density from multiple nearest-neighbor graphs for robust density modeling. Moreover, a unified training framework is proposed to combine label-guided self-training and density-guided geometry regularization to form complementary supervision on unlabeled data. Experimental results on PAS-CAL VOC and Cityscapes under various semi-supervised settings demonstrate that our proposed method achieves state-of-the-art performances. The project is available at https://github.com/Gavinwxy/DGCL.

        ----

        ## [297] Cut and Learn for Unsupervised Object Detection and Instance Segmentation

        **Authors**: *Xudong Wang, Rohit Girdhar, Stella X. Yu, Ishan Misra*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00305](https://doi.org/10.1109/CVPR52729.2023.00305)

        **Abstract**:

        We propose Cut-and-LEaRn (CutLER), a simple approach for training unsupervised object detection and seg-mentation models. We leverage the property of self-supervised models to ‘discover’ objects without supervision and amplify it to train a state-of-the-art localization model without any human labels. CutLER first uses our proposed MaskCut approach to generate coarse masks for multiple objects in an image, and then learns a detector on these masks using our robust loss function. We further improve performance by self-training the model on its predictions. Compared to prior work, CutLER is simpler, compatible with different detection architectures, and detects multiple objects. CutLER is also a zero-shot unsupervised detector and improves detection performance AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
 by over 2.7× on 11 benchmarks across domains like video frames, paintings, sketches, etc. With finetuning, CutLER serves as a low-shot detector surpassing MoCo-v2 by 7.3% Ap
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">box</sup>
 and 6.6% Ap
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">mask</sup>
 on COCO when training with 5% labels.

        ----

        ## [298] Extracting Class Activation Maps from Non-Discriminative Features as well

        **Authors**: *Zhaozheng Chen, Qianru Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00306](https://doi.org/10.1109/CVPR52729.2023.00306)

        **Abstract**:

        Extracting class activation maps (CAM) from a classification model often results in poor coverage on foreground objects, i.e., only the discriminative region (e.g., the “head” of “sheep”) is recognized and the rest (e.g., the “leg” of “sheep”) mistakenly as background. The crux behind is that the weight of the classifier (used to compute CAM) captures only the discriminative features of objects. We tackle this by introducing a new computation method for CAM that explicitly captures non-discriminative features as well, thereby expanding CAM to cover whole objects. Specifically, we omit the last pooling layer of the classification model, and perform clustering on all local features of an object class, where “local” means “at a spatial pixel position”. We call the resultant K cluster centers local prototypes — represent local semantics like the “head”, “leg”, and “body” of “sheep”. Given a new image of the class, we compare its unpooled features to every prototype, derive K similarity matrices, and then aggregate them into a heatmap (i.e., our CAM). Our CAM thus captures all local features of the class without discrimination. We evaluate it in the challenging tasks of weakly-supervised semantic segmentation (WSSS), and plug it in multiple state-of-the-art WSSS methods, such as MCTformer [45] and AMN [26], by simply replacing their original CAM with ours. Our extensive experiments on standard WSSS benchmarks (PASCAL VOC and MS COCO) show the superiority of our method: consistent improvements with little computational overhead. Our code is provided at https://github.com/zhaozhengChen/LPCAM.

        ----

        ## [299] BoxTeacher: Exploring High-Quality Pseudo Labels for Weakly Supervised Instance Segmentation

        **Authors**: *Tianheng Cheng, Xinggang Wang, Shaoyu Chen, Qian Zhang, Wenyu Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00307](https://doi.org/10.1109/CVPR52729.2023.00307)

        **Abstract**:

        Labeling objects with pixel-wise segmentation requires a huge amount of human labor compared to bounding boxes. Most existing methods for weakly supervised instance segmentation focus on designing heuristic losses with priors from bounding boxes. While, we find that box-supervised methods can produce some fine segmentation masks and we wonder whether the detectors could learn from these fine masks while ignoring low-quality masks. To answer this question, we present BoxTeacher, an efficient and end-to-end training framework for high-performance weakly supervised instance segmentation, which leverages a sophisticated teacher to generate high-quality masks as pseudo labels. Considering the massive noisy masks hurt the training, we present a mask-aware confidence score to estimate the quality of pseudo masks, and propose the noiseaware pixel loss and noise-reduced affinity loss to adaptively optimize the student with pseudo masks. Extensive experiments can demonstrate effectiveness of the proposed BoxTeacher. Without bells and whistles, BoxTeacher remarkably achieves 35.0 mask AP and 36.5 mask AP with ResNet-50 and ResNet-101 respectively on the challenging COCO dataset, which outperforms the previous state-of-the-art methods by a significant margin and bridges the gap between box-supervised and mask-supervised methods.

        ----

        ## [300] Hierarchical Fine-Grained Image Forgery Detection and Localization

        **Authors**: *Xiao Guo, Xiaohong Liu, Zhiyuan Ren, Steven Grosz, Iacopo Masi, Xiaoming Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00308](https://doi.org/10.1109/CVPR52729.2023.00308)

        **Abstract**:

        Differences in forgery attributes of images generated in CNN-synthesized and image-editing domains are large, and such differences make a unified image forgery detection and localization (IFDL) challenging. To this end, we present a hierarchical fine-grained formulation for IFDL representation learning. Specifically, we first represent forgery attributes of a manipulated image with multiple labels at different levels. Then we perform fine-grained classification at these levels using the hierarchical dependency between them. As a result, the algorithm is encouraged to learn both comprehensive features and inherent hierarchical nature of different forgery attributes, thereby improving the IFDL representation. Our proposed IFDL framework contains three components: multi-branch feature extractor, localization and classification modules. Each branch of the feature extractor learns to classify forgery attributes at one level, while localization and classification modules segment the pixel-level forgery region and detect image-level forgery, respectively. Lastly, we construct a hierarchical fine-grained dataset to facilitate our study. We demonstrate the effectiveness of our method on 7 different benchmarks, for both tasks of IFDL and forgery attribute classification. Our source code and dataset can be found: github.com/CHELSEA234/HiFi-IFDL.

        ----

        ## [301] Towards Professional Level Crowd Annotation of Expert Domain Data

        **Authors**: *Pei Wang, Nuno Vasconcelos*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00309](https://doi.org/10.1109/CVPR52729.2023.00309)

        **Abstract**:

        Image recognition on expert domains is usually fine-grained and requires expert labeling, which is costly. This limits dataset sizes and the accuracy of learning systems. To address this challenge, we consider annotating expert data with crowdsourcing. This is denoted as PrOfeSsional lEvel cRowd (POSER) annotation. A new approach, based on semi-supervised learning (SSL) and denoted as SSL with human filtering (SSL-HF) is proposed. It is a human-in-the-loop SSL method, where crowd-source workers act as filters of pseudo-labels, replacing the unreliable confidence thresholding used by state-of-the-art SSL methods. To enable annotation by non-experts, classes are specified implicitly, via positive and negative sets of examples and augmented with deliberative explanations, which highlight regions of class ambiguity. In this way, SSL-HF leverages the strong low-shot learning and confidence estimation ability of humans to create an intuitive but effective labeling experience. Experiments show that SSL-HF significantly outperforms various alternative approaches in several benchmarks.

        ----

        ## [302] Unsupervised Object Localization: Observing the Background to Discover Objects

        **Authors**: *Oriane Siméoni, Chloé Sekkat, Gilles Puy, Antonín Vobecký, Éloi Zablocki, Patrick Pérez*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00310](https://doi.org/10.1109/CVPR52729.2023.00310)

        **Abstract**:

        Recent advances in self-supervised visual representation learning have paved the way for unsupervised methods tackling tasks such as object discovery and instance segmentation. However, discovering objects in an image with no supervision is a very hard task; what are the desired objects, when to separate them into parts, how many are there, and of what classes? The answers to these questions de-pend on the tasks and datasets of evaluation. In this work, we take a different approach and propose to look for the background instead. This way, the salient objects emerge as a by-product without any strong assumption on what an object should be. We propose FOUND, a simple model made of a single conv1 x 1 initialized with coarse background masks extracted from self-supervised patch-based representations. After fast training and refining these seed masks, the model reaches state-of-the-art results on unsupervised saliency detection and object discovery benchmarks. Moreover, we show that our approach yields good results in the unsupervised semantic segmentation retrieval task. The code to reproduce our results is available at https://github.com/valeoai/FOUND.

        ----

        ## [303] Semi-supervised learning made simple with self-supervised clustering

        **Authors**: *Enrico Fini, Pietro Astolfi, Karteek Alahari, Xavier Alameda-Pineda, Julien Mairal, Moin Nabi, Elisa Ricci*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00311](https://doi.org/10.1109/CVPR52729.2023.00311)

        **Abstract**:

        Self-supervised learning models have been shown to learn rich visual representations without requiring human annotations. However, in many real-world scenarios, labels are partially available, motivating a recent line of work on semi-supervised methods inspired by self-supervised principles. In this paper, we propose a conceptually simple yet empirically powerful approach to turn clustering-based self-supervised methods such as SwAV or DINO into semi-supervised learners. More precisely, we introduce a multi-task framework merging a supervised objective using ground-truth labels and a self-supervised objective relying on clustering assignments with a single cross-entropy loss. This approach may be interpreted as imposing the cluster centroids to be class prototypes. Despite its simplicity, we provide empirical evidence that our approach is highly effective and achieves state-of-the-art performance on CI-FAR100 and ImageNet.

        ----

        ## [304] Unbalanced Optimal Transport: A Unified Framework for Object Detection

        **Authors**: *Henri De Plaen, Pierre-François De Plaen, Johan A. K. Suykens, Marc Proesmans, Tinne Tuytelaars, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00312](https://doi.org/10.1109/CVPR52729.2023.00312)

        **Abstract**:

        During training, supervised object detection tries to correctly match the predicted bounding boxes and associated classification scores to the ground truth. This is essential to determine which predictions are to be pushed towards which solutions, or to be discarded. Popular matching strategies include matching to the closest ground truth box (mostly used in combination with anchors), or matching via the Hungarian algorithm (mostly used in anchor free methods). Each of these strategies comes with its own properties, underlying losses, and heuristics. We show how Unbalanced Optimal Transport unifies these different approaches and opens a whole continuum of methods in between. This allows for a finer selection of the desired properties. Experimentally, we show that training an object detection model with Unbalanced Optimal Transport is able to reach the state-of-the-art both in terms of Average Precision and Average Recall as well as to provide a faster initial convergence. The approach is well suited for GPU implementation, which proves to be an advantage for large-scale models.

        ----

        ## [305] DiGeo: Discriminative Geometry-Aware Learning for Generalized Few-Shot Object Detection

        **Authors**: *Jiawei Ma, Yulei Niu, Jincheng Xu, Shiyuan Huang, Guangxing Han, Shih-Fu Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00313](https://doi.org/10.1109/CVPR52729.2023.00313)

        **Abstract**:

        Generalized few-shot object detection aims to achieve precise detection on both base classes with abundant annotations and novel classes with limited training data. Existing approaches enhance few-shot generalization with the sacrifice of base-class performance, or maintain high precision in base-class detection with limited improvement in novel-class adaptation. In this paper, we point out the reason is insufficient Discriminative feature learning for all of the classes. As such, we propose a new training framework, DiGeo, to learn Geometry-aware features of interclass separation and intra-class compactness. To guide the separation of feature clusters, we derive an offline simplex equiangular tight frame (ETF) classifier whose weights serve as class centers and are maximally and equally separated. To tighten the cluster for each class, we include adaptive class-specific margins into the classification loss and encourage the features close to the class centers. Experimental studies on two few-shot benchmark datasets (VOC, COCO) and one long-tail dataset (LVIS) demonstrate that, with a single model, our method can effectively improve generalization on novel classes without hurting the detection of base classes. Our code can be found here.

        ----

        ## [306] CLIP the Gap: A Single Domain Generalization Approach for Object Detection

        **Authors**: *Vidit Vidit, Martin Engilberge, Mathieu Salzmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00314](https://doi.org/10.1109/CVPR52729.2023.00314)

        **Abstract**:

        Single Domain Generalization (SDG) tackles the problem of training a model on a single source domain so that it generalizes to any unseen target domain. While this has been well studied for image classification, the literature on SDG object detection remains almost non-existent. To address the challenges of simultaneously learning robust object localization and representation, we propose to leverage a pre-trained vision-language model to introduce semantic domain concepts via textual prompts. We achieve this via a semantic augmentation strategy acting on the features extracted by the detector backbone, as well as a text-based classification loss. Our experiments evidence the benefits of our approach, outperforming by 10% the only existing SDG object detection method, Single-DGOD [52], on their own diverse weather-driving benchmark.

        ----

        ## [307] Unknown Sniffer for Object Detection: Don't Turn a Blind Eye to Unknown Objects

        **Authors**: *Wenteng Liang, Feng Xue, Yihao Liu, Guofeng Zhong, Anlong Ming*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00315](https://doi.org/10.1109/CVPR52729.2023.00315)

        **Abstract**:

        The recently proposed open-world object and open-set detection have achieved a breakthrough in finding never-seen-before objects and distinguishing them from known ones. However, their studies on knowledge transfer from known classes to unknown ones are not deep enough, resulting in the scanty capability for detecting unknowns hidden in the background. In this paper, we propose the unknown sniffer (UnSniffer) to find both unknown and known objects. Firstly, the generalized object confidence (GOC) score is introduced, which only uses known samples for supervision and avoids improper suppression of unknowns in the back-ground. Significantly, such confidence score learned from known objects can be generalized to unknown ones. Additionally, we propose a negative energy suppression loss to further suppress the non-object samples in the background. Next, the best box of each unknown is hard to obtain during inference due to lacking their semantic information in training. To solve this issue, we introduce a graph-based determination scheme to replace hand-designed non-maximum suppression (NMS) post-processing. Finally, we present the Unknown Object Detection Benchmark, the first publicly benchmark that encompasses precision evaluation for unknown detection to our knowledge. Experiments show that our method is far better than the existing state-of-the-art methods. Code is available at: https://github.com/Went-Liang/UnSniffer.

        ----

        ## [308] Consistent-Teacher: Towards Reducing Inconsistent Pseudo-Targets in Semi-Supervised Object Detection

        **Authors**: *Xinjiang Wang, Xingyi Yang, Shilong Zhang, Yijiang Li, Litong Feng, Shijie Fang, Chengqi Lyu, Kai Chen, Wayne Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00316](https://doi.org/10.1109/CVPR52729.2023.00316)

        **Abstract**:

        In this study, we dive deep into the inconsistency of pseudo targets in semi-supervised object detection (SSOD). Our core observation is that the oscillating pseudo-targets undermine the training of an accurate detector. It injects noise into the student's training, leading to severe overfitting problems. Therefore, we propose a systematic solution, termed Consistent-Teacher, to reduce the inconsistency. First, adaptive anchor assignment (ASA) substitutes the static IoU-based strategy, which enables the student network to be resistant to noisy pseudo-bounding boxes. Then we calibrate the subtask predictions by designing a 3D feature alignment module (FAM-3D). It allows each classification feature to adaptively query the optimal feature vector for the regression task at arbitrary scales and locations. Lastly, a Gaussian Mixture Model (GMM) dynamically revises the score threshold of pseudo-bboxes, which stabilizes the number of ground truths at an early stage and remedies the unreliable supervision signal during training. Consistent-Teacher provides strong results on a large range of SSOD evaluations. It achieves 40.0 mAP with ResNet-50 backbone given only 10% of annotated MS-COCO data, which surpasses previous base-lines using pseudo labels by around 3 mAP. When trained on fully annotated MS-COCO with additional unlabeled data, the performance further increases to 47.7 mAP. Our code is available at https://github.com/Adamdad/ConsistentTeacher.

        ----

        ## [309] Optimal Proposal Learning for Deployable End-to-End Pedestrian Detection

        **Authors**: *Xiaolin Song, Binghui Chen, Pengyu Li, Jun-Yan He, Biao Wang, Yifeng Geng, Xuansong Xie, Honggang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00317](https://doi.org/10.1109/CVPR52729.2023.00317)

        **Abstract**:

        End-to-end pedestrian detection focuses on training a pedestrian detection model via discarding the Non-Maximum Suppression (NMS) post-processing. Though a few methods have been explored, most of them still suffer from longer training time and more complex deployment, which cannot be deployed in the actual industrial applications. In this paper, we intend to bridge this gap and propose an Optimal Proposal Learning (OPL) framework for deployable end-to-end pedestrian detection. Specifically, we achieve this goal by using CNN-based light detector and introducing two novel modules, including a Coarse-to-Fine (C2F) learning strategy for proposing precise positive proposals for the Ground-Truth (GT) instances by reducing the ambiguity of sample assignment/output in training/testing respectively, and a Completed Proposal Network (CPN) for producing extra information compensation to further recall the hard pedestrian samples. Extensive experiments are conducted on CrowdHuman, TJU-Ped and Caltech, and the results show that our proposed OPL method significantly outperforms the competing methods.

        ----

        ## [310] AsyFOD: An Asymmetric Adaptation Paradigm for Few-Shot Domain Adaptive Object Detection

        **Authors**: *Yipeng Gao, Kun-Yu Lin, Junkai Yan, Yaowei Wang, Wei-Shi Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00318](https://doi.org/10.1109/CVPR52729.2023.00318)

        **Abstract**:

        In this work, we study few-shot domain adaptive object detection (FSDAOD), where only a few target labeled images are available for training in addition to sufficient source labeled images. Critically, in FSDAOD, the data scarcity in the target domain leads to an extreme data imbalance between the source and target domains, which potentially causes over-adaptation in traditional feature alignment. To address the data imbalance problem, we propose an asymmetric adaptation paradigm, namely AsyFOD, which leverages the source and target instances from different perspectives. Specifically, by using target distribution estimation, the AsyFOD first identifies the target-similar source instances, which serves to augment the limited target instances. Then, we conduct asynchronous alignment between target-dissimilar source instances and augmented target instances, which is simple yet effective for alleviating the over-adaptation. Extensive experiments demonstrate that the proposed AsyFOD outperforms all state-of-the-art methods on four FSDAOD benchmarks with various environmental variances, e.g., 3.1% mAP improvement on Cityscapes-to-FoggyCityscapes and 2.9% mAP increase on Sim10k-to-Cityscapes. The code is available at https://github.com/Hlings/AsyFPD.

        ----

        ## [311] Where is My Spot? Few-shot Image Generation via Latent Subspace Optimization

        **Authors**: *Chenxi Zheng, Bangzhen Liu, Huaidong Zhang, Xuemiao Xu, Shengfeng He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00319](https://doi.org/10.1109/CVPR52729.2023.00319)

        **Abstract**:

        Image generation relies on massive training data that can hardly produce diverse images of an unseen category according to a few examples. In this paper, we address this dilemma by projecting sparse few-shot samples into a continuous latent space that can potentially generate infinite unseen samples. The rationale behind is that we aim to locate a centroid latent position in a conditional StyleGAN, where the corresponding output image on that centroid can maximize the similarity with the given samples. Although the given samples are unseen for the conditional StyleGAN, we assume the neighboring latent subspace around the centroid belongs to the novel category, and therefore introduce two latent subspace optimization objectives. In the first one we use few-shot samples as positive anchors of the novel class, and adjust the StyleGAN to produce the corresponding results with the new class label condition. The second objective is to govern the generation process from the other way around, by altering the centroid and its surrounding latent subspace for a more precise generation of the novel class. These reciprocal optimization objectives inject a novel class into the StyleGAN latent subspace, and therefore new unseen samples can be easily produced by sampling images from it. Extensive experiments demonstrate superior few-shot generation performances compared with state-of-the-art methods, especially in terms of diversity and generation quality. Code is available at https://github.com/chansey0529/LSO.

        ----

        ## [312] Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection

        **Authors**: *Fan Lu, Kai Zhu, Wei Zhai, Kecheng Zheng, Yang Cao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00320](https://doi.org/10.1109/CVPR52729.2023.00320)

        **Abstract**:

        Semantically coherent out-of-distribution (SCOOD) detection aims to discern outliers from the intended data distribution with access to unlabeled extra set. The coexistence of in-distribution and out-of-distribution samples will exacerbate the model overfitting when no distinction is made. To address this problem, we propose a novel uncertainty-aware optimal transport scheme. Our scheme consists of an energy-based transport (ET) mechanism that estimates the fluctuating cost of uncertainty to promote the assignment of semantic-agnostic representation, and an inter-cluster extension strategy that enhances the discrimination of semantic property among different clusters by widening the corresponding margin distance. Furthermore, a T-energy score is presented to mitigate the magnitude gap between the parallel transport and classifier branches. Extensive experiments on two standard SCOOD benchmarks demonstrate the above-par OOD detection performance, outperforming the state-of-the-art methods by a margin of 27.69% and 34.4% on FPR@95, respectively. Code is available at https://github.com/LuFan3/IET-OOD.

        ----

        ## [313] MAESTER: Masked Autoencoder Guided Segmentation at Pixel Resolution for Accurate, Self-Supervised Subcellular Structure Recognition

        **Authors**: *Ronald Xie, Kuan Pang, Gary D. Bader, Bo Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00321](https://doi.org/10.1109/CVPR52729.2023.00321)

        **Abstract**:

        Accurate segmentation of cellular images remains an elusive task due to the intrinsic variability in morphology of biological structures. Complete manual segmentation is unfeasible for large datasets, and while supervised methods have been proposed to automate segmentation, they often rely on manually generated ground truths which are especially challenging and time consuming to generate in biology due to the requirement of domain expertise. Furthermore, these methods have limited generalization capacity, requiring additional manual labels to be generated for each dataset and use case. We introduce MAESTER (Masked AutoEncoder guided Segmen'Iation at pixEl Resolution), a self-supervised method for accurate, subcellular structure segmentation at pixel resolution. MAESTER treats segmentation as a representation learning and clustering problem. Specifically, MAESTER learns semantically meaningful token representations of multi-pixel image patches while simultaneously maintaining a sufficiently large field of view for contextual learning. We also develop a cover-and-stride inference strategy to achieve pixel-level subcellular structure segmentation. We evaluated MAESTER on a publicly available volumetric electron microscopy (VEM) dataset of primary mouse pancreatic islets ß cells and achieved up-wards of 29.1 % improvement over state-of-the-art under the same evaluation criteria. Furthermore, our results are competitive against supervised methods trained on the same tasks, closing the gap between self-supervised and supervised approaches. MAESTER shows promise for alleviating the critical bottleneck of ground truth generation for imaging related data analysis and thereby greatly increasing the rate of biological discovery. Code available at https://github.com/bowang-lab/MAESTER.

        ----

        ## [314] Orthogonal Annotation Benefits Barely-supervised Medical Image Segmentation

        **Authors**: *Heng Cai, Shumeng Li, Lei Qi, Qian Yu, Yinghuan Shi, Yang Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00322](https://doi.org/10.1109/CVPR52729.2023.00322)

        **Abstract**:

        Recent trends in semi-supervised learning have significantly boosted the performance of 3D semi-supervised medical image segmentation. Compared with 2D images, 3D medical volumes involve information from different directions, e.g., transverse, sagittal, and coronal planes, so as to naturally provide complementary views. These complementary views and the intrinsic similarity among adjacent 3D slices inspire us to develop a novel annotation way and its corresponding semi-supervised model for effective segmentation. Specifically, we firstly propose the orthogonal annotation by only labeling two orthogonal slices in a labeled volume, which significantly relieves the burden of annotation. Then, we perform registration to obtain the initial pseudo labels for sparsely labeled volumes. Subsequently, by introducing unlabeled volumes, we propose a dual-network paradigm named Dense-Sparse Co-training (DeSCO) that exploits dense pseudo labels in early stage and sparse labels in later stage and meanwhile forces consistent output of two networks. Experimental results on three benchmark datasets validated our effectiveness in performance and efficiency in annotation. For example, with only 10 annotated slices, our method reaches a Dice up to 86.93% on KiTS19 dataset. Our code and models are available at https://github.com/HengCai-NJU/DeSCO.

        ----

        ## [315] RepMode: Learning to Re-Parameterize Diverse Experts for Subcellular Structure Prediction

        **Authors**: *Donghao Zhou, Chunbin Gu, Junde Xu, Furui Liu, Qiong Wang, Guangyong Chen, Pheng-Ann Heng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00323](https://doi.org/10.1109/CVPR52729.2023.00323)

        **Abstract**:

        In biological research, fluorescence staining is a key technique to reveal the locations and morphology of subcellular structures. However, it is slow, expensive, and harmful to cells. In this paper, we model it as a deep learning task termed subcellular structure prediction (SSP), aiming to predict the 3D fluorescent images of multiple subcellular structures from a 3D transmitted-light image. Unfortunately, due to the limitations of current biotechnology, each image is partially labeled in SSP. Besides, naturally, subcellular structures vary considerably in size, which causes the multi-scale issue of SSP. To overcome these challenges, we propose Re-parameterizing Mixture-of-Diverse-Experts (RepMode), a network that dynamically organizes its parameters with task-aware priors to handle specified single-label prediction tasks. In RepMode, the Mixture-of-Diverse-Experts (MoDE) block is designed to learn the generalized parameters for all tasks, and gating re-parameterization (GatRep) is performed to generate the specialized parameters for each task, by which RepMode can maintain a compact practical topology exactly like a plain network, and meanwhile achieves a powerful theoretical topology. Comprehensive experiments show that RepMode can achieve state-of-the-art overall performance in SSP.

        ----

        ## [316] Topology-Guided Multi-Class Cell Context Generation for Digital Pathology

        **Authors**: *Shahira Abousamra, Rajarsi Gupta, Tahsin M. Kurç, Dimitris Samaras, Joel H. Saltz, Chao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00324](https://doi.org/10.1109/CVPR52729.2023.00324)

        **Abstract**:

        In digital pathology, the spatial context of cells is important for cell classification, cancer diagnosis and prognosis. To model such complex cell context, however, is challenging. Cells form different mixtures, lineages, clusters and holes. To model such structural patterns in a learnable fashion, we introduce several mathematical tools from spatial statistics and topological data analysis. We incorporate such structural descriptors into a deep generative model as both conditional inputs and a differentiable loss. This way, we are able to generate high quality multi-class cell layouts for the first time. We show that the topology-rich cell layouts can be used for data augmentation and improve the performance of downstream tasks such as cell classification.

        ----

        ## [317] Dynamic Graph Enhanced Contrastive Learning for Chest X-Ray Report Generation

        **Authors**: *Mingjie Li, Bingqian Lin, Zicong Chen, Haokun Lin, Xiaodan Liang, Xiaojun Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00325](https://doi.org/10.1109/CVPR52729.2023.00325)

        **Abstract**:

        Automatic radiology reporting has great clinical potential to relieve radiologists from heavy workloads and improve diagnosis interpretation. Recently, researchers have enhanced data-driven neural networks with medical knowledge graphs to eliminate the severe visual and textual bias in this task. The structures of such graphs are exploited by using the clinical dependencies formed by the disease topic tags via general knowledge and usually do not update during the training process. Consequently, the fixed graphs can not guarantee the most appropriate scope of knowledge and limit the effectiveness. To address the limitation, we propose a knowledge graph with Dynamic structure and nodes to facilitate chest X-ray report generation with Contrastive Learning, named DCL. In detail, the fundamental structure of our graph is pre-constructed from general knowledge. Then we explore specific knowledge extracted from the retrieved reports to add additional nodes or redefine their relations in a bottom-up manner. Each image feature is integrated with its very own updated graph before being fed into the decoder module for report generation. Finally, this paper introduces Image-Report Contrastive and Image-Report Matching losses to better represent visual features and textual information. Evaluated on IU-Xray and MIMIC-CXR datasets, our DCL outperforms previous state-of-the-art models on these two benchmarks.

        ----

        ## [318] Benchmarking Self-Supervised Learning on Diverse Pathology Datasets

        **Authors**: *Mingu Kang, Heon Song, Seonwook Park, Donggeun Yoo, Sérgio Pereira*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00326](https://doi.org/10.1109/CVPR52729.2023.00326)

        **Abstract**:

        Computational pathology can lead to saving human lives, but models are annotation hungry and pathology images are notoriously expensive to annotate. Self-supervised learning (SSL) has shown to be an effective method for utilizing unlabeled data, and its application to pathology could greatly benefit its downstream tasks. Yet, there are no principled studies that compare SSL methods and discuss how to adapt them for pathology. To address this need, we execute the largest-scale study of SSL pre-training on pathology image data, to date. Our study is conducted using 4 representative SSL methods on diverse downstream tasks. We establish that large-scale domain-aligned pre-training in pathology consistently out-performs ImageNet pre-training in standard SSL settings such as linear and fine-tuning evaluations, as well as in low-label regimes. Moreover, we propose a set of domain-specific techniques that we experimentally show leads to a performance boost. Lastly, for the first time, we apply SSL to the challenging task of nuclei instance segmentation and show large and consistent performance improvements. We release the pre-trained model weights
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://lunit-io.github.io/research/publications/pathology_ssl.

        ----

        ## [319] Multiple Instance Learning via Iterative Self-Paced Supervised Contrastive Learning

        **Authors**: *Kangning Liu, Weicheng Zhu, Yiqiu Shen, Sheng Liu, Narges Razavian, Krzysztof J. Geras, Carlos Fernandez-Granda*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00327](https://doi.org/10.1109/CVPR52729.2023.00327)

        **Abstract**:

        Learning representations for individual instances when only bag-level labels are available is a fundamental challenge in multiple instance learning (MIL). Recent works have shown promising results using contrastive self-supervised learning (CSSL), which learns to push apart representations corresponding to two different randomly-selected instances. Unfortunately, in real-world applications such as medical image classification, there is often class imbalance, so randomly-selected instances mostly belong to the same majority class, which precludes CSSL from learning inter-class differences. To address this issue, we propose a novel framework, Iterative Self-paced Supervised Contrastive Learning for MIL Representations (ItS2CLR), which improves the learned representation by exploiting instance-level pseudo labels derived from the bag-level labels. The framework employs a novel self-paced sampling strategy to ensure the accuracy of pseudo labels. We evaluate ItS2CLR on three medical datasets, showing that it improves the quality of instance-level pseudo labels and representations, and outperforms existing MIL methods in terms of both bag and instance level accuracy.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/Kangningthu/ItS2CLR

        ----

        ## [320] Learning Expressive Prompting With Residuals for Vision Transformers

        **Authors**: *Rajshekhar Das, Yonatan Dukler, Avinash Ravichandran, Ashwin Swaminathan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00328](https://doi.org/10.1109/CVPR52729.2023.00328)

        **Abstract**:

        Prompt learning is an efficient approach to adapt transformers by inserting learnable set of parameters into the input and intermediate representations of a pre-trained model. In this work, we present Expressive Prompts with Residuals (EXPRES) which modifies the prompt learning paradigm specifically for effective adaptation of vision transformers (ViT). Our method constructs downstream representations via learnable “output” tokens (shal-low prompts), that are akin to the learned class tokens of the ViT. Further for better steering of the downstream representation processed by the frozen transformer, we introduce residual learnable tokens that are added to the output of various computations. We apply EXPRES for image classification and few-shot semantic segmentation, and show our method is capable of achieving state of the art prompt tuning on 3/3 categories of the VTAB benchmark. In addition to strong performance, we observe that our approach is an order of magnitude more prompt efficient than existing visual prompting baselines. We analytically show the computational benefits of our approach over weight space adaptation techniques like finetuning. Lastly we systematically corroborate the architectural design of our method via a series of ablation experiments.

        ----

        ## [321] Detection of Out-of-Distribution Samples Using Binary Neuron Activation Patterns

        **Authors**: *Bartlomiej Olber, Krystian Radlak, Adam Popowicz, Michal Szczepankiewicz, Krystian Chachula*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00329](https://doi.org/10.1109/CVPR52729.2023.00329)

        **Abstract**:

        Deep neural networks (DNN) have outstanding performance in various applications. Despite numerous efforts of the research community, out-of-distribution (OOD) samples remain a significant limitation of DNN classifiers. The ability to identify previously unseen inputs as novel is crucial in safety-critical applications such as self-driving cars, unmanned aerial vehicles, and robots. Existing approaches to detect OOD samples treat a DNN as a black box and evaluate the confidence score of the output predictions. Unfortunately, this method frequently fails, because DNNs are not trained to reduce their confidence for OOD inputs. In this work, we introduce a novel method for OOD detection. Our method is motivated by theoretical analysis of neuron activation patterns (NAP) in ReLU-based architectures. The proposed method does not introduce a high computational overhead due to the binary representation of the activation patterns extracted from convolutional layers. The extensive empirical evaluation proves its high performance on various DNN architectures and seven image datasets.

        ----

        ## [322] Decoupling MaxLogit for Out-of-Distribution Detection

        **Authors**: *Zihan Zhang, Xiang Xiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00330](https://doi.org/10.1109/CVPR52729.2023.00330)

        **Abstract**:

        In machine learning, it is often observed that standard training outputs anomalously high confidence for both in-distribution (ID) and out-of-distribution (OOD) data. Thus, the ability to detect OOD samples is critical to the model deployment. An essential step for OOD detection is post-hoc scoring. MaxLogit is one of the simplest scoring functions which uses the maximum logits as OOD score. To provide a new viewpoint to study the logit-based scoring function, we reformulate the logit into cosine similarity and logit norm and propose to use MaxCosine and MaxNorm. We empirically find that MaxCosine is a core factor in the effectiveness of MaxLogit. And the performance of MaxLogit is encumbered by MaxNorm. To tackle the problem, we propose the Decoupling MaxLogit (DML) for flexibility to balance MaxCosine and MaxNorm. To further embody the core of our method, we extend DML to DML+ based on the new insights that fewer hard samples and compact feature space are the key components to make logit-based methods effective. We demonstrate the effectiveness of our logit-based OOD detection methods on CIFAR-10, CIFAR-100 and ImageNet and establish state-of-the-art performance.

        ----

        ## [323] Exploring Structured Semantic Prior for Multi Label Recognition with Incomplete Labels

        **Authors**: *Zixuan Ding, Ao Wang, Hui Chen, Qiang Zhang, Pengzhang Liu, Yongjun Bao, Weipeng Yan, Jungong Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00331](https://doi.org/10.1109/CVPR52729.2023.00331)

        **Abstract**:

        Multi-label recognition (MLR) with incomplete labels is very challenging. Recent works strive to explore the image-to-label correspondence in the vision-language model, i.e., CLIP [22], to compensate for insufficient annotations. In spite of promising performance, they generally overlook the valuable prior about the label-to-label correspondence. In this paper, we advocate remedying the deficiency of label supervision for the MLR with incomplete labels by deriving a structured semantic prior about the label-to-label corre-spondence via a semantic prior prompter. We then present a novel Semantic Correspondence Prompt Network (SCP-Net), which can thoroughly explore the structured semantic prior. A Prior-Enhanced Self-Supervised Learning method is further introduced to enhance the use of the prior. Comprehensive experiments and analyses on several widely used benchmark datasets show that our method significantly out-performs existing methods on all datasets, well demonstrating the effectiveness and the superiority of our method. Our code will be available at https://github.com/jameslahm/SCPNet.

        ----

        ## [324] Bridging the Gap Between Model Explanations in Partially Annotated Multi-Label Classification

        **Authors**: *Youngwook Kim, Jae-Myung Kim, Jieun Jeong, Cordelia Schmid, Zeynep Akata, Jungwoo Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00332](https://doi.org/10.1109/CVPR52729.2023.00332)

        **Abstract**:

        Due to the expensive costs of collecting labels in multi-label classification datasets, partially annotated multi-label classification has become an emerging field in computer vision. One baseline approach to this task is to assume unobserved labels as negative labels, but this assumption induces label noise as a form of false negative. To understand the negative impact caused by false negative labels, we study how these labels affect the model's explanation. We observe that the explanation of two models, trained with full and partial labels each, highlights similar regions but with different scaling, where the latter tends to have lower attribution scores. Based on these findings, we propose to boost the attribution scores of the model trained with partial labels to make its explanation resemble that of the model trained with full labels. Even with the conceptually simple approach, the multi-label classification performance improves by a large margin in three different datasets on a single positive label setting and one on a large-scale partial label setting. Code is available at https://github.com/youngwk/BridgeGapExplanationPAMC.

        ----

        ## [325] DivClust: Controlling Diversity in Deep Clustering

        **Authors**: *Ioannis Maniadis Metaxas, Georgios Tzimiropoulos, Ioannis Patras*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00333](https://doi.org/10.1109/CVPR52729.2023.00333)

        **Abstract**:

        Clustering has been a major research topic in the field of machine learning, one to which Deep Learning has recently been applied with significant success. However, an aspect of clustering that is not addressed by existing deep clustering methods, is that of efficiently producing multiple, diverse partitionings for a given dataset. This is particularly important, as a diverse set of base clusterings are necessary for consensus clustering, which has been found to produce better and more robust results than relying on a single clustering. To address this gap, we propose Div-Clust, a diversity controlling loss that can be incorporated into existing deep clustering frameworks to produce multiple clusterings with the desired degree of diversity. We conduct experiments with multiple datasets and deep clustering frameworks and show that: a) our method effectively controls diversity across frameworks and datasets with very small additional computational cost, b) the sets of clusterings learned by DivClust include solutions that significantly outperform single-clustering baselines, and c) using an off-the-shelf consensus clustering algorithm, DivClust produces consensus clustering solutions that consistently outperform single-clustering baselines, effectively improving the performance of the base deep clustering framework. Code is available at https://github.com/ManiadisG/DivClust.

        ----

        ## [326] Deep Semi-Supervised Metric Learning with Mixed Label Propagation

        **Authors**: *Furen Zhuang, Pierre Moulin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00334](https://doi.org/10.1109/CVPR52729.2023.00334)

        **Abstract**:

        Metric learning requires the identification of far-apart similar pairs and close dissimilar pairs during training, and this is difficult to achieve with unlabeled data because pairs are typically assumed to be similar if they are close. We present a novel metric learning method which circumvents this issue by identifying hard negative pairs as those which obtain dissimilar labels via label propagation (LP), when the edge linking the pair of data is removed in the affinity matrix. In so doing, the negative pairs can be identified despite their proximity, and we are able to utilize this information to significantly improve LP’ s ability to identify far-apart positive pairs and close negative pairs. This results in a considerable improvement in semi-supervised metric learning performance as evidenced by recall, precision and Normalized Mutual Information (NMI) performance metrics on Content-based Information Retrieval (CBIR) applications.

        ----

        ## [327] Leveraging Inter-Rater Agreement for Classification in the Presence of Noisy Labels

        **Authors**: *Maria Sofia Bucarelli, Lucas Cassano, Federico Siciliano, Amin Mantrach, Fabrizio Silvestri*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00335](https://doi.org/10.1109/CVPR52729.2023.00335)

        **Abstract**:

        In practical settings, classification datasets are obtained through a labelling process that is usually done by humans. Labels can be noisy as they are obtained by aggregating the different individual labels assigned to the same sample by multiple, and possibly disagreeing, annotators. The interrater agreement on these datasets can be measured while the underlying noise distribution to which the labels are subject is assumed to be unknown. In this work, we: (i) show how to leverage the inter-annotator statistics to estimate the noise distribution to which labels are subject; (ii) introduce methods that use the estimate of the noise distribution to learn from the noisy dataset; and (iii) establish generalization bounds in the empirical risk minimization framework that depend on the estimated quantities. We conclude the paper by providing experiments that illustrate our findings.

        ----

        ## [328] Modeling Inter-Class and Intra-Class Constraints in Novel Class Discovery

        **Authors**: *Wenbin Li, Zhichen Fan, Jing Huo, Yang Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00336](https://doi.org/10.1109/CVPR52729.2023.00336)

        **Abstract**:

        Novel class discovery (NCD) aims at learning a model that transfers the common knowledge from a class-disjoint labelled dataset to another unlabelled dataset and discovers new classes (clusters) within it. Many methods, as well as elaborate training pipelines and appropriate objectives, have been proposed and considerably boosted performance on NCD tasks. Despite all this, we find that the existing methods do not sufficiently take advantage of the essence of the NCD setting. To this end, in this paper, we propose to model both inter-class and intra-class constraints in NCD based on the symmetric Kullback-Leibler divergence (sKLD). Specifically, we propose an inter-class sKLD constraint to effectively exploit the disjoint relationship between labelled and unlabelled classes, enforcing the separability for different classes in the embedding space. In addition, we present an intra-class sKLD constraint to explicitly constrain the intra-relationship between a sample and its augmentations and ensure the stability of the training process at the same time. We conduct extensive experiments on the popular CIFAR10, CIFAR100 and ImageNet benchmarks and successfully demonstrate that our method can establish a new state of the art and can achieve significant performance improvements, e.g., 3.5%/3.7% clustering accuracy improvements on CIFAR100-50 dataset split under the task-aware/-agnostic evaluation protocol, over previous state-of-the-art methods. Code is available at https://github.com/FanZhichen/NCD-IIC.

        ----

        ## [329] Bootstrap Your Own Prior: Towards Distribution-Agnostic Novel Class Discovery

        **Authors**: *Muli Yang, Liancheng Wang, Cheng Deng, Hanwang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00337](https://doi.org/10.1109/CVPR52729.2023.00337)

        **Abstract**:

        Novel Class Discovery (NCD) aims to discover unknown classes without any annotation, by exploiting the transferable knowledge already learned from a base set of known classes. Existing works hold an impractical assumption that the novel class distribution prior is uniform, yet neglect the imbalanced nature of real-world data. In this paper, we relax this assumption by proposing a new challenging task: distribution-agnostic NCD, which allows data drawn from arbitrary unknown class distributions and thus renders existing methods useless or even harmful. We tackle this challenge by proposing a new method, dubbed “Boot-strapping Your Own Prior (BYOP)”, which iteratively estimates the class prior based on the model prediction it-self. At each iteration, we devise a dynamic temperature technique that better estimates the class prior by encouraging sharper predictions for less-confident samples. Thus, BYOP obtains more accurate pseudo-labels for the novel samples, which are beneficial for the next training iteration. Extensive experiments show that existing methods suffer from imbalanced class distributions, while BYOp
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/muliyangm/BYOP. out-performs them by clear margins, demonstrating its effectiveness across various distribution scenarios.

        ----

        ## [330] Towards Realistic Long-Tailed Semi-Supervised Learning: Consistency is All You Need

        **Authors**: *Tong Wei, Kai Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00338](https://doi.org/10.1109/CVPR52729.2023.00338)

        **Abstract**:

        While long-tailed semi-supervised learning (LTSSL) has received tremendous attention in many real-world classification problems, existing LTSSL algorithms typically assume that the class distributions of labeled and unlabeled data are almost identical. Those LTSSL algorithms built upon the assumption can severely suffer when the class distributions of labeled and unlabeled data are mismatched since they utilize biased pseudo-labels from the model. To alleviate this issue, we propose a new simple method that can effectively utilize unlabeled data of unknown class distributions by introducing the adaptive consistency regularizer (ACR). ACR realizes the dynamic refinery of pseudolabels for various distributions in a unified formula by estimating the true class distribution of unlabeled data. Despite its simplicity, we show that ACR achieves state-of-the-art performance on a variety of standard LTSSL benchmarks, e.g., an averaged 10% absolute increase of test accuracy against existing algorithms when the class distributions of labeled and unlabeled data are mismatched. Even when the class distributions are identical, ACR consistently outper-forms many sophisticated LTSSL algorithms. We carry out extensive ablation studies to tease apart the factors that are most important to ACR's success. Source code is available at https://github.com/Gank0078/ACR.

        ----

        ## [331] PromptCAL: Contrastive Affinity Learning via Auxiliary Prompts for Generalized Novel Category Discovery

        **Authors**: *Sheng Zhang, Salman H. Khan, Zhiqiang Shen, Muzammal Naseer, Guangyi Chen, Fahad Shahbaz Khan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00339](https://doi.org/10.1109/CVPR52729.2023.00339)

        **Abstract**:

        Although existing semi-supervised learning models achieve remarkable success in learning with unannotated in-distribution data, they mostly fail to learn on unlabeled data sampled from novel semantic classes due to their closed-set assumption. In this work, we target a pragmatic but under-explored Generalized Novel Category Discovery (GNCD) setting. The GNCD setting aims to categorize unlabeled training data coming from known and novel classes by leveraging the information of partially labeled known classes. We propose a two-stage Contrastive Affinity Learning method with auxiliary visual Prompts, dubbed PromptCAL, to address this challenging problem. Our approach discovers reliable pairwise sample affinities to learn better semantic clustering of both known and novel classes for the class token and visual prompts. First, we propose a discriminative prompt regularization loss to reinforce semantic discriminativeness of prompt-adapted pre-trained vision transformer for refined affinity relationships. Besides, we propose contrastive affinity learning to calibrate semantic representations based on our iterative semi-supervised affinity graph generation method for semantically-enhanced supervision. Extensive experimental evaluation demonstrates that our PromptCAL method is more effective in discovering novel classes even with limited annotations and surpasses the current state-of-the-art on generic and fine-grained benchmarks (e.g., with nearly 11% gain on CUB-200, and 9% on ImageNet-100) on overall accuracy. Our code is available at https://github.com/sheng-eatamath/PromptCAL.

        ----

        ## [332] Probabilistic Knowledge Distillation of Face Ensembles

        **Authors**: *Jianqing Xu, Shen Li, Ailin Deng, Miao Xiong, Jiaying Wu, Jiaxiang Wu, Shouhong Ding, Bryan Hooi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00340](https://doi.org/10.1109/CVPR52729.2023.00340)

        **Abstract**:

        Mean ensemble (i.e. averaging predictions from multiple models) is a commonly-used technique in machine learning that improves the performance of each individual model. We formalize it as feature alignment for ensemble in open-set face recognition and generalize it into Bayesian Ensemble Averaging (BEA) through the lens of probabilistic modeling. This generalization brings up two practical benefits that existing methods could not provide: (1) the uncertainty of a face image can be evaluated and further decomposed into aleatoric uncertainty and epistemic uncertainty, the latter of which can be used as a measure for out-of-distribution detection of faceness; (2) a BEA statistic provably reflects the aleatoric uncertainty of a face image, acting as a measure for face image quality to improve recognition performance. To inherit the uncertainty estimation capability from BEA without the loss of inference efficiency, we propose BEA-KD, a student model to distill knowledge from BEA. BEA-KD mimics the overall behavior of ensemble members and consistently outperforms SOTA knowledge distillation methods on various challenging benchmarks.

        ----

        ## [333] Class-Conditional Sharpness-Aware Minimization for Deep Long-Tailed Recognition

        **Authors**: *Zhipeng Zhou, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, Wei Gong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00341](https://doi.org/10.1109/CVPR52729.2023.00341)

        **Abstract**:

        It's widely acknowledged that deep learning models with flatter minima in its loss landscape tend to generalize better. However, such property is under-explored in deep long-tailed recognition (DLTR), a practical problem where the model is required to generalize equally well across all classes when trained on highly imbalanced label distribution. In this paper, through empirical observations, we argue that sharp minima are in fact prevalent in deep long-tailed models, whereas naï ve integration of existing flattening operations into long-tailed learning algorithms brings little improvement. Instead, we propose an effective two-stage sharpness-aware optimization approach based on the decoupling paradigm in DLTR. In the first stage, both the feature extractor and classifier are trained under parameter perturbations at a class-conditioned scale, which is theoretically motivated by the characteristic radius of flat minima under the PAC-Bayesian framework. In the second stage, we generate adversarial features with class-balanced sampling to further robustify the classifier with the backbone frozen. Extensive experiments on multiple long-tailed visual recognition benchmarks show that, our proposed Class-Conditional Sharpness-Aware Minimization (CC-SAM), achieves competitive performance compared to the state-of-the-arts. Code is available at https://github.com/zzpustc/CC-SAM.

        ----

        ## [334] Promoting Semantic Connectivity: Dual Nearest Neighbors Contrastive Learning for Unsupervised Domain Generalization

        **Authors**: *Yuchen Liu, Yaoming Wang, Yabo Chen, Wenrui Dai, Chenglin Li, Junni Zou, Hongkai Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00342](https://doi.org/10.1109/CVPR52729.2023.00342)

        **Abstract**:

        Domain Generalization (DG) has achieved great success in generalizing knowledge from source domains to unseen target domains. However, current DG methods rely heavily on labeled source data, which are usually costly and unavailable. Since unlabeled data are far more accessible, we study a more practical unsupervised domain generalization (UDG) problem. Learning invariant visual representation from different views, i.e., contrastive learning, promises well semantic features for in-domain unsupervised learning. However, it fails in cross-domain scenarios. In this paper, we first delve into the failure of vanilla contrastive learning and point out that semantic connectivity is the key to UDG. Specifically, suppressing the intra-domain connectiv-ity and encouraging the intra-class connectivity help to learn the domain-invariant semantic information. Then, we propose a novel unsupervised domain generalization approach, namely Dual Nearest Neighbors contrastive learning with strong Augmentation (DN
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
A). Our DN
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
A leverages strong augmentations to suppress the intra-domain connectivity and proposes a novel dual nearest neighbors search strategy to find trustworthy cross domain neighbors along with in-domain neighbors to encourage the intra-class connectivity. Experimental results demonstrate that our DN
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 A outperforms the state-of-the-art by a large margin, e.g., 12.01% and 13.11 % accuracy gain with only 1% labels for linear evaluation on PACS and DomainNet, respectively.

        ----

        ## [335] Instance Relation Graph Guided Source-Free Domain Adaptive Object Detection

        **Authors**: *Vibashan VS, Poojan Oza, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00343](https://doi.org/10.1109/CVPR52729.2023.00343)

        **Abstract**:

        Unsupervised Domain Adaptation (UDA) is an effective approach to tackle the issue of domain shift. Specifically, UDA methods try to align the source and target representations to improve generalization on the target domain. Further, UDA methods work under the assumption that the source data is accessible during the adaptation process. However, in real-world scenarios, the labelled source data is often restricted due to privacy regulations, data transmission constraints, or proprietary data concerns. The Source-Free Domain Adaptation (SFDA) setting aims to alleviate these concerns by adapting a source-trained model for the target domain without requiring access to the source data. In this paper, we explore the SFDA setting for the task of adaptive object detection. To this end, we propose a novel training strategy for adapting a source-trained object detector to the target domain without source data. More precisely, we design a novel contrastive loss to enhance the target representations by exploiting the objects relations for a given target domain input. These object instance relations are modelled using an Instance Relation Graph (IRG) network, which are then used to guide the contrastive representation learning. In addition, we utilize a student-teacher to effectively distill knowledge from source-trained model to target domain. Extensive experiments on multiple object detection benchmark datasets show that the proposed approach is able to efficiently adapt source-trained object detectors to the target domain, outperforming state-of-the-art domain adaptive detection methods. Code and models are provided in https://viudomain.github.io/irg-sfda-web/.

        ----

        ## [336] MOT: Masked Optimal Transport for Partial Domain Adaptation

        **Authors**: *You-Wei Luo, Chuan-Xian Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00344](https://doi.org/10.1109/CVPR52729.2023.00344)

        **Abstract**:

        As an important methodology to measure distribution discrepancy, optimal transport (OT) has been successfully applied to learn generalizable visual models under changing environments. However, there are still limitations, including strict prior assumption and implicit alignment, for current OT modeling in challenging real-world scenarios like partial domain adaptation, where the learned trans-port plan may be biased and negative transfer is inevitable. Thus, it is necessary to explore a more feasible OT methodology for real-world applications. In this work, we focus on the rigorous OT modeling for conditional distribution matching and label shift correction. A novel masked OT (MOT) methodology on conditional distributions is proposed by defining a mask operation with label information. Further, a relaxed and reweighting formulation is proposed to improve the robustness of OT in extreme scenarios. We prove the theoretical equivalence between conditional OT and MOT, which implies the well-defined MOT serves as a computation-friendly proxy. Extensive experiments validate the effectiveness of theoretical results and proposed model.

        ----

        ## [337] TOPLight: Lightweight Neural Networks with Task-Oriented Pretraining for Visible-Infrared Recognition

        **Authors**: *Hao Yu, Xu Cheng, Wei Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00345](https://doi.org/10.1109/CVPR52729.2023.00345)

        **Abstract**:

        Visible-infrared recognition (VI recognition) is a challenging task due to the enormous visual difference across heterogeneous images. Most existing works achieve promising results by transfer learning, such as pretraining on the ImageNet, based on advanced neural architectures like ResNet and ViT. However, such methods ignore the neg-ative influence of the pretrained colour prior knowledge, as well as their heavy computational burden makes them hard to deploy in actual scenarios with limited resources. In this paper, we propose a novel task-oriented pretrained lightweight neural network (TOPLight) for VI recognition. Specifically, the TOPLight method simulates the domain conflict and sample variations with the proposed fake do-main loss in the pretraining stage, which guides the network to learn how to handle those difficulties, such that a more general modality-shared feature representation is learned for the heterogeneous images. Moreover, an effective fine-grained dependency reconstruction module (FDR) is developed to discover substantial pattern dependencies shared in two modalities. Extensive experiments on VI person re-identification and VI face recognition datasets demonstrate the superiority of the proposed TOPLight, which signifi-cantly outperforms the current state of the arts while de-manding fewer computational resources.

        ----

        ## [338] OSAN: A One-Stage Alignment Network to Unify Multimodal Alignment and Unsupervised Domain Adaptation

        **Authors**: *Ye Liu, Lingfeng Qiao, Changchong Lu, Di Yin, Chen Lin, Haoyuan Peng, Bo Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00346](https://doi.org/10.1109/CVPR52729.2023.00346)

        **Abstract**:

        Extending from unimodal to multimodal is a critical challenge for unsupervised domain adaptation (UDA). Two major problems emerge in unsupervised multimodal domain adaptation: domain adaptation and modality alignment. An intuitive way to handle these two problems is to fulfill these tasks in two separate stages: aligning modalities followed by domain adaptation, or vice versa. However, domains and modalities are not associated in most existing two-stage studies, and the relationship between them is not leveraged which can provide complementary information to each other. In this paper, we unify these two stages into one to align domains and modalities simultaneously. In our model, a tensor-based alignment module (TAL) is presented to explore the relationship between domains and modalities. By this means, domains and modalities can interact sufficiently and guide them to utilize complementary information for better results. Furthermore, to establish a bridge between domains, a dynamic domain generator (DDG) module is proposed to build transitional samples by mixing the shared information of two domains in a self-supervised manner, which helps our model learn a domain-invariant common representation space. Extensive experiments prove that our method can achieve superior performance in two real-world applications. The code will be publicly available.

        ----

        ## [339] Patch-Mix Transformer for Unsupervised Domain Adaptation: A Game Perspective

        **Authors**: *Jinjing Zhu, Haotian Bai, Lin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00347](https://doi.org/10.1109/CVPR52729.2023.00347)

        **Abstract**:

        Endeavors have been recently made to leverage the vision transformer (ViT) for the challenging unsupervised domain adaptation (UDA) task. They typically adopt the cross-attention in ViT for direct domain alignment. However, as the performance of cross-attention highly relies on the quality of pseudo labels for targeted samples, it becomes less effective when the domain gap becomes large. We solve this problem from a game theory's perspective with the proposed model dubbed as PMTrans, which bridges source and target domains with an intermediate domain. Specifically, we propose a novel ViT-based module called PatchMix that effectively builds up the intermediate domain, i.e., probability distribution, by learning to sample patches from both domains based on the game-theoretical models. This way, it learns to mix the patches from the source and target domains to maximize the cross entropy (CE), while exploiting two semi-supervised mixup losses in the feature and label spaces to minimize it. As such, we interpret the process of UDA as a min-max CE game with three players, including the feature extractor, classifier, and PatchMix, to find the Nash Equilibria. Moreover, we leverage attention maps from ViT to re-weight the label of each patch by its importance, making it possible to obtain more domain-discriminative feature representations. We conduct extensive experiments on four benchmark datasets, and the results show that PMTrans significantly surpasses the ViT-based and CNN-based SoTA methods by +3.6% on Office-Home, +1.4% on Office-31, and +17.7% on DomainNet, respectively. https://vlis2022.github.io/cvpr23/PMTrans

        ----

        ## [340] ARO-Net: Learning Implicit Fields from Anchored Radial Observations

        **Authors**: *Yizhi Wang, Zeyu Huang, Ariel Shamir, Hui Huang, Hao Zhang, Ruizhen Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00348](https://doi.org/10.1109/CVPR52729.2023.00348)

        **Abstract**:

        We introduce anchored radial observations (ARO), a novel shape encoding for learning implicit field representation of 3D shapes that is category-agnostic and generalizable amid significant shape variations. The main idea behind our work is to reason about shapes through partial observations from a set of viewpoints, called anchors. We develop a general and unified shape representation by employing a fixed set of anchors, via Fibonacci sampling, and designing a coordinate-based deep neural network to predict the occupancy value of a query point in space. Differently from prior neural implicit models that use global shape feature, our shape encoder operates on contextual, query-specific features. To predict point occupancy, locally observed shape information from the perspective of the anchors surrounding the input query point are encoded and aggregated through an attention module, before implicit decoding is performed. We demonstrate the quality and generality of our network, coined ARO-Net, on surface reconstruction from sparse point clouds, with tests on novel and unseen object categories, “one-shape” training, and comparisons to state-of-the-art neural and classical methods for reconstruction and tessellation.

        ----

        ## [341] A Probabilistic Framework for Lifelong Test-Time Adaptation

        **Authors**: *Dhanajit Brahma, Piyush Rai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00349](https://doi.org/10.1109/CVPR52729.2023.00349)

        **Abstract**:

        Test-time adaptation (TTA) is the problem of updating a pre-trained source model at inference time given test input(s) from a different target domain. Most existing TTA approaches assume the setting in which the target domain is stationary, i.e., all the test inputs come from a single target domain. However, in many practical settings, the test input distribution might exhibit a lifelong/continual shift over time. Moreover, existing TTA approaches also lack the ability to provide reliable uncertainty estimates, which is crucial when distribution shifts occur between the source and target domain. To address these issues, we present PETAL (Probabilistic lifElong Test-time Adaptation with self-training prior), which solves lifelong TTA using a probabilistic approach, and naturally results in (1) a student-teacher framework, where the teacher model is an exponential moving average of the student model, and (2) regularizing the model updates at inference time using the source model as a regularizer. To prevent model drift in the lifelong/continual TTA setting, we also propose a data-driven parameter restoration technique which contributes to reducing the error accumulation and maintaining the knowledge of recent domains by restoring only the irrelevant parameters. In terms of predictive error rate as well as uncertainty based metrics such as Brier score and negative log-likelihood, our method achieves better results than the current state-of-the-art for online lifelong test-time adaptation across various benchmarks, such as CIFAR-10C, CIFAR-100C, ImageNetC, and ImageNet3DCC datasets. The source code for our approach is accessible at https://github.com/dhanajitb/petal.

        ----

        ## [342] Distribution Shift Inversion for Out-of-Distribution Prediction

        **Authors**: *Runpeng Yu, Songhua Liu, Xingyi Yang, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00350](https://doi.org/10.1109/CVPR52729.2023.00350)

        **Abstract**:

        Machine learning society has witnessed the emergence of a myriad of Out-of-Distribution (OoD) algorithms, which address the distribution shift between the training and the testing distribution by searching for a unified predictor or invariant feature representation. However, the task of directly mitigating the distribution shift in the unseen testing set is rarely investigated, due to the unavailability of the testing distribution during the training phase and thus the impossibility of training a distribution translator mapping between the training and testing distribution. In this paper, we explore how to bypass the requirement of testing distribution for distribution translator training and make the distribution translation useful for OoD prediction. We propose a portable Distribution Shift Inversion (DSI) algorithm, in which, before being fed into the prediction model, the OoD testing samples are first linearly combined with additional Gaussian noise and then transferred back towards the training distribution using a diffusion model trained only on the source distribution. Theoretical analysis reveals the feasibility of our method. Experimental results, on both multiple-domain generalization datasets and single-domain generalization datasets, show that our method provides a general performance gain when plugged into a wide range of commonly used OoD algorithms. Our code is available at https://github.com/yu-rp/Distribution-Shift-Iverson.

        ----

        ## [343] Learning Joint Latent Space EBM Prior Model for Multi-layer Generator

        **Authors**: *Jiali Cui, Ying Nian Wu, Tian Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00351](https://doi.org/10.1109/CVPR52729.2023.00351)

        **Abstract**:

        This paper studies the fundamental problem of learning multi-layer generator models. The multi-layer generator model builds multiple layers of latent variables as a prior model on top of the generator, which benefits learning complex data distribution and hierarchical representations. However, such a prior model usually focuses on modeling inter-layer relations between latent variables by assuming non-informative (conditional) Gaussian distributions, which can be limited in model expressivity. To tackle this issue and learn more expressive prior models, we propose an energy-based model (EBM) on the joint latent space over all layers of latent variables with the multi-layer generator as its backbone. Such joint latent space EBM prior model captures the intra-layer contextual relations at each layer through layer-wise energy terms, and latent variables across different layers are jointly corrected. We develop a joint training scheme via maximum likelihood estimation (MLE), which involves Markov Chain Monte Carlo (MCMC) sampling for both prior and posterior distributions of the latent variables from different layers. To ensure efficient inference and learning, we further propose a variational training scheme where an inference model is used to amortize the costly posterior MCMC sampling. Our experiments demonstrate that the learned model can be expressive in generating high-quality images and capturing hierarchical features for better outlier detection.

        ----

        ## [344] A Data-Based Perspective on Transfer Learning

        **Authors**: *Saachi Jain, Hadi Salman, Alaa Khaddaj, Eric Wong, Sung Min Park, Aleksander Madry*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00352](https://doi.org/10.1109/CVPR52729.2023.00352)

        **Abstract**:

        It is commonly believed that in transfer learning including more pre-training data translates into better performance. However, recent evidence suggests that removing data from the source dataset can actually help too. In this work, we take a closer look at the role of the source dataset's composition in transfer learning and present a framework for probing its impact on downstream performance. Our framework gives rise to new capabilities such as pinpointing transfer learning brittleness as well as detecting pathologies such as data-leakage and the presence of misleading examples in the source dataset. In particular, we demonstrate that removing detrimental datapoints identified by our framework indeed improves transfer learning performance from ImageNet on a variety of target tasks. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/MadryLab/data-transfer

        ----

        ## [345] A Meta-Learning Approach to Predicting Performance and Data Requirements

        **Authors**: *Achin Jain, Gurumurthy Swaminathan, Paolo Favaro, Hao Yang, Avinash Ravichandran, Hrayr Harutyunyan, Alessandro Achille, Onkar Dabeer, Bernt Schiele, Ashwin Swaminathan, Stefano Soatto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00353](https://doi.org/10.1109/CVPR52729.2023.00353)

        **Abstract**:

        We propose an approach to estimate the number of samples required for a model to reach a target performance. We find that the power law, the de facto principle to estimate model performance, leads to a large error when using a small dataset (e.g., 5 samples per class) for extrapolation. This is because the log-performance error against the log-dataset size follows a nonlinear progression in the few-shot regime followed by a linear progression in the high-shot regime. We introduce a novel piecewise power law (PPL) that handles the two data regimes differently. To estimate the parameters of the PPL, we introduce a random forest regressor trained via meta learning that generalizes across classification/detection tasks, ResNet/ViT based architectures, and random/pre-trained initializations. The PPL improves the performance estimation on average by 37% across 16 classification and 33% across 10 detection datasets, compared to the power law. We further extend the PPL to provide a confidence bound and use it to limit the prediction horizon that reduces over-estimation of data by 76% on classification and 91% on detection datasets.

        ----

        ## [346] Guided Recommendation for Model Fine-Tuning

        **Authors**: *Hao Li, Charless C. Fowlkes, Hao Yang, Onkar Dabeer, Zhuowen Tu, Stefano Soatto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00354](https://doi.org/10.1109/CVPR52729.2023.00354)

        **Abstract**:

        Model selection is essential for reducing the search cost of the best pre-trained model over a large-scale model zoo for a downstream task. After analyzing recent hand-designed model selection criteria with 400+ ImageNet pre-trained models and 40 downstream tasks, we find that they can fail due to invalid assumptions and intrinsic limitations. The prior knowledge on model capacity and dataset also can not be easily integrated into the existing criteria. To address these issues, we propose to convert model selection as a recommendation problem and to learn from the past training history. Specifically, we characterize the meta information of datasets and models as features, and use their transfer learning performance as the guided score. With thousands of historical training jobs, a recommendation system can be learned to predict the model selection score given the features of the dataset and the model as input. Our approach enables integrating existing model selection scores as additional features and scales with more historical data. We evaluate the prediction accuracy with 22 pre-trained models over 40 downstream tasks. With extensive evaluations, we show that the learned approach can outperform prior hand-designed model selection methods significantly when relevant training history is available.

        ----

        ## [347] EMT-NAS: Transferring architectural knowledge between tasks from different datasets

        **Authors**: *Peng Liao, Yaochu Jin, Wenli Du*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00355](https://doi.org/10.1109/CVPR52729.2023.00355)

        **Abstract**:

        The success of multitask learning (MTL) can largely be attributed to the shared representation of related tasks, allowing the models to better generalise. In deep learning, this is usually achieved by sharing a common neural network architecture and jointly training the weights. However, the joint training of weighting parameters on multiple related tasks may lead to performance degradation, known as negative transfer. To address this issue, this work proposes an evolutionary multitasking neural architecture search (EMT-NAS) algorithm to accelerate the search process by transferring architectural knowledge across multiple related tasks. In EMT-NAS, unlike the traditional MTL, the model for each task has a personalised network architecture and its own weights, thus offering the capability of effectively alleviating negative transfer. A fitness re-evaluation method is suggested to alleviate fluctuations in performance evaluations resulting from parameter sharing and the mini-batch gradient descent training method, thereby avoiding losing promising solutions during the search process. To rigorously verify the performance of EMT-NAS, the classification tasks used in the empirical assessments are derived from different datasets, including the CIFAR-10 and CIFAR-100, and four MedMNIST datasets. Extensive comparative experiments on different numbers of tasks demonstrate that EMT-NAS takes 8% and up to 40% on CIFAR and MedMNIST, respectively, less time to find competitive neural architectures than its single-task counterparts.

        ----

        ## [348] AttriCLIP: A Non-Incremental Learner for Incremental Knowledge Learning

        **Authors**: *Runqi Wang, Xiaoyue Duan, Guoliang Kang, Jianzhuang Liu, Shaohui Lin, Songcen Xu, Jinhu Lv, Baochang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00356](https://doi.org/10.1109/CVPR52729.2023.00356)

        **Abstract**:

        Continual learning aims to enable a model to incrementally learn knowledge from sequentially arrived data. Previous works adopt the conventional classification architecture, which consists of a feature extractor and a classifier. The feature extractor is shared across sequentially arrived tasks or classes, but one specific group of weights of the classifier corresponding to one new class should be incrementally expanded. Consequently, the parameters of a continual learner gradually increase. Moreover, as the classifier contains all historical arrived classes, a certain size of the memory is usually required to store rehearsal data to mitigate classifier bias and catastrophic forgetting. In this paper, we propose a non-incremental learner, named AttriCLIP, to incrementally extract knowledge of new classes or tasks. Specifically, AttriCLIP is built upon the pre-trained visual-language model CLIP. Its image encoder and text encoder are fixed to extract features from both images and text. Text consists of a category name and a fixed number of learnable parameters which are selected from our designed attribute word bank and serve as attributes. As we compute the visual and textual similarity for classification, AttriCLIP is a non-incremental learner. The attribute prompts, which encode the common knowledge useful for classification, can effectively mitigate the catastrophic forgetting and avoid constructing a replay memory. We evaluate our AttriCLIP and compare it with CLIP-based and previous state-of-the-art continual learning methods in realistic settings with domain-shift and long-sequence learning. The results show that our method performs favorably against previous state-of-the-arts. The implementation code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/AttriCLIP.

        ----

        ## [349] Batch Model Consolidation: A Multi-Task Model Consolidation Framework

        **Authors**: *Iordanis Fostiropoulos, Jiaye Zhu, Laurent Itti*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00357](https://doi.org/10.1109/CVPR52729.2023.00357)

        **Abstract**:

        In Continual Learning (CL), a model is required to learn a stream of tasks sequentially without significant performance degradation on previously learned tasks. Current approaches fail for a long sequence of tasks from diverse domains and difficulties. Many of the existing CL approaches are difficult to apply in practice due to excessive memory cost or training time, or are tightly coupled to a single device. With the intuition derived from the widely applied mini-batch training, we propose Batch Model Consolidation (BMC) to support more realistic CL under conditions where multiple agents are exposed to a range of tasks. During a regularization phase, BMC trains multiple expert models in parallel on a set of disjoint tasks. Each expert maintains weight similarity to a base model through a stability loss, and constructs a buffer from a fraction of the task's data. During the consolidation phase, we combine the learned knowledge on 'batches' of expert models using a batched consolidation loss in memory data that aggregates all buffers. We thoroughly evaluate each component of our method in an ablation study and demonstrate the effectiveness on standardized benchmark datasets Split-CIFAR-100, Tiny-ImageNet, and the Stream dataset composed of 71 image classification tasks from diverse domains and difficulties. Our method outperforms the next best CL approach by 70% and is the only approach that can maintain performance at the end of 71 tasks.

        ----

        ## [350] SmartAssign: Learning A Smart Knowledge Assignment Strategy for Deraining and Desnowing

        **Authors**: *Yinglong Wang, Chao Ma, Jianzhuang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00358](https://doi.org/10.1109/CVPR52729.2023.00358)

        **Abstract**:

        Existing methods mainly handle single weather types. However, the connections of different weather conditions at deep representation level are usually ignored. These connections, if used properly, can generate complementary representations for each other to make up insufficient training data, obtaining positive performance gains and better generalization. In this paper, we focus on the very correlated rain and snow to explore their connections at deep representation level. Because sub-optimal connections may cause negative effect, another issue is that if rain and snow are handled in a multi-task learning way, how to find an optimal connection strategy to simultaneously improve deraining and desnowing performance. To build desired connection, we propose a smart knowledge assignment strategy, called SmartAssign, to optimally assign the knowledge learned from both tasks to a specific one. In order to further enhance the accuracy of knowledge assignment, we propose a novel knowledge contrast mechanism, so that the knowledge assigned to different tasks preserves better uniqueness. The inherited inductive biases usually limit the modelling ability of CNNs, we introduce a novel transformer block to constitute the backbone of our network to effectively combine long-range context dependency and local image details. Extensive experiments on seven benchmark datasets verify that proposed SmartAssign explores effective connection between rain and snow, and improves the performances of both deraining and desnowing apparently. The implementation code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/SmartAssign.

        ----

        ## [351] TinyMIM: An Empirical Study of Distilling MIM Pre-trained Models

        **Authors**: *Sucheng Ren, Fangyun Wei, Zheng Zhang, Han Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00359](https://doi.org/10.1109/CVPR52729.2023.00359)

        **Abstract**:

        Masked image modeling (MIM) performs strongly in pretraining large vision Transformers (ViTs). However, small models that are critical for real-world applications can-not or only marginally benefit from this pre-training approach. In this paper, we explore distillation techniques to transfer the success of large MIM-based pre-trained models to smaller ones. We systematically study different options in the distillation framework, including distilling targets, losses, input, network regularization, sequential distillation, etc, revealing that: 1) Distilling token relations is more effective than CLS token- and feature-based distillation; 2) An intermediate layer of the teacher network as target perform better than that using the last layer when the depth of the student mismatches that of the teacher; 3) Weak regularization is preferred; etc. With these findings, we achieve significant fine-tuning accuracy improvements over the scratch MIM pre-training on ImageNet-1K classification, using all the ViT-Tiny, ViT-Small, and ViT-base models, with +4.2%/+2.4%/+1.4% gains, respectively. Our TinyMIM model of base size achieves 52.2 mIoU in AE20K semantic segmentation, which is +4.1 higher than the MAE baseline. Our TinyMIM model of tiny size achieves 79.6% top-1 accuracy on ImageNet-1K image classification, which sets a new record for small vision models of the same size and computation budget. This strong performance suggests an alternative way for developing small vision Transformer models, that is, by exploring better training methods rather than introducing inductive biases into architectures as in most previous works. Code is available at https://github.com/OliverRensu/TinyMIM.

        ----

        ## [352] Computationally Budgeted Continual Learning: What Does Matter?

        **Authors**: *Ameya Prabhu, Hasan Abed Al Kader Hammoud, Puneet K. Dokania, Philip H. S. Torr, Ser-Nam Lim, Bernard Ghanem, Adel Bibi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00360](https://doi.org/10.1109/CVPR52729.2023.00360)

        **Abstract**:

        Continual Learning (CL) aims to sequentially train models on streams of incoming data that vary in distribution by preserving previous knowledge while adapting to new data. Current CL literature focuses on restricted access to previously seen data, while imposing no constraints on the computational budget for training. This is unreasonable for applications in-the-wild, where systems are primarily constrained by computational and time budgets, not storage. We revisit this problem with a large-scale benchmark and analyze the performance of traditional CL approaches in a compute-constrained setting, where effective memory samples used in training can be implicitly restricted as a consequence of limited computation. We conduct experiments evaluating various CL sampling strategies, distillation losses, and partial fine-tuning on two large-scale datasets, namely ImageNet2K and Continual Google Landmarks V2 in data incremental, class incremental, and time incremental settings. Through extensive experiments amounting to a total of over 1500 GPU-hours, we find that, under compute-constrained setting, traditional CL approaches, with no exception, fail to outperform a simple minimal baseline that samples uniformly from memory. Our conclusions are consistent in a different number of stream time steps, e.g., 20 to 200, and under several computational budgets. This suggests that most existing CL methods are particularly too computationally expensive for realistic budgeted deployment. Code for this project is available at: https://github.com/drimpossible/BudgetCL.

        ----

        ## [353] GradMA: A Gradient-Memory-based Accelerated Federated Learning with Alleviated Catastrophic Forgetting

        **Authors**: *Kangyang Luo, Xiang Li, Yunshi Lan, Ming Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00361](https://doi.org/10.1109/CVPR52729.2023.00361)

        **Abstract**:

        Federated Learning (FL) has emerged as a de facto machine learning area and received rapid increasing research interests from the community. However, catastrophic forgetting caused by data heterogeneity and partial participation poses distinctive challenges for FL, which are detrimental to the performance. To tackle the problems, we propose a new FL approach (namely GradMA), which takes inspiration from continual learning to simultaneously correct the server-side and worker-side update directions as well as take full advantage of server's rich computing and memory resources. Furthermore, we elaborate a memory reduction strategy to enable GradMA to accommodate FL with a large scale of workers. We then analyze convergence of GradMA theoretically under the smooth non-convex setting and show that its convergence rate achieves a linear speed up w.r.t the increasing number of sampled active workers. At last, our extensive experiments on various image classification tasks show that GradMA achieves significant performance gains in accuracy and communication efficiency compared to SOTA baselines. We provide our code here: https://github.com/lkyddd/GradMA.

        ----

        ## [354] Rethinking Gradient Projection Continual Learning: Stability/Plasticity Feature Space Decoupling

        **Authors**: *Zhen Zhao, Zhizhong Zhang, Xin Tan, Jun Liu, Yanyun Qu, Yuan Xie, Lizhuang Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00362](https://doi.org/10.1109/CVPR52729.2023.00362)

        **Abstract**:

        Continual learning aims to incrementally learn novel classes over time, while not forgetting the learned knowledge. Recent studies have found that learning would not forget if the updated gradient is orthogonal to the feature space. However, previous approaches require the gradient to be fully orthogonal to the whole feature space, leading to poor plasticity, as the feasible gradient direction becomes narrow when the tasks continually come, i.e., feature space is unlimitedly expanded. In this paper, we propose a space decoupling (SD) algorithm to decouple the feature space into a pair of complementary subspaces, i.e., the stability space 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{I}$</tex>
 and the plasticity space 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{R}. \mathcal{I}$</tex>
 is established by conducting space intersection between the historic and current feature space, and thus 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{I}$</tex>
 contains more task-shared bases. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{R}$</tex>
 is constructed by seeking the orthogonal complementary subspace of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$T$</tex>
 and thus 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{R}$</tex>
 mainly contains task-specific bases. By putting distinguishing constraints on 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{R}$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{I}$</tex>
, our method achieves a better balance between stability and plasticity. Extensive experiments are conducted by applying SD to gradient projection baselines, and show SD is model-agnostic and achieves SOTA results on publicly available datasets.

        ----

        ## [355] Neuro-Modulated Hebbian Learning for Fully Test-Time Adaptation

        **Authors**: *Yushun Tang, Ce Zhang, Heng Xu, Shuoshuo Chen, Jie Cheng, Luziwei Leng, Qinghai Guo, Zhihai He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00363](https://doi.org/10.1109/CVPR52729.2023.00363)

        **Abstract**:

        Fully test-time adaptation aims to adapt the network model based on sequential analysis of input samples during the inference stage to address the cross-domain performance degradation problem of deep neural networks. We take inspiration from the biological plausibility learning where the neuron responses are tuned based on a local synapse-change procedure and activated by competitive lateral inhibition rules. Based on these feed-forward learning rules, we design a soft Hebbian learning process which provides an unsupervised and effective mechanism for online adaptation. We observe that the performance of this feed-forward Hebbian learning for fully test-time adaptation can be significantly improved by incorporating a feedback neuromodulation layer. It is able to fine-tune the neuron responses based on the external feedback generated by the error backpropagation from the top inference layers. This leads to our proposed neuro-modulated Hebbian learning (NHL) method for fully test-time adaptation. With the unsupervised feed-forward soft Hebbian learning being combined with a learned neuromodulator to capture feedback from external responses, the source model can be effectively adapted during the testing process. Experimental results on benchmark datasets demonstrate that our proposed method can significantly improve the adaptation performance of network models and outperforms existing state-of-the-art methods.

        ----

        ## [356] Generalizing Dataset Distillation via Deep Generative Prior

        **Authors**: *George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, Jun-Yan Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00364](https://doi.org/10.1109/CVPR52729.2023.00364)

        **Abstract**:

        Dataset Distillation aims to distill an entire dataset's knowledge into a few synthetic images. The idea is to synthe-size a small number of synthetic data points that, when given to a learning algorithm as training data, result in a model approximating one trained on the original data. Despite a recent upsurge of progress in the field, existing dataset dis-tillation methods fail to generalize to new architectures and scale to high-resolution datasets. To overcome the above issues, we propose to use the learned prior from pretrained deep generative models to synthesize the distilled data. To achieve this, we present a new optimization algorithm that distills a large number of images into a few intermediate feature vectors in the generative model's latent space. Our method augments existing techniques, significantly improving cross-architecture generalization in all settings.

        ----

        ## [357] Minimizing the Accumulated Trajectory Error to Improve Dataset Distillation

        **Authors**: *Jiawei Du, Yidi Jiang, Vincent Y. F. Tan, Joey Tianyi Zhou, Haizhou Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00365](https://doi.org/10.1109/CVPR52729.2023.00365)

        **Abstract**:

        Model-based deep learning has achieved astounding successes due in part to the availability of large-scale real-world data. However, processing such massive amounts of data comes at a considerable cost in terms of computations, storage, training and the search for good neural architectures. Dataset distillation has thus recently come to the fore. This paradigm involves distilling information from large real-world datasets into tiny and compact synthetic datasets such that processing the latter ideally yields similar performances as the former. State-of-the-art methods primarily rely on learning the synthetic dataset by matching the gradients obtained during training between the real and synthetic data. However, these gradient-matching methods suffer from the so-called accumulated trajectory error caused by the discrepancy between the distillation and subsequent evaluation. To mitigate the adverse impact of this accumulated trajectory error, we propose a novel approach that encourages the optimization algorithm to seek a flat trajectory. We show that the weights trained on synthetic data are robust against the accumulated errors perturbations with the regularization towards the flat trajectory. Our method, called Flat Trajectory Distillation (FTD), is shown to boost the performance of gradient-matching methods by up to 4.7% on a subset of images of the ImageNet dataset with higher resolution images. We also validate the effectiveness and generalizability of our method with datasets of different resolutions and demonstrate its applicability to neural architecture search. Code is available at. https://github.com/AngusDujw/FTD-distillation.

        ----

        ## [358] Slimmable Dataset Condensation

        **Authors**: *Songhua Liu, Jingwen Ye, Runpeng Yu, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00366](https://doi.org/10.1109/CVPR52729.2023.00366)

        **Abstract**:

        Dataset distillation, also known as dataset condensation, aims to compress a large dataset into a compact synthetic one. Existing methods perform dataset condensation by assuming a fixed storage or transmission budget. When the budget changes, however, they have to repeat the synthesizing process with access to original datasets, which is highly cumbersome if not infeasible at all. In this paper, we explore the problem of slimmable dataset condensation, to extract a smaller synthetic dataset given only previous condensation results. We first study the limitations of existing dataset condensation algorithms on such a successive compression setting and identify two key factors: (1) the inconsistency of neural networks over different compression times and (2) the underdetermined solution space for synthetic data. Accordingly, we propose a novel training objective for slimmable dataset condensation to explicitly account for both factors. Moreover, synthetic datasets in our method adopt a significance-aware parameterization. Theoretical derivation indicates that an upper-bounded error can be achieved by discarding the minor components without training. Alternatively, if training is allowed, this strategy can serve as a strong initialization that enables a fast convergence. Extensive comparisons and ablations demonstrate the superiority of the proposed solution over existing methods on multiple benchmarks.

        ----

        ## [359] Sharpness-Aware Gradient Matching for Domain Generalization

        **Authors**: *Pengfei Wang, Zhaoxiang Zhang, Zhen Lei, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00367](https://doi.org/10.1109/CVPR52729.2023.00367)

        **Abstract**:

        The goal of domain generalization (DG) is to enhance the generalization capability of the model learned from a source domain to other unseen domains. The recently developed Sharpness-Aware Minimization (SAM) method aims to achieve this goal by minimizing the sharpness measure of the loss landscape. Though SAM and its variants have demonstrated impressive DG performance, they may not always converge to the desired flat region with a small loss value. In this paper, we present two conditions to ensure that the model could converge to a flat minimum with a small loss, and present an algorithm, named Sharpness-Aware Gradient Matching (SAGM), to meet the two conditions for improving model generalization capability. Specifically, the optimization objective of SAGM will simultaneously minimize the empirical risk, the perturbed loss (i.e., the maximum loss within a neighborhood in the parameter space), and the gap between them. By implicitly aligning the gradient directions between the empirical risk and the perturbed loss, SAGM improves the generalization capability over SAM and its variants without increasing the computational cost. Extensive experimental results show that our proposed SAGM method consistently outperforms the state-of-the-art methods on five DG benchmarks, including PACS, VLCS, OfficeHome, TerraIncognita, and DomainNet. Codes are available at https://github.com/Wang-pengfei/SAGM.

        ----

        ## [360] Dynamic Neural Network for Multi-Task Learning Searching across Diverse Network Topologies

        **Authors**: *Wonhyeok Choi, Sunghoon Im*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00368](https://doi.org/10.1109/CVPR52729.2023.00368)

        **Abstract**:

        In this paper, we present a new MTL framework that searches for structures optimized for multiple tasks with diverse graph topologies and shares features among tasks. We design a restricted DAG-based central network with read-in/read-out layers to build topologically diverse task-adaptive structures while limiting search space and time. We search for a single optimized network that serves as multiple task adaptive sub-networks using our three-stage training process. To make the network compact and discretized, we propose a flow-based reduction algorithm and a squeeze loss used in the training process. We evaluate our optimized network on various public MTL datasets and show ours achieves state-of-the-art performance. An extensive ablation study experimentally validates the effectiveness of the sub-module and schemes in our framework.

        ----

        ## [361] SplineCam: Exact Visualization and Characterization of Deep Network Geometry and Decision Boundaries

        **Authors**: *Ahmed Imtiaz Humayun, Randall Balestriero, Guha Balakrishnan, Richard G. Baraniuk*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00369](https://doi.org/10.1109/CVPR52729.2023.00369)

        **Abstract**:

        Current Deep Network (DN) visualization and inter-pretability methods rely heavily on data space visualizations such as scoring which dimensions of the data are responsible for their associated prediction or generating new data features or samples that best match a given DN unit or representation. In this paper, we go one step further by developing the first provably exact method for computing the geometry of a DN's mapping - including its decision boundary - over a specified region of the data space. By lever-aging the theory of Continuous Piece- Wise Linear (CPWL) spline DNs, SplineCam exactly computes a DN's geometry without resorting to approximations such as sampling or architecture simplification. SplineCam applies to any DN architecture based on CPWL activation nonlinearities, including (leaky) ReLU, absolute value, maxout, and max-pooling and can also be applied to regression DNs such as implicit neural representations. Beyond decision boundary visualization and characterization, SplineCam enables one to compare architectures, measure generalizability, and sample from the decision boundary on or off the data manifold. Project website: bit.ly/splinecam.

        ----

        ## [362] VNE: An Effective Method for Improving Deep Representation by Manipulating Eigenvalue Distribution

        **Authors**: *Jaeill Kim, Suhyun Kang, Duhun Hwang, Jungwook Shin, Wonjong Rhee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00370](https://doi.org/10.1109/CVPR52729.2023.00370)

        **Abstract**:

        Since the introduction of deep learning, a wide scope of representation properties, such as decorrelation, whitening, disentanglement, rank, isotropy, and mutual information, have been studied to improve the quality of representation. However, manipulating such properties can be challenging in terms of implementational effectiveness and general applicability. To address these limitations, we propose to regularize von Neumann entropy (VNE) of representation. First, we demonstrate that the mathematical formulation of VNE is superior in effectively manipulating the eigenvalues of the representation autocorrelation matrix. Then, we demonstrate that it is widely applicable in improving state-of-the-art algorithms or popular benchmark algorithms by investigating domain-generalization, meta-learning, self-supervised learning, and generative models. In addition, we formally establish theoretical connections with rank, disentanglement, and isotropy of representation. Finally, we provide discussions on the dimension control of VNE and the relationship with Shannon entropy. Code is available at: https://github.com/jaeill/CVPR23-VNE.

        ----

        ## [363] Efficient On-Device Training via Gradient Filtering

        **Authors**: *Yuedong Yang, Guihong Li, Radu Marculescu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00371](https://doi.org/10.1109/CVPR52729.2023.00371)

        **Abstract**:

        Despite its importance for federated learning, continuous learning and many other applications, on-device training remains an open problem for EdgeAI. The problem stems from the large number of operations (e.g., floating point multiplications and additions) and memory consumption required during training by the back-propagation algorithm. Consequently, in this paper, we propose a new gradient filtering approach which enables on-device CNN model training. More precisely, our approach creates a special structure with fewer unique elements in the gradient map, thus significantly reducing the computational complexity and memory consumption of back propagation during training. Extensive experiments on image classification and semantic segmentation with multiple CNN models (e.g., MobileNet, DeepLabV3, UPerNet) and devices (e.g., Raspberry Pi and Jetson Nano) demonstrate the effectiveness and wide applicability of our approach. For example, compared to SOTA, we achieve up to 19× speedup and 77.1% memory savings on ImageNet classification with only 0.1% accuracy loss. Finally, our method is easy to implement and deploy; over 20× speedup and 90% energy savings have been observed compared to highly optimized baselines in MKLDNN and CUDNN on NVIDIA Jetson Nano. Consequently, our approach opens up a new direction of research with a huge potential for on-device training.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/SLDGroup/GradientFilter-CVPR23

        ----

        ## [364] Are Data-Driven Explanations Robust Against Out-of-Distribution Data?

        **Authors**: *Tang Li, Fengchun Qiao, Mengmeng Ma, Xi Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00372](https://doi.org/10.1109/CVPR52729.2023.00372)

        **Abstract**:

        As black-box models increasingly power high-stakes applications, a variety of data-driven explanation methods have been introduced. Meanwhile, machine learning models are constantly challenged by distributional shifts. A question naturally arises: Are data-driven explanations robust against out-of-distribution data? Our empirical results show that even though predict correctly, the model might still yield unreliable explanations under distributional shifts. How to develop robust explanations against out-of-distribution data? To address this problem, we propose an end-to-end model-agnostic learning framework Distributionally Robust Explanations (DRE). The key idea is, inspired by self-supervised learning, to fully utilizes the inter-distribution information to provide supervisory signals for the learning of explanations without human annotation. Can robust explanations benefit the model's generalization capability? We conduct extensive experiments on a wide range of tasks and data types, including classification and regression on image and scientific tabular data. Our results demonstrate that the proposed method significantly improves the model's performance in terms of explanation and prediction robustness against distributional shifts.

        ----

        ## [365] BiasAdv: Bias-Adversarial Augmentation for Model Debiasing

        **Authors**: *Jongin Lim, Youngdong Kim, Byungjai Kim, Chanho Ahn, Jinwoo Shin, Eunho Yang, Seungju Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00373](https://doi.org/10.1109/CVPR52729.2023.00373)

        **Abstract**:

        Neural networks are often prone to bias toward spurious correlations inherent in a dataset, thus failing to generalize unbiased test criteria. A key challenge to resolving the issue is the significant lack of bias-conflicting training data (i. e., samples without spurious correlations). In this paper, we propose a novel data augmentation approach termed Bias-Adversarial augmentation (BiasAdv) that supplements bias-conflicting samples with adversarial images. Our key idea is that an adversarial attack on a biased model that makes decisions based on spurious correlations may generate syn-thetic bias-conflicting samples, which can then be used as augmented training data for learning a debiased model. Specifically, we formulate an optimization problem for gen-erating adversarial images that attack the predictions of an auxiliary biased model without ruining the predictions of the desired debiased model. Despite its simplicity, we find that BiasAdv can generate surprisingly useful synthetic bias-conflicting samples, allowing the debiased model to learn generalizable representations. Furthermore, BiasAdv does not require any bias annotations or prior knowledge of the bias type, which enables its broad applicability to existing debiasing methods to improve their performances. Our extensive experimental results demonstrate the superiority of BiasAdv, achieving state-of-the-art performance on four popular benchmark datasets across various bias domains.

        ----

        ## [366] Q-DETR: An Efficient Low-Bit Quantized Detection Transformer

        **Authors**: *Sheng Xu, Yanjing Li, Mingbao Lin, Peng Gao, Guodong Guo, Jinhu Lü, Baochang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00374](https://doi.org/10.1109/CVPR52729.2023.00374)

        **Abstract**:

        The recent detection transformer (DETR) has advanced object detection, but its application on resource-constrained devices requires massive computation and memory resources. Quantization stands out as a solution by representing the network in low-bit parameters and operations. However, there is a significant performance drop when performing low-bit quantized DETR (Q-DETR) with existing quantization methods. We find that the bottle-necks of Q-DETR come from the query information distortion through our empirical analyses. This paper addresses this problem based on a distribution rectification distillation (DRD). We formulate our DRD as a bi-level optimization problem, which can be derived by generalizing the information bottleneck (IB) principle to the learning of Q-DETR. At the inner level, we conduct a distribution alignment for the queries to maximize the self-information entropy. At the upper level, we introduce a new foreground-aware query matching scheme to effectively transfer the teacher information to distillation-desired features to minimize the conditional information entropy. Extensive experimental results show that our method performs much better than prior arts. For example, the 4-bit Q-DETR can theoretically accelerate DETR with ResNet-50 backbone by 6.6× and achieve 39.4% AP, with only 2.6% performance gaps than its real-valued counterpart on the COCO dataset
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/SteveTsui/Q-DETR.

        ----

        ## [367] NIPQ: Noise proxy-based Integrated Pseudo-Quantization

        **Authors**: *Juncheol Shin, Junhyuk So, Sein Park, Seungyeop Kang, Sungjoo Yoo, Eunhyeok Park*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00375](https://doi.org/10.1109/CVPR52729.2023.00375)

        **Abstract**:

        Straight-through estimator (STE), which enables the gradient flow over the non-differentiable function via approximation, has been favored in studies related to quantization-aware training (QAT). However, STE incurs unstable convergence during QAT, resulting in notable quality degradation in low precision. Recently, pseudo-quantization training has been proposed as an alternative approach to updating the learnable parameters using the pseudo-quantization noise instead of STE. In this study, we propose a novel noise proxy-based integrated pseudo-quantization (NIPQ) that enables unified support of pseudo-quantization for both activation and weight by integrating the idea of truncation on the pseudo-quantization framework. NIPQ updates all of the quantization parameters (e.g., bit-width and truncation boundary) as well as the network parameters via gradient descent without STE instability. According to our extensive experiments, NIPQ outperforms existing quantization algorithms in various vision and language applications by a large margin.

        ----

        ## [368] CUDA: Convolution-Based Unlearnable Datasets

        **Authors**: *Vinu Sankar Sadasivan, Mahdi Soltanolkotabi, Soheil Feizi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00376](https://doi.org/10.1109/CVPR52729.2023.00376)

        **Abstract**:

        Large-scale training of modern deep learning models heavily relies on publicly available data on the web. This potentially unauthorized usage of online data leads to concerns regarding data privacy. Recent works aim to make unlearnable data for deep learning models by adding small, specially designed noises to tackle this issue. However, these methods are vulnerable to adversarial training (AT) and/or are computationally heavy. In this work, we propose a novel, model-free, Convolution-based Unlearnable DAtaset (CUDA) generation technique. CUDA is generated using controlled class-wise convolutions with filters that are randomly generated via a private key. CUDA encourages the network to learn the relation between filters and labels rather than informative features for classifying the clean data. We develop some theoretical analysis demonstrating that CUDA can successfully poison Gaussian mixture data by reducing the clean data performance of the optimal Bayes classifier. We also empirically demonstrate the effectiveness of CUDA with various datasets (CIFAR-10, CIFAR-100, ImageNet-100, and Tiny-ImageNet), and architectures (ResNet-18, VGG-16, Wide ResNet-34-10, DenseNet-121, DeIT, EfficientNetV2-S, and MobileNetV2). Our experiments show that CUDA is robust to various data augmentations and training approaches such as smoothing, AT with different budgets, transfer learning, and fine-tuning. For instance, training a ResNet-18 on ImageNet-100 CUDA achieves only 8.96%, 40.08%, and 20.58% clean test accuracies with empirical risk minimization (ERM), L
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">∞</inf>
AT, and L2 AT, respectively. Here, ERM on the clean training data achieves a clean test accuracy of 80.66%. CUDA exhibits unlearnability effect with ERM even when only a fraction of the training dataset is perturbed. Furthermore, we also show that CUDA is robust to adaptive defenses designed specifically to break it.

        ----

        ## [369] KD-DLGAN: Data Limited Image Generation via Knowledge Distillation

        **Authors**: *Kaiwen Cui, Yingchen Yu, Fangneng Zhan, Shengcai Liao, Shijian Lu, Eric P. Xing*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00377](https://doi.org/10.1109/CVPR52729.2023.00377)

        **Abstract**:

        Generative Adversarial Networks (GANs) rely heavily on large-scale training data for training high-quality image generation models. With limited training data, the GAN discriminator often suffers from severe overfitting which directly leads to degraded generation especially in generation diversity. Inspired by the recent advances in knowledge distillation (KD), we propose KD-DLGAN, a knowledge-distillation based generation framework that introduces pre-trained vision-language models for training effective data-limited generation models. KD-DLGAN consists of two innovative designs. The first is aggregated generative KD that mitigates the discriminator overfitting by challenging the discriminator with harder learning tasks and distilling more generalizable knowledge from the pre-trained models. The second is correlated generative KD that improves the generation diversity by distilling and preserving the diverse image-text correlation within the pre-trained models. Extensive experiments over multiple benchmarks show that KD-DLGAN achieves superior image generation with limited training data. In addition, KD-DLGAN complements the state-of-the-art with consistent and substantial performance gains. Note that codes will be released.

        ----

        ## [370] Spider GAN: Leveraging Friendly Neighbors to Accelerate GAN Training

        **Authors**: *Siddarth Asokan, Chandra Sekhar Seelamantula*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00378](https://doi.org/10.1109/CVPR52729.2023.00378)

        **Abstract**:

        Training Generative adversarial networks (GANs) stably is a challenging task. The generator in GANs transform noise vectors, typically Gaussian distributed, into realistic data such as images. In this paper, we propose a novel approach for training GANs with images as inputs, but without enforcing any pairwise constraints. The intuition is that images are more structured than noise, which the generator can leverage to learn a more robust transformation. The process can be made efficient by identifying closely related datasets, or a “friendly neighborhood” of the target distribution, inspiring the moniker, Spider GAN. To define friendly neighborhoods leveraging proximity between datasets, we propose a new measure called the signed inception distance (SID), inspired by the polyharmonic kernel. We show that the Spider GAN formulation results in faster convergence, as the generator can discover correspondence even between seemingly unrelated datasets, for instance, between TinyImageNet and CelebA faces. Further, we demonstrate cascading Spider GAN, where the output distribution from a pre-trained GAN generator is used as the input to the subsequent network. Effectively, transporting one distribution to another in a cascaded fashion until the target is learnt – a new flavor of transfer learning. We demonstrate the efficacy of the Spider approach on DCGAN, conditional GAN, PGGAN, StyleGAN2 and StyleGAN3. The proposed approach achieves state-of-the-art Fréchet inception distance (FID) values, with one-fifth of the training iterations, in comparison to their baseline counterparts on high-resolution small datasets such as MetFaces, Ukiyo-E Faces and AFHQ-Cats.

        ----

        ## [371] Efficient Verification of Neural Networks Against LVM-Based Specifications

        **Authors**: *Harleen Hanspal, Alessio Lomuscio*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00379](https://doi.org/10.1109/CVPR52729.2023.00379)

        **Abstract**:

        The deployment of perception systems based on neural networks in safety critical applications requires assurance on their robustness. Deterministic guarantees on network robustness require formal verification. Standard approaches for verifying robustness analyse invariance to analytically defined transformations, but not the diverse and ubiquitous changes involving object pose, scene viewpoint, occlusions, etc. To this end, we present an efficient approach for verifying specifications definable using Latent Variable Models that capture such diverse changes. The approach involves adding an invertible encoding head to the network to be verified, enabling the verification of latent space sets with minimal reconstruction overhead. We report verification experiments for three classes of proposed latent space specifications, each capturing different types of realistic input variations. Differently from previous work in this area, the proposed approach is relatively independent of input dimensionality and scales to a broad class of deep networks and real-world datasets by mitigating the inefficiency and decoder expressivity dependence in the present state-of-the-art.

        ----

        ## [372] Bi-directional Feature Fusion Generative Adversarial Network for Ultra-high Resolution Pathological Image Virtual Re-staining

        **Authors**: *Kexin Sun, Zhineng Chen, Gongwei Wang, Jun Liu, Xiongjun Ye, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00380](https://doi.org/10.1109/CVPR52729.2023.00380)

        **Abstract**:

        The cost of pathological examination makes virtual restaining of pathological images meaningful. However, due to the ultra-high resolution of pathological images, traditional virtual restaining methods have to divide a WSI image into patches for model training and inference. Such a limitation leads to the lack of global information, resulting in observable differences in color, brightness and contrast when the re-stained patches are merged to generate an image of larger size. We summarize this issue as the square effect. Some existing methods try to solve this issue through overlapping between patches or simple postprocessing. But the former one is not that effective, while the latter one requires carefully tuning. In order to eliminate the square effect, we design a bidirectional feature fusion generative adversarial network (BFF-GAN) with a global branch and a local branch. It learns the inter-patch connections through the fusion of global and local features plus patch-wise attention. We perform experiments on both the private dataset RCC and the public dataset ANHIR. The results show that our model achieves competitive performance and is able to generate extremely real images that are deceptive even for experienced pathologists, which means it is of great clinical significance.

        ----

        ## [373] DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection

        **Authors**: *Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, Ting Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00381](https://doi.org/10.1109/CVPR52729.2023.00381)

        **Abstract**:

        Visual anomaly detection, an important problem in computer vision, is usually formulated as a one-class classification and segmentation task. The student-teacher (S- T) framework has proved to be effective in solving this chal-lenge. However, previous works based on S-T only empirically applied constraints on normal data and fused multilevel information. In this study, we propose an improved model called DeS TSeg, which integrates a pre-trained teacher network, a denoising student encoder-decoder, and a segmentation network into one framework. First, to strengthen the constraints on anomalous data, we intro-duce a denoising procedure that allows the student net-work to learn more robust representations. From synthet-ically corrupted normal images, we train the student net-work to match the teacher network feature of the same images without corruption. Second, to fuse the multi-level S-T features adaptively, we train a segmentation network with rich supervision from synthetic anomaly masks, achieving a substantial performance improvement. Experiments on the industrial inspection benchmark dataset demonstrate that our method achieves state-of-the-art performance, 98.6% on image-level AUC, 75.8% on pixel-level average precision, and 76.4% on instance-level average precision.

        ----

        ## [374] OmniAL: A Unified CNN Framework for Unsupervised Anomaly Localization

        **Authors**: *Ying Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00382](https://doi.org/10.1109/CVPR52729.2023.00382)

        **Abstract**:

        Unsupervised anomaly localization and detection is crucial for industrial manufacturing processes due to the lack of anomalous samples. Recent unsupervised advances on industrial anomaly detection achieve high performance by training separate models for many different categories. The model storage and training time cost of this paradigm is high. Moreover, the setting of one-model-N-classes leads to fearful degradation of existing methods. In this paper, we propose a unified CNN framework for unsupervised anomaly localization, named OmniAL. This method conquers aforementioned problems by improving anomaly synthesis, reconstruction and localization. To prevent the model learning identical reconstruction, it trains the model with proposed panel-guided synthetic anomaly data rather than directly using normal data. It increases anomaly reconstruction error for multi-class distribution by using a network that is equipped with proposed Dilated Channel and Spatial Attention (DCSA) blocks. To better localize the anomaly regions, it employs proposed DiffNeck between reconstruction and localization sub-networks to explore multi-level differences. Experiments on 15-class MVTecAD and 12-class VisA datasets verify the advantage of proposed OmniAL that surpasses the state-of-the-art of unified models. On 15-class-MVTecAD/12-class-VisA, its single unified model achieves 97.2/87.8 image-AUROC, 98.3/96.6 pixel-AUROC and 73.4/41.7 pixel-AP for anomaly detection and localization respectively. Besides that, we make the first attempt to conduct a comprehensive study on the robustness of unsupervised anomaly localization and detection methods against different level adversarial attacks. Experiential results show OmniAL has good application prospects for its superior performance.

        ----

        ## [375] Federated Incremental Semantic Segmentation

        **Authors**: *Jiahua Dong, Duzhen Zhang, Yang Cong, Wei Cong, Henghui Ding, Dengxin Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00383](https://doi.org/10.1109/CVPR52729.2023.00383)

        **Abstract**:

        Federated learning-based semantic segmentation (FSS) has drawn widespread attention via decentralized training on local clients. However, most FSS models assume categories are fixed in advance, thus heavily undergoing forgetting on old categories in practical applications where local clients receive new categories incrementally while have no memory storage to access old classes. Moreover, new clients collecting novel classes may join in the global training of FSS, which further exacerbates catastrophic forgetting. To surmount the above challenges, we propose a Forgetting-Balanced Learning (FBL) model to address heterogeneous forgetting on old classes from both intra-client and interclient aspects. Specifically, under the guidance of pseudo labels generated via adaptive class-balanced pseudo labeling, we develop a forgetting-balanced semantic compensation loss and a forgetting-balanced relation consistency loss to rectify intra-client heterogeneous forgetting of old categories with background shift. It performs balanced gradient propagation and relation consistency distillation within local clients. Moreover, to tackle heterogeneous forgetting from inter-client aspect, we propose a task transition monitor. It can identify new classes under privacy protection and store the latest old global model for relation distillation. Qualitative experiments reveal large improvement of our model against comparison methods. The code is available at https://github.com/JiahuaDong/FISS.

        ----

        ## [376] Re-Thinking Federated Active Learning Based on Inter-Class Diversity

        **Authors**: *Sangmook Kim, Sangmin Bae, Hwanjun Song, Se-Young Yun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00384](https://doi.org/10.1109/CVPR52729.2023.00384)

        **Abstract**:

        Although federated learning has made awe-inspiring advances, most studies have assumed that the client's data are fully labeled. However, in a real-world scenario, every client may have a significant amount of unlabeled instances. Among the various approaches to utilizing un-labeled data, a federated active learning framework has emerged as a promising solution. In the decentralized setting, there are two types of available query selector models, namely ‘global’ and ‘local-only’ models, but little literature discusses their performance dominance and its causes. In this work, we first demonstrate that the superiority of two selector models depends on the global and local inter-class diversity. Furthermore, we observe that the global and local-only models are the keys to resolving the imbalance of each side. Based on our findings, we propose LoGo, a FAL sampling strategy robust to varying local heterogeneity levels and global imbalance ratio, that integrates both models by two steps of active selection scheme. LoGo consistently outperforms six active learning strategies in the total number of 38 experimental settings. The code is available at: https://github.com/raymin0223/LoGo.

        ----

        ## [377] Federated Domain Generalization with Generalization Adjustment

        **Authors**: *Ruipeng Zhang, Qinwei Xu, Jiangchao Yao, Ya Zhang, Qi Tian, Yanfeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00385](https://doi.org/10.1109/CVPR52729.2023.00385)

        **Abstract**:

        Federated Domain Generalization (FedDG) attempts to learn a global model in a privacy-preserving manner that generalizes well to new clients possibly with domain shift. Recent exploration mainly focuses on designing an unbiased training strategy within each individual domain. However, without the support of multi-domain data jointly in the minibatch training, almost all methods cannot guarantee the generalization under domain shift. To overcome this problem, we propose a novel global objective incorporating a new variance reduction regularizer to encourage fairness. A novel FL-friendly method named Generalization Adjustment (GA) is proposed to optimize the above objective by dynamically calibrating the aggregation weights. The theoretical analysis of GA demonstrates the possibility to achieve a tighter generalization bound with an explicit reweighted aggregation, substituting the implicit multi-domain data sharing that is only applicable to the conventional DG settings. Besides, the proposed algorithm is generic and can be combined with any local client training-based methods. Extensive experiments on several benchmark datasets have shown the effectiveness of the proposed method, with consistent improvements over several FedDG algorithms when used in combination. The source code is released at https://github.com/MediaBrain-SJTU/FedDG-GA

        ----

        ## [378] On the Effectiveness of Partial Variance Reduction in Federated Learning with Heterogeneous Data

        **Authors**: *Bo Li, Mikkel N. Schmidt, Tommy S. Alstrøm, Sebastian U. Stich*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00386](https://doi.org/10.1109/CVPR52729.2023.00386)

        **Abstract**:

        Data heterogeneity across clients is a key challenge in federated learning. Prior works address this by either aligning client and server models or using control variates to correct client model drift. Although these methods achieve fast convergence in convex or simple non-convex problems, the performance in over-parameterized models such as deep neural networks is lacking. In this paper, we first revisit the widely used FedAvg algorithm in a deep neural network to understand how data heterogeneity influences the gradient updates across the neural network layers. We observe that while the feature extraction layers are learned efficiently by FedAvg, the substantial diversity of the final classification layers across clients impedes the performance. Motivated by this, we propose to correct model drift by variance reduction only on the final layers. We demonstrate that this significantly outperforms existing benchmarks at a similar or lower communication cost. We furthermore provide proof for the convergence rate of our algorithm.

        ----

        ## [379] The Resource Problem of Using Linear Layer Leakage Attack in Federated Learning

        **Authors**: *Joshua C. Zhao, Ahmed Roushdy Elkordy, Atul Sharma, Yahya H. Ezzeldin, Salman Avestimehr, Saurabh Bagchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00387](https://doi.org/10.1109/CVPR52729.2023.00387)

        **Abstract**:

        Secure aggregation promises a heightened level of privacy in federated learning, maintaining that a server only has access to a decrypted aggregate update. Within this setting, linear layer leakage methods are the only data reconstruction attacks able to scale and achieve a high leakage rate regardless of the number of clients or batch size. This is done through increasing the size of an injected fully-connected (FC) layer. However, this results in a resource overhead which grows larger with an increasing number of clients. We show that this resource overhead is caused by an incorrect perspective in all prior work that treats an attack on an aggregate update in the same way as an individual update with a larger batch size. Instead, by attacking the update from the perspective that aggregation is combining multiple individual updates, this allows the application of sparsity to alleviate resource overhead. We show that the use of sparsity can decrease the model size overhead by over 327x and the computation time by 3.34x compared to SOTA while maintaining equivalent total leakage rate, 77% even with 1000 clients in aggregation.

        ----

        ## [380] Unlearnable Clusters: Towards Label-Agnostic Unlearnable Examples

        **Authors**: *Jiaming Zhang, Xingjun Ma, Qi Yi, Jitao Sang, Yu-Gang Jiang, Yaowei Wang, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00388](https://doi.org/10.1109/CVPR52729.2023.00388)

        **Abstract**:

        There is a growing interest in developing unlearnable examples (UEs) against visual privacy leaks on the Internet. UEs are training samples added with invisible but unlearnable noise, which have been found can prevent unauthorized training of machine learning models. UEs typically are generated via a bilevel optimization framework with a surrogate model to remove (minimize) errors from the original samples, and then applied to protect the data against unknown target models. However, existing UE generation methods all rely on an ideal assumption called label-consistency, where the hackers and protectors are assumed to hold the same label for a given sample. In this work, we propose and promote a more practical label-agnostic setting, where the hackers may exploit the protected data quite differently from the protectors. E.g., amclass unlearnable dataset held by the protector may be exploited by the hacker as a n-class dataset. Existing UE generation methods are rendered ineffective in this challenging setting. To tackle this challenge, we present a novel technique called Unlearnable Clusters (UCs) to generate label-agnostic unlearnable examples with cluster-wise perturbations. Furthermore, we propose to leverage Vision-and-Language Pre-trained Models (VLPMs) like CLIP as the surrogate model to improve the transferability of the crafted UCs to diverse domains. We empirically verify the effectiveness of our proposed approach under a variety of settings with different datasets, target models, and even commercial platforms Microsoft Azure and Baidu PaddlePaddle. Code is available at https://github.com/jiamingzhang94/Unlearnable-Clusters.

        ----

        ## [381] Implicit Identity Leakage: The Stumbling Block to Improving Deepfake Detection Generalization

        **Authors**: *Shichao Dong, Jin Wang, Renhe Ji, Jiajun Liang, Haoqiang Fan, Zheng Ge*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00389](https://doi.org/10.1109/CVPR52729.2023.00389)

        **Abstract**:

        In this paper, we analyse the generalization ability of binary classifiers for the task of deepfake detection. We find that the stumbling block to their generalization is caused by the unexpected learned identity representation on images. Termed as the Implicit Identity Leakage, this phenomenon has been qualitatively and quantitatively verified among various DNNs. Furthermore, based on such understanding, we propose a simple yet effective method named the ID-unaware Deepfake Detection Model to reduce the influence of this phenomenon. Extensive experimental results demonstrate that our method outperforms the state-of-the-art in both in-dataset and cross-dataset evaluation. The code is available at https://github.com/megvii-research/CADDM.

        ----

        ## [382] Backdoor Defense via Adaptively Splitting Poisoned Dataset

        **Authors**: *Kuofeng Gao, Yang Bai, Jindong Gu, Yong Yang, Shu-Tao Xia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00390](https://doi.org/10.1109/CVPR52729.2023.00390)

        **Abstract**:

        Backdoor defenses have been studied to alleviate the threat of deep neural networks (DNNs) being backdoor attacked and thus maliciously altered. Since DNNs usually adopt some external training data from an untrusted third party, a robust backdoor defense strategy during the training stage is of importance. We argue that the core of training-time defense is to select poisoned samples and to handle them properly. In this work, we summarize the training-time defenses from a unified framework as splitting the poisoned dataset into two data pools. Under our framework, we propose an adaptively splitting dataset-based defense (ASD). Concretely, we apply loss-guided split and meta-learning-inspired split to dynamically update two data pools. With the split clean data pool and polluted data pool, ASD successfully defends against backdoor attacks during training. Extensive experiments on multiple benchmark datasets and DNN models against six state-of-the-art backdoor attacks demonstrate the superiority of our ASD. Our code is available at https://github.com/KuofengGao/ASD.

        ----

        ## [383] How to Backdoor Diffusion Models?

        **Authors**: *Sheng-Yen Chou, Pin-Yu Chen, Tsung-Yi Ho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00391](https://doi.org/10.1109/CVPR52729.2023.00391)

        **Abstract**:

        Diffusion models are state-of-the-art deep learning empowered generative models that are trained based on the principle of learning forward and reverse diffusion processes via progressive noise-addition and denoising. To gain a better understanding of the limitations and potential risks, this paper presents the first study on the robustness of diffusion models against backdoor attacks. Specifically, we propose BadDiffusion, a novel attack framework that engineers compromised diffusion processes during model training for backdoor implantation. At the inference stage, the backdoored diffusion model will behave just like an untam-pered generator for regular data inputs, while falsely generating some targeted outcome designed by the bad actor upon receiving the implanted trigger signal. Such a critical risk can be dreadful for downstream tasks and applications built upon the problematic model. Our extensive experiments on various backdoor attack settings show that BadDiffusion can consistently lead to compromised diffusion models with high utility and target specificity. Even worse, BadDiffusion can be made cost-effective by simply finetuning a clean pre-trained diffusion model to implant backdoors. We also explore some possible countermeasures for risk mitigation. Our results call attention to potential risks and possible misuse of diffusion models. Our code is available on https://github.com/IBM/BadDiffusion.

        ----

        ## [384] TrojViT: Trojan Insertion in Vision Transformers

        **Authors**: *Mengxin Zheng, Qian Lou, Lei Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00392](https://doi.org/10.1109/CVPR52729.2023.00392)

        **Abstract**:

        Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform back-door attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Naively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack TrojViT. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses parameter distillation to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify 99.64% of test images to a target class by flipping 345 bits on a ViT for ImageNet.

        ----

        ## [385] TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets

        **Authors**: *Weixin Chen, Dawn Song, Bo Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00393](https://doi.org/10.1109/CVPR52729.2023.00393)

        **Abstract**:

        Diffusion models have achieved great success in a range of tasks, such as image synthesis and molecule design. As such successes hinge on large-scale training data collected from diverse sources, the trustworthiness of these collected data is hard to control or audit. In this work, we aim to explore the vulnerabilities of diffusion models under potential training data manipulations and try to answer: How hard is it to perform Trojan attacks on well-trained diffusion models? What are the adversarial targets that such Trojan attacks can achieve? To answer these questions, we propose an effective Trojan attack against diffusion models, TrojDiff, which optimizes the Trojan diffusion and generative processes during training. In particular, we design novel transitions during the Trojan diffusion process to diffuse adversarial targets into a biased Gaussian distribution and propose a new parameterization of the Trojan generative process that leads to an effective training objective for the attack. In addition, we consider three types of adversarial targets: the Trojaned diffusion models will always output instances belonging to a certain class from the in-domain distribution (In-D2D attack), out-of-domain distribution (Out-D2D-attack), and one specific instance (D2I attack). We evaluate TrojDiff on CIFAR-10 and CelebA datasets against both DDPM and DDIM diffusion models. We show that TrojDiff always achieves high attack performance under different adversarial targets using different types of triggers, while the performance in benign environments is preserved. The code is available at https://github.com/chenweixin107/TrojDiff.

        ----

        ## [386] Ensemble-based Blackbox Attacks on Dense Prediction

        **Authors**: *Zikui Cai, Yaoteng Tan, M. Salman Asif*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00394](https://doi.org/10.1109/CVPR52729.2023.00394)

        **Abstract**:

        We propose an approach for adversarial attacks on dense prediction models (such as object detectors and segmentation). It is well known that the attacks generated by a single surrogate model do not transfer to arbitrary (blackbox) victim models. Furthermore, targeted attacks are often more challenging than the untargeted attacks. In this paper, we show that a carefully designed ensemble can create effective attacks for a number of victim models. In particular, we show that normalization of the weights for individual models plays a critical role in the success of the attacks. We then demonstrate that by adjusting the weights of the ensemble according to the victim model can further improve the performance of the attacks. We performed a number of experiments for object detectors and segmentation to highlight the significance of the our proposed methods. Our proposed ensemble-based method outperforms existing blackbox attack methods for object detection and segmentation. Finally we show that our proposed method can also generate a single perturbation that can fool multiple blackbox detection and segmentation models simultaneously. Code is available at https://github.com/CSIPlab/EBAD

        ----

        ## [387] Efficient Loss Function by Minimizing the Detrimental Effect of Floating-Point Errors on Gradient-Based Attacks

        **Authors**: *Yunrui Yu, Cheng-Zhong Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00395](https://doi.org/10.1109/CVPR52729.2023.00395)

        **Abstract**:

        Attackers can deceive neural networks by adding human imperceptive perturbations to their input data; this reveals the vulnerability and weak robustness of current deep-learning networks. Many attack techniques have been proposed to evaluate the model's robustness. Gradient-based attacks suffer from severely overestimating the robustness. This paper identifies that the relative error in calculated gradients caused by floating-point errors, including floating-point underflow and rounding errors, is a fundamental reason why gradient-based attacks fail to accurately assess the model's robustness. Although it is hard to eliminate the relative error in the gradients, we can control its effect on the gradient-based attacks. Correspondingly, we propose an efficient loss function by minimizing the detrimental impact of the floating-point errors on the attacks. Experimental results show that it is more efficient and reliable than other loss functions when examined across a wide range of defence mechanisms.

        ----

        ## [388] The Best Defense is a Good Offense: Adversarial Augmentation Against Adversarial Attacks

        **Authors**: *Iuri Frosio, Jan Kautz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00396](https://doi.org/10.1109/CVPR52729.2023.00396)

        **Abstract**:

        Many defenses against adversarial attacks (e.g. robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">5</sup>
 (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. To this aim, we leverage existing automatic perturbation analysis tools for neural networks. We study the conditions to apply A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">5</sup>
 effectively, analyze the importance of the robustness of the to-be-defended classifier, and inspect the appearance of the robustified images. We show effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label, and demonstrate the benefits of robustifier and classifier co-training. In our tests, A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">5</sup>
consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet. We also show how to apply A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">5</sup>
to create certifiably robust physical objects. Our code at https://github.com/NVlabs/A5 allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested here, including the case of physical attacks.

        ----

        ## [389] Adversarial Robustness via Random Projection Filters

        **Authors**: *Minjing Dong, Chang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00397](https://doi.org/10.1109/CVPR52729.2023.00397)

        **Abstract**:

        Deep Neural Networks show superior performance in various tasks but are vulnerable to adversarial attacks. Most defense techniques are devoted to the adversarial training strategies, however, it is difficult to achieve satisfactory robust performance only with traditional adversarial training. We mainly attribute it to that aggressive perturbations which lead to the loss increment can always be found via gradient ascent in white-box setting. Although some noises can be involved to prevent attacks from deriving precise gradients on inputs, there exist trade-offs between the defense capability and natural generalization. Taking advantage of the properties of random projection, we propose to replace part of convolutional filters with random projection filters, and theoretically explore the geometric representation preservation of proposed synthesized filters via Johnson-Lindenstrauss lemma. We conduct sufficient evaluation on multiple networks and datasets. The experimental results showcase the superiority of proposed random projection filters to state-of-the-art baselines. The code is available on GitHub.

        ----

        ## [390] Jedi: Entropy-Based Localization and Removal of Adversarial Patches

        **Authors**: *Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael B. Abu-Ghazaleh, Ihsen Alouani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00398](https://doi.org/10.1109/CVPR52729.2023.00398)

        **Abstract**:

        Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).

        ----

        ## [391] Exploring the Relationship Between Architectural Design and Adversarially Robust Generalization

        **Authors**: *Aishan Liu, Shiyu Tang, Siyuan Liang, Ruihao Gong, Boxi Wu, Xianglong Liu, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00399](https://doi.org/10.1109/CVPR52729.2023.00399)

        **Abstract**:

        Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the adversarially robust generalization problem. Despite the preliminary understandings devoted to adversarially robust generalization, little is known from the architectural perspective. To bridge the gap, this paper for the first time systematically investigated the relationship between adversarially robust generalization and architectural design. In particular, we comprehensively evaluated 20 most representative adversarially trained architectures on ImageNette and CIFAR-10 datasets towards multiple 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{p}$</tex>
-norm adversarial attacks. Based on the extensive experiments, we found that, under aligned settings, Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization while CNNs tend to overfit on specific attacks and fail to generalize on multiple adversaries. To better understand the nature behind it, we conduct theoretical analysis via the lens of Rademacher complexity. We revealed the fact that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Transformers, which can be often achieved by the specially-designed attention blocks. We hope our paper could help to better understand the mechanism for designing robust DNNs. Our model weights can be found at http://robust.art.

        ----

        ## [392] Improving Robustness of Vision Transformers by Reducing Sensitivity to Patch Corruptions

        **Authors**: *Yong Guo, David Stutz, Bernt Schiele*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00400](https://doi.org/10.1109/CVPR52729.2023.00400)

        **Abstract**:

        Despite their success, vision transformers still remain vulnerable to image corruptions, such as noise or blur. Indeed, we find that the vulnerability mainly stems from the unstable self-attention mechanism, which is inherently built upon patch-based inputs and often becomes overly sensitive to the corruptions across patches. For example, when we only occlude a small number of patches with random noise (e.g., 10%), these patch corruptions would lead to severe accuracy drops and greatly distract intermediate attention layers. To address this, we propose a new training method that improves the robustness of transformers from a new perspective - reducing sensitivity to patch corruptions (RSPC). Specifically, we first identify and occlude/corrupt the most vulnerable patches and then explicitly reduce sensitivity to them by aligning the intermediate features between clean and corrupted examples. We highlight that the construction of patch corruptions is learned adversarially to the following feature alignment process, which is particularly effective and essentially different from existing methods. In experiments, our RSPC greatly improves the stability of attention layers and consistently yields better robustness on various benchmarks, including CIFAR-10/100-C, ImageNet-A, ImageNet-C, and ImageNet-P.

        ----

        ## [393] Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition

        **Authors**: *Xiao Yang, Chang Liu, Longlong Xu, Yikai Wang, Yinpeng Dong, Ning Chen, Hang Su, Jun Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00401](https://doi.org/10.1109/CVPR52729.2023.00401)

        **Abstract**:

        Face recognition is a prevailing authentication solution in numerous biometric applications. Physical adversarial attacks, as an important surrogate, can identify the weak-nesses of face recognition systems and evaluate their ro-bustness before deployed. However, most existing physical attacks are either detectable readily or ineffective against commercial recognition systems. The goal of this work is to develop a more reliable technique that can carry out an end-to-end evaluation of adversarial robustness for commercial systems. It requires that this technique can simultaneously deceive black-box recognition models and evade defensive mechanisms. To fulfill this, we design adversarial textured 3D meshes (AT3D) with an elaborate topology on a human face, which can be 3D-printed and pasted on the attacker's face to evade the defenses. However, the mesh-based op-timization regime calculates gradients in high-dimensional mesh space, and can be trapped into local optima with un-satisfactory transferability. To deviate from the mesh-based space, we propose to perturb the low-dimensional coefficient space based on 3D Morphable Model, which signifi-cantly improves black-box transferability meanwhile enjoying faster search efficiency and better visual quality. Exten-sive experiments in digital and physical scenarios show that our method effectively explores the security vulnerabilities of multiple popular commercial services, including three recognition A PIs, four anti-spoofing A PIs, two prevailing mobile phones and two automated access control systems.

        ----

        ## [394] AltFreezing for More General Video Face Forgery Detection

        **Authors**: *Zhendong Wang, Jianmin Bao, Wengang Zhou, Weilun Wang, Houqiang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00402](https://doi.org/10.1109/CVPR52729.2023.00402)

        **Abstract**:

        Existing face forgery detection models try to discriminate fake images by detecting only spatial artifacts (e.g., generative artifacts, blending) or mainly temporal artifacts (e.g., flickering, discontinuity). They may experience significant performance degradation when facing out-domain artifacts. In this paper, we propose to capture both spatial and temporal artifacts in one model for face forgery detection. A simple idea is to leverage a spatiotemporal model (3D ConvNet). However, we find that it may easily rely on one type of artifact and ignore the other. To address this issue, we present a novel training strategy called AltFreezing for more general face forgery detection. The AltFreezing aims to encourage the model to detect both spatial and temporal artifacts. It divides the weights of a spatiotemporal network into two groups: spatial-related and temporal-related. Then the two groups of weights are alternately frozen during the training process so that the model can learn spatial and temporal features to distinguish real or fake videos. Furthermore, we introduce various video-level data augmentation methods to improve the generalization capability of the forgery detection model. Extensive experiments show that our framework outperforms existing methods in terms of generalization to unseen manipulations and datasets.

        ----

        ## [395] Passive Micron-Scale Time-of-Flight with Sunlight Interferometry

        **Authors**: *Alankar Kotwal, Anat Levin, Ioannis Gkioulekas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00403](https://doi.org/10.1109/CVPR52729.2023.00403)

        **Abstract**:

        We introduce an interferometric technique for passive time-of-flight imaging and depth sensing at micrometer axial resolutions. Our technique uses a full-field Michelson interferometer, modified to use sunlight as the only light source. The large spectral bandwidth of sunlight makes it possible to acquire micrometer-resolution time-resolved scene responses, through a simple axial scanning operation. Additionally, the angular bandwidth of sunlight makes it possible to capture time-of-flight measurements insensitive to indirect illumination effects, such as interreflections and subsurface scattering. We build an experimental prototype that we operate outdoors, under direct sunlight, and in adverse environment conditions such as machine vibrations and vehicle traffic. We use this prototype to demonstrate, for the first time, passive imaging capabilities such as micrometer-scale depth sensing robust to indirect illumination, direct-only imaging, and imaging through diffusers.

        ----

        ## [396] F2-NeRF: Fast Neural Radiance Field Training with Free Camera Trajectories

        **Authors**: *Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, Wenping Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00404](https://doi.org/10.1109/CVPR52729.2023.00404)

        **Abstract**:

        This paper presents a novel grid-based NeRF called F
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
- NeRF (Fast-Free-NeRF) for novel view synthesis, which enables arbitrary input camera trajectories and only costs a few minutes for training. Existing fast grid-based NeRF training frameworks, like Instant-NGP, Plenoxels, DVGO, or TensoRF, are mainly designed for bounded scenes and rely on space warping to handle unbounded scenes. Existing two widely-used space-warping methods are only designed for the forward-facing trajectory or the 360° object-centric trajectory but cannot process arbitrary trajectories. In this paper, we delve deep into the mechanism of space warping to handle unbounded scenes. Based on our analysis, we further propose a novel space-warping method called perspective warping, which allows us to handle arbitrary trajectories in the grid-based NeRF framework. Extensive experiments demonstrate that F
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-NeRF is able to use the same perspective warping to render high-quality images on two standard datasets and a new free trajectory dataset collected by us. Project page: totoro97.github.io/projects/f2-nerf.

        ----

        ## [397] NoPe-NeRF: Optimising Neural Radiance Field with No Pose Prior

        **Authors**: *Wenjing Bian, Zirui Wang, Kejie Li, Jia-Wang Bian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00405](https://doi.org/10.1109/CVPR52729.2023.00405)

        **Abstract**:

        Training a Neural Radiance Field (NeRF) without precomputed camera poses is challenging. Recent advances in this direction demonstrate the possibility of jointly optimising a NeRF and camera poses in forward-facing scenes. However, these methods still face difficulties during dramatic camera movement. We tackle this challenging problem by incorporating undistorted monocular depth priors. These priors are generated by correcting scale and shift parameters during training, with which we are then able to constrain the relative poses between consecutive frames. This constraint is achieved using our proposed novel loss functions. Experiments on real-world indoor and outdoor scenes show that our method can handle challenging camera trajectories and outperforms existing methods in terms of novel view rendering quality and pose estimation accuracy. Our project page is https://nope-nerf.active.vision.

        ----

        ## [398] BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields

        **Authors**: *Peng Wang, Lingzhe Zhao, Ruijie Ma, Peidong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00406](https://doi.org/10.1109/CVPR52729.2023.00406)

        **Abstract**:

        Neural Radiance Fields (NeRF) have received considerable attention recently, due to its impressive capability in photo-realistic 3D reconstruction and novel view synthesis, given a set of posed camera images. Earlier work usually assumes the input images are of good quality. However, image degradation (e.g. image motion blur in low-light conditions) can easily happen in real-world scenarios, which would further affect the rendering quality of NeRF. In this paper, we present a novel bundle adjusted deblur Neural Radiance Fields (BAD-NeRF), which can be robust to severe motion blurred images and inaccurate camera poses. Our approach models the physical image formation process of a motion blurred image, and jointly learns the parameters of NeRF and recovers the camera motion trajectories during exposure time. In experiments, we show that by directly modeling the real physical image formation process, BADNeRF achieves superior performance over prior works on both synthetic and real datasets. Code and data are available at https://github.com/WU-CVGL/BAD-NeRF.

        ----

        ## [399] DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models

        **Authors**: *Jamie Wynn, Daniyar Turmukhambetov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00407](https://doi.org/10.1109/CVPR52729.2023.00407)

        **Abstract**:

        Under good conditions, Neural Radiance Fields (NeRFs) have shown impressive results on novel view synthesis tasks. NeRFs learn a scene's color and density fields by minimizing the photometric discrepancy between training views and differentiable renderings of the scene. Once trained from a sufficient set of views, NeRFs can generate novel views from arbitrary camera positions. However, the scene geometry and color fields are severely under-constrained, which can lead to artifacts, especially when trained with few input views. To alleviate this problem we learn a prior over scene geometry and color, using a denoising diffusion model (DDM). Our DDM is trained on RGBD patches of the synthetic Hypersim dataset and can be used to predict the gradient of the logarithm of a joint probability distribution of color and depth patches. We show that, these gradients of logarithms of RGBD patch priors serve to regularize geom-etry and color of a scene. During NeRF training, random RGBD patches are rendered and the estimated gradient of the log-likelihood is backpropagated to the color and density fields. Evaluations on LLFF, the most relevant dataset, show that our learned prior achieves improved quality in the reconstructed geometry and improved generalization to novel views. Evaluations on DTU show improved Reconstruction quality among NeRF methods.

        ----

        

[Go to the previous page](CVPR-2023-list01.md)

[Go to the next page](CVPR-2023-list03.md)

[Go to the catalog section](README.md)