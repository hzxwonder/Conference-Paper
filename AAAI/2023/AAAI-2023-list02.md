## [200] RADIANT: Radar-Image Association Network for 3D Object Detection

**Authors**: *Yunfei Long, Abhinav Kumar, Daniel D. Morris, Xiaoming Liu, Marcos Castro, Punarjay Chakravarty*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25270](https://doi.org/10.1609/aaai.v37i2.25270)

**Abstract**:

As a direct depth sensor, radar holds promise as a tool to improve monocular 3D object detection, which suffers from depth errors, due in part to the depth-scale ambiguity. On the other hand, leveraging radar depths is hampered by difficulties in precisely associating radar returns with 3D estimates from monocular methods, effectively erasing its benefits. This paper proposes a fusion network that addresses this radar-camera association challenge. We train our network to predict the 3D offsets between radar returns and object centers, enabling radar depths to enhance the accuracy of 3D monocular detection. By using parallel radar and camera backbones, our network fuses information at both the feature level and detection level, while at the same time leveraging a state-of-the-art monocular detection technique without retraining it. Experimental results show significant improvement in mean average precision and translation error on the nuScenes dataset over monocular counterparts. Our source code is available at https://github.com/longyunf/radiant.

----

## [201] CRIN: Rotation-Invariant Point Cloud Analysis and Rotation Estimation via Centrifugal Reference Frame

**Authors**: *Yujing Lou, Zelin Ye, Yang You, Nianjuan Jiang, Jiangbo Lu, Weiming Wang, Lizhuang Ma, Cewu Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25271](https://doi.org/10.1609/aaai.v37i2.25271)

**Abstract**:

Various recent methods attempt to implement rotation-invariant 3D deep learning by replacing the input coordinates of points with relative distances and angles. Due to the incompleteness of these low-level features, they have to undertake the expense of losing global information. In this paper, we propose the CRIN, namely Centrifugal Rotation-Invariant Network. CRIN directly takes the coordinates of points as input and transforms local points into rotation-invariant representations via centrifugal reference frames. Aided by centrifugal reference frames, each point corresponds to a discrete rotation so that the information of rotations can be implicitly stored in point features. Unfortunately, discrete points are far from describing the whole rotation space. We further introduce a continuous distribution for 3D rotations based on points. Furthermore, we propose an attention-based down-sampling strategy to sample points invariant to rotations. A relation module is adopted at last for reinforcing the long-range dependencies between sampled points and predicts the anchor point for unsupervised rotation estimation. Extensive experiments show that our method achieves rotation invariance, accurately estimates the object rotation, and obtains state-of-the-art results on rotation-augmented classification and part segmentation. Ablation studies validate the effectiveness of the network design.

----

## [202] See Your Emotion from Gait Using Unlabeled Skeleton Data

**Authors**: *Haifeng Lu, Xiping Hu, Bin Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25272](https://doi.org/10.1609/aaai.v37i2.25272)

**Abstract**:

This paper focuses on contrastive learning for gait-based emotion recognition. The existing contrastive learning approaches are rarely suitable for learning skeleton-based gait representations, which suffer from limited gait diversity and inconsistent semantics. In this paper, we propose a Cross-coordinate contrastive learning framework utilizing Ambiguity samples for self-supervised Gait-based Emotion representation (CAGE). First, we propose ambiguity transform to push positive samples into ambiguous semantic space. By learning similarities between ambiguity samples and positive samples, our model can learn higher-level semantics of the gait sequences and maintain semantic diversity. Second, to encourage learning the semantic invariance, we uniquely propose cross-coordinate contrastive learning between the Cartesian coordinate and the Spherical coordinate, which brings rich supervisory signals to learn the intrinsic semantic consistency information. Exhaustive experiments show that CAGE improves existing self-supervised methods by 5%–10% accuracy, and it achieves comparable or even superior performance to supervised methods.

----

## [203] Learning Progressive Modality-Shared Transformers for Effective Visible-Infrared Person Re-identification

**Authors**: *Hu Lu, Xuezhang Zou, Pingping Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25273](https://doi.org/10.1609/aaai.v37i2.25273)

**Abstract**:

Visible-Infrared Person Re-Identification (VI-ReID) is a challenging retrieval task under complex modality changes. Existing methods usually focus on extracting discriminative visual features while ignoring the reliability and commonality of visual features between different modalities. In this paper, we propose a novel deep learning framework named Progressive Modality-shared Transformer (PMT) for effective VI-ReID. To reduce the negative effect of modality gaps, we first take the gray-scale images as an auxiliary modality and propose a progressive learning strategy. Then, we propose a Modality-Shared Enhancement Loss (MSEL) to guide the model to explore more reliable identity information from modality-shared features. Finally, to cope with the problem of large intra-class differences and small inter-class differences, we propose a Discriminative Center Loss (DCL) combined with the MSEL to further improve the discrimination of reliable features. Extensive experiments on SYSU-MM01 and RegDB datasets show that our proposed framework performs better than most state-of-the-art methods. For model reproduction, we release the source code at https://github.com/hulu88/PMT.

----

## [204] Breaking Immutable: Information-Coupled Prototype Elaboration for Few-Shot Object Detection

**Authors**: *Xiaonan Lu, Wenhui Diao, Yongqiang Mao, Junxi Li, Peijin Wang, Xian Sun, Kun Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25274](https://doi.org/10.1609/aaai.v37i2.25274)

**Abstract**:

Few-shot object detection, expecting detectors to detect novel classes with a few instances, has made conspicuous progress. However, the prototypes extracted by existing meta-learning based methods still suffer from insufficient representative information and lack awareness of query images, which cannot be adaptively tailored to different query images. Firstly, only the support images are involved for extracting prototypes, resulting in scarce perceptual information of query images. Secondly, all pixels of all support images are treated equally when aggregating features into prototype vectors, thus the salient objects are overwhelmed by the cluttered background. In this paper, we propose an Information-Coupled Prototype Elaboration (ICPE) method to generate specific and representative prototypes for each query image. Concretely, a conditional information coupling module is introduced to couple information from the query branch to the support branch, strengthening the query-perceptual information in support features. Besides, we design a prototype dynamic aggregation module that dynamically adjusts intra-image and inter-image aggregation weights to highlight the salient information useful for detecting query images. Experimental results on both Pascal VOC and MS COCO demonstrate that our method achieves state-of-the-art performance in almost all settings. Code will be available at: https://github.com/lxn96/ICPE.

----

## [205] ParaFormer: Parallel Attention Transformer for Efficient Feature Matching

**Authors**: *Xiaoyong Lu, Yaping Yan, Bin Kang, Songlin Du*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25275](https://doi.org/10.1609/aaai.v37i2.25275)

**Abstract**:

Heavy computation is a bottleneck limiting deep-learning-based feature matching algorithms to be applied in many real-time applications. However, existing lightweight networks optimized for Euclidean data cannot address classical feature matching tasks, since sparse keypoint based descriptors are expected to be matched. This paper tackles this problem and proposes two concepts: 1) a novel parallel attention model entitled ParaFormer and 2) a graph based U-Net architecture with attentional pooling. First, ParaFormer fuses features and keypoint positions through the concept of amplitude and phase, and integrates self- and cross-attention in a parallel manner which achieves a win-win performance in terms of accuracy and efficiency. Second, with U-Net architecture and proposed attentional pooling, the ParaFormer-U variant significantly reduces computational complexity, and minimize performance loss caused by downsampling. Sufficient experiments on various applications, including homography estimation, pose estimation, and image matching, demonstrate that ParaFormer achieves state-of-the-art performance while maintaining high efficiency. The efficient ParaFormer-U variant achieves comparable performance with less than 50% FLOPs of the existing attention-based models.

----

## [206] Robust One-Shot Segmentation of Brain Tissues via Image-Aligned Style Transformation

**Authors**: *Jinxin Lv, Xiaoyu Zeng, Sheng Wang, Ran Duan, Zhiwei Wang, Qiang Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25276](https://doi.org/10.1609/aaai.v37i2.25276)

**Abstract**:

One-shot segmentation of brain tissues is typically a dual-model iterative learning: a registration model (reg-model) warps a carefully-labeled atlas onto unlabeled images to initialize their pseudo masks for training a segmentation model (seg-model); the seg-model revises the pseudo masks to enhance the reg-model for a better warping in the next iteration. However, there is a key weakness in such dual-model iteration that the spatial misalignment inevitably caused by the reg-model could misguide the seg-model, which makes it converge on an inferior segmentation performance eventually. In this paper, we propose a novel image-aligned style transformation to reinforce the dual-model iterative learning for robust one-shot segmentation of brain tissues. Specifically, we first utilize the reg-model to warp the atlas onto an unlabeled image, and then employ the Fourier-based amplitude exchange with perturbation to transplant the style of the unlabeled image into the aligned atlas. This allows the subsequent seg-model to learn on the aligned and style-transferred copies of the atlas instead of unlabeled images, which naturally guarantees the correct spatial correspondence of an image-mask training pair, without sacrificing the diversity of intensity patterns carried by the unlabeled images. Furthermore, we introduce a feature-aware content consistency in addition to the image-level similarity to constrain the reg-model for a promising initialization, which avoids the collapse of image-aligned style transformation in the first iteration. Experimental results on two public datasets demonstrate 1) a competitive segmentation performance of our method compared to the fully-supervised method, and 2) a superior performance over other state-of-the-art with an increase of average Dice by up to 4.67%. The source code is available at: https://github.com/JinxLv/One-shot-segmentation-via-IST.

----

## [207] HRDoc: Dataset and Baseline Method toward Hierarchical Reconstruction of Document Structures

**Authors**: *Jiefeng Ma, Jun Du, Pengfei Hu, Zhenrong Zhang, Jianshu Zhang, Huihui Zhu, Cong Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25277](https://doi.org/10.1609/aaai.v37i2.25277)

**Abstract**:

The problem of document structure reconstruction refers to converting digital or scanned documents into corresponding semantic structures. Most existing works mainly focus on splitting the boundary of each element in a single document page, neglecting the reconstruction of semantic structure in multi-page documents. This paper introduces hierarchical reconstruction of document structures as a novel task suitable for NLP and CV fields. To better evaluate the system performance on the new task, we built a large-scale dataset named HRDoc, which consists of 2,500 multi-page documents with nearly 2 million semantic units. Every document in HRDoc has line-level annotations including categories and relations obtained from rule-based extractors and human annotators. Moreover, we proposed an encoder-decoder-based hierarchical document structure parsing system (DSPS) to tackle this problem. By adopting a multi-modal bidirectional encoder and a structure-aware GRU decoder with soft-mask operation, the DSPS model surpass the baseline method by a large margin. All scripts and datasets will be made publicly available at https://github.com/jfma-USTC/HRDoc.

----

## [208] Semantic 3D-Aware Portrait Synthesis and Manipulation Based on Compositional Neural Radiance Field

**Authors**: *Tianxiang Ma, Bingchuan Li, Qian He, Jing Dong, Tieniu Tan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25278](https://doi.org/10.1609/aaai.v37i2.25278)

**Abstract**:

Recently 3D-aware GAN methods with neural radiance field have developed rapidly. However, current methods model the whole image as an overall neural radiance field, which limits the partial semantic editability of synthetic results. Since NeRF renders an image pixel by pixel, it is possible to split NeRF in the spatial dimension. We propose a Compositional Neural Radiance Field (CNeRF) for semantic 3D-aware portrait synthesis and manipulation. CNeRF divides the image by semantic regions and learns an independent neural radiance field for each region, and finally fuses them and renders the complete image. Thus we can manipulate the synthesized semantic regions independently, while fixing the other parts unchanged. Furthermore, CNeRF is also designed to decouple shape and texture within each semantic region. Compared to state-of-the-art 3D-aware GAN methods, our approach enables fine-grained semantic region manipulation, while maintaining high-quality 3D-consistent synthesis. The ablation studies show the effectiveness of the structure and loss function used by our method. In addition real image inversion and cartoon portrait 3D editing experiments demonstrate the application potential of our method.

----

## [209] CFFT-GAN: Cross-Domain Feature Fusion Transformer for Exemplar-Based Image Translation

**Authors**: *Tianxiang Ma, Bingchuan Li, Wei Liu, Miao Hua, Jing Dong, Tieniu Tan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25279](https://doi.org/10.1609/aaai.v37i2.25279)

**Abstract**:

Exemplar-based image translation refers to the task of generating images with the desired style, while conditioning on certain input image. Most of the current methods learn the correspondence between two input domains and lack the mining of information within the domain. In this paper, we propose a more general learning approach by considering two domain features as a whole and learning both inter-domain correspondence and intra-domain potential information interactions. Specifically, we propose a Cross-domain Feature Fusion Transformer (CFFT) to learn inter- and intra-domain feature fusion. Based on CFFT, the proposed CFFT-GAN works well on exemplar-based image translation. Moreover, CFFT-GAN is able to decouple and fuse features from multiple domains by cascading CFFT modules. We conduct rich quantitative and qualitative experiments on several image translation tasks, and the results demonstrate the superiority of our approach compared to state-of-the-art methods. Ablation studies show the importance of our proposed CFFT. Application experimental results reflect the potential of our method.

----

## [210] StyleTalk: One-Shot Talking Head Generation with Controllable Speaking Styles

**Authors**: *Yifeng Ma, Suzhen Wang, Zhipeng Hu, Changjie Fan, Tangjie Lv, Yu Ding, Zhidong Deng, Xin Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25280](https://doi.org/10.1609/aaai.v37i2.25280)

**Abstract**:

Different people speak with diverse personalized speaking styles. Although existing one-shot talking head methods have made significant progress in lip sync, natural facial expressions, and stable head motions, they still cannot generate diverse speaking styles in the final talking head videos. To tackle this problem, we propose a one-shot style-controllable talking face generation framework. In a nutshell, we aim to attain a speaking style from an arbitrary reference speaking video and then drive the one-shot portrait to speak with the reference speaking style and another piece of audio. Specifically, we first develop a style encoder to extract dynamic facial motion patterns of a style reference video and then encode them into a style code. Afterward, we introduce a style-controllable decoder to synthesize stylized facial animations from the speech content and style code. In order to integrate the reference speaking style into generated videos, we design a style-aware adaptive transformer, which enables the encoded style code to adjust the weights of the feed-forward layers accordingly. Thanks to the style-aware adaptation mechanism, the reference speaking style can be better embedded into synthesized videos during decoding. Extensive experiments demonstrate that our method is capable of generating talking head videos with diverse speaking styles from only one portrait image and an audio clip while achieving authentic visual effects. Project Page: https://github.com/FuxiVirtualHuman/styletalk.

----

## [211] Intriguing Findings of Frequency Selection for Image Deblurring

**Authors**: *Xintian Mao, Yiming Liu, Fengze Liu, Qingli Li, Wei Shen, Yan Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25281](https://doi.org/10.1609/aaai.v37i2.25281)

**Abstract**:

Blur was naturally analyzed in the frequency domain, by estimating the latent sharp image and the blur kernel given a blurry image. Recent progress on image deblurring always designs end-to-end architectures and aims at learning the difference between blurry and sharp image pairs from pixel-level, which inevitably overlooks the importance of blur kernels. This paper reveals an intriguing phenomenon that simply applying ReLU operation on the frequency domain of a blur image followed by inverse Fourier transform, i.e., frequency selection, provides faithful information about the blur pattern (e.g., the blur direction and blur level, implicitly shows the kernel pattern). Based on this observation, we attempt to leverage kernel-level information for image deblurring networks by inserting Fourier transform, ReLU operation, and inverse Fourier transform to the standard ResBlock. 1 × 1 convolution is further added to let the network modulate flexible thresholds for frequency selection. We term our newly built block as Res FFT-ReLU Block, which takes advantages of both kernel-level and pixel-level features via learning frequency-spatial dual-domain representations. Extensive experiments are conducted to acquire a thorough analysis on the insights of the method. Moreover, after plugging the proposed block into NAFNet, we can achieve 33.85 dB in PSNR on GoPro dataset. Our method noticeably improves backbone architectures without introducing many parameters, while maintaining low computational complexity. Code is available at https://github.com/DeepMed-Lab/DeepRFT-AAAI2023.

----

## [212] DocEdit: Language-Guided Document Editing

**Authors**: *Puneet Mathur, Rajiv Jain, Jiuxiang Gu, Franck Dernoncourt, Dinesh Manocha, Vlad I. Morariu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25282](https://doi.org/10.1609/aaai.v37i2.25282)

**Abstract**:

Professional document editing tools require a certain level of expertise to perform complex edit operations. To make editing tools accessible to increasingly novice users, we investigate intelligent document assistant systems that can make or suggest edits based on a user's natural language request. Such a system should be able to understand the user's ambiguous requests and contextualize them to the visual cues and textual content found in a document image to edit localized unstructured text and structured layouts. To this end, we propose a new task of language-guided localized document editing, where the user provides a document and an open vocabulary editing request, and the intelligent system produces a command that can be used to automate edits in real-world document editing software. In support of this task, we curate the DocEdit dataset, a collection of approximately 28K instances of user edit requests over PDF and design templates along with their corresponding ground truth software executable commands. To our knowledge, this is the first dataset that provides a diverse mix of edit operations with direct and indirect references to the embedded text and visual objects such as paragraphs, lists, tables, etc. We also propose DocEditor, a Transformer-based localization-aware multimodal (textual, spatial, and visual) model that performs the new task. The model attends to both document objects and related text contents which may be referred to in a user edit request, generating a multimodal embedding that is used to predict an edit command and associated bounding box localizing it. Our proposed model empirically outperforms other baseline deep learning approaches by 15-18%, providing a strong starting point for future work.

----

## [213] Progressive Few-Shot Adaptation of Generative Model with Align-Free Spatial Correlation

**Authors**: *Jongbo Moon, Hyunjun Kim, Jae-Pil Heo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25283](https://doi.org/10.1609/aaai.v37i2.25283)

**Abstract**:

In few-shot generative model adaptation, the model for target domain is prone to the mode-collapse. Recent studies attempted to mitigate the problem by matching the relationship among samples generated from the same latent codes in source and target domains. The objective is further extended to image patch-level to transfer the spatial correlation within an instance. However, the patch-level approach assumes the consistency of spatial structure between source and target domains. For example, the positions of eyes in two domains are almost identical. Thus, it can bring visual artifacts if source and target domain images are not nicely aligned. In this paper, we propose a few-shot generative model adaptation method free from such assumption, based on a motivation that generative models are progressively adapting from the source domain to the target domain. Such progressive changes allow us to identify semantically coherent image regions between instances generated by models at a neighboring training iteration to consider the spatial correlation. We also propose an importance-based patch selection strategy to reduce the complexity of patch-level correlation matching. Our method shows the state-of-the-art few-shot domain adaptation performance in the qualitative and quantitative evaluations.

----

## [214] Minority-Oriented Vicinity Expansion with Attentive Aggregation for Video Long-Tailed Recognition

**Authors**: *WonJun Moon, Hyun Seok Seong, Jae-Pil Heo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25284](https://doi.org/10.1609/aaai.v37i2.25284)

**Abstract**:

A dramatic increase in real-world video volume with extremely diverse and emerging topics naturally forms a long-tailed video distribution in terms of their categories, and it spotlights the need for Video Long-Tailed Recognition (VLTR).
In this work, we summarize the challenges in VLTR and explore how to overcome them.
The challenges are: (1) it is impractical to re-train the whole model for high-quality features, (2) acquiring frame-wise labels requires extensive cost, and (3) long-tailed data triggers biased training.
Yet, most existing works for VLTR unavoidably utilize image-level features extracted from pretrained models which are task-irrelevant, and learn by video-level labels.
Therefore, to deal with such (1) task-irrelevant features and (2) video-level labels, we introduce two complementary learnable feature aggregators.
Learnable layers in each aggregator are to produce task-relevant representations, and each aggregator is to assemble the snippet-wise knowledge into a video representative.
Then, we propose Minority-Oriented Vicinity Expansion (MOVE) that explicitly leverages the class frequency into approximating the vicinity distributions to alleviate (3) biased training.
By combining these solutions, our approach achieves state-of-the-art results on large-scale VideoLT and synthetically induced Imbalanced-MiniKinetics200. 
With VideoLT features from ResNet-50, it attains 18% and 58% relative improvements on head and tail classes over the previous state-of-the-art method, respectively. 
Code and dataset are available at https://github.com/wjun0830/MOVE.

----

## [215] Show, Interpret and Tell: Entity-Aware Contextualised Image Captioning in Wikipedia

**Authors**: *Khanh Nguyen, Ali Furkan Biten, Andrés Mafla, Lluís Gómez, Dimosthenis Karatzas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25285](https://doi.org/10.1609/aaai.v37i2.25285)

**Abstract**:

Humans exploit prior knowledge to describe images, and are able to adapt their explanation to specific contextual information given, even to the extent of inventing plausible explanations when contextual information and images do not match. In this work, we propose the novel task of captioning Wikipedia images by integrating contextual knowledge. Specifically, we produce models that jointly reason over Wikipedia articles, Wikimedia images and their associated descriptions to produce contextualized captions. The same Wikimedia image can be used to illustrate different articles, and the produced caption needs to be adapted to the specific context allowing us to explore the limits of the model to adjust captions to different contextual information. Dealing with out-of-dictionary words and Named Entities is a challenging task in this domain. To address this, we propose a pre-training objective, Masked Named Entity Modeling (MNEM), and show that this pretext task results to significantly improved models. Furthermore, we verify that a model pre-trained in Wikipedia generalizes well to News Captioning datasets. We further define two different test splits according to the difficulty of the captioning task. We offer insights on the role and the importance of each modality and highlight the limitations of our model.

----

## [216] TaCo: Textual Attribute Recognition via Contrastive Learning

**Authors**: *Chang Nie, Yiqing Hu, Yanqiu Qu, Hao Liu, Deqiang Jiang, Bo Ren*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25286](https://doi.org/10.1609/aaai.v37i2.25286)

**Abstract**:

As textual attributes like font are core design elements of document format and page style, automatic attributes recognition favor comprehensive practical applications. Existing approaches already yield satisfactory performance in differentiating disparate attributes, but they still suffer in distinguishing similar attributes with only subtle difference. Moreover, their performance drop severely in real-world scenarios where unexpected and obvious imaging distortions appear. In this paper, we aim to tackle these problems by proposing TaCo, a contrastive framework for textual attribute recognition tailored toward the most common document scenes. Specifically, TaCo leverages contrastive learning to dispel the ambiguity trap arising from vague and open-ended attributes. 
To realize this goal, we design the learning paradigm from three perspectives: 1) generating attribute views, 2) extracting subtle but crucial details, and 3) exploiting valued view pairs for learning, to fully unlock the pre-training potential. 
Extensive experiments show that TaCo surpasses the supervised counterparts and advances the state-of-the-art remarkably on multiple attribute recognition tasks. Online services of TaCo will be made available.

----

## [217] GLT-T: Global-Local Transformer Voting for 3D Single Object Tracking in Point Clouds

**Authors**: *Jiahao Nie, Zhiwei He, Yuxiang Yang, Mingyu Gao, Jing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25287](https://doi.org/10.1609/aaai.v37i2.25287)

**Abstract**:

Current 3D single object tracking methods are typically based on VoteNet, a 3D region proposal network. Despite the success, using a single seed point feature as the cue for offset learning in VoteNet prevents high-quality 3D proposals from being generated. Moreover, seed points with different importance are treated equally in the voting process, aggravating this defect. To address these issues, we propose a novel global-local transformer voting scheme to provide more informative cues and guide the model pay more attention on potential seed points, promoting the generation of high-quality 3D proposals. Technically, a global-local transformer (GLT) module is employed to integrate object- and patch-aware prior into seed point features to effectively form strong feature representation for geometric positions of the seed points, thus providing more robust and accurate cues for offset learning. Subsequently, a simple yet effective training strategy is designed to train the GLT module. We develop an importance prediction branch to learn the potential importance of the seed points and treat the output weights vector as a training constraint term. By incorporating the above components together, we exhibit a superior tracking method GLT-T. Extensive experiments on challenging KITTI and NuScenes benchmarks demonstrate that GLT-T achieves state-of-the-art performance in the 3D single object tracking task. Besides, further ablation studies show the advantages of the proposed global-local transformer voting scheme over the original VoteNet. Code and models will be available at https://github.com/haooozi/GLT-T.

----

## [218] Adapting Object Size Variance and Class Imbalance for Semi-supervised Object Detection

**Authors**: *Yuxiang Nie, Chaowei Fang, Lechao Cheng, Liang Lin, Guanbin Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25288](https://doi.org/10.1609/aaai.v37i2.25288)

**Abstract**:

Semi-supervised object detection (SSOD) attracts extensive research interest due to its great significance in reducing the data annotation effort. Collecting high-quality and category-balanced pseudo labels for unlabeled images is critical to addressing the SSOD problem. However, most of the existing pseudo-labeling-based methods depend on a large and fixed threshold to select high-quality pseudo labels from the predictions of a teacher model. Considering different object classes usually have different detection difficulty levels due to scale variance and data distribution imbalance, conventional pseudo-labeling-based methods are arduous to explore the value of unlabeled data sufficiently. To address these issues, we propose an adaptive pseudo labeling strategy, which can assign thresholds to classes with respect to their “hardness”. This is beneficial for ensuring the high quality of easier classes and increasing the quantity of harder classes simultaneously. Besides, label refinement modules are set up based on box jittering for guaranteeing the localization quality of pseudo labels. To further improve the algorithm’s robustness against scale variance and make the most of pseudo labels, we devise a joint feature-level and prediction-level consistency learning pipeline for transferring the information of the teacher model to the student model. Extensive experiments on COCO and VOC datasets indicate that our method achieves state-of-the-art performance. Especially, it brings mean average precision gains of 2.08 and 1.28 on MS-COCO dataset with 5% and 10% labeled images, respectively.

----

## [219] MIMO Is All You Need：A Strong Multi-in-Multi-Out Baseline for Video Prediction

**Authors**: *Shuliang Ning, Mengcheng Lan, Yanran Li, Chaofeng Chen, Qian Chen, Xunlai Chen, Xiaoguang Han, Shuguang Cui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25289](https://doi.org/10.1609/aaai.v37i2.25289)

**Abstract**:

The mainstream of the existing approaches for video prediction builds up their models based on a Single-In-Single-Out (SISO) architecture, which takes the current frame as input to predict the next frame in a recursive manner. This way often leads to severe performance degradation when they try to extrapolate a longer period of future, thus limiting the practical use of the prediction model. Alternatively, a Multi-In-Multi-Out (MIMO) architecture that outputs all the future frames at one shot naturally breaks the recursive manner and therefore prevents error accumulation. However, only a few MIMO models for video prediction are proposed and they only achieve inferior performance due to the date. 
The real strength of the MIMO model in this area is not well noticed and is largely under-explored. Motivated by that, we conduct a comprehensive investigation in this paper to thoroughly exploit how far a simple MIMO architecture can go. Surprisingly, our empirical studies reveal that a simple MIMO model can outperform the state-of-the-art work with a large margin much more than expected, especially in dealing with long-term error accumulation. 
After exploring a number of ways and designs, we propose a new MIMO architecture based on extending the pure Transformer with local spatio-temporal blocks and a new multi-output decoder, namely MIMO-VP, to establish a new standard in video prediction. We evaluate our model in four highly competitive benchmarks. 
Extensive experiments show that our model wins 1st place on all the benchmarks with remarkable performance gains and surpasses the best SISO model in all aspects including efficiency, quantity, and quality. A dramatic error reduction is achieved when predicting 10 frames on Moving MNIST and Weather datasets respectively. We believe our model can serve as a new baseline to facilitate the future research of video prediction tasks. The code will be released.

----

## [220] Universe Points Representation Learning for Partial Multi-Graph Matching

**Authors**: *Zhakshylyk Nurlanov, Frank R. Schmidt, Florian Bernard*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25290](https://doi.org/10.1609/aaai.v37i2.25290)

**Abstract**:

Many challenges from natural world can be formulated as a graph matching problem. Previous deep learning-based methods mainly consider a full two-graph matching setting. In this work, we study the more general partial matching problem with multi-graph cycle consistency guarantees. Building on a recent progress in deep learning on graphs, we propose a novel data-driven method (URL) for partial multi-graph matching, which uses an object-to-universe formulation and learns latent representations of abstract universe points. The proposed approach advances the state of the art in semantic keypoint matching problem, evaluated on Pascal VOC, CUB, and Willow datasets. Moreover, the set of controlled experiments on a synthetic graph matching dataset demonstrates the scalability of our method to graphs with large number of nodes and its robustness to high partiality.

----

## [221] Robust Image Denoising of No-Flash Images Guided by Consistent Flash Images

**Authors**: *Geunwoo Oh, Jonghee Back, Jae-Pil Heo, Bochang Moon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25291](https://doi.org/10.1609/aaai.v37i2.25291)

**Abstract**:

Images taken in low light conditions typically contain distracting noise, and eliminating such noise is a crucial computer vision problem. Additional photos captured with a camera flash can guide an image denoiser to preserve edges since the flash images often contain fine details with reduced noise. Nonetheless, a denoiser can be misled by inconsistent flash images, which have image structures (e.g., edges) that do not exist in no-flash images. Unfortunately, this disparity frequently occurs as the flash/no-flash pairs are taken in different light conditions. We propose a learning-based technique that robustly fuses the image pairs while considering their inconsistency. Our framework infers consistent flash image patches locally, which have similar image structures with the ground truth, and denoises no-flash images using the inferred ones via a combination model. We demonstrate that our technique can produce more robust results than state-of-the-art methods, given various flash/no-flash pairs with inconsistent image structures. The source code is available at https://github.com/CGLab-GIST/RIDFnF.

----

## [222] Coarse2Fine: Local Consistency Aware Re-prediction for Weakly Supervised Object Localization

**Authors**: *Yixuan Pan, Yao Yao, Yichao Cao, Chongjin Chen, Xiaobo Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25292](https://doi.org/10.1609/aaai.v37i2.25292)

**Abstract**:

Weakly supervised object localization aims to localize objects of interest by using only image-level labels. Existing methods generally segment activation map by threshold to obtain mask and generate bounding box. However, the activation map is locally inconsistent, i.e., similar neighboring pixels of the same object are not equally activated, which leads to the blurred boundary issue: the localization result is sensitive to the threshold, and the mask obtained directly from the activation map loses the fine contours of the object, making it difficult to obtain a tight bounding box. In this paper, we introduce the Local Consistency Aware Re-prediction (LCAR) framework, which aims to recover the complete fine object mask from locally inconsistent activation map and hence obtain a tight bounding box. To this end, we propose the self-guided re-prediction module (SGRM), which employs a novel superpixel aggregation network to replace the post-processing of threshold segmentation. In order to derive more reliable pseudo label from the activation map to supervise the SGRM, we further design an affinity refinement module (ARM) that utilizes the original image feature to better align the activation map with the image appearance, and design a self-distillation CAM (SD-CAM) to alleviate the locator dependence on saliency. Experiments demonstrate that our LCAR outperforms the state-of-the-art on both the CUB-200-2011 and ILSVRC datasets, achieving 95.89% and 70.72% of GT-Know localization accuracy, respectively.

----

## [223] Find Beauty in the Rare: Contrastive Composition Feature Clustering for Nontrivial Cropping Box Regression

**Authors**: *Zhiyu Pan, Yinpeng Chen, Jiale Zhang, Hao Lu, Zhiguo Cao, Weicai Zhong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25293](https://doi.org/10.1609/aaai.v37i2.25293)

**Abstract**:

Automatic image cropping algorithms aim to recompose images like human-being photographers by generating the cropping boxes with improved composition quality. Cropping box regression approaches learn the beauty of composition from annotated cropping boxes. However, the bias of annotations leads to quasi-trivial recomposing results, which has an obvious tendency to the average location of training samples. The crux of this predicament is that the task is naively treated as a box regression problem, where rare samples might be dominated by normal samples, and the composition patterns of rare samples are not well exploited. Observing that similar composition patterns tend to be shared by the cropping boundaries annotated nearly, we argue to find the beauty of composition from the rare samples by clustering the samples with similar cropping boundary annotations, i.e., similar composition patterns. We propose a novel Contrastive Composition Clustering (C2C) to regularize the composition features by contrasting dynamically established similar and dissimilar pairs. In this way, common composition patterns of multiple images can be better summarized, which especially benefits the rare samples and endows our model with better generalizability to render nontrivial results. Extensive experimental results show the superiority of our model compared with prior arts. We also illustrate the philosophy of our design with an interesting analytical visualization.

----

## [224] Domain Decorrelation with Potential Energy Ranking

**Authors**: *Sen Pei, Jiaxi Sun, Richard Yi Da Xu, Shiming Xiang, Gaofeng Meng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25294](https://doi.org/10.1609/aaai.v37i2.25294)

**Abstract**:

Machine learning systems, especially the methods based on deep learning, enjoy great success in modern computer vision tasks under ideal experimental settings. Generally, these classic deep learning methods are built on the i.i.d. assumption, supposing the training and test data are drawn from the same distribution independently and identically. However, the aforementioned i.i.d. assumption is, in general, unavailable in the real-world scenarios, and as a result, leads to sharp performance decay of deep learning algorithms. Behind this, domain shift is one of the primary factors to be blamed. In order to tackle this problem, we propose using Potential Energy Ranking (PoER) to decouple the object feature and the domain feature in given images, promoting the learning of label-discriminative representations while filtering out the irrelevant correlations between the objects and the background. PoER employs the ranking loss in shallow layers to make features with identical category and domain labels close to each other and vice versa. This makes the neural networks aware of both objects and background characteristics, which is vital for generating domain-invariant features. Subsequently, with the stacked convolutional blocks, PoER further uses the contrastive loss to make features within the same categories distribute densely no matter domains, filtering out the domain information progressively for feature alignment. PoER reports superior performance on domain generalization benchmarks, improving the average top-1 accuracy by at least 1.20% compared to the existing methods. Moreover, we use PoER in the ECCV 2022 NICO Challenge, achieving top place with only a vanilla ResNet-18 and winning the jury award. The code has been made publicly available at: https://github.com/ForeverPs/PoER.

----

## [225] PDRF: Progressively Deblurring Radiance Field for Fast Scene Reconstruction from Blurry Images

**Authors**: *Cheng Peng, Rama Chellappa*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25295](https://doi.org/10.1609/aaai.v37i2.25295)

**Abstract**:

We present Progressively Deblurring Radiance Field (PDRF), a novel approach to efficiently reconstruct high quality radiance fields from blurry images. While current State-of-The-Art (SoTA) scene reconstruction methods achieve photo-realistic renderings from clean source views, their performances suffer when the source views are affected by blur, which is commonly observed in the wild. Previous deblurring methods either do not account for 3D geometry, or are computationally intense. To addresses these issues, PDRF uses a progressively deblurring scheme for radiance field modeling, which can accurately model blur with 3D scene context. PDRF further uses an efficient importance sampling scheme that results in fast scene optimization. We perform extensive experiments and show that PDRF is 15X faster than previous SoTA while achieving better performance on both synthetic and real scenes.

----

## [226] Efficient End-to-End Video Question Answering with Pyramidal Multimodal Transformer

**Authors**: *Min Peng, Chongyang Wang, Yu Shi, Xiang-Dong Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25296](https://doi.org/10.1609/aaai.v37i2.25296)

**Abstract**:

This paper presents a new method for end-to-end Video Question Answering (VideoQA), aside from the current popularity of using large-scale pre-training with huge feature extractors. We achieve this with a pyramidal multimodal transformer (PMT) model, which simply incorporates a learnable word embedding layer, a few convolutional and transformer layers. We use the anisotropic pyramid to fulfill video-language interactions across different spatio-temporal scales. In addition to the canonical pyramid, which includes both bottom-up and top-down pathways with lateral connections, novel strategies are proposed to decompose the visual feature stream into spatial and temporal sub-streams at different scales and implement their interactions with the linguistic semantics while preserving the integrity of local and global semantics. We demonstrate better or on-par performances with high computational efficiency against state-of-the-art methods on five VideoQA benchmarks. Our ablation study shows the scalability of our model that achieves competitive results for text-to-video retrieval by leveraging feature extractors with reusable pre-trained weights, and also the effectiveness of the pyramid. Code available at: https://github.com/Trunpm/PMT-AAAI23.

----

## [227] CL3D: Unsupervised Domain Adaptation for Cross-LiDAR 3D Detection

**Authors**: *Xidong Peng, Xinge Zhu, Yuexin Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25297](https://doi.org/10.1609/aaai.v37i2.25297)

**Abstract**:

Domain adaptation for Cross-LiDAR 3D detection is challenging due to the large gap on the raw data representation with disparate point densities and point arrangements. By exploring domain-invariant 3D geometric characteristics and motion patterns, we present an unsupervised domain adaptation method that overcomes above difficulties. First, we propose the Spatial Geometry Alignment module to extract similar 3D shape geometric features of the same object class to align two domains, while eliminating the effect of distinct point distributions. Second, we present Temporal Motion Alignment module to utilize motion features in sequential frames of data to match two domains. Prototypes generated from two modules are incorporated into the pseudo-label reweighting procedure and contribute to our effective self-training framework for the target domain. Extensive experiments show that our method achieves state-of-the-art performance on cross-device datasets, especially for the datasets with large gaps captured by mechanical scanning LiDARs and solid-state LiDARs in various scenes. Project homepage is at https://github.com/4DVLab/CL3D.git.

----

## [228] Better and Faster: Adaptive Event Conversion for Event-Based Object Detection

**Authors**: *Yansong Peng, Yueyi Zhang, Peilin Xiao, Xiaoyan Sun, Feng Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25298](https://doi.org/10.1609/aaai.v37i2.25298)

**Abstract**:

Event cameras are a kind of bio-inspired imaging sensor, which asynchronously collect sparse event streams with many advantages. In this paper, we focus on building better and faster event-based object detectors. To this end, we first propose a computationally efficient event representation Hyper Histogram, which adequately preserves both the polarity and temporal information of events. Then we devise an Adaptive Event Conversion module, which converts events into Hyper Histograms according to event density via an adaptive queue. Moreover, we introduce a novel event-based augmentation method Shadow Mosaic, which significantly improves the event sample diversity and enhances the generalization ability of detection models. We equip our proposed modules on three representative object detection models: YOLOv5, Deformable-DETR, and RetinaNet. Experimental results on three event-based detection datasets (1Mpx, Gen1, and MVSEC-NIGHTL21) demonstrate that our proposed approach outperforms other state-of-the-art methods by a large margin, while achieving a much faster running speed (< 14 ms and < 4 ms for 50 ms event data on the 1Mpx and Gen1 datasets).

----

## [229] CSTAR: Towards Compact and Structured Deep Neural Networks with Adversarial Robustness

**Authors**: *Huy Phan, Miao Yin, Yang Sui, Bo Yuan, Saman A. Zonouz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25299](https://doi.org/10.1609/aaai.v37i2.25299)

**Abstract**:

Model compression and model defense for deep neural networks (DNNs) have been extensively and individually studied. Considering the co-importance of model compactness and robustness in practical applications, several prior works have explored to improve the adversarial robustness of the sparse neural networks. However, the structured sparse models obtained by the existing works suffer severe performance degradation for both benign and robust accuracy, thereby causing a challenging dilemma between robustness and structuredness of compact DNNs.
To address this problem, in this paper, we propose CSTAR, an efficient solution that simultaneously impose Compactness, high STructuredness and high Adversarial Robustness on the target DNN models. By formulating the structuredness and robustness requirement within the same framework, the compressed DNNs can simultaneously achieve high compression performance and strong adversarial robustness. Evaluations for various DNN models on different datasets demonstrate the effectiveness of CSTAR. Compared with the state-of-the-art robust structured pruning, CSTAR shows consistently better performance. For instance, when compressing ResNet-18 on CIFAR-10, CSTAR achieves up to 20.07% and 11.91% improvement for benign accuracy and robust accuracy, respectively. For compressing ResNet-18 with 16x compression ratio on Imagenet, CSTAR obtains 8.58% benign accuracy gain and 4.27% robust accuracy gain compared to the existing robust structured pruning.

----

## [230] Exploring Stochastic Autoregressive Image Modeling for Visual Representation

**Authors**: *Yu Qi, Fan Yang, Yousong Zhu, Yufei Liu, Liwei Wu, Rui Zhao, Wei Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25300](https://doi.org/10.1609/aaai.v37i2.25300)

**Abstract**:

Autoregressive language modeling (ALM) has been successfully used in self-supervised pre-training in Natural language processing (NLP). However, this paradigm has not achieved comparable results with other self-supervised approaches in computer vision (e.g., contrastive learning, masked image modeling). In this paper, we try to find the reason why autoregressive modeling does not work well on vision tasks. To tackle this problem, we fully analyze the limitation of visual autoregressive methods and proposed a novel stochastic autoregressive image modeling (named SAIM) by the two simple designs. First, we serialize the image into patches. Second, we employ the stochastic permutation strategy to generate an effective and robust image context which is critical for vision tasks. To realize this task, we create a parallel encoder-decoder training process in which the encoder serves a similar role to the standard vision transformer focusing on learning the whole contextual information, and meanwhile the decoder predicts the content of the current position so that the encoder and decoder can reinforce each other. Our method significantly improves the performance of autoregressive image modeling and achieves the best accuracy (83.9%) on the vanilla ViT-Base model among methods using only ImageNet-1K data. Transfer performance in downstream tasks also shows that our model achieves competitive performance. Code is available at https://github.com/qiy20/SAIM.

----

## [231] Context-Aware Transformer for 3D Point Cloud Automatic Annotation

**Authors**: *Xiaoyan Qian, Chang Liu, Xiaojuan Qi, Siew-Chong Tan, Edmund Y. Lam, Ngai Wong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25301](https://doi.org/10.1609/aaai.v37i2.25301)

**Abstract**:

3D automatic annotation has received increased attention since manually annotating 3D point clouds is laborious. However, existing methods are usually complicated, e.g., pipelined training for 3D foreground/background segmentation, cylindrical object proposals, and point completion. Furthermore, they often overlook the inter-object feature correlation that is particularly informative to hard samples for 3D annotation. 
To this end, we propose a simple yet effective end-to-end Context-Aware Transformer (CAT) as an automated 3D-box labeler to generate precise 3D box annotations from 2D boxes, trained with a small number of human annotations. We adopt the general encoder-decoder architecture, where the CAT encoder consists of an intra-object encoder (local) and an inter-object encoder (global), performing self-attention along the sequence and batch dimensions, respectively. The former models intra-object interactions among points and the latter extracts feature relations among different objects, thus boosting scene-level understanding.
Via local and global encoders, CAT can generate high-quality 3D box annotations with a streamlined workflow, allowing it to outperform existing state-of-the-arts by up to 1.79% 3D AP on the hard task of the KITTI test set.

----

## [232] Data-Efficient Image Quality Assessment with Attention-Panel Decoder

**Authors**: *Guanyi Qin, Runze Hu, Yutao Liu, Xiawu Zheng, Haotian Liu, Xiu Li, Yan Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25302](https://doi.org/10.1609/aaai.v37i2.25302)

**Abstract**:

Blind Image Quality Assessment (BIQA) is a fundamental task in computer vision, which however remains unresolved due to the complex distortion conditions and diversified image contents. To confront this challenge, we in this paper propose a novel BIQA pipeline based on the Transformer architecture, which achieves an efficient quality-aware feature representation with much fewer data. More specifically, we consider the traditional fine-tuning in BIQA as an interpretation of the pre-trained model. In this way, we further introduce a Transformer decoder to refine the perceptual information of the CLS token from different perspectives. This enables our model to establish the quality-aware feature manifold efficiently while attaining a strong generalization capability. Meanwhile, inspired by the subjective evaluation behaviors of human, we introduce a novel attention panel mechanism, which improves the model performance and reduces the prediction uncertainty simultaneously. The proposed BIQA method maintains a light-weight design with only one layer of the decoder, yet extensive experiments on eight standard BIQA datasets (both synthetic and authentic) demonstrate its superior performance to the state-of-the-art BIQA methods, i.e., achieving the SRCC values of 0.875 (vs. 0.859 in LIVEC) and 0.980 (vs. 0.969 in LIVE). Checkpoints, logs and code will be available at https://github.com/narthchin/DEIQT.

----

## [233] FoPro: Few-Shot Guided Robust Webly-Supervised Prototypical Learning

**Authors**: *Yulei Qin, Xingyu Chen, Chao Chen, Yunhang Shen, Bo Ren, Yun Gu, Jie Yang, Chunhua Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25303](https://doi.org/10.1609/aaai.v37i2.25303)

**Abstract**:

Recently, webly supervised learning (WSL) has been studied to leverage numerous and accessible data from the Internet. Most existing methods focus on learning noise-robust models from web images while neglecting the performance drop caused by the differences between web domain and real-world domain. However, only by tackling the performance gap above can we fully exploit the practical value of web datasets. To this end, we propose a Few-shot guided Prototypical (FoPro) representation learning method, which only needs a few labeled examples from reality and can significantly improve the performance in the real-world domain. Specifically, we initialize each class center with few-shot real-world data as the ``realistic" prototype. Then, the intra-class distance between web instances and ``realistic" prototypes is narrowed by contrastive learning. Finally, we measure image-prototype distance with a learnable metric. Prototypes are polished by adjacent high-quality web images and involved in removing distant out-of-distribution samples. In experiments, FoPro is trained on web datasets with a few real-world examples guided and evaluated on real-world datasets. Our method achieves the state-of-the-art performance on three fine-grained datasets and two large-scale datasets. Compared with existing WSL methods under the same few-shot settings, FoPro still excels in real-world generalization. Code is available at https://github.com/yuleiqin/fopro.

----

## [234] Exposing the Self-Supervised Space-Time Correspondence Learning via Graph Kernels

**Authors**: *Zheyun Qin, Xiankai Lu, Xiushan Nie, Yilong Yin, Jianbing Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25304](https://doi.org/10.1609/aaai.v37i2.25304)

**Abstract**:

Self-supervised space-time correspondence learning is emerging as a promising way of leveraging unlabeled video. Currently, most methods adapt contrastive learning with mining negative samples or reconstruction adapted from the image domain, which requires dense affinity across multiple frames or optical flow constraints. Moreover, video correspondence predictive models require mining more inherent properties in videos, such as structural information. In this work, we propose the VideoHiGraph, a space-time correspondence framework based on a learnable graph kernel. Concerning the video as the spatial-temporal graph, the learning objectives of VideoHiGraph are emanated in a self-supervised manner for predicting unobserved hidden graphs via graph kernel manner. We learn a representation of the temporal coherence across frames in which pairwise similarity defines the structured hidden graph, such that a biased random walk graph kernel along the sub-graph can predict long-range correspondence. Then, we learn a refined representation across frames on the node-level via a dense graph kernel. The self-supervision of the model training is formed by the structural and temporal consistency of the graph. VideoHiGraph achieves superior performance and demonstrates its robustness across the benchmark of label propagation tasks involving objects, semantic parts, keypoints, and instances. Our algorithm implementations have been made publicly available at https://github.com/zyqin19/VideoHiGraph.

----

## [235] Exploring Stroke-Level Modifications for Scene Text Editing

**Authors**: *Yadong Qu, Qingfeng Tan, Hongtao Xie, Jianjun Xu, YuXin Wang, Yongdong Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25305](https://doi.org/10.1609/aaai.v37i2.25305)

**Abstract**:

Scene text editing (STE) aims to replace text with the desired one while preserving background and styles of the original text. However, due to the complicated background textures and various text styles, existing methods fall short in generating clear and legible edited text images. In this study, we attribute the poor editing performance to two problems: 1) Implicit decoupling structure. Previous methods of editing the whole image have to learn different translation rules of background and text regions simultaneously. 2) Domain gap. Due to the lack of edited real scene text images, the network can only be well trained on synthetic pairs and performs poorly on real-world images. To handle the above problems, we propose a novel network by MOdifying Scene Text image at strokE Level (MOSTEL). Firstly, we generate stroke guidance maps to explicitly indicate regions to be edited. Different from the implicit one by directly modifying all the pixels at image level, such explicit instructions filter out the distractions from background and guide the network to focus on editing rules of text regions. Secondly, we propose a Semi-supervised Hybrid Learning to train the network with both labeled synthetic images and unpaired real scene text images. Thus, the STE model is adapted to real-world datasets distributions. Moreover, two new datasets (Tamper-Syn2k and Tamper-Scene) are proposed to fill the blank of public evaluation datasets. Extensive experiments demonstrate that our MOSTEL outperforms previous methods both qualitatively and quantitatively. Datasets and code will be available at https://github.com/qqqyd/MOSTEL.

----

## [236] Unsupervised Deep Learning for Phase Retrieval via Teacher-Student Distillation

**Authors**: *Yuhui Quan, Zhile Chen, Tongyao Pang, Hui Ji*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25306](https://doi.org/10.1609/aaai.v37i2.25306)

**Abstract**:

Phase retrieval (PR) is a challenging nonlinear inverse problem in scientific imaging that involves reconstructing the phase of a signal from its intensity measurements. Recently, there has been an increasing interest in deep learning-based PR. Motivated by the challenge of collecting ground-truth (GT) images in many domains, this paper proposes a fully-unsupervised learning approach for PR, which trains an end-to-end deep model via a GT-free teacher-student online distillation framework. Specifically, a teacher model is trained using a self-expressive loss with noise resistance, while a student model is trained with a consistency loss on augmented data to exploit the teacher's dark knowledge. Additionally, we develop an enhanced unfolding  network for both the teacher and student models. Extensive experiments show that our proposed approach outperforms existing unsupervised PR methods with higher computational efficiency and performs competitively against supervised methods.

----

## [237] A Learnable Radial Basis Positional Embedding for Coordinate-MLPs

**Authors**: *Sameera Ramasinghe, Simon Lucey*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25307](https://doi.org/10.1609/aaai.v37i2.25307)

**Abstract**:

We propose a novel method to enhance the performance of coordinate-MLPs (also referred to as neural fields) by learning instance-specific positional embeddings. End-to-end optimization of positional embedding parameters along with network weights leads to poor generalization performance. Instead, we develop a generic framework to learn the positional embedding based on the classic graph-Laplacian regularization, which can implicitly balance the trade-off between memorization and generalization. This framework is then used to propose a novel positional embedding scheme, where the hyperparameters are learned per coordinate (i.e instance)  to deliver optimal performance. We show that the proposed embedding achieves better performance with higher stability compared to the well-established random Fourier features (RFF). Further, we demonstrate that the proposed embedding scheme yields stable gradients, enabling seamless integration into deep architectures as intermediate layers.

----

## [238] Action-Conditioned Generation of Bimanual Object Manipulation Sequences

**Authors**: *Haziq Razali, Yiannis Demiris*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25308](https://doi.org/10.1609/aaai.v37i2.25308)

**Abstract**:

The generation of bimanual object manipulation sequences given a semantic action label has broad applications in collaborative robots or augmented reality. This relatively new problem differs from existing works that generate whole-body motions without any object interaction as it now requires the model to additionally learn the spatio-temporal relationship that exists between the human joints and object motion given said label. To tackle this task, we leverage the varying degree each muscle or joint is involved during object manipulation. For instance, the wrists act as the prime movers for the objects while the finger joints are angled to provide a firm grip. The remaining body joints are the least involved in that they are positioned as naturally and comfortably as possible. We thus design an architecture that comprises 3 main components: (i) a graph recurrent network that generates the wrist and object motion, (ii) an attention-based recurrent network that estimates the required finger joint angles given the graph configuration, and (iii) a recurrent network that reconstructs the body pose given the locations of the wrist. We evaluate our approach on the KIT Motion Capture and KIT RGBD Bimanual Manipulation datasets and show improvements over a simplified approach that treats the entire body as a single entity, and existing whole-body-only methods.

----

## [239] Mean-Shifted Contrastive Loss for Anomaly Detection

**Authors**: *Tal Reiss, Yedid Hoshen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25309](https://doi.org/10.1609/aaai.v37i2.25309)

**Abstract**:

Deep anomaly detection methods learn representations that separate between normal and anomalous images. Although self-supervised representation learning is commonly used, small dataset sizes limit its effectiveness. It was previously shown that utilizing external, generic datasets (e.g. ImageNet classification) can significantly improve anomaly detection performance. One approach is outlier exposure, which fails when the external datasets do not resemble the anomalies. We take the approach of transferring representations pre-trained on external datasets for anomaly detection. Anomaly detection performance can be significantly improved by fine-tuning the pre-trained representations on the normal training images. In this paper, we first demonstrate and analyze that contrastive learning, the most popular self-supervised learning paradigm cannot be naively applied to pre-trained features.  The reason is that pre-trained feature initialization causes poor conditioning for standard contrastive objectives, resulting in bad optimization dynamics. Based on our analysis, we provide a modified contrastive objective, the Mean-Shifted Contrastive Loss. Our method is highly effective and achieves a new state-of-the-art anomaly detection performance including 98.6% ROC-AUC on the CIFAR-10 dataset.

----

## [240] Two Heads Are Better than One: Image-Point Cloud Network for Depth-Based 3D Hand Pose Estimation

**Authors**: *Pengfei Ren, Yuchen Chen, Jiachang Hao, Haifeng Sun, Qi Qi, Jingyu Wang, Jianxin Liao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25310](https://doi.org/10.1609/aaai.v37i2.25310)

**Abstract**:

Depth images and point clouds are the two most commonly used data representations for depth-based 3D hand pose estimation. Benefiting from the structuring of image data and the inherent inductive biases of the 2D Convolutional Neural Network (CNN), image-based methods are highly efficient and effective. However, treating the depth data as a 2D image inevitably ignores the 3D nature of depth data. Point cloud-based methods can better mine the 3D geometric structure of depth data. However, these methods suffer from the disorder and non-structure of point cloud data, which is computationally inefficient. In this paper, we propose an Image-Point cloud Network (IPNet) for accurate and robust 3D hand pose estimation. IPNet utilizes 2D CNN to extract visual representations in 2D image space and performs iterative correction in 3D point cloud space to exploit the 3D geometry information of depth data. In particular, we propose a sparse anchor-based "aggregation-interaction-propagation'' paradigm to enhance point cloud features and refine the hand pose, which reduces irregular data access. Furthermore, we introduce a 3D hand model to the iterative correction process, which significantly improves the robustness of IPNet to occlusion and depth holes. Experiments show that IPNet outperforms state-of-the-art methods on three challenging hand datasets.

----

## [241] MAGIC: Mask-Guided Image Synthesis by Inverting a Quasi-robust Classifier

**Authors**: *Mozhdeh Rouhsedaghat, Masoud Monajatipoor, C.-C. Jay Kuo, Iacopo Masi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25311](https://doi.org/10.1609/aaai.v37i2.25311)

**Abstract**:

We offer a method for one-shot mask-guided image synthesis that allows controlling manipulations of a single image by inverting a quasi-robust classifier equipped with strong regularizers. Our proposed method, entitled MAGIC, leverages structured gradients from a pre-trained quasi-robust classifier to better preserve the input semantics while preserving its classification accuracy, thereby guaranteeing credibility in the synthesis.
Unlike current methods that use complex primitives to supervise the process or use attention maps as a weak supervisory signal, MAGIC aggregates gradients over the input, driven by a guide binary mask that enforces a strong, spatial prior. MAGIC implements a series of manipulations with a single framework achieving shape and location control, intense non-rigid shape deformations, and copy/move operations in the presence of repeating objects and gives users firm control over the synthesis by requiring to simply specify binary guide masks. 
Our study and findings are supported by various qualitative comparisons with the state-of-the-art on the same images sampled from ImageNet and quantitative analysis using machine perception along with a user survey of 100+ participants that endorse our synthesis quality.

----

## [242] Domain Generalised Faster R-CNN

**Authors**: *Karthik Seemakurthy, Charles Fox, Erchan Aptoula, Petra Bosilj*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25312](https://doi.org/10.1609/aaai.v37i2.25312)

**Abstract**:

Domain generalisation (i.e. out-of-distribution generalisation)
is an open problem in machine learning, where the goal is
to train a model via one or more source domains, that will
generalise well to unknown target domains. While the topic
is attracting increasing interest, it has not been studied in
detail in the context of object detection. The established approaches
all operate under the covariate shift assumption,
where the conditional distributions are assumed to be approximately
equal across source domains. This is the first
paper to address domain generalisation in the context of object
detection, with a rigorous mathematical analysis of domain
shift, without the covariate shift assumption. We focus on
improving the generalisation ability of object detection by
proposing new regularisation terms to address the domain
shift that arises due to both classification and bounding box
regression. Also, we include an additional consistency regularisation
term to align the local and global level predictions.
The proposed approach is implemented as a Domain
Generalised Faster R-CNN and evaluated using four object
detection datasets which provide domain metadata (GWHD,
Cityscapes, BDD100K, Sim10K) where it exhibits a consistent
performance improvement over the baselines. All the
codes for replicating the results in this paper can be found at
https://github.com/karthikiitm87/domain-generalisation.git

----

## [243] MIDMs: Matching Interleaved Diffusion Models for Exemplar-Based Image Translation

**Authors**: *Junyoung Seo, Gyuseong Lee, Seokju Cho, Jiyoung Lee, Seungryong Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25313](https://doi.org/10.1609/aaai.v37i2.25313)

**Abstract**:

We present a novel method for exemplar-based image translation, called matching interleaved diffusion models (MIDMs). Most existing methods for this task were formulated as GAN-based matching-then-generation framework. However, in this framework, matching errors induced by the difficulty of semantic matching across cross-domain, e.g., sketch and photo, can be easily propagated to the generation step, which in turn leads to the degenerated results. Motivated by the recent success of diffusion models, overcoming the shortcomings of GANs, we incorporate the diffusion models to overcome these limitations. Specifically, we formulate a diffusion-based matching-and-generation framework that interleaves cross-domain matching and diffusion steps in the latent space by iteratively feeding the intermediate warp into the noising process and denoising it to generate a translated image. In addition, to improve the reliability of diffusion process, we design confidence-aware process using cycle-consistency to consider only confident regions during translation. Experimental results show that our MIDMs generate more plausible images than state-of-the-art methods.

----

## [244] JR2Net: Joint Monocular 3D Face Reconstruction and Reenactment

**Authors**: *Jiaxiang Shang, Yu Zeng, Xin Qiao, Xin Wang, Runze Zhang, Guangyuan Sun, Vishal Patel, Hongbo Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25314](https://doi.org/10.1609/aaai.v37i2.25314)

**Abstract**:

Face reenactment and reconstruction benefit various applications in self-media, VR, etc. Recent face reenactment methods use 2D facial landmarks to implicitly retarget facial expressions and poses from driving videos to source images,
while they suffer from pose and expression preservation issues for cross-identity scenarios, i.e., when the source and the driving subjects are different. Current self-supervised face reconstruction methods also demonstrate impressive results.
However, these methods do not handle large expressions well, since their training data lacks samples of large expressions, and 2D facial attributes are inaccurate on such samples. To mitigate the above problems, we propose to explore the inner connection between the two tasks, i.e., using face reconstruction to provide 
sufficient 3D information for reenactment, and synthesizing videos paired with captured face model parameters through face reenactment to enhance the
expression module of face reconstruction. In particular, we propose a novel cascade framework named JR2Net for Joint Face Reconstruction and Reenactment, which begins with the training of a coarse reconstruction network, followed by a 3D-aware face reenactment network based on the coarse reconstruction results. In the end, we train an expression tracking network based on our synthesized videos composed by image-face model parameter pairs. Such an expression tracking network can further enhance the coarse face reconstruction. Extensive experiments show that our JR2Net outperforms the state-of-the-art methods on several face reconstruction and reenactment benchmarks.

----

## [245] HVTSurv: Hierarchical Vision Transformer for Patient-Level Survival Prediction from Whole Slide Image

**Authors**: *Zhuchen Shao, Yang Chen, Hao Bian, Jian Zhang, Guojun Liu, Yongbing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25315](https://doi.org/10.1609/aaai.v37i2.25315)

**Abstract**:

Survival prediction based on whole slide images (WSIs) is a challenging task for patient-level multiple instance learning (MIL). Due to the vast amount of data for a patient (one or multiple gigapixels WSIs) and the irregularly shaped property of WSI, it is difficult to fully explore spatial, contextual, and hierarchical interaction in the patient-level bag. Many studies adopt random sampling pre-processing strategy and WSI-level aggregation models, which inevitably lose critical prognostic information in the patient-level bag. In this work, we propose a hierarchical vision Transformer framework named HVTSurv, which can encode the local-level relative spatial information, strengthen WSI-level context-aware communication, and establish patient-level hierarchical interaction. Firstly, we design a feature pre-processing strategy, including feature rearrangement and random window masking. Then, we devise three layers to progressively obtain patient-level representation, including a local-level interaction layer adopting Manhattan distance, a WSI-level interaction layer employing spatial shuffle, and a patient-level interaction layer using attention pooling. Moreover, the design of hierarchical network helps the model become more computationally efficient. Finally, we validate HVTSurv with 3,104 patients and 3,752 WSIs across 6 cancer types from The Cancer Genome Atlas (TCGA). The average C-Index is  2.50-11.30% higher than all the prior weakly supervised methods over 6 TCGA datasets. Ablation study and attention visualization further verify the superiority of the proposed HVTSurv. Implementation is available at: https://github.com/szc19990412/HVTSurv.

----

## [246] Channel Regeneration: Improving Channel Utilization for Compact DNNs

**Authors**: *Ankit Kumar Sharma, Hassan Foroosh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25316](https://doi.org/10.1609/aaai.v37i2.25316)

**Abstract**:

Overparameterized deep neural networks have redundant neurons that do not contribute to the network's accuracy. In this paper, we introduce a novel channel regeneration technique that reinvigorates these redundant channels by re-initializing its batch normalization scaling factor gamma. This re-initialization of BN gamma promotes regular weight updates during training. Furthermore, we show that channel regeneration encourages the channels to contribute equally to the learned representation and further boosts the generalization accuracy. We apply our technique at regular intervals of the training cycle to improve channel utilization. The solutions proposed in previous works either raise the total computational cost or increase the model complexity. Integrating the proposed channel regeneration technique into the training methodology of efficient architectures requires minimal effort and comes at no additional cost in size or memory. Extensive experiments on several image classification and semantic segmentation benchmarks demonstrate the effectiveness of applying the channel regeneration technique to compact architectures.

----

## [247] Adaptive Dynamic Filtering Network for Image Denoising

**Authors**: *Hao Shen, Zhong-Qiu Zhao, Wandi Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25317](https://doi.org/10.1609/aaai.v37i2.25317)

**Abstract**:

In image denoising networks, feature scaling is widely used to enlarge the receptive field size and reduce computational costs. This practice, however, also leads to the loss of high-frequency information and fails to consider within-scale characteristics. Recently, dynamic convolution has exhibited powerful capabilities in processing high-frequency information (e.g., edges, corners, textures), but previous works lack sufficient spatial contextual information in filter generation. To alleviate these issues, we propose to employ dynamic convolution to improve the learning of high-frequency and multi-scale features. Specifically, we design a spatially enhanced kernel generation (SEKG) module to improve dynamic convolution, enabling the learning of spatial context information with a very low computational complexity. Based on the SEKG module, we propose a dynamic convolution block (DCB) and a multi-scale dynamic convolution block (MDCB). The former enhances the high-frequency information via dynamic convolution and preserves low-frequency information via skip connections. The latter utilizes shared adaptive dynamic kernels and the idea of dilated convolution to achieve efficient multi-scale feature extraction. The proposed multi-dimension feature integration (MFI) mechanism further fuses the multi-scale features, providing precise and contextually enriched feature representations. Finally, we build an efficient denoising network with the proposed DCB and MDCB, named ADFNet. It achieves better performance with low computational complexity on real-world and synthetic Gaussian noisy datasets. The source code is available at https://github.com/it-hao/ADFNet.

----

## [248] Edge Structure Learning via Low Rank Residuals for Robust Image Classification

**Authors**: *Xiangjun Shen, Stanley Ebhohimhen Abhadiomhen, Yang Yang, Zhifeng Liu, Sirui Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25318](https://doi.org/10.1609/aaai.v37i2.25318)

**Abstract**:

Traditional low-rank methods overlook residuals as corruptions, but we discovered that low-rank residuals actually keep image edges together with corrupt components. Therefore, filtering out such structural information could hamper the discriminative details in images, especially in heavy corruptions. In order to address this limitation, this paper proposes a novel method named ESL-LRR, which preserves image edges by finding image projections from low-rank residuals. Specifically, our approach is built in a manifold learning framework where residuals are regarded as another view of image data. Edge preserved image projections are then pursued using a dynamic affinity graph regularization to capture the more accurate similarity between residuals while suppressing the influence of corrupt ones. With this adaptive approach, the proposed method can also find image intrinsic low-rank representation, and much discriminative edge preserved projections. As a result, a new classification strategy is introduced, aligning both modalities to enhance accuracy. Experiments are conducted on several benchmark image datasets, including MNIST, LFW, and COIL100. The results show that the proposed method has clear advantages over compared state-of-the-art (SOTA) methods, such as Low-Rank Embedding (LRE), Low-Rank Preserving Projection via Graph Regularized Reconstruction (LRPP_GRR), and Feature Selective Projection (FSP) with more than 2% improvement, particularly in corrupted cases.

----

## [249] Memory-Oriented Structural Pruning for Efficient Image Restoration

**Authors**: *Xiangsheng Shi, Xuefei Ning, Lidong Guo, Tianchen Zhao, Enshu Liu, Yi Cai, Yuhan Dong, Huazhong Yang, Yu Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25319](https://doi.org/10.1609/aaai.v37i2.25319)

**Abstract**:

Deep learning (DL) based methods have significantly pushed forward the state-of-the-art for image restoration (IR) task. Nevertheless, DL-based IR models are highly computation- and memory-intensive. The surging demands for processing higher-resolution images and multi-task paralleling in practical mobile usage further add to their computation and memory burdens. In this paper, we reveal the overlooked memory redundancy of the IR models and propose a Memory-Oriented Structural Pruning (MOSP) method. To properly compress the long-range skip connections (a major source of the memory burden), we introduce a compactor module onto each skip connection to decouple the pruning of the skip connections and the main branch. MOSP progressively prunes the original model layers and the compactors to cut down the peak memory while maintaining high IR quality. Experiments on real image denoising, image super-resolution and low-light image enhancement show that MOSP can yield models with higher memory efficiency while better preserving performance compared with baseline pruning methods.

----

## [250] YOLOV: Making Still Image Object Detectors Great at Video Object Detection

**Authors**: *Yuheng Shi, Naiyan Wang, Xiaojie Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25320](https://doi.org/10.1609/aaai.v37i2.25320)

**Abstract**:

Video object detection (VID) is challenging because of the high variation of object appearance as well as the diverse deterioration in some frames. On the positive side, the detection in a certain frame of a video, compared with that in a still image, can draw support from other frames. Hence, how to aggregate features across different frames is pivotal to VID problem. Most of existing aggregation algorithms are customized for two-stage detectors. However, these detectors are usually computationally expensive due to their two-stage nature. This work proposes a simple yet effective strategy to address the above concerns, which costs marginal overheads with significant gains in accuracy. Concretely, different from traditional two-stage pipeline, we select important regions after the one-stage detection to avoid processing massive low-quality candidates. Besides, we evaluate the relationship between a target frame and reference frames to guide the aggregation. We conduct extensive experiments and ablation studies to verify the efficacy of our design, and reveal its superiority over other state-of-the-art VID approaches in both effectiveness and efficiency. Our YOLOX-based model can achieve promising performance (e.g., 87.5% AP50 at over 30 FPS on the ImageNet VID dataset on a single 2080Ti GPU), making it attractive for large-scale or real-time applications. The implementation is simple, we have made the demo codes and models available at https://github.com/YuHengsss/YOLOV.

----

## [251] FeedFormer: Revisiting Transformer Decoder for Efficient Semantic Segmentation

**Authors**: *Jae-hun Shim, Hyunwoo Yu, Kyeongbo Kong, Suk-Ju Kang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25321](https://doi.org/10.1609/aaai.v37i2.25321)

**Abstract**:

With the success of Vision Transformer (ViT) in image classification, its variants have yielded great success in many downstream vision tasks. Among those, the semantic segmentation task has also benefited greatly from the advance of ViT variants. However, most studies of the transformer for semantic segmentation only focus on designing efficient transformer encoders, rarely giving attention to designing the decoder. Several studies make attempts in using the transformer decoder as the segmentation decoder with class-wise learnable query. Instead, we aim to directly use the encoder features as the queries. This paper proposes the Feature Enhancing Decoder transFormer (FeedFormer) that enhances structural information using the transformer decoder. Our goal is to decode the high-level encoder features using the lowest-level encoder feature. We do this by formulating high-level features as queries, and the lowest-level feature as the key and value. This enhances the high-level features by collecting the structural information from the lowest-level feature. Additionally, we use a simple reformation trick of pushing the encoder blocks to take the place of the existing self-attention module of the decoder to improve efficiency. We show the superiority of our decoder with various light-weight transformer-based decoders on popular semantic segmentation datasets. Despite the minute computation, our model has achieved state-of-the-art performance in the performance computation trade-off. Our model FeedFormer-B0 surpasses SegFormer-B0 with 1.8% higher mIoU and 7.1% less computation on ADE20K, and 1.7% higher mIoU and 14.4% less computation on Cityscapes, respectively. Code will be released at: https://github.com/jhshim1995/FeedFormer.

----

## [252] Task-Specific Scene Structure Representations

**Authors**: *Jisu Shin, SeungHyun Shin, Hae-Gon Jeon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25322](https://doi.org/10.1609/aaai.v37i2.25322)

**Abstract**:

Understanding the informative structures of scenes is essential for low-level vision tasks. Unfortunately, it is difficult to obtain a concrete visual definition of the informative structures because influences of visual features are task-specific. In this paper, we propose a single general neural network architecture for extracting task-specific structure guidance for scenes.
To do this, we first analyze traditional spectral clustering methods, which computes a set of eigenvectors to model a segmented graph forming small compact structures on image domains. We then unfold the traditional graph-partitioning problem into a learnable network, named Scene Structure Guidance Network (SSGNet), to represent the task-specific informative structures. The SSGNet yields a set of coefficients of eigenvectors that produces explicit feature representations of image structures. In addition, our SSGNet is light-weight (56K parameters), and can be used as a plug-and-play module for off-the-shelf architectures. We optimize the SSGNet without any supervision by proposing two novel training losses that enforce task-specific scene structure generation during training. Our main contribution is to show that such a simple network can achieve state-of-the-art results for several low-level vision applications including joint upsampling and image denoising. We also demonstrate that our SSGNet generalizes well on unseen datasets, compared to existing methods which use structural embedding frameworks. Our source codes are available at https://github.com/jsshin98/SSGNet.

----

## [253] Diversified and Realistic 3D Augmentation via Iterative Construction, Random Placement, and HPR Occlusion

**Authors**: *Jungwook Shin, Jaeill Kim, Kyungeun Lee, Hyunghun Cho, Wonjong Rhee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25323](https://doi.org/10.1609/aaai.v37i2.25323)

**Abstract**:

In autonomous driving, data augmentation is commonly used for improving 3D object detection. The most basic methods include insertion of copied objects and rotation and scaling of the entire training frame. Numerous variants have been developed as well. The existing methods, however, are considerably limited when compared to the variety of the real world possibilities. In this work, we develop a diversified and realistic augmentation method that can flexibly construct a whole-body object, freely locate and rotate the object, and apply self-occlusion and external-occlusion accordingly. To improve the diversity of the whole-body object construction, we develop an iterative method that stochastically combines multiple objects observed from the real world into a single object. Unlike the existing augmentation methods, the constructed objects can be randomly located and rotated in the training frame because proper occlusions can be reflected to the whole-body objects in the final step. Finally, proper self-occlusion at each local object level and external-occlusion at the global frame level are applied using the Hidden Point Removal (HPR) algorithm that is computationally efficient. HPR is also used for adaptively controlling the point density of each object according to the object's distance from the LiDAR. Experiment results show that the proposed DR.CPO algorithm is data-efficient and model-agnostic without incurring any computational overhead. Also, DR.CPO can improve mAP performance by 2.08% when compared to the best 3D detection result known for KITTI dataset.

----

## [254] SHUNIT: Style Harmonization for Unpaired Image-to-Image Translation

**Authors**: *Seokbeom Song, Suhyeon Lee, Hongje Seong, Kyoungwon Min, Euntai Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25324](https://doi.org/10.1609/aaai.v37i2.25324)

**Abstract**:

We propose a novel solution for unpaired image-to-image (I2I) translation. To translate complex images with a wide range of objects to a different domain, recent approaches often use the object annotations to perform per-class source-to-target style mapping. However, there remains a point for us to exploit in the I2I. An object in each class consists of multiple components, and all the sub-object components have different characteristics. For example, a car in CAR class consists of a car body, tires, windows and head and tail lamps, etc., and they should be handled separately for realistic I2I translation. The simplest solution to the problem will be to use more detailed annotations with sub-object component annotations than the simple object annotations, but it is not possible. The key idea of this paper is to bypass the sub-object component annotations by leveraging the original style of the input image because the original style will include the information about the characteristics of the sub-object components. Specifically, for each pixel, we use not only the per-class style gap between the source and target domains but also the pixel’s original style to determine the target style of a pixel. To this end, we present Style Harmonization for unpaired I2I translation (SHUNIT). Our SHUNIT generates a new style by harmonizing the target domain style retrieved from a class memory and an original source image style. Instead of direct source-to-target style mapping, we aim for source and target styles harmonization. We validate our method with extensive experiments and achieve state-of-the-art performance on the latest benchmark sets. The source code is available online: https://github.com/bluejangbaljang/SHUNIT.

----

## [255] Siamese-Discriminant Deep Reinforcement Learning for Solving Jigsaw Puzzles with Large Eroded Gaps

**Authors**: *Xingke Song, Jiahuan Jin, Chenglin Yao, Shihe Wang, Jianfeng Ren, Ruibin Bai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25325](https://doi.org/10.1609/aaai.v37i2.25325)

**Abstract**:

Jigsaw puzzle solving has recently become an emerging research area. The developed techniques have been widely used in applications beyond puzzle solving. This paper focuses on solving Jigsaw Puzzles with Large Eroded Gaps (JPwLEG). We formulate the puzzle reassembly as a combinatorial optimization problem and propose a Siamese-Discriminant Deep Reinforcement Learning (SD2RL) to solve it. A Deep Q-network (DQN) is designed to visually understand the puzzles, which consists of two sets of Siamese Discriminant Networks, one set to perceive the pairwise relations between vertical neighbors and another set for horizontal neighbors. The proposed DQN considers not only the evidence from the incumbent fragment but also the support from its four neighbors. The DQN is trained using replay experience with carefully designed rewards to guide the search for a sequence of fragment swaps to reach the correct puzzle solution. Two JPwLEG datasets are constructed to evaluate the proposed method, and the experimental results show that the proposed SD2RL significantly outperforms state-of-the-art methods.

----

## [256] CLIPVG: Text-Guided Image Manipulation Using Differentiable Vector Graphics

**Authors**: *Yiren Song, Xuning Shao, Kang Chen, Weidong Zhang, Zhongliang Jing, Minzhe Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25326](https://doi.org/10.1609/aaai.v37i2.25326)

**Abstract**:

Considerable progress has recently been made in leveraging CLIP (Contrastive Language-Image Pre-Training) models for text-guided image manipulation. However, all existing works rely on additional generative models to ensure the quality of results, because CLIP alone cannot provide enough guidance information for fine-scale pixel-level changes. In this paper, we introduce CLIPVG, a text-guided image manipulation framework using differentiable vector graphics,  which is also the first CLIP-based general image manipulation framework that does not require any additional generative models. We demonstrate that CLIPVG can not only achieve state-of-art performance in both semantic correctness and synthesis quality, but also is flexible enough to support various applications far beyond the capability of all existing methods.

----

## [257] Compact Transformer Tracker with Correlative Masked Modeling

**Authors**: *Zikai Song, Run Luo, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25327](https://doi.org/10.1609/aaai.v37i2.25327)

**Abstract**:

Transformer framework has been showing superior performances in visual object tracking for its great strength in information aggregation across the template and search image with the well-known attention mechanism. Most recent advances focus on exploring attention mechanism variants for better information aggregation. We find these schemes are equivalent to or even just a subset of the basic self-attention mechanism. In this paper, we prove that the vanilla self-attention structure is sufficient for information aggregation, and structural adaption is unnecessary. The key is not the attention structure,  but how to extract the discriminative feature for tracking and enhance the communication between the target and search image. Based on this finding, we adopt the basic vision transformer (ViT) architecture as our main tracker and concatenate the template and search image for feature embedding. To guide the encoder to capture the invariant feature for tracking, we attach a lightweight correlative masked decoder which reconstructs the original template and search image from the corresponding masked tokens. The correlative masked decoder serves as a plugin for the compact transformer tracker and is skipped in inference. Our compact tracker uses the most simple structure which only consists of a ViT backbone and a box head, and can run at 40 fps. Extensive experiments show the proposed compact transform tracker outperforms existing approaches, including advanced attention variants, and demonstrates the sufficiency of self-attention in tracking tasks. Our method achieves state-of-the-art performance on five challenging datasets, along with the VOT2020, UAV123, LaSOT, TrackingNet, and GOT-10k benchmarks. Our project is available at https://github.com/HUSTDML/CTTrack.

----

## [258] Text-DIAE: A Self-Supervised Degradation Invariant Autoencoder for Text Recognition and Document Enhancement

**Authors**: *Mohamed Ali Souibgui, Sanket Biswas, Andrés Mafla, Ali Furkan Biten, Alicia Fornés, Yousri Kessentini, Josep Lladós, Lluís Gómez, Dimosthenis Karatzas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25328](https://doi.org/10.1609/aaai.v37i2.25328)

**Abstract**:

In this paper, we propose  a Text-Degradation Invariant Auto Encoder (Text-DIAE), a self-supervised model designed to tackle two tasks, text recognition (handwritten or scene-text) and document image enhancement. We start by employing a transformer-based architecture that incorporates three pretext tasks as learning objectives to be optimized during pre-training without the usage of labelled data. Each of the pretext objectives is specifically tailored for the final downstream tasks. We conduct several ablation experiments that confirm the design choice of the selected pretext tasks. Importantly, the proposed model does not exhibit limitations of previous state-of-the-art methods based on contrastive losses, while at the same time requiring substantially fewer data samples to converge. Finally, we demonstrate that our method surpasses the state-of-the-art in existing supervised and self-supervised settings in handwritten and scene text recognition and document image enhancement. Our code and trained models will be made publicly available at https://github.com/dali92002/SSL-OCR

----

## [259] PUPS: Point Cloud Unified Panoptic Segmentation

**Authors**: *Shihao Su, Jianyun Xu, Huanyu Wang, Zhenwei Miao, Xin Zhan, Dayang Hao, Xi Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25329](https://doi.org/10.1609/aaai.v37i2.25329)

**Abstract**:

Point cloud panoptic segmentation is a challenging task that seeks a holistic solution for both semantic and instance segmentation to predict groupings of coherent points. Previous approaches treat semantic and instance segmentation as surrogate tasks, and they either use clustering methods or bounding boxes to gather instance groupings with costly computation and hand-craft designs in the instance segmentation task. In this paper, we propose a simple but effective point cloud unified panoptic segmentation (PUPS) framework, which use a set of point-level classifiers to directly predict semantic and instance groupings in an end-to-end manner. To realize PUPS, we introduce bipartite matching to our training pipeline so that our classifiers are able to exclusively predict groupings of instances, getting rid of hand-crafted designs, e.g. anchors and Non-Maximum Suppression (NMS). In order to achieve better grouping results, we utilize a transformer decoder to iteratively refine the point classifiers and develop a context-aware CutMix augmentation to overcome the class imbalance problem. As a result, PUPS achieves 1st place on the leader board of SemanticKITTI panoptic segmentation task and state-of-the-art results on nuScenes.

----

## [260] Efficient Edge-Preserving Multi-View Stereo Network for Depth Estimation

**Authors**: *Wanjuan Su, Wenbing Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25330](https://doi.org/10.1609/aaai.v37i2.25330)

**Abstract**:

Over the years, learning-based multi-view stereo methods have achieved great success based on their coarse-to-fine depth estimation frameworks.  However, 3D CNN-based cost volume regularization inevitably leads to over-smoothing problems at object boundaries due to its smooth properties. Moreover, discrete and sparse depth hypothesis sampling exacerbates the difficulty in recovering the depth of thin structures and object boundaries. To this end, we present an Efficient edge-Preserving multi-view stereo Network (EPNet) for practical depth estimation. To keep delicate estimation at details, a Hierarchical Edge-Preserving Residual learning (HEPR) module is proposed to progressively rectify the upsampling errors and help refine multi-scale depth estimation. After that, a Cross-view Photometric Consistency (CPC) is proposed to enhance the gradient flow for detailed structures, which further boosts the estimation accuracy. Last, we design a lightweight cascade framework and inject the above two strategies into it to achieve better efficiency and performance trade-offs. Extensive experiments show that our method achieves state-of-the-art performance with fast inference speed and low memory usage. Notably, our method tops the first place on challenging Tanks and Temples advanced dataset and ETH3D high-res benchmark among all published learning-based methods. Code will be available at https://github.com/susuwj/EPNet.

----

## [261] Referring Expression Comprehension Using Language Adaptive Inference

**Authors**: *Wei Su, Peihan Miao, Huanzhang Dou, Yongjian Fu, Xi Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25331](https://doi.org/10.1609/aaai.v37i2.25331)

**Abstract**:

Different from universal object detection, referring expression comprehension (REC) aims to locate specific objects referred to by natural language expressions. The expression provides high-level concepts of relevant visual and contextual patterns, which vary significantly with different expressions and account for only a few of those encoded in the REC model. This leads us to a question: do we really need the entire network with a fixed structure for various referring expressions? Ideally, given an expression, only expression-relevant components of the REC model are required. These components should be small in number as each expression only contains very few visual and contextual clues. This paper explores the adaptation between expressions and REC models for dynamic inference. Concretely, we propose a neat yet efficient framework named Language Adaptive Dynamic Subnets (LADS), which can extract language-adaptive subnets from the REC model conditioned on the referring expressions. By using the compact subnet, the inference can be more economical and efficient. Extensive experiments on RefCOCO, RefCOCO+, RefCOCOg, and Referit show that the proposed method achieves faster inference speed and higher accuracy against state-of-the-art approaches.

----

## [262] Rethinking Data Augmentation for Single-Source Domain Generalization in Medical Image Segmentation

**Authors**: *Zixian Su, Kai Yao, Xi Yang, Kaizhu Huang, Qiufeng Wang, Jie Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25332](https://doi.org/10.1609/aaai.v37i2.25332)

**Abstract**:

Single-source domain generalization (SDG) in medical image segmentation is a challenging yet essential task as domain shifts are quite common among clinical image datasets. Previous attempts most conduct global-only/random augmentation.  Their augmented samples are usually insufficient in diversity and informativeness, thus failing to cover the possible target domain distribution. In this paper, we rethink the data augmentation strategy for SDG in medical image segmentation. Motivated by the class-level representation invariance and style mutability of medical images, we hypothesize that  unseen target data can be sampled from a linear combination of C (the class number) random variables, where each variable follows a location-scale distribution at the class level. Accordingly, data augmented can be readily made by sampling the random variables through a general form. On the empirical front, we implement such strategy with constrained Bezier transformation on both  global and  local (i.e. class-level) regions, which can largely increase the augmentation diversity.  A Saliency-balancing Fusion mechanism is further proposed to enrich the informativeness by engaging the gradient information, guiding augmentation with proper orientation and magnitude. As an important contribution, we prove theoretically that our proposed augmentation can lead to an upper bound of the generalization risk on the unseen  target domain, thus confirming our hypothesis. Combining the two strategies, our Saliency-balancing Location-scale Augmentation (SLAug) exceeds the state-of-the-art works by a large margin in two challenging SDG tasks. Code is available at https://github.com/Kaiseem/SLAug.

----

## [263] Hybrid Pixel-Unshuffled Network for Lightweight Image Super-resolution

**Authors**: *Bin Sun, Yulun Zhang, Songyao Jiang, Yun Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25333](https://doi.org/10.1609/aaai.v37i2.25333)

**Abstract**:

Convolutional neural network (CNN) has achieved great success on image super-resolution (SR). However, most deep CNN-based SR models take massive computations to obtain high performance. Downsampling features for multi-resolution fusion is an efficient and effective way to improve the performance of visual recognition. Still, it is counter-intuitive in the SR task, which needs to project a low-resolution input to high-resolution. In this paper, we propose a novel Hybrid Pixel-Unshuffled Network (HPUN) by introducing an efficient and effective downsampling module into the SR task. The network contains pixel-unshuffled downsampling and Self-Residual Depthwise Separable Convolutions. Specifically, we utilize pixel-unshuffle operation to downsample the input features and use grouped convolution to reduce the channels. Besides, we enhance the depthwise convolution's performance by adding the input feature to its output. The comparison findings demonstrate that, with fewer parameters and computational costs, our HPUN achieves and surpasses the state-of-the-art performance on SISR. All results are provided in the github https://github.com/Sun1992/HPUN.

----

## [264] Learning Event-Relevant Factors for Video Anomaly Detection

**Authors**: *Che Sun, Chenrui Shi, Yunde Jia, Yuwei Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25334](https://doi.org/10.1609/aaai.v37i2.25334)

**Abstract**:

Most video anomaly detection methods discriminate events that deviate from normal patterns as anomalies. However, these methods are prone to interferences from event-irrelevant factors,  such as background textures and object scale variations,  incurring an increased false detection rate. In this paper, we propose to explicitly learn event-relevant factors to eliminate the interferences from event-irrelevant factors on anomaly predictions.  To this end, we introduce a causal generative model to separate the event-relevant factors and event-irrelevant ones in videos, and learn the prototypes of event-relevant factors in a memory augmentation module.   We design a causal objective function to optimize the causal generative model and develop a counterfactual learning strategy to guide anomaly predictions, which increases the influence of the event-relevant factors. The extensive experiments show the effectiveness of our method for video anomaly detection.

----

## [265] Superpoint Transformer for 3D Scene Instance Segmentation

**Authors**: *Jiahao Sun, Chunmei Qing, Junpeng Tan, Xiangmin Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25335](https://doi.org/10.1609/aaai.v37i2.25335)

**Abstract**:

Most existing methods realize 3D instance segmentation by extending those models used for 3D object detection or 3D semantic segmentation. However, these non-straightforward methods suffer from two drawbacks: 1) Imprecise bounding boxes or unsatisfactory semantic predictions limit the performance of the overall 3D instance segmentation framework. 2) Existing method requires a time-consuming intermediate step of aggregation. To address these issues, this paper proposes a novel end-to-end 3D instance segmentation method based on Superpoint Transformer, named as SPFormer. It groups potential features from point clouds into superpoints, and directly predicts instances through query vectors without relying on the results of object detection or semantic segmentation. The key step in this framework is a novel query decoder with transformers that can capture the instance information through the superpoint cross-attention mechanism and generate the superpoint masks of the instances. Through bipartite matching based on superpoint masks, SPFormer can implement the network training without the intermediate aggregation step, which accelerates the network. Extensive experiments on ScanNetv2 and S3DIS benchmarks verify that our method is concise yet efficient. Notably, SPFormer exceeds compared state-of-the-art methods by 4.3% on ScanNetv2 hidden test set in terms of mAP and keeps fast inference speed (247ms per frame) simultaneously. Code is available at https://github.com/sunjiahao1999/SPFormer.

----

## [266] Asynchronous Event Processing with Local-Shift Graph Convolutional Network

**Authors**: *Linhui Sun, Yifan Zhang, Jian Cheng, Hanqing Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25336](https://doi.org/10.1609/aaai.v37i2.25336)

**Abstract**:

Event cameras are bio-inspired sensors that produce sparse and asynchronous event streams instead of frame-based images at a high-rate. Recent works utilizing graph convolutional networks (GCNs) have achieved remarkable performance in recognition tasks, which model event stream as spatio-temporal graph. However, the computational mechanism of graph convolution introduces redundant computation when aggregating neighbor features, which limits the low-latency nature of the events. And they perform a synchronous inference process, which can not achieve a fast response to the asynchronous event signals. This paper proposes a local-shift graph convolutional network (LSNet), which utilizes a novel local-shift operation equipped with a local spatio-temporal attention component to achieve efficient and adaptive aggregation of neighbor features. To improve the efficiency of pooling operation in feature extraction, we design a node-importance based parallel pooling method (NIPooling) for sparse and low-latency event data. Based on the calculated importance of each node, NIPooling can efficiently obtain uniform sampling results in parallel, which retains the diversity of event streams. Furthermore, for achieving a fast response to asynchronous event signals, an asynchronous event processing procedure is proposed to restrict the network nodes which need to recompute activations only to those affected by the new arrival event. Experimental results show that the computational cost can be reduced by nearly 9 times through using local-shift operation and the proposed asynchronous procedure can further improve the inference efficiency, while achieving state-of-the-art performance on gesture recognition and object recognition.

----

## [267] DENet: Disentangled Embedding Network for Visible Watermark Removal

**Authors**: *Ruizhou Sun, Yukun Su, Qingyao Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25337](https://doi.org/10.1609/aaai.v37i2.25337)

**Abstract**:

Adding visible watermark into image is a common copyright protection method of medias. Meanwhile, public research on watermark removal can be utilized as an adversarial technology to help the further development of watermarking. 
Existing watermark removal methods mainly adopt multi-task learning networks, which locate the watermark and restore the background simultaneously. However, these approaches view the task as an image-to-image reconstruction problem, where they only impose supervision after the final output, making the high-level semantic features shared between different tasks. 
To this end, inspired by the two-stage coarse-refinement network, we propose a novel contrastive learning mechanism to disentangle the high-level embedding semantic information of the images and watermarks, driving the respective network branch more oriented.
Specifically, the proposed mechanism is leveraged for watermark image decomposition, which aims to decouple the clean image and watermark hints in the high-level embedding space. This can guarantee the learning representation of the restored image enjoy more task-specific cues.
In addition, we introduce a self-attention-based enhancement module, which promotes the network's ability to capture semantic information among different regions, leading to further improvement on the contrastive learning mechanism. 
To validate the effectiveness of our proposed method, extensive experiments are conducted on different challenging benchmarks. Experimental evaluations show that our approach can achieve state-of-the-art performance and yield high-quality images. The code is available at: https://github.com/lianchengmingjue/DENet.

----

## [268] Deep Manifold Attack on Point Clouds via Parameter Plane Stretching

**Authors**: *Keke Tang, Jianpeng Wu, Weilong Peng, Yawen Shi, Peng Song, Zhaoquan Gu, Zhihong Tian, Wenping Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25338](https://doi.org/10.1609/aaai.v37i2.25338)

**Abstract**:

Adversarial attack on point clouds plays a vital role in evaluating and improving the adversarial robustness of 3D deep learning models. Current attack methods are mainly applied by point perturbation in a non-manifold manner. In this paper, we formulate a novel manifold attack, which deforms the underlying 2-manifold surfaces via parameter plane stretching to generate adversarial point clouds. First, we represent the mapping between the parameter plane and underlying surface using generative-based networks. Second, the stretching is learned in the 2D parameter domain such that the generated 3D point cloud fools a pretrained classifier with minimal geometric distortion. Extensive experiments show that adversarial point clouds generated by manifold attack are smooth, undefendable and transferable, and outperform those samples generated by the state-of-the-art non-manifold ones.

----

## [269] Fair Generative Models via Transfer Learning

**Authors**: *Christopher T. H. Teo, Milad Abdollahzadeh, Ngai-Man Cheung*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25339](https://doi.org/10.1609/aaai.v37i2.25339)

**Abstract**:

This work addresses fair generative models. Dataset biases have been a major cause of unfairness in deep generative models. Previous work had proposed to augment large, biased datasets with small, unbiased reference datasets. Under this setup, a weakly-supervised approach has been proposed, which achieves state-of-the-art quality and fairness in generated samples. In our work, based on this setup, we propose a simple yet effective approach. Specifically, first, we propose fairTL, a transfer learning approach to learn fair generative models. Under fairTL, we pre-train the generative model with the available large, biased datasets and subsequently adapt the model using the small, unbiased reference dataset. We find that our fairTL can learn expressive sample generation during pre-training, thanks to the large (biased) dataset. This knowledge is then transferred to the target model during adaptation, which also learns to capture the underlying fair distribution of the small reference dataset. Second, we propose fairTL++, where we introduce two additional innovations to improve upon fairTL: (i) multiple feedback and (ii) Linear-Probing followed by Fine-Tuning (LP-FT). Taking one step further, we consider an alternative, challenging setup when only a pre-trained (potentially biased) model is available but the dataset that was used to pre-train the model is inaccessible. We demonstrate that our proposed fairTL and fairTL++ remain very effective under this setup. We note that previous work requires access to the large, biased datasets and is incapable of handling this more challenging setup. Extensive experiments show that fairTL and fairTL++ achieve state-of-the-art in both quality and fairness of generated samples. The code and additional resources can be found at bearwithchris.github.io/fairTL/.

----

## [270] Learning Context-Aware Classifier for Semantic Segmentation

**Authors**: *Zhuotao Tian, Jiequan Cui, Li Jiang, Xiaojuan Qi, Xin Lai, Yixin Chen, Shu Liu, Jiaya Jia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25340](https://doi.org/10.1609/aaai.v37i2.25340)

**Abstract**:

Semantic segmentation is still a challenging task for parsing diverse contexts in different scenes, thus the fixed classifier might not be able to well address varying feature distributions during testing.  Different from the mainstream literature where the efficacy of strong backbones and effective decoder heads has been well studied, in this paper, additional contextual hints are instead exploited via learning a context-aware classifier whose content is data-conditioned, decently adapting to different latent distributions. Since only the classifier is dynamically altered, our method is model-agnostic and can be easily applied to generic segmentation models. Notably, with only negligible additional parameters and +2\% inference time, decent performance gain has been achieved on both small and large models with challenging benchmarks, manifesting substantial practical merits brought by our simple yet effective method. The implementation is available at https://github.com/tianzhuotao/CAC.

----

## [271] TopicFM: Robust and Interpretable Topic-Assisted Feature Matching

**Authors**: *Khang Truong Giang, Soohwan Song, Sungho Jo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25341](https://doi.org/10.1609/aaai.v37i2.25341)

**Abstract**:

This study addresses an image-matching problem in challenging cases, such as large scene variations or textureless scenes. To gain robustness to such situations, most previous studies have attempted to encode the global contexts of a scene via graph neural networks or transformers. However, these contexts do not explicitly represent high-level contextual information, such as structural shapes or semantic instances; therefore, the encoded features are still not sufficiently discriminative in challenging scenes. We propose a novel image-matching method that applies a topic-modeling strategy to encode high-level contexts in images. The proposed method trains latent semantic instances called topics. It explicitly models an image as a multinomial distribution of topics, and then performs probabilistic feature matching. This approach improves the robustness of matching by focusing on the same semantic areas between the images. In addition, the inferred topics provide interpretability for matching the results, making our method explainable. Extensive experiments on outdoor and indoor datasets show that our method outperforms other state-of-the-art methods, particularly in challenging cases.

----

## [272] Learning Fractals by Gradient Descent

**Authors**: *Cheng-Hao Tu, Hong-You Chen, David Carlyn, Wei-Lun Chao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25342](https://doi.org/10.1609/aaai.v37i2.25342)

**Abstract**:

Fractals are geometric shapes that can display complex and self-similar patterns found in nature (e.g., clouds and plants). Recent works in visual recognition have leveraged this property to create random fractal images for model pre-training. In this paper, we study the inverse problem --- given a target image (not necessarily a fractal), we aim to generate a fractal image that looks like it. We propose a novel approach that learns the parameters underlying a fractal image via gradient descent. We show that our approach can find fractal parameters of high visual quality and be compatible with different loss functions, opening up several potentials, e.g., learning fractals for downstream tasks, scientific understanding, etc.

----

## [273] Leveraging Weighted Cross-Graph Attention for Visual and Semantic Enhanced Video Captioning Network

**Authors**: *Deepali Verma, Arya Haldar, Tanima Dutta*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25343](https://doi.org/10.1609/aaai.v37i2.25343)

**Abstract**:

Video captioning has become a broad and interesting research area. Attention-based encoder-decoder methods are extensively used for caption generation. However, these methods mostly utilize the visual attentive feature to highlight the video regions while overlooked the semantic features of the available captions. These semantic features contain significant information that helps to generate highly informative human description-like captions. Therefore, we propose a novel visual and semantic enhanced video captioning network, named as VSVCap, that efficiently utilizes multiple ground truth captions. We aim to generate captions that are visually and semantically enhanced by exploiting both video and text modalities. To achieve this, we propose a fine-grained cross-graph attention mechanism that captures detailed graph embedding correspondence between visual graphs and textual knowledge graphs. We have performed node-level matching and structure-level reasoning between the weighted regional graph and knowledge graph. The proposed network achieves promising results on three benchmark datasets, i.e., YouTube2Text, MSR-VTT, and VATEX. The experimental results show that our network accurately captures all key objects, relationships, and semantically enhanced events of a video to generate human annotation-like captions.

----

## [274] Doodle to Object: Practical Zero-Shot Sketch-Based 3D Shape Retrieval

**Authors**: *Bingrui Wang, Yuan Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25344](https://doi.org/10.1609/aaai.v37i2.25344)

**Abstract**:

Zero-shot (ZS) sketch-based three-dimensional (3D) shape retrieval (SBSR) is challenging due to the abstraction of sketches, cross-domain discrepancies between two-dimensional sketches and 3D shapes, and ZS-driven semantic knowledge transference from seen to unseen categories. Extant SBSR datasets suffer from lack of data, and no current SBSR methods consider ZS scenarios. In this paper, we contribute a new Doodle2Object (D2O) dataset consisting of 8,992 3D shapes and over 7M sketches spanning 50 categories. Then, we propose a novel prototype contrastive learning (PCL) method that effectively extracts features from different domains and adapts them to unseen categories. Specifically, our PCL method combines the ideas of contrastive and cluster-based prototype learning, and several randomly selected prototypes of different classes are assigned to each sample. By comparing these prototypes, a given sample can be moved closer to the same semantic class of samples while moving away from negative ones. Extensive experiments on two common SBSR benchmarks and our D2O dataset demonstrate the efficacy of the proposed PCL method for ZS-SBSR. Resource is available at https://github.com/yigohw/doodle2object.

----

## [275] Controlling Class Layout for Deep Ordinal Classification via Constrained Proxies Learning

**Authors**: *Cong Wang, Zhiwei Jiang, Yafeng Yin, Zifeng Cheng, Shiping Ge, Qing Gu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25345](https://doi.org/10.1609/aaai.v37i2.25345)

**Abstract**:

For deep ordinal classification, learning a well-structured feature space specific to ordinal classification is helpful to properly capture the ordinal nature among classes. Intuitively, when Euclidean distance metric is used, an ideal ordinal layout in feature space would be that the sample clusters are arranged in class order along a straight line in space. However, enforcing samples to conform to a specific layout in the feature space is a challenging problem. To address this problem, in this paper, we propose a novel Constrained Proxies Learning (CPL) method, which can learn a proxy for each ordinal class and then adjusts the global layout of classes by constraining these proxies. Specifically, we propose two kinds of strategies: hard layout constraint and soft layout constraint. The hard layout constraint is realized by directly controlling the generation of proxies to force them to be placed in a strict linear layout or semicircular layout (i.e., two instantiations of strict ordinal layout). The soft layout constraint is realized by constraining that the proxy layout should always produce unimodal proxy-to-proxies similarity distribution for each proxy (i.e., to be a relaxed ordinal layout). Experiments show that the proposed CPL method outperforms previous deep ordinal classification methods under the same setting of feature extractor.

----

## [276] Dual Memory Aggregation Network for Event-Based Object Detection with Learnable Representation

**Authors**: *Dongsheng Wang, Xu Jia, Yang Zhang, Xinyu Zhang, Yaoyuan Wang, Ziyang Zhang, Dong Wang, Huchuan Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25346](https://doi.org/10.1609/aaai.v37i2.25346)

**Abstract**:

Event-based cameras are bio-inspired sensors that capture brightness change of every pixel in an asynchronous manner. Compared with frame-based sensors, event cameras have microsecond-level latency and high dynamic range, hence showing great potential for object detection under high-speed motion and poor illumination conditions. Due to sparsity and asynchronism nature with event streams, most of existing approaches resort to hand-crafted methods to convert event data into 2D grid representation. However, they are sub-optimal in aggregating information from event stream for object detection. In this work, we propose to learn an event representation optimized for event-based object detection. Specifically, event streams are divided into grids in the x-y-t coordinates for both positive and negative polarity, producing a set of pillars as 3D tensor representation. To fully exploit information with event streams to detect objects, a dual-memory aggregation network (DMANet) is proposed to leverage both long and short memory along event streams to aggregate effective information for object detection. Long memory is encoded in the hidden state of adaptive convLSTMs while short memory is modeled by computing spatial-temporal correlation between event pillars at neighboring time intervals. Extensive experiments on the recently released event-based automotive detection dataset demonstrate the effectiveness of the proposed method.

----

## [277] Text to Point Cloud Localization with Relation-Enhanced Transformer

**Authors**: *Guangzhi Wang, Hehe Fan, Mohan S. Kankanhalli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25347](https://doi.org/10.1609/aaai.v37i2.25347)

**Abstract**:

Automatically localizing a position based on a few natural language instructions is essential for future robots to communicate and collaborate with humans. To approach this goal, we focus on a text-to-point-cloud cross-modal localization
problem. Given a textual query, it aims to identify the described location from city-scale point clouds. The task involves two challenges. 1) In city-scale point clouds, similar ambient instances may exist in several locations. Searching each location in a huge point cloud with only instances as guidance may lead to less discriminative signals and incorrect results. 2) In textual descriptions, the hints are provided separately. In this case, the relations among those hints are not
explicitly described, leaving the difficulties of learning relations to the agent itself. To alleviate the two challenges, we propose a unified Relation-Enhanced Transformer (RET) to improve representation discriminability for both point cloud
and nature language queries. The core of the proposed RET is a novel Relation-enhanced Self-Attention (RSA) mechanism, which explicitly encodes instance (hint)-wise relations for the two modalities. Moreover, we propose a fine-grained cross-modal matching method to further refine the location predictions in a subsequent instance-hint matching stage. Experimental results on the KITTI360Pose dataset demonstrate that our approach surpasses the previous state-of-the-art method by large margins.

----

## [278] UCoL: Unsupervised Learning of Discriminative Facial Representations via Uncertainty-Aware Contrast

**Authors**: *Hao Wang, Min Li, Yangyang Song, Youjian Zhang, Liying Chi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25348](https://doi.org/10.1609/aaai.v37i2.25348)

**Abstract**:

This paper presents Uncertainty-aware Contrastive Learning (UCoL): a fully unsupervised framework for discriminative facial representation learning. Our UCoL is built upon a momentum contrastive network, referred to as Dual-path Momentum Network. Specifically, two flows of pairwise contrastive training are conducted simultaneously:  one is formed with intra-instance self augmentation, and the other is to identify positive pairs collected by online pairwise prediction. We introduce a novel uncertainty-aware consistency K-nearest neighbors algorithm to generate predicted positive pairs, which enables efficient discriminative learning from large-scale open-world unlabeled data. Experiments show that UCoL significantly improves the baselines of unsupervised models and performs on par with the semi-supervised and supervised face representation learning methods.

----

## [279] Calibrated Teacher for Sparsely Annotated Object Detection

**Authors**: *Haohan Wang, Liang Liu, Boshen Zhang, Jiangning Zhang, Wuhao Zhang, Zhenye Gan, Yabiao Wang, Chengjie Wang, Haoqian Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25349](https://doi.org/10.1609/aaai.v37i2.25349)

**Abstract**:

Fully supervised object detection requires training images in which all instances are annotated. This is actually impractical due to the high labor and time costs and the unavoidable missing annotations. As a result, the incomplete annotation in each image could provide misleading supervision and harm the training. Recent works on sparsely annotated object detection alleviate this problem by generating pseudo labels for the missing annotations. Such a mechanism is sensitive to the threshold of the pseudo label score. However, the effective threshold is different in different training stages and among different object detectors. Therefore, the current methods with fixed thresholds have sub-optimal performance, and are difficult to be applied to other detectors. In order to resolve this obstacle, we propose a Calibrated Teacher, of which the confidence estimation of the prediction is well calibrated to match its real precision. In this way, different detectors in different training stages would share a similar distribution of the output confidence, so that multiple detectors could share the same fixed threshold and achieve better performance. Furthermore, we present a simple but effective Focal IoU Weight (FIoU) for the classification loss. FIoU aims at reducing the loss weight of false negative samples caused by the missing annotation, and thus works as the complement of the teacher-student paradigm. Extensive experiments show that our methods set new state-of-the-art under all different sparse settings in COCO. Code will be available at https://github.com/Whileherham/CalibratedTeacher.

----

## [280] Towards Real-Time Panoptic Narrative Grounding by an End-to-End Grounding Network

**Authors**: *Haowei Wang, Jiayi Ji, Yiyi Zhou, Yongjian Wu, Xiaoshuai Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25350](https://doi.org/10.1609/aaai.v37i2.25350)

**Abstract**:

Panoptic Narrative Grounding (PNG) is an emerging cross-modal grounding task, which locates the target regions of an image corresponding to the text description.
Existing approaches for PNG are mainly based on a two-stage paradigm, which is computationally expensive. In this paper, we propose a one-stage network for real-time PNG, termed End-to-End Panoptic Narrative Grounding network (EPNG), which directly generates masks for referents. Specifically, we propose two innovative designs, i.e., Locality-Perceptive Attention (LPA) and a bidirectional Semantic Alignment Loss (SAL), to properly handle the many-to-many relationship between textual expressions and visual objects. LPA embeds the local spatial priors into attention modeling, i.e., a pixel may belong to multiple masks at different scales, thereby improving segmentation. To help understand the complex semantic relationships, SAL proposes a bidirectional contrastive objective to regularize the semantic consistency inter modalities. Extensive experiments on the PNG benchmark dataset demonstrate the effectiveness and efficiency of our method. Compared to the single-stage baseline, our method achieves a significant improvement of up to 9.4% accuracy. More importantly, our EPNG is 10 times faster than the two-stage model. Meanwhile, the generalization ability of EPNG is also validated by zero-shot experiments on other grounding tasks. The source codes and trained models for all our experiments are publicly available at https://github.com/Mr-Neko/EPNG.git.

----

## [281] LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise

**Authors**: *He Wang, Lin Wan, He Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25351](https://doi.org/10.1609/aaai.v37i2.25351)

**Abstract**:

Pixel-wise prediction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remarkable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust saliency (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected conditional random field (CRF). Different from ROSA that rely on various pre- and post-processings, this paper proposes a light-weight Learnable Noise (LeNo) to defend adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also on clean images, which contributes stronger robustness for SOD. Our code is available at https://github.com/ssecv/LeNo.

----

## [282] Defending Black-Box Skeleton-Based Human Activity Classifiers

**Authors**: *He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25352](https://doi.org/10.1609/aaai.v37i2.25352)

**Abstract**:

Skeletal motions have been heavily relied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks. Appendix and code are available.

----

## [283] Exploring CLIP for Assessing the Look and Feel of Images

**Authors**: *Jianyi Wang, Kelvin C. K. Chan, Chen Change Loy*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25353](https://doi.org/10.1609/aaai.v37i2.25353)

**Abstract**:

Measuring the perception of visual content is a long-standing problem in computer vision. Many mathematical models have been developed to evaluate the look or quality of an image. Despite the effectiveness of such tools in quantifying degradations such as noise and blurriness levels, such quantification is loosely coupled with human language. When it comes to more abstract perception about the feel of visual content, existing methods can only rely on supervised models that are explicitly trained with labeled data collected via laborious user study. In this paper, we go beyond the conventional paradigms by exploring the rich visual language prior encapsulated in Contrastive Language-Image Pre-training (CLIP) models for assessing both the quality perception (look) and abstract perception (feel) of images without explicit task-specific training. In particular, we discuss effective prompt designs and show an effective prompt pairing strategy to harness the prior. We also provide extensive experiments on controlled datasets and Image Quality Assessment (IQA) benchmarks. Our results show that CLIP captures meaningful priors that generalize well to different perceptual assessments.

----

## [284] Robust Video Portrait Reenactment via Personalized Representation Quantization

**Authors**: *Kaisiyuan Wang, Changcheng Liang, Hang Zhou, Jiaxiang Tang, Qianyi Wu, Dongliang He, Zhibin Hong, Jingtuo Liu, Errui Ding, Ziwei Liu, Jingdong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25354](https://doi.org/10.1609/aaai.v37i2.25354)

**Abstract**:

While progress has been made in the field of portrait reenactment, the problem of how to produce high-fidelity and robust videos remains. Recent studies normally find it challenging to handle rarely seen target poses due to the limitation of source data. This paper proposes the Video Portrait via Non-local Quantization Modeling (VPNQ) framework, which produces pose- and disturbance-robust reenactable video portraits. Our key insight is to learn position-invariant quantized local patch representations and build a mapping between simple driving signals and local textures with non-local spatial-temporal modeling. Specifically, instead of learning a universal quantized codebook, we identify that a personalized one can be trained to preserve desired position-invariant local details better. Then, a simple representation of projected landmarks can be used as sufficient driving signals to avoid 3D rendering. Following, we employ a carefully designed Spatio-Temporal Transformer to predict reasonable and temporally consistent quantized tokens from the driving signal. The predicted codes can be decoded back to robust and high-quality videos. Comprehensive experiments have been conducted to validate the effectiveness of our approach.

----

## [285] De-biased Teacher: Rethinking IoU Matching for Semi-supervised Object Detection

**Authors**: *Kuo Wang, Jingyu Zhuang, Guanbin Li, Chaowei Fang, Lechao Cheng, Liang Lin, Fan Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25355](https://doi.org/10.1609/aaai.v37i2.25355)

**Abstract**:

Most of the recent research in semi-supervised object detection follows the pseudo-labeling paradigm evolved from the semi-supervised image classification task. However, the training paradigm of the two-stage object detector inevitably makes the pseudo-label learning process for unlabeled images full of bias. Specifically, the IoU matching scheme used for selecting and labeling candidate boxes is based on the assumption that the matching source~(ground truth) is accurate enough in terms of the number of objects, object position and object category. Obviously, pseudo-labels generated for unlabeled images cannot satisfy such a strong assumption, which makes the produced training proposals extremely unreliable and thus severely spoil the follow-up training. To de-bias the training proposals generated by the pseudo-label-based IoU matching, we propose a general framework -- De-biased Teacher, which abandons both the IoU matching and pseudo labeling processes by directly generating favorable training proposals for consistency regularization between the weak/strong augmented image pairs. Moreover, a distribution-based refinement scheme is designed to eliminate the scattered class predictions of significantly low values for higher efficiency. Extensive experiments demonstrate that the proposed De-biased Teacher consistently outperforms other state-of-the-art methods on the MS-COCO and PASCAL VOC benchmarks. Source codes are available at https://github.com/wkfdb/De-biased-Teracher.

----

## [286] Learning to Generate an Unbiased Scene Graph by Using Attribute-Guided Predicate Features

**Authors**: *Lei Wang, Zejian Yuan, Badong Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25356](https://doi.org/10.1609/aaai.v37i2.25356)

**Abstract**:

Scene Graph Generation (SGG) aims to capture the semantic information in an image and build a structured representation, which facilitates downstream tasks. The current challenge in SGG is to tackle the biased predictions caused by the long-tailed distribution of predicates. Since multiple predicates in SGG are coupled in an image, existing data re-balancing methods cannot completely balance the head and tail predicates. In this work, a decoupled learning framework is proposed for unbiased scene graph generation by using attribute-guided predicate features to construct a balanced training set. Specifically, the predicate recognition is decoupled into Predicate Feature Representation Learning (PFRL) and predicate classifier training with a class-balanced predicate feature set, which is constructed by our proposed Attribute-guided Predicate Feature Generation (A-PFG) model. In the A-PFG model, we first define the class labels of  and corresponding visual feature as attributes to describe a predicate. Then the predicate feature and the attribute embedding are mapped into a shared hidden space by a dual Variational Auto-encoder (VAE), and finally the synthetic predicate features are forced to learn the contextual information in the attributes via cross reconstruction and distribution alignment. To demonstrate the effectiveness of our proposed method, our decoupled learning framework and A-PFG model are applied to various SGG models. The empirical results show that our method is substantially improved on all benchmarks and achieves new state-of-the-art performance for unbiased scene graph generation. Our code is available at https://github.com/wanglei0618/A-PFG.

----

## [287] Alignment-Enriched Tuning for Patch-Level Pre-trained Document Image Models

**Authors**: *Lei Wang, Jiabang He, Xing Xu, Ning Liu, Hui Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25357](https://doi.org/10.1609/aaai.v37i2.25357)

**Abstract**:

Alignment between image and text has shown promising improvements on patch-level pre-trained document image models. However, investigating more effective or finer-grained alignment techniques during pre-training requires a large amount of computation cost and time. Thus, a question naturally arises: Could we fine-tune the pre-trained models adaptive to downstream tasks with alignment objectives and achieve comparable or better performance? In this paper, we propose a new model architecture with alignment-enriched tuning (dubbed AETNet) upon pre-trained document image models, to adapt downstream tasks with the joint task-specific supervised and alignment-aware contrastive objective. Specifically, we introduce an extra visual transformer as the alignment-ware image encoder and an extra text transformer as the alignment-ware text encoder before multimodal fusion. We consider alignment in the following three aspects: 1) document-level alignment by leveraging the cross-modal and intra-modal contrastive loss; 2) global-local alignment for modeling localized and structural information in document images; and 3) local-level alignment for more accurate patch-level information. Experiments on various downstream tasks show that AETNet can achieve state-of-the-art performance on various downstream tasks. Notably, AETNet consistently outperforms state-of-the-art pre-trained models, such as LayoutLMv3 with fine-tuning techniques, on three different downstream tasks. Code is available at https://github.com/MAEHCM/AET.

----

## [288] Flora: Dual-Frequency LOss-Compensated ReAl-Time Monocular 3D Video Reconstruction

**Authors**: *Likang Wang, Yue Gong, Qirui Wang, Kaixuan Zhou, Lei Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25358](https://doi.org/10.1609/aaai.v37i2.25358)

**Abstract**:

In this work, we propose a real-time monocular 3D video reconstruction approach named Flora for reconstructing delicate and complete 3D scenes from RGB video sequences in an end-to-end manner. Specifically, we introduce a novel method with two main contributions. Firstly, the proposed feature aggregation module retains both color and reliability in a dual-frequency form. Secondly, the loss compensation module solves missing structure by correcting losses for falsely pruned voxels. The dual-frequency feature aggregation module enhances reconstruction quality in both precision and recall, and the loss compensation module benefits the recall. Notably, both proposed contributions achieve great results with negligible inferencing overhead. Our state-of-the-art experimental results on real-world datasets demonstrate Flora's leading performance in both effectiveness and efficiency. The code is available at https://github.com/NoOneUST/Flora.

----

## [289] Efficient Image Captioning for Edge Devices

**Authors**: *Ning Wang, Jiangrong Xie, Hang Luo, Qinglin Cheng, Jihao Wu, Mingbo Jia, Linlin Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25359](https://doi.org/10.1609/aaai.v37i2.25359)

**Abstract**:

Recent years have witnessed the rapid progress of image captioning. However, the demands for large memory storage and heavy computational burden prevent these captioning models from being deployed on mobile devices. The main obstacles lie in the heavyweight visual feature extractors (i.e., object detectors) and complicated cross-modal fusion networks. To this end, we propose LightCap, a lightweight image captioner for resource-limited devices. The core design is built on the recent CLIP model for efficient image captioning. To be specific, on the one hand, we leverage the CLIP model to extract the compact grid features without relying on the time-consuming object detectors. On the other hand, we transfer the image-text retrieval design of CLIP to image captioning scenarios by devising a novel visual concept extractor and a cross-modal modulator. We further optimize the cross-modal fusion model and parallel prediction heads via sequential and ensemble distillations. With the carefully designed architecture, our model merely contains 40M parameters, saving the model size by more than 75% and the FLOPs by more than 98% in comparison with the current state-of-the-art methods. In spite of the low capacity, our model still exhibits state-of-the-art performance on prevalent datasets, e.g., 136.6 CIDEr on COCO Karpathy test split. Testing on the smartphone with only a single CPU, the proposed LightCap exhibits a fast inference speed of 188ms per image, which is ready for practical applications.

----

## [290] Controllable Image Captioning via Prompting

**Authors**: *Ning Wang, Jiahao Xie, Jihao Wu, Mingbo Jia, Linlin Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25360](https://doi.org/10.1609/aaai.v37i2.25360)

**Abstract**:

Despite the remarkable progress of image captioning, existing captioners typically lack the controllable capability to generate desired image captions, e.g., describing the image in a rough or detailed manner, in a factual or emotional view, etc. In this paper, we show that a unified model is qualified to perform well in diverse domains and freely switch among multiple styles. Such a controllable capability is achieved by embedding the prompt learning into the image captioning framework. To be specific, we design a set of prompts to fine-tune the pre-trained image captioner. These prompts allow the model to absorb stylized data from different domains for joint training, without performance degradation in each domain. Furthermore, we optimize the prompts with learnable vectors in the continuous word embedding space, avoiding the heuristic prompt engineering and meanwhile exhibiting superior performance. In the inference stage, our model is able to generate desired stylized captions by choosing the corresponding prompts. Extensive experiments verify the controllable capability of the proposed method. Notably, we achieve outstanding performance on two diverse image captioning benchmarks including COCO Karpathy split and TextCaps using a unified model.

----

## [291] ECO-3D: Equivariant Contrastive Learning for Pre-training on Perturbed 3D Point Cloud

**Authors**: *Ruibin Wang, Xianghua Ying, Bowei Xing, Jinfa Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25361](https://doi.org/10.1609/aaai.v37i2.25361)

**Abstract**:

In this work, we investigate contrastive learning on perturbed point clouds and find that the contrasting process may widen the domain gap caused by random perturbations, making the pre-trained network fail to generalize on testing data. To this end, we propose the Equivariant COntrastive framework which closes the domain gap before contrasting, further introduces the equivariance property, and enables pre-training networks under more perturbation types to obtain meaningful features. Specifically, to close the domain gap, a pre-trained VAE is adopted to convert perturbed point clouds into less perturbed point embedding of similar domains and separated perturbation embedding. The contrastive pairs can then be generated by mixing the point embedding with different perturbation embedding. Moreover, to pursue the equivariance property, a Vector Quantizer is adopted during VAE training, discretizing the perturbation embedding into one-hot tokens which indicate the perturbation labels. By correctly predicting the perturbation labels from the perturbed point cloud, the property of equivariance can be encouraged in the learned features. Experiments on synthesized and real-world perturbed datasets show that ECO-3D outperforms most existing pre-training strategies under various downstream tasks, achieving SOTA performance for lots of perturbations.

----

## [292] Global-Local Characteristic Excited Cross-Modal Attacks from Images to Videos

**Authors**: *Ruikui Wang, Yuanfang Guo, Yunhong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25362](https://doi.org/10.1609/aaai.v37i2.25362)

**Abstract**:

The transferability of adversarial examples is the key property in practical black-box scenarios. Currently, numerous methods improve the transferability across different models trained on the same modality of data. The investigation of generating video adversarial examples with imagebased substitute models to attack the target video models, i.e., cross-modal transferability of adversarial examples, is rarely explored. A few works on cross-modal transferability directly apply image attack methods for each frame and no factors especial for video data are considered, which limits the cross-modal transferability of adversarial examples. In this paper, we propose an effective cross-modal attack method which considers both the global and local characteristics of video data. Firstly, from the global perspective, we introduce inter-frame interaction into attack process to induce more diverse and stronger gradients rather than perturb each frame separately. Secondly, from the local perspective, we disrupt the inherently local correlation of frames within a video, which prevents black-box video model from capturing valuable temporal clues. Extensive experiments on the UCF-101 and Kinetics-400 validate the proposed method significantly improves cross-modal transferability and even surpasses strong baseline using video models as substitute model. Our source codes are available at https://github.com/lwmming/Cross-Modal-Attack.

----

## [293] Fine-Grained Retrieval Prompt Tuning

**Authors**: *Shijie Wang, Jianlong Chang, Zhihui Wang, Haojie Li, Wanli Ouyang, Qi Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25363](https://doi.org/10.1609/aaai.v37i2.25363)

**Abstract**:

Fine-grained object retrieval aims to learn discriminative representation to retrieve visually similar objects. However, existing top-performing works usually impose pairwise similarities on the semantic embedding spaces or design a localization sub-network to continually fine-tune the entire model in limited data scenarios, thus resulting in convergence to suboptimal solutions. In this paper, we develop Fine-grained Retrieval Prompt Tuning (FRPT), which steers a frozen pre-trained model to perform the fine-grained retrieval task from the perspectives of sample prompting and feature adaptation. Specifically, FRPT only needs to learn fewer parameters in the prompt and adaptation instead of fine-tuning the entire model, thus solving the issue of convergence to suboptimal solutions caused by fine-tuning the entire model. Technically, a discriminative perturbation prompt (DPP) is introduced and deemed as a sample prompting process, which amplifies and even exaggerates some discriminative elements contributing to category prediction via a content-aware inhomogeneous sampling operation. In this way, DPP can make the fine-grained retrieval task aided by the perturbation prompts close to the solved task during the original pre-training. Thereby, it preserves the generalization and discrimination of representation extracted from input samples. Besides, a category-specific awareness head is proposed and regarded as feature adaptation, which removes the species discrepancies in features extracted by the pre-trained model using category-guided instance normalization. And thus, it makes the optimized features only include the discrepancies among subcategories. Extensive experiments demonstrate that our FRPT with fewer learnable parameters achieves the state-of-the-art performance on three widely-used fine-grained datasets.

----

## [294] Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method

**Authors**: *Tao Wang, Kaihao Zhang, Tianrun Shen, Wenhan Luo, Björn Stenger, Tong Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25364](https://doi.org/10.1609/aaai.v37i3.25364)

**Abstract**:

As the quality of optical sensors improves, there is a need for processing large-scale images. In particular, the ability of devices to capture ultra-high definition (UHD) images and video places new demands on the image processing pipeline. In this paper, we consider the task of low-light image enhancement (LLIE) and introduce a large-scale database consisting of images at 4K and 8K resolution. We conduct systematic benchmarking studies and provide a comparison of current LLIE algorithms. As a second contribution, we introduce LLFormer, a transformer-based low-light enhancement method. The core components of LLFormer are the axis-based multi-head self-attention and cross-layer attention fusion block, which significantly reduces the linear complexity. Extensive experiments on the new dataset and existing public datasets show that LLFormer outperforms state-of-the-art methods. We also show that employing existing LLIE methods trained on our benchmark as a pre-processing step significantly improves the performance of downstream tasks, e.g., face detection in low-light conditions. The source code and pre-trained models are available at https://github.com/TaoWangzj/LLFormer.

----

## [295] 3D Assembly Completion

**Authors**: *Weihao Wang, Rufeng Zhang, Mingyu You, Hongjun Zhou, Bin He*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25365](https://doi.org/10.1609/aaai.v37i3.25365)

**Abstract**:

Automatic assembly is a promising research topic in 3D computer vision and robotics. Existing works focus on generating assembly (e.g., IKEA furniture) from scratch with a set of parts, namely 3D part assembly. In practice, there are higher demands for the robot to take over and finish an incomplete assembly (e.g., a half-assembled IKEA furniture) with an off-the-shelf toolkit, especially in human-robot and multi-agent collaborations. Compared to 3D part assembly, it is more complicated in nature and remains unexplored yet. The robot must understand the incomplete structure, infer what parts are missing, single out the correct parts from the toolkit and finally, assemble them with appropriate poses to finish the incomplete assembly. Geometrically similar parts in the toolkit can interfere, and this problem will be exacerbated with more missing parts. To tackle this issue, we propose a novel task called 3D assembly completion. Given an incomplete assembly, it aims to find its missing parts from a toolkit and predict the 6-DoF poses to make the assembly complete. To this end, we propose FiT, a framework for Finishing the incomplete 3D assembly with Transformer. We employ the encoder to model the incomplete assembly into memories. Candidate parts interact with memories in a memory-query paradigm for final candidate classification and pose prediction. Bipartite part matching and symmetric transformation consistency are embedded to refine the completion. For reasonable evaluation and further reference, we design two standard toolkits of different difficulty, containing different compositions of candidate parts. We conduct extensive comparisons with several baseline methods and ablation studies, demonstrating the effectiveness of the proposed method.

----

## [296] A Benchmark and Asymmetrical-Similarity Learning for Practical Image Copy Detection

**Authors**: *Wenhao Wang, Yifan Sun, Yi Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25366](https://doi.org/10.1609/aaai.v37i3.25366)

**Abstract**:

Image copy detection (ICD) aims to determine whether a query image is an edited copy of any image from a reference set. Currently, there are very limited public benchmarks for ICD, while all overlook a critical challenge in real-world applications, i.e., the distraction from hard negative queries. Specifically, some queries are not edited copies but are inherently similar to some reference images. These hard negative queries are easily false recognized as edited copies, significantly compromising the ICD accuracy. This observation motivates us to build the first ICD benchmark featuring this characteristic. Based on existing ICD datasets, this paper constructs a new dataset by additionally adding 100,000 and 24, 252 hard negative pairs into the training and test set, respectively. Moreover, this paper further reveals a unique difficulty for solving the hard negative problem in ICD, i.e., there is a fundamental conflict between current metric learning and ICD. This conflict is: the metric learning adopts symmetric distance while the edited copy is an asymmetric (unidirectional) process, e.g., a partial crop is close to its holistic reference image and is an edited copy, while the latter cannot be the edited copy of the former (in spite the distance is equally small). This insight results in an Asymmetrical-Similarity Learning (ASL) method, which allows the similarity in two directions (the query ↔ the reference image) to be different from each other. Experimental results show that ASL outperforms state-of-the-art methods by a clear margin, confirming that solving the symmetric-asymmetric conflict is critical for ICD. The NDEC dataset and code are available at https://github.com/WangWenhao0716/ASL.

----

## [297] Revisiting Unsupervised Local Descriptor Learning

**Authors**: *Wufan Wang, Lei Zhang, Hua Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25367](https://doi.org/10.1609/aaai.v37i3.25367)

**Abstract**:

Constructing accurate training tuples is crucial for unsupervised local descriptor learning, yet challenging due to the absence of patch labels. The state-of-the-art approach constructs tuples with heuristic rules, which struggle to precisely depict real-world patch transformations, in spite of enabling fast model convergence. A possible solution to alleviate the problem is the clustering-based approach, which can capture realistic patch variations and learn more accurate class decision boundaries, but suffers from slow model convergence. This paper presents HybridDesc, an unsupervised approach that learns powerful local descriptor models with fast convergence speed by combining the rule-based and clustering-based approaches to construct training tuples. In addition, HybridDesc also contributes two concrete enhancing mechanisms: (1) a Differentiable Hyperparameter Search (DHS) strategy to find the optimal hyperparameter setting of the rule-based approach so as to provide accurate prior for the clustering-based approach, (2) an On-Demand Clustering (ODC) method to reduce the clustering overhead of the clustering-based approach without eroding its advantage. Extensive experimental results show that HybridDesc can efficiently learn local descriptors that surpass existing unsupervised local descriptors and even rival competitive supervised ones.

----

## [298] Crafting Monocular Cues and Velocity Guidance for Self-Supervised Multi-Frame Depth Learning

**Authors**: *Xiaofeng Wang, Zheng Zhu, Guan Huang, Xu Chi, Yun Ye, Ziwei Chen, Xingang Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25368](https://doi.org/10.1609/aaai.v37i3.25368)

**Abstract**:

Self-supervised monocular methods can efficiently learn depth information of weakly textured surfaces or reflective objects. However, the depth accuracy is limited due to the inherent ambiguity in monocular geometric modeling. In contrast, multi-frame depth estimation methods improve depth accuracy thanks to the success of Multi-View Stereo (MVS), which directly makes use of geometric constraints. Unfortunately, MVS often suffers from texture-less regions, non-Lambertian surfaces, and moving objects, especially in real-world video sequences without known camera motion and depth supervision. Therefore, we propose MOVEDepth, which exploits the MOnocular cues and VElocity guidance to improve multi-frame Depth learning. Unlike existing methods that enforce consistency between MVS depth and monocular depth, MOVEDepth boosts multi-frame depth learning by directly addressing the inherent problems of MVS. The key of our approach is to utilize monocular depth as a geometric priority to construct MVS cost volume, and adjust depth candidates of cost volume under the guidance of predicted camera velocity. We further fuse monocular depth and MVS depth by learning uncertainty in the cost volume, which results in a robust depth estimation against ambiguity in multi-view geometry. Extensive experiments show MOVEDepth achieves state-of-the-art performance: Compared with Monodepth2 and PackNet, our method relatively improves the depth accuracy by 20% and 19.8% on the KITTI benchmark. MOVEDepth also generalizes to the more challenging DDAD benchmark, relatively outperforming ManyDepth by 7.2%. The code is available at https://github.com/JeffWang987/MOVEDepth.

----

## [299] Learning Continuous Depth Representation via Geometric Spatial Aggregator

**Authors**: *Xiaohang Wang, Xuanhong Chen, Bingbing Ni, Zhengyan Tong, Hang Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25369](https://doi.org/10.1609/aaai.v37i3.25369)

**Abstract**:

Depth map super-resolution (DSR) has been a fundamental task for 3D computer vision. While arbitrary scale DSR is a more realistic setting in this scenario, previous approaches predominantly suffer from the issue of inefficient real-numbered scale upsampling. To explicitly address this issue, we propose a novel continuous depth representation for DSR. The heart of this representation is our proposed Geometric Spatial Aggregator (GSA), which exploits a distance field modulated by arbitrarily upsampled target gridding, through which the geometric information is explicitly introduced into feature aggregation and target generation. Furthermore, bricking with GSA, we present a transformer-style backbone named GeoDSR, which possesses a principled way to construct the functional mapping between local coordinates and the high-resolution output results, empowering our model with the advantage of arbitrary shape transformation ready to help diverse zooming demand. Extensive experimental results on standard depth map benchmarks, e.g., NYU v2, have demonstrated that the proposed framework achieves significant restoration gain in arbitrary scale depth map super-resolution compared with the prior art. Our codes are available at https://github.com/nana01219/GeoDSR.

----

## [300] SSDA3D: Semi-supervised Domain Adaptation for 3D Object Detection from Point Cloud

**Authors**: *Yan Wang, Junbo Yin, Wei Li, Pascal Frossard, Ruigang Yang, Jianbing Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25370](https://doi.org/10.1609/aaai.v37i3.25370)

**Abstract**:

LiDAR-based 3D object detection is an indispensable task in advanced autonomous driving systems. Though impressive detection results have been achieved by superior 3D detectors, they suffer from significant performance degeneration when facing unseen domains, such as different LiDAR configurations, different cities, and weather conditions. The mainstream approaches tend to solve these challenges by leveraging unsupervised domain adaptation (UDA) techniques. However, these UDA solutions just yield unsatisfactory 3D detection results when there is a severe domain shift, e.g., from Waymo (64-beam) to nuScenes (32-beam). To address this, we present a novel Semi-Supervised Domain Adaptation method for 3D object detection (SSDA3D), where only a few labeled target data is available, yet can significantly improve the adaptation performance. In particular, our SSDA3D includes an Inter-domain Adaptation stage and an Intra-domain Generalization stage. In the first stage, an Inter-domain Point-CutMix module is presented to efficiently align the point cloud distribution across domains. The Point-CutMix generates mixed samples of an intermediate domain, thus encouraging to learn domain-invariant knowledge. Then, in the second stage, we further enhance the model for better generalization on the unlabeled target set. This is achieved by exploring Intra-domain Point-MixUp in semi-supervised learning, which essentially regularizes the pseudo label distribution. Experiments from Waymo to nuScenes show that, with only 10% labeled target data, our SSDA3D can surpass the fully-supervised oracle model with 100% target label. Our code is available at  https://github.com/yinjunbo/SSDA3D.

----

## [301] High-Resolution GAN Inversion for Degraded Images in Large Diverse Datasets

**Authors**: *Yanbo Wang, Chuming Lin, Donghao Luo, Ying Tai, Zhizhong Zhang, Yuan Xie*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25371](https://doi.org/10.1609/aaai.v37i3.25371)

**Abstract**:

The last decades are marked by massive and diverse image data, which shows increasingly high resolution and quality. However, some images we obtained may be corrupted, affecting the perception and the application of downstream tasks. A generic method for generating a high-quality image from the degraded one is in demand. In this paper, we present a novel GAN inversion framework that utilizes the powerful generative ability of StyleGAN-XL for this problem. To ease the inversion challenge with StyleGAN-XL, Clustering \& Regularize Inversion (CRI) is proposed. Specifically, the latent space is firstly divided into finer-grained sub-spaces by clustering. Instead of initializing the inversion with the average latent vector, we approximate a centroid latent vector from the clusters, which generates an image close to the input image. Then, an offset with a regularization term is introduced to keep the inverted latent vector within a certain range. We validate our CRI scheme on multiple restoration tasks (i.e., inpainting, colorization, and super-resolution) of complex natural images, and show preferable quantitative and qualitative results. We further demonstrate our technique is robust in terms of data and different GAN models. To our best knowledge, we are the first to adopt StyleGAN-XL for generating high-quality natural images from diverse degraded inputs. Code is available at https://github.com/Booooooooooo/CRI.

----

## [302] GAN Prior Based Null-Space Learning for Consistent Super-resolution

**Authors**: *Yinhuai Wang, Yujie Hu, Jiwen Yu, Jian Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25372](https://doi.org/10.1609/aaai.v37i3.25372)

**Abstract**:

Consistency and realness have always been the two critical issues of image super-resolution. While the realness has been dramatically improved with the use of GAN prior, the state-of-the-art methods still suffer inconsistencies in local structures and colors (e.g., tooth and eyes). In this paper, we show that these inconsistencies can be analytically eliminated by learning only the null-space component while fixing the range-space part. Further, we design a pooling-based decomposition (PD), a universal range-null space decomposition for super-resolution tasks, which is concise, fast, and parameter-free. PD can be easily applied to state-of-the-art GAN Prior based SR methods to eliminate their inconsistencies, neither compromise the realness nor bring extra parameters or computational costs. Besides, our ablation studies reveal that PD can replace pixel-wise losses for training and achieve better generalization performance when facing unseen downsamplings or even real-world degradation. Experiments show that the use of PD refreshes state-of-the-art SR performance and speeds up the convergence of training up to 2~10 times.

----

## [303] Contrastive Masked Autoencoders for Self-Supervised Video Hashing

**Authors**: *Yuting Wang, Jinpeng Wang, Bin Chen, Ziyun Zeng, Shu-Tao Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25373](https://doi.org/10.1609/aaai.v37i3.25373)

**Abstract**:

Self-Supervised Video Hashing (SSVH) models learn to generate short binary representations for videos without ground-truth supervision, facilitating large-scale video retrieval efficiency and attracting increasing research attention. The success of SSVH lies in the understanding of video content and the ability to capture the semantic relation among unlabeled videos. Typically, state-of-the-art SSVH methods consider these two points in a two-stage training pipeline, where they firstly train an auxiliary network by instance-wise mask-and-predict tasks and secondly train a hashing model to preserve the pseudo-neighborhood structure transferred from the auxiliary network. This consecutive training strategy is inflexible and also unnecessary. In this paper, we propose a simple yet effective one-stage SSVH method called ConMH, which incorporates video semantic information and video similarity relationship understanding in a single stage. To capture video semantic information for better hashing learning, we adopt an encoder-decoder structure to reconstruct the video from its temporal-masked frames. Particularly, we find that a higher masking ratio helps video understanding. Besides, we fully exploit the similarity relationship between videos by maximizing agreement between two augmented views of a video, which contributes to more discriminative and robust hash codes. Extensive experiments on three large-scale video datasets (i.e., FCVID, ActivityNet and YFCC) indicate that ConMH achieves state-of-the-art results. Code is available at https://github.com/huangmozhi9527/ConMH.

----

## [304] MicroAST: Towards Super-fast Ultra-Resolution Arbitrary Style Transfer

**Authors**: *Zhizhong Wang, Lei Zhao, Zhiwen Zuo, Ailin Li, Haibo Chen, Wei Xing, Dongming Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25374](https://doi.org/10.1609/aaai.v37i3.25374)

**Abstract**:

Arbitrary style transfer (AST) transfers arbitrary artistic styles onto content images. Despite the recent rapid progress, existing AST methods are either incapable or too slow to run at ultra-resolutions (e.g., 4K) with limited resources, which heavily hinders their further applications. In this paper, we tackle this dilemma by learning a straightforward and lightweight model, dubbed MicroAST. The key insight is to completely abandon the use of cumbersome pre-trained Deep Convolutional Neural Networks (e.g., VGG) at inference. Instead, we design two micro encoders (content and style encoders) and one micro decoder for style transfer. The content encoder aims at extracting the main structure of the content image. The style encoder, coupled with a modulator, encodes the style image into learnable dual-modulation signals that modulate both intermediate features and convolutional filters of the decoder, thus injecting more sophisticated and flexible style signals to guide the stylizations. In addition, to boost the ability of the style encoder to extract more distinct and representative style signals, we also introduce a new style signal contrastive loss in our model. Compared to the state of the art, our MicroAST not only produces visually superior results but also is 5-73 times smaller and 6-18 times faster, for the first time enabling super-fast (about 0.5 seconds) AST at 4K ultra-resolutions.

----

## [305] Truncate-Split-Contrast: A Framework for Learning from Mislabeled Videos

**Authors**: *Zixiao Wang, Junwu Weng, Chun Yuan, Jue Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25375](https://doi.org/10.1609/aaai.v37i3.25375)

**Abstract**:

Learning with noisy label is a classic problem that has been extensively studied for image tasks, but much less for video in the literature. A straightforward migration from images to videos without considering temporal semantics and computational cost is not a sound choice. In this paper, we propose two new strategies for video analysis with noisy labels: 1) a lightweight channel selection method dubbed as Channel Truncation for feature-based label noise detection. This method selects the most discriminative channels to split clean and noisy instances in each category. 2) A novel contrastive strategy dubbed as Noise Contrastive Learning, which constructs the relationship between clean and noisy instances to regularize model training. Experiments on three well-known benchmark datasets for video classification show that our proposed truNcatE-split-contrAsT (NEAT) significantly outperforms the existing baselines. By reducing the dimension to 10% of it, our method achieves over 0.4 noise detection F1-score and 5% classification accuracy improvement on Mini-Kinetics dataset under severe noise (symmetric-80%). Thanks to Noise Contrastive Learning, the average classification accuracy improvement on Mini-Kinetics and Sth-Sth-V1 is over 1.6%.

----

## [306] Active Token Mixer

**Authors**: *Guoqiang Wei, Zhizheng Zhang, Cuiling Lan, Yan Lu, Zhibo Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25376](https://doi.org/10.1609/aaai.v37i3.25376)

**Abstract**:

The three existing dominant network families, i.e., CNNs, Transformers and MLPs, differ from each other mainly in the ways of fusing spatial contextual information, leaving designing more effective token-mixing mechanisms at the core of backbone architecture development. In this work, we propose an innovative token-mixer, dubbed Active Token Mixer (ATM), to actively incorporate contextual information from other tokens in the global scope into the given query token. This fundamental operator actively predicts where to capture useful contexts and learns how to fuse the captured contexts with the query token at channel level. In this way, the spatial range of token-mixing can be expanded to a global scope with limited computational complexity, where the way of token-mixing is reformed. We take ATMs as the primary operators and assemble them into a cascade architecture, dubbed ATMNet. Extensive experiments demonstrate that ATMNet is generally applicable and comprehensively surpasses different families of SOTA vision backbones by a clear margin on a broad range of vision tasks, including visual recognition and dense prediction tasks.  Code is available at https://github.com/microsoft/ActiveMLP.

----

## [307] Exploring Non-target Knowledge for Improving Ensemble Universal Adversarial Attacks

**Authors**: *Juanjuan Weng, Zhiming Luo, Zhun Zhong, Dazhen Lin, Shaozi Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25377](https://doi.org/10.1609/aaai.v37i3.25377)

**Abstract**:

The ensemble attack with average weights can be leveraged for increasing the transferability of universal adversarial perturbation (UAP) by training with multiple Convolutional Neural Networks (CNNs). However, after analyzing the Pearson Correlation Coefficients (PCCs) between the ensemble logits and individual logits of the crafted UAP trained by the ensemble attack, we find that one CNN plays a dominant role during the optimization. Consequently, this average weighted strategy will weaken the contributions of other CNNs and thus limit the transferability for other black-box CNNs. To deal with this bias issue, the primary attempt is to leverage the Kullback–Leibler (KL) divergence loss to encourage the joint contribution from different CNNs, which is still insufficient. After decoupling the KL loss into a target-class part and a non-target-class part, the main issue lies in that the non-target knowledge will be significantly suppressed due to the increasing logit of the target class. 
In this study, we simply adopt a KL loss that only considers the non-target classes for addressing the dominant bias issue. Besides, to further boost the transferability, we incorporate the min-max learning framework to self-adjust the ensemble weights for each CNN. Experiments results validate that considering the non-target KL loss can achieve superior transferability than the original KL loss by a large margin, and the min-max training can provide a mutual benefit in adversarial ensemble attacks.
The source code is available at: https://github.com/WJJLL/ND-MM.

----

## [308] Towards Good Practices for Missing Modality Robust Action Recognition

**Authors**: *Sangmin Woo, Sumin Lee, Yeonju Park, Muhammad Adi Nugroho, Changick Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25378](https://doi.org/10.1609/aaai.v37i3.25378)

**Abstract**:

Standard multi-modal models assume the use of the same modalities in training and inference stages. However, in practice, the environment in which multi-modal models operate may not satisfy such assumption. As such, their performances degrade drastically if any modality is missing in the inference stage. We ask: how can we train a model that is robust to missing modalities? This paper seeks a set of good practices for multi-modal action recognition, with a particular interest in circumstances where some modalities are not available at an inference time. First, we show how to effectively regularize the model during training (e.g., data augmentation). Second, we investigate on fusion methods for robustness to missing modalities: we find that transformer-based fusion shows better robustness for missing modality than summation or concatenation. Third, we propose a simple modular network, ActionMAE, which learns missing modality predictive coding by randomly dropping modality features and tries to reconstruct them with the remaining modality features. Coupling these good practices, we build a model that is not only effective in multi-modal action recognition but also robust to modality missing. Our model achieves the state-of-the-arts on multiple benchmarks and maintains competitive performances even in missing modality scenarios.

----

## [309] Reject Decoding via Language-Vision Models for Text-to-Image Synthesis

**Authors**: *Fuxiang Wu, Liu Liu, Fusheng Hao, Fengxiang He, Lei Wang, Jun Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25379](https://doi.org/10.1609/aaai.v37i3.25379)

**Abstract**:

Transformer-based text-to-image synthesis generates images from abstractive textual conditions and achieves prompt results. Since transformer-based models predict visual tokens step by step in testing, where the early error is hard to be corrected and would be propagated. To alleviate this issue, the common practice is drawing multi-paths from the transformer-based models and re-ranking the multi-images decoded from multi-paths to find the best one and filter out others. Therefore, the computing procedure of excluding images may be inefficient. To improve the effectiveness and efficiency of decoding, we exploit a reject decoding algorithm with tiny multi-modal models to enlarge the searching space and exclude the useless paths as early as possible. Specifically, we build tiny multi-modal models to evaluate the similarities between the partial paths and the caption at multi scales. Then, we propose a reject decoding algorithm to exclude some lowest quality partial paths at the inner steps. Thus, under the same computing load as the original decoding, we could search across more multi-paths to improve the decoding efficiency and synthesizing quality. The experiments conducted on the MS-COCO dataset and large-scale datasets show that the proposed reject decoding algorithm can exclude the useless paths and enlarge the searching paths to improve the synthesizing quality by consuming less time.

----

## [310] Transformation-Equivariant 3D Object Detection for Autonomous Driving

**Authors**: *Hai Wu, Chenglu Wen, Wei Li, Xin Li, Ruigang Yang, Cheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25380](https://doi.org/10.1609/aaai.v37i3.25380)

**Abstract**:

3D object detection received increasing attention in autonomous driving recently. Objects in 3D scenes are distributed with diverse orientations. Ordinary detectors do not explicitly model the variations of rotation and reflection transformations. Consequently, large networks and extensive data augmentation are required for robust detection. Recent equivariant networks explicitly model the transformation variations by applying shared networks on multiple transformed point clouds, showing great potential in object geometry modeling. However, it is difficult to apply such networks to 3D object detection in autonomous driving due to its large computation cost and slow reasoning speed. In this work, we present TED, an efficient Transformation-Equivariant 3D Detector to overcome the computation cost and speed issues. TED first applies a sparse convolution backbone to extract multi-channel transformation-equivariant voxel features; and then aligns and aggregates these equivariant features into lightweight and compact representations for high-performance 3D object detection. On the highly competitive KITTI 3D car detection leaderboard, TED ranked 1st among all submissions with competitive efficiency. Code is available at https://github.com/hailanyi/TED.

----

## [311] Super-efficient Echocardiography Video Segmentation via Proxy- and Kernel-Based Semi-supervised Learning

**Authors**: *Huisi Wu, Jingyin Lin, Wende Xie, Jing Qin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25381](https://doi.org/10.1609/aaai.v37i3.25381)

**Abstract**:

Automatic segmentation of left ventricular endocardium in echocardiography videos is critical for assessing various cardiac functions and improving the diagnosis of cardiac diseases. It is yet a challenging task due to heavy speckle noise, significant shape variability of cardiac structure, and limited labeled data. Particularly, the real-time demand in clinical practice makes this task even harder. In this paper, we propose a novel proxy- and kernel-based semi-supervised segmentation network (PKEcho-Net) to comprehensively address these challenges. We first propose a multi-scale region proxy (MRP) mechanism to model the region-wise contexts, in which a learnable region proxy with an arbitrary shape is developed in each layer of the encoder, allowing the network to identify homogeneous semantics and hence alleviate the influence of speckle noise on segmentation. To sufficiently and efficiently exploit temporal consistency, different from traditional methods which only utilize the temporal contexts of two neighboring frames via feature warping or self-attention mechanism, we formulate the semi-supervised segmentation with a group of learnable kernels, which can naturally and uniformly encode the appearances of left ventricular endocardium, as well as extracting the inter-frame contexts across the whole video to resist the fast shape variability of cardiac structures. Extensive experiments have been conducted on two famous public echocardiography video datasets, EchoNet-Dynamic and CAMUS. Our model achieves the best performance-efficiency trade-off when compared with other state-of-the-art approaches, attaining comparative accuracy with a much faster speed. The code is available at https://github.com/JingyinLin/PKEcho-Net.

----

## [312] ACL-Net: Semi-supervised Polyp Segmentation via Affinity Contrastive Learning

**Authors**: *Huisi Wu, Wende Xie, Jingyin Lin, Xinrong Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25382](https://doi.org/10.1609/aaai.v37i3.25382)

**Abstract**:

Automatic polyp segmentation from colonoscopy images is an essential prerequisite for the development of computer-assisted therapy. However, the complex semantic information and the blurred edges of polyps make segmentation extremely difficult. In this paper, we propose a novel semi-supervised polyp segmentation framework using affinity contrastive learning (ACL-Net), which is implemented between student and teacher networks to consistently refine the pseudo-labels for semi-supervised polyp segmentation. By aligning the affinity maps between the two branches, a better polyp region activation can be obtained to fully exploit the appearance-level context encoded in the feature maps, thereby improving the capability of capturing not only global localization and shape context, but also the local textural and boundary details. By utilizing the rich inter-image affinity context and establishing a global affinity context based on the memory bank, a cross-image affinity aggregation (CAA) module is also implemented to further refine the affinity aggregation between the two branches. By continuously and adaptively refining pseudo-labels with optimized affinity, we can improve the semi-supervised polyp segmentation based on the mutually reinforced knowledge interaction among contrastive learning and consistency learning iterations. Extensive experiments on five benchmark datasets, including Kvasir-SEG, CVC-ClinicDB, CVC-300, CVC-ColonDB and ETIS, demonstrate the effectiveness and superiority of our method. Codes are available at https://github.com/xiewende/ACL-Net.

----

## [313] Bi-directional Feature Reconstruction Network for Fine-Grained Few-Shot Image Classification

**Authors**: *Jijie Wu, Dongliang Chang, Aneeshan Sain, Xiaoxu Li, Zhanyu Ma, Jie Cao, Jun Guo, Yi-Zhe Song*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25383](https://doi.org/10.1609/aaai.v37i3.25383)

**Abstract**:

The main challenge for fine-grained few-shot image classification is to learn feature representations with higher inter-class and lower intra-class variations, with a mere few labelled samples. Conventional few-shot learning methods however cannot be naively adopted for this fine-grained setting -- a quick pilot study reveals that they in fact push for the opposite (i.e., lower inter-class variations and higher intra-class variations). To alleviate this problem, prior works predominately use a support set to reconstruct the query image and then utilize metric learning to determine its category. Upon careful inspection, we further reveal that such unidirectional reconstruction methods only help to increase inter-class variations and are not effective in tackling intra-class variations. In this paper, we for the first time introduce a bi-reconstruction mechanism that can simultaneously accommodate for inter-class and intra-class variations. In addition to using the support set to reconstruct the query set for increasing inter-class variations, we further use the query set to reconstruct the support set for reducing intra-class variations. This design effectively helps the model to explore more subtle and discriminative features which is key for the fine-grained problem in hand. Furthermore, we also construct a self-reconstruction module to work alongside the bi-directional module to make the features even more discriminative. Experimental results on three widely used fine-grained image classification datasets consistently show considerable improvements compared with other methods. Codes are available at: https://github.com/PRIS-CV/Bi-FRN.

----

## [314] Preserving Structural Consistency in Arbitrary Artist and Artwork Style Transfer

**Authors**: *Jingyu Wu, Lefan Hou, Zejian Li, Jun Liao, Li Liu, Lingyun Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25384](https://doi.org/10.1609/aaai.v37i3.25384)

**Abstract**:

Deep generative models are effective in style transfer. 
Previous methods learn one or several specific artist-style from a collection of artworks.
These methods not only homogenize the artist-style of different artworks of the same artist but also lack generalization for the unseen artists.
To solve these challenges, we propose a double-style transferring module (DSTM).
It extracts different artist-style and artwork-style from different artworks (even untrained) and preserves the intrinsic diversity between different artworks of the same artist.
DSTM swaps the two styles in the adversarial training and encourages realistic image generation given arbitrary style combinations.
However, learning style from single artwork can often cause over-adaption to it, resulting in the introduction of structural features of style image.
We further propose an edge enhancing module (EEM) which derives edge information from multi-scale and multi-level features to enhance structural consistency.
We broadly evaluate our method across six large-scale benchmark datasets.
Empirical results show that our method achieves arbitrary artist-style and artwork-style extraction from a single artwork, and effectively avoids introducing the style image’s structural features.
Our method improves the state-of-the-art deception rate from 58.9% to 67.2% and the average FID from 48.74 to 42.83.

----

## [315] End-to-End Zero-Shot HOI Detection via Vision and Language Knowledge Distillation

**Authors**: *Mingrui Wu, Jiaxin Gu, Yunhang Shen, Mingbao Lin, Chao Chen, Xiaoshuai Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25385](https://doi.org/10.1609/aaai.v37i3.25385)

**Abstract**:

Most existing Human-Object Interaction (HOI) Detection methods rely heavily on full annotations with predefined HOI categories, which is limited in diversity and costly to scale further. We aim at advancing zero-shot HOI detection to detect both seen and unseen HOIs simultaneously. The fundamental challenges are to discover potential human-object pairs and identify novel HOI categories. To overcome the above challenges, we propose a novel End-to-end zero-shot HOI Detection (EoID) framework via vision-language knowledge distillation. We first design an Interactive Score module combined with a Two-stage Bipartite Matching algorithm to achieve interaction distinguishment for human-object pairs in an action-agnostic manner.
Then we transfer the distribution of action probability from the pretrained vision-language teacher as well as the seen ground truth to the HOI model to attain zero-shot HOI classification. Extensive experiments on HICO-Det dataset demonstrate that our model discovers potential interactive pairs and enables the recognition of unseen HOIs. Finally, our method outperforms the previous SOTA under various zero-shot settings. Moreover, our method is generalizable to large-scale object detection data to further scale up the action sets. The source code is available at: https://github.com/mrwu-mac/EoID.

----

## [316] Revisiting Classifier: Transferring Vision-Language Models for Video Recognition

**Authors**: *Wenhao Wu, Zhun Sun, Wanli Ouyang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25386](https://doi.org/10.1609/aaai.v37i3.25386)

**Abstract**:

Transferring knowledge from task-agnostic pre-trained deep models for downstream tasks is an important topic in computer vision research. Along with the growth of computational capacity, we now have open-source vision-language pre-trained models in large scales of the model architecture and amount of data. In this study, we focus on transferring knowledge for video classification tasks. Conventional methods randomly initialize the linear classifier head for vision classification, but they leave the usage of the text encoder for downstream visual recognition tasks undiscovered. In this paper, we revise the role of the linear classifier and replace the classifier with the different knowledge from pre-trained model. We utilize the well-pretrained language model to generate good semantic target for efficient transferring learning. The empirical study shows that our method improves both the performance and the training speed of video classification, with a negligible change in the model. Our simple yet effective tuning paradigm achieves state-of-the-art performance and efficient training on various video recognition scenarios, i.e., zero-shot, few-shot, general recognition. In particular, our paradigm achieves the state-of-the-art accuracy of 87.8% on Kinetics-400, and also surpasses previous methods by 20~50% absolute top-1 accuracy under zero-shot, few-shot settings on five video datasets. Code and models are available at https://github.com/whwu95/Text4Vis.

----

## [317] Scene Graph to Image Synthesis via Knowledge Consensus

**Authors**: *Yang Wu, Pengxu Wei, Liang Lin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25387](https://doi.org/10.1609/aaai.v37i3.25387)

**Abstract**:

In this paper, we study graph-to-image generation conditioned exclusively on scene graphs, in which we seek to disentangle the veiled semantics between knowledge graphs and images.  While most existing research resorts to laborious auxiliary information such as object layouts or segmentation masks, it is also of interest to unveil the generality of the model with limited supervision, moreover, avoiding extra cross-modal alignments.  To tackle this challenge, we delve into the causality of the adversarial generation process, and reason out a new principle to realize a simultaneous semantic disentanglement with an alignment on target and model distributions.  This principle is named knowledge consensus, which explicitly describes a triangle causal dependency among observed images, graph semantics and hidden visual representations.  The consensus also determines a new graph-to-image generation framework, carried on several adversarial optimization objectives.  Extensive experimental results demonstrate that, even conditioned only on scene graphs, our model surprisingly achieves superior performance on semantics-aware image generation, without losing the competence on manipulating the generation through knowledge graphs.

----

## [318] Synthetic Data Can Also Teach: Synthesizing Effective Data for Unsupervised Visual Representation Learning

**Authors**: *Yawen Wu, Zhepeng Wang, Dewen Zeng, Yiyu Shi, Jingtong Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25388](https://doi.org/10.1609/aaai.v37i3.25388)

**Abstract**:

Contrastive learning (CL), a self-supervised learning approach, can effectively learn visual representations from unlabeled data. Given the CL training data, generative models can be trained to generate synthetic data to supplement the real data. Using both synthetic and real data for CL training has the potential to improve the quality of learned representations. However, synthetic data usually has lower quality than real data, and using synthetic data may not improve CL compared with using real data. To tackle this problem, we propose a data generation framework with two methods to improve CL training by joint sample generation and contrastive learning. The first approach generates hard samples for the main model. The generator is jointly learned with the main model to dynamically customize hard samples based on the training state of the main model. Besides, a pair of data generators are proposed to generate similar but distinct samples as positive pairs. In joint learning, the hardness of a positive pair is progressively increased by decreasing their similarity. Experimental results on multiple datasets show superior accuracy and data efficiency of the proposed data generation methods applied to CL. For example, about 4.0%, 3.5%, and 2.6% accuracy improvements for linear classification are observed on ImageNet-100, CIFAR-100, and CIFAR-10, respectively. Besides, up to 2× data efficiency for linear classification and up to 5× data efficiency for transfer learning are achieved.

----

## [319] Multi-Stream Representation Learning for Pedestrian Trajectory Prediction

**Authors**: *Yuxuan Wu, Le Wang, Sanping Zhou, Jinghai Duan, Gang Hua, Wei Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25389](https://doi.org/10.1609/aaai.v37i3.25389)

**Abstract**:

Forecasting the future trajectory of pedestrians is an important task in computer vision with a range of applications, from security cameras to autonomous driving. It is very challenging because pedestrians not only move individually across time but also interact spatially, and the spatial and temporal information is deeply coupled with one another in a multi-agent scenario. Learning such complex spatio-temporal correlation is a fundamental issue in pedestrian trajectory prediction. Inspired by the procedure that the hippocampus processes and integrates spatio-temporal information to form memories, we propose a novel multi-stream representation learning module to learn complex spatio-temporal features of pedestrian trajectory. Specifically, we learn temporal, spatial and cross spatio-temporal correlation features in three respective pathways and then adaptively integrate these features with learnable weights by a gated network. Besides, we leverage the sparse attention gate to select informative interactions and correlations brought by complex spatio-temporal modeling and reduce complexity of our model. We evaluate our proposed method on two commonly used datasets, i.e. ETH-UCY and SDD, and the experimental results demonstrate our method achieves the state-of-the-art performance. Code: https://github.com/YuxuanIAIR/MSRL-master

----

## [320] Pixel Is All You Need: Adversarial Trajectory-Ensemble Active Learning for Salient Object Detection

**Authors**: *Zhenyu Wu, Lin Wang, Wei Wang, Qing Xia, Chenglizhao Chen, Aimin Hao, Shuo Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25390](https://doi.org/10.1609/aaai.v37i3.25390)

**Abstract**:

Although weakly-supervised techniques can reduce the labeling effort, it is unclear whether a saliency model trained with weakly-supervised data (e.g., point annotation) can achieve the equivalent performance of its fully-supervised version. This paper attempts to answer this unexplored question by proving a hypothesis: there is a point-labeled dataset where saliency models trained on it can achieve equivalent performance when trained on the densely annotated dataset. To prove this conjecture, we proposed a novel yet effective adversarial trajectory-ensemble active learning (ATAL). Our contributions are three-fold:  1) Our proposed adversarial attack triggering uncertainty can conquer the overconfidence of existing active learning methods and accurately locate these uncertain pixels. 2) Our proposed trajectory-ensemble uncertainty estimation method maintains the advantages of the ensemble networks while significantly reducing the computational cost. 3) Our proposed relationship-aware diversity sampling algorithm can conquer oversampling while boosting performance. Experimental results show that our ATAL can find such a point-labeled dataset, where a saliency model trained on it obtained 97%-99% performance of its fully-supervised version with only 10 annotated points per image.

----

## [321] Attention-Based Depth Distillation with 3D-Aware Positional Encoding for Monocular 3D Object Detection

**Authors**: *Zizhang Wu, Yunzhe Wu, Jian Pu, Xianzhi Li, Xiaoquan Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25391](https://doi.org/10.1609/aaai.v37i3.25391)

**Abstract**:

Monocular 3D object detection is a low-cost but challenging task, as it requires generating accurate 3D localization solely from a single image input. Recent developed depth-assisted methods show promising results by using explicit depth maps as intermediate features, which are either precomputed by monocular depth estimation networks or jointly evaluated with 3D object detection. However, inevitable errors from estimated depth priors may lead to misaligned semantic information and 3D localization, hence resulting in feature smearing and suboptimal predictions. To mitigate this issue, we propose ADD, an Attention-based Depth knowledge Distillation framework with 3D-aware positional encoding. Unlike previous knowledge distillation frameworks that adopt stereo- or LiDAR-based teachers, we build up our teacher with identical architecture as the student but with extra ground-truth depth as input. Credit to our teacher design, our framework is seamless, domain-gap free, easily implementable, and is compatible with object-wise ground-truth depth. Specifically, we leverage intermediate features and responses for knowledge distillation. Considering long-range 3D dependencies, we propose 3D-aware self-attention and target-aware cross-attention modules for student adaptation. Extensive experiments are performed to verify the effectiveness of our framework on the challenging KITTI 3D object detection benchmark. We implement our framework on three representative monocular detectors, and we achieve state-of-the-art performance with no additional inference computational cost relative to baseline models. Our code is available at https://github.com/rockywind/ADD.

----

## [322] Skating-Mixer: Long-Term Sport Audio-Visual Modeling with MLPs

**Authors**: *Jingfei Xia, Mingchen Zhuge, Tiantian Geng, Shun Fan, Yuantai Wei, Zhenyu He, Feng Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25392](https://doi.org/10.1609/aaai.v37i3.25392)

**Abstract**:

Figure skating scoring is challenging because it requires judging players’ technical moves as well as coordination with the background music. Most learning-based methods struggle for two reasons: 1) each move in figure skating changes quickly, hence simply applying traditional frame sampling will lose a lot of valuable information, especially in 3 to 5 minutes lasting videos; 2) prior methods rarely considered the critical audio-visual relationship in their models. Due to these reasons, we introduce a novel architecture, named Skating-Mixer. It extends the MLP framework into a multimodal fashion and effectively learns long-term representations through our designed memory recurrent unit (MRU). Aside from the model, we collected a high-quality audio-visual FS1000 dataset, which contains over 1000 videos on 8 types of programs with 7 different rating metrics, overtaking other datasets in both quantity and diversity. Experiments show the proposed method achieves SOTAs over all major metrics on the public Fis-V and our FS1000 dataset. In addition, we include an analysis applying our method to the recent competitions in Beijing 2022 Winter Olympic Games, proving our method has strong applicability.

----

## [323] SVFI: Spiking-Based Video Frame Interpolation for High-Speed Motion

**Authors**: *Lujie Xia, Jing Zhao, Ruiqin Xiong, Tiejun Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25393](https://doi.org/10.1609/aaai.v37i3.25393)

**Abstract**:

Occlusion and motion blur make it challenging to interpolate video frame, since estimating complex motions between two frames is hard and unreliable, especially in highly dynamic scenes. This paper aims to address these issues by exploiting spike stream as auxiliary visual information between frames to synthesize target frames. Instead of estimating motions by optical flow from RGB frames, we present a new dual-modal pipeline adopting both RGB frames and the corresponding spike stream as inputs (SVFI). It extracts the scene structure and objects' outline feature maps of the target frames from spike stream. Those feature maps are fused with the color and texture feature maps extracted from RGB frames to synthesize target frames. Benefited by the spike stream that contains consecutive information between two frames, SVFI can directly extract the information in occlusion and motion blur areas of target frames from spike stream, thus it is more robust than previous optical flow-based methods. Experiments show SVFI outperforms the SOTA methods on wide variety of datasets. For instance, in 7 and 15 frame skip evaluations, it shows up to 5.58 dB and 6.56 dB improvements in terms of PSNR over the corresponding second best methods BMBC and DAIN. SVFI also shows visually impressive performance in real-world scenes.

----

## [324] FEditNet: Few-Shot Editing of Latent Semantics in GAN Spaces

**Authors**: *Mengfei Xia, Yezhi Shu, Yuji Wang, Yu-Kun Lai, Qiang Li, Pengfei Wan, Zhongyuan Wang, Yong-Jin Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25394](https://doi.org/10.1609/aaai.v37i3.25394)

**Abstract**:

Generative Adversarial networks (GANs) have demonstrated their powerful capability of synthesizing high-resolution images, and great efforts have been made to interpret the semantics in the latent spaces of GANs. However, existing works still have the following limitations: (1) the majority of works rely on either pretrained attribute predictors or large-scale labeled datasets, which are difficult to collect in most cases, and (2) some other methods are only suitable for restricted cases, such as focusing on interpretation of human facial images using prior facial semantics. In this paper, we propose a GAN-based method called FEditNet, aiming to discover latent semantics using very few labeled data without any pretrained predictors or prior knowledge. Specifically, we reuse the knowledge from the pretrained GANs, and by doing so, avoid overfitting during the few-shot training of FEditNet. Moreover, our layer-wise objectives which take content consistency into account also ensure the disentanglement between attributes. Qualitative and quantitative results demonstrate that our method outperforms the state-of-the-art methods on various datasets. The code is available at https://github.com/THU-LYJ-Lab/FEditNet.

----

## [325] Toward Robust Diagnosis: A Contour Attention Preserving Adversarial Defense for COVID-19 Detection

**Authors**: *Kun Xiang, Xing Zhang, Jinwen She, Jinpeng Liu, Haohan Wang, Shiqi Deng, Shancheng Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25395](https://doi.org/10.1609/aaai.v37i3.25395)

**Abstract**:

As the COVID-19 pandemic puts pressure on healthcare systems worldwide, the computed tomography image based AI diagnostic system has become a sustainable solution for early diagnosis. However, the model-wise vulnerability under adversarial perturbation hinders its deployment in practical situation. The existing adversarial training strategies are difficult to generalized into medical imaging field challenged by complex medical texture features. To overcome this challenge, we propose a Contour Attention Preserving (CAP) method based on lung cavity edge extraction. The contour prior features are injected to attention layer via a parameter regularization and we optimize the robust empirical risk with hybrid distance metric. We then introduce a new cross-nation CT scan dataset to evaluate the generalization capability of the adversarial robustness under distribution shift. Experimental results indicate that the proposed method achieves state-of-the-art performance in multiple adversarial defense and generalization tasks. The code and dataset are available at https://github.com/Quinn777/CAP.

----

## [326] Boosting Semi-Supervised Semantic Segmentation with Probabilistic Representations

**Authors**: *Haoyu Xie, Changqi Wang, Mingkai Zheng, Minjing Dong, Shan You, Chong Fu, Chang Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25396](https://doi.org/10.1609/aaai.v37i3.25396)

**Abstract**:

Recent breakthroughs in semi-supervised semantic segmentation have been developed through contrastive learning. In prevalent pixel-wise contrastive learning solutions, the model maps pixels to deterministic representations and regularizes them in the latent space. However, there exist inaccurate pseudo-labels which map the ambiguous representations of pixels to the wrong classes due to the limited cognitive ability of the model. In this paper, we define pixel-wise representations from a new perspective of probability theory and propose a Probabilistic Representation Contrastive Learning (PRCL) framework that improves representation quality by taking its probability into consideration. Through modelling the mapping from pixels to representations as the probability via multivariate Gaussian distributions, we can tune the contribution of the ambiguous representations to tolerate the risk of inaccurate pseudo-labels. Furthermore, we define prototypes in the form of distributions, which indicates the confidence of a class, while the point prototype cannot. More- over, we propose to regularize the distribution variance to enhance the reliability of representations. Taking advantage of these benefits, high-quality feature representations can be derived in the latent space, thereby the performance of se- mantic segmentation can be further improved. We conduct sufficient experiment to evaluate PRCL on Pascal VOC and CityScapes to demonstrate its superiority. The code is available at https://github.com/Haoyu-Xie/PRCL.

----

## [327] Less Is More Important: An Attention Module Guided by Probability Density Function for Convolutional Neural Networks

**Authors**: *Jingfen Xie, Jian Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25397](https://doi.org/10.1609/aaai.v37i3.25397)

**Abstract**:

Attention modules, which adaptively weight and refine features according to the importance of the input, have become a critical technique to boost the capability of convolutional neural networks. However, most existing attention modules are heuristic without a sound interpretation, and thus, require empirical engineering to design structure and operators within the modules. To handle the above issue, based on our 'less is more important' observation, we propose an Attention Module guided by Probability Density Function (PDF), dubbed PdfAM, which enjoys a rational motivation and requires few empirical structure designs. Concretely, we observe that pixels with less occurrence are prone to be textural details or foreground objects with much importance to aid vision tasks. Thus, with PDF values adopted as a smooth and anti-noise alternative to the pixel occurrence frequency, we design our PdfAM by first estimating the PDF based on some distribution assumption, and then predicting a 3D attention map via applying a negative correlation between the attention weights and the estimated PDF values. Furthermore, we develop learnable PDF-rescale parameters so as to adaptively transform the estimated PDF and predict a customized negative correlation. Experiments show that our PdfAM consistently boosts various networks under both high- and low-level vision tasks, and also performs favorably against other attention modules in terms of accuracy and convergence.

----

## [328] Mitigating Artifacts in Real-World Video Super-resolution Models

**Authors**: *Liangbin Xie, Xintao Wang, Shuwei Shi, Jinjin Gu, Chao Dong, Ying Shan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25398](https://doi.org/10.1609/aaai.v37i3.25398)

**Abstract**:

The recurrent structure is a prevalent framework for the task of video super-resolution, which models the temporal dependency between frames via hidden states. When applied to real-world scenarios with unknown and complex degradations, hidden states tend to contain unpleasant artifacts and propagate them to restored frames. In this circumstance, our analyses show that such artifacts can be largely alleviated when the hidden state is replaced with a cleaner counterpart. Based on the observations, we propose a Hidden State Attention (HSA) module to mitigate artifacts in real-world video super-resolution. Specifically, we first adopt various cheap filters to produce a hidden state pool. For example, Gaussian blur filters are for smoothing artifacts while sharpening filters are for enhancing details. To aggregate a new hidden state that contains fewer artifacts from the hidden state pool, we devise a Selective Cross Attention (SCA) module, in which the attention between input features and each hidden state is calculated. Equipped with HSA, our proposed method, namely FastRealVSR, is able to achieve 2x speedup while obtaining better performance than Real-BasicVSR. Codes will be available at https://github.com/TencentARC/FastRealVSR.

----

## [329] Just Noticeable Visual Redundancy Forecasting: A Deep Multimodal-Driven Approach

**Authors**: *Wuyuan Xie, Shukang Wang, Sukun Tian, Lirong Huang, Ye Liu, Miaohui Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25399](https://doi.org/10.1609/aaai.v37i3.25399)

**Abstract**:

Just noticeable difference (JND) refers to the maximum visual change that human eyes cannot perceive, and it has a wide range of applications in multimedia systems. However, most existing JND approaches only focus on a single modality, and rarely consider the complementary effects of multimodal information. In this article, we investigate the JND modeling from an end-to-end homologous multimodal perspective, namely hmJND-Net. Specifically, we explore three important visually sensitive modalities, including saliency, depth, and segmentation. To better utilize homologous multimodal information, we establish an effective fusion method via summation enhancement and subtractive offset, and align homologous multimodal features based on a self-attention driven encoder-decoder paradigm. Extensive experimental results on eight different benchmark datasets validate the superiority of our hmJND-Net over eight representative methods.

----

## [330] Cross-Modal Contrastive Learning for Domain Adaptation in 3D Semantic Segmentation

**Authors**: *Bowei Xing, Xianghua Ying, Ruibin Wang, Jinfa Yang, Taiyan Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25400](https://doi.org/10.1609/aaai.v37i3.25400)

**Abstract**:

Domain adaptation for 3D point cloud has attracted a lot of interest since it can avoid the time-consuming labeling process of 3D data to some extent. A recent work named xMUDA leveraged multi-modal data to domain adaptation task of 3D semantic segmentation by mimicking the predictions between 2D and 3D modalities, and outperformed the previous single modality methods only using point clouds. Based on it, in this paper, we propose a novel cross-modal contrastive learning scheme to further improve the adaptation effects. By employing constraints from the correspondences between 2D pixel features and 3D point features, our method not only facilitates interaction between the two different modalities, but also boosts feature representations in both labeled source domain and unlabeled target domain. Meanwhile, to sufficiently utilize 2D context information for domain adaptation through cross-modal learning, we introduce a neighborhood feature aggregation module to enhance pixel features. The module employs neighborhood attention to aggregate nearby pixels in the 2D image, which relieves the mismatching between the two different modalities, arising from projecting relative sparse point cloud to dense image pixels. We evaluate our method on three unsupervised domain adaptation scenarios, including country-to-country, day-to-night, and dataset-to-dataset. Experimental results show that our approach outperforms existing methods, which demonstrates the effectiveness of the proposed method.

----

## [331] ROIFormer: Semantic-Aware Region of Interest Transformer for Efficient Self-Supervised Monocular Depth Estimation

**Authors**: *Daitao Xing, Jinglin Shen, Chiuman Ho, Anthony Tzes*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25401](https://doi.org/10.1609/aaai.v37i3.25401)

**Abstract**:

The exploration of mutual-benefit cross-domains has shown great potential toward accurate self-supervised depth estimation. In this work, we revisit feature fusion between depth and semantic information and propose an efficient local adaptive attention method for geometric aware representation enhancement. Instead of building global connections or deforming attention across the feature space without restraint, we bound the spatial interaction within a learnable region of interest. In particular, we leverage geometric cues from semantic information to learn local adaptive bounding boxes to guide unsupervised feature aggregation. The local areas preclude most irrelevant reference points from attention space, yielding more selective feature learning and faster convergence. We naturally extend the paradigm into a multi-head and hierarchic way to enable the information distillation in different semantic levels and improve the feature discriminative ability for fine-grained depth estimation. Extensive experiments on the KITTI dataset show that our proposed method establishes a new state-of-the-art in self-supervised monocular depth estimation task, demonstrating the effectiveness of our approach over former Transformer variants.

----

## [332] LORE: Logical Location Regression Network for Table Structure Recognition

**Authors**: *Hangdi Xing, Feiyu Gao, Rujiao Long, Jiajun Bu, Qi Zheng, Liangcheng Li, Cong Yao, Zhi Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25402](https://doi.org/10.1609/aaai.v37i3.25402)

**Abstract**:

Table structure recognition (TSR) aims at extracting tables in images into machine-understandable formats. Recent methods solve this problem by predicting the adjacency relations of detected cell boxes, or learning to generate the corresponding markup sequences from the table images. However, they either count on additional heuristic rules to recover the table structures, or require a huge amount of training data and time-consuming sequential decoders. In this paper, we propose an alternative paradigm. We model TSR as a logical location regression problem and propose a new TSR framework called LORE, standing for LOgical location REgression network, which for the first time combines logical location regression together with spatial location regression of table cells. Our proposed LORE is conceptually simpler, easier to train and more accurate than previous TSR models of other paradigms. Experiments on standard benchmarks demonstrate that LORE consistently outperforms prior arts. Code is available at https:// github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/DocumentUnderstanding/LORE-TSR.

----

## [333] Revisiting the Spatial and Temporal Modeling for Few-Shot Action Recognition

**Authors**: *Jiazheng Xing, Mengmeng Wang, Yong Liu, Boyu Mu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25403](https://doi.org/10.1609/aaai.v37i3.25403)

**Abstract**:

Spatial and temporal modeling is one of the most core aspects of few-shot action recognition. Most previous works mainly focus on long-term temporal relation modeling based on high-level spatial representations, without considering the crucial low-level spatial features and short-term temporal relations. Actually, the former feature could bring rich local semantic information, and the latter feature could represent motion characteristics of adjacent frames, respectively. In this paper, we propose SloshNet, a new framework that revisits the spatial and temporal modeling for few-shot action recognition in a finer manner. First, to exploit the low-level spatial features, we design a feature fusion architecture search module to automatically search for the best combination of the low-level and high-level spatial features. Next, inspired by the recent transformer, we introduce a long-term temporal modeling module to model the global temporal relations based on the extracted spatial appearance features. Meanwhile, we design another short-term temporal modeling module to encode the motion characteristics between adjacent frame representations. After that, the final predictions can be obtained by feeding the embedded rich spatial-temporal features to a common frame-level class prototype matcher. We extensively validate the proposed SloshNet on four few-shot action recognition datasets, including Something-Something V2, Kinetics, UCF101, and HMDB51. It achieves favorable results against state-of-the-art methods in all datasets.

----

## [334] Unsupervised Multi-Exposure Image Fusion Breaking Exposure Limits via Contrastive Learning

**Authors**: *Han Xu, Liang Haochen, Jiayi Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25404](https://doi.org/10.1609/aaai.v37i3.25404)

**Abstract**:

This paper proposes an unsupervised multi-exposure image fusion (MEF) method via contrastive learning, termed as MEF-CL. It breaks exposure limits and performance bottleneck faced by existing methods. MEF-CL firstly designs similarity constraints to preserve contents in source images. It eliminates the need for ground truth (actually not exist and created artificially) and thus avoids negative impacts of inappropriate ground truth on performance and generalization. Moreover, we explore a latent feature space and apply contrastive learning in this space to guide fused image to approximate normal-light samples and stay away from inappropriately exposed ones. In this way, characteristics of fused images (e.g., illumination, colors) can be further improved without being subject to source images. Therefore, MEF-CL is applicable to image pairs of any multiple exposures rather than a pair of under-exposed and over-exposed images mandated by existing methods. By alleviating dependence on source images, MEF-CL shows better generalization for various scenes. Consequently, our results exhibit appropriate illumination, detailed textures, and saturated colors. Qualitative, quantitative, and ablation experiments validate the superiority and generalization of MEF-CL. Our code is publicly available at https://github.com/hanna-xu/MEF-CL.

----

## [335] CasFusionNet: A Cascaded Network for Point Cloud Semantic Scene Completion by Dense Feature Fusion

**Authors**: *Jinfeng Xu, Xianzhi Li, Yuan Tang, Qiao Yu, Yixue Hao, Long Hu, Min Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25405](https://doi.org/10.1609/aaai.v37i3.25405)

**Abstract**:

Semantic scene completion (SSC) aims to complete a partial 3D scene and predict its semantics simultaneously. Most existing works adopt the voxel representations, thus suffering from the growth of memory and computation cost as the voxel resolution increases. Though a few works attempt to solve SSC from the perspective of 3D point clouds, they have not fully exploited the correlation and complementarity between the two tasks of scene completion and semantic segmentation. In our work, we present CasFusionNet, a novel cascaded network for point cloud semantic scene completion by dense feature fusion. Specifically, we design (i) a global completion module (GCM) to produce an upsampled and completed but coarse point set, (ii) a semantic segmentation module (SSM) to predict the per-point semantic labels of the completed points generated by GCM, and (iii) a local refinement module (LRM) to further refine the coarse completed points and the associated labels from a local perspective. We organize the above three modules via dense feature fusion in each level, and cascade a total of four levels, where we also employ feature fusion between each level for sufficient information usage. Both quantitative and qualitative results on our compiled two point-based datasets validate the effectiveness and superiority of our CasFusionNet compared to state-of-the-art methods in terms of both scene completion and semantic segmentation. The codes and datasets are available at: https://github.com/JinfengX/CasFusionNet.

----

## [336] Learning a Generalized Gaze Estimator from Gaze-Consistent Feature

**Authors**: *Mingjie Xu, Haofei Wang, Feng Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25406](https://doi.org/10.1609/aaai.v37i3.25406)

**Abstract**:

Gaze estimator computes the gaze direction based on face images. Most existing gaze estimation methods perform well under within-dataset settings, but can not generalize to unseen domains. In particular, the ground-truth labels in unseen domain are often unavailable. In this paper, we propose a new domain generalization method based on gaze-consistent features. Our idea is to consider the gaze-irrelevant factors as unfavorable interference and disturb the training data against them, so that the model cannot fit to these gaze-irrelevant factors, instead, only fits to the gaze-consistent features. To this end, we first disturb the training data via adversarial attack or data augmentation based on the gaze-irrelevant factors, i.e., identity, expression, illumination and tone. Then we extract the gaze-consistent features by aligning the gaze features from disturbed data with non-disturbed gaze features. Experimental results show that our proposed method achieves state-of-the-art performance on gaze domain generalization task. Furthermore, our proposed method also improves domain adaption performance on gaze estimation. Our work provides new insight on gaze domain generalization task.

----

## [337] Class Overwhelms: Mutual Conditional Blended-Target Domain Adaptation

**Authors**: *Pengcheng Xu, Boyu Wang, Charles Ling*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25407](https://doi.org/10.1609/aaai.v37i3.25407)

**Abstract**:

Current methods of blended targets domain adaptation (BTDA) usually infer or consider domain label information but underemphasize hybrid categorical feature structures of targets, which yields limited performance, especially under the label distribution shift. We demonstrate that domain labels are not directly necessary for BTDA if categorical distributions of various domains are sufficiently aligned even facing the imbalance of domains and the label distribution shift of classes. However, we observe that the cluster assumption in BTDA does not comprehensively hold. The hybrid categorical feature space hinders the modeling of categorical distributions and the generation of reliable pseudo labels for categorical alignment. To address these, we propose a categorical domain discriminator guided by uncertainty to explicitly model and directly align categorical distributions P(Z|Y). Simultaneously, we utilize the low-level features to augment the single source features with diverse target styles to rectify the biased classifier P(Y|Z) among diverse targets. Such a mutual conditional alignment of P(Z|Y) and P(Y|Z) forms a mutual reinforced mechanism. Our approach outperforms the state-of-the-art in BTDA even compared with methods utilizing domain labels, especially under the label distribution shift, and in single target DA on DomainNet.

----

## [338] Self Correspondence Distillation for End-to-End Weakly-Supervised Semantic Segmentation

**Authors**: *Rongtao Xu, Changwei Wang, Jiaxi Sun, Shibiao Xu, Weiliang Meng, Xiaopeng Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25408](https://doi.org/10.1609/aaai.v37i3.25408)

**Abstract**:

Efficiently training accurate deep models for weakly supervised semantic segmentation (WSSS) with image-level labels is challenging and important. Recently, end-to-end WSSS methods have become the focus of research due to their high training efficiency. However, current methods suffer from insufficient extraction of comprehensive semantic information, resulting in low-quality pseudo-labels and sub-optimal solutions for end-to-end WSSS. To this end, we propose a simple and novel Self Correspondence Distillation (SCD) method to refine pseudo-labels without introducing external supervision. Our SCD enables the network to utilize feature correspondence derived from itself as a distillation target, which can enhance the network's feature learning process by complementing semantic information. In addition, to further improve the segmentation accuracy, we design a Variation-aware Refine Module to enhance the local consistency of pseudo-labels by computing pixel-level variation. Finally, we present an efficient end-to-end Transformer-based framework (TSCD) via SCD and Variation-aware Refine Module for the accurate WSSS task. Extensive experiments on the PASCAL VOC 2012 and MS COCO 2014 datasets demonstrate that our method significantly outperforms other state-of-the-art methods.  Our code is available at https://github.com/Rongtao-Xu/RepresentationLearning/tree/main/SCD-AAAI2023.

----

## [339] Deep Parametric 3D Filters for Joint Video Denoising and Illumination Enhancement in Video Super Resolution

**Authors**: *Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, Jiaya Jia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25409](https://doi.org/10.1609/aaai.v37i3.25409)

**Abstract**:

Despite the quality improvement brought by the recent methods, video super-resolution (SR) is still very challenging, especially for videos that are low-light and noisy. The current best solution is to subsequently employ best models of video SR, denoising, and illumination enhancement, but doing so often lowers the image quality, due to the inconsistency between the models. This paper presents a new parametric representation called the Deep Parametric 3D Filters (DP3DF), which incorporates local spatiotemporal information to enable simultaneous denoising, illumination enhancement, and SR efficiently in a single encoder-and-decoder network. Also, a dynamic residual frame is jointly learned with the DP3DF via a shared backbone to further boost the SR quality. We performed extensive experiments, including a large-scale user study, to show our method's effectiveness. Our method consistently surpasses the best state-of-the-art methods on all the challenging real datasets with top PSNR and user ratings, yet having a very fast run time. The code is available at https://github.com/xiaogang00/DP3DF.

----

## [340] Inter-image Contrastive Consistency for Multi-Person Pose Estimation

**Authors**: *Xixia Xu, Yingguo Gao, Xingjia Pan, Ke Yan, Xiaoyu Chen, Qi Zou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25410](https://doi.org/10.1609/aaai.v37i3.25410)

**Abstract**:

Multi-person pose estimation (MPPE) has achieved impressive progress in recent years. However, due to the large variance of appearances among images or occlusions, the model can hardly learn consistent patterns enough, which leads to severe location jitter and missing issues. In this study, we propose a novel framework, termed Inter-image Contrastive consistency (ICON), to strengthen the keypoint consistency among images for MPPE. Concretely, we consider two-fold consistency constraints, which include single keypoint contrastive consistency (SKCC) and pair relation contrastive consistency (PRCC). The SKCC learns to strengthen the consistency of individual keypoints across images in the same category to improve the category-specific robustness. Only with SKCC, the model can effectively reduce location errors caused by large appearance variations, but remains challenging with extreme postures (e.g., occlusions) due to lack of relational guidance. Therefore, PRCC is proposed to strengthen the consistency of pair-wise joint relation between images to preserve the instructive relation. Cooperating with SKCC, PRCC further improves structure aware robustness in handling extreme postures. Extensive experiments on kinds of architectures across three datasets (i.e., MS-COCO, MPII, CrowdPose) show the proposed ICON achieves substantial improvements over baselines. Furthermore, ICON under the semi-supervised setup can obtain comparable results with the fully-supervised methods using only 30% labeled data.

----

## [341] DeMT: Deformable Mixer Transformer for Multi-Task Learning of Dense Prediction

**Authors**: *Yangyang Xu, Yibo Yang, Lefei Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25411](https://doi.org/10.1609/aaai.v37i3.25411)

**Abstract**:

Convolution neural networks (CNNs) and Transformers have their own advantages and both have been widely used for dense prediction in multi-task learning (MTL). Most of the current studies on MTL solely rely on CNN or Transformer. In this work, we present a novel MTL model by combining both merits of deformable CNN and query-based Transformer for multi-task learning of dense prediction. Our method, named DeMT, is based on a simple and effective encoder-decoder architecture (i.e., deformable mixer encoder and task-aware transformer decoder). First, the deformable mixer encoder contains two types of operators: the channel-aware mixing operator leveraged to allow communication among different channels (i.e., efficient channel location mixing), and the spatial-aware deformable operator with deformable convolution applied to efficiently sample more informative spatial locations (i.e., deformed features). Second, the task-aware transformer decoder consists of the task interaction block and task query block. The former is applied to capture task interaction features via self-attention. The latter leverages the deformed features and task-interacted features to generate the corresponding task-specific feature through a query-based Transformer for corresponding task predictions. Extensive experiments on two dense image prediction datasets, NYUD-v2 and PASCAL-Context, demonstrate that our model uses fewer GFLOPs and significantly outperforms current Transformer- and CNN-based competitive models on a variety of metrics. The code is available at https://github.com/yangyangxu0/DeMT.

----

## [342] VLTinT: Visual-Linguistic Transformer-in-Transformer for Coherent Video Paragraph Captioning

**Authors**: *Kashu Yamazaki, Khoa Vo, Quang Sang Truong, Bhiksha Raj, Ngan Le*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25412](https://doi.org/10.1609/aaai.v37i3.25412)

**Abstract**:

Video Paragraph Captioning aims to generate a multi-sentence description of an untrimmed video with multiple temporal event locations in a coherent storytelling. 
Following the human perception process, where the scene is effectively understood by decomposing it into visual (e.g. human, animal) and non-visual components (e.g. action, relations) under the mutual influence of vision and language, we first propose a visual-linguistic (VL) feature. In the proposed VL feature, the scene is modeled by three modalities including (i) a global visual environment; (ii) local visual main agents; (iii) linguistic scene elements. We then introduce an autoregressive Transformer-in-Transformer (TinT) to simultaneously capture the semantic coherence of intra- and inter-event contents within a video. Finally, we present a new VL contrastive loss function to guarantee the learnt embedding features are consistent with the captions semantics. Comprehensive experiments and extensive ablation studies on the ActivityNet Captions and YouCookII datasets show that the proposed Visual-Linguistic Transformer-in-Transform (VLTinT) outperforms previous state-of-the-art methods in terms of accuracy and diversity. The source code is made publicly available at: https://github.com/UARK-AICV/VLTinT.

----

## [343] Rethinking Disparity: A Depth Range Free Multi-View Stereo Based on Disparity

**Authors**: *Qingsong Yan, Qiang Wang, Kaiyong Zhao, Bo Li, Xiaowen Chu, Fei Deng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25413](https://doi.org/10.1609/aaai.v37i3.25413)

**Abstract**:

Existing learning-based multi-view stereo (MVS) methods rely on the depth range to build the 3D cost volume and may fail when the range is too large or unreliable. To address this problem, we propose a disparity-based MVS method based on the epipolar disparity flow (E-flow), called DispMVS, which infers the depth information from the pixel movement between two views. The core of DispMVS is to construct a 2D cost volume on the image plane along the epipolar line between each pair (between the reference image and several source images) for pixel matching and fuse uncountable depths triangulated from each pair by multi-view geometry to ensure multi-view consistency. To be robust, DispMVS starts from a randomly initialized depth map and iteratively refines the depth map with the help of the coarse-to-fine strategy. Experiments on DTUMVS and Tanks\&Temple datasets show that DispMVS is not sensitive to the depth range and achieves state-of-the-art results with lower GPU memory.

----

## [344] Video-Text Pre-training with Learned Regions for Retrieval

**Authors**: *Rui Yan, Mike Zheng Shou, Yixiao Ge, Jinpeng Wang, Xudong Lin, Guanyu Cai, Jinhui Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25414](https://doi.org/10.1609/aaai.v37i3.25414)

**Abstract**:

Video-Text pre-training aims at learning transferable representations from large-scale video-text pairs via aligning the semantics between visual and textual information. State-of-the-art approaches extract visual features from raw pixels in an end-to-end fashion. However, these methods operate at frame-level directly and thus overlook the spatio-temporal structure of objects in video, which yet has a strong synergy with nouns in textual descriptions. In this work, we propose a simple yet effective module for video-text representation learning, namely RegionLearner, which can take into account the structure of objects during pre-training on large-scale video-text pairs. Given a video, our module (1) first quantizes continuous visual features via clustering patch-features into the same cluster according to content similarity, then (2) generates learnable masks to aggregate fragmentary features into regions with complete semantics, and finally (3) models the spatio-temporal dependencies between different semantic regions. In contrast to using off-the-shelf object detectors, our proposed module does not require explicit supervision and is much more computationally efficient. We pre-train the proposed approach on the public WebVid2M and CC3M datasets. Extensive evaluations on four downstream video-text retrieval benchmarks clearly demonstrate the effectiveness of our RegionLearner.

----

## [345] DesNet: Decomposed Scale-Consistent Network for Unsupervised Depth Completion

**Authors**: *Zhiqiang Yan, Kun Wang, Xiang Li, Zhenyu Zhang, Jun Li, Jian Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25415](https://doi.org/10.1609/aaai.v37i3.25415)

**Abstract**:

Unsupervised depth completion aims to recover dense depth from the sparse one without using the ground-truth annotation. Although depth measurement obtained from LiDAR is usually sparse, it contains valid and real distance information, i.e., scale-consistent absolute depth values. Meanwhile, scale-agnostic counterparts seek to estimate relative depth and have achieved impressive performance. To leverage both the inherent characteristics, we thus suggest to model scale-consistent depth upon unsupervised scale-agnostic frameworks. Specifically, we propose the decomposed scale-consistent learning (DSCL) strategy, which disintegrates the absolute depth into relative depth prediction and global scale estimation, contributing to individual learning benefits. But unfortunately, most existing unsupervised scale-agnostic frameworks heavily suffer from depth holes due to the extremely sparse depth input and weak supervisory signal. To tackle this issue, we introduce the global depth guidance (GDG) module, which attentively propagates dense depth reference into the sparse target via novel dense-to-sparse attention. Extensive experiments show the superiority of our method on outdoor KITTI, ranking 1st and outperforming the best KBNet more than 12% in RMSE. Additionally, our approach achieves state-of-the-art performance on indoor NYUv2 benchmark as well.

----

## [346] Self-Supervised Video Representation Learning via Latent Time Navigation

**Authors**: *Di Yang, Yaohui Wang, Quan Kong, Antitza Dantcheva, Lorenzo Garattoni, Gianpiero Francesca, François Brémond*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25416](https://doi.org/10.1609/aaai.v37i3.25416)

**Abstract**:

Self-supervised video representation learning aimed at maximizing similarity between different temporal segments of one video, in order to enforce feature persistence over time. This leads to loss of pertinent information related to temporal relationships, rendering actions such as `enter' and `leave' to be indistinguishable. To mitigate this limitation, we propose Latent Time Navigation (LTN), a time parameterized contrastive learning strategy that is streamlined to capture fine-grained motions. Specifically, we maximize the representation similarity between different video segments from one video, while maintaining their representations time-aware along a subspace of the latent representation code including an orthogonal basis to represent temporal changes. Our extensive experimental analysis suggests that learning video representations by LTN consistently improves performance of action classification in fine-grained and human-oriented tasks (e.g., on Toyota Smarthome dataset). In addition, we demonstrate that our proposed model, when pre-trained on Kinetics-400, generalizes well onto the unseen real world video benchmark datasets UCF101 and HMDB51, achieving state-of-the-art performance in action recognition.

----

## [347] One-Shot Replay: Boosting Incremental Object Detection via Retrospecting One Object

**Authors**: *Dongbao Yang, Yu Zhou, Xiaopeng Hong, Aoting Zhang, Weiping Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25417](https://doi.org/10.1609/aaai.v37i3.25417)

**Abstract**:

Modern object detectors are ill-equipped to incrementally learn new emerging object classes over time due to the well-known phenomenon of catastrophic forgetting. Due to data privacy or limited storage, few or no images of the old data can be stored for replay. In this paper, we design a novel One-Shot Replay (OSR) method for incremental object detection, which is an augmentation-based method. Rather than storing original images, only one object-level sample for each old class is stored to reduce memory usage significantly, and we find that copy-paste is a harmonious way to replay for incremental object detection. In the incremental learning procedure, diverse augmented samples with co-occurrence of old and new objects to existing training data are generated. To introduce more variants for objects of old classes, we propose two augmentation modules. The object augmentation module aims to enhance the ability of the detector to perceive potential unknown objects. The feature augmentation module explores the relations between old and new classes and augments the feature space via analogy. Extensive experimental results on VOC2007 and COCO demonstrate that OSR can outperform the state-of-the-art incremental object detection methods without using extra wild data.

----

## [348] Video Event Extraction via Tracking Visual States of Arguments

**Authors**: *Guang Yang, Manling Li, Jiajie Zhang, Xudong Lin, Heng Ji, Shih-Fu Chang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25418](https://doi.org/10.1609/aaai.v37i3.25418)

**Abstract**:

Video event extraction aims to detect salient events from a video and identify the arguments for each event as well as their semantic roles. Existing methods focus on capturing the overall visual scene of each frame, ignoring fine-grained argument-level information. Inspired by the definition of events as changes of  states, we propose a novel framework to detect video events by tracking the changes in the visual states of all involved arguments, which are expected to provide the most informative evidence for the extraction of video events. In order to capture the visual state changes of arguments, we decompose them into changes in pixels within objects, displacements of objects, and interactions among multiple arguments. We further propose Object State Embedding, Object Motion-aware Embedding and Argument Interaction Embedding to encode and track these changes respectively. Experiments on various video event extraction tasks demonstrate significant improvements compared to state-of-the-art models. In particular, on verb classification, we achieve 3.49% absolute gains (19.53% relative gains) in F1@5 on Video Situation Recognition. Our Code is publicly available at https://github.com/Shinetism/VStates for research purposes.

----

## [349] CoMAE: Single Model Hybrid Pre-training on Small-Scale RGB-D Datasets

**Authors**: *Jiange Yang, Sheng Guo, Gangshan Wu, Limin Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25419](https://doi.org/10.1609/aaai.v37i3.25419)

**Abstract**:

Current RGB-D scene recognition approaches often train two standalone backbones for RGB and depth modalities with the same Places or ImageNet pre-training. However, the pre-trained depth network is still biased by RGB-based models which may result in a suboptimal solution. In this paper, we present a single-model self-supervised hybrid pre-training framework for RGB and depth modalities, termed as CoMAE. Our CoMAE presents a curriculum learning strategy to unify the two popular self-supervised representation learning algorithms: contrastive learning and masked image modeling.  Specifically, we first build a patch-level alignment task to pre-train a single encoder shared by two modalities via cross-modal contrastive learning. Then, the pre-trained contrastive encoder is passed to a multi-modal masked autoencoder to capture the finer context features from a generative perspective.  In addition, our single-model design without requirement of fusion module is very flexible and robust to generalize to unimodal scenario in both training and testing phases.  Extensive experiments on SUN RGB-D and NYUDv2 datasets demonstrate the effectiveness of our CoMAE for RGB and depth representation learning.  In addition, our experiment results reveal that CoMAE is a data-efficient representation learner. Although we only use the small-scale and unlabeled training set for pre-training, our CoMAE pre-trained models are still competitive to the state-of-the-art methods with extra large-scale and supervised RGB dataset pre-training. Code will be released at https://github.com/MCG-NJU/CoMAE.

----

## [350] Self-Asymmetric Invertible Network for Compression-Aware Image Rescaling

**Authors**: *Jinhai Yang, Mengxi Guo, Shijie Zhao, Junlin Li, Li Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25420](https://doi.org/10.1609/aaai.v37i3.25420)

**Abstract**:

High-resolution (HR) images are usually downscaled to low-resolution (LR) ones for better display and afterward upscaled back to the original size to recover details. Recent work in image rescaling formulates downscaling and upscaling as a unified task and learns a bijective mapping between HR and LR via invertible networks. However, in real-world applications (e.g., social media), most images are compressed for transmission. Lossy compression will lead to irreversible information loss on LR images, hence damaging the inverse upscaling procedure and degrading the reconstruction accuracy. In this paper, we propose the Self-Asymmetric Invertible Network (SAIN) for compression-aware image rescaling. To tackle the distribution shift, we first develop an end-to-end asymmetric framework with two separate bijective mappings for high-quality and compressed LR images, respectively. Then, based on empirical analysis of this framework, we model the distribution of the lost information (including downscaling and compression) using isotropic Gaussian mixtures and propose the Enhanced Invertible Block to derive high-quality/compressed LR images in one forward pass. Besides, we design a set of losses to regularize the learned LR images and enhance the invertibility. Extensive experiments demonstrate the consistent improvements of SAIN across various image rescaling datasets in terms of both quantitative and qualitative evaluation under standard image compression formats (i.e., JPEG and WebP). Code is available at https://github.com/yang-jin-hai/SAIN.

----

## [351] Stop-Gradient Softmax Loss for Deep Metric Learning

**Authors**: *Lu Yang, Peng Wang, Yanning Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25421](https://doi.org/10.1609/aaai.v37i3.25421)

**Abstract**:

Deep metric learning aims to learn a feature space that models the similarity between images, and feature normalization is a critical step for boosting performance. However directly optimizing L2-normalized softmax loss cause the network to fail to converge. Therefore some SOTA approaches appends a scale layer after the inner product to relieve the convergence problem, but it incurs a new problem that it's difficult to learn the best scaling parameters. In this letter, we look into the characteristic of softmax-based approaches and propose a novel learning objective function Stop-Gradient Softmax Loss (SGSL) to solve the convergence problem in softmax-based deep metric learning with L2-normalization. In addition, we found a useful trick named Remove the last BN-ReLU (RBR). It removes the last BN-ReLU in the backbone to reduce the learning burden of the model.
Experimental results on four fine-grained image retrieval benchmarks show that our proposed approach outperforms most existing approaches, i.e., our approach achieves 75.9% on CUB-200-2011, 94.7% on CARS196 and 83.1% on SOP which outperforms other approaches at least 1.7%, 2.9% and 1.7% on Recall@1.

----

## [352] Local Path Integration for Attribution

**Authors**: *Peiyu Yang, Naveed Akhtar, Zeyi Wen, Ajmal Mian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25422](https://doi.org/10.1609/aaai.v37i3.25422)

**Abstract**:

Path attribution methods are a popular tool to interpret a visual model's prediction on an input. They integrate model gradients for the input features over a path defined between the input and a reference, thereby satisfying certain desirable theoretical properties. However, their reliability hinges on the choice of the reference. Moreover, they do not exhibit weak dependence on the input, which leads to counter-intuitive feature attribution mapping. We show that path-based attribution can account for the weak dependence property by choosing the reference from the local distribution of the input. We devise a method to identify the local input distribution and propose a technique to stochastically integrate the model gradients over the paths defined by the references sampled from that distribution. Our local path integration (LPI) method is found to consistently outperform existing path attribution techniques when evaluated on deep visual models. Contributing to the ongoing search of reliable evaluation metrics for the interpretation methods, we also introduce DiffID metric that uses the relative difference between insertion and deletion games to alleviate the distribution shift problem faced by existing metrics. Our code is available at https://github.com/ypeiyu/LPI.

----

## [353] Spatiotemporal Deformation Perception for Fisheye Video Rectification

**Authors**: *Shangrong Yang, Chunyu Lin, Kang Liao, Yao Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25423](https://doi.org/10.1609/aaai.v37i3.25423)

**Abstract**:

Although the distortion correction of fisheye images has been extensively studied, the correction of fisheye videos is still an elusive challenge. For different frames of the fisheye video, the existing image correction methods ignore the correlation of sequences, resulting in temporal jitter in the corrected video. To solve this problem, we propose a temporal weighting scheme to get a plausible global optical flow, which mitigates the jitter effect by progressively reducing the weight of frames. Subsequently, we observe that the inter-frame optical flow of the video is facilitated to perceive the local spatial deformation of the fisheye video. Therefore, we derive the spatial deformation through the flows of fisheye and distorted-free videos, thereby enhancing the local accuracy of the predicted result. However, the independent correction for each frame disrupts the temporal correlation. Due to the property of fisheye video, a distorted moving object may be able to find its distorted-free pattern at another moment. To this end, a temporal deformation aggregator is designed to reconstruct the deformation correlation between frames and provide a reliable global feature. Our method achieves an end-to-end correction and demonstrates superiority in correction quality and stability compared with the SOTA correction methods.

----

## [354] Contrastive Multi-Task Dense Prediction

**Authors**: *Siwei Yang, Hanrong Ye, Dan Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25424](https://doi.org/10.1609/aaai.v37i3.25424)

**Abstract**:

This paper targets the problem of multi-task dense prediction
which aims to achieve simultaneous learning and inference on
a bunch of multiple dense prediction tasks in a single framework. A core objective in design is how to effectively model
cross-task interactions to achieve a comprehensive improvement on different tasks based on their inherent complementarity and consistency. Existing works typically design extra
expensive distillation modules to perform explicit interaction
computations among different task-specific features in both
training and inference, bringing difficulty in adaptation for
different task sets, and reducing efficiency due to clearly increased size of multi-task models. In contrast, we introduce
feature-wise contrastive consistency into modeling the cross-task interactions for multi-task dense prediction. We propose
a novel multi-task contrastive regularization method based on
the consistency to effectively boost the representation learning of the different sub-tasks, which can also be easily generalized to different multi-task dense prediction frameworks,
and costs no additional computation in the inference. Extensive experiments on two challenging datasets (i.e. NYUD-v2
and Pascal-Context) clearly demonstrate the superiority of the
proposed multi-task contrastive learning approach for dense
predictions, establishing new state-of-the-art performances.

----

## [355] AutoStegaFont: Synthesizing Vector Fonts for Hiding Information in Documents

**Authors**: *Xi Yang, Jie Zhang, Han Fang, Chang Liu, Zehua Ma, Weiming Zhang, Nenghai Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25425](https://doi.org/10.1609/aaai.v37i3.25425)

**Abstract**:

Hiding information in text documents has been a hot topic recently, with the most typical schemes of utilizing fonts. By constructing several fonts with similar appearances, information can be effectively represented and embedded in documents. However, due to the unstructured characteristic, font vectors are more difficult to synthesize than font images. Existing methods mainly use handcrafted features to design the fonts manually, which is time-consuming and labor-intensive. Moreover, due to the diversity of fonts, handcrafted features are not generalizable to different fonts. Besides, in practice, since documents might be distorted through transmission, ensuring extractability under distortions is also an important requirement. Therefore, three requirements are imposed on vector font generation in this domain: automaticity, generalizability, and robustness. However, none of the existing methods can satisfy these requirements well and simultaneously.
To satisfy the above requirements, we propose AutoStegaFont, an automatic vector font synthesis scheme for hiding information in documents. Specifically, we design a two-stage and dual-modality learning framework. In the first stage, we jointly train an encoder and a decoder to invisibly encode the font images with different information. To ensure robustness, we target designing a noise layer to work with the encoder and decoder during training. In the second stage, we employ a differentiable rasterizer to establish a connection between the image and the vector modality. Then, we design an optimization algorithm to convey the information from the encoded image to the corresponding vector. Thus the encoded font vectors can be automatically generated. Extensive experiments demonstrate the superior performance of our scheme in automatically synthesizing vector fonts for hiding information in documents, with robustness to distortions caused by low-resolution screenshots, printing, and photography. Besides, the proposed framework has better generalizability to fonts with diverse styles and languages.

----

## [356] Towards Global Video Scene Segmentation with Context-Aware Transformer

**Authors**: *Yang Yang, Yurui Huang, Weili Guo, Baohua Xu, Dingyin Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25426](https://doi.org/10.1609/aaai.v37i3.25426)

**Abstract**:

Videos such as movies or TV episodes usually need to divide the long storyline into cohesive units, i.e., scenes, to facilitate the understanding of video semantics. The key challenge lies in finding the boundaries of scenes by comprehensively considering the complex temporal structure and semantic information. To this end, we introduce a novel Context-Aware Transformer (CAT) with a self-supervised learning framework to learn high-quality shot representations, for generating well-bounded scenes. More specifically, we design the CAT with local-global self-attentions, which can effectively consider both the long-term and short-term context to improve the shot encoding. For training the CAT, we adopt the self-supervised learning schema. Firstly, we leverage shot-to-scene level pretext tasks to facilitate the pre-training with pseudo boundary, which guides CAT to learn the discriminative shot representations that maximize intra-scene similarity and inter-scene discrimination in an unsupervised manner. Then, we transfer contextual representations for fine-tuning the CAT with supervised data, which encourages CAT to accurately detect the boundary for scene segmentation. As a result, CAT is able to learn the context-aware shot representations and provides global guidance for scene segmentation. Our empirical analyses show that CAT can achieve state-of-the-art performance when conducting the scene segmentation task on the MovieNet dataset, e.g., offering 2.15 improvements on AP.

----

## [357] Low-Light Image Enhancement Network Based on Multi-Scale Feature Complementation

**Authors**: *Yong Yang, Wenzhi Xu, Shuying Huang, Weiguo Wan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25427](https://doi.org/10.1609/aaai.v37i3.25427)

**Abstract**:

Images captured in low-light environments have problems of insufficient brightness and low contrast, which will affect subsequent image processing tasks. Although most current enhancement methods can obtain high-contrast images, they still suffer from noise amplification and color distortion. To address these issues, this paper proposes a low-light image enhancement network based on multi-scale feature complementation (LIEN-MFC), which is a U-shaped encoder-decoder network supervised by multiple images of different scales. In the encoder, four feature extraction branches are constructed to extract features of low-light images at different scales. In the decoder, to ensure the integrity of the learned features at each scale, a feature supplementary fusion module (FSFM) is proposed to complement and integrate features from different branches of the encoder and decoder. In addition, a feature restoration module (FRM) and an image reconstruction module (IRM) are built in each branch to reconstruct the restored features and output enhanced images. To better train the network, a joint loss function is defined, in which a discriminative loss term is designed to ensure that the enhanced results better meet the visual properties of the human eye. Extensive experiments on benchmark datasets show that the proposed method outperforms some state-of-the-art methods subjectively and objectively.

----

## [358] Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation

**Authors**: *Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, Philip H. S. Torr*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25428](https://doi.org/10.1609/aaai.v37i3.25428)

**Abstract**:

Referring image segmentation segments an image from a language expression. With the aim of producing high-quality masks, existing methods often adopt iterative learning approaches that rely on RNNs or stacked attention layers to refine vision-language features. Despite their complexity, RNN-based methods are subject to specific encoder choices, while attention-based methods offer limited gains. In this work, we introduce a simple yet effective alternative for progressively learning discriminative multi-modal features. The core idea of our approach is to leverage a continuously updated query as the representation of the target object and at each iteration, strengthen multi-modal features strongly correlated to the query while weakening less related ones. As the query is initialized by language features and successively updated by object features, our algorithm gradually shifts from being localization-centric to segmentation-centric. This strategy enables the incremental recovery of missing object parts and/or removal of extraneous parts through iteration. Compared to its counterparts, our method is more versatile—it can be plugged into prior arts straightforwardly and consistently bring improvements. Experimental results on the challenging datasets of RefCOCO, RefCOCO+, and G-Ref demonstrate its advantage with respect to the state-of-the-art methods.

----

## [359] LidarMultiNet: Towards a Unified Multi-Task Network for LiDAR Perception

**Authors**: *Dongqiangzi Ye, Zixiang Zhou, Weijia Chen, Yufei Xie, Yu Wang, Panqu Wang, Hassan Foroosh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25429](https://doi.org/10.1609/aaai.v37i3.25429)

**Abstract**:

LiDAR-based 3D object detection, semantic segmentation, and panoptic segmentation are usually implemented in specialized networks with distinctive architectures that are difficult to adapt to each other. This paper presents LidarMultiNet, a LiDAR-based multi-task network that unifies these three major LiDAR perception tasks. Among its many benefits, a multi-task network can reduce the overall cost by sharing weights and computation among multiple tasks. However, it typically underperforms compared to independently combined single-task models. The proposed LidarMultiNet aims to bridge the performance gap between the multi-task network and multiple single-task networks. At the core of LidarMultiNet is a strong 3D voxel-based encoder-decoder architecture with a Global Context Pooling (GCP) module extracting global contextual features from a LiDAR frame. Task-specific heads are added on top of the network to perform the three LiDAR perception tasks. More tasks can be implemented simply by adding new task-specific heads while introducing little additional cost. A second stage is also proposed to refine the first-stage segmentation and generate accurate panoptic segmentation results. LidarMultiNet is extensively tested on both Waymo Open Dataset and nuScenes dataset, demonstrating for the first time that major LiDAR perception tasks can be unified in a single strong network that is trained end-to-end and achieves state-of-the-art performance. Notably, LidarMultiNet reaches the official 1 place in the Waymo Open Dataset 3D semantic segmentation challenge 2022 with the highest mIoU and the best accuracy for most of the 22 classes on the test set, using only LiDAR points as input. It also sets the new state-of-the-art for a single model on the Waymo 3D object detection benchmark and three nuScenes benchmarks.

----

## [360] DPText-DETR: Towards Better Scene Text Detection with Dynamic Points in Transformer

**Authors**: *Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Bo Du, Dacheng Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25430](https://doi.org/10.1609/aaai.v37i3.25430)

**Abstract**:

Recently, Transformer-based methods, which predict polygon points or Bezier curve control points for localizing texts, are popular in scene text detection. However, these methods built upon detection transformer framework might achieve sub-optimal training efficiency and performance due to coarse positional query modeling. In addition, the point label form exploited in previous works implies the reading order of humans, which impedes the detection robustness from our observation. To address these challenges, this paper proposes a concise Dynamic Point Text DEtection TRansformer network, termed DPText-DETR. In detail, DPText-DETR directly leverages explicit point coordinates to generate position queries and dynamically updates them in a progressive way. Moreover, to improve the spatial inductive bias of non-local self-attention in Transformer, we present an Enhanced Factorized Self-Attention module which provides point queries within each instance with circular shape guidance. Furthermore, we design a simple yet effective positional label form to tackle the side effect of the previous form. To further evaluate the impact of different label forms on the detection robustness in real-world scenario, we establish an Inverse-Text test set containing 500 manually labeled images. Extensive experiments prove the high training efficiency, robustness, and state-of-the-art performance of our method on popular benchmarks. The code and the Inverse-Text test set are available at https://github.com/ymy-k/DPText-DETR.

----

## [361] Learning Second-Order Attentive Context for Efficient Correspondence Pruning

**Authors**: *Xinyi Ye, Weiyue Zhao, Hao Lu, Zhiguo Cao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25431](https://doi.org/10.1609/aaai.v37i3.25431)

**Abstract**:

Correspondence pruning aims to search consistent correspondences (inliers) from a set of putative correspondences. It is challenging because of the disorganized spatial distribution of numerous outliers, especially when putative correspondences are largely dominated by outliers. It's more challenging to ensure effectiveness while maintaining efficiency. In this paper, we propose an effective and efficient method for correspondence pruning. Inspired by the success of attentive context in correspondence problems, we first extend the attentive context to the first-order attentive context and then introduce the idea of attention in attention (ANA) to model second-order attentive context for correspondence pruning. Compared with first-order attention that focuses on feature-consistent context, second-order attention dedicates to attention weights itself and provides an additional source to encode consistent context from the attention map. For efficiency, we derive two approximate formulations for the naive implementation of second-order attention to optimize the cubic complexity to linear complexity, such that second-order attention can be used with negligible computational overheads. We further implement our formulations in a second-order context layer and then incorporate the layer in an ANA block. Extensive experiments demonstrate that our method is effective and efficient in pruning outliers, especially in high-outlier-ratio cases. Compared with the state-of-the-art correspondence pruning approach LMCNet, our method  runs 14 times faster while maintaining a competitive accuracy.

----

## [362] Infusing Definiteness into Randomness: Rethinking Composition Styles for Deep Image Matting

**Authors**: *Zixuan Ye, Yutong Dai, Chaoyi Hong, Zhiguo Cao, Hao Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25432](https://doi.org/10.1609/aaai.v37i3.25432)

**Abstract**:

We study the composition style in deep image matting, a notion that characterizes a data generation flow on how to exploit limited foregrounds and random backgrounds to form a training dataset. Prior art executes this flow in a completely random manner by simply going through the foreground pool or by optionally combining two foregrounds before foreground-background composition. In this work, we first show that naive foreground combination can be problematic and therefore derive an alternative formulation to reasonably combine foregrounds. Our second contribution is an observation that matting performance can benefit from a certain occurrence frequency of combined foregrounds and their associated source foregrounds during training. Inspired by this, we introduce a novel composition style that binds the source and combined foregrounds in a definite triplet. In addition, we also find that different orders of foreground combination lead to different foreground patterns, which further inspires a quadruplet-based composition style. Results under controlled experiments on four matting baselines show that our composition styles outperform existing ones and invite consistent performance improvement on both composited and real-world datasets. Code is available at: https://github.com/coconuthust/composition_styles

----

## [363] Can We Find Strong Lottery Tickets in Generative Models?

**Authors**: *Sangyeop Yeo, Yoojin Jang, Jy-yong Sohn, Dongyoon Han, Jaejun Yoo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25433](https://doi.org/10.1609/aaai.v37i3.25433)

**Abstract**:

Yes. In this paper, we investigate strong lottery tickets in generative models, the subnetworks that achieve good generative performance without any weight update. Neural network pruning is considered the main cornerstone of model compression for reducing the costs of computation and memory. Unfortunately, pruning a generative model has not been extensively explored, and all existing pruning algorithms suffer from excessive weight-training costs, performance degradation, limited generalizability, or complicated training. To address these problems, we propose to find a strong lottery ticket via moment-matching scores. Our experimental results show that the discovered subnetwork can perform similarly or better than the trained dense model even when only 10% of the weights remain. To the best of our knowledge, we are the first to show the existence of strong lottery tickets in generative models and provide an algorithm to find it stably. Our code and supplementary materials are publicly available at https://lait-cvlab.github.io/SLT-in-Generative-Models/.

----

## [364] Class-Independent Regularization for Learning with Noisy Labels

**Authors**: *Rumeng Yi, Dayan Guan, Yaping Huang, Shijian Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25434](https://doi.org/10.1609/aaai.v37i3.25434)

**Abstract**:

Training deep neural networks (DNNs) with noisy labels often leads to poorly generalized models as DNNs tend to memorize the noisy labels in training. Various strategies have been developed for improving sample selection precision and mitigating the noisy label memorization issue. However, most existing works adopt a class-dependent softmax classifier that is vulnerable to noisy labels by entangling the classification of multi-class features. This paper presents a class-independent regularization (CIR) method that can effectively alleviate the negative impact of noisy labels in DNN training. CIR regularizes the class-dependent softmax classifier by introducing multi-binary classifiers each of which takes care of one class only. Thanks to its class-independent nature, CIR is tolerant to noisy labels as misclassification by one binary classifier does not affect others. For effective training of CIR, we design a heterogeneous adaptive co-teaching strategy that forces the class-independent and class-dependent classifiers to focus on sample selection and image classification, respectively, in a cooperative manner. Extensive experiments show that CIR achieves superior performance consistently across multiple benchmarks with both synthetic and real images. Code is
available at https://github.com/RumengYi/CIR.

----

## [365] Unbiased Heterogeneous Scene Graph Generation with Relation-Aware Message Passing Neural Network

**Authors**: *Kanghoon Yoon, Kibum Kim, Jinyoung Moon, Chanyoung Park*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25435](https://doi.org/10.1609/aaai.v37i3.25435)

**Abstract**:

Recent scene graph generation (SGG) frameworks have focused on learning complex relationships among multiple objects in an image. Thanks to the nature of the message passing neural network (MPNN) that models high-order interactions between objects and their neighboring objects, they are dominant representation learning modules for SGG. However, existing MPNN-based frameworks assume the scene graph as a homogeneous graph, which restricts the context-awareness of visual relations between objects. That is, they overlook the fact that the relations tend to be highly dependent on the objects with which the relations are associated. In this paper, we propose an unbiased heterogeneous scene graph generation (HetSGG) framework that captures
relation-aware context using message passing neural networks. We devise a novel message passing layer, called relation-aware message passing neural network (RMP), that aggregates the contextual information of an image considering the predicate type between objects. Our extensive evaluations demonstrate that HetSGG outperforms state-of-the-art methods, especially outperforming on tail predicate classes. The source code for HetSGG is available at https://github.com/KanghoonYoon/hetsgg-torch

----

## [366] Lifelong Person Re-identification via Knowledge Refreshing and Consolidation

**Authors**: *Chunlin Yu, Ye Shi, Zimo Liu, Shenghua Gao, Jingya Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25436](https://doi.org/10.1609/aaai.v37i3.25436)

**Abstract**:

Lifelong person re-identification (LReID) is in significant demand for real-world development as a large amount of ReID data is captured from diverse locations over time and cannot be accessed at once inherently. However, a key challenge for LReID is how to incrementally preserve old knowledge and gradually add new capabilities to the system. Unlike most existing LReID methods, which mainly focus on dealing with catastrophic forgetting, our focus is on a more challenging problem, which is, not only trying to reduce the forgetting on old tasks but also aiming to improve the model performance on both new and old tasks during the lifelong learning process. Inspired by the biological process of human cognition where the somatosensory neocortex and the hippocampus work together in memory consolidation, we formulated a model called Knowledge Refreshing and Consolidation (KRC) that achieves both positive forward and backward transfer. More specifically, a knowledge refreshing scheme is incorporated with the knowledge rehearsal mechanism to enable bi-directional knowledge transfer by introducing a dynamic memory model and an adaptive working model. Moreover, a knowledge consolidation scheme operating on the dual space further improves model stability over the long-term. Extensive evaluations show KRC’s superiority over the state-of-the-art LReID methods with challenging pedestrian benchmarks.  Code is available at https://github.com/cly234/LReID-KRKC.

----

## [367] Generalizing Multiple Object Tracking to Unseen Domains by Introducing Natural Language Representation

**Authors**: *En Yu, Songtao Liu, Zhuoling Li, Jinrong Yang, Zeming Li, Shoudong Han, Wenbing Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25437](https://doi.org/10.1609/aaai.v37i3.25437)

**Abstract**:

Although existing multi-object tracking (MOT) algorithms have obtained competitive performance on various benchmarks, almost all of them train and validate models on the same domain. The domain generalization problem of MOT is hardly studied. To bridge this gap, we first draw the observation that the high-level information contained in natural language is domain invariant to different tracking domains. Based on this observation, we propose to introduce natural language representation into visual MOT models for boosting the domain generalization ability. However, it is infeasible to label every tracking target with a textual description. To tackle this problem, we design two modules, namely visual context prompting (VCP) and visual-language mixing (VLM). Specifically, VCP generates visual prompts based on the input frames. VLM joints the information in the generated visual prompts and the textual prompts from a pre-defined Trackbook to obtain instance-level pseudo textual description, which is domain invariant to different tracking scenes. Through training models on MOT17 and validating them on MOT20, we observe that the pseudo textual descriptions generated by our proposed modules improve the generalization performance of query-based trackers by large margins.

----

## [368] Rethinking Rotation Invariance with Point Cloud Registration

**Authors**: *Jianhui Yu, Chaoyi Zhang, Weidong Cai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25438](https://doi.org/10.1609/aaai.v37i3.25438)

**Abstract**:

Recent investigations on rotation invariance for 3D point clouds have been devoted to devising rotation-invariant feature descriptors or learning canonical spaces where objects are semantically aligned. Examinations of learning frameworks for invariance have seldom been looked into. In this work, we review rotation invariance (RI) in terms of point cloud registration (PCR) and propose an effective framework for rotation invariance learning via three sequential stages, namely rotation-invariant shape encoding, aligned feature integration, and deep feature registration. We first encode shape descriptors constructed with respect to reference frames defined over different scales, e.g., local patches and global topology, to generate rotation-invariant latent shape codes. Within the integration stage, we propose an Aligned Integration Transformer (AIT) to produce a discriminative feature representation by integrating point-wise self- and cross-relations established within the shape codes. Meanwhile, we adopt rigid transformations between reference frames to align the shape codes for feature consistency across different scales. Finally, the deep integrated feature is registered to both rotation-invariant shape codes to maximize their feature similarities, such that rotation invariance of the integrated feature is preserved and shared semantic information is implicitly extracted from shape codes. Experimental results on 3D shape classification, part segmentation, and retrieval tasks prove the feasibility of our framework. Our project page is released at: https://rotation3d.github.io/.

----

## [369] Frame-Level Label Refinement for Skeleton-Based Weakly-Supervised Action Recognition

**Authors**: *Qing Yu, Kent Fujiwara*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25439](https://doi.org/10.1609/aaai.v37i3.25439)

**Abstract**:

In recent years, skeleton-based action recognition has achieved remarkable performance in understanding human motion from sequences of skeleton data, which is an important medium for synthesizing realistic human movement in various applications. However, existing methods assume that each action clip is manually trimmed to contain one specific action, which requires a significant amount of effort for annotation. To solve this problem, we consider a novel problem of skeleton-based weakly-supervised temporal action localization (S-WTAL), where we need to recognize and localize human action segments in untrimmed skeleton videos given only the video-level labels. Although this task is challenging due to the sparsity of skeleton data and the lack of contextual clues from interaction with other objects and the environment, we present a frame-level label refinement framework based on a spatio-temporal graph convolutional network (ST-GCN) to overcome these difficulties. We use multiple instance learning (MIL) with video-level labels to generate the frame-level predictions. Inspired by advances in handling the noisy label problem, we introduce a label cleaning strategy of the frame-level pseudo labels to guide the learning process. The network parameters and the frame-level predictions are alternately updated to obtain the final results. We extensively evaluate the effectiveness of our learning approach on skeleton-based action recognition benchmarks. The state-of-the-art experimental results demonstrate that the proposed method can recognize and localize action segments of the skeleton data.

----

## [370] Recurrent Structure Attention Guidance for Depth Super-resolution

**Authors**: *Jiayi Yuan, Haobo Jiang, Xiang Li, Jianjun Qian, Jun Li, Jian Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25440](https://doi.org/10.1609/aaai.v37i3.25440)

**Abstract**:

Image guidance is an effective strategy for depth super-resolution. Generally, most existing methods employ hand-crafted operators to decompose the high-frequency (HF) and low-frequency (LF) ingredients from low-resolution depth maps and guide the HF ingredients by directly concatenating them with image features. However, the hand-designed operators usually cause inferior HF maps (e.g., distorted or structurally missing) due to the diverse appearance of complex depth maps. Moreover, the direct concatenation often results in weak guidance because not all image features have a positive effect on the HF maps. In this paper, we develop a recurrent structure attention guided (RSAG) framework, consisting of two important parts. First, we introduce a deep contrastive network with multi-scale filters for adaptive frequency-domain separation, which adopts contrastive networks from large filters to small ones to calculate the pixel contrasts for adaptive high-quality HF predictions. Second, instead of the coarse concatenation guidance, we propose a recurrent structure attention block, which iteratively utilizes the latest depth estimation and the image features to jointly select clear patterns and boundaries, aiming at providing refined guidance for accurate depth recovery. In addition, we fuse the features of HF maps to enhance the edge structures in the decomposed LF maps. Extensive experiments show that our approach obtains superior performance compared with state-of-the-art depth super-resolution methods. Our code is available at: https://github.com/Yuanjiayii/DSR-RSAG.

----

## [371] Structure Flow-Guided Network for Real Depth Super-resolution

**Authors**: *Jiayi Yuan, Haobo Jiang, Xiang Li, Jianjun Qian, Jun Li, Jian Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25441](https://doi.org/10.1609/aaai.v37i3.25441)

**Abstract**:

Real depth super-resolution (DSR), unlike synthetic settings, is a challenging task due to the structural distortion and the edge noise caused by the natural degradation in real-world low-resolution (LR) depth maps. These defeats result in significant structure inconsistency between the depth map and the RGB guidance, which potentially confuses the RGB-structure guidance and thereby degrades the DSR quality. In this paper, we propose a novel structure flow-guided DSR framework, where a cross-modality flow map is learned to guide the RGB-structure information transferring for precise depth upsampling. Specifically, our framework consists of a cross-modality flow-guided upsampling network (CFUNet) and a flow-enhanced pyramid edge attention network (PEANet). CFUNet contains a trilateral self-attention module combining both the geometric and semantic correlations for reliable cross-modality flow learning. Then, the learned flow maps are combined with the grid-sampling mechanism for coarse high-resolution (HR) depth prediction. PEANet targets at integrating the learned flow map as the edge attention into a pyramid network to hierarchically learn the edge-focused guidance feature for depth edge refinement. Extensive experiments on real and synthetic DSR datasets verify that our approach achieves excellent performance compared to state-of-the-art methods. Our code is available at: https://github.com/Yuanjiayii/DSR-SFG.

----

## [372] Pseudo Label-Guided Model Inversion Attack via Conditional Generative Adversarial Network

**Authors**: *Xiaojian Yuan, Kejiang Chen, Jie Zhang, Weiming Zhang, Nenghai Yu, Yang Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25442](https://doi.org/10.1609/aaai.v37i3.25442)

**Abstract**:

Model inversion (MI) attacks have raised increasing concerns about privacy, which can reconstruct training data from public models. Indeed, MI attacks can be formalized as an optimization problem that seeks private data in a certain space. Recent MI attacks leverage a generative adversarial network (GAN) as an image prior to narrow the search space, and can successfully reconstruct even the high-dimensional data (e.g., face images). However, these generative MI attacks do not fully exploit the potential capabilities of the target model, still leading to a vague and coupled search space, i.e., different classes of images are coupled in the search space. Besides, the widely used cross-entropy loss in these attacks suffers from gradient vanishing. To address these problems, we propose Pseudo Label-Guided MI (PLG-MI) attack via conditional GAN (cGAN). At first, a top-n selection strategy is proposed to provide pseudo-labels for public data, and use pseudo-labels to guide the training of the cGAN. In this way, the search space is decoupled for different classes of images. Then a max-margin loss is introduced to improve the search process on the subspace of a target class. Extensive experiments demonstrate that our PLG-MI attack significantly improves the attack success rate and visual quality for various datasets and models, notably, 2 ∼ 3× better than state-of-the-art attacks under large distributional shifts. Our code is available at: https://github.com/LetheSec/PLG-MI-Attack.

----

## [373] Cyclically Disentangled Feature Translation for Face Anti-spoofing

**Authors**: *Haixiao Yue, Keyao Wang, Guosheng Zhang, Haocheng Feng, Junyu Han, Errui Ding, Jingdong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25443](https://doi.org/10.1609/aaai.v37i3.25443)

**Abstract**:

Current domain adaptation methods for face anti-spoofing leverage labeled source domain data and unlabeled target domain data to obtain a promising generalizable decision boundary. However, it is usually difficult for these methods to achieve a perfect domain-invariant liveness feature disentanglement, which may degrade the final classification performance by domain differences in illumination, face category, spoof type, etc. In this work, we tackle cross-scenario face anti-spoofing by proposing a novel domain adaptation method called cyclically disentangled feature translation network (CDFTN). Specifically, CDFTN generates pseudo-labeled samples that possess: 1) source domain-invariant liveness features and 2) target domain-specific content features, which are disentangled through domain adversarial training. A robust classifier is trained based on the synthetic pseudo-labeled images under the supervision of source domain labels. We further extend CDFTN for multi-target domain adaptation by leveraging data from more unlabeled target domains. Extensive experiments on several public datasets demonstrate that our proposed approach significantly outperforms the state of the art. Code and models are available at https://github.com/vis-face/CDFTN.

----

## [374] FlowFace: Semantic Flow-Guided Shape-Aware Face Swapping

**Authors**: *Hao Zeng, Wei Zhang, Changjie Fan, Tangjie Lv, Suzhen Wang, Zhimeng Zhang, Bowen Ma, Lincheng Li, Yu Ding, Xin Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25444](https://doi.org/10.1609/aaai.v37i3.25444)

**Abstract**:

In this work, we propose a semantic flow-guided two-stage framework for shape-aware face swapping, namely FlowFace. Unlike most previous methods that focus on transferring the source inner facial features but neglect facial contours, our FlowFace can transfer both of them to a target face, thus leading to more realistic face swapping. Concretely, our FlowFace consists of a face reshaping network and a face swapping network. The face reshaping network addresses the shape outline differences between the source and target faces. It first estimates a semantic flow (i.e. face shape differences) between the source and the target face, and then explicitly warps the target face shape with the estimated semantic flow. After reshaping, the face swapping network generates inner facial features that exhibit the identity of the source face. We employ a pre-trained face masked autoencoder (MAE) to extract facial features from both the source face and the target face. In contrast to previous methods that use identity embedding to preserve identity information, the features extracted by our encoder can better capture facial appearances and identity information. Then, we develop a cross-attention fusion module to adaptively fuse inner facial features from the source face with the target facial attributes, thus leading to better identity preservation. Extensive quantitative and qualitative experiments on in-the-wild faces demonstrate that our FlowFace outperforms the state-of-the-art significantly.

----

## [375] Multi-Modal Knowledge Hypergraph for Diverse Image Retrieval

**Authors**: *Yawen Zeng, Qin Jin, Tengfei Bao, Wenfeng Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25445](https://doi.org/10.1609/aaai.v37i3.25445)

**Abstract**:

The task of keyword-based diverse image retrieval has received considerable attention due to its wide demand in real-world scenarios. Existing methods either rely on a multi-stage re-ranking strategy based on human design to diversify results, or extend sub-semantics via an implicit generator, which either relies on manual labor or lacks explainability. To learn more diverse and explainable representations, we capture sub-semantics in an explicit manner by leveraging the multi-modal knowledge graph (MMKG) that contains richer entities and relations. However, the huge domain gap between the off-the-shelf MMKG and retrieval datasets, as well as the semantic gap between images and texts, make the fusion of MMKG difficult. In this paper, we pioneer a degree-free hypergraph solution that models many-to-many relations to address the challenge of heterogeneous sources and heterogeneous modalities. Specifically, a hyperlink-based solution, Multi-Modal Knowledge Hyper Graph (MKHG) is proposed, which bridges heterogeneous data via various hyperlinks to diversify sub-semantics. Among them, a hypergraph construction module first customizes various hyperedges to link the heterogeneous MMKG and retrieval databases. A multi-modal instance bagging module then explicitly selects instances to diversify the semantics. Meanwhile, a diverse concept aggregator flexibly adapts key sub-semantics. Finally, several losses are adopted to optimize the semantic space. Extensive experiments on two real-world datasets have well verified the effectiveness and explainability of our proposed method.

----

## [376] Learnable Blur Kernel for Single-Image Defocus Deblurring in the Wild

**Authors**: *Jucai Zhai, Pengcheng Zeng, Chihao Ma, Jie Chen, Yong Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25446](https://doi.org/10.1609/aaai.v37i3.25446)

**Abstract**:

Recent research showed that the dual-pixel sensor has made great progress in defocus map estimation and image defocus deblurring.
However, extracting real-time dual-pixel views is troublesome and complex in algorithm deployment.
Moreover, the deblurred image generated by the defocus deblurring network lacks high-frequency details, which is unsatisfactory in human perception. To overcome this issue, we propose a novel defocus deblurring method that uses the guidance of the defocus map to implement image deblurring.
The proposed method consists of a learnable blur kernel to estimate the defocus map, which is an unsupervised method, and a single-image defocus deblurring generative adversarial network (DefocusGAN) for the first time.
The proposed network can learn the deblurring of different regions and recover realistic details. We propose a defocus adversarial loss to guide this training process.
Competitive experimental results confirm that with a learnable blur kernel, the generated defocus map can achieve results comparable to supervised methods.
In the single-image defocus deblurring task, the proposed method achieves state-of-the-art results, especially significant improvements in perceptual quality, where PSNR reaches 25.56 dB and LPIPS reaches 0.111.

----

## [377] Darwinian Model Upgrades: Model Evolving with Selective Compatibility

**Authors**: *Binjie Zhang, Shupeng Su, Yixiao Ge, Xuyuan Xu, Yexin Wang, Chun Yuan, Mike Zheng Shou, Ying Shan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25447](https://doi.org/10.1609/aaai.v37i3.25447)

**Abstract**:

The traditional model upgrading paradigm for retrieval requires recomputing all gallery embeddings before deploying the new model (dubbed as "backfilling"), which is quite expensive and time-consuming considering billions of instances in industrial applications. BCT presents the first step towards backward-compatible model upgrades to get rid of backfilling. It is workable but leaves the new model in a dilemma between new feature discriminativeness and new-to-old compatibility due to the undifferentiated compatibility constraints. In this work, we propose Darwinian Model Upgrades (DMU), which disentangle the inheritance and variation in the model evolving with selective backward compatibility and forward adaptation, respectively. The old-to-new heritable knowledge is measured by old feature discriminativeness, and the gallery features, especially those of poor quality, are evolved in a lightweight manner to become more adaptive in the new latent space. We demonstrate the superiority of DMU through comprehensive experiments on large-scale landmark retrieval and face recognition benchmarks. DMU effectively alleviates the new-to-new degradation at the same time improving new-to-old compatibility, rendering a more proper model upgrading paradigm in large-scale retrieval systems.Code: https://github.com/TencentARC/OpenCompatible.

----

## [378] Mx2M: Masked Cross-Modality Modeling in Domain Adaptation for 3D Semantic Segmentation

**Authors**: *Boxiang Zhang, Zunran Wang, Yonggen Ling, Yuanyuan Guan, Shenghao Zhang, Wenhui Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25448](https://doi.org/10.1609/aaai.v37i3.25448)

**Abstract**:

Existing methods of cross-modal domain adaptation for 3D semantic segmentation predict results only via 2D-3D complementarity that is obtained by cross-modal feature matching. However, as lacking supervision in the target domain, the complementarity is not always reliable. The results are not ideal when the domain gap is large. To solve the problem of lacking supervision, we introduce masked modeling into this task and propose a method Mx2M, which utilizes masked cross-modality modeling to reduce the large domain gap. Our Mx2M contains two components. One is the core solution, cross-modal removal and prediction (xMRP), which makes the Mx2M adapt to various scenarios and provides cross-modal self-supervision. The other is a new way of cross-modal feature matching, the dynamic cross-modal filter (DxMF) that ensures the whole method dynamically uses more suitable 2D-3D complementarity. Evaluation of the Mx2M on three DA scenarios, including Day/Night, USA/Singapore, and A2D2/SemanticKITTI, brings large improvements over previous methods on many metrics.

----

## [379] Few-Shot 3D Point Cloud Semantic Segmentation via Stratified Class-Specific Attention Based Transformer Network

**Authors**: *Canyu Zhang, Zhenyao Wu, Xinyi Wu, Ziyu Zhao, Song Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25449](https://doi.org/10.1609/aaai.v37i3.25449)

**Abstract**:

3D point cloud semantic segmentation aims to group all points into different semantic categories, which benefits important applications such as point cloud scene reconstruction and understanding. Existing supervised point cloud semantic segmentation methods usually require large-scale annotated point clouds for training and cannot handle new categories. While a few-shot learning method was proposed recently to address these two problems, it suffers from high computational complexity caused by graph construction and inability to learn fine-grained relationships among points due to the use of pooling operations. In this paper, we further address these problems by developing a new multi-layer transformer network for few-shot point cloud semantic segmentation. In the proposed network, the query point cloud features are aggregated based on the class-specific support features in different scales. Without using pooling operations, our method makes full use of all pixel-level features from the support samples. By better leveraging the support features for few-shot learning, the proposed method achieves the new state-of-the-art performance, with 15% less inference time, over existing few-shot 3D point cloud segmentation models on the S3DIS dataset and the ScanNet dataset. Our code is available
at https://github.com/czzhang179/SCAT.

----

## [380] PaRot: Patch-Wise Rotation-Invariant Network via Feature Disentanglement and Pose Restoration

**Authors**: *Dingxin Zhang, Jianhui Yu, Chaoyi Zhang, Weidong Cai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25450](https://doi.org/10.1609/aaai.v37i3.25450)

**Abstract**:

Recent interest in point cloud analysis has led rapid progress in designing deep learning methods for 3D models. However, state-of-the-art models are not robust to rotations, which remains an unknown prior to real applications and harms the model performance. In this work, we introduce a novel Patch-wise Rotation-invariant network (PaRot), which achieves rotation invariance via feature disentanglement and produces consistent predictions for samples with arbitrary rotations. Specifically, we design a siamese training module which disentangles rotation invariance and equivariance from patches defined over different scales, e.g., the local geometry and global shape, via a pair of rotations. However, our disentangled invariant feature loses the intrinsic pose information of each patch. To solve this problem, we propose a rotation-invariant geometric relation to restore the relative pose with equivariant information for patches defined over different scales. Utilising the pose information, we propose a hierarchical module which implements intra-scale and inter-scale feature aggregation for 3D shape learning. Moreover, we introduce a pose-aware feature propagation process with the rotation-invariant relative pose information embedded. Experiments show that our disentanglement module extracts high-quality rotation-robust features and the proposed lightweight model achieves competitive results in rotated 3D object classification and part segmentation tasks.

----

## [381] Hierarchical Consistent Contrastive Learning for Skeleton-Based Action Recognition with Growing Augmentations

**Authors**: *Jiahang Zhang, Lilang Lin, Jiaying Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25451](https://doi.org/10.1609/aaai.v37i3.25451)

**Abstract**:

Contrastive learning has been proven beneficial for self-supervised skeleton-based action recognition. Most contrastive learning methods utilize carefully designed augmentations to generate different movement patterns of skeletons for the same semantics. However, it is still a pending issue to apply strong augmentations, which distort the images/skeletons’ structures and cause semantic loss, due to their resulting unstable training. In this paper, we investigate the potential of adopting strong augmentations and propose a general hierarchical consistent contrastive learning framework (HiCLR) for skeleton-based action recognition. Specifically, we first design a gradual growing augmentation policy to generate multiple ordered positive pairs, which guide to achieve the consistency of the learned representation from different views. Then, an asymmetric loss is proposed to enforce the hierarchical consistency via a directional clustering operation in the feature space, pulling the representations from strongly augmented views closer to those from weakly augmented views for better generalizability. Meanwhile, we propose and evaluate three kinds of strong augmentations for 3D skeletons to demonstrate the effectiveness of our method. Extensive experiments show that HiCLR outperforms the state-of-the-art methods notably on three large-scale datasets, i.e., NTU60, NTU120, and PKUMMD. Our project is publicly available at: https://jhang2020.github.io/Projects/HiCLR/HiCLR.html.

----

## [382] ImageNet Pre-training Also Transfers Non-robustness

**Authors**: *Jiaming Zhang, Jitao Sang, Qi Yi, Yunfan Yang, Huiwen Dong, Jian Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25452](https://doi.org/10.1609/aaai.v37i3.25452)

**Abstract**:

ImageNet pre-training has enabled state-of-the-art results on many tasks. In spite of its recognized contribution to generalization, we observed in this study that ImageNet pre-training also transfers adversarial non-robustness from pre-trained model into fine-tuned model in the downstream classification tasks. We first conducted experiments on various datasets and network backbones to uncover the adversarial non-robustness in fine-tuned model. Further analysis was conducted on examining the learned knowledge of fine-tuned model and standard model, and revealed that the reason leading to the non-robustness is the non-robust features transferred from ImageNet pre-trained model. Finally, we analyzed the preference for feature learning of the pre-trained model, explored the factors influencing robustness, and introduced a simple robust ImageNet pre-training solution. Our code is available at https://github.com/jiamingzhang94/ImageNet-Pretraining-transfers-non-robustness.

----

## [383] Language-Assisted 3D Feature Learning for Semantic Scene Understanding

**Authors**: *Junbo Zhang, Guofan Fan, Guanghan Wang, Zhengyuan Su, Kaisheng Ma, Li Yi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25453](https://doi.org/10.1609/aaai.v37i3.25453)

**Abstract**:

Learning descriptive 3D features is crucial for understanding 3D scenes with diverse objects and complex structures. However, it is usually unknown whether important geometric attributes and scene context obtain enough emphasis in an end-to-end trained 3D scene understanding network. To guide 3D feature learning toward important geometric attributes and scene context, we explore the help of textual scene descriptions. Given some free-form descriptions paired with 3D scenes, we extract the knowledge regarding the object relationships and object attributes. We then inject the knowledge to 3D feature learning through three classification-based auxiliary tasks. This language-assisted training can be combined with modern object detection and instance segmentation methods to promote 3D semantic scene understanding, especially in a label-deficient regime. Moreover, the 3D feature learned with language assistance is better aligned with the language features, which can benefit various 3D-language multimodal tasks. Experiments on several benchmarks of 3D-only and 3D-language tasks demonstrate the effectiveness of our language-assisted 3D feature learning.  Code is available at https://github.com/Asterisci/Language-Assisted-3D.

----

## [384] IKOL: Inverse Kinematics Optimization Layer for 3D Human Pose and Shape Estimation via Gauss-Newton Differentiation

**Authors**: *Juze Zhang, Ye Shi, Yuexin Ma, Lan Xu, Jingyi Yu, Jingya Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25454](https://doi.org/10.1609/aaai.v37i3.25454)

**Abstract**:

This paper presents an inverse kinematic optimization layer (IKOL) for 3D human pose and shape estimation that leverages the strength of both optimization- and regression-based methods within an end-to-end framework. IKOL involves a nonconvex optimization that establishes an implicit mapping from an image’s 3D keypoints and body shapes to the relative body-part rotations. The 3D keypoints and the body shapes are the inputs and the relative body-part rotations are the solutions. However, this procedure is implicit and hard to make differentiable. So, to overcome this issue, we designed a Gauss-Newton differentiation (GN-Diff) procedure to differentiate IKOL. GN-Diff iteratively linearizes the nonconvex objective function to obtain Gauss-Newton directions with closed form solutions. Then, an automatic differentiation procedure is directly applied to generate a Jacobian matrix for end-to-end training. Notably, the GN-Diff procedure works fast because it does not rely on a time-consuming implicit differentiation procedure. The twist rotation and shape parameters are learned from the neural networks and, as a result, IKOL has a much lower computational overhead than most existing optimization-based methods. Additionally, compared to existing regression-based methods, IKOL provides a more accurate mesh-image correspondence. This is because it iteratively reduces the distance between the keypoints and also enhances the reliability of the pose structures. Extensive experiments demonstrate the superiority of our proposed framework over a wide range of 3D human pose and shape estimation methods. Code is available at  https://github.com/Juzezhang/IKOL

----

## [385] Mind the Gap: Polishing Pseudo Labels for Accurate Semi-supervised Object Detection

**Authors**: *Lei Zhang, Yuxuan Sun, Wei Wei*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25455](https://doi.org/10.1609/aaai.v37i3.25455)

**Abstract**:

Exploiting pseudo labels (e.g., categories and bounding boxes) of unannotated objects produced by a teacher detector have underpinned much of recent progress in semi-supervised object detection (SSOD). However, due to the limited generalization capacity of the teacher detector caused by the scarce annotations, the produced pseudo labels often deviate from ground truth, especially those with relatively low classification confidences, thus limiting the generalization performance of SSOD. To mitigate this problem, we propose a dual pseudo-label polishing framework for SSOD. Instead of directly exploiting the pseudo labels produced by the teacher detector, we take the first attempt at reducing their deviation from ground truth using dual polishing learning, where two differently structured polishing networks are elaborately developed and trained using synthesized paired pseudo labels and the corresponding ground truth for categories and bounding boxes on the given annotated objects, respectively. By doing this, both polishing networks can infer more accurate pseudo labels for unannotated objects through sufficiently exploiting their context knowledge based on the initially produced pseudo labels, and thus improve the generalization performance of SSOD. Moreover, such a scheme can be seamlessly plugged into the existing SSOD framework for joint end-to-end learning. In addition, we propose to disentangle the polished pseudo categories and bounding boxes of unannotated objects for separate category classification and bounding box regression in SSOD, which enables introducing more unannotated objects during model training and thus further improves the performance. Experiments on both PASCAL VOC and MS-COCO benchmarks demonstrate the superiority of the proposed method over existing state-of-the-art baselines. The code can be found at https://github.com/snowdusky/DualPolishLearning.

----

## [386] ConvMatch: Rethinking Network Design for Two-View Correspondence Learning

**Authors**: *Shihua Zhang, Jiayi Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25456](https://doi.org/10.1609/aaai.v37i3.25456)

**Abstract**:

Multilayer perceptron (MLP) has been widely used in two-view correspondence learning for only unordered correspondences provided, and it extracts deep features from individual correspondence effectively. However, the problem of lacking context information limits its performance and hence, many extra complex blocks are designed to capture such information in the follow-up studies. In this paper, from a novel perspective, we design a correspondence learning network called ConvMatch that for the first time can leverage convolutional neural network (CNN) as the backbone to capture better context, thus avoiding the complex design of extra blocks. Specifically, with the observation that sparse motion vectors and dense motion field can be converted into each other with interpolating and sampling, we regularize the putative motion vectors by estimating dense motion field implicitly, then rectify the errors caused by outliers in local areas with CNN, and finally obtain correct motion vectors from the rectified motion field. Extensive experiments reveal that ConvMatch with a simple CNN backbone consistently outperforms state-of-the-arts including MLP-based methods for relative pose estimation and homography estimation, and shows promising generalization ability to different datasets and descriptors. Our code is publicly available at https://github.com/SuhZhang/ConvMatch.

----

## [387] Cross-View Geo-Localization via Learning Disentangled Geometric Layout Correspondence

**Authors**: *Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, Safwan Wshah*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25457](https://doi.org/10.1609/aaai.v37i3.25457)

**Abstract**:

Cross-view geo-localization aims to estimate the location of a query ground image by matching it to a reference geo-tagged aerial images database. As an extremely challenging task, its difficulties root in the drastic view changes and different capturing time between two views. Despite these difficulties, recent works achieve outstanding progress on cross-view geo-localization benchmarks. However, existing methods still suffer from poor performance on the cross-area benchmarks, in which the training and testing data are captured from two different regions. We attribute this deficiency to the lack of ability to extract the spatial configuration of visual feature layouts and models' overfitting on low-level details from the training set. In this paper, we propose GeoDTR which explicitly disentangles geometric information from raw features and learns the spatial correlations among visual features from aerial and ground pairs with a novel geometric layout extractor module. This module generates a set of geometric layout descriptors, modulating the raw features and producing high-quality latent representations. In addition, we elaborate on two categories of data augmentations, (i) Layout simulation, which varies the spatial configuration while keeping the low-level details intact. (ii) Semantic augmentation, which alters the low-level details and encourages the model to capture spatial configurations. These augmentations help to improve the performance of the cross-view geo-localization models, especially on the cross-area benchmarks. Moreover, we propose a counterfactual-based learning process to benefit the geometric layout extractor in exploring spatial information. Extensive experiments show that GeoDTR not only achieves state-of-the-art results but also significantly boosts the performance on same-area and cross-area benchmarks. Our code can be found at https://gitlab.com/vail-uvm/geodtr.

----

## [388] Video Compression Artifact Reduction by Fusing Motion Compensation and Global Context in a Swin-CNN Based Parallel Architecture

**Authors**: *Xinjian Zhang, Su Yang, Wuyang Luo, Longwen Gao, Weishan Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25458](https://doi.org/10.1609/aaai.v37i3.25458)

**Abstract**:

Video Compression Artifact Reduction aims to reduce the artifacts caused by video compression algorithms and improve the quality of compressed video frames. The critical challenge in this task is to make use of the redundant high-quality information in compressed frames for compensation as much as possible. Two important possible compensations: Motion compensation and global context, are not comprehensively considered in previous works, leading to inferior results. The key idea of this paper is to fuse the motion compensation and global context together to gain more compensation information to improve the quality of compressed videos. Here, we propose a novel Spatio-Temporal Compensation Fusion (STCF) framework with the Parallel Swin-CNN Fusion (PSCF) block, which can simultaneously learn and merge the motion compensation and global context to reduce the video compression artifacts. Specifically, a temporal self-attention strategy based on shifted windows is developed to capture the global context in an efficient way, for which we use the Swin transformer layer in the PSCF block. Moreover, an additional Ada-CNN layer is applied in the PSCF block to extract the motion compensation. Experimental results demonstrate that our proposed STCF framework outperforms the state-of-the-art methods up to 0.23dB (27% improvement) on the MFQEv2 dataset.

----

## [389] MRCN: A Novel Modality Restitution and Compensation Network for Visible-Infrared Person Re-identification

**Authors**: *Yukang Zhang, Yan Yan, Jie Li, Hanzi Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25459](https://doi.org/10.1609/aaai.v37i3.25459)

**Abstract**:

Visible-infrared person re-identification (VI-ReID), which aims to search identities across different spectra, is a challenging task due to large cross-modality discrepancy between visible and infrared images. The key to reduce the discrepancy is to filter out identity-irrelevant interference and effectively learn modality-invariant person representations. In this paper, we propose a novel Modality Restitution and Compensation Network (MRCN) to narrow the gap between the two modalities. Specifically, we first reduce the modality discrepancy by using two Instance Normalization (IN) layers. Next, to reduce the influence of IN layers on removing discriminative information and to reduce modality differences, we propose a Modality Restitution Module (MRM) and a Modality Compensation Module (MCM) to respectively distill modality-irrelevant and modality-relevant features from the removed information. Then, the modality-irrelevant features are used to restitute to the normalized visible and infrared features, while the modality-relevant features are used to compensate for the features of the other modality. Furthermore, to better disentangle the modality-relevant features and the modality-irrelevant features, we propose a novel Center-Quadruplet Causal (CQC) loss to encourage the network to effectively learn the modality-relevant features and the modality-irrelevant features. Extensive experiments are conducted to validate the superiority of our method on the challenging SYSU-MM01 and RegDB datasets. More remarkably, our method achieves 95.1% in terms of Rank-1 and 89.2% in terms of mAP on the RegDB dataset.

----

## [390] A Simple Baseline for Multi-Camera 3D Object Detection

**Authors**: *Yunpeng Zhang, Wenzhao Zheng, Zheng Zhu, Guan Huang, Jiwen Lu, Jie Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25460](https://doi.org/10.1609/aaai.v37i3.25460)

**Abstract**:

3D object detection with surrounding cameras has been a promising direction for autonomous driving. In this paper, we present SimMOD, a Simple baseline for Multi-camera Object Detection, to solve the problem. To incorporate multiview information as well as build upon previous efforts on monocular 3D object detection, the framework is built on sample-wise object proposals and designed to work in a twostage manner. First, we extract multi-scale features and generate the perspective object proposals on each monocular image. Second, the multi-view proposals are aggregated and then iteratively refined with multi-view and multi-scale visual features in the DETR3D-style. The refined proposals are endto-end decoded into the detection results. To further boost the performance, we incorporate the auxiliary branches alongside the proposal generation to enhance the feature learning. Also, we design the methods of target filtering and teacher forcing to promote the consistency of two-stage training. We conduct extensive experiments on the 3D object detection benchmark of nuScenes to demonstrate the effectiveness of SimMOD and achieve competitive performance. Code will be available at https://github.com/zhangyp15/SimMOD.

----

## [391] Positional Label for Self-Supervised Vision Transformer

**Authors**: *Zhemin Zhang, Xun Gong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25461](https://doi.org/10.1609/aaai.v37i3.25461)

**Abstract**:

Positional encoding is important for vision transformer (ViT) to capture the spatial structure of the input image. General effectiveness has been proven in ViT. In our work we propose to train ViT to recognize the positional label of patches of the input image, this apparently simple task actually yields a meaningful self-supervisory task. Based on previous work on ViT positional encoding, we propose two positional labels dedicated to 2D images including absolute position and relative position. Our positional labels can be easily plugged into various current ViT variants. It can work in two ways: (a) As an auxiliary training target for vanilla ViT for better performance. (b) Combine the self-supervised ViT to provide a more powerful self-supervised signal for semantic feature learning. Experiments demonstrate that with the proposed self-supervised methods, ViT-B and Swin-B gain improvements of 1.20% (top-1 Acc) and 0.74% (top-1 Acc) on ImageNet, respectively, and 6.15% and 1.14% improvement on Mini-ImageNet. The code is publicly available at: https://github.com/zhangzhemin/PositionalLabel.

----

## [392] Cross-Category Highlight Detection via Feature Decomposition and Modality Alignment

**Authors**: *Zhenduo Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25462](https://doi.org/10.1609/aaai.v37i3.25462)

**Abstract**:

Learning an autonomous highlight video detector with good transferability across video categories,  called Cross-Category Video Highlight Detection(CC-VHD),  is crucial for the practical application on video-based media platforms. To tackle this problem, we first propose a framework that treats the CC-VHD as learning category-independent highlight feature representation. Under this framework,  we propose a novel module, named Multi-task Feature Decomposition Branch which jointly conducts label prediction,  cyclic feature reconstruction, and adversarial feature reconstruction to decompose the video features into two independent components: highlight-related component and category-related component. Besides,  we propose to align the visual and audio modalities to one aligned feature space before conducting modality fusion, which has not been considered in previous works. Finally, the extensive experimental results on three challenging public benchmarks validate the efficacy of our paradigm and the superiority over the existing state-of-the-art approaches to video highlight detection.

----

## [393] TrEP: Transformer-Based Evidential Prediction for Pedestrian Intention with Uncertainty

**Authors**: *Zhengming Zhang, Renran Tian, Zhengming Ding*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25463](https://doi.org/10.1609/aaai.v37i3.25463)

**Abstract**:

With rapid development in hardware (sensors and processors) and AI algorithms, automated driving techniques have entered the public’s daily life and achieved great success in supporting human driving performance. However, due to the high contextual variations and temporal dynamics in pedestrian behaviors, the interaction between autonomous-driving cars and pedestrians remains challenging, impeding the development of fully autonomous driving systems. This paper focuses on predicting pedestrian intention with a novel transformer-based evidential prediction (TrEP) algorithm. We develop a transformer module towards the temporal correlations among the input features within pedestrian video sequences and a deep evidential learning model to capture the AI uncertainty under scene complexities. Experimental results on three popular pedestrian intent benchmarks have verified the effectiveness of our proposed model over the state-of-the-art. The algorithm performance can be further boosted by controlling the uncertainty level. We systematically compare human disagreements with AI uncertainty to further evaluate AI performance in confusing scenes. The code is released at https://github.com/zzmonlyyou/TrEP.git.

----

## [394] DINet: Deformation Inpainting Network for Realistic Face Visually Dubbing on High Resolution Video

**Authors**: *Zhimeng Zhang, Zhipeng Hu, Wenjin Deng, Changjie Fan, Tangjie Lv, Yu Ding*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25464](https://doi.org/10.1609/aaai.v37i3.25464)

**Abstract**:

For few-shot learning, it is still a critical challenge to realize photo-realistic face visually dubbing on high-resolution videos. Previous works fail to generate high-fidelity dubbing results. To address the above problem, this paper proposes a Deformation Inpainting Network (DINet) for high-resolution face visually dubbing. Different from previous works relying on multiple up-sample layers to directly generate pixels from latent embeddings, DINet performs spatial deformation on feature maps of reference images to better preserve high-frequency textural details. Specifically, DINet consists of one deformation part and one inpainting part. In the first part, five reference facial images adaptively perform spatial deformation to create deformed feature maps encoding mouth shapes at each frame, in order to align with input driving audio and also the head poses of input source images. In the second part, to produce face visually dubbing, a feature decoder is responsible for adaptively incorporating mouth movements from the deformed feature maps and other attributes (i.e., head pose and upper facial expression) from the source feature maps together. Finally, DINet achieves face visually dubbing with rich textural details. We conduct qualitative and quantitative comparisons to validate our DINet on high-resolution videos. The experimental results show that our method outperforms state-of-the-art works.

----

## [395] ShiftDDPMs: Exploring Conditional Diffusion Models by Shifting Diffusion Trajectories

**Authors**: *Zijian Zhang, Zhou Zhao, Jun Yu, Qi Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25465](https://doi.org/10.1609/aaai.v37i3.25465)

**Abstract**:

Diffusion models have recently exhibited remarkable abilities to synthesize striking image samples since the introduction of denoising diffusion probabilistic models (DDPMs). Their key idea is to disrupt images into noise through a fixed forward process and learn its reverse process to generate samples from noise in a denoising way. For conditional DDPMs, most existing practices relate conditions only to the reverse process and fit it to the reversal of unconditional forward process. We find this will limit the condition modeling and generation in a small time window. In this paper, we propose a novel and flexible conditional diffusion model by introducing conditions into the forward process. We utilize extra latent space to allocate an exclusive diffusion trajectory for each condition based on some shifting rules, which will disperse condition modeling to all timesteps and improve the learning capacity of model. We formulate our method, which we call ShiftDDPMs, and provide a unified point of view on existing related methods. Extensive qualitative and quantitative experiments on image synthesis demonstrate the feasibility and effectiveness of ShiftDDPMs.

----

## [396] Combating Unknown Bias with Effective Bias-Conflicting Scoring and Gradient Alignment

**Authors**: *Bowen Zhao, Chen Chen, Qian-Wei Wang, Anfeng He, Shutao Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25466](https://doi.org/10.1609/aaai.v37i3.25466)

**Abstract**:

Models notoriously suffer from dataset biases which are detrimental to robustness and generalization. The identify-emphasize paradigm shows a promising effect in dealing with unknown biases. However, we find that it is still plagued by two challenges: A, the quality of the identified bias-conflicting samples is far from satisfactory; B, the emphasizing strategies just yield suboptimal performance. In this work, for challenge A, we propose an effective bias-conflicting scoring method to boost the identification accuracy with two practical strategies --- peer-picking and epoch-ensemble. For challenge B, we point out that the gradient contribution statistics can be a reliable indicator to inspect whether the optimization is dominated by bias-aligned samples. Then, we propose gradient alignment, which employs gradient statistics to balance the contributions of the mined bias-aligned and bias-conflicting samples dynamically throughout the learning process, forcing models to leverage intrinsic features to make fair decisions. Experiments are conducted on multiple datasets in various settings, demonstrating that the proposed solution can alleviate the impact of unknown biases and achieve state-of-the-art performance.

----

## [397] RLogist: Fast Observation Strategy on Whole-Slide Images with Deep Reinforcement Learning

**Authors**: *Boxuan Zhao, Jun Zhang, Deheng Ye, Jian Cao, Xiao Han, Qiang Fu, Wei Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25467](https://doi.org/10.1609/aaai.v37i3.25467)

**Abstract**:

Whole-slide images (WSI) in computational pathology have high resolution with gigapixel size, but are generally with sparse regions of interest, which leads to weak diagnostic relevance and data inefficiency for each area in the slide. Most of the existing methods rely on a multiple instance learning framework that requires densely sampling local patches at high magnification. The limitation is evident in the application stage as the heavy computation for extracting patch-level features is inevitable. In this paper, we develop RLogist, a benchmarking deep reinforcement learning (DRL) method for fast observation strategy on WSIs. Imitating the diagnostic logic of human pathologists, our RL agent learns how to find regions of observation value and obtain representative features across multiple resolution levels, without having to analyze each part of the WSI at the high magnification. We benchmark our method on two whole-slide level classification tasks, including detection of metastases in WSIs of lymph node sections, and subtyping of lung cancer. Experimental results demonstrate that RLogist achieves competitive classification performance compared to typical multiple instance learning algorithms, while having a significantly short observation path. In addition, the observation path given by RLogist provides good decision-making interpretability, and its ability of reading path navigation can potentially be used by pathologists for educational/assistive purposes. Our code is available at: https://github.com/tencent-ailab/RLogist.

----

## [398] Learning to Super-resolve Dynamic Scenes for Neuromorphic Spike Camera

**Authors**: *Jing Zhao, Ruiqin Xiong, Jian Zhang, Rui Zhao, Hangfan Liu, Tiejun Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25468](https://doi.org/10.1609/aaai.v37i3.25468)

**Abstract**:

Spike camera is a kind of neuromorphic sensor that uses a novel ``integrate-and-fire'' mechanism to generate a continuous spike stream to record the dynamic light intensity at extremely high temporal resolution. However, as a trade-off for high temporal resolution, its spatial resolution is limited, resulting in inferior reconstruction details. To address this issue, this paper develops a network (SpikeSR-Net) to super-resolve a high-resolution image sequence from the low-resolution binary spike streams. SpikeSR-Net is designed based on the observation model of spike camera and exploits both the merits of model-based and learning-based methods. To deal with the limited representation capacity of binary data, a pixel-adaptive spike encoder is proposed to convert spikes to latent representation to infer clues on intensity and motion. Then, a motion-aligned super resolver is employed to exploit long-term correlation, so that the dense sampling in temporal domain can be exploited to enhance the spatial resolution without introducing motion blur. Experimental results show that SpikeSR-Net is  promising in super-resolving higher-quality images for spike camera.

----

## [399] TinyNeRF: Towards 100 x Compression of Voxel Radiance Fields

**Authors**: *Tianli Zhao, Jiayuan Chen, Cong Leng, Jian Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i3.25469](https://doi.org/10.1609/aaai.v37i3.25469)

**Abstract**:

Voxel grid representation of 3D scene properties has been widely used to improve the training or rendering speed of the Neural Radiance Fields (NeRF) while at the same time achieving high synthesis quality. However, these methods accelerate the original NeRF at the expense of extra storage demand, which hinders their applications in many scenarios. To solve this limitation, we present TinyNeRF, a three-stage pipeline: frequency domain transformation, pruning and quantization that work together to reduce the storage demand of the voxel grids with little to no effects on their speed and synthesis quality. Based on the prior knowledge of visual signals sparsity in the frequency domain, we convert the original voxel grids in the frequency domain via block-wise discrete cosine transformation (DCT). Next, we apply pruning and quantization to enforce the DCT coefficients to be sparse and low-bit. Our method can be optimized from scratch in an end-to-end manner, and can typically compress the original models by 2 orders of magnitude with minimal sacrifice on speed and synthesis quality.

----



[Go to the previous page](AAAI-2023-list01.md)

[Go to the next page](AAAI-2023-list03.md)

[Go to the catalog section](README.md)