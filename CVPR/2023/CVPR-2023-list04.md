## [600] IS-GGT: Iterative Scene Graph Generation with Generative Transformers

**Authors**: *Sanjoy Kundu, Sathyanarayanan N. Aakur*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00609](https://doi.org/10.1109/CVPR52729.2023.00609)

**Abstract**:

Scene graphs provide a rich, structured representation of a scene by encoding the entities (objects) and their spatial relationships in a graphical format. This representation has proven useful in several tasks, such as question answering, captioning, and even object detection, to name a few. Current approaches take a generation-by-classification approach where the scene graph is generated through labeling of all possible edges between objects in a scene, which adds computational overhead to the approach. This work introduces a generative transformer-based approach to generating scene graphs beyond link prediction. Using two transformer-based components, we first sample a possible scene graph structure from detected objects and their visual features. We then perform predicate classification on the sampled edges to generate the final scene graph. This approach allows us to efficiently generate scene graphs from images with minimal inference overhead. Extensive experiments on the Visual Genome dataset demonstrate the efficiency of the proposed approach. Without bells and whistles, we obtain, on average, 20.7% mean recall (mR@100) across different settings for scene graph generation (SGG), outperforming state-of-the-art SGG approaches while offering competitive performance to unbiased SGG approaches.

----

## [601] Fast Contextual Scene Graph Generation with Unbiased Context Augmentation

**Authors**: *Tianlei Jin, Fangtai Guo, Qiwei Meng, Shiqiang Zhu, Xiangming Xi, Wen Wang, Zonghao Mu, Wei Song*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00610](https://doi.org/10.1109/CVPR52729.2023.00610)

**Abstract**:

Scene graph generation (SGG) methods have historically suffered from long-tail bias and slow inference speed. In this paper, we notice that humans can analyze relationships between objects relying solely on context descriptions, and this abstract cognitive process may be guided by experience. For example, given descriptions of cup and table with their spatial locations, humans can speculate possible relationships < cup, on, table > or < table, near, cup >. Even without visual appearance information, some impossible predicates like flying in and looking at can be empirically excluded. Accordingly, we propose a contextual scene graph generation (C-SGG) method without using visual information and introduce a context augmentation method. We propose that slight perturbations in the position and size of objects do not essentially affect the relationship between objects. Therefore, at the context level, we can produce diverse context descriptions by using a context augmentation method based on the original dataset. These diverse context descriptions can be used for unbiased training of C-SGG to alleviate long-tail bias. In addition, we also introduce a context guided visual scene graph generation (CV-SGG) method, which leverages the C-SGG experience to guide vision to focus on possible predicates. Through extensive experiments on the publicly available dataset, C-SGG alleviates long-tail bias and omits the huge computation of visual feature extraction to realize real-time SGG. CV-SGG achieves a great trade-off between common predicates and tail predicates.

----

## [602] Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning

**Authors**: *Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Lu Yuan, Yu-Gang Jiang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00611](https://doi.org/10.1109/CVPR52729.2023.00611)

**Abstract**:

Benefiting from masked visual modeling, self-supervised video representation learning has achieved remarkable progress. However, existing methods focus on learning representations from scratch through reconstructing low-level features like raw pixel values. In this paper, we propose masked video distillation (MVD), a simple yet effective two-stage masked feature modeling framework for video representation learning: firstly we pretrain an image (or video) model by recovering low-level features of masked patches, then we use the resulting features as targets for masked feature modeling. For the choice of teacher models, we observe that students taught by video teachers perform better on temporally-heavy video tasks, while image teachers transfer stronger spatial representations for spatially-heavy video tasks. Visualization analysis also indicates different teachers produce different learned patterns for students. To leverage the advantage of different teachers, we design a spatial-temporal co-teaching method for MVD. Specifically, we distill student models from both video teachers and image teachers by masked feature modeling. Extensive experimental results demonstrate that video transformers pre-trained with spatial-temporal co-teaching outperform models distilled with a single teacher on a multitude of video datasets. Our MVD with vanilla ViT achieves state-of-the-art performance compared with previous methods on several challenging video downstream tasks. For example, with the ViT-Large model, our MVD achieves 86.4% and 76.7% Top-1 accuracy on Kinetics-400 and Something-Something-v2, outperforming VideoMAE by 1.2% and 2.4% respectively. When a larger ViT-Huge model is adopted, MVD achieves the state-of-the-art performance with 77.3% Top-1 accuracy on Something-Something-v2. Code will be available at https://github.com/ruiwang2021/mvd.

----

## [603] MED-VT: Multiscale Encoder-Decoder Video Transformer with Application to Object Segmentation

**Authors**: *Rezaul Karim, He Zhao, Richard P. Wildes, Mennatullah Siam*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00612](https://doi.org/10.1109/CVPR52729.2023.00612)

**Abstract**:

Multiscale video transformers have been explored in a wide variety of vision tasks. To date, however, the multiscale processing has been confined to the encoder or decoder alone. We present a unified multiscale encoder-decoder transformer that is focused on dense prediction tasks in videos. Multiscale representation at both encoder and decoder yields key benefits of implicit extraction of spatiotemporal features (i.e. without reliance on input optical flow) as well as temporal consistency at encoding and coarse-to-fine detection for high-level (e.g. object) semantics to guide precise localization at decoding. Moreover, we propose a transductive learning scheme through many-to-many label propagation to provide temporally consistent predictions. We showcase our Multiscale Encoder-Decoder Video Transformer (MED-VT) on Automatic Video Object Segmentation (AVOS) and actor/action segmentation, where we outperform state-of-the-art approaches on multiple benchmarks using only raw images, without using optical flow.

----

## [604] MOVES: Manipulated Objects in Video Enable Segmentation

**Authors**: *Richard E. L. Higgins, David F. Fouhey*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00613](https://doi.org/10.1109/CVPR52729.2023.00613)

**Abstract**:

Our method uses manipulation in video to learn to understand held-objects and hand-object contact. We train a system that takes a single RGB image and produces a pixel-embedding that can be used to answer grouping questions (do these two pixels go together) as well as hand-association questions (is this hand holding that pixel). Rather than painstakingly annotate segmentation masks, we observe people in realistic video data. We show that pairing epipolar geometry with modern optical flow produces simple and effective pseudo-labels for grouping. Given people segmentations, we can further associate pixels with hands to understand contact. Our system achieves competitive results on hand and hand-held object tasks.

----

## [605] InstMove: Instance Motion for Object-centric Video Segmentation

**Authors**: *Qihao Liu, Junfeng Wu, Yi Jiang, Xiang Bai, Alan L. Yuille, Song Bai*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00614](https://doi.org/10.1109/CVPR52729.2023.00614)

**Abstract**:

Despite significant efforts, cutting-edge video segmentation methods still remain sensitive to occlusion and rapid movement, due to their reliance on the appearance of objects in the form of object embeddings, which are vulnerable to these disturbances. A common solution is to use optical flow to provide motion information, but essentially it only considers pixel-level motion, which still relies on appearance similarity and hence is often inaccurate under occlusion and fast movement. In this work, we study the instance-level motion and present InstMove, which stands for Instance Motion for Object-centric Video Segmentation. In comparison to pixel-wise motion, Inst-Move mainly relies on instance-level motion information that is free from image feature embeddings, and features physical interpretations, making it more accurate and robust toward occlusion and fast-moving objects. To better fit in with the video segmentation tasks, InstMove uses instance masks to model the physical presence of an object and learns the dynamic model through a memory network to predict its position and shape in the next frame. With only a few lines of code, InstMove can be integrated into current SOTA methods for three different video segmentation tasks and boost their performance. Specifically, we improve the previous arts by 1.5 AP on OVIS dataset, which features heavy occlusions, and 4.9 AP on YouTube VIS-Long dataset, which mainly contains fast moving objects. These results suggest that instance-level motion is robust and accurate, and hence serving as a powerful solution in complex scenarios for object-centric video segmentation.

----

## [606] ZBS: Zero-Shot Background Subtraction via Instance-Level Background Modeling and Foreground Selection

**Authors**: *Yongqi An, Xu Zhao, Tao Yu, Haiyun Gu, Chaoyang Zhao, Ming Tang, Jinqiao Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00615](https://doi.org/10.1109/CVPR52729.2023.00615)

**Abstract**:

Background subtraction (BGS) aims to extract all moving objects in the video frames to obtain binary foreground segmentation masks. Deep learning has been widely used in this field. Compared with supervised-based BGS methods, unsupervised methods have better generalization. However, previous unsupervised deep learning BGS algorithms perform poorly in sophisticated scenarios such as shadows or night lights, and they cannot detect objects outside the pre-defined categories. In this work, we propose an unsuper-vised BGS algorithm based on zero-shot object detection called Zero-shot Background Subtraction (ZBS). The proposed method fully utilizes the advantages of zero-shot object detection to build the open-vocabulary instance-level background model. Based on it, the foreground can be effectively extracted by comparing the detection results of new frames with the background model. ZBS performs well for sophisticated scenarios, and it has rich and extensible categories. Furthermore, our method can easily generalize to other tasks, such as abandoned object detection in unseen environments. We experimentally show that ZBS surpasses state-of-the-art unsupervised BGS methods by 4.70% F-Measure on the CDnet 2014 dataset. The code is released at https://github.com/CASIA-IVA-Lab/ZBS.

----

## [607] Feature Aggregated Queries for Transformer-Based Video Object Detectors

**Authors**: *Yiming Cui*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00616](https://doi.org/10.1109/CVPR52729.2023.00616)

**Abstract**:

Video object detection needs to solve feature degradation situations that rarely happen in the image domain. One solution is to use the temporal information and fuse the features from the neighboring frames. With Transformer-based object detectors getting a better performance on the image domain tasks, recent works began to extend those methods to video object detection. However, those existing Transformer-based video object detectors still follow the same pipeline as those used for classical object detectors, like enhancing the object feature representations by aggregation. In this work, we take a different perspective on video object detection. In detail, we improve the qualities of queries for the Transformer-based models by aggregation. To achieve this goal, we first propose a vanilla query aggregation module that weighted averages the queries according to the features of the neighboring frames. Then, we extend the vanilla module to a more practical version, which generates and aggregates queries according to the features of the input frames. Extensive experimental results validate the effectiveness of our proposed methods: On the challenging ImageNet VID benchmark, when integrated with our proposed modules, the current state-of-the-art Transformer-based object detectors can be improved by more than 2.4% on mAP and 4.2% on AP50. Code is available at https://github.com/YimingCuiCuiCui/FAQ.

----

## [608] Context-Aware Relative Object Queries to Unify Video Instance and Panoptic Segmentation

**Authors**: *Anwesa Choudhuri, Girish Chowdhary, Alexander G. Schwing*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00617](https://doi.org/10.1109/CVPR52729.2023.00617)

**Abstract**:

Object queries have emerged as a powerful abstraction to generically represent object proposals. However, their use for temporal tasks like video segmentation poses two questions: 1) How to process frames sequentially and propagate object queries seamlessly across frames. Using independent object queries per frame doesn't permit tracking, and requires post-processing. 2) How to produce temporally consistent, yet expressive object queries that model both appearance and position changes. Using the entire video at once doesn't capture position changes and doesn't scale to long videos. As one answer to both questions we propose 'context-aware relative object queries', which are continuously propagated frame-by-frame. They seamlessly track objects and deal with occlusion and re-appearance of objects, without post-processing. Further, we find context-aware relative object queries better capture position changes of objects in motion. We evaluate the proposed approach across three challenging tasks: video instance segmentation, multi-object tracking and segmentation, and video panoptic segmentation. Using the same approach and architecture, we match or surpass state-of-the art results on the diverse and challenging OVIS, Youtube-VIS, Cityscapes-VPS, MOTS 2020 and KITTI-MOTS data.

----

## [609] Selective Structured State-Spaces for Long-Form Video Understanding

**Authors**: *Jue Wang, Wentao Zhu, Pichao Wang, Xiang Yu, Linda Liu, Mohamed Omar, Raffay Hamid*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00618](https://doi.org/10.1109/CVPR52729.2023.00618)

**Abstract**:

Effective modeling of complex spatiotemporal dependencies in long-form videos remains an open problem. The recently proposed Structured State-Space Sequence (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S4$</tex>
) model with its linear complexity offers a promising direction in this space. However, we demonstrate that treating all imagetokens equally as done by 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S4$</tex>
 model can adversely affect its efficiency and accuracy. To address this limitation, we present a novel Selective 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S4$</tex>
 (i.e., 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S5)$</tex>
 model that employs a lightweight mask generator to adaptively select informative image tokens resulting in more efficient and accurate modeling of long-term spatiotemporal dependencies in videos. Unlike previous mask-based token reduction methods used in transformers, our 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S5$</tex>
 model avoids the dense self-attention calculation by making use of the guidance of the momentum-updated 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$S4$</tex>
 model. This enables our model to efficiently discard less informative tokens and adapt to various long-form video understanding tasks more effectively. However, as is the case for most token reduction methods, the informative image tokens could be dropped incorrectly. To improve the robustness and the temporal horizon of our model, we propose a novel long-short masked contrastive learning (LSMCL) approach that enables our model to predict longer temporal context using shorter input videos. We present extensive comparative results using three challenging long-form video understanding datasets (LVU, COIN and Breakfast), demonstrating that our approach consistently outperforms the previous state-of-the-art S4 model by up to 9.6% accuracy while reducing its memory footprint by 23%.

----

## [610] Relational Space-Time Query in Long-Form Videos

**Authors**: *Xitong Yang, Fu-Jen Chu, Matt Feiszli, Raghav Goyal, Lorenzo Torresani, Du Tran*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00619](https://doi.org/10.1109/CVPR52729.2023.00619)

**Abstract**:

Egocentric videos are often available in the form of uninterrupted, uncurated long videos capturing the camera wearers' daily life activities. Understanding these videos requires models to be able to reason about activities, objects, and their interactions. However, current video benchmarks study these problems independently and under short, curated clips. In contrast, real-world applications, e.g. AR assistants, require bundling these problems for both model development and evaluation. In this paper, we propose to study these problems in a joint framework for long video understanding. Our contributions are three-fold. First, we propose an integrated framework, namely Relational Space-Time Query (ReST), for evaluating video understanding models via templated spatiotemporal queries. Second, we introduce two new benchmarks, ReST-ADL and ReST-Ego4D
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The latest version of our benchmark and models will be available here., which augment the existing egocentric video datasets with abundant query annotations generated by the ReST framework. Finally, we present a set of baselines and in-depth analysis on the two benchmarks and provide insights about the query tasks. We view our integrated framework and benchmarks as a step towards comprehensive, multi-step reasoning in long videos, and believe it will facilitate the development of next generations of video understanding models.

----

## [611] Novel-View Acoustic Synthesis

**Authors**: *Changan Chen, Alexander Richard, Roman Shapovalov, Vamsi Krishna Ithapu, Natalia Neverova, Kristen Grauman, Andrea Vedaldi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00620](https://doi.org/10.1109/CVPR52729.2023.00620)

**Abstract**:

We introduce the novel-view acoustic synthesis (NVAS) task: given the sight and sound observed at a source view-point, can we synthesize the sound of that scene from an unseen target viewpoint? We propose a neural rendering approach: Visually-Guided Acoustic Synthesis (ViGAS) network that learns to synthesize the sound of an arbitrary point in space by analyzing the input audio-visual cues. To benchmark this task, we collect two first-of-their-kind large-scale multi-view audio-visual datasets, one synthetic and one real. We show that our model successfully reasons about the spatial cues and synthesizes faithful audio on both datasets. To our knowledge, this work represents the very first formulation, dataset, and approach to solve the novel-view acoustic synthesis task, which has exciting potential applications ranging from AR/VR to art and design. Unlocked by this work, we believe that the future of novel-view synthesis is in multi-modal learning from videos.

----

## [612] Learning Audio-Visual Source Localization via False Negative Aware Contrastive Learning

**Authors**: *Weixuan Sun, Jiayi Zhang, Jianyuan Wang, Zheyuan Liu, Yiran Zhong, Tianpeng Feng, Yandong Guo, Yanhao Zhang, Nick Barnes*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00621](https://doi.org/10.1109/CVPR52729.2023.00621)

**Abstract**:

Self-supervised audio-visual source localization aims to locate sound-source objects in video frames without extra annotations. Recent methods often approach this goal with the help of contrastive learning, which assumes only the audio and visual contents from the same video are positive samples for each other. However, this assumption would suffer from false negative samples in real-world training. For example, for an audio sample, treating the frames from the same audio class as negative samples may mislead the model and therefore harm the learned representations (e.g., the audio of a siren wailing may reasonably correspond to the ambulances in multiple images). Based on this observation, we propose a new learning strategy named False Negative Aware Contrastive (FNAC) to mitigate the problem of misleading the training with such false negative samples. Specifically, we utilize the intra-modal similarities to identify potentially similar samples and construct corresponding adjacency matrices to guide contrastive learning. Further, we propose to strengthen the role of true negative samples by explicitly leveraging the visual features of sound sources to facilitate the differentiation of authentic sounding source regions. FNAC achieves state-of-the-art performances on Flickr-SoundNet, VGG-Sound, and AVSBench, which demonstrates the effectiveness of our method in mitigating the false negative issue. The code is available at https://github.com/OpenNLPLab/FNAC_AVL.

----

## [613] Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment

**Authors**: *Kim Sung-Bin, Arda Senocak, Hyunwoo Ha, Andrew Owens, Tae-Hyun Oh*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00622](https://doi.org/10.1109/CVPR52729.2023.00622)

**Abstract**:

How does audio describe the world around us? In this paper, we propose a method for generating an image of a scene from sound. Our method addresses the challenges of dealing with the large gaps that often exist between sight and sound. We design a model that works by scheduling the learning procedure of each model component to associate audio-visual modalities despite their information gaps. The key idea is to enrich the audio features with visual information by learning to align audio to visual latent space. We translate the input audio to visual features, then use a pre-trained generator to produce an image. To further improve the quality of our generated images, we use sound source localization to select the audio-visual pairs that have strong cross-modal correlations. We obtain substantially better results on the VEGAS and VGGSound datasets than prior approaches. We also show that we can control our model's predictions by applying simple manipulations to the input waveform, or to the latent space.

----

## [614] CASP-Net: Rethinking Video Saliency Prediction from an Audio-Visual Consistency Perceptual Perspective

**Authors**: *Junwen Xiong, Ganglai Wang, Peng Zhang, Wei Huang, Yufei Zha, Guangtao Zhai*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00623](https://doi.org/10.1109/CVPR52729.2023.00623)

**Abstract**:

Incorporating the audio stream enables Video Saliency Prediction (VSP) to imitate the selective attention mechanism of human brain. By focusing on the benefits of joint auditory and visual information, most VSP methods are capable of exploiting semantic correlation between vision and audio modalities but ignoring the negative effects due to the temporal inconsistency of audio-visual intrinsics. Inspired by the biological inconsistency-correction within multi-sensory information, in this study, a consistency-aware audio-visual saliency prediction network (CASP-Net) is proposed, which takes a comprehensive consideration of the audio-visual semantic interaction and consistent perception. In addition a two-stream encoder for elegant association between video frames and corresponding sound source, a novel consistency-aware predictive coding is also designed to improve the consistency within audio and visual representations iteratively. To further aggregate the multi-scale audio-visual information, a saliency decoder is introduced for the final saliency map generation. Substantial experiments demonstrate that the proposed CASP-Net outperforms the other state-of-the-art methods on six challenging audio-visual eye-tracking datasets. For a demo of our system please see our project webpage.

----

## [615] Decompose More and Aggregate Better: Two Closer Looks at Frequency Representation Learning for Human Motion Prediction

**Authors**: *Xuehao Gao, Shaoyi Du, Yang Wu, Yang Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00624](https://doi.org/10.1109/CVPR52729.2023.00624)

**Abstract**:

Encouraged by the effectiveness of encoding temporal dynamics within the frequency domain, recent human motion prediction systems prefer to first convert the motion representation from the original pose space into the frequency space. In this paper, we introduce two closer looks at effective frequency representation learning for robust motion prediction and summarize them as: decompose more and aggregate better. Motivated by these two insights, we develop two powerful units that factorize the frequency representation learning task with a novel decomposition-aggregation two-stage strategy: (1) frequency decomposition unit unweaves multi-view frequency representations from an input body motion by embedding its frequency features into multiple spaces; (2) feature aggregation unit deploys a series of intra-space and inter-space feature aggregation layers to collect comprehensive frequency representations from these spaces for robust human motion prediction. As evaluated on large-scale datasets, we develop a strong baseline model for the human motion prediction task that outperforms state-of-the-art methods by large margins: 8%∼12% on Human3.6M, 3%∼7% on CMU MoCap, and 7%∼10% on 3DPW.

----

## [616] TempSAL - Uncovering Temporal Information for Deep Saliency Prediction

**Authors**: *Bahar Aydemir, Ludo Hoffstetter, Tong Zhang, Mathieu Salzmann, Sabine Süsstrunk*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00625](https://doi.org/10.1109/CVPR52729.2023.00625)

**Abstract**:

Deep saliency prediction algorithms complement the object recognition features, they typically rely on additional information such as scene context, semantic relationships, gaze direction, and object dissimilarity. However, none of these models consider the temporal nature of gaze shifts during image observation. We introduce a novel saliency prediction model that learns to output saliency maps in sequential time intervals by exploiting human temporal attention patterns. Our approach locally modulates the saliency predictions by combining the learned temporal maps. Our experiments show that our method outperforms the state-of-the-art models, including a multi-duration saliency model, on the SALICON benchmark and CodeCharts1k dataset. Our code is publicly available on GitHub
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://ivrl.github.io/Tempsal/.

----

## [617] Prompt-Guided Zero-Shot Anomaly Action Recognition using Pretrained Deep Skeleton Features

**Authors**: *Fumiaki Sato, Ryo Hachiuma, Taiki Sekii*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00626](https://doi.org/10.1109/CVPR52729.2023.00626)

**Abstract**:

This study investigates unsupervised anomaly action recognition, which identifies video-level abnormal-human-behavior events in an unsupervised manner without abnormal samples, and simultaneously addresses three limitations in the conventional skeleton-based approaches: target domain-dependent DNN training, robustness against skeleton errors, and a lack of normal samples. We present a unified, user prompt-guided zero-shot learning framework using a target domain-independent skeleton feature extractor, which is pretrained on a large-scale action recognition dataset. Particularly, during the training phase using normal samples, the method models the distribution of skeleton features of the normal actions while freezing the weights of the DNNs and estimates the anomaly score using this distribution in the inference phase. Additionally, to increase robustness against skeleton errors, we introduce a DNN architecture inspired by a point cloud deep learning paradigm, which sparsely propagates the features between Joints. Furthermore, to prevent the unobserved normal actions from being misidentified as abnormal actions, we incorporate a similarity score between the user prompt embeddings and skeleton features aligned in the common space into the anomaly score, which indirectly supplements normal actions. On two publicly available datasets, we conduct experiments to test the effectiveness of the proposed method with respect to above-mentioned limitations.

----

## [618] MMG-Ego4D: Multi-Modal Generalization in Egocentric Action Recognition

**Authors**: *Xinyu Gong, Sreyas Mohan, Naina Dhingra, Jean-Charles Bazin, Yilei Li, Zhangyang Wang, Rakesh Ranjan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00627](https://doi.org/10.1109/CVPR52729.2023.00627)

**Abstract**:

In this paper, we study a novel problem in egocentric action recognition, which we term as “Multimodal Generalization“ (MMG). MMG aims to study how systems can generalize when data from certain modalities is limited or even completely missing. We thoroughly investigate MMG in the context of standard supervised action recognition and the more challenging few-shot setting for learning new action categories. MMG consists of two novel scenarios, designed to support security, and efficiency considerations in real-world applications: (1) missing modality generalization where some modalities that were present during the train time are missing during the inference time, and (2) cross-modal zero-shot generalization, where the modalities present during the inference time and the training time are disjoint. To enable this investigation, we construct a new dataset MMG-Ego4D containing data points with video, audio, and inertial motion sensor (IMU) modalities. Our dataset is derived from Ego4D [27] dataset, but processed and thoroughly re-annotated by human experts to facilitate research in the MMG problem. We evaluate a diverse array of models on MMG-Ego4D and propose new methods with improved generalization ability. In particular, we introduce a new fusion module with modality dropout training, contrastive-based alignment training, and a novel cross-modal prototypical loss for better few-shot performance. We hope this study will serve as a benchmark and guide future research in multimodal generalization problems. The benchmark and code are available at https://github.com/facebookresearch/MMG_Ego4D

----

## [619] Active Exploration of Multimodal Complementarity for Few-Shot Action Recognition

**Authors**: *Yuyang Wanyan, Xiaoshan Yang, Chaofan Chen, Changsheng Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00628](https://doi.org/10.1109/CVPR52729.2023.00628)

**Abstract**:

Recently, few-shot action recognition receives increasing attention and achieves remarkable progress. However, previous methods mainly rely on limited unimodal data (e.g., RGB frames) while the multimodal information remains relatively underexplored. In this paper, we propose a novel Active Multimodal Few-shot Action Recognition (AMFAR) framework, which can actively find the reliable modality for each sample based on task-dependent context information to improve few-shot reasoning procedure. In meta-training, we design an Active Sample Selection (ASS) module to organize query samples with large differences in the reliability of modalities into different groups based on modality-specific posterior distributions. In addition, we design an Active Mutual Distillation (AMD) to capture discriminative task-specific knowledge from the reliable modality to improve the representation learning of unreliable modality by bidirectional knowledge distillation. In meta-test, we adopt Adaptive Multimodal Inference (AMI) to adaptively fuse the modality-specific posterior distributions with a larger weight on the reliable modality. Extensive experimental results on four public benchmarks demonstrate that our model achieves significant improvements over existing unimodal and multimodal methods.

----

## [620] Reducing the Label Bias for Timestamp Supervised Temporal Action Segmentation

**Authors**: *Kaiyuan Liu, Yunheng Li, Shenglan Liu, Chenwei Tan, Zihang Shao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00629](https://doi.org/10.1109/CVPR52729.2023.00629)

**Abstract**:

Timestamp supervised temporal action segmentation (TSTAS) is more cost-effective than fully supervised counterparts. However, previous approaches suffer from severe label bias due to over-reliance on sparse timestamp annotations, resulting in unsatisfactory performance. In this paper, we propose the Debiasing-TSTAS (D-TSTAS) framework by exploiting unannotated frames to alleviate this bias from two phases: 1) Initialization. To reduce the dependencies on annotated frames, we propose masked timestamp predictions (MTP) to ensure that initialized model captures more contextual information. 2) Refinement. To overcome the limitation of the expressiveness from sparsely annotated timestamps, we propose a center-oriented timestamp expansion (CTE) approach to progressively expand pseudo-timestamp groups which contain semantic-rich motion representation of action segments. Then, these pseudo-timestamp groups and the model output are used to iteratively generate pseudo-labels for refining the model in a fully supervised setup. We further introduce segmental confidence loss to enable the model to have high confidence predictions within the pseudo-timestamp groups and more accurate action boundaries. Our D-TSTAS outperforms the state-of-the-art TSTAS method as well as achieves competitive results compared with fully supervised approaches on three benchmark datasets.

----

## [621] Soft-Landing Strategy for Alleviating the Task Discrepancy Problem in Temporal Action Localization Tasks

**Authors**: *Hyolim Kang, Hanjung Kim, Joungbin An, Minsu Cho, Seon Joo Kim*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00630](https://doi.org/10.1109/CVPR52729.2023.00630)

**Abstract**:

Temporal Action Localization (TAL) methods typically operate on top of feature sequences from a frozen snippet encoder that is pretrained with the Trimmed Action Classification (TAC) tasks, resulting in a task discrepancy problem. While existing TAL methods mitigate this issue either by re-training the encoder with a pretext task or by end-to-end fine-tuning, they commonly require an overload of high memory and computation. In this work, we introduce Soft-Landing (SoLa) strategy, an efficient yet effective framework to bridge the transferability gap between the pretrained encoder and the downstream tasks by incorporating a light-weight neural network, i.e., a SoLa module, on top of the frozen encoder. We also propose an unsupervised training scheme for the SoLa module; it learns with inter-frame Similarity Matching that uses the frame interval as its supervisory signal, eliminating the need for temporal annotations. Experimental evaluation on various benchmarks for downstream TAL tasks shows that our method effectively alleviates the task discrepancy problem with remarkable computational efficiency.

----

## [622] Iterative Proposal Refinement for Weakly-Supervised Video Grounding

**Authors**: *Meng Cao, Fangyun Wei, Can Xu, Xiubo Geng, Long Chen, Can Zhang, Yuexian Zou, Tao Shen, Daxin Jiang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00631](https://doi.org/10.1109/CVPR52729.2023.00631)

**Abstract**:

Weakly-Supervised Video Grounding (WSVG) aims to localize events of interest in untrimmed videos with only video-level annotations. To date, most of the state-of-the-art WSVG methods follow a two-stage pipeline, i.e., firstly generating potential temporal proposals and then grounding with these proposal candidates. Despite the recent progress, existing proposal generation methods suffer from two draw-backs: 1) lack of explicit correspondence modeling; and 2) partial coverage of complex events. To this end, we propose a novel IteRative prOposal refiNement network (dubbed as IRON) to gradually distill the prior knowledge into each proposal and encourage proposals with more complete coverage. Specifically, we set up two lightweight distillation branches to uncover the cross-modal correspondence on both the semantic and conceptual levels. Then, an iterative Label Propagation (LP) strategy is devised to prevent the network from focusing excessively on the most discriminative events instead of the whole sentence content. Precisely, during each iteration, the proposal with the minimal distillation loss and its adjacent ones are regarded as the positive samples, which refines proposal confidence scores in a cascaded manner. Extensive experiments and ablation studies on two challenging WSVG datasets have attested to the effectiveness of our IRON. The code will be available at https://github.com/mengcaopku/IRON.

----

## [623] Movies2Scenes: Using Movie Metadata to Learn Scene Representation

**Authors**: *Shixing Chen, Chun-Hao Liu, Xiang Hao, Xiaohan Nie, Maxim Arap, Raffay Hamid*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00632](https://doi.org/10.1109/CVPR52729.2023.00632)

**Abstract**:

Understanding scenes in movies is crucial for a variety of applications such as video moderation, search, and recommendation. However, labeling individual scenes is a time-consuming process. In contrast, movie level metadata (e.g., genre, synopsis, etc.) regularly gets produced as part of the film production process, and is therefore significantly more commonly available. In this work, we propose a novel contrastive learning approach that uses movie metadata to learn a general-purpose scene representation. Specifically, we use movie metadata to define a measure of movie similarity, and use it during contrastive learning to limit our search for positive scene-pairs to only the movies that are considered similar to each other. Our learned scene representation consistently outperforms existing state-of-the-art methods on a diverse set of tasks evaluated using multiple benchmark datasets. Notably, our learned representation offers an average improvement of 7.9% on the seven classification tasks and 9.7% improvement on the two regression tasks in LVU dataset. Furthermore, using a newly collected movie dataset, we present comparative results of our scene representation on a set of video moderation tasks to demonstrate its generalizability on previously less explored tasks.

----

## [624] Fine-tuned CLIP Models are Efficient Video Learners

**Authors**: *Hanoona Abdul Rasheed, Muhammad Uzair Khattak, Muhammad Maaz, Salman H. Khan, Fahad Shahbaz Khan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00633](https://doi.org/10.1109/CVPR52729.2023.00633)

**Abstract**:

Large-scale multi-modal training with image-text pairs imparts strong generalization to CLIP model. Since training on a similar scale for videos is infeasible, recent approaches focus on the effective transfer of image-based CLIP to the video domain. In this pursuit, new parametric modules are added to learn temporal information and inter-frame relationships which require meticulous design efforts. Furthermore, when the resulting models are learned on videos, they tend to overfit on the given task distribution and lack in generalization aspect. This begs the following question: How to effectively transfer image-level CLIP representations to videos? In this work, we show that a simple Video Fine-tuned CLIP (ViFi-CLIP) baseline is generally sufficient to bridge the domain gap from images to videos. Our qualitative analysis illustrates that the frame-level processing from CLIP image-encoder followed by feature pooling and similarity matching with corresponding text embeddings helps in implicitly modeling the temporal cues within ViFi-CLIP. Such fine-tuning helps the model to focus on scene dynamics, moving objects and inter-object relationships. For low-data regimes where full fine-tuning is not viable, we propose a ‘bridge and prompt’ approach that first uses fine-tuning to bridge the domain gap and then learns prompts on language and vision side to adapt CLIP representations. We extensively evaluate this simple yet strong baseline on zero-shot, base-to-novel generalization, few-shot and fully supervised settings across five video benchmarks. Our code and pre-trained models are available at https://github.com/muzairkhattak/ViFi-CLIP.

----

## [625] Revisiting Temporal Modeling for CLIP-Based Image-to-Video Knowledge Transferring

**Authors**: *Ruyang Liu, Jingjia Huang, Ge Li, Jiashi Feng, Xinglong Wu, Thomas H. Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00634](https://doi.org/10.1109/CVPR52729.2023.00634)

**Abstract**:

Image-text pretrained models, e.g., CLIP, have shown impressive general multi-modal knowledge learned from large-scale image-text data pairs, thus attracting increasing attention for their potential to improve visual representation learning in the video domain. In this paper, based on the CLIP model, we revisit temporal modeling in the context of image-to-video knowledge transferring, which is the key point for extending image-text pretrained models to the video domain. We find that current temporal modeling mechanisms are tailored to either high-level semantic-dominant tasks (e.g., retrieval) or low-level visual pattern-dominant tasks (e.g., recognition), and fail to work on the two cases simultaneously. The key difficulty lies in modeling temporal dependency while taking advantage of both high-level and low-level knowledge in CLIP model. To tackle this problem, we present Spatial-Temporal Auxiliary Network (STAN) - a simple and effective temporal modeling mechanism extending CLIP model to diverse video tasks. Specifically, to realize both low-level and high-level knowledge transferring, STAN adopts a branch structure with decomposed spatial-temporal modules that enable multi-level CLIP features to be spatial-temporally contextualized. We evaluate our method on two representative video tasks: Video-Text Retrieval and Video Recognition. Extensive experiments demonstrate the superiority of our model over the state-of-the-art methods on various datasets, including MSR-VTT, DiDeMo, LSMDC, MSVD, Kinetics-400, and Something-Something- V2. Codes will be available at https://github.com/farewellthree/STAN

----

## [626] VoP: Text-Video Co-Operative Prompt Tuning for Cross-Modal Retrieval

**Authors**: *Siteng Huang, Biao Gong, Yulin Pan, Jianwen Jiang, Yiliang Lv, Yuyuan Li, Donglin Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00635](https://doi.org/10.1109/CVPR52729.2023.00635)

**Abstract**:

Many recent studies leverage the pre-trained CLIP for text-video cross-modal retrieval by tuning the backbone with additional heavy modules, which not only brings huge computational burdens with much more parameters, but also leads to the knowledge forgetting from upstream models. In this work, we propose the VoP: Text-Video Cooperative Prompt Tuning for efficient tuning on the text-video retrieval task. The proposed VoP is an end-to-end framework with both video & text prompts introducing, which can be regarded as a powerful baseline with only 0.1% trainable parameters. Further, based on the spatiotemporal characteristics of videos, we develop three novel video prompt mechanisms to improve the performance with different scales of trainable parameters. The basic idea of the VoP enhancement is to model the frame position, frame context, and layer function with specific trainable prompts, respectively. Extensive experiments show that compared to full fine-tuning, the enhanced VoP achieves a 1.4% average R@1 gain across five text-video retrieval benchmarks with 6× less parameter overhead. The code will be available at https://github.com/bighuang624/VoP.

----

## [627] ProTéGé: Untrimmed Pretraining for Video Temporal Grounding by Video Temporal Grounding

**Authors**: *Lan Wang, Gaurav Mittal, Sandra Sajeev, Ye Yu, Matthew Hall, Vishnu Naresh Boddeti, Mei Chen*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00636](https://doi.org/10.1109/CVPR52729.2023.00636)

**Abstract**:

Video temporal grounding (VTG) is the task of localizing a given natural language text query in an arbitrarily long untrimmed video. While the task involves untrimmed videos, all existing VTG methods leverage features from video backbones pretrained on trimmed videos. This is largely due to the lack of large-scale well-annotated VTG dataset to perform pretraining. As a result, the pretrained features lack a notion of temporal boundaries leading to the video-text alignment being less distinguishable between correct and incorrect locations. We present ProTéGé as the first method to perform VTG-based untrimmed pretraining to bridge the gap between trimmed pretrained backbones and downstream VTG tasks. ProTéGé reconfigures the HowTo100M dataset, with noisily correlated video-text pairs, into a VTG dataset and introduces a novel Video-Text Similarity-based Grounding Module and a pretraining objective to make pretraining robust to noise in HowTo100M. Extensive experiments on multiple datasets across downstream tasks with all variations of supervision validate that pretrained features from ProTéGé can significantly outperform features from trimmed pretrained backbones on VTG.

----

## [628] Learning Video Representations from Large Language Models

**Authors**: *Yue Zhao, Ishan Misra, Philipp Krähenbühl, Rohit Girdhar*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00637](https://doi.org/10.1109/CVPR52729.2023.00637)

**Abstract**:

We introduce LAVILA, a new approach to learning video-language representations by leveraging Large Language Models (LLMs). We repurpose pre-trained LLMs to be conditioned on visual input, and finetune them to create automatic video narrators. Our auto-generated narrations offer a number of advantages, including dense coverage of long videos, better temporal synchronization of the visual information and text, and much higher diversity of text. The video-language embedding learned contrastively with these narrations outperforms the previous state-of-the-art on multiple first-person and third-person video tasks, both in zero-shot and finetuned setups. Most notably, Lavilaobtains an absolute gain of 10.1% on EGTEA classification and 5.9% Epic-Kitchens-100 multi-instance retrieval benchmarks. Furthermore, LaVilatrained with only half the narrations from the Ego4D dataset outperforms models trained on the full set, and shows positive scaling behavior on increasing pre-training data and model size.

----

## [629] All in One: Exploring Unified Video-Language Pre-Training

**Authors**: *Jinpeng Wang, Yixiao Ge, Rui Yan, Yuying Ge, Kevin Qinghong Lin, Satoshi Tsutsui, Xudong Lin, Guanyu Cai, Jianping Wu, Ying Shan, Xiaohu Qie, Mike Zheng Shou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00638](https://doi.org/10.1109/CVPR52729.2023.00638)

**Abstract**:

Mainstream Video-Language Pre-training (VLP) models [10, 26, 64] consist of three parts, a video encoder, a text encoder, and a video-text fusion Transformer. They pursue better performance via utilizing heavier unimodal encoders or multimodal fusion Transformers, resulting in increased parameters with lower efficiency in downstream tasks. In this work, we for the first time introduce an end-to-end VLP model, namely all-in-one Transformer, that embeds raw video and textual signals into joint representations using a unified backbone architecture. We argue that the unique temporal information of video data turns out to be a key barrier hindering the design of a modality-agnostic Transformer. To overcome the challenge, we introduce a novel and effective token rolling operation to encode temporal representations from video clips in a non-parametric manner. The careful design enables the representation learning of both video-text multimodal inputs and unimodal inputs using a unified model. Our pretrained ali-in-one Transformer is transferred to various downstream video-text tasks after fine-tuning, including text-video retrieval, video-question answering, multiple choice and video captioning. State-of-the-art performances with the minimal model FLOPs on ten datasets demonstrate the superiority of our method compared to the competitive counterparts. The code and pretrained models are available at https://github.com/showlab/all-in-one.

----

## [630] High-Fidelity Generalized Emotional Talking Face Generation with Multi-Modal Emotion Space Learning

**Authors**: *Chao Xu, Junwei Zhu, Jiangning Zhang, Yue Han, Wenqing Chu, Ying Tai, Chengjie Wang, Zhifeng Xie, Yong Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00639](https://doi.org/10.1109/CVPR52729.2023.00639)

**Abstract**:

Recently, emotional talking face generation has received considerable attention. However, existing methods only adopt one-hot coding, image, or audio as emotion conditions, thus lacking flexible control in practical applications and failing to handle unseen emotion styles due to limited semantics. They either ignore the one-shot setting or the quality of generated faces. In this paper, we propose a more flexible and generalized framework. Specifically, we supplement the emotion style in text prompts and use an Aligned Multi-modal Emotion encoder to embed the text, image, and audio emotion modality into a unified space, which inherits rich semantic prior from CLIP. Consequently, effective multi-modal emotion space learning helps our method support arbitrary emotion modality during testing and could generalize to unseen emotion styles. Besides, an Emotion-aware Audio-to-3DMM Convertor is proposed to connect the emotion condition and the audio sequence to structural representation. A followed style-based High-fidelity Emotional Face generator is designed to generate arbitrary high-resolution realistic identities. Our texture generator hierarchically learns flow fields and animated faces in a residual manner. Extensive experiments demonstrate the flexibility and generalization of our method in emotion control and the effectiveness of high-quality face synthesis.

----

## [631] Bidirectional Cross-Modal Knowledge Exploration for Video Recognition with Pre-trained Vision-Language Models

**Authors**: *Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, Wanli Ouyang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00640](https://doi.org/10.1109/CVPR52729.2023.00640)

**Abstract**:

Vision-language models (VLMs) pre-trained on large- scale image-text pairs have demonstrated impressive transferability on various visual tasks. Transferring knowledge from such powerful VLMs is a promising direction for building effective video recognition models. However, current exploration in this field is still limited. We believe that the greatest value of pre-trained VLMs lies in building a bridge between visual and textual domains. In this paper, we propose a novel framework called BIKE, which utilizes the cross-modal bridge to explore bidirectional knowledge: i) We introduce the Video Attribute Association mechanism, which leverages the Video-to-Text knowledge to generate textual auxiliary attributes for complementing video recognition. ii) We also present a Temporal Concept Spotting mechanism that uses the Text-to-Video expertise to capture temporal saliency in a parameter-free manner, leading to enhanced video representation. Extensive studies on six popular video datasets, including Kinetics-400 & 600, UCF-101, HMDB-51, ActivityNet and Charades, show that our method achieves state-of-the-art performance in various recognition scenarios, such as general, zero-shot, and few-shot video recognition. Our best model achieves a state-of-the-art accuracy of 88.6% on the challenging Kinetics-400 using the released CLIP model. The code is available at https://github.com/whwu95/BIKE.

----

## [632] Decoupled Multimodal Distilling for Emotion Recognition

**Authors**: *Yong Li, Yuanzhi Wang, Zhen Cui*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00641](https://doi.org/10.1109/CVPR52729.2023.00641)

**Abstract**:

Human multimodal emotion recognition (MER) aims to perceive human emotions via language, visual and acoustic modalities. Despite the impressive performance of previous MER approaches, the inherent multimodal heterogeneities still haunt and the contribution of different modalities varies significantly. In this work, we mitigate this issue by proposing a decoupled multimodal distillation (DMD) approach that facilitates flexible and adaptive crossmodal knowledge distillation, aiming to enhance the discriminative features of each modality. Specially, the representation of each modality is decoupled into two parts, i.e., modality-irrelevant/-exclusive spaces, in a self-regression manner. DMD utilizes a graph distillation unit (GD-Unit) for each decoupled part so that each GD can be performed in a more specialized and effective manner. A GD-Unit consists of a dynamic graph where each vertice represents a modality and each edge indicates a dynamic knowledge distillation. Such GD paradigm provides a flexible knowledge transfer manner where the distillation weights can be automatically learned, thus enabling diverse crossmodal knowledge transfer patterns. Experimental results show DMD consistently obtains superior performance than state-of-the-art MER methods. Visualization results show the graph edges in DMD exhibit meaningful distributional patterns w.r.t. the modality-irrelevant/-exclusive feature spaces. Codes are re leased at https://github.com/mdswyz/DMD.

----

## [633] Affection: Learning Affective Explanations for Real-World Visual Data

**Authors**: *Panos Achlioptas, Maks Ovsjanikov, Leonidas J. Guibas, Sergey Tulyakov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00642](https://doi.org/10.1109/CVPR52729.2023.00642)

**Abstract**:

In this work, we explore the space of emotional reactions induced by real-world images. For this, we first introduce a large-scale dataset that contains both categorical emotional reactions and free-form textual explanations for 85,007 publicly available images, analyzed by 6,283 annotators who were asked to indicate and explain how and why they felt when observing a particular image, with a total of 526,749 responses. Although emotional reactions are subjective and sensitive to context (personal mood, social status, past experiences) - we show that there is significant common ground to capture emotional responses with a large support in the subject population. In light of this observation, we ask the following questions: i) Can we develop neural networks that provide plausible affective responses to real-world visual data explained with language? ii) Can we steer such methods towards producing explanations with varying degrees of pragmatic language, justifying different emotional reactions by grounding them in the visual stimulus? Finally, iii) How to evaluate the performance of such methods for this novel task? In this work, we take the first steps in addressing all of these questions, paving the way for more human-centric and emotionally-aware image analysis systems. Our code and data are publicly available at https://affective-explanations.org.

----

## [634] An Actor-centric Causality Graph for Asynchronous Temporal Inference in Group Activity

**Authors**: *Zhao Xie, Tian Gao, Kewei Wu, Jiao Chang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00643](https://doi.org/10.1109/CVPR52729.2023.00643)

**Abstract**:

The causality relation modeling remains a challenging task for group activity recognition. The causality relations describe the influence on the centric actor (effect actor) from its correlative actors (cause actors). Most existing graph models focus on learning the actor relation with synchronous temporal features, which is insufficient to deal with the causality relation with asynchronous temporal features. In this paper, we propose an Actor-Centric Causality Graph Model, which learns the asynchronous temporal causality relation with three modules, i.e., an asynchronous temporal causality relation detection module, a causality feature fusion module, and a causality relation graph inference module. First, given a centric actor and its correlative actor, we analyze their influences to detect causality relation. We estimate the self influence of the centric actor with self regression. We estimate the correlative influence from the correlative actor to the centric actor with correlative regression, which uses asynchronous features at different timestamps. Second, we synchronize the two action features by estimating the temporal delay between the cause action and the effect action. The synchronized features are used to enhance the feature of the effect action with a channel-wise fusion. Third, we describe the nodes (actors) with causality features and learn the edges by fusing the causality relation with the appearance relation and distance relation. The causality relation graph inference provides crucial features of effect action, which are complementary to the base model using synchronous relation inference. Experiments show that our method achieves state-of-the-art performance on the Volleyball dataset and Collective Activity dataset.

----

## [635] VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision

**Authors**: *Mengyin Liu, Jie Jiang, Chao Zhu, Xu-Cheng Yin*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00644](https://doi.org/10.1109/CVPR52729.2023.00644)

**Abstract**:

Detecting pedestrians accurately in urban scenes is significant for realistic applications like autonomous driving or video surveillance. However, confusing human-like objects often lead to wrong detections, and small scale or heavily occluded pedestrians are easily missed due to their unusual appearances. To address these challenges, only object regions are inadequate, thus how to fully utilize more explicit and semantic contexts becomes a key problem. Meanwhile, previous context-aware pedestrian detectors either only learn latent contexts with visual clues, or need laborious annotations to obtain explicit and semantic contexts. Therefore, we propose in this paper a novel approach via Vision-Language semantic self-supervision for context-aware Pedestrian Detection (VLPD) to model explicitly semantic contexts without any extra annotations. Firstly, we propose a self-supervised Vision-Language Semantic (VLS) segmentation method, which learns both fully-supervised pedestrian detection and contextual segmentation via self-generated explicit labels of semantic classes by vision-language models. Furthermore, a self-supervised Prototypical Semantic Contrastive (PSC) learning method is proposed to better discriminate pedestrians and other classes, based on more explicit and semantic contexts obtained from VLS. Extensive experiments on popular benchmarks show that our proposed VLPD achieves superior performances over the previous state-of-the-arts, particularly under challenging circumstances like small scale and heavy occlusion. Code is available at https://github.com/lmy98129/VLPD.

----

## [636] 3D-Aware Object Goal Navigation via Simultaneous Exploration and Identification

**Authors**: *Jiazhao Zhang, Liu Dai, Fanpeng Meng, Qingnan Fan, Xuelin Chen, Kai Xu, He Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00645](https://doi.org/10.1109/CVPR52729.2023.00645)

**Abstract**:

Object goal navigation (ObjectNav) in unseen environments is a fundamental task for Embodied AI. Agents in existing works learn ObjectNav policies based on 2D maps, scene graphs, or image sequences. Considering this task happens in 3D space, a 3D-aware agent can advance its ObjectNav capability via learning from fine-grained spatial information. However, leveraging 3D scene representation can be prohibitively unpractical for policy learning in this floor-level task, due to low sample efficiency and expensive computational cost. In this work, we propose a framework for the challenging 3D-aware ObjectNav based on two straightforward sub-policies. The two sub-polices, namely corner-guided exploration policy and category-aware identification policy, simultaneously perform by utilizing online fused 3D points as observation. Through extensive experiments, we show that this framework can dramatically improve the performance in ObjectNav through learning from 3D scene representation. Our framework achieves the best performance among all modular-based methods on the Matterport3D and Gibson datasets, while requiring (up to 30x) less computational cost for training. The code will be released to benefit the community.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Homepage: https://pku-epic.github.io/3D-Aware-ObjectNav/

----

## [637] Meta-Explore: Exploratory Hierarchical Vision-and-Language Navigation Using Scene Object Spectrum Grounding

**Authors**: *Minyoung Hwang, Jaeyeon Jeong, Minsoo Kim, Yoonseon Oh, Songhwai Oh*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00646](https://doi.org/10.1109/CVPR52729.2023.00646)

**Abstract**:

The main challenge in vision-and-language navigation (VLN) is how to understand natural-language instructions in an unseen environment. The main limitation of conventional VLN algorithms is that if an action is mistaken, the agent fails to follow the instructions or explores unnecessary regions, leading the agent to an irrecoverable path. To tackle this problem, we propose Meta-Explore, a hierarchical navigation method deploying an exploitation policy to correct misled recent actions. We show that an exploitation policy, which moves the agent toward a well-chosen local goal among unvisited but observable states, outperforms a method which moves the agent to a previously visited state. We also highlight the demand for imagining regretful explorations with semantically meaningful clues. The key to our approach is understanding the object placements around the agent in spectral-domain. Specifically, we present a novel visual representation, called scene object spectrum (SOS), which performs category-wise 2D Fourier transform of detected objects. Combining exploitation policy and SOS features, the agent can correct its path by choosing a promising local goal. We evaluate our method in three VLN benchmarks: R2R, SOON, and REVERIE. Meta-Explore outper-forms other baselines and shows significant generalization performance. In addition, local goal search using the proposed spectral-domain SOS features significantly improves the success rate by 17.1% and SPL by 20. 6% against the state-of-the-art method of the SOON benchmark. Project page: https://rllab-snu.github.io/projects/Meta-Explore/doc.html

----

## [638] NaQ: Leveraging Narrations as Queries to Supervise Episodic Memory

**Authors**: *Santhosh Kumar Ramakrishnan, Ziad Al-Halah, Kristen Grauman*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00647](https://doi.org/10.1109/CVPR52729.2023.00647)

**Abstract**:

Searching long egocentric videos with natural language queries (NLQ) has compelling applications in augmented reality and robotics, where a fluid index into everything that a person (agent) has seen before could augment human memory and surface relevant information on demand. However, the structured nature of the learning problem (free-form text query inputs, localized video temporal window outputs) and its needle-in-a-haystack nature makes it both technically challenging and expensive to supervise. We introduce Narrations-as-Queries (NaQ), a data augmentation strategy that transforms standard video-text narrations into training data for a video query localization model. Validating our idea on the Ego4D benchmark, we find it has tremendous impact in practice. NaQ improves multiple top models by substantial margins (even doubling their accuracy), and yields the very best results to date on the Ego4D NLQ challenge, soundly outperforming all challenge winners in the CVPR and ECCV 2022 competitions and topping the current public leaderboard. Beyond achieving the state-of-the-art for NLQ, we also demonstrate unique properties of our approach such as the ability to perform zero-shot and fewshot NLQ, and improved performance on queries about long-tail object categories. Code and models: http://vision.cs.utexas.edu/projects/naq.

----

## [639] EC2: Emergent Communication for Embodied Control

**Authors**: *Yao Mu, Shunyu Yao, Mingyu Ding, Ping Luo, Chuang Gan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00648](https://doi.org/10.1109/CVPR52729.2023.00648)

**Abstract**:

Embodied control requires agents to leverage multimodal pre-training to quickly learn how to act in new environments, where video demonstrations contain visual and motion details needed for low-level perception and control, and language instructions support generalization with abstract, symbolic structures. While recent approaches apply contrastive learning to force alignment between the two modalities, we hypothesize better modeling their complementary differences can lead to more holistic representations for downstream adaption. To this end, we propose Emergent Communication for Embodied Control (EC
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
), a novel scheme to pre-train video-language representations for few-shot embodied control. The key idea is to learn an unsupervised “language” of videos via emergent communication, which bridges the semantics of video details and structures of natural language. We learn embodied representations of video trajectories, emergent language, and natural language using a language model, which is then used to finetune a lightweight policy network for downstream control. Through extensive experiments in Metaworld and Franka Kitchen embodied benchmarks, EC
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 is shown to consistently outperform previous contrastive learning methods for both videos and texts as task inputs. Further ablations confirm the importance of the emergent language, which is beneficial for both video and language learning, and significantly superior to using pre-trained video captions. We also present a quantitative and qualitative analysis of the emergent language and discuss future directions toward better understanding and leveraging emergent communication in embodied tasks.

----

## [640] Abstract Visual Reasoning: An Algebraic Approach for Solving Raven's Progressive Matrices

**Authors**: *Jingyi Xu, Tushar Vaidya, Yufei Wu, Saket Chandra, Zhangsheng Lai, Kai Fong Ernest Chong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00649](https://doi.org/10.1109/CVPR52729.2023.00649)

**Abstract**:

We introduce algebraic machine reasoning, a new reasoning framework that is well-suited for abstract reasoning. Effectively, algebraic machine reasoning reduces the difficult process of novel problem-solving to routine algebraic computation. The fundamental algebraic objects of interest are the ideals of some suitably initialized polynomial ring. We shall explain how solving Raven's Progressive Matrices (RPMs) can be realized as computational problems in algebra, which combine various well-known algebraic subroutines that include: Computing the Gröbner basis of an ideal, checking for ideal containment, etc. Crucially, the additional algebraic structure satisfied by ideals allows for more operations on ideals beyond set-theoretic operations. Our algebraic machine reasoning framework is not only able to select the correct answer from a given answer set, but also able to generate the correct answer with only the question matrix given. Experiments on the I-RAVEN dataset yield an overall 93.2% accuracy, which significantly out-performs the current state-of-the-art accuracy of77.0% and exceeds human performance at 84.4% accuracy.

----

## [641] Logical Implications for Visual Question Answering Consistency

**Authors**: *Sergio Tascon-Morales, Pablo Márquez-Neila, Raphael Sznitman*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00650](https://doi.org/10.1109/CVPR52729.2023.00650)

**Abstract**:

Despite considerable recent progress in Visual Question Answering (VQA) models, inconsistent or contradictory answers continue to cast doubt on their true reasoning capabilities. However, most proposed methods use indirect strategies or strong assumptions on pairs of questions and answers to enforce model consistency. Instead, we propose a novel strategy intended to improve model performance by directly reducing logical inconsistencies. To do this, we introduce a new consistency loss term that can be used by a wide range of the VQA models and which relies on knowing the logical relation between pairs of questions and answers. While such information is typically not available in VQA datasets, we propose to infer these logical relations using a dedicated language model and use these in our proposed consistency loss function. We conduct extensive experiments on the VQA Introspect and DME datasets and show that our method brings improvements to state-of-the-art VQA models while being robust across different architectures and settings.

----

## [642] Divide and Conquer: Answering Questions with Object Factorization and Compositional Reasoning

**Authors**: *Shi Chen, Qi Zhao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00651](https://doi.org/10.1109/CVPR52729.2023.00651)

**Abstract**:

Humans have the innate capability to answer diverse questions, which is rooted in the natural ability to correlate different concepts based on their semantic relationships and decompose difficult problems into sub-tasks. On the contrary, existing visual reasoning methods assume training samples that capture every possible object and reasoning problem, and rely on black-boxed models that commonly exploit statistical priors. They have yet to develop the capability to address novel objects or spurious biases in real-world scenarios, and also fall short of interpreting the rationales behind their decisions. Inspired by humans' reasoning of the visual world, we tackle the aforementioned challenges from a compositional perspective, and propose an integral framework consisting of a principled object factorization method and a novel neural module network. Our factorization method decomposes objects based on their key characteristics, and automatically derives prototypes that represent a wide range of objects. With these prototypes encoding important semantics, the proposed network then correlates objects by measuring their similarity on a common semantic space and makes decisions with a compositional reasoning process. It is capable of answering questions with diverse objects regard-less of their availability during training, and overcoming the issues of biased question-answer distributions. In addition to the enhanced generalizability, our framework also provides an interpretable interface for understanding the decision-making process of models. Our code is available at https://github.com/szzexpoi/POEM.

----

## [643] The Dialog Must Go On: Improving Visual Dialog via Generative Self-Training

**Authors**: *Gi-Cheon Kang, Sungdong Kim, Jin-Hwa Kim, Donghyun Kwak, Byoung-Tak Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00652](https://doi.org/10.1109/CVPR52729.2023.00652)

**Abstract**:

Visual dialog (VisDial) is a task of answering a sequence of questions grounded in an image, using the dialog history as context. Prior work has trained the dialog agents solely on VisDial data via supervised learning or leveraged pretraining on related vision-and-language datasets. This paper presents a semi-supervised learning approach for visually-grounded dialog, called Generative Self-Training (GST), to leverage unlabeled images on the Web. Specifically, GST first retrieves in-domain images through out-of-distribution detection and generates synthetic dialogs regarding the images via multimodal conditional text generation. GST then trains a dialog agent on the synthetic and the original Vis-Dial data. As a result, GST scales the amount of training data up to an order of magnitude that of VisDial (1.2M → 12.9M QA data). For robust training of the synthetic dialogs, we also propose perplexity-based data selection and multimodal consistency regularization. Evaluation on Vis-Dial v1.0 and v0.9 datasets shows that GST achieves new state-of-the-art results on both datasets. We further observe the robustness of GST against both visual and textual adversarial attacks. Finally, GST yields strong performance gains in the low-data regime. Code is available at https:/github.com/gicheonkang/gst-visdial.

----

## [644] Visual-Language Prompt Tuning with Knowledge-Guided Context Optimization

**Authors**: *Hantao Yao, Rui Zhang, Changsheng Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00653](https://doi.org/10.1109/CVPR52729.2023.00653)

**Abstract**:

Prompt tuning is an effective way to adapt the pretrained visual-language model (VLM) to the downstream task using task-related textual tokens. Representative CoOp-based work combines the learnable textual tokens with the class tokens to obtain specific textual knowledge. However, the specific textual knowledge is worse generalization to the unseen classes because it forgets the essential general textual knowledge having a strong generalization ability. To tackle this issue, we introduce a novel Knowledge-guided Context Optimization (KgCoOp) to enhance the generalization ability of the learnable prompt for unseen classes. The key insight of KgCoOp is that the forgetting about essential knowledge can be alleviated by reducing the discrepancy between the learnable prompt and the hand-crafted prompt. Especially, KgCoOp minimizes the discrepancy between the textual embeddings generated by learned prompts and the hand-crafted prompts. Finally, adding the KgCoOp upon the contrastive loss can make a discriminative prompt for both seen and unseen tasks. Extensive evaluation of several benchmarks demonstrates that the proposed Knowledge-guided Context Optimization is an efficient method for prompt tuning, i.e., achieves better performance with less training time. code.

----

## [645] Probabilistic Prompt Learning for Dense Prediction

**Authors**: *Hyeongjun Kwon, Taeyong Song, Somi Jeong, Jin Kim, Jinhyun Jang, Kwanghoon Sohn*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00654](https://doi.org/10.1109/CVPR52729.2023.00654)

**Abstract**:

Recent progress in deterministic prompt learning has become a promising alternative to various downstream vision tasks, enabling models to learn powerful visual representations with the help of pre-trained vision-language models. However, this approach results in limited performance for dense prediction tasks that require handling more complex and diverse objects, since a single and deterministic description cannot sufficiently represent the entire image. In this paper, we present a novel probabilistic prompt learning to fully exploit the vision-language knowledge in dense prediction tasks. First, we introduce learnable class-agnostic attribute prompts to describe universal attributes across the object class. The attributes are combined with class information and visual-context knowledge to define the class-specific textual distribution. Text representations are sampled and used to guide the dense prediction task using the probabilistic pixel-text matching loss, enhancing the stability and generalization capability of the proposed method. Extensive experiments on different dense prediction tasks and ablation studies demonstrate the effectiveness of our proposed method.

----

## [646] Is BERT Blind? Exploring the Effect of Vision-and-Language Pretraining on Visual Language Understanding

**Authors**: *Morris Alper, Michael Fiman, Hadar Averbuch-Elor*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00655](https://doi.org/10.1109/CVPR52729.2023.00655)

**Abstract**:

Most humans use visual imagination to understand and reason about language, but models such as BERT reason about language using knowledge acquired during text-only pretraining. In this work, we investigate whether vision-and-language pretraining can improve performance on text-only tasks that involve implicit visual reasoning, focusing primarily on zero-shot probing methods. We propose a suite of visual language understanding (VLU) tasks for probing the visual reasoning abilities of text encoder models, as well as various non-visual natural language understanding (NLU) tasks for comparison. We also contribute a novel zero-shot knowledge probing method, Stroop probing, for applying models such as CLIP to text-only tasks without needing a prediction head such as the masked language modelling head of models like BERT. We show that SOTA multimodally trained text encoders outperform unimodally trained text encoders on the VLU tasks while being under-performed by them on the NLU tasks, lending new context to previously mixed results regarding the NLU capabilities of multimodal models. We conclude that exposure to images during pretraining affords inherent visual reasoning knowledge that is reflected in language-only tasks that require implicit visual reasoning. Our findings bear importance in the broader context of multimodal learning, providing principled guidelines for the choice of text encoders used in such contexts
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code will be made available at https://isbertblind.github.io/.

----

## [647] Seeing What You Miss: Vision-Language Pre-training with Semantic Completion Learning

**Authors**: *Yatai Ji, Rongcheng Tu, Jie Jiang, Weijie Kong, Chengfei Cai, Wenzhe Zhao, Hongfa Wang, Yujiu Yang, Wei Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00656](https://doi.org/10.1109/CVPR52729.2023.00656)

**Abstract**:

Cross-modal alignment is essential for vision-language pre-training (VLP) models to learn the correct corresponding information across different modalities. For this purpose, inspired by the success of masked language modeling (MLM) tasks in the NLP pre-training area, numerous masked modeling tasks have been proposed for VLP to further promote cross-modal interactions. The core idea of previous masked modeling tasks is to focus on reconstructing the masked tokens based on visible context for learning local-to-local alignment. However, most of them pay little attention to the global semantic features generated for the masked data, resulting in a limited cross-modal alignment ability of global representations. Therefore, in this paper, we propose a novel Semantic Completion Learning (SCL) task, complementary to existing masked modeling tasks, to facilitate global-to-local alignment. Specifically, the SCL task complements the missing semantics of masked data by capturing the corresponding information from the other modality, promoting learning more representative global features which have a great impact on the performance of downstream tasks. Moreover, we present a flexible vision encoder, which enables our model to perform image-text and video-text multimodal tasks simultaneously. Experimental results show that our proposed method obtains state-of-the-art performance on various vision-language benchmarks, such as visual question answering, image-text retrieval, and video-text retrieval.

----

## [648] Affordance Grounding from Demonstration Video to Target Image

**Authors**: *Joya Chen, Difei Gao, Kevin Qinghong Lin, Mike Zheng Shou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00657](https://doi.org/10.1109/CVPR52729.2023.00657)

**Abstract**:

Humans excel at learning from expert demonstrations and solving their own problems. To equip intelligent robots and assistants, such as AR glasses, with this ability, it is essential to ground human hand interactions (i.e., affordances) from demonstration videos and apply them to a target image like a user's AR glass view. This video-to-image affordance grounding task is challenging due to (1) the need to predict fine-grained affordances, and (2) the limited training data, which inadequately covers video-image discrepancies and negatively impacts grounding. To tackle them, we propose Affordance Transformer (Afformer), which has a fine-grained transformer-based decoder that gradually refines affordance grounding. Moreover, we introduce Mask Affordance Hand (MaskAHand), a self-supervised pre-training technique for synthesizing video-image data and simulating context changes, enhancing affordance grounding across video-image discrepancies. Afformer with MaskAHand pre-training achieves state-of-the-art performance on multiple benchmarks, including a sub-stantial 37% improvement on the OPRA dataset. Code is made available at https://github.com/showlab/afformer.

----

## [649] Leverage Interactive Affinity for Affordance Learning

**Authors**: *Hongchen Luo, Wei Zhai, Jing Zhang, Yang Cao, Dacheng Tao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00658](https://doi.org/10.1109/CVPR52729.2023.00658)

**Abstract**:

Perceiving potential “action possibilities” (i.e., affordance) regions of images and learning interactive functionalities of objects from human demonstration is a challenging task due to the diversity of human-object interactions. Prevailing affordance learning algorithms often adopt the label assignment paradigm and presume that there is a unique relationship between functional region and affordance label, yielding poor performance when adapting to unseen environments with large appearance variations. In this paper, we propose to leverage interactive affinity for affordance learning, i.e. extracting interactive affinity from human-object interaction and transferring it to non-interactive objects. Interactive affinity, which represents the contacts between different parts of the human body and local regions of the target object, can provide inherent cues of interconnectivity between humans and objects, thereby reducing the ambiguity of the perceived action possibilities. Specifically, we propose a pose-aided interactive affinity learning framework that exploits human pose to guide the network to learn the interactive affinity from human-object interactions. Particularly, a keypoint heuristic perception (KHP) scheme is devised to exploit the keypoint association of human pose to alleviate the uncertainties due to interaction diversities and contact occlusions. Besides, a contact-driven affordance learning (CAL) dataset is constructed by collecting and labeling over 5, 000 images. Experimental results demonstrate that our method outperforms the representative models regarding objective metrics and visual quality. Code and dataset: github.com/lhc1224/PIAL-Net.

----

## [650] DeAR: Debiasing Vision-Language Models with Additive Residuals

**Authors**: *Ashish Seth, Mayur Hemani, Chirag Agarwal*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00659](https://doi.org/10.1109/CVPR52729.2023.00659)

**Abstract**:

Large pre-trained vision-language models (VLMs) reduce the time for developing predictive models for various vision-grounded language downstream tasks by providing rich, adaptable image and text representations. However, these models suffer from societal biases owing to the skewed distribution of various identity groups in the training data. These biases manifest as the skewed similarity between the representations for specific text concepts and images of people of different identity groups and, therefore, limit the usefulness of such models in real-world high-stakes applications. In this work, we present Dear(Debiasing with Additive Residuals), a novel debiasing method that learns additive residual image representations to offset the original representations, ensuring fair output representations. In doing so, it reduces the ability of the representations to distinguish between the different identity groups. Further, we observe that the current fairness tests are performed on limited face image datasets that fail to indicate why a specific text concept should/should not apply to them. To bridge this gap and better evaluate Dear,we introduce the Protected Attribute Tag Association (pata)dataset - a new context-based bias benchmarking dataset for evaluating the fairness of large pre-trained VLMs. Additionally, Pataprovides visual context for a diverse human population in different scenarios with both positive and negative connotations. Experimental results for fairness and zero-shot performance preservation using multiple datasets demonstrate the efficacy of our framework. The dataset is released here.

----

## [651] Images Speak in Images: A Generalist Painter for In-Context Visual Learning

**Authors**: *Xinlong Wang, Wen Wang, Yue Cao, Chunhua Shen, Tiejun Huang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00660](https://doi.org/10.1109/CVPR52729.2023.00660)

**Abstract**:

In-context learning, as a new paradigm in NLP, allows the model to rapidly adapt to various tasks with only a handful of prompts and examples. But in computer vision, the difficulties for in-context learning lie in that tasks vary significantly in the output representations, thus it is unclear how to define the general-purpose task prompts that the vision model can understand and transfer to out-of-domain tasks. In this work, we present Painter, a generalist model which addresses these obstacles with an “image”-centric solution, that is, to redefine the output of core vision tasks as images, and specify task prompts as also images. With this idea, our training process is extremely simple, which performs standard masked image modeling on the stitch of input and output image pairs. This makes the model capable of performing tasks conditioned on visible image patches. Thus, during inference, we can adopt a pair of input and output images from the same task as the input condition, to indicate which task to perform. Without bells and whistles, our generalist Painter can achieve competitive performance compared to well-established task-specific models, on seven representative vision tasks ranging from high-level visual understanding to low-level image processing. In addition, Painter significantly outperforms recent generalist models on several challenging tasks.

----

## [652] Hyperbolic Contrastive Learning for Visual Representations beyond Objects

**Authors**: *Songwei Ge, Shlok Mishra, Simon Kornblith, Chun-Liang Li, David Jacobs*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00661](https://doi.org/10.1109/CVPR52729.2023.00661)

**Abstract**:

Although self-/un-supervised methods have led to rapid progress in visual representation learning, these methods generally treat objects and scenes using the same lens. In this paper, we focus on learning representations for objects and scenes that preserve the structure among them. Motivated by the observation that visually similar objects are close in the representation space, we argue that the scenes and objects should instead follow a hierarchical structure based on their compositionality. To exploit such a structure, we propose a contrastive learning framework where a Euclidean loss is used to learn object representations and a hyperbolic loss is used to encourage representations of scenes to lie close to representations of their constituent objects in a hyperbolic space. This novel hyperbolic objective encourages the scene-object hypernymy among the representations by optimizing the magnitude of their norms. We show that when pretraining on the COCO and OpenImages datasets, the hyperbolic loss improves downstream performance of several baselines across multiple datasets and tasks, including image classification, object detection, and semantic segmentation. We also show that the properties of the learned representations allow us to solve various vision tasks that involve the interaction between scenes and objects in a zero-shot fashion.

----

## [653] Picture that Sketch: Photorealistic Image Generation from Abstract Sketches

**Authors**: *Subhadeep Koley, Ayan Kumar Bhunia, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00662](https://doi.org/10.1109/CVPR52729.2023.00662)

**Abstract**:

Given an abstract, deformed, ordinary sketch from untrained amateurs like you and me, this paper turns it into a photorealistic image – just like those shown in Fig. 1(a), all non-cherry-picked. We differ significantly from prior art in that we do not dictate an edgemap-like sketch to start with, but aim to work with abstract free-hand human sketches. In doing so, we essentially democratise the sketch-to-photo pipeline, “picturing” a sketch regardless of how good you sketch. Our contribution at the outset is a decoupled encoder-decoder training paradigm, where the decoder is a StyleGAN trained on photos only. This importantly ensures that generated results are always photorealistic. The rest is then all centred around how best to deal with the abstraction gap between sketch and photo. For that, we propose an autoregressive sketch mapper trained on sketch-photo pairs that maps a sketch to the StyleGAN latent space. We further introduce specific designs to tackle the abstract nature of human sketches, including a fine-grained discriminative loss on the back of a trained sketch-photo retrieval model, and a partial-aware sketch augmentation strategy. Finally, we showcase a few downstream tasks our generation model enables, amongst them is showing how fine-grained sketch-based image retrieval, a well-studied problem in the sketch community, can be reduced to an image (generated) to image retrieval task, surpassing state-of-the-arts. We put forward generated results in the supplementary for everyone to scrutinise. Project page: https://subhadeepkoley.github.io/PictureThatSketch

----

## [654] GeneCIS: A Benchmark for General Conditional Image Similarity

**Authors**: *Sagar Vaze, Nicolas Carion, Ishan Misra*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00663](https://doi.org/10.1109/CVPR52729.2023.00663)

**Abstract**:

We argue that there are many notions of ‘similarity’ and that models, like humans, should be able to adapt to these dynamically. This contrasts with most representation learning methods, supervised or self-supervised, which learn a fixed embedding function and hence implicitly assume a single notion of similarity. For instance, models trained on ImageNet are biased towards object categories, while a user might prefer the model to focus on colors, textures or specific elements in the scene. In this paper, we propose the GeneCIS ('genesis') benchmark, which measures models' ability to adapt to a range of similarity conditions. Extending prior work, our benchmark is designed for zero-shot evaluation only, and hence considers an open-set of similarity conditions. We find that baselines from powerful CLIP models struggle on GeneCIS and that performance on the benchmark is only weakly correlated with ImageNet accuracy, suggesting that simply scaling existing methods is not fruitful. We further propose a simple, scalable solution based on automatically mining information from existing image-caption datasets. We find our method offers a substantial boost over the baselines on GeneCIS, and further improves zero-shot performance on related image retrieval benchmarks. In fact, though evaluated zero-shot, our model surpasses state-of-the-art supervised models on MIT-States.

----

## [655] Exploiting Unlabelled Photos for Stronger Fine-Grained SBIR

**Authors**: *Aneeshan Sain, Ayan Kumar Bhunia, Subhadeep Koley, Pinaki Nath Chowdhury, Soumitri Chattopadhyay, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00664](https://doi.org/10.1109/CVPR52729.2023.00664)

**Abstract**:

This paper advances the fine-grained sketch-based image retrieval (FG-SBIR) literature by putting forward a strong baseline that overshoots prior state-of-the-arts by ≈11 %. This is not via complicated design though, but by addressing two critical issues facing the community (i) the gold standard triplet loss does not enforce holistic latent space geometry, and (ii) there are never enough sketches to train a high accuracy model. For the former, we propose a simple modification to the standard triplet loss, that explicitly enforces separation amongst photos/sketch instances. For the latter, we put forward a novel knowledge distillation module can leverage photo data for model training. Both modules are then plugged into a novel plug-n-playable training paradigm that allows for more stable training. More specifically, for (i) we employ an intra-modal triplet loss amongst sketches to bring sketches of the same instance closer from others, and one more amongst photos to push away different photo instances while bringing closer a structurally augmented version of the same photo (offering a gain of ≈4-6%). To tackle (ii), we first pre-train a teacher on the large set of unlabelled photos over the aforementioned intra-modal photo triplet loss. Then we distill the contextual similarity present amongst the instances in the teacher's embedding space to that in the student's embedding space, by matching the distribution over inter-feature distances of respective samples in both embedding spaces (delivering a further gain of ≈ 4-5%). Apart from outperforming prior arts significantly, our model also yields satisfactory results on generalising to new classes. Project page: https://aneeshan95.github.io/Sketch_PVT/

----

## [656] Parts2Words: Learning Joint Embedding of Point Clouds and Texts by Bidirectional Matching Between Parts and Words

**Authors**: *Chuan Tang, Xi Yang, Bojian Wu, Zhizhong Han, Yi Chang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00665](https://doi.org/10.1109/CVPR52729.2023.00665)

**Abstract**:

Shape-Text matching is an important task of high-level shape understanding. Current methods mainly represent a 3D shape as multiple 2D rendered views, which obviously can not be understood well due to the structural ambiguity caused by self-occlusion in the limited number of views. To resolve this issue, we directly represent 3D shapes as point clouds, and propose to learn joint embedding of point clouds and texts by bidirectional matching between parts from shapes and words from texts. Specifically, we first segment the point clouds into parts, and then leverage optimal transport method to match parts and words in an optimized feature space, where each part is represented by aggregating features of all points within it and each word is abstracted by its contextual information. We optimize the feature space in order to enlarge the similarities between the paired training samples, while simultaneously maximizing the margin between the unpaired ones. Experiments demonstrate that our method achieves a significant improvement in accuracy over the SOTAs on multi-modal retrieval tasks under the Text2Shape dataset. Codes are available at here.

----

## [657] Detecting and Grounding Multi-Modal Media Manipulation

**Authors**: *Rui Shao, Tianxing Wu, Ziwei Liu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00667](https://doi.org/10.1109/CVPR52729.2023.00667)

**Abstract**:

Misinformation has become a pressing issue. Fake media, in both visual and textual forms, is widespread on the web. While various deepfake detection and text fake news detection methods have been proposed, they are only designed for single-modality forgery based on binary classification, let alone analyzing and reasoning subtle forgery traces across different modalities. In this paper, we high-light a new research problem for multi-modal fake media, namely Detecting and Grounding Multi-Modal Media Manipulation (DGM
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
). DGM
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
 aims to not only detect the authenticity of multi-modal media, but also ground the manipulated content (i.e., image bounding boxes and text tokens), which requires deeper reasoning of multi-modal media manipulation. To support a large-scale investigation, we construct the first DGM
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
 dataset, where image-text pairs are manipulated by various approaches, with rich annotation of diverse manipulations. Moreover, we propose a novel HierArchical Multi-modal Manipulation rEasoning tRansformer (HAMMER) to fully capture the fine-grained interaction between different modalities. HAMMER performs 1) manipulation-aware contrastive learning between two uni-modal encoders as shallow manipulation reasoning, and 2) modality-aware cross-attention by multi-modal aggregator as deep manipulation reasoning. Dedicated manipulation detection and grounding heads are integrated from shallow to deep levels based on the interacted multi-modal information. Finally, we build an extensive bench-mark and set up rigorous evaluation metrics for this new research problem. Comprehensive experiments demonstrate the superiority of our model; several valuable observations are also revealed to facilitate future research in multi-modal media manipulation.

----

## [658] Positive-Augmented Contrastive Learning for Image and Video Captioning Evaluation

**Authors**: *Sara Sarto, Manuele Barraco, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00668](https://doi.org/10.1109/CVPR52729.2023.00668)

**Abstract**:

The CLIP model has been recently proven to be very effective for a variety of cross-modal tasks, including the evaluation of captions generated from vision-and-language architectures. In this paper, we propose a new recipe for a contrastive-based evaluation metric for image captioning, namely Positive-Augmented Contrastive learning Score (PAC-S), that in a novel way unifies the learning of a contrastive visual-semantic space with the addition of generated images and text on curated data. Experiments spanning several datasets demonstrate that our new metric achieves the highest correlation with human judgments on both images and videos, outperforming existing referencebased metrics like CIDEr and SPICE and reference-free metrics like CLIP-Score. Finally, we test the system-level correlation of the proposed metric when considering popular image captioning approaches, and assess the impact of employing different cross-modal features. Our source code and trained models are publicly available at: https://github.com/aimagelab/pacscore.

----

## [659] Similarity Maps for Self-Training Weakly-Supervised Phrase Grounding

**Authors**: *Tal Shaharabany, Lior Wolf*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00669](https://doi.org/10.1109/CVPR52729.2023.00669)

**Abstract**:

A phrase grounding model receives an input image and a text phrase and outputs a suitable localization map. We present an effective way to refine a phrase ground model by considering self-similarity maps extracted from the latent representation of the model's image encoder. Our main insights are that these maps resemble localization maps and that by combining such maps, one can obtain useful pseudo-labels for performing self-training. Our results surpass, by a large margin, the state of the art in weakly supervised phrase grounding. A similar gap in performance is obtained for a recently proposed downstream task called WWbL, in which only the image is input, without any text. Our code is available at https://github.com/talshaharabany/Similarity-Maps-for-Self-Training-Weakly-Supervised-Phrase-Grounding

----

## [660] Cross-Domain Image Captioning with Discriminative Finetuning

**Authors**: *Roberto Dessì, Michele Bevilacqua, Eleonora Gualdoni, Nathanaël Carraz Rakotonirina, Francesca Franzon, Marco Baroni*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00670](https://doi.org/10.1109/CVPR52729.2023.00670)

**Abstract**:

Neural captioners are typically trained to mimic human-generated references without optimizing for any specific communication goal, leading to problems such as the generation of vague captions. In this paper, we show that fine-tuning an out-of-the-box neural captioner with a self-supervised discriminative communication objective helps to recover a plain, visually descriptive language that is more informative about image contents. Given a target image, the system must learn to produce a description that enables an out-of-the-box text-conditioned image retriever to iden-tify such image among a set of candidates. We experiment with the popular ClipCap captioner; also replicating the main results with BLIP. In terms of similarity to ground-truth human descriptions, the captions emerging from dis-criminative finetuning lag slightly behind those generated by the non-finetuned model, when the latter is trained and tested on the same caption dataset. However, when the model is used without further tuning to generate captions for out-of-domain datasets, our discriminatively-finetuned captioner generates descriptions that resemble human ref-erences more than those produced by the same captioner wihtout finetuning. We further show that, on the Concep-tual Captions dataset, discriminatively finetuned captions are more helpful than either vanilla ClipCap captions or ground-truth captions for human annotators tasked with an image discrimination task.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at https://github.com/facebookresearch/EGG/tree/main/egg/zoo/discriminative_captioner.

----

## [661] EXIF as Language: Learning Cross-Modal Associations between Images and Camera Metadata

**Authors**: *Chenhao Zheng, Ayush Shrivastava, Andrew Owens*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00671](https://doi.org/10.1109/CVPR52729.2023.00671)

**Abstract**:

We learn a visual representation that captures information about the camera that recorded a given photo. To do this, we train a multimodal embedding between image patches and the EXIF metadata that cameras automatically insert into image files. Our model represents this meta-data by simply converting it to text and then processing it with a transformer. The features that we learn significantly outperform other self-supervised and supervised features on downstream image forensics and calibration tasks. In particular, we successfully localize spliced image regions “zero shot” by clustering the visual embeddings for all of the patches within an image.

----

## [662] Uncurated Image-Text Datasets: Shedding Light on Demographic Bias

**Authors**: *Noa Garcia, Yusuke Hirota, Yankun Wu, Yuta Nakashima*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00672](https://doi.org/10.1109/CVPR52729.2023.00672)

**Abstract**:

The increasing tendency to collect large and uncurated datasets to train vision-and-language models has raised concerns about fair representations. It is known that even small but manually annotated datasets, such as MSCOCO, are affected by societal bias. This problem, far from being solved, may be getting worse with data crawled from the Internet without much control. In addition, the lack of tools to analyze societal bias in big collections of images makes addressing the problem extremely challenging. Our first contribution is to annotate part of the Google Conceptual Captions dataset, widely used for training vision-and-language models, with four demographic and two contextual attributes. Our second contribution is to conduct a comprehensive analysis of the annotations, focusing on how different demographic groups are represented. Our last contribution lies in evaluating three prevailing vision-and-language tasks: image captioning, text-image CLIP embeddings, and text-to-image generation, showing that societal bias is a persistent problem in all of them. https://github.com/noagarcia/phase

----

## [663] Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training

**Authors**: *Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, Dhruv Mahajan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00673](https://doi.org/10.1109/CVPR52729.2023.00673)

**Abstract**:

Vision-language models trained with contrastive learning on large-scale noisy data are becoming increasingly popular for zero-shot recognition problems. In this paper we improve the following three aspects of the contrastive pre-training pipeline: dataset noise, model initialization and the training objective. First, we propose a straightforward filtering strategy titled Complexity, Action, and Text-spotting (CAT) that significantly reduces dataset size, while achieving improved performance across zero-shot vision-language tasks. Next, we propose an approach titled Concept Distillation to leverage strong unimodal representations for contrastive training that does not increase training complexity while outperforming prior work. Finally, we modify the traditional contrastive alignment objective, and propose an importance-sampling approach to up-sample the importance of hard-negatives without adding additional complexity. On an extensive zero-shot benchmark of 29 tasks, our Distilled and Hard-negative Training (DiHT) approach improves on 20 tasks compared to the baseline. Furthermore, for few-shot linear probing, we propose a novel approach that bridges the gap between zero-shot and few-shot performance, substantially improving over prior work. Models are available at github.com/facebookresearch/diht.

----

## [664] Turning a CLIP Model into a Scene Text Detector

**Authors**: *Wenwen Yu, Yuliang Liu, Wei Hua, Deqiang Jiang, Bo Ren, Xiang Bai*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00674](https://doi.org/10.1109/CVPR52729.2023.00674)

**Abstract**:

The recent large-scale Contrastive Language-Image Pretraining (CLIP) model has shown great potential in various downstream tasks via leveraging the pretrained vision and language knowledge. Scene text, which contains rich textual and visual information, has an inherent connection with a model like CLIP. Recently, pretraining approaches based on vision language models have made effective progresses in the field of text detection. In contrast to these works, this paper proposes a new method, termed TCM, focusing on Turning the CLIP Model directly for text detection without pretraining process. We demonstrate the advantages of the proposed TCM as follows: (1) The underlying principle of our framework can be applied to improve existing scene text detector. (2) It facilitates the few-shot training capability of existing methods, e.g., by using 10% of labeled data, we significantly improve the performance of the baseline method with an average of 22% in terms of the F-measure on 4 benchmarks. (3) By turning the CLIP model into existing scene text detection methods, we further achieve promising domain adaptation ability. The code will be publicly released at https://github.com/wenwenyu/TCM.

----

## [665] ScanDMM: A Deep Markov Model of Scanpath Prediction for 360° Images

**Authors**: *Xiangjie Sui, Yuming Fang, Hanwei Zhu, Shiqi Wang, Zhou Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00675](https://doi.org/10.1109/CVPR52729.2023.00675)

**Abstract**:

Scanpath prediction for 360° images aims to produce dynamic gaze behaviors based on the human visual perception mechanism. Most existing scanpath prediction methods for 360° images do not give a complete treatment of the time-dependency when predicting human scanpath, resulting in inferior performance and poor generalizability. In this paper, we present a scanpath prediction method for 360° images by designing a novel Deep Markov Model (DMM) architecture, namely ScanDMM. We propose a semantics-guided transition function to learn the nonlinear dynamics of time-dependent attentional landscape. Moreover, a state initialization strategy is proposed by considering the starting point of viewing, enabling the model to learn the dynamics with the correct “launcher”. We further demonstrate that our model achieves state-of-the-art performance on four 360° image databases, and exhibit its generalizability by presenting two applications of applying scanpath prediction models to other visual tasks - saliency detection and image quality assessment, expecting to provide profound insights into these fields.

----

## [666] CrOC: Cross-View Online Clustering for Dense Visual Representation Learning

**Authors**: *Thomas Stegmüller, Tim Lebailly, Behzad Bozorgtabar, Tinne Tuytelaars, Jean-Philippe Thiran*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00676](https://doi.org/10.1109/CVPR52729.2023.00676)

**Abstract**:

Learning dense visual representations without labels is an arduous task and more so from scene-centric data. We propose to tackle this challenging problem by proposing a Cross-view consistency objective with an Online Clustering mechanism (CrOC) to discover and segment the semantics of the views. In the absence of hand-crafted priors, the resulting method is more generalizable and does not require a cumbersome pre-processing step. More importantly, the clustering algorithm conjointly operates on the features of both views, thereby elegantly bypassing the issue of content not represented in both views and the ambiguous matching of objects from one crop to the other. We demonstrate excellent performance on linear and unsupervised segmentation transfer tasks on various datasets and similarly for video object segmentation. Our code and pre-trained models are publicly available at https://github.com/stegmuel/CrOC.

----

## [667] PLA: Language-Driven Open-Vocabulary 3D Scene Understanding

**Authors**: *Runyu Ding, Jihan Yang, Chuhui Xue, Wenqing Zhang, Song Bai, Xiaojuan Qi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00677](https://doi.org/10.1109/CVPR52729.2023.00677)

**Abstract**:

Open-vocabulary scene understanding aims to localize and recognize unseen categories beyond the annotated label space. The recent breakthrough of 2D open-vocabulary perception is largely driven by Internet-scale paired image-text data with rich vocabulary concepts. However, this success cannot be directly transferred to 3D scenarios due to the inaccessibility of large-scale 3D-text pairs. To this end, we propose to distill knowledge encoded in pretrained vision-language (VL) foundation models through captioning multi-view images from 3D, which allows explicitly associating 3D and semantic-rich captions. Further, to foster coarse-to-fine visual-semantic representation learning from captions, we design hierarchical 3D-caption pairs, leveraging geometric constraints between 3D scenes and multi-view images. Finally, by employing contrastive learning, the model learns language-aware embeddings that connect 3D and text for open-vocabulary tasks. Our method not only remarkably outperforms baseline methods by 25.8% ~ 44.7% hIoU and 14.5% ~ 50.4% hAP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
 in open-vocabulary semantic and instance segmentation, but also shows robust transferability on challenging zero-shot domain transfer tasks. See the project website at https://dingry.github.io/projects/PLA.

----

## [668] CLIP2Scene: Towards Label-efficient 3D Scene Understanding by CLIP

**Authors**: *Runnan Chen, Youquan Liu, Lingdong Kong, Xinge Zhu, Yuexin Ma, Yikang Li, Yuenan Hou, Yu Qiao, Wenping Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00678](https://doi.org/10.1109/CVPR52729.2023.00678)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) achieves promising results in 2D zero-shot and few-shot learning. Despite the impressive performance in 2D, applying CLIP to help the learning in 3D scene understanding has yet to be explored. In this paper, we make the first attempt to investigate how CLIP knowledge benefits 3D scene understanding. We propose CLIP2Scene, a simple yet effective framework that transfers CLIP knowledge from 2D image-text pre-trained models to a 3D point cloud network. We show that the pre-trained 3D network yields impressive performance on various downstream tasks, i.e., annotation-free and fine-tuning with labelled data for semantic segmentation. Specifically, built upon CLIP, we design a Semantic-driven Cross-modal Contrastive Learning framework that pre-trains a 3D network via semantic and spatial-temporal consistency regularization. For the former, we first leverage CLIP's text semantics to select the positive and negative point samples and then employ the contrastive loss to train the 3D network. In terms of the latter, we force the consistency between the temporally coherent point cloud features and their corresponding image features. We conduct experiments on SemanticKITTI, nuScenes, and ScanNet. For the first time, our pre-trained network achieves annotation-free 3D semantic segmentation with 20.8% and 25.08% mIoU on nuScenes and ScanNet, respectively. When fine-tuned with 1% or 100% labelled data, our method significantly outperforms other self-supervised methods, with improvements of 8% and 1% mIoU, respectively. Furthermore, we demonstrate the generalizability for handling cross-domain datasets. Code is publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/runnanchen/CLIP2Scene..

----

## [669] CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching

**Authors**: *Xiaoshi Wu, Feng Zhu, Rui Zhao, Hongsheng Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00679](https://doi.org/10.1109/CVPR52729.2023.00679)

**Abstract**:

Open-vocabulary detection (OVD) is an object detection task aiming at detecting objects from novel categories beyond the base categories on which the detector is trained. Recent OVD methods rely on large-scale visual-language pre-trained models, such as CLIP, for recognizing novel objects. We identify the two core obstacles that need to be tackled when incorporating these models into detector training: (1) the distribution mismatch that happens when applying a VL-model trained on whole images to region recognition tasks; (2) the difficulty of localizing objects of unseen classes. To overcome these obstacles, we propose CORA, a DETR-style framework that adapts CLIP for Open-vocabulary detection by Region prompting and Anchor pre-matching. Region prompting mitigates the whole-to-region distribution gap by prompting the region features of the CLIP-based region classifier. Anchor pre-matching helps learning generalizable object localization by a class-aware matching mechanism. We evaluate CORA on the COCO OVD benchmark, where we achieve 41.7 AP50 on novel classes, which outperforms the previous SOTA by 2.4 AP50 even without resorting to extra training data. When extra training data is available, we train CORA+ on both ground-truth base-category annotations and additional pseudo bounding box labels computed by CORA. CORA+ achieves 43.1 AP50 on the COCO OVD benchmark and 28.1 box APr on the LVIS OVD benchmark. The code is available at https://github.com/tgxs002/CORA.

----

## [670] Open-vocabulary Attribute Detection

**Authors**: *María Alejandra Bravo, Sudhanshu Mittal, Simon Ging, Thomas Brox*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00680](https://doi.org/10.1109/CVPR52729.2023.00680)

**Abstract**:

Vision-language modeling has enabled open-vocabulary tasks where predictions can be queried using any text prompt in a zero-shot manner. Existing open-vocabulary tasks focus on object classes, whereas research on object attributes is limited due to the lack of a reliable attribute-focused evaluation benchmark. This paper introduces the Open- Vocabulary Attribute Detection (OVAD) task and the corresponding OVAD benchmark. The objective of the novel task and benchmark is to probe object-level attribute information learned by vision-language models. To this end, we created a clean and densely annotated test set covering 117 attribute classes on the 80 object classes of MS COCO. It includes positive and negative annotations, which enables open-vocabulary evaluation. Overall, the benchmark consists of 1.4 million annotations. For reference, we provide a first baseline method for open-vocabulary attribute detection. Moreover, we demonstrate the benchmark's value by studying the attribute detection performance of several foundation models.

----

## [671] Learning to Detect and Segment for Open Vocabulary Object Detection

**Authors**: *Tao Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00681](https://doi.org/10.1109/CVPR52729.2023.00681)

**Abstract**:

Open vocabulary object detection has been greatly advanced by the recent development of vision-language pretrained model, which helps recognize novel objects with only semantic categories. The prior works mainly focus on knowledge transferring to the object proposal classification and employ class-agnostic box and mask prediction. In this work, we propose CondHead, a principled dynamic network design to better generalize the box regression and mask segmentation for open vocabulary setting. The core idea is to conditionally parameterize the network heads on semantic embedding and thus the model is guided with class-specific knowledge to better detect novel categories. Specifically, CondHead is composed of two streams of network heads, the dynamically aggregated head and dynamically generated head. The former is instantiated with a set of static heads that are conditionally aggregated, these heads are optimized as experts and are expected to learn sophisticated prediction. The latter is instantiated with dynamically generated parameters and encodes general class-specific information. With such a conditional design, the detection model is bridged by the semantic embedding to offer strongly generalizable class-wise box and mask prediction. Our method brings significant improvement to the state-of-the-art open vocabulary object detection methods with very minor overhead, e.g., it surpasses a RegionClip model by 3.0 detection AP on novel categories, with only 1.1% more computation.

----

## [672] Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP

**Authors**: *Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, Diana Marculescu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00682](https://doi.org/10.1109/CVPR52729.2023.00682)

**Abstract**:

Open-vocabulary semantic segmentation aims to segment an image into semantic regions according to text descriptions, which may not have been seen during training. Recent two-stage methods first generate class-agnostic mask proposals and then leverage pre-trained vision-language models, e.g., CLIP, to classify masked regions. We identify the performance bottleneck of this paradigm to be the pre-trained CLIP model, since it does not perform well on masked images. To address this, we propose to finetune CLIP on a collection of masked image regions and their corresponding text descriptions. We collect training data by mining an existing image-caption dataset (e.g., COCO Captions), using CLIP to match masked image regions to nouns in the image captions. Compared with the more precise and manually annotated segmentation labels with fixed classes (e.g., COCO-Stuff), we find our noisy but diverse dataset can better retain CLIP's generalization ability. Along with finetuning the entire model, we utilize the “blank” areas in masked images using a method we dub mask prompt tuning. Experiments demonstrate mask prompt tuning brings significant improvement without modifying any weights of CLIP, and it can further improve a fully finetuned model. In particular, when trained on COCO and evaluated on ADE20K-150, our best model achieves 29.6% mIoU, which is +8.5% higher than the previous state-of-the-art. For the first time, open-vocabulary generalist models match the performance of supervised specialist models in 2017 without dataset specific adaptations.

----

## [673] A Simple Framework for Text-Supervised Semantic Segmentation

**Authors**: *Muyang Yi, Quan Cui, Hao Wu, Cheng Yang, Osamu Yoshie, Hongtao Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00683](https://doi.org/10.1109/CVPR52729.2023.00683)

**Abstract**:

Text-supervised semantic segmentation is a novel research topic that allows semantic segments to emerge with image-text contrasting. However, pioneering methods could be subject to specifically designed network architectures. This paper shows that a vanilla contrastive language-image pretraining (CLIP) model is an effective text-supervised semantic segmentor by itself. First, we reveal that a vanilla CLIP is inferior to localization and segmentation due to its optimization being driven by densely aligning visual and language representations. Second, we propose the locality-driven alignment (LoDA) to address the problem, where CLIP optimization is driven by sparsely aligning local representations. Third, we propose a simple segmentation (SimSeg) framework. LoDA and SimSeg jointly amelio-rate a vanilla CLIP to produce impressive semantic segmentation results. Our method outperforms previous state-of-the-art methods on PASCAL VOC 2012, PASCAL Context and COCO datasets by large margins. Code and models are available at github.com/muyangyi/SimSeg.

----

## [674] GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts

**Authors**: *Haoran Geng, Helin Xu, Chengyang Zhao, Chao Xu, Li Yi, Siyuan Huang, He Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00684](https://doi.org/10.1109/CVPR52729.2023.00684)

**Abstract**:

For years, researchers have been devoted to generalizable object perception and manipulation, where cross-category generalizability is highly desired yet underexplored. In this work, we propose to learn such cross-category skills via Generalizable and Actionable Parts (GAParts). By identifying and defining 9 GAPart classes (lids, handles, etc.) in 27 object categories, we construct a large-scale part-centric interactive dataset, GAPartNet, where we provide rich, part-level annotations (semantics, poses) for 8,489 part instances on 1,166 objects. Based on GAPartNet, we investigate three cross-category tasks: part segmentation, part pose estimation, and partbased object manipulation. Given the significant domain gaps between seen and unseen object categories, we propose a robust 3D segmentation method from the perspective of domain generalization by integrating adversarial learning techniques. Our method outperforms all existing methods by a large margin, no matter on seen or unseen categories. Furthermore, with part segmentation and pose estimation results, we leverage the GAPart pose definition to design part-based manipulation heuristics that can generalize well to unseen object categories in both the simulator and the real world.

----

## [675] GeoLayoutLM: Geometric Pre-training for Visual Information Extraction

**Authors**: *Chuwei Luo, Changxu Cheng, Qi Zheng, Cong Yao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00685](https://doi.org/10.1109/CVPR52729.2023.00685)

**Abstract**:

Visual information extraction (VIE) plays an important role in Document Intelligence. Generally, it is divided into two tasks: semantic entity recognition (SER) and relation extraction (RE). Recently, pre-trained models for documents have achieved substantial progress in VIE, particularly in SER. However, most of the existing models learn the geometric representation in an implicit way, which has been found insufficient for the RE task since geometric information is especially crucial for RE. Moreover, we reveal another factor that limits the performance of RE lies in the objective gap between the pre-training phase and the finetuning phase for RE. To tackle these issues, we propose in this paper a multi-modal framework, named GeoLayoutLM, for VIE. GeoLayoutLM explicitly models the geometric relations in pre-training, which we call geometric pre-training. Geometric pre-training is achieved by three specially designed geometry-related pre-training tasks. Additionally, novel relation heads, which are pre-trained by the geometric pre-training tasks and fine-tuned for RE, are elaborately designed to enrich and enhance the feature representation. According to extensive experiments on standard VIE benchmarks, GeoLayoutLM achieves highly competitive scores in the SER task and significantly outperforms the previous state-of-the-arts for RE (e.g., the F1 score of RE on FUNSD is boosted from 80.35% to 89.45%) 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/AlibabaResearch/AdvancedLiterateMachinery.

----

## [676] Self-Supervised Image-to-Point Distillation via Semantically Tolerant Contrastive Loss

**Authors**: *Anas Mahmoud, Jordan S. K. Hu, Tianshu Kuai, Ali Harakeh, Liam Paull, Steven L. Waslander*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00686](https://doi.org/10.1109/CVPR52729.2023.00686)

**Abstract**:

An effective framework for learning 3D representations for perception tasks is distilling rich self-supervised image features via contrastive learning. However, image-to-point representation learning for autonomous driving datasets faces two main challenges: 1) the abundance of self-similarity, which results in the contrastive losses pushing away semantically similar point and image regions and thus disturbing the local semantic structure of the learned representations, and 2) severe class imbalance as pretraining gets dominated by over-represented classes. We propose to alleviate the self-similarity problem through a novel semantically tolerant image-to-point contrastive loss that takes into consideration the semantic distance between positive and negative image regions to minimize contrasting semantically similar point and image regions. Additionally, we address class imbalance by designing a class-agnostic balanced loss that approximates the degree of class imbalance through an aggregate sample-to-samples semantic similarity measure. We demonstrate that our semantically-tolerant contrastive loss with class balancing improves state-of-the-art 2D-to-3D representation learning in all evaluation settings on 3D semantic segmentation. Our method consistently outperforms state-of-the-art 2D-to-3D representation learning frameworks across a wide range of 2D self-supervised pretrained models.

----

## [677] Generative Semantic Segmentation

**Authors**: *Jiaqi Chen, Jiachen Lu, Xiatian Zhu, Li Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00687](https://doi.org/10.1109/CVPR52729.2023.00687)

**Abstract**:

We present Generative Semantic Segmentation (GSS), a generative learning approach for semantic segmentation. Uniquely, we cast semantic segmentation as an image-conditioned mask generation problem. This is achieved by replacing the conventional per-pixel discriminative learning with a latent prior learning process. Specifically, we model the variational posterior distribution of latent vari-ables given the segmentation mask. To that end, the segmentation mask is expressed with a special type of image (dubbed as maskige). This posterior distribution allows to generate segmentation masks unconditionally. To achieve semantic segmentation on a given image, we further intro-duce a conditioning network. It is optimized by minimizing the divergence between the posterior distribution of maskige (i.e. segmentation masks) and the latent prior distribution of input training images. Extensive experiments on standard benchmarks show that our GSS can perform competitively to prior art alternatives in the standard semantic segmentation setting, whilst achieving a new state of the art in the more challenging cross-domain setting.

----

## [678] MISC210K: A Large-Scale Dataset for Multi-Instance Semantic Correspondence

**Authors**: *Yixuan Sun, Yiwen Huang, Haijing Guo, Yuzhou Zhao, Runmin Wu, Yizhou Yu, Weifeng Ge, Wenqiang Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00688](https://doi.org/10.1109/CVPR52729.2023.00688)

**Abstract**:

Semantic correspondence have built up a new way for object recognition. However current single-object matching schema can be hard for discovering commonalities for a category and far from the real-world recognition tasks. To fill this gap, we design the multi-instance semantic correspondence task which aims at constructing the correspondence between multiple objects in an image pair. To support this task, we build a multi-instance semantic correspondence (MISC) dataset from COCO Detection 2017 task called MISC210K. We construct our dataset as three steps: (1) category selection and data cleaning; (2) keypoint design based on 3D models and object description rules; (3) human-machine collaborative annotation. Following these steps, we select 34 classes of objects with 4,812 challenging images annotated via a well designed semi-automatic workflow, and finally acquire 218,179 image pairs with instance masks and instance-level keypoint pairs annotated. We design a dual-path collaborative learning pipeline to train instance-level co-segmentation task and fine-grained level correspondence task together. Benchmark evaluation and further ablation results with detailed analysis are provided with three future directions proposed. Our project is available on https://github.com/YXSUNMADMAX/MISC210K.

----

## [679] MIANet: Aggregating Unbiased Instance and General Information for Few-Shot Semantic Segmentation

**Authors**: *Yong Yang, Qiong Chen, Yuan Feng, Tianlin Huang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00689](https://doi.org/10.1109/CVPR52729.2023.00689)

**Abstract**:

Existing few-shot segmentation methods are based on the meta-learning strategy and extract instance knowledge from a support set and then apply the knowledge to segment target objects in a query set. However, the extracted knowledge is insufficient to cope with the variable intra-class differences since the knowledge is obtained from a few samples in the support set. To address the problem, we propose a multi-information aggregation network (MIANet) that effectively leverages the general knowledge, i.e., semantic word embeddings, and instance information for accurate segmentation. Specifically, in MIANet, a general information module (GIM) is proposed to extract a general class prototype from word embeddings as a supplement to instance information. To this end, we design a triplet loss that treats the general class prototype as an anchor and samples positive-negative pairs from local features in the support set. The calculated triplet loss can transfer semantic similarities among language identities from a word embedding space to a visual representation space. To alleviate the model biasing towards the seen training classes and to obtain multi-scale information, we then introduce a non-parametric hierarchical prior module (HPM) to generate unbiased instance-level information via calculating the pixel-level similarity between the support and query image features. Finally, an information fusion module (IFM) combines the general and instance information to make predictions for the query image. Extensive experiments on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 show that MIANet yields superior performance and set a new state-of-the-art. Code is available at github.com/Aldrich2y/MIANet.

----

## [680] PACO: Parts and Attributes of Common Objects

**Authors**: *Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, Amir Mousavi, Yiwen Song, Abhimanyu Dubey, Dhruv Mahajan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00690](https://doi.org/10.1109/CVPR52729.2023.00690)

**Abstract**:

Object models are gradually progressing from predicting just category labels to providing detailed descriptions of object instances. This motivates the need for large datasets which go beyond traditional object masks and provide richer annotations such as part masks and attributes. Hence, we introduce PACO: Parts and Attributes of Common Objects. It spans 75 object categories, 456 object-part categories and 55 attributes across image (LVIS) and video (Eg04D) datasets. We provide 641K part masks an-notated across 260K object boxes, with roughly half of them exhaustively annotated with attributes as well. We design evaluation metrics and provide benchmark results for three tasks on the dataset: part mask segmentation, object and part attribute prediction and zero-shot instance detection. Dataset, models, and code are open-sourced at https://github.com/jacebookresearch/paco.

----

## [681] PartDistillation: Learning Parts from Instance Segmentation

**Authors**: *Jang Hyun Cho, Philipp Krähenbühl, Vignesh Ramanathan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00691](https://doi.org/10.1109/CVPR52729.2023.00691)

**Abstract**:

We present a scalable framework to learn part segmentation from object instance labels. State-of-the-art instance segmentation models contain a surprising amount of part information. However, much of this information is hidden from plain view. For each object instance, the part information is noisy, inconsistent, and incomplete. PartDistillation transfers the part information of an instance segmentation model into a part segmentation model through self-supervised self-training on a large dataset. The resulting segmentation model is robust, accurate, and generalizes well. We evaluate the model on various part segmentation datasets. Our model outperforms supervised part segmentation in zero-shot generalization performance by a large margin. Our model outperforms when finetuned on target datasets compared to supervised counterpart and other baselines especially in few-shot regime. Finally, our model provides a wider coverage of rare parts when evaluated over 10K object classes. Code is at https://github.com/facebookresearch/PartDistillation.

----

## [682] ACSeg: Adaptive Conceptualization for Unsupervised Semantic Segmentation

**Authors**: *Kehan Li, Zhennan Wang, Zesen Cheng, Runyi Yu, Yian Zhao, Guoli Song, Chang Liu, Li Yuan, Jie Chen*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00692](https://doi.org/10.1109/CVPR52729.2023.00692)

**Abstract**:

Recently, self-supervised large-scale visual pre-training models have shown great promise in representing pixel-level semantic relationships, significantly promoting the development of unsupervised dense prediction tasks, e.g., unsupervised semantic segmentation (USS). The extracted relationship among pixel-level representations typically contains rich class-aware information that semantically identical pixel embeddings in the representation space gather together to form sophisticated concepts. However, leveraging the learned models to ascertain semantically consistent pixel groups or regions in the image is non-trivial since over/ under-clustering overwhelms the conceptualization procedure under various semantic distributions of different images. In this work, we investigate the pixel-level semantic aggregation in self-supervised ViT pre-trained models as image Segmentation and propose the Adaptive Conceptualization approach for USS, termed ACSeg. Concretely, we explicitly encode concepts into learnable prototypes and design the Adaptive Concept Generator (ACG), which adaptively maps these prototypes to informative concepts for each image. Meanwhile, considering the scene complexity of different images, we propose the modularity loss to optimize ACG independent of the concept number based on estimating the intensity of pixel pairs belonging to the same concept. Finally, we turn the USS task into classifying the discovered concepts in an unsupervised manner. Extensive experiments with state-of-the-art results demonstrate the effectiveness of the proposed ACSeg.

----

## [683] Reliability in Semantic Segmentation: Are we on the Right Track?

**Authors**: *Pau de Jorge, Riccardo Volpi, Philip H. S. Torr, Grégory Rogez*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00693](https://doi.org/10.1109/CVPR52729.2023.00693)

**Abstract**:

Motivated by the increasing popularity of transformers in computer vision, in recent times there has been a rapid development of novel architectures. While in-domain performance follows a constant, upward trend, properties like robustness or uncertainty estimation are less explored-leaving doubts about advances in model reliability. Studies along these axes exist, but they are mainly limited to classification models. In contrast, we carry out a study on semantic segmentation, a relevant task for many real-world applications where model reliability is paramount. We analyze a broad variety of models, spanning from older ResNet-based architectures to novel transformers and assess their reliability based on four metrics: robustness, calibration, misclassification detection and out-of-distribution (OOD) detection. We find that while recent models are significantly more robust, they are not overall more reliable in terms of uncertainty estimation. We further explore methods that can come to the rescue and show that improving calibration can also help with other uncertainty metrics such as misclassification or OOD detection. This is the first study on modern segmentation models focused on both robustness and uncertainty estimation and we hope it will help practitioners and researchers interested in this fundamental vision task
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/naver/relis.

----

## [684] Rethinking the Correlation in Few-Shot Segmentation: A Buoys View

**Authors**: *Yuan Wang, Rui Sun, Tianzhu Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00694](https://doi.org/10.1109/CVPR52729.2023.00694)

**Abstract**:

Few-shot segmentation (FSS) aims to segment novel ob-jects in a given query image with only a few annotated support images. However, most previous best-performing methods, whether prototypical learning methods or affinity learning methods, neglect to alleviate false matches caused by their own pixel-level correlation. In this work, we rethink how to mitigate the false matches from the perspective of representative reference features (referred to as buoys), and propose a novel adaptive buoys correlation (ABC) network to rectify direct pairwise pixel-level correlation, including a buoys mining module and an adaptive correlation module. The proposed ABC enjoys several merits. First, to learn the buoys well without any correspondence supervision, we customize the buoys mining module according to the three characteristics of representativeness, task awareness and re-silience. Second, the proposed adaptive correlation module is responsible for further endowing buoy-correlation-based pixel matching with an adaptive ability. Extensive experimen-tal results with two different backbones on two challenging benchmarks demonstrate that our ABC, as a general plu-gin, achieves consistent improvements over several leading methods on both I-shot and 5-shot settings.

----

## [685] SIM: Semantic-aware Instance Mask Generation for Box-Supervised Instance Segmentation

**Authors**: *Ruihuang Li, Chenhang He, Yabin Zhang, Shuai Li, Liyi Chen, Lei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00695](https://doi.org/10.1109/CVPR52729.2023.00695)

**Abstract**:

Weakly supervised instance segmentation using only bounding box annotations has recently attracted much re-search attention. Most of the current efforts leverage low-level image features as extra supervision without explicitly exploiting the high-level semantic information of the objects, which will become ineffective when the foreground objects have similar appearances to the background or other objects nearby. We propose a new box-supervised instance segmentation approach by developing a Semantic-aware In-stance Mask (SIM) generation paradigm. Instead of heavily relying on local pair-wise affinities among neighboring pixels, we construct a group of category-wise feature cen-troids as prototypes to identify foreground objects and as-sign them semantic-level pseudo labels. Considering that the semantic-aware prototypes cannot distinguish different instances of the same semantics, we propose a self-correction mechanism to rectify the falsely activated regions while enhancing the correct ones. Furthermore, to handle the occlusions between objects, we tailor the Copy-Paste operation for the weakly-supervised instance segmentation task to augment challenging training data. Extensive experimental results demonstrate the superiority of our proposed SIM approach over other state-of-the-art methods. The source code: https://github.com/1slrh/SIM.

----

## [686] Endpoints Weight Fusion for Class Incremental Semantic Segmentation

**Authors**: *Jia-Wen Xiao, Chang-Bin Zhang, Jiekang Feng, Xialei Liu, Joost van de Weijer, Ming-Ming Cheng*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00696](https://doi.org/10.1109/CVPR52729.2023.00696)

**Abstract**:

Class incremental semantic segmentation (CISS) focuses on alleviating catastrophic forgetting to improve discrimination. Previous work mainly exploits regularization (e.g., knowledge distillation) to maintain previous knowledge in the current model. However, distillation alone often yields limited gain to the model since only the representations of old and new models are restricted to be consistent. In this paper, we propose a simple yet effective method to obtain a model with a strong memory of old knowledge, named Endpoints Weight Fusion (EWF). In our method, the model containing old knowledge is fused with the model retaining new knowledge in a dynamic fusion manner, strengthening the memory of old classes in ever changing distributions. In addition, we analyze the relationship between our fusion strategy and a popular moving average technique EMA, which reveals why our method is more suitable for class-incremental learning. To facilitate parameter fusion with closer distance in the parameter space, we use distillation to enhance the optimization process. Furthermore, we conduct experiments on two widely used datasets, achieving state-of-the-art performance.

----

## [687] Incrementer: Transformer for Class-Incremental Semantic Segmentation with Knowledge Distillation Focusing on Old Class

**Authors**: *Chao Shang, Hongliang Li, Fanman Meng, Qingbo Wu, Heqian Qiu, Lanxiao Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00697](https://doi.org/10.1109/CVPR52729.2023.00697)

**Abstract**:

Class-incremental semantic segmentation aims to incrementally learn new classes while maintaining the capability to segment old ones, and suffers catastrophic forgetting since the old-class labels are unavailable. Most existing methods are based on convolutional networks and prevent forgetting through knowledge distillation, which (1) need to add additional convolutional layers to predict new classes, and (2) ignore to distinguish different regions corresponding to old and new classes during knowledge distillation and roughly distill all the features, thus limiting the learning of new classes. Based on the above observations, we propose a new transformer framework for class-incremental semantic segmentation, dubbed Incrementer, which only needs to add new class tokens to the transformer decoder for new-class learning. Based on the Incrementer, we propose a new knowledge distillation scheme that focuses on the distillation in the old-class regions, which reduces the constraints of the old model on the new-class learning, thus improving the plasticity. Moreover, we propose a class deconfusion strategy to alleviate the overfitting to new classes and the confusion of similar classes. Our method is simple and effective, and extensive experiments show that our method outperforms the SOTAs by a large margin (5~15 absolute points boosts on both Pascal VOC and ADE20k). We hope that our Incrementer can serve as a new strong pipeline for class-incremental semantic segmentation.

----

## [688] Continuous Pseudo-Label Rectified Domain Adaptive Semantic Segmentation with Implicit Neural Representations

**Authors**: *Rui Gong, Qin Wang, Martin Danelljan, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00698](https://doi.org/10.1109/CVPR52729.2023.00698)

**Abstract**:

Unsupervised domain adaptation (UDA) for semantic segmentation aims at improving the model performance on the unlabeled target domain by leveraging a labeled source domain. Existing approaches have achieved impressive progress by utilizing pseudo-labels on the unlabeled target-domain images. Yet the low-quality pseudo-labels, arising from the domain discrepancy, inevitably hinder the adaptation. This calls for effective and accurate approaches to estimating the reliability of the pseudo-labels, in order to rectify them. In this paper, we propose to estimate the rectification values of the predicted pseudo-labels with implicit neural representations. We view the rectification value as a signal defined over the continuous spatial domain. Taking an image coordinate and the nearby deep features as inputs, the rectification value at a given coordinate is predicted as an output. This allows us to achieve high-resolution and detailed rectification values estimation, important for accurate pseudo-label generation at mask boundaries in particular. The rectified pseudo-labels are then leveraged in our rectification-aware mixture model (RMM) to be learned end-to-end and help the adaptation. We demonstrate the effectiveness of our approach on different UDA benchmarks, including synthetic-to-real and day-to-night. Our approach achieves superior results compared to state-of-the-art. The implementation is available at https://github.com/ETHRuiGong/IR2F.

----

## [689] Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation

**Authors**: *Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00699](https://doi.org/10.1109/CVPR52729.2023.00699)

**Abstract**:

In this work, we revisit the weak-to-strong consistency framework, popularized by FixMatch from semi-supervised classification, where the prediction of a weakly perturbed image serves as supervision for its strongly perturbed version. Intriguingly, we observe that such a simple pipeline already achieves competitive results against recent advanced works, when transferred to our segmentation scenario. Its success heavily relies on the manual design of strong data augmentations, however, which may be limited and inadequate to explore a broader perturbation space. Motivated by this, we propose an auxiliary feature perturbation stream as a supplement, leading to an expanded perturbation space. On the other, to sufficiently probe original image-level augmentations, we present a dual-stream perturbation technique, enabling two strong views to be simultaneously guided by a common weak view. Consequently, our overall Unified Dual-Stream Perturbations approach (UniMatch) surpasses all existing methods significantly across all evaluation protocols on the Pascal, Cityscapes, and COCO benchmarks. Its superiority is also demonstrated in remote sensing interpretation and medical image analysis. We hope our reproduced FixMatch and our results can inspire more future works.

----

## [690] Discriminative Co-Saliency and Background Mining Transformer for Co-Salient Object Detection

**Authors**: *Long Li, Junwei Han, Ni Zhang, Nian Liu, Salman H. Khan, Hisham Cholakkal, Rao Muhammad Anwer, Fahad Shahbaz Khan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00700](https://doi.org/10.1109/CVPR52729.2023.00700)

**Abstract**:

Most previous co-salient object detection works mainly focus on extracting co-salient cues via mining the consistency relations across images while ignore explicit exploration of background regions. In this paper, we propose a Discriminative co-saliency and background Mining Transformer framework (DMT) based on several economical multi-grained correlation modules to explicitly mine both co-saliency and background information and effectively model their discrimination. Specifically, we first propose a region-to-region correlation module for introducing inter-image relations to pixel-wise segmentation features while maintaining computational efficiency. Then, we use two types of pre-defined tokens to mine co-saliency and background information via our proposed contrast-induced pixel-to-token correlation and co-saliency token-to-token correlation modules. We also design a token-guided feature refinement module to enhance the discriminability of the segmentation features under the guidance of the learned tokens. We perform iterative mutual promotion for the segmentation feature extraction and token construction. Experimental results on three benchmark datasets demonstrate the effectiveness of our proposed method. The source code is available at: https://github.com/dragonlee258079/DMT.

----

## [691] Texture-Guided Saliency Distilling for Unsupervised Salient Object Detection

**Authors**: *Huajun Zhou, Bo Qiao, Lingxiao Yang, Jianhuang Lai, Xiaohua Xie*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00701](https://doi.org/10.1109/CVPR52729.2023.00701)

**Abstract**:

Deep Learning-based Unsupervised Salient Object Detection (USOD) mainly relies on the noisy saliency pseudo labels that have been generated from traditional handcraft methods or pre-trained networks. To cope with the noisy labels problem, a class of methods focus on only easy samples with reliable labels but ignore valuable knowledge in hard samples. In this paper, we propose a novel USOD method to mine rich and accurate saliency knowledge from both easy and hard samples. First, we propose a Confidence-aware Saliency Distilling (CSD) strategy that scores samples conditioned on samples' confidences, which guides the model to distill saliency knowledge from easy samples to hard samples progressively. Second, we propose a Boundary-aware Texture Matching (BTM) strategy to refine the boundaries of noisy labels by matching the textures around the predicted boundaries. Extensive experiments on RGB, RGB-D, RGB-T, and video SOD benchmarks prove that our method achieves state-of-the-art USOD performance. Code is available at www.github.com/moothes/A2S-v2.

----

## [692] An Erudite Fine-Grained Visual Classification Model

**Authors**: *Dongliang Chang, Yujun Tong, Ruoyi Du, Timothy M. Hospedales, Yi-Zhe Song, Zhanyu Ma*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00702](https://doi.org/10.1109/CVPR52729.2023.00702)

**Abstract**:

Current fine-grained visual classification (FGVC) models are isolated. In practice, we first need to identify the coarse-grained label of an object, then select the corresponding FGVC model for recognition. This hinders the application of FGVC algorithms in real-life scenarios. In this paper, we propose an erudite FGVC model jointly trained by several different datasets
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
In this paper, different datasets mean different fine-grained visual classification datasets., which can efficiently and accurately predict an object's fine-grained label across the combined label space. We found through a pilot study that positive and negative transfers co-occur when different datasets are mixed for training, i.e., the knowledge from other datasets is not always useful. Therefore, we first propose a feature disentanglement module and a feature re-fusion module to reduce negative transfer and boost positive transfer between different datasets. In detail, we reduce negative transfer by decoupling the deep features through many dataset-specific feature extractors. Subsequently, these are channel-wise re-fused to facilitate positive transfer. Finally, we propose a meta-learning based dataset-agnostic spatial attention layer to take full advantage of the multi-dataset training data, given that localisation is dataset-agnostic between different datasets. Experimental results across 11 different mixed-datasets built on four different FGVC datasets demonstrate the effectiveness of the proposed method. Furthermore, the proposed method can be easily combined with existing FGVC methods to obtain state-of-the-art results. Our code is available at https://github.com/PRIS-CV/An-Erudite-FGVC-Model.

----

## [693] Dynamic Graph Learning with Content-guided Spatial-Frequency Relation Reasoning for Deepfake Detection

**Authors**: *Yuan Wang, Kun Yu, Chen Chen, Xiyuan Hu, Silong Peng*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00703](https://doi.org/10.1109/CVPR52729.2023.00703)

**Abstract**:

With the springing up of face synthesis techniques, it is prominent in need to develop powerful face forgery detection methods due to security concerns. Some existing methods attempt to employ auxiliary frequency-aware information combined with CNN backbones to discover the forged clues. Due to the inadequate information interaction with image content, the extracted frequency features are thus spatially irrelavant, struggling to generalize well on increasingly realistic counterfeit types. To address this issue, we propose a Spatial-Frequency Dynamic Graph method to exploit the relation-aware features in spatial and frequency domains via dynamic graph learning. To this end, we introduce three well-designed components: 1) Content-guided Adaptive Frequency Extraction module to mine the content-adaptive forged frequency clues. 2) Multiple Domains Attention Map Learning module to enrich the spatial-frequency contextual features with multiscale attention maps. 3) Dynamic Graph Spatial-Frequency Feature Fusion Network to explore the high-order relation of spatial and frequency features. Extensive experiments on several benchmark show that our proposed method sustainedly exceeds the state-of-the-arts by a considerable margin.

----

## [694] ScaleDet: A Scalable Multi-Dataset Object Detector

**Authors**: *Yanbei Chen, Manchen Wang, Abhay Mittal, Zhenlin Xu, Paolo Favaro, Joseph Tighe, Davide Modolo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00704](https://doi.org/10.1109/CVPR52729.2023.00704)

**Abstract**:

Multi-dataset training provides a viable solution for exploiting heterogeneous large-scale datasets without extra annotation cost. In this work, we propose a scalable multi-dataset detector (ScaleDet) that can scale up its generalization across datasets when increasing the number of training datasets. Unlike existing multi-dataset learners that mostly rely on manual relabelling efforts or sophisticated optimizations to unify labels across datasets, we introduce a simple yet scalable formulation to derive a unified semantic label space for multi-dataset training. ScaleDet is trained by visual-textual alignment to learn the label assignment with label semantic similarities across datasets. Once trained, ScaleDet can generalize well on any given upstream and downstream datasets with seen and unseen classes. We conduct extensive experiments using LVIS, COCO, Objects365, OpenImages as upstream datasets, and 13 datasets from Object Detection in the Wild (ODinW) as downstream datasets. Our results show that ScaleDet achieves compelling strong model performance with an mAP of 50.7 on LVIS, 58.8 on COCO, 46.8 on Objects365, 76.2 on OpenImages, and 71.8 on ODinW, surpassing state-of-the-art detectors with the same backbone.

----

## [695] Multi-Centroid Task Descriptor for Dynamic Class Incremental Inference

**Authors**: *Tenghao Cai, Zhizhong Zhang, Xin Tan, Yanyun Qu, Guannan Jiang, Chengjie Wang, Yuan Xie*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00705](https://doi.org/10.1109/CVPR52729.2023.00705)

**Abstract**:

Incremental learning could be roughly divided into two categories, i.e., class- and task-incremental learning. The main difference is whether the task ID is given during evaluation. In this paper, we show this task information is indeed a strong prior knowledge, which will bring significant improvement over class-incremental learning baseline, e.g., DER [39]. Based on this observation, we propose a gate network to predict the task ID for class incremental inference. This is challenging as there is no explicit semantic relationship between categories in the concept of task. Therefore, we propose a multi-centroid task descriptor by assuming the data within a task can form multiple clusters. The cluster centers are optimized by pulling relevant sample-centroid pairs while pushing others away, which ensures that there is at least one centroid close to a given sample. To select relevant pairs, we use class prototypes as proxies and solve a bipartite matching problem, making the task descriptor representative yet not degenerate to uni-modal. As a result, our dynamic inference network is trained independently of baseline and provides a flexible, efficient solution to distinguish between tasks. Extensive experiments show our approach achieves state-of-the-art results, e.g., we achieve 72.41% average accuracy on CIFAR100-BOS50, outperforming DER by 3.40%.

----

## [696] Matching Is Not Enough: A Two-Stage Framework for Category-Agnostic Pose Estimation

**Authors**: *Min Shi, Zihao Huang, Xianzheng Ma, Xiaowei Hu, Zhiguo Cao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00706](https://doi.org/10.1109/CVPR52729.2023.00706)

**Abstract**:

Category-agnostic pose estimation (CAPE) aims to predict keypoints for arbitrary categories given support images with keypoint annotations. Existing approaches match the keypoints across the image for localization. However, such a one-stage matching paradigm shows inferior accuracy: the prediction heavily relies on the matching results, which can be noisy due to the open set nature in CAPE. For example, two mirror-symmetric keypoints (e.g., left and right eyes) in the query image can both trigger high similarity on certain support keypoints (eyes), which leads to duplicated or opposite predictions. To calibrate the inaccurate matching results, we introduce a two-stage framework, where matched keypoints from the first stage are viewed as similarity-aware position proposals. Then, the model learns to fetch relevant features to correct the initial proposals in the second stage. We instantiate the framework with a transformer model tailored for CAPE. The transformer encoder incorporates specific designs to improve the representation and similarity modeling in the first matching stage. In the second stage, similarity-aware proposals are packed as queries in the decoder for refinement via cross-attention. Our method surpasses the previous best approach by large margins on CAPE benchmark MP-100 on both accuracy and efficiency. Code available at github.com/flyinglynx/CapeFormer

----

## [697] Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection

**Authors**: *Chang Xu, Jian Ding, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00707](https://doi.org/10.1109/CVPR52729.2023.00707)

**Abstract**:

Detecting arbitrarily oriented tiny objects poses intense challenges to existing detectors, especially for label assignment. Despite the exploration of adaptive label assignment in recent oriented object detectors, the extreme geometry shape and limited feature of oriented tiny objects still induce severe mismatch and imbalance issues. Specifically, the position prior, positive sample feature, and instance are mismatched, and the learning of extreme-shaped objects is biased and unbalanced due to little proper feature supervision. To tackle these issues, we propose a dynamic prior along with the coarse-to-fine assigner, dubbed DCFL. For one thing, we model the prior, label assignment, and object representation all in a dynamic manner to alleviate the mismatch issue. For another, we leverage the coarse prior matching and finer posterior constraint to dynamically assign labels, providing appropriate and relatively balanced supervision for diverse instances. Extensive experiments on six datasets show substantial improvements to the baseline. Notably, we obtain the state-of-the-art performance for one-stage detectors on the DOTA-v1.5, DOTA-v2.0, and DIOR-R datasets under single-scale training and testing. Codes are available at https://github.com/Chasel-Tsui/mmrotate-dcfl.

----

## [698] Dense Distinct Query for End-to-End Object Detection

**Authors**: *Shilong Zhang, Xinjiang Wang, Jiaqi Wang, Jiangmiao Pang, Chengqi Lyu, Wenwei Zhang, Ping Luo, Kai Chen*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00708](https://doi.org/10.1109/CVPR52729.2023.00708)

**Abstract**:

One-to-one label assignment in object detection has successfully obviated the need for non-maximum suppression (NMS) as postprocessing and makes the pipeline end-to-end. However, it triggers a new dilemma as the widely used sparse queries cannot guarantee a high recall, while dense queries inevitably bring more similar queries and encounter optimization difficulties. As both sparse and dense queries are problematic, then what are the expected queries in end-to-end object detection? This paper shows that the solution should be Dense Distinct Queries (DDQ). Concretely, we first lay dense queries like traditional detectors and then select distinct ones for one-to-one assignments. DDQ blends the advantages of traditional and recent end-to-end detectors and significantly improves the performance of various detectors including FCN, R-CNN, and DETRs. Most impressively, DDQ-DETR achieves 52.1 AP on MS-COCO dataset within 12 epochs using a ResNet-50 backbone, outperforming all existing detectors in the same setting. DDQ also shares the benefit of end-to-end detectors in crowded scenes and achieves 93.8 AP on Crowd-Human. We hope DDQ can inspire researchers to consider the complementarity between traditional methods and end-to-end detectors. The source code can be found at https://github.com/jshilong/DDQ.

----

## [699] Meta-Tuning Loss Functions and Data Augmentation for Few-Shot Object Detection

**Authors**: *Berkan Demirel, Orhun Bugra Baran, Ramazan Gokberk Cinbis*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00709](https://doi.org/10.1109/CVPR52729.2023.00709)

**Abstract**:

Few-shot object detection, the problem of modelling novel object detection categories with few training instances, is an emerging topic in the area of few-shot learning and object detection. Contemporary techniques can be divided into two groups: fine-tuning based and meta-learning based approaches. While meta-learning approaches aim to learn dedicated meta-models for mapping samples to novel class models, fine-tuning approaches tackle few-shot detection in a simpler manner, by adapting the detection model to novel classes through gradient based optimization. Despite their simplicity, fine-tuning based approaches typically yield competitive detection results. Based on this observation, we focus on the role of loss functions and augmentations as the force driving the fine-tuning process, and propose to tune their dynamics through meta-learning principles. The proposed training scheme, therefore, allows learning inductive biases that can boost few-shot detection, while keeping the advantages of fine-tuning based approaches. In addition, the proposed approach yields interpretable loss functions, as opposed to highly parametric and complex few-shot meta-models. The experimental results highlight the merits of the proposed scheme, with significant improvements over the strong fine-tuning based few-shot detection baselines on benchmark Pascal VOC and MS-COCO datasets, in terms of both standard and generalized few-shot performance metrics.

----

## [700] One-to-Few Label Assignment for End-to-End Dense Detection

**Authors**: *Shuai Li, Minghan Li, Ruihuang Li, Chenhang He, Lei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00710](https://doi.org/10.1109/CVPR52729.2023.00710)

**Abstract**:

One-to-one (o2o) label assignment plays a key role for transformer based end-to-end detection, and it has been recently introduced in fully convolutional detectors for end-to-end dense detection. However, o2o can degrade the feature learning efficiency due to the limited number of positive samples. Though extra positive samples are introduced to mitigate this issue in recent DETRs, the computation of self- and cross- attentions in the decoder limits its practical application to dense and fully convolutional detectors. In this work, we propose a simple yet effective one-to-few (o2f) label assignment strategy for end-to-end dense detection. Apart from defining one positive and many negative anchors for each object, we define several soft anchors, which serve as positive and negative samples simultaneously. The positive and negative weights of these soft anchors are dynamically adjusted during training so that they can contribute more to “representation learning” in the early training stage, and contribute more to “duplicated prediction removal” in the later stage. The detector trained in this way can not only learn a strong feature representation but also perform end-to-end dense detection. Experiments on COCO and CrowdHuman datasets demonstrate the effectiveness of the o2f scheme. Code is available at https://github.com/strongwolf/o2f

----

## [701] Test Time Adaptation with Regularized Loss for Weakly Supervised Salient Object Detection

**Authors**: *Olga Veksler*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00711](https://doi.org/10.1109/CVPR52729.2023.00711)

**Abstract**:

It is well known that CNNs tend to overfit to the training data. Test-time adaptation is an extreme approach to deal with overfitting: given a test image, the aim is to adapt the trained model to that image. Indeed nothing can be closer to the test data than the test image itself. The main difficulty of test-time adaptation is that the ground truth is not available. Thus test-time adaptation, while intriguing, applies to only a few scenarios where one can design an effective loss function that does not require ground truth. We propose the first approach for test-time Salient Object Detection (SOD) in the context of weak supervision. Our approach is based on a so called regularized loss function, which can be used for training CNN when pixel precise ground truth is unavail-able. Regularized loss tends to have lower values for the more likely object segments, and thus it can be used to fine-tune an already trained CNN to a given test image, adapting to images unseen during training. We develop a regularized loss function particularly suitable for test-time adaptation and show that our approach significantly outperforms prior work for weakly supervised SOD.

----

## [702] MixTeacher: Mining Promising Labels with Mixed Scale Teacher for Semi-Supervised Object Detection

**Authors**: *Liang Liu, Boshen Zhang, Jiangning Zhang, Wuhao Zhang, Zhenye Gan, Guanzhong Tian, Wenbing Zhu, Yabiao Wang, Chengjie Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00712](https://doi.org/10.1109/CVPR52729.2023.00712)

**Abstract**:

Scale variation across object instances remains a key challenge in object detection task. Despite the remarkable progress made by modern detection models, this challenge is particularly evident in the semi-supervised case. While existing semi-supervised object detection methods rely on strict conditions to filter high-quality pseudo labels from network predictions, we observe that objects with extreme scale tend to have low confidence, resulting in a lack of positive supervision for these objects. In this paper, we propose a novel framework that addresses the scale variation problem by introducing a mixed scale teacher to improve pseudo label generation and scale-invariant learning. Additionally, we propose mining pseudo labels using score promotion of predictions across scales, which benefits from better predictions from mixed scale features. Our extensive experiments on MS COCO and PASCAL VOC benchmarks under various semi-supervised settings demonstrate that our method achieves new state-of-the-art performance. The code and models are available at https://github.com/lliuz/MixTeacher.

----

## [703] Exploring Incompatible Knowledge Transfer in Few-shot Image Generation

**Authors**: *Yunqing Zhao, Chao Du, Milad Abdollahzadeh, Tianyu Pang, Min Lin, Shuicheng Yan, Ngai-Man Cheung*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00713](https://doi.org/10.1109/CVPR52729.2023.00713)

**Abstract**:

Few-shot image generation (FSIG) learns to generate diverse and high-fidelity images from a target domain using a few (e.g., 10) reference samples. Existing FSIG methods select, preserve and transfer prior knowledge from a source generator (pretrained on a related domain) to learn the target generator. In this work, we investigate an underexplored issue in FSIG, dubbed as incompatible knowledge transfer, which would significantly degrade the realisticness of synthetic samples. Empirical observations show that the issue stems from the least significant filters from the source generator. To this end, we propose knowledge truncation to mitigate this issue in FSIG, which is a complementary operation to knowledge preservation and is implemented by a lightweight pruning-based method. Extensive experiments show that knowledge truncation is simple and effective, consistently achieving state-of-the-art performance, including challenging setups where the source and target domains are more distant. Project Page: yunqing-me.github.io/RICK.

----

## [704] Exploring Intra-class Variation Factors with Learnable Cluster Prompts for Semi-supervised Image Synthesis

**Authors**: *Yunfei Zhang, Xiaoyang Huo, Tianyi Chen, Si Wu, Hau-San Wong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00714](https://doi.org/10.1109/CVPR52729.2023.00714)

**Abstract**:

Semi-supervised class-conditional image synthesis is typically performed by inferring and injecting class labels into a conditional Generative Adversarial Network (GAN). The supervision in the form of class identity may be inadequate to model classes with diverse visual appearances. In this paper, we propose a Learnable Cluster Prompt-based GAN (LCP-GAN) to capture class-wise characteristics and intra-class variation factors with a broader source of supervision. To exploit partially labeled data, we perform soft partitioning on each class, and explore the possibility of associating intra-class clusters with learnable visual concepts in the feature space of a pre-trained language-vision model, e.g., CLIP. For class-conditional image generation, we design a cluster-conditional generator by injecting a combination of intra-class cluster label embeddings, and further incorporate a real-fake classification head on top of CLIP to distinguish real instances from the synthesized ones, conditioned on the learnable cluster prompts. This significantly strengthens the generator with more semantic language supervision. LCP-GAN not only possesses superior generation capability but also matches the performance of the fully supervised version of the base models: BigGAN and StyleGAN2-ADA, on multiple standard benchmarks.

----

## [705] A Soma Segmentation Benchmark in Full Adult Fly Brain

**Authors**: *Xiaoyu Liu, Bo Hu, Mingxing Li, Wei Huang, Yueyi Zhang, Zhiwei Xiong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00715](https://doi.org/10.1109/CVPR52729.2023.00715)

**Abstract**:

Neuron reconstruction in a full adult fly brain from high-resolution electron microscopy (EM) data is regarded as a cornerstone for neuroscientists to explore how neurons inspire intelligence. As the central part of neurons, somas in the full brain indicate the origin of neurogenesis and neural functions. However, due to the absence of EM datasets specifically annotated for somas, existing deep learning-based neuron reconstruction methods cannot directly provide accurate soma distribution and morphology. Moreover, full brain neuron reconstruction remains extremely time-consuming due to the unprecedentedly large size of EM data. In this paper, we develop an efficient soma reconstruction method for obtaining accurate soma distribution and morphology information in a full adult fly brain. To this end, we first make a high-resolution EM dataset with fine-grained 3D manual annotations on somas. Relying on this dataset, we propose an efficient, two-stage deep learning algorithm for predicting accurate locations and boundaries of 3D soma instances. Further, we deploy a parallelized, high-throughput data processing pipeline for executing the above algorithm on the full brain. Finally, we provide quantitative and qualitative benchmark comparisons on the testset to validate the superiority of the proposed method, as well as preliminary statistics of the reconstructed somas in the full adult fly brain from the biological perspective. We release our code and dataset at https://github.com/liuxy1103/EMADS.

----

## [706] SDC-UDA: Volumetric Unsupervised Domain Adaptation Framework for Slice-Direction Continuous Cross-Modality Medical Image Segmentation

**Authors**: *Hyungseob Shin, Hyeongyu Kim, Sewon Kim, Yohan Jun, Taejoon Eo, Dosik Hwang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00716](https://doi.org/10.1109/CVPR52729.2023.00716)

**Abstract**:

Recent advances in deep learning-based medical image segmentation studies achieve nearly human-level performance in fully supervised manner. However, acquiring pixel-level expert annotations is extremely expensive and laborious in medical imaging fields. Unsupervised domain adaptation (UDA) can alleviate this problem, which makes it possible to use annotated data in one imaging modality to train a network that can successfully perform segmentation on target imaging modality with no labels. In this work, we propose SDC-UDA, a simple yet effective volumetric UDA framework for Slice-Direction Continuous cross-modality medical image segmentation which combines intra- and inter-slice self-attentive image translation, uncertainty-constrained pseudo-label refinement, and volumetric self-training. Our method is distinguished from previous methods on UDA for medical image segmentation in that it can obtain continuous segmentation in the slice direction, thereby ensuring higher accuracy and potential in clinical practice. We validate SDC-UDA with multiple publicly available cross-modality medical image segmentation datasets and achieve state-of-the-art segmentation performance, not to mention the superior slice-direction continuity of prediction compared to previous studies.

----

## [707] Label-Free Liver Tumor Segmentation

**Authors**: *Qixin Hu, Yixiong Chen, Junfei Xiao, Shuwen Sun, Jieneng Chen, Alan L. Yuille, Zongwei Zhou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00717](https://doi.org/10.1109/CVPR52729.2023.00717)

**Abstract**:

We demonstrate that AI models can accurately segment liver tumors without the need for manual annotation by using synthetic tumors in CT scans. Our synthetic tumors have two intriguing advantages: (I) realistic in shape and texture, which even medical professionals can confuse with real tumors; (II) effective for training AI models, which can perform liver tumor segmentation similarly to the model trained on real tumors—this result is exciting because no existing work, using synthetic tumors only, has thus far reached a similar or even close performance to real tumors. This result also implies that manual efforts for annotating tumors voxel by voxel (which took years to create) can be significantly reduced in the future. Moreover, our synthetic tumors can automatically generate many examples of small (or even tiny) synthetic tumors and have the potential to im-prove the success rate of detecting small liver tumors, which is critical for detecting the early stages of cancer. In addition to enriching the training data, our synthesizing strategy also enables us to rigorously assess the AI robustness.

----

## [708] Interactive and Explainable Region-guided Radiology Report Generation

**Authors**: *Tim Tanida, Philip Müller, Georgios Kaissis, Daniel Rueckert*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00718](https://doi.org/10.1109/CVPR52729.2023.00718)

**Abstract**:

The automatic generation of radiology reports has the potential to assist radiologists in the time-consuming task of report writing. Existing methods generate the full report from image-level features, failing to explicitly focus on anatomical regions in the image. We propose a simple yet effective region-guided report generation model that detects anatomical regions and then describes individual, salient regions to form the final report. While previous methods generate reports without the possibility of human intervention and with limited explainability, our method opens up novel clinical use cases through additional interactive capabilities and introduces a high degree of transparency and explainability. Comprehensive experiments demonstrate our method's effectiveness in report generation, outperforming previous state-of-the-art models, and highlight its interactive capabilities. The code and checkpoints are available at https://github.com/ttanida/rgrg.

----

## [709] A Loopback Network for Explainable Microvascular Invasion Classification

**Authors**: *Shengxuming Zhang, Tianqi Shi, Yang Jiang, Xiuming Zhang, Jie Lei, Zunlei Feng, Mingli Song*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00719](https://doi.org/10.1109/CVPR52729.2023.00719)

**Abstract**:

Microvascular invasion (MVI) is a critical factor for prognosis evaluation and cancer treatment. The current diagnosis of MVI relies on pathologists to manually find out cancerous cells from hundreds of blood vessels, which is time-consuming, tedious, and subjective. Recently, deep learning has achieved promising results in medical image analysis tasks. However, the unexplainability of black box models and the requirement of massive annotated samples limit the clinical application of deep learning based diagnostic methods. In this paper, aiming to develop an accurate, objective, and explainable diagnosis tool for MVI, we propose a Loopback Network (LoopNet) for classifying MVI efficiently. With the image-level category annotations of the collected Pathologic Vessel Image Dataset (PVID), LoopNet is devised to be composed binary classification branch and cell locating branch. The latter is devised to locate the area of cancerous cells, regular non-cancerous cells, and background. For healthy samples, the pseudo masks of cells supervise the cell locating branch to distinguish the area of regular non-cancerous cells and background. For each MVI sample, the cell locating branch predicts the mask of cancerous cells. Then the masked cancerous and non-cancerous areas of the same sample are input back to the binary classification branch separately. The loopback between two branches enables the category label to supervise the cell locating branch to learn the locating ability for cancerous areas. Experiment results show that the proposed LoopNet achieves 97.5% accuracy on MVI classification. Surprisingly, the proposed loopback mechanism not only enables LoopNet to predict the cancerous area but also facilitates the classification backbone to achieve better classification performance.

----

## [710] Task-Specific Fine-Tuning via Variational Information Bottleneck for Weakly-Supervised Pathology Whole Slide Image Classification

**Authors**: *Honglin Li, Chenglu Zhu, Yunlong Zhang, Yuxuan Sun, Zhongyi Shui, Wenwei Kuang, Sunyi Zheng, Lin Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00720](https://doi.org/10.1109/CVPR52729.2023.00720)

**Abstract**:

While Multiple Instance Learning (MIL) has shown promising results in digital Pathology Whole Slide Image (WSI) analysis, such a paradigm still faces performance and generalization problems due to high computational costs and limited supervision of Gigapixel WSIs. To deal with the computation problem, previous methods utilize a frozen model pretrained from ImageNet to obtain representations, however, it may lose key information owing to the large domain gap and hinder the generalization ability without image-level training-time augmentation. Though Self-supervised Learning (SSL) proposes viable representation learning schemes, the downstream task-specific features via partial label tuning are not explored. To alleviate this problem, we propose an efficient WSI fine-tuning framework motivated by the Information Bottleneck theory. The theory enables the framework to find the minimal sufficient statistics of WSI, thus supporting us to fine-tune the backbone into a task-specific representation only depending on WSI-level weak labels. The WSI-MIL problem is further analyzed to theoretically deduce our fine-tuning method. We evaluate the method on five pathological WSI datasets on various WSI heads. The experimental results show significant improvements in both accuracy and generalization compared with previous works. Source code will be available at https://github.com/invoker-LL/WSI-finetuning.

----

## [711] YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors

**Authors**: *Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00721](https://doi.org/10.1109/CVPR52729.2023.00721)

**Abstract**:

Real-time object detection is one of the most important research topics in computer vision. As new approaches regarding architecture optimization and training optimization are continually being developed, we have found two research topics that have spawned when dealing with these latest state-of-the-art methods. To address the topics, we propose a trainable bag-of-freebies oriented solution. We combine the flexible and efficient training tools with the proposed architecture and the compound scaling method. YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 120 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100. Source code is released in https://github.com/WongKinYiu/yolov7.

----

## [712] Two-Way Multi-Label Loss

**Authors**: *Takumi Kobayashi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00722](https://doi.org/10.1109/CVPR52729.2023.00722)

**Abstract**:

A natural image frequently contains multiple classification targets, accordingly providing multiple class labels rather than a single label per image. While the single-label classification is effectively addressed by applying a softmax cross-entropy loss, the multi-label task is tackled mainly in a binary cross-entropy (BCE) framework. In contrast to the softmax loss, the BCE loss involves issues regarding imbalance as multiple classes are decomposed into a bunch of binary classifications; recent works improve the BCE loss to cope with the issue by means of weighting. In this paper, we propose a multi-label loss by bridging a gap between the softmax loss and the multi-label scenario. The proposed loss function is formulated on the basis of relative comparison among classes which also enables us to further improve discriminative power of features by enhancing classification margin. The loss function is so flexible as to be applicable to a multi-label setting in two ways for discriminating classes as well as samples. In the experiments on multi-label classification, the proposed method exhibits competitive performance to the other multi-label losses, and it also provides transferrable features on single-label ImageNet training. Codes are available at https://github.com/tk1980/TwowayMultiLabelLoss.

----

## [713] Teaching Matters: Investigating the Role of Supervision in Vision Transformers

**Authors**: *Matthew Walmer, Saksham Suri, Kamal Gupta, Abhinav Shrivastava*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00723](https://doi.org/10.1109/CVPR52729.2023.00723)

**Abstract**:

Vision Transformers (ViTs) have gained significant popularity in recent years and have proliferated into many applications. However, their behavior under different learning paradigms is not well explored. We compare ViTs trained through different methods of supervision, and show that they learn a diverse range of behaviors in terms of their attention, representations, and downstream performance. We also discover ViT behaviors that are consistent across supervision, including the emergence of Offset Local Attention Heads. These are self-attention heads that attend to a token adjacent to the current token with a fixed directional offset, a phenomenon that to the best of our knowledge has not been highlighted in any prior work. Our analysis shows that ViTs are highly flexible and learn to process local and global information in different orders depending on their training method. We find that contrastive self-supervised methods learn features that are competitive with explicitly supervised features, and they can even be superior for part-level tasks. We also find that the representations of reconstruction-based models show non-trivial similarity to contrastive self-supervised models.

----

## [714] Label Information Bottleneck for Label Enhancement

**Authors**: *Qinghai Zheng, Jihua Zhu, Haoyu Tang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00724](https://doi.org/10.1109/CVPR52729.2023.00724)

**Abstract**:

In this work, we focus on the challenging problem of Label Enhancement (LE), which aims to exactly recover label distributions from logical labels, and present a novel Label Information Bottleneck (LIB) method for LE. For the recovery process of label distributions, the label irrelevant information contained in the dataset may lead to unsatisfactory recovery performance. To address this limitation, we make efforts to excavate the essential label relevant information to improve the recovery performance. Our method formulates the LE problem as the following two joint processes: 1) learning the representation with the essential label relevant information, 2) recovering label distributions based on the learned representation. The label relevant information can be excavated based on the “bottleneck” formed by the learned representation. Significantly, both the label relevant information about the label assignments and the label relevant information about the label gaps can be explored in our method. Evaluation experiments conducted on several benchmark label distribution learning datasets verify the effectiveness and competitiveness of LIB. Our source codes are available at https://github.com/qinghai-zheng/LIBLE.

----

## [715] Glocal Energy-based Learning for Few-Shot Open-Set Recognition

**Authors**: *Haoyu Wang, Guansong Pang, Peng Wang, Lei Zhang, Wei Wei, Yanning Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00725](https://doi.org/10.1109/CVPR52729.2023.00725)

**Abstract**:

Few-shot open-set recognition (FSOR) is a challenging task of great practical value. It aims to categorize a sample to one of the predefined, closed-set classes illustrated by few examples while being able to reject the sample from unknown classes. In this work, we approach the FSOR task by proposing a novel energy-based hybrid model. The model is composed of two branches, where a classification branch learns a metric to classify a sample to one of closed-set classes and the energy branch explicitly estimates the open-set probability. To achieve holistic detection of open-set samples, our model leverages both class-wise and pixel-wise features to learn a glocal energy-based score, in which a global energy score is learned using the class-wise features, while a local energy score is learned using the pixel-wise features. The model is enforced to assign large energy scores to samples that are deviated from the few-shot examples in either the class-wise features or the pixel-wise features, and to assign small energy scores otherwise. Experiments on three standard FSOR datasets show the superior performance of our model.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/00why00/Glocal

----

## [716] Noisy Correspondence Learning with Meta Similarity Correction

**Authors**: *Haochen Han, Kaiyao Miao, Qinghua Zheng, Minnan Luo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00726](https://doi.org/10.1109/CVPR52729.2023.00726)

**Abstract**:

Despite the success of multimodal learning in crossmodal retrieval task, the remarkable progress relies on the correct correspondence among multimedia data. However, collecting such ideal data is expensive and time-consuming. In practice, most widely used datasets are harvested from the Internet and inevitably contain mismatched pairs. Training on such noisy correspondence datasets causes performance degradation because the cross-modal retrieval methods can wrongly enforce the mismatched data to be similar. To tackle this problem, we propose a Meta Similarity Correction Network (MSCN) to provide reliable similarity scores. We view a binary classification task as the meta-process that encourages the MSCN to learn discrimination from positive and negative meta-data. To further alleviate the influence of noise, we design an effective data purification strategy using meta-data as prior knowledge to remove the noisy samples. Extensive experiments are conducted to demonstrate the strengths of our method in both synthetic and real-world noises, including Flickr30K, MS-COCO, and Conceptual Captions. Our code is publicly available.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/hhc1997/MSCN

----

## [717] Hubs and Hyperspheres: Reducing Hubness and Improving Transductive Few-Shot Learning with Hyperspherical Embeddings

**Authors**: *Daniel J. Trosten, Rwiddhi Chakraborty, Sigurd Løkse, Kristoffer Knutsen Wickstrøm, Robert Jenssen, Michael C. Kampffmeyer*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00727](https://doi.org/10.1109/CVPR52729.2023.00727)

**Abstract**:

Distance-based classification is frequently used in transductive few-shot learning (FSL). However, due to the high-dimensionality of image representations, FSL classifiers are prone to suffer from the hubness problem, where a few points (hubs) occur frequently in multiple nearest neighbour lists of other points. Hubness negatively impacts distance-based classification when hubs from one class appear often among the nearest neighbors of points from another class, degrading the classifier's performance. To address the hubness problem in FSL, we first prove that hubness can be eliminated by distributing representations uniformly on the hypersphere. We then propose two new approaches to embed representations on the hypersphere, which we prove optimize a tradeoff between uniformity and local similarity preservation - reducing hubness while retaining class structure. Our experiments show that the proposed methods reduce hubness, and significantly improves transductive FSL accuracy for a wide range of classifiers 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/uitml/noHub..

----

## [718] Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning

**Authors**: *Sungnyun Kim, Sangmin Bae, Se-Young Yun*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00728](https://doi.org/10.1109/CVPR52729.2023.00728)

**Abstract**:

Deep learning in general domains has constantly been extended to domain-specific tasks requiring the recognition of fine-grained characteristics. However, real-world applications for fine-grained tasks suffer from two challenges: a high reliance on expert knowledge for annotation and necessity of a versatile model for various downstream tasks in a specific domain (e.g., prediction of categories, bounding boxes, or pixel-wise annotations). Fortunately, the recent self-supervised learning (SSL) is a promising approach to pretrain a model without annotations, serving as an effective initialization for any downstream tasks. Since SSL does not rely on the presence of annotation, in general, it utilizes the large-scale unlabeled dataset, referred to as an open-set. In this sense, we introduce a novel Open-Set Self-Supervised Learning problem under the assumption that a large-scale unlabeled open-set is available, as well as the fine-grained target dataset, during a pretraining phase. In our problem setup, it is crucial to consider the distribution mismatch between the open-set and target dataset. Hence, we propose SimCore algorithm to sample a coreset, the subset of an open-set that has a minimum distance to the target dataset in the latent space. We demonstrate that SimCore significantly improves representation learning performance through extensive experimental settings, including eleven fine-grained datasets and seven open-sets in various downstream tasks.

----

## [719] Boosting Semi-Supervised Learning by Exploiting All Unlabeled Data

**Authors**: *Yuhao Chen, Xin Tan, Borui Zhao, Zhaowei Chen, Renjie Song, Jiajun Liang, Xuequan Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00729](https://doi.org/10.1109/CVPR52729.2023.00729)

**Abstract**:

Semi-supervised learning (SSL) has attracted enormous attention due to its vast potential of mitigating the dependence on large labeled datasets. The latest methods (e.g., FixMatch) use a combination of consistency regularization and pseudo-labeling to achieve remarkable successes. However, these methods all suffer from the waste of complicated examples since all pseudo-labels have to be selected by a high threshold to filter out noisy ones. Hence, the examples with ambiguous predictions will not contribute to the training phase. For better leveraging all unlabeled examples, we propose two novel techniques: Entropy Meaning Loss (EML) and Adaptive Negative Learning (ANL). EML incorporates the prediction distribution of non-target classes into the optimization objective to avoid competition with target class, and thus generating more high-confidence predictions for selecting pseudo-label. ANL introduces the additional negative pseudo-label for all unlabeled data to leverage low-confidence examples. It adaptively allocates this label by dynamically evaluating the top-k performance of the model. EML and ANL do not introduce any additional parameter and hyperparameter. We integrate these techniques with FixMatch, and develop a simple yet powerful framework called FullMatch. Extensive experiments on several common SSL benchmarks (CIFAR-10/100, SVHN, STL-10 and ImageNet) demonstrate that FullMatch exceeds FixMatch by a large margin. Integrated with FlexMatch (an advanced FixMatch-based framework), we achieve state-of-the-art performance. Source code is available at https://github.com/megvii-research/FullMatch.

----

## [720] Trade-off between Robustness and Accuracy of Vision Transformers

**Authors**: *Yanxi Li, Chang Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00730](https://doi.org/10.1109/CVPR52729.2023.00730)

**Abstract**:

Although deep neural networks (DNNs) have shown great successes in computer vision tasks, they are vulnerable to perturbations on inputs, and there exists a trade-off between the natural accuracy and robustness to such perturbations, which is mainly caused by the existence of robust non-predictive features and non-robust predictive features. Recent empirical analyses find Vision Transformers (ViTs) are inherently robust to various kinds of perturbations, but the aforementioned trade-off still exists for them. In this work, we propose Trade-off between Robustness and Accuracy of Vision Transformers (TORA-ViTs), which aims to efficiently transfer ViT models pretrained on natural tasks for both accuracy and robustness. TORA-ViTs consist of two major components, including a pair of accuracy and robustness adapters to extract predictive and robust features, respectively, and a gated fusion module to adjust the trade-off. The gated fusion module takes outputs of a pre-trained ViT block as queries and outputs of our adapters as keys and values, and tokens from different adapters at different spatial locations are compared with each other to generate attention scores for a balanced mixing of predictive and robust features. Experiments on ImageNet with various robust benchmarks show that our TORA-ViTs can efficiently improve the robustness of naturally pretrained ViTs while maintaining competitive natural accuracy. Our most balanced setting (TORA-ViTs with λ= 0.5) can maintain 83.7% accuracy on clean ImageNet and reach 54.7% and 38.0% accuracy under FGSM and PGD white-box attacks, respectively. In terms of various ImageNet variants, it can reach 39.2% and 56.3% accuracy on ImageNet-A and ImageNet-R and reach 34.4% mCE on ImageNet-C.

----

## [721] Exploring and Utilizing Pattern Imbalance

**Authors**: *Shibin Mei, Chenglong Zhao, Shengchao Yuan, Bingbing Ni*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00731](https://doi.org/10.1109/CVPR52729.2023.00731)

**Abstract**:

In this paper, we identify pattern imbalance from several aspects, and further develop a new training scheme to avert pattern preference as well as spurious correlation. In contrast to prior methods which are mostly concerned with category or domain granularity, ignoring the potential finer structure that existed in datasets, we give a new definition of seed category as an appropriate optimization unit to distinguish different patterns in the same category or domain. Extensive experiments on domain generalization datasets of diverse scales demonstrate the effectiveness of the proposed method.

----

## [722] Dynamic Conceptional Contrastive Learning for Generalized Category Discovery

**Authors**: *Nan Pu, Zhun Zhong, Nicu Sebe*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00732](https://doi.org/10.1109/CVPR52729.2023.00732)

**Abstract**:

Generalized category discovery (GCD) is a recently proposed open-world problem, which aims to automatically cluster partially labeled data. The main challenge is that the unlabeled data contain instances that are not only from known categories of the labeled data but also from novel categories. This leads traditional novel category discovery (NCD) methods to be incapacitated for GCD, due to their assumption of unlabeled data are only from novel categories. One effective way for GCD is applying self-supervised learning to learn discriminate representation for unlabeled data. However, this manner largely ignores underlying relationships between instances of the same concepts (e.g., class, super-class, and sub-class), which results in inferior representation learning. In this paper, we propose a Dynamic Conceptional Contrastive Learning (DCCL)framework, which can effectively improve clustering accuracy by alternately estimating underlying visual conceptions and learning conceptional representation. In addition, we design a dynamic conception generation and update mechanism, which is able to ensure consistent conception learning and thus further facilitate the optimization of DCCL. Extensive experiments show that DCCL achieves new state-of-the-art performances on six generic and fine-grained visual recognition datasets, especially on fine-grained ones. For example, our method significantly surpasses the best competitor by 16.2% on the new classes for the CUB-200 dataset. Code is available at https://github.com/TPCD/DCCL

----

## [723] Towards Better Decision Forests: Forest Alternating Optimization

**Authors**: *Miguel Á. Carreira-Perpiñán, Magzhan Gabidolla, Arman Zharmagambetov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00733](https://doi.org/10.1109/CVPR52729.2023.00733)

**Abstract**:

Decision forests are among the most accurate models in machine learning. This is remarkable given that the way they are trained is highly heuristic: neither the individual trees nor the overall forest optimize any well-defined loss. While diversity mechanisms such as bagging or boosting have been until now critical in the success of forests, we think that a better optimization should lead to better forests—ideally eliminating any need for an ensembling heuristic. However, unlike for most other models, such as neural networks, optimizing forests or trees is not easy, because they define a non-differentiable function. We show, for the first time, that it is possible to learn a forest by optimizing a desirable loss and regularization jointly over all its trees and parameters. Our algorithm, Forest Alternating Optimization, is based on defining a forest as a parametric model with a fixed number of trees and structure (rather than adding trees indefinitely as in bagging or boosting). It then iteratively updates each tree in alternation so that the objective function decreases monotonically. The algorithm is so effective at optimizing that it easily overfits, but this can be corrected by averaging. The result is a forest that consistently exceeds the accuracy of the state-of-the-art while using fewer, smaller trees.

----

## [724] Learning Debiased Representations via Conditional Attribute Interpolation

**Authors**: *Yi-Kai Zhang, Qi-Wei Wang, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00734](https://doi.org/10.1109/CVPR52729.2023.00734)

**Abstract**:

An image is usually described by more than one attribute like “shape” and “color”. When a dataset is biased, i.e., most samples have attributes spuriously correlated with the target label, a Deep Neural Network (DNN) is prone to make predictions by the “unintended” attribute, especially if it is easier to learn. To improve the generalization ability when training on such a biased dataset, we propose a X
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-model to learn debiased representations. First, we design a x-shape pattern to match the training dynamics of a DNN and find Intermediate Attribute Samples (IASs) — samples near the attribute decision boundaries, which indicate how the value of an attribute changes from one extreme to another. Then we rectify the representation with a X-structured metric learning objective. Conditional interpolation among IASs eliminates the negative effect of periph-eral attributes and facilitates retaining the intra-class compactness. Experiments show that X
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-modellearns debiased representation effectively and achieves remarkable improvements on various datasets. Code is available at: https://github.com/ZhangYikaii/chi-square

----

## [725] On the Pitfall of Mixup for Uncertainty Calibration

**Authors**: *Deng-Bao Wang, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, Min-Ling Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00735](https://doi.org/10.1109/CVPR52729.2023.00735)

**Abstract**:

By simply taking convex combinations between pairs of samples and their labels, mixup training has been shown to easily improve predictive accuracy. It has been recently found that models trained with mixup also perform well on uncertainty calibration. However, in this study, we found that mixup training usually makes models less calibratable than vanilla empirical risk minimization, which means that it would harm uncertainty estimation when post-hoc calibration is considered. By decomposing the mixup process into data transformation and random perturbation, we suggest that the confidence penalty nature of the data transformation is the reason of calibration degradation. To mitigate this problem, we first investigate the mixup inference strategy and found that despite it improves calibration on mixup, this ensemble-like strategy does not necessarily out-perform simple ensemble. Then, we propose a general strategy named mixup inference in training, which adopts a simple decoupling principle for recovering the outputs of raw samples at the end of forward network pass. By embedding the mixup inference, models can be learned from the original one-hot labels and hence avoid the negative impact of confidence penalty. Our experiments show this strategy properly solves mixup's calibration issue without sacrificing the predictive performance, while even improves accuracy than vanilla mixup.

----

## [726] Class Relationship Embedded Learning for Source-Free Unsupervised Domain Adaptation

**Authors**: *Yixin Zhang, Zilei Wang, Weinan He*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00736](https://doi.org/10.1109/CVPR52729.2023.00736)

**Abstract**:

This work focuses on a practical knowledge transfer task defined as Source-Free Unsupervised Domain Adaptation (SFUDA), where only a well-trained source model and unlabeled target data are available. To fully utilize source knowledge, we propose to transfer the class relationship, which is domain-invariant but still under-explored in previous works. To this end, we first regard the classifier weights of the source model as class prototypes to compute class relationship, and then propose a novel probability-based similarity between target-domain samples by embedding the source-domain class relationship, resulting in Class Relationship embedded Similarity (CRS). Here the inter-class term is particularly considered in order to more accurately represent the similarity between two samples, in which the source prior of class relationship is utilized by weighting. Finally, we propose to embed CRS into contrastive learning in a unified form. Here both class-aware and instance discrimination contrastive losses are employed, which are complementary to each other. We combine the proposed method with existing representative methods to evaluate its efficacy in multiple SFUDA settings. Extensive experimental results reveal that our method can achieve state-of-the-art performance due to the transfer of domain-invariant class relationship. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/zhyx12/CRCo

----

## [727] FeatureBooster: Boosting Feature Descriptors with a Lightweight Neural Network

**Authors**: *Xinjiang Wang, Zeyu Liu, Yu Hu, Wei Xi, Wenxian Yu, Danping Zou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00737](https://doi.org/10.1109/CVPR52729.2023.00737)

**Abstract**:

We introduce a lightweight network to improve descriptors of keypoints within the same image. The network takes the original descriptors and the geometric properties of keypoints as the input, and uses an MLP-based self-boosting stage and a Transformer-based cross-boosting stage to enhance the descriptors. The boosted descriptors can be either real-valued or binary ones. We use the proposed network to boost both hand-crafted (ORB [34], SIFT [24]) and the state-of-the-art learning-based descriptors (SuperPoint [10], ALIKE [53]) and evaluate them on image matching, visual localization, and structure-from-motion tasks. The results show that our method significantly improves the performance of each task, particularly in challenging cases such as large illumination changes or repetitive patterns. Our method requires only 3.2ms on desktop GPU and 27ms on embedded GPU to process 2000 features, which is fast enough to be applied to a practical system. The code and trained weights are publicly available at github.com/SJTU-ViSYS/FeatureBooster.

----

## [728] Guiding Pseudo-labels with Uncertainty Estimation for Source-free Unsupervised Domain Adaptation

**Authors**: *Mattia Litrico, Alessio Del Bue, Pietro Morerio*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00738](https://doi.org/10.1109/CVPR52729.2023.00738)

**Abstract**:

Standard Unsupervised Domain Adaptation (UDA) methods assume the availability of both source and target data during the adaptation. In this work, we investigate Source-free Unsupervised Domain Adaptation (SF-UDA), a specific case of UDA where a model is adapted to a target domain without access to source data. We propose a novel approach for the SF-UDA setting based on a loss reweighting strategy that brings robustness against the noise that inevitably affects the pseudo-labels. The classification loss is reweighted based on the reliability of the pseudo-labels that is measured by estimating their uncertainty. Guided by such reweighting strategy, the pseudo-labels are progressively refined by aggregating knowledge from neighbouring samples. Furthermore, a self-supervised contrastive framework is leveraged as a target space regulariser to enhance such knowledge aggregation. A novel negative pairs exclusion strategy is proposed to identify and exclude negative pairs made of samples sharing the same class, even in presence of some noise in the pseudo-labels. Our method outperforms previous methods on three major benchmarks by a large margin. We set the new SF-UDA state-of-the-art on VisDA-C and DomainNet with a performance gain of + 1.8% on both benchmarks and on PACS with + 12.3% in the single-source setting and +6.6% in multi-target adaptation. Additional analyses demonstrate that the proposed approach is robust to the noise, which results in significantly more accurate pseudo-labels compared to state-of-the-art approaches.

----

## [729] Divide and Adapt: Active Domain Adaptation via Customized Learning

**Authors**: *Duojun Huang, Jichang Li, Weikai Chen, Junshi Huang, Zhenhua Chai, Guanbin Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00739](https://doi.org/10.1109/CVPR52729.2023.00739)

**Abstract**:

Active domain adaptation (ADA) aims to improve the model adaptation performance by incorporating active learning (AL) techniques to label a maximally-informative subset of target samples. Conventional AL methods do not consider the existence of domain shift, and hence, fail to identify the truly valuable samples in the context of domain adaptation. To accommodate active learning and domain adaption, the two naturally different tasks, in a collaborative framework, we advocate that a customized learning strategy for the target data is the key to the success of ADA solutions. We present Divide-and-Adapt (DiaNA), a new ADA framework that partitions the target instances into four categories with stratified transferable properties. With a novel data subdivision protocol based on uncertainty and domainness, DiaNA can accurately recognize the most gainful samples. While sending the informative instances for annotation, DiaNA employs tailored learning strategies for the remaining categories. Furthermore, we propose an informativeness score that unifies the data partitioning criteria. This enables the use of a Gaussian mixture model (GMM) to automatically sample unlabeled data into the proposed four categories. Thanks to the “divide-and-adapt” spirit, DiaNA can handle data with large variations of domain gap. In addition, we show that DiaNA can generalize to different domain adaptation settings, such as unsupervised domain adaptation (UDA), semi-supervised domain adaptation (SSDA), source-free domain adaptation (SFDA), etc.

----

## [730] Understanding and Constructing Latent Modality Structures in Multi-Modal Representation Learning

**Authors**: *Qian Jiang, Changyou Chen, Han Zhao, Liqun Chen, Qing Ping, Son Dinh Tran, Yi Xu, Belinda Zeng, Trishul Chilimbi*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00740](https://doi.org/10.1109/CVPR52729.2023.00740)

**Abstract**:

Contrastive loss has been increasingly used in learning representations from multiple modalities. In the limit, the nature of the contrastive loss encourages modalities to exactly match each other in the latent space. Yet it remains an open question how the modality alignment affects the downstream task performance. In this paper, based on an information-theoretic argument, we first prove that exact modality alignment is sub-optimal in general for down-stream prediction tasks. Hence we advocate that the key of better performance lies in meaningful latent modality structures instead of perfect modality alignment. To this end, we propose three general approaches to construct latent modality structures. Specifically, we design 1) a deep feature separation loss for intra-modality regularization; 2) a Brownian-bridge loss for inter-modality regularization; and 3) a geometric consistency loss for both intra- and intermodality regularization. Extensive experiments are conducted on two popular multi-modal representation learning frameworks: the CLIP-based two-tower model and the ALBEF-based fusion model. We test our model on a variety of tasks including zero/few-shot image classification, image-text retrieval, visual question answering, visual reasoning, and visual entailment. Our method achieves consistent improvements over existing methods, demonstrating the effectiveness and generalizability of our proposed approach on latent modality structure regularization.

----

## [731] Deep Factorized Metric Learning

**Authors**: *Chengkun Wang, Wenzhao Zheng, Junlong Li, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00741](https://doi.org/10.1109/CVPR52729.2023.00741)

**Abstract**:

Learning a generalizable and comprehensive similarity metric to depict the semantic discrepancies between images is the foundation of many computer vision tasks. While existing methods approach this goal by learning an ensemble of embeddings with diverse objectives, the backbone network still receives a mix of all the training signals. Differently, we propose a deep factorized metric learning (DFML) method to factorize the training signal and employ different samples to train various components of the backbone network. We factorize the network to different sub-blocks and devise a learnable router to adaptively allocate the training samples to each sub-block with the objective to capture the most information. The metric model trained by DFML capture different characteristics with different sub-blocks and constitutes a generalizable metric when using all the sub-blocks. The proposed DFML achieves state-of-the-art performance on all three benchmarks for deep metric learning including CUB-200-20ll, Cars196, and Stanford Online Products. We also generalize DFML to the image classification task on ImageNet-1K and observe consistent improvement in accuracy/computation trade-off. Specifically, we improve the performance of ViT-B on ImageNet (+0.2% accuracy) with less computation load (-24% FLOPs).
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.com/wangck20/DFML.

----

## [732] Meta-Causal Learning for Single Domain Generalization

**Authors**: *Jin Chen, Zhi Gao, Xinxiao Wu, Jiebo Luo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00742](https://doi.org/10.1109/CVPR52729.2023.00742)

**Abstract**:

Single domain generalization aims to learn a model from a single training domain (source domain) and apply it to multiple unseen test domains (target domains). Existing methods focus on expanding the distribution of the training domain to cover the target domains, but without estimating the domain shift between the source and target domains. In this paper, we propose a new learning paradigm, namely simulate-analyze-reduce, which first simulates the domain shift by building an auxiliary domain as the target domain, then learns to analyze the causes of domain shift, and finally learns to reduce the domain shift for model adaptation. Under this paradigm, we propose a meta-causal learning method to learn meta-knowledge, that is, how to infer the causes of domain shift between the auxiliary and source domains during training. We use the meta-knowledge to analyze the shift between the target and source domains during testing. Specifically, we perform multiple transformations on source data to generate the auxiliary domain, perform counterfactual inference to learn to discover the causal factors of the shift between the auxiliary and source domains, and incorporate the inferred causality into factor-aware domain alignments. Extensive experiments on several benchmarks of image classification show the effectiveness of our method.

----

## [733] Meta Omnium: A Benchmark for General-Purpose Learning-to-Learn

**Authors**: *Ondrej Bohdal, Yinbing Tian, Yongshuo Zong, Ruchika Chavhan, Da Li, Henry Gouk, Li Guo, Timothy M. Hospedales*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00743](https://doi.org/10.1109/CVPR52729.2023.00743)

**Abstract**:

Meta-learning and other approaches to few-shot learning are widely studied for image recognition, and are increasingly applied to other vision tasks such as pose estimation and dense prediction. This naturally raises the question of whether there is any fewshot metalearning algorithm capable of generalizing across these diverse task types? To support the community in answering this question, we introduce Meta Omnium, a dataset-of-datasets spanning multiple vision tasks including recognition, keypoint localization, semantic segmentation and regression. We experiment with popular fewshot metalearning baselines and analyze their ability to generalize across tasks and to transfer knowledge between them. Meta Omnium enables metalearning researchers to evaluate model generalization to a much wider array of tasks than previously possible, and provides a single framework for evaluating meta-learners across a wide suite of vision applications in a consistent manner. Code and dataset are available at https://github.com/edi-meta-learning/meta-omnium.

----

## [734] Robust Mean Teacher for Continual and Gradual Test-Time Adaptation

**Authors**: *Mario Döbler, Robert A. Marsden, Bin Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00744](https://doi.org/10.1109/CVPR52729.2023.00744)

**Abstract**:

Since experiencing domain shifts during test-time is inevitable in practice, test-time adaption (TTA) continues to adapt the model after deployment. Recently, the area of continual and gradual test-time adaptation (TTA) emerged. In contrast to standard TTA, continual TTA considers not only a single domain shift, but a sequence of shifts. Gradual TTA further exploits the property that some shifts evolve gradually over time. Since in both settings long test sequences are present, error accumulation needs to be addressed for methods relying on self-training. In this work, we propose and show that in the setting of TTA, the symmetric cross-entropy is better suited as a consistency loss for mean teachers compared to the commonly used cross-entropy. This is justified by our analysis with respect to the (symmetric) cross-entropy's gradient properties. To pull the test feature space closer to the source domain, where the pre-trained model is well posed, contrastive learning is leveraged. Since applications differ in their requirements, we address several settings, including having source data available and the more challenging source-free setting. We demonstrate the effectiveness of our proposed method “robust mean teacher” (RMT) on the continual and gradual corruption benchmarks CIFAR10C, CIFAR100C, and Imagenet-C. We further consider ImageNet-R and propose a new continual DomainNet-126 benchmark. State-of-the-art results are achieved on all benchmarks.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.com/mariodoebler/test-time-adaptation

----

## [735] NAR-Former: Neural Architecture Representation Learning Towards Holistic Attributes Prediction

**Authors**: *Yun Yi, Haokui Zhang, Wenze Hu, Nannan Wang, Xiaoyu Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00745](https://doi.org/10.1109/CVPR52729.2023.00745)

**Abstract**:

With the wide and deep adoption of deep learning models in real applications, there is an increasing need to model and learn the representations of the neural networks themselves. These models can be used to estimate attributes of different neural network architectures such as the accuracy and latency, without running the actual training or inference tasks. In this paper, we propose a neural architecture representation model that can be used to estimate these attributes holistically. Specifically, we first propose a simple and effective tokenizer to encode both the operation and topology information of a neural network into a single sequence. Then, we design a multi-stage fusion transformer to build a compact vector representation from the converted sequence. For efficient model training, we further propose an information flow consistency augmentation and correspondingly design an architecture consistency loss, which brings more benefits with less augmentation samples compared with previous random augmentation strategies. Experiment results on NAS-Bench-101, NAS-Bench-201, DARTS search space and NNLQP show that our proposed framework can be used to predict the aforementioned latency and accuracy attributes of both cell architectures and whole deep neural networks, and achieves promising performance. Code is available at https://github.com/yuny220/NAR-Former.

----

## [736] Visual Query Tuning: Towards Effective Usage of Intermediate Representations for Parameter and Memory Efficient Transfer Learning

**Authors**: *Cheng-Hao Tu, Zheda Mai, Wei-Lun Chao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00746](https://doi.org/10.1109/CVPR52729.2023.00746)

**Abstract**:

Intermediate features of a pretrained model have been shown informative for making accurate predictions on downstream tasks, even if the model backbone is kept frozen. The key challenge is how to utilize these intermediate features given their gigantic amount. We propose visual query tuning (VQT), a simple yet effective approach to aggregate intermediate features of Vision Transformers. Through introducing a handful of learnable “query” tokens to each layer, VQT leverages the inner workings of Transformers to “summarize” rich intermediate features of each layer, which can then be used to train the prediction heads of downstream tasks. As VQT keeps the intermediate features intact and only learns to combine them, it enjoys memory efficiency in training, compared to many other parameter-efficient fine-tuning approaches that learn to adapt features and need back-propagation through the entire backbone. This also suggests the complementary role between VQT and those approaches in transfer learning. Empirically, VQT consistently surpasses the state-of-the-art approach that utilizes intermediate features for transfer learning and outperforms full fine-tuning in many cases. Compared to parameter-efficient approaches that adapt features, VQT achieves much higher accuracy under memory constraints. Most importantly, VQT is compatible with these approaches to attain even higher accuracy, making it a simple add-on to further boost transfer learning. Code is available at https://github.com/andytu28/VQT.

----

## [737] Architecture, Dataset and Model-Scale Agnostic Data-free Meta-Learning

**Authors**: *Zixuan Hu, Li Shen, Zhenyi Wang, Tongliang Liu, Chun Yuan, Dacheng Tao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00747](https://doi.org/10.1109/CVPR52729.2023.00747)

**Abstract**:

The goal of data-free meta-learning is to learn useful prior knowledge from a collection of pre-trained models without accessing their training data. However, existing works only solve the problem in parameter space, which (i) ignore the fruitful data knowledge contained in the pretrained models; (ii) can not scale to large-scale pre-trained models; (iii) can only meta-learn pre-trained models with the same network architecture. To address those issues, we propose a unified framework, dubbed PURER, which contains: (1) ePisode cUrriculum inveRsion (ECI) during data-free meta training; and (2) invErsion calibRation following inner loop (ICFIL) during meta testing. During meta training, we propose ECI to perform pseudo episode training for learning to adapt fast to new unseen tasks. Specifically, we progressively synthesize a sequence of pseudo episodes by distilling the training data from each pre-trained model. The ECI adaptively increases the difficulty level of pseudo episodes according to the real-time feedback of the meta model. We formulate the optimization process of meta training with ECI as an adversarial form in an end-to-end manner. During meta testing, we further propose a simple plug-and-play supplement—ICFIL—only used during meta testing to narrow the gap between meta training and meta testing task distribution. Extensive experiments in various real-world scenarios show the superior performance of ours.

----

## [738] GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-Shot Class Incremental Task

**Authors**: *Huiping Zhuang, Zhenyu Weng, Run He, Zhiping Lin, Ziqian Zeng*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00748](https://doi.org/10.1109/CVPR52729.2023.00748)

**Abstract**:

Few-shot class incremental learning (FSCIL) aims to address catastrophic forgetting during class incremental learning in a few-shot learning setting. In this paper, we approach the FSCIL by adopting analytic learning, a technique that converts network training into linear problems. This is inspired by the fact that the recursive implementation (batch-by-batch learning) of analytic learning gives identical weights to that produced by training on the entire dataset at once. The recursive implementation and the weight-identical property highly resemble the FSCIL setting (phase-by-phase learning) and its goal of avoiding catastrophic forgetting. By bridging the FSCIL with the analytic learning, we propose a Gaussian kernel embedded analytic learning (GKEAL) for FSCIL. The key components of GKEAL include the kernel analytic module which allows the GKEAL to conduct FSCIL in a recursive manner, and the augmented feature concatenation module that balances the preference between old and new tasks especially effectively under the few-shot setting. Our experiments show that the GKEAL gives state-of-the-art performance on several benchmark datasets.

----

## [739] Mitigating Task Interference in Multi-Task Learning via Explicit Task Routing with Non-Learnable Primitives

**Authors**: *Chuntao Ding, Zhichao Lu, Shangguang Wang, Ran Cheng, Vishnu Naresh Boddeti*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00749](https://doi.org/10.1109/CVPR52729.2023.00749)

**Abstract**:

Multi-task learning (MTL) seeks to learn a single model to accomplish multiple tasks by leveraging shared information among the tasks. Existing MTL models, however, have been known to suffer from negative interference among tasks. Efforts to mitigate task interference have focused on either loss/gradient balancing or implicit parameter partitioning with partial overlaps among the tasks. In this paper, we propose ETR-NLP to mitigate task interference through a synergistic combination of non-learnable primitives (NLPs) and explicit task routing (ETR). Our key idea is to employ non-learnable primitives to extract a diverse set of task-agnostic features and recombine them into a shared branch common to all tasks and explicit task-specific branches reserved for each task. The non-learnable primitives and the explicit decoupling of learnable parameters into shared and task-specific ones afford the flexibility needed for minimizing task interference. We evaluate the efficacy of ETR-NLP networks for both image-level classification and pixel-level dense prediction MTL problems. Experimental results indicate that ETR-NLP significantly outperforms state-of-the-art baselines with fewer learnable parameters and similar FLOPs across all datasets. Code is available at this URL.

----

## [740] Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary

**Authors**: *Min Chen, Weizhuo Gao, Gaoyang Liu, Kai Peng, Chen Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00750](https://doi.org/10.1109/CVPR52729.2023.00750)

**Abstract**:

The practical needs of the “right to be forgotten” and poisoned data removal call for efficient machine unlearning techniques, which enable machine learning models to unlearn, or to forget a fraction of training data and its lineage. Recent studies on machine unlearning for deep neural networks (DNNs) attempt to destroy the influence of the forgetting data by scrubbing the model parameters. However, it is prohibitively expensive due to the large dimension of the parameter space. In this paper, we refocus our attention from the parameter space to the decision space of the DNN model, and propose Boundary Unlearning, a rapid yet effective way to unlearn an entire class from a trained DNN model. The key idea is to shift the decision boundary of the original DNN model to imitate the decision behavior of the model retrained from scratch. We develop two novel boundary shift methods, namely Boundary Shrink and Boundary Expanding, both of which can rapidly achieve the utility and privacy guarantees. We extensively evaluate Boundary Unlearning on CIFAR-10 and Vggface2 datasets, and the results show that Boundary Unlearning can effectively forget the forgetting class on image classification and face recognition tasks, with an expected speed-up of 17x and 19x, respectively, compared with retraining from the scratch.

----

## [741] Task Difficulty Aware Parameter Allocation & Regularization for Lifelong Learning

**Authors**: *Wenjin Wang, Yunqing Hu, Qianglong Chen, Yin Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00751](https://doi.org/10.1109/CVPR52729.2023.00751)

**Abstract**:

Parameter regularization or allocation methods are effective in overcoming catastrophic forgetting in lifelong learning. However, they solve all tasks in a sequence uniformly and ignore the differences in the learning difficulty of different tasks. So parameter regularization methods face significant forgetting when learning a new task very different from learned tasks, and parameter allocation methods face unnecessary parameter overhead when learning simple tasks. In this paper, we propose the Parameter Allocation & Regularization (PAR), which adaptively select an appropriate strategy for each task from parameter allocation and regularization based on its learning difficulty. A task is easy for a model that has learned tasks related to it and vice versa. We propose a divergence estimation method based on the Nearest-Prototype distance to measure the task relatedness using only features of the new task. Moreover, we propose a time-efficient relatedness-aware sampling-based architecture search strategy to reduce the parameter overhead for allocation. Experimental results on multiple benchmarks demonstrate that, compared with SOTAs, our method is scalable and significantly reduces the model's redundancy while improving the model's performance. Further qualitative analysis indicates that PAR obtains reasonable task-relatedness.

----

## [742] Learning to Retain while Acquiring: Combating Distribution-Shift in Adversarial Data-Free Knowledge Distillation

**Authors**: *Gaurav Patel, Konda Reddy Mopuri, Qiang Qiu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00752](https://doi.org/10.1109/CVPR52729.2023.00752)

**Abstract**:

Data-free Knowledge Distillation (DFKD) has gained popularity recently, with the fundamental idea of carrying out knowledge transfer from a Teacher neural network to a Student neural network in the absence of training data. However, in the Adversarial DFKD framework, the student network's accuracy, suffers due to the non-stationary distribution of the pseudo-samples under multiple generator updates. To this end, at every generator update, we aim to maintain the student's performance on previously encountered examples while acquiring knowledge from samples of the current distribution. Thus, we propose a meta-learning inspired framework by treating the task of Knowledge-Acquisition (learning from newly generated samples) and Knowledge-Retention (retaining knowledge on previously met samples) as meta-train and meta-test, respectively. Hence, we dub our method as Learning to Retain while Acquiring. Moreover, we identify an implicit aligning factor between the Knowledge-Retention and Knowledge-Acquisition tasks indicating that the proposed student update strategy enforces a common gradient direction for both tasks, alleviating interference between the two objectives. Finally, we support our hypothesis by exhibiting extensive evaluation and comparison of our method with prior arts on multiple datasets.

----

## [743] A Unified Knowledge Distillation Framework for Deep Directed Graphical Models

**Authors**: *Yizhuo Chen, Kaizhao Liang, Zhe Zeng, Shuochao Yao, Huajie Shao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00753](https://doi.org/10.1109/CVPR52729.2023.00753)

**Abstract**:

Knowledge distillation (KD) is a technique that transfers the knowledge from a large teacher network to a small student network. It has been widely applied to many different tasks, such as model compression and federated learning. However, existing KD methods fail to generalize to general deep directed graphical models (DGMs) with arbitrary layers of random variables. We refer by deep DGMs to DGMs whose conditional distributions are parameterized by deep neural networks. In this work, we propose a novel unified knowledge distillation framework for deep DGMs on various applications. Specifically, we leverage the reparameterization trick to hide the intermediate latent variables, resulting in a compact DGM. Then we develop a surrogate distillation loss to reduce error accumulation through multiple layers of random variables. Moreover, we present the connections between our method and some existing knowledge distillation approaches. The proposed framework is evaluated on four applications: data-free hierarchical variational autoencoder (VAE) compression, data-free variational recurrent neural networks (VRNN) compression, data-free Helmholtz Machine (HM) compression, and VAE continual learning. The results show that our distillation method out-performs the baselines in data-free model compression tasks. We further demonstrate that our method significantly improves the performance of KD-based continual learning for data generation. Our source code is available at https://github.com/YizhuoChen99/KD4DGM-CVPR.

----

## [744] Coaching a Teachable Student

**Authors**: *Jimuyang Zhang, Zanming Huang, Eshed Ohn-Bar*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00754](https://doi.org/10.1109/CVPR52729.2023.00754)

**Abstract**:

We propose a novel knowledge distillation framework for effectively teaching a sensorimotor student agent to drive from the supervision of a privileged teacher agent. Current distillation for sensorimotor agents methods tend to result in suboptimal learned driving behavior by the student, which we hypothesize is due to inherent differences between the input, modeling capacity, and optimization processes of the two agents. We develop a novel distillation scheme that can address these limitations and close the gap between the sensorimotor agent and its privileged teacher. Our key insight is to design a student which learns to align their input features with the teacher's privileged Bird's Eye View (BEV) space. The student then can benefit from direct supervision by the teacher over the internal representation learning. To scaffold the difficult sensorimotor learning task, the student model is optimized via a student-paced coaching mechanism with various auxiliary supervision. We further propose a high-capacity imitation learned privileged agent that surpasses prior privileged agents in CARLA and ensures the student learns safe driving behavior. Our proposed sensorimotor agent results in a robust image-based behavior cloning agent in CARLA, improving over current models by over 20.6% in driving score without requiring LiDAR, historical observations, ensemble of models, on-policy data aggregation or reinforcement learning.

----

## [745] Adaptive Plasticity Improvement for Continual Learning

**Authors**: *Yan-Shuo Liang, Wu-Jun Li*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00755](https://doi.org/10.1109/CVPR52729.2023.00755)

**Abstract**:

Many works have tried to solve the catastrophic forgetting (CF) problem in continual learning (lifelong learning). However, pursuing non-forgetting on old tasks may damage the model's plasticity for new tasks. Although some methods have been proposed to achieve stability-plasticity trade-off, no methods have considered evaluating a model's plasticity and improving plasticity adaptively for a new task. In this work, we propose a new method, called adaptive plasticity improvement (API), for continual learning. Besides the ability to overcome CF on old tasks, API also tries to evaluate the model's plasticity and then adaptively improve the model's plasticity for learning a new task if necessary. Experiments on several real datasets show that API can outperform other state-of-the-art baselines in terms of both accuracy and memory usage.

----

## [746] Improving Generalization of Meta-Learning with Inverted Regularization at Inner-Level

**Authors**: *Lianzhe Wang, Shiji Zhou, Shanghang Zhang, Xu Chu, Heng Chang, Wenwu Zhu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00756](https://doi.org/10.1109/CVPR52729.2023.00756)

**Abstract**:

Despite the broad interest in meta-learning, the generalization problem remains one of the significant challenges in this field. Existing works focus on meta-generalization to unseen tasks at the meta-level by regularizing the meta-loss, while ignoring that adapted models may not generalize to the task domains at the adaptation level. In this paper, we propose a new regularization mechanism for meta-learning - Minimax-Meta Regularization, which employs inverted regularization at the inner loop and ordinary regularization at the outer loop during training. In particular, the inner inverted regularization makes the adapted model more difficult to generalize to task domains; thus, optimizing the outer-loop loss forces the meta-model to learn meta-knowledge with better generalization. Theoretically, we prove that inverted regularization improves the meta-testing performance by reducing generalization errors. We conduct extensive experiments on the representative scenarios, and the results show that our method consistently improves the performance of meta-learning algorithms.

----

## [747] Trainable Projected Gradient Method for Robust Fine-Tuning

**Authors**: *Junjiao Tian, Xiaoliang Dai, Chih-Yao Ma, Zecheng He, Yen-Cheng Liu, Zsolt Kira*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00757](https://doi.org/10.1109/CVPR52729.2023.00757)

**Abstract**:

Recent studies on transfer learning have shown that selectively fine-tuning a subset of layers or customizing different learning rates for each layer can greatly improve robustness to out-of-distribution (OOD) data and retain generalization capability in the pre-trained models. However, most of these methods employ manually crafted heuristics or expensive hyper-parameter searches, which prevent them from scaling up to large datasets and neural networks. To solve this problem, we propose Trainable Projected Gradient Method (TPGM) to automatically learn the constraint imposed for each layer for a fine-grained fine-tuning regularization. This is motivated by formulating fine-tuning as a bi-level constrained optimization problem. Specifically, TPGM maintains a set of projection radii, i.e., distance constraints between the fine-tuned model and the pretrained model, for each layer, and enforces them through weight projections. To learn the constraints, we propose a bi-level optimization to automatically learn the best set of projection radii in an end-to-end manner. Theoretically, we show that the bi-level optimization formulation is the key to learning different constraints for each layer. Empirically, with little hyper-parameter search cost, TPGM outperforms existing fine-tuning methods in OOD performance while matching the best in-distribution (ID) performance. For example, when fine-tuned on DomainNet-Real and ImageNet, compared to vanilla fine-tuning, TPGM shows 22% and 10% relative OOD improvement respectively on their sketch counterparts. Code is available at https://github.com/PotatoTian/TPGM.

----

## [748] Imitation Learning as State Matching via Differentiable Physics

**Authors**: *Siwei Chen, Xiao Ma, Zhongwen Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00758](https://doi.org/10.1109/CVPR52729.2023.00758)

**Abstract**:

Existing imitation learning (IL) methods such as inverse reinforcement learning (IRL) usually have a double-loop training process, alternating between learning a reward function and a policy and tend to suffer long training time and high variance. In this work, we identify the benefits of differentiable physics simulators and propose a new IL method, i.e., Imitation Learning as State Matching via Differentiable Physics (ILD), which gets rid of the double-loop design and achieves significant improvements in final performance, convergence speed, and stability. The proposed ILD incorporates the differentiable physics simulator as a physics prior into its computational graph for policy learning. ILD unrolls the dynamics by sampling actions from a parameterized policy and minimizing the distance between the expert trajectory and the agent trajectory. It back-propagates the gradient into the policy via temporal physics operators, which improves the transferability to unseen environments and yields higher final performance. ILD has a single-loop structure that stabilizes and speeds up training. It dynamically selects learning objectives for each state during optimization to simplify the complex optimization land-scape. Experiments show that ILD outperforms state-of-the-art methods in continuous control tasks with Brax, and can be applied to deformable object manipulation tasks, generalized to unseen configurations.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The link to the code: https://github.com/sail-sg/ILD

----

## [749] Improved Distribution Matching for Dataset Condensation

**Authors**: *Ganlong Zhao, Guanbin Li, Yipeng Qin, Yizhou Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00759](https://doi.org/10.1109/CVPR52729.2023.00759)

**Abstract**:

Dataset Condensation aims to condense a large dataset into a smaller one while maintaining its ability to train a well-performing model, thus reducing the storage cost and training effort in deep learning applications. However, conventional dataset condensation methods are optimization-oriented and condense the dataset by performing gradient or parameter matching during model optimization, which is computationally intensive even on small datasets and models. In this paper, we propose a novel dataset condensation method based on distribution matching, which is more efficient and promising. Specifically, we identify two important shortcomings of naive distribution matching (i.e., imbalanced feature numbers and unvalidated embeddings for distance computation) and address them with three novel techniques (i.e., partitioning and expansion augmentation, efficient and enriched model sampling, and class-aware distribution regularization). Our simple yet effective method outperforms most previous optimization-oriented methods with much fewer computational resources, thereby scaling data condensation to larger datasets and models. Extensive experiments demonstrate the effectiveness of our method. Codes are available at https://github.com/uitrbn/IDM

----

## [750] A General Regret Bound of Preconditioned Gradient Method for DNN Training

**Authors**: *Hongwei Yong, Ying Sun, Lei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00760](https://doi.org/10.1109/CVPR52729.2023.00760)

**Abstract**:

While adaptive learning rate methods, such as Adam, have achieved remarkable improvement in optimizing Deep Neural Networks (DNNs), they consider only the diagonal elements of the full preconditioned matrix. Though the full-matrix preconditioned gradient methods theoretically have a lower regret bound, they are impractical for use to train DNNs because of the high complexity. In this paper, we present a general regret bound with a constrained full-matrix preconditioned gradient, and show that the updating formula of the preconditioner can be derived by solving a cone-constrained optimization problem. With the block-diagonal and Kronecker-factorized constraints, a specific guide function can be obtained. By minimizing the upper bound of the guide function, we develop a new DNN optimizer, termed AdaBK. A series of techniques, including statistics updating, dampening, efficient matrix inverse root computation, and gradient amplitude preservation, are developed to make AdaBK effective and efficient to implement. The proposed AdaBK can be readily embedded into many existing DNN optimizers, e.g., SGDM and Adam W, and the corresponding SGDM _BK and Adam W _BK algorithms demonstrate significant improvements over existing DNN optimizers on benchmark vision tasks, including image classification, object detection and segmentation. The code is publicly available at https://github.com/Yonghongwei/AdaBK.

----

## [751] From Node Interaction to Hop Interaction: New Effective and Scalable Graph Learning Paradigm

**Authors**: *Jie Chen, Zilong Li, Yin Zhu, Junping Zhang, Jian Pu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00761](https://doi.org/10.1109/CVPR52729.2023.00761)

**Abstract**:

Existing Graph Neural Networks (GNNs) follow the message-passing mechanism that conducts information interaction among nodes iteratively. While considerable progress has been made, such node interaction paradigms still have the following limitation. First, the scalability limitation precludes the broad application of GNNs in large-scale industrial settings since the node interaction among rapidly expanding neighbors incurs high computation and memory costs. Second, the over-smoothing problem restricts the discrimination ability of nodes, i.e., node representations of different classes will converge to indistinguishable after repeated node interactions. In this work, we propose a novel hop interaction paradigm to address these limitations simultaneously. The core idea is to convert the interaction target among nodes to pre-processed multi-hop features inside each node. We design a simple yet effective HopGNN framework that can easily utilize existing GNNs to achieve hop interaction. Furthermore, we propose a multi-task learning strategy with a self-supervised learning objective to enhance HopGNN. We conduct extensive experiments on 12 benchmark datasets in a wide range of domains, scales, and smoothness of graphs. Experimental results show that our methods achieve superior performance while maintaining high scalability and efficiency. The code is at https://github.com/JC-202/HopGNN.

----

## [752] Constructing Deep Spiking Neural Networks from Artificial Neural Networks with Knowledge Distillation

**Authors**: *Qi Xu, Yaxin Li, Jiangrong Shen, Jian K. Liu, Huajin Tang, Gang Pan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00762](https://doi.org/10.1109/CVPR52729.2023.00762)

**Abstract**:

Spiking neural networks (SNNs) are well-known as brain-inspired models with high computing efficiency, due to a key component that they utilize spikes as information units, close to the biological neural systems. Although spiking based models are energy efficient by taking advantage of discrete spike signals, their performance is limited by current network structures and their training methods. As discrete signals, typical SNNs cannot apply the gradient descent rules directly into parameter adjustment as artificial neural networks (ANNs). Aiming at this limitation, here we propose a novel method of constructing deep SNN models with knowledge distillation (KD) that uses ANN as the teacher model and SNN as the student model. Through the ANN-SNN joint training algorithm, the student SNN model can learn rich feature information from the teacher ANN model through the KD method, yet it avoids training SNN from scratch when communicating with non-differentiable spikes. Our method can not only build a more efficient deep spiking structure feasibly and reasonably but use few time steps to train the whole model compared to direct training or ANN to SNN methods. More importantly, it has a superb ability of noise immunity for various types of artificial noises and natural signals. The proposed novel method provides efficient ways to improve the performance of SNN through constructing deeper structures in a high-throughput fashion, with potential usage for light and efficient brain-inspired computing of practical scenarios.

----

## [753] Rate Gradient Approximation Attack Threats Deep Spiking Neural Networks

**Authors**: *Tong Bu, Jianhao Ding, Zecheng Hao, Zhaofei Yu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00763](https://doi.org/10.1109/CVPR52729.2023.00763)

**Abstract**:

Spiking Neural Networks (SNNs) have attracted significant attention due to their energy-efficient properties and potential application on neuromorphic hardware. State-of-the-art SNNs are typically composed of simple Leaky Integrate-and-Fire (LIF) neurons and have become comparable to ANNs in image classification tasks on large-scale datasets. However, the robustness of these deep SNNs has not yet been fully uncovered. In this paper, we first experimentally observe that layers in these SNNs mostly communicate by rate coding. Based on this rate coding property, we develop a novel rate coding SNN-specified attack method, Rate Gradient Approximation Attack (RGA). We generalize the RGA attack to SNNs composed of LIF neurons with different leaky parameters and input encoding by designing surrogate gradients. In addition, we develop the time-extended enhancement to generate more effective adversarial examples. The experiment results indicate that our proposed RGA attack is more effective than the previous attack and is less sensitive to neuron hyperparameters. We also conclude from the experiment that rate-coded SNN composed of LIF neurons is not secure, which calls for exploring training methods for SNNs composed of complex neurons and other neuronal codings. Code is available at https://github.com/putshua/SNN_attack_RGA

----

## [754] MobileOne: An Improved One millisecond Mobile Backbone

**Authors**: *Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, Anurag Ranjan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00764](https://doi.org/10.1109/CVPR52729.2023.00764)

**Abstract**:

Efficient neural network backbones for mobile devices are often optimized for metrics such as FLOPs or parameter count. However, these metrics may not correlate well with latency of the network when deployed on a mobile device. Therefore, we perform extensive analysis of different metrics by deploying several mobile-friendly networks on a mobile device. We identify and analyze architectural and optimization bottlenecks in recent efficient neural networks and provide ways to mitigate these bottlenecks. To this end, we design an efficient backbone MobileOne, with variants achieving an inference time under 1 ms on an iPhone12 with 75.9% top-1 accuracy on ImageNet. We show that MobileOne achieves state-of-the-art performance within the efficient architectures while being many times faster on mobile. Our best model obtains similar performance on ImageNet as MobileFormer while being 38×faster. Our model obtains 2.3% better top-1 accuracy on ImageNet than EfficientNet at similar latency. Furthermore, we show that our model generalizes to multiple tasks - image classification, object detection, and semantic segmentation with significant improvements in latency and accuracy as compared to existing efficient architectures when deployed on a mobile device. Code and models are available at https://github.com/apple/ml-mobileone

----

## [755] Understanding Masked Autoencoders via Hierarchical Latent Variable Models

**Authors**: *Lingjing Kong, Martin Q. Ma, Guangyi Chen, Eric P. Xing, Yuejie Chi, Louis-Philippe Morency, Kun Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00765](https://doi.org/10.1109/CVPR52729.2023.00765)

**Abstract**:

Masked autoencoder (MAE), a simple and effective self-supervised learning framework based on the reconstruction of masked image regions, has recently achieved prominent success in a variety of vision tasks. Despite the emergence of intriguing empirical observations on MAE, a theoretically principled understanding is still lacking. In this work, we formally characterize and justify existing empirical in-sights and provide theoretical guarantees of MAE. We formulate the underlying data-generating process as a hierarchical latent variable model, and show that under reasonable assumptions, MAE provably identifies a set of latent variables in the hierarchical model, explaining why MAE can extract high-level information from pixels. Further, we show how key hyperparameters in MAE (the masking ratio and the patch size) determine which true latent variables to be recovered, therefore influencing the level of semantic information in the representation. Specifically, extremely large or small masking ratios inevitably lead to low-level representations. Our theory offers coherent explanations of existing empirical observations and provides insights for potential empirical improvements and fundamental limitations of the masked-reconstruction paradigm. We conduct extensive experiments to validate our theoretical insights.

----

## [756] Training Debiased Subnetworks with Contrastive Weight Pruning

**Authors**: *Geon Yeong Park, Sangmin Lee, Sang Wan Lee, Jong Chul Ye*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00766](https://doi.org/10.1109/CVPR52729.2023.00766)

**Abstract**:

Neural networks are often biased to spuriously correlated features that provide misleading statistical evidence that does not generalize. This raises an interesting question: “Does an optimal unbiased functional subnetwork exist in a severely biased network? If so, how to extract such subnetwork?” While empirical evidence has been accumulated about the existence of such unbiased subnetworks, these observations are mainly based on the guidance of ground-truth unbiased samples. Thus, it is unexplored how to discover the optimal subnetworks with biased training datasets in practice. To address this, here we first present our theoretical insight that alerts potential limitations of existing algorithms in exploring unbiased subnetworks in the presence of strong spurious correlations. We then further elucidate the importance of bias-conflicting samples on structure learning. Motivated by these observations, we propose a Debiased Contrastive Weight Pruning (DCWP) algorithm, which probes unbiased subnetworks without expensive group annotations. Experimental results demonstrate that our approach significantly outperforms state-of-the-art debiasing methods despite its considerable reduction in the number of parameters.

----

## [757] One-Shot Model for Mixed-Precision Quantization

**Authors**: *Ivan Koryakovskiy, Alexandra Yakovleva, Valentin Buchnev, Temur Isaev, Gleb Odinokikh*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00767](https://doi.org/10.1109/CVPR52729.2023.00767)

**Abstract**:

Neural network quantization is a popular approach for model compression. Modern hardware supports quantization in mixed-precision mode, which allows for greater compression rates but adds the challenging task of searching for the optimal bit width. The majority of existing searchers find a single mixed-precision architecture. To select an architecture that is suitable in terms of performance and resource consumption, one has to restart searching multiple times. We focus on a specific class of methods that find tensor bit width using gradient-based optimization. First, we theoretically derive several methods that were empirically proposed earlier. Second, we present a novel One-Shot method that finds a diverse set of Pareto-front architectures in 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(1)$</tex>
 time. For large models, the proposed method is 5 times more efficient than existing methods. We verify the method on two classification and super-resolution models and show above 0.93 correlation score between the predicted and actual model performance. The Paretofront architecture selection is straightforward and takes only 20 to 40 supernet evaluations, which is the new state-of-the-art result to the best of our knowledge.

----

## [758] Solving Oscillation Problem in Post-Training Quantization Through a Theoretical Perspective

**Authors**: *Yuexiao Ma, Huixia Li, Xiawu Zheng, Xuefeng Xiao, Rui Wang, Shilei Wen, Xin Pan, Fei Chao, Rongrong Ji*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00768](https://doi.org/10.1109/CVPR52729.2023.00768)

**Abstract**:

Post-training quantization (PTQ) is widely regarded as one of the most efficient compression methods practically, benefitting from its data privacy and low computation costs. We argue that an overlooked problem of oscillation is in the PTQ methods. In this paper, we take the initiative to explore and present a theoretical proof to explain why such a problem is essential in PTQ. And then, we try to solve this problem by introducing a principled and generalized frame-work theoretically. In particular, we first formulate the oscillation in PTQ and prove the problem is caused by the difference in module capacity. To this end, we define the module capacity (ModCap) under data-dependent and data-free scenarios, where the differentials between adjacent modules are used to measure the degree of oscillation. The problem is then solved by selecting top-k differentials, in which the corresponding modules are jointly optimized and quantized. Extensive experiments demonstrate that our method successfully reduces the performance drop and is generalized to different neural networks and PTQ methods. For example, with 2/4 bit ResNet-50 quantization, our method surpasses the previous state-of-the-art method by 1.9%. It becomes more significant on small model quantization, e.g. surpasses BRECQ method by 6.61% on MobileNetV2 × 0.5.

----

## [759] Adaptive Data-Free Quantization

**Authors**: *Biao Qian, Yang Wang, Richang Hong, Meng Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00769](https://doi.org/10.1109/CVPR52729.2023.00769)

**Abstract**:

Data-free quantization (DFQ) recovers the performance of quantized network (Q) without the original data, but generates the fake sample via a generator (G) by learning from full-precision network (P), which, however, is totally independent of Q, overlooking the adaptability of the knowledge from generated samples, i.e., informative or not to the learning process of Q, resulting into the overflow of generalization error. Building on this, several critical questions — how to measure the sample adaptability to Q under varied bit-width scenarios? whether the largest adaptability is the best? how to generate the samples with adaptive adaptability to improve Q's generalization? To answer the above questions, in this paper, we propose an Adaptive Data-Free Quantization (AdaDFQ) method, which revisits DFQ from a zero-sum game perspective upon the sample adaptability between two players — a generator and a quantized network. Following this viewpoint, we further define the disagreement and agreement samples to form two boundaries, where the margin between two boundaries is optimized to adaptively regulate the adaptability of generated samples to Q, so as to address the over-and-under fitting issues. Our AdaDFQ reveals: 1) the largest adaptability is NOT the best for sample generation to benefit Q's generalization; 2) the knowledge of the generated sample should not be informative to Q only, but also related to the category and distribution information of the training data for P. The theoretical and empirical analysis validate the advantages of AdaDFQ over the state-of-the-arts. Our code is available at https://github.com/hfutqian/AdaDFQ.

----

## [760] Learning to Generate Image Embeddings with User-Level Differential Privacy

**Authors**: *Zheng Xu, Maxwell D. Collins, Yuxiao Wang, Liviu Panait, Sewoong Oh, Sean Augenstein, Ting Liu, Florian Schroff, H. Brendan McMahan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00770](https://doi.org/10.1109/CVPR52729.2023.00770)

**Abstract**:

Small on-device models have been successfully trained with user-level differential privacy (DP) for next word prediction and image classification tasks in the past. However, existing methods can fail when directly applied to learn embedding models using supervised training data with a large class space. To achieve user-level DP for large image-to-embedding feature extractors, we propose DP-FedEmb, a variant of federated learning algorithms with per-user sensitivity control and noise addition, to train from user-partitioned data centralized in the datacenter. DP-FedEmb combines virtual clients, partial aggregation, private local fine-tuning, and public Pre-training to achieve strong privacy utility trade-offs. We apply DP-FedEmb to train image embedding models for faces, landmarks and natural species, and demonstrate its superior utility under same privacy budget on benchmark datasets DigiFace. EMNIST, GLD and iNaturalist. We further illustrate it is possible to achieve strong user-level DP guarantees of ∊ < 2 while controlling the utility drop within 5%, when millions of users can participate in training.

----

## [761] Cross-GAN Auditing: Unsupervised Identification of Attribute Level Similarities and Differences Between Pretrained Generative Models

**Authors**: *Matthew L. Olson, Shusen Liu, Rushil Anirudh, Jayaraman J. Thiagarajan, Peer-Timo Bremer, Weng-Keen Wong*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00771](https://doi.org/10.1109/CVPR52729.2023.00771)

**Abstract**:

Generative Adversarial Networks (GANs) are notoriously difficult to train especially for complex distributions and with limited data. This has driven the need for tools to audit trained networks in human intelligible format, for example, to identify biases or ensure fairness. Existing GAN audit tools are restricted to coarse-grained, modeldata comparisons based on summary statistics such as FID or recall. In this paper, we propose an alternative approach that compares a newly developed GAN against a prior baseline. To this end, we introduce Cross-GAN Auditing (xGA) that, given an established “reference” GAN and a newly proposed “client” GAN, jointly identifies intelligible attributes that are either common across both GANs, novel to the client GAN, or missing from the client GAN. This provides both users and model developers an intuitive assessment of similarity and differences between GANs. We introduce novel metrics to evaluate attribute-based GAN auditing approaches and use these metrics to demonstrate quantitatively that xGA outperforms baseline approaches. We also include qualitative results that illustrate the common, novel and missing attributes identified by xGA from GANs trained on a variety of image datasets 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

----

## [762] HandsOff: Labeled Dataset Generation With No Additional Human Annotations

**Authors**: *Austin Xu, Mariya I. Vasileva, Achal Dave, Arjun Seshadri*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00772](https://doi.org/10.1109/CVPR52729.2023.00772)

**Abstract**:

Recent work leverages the expressive power of generative adversarial networks (GANs) to generate labeled synthetic datasets. These dataset generation methods often require new annotations of synthetic images, which forces practitioners to seek out annotators, curate a set of synthetic images, and ensure the quality of generated labels. We introduce the HandsOff framework, a technique capable of producing an unlimited number of synthetic images and corresponding labels after being trained on less than 50 preexisting labeled images. Our framework avoids the practical drawbacks of prior work by unifying the field of GAN inversion with dataset generation. We generate datasets with rich pixel-wise labels in multiple challenging domains such as faces, cars, full-body human poses, and urban driving scenes. Our method achieves state-of-the-art performance in semantic segmentation, keypoint detection, and depth estimation compared to prior dataset generation approaches and transfer learning baselines. We additionally showcase its ability to address broad challenges in model development which stem from fixed, hand-annotated datasets, such as the long-tail problem in semantic segmentation. Project page: austinxu87.github.io/handsoff.

----

## [763] Attribute-Preserving Face Dataset Anonymization via Latent Code Optimization

**Authors**: *Simone Barattin, Christos Tzelepis, Ioannis Patras, Nicu Sebe*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00773](https://doi.org/10.1109/CVPR52729.2023.00773)

**Abstract**:

This work addresses the problem of anonymizing the identity of faces in a dataset of images, such that the privacy of those depicted is not violated, while at the same time the dataset is useful for downstream task such as for training machine learning models. To the best of our knowledge, we are the first to explicitly address this issue and deal with two major drawbacks of the existing state-of-the-art approaches, namely that they (i) require the costly training of additional, purpose-trained neural networks, and/or (ii) fail to retain the facial attributes of the original images in the anonymized counterparts, the preservation of which is of paramount importance for their use in downstream tasks. We accordingly present a task-agnostic anonymization procedure that directly optimizes the images' latent representation in the latent space of a pretrained GAN. By optimizing the latent codes directly, we ensure both that the identity is of a desired distance away from the original (with an identity obfuscation loss), whilst preserving the facial attributes (using a novel feature-matching loss in FaRL's [48] deep feature space). We demonstrate through a series of both qualitative and quantitative experiments that our method is capable of anonymizing the identity of the images whilst-crucially-better-preserving the facial attributes. We make the code and the pretrained models publicly available at: https://github.com/chi0tzp/FALCO.

----

## [764] Fake it Till You Make it: Learning Transferable Representations from Synthetic ImageNet Clones

**Authors**: *Mert Bülent Sariyildiz, Karteek Alahari, Diane Larlus, Yannis Kalantidis*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00774](https://doi.org/10.1109/CVPR52729.2023.00774)

**Abstract**:

Recent image generation models such as Stable Diffusion have exhibited an impressive ability to generate fairly realistic images starting from a simple text prompt. Could such models render real images obsolete for training image prediction models? In this paper, we answer part of this provocative question by investigating the need for real images when training models for ImageNet classification. Provided only with the class names that have been used to build the dataset, we explore the ability of Stable Diffusion to generate synthetic clones of ImageNet and measure how useful these are for training classification models from scratch. We show that with minimal and class-agnostic prompt engineering, ImageNet clones are able to close a large part of the gap between models produced by synthetic images and models trained with real images, for the several standard classification benchmarks that we consider in this study. More importantly, we show that models trained on synthetic images exhibit strong generalization properties and perform on par with models trained on real data for transfer. Project page: https://europe.naverlabs.com/imagenet-sd

----

## [765] Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection

**Authors**: *Hui Lv, Zhongqi Yue, Qianru Sun, Bin Luo, Zhen Cui, Hanwang Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00775](https://doi.org/10.1109/CVPR52729.2023.00775)

**Abstract**:

Weakly Supervised Video Anomaly Detection (WSVAD) is challenging because the binary anomaly label is only given on the video level, but the output requires snippet-level predictions. So, Multiple Instance Learning (MIL) is prevailing in WSVAD. However, MIL is notoriously known to suffer from many false alarms because the snippet-level detector is easily biased towards the abnormal snippets with simple context, confused by the normality with the same bias, and missing the anomaly with a different pattern. To this end, we propose a new MIL framework: Unbiased MIL (UMIL), to learn unbiased anomaly features that improve WSVAD. At each MIL training iteration, we use the current detector to divide the samples into two groups with different context biases: the most confident abnormal/normal snippets and the rest ambiguous ones. Then, by seeking the invariant features across the two sample groups, we can remove the variant context biases. Extensive experiments on benchmarks UCF-Crime and TAD demonstrate the effectiveness of our UMIL. Our code is provided at https://github.com/ktr-hubrt/UMIL.

----

## [766] Multimodal Industrial Anomaly Detection via Hybrid Fusion

**Authors**: *Yue Wang, Jinlong Peng, Jiangning Zhang, Ran Yi, Yabiao Wang, Chengjie Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00776](https://doi.org/10.1109/CVPR52729.2023.00776)

**Abstract**:

2D-based Industrial Anomaly Detection has been widely discussed, however, multimodal industrial anomaly detection based on 3D point clouds and RGB images still has many untouched fields. Existing multimodal industrial anomaly detection methods directly concatenate the multimodal features, which leads to a strong disturbance between features and harms the detection performance. In this paper, we propose Multi-3D-Memory (M3DM), a novel multimodal anomaly detection method with hybrid fusion scheme: firstly, we design an unsupervised feature fusion with patch-wise contrastive learning to encourage the interaction of different modal features; secondly, we use a decision layer fusion with multiple memory banks to avoid loss of information and additional novelty classifiers to make the final decision. We further propose a point feature alignment operation to better align the point cloud and RGB features. Extensive experiments show that our multi-modal industrial anomaly detection model outperforms the state-of-the-art (SOTA) methods on both detection and segmentation precision on MVTec-3D AD dataset. Code at github.com/nomewang/M3DM.

----

## [767] FedSeg: Class-Heterogeneous Federated Learning for Semantic Segmentation

**Authors**: *Jiaxu Miao, Zongxin Yang, Leilei Fan, Yi Yang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00777](https://doi.org/10.1109/CVPR52729.2023.00777)

**Abstract**:

Federated Learning (FL) is a distributed learning paradigm that collaboratively learns a global model by multiple clients with data privacy-preserving. Although many FL algorithms have been proposed for classification tasks, few works focus on more challenging semantic segmentation tasks, especially in the class-heterogeneous FL situation. Compared with classification, the issues from heterogeneous FL for semantic segmentation are more severe: (1) Due to the non-IID distribution, different clients may contain inconsistent foreground-background classes, resulting in divergent local updates. (2) Class-heterogeneity for complex dense prediction tasks makes the local optimum of clients farther from the global optimum. In this work, we propose FedSeg, a basic federated learning approach for class-heterogeneous semantic segmentation. We first propose a simple but strong modified cross-entropy loss to correct the local optimization and address the foreground-background inconsistency problem. Based on it, we introduce pixel-level contrastive learning to enforce local pixel embeddings belonging to the global semantic space. Extensive experiments on four semantic segmentation benchmarks (Cityscapes, CamVID, PascalVOC and ADE20k) demonstrate the effectiveness of our FedSeg. We hope this work will attract more attention from the FL community to the challenging semantic segmentation federated learning.

----

## [768] Decentralized Learning with Multi-Headed Distillation

**Authors**: *Andrey Zhmoginov, Mark Sandler, Nolan Miller, Gus Kristiansen, Max Vladymyrov*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00778](https://doi.org/10.1109/CVPR52729.2023.00778)

**Abstract**:

Decentralized learning with private data is a central problem in machine learning. We propose a novel distillation-based decentralized learning technique that allows multiple agents with private non-lid data to learn from each other, without having to share their data, weights or weight updates. Our approach is communication efficient, utilizes an unlabeled public dataset and uses multiple auxiliary heads for each client, greatly improving training efficiency in the case of heterogeneous data. This approach allows individual models to preserve and enhance performance on their private tasks while also dramatically improving their performance on the global aggregated data distribution. We study the effects of data and model architecture heterogeneity and the impact of the underlying communication graph topology on learning efficiency and show that our agents can significantly improve their performance compared to learning in isolation.

----

## [769] Learning Federated Visual Prompt in Null Space for MRI Reconstruction

**Authors**: *Chun-Mei Feng, Bangjun Li, Xinxing Xu, Yong Liu, Huazhu Fu, Wangmeng Zuo*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00779](https://doi.org/10.1109/CVPR52729.2023.00779)

**Abstract**:

Federated Magnetic Resonance Imaging (MRI) reconstruction enables multiple hospitals to collaborate distributedly without aggregating local data, thereby protecting patient privacy. However, the data heterogeneity caused by different MRI protocols, insufficient local training data, and limited communication bandwidth inevitably impair global model convergence and updating. In this paper, we propose a new algorithm, FedPR, to learn federated visual prompts in the null space of global prompt for MRI reconstruction. FedPR is a new federated paradigm that adopts a powerful pre-trained model while only learning and communicating the prompts with few learnable parameters, thereby significantly reducing communication costs and achieving competitive performance on limited local data. Moreover, to deal with catastrophic forgetting caused by data heterogeneity, FedPR also updates efficient federated visual prompts that project the local prompts into an approximate null space of the global prompt, thereby suppressing the interference of gradients on the server performance. Extensive experiments on federated MRI show that FedPR significantly outperforms state-of-the-art FL algorithms with < 6% of communication costs when given the limited amount of local training data.

----

## [770] Federated Learning with Data-Agnostic Distribution Fusion

**Authors**: *Jian-Hui Duan, Wenzhong Li, Derun Zou, Ruichen Li, Sanglu Lu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00780](https://doi.org/10.1109/CVPR52729.2023.00780)

**Abstract**:

Federated learning has emerged as a promising distributed machine learning paradigm to preserve data privacy. One of the fundamental challenges of federated learning is that data samples across clients are usually not independent and identically distributed (non-IID), leading to slow convergence and severe performance drop of the aggregated global model. To facilitate model aggregation on non-IID data, it is desirable to infer the unknown global distributions without violating privacy protection policy. In this paper, we propose a novel data-agnostic distribution fusion based model aggregation method called FedFusion to optimize federated learning with non-IID local datasets, based on which the heterogeneous clients' data distributions can be represented by a global distribution of several virtual fusion components with different parameters and weights. We develop a Variational AutoEncoder (VAE) method to learn the optimal parameters of the distribution fusion components based on limited statistical information extracted from the local models, and apply the derived distribution fusion model to optimize federated model aggregation with non-IID data. Extensive experiments based on various federated learning scenarios with real-world datasets show that FedFusion achieves significant performance improvement compared to the state-of-the-art.

----

## [771] CaPriDe Learning: Confidential and Private Decentralized Learning Based on Encryption-Friendly Distillation Loss

**Authors**: *Nurbek Tastan, Karthik Nandakumar*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00781](https://doi.org/10.1109/CVPR52729.2023.00781)

**Abstract**:

Large volumes of data required to train accurate deep neural networks (DNNs) are seldom available with any single entity. Often, privacy concerns prevent entities from sharing data with each other or with a third-party learning service provider. While crosssilo federated learning (FL) allows collaborative learning of large DNNs without sharing the data itself, most existing cross-silo FL algorithms have an unacceptable utility-privacy trade-off. In this work, we propose a framework called Confidential and Private Decentralized (CaPriDe) learning, which optimally leverages the power of fully homomorphic encryption (FHE) to enable collaborative learning without compromising on the confidentiality and privacy of data. In CaPridDe learning, participating entities release their private data in an encrypted form allowing other participants to perform inference in the encrypted domain. The crux of CaPriDe learning is mutual knowledge distillation between multiple local models through a novel distillation loss, which is an approximation of the Kullback-Leibler (KL) divergence between the local predictions and encrypted inferences of other participants on the same data that can be computed in the encrypted domain. Extensive experiments on three datasets show that CaPriDe learning can improve the accuracy of local models without any central coordination, provide strong guarantees of data confidentiality and privacy, and has the ability to handle statistical heterogeneity. Constraints on the model architecture (arising from the need to be FHE-friendly), limited scalability, and computational complexity of encrypted domain inference are the main limitations of the proposed approach. The code can be found at https://github.com/tnurbek/capride-learning.

----

## [772] Multi-view Adversarial Discriminator: Mine the Non-causal Factors for Object Detection in Unseen Domains

**Authors**: *Mingjun Xu, Lingyun Qin, Weijie Chen, Shiliang Pu, Lei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00783](https://doi.org/10.1109/CVPR52729.2023.00783)

**Abstract**:

Domain shift degrades the performance of object detection models in practical applications. To alleviate the influence of domain shift, plenty of previous work try to decouple and learn the domain-invariant (common) features from source domains via domain adversarial learning (DAL). However, inspired by causal mechanisms, we find that previous methods ignore the implicit insignificant non-causal factors hidden in the common features. This is mainly due to the single-view nature of DAL. In this work, we present an idea to remove non-causal factors from common features by multi-view adversarial training on source domains, because we observe that such insignificant non-causal factors may still be significant in other latent spaces (views) due to the multi-mode structure of data. To summarize, we propose a Multi-view Adversarial Discriminator (MAD) based domain generalization model, consisting of a Spurious Correlations Generator (SCG) that increases the diversity of source domain by random augmentation and a Multi-View Domain Classifier (MVDC) that maps features to multiple latent spaces, such that the non-causal factors are removed and the domain-invariant features are purified. Extensive experiments on six benchmarks show our MAD obtains state-of-the-art performance.

----

## [773] Single Image Backdoor Inversion via Robust Smoothed Classifiers

**Authors**: *Mingjie Sun, Zico Kolter*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00784](https://doi.org/10.1109/CVPR52729.2023.00784)

**Abstract**:

Backdoor inversion, the process of finding a backdoor “trigger” inserted into a machine learning model, has become the pillar of many backdoor detection and defense methods. Previous works on backdoor inversion often recover the backdoor through an optimization process to flip a support set of clean images into the target class. However, it is rarely studied and understood how large this support set should be to recover a successful backdoor. In this work, we show that one can reliably recover the backdoor trigger with as few as a single image. Specifically, we propose the SmoothInv method, which first constructs a robust smoothed version of the backdoored classifier and then performs guided image synthesis towards the target class to reveal the backdoor pattern. SmoothInv requires neither an explicit modeling of the backdoor via a mask variable, nor any complex regularization schemes, which has become the standard practice in backdoor inversion methods. We perform both quantitaive and qualitative study on backdoored classifiers from previous published backdoor attacks. We demonstrate that compared to existing methods, SmoothInv is able to recover successful backdoors from single images, while maintaining high fidelity to the original backdoor. We also show how we identify the target backdoored class from the backdoored classifier. Last, we propose and analyze two countermeasures to our approach and show that SmoothInv remains robust in the face of an adaptive attacker. Our code is available at https://github.com/locuslab/smoothinv.

----

## [774] Effective Ambiguity Attack Against Passport-based DNN Intellectual Property Protection Schemes through Fully Connected Layer Substitution

**Authors**: *Yiming Chen, Jinyu Tian, Xiangyu Chen, Jiantao Zhou*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00785](https://doi.org/10.1109/CVPR52729.2023.00785)

**Abstract**:

Since training a deep neural network (DNN) is costly, the well-trained deep models can be regarded as valuable intellectual property (IP) assets. The IP protection associated with deep models has been receiving increasing attentions in recent years. Passport-based method, which replaces normalization layers with passport layers, has been one of the few protection solutions that are claimed to be secure against advanced attacks. In this work, we tackle the issue of evaluating the security of passport-based IP protection methods. We propose a novel and effective ambiguity attack against passport-based method, capable of successfully forging multiple valid passports with a small training dataset. This is accomplished by inserting a specially designed accessory block ahead of the passport parameters. Using less than 10% of training data, with the forged passport, the model exhibits almost indistinguishable performance difference (less than 2%) compared with that of the authorized passport. In addition, it is shown that our attack strategy can be readily generalized to attack other IP protection methods based on watermark embedding. Directions for potential remedy solutions are also given.

----

## [775] Color Backdoor: A Robust Poisoning Attack in Color Space

**Authors**: *Wenbo Jiang, Hongwei Li, Guowen Xu, Tianwei Zhang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00786](https://doi.org/10.1109/CVPR52729.2023.00786)

**Abstract**:

Backdoor attacks against neural networks have been intensively investigated, where the adversary compromises the integrity of the victim model, causing it to make wrong predictions for inference samples containing a specific trigger. To make the trigger more imperceptible and human-unnoticeable, a variety of stealthy backdoor attacks have been proposed, some works employ imperceptible perturbations as the backdoor triggers, which restrict the pixel differences of the triggered image and clean image. Some works use special image styles (e.g., reflection, Instagram filter) as the backdoor triggers. However, these attacks sacrifice the robustness, and can be easily defeated by common preprocessing-based defenses. This paper presents a novel color backdoor attack, which can exhibit robustness and stealthiness at the same time. The key insight of our attack is to apply a uniform color space shift for all pixels as the trigger. This global feature is robust to image transformation operations and the triggered samples maintain natural-looking. To find the optimal trigger, we first define naturalness restrictions through the metrics of PSNR, SSIM and LPIPS. Then we employ the Particle Swarm Optimization (PSO) algorithm to searchfor the optimal trigger that can achieve high attack effectiveness and robustness while satisfying the restrictions. Extensive experiments demonstrate the superiority of PSO and the robustness of color backdoor against different main-stream backdoor defenses.

----

## [776] Adversarially Robust Neural Architecture Search for Graph Neural Networks

**Authors**: *Beini Xie, Heng Chang, Ziwei Zhang, Xin Wang, Daixin Wang, Zhiqiang Zhang, Rex Ying, Wenwu Zhu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00787](https://doi.org/10.1109/CVPR52729.2023.00787)

**Abstract**:

Graph Neural Networks (GNNs) obtain tremendous success in modeling relational data. Still, they are prone to adversarial attacks, which are massive threats to applying GNNs to risk-sensitive domains. Existing defensive methods neither guarantee performance facing new data/tasks or adversarial attacks nor provide insights to understand GNN robustness from an architectural perspective. Neural Architecture Search (NAS) has the potential to solve this problem by automating GNN architecture designs. Nevertheless, current graph NAS approaches lack robust design and are vulnerable to adversarial attacks. To tackle these challenges, we propose a novel Robust Neural Architecture search framework for GNNs (G-RNA). Specifically, we design a robust search space for the message-passing mechanism by adding graph structure mask operations into the search space, which comprises various defensive operation candidates and allows us to search for defensive GNNs. Furthermore, we define a robustness metric to guide the search procedure, which helps to filter robust architectures. In this way, G-RNA helps understand GNN robustness from an architectural perspective and effectively searches for optimal adversarial robust GNNs. Extensive experimental results on benchmark datasets show that G-RNA significantly outperforms manually designed robust GNNs and vanilla graph NAS baselines by 12.1% to 23.4% under adversarial attacks.

----

## [777] Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks

**Authors**: *Anqi Zhao, Tong Chu, Yahao Liu, Wen Li, Jingjing Li, Lixin Duan*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00788](https://doi.org/10.1109/CVPR52729.2023.00788)

**Abstract**:

In this work, we study the black-box targeted attack problem from the model discrepancy perspective. On the theoretical side, we present a generalization error bound for black-box targeted attacks, which gives a rigorous theoretical analysis for guaranteeing the success of the attack. We reveal that the attack error on a target model mainly depends on empirical attack error on the substitute model and the maximum model discrepancy among substitute models. On the algorithmic side, we derive a new algorithm for black-box targeted attacks based on our theoretical analysis, in which we additionally minimize the maximum model discrepancy (M3D) of the substitute models when training the generator to generate adversarial examples. In this way, our model is capable of crafting highly transferable adversarial examples that are robust to the model variation, thus improving the success rate for attacking the black-box model. We conduct extensive experiments on the ImageNet dataset with different classification models, and our proposed approach outperforms existing state-of-the-art methods by a significant margin. The code will be available at https://github.com/Asteriajojo/M3D.

----

## [778] StyLess: Boosting the Transferability of Adversarial Examples

**Authors**: *Kaisheng Liang, Bin Xiao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00789](https://doi.org/10.1109/CVPR52729.2023.00789)

**Abstract**:

Adversarial attacks can mislead deep neural networks (DNNs) by adding imperceptible perturbations to benign examples. The attack transferability enables adversarial examples to attack blackbox DNNs with unknown architectures or parameters, which poses threats to many realworld applications. We find that existing transferable attacks do not distinguish between style and content features during optimization, limiting their attack transferability. To improve attack transferability, we propose a novel attack method called style-less perturbation (StyLess). Specifically, instead of using a vanilla network as the surrogate model, we advocate using stylized networks, which encode different style features by perturbing an adaptive instance normalization. Our method can prevent adversarial examples from using non-robust style features and help generate transferable perturbations. Comprehensive experiments show that our method can significantly improve the transferability of adversarial examples. Furthermore, our approach is generic and can outperform state-of-the-art transferable attacks when combined with other attack techniques.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at https://github.com/uhiu/StyLess

----

## [779] Improving the Transferability of Adversarial Samples by Path-Augmented Method

**Authors**: *Jianping Zhang, Jen-tse Huang, Wenxuan Wang, Yichen Li, Weibin Wu, Xiaosen Wang, Yuxin Su, Michael R. Lyu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00790](https://doi.org/10.1109/CVPR52729.2023.00790)

**Abstract**:

Deep neural networks have achieved unprecedented success on diverse vision tasks. However, they are vulnerable to adversarial noise that is imperceptible to humans. This phenomenon negatively affects their deployment in real-world scenarios, especially security-related ones. To evaluate the robustness of a target model in practice, transfer-based attacks craft adversarial samples with a local model and have attracted increasing attention from researchers due to their high efficiency. The state-of-the-art transfer-based attacks are generally based on data augmentation, which typically augments multiple training images from a linear path when learning adversarial samples. However, such methods selected the image augmentation path heuristically and may augment images that are semantics-inconsistent with the target images, which harms the transferability of the generated adversarial samples. To overcome the pitfall, we propose the Path-Augmented Method (PAM). Specifically, PAM first constructs a candidate augmentation path pool. It then settles the employed augmentation paths during adversarial sample generation with greedy search. Furthermore, to avoid augmenting semantics-inconsistent images, we train a Semantics Predictor (SP) to constrain the length of the augmentation path. Extensive experiments confirm that PAM can achieve an improvement of over 4.8% on average compared with the state-of-the-art baselines in terms of the attack success rates.

----

## [780] Feature Separation and Recalibration for Adversarial Robustness

**Authors**: *Woo Jae Kim, Yoonki Cho, Junsik Jung, Sung-Eui Yoon*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00791](https://doi.org/10.1109/CVPR52729.2023.00791)

**Abstract**:

Deep neural networks are susceptible to adversarial attacks due to the accumulation of perturbations in the feature level, and numerous works have boosted model robustness by deactivating the non-robust feature activations that cause model mispredictions. However, we claim that these malicious activations still contain discriminative cues and that with recalibration, they can capture additional useful information for correct model predictions. To this end, we propose a novel, easy-to-plugin approach named Feature Separation and Recalibration (FSR) that recalibrates the malicious, non-robust activations for more robust feature maps through Separation and Recalibration. The Separation part disentangles the input feature map into the robust feature with activations that help the model make correct predictions and the non-robust feature with activations that are responsible for model mispredictions upon adversarial attack. The Recalibration part then adjusts the non-robust activations to restore the potentially useful cues for model predictions. Extensive experiments verify the superiority of FSR compared to traditional deactivation techniques and demonstrate that it improves the robustness of existing adversarial training methods by up to 8.57% with small computational overhead. Codes are available at https://github.com/wkim97/FSR

----

## [781] CFA: Class-Wise Calibrated Fair Adversarial Training

**Authors**: *Zeming Wei, Yifei Wang, Yiwen Guo, Yisen Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00792](https://doi.org/10.1109/CVPR52729.2023.00792)

**Abstract**:

Adversarial training has been widely acknowledged as the most effective method to improve the adversarial robustness against adversarial examples for Deep Neural Networks (DNNs). So far, most existing works focus on en-hancing the overall model robustness, treating each class equally in both the training and testing phases. Although revealing the disparity in robustness among classes, few works try to make adversarial training fair at the class level without sacrificing overall robustness. In this paper, we are the first to theoretically and empirically investigate the preference of different classes for adversarial configu-rations, including perturbation margin, regularization, and weight averaging. Motivated by this, we further propose a Class-wise calibrated Fair Adversarial training frame-work, named CFA, which customizes specific training con-figurations for each class automatically. Experiments on benchmark datasets demonstrate that our proposed CFA can improve both overall robustness and fairness notably over other state-of-the-art methods. Code is available at https://github.com/PKU-ML/CFA.

----

## [782] Revisiting Residual Networks for Adversarial Robustness

**Authors**: *Shihua Huang, Zhichao Lu, Kalyanmoy Deb, Vishnu Naresh Boddeti*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00793](https://doi.org/10.1109/CVPR52729.2023.00793)

**Abstract**:

Efforts to improve the adversarial robustness of convolutional neural networks have primarily focused on developing more effective adversarial training methods. In contrast, little attention was devoted to analyzing the role of architectural elements (e.g., topology, depth, and width) on adversarial robustness. This paper seeks to bridge this gap and present a holistic study on the impact of architectural design on adversarial robustness. We focus on residual networks and consider architecture design at the block level as well as at the network scaling level. In both cases, we first derive insights through systematic experiments. Then we design a robust residual block, dubbed RobustResBlock, and a compound scaling rule, dubbed RobustScaling, to distribute depth and width at the desired FLOP count. Finally, we combine RobustResBlock and RobustScaling and present a portfolio of adversarially robust residual networks, RobustResNets, spanning a broad spectrum of model capacities. Experimental validation across multiple datasets and adversarial attacks demonstrate that RobustResNets consistently outperform both the standard WRNs and other existing robust architectures, achieving state-of-the-art AutoAttack robust accuracy 63.7% with 500K external data while being 2× more compact in terms of parameters. Code is available at this URL.

----

## [783] Privacy-preserving Adversarial Facial Features

**Authors**: *Zhibo Wang, He Wang, Shuaifan Jin, Wenwen Zhang, Jiahui Hu, Yan Wang, Peng Sun, Wei Yuan, Kaixin Liu, Kui Ren*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00794](https://doi.org/10.1109/CVPR52729.2023.00794)

**Abstract**:

Face recognition service providers protect face privacy by extracting compact and discriminative facial features (representations) from images, and storing the facial features for real-time recognition. However, such features can still be exploited to recover the appearance of the original face by building a reconstruction network. Although sev-eral privacy-preserving methods have been proposed, the enhancement offace privacy protection is at the expense of accuracy degradation. In this paper, we propose an adver-sarial features-based face privacy protection (AdvFace) approach to generate privacy-preserving adversarial features, which can disrupt the mapping from adversarial features to facial images to defend against reconstruction attacks. To this end, we design a shadow model which simulates the attackers' behavior to capture the mapping function from facial features to images and generate adversarial la-tent noise to disrupt the mapping. The adversarial features rather than the original features are stored in the server's database to prevent leaked features from exposing facial information. Moreover, the AdvFace requires no changes to the face recognition network and can be implemented as a privacy-enhancing plugin in deployed face recognition systems. Extensive experimental results demonstrate that Adv Face outperforms the state-of-the-art face privacy-preserving methods in defending against reconstruction at-tacks while maintaining face recognition accuracy.

----

## [784] Edge-aware Regional Message Passing Controller for Image Forgery Localization

**Authors**: *Dong Li, Jiaying Zhu, Menglu Wang, Jiawei Liu, Xueyang Fu, Zheng-Jun Zha*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00795](https://doi.org/10.1109/CVPR52729.2023.00795)

**Abstract**:

Digital image authenticity has promoted research on image forgery localization. Although deep learning-based methods achieve remarkable progress, most of them usually suffer from severe feature coupling between the forged and authentic regions. In this work, we propose a two-step Edge-aware Regional Message Passing Controlling strategy to address the above issue. Specifically, the first step is to account for fully exploiting the edge information. It consists of two core designs: context-enhanced graph construction and threshold-adaptive differentiable binarization edge algorithm. The former assembles the global semantic information to distinguish the features between the forged and authentic regions, while the latter stands on the output of the former to provide the learnable edges. In the second step, guided by the learnable edges, a region message passing controller is devised to weaken the message passing between the forged and authentic regions. In this way, our ERMPC is capable of explicitly modeling the inconsistency between the forged and authentic regions and enabling it to perform well on refined forged images. Extensive experiments on several challenging benchmarks show that our method is superior to state-of-the-art image forgery localization methods qualitatively and quantitatively.

----

## [785] Swept-Angle Synthetic Wavelength Interferometry

**Authors**: *Alankar Kotwal, Anat Levin, Ioannis Gkioulekas*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00796](https://doi.org/10.1109/CVPR52729.2023.00796)

**Abstract**:

We present a new imaging technique, swept-angle synthetic wavelength interferometry, for full-field micron-scale 3D sensing. As in conventional synthetic wavelength interferometry, our technique uses light consisting of two narrowly-separated optical wavelengths, resulting in per-pixel inter-ferometric measurements whose phase encodes scene depth. Our technique additionally uses a new type of light source that, by emulating spatially-incoherent illumination, makes interferometric measurements insensitive to aberrations and (sub)surface scattering, effects that corrupt phase measurements. The resulting technique combines the robustness to such corruptions of scanning interferometric setups, with the speed of full-field interferometric setups. Overall, our technique can recover full-frame depth at a lateral and axial resolution of 5 µm, at frame rates of 5 Hz, even under strong ambient light. We build an experimental prototype, and use it to demonstrate these capabilities by scanning a variety of objects, including objects representative of applications in inspection and fabrication, and objects that contain challenging light scattering effects.

----

## [786] RefSR-NeRF: Towards High Fidelity and Super Resolution View Synthesis

**Authors**: *Xudong Huang, Wei Li, Jie Hu, Hanting Chen, Yunhe Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00797](https://doi.org/10.1109/CVPR52729.2023.00797)

**Abstract**:

We present Reference-guided Super-Resolution Neural Radiance Field (RefSR-NeRF) that extends NeRF to super resolution and photorealistic novel view synthesis. Despite NeRF's extraordinary success in the neural rendering field, it suffers from blur in high resolution rendering because its inherent multilayer perceptron struggles to learn high frequency details and incurs a computational explosion as resolution increases. Therefore, we propose RefSR-NeRF, an end-to-end framework that first learns a low resolution NeRF representation, and then reconstructs the high frequency details with the help of a high resolution reference image. We observe that simply introducing the pre-trained models from the literature tends to produce unsatisfied artifacts due to the divergence in the degradation model. To this end, we design a novel lightweight RefSR model to learn the inverse degradation process from NeRF renderings to target HR ones. Extensive experiments on multiple benchmarks demonstrate that our method exhibits an impressive trade-off among rendering quality, speed, and memory usage, outperforming or on par with NeRF and its variants while being 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$52\times$</tex>
 speedup with minor extra memory usage. Code will be available at: Mindspore and Pytorch

----

## [787] FreeNeRF: Improving Few-Shot Neural Rendering with Free Frequency Regularization

**Authors**: *Jiawei Yang, Marco Pavone, Yue Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00798](https://doi.org/10.1109/CVPR52729.2023.00798)

**Abstract**:

Novel view synthesis with sparse inputs is a challenging problem for neural radiance fields (NeRF). Recent efforts alleviate this challenge by introducing external supervision, such as pre-trained models and extra depth signals, or by using non-trivial patch-based rendering. In this paper, we present Frequency regularized NeRF (FreeNeRF), a surprisingly simple baseline that outperforms previous methods with minimal modifications to plain NeRF. We analyze the key challenges in few-shot neural rendering and find that frequency plays an important role in NeRF's training. Based on this analysis, we propose two regularization terms: one to regularize the frequency range of NeRF's inputs, and the other to penalize the near-camera density fields. Both techniques are “free lunches” that come at no additional computational cost. We demonstrate that even with just one line of code change, the original NeRF can achieve similar performance to other complicated methods in the few-shot setting. FreeNeRF achieves state-of-the-art performance across diverse datasets, including Blender, DTU, and LLFF. We hope that this simple baseline will motivate a rethinking of the fundamental role of frequency in NeRF's training, under both the low-data regime and beyond. This project is released at FreeNeRF.

----

## [788] Local-to-Global Registration for Bundle-Adjusting Neural Radiance Fields

**Authors**: *Yue Chen, Xingyu Chen, Xuan Wang, Qi Zhang, Yu Guo, Ying Shan, Fei Wang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00799](https://doi.org/10.1109/CVPR52729.2023.00799)

**Abstract**:

Neural Radiance Fields (NeRF) have achieved photorealistic novel views synthesis; however, the requirement of accurate camera poses limits its application. Despite analysis-by-synthesis extensions for jointly learning neural3D representations and registering camera frames exist, they are susceptible to suboptimal solutions if poorly initialized. We propose L2G-NeRF, a Local-to-Global registration method for bundle-adjusting Neural Radiance Fields: first, a pixel-wise flexible alignment, followed by a framewise constrained parametric alignment. Pixel-wise local alignment is learned in an unsupervised way via a deep network which optimizes photometric reconstruction errors. framewise global alignment is performed using differentiable parameter estimation solvers on the pixel-wise correspondences to find a global transformation. Experiments on synthetic and real-world data show that our method outperforms the current state-of-the-art in terms of high-fidelity reconstruction and resolving large camera pose misalignment. Our module is an easy-to-use plugin that can be applied to NeRF variants and other neural field applications. The Code and supplementary materials are available at https://rover-xingyu.github.io/L2G-NeRF/.

----

## [789] Nerflets: Local Radiance Fields for Efficient Structure-Aware 3D Scene Representation from 2D Supervision

**Authors**: *Xiaoshuai Zhang, Abhijit Kundu, Thomas A. Funkhouser, Leonidas J. Guibas, Hao Su, Kyle Genova*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00800](https://doi.org/10.1109/CVPR52729.2023.00800)

**Abstract**:

We address efficient and structure-aware 3D scene representation from images. Nerflets are our key contribution-a set of local neural radiance fields that together represent a scene. Each nerflet maintains its own spatial position, orientation, and extent, within which it contributes to panoptic, density, and radiance reconstructions. By leveraging only photometric and inferred panoptic image supervision, we can directly and jointly optimize the parameters of a set of nerflets so as to form a decomposed representation of the scene, where each object instance is represented by a group of nerflets. During experiments with indoor and outdoor environments, we find that nerflets: (1) fit and approximate the scene more efficiently than traditional global NeRFs, (2) allow the extraction of panoptic and photometric renderings from arbitrary views, and (3) enable tasks rare for NeRFs, such as 3D panoptic segmentation and interactive editing. Our project page.

----

## [790] NeRF-DS: Neural Radiance Fields for Dynamic Specular Objects

**Authors**: *Zhiwen Yan, Chen Li, Gim Hee Lee*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00801](https://doi.org/10.1109/CVPR52729.2023.00801)

**Abstract**:

Dynamic Neural Radiance Field (NeRF) is a powerful algorithm capable of rendering photo-realistic novel view images from a monocular RGB video of a dynamic scene. Although it warps moving points across frames from the observation spaces to a common canonical space for rendering, dynamic NeRF does not model the change of the reflected color during the warping. As a result, this approach often fails drastically on challenging specular objects in motion. We address this limitation by reformulating the neural radiance field function to be conditioned on surface position and orientation in the observation space. This allows the specular surface at different poses to keep the different reflected colors when mapped to the common canonical space. Additionally, we add the mask of moving objects to guide the deformation field. As the specular surface changes color during motion, the mask mitigates the problem of failure to find temporal correspondences with only RGB supervision. We evaluate our model based on the novel view synthesis quality with a self-collected dataset of different moving specular objects in realistic environments. The experimental results demonstrate that our method significantly improves the reconstruction quality of moving specular objects from monocular RGB videos compared to the existing NeRF models. Our code and data are available at the project website
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/JokerYan/NeRF-DS.

----

## [791] Grid-guided Neural Radiance Fields for Large Urban Scenes

**Authors**: *Linning Xu, Yuanbo Xiangli, Sida Peng, Xingang Pan, Nanxuan Zhao, Christian Theobalt, Bo Dai, Dahua Lin*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00802](https://doi.org/10.1109/CVPR52729.2023.00802)

**Abstract**:

Purely MLP-based neural radiance fields (NeRF-based methods) often suffer from underfitting with blurred renderings on large-scale scenes due to limited model capacity. Recent approaches propose to geographically divide the scene and adopt multiple sub-NeRFs to model each region individually, leading to linear scale-up in training costs and the number of sub-NeRFs as the scene expands. An alternative solution is to use a feature grid representation, which is computationally efficient and can naturally scale to a large scene with increased grid resolutions. However, the feature grid tends to be less constrained and often reaches suboptimal solutions, producing noisy artifacts in renderings, especially in regions with complex geometry and texture. In this work, we present a new framework that realizes high-fidelity rendering on large urban scenes while being computationally efficient. We propose to use a compact multi-resolution ground feature plane representation to coarsely capture the scene, and complement it with positional encoding inputs through another NeRF branch for rendering in a joint learning fashion. We show that such an integration can utilize the advantages of two alternative solutions: a light-weighted NeRF is sufficient, under the guidance of the feature grid representation, to render photorealistic novel views with fine details; and the jointly optimized ground feature planes, can meanwhile gain further refinements, forming a more accurate and compact feature space and output much more natural rendering results.

----

## [792] Learning Neural Duplex Radiance Fields for Real-Time View Synthesis

**Authors**: *Ziyu Wan, Christian Richardt, Aljaz Bozic, Chao Li, Vijay Rengarajan, Seonghyeon Nam, Xiaoyu Xiang, Tuotuo Li, Bo Zhu, Rakesh Ranjan, Jing Liao*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00803](https://doi.org/10.1109/CVPR52729.2023.00803)

**Abstract**:

Neural radiance fields (NeRFs) enable novel-view synthesis with unprecedented visual quality. However, to render photorealistic images, NeRFs require hundreds of deep multilayer perceptron (MLP) evaluations - for each pixel. This is prohibitively expensive and makes realtime rendering infeasible, even on powerful modern GPUs. In this paper, we propose a novel approach to distill and bake NeRFs into highly efficient mesh-based neural representations that are fully compatible with the massively parallel graphics rendering pipeline. We represent scenes as neural radiance features encoded on a two-layer duplex mesh, which effectively over-comes the inherent inaccuracies in 3D surface reconstruction by learning the aggregated radiance information from a reliable interval of ray-surface intersections. To exploit local geometric relationships of nearby pixels, we leverage screen-space convolutions instead of the MLPs used in NeRFs to achieve high-quality appearance. Finally, the performance of the whole framework is further boosted by a novel multi-view distillation optimization strategy. We demonstrate the effectiveness and superiority of our approach via extensive experiments on a range of standard datasets.

----

## [793] EditableNeRF: Editing Topologically Varying Neural Radiance Fields by Key Points

**Authors**: *Chengwei Zheng, Wenbin Lin, Feng Xu*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00804](https://doi.org/10.1109/CVPR52729.2023.00804)

**Abstract**:

Neural radiance fields (NeRF) achieve highly photorealistic novel-view synthesis, but it's a challenging problem to edit the scenes modeled by NeRF-based methods, especially for dynamic scenes. We propose editable neural radiance fields that enable end-users to easily edit dynamic scenes and even support topological changes. Input with an image sequence from a single camera, our network is trained fully automatically and models topologically varying dynamics using our picked-out surface key points. Then end-users can edit the scene by easily dragging the key points to desired new positions. To achieve this, we propose a scene analysis method to detect and initialize key points by considering the dynamics in the scene, and a weighted key points strategy to model topologically varying dynamics by joint key points and weights optimization. Our method supports intuitive multi-dimensional (up to 3D) editing and can generate novel scenes that are unseen in the input sequence. Experiments demonstrate that our method achieves high-quality editing on various dynamic scenes and outperforms the state-of-the-art. Our code and captured data are available at https://chengwei-zheng.github.io/EditableNeRF/

----

## [794] Real-Time Neural Light Field on Mobile Devices

**Authors**: *Junli Cao, Huan Wang, Pavlo Chemerys, Vladislav Shakhrai, Ju Hu, Yun Fu, Denys Makoviichuk, Sergey Tulyakov, Jian Ren*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00805](https://doi.org/10.1109/CVPR52729.2023.00805)

**Abstract**:

Recent efforts in Neural Rendering Fields (NeRF) have shown impressive results on novel view synthesis by utilizing implicit neural representation to represent 3D scenes. Due to the process of volumetric rendering, the inference speed for NeRF is extremely slow, limiting the application scenarios of utilizing NeRF on resource-constrained hardware, such as mobile devices. Many works have been conducted to reduce the latency of running NeRF models. However, most of them still require high-end GPU for acceleration or extra storage memory, which is all unavailable on mobile devices. Another emerging direction utilizes the neural light field (NeLF) for speedup, as only one forward pass is performed on a ray to predict the pixel color. Nevertheless, to reach a similar rendering quality as NeRF, the network in NeLF is designed with intensive computation, which is not mobile-friendly. In this work, we propose an efficient network that runs in real-time on mobile devices for neural rendering. We follow the setting of NeLF to train our network. Unlike existing works, we introduce a novel network architecture that runs efficiently on mobile devices with low latency and small size, i.e., saving 15 x ~ 24 x storage compared with MobileNeRF. Our model achieves high-resolution generation while maintaining real-time inference for both synthetic and real-world scenes on mobile devices, e.g., 18.04ms (iPhone 13) for rendering one 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1008 \times 756$</tex>
 image of real 3D scenes. Additionally, we achieve similar image quality as NeRF and better quality than MobileNeRF (PSNR 26.15 vs. 25.91 on the real-world forward-facing dataset)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
More demo examples in our Webpage..

----

## [795] StyleRF: Zero-Shot 3D Style Transfer of Neural Radiance Fields

**Authors**: *Kunhao Liu, Fangneng Zhan, Yiwen Chen, Jiahui Zhang, Yingchen Yu, Abdulmotaleb El-Saddik, Shijian Lu, Eric P. Xing*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00806](https://doi.org/10.1109/CVPR52729.2023.00806)

**Abstract**:

3D style transfer aims to render stylized novel views of a 3D scene with multiview consistency. However, most existing work suffers from a three-way dilemma over accurate geometry reconstruction, high-quality stylization, and being generalizable to arbitrary new styles. We propose StyleRF (Style Radiance Fields), an innovative 3D style transfer technique that resolves the three-way dilemma by performing style transformation within the feature space of a radiance field. StyleRF employs an explicit grid of high-level features to represent 3D scenes, with which highfidelity geometry can be reliably restored via volume rendering. In addition, it transforms the grid features according to the reference style which directly leads to high-quality zero-shot style transfer. StyleRF consists of two innovative designs. The first is sampling-invariant content transformation that makes the transformation invariant to the holistic statistics of the sampled 3D points and accordingly ensures multi-view consistency. The second is deferred style transformation of 2D feature maps which is equivalent to the transformation of 3D points but greatly reduces memory footprint without degrading multi-view consistency. Extensive experiments show that StyleRF achieves superior 3D stylization quality with precise geometry reconstruction and it can generalize to various new styles in a zero-shot manner. Project website: https://kunhao-liu.github.io/StyleRF/

----

## [796] Point2Pix: Photo-Realistic Point Cloud Rendering via Neural Radiance Fields

**Authors**: *Tao Hu, Xiaogang Xu, Shu Liu, Jiaya Jia*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00807](https://doi.org/10.1109/CVPR52729.2023.00807)

**Abstract**:

Synthesizing photo-realistic images from a point cloud is challenging because of the sparsity of point cloud representation. Recent Neural Radiance Fields and extensions are proposed to synthesize realistic images from 2D input. In this paper, we present Point2Pix as a novel point renderer to link the 3D sparse point clouds with 2D dense image pixels. Taking advantage of the point cloud 3D prior and NeRF rendering pipeline, our method can synthesize high-quality images from colored point clouds, generally for novel indoor scenes. To improve the efficiency of ray sampling, we propose point-guided sampling, which focuses on valid samples. Also, we present Point Encoding to build Multiscale Radiance Fields that provide discriminative 3D point features. Finally, we propose Fusion Encoding to efficiently synthesize high-quality images. Extensive experiments on the ScanNet and ArkitScenes datasets demonstrate the effectiveness and generalization.

----

## [797] Pointersect: Neural Rendering with Cloud-Ray Intersection

**Authors**: *Jen-Hao Rick Chang, Wei-Yu Chen, Anurag Ranjan, Kwang Moo Yi, Oncel Tuzel*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00808](https://doi.org/10.1109/CVPR52729.2023.00808)

**Abstract**:

We propose a novel method that renders point clouds as if they are surfaces. The proposed method is differentiable and requires no scene-specific optimization. This unique capability enables, out-of-the-box, surface normal estimation, rendering room-scale point clouds, inverse rendering, and ray tracing with global illumination. Unlike existing work that focuses on converting point clouds to other representations—e.g., surfaces or implicit functions—our key idea is to directly infer the intersection of a light ray with the underlying surface represented by the given point cloud. Specifically, we train a set transformer that, given a small number of local neighbor points along a light ray, provides the intersection point, the surface normal, and the material blending weights, which are used to render the outcome of this light ray. Localizing the problem into small neighborhoods enables us to train a model with only 48 meshes and apply it to unseen point clouds. Our model achieves higher estimation accuracy than state-of-the-art surface reconstruction and point-cloud rendering methods on three test sets. When applied to room-scale point clouds, without any scene-specific optimization, the model achieves competitive quality with the state-of-the-art novel-view rendering methods. Moreover, we demonstrate ability to render and manipulate Lidar-scanned point clouds such as lighting control and object insertion.

----

## [798] Neural Fields Meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes

**Authors**: *Zian Wang, Tianchang Shen, Jun Gao, Shengyu Huang, Jacob Munkberg, Jon Hasselgren, Zan Gojcic, Wenzheng Chen, Sanja Fidler*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00809](https://doi.org/10.1109/CVPR52729.2023.00809)

**Abstract**:

Reconstruction and intrinsic decomposition of scenes from captured imagery would enable many applications such as relighting and virtual object insertion. Recent NeRF based methods achieve impressive fidelity of 3D reconstruction, but bake the lighting and shadows into the radiance field, while mesh-based methods that facilitate intrinsic decomposition through differentiable rendering have not yet scaled to the complexity and scale of outdoor scenes. We present a novel inverse rendering framework for large urban scenes capable of jointly reconstructing the scene geometry, spatially-varying materials, and HDR lighting from a set of posed RGB images with optional depth. Specifically, we use a neural field to account for the primary rays, and use an explicit mesh (reconstructed from the underlying neural field) for modeling secondary rays that produce higher-order lighting effects such as cast shadows. By faithfully disentangling complex geometry and materials from lighting effects, our method enables photorealistic relighting with specular and shadow effects on several outdoor datasets. Moreover, it supports physics-based scene manipulations such as virtual object insertion with ray-traced shadow casting.

----

## [799] DANI-Net: Uncalibrated Photometric Stereo by Differentiable Shadow Handling, Anisotropic Reflectance Modeling, and Neural Inverse Rendering

**Authors**: *Zongrui Li, Qian Zheng, Boxin Shi, Gang Pan, Xudong Jiang*

**Conference**: *cvpr 2023*

**URL**: [https://doi.org/10.1109/CVPR52729.2023.00810](https://doi.org/10.1109/CVPR52729.2023.00810)

**Abstract**:

Uncalibrated photometric stereo (UPS) is challenging due to the inherent ambiguity brought by the unknown light. Although the ambiguity is alleviated on non-Lambertian objects, the problem is still difficult to solve for more general objects with complex shapes introducing irregular shadows and general materials with complex reflectance like anisotropic reflectance. To exploit cues from shadow and reflectance to solve UPS and improve performance on general materials, we propose DANI-Net, an inverse rendering framework with differentiable shadow handling and anisotropic reflectance modeling. Unlike most previous methods that use non-differentiable shadow maps and assume isotropic material, our network benefits from cues of shadow and anisotropic reflectance through two differentiable paths. Experiments on multiple real-world datasets demonstrate our superior and robust performance.

----



[Go to the previous page](CVPR-2023-list03.md)

[Go to the next page](CVPR-2023-list05.md)

[Go to the catalog section](README.md)