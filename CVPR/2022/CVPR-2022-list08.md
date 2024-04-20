## [1400] MUM: Mix Image Tiles and UnMix Feature Tiles for Semi-Supervised Object Detection

**Authors**: *Jongmok Kim, Jooyoung Jang, Seunghyeon Seo, Jisoo Jeong, Jongkeun Na, Nojun Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01411](https://doi.org/10.1109/CVPR52688.2022.01411)

**Abstract**:

Many recent semi-supervised learning (SSL) studies build teacher-student architecture and train the student net-work by the generated supervisory signal from the teacher. Data augmentation strategy plays a significant role in the SSL framework since it is hard to create a weak-strong aug-mented input pair without losing label information. Espe-cially when extending SSL to semi-supervised object de-tection (SSOD), many strong augmentation methodologies related to image geometry and interpolation-regularization are hard to utilize since they possibly hurt the location information of the bounding box in the object detection task. To address this, we introduce a simple yet effective data augmentation method, Mix/UnMix (MUM), which un-mixes feature tiles for the mixed image tiles for the SSOD framework. Our proposed method makes mixed input image tiles and reconstructs them in the feature space. Thus, MUM can enjoy the interpolation-regularization effect from non-interpolated pseudo-labels and successfully generate a meaningful weak-strong pair. Furthermore, MUM can be easily equipped on top of various SSOD methods. Exten-sive experiments on MS-COCO and PASCAL VOC datasets demonstrate the superiority of MUM by consistently im-proving the mAP performance over the baseline in all the tested SSOD benchmark protocols. The code is released at https.//github.com/JongMokKim/mix-unmix.

----

## [1401] Scale-Equivalent Distillation for Semi-Supervised Object Detection

**Authors**: *Qiushan Guo, Yao Mu, Jianyu Chen, Tianqi Wang, Yizhou Yu, Ping Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01412](https://doi.org/10.1109/CVPR52688.2022.01412)

**Abstract**:

Recent Semi-Supervised Object Detection (SS-OD) methods are mainly based on self-training, i.e., generating hard pseudo-labels by a teacher model on unlabeled data as supervisory signals. Although they achieved certain success, the limited labeled data in semi-supervised learning scales up the challenges of object detection. We analyze the challenges these methods meet with the empirical experiment results. We find that the massive False Negative samples and inferior localization precision lack consideration. Besides, the large variance of object sizes and class imbalance (i.e., the extreme ratio between back-ground and object) hinder the performance of prior arts. Further, we overcome these challenges by introducing a novel approach, Scale-Equivalent Distillation (SED), which is a simple yet effective end-to-end knowledge distillation framework robust to large object size variance and class imbalance. SED has several appealing benefits compared to the previous works. (1) SED imposes a consistency regularization to handle the large scale variance problem. (2) SED alleviates the noise problem from the False Negative samples and inferior localization precision. (3) A re-weighting strategy can implicitly screen the potential foreground regions of the unlabeled data to reduce the effect of class imbalance. Extensive experiments show that SED consistently outperforms the recent state-of-the-art methods on different datasets with significant margins. For example, it surpasses the supervised counterpart by more than 10 mAP when using 5% and 10% labeled data on MS-COCO.

----

## [1402] A Self-Supervised Descriptor for Image Copy Detection

**Authors**: *Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, Matthijs Douze*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01413](https://doi.org/10.1109/CVPR52688.2022.01413)

**Abstract**:

Image copy detection is an important task for content moderation. We introduce SSCD, a model that builds on a recent self-supervised contrastive training objective. We adapt this method to the copy detection task by changing the architecture and training objective, including a pooling operator from the instance matching literature, and adapting contrastive learning to augmentations that combine images. Our approach relies on an entropy regularization term, promoting consistent separation between descriptor vectors, and we demonstrate that this significantly improves copy detection accuracy. Our method produces a compact descriptor vector, suitable for real-world web scale applications. Statistical information from a background image distribution can be incorporated into the descriptor. On the recent DISC2021 benchmark, SSCD is shown to outperform both baseline copy detection models and self-supervised architectures designed for image classification by huge margins, in all settings. For example, SSCD out-performs SimCLR descriptors by 48% absolute. Code is available at https://github.com/facebookresearch/sscd-copy-detection.

----

## [1403] Self-Supervised Transformers for Unsupervised Object Discovery using Normalized Cut

**Authors**: *Yangtao Wang, Xi Shen, Shell Xu Hu, Yuan Yuan, James L. Crowley, Dominique Vaufreydaz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01414](https://doi.org/10.1109/CVPR52688.2022.01414)

**Abstract**:

Transformers trained with self-supervision using selfdistillation loss (DINO) have been shown to produce attention maps that highlight salient foreground objects. In this paper, we show a graph-based method that uses the selfsupervised transformer features to discover an object from an image. Visual tokens are viewed as nodes in a weighted graph with edges representing a connectivity score based on the similarity of tokens. Foreground objects can then be segmented using a normalized graph-cut to group self-similar regions. We solve the graph-cut problem using spectral clustering with generalized eigen-decomposition and show that the second smallest eigenvector provides a cutting solution since its absolute value indicates the likelihood that a token belongs to a foreground object. Despite its simplicity, this approach significantly boosts the performance of unsupervised object discovery: we improve over the recent state-of-the-art LOST by a margin of 6.9%, 8.1%, and 8.1% respectively on the VOC07, VOC12, and COCO20K. The performance can be further improved by adding a second stage class-agnostic detector (CAD). Our proposed method can be easily extended to unsupervised saliency detection and weakly supervised object detection. For unsupervised saliency detection, we improve IoU for 4.9%, 5.2%, 12.9% on ECSSD, DUTS, DUT-OMRON respectively compared to state-of-the-art. For weakly supervised object detection, we achieve competitive performance on CUB and ImageNet. Our code is available at: https://www.m-psi.fr/Papers/TokenCut2022/

----

## [1404] CAD: Co-Adapting Discriminative Features for Improved Few-Shot Classification

**Authors**: *Philip Chikontwe, Soopil Kim, Sang Hyun Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01415](https://doi.org/10.1109/CVPR52688.2022.01415)

**Abstract**:

Few-shot classification is a challenging problem that aims to learn a model that can adapt to unseen classes given a few labeled samples. Recent approaches pre-train a feature extractor, and then fine-tune for episodic metalearning. Other methods leverage spatial features to learn pixel-level correspondence while jointly training a classifier. However, results using such approaches show marginal improvements. In this paper, inspired by the transformer style self-attention mechanism, we propose a strategy to cross-attend and re-weight discriminative features for fewshot classification. Given a base representation of support and query images after global pooling, we introduce a single shared module that projects features and cross-attends in two aspects: (i) query to support, and (ii) support to query. The module computes attention scores between features to produce an attention pooled representation of features in the same class that is later added to the original representation followed by a projection head. This effectively re-weights features in both aspects (i & ii) to produce features that better facilitate improved metric-based metalearning. Extensive experiments on public benchmarks show our approach outperforms state-of-the-art methods by 3%~5%.

----

## [1405] Semi-Supervised Few-shot Learning via Multi-Factor Clustering

**Authors**: *Jie Ling, Lei Liao, Meng Yang, Jia Shuai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01416](https://doi.org/10.1109/CVPR52688.2022.01416)

**Abstract**:

The scarcity of labeled data and the problem of model overfitting have been the challenges in few-shot learning. Recently, semi-supervised few-shot learning has been developed to obtain pseudo-labels of unlabeled samples for expanding the support set. However, the relationship between unlabeled and labeled data is not well exploited in generating pseudo labels, the noise of which will di-rectly harm the model learning. In this paper, we propose a Clustering-based semi-supervised Few-Shot Learning (cluster-FSL) method to solve the above problems in image classification. By using multi-factor collaborative representation, a novel Multi-Factor Clustering (MFC) is designed to fuse the information of few-shot data distribution, which can generate soft and hard pseudo-labels for unlabeled samples based on labeled data. And we exploit the pseudo labels of unlabeled samples by MFC to expand the support set for obtaining more distribution information. Furthermore, robust data augmentation is used for support set in the fine-tuning phase to increase the labeled samples' diversity. We verified the validity of the cluster-FSL by comparing it with other few-shot learning methods on three popular benchmark datasets, miniImageNet, tieredImageNet, and CUB-200-2011. The ablation experiments further demonstrate that our MFC can effectively fuse distribution information of labeled samples and provide high-quality pseudo-labels. Our code is available at: https://gitlab.com/smartllvlab/cluster-fsl

----

## [1406] CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning

**Authors**: *Yue Fan, Dengxin Dai, Anna Kukleva, Bernt Schiele*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01417](https://doi.org/10.1109/CVPR52688.2022.01417)

**Abstract**:

Standard semi-supervised learning (SSL) using classbalanced datasets has shown great progress to leverage unlabeled data effectively. However, the more realistic setting of class-imbalanced data - called imbalanced SSL - is largely underexplored and standard SSL tends to under-perform. In this paper, we propose a novel co-learning framework (CoSSL), which decouples representation and classifier learning while coupling them closely. To handle the data imbalance, we devise Tail-class Feature Enhancement (TFE) for classifier learning. Furthermore, the current evaluation protocol for imbalanced SSL focuses only on balanced test sets, which has limited practicality in real-world scenarios. Therefore, we further conduct a comprehensive evaluation under various shifted test distributions. In experiments, we show that our approach outperforms other methods over a large range of shifted distributions, achieving state-of-the-art performance on benchmark datasets ranging from CIFAR-10, CIFAR-100, ImageNet, to Food-101. Our code will be made publicly available.

----

## [1407] Safe-Student for Safe Deep Semi-Supervised Learning with Unseen-Class Unlabeled Data

**Authors**: *Rundong He, Zhongyi Han, Xiankai Lu, Yilong Yin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01418](https://doi.org/10.1109/CVPR52688.2022.01418)

**Abstract**:

Deep semi-supervised learning (SSL) methods aim to take advantage of abundant unlabeled data to improve the algorithm performance. In this paper, we consider the problem of safe SSL scenario where unseen-class instances appear in the unlabeled data. This setting is essential and commonly appears in a variety of real applications. One intuitive solution is removing these unseen-class instances after detecting them during the SSL process. Nevertheless, the performance of unseen-class identification is limited by the small number of labeled data and ignoring the availability of unlabeled data. To take advantage of these unseen-class data and ensure performance, we propose a safe SSL method called SAFE-STUDENT from the teacher-student view. Firstly, a new scoring function called energy-discrepancy (ED) is proposed to help the teacher model improve the security of instances selection. Then, a novel unseen-class label distribution learning mechanism mitigates the unseen-class perturbation by calibrating the unseen-class label distribution. Finally, we propose an iterative optimization strategy to facilitate teacher-student network learning. Extensive studies on several representative datasets show that SAFE-STUDENT remarkably outperforms the state-of-the-art, verifying the feasibility and robustness of our method in the under-explored problem.

----

## [1408] A Simple Data Mixing Prior for Improving Self-Supervised Learning

**Authors**: *Sucheng Ren, Huiyu Wang, Zhengqi Gao, Shengfeng He, Alan L. Yuille, Yuyin Zhou, Cihang Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01419](https://doi.org/10.1109/CVPR52688.2022.01419)

**Abstract**:

Data mixing (e.g., Mixup, Cutmix, ResizeMix) is an essential component for advancing recognition models. In this paper, we focus on studying its effectiveness in the self-supervised setting. By noticing the mixed images that share the same source images are intrinsically related to each other, we hereby propose SDMP, short for Simple Data Mixing Prior, to capture this straightforward yet essential prior, and position such mixed images as additional positive pairs to facilitate self-supervised representation learning. Our experiments verify that the proposed SDMP enables data mixing to help a set of self-supervised learning frameworks (e.g., MoCo) achieve better accuracy and out-of-distribution robustness. More notably, our SDMP is the first method that successfully leverages data mixing to improve (rather than hurt) the performance of Vision Transformers in the self-supervised setting. Code is publicly available at https://github.com/OliverRensu/SDMP.

----

## [1409] DETReg: Unsupervised Pretraining with Region Priors for Object Detection

**Authors**: *Amir Bar, Xin Wang, Vadim Kantorov, Colorado J. Reed, Roei Herzig, Gal Chechik, Anna Rohrbach, Trevor Darrell, Amir Globerson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01420](https://doi.org/10.1109/CVPR52688.2022.01420)

**Abstract**:

Recent self-supervised pretraining methods for object detection largely focus on pretraining the backbone of the object detector, neglecting key parts of detection architecture. Instead, we introduce DETReg, a new self-supervised method that pretrains the entire object detection network, including the object localization and embedding components. During pretraining, DETReg predicts object localizations to match the localizations from an unsupervised region proposal generator and simultaneously aligns the corresponding feature embeddings with embeddings from a self-supervised image encoder. We implement DETReg using the DETR family of detectors and show that it improves over competitive baselines when finetuned on COCO, PASCAL VOC, and Airbus Ship benchmarks. In low-data regimes, including semi-supervised and few-shot learning settings, DETReg establishes many state-of-the-art results, e.g., on COCO we see a +6.0 AP improvement for 10-shot detection and over 2 AP improvements when training with only 1 % of the labels.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://www.amirbar.net/detreg/.

----

## [1410] Sound and Visual Representation Learning with Multiple Pretraining Tasks

**Authors**: *Arun Balajee Vasudevan, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01421](https://doi.org/10.1109/CVPR52688.2022.01421)

**Abstract**:

Different self-supervised tasks (SSL) reveal different features from the data. The learned feature representations can exhibit different performance for each downstream task. In this light, this work aims to combine Multiple SSL tasks (Multi-SSL) that generalizes wellfor all downstream tasks. For this study, we investigate binaural sounds and image data. For binaural sounds, we propose three SSL tasks namely, spatial alignment, temporal synchronization offore-ground objects and binaural sounds and temporal gap prediction. We investigate several approaches of Multi-SSL and give insights into the downstream task performance on video retrieval, spatial sound super resolution, and semantic prediction using OmniAudio dataset. Our experiments on binaural sound representations demonstrate that Multi-SSL via incremental learning (IL) of SSL tasks outperforms single SSL task models and fully supervised models in the downstream task performance. As a check of applicability on other modalities, we also formulate our Multi-SSL models for image representation learning and we use the recently proposed SSL tasks, MoCov2 and DenseCL. Here, Multi-SSL surpasses recent methods such as MoCov2, DenseCL and DetCo by 2.06%,3.27% and 1.19% on VOC07 classification and +2.83, +1.56 and +1.61 AP on COCO detection.

----

## [1411] UniVIP: A Unified Framework for Self-Supervised Visual Pre-training

**Authors**: *Zhaowen Li, Yousong Zhu, Fan Yang, Wei Li, Chaoyang Zhao, Yingying Chen, Zhiyang Chen, Jiahao Xie, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01422](https://doi.org/10.1109/CVPR52688.2022.01422)

**Abstract**:

Self-supervised learning (SSL) holds promise in leveraging large amounts of unlabeled data. However, the success of popular SSL methods has limited on single-centric-object images like those in ImageNet and ignores the correlation among the scene and instances, as well as the semantic difference of instances in the scene. To address the above problems, we propose a Unified Self-supervised Visual Pre-training (UniVIP), a novel self-supervised framework to learn versatile visual representations on either single-centric-object or non-iconic dataset. The framework takes into account the representation learning at three levels: 1) the similarity of scene-scene, 2) the correlation of scene-instance, 3) the discrimination of instance-instance. During the learning, we adopt the optimal transport algorithm to automatically measure the discrimination of instances. Massive experiments show that Uni-VIP pre-trained on non-iconic COCO achieves state-of-the-art transfer performance on a variety of downstream tasks, such as image classification, semi-supervised learning, object detection and segmentation. Furthermore, our method can also exploit single-centric-object dataset such as ImageNet and outperforms BYOL by 2.5% with the same pre-training epochs in linear probing, and surpass current self-supervised object detection methods on COCO dataset, demonstrating its universality and potential.

----

## [1412] Weakly Supervised Object Localization as Domain Adaption

**Authors**: *Lei Zhu, Qi She, Qian Chen, Yunfei You, Boyu Wang, Yanye Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01423](https://doi.org/10.1109/CVPR52688.2022.01423)

**Abstract**:

Weakly supervised object localization (WSOL) focuses on localizing objects only with the supervision of image-level classification masks. Most previous WSOL methods follow the classification activation map (CAM) that localizes objects based on the classification structure with the multi-instance learning (MIL) mechanism. However, the MIL mechanism makes CAM only activate discriminative object parts rather than the whole object, weakening its performance for localizing objects. To avoid this problem, this work provides a novel perspective that models WSOL as a domain adaption (DA) task, where the score estimator trained on the source/image domain is tested on the target/pixel domain to locate objects. Under this perspective, a DA-WSOL pipeline is designed to better engage DA approaches into WSOL to enhance localization performance. It utilizes a proposed target sampling strategy to select different types of target samples. Based on these types of target samples, domain adaption localization (DAL) loss is elaborated. It aligns the feature distribution between the two domains by DA and makes the estimator perceive target domain cues by Universum regularization. Experiments show that our pipeline outperforms SOTA methods on multi benchmarks. Code are released at https://github.com/zh460045050/DA-WSOL_CVPR2022.

----

## [1413] Debiased Learning from Naturally Imbalanced Pseudo-Labels

**Authors**: *Xudong Wang, Zhirong Wu, Long Lian, Stella X. Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01424](https://doi.org/10.1109/CVPR52688.2022.01424)

**Abstract**:

Pseudo-labels are confident predictions made on unlabeled target data by a classifier trained on labeled source data. They are widely used for adapting a model to unlabeled data, e.g., in a semi-supervised learning setting. Our key insight is that pseudo-labels are naturally imbalanced due to intrinsic data similarity, even when a model is trained on balanced source data and evaluated on balanced target data. If we address this previously unknown imbalanced classification problem arising from pseudo-labels instead of ground-truth training labels, we could remove model biases towards false majorities created by pseudo-labels. We propose a novel and effective debiased learning method with pseudo-labels, based on counterfactual reasoning and adaptive margins: The former removes the classifier response bias, whereas the latter adjusts the margin of each class according to the imbalance of pseudo-labels. Validated by extensive experimentation, our simple debiased learning delivers significant accuracy gains over the state-of-the-art on ImageNet-1K: 26% for semi-supervised learning with 0.2% annotations and 9% for zero-shot learning. Our code is available at: https://github.com/frank-xwang/debiased-pseudo-labeling.

----

## [1414] Towards Discovering the Effectiveness of Moderately Confident Samples for Semi-Supervised Learning

**Authors**: *Hui Tang, Kui Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01425](https://doi.org/10.1109/CVPR52688.2022.01425)

**Abstract**:

Semi-supervised learning (SSL) has been studied for a long time to solve vision tasks in data-efficient application scenarios. SSL aims to learn a good classification model using a few labeled data together with large-scale unlabeled data. Recent advances achieve the goal by combining multiple SSL techniques, e.g., self-training and consistency regularization. From unlabeled samples, they usually adopt a confidence filter (CF) to select reliable ones with high prediction confidence. In this work, we study whether the moderately confident samples are useless and how to select the useful ones to improve model optimization. To answer these problems, we propose a novel Taylor expansion inspired filtration (TEIF) framework, which admits the samples of moderate confidence with similar feature or gradient to the respective one averaged over the labeled and highly confident unlabeled data. It can produce a stable and new information induced network update, leading to better generalization. Two novel filters are derived from this framework and can be naturally explained in two perspectives. One is gradient synchronization filter (GSF), which strengthens the optimization dynamic of fully-supervised learning; it selects the samples whose gradients are similar to classwise majority gradients. The other is prototype proximity filter (PPF), which involves more prototypical samples in training to learn better semantic representations; it selects the samples near classwise prototypes. They can be integrated into SSL methods with CF. We use the state-of-the-art Fix-Match as the baseline. Experiments on popular SSL benchmarks show that we achieve the new state of the art.

----

## [1415] Masked Feature Prediction for Self-Supervised Visual Pre-Training

**Authors**: *Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan L. Yuille, Christoph Feichtenhofer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01426](https://doi.org/10.1109/CVPR52688.2022.01426)

**Abstract**:

We present Masked Feature Prediction (MaskFeat) for self-supervised pre-training of video models. Our approach first randomly masks out a portion of the input sequence and then predicts the feature of the masked regions. We study five different types of features and find Histograms of Oriented Gradients (HOG), a hand-crafted feature descriptor, works particularly well in terms of both performance and efficiency. We observe that the local contrast normalization in HOG is essential for good results, which is in line with earlier work using HOG for visual recognition. Our approach can learn abundant visual knowledge and drive large-scale Transformer based models. Without using extra model weights or supervision, MaskFeat pretrained on unlabeled videos achieves unprecedented results of 86.7% with MViTv2-L on Kinetics-400, 88.3% on Kinetics 600, 80.4% on Kinetics-700, 38.8 mAP on AVA, and 75.0% on SSv2. MaskFeat further generalizes to image input, which can be interpreted as a video with a single frame and obtains competitive results on ImageN et.

----

## [1416] Contrastive Learning for Space-time Correspondence via Self-cycle Consistency

**Authors**: *Jeany Son*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01427](https://doi.org/10.1109/CVPR52688.2022.01427)

**Abstract**:

We propose a novel probabilistic method employing Bayesian Model Averaging and self-cycle regularization for spatio-temporal correspondence learning in videos within a self-supervised learning framework. Most existing methods for self-supervised correspondence learning suffer from noisy labels that come with the data for free, and the presence of occlusion exacerbates the problem. We tackle this issue within a probabilistic framework that handles model uncertainty inherent in the path selection problem built on a complete graph. We propose a self-cycle regularization to consider a cycle-consistency property on individual edges in order to prevent converging on noisy matching or trivial solutions. We also utilize a mixture of sequential Bayesian filters to estimate posterior distribution for targets. In addition, we present a domain contrastive loss to learn discriminative representation among videos. Our algorithm is evaluated on various datasets for video label propagation tasks including DAVIS2017, VIP and JHMDB, and shows outstanding performances compared to the state-of-the-art self-supervised learning based video correspondence algorithms. Moreover, our method converges significantly faster than previous methods.

----

## [1417] Id-Free Person Similarity Learning

**Authors**: *Bing Shuai, Xinyu Li, Kaustav Kundu, Joseph Tighe*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01428](https://doi.org/10.1109/CVPR52688.2022.01428)

**Abstract**:

Learning a unified person detection and re-identification model is a key component of modern trackers. However, training such models usually relies on the availability of training images / videos that are manually labeled with both person boxes and their identities. In this work, we explore training such a model by only using person box annotations, thus removing the necessity of manually labeling a training dataset with additional person identity annotation as these are expensive to collect. To this end, we present a contrastive learning framework to learn person similarity without using manually labeled identity annotations. First, we apply image-level augmentation to images on public person detection datasets, based on which we learn a strong model for general person detection as well as for short-term person re-identification. To learn a model capable of longerterm re-identification, we leverage the natural appearance evolution of each person in videos to serve as instance-level appearance augmentation in our contrastive loss formulation. Without access to the target dataset or person identity annotation, our model achieves competitive results compared to existing fully-supervised state-of-the-art methods on both person search and person tracking tasks. Our model also shows promising results for saving the annotation cost that is needed to achieve a certain level of performance on the person search task.

----

## [1418] End-to-End Semi-Supervised Learning for Video Action Detection

**Authors**: *Akash Kumar, Yogesh Singh Rawat*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01429](https://doi.org/10.1109/CVPR52688.2022.01429)

**Abstract**:

In this work, we focus on semi-supervised learning for video action detection which utilizes both labeled as well as unlabeled data. We propose a simple end-to-end consistency based approach which effectively utilizes the unlabeled data. Video action detection requires both, action class prediction as well as a spatio-temporal localization of actions. Therefore, we investigate two types of constraints, classification consistency, and spatio-temporal consistency. The presence of predominant background and static regions in a video makes it challenging to utilize spatio-temporal consistency for action detection. To address this, we propose two novel regularization constraints for spatio-temporal consistency; 1) temporal coherency, and 2) gradient smoothness. Both these aspects exploit the temporal continuity of action in videos and are found to be effective for utilizing unlabeled videos for action detection. We demonstrate the effectiveness of the proposed approach on two different action detection benchmark datasets, UCF101-24 and IHMDB-21. In addition, we also show the effectiveness of the proposed approach for video object segmentation on the Youtube-VOS which demonstrates its generalization capability The proposed approach achieves competitive performance by using merely 20% of annotations on UCF101-24 when compared with recent fully supervised methods. On UCF101-24, it improves the score by +8.9% and +11% at 0.5 f-mAP and v-mAP respectively, compared to supervised approach. The code and models will be made publicly available at: https://github.com/AKASH2907/End-to-End-Semi-Supervised-Learning-for-Video-Action-Detection.

----

## [1419] Probabilistic Representations for Video Contrastive Learning

**Authors**: *Jungin Park, Jiyoung Lee, Ig-Jae Kim, Kwanghoon Sohn*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01430](https://doi.org/10.1109/CVPR52688.2022.01430)

**Abstract**:

This paper presents Probabilistic Video Contrastive Learning, a self-supervised representation learning method that bridges contrastive learning with probabilistic representation. We hypothesize that the clips composing the video have different distributions in short-term duration, but can represent the complicated and sophisticated video distribution through combination in a common embedding space. Thus, the proposed method represents video clips as normal distributions and combines them into a Mixture of Gaussians to model the whole video distribution. By sampling embeddings from the whole video distribution, we can circumvent the careful sampling strategy or transformations to generate augmented views of the clips, unlike previous deterministic methods that have mainly focused on such sample generation strategies for contrastive learning. We further propose a stochastic contrastive loss to learn proper video distributions and handle the inherent uncertainty from the nature of the raw video. Experimental results verify that our probabilistic embedding stands as a state-of-the-art video representation learning for action recognition and video retrieval on the most popular benchmarks, including UCF101 and HMDB51.

----

## [1420] Interact before Align: Leveraging Cross-Modal Knowledge for Domain Adaptive Action Recognition

**Authors**: *Lijin Yang, Yifei Huang, Yusuke Sugano, Yoichi Sato*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01431](https://doi.org/10.1109/CVPR52688.2022.01431)

**Abstract**:

Unsupervised domain adaptive video action recognition aims to recognize actions of a target domain using a model trained with only out-of-domain (source) annotations. The inherent complexity of videos makes this task challenging but also provides ground for leveraging multi-modal inputs (e.g., RGB, Flow, Audio). Most previous works utilize the multi-modal information by either aligning each modality individually or learning representation via cross-modal self-supervision. Different from previous works, we find that the cross-domain alignment can be more effectively done by using cross-modal interaction first. Cross-modal knowledge interaction allows other modalities to supplement missing transferable information because of the cross-modal complementarity. Also, the most transferable aspects of data can be highlighted using cross-modal consensus. In this work, we present a novel model that jointly considers these two characteristics for domain adaptive action recognition. We achieve this by implementing two modules, where the first module exchanges complementary transferable information across modalities through the semantic space, and the second module finds the most transferable spatial region based on the consensus of all modalities. Extensive experiments validate that our proposed method can significantly outperform state-of-the-art methods on multiple benchmark datasets, including the complex fine-grained dataset EPIC-Kitchens-100.

----

## [1421] BEVT: BERT Pretraining of Video Transformers

**Authors**: *Rui Wang, Dongdong Chen, Zuxuan Wu, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Yu-Gang Jiang, Luowei Zhou, Lu Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01432](https://doi.org/10.1109/CVPR52688.2022.01432)

**Abstract**:

This paper studies the BERT pretraining of video transformers. It is a straightforward but worth-studying extension given the recent success from BERT pretraining of image transformers. We introduce BEVT which decouples video representation learning into spatial representation learning and temporal dynamics learning. In particular, BEVT first performs masked image modeling on image data, and then conducts masked image modeling jointly with masked video modeling on video data. This design is motivated by two observations: 1) transformers learned on image datasets provide decent spatial priors that can ease the learning of video transformers, which are often times computationally-intensive if trained from scratch; 2) discriminative clues, i.e., spatial and temporal information, needed to make correct predictions vary among different videos due to large intra-class and inter-class variations. We conduct extensive experiments on three challenging video benchmarks where BEVT achieves very promising results. On Kinetics 400, for which recognition mostly relies on discriminative spatial representations, BEVT achieves comparable results to strong supervised baselines. On Something-Something-V2 and Diving 48, which contain videos relying on temporal dynamics, BEVT outperforms by clear margins all alternative baselines and achieves state-of-the-art performance with a 71.4% and 87.2% Top-1 accuracy respectively. Code is available at https://github.com/xyzforever/BEVT.

----

## [1422] Generative Cooperative Learning for Unsupervised Video Anomaly Detection

**Authors**: *Muhammad Zaigham Zaheer, Arif Mahmood, Muhammad Haris Khan, Mattia Segù, Fisher Yu, Seung-Ik Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01433](https://doi.org/10.1109/CVPR52688.2022.01433)

**Abstract**:

Video anomaly detection is well investigated in weakly-supervised and one-class classification (OCC) settings. However, unsupervised video anomaly detection methods are quite sparse, likely because anomalies are less frequent in occurrence and usually not well-defined, which when coupled with the absence of ground truth supervision, could adversely affect the performance of the learning algorithms. This problem is challenging yet rewarding as it can completely eradicate the costs of obtaining laborious annotations and enable such systems to be deployed without human intervention. To this end, we propose a novel unsupervised Generative Cooperative Learning (GCL) approach for video anomaly detection that exploits the low frequency of anomalies towards building a cross-supervision between a generator and a discriminator. In essence, both networks get trained in a cooperative fashion, thereby allowing unsupervised learning. We conduct extensive experiments on two large-scale video anomaly detection datasets, UCF crime and ShanghaiTech. Consistent improvement over the existing state-of-the-art unsupervised and OCC methods corroborate the effectiveness of our approach.

----

## [1423] The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization

**Authors**: *Muhammad Jehanzeb Mirza, Jakub Micorek, Horst Possegger, Horst Bischof*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01435](https://doi.org/10.1109/CVPR52688.2022.01435)

**Abstract**:

Domain adaptation is crucial to adapt a learned model to new scenarios, such as domain shifts or changing data distributions. Current approaches usually require a large amount of labeled or unlabeled data from the shifted domain. This can be a hurdle in fields which require continuous dynamic adaptation or suffer from scarcity of data, e.g. autonomous driving in challenging weather conditions. To address this problem of continuous adaptation to distribution shifts, we propose Dynamic Unsupervised Adaptation (DUA). By continuously adapting the statistics of the batch normalization layers we modify the feature representations of the model. We show that by sequentially adapting a model with only a fraction of unlabeled data, a strong performance gain can be achieved. With even less than 1% of unlabeled data from the target domain, DUA already achieves competitive results to strong baselines. In addition, the computational overhead is minimal in contrast to previous approaches. Our approach is simple, yet effective and can be applied to any architecture which uses batch normalization as one of its components. We show the utility of DUA by evaluating it on a variety of domain adaptation datasets and tasks including object recognition, digit recognition and object detection.

----

## [1424] What Matters For Meta-Learning Vision Regression Tasks?

**Authors**: *Ning Gao, Hanna Ziesche, Ngo Anh Vien, Michael Volpp, Gerhard Neumann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01436](https://doi.org/10.1109/CVPR52688.2022.01436)

**Abstract**:

Meta-learning is widely used in few-shot classification and function regression due to its ability to quickly adapt to unseen tasks. However, it has not yet been well explored on regression tasks with high dimensional inputs such as images. This paper makes two main contributions that help understand this barely explored area. First, we design two new types of cross-category level vision regression tasks, namely object discovery and pose estimation of unprecedented complexity in the meta-learning domain for computer vision. To this end, we (i) exhaustively evaluate common meta-learning techniques on these tasks, and (ii) quantitatively analyze the effect of various deep learning techniques commonly used in recent meta-learning algorithms in order to strengthen the generalization capability: data augmentation, domain randomization, task augmentation and meta-regularization. Finally, we (iii) provide some insights and practical recommendations for training meta-learning algorithms on vision regression tasks. Second, we propose the addition of functional contrastive learning (FCL) over the task representations in Conditional Neural Processes (CNPs) and train in an end-to-end fashion. The experimental results show that the results of prior work are misleading as a consequence of a poor choice of the loss function as well as too small meta-training sets. Specifically, we find that CNPs outperform MAML on most tasks without fine-tuning. Furthermore, we observe that naive task augmentation without a tailored design results in underfitting.

----

## [1425] IFOR: Iterative Flow Minimization for Robotic Object Rearrangement

**Authors**: *Ankit Goyal, Arsalan Mousavian, Chris Paxton, Yu-Wei Chao, Brian Okorn, Jia Deng, Dieter Fox*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01437](https://doi.org/10.1109/CVPR52688.2022.01437)

**Abstract**:

Accurate object rearrangement from vision is a crucial problem for a wide variety of real-world robotics applications in unstructured environments. We propose IFOR, Iterative Flow Minimization for Robotic Object Rearrangement, an end-to-end method for the challenging problem of object rearrangement for unknown objects given an RGBD image of the original and final scenes. First, we learn an optical flow model based on RAFT to estimate the relative transformation of the objects purely from synthetic data. This flow is then used in an iterative minimization algorithm to achieve accurate positioning of previously unseen objects. Crucially, we show that our method applies to cluttered scenes, and in the real world, while training only on synthetic data. Videos are available at h t t ps: //imankgoyal.github.io/ifor.html.

----

## [1426] TCTrack: Temporal Contexts for Aerial Tracking

**Authors**: *Ziang Cao, Ziyuan Huang, Liang Pan, Shiwei Zhang, Ziwei Liu, Changhong Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01438](https://doi.org/10.1109/CVPR52688.2022.01438)

**Abstract**:

Temporal contexts among consecutive frames are far from being fully utilized in existing visual trackers. In this work, we present TCTrack
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/vision4robotics/TCTrack, a comprehensive framework to fully exploit temporal contexts for aerial tracking. The temporal contexts are incorporated at two levels: the extraction of features and the refinement of similarity maps. Specifically, for feature extraction, an online temporally adaptive convolution is proposed to enhance the spatial features using temporal information, which is achieved by dynamically calibrating the convolution weights according to the previous frames. For similarity map refinement, we propose an adaptive temporal transformer, which first effectively encodes temporal knowledge in a memory-efficient way, before the temporal knowledge is decoded for accurate adjustment of the similarity map. TCTrack is effective and efficient: evaluation on four aerial tracking benchmarks shows its impressive performance; real-world UAV tests show its high speed of over 27 FPS on NVIDIA Jetson AGX Xavier.

----

## [1427] AKB-48: A Real-World Articulated Object Knowledge Base

**Authors**: *Liu Liu, Wenqiang Xu, Haoyuan Fu, Sucheng Qian, Qiaojun Yu, Yang Han, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01439](https://doi.org/10.1109/CVPR52688.2022.01439)

**Abstract**:

Human life is populated with articulated objects. A comprehensive understanding of articulated objects, namely appearance, structure, physical property, and semantics, will benefit many research communities. As current articulated object understanding solutions are usually based on synthetic object dataset with CAD models without physics properties, which prevent satisfied generalization from simulation to real-world applications in visual and robotics tasks. To bridge the gap, we present AKB-48: a large-scale Articulated object Knowledge Base which consists of 2,037 real-world 3D articulated object models of 48 categories. Each object is described by a knowledge graph ArtiKG. To build the AKB-48, we present a fast articulation knowledge modeling (FArM) pipeline, which can fulfill the ArtiKG for an articulated object within 10–15 minutes, and largely reduce the cost for object modeling in the real world. Using our dataset, we propose AKBNet, an integral pipeline for Category-level Visual Articulation Manipulation (C-VAM) task, in which we benchmark three sub-tasks, namely pose estimation, object reconstruction and manipulation. Dataset, codes, and models are publicly available at https://liuliu66.github.io/AKB-48.

----

## [1428] 3DAC: Learning Attribute Compression for Point Clouds

**Authors**: *Guangchi Fang, Qingyong Hu, Hanyun Wang, Yiling Xu, Yulan Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01440](https://doi.org/10.1109/CVPR52688.2022.01440)

**Abstract**:

We study the problem of attribute compression for large-scale unstructured 3D point clouds. Through an in-depth exploration of the relationships between different encoding steps and different attribute channels, we introduce a deep compression network, termed 3DAC, to explicitly compress the attributes of 3D point clouds and reduce storage usage in this paper. Specifically, the point cloud attributes such as color and reflectance are firstly converted to transform coefficients. We then propose a deep entropy model to model the probabilities of these coefficients by considering information hidden in attribute transforms and previous encoded attributes. Finally, the estimated probabilities are used to further compress these transform coefficients to a final attributes bitstream. Extensive experiments conducted on both indoor and outdoor large-scale open point cloud datasets, including ScanNet and SemanticKITTI, demonstrated the superior compression rates and reconstruction quality of the proposed method.

----

## [1429] Simple but Effective: CLIP Embeddings for Embodied AI

**Authors**: *Apoorv Khandelwal, Luca Weihs, Roozbeh Mottaghi, Aniruddha Kembhavi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01441](https://doi.org/10.1109/CVPR52688.2022.01441)

**Abstract**:

Contrastive language image pretraining (CLIP) encoders have been shown to be beneficial for a range of visual tasks from classification and detection to captioning and image manipulation. We investigate the effectiveness of CLIP visual backbones for Embodied AI tasks. We build incredibly simple baselines, named EmbCLIP, with no task specific architectures, inductive biases (such as the use of semantic maps), auxiliary tasks during training, or depth maps-yet we find that our improved baselines perform very well across a range of tasks and simulators. EmbCLIP tops the RoboTHOR ObjectNav leader-board by a huge margin of 20 pts (Success Rate). It tops the iTHOR 1-Phase Rearrangement leaderboard, beating the next best submission, which employs Active Neural Mapping, and more than doubling the % Fixed Strict metric (0.08 to 0.17). It also beats the winners of the 2021 Habitat ObjectNav Challenge, which employ auxiliary tasks, depth maps, and human demonstrations, and those of the 2019 Habitat PointNav Challenge. We evaluate the ability of CLIP's visual representations at capturing semantic information about input observations-primitives that are useful for navigation-heavy embodied tasks- and find that CLIP's representations encode these primitives more effectively than ImageNet-pretrained backbones. Finally, we extend one of our baselines, producing an agent capable of zero-shot object navigation that can navigate to objects that were not used as targets during training. Our code and models are available at https://github.com/allenai/embodied-clip.

----

## [1430] Multi-Robot Active Mapping via Neural Bipartite Graph Matching

**Authors**: *Kai Ye, Siyan Dong, Qingnan Fan, He Wang, Li Yi, Fei Xia, Jue Wang, Baoquan Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01442](https://doi.org/10.1109/CVPR52688.2022.01442)

**Abstract**:

We study the problem of multi-robot active mapping, which aims for complete scene map construction in minimum time steps. The key to this problem lies in the goal position estimation to enable more efficient robot movements. Previous approaches either choose the frontier as the goal position via a myopic solution that hinders the time efficiency, or maximize the long-term value via reinforcement learning to directly regress the goal position, but does not guarantee the complete map construction. In this paper, we propose a novel algorithm, namely NeuralCoMapping, which takes advantage of both approaches. We reduce the problem to bipartite graph matching, which establishes the node correspondences between two graphs, denoting robots and frontiers. We introduce a multiplex graph neural network (mGNN) that learns the neural distance to fill the affinity matrix for more effective graph matching. We optimize the mGNN with a differentiable linear assignment layer by maximizing the long-term values that favor time efficiency and map completeness via reinforcement learning. We compare our algorithm with several state-of-the-art multi-robot active mapping approaches and adapted reinforcement-learning baselines. Experimental results demonstrate the superior performance and exceptional generalization ability of our algorithm on various indoor scenes and unseen number of robots, when only trained with 9 indoor scenes.

----

## [1431] Continuous Scene Representations for Embodied AI

**Authors**: *Samir Yitzhak Gadre, Kiana Ehsani, Shuran Song, Roozbeh Mottaghi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01443](https://doi.org/10.1109/CVPR52688.2022.01443)

**Abstract**:

We propose Continuous Scene Representations (CSR), a scene representation constructed by an embodied agent navigating within a space, where objects and their relationships are modeled by continuous valued embeddings. Our method captures feature relationships between objects, composes them into a graph structure on-the-fly, and situates an embodied agent within the representation. Our key insight is to embed pair-wise relationships between objects in a latent space. This allows for a richer representation compared to discrete relations (e.g., [SUPPORT], [NEXT-TO]) commonly used for building scene representations. CSR can track objects as the agent moves in a scene, update the representation accordingly, and detect changes in room configurations. Using CSR, we outperform state-of-the-art approaches for the challenging downstream task of visual room rearrangement, without any task specific training. Moreover, we show the learned embeddings capture salient spatial details of the scene and show applicability to real world data. A summery video and code is available at prior.allenai.org/projects/csr.

----

## [1432] Interactron: Embodied Adaptive Object Detection

**Authors**: *Klemen Kotar, Roozbeh Mottaghi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01444](https://doi.org/10.1109/CVPR52688.2022.01444)

**Abstract**:

Over the years various methods have been proposed for the problem of object detection. Recently, we have wit-nessed great strides in this domain owing to the emergence of powerful deep neural networks. However, there are typically two main assumptions common among these approaches. First, the model is trained on a fixed training set and is evaluated on a pre-recorded test set. Second, the model is kept frozen after the training phase, so no further updates are performed after the training is finished. These two assumptions limit the applicability of these methods to real-world settings. In this paper, we propose Interactron, a method for adaptive object detection in an interactive setting, where the goal is to perform object detection in images observed by an embodied agent navigating in different environments. Our idea is to continue training during inference and adapt the model at test time without any explicit supervision via interacting with the environment. Our adaptive object detection model provides a 11.8 point improvement in AP (and 19.1 points in 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$AP_{50}$</tex>
) over DETR [5]. a recent, high-performance object detector. Moreover, we show that our object detection model adapts to environments with completely different appearance characteristics, and its performance is on par with a model trained with full supervision for those environments. The code is available at: https://github.com/allenai/interactron.

----

## [1433] Online Learning of Reusable Abstract Models for Object Goal Navigation

**Authors**: *Tommaso Campari, Leonardo Lamanna, Paolo Traverso, Luciano Serafini, Lamberto Ballan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01445](https://doi.org/10.1109/CVPR52688.2022.01445)

**Abstract**:

In this paper, we present a novel approach to incrementally learn an Abstract Model of an unknown environment, and show how an agent can reuse the learned model for tackling the Object Goal Navigation task. The Abstract Model is a finite state machine in which each state is an abstraction of a state of the environment, as perceived by the agent in a certain position and orientation. The perceptions are high-dimensional sensory data (e.g., RGB-D images), and the abstraction is reached by exploiting image segmentation and the Taskonomy model bank. The learning of the Abstract Model is accomplished by executing actions, observing the reached state, and updating the Abstract Model with the acquired information. The learned models are memorized by the agent, and they are reused whenever it recognizes to be in an environment that corresponds to the stored model. We investigate the effectiveness of the proposed approach for the Object Goal Navigation task, relying on public benchmarks. Our results show that the reuse of learned Abstract Models can boost performance on Object Goal Navigation.

----

## [1434] RNNPose: Recurrent 6-DoF Object Pose Refinement with Robust Correspondence Field Estimation and Pose Optimization

**Authors**: *Yan Xu, Kwan-Yee Lin, Guofeng Zhang, Xiaogang Wang, Hongsheng Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01446](https://doi.org/10.1109/CVPR52688.2022.01446)

**Abstract**:

6-DoF object pose estimation from a monocular image is challenging, and a post-refinement procedure is generally needed for high-precision estimation. In this paper, we propose a framework based on a recurrent neural network (RNN) for object pose refinement, which is robust to erroneous initial poses and occlusions. During the recurrent iterations, object pose refinement is formulated as a nonlinear least squares problem based on the estimated correspondence field (between a rendered image and the observed image). The problem is then solved by a differentiable Levenberg-Marquardt (LM) algorithm enabling end-to-end training. The correspondence field estimation and pose refinement are conducted alternatively in each iteration to recover the object poses. Furthermore, to improve the robustness to occlusion, we introduce a consistency-check mechanism based on the learned descriptors of the 3D model and observed 2D images, which downweights the unreliable correspondences during pose optimization. Extensive experiments on LINEMOD, Occlusion-LINEMOD, and YCB-Video datasets validate the effectiveness of our method and demonstrate state-of-the-art performance.

----

## [1435] UDA-COPE: Unsupervised Domain Adaptation for Category-level Object Pose Estimation

**Authors**: *Taeyeop Lee, Byeong-Uk Lee, Inkyu Shin, Jaesung Choe, Ukcheol Shin, In So Kweon, Kuk-Jin Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01447](https://doi.org/10.1109/CVPR52688.2022.01447)

**Abstract**:

Learning to estimate object pose often requires ground-truth (GT) labels, such as CAD model and absolute-scale object pose, which is expensive and laborious to obtain in the real world. To tackle this problem, we propose an unsupervised domain adaptation (UDA) for category-level object pose estimation, called UDA-COPE. Inspired by recent multi-modal UDA techniques, the proposed method exploits a teacher-student self-supervised learning scheme to train a pose estimation network without using target domain pose labels. We also introduce a bidirectional filtering method between the predicted normalized object coordinate space (NOCS) map and observed point cloud, to not only make our teacher network more robust to the target domain but also to provide more reliable pseudo labels for the student network training. Extensive experimental results demonstrate the effectiveness of our proposed method both quantitatively and qualitatively. Notably, without leveraging target-domain GT labels, our proposed method achieved comparable or sometimes superior performance to existing methods that depend on the GT labels.

----

## [1436] Symmetry and Uncertainty-Aware Object SLAM for 6DoF Object Pose Estimation

**Authors**: *Nathaniel Merrill, Yuliang Guo, Xingxing Zuo, Xinyu Huang, Stefan Leutenegger, Xi Peng, Liu Ren, Guoquan Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01448](https://doi.org/10.1109/CVPR52688.2022.01448)

**Abstract**:

We propose a keypoint-based object-level SLAM framework that can provide globally consistent 6DoF pose estimates for symmetric and asymmetric objects alike. To the best of our knowledge, our system is among the first to utilize the camera pose information from SLAM to provide prior knowledge for tracking keypoints on symmetric objects - ensuring that new measurements are consistent with the current 3D scene. Moreover, our semantic key-point network is trained to predict the Gaussian covariance for the keypoints that captures the true error of the prediction, and thus is not only useful as a weight for the residuals in the system's optimization problems, but also as a means to detect harmful statistical outliers without choosing a manual threshold. Experiments show that our method provides competitive performance to the state of the art in 6DoF object pose estimation, and at a real-time speed. Our code, pre-trained models, and keypoint labels are available https://github.com/rpng/suo_slam.

----

## [1437] Upright-Net: Learning Upright Orientation for 3D Point Cloud

**Authors**: *Xufang Pang, Feng Li, Ning Ding, Xiaopin Zhong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01449](https://doi.org/10.1109/CVPR52688.2022.01449)

**Abstract**:

A mass of experiments shows that the pose of the input 3D models exerts a tremendous influence on automatic 3D shape analysis. In this paper, we propose Upright-Net, a deep-learning-based approach for estimating the upright orientation of 3D point clouds. Based on a well-known postulate of design states that “form ever follows function”, we treat the natural base of an object as a common functional structure, which supports the object in a most commonly seen pose following a set of specific rules, e.g. physical laws, functionality-related geometric properties, semantic cues, and so on. Thus we apply a data-driven deep learning method to automatically encode those rules and formulate the upright orientation estimation problem as a classification model, i.e. extract the points on a 3D model that forms the natural base. And then the upright orientation is computed as the normal of the natural base. Our proposed new approach has three advantages. First, it formulates the continuous orientation estimation task as a discrete classification task while preserving the continuity of the solution space. Second, it automatically learns the comprehensive criteria defining a natural base of general 3D models even with asymmetric geometry. Third, the learned orientation-aware features can serve well in downstream tasks. Results show that our network outperforms previous approaches on orientation estimation and also achieves remarkable generalization capability and transfer capability.

----

## [1438] DeepFake Disrupter: The Detector of DeepFake Is My Friend

**Authors**: *Xueyu Wang, Jiajun Huang, Siqi Ma, Surya Nepal, Chang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01450](https://doi.org/10.1109/CVPR52688.2022.01450)

**Abstract**:

In recent years, with the advances of generative models, many powerful face manipulation systems have been developed based on Deep Neural Networks (DNNs), called DeepFakes. If DeepFakes are not controlled timely and properly, they would become a real threat to both celebrities and ordinary people. Precautions such as adding perturbations to the source inputs will make DeepFake results look distorted from the perspective of human eyes. However, previous method doesn't explore whether the disrupted images can still spoof DeepFake detectors. This is critical for many applications where DeepFake detectors are used to discriminate between DeepFake data and real data due to the huge cost of examining a large amount of data manually. We argue that the detectors do not share a similar perspective as human eyes, which might still be spoofed by the disrupted data. Besides, the existing disruption methods rely on iteration-based perturbation generation algorithms, which is time-consuming. In this paper, we propose a novel DeepFake disruption algorithm called “DeepFake Disrupter”. By training a perturbation generator, we can add the human-imperceptible perturbations to source images that need to be protected without any backpropagation update. The DeepFake results of these protected source inputs would not only look unrealistic by the human eye but also can be distinguished by DeepFake detectors easily. For example, experimental results show that by adding our trained perturbations, fake images generated by StarGAN [5] can result in a 10 ∼ 20% increase in F1-score evaluated by various DeepFake detectors.

----

## [1439] HybridCR: Weakly-Supervised 3D Point Cloud Semantic Segmentation via Hybrid Contrastive Regularization

**Authors**: *Mengtian Li, Yuan Xie, Yunhang Shen, Bo Ke, Ruizhi Qiao, Bo Ren, Shaohui Lin, Lizhuang Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01451](https://doi.org/10.1109/CVPR52688.2022.01451)

**Abstract**:

To address the huge labeling cost in large-scale point cloud semantic segmentation, we propose a novel hybrid contrastive regularization (HybridCR) framework in weakly-supervised setting, which obtains competitive performance compared to its fully-supervised counterpart. Specifically, HybridCR is the first framework to leverage both point consistency and employ contrastive regularization with pseudo labeling in an end-to-end manner. Fundamentally, HybridCR explicitly and effectively considers the semantic similarity between local neighboring points and global characteristics of 3D classes. We further design a dynamic point cloud augmentor to generate diversity and robust sample views, whose transformation parameter is jointly optimized with model training. Through extensive experiments, HybridCR achieves significant performance improvement against the SOTA methods on both indoor and outdoor datasets, e.g., S3DIS, ScanNet-V2, Semantic3D, and SemanticKITTI.

----

## [1440] Open-Domain, Content-based, Multi-modal Fact-checking of Out-of-Context Images via Online Resources

**Authors**: *Sahar Abdelnabi, Rakibul Hasan, Mario Fritz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01452](https://doi.org/10.1109/CVPR52688.2022.01452)

**Abstract**:

Misinformation is now a major problem due to its poten-tial high risks to our core democratic and societal values and orders. Out-of-context misinformation is one of the easiest and effective ways used by adversaries to spread vi-ral false stories. In this threat, a real image is re-purposed to support other narratives by misrepresenting its context and/or elements. The internet is being used as the go-to way to verify information using different sources and modali-ties. Our goal is an inspectable method that automates this time-consuming and reasoning-intensive process by fact-checking the image-caption pairing using Web evidence. To integrate evidence and cues from both modalities, we intro-duce the concept of ‘multi-modal cycle-consistency check’ starting from the image/caption, we gather tex-tual/visual evidence, which will be compared against the other paired caption/image, respectively. Moreover, we propose a novel architecture, Consistency-Checking Network (CCN), that mimics the layered human reasoning across the same and different modalities: the caption vs. textual evidence, the image vs. visual evidence, and the image vs. caption. Our work offers the first step and bench-mark for open-domain, content-based, multi-modal fact-checking, and significantly outperforms previous baselines that did not leverage external evidence
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
For code, checkpoints, and dataset, check: https://s-abdelnabi.github.io/OoC-multi-modal-fc/.

----

## [1441] Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection

**Authors**: *Alexandros Haliassos, Rodrigo Mira, Stavros Petridis, Maja Pantic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01453](https://doi.org/10.1109/CVPR52688.2022.01453)

**Abstract**:

One of the most pressing challenges for the detection of face-manipulated videos is generalising to forgery methods not seen during training while remaining effective under common corruptions such as compression. In this paper, we examine whether we can tackle this issue by harnessing videos of real talking faces, which contain rich information on natural facial appearance and behaviour and are readily available in large quantities online. Our method, termed RealForensics, consists of two stages. First, we exploit the natural correspondence between the visual and auditory modalities in real videos to learn, in a self-supervised cross-modal manner, temporally dense video representations that capture factors such as facial movements, expression, and identity. Second, we use these learned representations as targets to be predicted by our forgery detector along with the usual binary forgery classification task; this encourages it to base its real/fake decision on said factors. We show that our method achieves state-of-the-art performance on cross-manipulation generalisation and robustness experiments, and examine the factors that contribute to its per-formance. Our results suggest that leveraging natural and unlabelled videos is a promising direction for the development of more robust face forgery detectors.

----

## [1442] Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection

**Authors**: *Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01455](https://doi.org/10.1109/CVPR52688.2022.01455)

**Abstract**:

Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.

----

## [1443] Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability

**Authors**: *Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01456](https://doi.org/10.1109/CVPR52688.2022.01456)

**Abstract**:

The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security. Meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly. Code is available at https://github.com/JHL-HUST/SVRE.

----

## [1444] Improving Adversarial Transferability via Neuron Attribution-based Attacks

**Authors**: *Jianping Zhang, Weibin Wu, Jen-tse Huang, Yizhan Huang, Wenxuan Wang, Yuxin Su, Michael R. Lyu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01457](https://doi.org/10.1109/CVPR52688.2022.01457)

**Abstract**:

Deep neural networks (DNNs) are known to be vulnerable to adversarial examples. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs beforehand in security-sensitive applications. To efficiently tackle the black-box setting where the target model's particulars are unknown, feature-level transfer-based attacks propose to contaminate the intermediate feature outputs of local models, and then directly employ the crafted adversarial samples to attack the target model. Due to the transferability of features, feature-level attacks have shown promise in synthesizing more transferable adversarial samples. However, existing feature-level attacks generally employ inaccurate neuron importance estimations, which deteriorates their transferability. To overcome such pitfalls, in this paper, we propose the Neuron Attribution-based Attack (NAA), which conducts feature-level attacks with more accurate neuron importance estimations. Specifically, we first completely attribute a model's output to each neuron in a middle layer. We then derive an approximation scheme of neuron attribution to tremendously reduce the computation overhead. Finally, we weight neurons based on their attribution results and launch feature-level attacks. Extensive experiments confirm the superiority of our approach to the state-of-the-art benchmarks. Our code is available at: hups.//rgithub.com/jprhang1810/NAA.

----

## [1445] Complex Backdoor Detection by Symmetric Feature Differencing

**Authors**: *Yingqi Liu, Guangyu Shen, Guanhong Tao, Zhenting Wang, Shiqing Ma, Xiangyu Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01458](https://doi.org/10.1109/CVPR52688.2022.01458)

**Abstract**:

Many existing backdoor scanners work by finding a small and fixed trigger. However, advanced attacks have large and pervasive triggers, rendering existing scanners less effective. We develop a new detection method. It first uses a trigger inversion technique to generate triggers, namely, universal input patterns flipping victim class samples to a target class. It then checks if any such trigger is composed of features that are not natural distinctive features between the victim and target classes. It is based on a novel symmetric feature differencing method that identifies features separating two sets of samples (e.g., from two respective classes). We evaluate the technique on a number of advanced attacks including composite attack, reflection attack, hidden attack, filter attack, and also on the traditional patch attack. The evaluation is on thousands of models, including both clean and trojaned models, with various architectures. We compare with three state-of-the-art scanners. Our technique can achieve 80-88% accuracy while the baselines can only achieve 50-70% on complex attacks. Our results on the TrojAI competition rounds 2–4, which have patch backdoors and filter backdoors, show that existing scanners may produce hundreds of false positives (i.e., clean models recognized as trojaned), while our technique removes 78-100% of them with a small increase of false negatives by 0-30%, leading to 17-41% overall accuracy improvement. This allows us to achieve top performance on the leaderboard.

----

## [1446] Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer

**Authors**: *Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01459](https://doi.org/10.1109/CVPR52688.2022.01459)

**Abstract**:

While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN) 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/CGCL-codes/AMT-GAN, a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

----

## [1447] Zero-Query Transfer Attacks on Context-Aware Object Detectors

**Authors**: *Zikui Cai, Shantanu Rane, Alejandro E. Brito, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01460](https://doi.org/10.1109/CVPR52688.2022.01460)

**Abstract**:

Adversarial attacks perturb images such that a deep neural network produces incorrect classification results. A promising approach to defend against adversarial attacks on natural multi-object scenes is to impose a context-consistency check, wherein, if the detected objects are not consistent with an appropriately defined context, then an attack is suspected. Stronger attacks are needed to fool such context-aware detectors. We present the first approach for generating context-consistent adversarial attacks that can evade the context-consistency check of black-box object detectors operating on complex, natural scenes. Unlike many black-box attacks that perform repeated attempts and open themselves to detection, we assume a “zero-query” setting, where the attacker has no knowledge of the classification decisions of the victim system. First, we derive multiple attack plans that assign incorrect labels to victim objects in a context-consistent manner. Then we design and use a novel data structure that we call the perturbation success probability matrix, which enables us to filter the attack plans and choose the one most likely to succeed. This final attack plan is implemented using a perturbation-bounded adversarial attack algorithm. We compare our zero-query attack against a few-query scheme that repeatedly checks if the victim system is fooled. We also compare against state-of-the-art context-agnostic attacks. Against a context-aware defense, the fooling rate of our zero-query approach is significantly higher than context-agnostic approaches and higher than that achievable with up to three rounds of the fewquery scheme.

----

## [1448] 360-Attack: Distortion-Aware Perturbations from Perspective-Views

**Authors**: *Yunjian Zhang, Yanwei Liu, Jinxia Liu, Jingbo Miao, Antonios Argyriou, Liming Wang, Zhen Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01461](https://doi.org/10.1109/CVPR52688.2022.01461)

**Abstract**:

The application of deep neural networks (DNNs) on 360-degree images has achieved remarkable progress in the recent years. However, DNNs have been demonstrated to be vulnerable to well-crafted adversarial examples, which may trigger severe safety problems in the real-world applications based on 360-degree images. In this paper, we propose an adversarial attack targeting spherical images, called 360-attactk, that transfers adversarial perturbations from perspective-view (PV) images to a final adversarial spherical image. Given a target spherical image, we first represent it with a set of planar PV images, and then perform 2D attacks on them to obtain adversarial PV images. Considering the issue of the projective distortion between spherical and PV images, we propose a distortion-aware attack to reduce the negative impact of distortion on attack. Moreover, to reconstruct the final adversarial spherical image with high aggressiveness, we calculate the spherical saliency map with a novel spherical spectrum method and next propose a saliency-aware fusion strategy that merges multiple inverse perspective projections for the same position on the spherical image. Extensive experimental results show that 360-attack is effective for disturbing spherical images in the black-box setting. Our attack also proves the presence of adversarial transferability from Z2 to SO(3) groups.

----

## [1449] Label-Only Model Inversion Attacks via Boundary Repulsion

**Authors**: *Mostafa Kahla, Si Chen, Hoang Anh Just, Ruoxi Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01462](https://doi.org/10.1109/CVPR52688.2022.01462)

**Abstract**:

Recent studies show that the state-of-the-art deep neural networks are vulnerable to model inversion attacks, in which access to a model is abused to reconstruct private training data of any given target class. Existing attacks rely on having access to either the complete target model (whitebox) or the model's soft-labels (blackbox). However, no prior work has been done in the harder but more practical scenario, in which the attacker only has access to the model's predicted label, without a confidence measure. In this paper, we introduce an algorithm, Boundary-Repelling Model Inversion (BREP-MI), to invert private training data using only the target model's predicted labels. The key idea of our algorithm is to evaluate the model's predicted labels over a sphere and then estimate the direction to reach the target class's centroid. Using the example of face recognition, we show that the images reconstructed by BREP-MI successfully reproduce the semantics of the private training data for various datasets and target model architectures. We compare BREP-MI with the state-of-the-art white-box and blackbox model inversion attacks, and the results show that despite assuming less knowledge about the target model, BREP-MI outperforms the blackbox attack and achieves comparable results to the whitebox attack. Our code is available online.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/m-kahla/Label-Only-Model-Inversion-Attacks-via-Boundary-Repulsion

----

## [1450] Merry Go Round: Rotate a Frame and Fool a DNN

**Authors**: *Daksh Thapar, Aditya Nigam, Chetan Arora*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01463](https://doi.org/10.1109/CVPR52688.2022.01463)

**Abstract**:

A large proportion of videos captured today are first person videos shot from wearable cameras. Similar to other computer vision tasks, Deep Neural Networks (DNNs) are the workhorse for most state-of-the-art (SOTA) egocentric vision techniques. On the other hand DNNs are known to be susceptible to Adversarial Attacks (AAs) which add imperceptible noise to the input. Both black-box, as well as white-box attacks on image as well as video analysis tasks have been shown. We observe that most AA techniques basically add intensity perturbation to an image. Even for videos, the same process is essentially repeated for each frame independently. We note that definition of imperceptibility used for images may not be applicable for videos, where a small intensity change happening randomly in two consecutive frames may still be perceptible. In this paper we make a key novel suggestion to use perturbation in optical flow to carry out AAs on a video analysis system. Such perturbation is especially useful for egocentric videos, because there is lot of shake in the egocentric videos anyways, and adding a little more, keeps it highly imperceptible. In general our idea can be seen as adding structured, parametric noise as the adversarial perturbation. Our implementation of the idea by adding 3D rotations to the frames, reveal that using our technique, one can mount a black-box AA on an egocentric activity detection system in one-third of the queries compared to the SOTA AA technique.

----

## [1451] Cross-Modal Transferable Adversarial Attacks from Images to Videos

**Authors**: *Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01464](https://doi.org/10.1109/CVPR52688.2022.01464)

**Abstract**:

Recent studies have shown that adversarial examples handcrafted on one white-box model can be used to at-tack other black-box models. Such cross-model transferability makes it feasible to perform black-box attacks, which has raised security concerns for real-world DNNs applications. Nevertheless, existing works mostly focus on investigating the adversarial transferability across different deep models that share the same modality of input data. The cross-modal transferability of adversarial perturbation has never been explored. This paper investigates the transferability of adversarial perturbation across different modalities, i.e., leveraging adversarial perturbation generated on white-box image models to attack black-box video models. Specifically, motivated by the observation that the low-level feature space between images and video frames are similar, we propose a simple yet effective cross-modal attack method, named as Image To Video (I2V) attack. I2V generates adversarial frames by minimizing the cosine similarity between features of pretrained image models from adversarial and benign examples, then combines the generated adversarial frames to perform black-box attacks on video recognition models. Extensive experiments demonstrate that I2V can achieve high attack success rates on different black-box video recognition models. On Kinetics-400 and UCF-101, I2V achieves an average attack success rate of 77.88% and 65.68%, respectively, which sheds light on the feasibility of cross-modal adversarial attacks.

----

## [1452] BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning

**Authors**: *Zhenting Wang, Juan Zhai, Shiqing Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01465](https://doi.org/10.1109/CVPR52688.2022.01465)

**Abstract**:

Deep neural networks are vulnerable to Trojan attacks. Existing attacks use visible patterns (e.g., a patch or image transformations) as triggers, which are vulnerable to human inspection. In this paper, we propose stealthy and efficient Trojan attacks, BppAttack. Based on existing biology literature on human visual systems, we propose to use image quantization and dithering as the Trojan trigger, making imperceptible changes. It is a stealthy and efficient attack without training auxiliary models. Due to the small changes made to images, it is hard to inject such triggers during training. To alleviate this problem, we propose a contrastive learning based approach that leverages adversarial attacks to generate negative sample pairs so that the learned trigger is precise and accurate. The proposed method achieves high attack success rates on four benchmark datasets, including MNIST, CIFAR-10, GTSRB, and CelebA. It also effectively bypasses existing Trojan defenses and human inspection. Our code can be found in https://github.com/RU-System-Software-and-Security/BppAttack.

----

## [1453] Investigating Top-k White-Box and Transferable Black-box Attack

**Authors**: *Chaoning Zhang, Philipp Benz, Adil Karjauv, Jae-Won Cho, Kang Zhang, In So Kweon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01466](https://doi.org/10.1109/CVPR52688.2022.01466)

**Abstract**:

Existing works have identified the limitation of top-1 attack success rate (ASR) as a metric to evaluate the attack strength but exclusively investigated it in the white-box setting, while our work extends it to a more practical black-box setting: transferable attack. It is widely reported that stronger I-FGSM transfers worse than simple FGSM, leading to a popular belief that transferability is at odds with the white-box attack strength. Our work challenges this belief with empirical finding that stronger attack actually transfers better for the general top-k ASR indicated by the interest class rank (ICR) after attack. For increasing the attack strength, with an intuitive analysis on the logit gradient from the geometric perspective, we identify that the weakness of the commonly used losses lie in prioritizing the speed to fool the network instead of maximizing its strength. To this end, we propose a new normalized CE loss that guides the logit to be updated in the direction of implicitly maximizing its rank distance from the ground-truth class. Extensive results in various settings have verified that our proposed new loss is simple yet effective for top-k attack. Code is available at: https://bit.ly/3uCiomP

----

## [1454] Boosting Black-Box Attack with Partially Transferred Conditional Adversarial Distribution

**Authors**: *Yan Feng, Baoyuan Wu, Yanbo Fan, Li Liu, Zhifeng Li, Shu-Tao Xia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01467](https://doi.org/10.1109/CVPR52688.2022.01467)

**Abstract**:

This work studies black-box adversarial attacks against deep neural networks (DNNs), where the attacker can only access the query feedback returned by the attacked DNN model, while other information such as model parameters or the training datasets are unknown. One promising approach to improve attack performance is utilizing the adversarial transferability between some white-box surrogate models and the target model (i.e., the attacked model). However, due to the possible differences on model architectures and training datasets between surrogate and target models, dubbed “surrogate biases”, the contribution of adversarial transferability to improving the attack performance may be weakened. To tackle this issue, we innovatively propose a black-box attack method by developing a novel mechanism of adversarial transferability, which is robust to the surrogate biases. The general idea is transferring partial parameters of the conditional adversarial distribution (CAD) of surrogate models, while learning the untransferred parameters based on queries to the target model, to keep the flexibility to adjust the CAD of the target model on any new benign sample. Extensive experiments on benchmark datasets and attacking against real-world API demonstrate the superior attack performance of the proposed method. The code will be available at https://github.com/Kira0096/CGATTACK.

----

## [1455] Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack

**Authors**: *Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01468](https://doi.org/10.1109/CVPR52688.2022.01468)

**Abstract**:

Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments on nearly 50 widely-used defense models demonstrate the effectiveness of our A
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
. By consuming much fewer iterations than existing methods, i.e., 1/10 on average (10× speed up), we achieve lower robust accuracy in all cases. Notably, we won first place out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: https://github.com/liuye6666/adaptive_auto_attack

----

## [1456] Towards Efficient Data Free Blackbox Adversarial Attack

**Authors**: *Jie Zhang, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Lei Zhang, Chao Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01469](https://doi.org/10.1109/CVPR52688.2022.01469)

**Abstract**:

Classic black-box adversarial attacks can take advantage of transferable adversarial examples generated by a similar substitute model to successfully fool the target model. However, these substitute models need to be trained by target models' training data, which is hard to acquire due to privacy or transmission reasons. Recognizing the limited availability of real data for adversarial queries, recent works proposed to train substitute models in a data-free black-box scenario. However, their generative adversarial networks (GANs) based framework suffers from the convergence failure and the model collapse, resulting in low efficiency. In this paper, by rethinking the collaborative relationship between the generator and the substitute model, we design a novel black-box attack framework. The proposed method can efficiently imitate the target model through a small number of queries and achieve high attack success rate. The comprehensive experiments over six datasets demonstrate the effectiveness of our method against the state-of-the-art attacks. Especially, we conduct both label-only and probability-only attacks on the Microsoft Azure online model, and achieve a 100% attack success rate with only 0.46% query budget of the SOTA method [49].

----

## [1457] Masking Adversarial Damage: Finding Adversarial Saliency for Robust and Sparse Network

**Authors**: *Byung-Kwan Lee, Junho Kim, Yong Man Ro*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01470](https://doi.org/10.1109/CVPR52688.2022.01470)

**Abstract**:

Adversarial examples provoke weak reliability and potential security issues in deep neural networks. Although adversarial training has been widely studied to improve adversarial robustness, it works in an over-parameterized regime and requires high computations and large memory budgets. To bridge adversarial robustness and model compression, we propose a novel adversarial pruning method, Masking Adversarial Damage (MAD) that employs second-order information of adversarial loss. By using it, we can accurately estimate adversarial saliency for model parameters and determine which parameters can be pruned without weakening adversarial robustness. Furthermore, we reveal that model parameters of initial layer are highly sensitive to the adversarial examples and show that compressed feature representation retains semantic information for the target objects. Through extensive experiments on three public datasets, we demonstrate that MAD effectively prunes adversarially trained networks without loosing adversarial robustness and shows better performance than previous adversarial pruning methods.

----

## [1458] Certified Patch Robustness via Smoothed Vision Transformers

**Authors**: *Hadi Salman, Saachi Jain, Eric Wong, Aleksander Madry*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01471](https://doi.org/10.1109/CVPR52688.2022.01471)

**Abstract**:

Certified patch defenses can guarantee robustness of an image classifier to arbitrary changes within a bounded contiguous region. But, currently, this robustness comes at a cost of degraded standard accuracies and slower inference times. We demonstrate how using vision transformers enables significantly better certified patch robustness that is also more computationally efficient and does not incur a substantial drop in standard accuracy. These improvements stem from the inherent ability of the vision transformer to gracefully handle largely masked images.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at https://github.com/MadryLab/smoothed-vit..

----

## [1459] Towards Practical Certifiable Patch Defense with Vision Transformer

**Authors**: *Zhaoyu Chen, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Wenqiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01472](https://doi.org/10.1109/CVPR52688.2022.01472)

**Abstract**:

Patch attacks, one of the most threatening forms of physical attack in adversarial examples, can lead networks to induce misclassification by modifying pixels arbitrarily in a continuous region. Certifiable patch defense can guarantee robustness that the classifier is not affected by patch attacks. Existing certifiable patch defenses sacrifice the clean accuracy of classifiers and only obtain a low certified accuracy on toy datasets. Furthermore, the clean and certified accuracy of these methods is still significantly lower than the accuracy of normal classification networks, which limits their application in practice. To move towards a practical certifiable patch defense, we introduce Vision Transformer (ViT) into the framework of Derandomized Smoothing (DS). Specifically, we propose a progressive smoothed image modeling task to train Vision Transformer, which can capture the more discriminable local context of an image while preserving the global semantic information. For efficient inference and deployment in the real world, we innovatively reconstruct the global self-attention structure of the original ViT into isolated band unit self-attention. On ImageNet, under 2% area patch attacks our method achieves 41.70% certified accuracy, a nearly 1-fold increase over the previous best method (26.00%). Simultaneously, our method achieves 78.58% clean accuracy, which is quite close to the normal ResNet-101 accuracy. Extensive experiments show that our method obtains state-of-the-art clean and certified accuracy with inferring efficiently on CIFAR-10 and ImageNet.

----

## [1460] On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles

**Authors**: *Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01473](https://doi.org/10.1109/CVPR52688.2022.01473)

**Abstract**:

Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing.

----

## [1461] 3DeformRS: Certifying Spatial Deformations on Point Clouds

**Authors**: *Gabriel Pérez S., Juan C. Pérez, Motasem Alfarra, Silvio Giancola, Bernard Ghanem*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01474](https://doi.org/10.1109/CVPR52688.2022.01474)

**Abstract**:

3D computer vision models are commonly used in security-critical applications such as autonomous driving and surgical robotics. Emerging concerns over the robustness of these models against real-world deformations must be addressed practically and reliably. In this work, we propose 3DeformRS, a method to certify the robustness of point cloud Deep Neural Networks (DNNs) against real-world deformations. We developed 3DeformRS by building upon recent work that generalized Randomized Smoothing (RS) from pixel-intensity perturbations to vector-field deformations. In particular, we specialized RS to certify DNNs against parameterized deformations (e.g. rotation, twisting), while enjoying practical computational costs. We leverage the virtues of 3DeformRS to conduct a comprehensive empirical study on the certified robustness of four representative point cloud DNNs on two datasets and against seven different deformations. Compared to previous approaches for certifying point cloud DNNs, 3DeformRS is fast, scales well with point cloud size, and provides comparable-to-better certificates. For instance, when certifying a plain PointNet against a 3° z-rotation on 1024-point clouds, 3DeformRS grants a certificate 3× larger and 20× faster than previous work 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code:https://github.com/gaperezsa/3DeformRS.

----

## [1462] Stereoscopic Universal Perturbations across Different Architectures and Datasets

**Authors**: *Zachary Berger, Parth Agrawal, Tian Yu Liu, Stefano Soatto, Alex Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01475](https://doi.org/10.1109/CVPR52688.2022.01475)

**Abstract**:

We study the effect of adversarial perturbations of images on deep stereo matching networks for the disparity estimation task. We present a method to craft a single set of perturbations that, when added to any stereo image pair in a dataset, can fool a stereo network to significantly alter the perceived scene geometry. Our perturbation images are “universal” in that they not only corrupt estimates of the network on the dataset they are optimized for, but also generalize to different architectures trained on different datasets. We evaluate our approach on multiple benchmark datasets where our perturbations can increase the D1-error (akin to fooling rate) of state-of-the-art stereo networks from 1% to as much as 87%. We investigate the effect of perturbations on the estimated scene geometry and identify object classes that are most vulnerable. Our analysis on the activations of registered points between left and right images led us to find architectural components that can increase robustness against adversaries. By simply designing networks with such components, one can reduce the effect of adversaries by up to 60.5%, which rivals the robustness of networks finetuned with costly adversarial data augmentation. Our design principle also improves their robustness against common image corruptions by an average of 70%.

----

## [1463] Aug-NeRF: Training Stronger Neural Radiance Fields with Triple-Level Physically-Grounded Augmentations

**Authors**: *Tianlong Chen, Peihao Wang, Zhiwen Fan, Zhangyang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01476](https://doi.org/10.1109/CVPR52688.2022.01476)

**Abstract**:

Neural Radiance Field (NeRF) regresses a neural param-eterized scene by differentially rendering multi-view images with ground-truth supervision. However, when interpolating novel views, NeRF often yields inconsistent and visually non-smooth geometric results, which we consider as a generalization gap between seen and unseen views. Recent advances in convolutional neural networks have demonstrated the promise of advanced robust data augmentations, either random or learned, in enhancing both in-distribution and out-of-distribution generalization. Inspired by that, we propose Augmented NeRF (Aug-NeRF), which for the first time brings the power of robust data augmentations into regular-izing the NeRF training. Particularly, our proposal learns to seamlessly blend worst-case perturbations into three distinct levels of the NeRF pipeline with physical grounds, including (1) the input coordinates, to simulate imprecise camera parameters at image capture; (2) intermediate features, to smoothen the intrinsic feature manifold; and (3) pre-rendering output, to account for the potential degra-dation factors in the multi-view image supervision. Extensive results demonstrate that Aug-NeRF effectively boosts NeRF performance in both novel view synthesis (up to 1.5dB PSNR gain) and underlying geometry reconstruction. Fur-thermore, thanks to the implicit smooth prior injected by the triple-level augmentations, Aug-NeRF can even recover scenes from heavily corrupted images, a highly challenging setting untackled before. Our codes are available in https://github.com/VITA-Group/Aug-NeRF.

----

## [1464] Bounded Adversarial Attack on Deep Content Features

**Authors**: *Qiuling Xu, Guanhong Tao, Xiangyu Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01477](https://doi.org/10.1109/CVPR52688.2022.01477)

**Abstract**:

We propose a novel adversarial attack targeting content features in some deep layer, that is, individual neurons in the layer. A naive method that enforces a fixed value/percentage bound for neuron activation values can hardly work and generates very noisy samples. The reason is that the level of perceptual variation entailed by a fixed value bound is non-uniform across neurons and even for the same neuron. We hence propose a novel distribution quantile bound for activation values and a polynomial barrier loss function. Given a benign input, a fixed quantile bound is translated to many value bounds, one for each neuron, based on the distributions of the neuron's activations and the current activation value on the given input. These individualized bounds enable fine-grained regulation, allowing content feature mutations with bounded perceptional variations. Our evaluation on ImageNet and five different model architectures demonstrates that our attack is effective. Compared to seven other latest adversarial attacks in both the pixel space and the feature space, our attack can achieve the state-of-the-art trade-off between attack success rate and imperceptibility. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and Samples are available on Github [37].

----

## [1465] DEFEAT: Deep Hidden Feature Backdoor Attacks by Imperceptible Perturbation and Latent Representation Constraints

**Authors**: *Zhendong Zhao, Xiaojun Chen, Yuexin Xuan, Ye Dong, Dakui Wang, Kaitai Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01478](https://doi.org/10.1109/CVPR52688.2022.01478)

**Abstract**:

Backdoor attack is a type of serious security threat to deep learning models. An adversary can provide users with a model trained on poisoned data to manipulate prediction behavior in test stage using a backdoor. The backdoored models behave normally on clean images, yet can be activated and output incorrect prediction if the input is stamped with a specific trigger pattern. Most existing backdoor attacks focus on manually defining imperceptible triggers in input space without considering the abnormality of triggers' latent representations in the poisoned model. These attacks are susceptible to backdoor detection algorithms and even visual inspection. In this paper, We propose a novel and stealthy backdoor attack - DEFEAT. It poisons the clean data using adaptive imperceptible perturbation and restricts latent representation during training process to strengthen our attack's stealthiness and resistance to defense algorithms. We conduct extensive experiments on multiple image classifiers using real-world datasets to demonstrate that our attack can 1) hold against the state-of-the-art defenses, 2) deceive the victim model with high attack success without jeopardizing model utility, and 3) provide practical stealthiness on image data.

----

## [1466] Two Coupled Rejection Metrics Can Tell Adversarial Examples Apart

**Authors**: *Tianyu Pang, Huishuai Zhang, Di He, Yinpeng Dong, Hang Su, Wei Chen, Jun Zhu, Tie-Yan Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01479](https://doi.org/10.1109/CVPR52688.2022.01479)

**Abstract**:

Correctly classifying adversarial examples is an essential but challenging requirement for safely deploying machine learning models. As reported in RobustBench, even the state-of-the-art adversarially trained models struggle to exceed 67% robust test accuracy on CIFAR-10, which is far from practical. A complementary way towards robustness is to introduce a rejection option, allowing the model to not return predictions on uncertain inputs, where confidence is a commonly used certainty proxy. Along with this routine, we find that confidence and a rectified confidence (R-Con) can form two coupled rejection metrics, which could provably distinguish wrongly classified inputs from correctly classified ones. This intriguing property sheds light on using coupling strategies to better detect and reject adversarial examples. We evaluate our rectified rejection (RR) module on CIFAR-10, CIFAR-10-C, and CIFAR-100 under several attacks including adaptive ones, and demonstrate that the RR module is compatible with different adversarial training frameworks on improving robustness, with little extra computation.

----

## [1467] Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness

**Authors**: *Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01480](https://doi.org/10.1109/CVPR52688.2022.01480)

**Abstract**:

Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.

----

## [1468] Improving the Transferability of Targeted Adversarial Examples through Object-Based Diverse Input

**Authors**: *Junyoung Byun, Seungju Cho, Myung-Joon Kwon, Heeseon Kim, Changick Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01481](https://doi.org/10.1109/CVPR52688.2022.01481)

**Abstract**:

The transferability of adversarial examples allows the deception on black-box models, and transfer-based targeted attacks have attracted a lot of interest due to their practical applicability. To maximize the transfer success rate, adversarial examples should avoid overfitting to the source model, and image augmentation is one of the primary approaches for this. However, prior works utilize simple image transformations such as resizing, which limits input diversity. To tackle this limitation, we propose the object-based diverse input (ODI) method that draws an adversarial image on a 3D object and induces the rendered image to be classified as the target class. Our motivation comes from the humans' superior perception of an image printed on a 3D object. If the image is clear enough, humans can recognize the image content in a variety of viewing conditions. Likewise, if an adversarial example looks like the target class to the model, the model should also classify the rendered image of the 3D object as the target class. The ODI method effectively diversifies the input by leveraging an ensemble of multiple source objects and randomizing viewing conditions. In our experimental results on the ImageNet-Compatible dataset, this method boosts the average targeted attack success rate from 28.3% to 47.0% compared to the state-of-the-art methods. We also demonstrate the applicability of the ODI method to adversarial examples on the face verification task and its superior performance improvement. Our code is available at https://github.com/dreamflake/ODI.

----

## [1469] Adversarial Eigen Attack on BlackBox Models

**Authors**: *Linjun Zhou, Peng Cui, Xingxuan Zhang, Yinan Jiang, Shiqiang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01482](https://doi.org/10.1109/CVPR52688.2022.01482)

**Abstract**:

Black-box adversarial attack has aroused much research attention for its difficulty on nearly no available information of the attacked model and the additional constraint on the query budget. A common way to improve attack efficiency is to transfer the gradient information of a white-box substitute model trained on an extra dataset. In this paper, we deal with a more practical setting where a pre-trained white-box model with network parameters is provided without extra training data. To solve the model mismatch problem between the white-box and black-box models, we propose a novel algorithm EigenBA by systematically integrating gradient-based white-box method and zeroth-order optimization in black-box methods. We theoretically show the optimal directions of perturbations for each step are closely related to the right singular vectors of the Jacobian matrix of the pretrained white-box model. Extensive experiments on ImageNet, CIFAR-10 and WebVision show that EigenBA can consistently and significantly outperform state-of-the-art baselines in terms of success rate and attack efficiency.

----

## [1470] Appearance and Structure Aware Robust Deep Visual Graph Matching: Attack, Defense and Beyond

**Authors**: *Qibing Ren, Qingquan Bao, Runzhong Wang, Junchi Yan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01483](https://doi.org/10.1109/CVPR52688.2022.01483)

**Abstract**:

Despite the recent breakthrough of high accuracy deep graph matching (GM) over visual images, the robustness of deep GM models is rarely studied which yet has been revealed an important issue in modern deep nets, ranging from image recognition to graph learning tasks. We first show that an adversarial attack on keypoint localities and the hidden graphs can cause significant accuracy drop to deep GM models. Accordingly, we propose our defense strategy, namely Appearance and Structure Aware Robust Graph Matching (ASAR-GM). Specifically, orthogonal to de facto adversarial training (AT), we devise the Appearance Aware Regularizer (AAR) on those appearance-similar keypoints between graphs that are likely to confuse. Experimental results show that our ASAR-GM achieves better robustness compared to AT. Moreover, our locality attack can serve as a data augmentation technique, which boosts the state-of-the-art GM models even on the clean test dataset. Code is available at https://github.com/Thinklab-SJTU/RobustMatch.

----

## [1471] Enhancing Adversarial Training with Second-Order Statistics of Weights

**Authors**: *Gaojie Jin, Xinping Yi, Wei Huang, Sven Schewe, Xiaowei Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01484](https://doi.org/10.1109/CVPR52688.2022.01484)

**Abstract**:

Adversarial training has been shown to be one of the most effective approaches to improve the robustness of deep neural networks. It is formalized as a min-max optimization over model weights and adversarial perturbations, where the weights can be optimized through gradient descent methods like SGD. In this paper, we show that treating model weights as random variables allows for enhancing adversarial training through Second-Order Statistics Optimization (S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
O) with respect to the weights. By relaxing a common (but unrealistic) assumption of previous PAC-Bayesian frameworks that all weights are statistically independent, we derive an improved PAC-Bayesian adversarial generalization bound, which suggests that optimizing second-order statistics of weights can effectively tighten the bound. In addition to this theoretical insight, we conduct an extensive set of experiments, which show that S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
O not only improves the robustness and generalization of the trained neural networks when used in isolation, but also integrates easily in state-of-the-art adversarial training techniques like TRADES, AWP, MART, and AVMixup, leading to a measurable improvement of these techniques. The code is available at https://github.com/Alexkael/S2O.

----

## [1472] Towards Data-Free Model Stealing in a Hard Label Setting

**Authors**: *Sunandini Sanyal, Sravanti Addepalli, R. Venkatesh Babu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01485](https://doi.org/10.1109/CVPR52688.2022.01485)

**Abstract**:

Machine learning models deployed as a service (MLaaS) are susceptible to model stealing attacks, where an adversary attempts to steal the model within a restricted access framework. While existing attacks demonstrate near-perfect clone-model performance using softmax predictions of the classification network, most of the APIs allow access to only the top-1 labels. In this work, we show that it is indeed possible to steal Machine Learning models by accessing only top-1 predictions (Hard Label setting) as well, without access to model gradients (Black-Box setting) or even the training dataset (Data-Free setting) within a low query budget. We propose a novel GAN-based framework
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project Page: https://sites.google.com/view/dfms-hl that trains the student and generator in tandem to steal the model effectively while overcoming the challenge of the hard label setting by utilizing gradients of the clone network as a proxy to the victim's gradients. We propose to overcome the large query costs associated with a typical Data-Free setting by utilizing publicly available (potentially unrelated) datasets as a weak image prior. We additionally show that even in the absence of such data, it is possible to achieve state-of-the-art results within a low query budget using synthetically crafted samples. We are the first to demonstrate the scalability of Model Stealing in a restricted access setting on a 100 class dataset as well.

----

## [1473] Robust Structured Declarative Classifiers for 3D Point Clouds: Defending Adversarial Attacks with Implicit Gradients

**Authors**: *Kaidong Li, Ziming Zhang, Cuncong Zhong, Guanghui Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01486](https://doi.org/10.1109/CVPR52688.2022.01486)

**Abstract**:

Deep neural networks for 3D point cloud classification, such as PointNet, have been demonstrated to be vulnerable to adversarial attacks. Current adversarial defenders often learn to denoise the (attacked) point clouds by reconstruction, and then feed them to the classifiers as input. In contrast to the literature, we propose a family of robust structured declarative classifiers for point cloud classification, where the internal constrained optimization mechanism can effectively defend adversarial attacks through implicit gradients. Such classifiers can be formulated using a bilevel optimization framework. We further propose an effective and efficient instantiation of our approach, namely, Lattice Point Classifier (LPC), based on structured sparse coding in the permutohedral lattice and 2D convolutional neural networks (CNNs) that is end-to-end trainable. We demonstrate state-of-the-art robust point cloud classification performance on ModelNet40 and ScanNet under seven different attackers. For instance, we achieve 89.51% and 83.16% test accuracy on each dataset under the recent JGBA attacker that outperforms DUP-Net and IF-Defense with PointNet by ~70%. The demo code is available at https://zhang-vislab.github.io.

----

## [1474] DTA: Physical Camouflage Attacks using Differentiable Transformation Network

**Authors**: *Naufal Suryanto, Yongsu Kim, Hyoeun Kang, Harashta Tatimma Larasati, Youngyeo Yun, Thi-Thu-Huong Le, Hunmin Yang, Se-Yoon Oh, Howon Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01487](https://doi.org/10.1109/CVPR52688.2022.01487)

**Abstract**:

To perform adversarial attacks in the physical world, many studies have proposed adversarial camouflage, a method to hide a target object by applying camouflage patterns on 3D object surfaces. For obtaining optimal physical adversarial camouflage, previous studies have utilized the so-called neural renderer, as it supports differentiability. However, existing neural renderers cannot fully represent various real-world transformations due to a lack of control of scene parameters compared to the legacy photo-realistic renderers. In this paper, we propose the Differentiable Transformation Attack (DTA), a framework for generating a robust physical adversarial pattern on a target object to camouflage it against object detection models with a wide range of transformations. It utilizes our novel Differentiable Transformation Network (DTN), which learns the expected transformation of a rendered object when the texture is changed while preserving the original properties of the target object. Using our attack framework, an adversary can gain both the advantages of the legacy photo-realistic renderers including various physical-world transformations and the benefit of white-box access by offering differentiability. Our experiments show that our camouflaged 3D vehicles can successfully evade state-of-the-art object detection models in the photo-realistic environment (i.e., CARLA on Unreal Engine). Furthermore, our demonstration on a scaled Tesla Model 3 proves the applicability and transferability of our method to the real world.

----

## [1475] Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity

**Authors**: *Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01488](https://doi.org/10.1109/CVPR52688.2022.01488)

**Abstract**:

Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at https://github.com/LinQinLiang/SSAH-adversarial-attack.

----

## [1476] Enhancing Adversarial Robustness for Deep Metric Learning

**Authors**: *Mo Zhou, Vishal M. Patel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01489](https://doi.org/10.1109/CVPR52688.2022.01489)

**Abstract**:

Owing to security implications of adversarial vulnerability, adversarial robustness of deep metric learning models has to be improved. In order to avoid model collapse due to excessively hard examples, the existing defenses dismiss the min-max adversarial training, but instead learn from a weak adversary inefficiently. Conversely, we propose Hardness Manipulation to efficiently perturb the training triplet till a specified level of hardness for adversarial training, according to a harder benign triplet or a pseudo-hardness function. It is flexible since regular training and min-max adversarial training are its boundary cases. Besides, Gradual Adversary, a family of pseudo-hardness functions is proposed to gradually increase the specified hardness level during training for a better balance between performance and robustness. Additionally, an Intra-Class Structure loss term among benign and adversarial examples further improves model robust-ness and efficiency. Comprehensive experimental results suggest that the proposed method, although simple in its form, overwhelmingly outperforms the state-of-the-art de-fenses in terms of robustness, training efficiency, as well as performance on benign examples.

----

## [1477] Shape-invariant 3D Adversarial Point Clouds

**Authors**: *Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01490](https://doi.org/10.1109/CVPR52688.2022.01490)

**Abstract**:

Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an “implicit constrain” like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to constrain its perturbation with a simple loss or metric properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an “explicit constrain” instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.

----

## [1478] Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon

**Authors**: *Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01491](https://doi.org/10.1109/CVPR52688.2022.01491)

**Abstract**:

Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the “sticker-pasting” strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at https://github.com/hncszyq/ShadowAttack.

----

## [1479] Exploring Effective Data for Surrogate Training Towards Black-box Attack

**Authors**: *Xuxiang Sun, Gong Cheng, Hongda Li, Lei Pei, Junwei Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01492](https://doi.org/10.1109/CVPR52688.2022.01492)

**Abstract**:

Without access to the training data where a black-box victim model is deployed, training a surrogate model for black-box adversarial attack is still a struggle. In terms of data, we mainly identify three key measures for effective surrogate training in this paper. First, we show that leveraging the loss introduced in this paper to enlarge the inter-class similarity makes more sense than enlarging the inter-class diversity like existing methods. Next, unlike the approaches that expand the intra-class diversity in an implicit model-agnostic fashion, we propose a loss function specific to the surrogate model for our generator to enhance the intra-class diversity. Finally, in accordance with the in-depth observations for the methods based on proxy data, we argue that leveraging the proxy data is still an effective way for surrogate training. To this end, we propose a triple-player framework by introducing a discriminator into the traditional data-free framework. In this way, our method can be competitive when there are few semantic overlaps between the scarce proxy data (with the size between 1 k and 5k) and the training data. We evaluate our method on a range of victim models and datasets. The extensive results witness the effectiveness of our method. Our source code is available at https://github.com/xuxiangsun/ST-Data.

----

## [1480] NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models

**Authors**: *Simin Chen, Zihe Song, Mirazul Haque, Cong Liu, Wei Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01493](https://doi.org/10.1109/CVPR52688.2022.01493)

**Abstract**:

Neural image caption generation (NICG) models have received massive attention from the research community due to their excellent performance in visual understanding. Existing work focuses on improving NICG model ac-curacy while efficiency is less explored. However, many real-world applications require real-time feedback, which highly relies on the efficiency of NICG models. Recent re-search observed that the efficiency of NICG models could vary for different inputs. This observation brings in a new attack surface of NICG models, i.e., An adversary might be able to slightly change inputs to cause the NICG mod-els to consume more computational resources. To further understand such efficiency-oriented threats, we propose a new attack approach, NICGSlowDown, to evaluate the ef-ficiency robustness of NICG models. Our experimental re-sults show that NICGSlowDown can generate images with human-unnoticeable perturbations that will increase the NICG model latency up to 483.86%. We hope this research could raise the community's concern about the efficiency robustness of NICG models.

----

## [1481] Dual-Key Multimodal Backdoors for Visual Question Answering

**Authors**: *Matthew Walmer, Karan Sikka, Indranil Sur, Abhinav Shrivastava, Susmit Jha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01494](https://doi.org/10.1109/CVPR52688.2022.01494)

**Abstract**:

The success of deep learning has enabled advances in multimodal tasks that require non-trivial fusion of multiple input domains. Although multimodal models have shown potential in many problems, their increased complexity makes them more vulnerable to attacks. A Backdoor (or Trojan) attack is a class of security vulnerability wherein an attacker embeds a malicious secret behavior into a network (e.g. targeted misclassification) that is activated when an attacker-specified trigger is added to an input. In this work, we show that multimodal networks are vulnerable to a novel type of attack that we refer to as Dual-Key Multimodal Backdoors. This attack exploits the complex fusion mechanisms used by state-of-the-art networks to embed backdoors that are both effective and stealthy. Instead of using a single trigger, the proposed attack embeds a trigger in each of the input modalities and activates the malicious behavior only when both the triggers are present. We present an extensive study of multimodal backdoors on the Visual Question Answering (VQA) task with multiple architectures and visual feature backbones. A major challenge in embedding backdoors in VQA models is that most models use visual features extracted from a fixed pretrained object detector. This is challenging for the attacker as the detector can distort or ignore the visual trigger entirely, which leads to models where backdoors are over-reliant on the language trigger. We tackle this problem by proposing a visual trigger optimization strategy designed for pretrained object detectors. Through this method, we create Dual-Key Backdoors with over a 98% attack success rate while only poisoning 1% of the training data. Finally, we release TrojVQA, a large collection of clean and trojan VQA models to enable research in defending against multimodal backdoors.

----

## [1482] Proactive Image Manipulation Detection

**Authors**: *Vishal Asnani, Xi Yin, Tal Hassner, Sijia Liu, Xiaoming Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01495](https://doi.org/10.1109/CVPR52688.2022.01495)

**Abstract**:

Image manipulation detection algorithms are often trained to discriminate between images manipulated with particular Generative Models (GMs) and genuine/real images, yet generalize poorly to images manipulated with GMs unseen in the training. Conventional detection algorithms receive an input image passively. By contrast, we propose a proactive scheme to image manipulation detection. Our key enabling technique is to estimate a set of templates which when added onto the real image would lead to more accurate manipulation detection. That is, a template protected real image, and its manipulated version, is better discriminated compared to the original real image vs. its manipulated one. These templates are estimated using certain constraints based on the desired properties of templates. For image manipulation detection, our proposed approach outperforms the prior work by an average precision of 16%for CycleGAN and 32% for GauGAN. Our approach is generalizable to a variety of GMs showing an improvement over prior work by an average precision of 10% averaged across 12 GMs. Our code is available at https://www.github.com/vishal3477/proactive_IMD.

----

## [1483] ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts

**Authors**: *Bingqian Lin, Yi Zhu, Zicong Chen, Xiwen Liang, Jianzhuang Liu, Xiaodan Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01496](https://doi.org/10.1109/CVPR52688.2022.01496)

**Abstract**:

Vision-Language Navigation (VLN) is a challenging task that requires an embodied agent to perform action-level modality alignment, i.e., make instruction-asked actions sequentially in complex visual environments. Most existing VLN agents learn the instruction-path data directly and cannot sufficiently explore action-level alignment knowledge inside the multi-modal inputs. In this paper, we propose modAlity-aligneD Action PrompTs (ADAPT), which provides the VLN agent with action prompts to enable the explicit learning of action-level modality alignment to pursue successful navigation. Specifically, an action prompt is defined as a modality-aligned pair of an image sub-prompt and a text sub-prompt, where the former is a single-view observation and the latter is a phrase like “walk past the chair”. When starting navigation, the instruction-related action prompt set is retrieved from a prebuilt action prompt base and passed through a prompt encoder to obtain the prompt feature. Then the prompt feature is concatenated with the original instruction feature and fed to a multilayer transformer for action prediction. To collect high-quality action prompts into the prompt base, we use the Contrastive Language-Image Pretraining (CLIP) model which has powerful cross-modality alignment ability. A modality alignment loss and a sequential consistency loss are further introduced to enhance the alignment of the action prompt and enforce the agent to focus on the related prompt sequentially. Experimental results on both R2R and RxR show the superiority of ADAPT over state-of-the-art methods.

----

## [1484] Envedit: Environment Editing for Vision-and-Language Navigation

**Authors**: *Jialu Li, Hao Tan, Mohit Bansal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01497](https://doi.org/10.1109/CVPR52688.2022.01497)

**Abstract**:

In Vision-and-Language Navigation (VLN), an agent needs to navigate through the environment based on nat-ural language instructions. Due to limited available data for agent training and finite diversity in navigation environments, it is challenging for the agent to generalize to new, unseen environments. To address this problem, we propose Envedit, a data augmentation method that cre-ates new environments by editing existing environments, which are used to train a more generalizable agent. Our augmented environments can differ from the seen environ-ments in three diverse aspects: style, object appearance, and object classes. Training on these edit-augmented environments prevents the agent from overfitting to existing en-vironments and helps generalize better to new, unseen en-vironments. Empirically, on both the Room-to-Room and the multi-lingual Room-Across-Room datasets, we show that our proposed Envedit method gets significant im-provements in all metrics on both pre-trained and non-pre-trained VLN agents, and achieves the new state-of-the-art on the test leaderboard. We further ensemble the VLN agents augmented on different edited environments and show that these edit methods are complementary.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and data are available at https://github.com/jialuli-luka/EnvEdit.

----

## [1485] HOP: History-and-Order Aware Pretraining for Vision-and-Language Navigation

**Authors**: *Yanyuan Qiao, Yuankai Qi, Yicong Hong, Zheng Yu, Peng Wang, Qi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01498](https://doi.org/10.1109/CVPR52688.2022.01498)

**Abstract**:

Pretraining has been adopted in a few of recent works for Vision-and-Language Navigation (VLN). However, pre-vious pre-training methods for VLN either lack the ability to predict future actions or ignore the trajectory contexts, which are essential for a greedy navigation process. In this work, to promote the learning of spatio-temporal visual-textual correspondence as well as the agent's capability of decision making, we propose a novel history-and-order aware pre-training paradigm (HOP) with VLN-specific objectives that exploit the past observations and support future action prediction. Specifically, in addition to the commonly used Masked Language Modeling (MLM) and Trajectory-Instruction Matching (TIM), we design two proxy tasks to model temporal order information: Trajectory Order Modeling (TOM) and Group Order Modeling (GOM). Moreover, our navigation action prediction is also enhanced by intro-ducing the task of Action Prediction with History (APH), which takes into account the history visual perceptions. Extensive experimental results on four downstream VLN tasks (R2R, REVERIE, NDH, RxR) demonstrate the effectiveness of our proposed method compared against several state-of-the-art agents.

----

## [1486] Less is More: Generating Grounded Navigation Instructions from Landmarks

**Authors**: *Su Wang, Ceslee Montgomery, Jordi Orbay, Vighnesh Birodkar, Aleksandra Faust, Izzeddin Gur, Natasha Jaques, Austin Waters, Jason Baldridge, Peter Anderson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01499](https://doi.org/10.1109/CVPR52688.2022.01499)

**Abstract**:

We study the automatic generation of navigation instructions from 360째 images captured on indoor routes. Existing generators suffer from poor visual grounding, causing them to rely on language priors and hallucinate objects. Our Marky-mt5 system addresses this by focusing on visual landmarks; it comprises a first stage landmark detector and a second stage generator-a multimodal, multilingual, multi-task encoder-decoder. To train it, we bootstrap grounded landmark annotations on top of the Room-across-Room (RxR) dataset. Using text parsers, weak supervision from RxR's pose traces, and a multilingual image-text encoder trained on 1.8b images, we identify 971k English, Hindi and Telugu landmark descriptions and ground them to specific regions in panoramas. On Room-to-Room, human wayfind-ers obtain success rates (SR) of 71% following Marky-mt5's instructions, just shy of their 75% SR following human instructions-and well above SRs with other genera-tors. Evaluations on RxR's longer, diverse paths obtain 61-64% SRs on three languages. Generating such high-quality navigation instructions in novel environments is a step to-wards conversational navigation tools and could facilitate larger-scale training of instruction-following agents.

----

## [1487] Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation

**Authors**: *Yicong Hong, Zun Wang, Qi Wu, Stephen Gould*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01500](https://doi.org/10.1109/CVPR52688.2022.01500)

**Abstract**:

Most existing works in vision-and-language navigation (VLN) focus on either discrete or continuous environments, training agents that cannot generalize across the two. Although learning to navigate in continuous spaces is closer to the real-world, training such an agent is significantly more difficult than training an agent in discrete spaces. However, recent advances in discrete VLN are challenging to translate to continuous VLN due to the domain gap. The fundamental difference between the two setups is that discrete navigation assumes prior knowledge of the connectivity graph of the environment, so that the agent can effectively transfer the problem of navigation with low-level controls to jumping from node to node with high-level actions by grounding to an image of a navigable direction. To bridge the discrete-to-continuous gap, we propose a predictor to generate a set of candidate waypoints during navigation, so that agents designed with high-level actions can be transferred to and trained in continuous environments. We refine the connectivity graph of Matterport3D to fit the continuous Habitat-Matterport3D, and train the waypoints predictor with the refined graphs to produce accessible waypoints at each time step. Moreover, we demonstrate that the predicted waypoints can be augmented during training to diversify the views and paths, and therefore enhance agent's generalization ability. Through extensive experiments we show that agents navigating in continuous environments with predicted waypoints perform significantly better than agents using low-level actions, which reduces the absolute discrete-to-continuous gap by 11.76% Success Weighted by Path Length (SPL) for the Cross-Modal Matching Agent and 18.24% SPL for the VLN$$BERT. Our agents, trained with a simple imitation learning objective, outperform previous methods by a large margin, achieving new state-of-the-art results on the testing environments of the R2R-CE and the RxR-CE datasets.

----

## [1488] Reinforced Structured State-Evolution for Vision-Language Navigation

**Authors**: *Jinyu Chen, Chen Gao, Erli Meng, Qiong Zhang, Si Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01501](https://doi.org/10.1109/CVPR52688.2022.01501)

**Abstract**:

Vision-and-language Navigation (VLN) task requires an embodied agent to navigate to a remote location following a natural language instruction. Previous methods usually adopt a sequence model (e.g., Transformer and LSTM) as the navigator. In such a paradigm, the sequence model predicts action at each step through a maintained navigation state, which is generally represented as a one-dimensional vector. However, the crucial navigation clues (i.e., object-level environment layout) for embodied navigation task is discarded since the maintained vector is essentially unstructured. In this paper, we propose a novel Structured state-Evolution (SEvol) model to effectively maintain the environment layout clues for VLN. Specifically, we utilise the graph-based feature to represent the navigation state instead of the vector-based state. Accordingly, we devise a Reinforced Layout clues Miner (RLM) to mine and detect the most crucial layout graph for long-term navigation via a customised reinforcement learning strategy. Moreover, the Structured Evolving Module (SEM) is proposed to maintain the structured graph-based state during navigation, where the state is gradually evolved to learn the object-level spatial-temporal relationship. The experiments on the R2R and R4R datasets show that the proposed SEvol model improves VLN models' performance by large margins, e.g., +3% absolute SPL accuracy for NvEM and +8% for EnvDrop on the R2R test set.

----

## [1489] Cross-modal Map Learning for Vision and Language Navigation

**Authors**: *Georgios Georgakis, Karl Schmeckpeper, Karan Wanchoo, Soham Dan, Eleni Miltsakaki, Dan Roth, Kostas Daniilidis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01502](https://doi.org/10.1109/CVPR52688.2022.01502)

**Abstract**:

We consider the problem of Vision-and-Language Navigation (VLN). The majority of current methods for VLN are trained end-to-end using either unstructured memory such as LSTM, or using cross-modal attention over the egocentric observations of the agent. In contrast to other works, our key insight is that the association between language and vision is stronger when it occurs in explicit spatial representations. In this work, we propose a cross-modal map learning model for vision-and-language navigation that first learns to predict the top-down semantics on an egocentric map for both observed and unobserved regions, and then predicts a path towards the goal as a set of way-points. In both cases, the prediction is informed by the language through cross-modal attention mechanisms. We experimentally test the basic hypothesis that language-driven navigation can be solved given a map, and then show competitive results on the full VLN-CE benchmark.

----

## [1490] Counterfactual Cycle-Consistent Learning for Instruction Following and Generation in Vision-Language Navigation

**Authors**: *Hanqing Wang, Wei Liang, Jianbing Shen, Luc Van Gool, Wenguan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01503](https://doi.org/10.1109/CVPR52688.2022.01503)

**Abstract**:

Since the rise of vision-language navigation (VLN), great progress has been made in instruction following - building a follower to navigate environments under the guidance of instructions. However, far less attention has been paid to the inverse task: instruction generation - learning a speaker to generate grounded descriptions for navigation routes. Existing VLN methods train a speaker independently and often treat it as a data augmentation tool to strengthen the follower, while ignoring rich cross-task relations. Here we describe an approach that learns the two tasks simultaneously and exploits their intrinsic correlations to boost the training of each: the follower judges whether the speaker-created instruction explains the original navigation route correctly, and vice versa. Without the need of aligned instruction-path pairs, such cycle-consistent learning scheme is complementary to task-specific training targets defined on labeled data, and can also be applied over unlabeled paths (sampled without paired instructions). Another agent, called creator is added to generate counterfactual environments. It greatly changes current scenes yet leaves novel items - which are vital for the execution of original instructions - unchanged. Thus more informative training scenes are synthesized and the three agents compose a powerful VLN learning system. Extensive experiments on a standard benchmark show that our approach improves the performance of various follower models and produces accurate navigation instructions.

----

## [1491] One Step at a Time: Long-Horizon Vision-and-Language Navigation with Milestones

**Authors**: *Chan Hee Song, Jihyung Kil, Tai-Yu Pan, Brian M. Sadler, Wei-Lun Chao, Yu Su*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01504](https://doi.org/10.1109/CVPR52688.2022.01504)

**Abstract**:

We study the problem of developing autonomous agents that can follow human instructions to infer and perform a sequence of actions to complete the underlying task. Significant progress has been made in recent years, especially for tasks with short horizons. However, when it comes to long-horizon tasks with extended sequences of actions, an agent can easily ignore some instructions or get stuck in the middle of the long instructions and eventually fail the task. To address this challenge, we propose a modelagnostic milestone-based task tracker (M-TRACK) to guide the agent and monitor its progress. Specifically, we propose a milestone builder that tags the instructions with navigation and interaction milestones which the agent needs to complete step by step, and a milestone checker that systemically checks the agent's progress in its current milestone and determines when to proceed to the next. On the challenging ALFRED dataset, our M-Track leads to a notable 33% and 52% relative improvement in unseen success rate over two competitive base models.

----

## [1492] Expanding Large Pre-trained Unimodal Models with Multimodal Information Injection for Image-Text Multimodal Classification

**Authors**: *Tao Liang, Guosheng Lin, Mingyang Wan, Tianrui Li, Guojun Ma, Fengmao Lv*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01505](https://doi.org/10.1109/CVPR52688.2022.01505)

**Abstract**:

Fine-tuning pre-trained models for downstream tasks is mainstream in deep learning. However, the pre-trained models are limited to be fine-tuned by data from a specific modality. For example, as a visual model, DenseNet cannot directly take the textual data as its input. Hence, although the large pre-trained models such as DenseNet or BERT have a great potential for the downstream recognition tasks, they have weaknesses in leveraging multimodal information, which is a new trend of deep learning. This work focuses on fine-tuning pre-trained unimodal models with multimodal inputs of image-text pairs and expanding them for image-text multimodal recognition. To this end, we propose the Multimodal Information Injection Plug-in (MI2P) which is attached to different layers of the unimodal models (e.g., DenseNet and BERT). The proposed MI2P unit provides the path to integrate the information of other modalities into the unimodal models. Specifically, MI2P performs cross-modal feature transformation by learning the fine-grained correlations between the visual and textual features. Through the proposed MI2P unit, we can inject the language information into the vision backbone by attending the word-wise textual features to different visual channels, as well as inject the visual information into the language backbone by attending the channel-wise visual features to different textual words. Armed with the MI2P attachments, the pre-trained unimodal models can be expanded to process multimodal data without the need to change the network structures.

----

## [1493] Shifting More Attention to Visual Backbone: Query-modulated Refinement Networks for End-to-End Visual Grounding

**Authors**: *Jiabo Ye, Junfeng Tian, Ming Yan, Xiaoshan Yang, Xuwu Wang, Ji Zhang, Liang He, Xin Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01506](https://doi.org/10.1109/CVPR52688.2022.01506)

**Abstract**:

Visual grounding focuses on establishing fine-grained alignment between vision and natural language, which has essential applications in multimodal reasoning systems. Existing methods use pre-trained query-agnostic visual backbones to extract visual feature maps independently without considering the query information. We argue that the visual features extracted from the visual backbones and the features really needed for multimodal reasoning are inconsistent. One reason is that there are differences between pre-training tasks and visual grounding. Moreover, since the backbones are query-agnostic, it is difficult to completely avoid the inconsistency issue by training the visual backbone end-to-end in the visual grounding framework. In this paper, we propose a Query-modulated Refinement Network (QRNet) to address the inconsistent issue by adjusting intermediate features in the visual backbone with a novel Query-aware Dynamic Attention (QD-ATT) mechanism and query-aware multiscale fusion. The QD-ATT can dynamically compute query-dependent visual attention at the spatial and channel levels of the feature maps produced by the visual backbone. We apply the QRNet to an end-to-end visual grounding framework. Extensive experiments show that the proposed method outperforms state-of-the-art methods on five widely used datasets. Our code is available at https://github.com/LukeForeverYoung/QRNet.

----

## [1494] Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding

**Authors**: *Haojun Jiang, Yuanze Lin, Dongchen Han, Shiji Song, Gao Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01507](https://doi.org/10.1109/CVPR52688.2022.01507)

**Abstract**:

Visual grounding, i.e., localizing objects in images ac-cording to natural language queries, is an important topic in visual language understanding. The most effective approaches for this task are based on deep learning, which generally require expensive manually labeled image-query or patch-query pairs. To eliminate the heavy depen-dence on human annotations, we present a novel method, named Pseudo-Q, to automatically generate pseudo language queries for supervised training. Our method lever-ages an off-the-shelf object detector to identify visual ob-jects from unlabeled images, and then language queries for these objects are obtained in an unsupervised fashion with a pseudo-query generation module. Then, we design a task-related query prompt module to specifically tailor generated pseudo language queries for visual grounding tasks. Further, in order to fully capture the contextual re-lationships between images and language queries, we de-velop a visual-language model equipped with multi-level cross-modality attention mechanism. Extensive experimen-tal results demonstrate that our method has two notable benefits: (1) it can reduce human annotation costs signifi-cantly, e.g., 31% on Ref Coco [65] without degrading orig-inal model's performance under the fully supervised set-ting, and (2) without bells and whistles, it achieves supe-rior or comparable performance compared to state-of-the-art weakly-supervised visual grounding methods on all the five datasets we have experimented. Code is available at https://github.com/LeapLabTHU/Pseudo-Q.

----

## [1495] Multi-View Transformer for 3D Visual Grounding

**Authors**: *Shijia Huang, Yilun Chen, Jiaya Jia, Liwei Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01508](https://doi.org/10.1109/CVPR52688.2022.01508)

**Abstract**:

The 3D visual grounding task aims to ground a natural language description to the targeted object in a 3D scene, which is usually represented in 3D point clouds. Previous works studied visual grounding under specific views. The vision-language correspondence learned by this way can easily fail once the view changes. In this paper, we propose a Multi-View Transformer (MVT) for 3D visual grounding. We project the 3D scene to a multi-view space, in which the position information of the 3D scene under different views are modeled simultaneously and aggregated together. The multi-view space enables the network to learn a more robust multi-modal representation for 3D visual grounding and eliminates the dependence on specific views. Extensive experiments show that our approach significantly outperforms all state-of-the-art methods. Specifically, on Nr3D and Sr3D datasets, our method outperforms the best competitor by 11.2% and 7.1% and even surpasses recent work with extra 2D assistance by 5.9% and 6.6%. Our code is available at https://github.com/sega-hsj/MVT-3DVG.

----

## [1496] Multi-Modal Dynamic Graph Transformer for Visual Grounding

**Authors**: *Sijia Chen, Baochun Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01509](https://doi.org/10.1109/CVPR52688.2022.01509)

**Abstract**:

Visual grounding (VG) aims to align the correct regions of an image with a natural language query about that image. We found that existing VG methods are trapped by the single-stage grounding process that performs a sole evaluate-and-rank for meticulously prepared regions. Their performance depends on the density and quality of the candidate regions, and is capped by the inability to optimize the located regions continuously. To address these issues, we propose to remodel VG into a progressively optimized visual semantic alignment process. Our proposed multi-modal dynamic graph transformer (M-DGT) achieves this by building upon the dynamic graph structure with regions as nodes and their semantic relations as edges. Starting from a few randomly initialized regions, M-DGT is able to make sustainable adjustments (i.e., 2D spatial transformation and deletion) to the nodes and edges of the graph based on multi-modal information and the graph feature, thereby efficiently shrinking the graph to approach the ground truth regions. Experiments show that with an average of 48 boxes as initialization, the performance of M-DGT on the Flickr30k Entities and RefCOCO datasets outperforms existing state-of-the-art methods by a substantial margin, in terms of both accuracy and Intersect over Union (IOU) scores. Furthermore, introducing M-DGT to optimize the predicted regions of existing methods can further significantly improve their performance. The source codes are available at https://github.com/iQua/M-DGT.

----

## [1497] Weakly-Supervised Generation and Grounding of Visual Descriptions with Conditional Generative Models

**Authors**: *Effrosyni Mavroudi, René Vidal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01510](https://doi.org/10.1109/CVPR52688.2022.01510)

**Abstract**:

Given weak supervision from image- or video-caption pairs, we address the problem of grounding (localizing) each object word of a ground-truth or generated sentence describing a visual input. Recent weakly-supervised approaches leverage region proposals and ground words based on the region attention coefficients of captioning models. To predict each next word in the sentence they attend over regions using a summary of the previous words as a query, and then ground the word by selecting the most attended regions. However, this leads to sub-optimal grounding, since attention coefficients are computed without taking into account the word that needs to be localized. To address this shortcoming, we propose a novel Grounded Visual Description Conditional Variational Autoencoder (GVD-CVAE) and leverage its latent variables for grounding. In particular, we introduce a discrete random variable that models each word-to-region alignment, and learn its approximate posterior distribution given the full sentence. Experiments on challenging image and video datasets (Flickr30k Entities, YouCook2, ActivityNet Entities) validate the effectiveness of our conditional generative model, showing that it can substantially outperform soft-attention-based baselines in grounding.

----

## [1498] Weakly Supervised Temporal Sentence Grounding with Gaussian-based Contrastive Proposal Learning

**Authors**: *Minghang Zheng, Yanjie Huang, Qingchao Chen, Yuxin Peng, Yang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01511](https://doi.org/10.1109/CVPR52688.2022.01511)

**Abstract**:

Temporal sentence grounding aims to detect the most salient moment corresponding to the natural language query from untrimmed videos. As labeling the temporal boundaries is labor-intensive and subjective, the weakly- supervised methods have recently received increasing attention. Most of the existing weakly-supervised methods gen-erate the proposals by sliding windows, which are content- independent and of low quality. Moreover, they train their model to distinguish positive visual-language pairs from negative ones randomly collected from other videos, ignoring the highly confusing video segments within the same video. In this paper, we propose Contrastive Proposal Learning(CPL) to overcome the above limitations. Specifi-cally, we use multiple learnable Gaussian functions to gen-erate both positive and negative proposals within the same video that can characterize the multiple events in a long video. Then, we propose a controllable easy to hard neg-ative proposal mining strategy to collect negative samples within the same video, which can ease the model opti-mization and enables CPL to distinguish highly confusing scenes. The experiments show that our method achieves state-of-the-art performance on Charades-STA and Activi-tyNet Captions datasets. The code and models are available at https://github.com/minghangz/cpl.

----

## [1499] Visual Abductive Reasoning

**Authors**: *Chen Liang, Wenguan Wang, Tianfei Zhou, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01512](https://doi.org/10.1109/CVPR52688.2022.01512)

**Abstract**:

Abductive reasoning seeks the likeliest possible explanation for partial observations. Although abduction is frequently employed in human daily reasoning, it is rarely explored in computer vision literature. In this paper, we propose a new task and dataset, Visual Abductive Reasoning (VAR), for examining abductive reasoning ability of machine intelligence in everyday visual situations. Given an incomplete set of visual events, AI systems are required to not only describe what is observed, but also infer the hypothesis that can best explain the visual premise. Based on our large-scale VAR dataset, we devise a strong baseline model, REASONER (causal-and-cascaded reasoning Transformer). First, to capture the causal structure of the observations, a contextualized directional position embedding strategy is adopted in the encoder, that yields discriminative represen-tations for the premise and hypothesis. Then, multiple de-coders are cascaded to generate and progressively refine the premise and hypothesis sentences. The prediction scores of the sentences are used to guide cross-sentence information flow in the cascaded reasoning procedure. Our VAR bench-marking results show that REASONER surpasses many famous video-language models, while still being far behind human performance. This work is expected to foster future efforts in the reasoning-beyond-observation paradigm.

----

## [1500] Query and Attention Augmentation for Knowledge-Based Explainable Reasoning

**Authors**: *Yifeng Zhang, Ming Jiang, Qi Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01513](https://doi.org/10.1109/CVPR52688.2022.01513)

**Abstract**:

Explainable visual question answering (VQA) models have been developed with neural modules and query-based knowledge incorporation to answer knowledge-requiring questions. Yet, most reasoning methods cannot effectively generate queries or incorporate external knowledge during the reasoning process, which may lead to suboptimal results. To bridge this research gap, we present Query and Attention Augmentation, a general approach that augments neural module networks to jointly reason about visual and external knowledge. To take both knowledge sources into account during reasoning, it parses the input question into a functional program with queries augmented through a novel reinforcement learning method, and jointly directs augmented attention to visual and external knowledge based on intermediate reasoning results. With extensive experiments on multiple VQA datasets, our method demonstrates significant performance, explainability, and generalizability over state-of-the-art models in answering questions requiring different extents of knowledge. Our source code is available at https://github.com/SuperJohnZhang/QAA.

----

## [1501] REX: Reasoning-aware and Grounded Explanation

**Authors**: *Shi Chen, Qi Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01514](https://doi.org/10.1109/CVPR52688.2022.01514)

**Abstract**:

Effectiveness and interpretability are two essential properties for trustworthy AI systems. Most recent studies in visual reasoning are dedicated to improving the accuracy of predicted answers, and less attention is paid to explaining the rationales behind the decisions. As a result, they commonly take advantage of spurious biases instead of actually reasoning on the visual-textual data, and have yet developed the capability to explain their decision making by considering key information from both modalities. This paper aims to close the gap from three distinct perspectives: first, we define a new type of multi-modal explanations that explain the decisions by progressively traversing the reasoning process and grounding keywords in the images. We develop a functional program to sequentially ex-ecute different reasoning steps and construct a new dataset with 1,040,830 multi-modal explanations. Second, we iden-tify the critical need to tightly couple important components across the visual and textual modalities for explaining the decisions, and propose a novel explanation generation method that explicitly models the pairwise correspon-dence between words and regions of interest. It improves the visual grounding capability by a considerable margin, resulting in enhanced interpretability and reasoning performance. Finally, with our new data and method, we perform extensive analyses to study the effectiveness of our explanation under different settings, including multi-task learning and transfer learning. Our code and data are available at https://github.com/szzexpoi/rex.

----

## [1502] Not All Relations are Equal: Mining Informative Labels for Scene Graph Generation

**Authors**: *Arushi Goel, Basura Fernando, Frank Keller, Hakan Bilen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01515](https://doi.org/10.1109/CVPR52688.2022.01515)

**Abstract**:

Scene graph generation (SGG) aims to capture a wide variety of interactions between pairs of objects, which is essential for full scene understanding. Existing SGG methods trained on the entire set of relations fail to acquire complex reasoning about visual and textual correlations due to various biases in training data. Learning on trivial relations that indicate generic spatial configuration like ‘on’ instead of informative relations such as ‘parked on’ does not enforce this complex reasoning, harming generalization. To address this problem, we propose a novel framework for SGG training that exploits relation labels based on their informativeness. Our model-agnostic training procedure imputes missing informative relations for less informative samples in the training data and trains a SGG model on the imputed labels along with existing annotations. We show that this approach can successfully be used in conjunction with state-of-the-art SGG methods and improves their performance significantly in multiple metrics on the standard Visual Genome benchmark. Furthermore, we obtain considerable improvements for unseen triplets in a more challenging zero-shot setting.

----

## [1503] Unsupervised Vision-Language Parsing: Seamlessly Bridging Visual Scene Graphs with Language Structures via Dependency Relationships

**Authors**: *Chao Lou, Wenjuan Han, Yuhuan Lin, Zilong Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01516](https://doi.org/10.1109/CVPR52688.2022.01516)

**Abstract**:

Understanding realistic visual scene images together with language descriptions is a fundamental task towards generic visual understanding. Previous works have shown compelling comprehensive results by building hierarchical structures for visual scenes (e.g., scene graphs) and natural languages (e.g., dependency trees), individually. However, how to construct a joint vision-language (VL) structure has barely been investigated. More challenging but worthwhile, we introduce a new task that targets on inducing such a joint VL structure in an unsupervised manner. Our goal is to bridge the visual scene graphs and linguistic dependency trees seamlessly. Due to the lack of VL structural data, we start by building a new dataset VLParse. Rather than using labor-intensive labeling from scratch, we propose an automatic alignment procedure to produce coarse structures followed by human refinement to produce high-quality ones. Moreover, we benchmark our dataset by proposing a contrastive learning (CL)-based framework VLGAE, short for Vision-Language Graph Autoencoder. Our model obtains superior performance on two derived tasks, i.e., language grammar induction and VL phrase grounding. Ablations show the effectiveness of both visual cues and dependency relationships on fine-grained VL structure construction.

----

## [1504] Scene Graph Expansion for Semantics-Guided Image Outpainting

**Authors**: *Chiao-An Yang, Cheng-Yo Tan, Wan-Cyuan Fan, Cheng-Fu Yang, Meng-Lin Wu, Yu-Chiang Frank Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01517](https://doi.org/10.1109/CVPR52688.2022.01517)

**Abstract**:

In this paper, we address the task of semantics-guided image outpainting, which is to complete an image by generating semantically practical content. Different from most existing image outpainting works, we approach the above task by understanding and completing image semantics at the scene graph level. In particular, we propose a novel network of Scene Graph Transformer (SGT), which is designed to take node and edge features as inputs for modeling the associated structural information. To better understand and process graph-based inputs, our SGT uniquely performs feature attention at both node and edge levels. While the former views edges as relationship regularization, the latter observes the co-occurrence of nodes for guiding the attention process. We demonstrate that, given a partial input image with its layout and scene graph, our SGT can be applied for scene graph expansion and its conversion to a complete layout. Following state-of-the-art layout-to-image conversions works, the task of image outpainting can be completed with sufficient and practical semantics introduced. Extensive experiments are conducted on the datasets of MS-COCO and Visual Genome, which quantitatively and qualitatively confirm the effectiveness of our proposed SGT and outpainting frameworks.

----

## [1505] VisualHow: Multimodal Problem Solving

**Authors**: *Jinhui Yang, Xianyu Chen, Ming Jiang, Shi Chen, Louis Wang, Qi Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01518](https://doi.org/10.1109/CVPR52688.2022.01518)

**Abstract**:

Recent progress in the interdisciplinary studies of computer vision (CV) and natural language processing (NLP) has enabled the development of intelligent systems that can describe what they see and answer questions accordingly. However, despite showing usefulness in performing these vision-language tasks, existing methods still struggle in understanding real-life problems (i.e., how to do something) and suggesting step-by-step guidance to solve them. With an overarching goal of developing intelligent systems to assist humans in various daily activities, we propose VisualHow, a free-form and open-ended research that focuses on understanding a real-life problem and deriving its solution by incorporating key components across multiple modalities. We develop a new dataset with 20,028 real-life problems and 102,933 steps that constitute their solutions, where each step consists of both a visual illustration and a textual description that guide the problem solving. To establish better understanding of problems and solutions, we also provide annotations of multimodal attention that localizes important components across modalities and solution graphs that encapsulate different steps in structured representations. These data and annotations enable a family of new vision-language tasks that solve real-life problems. Through extensive experiments with representative models, we demonstrate their effectiveness on training and testing models for the new tasks, and there is significant scope for improvement by learning effective attention mechanisms. Our dataset and models are available at https://github.com/formidify/VisualHow.

----

## [1506] FLAVA: A Foundational Language And Vision Alignment Model

**Authors**: *Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, Douwe Kiela*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01519](https://doi.org/10.1109/CVPR52688.2022.01519)

**Abstract**:

State-of-the-art vision and vision-and-language models rely on large-scale visio-linguistic pretraining for obtaining good performance on a variety of downstream tasks. Generally, such models are often either cross-modal (contrastive) or multi-modal (with earlier fusion) but not both; and they often only target specific modalities or tasks. A promising direction would be to use a single holistic universal model, as a “foundation”, that targets all modalities at once-a true vision and language foundation model should be good at vision tasks, language tasks, and cross- and multi-modal vision and language tasks. We introduce FLAVA as such a model and demonstrate impressive performance on a wide range of 35 tasks spanning these target modalities.

----

## [1507] Multi-modal Alignment using Representation Codebook

**Authors**: *Jiali Duan, Liqun Chen, Son Tran, Jinyu Yang, Yi Xu, Belinda Zeng, Trishul Chilimbi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01520](https://doi.org/10.1109/CVPR52688.2022.01520)

**Abstract**:

Aligning signals from different modalities is an important step in vision-language representation learning as it affects the performance of later stages such as cross-modality fusion. Since image and text typically reside in different regions of the feature space, directly aligning them at instance level is challenging especially when features are still evolving during training. In this paper, we propose to align at a higher and more stable level using cluster representation. Specifically, we treat image and text as two “views” of the same entity, and encode them into a joint vision-language coding space spanned by a dictionary of cluster centers (codebook). We contrast positive and negative samples via their cluster assignments while simultaneously optimizing the cluster centers. To further smooth out the learning process, we adopt a teacher-student distillation paradigm, where the momentum teacher of one view guides the student learning of the other. We evaluated our approach on common vision language benchmarks and obtain new SoTA on zero-shot cross modality retrieval while being competitive on various other transfer tasks.

----

## [1508] Negative-Aware Attention Framework for Image-Text Matching

**Authors**: *Kun Zhang, Zhendong Mao, Quan Wang, Yongdong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01521](https://doi.org/10.1109/CVPR52688.2022.01521)

**Abstract**:

Image-text matching, as a fundamental task, bridges the gap between vision and language. The key of this task is to accurately measure similarity between these two modalities. Prior work measuring this similarity mainly based on matched fragments (i.e., word/region with high relevance), while underestimating or even ignoring the effect of mismatched fragments (i.e., word/region with low relevance), e.g., via a typical LeaklyReLU or ReLU operation that forces negative scores close or exact to zero in attention. This work argues that mismatched textual fragments, which contain rich mismatching clues, are also crucial for image-text matching. We thereby propose a novel Negative-Aware Attention Framework (NAAF), which explicitly exploits both the positive effect of matched fragments and the negative effect of mismatched fragments to jointly infer image-text similarity. NAAF (1) delicately designs an iterative optimization method to maximally mine the mismatched fragments, facilitating more discriminative and robust negative effects, and (2) devises the two-branch matching mechanism to precisely calculate similarity/dissimilarity degrees for matched/mismatched fragments with different masks. Extensive experiments on two benchmark datasets, i.e., Flickr30K and MSCOCO, demonstrate the superior effectiveness of our NAAF, achieving state-of-the-art performance. Code will be released at: https://github.com/CrossmodalGroup/NAAF.

----

## [1509] Vision-Language Pre-Training with Triple Contrastive Learning

**Authors**: *Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, Junzhou Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01522](https://doi.org/10.1109/CVPR52688.2022.01522)

**Abstract**:

Vision-language representation learning largely benefits from image-text alignment through contrastive losses (e.g., InfoNCE loss). The success of this alignment strategy is attributed to its capability in maximizing the mutual information (MI) between an image and its matched text. However, simply performing cross-modal alignment (CMA) ignores data potential within each modality, which may result in degraded representations. For instance, although CMA-based models are able to map image-text pairs close together in the embedding space, they fail to ensure that similar inputs from the same modality stay close by. This problem can get even worse when the pre-training data is noisy. In this paper, we propose triple contrastive learning (TCL) for vision-language pre-training by leveraging both cross-modal and intra-modal self-supervision. Besides CMA, TCL introduces an intra-modal contrastive objective to provide complementary benefits in representation learning. To take advantage of localized and structural information from image and text input, TCL further maximizes the average MI between local regions of image/text and their global summary. To the best of our knowledge, ours is the first work that takes into account local structure information for multi-modality representation learning. Experimental evaluations show that our approach is competitive and achieves the new state of the art on various common downstream vision-language tasks such as image-text retrieval and visual question answering.

----

## [1510] Vision-Language Pre-Training for Boosting Scene Text Detectors

**Authors**: *Sibo Song, Jianqiang Wan, Zhibo Yang, Jun Tang, Wenqing Cheng, Xiang Bai, Cong Yao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01523](https://doi.org/10.1109/CVPR52688.2022.01523)

**Abstract**:

Recently, vision-language joint representation learning has proven to be highly effective in various scenarios. In this paper, we specifically adapt vision-language joint learning for scene text detection, a task that intrinsically involves cross-modal interaction between the two modalities: vision and language, since text is the written form of language. Concretely, we propose to learn contextualized, joint representations through vision-language pretraining, for the sake of enhancing the performance of scene text detectors. Towards this end, we devise a pre-training architecture with an image encoder, a text encoder and a cross-modal encoder, as well as three pretext tasks: image-text contrastive learning (ITC), masked language modeling (MLM) and word-in-image prediction (WIP). The pretrained model is able to produce more informative representations with richer semantics, which could readily benefit existing scene text detectors (such as EAST and PSENet) in the down-stream text detection task. Extensive experiments on standard benchmarks demonstrate that the proposed paradigm can significantly improve the performance of various representative text detectors, outperforming previous pre-training approaches. The code and pre-trained models will be publicly released.

----

## [1511] COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval

**Authors**: *Haoyu Lu, Nanyi Fei, Yuqi Huo, Yizhao Gao, Zhiwu Lu, Ji-Rong Wen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01524](https://doi.org/10.1109/CVPR52688.2022.01524)

**Abstract**:

Large-scale single-stream pre-training has shown dramatic performance in image-text retrieval. Regrettably, it faces low inference efficiency due to heavy attention layers. Recently, two-stream methods like CLIP and ALIGN with high inference efficiency have also shown promising performance, however, they only consider instance-level alignment between the two streams (thus there is still room for improvement). To overcome these limitations, we propose a novel COllaborative Two-Stream vision-language pretraining model termed COTS for image-text retrieval by enhancing cross-modal interaction. In addition to instance-level alignment via momentum contrastive learning, we leverage two extra levels of cross-modal interactions in our COTS: (1) Token-level interaction - a masked vision-language modeling (MVLM) learning objective is devised without using a cross-stream network module, where variational autoencoder is imposed on the visual encoder to generate visual tokens for each image. (2) Task-level interaction - a KL-alignment learning objective is devised between text-to-image and image-to-text retrieval tasks, where the probability distribution per task is computed with the negative queues in momentum contrastive learning. Under a fair comparison setting, our COTS achieves the highest performance among all two-stream methods and comparable performance (but with 10,800× faster in inference) w.r.t. the latest single-stream methods. Importantly, our COTS is also applicable to text-to-video retrieval, yielding new state-of-the-art on the widely-used MSR-VTT dataset.

----

## [1512] NeurMiPs: Neural Mixture of Planar Experts for View Synthesis

**Authors**: *Zhi-Hao Lin, Wei-Chiu Ma, Hao-Yu Hsu, Yu-Chiang Frank Wang, Shenlong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01525](https://doi.org/10.1109/CVPR52688.2022.01525)

**Abstract**:

We present Neural Mixtures of Planar Experts (Neur-MiPs), a novel planar-based scene representation for modeling geometry and appearance. NeurMiPs leverages a collection of local planar experts in 3D space as the scene representation. Each planar expert consists of the parameters of the local rectangular shape representing geometry and a neural radiance field modeling the color and opacity. We render novel views by calculating ray-plane intersections and composite output colors and densities at intersected points to the image. NeurMiPs blends the efficiency of explicit mesh rendering and flexibility of the neural radiance field. Experiments demonstrate superior performance and speed of our proposed method, compared to other 3D representations in novel view synthesis.

----

## [1513] FWD: Real-time Novel View Synthesis with Forward Warping and Depth

**Authors**: *Ang Cao, Chris Rockwell, Justin Johnson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01526](https://doi.org/10.1109/CVPR52688.2022.01526)

**Abstract**:

Novel view synthesis (NVS) is a challenging task requiring systems to generate photorealistic images of scenes from new viewpoints, where both quality and speed are important for applications. Previous image-based rendering (IBR) methods are fast, but have poor quality when input views are sparse. Recent Neural Radiance Fields (NeRF) and generalizable variants give impressive results but are not real-time. In our paper, we propose a generalizable NVS method with sparse inputs, called FWD, which gives high-quality synthesis in real-time. With explicit depth and differentiable rendering, it achieves competitive results to the SOTA methods with 130-1000× speedup and better perceptual quality. If available, we can seamlessly integrate sensor depth during either training or inference to improve image quality while retaining real-time speed. With the growing prevalence of depths sensors, we hope that methods making use of depth will become increasingly useful.

----

## [1514] SOMSI: Spherical Novel View Synthesis with Soft Occlusion Multi-Sphere Images

**Authors**: *Tewodros Habtegebrial, Christiano Couto Gava, Marcel Rogge, Didier Stricker, Varun Jampani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01527](https://doi.org/10.1109/CVPR52688.2022.01527)

**Abstract**:

Spherical novel view synthesis (SNVS) is the task of estimating 360
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">○</sup>
 views at dynamic novel views given a set of 360
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">○</sup>
 input views. Prior arts learn multi-sphere image (MSI) representations that enable fast rendering times but are only limited to modelling low-dimensional color values. Modelling high-dimensional appearance features in MSI can result in better view synthesis, but it is not feasible to represent high-dimensional features in a large number (> 64) of MSI spheres. We propose a novel MSI representation called Soft Occlusion MSI (SOMSI) that enables modelling high-dimensional appearance features in MSI while retaining the fast rendering times of a standard MSI. Our key insight is to model appearance features in a smaller set (e.g. 3) of occlusion levels instead of larger number of MSI levels. Experiments on both synthetic and real-world scenes demonstrate that using SOMSI can provide a good balance between accuracy and run-time. SOMSI can produce considerably better results compared to MSI based MODS [1], while having similar fast rendering time. SOMSI view synthesis quality is on-par with state-of-the-art NeRF [24] like model while being 2 orders of magnitude faster. For code, additional results and data, please visit https://tedyhabtegebrial.github.io/somsi.

----

## [1515] Fast, Accurate and Memory-Efficient Partial Permutation Synchronization

**Authors**: *Shaohan Li, Yunpeng Shi, Gilad Lerman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01528](https://doi.org/10.1109/CVPR52688.2022.01528)

**Abstract**:

Previous partial permutation synchronization (PPS) algorithms, which are commonly used for multi-object matching, often involve computation-intensive and memory-demanding matrix operations. These operations become intractable for large scale structure-from-motion datasets. For pure permutation synchronization, the recent Cycle-Edge Message Passing (CEMP) framework suggests a memory-efficient and fast solution. Here we overcome the restriction of CEMP to compact groups and propose an improved algorithm, CEMP-Partial, for estimating the corruption levels of the observed partial permutations. It allows us to subsequently implement a nonconvex weighted projected power method without the need of spectral initialization. The resulting new PPS algorithm, MatchFAME (Fast, Accurate and Memory-Efficient Matching), only involves sparse matrix operations, and thus enjoys lower time and space complexities in comparison to previous PPS algorithms. We prove that under adversarial corruption, though without additive noise and with certain assumptions, CEMP-Partial is able to exactly classify corrupted and clean partial permutations. We demonstrate the state-of-the-art accuracy, speed and memory efficiency of our method on both synthetic and real datasets.

----

## [1516] Learning to Find Good Models in RANSAC

**Authors**: *Daniel Barath, Luca Cavalli, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01529](https://doi.org/10.1109/CVPR52688.2022.01529)

**Abstract**:

We propose the Model Quality Network, MQ-Net in short, for predicting the quality, e.g. the pose error of essential matrices, of models generated inside RANSAC. It replaces the traditionally used scoring techniques, e.g., inlier counting of RANSAC, truncated loss of MSAC, and the marginalization-based loss of MAGSAC++. Moreover, Minimal samples Filtering Network (MF-Net) is proposed for the early rejection of minimal samples that likely lead to degenerate models or to ones that are inconsistent with the scene geometry, e.g., due to the chirality constraint. We show on 54450 image pairs from public real-world datasets that the proposed MQ-Net leads to results superior to the state-of-the-art in terms of accuracy by a large margin. The proposed MF-Net accelerates the fundamental matrix estimation by five times and significantly reduces the essential matrix estimation time while slightly improving accuracy as well. Also, we show experimentally that consensus maximization, i.e. inlier counting, is not an inherently good measure of the model quality for relative pose estimation. The code is at https://github.com/danini/learning-goad-models-in-ransac.

----

## [1517] Optimizing Elimination Templates by Greedy Parameter Search

**Authors**: *Evgeniy Martyushev, Jana Vráblíková, Tomás Pajdla*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01530](https://doi.org/10.1109/CVPR52688.2022.01530)

**Abstract**:

We propose a new method for constructing elimination templates for efficient polynomial system solving of minimal problems in structure from motion, image matching, and camera tracking. We first construct a particular affine parameterization of the elimination templates for systems with a finite number of distinct solutions. Then, we use a heuristic greedy optimization strategy over the space of parameters to get a template with a small size. We test our method on 34 minimal problems in computer vision. For all of them, we found the templates either of the same or smaller size compared to the state-of-the-art. For some difficult examples, our templates are, e.g., 2.1, 2.5, 3.8, 6.6 times smaller. For the problem of refractive absolute pose estimation with unknown focal length, we have found a template that is 20 times smaller. Our experiments on synthetic data also show that the new solvers are fast and numerically accurate. We also present a fast and numerically accurate solver for the problem of relative pose estimation with unknown common focal length and radial distortion.

----

## [1518] GPU-Based Homotopy Continuation for Minimal Problems in Computer Vision

**Authors**: *Chiang-Heng Chien, Hongyi Fan, Ahmad Abdelfattah, Elias P. Tsigaridas, Stanimire Tomov, Benjamin B. Kimia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01531](https://doi.org/10.1109/CVPR52688.2022.01531)

**Abstract**:

Systems of polynomial equations arise frequently in computer vision, especially in multiview geometry problems. Traditional methods for solving these systems typically aim to eliminate variables to reach a univariate polynomial, e.g., a tenth-order polynomial for 5-point pose estimation, using clever manipulations, or more generally using Grobner basis, resultants, and elimination templates, leading to successful algorithms for multiview geometry and other problems. However, these methods do not work when the problem is complex and when they do, they face efficiency and stability issues. Homotopy Continuation (HC) can solve more complex problems without the stability issues, and with guarantees of a global solution, but they are known to be slow. In this paper we show that HC can be parallelized on a GPU, showing significant speedups up to 56 times on polynomial benchmarks. We also show that GPU-HC can be generically applied to a range of computer vision problems, including 4-view triangulation and trifocal pose estimation with unknown focal length, which cannot be solved with elimination template but they can be efficiently solved with HC. GPU-HC opens the door to easy formulation and solution of a range of computer vision problems.

----

## [1519] HARA: A Hierarchical Approach for Robust Rotation Averaging

**Authors**: *Seong Hun Lee, Javier Civera*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01532](https://doi.org/10.1109/CVPR52688.2022.01532)

**Abstract**:

We propose a novel hierarchical approach for multiple rotation averaging, dubbed HARA. Our method incrementally initializes the rotation graph based on a hierarchy of triplet support. The key idea is to build a spanning tree by prioritizing the edges with many strong triplet supports and gradually adding those with weaker and fewer supports. This reduces the risk of adding outliers in the spanning tree. As a result, we obtain a robust initial solution that enables us to filter outliers prior to nonlinear optimization. With minimal modification, our approach can also integrate the knowledge of the number of valid 2D-2D correspondences. We perform extensive evaluations on both synthetic and real datasets, demonstrating state-of-the-art results.

----

## [1520] RAGO: Recurrent Graph Optimizer For Multiple Rotation Averaging

**Authors**: *Heng Li, Zhaopeng Cui, Shuaicheng Liu, Ping Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01533](https://doi.org/10.1109/CVPR52688.2022.01533)

**Abstract**:

This paper proposes a deep recurrent Rotation Averaging Graph Optimizer (RAGO) for Multiple Rotation Averaging (MRA). Conventional optimization-based methods usually fail to produce accurate results due to corrupted and noisy relative measurements. Recent learning-based approaches regard MRA as a regression problem, while these methods are sensitive to initialization due to the gauge freedom problem. To handle these problems, we propose a learnable iterative graph optimizer minimizing a gauge- invariant cost function with an edge rectification strategy to mitigate the effect of inaccurate measurements. Our graph optimizer iteratively refines the global camera rotations by minimizing each node's single rotation objective function. Besides, our approach iteratively rectifies relative rotations to make them more consistent with the current camera orientations and observed relative rotations. Furthermore, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$we$</tex>
 employ a gated recurrent unit to improve the result by tracing the temporal information of the cost graph. Our framework is a real-time learning-to-optimize rotation averaging graph optimizer with a tiny size deployed for real-world applications. RAGO outperforms previous traditional and deep methods on real-world and synthetic datasets. The code is available at github.com/sfu-gruvi-3dv/RAGO.

----

## [1521] A Unified Model for Line Projections in Catadioptric Cameras with Rotationally Symmetric Mirrors

**Authors**: *Pedro Miraldo, José Pedro Iglesias*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01534](https://doi.org/10.1109/CVPR52688.2022.01534)

**Abstract**:

Lines are among the most used computer vision features, in applications such as camera calibration to object detection. Catadioptric cameras with rotationally symmetric mirrors are omnidirectional imaging devices, capturing up to a 360 degrees field of view. These are used in many applications ranging from robotics to panoramic vision. Although known for some specific configurations, the modeling of line projection was never fully solved for general central and non-central catadioptric cameras. We start by taking some general point reflection assumptions and derive a line reflection constraint. This constraint is then used to define a line projection into the image. Next, we compare our model with previous methods, showing that our general approach outputs the same polynomial degrees as previous configuration-specific systems. We run several experiments using synthetic and real-world data, validating our line projection model. Lastly, we show an application of our methods to an absolute camera pose problem.

----

## [1522] ELSR: Efficient Line Segment Reconstruction with Planes and Points Guidance

**Authors**: *Dong Wei, Yi Wan, Yongjun Zhang, Xinyi Liu, Bin Zhang, Xiqi Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01535](https://doi.org/10.1109/CVPR52688.2022.01535)

**Abstract**:

Three-dimensional (3D) line segments are helpful for scene reconstruction. Most of the existing 3D-line-segment reconstruction algorithms deal with two views or dozens of small-size images; while in practice there are usually hundreds or thousands of large-size images. In this paper, we propose an efficient line segment reconstruction method called ELSR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Available at https://skyearth.org/publication/project/ELSR. ELSR exploits scene planes that are commonly seen in city scenes and sparse 3D points that can be acquired easily from the structure-from-motion (SfM) approach. For two views, ELSR efficiently finds the local scene plane to guide the line matching and exploits sparse 3D points to accelerate and constrain the matching. To reconstruct a 3D line segment with multiple views, ELSR utilizes an efficient abstraction approach that selects representative 3D lines based on their spatial consistence. Our experiments demonstrated that ELSR had a higher accuracy and efficiency than the existing methods. Moreover, our results showed that ELSR could reconstruct 3D lines efficiently for large and complex scenes that contain thousands of large-size images.

----

## [1523] Self-supervised Neural Articulated Shape and Appearance Models

**Authors**: *Fangyin Wei, Rohan Chabra, Lingni Ma, Christoph Lassner, Michael Zollhöfer, Szymon Rusinkiewicz, Chris Sweeney, Richard A. Newcombe, Mira Slavcheva*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01536](https://doi.org/10.1109/CVPR52688.2022.01536)

**Abstract**:

Learning geometry, motion, and appearance priors of object classes is important for the solution of a large variety of computer vision problems. While the majority of approaches has focused on static objects, dynamic objects, especially with controllable articulation, are less explored. We propose a novel approach for learning a representation of the geometry, appearance, and motion of a class of articulated objects given only a set of color images as input. In a self-supervised manner, our novel representation learns shape, appearance, and articulation codes that enable independent control of these semantic dimensions. Our model is trained end-to-end without requiring any articulation annotations. Experiments show that our approach performs well for different joint types, such as revolute and prismatic joints, as well as different combinations of these joints. Compared to state of the art that uses direct 3D supervision and does not output appearance, we recover more faithful geometry and appearance from 2D observations only. In addition, our representation enables a large variety of applications, such as few-shot reconstruction, the generation of novel articulations, and novel view-synthesis. Project page: https://weify627.github.io/nasam/.

----

## [1524] Virtual Elastic Objects

**Authors**: *Hsiao-Yu Chen, Edith Tretschk, Tuur Stuyck, Petr Kadlecek, Ladislav Kavan, Etienne Vouga, Christoph Lassner*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01537](https://doi.org/10.1109/CVPR52688.2022.01537)

**Abstract**:

We present Virtual Elastic Objects (VEOs): virtual objects that not only look like their real-world counterparts but also behave like them, even when subject to novel interactions. Achieving this presents multiple challenges: not only do objects have to be captured including the physical forces acting on them, then faithfully reconstructed and rendered, but also plausible material parameters found and simulated. To create VEOs, we built a multi-view capture system that captures objects under the influence of a compressed air stream. Building on recent advances in model-free, dynamic Neural Radiance Fields, we reconstruct the objects and corresponding deformation fields. We propose to use a differentiable, particle-based simulator to use these deformation fields to find representative material parameters, which enable us to run new simulations. To render simulated objects, we devise a method for integrating the simulation results with Neural Radiance Fields. The resulting method is applicable to a wide range of scenarios: it can handle objects composed of inhomogeneous material, with very different shapes, and it can simulate interactions with other virtual objects. We present our results using a newly collected dataset of 12 objects under a variety of force fields, which will be made available upon publication.

----

## [1525] Decoupling Makes Weakly Supervised Local Feature Better

**Authors**: *Kunhong Li, Longguang Wang, Li Liu, Qing Ran, Kai Xu, Yulan Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01538](https://doi.org/10.1109/CVPR52688.2022.01538)

**Abstract**:

Weakly supervised learning can help local feature methods to overcome the obstacle of acquiring a large-scale dataset with densely labeled correspondences. However, since weak supervision cannot distinguish the losses caused by the detection and description steps, directly conducting weakly supervised learning within a joint training describe-then-detect pipeline suffers limited performance. In this paper, we propose a decoupled training describe-then-detect pipeline tailored for weakly supervised local feature learning. Within our pipeline, the detection step is decoupled from the description step and postponed until discriminative and robust descriptors are learned. In addition, we introduce a line-to-window search strategy to explicitly use the camera pose information for better descriptor learning. Extensive experiments show that our method, namely PoSFeat (Camera Pose Supervised Feature), outperforms previous fully and weakly supervised methods and achieves state-of-the-art performance on a wide range of downstream task.

----

## [1526] JoinABLe: Learning Bottom-up Assembly of Parametric CAD Joints

**Authors**: *Karl D. D. Willis, Pradeep Kumar Jayaraman, Hang Chu, Yunsheng Tian, Yifei Li, Daniele Grandi, Aditya Sanghi, Linh Tran, Joseph G. Lambourne, Armando Solar-Lezama, Wojciech Matusik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01539](https://doi.org/10.1109/CVPR52688.2022.01539)

**Abstract**:

Physical products are often complex assemblies combining a multitude of 3D parts modeled in computer-aided design (CAD) software. CAD designers build up these assemblies by aligning individual parts to one another using constraints called joints. In this paper we introduce JoinABLe, a learning-based method that assembles parts together to form joints. JoinABLe uses the weak supervision available in standard parametric CAD files without the help of object class labels or human guidance. Our results show that by making network predictions over a graph representation of solid models we can outperform multiple baseline methods with an accuracy (79.53%) that approaches human performance (80%). Finally, to support future research we release the Fusion 360 Gallery assembly dataset, containing assemblies with rich information on joints, contact surfaces, holes, and the underlying assembly graph structure.

----

## [1527] ImplicitAtlas: Learning Deformable Shape Templates in Medical Imaging

**Authors**: *Jiancheng Yang, Udaranga Wickramasinghe, Bingbing Ni, Pascal Fua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01540](https://doi.org/10.1109/CVPR52688.2022.01540)

**Abstract**:

Deep implicit shape models have become popular in the computer vision community at large but less so for biomed-ical applications. This is in part because large training databases do not exist and in part because biomedical an-notations are often noisy. In this paper, we show that by introducing templates within the deep learning pipeline we can overcome these problems. The proposed framework, named ImplicitAtlas, represents a shape as a deformation field from a learned template field, where multiple templates could be integrated to improve the shape representation ca-pacity at negligible computational cost. Extensive experi-ments on three medical shape datasets prove the superiority over current implicit representation methods.

----

## [1528] DoubleField: Bridging the Neural Surface and Radiance Fields for High-fidelity Human Reconstruction and Rendering

**Authors**: *Ruizhi Shao, Hongwen Zhang, He Zhang, Mingjia Chen, Yanpei Cao, Tao Yu, Yebin Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01541](https://doi.org/10.1109/CVPR52688.2022.01541)

**Abstract**:

We introduce DoubleField, a novel framework combining the merits of both surface field and radiance field for high-fidelity human reconstruction and rendering. Within DoubleField, the surface field and radiance field are associated together by a shared feature embedding and a surface-guided sampling strategy. Moreover, a view-to-view transformer is introduced to fuse multi-view features and learn view-dependent features directly from high-resolution inputs. With the modeling power of DoubleField and the view-to-view transformer, our method significantly improves the reconstruction quality of both geometry and appearance, while supporting direct inference, scene-specific high-resolution finetuning, and fast rendering. The efficacy of DoubleField is validated by the quantitative evaluations on several datasets and the qualitative results in a real-world sparse multi-view system, showing its superior capability for high-quality human model reconstruction and photo-realistic free-viewpoint human rendering. Data and source code will be made public for the research purpose.

----

## [1529] Surface-Aligned Neural Radiance Fields for Controllable 3D Human Synthesis

**Authors**: *Tianhan Xu, Yasuhiro Fujita, Eiichi Matsumoto*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01542](https://doi.org/10.1109/CVPR52688.2022.01542)

**Abstract**:

We propose a new method for reconstructing control-lable implicit 3D human models from sparse multi-view RGB videos. Our method defines the neural scene repre-sentation on the mesh surface points and signed distances from the surface of a human body mesh. We identify an indistinguishability issue that arises when a point in 3D space is mapped to its nearest surface point on a mesh for learning surface-aligned neural scene representation. To address this issue, we propose projecting a point onto a mesh surface using a barycentric interpolation with modi-fied vertex normals. Experiments with the ZJU-MoCap and Human3.6M datasets show that our approach achieves a higher quality in a novel-view and novel-pose synthesis than existing methods. We also demonstrate that our method eas-ily supports the control of body shape and clothes. Project page: https://pfnet-research.github.io/surface-aligned-nerf/

----

## [1530] Structured Local Radiance Fields for Human Avatar Modeling

**Authors**: *Zerong Zheng, Han Huang, Tao Yu, Hongwen Zhang, Yandong Guo, Yebin Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01543](https://doi.org/10.1109/CVPR52688.2022.01543)

**Abstract**:

It is extremely challenging to create an animatable clothed human avatar from RGB videos, especially for loose clothes due to the difficulties in motion modeling. To address this problem, we introduce a novel representation on the basis of recent neural scene rendering techniques. The core of our representation is a set of structured local radiance fields, which are anchored to the pre-defined nodes sampled on a statistical human body template. These local radiance fields not only leverage the flexibility of implicit representation in shape and appearance modeling, but also factorize cloth deformations into skeleton motions, node residual translations and the dynamic detail variations inside each individual radiance field. To learn our representation from RGB data and facilitate pose generalization, we propose to learn the node translations and the detail variations in a conditional generative latent space. Overall, our method enables automatic construction of animatable human avatars for various types of clothes without the need for scanning subject-specific templates, and can generate realistic images with dynamic details for novel poses. Experiment show that our method outperforms state-of-the-art methods both qualitatively and quantitatively.

----

## [1531] High-Fidelity Human Avatars from a Single RGB Camera

**Authors**: *Hao Zhao, Jinsong Zhang, Yu-Kun Lai, Zerong Zheng, Yingdi Xie, Yebin Liu, Kun Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01544](https://doi.org/10.1109/CVPR52688.2022.01544)

**Abstract**:

In this paper, we propose a coarse-to-fine framework to reconstruct a personalized high-fidelity human avatar from a monocular video. To deal with the misalignment problem caused by the changed poses and shapes in different frames, we design a dynamic surface network to recover pose-dependent surface deformations, which help to decouple the shape and texture of the person. To cope with the complexity of textures and generate photo-realistic results, we propose a reference-based neural rendering network and exploit a bottom-up sharpening-guided fine-tuning strategy to obtain detailed textures. Our frame-work also enables photo-realistic novel view/pose syn-thesis and shape editing applications. Experimental re-sults on both the public dataset and our collected dataset demonstrate that our method outperforms the state-of-the-art methods. The code and dataset will be available at http://cic.tju.edu.cn/faculty/likun/projects/HF-Avatar.

----

## [1532] Forecasting Characteristic 3D Poses of Human Actions

**Authors**: *Christian Diller, Thomas A. Funkhouser, Angela Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01545](https://doi.org/10.1109/CVPR52688.2022.01545)

**Abstract**:

We propose the task of forecasting characteristic 3d poses: from a short sequence observation of a person, predict a future 3d pose of that person in a likely action-defining, characteristic pose - for instance, from observing a person picking up an apple, predict the pose of the person eating the apple. Prior work on human motion prediction estimates future poses at fixed time intervals. Although easy to define, this frame-by-frame formulation confounds temporal and intentional aspects of human action. Instead, we define a semantically meaningful pose prediction task that decouples the predicted pose from time, taking inspiration from goal-directed behavior. To predict characteristic poses, we propose a probabilistic approach that models the possible multimodality in the distribution of likely characteristic poses. We then sample future pose hypotheses from the predicted distribution in an autoregressive fashion to model dependencies between joints. To evaluate our method, we construct a dataset of manually annotated characteristic 3d poses. Our experiments with this dataset suggest that our proposed probabilistic approach outperforms state-of-the-art methods by 26% on average.

----

## [1533] Virtual Correspondence: Humans as a Cue for Extreme-View Geometry

**Authors**: *Wei-Chiu Ma, Anqi Joyce Yang, Shenlong Wang, Raquel Urtasun, Antonio Torralba*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01546](https://doi.org/10.1109/CVPR52688.2022.01546)

**Abstract**:

Recovering the spatial layout of the cameras and the geometry of the scene from extreme-view images is a longstanding challenge in computer vision. Prevailing 3D reconstruction algorithms often adopt the image matching paradigm and presume that a portion of the scene is covisible across images, yielding poor performance when there is little overlap among inputs. In contrast, humans can associate visible parts in one image to the corresponding invisible components in another image via prior knowledge of the shapes. Inspired by this fact, we present a novel concept called virtual correspondences (VCs). VCs are a pair of pixels from two images whose camera rays intersect in 3D. Similar to classic correspondences, VCs conform with epipolar geometry; unlike classic correspondences, VCs do not need to be co-visible across views. Therefore VCs can be established and exploited even if images do not overlap. We introduce a method to find virtual correspondences based on humans in the scene. We showcase how VCs can be seamlessly integrated with classic bundle adjustment to recover camera poses across extreme views. Experiments show that our method significantly outperforms state-of-the-art camera pose estimation methods in challenging scenarios and is comparable in the traditional densely captured setup. Our approach also unleashes the potential of multiple down-stream tasks such as scene reconstruction from multi-view stereo and novel view synthesis in extreme-view scenarios
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://people.csail.mit.edu/weichium/virtual-correspondence/.

----

## [1534] BEHAVE: Dataset and Method for Tracking Human Object Interactions

**Authors**: *Bharat Lal Bhatnagar, Xianghui Xie, Ilya A. Petrov, Cristian Sminchisescu, Christian Theobalt, Gerard Pons-Moll*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01547](https://doi.org/10.1109/CVPR52688.2022.01547)

**Abstract**:

Modelling interactions between humans and objects in natural environments is central to many applications including gaming, virtual and mixed reality, as well as human behavior analysis and human-robot collaboration. This challenging operation scenario requires generalization to vast number of objects, scenes, and human actions. Unfortunately, there exist no such dataset. Moreover, this data needs to be acquired in diverse natural environments, which rules out 4D scanners and marker based capture systems. We present BEHAVE dataset, the first full body human-object interaction dataset with multi-view RGBD frames and corresponding 3D SMPL and object fits along with the annotated contacts between them. We record ~15k frames at 5 locations with 8 subjects performing a wide range of interactions with 20 common objects. We use this data to learn a model that can jointly track humans and objects in natural environments with an easy-to-use portable multi-camera setup. Our key insight is to predict correspondences from the human and the object to a statistical body model to obtain human-object contacts during interactions. Our approach can record and track not just the humans and objects but also their interactions, modeled as surface contacts, in 3D. Our code and data can be found at: http://virtualhumans.mpi-inf.mpg.de/behave.

----

## [1535] Primitive3D: 3D Object Dataset Synthesis from Randomly Assembled Primitives

**Authors**: *Xinke Li, Henghui Ding, Zekun Tong, Yuwei Wu, Yeow Meng Chee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01548](https://doi.org/10.1109/CVPR52688.2022.01548)

**Abstract**:

Numerous advancements in deep learning can be attributed to the access to large-scale and well-annotated datasets. However, such a dataset is prohibitively expensive in 3D computer vision due to the substantial collection cost. To alleviate this issue, we propose a cost-effective method for automatically generating a large amount of 3D objects with annotations. In particular, we synthesize objects simply by assembling multiple random primitives. These objects are thus auto-annotated with part labels originating from primitives. This allows us to perform multi-task learning by combining the supervised segmentation with unsupervised reconstruction. Considering the large overhead of learning on the generated dataset, we further propose a dataset distillation strategy to remove redundant samples regarding a target dataset. We conduct extensive experiments for the downstream tasks of 3D object classification. The results indicate that our dataset, together with multitask pretraining on its annotations, achieves the best performance compared to other commonly used datasets. Further study suggests that our strategy can improve the model performance by pretraining and fine-tuning scheme, especially for the dataset with a small scale. In addition, pretraining with the proposed dataset distillation method can save 86% of the pretraining time with negligible performance degradation. We expect that our attempt provides a new data-centric perspective for training 3D deep models.

----

## [1536] RGB-Multispectral Matching: Dataset, Learning Methodology, Evaluation

**Authors**: *Fabio Tosi, Pierluigi Zama Ramirez, Matteo Poggi, Samuele Salti, Stefano Mattoccia, Luigi Di Stefano*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01549](https://doi.org/10.1109/CVPR52688.2022.01549)

**Abstract**:

We address the problem of registering synchronized color (RGB) and multi-spectral (MS) images featuring very different resolution by solving stereo matching correspondences. Purposely, we introduce a novel RGB-MS dataset framing 13 different scenes in indoor environments and providing a total of 34 image pairs annotated with semi-dense, high-resolution ground-truth labels in the form of disparity maps. To tackle the task, we propose a deep learning architecture trained in a self-supervised manner by exploiting a further RGB camera, required only during training data acquisition. In this setup, we can conveniently learn cross-modal matching in the absence of ground-truth labels by distilling knowledge from an easier RGB-RGB matching task based on a collection of about 11K unlabeled image triplets. Experiments show that the proposed pipeline sets a good performance bar (1.16 pixels average registration error) for future research on this novel, challenging task.

----

## [1537] NPBG++: Accelerating Neural Point-Based Graphics

**Authors**: *Ruslan Rakhimov, Andrei-Timotei Ardelean, Victor Lempitsky, Evgeny Burnaev*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01550](https://doi.org/10.1109/CVPR52688.2022.01550)

**Abstract**:

We present a new system 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(NPBG++)$</tex>
 for the novel view synthesis (NVS) task that achieves high rendering realism with low scene fitting time. Our method efficiently lever-ages the multiview observations and the point cloud of a static scene to predict a neural descriptor for each point, improving upon the pipeline of Neural Point-Based Graph-ics [1] in several important ways. By predicting the descrip-tors with a single pass through the source images, we lift the requirement of per-scene optimization while also making the neural descriptors view-dependent and more suit-able for scenes with strong non-Lambertian effects. In our comparisons, the proposed system outperforms previous NVS approaches in terms of fitting and rendering runtimes while producing images of similar quality. Project page: https://rakhimovv.github.io/npbgpp/.

----

## [1538] Depth-Guided Sparse Structure-from-Motion for Movies and TV Shows

**Authors**: *Sheng Liu, Xiaohan Nie, Raffay Hamid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01551](https://doi.org/10.1109/CVPR52688.2022.01551)

**Abstract**:

Existing approaches for Structure from Motion (SfM) produce impressive 3-D reconstruction results especially when using imagery captured with large parallax. However, to create engaging video-content in movies and TV shows, the amount by which a camera can be moved while filming a particular shot is often limited. The resulting small-motion parallax between video frames makes standard geometry-based SfM approaches not as effective for movies and TV shows. To address this challenge, we propose a simple yet effective approach that uses single-frame depth-prior obtained from a pretrained network to significantly improve geometry-based SfM for our small-parallax setting. To this end, we first use the depth-estimates of the detected keypoints to reconstruct the point cloud and camera-pose for initial two-view reconstruction. We then perform depth-regularized optimization to register new images and triangulate the new points during incremental reconstruction. To comprehensively evaluate our approach, we introduce a new dataset (StudioSfM) consisting of 130 shots with 21K frames from 15 studio-produced videos that are manually annotated by a professional CG studio. We demonstrate that our approach: (a) significantly improves the quality of 3-D reconstruction for our small-parallax setting, (b) does not cause any degradation for data with large-parallax, and (c) maintains the generalizability and scalability of geometry-based sparse SfM. Our dataset can be obtained at https://github.com/amazon-researchlsmall-baseline-camera-tracking.

----

## [1539] Motion-from-Blur: 3D Shape and Motion Estimation of Motion-blurred Objects in Videos

**Authors**: *Denys Rozumnyi, Martin R. Oswald, Vittorio Ferrari, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01552](https://doi.org/10.1109/CVPR52688.2022.01552)

**Abstract**:

We propose a method for jointly estimating the 3D motion, 3D shape, and appearance of highly motion-blurred objects from a video. To this end, we model the blurred appearance of a fast moving object in a generative fashion by parametrizing its 3D position, rotation, velocity, acceleration, bounces, shape, and texture over the duration of a predefined time window spanning multiple frames. Using differentiable rendering, we are able to estimate all parameters by minimizing the pixel-wise reprojection error to the input video via backpropagating through a rendering pipeline that accounts for motion blur by averaging the graphics output over short time intervals. For that purpose, we also estimate the camera exposure gap time within the same optimization. To account for abrupt motion changes like bounces, we model the motion trajectory as a piece-wise polynomial, and we are able to estimate the specific time of the bounce at sub-frame accuracy. Experiments on established benchmark datasets demonstrate that our method outperforms previous methods for fast moving object deblurring and 3D reconstruction.

----

## [1540] Masked Autoencoders Are Scalable Vision Learners

**Authors**: *Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross B. Girshick*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01553](https://doi.org/10.1109/CVPR52688.2022.01553)

**Abstract**:

This paper shows that masked autoencoders (MAE) are scalable self-supervised learners for computer vision. Our MAE approach is simple: we mask random patches of the input image and reconstruct the missing pixels. It is based on two core designs. First, we develop an asymmetric encoder-decoder architecture, with an encoder that operates only on the visible subset of patches (without mask tokens), along with a lightweight decoder that reconstructs the original image from the latent representation and mask tokens. Second, we find that masking a high proportion of the input image, e.g., 75%, yields a nontrivial and meaningful self-supervisory task. Coupling these two designs enables us to train large models efficiently and effectively: we accelerate training (by 3× or more) and improve accuracy. Our scalable approach allows for learning high-capacity models that generalize well: e.g., a vanilla ViT-Huge model achieves the best accuracy (87.8%) among methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

----

## [1541] Learning ABCs: Approximate Bijective Correspondence for isolating factors of variation with weak supervision

**Authors**: *Kieran A. Murphy, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01554](https://doi.org/10.1109/CVPR52688.2022.01554)

**Abstract**:

Representational learning forms the backbone of most deep learning applications, and the value of a learned representation is intimately tied to its information content regarding different factors of variation. Finding good representations depends on the nature of supervision and the learning algorithm. We propose a novel algorithm that utilizes a weak form of supervision where the data is partitioned into sets according to certain inactive (common) factors of variation which are invariant across elements of each set. Our key insight is that by seeking correspondence between elements of different sets, we learn strong representations that exclude the inactive factors of variation and isolate the active factors that vary within all sets. As a consequence of focusing on the active factors, our method can leverage a mix of setsupervised and wholly unsupervised data, which can even belong to a different domain. We tackle the challenging problem of synthetic-to-real object pose transfer, without pose annotations on anything, by isolating pose information which generalizes to the category level and across the synthetic/real domain gap. The method can also boost performance in supervised settings, by strengthening intermediate representations, as well as operate in practically attainable scenarios with set-supervised natural images, where quantity is limited and nuisance factors of variation are more plentiful. Accompanying code may be found on github.

----

## [1542] Bayesian Invariant Risk Minimization

**Authors**: *Yong Lin, Hanze Dong, Hao Wang, Tong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01555](https://doi.org/10.1109/CVPR52688.2022.01555)

**Abstract**:

Generalization under distributional shift is an open challenge for machine learning. Invariant Risk Minimization (IRM) is a promising framework to tackle this issue by extracting invariant features. However, despite the potential and popularity of IRM, recent works have reported negative results of it on deep models. We argue that the failure can be primarily attributed to deep models' tendency to overfit the data. Specifically, our theoretical analysis shows that IRM degenerates to empirical risk minimization (ERM) when overfitting occurs. Our empirical evidence also provides supports: IRM methods that work well in typical settings significantly deteriorate even if we slightly enlarge the model size or lessen the training data. To alleviate this issue, we propose Bayesian Invariant Risk Min-imization (BIRM) by introducing Bayesian inference into the IRM. The key motivation is to estimate the penalty of IRM based on the posterior distribution of classifiers (as opposed to a single classifier), which is much less prone to overfitting. Extensive experimental results on four datasets demonstrate that BIRM consistently outperforms the existing IRM baselines significantly.

----

## [1543] Crafting Better Contrastive Views for Siamese Representation Learning

**Authors**: *Xiangyu Peng, Kai Wang, Zheng Zhu, Mang Wang, Yang You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01556](https://doi.org/10.1109/CVPR52688.2022.01556)

**Abstract**:

Recent self-supervised contrastive learning methods greatly benefit from the Siamese structure that aims at minimizing distances between positive pairs. For high performance Siamese representation learning, one of the keys is to design good contrastive pairs. Most previous works simply apply random sampling to make different crops of the same image, which overlooks the semantic information that may degrade the quality of views. In this work, we propose ContrastiveCrop, which could effectively generate better crops for Siamese representation learning. Firstly, a semantic-aware object localization strategy is proposed within the training process in a fully unsupervised manner. This guides us to generate contrastive views which could avoid most false positives (i.e., object vs. background). Moreover, we empirically find that views with similar appearances are trivial for the Siamese model training. Thus, a center-suppressed sampling is further designed to enlarge the variance of crops. Remarkably, our method takes a careful consideration of positive pairs for contrastive learning with negligible extra training overhead. As a plug-and-play and framework-agnostic module, ContrastiveCrop consistently improves SimCLR, MoCo, BYOL, SimSiam by 0.4% ∼ 2.0% classification accuracy on CIFAR-10, CIFAR-100, Tiny ImageNet and STL-10. Superior results are also achieved on downstream detection and segmentation tasks when pre-trained on ImageNet-1K.

----

## [1544] Rethinking Minimal Sufficient Representation in Contrastive Learning

**Authors**: *Haoqing Wang, Xun Guo, Zhi-Hong Deng, Yan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01557](https://doi.org/10.1109/CVPR52688.2022.01557)

**Abstract**:

Contrastive learning between different views of the data achieves outstanding success in the field of self-supervised representation learning and the learned representations are useful in broad downstream tasks. Since all supervision information for one view comes from the other view, contrastive learning approximately obtains the minimal sufficient representation which contains the shared information and eliminates the non-shared information between views. Considering the diversity of the downstream tasks, it cannot be guaranteed that all task-relevant information is shared between views. Therefore, we assume the non-shared task-relevant information cannot be ignored and theoretically prove that the minimal sufficient representation in contrastive learning is not sufficient for the downstream tasks, which causes performance degradation. This reveals a new problem that the contrastive learning models have the risk of overfitting to the shared information between views. To alleviate this problem, we propose to increase the mutual information between the representation and input as regularization to approximately introduce more task-relevant information, since we cannot utilize any downstream task information during training. Extensive experiments verify the rationality of our analysis and the effectiveness of our method. It significantly improves the performance of several classic contrastive learning models in downstream tasks. Our code is available at https://github.com/Haoqing-Wang/InfoCL.

----

## [1545] Multi-level Feature Learning for Contrastive Multi-view Clustering

**Authors**: *Jie Xu, Huayi Tang, Yazhou Ren, Liang Peng, Xiaofeng Zhu, Lifang He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01558](https://doi.org/10.1109/CVPR52688.2022.01558)

**Abstract**:

Multi-view clustering can explore common semantics from multiple views and has attracted increasing attention. However, existing works punish multiple objectives in the same feature space, where they ignore the conflict between learning consistent common semantics and reconstructing inconsistent view-private information. In this paper, we propose a new framework of multi-level feature learning for contrastive multi-view clustering to address the aforementioned issue. Our method learns different levels of features from the raw features, including low-level features, high-level features, and semantic labels/features in a fusion-free manner, so that it can effectively achieve the reconstruction objective and the consistency objectives in different feature spaces. Specifically, the reconstruction objective is conducted on the low-level features. Two consistency objectives based on contrastive learning are conducted on the high-level features and the semantic labels, respectively. They make the high-level features effectively explore the common semantics and the semantic labels achieve the multi-view clustering. As a result, the proposed framework can reduce the adverse influence of view-private information. Extensive experiments on public datasets demonstrate that our method achieves state-of-the-art clustering effectiveness.

----

## [1546] Point-Level Region Contrast for Object Detection Pre-Training

**Authors**: *Yutong Bai, Xinlei Chen, Alexander Kirillov, Alan L. Yuille, Alexander C. Berg*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01559](https://doi.org/10.1109/CVPR52688.2022.01559)

**Abstract**:

In this work we present point-level region contrast, a self-supervised pre-training approach for the task of object detection. This approach is motivated by the two key factors in detection: localization and recognition. While accurate localization favors models that operate at the pixel- or point-level, correct recognition typically relies on a more holistic, region-level view of objects. Incorporating this perspective in pre-training, our approach performs contrastive learning by directly sampling individual point pairs from different regions. Compared to an aggregated representation per region, our approach is more robust to the change in input region quality, and further enables us to implicitly improve initial region assignments via online knowledge distillation during training. Both advantages are important when dealing with imperfect regions encountered in the unsupervised setting. Experiments show point-level region contrast improves on state-of-the-art pre-training methods for object detection and segmentation across multiple tasks and datasets, and we provide extensive ablation studies and visualizations to aid understanding. Code will be made available.

----

## [1547] Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation

**Authors**: *Minsoo Kang, Jaeyoo Park, Bohyung Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01560](https://doi.org/10.1109/CVPR52688.2022.01560)

**Abstract**:

We present a novel class incremental learning approach based on deep neural networks, which continually learns new tasks with limited memory for storing examples in the previous tasks. Our algorithm is based on knowledge distillation and provides a principled way to maintain the representations of old models while adjusting to new tasks effectively. The proposed method estimates the relationship between the representation changes and the resulting loss increases incurred by model updates. It minimizes the upper bound of the loss increases using the representations, which exploits the estimated importance of each feature map within a backbone model. Based on the importance, the model restricts updates of important features for robustness while allowing changes in less critical features for flexibility. This optimization strategy effectively alleviates the notorious catastrophic forgetting problem despite the limited accessibility of data in the previous tasks. The experimental results show significant accuracy improvement of the proposed algorithm over the existing methods on the standard datasets. Code is available.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/kminsoo/AFC

----

## [1548] A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration

**Authors**: *Ramya Hebbalaguppe, Jatin Prakash, Neelabh Madan, Chetan Arora*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01561](https://doi.org/10.1109/CVPR52688.2022.01561)

**Abstract**:

Deep Neural Networks (dnns) are known to make over-confident mistakes, which makes their use problematic in safety-critical applications. State-of-the-art (sota) calibration techniques improve on the confidence of predicted labels alone, and leave the confidence of non-max classes (e.g. top-2, top-5) uncalibrated. Such calibration is not suitable for label refinement using post-processing. Further, most sota techniques learn a few hyper-parameters post-hoc, leaving out the scope for image, or pixel specific calibration. This makes them unsuitable for calibration under domain shift, or for dense prediction tasks like semantic segmentation. In this paper, we argue for intervening at the train time itself, so as to directly produce calibrated dnn models. We propose a novel auxiliary loss function: Multi-class Difference in Confidence and Accuracy (mdca), to achieve the same. mdca can be used in conjunction with other application/task specific loss functions. We show that training with mdca leads to better calibrated models in terms of Expected Calibration Error (ece), and Static Calibration Error (sce) on image classification, and segmentation tasks. We report ece (sce) score of 0.72 (1.60) on the cifar 100 dataset, in comparison to 1.90 (1.71) by the sota. Under domain shift, a ResNet-18 model trained on pacs dataset using mdca gives a average ece (sce) score of 19.7 (9.7) across all domains, compared to 24.2 (11.8) by the sota. For segmentation task, we report a 2× reduction in calibration error on pascal-voc dataset in comparison to Focal Loss [32]. Finally, mdca training improves calibration even on imbalanced data, and for natural language classification tasks.

----

## [1549] SLIC: Self-Supervised Learning with Iterative Clustering for Human Action Videos

**Authors**: *Salar Hosseini Khorasgani, Yuxuan Chen, Florian Shkurti*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01562](https://doi.org/10.1109/CVPR52688.2022.01562)

**Abstract**:

Self-supervised methods have significantly closed the gap with end-to-end supervised learning for image classification [13], [24]. In the case of human action videos, however, where both appearance and motion are significant factors of variation, this gap remains significant [28], [58]. One of the key reasons for this is that sampling pairs of similar video clips, a required step for many self-supervised contrastive learning methods, is currently done conservatively to avoid false positives. A typical assumption is that similar clips only occur temporally close within a single video, leading to insufficient examples of motion similarity. To mitigate this, we propose SLIC, a clustering-based self-supervised contrastive learning method for human action videos. Our key contribution is that we improve upon the traditional intra-video positive sampling by using iterative clustering to group similar video instances. This enables our method to leverage pseudo-labels from the cluster assignments to sample harder positives and negatives. SLIC outperforms state-of-the-art video retrieval baselines by 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$+15.4\%$</tex>
 on top-1 recall on UCF101 and by 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$+5.7\%$</tex>
 when directly transferred to HMDB51. With end-to-end finetuning for action classi-fication, SLIC achieves 83.2% top-1 accuracy 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(+0.8\%)$</tex>
 on UCF101 and 54.5% on HMDB51 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(+1.6\%$</tex>
,. SLIC is also competitive with the state-of-the-art in action classification after self-supervised pretraining on Kinetics400.

----

## [1550] Omnivore: A Single Model for Many Visual Modalities

**Authors**: *Rohit Girdhar, Mannat Singh, Nikhila Ravi, Laurens van der Maaten, Armand Joulin, Ishan Misra*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01563](https://doi.org/10.1109/CVPR52688.2022.01563)

**Abstract**:

Prior work has studied different visual modalities in isolation and developed separate architectures for recognition of images, videos, and 3D data. Instead, in this paper, we propose a single model which excels at classifying images, videos, and single-view 3D data using exactly the same model parameters. Our ‘OMNIVORE’ model leverages the flexibility of transformer-based architectures and is trained jointly on classification tasks from different modalities. Omnivoreis simple to train, uses off-the-shelf standard datasets, and performs at-par or better than modality-specific models of the same size. A single Omnivoremodel obtains 86.0% on ImageNet, 84.1% on Kinetics, and 67.1% on SUN RGB-D. After finetuning, our models outperform prior work on a variety of vision tasks and generalize across modalities. OMNIVORE's shared visual representation naturally enables cross-modal recognition without access to correspondences between modalities. We hope our results motivate researchers to model visual modalities together.

----

## [1551] DPICT: Deep Progressive Image Compression Using Trit-Planes

**Authors**: *Jae-Han Lee, Seungmin Jeon, Kwang Pyo Choi, Youngo Park, Chang-Su Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01564](https://doi.org/10.1109/CVPR52688.2022.01564)

**Abstract**:

We propose the deep progressive image compression using trit-planes (DPICT) algorithm, which is the first learning-based codec supporting fine granular scalability (FGS). First, we transform an image into a latent tensor using an analysis network. Then, we represent the latent tensor in ternary digits (trits) and encode it into a compressed bitstream trit-plane by trit-plane in the decreasing order of significance. Moreover, within each trit-plane, we sort the trits according to their rate-distortion priorities and transmit more important information first. Since the compression network is less optimized for the cases of using fewer tritplanes, we develop a postprocessing network for refining reconstructed images at low rates. Experimental results show that DPICT outperforms conventional progressive codecs significantly, while enabling FGS transmission. Codes are available at https://github.com/jaehanlee-mcl/DPICT.

----

## [1552] Efficient Geometry-aware 3D Generative Adversarial Networks

**Authors**: *Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J. Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, Gordon Wetzstein*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01565](https://doi.org/10.1109/CVPR52688.2022.01565)

**Abstract**:

Unsupervised generation of high-quality multi-view-consistent images and 3D shapes using only collections of single-view 2D photographs has been a long-standing challenge. Existing 3D GANs are either compute intensive or make approximations that are not 3D-consistent; the former limits quality and resolution of the generated images and the latter adversely affects multi-view consistency and shape quality. In this work, we improve the computational efficiency and image quality of 3D GANs without overly relying on these approximations. We introduce an expressive hybrid explicit implicit network architecture that, together with other design choices, synthesizes not only high-resolution multi-view-consistent images in real time but also produces high-quality 3D geometry. By decoupling feature generation and neural rendering, our framework is able to leverage state-of-the-art 2D CNN generators, such as StyleGAN2, and inherit their efficiency and expressiveness. We demonstrate state-of-the-art 3D-aware synthesis with FFHQ and AFHQ Cats, among other experiments.

----

## [1553] Geometric Anchor Correspondence Mining with Uncertainty Modeling for Universal Domain Adaptation

**Authors**: *Liang Chen, Yihang Lou, Jianzhong He, Tao Bai, Minghua Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01566](https://doi.org/10.1109/CVPR52688.2022.01566)

**Abstract**:

Universal domain adaptation (UniDA) aims to transfer the knowledge learned from a label-rich source domain to a label-scarce target domain without any constraints on the label space. However, domain shift and category shift make UniDA extremely challenging, which mainly lies in how to recognize both shared “known” samples and private “unknown” samples. Previous works rarely explore the intrinsic geometrical relationship between the two domains, and they manually set a threshold for the overconfident closed-world classifier to reject “unknown” samples. Therefore, in this paper, we propose a Geometric anchor-guided Adversarial and conTrastive learning framework with uncErtainty modeling called GATE to alleviate these issues. Specifically, we first develop a random walk-based anchor mining strategy together with a high-order attention mechanism to build correspondence across domains. Then a global joint local domain alignment paradigm is designed, i.e., geometric adversarial learning for global distribution calibration and subgraph-level contrastive learning for local region aggregation. Toward accurate target private samples detection, GATE introduces a universal incremental classifier by modeling the energy uncertainty. We further efficiently generate novel categories by manifold mixup, and minimize the open-set entropy to learn the “unknown” threshold adaptively. Extensive experiments on three benchmarks demonstrate that GATE significantly out-performs previous state-of-the-art UniDA methods.

----

## [1554] Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning

**Authors**: *Richard J. Chen, Chengkuan Chen, Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan, Faisal Mahmood*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01567](https://doi.org/10.1109/CVPR52688.2022.01567)

**Abstract**:

Vision Transformers (ViTs) and their multi-scale and hierarchical variations have been successful at capturing image representations but their use has been generally studied for low-resolution images (e.g. 256 × 256, 384 × 384). For gigapixel whole-slide imaging (WSI) in computational pathology, WSIs can be as large as 150000 × 150000 pixels at 20 × magnification and exhibit a hierarchical structure of visual tokens across varying resolutions: from 16 × 16 images capturing individual cells, to 4096 × 4096 images characterizing interactions within the tissue microenvironment. We introduce a new ViT architecture called the Hierarchical Image Pyramid Transformer (HIPT), which leverages the natural hierarchical structure inherent in WSIs using two levels of self-supervised learning to learn high-resolution image representations. HIPT is pretrained across 33 cancer types using 10,678 gigapixel WSIs, 408,218 4096 × 4096 images, and 104M 256 × 256 images. We benchmark HIPT representations on 9 slide-level tasks, and demonstrate that: 1) HIPT with hierarchical pretraining outperforms current state-of-the-art methods for cancer subtyping and survival prediction, 2) self-supervised ViTs are able to model important inductive biases about the hierarchical structure of phenotypes in the tumor microenvironment.

----

## [1555] Versatile Multi-Modal Pre-Training for Human-Centric Perception

**Authors**: *Fangzhou Hong, Liang Pan, Zhongang Cai, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01568](https://doi.org/10.1109/CVPR52688.2022.01568)

**Abstract**:

Human-centric perception plays a vital role in vision and graphics. But their data annotations are prohibitively expensive. Therefore, it is desirable to have a versatile pretrain model that serves as a foundation for data-efficient downstream tasks transfer. To this end, we propose the Human-Centric Multi-Modal Contrastive Learning framework HCMoCo that leverages the multi-modal nature of human data (e.g. RGB, depth, 2D keypoints) for effective representation learning. The objective comes with two main challenges: dense pre-train for multi-modality data, efficient usage of sparse human priors. To tackle the challenges, we design the novel Dense Intra-sample Contrastive Learning and Sparse Structure-aware Contrastive Learning targets by hierarchically learning a modal-invariant latent space featured with continuous and ordinal feature distribution and structure-aware semantic consistency. HCMoCo provides pre-train for different modalities by combining heterogeneous datasets, which allows efficient usage of existing task-specific human data. Extensive experiments on four downstream tasks of different modalities demonstrate the effectiveness of HCMoCo, especially under data-efficient settings (7.16% and 12% improvement on DensePose Estimation and Human Parsing). Moreover, we demonstrate the versatility of HCMoCo by exploring cross-modality super-vision and missing-modality inference, validating its strong ability in cross-modal association and reasoning. Codes are available at https://github.com/hongfz16/HCMoCo.

----

## [1556] Bridging Video-text Retrieval with Multiple Choice Questions

**Authors**: *Yuying Ge, Yixiao Ge, Xihui Liu, Dian Li, Ying Shan, Xiaohu Qie, Ping Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01569](https://doi.org/10.1109/CVPR52688.2022.01569)

**Abstract**:

Pretraining a model to learn transferable video-text representation for retrieval has attracted a lot of attention in recent years. Previous dominant works mainly adopt two separate encoders for efficient retrieval, but ignore local associations between videos and texts. Another line of research uses a joint encoder to interact video with texts, but results in low efficiency since each text-video pair needs to be fed into the model. In this work, we enable fine-grained video-text interactions while maintaining high efficiency for retrieval via a novel pretext task, dubbed as Multiple Choice Questions (MCQ), where a parametric module BridgeFormer is trained to answer the “questions” constructed by the text features via resorting to the video features. Specifically, we exploit the rich semantics of text (i.e., nouns and verbs) to build questions, with which the video encoder can be trained to capture more regional content and temporal dynamics. In the form of questions and answers, the semantic associations between local video-text features can be properly established. BridgeFormer is able to be removed for downstream retrieval, rendering an efficient and flexible model with only two encoders. Our method outperforms state-of-the-art methods on the popular text-to-video retrieval task in five datasets with different experimental setups (i.e., zero-shot andfine-tune), including HowTo100M (one million videos). We further conduct zero-shot action recognition, which can be cast as video-to-text retrieval, and our approach also significantly surpasses its counterparts. As an additional benefit, our method achieves competitive results with much shorter pre-training videos on single-modality downstream tasks, e.g., action recognition with linear evaluation.

----

## [1557] Integrating Language Guidance into Vision-based Deep Metric Learning

**Authors**: *Karsten Roth, Oriol Vinyals, Zeynep Akata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01570](https://doi.org/10.1109/CVPR52688.2022.01570)

**Abstract**:

Deep Metric Learning (DML) proposes to learn metric spaces which encode semantic similarities as embedding space distances. These spaces should be transferable to classes beyond those seen during training. Commonly, DML methods task networks to solve contrastive ranking tasks defined over binary class assignments. However, such approaches ignore higher-level semantic relations between the actual classes. This causes learned embedding spaces to encode incomplete semantic context and misrepresent the semantic relation between classes, impacting the generalizability of the learned metric space. To tackle this issue, we propose a language guidance objective for visual similarity learning. Leveraging language embeddings of expert- and pseudo-classnames, we contextualize and realign visual representation spaces corresponding to meaningful language semantics for better semantic consistency. Extensive experiments and ablations provide a strong motivation for our proposed approach and show language guidance offering significant, model-agnostic improvements for DML, achieving competitive and state-of-the-art results on all benchmarks. Code available at github.com/ExplainableML/LanguageGuidance-for_DML.

----

## [1558] NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images

**Authors**: *Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan, Jonathan T. Barron*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01571](https://doi.org/10.1109/CVPR52688.2022.01571)

**Abstract**:

Neural Radiance Fields (NeRF) is a technique for high quality novel view synthesis from a collection of posed input images. Like most view synthesis methods, NeRF uses tonemapped low dynamic range (LDR) as input; these images have been processed by a lossy camera pipeline that smooths detail, clips highlights, and distorts the simple noise distribution of raw sensor data. We modify NeRF to instead train directly on linear raw images, preserving the scene's full dynamic range. By rendering raw output images from the resulting NeRF, we can perform novel high dynamic range (HDR) view synthesis tasks. In addition to changing the camera viewpoint, we can manipulate focus, exposure, and tonemapping after the fact. Although a single raw image appears significantly more noisy than a postprocessed one, we show that NeRF is highly robust to the zeromean distribution of raw noise. When optimized over many noisy raw inputs (25–200), NeRF produces a scene representation so accurate that its rendered novel views outperform dedicated single and multi-image deep raw denoisers run on the same wide baseline input images. As a result, our method, which we call RawNeRF, can reconstruct scenes from extremely noisy images captured in near-darkness.

----

## [1559] DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering

**Authors**: *Liwen Wu, Jae Yong Lee, Anand Bhattad, Yu-Xiong Wang, David A. Forsyth*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01572](https://doi.org/10.1109/CVPR52688.2022.01572)

**Abstract**:

DIVeR builds on the key ideas of NeRF and its variants-density models and volume rendering – to learn 3D object models that can be rendered realistically from small numbers of images. In contrast to all previous NeRF methods, DIVeR uses deterministic rather than stochastic estimates of the volume rendering integral. DIVeR's representation is a voxel based field of features. To compute the volume rendering integral, a ray is broken into intervals, one per voxel; components of the volume rendering integral are estimated from the features for each interval using an MLP, and the components are aggregated. As a result, DIVeR can render thin translucent structures that are missed by other integrators. Furthermore, DIVeR's representation has semantics that is relatively exposed compared to other such methods – moving feature vectors around in the voxel space results in natural edits. Extensive qualitative and quantitative comparisons to current state-of-the-art methods show that DIVeR produces models that (1) render at or above state-of-the-art quality, (2) are very small without being baked, (3) render very fast without being baked, and (4) can be edited in natural ways. Our real-time code is available at: https://github.com/lwwu2/diver-rt

----

## [1560] HumanNeRF: Free-viewpoint Rendering of Moving People from Monocular Video

**Authors**: *Chung-Yi Weng, Brian Curless, Pratul P. Srinivasan, Jonathan T. Barron, Ira Kemelmacher-Shlizerman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01573](https://doi.org/10.1109/CVPR52688.2022.01573)

**Abstract**:

We introduce a free-viewpoint rendering method - HumanNeRF - that works on a given monocular video of a human performing complex body motions, e.g. a video from YouTube. Our method enables pausing the video at any frame and rendering the subject from arbitrary new camera viewpoints or even a full 360-degree camera path for that particular frame and body pose. This task is particularly challenging, as it requires synthesizing photorealistic details of the body, as seen from various camera angles that may not exist in the input video, as well as synthesizing fine details such as cloth folds and facial appearance. Our method optimizes for a volumetric representation of the person in a canonical T-pose, in concert with a motion field that maps the estimated canonical representation to every frame of the video via backward warps. The motion field is decomposed into skeletal rigid and non-rigid motions, produced by deep networks. We show significant performance improvements over prior work, and compelling examples of free-viewpoint renderings from monocular video of moving humans in challenging uncontrolled capture scenarios.

----

## [1561] Neural Reflectance for Shape Recovery with Shadow Handling

**Authors**: *Junxuan Li, Hongdong Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01574](https://doi.org/10.1109/CVPR52688.2022.01574)

**Abstract**:

This paper aims at recovering the shape of a scene with unknown, non-Lambertian, and possibly spatially-varying surface materials. When the shape of the object is highly complex and that shadows cast on the surface, the task becomes very challenging. To overcome these challenges, we propose a coordinate-based deep MLP (multilayer perceptron) to parameterize both the unknown 3D shape and the unknown reflectance at every surface point. This network is able to leverage the observed photometric variance and shadows on the surface, and recover both surface shape and general non-Lambertian reflectance. We explicitly predict cast shadows, mitigating possible artifacts on these shadowing regions, leading to higher estimation accuracy. Our framework is entirely self-supervised, in the sense that it requires neither ground truth shape nor BRDF. Tests on real-world images demonstrate that our method outperform existing methods by a significant margin. Thanks to the small size of the MLP-net, our method is an order of magnitude faster than previous CNN-based methods.

----

## [1562] Visual Vibration Tomography: Estimating Interior Material Properties from Monocular Video

**Authors**: *Berthy T. Feng, Alexander C. Ogren, Chiara Daraio, Katherine L. Bouman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01575](https://doi.org/10.1109/CVPR52688.2022.01575)

**Abstract**:

An object's interior material properties, while invisible to the human eye, determine motion observed on its surface. We propose an approach that estimates heterogeneous material properties of an object from a monocular video of its surface vibrations. Specifically, we show how to estimate Young's modulus and density throughout a 3D object with known geometry. Knowledge of how these values change across the object is useful for simulating its motion and characterizing any defects. Traditional nondestructive testing approaches, which often require expensive instruments, generally estimate only homogenized material properties or simply identify the presence of defects. In contrast, our approach leverages monocular video to (1) identify image-space modes from an object's sub-pixel motion, and (2) directly infer spatially-varying Young's modulus and density values from the observed modes. We demonstrate our approach on both simulated and real videos.

----

## [1563] Dancing under the stars: video denoising in starlight

**Authors**: *Kristina Monakhova, Stephan R. Richter, Laura Waller, Vladlen Koltun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01576](https://doi.org/10.1109/CVPR52688.2022.01576)

**Abstract**:

Imaging in low light is extremely challenging due to low photon counts. Using sensitive CMOS cameras, it is currently possible to take videos at night under moonlight (0.05-0.3 lux illumination). In this paper, we demonstrate photorealistic video under starlight (no moon present, <0.001 lux) for the first time. To enable this, we develop a GAN-tuned physics-based noise model to more accurately represent camera noise at the lowest light levels. Using this noise model, we train a video denoiser using a combination of simulated noisy video clips and real noisy still images. We capture a 5–10 fps video dataset with significant motion at approximately 0.6-0.7 millilux with no active illumination. Comparing against alternative methods, we achieve improved video quality at the lowest light levels, demonstrating photorealistic video denoising in starlight for the first time.

----

## [1564] Bacon: Band-limited Coordinate Networks for Multiscale Scene Representation

**Authors**: *David B. Lindell, Dave Van Veen, Jeong Joon Park, Gordon Wetzstein*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01577](https://doi.org/10.1109/CVPR52688.2022.01577)

**Abstract**:

Coordinate-based networks have emerged as a powerful tool for 3D representation and scene reconstruction. These networks are trained to map continuous input coordinates to the value of a signal at each point. Still, current architectures are black boxes: their spectral characteristics cannot be easily analyzed, and their behavior at unsupervised points is difficult to predict. Moreover, these networks are typically trained to represent a signal at a single scale, so naive downsampling or upsampling results in artifacts. We introduce band-limited coordinate networks (BACON), a network architecture with an analytical Fourier spectrum. Bacon has constrained behavior at unsupervised points, can be designed based on the spectral characteristics of the represented signal, and can represent signals at multiple scales without per-scale supervision. We demonstrate Bacon for multiscale neural representation of images, radiance fields, and 3D scenes using signed distance functions and show that it outperforms conventional single-scale coordinate networks in terms of interpretability and quality.

----

## [1565] Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation

**Authors**: *Jiankun Li, Peisen Wang, Pengfei Xiong, Tao Cai, Ziwei Yan, Lei Yang, Jiangyu Liu, Haoqiang Fan, Shuaicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01578](https://doi.org/10.1109/CVPR52688.2022.01578)

**Abstract**:

With the advent of convolutional neural networks, stereo matching algorithms have recently gained tremendous progress. However, it remains a great challenge to accurately extract disparities from real-world image pairs taken by consumer-level devices like smartphones, due to practical complicating factors such as thin structures, non-ideal rectification, camera module inconsistencies and various hard-case scenes. In this paper, we propose a set of innovative designs to tackle the problem of practical stereo matching: 1) to better recover fine depth details, we design a hierarchical network with recurrent refinement to update disparities in a coarse-to-fine manner, as well as a stacked cascaded architecture for inference; 2) we propose an adaptive group correlation layer to mitigate the impact of erroneous rectification; 3) we introduce a new synthetic dataset with special attention to difficult cases for better generalizing to real-world scenes. Our results not only rank 1
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">st</sup>
 on both Middlebury and ETH3D benchmarks, outperforming existing state-of-the-art methods by a notable margin, but also exhibit high-quality details for real-life photos, which clearly demonstrates the efficacy of our contributions.

----

## [1566] 3D Photo Stylization: Learning to Generate Stylized Novel Views from a Single Image

**Authors**: *Fangzhou Mu, Jian Wang, Yicheng Wu, Yin Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01579](https://doi.org/10.1109/CVPR52688.2022.01579)

**Abstract**:

Visual content creation has spurred a soaring interest given its applications in mobile photography and AR / VR. Style transfer and single-image 3D photography as two representative tasks have so far evolved independently. In this paper, we make a connection between the two, and address the challenging task of 3D photo stylization - generating stylized novel views from a single image given an arbitrary style. Our key intuition is that style transfer and view synthesis have to be jointly modeled. To this end, we propose a deep model that learns geometry-aware content features for stylization from a point cloud representation of the scene, resulting in high-quality stylized images that are consistent across views. Further, we introduce a novel training protocol to enable the learning using only 2D images. We demonstrate the superiority of our method via extensive qualitative and quantitative studies, and showcase key applications of our method in light of the growing demand for 3D content creation from 2D image assets. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: http://pages.es.wise.edu/-fmu/style3d

----

## [1567] BokehMe: When Neural Rendering Meets Classical Rendering

**Authors**: *Juewen Peng, Zhiguo Cao, Xianrui Luo, Hao Lu, Ke Xian, Jianming Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01580](https://doi.org/10.1109/CVPR52688.2022.01580)

**Abstract**:

We propose BokehMe, a hybrid bokeh rendering framework that marries a neural renderer with a classical physically motivated renderer. Given a single image and a potentially imperfect disparity map, BokehMe generates high-resolution photo-realistic bokeh effects with adjustable blur size, focal plane, and aperture shape. To this end, we analyze the errors from the classical scattering-based method and derive a formulation to calculate an error map. Based on this formulation, we implement the classical renderer by a scattering-based method and propose a two-stage neural renderer to fix the erroneous areas from the classical renderer. The neural renderer employs a dynamic multi-scale scheme to efficiently handle arbitrary blur sizes, and it is trained to handle imperfect disparity input. Experiments show that our method compares favorably against previous methods on both synthetic image data and real image data with predicted disparity. A user study is further conducted to validate the advantage of our method.

----

## [1568] Deblurring via Stochastic Refinement

**Authors**: *Jay Whang, Mauricio Delbracio, Hossein Talebi, Chitwan Saharia, Alexandros G. Dimakis, Peyman Milanfar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01581](https://doi.org/10.1109/CVPR52688.2022.01581)

**Abstract**:

Image deblurring is an ill-posed problem with multiple plausible solutions for a given input image. However, most existing methods produce a deterministic estimate of the clean image and are trained to minimize pixel-level distortion. These metrics are known to be poorly correlated with human perception, and often lead to unrealistic reconstructions. We present an alternative framework for blind deblurring based on conditional diffusion models. Unlike existing techniques, we train a stochastic sampler that refines the output of a deterministic predictor and is capable of producing a diverse set of plausible reconstructions for a given input. This leads to a significant improvement in perceptual quality over existing state-of-the-art methods across multiple standard benchmarks. Our predict-and-refine approach also enables much more efficient sampling compared to typical diffusion models. Combined with a carefully tuned network architecture and inference procedure, our method is competitive in terms of distortion metrics such as PSNR. These results show clear benefits of our diffusion-based method for deblurring and challenge the widely used strategy of producing a single, deterministic reconstruction.

----

## [1569] Learning to Deblur using Light Field Generated and Real Defocus Images

**Authors**: *Lingyan Ruan, Bin Chen, Jizhou Li, Miu-Ling Lam*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01582](https://doi.org/10.1109/CVPR52688.2022.01582)

**Abstract**:

Defocus deblurring is a challenging task due to the spatially varying nature of defocus blur. While deep learning approach shows great promise in solving image restoration problems, defocus deblurring demands accurate training data that consists of all-in-focus and defocus image pairs, which is difficult to collect. Naive two-shot capturing cannot achieve pixel-wise correspondence between the defocused and all-in-focus image pairs. Synthetic aperture of light fields is suggested to be a more reliable way to generate accurate image pairs. However, the defocus blur generated from light field data is different from that of the images captured with a traditional digital camera. In this paper, we propose a novel deep defocus deblurring network that leverages the strength and overcomes the shortcoming of light fields. We first train the network on a light field-generated dataset for its highly accurate image correspondence. Then, we fine-tune the network using feature loss on another dataset collected by the two-shot method to alleviate the differences between the defocus blur exists in the two domains. This strategy is proved to be highly effective and able to achieve the state-of-the-art performance both quantitatively and qualitatively on multiple test sets. Extensive ablation studies have been conducted to analyze the effect of each network module to the final performance.

----

## [1570] Towards Layer-wise Image Vectorization

**Authors**: *Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, Humphrey Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01583](https://doi.org/10.1109/CVPR52688.2022.01583)

**Abstract**:

Image rasterization is a mature technique in computer graphics, while image vectorization, the reverse path of rasterization, remains a major challenge. Recent advanced deep learning-based models achieve vectorization and semantic interpolation of vector graphs and demonstrate a better topology of generating new figures. However, deep models cannot be easily generalized to out-of-domain testing data. The generated SVGs also contain complex and redundant shapes that are not quite convenient for further editing. Specifically, the crucial layer-wise topology and fundamental semantics in images are still not well understood and thus not fully explored. In this work, we propose Layer-wise Image Vectorization, namely LIVE, to convert raster images to SVGs and simultaneously maintain its image topology. LIVE can generate compact SVG forms with layer-wise structures that are semantically consistent with human perspective. We progressively add new bezier paths and optimize these paths with the layer-wise framework, newly designed loss functions, and component-wise path initialization technique. Our experiments demonstrate that LIVE presents more plausible vectorized forms than prior works and can be generalized to new images. With the help of this newly learned topology, LIVE initiates human editable SVGs for both designers and other downstream applications. Codes are made available at https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization.

----

## [1571] Dual-Shutter Optical Vibration Sensing

**Authors**: *Mark Sheinin, Dorian Chan, Matthew O'Toole, Srinivasa G. Narasimhan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01584](https://doi.org/10.1109/CVPR52688.2022.01584)

**Abstract**:

Visual vibrometry is a highly useful tool for remote capture of audio, as well as the physical properties of materials, human heart rate, and more. While visually-observable vibrations can be captured directly with a high-speed camera, minute imperceptible object vibrations can be optically amplified by imaging the displacement of a speckle pattern, created by shining a laser beam on the vibrating surface. In this paper, we propose a novel method for sensing vibrations at high speeds (up to 63kHz), for multiple scene sources at once, using sensors rated for only 130Hz operation. Our method relies on simultaneously capturing the scene with two cameras equipped with rolling and global shutter sensors, respectively. The rolling shutter camera captures distorted speckle images that encode the high-speed object vibrations. The global shutter camera captures undistorted reference images of the speckle pattern, helping to decode the source vibrations. We demonstrate our method by capturing vibration caused by audio sources (e.g. speakers, human voice, and musical instruments) and analyzing the vibration modes of a tuning fork.

----

## [1572] Fisher Information Guidance for Learned Time-of-Flight Imaging

**Authors**: *Jiaqu Li, Tao Yue, Sijie Zhao, Xuemei Hu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01585](https://doi.org/10.1109/CVPR52688.2022.01585)

**Abstract**:

Indirect Time-of-Flight (ToF) imaging is widely applied in practice for its superiorities on cost and spatial resolution. However, lower signal-to-noise ratio (SNR) of measurement leads to larger error in ToF imaging, especially for imaging scenes with strong ambient light or long distance. In this paper, we propose a Fisher-information guided framework to jointly optimize the coding functions (light modulation and sensor demodulation functions) and the reconstruction network of iToF imaging, with the super-vision of the proposed discriminative fisher loss. By introducing the differentiable modeling of physical imaging process considering various real factors and constraints, e.g., light-falloff with distance, physical implementability of coding functions, etc., followed by a dual-branch depth reconstruction neural network, the proposed method could learn the optimal iToF imaging system in an end-to-end manner. The effectiveness of the proposed method is extensively verified with both simulations and prototype experiments.

----

## [1573] Autofocus for Event Cameras

**Authors**: *Shijie Lin, Yinqiang Zhang, Lei Yu, Bin Zhou, Xiaowei Luo, Jia Pan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01586](https://doi.org/10.1109/CVPR52688.2022.01586)

**Abstract**:

Focus control (FC) is crucial for cameras to capture sharp images in challenging real-world scenarios. The autofocus (AF) facilitates the FC by automatically adjusting the focus settings. However, due to the lack of effective AF methods for the recently introduced event cameras, their FC still relies on naive AF like manual focus adjustments, leading to poor adaptation in challenging real-world conditions. In particular, the inherent differences between event and frame data in terms of sensing modality, noise, temporal resolutions, etc., bring many challenges in designing an effective AF method for event cameras. To address these challenges, we develop a novel event-based autofocus framework consisting of an event-specific focus measure called event rate (ER) and a robust search strategy called event-based golden search (EGS). To verify the performance of our method, we have collected an event-based autofocus dataset (EAD) containing well-synchronized frames, events, and focal positions in a wide variety of challenging scenes with severe lighting and motion conditions. The experiments on this dataset and additional real-world scenarios demonstrated the superiority of our method over state-of-the-art approaches in terms of efficiency and accuracy.

----

## [1574] Adaptive Gating for Single-Photon 3D Imaging

**Authors**: *Ryan Po, Adithya Pediredla, Ioannis Gkioulekas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01587](https://doi.org/10.1109/CVPR52688.2022.01587)

**Abstract**:

Single-photon avalanche diodes (SPADs) are growing in popularity for depth sensing tasks. However, SPADs still struggle in the presence of high ambient light due to the effects of pile-up. Conventional techniques leverage fixed or asynchronous gating to minimize pile-up effects, but these gating schemes are all non-adaptive, as they are unable to incorporate factors such as scene priors and previous photon detections into their gating strategy. We propose an adaptive gating scheme built upon Thompson sampling. Adaptive gating periodically updates the gate position based on prior photon observations in order to minimize depth errors. Our experiments show that our gating strategy results in significantly reduced depth reconstruction error and acquisition time, even when operating outdoors under strong sunlight conditions.

----

## [1575] LiDAR Snowfall Simulation for Robust 3D Object Detection

**Authors**: *Martin Hahner, Christos Sakaridis, Mario Bijelic, Felix Heide, Fisher Yu, Dengxin Dai, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01588](https://doi.org/10.1109/CVPR52688.2022.01588)

**Abstract**:

3D object detection is a central task for applications such as autonomous driving, in which the system needs to localize and classify surrounding traffic agents, even in the presence of adverse weather. In this paper, we address the problem of LiDAR-based 3D object detection under snow-fall. Due to the difficulty of collecting and annotating training data in this setting, we propose a physically based method to simulate the effect of snowfall on real clear-weather LiDAR point clouds. Our method samples snow particles in 2D space for each LiDAR line and uses the in-duced geometry to modify the measurement for each LiDAR beam accordingly. Moreover, as snowfall often causes wet-ness on the ground, we also simulate ground wetness on LiDAR point clouds. We use our simulation to generate par-tially synthetic snowy LiDAR data and leverage these data for training 3D object detection models that are robust to snowfall. We conduct an extensive evaluation using several state-of-the-art 3D object detection methods and show that our simulation consistently yields significant performance gains on the real snowy STF dataset compared to clear-weather baselines and competing simulation approaches, while not sacrificing performance in clear weather. Our code is available at github.com/SysCV/LiDAR_snow_sim.

----

## [1576] MERLOT RESERVE: Neural Script Knowledge through Vision and Language and Sound

**Authors**: *Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, Yejin Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01589](https://doi.org/10.1109/CVPR52688.2022.01589)

**Abstract**:

As humans, we navigate a multimodal world, building a holistic understanding from all our senses. We introduce @MERLOT RESERVE, a model that represents videos jointly over time - through a new training objective that learns from audio, subtitles, and video frames. Given a video, we replace snippets of text and audio with a MASK token; the model learns by choosing the correct masked-out snippet. Our objective learns faster than alternatives, and performs well at scale: we pretrain on 20 million YouTube videos. Empirical results show that @MERLOT RESERVE learns strong multimodal representations. When finetuned, it sets state-of-the-art on Visual Commonsense Reasoning (VCR), TVQA, and Kinetics-600; outperforming prior work by 5%, 7%, and 1.5% respectively. Ablations show that these tasks benefit from audio pretraining - even VCR, a QA task centered around images (without sound). Moreover, our objective enables out-of-the-box prediction, revealing strong multimodal commonsense understanding. In a fully zero-shot setting, our model obtains competitive results on four video tasks, even outperforming supervised approaches on the recently proposed Situated Reasoning (STAR) benchmark. We analyze why audio enables better vision-language representations, suggesting significant opportunities for future research. We conclude by discussing ethical and societal implications of multimodal pretraining.

----

## [1577] Joint Video Summarization and Moment Localization by Cross-Task Sample Transfer

**Authors**: *Hao Jiang, Yadong Mu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01590](https://doi.org/10.1109/CVPR52688.2022.01590)

**Abstract**:

Video summarization has recently engaged increasing attention in computer vision communities. However, the scarcity of annotated data has been a key obstacle in this task. To address it, this work explores a new solution for video summarization by transferring samples from a correlated task (i.e., video moment localization) equipped with abundant training data. Our main insight is that the annotated video moments also indicate the semantic highlights of a video, essentially similar to video summary. Approximately, the video summary can be treated as a sparse, redundancy-free version of the video moments. Inspired by this observation, we propose an importance Propagation based collaborative Teaching Network (iPTNet). It consists of two separate modules that conduct video summarization and moment localization, respectively. Each module estimates a frame-wise importance map for indicating keyframes or moments. To perform cross-task sample transfer, we devise an importance propagation module that realizes the conversion between summarization-guided and localization-guided importance maps. This way critically enables optimizing one of the tasks using the data from the other task. Additionally, in order to avoid error amplification caused by batch-wise joint training, we devise a collaborative teaching scheme, which adopts a crosstask mean teaching strategy to realize the joint optimization of the two tasks and provide robust frame-level teaching signals. Extensive experiments on video summarization benchmarks demonstrate that iPTNet significantly outperforms previous state-of-the-art video summarization methods, serving as an effective solution that overcomes the data scarcity issue in video summarization.

----

## [1578] Towards General Purpose Vision Systems: An End-to-End Task-Agnostic Vision-Language Architecture

**Authors**: *Tanmay Gupta, Amita Kamath, Aniruddha Kembhavi, Derek Hoiem*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01591](https://doi.org/10.1109/CVPR52688.2022.01591)

**Abstract**:

Computer vision systems today are primarily N-purpose systems, designed and trained for a predefined set of tasks. Adapting such systems to new tasks is challenging and often requires nontrivial modifications to the network architecture (e.g. adding new output heads) or training process (e.g. adding new losses). To reduce the time and expertise required to develop new applications, we would like to create general purpose vision systems that can learn and perform a range of tasks without any modification to the architecture or learning process. In this paper, we propose GPV-1, a task-agnostic vision-language architecture that can learn and perform tasks that involve receiving an image and producing text and/or bounding boxes, including classification, localization, visual question answering, captioning, and more. We also propose evaluations of generality of architecture, skill-concept
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
For this work, we define concepts, skills and tasks as follows: Concepts - nouns (e.g. car, person, dog), Skills - operations that we wish to perform on the given inputs (e.g. classification, object detection, image captioning), Tasks - predefined combinations of a set of skills performed on a set of concepts (e.g. ImageNet classification task involves the skill of image classification across 1000 concepts). transfer, and learning efficiency that may informfuture work on general purpose vision. Our experiments indicate GPV-1 is effective at multiple tasks, reuses some concept knowledge across tasks, can perform the Referring Expressions task zero-shot, and further improves upon the zero-shot performance using a few training samples.

----

## [1579] Disentangling visual and written concepts in CLIP

**Authors**: *Joanna Materzynska, Antonio Torralba, David Bau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01592](https://doi.org/10.1109/CVPR52688.2022.01592)

**Abstract**:

The CLIP network measures the similarity between natural text and images; in this work, we investigate the entanglement of the representation of word images and natural images in its image encoder. First, we find that the image encoder has an ability to match word images with natural images of scenes described by those words. This is consistent with previous research that suggests that the meaning and the spelling of a word might be entangled deep within the network. On the other hand, we also find that CLIP has a strong ability to match nonsense words, suggesting that processing of letters is separated from processing of their meaning. To explicitly determine whether the spelling capability of CLIP is separable, we devise a procedure for identifying representation subspaces that selectively isolate or eliminate spelling capabilities. We benchmark our methods against a range of retrieval tasks, and we also test them by measuring the appearance of text in CLIP-guided generated images. We find that our methods are able to cleanly separate spelling capabilities of CLIP from the visual processing of natural images.

----

## [1580] CLIP-Event: Connecting Text and Images with Event Structures

**Authors**: *Manling Li, Ruochen Xu, Shuohang Wang, Luowei Zhou, Xudong Lin, Chenguang Zhu, Michael Zeng, Heng Ji, Shih-Fu Chang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01593](https://doi.org/10.1109/CVPR52688.2022.01593)

**Abstract**:

Vision-language (V+L) pretraining models have achieved great success in supporting multimedia applications by understanding the alignments between images and text. While existing vision-language pretraining models primarily focus on understanding objects in images or entities in text, they often ignore the alignment at the level of events and their argument structures. In this work, we propose a contrastive learning framework to enforce vision-language pretraining models to comprehend events and associated argument (participant) roles. To achieve this, we take advantage of text information extraction technologies to obtain event structural knowledge, and utilize multiple prompt functions to contrast difficult negative descriptions by manipulating event structures. We also design an event graph alignment loss based on optimal transport to capture event argument structures. In addition, we collect a large event-rich dataset (106,875 images) for pretraining, which provides a more challenging image retrieval benchmark to assess the understanding of complicated lengthy sentences
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The data and code are publicly available for research purpose in https://github.com/limanling/clip-event.. Experiments show that our zero-shot CLIP-Event outperforms the state-of-the-art supervised model in argument extraction on Multimedia Event Extraction, achieving more than 5% absolute F-score gain in event extraction, as well as significant improvements on a variety of downstream tasks under zero-shot settings.

----

## [1581] Robust Cross-Modal Representation Learning with Progressive Self-Distillation

**Authors**: *Alex Andonian, Shixing Chen, Raffay Hamid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01594](https://doi.org/10.1109/CVPR52688.2022.01594)

**Abstract**:

The learning objective of vision-language approach of CLIP [63] does not effectively account for the noisy many-to-many correspondences found in web-harvested image captioning datasets, which contributes to its compute and data inefficiency. To address this challenge, we introduce a novel training framework based on cross-modal contrastive learning that uses progressive self-distillation and soft image-text alignments to more efficiently learn robust representations from noisy data. Our model distills its own knowledge to dynamically generate soft-alignment targets for a subset of images and captions in every minibatch, which are then used to update its parameters. Extensive evaluation across 14 benchmark datasets shows that our method consistently outperforms its CLIP counterpart in multiple settings, including: (a) zero-shot classification, (b) linear probe transfer, and (c) image-text retrieval, without incurring extra computational cost. Analysis using an ImageNet-based robustness test-bed [70] reveals that our method offers better effective robustness to natural distribution shifts compared to both ImageNet-trained models and CLIP itself. Lastly, pretraining with datasets spanning two orders of magnitude in size shows that our improvements over CLIP tend to scale with number of training examples.

----

## [1582] TubeDETR: Spatio-Temporal Video Grounding with Transformers

**Authors**: *Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, Cordelia Schmid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01595](https://doi.org/10.1109/CVPR52688.2022.01595)

**Abstract**:

We consider the problem of localizing a spatio-temporal tube in a video corresponding to a given text query. This is a challenging task that requires the joint and efficient modeling of temporal, spatial and multi-modal interactions. To address this task, we propose TubeDETR, a transformer-based architecture inspired by the recent success of such models for text-conditioned object detection. Our model notably includes: (i) an efficient video and text encoder that models spatial multi-modal interactions over sparsely sampled frames and (ii) a space-time decoder that jointly performs spatio-temporal localization. We demonstrate the advantage of our proposed components through an extensive ablation study. We also evaluate our full approach on the spatio-temporal video grounding task and demonstrate improvements over the state of the art on the challenging VidSTG and HC-STVG benchmarks.

----

## [1583] 3D-SPS: Single-Stage 3D Visual Grounding via Referred Point Progressive Selection

**Authors**: *Junyu Luo, Jiahui Fu, Xianghao Kong, Chen Gao, Haibing Ren, Hao Shen, Huaxia Xia, Si Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01596](https://doi.org/10.1109/CVPR52688.2022.01596)

**Abstract**:

3D visual grounding aims to locate the referred target object in 3D point cloud scenes according to a free-form language description. Previous methods mostly follow a two-stage paradigm, i.e., language-irrelevant detection and cross-modal matching, which is limited by the isolated architecture. In such a paradigm, the detector needs to sample keypoints from raw point clouds due to the inherent properties of 3D point clouds (irregular and large-scale), to generate the corresponding object proposal for each keypoint. However, sparse proposals may leave out the target in detection, while dense proposals may confuse the matching model. Moreover, the language-irrelevant detection stage can only sample a small proportion of keypoints on the target, deteriorating the target prediction. In this paper, we propose a 3D Single-Stage Referred Point Progressive Selection (3D-SPS) method, which progressively selects keypoints with the guidance of language and directly locates the target. Specifically, we propose a Description-aware Keypoint Sampling (DKS) module to coarsely focus on the points of language-relevant objects, which are significant clues for grounding. Besides, we devise a Target-oriented Progressive Mining (TPM) module to finely concentrate on the points of the target, which is enabled by progressive intra-modal relation modeling and inter-modal target mining. 3D-SPS bridges the gap between detection and matching in the 3D visual grounding task, localizing the target at a single stage. Experiments demonstrate that 3D-SPS achieves state-of-the-art performance on both ScanRe-fer and Nr3D/Sr3D datasets.

----

## [1584] 3DJCG: A Unified Framework for Joint Dense Captioning and Visual Grounding on 3D Point Clouds

**Authors**: *Daigang Cai, Lichen Zhao, Jing Zhang, Lu Sheng, Dong Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01597](https://doi.org/10.1109/CVPR52688.2022.01597)

**Abstract**:

Observing that the 3D captioning task and the 3D grounding task contain both shared and complementary information in nature, in this work, we propose a unified framework to jointly solve these two distinct but closely related tasks in a synergistic fashion, which consists of both shared task-agnostic modules and lightweight task-specific modules. On one hand, the shared task-agnostic modules aim to learn precise locations of objects, fine-grained attribute features to characterize different objects, and complex relations between objects, which benefit both captioning and visual grounding. On the other hand, by casting each of the two tasks as the proxy task of another one, the lightweight task-specific modules solve the captioning task and the grounding task respectively. Extensive experiments and ablation study on three 3D vision and language datasets demonstrate that our joint training frame-work achieves significant performance gains for each individual task and finally improves the state-of-the-art performance for both captioning and grounding tasks.

----

## [1585] Globetrotter: Connecting Languages by Connecting Images

**Authors**: *Dídac Surís, Dave Epstein, Carl Vondrick*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01598](https://doi.org/10.1109/CVPR52688.2022.01598)

**Abstract**:

Machine translation between many languages at once is highly challenging, since training with ground truth re-quires supervision between all language pairs, which is dif-ficult to obtain. Our key insight is that, while languages may vary drastically, the underlying visual appearance of the world remains consistent. We introduce a method that uses visual observations to bridge the gap between languages, rather than relying on parallel corpora or topo-logical properties of the representations. We train a model that aligns segments of text from different languages if and only if the images associated with them are similar and each image in turn is well-aligned with its textual description. We train our model from scratch on a new dataset of text in over fifty languages with accompanying images. Experiments show that our method outperforms previous work on unsupervised word and sentence translation using retrieval. Code, models and data are available on globetrotter.cs.columbia.edu

----

## [1586] Unsupervised Vision-and-Language Pretraining via Retrieval-based Multi-Granular Alignment

**Authors**: *Mingyang Zhou, Licheng Yu, Amanpreet Singh, Mengjiao Wang, Zhou Yu, Ning Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01599](https://doi.org/10.1109/CVPR52688.2022.01599)

**Abstract**:

Vision-and-Language (V+L) pre-training models have achieved tremendous success in recent years on various multi-modal benchmarks. However, the majority of existing models require pre-training on a large set of parallel imagetext data, which is costly to collect, compared to image-only or text-only data. In this paper, we explore unsupervised Vision-and-Language pre-training (UVLP) to learn the cross-modal representation from non-parallel image and text datasets. We found two key factors that lead to good unsupervised V + L pre-training without parallel data: (i) joint image-and-text input (ii) overall imagetext alignment (even for non-parallel data). Accordingly, we propose a novel unsupervised V + L pre-training curriculum for non-parallel texts and images. We first construct a weakly aligned imagetext corpus via a retrieval-based approach, then apply a set of multi-granular alignment pre-training tasks, including region-to-tag, region-to-phrase, and image-to-sentence alignment, to bridge the gap between the two modalities. A comprehensive ablation study shows each granularity is helpful to learn a stronger pre-trained model. We adapt our pre-trained model to a set of V+L downstream tasks, including VQA, NLVR2, Visual Entailment, and Ref-COCO+. Our model achieves the state-of-art performance in all these tasks under the unsupervised setting.

----

## [1587] WebQA: Multihop and Multimodal QA

**Authors**: *Yingshan Chang, Guihong Cao, Mridu Narang, Jianfeng Gao, Hisami Suzuki, Yonatan Bisk*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01600](https://doi.org/10.1109/CVPR52688.2022.01600)

**Abstract**:

Scaling Visual Question Answering (VQA) to the open-domain and multi-hop nature of web searches, requires fundamental advances in visual representation learning, knowledge aggregation, and language generation. In this work, we introduce WEBQA, a challenging new benchmark that proves difficult for large-scale state-of-the-art models which lack language groundable visual representations for novel objects and the ability to reason, yet trivial for humans. WebQA mirrors the way humans use the web: 1) Ask a question, 2) Choose sources to aggregate, and 3) Produce a fluent language response. This is the behavior we should be expecting from IoT devices and digital assistants. Existing work prefers to assume that a model can either reason about knowledge in images or in text. WebQA includes a secondary text-only QA task to ensure improved visual performance does not come at the cost of language understanding. Our challenge for the community is to create unified multimodal reasoning models that answer questions regardless of the source modality, moving us closer to digital assistants that not only query language knowledge, but also the richer visual online world.

----

## [1588] PartGlot: Learning Shape Part Segmentation from Language Reference Games

**Authors**: *Juil Koo, Ian Huang, Panos Achlioptas, Leonidas J. Guibas, Minhyuk Sung*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01601](https://doi.org/10.1109/CVPR52688.2022.01601)

**Abstract**:

We introduce PartGlot, a neural framework and associated architectures for learning semantic part segmentation of 3D shape geometry, based solely on part referential language. We exploit the fact that linguistic descriptions of a shape can provide priors on the shape's parts - as natural language has evolved to reflect human perception of the compositional structure of objects, essential to their recognition and use. For training we use ShapeGlot's paired geometry /language data collected via a reference game where a speaker produces an utterance to differentiate a target shape from two distractors and the listener has to find the target based on this utterance [3]. Our network is designed to solve this target multi-modal recognition problem, by carefully incorporating a Transformer-based attention module so that the output attention can precisely highlight the semantic part or parts described in the language. Remarkably, the network operates without any direct supervision on the 3D geometry itself. Furthermore, we also demonstrate that the learned part information is generaliz-able to shape classes unseen during training. Our approach opens the possibility of learning 3D shape parts from language alone, without the need for large-scale part geometry annotations, thus facilitating annotation acquisition. The code is available at https://github.com/63days/PartGlot.

----

## [1589] DF-GAN: A Simple and Effective Baseline for Text-to-Image Synthesis

**Authors**: *Ming Tao, Hao Tang, Fei Wu, Xiaoyuan Jing, Bing-Kun Bao, Changsheng Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01602](https://doi.org/10.1109/CVPR52688.2022.01602)

**Abstract**:

Synthesizing high-quality realistic images from text descriptions is a challenging task. Existing text-to-image Generative Adversarial Networks generally employ a stacked architecture as the backbone yet still remain three flaws. First, the stacked architecture introduces the entanglements between generators of different image scales. Second, existing studies prefer to apply and fix extra networks in adversarial learning for text-image semantic consistency, which limits the supervision capability of these networks. Third, the cross-modal attention-based text-image fusion that widely adopted by previous works is limited on several special image scales because of the computational cost. To these ends, we propose a simpler but more effective Deep Fusion Generative Adversarial Networks (DF-GAN). To be specific, we propose: (i) a novel one-stage text-to-image backbone that directly synthesizes high-resolution images without entanglements between different generators, (ii) a novel Target-Aware Discriminator composed of Matching-Aware Gradient Penalty and One-Way Output, which enhances the text-image semantic consistency without introducing extra networks, (iii) a novel deep text-image fusion block, which deepens the fusion process to make a full fusion between text and visual features. Compared with current state-of-the-art methods, our proposed DF-GAN is simpler but more efficient to synthesize realistic and text-matching images and achieves better performance on widely used datasets. Code is available at https://github.com/tobran/DF-GAN.

----

## [1590] L-Verse: Bidirectional Generation Between Image and Text

**Authors**: *Taehoon Kim, Gwangmo Song, Sihaeng Lee, Sangyun Kim, Yewon Seo, Soonyoung Lee, Seung Hwan Kim, Honglak Lee, Kyunghoon Bae*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01603](https://doi.org/10.1109/CVPR52688.2022.01603)

**Abstract**:

Far beyond learning long-range interactions of natural language, transformers are becoming the de-facto standard for many vision tasks with their power and scalability. Especially with cross-modal tasks between image and text, vector quantized variational autoencoders (VQ-VAEs) are widely used to make a raw RGB image into a sequence of feature vectors. To better leverage the correlation between image and text, we propose L-Verse, a novel architecture consisting of feature-augmented variational autoencoder (AugVAE) and bidirectional auto-regressive transformer (BiART) for image-to-text and text-to-image generation. Our AugVAE shows the state-of-the-art reconstruction performance on ImageNetlK validation set, along with the robustness to unseen images in the wild. Unlike other models, BiART can distinguish between image (or text) as a conditional reference and a generation target. L-Verse can be directly used for image-to-text or text-to-image generation without any finetuning or extra object detection framework. In quantitative and qualitative experiments, L-Verse shows impressive results against previous methods in both image-to-text and text-to-image generation on MS-COCO Captions. We furthermore assess the scalability of L-Verse architecture on Conceptual Captions and present the initial result of bidirectional vision-language representation learning on general domain.

----

## [1591] Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation

**Authors**: *Shizhe Chen, Pierre-Louis Guhur, Makarand Tapaswi, Cordelia Schmid, Ivan Laptev*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01604](https://doi.org/10.1109/CVPR52688.2022.01604)

**Abstract**:

Following language instructions to navigate in unseenaenvironments is a challenging problem for autonomous embodied agents. The agent not only needs to ground languages in visual scenes, but also should explore the environment to reach its target. In this work, we propose a dual-scale graph transformer (DUET) for joint long-term action planning and fine-grained cross-modal understanding. We build a topological map on-the-fly to enable efficient exploration in global action space. To balance the complexity of large action space reasoning and fine-grained language grounding, we dynamically combine a fine-scale encoding over local observations and a coarse-scale encoding on a global map via graph transformers. The proposed approach, DUET, significantly outperforms state-of-the-art methods on goal-oriented vision-and-language navigation (VLN) benchmarks REVERIE and SOON. It also improves the success rate on the fine-grained VLN benchmark R2R.

----

## [1592] LaTr: Layout-Aware Transformer for Scene-Text VQA

**Authors**: *Ali Furkan Biten, Ron Litman, Yusheng Xie, Srikar Appalaraju, R. Manmatha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01605](https://doi.org/10.1109/CVPR52688.2022.01605)

**Abstract**:

We propose a novel multimodal architecture for Scene Text Visual Question Answering (STVQA), named Layout-Aware Transformer (LaTr). The task of STVQA requires models to reason over different modalities. Thus, we first investigate the impact of each modality, and reveal the importance of the language module, especially when enriched with layout information. Accounting for this, we propose a single objective pre-training scheme that requires only text and spatial cues. We show that applying this pre-training scheme on scanned documents has certain advantages over using natural images, despite the domain gap. Scanned documents are easy to procure, text-dense and have a variety of layouts, helping the model learn various spatial cues (e.g. left-of, below etc.) by tying together language and layout information. Compared to existing approaches, our method performs vocabulary-free decoding and, as shown, generalizes well beyond the training vocabulary. We further demonstrate that LaTr improves robustness towards OCR errors, a common reason for failure cases in STVQA. In addition, by leveraging a vision transformer, we eliminate the need for an external object detector. LaTr outperforms state-of-the-art STVQA methods on multiple datasets. In particular, +7.6% on TextVQA, +10.8% on ST-VQA and +4.0% on OCR-VQA (all absolute accuracy numbers).

----

## [1593] Learning Program Representations for Food Images and Cooking Recipes

**Authors**: *Dim P. Papadopoulos, Enrique Mora, Nadiia Chepurko, Kuan Wei Huang, Ferda Ofli, Antonio Torralba*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01606](https://doi.org/10.1109/CVPR52688.2022.01606)

**Abstract**:

In this paper, we are interested in modeling a how-to instructional procedure, such as a cooking recipe, with a meaningful and rich high-level representation. Specifically, we propose to represent cooking recipes and food images as cooking programs. Programs provide a structured repre-sentation of the task, capturing cooking semantics and se-quential relationships of actions in the form of a graph. This allows them to be easily manipulated by users and executed by agents. To this end, we build a model that is trained to learn a joint embedding between recipes and food images via self-supervision and jointly generate a program from this embedding as a sequence. To validate our idea, we crowdsource programs for cooking recipes and show that: (a) projecting the image-recipe embeddings into programs leads to better cross-modal retrieval results; (b) generating programs from images leads to better recognition re-sults compared to predicting raw cooking instructions; and (c) we can generate food images by manipulating programs via optimizing the latent code of a GAN. Code, data, and models are available online
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
http://cookingprograms.csail.mit.edu.

----

## [1594] On the Importance of Asymmetry for Siamese Representation Learning

**Authors**: *Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, Xinlei Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01607](https://doi.org/10.1109/CVPR52688.2022.01607)

**Abstract**:

Many recent self-supervised frameworks for visual representation learning are based on certain forms of Siamese networks. Such networks are conceptually symmetric with two parallel encoders, but often practically asymmetric as numerous mechanisms are devised to break the symmetry. In this work, we conduct a formal study on the importance of asymmetry by explicitly distinguishing the two encoders within the network - one produces source encodings and the other targets. Our key insight is keeping a relatively lower variance in target than source generally benefits learning. This is empirically justified by our results from five case studies covering different variance-oriented designs, and is aligned with our preliminary theoretical analysis on the baseline. Moreover, we find the improvements from asymmetric designs generalize well to longer training schedules, multiple other frameworks and newer backbones. Finally, the combined effect of several asymmetric designs achieves a state-of-the-art accuracy on ImageNet linear probing and competitive results on downstream transfer. We hope our exploration will inspire more research in exploiting asymmetry for Siamese representation learning.

----

## [1595] Leverage Your Local and Global Representations: A New Self-Supervised Learning Strategy

**Authors**: *Tong Zhang, Congpei Qiu, Wei Ke, Sabine Süsstrunk, Mathieu Salzmann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01608](https://doi.org/10.1109/CVPR52688.2022.01608)

**Abstract**:

Self-supervised learning (SSL) methods aim to learn view-invariant representations by maximizing the similar-ity between the features extracted from different crops of the same image regardless of cropping size and content. In essence, this strategy ignores the fact that two crops may truly contain different image information, e.g., background and small objects, and thus tends to restrain the diversity of the learned representations. In this work, we address this issue by introducing a new self-supervised learning strat-egy, LoGo, that explicitly reasons about Local and Global crops. To achieve view invariance, LoGo encourages similarity between global crops from the same image, as well as between a global and a local crop. However, to correctly encode the fact that the content of smaller crops may differ entirely, LoGo promotes two local crops to have dissimi-lar representations, while being close to global crops. Our LoGo strategy can easily be applied to existing SSL meth-ods. Our extensive experiments on a variety of datasets and using different self-supervised learning frameworks vali-date its superiority over existing approaches. Noticeably, we achieve better results than supervised models on trans-fer learning when using only 1/10 of the data. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code and pretrained models can be found at https://github.com/ztt1024/LoGo-SSL.

----

## [1596] Exploring Set Similarity for Dense Self-supervised Representation Learning

**Authors**: *Zhaoqing Wang, Qiang Li, Guoxin Zhang, Pengfei Wan, Wen Zheng, Nannan Wang, Mingming Gong, Tongliang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01609](https://doi.org/10.1109/CVPR52688.2022.01609)

**Abstract**:

By considering the spatial correspondence, dense self-supervised representation learning has achieved superior performance on various dense prediction tasks. However, the pixel-level correspondence tends to be noisy because of many similar misleading pixels, e.g., backgrounds. To address this issue, in this paper, we propose to explore set similarity (SetSim) for dense self-supervised representation learning. We generalize pixel-wise similarity learning to set-wise one to improve the robustness because sets contain more semantic and structure information. Specifically, by resorting to attentional features of views, we establish the corresponding set, thus filtering out noisy backgrounds that may cause incorrect correspondences. Meanwhile, these at-tentional features can keep the coherence of the same image across different views to alleviate semantic inconsistency. We further search the cross-view nearest neighbours of sets and employ the structured neighbourhood information to enhance the robustness. Empirical evaluations demonstrate that SetSim surpasses or is on par with state-of-the-art meth-ods on object detection, keypoint detection, instance segmen-tation, and semantic segmentation.

----

## [1597] Align Representations with Base: A New Approach to Self-Supervised Learning

**Authors**: *Shaofeng Zhang, Lyn Qiu, Feng Zhu, Junchi Yan, Hengrui Zhang, Rui Zhao, Hongyang Li, Xiaokang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01610](https://doi.org/10.1109/CVPR52688.2022.01610)

**Abstract**:

Existing symmetric contrastive learning methods suffer from collapses (complete and dimensional) or quadratic complexity of objectives. Departure from these methods which maximize mutual information of two generated views, along either instance or feature dimension, the proposed paradigm introduces intermediate variables at the feature level, and maximizes the consistency between variables and representations of each view. Specifically, the proposed intermediate variables are the nearest group of base vectors to representations. Hence, we call the proposed method ARB (Align Representations with Base). Compared with other symmetric approaches, ARB 1) does not require negative pairs, which leads the complexity of the overall objective function is in linear order, 2) reduces feature redundancy, increasing the information density of training samples, 3) is more robust to output dimension size, which out-performs previous feature-wise arts over 28% Top-1 accuracy on ImageNet-100under low-dimension settings.

----

## [1598] Identifying Ambiguous Similarity Conditions via Semantic Matching

**Authors**: *Han-Jia Ye, Yi Shi, De-Chuan Zhan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01611](https://doi.org/10.1109/CVPR52688.2022.01611)

**Abstract**:

Rich semantics inside an image result in its ambiguous relationship with others, i.e., two images could be similar in one condition but dissimilar in another. Given triplets like “aircraft” is similar to “bird” than “train”, Weakly Supervised Conditional Similarity Learning (WS-CSL) learns multiple embeddings to match semantic conditions without explicit condition labels such as “can fly”. However, similarity relationships in a triplet are uncertain except providing a condition. For example, the previous comparison becomes invalid once the conditional label changes to “is vehicle”. To this end, we introduce a novel evaluation criterion by predicting the comparison's correctness after assigning the learned embeddings to their optimal conditions, which measures how much WS-CSL could cover latent semantics as the supervised model. Furthermore, we propose the Distance Induced Semantic COndition VER-ification Network (DiscoverNet), which characterizes the instance-instance and triplets-condition relations in a “decompose-and-fuse” manner. To make the learned embeddings cover all semantics, Discovernet utilizes a set module or an additional regularizer over the correspondence between a triplet and a condition. DiscoverNet achieves state-of-the-art performance on benchmarks like UT-Zappos-50k and Celeb-A w.r.t. different criteria.

----

## [1599] Node Representation Learning in Graph via Node-to-Neighbourhood Mutual Information Maximization

**Authors**: *Wei Dong, Junsheng Wu, Yi Luo, Zongyuan Ge, Peng Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01612](https://doi.org/10.1109/CVPR52688.2022.01612)

**Abstract**:

The key towards learning informative node representations in graphs lies in how to gain contextual information from the neighbourhood. In this work, we present a simple-yet-effective self-supervised node representation learning strategy via directly maximizing the mutual information between the hidden representations of nodes and their neighbourhood, which can be theoretically justified by its link to graph smoothing. Following InfoNCE, our framework is optimized via a surrogate contrastive loss, where the positive selection underpins the quality and efficiency of rep-resentation learning. To this end, we propose a topology-aware positive sampling strategy, which samples positives from the neighbourhood by considering the structural dependencies between nodes and thus enables positive selection upfront. In the extreme case when only one positive is sampled, we fully avoid expensive neighbourhood aggregation. Our methods achieve promising performance on various node classification datasets. It is also worth mentioning by applying our loss function to MLP based node encoders, our methods can be orders of faster than existing solutions. Our codes and supplementary materials are available at https://github.com/dongwei156/n2n.

----



[Go to the previous page](CVPR-2022-list07.md)

[Go to the next page](CVPR-2022-list09.md)

[Go to the catalog section](README.md)