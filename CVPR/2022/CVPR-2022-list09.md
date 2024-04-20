## [1600] Instance-Dependent Label-Noise Learning with Manifold-Regularized Transition Matrix Estimation

**Authors**: *De Cheng, Tongliang Liu, Yixiong Ning, Nannan Wang, Bo Han, Gang Niu, Xinbo Gao, Masashi Sugiyama*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01613](https://doi.org/10.1109/CVPR52688.2022.01613)

**Abstract**:

In label-noise learning, estimating the transition matrix has attracted more and more attention as the matrix plays an important role in building statistically consistent classifiers. However, it is very challenging to estimate the transition matrix T(x), where x denotes the instance, because it is unidentifiable under the instance-dependent noise (IDN). To address this problem, we have noticed that, there are psychological and physiological evidences showing that we humans are more likely to annotate instances of similar appearances to the same classes, and thus poor-quality or ambiguous instances of similar appearances are easier to be mislabeled to the correlated or same noisy classes. Therefore, we propose assumption on the geometry of T(x) that “the closer two instances are, the more similar their corresponding transition matrices should be”. More specifically, we formulate above assumption into the manifold embedding, to effectively reduce the degree of freedom of T(x) and make it stably estimable in practice. The proposed manifold-regularized technique works by directly reducing the estimation error without hurting the approximation error about the estimation problem of T(x). Experimental evaluations on four synthetic and two real-world datasets demonstrate that our method is superior to state-of-the-art approaches for label-noise learning under the challenging IDN.

----

## [1601] Unsupervised Visual Representation Learning by Online Constrained K-Means

**Authors**: *Qi Qian, Yuanhong Xu, Juhua Hu, Hao Li, Rong Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01614](https://doi.org/10.1109/CVPR52688.2022.01614)

**Abstract**:

Cluster discrimination is an effective pretext task for unsupervised representation learning, which often consists of two phases: clustering and discrimination. Clustering is to assign each instance a pseudo label that will be used to learn representations in discrimination. The main challenge resides in clustering since prevalent clustering methods (e.g., k-means) have to run in a batch mode. Besides, there can be a trivial solution consisting of a dominating cluster. To address these challenges, we first investigate the objective of clustering-based representation learning. Based on this, we propose a novel clustering-based pretext task with online Constrained K-means (CoKe). Compared with the balanced clustering that each cluster has exactly the same size, we only constrain the minimal size of each cluster to flexibly capture the inherent data structure. More importantly, our online assignment method has a theoretical guarantee to approach the global optimum. By decoupling clustering and discrimination, CoKe can achieve competitive performance when optimizing with only a single view from each instance. Extensive experiments on ImageNet and other benchmark data sets verify both the efficacy and efficiency of our proposal.

----

## [1602] Rethinking the Augmentation Module in Contrastive Learning: Learning Hierarchical Augmentation Invariance with Expanded Views

**Authors**: *Junbo Zhang, Kaisheng Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01615](https://doi.org/10.1109/CVPR52688.2022.01615)

**Abstract**:

A data augmentation module is utilized in contrastive learning to transform the given data example into two views, which is considered essential and irreplaceable. However, the pre-determined composition of multiple data augmentations brings two drawbacks. First, the artificial choice of augmentation types brings specific representational invariances to the model, which have different de-grees of positive and negative effects on different down-stream tasks. Treating each type of augmentation equally during training makes the model learn non-optimal repre-sentations for various downstream tasks and limits the flex-ibility to choose augmentation types beforehand. Second, the strong data augmentations used in classic contrastive learning methods may bring too much invariance in some cases, and fine- grained information that is essential to some downstream tasks may be lost. This paper proposes a gen-eral method to alleviate these two problems by considering “where” and “what” to contrast in a general contrastive learning framework. We first propose to learn different aug-mentation invariances at different depths of the model ac-cording to the importance of each data augmentation in-stead of learning representational invariances evenly in the backbone. We then propose to expand the contrast content with augmentation embeddings to reduce the misleading ef-fects of strong data augmentations. Experiments based on several baseline methods demonstrate that we learn better representations for various benchmarks on classification, detection, and segmentation downstream tasks.

----

## [1603] Use All The Labels: A Hierarchical Multi-Label Contrastive Learning Framework

**Authors**: *Shu Zhang, Ran Xu, Caiming Xiong, Chetan Ramaiah*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01616](https://doi.org/10.1109/CVPR52688.2022.01616)

**Abstract**:

Current contrastive learning frameworks focus on leveraging a single supervisory signal to learn representations, which limits the efficacy on unseen data and downstream tasks. In this paper, we present a hierarchical multi-label representation learning framework that can leverage all available labels and preserve the hierarchical relationship between classes. We introduce novel hierarchy preserving losses, which jointly apply a hierarchical penalty to the contrastive loss, and enforce the hierarchy constraint. The loss function is data driven and automatically adapts to arbitrary multi-label structures. Experiments on several datasets show that our relationship-preserving embedding performs well on a variety of tasks and outperform the base-line supervised and self-supervised approaches. Code is available at https://github.com/salesforce/hierarchicalContrastiveLearning.

----

## [1604] Robust Contrastive Learning against Noisy Views

**Authors**: *Ching-Yao Chuang, R. Devon Hjelm, Xin Wang, Vibhav Vineet, Neel Joshi, Antonio Torralba, Stefanie Jegelka, Yale Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01617](https://doi.org/10.1109/CVPR52688.2022.01617)

**Abstract**:

Contrastive learning relies on an assumption that positive pairs contain related views that share certain underlying information about an instance, e.g., patches of an image or co-occurring multimodal signals of a video. What if this assumption is violated? The literature suggests that contrastive learning produces suboptimal representations in the presence of noisy views, e.g., false positive pairs with no apparent shared information. In this work, we pro-pose a new contrastive loss function that is robust against noisy views. We provide rigorous theoretical justifications by showing connections to robust symmetric losses for noisy binary classification and by establishing a new contrastive bound for mutual information maximization based on the Wasserstein distance measure. The proposed loss is completely modality-agnostic and a simple drop-in replacement for the InfoNCE loss, which makes it easy to apply to ex-isting contrastive frameworks. We show that our approach provides consistent improvements over the state-of-the-art on image, video, and graph contrastive learning bench-marks that exhibit a variety of real-world noise patterns.

----

## [1605] On Learning Contrastive Representations for Learning with Noisy Labels

**Authors**: *Li Yi, Sheng Liu, Qi She, A. Ian McLeod, Boyu Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01618](https://doi.org/10.1109/CVPR52688.2022.01618)

**Abstract**:

Deep neural networks are able to memorize noisy labels easily with a softmax cross entropy (CE) loss. Previous studies attempted to address this issue focus on incorporating a noise-robust loss function to the CE loss. However, the memorization issue is alleviated but still remains due to the non-robust CE loss. To address this issue, we focus on learning robust contrastive representations of data on which the classifier is hard to memorize the label noise under the CE loss. We propose a novel contrastive regularization function to learn such representations over noisy data where label noise does not dominate the representation learning. By theoretically investigating the representations induced by the proposed regularization function, we reveal that the learned representations keep information related to true labels and discard information related to corrupted labels. Moreover, our theoretical results also indicate that the learned representations are robust to the label noise. The effectiveness of this method is demonstrated with experiments on benchmark datasets.

----

## [1606] Directional Self-supervised Learning for Heavy Image Augmentations

**Authors**: *Yalong Bai, Yifan Yang, Wei Zhang, Tao Mei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01619](https://doi.org/10.1109/CVPR52688.2022.01619)

**Abstract**:

Despite the large augmentation family, only a few cherry-picked robust augmentation policies are beneficial to self-supervised image representation learning. In this paper, we propose a directional self-supervised learning paradigm (DSSL), which is compatible with significantly more augmentations. Specifically, we adapt heavy augmentation policies after the views lightly augmented by standard augmentations, to generate harder view (HV). HV usually has a higher deviation from the original image than the lightly augmented standard view (SV). Unlike previous methods equally pairing all augmented views to symmetrically maximize their similarities, DSSL treats augmented views of the same instance as a partially ordered set (with directions as SV↔SV, SV↔HV), and then equips a directional objective function respecting to the derived relationships among views. DSSL can be easily implemented with a few lines of codes and is highly flexible to popular self-supervised learning frameworks, including SimCLR, Sim-Siam, BYOL. Extensive experimental results on CIFAR and ImageNet demonstrated that DSSL can stably improve various baselines with compatibility to a wider range of augmentations. Code is available at: https://github.com/Yif-Yang/DSSL.

----

## [1607] Continual Learning for Visual Search with Backward Consistent Feature Embedding

**Authors**: *Timmy S. T. Wan, Jun-Cheng Chen, Tzer-Yi Wu, Chu-Song Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01620](https://doi.org/10.1109/CVPR52688.2022.01620)

**Abstract**:

In visual search, the gallery set could be incrementally growing and added to the database in practice. However, existing methods rely on the model trained on the entire dataset, ignoring the continual updating of the model. Besides, as the model updates, the new model must re-extract features for the entire gallery set to maintain compatible feature space, imposing a high computational cost for a large gallery set. To address the issues of long-term visual search, we introduce a continual learning (CL) approach that can handle the incrementally growing gallery set with backward embedding consistency. We enforce the losses of inter-session data coherence, neighbor-session model coherence, and intra-session discrimination to conduct a continual learner. In addition to the disjoint setup, our CL solution also tackles the situation of increasingly adding new classes for the blurry boundary without assuming all categories known in the beginning and during model update. To our knowledge, this is the first CL method both tackling the issue of backward-consistent feature embedding and allowing novel classes to occur in the new sessions. Extensive experiments on various benchmarks show the efficacy of our approach under a wide range of setups 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/ivclab/CVS.

----

## [1608] Probing Representation Forgetting in Supervised and Unsupervised Continual Learning

**Authors**: *MohammadReza Davari, Nader Asadi, Sudhir Mudur, Rahaf Aljundi, Eugene Belilovsky*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01621](https://doi.org/10.1109/CVPR52688.2022.01621)

**Abstract**:

Continual Learning (CL) research typically focuses on tackling the phenomenon of catastrophic forgetting in neural networks. Catastrophic forgetting is associated with an abrupt loss of knowledge previously learned by a model when the task, or more broadly the data distribution, being trained on changes. In supervised learning problems this forgetting, resulting from a change in the model's representation, is typically measured or observed by evaluating the decrease in old task performance. However, a model's representation can change without losing knowledge about prior tasks. In this work we consider the concept of representation forgetting, observed by using the difference in performance of an optimal linear classifier before and after a new task is introduced. Using this tool we revisit a number of standard continual learning benchmarks and observe that, through this lens, model representations trained without any explicit control for forgetting often experience small representation forgetting and can sometimes be comparable to methods which explicitly control for forgetting, especially in longer task sequences. We also show that representation forgetting can lead to new insights on the effect of model capacity and loss function used in continual learning. Based on our results, we show that a simple yet competitive approach is to learn representations continually with standard supervised contrastive learning while constructing prototypes of class samples when queried on old samples.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code to reproduce our results is publicly available at: https://github.com/rezazzr/Probing-Representation-Forgetting

----

## [1609] Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning

**Authors**: *Yujun Shi, Kuangqi Zhou, Jian Liang, Zihang Jiang, Jiashi Feng, Philip H. S. Torr, Song Bai, Vincent Y. F. Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01622](https://doi.org/10.1109/CVPR52688.2022.01622)

**Abstract**:

Class Incremental Learning (CIL) aims at learning a classifier in a phase-by-phase manner, in which only data of a subset of the classes are provided at each phase. Previous works mainly focus on mitigating forgetting in phases after the initial one. However, we find that improving CIL at its initial phase is also a promising direction. Specifically, we experimentally show that directly encouraging CIL Learner at the initial phase to output similar representations as the model jointly trained on all classes can greatly boost the CIL performance. Motivated by this, we study the differ-ence between a naively-trained initial-phase model and the oracle model. Specifically, since one major difference be-tween these two models is the number of training classes, we investigate how such difference affects the model rep-resentations. We find that, with fewer training classes, the data representations of each class lie in a long and narrow region; with more training classes, the representations of each class scatter more uniformly. Inspired by this obser-vation, we propose Class-wise Decorrelation (CwD) that ef-fectively regularizes representations of each class to scatter more uniformly, thus mimicking the model jointly trained with all classes (i.e., the oracle model). Our CwD is simple to implement and easy to plug into existing methods. Ex-tensive experiments on various benchmark datasets show that CwD consistently and significantly improves the per-formance of existing state-of-the-art methods by around 1% to 3%. Code: https://github.com/Yujun-Shi/CwD.

----

## [1610] Bring Evanescent Representations to Life in Lifelong Class Incremental Learning

**Authors**: *Marco Toldo, Mete Ozay*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01623](https://doi.org/10.1109/CVPR52688.2022.01623)

**Abstract**:

In Class Incremental Learning (CIL), a classification model is progressively trained at each incremental step on an evolving dataset of new classes, while at the same time, it is required to preserve knowledge of all the classes ob-served so far. Prototypical representations can be lever-aged to model feature distribution for the past data and in-ject information of former classes in later incremental steps without resorting to stored exemplars. However, if not up-dated, those representations become increasingly outdated as the incremental learning progresses with new classes. To address the aforementioned problems, we propose a frame-work which aims to (i) model the semantic drift by learning the relationship between representations of past and novel classes among incremental steps, and (ii) estimate the feature drift, defined as the evolution of the represen-tations learned by models at each incremental step. Se-mantic and feature drifts are then jointly exploited to infer up-to-date representations of past classes (evanescent rep-resentations), and thereby infuse past knowledge into incre-mental training. We experimentally evaluate our framework achieving exemplar-free SotA results on multiple bench-marks. In the ablation study, we investigate nontrivial relationships between evanescent representations and models.

----

## [1611] Unsupervised Learning of Debiased Representations with Pseudo-Attributes

**Authors**: *Seonguk Seo, Joon-Young Lee, Bohyung Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01624](https://doi.org/10.1109/CVPR52688.2022.01624)

**Abstract**:

Dataset bias is a critical challenge in machine learning since it often leads to a negative impact on a model due to the unintended decision rules captured by spurious correlations. Although existing works often handle this issue based on human supervision, the availability of the proper annotations is impractical and even unrealistic. To better tackle the limitation, we propose a simple but effective unsupervised debiasing technique. Specifically, we first identify pseudo-attributes based on the results from clustering performed in the feature embedding space even without an explicit bias attribute supervision. Then, we employ a novel cluster-wise reweighting scheme to learn debiased representation; the proposed method prevents minority groups from being discounted for minimizing the overall loss, which is desirable for worst-case generalization. The extensive experiments demonstrate the outstanding performance of our approach on multiple standard benchmarks, even achieving the competitive accuracy to the supervised counterpart. The source code is available at our project page
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/skynbe/pseudo-attributes .

----

## [1612] A Conservative Approach for Unbiased Learning on Unknown Biases

**Authors**: *Myeongho Jeon, Daekyung Kim, Woochul Lee, Myungjoo Kang, Joonseok Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01625](https://doi.org/10.1109/CVPR52688.2022.01625)

**Abstract**:

Although convolutional neural networks (CNNs) achieve state-of-the-art in image classification, recent works address their unreliable predictions due to their excessive dependence on biased training data. Existing unbiased modeling postulates that the bias in the dataset is obvious to know, but it is actually unsuited for image datasets including countless sensory attributes. To mitigate this issue, we present a new scenario that does not necessitate a predefined bias. Under the observation that CNNs do have multi-variant and unbiased representations in the model, we propose a conservative framework that employs this internal information for unbiased learning. Specifically, this mechanism is implemented via hierarchical features captured along the multiple layers and orthogonal regularization. Extensive evaluations on public benchmarks demonstrate our method is effective for unbiased learning. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Source code: https://github.com/aandyjeon/UBNet

----

## [1613] Evading the Simplicity Bias: Training a Diverse Set of Models Discovers Solutions with Superior OOD Generalization

**Authors**: *Damien Teney, Ehsan Abbasnejad, Simon Lucey, Anton van den Hengel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01626](https://doi.org/10.1109/CVPR52688.2022.01626)

**Abstract**:

Neural networks trained with SGD were recently shown to rely preferentially on linearly-predictive features and can ignore complex, equally-predictive ones. This simplicity bias can explain their lack of robustness out of distribution (OOD). The more complex the task to learn, the more likely it is that statistical artifacts (i.e. selection biases, spurious correlations) are simpler than the mechanisms to learn. We demonstrate that the simplicity bias can be mitigated and OOD generalization improved. We train a set of similar models to fit the data in different ways using a penalty on the alignment of their input gradients. We show theoretically and empirically that this induces the learning of more com-plex predictive patterns. OOD generalization fundamentally requires information beyond i. i.d. examples, such as multiple training environ-ments, counterfactual examples, or other side information. Our approach shows that we can defer this requirement to an independent model selection stage. We obtain SOTA re-sults in visual recognition on biased data and generalization across visual domains. The method - the first to evade the simplicity bias - highlights the need for a better under-standing and control of inductive biases in deep learning.

----

## [1614] Co-advise: Cross Inductive Bias Distillation

**Authors**: *Sucheng Ren, Zhengqi Gao, Tianyu Hua, Zihui Xue, Yonglong Tian, Shengfeng He, Hang Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01627](https://doi.org/10.1109/CVPR52688.2022.01627)

**Abstract**:

The inductive bias of vision transformers is more relaxed that cannot work well with insufficient data. Knowledge distillation is thus introduced to assist the training of transformers. Unlike previous works, where merely heavy convolution-based teachers are provided, in this paper, we delve into the influence of models inductive biases in knowledge distillation (e.g., convolution and involution). Our key observation is that the teacher accuracy is not the dominant reason for the student accuracy, but the teacher inductive bias is more important. We demonstrate that lightweight teachers with different architectural inductive biases can be used to co-advise the student transformer with outstanding performances. The rationale behind is that models designed with different inductive biases tend to focus on diverse patterns, and teachers with different inductive biases attain various knowledge despite being trained on the same dataset. The diverse knowledge provides a more precise and comprehensive description of the data and compounds and boosts the performance of the student during distillation. Furthermore, we propose a token inductive bias alignment to align the inductive bias of the token with its target teacher model. With only lightweight teachers provided and using this cross inductive bias distillation method, our vision transformers (termed as CiT) outperform all previous vision transformers (ViT) of the same architecture on ImageNet. Moreover, our small size model CiT-SAK further achieves 82.7% Top-1 accuracy on ImageNet without modifying the attention module of the ViT. Code is available at https://github.com/OliverRensu/co-advise.

----

## [1615] PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures

**Authors**: *Dan Hendrycks, Andy Zou, Mantas Mazeika, Leonard Tang, Bo Li, Dawn Song, Jacob Steinhardt*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01628](https://doi.org/10.1109/CVPR52688.2022.01628)

**Abstract**:

In real-world applications of machine learning, reliable and safe systems must consider measures of performance beyond standard test set accuracy. These other goals include out-of-distribution (OOD) robustness, prediction consistency, resilience to adversaries, calibrated uncertainty estimates, and the ability to detect anomalous inputs. However, improving performance towards these goals is often a balancing act that today's methods cannot achieve without sacrificing performance on other safety axes. For instance, adversarial training improves adversarial robustness but sharply degrades other classifier performance metrics. Similarly, strong data augmentation and regularization techniques often improve OOD robustness but harm anomaly detection, raising the question of whether a Pareto improvement on all existing safety measures is possible. To meet this challenge, we design a new data augmentation strategy utilizing the natural structural complexity of pictures such as fractals, which outperforms numerous baselines, is near Pareto-optimal, and roundly improves safety measures.

----

## [1616] RegionCLIP: Region-based Language-Image Pretraining

**Authors**: *Yiwu Zhong, Jianwei Yang, Pengchuan Zhang, Chunyuan Li, Noel Codella, Liunian Harold Li, Luowei Zhou, Xiyang Dai, Lu Yuan, Yin Li, Jianfeng Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01629](https://doi.org/10.1109/CVPR52688.2022.01629)

**Abstract**:

Contrastive language-image pretraining (CLIP) using image-text pairs has achieved impressive results on image classification in both zero-shot and transfer learning set-tings. However, we show that directly applying such mod-els to recognize image regions for object detection leads to unsatisfactory performance due to a major domain shift: CLIP was trained to match an image as a whole to a text de-scription, without capturing the fine-grained alignment be-tween image regions and text spans. To mitigate this issue, we propose a new method called RegionCLIP that signifi-cantly extends CLIP to learn region-level visual representations, thus enabling fine-grained alignment between image regions and textual concepts. Our method leverages a CLIP model to match image regions with template captions, and then pretrains our model to align these region-text pairs in the feature space. When transferring our pretrained model to the open-vocabulary object detection task, our method outperforms the state of the art by 3.8 AP50 and 2.2 AP for novel categories on COCO and LVIS datasets, respectively. Further, the learned region representations support zero-shot inference for object detection, showing promising results on both COCO and LVIS datasets. Our code is available at https://github.com/microsoft/RegionCLIP.

----

## [1617] Uni-Perceiver: Pre-training Unified Architecture for Generic Perception for Zero-shot and Few-shot Tasks

**Authors**: *Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Hongsheng Li, Xiaohua Wang, Jifeng Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01630](https://doi.org/10.1109/CVPR52688.2022.01630)

**Abstract**:

Biological intelligence systems of animals perceive the world by integrating information in different modalities and processing simultaneously for various tasks. In contrast, current machine learning research follows a task-specific paradigm, leading to inefficient collaboration between tasks and high marginal costs of developing perception models for new tasks. In this paper, we present a generic perception architecture named Uni-Perceiver, which processes a variety of modalities and tasks with unified modeling and shared parameters. Specifically, Uni-Perceiver encodes different task inputs and targets from arbitrary modalities into a unified representation space with a modality-agnostic Transformer encoder and lightweight modality-specific tokenizers. Different perception tasks are modeled as the same formulation, that is, finding the maximum likelihood target for each input through the similarity of their representations. The model is pre-trained on several uni-modal and multi-modal tasks, and evaluated on a variety of downstream tasks, including novel tasks that did not appear in the pre-training stage. Results show that our pre-trained model without any tuning can achieve reasonable performance even on novel tasks. The performance can be improved to a level close to state-of-the-art methods by conducting prompt tuning on 1% of downstream task data. Full-data fine-tuning further delivers results on par with or better than state-of-the-art results. Code and pre-trained weights shall be released.

----

## [1618] Conditional Prompt Learning for Vision-Language Models

**Authors**: *Kaiyang Zhou, Jingkang Yang, Chen Change Loy, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01631](https://doi.org/10.1109/CVPR52688.2022.01631)

**Abstract**:

With the rise of powerful pre-trained vision-language models like CLIP, it becomes essential to investigate ways to adapt these models to downstream datasets. A recently proposed method named Context Optimization (CoOp) introduces the concept of prompt learning—a recent trend in NLP—to the vision domain for adapting pre-trained vision-language models. Specifically, CoOp turns context words in a prompt into a set of learnable vectors and, with only a few labeled images for learning, can achieve huge improvements over intensively-tuned manual prompts. In our study we identify a critical problem of CoOp: the learned context is not generalizable to wider unseen classes within the same dataset, suggesting that CoOp overfits base classes observed during training. To address the problem, we propose Conditional Context Optimization (CoCoOp), which extends CoOp by further learning a lightweight neural network to generate for each image an input-conditional token (vector). Compared to CoOp's static prompts, our dynamic prompts adapt to each instance and are thus less sensitive to class shift. Extensive experiments show that CoCoOp generalizes much better than CoOp to unseen classes, even showing promising transferability beyond a single dataset; and yields stronger domain generalization performance as well. Code is available at https://github.com/KaiyangZhou/CoOp.

----

## [1619] Noisy Boundaries: Lemon or Lemonade for Semi-supervised Instance Segmentation?

**Authors**: *Zhenyu Wang, Yali Li, Shengjin Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01632](https://doi.org/10.1109/CVPR52688.2022.01632)

**Abstract**:

Current instance segmentation methods rely heavily on pixel-level annotated images. The huge cost to obtain such fully-annotated images restricts the dataset scale and limits the performance. In this paper, we formally address semi-supervised instance segmentation, where unlabeled images are employed to boost the performance. We construct a framework for semi-supervised instance segmentation by assigning pixel-level pseudo labels. Under this framework, we point out that noisy boundaries associated with pseudo labels are double-edged. We propose to exploit and resist them in a unified manner simultaneously: 1) To combat the negative effects of noisy boundaries, we propose a noise-tolerant mask head by leveraging low-resolution features. 2) To enhance the positive impacts, we introduce a boundary-preserving map for learning detailed information within boundary-relevant regions. We evaluate our approach by extensive experiments. It behaves extraordinarily, outperforming the supervised baseline by a large margin, more than 6% on Cityscapes, 7% on COCO and 4.5% on BDD100k. On Cityscapes, our method achieves comparable performance by utilizing only 30% labeled images.

----

## [1620] Partial Class Activation Attention for Semantic Segmentation

**Authors**: *Sun'ao Liu, Hongtao Xie, Hai Xu, Yongdong Zhang, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01633](https://doi.org/10.1109/CVPR52688.2022.01633)

**Abstract**:

Current attention-based methods for semantic segmentation mainly model pixel relation through pairwise affinity and coarse segmentation. For the first time, this paper explores modeling pixel relation via Class Activation Map (CAM). Beyond the previous CAM generated from image-level classification, we present Partial CAM, which sub-divides the task into region-level prediction and achieves better localization performance. In order to eliminate the intra-class inconsistency caused by the variances of local context, we further propose Partial Class Activation Attention (PCAA) that simultaneously utilizes local and global class-level representations for attention calculation. Once obtained the partial CAM, PCAA collects local class centers and computes pixel-to-class relation locally. Applying local-specific representations ensures reliable results under different local contexts. To guarantee global consistency, we gather global representations from all local class centers and conduct feature aggregation. Experimental results confirm that Partial CAM outperforms the previous two strategies as pixel relation. Notably, our method achieves state-of-the-art performance on several challenging benchmarks including Cityscapes, Pascal Context, and ADE20K. Code is available at https://github.com/lsa1997/PCAA.

----

## [1621] Learning Affinity from Attention: End-to-End Weakly-Supervised Semantic Segmentation with Transformers

**Authors**: *Lixiang Ru, Yibing Zhan, Baosheng Yu, Bo Du*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01634](https://doi.org/10.1109/CVPR52688.2022.01634)

**Abstract**:

Weakly-supervised semantic segmentation (WSSS) with image-level labels is an important and challenging task. Due to the high training efficiency, end-to-end solutions for WSSS have received increasing attention from the community. However, current methods are mainly based on convolutional neural networks and fail to explore the global information properly, thus usually resulting in incomplete object regions. In this paper, to address the aforementioned problem, we introduce Transformers, which naturally integrate global information, to generate more integral initial pseudo labels for end-to-end WSSS. Motivated by the inherent consistency between the self-attention in Transformers and the semantic affinity, we propose an Affinity from Attention (AFA) module to learn semantic affinity from the multi-head self-attention (MHSA) in Transformers. The learned affinity is then leveraged to refine the initial pseudo labels for segmentation. In addition, to efficiently derive reliable affinity labels for supervising AFA and ensure the local consistency of pseudo labels, we devise a Pixel-Adaptive Refinement module that incorporates low-level image appearance information to refine the pseudo labels. We perform extensive experiments and our method achieves 66.0% and 38.9% mIoU on the PASCAL VOC 2012 and MS COCO 2014 datasets, respectively, significantly outperforming recent end-to-end methods and several multi-stage competitors. Code is available at https://github.com/rulixiang/afa.

----

## [1622] Towards Noiseless Object Contours for Weakly Supervised Semantic Segmentation

**Authors**: *Jing Li, Junsong Fan, Zhaoxiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01635](https://doi.org/10.1109/CVPR52688.2022.01635)

**Abstract**:

Image-level label based weakly supervised semantic segmentation has attracted much attention since image labels are very easy to obtain. Existing methods usually generate pseudo labels from class activation map (CAM) and then train a segmentation model. CAM usually highlights partial objects and produce incomplete pseudo labels. Some methods explore object contour by training a contour model with CAM seed label supervision and then propagate CAM score from discriminative regions to nondiscriminative regions with contour guidance. The propagation process suffers from the noisy intra-object contours, and inadequate propagation results produce incomplete pseudo labels. This is because the coarse CAM seed label lacks sufficient precise semantic information to suppress contour noise. In this paper, we train a SANCE model which utilizes an auxiliary segmentation module to supplement high-level semantic information for contour training by backbone feature sharing and online label supervision. The auxiliary segmentation module also provides more accurate localization map than CAM for pseudo label generation. We evaluate our approach on Pascal VOC 2012 and MS COCO 2014 benchmarks and achieve stateof- the-art performance, demonstrating the effectiveness of our method. The source code can be found at https://github.com/BraveGroup/SANCE

----

## [1623] Class Similarity Weighted Knowledge Distillation for Continual Semantic Segmentation

**Authors**: *Minh-Hieu Phan, The-Anh Ta, Son Lam Phung, Long Tran-Thanh, Abdesselam Bouzerdoum*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01636](https://doi.org/10.1109/CVPR52688.2022.01636)

**Abstract**:

Deep learning models are known to suffer from the problem of catastrophic forgetting when they incrementally learn new classes. Continual learning for semantic segmentation (CSS) is an emerging field in computer vision. We identify a problem in CSS: A model tends to be confused between old and new classes that are visually similar, which makes it forget the old ones. To address this gap, we propose REMINDER - a new CSS framework and a novel class similarity knowledge distillation (CSW-KD) method. Our CSW-KD method distills the knowledge of a previous model on old classes that are similar to the new one. This provides two main benefits: (i) selectively revising old classes that are more likely to be forgotten, and (ii) better learning new classes by relating them with the previously seen classes. Extensive experiments on Pascal-Voc 2012 and ADE20k datasets show that our approach outperforms state-of-the-art methods on standard CSS settings by up to 7.07% and 8.49%, respectively.

----

## [1624] Structural and Statistical Texture Knowledge Distillation for Semantic Segmentation

**Authors**: *Deyi Ji, Haoran Wang, Mingyuan Tao, Jianqiang Huang, Xian-Sheng Hua, Hongtao Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01637](https://doi.org/10.1109/CVPR52688.2022.01637)

**Abstract**:

Existing knowledge distillation works for semantic seg-mentation mainly focus on transfering high-level contextual knowledge from teacher to student. However, low-level texture knowledge is also of vital importance for characterizing the local structural pattern and global statistical prop-erty, such as boundary, smoothness, regularity and color contrast, which may not be well addressed by high-level deep features. In this paper, we are intended to take full advantage of both structural and statistical texture knowledge and propose a novel Structural and Statistical Texture Knowledge Distillation (SSTKD) framework for Semantic Segmentation. Specifically, for structural texture knowledge, we introduce a Contourlet Decomposition Module (CDM) that decomposes low-level features with iterative laplacian pyramid and directional filter bank to mine the structural texture knowledge. For statistical knowledge, we propose a Denoised Texture Intensity Equalization Module (DTIEM) to adaptively extract and enhance statistical texture knowledge through heuristics iterative quantization and denoised operation. Finally, each knowledge learning is supervised by an individual loss function, forcing the student network to mimic the teacher better from a broader perspective. Experiments show that the proposed method achieves state-of-the-art performance on Cityscapes, Pascal VOC 2012 and ADE20K datasets.

----

## [1625] L2G: A Simple Local-to-Global Knowledge Transfer Framework for Weakly Supervised Semantic Segmentation

**Authors**: *Peng-Tao Jiang, Yuqi Yang, Qibin Hou, Yunchao Wei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01638](https://doi.org/10.1109/CVPR52688.2022.01638)

**Abstract**:

Mining precise class-aware attention maps, a.k.a, class activation maps, is essential for weakly supervised semantic segmentation. In this paper, we present L2G, a simple online local-to-global knowledge transfer framework for high-quality object attention mining. We observe that classification models can discover object regions with more details when replacing the input image with its local patches. Taking this into account, we first leverage a local classification network to extract attentions from multiple local patches randomly cropped from the input image. Then, we utilize a global network to learn complementary attention knowledge across multiple local attention maps online. Our framework conducts the global network to learn the captured rich object detail knowledge from a global view and thereby produces high-quality attention maps that can be directly used as pseudo annotations for semantic segmentation networks. Experiments show that our method attains 72.1% and 44.2% mIoU scores on the validation set of PASCAL VOC 2012 and MS COCO 2014, respectively, setting new state-of-the-art records. Code is available at https://github.com/PengtaoJiang/L2G.

----

## [1626] Weakly Supervised Semantic Segmentation using Out-of-Distribution Data

**Authors**: *Jungbeom Lee, Seong Joon Oh, Sangdoo Yun, Junsuk Choe, Eunji Kim, Sungroh Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01639](https://doi.org/10.1109/CVPR52688.2022.01639)

**Abstract**:

Weakly supervised semantic segmentation (WSSS) methods are often built on pixel-level localization maps obtained from a classifier. However, training on class labels only, classifiers suffer from the spurious correlation between fore-ground and background cues (e.g. train and rail), fundamentally bounding the performance of WSSS. There have been previous endeavors to address this issue with additional supervision. We propose a novel source of information to distinguish foreground from the background: Out-of-Distribution (OoD) data, or images devoid of foreground object classes. In particular, we utilize the hard OoDs that the classifier is likely to make false-positive predictions. These samples typically carry key visual features on the background (e.g. rail) that the classifiers often confuse as foreground (e.g. train), so these cues let classifiers correctly suppress spurious background cues. Acquiring such hard OoDs does not require an extensive amount of annotation efforts; it only incurs a few additional image-level labeling costs on top of the original efforts to collect class labels. We propose a method, W-OoD, for utilizing the hard OoDs. W-OoD achieves state-of-the-art performance on Pascal VOC 2012. The code is available at: https://github.com/naver-ai/w-ood.

----

## [1627] Tree Energy Loss: Towards Sparsely Annotated Semantic Segmentation

**Authors**: *Zhiyuan Liang, Tiancai Wang, Xiangyu Zhang, Jian Sun, Jianbing Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01640](https://doi.org/10.1109/CVPR52688.2022.01640)

**Abstract**:

Sparsely annotated semantic segmentation (SASS) aims to train a segmentation network with coarse-grained (i.e., point-, scribble-, and block-wise) supervisions, where only a small proportion of pixels are labeled in each image. In this paper, we propose a novel tree energy loss for SASS by providing semantic guidance for unlabeled pixels. The tree energy loss represents images as minimum spanning trees to model both low-level and high-level pair-wise affini-ties. By sequentially applying these affinities to the net-work prediction, soft pseudo labels for unlabeled pixels are generated in a coarse-to-fine manner, achieving dynamic online self-training. The tree energy loss is effective and easy to be incorporated into existing frameworks by com-bining it with a traditional segmentation loss. Compared with previous SASS methods, our method requires no multi-stage training strategies, alternating optimization proce-dures, additional supervised data, or time-consuming post-processing while outperforming them in all SASS settings. Code is available at https://github.com/megvii-research/TreeEnergyLoss.

----

## [1628] Bending Reality: Distortion-aware Transformers for Adapting to Panoramic Semantic Segmentation

**Authors**: *Jiaming Zhang, Kailun Yang, Chaoxiang Ma, Simon Reiß, Kunyu Peng, Rainer Stiefelhagen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01641](https://doi.org/10.1109/CVPR52688.2022.01641)

**Abstract**:

Panoramic images with their 360° directional view encompass exhaustive information about the surrounding space, providing a rich foundation for scene understanding. To unfold this potential in the form of robust panoramic segmentation models, large quantities of expensive, pixel-wise annotations are crucial for success. Such annotations are available, but predominantly for narrow-angle, pinhole-camera images which, off the shelf, serve as sub-optimal resources for training panoramic models. Distortions and the distinct image-feature distribution in 360° panoramas impede the transfer from the annotation-rich pinhole domain and therefore come with a big dent in performance. To get around this domain difference and bring together semantic annotations from pinhole- and 360° surround-visuals, we propose to learn object deformations and panoramic image distortions in the Deformable Patch Embedding (DPE) and Deformable MLP (DMLP) components which blend into our Transformer for PAnoramic Semantic Segmentation (Trans4PASS) model. Finally, we tie together shared semantics in pinhole- and panoramic feature embeddings by generating multi-scale prototype features and aligning them in our Mutual Prototypical Adaptation (MPA) for unsupervised domain adaptation. On the indoor Stanford2D3D dataset, our Trans4PASS with MPA maintains comparable performance to fully-supervised state-of-the-arts, cutting the need for over 1,400 labeled panoramas. On the outdoor DensePASS dataset, we break state-of-the-art by 14.39% mIoU and set the new bar at 56.38%.

----

## [1629] MM-TTA: Multi-Modal Test-Time Adaptation for 3D Semantic Segmentation

**Authors**: *Inkyu Shin, Yi-Hsuan Tsai, Bingbing Zhuang, Samuel Schulter, Buyu Liu, Sparsh Garg, In So Kweon, Kuk-Jin Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01642](https://doi.org/10.1109/CVPR52688.2022.01642)

**Abstract**:

Test-time adaptation approaches have recently emerged as a practical solution for handling domain shift without access to the source domain data. In this paper, we propose and explore a new multi-modal extension of test-time adaptation for 3D semantic segmentation. We find that, directly applying existing methods usually results in performance instability at test time, because multi-modal input is not considered jointly. To design a framework that can take full advantage of multi-modality, where each modality provides regularized self-supervisory signals to other modalities, we propose two complementary modules within and across the modalities. First, Intra-modal Pseudo-label Generation (Intra-PG) is introduced to obtain reliable pseudo labels within each modality by aggregating information from two models that are both pre-trained on source data but updated with target data at different paces. Second, Inter-modal Pseudo-label Refinement (Inter-PR) adaptively selects more reliable pseudo labels from different modalities based on a proposed consistency scheme. Experiments demonstrate that our regularized pseudo labels produce stable self-learning signals in numerous multi-modal test-time adaptation scenarios for 3D semantic segmentation. Visit our project website at https://www.nec-labs.com/~mas/MM-TTA

----

## [1630] NightLab: A Dual-level Architecture with Hardness Detection for Segmentation at Night

**Authors**: *Xueqing Deng, Peng Wang, Xiaochen Lian, Shawn D. Newsam*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01643](https://doi.org/10.1109/CVPR52688.2022.01643)

**Abstract**:

The semantic segmentation of nighttime scenes is a challenging problem that is key to impactful applications like self-driving cars. Yet, it has received little attention compared to its daytime counterpart. In this paper, we propose NightLab, a novel nighttime segmentation framework that leverages multiple deep learning models imbued with night-aware features to yield State-of-The-Art (SoTA) performance on multiple night segmentation benchmarks. Notably, NightLab contains models at two levels of granularity, i.e. image and regional, and each level is composed of light adaptation and segmentation modules. Given a nighttime image, the image level model provides an initial segmentation estimate while, in parallel, a hardness detection module identifies regions and their surrounding context that need further analysis. A regional level model focuses on these difficult regions to provide a significantly improved segmentation. All the models in NightLab are trained end-to-end using a set of proposed night-aware losses without handcrafted heuristics. Extensive experiments on the NightCity [44] and BDD100K [59] datasets show NightLab achieves SoTA performance compared to concurrent methods. Code and dataset are available at https://github.com/xdeng7/NightLab.

----

## [1631] Fast Point Transformer

**Authors**: *Chunghyun Park, Yoonwoo Jeong, Minsu Cho, Jaesik Park*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01644](https://doi.org/10.1109/CVPR52688.2022.01644)

**Abstract**:

The recent success of neural networks enables a better interpretation of 3D point clouds, but processing a large-scale 3D scene remains a challenging problem. Most current approaches divide a large-scale scene into small regions and combine the local predictions together. However, this scheme inevitably involves additional stages for pre- and post-processing and may also degrade the final output due to predictions in a local perspective. This paper introduces Fast Point Transformer that consists of a new lightweight self-attention layer. Our approach encodes continuous 3D coordinates, and the voxel hashing-based architecture boosts computational efficiency. The proposed method is demonstrated with 3D semantic segmentation and 3D detection. The accuracy of our approach is competitive to the best voxel-based method, and our network achieves 129 times faster inference time than the state-of-the-art, Point Transformer, with a reasonable accuracy trade-off in 3D semantic segmentation on S3DIS dataset.

----

## [1632] RigidFlow: Self-Supervised Scene Flow Learning on Point Clouds by Local Rigidity Prior

**Authors**: *Ruibo Li, Chi Zhang, Guosheng Lin, Zhe Wang, Chunhua Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01645](https://doi.org/10.1109/CVPR52688.2022.01645)

**Abstract**:

In this work, we focus on scene flow learning on point clouds in a self-supervised manner. A real-world scene can be well modeled as a collection of rigidly moving parts, therefore its scene flow can be represented as a combination of rigid motion of each part. Inspired by this observation, we propose to generate pseudo scene flow for self-supervised learning based on piecewise rigid motion estimation, in which the source point cloud is decomposed into a set of local regions and each region is treated as rigid. By rigidly aligning each region with its potential counterpart in the target point cloud, we obtain a region-specific rigid transformation to represent the flow, which together constitutes the pseudo scene flow labels of the entire scene to enable network training. Compared with most existing approaches relying on point-wise similarities for scene flow approximation, our method explicitly enforces region-wise rigid alignments, yielding locally rigid pseudo scene flow labels. We demonstrate the effectiveness of our self-supervised learning method on FlyingThings3D and KITTI datasets. Comprehensive experiments show that our method achieves new state-of-the-art performance in self-supervised scene flow learning, without any ground truth scene flow for supervision, even outperforming some super-vised counterparts.

----

## [1633] ConDor: Self-Supervised Canonicalization of 3D Pose for Partial Shapes

**Authors**: *Rahul Sajnani, Adrien Poulenard, Jivitesh Jain, Radhika Dua, Leonidas J. Guibas, Srinath Sridhar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01646](https://doi.org/10.1109/CVPR52688.2022.01646)

**Abstract**:

Progress in 3D object understanding has relied on manually “canonicalized” shape datasets that contain instances with consistent position and orientation (3D pose). This has made it hard to generalize these methods to in-the-wild shapes, e.g., from internet model collections or depth sensors. ConDor is a self-supervised method that learns to Canonicalize the 3D orientation and position for full and partial 3D point clouds. We build on top of Tensor Field Networks (TFNs), a class of permutation- and rotation-equivariant, and translation-invariant 3D networks. During inference, our method takes an unseen full or partial 3D point cloud at an arbitrary pose and outputs an equivariant canonical pose. During training, this network uses self-supervision losses to learn the canonical pose from an un-canonicalized collection of full and partial 3D point clouds. ConDor can also learn to consistently co-segment object parts without any supervision. Extensive quantitative results on four new metrics show that our approach out-performs existing methods while enabling new applications such as operation on depth images and annotation transfer.

----

## [1634] DisARM: Displacement Aware Relation Module for 3D Detection

**Authors**: *Yao Duan, Chenyang Zhu, Yuqing Lan, Renjiao Yi, Xinwang Liu, Kai Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01647](https://doi.org/10.1109/CVPR52688.2022.01647)

**Abstract**:

We introduce Displacement Aware Relation Module (DisARM), a novel neural network module for enhancing the performance of 3D object detection in point cloud scenes. The core idea is extracting the most principal contextual information is critical for detection while the target is incomplete or featureless. We find that relations between proposals provide a good representation to describe the context. However, adopting relations between all the object or patch proposals for detection is inefficient, and an imbalanced combination of local and global relations brings extra noise that could mislead the training. Rather than working with all relations, we find that training with relations only between the most representative ones, or an-chors, can significantly boost the detection performance. Good anchors should be semantic-aware with no ambiguity and able to describe the whole layout of a scene with no redundancy. To find the anchors, we first perform a preliminary relation anchor module with an objectness-aware sampling approach and then devise a displacement based module for weighing the relation importance for better utilization of contextual information. This lightweight relation module leads to significantly higher accuracy of object instance detection when being plugged into the state-of-the-art detectors. Evaluations on the public benchmarks of real-world scenes show that our method achieves the state-of-the-art performance on both SUN RGB-D and Scan-Net V2. The code and models are publicly available at https://github.com/YaraDuan/DisARM.

----

## [1635] Learning Object Context for Novel-view Scene Layout Generation

**Authors**: *Xiaotian Qiao, Gerhard P. Hancke, Rynson W. H. Lau*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01648](https://doi.org/10.1109/CVPR52688.2022.01648)

**Abstract**:

Novel-view prediction of a scene has many applications. Existing works mainly focus on generating novel-view images via pixel-wise prediction in the image space, often resulting in severe ghosting and blurry artifacts. In this paper, we make the first attempt to explore novel-view prediction in the layout space, and introduce the new problem of novel-view scene layout generation. Given a single scene layout and the camera transformation as inputs, our goal is to generate a plausible scene layout for a specified viewpoint. Such a problem is challenging as it involves accurate understanding of the 3D geometry and semantics of the scene from as little as a single 2D scene layout. To tackle this challenging problem, we propose a deep model to capture contextualized object representation by explicitly modeling the object context transformation in the scene. The contextualized object representation is essential in generating geometrically and semantically consistent scene layouts of different views. Experiments show that our model outperforms several strong baselines on many indoor and outdoor scenes, both qualitatively and quantitatively. We also show that our model enables a wide range of applications, including novel-view image synthesis, novel-view image editing, and amodal object estimation.

----

## [1636] Weakly But Deeply Supervised Occlusion-Reasoned Parametric Road Layouts

**Authors**: *Buyu Liu, Bingbing Zhuang, Manmohan Chandraker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01649](https://doi.org/10.1109/CVPR52688.2022.01649)

**Abstract**:

We propose an end-to-end network that takes a single perspective RGB image of a complex road scene as input, to produce occlusion-reasoned layouts in perspective space as well as a parametric bird's-eye-view (BEV) space. In contrast to prior works that require dense supervision such as semantic labels in perspective view, our method only requires human annotations for parametric attributes that are cheaper and less ambiguous to obtain. To solve this challenging task, our design is comprised of modules that incorporate inductive biases to learn occlusion-reasoning, geometric transformation and semantic abstraction, where each module may be supervised by appropriately transforming the parametric annotations. We demonstrate how our design choices and proposed deep supervision help achieve meaningful representations and accurate predictions. We validate our approach on two public datasets, KITTI and NuScenes, to achieve state-of-the-art results with considerably less human supervision.

----

## [1637] Beyond Cross-view Image Retrieval: Highly Accurate Vehicle Localization Using Satellite Image

**Authors**: *Yujiao Shi, Hongdong Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01650](https://doi.org/10.1109/CVPR52688.2022.01650)

**Abstract**:

This paper addresses the problem of vehicle-mounted camera localization by matching a ground-level image with an overhead-view satellite map. Existing methods often treat this problem as cross-view image retrieval, and use learned deep features to match the ground-level query im-age to a partition (e.g., a small patch) of the satellite map. By these methods, the localization accuracy is limited by the partitioning density of the satellite map (often in the order of tens meters). Departing from the conventional wisdom of image retrieval, this paper presents a novel solution that can achieve highly-accurate localization. The key idea is to formulate the task as pose estimation and solve it by neural-net based optimization. Specifically, we design a two-branch CNN to extract robust features from the ground and satellite images, respectively. To bridge the vast cross-view domain gap, we resort to a Geometry Projection module that projects features from the satellite map to the ground-view, based on a relative camera pose. Aiming to minimize the differences between the projected features and the observed features, we employ a differentiable Levenberg-Marquardt (LM) module to search for the optimal camera pose iteratively. The entire pipeline is differen-tiable and runs end-to-end. Extensive experiments on standard autonomous vehicle localization datasets have confirmed the superiority of the proposed method. Notably, e.g., starting from a coarse estimate of camera location within a wide region of 40m × 40m, with an 80% likelihood our method quickly reduces the lateral location error to be within 5m on a new KITTI cross-view dataset.

----

## [1638] Raw High-Definition Radar for Multi-Task Learning

**Authors**: *Julien Rebut, Arthur Ouaknine, Waqas Malik, Patrick Pérez*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01651](https://doi.org/10.1109/CVPR52688.2022.01651)

**Abstract**:

With their robustness to adverse weather conditions and ability to measure speeds, radar sensors have been part of the automotive landscape for more than two decades. Recent progress toward High Definition (HD) Imaging radar has driven the angular resolution below the degree, thus approaching laser scanning performance. However, the amount of data a HD radar delivers and the computational cost to estimate the angular positions remain a challenge. In this paper, we propose a novel HD radar sensing model, FFT-RadNet, that eliminates the overhead of computing the range-azimuth-Doppler 3D tensor, learning instead to recover angles from a range-Doppler spectrum. FFT-RadNet is trained both to detect vehicles and to segment free driving space. On both tasks, it competes with the most recent radar-based models while requiring less compute and memory. Also, we collected and annotated 2-hour worth of raw data from synchronized automotive-grade sensors (camera, laser, HD radar) in various environments (city street, highway, countryside road). This unique dataset, nick-named RADIal for “Radar, LiDAR et al.”, is available at https://github.com/valeoai/RADIal.

----

## [1639] Zero Experience Required: Plug & Play Modular Transfer Learning for Semantic Visual Navigation

**Authors**: *Ziad Al-Halah, Santhosh K. Ramakrishnan, Kristen Grauman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01652](https://doi.org/10.1109/CVPR52688.2022.01652)

**Abstract**:

In reinforcement learning for visual navigation, it is common to develop a model for each new task, and train that model from scratch with task-specific interactions in 3D environments. However, this process is expensive; mas-sive amounts of interactions are needed for the model to generalize well. Moreover, this process is repeated when-ever there is a change in the task type or the goal modality. We present a unified approach to visual navigation using a novel modular transfer learning model. Our model can ef-fectively leverage its experience from one source task and apply it to multiple target tasks (e.g., ObjectNav, Room-Nav, Vi ewNav) with various goal modalities (e.g., image, sketch, audio, label). Furthermore, our model enables zero-shot experience learning, whereby it can solve the target tasks without receiving any task-specific interactive training. Our experiments on multiple photorealistic datasets and challenging tasks show that our approach learns faster, generalizes better, and outperforms SoTA models by a sig-nificant margin. Project page: https://vision.cs.utexas.edu/projects/zsel/

----

## [1640] UKPGAN: A General Self-Supervised Keypoint Detector

**Authors**: *Yang You, Wenhai Liu, Yanjie Ze, Yong-Lu Li, Weiming Wang, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01653](https://doi.org/10.1109/CVPR52688.2022.01653)

**Abstract**:

Keypoint detection is an essential component for the object registration and alignment. In this work, we reckon keypoint detection as information compression, and force the model to distill out important points of an object. Based on this, we propose UKPGAN, a general self-supervised 3D keypoint detector where keypoints are detected so that they could reconstruct the original object shape. Two modules: GAN-based keypoint sparsity control and salient information distillation modules are proposed to locate those important keypoints. Extensive experiments show that our keypoints align well with human annotated keypoint labels, and can be applied to SMPL human bodies under various non-rigid deformations. Furthermore, our keypoint detector trained on clean object collections generalizes well to real-world scenarios, thus further improves geometric registration when combined with off-the-shelf point descriptors. Repeatability experiments show that our model is stable under both rigid and non-rigid transformations, with local reference frame estimation. Our code is available on https://github.com/qq456cvb/UKPGAN.

----

## [1641] Cannot See the Forest for the Trees: Aggregating Multiple Viewpoints to Better Classify Objects in Videos

**Authors**: *Sukjun Hwang, Miran Heo, Seoung Wug Oh, Seon Joo Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01654](https://doi.org/10.1109/CVPR52688.2022.01654)

**Abstract**:

Recently, both long-tailed recognition and object tracking have made great advances individually. TAO benchmark presented a mixture of the two, long-tailed object tracking, in order to further reflect the aspect of the real-world. To date, existing solutions have adopted detectors showing robustness in long-tailed distributions, which derive per-frame results. Then, they used tracking algorithms that combine the temporally independent detections to finalize tracklets. However, as the approaches did not take temporal changes in scenes into account, inconsistent classification results in videos led to low overall performance. In this paper, we present a set classifier that improves accuracy of classifying tracklets by aggregating information from multiple viewpoints contained in a tracklet. To cope with sparse annotations in videos, we further propose augmentation of tracklets that can maximize data efficiency. The set classifier is plug-and-playable to existing object trackers, and highly improves the performance of long-tailed object tracking. By simply attaching our method to QDTrack on top of ResNet-101, we achieve the new state-of-the-art, 19.9% and 15.7% 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$TrackAP_{50}$</tex>
 on TAO validation and test sets, respectively. Our code is available at this link
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/sukjunhwang/setclassifier.

----

## [1642] Rethinking Efficient Lane Detection via Curve Modeling

**Authors**: *Zhengyang Feng, Shaohua Guo, Xin Tan, Ke Xu, Min Wang, Lizhuang Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01655](https://doi.org/10.1109/CVPR52688.2022.01655)

**Abstract**:

This paper presents a novel parametric curve-based method for lane detection in RGB images. Unlike state-of-the-art segmentation-based and point detection-based methods that typically require heuristics to either decode predictions or formulate a large sum of anchors, the curve-based methods can learn holistic lane representations naturally. To handle the optimization difficulties of existing poly-nomial curve methods, we propose to exploit the parametric Bézier curve due to its ease of computation, stability, and high freedom degrees of transformations. In addition, we propose the deformable convolution-based feature flip fusion, for exploiting the symmetry properties of lanes in driving scenes. The proposed method achieves a new state-of-the-art performance on the popular LLAMAS benchmark. It also achieves favorable accuracy on the TuSimple and CULane datasets, while retaining both low latency (>150 FPS) and small model size (<10M). Our method can serve as a new baseline, to shed the light on the parametric curves modeling for lane detection. Codes of our model and PytorchAutoDrive: a unified framework for self-driving perception, are available at: https://github.com/voldemortX/pytorch-auto-drive.

----

## [1643] Exploiting Temporal Relations on Radar Perception for Autonomous Driving

**Authors**: *Peizhao Li, Pu Wang, Karl Berntorp, Hongfu Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01656](https://doi.org/10.1109/CVPR52688.2022.01656)

**Abstract**:

We consider the object recognition problem in autonomous driving using automotive radar sensors. Comparing to Lidar sensors, radar is cost-effective and robust in all- weather conditions for perception in autonomous driving. However, radar signals suffer from low angular resolution and precision in recognizing surrounding objects. To enhance the capacity of automotive radar, in this work, we exploit the temporal information from successive ego-centric bird-eye-view radar image frames for radar object recognition. We leverage the consistency of an object's existence and attributes (size, orientation, etc.), and propose a temporal relational layer to explicitly model the relations between objects within successive radar images. In both object detection and multiple object tracking, we show the superiority of our method compared to several baseline approaches.

----

## [1644] Towards Robust and Adaptive Motion Forecasting: A Causal Representation Perspective

**Authors**: *Yuejiang Liu, Riccardo Cadei, Jonas Schweizer, Sherwin Bahmani, Alexandre Alahi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01657](https://doi.org/10.1109/CVPR52688.2022.01657)

**Abstract**:

Learning behavioral patterns from observational data has been a de-facto approach to motion forecasting. Yet, the current paradigm suffers from two shortcomings: brittle under distribution shifts and inefficient for knowledge transfer. In this work, we propose to address these challenges from a causal representation perspective. We first introduce a causal formalism of motion forecasting, which casts the problem as a dynamic process with three groups of latent variables, namely invariant variables, style confounders, and spurious features. We then introduce a learning framework that treats each group separately: (i) unlike the common practice mixing datasets collected from different locations, we exploit their subtle distinctions by means of an invariance loss encouraging the model to suppress spurious correlations; (ii) we devise a modular architecture that factorizes the representations of invariant mechanisms and style confounders to approximate a sparse causal graph; (iii) we introduce a style contrastive loss that not only enforces the structure of style representations but also serves as a self-supervisory signal for test-time refinement on the fly. Experiments on synthetic and real datasets show that our proposed method improves the robustness and reusability of learned motion representations, significantly outperforming prior state-of-the-art motion forecasting models for out-of-distribution generalization and low-shot transfer.

----

## [1645] BE-STI: Spatial-Temporal Integrated Network for Class-agnostic Motion Prediction with Bidirectional Enhancement

**Authors**: *Yunlong Wang, Hongyu Pan, Jun Zhu, Yu-Huan Wu, Xin Zhan, Kun Jiang, Diange Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01658](https://doi.org/10.1109/CVPR52688.2022.01658)

**Abstract**:

Determining the motion behavior of inexhaustible categories of traffic participants is critical for autonomous driving. In recent years, there has been a rising concern in performing class-agnostic motion prediction directly from the captured sensor data, like LiDAR point clouds or the combination of point clouds and images. Current motion prediction frameworks tend to perform joint semantic segmentation and motion prediction and face the trade-off between the performance of these two tasks. In this paper, we propose a novel Spatial-Temporal Integrated network with Bidirectional Enhancement, BE-STI, to improve the temporal motion prediction performance by spatial semantic features, which points out an efficient way to combine semantic segmentation and motion prediction. Specifically, we propose to enhance the spatial features of each individual point cloud with the similarity among temporal neighboring frames and enhance the global temporal features with the spatial difference among non-adjacent frames in a coarse-to-fine fashion. Extensive experiments on nuScenes and Waymo Open Dataset show that our proposed framework outperforms all state-of-the-art LiDAR-based and RGB+LiDAR-based methods with remarkable margins by using only point clouds as input.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code will be released at https://github.com/be-sti/be-sti.

----

## [1646] ScePT: Scene-consistent, Policy-based Trajectory Predictions for Planning

**Authors**: *Yuxiao Chen, Boris Ivanovic, Marco Pavone*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01659](https://doi.org/10.1109/CVPR52688.2022.01659)

**Abstract**:

Trajectory prediction is a critical functionality of autonomous systems that share environments with uncontrolled agents, one prominent example being self-driving vehicles. Currently, most prediction methods do not enforce scene consistency, i.e., there are a substantial amount of self-collisions between predicted trajectories of different agents in the scene. Moreover, many approaches generate individual trajectory predictions per agent instead of joint trajectory predictions of the whole scene, which makes downstream planning difficult. In this work, we present ScePT, a policy planning-based trajectory prediction model that generates accurate, scene-consistent trajectory predictions suitable for autonomous system motion planning. It explicitly enforces scene consistency and learns an agent interaction policy that can be used for conditional prediction. Experiments on multiple real-world pedestrians and autonomous vehicle datasets show that ScePT matches current state-of-the-art prediction accuracy with significantly improved scene consistency. We also demonstrate ScePT's ability to work with a downstream contingency planner.

----

## [1647] Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion

**Authors**: *Tianpei Gu, Guangyi Chen, Junlong Li, Chunze Lin, Yongming Rao, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01660](https://doi.org/10.1109/CVPR52688.2022.01660)

**Abstract**:

Human behavior has the nature of indeterminacy, which requires the pedestrian trajectory prediction system to model the multi-modality of future motion states. Unlike existing stochastic trajectory prediction methods which usually use a latent variable to represent multi-modality, we explicitly simulate the process of human motion variation from indeterminate to determinate. In this paper, we present a new framework to formulate the trajectory prediction task as a reverse process of motion indeterminacy diffusion (MID), in which we progressively discard indeterminacy from all the walkable areas until reaching the desired trajectory. This process is learned with a parameterized Markov chain conditioned by the observed trajectories. We can adjust the length of the chain to control the degree of indeterminacy and balance the diversity and determinacy of the predictions. Specifically, we encode the history behavior information and the social interactions as a state embedding and devise a Transformer-based diffusion model to capture the temporal dependencies of trajectories. Extensive experiments on the human trajectory prediction benchmarks including the Stanford Drone and ETH/UCY datasets demonstrate the superiority of our method. Code is available at https://github.com/gutianpei/MID.

----

## [1648] Vehicle trajectory prediction works, but not everywhere

**Authors**: *Mohammadhossein Bahari, Saeed Saadatnejad, Ahmad Rahimi, Mohammad Shaverdikondori, Amir Hossein Shahidzadeh, Seyed-Mohsen Moosavi-Dezfooli, Alexandre Alahi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01661](https://doi.org/10.1109/CVPR52688.2022.01661)

**Abstract**:

Vehicle trajectory prediction is nowadays a fundamental pillar of self-driving cars. Both the industry and research communities have acknowledged the need for such a pillar by providing public benchmarks. While state-of-the-art methods are impressive, i.e., they have no off-road prediction, their generalization to cities outside of the benchmark remains unexplored. In this work, we show that those methods do not generalize to new scenes. We present a method that automatically generates realistic scenes causing state-of-the-art models to go off-road. We frame the problem through the lens of adversarial scene generation. The method is a simple yet effective generative model based on atomic scene generation functions along with physical constraints. Our experiments show that more than 60% of existing scenes from the current benchmarks can be modified in a way to make prediction methods fail (i.e., predicting off-road). We further show that the generated scenes (i) are realistic since they do exist in the real world, and (ii) can be used to make existing models more robust, yielding 30-40% reductions in the off-road rate. The code is available online: https://s-attack.github.io/.

----

## [1649] LTP: Lane-based Trajectory Prediction for Autonomous Driving

**Authors**: *Jingke Wang, Tengju Ye, Ziqing Gu, Junbo Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01662](https://doi.org/10.1109/CVPR52688.2022.01662)

**Abstract**:

The reasonable trajectory prediction of surrounding traf-fic participants is crucial for autonomous driving. Espe-cially, how to predict multiple plausible trajectories is still a challenging problem because of the multiple possibilities of the future. Proposal-based prediction methods address the multi-modality issues with a two-stage approach, com-monly using intention classification followed by motion re-gression. This paper proposes a two-stage proposal-based motion forecasting method that exploits the sliced lane seg-ments as fine-grained, shareable, and interpretable propos-als. We use Graph neural network and Transformer to en-code the shape and interaction information among the map sub-graphs and the agents sub-graphs. In addition, we propose a variance-based non-maximum suppression strategy to select representative trajectories that ensure the diversity of the final output. Experiments on the Argoverse dataset show that the proposed method outperforms state-of-the-art methods, and the lane segments-based proposals as well as the variance-based non-maximum suppression strategy both contribute to the performance improvement. More-over, we demonstrate that the proposed method can achieve reliable performance with a lower collision rate and fewer off-road scenarios in the closed-loop simulation.

----

## [1650] ONCE-3DLanes: Building Monocular 3D Lane Detection

**Authors**: *Fan Yan, Ming Nie, Xinyue Cai, Jianhua Han, Hang Xu, Zhen Yang, Chaoqiang Ye, Yanwei Fu, Michael Bi Mi, Li Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01663](https://doi.org/10.1109/CVPR52688.2022.01663)

**Abstract**:

We present ONCE-3DLanes, a real-world autonomous driving dataset with lane layout annotation in 3D space. Conventional 2D lane detection from a monocular image yields poor performance of following planning and control tasks in autonomous driving due to the case of uneven road. Predicting the 3D lane layout is thus necessary and enables effective and safe driving. However, existing 3D lane detection datasets are either unpublished or synthesized from a simulated environment, severely hampering the development of this field. In this paper, we take steps towards addressing these issues. By exploiting the explicit relationship between point clouds and image pixels, a dataset annotation pipeline is designed to automatically generate high-quality 3D lane locations from 2D lane annotations in 211K road scenes. In addition, we present an extrinsic-free, anchorfree method, called SALAD, regressing the 3D coordinates of lanes in image view without converting the feature map into the bird's-eye view (BEV). To facilitate future research on 3D lane detection, we benchmark the dataset and provide a novel evaluation metric, performing extensive experiments of both existing approaches and our proposed method. The aim of our work is to revive the interest of 3D lane detection in a real-world scenario. We believe our work can lead to the expected and unexpected innovations in both academia and industry.

----

## [1651] Towards Driving-Oriented Metric for Lane Detection Models

**Authors**: *Takami Sato, Qi Alfred Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01664](https://doi.org/10.1109/CVPR52688.2022.01664)

**Abstract**:

After the 2017 TuSimple Lane Detection Challenge, its dataset and evaluation based on accuracy and F1 score have become the de facto standard to measure the performance of lane detection methods. While they have played a major role in improving the performance of lane detection methods, the validity of this evaluation method in down-stream tasks has not been adequately researched. In this study, we design 2 new driving-oriented metrics for lane detection: End-to-End Lateral Deviation metric (E2E-LD) is directly formulated based on the requirements of autonomous driving, a core downstream task of lane detection; Per-frame Simulated Lateral Deviation metric (PSLD) is a lightweight surrogate metric of E2E-LD. To evaluate the validity of the metrics, we conduct a large-scale empirical study with 4 major types of lane detection approaches on the TuSimple dataset and our newly constructed dataset Comma2k19-LD. Our results show that the conventional metrics have strongly negative correlations (≤-0.55) with E2E-LD, meaning that some recent improvements purely targeting the conventional metrics may not have led to meaningful improvements in autonomous driving, but rather may actually have made it worse by over-fitting to the conventional metrics. As autonomous driving is a security/safety-critical system, the underestimation of robustness hinders the sound development of practical lane detection models. We hope that our study will help the community achieve more downstream task-aware evaluations for lane detection.

----

## [1652] Eigenlanes: Data-Driven Lane Descriptors for Structurally Diverse Lanes

**Authors**: *Dongkwon Jin, Wonhui Park, Seong-Gyun Jeong, Heeyeon Kwon, Chang-Su Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01665](https://doi.org/10.1109/CVPR52688.2022.01665)

**Abstract**:

A novel algorithm to detect road lanes in the eigen-lane space is proposed in this paper. First, we introduce the notion of eigenlanes, which are data-driven descriptors for structurally diverse lanes, including curved, as well as straight, lanes. To obtain eigenlanes, we perform the best rank-M approximation of a lane matrix containing all lanes in a training set. Second, we generate a set of lane candi-dates by clustering the training lanes in the eigenlane space. Third, using the lane candidates, we determine an optimal set of lanes by developing an anchor-based detection net-work, called SIIC-Net. Experimental results demonstrate that the proposed algorithm provides excellent detection performance for structurally diverse lanes. Our codes are available at https://github.com/dongkwonjin/Eigenlanes.

----

## [1653] LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection

**Authors**: *Yihan Zeng, Da Zhang, Chunwei Wang, Zhenwei Miao, Ting Liu, Xin Zhan, Dayang Hao, Chao Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01666](https://doi.org/10.1109/CVPR52688.2022.01666)

**Abstract**:

LiDAR and camera are two common sensors to collect data in time for 3D object detection under the autonomous driving context. Though the complementary information across sensors and time has great potential of benefiting 3D perception, taking full advantage of sequential cross-sensor data still remains challenging. In this paper, we propose a novel LiDAR Image Fusion Transformer (LIFT) to model the mutual interaction relationship of cross-sensor data over time. LIFT learns to align the input 4D sequential cross-sensor data to achieve multi-frame multi-modal information aggregation. To alleviate computational load, we project both point clouds and images into the bird-eye-view maps to compute sparse grid-wise self-attention. LIFT also benefits from a cross-sensor and cross-time data augmentation scheme. We evaluate the proposed approach on the challenging nuScenes and Waymo datasets, where our LIFT performs well over the state-of-the-art and strong baselines.

----

## [1654] DeepFusion: Lidar-Camera Deep Fusion for Multi-Modal 3D Object Detection

**Authors**: *Yingwei Li, Adams Wei Yu, Tianjian Meng, Benjamin Caine, Jiquan Ngiam, Daiyi Peng, Junyang Shen, Yifeng Lu, Denny Zhou, Quoc V. Le, Alan L. Yuille, Mingxing Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01667](https://doi.org/10.1109/CVPR52688.2022.01667)

**Abstract**:

Lidars and cameras are critical sensors that provide complementary information for 3D detection in autonomous driving. While prevalent multi-modal methods [34], [36] simply decorate raw lidar point clouds with camera features and feed them directly to existing 3D detection models, our study shows that fusing camera features with deep lidar features instead of raw points, can lead to better performance. However, as those features are often augmented and aggregated, a key challenge in fusion is how to effectively align the transformed features from two modalities. In this paper, we propose two novel techniques: InverseAug that inverses geometric-related augmentations, e.g., rotation, to enable accurate geometric alignment between lidar points and image pixels, and LearnableAlign that leverages cross-attention to dynamically capture the correlations between image and lidar features during fusion. Based on InverseAug and LearnableAlign, we develop a family of generic multi-modal 3D detection models named DeepFusion, which is more accurate than previous methods. For example, DeepFusion improves Point-Pillars, CenterPoint, and 3D-MAN baselines on Pedestrian detection for 6.7,8.9, and 6.2 LEVEL_2 APH, respectively. Notably, our models achieve state-of-the-art performance on Waymo Open Dataset, and show strong model robustness against input corruptions and out-of-distribution data. Code will be publicly available at https://github.com/tensorflow/lingvo.

----

## [1655] A Versatile Multi-View Framework for LiDAR-based 3D Object Detection with Guidance from Panoptic Segmentation

**Authors**: *Hamidreza Fazlali, Yixuan Xu, Yuan Ren, Bingbing Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01668](https://doi.org/10.1109/CVPR52688.2022.01668)

**Abstract**:

3D object detection using LiDAR data is an indispensable component for autonomous driving systems. Yet, only a few LiDAR-based 3D object detection methods leverage segmentation information to further guide the detection process. In this paper, we propose a novel multi-task framework that jointly performs 3D object detection and panoptic segmentation. In our method, the 3D object detection backbone in Bird’s-Eye-View (BEV) plane is augmented by the injection of Range-View (RV) feature maps from the 3D panoptic segmentation backbone. This enables the detection backbone to leverage multi-view information to address the shortcomings of each projection view. Furthermore, foreground semantic information is incorporated to ease the detection task by highlighting the locations of each object class in the feature maps. Finally, a new center density heatmap generated based on the instance-level information further guides the detection backbone by suggesting possible box center locations for objects. Our method works with any BEV-based 3D object detection method and, based on experiments on the nuScenes dataset, it provides significant performance gains. Notably, the proposed method based on a single-stage CenterPoint 3D object detection network achieve state-of-the-art performance on nuScenes 3D Detection Benchmark with 67.3 NDS.

----

## [1656] Forecasting from LiDAR via Future Object Detection

**Authors**: *Neehar Peri, Jonathon Luiten, Mengtian Li, Aljosa Osep, Laura Leal-Taixé, Deva Ramanan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01669](https://doi.org/10.1109/CVPR52688.2022.01669)

**Abstract**:

Object detection and forecasting are fundamental components of embodied perception. These two problems, how-ever, are largely studied in isolation by the community. In this paper, we propose an end-to-end approachfor detection and motion forecasting based on raw sensor measurement as opposed to ground truth tracks. Instead of predicting the current frame locations and forecasting forward in time, we directly predict future object locations and backcast to determine where each trajectory began. Our approach not only improves overall accuracy compared to other modular or end-to-end baselines, it also prompts us to rethink the role of explicit tracking for embodied perception. Additionally, by linking future and current locations in a many-to-one manner, our approach is able to reason about multiple futures, a capability that was previously considered difficult for end-to-end approaches. We conduct extensive experi-ments on the popular nuScenes dataset and demonstrate the empirical effectiveness of our approach. In addition, we investigate the appropriateness of reusing standard forecasting metrics for an end-to-end setup, and find a number of limitations which allow us to build simple baselines to game these metrics. We address this issue with a novel set of joint forecasting and detection metrics that extend the commonly used AP metrics from the detection community to measuring forecasting accuracy. Our code is available on GitHub.

----

## [1657] RIDDLE: Lidar Data Compression with Range Image Deep Delta Encoding

**Authors**: *Xuanyu Zhou, Charles R. Qi, Yin Zhou, Dragomir Anguelov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01670](https://doi.org/10.1109/CVPR52688.2022.01670)

**Abstract**:

Lidars are depth measuring sensors widely used in autonomous driving and augmented reality. However, the large volume of data produced by lidars can lead to high costs in data storage and transmission. While lidar data can be represented as two interchangeable representations: 3D point clouds and range images, most previous work focus on compressing the generic 3D point clouds. In this work, we show that directly compressing the range images can leverage the lidar scanning pattern, compared to compressing the unprojected point clouds. We propose a novel datadriven range image compression algorithm, named RIDDLE (Range Image Deep DeLta Encoding). At its core is a deep model that predicts the next pixel value in a raster scanning order, based on contextual laser shots from both the current and past scans (represented as a 4D point cloud of spherical coordinates and time). The deltas between predictions and original values can then be compressed by entropy encoding. Evaluated on the Waymo Open Dataset and KITTI, our method demonstrates significant improvement in the compression rate (under the same distortion) compared to widely used point cloud and range image compression algorithms as well as recent deep methods.

----

## [1658] Learning from All Vehicles

**Authors**: *Dian Chen, Philipp Krähenbühl*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01671](https://doi.org/10.1109/CVPR52688.2022.01671)

**Abstract**:

In this paper, we present a system to train driving policies from experiences collected not just from the ego-vehicle, but all vehicles that it observes. This system uses the behaviors of other agents to create more diverse driving scenarios without collecting additional data. The main difficulty in learning from other vehicles is that there is no sensor information. We use a set of supervisory tasks to learn an intermediate representation that is invariant to the viewpoint of the controlling vehicle. This not only provides a richer signal at training time but also allows more complex reasoning during inference. Learning how all vehicles drive helps predict their behavior at test time and can avoid collisions. We evaluate this system in closed-loop driving simulations. Our system outperforms all prior methods on the public CARLA Leaderboard by a wide margin, improving driving score by 25 and route completion rate by 24 points.

----

## [1659] Is Mapping Necessary for Realistic PointGoal Navigation?

**Authors**: *Ruslan Partsey, Erik Wijmans, Naoki Yokoyama, Oles Dobosevych, Dhruv Batra, Oleksandr Maksymets*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01672](https://doi.org/10.1109/CVPR52688.2022.01672)

**Abstract**:

Can an autonomous agent navigate in a new environment without building an explicit map? For the task of PointGoal navigation ('Go to Δx, Δy’) under idealized settings (no RGB-D and actuation noise, perfect GPS+Compass), the answer is a clear ‘yes' - mapless neural models composed of task-agnostic components (CNNs and RNNs) trained with large-scale reinforcement learning achieve 100% Success on a standard dataset (Gibson [24] ). However, for PointNav in a realistic setting (RGB-D and actuation noise, no GPS+Compass), this is an open question; one we tackle in this paper. The strongest published result for this task is 71.7% Success [39]. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
According to Habitat Challenge 2020 PointNav benchmark held annually. A concurrent as-yet-unpublished result has reported 91% Success on 2021's benchmark, but we are unable to comment on the details because an associated report is not available.First, we identify the main (perhaps, only) cause of the drop in performance: absence of GPS+Compass. An agent with perfect GPS+Compass faced with RGB-D sensing and actuation noise achieves 99.8% Success (Gibson- v2 val). This suggests that (to paraphrase a meme) robust visual odometry is all we need for realistic PointNav; if we can achieve that, we can ignore the sensing and actuation noise. With that as our operating hypothesis, we scale dataset size, model size, and develop human-annotation-free dataaugmentation techniques to train neural models for visual odometry. We advance state of the art on the Habitat Realistic PointNav Challenge - SPL by 40% (relative), 53 to 74, and Success by 31% (relative), 71 to 94. While our approach does not saturate or ‘solve’ this dataset, this strong improvement combined with promising zero-shot sim2real transfer (to a LoCoBot robot) provides evidence consistent with the hypothesis that explicit mapping may not be necessary for navigation, even in a realistic setting.

----

## [1660] Symmetry-aware Neural Architecture for Embodied Visual Exploration

**Authors**: *Shuang Liu, Takayuki Okatani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01673](https://doi.org/10.1109/CVPR52688.2022.01673)

**Abstract**:

Visual exploration is a task that seeks to visit all the navigable areas of an environment as quickly as possible. The existing methods employ deep reinforcement learning (RL) as the standard tool for the task. However, they tend to be vulnerable to statistical shifts between the training and test data, resulting in poor generalization over novel environments that are out-of-distribution (OOD) from the training data. In this paper, we attempt to improve the generalization ability by utilizing the inductive biases available for the task. Employing the active neural SLAM (ANS) that learns exploration policies with the advantage actor-critic (A2C) method as the base framework, we first point out that the mappings represented by the actor and the critic should satisfy specific symmetries. We then propose a network design for the actor and the critic to inherently attain these symmetries. Specifically, we use G-convolution instead of the standard convolution and insert the semi-global polar pooling (SGPP) layer, which we newly design in this study, in the last section of the critic network. Experimental results show that our method increases area coverage by 8.1m
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 when trained on the Gibson dataset and tested on the Matterport3D dataset, establishing the new state-of-the-art.

----

## [1661] Coopernaut: End-to-End Driving with Cooperative Perception for Networked Vehicles

**Authors**: *Jiaxun Cui, Hang Qiu, Dian Chen, Peter Stone, Yuke Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01674](https://doi.org/10.1109/CVPR52688.2022.01674)

**Abstract**:

Optical sensors and learning algorithms for autonomous vehicles have dramatically advanced in the past few years. Nonetheless, the reliability of today's autonomous vehicles is hindered by the limited line-of-sight sensing capability and the brittleness of data-driven methods in handling extreme situations. With recent developments of telecommunication technologies, cooperative perception with vehicle-to-vehicle communications has become a promising paradigm to enhance autonomous driving in dangerous or emergency situations. We introduce Coopernaut,an end-to-end learning model that uses cross-vehicle perception for vision-based cooperative driving. Our model encodes Li-DAR information into compact point-based representations that can be transmitted as messages between vehicles via realistic wireless channels. To evaluate our model, we develop Autocastsim,a network-augmented driving simulation framework with example accident-prone scenarios. Our experiments on Autocastsim suggest that our cooperative perception driving models lead to a 40% improvement in average success rate over egocentric driving mod-els in these challenging driving situations and a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5\times$</tex>
 smaller bandwidth requirement than prior work V2VNet. Cooper-nautand Autocastsim are available at https://ut-austin-rpl.github.io/Coopernaut/.

----

## [1662] Topology Preserving Local Road Network Estimation from Single Onboard Camera Image

**Authors**: *Yigit Baran Can, Alexander Liniger, Danda Pani Paudel, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01675](https://doi.org/10.1109/CVPR52688.2022.01675)

**Abstract**:

Knowledge of the road network topology is crucial for autonomous planning and navigation. Yet, recovering such topology from a single image has only been explored in part. Furthermore, it needs to refer to the ground plane, where also the driving actions are taken. This paper aims at extracting the local road network topology, directly in the bird’ s-eye- view (BEV), all in a complex urban set-ting. The only input consists of a single onboard, for-ward looking camera image. We represent the road topology using a set of directed lane curves and their interactions, which are captured using their intersection points. To better capture topology, we introduce the concept of minimal cycles and their covers. A minimal cycle is the smallest cycle formed by the directed curve segments (be-tween two intersections). The cover is a set of curves whose segments are involved in forming a minimal cycle. We first show that the covers suffice to uniquely represent the road topology. The covers are then used to supervise deep neural networks, along with the lane curve supervision. These learn to predict the road topology from a single input image. The results on the NuScenes and Argo-verse benchmarks are significantly better than those ob-tained with baselines. Code: https://github.com/ybarancan/TopologicalLaneGraph.

----

## [1663] Coupling Vision and Proprioception for Navigation of Legged Robots

**Authors**: *Zipeng Fu, Ashish Kumar, Ananye Agarwal, Haozhi Qi, Jitendra Malik, Deepak Pathak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01676](https://doi.org/10.1109/CVPR52688.2022.01676)

**Abstract**:

We exploit the complementary strengths of vision and pro-prioception to develop a point-goal navigation system for legged robots, called VP-Nav. Legged systems are capable of traversing more complex terrain than wheeled robots, but to fully utilize this capability, we need a high-level path planner in the navigation system to be aware of the walking capabilities of the low-level locomotion policy in varying environments. We achieve this by using proprioceptive feedback to ensure the safety of the planned path by sensing unexpected obstacles like glass walls, terrain properties like slipperiness or softness of the ground and robot properties like extra payload that are likely missed by vision. The navigation system uses onboard cameras to generate an occupancy map and a corresponding cost map to reach the goal. A fast marching planner then generates a target path. A velocity command generator takes this as input to generate the desired velocity for the walking policy. A safety advisor module adds sensed unexpected obstacles to the occupancy map and environment-determined speed limits to the velocity command generator. We show superior performance compared to wheeled robot baselines, and ablation studies which have disjoint high-level planning and low-level control. We also show the real-world deployment of VP-Nav on a quadruped robot with onboard sensors and computation. Videos at https://navigation-locomotion.github.io

----

## [1664] Pyramid Architecture for Multi-Scale Processing in Point Cloud Segmentation

**Authors**: *Dong Nie, Rui Lan, Ling Wang, Xiaofeng Ren*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01677](https://doi.org/10.1109/CVPR52688.2022.01677)

**Abstract**:

Semantic segmentation of point cloud data is a critical task for autonomous driving and other applications. Recent advances of point cloud segmentation are mainly driven by new designs of local aggregation operators and point sampling methods. Unlike image segmentation, few efforts have been made to understand the fundamental issue of scale and how scales should interact and be fused. In this work, we investigate how to efficiently and effectively integrate features at varying scales and varying stages in a point cloud segmentation network. In particular, we open up the commonly used encoder-decoder architecture, and design scale pyramid architectures that allow information to flow more freely and systematically, both laterally and upward/downward in scale. Moreover, a cross-scale attention feature learning block has been designed to enhance the multi-scale feature fusion which occurs everywhere in the network. Such a design of multi-scale processing and fusion gains large improvements in accuracy without adding much additional computation. When built on top of the popular KPConv network, we see consistent improvements on a wide range of datasets, including achieving state-of-the-art performance on NPM3D and S3DIS. Moreover, the pyramid architecture is generic and can be applied to other network designs: we show an example of similar improvements over RandLANet.

----

## [1665] 3D-VField: Adversarial Augmentation of Point Clouds for Domain Generalization in 3D Object Detection

**Authors**: *Alexander Lehner, Stefano Gasperini, Alvaro Marcos-Ramiro, Michael Schmidt, Mohammad-Ali Nikouei Mahani, Nassir Navab, Benjamin Busam, Federico Tombari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01678](https://doi.org/10.1109/CVPR52688.2022.01678)

**Abstract**:

As 3D object detection on point clouds relies on the geometrical relationships between the points, non-standard object shapes can hinder a method's detection capability. However, in safety-critical settings, robustness to out-of-domain and long-tail samples is fundamental to circumvent dangerous issues, such as the misdetection of damaged or rare cars. In this work, we substantially improve the generalization of 3D object detectors to out-of-domain data by deforming point clouds during training. We achieve this with 3D-VField: a novel data augmentation method that plausibly deforms objects via vector fields learned in an adversarial fashion. Our approach constrains 3D points to slide along their sensor view rays while neither adding nor removing any of them. The obtained vectors are transferable, sample-independent and preserve shape and occlusions. Despite training only on a standard dataset, such as KITTI, augmenting with our vector fields significantly improves the generalization to differently shaped objects and scenes. Towards this end, we propose and share CrashD: a synthetic dataset of realistic damaged and rare cars, with a variety of crash scenarios. Extensive experiments on KITTI, Waymo, our CrashD and SUN RGB-D show the generalizability of our techniques to out-of-domain data, different models and sensors, namely LiDAR and ToF cameras, for both indoor and outdoor scenes. Our CrashD dataset is available at https://crashd-cars.github.io.

----

## [1666] Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior

**Authors**: *Davis Rempe, Jonah Philion, Leonidas J. Guibas, Sanja Fidler, Or Litany*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01679](https://doi.org/10.1109/CVPR52688.2022.01679)

**Abstract**:

Evaluating and improving planning for autonomous vehicles requires scalable generation of long-tail traffic scenarios. To be useful, these scenarios must be realistic and challenging, but not impossible to drive through safely. In this work, we introduce STRIVE, a method to automatically generate challenging scenarios that cause a given planner to produce undesirable behavior, like collisions. To maintain scenario plausibility, the key idea is to leverage a learned model of traffic motion in the form of a graph-based conditional VAE. Scenario generation is formulated as an optimization in the latent space of this traffic model, perturbing an initial real-world scene to produce trajectories that collide with a given planner. A subsequent optimization is used to find a “solution” to the scenario, ensuring it is useful to improve the given planner. Further analysis clusters generated scenarios based on collision type. We attack two planners and show that STRIVE successfully generates realistic, challenging scenarios in both cases. We additionally “close the loop” and use these scenarios to optimize hyperparameters of a rule-based planner.

----

## [1667] SelfD: Self-Learning Large-Scale Driving Policies From the Web

**Authors**: *Jimuyang Zhang, Ruizhao Zhu, Eshed Ohn-Bar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01680](https://doi.org/10.1109/CVPR52688.2022.01680)

**Abstract**:

Effectively utilizing the vast amounts of ego-centric navigation data that is freely available on the internet can advance generalized intelligent systems, i.e., to robustly scale across perspectives, platforms, environmental conditions, scenarios, and geographical locations. However, it is difficult to directly leverage such large amounts of unlabeled and highly diverse datafor complex 3D reasoning and planning tasks. Consequently, researchers have primarily focused on its use for various auxiliary pixel- and image-level computer vision tasks that do not consider an ultimate navigational objective. In this work, we introduce SelfD, a framework for learning scalable driving by utilizing large amounts of online monocular images. Our key idea is to leverage iterative semi-supervised training when learning imitative agents from unlabeled data. To handle unconstrained viewpoints, scenes, and camera parameters, we train an image-based model that directly learns to plan in the Bird's Eye View (BEV) space. Next, we use unla-beled data to augment the decision-making knowledge and robustness of an initially trained model via self-training. In particular, we propose a pseudo-labeling step which enables making full use of highly diverse demonstration data through “hypothetical” planning-based data augmentation. We employ a large dataset of publicly available YouTube videos to train SelfD and comprehensively analyze its generalization benefits across challenging navigation scenarios. Without requiring any additional data collection or annotation efforts, SelfD demonstrates consistent improvements (by up to 24%) in driving performance evaluation on nuScenes, Argoverse, Waymo, and CARLA.

----

## [1668] Towards real-world navigation with deep differentiable planners

**Authors**: *Shu Ishida, João F. Henriques*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01681](https://doi.org/10.1109/CVPR52688.2022.01681)

**Abstract**:

We train embodied neural networks to plan and navigate unseen complex 3D environments, emphasising real-world deployment. Rather than requiring prior knowledge of the agent or environment, the planner learns to model the state transitions and rewards. To avoid the potentially hazardous trial-and-error of reinforcement learning, we focus on differentiable planners such as Value Iteration Networks (VIN), which are trained offline from safe expert demonstrations. Although they work well in small simulations, we address two major limitations that hinder their deployment. First, we observed that current differentiable planners struggle to plan long-term in environments with a high branching complexity. While they should ideally learn to assign low rewards to obstacles to avoid collisions, these penalties are not strong enough to guarantee collision-free operation. We thus impose a structural constraint on the value iteration, which explicitly learns to model impossible actions and noisy motion. Secondly, we extend the model to plan exploration with a limited perspective camera under translation and fine rotations, which is crucial for real robot deployment. Our proposals significantly improve semantic navigation and exploration on several 2D and 3D environments, succeeding in settings that are otherwise challenging for differentiable planners. As far as we know, we are the first to successfully apply them to the difficult Active Vision Dataset, consisting of real images captured from a robot. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available: https://github.com/shuishida/calvin

----

## [1669] Privacy Preserving Partial Localization

**Authors**: *Marcel Geppert, Viktor Larsson, Johannes L. Schönberger, Marc Pollefeys*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01682](https://doi.org/10.1109/CVPR52688.2022.01682)

**Abstract**:

Recently proposed privacy preserving solutions for cloud-based localization rely on lifting traditional point-based maps to randomized 3D line clouds. While the lifted representation is effective in concealing private information, there are two fundamental limitations. First, without careful construction of the line clouds, the representation is vulnerable to density-based inversion attacks. Secondly, after successful localization, the precise camera orientation and position is revealed to the server. However, in many scenarios, the pose itself might be sensitive information. We propose a principled approach overcoming these limitations, based on two observations. First, a full 6 DoF pose is not always necessary, and in combination with egomotion tracking even a one dimensional localization can reduce uncertainty and correct drift. Secondly, by lifting to parallel planes instead of lines, the map only provides partial constraints on the query pose, preventing the server from knowing the exact query location. If the client requires a full 6 DoF pose, it can be obtained by fusing the result from multiple queries, which can be temporally and spatially disjoint. We demonstrate the practical feasibility of this approach and show a small performance drop compared to both the conventional and privacy preserving approaches.

----

## [1670] Efficient Large-scale Localization by Global Instance Recognition

**Authors**: *Fei Xue, Ignas Budvytis, Daniel Olmeda Reino, Roberto Cipolla*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01683](https://doi.org/10.1109/CVPR52688.2022.01683)

**Abstract**:

Hierarchical frameworks consisting of both coarse and fine localization are often used as the standard pipeline for large-scale visual localization. Despite their promising performance in simple environments, they still suffer from low efficiency and accuracy in large-scale scenes, especially under challenging conditions. In this paper, we propose an efficient and accurate large-scale localization framework based on the recognition of buildings, which are not only discriminative for coarse localization but also robust for fine localization. Specifically, we assign each building instance a global ID and perform pixel-wise recognition of these global instances in the localization process. For coarse localization, we employ an efficient reference search strategy to find candidates progressively from the local map observing recognized instances instead of the whole database. For fine localization, predicted labels are further used for instance-wise feature detection and matching, allowing our model to focus on fewer but more robust keypoints for establishing correspondences. The experiments in long-term large-scale localization datasets including Aachen and RobotCar-Seasons demonstrate that our method outperforms previous approaches consistently in terms of both efficiency and accuracy.

----

## [1671] CrossLoc: Scalable Aerial Localization Assisted by Multimodal Synthetic Data

**Authors**: *Qi Yan, Jianhao Zheng, Simon Reding, Shanci Li, Iordan Doytchinov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01684](https://doi.org/10.1109/CVPR52688.2022.01684)

**Abstract**:

We present a visual localization system that learns to estimate camera poses in the real world with the help of synthetic data. Despite significant progress in recent years, most learning-based approaches to visual localization target at a single domain and require a dense database of geo-tagged images to function well. To mitigate the data scarcity issue and improve the scalability of the neural localization models, we introduce TOPO-DataGen, a versatile synthetic data generation tool that traverses smoothly between the real and virtual world, hinged on the geographic camera viewpoint. New large-scale sim-to-real benchmark datasets are proposed to showcase and evaluate the utility of the said synthetic data. Our experiments reveal that synthetic data generically enhances the neural network performance on real data. Furthermore, we introduce CrossLoc, a cross-modal visual representation learning approach to pose estimation that makes full use of the scene coordinate ground truth via self-supervision. Without any extra data, CrossLoc significantly outperforms the state-of-the-art methods and achieves substantially higher real-data sample efficiency. Our code and datasets are all available at crossloc. github. io.

----

## [1672] Bilateral Video Magnification Filter

**Authors**: *Shoichiro Takeda, Kenta Niwa, Mariko Isogawa, Shinya Shimizu, Kazuki Okami, Yushi Aono*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01685](https://doi.org/10.1109/CVPR52688.2022.01685)

**Abstract**:

Eulerian video magnification (EVM) has progressed to magnify subtle motions with a target frequency even under the presence of large motions of objects. However, existing EVM methods often fail to produce desirable results in real videos due to (1) misextracting subtle motions with a non-target frequency and (2) collapsing results when large de/acceleration motions occur (e.g., objects suddenly start, stop, or change direction). To enhance EVM performance on real videos, this paper proposes a bilateral video magnification filter (BVMF) that offers simple yet robust temporal filtering. BVMF has two kernels; (I) one kernel performs temporal bandpass filtering via a Laplacian of Gaussian whose passband peaks at the target frequency with unity gain and (II) the other kernel excludes large motions outside the magnitude of interest by Gaussian filtering on the intensity of the input signal via the Fourier shift theorem. Thus, BVMF extracts only subtle motions with the target frequency while excluding large motions outside the magnitude of interest, regardless of motion dynamics. In addition, BVMF runs the two kernels in the temporal and intensity domains simultaneously like the bilateral filter does in the spatial and intensity domains. This simplifies implementation and, as a secondary effect, keeps the memory usage low. Experiments conducted on synthetic and real videos show that BVMF outperforms state-of-the-art methods.

----

## [1673] Neural Data-Dependent Transform for Learned Image Compression

**Authors**: *Dezhao Wang, Wenhan Yang, Yueyu Hu, Jiaying Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01686](https://doi.org/10.1109/CVPR52688.2022.01686)

**Abstract**:

Learned image compression has achieved great success due to its excellent modeling capacity, but seldom further considers the Rate-Distortion Optimization (RDO) of each input image. To explore this potential in the learned codec, we make the first attempt to build a neural data-dependent transform and introduce a continuous online mode decision mechanism to jointly optimize the coding efficiency for each individual image. Specifically, apart from the image content stream, we employ an additional model stream to generate the transform parameters at the decoder side. The pres-ence of a model stream enables our model to learn more abstract neural-syntax, which helps cluster the latent repre-sentations of images more compactly. Beyond the transform stage, we also adopt neural-syntax based post-processing for the scenarios that require higher quality reconstructions regardless of extra decoding overhead. Moreover, the in-volvement of the model stream further makes it possible to optimize both the representation and the decoder in an on-line way, i. e. RDO at the testing time. It is equivalent to a continuous online mode decision, like coding modes in the traditional codecs, to improve the coding efficiency based on the individual input image. The experimental results show the effectiveness of the proposed neural-syntax de-sign and the continuous online mode decision mechanism, demonstrating the superiority of our method in coding effi-ciency. Our project is available at: https://dezhao-wang.github.io/Neural-Syntax-Website/.

----

## [1674] Towards Bidirectional Arbitrary Image Rescaling: Joint Optimization and Cycle Idempotence

**Authors**: *Zhihong Pan, Baopu Li, Dongliang He, Mingde Yao, Wenhao Wu, Tianwei Lin, Xin Li, Errui Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01687](https://doi.org/10.1109/CVPR52688.2022.01687)

**Abstract**:

Deep learning based single image super-resolution models have been widely studied and superb results are achieved in upscaling low-resolution images with fixed scale factor and downscaling degradation kernel. To improve real world applicability of such models, there are growing interests to develop models optimized for arbitrary upscaling factors. Our proposed method is the first to treat arbitrary rescaling, both upscaling and downscaling, as one unified process. Using joint optimization of both directions, the proposed model is able to learn upscaling and downscaling simultaneously and achieve bidirectional arbitrary image rescaling. It improves the performance of current arbitrary upscaling models by a large margin while at the same time learns to maintain visual perception quality in downscaled images. The proposed model is further shown to be robust in cycle idempotence test, free of severe degradations in reconstruction accuracy when the downscaling-to-upscaling cycle is applied repetitively. This robustness is beneficial for image rescaling in the wild when this cycle could be applied to one image for multiple times. It also performs well on tests with arbitrary large scales and asymmetric scales, even when the model is not trained with such tasks. Extensive experiments are conducted to demonstrate the superior performance of our model.

----

## [1675] Deep Generalized Unfolding Networks for Image Restoration

**Authors**: *Chong Mou, Qian Wang, Jian Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01688](https://doi.org/10.1109/CVPR52688.2022.01688)

**Abstract**:

Deep neural networks (DNN) have achieved great suc-cess in image restoration. However, most DNN methods are designed as a black box, lacking transparency and inter-pretability. Although some methods are proposed to combine traditional optimization algorithms with DNN, they usually demand pre-defined degradation processes or hand-crafted assumptions, making it difficult to deal with complex and real-world applications. In this paper, we propose a Deep Generalized Unfolding Network (DGUNet) for image restoration. Concretely, without loss of interpretability, we integrate a gradient estimation strategy into the gradi-ent descent step of the Proximal Gradient Descent (PGD) algorithm, driving it to deal with complex and real-world image degradation. In addition, we design inter-stage in-formation pathways across proximal mapping in different PGD iterations to rectify the intrinsic information loss in most deep unfolding networks (DUN) through a multi-scale and spatial-adaptive way. By integrating the flexible gradi-ent descent and informative proximal mapping, we unfold the iterative PGD algorithm into a trainable DNN. Exten-sive experiments on various image restoration tasks demon-strate the superiority of our method in terms of state-of-the-art performance, interpretability, and generalizability. The source code is available at github.com/MC-E/DGUNet.

----

## [1676] Look Back and Forth: Video Super-Resolution with Explicit Temporal Difference Modeling

**Authors**: *Takashi Isobe, Xu Jia, Xin Tao, Changlin Li, Ruihuang Li, Yongjie Shi, Jing Mu, Huchuan Lu, Yu-Wing Tai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01689](https://doi.org/10.1109/CVPR52688.2022.01689)

**Abstract**:

Temporal modeling is crucial for video super-resolution. Most of the video super-resolution methods adopt the optical flow or deformable convolution for explicitly motion compensation. However, such temporal modeling techniques increase the model complexity and might fail in case of occlusion or complex motion, resulting in serious distortion and artifacts. In this paper, we propose to explore the role of explicit temporal difference modeling in both LR and HR space. Instead of directly feeding consecutive frames into a VSR model, we propose to compute the temporal difference between frames and divide those pixels into two subsets according to the level of difference. They are separately processed with two branches of different receptive fields in order to better extract complementary information. To further enhance the super-resolution result, not only spatial residual features are extracted, but the difference between consecutive frames in high-frequency domain is also computed. It allows the model to exploit intermediate SR results in both future and past to refine the current SR output. The difference at different time steps could be cached such that information from further distance in time could be propagated to the current frame for refinement. Experiments on several video super-resolution benchmark datasets demonstrate the effectiveness of the proposed method and its favorable performance against state-of-the-art methods.

----

## [1677] XYDeblur: Divide and Conquer for Single Image Deblurring

**Authors**: *Seo-Won Ji, Jeongmin Lee, Seung-Wook Kim, Jun-Pyo Hong, Seung-Jin Baek, Seung-Won Jung, Sung-Jea Ko*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01690](https://doi.org/10.1109/CVPR52688.2022.01690)

**Abstract**:

Many convolutional neural networks (CNNs) for single image deblurring employ a U-Net structure to estimate latent sharp images. Having long been proven to be effective in image restoration tasks, a single lane of encoder-decoder architecture overlooks the characteristic of deblurring, where a blurry image is generated from complicated blur kernels caused by tangled motions. Toward an effective network architecture for single image deblurring, we present complemental sub-solution learning with a one-encoder-two-decoder architecture. Observing that multiple decoders successfully learn to decompose encoded feature information into directional components, we further improve both the network efficiency and the deblurring performance by rotating and sharing kernels exploited in the decoders, which prevents the decoders from separating unnecessary components such as color shift. As a result, our proposed network shows superior results compared to U-Net while preserving the network parameters, and using the proposed network as the base network can improve the performance of existing state-of-the-art deblurring networks.

----

## [1678] Abandoning the Bayer-Filter to See in the Dark

**Authors**: *Xingbo Dong, Wanyan Xu, Zhihui Miao, Lan Ma, Chao Zhang, Jiewen Yang, Zhe Jin, Andrew Beng Jin Teoh, Jiajun Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01691](https://doi.org/10.1109/CVPR52688.2022.01691)

**Abstract**:

Low-light image enhancement, a pervasive but challenging problem, plays a central role in enhancing the visibility of an image captured in a poor illumination environment. Due to the fact that not all photons can pass the Bayer-Filter on the sensor of the color camera, in this work, we first present a De-Bayer-Filter simulator based on deep neural networks to generate a monochrome raw image from the colored raw image. Next, a fully convolutional network is proposed to achieve the low-light image enhancement by fusing colored raw data with synthesized monochrome data. Channel-wise attention is also introduced to the fusion process to establish a complementary interaction between features from colored and monochrome raw images. To train the convolutional networks, we propose a dataset with monochrome and color raw pairs named Mono-Colored Raw paired dataset (MCR) collected by using a monochrome camera without Bayer-Filter and a color camera with Bayer-Filter. The proposed pipeline takes advantages of the fusion of the virtual monochrome and the color raw images, and our extensive experiments indicate that significant improvement can be achieved by leveraging raw sensor data and data-driven learning. The project is available at https://github.com/TCL-AILab/Abandon_Bayer-Filter_See_in_the_Dark.

----

## [1679] RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution

**Authors**: *Zhicheng Geng, Luming Liang, Tianyu Ding, Ilya Zharkov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01692](https://doi.org/10.1109/CVPR52688.2022.01692)

**Abstract**:

Space-time video super-resolution (STVSR) is the task of interpolating videos with both Low Frame Rate (LFR) and Low Resolution (LR) to produce High-Frame-Rate (HFR) and also High-Resolution (HR) counterparts. The existing methods based on Convolutional Neural Network (CNN) succeed in achieving visually satisfied results while suffer from slow inference speed due to their heavy architec-tures. We propose to resolve this issue by using a spatial-temporal transformer that naturally incorporates the spa-tial and temporal super resolution modules into a single model. Unlike CNN-based methods, we do not explic-itly use separated building blocks for temporal interpolations and spatial super-resolutions; instead, we only use a single end-to-end transformer architecture. Specifically, a reusable dictionary is built by encoders based on the in-put LFR and LR frames, which is then utilized in the de-coder part to synthesize the HFR and HR frames. compared with the state-of-the-art TMNet [54], our network is 60% smaller (4.5M vs 12.3M parameters) and 80% faster (26.2fps vs 14.3fps on 720 x 576 frames) without sacri-ficing much performance. The source code is available at https://github.com/llmpass/RSTT.

----

## [1680] All-In-One Image Restoration for Unknown Corruption

**Authors**: *Boyun Li, Xiao Liu, Peng Hu, Zhongqin Wu, Jiancheng Lv, Xi Peng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01693](https://doi.org/10.1109/CVPR52688.2022.01693)

**Abstract**:

In this paper, we study a challenging problem in image restoration, namely, how to develop an all-in-one method that could recover images from a variety of unknown corruption types and levels. To this end, we propose an All-in-one Image Restoration Network (AirNet) consisting of two neural modules, named Contrastive-Based Degraded Encoder (CBDE) and Degradation-Guided Restoration Network (DGRN). The major advantages of AirNet are two-fold. First, it is an all-in-one solution which could recover various degraded images in one network. Second, AirNet is free from the prior of the corruption types and levels, which just uses the observed corrupted image to perform inference. These two advantages enable AirNet to enjoy better flexibility and higher economy in real world scenarios wherein the priors on the corruptions are hard to know and the degradation will change with space and time. Extensive experimental results show the proposed method outperforms 17 image restoration baselines on four challenging datasets. The code is available at https://github.com/XLearning-SCU/2022-CVPR-AirNet.

----

## [1681] Modeling sRGB Camera Noise with Normalizing Flows

**Authors**: *Shayan Kousha, Ali Maleky, Michael S. Brown, Marcus A. Brubaker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01694](https://doi.org/10.1109/CVPR52688.2022.01694)

**Abstract**:

Noise modeling and reduction are fundamental tasks in low-level computer vision. They are particularly important for smartphone cameras relying on small sensors that exhibit visually noticeable noise. There has recently been renewed interest in using data-driven approaches to improve camera noise models via neural networks. These data-driven approaches target noise present in the raw-sensor image before it has been processed by the camera's image signal processor (ISP). Modeling noise in the RAW-rgb domain is useful for improving and testing the in-camera denoising algorithm; however, there are situations where the camera's ISP does not apply denoising or additional denoising is desired when the RAW-rgb domain image is no longer available. In such cases, the sensor noise propagates through the ISP to the final rendered image encoded in standard RGB (sRGB). The nonlinear steps on the ISP culminate in a significantly more complex noise distribution in the sRGB domain and existing raw-domain noise models are unable to capture the sRGB noise distribution. We propose a new sRGB-domain noise model based on normalizing flows that is capable of learning the complex noise distribution found in sRGB images under various ISO levels. Our normalizing flows-based approach outperforms other models by a large margin in noise modeling and synthesis tasks. We also show that image denoisers trained on noisy images synthesized with our noise model outperforms those trained with noise from baselines models.

----

## [1682] A Differentiable Two-stage Alignment Scheme for Burst Image Reconstruction with Large Shift

**Authors**: *Shi Guo, Xi Yang, Jianqi Ma, Gaofeng Ren, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01695](https://doi.org/10.1109/CVPR52688.2022.01695)

**Abstract**:

Denoising and demosaicking are two essential steps to reconstruct a clean full-color image from the raw data. Recently, joint denoising and demosaicking (JDD) for burst images, namely JDD-B, has attracted much attention by using multiple raw images captured in a short time to reconstruct a single high-quality image. One key challenge of JDD-B lies in the robust alignment of image frames. State-of-the-art alignment methods in feature domain cannot effectively utilize the temporal information of burst images, where large shifts commonly exist due to camera and object motion. In addition, the higher resolution (e.g., 4K) of modern imaging devices results in larger displacement between frames. To address these challenges, we design a differentiable two-stage alignment scheme sequentially in patch and pixel level for effective JDD-B. The input burst images are firstly aligned in the patch level by using a differentiable progressive block matching method, which can estimate the offset between distant frames with small computational cost. Then we perform implicit pixel-wise alignment in full-resolution feature domain to refine the alignment results. The two stages are jointly trained in an end-to-end manner. Extensive experiments demonstrate the significant improvement of our method over existing JDD-B methods. Codes are available at https://github.com/GuoShi28/2StageAlign.

----

## [1683] Video Frame Interpolation Transformer

**Authors**: *Zhihao Shi, Xiangyu Xu, Xiaohong Liu, Jun Chen, Ming-Hsuan Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01696](https://doi.org/10.1109/CVPR52688.2022.01696)

**Abstract**:

Existing methods for video interpolation heavily rely on deep convolution neural networks, and thus suffer from their intrinsic limitations, such as content-agnostic kernel weights and restricted receptive field. To address these issues, we propose a Transformer-based video interpolation framework that allows content-aware aggregation weights and considers long-range dependencies with the self-attention operations. To avoid the high computational cost of global self-attention, we introduce the concept of local attention into video interpolation and extend it to the spatial-temporal domain. Furthermore, we propose a space-time separation strategy to save memory usage, which also improves performance. In addition, we develop a multi-scale frame synthesis scheme to fully realize the potential of Transformers. Extensive experiments demonstrate the proposed model performs favorably against the state-of-the-art methods both quantitatively and qualitatively on a variety of benchmark datasets. The code and models are released at https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer.

----

## [1684] The Devil Is in the Details: Window-based Attention for Image Compression

**Authors**: *Renjie Zou, Chunfeng Song, Zhaoxiang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01697](https://doi.org/10.1109/CVPR52688.2022.01697)

**Abstract**:

Learned image compression methods have exhibited superior rate-distortion performance than classical image compression standards. Most existing learned image compression models are based on Convolutional Neural Networks (CNNs). Despite great contributions, a main drawback of CNN based model is that its structure is not designed for capturing local redundancy, especially the nonrepetitive textures, which severely affects the reconstruction quality. Therefore, how to make full use of both global structure and local texture becomes the core problem for learning-based image compression. Inspired by recent progresses of Vision Transformer (ViT) and Swin Transformer, we found that combining the local-aware attention mechanism with the global-related feature learning could meet the expectation in image compression. In this paper, we first extensively study the effects of multiple kinds of attention mechanisms for local features learning, then introduce a more straightforward yet effective window-based local attention block. The proposed window-based attention is very flexible which could work as a plug-and-play component to enhance CNN and Transformer models. Moreover, we propose a novel Symmetrical TransFormer (STF) framework with absolute transformer blocks in the down-sampling encoder and up-sampling decoder. Extensive experimental evaluations have shown that the proposed method is effective and outperforms the state-of-the-art methods. The code is publicly available at https://github.com/Googolxx/STF.

----

## [1685] Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction

**Authors**: *Yuanhao Cai, Jing Lin, Xiaowan Hu, Haoqian Wang, Xin Yuan, Yulun Zhang, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01698](https://doi.org/10.1109/CVPR52688.2022.01698)

**Abstract**:

Hyperspectral image (HSI) reconstruction aims to recover the 3D spatial-spectral signal from a 2D measurement in the coded aperture snapshot spectral imaging (CASSI) system. The HSI representations are highly similar and correlated across the spectral dimension. Modeling the inter-spectra interactions is beneficial for HSI reconstruction. However, existing CNN-based methods show limitations in capturing spectral-wise similarity and long-range dependencies. Besides, the HSI information is modulated by a coded aperture (physical mask) in CASSI. Nonetheless, current algorithms have not fully explored the guidance effect of the mask for HSI restoration. In this paper, we propose a novel framework, Mask-guided Spectral-wise Transformer (MST), for HSI reconstruction. Specifically, we present a Spectral-wise Multi-head Self-Attention (S-MSA) that treats each spectral feature as a token and calculates self-attention along the spectral dimension. In addition, we customize a Mask-guided Mechanism (MM) that directs S- MSA to pay attention to spatial regions with high-fidelity spectral representations. Extensive experiments show that our MST significantly outperforms state-of-the-art (SOTA) methods on simulation and real HSI datasets while requiring dramatically cheaper computational and memory costs. https://github.com/caiyuanhao1998/MST/

----

## [1686] RestoreFormer: High-Quality Blind Face Restoration from Undegraded Key-Value Pairs

**Authors**: *Zhouxia Wang, Jiawei Zhang, Runjian Chen, Wenping Wang, Ping Luo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01699](https://doi.org/10.1109/CVPR52688.2022.01699)

**Abstract**:

Blind face restoration is to recover a high-quality face image from unknown degradations. As face image contains abundant contextual information, we propose a method, RestoreFormer, which explores fully-spatial attentions to model contextual information and surpasses existing works that use local operators. RestoreFormer has several benefits compared to prior arts. First, unlike the conventional multi-head self-attention in previous Vision Transformers (ViTs), RestoreFormer incorporates a multi-head cross-attention layer to learn fully-spatial interactions between corrupted queries and high-quality key-value pairs. Second, the key-value pairs in ResotreFormer are sampled from a reconstruction-oriented high-quality dictionary, whose elements are rich in high-quality facial features specifically aimed for face reconstruction, leading to superior restoration results. Third, RestoreFormer outperforms advanced state-of-the-art methods on one synthetic dataset and three real-world datasets, as well as produces images with better visual quality. Code is available at https://github.com/wzhouxiff/RestoreFormer.git.

----

## [1687] AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement

**Authors**: *Canqian Yang, Meiguang Jin, Xu Jia, Yi Xu, Ying Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01700](https://doi.org/10.1109/CVPR52688.2022.01700)

**Abstract**:

The 3D Lookup Table (3D LUT) is a highly-efficient tool for real-time image enhancement tasks, which models a non-linear 3D color transform by sparsely sampling it into a discretized 3D lattice. Previous works have made efforts to learn image-adaptive output color values of LUTs for flexible enhancement but neglect the importance of sampling strategy. They adopt a sub-optimal uniform sampling point allocation, limiting the expressiveness of the learned LUTs since the (tri-)linear interpolation between uniform sampling points in the LUT transform might fail to model local non-linearities of the color transform. Focusing on this problem, we present AdaInt (Adaptive Intervals Learning), a novel mechanism to achieve a more flexible sampling point allocation by adaptively learning the non-uniform sampling intervals in the 3D color space. In this way, a 3D LUT can increase its capability by conducting dense sampling in color ranges requiring highly non-linear transforms and sparse sampling for near-linear transforms. The proposed AdaInt could be implemented as a compact and efficient plug-and-play module for a 3D LUT-based method. To enable the end-to-end learning of AdaInt, we design a novel differentiable operator called AiLUT-Transform (Adaptive Interval LUT Transform) to locate input colors in the non-uniform 3D LUT and provide gradients to the sampling intervals. Experiments demonstrate that methods equipped with AdaInt can achieve state-of-the-art performance on two public benchmark datasets with a negligible overhead increase. Our source code is available at https://github.com/ImCharlesY/AdaInt.

----

## [1688] HerosNet: Hyperspectral Explicable Reconstruction and Optimal Sampling Deep Network for Snapshot Compressive Imaging

**Authors**: *Xuanyu Zhang, Yongbing Zhang, Ruiqin Xiong, Qilin Sun, Jian Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01701](https://doi.org/10.1109/CVPR52688.2022.01701)

**Abstract**:

Hyperspectral imaging is an essential imaging modality for a wide range of applications, especially in remote sensing, agriculture, and medicine. Inspired by existing hyperspectral cameras that are either slow, expensive, or bulky, reconstructing hyperspectral images (HSIs) from a low-budget snapshot measurement has drawn wide attention. By mapping a truncated numerical optimization algorithm into a network with a fixed number of phases, recent deep unfolding networks (DUNs) for spectral snapshot compressive sensing (SCI) have achieved remarkable success. However, DUNs are far from reaching the scope of industrial applications limited by the lack of cross-phase feature interaction and adaptive parameter adjustment. In this paper, we propose a novel Hyperspectral Explicable Reconstruction and Optimal Sampling deep Network for SCI, dubbed HerosNet, which includes several phases under the ISTA-unfolding framework. Each phase can flexibly simulate the sensing matrix and contextually adjust the step size in the gradient descent step, and hierarchically fuse and interact the hidden states of previous phases to effectively recover current HSI frames in the proximal mapping step. Simultaneously, a hardware-friendly optimal binary mask is learned end-to-end to further improve the reconstruction performance. Finally, our HerosNet is validated to outperform the state-of-the-art methods on both simulation and real datasets by large margins. The source code is available at https://github.com/jianzhangcs/HerosNet.

----

## [1689] HDNet: High-resolution Dual-domain Learning for Spectral Compressive Imaging

**Authors**: *Xiaowan Hu, Yuanhao Cai, Jing Lin, Haoqian Wang, Xin Yuan, Yulun Zhang, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01702](https://doi.org/10.1109/CVPR52688.2022.01702)

**Abstract**:

The rapid development of deep learning provides a better solution for the end-to-end reconstruction of hyperspectral image (HSI). However, existing learning-based methods have two major defects. Firstly, networks with self-attention usually sacrifice internal resolution to balance model performance against complexity, losing fine-grained high-resolution (HR) features. Secondly, even if the optimization focusing on spatial-spectral domain learning (SDL) converges to the ideal solution, there is still a significant visual difference between the reconstructed HSI and the truth. So we propose a high-resolution dual-domain learning network (HDNet) for HSI reconstruction. On the one hand, the proposed HR spatial-spectral attention module with its efficient feature fusion provides continuous and fine pixel-level features. On the other hand, frequency domain learning (FDL) is introduced for HSI reconstruction to narrow the frequency domain discrepancy. Dynamic FDL supervision forces the model to reconstruct fine-grained frequencies and compensate for excessive smoothing and distortion caused by pixel-level losses. The HR pixel-level attention and frequency-level refinement in our HDNet mutually promote HSI perceptual quality. Extensive quantitative and qualitative experiments show that our method achieves SOTA performance on simulated and real HSI datasets. https://github.com/Huxiaowan/HDNet

----

## [1690] Learning to Zoom Inside Camera Imaging Pipeline

**Authors**: *Chengzhou Tang, Yuqiang Yang, Bing Zeng, Ping Tan, Shuaicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01703](https://doi.org/10.1109/CVPR52688.2022.01703)

**Abstract**:

Existing single image super-resolution methods are either designed for synthetic data, or for real data but in the RGB-to-RGB or the RAW-to-RGB domain. This paper proposes to zoom an image from RAW to RAW inside the camera imaging pipeline. The RAW-to-RAW domain closes the gap between the ideal and the real degradation models. It also excludes the image signal processing pipeline, which refocuses the model learning onto the super-resolution. To these ends, we design a method that receives a low-resolution RAW as the input and estimates the desired higher-resolution RAW jointly with the degradation model. In our method, two convolutional neural networks are learned to constrain the high-resolution image and the degradation model in lower-dimensional subspaces. This subspace constraint converts the ill-posed SISR problem to a well-posed one. To demonstrate the superiority of the proposed method and the RAW-to-RAW domain, we conduct evaluations on the RealSR and the SR-RAW datasets. The results show that our method performs superiorly over the state-of-the-arts both qualitatively and quantitatively, and it also generalizes well and enables zero-shot transfer across different sensors.

----

## [1691] Towards An End-to-End Framework for Flow-Guided Video Inpainting

**Authors**: *Zhen Li, Chengze Lu, Jianhua Qin, Chun-Le Guo, Ming-Ming Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01704](https://doi.org/10.1109/CVPR52688.2022.01704)

**Abstract**:

Optical flow, which captures motion information across frames, is exploited in recent video inpainting methods through propagating pixels along its trajectories. However, the hand-crafted flow-based processes in these methods are applied separately to form the whole inpainting pipeline. Thus, these methods are less efficient and rely heavily on the intermediate results from earlier stages. In this paper, we propose an End-to-End framework for Flow-Guided Video Inpainting (E
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 FGVI) through elaborately designed three trainable modules, namely, flow completion, feature propagation, and content hallucination modules. The three modules correspond with the three stages of previous flow-based methods but can be Jointly optimized, leading to a more efficient and effective inpainting process. Experimental results demonstrate that the proposed method outperforms state-of-the-art methods both qualitatively and quantitatively and shows promising efficiency. The code is available at https://github.com/MCG-NKU/E2FGVI.

----

## [1692] Context-Aware Video Reconstruction for Rolling Shutter Cameras

**Authors**: *Bin Fan, Yuchao Dai, Zhiyuan Zhang, Qi Liu, Mingyi He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01705](https://doi.org/10.1109/CVPR52688.2022.01705)

**Abstract**:

With the ubiquity of rolling shutter (RS) cameras, it is becoming increasingly attractive to recover the latent global shutter (GS) video from two consecutive RS frames, which also places a higher demand on realism. Existing solutions, using deep neural networks or optimization, achieve promising performance. However, these methods generate intermediate GS frames through image warping based on the RS model, which inevitably result in black holes and noticeable motion artifacts. In this paper, we alleviate these issues by proposing a context-aware GS video reconstruction architecture. It facilitates the advantages such as occlusion reasoning, motion compensation, and temporal abstraction. Specifically, we first estimate the bilateral motion field so that the pixels of the two RS frames are warped to a common GS frame accordingly. Then, a refinement scheme is proposed to guide the GS frame synthesis along with bilateral occlusion masks to produce high-fidelity GS video frames at arbitrary times. Furthermore, we derive an approximated bilateral motion field model, which can serve as an alternative to provide a simple but effective GS frame initialization for related tasks. Experiments on synthetic and real data show that our approach achieves superior performance over state-of-the-art methods in terms of objective metrics and subjective visual quality. Code is available at https://github.com/GitCVfb/CVR.

----

## [1693] CVF-SID: Cyclic multi-Variate Function for Self-Supervised Image Denoising by Disentangling Noise from Image

**Authors**: *Reyhaneh Neshatavar, Mohsen Yavartanoo, Sanghyun Son, Kyoung Mu Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01706](https://doi.org/10.1109/CVPR52688.2022.01706)

**Abstract**:

Recently, significant progress has been made on image denoising with strong supervision from large-scale datasets. However, obtaining well-aligned noisy-clean training image pairs for each specific scenario is complicated and costly in practice. Consequently, applying a conventional supervised denoising network on in-the-wild noisy inputs is not straightforward. Although several studies have challenged this problem without strong supervision, they rely on less practical assumptions and cannot be applied to practical situations directly. To address the aforementioned challenges, we propose a novel and powerful self-supervised denoising method called CVF-SID based on a Cyclic multi-Variate Function (CVF) module and a self-supervised image disentangling (SID) framework. The CVF module can output multiple decomposed variables of the input and take a combination of the outputs back as an input in a cyclic manner. Our CVF-SID can disentangle a clean image and noise maps from the input by leveraging various self-supervised loss terms. Unlike several methods that only consider the signal-independent noise models, we also deal with signal-dependent noise components for real-world applications. Furthermore, we do not rely on any prior assumptions about the underlying noise distribution, making CVF-SID more generalizable toward realistic noise. Extensive experiments on real-world datasets show that CVF-SID achieves state-of-the-art self-supervised image denoising performance and is comparable to other existing approaches. The code is publicly available from this link.

----

## [1694] Global Matching with Overlapping Attention for Optical Flow Estimation

**Authors**: *Shiyu Zhao, Long Zhao, Zhixing Zhang, Enyu Zhou, Dimitris N. Metaxas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01707](https://doi.org/10.1109/CVPR52688.2022.01707)

**Abstract**:

Optical flow estimation is a fundamental task in computer vision. Recent direct-regression methods using deep neural networks achieve remarkable performance improvement. However, they do not explicitly capture long-term motion correspondences and thus cannot handle large motions effectively. In this paper, inspired by the traditional matching-optimization methods where matching is introduced to handle large displacements before energy-based optimizations, we introduce a simple but effective global matching step before the direct regression and develop a learning-based matching-optimization framework, namely GMFlowNet. In GMFlowNet, global matching is efficiently calculated by applying argmax on 4D cost volumes. Additionally, to improve the matching quality, we propose patch-based overlapping attention to extract large context features. Extensive experiments demonstrate that GM-FlowNet outperforms RAFT, the most popular optimization-only method, by a large margin and achieves state-of-the-art performance on standard benchmarks. Thanks to the matching and overlapping attention, GMFlowNet obtains major improvements on the predictions for textureless regions and large motions. Our code is made publicly available at https://github.com/xiaofeng94/GMFlowNet.

----

## [1695] CRAFT: Cross-Attentional Flow Transformer for Robust Optical Flow

**Authors**: *Xiuchao Sui, Shaohua Li, Xue Geng, Yan Wu, Xinxing Xu, Yong Liu, Rick Siow Mong Goh, Hongyuan Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01708](https://doi.org/10.1109/CVPR52688.2022.01708)

**Abstract**:

Optical flow estimation aims to find the 2D motion field by identifying corresponding pixels between two images. Despite the tremendous progress of deep learning-based optical flow methods, it remains a challenge to accurately estimate large displacements with motion blur. This is mainly because the correlation volume, the basis of pixel matching, is computed as the dot product of the convolutional features of the two images. The locality of convolutional features makes the computed correlations susceptible to various noises. On large displacements with motion blur, noisy correlations could cause severe errors in the estimated flow. To overcome this challenge, we propose a new architecture “CRoss-Attentional Flow Trans-former” (CRAFT), aiming to revitalize the correlation volume computation. In CRAFT, a Semantic Smoothing Trans-former layer transforms the features of one frame, making them more global and semantically stable. In addition, the dot-product correlations are replaced with trans-former Cross-Frame Attention. This layer filters out feature noises through the Query and Key projections, and computes more accurate correlations. On Sintel (Final) and KITTI (foreground) benchmarks, CRAFT has achieved new state-of-the-art performance. Moreover, to test the robust-ness of different models on large motions, we designed an image shifting attack that shifts input images to generate large artificial motions. Under this attack, CRAFT per-forms much more robustly than two representative meth-ods, RAFT and GMA. The code of CRAFT is is available at https://github.com/askerlee/craft.

----

## [1696] Unified Multivariate Gaussian Mixture for Efficient Neural Image Compression

**Authors**: *Xiaosu Zhu, Jingkuan Song, Lianli Gao, Feng Zheng, Heng Tao Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01709](https://doi.org/10.1109/CVPR52688.2022.01709)

**Abstract**:

Modeling latent variables with priors and hyperpriors is an essential problem in variational image compression. Formally, trade-off between rate and distortion is handled well if priors and hyperpriors precisely describe latent variables. Current practices only adopt univariate priors and process each variable individually. However, we find inter-correlations and intra-correlations exist when observing latent variables in a vectorized perspective. These findings reveal visual redundancies to improve rate-distortion performance and parallel processing ability to speed up compression. This encourages us to propose a novel vectorized prior. Specifically, a multivariate Gaussian mixture is proposed with means and covariances to be estimated. Then, a novel probabilistic vector quantization is utilized to effectively approximate means, and remaining covariances are further induced to a unified mixture and solved by cascaded estimation without context models involved. Furthermore, code books involved in quantization are extended to multi-codebooks for complexity reduction, which formulates an efficient compression procedure. Extensive experiments on benchmark datasets against state-of-the-art indicate our model has better rate-distortion performance and an impressive 3.18x compression speed up, giving us the ability to perform real-time, high-quality variational image compression in practice. Our source code is publicly available at https://github.com/xiaosu-zhu/McQuic.

----

## [1697] Video Demoiréing with Relation-Based Temporal Consistency

**Authors**: *Peng Dai, Xin Yu, Lan Ma, Baoheng Zhang, Jia Li, Wenbo Li, Jiajun Shen, Xiaojuan Qi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01710](https://doi.org/10.1109/CVPR52688.2022.01710)

**Abstract**:

Moiré patterns, appearing as color distortions, severely degrade image and video qualities when filming a screen with digital cameras. Considering the increasing demands for capturing videos, we study how to remove such undesirable moiré patterns in videos, namely video demoiréing. To this end, we introduce the first hand-held video demoiréing dataset with a dedicated data collection pipeline to ensure spatial and temporal alignments of captured data. Further, a baseline video demoiréing model with implicit feature space alignment and selective feature aggregation is developed to leverage complementary information from nearby frames to improve frame-level video demoiréing. More importantly, we propose a relation-based temporal consistency loss to encourage the model to learn temporal consistency priors directly from ground-truth reference videos, which facilitates producing temporally consistent predictions and effectively maintains frame-level qualities. Extensive experiments manifest the superiority of our model. Code is available at ht tps:// daipengwa. github.io/VDmoire_ProjectPage/.

----

## [1698] Noise2NoiseFlow: Realistic Camera Noise Modeling without Clean Images

**Authors**: *Ali Maleky, Shayan Kousha, Michael S. Brown, Marcus A. Brubaker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01711](https://doi.org/10.1109/CVPR52688.2022.01711)

**Abstract**:

Image noise modeling is a long-standing problem with many applications in computer vision. Early attempts that propose simple models, such as signal-independent additive white Gaussian noise or the heteroscedastic Gaussian noise model (a.k.a., camera noise level function) are not sufficient to learn the complex behavior of the camera sensor noise. Recently, more complex learning-based models have been proposed that yield better results in noise synthesis and downstream tasks, such as denoising. However, their dependence on supervised data (i.e., paired clean images) is a limiting factor given the challenges in producing ground-truth images. This paper proposes a framework for training a noise model and a denoiser simultaneously while relying only on pairs of noisy images rather than noisy/clean paired image data. We apply this framework to the training of the Noise Flow architecture. The noise synthesis and density estimation results show that our framework outperforms previous signal-processing-based noise models and is on par with its supervised counterpart. The trained denoiser is also shown to significantly improve upon both supervised and weakly supervised baseline denoising approaches. The results indicate that the joint training of a denoiser and a noise model yields significant improvements in the denoiser.

----

## [1699] Deep Constrained Least Squares for Blind Image Super-Resolution

**Authors**: *Ziwei Luo, Haibin Huang, Lei Yu, Youwei Li, Haoqiang Fan, Shuaicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01712](https://doi.org/10.1109/CVPR52688.2022.01712)

**Abstract**:

In this paper, we tackle the problem of blind image super-resolution(SR) with a reformulated degradation model and two novel modules. Following the common practices of blind SR, our method proposes to improve both the kernel estimation as well as the kernel based high resolution image restoration. To be more specific, we first reformulate the degradation model such that the deblurring kernel estimation can be transferred into the low resolution space. On top of this, we introduce a dynamic deep linear filter module. Instead of learning a fixed kernel for all images, it can adaptively generate deblurring kernel weights conditional on the input and yields more robust kernel estimation. Subsequently, a deep constrained least square filtering module is applied to generate clean features based on the reformulation and estimated kernel. The deblurred feature and the low input image feature are then fed into a dual-path structured SR network and restore the final high resolution result. To evaluate our method, we further conduct evaluations on several benchmarks, including Gaussian8 and DIV2KRK. Our experiments demonstrate that the proposed method achieves better accuracy and visual improvements against state-of-the-art methods. Codes and models are available at https://github.com/megvii-research/DCLS-SR.

----

## [1700] Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization: Toward a Unified Model

**Authors**: *Wei-Ting Chen, Zhi-Kai Huang, Cheng-Che Tsai, Hao-Hsiang Yang, Jian-Jiun Ding, Sy-Yen Kuo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01713](https://doi.org/10.1109/CVPR52688.2022.01713)

**Abstract**:

In this paper, an ill-posed problem of multiple adverse weather removal is investigated. Our goal is to train a model with a ‘unified’ architecture and only one set of pretrained weights that can tackle multiple types of adverse weathers such as haze, snow, and rain simultaneously. To this end, a two-stage knowledge learning mechanism including knowledge collation (KC) and knowledge examination (KE) based on a multi-teacher and student architecture is proposed. At the KC, the student network aims to learn the comprehensive bad weather removal problem from multiple well-trained teacher networks where each of them is specialized in a specific bad weather removal problem. To accomplish this process, a novel collaborative knowledge transfer is proposed. At the KE, the student model is trained without the teacher networks and examined by challenging pixel loss derived by the ground truth. Moreover, to improve the performance of our training framework, a novel loss function called multi-contrastive knowledge regularization (MCR) loss is proposed. Experiments on several datasets show that our student model can achieve promising results on different bad weather removal tasks simultaneously. The code is available in our project page.

----

## [1701] Unsupervised Homography Estimation with Coplanarity-Aware GAN

**Authors**: *Mingbo Hong, Yuhang Lu, Nianjin Ye, Chunyu Lin, Qijun Zhao, Shuaicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01714](https://doi.org/10.1109/CVPR52688.2022.01714)

**Abstract**:

Estimating homography from an image pair is a fundamental problem in image alignment. Unsupervised learning methods have received increasing attention in this field due to their promising performance and label-free training. However, existing methods do not explicitly consider the problem of plane-induced parallax, which will make the predicted homography compromised on multiple planes. In this work, we propose a novel method HomoGAN to guide unsupervised homography estimation to focus on the dominant plane. First, a multi-scale transformer network is designed to predict homography from the feature pyramids of input images in a coarse-to-fine fashion. Moreover, we propose an unsupervised GAN to impose coplanarity constraint on the predicted homography, which is realized by using a generator to predict a mask of aligned regions, and then a discriminator to check if two masked feature maps are induced by a single homography. To validate the effectiveness of HomoGAN and its components, we conduct extensive experiments on a large-scale dataset, and results show that our matching error is 22% lower than the previous SOTA method. Code is available at https://github.com/megvii-research/HomoGAN

----

## [1702] Attentive Fine-Grained Structured Sparsity for Image Restoration

**Authors**: *Junghun Oh, Heewon Kim, Seungjun Nah, Cheeun Hong, Jonghyun Choi, Kyoung Mu Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01715](https://doi.org/10.1109/CVPR52688.2022.01715)

**Abstract**:

Image restoration tasks have witnessed great performance improvement in recent years by developing large deep models. Despite the outstanding performance, the heavy computation demanded by the deep models has restricted the application of image restoration. To lift the restriction, it is required to reduce the size of the networks while maintaining accuracy. Recently, N:M structured pruning has appeared as one of the effective and practical pruning approaches for making the model efficient with the accuracy constraint. However, it fails to account for different computational complexities and performance requirements for different layers of an image restoration network. To further optimize the trade-off between the efficiency and the restoration accuracy, we propose a novel pruning method that determines the pruning ratio for N:M structured sparsity at each layer. Extensive experimental results on super-resolution and deblurring tasks demonstrate the efficacy of our method which outperforms previous pruning methods significantly. PyTorch implementation for the proposed methods will be publicly available at https://github.com/JungHunOh/SLS_CVPR2022

----

## [1703] Uformer: A General U-Shaped Transformer for Image Restoration

**Authors**: *Zhendong Wang, Xiaodong Cun, Jianmin Bao, Wengang Zhou, Jianzhuang Liu, Houqiang Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01716](https://doi.org/10.1109/CVPR52688.2022.01716)

**Abstract**:

In this paper, we present Uformer, an effective and efficient Transformer-based architecture for image restoration, in which we build a hierarchical encoder-decoder network using the Transformer block. In Uformer, there are two core designs. First, we introduce a novel locally-enhanced window (LeWin) Transformer block, which performs non-overlapping window-based self-attention instead of global self-attention. It significantly reduces the computational complexity on high resolution feature map while capturing local context. Second, we propose a learnable multi-scale restoration modulator in the form of a multi-scale spatial bias to adjust features in multiple layers of the Uformer decoder. Our modulator demonstrates superior capability for restoring details for various image restoration tasks while introducing marginal extra parameters and computational cost. Powered by these two designs, Uformer enjoys a high capability for capturing both local and global dependencies for image restoration. To evaluate our approach, extensive experiments are conducted on several image restoration tasks, including image denoising, motion deblurring, defocus deblurring and deraining. Without bells and whistles, our Uformer achieves superior or comparable performance compared with the state-of-the-art algorithms. The code and models are available at https://github.com/ZhendongWang6/Uformer.

----

## [1704] Bringing Old Films Back to Life

**Authors**: *Ziyu Wan, Bo Zhang, Dongdong Chen, Jing Liao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01717](https://doi.org/10.1109/CVPR52688.2022.01717)

**Abstract**:

We present a learning-based framework, recurrent transformer network (RTN), to restore heavily degraded old films. Instead of performing frame-wise restoration, our method is based on the hidden knowledge learned from adjacent frames that contain abundant information about the occlusion, which is beneficial to restore challenging artifacts of each frame while ensuring temporal coherency. Moreover, contrasting the representation of the current frame and the hidden knowledge makes it possible to infer the scratch position in an unsupervised manner, and such defect localization generalizes well to real-world degradations. To better resolve mixed degradation and compensate for the flow estimation error during frame alignment, we propose to leverage more expressive transformer blocks for spatial restoration. Experiments on both synthetic dataset and real-world old films demonstrate the significant superiority of the proposed RTN over existing solutions. In addition, the same framework can effectively propagate the color from keyframes to the whole video, ultimately yielding compelling restored films. The implementation and model will be released at https://github.com/raywzy/Bringing-Old-Films-Back-to-Life.

----

## [1705] Learning sRGB-to-Raw-RGB De-rendering with Content-Aware Metadata

**Authors**: *Seonghyeon Nam, Abhijith Punnappurath, Marcus A. Brubaker, Michael S. Brown*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01718](https://doi.org/10.1109/CVPR52688.2022.01718)

**Abstract**:

Most camera images are rendered and saved in the standard RGB (sRGB) format by the camera's hardware. Due to the in-camera photo-finishing routines, nonlinear sRGB images are undesirable for computer vision tasks that assume a direct relationship between pixel values and scene radiance. For such applications, linear raw-RGB sensor images are preferred. Saving images in their raw-RGB format is still uncommon due to the large storage requirement and lack of support by many imaging applications. Several “raw reconstruction” methods have been proposed that utilize specialized metadata sampled from the raw-RGB image at capture time and embedded in the sRGB image. This metadata is used to parameterize a mapping function to derender the sRGB image back to its original raw-RGB format when needed. Existing raw reconstruction methods rely on simple sampling strategies and global mapping to perform the de-rendering. This paper shows how to improve the derendering results by jointly learning sampling and reconstruction. Our experiments show that our learned sampling can adapt to the image content to produce better raw reconstructions than existing methods. We also describe an online fine-tuning strategy for the reconstruction network to improve results further.

----

## [1706] SNR-Aware Low-light Image Enhancement

**Authors**: *Xiaogang Xu, Ruixing Wang, Chi-Wing Fu, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01719](https://doi.org/10.1109/CVPR52688.2022.01719)

**Abstract**:

This paper presents a new solution for low-light image enhancement by collectively exploiting Signal-to-Noise-Ratio-aware transformers and convolutional models to dynamically enhance pixels with spatial-varying operations. They are long-range operations for image regions of extremely low Signal-to-Noise-Ratio (SNR) and short-range operations for other regions. We propose to take an SNR prior to guide the feature fusion and formulate the SNR-aware transformer with a new self-attention model to avoid tokens from noisy image regions of very low SNR. Extensive experiments show that our framework consistently achieves better performance than SOTA approaches on seven representative benchmarks with the same structure. Also, we conducted a large-scale user study with 100 participants to verify the superior perceptual quality of our results. The code is available at https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance.

----

## [1707] AP-BSN: Self-Supervised Denoising for Real-World Images via Asymmetric PD and Blind-Spot Network

**Authors**: *Wooseok Lee, Sanghyun Son, Kyoung Mu Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01720](https://doi.org/10.1109/CVPR52688.2022.01720)

**Abstract**:

Blind-spot network (BSN) and its variants have made significant advances in self-supervised denoising. Never-theless, they are still bound to synthetic noisy inputs due to less practical assumptions like pixel-wise independent noise. Hence, it is challenging to deal with spatially corre-lated real-world noise using self-supervised BSN. Recently, pixel-shuffle downsampling (PD) has been proposed to re-move the spatial correlation of real-world noise. However, it is not trivial to integrate PD and BSN directly, which prevents the fully self-supervised denoising model on real-world images. We propose an Asymmetric PD (AP) to ad-dress this issue, which introduces different P D stride factors for training and inference. We systematically demonstrate that the proposed AP can resolve inherent trade-offs caused by specific PD stride factors and make BSN applicable to practical scenarios. To this end, we develop AP-BSN, a state-of-the-art self-supervised denoising method for real-world sRGB images. We further propose random-replacing refinement, which significantly improves the performance of our AP-BSN without any additional parameters. Extensive studies demonstrate that our method outperforms the other self-supervised and even unpaired denoising methods by a large margin, without using any additional knowledge, e.g., noise level, regarding the underlying unknown noise.

----

## [1708] Synthetic Aperture Imaging with Events and Frames

**Authors**: *Wei Liao, Xiang Zhang, Lei Yu, Shijie Lin, Wen Yang, Ning Qiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01721](https://doi.org/10.1109/CVPR52688.2022.01721)

**Abstract**:

The Event-based Synthetic Aperture Imaging (E-SAI) has recently been proposed to see through extremely dense occlusions. However, the performance of E-SAI is not consistent under sparse occlusions due to the dramatic de-crease of signal events. This paper addresses this problem by leveraging the merits of both events and frames, leading to a fusion-based SAl (EF-SAI) that performs consistently under the different densities of occlusions. In particular, we first extract the feature from events and frames via multi-modal feature encoders and then apply a multi-stage fusion network for cross-modal enhancement and density-aware feature selection. Finally, a CNN decoder is employed to generate occlusion-free visual images from selected features. Extensive experiments show that our method effectively tackles varying densities of occlusions and achieves superior performance to the state-of-the-art SAl methods. Codes and datasets are available at https://github.com/smjsc/EF-SAI

----

## [1709] Ev-TTA: Test-Time Adaptation for Event-Based Object Recognition

**Authors**: *Junho Kim, Inwoo Hwang, Young Min Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01722](https://doi.org/10.1109/CVPR52688.2022.01722)

**Abstract**:

We introduce Ev-TTA, a simple, effective test-time adaptation algorithm for event-based object recognition. While event cameras are proposed to provide measurements of scenes with fast motions or drastic illumination changes, many existing event-based recognition algorithms suffer from performance deterioration under extreme conditions due to significant domain shifts. Ev-TTA mitigates the severe domain gaps by fine-tuning the pre-trained classifiers during the test phase using loss functions inspired by the spatio-temporal characteristics of events. Since the event data is a temporal stream of measurements, our loss function enforces similar predictions for adjacent events to quickly adapt to the changed environment online. Also, we utilize the spatial correlations between two polarities of events to handle noise under extreme illumination, where different polarities of events exhibit distinctive noise distributions. Ev-TTA demonstrates a large amount of performance gain on a wide range of event-based object recognition tasks without extensive additional training. Our formulation can be successfully applied regardless of input representations and further extended into regression tasks. We expect Ev-TTA to provide the key technique to deploy event-based vision algorithms in challenging real-world applications where significant domain shift is inevitable.

----

## [1710] Time Lens++: Event-based Frame Interpolation with Parametric Nonlinear Flow and Multi-scale Fusion

**Authors**: *Stepan Tulyakov, Alfredo Bochicchio, Daniel Gehrig, Stamatios Georgoulis, Yuanyou Li, Davide Scaramuzza*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01723](https://doi.org/10.1109/CVPR52688.2022.01723)

**Abstract**:

Recently, video frame interpolation using a combination of frame- and event-based cameras has surpassed traditional image-based methods both in terms of performance and memory efficiency. However, current methods still suffer from (i) brittle image-level fusion of complementary interpolation results, that fails in the presence of artifacts in the fused image, (ii) potentially temporally inconsistent and inefficient motion estimation procedures, that run for every inserted frame and (iii) low contrast regions that do not trigger events, and thus cause events-only motion estimation to generate artifacts. Moreover, previous methods were only tested on datasets consisting of planar and far-away scenes, which do not capture the full complexity of the real world. In this work, we address the above problems by introducing multi-scale feature-level fusion and computing one-shot non-linear inter-frame motion-which can be efficiently sampled for image warping-from events and images. We also collect the first large-scale events and frames dataset consisting of more than 100 challenging scenes with depth variations, captured with a new experimental setup based on a beamsplitter. We show that our method improves the reconstruction quality by up to 0.2 dB in terms of PSNR and up to 15% in LPIPS score.

----

## [1711] Unifying Motion Deblurring and Frame Interpolation with Events

**Authors**: *Xiang Zhang, Lei Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01724](https://doi.org/10.1109/CVPR52688.2022.01724)

**Abstract**:

Slow shutter speed and long exposure time of frame-based cameras often cause visual blur and loss of inter-frame information, degenerating the overall quality of captured videos. To this end, we present a unified framework of event-based motion deblurring and frame interpolation for blurry video enhancement, where the extremely low latency of events is leveraged to alleviate motion blur and facilitate intermediate frame prediction. Specifically, the mapping relation between blurry frames and sharp latent images is first predicted by a learnable double integral network, and a fusion network is then proposed to refine the coarse results via utilizing the information from consecutive blurry inputs and the concurrent events. By exploring the mutual constraints among blurry frames, latent images, and event streams, we further propose a self-supervised learning framework to enable network training with real-world blurry videos and events. Extensive experiments demonstrate that our method compares favorably against the state-of-the-art approaches and achieves remarkable performance on both synthetic and real-world datasets. Codes are available at https://github.com/XiangZ-0/EVDI.

----

## [1712] EvUnroll: Neuromorphic Events based Rolling Shutter Image Correction

**Authors**: *Xinyu Zhou, Peiqi Duan, Yi Ma, Boxin Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01725](https://doi.org/10.1109/CVPR52688.2022.01725)

**Abstract**:

This paper proposes to use neuromorphic events for correcting rolling shutter (RS) images as consecutive global shutter (GS) frames. RS effect introduces edge distortion and region occlusion into images caused by row-wise read-out of CMOS sensors. We introduce a novel computational imaging setup consisting of an RS sensor and an event sensor, and propose a neural network called EvUnroll to solve this problem by exploring the high-temporal-resolution property of events. We use events to bridge a spatio-temporal connection between RS and GS, establish a flow estimation module to correct edge distortions, and design a synthesis-based restoration module to restore occluded regions. The results of two branches are fused through a refining module to generate corrected GS images. We further propose datasets captured by a high-speed camera and an RS-Event hybrid camera system for training and testing our network. Experimental results on both public and proposed datasets show a systematic performance improvement compared to state-of-the-art methods.

----

## [1713] Learning Adaptive Warping for RealWorld Rolling Shutter Correction

**Authors**: *Mingdeng Cao, Zhihang Zhong, Jiahao Wang, Yinqiang Zheng, Yujiu Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01726](https://doi.org/10.1109/CVPR52688.2022.01726)

**Abstract**:

This paper proposes the first real-world rolling shutter (RS) correction dataset, BS-RSC, and a corresponding model to correct the RS frames in a distorted video. Mobile devices in the consumer market with CMOS-based sensors for video capture often result in rolling shutter effects when relative movements occur during the video acquisition process, calling for RS effect removal techniques. However, current state-of-the-art RS correction methods often fail to remove RS effects in real scenarios since the motions are various and hard to model. To address this issue, we propose a real-world RS correction dataset BS-RSC. Real distorted videos with corresponding ground truth are recorded simultaneously via a well-designed beam-splitter-based acquisition system. BS-RSC contains various motions of both camera and objects in dynamic scenes. Further, an RS correction model with adaptive warping is proposed. Our model can warp the learned RS features into global shutter counterparts adaptively with predicted multiple displacement fields. These warped features are aggregated and then reconstructed into high-quality global shutter frames in a coarse-to-fine strategy. Experimental results demonstrate the effectiveness of the proposed method, and our dataset can improve the model's ability to remove the RS effects in the real world. The project is available at https://github.com/ljzycmd/BSRSC.

----

## [1714] Neural Global Shutter: Learn to Restore Video from a Rolling Shutter Camera with Global Reset Feature

**Authors**: *Zhixiang Wang, Xiang Ji, Jia-Bin Huang, Shin'ichi Satoh, Xiao Zhou, Yinqiang Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01727](https://doi.org/10.1109/CVPR52688.2022.01727)

**Abstract**:

Most computer vision systems assume distortion-free images as inputs. The widely used rolling-shutter (RS) image sensors, however, suffer from geometric distortion when the camera and object undergo motion during capture. Extensive researches have been conducted on correcting RS distortions. However, most of the existing work relies heavily on the prior assumptions of scenes or motions. Besides, the motion estimation steps are either oversimplified or computationally inefficient due to the heavy flow warping, limiting their applicability. In this paper, we investigate using rolling shutter with a global reset feature (RSGR) to restore clean global shutter (GS) videos. This feature enables us to turn the rectification problem into a deblur-like one, getting rid of inaccurate and costly explicit motion estimation. First, we build an optic system that captures paired RSGR/GS videos. Second, we develop a novel algorithm incorporating spatial and temporal designs to correct the spatial-varying RSGR distortion. Third, we demonstrate that existing image-to-image translation algorithms can recover clean GS videos from distorted RSGR inputs, yet our algorithm achieves the best performance with the specific designs. Our rendered results are not only visually appealing but also beneficial to downstream tasks. Compared to the state-of-the-art RS solution, our RSGR solution is superior in both effectiveness and efficiency. Considering it is easy to realize without changing the hardware, we believe our RSGR solution can potentially replace the RS solution in taking distortion-free videos with low noise and low budget.

----

## [1715] TimeReplayer: Unlocking the Potential of Event Cameras for Video Interpolation

**Authors**: *Weihua He, Kaichao You, Zhendong Qiao, Xu Jia, Ziyang Zhang, Wenhui Wang, Huchuan Lu, Yaoyuan Wang, Jianxing Liao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01728](https://doi.org/10.1109/CVPR52688.2022.01728)

**Abstract**:

Recording fast motion in a high FPS (frame-per-second) requires expensive high-speed cameras. As an alternative, interpolating low-FPS videos from commodity cameras has attracted significant attention. If only low-FPS videos are available, motion assumptions (linear or quadratic) are necessary to infer intermediate frames, which fail to model complex motions. Event camera, a new camera with pixels producing events of brightness change at the temporal resolution of μs (10–
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">6</sup>
 second), is a game-changing device to enable video interpolation at the presence of arbitrarily complex motion. Since event camera is a novel sensor, its potential has not been fulfilled due to the lack of processing algorithms. The pioneering work Time Lens introduced event cameras to video interpolation by designing optical devices to collect a large amount of paired training data of high-speed frames and events, which is too costly to scale. To fully unlock the potential of event cameras, this paper proposes a novel TimeReplayer algorithm to interpolate videos captured by commodity cameras with events. It is trained in an unsupervised cycleconsistent style, canceling the necessity of high-speed training data and bringing the additional ability of video extrapolation. Its state-of-the-art results and demo videos in supplementary reveal the promising future of event-based vision.

----

## [1716] Optimizing Video Prediction via Video Frame Interpolation

**Authors**: *Yue Wu, Qiang Wen, Qifeng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01729](https://doi.org/10.1109/CVPR52688.2022.01729)

**Abstract**:

Video prediction is an extrapolation task that predicts future frames given past frames, and video frame interpolation is an interpolation task that estimates intermediate frames between two frames. We have witnessed the tremendous advancement of video frame interpolation, but the general video prediction in the wild is still an open question. Inspired by the photo-realistic results of video frame interpolation, we present a new optimization framework for video prediction via video frame interpolation, in which we solve an extrapolation problem based on an interpolation model. Our video prediction framework is based on optimization with a pretrained differentiable video frame interpolation module without the need for a training dataset, and thus there is no domain gap issue between training and test data. Also, our approach does not need any additional information such as semantic or instance maps, which makes our framework applicable to any video. Extensive experiments on the Cityscapes, KITTI, DAVIS, Middlebury, and Vimeo90K datasets show that our video prediction results are robust in general scenarios, and our approach outperforms other video prediction methods that require a large amount of training data or extra semantic information.

----

## [1717] Reference-based Video Super-Resolution Using Multi-Camera Video Triplets

**Authors**: *Junyong Lee, Myeonghee Lee, Sunghyun Cho, Seungyong Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01730](https://doi.org/10.1109/CVPR52688.2022.01730)

**Abstract**:

We propose the first reference-based video super-resolution (RefVSR) approach that utilizes reference videos for high-fidelity results. We focus on RefVSR in a triple-camera setting, where we aim at super-resolving a low-resolution ultra-wide video utilizing wide-angle and tele-photo videos. We introduce the first RefVSR network that re-currently aligns and propagates temporal reference features fused with features extracted from low-resolution frames. To facilitate the fusion and propagation of temporal reference features, we propose a propagative temporal fusion module. For learning and evaluation of our network, we present the first RefVSR dataset consisting of triplets of ultra-wide, wide-angle, and telephoto videos concurrently taken from triple cameras of a smartphone. We also propose a two-stage training strategy fully utilizing video triplets in the proposed dataset for real-world 4 × video super-resolution. We extensively evaluate our method, and the result shows the state-of-the-art performance in 4 × super-resolution.

----

## [1718] Memory-Augmented Non-Local Attention for Video Super-Resolution

**Authors**: *Jiyang Yu, Jingen Liu, Liefeng Bo, Tao Mei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01731](https://doi.org/10.1109/CVPR52688.2022.01731)

**Abstract**:

In this paper, we propose a simple yet effective video super-resolution method that aims at generating highfidelity high-resolution (HR) videos from low-resolution (LR) ones. Previous methods predominantly leverage temporal neighbor frames to assist the super-resolution of the current frame. Those methods achieve limited performance as they suffer from the challenges in spatial frame alignment and the lack of useful information from similar LR neighbor frames. In contrast, we devise a cross-frame non-local attention mechanism that allows video superresolution without frame alignment, leading to being more robust to large motions in the video. In addition, to acquire general video prior information beyond neighbor frames, and to compensate for the information loss caused by large motions, we design a novel memory-augmented attention module to memorize general video details during the superresolution training. We have thoroughly evaluated our work on various challenging datasets. Compared to other recent video super-resolution approaches, our method not only achieves significant performance gains on large motion videos but also shows better generalization. Our source code and the new Parkour benchmark dataset is available at https://github.com/jiy173/MANA.

----

## [1719] Optical Flow Estimation for Spiking Camera

**Authors**: *Liwen Hu, Rui Zhao, Ziluo Ding, Lei Ma, Boxin Shi, Ruiqin Xiong, Tiejun Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01732](https://doi.org/10.1109/CVPR52688.2022.01732)

**Abstract**:

As a bio-inspired sensor with high temporal resolution, the spiking camera has an enormous potential in real applications, especially for motion estimation in high-speed scenes. However, frame-based and event-based methods are not well suited to spike streams from the spiking camera due to the different data modalities. To this end, we present, SCFlow, a tailored deep learning pipeline to estimate optical flow in high-speed scenes from spike streams. Importantly, a novel input representation is introduced which can adaptively remove the motion blur in spike streams according to the prior motion. Further, for training SCFlow, we synthesize two sets of optical flow data for the spiking camera, SPIkingly Flying Things and Photo-realistic Highspeed Motion, denoted as SPIFT and PHM respectively, corresponding to random high-speed and well-designed scenes. Experimental results show that the SCFlow can predict optical flow from spike streams in different high-speed scenes. Moreover, SCFlow shows promising generalization on real spike streams. Codes and datasets refer to https://github.com/Acnext/Optical-Flow-For-Spiking-Camera.

----

## [1720] Compressive Single-Photon 3D Cameras

**Authors**: *Felipe Gutierrez-Barragan, Atul Ingle, Trevor Seets, Mohit Gupta, Andreas Velten*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01733](https://doi.org/10.1109/CVPR52688.2022.01733)

**Abstract**:

Single-photon avalanche diodes (SPADs) are an emerging pixel technology for time-of-flight (ToF) 3D cameras that can capture the time-of-arrival of individual photons at picosecond resolution. To estimate depths, current SPAD-based 3D cameras measure the round-trip time of a laser pulse by building a per-pixel histogram of photon times-tamps. As the spatial and timestamp resolution of SPAD-based cameras increase, their output data rates far exceed the capacity of existing data transfer technologies. One major reason for SPAD's bandwidth-intensive operation is the tight coupling that exists between depth resolution and histogram resolution. To weaken this coupling, we propose compressive single-photon histograms (CSPH). CSPHs are a per-pixel compressive representation of the high-resolution histogram, that is built on-the-fly, as each photon is detected. They are based on a family of linear coding schemes that can be expressed as a simple matrix operation. We design different CSPH coding schemes for 3D imaging and evaluate them under different signal and background levels, laser waveforms, and illumination setups. Our results show that a well-designed CSPH can consistently reduce data rates by 1–2 orders of magnitude without compromising depth precision.

----

## [1721] Single-Photon Structured Light

**Authors**: *Varun Sundar, Sizhuo Ma, Aswin C. Sankaranarayanan, Mohit Gupta*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01734](https://doi.org/10.1109/CVPR52688.2022.01734)

**Abstract**:

We present a novel structured light technique that uses Single Photon Avalanche Diode (SPAD) arrays to enable 3D scanning at high-frame rates and low-light levels. This technique, called “Single-Photon Structured Light”, works by sensing binary images that indicates the presence or absence of photon arrivals during each exposure; the SPAD array is used in conjunction with a high-speed binary projector, with both devices operated at speeds as high as 20 kHz. The binary images that we acquire are heavily influenced by photon noise and are easily corrupted by ambient sources of light. To address this, we develop novel temporal sequences using error correction codes that are designed to be robust to short-range effects like projector and camera defocus as well as resolution mismatch between the two devices. Our lab prototype is capable of 3D imaging in challenging scenarios involving objects with extremely low albedo or undergoing fast motion, as well as scenes under strong ambient illumination.

----

## [1722] All-photon Polarimetric Time-of-Flight Imaging

**Authors**: *Seung-Hwan Baek, Felix Heide*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01735](https://doi.org/10.1109/CVPR52688.2022.01735)

**Abstract**:

Time-of-flight (ToF) sensors provide an image modal-ity fueling diverse applications, including LiDAR in au-tonomous driving, robotics, and augmented reality. Con-ventional ToF imaging methods estimate depth by sending pulses of light into a scene and measuring the ToF of the first-arriving photons directly reflected from a scene surface without any temporal delay. As such, all photons following this first response are typically considered as unwanted noise. In this paper, we depart from the principle of using first-arriving photons and propose an all-photon ToF imaging method that relies on the temporal-polarimetric analysis of first- and late-arriving photons which encode rich scene information in terms of geometry and material. To this end, we propose a novel temporal-polarimetric re-flectance model, an efficient capture method, and a reconstruction method that exploits the temporal-polarimetric changes of light reflected by the surface and sub-surface reflection. The proposed all-photon polarimetric ToF imaging method allows us to acquire depth, surface normals, and material parameters of a scene by utilizing all photons captured by the system, whereas conventional ToF imaging only obtains coarse depth from the first-arriving photons. We validate our method in simulation and experimentally with a prototype system.

----

## [1723] Holocurtains: Programming Light Curtains via Binary Holography

**Authors**: *Dorian Chan, Srinivasa G. Narasimhan, Matthew O'Toole*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01736](https://doi.org/10.1109/CVPR52688.2022.01736)

**Abstract**:

Light curtain systems are designed for detecting the presence of objects within a user-defined 3D region of space, which has many applications across vision and robotics. However, the shape of light curtains have so far been limited to ruled surfaces, i.e., surfaces composed of straight lines. In this work, we propose Holocurtains: a light-efficient approach to producing light curtains of arbitrary shape. The key idea is to synchronize a rolling-shutter camera with a 2D holographic projector, which steers (rather than block) light to generate bright structured light patterns. Our prototype projector uses a binary digital micromirror device (DMD) to generate the holographic interference patterns at high speeds. Our system produces 3D light curtains that cannot be achieved with traditional light curtain setups and thus enables all-new applications, including the ability to simultaneously capture multiple light curtains in a single frame, detect subtle changes in scene geometry, and transform any 3D surface into an optical touch interface.

----

## [1724] Towards Implicit Text-Guided 3D Shape Generation

**Authors**: *Zhengzhe Liu, Yi Wang, Xiaojuan Qi, Chi-Wing Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01737](https://doi.org/10.1109/CVPR52688.2022.01737)

**Abstract**:

In this work, we explore the challenging task of generating 3D shapes from text. Beyond the existing works, we propose a new approach for text-guided 3D shape generation, capable of producing high-fidelity shapes with colors that match the given text description. This work has several technical contributions. First, we decouple the shape and color predictions for learning features in both texts and shapes, and propose the word-level spatial transformer to correlate word features from text with spatial features from shape. Also, we design a cyclic loss to encourage consistency between text and shape, and introduce the shape IMLE to diversify the generated shapes. Further, we extend the framework to enable text-guided shape manipulation. Extensive experiments on the largest existing text-shape benchmark [10] manifest the superiority of this work. The code and the models are available at https://github.com/liuzhengzhe/Towards-Implicit-Text-Guided-Shape-Generation.

----

## [1725] Towards Language-Free Training for Text-to-Image Generation

**Authors**: *Yufan Zhou, Ruiyi Zhang, Changyou Chen, Chunyuan Li, Chris Tensmeyer, Tong Yu, Jiuxiang Gu, Jinhui Xu, Tong Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01738](https://doi.org/10.1109/CVPR52688.2022.01738)

**Abstract**:

One of the major challenges in training text-to-image generation models is the need of a large number of highquality image-text pairs. While image samples are often easily accessible, the associated text descriptions typically require careful human captioning, which is particularly time- and cost-consuming. In this paper, we propose the first work to train text-to-image generation models without any text data. Our method leverages the well-aligned multi-modal semantic space of the powerful pre-trained CLIP model: the requirement of text-conditioning is seamlessly alleviated via generating text features from image features. Extensive experiments are conducted to illustrate the effectiveness of the proposed method. We obtain state-of-the-art results in the standard text-to-image generation tasks. Importantly, the proposed language-free model outperforms most existing models trained with full image-text pairs. Furthermore, our method can be applied in fine-tuning pretrained models, which saves both training time and cost in training text-to-image generation models. Our pre-trained model obtains competitive results in zero-shot text-to-image generation on the MS-COCO dataset, yet with around only 1% of the model size and training data size relative to the recently proposed large DALL-E model.

----

## [1726] ZeroCap: Zero-Shot Image-to-Text Generation for Visual-Semantic Arithmetic

**Authors**: *Yoad Tewel, Yoav Shalev, Idan Schwartz, Lior Wolf*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01739](https://doi.org/10.1109/CVPR52688.2022.01739)

**Abstract**:

Recent text-to-image matching models apply contrastive learning to large corpora of uncurated pairs of images and sentences. While such models can provide a powerful score for matching and subsequent zero-shot tasks, they are not capable of generating caption given an image. In this work, we repurpose such models to generate a descriptive text given an image at inference time, without any further training or tuning step. This is done by combining the visual-semantic model with a large language model, benefiting from the knowledge in both web-scale models. The resulting captions are much less restrictive than those obtained by supervised captioning methods. Moreover, as a zero-shot learning method, it is extremely flexible and we demonstrate its ability to perform image arithmetic in which the inputs can be either images or text and the output is a sentence. This enables novel high-level vision capabilities such as comparing two images or solving visual analogy tests. Our code is available at: https://github.com/YoadTew/zero-shot-image-to-text.

----

## [1727] EMScore: Evaluating Video Captioning via Coarse-Grained and Fine-Grained Embedding Matching

**Authors**: *Yaya Shi, Xu Yang, Haiyang Xu, Chunfeng Yuan, Bing Li, Weiming Hu, Zheng-Jun Zha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01740](https://doi.org/10.1109/CVPR52688.2022.01740)

**Abstract**:

Current metrics for video captioning are mostly based on the text-level comparison between reference and candidate captions. However, they have some insuperable drawbacks, e.g., they cannot handle videos without references, and they may result in biased evaluation due to the one-to-many nature of video-to-text and the neglect of visual relevance. From the human evaluator's viewpoint, a high-quality caption should be consistent with the provided video, but not necessarily be similar to the reference in literal or semantics. Inspired by human evaluation, we propose EMScore (Embedding Matching-based score), a novel reference-free metric for video captioning, which directly measures similarity between video and candidate captions. Benefiting from the recent development of large-scale pre-training models, we exploit a well pre-trained vision-language model to extract visual and linguistic embeddings for computing EMScore. Specifically, EMScore combines matching scores of both coarse-grained (video and caption) and fine-grained (frames and words) levels, which takes the overall understanding and detailed characteristics of the video into account. Furthermore, considering the potential information gain, EMScore can be flexibly extended to the conditions where human-labeled references are available. Last but not least, we collect VATEX-EVAL and ActivityNet-FOIl datasets to systematically evaluate the existing metrics. VATEX-EVAL experiments demonstrate that EMScore has higher human correlation and lower reference dependency. ActivityNet-FOIL experiment verifies that EMScore can effectively identify “hallucinating” captions. Code and datasets are available at https://github.com/shiyaya/emscore.

----

## [1728] Hierarchical Modular Network for Video Captioning

**Authors**: *Hanhua Ye, Guorong Li, Yuankai Qi, Shuhui Wang, Qingming Huang, Ming-Hsuan Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01741](https://doi.org/10.1109/CVPR52688.2022.01741)

**Abstract**:

Video captioning aims to generate natural language descriptions according to the content, where representation learning plays a crucial role. Existing methods are mainly developed within the supervised learning framework via word-by-word comparison of the generated caption against the ground-truth text without fully exploiting linguistic semantics. In this work, we propose a hierarchical modular network to bridge video representations and linguistic semantics from three levels before generating captions. In particular, the hierarchy is composed of: (I) Entity level, which highlights objects that are most likely to be mentioned in captions. (II) Predicate level, which learns the actions conditioned on highlighted objects and is supervised by the predicate in captions. (III) Sentence level, which learns the global semantic representation and is supervised by the whole caption. Each level is implemented by one module. Extensive experimental results show that the proposed method performs favorably against the state-of-the-art models on the two widely-used benchmarks: MSVD 104.0% and MSR-VTT 51.5% in CIDEr score. Code will be made available at https://github.com/MarcusNerva/HMN.

----

## [1729] SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning

**Authors**: *Kevin Lin, Linjie Li, Chung-Ching Lin, Faisal Ahmed, Zhe Gan, Zicheng Liu, Yumao Lu, Lijuan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01742](https://doi.org/10.1109/CVPR52688.2022.01742)

**Abstract**:

The canonical approach to video captioning dictates a caption generation model to learn from offline-extracted dense video features. These feature extractors usually operate on video frames sampled at a fixed frame rate and are often trained on image/video understanding tasks, without adaption to video captioning data. In this work, we present SwinBERT, an end-to-end transformer-based model for video captioning, which takes video frame patches directly as inputs, and outputs a natural language description. Instead of leveraging multiple 2D/3D feature extractors, our method adopts a video transformer to encode spatial-temporal representations that can adapt to variable lengths of video input without dedicated design for different frame rates. Based on this model architecture, we show that video captioning can benefit significantly from more densely sampled video frames as opposed to previous successes with sparsely sampled video frames for video-and-language understanding tasks (e.g., video question answering). Moreover, to avoid the inherent redundancy in consecutive video frames, we propose adaptively learning a sparse attention mask and optimizing it for task-specific performance improvement through better long-range video sequence modeling. Through extensive experiments on 5 video captioning datasets, we show that Swinbert achieves across-the-board performance improvements over previous methods, often by a large margin. The learned sparse attention masks in addition push the limit to new state of the arts, and can be transferred between different video lengths and between different datasets. Code is available at https://github.com/microsoft/SwinBERT.

----

## [1730] End-to-end Generative Pretraining for Multimodal Video Captioning

**Authors**: *Paul Hongsuck Seo, Arsha Nagrani, Anurag Arnab, Cordelia Schmid*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01743](https://doi.org/10.1109/CVPR52688.2022.01743)

**Abstract**:

Recent video and language pretraining frameworks lack the ability to generate sentences. We present Multimodal Video Generative Pretraining (MV-GPT), a new pretraining framework for learning from unlabelled videos which can be effectively used for generative tasks such as multimodal video captioning. Unlike recent video-language pretraining frameworks, our framework trains both a multimodal video encoder and a sentence decoder jointly. To overcome the lack of captions in unlabelled videos, we leverage the future utterance as an additional text source and propose a bidirectional generation objective - we generate future utterances given the present mulitmodal context, and also the present utterance given future observations. With this objective, we train an encoder-decoder model end-to-end to generate a caption from raw pixels and transcribed speech directly. Our model achieves state-of the-art performance for multimodal video captioning on four standard benchmarks, as well as for other video understanding tasks such as VideoQA, video retrieval and action classification.

----

## [1731] Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning

**Authors**: *Chia-Wen Kuo, Zsolt Kira*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01744](https://doi.org/10.1109/CVPR52688.2022.01744)

**Abstract**:

Significant progress has been made on visual captioning, largely relying on pre-trained features and later fixed object detectors that serve as rich inputs to auto-regressive models. A key limitation of such methods, however, is that the output of the model is conditioned only on the object detector's outputs. The assumption that such outputs can represent all necessary information is unrealistic, especially when the detector is transferred across datasets. In this work, we reason about the graphical model induced by this assumption, and propose to add an auxiliary input to represent missing information such as object relationships. We specifically propose to mine attributes and relationships from the Visual Genome dataset and condition the captioning model on them. Crucially, we propose (and show to be important) the use of a multi-modal pre-trained model (CLIP) to retrieve such contextual descriptions. Further, object detector models are frozen and do not have sufficient richness to allow the captioning model to properly ground them. As a result, we propose to condition both the detector and description outputs on the image, and show qualitatively and quantitatively that this can improve grounding. We validate our method on image captioning, perform thorough analyses of each component and importance of the pre-trained multi-modal model, and demonstrate significant improvements over the current state of the art, specifically +7.5% in CIDEr and +1.3% in BLEU-4 metrics.

----

## [1732] Scaling Up Vision-Language Pretraining for Image Captioning

**Authors**: *Xiaowei Hu, Zhe Gan, Jianfeng Wang, Zhengyuan Yang, Zicheng Liu, Yumao Lu, Lijuan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01745](https://doi.org/10.1109/CVPR52688.2022.01745)

**Abstract**:

In recent years, we have witnessed significant performance boost in the image captioning task based on vision-language pre-training (VLP). Scale is believed to be an important factor for this advance. However, most existing work only focuses on pre-training transformers with moderate sizes (e.g., 12 or 24 layers) on roughly 4 million images. In this paper, we present LEMON O, a LargE-scale iMage captiONer, and provide the first empirical study on the scaling behavior of VLP for image captioning. We use the state-of-the-art Vin VL model as our reference model, which consists of an image feature extractor and a transformer model, and scale the transformer both up and down, with model sizes ranging from 13 to 675 million parameters. In terms of data, we conduct experiments with up to 200 million imagetext pairs which are automatically collected from web based on the alt attribute of the image (dubbed as ALT200M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The dataset is released at https://github.com/xiaoweihu/ALT200M). Extensive analysis helps to characterize the performance trend as the model size and the pre-training data size increase. We also compare different training recipes, especially for training on large-scale noisy data. As a result, LEMON achieves new state of the arts on several major image captioning benchmarks, including COCO Caption, nocaps, and Conceptual Captions. We also show LEMON can generate captions with long-tail vi-sual concepts when used in a zero-shot manner.

----

## [1733] Comprehending and Ordering Semantics for Image Captioning

**Authors**: *Yehao Li, Yingwei Pan, Ting Yao, Tao Mei*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01746](https://doi.org/10.1109/CVPR52688.2022.01746)

**Abstract**:

Comprehending the rich semantics in an image and ordering them in linguistic order are essential to compose a visually-grounded and linguistically coherent description for image captioning. Modern techniques commonly capitalize on a pre-trained object detector/classifier to mine the semantics in an image, while leaving the inherent linguistic ordering of semantics under-exploited. In this paper, we propose a new recipe of Transformer-style structure, namely Comprehending and Ordering Semantics Networks (COS-Net), that novelly unifies an enriched semantic comprehending and a learnable semantic ordering processes into a single architecture. Technically, we initially utilize a cross-modal retrieval model to search the relevant sentences of each image, and all words in the searched sentences are taken as primary semantic cues. Next, a novel semantic comprehender is devised to filter out the irrelevant semantic words in primary semantic cues, and mean-while infer the missing relevant semantic words visually grounded in the image. After that, we feed all the screened and enriched semantic words into a semantic ranker, which learns to allocate all semantic words in linguistic order as humans. Such sequence of ordered semantic words are further integrated with visual tokens of images to trigger sentence generation. Empirical evidences show that COS-Net clearly surpasses the state-of-the-art approaches on COCO and achieves to-date the best CIDEr score of 141.1% on Karpathy test split. Source code is available at https://github.com/YehLi/xmodaler/tree/master/configs/image_caption/cosnet.

----

## [1734] NOC-REK: Novel Object Captioning with Retrieved Vocabulary from External Knowledge

**Authors**: *Duc Minh Vo, Hong Chen, Akihiro Sugimoto, Hideki Nakayama*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01747](https://doi.org/10.1109/CVPR52688.2022.01747)

**Abstract**:

Novel object captioning aims at describing objects absent from training data, with the key ingredient being the provision of object vocabulary to the model. Although existing methods heavily rely on an object detection model, we view the detection step as vocabulary retrieval from an external knowledge in the form of embeddings for any object's definition from Wiktionary, where we use in the retrieval image region features learned from a transformers model. We propose an end-to-end Novel Object Captioning with Retrieved vocabulary from External Knowledge method (NOC-REK), which simultaneously learns vocabulary retrieval and caption generation, successfully describing novel objects outside of the training dataset. Furthermore, our model eliminates the requirement for model retraining by simply updating the external knowledge whenever a novel object appears. Our comprehensive experiments on held-out COCO and Nocaps datasets show that our NOCREK is considerably effective against SOTAs.

----

## [1735] Injecting Semantic Concepts into End-to-End Image Captioning

**Authors**: *Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lin Liang, Zhe Gan, Lijuan Wang, Yezhou Yang, Zicheng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01748](https://doi.org/10.1109/CVPR52688.2022.01748)

**Abstract**:

Tremendous progresses have been made in recent years in developing better image captioning models, yet most of them rely on a separate object detector to extract regional features. Recent vision-language studies are shifting towards the detector-free trend by leveraging grid representations for more flexible model training and faster inference speed. However, such development is primarily focused on image understanding tasks, and remains less investigated for the caption generation task. In this paper, we are concerned with a better-performing detector-free image captioning model, and propose a pure vision transformer-based image captioning model, dubbed as ViTCAP, in which grid representations are used without extracting the regional features. For improved performance, we introduce a novel Concept Token Network (CTN) to predict the semantic concepts and then incorporate them into the end-to-end captioning. In particular, the CTN is built on the basis of a vision transformer, and is designed to predict the concept tokens through a classification task, from which the rich semantic information contained greatly benefits the captioning task. Compared with the previous detector-based models, ViTCAP drastically simplifies the architectures and at the same time achieves competitive performance on various challenging image captioning datasets. In particular, ViTCAP reaches 138.1 CIDEr scores on COCO-caption Karpathy-split, 93.8 and 108.6 CIDEr scores on nocaps and Google-CC captioning datasets, respectively.

----

## [1736] DIFNet: Boosting Visual Information Flow for Image Captioning

**Authors**: *Mingrui Wu, Xuying Zhang, Xiaoshuai Sun, Yiyi Zhou, Chao Chen, Jiaxin Gu, Xing Sun, Rongrong Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01749](https://doi.org/10.1109/CVPR52688.2022.01749)

**Abstract**:

Current Image Captioning (IC) methods predict textual words sequentially based on the input visual information from the visual feature extractor and the partially generated sentence information. However, for most cases, the partially generated sentence may dominate the target word prediction due to the insufficiency of visual information, making the generated descriptions irrelevant to the content of the given image. In this paper, we propose a Dual Information Flow Network (DIFNet
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Source code is available at: https://github.com/mrwu-mac/DIFNet) to address this issue, which takes segmentation feature as another visual information source to enhance the contribution of visual information for prediction. To maximize the use of two information flows, we also propose an effective feature fusion module termed Iterative Independent Layer Normalization (IILN) which can condense the most relevant inputs while retraining modality-specific information in each flow. Experiments show that our method is able to enhance the dependence of prediction on visual information, making word prediction more focused on the visual content, and thus achieves new state-of-the-art performance on the MSCOCO dataset, e.g., 136.2 CIDEr on COCO Karpathy test split.

----

## [1737] VisualGPT: Data-efficient Adaptation of Pretrained Language Models for Image Captioning

**Authors**: *Jun Chen, Han Guo, Kai Yi, Boyang Li, Mohamed Elhoseiny*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01750](https://doi.org/10.1109/CVPR52688.2022.01750)

**Abstract**:

The limited availability of annotated data often hinders real-world applications of machine learning. To efficiently learn from small quantities of multimodal data, we leverage the linguistic knowledge from a large pre-trained language model (PLM) and quickly adapt it to new domains of image captioning. To effectively utilize a pretrained model, it is critical to balance the visual input and prior linguistic knowledge from pretraining. We propose VisualGPT, which employs a novel self-resurrecting encoder-decoder attention mechanism to quickly adapt the PLM with a small amount of in-domain image-text data. The proposed self-resurrecting activation unit produces sparse activations that prevent accidental overwriting of linguistic knowledge. When trained on 0.1%, 0.5% and 1% of the respective training sets, VisualGPT surpasses the best baseline by up to 10.0% CIDEr on MS COCO [43] and 17.9% CIDEr on Conceptual Captions [63]. Furthermore, VisualGPT achieves the state-of-the-art result on IU X-ray [15], a medical report generation dataset. Our code is available at https://github.com/Vision-CAIR/VisualGPT.

----

## [1738] Show, Deconfound and Tell: Image Captioning with Causal Inference

**Authors**: *Bing Liu, Dong Wang, Xu Yang, Yong Zhou, Rui Yao, Zhiwen Shao, Jiaqi Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01751](https://doi.org/10.1109/CVPR52688.2022.01751)

**Abstract**:

The transformer-based encoder-decoder framework has shown remarkable performance in image captioning. However, most transformer-based captioning methods ever overlook two kinds of elusive confounders: the visual confounder and the linguistic confounder, which generally lead to harmful bias, induce the spurious correlations during training, and degrade the model generalization. In this paper, we first use Structural Causal Models (SCMs) to show how two confounders damage the image captioning. Then we apply the backdoor adjustment to propose a novel causal inference based image captioning (CIIC) framework, which consists of an interventional object detector (IOD) and an interventional transformer decoder (ITD) to jointly confront both confounders. In the encoding stage, the IOD is able to disentangle the region-based visual features by deconfounding the visual confounder. In the decoding stage, the ITD introduces causal intervention into the transformer decoder and deconfounds the visual and linguistic confounders simultaneously. Two modules collaborate with each other to alleviate the spurious correlations caused by the unobserved confounders. When tested on MSCOCO, our proposal significantly outperforms the state-of-the-art encoder-decoder models on Karpathy split and online test split. Code is published in https://github.com/CUMTGG/CIIC.

----

## [1739] EI-CLIP: Entity-aware Interventional Contrastive Learning for E-commerce Cross-modal Retrieval

**Authors**: *Haoyu Ma, Handong Zhao, Zhe Lin, Ajinkya Kale, Zhangyang Wang, Tong Yu, Jiuxiang Gu, Sunav Choudhary, Xiaohui Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01752](https://doi.org/10.1109/CVPR52688.2022.01752)

**Abstract**:

Cross language-image modality retrieval in E-commerce is a fundamental problem for product search, recommendation, and marketing services. Extensive efforts have been made to conquer the cross-modal retrieval problem in the general domain. When it comes to E-commerce, a com-mon practice is to adopt the pretrained model and finetune on E-commerce data. Despite its simplicity, the performance is sub-optimal due to overlooking the uniqueness of E-commerce multimodal data. A few recent efforts [10], [72] have shown significant improvements over generic methods with customized designs for handling product images. Unfortunately, to the best of our knowledge, no existing method has addressed the unique challenges in the e-commerce language. This work studies the outstanding one, where it has a large collection of special meaning entities, e.g., “Di s s e l (brand)”, “Top (category)”, “relaxed (fit)” in the fashion clothing business. By formulating such out-of-distribution finetuning process in the Causal Inference paradigm, we view the erroneous semantics of these special entities as confounders to cause the retrieval failure. To rectify these semantics for aligning with e-commerce do-main knowledge, we propose an intervention-based entity-aware contrastive learning framework with two modules, i.e., the Confounding Entity Selection Module and Entity-Aware Learning Module. Our method achieves competitive performance on the E-commerce benchmark Fashion-Gen. Particularly, in top-1 accuracy (R@l), we observe 10.3% and 10.5% relative improvements over the closest baseline in image-to-text and text-to-image retrievals, respectively.

----

## [1740] CLIPstyler: Image Style Transfer with a Single Text Condition

**Authors**: *Gihyun Kwon, Jong Chul Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01753](https://doi.org/10.1109/CVPR52688.2022.01753)

**Abstract**:

Existing neural style transfer methods require reference style images to transfer texture information of style images to content images. However, in many practical situations, users may not have reference style images but still be inter-ested in transferring styles by just imagining them. In order to deal with such applications, we propose a new framework that enables a style transfer ‘without’ a style image, but only with a text description of the desired style. Using the pre-trained text-image embedding model of CLIP, we demonstrate the modulation of the style of content images only with a single text condition. Specifically, we propose a patch-wise text-image matching loss with multiview augmentations for realistic texture transfer. Extensive experimental results confirmed the successful image style transfer with realistic textures that reflect semantic query texts.

----

## [1741] HairCLIP: Design Your Hair by Text and Reference Image

**Authors**: *Tianyi Wei, Dongdong Chen, Wenbo Zhou, Jing Liao, Zhentao Tan, Lu Yuan, Weiming Zhang, Nenghai Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01754](https://doi.org/10.1109/CVPR52688.2022.01754)

**Abstract**:

Hair editing is an interesting and challenging problem in computer vision and graphics. Many existing methods require well-drawn sketches or masks as conditional inputs for editing, however these interactions are neither straight-forward nor efficient. In order to free users from the tedious interaction process, this paper proposes a new hair editing interaction mode, which enables manipulating hair attributes individually or jointly based on the texts or reference images provided by users. For this purpose, we encode the image and text conditions in a shared embedding space and propose a unified hair editing framework by leveraging the powerful image text representation capability of the Contrastive Language-Image Pre-Training (CLIP) model. With the carefully designed network structures and loss functions, our framework can perform high-quality hair editing in a disentangled manner. Extensive experiments demonstrate the superiority of our approach in terms of manipulation accuracy, visual realism of editing results, and irrelevant attribute preservation.

----

## [1742] DenseCLIP: Language-Guided Dense Prediction with Context-Aware Prompting

**Authors**: *Yongming Rao, Wenliang Zhao, Guangyi Chen, Yansong Tang, Zheng Zhu, Guan Huang, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01755](https://doi.org/10.1109/CVPR52688.2022.01755)

**Abstract**:

Recent progress has shown that large-scale pre-training using contrastive image-text pairs can be a promising alternative for high-quality visual representation learning from natural language supervision. Benefiting from a broader source of supervision, this new paradigm exhibits impressive transferability to downstream classification tasks and datasets. However, the problem of transferring the knowledge learned from image-text pairs to more complex dense prediction tasks has barely been visited. In this work, we present a new framework for dense prediction by implicitly and explicitly leveraging the pre-trained knowledge from CLIP. Specifically, we convert the original image-text matching problem in CLIP to a pixel-text matching problem and use the pixel-text score maps to guide the learning of dense prediction models. By further using the contextual information from the image to prompt the language model, we are able to facilitate our model to better exploit the pretrained knowledge. Our method is model-agnostic, which can be applied to arbitrary dense prediction systems and various pre-trained visual backbones including both CLIP models and ImageNet pre-trained models. Extensive experiments demonstrate the superior performance of our methods on semantic segmentation, object detection, and instance segmentation tasks. Code is available at https://github.com/raoyongming/DenseCLIP.

----

## [1743] On Guiding Visual Attention with Language Specification

**Authors**: *Suzanne Petryk, Lisa Dunlap, Keyan Nasseri, Joseph Gonzalez, Trevor Darrell, Anna Rohrbach*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01756](https://doi.org/10.1109/CVPR52688.2022.01756)

**Abstract**:

While real world challenges typically define visual categories with language words or phrases, most visual classification methods define categories with numerical indices. However, the language specification of the classes provides an especially useful prior for biased and noisy datasets, where it can help disambiguate what features are task-relevant. Recently, large-scale multimodal models have been shown to recognize a wide variety of high-level concepts from a language specification even without additional image training data, but they are often unable to distinguish classes for more fine-grained tasks. CNNs, in contrast, can extract subtle image features that are required for fine-grained discrimination, but will overfit to any bias or noise in datasets. Our insight is to use high-level language specification as advice for constraining the classification evidence to task-relevant features, instead of distractors. To do this, we ground task-relevant words or phrases with attention maps from a pretrained large-scale model. We then use this grounding to supervise a classifier's spatial attention away from distracting context. We show that supervising spatial attention in this way improves performance on classification tasks with biased and noisy data, including ~3 −15% worst-group accuracy improvements and ~41-45% relative improvements on fairness metrics.

----

## [1744] UTC: A Unified Transformer with Inter-Task Contrastive Learning for Visual Dialog

**Authors**: *Cheng Chen, Zhenshan Tan, Qingrong Cheng, Xin Jiang, Qun Liu, Yudong Zhu, Xiaodong Gu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01757](https://doi.org/10.1109/CVPR52688.2022.01757)

**Abstract**:

Visual Dialog aims to answer multi-round, interactive questions based on the dialog history and image content. Existing methods either consider answer ranking and generating individually or only weakly capture the relation across the two tasks implicitly by two separate models. The research on a universal framework that jointly learns to rank and generate answers in a single model is seldom explored. In this paper, we propose a contrastive learning-based framework UTC to unify and facilitate both discriminative and generative tasks in visual dialog with a single model. Specifically, considering the inherent limitation of the previous learning paradigm, we devise two inter-task contrastive losses i.e., context contrastive loss and answer contrastive loss to make the discriminative and generative tasks mutually reinforce each other. These two com-plementary contrastive losses exploit dialog context and target answer as anchor points to provide representation learning signals from different perspectives. We evaluate our proposed UTC on the VisDial v1.0 dataset, where our method outperforms the state-of-the-art on both discriminative and generative tasks and surpasses previous state-of-the-art generative methods by more than 2 absolute points on Recall@1.

----

## [1745] Text-to-Image Synthesis based on Object-Guided Joint-Decoding Transformer

**Authors**: *Fuxiang Wu, Liu Liu, Fusheng Hao, Fengxiang He, Jun Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01758](https://doi.org/10.1109/CVPR52688.2022.01758)

**Abstract**:

Object-guided text-to-image synthesis aims to generate images from natural language descriptions built by two-step frameworks, i.e., the model generates the layout and then synthesizes images from the layout and captions. However, such frameworks have two issues: 1) complex structure, since generating language-related layout is not a trivial task; 2) error propagation, because the inappropriate layout will mislead the image synthesis and is hard to be revised. In this paper, we propose an object-guided joint-decoding module to simultaneously generate the image and the corresponding layout. Specially, we present the joint-decoding transformer to model the joint probability on images tokens and the corresponding layouts tokens, where layout tokens provide additional observed data to model the complex scene better. Then, we describe a novel Layout-Vqgan for layout encoding and decoding to provide more information about the complex scene. After that, we present the detail-enhanced module to enrich the language-related details based on two facts: 1) visual details could be omitted in the compression of VQGANs; 2) the joint-decoding transformer would not have sufficient generating capacity. The experiments show that our approach is competitive with previous object-centered models and can generate diverse and high-quality objects under the given layouts.

----

## [1746] LiT: Zero-Shot Transfer with Locked-image text Tuning

**Authors**: *Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, Lucas Beyer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01759](https://doi.org/10.1109/CVPR52688.2022.01759)

**Abstract**:

This paper presents contrastive-tuning, a simple method employing contrastive training to align image and text mod-els while still taking advantage of their pre-training. In our empirical study we find that locked pre-trained image mod-els with unlocked text models work best. We call this in-stance of contrastive-tuning “Locked-image Tuning” (LiT), which just teaches a text model to read out good repre-sentations from a pre-trained image model for new tasks. A LiT model gains the capability of zero-shot transfer to new vision tasks, such as image classification or retrieval. The proposed LiT is widely applicable; it works reliably with multiple pre-training methods (supervised and unsu-pervised) and across diverse architectures (ResNet, Vision Transformers and MLP-Mixer) using three different image-text datasets. With the transformer-based pre-trained ViT-g/14 model, the LiT model achieves 84.5% zero-shot trans-fer accuracy on the ImageNet test set, and 81.1% on the challenging out-of-distribution ObjectNet test set.

----

## [1747] GroupViT: Semantic Segmentation Emerges from Text Supervision

**Authors**: *Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas M. Breuel, Jan Kautz, Xiaolong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01760](https://doi.org/10.1109/CVPR52688.2022.01760)

**Abstract**:

Grouping and recognition are important components of visual scene understanding, e.g., for object detection and semantic segmentation. With end-to-end deep learning systems, grouping of image regions usually happens implicitly via top-down supervision from pixel-level recognition labels. Instead, in this paper, we propose to bring back the grouping mechanism into deep networks, which allows semantic segments to emerge automatically with only text supervision. We propose a hierarchical Grouping Vision Transformer (GroupViT), which goes beyond the regular grid structure representation and learns to group image regions into progressively larger arbitrary-shaped segments. We train GroupViT jointly with a text encoder on a large-scale image-text dataset via contrastive losses. With only text supervision and without any pixel-level annotations, GroupViT learns to group together semantic regions and successfully transfers to the task of semantic segmentation in a zero-shot manner, i.e., without any further fine-tuning. It achieves a zero-shot accuracy of 52.3% mIoU on the PASCAL VOC 2012 and 22.4% mIoU on PASCAL Context datasets, and performs competitively to state-of-the-art transfer-learning methods requiring greater levels of supervision. We open-source our code at https://github.com/NVlabs/GroupViT.

----

## [1748] ReSTR: Convolution-free Referring Image Segmentation Using Transformers

**Authors**: *Namyup Kim, Dongwon Kim, Suha Kwak, Cuiling Lan, Wenjun Zeng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01761](https://doi.org/10.1109/CVPR52688.2022.01761)

**Abstract**:

Referring image segmentation is an advanced semantic segmentation task where target is not a predefined class but is described in natural language. Most of existing methods for this task rely heavily on convolutional neural networks, which however have trouble capturing long-range dependencies between entities in the language expression and are not flexible enough for modeling interactions between the two different modalities. To address these issues, we present the first convolution-free model for referring image segmentation using transformers, dubbed ReSTR. Since it extracts features of both modalities through transformer encoders, it can capture long-range dependencies between entities within each modality. Also, ReSTR fuses features of the two modalities by a self-attention encoder, which enables flexible and adaptive interactions between the two modalities in the fusion process. The fused features are fed to a segmentation module, which works adaptively according to the image and language expression in hand. ReSTR is evaluated and compared with previous work on all public benchmarks, where it outperforms all existing models.

----

## [1749] LAVT: Language-Aware Vision Transformer for Referring Image Segmentation

**Authors**: *Zhao Yang, Jiaqi Wang, Yansong Tang, Kai Chen, Hengshuang Zhao, Philip H. S. Torr*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01762](https://doi.org/10.1109/CVPR52688.2022.01762)

**Abstract**:

Referring image segmentation is a fundamental vision-language task that aims to segment out an object referred to by a natural language expression from an image. One of the key challenges behind this task is leveraging the referring expression for highlighting relevant positions in the image. A paradigm for tackling this problem is to leverage a powerful vision-language (“cross-madal”) decoder to fuse features independently extracted from a vision encoder and a language encoder. Recent methods have made remarkable advancements in this paradigm by exploiting Transformers as cross-modal decoders, concurrent to the Transformer's overwhelming success in many other vision-language tasks. Adopting a different approach in this work, we show that significantly better cross-modal alignments can be achieved through the early fusion of linguistic and visual features in intermediate layers of a vision Transformer encoder network. By conducting cross-modal feature fusion in the visual feature encoding stage, we can leverage the well-proven correlation modeling power of a Transformer encoder for excavating helpful multi-modal context. This way, accurate segmentation results are readily harvested with a light-weight mask predictor. Without bells and whistles, our method surpasses the previous state-of-the-art methods on Ref CoCo, RefCOCO+, and G-Ref by large margins.

----

## [1750] An Empirical Study of Training End-to-End Vision-and-Language Transformers

**Authors**: *Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, Zicheng Liu, Michael Zeng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01763](https://doi.org/10.1109/CVPR52688.2022.01763)

**Abstract**:

Vision-and-language (VL) pre-training has proven to be highly effective on various VL downstream tasks. While recent work has shown that fully transformer-based VL models can be more efficient than previous region-feature-based methods, their performance on downstream tasks often degrades significantly. In this paper, we present Meter, a Multimodal End-to-end TransformER framework, through which we investigate how to design and pre-train a fully transformer-based VL model in an end-to-end manner. Specifically, we dissect the model designs along multiple dimensions: vision encoders (e.g., CLIP-ViT, Swin transformer), text encoders (e.g., RoBERTa, De-BERTa), multimodal fusion module (e.g., merged attention vs. co-attention), architectural design (e.g., encoder-only vs. encoder-decoder), and pre-training objectives (e.g., masked image modeling). We conduct comprehensive experiments and provide insights on how to train a performant VL transformer. Meterachieves an accuracy of 77.64% on the VQAv2 test-std set using only 4M images for pre-training, surpassing the state-of-the-art region-feature-based model by 1.04%, and outperforming the previous best fully transformer-based model by 1.6%. Notably, when further scaled up, our best VQA model achieves an accuracy of 80.54%. Code and pre-trained models are released at https://github.com/zdou0830/METER.

----

## [1751] Are Multimodal Transformers Robust to Missing Modality?

**Authors**: *Mengmeng Ma, Jian Ren, Long Zhao, Davide Testuggine, Xi Peng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01764](https://doi.org/10.1109/CVPR52688.2022.01764)

**Abstract**:

Multimodal data collected from the real world are often imperfect due to missing modalities. Therefore multimodal models that are robust against modal-incomplete data are highly preferred. Recently, Transformer models have shown great success in processing multimodal data. However, existing work has been limited to either architecture designs or pre-training strategies; whether Transformer models are naturally robust against missing-modal data has rarely been investigated. In this paper, we present the first-of-its-kind work to comprehensively investigate the behavior of Transformers in the presence of modal-incomplete data. Unsurprising, we find Transformer models are sensitive to missing modalities while different modal fusion strategies will significantly affect the robustness. What surprised us is that the optimal fusion strategy is dataset dependent even for the same Transformer model; there does not exist a universal strategy that works in general cases. Based on these findings, we propose a principle method to improve the robustness of Transformer models byautomatically searching for an optimal fusion strategy regarding input data. Experimental validations on three benchmarks support the superior performance of the proposed method.

----

## [1752] Text to Image Generation with Semantic-Spatial Aware GAN

**Authors**: *Wentong Liao, Kai Hu, Michael Ying Yang, Bodo Rosenhahn*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01765](https://doi.org/10.1109/CVPR52688.2022.01765)

**Abstract**:

Text-to-image synthesis (T2I) aims to generate photorealistic images which are semantically consistent with the text descriptions. Existing methods are usually built upon conditional generative adversarial networks (GANs) and initialize an image from noise with sentence embedding, and then refine the features with fine-grained word embedding iteratively. A close inspection of their generated images reveals a major limitation: even though the generated image holistically matches the description, individual image regions or parts of somethings are often not recognizable or consistent with words in the sentence, e.g. “a white crown”. To address this problem, we propose a novel framework Semantic-Spatial Aware GAN for synthesizing images from input text. Concretely, we introduce a simple and effective Semantic-Spatial Aware block, which (1) learns semantic-adaptive transformation conditioned on text to effectively fuse text features and image features, and (2) learns a semantic mask in a weakly-supervised way that depends on the current text-image fusion process in order to guide the transformation spatially. Experiments on the challenging COCO and CUB bird datasets demonstrate the advantage of our method over the recent state-of-the-art approaches, regarding both visual fidelity and alignment with input text description. Code available at https://github.com/wtliao/text2image.

----

## [1753] StyleT2I: Toward Compositional and High-Fidelity Text-to-Image Synthesis

**Authors**: *Zhiheng Li, Martin Renqiang Min, Kai Li, Chenliang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01766](https://doi.org/10.1109/CVPR52688.2022.01766)

**Abstract**:

Although progress has been made for text-to-image synthesis, previous methods fall short of generalizing to unseen or underrepresented attribute compositions in the input text. Lacking compositionality could have severe implications for robustness and fairness, e.g., inability to synthesize the face images of underrepresented demographic groups. In this paper, we introduce a new framework, StyleT2I, to improve the compositionality of text-to-image synthesis. Specifically, we propose a CLIP-guided Contrastive Loss to better distinguish different compositions among different sentences. To further improve the compositionality, we design a novel Semantic Matching Loss and a Spatial Constraint to identify attributes' latent directions for intended spatial region manipulations, leading to better disentangled latent representations of attributes. Based on the identified latent directions of attributes, we propose Compositional Attribute Adjustment to adjust the latent code, resulting in better compositionality of image synthesis. In addition, we leverage the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$l_{2}$</tex>
-norm regularization of identified latent directions (norm penalty) to strike a nice balance between image-text alignment and image fidelity. In the experiments, we devise a new dataset split and an evaluation metric to evaluate the compositionality of text-to-image synthesis models. The results show that StyleT2I outperforms previous approaches in terms of the consistency between the input text and synthesized images and achieves higher fidelity.

----

## [1754] Blended Diffusion for Text-driven Editing of Natural Images

**Authors**: *Omri Avrahami, Dani Lischinski, Ohad Fried*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01767](https://doi.org/10.1109/CVPR52688.2022.01767)

**Abstract**:

Natural language offers a highly intuitive interface for image editing. In this paper, we introduce the first solution for performing local (region-based) edits in generic natural images, based on a natural language description along with an ROI mask. We achieve our goal by leveraging and combining a pretrained language-image model (CLIP), to steer the edit towards a user-provided text prompt, with a denoising diffusion probabilistic model (DDPM) to generate natural-looking results. To seamlessly fuse the edited region with the unchanged parts of the image, we spatially blend noised versions of the input image with the local text-guided diffusion latent at a progression of noise levels. In addition, we show that adding augmentations to the diffusion process mitigates adversarial results. We compare against several baselines and related methods, both qualitatively and quantitatively, and show that our method outperforms these solutions in terms of overall realism, ability to preserve the background and matching the text. Finally, we show several text-driven editing applications, including adding a new object to an image, removing/replacing/altering existing objects, background replacement, and image extrapolation.

----

## [1755] Make It Move: Controllable Image-to-Video Generation with Text Descriptions

**Authors**: *Yaosi Hu, Chong Luo, Zhenzhong Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01768](https://doi.org/10.1109/CVPR52688.2022.01768)

**Abstract**:

Generating controllable videos conforming to user intentions is an appealing yet challenging topic in computer vision. To enable maneuverable control in line with user intentions, a novel video generation task, named Text-Image-to-Video generation (TI2V), is proposed. With both controllable appearance and motion, TI2V aims at generating videos from a static image and a text description. The key challenges of TI2V task lie both in aligning appearance and motion from different modalities, and in handling uncertainty in text descriptions. To address these challenges, we propose a Motion Anchor-based video GEnerator (MAGE) with an innovative motion anchor (MA) structure to store appearance-motion aligned representation. To model the uncertainty and increase the diversity, it further allows the injection of explicit condition and implicit randomness. Through three-dimensional axial transformers, MA is interacted with given image to generate next frames recursively with satisfying controllability and diversity. Accompanying the new task, we build two new video-text paired datasets based on MNIST and CATER for evaluation. Experiments conducted on these datasets verify the effectiveness of MAGE and show appealing potentials of TI2V task. Datasets are available at https://github.com/Youncy-Hu/MAGE.

----

## [1756] Predict, Prevent, and Evaluate: Disentangled Text-Driven Image Manipulation Empowered by Pre-Trained Vision-Language Model

**Authors**: *Zipeng Xu, Tianwei Lin, Hao Tang, Fu Li, Dongliang He, Nicu Sebe, Radu Timofte, Luc Van Gool, Errui Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01769](https://doi.org/10.1109/CVPR52688.2022.01769)

**Abstract**:

To achieve disentangled image manipulation, previous works depend heavily on manual annotation. Meanwhile, the available manipulations are limited to a pre-defined set the models were trainedfor. We propose a novelframework, i.e., Predict, Prevent, and Evaluate (PPE), for disentangled text-driven image manipulation that requires little manual annotation while being applicable to a wide variety of ma-nipulations. Our method approaches the targets by deeply exploiting the power of the large-scale pre-trained vision-language model CLIP [32]. Concretely, we firstly Predict the possibly entangled attributes for a given text command. Then, based on the predicted attributes, we introduce an entanglement loss to Prevent entanglements during training. Finally, we propose a new evaluation metric to Evaluate the disentangled image manipulation. We verify the effectiveness of our method on the challenging face editing task. Extensive experiments show that the proposed PPE frame-work achieves much better quantitative and qualitative re-sults than the up-to-date StyleCLIP [31] baseline. Code is available at https://github.com/zipengxuc/PPE.

----

## [1757] A Style-aware Discriminator for Controllable Image Translation

**Authors**: *Kunhee Kim, Sanghun Park, Eunyeong Jeon, Taehun Kim, Daijin Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01770](https://doi.org/10.1109/CVPR52688.2022.01770)

**Abstract**:

Current image-to-image translations do not control the output domain beyond the classes used during training, nor do they interpolate between different domains well, leading to implausible results. This limitation largely arises because labels do not consider the semantic distance. To mitigate such problems, we propose a style-aware discriminator that acts as a critic as well as a style encoder to provide conditions. The style-aware discriminator learns a controllable style space using prototype-based self-supervised learning and simultaneously guides the generator. Experiments on multiple datasets verify that the proposed model outperforms current state-of-the-art image-to-image translation methods. In contrast with current methods, the proposed approach supports various applications, including style interpolation, content transplantation, and local image translation. The code is available at github.com/kunheek/style-aware-discriminator.

----

## [1758] Alleviating Semantics Distortion in Unsupervised Low-Level Image-to-Image Translation via Structure Consistency Constraint

**Authors**: *Jiaxian Guo, Jiachen Li, Huan Fu, Mingming Gong, Kun Zhang, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01771](https://doi.org/10.1109/CVPR52688.2022.01771)

**Abstract**:

Unsupervised image-to-image (I21) translation aims to learn a domain mapping function that can preserve the semantics of the input images without paired data. However, because the underlying semantics distributions in the source and target domains are often mismatched, current distribution matching-based methods may distort the semantics when matching distributions, resulting in the inconsistency between the input and translated images, which is known as the semantics distortion problem. In this paper, we focus on the low-level I21 translation, where the structure of images is highly related to their semantics. To alleviate semantic distortions in such translation tasks without paired supervision, we propose a novel I21 translation constraint, called Structure Consistency Constraint (SCC), to promote the consistency of image structures by reducing the randomness of color transformation in the translation process. To facilitate estimation and maximization of SCC, we propose an approximate representation of mutual information called relative Squared-loss Mutual Information (rSMI) that enjoys efficient analytic solutions. Our SCC can be easily incorporated into most existing translation models. Quantitative and qualitative comparisons on a range of low-level I21 translation tasks show that translation models with SCC outperform the original models by a significant margin with little additional computational and memory costs.

----

## [1759] Exploring Patch-wise Semantic Relation for Contrastive Learning in Image-to-Image Translation Tasks

**Authors**: *Chanyong Jung, Gihyun Kwon, Jong Chul Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01772](https://doi.org/10.1109/CVPR52688.2022.01772)

**Abstract**:

Recently, contrastive learning-based image translation methods have been proposed, which contrasts different spatial locations to enhance the spatial correspondence. However, the methods often ignore the diverse semantic relation within the images. To address this, here we propose a novel semantic relation consistency (SRC) regularization along with the decoupled contrastive learning, which utilize the diverse semantics by focusing on the heterogeneous semantics between the image patches of a single image. To further improve the performance, we present a hard negative mining by exploiting the semantic relation. We verified our method for three tasks: single-modal and multi-modal image translations, and GAN compression task for image translation. Experimental results confirmed the state-of-art performance of our method in all the three tasks.

----

## [1760] FlexIT: Towards Flexible Semantic Image Translation

**Authors**: *Guillaume Couairon, Asya Grechka, Jakob Verbeek, Holger Schwenk, Matthieu Cord*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01773](https://doi.org/10.1109/CVPR52688.2022.01773)

**Abstract**:

Deep generative models, like GANs, have considerably improved the state of the art in image synthesis, and are able to generate near photo-realistic images in structured domains such as human faces. Based on this success, recent work on image editing proceeds by projecting images to the GAN latent space and manipulating the latent vector. However, these approaches are limited in that only images from a narrow domain can be transformed, and with only a limited number of editing operations. We propose FlexIT, a novel method which can take any input image and a user-defined text instruction for editing. Our method achieves flexible and natural editing, pushing the limits of semantic image translation. First, FlexIT combines the input image and text into a single target point in the CLIP multimodal embedding space. Via the latent space of an autoencoder, we iteratively transform the input image toward the target point, ensuring coherence and quality with a variety of novel regularization terms. We propose an evaluation protocol for semantic image translation, and thoroughly evaluate our method on ImageNet. Code will be available at https://github.com/facebookresearch/SemanticImageTranslation/.

----

## [1761] Modulated Contrast for Versatile Image Synthesis

**Authors**: *Fangneng Zhan, Jiahui Zhang, Yingchen Yu, Rongliang Wu, Shijian Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01774](https://doi.org/10.1109/CVPR52688.2022.01774)

**Abstract**:

Perceiving the similarity between images has been a long-standing and fundamental problem underlying various visual generation tasks. Predominant approaches measure the inter-image distance by computing pointwise absolute deviations, which tends to estimate the median of instance distributions and leads to blurs and artifacts in the generated images. This paper presents MoNCE, a versatile metric that introduces image contrast to learn a calibrated metric for the perception of multifaceted inter-image distances. Unlike vanilla contrast which indiscriminately pushes negative samples from the anchor regardless of their similarity, we propose to re-weight the pushing force of negative samples adaptively according to their similarity to the anchor, which facilitates the contrastive learning from informative negative samples. Since multiple patch-level contrastive objectives are involved in image distance measurement, we introduce optimal transport in MoNCE to modulate the pushing force of negative samples collaboratively across multiple contrastive objectives. Extensive experiments over multiple image translation tasks show that the proposed MoNCE outperforms various prevailing metrics substantially. The code is available at MoNCE.

----

## [1762] QS-Attn: Query-Selected Attention for Contrastive Learning in I2I Translation

**Authors**: *Xueqi Hu, Xinyue Zhou, Qiusheng Huang, Zhengyi Shi, Li Sun, Qingli Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01775](https://doi.org/10.1109/CVPR52688.2022.01775)

**Abstract**:

Unpaired image-to-image (I2I) translation often requires to maximize the mutual information between the source and the translated images across different domains, which is critical for the generator to keep the source content and prevent it from unnecessary modifications. The self-supervised contrastive learning has already been successfully applied in the I2I. By constraining features from the same location to be closer than those from different ones, it implicitly ensures the result to take content from the source. However, previous work uses the features from random locations to impose the constraint, which may not be appropriate since some locations contain less information of source domain. Moreover, the feature itself does not reflect the relation with others. This paper deals with these problems by intentionally selecting significant anchor points for contrastive learning. We design a query-selected attention (QS-Attn) module, which compares feature distances in the source domain, giving an attention matrix with a probability distribution in each row. Then we select queries according to their measurement of significance, computed from the distribution. The selected ones are regarded as anchors for contrastive loss. At the same time, the reduced attention matrix is employed to route features in both domains, so that source relations maintain in the synthesis. We validate our proposed method in three different I2I datasets, showing that it increases the image quality with-out adding learnable parameters. Codes are available at https://github.com/sapphire497/query-selected-attention.

----

## [1763] Self-Supervised Dense Consistency Regularization for Image-to-Image Translation

**Authors**: *Minsu Ko, Eunju Cha, Sungjoo Suh, Huijin Lee, Jae-Joon Han, Jinwoo Shin, Bohyung Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01776](https://doi.org/10.1109/CVPR52688.2022.01776)

**Abstract**:

Unsupervised image-to-image translation has gained considerable attention due to recent impressive advances in generative adversarial networks (GANs). This paper presents a simple but effective regularization technique for improving GAN-based image-to-image translation. To generate images with realistic local semantics and structures, we propose an auxiliary self-supervision loss that enforces point-wise consistency of the overlapping region between a pair of patches cropped from a single real image during training the discriminator of a GAN. Our experiment shows that the proposed dense consistency regularization improves performance substantially on various image-to-image translation scenarios. It also leads to extra performance gains through the combination with instance-level regularization methods. Furthermore, we verify that the proposed model captures domain-specific characteristics more effectively with only a small fraction of training data.

----

## [1764] Maximum Spatial Perturbation Consistency for Unpaired Image-to-Image Translation

**Authors**: *Yanwu Xu, Shaoan Xie, Wenhao Wu, Kun Zhang, Mingming Gong, Kayhan Batmanghelich*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01777](https://doi.org/10.1109/CVPR52688.2022.01777)

**Abstract**:

Unpaired image-to-image translation (I2I) is an ill-posed problem, as an infinite number of translation functions can map the source domain distribution to the target distribution. Therefore, much effort has been put into designing suitable constraints, e.g., cycle consistency (CycleGAN), geometry consistency (GCGAN), and contrastive learning-based constraints (CUTGAN), that help better pose the problem. However, these well-known constraints have limitations: (1) they are either too restrictive or too weak for specific I2I tasks; (2) these methods result in content distortion when there is a significant spatial variation between the source and target domains. This paper proposes a universal regularization technique called maximum spatial perturbation consistency (MSPC), which enforces a spatial perturbation function 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(T)$</tex>
 and the translation operator 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(G)$</tex>
 to be commutative (i.e., 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$T\circ G=G\circ T)$</tex>
. In addition, we introduce two adversarial training components for learning the spatial perturbation function. The first one lets 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$T$</tex>
 compete with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$G$</tex>
 to achieve maximum perturbation. The second one lets 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$G$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$T$</tex>
 compete with discriminators to align the spatial variations caused by the change of object size, object distortion, background interruptions, etc. Our method outperforms the state-of-the-art methods on most I2I benchmarks. We also introduce a new benchmark, namely the front face to profile face dataset, to emphasize the underlying challenges of I2I for real-world applications. We finally perform ablation experiments to study the sensitivity of our method to the severity of spatial perturbation and its effectiveness for distribution alignment.

----

## [1765] InstaFormer: Instance-Aware Image-to-Image Translation with Transformer

**Authors**: *Soohyun Kim, Jongbeom Baek, Jihye Park, Gyeongnyeon Kim, Seungryong Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01778](https://doi.org/10.1109/CVPR52688.2022.01778)

**Abstract**:

We present a novel Transformer-based network architecture for instance-aware image-to-image translation, dubbed InstaFormer, to effectively integrate global- and instance-level information. By considering extracted content featuresfrom an image as tokens, our networks discover global consensus of content features by considering context information through a self-attention module in Transformers. By augmenting such tokens with an instance-level feature extracted from the content feature with respect to bounding box information, our framework is capable of learning an interaction between object instances and the global image, thus boosting the instance-awareness. We replace layer normalization (LayerNorm) in standard Transformers with adaptive instance normalization (AdaIN) to enable a multi-modal translation with style codes. In addition, to improve the instance-awareness and translation quality at object regions, we present an instance-level content contrastive loss defined between input and translated image. We conduct experiments to demonstrate the effectiveness of our InstaFormer over the latest methods and provide extensive ablation studies.

----

## [1766] Unsupervised Image-to-Image Translation with Generative Prior

**Authors**: *Shuai Yang, Liming Jiang, Ziwei Liu, Chen Change Loy*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01779](https://doi.org/10.1109/CVPR52688.2022.01779)

**Abstract**:

Unsupervised image-to-image translation aims to learn the translation between two visual domains without paired data. Despite the recent progress in image translation models, it remains challenging to build mappings between complex domains with drastic visual discrepancies. In this work, we present a novel framework, Generative Priorguided UNsupervised Image-to-image Translation (GP-UNIT), to improve the overall quality and applicability of the translation algorithm. Our key insight is to leverage the generative prior from pre-trained class-conditional GANs (e.g., BigGAN) to learn rich content correspondences across various domains. We propose a novel coarse-to-fine scheme: we first distill the generative prior to capture a robust coarse-level content representation that can link objects at an abstract semantic level, based on which finelevel content features are adaptively learned for more accurate multi-level content correspondences. Extensive experiments demonstrate the superiority of our versatile framework over state-of-the-art methods in robust, high-quality and diversified translations, even for challenging and distant domains. Code is available at https://github.com/williamyang1991/GP-UNIT.

----

## [1767] StylizedNeRF: Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D Mutual Learning

**Authors**: *Yihua Huang, Yue He, Yu-Jie Yuan, Yu-Kun Lai, Lin Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01780](https://doi.org/10.1109/CVPR52688.2022.01780)

**Abstract**:

3D scene stylization aims at generating stylized images of the scene from arbitrary novel views following a given set of style examples, while ensuring consistency when rendered from different views. Directly applying methods for image or video stylization to 3D scenes cannot achieve such consistency. Thanks to recently proposed neural radiance fields (NeRF), we are able to represent a 3D scene in a consistent way. Consistent 3D scene stylization can be effectively achieved by stylizing the corresponding NeRF. However, there is a significant domain gap between style examples which are 2D images and NeRF which is an implicit volumetric representation. To address this problem, we propose a novel mutual learning framework for 3D scene stylization that combines a 2D image stylization network and NeRF to fuse the stylization ability of 2D stylization network with the 3D consistency of NeRF. We first pre-train a standard NeRF of the 3D scene to be stylized and replace its color prediction module with a style network to obtain a stylized NeRF. It is followed by distilling the prior knowledge of spatial consistency from NeRF to the 2D stylization network through an introduced consistency loss. We also introduce a mimic loss to supervise the mutual learning of the NeRF style module and fine-tune the 2D stylization decoder. In order to further make our model handle ambiguities of 2D stylization results, we introduce learnable latent codes that obey the probability distributions conditioned on the style. They are attached to training samples as conditional inputs to better learn the style module in our novel stylized NeRF. Experimental results demonstrate that our method is superior to existing approaches in both visual quality and long-range consistency.

----

## [1768] NeRF-Editing: Geometry Editing of Neural Radiance Fields

**Authors**: *Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, Lin Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01781](https://doi.org/10.1109/CVPR52688.2022.01781)

**Abstract**:

Implicit neural rendering, especially Neural Radiance Field (NeRF), has shown great potential in novel view synthesis of a scene. However, current NeRF-based methods cannot enable users to perform user-controlled shape deformation in the scene. While existing works have proposed some approaches to modify the radiance field according to the user's constraints, the modification is limited to color editing or object translation and rotation. In this paper, we propose a method that allows users to perform controllable shape deformation on the implicit representation of the scene, and synthesizes the novel view images of the edited scene without re-training the network. Specifically, we establish a correspondence between the extracted explicit mesh representation and the implicit neural representation of the target scene. Users can first utilize well-developed mesh-based deformation methods to deform the mesh representation of the scene. Our method then utilizes user edits from the mesh representation to bend the camera rays by introducing a tetrahedra mesh as a proxy, obtaining the rendering results of the edited scene. Extensive experiments demonstrate that our framework can achieve ideal editing results not only on synthetic data, but also on real scenes captured by users.

----

## [1769] GeoNeRF: Generalizing NeRF with Geometry Priors

**Authors**: *Mohammad Mahdi Johari, Yann Lepoittevin, François Fleuret*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01782](https://doi.org/10.1109/CVPR52688.2022.01782)

**Abstract**:

We present GeoNeRF, a generalizable photorealistic novel view synthesis method based on neural radiance fields. Our approach consists of two main stages: a ge-ometry reasoner and a renderer. To render a novel view, the geometry reasoner first constructs cascaded cost volumes for each nearby source view. Then, using a Transformer- based attention mechanism and the cascaded cost volumes, the renderer infers geometry and appearance, and ren-ders detailed images via classical volume rendering techniques. This architecture, in particular, allows sophis-ticated occlusion reasoning, gathering information from consistent source views. Moreover, our method can eas-ily be fine-tuned on a single scene, and renders com-petitive results with per-scene optimized neural rendering methods with a fraction of computational cost. Ex-periments show that GeoNeRF outperforms state-of-the- art generalizable neural rendering models on various syn-thetic and real datasets. Lastly, with a slight modification to the geometry reasoner, we also propose an alter-native model that adapts to RGBD images. This model di-rectly exploits the depth information often available thanks to depth sensors. The implementation code is available at https://www.idiap.ch/paper/geonerf.

----

## [1770] Ray Priors through Reprojection: Improving Neural Radiance Fields for Novel View Extrapolation

**Authors**: *Jian Zhang, Yuanqing Zhang, Huan Fu, Xiaowei Zhou, Bowen Cai, Jinchi Huang, Rongfei Jia, Binqiang Zhao, Xing Tang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01783](https://doi.org/10.1109/CVPR52688.2022.01783)

**Abstract**:

Neural Radiance Fields (NeRF) [22] have emerged as a potent paradigm for representing scenes and synthesizing photo-realistic images. A main limitation of conventional NeRFs is that they often fail to produce high-quality renderings under novel viewpoints that are significantly different from the training viewpoints. In this paper, instead of ex-ploiting few-shot image synthesis, we study the novel view extrapolation setting that (1) the training images can well describe an object, and (2) there is a notable discrepancy between the training and test viewpoints' distributions. We present RapNeRF (RAy Priors) as a solution. Our insight is that the inherent appearances of a 3D surface's arbitrary visible projections should be consistent. We thus propose a random ray casting policy that allows training unseen views using seen views. Furthermore, we show that a ray atlas pre-computed from the observed rays' viewing directions could further enhance the rendering quality for ex-trapolated views. A main limitation is that RapNeRF would remove the strong view-dependent effects because it lever-ages the multi-view consistency property.

----

## [1771] AR-NeRF: Unsupervised Learning of Depth and Defocus Effects from Natural Images with Aperture Rendering Neural Radiance Fields

**Authors**: *Takuhiro Kaneko*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01784](https://doi.org/10.1109/CVPR52688.2022.01784)

**Abstract**:

Fully unsupervised 3D representation learning has gained attention owing to its advantages in data collection. A successful approach involves a viewpoint-aware approach that learns an image distribution based on generative models (e.g., generative adversarial networks (GANs)) while generating various view images based on 3D-aware models (e.g., neural radiance fields (NeRFs)). However, they require images with various views for training, and consequently, their application to datasets with few or limited viewpoints remains a challenge. As a complementary approach, an aperture rendering GAN (AR-GAN) that employs a defocus cue was proposed. However, an AR-GAN is a CNN-based model and represents a defocus independently from a viewpoint change despite its high correlation, which is one of the reasons for its performance. As an alternative to an AR-GAN, we propose an aperture rendering NeRF (AR-NeRF), which can utilize viewpoint and defocus cues in a unified manner by representing both factors in a common ray-tracing framework. Moreover, to learn defocus-aware and defocus-independent representations in a disentangled manner, we propose aperture randomized training, for which we learn to generate images while randomizing the aperture size and latent codes independently. During our experiments, we applied AR-NeRF to various natural image datasets, including flower, bird, and face images, the results of which demonstrate the utility of AR-NeRF for un-supervised learning of the depth and defocus effects.

----

## [1772] HDR-NeRF: High Dynamic Range Neural Radiance Fields

**Authors**: *Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Xuan Wang, Qing Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01785](https://doi.org/10.1109/CVPR52688.2022.01785)

**Abstract**:

We present High Dynamic Range Neural Radiance Fields (HDR-NeRF) to recover an HDR radiance field from a set of low dynamic range (LDR) views with different exposures. Using the HDR-NeRF, we are able to generate both novel HDR views and novel LDR views under different exposures. The key to our method is to model the simplified physical imaging process, which dictates that the radiance of a scene point transforms to a pixel value in the LDR image with two implicit functions: a radiance field and a tone mapper. The radiance field encodes the scene radiance (values vary from 0 to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$+\infty$</tex>
), which outputs the density and radiance of a ray by giving corresponding ray origin and ray direction. The tone mapper models the mapping process that a ray hitting on the camera sensor becomes a pixel value. The color of the ray is predicted by feeding the radiance and the corresponding exposure time into the tone mapper. We use the classic volume rendering technique to project the output radiance, colors and densities into HDR and LDR images, while only the input LDR images are used as the supervision. We collect a new forward-facing HDR dataset to evaluate the proposed method. Experimental results on synthetic and real-world scenes validate that our method can not only accurately control the exposures of synthesized views but also render views with a high dynamic range.

----

## [1773] NeRFReN: Neural Radiance Fields with Reflections

**Authors**: *Yuan-Chen Guo, Di Kang, Linchao Bao, Yu He, Song-Hai Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01786](https://doi.org/10.1109/CVPR52688.2022.01786)

**Abstract**:

Neural Radiance Fields (NeRF) has achieved unprece-dented view synthesis quality using coordinate-based neu-ral scene representations. However, NeRF's view depen-dency can only handle simple reflections like highlights but cannot deal with complex reflections such as those from glass and mirrors. In these scenarios, NeRF models the virtual image as real geometries which leads to inaccurate depth estimation, and produces blurry renderings when the multi-view consistency is violated as the reflected objects may only be seen under some of the viewpoints. To over-come these issues, we introduce NeRFReN, which is built upon NeRF to model scenes with reflections. Specifically, we propose to split a scene into transmitted and reflected components, and model the two components with separate neural radiance fields. Considering that this decomposition is highly under-constrained, we exploit geometric priors and apply carefully-designed training strategies to achieve reasonable decomposition results. Experiments on various self-captured scenes show that our method achieves high-quality novel view synthesis and physically sound depth es-timation results while enabling scene editing applications.

----

## [1774] Neural Point Light Fields

**Authors**: *Julian Ost, Issam Laradji, Alejandro Newell, Yuval Bahat, Felix Heide*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01787](https://doi.org/10.1109/CVPR52688.2022.01787)

**Abstract**:

We introduce Neural Point Light Fields that represent scenes implicitly with a light field living on a sparse point cloud. Combining differentiable volume rendering with learned implicit density representations has made it possible to synthesize photo-realistic images for novel views of small scenes. As neural volumetric rendering methods require dense sampling of the underlying functional scene representation, at hundreds of samples along a ray cast through the volume, they are fundamentally limited to small scenes with the same objects projected to hundreds of training views. Promoting sparse point clouds to neural implicit light fields allows us to represent large scenes effectively with only a single radiance evaluation per ray. These point light fields are as a function of the ray direction, and local point feature neighborhood, allowing us to interpolate the light field conditioned training images without dense object coverage and parallax. We assess the proposed method for novel view synthesis on large driving scenarios, where we synthesize realistic unseen views that existing implicit approaches fail to represent. We validate that Neural Point Light Fields make it possible to predict videos along unseen trajectories previously only feasible to generate by explicitly modeling the scene.

----

## [1775] 3D-aware Image Synthesis via Learning Structural and Textural Representations

**Authors**: *Yinghao Xu, Sida Peng, Ceyuan Yang, Yujun Shen, Bolei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01788](https://doi.org/10.1109/CVPR52688.2022.01788)

**Abstract**:

Making generative models 3D-aware bridges the 2D image space and the 3D physical world yet remains challenging. Recent attempts equip a Generative Adversarial Network (GAN) with a Neural Radiance Field (NeRF), which maps 3D coordinates to pixel values, as a 3D prior. However, the implicit function in NeRF has a very local receptive field, making the generator hard to become aware of the global structure. Meanwhile, NeRF is built on volume rendering which can be too costly to produce high-resolution results, increasing the optimization difficulty. To alleviate these two problems, we propose a novel framework, termed as VolumeGAN, for high-fidelity 3D-aware image synthesis, through explicitly learning a structural representation and a textural representation. We first learn a feature volume to represent the underlying structure, which is then converted to a feature field using a NeRF-like model. The feature field is further accumulated into a 2D feature map as the textural representation, followed by a neural renderer for appearance synthesis. Such a design enables independent control of the shape and the appearance. Project page is at https://genforce.github.io/volumegan.

----

## [1776] GIRAFFE HD: A High-Resolution 3D-aware Generative Model

**Authors**: *Yang Xue, Yuheng Li, Krishna Kumar Singh, Yong Jae Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01789](https://doi.org/10.1109/CVPR52688.2022.01789)

**Abstract**:

3D-aware generative models have shown that the introduction of 3D information can lead to more controllable image generation. In particular, the current state-of-the-art model GIRAFFE [38] can control each object's rotation, translation, scale, and scene camera pose without corresponding supervision. However, GIRAFFE only operates well when the image resolution is low. We propose GIRAFFE HD, a high-resolution 3D-aware generative model that inherits all of GIRAFFE's controllable features while generating high-quality, high-resolution images (512
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
resolution and above). The key idea is to leverage a style- based neural renderer, and to independently generate the foreground and background to force their disentanglement while imposing consistency constraints to stitch them together to composite a coherent final image. We demonstrate state-of-the-art 3D controllable high-resolution image generation on multiple natural image datasets.

----

## [1777] Multi-View Consistent Generative Adversarial Networks for 3D-aware Image Synthesis

**Authors**: *Xuanmeng Zhang, Zhedong Zheng, Daiheng Gao, Bang Zhang, Pan Pan, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01790](https://doi.org/10.1109/CVPR52688.2022.01790)

**Abstract**:

3D-aware image synthesis aims to generate images of objects from multiple views by learning a 3D representation. However, one key challenge remains: existing approaches lack geometry constraints, hence usually fail to generate multi-view consistent images. To address this challenge, we propose Multi-View Consistent Generative Adversarial Networks (MVCGAN) for high-quality 3D-aware image synthesis with geometry constraints. By leveraging the underlying 3D geometry information of generated images, i.e., depth and camera transformation matrix, we explicitly establish stereo correspondence between views to perform multi-view joint optimization. In particular, we enforce the photometric consistency between pairs of views and integrate a stereo mixup mechanism into the training process, encouraging the model to reason about the correct 3D shape. Besides, we design a two-stage training strategy with feature-level multi-view joint optimization to improve the image quality. Extensive experiments on three datasets demonstrate that MVCGAN achieves the state-of-the-art performance for 3D-aware image synthesis.

----

## [1778] Bi-level Doubly Variational Learning for Energy-based Latent Variable Models

**Authors**: *Ge Kan, Jinhu Lü, Tian Wang, Baochang Zhang, Aichun Zhu, Lei Huang, Guodong Guo, Hichem Snoussi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01791](https://doi.org/10.1109/CVPR52688.2022.01791)

**Abstract**:

Energy-based latent variable models (EBLVMs) are more expressive than conventional energy-based models. However, its potential on visual tasks are limited by its training process based on maximum likelihood estimate that requires sampling from two intractable distributions. In this paper, we propose Bi-level doubly variational learning (BiDVL), which is based on a new bi-level optimization framework and two tractable variational distributions to facilitate learning EBLVMs. Particularly, we lead a decoupled EBLVM consisting of a marginal energy-based distribution and a structural posterior to handle the difficulties when learning deep EBLVMs on images. By choosing a symmetric KL divergence in the lower level of our framework, a compact BiDVL for visual tasks can be obtained. Our model achieves impressive image generation performance over related works. It also demonstrates the significant capacity of testing image reconstruction and out-of-distribution detection.

----

## [1779] High-Resolution Image Harmonization via Collaborative Dual Transformations

**Authors**: *Wenyan Cong, Xinhao Tao, Li Niu, Jing Liang, Xuesong Gao, Qihao Sun, Liqing Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01792](https://doi.org/10.1109/CVPR52688.2022.01792)

**Abstract**:

Given a composite image, image harmonization aims to adjust the foreground to make it compatible with the background. High-resolution image harmonization is in high demand, but still remains unexplored. Conventional image harmonization methods learn global RGB-to-RGB transformation which could effortlessly scale to high resolution, but ignore diverse local context. Recent deep learning methods learn the dense pixel-to-pixel transformation which could generate harmonious outputs, but are highly constrained in low resolution. In this work, we propose a high-resolution image harmonization network with Collaborative Dual Transformation (CDTNet) to combine pixel-to-pixel transformation and RGB-to-RGB transformation coherently in an end-to-end network. Our CDTNet consists of a low-resolution generator for pixel-to-pixel transformation, a color mapping module for RGB-to-RGB transformation, and a refinement module to take advantage of both. Extensive experiments on high-resolution bench-mark dataset and our created high-resolution real composite images demonstrate that our CDTNet strikes a good balance between efficiency and effectiveness. Our used datasets can be found in https://github.com/bcmi/CDTNet-High-Resolution-Image-Harmonization.

----

## [1780] Brain-Supervised Image Editing

**Authors**: *Keith M. Davis, Carlos de la Torre-Ortiz, Tuukka Ruotsalo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01793](https://doi.org/10.1109/CVPR52688.2022.01793)

**Abstract**:

Despite recent advances in deep neural models for semantic image editing, present approaches are dependent on explicit human input. Previous work assumes the availability of manually curated datasets for supervised learning, while for unsupervised approaches the human inspection of discovered components is required to identify those which modify worthwhile semantic features. Here, we present a novel alternative: the utilization of brain responses as a supervision signal for learning semantic feature representations. Participants 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(N=30)$</tex>
 in a neurophysiological experiment were shown artificially generated faces and instructed to look for a particular semantic feature, such as “old” or “smiling”, while their brain responses were recorded via electroencephalography (EEG). Using supervision signals inferred from these responses, semantic features within the latent space of a generative adversarial network (GAN) were learned and then used to edit semantic features of new images. We show that implicit brain supervision achieves comparable semantic image editing performance to explicit manual labeling. This work demonstrates the feasibility of utilizing implicit human reactions recorded via brain-computer interfaces for semantic image editing and interpretation.

----

## [1781] De-rendering 3D Objects in the Wild

**Authors**: *Felix Wimbauer, Shangzhe Wu, Christian Rupprecht*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01794](https://doi.org/10.1109/CVPR52688.2022.01794)

**Abstract**:

With increasing focus on augmented and virtual reality (XR) applications comes the demand for algorithms that can lift objects from images into representations that are suitable for a wide variety of related 3D tasks. Large-scale deployment of XR devices and applications means that we cannot solely rely on supervised learning, as collecting and annotating data for the unlimited variety of objects in the real world is infeasible. We present a weakly supervised method that is able to decompose a single image of an object into shape (depth and normals), material (albedo, reflectivity and shininess) and global lighting parameters. For training, the method only relies on a rough initial shape estimate of the training objects to bootstrap the learning process. This shape supervision can come for example from a pretrained depth network or-more generically-from a traditional structure-from-motion pipeline. In our experiments, we show that the method can successfully de-render 2D images into a decomposed 3D representation and generalizes to unseen object categories. Since in-the-wild evaluation is difficult due to the lack of ground truth data, we also introduce a photo-realistic synthetic test set that allows for quantitative evaluation. Please find our project page at: https://github.com/Brummi/derender3d

----

## [1782] Neural Fields as Learnable Kernels for 3D Reconstruction

**Authors**: *Francis Williams, Zan Gojcic, Sameh Khamis, Denis Zorin, Joan Bruna, Sanja Fidler, Or Litany*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01795](https://doi.org/10.1109/CVPR52688.2022.01795)

**Abstract**:

We present Neural Kernel Fields: a novel method for reconstructing implicit 3D shapes based on a learned kernel ridge regression. Our technique achieves state-of-the-art results when reconstructing 3D objects and large scenes from sparse oriented points, and can reconstruct shape categories outside the training set with almost no drop in accuracy. The core insight of our approach is that kernel methods are extremely effective for reconstructing shapes when the chosen kernel has an appropriate inductive bias. We thus factor the problem of shape reconstruction into two parts: (1) a backbone neural network which learns kernel parameters from data, and (2) a kernel ridge regression that fits the input points on-the-fly by solving a simple positive definite linear system using the learned kernel. As a result of this factorization, our reconstruction gains the benefits of data-driven methods under sparse point density while maintaining interpolatory behavior, which converges to the ground truth shape as input sampling density increases. Our experiments demonstrate a strong generalization capability to objects outside the train-set category and scanned scenes. Source code and pretrained models are available at https://nv-tlabs.github.io/nkf.

----

## [1783] HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing

**Authors**: *Yuval Alaluf, Omer Tov, Ron Mokady, Rinon Gal, Amit Bermano*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01796](https://doi.org/10.1109/CVPR52688.2022.01796)

**Abstract**:

The inversion of real images into StyleGAN's latent space is a well-studied problem. Nevertheless, applying existing approaches to real-world scenarios remains an open challenge, due to an inherent trade-off between reconstruction and editability: latent space regions which can accurately represent real images typically suffer from degraded semantic control. Recent work proposes to mitigate this trade-off by fine-tuning the generator to add the target image to well-behaved, editable regions of the latent space. While promising, this fine-tuning scheme is impractical for prevalent use as it requires a lengthy training phase for each new image. In this work, we introduce this approach into the realm of encoder-based inversion. We propose HyperStyle, a hypernetwork that learns to modulate StyleGAN's weights to faithfully express a given image in editable regions of the latent space. A naive modulation approach would require training a hypernetwork with over three billion parameters. Through careful network design, we reduce this to be in line with existing encoders. HyperStyle yields reconstructions comparable to those of optimization techniques with the near real-time inference capabilities of encoders. Lastly, we demonstrate HyperStyle's effectiveness on several applications beyond the inversion task, including the editing of out-of-domain images which were never seen during training. Code is available on our project page: https://yuval-alaluf.github.io/hyperstyle/.

----

## [1784] 3PSDF: Three-Pole Signed Distance Function for Learning Surfaces with Arbitrary Topologies

**Authors**: *Weikai Chen, Cheng Lin, Weiyang Li, Bo Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01797](https://doi.org/10.1109/CVPR52688.2022.01797)

**Abstract**:

Recent advances in learning 3D shapes using neural implicit functions have achieved impressive results by breaking the previous barrier of resolution and diversity for varying topologies. However, most of such approaches are limited to closed surfaces as they require the space to be divided into inside and outside. More recent works based on unsigned distance function have been proposed to handle complex geometry containing both the open and closed surfaces. Nonetheless, as their direct outputs are point clouds, robustly obtaining high-quality meshing results from discrete points remains an open question. We present a novel learnable implicit representation, called three-pole signed distance function (3PSDF), that can represent non-watertight 3D shapes with arbitrary topologies while supporting easy field-to-mesh conversion using the classic Marching Cubes algorithm. The key to our method is the introduction of a new sign, the NULL sign, in addition to the conventional in and out labels. The existence of the null sign could stop the formation of a closed isosurface derived from the bisector of the in/out regions. Further, we propose a dedicated learning framework to effectively learn 3PSDF without worrying about the vanishing gradient due to the null labels. Experimental results show that our approach outperforms the previous state-of-the-art methods in a wide range of benchmarks both quantitatively and qualitatively.

----

## [1785] Pop-Out Motion: 3D-Aware Image Deformation via Learning the Shape Laplacian

**Authors**: *Jihyun Lee, Minhyuk Sung, Hyunjin Kim, Tae-Kyun Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01798](https://doi.org/10.1109/CVPR52688.2022.01798)

**Abstract**:

We propose a framework that can deform an object in a 2D image as it exists in 3D space. Most existing methods for 3D-aware image manipulation are limited to (1) only changing the global scene information or depth, or (2) manipulating an object of specific categories. In this paper, we present a 3D-aware image deformation method with minimal restrictions on shape category and deformation type. While our framework leverages 2D-to-3D reconstruction, we argue that reconstruction is not sufficient for realistic deformations due to the vulnerability to topological errors. Thus, we propose to take a supervised learning-based approach to predict the shape Laplacian of the underlying volume of a 3D reconstruction represented as a point cloud. Given the deformation energy calculated using the predicted shape Laplacian and user-defined deformation handles (e.g., keypoints), we obtain bounded biharmonic weights to model plausible handle-based image deformation. In the experiments, we present our results of deforming 2D character and clothed human images. We also quantitatively show that our approach can produce more accurate deformation weights compared to alternative methods (i.e., mesh reconstruction and point cloud Laplacian methods).

----

## [1786] Deep Image-based Illumination Harmonization

**Authors**: *Zhongyun Bao, Chengjiang Long, Gang Fu, Daquan Liu, Yuanzhen Li, Jiaming Wu, Chunxia Xiao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01799](https://doi.org/10.1109/CVPR52688.2022.01799)

**Abstract**:

Integrating a foreground object into a background scene with illumination harmonization is an important but challenging task in computer vision and augmented reality community. Existing methods mainly focus on foreground and background appearance consistency or the foreground object shadow generation, which rarely consider global appearance and illumination harmonization. In this paper, we formulate seamless illumination harmonization as an illumination exchange and aggregation problem. Specifically, we firstly apply a physically-based rendering method to construct a large-scale, high-quality dataset (named IH) for our task, which contains various types of foreground objects and background scenes with different lighting conditions. Then, we propose a deep image-based illumination harmonization GAN framework named DIH-GAN, which makes full use of a multi-scale attention mechanism and illumination exchange strategy to directly infer mapping relationship between the inserted foreground object and the corresponding background scene. Meanwhile, we also use adversarial learning strategy to further refine the illumination harmonization result. Our method can not only achieve harmonious appearance and illumination for the foreground object but also can generate compelling shadow cast by the foreground object. Comprehensive experiments on both our IH dataset and real-world images show that our proposed DIH-GAN provides a practical and effective solution for image-based object illumination harmonization editing, and validate the superiority of our method against state-of-the-art methods. Our IH dataset is available at https://github.com/zhongyunbao/Dataset.

----

## [1787] PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes

**Authors**: *Yu-Ying Yeh, Zhengqin Li, Yannick Hold-Geoffroy, Rui Zhu, Zexiang Xu, Milos Hasan, Kalyan Sunkavalli, Manmohan Chandraker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01801](https://doi.org/10.1109/CVPR52688.2022.01801)

**Abstract**:

Most indoor 3D scene reconstruction methods focus on recovering 3D geometry and scene layout. In this work, we go beyond this to propose PhotoScene
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/ViLab-UCSD/PhotoScene, a framework that takes input image(s) of a scene along with approximately aligned CAD geometry (either reconstructed automatically or manually specified) and builds a photorealistic digital twin with high-quality materials and similar lighting. We model scene materials using procedural material graphs; such graphs represent photorealistic and resolution-independent materials. We optimize the parameters of these graphs and their texture scale and rotation, as well as the scene lighting to best match the input image via a differentiable rendering layer. We evaluate our technique on objects and layout reconstructions from ScanNet, SUN RGB-D and stock photographs, and demonstrate that our method reconstructs high-quality, fully relightable 3D scenes that can be re-rendered under arbitrary viewpoints, zooms and lighting.

----

## [1788] Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes

**Authors**: *Ka-Hei Hui, Ruihui Li, Jingyu Hu, Chi-Wing Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01802](https://doi.org/10.1109/CVPR52688.2022.01802)

**Abstract**:

This paper introduces a novel framework called DTNet for 3D mesh reconstruction and generation via Disentangled Topology. Beyond previous works, we learn a topology-aware neural template specific to each input then deform the template to reconstruct a detailed mesh while preserving the learned topology. One key insight is to decouple the complex mesh reconstruction into two sub-tasks: topology formulation and shape deformation. Thanks to the decoupling, DT-Net implicitly learns a disentangled representation for the topology and shape in the latent space. Hence, it can enable novel disentangled controls for supporting various shape generation applications, e.g., remix the topologies of 3D objects, that are not achievable by previous reconstruction works. Extensive experimental results demonstrate that our method
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/edward1997104/Neural-Template. is able to produce high-quality meshes, particularly with diverse topologies, as compared with the state-of-the-art methods.

----

## [1789] Neural Mesh Simplification

**Authors**: *Rolandos Alexandros Potamias, Stylianos Ploumpis, Stefanos Zafeiriou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01803](https://doi.org/10.1109/CVPR52688.2022.01803)

**Abstract**:

Despite the advent in rendering, editing and preprocessing methods of 3D meshes, their real-time execution remains still infeasible for large-scale meshes. To ease and accelerate such processes, mesh simplification methods have been introduced with the aim to reduce the mesh resolution while preserving its appearance. In this work we attempt to tackle the novel task of learnable and differentiable mesh simplification. Compared to traditional simplification approaches that collapse edges in a greedy iterative manner, we propose a fast and scalable method that simplifies a given mesh in one-pass. The proposed method unfolds in three steps. Initially, a subset of the input vertices is sampled using a sophisticated extension of random sampling. Then, we train a sparse attention network to propose candidate triangles based on the edge connectivity of the sampled vertices. Finally, a classification network estimates the probability that a candidate triangle will be included in the final mesh. The fast, lightweight and differentiable properties of the proposed method makes it possible to be plugged in every learnable pipeline without introducing a significant overhead. We evaluate both the sampled vertices and the generated triangles under several appearance error measures and compare its performance against several state-of-the-art baselines. Furthermore, we showcase that the running performance can be up to 10× faster than traditional methods.

----

## [1790] SkinningNet: Two-Stream Graph Convolutional Neural Network for Skinning Prediction of Synthetic Characters

**Authors**: *Albert Mosella-Montoro, Javier Ruiz Hidalgo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01804](https://doi.org/10.1109/CVPR52688.2022.01804)

**Abstract**:

This work presents SkinningNet, an end-to-end Two-Stream Graph Neural Network architecture that computes skinning weights from an input mesh and its associated skeleton, without making any assumptions on shape class and structure of the provided mesh. Whereas previous meth-ods pre-compute handcrafted features that relate the mesh and the skeleton or assume a fixed topology of the skeleton, the proposed method extracts this information in an end-to-end learnable fashion by jointly learning the best relationship between mesh vertices and skeleton joints. The proposed method exploits the benefits of the novel Multi-Aggregator Graph Convolution that combines the results of different aggregators during the summarizing step of the Message-Passing scheme, helping the operation to general-ize for unseen topologies. Experimental results demonstrate the effectiveness of the contributions of our novel architecture, with SkinningNet outperforming current state-of-the-art alternatives.

----

## [1791] CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation

**Authors**: *Aditya Sanghi, Hang Chu, Joseph G. Lambourne, Ye Wang, Chin-Yi Cheng, Marco Fumero, Kamal Rahimi Malekshan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01805](https://doi.org/10.1109/CVPR52688.2022.01805)

**Abstract**:

Generating shapes using natural language can enable new ways of imagining and creating the things around us. While significant recent progress has been made in text-to-image generation, text-to-shape generation remains a challenging problem due to the unavailability of paired text and shape data at a large scale. We present a simple yet effective method for zero-shot text-to-shape gener-ation that circumvents such data scarcity. Our proposed method, named CLIP-Forge, is based on a two-stage training process, which only depends on an unlabelled shape dataset and a pre-trained image-text network such as CLIP. Our method has the benefits of avoiding expensive inference time optimization, as well as the ability to generate multiple shapes for a given text. We not only demonstrate promising zero-shot generalization of the CLIP-Forge model qualitatively and quantitatively, but also provide extensive compar-ative evaluations to better understand its behavior.

----

## [1792] UNIST: Unpaired Neural Implicit Shape Translation Network

**Authors**: *Qimin Chen, Johannes Merz, Aditya Sanghi, Hooman Shayani, Ali Mahdavi-Amiri, Hao Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01806](https://doi.org/10.1109/CVPR52688.2022.01806)

**Abstract**:

We introduce UNIST, the first deep neural implicit model for general-purpose, unpaired shape-to-shape translation, in both 2D and 3D domains. Our model is built on autoencoding implicit fields, rather than point clouds which represents the state of the art. Furthermore, our translation network is trained to perform the task over a latent grid representation which combines the merits of both latent-space processing and position awareness, to not only enable drastic shape transforms but also well preserve spatial features and fine local details for natural shape translations. With the same network architecture and only dictated by the input domain pairs, our model can learn both style-preserving content alteration and content-preserving style transfer. We demonstrate the generality and quality of the translation results, and compare them to well-known baselines. Code is available at https://qiminchen.github.io/unist/.

----

## [1793] CoNeRF: Controllable Neural Radiance Fields

**Authors**: *Kacper Kania, Kwang Moo Yi, Marek Kowalski, Tomasz Trzcinski, Andrea Tagliasacchi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01807](https://doi.org/10.1109/CVPR52688.2022.01807)

**Abstract**:

We extend neural 3D representations to allow for intu-itive and interpretable user control beyond novel view ren-dering (i. e. camera control). We allow the user to annotate which part of the scene one wishes to control with just a small number of mask annotations in the training images. Our key idea is to treat the attributes as latent variables that are regressed by the neural network given the scene en-coding. This leads to afew-shot learning framework, where attributes are discovered automatically by the framework, when annotations are not provided. We apply our method to various scenes with different types of controllable attributes (e.g. expression control on human faces, or state control in movement of inanimate objects). Overall, we demonstrate, to the best of our knowledge, for the first time novel view and novel attribute re-rendering of scenes from a single video.

----

## [1794] Neural Points: Point Cloud Representation with Neural Fields for Arbitrary Upsampling

**Authors**: *Wanquan Feng, Jin Li, Hongrui Cai, Xiaonan Luo, Juyong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01808](https://doi.org/10.1109/CVPR52688.2022.01808)

**Abstract**:

In this paper, we propose Neural Points, a novel point cloud representation and apply it to the arbitrary-factored upsampling task. Different from traditional point cloud representation where each point only represents a position or a local plane in the 3D space, each point in Neural Points represents a local continuous geometric shape via neural fields. Therefore, Neural Points contain more shape information and thus have a stronger representation ability. Neural Points is trained with surface containing rich geometric details, such that the trained model has enough expression ability for various shapes. Specifically, we extract deep local features on the points and construct neural fields through the local isomorphism between the 2D parametric domain and the 3D local patch. In the final, local neural fields are integrated together to form the global surface. Experimental results show that Neural Points has powerful representation ability and demonstrate excellent robustness and generalization ability. With Neural Points, we can resample point cloud with arbitrary resolutions, and it outperforms the state-of-the-art point cloud upsampling methods. Code is available at https://github.com/WanquanF/NeuralPoints.

----

## [1795] Modeling Indirect Illumination for Inverse Rendering

**Authors**: *Yuanqing Zhang, Jiaming Sun, Xingyi He, Huan Fu, Rongfei Jia, Xiaowei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01809](https://doi.org/10.1109/CVPR52688.2022.01809)

**Abstract**:

Recent advances in implicit neural representations and differentiable rendering make it possible to simultaneously recover the geometry and materials of an object from multi-view RGB images captured under unknown static illumination. Despite the promising results achieved, indirect illumination is rarely modeled in previous methods, as it requires expensive recursive path tracing which makes the inverse rendering computationally intractable. In this paper, we propose a novel approach to efficiently recovering spatially-varying indirect illumination. The key insight is that indirect illumination can be conveniently derived from the neural radiance field learned from input images instead of being estimated jointly with direct illumination and materials. By properly modeling the indirect illumination and visibility of direct illumination, interreflection- and shadow-free albedo can be recovered. The experiments on both synthetic and real data demonstrate the superior performance of our approach compared to previous work and its capambility to synthesize realistic renderings under novel view-points and illumination. Our code and data are available at https://zju3dv.github.io/invrender/.

----

## [1796] Neural Head Avatars from Monocular RGB Videos

**Authors**: *Philip-William Grassal, Malte Prinzler, Titus Leistner, Carsten Rother, Matthias Nießner, Justus Thies*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01810](https://doi.org/10.1109/CVPR52688.2022.01810)

**Abstract**:

We present Neural Head Avatars, a novel neural representation that explicitly models the surface geometry and appearance of an animatable human avatar that can be used for teleconferencing in AR/VR or other applications in the movie or games industry that rely on a digital human.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
philgras.github.io/neural_head_avatars/neural_head_avatars.html Our representation can be learned from a monocular RGB portrait video that features a range of different expressions and views. Specifically, we propose a hybrid representation consisting of a morphable model for the coarse shape and expressions of the face, and two feed-forward networks, predicting vertex offsets of the underlying mesh as well as a view- and expression-dependent texture. We demonstrate that this representation is able to accurately extrapolate to unseen poses and view points, and generates natural expressions while providing sharp texture details. Compared to previous works on head avatars, our method provides a disentangled shape and appearance model of the complete human head (including hair) that is compatible with the standard graphics pipeline. Moreover, it quantitatively and qualitatively outperforms current state of the art in terms of reconstruction quality and novel-view synthesis.

----

## [1797] DeepCurrents: Learning Implicit Representations of Shapes with Boundaries

**Authors**: *David R. Palmer, Dmitriy Smirnov, Stephanie Wang, Albert Chern, Justin Solomon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01811](https://doi.org/10.1109/CVPR52688.2022.01811)

**Abstract**:

Recent techniques have been successful in reconstructing surfaces as level sets of learned functions (such as signed distance fields) parameterized by deep neural networks. Many of these methods, however, learn only closed surfaces and are unable to reconstruct shapes with boundary curves. We propose a hybrid shape representation that combines explicit boundary curves with implicit learned interiors. Using machinery from geometric measure theory, we parameterize currents using deep networks and use stochastic gradient descent to solve a minimal surface problem. By modifying the metric according to target geometry coming, e.g., from a mesh or point cloud, we can use this approach to represent arbitrary surfaces, learning implicitly defined shapes with explicitly defined boundary curves. We further demonstrate learning families of shapes jointly parameterized by boundary curves and latent codes.

----

## [1798] Escaping Data Scarcity for High-Resolution Heterogeneous Face Hallucination

**Authors**: *Yiqun Mei, Pengfei Guo, Vishal M. Patel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01812](https://doi.org/10.1109/CVPR52688.2022.01812)

**Abstract**:

In Heterogeneous Face Recognition (HFR), the objective is to match faces across two different domains such as visible and thermal. Large domain discrepancy makes HFR a difficult problem. Recent methods attempting to fill the gap via synthesis have achieved promising results, but their performance is still limited by the scarcity of paired training data. In practice, large-scale heterogeneous face data are often inaccessible due to the high cost of acquisition and annotation process as well as privacy regulations. In this paper, we propose a new face hallucination paradigm for HFR, which not only enables data-efficient synthesis but also allows to scale up model training without breaking any privacy policy. Unlike existing methods that learn face synthesis entirely from scratch, our approach is particularly designed to take advantage of rich and diverse facial priors from visible domain for more faithful hallucination. On the other hand, large-scale training is enabled by introducing a new federated learning scheme to allow institution-wise collaborations while avoiding explicit data sharing. Extensive experiments demonstrate the advantages of our approach in tackling HFR under current data limitations. In a unified framework, our method yields the state-of-the-art hallucination results on multiple HFR datasets.

----

## [1799] AnyFace: Free-style Text-to-Face Synthesis and Manipulation

**Authors**: *Jianxin Sun, Qiyao Deng, Qi Li, Muyi Sun, Min Ren, Zhenan Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01813](https://doi.org/10.1109/CVPR52688.2022.01813)

**Abstract**:

Existing text-to-image synthesis methods generally are only applicable to words in the training dataset. However, human faces are so variable to be described with limited words. So this paper proposes the first free-style text-to-face method namely AnyFace enabling much wider open world applications such as metaverse, social media, cosmetics, forensics, etc. AnyFace has a novel two-stream framework for face image synthesis and manipulation given arbitrary descriptions of the human face. Specifically, one stream performs text-to-face generation and the other conducts face image reconstruction. Facial text and image features are extracted using the CLIP (Contrastive Language-Image Pre-training) encoders. And a collaborative Cross Modal Distillation (CMD) module is designed to align the linguistic and visual features across these two streams. Furthermore, a Diverse Triplet Loss (DT loss) is developed to model fine-grained features and improve facial diversity. Extensive experiments on Multi-modal CelebA-HQ and CelebAText-HQ demonstrate significant advantages of AnyFace over state-of-the-art methods. AnyFace can achieve high-quality, high-resolution, and high-diversity face synthesis and manipulation results without any constraints on the number and content of input captions.

----



[Go to the previous page](CVPR-2022-list08.md)

[Go to the next page](CVPR-2022-list10.md)

[Go to the catalog section](README.md)