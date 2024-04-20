## [0] Pixel screening based intermediate correction for blind deblurring

**Authors**: *Meina Zhang, Yingying Fang, Guoxi Ni, Tieyong Zeng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00580](https://doi.org/10.1109/CVPR52688.2022.00580)

**Abstract**:

Blind deblurring has attracted much interest with its wide applications in reality. The blind deblurring problem is usually solved by estimating the intermediate kernel and the intermediate image alternatively, which will finally converge to the blurring kernel of the observed image. Numerous works have been proposed to obtain intermediate images with fewer undesirable artifacts by designing delicate regularization on the latent solution. However, these methods still fail while dealing with images containing saturations and large blurs. To address this problem, we propose an intermediate image correction method which utilizes Bayes posterior estimation to screen through the intermediate image and exclude those unfavorable pixels to reduce their influence for kernel estimation. Extensive experiments have proved that the proposed method can effectively improve the accuracy of the final derived kernel against the state-of-the-art methods on benchmark datasets by both quantitative and qualitative comparisons.

----

## [1] When Does Contrastive Visual Representation Learning Work?

**Authors**: *Elijah Cole, Xuan Yang, Kimberly Wilber, Oisin Mac Aodha, Serge J. Belongie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01434](https://doi.org/10.1109/CVPR52688.2022.01434)

**Abstract**:

Recent self-supervised representation learning techniques have largely closed the gap between supervised and unsupervised learning on ImageNet classification. While the particulars of pretraining on ImageNet are now relatively well understood, the field still lacks widely accepted best practices for replicating this success on other datasets. As a first step in this direction, we study contrastive self-supervised learning on four diverse large-scale datasets. By looking through the lenses of data quantity, data domain, data quality, and task granularity, we provide new insights into the necessary conditions for successful self-supervised learning. Our key findings include observations such as: (i) the benefit of additional pretraining data beyond 500k images is modest, (ii) adding pretraining images from another domain does not lead to more general representations, (iii) corrupted pretraining images have a disparate impact on supervised and self-supervised pretraining, and (iv) contrastive learning lags far behind supervised learning on finegrained visual classification tasks.

----

## [2] Large-Scale Pre-training for Person Re-identification with Noisy Labels

**Authors**: *Dengpan Fu, Dongdong Chen, Hao Yang, Jianmin Bao, Lu Yuan, Lei Zhang, Houqiang Li, Fang Wen, Dong Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00251](https://doi.org/10.1109/CVPR52688.2022.00251)

**Abstract**:

This paper aims to address the problem of pretraining for person re-identification (Re-ID) with noisy labels. To setup the pretraining task, we apply a simple online multi-object tracking system on raw videos of an existing un-labeled Re-ID dataset “LUPerson” and build the Noisy Labeled variant called “LUPerson-NL”. Since theses ID labels automatically derived from tracklets inevitably con-tain noises, we develop a large-scale Pre-training frame-work utilizing Noisy Labels (PNL), which consists of three learning modules: supervised Re-ID learning, prototype-based contrastive learning, and label-guided contrastive learning. In principle, joint learning of these three mod-ules not only clusters similar examples to one prototype, but also rectifies noisy labels based on the prototype as-signment. We demonstrate that learning directly from raw videos is a promising alternative for pre-training, which utilizes spatial and temporal correlations as weak super-vision. This simple pre-training task provides a scalable way to learn SOTA Re-ID representations from scratch on “LUPerson-NL” without bells and whistles. For example, by applying on the same supervised Re-ID method MGN, our pre-trained model improves the mAP over the unsu-pervised pre-training counterpart by 5.7%, 2.2%, 2.3% on CUHK03, DukeMTMC, and MSMT17 respectively. Under the small-scale or few-shot setting, the performance gain is even more significant, suggesting a better transferability of the learned representation. Code is available at https://github.com/DengpanFu/LUPerson-NL.

----

## [3] Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers

**Authors**: *Yunhui Guo, Xudong Wang, Yubei Chen, Stella X. Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00010](https://doi.org/10.1109/CVPR52688.2022.00010)

**Abstract**:

Hyperbolic space can naturally embed hierarchies, unlike Euclidean space. Hyperbolic Neural Networks (HNNs) exploit such representational power by lifting Euclidean features into hyperbolic space for classification, outperforming Euclidean neural networks (ENNs) on datasets with known semantic hierarchies. However, HNNs underperform ENNs on standard benchmarks without clear hierarchies, greatly restricting HNNs' applicability in practice. Our key insight is that HNNs' poorer general classification performance results from vanishing gradients during backpropagation, caused by their hybrid architecture connecting Euclidean features to a hyperbolic classifier. We propose an effective solution by simply clipping the Euclidean feature magnitude while training HNNs. Our experiments demonstrate that clipped HNNs become super-hyperbolic classifiers: They are not only consistently better than HNNs which already outperform ENNs on hierarchical data, but also on-par with ENNs on MNIST, CIFAR10, CIFAR100 and ImageNet benchmarks, with better adversarial robustness and out-of-distribution detection.

----

## [4] CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data

**Authors**: *Yunhui Guo, Haoran Guo, Stella X. Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00011](https://doi.org/10.1109/CVPR52688.2022.00011)

**Abstract**:

Hyperbolic space can naturally embed hierarchies that often exist in real-world data and semantics. While high-dimensional hyperbolic embeddings lead to better representations, most hyperbolic models utilize low-dimensional embeddings, due to non-trivial optimization and visualization of high-dimensional hyperbolic data. We propose CO-SNE, which extends the Euclidean space visualization tool, t-SNE, to hyperbolic space. Like t-SNE, it converts distances between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of high-dimensional data 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$X$</tex>
 and low-dimensional embedding 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Y$</tex>
. However, unlike Euclidean space, hyperbolic space is inhomogeneous: A volume could contain a lot more points at a location far from the origin. CO-SNE thus uses hyperbolic normal distributions for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$X$</tex>
 and hyperbolic Cauchy instead of t-SNE's Student's t-distribution for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Y$</tex>
, and it additionally seeks to preserve 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$X$</tex>
's individual distances to the Origin in 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Y$</tex>
. We apply CO-SNE to naturally hyperbolic data and supervisedly learned hyperbolic features. Our results demonstrate that CO-SNE deflates high-dimensional hyperbolic data into a low-dimensional space without losing their hyperbolic characteristics, significantly outperforming popular visualization tools such as PCA, t-SNE, UMAP, and HoroPCA which is also designed for hyperbolic data.

----

## [5] Efficient Deep Embedded Subspace Clustering

**Authors**: *Jinyu Cai, Jicong Fan, Wenzhong Guo, Shiping Wang, Yunhe Zhang, Zhao Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00012](https://doi.org/10.1109/CVPR52688.2022.00012)

**Abstract**:

Recently deep learning methods have shown significant progress in data clustering tasks. Deep clustering methods (including distance-based methods and subspace-based methods) integrate clustering and feature learning into a unified framework, where there is a mutual promotion between clustering and representation. However, deep subspace clustering methods are usually in the framework of self-expressive model and hence have quadratic time and space complexities, which prevents their applications in large-scale clustering and real-time clustering. In this paper, we propose a new mechanism for deep clustering. We aim to learn the subspace bases from deep representation in an iterative refining manner while the refined subspace bases help learning the representation of the deep neural networks in return. The proposed method is out of the self-expressive framework, scales to the sample size linearly, and is applicable to arbitrarily large datasets and online clustering scenarios. More importantly, the clustering accuracy of the proposed method is much higher than its competitors. Extensive comparison studies with state-of-the-art clustering approaches on benchmark datasets demonstrate the superiority of the proposed method.

----

## [6] Noise Is Also Useful: Negative Correlation-Steered Latent Contrastive Learning

**Authors**: *Jiexi Yan, Lei Luo, Chenghao Xu, Cheng Deng, Heng Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00013](https://doi.org/10.1109/CVPR52688.2022.00013)

**Abstract**:

How to effectively handle label noise has been one of the most practical but challenging tasks in Deep Neural Networks (DNNs). Recent popular methods for training DNNs with noisy labels mainly focus on directly filtering out samples with low confidence or repeatedly mining valuable information from low-confident samples. However, they cannot guarantee the robust generalization of models due to the ignorance of useful information hidden in noisy data. To address this issue, we propose a new effective method named as LaCoL (Latent Contrastive Learning) to leverage the negative correlations from the noisy data. Specifically, in label space, we exploit the weakly-augmented data to filter samples and adopt classification loss on strong augmentations of the selected sample set, which can preserve the training diversity. While in metric space, we utilize weakly-supervised contrastive learning to excavate these negative correlations hidden in noisy data. Moreover, a cross-space similarity consistency regularization is provided to constrain the gap between label space and metric space. Extensive experiments have validated the superiority of our approach over existing state-of-the-art methods.

----

## [7] Active Learning for Open-set Annotation

**Authors**: *Kun-Peng Ning, Xun Zhao, Yu Li, Sheng-Jun Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00014](https://doi.org/10.1109/CVPR52688.2022.00014)

**Abstract**:

Existing active learning studies typically work in the closed-set setting by assuming that all data examples to be labeled are drawn from known classes. However, in real annotation tasks, the unlabeled data usually contains a large amount of examples from unknown classes, resulting in the failure of most active learning methods. To tackle this open-set annotation (OSA) problem, we propose a new active learning framework called LfOSA, which boosts the classification performance with an effective sampling strategy to precisely detect examples from known classes for annotation. The LfOSA framework introduces an auxiliary network to model the perexample max activation value (MAV) distribution with a Gaussian Mixture Model, which can dynamically select the examples with highest probability from known classes in the unlabeled set. Moreover, by reducing the temperature T of the loss function, the detection model will be further optimized by exploiting both known and unknown supervision. The experimental results show that the proposed method can significantly improve the selection quality of known classes, and achieve higher classification accuracy with lower annotation cost than state-of-the-art active learning methods. To the best of our knowledge, this is the first work of active learning for open-set annotation.

----

## [8] Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training

**Authors**: *Theodoros Tsiligkaridis, Jay Roberts*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00015](https://doi.org/10.1109/CVPR52688.2022.00015)

**Abstract**:

Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical frameworkfor adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the distortion of l-inf FW attacks (the attack's l–2 norm). Specifically, we analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that l-inf attacks against robust models achieve near maximal l-2 distortion, while standard networks have lower distortion. Furthermore, it is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from the more popular Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and improves closing the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

----

## [9] Robust Optimization as Data Augmentation for Large-scale Graphs

**Authors**: *Kezhi Kong, Guohao Li, Mucong Ding, Zuxuan Wu, Chen Zhu, Bernard Ghanem, Gavin Taylor, Tom Goldstein*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00016](https://doi.org/10.1109/CVPR52688.2022.00016)

**Abstract**:

Data augmentation helps neural networks generalize better by enlarging the training set, but it remains an open question how to effectively augment graph data to enhance the performance of GNNs (Graph Neural Networks). While most existing graph regularizers focus on manipulating graph topological structures by adding/removing edges, we offer a method to augment node features for better performance. We propose FLAG (Free Large-scale Adversarial Augmentation on Graphs), which iteratively augments node features with gradient-based adversarial perturbations during training. By making the model invariant to small fluctuations in input data, our method helps models generalize to out-of-distribution samples and boosts model performance at test time. FLAG is a general-purpose approach for graph data, which universally works in node classification, link prediction, and graph classification tasks. FLAG is also highly flexible and scalable, and is deployable with arbitrary GNN backbones and large-scale datasets. We demon-strate the efficacy and stability of our method through ex-tensive experiments and ablation studies. We also provide intuitive observations for a deeper understanding of our method. We open source our implementation at https://github.com/devnkong/FLAG.

----

## [10] A Re-Balancing Strategy for Class-Imbalanced Classification Based on Instance Difficulty

**Authors**: *Sihao Yu, Jiafeng Guo, Ruqing Zhang, Yixing Fan, Zizhen Wang, Xueqi Cheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00017](https://doi.org/10.1109/CVPR52688.2022.00017)

**Abstract**:

Real-world data often exhibits class-imbalanced distributions, where a few classes (a.k.a. majority classes) occupy most instances and lots of classes (a.k.a. minority classes) have few instances. Neural classification models usually perform poorly on minority classes when training on such imbalanced datasets. To improve the performance on minority classes, existing methods typically re-balance the data distribution at the class level, i.e., assigning higher weights to minority classes and lower weights to majority classes during the training process. However, we observe that even the majority classes contain difficult instances to learn. By reducing the weights of the majority classes, such instances would become more difficult to learn and hurt the overall performance consequently. To tackle this problem, we propose a novel instance-level re-balancing strategy, which dynamically adjusts the sampling probabilities of instances according to the instance difficulty. Here the instance difficulty is measured based on the learning speed of instance, which is inspired by the human-leaning process (i.e., easier instances will be learned faster). We theoretically prove the correctness and convergence of our resampling algorithm. Empirical experiments demonstrate that our method significantly outperforms state-of-the-art re-balancing methods on the class-imbalanced datasets.

----

## [11] The Devil is in the Margin: Margin-based Label Smoothing for Network Calibration

**Authors**: *Bingyuan Liu, Ismail Ben Ayed, Adrian Galdran, Jose Dolz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00018](https://doi.org/10.1109/CVPR52688.2022.00018)

**Abstract**:

In spite of the dominant performances of deep neural networks, recent works have shown that they are poorly calibrated, resulting in over-confident predictions. Miscalibration can be exacerbated by overfitting due to the minimization of the cross-entropy during training, as it promotes the predicted softmax probabilities to match the one-hot label assignments. This yields a pre-softmax activation of the correct class that is significantly larger than the remaining activations. Recent evidence from the literature suggests that loss functions that embed implicit or explicit maximization of the entropy of predictions yield state-of-the-art calibration performances. We provide a unifying constrained-optimization perspective of current state-of-the-art calibration losses. Specifically, these losses could be viewed as approximations of a linear penalty (or a Lagrangian term) imposing equality constraints on logit distances. This points to an important limitation of such underlying equality constraints, whose ensuing gradients constantly push towards a non-informative solution, which might prevent from reaching the best compromise between the discriminative performance and calibration of the model during gradient-based optimization. Following our observations, we propose a simple and flexible generalization based on inequality constraints, which imposes a controllable margin on logit distances. Comprehensive experiments on a variety of image classification, semantic segmentation and NLP benchmarks demonstrate that our method sets novel state-of-the-art results on these tasks in terms of network calibration, without affecting the discriminative performance. The code is available at https://github.com/by-liu/MbLS.

----

## [12] Towards Better Plasticity-Stability Trade-off in Incremental Learning: A Simple Linear Connector

**Authors**: *Guoliang Lin, Hanlu Chu, Hanjiang Lai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00019](https://doi.org/10.1109/CVPR52688.2022.00019)

**Abstract**:

Plasticity-stability dilemma is a main problem for incremental learning, where plasticity is referring to the ability to learn new knowledge, and stability retains the knowledge of previous tasks. Many methods tackle this problem by storing previous samples, while in some applications, training data from previous tasks cannot be legally stored. In this work, we propose to employ mode connectivity in loss landscapes to achieve better plasticity-stability trade-off without any previous samples. We give an analysis of why and how to connect two independently optimized optima of networks, null-space projection for previous tasks and simple SGD for the current task, can attain a meaningful balance between preserving already learned knowledge and granting sufficient flexibility for learning a new task. This analysis of mode connectivity also provides us a new perspective and technology to control the trade-off between plasticity and stability. We evaluate the proposed method on several benchmark datasets. The results indicate our simple method can achieve notable improvement, and perform well on both the past and current tasks. On 10-split-CIFAR-100 task, our method achieves 79.79% accuracy, which is 6.02% higher. Our method also achieves 6.33% higher accuracy on TinyImageNet. Code is available at https://github.com/lingl1024/Connector.

----

## [13] GCR: Gradient Coreset based Replay Buffer Selection for Continual Learning

**Authors**: *Rishabh Tiwari, KrishnaTeja Killamsetty, Rishabh K. Iyer, Pradeep Shenoy*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00020](https://doi.org/10.1109/CVPR52688.2022.00020)

**Abstract**:

Continual learning (CL) aims to develop techniques by which a single model adapts to an increasing number of tasks encountered sequentially, thereby potentially leveraging learnings across tasks in a resource-efficient manner. A major challenge for CL systems is catastrophic forgetting, where earlier tasks are forgotten while learning a new task. To address this, replay-based CL approaches maintain and repeatedly retrain on a small buffer of data selected across encountered tasks. We propose Gradient Coreset Replay (GCR), a novel strategy for replay buffer selection and update using a carefully designed optimization criterion. Specifically, we select and maintain a ‘coreset’ that closely approximates the gradient of all the data seen so far with respect to current model parameters, and discuss key strategies needed for its effective application to the continual learning setting. We show significant gains (2%-4% absolute) over the state-of-the-art in the well-studied offline continual learning setting. Our findings also effectively transfer to online / streaming CL settings, showing up to 5% gains over existing approaches. Finally, we demonstrate the value of supervised contrastive loss for continual learning, which yields a cumulative gain of up to 5% accuracy when combined with our subset selection strategy.

----

## [14] Learning Bayesian Sparse Networks with Full Experience Replay for Continual Learning

**Authors**: *Qingsen Yan, Dong Gong, Yuhang Liu, Anton van den Hengel, Javen Qinfeng Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00021](https://doi.org/10.1109/CVPR52688.2022.00021)

**Abstract**:

Continual Learning (CL) methods aim to enable machine learning models to learn new tasks without catastrophic forgetting of those that have been previously mastered. Existing CL approaches often keep a buffer of previously-seen samples, perform knowledge distillation, or use regularization techniques towards this goal. Despite their performance, they still suffer from interference across tasks which leads to catastrophic forgetting. To ameliorate this problem, we propose to only activate and select sparse neurons for learning current and past tasks at any stage. More parameters space and model capacity can thus be reserved for the future tasks. This minimizes the interference between parameters for different tasks. To do so, we propose a Sparse neural Network for Continual Learning (SNCL), which employs variational Bayesian sparsity priors on the activations of the neurons in all layers. Full Experience Replay (FER) provides effective supervision in learning the sparse activations of the neurons in different layers. A loss-aware reservoir-sampling strategy is developed to maintain the memory buffer. The proposed method is agnostic as to the network structures and the task boundaries. Experiments on different datasets show that SNCL achieves state-of-the-art result for mitigating forgetting.

----

## [15] A variational Bayesian method for similarity learning in non-rigid image registration

**Authors**: *Daniel Grzech, Mohammad Farid Azampour, Ben Glocker, Julia A. Schnabel, Nassir Navab, Bernhard Kainz, Loïc Le Folgoc*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00022](https://doi.org/10.1109/CVPR52688.2022.00022)

**Abstract**:

We propose a novel variational Bayesian formulation for diffeomorphic non-rigid registration of medical images, which learns in an unsupervised way a data-specific similarity metric. The proposed framework is general and may be used together with many existing image registration models. We evaluate it on brain MRI scans from the UK Biobank and show that use of the learnt similarity metric, which is parametrised as a neural network, leads to more accurate results than use of traditional functions, e.g. SSD and LCC, to which we initialise the model, without a negative impact on image registration speed or transformation smoothness. In addition, the method estimates the uncertainty associated with the transformation. The code and the trained models are available in a public repository: https://github.com/dgrzech/learnsim.

----

## [16] Learning to Learn by Jointly Optimizing Neural Architecture and Weights

**Authors**: *Yadong Ding, Yu Wu, Chengyue Huang, Siliang Tang, Yi Yang, Longhui Wei, Yueting Zhuang, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00023](https://doi.org/10.1109/CVPR52688.2022.00023)

**Abstract**:

Meta-learning enables models to adapt to new environments rapidly with a few training examples. Current gradient-based meta-learning methods concentrate on finding good model-agnostic initialization (meta-weights) for learners. In this paper, we aim to obtain better meta-learners by co-optimizing the architecture and meta-weights simultaneously. Existing NAS-based meta-learning methods apply a two-stage strategy, i.e., first searching architectures and then re-training meta-weights on the searched architecture. However, this two-stage strategy would break the mutual impact of the architecture and meta-weights since they are optimized separately. Differently, we propose progressive connection consolidation, fixing the architecture layer by layer, in which the layer with the largest weight value would be fixed first. In this way, we can jointly search architectures and train the meta-weights on fixed layers. Besides, to improve the generalization performance of the searched meta-learner on all tasks, we propose a more effective rule for co-optimization, namely Connection-Adaptive Meta-learning (CAML). By searching only once, we can obtain both adaptive architecture and meta-weights for meta-learning. Extensive experiments show that our method achieves state-of-the-art performance with 3x less computational cost, revealing our method's effectiveness and efficiency.

----

## [17] Learning to Prompt for Continual Learning

**Authors**: *Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer G. Dy, Tomas Pfister*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00024](https://doi.org/10.1109/CVPR52688.2022.00024)

**Abstract**:

The mainstream paradigm behind continual learning has been to adapt the model parameters to non-stationary data distributions, where catastrophic forgetting is the central challenge. Typical methods rely on a rehearsal buffer or known task identity at test time to retrieve learned knowl-edge and address forgetting, while this work presents a new paradigm for continual learning that aims to train a more succinct memory system without accessing task identity at test time. Our method learns to dynamically prompt (L2P) a pre-trained model to learn tasks sequen-tially under different task transitions. In our proposed framework, prompts are small learnable parameters, which are maintained in a memory space. The objective is to optimize prompts to instruct the model prediction and ex-plicitly manage task-invariant and task-specific knowledge while maintaining model plasticity. We conduct comprehen-sive experiments under popular image classification bench-marks with different challenging continual learning set-tings, where L2P consistently outperforms prior state-of-the-art methods. Surprisingly, L2P achieves competitive results against rehearsal-based methods even without a re-hearsal buffer and is directly applicable to challenging task-agnostic continual learning. Source code is available at https://github.com/google-research/12p.

----

## [18] Meta-attention for ViT-backed Continual Learning

**Authors**: *Mengqi Xue, Haofei Zhang, Jie Song, Mingli Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00025](https://doi.org/10.1109/CVPR52688.2022.00025)

**Abstract**:

Continual learning is a longstanding research topic due to its crucial role in tackling continually arriving tasks. Up to now, the study of continual learning in computer vision is mainly restricted to convolutional neural networks (CNNs). However, recently there is a tendency that the newly emerging vision transformers (ViTs) are gradually dominating the field of computer vision, which leaves CNN-based continual learning lagging behind as they can suffer from severe performance degradation if straightforwardly applied to ViTs. In this paper, we study ViT-backed continual learning to strive for higher performance riding on recent advances of ViTs. Inspired by mask-based continual learning methods in CNNs, where a mask is learned per task to adapt the pre-trained ViT to the new task, we propose MEta-ATtention (MEAT), i.e., attention to self-attention, to adapt a pre-trained ViT to new tasks without sacrificing performance on already learned tasks. Unlike prior mask-based methods like Piggyback, where all parameters are associated with corresponding masks, MEAT leverages the characteristics of ViTs and only masks a portion of its parameters. It renders MEAT more efficient and effective with less overhead and higher accuracy. Extensive experiments demonstrate that MEAT exhibits significant superiority to its state-of-the-art CNN counterparts, with 4.0 ∼ 6.0% absolute boosts in accuracy. Our code has been released at https://github.com/zju-vipa/MEAT-TIL.

----

## [19] Multi-Frame Self-Supervised Depth with Transformers

**Authors**: *Vitor Guizilini, Rares Ambrus, Dian Chen, Sergey Zakharov, Adrien Gaidon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00026](https://doi.org/10.1109/CVPR52688.2022.00026)

**Abstract**:

Multi-frame depth estimation improves over single-frame approaches by also leveraging geometric relationships between images via feature matching, in addition to learning appearance-based features. In this paper we revisit feature matching for self-supervised monocular depth estimation, and propose a novel transformer architecture for cost volume generation. We use depth-discretized epipolar sampling to select matching candidates, and refine predictions through a series of self- and cross-attention layers. These layers sharpen the matching probability between pixel features, improving over standard similarity metrics prone to ambiguities and local minima. The refined cost volume is decoded into depth estimates, and the whole pipeline is trained end-to-end from videos using only a photometric objective. Experiments on the KITTI and DDAD datasets show that our DepthFormer architecture establishes a new state of the art in self-supervised monocular depth estimation, and is even competitive with highly specialized supervised single-frame architectures. We also show that our learned cross-attention network yields representations transferable across datasets, increasing the effectiveness of pre-training strategies. Project page: https://sites.google.com/tri.global/depthformer.

----

## [20] Continual Learning with Lifelong Vision Transformer

**Authors**: *Zhen Wang, Liu Liu, Yiqun Duan, Yajing Kong, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00027](https://doi.org/10.1109/CVPR52688.2022.00027)

**Abstract**:

Continual learning methods aim at training a neural network from sequential data with streaming labels, relieving catastrophic forgetting. However, existing methods are based on and designed for convolutional neural networks (CNNs), which have not utilized the full potential of newly emerged powerful vision transformers. In this paper, we propose a novel attention-based framework Lifelong Vision Transformer (LVT), to achieve a better stability-plasticity trade-off for continual learning. Specifically, an inter-task attention mechanism is presented in LVT, which implicitly absorbs the previous tasks' information and slows down the drift of important attention between previous tasks and the current task. LVT designs a dual-classifier structure that independently injects new representation to avoid catas-trophic interference and accumulates the new and previous knowledge in a balanced manner to improve the overall performance. Moreover, we develop a confidence-aware memory update strategy to deepen the impression of the previous tasks. The extensive experimental results show that our approach achieves state-of-the-art performance with even fewer parameters on continual learning benchmarks.

----

## [21] Rethinking Bayesian Deep Learning Methods for Semi-Supervised Volumetric Medical Image Segmentation

**Authors**: *Jianfeng Wang, Thomas Lukasiewicz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00028](https://doi.org/10.1109/CVPR52688.2022.00028)

**Abstract**:

Recently, several Bayesian deep learning methods have been proposed for semi-supervised medical image segmentation. Although they have achieved promising results on medical benchmarks, some problems are still existing. Firstly, their overall architectures belong to the discriminative models, and hence, in the early stage of training, they only use labeled data for training, which might make them overfit to the labeled data. Secondly, in fact, they are only partially based on Bayesian deep learning, as their overall architectures are not designed under the Bayesian framework. However, unifying the overall architecture under the Bayesian perspective can make the architecture have a rigorous theoretical basis, so that each part of the architecture can have a clear probabilistic interpretation. Therefore, to solve the problems, we propose a new generative Bayesian deep learning (GBDL) architecture. GBDL belongs to the generative models, whose target is to estimate the joint distribution of input medical volumes and their corresponding labels. Estimating the joint distribution implicitly involves the distribution of data, so both labeled and unlabeled data can be utilized in the early stage of training, which alleviates the potential overfitting problem. Besides, GBDL is completely designed under the Bayesian framework, and thus we give its full Bayesian formulation, which lays a theoretical probabilistic foundation for our architecture. Extensive experiments show that our GBDL outperforms previous state-of-the-art methods in terms of four commonly used evaluation indicators on three public medical datasets.

----

## [22] Revisiting Random Channel Pruning for Neural Network Compression

**Authors**: *Yawei Li, Kamil Adamczewski, Wen Li, Shuhang Gu, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00029](https://doi.org/10.1109/CVPR52688.2022.00029)

**Abstract**:

Channel (or 3D filter) pruning serves as an effective way to accelerate the inference of neural networks. There has been a flurry of algorithms that try to solve this practical problem, each being claimed effective in some ways. Yet, a benchmark to compare those algorithms directly is lacking, mainly due to the complexity of the algorithms and some custom settings such as the particular network configuration or training procedure. A fair benchmark is important for the further development of channel pruning. Meanwhile, recent investigations reveal that the channel configurations discovered by pruning algorithms are at least as important as the pre-trained weights. This gives channel pruning a new role, namely searching the optimal channel configuration. In this paper, we try to determine the channel configuration of the pruned models by random search. The proposed approach provides a new way to compare different methods, namely how well they behave compared with random pruning. We show that this simple strategy works quite well compared with other channel pruning methods. We also show that under this setting, there are surprisingly no clear winners among different channel importance evaluation methods, which then may tilt the research efforts into advanced channel configuration searching methods. Code will be released at https://github.com/ofsoundof/random_channel_pruning.

----

## [23] Deep Safe Multi-view Clustering: Reducing the Risk of Clustering Performance Degradation Caused by View Increase

**Authors**: *Huayi Tang, Yong Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00030](https://doi.org/10.1109/CVPR52688.2022.00030)

**Abstract**:

Multi-view clustering has been shown to boost clustering performance by effectively mining the complementary information from multiple views. However, we observe that learning from data with more views is not guaranteed to achieve better clustering performance than from data with fewer views. To address this issue, we propose a general deep learning based framework that is guaranteed to reduce the risk of performance degradation caused by view increase. Concretely, the model is trained to simultaneously extract complementary information and discard the meaningless noise by automatically selecting features. These two learning procedures are incorporated into one unified framework by the proposed optimization objective. In theory, the empirical clustering risk of the model is no higher than learning from data before the view increase and data of the new increased single view. Also, the expected clustering risk of the model under divergence-based loss is no higher than that with high probability. Comprehensive experiments on benchmark datasets demonstrate the effectiveness and superiority of the proposed framework in achieving safe multi-view clustering.

----

## [24] Hypergraph-Induced Semantic Tuplet Loss for Deep Metric Learning

**Authors**: *Jongin Lim, Sangdoo Yun, Seulki Park, Jin Young Choi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00031](https://doi.org/10.1109/CVPR52688.2022.00031)

**Abstract**:

In this paper, we propose Hypergraph-Induced Semantic Tuplet (HIST) loss for deep metric learning that leverages the multilateral semantic relations of multiple samples to multiple classes via hypergraph modeling. We formulate deep metric learning as a hypergraph node classification problem in which each sample in a mini-batch is regarded as a node and each hyperedge models class-specific semantic relations represented by a semantic tuplet. Unlike previous graph-based losses that only use a bundle of pairwise relations, our HIST loss takes advantage of the multilateral semantic relations provided by the semantic tuplets through hypergraph modeling. Notably, by leveraging the rich multilateral semantic relations, HIST loss guides the embedding model to learn class-discriminative visual semantics, contributing to better generalization performance and model robustness against input corruptions. Extensive experiments and ablations provide a strong motivation for the proposed method and show that our HIST loss leads to improved feature learning, achieving state-of-the-art results on three widely used benchmarks. Code is available at https://github.com/ljin0429/HIST.

----

## [25] Towards Robust and Reproducible Active Learning using Neural Networks

**Authors**: *Prateek Munjal, Nasir Hayat, Munawar Hayat, Jamshid Sourati, Shadab Khan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00032](https://doi.org/10.1109/CVPR52688.2022.00032)

**Abstract**:

Active learning (AL) is a promising ML paradigm that has the potential to parse through large unlabeled data and help reduce annotation cost in domains where labeling data can be prohibitive. Recently proposed neural network based AL methods use different heuristics to accomplish this goal. In this study, we demonstrate that under identical experimental settings, different types of AL algorithms (uncertainty based, diversity based, and committee based) produce an inconsistent gain over random sampling baseline. Through a variety of experiments, controlling for sources of stochasticity, we show that variance in performance metrics achieved by AL algorithms can lead to results that are not consistent with the previously reported results. We also found that under strong regularization, AL methods show marginal or no advantage over the random sampling baseline under a variety of experimental conditions. Finally, we conclude with a set of recommendations on how to assess the results using a new AL algorithm to ensure results are reproducible and robust under changes in experimental conditions. We share our codes to facilitate AL evaluations. We believe our findings and recommendations will help advance reproducible research in AL using neural networks.

----

## [26] Non-Iterative Recovery from Nonlinear Observations using Generative Models

**Authors**: *Jiulong Liu, Zhaoqiang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00033](https://doi.org/10.1109/CVPR52688.2022.00033)

**Abstract**:

In this paper, we aim to estimate the direction of an underlying signal from its nonlinear observations following the semi-parametric single index model (SIM). Unlike for conventional compressed sensing where the signal is assumed to be sparse, we assume that the signal lies in the range of an L-Lipschitz continuous generative model with bounded k-dimensional inputs. This is mainly motivated by the tremendous success of deep generative models in various real applications. Our reconstruction method is noniterative (though approximating the projection step may require an iterative procedure) and highly efficient, and it is shown to attain the near-optimal statistical rate of order 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\sqrt{(k\log L)}/m$</tex>
, where m is the number of measurements. We consider two specific instances of the SIM, namely noisy 1-bit and cubic measurement models, and perform experiments on image datasets to demonstrate the efficacy of our method. In particular, for the noisy 1-bit measurement model, we show that our non-iterative method significantly outperforms a state-of-the-art iterative method in terms of both accuracy and efficiency.

----

## [27] Gaussian Process Modeling of Approximate Inference Errors for Variational Autoencoders

**Authors**: *Minyoung Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00034](https://doi.org/10.1109/CVPR52688.2022.00034)

**Abstract**:

Variational autoencoder (VAE) is a very successful generative model whose key element is the so-called amortized inference network, which can perform test time inference using a single feed forward pass. Unfortunately, this comes at the cost of degraded accuracy in posterior approximation, often underperforming the instance-wise variational optimization. Although the latest semi-amortized approaches mitigate the issue by performing a few variational optimization updates starting from the VAE's amortized inference output, they inherently suffer from computational overhead for inference at test time. In this paper, we address the problem in a completely different way by considering a random inference model, where we model the mean and variance functions of the variational posterior as random Gaussian processes (GP). The motivation is that the deviation of the VAE's amortized posterior distribution from the true posterior can be regarded as random noise, which allows us to view the approximation error as uncertainty in posterior approximation that can be dealt with in a principled GP manner. In particular, our model can quantify the difficulty in posterior approximation by a Gaussian variational density. Inference in our GP model is done by a single feed forward pass through the network, significantly faster than semi-amortized methods. We show that our approach attains higher test data likelihood than the state-of-the-arts on several benchmark datasets.

----

## [28] Robust Combination of Distributed Gradients Under Adversarial Perturbations

**Authors**: *Kwang In Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00035](https://doi.org/10.1109/CVPR52688.2022.00035)

**Abstract**:

We consider distributed (gradient descent-based) learning scenarios where the server combines the gradients of learning objectives gathered from local clients. As individual data collection and learning environments can vary, some clients could transfer erroneous gradients e.g. due to ad-versarial data or gradient perturbations. Further, for data privacy and security, the identities of such affected clients are often unknown to the server. In such cases, naively ag-gregating the resulting gradients can mislead the learning process. We propose a new server-side learning algorithm that robustly combines gradients. Our algorithm embeds the local gradients into the manifold of normalized gradients and refines their combinations via simulating a diffusion process therein. The resulting algorithm is instantiated as a compu-tationally simple and efficient weighted gradient averaging algorithm. In the experiments with five classification and three regression benchmark datasets, our algorithm demon-strated significant performance improvements over existing robust gradient combination algorithms as well as the base-line uniform gradient averaging algorithm.

----

## [29] Do learned representations respect causal relationships?

**Authors**: *Lan Wang, Vishnu Naresh Boddeti*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00036](https://doi.org/10.1109/CVPR52688.2022.00036)

**Abstract**:

Data often has many semantic attributes that are causally associated with each other. But do attribute-specific learned representations of data also respect the same causal relations? We answer this question in three steps. First, we introduce NCINet, an approach for obser-vational causal discovery from high-dimensional data. It is trained purely on synthetically generated representations and can be applied to real representations, and is specif-ically designed to mitigate the domain gap between the two. Second, we apply NCINet to identify the causal relations between image representations of different pairs of at-tributes with known and unknown causal relations between the labels. For this purpose, we consider image represen-tations learned for predicting attributes on the 3D Shapes, CelebA, and the CASIA-WebFace datasets, which we an-notate with multiple multi-class attributes. Third, we an-alyze the effect on the underlying causal relation between learned representations induced by various design choices in representation learning. Our experiments indicate that (1) NCINet significantly outperforms existing observational causal discovery approaches for estimating the causal relation between pairs of random samples, both in the presence and absence of an unobserved confounder, (2) under controlled scenarios, learned representations can indeed satisfy the underlying causal relations between their respective labels, and (3) the causal relations are positively correlated with the predictive capability of the representations. Code and annotations are available at: https://github.com/human-analysis/causal-relations-between-representations.

----

## [30] How Much More Data Do I Need? Estimating Requirements for Downstream Tasks

**Authors**: *Rafid Mahmood, James Lucas, David Acuna, Daiqing Li, Jonah Philion, José M. Álvarez, Zhiding Yu, Sanja Fidler, Marc T. Law*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00037](https://doi.org/10.1109/CVPR52688.2022.00037)

**Abstract**:

Given a small training data set and a learning algorithm, how much more data is necessary to reach a target validation or test performance? This question is of critical importance in applications such as autonomous driving or medical imaging where collecting data is expensive and time-consuming. Overestimating or underestimating data requirements incurs substantial costs that could be avoided with an adequate budget. Prior work on neural scaling laws suggest that the power-law function can fit the validation performance curve and extrapolate it to larger data set sizes. We find that this does not immediately translate to the more difficult downstream task of estimating the required data set size to meet a target performance. In this work, we consider a broad class of computer vision tasks and systematically investigate a family of functions that generalize the power-law function to allow for better estimation of data requirements. Finally, we show that incorporating a tuned correction factor and collecting over multiple rounds significantly improves the performance of the data estimators. Using our guidelines, practitioners can accurately estimate data requirements of machine learning systems to gain savings in both development time and data acquisition costs.

----

## [31] Pushing the Envelope of Gradient Boosting Forests via Globally-Optimized Oblique Trees

**Authors**: *Magzhan Gabidolla, Miguel Á. Carreira-Perpiñán*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00038](https://doi.org/10.1109/CVPR52688.2022.00038)

**Abstract**:

Ensemble methods based on decision trees, such as Random Forests or boosted forests, have long been established as some of the most powerful, off-the-shelf machine learning models, and have been widely used in computer vision and other areas. In recent years, a specific form of boosting, gradient boosting (GB), has gained prominence. This is partly because of highly optimized implementations such as XGBoost or LightGBM, which incorporate many clever modifications and heuristics. However, one gaping hole remains unexplored in GB: the construction of individual trees. To date, all successful GB versions use axis-aligned trees trained in a suboptimal way via greedy recursive partitioning. We address this gap by using a more powerful type of trees (having hyperplane splits) and an algorithm that can optimize, globally over all the tree parameters, the objective function that GB dictates. We show, in several benchmarks of image and other data types, that GB forests of these stronger, well-optimized trees consistently exceed the test accuracy of axis-aligned forests from XGBoost, Light-GBM and other strong baselines. Further, this happens using many fewer trees and sometimes even fewer parameters overall.

----

## [32] Contrastive Test-Time Adaptation

**Authors**: *Dian Chen, Dequan Wang, Trevor Darrell, Sayna Ebrahimi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00039](https://doi.org/10.1109/CVPR52688.2022.00039)

**Abstract**:

Test-time adaptation is a special setting of unsupervised domain adaptation where a trained model on the source domain has to adapt to the target domain without accessing source data. We propose a novel way to leverage self-supervised contrastive learning to facilitate target feature learning, along with an online pseudo labeling scheme with refinement that significantly denoises pseudo labels. The contrastive learning task is applied jointly with pseudo labeling, contrasting positive and negative pairs constructed similarly as MoCo but with source-initialized encoder, and excluding same-class negative pairs indicated by pseudo labels. Meanwhile, we produce pseudo labels online and refine them via soft voting among their nearest neighbors in the target feature space, enabled by maintaining a memory queue. Our method, AdaContrast, achieves state-of-the-art performance on major benchmarks while having several desirable properties compared to existing works, including memory efficiency, insensitivity to hyper-parameters, and better model calibration. Code is released at https://github.com/DianCh/AdaContrast.

----

## [33] AutoSDF: Shape Priors for 3D Completion, Reconstruction and Generation

**Authors**: *Paritosh Mittal, Yen-Chi Cheng, Maneesh Singh, Shubham Tulsiani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00040](https://doi.org/10.1109/CVPR52688.2022.00040)

**Abstract**:

Powerful priors allow us to perform inference with in-sufficient information. In this paper, we propose an au-toregressive prior for 3D shapes to solve multimodal 3D tasks such as shape completion, reconstruction, and gener-ation. We model the distribution over 3D shapes as a non-sequential autoregressive distribution over a discretized, low-dimensional, symbolic grid-like latent representation of 3D shapes. This enables us to represent distributions over 3D shapes conditioned on information from an arbitrary set of spatially anchored query locations and thus perform shape completion in such arbitrary settings (e.g. generating a complete chair given only a view of the back leg). We also show that the learned autoregressive prior can be leveraged for conditional tasks such as single-view reconstruction and language-based generation. This is achieved by learning task-specific ‘naive’ conditionals which can be approxi-mated by light-weight models trained on minimal paired data. We validate the effectiveness of the proposed method using both quantitative and qualitative evaluation and show that the proposed method outperforms the specialized state-of-the-art methods trained for individual tasks. The project page with code and video visualizations can be found at https://yccyenchicheng.github.io/AutoSDF/.

----

## [34] Selective-Supervised Contrastive Learning with Noisy Labels

**Authors**: *Shikun Li, Xiaobo Xia, Shiming Ge, Tongliang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00041](https://doi.org/10.1109/CVPR52688.2022.00041)

**Abstract**:

Deep networks have strong capacities of embedding data into latent representations and finishing following tasks. However, the capacities largely come from high-quality annotated labels, which are expensive to collect. Noisy labels are more affordable, but result in corrupted representations, leading to poor generalization performance. To learn robust representations and handle noisy labels, we propose selective-supervised contrastive learning (Sel-CL) in this paper. Specifically, Sel-CL extend supervised contrastive learning (Sup-CL), which is powerful in representation learning, but is degraded when there are noisy labels. Sel-CL tackles the direct cause of the problem of Sup-CL. That is, as Sup-CL works in a pair-wise manner, noisy pairs built by noisy labels mislead representation learning. To alleviate the issue, we select confident pairs out of noisy ones for Sup-CL without knowing noise rates. In the selection process, by measuring the agreement between learned representations and given labels, we first identify confident examples that are exploited to build confident pairs. Then, the representation similarity distribution in the built confident pairs is exploited to identify more confident pairs out of noisy pairs. All obtained confident pairs are finally used for Sup-CL to enhance representations. Experiments on multiple noisy datasets demonstrate the robustness of the learned representations by our method, following the state-of-the-art performance. Source codes are available at https://github.com/ShikunLi/Sel-Cl.

----

## [35] RecDis-SNN: Rectifying Membrane Potential Distribution for Directly Training Spiking Neural Networks

**Authors**: *Yufei Guo, Xinyi Tong, Yuanpei Chen, Liwen Zhang, Xiaode Liu, Zhe Ma, Xuhui Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00042](https://doi.org/10.1109/CVPR52688.2022.00042)

**Abstract**:

The brain-inspired and event-driven Spiking Neural Network (SNN) aiming at mimicking the synaptic activity of biological neurons has received increasing attention. It transmits binary spike signals between network units when the membrane potential exceeds the firing threshold. This biomimetic mechanism of SNN appears energy-efficiency with its power sparsity and asynchronous operations on spike events. Unfortunately, with the propagation of binary spikes, the distribution of membrane potential will shift, leading to degeneration, saturation, and gradient mismatch problems, which would be disadvantageous to the network optimization and convergence. Such undesired shifts would prevent the SNN from performing well and going deep. To tackle these problems, we attempt to rectify the membrane potential distribution (MPD) by designing a novel distribution loss, MPD-Loss, which can explicitly penalize the un-desired shifts without introducing any additional operations in the inference phase. Moreover, the proposed method can also mitigate the quantization error in SNNs, which is usually ignored in other works. Experimental results demonstrate that the proposed method can directly train a deeper, larger, and better-performing SNN within fewer timesteps.

----

## [36] Hierarchical Nearest Neighbor Graph Embedding for Efficient Dimensionality Reduction

**Authors**: *M. Saquib Sarfraz, Marios Koulakis, Constantin Seibold, Rainer Stiefelhagen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00043](https://doi.org/10.1109/CVPR52688.2022.00043)

**Abstract**:

Dimensionality reduction is crucial both for visualization and preprocessing high dimensional data for machine learning. We introduce a novel method based on a hierarchy built on 1-nearest neighbor graphs in the original space which is used to preserve the grouping properties of the data distribution on multiple levels. The core of the proposal is an optimization-free projection that is competitive with the latest versions of t-SNE and UMAP in performance and visualization quality while being an order of magnitude faster at run-time. Furthermore, its interpretable mechanics, the ability to project new data, and the natural separation of data clusters in visualizations make it a general purpose unsupervised dimension reduction technique. In the paper, we argue about the soundness of the proposed method and evaluate it on a diverse collection of datasets with sizes varying from 1 K to 11M samples and dimensions from 28 to 16K. We perform comparisons with other state-of-the-art methods on multiple metrics and target dimensions high-lighting its efficiency and performance. Code is available at https://github.com/koulakis/h-nne

----

## [37] Scalable Penalized Regression for Noise Detection in Learning with Noisy Labels

**Authors**: *Yikai Wang, Xinwei Sun, Yanwei Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00044](https://doi.org/10.1109/CVPR52688.2022.00044)

**Abstract**:

Noisy training set usually leads to the degradation of generalization and robustness of neural networks. In this paper, we propose using a theoretically guaranteed noisy label detection framework to detect and remove noisy data for Learning with Noisy Labels (LNL). Specifically, we design a penalized regression to model the linear relation between network features and one-hot labels, where the noisy data are identified by the non-zero mean shift parameters solved in the regression model. To make the framework scalable to datasets that contain a large number of categories and training data, we propose a split algorithm to divide the whole training set into small pieces that can be solved by the penalized regression in parallel, leading to the Scalable Penalized Regression (SPR) framework. We provide the non-asymptotic probabilistic condition for SP R to correctly identify the noisy data. While SPR can be regarded as a sample selection module for standard supervised training pipeline, we further combine it with semi-supervised algorithm to further exploit the support of noisy data as unlabeled data. Experimental results on several benchmark datasets and real-world noisy datasets show the effectiveness of our framework. Our code and pretrained models are released at https://github.com/Yikai-Wang/SPR-LNL.

----

## [38] Nested Hyperbolic Spaces for Dimensionality Reduction and Hyperbolic NN Design

**Authors**: *Xiran Fan, Chun-Hao Yang, Baba C. Vemuri*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00045](https://doi.org/10.1109/CVPR52688.2022.00045)

**Abstract**:

Hyperbolic neural networks have been popular in the recent past due to their ability to represent hierarchical data sets effectively and efficiently. The challenge in developing these networks lies in the nonlinearity of the embedding space namely, the Hyperbolic space. Hyperbolic space is a homogeneous Riemannian manifold of the Lorentz group which is a semi-Riemannian manifold, i.e. a mani-fold equipped with an indefinite metric. Most existing methods (with some exceptions) use local linearization to define a variety of operations paralleling those used in traditional deep neural networks in Euclidean spaces. In this paper, we present a novel fully hyperbolic neural network which uses the concept of projections (embeddings) followed by an intrinsic aggregation and a nonlinearity all within the hyperbolic space. The novelty here lies in the projection which is designed to project data on to a lower-dimensional embedded hyperbolic space and hence leads to a nested hyperbolic space representation independently useful for dimensionality reduction. The main theoretical contribution is that the proposed embedding is proved to be isometric and equivariant under the Lorentz transformations, which are the natural isometric transformations in hyperbolic spaces. This projection is computationally efficient since it can be expressed by simple linear operations, and, due to the aforementioned equivariance property, it allows for weight sharing. The nested hyperbolic space representation is the core component of our network and therefore, we first compare this representation - independent of the network - with other dimensionality reduction methods such as tangent PCA, principal geodesic analysis (PGA) and HoroPCA. Based on this equivariant embedding, we develop a novel fully hyperbolic graph convolutional neural network architecture to learn the parameters of the projection. Finally, we present experiments demonstrating comparative performance of our network on several publicly available data sets.

----

## [39] Learning Structured Gaussians to Approximate Deep Ensembles

**Authors**: *Ivor J. A. Simpson, Sara Vicente, Neill D. F. Campbell*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00046](https://doi.org/10.1109/CVPR52688.2022.00046)

**Abstract**:

This paper proposes using a sparse-structured multivari-ate Gaussian to provide a closed-form approximator for the output of probabilistic ensemble models used for dense im-age prediction tasks. This is achieved through a convolutional neural network that predicts the mean and covari-ance of the distribution, where the inverse covariance is parameterised by a sparsely structured Cholesky matrix. Similarly to distillation approaches, our single network is trained to maximise the probability of samples from pre-trained probabilistic models, in this work we use a fixed en-semble of networks. Once trained, our compact represen-tation can be used to efficiently draw spatially correlated samples from the approximated output distribution. Impor-tantly, this approach captures the uncertainty and struc-tured correlations in the predictions explicitly in a formal distribution, rather than implicitly through sampling alone. This allows direct introspection of the model, enabling vi-sualisation of the learned structure. Moreover, this formu-lation provides two further benefits: estimation of a sample probability, and the introduction of arbitrary spatial conditioning at test time. We demonstrate the merits of our approach on monocular depth estimation and show that the advantages of our approach are obtained with comparable quantitative performance.

----

## [40] Out-of-distribution Generalization with Causal Invariant Transformations

**Authors**: *Ruoyu Wang, Mingyang Yi, Zhitang Chen, Shengyu Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00047](https://doi.org/10.1109/CVPR52688.2022.00047)

**Abstract**:

In real-world applications, it is important and desirable to learn a model that performs well on out-of-distribution (OOD) data. Recently, causality has become a powerful tool to tackle the OOD generalization problem, with the idea resting on the causal mechanism that is invariant across domains of interest. To leverage the generally unknown causal mechanism, existing works assume a linear form of causal feature or require sufficiently many and diverse training domains, which are usually restrictive in practice. In this work, we obviate these assumptions and tackle the OOD problem without explicitly recovering the causal feature. Our approach is based on transformations that modify the non-causal feature but leave the causal part unchanged, which can be either obtained from prior knowledge or learned from the training data in the multi-domain scenario. Under the setting of invariant causal mechanism, we theoretically show that if all such transformations are available, then we can learn a minimax optimal model across the domains using only single domain data. Noticing that knowing a complete set of these causal invariant transformations may be impractical, we further show that it suffices to know only a subset of these transformations. Based on the theoretical findings, a regularized training procedure is proposed to improve the OOD generalization capability. Extensive experimental results on both synthetic and real datasets verify the effectiveness of the proposed algorithm, even with only a few causal invariant transformations.

----

## [41] Split Hierarchical Variational Compression

**Authors**: *Tom Ryder, Chen Zhang, Ning Kang, Shifeng Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00048](https://doi.org/10.1109/CVPR52688.2022.00048)

**Abstract**:

Variational autoencoders (VAEs) have witnessed great success in performing the compression of image datasets. This success, made possible by the bits-back coding framework, has produced competitive compression performance across many benchmarks. However, despite this, VAE architectures are currently limited by a combination of coding practicalities and compression ratios. That is, not only do state-of the-art methods, such as normalizing flows, often demonstrate out-performance, but the initial bits required in coding makes single and parallel image compression challenging. To remedy this, we introduce Split Hierarchical Variational Compression (SHVC). SHVC introduces two novelties. Firstly, we propose an efficient autoregressive prior, the autoregressive sub-pixel convolution, that allows a generalisation between per-pixel autoregressions and fully factorised probability models. Secondly, we define our coding framework, the autoregressive initial bits, that flexibly supports parallel coding and avoids -for the first time - many of the practicalities commonly associated with bits-back coding. In our experiments, we demonstrate SHVC is able to achieve state-of the-art compression performance across full-resolution lossless image compression tasks, with up to 100x fewer model parameters than competing VAE approaches.

----

## [42] Implicit Feature Decoupling with Depthwise Quantization

**Authors**: *Iordanis Fostiropoulos, Barry W. Boehm*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00049](https://doi.org/10.1109/CVPR52688.2022.00049)

**Abstract**:

Quantization has been applied to multiple domains in Deep Neural Networks (DNNs). We propose Depthwise Quantization (DQ) where quantization is applied to a de-composed sub-tensor along the feature axis of weak statis-tical dependence. The feature decomposition leads to an exponential increase in representation capacity with a linear increase in memory and parameter cost. In addition, DQ can be directly applied to existing encoder-decoder frame-works without modification of the DNN architecture. We use DQ in the context of Hierarchical Auto-Encoders and train end-to-end on an image feature representation. We provide an analysis of the cross-correlation between spatial and channel features and propose a decomposition of the image feature representation along the channel axis. The improved performance of the depthwise operator is due to the increased representation capacity from implicit feature decoupling. We evaluate DQ on the likelihood estimation task, where it outperforms the previous state-of-the-art on CIFAR-10, ImageNet-32 and ImageNet-64. We progressively train with increasing image size a single hierarchical model that uses 69% fewer parameters and has faster convergence than the previous work.

----

## [43] Understanding Uncertainty Maps in Vision with Statistical Testing

**Authors**: *Jurijs Nazarovs, Zhichun Huang, Songwong Tasneeyapant, Rudrasis Chakraborty, Vikas Singh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00050](https://doi.org/10.1109/CVPR52688.2022.00050)

**Abstract**:

Quantitative descriptions of confidence intervals and uncertainties of the predictions of a model are needed in many applications in vision and machine learning. Mechanisms that enable this for deep neural network (DNN) models are slowly becoming available, and occasionally, being integrated within production systems. But the literature is sparse in terms of how to perform statistical tests with the uncertainties produced by these overparameterized models. For two models with a similar accuracy profile, is the former model's uncertainty behavior better in a statistically significant sense compared to the second model? For high resolution images, performing hypothesis tests to generate meaningful actionable information (say, at a user specified significance level 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\alpha=0.05)$</tex>
 is difficult but needed in both mission critical settings and elsewhere. In this paper, specifically for uncertainties defined on images, we show how revisiting results from Random Field theory (RFT) when paired with DNN tools (to get around computational hurdles) leads to efficient frameworks that can provide a hypothesis test capabilities, not otherwise available, for uncertainty maps from models used in many vision tasks. We show via many different experiments the viability of this framework.

----

## [44] A Hybrid Quantum-Classical Algorithm for Robust Fitting

**Authors**: *Anh-Dzung Doan, Michele Sasdelli, David Suter, Tat-Jun Chin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00051](https://doi.org/10.1109/CVPR52688.2022.00051)

**Abstract**:

Fitting geometric models onto outlier contaminated data is provably intractable. Many computer vision systems rely on random sampling heuristics to solve robust fitting, which do not provide optimality guarantees and error bounds. It is therefore critical to develop novel approaches that can bridge the gap between exact solutions that are costly, and fast heuristics that offer no quality assurances. In this paper, we propose a hybrid quantum-classical algorithm for robust fitting. Our core contribution is a novel robust fitting formulation that solves a sequence of integer programs and terminates with a global solution or an error bound. The combinatorial subproblems are amenable to a quantum annealer, which helps to tighten the bound efficiently. While our usage of quantum computing does not surmount the fundamental intractability of robust fitting, by providing error bounds our algorithm is a practical improvement over randomised heuristics. Moreover, our work represents a concrete application of quantum computing in computer vision. We present results obtained using an actual quantum computer (D-Wave Advantage) and via simulation
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Source code: https://github.com/dadung/HQC-robust-fitting.

----

## [45] A Scalable Combinatorial Solver for Elastic Geometrically Consistent 3D Shape Matching

**Authors**: *Paul Roetzer, Paul Swoboda, Daniel Cremers, Florian Bernard*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00052](https://doi.org/10.1109/CVPR52688.2022.00052)

**Abstract**:

We present a scalable combinatorial algorithm for globally optimizing over the space of geometrically consistent mappings between 3D shapes. We use the mathematically elegant formalism proposed by Windheuser et al. [66] where 3D shape matching was formulated as an integer linear program over the space of orientation-preserving diffeomorphisms. Until now, the resulting formulation had limited practical applicability due to its complicated constraint structure and its large size. We propose a novel primal heuristic coupled with a Lagrange dual problem that is several orders of magnitudes faster compared to previous solvers. This allows us to handle shapes with substantially more triangles than previously solvable. We demonstrate compelling results on diverse datasets, and, even showcase that we can address the challenging setting of matching two partial shapes without availability of complete shapes. Our code is publicly available at http://github.com/paulOnoah/sm-comb.

----

## [46] FastDOG: Fast Discrete Optimization on GPU

**Authors**: *Ahmed Abbas, Paul Swoboda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00053](https://doi.org/10.1109/CVPR52688.2022.00053)

**Abstract**:

We present a massively parallel Lagrange decomposition method for solving 0–1 integer linear programs occurring in structured prediction. We propose a new iterative update scheme for solving the Lagrangean dual and a perturbation technique for decoding primal solutions. For representing subproblems we follow [40] and use binary decision diagrams (BDDs). Our primal and dual algorithms require little synchronization between subproblems and optimization over BDDs needs only elementary operations without complicated control flow. This allows us to exploit the parallelism offered by GPUs for all components of our method. We present experimental results on combinatorial problems from MAP inference for Markov Random Fields, quadratic assignment and cell tracking for developmental biology. Our highly parallel GPU implementation improves upon the running times of the algorithms from [40] by up to an order of magnitude. In particular, we come close to or outperform some state-of-the-art specialized heuristics while being problem agnostic. Our implementation is available at https://github.com/LPMP/BDD.

----

## [47] Data-Free Network Compression via Parametric Non-uniform Mixed Precision Quantization

**Authors**: *Vladimir Chikin, Mikhail Antiukh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00054](https://doi.org/10.1109/CVPR52688.2022.00054)

**Abstract**:

Deep Neural Networks (DNNs) usually have a large number of parameters and consume a huge volume of storage space, which limits the application of DNNs on memory-constrained devices. Network quantization is an appealing way to compress DNNs. However, most of existing quantization methods require the training dataset and a fine-tuning procedure to preserve the quality of a full-precision model. These are unavailable for the confidential scenarios due to personal privacy and security problems. Focusing on this issue, we propose a novel data-free method for network compression called PNMQ, which employs the Parametric Non-uniform Mixed precision Quantization to generate a quantized network. During the compression stage, the optimal parametric non-uniform quantization grid is calculated directly for each layer to minimize the quantization error. User can directly specify the required compression ratio of a network, which is used by the PNMQ algorithm to select bitwidths of layers. This method does not require any model retraining or expensive calculations, which allows efficient implementations for network compression on edge devices. Extensive experiments have been conducted on various computer vision tasks and the results demonstrate that PNMQ achieves better performance than other state-of-the-art methods of network compression.

----

## [48] AdaSTE: An Adaptive Straight-Through Estimator to Train Binary Neural Networks

**Authors**: *Huu Le, Rasmus Kjær Høier, Che-Tsung Lin, Christopher Zach*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00055](https://doi.org/10.1109/CVPR52688.2022.00055)

**Abstract**:

We propose a new algorithm for training deep neural networks (DNNs) with binary weights. In particular, we first cast the problem of training binary neural networks (BiNNs) as a bilevel optimization instance and subsequently construct flexible relaxations of this bilevel program. The resulting training method shares its algorithmic simplicity with several existing approaches to train BiNNs, in particular with the straight-through gradient estimator successfully employed in BinaryConnect and subsequent methods. Infact, our proposed method can be interpreted as an adaptive variant of the original straight-through estimator that conditionally (but not always) acts like a linear mapping in the backward pass of error propagation. Experimental results demonstrate that our new algorithm offers favorable performance compared to existing approaches.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
This work was partially supported by theWallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation.

----

## [49] GLASS: Geometric Latent Augmentation for Shape Spaces

**Authors**: *Sanjeev Muralikrishnan, Siddhartha Chaudhuri, Noam Aigerman, Vladimir G. Kim, Matthew Fisher, Niloy J. Mitra*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01800](https://doi.org/10.1109/CVPR52688.2022.01800)

**Abstract**:

We investigate the problem of training generative models on very sparse collections of 3D models. Particularly, instead of using difficult-to-obtain large sets of 3D models, we demonstrate that geometrically-motivated energy functions can be used to effectively augment and boost only a sparse collection of example (training) models. Technically, we analyze the Hessian of the as-rigid-as-possible (ARAP) energy to adaptively sample from and project to the underlying (local) shape space, and use the augmented dataset to train a variational autoencoder (VAE). We iterate the process, of building latent spaces of VAE and augmenting the associated dataset, to progressively reveal a richer and more expressive generative space for creating geometrically and semantically valid samples. We evaluate our method against a set of strong baselines, provide ablation studies, and demonstrate application towards establishing shape correspondences. Glassproduces multiple interesting and meaningful shape variations even when starting from as few as 3–10 training shapes. Our code is available at https://sanjeevmk.github.io/glass_webpage/.

----

## [50] Training Quantised Neural Networks with STE Variants: the Additive Noise Annealing Algorithm

**Authors**: *Matteo Spallanzani, Gian Paolo Leonardi, Luca Benini*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00056](https://doi.org/10.1109/CVPR52688.2022.00056)

**Abstract**:

Training quantised neural networks (QNNs) is a non-differentiable optimisation problem since weights and features are output by piecewise constant functions. The standard solution is to apply the straight-through estimator (STE), using different functions during the inference and gradient computation steps. Several STE variants have been proposed in the literature aiming to maximise the task accuracy of the trained network. In this paper, we analyse STE variants and study their impact on QNN training. We first observe that most such variants can be modelled as stochastic regularisations of stair functions; although this intuitive interpretation is not new, our rigorous discussion generalises to further variants. Then, we analyse QNNs mixing different regularisations, finding that some suitably synchronised smoothing of each layer map is required to guarantee pointwise compositional convergence to the target discontinuous function. Based on these theoretical insights, we propose additive noise annealing (ANA), a new algorithm to train QNNs encompassing standard STE and its variants as special cases. When testing ANA on the CIFAR-J0 image classification benchmark, we find that the major impact on task accuracy is not due to the qualitative shape of the regularisations but to the proper synchronisation of the different STE variants used in a network, in accordance with the theoretical results.

----

## [51] AME: Attention and Memory Enhancement in Hyper-Parameter Optimization

**Authors**: *Nuo Xu, Jianlong Chang, Xing Nie, Chunlei Huo, Shiming Xiang, Chunhong Pan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00057](https://doi.org/10.1109/CVPR52688.2022.00057)

**Abstract**:

Training Deep Neural Networks (DNNs) is inherently subject to sensitive hyper-parameters and untimely feedbacks of performance evaluation. To solve these two difficulties, an efficient parallel hyper-parameter optimization model is proposed under the framework of Deep Reinforcement Learning (DRL). Technically, we develop Attention and Memory Enhancement (AME), that includes multi-head attention and memory mechanism to enhance the ability to capture both the short-term and long-term relationships between different hyper-parameter configurations, yielding an attentive sampling mechanism for searching high-performance configurations embedded into a huge search space. During the optimization of transformer-structured configuration searcher, a conceptually intuitive yet powerful strategy is applied to solve the problem of insufficient number of samples due to the untimely feedback. Experiments on three visual tasks, including image classification, object detection, semantic segmentation, demonstrate the effectiveness of AME.

----

## [52] Efficient Maximal Coding Rate Reduction by Variational Forms

**Authors**: *Christina Baek, Ziyang Wu, Kwan Ho Ryan Chan, Tianjiao Ding, Yi Ma, Benjamin D. Haeffele*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00058](https://doi.org/10.1109/CVPR52688.2022.00058)

**Abstract**:

The principle of Maximal Coding Rate Reduction (MCR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) has recently been proposed as a training objective for learning discriminative low-dimensional structures intrinsic to high-dimensional data to allow for more robust training than standard approaches, such as cross-entropy minimization. However, despite the advantages that have been shown for MCR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 training, MCR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 suffers from a significant computational cost due to the need to evaluate and differentiate a significant number of log-determinant terms that grows linearly with the number of classes. By taking advantage of variational forms of spectral functions of a matrix, we reformulate the MCR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 objective to a form that can scale significantly without compromising training accuracy. Experiments in image classification demonstrate that our proposed formulation results in a significant speed up over optimizing the original MCR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 objective directly and often results in higher quality learned representations. Further, our approach may be of independent interest in other models that require computation of log-determinant forms, such as in system identification or normalizing flow models.

----

## [53] A Unified Framework for Implicit Sinkhorn Differentiation

**Authors**: *Marvin Eisenberger, Aysim Toker, Laura Leal-Taixé, Florian Bernard, Daniel Cremers*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00059](https://doi.org/10.1109/CVPR52688.2022.00059)

**Abstract**:

The Sinkhorn operator has recently experienced a surge of popularity in computer vision and related fields. One major reason is its ease of integration into deep learning frameworks. To allow for an efficient training of respective neural networks, we propose an algorithm that obtains analytical gradients of a Sinkhorn layer via implicit differentiation. In comparison to prior work, our framework is based on the most general formulation of the Sinkhorn operator. It allows for any type of loss function, while both the target capacities and cost matrices are differentiated jointly. We further construct error bounds of the resulting algorithm for approximate inputs. Finally, we demonstrate that for a number of applications, simply replacing automatic differentiation with our algorithm directly improves the stability and accuracy of the obtained gradients. Moreover, we show that it is computationally more efficient, particularly when resources like GPU memory are scarce. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our implementation is available under the following link: https://github.com/marvin-eisenberger/implicit-sinkhorn

----

## [54] Computing Wasserstein-$p$ Distance Between Images with Linear Cost

**Authors**: *Yidong Chen, Chen Li, Zhonghua Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00060](https://doi.org/10.1109/CVPR52688.2022.00060)

**Abstract**:

When the images are formulated as discrete measures, computing Wasserstein-p distance between them is challenging due to the complexity of solving the corresponding Kantorovich's problem. In this paper, we propose a novel algorithm to compute the Wasserstein-p distance between discrete measures by restricting the optimal transport (OT) problem on a subset. First, we define the restricted OT problem and prove the solution of the restricted problem converges to Kantorovich's OT solution. Second, we propose the SparseSinkhorn algorithm for the restricted problem and provide a multi-scale algorithm to estimate the subset. Finally, we implement the proposed algorithm on CUDA and illustrate the linear computational cost in terms of time and memory requirements. We compute Wasserstein-p distance, estimate the transport mapping, and transfer color between color images with size ranges from 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$64\times 64$</tex>
 to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1920\times 1200$</tex>
. (Our code is available at https://github.com/ucascnic/CudaOT)

----

## [55] An Iterative Quantum Approach for Transformation Estimation from Point Sets

**Authors**: *Natacha Kuete Meli, Florian Mannel, Jan Lellmann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00061](https://doi.org/10.1109/CVPR52688.2022.00061)

**Abstract**:

We propose an iterative method for estimating rigid transformations from point sets using adiabatic quantum computation. Compared to existing quantum approaches, our method relies on an adaptive scheme to solve the problem to high precision, and does not suffer from inconsistent rotation matrices. Experimentally, our method performs robustly on several 2D and 3D datasets even with high outlier ratio.

----

## [56] BoosterNet: Improving Domain Generalization of Deep Neural Nets using Culpability-Ranked Features

**Authors**: *Nourhan Bayasi, Ghassan Hamarneh, Rafeef Garbi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00062](https://doi.org/10.1109/CVPR52688.2022.00062)

**Abstract**:

Deep learning (DL) models trained to minimize empirical risk on a single domain often fail to generalize when applied to other domains. Model failures due to poor generalizability are quite common in practice and may prove quite perilous in mission-critical applications, e.g., diagnostic imaging where real-world data often exhibits pronounced variability. Such limitations have led to increased interest in domain generalization (DG) approaches that improve the ability of models learned from a single or multiple source domains to generalize to out-of-distribution (OOD) test domains. In this work, we propose BoosterNet, a lean add-on network that can be simply appended to any arbitrary core network to improve its generalization capability without requiring any changes in its architecture or training procedure. Specifically, using a novel measure of feature culpability, BoosterNet is trained episodically on the most and least culpable data features extracted from critical units in the core network based on their contribution towards class-specific prediction errors, which have shown to improve generalization. At inference time, corresponding test image features are extracted from the closest class-specific units, determined by smart gating via a Siamese network, and fed to BoosterNet for improved generalization. We evaluate the performance of BoosterNet within two very different classification problems, digits and skin lesions, and demonstrate a marked improvement in model generalization to OOD test domains compared to SOTA.

----

## [57] Pooling Revisited: Your Receptive Field is Suboptimal

**Authors**: *Dong-Hwan Jang, Sanghyeok Chu, Joonhyuk Kim, Bohyung Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00063](https://doi.org/10.1109/CVPR52688.2022.00063)

**Abstract**:

The size and shape of the receptive field determine how the network aggregates local features, and affect the overall performance of a model considerably. Many components in a neural network, such as depth, kernel sizes, and strides for convolution and pooling, influence the receptive field. However, they still rely on hyperparameters, and the receptive fields of existing models result in suboptimal shapes and sizes. Hence, we propose a simple yet effective Dynamically Optimized Pooling operation, referred to as DynOPool, which learns the optimized scale factors offeature maps end-to-end. Moreover, DynOPool determines the proper resolution of a feature map by learning the desirable size and shape of its receptive field, which allows an operator in a deeper layer to observe an input image in the optimal scale. Any kind of resizing modules in a deep neural network can be replaced by DynOPool with minimal cost. Also, DynOPool controls the complexity of the model by introducing an additional loss term that constrains computational cost. Our experiments show that the models equipped with the proposed learnable resizing module outperform the baseline algorithms on multiple datasets in image classification and semantic segmentation.

----

## [58] Why Discard if You can Recycle?: A Recycling Max Pooling Module for 3D Point Cloud Analysis

**Authors**: *Jiajing Chen, Burak Kakillioglu, Huantao Ren, Senem Velipasalar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00064](https://doi.org/10.1109/CVPR52688.2022.00064)

**Abstract**:

In recent years, most 3D point cloud analysis models have focused on developing either new network architectures or more efficient modules for aggregating point features from a local neighborhood. Regardless of the network architecture or the methodology used for improved feature learning, these models share one thing, which is the use of max-pooling in the end to obtain permutation invariant features. We first show that this traditional approach causes only a fraction of 3D points contribute to the permutation-invariant features, and discards the rest of the points. In order to address this issue and improve the performance of any baseline 3D point classification or segmentation model, we propose a new module, referred to as the Recycling Max-Pooling (RMP) module, to recycle and utilize the features of some of the discarded points. We incorporate a refinement loss that uses the recycled features to refine the prediction loss obtained from the features kept by traditional max-pooling. To the best of our knowledge, this is the first work that explores recycling of still useful points that are traditionally discarded by max-pooling. We demonstrate the effectiveness of the proposed RMP module by incorporating it into several milestone baselines and state-of-the-art networks for point cloud classification and indoor semantic segmentation tasks. We show that RPM, without any bells and whistles, consistently improves the performance of all the tested networks by using the same base network implementation and hyper-parameters. The code is provided in the supplementary material.

----

## [59] Online Convolutional Reparameterization

**Authors**: *Mu Hu, Junyi Feng, Jiashen Hua, Baisheng Lai, Jianqiang Huang, Xiaojin Gong, Xiansheng Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00065](https://doi.org/10.1109/CVPR52688.2022.00065)

**Abstract**:

Structural re-parameterization has drawn increasing attention in various computer vision tasks. It aims at improving the performance of deep models without introducing any inference-time cost. Though efficient during inference, such models rely heavily on the complicated training-time blocks to achieve high accuracy, leading to large extra training cost. In this paper, we present online convolutional re-parameterization (OREPA), a two-stage pipeline, aiming to reduce the huge training overhead by squeezing the complex training-time block into a single convolution. To achieve this goal, we introduce a linear scaling layer for better optimizing the online blocks. Assisted with the reduced training cost, we also explore some more effective re-param components. Compared with the state-of-the-art re-param models, OREPA is able to save the training-time memory cost by about 70% and accelerate the training speed by around 2×. Meanwhile, equipped with OREPA, the models out-perform previous methods on ImageNet by up to +0.6%. We also conduct experiments on object detection and semantic segmentation and show consistent improvements on the downstream tasks. Codes are available at https://github.com/JUGGHM/OREPA_CVPR2022.

----

## [60] RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality

**Authors**: *Xiaohan Ding, Honghao Chen, Xiangyu Zhang, Jungong Han, Guiguang Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00066](https://doi.org/10.1109/CVPR52688.2022.00066)

**Abstract**:

Compared to convolutional layers, fully-connected (FC) layers are better at modeling the long-range dependencies but worse at capturing the local patterns, hence usually less favored for image recognition. In this paper, we propose a methodology, Locality Injection, to incorporate local priors into an FC layer via merging the trained parameters of a parallel conv kernel into the FC kernel. Locality Injection can be viewed as a novel Structural Re-parameterization method since it equivalently converts the structures via transforming the parameters. Based on that, we propose a multi-layer-perceptron (MLP) block named RepMLP Block, which uses three FC layers to extract features, and a novel architecture named RepMLPNet. The hierarchical design distinguishes RepMLPNet from the other concurrently proposed vision MLPs. As it produces feature maps of different levels, it qualifies as a backbone model for downstream tasks like semantic segmentation. Our results reveal that 1) Locality Injection is a general methodology for MLP models; 2) RepMLPNet has favorable accuracy-efficiency trade-off compared to the other MLPs; 3) RepMLPNet is the first MLP that seamlessly transfer to Cityscapes semantic segmentation. The code and models are available at https://github.com/DingXiaoH/RepMLP.

----

## [61] DyRep: Bootstrapping Training with Dynamic Re-parameterization

**Authors**: *Tao Huang, Shan You, Bohan Zhang, Yuxuan Du, Fei Wang, Chen Qian, Chang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00067](https://doi.org/10.1109/CVPR52688.2022.00067)

**Abstract**:

Structural re-parameterization (Rep) methods achieve noticeable improvements on simple VGG-style networks. Despite the prevalence, current Rep methods simply re-parameterize all operations into an augmented network, including those that rarely contribute to the model's performance. As such, the price to pay is an expensive computational overhead to manipulate these unnecessary behaviors. To eliminate the above caveats, we aim to boot-strap the training with minimal cost by devising a dynamic re-parameterization (DyRep) method, which encodes Rep technique into the training process that dynamically evolves the network structures. Concretely, our proposal adaptively finds the operations which contribute most to the loss in the network, and applies Rep to enhance their representational capacity. Besides, to suppress the noisy and redundant operations introduced by Rep, we devise a de-parameterization technique for a more compact re-parameterization. With this regard, DyRep is more efficient than Rep since it smoothly evolves the given network instead of constructing an over-parameterized network. Experimental results demonstrate our effectiveness, e.g., DyRep improves the accuracy of ResNet-18 by 2.04% on ImageNet and reduces 22% runtime over the baseline. Code is avail-able at: https://github.com/hunto/DyRep.

----

## [62] Quarantine: Sparsity Can Uncover the Trojan Attack Trigger for Free

**Authors**: *Tianlong Chen, Zhenyu Zhang, Yihua Zhang, Shiyu Chang, Sijia Liu, Zhangyang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00068](https://doi.org/10.1109/CVPR52688.2022.00068)

**Abstract**:

Trojan attacks threaten deep neural networks (DNNs) by poisoning them to behave normally on most samples, yet to produce manipulated results for inputs attached with a particular trigger. Several works attempt to detect whether a given DNN has been injected with a specific trigger during the training. In a parallel line of research, the lottery ticket hypothesis reveals the existence of sparse sub-networks which are capable of reaching competitive performance as the dense network after independent training. Connecting these two dots, we investigate the problem of Trojan DNN detection from the brand new lens of sparsity, even when no clean training data is available. Our crucial observation is that the Trojan features are significantly more stable to network pruning than benign features. Leveraging that, we propose a novel Trojan network detection regime: first locating a “winning Trojan lottery ticket” which preserves nearly full Trojan information yet only chance-level performance on clean inputs; then recovering the trigger embedded in this already isolated sub-network. Extensive experiments on various datasets, i.e., CIFAR-10, CIFAR-100, and ImageNet, with different network architectures, i.e., VGG-16, ResNet-18, ResNet-20s, and DenseNet-100 demonstrate the effectiveness of our proposal. Codes are available at https://github.com/VITA-Group/Backdoor-LTH.

----

## [63] Condensing CNNs with Partial Differential Equations

**Authors**: *Anil Kag, Venkatesh Saligrama*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00069](https://doi.org/10.1109/CVPR52688.2022.00069)

**Abstract**:

Convolutional neural networks (CNNs) rely on the depth of the architecture to obtain complex features. It results in computationally expensive models for low-resource IoT devices. Convolutional operators are local and restricted in the receptive field, which increases with depth. We explore partial differential equations (PDEs) that offer a global receptive field without the added overhead of maintaining large kernel convolutional filters. We propose a new feature layer, called the Global layer, that enforces PDE constraints on the feature maps, resulting in rich features. These constraints are solved by embedding iterative schemes in the network. The proposed layer can be embedded in any deep CNN to transform it into a shallower network. Thus, resulting in compact and computationally efficient architectures achieving similar performance as the original network. Our experimental evaluation demonstrates that architectures with global layers require 2 - 5 × less computational and storage budget without any significant loss in performance.

----

## [64] Deep Equilibrium Optical Flow Estimation

**Authors**: *Shaojie Bai, Zhengyang Geng, Yash Savani, J. Zico Kolter*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00070](https://doi.org/10.1109/CVPR52688.2022.00070)

**Abstract**:

Many recent state-of-the-art (SOTA) optical flow models use finite-step recurrent update operations to emulate traditional algorithms by encouraging iterative refinements toward a stable flow estimation. However, these RNNs impose large computation and memory overheads, and are not directly trained to model such “stable estimation”. They can converge poorly and thereby suffer from performance degradation. To combat these drawbacks, we propose deep equilibrium (DEQ)flow estimators, an approach that directly solves for the flow as the infinite-level fixed point of an implicit layer (using any black-box solver) [3], and differentiates through this fixed point analytically (thus requiring 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(1)$</tex>
 training memory). This implicit-depth approach is not predicated on any specific model, and thus can be applied to a wide range of SOTA flow estimation model designs (e.g., RAFT [1] and GMA [2]). The use of these DEQflow estimators allows us to compute the flow faster using, e.g., fixed-point reuse and inexact gradients, consumes 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$4\sim 6\times less$</tex>
 training memory than the recurrent counterpart, and achieves better results with the same computation budget. In addition, we propose a novel, sparse fixed-point correction scheme to stabilize our DEQ flow estimators, which addresses a longstanding challenge for DEQ models in general. We test our approach in various realistic settings and show that it improves SOTA methods on Sintel and KITTI datasets with substantially better computational and memory efficiency.

----

## [65] Frame Averaging for Equivariant Shape Space Learning

**Authors**: *Matan Atzmon, Koki Nagano, Sanja Fidler, Sameh Khamis, Yaron Lipman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00071](https://doi.org/10.1109/CVPR52688.2022.00071)

**Abstract**:

The task of shape space learning involves mapping a train set of shapes to and from a latent representation space with good generalization properties. Often, real-world collections of shapes have symmetries, which can be defined as transformations that do not change the essence of the shape. A natural way to incorporate symmetries in shape space learning is to ask that the mapping to the shape space (encoder) and mapping from the shape space (decoder) are equivariant to the relevant symmetries. In this paper, we present a framework for incorporating equivariance in encoders and decoders by introducing two contributions: (i) adapting the recent Frame Averaging (FA) framework for building generic, efficient, and maximally expressive Equivariant autoencoders; and (ii) constructing autoencoders equivariant to piecewise Euclidean motions applied to different parts of the shape. To the best of our knowledge, this is the first fully piecewise Euclidean equivariant autoencoder construction. Training our framework is simple: it uses standard reconstruction losses, and does not require the introduction of new losses. Our architectures are built of standard (backbone) architectures with the appropriate frame averaging to make them equivariant. Testing our framework on both rigid shapes dataset using implicit neural representations, and articulated shape datasets using mesh-based neural networks show state of the art generalization to unseen test shapes, improving relevant baselines by a large margin. In particular, our method demonstrates significant improvement in generalizing to unseen articulated poses.

----

## [66] Dual-Generator Face Reenactment

**Authors**: *Gee-Sern Hsu, Chun-Hung Tsai, Hung-Yi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00072](https://doi.org/10.1109/CVPR52688.2022.00072)

**Abstract**:

We propose the Dual-Generator (DG) network for largepose face reenactment. Given a source face and a reference face as inputs, the DG network can generate an output face that has the same pose and expression as the reference face, and has the same identity as the source face. As most approaches do not particularly consider large-pose reenactment, the proposed approach addresses this issue by incorporating a 3D landmark detector into the framework and considering a loss function to capture visible local shape variation across large pose. The DG network consists of two modules, the ID-preserving Shape Generator (IDSG) and the Reenacted Face Generator (RFG). The IDSG encodes the 3D landmarks of the reference face into a reference landmark code, and encodes the source face into a source face code. The reference landmark code and the source face code are concatenated and decoded to a set of target landmarks that exhibits the pose and expression of the reference face and preserves the identity of the source face. The RFG is partially built on the StarGAN2 generator with modifications on the input and layer settings, and with a facial style encoder added in. Given the target landmarks made by the IDSG and the source face as inputs, the RFG generates the target face with the desired identity, pose and expression. We evaluate our approach on the RaFD, MPIE, VoxCeleb1, and VoxCeleb2 benchmarks and compare with state-of-the-art methods.

----

## [67] Convolution of Convolution: Let Kernels Spatially Collaborate

**Authors**: *Rongzhen Zhao, Jian Li, Zhenzhi Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00073](https://doi.org/10.1109/CVPR52688.2022.00073)

**Abstract**:

In the biological visual pathway especially the retina, neurons are tiled along spatial dimensions with the electrical coupling as their local association, while in a convolution layer, kernels are placed along the channel dimension singly. We propose convolution of convolution, associating kernels in a layer and letting them collaborate spatially. With this method, a layer can provide feature maps with extra transformations and learn its kernels together instead of isolatedly. It is only used during training, bringing in negligible extra costs; then it can be re-parameterized to common convolution before testing, boosting performance gratuitously in tasks like classification, detection and segmentation. Our method works even better when larger receptive fields are demanded. The code is available on site: https://github.com/Genera1Z/ConvolutionOfConvolution.

----

## [68] SASIC: Stereo Image Compression with Latent Shifts and Stereo Attention

**Authors**: *Matthias Wödlinger, Jan Kotera, Jan Xu, Robert Sablatnig*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00074](https://doi.org/10.1109/CVPR52688.2022.00074)

**Abstract**:

We propose a learned method for stereo image compression that leverages the similarity of the left and right images in a stereo pair due to overlapping fields of view. The left image is compressed by a learned compression method based on an autoencoder with a hyperprior entropy model. The right image uses this information from the previously encoded left image in both the encoding and decoding stages. In particular, for the right image, we encode only the residual of its latent representation to the optimally shifted latent of the left image. On top of that, we also employ a stereo attention module to connect left and right images during decoding. The performance of the proposed method is evaluated on two benchmark stereo image datasets (Cityscapes and InStereo2K) and outperforms previous stereo image compression methods while being significantly smaller in model size.

----

## [69] RADU: Ray-Aligned Depth Update Convolutions for ToF Data Denoising

**Authors**: *Michael Schelling, Pedro Hermosilla, Timo Ropinski*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00075](https://doi.org/10.1109/CVPR52688.2022.00075)

**Abstract**:

Time-of-Flight (ToF) cameras are subject to high levels of noise and distortions due to Multi-Path-Interference (MPI). While recent research showed that 2D neural networks are able to outperform previous traditional State-of-the-Art (SOTA) methods on correcting ToF-Data, little research on learning-based approaches has been done to make direct use of the 3D information present in depth images. In this paper, we propose an iterative correcting approach operating in 3D space, that is designed to learn on 2.5D data by enabling 3D point convolutions to correct the points’ positions along the view direction. As labeled real world data is scarce for this task, we further train our network with a self-training approach on unlabeled real world data to account for real world statistics. We demonstrate that our method is able to outperform SOTA methods on several datasets, including two real world datasets and a new large-scale synthetic data set introduced in this paper.

----

## [70] Co-domain Symmetry for Complex-Valued Deep Learning

**Authors**: *Utkarsh Singhal, Yifei Xing, Stella X. Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00076](https://doi.org/10.1109/CVPR52688.2022.00076)

**Abstract**:

We study complex-valued scaling as a type of symmetry natural and unique to complex-valued measurements and representations. Deep Complex Networks (DCN) extend real-valued algebra to the complex domain without addressing complex-valued scaling. SurReal extends manifold learning to the complex plane, achieving scaling invariance with manifold distances that discard phase information. Treating complex-valued scaling as a co-domain transformation, we design novel equivariant/invariant layer functions and architectures that exploit co-domain symmetry. We also propose novel complex-valued representations of RGB images, where complex-valued scaling indicates hue shift or correlated changes across color channels. Benchmarked on MSTAR, CIFAR10, CIFAR100, and SVHN, our co-domain symmetric (CDS) classifiers deliver higher accuracy, better generalization, more robustness to co-domain transformations, and lower model bias and variance than DCN and SurReal with far fewer parameters.

----

## [71] Paramixer: Parameterizing Mixing Links in Sparse Factors Works Better than Dot-Product Self-Attention

**Authors**: *Tong Yu, Ruslan Khalitov, Lei Cheng, Zhirong Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00077](https://doi.org/10.1109/CVPR52688.2022.00077)

**Abstract**:

Self-Attention is a widely used building block in neural modeling to mix long-range data elements. Most self-attention neural networks employ pairwise dot-products to specify the attention coefficients. However, these methods require O(N
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) computing cost for sequence length N. Even though some approximation methods have been introduced to relieve the quadratic cost, the performance of the dot-product approach is still bottlenecked by the lowrank constraint in the attention matrix factorization. In this paper, we propose a novel scalable and effective mixing building block called Paramixer. Our method factorizes the interaction matrix into several sparse matrices, where we parameterize the non-zero entries by MLPs with the data elements as input. The overall computing cost of the new building block is as low as O(N log N). Moreover, all factorizing matrices in Paramixer are full-rank, so it does not suffer from the low-rank bottleneck. We have tested the new method on both synthetic and various real-world long sequential data sets and compared it with several state-of-the-art attention networks. The experimental results show that Paramixer has better performance in most learning tasks.

----

## [72] Compressing Models with Few Samples: Mimicking then Replacing

**Authors**: *Huanyu Wang, Junjie Liu, Xin Ma, Yang Yong, Zhenhua Chai, Jianxin Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00078](https://doi.org/10.1109/CVPR52688.2022.00078)

**Abstract**:

Few-sample compression aims to compress a big redundant model into a small compact one with only few samples. If we fine-tune models with these limited few samples directly, models will be vulnerable to overfit and learn almost nothing. Hence, previous methods optimize the compressed model layer-by-layer and try to make every layer have the same outputs as the corresponding layer in the teacher model, which is cumbersome. In this paper, we propose a new framework named Mimicking then Replacing (MiR) for few-sample compression, which firstly urges the pruned model to output the same features as the teacher's in the penultimate layer, and then replaces teacher's layers before penultimate with a well-tuned compact one. Unlike previous layer-wise reconstruction methods, our MiR optimizes the entire network holistically, which is not only simple and effective, but also unsupervised and general. MiR outperforms previous methods with large margins. Codes is available at https://github.com/cjnjuwhy/MiR.

----

## [73] Total Variation Optimization Layers for Computer Vision

**Authors**: *Raymond A. Yeh, Yuan-Ting Hu, Zhongzheng Ren, Alexander G. Schwing*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00079](https://doi.org/10.1109/CVPR52688.2022.00079)

**Abstract**:

Optimization within a layer of a deep-net has emerged as a new direction for deep-net layer design. However, there are two main challenges when applying these layers to computer vision tasks: (a) which optimization problem within a layer is useful?; (b) how to ensure that computation within a layer remains efficient? To study question (a), in this work, we propose total variation (TV) minimization as a layer for computer vision. Motivated by the success of total variation in image processing, we hypothesize that TV as a layer provides useful inductive bias for deep-nets too. We study this hypothesis on five computer vision tasks: image classification, weakly supervised object localization, edge-preserving smoothing, edge detection, and image denoising, improving over existing baselines. To achieve these results we had to address question (b): we developed a GPU-based projected-Newton method which is 37× faster than existing solutions.

----

## [74] AIM: an Auto-Augmenter for Images and Meshes

**Authors**: *Vinit Veerendraveer Singh, Chandra Kambhamettu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00080](https://doi.org/10.1109/CVPR52688.2022.00080)

**Abstract**:

Data augmentations are commonly used to increase the robustness of deep neural networks. In most contemporary research, the networks do not decide the augmentations; they are task-agnostic, and grid search determines their magnitudes. Furthermore, augmentations applicable to lower-dimensional data do not easily extend to higher-dimensional data and vice versa. This paper presents an auto-augmenter for images and meshes (AIM) that easily incorporates into neural networks at training and inference times. It Jointly optimizes with the network to produce constrained, non-rigid deformations in the data. AIM predicts sample-aware deformations suited for a task, and our experiments confirm its effectiveness with various networks.

----

## [75] Recurrent Variational Network: A Deep Learning Inverse Problem Solver applied to the task of Accelerated MRI Reconstruction

**Authors**: *George Yiasemis, Jan-Jakob Sonke, Clarisa Sánchez, Jonas Teuwen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00081](https://doi.org/10.1109/CVPR52688.2022.00081)

**Abstract**:

Magnetic Resonance Imaging can produce detailed images of the anatomy and physiology of the human body that can assist doctors in diagnosing and treating pathologies such as tumours. However, MRI suffers from very long acquisition times that make it susceptible to patient motion artifacts and limit its potential to deliver dynamic treatments. Conventional approaches such as Parallel Imaging and Compressed Sensing allow for an increase in MRI acquisition speed by reconstructing MR images from sub-sampled MRI data acquired using multiple receiver coils. Recent advancements in Deep Learning combined with Parallel Imaging and Compressed Sensing techniques have the potential to produce high-fidelity reconstructions from highly accelerated MRI data. In this work we present a novel Deep Learning-based Inverse Problem solver applied to the task of Accelerated MRI Reconstruction, called the Recurrent Variational Network (RecurrentVarNet), by exploiting the properties of Convolutional Recurrent Neural Networks and unrolled algorithms for solving Inverse Problems. The RecurrentVarNet consists of multiple recurrent blocks, each responsible for one iteration of the unrolled variational optimization scheme for solving the inverse problem of multi-coil Accelerated MRI Reconstruction. Contrary to traditional approaches, the optimization steps are performed in the observation domain (k-space) instead of the image domain. Each block of the RecurrentVarNet refines the observed k-space and comprises a data consistency term and a recurrent unit which takes as input a learned hidden state and the prediction of the previous block. Our proposed method achieves new state of the art qualitative and quantitative reconstruction results on 5-fold and 10-fold accelerated data from a public multi-coil brain dataset, outperforming previous conventional and deep learning-based approaches. Our code is publicly available at https://github.com/NKI-AI/direct.

----

## [76] Deep orientation-aware functional maps: Tackling symmetry issues in Shape Matching

**Authors**: *Nicolas Donati, Etienne Corman, Maks Ovsjanikov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00082](https://doi.org/10.1109/CVPR52688.2022.00082)

**Abstract**:

State-of-the-art fully intrinsic network for non-rigid shape matching are unable to disambiguate between shape inner symmetries. Meanwhile, recent advances in the functional map framework allow to enforce orientation preservation using a functional representation for tangent vector field transfer, through so-called complex functional maps. Using this representation, we propose a new deep learning approach to learn orientation-aware features in afully unsupervised setting. Our architecture is built on DiffusionNet, which makes our method robust to discretization changes, while adding a vector-field-based loss, which promotes orientation preservation without using (often unstable) extrinsic descriptors. Our source code is available at: https://github.com/nicolasdonati/DUO-FM.

----

## [77] Weakly-supervised Metric Learning with Cross-Module Communications for the Classification of Anterior Chamber Angle Images

**Authors**: *Jingqi Huang, Yue Ning, Dong Nie, Linan Guan, Xiping Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00083](https://doi.org/10.1109/CVPR52688.2022.00083)

**Abstract**:

As the basis for developing glaucoma treatment strategies, Anterior Chamber Angle (ACA) evaluation is usually dependent on experts' Judgements. However, experienced ophthalmologists needed for these Judgements are not widely available. Thus, computer-aided ACA evaluations become a pressing and efficient solution for this issue. In this paper, we propose a novel end-to-end frame-work GCNet for automated Glaucoma Classification based on ACA images or other Glaucoma-related medical images. We first collect and label an ACA image dataset with some pixel-level annotations. Next, we introduce a segmentation module and an embedding module to enhance the performance of classifying ACA images. Within GCNet, we design a Cross-Module Aggregation Net (CMANet) which is a weakly-supervised metric learning network to capture contextual information exchanging across these modules. We conduct experiments on the ACA dataset and two public datasets REFUGE and SIGF. Our experimental results demonstrate that GCNet outperforms several state-of-the-art deep models in the tasks of glaucoma medical image classifications. The source code of GCNet can be found at https://github.com/Jingqi-H/GCNet.

----

## [78] Delving into the Estimation Shift of Batch Normalization in a Network

**Authors**: *Lei Huang, Yi Zhou, Tian Wang, Jie Luo, Xianglong Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00084](https://doi.org/10.1109/CVPR52688.2022.00084)

**Abstract**:

Batch normalization (BN) is a milestone technique in deep learning. It normalizes the activation using mini-batch statistics during training but the estimated population statistics during inference. This paper focuses on investigating the estimation of population statistics. We define the estimation shift magnitude of BN to quantitatively measure the difference between its estimated population statistics and expected ones. Our primary observation is that the estimation shift can be accumulated due to the stack of BN in a network, which has detriment effects for the test performance. We further find a batch-free normalization (BFN) can block such an accumulation of estimation shift. These observations motivate our design of XBNBlock that replace one BN with BFN in the bottleneck block of residual-style networks. Experiments on the ImageNet and COCO benchmarks show that XBNBlock consistently improves the performance of different architectures, including ResNet and ResNeXt, by a significant margin and seems to be more robust to distribution shift.

----

## [79] Generalizing Interactive Backpropagating Refinement for Dense Prediction Networks

**Authors**: *Fanqing Lin, Brian Price, Tony R. Martinez*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00085](https://doi.org/10.1109/CVPR52688.2022.00085)

**Abstract**:

As deep neural networks become the state-of-the-art approach in the field of computer vision for dense prediction tasks, many methods have been developed for automatic estimation of the target outputs given the visual inputs. Although the estimation accuracy of the proposed automatic methods continues to improve, interactive refinement is oftentimes necessary for further correction. Recently, feature backpropagating refinement scheme [25] (f-BRS) has been proposed for the task of interactive segmentation, which enables efficient optimization of a small set of auxiliary variables inserted into the pretrained network to produce object segmentation that better aligns with user inputs. However, the proposed auxiliary variables only contain channel-wise scale and bias, limiting the optimization to global refinement only. In this work, in order to generalize backpropagating refinement for a wide range of dense prediction tasks, we introduce a set of G-BRS (Generalized Backpropagating Refinement Scheme) layers that enable both global and localized refinement for the following tasks: interactive segmentation, semantic segmentation, image matting and monocular depth estimation. Experiments on SBD, Cityscapes, Mapillary Vista, Composition-1k and NYU-Depth-V2 show that our method can successfully generalize and significantly improve performance of existing pretrained state-of-the-art models with only a few clicks.

----

## [80] Brain-inspired Multilayer Perceptron with Spiking Neurons

**Authors**: *Wenshuo Li, Hanting Chen, Jianyuan Guo, Ziyang Zhang, Yunhe Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00086](https://doi.org/10.1109/CVPR52688.2022.00086)

**Abstract**:

Recently, Multilayer Perceptron (MLP) becomes the hotspot in the field of computer vision tasks. Without in-ductive bias, MLPs perform well on feature extraction and achieve amazing results. However, due to the simplic-ity of their structures, the performance highly depends on the local features communication machenism. To further improve the performance of MLP, we introduce information communication mechanisms from brain-inspired neu-ral networks. Spiking Neural Network (SNN) is the most famous brain-inspired neural network, and achieve great success on dealing with sparse data. Leaky Integrate and Fire (LIF) neurons in SNNs are used to communicate be-tween different time steps. In this paper, we incorporate the machanism of LIF neurons into the MLP models, to achieve better accuracy without extra FLOPs. We pro-pose a full-precision LIF operation to communicate be-tween patches, including horizontal LIF and vertical LIF in different directions. We also propose to use group LIF to extract better local features. With LIF modules, our SNN-MLP model achieves 81.9%, 83.3% and 83.5% top-1 accuracy on ImageNet dataset with only 4.4G, 8.5G and 15.2G FLOPs, respectively, which are state-of-the-art re-sults as far as we know. The source code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/snn_mlp.

----

## [81] Smooth Maximum Unit: Smooth Activation Function for Deep Networks using Smoothing Maximum Technique

**Authors**: *Koushik Biswas, Sandeep Kumar, Shilpak Banerjee, Ashish Kumar Pandey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00087](https://doi.org/10.1109/CVPR52688.2022.00087)

**Abstract**:

Deep learning researchers have a keen interest in proposing new novel activation functions that can boost neural network performance. A good choice of activation function can have a significant effect on improving network performance and training dynamics. Rectified Linear Unit (ReLU) is a popular hand-designed activation function and is the most common choice in the deep learning community due to its simplicity though ReLU has some drawbacks. In this paper, we have proposed two new novel activation functions based on approximation of the maximum function, and we call these functions Smooth Maximum Unit (SMU and SMU-1). We show that SMU and SMU-1 can smoothly approximate ReLU, Leaky ReLU, or more general Maxout family, and GELU is a particular case of SMU. Replacing ReLU by SMU, Top-1 classification accuracy improves by 6.22%, 3.39%, 3.51%, and 3.08% on the CIFAR100 dataset with ShuffleNet V2, PreActResNet-50, ResNet-50, and SeNet-50 models respectively. Also, our experimental evaluation shows that SMU and SMU-1 improve network performance in a variety of deep learning tasks like image classification, object detection, semantic segmentation, and machine translation compared to widely used activation functions.

----

## [82] Revisiting Weakly Supervised Pre-Training of Visual Perception Models

**Authors**: *Mannat Singh, Laura Gustafson, Aaron Adcock, Vinicius de Freitas Reis, Bugra Gedik, Raj Prateek Kosaraju, Dhruv Mahajan, Ross B. Girshick, Piotr Dollár, Laurens van der Maaten*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00088](https://doi.org/10.1109/CVPR52688.2022.00088)

**Abstract**:

Model pre-training is a cornerstone of modern visual recognition systems. Although fully supervised pre-training on datasets like ImageNet is still the de-facto standard, recent studies suggest that large-scale weakly supervised pretraining can outperform fully supervised approaches. This paper revisits weakly-supervised pre-training of models using hashtag supervision with modern versions of residual networks and the largest-ever dataset of images and corresponding hashtags. We study the performance of the resulting models in various transfer-learning settings including zero-shot transfer. We also compare our models with those obtained via large-scale self-supervised learning. We find our weakly-supervised models to be very competitive across all settings, and find they substantially outperform their self-supervised counterparts. We also include an investigation into whether our models learned potentially troubling associations or stereotypes. Overall, our results provide a compelling argument for the use of weakly supervised learning in the development of visual recognition systems. Our models, Supervised Weakly through hashtAGs (SWAG), are available publicly.

----

## [83] On the Integration of Self-Attention and Convolution

**Authors**: *Xuran Pan, Chunjiang Ge, Rui Lu, Shiji Song, Guanfu Chen, Zeyi Huang, Gao Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00089](https://doi.org/10.1109/CVPR52688.2022.00089)

**Abstract**:

Convolution and self-attention are two powerful techniques for representation learning, and they are usually considered as two peer approaches that are distinct from each other. In this paper, we show that there exists a strong underlying relation between them, in the sense that the bulk of computations of these two paradigms are in fact done with the same operation. Specifically, we first show that a traditional convolution with kernel size k × k can be decomposed into k
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 individual 1 × 1 convolutions, followed by shift and summation operations. Then, we interpret the projections of queries, keys, and values in self-attention module as multiple 1 × 1 convolutions, followed by the computation of attention weights and aggregation of the values. Therefore, the first stage of both two modules comprises the similar operation. More importantly, the first stage contributes a dominant computation complexity (square of the channel size) comparing to the second stage. This observation naturally leads to an elegant integration of these two seemingly distinct paradigms, i.e., a mixed model that enjoys the benefit of both self-Attention and Convolution (ACmix), while having minimum compu-tational overhead compared to the pure convolution or self-attention counterpart. Extensive experiments show that our model achieves consistently improved results over com-petitive baselines on image recognition and downstream tasks. Code and pre-trained models will be released at https://github.com/LeapLabTHU/ACmix and https://gitee.com/mindspore/models.

----

## [84] Hire-MLP: Vision MLP via Hierarchical Rearrangement

**Authors**: *Jianyuan Guo, Yehui Tang, Kai Han, Xinghao Chen, Han Wu, Chao Xu, Chang Xu, Yunhe Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00090](https://doi.org/10.1109/CVPR52688.2022.00090)

**Abstract**:

Previous vision MLPs such as MLP-Mixer and ResMLP accept linearly flattened image patches as input, making them inflexible for different input sizes and hard to capture spatial information. Such approach withholds MLPs from getting comparable performance with their transformer-based counterparts and prevents them from becoming a general backbone for computer vision. This paper presents Hire-MLP, a simple yet competitive vision MLP architecture via Hierarchical rearrangement, which contains two levels of rearrangements. Specifically, the inner-region rearrangement is proposed to capture local information inside a spatial region, and the cross-region rearrangement is proposed to enable information communication between different regions and capture global context by circularly shifting all tokens along spatial directions. Extensive experiments demonstrate the effectiveness of Hire-MLP as a versatile backbone for various vision tasks. In particular, Hire-MLP achieves competitive results on image classification, object detection and semantic segmentation tasks, e.g., 83.8% top-1 accuracy on ImageNet, 51.7% box AP and 44.8% mask AP on COCO val2017, and 49.9% mIoU on ADE20K, surpassing previous transformer-based and MLP-based models with better trade-off for accuracy and throughput.

----

## [85] Stable Long-Term Recurrent Video Super-Resolution

**Authors**: *Benjamin Naoto Chiche, Arnaud Woiselle, Joana Frontera-Pons, Jean-Luc Starck*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00091](https://doi.org/10.1109/CVPR52688.2022.00091)

**Abstract**:

Recurrent models have gained popularity in deep learning (DL) based video super-resolution (VSR), due to their increased computational efficiency, temporal receptive field and temporal consistency compared to sliding-window based models. However, when inferring on long video sequences presenting low motion (i.e. in which some parts of the scene barely move), recurrent models diverge through recurrent processing, generating high frequency artifacts. To the best of our knowledge, no study about VSR pointed out this instability problem, which can be critical for some real-world applications. Video surveillance is a typical example where such artifacts would occur, as both the camera and the scene stay static for a long time. In this work, we expose instabilities of existing recurrent VSR networks on long sequences with low motion. We demonstrate it on a new long sequence dataset Quasi-Static Video Set, that we have created. Finally, we introduce a new framework of recurrent VSR networks that is both stable and competitive, based on Lipschitz stability theory. We propose a new recurrent VSR network, coined Middle Recurrent Video Super-Resolution (MRVSR), based on this framework. We empirically show its competitive performance on long sequences with low motion.

----

## [86] Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation

**Authors**: *Aming Wu, Cheng Deng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00092](https://doi.org/10.1109/CVPR52688.2022.00092)

**Abstract**:

In this paper, we are concerned with enhancing the generalization capability of object detectors. And we consider a realistic yet challenging scenario, namely Single-Domain Generalized Object Detection (Single-DGOD), which aims to learn an object detector that performs well on many unseen target domains with only one source domain for training. Towards Single-DGOD, it is important to extract domain-invariant representations (DIR) containing intrinsical object characteristics, which is beneficial for improving the robustness for unseen domains. Thus, we present a method, i.e., cyclic-disentangled self-distillation, to disentangle DIR from domain-specific representations without the supervision of domain-related annotations (e.g., domain labels). Concretely, a cyclic-disentangled module is first proposed to cyclically extract DIR from the input visual features. Through the cyclic operation, the disentangled ability can be promoted without the reliance on domain-related annotations. Then, taking the DIR as the teacher, we design a self-distillation module to further enhance the generalization ability. In the experiments, our method is evaluated in urban-scene object detection. Experimental results of five weather conditions show that our method obtains a significant performance gain over baseline methods. Particularly, for the night-sunny scene, our method outperforms baselines by 3%, which indicates that our method is instrumental in enhancing generalization ability. Data and code are available at https://github.com/AmingWu/Single-DgoD.

----

## [87] Progressive End-to-End Object Detection in Crowded Scenes

**Authors**: *Anlin Zheng, Yuang Zhang, Xiangyu Zhang, Xiaojuan Qi, Jian Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00093](https://doi.org/10.1109/CVPR52688.2022.00093)

**Abstract**:

In this paper, we propose a new query-based detection framework for crowd detection. Previous query-based detectors suffer from two drawbacks: first, multiple predictions will be inferred for a single object, typically in crowded scenes; second, the performance saturates as the depth of the decoding stage increases. Benefiting from the nature of the one-to-one label assignment rule, we propose a progressive predicting method to address the above issues. Specifically, we first select accepted queries prone to generate true positive predictions, then refine the rest noisy queries according to the previously accepted predictions. Experiments show that our method can significantly boost the performance of query-based detectors in crowded scenes. Equipped with our approach, Sparse RCNN achieves 92.0% AP, 41.4% MR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">−2</sup>
 and 83.2% JI on the challenging CrowdHuman [35] dataset, outperforming the box-based method MIP [8] that specifies in handling crowded scenarios. Moreover, the proposed method, robust to crowdedness, can still obtain consistent improvements on moderately and slightly crowded datasets like CityPersons [47] and COCO [26]. Code will be made publicly available at https://github.com/megvii-model/Iter-E2EDET.

----

## [88] Zero-Shot Text-Guided Object Generation with Dream Fields

**Authors**: *Ajay Jain, Ben Mildenhall, Jonathan T. Barron, Pieter Abbeel, Ben Poole*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00094](https://doi.org/10.1109/CVPR52688.2022.00094)

**Abstract**:

We combine neural rendering with multi-modal image and text representations to synthesize diverse 3D objects solely from natural language descriptions. Our method, Dream Fields, can generate the geometry and color of a wide range of objects without 3D supervision. Due to the scarcity of diverse, captioned 3D data, prior methods only generate objectsfrom a handful of categories, such as ShapeNet. Instead, we guide generation with image-text models pre-trained on large datasets of captioned images from the web. Our method optimizes a Neural Radiance Field from many camera views so that rendered images score highly with a target caption according to a pre-trained CLIP model. To improve fidelity and visual quality, we introduce simple geometric priors, including sparsity-inducing transmittance regularization, scene bounds, and new MLP architectures. In experiments, Dream Fields produce realistic, multi-view consistent object geometry and color from a variety of natural language captions.

----

## [89] ISNet: Shape Matters for Infrared Small Target Detection

**Authors**: *Mingjin Zhang, Rui Zhang, Yuxiang Yang, Haichen Bai, Jing Zhang, Jie Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00095](https://doi.org/10.1109/CVPR52688.2022.00095)

**Abstract**:

Infrared small target detection (IRSTD) refers to extracting small and dim targets from blurred backgrounds, which has a wide range of applications such as traffic management and marine rescue. Due to the low signal-to-noise ratio and low contrast, infrared targets are easily submerged in the background of heavy noise and clutter. How to detect the precise shape information of infrared targets remains challenging. In this paper, we propose a novel infrared shape network (ISNet), where Taylor finite difference (TFD) -inspired edge block and two-orientation attention aggregation (TOAA) block are devised to address this problem. Specifically, TFD-inspired edge block aggregates and enhances the comprehensive edge information from different levels, in order to improve the contrast between target and background and also lay a foundation for extracting shape information with mathematical interpretation. TOAA block calculates the lowlevel information with attention mechanism in both row and column directions and fuses it with the high-level information to capture the shape characteristic of targets and suppress noises. In addition, we construct a new benchmark consisting of 1, 000 realistic images in various target shapes, different target sizes, and rich clutter backgrounds with accurate pixel-level annotations, called IRSTD-1k. Experiments on public datasets and IRSTD-1 k demonstrate the superiority of our approach over representative state-of-the-art IRSTD methods. The dataset and code are available at github.com/RuiZhang97/ISNet.

----

## [90] Pseudo-Stereo for Monocular 3D Object Detection in Autonomous Driving

**Authors**: *Yi-Nan Chen, Hang Dai, Yong Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00096](https://doi.org/10.1109/CVPR52688.2022.00096)

**Abstract**:

Pseudo-LiDAR 3D detectors have made remarkable progress in monocular 3D detection by enhancing the capability of perceiving depth with depth estimation networks, and using LiDAR-based 3D detection architectures. The Advanced stereo 3D detectors can also accurately localize 3D objects. The gap in image-to-image generation for stereo views is much smaller than that in image-to-LiDAR generation. Motivated by this, we propose a Pseudo-Stereo 3D detection framework with three novel virtual view generation methods, including image-level generation, feature-level generation, and feature-clone, for detecting 3D objects from a single image. Our analysis of depth-aware learning shows that the depth loss is effective in only feature-level virtual view generation and the estimated depth map is effective in both image-level and feature-level in our framework. We propose a disparity-wise dynamic convolution with dynamic kernels sampled from the disparity feature map to filter the features adaptively from a single image for generating virtual image features, which eases the feature degradation caused by the depth estimation errors. Till submission (November 18, 2021), our Pseudo-Stereo 3D detection framework ranks 1 st on car, pedestrian, and cyclist among the monocular 3D detectors with publications on the KITTI-3D benchmark. The code is released at https://github.com/revisitq/Pseudo-Stereo-3D.

----

## [91] CLRNet: Cross Layer Refinement Network for Lane Detection

**Authors**: *Tu Zheng, Yifei Huang, Yang Liu, Wenjian Tang, Zheng Yang, Deng Cai, Xiaofei He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00097](https://doi.org/10.1109/CVPR52688.2022.00097)

**Abstract**:

Lane is critical in the vision navigation system of the intelligent vehicle. Naturally, lane is a traffic sign with high-level semantics, whereas it owns the specific local pattern which needs detailed low-level features to localize accurately. Using different feature levels is of great importance for accurate lane detection, but it is still under-explored. In this work, we present Cross Layer Refinement Network (CLRNet) aiming at fully utilizing both high-level and low-level features in lane detection. In particular, it first detects lanes with high-level semantic features then performs refinement based on low-level features. In this way, we can exploit more contextual information to detect lanes while leveraging local detailed lane features to improve localization accuracy. We present ROIGather to gather global context, which further enhances the feature representation of lanes. In addition to our novel network design, we introduce Line IoU loss which regresses the lane line as a whole unit to improve the localization accuracy. Experiments demonstrate that the proposed method greatly outperforms the state-of-the-art lane detection approaches. Code is available at: https://github.com/Turoad/CLRNet.

----

## [92] CAT-Det: Contrastively Augmented Transformer for Multimodal 3D Object Detection

**Authors**: *Yanan Zhang, Jiaxin Chen, Di Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00098](https://doi.org/10.1109/CVPR52688.2022.00098)

**Abstract**:

In autonomous driving, LiDAR point-clouds and RGB images are two major data modalities with complementary cues for 3D object detection. However, it is quite difficult to sufficiently use them, due to large inter-modal discrepancies. To address this issue, we propose a novel framework, namely Contrastively Augmented Transformer for multi-modal 3D object Detection (CAT-Det). Specifically, CAT-Det adopts a two-stream structure consisting of a Pointformer (PT) branch, an Imageformer (IT) branch along with a Cross-Modal Transformer (CMT) module. PT, IT and CMT jointly encode intra-modal and inter-modal long-range contexts for representing an object, thus fully exploring multi-modal information for detection. Furthermore, we propose an effective One-way Multimodal Data Augmentation (OMDA) approach via hierarchical contrastive learning at both the point and object levels, significantly improving the accuracy only by augmenting point-clouds, which is free from complex generation of paired samples of the two modalities. Extensive experiments on the KITTI benchmark show that CAT-Det achieves a new state-of-the-art, highlighting its effectiveness.

----

## [93] Modality-Agnostic Learning for Radar-Lidar Fusion in Vehicle Detection

**Authors**: *Yu-Jhe Li, Jinhyung Park, Matthew O'Toole, Kris Kitani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00099](https://doi.org/10.1109/CVPR52688.2022.00099)

**Abstract**:

Fusion of multiple sensor modalities such as camera, Lidar, and Radar, which are commonly found on autonomous vehicles, not only allows for accurate detection but also robustifies perception against adverse weather conditions and individual sensor failures. Due to inherent sensor characteristics, Radar performs well under extreme weather conditions (snow, rain, fog) that significantly degrade camera and Lidar. Recently, a few works have developed vehicle detection methods fusing Lidar and Radar signals, i.e., MVD-Net. However, these models are typically developed under the assumption that the models always have access to two error-free sensor streams. If one of the sensors is unavailable or missing, the model may fail catastrophically. To mitigate this problem, we propose the Self-Training Multimodal Vehicle Detection Network (ST-MVDNet) which leverages a Teacher-Student mutual learning framework and a simulated sensor noise model used in strong data augmentation for Lidar and Radar. We show that by (1) enforcing output consistency between a Teacher network and a Student network and by (2) introducing missing modalities (strong augmentations) during training, our learned model breaks away from the error-free sensor assumption. This consistency enforcement enables the Student model to handle missing data properly and improve the Teacher model by updating it with the Student model's exponential moving average. Our experiments demonstrate that our proposed learning framework for multi-modal detection is able to better handle missing sensor data during inference. Furthermore, our method achieves new state-of-the-art performance (5% gain) on the Oxford Radar Robotcar dataset under various evaluation settings.

----

## [94] Group Contextualization for Video Recognition

**Authors**: *Yanbin Hao, Hao Zhang, Chong-Wah Ngo, Xiangnan He*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00100](https://doi.org/10.1109/CVPR52688.2022.00100)

**Abstract**:

Learning discriminative representation from the complex spatio-temporal dynamic space is essential for video recognition. On top of those stylized spatio-temporal computational units, further refining the learnt feature with axial contexts is demonstrated to be promising in achieving this goal. However, previous works generally focus on utilizing a single kind of contexts to calibrate entire feature channels and could hardly apply to deal with diverse video activities. The problem can be tackled by using pair-wise spatio-temporal attentions to recompute feature response with cross-axis contexts at the expense of heavy computations. In this paper, we propose an efficient feature refinement method that decomposes the feature channels into several groups and separately refines them with different axial contexts in parallel. We refer this lightweight feature calibration as group contextualization (GC). Specifically, we design a family of efficient element-wise calibrators, i.e., ECal-G/S/T/L, where their axial contexts are information dynamics aggregated from other axes either globally or locally, to contextualize feature channel groups. The GC module can be densely plugged into each residual layer of the off-the-shelf video networks. With little computational overhead, consistent improvement is observed when plugging in GC on different networks. By utilizing calibrators to embed feature with four different kinds of contexts in parallel, the learnt representation is expected to be more resilient to diverse types of activities. On videos with rich temporal variations, empirically GC can boost the performance of 2D-CNN (e.g., TSN and TSM) to a level comparable to the state-of-the-art video networks. Code is available at https://github.com/haoyanbin918/Group-Contextualization.

----

## [95] Learning Transferable Human-Object Interaction Detector with Natural Language Supervision

**Authors**: *Suchen Wang, Yueqi Duan, Henghui Ding, Yap-Peng Tan, Kim-Hui Yap, Junsong Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00101](https://doi.org/10.1109/CVPR52688.2022.00101)

**Abstract**:

It is difficult to construct a data collection including all possible combinations of human actions and interacting objects due to the combinatorial nature of human-object interactions (HOI). In this work, we aim to develop a transferable HOI detector for unseen interactions. Existing HOI detectors often treat interactions as discrete labels and learn a classifier according to a predetermined category space. This is inherently inapt for detecting unseen interactions which are out of the predefined categories. Conversely, we treat independent HOI labels as the natural language supervision of interactions and embed them into a joint visual-and-text space to capture their correlations. More specifically, we propose a new HOI visual encoder to detect the interacting humans and objects, and map them to a joint feature space to perform interaction recognition. Our visual encoder is instantiated as a Vision Transformer with new learnable HOI tokens and a sequence parser to generate unique HOI predictions. It distills and leverages the transferable knowledge from the pretrained CLIP model to perform the zero-shot interaction detection. Experiments on two datasets, SWIG-HOI and HICO-DET, validate that our proposed method can achieve a notable mAP improvement on detecting both seen and unseen HOIs. Our code is available at https://github.com/scwangdyd/promting_hoi.

----

## [96] Accelerating DETR Convergence via Semantic-Aligned Matching

**Authors**: *Gongjie Zhang, Zhipeng Luo, Yingchen Yu, Kaiwen Cui, Shijian Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00102](https://doi.org/10.1109/CVPR52688.2022.00102)

**Abstract**:

The recently developed DEtection TRansformer (DETR) establishes a new object detection paradigm by eliminating a series of hand-crafted components. However, DETR suffers from extremely slow convergence, which increases the training cost significantly. We observe that the slow convergence is largely attributed to the complication in matching object queries with target features in different feature embedding spaces. This paper presents SAM-DETR, a Semantic-Aligned-Matching DETR that greatly accelerates DETR's convergence without sacrificing its accuracy. SAM-DETR addresses the convergence issue from two perspectives. First, it projects object queries into the same embedding space as encoded image features, where the matching can be accomplished efficiently with aligned semantics. Second, it explicitly searches salient points with the most discriminative features for semantic-aligned matching, which further speeds up the convergence and boosts detection accuracy as well. Being like a plug and play, SAM-DETR complements existing convergence solutions well yet only introduces slight computational overhead. Extensive experiments show that the proposed SAM-DETR achieves superior convergence as well as competitive detection accuracy. The implementation codes are publicly available at https://github.com/ZhangGongjie/SAM-DETR.

----

## [97] Efficient Video Instance Segmentation via Tracklet Query and Proposal

**Authors**: *Jialian Wu, Sudhir Yarram, Hui Liang, Tian Lan, Junsong Yuan, Jayan Eledath, Gérard G. Medioni*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00103](https://doi.org/10.1109/CVPR52688.2022.00103)

**Abstract**:

Video Instance Segmentation (VIS) aims to simultaneously classify, segment, and track multiple object instances in videos. Recent clip-level VIS takes a short video clip as input each time showing stronger performance than frame-level VIS (tracking-by-segmentation), as more temporal context from multiple frames is utilized. Yet, most clip-level methods are neither end-to-end learnable nor real-time. These limitations are addressed by the recent VIS transformer (VisTR) [25] which performs VIS end-to-end within a clip. However, VisTR suffers from long training time due to its frame-wise dense attention. In addition, VisTR is not fully end-to-end learnable in multiple video clips as it requires a hand-crafted data association to link instance tracklets between successive clips. This paper proposes EfficientVIS, a fully end-to-end framework with efficient training and inference. At the core are tracklet query and tracklet proposal that associate and segment regions-of-interest (RoIs) across space and time by an iterative query-video interaction. We further propose a correspondence learning that makes tracklets linking between clips end-to-end learnable. Compared to VisTR, EfficientVIS requires 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$15\times$</tex>
 fewer training epochs while achieving state-of-the-art accuracy on the YouTube-VIS benchmark. Meanwhile, our method enables whole video instance segmentation in a single end-to-end pass without data association at all.

----

## [98] Class Re-Activation Maps for Weakly-Supervised Semantic Segmentation

**Authors**: *Zhaozheng Chen, Tan Wang, Xiongwei Wu, Xian-Sheng Hua, Hanwang Zhang, Qianru Sun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00104](https://doi.org/10.1109/CVPR52688.2022.00104)

**Abstract**:

Extracting class activation maps (CAM) is arguably the most standard step of generating pseudo masks for weakly-supervised semantic segmentation (WSSS). Yet, we find that the crux of the unsatisfactory pseudo masks is the binary crossentropy loss (BCE) widely used in CAM. Specifically, due to the sum-over-class pooling nature of BCE, each pixel in CAM may be responsive to multiple classes co-occurring in the same receptive field. As a result, given a class, its hot CAM pixels may wrongly invade the area belonging to other classes, or the non-hot ones may be actually a part of the class. To this end, we introduce an embarrassingly simple yet surprisingly effective method: Reactivating the converged CAM with BCE by using softmax crossentropy loss (SCE), dubbed ReCAM. Given an image, we use CAM to extract the feature pixels of each single class, and use them with the class label to learn another fully-connected layer (after the backbone) with SCE. Once converged, we extract ReCAM in the same way as in CAM. Thanks to the contrastive nature of SCE, the pixel response is disentangled into different classes and hence less mask ambiguity is expected. The evaluation on both PASCAL VOC and MS COCO shows that ReCAM not only generates high-quality masks, but also supports plug-and-play in any CAM variant with little overhead. Our code is public at https://github.com/zhaozhengChenIReCAM.

----

## [99] Democracy Does Matter: Comprehensive Feature Mining for Co-Salient Object Detection

**Authors**: *Siyue Yu, Jimin Xiao, Bingfeng Zhang, Eng Gee Lim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00105](https://doi.org/10.1109/CVPR52688.2022.00105)

**Abstract**:

Co-salient object detection, with the target of detecting co-existed salient objects among a group of images, is gaining popularity. Recent works use the attention mechanism or extra information to aggregate common co-salient features, leading to incomplete even incorrect responses for target objects. In this paper, we aim to mine comprehensive co-salient features with democracy and reduce background interference without introducing any extra information. To achieve this, we design a democratic prototype generation module to generate democratic response maps, covering sufficient co-salient regions and thereby involving more shared attributes of co-salient objects. Then a comprehensive prototype based on the response maps can be generated as a guide for final prediction. To suppress the noisy background information in the prototype, we propose a self-contrastive learning module, where both positive and negative pairs are formed without relying on additional classification information. Besides, we also design a democratic feature enhancement module to further strengthen the co-salient features by readjusting attention values. Extensive experiments show that our model obtains better performance than previous state-of-the-art methods, especially on challenging real-world cases (e.g., for CoCA, we obtain a gain of 2.0% for MAE, 5.4% for maximum F-measure, 2.3% for maximum E-measure, and 3.7% for S-measure) under the same settings. Source code is available at https://github.com/siyueyu/DCFM.

----

## [100] C2 AM: Contrastive learning of Class-agnostic Activation Map for Weakly Supervised Object Localization and Semantic Segmentation

**Authors**: *Jinheng Xie, Jianfeng Xiang, Junliang Chen, Xianxu Hou, Xiaodong Zhao, Linlin Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00106](https://doi.org/10.1109/CVPR52688.2022.00106)

**Abstract**:

While class activation map (CAM) generated by image classification network has been widely used for weakly su-pervised object localization (WSOL) and semantic segmentation (WSSS), such classifiers usually focus on discriminative object regions. In this paper, we propose Contrastive learning for Class-agnostic Activation Map (C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
AM) generation only using unlabeled image data, without the involvement of image-level supervision. The core idea comes from the observation that i) semantic information of fore-ground objects usually differs from their backgrounds; ii) foreground objects with similar appearance or background with similar color/texture have similar representations in the feature space. We form the positive and negative pairs based on the above relations and force the network to disentangle foreground and background with a class-agnostic activation map using a novel contrastive loss. As the network is guided to discriminate cross-image foreground-background, the class-agnostic activation maps learned by our approach generate more complete object regions. We successfully extracted from C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 AM class-agnostic object bounding boxes for object localization and background cues to refine CAM generated by classification network for semantic segmentation. Extensive experiments on CUB-200-2011, ImageNet-1K, and PASCAL VOC2012 datasets show that both WSOL and WSSS can benefit from the proposed C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
AM. Code will be available at https://github.com/CVI-SZUICCAM.

----

## [101] Sketching without Worrying: Noise-Tolerant Sketch-Based Image Retrieval

**Authors**: *Ayan Kumar Bhunia, Subhadeep Koley, Abdullah Faiz Ur Rahman Khilji, Aneeshan Sain, Pinaki Nath Chowdhury, Tao Xiang, Yi-Zhe Song*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00107](https://doi.org/10.1109/CVPR52688.2022.00107)

**Abstract**:

Sketching enables many exciting applications, notably, image retrieval. The fear-to-sketch problem (i.e., “I can't sketch”) has however proven to be fatal for its widespread adoption. This paper tackles this “fear” head on, and for the first time, proposes an auxiliary module for existing retrieval models that predominantly lets the users sketch without having to worry. We first conducted a pilot study that revealed the secret lies in the existence of noisy strokes, but not so much of the “I can't sketch”. We consequently design a stroke subset selector that detects noisy strokes, leaving only those which make a positive contribution towards successful retrieval. Our Reinforcement Learning based formulation quantifies the importance of each stroke present in a given subset, based on the extent to which that stroke contributes to retrieval. When combined with pre-trained retrieval models as a pre-processing module, we achieve a significant gain of 8%-10% over standard baselines and in turn report new state-of-the-art performance. Last but not least, we demonstrate the selector once trained, can also be used in a plug-and-play manner to empower various sketch applications in ways that were not previously possible.

----

## [102] AutoLoss-Zero: Searching Loss Functions from Scratch for Generic Tasks

**Authors**: *Hao Li, Tianwen Fu, Jifeng Dai, Hongsheng Li, Gao Huang, Xizhou Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00108](https://doi.org/10.1109/CVPR52688.2022.00108)

**Abstract**:

Significant progress has been achieved in automating the design of various components in deep networks. However, the automatic design of loss functions for generic tasks with various evaluation metrics remains under-investigated. Previous works on handcrafting loss functions heavily rely on human expertise, which limits their extensibility. Meanwhile, searching for loss functions is nontrivial due to the vast search space. Existing efforts mainly tackle the issue by employing task-specific heuristics on specific tasks and particular metrics. Such work cannot be extended to other tasks without arduous human effort. In this paper, we propose AutoLoss-Zero, which is a general framework for searching loss functions from scratch for generic tasks. Specifically, we design an elementary search space composed only of primitive mathematical operators to accommodate the heterogeneous tasks and evaluation metrics. A variant of the evolutionary algorithm is employed to discover loss functions in the elementary search space. A loss-rejection protocol and a gradient-equivalence-check strategy are developed so as to improve the search efficiency, which are applicable to generic tasks. Extensive experiments on various computer vision tasks demonstrate that our searched loss functions are on par with or superior to existing loss functions, which generalize well to different datasets and networks. Code shall be released.

----

## [103] Consistency Learning via Decoding Path Augmentation for Transformers in Human Object Interaction Detection

**Authors**: *Jihwan Park, Seungjun Lee, Hwan Heo, Hyeong Kyu Choi, Hyunwoo J. Kim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00109](https://doi.org/10.1109/CVPR52688.2022.00109)

**Abstract**:

Human-Object Interaction detection is a holistic visual recognition task that entails object detection as well as interaction classification. Previous works of HOI detection has been addressed by the various compositions of subset predictions, e.g., Image → HO → I, Image → HI → 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O$</tex>
. Recently, transformer based architecture for HOI has emerged, which directly predicts the HOI triplets in an end-to-end fashion (Image → HOI). Motivated by various inference paths for HOI detection, we propose cross-path consistency learning (CPC), which is a novel end-to-end learning strategy to improve HOI detection for transformers by leveraging augmented decoding paths. CPC learning enforces all the possible predictions from permuted inference sequences to be consistent. This simple scheme makes the model learn consistent representations, thereby improving generalization without increasing model capacity. Our experiments demonstrate the effectiveness of our method, and we achieved significant improvement on V-COCO and HICO-DET compared to the baseline models. Our code is available at https://github.com/mlvlab/CPChoi.

----

## [104] A Proposal-based Paradigm for Self-supervised Sound Source Localization in Videos

**Authors**: *Hanyu Xuan, Zhiliang Wu, Jian Yang, Yan Yan, Xavier Alameda-Pineda*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00110](https://doi.org/10.1109/CVPR52688.2022.00110)

**Abstract**:

Humans can easily recognize where and how the sound is produced via watching a scene and listening to corresponding audio cues. To achieve such cross-modal perception on machines, existing methods only use the maps generated by interpolation operations to localize the sound source. As semantic object-level localization is more attractive for potential practical applications, we argue that these existing map-based approaches only provide a coarse-grained and indirect description of the sound source. In this pa-per, we advocate a novel proposal-based paradigm that can directly perform semantic object-level localization, without any manual annotations. We incorporate the global re-sponse map as an unsupervised spatial constraint to weight the proposals according to how well they cover the esti-mated global shape of the sound source. As a result, our proposal-based sound source localization can be cast into a simpler Multiple Instance Learning (MIL) problem by filtering those instances corresponding to large sound-unrelated regions. Our method achieves state-of-the-art (SOTA) per-formance when compared to several baselines on multiple datasets.

----

## [105] SimAN: Exploring Self-Supervised Representation Learning of Scene Text via Similarity-Aware Normalization

**Authors**: *Canjie Luo, Lianwen Jin, Jingdong Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00111](https://doi.org/10.1109/CVPR52688.2022.00111)

**Abstract**:

Recently self-supervised representation learning has drawn considerable attention from the scene text recognition community. Different from previous studies using contrastive learning, we tackle the issue from an alternative perspective, i.e., by formulating the representation learning scheme in a generative manner. Typically, the neighboring image patches among one text line tend to have similar styles, including the strokes, textures, colors, etc. Motivated by this common sense, we augment one image patch and use its neighboring patch as guidance to recover itself. Specifically, we propose a Similarity-Aware Normalization (SimAN) module to identify the different patterns and align the corresponding styles from the guiding patch. In this way, the network gains representation capability for distinguishing complex patterns such as messy strokes and cluttered backgrounds. Experiments show that the proposed SimAN significantly improves the representation quality and achieves promising performance. Moreover, we surprisingly find that our self-supervised generative network has impressive potential for data synthesis, text image editing, and font interpolation, which suggests that the proposed SimAN has a wide range of practical applications.

----

## [106] Towards End-to-End Unified Scene Text Detection and Layout Analysis

**Authors**: *Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, Michalis Raptis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00112](https://doi.org/10.1109/CVPR52688.2022.00112)

**Abstract**:

Scene text detection and document layout analysis have long been treated as two separate tasks in different image domains. In this paper, we bring them together and introduce the task of unified scene text detection and layout analysis. The first hierarchical scene text dataset is introduced to enable this novel research task. We also propose a novel method that is able to simultaneously detect scene text and form text clusters in a unified way. Comprehensive experiments show that our unified model achieves better performance than multiple well-designed baseline methods. Additionally, this model achieves state-of-the-art results on multiple scene text detection datasets without the need of complex post-processing. Dataset and code: https://github.com/google-research-datasets/hiertext.

----

## [107] Clothes-Changing Person Re-identification with RGB Modality Only

**Authors**: *Xinqian Gu, Hong Chang, Bingpeng Ma, Shutao Bai, Shiguang Shan, Xilin Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00113](https://doi.org/10.1109/CVPR52688.2022.00113)

**Abstract**:

The key to address clothes-changing person re-identification (re-id) is to extract clothes-irrelevant features, e.g., face, hairstyle, body shape, and gait. Most current works mainly focus on modeling body shape from multi-modality information (e.g., silhouettes and sketches), but do not make full use of the clothes-irrelevant information in the original RGB images. In this paper, we propose a Clothes-based Adversarial Loss (CAL) to mine clothes-irrelevant features from the original RGB images by penalizing the predictive power of re-id model w.r.t. clothes. Extensive experiments demonstrate that using RGB images only, CAL outperforms all state-of-the-art methods on widely-used clothes-changing person re-id benchmarks. Besides, compared with images, videos contain richer appearance and additional temporal information, which can be used to model proper spatiotemporal patterns to assist clothes-changing re-id. Since there is no publicly available clothes-changing video re-id dataset, we contribute a new dataset named CCVID and show that there exists much room for improvement in modeling spatiotemporal information. The code and new dataset are available at: h t t 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$p$</tex>
 s: //github.com/guxinqian/Simple-CCReID.

----

## [108] MonoJSG: Joint Semantic and Geometric Cost Volume for Monocular 3D Object Detection

**Authors**: *Qing Lian, Peiliang Li, Xiaozhi Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00114](https://doi.org/10.1109/CVPR52688.2022.00114)

**Abstract**:

Due to the inherent ill-posed nature of 2D-3D projection, monocular 3D object detection lacks accurate depth recovery ability. Although the deep neural network (DNN) enables monocular depth-sensing from high-level learned features, the pixel-level cues are usually omitted due to the deep convolution mechanism. To benefit from both the pow-erful feature representation in DNN and pixel-level geomet-ric constraints, we reformulate the monocular object depth estimation as a progressive refinement problem and propose a joint semantic and geometric cost volume to model the depth error. Specifically, we first leverage neural networks to learn the object position, dimension, and dense normal-ized 3D object coordinates. Based on the object depth, the dense coordinates patch together with the corresponding object features is reprojected to the image space to build a cost volume in a joint semantic and geometric error man-ner. The final depth is obtained by feeding the cost volume to a refinement network, where the distribution of semantic and geometric error is regularized by direct depth supervision. Through effectively mitigating depth error by the re-finement framework, we achieve state-of-the-art results on both the KITTI and Waymo datasets.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/lianqingll/MonoJSG

----

## [109] Homography Loss for Monocular 3D Object Detection

**Authors**: *Jiaqi Gu, Bojian Wu, Lubin Fan, Jianqiang Huang, Shen Cao, Zhiyu Xiang, Xian-Sheng Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00115](https://doi.org/10.1109/CVPR52688.2022.00115)

**Abstract**:

Monocular 3D object detection is an essential task in autonomous driving. However, most current methods consider each 3D object in the scene as an independent training sample, while ignoring their inherent geometric relations, thus inevitably resulting in a lack of leveraging spatial constraints. In this paper, we propose a novel method that takes all the objects into consideration and explores their mutual relationships to help better estimate the 3D boxes. More-over, since 2D detection is more reliable currently, we also investigate how to use the detected 2D boxes as guidance to globally constrain the optimization of the corresponding predicted 3D boxes. To this end, a differentiable loss function, termed as Homography Loss, is proposed to achieve the goal, which exploits both 2D and 3D information, aiming at balancing the positional relationships between different objects by global constraints, so as to obtain more ac-curately predicted 3D boxes. Thanks to the concise design, our loss function is universal and can be plugged into any mature monocular 3D detector, while significantly boosting the performance over their baseline. Experiments demon-strate that our method yields the best performance (Nov. 2021) compared with the other state-of-the-arts by a large margin on KITTI3D datasets.

----

## [110] TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers

**Authors**: *Xuyang Bai, Zeyu Hu, Xinge Zhu, Qingqiu Huang, Yilun Chen, Hongbo Fu, Chiew-Lan Tai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00116](https://doi.org/10.1109/CVPR52688.2022.00116)

**Abstract**:

LiDAR and camera are two important sensors for 3D object detection in autonomous driving. Despite the increasing popularity of sensor fusion in this field, the robustness against inferior image conditions, e.g., bad illumination and sensor misalignment, is under-explored. Existing fusion methods are easily affected by such conditions, mainly due to a hard association of LiDAR points and image pixels, established by calibration matrices. We propose TransFusion, a robust solution to LiDAR-camera fusion with a soft-association mechanism to handle inferior image conditions. Specifically, our TransFusion consists of convolutional backbones and a detection head based on a transformer decoder. The first layer of the decoder predicts initial bounding boxes from a LiDAR point cloud using a sparse set of object queries, and its second decoder layer adaptively fuses the object queries with useful image features, leveraging both spatial and contextual relationships. The attention mechanism of the transformer enables our model to adaptively determine where and what information should be taken from the image, leading to a robust and effective fusion strategy. We additionally design an image-guided query initialization strategy to deal with objects that are difficult to detect in point clouds. TransFusion achieves state-of-the-art performance on large-scale datasets. We provide extensive experiments to demonstrate its robustness against degenerated image quality and calibration errors. We also extend the proposed method to the 3D tracking task and achieve the 1st place in the leader-board of nuScenes tracking, showing its effectiveness and generalization capability. [code release]

----

## [111] TWIST: Two-Way Inter-label Self-Training for Semi-supervised 3D Instance Segmentation

**Authors**: *Ruihang Chu, Xiaoqing Ye, Zhengzhe Liu, Xiao Tan, Xiaojuan Qi, Chi-Wing Fu, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00117](https://doi.org/10.1109/CVPR52688.2022.00117)

**Abstract**:

We explore the way to alleviate the label-hungry problem in a semi-supervised setting for 3D instance segmentation. To leverage the unlabeled data to boost model performance, we present a novel Two-Way Inter-label Self-Training framework named TWIST. It exploits inherent correlations between semantic understanding and instance information of a scene. Specifically, we consider two kinds of pseudo labels for semantic- and instance-level supervision. Our key design is to provide object-level information for denoising pseudo labels and make use of their correlation for two-way mutual enhancement, thereby iteratively promoting the pseudo-label qualities. TWIST attains leading performance on both ScanNet and S3DIS, compared to recent 3D pre-training approaches, and can cooperate with them to further enhance performance, e.g., +4.4% AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
 on 1%-label ScanNet data-efficient benchmark. Code is available at https://github.com/dvlab-research/TWIST.

----

## [112] RBGNet: Ray-based Grouping for 3D Object Detection

**Authors**: *Haiyang Wang, Shaoshuai Shi, Ze Yang, Rongyao Fang, Qi Qian, Hongsheng Li, Bernt Schiele, Liwei Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00118](https://doi.org/10.1109/CVPR52688.2022.00118)

**Abstract**:

As a fundamental problem in computer vision, 3D object detection is experiencing rapid growth. To extract the point-wise features from the irregularly and sparsely distributed points, previous methods usually take a feature grouping module to aggregate the point features to an object candidate. However, these methods have not yet leveraged the surface geometry of foreground objects to enhance grouping and 3D box generation. In this paper, we propose the RBGNet framework, a voting-based 3D detector for accurate 3D object detection from point clouds. In order to learn better representations of object shape to enhance cluster features for predicting 3D boxes, we propose a ray-based feature grouping module, which aggregates the point-wise features on object surfaces using a group of determined rays uniformly emitted from cluster centers. Considering the fact that foreground points are more meaningful for box estimation, we design a novel foreground biased sampling strategy in downsample process to sample more points on object surfaces and further boost the detection performance. Our model achieves state-of-the-art 3D detection performance on ScanNet V2 and SUN RGB-D with remarkable performance gains. Code will be available at https://github.com/Haiyang-W/RBGNet.

----

## [113] Voxel Field Fusion for 3D Object Detection

**Authors**: *Yanwei Li, Xiaojuan Qi, Yukang Chen, Liwei Wang, Zeming Li, Jian Sun, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00119](https://doi.org/10.1109/CVPR52688.2022.00119)

**Abstract**:

In this work, we present a conceptually simple yet effective framework for cross-modality 3D object detection, named voxel field fusion. The proposed approach aims to maintain cross-modality consistency by representing and fusing augmented image features as a ray in the voxel field. To this end, the learnable sampler is first designed to sample vital features from the image plane that are projected to the voxel grid in a point-to-ray manner, which maintains the consistency in feature representation with spatial context. In addition, ray-wise fusion is conducted to fuse features with the supplemental context in the constructed voxel field. We further develop mixed augmentor to align feature-variant transformations, which bridges the modality gap in data augmentation. The proposed framework is demonstrated to achieve consistent gains in various bench-marks and outperforms previous fusion-based methods on KITTI and nuScenes datasets. Code is made available at https://github.com/dvlab-research/VFF
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Part of the work was done in MEGVII Research..

----

## [114] Learning to Detect Mobile Objects from LiDAR Scans Without Labels

**Authors**: *Yurong You, Katie Luo, Cheng Perng Phoo, Wei-Lun Chao, Wen Sun, Bharath Hariharan, Mark E. Campbell, Kilian Q. Weinberger*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00120](https://doi.org/10.1109/CVPR52688.2022.00120)

**Abstract**:

Current 3D object detectors for autonomous driving are almost entirely trained on human-annotated data. Although of high quality, the generation of such data is laborious and costly, restricting them to a few specific locations and object types. This paper proposes an alternative approach entirely based on unlabeled data, which can be collected cheaply and in abundance almost everywhere on earth. Our approach leverages several simple common sense heuristics to create an initial set of approximate seed labels. For example, relevant traffic participants are generally not persistent across multiple traversals of the same route, do not fly, and are never under ground. We demonstrate that these seed labels are highly effective to bootstrap a surprisingly accurate detector through repeated self-training without a single human annotated label. Code is available at https://github.com/YurongYou/MODEST.

----

## [115] OccAM's Laser: Occlusion-based Attribution Maps for 3D Object Detectors on LiDAR Data

**Authors**: *David Schinagl, Georg Krispel, Horst Possegger, Peter M. Roth, Horst Bischof*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00121](https://doi.org/10.1109/CVPR52688.2022.00121)

**Abstract**:

While 3D object detection in LiDAR point clouds is well-established in academia and industry, the explainability of these models is a largely unexplored field. In this paper, we propose a method to generate attribution maps for the detected objects in order to better understand the behavior of such models. These maps indicate the importance of each 3D point in predicting the specific objects. Our method works with black-box models: We do not require any prior knowledge of the architecture nor access to the model's internals, like parameters, activations or gradients. Our efficient perturbation-based approach empirically estimates the importance of each point by testing the model with randomly generated subsets of the input point cloud. Our sub-sampling strategy takes into account the special characteristics of LiDAR data, such as the depth-dependent point density. We show a detailed evaluation of the attribution maps and demonstrate that they are interpretable and highly informative. Furthermore, we compare the attribution maps of recent 3D object detection architectures to provide insights into their decision-making processes.

----

## [116] Confidence Propagation Cluster: Unleash Full Potential of Object Detectors

**Authors**: *Yichun Shen, Wanli Jiang, Zhen Xu, Rundong Li, Junghyun Kwon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00122](https://doi.org/10.1109/CVPR52688.2022.00122)

**Abstract**:

It's been a long history that most object detection methods obtain objects by using the non-maximum suppression (NMS) and its improved versions like Soft-NMS to remove redundant bounding boxes. We challenge those NMS-based methods from three aspects: 1) The bounding box with highest confidence value may not be the true positive having the biggest overlap with the ground-truth box. 2) Not only suppression is required for redundant boxes, but also confidence enhancement is needed for those true positives. 3) Sorting candidate boxes by confidence values is not necessary so that full parallelism is achievable. In this paper, inspired by belief propagation (BP), we propose the Confidence Propagation Cluster (CP-Cluster) to replace NMS-based methods, which is fully parallelizable as well as better in accuracy. In CP-Cluster, we borrow the message passing mechanism from BP to penalize redundant boxes and enhance true positives simultaneously in an iterative way until convergence. We verified the effectiveness of CP-Cluster by applying it to various mainstream detectors such as FasterRCNN, SSD, FCOS, YOLOv3, YOLOv5, Centernet etc. Experiments on MS COCO show that our plug and play method, without retraining detectors, is able to steadily improve average mAP of all those state-of-the-art models with a clear margin from 0.3 to 1.9 respectively when compared with NMS-based methods.

----

## [117] TransGeo: Transformer Is All You Need for Cross-view Image Geo-localization

**Authors**: *Sijie Zhu, Mubarak Shah, Chen Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00123](https://doi.org/10.1109/CVPR52688.2022.00123)

**Abstract**:

The dominant CNN-based methods for cross-view image geo-localization rely on polar transform and fail to model global correlation. We propose a pure transformer-based approach (TransGeo) to address these limitations from a different perspective. TransGeo takes full advantage of the strengths of transformer related to global information modeling and explicit position information encoding. We further leverage the flexibility of transformer input and propose an attention-guided non-uniform cropping method, so that uninformative image patches are removed with negligible drop on performance to reduce computation cost. The saved computation can be reallocated to increase resolution only for informative patches, resulting in performance improvement with no additional computation cost. This “attend and zoom-in” strategy is highly similar to human behavior when observing images. Remarkably, TransGeo achieves state-of-the-art results on both urban and rural datasets, with significantly less computation cost than CNN-based methods. It does not rely on polar transform and infers faster than CNN-based methods. Code is available at https://github.com/Jeff-Zilence/TransGeo2022.

----

## [118] A Voxel Graph CNN for Object Classification with Event Cameras

**Authors**: *Yongjian Deng, Hao Chen, Hai Liu, Youfu Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00124](https://doi.org/10.1109/CVPR52688.2022.00124)

**Abstract**:

Event cameras attract researchers' attention due to their low power consumption, high dynamic range, and extremely high temporal resolution. Learning models on event-based object classification have recently achieved massive success by accumulating sparse events into dense frames to apply traditional 2D learning methods. Yet, these approaches necessitate heavy-weight models and are with high computational complexity due to the redundant information introduced by the sparse-to-dense conversion, limiting the potential of event cameras on real-life applications. This study aims to address the core problem of balancing accuracy and model complexity for event-based classification models. To this end, we introduce a novel graph representation for event data to exploit their sparsity better and customize a lightweight voxel graph convolutional neural network (EV-VGCNN) for event-based classification. Specifically, (1) using voxel-wise vertices rather than previous point-wise inputs to explicitly exploit regional 2D semantics of event streams while keeping the sparsity; (2) proposing a multi-scale feature relational layer (MFRL) to extract spatial and motion cues from each vertex discriminatively concerning its distances to neighbors. Comprehensive experiments show that our model can advance state-of-the-art classification accuracy with extremely low model complexity (merely 0.84M parameters).

----

## [119] OSKDet: Orientation-sensitive Keypoint Localization for Rotated Object Detection

**Authors**: *Dongchen Lu, Dongmei Li, Yali Li, Shengjin Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00125](https://doi.org/10.1109/CVPR52688.2022.00125)

**Abstract**:

Rotated object detection is a challenging issue in computer vision field. Inadequate rotated representation and the confusion of parametric regression have been the bottleneck for high performance rotated detection. In this paper, we propose an orientation-sensitive keypoint based rotated detector OSKDet. First, we adopt a set of keypoints to represent the target and predict the keypoint heatmap on ROI to get the rotated box. By proposing the orientation-sensitive heatmap, OSKDet could learn the shape and direction of rotated target implicitly and has stronger modeling capabilities for rotated representation, which improves the localization accuracy and acquires high quality detection results. Second, we explore a new unordered keypoint representation paradigm, which could avoid the confusion of keypoint regression caused by rule based ordering. Further-more, we propose a localization quality uncertainty module to better predict the classification score by the distribution uncertainty of keypoints heatmap. Experimental results on several public benchmarks show the state-of-the-art performance of OSKDet. Specifically, we achieve an AP of 80.91% on DOTA, 89.98% on HRSC2016, 97.27% on UCAS-AOD, and a F-measure of 92.18% on ICDAR2015, 81.43% on ICDAR2017, respectively.

----

## [120] Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes

**Authors**: *Yang You, Zelin Ye, Yujing Lou, Chengkun Li, Yong-Lu Li, Lizhuang Ma, Weiming Wang, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00126](https://doi.org/10.1109/CVPR52688.2022.00126)

**Abstract**:

3D object detection has attracted much attention thanks to the advances in sensors and deep learning methods for point clouds. Current state-of-the-art methods like VoteNet regress direct offset towards object centers and box orientations with an additional Multi-Layer-Perceptron network. Both their offset and orientation predictions are not accurate due to the fundamental difficulty in rotation classification. In the work, we disentangle the direct offset into Local Canonical Coordinates (LCC), box scales and box orientations. Only LCC and box scales are regressed, while box orientations are generated by a canonical voting scheme. Finally, an LCC-aware back-projection checking algorithm iteratively cuts out bounding boxes from the generated vote maps, with the elimination of false positives. Our model achieves state-of-the-art performance on three standard real-world benchmarks: ScanNet, SceneNN and SUN RGB-D. Our code is available on https://github.com/qq456cvb/CanonicalVoting.

----

## [121] Category Contrast for Unsupervised Domain Adaptation in Visual Tasks

**Authors**: *Jiaxing Huang, Dayan Guan, Aoran Xiao, Shijian Lu, Ling Shao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00127](https://doi.org/10.1109/CVPR52688.2022.00127)

**Abstract**:

Instance contrast for unsupervised representation learning has achieved great success in recent years. In this work, we explore the idea of instance contrastive learning in unsupervised domain adaptation (UDA) and propose a novel Category Contrast technique (CaCo) that introduces semantic priors on top of instance discrimination for visual UDA tasks. By considering instance contrastive learning as a dictionary look-up operation, we construct a semantics-aware dictionary with samples from both source and target domains where each target sample is assigned a (pseudo) category label based on the category priors of source samples. This allows category contrastive learning (between target queries and the category-level dictionary) for category-discriminative yet domain-invariant feature representations: samples of the same category (from either source or target domain) are pulled closer while those of different categories are pushed apart simultaneously. Extensive UDA experiments in multiple visual tasks (e.g., segmentation, classification and detection) show that CaCo achieves superior performance as compared with state-of-the-art methods. The experiments also demonstrate that CaCo is complementary to existing UDA methods and gen-eralizable to other learning setups such as unsupervised model adaptation, open-/partial-set adaptation etc.

----

## [122] Scaling Vision Transformers

**Authors**: *Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, Lucas Beyer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01179](https://doi.org/10.1109/CVPR52688.2022.01179)

**Abstract**:

Attention-based neural networks such as the Vision Transformer (ViT) have recently attained state-of-the-art results on many computer vision benchmarks. Scale is a primary ingredient in attaining excellent results, therefore, understanding a model's scaling properties is a key to designing future generations effectively. While the laws for scaling Transformer language models have been studied, it is unknown how Vision Transformers scale. To address this, we scale ViT models and data, both up and down, and characterize the relationships between error rate, data, and compute. Along the way, we refine the architecture and training of ViT, reducing memory consumption and increasing accuracy of the resulting models. As a result, we successfully train a ViT model with two billion parameters, which attains a new state-of-the-art on ImageNet of 90.45% top-1 accuracy. The model also performs well for few-shot transfer, for example, reaching 84.86% top-1 accuracy on ImageNet with only 10 examples per class.

----

## [123] Amodal Segmentation through Out-of-Task and Out-of-Distribution Generalization with a Bayesian Model

**Authors**: *Yihong Sun, Adam Kortylewski, Alan L. Yuille*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00128](https://doi.org/10.1109/CVPR52688.2022.00128)

**Abstract**:

Amodal completion is a visual task that humans perform easily but which is difficult for computer vision algorithms. The aim is to segment those object boundaries which are occluded and hence invisible. This task is particularly challenging for deep neural networks because data is difficult to obtain and annotate. Therefore, we formulate amodal segmentation as an out-of-task and out-of-distribution generalization problem. Specifically, we replace the fully connected classifier in neural networks with a Bayesian generative model of the neural network features. The model is trained from non-occluded images using bounding box annotations and class labels only, but is applied to generalize out-of-task to object segmentation and to generalize out-of-distribution to segment occluded objects. We demonstrate how such Bayesian models can naturally generalize beyond the training task labels when they learn a prior that models the object's background context and shape. Moreover, by leveraging an outlier process, Bayesian models can further generalize out-of-distribution to segment partially occluded objects and to predict their amodal object boundaries. Our algorithm outperforms alternative methods that use the same supervision by a large margin, and even outperforms methods where annotated amodal segmentations are used during training, when the amount of occlusion is large. Code is publicly available at https://github.com/YihongSun/Bayesian-Amodal.

----

## [124] GANSeg: Learning to Segment by Unsupervised Hierarchical Image Generation

**Authors**: *Xingzhe He, Bastian Wandt, Helge Rhodin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00129](https://doi.org/10.1109/CVPR52688.2022.00129)

**Abstract**:

Segmenting an image into its parts is a common pre-process for high-level vision tasks such as image editing. However, annotating masks for supervised training is expensive. Weakly-supervised and unsupervised methods exist, but they depend on the comparison of pairs of images, such as from multi-views, frames of videos, and image augmentation, which limit their applicability. To address this, we propose a GAN-based approach that generates images conditioned on latent masks, thereby alleviating full or weak annotations required by previous approaches. We show that such mask-conditioned image generation can be learned faithfully when conditioning the masks in a hierarchical manner on 2D latent points that define the position of parts explicitly. Without requiring supervision of masks or points, this strategy increases robustness of mask to viewpoint and object position changes. It also lets us generate image-mask pairs for training a segmentation network, which outperforms state-of-the-art unsupervised segmentation methods on established benchmarks. Code can be found at https://github.com/xingzhehe/GANSeg.

----

## [125] Segment-Fusion: Hierarchical Context Fusion for Robust 3D Semantic Segmentation

**Authors**: *Anirud Thyagharajan, Benjamin Ummenhofer, Prashant Laddha, Om Ji Omer, Sreenivas Subramoney*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00130](https://doi.org/10.1109/CVPR52688.2022.00130)

**Abstract**:

3D semantic segmentation is a fundamental building block for several scene understanding applications such as autonomous driving, robotics and AR/VR. Several state-of-the-art semantic segmentation models suffer from the part-misclassification problem, wherein parts of the same object are labelled incorrectly. Previous methods have utilized hierarchical, iterative methods to fuse semantic and instance information, but they lack learnability in context fusion, and are computationally complex and heuristic driven. This paper presents Segment-Fusion, a novel attention-based method for hierarchical fusion of semantic and instance in-formation to address the part misclassifications. The presented method includes a graph segmentation algorithmfor grouping points into segments that pools point-wise features into segment-wise features, a learnable attention-based net-work to fuse these segments based on their semantic and instance features, and followed by a simple yet effective connected component labelling algorithm to convert seg-ment features to instance labels. Segment-Fusion can be flexibly employed with any network architecture for semantic/instance segmentation. It improves the qualitative and quantitative performance of several semantic segmentation backbones by upto 5% on the ScanNet and S3DIS datasets.

----

## [126] Deep Hierarchical Semantic Segmentation

**Authors**: *Liulei Li, Tianfei Zhou, Wenguan Wang, Jianwu Li, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00131](https://doi.org/10.1109/CVPR52688.2022.00131)

**Abstract**:

Humans are able to recognize structured relations in observation, allowing us to decompose complex scenes into simpler parts and abstract the visual world in multiple levels. However, such hierarchical reasoning ability of human perception remains largely unexplored in current literature of semantic segmentation. Existing work is often aware of flatten labels and predicts target classes exclusively for each pixel. In this paper, we instead address hierarchical semantic segmentation (HSS), which aims at structured, pixel-wise description of visual observation in terms of a class hierarchy. We devise H
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">SSN</inf>
, a general HSS framework that tackles two critical issues in this task: i) how to efficiently adapt existing hierarchy-agnostic segmentation networks to the HSS setting, and ii) how to leverage the hierarchy information to regularize HSS network learning. To address i), HSSN directly casts HSS as a pixel-wise multi-label classification task, only bringing minimal architecture change to current segmentation models. To solve ii), HSSN first explores inherent properties of the hierarchy as a training objective, which enforces segmentation predictions to obey the hierarchy structure. Further, with hierarchy-induced margin constraints, HSSNreshapes the pixel embedding space, so as to generate well-structured pixel representations and improve segmentation eventually. We conduct experiments on four semantic segmentation datasets (i.e., Mapillary Vistas 2.0, City-scapes, LIP, and PASCAL-Person-Part), with different class hierarchies, segmentation network architectures and backbones, showing the generalization and superiority of HSSN.

----

## [127] Semantic Segmentation by Early Region Proxy

**Authors**: *Yifan Zhang, Bo Pang, Cewu Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00132](https://doi.org/10.1109/CVPR52688.2022.00132)

**Abstract**:

Typical vision backbones manipulate structured features. As a compromise, semantic segmentation has long been modeled as per-point prediction on dense regular grids. In this work, we present a novel and efficient modeling that starts from interpreting the image as a tessellation of learnable regions, each of which has flexible geometries and carries homogeneous semantics. To model region-wise context, we exploit Transformer to encode regions in a sequence-to-sequence manner by applying multi-layer self-attention on the region embeddings, which serve as proxies of specific regions. Semantic segmentation is now carried out as per-region prediction on top of the encoded region embeddings using a single linear classifier, where a decoder is no longer needed. The proposed RegProxy model discards the common Cartesian feature layout and operates purely at region level. Hence, it exhibits the most competitive performance-efficiency trade-off compared with the conventional dense prediction methods. For example, on ADE20K, the small-sized RegProxy-S/16 out-performs the best CNN model using 25% parameters and 4% computation, while the largest RegProxy-U16 achieves 52.9mIoU which outperforms the state-of-the-art by 2.1% with fewer resources. Codes and models are available at https://github.com/YiF-Zhang/RegionProxy

----

## [128] Panoptic, Instance and Semantic Relations: A Relational Context Encoder to Enhance Panoptic Segmentation

**Authors**: *Shubhankar Borse, Hyojin Park, Hong Cai, Debasmit Das, Risheek Garrepalli, Fatih Porikli*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00133](https://doi.org/10.1109/CVPR52688.2022.00133)

**Abstract**:

This paper presents a novel framework to integrate both semantic and instance contexts for panoptic segmentation. In existing works, it is common to use a shared backbone to extract features for both things (countable classes such as vehicles) and stuff (uncountable classes such as roads). This, however, fails to capture the rich relations among them, which can be utilized to enhance visual understanding and segmentation performance. To address this short-coming, we propose a novel Panoptic, Instance, and Semantic Relations (PISR) module to exploit such contexts. First, we generate panoptic encodings to summarize key features of the semantic classes and predicted instances. A Panoptic Relational Attention (PRA) module is then applied to the encodings and the global feature map from the backbone. It produces a feature map that captures 1) the relations across semantic classes and instances and 2) the relations between these panoptic categories and spatial features. PISR also automatically learns to focus on the more important instances, making it robust to the number of instances used in the relational attention module. Moreover, PISR is a general module that can be applied to any existing panoptic segmentation architecture. Through extensive evaluations on panoptic segmentation benchmarks like Cityscapes, COCO, and ADE20K, we show that PISR attains considerable improvements over existing approaches.

----

## [129] Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers

**Authors**: *Zhiqi Li, Wenhai Wang, Enze Xie, Zhiding Yu, Anima Anandkumar, José M. Álvarez, Ping Luo, Tong Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00134](https://doi.org/10.1109/CVPR52688.2022.00134)

**Abstract**:

Panoptic segmentation involves a combination of joint semantic segmentation and instance segmentation, where image contents are divided into two types: things and stuff. We present Panoptic SegFormer, a general framework for panoptic segmentation with transformers. It contains three innovative components: an efficient deeply-supervised mask decoder, a query decoupling strategy, and an improved postprocessing method. We also use Deformable DETR to efficiently process multiscale features, which is a fast and efficient version of DETR. Specifically, we supervise the attention modules in the mask decoder in a layer-wise manner. This deep supervision strategy lets the attention modules quickly focus on meaningful semantic regions. It improves performance and reduces the number of required training epochs by half compared to Deformable DETR. Our query decoupling strategy decouples the responsibilities of the query set and avoids mutual interference between things and stuff. In addition, our post-processing strategy improves performance without additional costs by jointly considering classification and segmentation qualities to resolve conflicting mask overlaps. Our approach increases the accuracy 6.2% PQ over the baseline DETR model. Panoptic SegFormer achieves state-of-the-art results on COCO testdev with 56.2% PQ. It also shows stronger zero-shot robustness over existing methods.

----

## [130] Masked-attention Mask Transformer for Universal Image Segmentation

**Authors**: *Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, Rohit Girdhar*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00135](https://doi.org/10.1109/CVPR52688.2022.00135)

**Abstract**:

Image segmentation groups pixels with different semantics, e.g., category or instance membership. Each choice of semantics defines a task. While only the semantics of each task differ, current research focuses on designing spe-cialized architectures for each task. We present Masked- attention Mask Transformer (Mask2Former), a new archi-tecture capable of addressing any image segmentation task (panoptic, instance or semantic). Its key components in-clude masked attention, which extracts localized features by constraining cross-attention within predicted mask regions. In addition to reducing the research effort by at least three times, it outperforms the best specialized architectures by a significant margin on four popular datasets. Most no-tably, Mask2Former sets a new state-of-the-art for panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO) and semantic segmentation (57.7 mIoU onADE20K).

----

## [131] FocalClick: Towards Practical Interactive Image Segmentation

**Authors**: *Xi Chen, Zhiyan Zhao, Yilei Zhang, Manni Duan, Donglian Qi, Hengshuang Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00136](https://doi.org/10.1109/CVPR52688.2022.00136)

**Abstract**:

Interactive segmentation allows users to extract target masks by making positive/negative clicks. Although explored by many previous works, there is still a gap between academic approaches and industrial needs: first, existing models are not efficient enough to work on low-power devices; second, they perform poorly when used to refine preexisting masks as they could not avoid destroying the correct part. FocalClick solves both issues at once by predicting and updating the mask in localized areas. For higher efficiency, we decompose the slow prediction on the entire image into two fast inferences on small crops: a coarse segmentation on the Target Crop, and a local refinement on the Focus Crop. To make the model work with preexisting masks, we formulate a sub-task termed Inter-active Mask Correction, and propose Progressive Merge as the solution. Progressive Merge exploits morphological information to decide where to preserve and where to update, enabling users to refine any preexisting mask effectively. FocalClick achieves competitive results against SOTA methods with significantly smaller FLOPs. It also shows significant superiority when making corrections on preexisting masks. Code and data will be released at github.com/XavierCHEN34/ClickSEG

----

## [132] High Quality Segmentation for Ultra High-resolution Images

**Authors**: *Tiancheng Shen, Yuechen Zhang, Lu Qi, Jason Kuen, Xingyu Xie, Jianlong Wu, Zhe Lin, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00137](https://doi.org/10.1109/CVPR52688.2022.00137)

**Abstract**:

To segment 4K or 6K ultra high-resolution images needs extra computation consideration in image segmentation. Common strategies, such as downsampling, patch cropping, and cascade model, cannot address well the balance issue between accuracy and computation cost. Motivated by the fact that humans distinguish among objects continuously from coarse to precise levels, we propose the Continuous Refinement Model (CRM) for the ultra high-resolution segmentation refinement task. CRM continuously aligns the feature map with the refinement target and aggregates features to reconstruct these image details. Besides, our CRM shows its significant generalization ability to fill the resolution gap between low-resolution training images and ultra high-resolution testing ones. We present quantitative performance evaluation and visualization to show that our proposed method is fast and effective on image segmentation refinement. Code is available at https://github.com/dvlab-research/Entity/tree/main/CRM.

----

## [133] Wnet: Audio-Guided Video Object Segmentation via Wavelet-Based Cross- Modal Denoising Networks

**Authors**: *Wenwen Pan, Haonan Shi, Zhou Zhao, Jieming Zhu, Xiuqiang He, Zhigeng Pan, Lianli Gao, Jun Yu, Fei Wu, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00138](https://doi.org/10.1109/CVPR52688.2022.00138)

**Abstract**:

Audio-Guided video object segmentation is a challenging problem in visual analysis and editing, which automatically separates foreground objects from the background in a video sequence according to the referring audio expressions. However, existing referring video object segmentation works mainly focus on the guidance of text-based referring expressions, due to the lack of modeling the semantic representations of audio-video interaction contents. In this paper, we consider the problem of audio-guided video semantic segmentation from the viewpoint of end-to-end denoising encoder-decoder network learning. We propose the wavelet-based encoder network to learn the cross-modal representations of the video contents with audio-form queries. Specifically, we adopt the multi-head cross-modal attention layers to explore the potential relations of video and query contents. A 2-dimension discrete wavelet trans-form is merged into the transformer encoder to decompose the audio-video features. Next, we maximize mutual information between the encoded features and multi-modal features after cross-modal attention layers to enhance the au-dio guidance. Then, a self attention-free decoder network is developed to generate the target masks with frequency-domain transforms. In addition, we construct the first large-scale audio-guided video semantic segmentation dataset. The extensive experiments show the effectiveness of our method
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.com/asudahkzj/Wnet.git.

----

## [134] Recurrent Dynamic Embedding for Video Object Segmentation

**Authors**: *Mingxing Li, Li Hu, Zhiwei Xiong, Bang Zhang, Pan Pan, Dong Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00139](https://doi.org/10.1109/CVPR52688.2022.00139)

**Abstract**:

Space-time memory (STM) based video object segmentation (VOS) networks usually keep increasing memory bank every several frames, which shows excellent performance. However, 1) the hardware cannot withstand the ever-increasing memory requirements as the video length increases. 2) Storing lots of information inevitably introduces lots of noise, which is not conducive to reading the most important information from the memory bank. In this paper, we propose a Recurrent Dynamic Embedding (RDE) to build a memory bank of constant size. Specifically, we explicitly generate and update RDE by the proposed Spatio-temporal Aggregation Module (SAM), which exploits the cue of historical information. To avoid error accumulation owing to the recurrent usage of SAM, we propose an unbiased guidance loss during the training stage, which makes SAM more robust in long videos. Moreover, the predicted masks in the memory bank are inaccurate due to the inaccurate network inference, which affects the seg-mentation of the query frame. To address this problem, we design a novel self-correction strategy so that the network can repair the embeddings of masks with different qualities in the memory bank. Extensive experiments show our method achieves the best tradeoff between performance and speed. Code is available at https://github.com/Limingxing00/RDE-VOS-CVPR2022.

----

## [135] Accelerating Video Object Segmentation with Compressed Video

**Authors**: *Kai Xu, Angela Yao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00140](https://doi.org/10.1109/CVPR52688.2022.00140)

**Abstract**:

We propose an efficient plug-and-play acceleration framework for semi-supervised video object segmentation by exploiting the temporal redundancies in videos presented by the compressed bitstream. Specifically, we propose a motion vector-based warping method for propagating segmentation masks from keyframes to other frames in a bidirectional and multi-hop manner. Additionally, we introduce a residual-based correction module that can fix wrongly propagated segmentation masks from noisy or erroneous motion vectors. Our approach is flexible and can be added on top of several existing video object segmentation algorithms. We achieved highly competitive results on DAVIS17 and YouTube-VOS on various base models with substantial speed-ups of up to 3.5X with minor drops in accuracy. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/kai422/CoVOS

----

## [136] Per-Clip Video Object Segmentation

**Authors**: *Kwanyong Park, Sanghyun Woo, Seoung Wug Oh, In So Kweon, Joon-Young Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00141](https://doi.org/10.1109/CVPR52688.2022.00141)

**Abstract**:

Recently, memory-based approaches show promising results on semi-supervised video object segmentation. These methods predict object masks frame-by-frame with the help of frequently updated memory of the previous mask. Different from this per-frame inference, we investigate an alternative perspective by treating video object segmentation as clip-wise mask propagation. In this per-clip inference scheme, we update the memory with an interval and simul-taneously process a set of consecutive frames (i.e. clip) between the memory updates. The scheme provides two potential benefits: accuracy gain by clip-level optimization and efficiency gain by parallel computation of multiple frames. To this end, we propose a new method tailored for the perclip inference. Specifically, we first introduce a clip-wise operation to refine the features based on intra-clip correlation. In addition, we employ a progressive matching mechanism for efficient information-passing within a clip. With the synergy of two modules and a newly proposed perclip based training, our network achieves state-of-the-art performance on Youtube-VOS 2018/2019 val (84.6% and 84.6%) and DAVIS 2016/2017 val (91.9% and 86.1%). Fur-thermore, our model shows a great speed-accuracy trade-off with varying memory update intervals, which leads to huge flexibility.

----

## [137] SWEM: Towards Real-Time Video Object Segmentation with Sequential Weighted Expectation-Maximization

**Authors**: *Zhihui Lin, Tianyu Yang, Maomao Li, Ziyu Wang, Chun Yuan, Wenhao Jiang, Wei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00142](https://doi.org/10.1109/CVPR52688.2022.00142)

**Abstract**:

Matching-based methods, especially those based on space-time memory, are significantly ahead of other solutions in semi-supervised video object segmentation (VOS). However, continuously growing and redundant template features lead to an inefficient inference. To alleviate this, we propose a novel Sequential Weighted Expectation-Maximization (SWEM) network to greatly reduce the redundancy of memory features. Different from the previous methods which only detect feature redundancy between frames, SWEM merges both intra-frame and inter-frame similar features by leveraging the sequential weighted EM algorithm. Further, adaptive weights for frame features endow SWEM with the flexibility to represent hard samples, improving the discrimination of templates. Besides, the proposed method maintains a fixed number of template features in memory, which ensures the stable inference complexity of the VOS system. Extensive experiments on commonly used DAVIS and YouTube-VOS datasets verify the high efficiency (36 FPS) and high performance (84.3% J&F on DAVIS 2017 validation dataset) of SWEM.

----

## [138] Neural Recognition of Dashed Curves with Gestalt Law of Continuity

**Authors**: *Hanyuan Liu, Chengze Li, Xueting Liu, Tien-Tsin Wong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00143](https://doi.org/10.1109/CVPR52688.2022.00143)

**Abstract**:

Dashed curve is a frequently used curve form and is widely used in various drawing and illustration applications. While humans can intuitively recognize dashed curves from disjoint curve segments based on the law of continuity in Gestalt psychology, it is extremely difficult for computers to model the Gestalt law of continuity and recognize the dashed curves since high-level semantic understanding is needed for this task. The various appear-ances and styles of the dashed curves posed on a potentially noisy background further complicate the task. In this paper, we propose an innovative Transformer-based framework to recognize dashed curves based on both high-level features and low-level clues. The framework manages to learn the computational analogy of the Gestalt Law in various do-mains to locate and extract instances of dashed curves in both raster and vector representations. Qualitative and quantitative evaluations demonstrate the efficiency and ro-bustness of our framework over all existing solutions.

----

## [139] CVNet: Contour Vibration Network for Building Extraction

**Authors**: *Ziqiang Xu, Chunyan Xu, Zhen Cui, Xiangwei Zheng, Jian Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00144](https://doi.org/10.1109/CVPR52688.2022.00144)

**Abstract**:

The classic active contour model raises a great promising solution to polygon-based object extraction with the progress of deep learning recently. Inspired by the physical vibration theory, we propose a contour vibration network (CVNet) for automatic building boundary delineation. Different from the previous contour models, the CVNet originally roots in the force and motion principle of contour string. Through the infinitesimal analysis and Newton's second law, we derive the spatial-temporal contour vibration model of object shapes, which is mathematically reduced to second-order differential equation. To concretize the dynamic model, we transform the vibration model into the space of image features, and reparameterize the equation coefficients as the learnable state from feature domain. The contour changes are finally evolved in a progressive mode through the computation of contour vibration equation. Both the polygon contour evolution and the model optimization are modulated to form a close-looping end-to-end network. Comprehensive experiments on three datasets demonstrate the effectiveness and superiority of our CVNet over other baselines and state-of-the-art methods for the polygon-based building extraction. The code is available at https://github.com/xzq-njust/CVNet.

----

## [140] A Keypoint-based Global Association Network for Lane Detection

**Authors**: *Jinsheng Wang, Yinchao Ma, Shaofei Huang, Tianrui Hui, Fei Wang, Chen Qian, Tianzhu Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00145](https://doi.org/10.1109/CVPR52688.2022.00145)

**Abstract**:

Lane detection is a challenging task that requires predicting complex topology shapes of lane lines and distinguishing different types of lanes simultaneously. Earlier works follow a top-down roadmap to regress predefined anchors into various shapes of lane lines, which lacks enough flexibility to fit complex shapes of lanes due to the fixed anchor shapes. Lately, some works propose to formulate lane detection as a keypoint estimation problem to describe the shapes of lane lines more flexibly and gradually group adjacent keypoints belonging to the same lane line in a point-by-point manner, which is inefficient and time-consuming during postprocessing. In this paper, we propose a Global Association Network (GANet) to formulate the lane detection problem from a new perspective, where each keypoint is directly regressed to the starting point of the lane line instead of point-by-point extension. Concretely, the association of keypoints to their belonged lane line is conducted by predicting their offsets to the corresponding starting points of lanes globally without dependence on each other, which could be done in parallel to greatly improve efficiency. In addition, we further propose a Lane-aware Feature Aggregator (LFA), which adaptively captures the local correlations between adjacent keypoints to supplement local information to the global association. Extensive experiments on two popular lane detection benchmarks show that our method outperforms previous methods with F1 score of 79.63% on CULane and 97.71% on Tusimple dataset with high FPS.

----

## [141] EDTER: Edge Detection with Transformer

**Authors**: *Mengyang Pu, Yaping Huang, Yuming Liu, Qingji Guan, Haibin Ling*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00146](https://doi.org/10.1109/CVPR52688.2022.00146)

**Abstract**:

Convolutional neural networks have made significant progresses in edge detection by progressively exploring the context and semantic features. However, local details are gradually suppressed with the enlarging of receptive fields. Recently, vision transformer has shown excellent capability in capturing long-range dependencies. Inspired by this, we propose a novel transformer-based edge detector, Edge Detection TransformER (EDTER), to extract clear and crisp object boundaries and meaningful edges by exploiting the full image context information and detailed local cues simultaneously. EDTER works in two stages. In Stage I, a global transformer encoder is used to capture long-range global context on coarse-grained image patches. Then in Stage II, a local transformer encoder works on fine-grained patches to excavate the short-range local cues. Each transformer encoder is followed by an elaborately designed Bi-directional Multi-Level Aggregation decoder to achieve high-resolution features. Finally, the global context and local cues are combined by a Feature Fusion Module and fed into a decision head for edge prediction. Extensive experiments on BSDS500, NYUDv2, and Multicue demonstrate the superiority of EDTER in comparison with state-of-the-arts. The source code is available at https://github.com/MengyangPu/EDTER.

----

## [142] Fixing Malfunctional Objects With Learned Physical Simulation and Functional Prediction

**Authors**: *Yining Hong, Kaichun Mo, Li Yi, Leonidas J. Guibas, Antonio Torralba, Joshua B. Tenenbaum, Chuang Gan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00147](https://doi.org/10.1109/CVPR52688.2022.00147)

**Abstract**:

This paper studies the problem of fixing malfunctional 3D objects. While previous works focus on building passive perception models to learn the functionality from static 3D objects, we argue that functionality is reckoned with respect to the physical interactions between the object and the user. Given a malfunctional object, humans can perform mental simulations to reason about its functionality and figure out how to fix it. Inspired by this, we propose FixIt, a dataset that contains about 5k poorly-designed 3D physical objects paired with choices to fix them. To mimic humans' mental simulation process, we present FixNet, a novel framework that seamlessly incorporates perception and physical dynamics. Specifically, FixNet consists of a perception module to extract the structured representation from the 3D point cloud, a physical dynamics prediction module to simulate the results of interactions on 3D objects, and a functionality prediction module to evaluate the functionality and choose the correct fix. Experimental results show that our framework outperforms baseline models by a large margin, and can generalize well to objects with similar interaction types. Code and dataset are publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
http://fixing-malfunctional.csail.mit.edu.

----

## [143] Coherent Point Drift Revisited for Non-rigid Shape Matching and Registration

**Authors**: *Aoxiang Fan, Jiayi Ma, Xin Tian, Xiaoguang Mei, Wei Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00148](https://doi.org/10.1109/CVPR52688.2022.00148)

**Abstract**:

In this paper, we explore a new type of extrinsic method to directly align two geometric shapes with point-to-point correspondences in ambient space by recovering a deformation, which allows more continuous and smooth maps to be obtained. Specifically, the classic coherent point drift is revisited and generalizations have been proposed. First, by observing that the deformation model is essentially defined with respect to Euclidean space, we generalize the kernel method to non-Euclidean domains. This generally leads to better results for processing shapes, which are known as two-dimensional manifolds. Second, a generalized probabilistic model is proposed to address the sensibility of coherent point drift method to local optima. Instead of directly optimizing over the objective of coherent point drift, the new model allows to focus on a group of most confident ones, thus improves the robustness of the registration system. Experiments are conducted on multiple public datasets with comparison to state-of-the-art competitors, demonstrating the superiority of our method which is both flexible and efficient to improve the matching accuracy due to our extrinsic alignment objective in ambient space.

----

## [144] CodedVTR: Codebook-based Sparse Voxel Transformer with Geometric Guidance

**Authors**: *Tianchen Zhao, Niansong Zhang, Xuefei Ning, He Wang, Li Yi, Yu Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00149](https://doi.org/10.1109/CVPR52688.2022.00149)

**Abstract**:

Transformers have gained much attention by outperforming convolutional neural networks in many 2D vision tasks. However, they are known to have generalization problems and rely on massive-scale pre-training and sophisticated training techniques. When applying to 3D tasks, the irregular data structure and limited data scale add to the difficulty of transformer's application. We propose CodedVTR (Codebook-based Voxel TRansformer), which improves data efficiency and generalization ability for 3D sparse voxel transformers. On the one hand, we propose the codebook-based attention that projects an attention space into its subspace represented by the combination of “prototypes” in a learnable codebook. It regularizes attention learning and improves generalization. On the other hand, we propose geometry-aware self-attention that utilizes geometric information (geometric pattern, density) to guide attention learning. CodedVTR could be embedded into existing sparse convolution-based methods, and bring consistent performance improvements for indoor and outdoor 3D semantic segmentation tasks.

----

## [145] FLOAT: Factorized Learning of Object Attributes for Improved Multi-object Multi-part Scene Parsing

**Authors**: *Rishubh Singh, Pranav Gupta, Pradeep Shenoy, Ravikiran Sarvadevabhatla*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00150](https://doi.org/10.1109/CVPR52688.2022.00150)

**Abstract**:

Multi-object multi-part scene parsing is a challenging task which requires detecting multiple object classes in a scene and segmenting the semantic parts within each object. In this paper, we propose FLOAT, a factorized label space framework for scalable multi-object multi-part parsing. Our framework involves independent dense prediction of object category and part attributes which increases scalability and reduces task complexity compared to the monolithic label space counterpart. In addition, we propose an inference-time ‘zoom’ refinement technique which significantly improves segmentation quality, especially for smaller objects/parts. Compared to state of the art, FLOAT obtains an absolute improvement of 2.0% for mean IOU (mIOU) and 4.8% for segmentation quality IOU (sqIOU) on the Pascal-Part-58 dataset. For the larger Pascal-Part-108 dataset, the improvements are 2.1% for mIOU and 3.9% for sqIOU. We incorporate previously excluded part attributes and other minor parts of the Pascal-Part dataset to create the most comprehensive and challenging version which we dub Pascal-Part-201. FLOAT obtains improvements of 8.6% for mIOU and 7.5% for sqIOU on the new dataset, demonstrating its parsing effectiveness across a challenging diversity of objects and parts. The code and datasets are available at floatseg.github.io.

----

## [146] Rotationally Equivariant 3D Object Detection

**Authors**: *Hong-Xing Yu, Jiajun Wu, Li Yi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00151](https://doi.org/10.1109/CVPR52688.2022.00151)

**Abstract**:

Rotation equivariance has recently become a strongly desired property in the 3D deep learning community. Yet most existing methods focus on equivariance regarding a global input rotation while ignoring the fact that rotation symmetry has its own spatial support. Specifically, we consider the object detection problem in 3D scenes, where an object bounding box should be equivariant regarding the object pose, independent of the scene motion. This suggests a new desired property we call object-level rotation equivariance. To incorporate object-level rotation equivariance into 3D object detectors, we need a mechanism to extract equivariant features with local object-level spatial support while being able to model cross-object context information. To this end, we propose Equivariant Object detection Network (EON) with a rotation equivariance suspension design to achieve object-level equivariance. EON can be applied to modern point cloud object detectors, such as VoteNet and PointR-CNN, enabling them to exploit object rotation symmetry in scene-scale inputs. Our experiments on both indoor scene and autonomous driving datasets show that significant improvements are obtained by plugging our EON design into existing state-of-the-art 3D object detectors. Project website: https://kovenyu.com/EON/

----

## [147] AUV-Net: Learning Aligned UV Maps for Texture Transfer and Synthesis

**Authors**: *Zhiqin Chen, Kangxue Yin, Sanja Fidler*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00152](https://doi.org/10.1109/CVPR52688.2022.00152)

**Abstract**:

In this paper, we address the problem of texture representation for 3D shapes for the challenging and under-explored tasks of texture transfer and synthesis. Previous works either apply spherical texture maps which may lead to large distortions, or use continuous texture fields that yield smooth outputs lacking details. We argue that the traditional way of representing textures with images and linking them to a 3D mesh via UV mapping is more desirable, since synthesizing 2D images is a well-studied problem. We propose AUV-Net which learns to embed 3D surfaces into a 2D aligned UV space, by mapping the corresponding semantic parts of different 3D shapes to the same location in the UV space. As a result, textures are aligned across objects, and can thus be easily synthesized by generative models of images. Texture alignment is learned in an unsupervised manner by a simple yet effective texture alignment module, taking inspiration from traditional works on linear subspace learning. The learned UV mapping and aligned texture representations enable a variety of applications including texture transfer, texture synthesis, and textured single view 3D reconstruction. We conduct experiments on multiple datasets to demonstrate the effectiveness of our method.

----

## [148] Learning to Estimate Robust 3D Human Mesh from In-the-Wild Crowded Scenes

**Authors**: *Hongsuk Choi, Gyeongsik Moon, JoonKyu Park, Kyoung Mu Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00153](https://doi.org/10.1109/CVPR52688.2022.00153)

**Abstract**:

We consider the problem of recovering a single person's 3D human mesh from in-the-wild crowded scenes. While much progress has been in 3D human mesh estimation, existing methods struggle when test input has crowded scenes. The first reason for the failure is a domain gap between training and testing data. A motion capture dataset, which provides accurate 3D labels for training, lacks crowd data and impedes a network from learning crowded scene-robust image features of a target person. The second reason is a feature processing that spatially averages the feature map of a localized bounding box containing multiple people. Averaging the whole feature map makes a target person's feature indistinguishable from others. We present 3DCrowdNet that firstly explicitly targets in-the-wild crowded scenes and estimates a robust 3D human mesh by addressing the above issues. First, we leverage 2D human pose estimation that does not require a motion capture dataset with 3D labels for training and does not suffer from the domain gap. Second, we propose a joint-based regressor that distinguishes a target person's feature from others. Our joint-based regressor preserves the spatial activation of a target by sampling features from the target's joint locations and regresses human model parameters. As a result, 3DCrowdNet learns target-focused features and effectively excludes the irrelevant features of nearby persons. We conduct experiments on various benchmarks and prove the robustness of 3D CrowdNet to the in-the-wild crowded scenes both quantitatively and qualitatively. Codes are available here 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/hongsukchoi/3DCrowdNet_RELEASE.

----

## [149] Human Mesh Recovery from Multiple Shots

**Authors**: *Georgios Pavlakos, Jitendra Malik, Angjoo Kanazawa*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00154](https://doi.org/10.1109/CVPR52688.2022.00154)

**Abstract**:

Videos from edited media like movies are a useful, yet underexplored source of information, with rich variety of appearance and interactions between humans depicted over a large temporal context. However, the richness of data comes at the expense of fundamental challenges such as abrupt shot changes and close up shots of actors with heavy truncation, which limits the applicability of existing 3D human understanding methods. In this paper, we address these limitations with the insight that while shot changes of the same scene incur a discontinuity between frames, the 3D structure of the scene still changes smoothly. This allows us to handle frames before and after the shot change as multi-view signal that provide strong cues to recover the 3D state of the actors. We propose a multi-shot optimization framework that realizes this insight, leading to improved 3D reconstruction and mining of sequences with pseudo-ground truth 3D human mesh. We treat this data as valuable supervision for models that enable human mesh recovery from movies; both from single image and from video, where we propose a transformer-based temporal encoder that can naturally handle missing observations due to shot changes in the input frames. We demonstrate the importance of our insight and proposed models through extensive experiments. The tools we develop open the door to processing and analyzing in 3D content from a large library of edited media, which could be helpful for many downstream applications. Code, models and data are available at: https://geopavlakos.github.io/multishot/.

----

## [150] HandOccNet: Occlusion-Robust 3D Hand Mesh Estimation Network

**Authors**: *JoonKyu Park, Yeonguk Oh, Gyeongsik Moon, Hongsuk Choi, Kyoung Mu Lee*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00155](https://doi.org/10.1109/CVPR52688.2022.00155)

**Abstract**:

Hands are often severely occluded by objects, which makes 3D hand mesh estimation challenging. Previous works often have disregarded information at occluded regions. However, we argue that occluded regions have strong correlations with hands so that they can provide highly beneficial information for complete 3D hand mesh estimation. Thus, in this work, we propose a novel 3D hand mesh estimation network HandOccNet, that can fully exploits the information at occluded regions as a secondary means to enhance image features and make it much richer. To this end, we design two successive Transformer-based modules, called feature injecting transformer (FIT) and self-enhancing transformer (SET). FIT injects hand information into occluded region by considering their correlation. SET refines the output of FIT by using a self-attention mechanism. By injecting the hand information to the occluded region, our HandOccNet reaches the state-of-the-art performance on 3D hand mesh benchmarks that contain challenging hand-object occlusions. The codes are available in: https://github.com/namepllet/HandOccNet.

----

## [151] Photorealistic Monocular 3D Reconstruction of Humans Wearing Clothing

**Authors**: *Thiemo Alldieck, Mihai Zanfir, Cristian Sminchisescu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00156](https://doi.org/10.1109/CVPR52688.2022.00156)

**Abstract**:

We present PHORHUM, a novel, end-to-end trainable, deep neural network methodology for photorealistic 3D human reconstruction given just a monocular RGB image. Our pixel-aligned method estimates detailed 3D geometry and, for the first time, the unshaded surface color together with the scene illumination. Observing that 3D supervision alone is not sufficient for high fidelity color reconstruction, we introduce patch-based rendering losses that enable reliable color reconstruction on visible parts of the human, and detailed and plausible color estimation for the non-visible parts. Moreover, our method specifically addresses methodological and practical limitations of prior work in terms of representing geometry, albedo, and illumination effects, in an end-to-end model where factors can be effectively disentangled. In extensive experiments, we demonstrate the versatility and robustness of our approach. Our state-of-the-art results validate the method qualitatively and for different metrics, for both geometric and color reconstruction.

----

## [152] Disentangled3D: Learning a 3D Generative Model with Disentangled Geometry and Appearance from Monocular Images

**Authors**: *Ayush Tewari, Mallikarjun B. R., Xingang Pan, Ohad Fried, Maneesh Agrawala, Christian Theobalt*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00157](https://doi.org/10.1109/CVPR52688.2022.00157)

**Abstract**:

Learning 3D generative models from a dataset of monocular images enables self-supervised 3D reasoning and controllable synthesis. State-of-the-art 3D generative models are GANs that use neural 3D volumetric representations for synthesis. Images are synthesized by rendering the volumes from a given camera. These models can disentangle the 3D scene from the camera viewpoint in any generated image. However, most models do not disentangle other factors of image formation, such as geometry and appearance. In this paper, we design a 3D GAN which can learn a disentangled model of objects, just from monocular observations. Our model can disentangle the geometry and appearance variations in the scene, i.e., we can independently sample from the geometry and appearance spaces of the generative model. This is achieved using a novel non-rigid deformable scene formulation. A 3D volume that represents an object instance is computed as a non-rigidly deformed canonical 3D volume. Our method learns the canonical volume, as well as its deformations, jointly during training. This formulation also helps us improve the disentanglement between the 3D scene and the camera viewpoints using a novel pose regularization loss defined on the 3D deformation field. In addition, we model the inverse deformations, enabling the computation of dense correspondences between images generated by our model. Finally, we design an approach to embed real images into the latent space of our model, enabling editing of real images.

----

## [153] NeuralHDHair: Automatic High-fidelity Hair Modeling from a Single Image Using Implicit Neural Representations

**Authors**: *Keyu Wu, Yifan Ye, Lingchen Yang, Hongbo Fu, Kun Zhou, Youyi Zheng*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00158](https://doi.org/10.1109/CVPR52688.2022.00158)

**Abstract**:

Undoubtedly, high-fidelity 3D hair plays an indispensable role in digital humans. However, existing monocular hair modeling methods are either tricky to deploy in digital systems (e.g., due to their dependence on complex user interactions or large databases) or can produce only a coarse geometry. In this paper, we introduce NeuralHDHair, a flexible, fully automatic system for modeling high-fidelity hair from a single image. The key enablers of our system are two carefully designed neural networks: an IRHairNet (Im-plicit representation for hair using neural network) for inferring high-fidelity 3D hair geometric features (3D orientation field and 3D occupancy field) hierarchically and a GrowingNet (Growing hair strands using neural network) to efficiently generate 3D hair strands in parallel. Specifically, we perform a coarse-to-fine manner and propose a novel voxel-aligned implicit function (VIFu) to represent the global hair feature, which is further enhanced by the local details extracted from a hair luminance map. To improve the efficiency of a traditional hair growth algorithm, we adopt a local neural implicit function to grow strands based on the estimated 3D hair geometric features. Extensive ex-periments show that our method is capable of constructing a high-fidelity 3D hair model from a single image, both efficiently and effectively, and achieves the-state-of-the-art performance.

----

## [154] Topologically-Aware Deformation Fields for Single-View 3D Reconstruction

**Authors**: *Shivam Duggal, Deepak Pathak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00159](https://doi.org/10.1109/CVPR52688.2022.00159)

**Abstract**:

We present a new framework to learn dense 3D re-construction and correspondence from a single 2D image. The shape is represented implicitly as deformation over a category-level occupancy field and learned in an unsupervised manner from an unaligned image collection without using any 3D supervision. However, image collections usually contain large intra-category topological variation, e.g. images of different chair instances, posing a major challenge. Hence, prior methods are either restricted only to categories with no topological variation for estimating shape and correspondence or focus only on learning shape independently for each instance without any correspondence. To address this issue, we propose a topologically-aware deformation field that maps 3D points in object space to a higher-dimensional canonical space. Given a single image, we first implicitly deform a 3D point in the object space to a learned category-specific canonical space using the topologically-aware field and then learn the 3D shape in the canonical space. Both the canonical shape and deformation field are trained end-to-end using differentiable rendering via learned recurrent ray marcher. Our approach, dubbed TARS, achieves state-of-the-art reconstruction fidelity on several datasets: ShapeNet, Pascal3D+, CUB, and Pix3D chairs.

----

## [155] Generating Diverse 3D Reconstructions from a Single Occluded Face Image

**Authors**: *Rahul Dey, Vishnu Naresh Boddeti*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00160](https://doi.org/10.1109/CVPR52688.2022.00160)

**Abstract**:

Occlusions are a common occurrence in unconstrained face images. Single image 3D reconstruction from such face images often suffers from corruption due to the pres-ence of occlusions. Furthermore, while a plurality of 3D reconstructions is plausible in the occluded regions, existing approaches are limited to generating only a single so-lution. To address both of these challenges, we present Diverse3DFace, which is specifically designed to simulta-neously generate a diverse and realistic set of 3D reconstructions from a single occluded face image. It comprises three components; a global+local shape fitting process, a graph neural network-based mesh VAE, and a determinan-tal point process based diversity-promoting iterative opti-mization procedure. Quantitative and qualitative comparisons of 3D reconstruction on occluded faces show that Di-verse3DFace can estimate 3D shapes that are consistent with the visible regions in the target image while exhibiting high, yet realistic, levels of diversity in the occluded regions. On face images occluded by masks, glasses, and other random objects, Diverse3DFace generates a distri-bution of 3D shapes having ~50% higher diversity on the occluded regions compared to the baselines. Moreover, our closest sample to the ground truth has ~40% lower MSE than the singular reconstructions by existing approaches. Code and data available at: https://github.com/human-analysis/diverse3dface

----

## [156] LOLNeRF: Learn from One Look

**Authors**: *Daniel Rebain, Mark J. Matthews, Kwang Moo Yi, Dmitry Lagun, Andrea Tagliasacchi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00161](https://doi.org/10.1109/CVPR52688.2022.00161)

**Abstract**:

We present a method for learning a generative 3D model based on neural radiance fields, trained solely from data with only single views of each object. While generating realistic images is no longer a difficult task, producing the corresponding 3D structure such that they can be rendered from different views is non-trivial. We show that, unlike existing methods, one does not need multi-view data to achieve this goal. Specifically, we show that by reconstructing many images aligned to an approximate canonical pose with a single network conditioned on a shared latent space, you can learn a space of radiance fields that models shape and appearance for a class of objects. We demonstrate this by training models to reconstruct object categories using datasets that contain only one view of each subject without depth or geometry information. Our experiments show that we achieve state-of-the-art results in novel view synthesis and high-quality results for monocular depth prediction. https://lolnerf.github.io.

----

## [157] Learning Local Displacements for Point Cloud Completion

**Authors**: *Yida Wang, David Joseph Tan, Nassir Navab, Federico Tombari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00162](https://doi.org/10.1109/CVPR52688.2022.00162)

**Abstract**:

We propose a novel approach aimed at object and semantic scene completion from a partial scan represented as a 3D point cloud. Our architecture relies on three novel layers that are used successively within an encoder-decoder structure and specifically developed for the task at hand. The first one carries out feature extraction by matching the point features to a set of pre-trained local descriptors. Then, to avoid losing individual descriptors as part of standard operations such as max-pooling, we propose an alternative neighbor-pooling operation that relies on adopting the feature vectors with the highest activations. Finally, upsampling in the decoder modifies our feature extraction in order to increase the output dimension. While this model is already able to achieve competitive results with the state of the art, we further propose a way to increase the versatility of our approach to process point clouds. To this aim, we introduce a second model that assembles our layers within a transformer architecture. We evaluate both architectures on object and indoor scene completion tasks, achieving state-of-the-art performance.

----

## [158] Exploiting Pseudo Labels in a Self-Supervised Learning Framework for Improved Monocular Depth Estimation

**Authors**: *Andra Petrovai, Sergiu Nedevschi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00163](https://doi.org/10.1109/CVPR52688.2022.00163)

**Abstract**:

We present a novel self-distillation based self-supervised monocular depth estimation (SD-SSMDE) learning framework. In the first step, our network is trained in a self-supervised regime on high-resolution images with the photometric loss. The network is further used to generate pseudo depth labels for all the images in the training set. To improve the performance of our estimates, in the second step, we re-train the network with the scale invariant logarithmic loss supervised by pseudo labels. We resolve scale ambiguity and inter-frame scale consistency by introducing an automatically computed scale in our depth labels. To filter out noisy depth values, we devise a filtering scheme based on the 3D consistency between consecutive views. Extensive experiments demonstrate that each proposed component and the self-supervised learning framework improve the quality of the depth estimation over the baseline and achieve state-of-the-art results on the KITTI and Cityscapes datasets.

----

## [159] Dimension Embeddings for Monocular 3D Object Detection

**Authors**: *Yunpeng Zhang, Wenzhao Zheng, Zheng Zhu, Guan Huang, Dalong Du, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00164](https://doi.org/10.1109/CVPR52688.2022.00164)

**Abstract**:

Most existing deep learning-based approaches for monocular 3D object detection directly regress the dimensions of objects and overlook their importance in solving the illposed problem. In this paper, we propose a general method to learn appropriate embeddings for dimension estimation in monocular 3D object detection. Specifically, we consider two intuitive clues in learning the dimension-aware embeddings with deep neural networks. First, we constrain the pair-wise distance on the embedding space to reflect the similarity of corresponding dimensions so that the model can take advantage of inter-object information to learn more discriminative embeddings for dimension estimation. Second, we propose to learn representative shape templates on the dimension-aware embedding space. Through the attention mechanism, each object can interact with the learnable templates and obtain the attentive dimensions as the initial estimation, which is further refined by the combined features from both the object and the attentive templates. Experimental results on the well-established KITTI dataset demonstrate the proposed method of dimension embeddings can bring consistent improvements with negligible computation cost overhead. We achieve new state-of-the-art performance on the KITTI 3D object detection benchmark.

----

## [160] Understanding 3D Object Articulation in Internet Videos

**Authors**: *Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David F. Fouhey*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00165](https://doi.org/10.1109/CVPR52688.2022.00165)

**Abstract**:

We propose to investigate detecting and characterizing the 3D planar articulation of objects from ordinary RGB videos. While seemingly easy for humans, this problem poses many challenges for computers. Our approach is based on a top-down detection system that finds planes that can be articulated. This approach is followed by optimizing for a 3D plane that explains a sequence of detected articulations. We show that this system can be trained on a combination of videos and 3D scan datasets. When tested on a dataset of challenging Internet videos and the Charades dataset, our approach obtains strong performance.

----

## [161] P3Depth: Monocular Depth Estimation with a Piecewise Planarity Prior

**Authors**: *Vaishakh Patil, Christos Sakaridis, Alexander Liniger, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00166](https://doi.org/10.1109/CVPR52688.2022.00166)

**Abstract**:

Monocular depth estimation is vital for scene understanding and downstream tasks. We focus on the supervised setup, in which ground-truth depth is available only at training time. Based on knowledge about the high regularity of real 3D scenes, we propose a method that learns to selectively leverage information from coplanar pixels to improve the predicted depth. In particular, we introduce a piecewise planarity prior which states that for each pixel, there is a seed pixel which shares the same planar 3D surface with the former. Motivated by this prior, we design a network with two heads. The first head outputs pixel-level plane coefficients, while the second one outputs a dense off-set vector field that identifies the positions of seed pixels. The plane coefficients of seed pixels are then used to predict depth at each position. The resulting prediction is adaptively fused with the initial prediction from the first head via a learned confidence to account for potential deviations from precise local planarity. The entire architecture is trained end-to-end thanks to the differentiability of the proposed modules and it learns to predict regular depth maps, with sharp edges at occlusion boundaries. An extensive evaluation of our method shows that we set the new state of the art in supervised monocular depth estimation, surpassing prior methods on NYU Depth-v2 and on the Garg split of KITTI. Our method delivers depth maps that yield plausible 3D reconstructions of the input scenes. Code is available at: https://github.com/SysCV/P3Depth

----

## [162] Neural Face Identification in a 2D Wireframe Projection of a Manifold Object

**Authors**: *Kehan Wang, Jia Zheng, Zihan Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00167](https://doi.org/10.1109/CVPR52688.2022.00167)

**Abstract**:

In computer-aided design (CAD) systems, 2D line drawings are commonly used to illustrate 3D object designs. To reconstruct the 3D models depicted by a single 2D line drawing, an important key is finding the edge loops in the line drawing which correspond to the actual faces of the 3D object. In this paper, we approach the classical problem of face identification from a novel data-driven point of view. We cast it as a sequence generation problem: starting from an arbitrary edge, we adopt a variant of the popular Transformer model to predict the edges associated with the same face in a natural order. This allows us to avoid searching the space of all possible edge loops with various handcrafted rules and heuristics as most existing methods do, deal with challenging cases such as curved surfaces and nested edge loops, and leverage additional cues such as face types. We further discuss how possibly imperfect predictions can be used for 3D object reconstruction. The project page is at https://manycore-research.github.io/faceformer.

----

## [163] PanopticDepth: A Unified Framework for Depth-aware Panoptic Segmentation

**Authors**: *Naiyu Gao, Fei He, Jian Jia, Yanhu Shan, Haoyang Zhang, Xin Zhao, Kaiqi Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00168](https://doi.org/10.1109/CVPR52688.2022.00168)

**Abstract**:

This paper presents a unified framework for depth-aware panoptic segmentation (DPS), which aims to reconstruct 3D scene with instance-level semantics from one single image. Prior works address this problem by simply adding a dense depth regression head to panoptic segmentation (PS) networks, resulting in two independent task branches. This neglects the mutually-beneficial relations between these two tasks, thus failing to exploit handy instance-level semantic cues to boost depth accuracy while also producing sub-optimal depth maps. To overcome these limitations, we propose a unified framework for the DPS task by applying a dynamic convolution technique to both the PS and depth prediction tasks. Specifically, instead of predicting depth for all pixels at a time, we generate instance-specific kernels to predict depth and segmentation masks for each instance. Moreover, leveraging the instance-wise depth estimation scheme, we add additional instance-level depth cues to assist with supervising the depth learning via a new depth loss. Extensive experiments on Cityscapes-DPS and SemKITTI-DPS show the effectiveness and promise of our method. We hope our unified solution to DPS can lead a new paradigm in this area. Code is available at https://github.com/NaiyuGao/PanopticDepth.

----

## [164] Stability-driven Contact Reconstruction From Monocular Color Images

**Authors**: *Zimeng Zhao, Binghui Zuo, Wei Xie, Yangang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00169](https://doi.org/10.1109/CVPR52688.2022.00169)

**Abstract**:

Physical contact provides additional constraints for hand-object state reconstruction as well as a basis for further understanding of interaction affordances. Estimating these severely occluded regions from monocular images presents a considerable challenge. Existing methods optimize the hand-object contact driven by distance threshold or prior from contact-labeled datasets. However, due to the number of subjects and objects involved in these indoor datasets being limited, the learned contact patterns could not be generalized easily. Our key idea is to reconstruct the contact pattern directly from monocular images, and then utilize the physical stability criterion in the simulation to optimize it. This criterion is defined by the resultant forces and contact distribution computed by the physics engine. Compared to existing solutions, our framework can be adapted to more personalized hands and diverse object shapes. Furthermore, an interaction dataset with extra physical attributes is created to verify the sim-to-real consistency of our methods. Through comprehensive evaluations, hand-object contact can be reconstructed with both accuracy and stability by the proposed framework.

----

## [165] LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network

**Authors**: *Zhigang Jiang, Zhongzheng Xiang, Jinhua Xu, Ming Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00170](https://doi.org/10.1109/CVPR52688.2022.00170)

**Abstract**:

3D room layout estimation by a single panorama using deep neural networks has made great progress. However, previous approaches can not obtain efficient geometry awareness of room layout with the only latitude of boundaries or horizon-depth. We present that using horizondepth along with room height can obtain omnidirectionalgeometry awareness of room layout in both horizontal and vertical directions. In addition, we propose a planar-geometry aware loss function with normals and gradients of normals to supervise the planeness of walls and turning of corners. We propose an efficient network, LGT-Net, for room layout estimation, which contains a novel Transformer architecture called SWG-Transformer to model geometry relations. SWG-Transformer consists of (Shifted) Window Blocks and Global Blocks to combine the local and global geometry relations. Moreover, we design a novel relative position embedding of Transformer to enhance the spatial identification ability for the panorama. Experiments show that the proposed LGT-Net achieves better performance than current state-of-the-arts (SOTA) on benchmark datasets. The code is publicly available at https://github.com/zhigangjiang/LGT-Net.

----

## [166] Collaborative Learning for Hand and Object Reconstruction with Attention-guided Graph Convolution

**Authors**: *Tze Ho Elden Tse, Kwang In Kim, Ales Leonardis, Hyung Jin Chang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00171](https://doi.org/10.1109/CVPR52688.2022.00171)

**Abstract**:

Estimating the pose and shape of hands and objects under interaction finds numerous applications including aug-mented and virtual reality. Existing approaches for hand and object reconstruction require explicitly defined physical constraints and known objects, which limits its application domains. Our algorithm is agnostic to object models, and it learns the physical rules governing hand-object interaction. This requires automatically inferring the shapes and physi-cal interaction of hands and (potentially unknown) objects. We seek to approach this challenging problem by proposing a collaborative learning strategy where two-branches of deep networks are learning from each other. Specifically, we transfer hand mesh information to the object branch and vice versa for the hand branch. The resulting optimi-sation (training) problem can be unstable, and we address this via two strategies: (i) attention-guided graph convo-lution which helps identify and focus on mutual occlusion and (ii) unsupervised associative loss which facilitates the transfer of information between the branches. Experiments using four widely-used benchmarks show that our frame-work achieves beyond state-of-the-art accuracy in 3D pose estimation, as well as recovers dense 3D hand and object shapes. Each technical component above contributes meaningfully in the ablation study.

----

## [167] RM-Depth: Unsupervised Learning of Recurrent Monocular Depth in Dynamic Scenes

**Authors**: *Tak-Wai Hui*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00172](https://doi.org/10.1109/CVPR52688.2022.00172)

**Abstract**:

Unsupervised methods have showed promising results on monocular depth estimation. However, the training data must be captured in scenes without moving objects. To push the envelope of accuracy, recent methods tend to increase their model parameters. In this paper, an unsupervised learning framework is proposed to jointly predict monocular depth and complete 3D motion including the motions of moving objects and camera. (1) Recurrent modulation units are used to adaptively and iteratively fuse encoder and decoder features. This improves the single-image depth inference without overspending model parameters. (2) Instead of using a single set of filters for upsampling, multiple sets of filters are devised for the residual upsampling. This facilitates the learning of edge-preserving filters and leads to the improved performance. (3) A warping-based network is used to estimate a motion field of moving objects without using semantic priors. This breaks down the requirement of scene rigidity and allows to use general videos for the unsupervised learning. The motion field is further regularized by an outlier-aware training loss. Despite the depth model just uses a single image in test time and 2.97M parameters, it achieves state-of-the-art results on the KITTI and Cityscapes benchmarks.

----

## [168] Exploring Geometric Consistency for Monocular 3D Object Detection

**Authors**: *Qing Lian, Botao Ye, Ruijia Xu, Weilong Yao, Tong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00173](https://doi.org/10.1109/CVPR52688.2022.00173)

**Abstract**:

This paper investigates the geometric consistency for monocular 3D object detection, which suffers from the ill-posed depth estimation. We first conduct a thorough analysis to reveal how existing methods fail to consistently localize objects when different geometric shifts occur. In particular, we design a series of geometric manipulations to diagnose existing detectors and then illustrate their vulnerability to consistently associate the depth with object apparent sizes and positions. To alleviate this issue, we propose four geometry-aware data augmentation approaches to enhance the geometric consistency of the detectors. We first modify some commonly used data augmentation methods for 2D images so that they can maintain geometric consistency in 3D spaces. We demonstrate such modifications are important. In addition, we propose a 3D-specific image perturbation method that employs the camera movement. During the augmentation process, the camera system with the corresponding image is manipulated, while the geometric visual cues for depth recovery are preserved. We show that by using the geometric consistency constraints, the proposed augmentation techniques lead to improvements on the KITTI and nuScenes monocular 3D detection benchmarks with state-of-the-art results. In addition, we demonstrate that the augmentation methods are well suited for semisupervised training and cross-dataset generalization.

----

## [169] Learning 3D Object Shape and Layout without 3D Supervision

**Authors**: *Georgia Gkioxari, Nikhila Ravi, Justin Johnson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00174](https://doi.org/10.1109/CVPR52688.2022.00174)

**Abstract**:

A 3D scene consists of a set of objects, each with a shape and a layout giving their position in space. Understanding 3D scenes from 2D images is an important goal, with ap-plications in robotics and graphics. While there have been recent advances in predicting 3D shape and layout from a single image, most approaches rely on 3D ground truth for training which is expensive to collect at scale. We overcome these limitations and propose a method that learns to predict 3D shape and layout for objects without any ground truth shape or layout information: instead we rely on multi-view images with 2D supervision which can more easily be col-lected at scale. Through extensive experiments on ShapeNet, Hypersim, and ScanNet we demonstrate that our approach scales to large datasets of realistic images, and compares favorably to methods relying on 3D ground truth. On Hy-persim and ScanNet where reliable 3D ground truth is not available, our approach outperforms supervised approaches trained on smaller and less diverse datasets.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page https://gkioxari.github.io/usl/

----

## [170] Single-Stage 3D Geometry-Preserving Depth Estimation Model Training on Dataset Mixtures with Uncalibrated Stereo Data

**Authors**: *Nikolay Patakin, Anna Vorontsova, Mikhail Artemyev, Anton Konushin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00175](https://doi.org/10.1109/CVPR52688.2022.00175)

**Abstract**:

Nowadays, robotics, AR, and 3D modeling applications attract considerable attention to single-view depth estimation (SVDE) as it allows estimating scene geometry from a single RGB image. Recent works have demonstrated that the accuracy of an SVDE method hugely depends on the diversity and volume of the training data. However, RGB-D datasets obtained via depth capturing or 3D re-construction are typically small, synthetic datasets are not photorealistic enough, and all these datasets lack diversity. The large-scale and diverse data can be sourced from stereo images or stereo videos from the web. Typically being uncalibrated, stereo data provides disparities up to unknown shift (geometrically incomplete data), so stereo-trained SVDE methods cannot recover 3D geometry. It was recently shown that the distorted point clouds obtained with a stereo-trained SVDE method can be corrected with additional point cloud modules (PCM) separately trained on the geometrically complete data. On the contrary, we propose 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$GP^{2}$</tex>
, General-Purpose and Geometry-Preserving training scheme, and show that conventional SVDE models can learn correct shifts themselves without any post-processing, benefiting from using stereo data even in the geometry-preserving setting. Through experiments on dif-ferent dataset mixtures, we prove that 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$GP^{2}$</tex>
-trained mod-els outperform methods relying on PCM in both accuracy and speed, and report the state-of-the-art results in the general-purpose geometry-preserving SVDE. Moreover, we show that SVDE models can learn to predict geometrically correct depth even when geometrically complete data com-prises the minor part of the training set.

----

## [171] Occluded Human Mesh Recovery

**Authors**: *Rawal Khirodkar, Shashank Tripathi, Kris Kitani*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00176](https://doi.org/10.1109/CVPR52688.2022.00176)

**Abstract**:

Top-down methods for monocular human mesh recovery have two stages: (1) detect human bounding boxes; (2) treat each bounding box as an independent single-human mesh recovery task. Unfortunately, the single-human assumption does not hold in images with multi-human occlusion and crowding. Consequently, top-down methods have difficulties in recovering accurate 3D human meshes under severe person-person occlusion. To address this, we present Occluded Human Mesh Recovery (OCHMR) - a novel top-down mesh recovery approach that incorporates image spatial context to overcome the limitations of the single-human assumption. The approach is conceptually simple and can be applied to any existing top-down architecture. Along with the input image, we condition the top-down model on spatial context from the image in the form of body-center heatmaps. To reason from the predicted body centermaps, we introduce Contextual Normalization (CoNorm) blocks to adaptively modulate intermediate features of the top-down model. The contextual conditioning helps our model disambiguate between two severely overlapping human boundingboxes, making it robust to multi-person occlusion. Compared with state-of-the-art methods, OCHMR achieves superior performance on challenging multi-person benchmarks like 3DPW, CrowdPose and OCHuman. Specifically, our proposed contextual reasoning architecture applied to the SPIN model with ResNet-50 backbone results in 75.2 PMPJPE on 3DPW-PC, 23.6 AP on CrowdPose and 37.7 AP on OCHu- man datasets, a significant improvement of 6.9 mm, 6.4 AP and 20.8 AP respectively over the baseline.

----

## [172] LAKe-Net: Topology-Aware Point Cloud Completion by Localizing Aligned Keypoints

**Authors**: *Junshu Tang, Zhijun Gong, Ran Yi, Yuan Xie, Lizhuang Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00177](https://doi.org/10.1109/CVPR52688.2022.00177)

**Abstract**:

Point cloud completion aims at completing geometric and topological shapes from a partial observation. However, some topology of the original shape is missing, existing methods directly predict the location of complete points, without predicting structured and topological information of the complete shape, which leads to inferior performance. To better tackle the missing topology part, we propose LAKe-Net, a novel topology-aware point cloud completion model by localizing aligned keypoints, with a novel Keypoints-Skeleton-Shape prediction manner. Specifically, our method completes missing topology using three steps: 1) Aligned Keypoint Localization. An asymmetric keypoint locator, including an unsupervised multi-scale keypoint detector and a complete keypoint generator, is proposed for localizing aligned keypoints from complete and partial point clouds. We theoretically prove that the detector can capture aligned keypoints for objects within a sub-category. 2) Surface-skeleton Generation. A new type of skeleton, named Surface-skeleton, is generated from keypoints based on geometric priors to fully represent the topological information captured from keypoints and better recover the local details. 3) Shape Refinement. We design a refinement subnet where multi-scale surface-skeletons are fed into each recursive skeleton-assisted refinement module to assist the completion process. Experimental results show that our method achieves the state-of-the-art performance on point cloud completion.

----

## [173] OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction

**Authors**: *Wenbin Lin, Chengwei Zheng, Jun-Hai Yong, Feng Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00178](https://doi.org/10.1109/CVPR52688.2022.00178)

**Abstract**:

RGBD-based real-time dynamic 3D reconstruction suffers from inaccurate inter-frame motion estimation as errors may accumulate with online tracking. This problem is even more severe for single-view-based systems due to strong occlusions. Based on these observations, we propose OcclusionFusion, a novel method to calculate occlusion-aware 3D motion to guide the reconstruction. In our technique, the motion of visible regions is first estimated and combined with temporal information to infer the motion of the occluded regions through an LSTM-involved graph neural network. Furthermore, our method computes the confidence of the estimated motion by modeling the network output with a probabilistic model, which alleviates untrust-worthy motions and enables robust tracking. Experimental results on public datasets and our own recorded data show that our technique outperforms existing single-view-based real-time methods by a large margin. With the reduction of the motion errors, the proposed technique can handle long and challenging motion sequences. Please check out the project page for sequence results: https://wenbinlin.github.io/OcclusionFusion.

----

## [174] Depth Estimation by Combining Binocular Stereo and Monocular Structured-Light

**Authors**: *Yuhua Xu, Xiaoli Yang, Yushan Yu, Wei Jia, Zhaobi Chu, Yulan Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00179](https://doi.org/10.1109/CVPR52688.2022.00179)

**Abstract**:

It is well known that the passive stereo system cannot adapt well to weak texture objects, e.g., white walls. However, these weak texture targets are very common in indoor environments. In this paper, we present a novel stereo system, which consists of two cameras (an RGB camera and an IR camera) and an IR speckle projector. The RGB camera is used both for depth estimation and texture acquisition. The IR camera and the speckle projector can form a monocular structured-light (MSL) subsystem, while the two cameras can form a binocular stereo subsystem. The depth map generated by the MSL subsystem can provide external guidance for the stereo matching networks, which can improve the matching accuracy significantly. In order to verify the effectiveness of the proposed system, we build a prototype and collect a test dataset in indoor scenes. The evaluation results show that the Bad 2.0 error of the proposed system is 28.2% of the passive stereo system when the network RAFT is used. The dataset and trained models are available at https://github.com/YuhuaXu/MonoStereoFusion.

----

## [175] Learning from Pixel-Level Noisy Label : A New Perspective for Light Field Saliency Detection

**Authors**: *Mingtao Feng, Kendong Liu, Liang Zhang, Hongshan Yu, Yaonan Wang, Ajmal Mian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00180](https://doi.org/10.1109/CVPR52688.2022.00180)

**Abstract**:

Saliency detection with light field images is becoming attractive given the abundant cues available, however, this comes at the expense of large-scale pixel level annotated data which is expensive to generate. In this paper, we propose to learn light field saliency from pixel-level noisy labels obtained from unsupervised hand crafted featured-based saliency methods. Given this goal, a natural question is: can we efficiently incorporate the relationships among light field cues while identifying clean labels in a unified framework? We address this question by formulating the learning as a joint optimization of intra light field features fusion stream and inter scenes correlation stream to generate the predictions. Specially, we first introduce a pixel forgetting guided fusion module to mutually enhance the light field features and exploit pixel consistency across iterations to identify noisy pixels. Next, we introduce a cross scene noise penalty loss for better reflecting latent structures of training data and enabling the learning to be invariant to noise. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our framework showing that it learns saliency prediction comparable to state-of-the-art fully supervised light field saliency methods. Our code is available at h t tps://github.com/ OLobbCode/NoiseLF.

----

## [176] HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening

**Authors**: *Wele Gedara Chaminda Bandara, Vishal M. Patel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00181](https://doi.org/10.1109/CVPR52688.2022.00181)

**Abstract**:

Pansharpening aims to fuse a registered high-resolution panchromatic image (PAN) with a low-resolution hyper-spectral image (LR-HSI) to generate an enhanced HSI with high spectral and spatial resolution. Existing pansharpening approaches neglect using an attention mechanism to transfer HR texture features from PAN to LR-HSI features, resulting in spatial and spectral distortions. In this paper, we present a novel attention mechanism for pansharpening called HyperTransformer, in which features of LR-HSI and PAN are formulated as queries and keys in a transformer, respectively. HyperTransformer consists of three main modules, namely two separate feature extractors for PAN and HSI, a multi-head feature soft-attention module, and a spatial-spectral feature fusion module. Such a network improves both spatial and spectral quality measures of the pansharpened HSI by learning cross-feature space dependencies and long-range details of PAN and LR-HSI. Furthermore, HyperTransformer can be utilized across multiple spatial scales at the backbone for obtaining improved performance. Extensive experiments conducted on three widely used datasets demonstrate that HyperTransformer achieves significant improvement over the state-of-the-art methods on both spatial and spectral quality measures. Implementation code and pretrained weights can be accessed at https://github.com/wgcban/HyperTransformer.

----

## [177] Revisiting Near/Remote Sensing with Geospatial Attention

**Authors**: *Scott Workman, Muhammad Usman Rafique, Hunter Blanton, Nathan Jacobs*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00182](https://doi.org/10.1109/CVPR52688.2022.00182)

**Abstract**:

This work addresses the task of overhead image segmentation when auxiliary ground-level images are available. Recent work has shown that performing joint inference over these two modalities, often called near/remote sensing, can yield significant accuracy improvements. Extending this line of work, we introduce the concept of geospatial attention, a geometry-aware attention mechanism that explicitly considers the geospatial relationship between the pixels in a ground-level image and a geographic location. We propose an approach for computing geospatial attention that incorporates geometric features and the appearance of the overhead and ground-level imagery. We introduce a novel architecture for near/remote sensing that is based on geospatial attention and demonstrate its use for five segmentation tasks. The results demonstrate that our method significantly outperforms the previous state-of-the-art methods.

----

## [178] Memory-augmented Deep Conditional Unfolding Network for Pansharpening

**Authors**: *Gang Yang, Man Zhou, Keyu Yan, Aiping Liu, Xueyang Fu, Fan Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00183](https://doi.org/10.1109/CVPR52688.2022.00183)

**Abstract**:

Pansharpening aims to obtain high-resolution multispectral (MS) images for remote sensing systems and deep learning-based methods have achieved remarkable success. However, most existing methods are designed in a black-box principle, lacking sufficient interpretability. Additionally, they ignore the different characteristics of each band of MS images and directly concatenate them with panchromatic (PAN) images, leading to severe copy artifacts [9]. To address the above issues, we propose an interpretable deep neural network, namely Memory-augmented Deep Conditional Unfolding Network with two specified core designs. Firstly, considering the degradation process, it formulates the Pansharpening problem as the minimization of a variational model with denoising-based prior and non-local auto-regression prior which is capable of searching the similarities between long-range patches, benefiting the texture enhancement. A novel iteration algorithm with built-in CNNs is exploited for transparent model design. Secondly, to fully explore the potentials of different bands of MS images, the PAN image is combined with each band of MS images, selectively providing the high-frequency details and alleviating the copy artifacts. Extensive experimental results validate the superiority of the proposed algorithm against other state-of-the-art methods.

----

## [179] Mutual Information-driven Pan-sharpening

**Authors**: *Man Zhou, Keyu Yan, Jie Huang, Zihe Yang, Xueyang Fu, Feng Zhao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00184](https://doi.org/10.1109/CVPR52688.2022.00184)

**Abstract**:

Pan-sharpening aims to integrate the complementary information of texture-rich PAN images and multi-spectral (MS) images to produce the texture-rich MS images. Despite the remarkable progress, existing state-of-the-art Pansharpening methods don't explicitly enforce the complementary information learning between two modalities of PAN and MS images. This leads to information redundancy not being handled well, which further limits the performance of these methods. To address the above issue, we propose a novel mutual information-driven Pan-sharpening framework in this paper. To be specific, we first project the PAN and MS image into modality-aware feature space independently, and then impose the mutual information minimization over them to explicitly encourage the complementary information learning. Such operation is capable of reducing the information redundancy and improving the model performance. Extensive experimental results over multiple satellite datasets demonstrate that the proposed algorithm outperforms other state-of-the-art methods qualitatively and quantitatively with great generalization ability to real-world scenes.

----

## [180] Sparse and Complete Latent Organization for Geospatial Semantic Segmentation

**Authors**: *Fengyu Yang, Chenyang Ma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00185](https://doi.org/10.1109/CVPR52688.2022.00185)

**Abstract**:

Geospatial semantic segmentation on remote sensing images suffers from large intra-class variance in both fore-ground and background classes. First, foreground objects are tiny in the remote sensing images and are represented by only a few pixels, which leads to large foreground intra-class variance and undermines the discrimination between foreground classes (issue firstly considered in this work). Second, background class contains complex context, which results in false alarms due to large background intra-class variance. To alleviate these two issues, we construct a sparse and complete latent structure via prototypes. In particular, to enhance the sparsity of the latent space, we design a prototypical contrastive learning to have prototypes of the same category clustering together and prototypes of different categories to be far away from each other. Also, we strengthen the completeness of the latent space by modeling all foreground categories and hardest (nearest) background objects. We further design a patch shuffle augmentation for remote sensing images with complicated contexts. Our augmentation encourages the semantic information of an object to be correlated only to the limited context within the patch that is specific to its category, which further reduces large intra-class variance. We conduct extensive evaluations on a large scale remote sensing dataset, showing our approach significantly outperforms state-of-the-art methods by a large margin.

----

## [181] The Probabilistic Normal Epipolar Constraint for Frame- To-Frame Rotation Optimization under Uncertain Feature Positions

**Authors**: *Dominik Muhle, Lukas Koestler, Nikolaus Demmel, Florian Bernard, Daniel Cremers*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00186](https://doi.org/10.1109/CVPR52688.2022.00186)

**Abstract**:

The estimation of the relative pose of two camera views is a fundamental problem in computer vision. Kneip et al. proposed to solve this problem by introducing the normal epipolar constraint (NEC). However, their approach does not take into account uncertainties, so that the accuracy of the estimated relative pose is highly dependent on accurate feature positions in the target frame. In this work, we introduce the probabilistic normal epipolar constraint (PNEC) that overcomes this limitation by accounting for anisotropic and inhomogeneous uncertainties in the feature positions. To this end, we propose a novel objective function, along with an efficient optimization scheme that effectively minimizes our objective while maintaining real-time performance. In experiments on synthetic data, we demonstrate that the novel PNEC yields more accurate rotation estimates than the original NEC and several popular relative rotation estimation algorithms. Furthermore, we integrate the proposed method into a state-of-the-art monocular rotation-only odometry system and achieve consistently improved results for the real-world KITTI dataset.

----

## [182] Oriented RepPoints for Aerial Object Detection

**Authors**: *Wentong Li, Yijie Chen, Kaixuan Hu, Jianke Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00187](https://doi.org/10.1109/CVPR52688.2022.00187)

**Abstract**:

In contrast to the generic object, aerial targets are often non-axis aligned with arbitrary orientations having the cluttered surroundings. Unlike the mainstreamed approaches regressing the bounding box orientations, this paper proposes an effective adaptive points learning approach to aerial object detection by taking advantage of the adaptive points representation, which is able to capture the geometric information of the arbitrary-oriented instances. To this end, three oriented conversion functions are presented to facilitate the classification and localization with accurate orientation. Moreover, we propose an effective quality assessment and sample assignment scheme for adaptive points learning toward choosing the representative oriented reppoints samples during training, which is able to capture the non-axis aligned features from adjacent objects or background noises. A spatial constraint is introduced to penalize the outlier points for roust adaptive learning. Experimental results on four challenging aerial datasets including DOTA, HRSC2016, UCAS-AOD and DIOR-R, demonstrate the efficacy of our proposed approach. The source code is availabel at: https://github.com/LiWentomng/OrientedRepPoints.

----

## [183] Using 3D Topological Connectivity for Ghost Particle Reduction in Flow Reconstruction

**Authors**: *Christina Tsalicoglou, Thomas Rösgen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00188](https://doi.org/10.1109/CVPR52688.2022.00188)

**Abstract**:

Volumetric flow velocimetry for experimental fluid dynamics relies primarily on the 3D reconstruction of point objects, which are the detected positions of tracer particles identified in images obtained by a multi-camera setup. By assuming that the particles accurately follow the observed flow, their displacement over a known time interval is a measure of the local flow velocity. The number of particles imaged in a 1 Megapixel image is typically in the order of 10
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
-1 0
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
, resulting in a large number of consistent but in-correct reconstructions (no real particle in 3D), that must be eliminated through tracking or intensity constraints. In an alternative method, 3D Particle Streak Velocimetry (3D-PSV), the exposure time is increased, and the particles' pathlines are imaged as “streaks”. We treat these streaks (a) as connected endpoints and (b) as conic section segments and develop a theoretical model that describes the mechanisms of 3D ambiguity generation and shows that streaks can drastically reduce reconstruction ambiguities. Moreover, we propose a method for simultaneously estimating these short, low-curvature conic section segments and their 3D position from multiple camera views. Our results validate the theory, and the streak and conic section reconstruction method produces far fewer ambiguities than simple particle reconstruction, outperforming current state-of-the-art particle tracking software on the evaluated cases.

----

## [184] Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites

**Authors**: *Ngoc Long Nguyen, Jérémy Anger, Axel Davy, Pablo Arias, Gabriele Facciolo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00190](https://doi.org/10.1109/CVPR52688.2022.00190)

**Abstract**:

Modern Earth observation satellites capture multi-exposure bursts of push-frame images that can be super-resolved via computational means. In this work, we propose a super-resolution method for such multi-exposure sequences, a problem that has received very little attention in the literature. The proposed method can handle the signal-dependent noise in the inputs, process sequences of any length, and be robust to inaccuracies in the exposure times. Furthermore, it can be trained end-to-end with self-supervision, without requiring ground truth high resolution frames, which makes it especially suited to handle real data. Central to our method are three key contributions: i) a base-detail decomposition for handling errors in the exposure times, ii) a noise-level-aware feature encoding for improved fusion of frames with varying signal-to-noise ratio and iii) a permutation invariant fusion strategy by temporal pooling operators. We evaluate the proposed method on synthetic and real data and show that it outperforms by a significant margin existing single-exposure approaches that we adapted to the multi-exposure case.

----

## [185] MISF: Multi-level Interactive Siamese Filtering for High-Fidelity Image Inpainting

**Authors**: *Xiaoguang Li, Qing Guo, Di Lin, Ping Li, Wei Feng, Song Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00191](https://doi.org/10.1109/CVPR52688.2022.00191)

**Abstract**:

Although achieving significant progress, existing deep generative inpainting methods still show low generalization across different scenes. As a result, the generated images usually contain artifacts or the filled pixels differ greatly from the ground truth, making them far from real-world applications. Image-level predictive filtering is a widely used restoration technique by predicting suitable kernels adaptively according to different input scenes. Inspired by this inherent advantage, we explore the possibility of addressing image inpainting as a filtering task. To this end, we first study the advantages and challenges of the image-level predictive filtering for inpainting: the method can preserve local structures and avoid artifacts but fails to fill large missing areas. Then, we propose the semantic filtering by conducting filtering on deep feature level, which fills the missing semantic information but fails to recover the details. To address the issues while adopting the respective advantages, we propose a novel filtering technique, i.e., Multi-level Interactive Siamese Filtering (MISF) containing two branches: kernel prediction branch (KPB) and semantic & image filtering branch (SIFB). These two branches are interactively linked: SIFB provides multi-level features for KPB while KPB predicts dynamic kernels for SIFB. As a result, the final method takes the advantage of effective semantic & image-level filling for high-fidelity inpainting. Moreover, we discuss the relationship between MISF and the naive encoder-decoder-based inpainting, inferring that MISF provides novel dynamic convolutional operations to enhance the high generalization capability across scenes. We validate our method on three challenging datasets, i.e., Dunhuang, Places2, and CelebA. Our method outperforms state-of-the-art baselines on four metrics, i.e., 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$L_{1}$</tex>
, PSNR, SSIM, and LPIPS.

----

## [186] Iterative Deep Homography Estimation

**Authors**: *Si-Yuan Cao, Jianxin Hu, Ze-Hua Sheng, Hui-Liang Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00192](https://doi.org/10.1109/CVPR52688.2022.00192)

**Abstract**:

We propose Iterative Homography Network, namely IHN, a new deep homography estimation architecture. Different from previous works that achieve iterative refinement by network cascading or untrainable IC-LK iterator; the iterator of IHN has tied weights and is completely trainable. IHN achieves state-of-the-art accuracy on several datasets including challenging scenes. We propose 2 versions of IHN: (1) IHN for static scenes, (2) IHN-mov for dynamic scenes with moving objects. Both versions can be arranged in 1-scale for efficiency or 2-scale for accuracy. We show that the basic 1-scale IHN already outperforms most of the existing methods. On a variety of datasets, the 2-scale IHN outperforms all competitors by a large gap. We introduce IHN-mov by producing an inlier mask to further improve the estimation accuracy of moving-objects scenes. We experimentally show that the iterative framework of IHN can achieve 95% error reduction while considerably saving network parameters. When processing sequential image pairs, IHN can achieve 32.7 fps, which is about 8× the speed of IC-LK iterator: Source code is available at https://github.com/imdump178/IHN.

----

## [187] GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors

**Authors**: *Jingwen He, Wu Shi, Kai Chen, Lean Fu, Chao Dong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00193](https://doi.org/10.1109/CVPR52688.2022.00193)

**Abstract**:

Face image super resolution (face hallucination) usu-ally relies on facial priors to restore realistic details and preserve identity information. Recent advances can achieve impressive results with the help of GAN prior. They ei-ther design complicated modules to modify the fixed GAN prior or adopt complex training strategies to finetune the generator. In this work, we propose a generative and controllable face SR framework, called GCFSR, which can re-construct images with faithful identity information without any additional priors. Generally, GCFSR has an encoder-generator architecture. Two modules called style modu-lation and feature modulation are designed for the multi-factor SR task. The style modulation aims to generate real-istic face details and the feature modulation dynamically fuses the multi-level encoded features and the generated ones conditioned on the upscaling factor. The simple and elegant architecture can be trained from scratch in an end-to-end manner. For small upscaling factors (≤8), GCFSR can produce surprisingly good results with only adversar-ialloss. After adding L1 and perceptual losses, GCFSR can outperform state-of-the-art methods for large upscalingfac-tors (16, 32, 64). During the test phase, we can modulate the generative strength via feature modulation by changing the conditional upscaling factor continuously to achieve various generative effects. Code is available at https://github.com/hejingwenhejingwen/GCFSR

----

## [188] Deep Color Consistent Network for Low-Light Image Enhancement

**Authors**: *Zhao Zhang, Huan Zheng, Richang Hong, Mingliang Xu, Shuicheng Yan, Meng Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00194](https://doi.org/10.1109/CVPR52688.2022.00194)

**Abstract**:

Low-light image enhancement (LLIE) explores how to refine the illumination and obtain natural normal-light images. Current LLIE methods mainly focus on improving the illumination, but do not consider the color consistency by reasonably incorporating color information into the LLIE process. As a result, color difference usually exists between the enhanced image and ground-truth. To address this issue, we propose a new deep color consistent network termed DCC-Net to retain the color consistency for LLIE. A new “divide and conquer” collaborative strategy is presented, which can jointly preserve color information and enhance the illumination. Specifically, the decoupling strategy of our DCC-Net decouples each color image into two main components, i.e., gray image plus color histogram. Gray image is used to generate reasonable structures and textures, and the color histogram is beneficial for preserving the color consistency. That is, they both are utilized to complete the LLIE task collaboratively. To match the color and content features, and reduce the color consistency gap between enhanced image and ground-truth, we also design a new pyramid color embedding (PCE) module, which can better embed color information into the LLIE process. Extensive experiments on six real datasets show that the enhanced images of our DCC-Net are more natural and colorful, and perform favorably against the state-of-the-art methods.

----

## [189] LAR-SR: A Local Autoregressive Model for Image Super-Resolution

**Authors**: *Baisong Guo, Xiaoyun Zhang, Haoning Wu, Yu Wang, Ya Zhang, Yan-Feng Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00195](https://doi.org/10.1109/CVPR52688.2022.00195)

**Abstract**:

Previous super-resolution (SR) approaches often formulate SR as a regression problem and pixel wise restoration, which leads to a blurry and unreal SR output. Recent works combine adversarial loss with pixel-wise loss to train a GAN-based model or introduce normalizing flows into SR problems to generate more realistic images. As another powerful generative approach, autoregressive (AR) model has not been noticed in low level tasks due to its limitation. Based on the fact that given the structural in-formation, the textural details in the natural images are locally related without long term dependency, in this paper we propose a novel autoregressive model-based SR approach, namely LAR-SR, which can efficiently generate realistic SR images using a novel local autoregressive (LAR) module. The proposed LAR module can sample all the patches of textural components in parallel, which greatly reduces the time consumption. In addition to high time efficiency, it is also able to leverage contextual information of pixels and can be optimized with a consistent loss. Experimental results on the widely-used datasets show that the proposed LAR-SR approach achieves superior performance on the vi-sual quality and quantitative metrics compared with other generative models such as GAN, Flow, and is competitive with the mixture generative model.

----

## [190] Multi-Scale Memory-Based Video Deblurring

**Authors**: *Bo Ji, Angela Yao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00196](https://doi.org/10.1109/CVPR52688.2022.00196)

**Abstract**:

Video deblurring has achieved remarkable progress thanks to the success of deep neural networks. Most methods solve for the deblurring end-to-end with limited information propagation from the video sequence. However, different frame regions exhibit different characteristics and should be provided with corresponding relevant information. To achieve fine-grained deblurring, we designed a memory branch to memorize the blurry-sharp feature pairs in the memory bank, thus providing useful information for the blurry query input. To enrich the memory of our memory bank, we further designed a bidirectional recurrency and multi-scale strategy based on the memory bank. Experimental results demonstrate that our model outperforms other state-of-the-art methods while keeping the model complexity and inference time low. The code is available at https://github.com/jibo27/MemDeblur.

----

## [191] Local Texture Estimator for Implicit Representation Function

**Authors**: *Jaewon Lee, Kyong Hwan Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00197](https://doi.org/10.1109/CVPR52688.2022.00197)

**Abstract**:

Recent works with an implicit neural function shed light on representing images in arbitrary resolution. However, a standalone multi-layer perceptron shows limited performance in learning high-frequency components. In this paper, we propose a Local Texture Estimator (LTE), a dominant-frequency estimator for natural images, enabling an implicit function to capture fine details while reconstructing images in a continuous manner. When jointly trained with a deep super-resolution (SR) architecture, LTE is capable of characterizing image textures in 2D Fourier space. We show that an LTE-based neuralfunction achieves favorable performance against existing deep SR methods within an arbitrary-scale factor. Furthermore, we demonstrate that our implementation takes the shortest running time compared to previous works.

----

## [192] ChiTransformer: Towards Reliable Stereo from Cues

**Authors**: *Qing Su, Shihao Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00198](https://doi.org/10.1109/CVPR52688.2022.00198)

**Abstract**:

Current stereo matching techniques are challenged by restricted searching space, occluded regions and sheer size. While single image depth estimation is spared from these challenges and can achieve satisfactory results with the extracted monocular cues, the lack of stereoscopic relationship renders the monocular prediction less reliable on its own especially in highly dynamic or cluttered environments. To address these issues in both scenarios, we present an optic-chiasm-inspired self-supervised binocular depth estimation method, wherein vision transformer (ViT) with a gated positional cross-attention (GPCA) layer is designed to enable feature-sensitive pattern retrieval between views, while retaining the extensive context information aggregated through self-attentions. Monocular cues from a single view are thereafter conditionally rectified by a blending layer with the retrieved pattern pairs. This crossover design is biologically analogous to the optic-chasma structure in human visual system and hence the name, Chi-Transformer. Our experiments show that this architecture yields substantial improvements over state-of-the-art self-supervised stereo approaches by 11%, and can be used on both rectilinear and non-rectilinear (e.g., fisheye) images.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/ISL-CV/ChiTransformer.git

----

## [193] PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images

**Authors**: *Stefano Zorzi, Shabab Bazrafkan, Stefan Habenschuss, Friedrich Fraundorfer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00189](https://doi.org/10.1109/CVPR52688.2022.00189)

**Abstract**:

While most state-of-the-art instance segmentation methods produce binary segmentation masks, geographic and cartographic applications typically require precise vector polygons of extracted objects instead of rasterized output. This paper introduces PolyWorld, a neural network that directly extracts building vertices from an image and connects them correctly to create precise polygons. The model predicts the connection strength between each pair of vertices using a graph neural network and estimates the assignments by solving a differentiable optimal transport problem. Moreover, the vertex positions are optimized by minimizing a combined segmentation and polygonal angle difference loss. PolyWorld significantly outperforms the state of the art in building polygonization and achieves not only notable quantitative results, but also produces visually pleasing building polygons. Code and trained weights are publicly available at https://thub.com/zorzis/yWorl-PoldPretrainedNetwork.

----

## [194] BNUDC: A Two-Branched Deep Neural Network for Restoring Images from Under-Display Cameras

**Authors**: *Jaihyun Koh, Jangho Lee, Sungroh Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00199](https://doi.org/10.1109/CVPR52688.2022.00199)

**Abstract**:

The images captured by under-display cameras (UDCs) are degraded by the screen in front of them. We model this degradation in terms of a) diffraction by the pixel grid, which attenuates high-spatial-frequency components of the image; and b) diffuse intensity and color changes caused by the multiple thin-film layers in an OLED, which modulate the low-spatial-frequency components of the image. We introduce a deep neural network with two branches to reverse each type of degradation, which is more effective than performing both restorations in a single forward network. We also propose an affine transform connection to replace the skip connection used in most existing DNNs for restoring UDC images. Confining the solution space to the linear transform domain reduces the blurring caused by convolution; and any gross color shift in the training images is eliminated by inverse color filtering. Trained on three datasets of UDC images, our network outperformed existing methods in terms of measures of distortion and of perceived image quality.

----

## [195] ISNAS-DIP: Image-Specific Neural Architecture Search for Deep Image Prior

**Authors**: *Metin Ersin Arican, Ozgur Kara, Gustav Bredell, Ender Konukoglu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00200](https://doi.org/10.1109/CVPR52688.2022.00200)

**Abstract**:

Recent works show that convolutional neural network (CNN) architectures have a spectral bias towards lower frequencies, which has been leveraged for various image restoration tasks in the Deep Image Prior (DIP) framework. The benefit of the inductive bias the network imposes in the DIP framework depends on the architecture. Therefore, re-searchers have studied how to automate the search to de-termine the best-performing model. However, common neu-ral architecture search (NAS) techniques are resource and time-intensive. Moreover, best-performing models are de-termined for a whole dataset of images instead of for each image independently, which would be prohibitively expen-sive. In this work, we first show that optimal neural archi-tectures in the DIP framework are image-dependent. Lever-aging this insight, we then propose an image-specific NAS strategy for the DIP framework that requires substantially less training than typical NAS approaches, effectively en-abling image-specific NAS. We justify the proposed strat-egy's effectiveness by (1) demonstrating its performance on a NAS Dataset for DIP that includes 522 models from a particular search space (2) conducting extensive experi-ments on image denoising, inpainting, and super-resolution tasks. Our experiments show that image-specific metrics can reduce the search space to a small cohort of models, of which the best model outperforms current NAS approaches for image restoration. Codes and datasets are available at https://github.com/ozgurkara99/ISNAS-DIP.

----

## [196] IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation

**Authors**: *Lingtong Kong, Boyuan Jiang, Donghao Luo, Wenqing Chu, Xiaoming Huang, Ying Tai, Chengjie Wang, Jie Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00201](https://doi.org/10.1109/CVPR52688.2022.00201)

**Abstract**:

Prevailing video frame interpolation algorithms, that generate the intermediate frames from consecutive inputs, typically rely on complex model architectures with heavy parameters or large delay, hindering them from diverse real-time applications. In this work, we devise an efficient encoder-decoder based network, termed IFRNet, for fast in-termediate frame synthesizing. It first extracts pyramid features from given inputs, and then refines the bilateral in-termediate flow fields together with a powerful intermedi-ate feature until generating the desired output. The gradu-ally refined intermediate feature can not only facilitate in-termediate flow estimation, but also compensate for con-textual details, making IFRNet do not need additional syn-thesis or refinement module. To fully release its potential, we further propose a novel task-oriented optical flow dis-tillation loss to focus on learning the useful teacher knowl-edge towards frame synthesizing. Meanwhile, a new ge-ometry consistency regularization term is imposed on the gradually refined intermediate features to keep better structure layout. Experiments on various benchmarks demon-strate the excellent performance and fast inference speed of proposed approaches. Code is available at https://github.com/ltkong218/IFRNet.

----

## [197] Learning Graph Regularisation for Guided Super-Resolution

**Authors**: *Riccardo de Lutio, Alexander Becker, Stefano D'Aronco, Stefania Russo, Jan D. Wegner, Konrad Schindler*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00202](https://doi.org/10.1109/CVPR52688.2022.00202)

**Abstract**:

We introduce a novel formulation for guided super-resolution. Its core is a differentiable optimisation layer that operates on a learned affinity graph. The learned graph potentials make it possible to leverage rich contextual information from the guide image, while the explicit graph optimisation within the architecture guarantees rigorous fidelity of the high-resolution target to the low-resolution source. With the decision to employ the source as a constraint rather than only as an input to the prediction, our method differs from state-of-the-art deep architectures for guided super-resolution, which produce targets that, when downsampled, will only approximately reproduce the source. This is not only theoretically appealing, but also produces crisper, more natural-looking images. A key property of our method is that, although the graph connectivity is restricted to the pixel lattice, the associated edge potentials are learned with a deep feature extractor and can encode rich context information over large receptive fields. By taking advantage of the sparse graph connectivity, it becomes possible to propagate gradients through the optimisation layer and learn the edge potentials from data. We extensively evaluate our method on several datasets, and consistently outperform recent baselines in terms of quantitative reconstruction errors, while also delivering visually sharper outputs. Moreover, we demonstrate that our method generalises particularly well to new datasets not seen during training.

----

## [198] Self-supervised Deep Image Restoration via Adaptive Stochastic Gradient Langevin Dynamics

**Authors**: *Weixi Wang, Ji Li, Hui Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00203](https://doi.org/10.1109/CVPR52688.2022.00203)

**Abstract**:

While supervised deep learning has been a prominent tool for solving many image restoration problems, there is an increasing interest on studying self-supervised or un-supervised methods to address the challenges and costs of collecting truth images. Based on the neuralization of a Bayesian estimator of the problem, this paper presents a self-supervised deep learning approach to general image restoration problems. The key ingredient of the neuralized estimator is an adaptive stochastic gradient Langevin dy-namics algorithm for efficiently sampling the posterior distri-bution of network weights. The proposed method is applied on two image restoration problems: compressed sensing and phase retrieval. The experiments on these applications showed that the proposed method not only outperformed existing non-learning and unsupervised solutions in terms of image restoration quality, but also is more computationally efficient.

----

## [199] Self-Supervised Arbitrary-Scale Point Clouds Upsampling via Implicit Neural Representation

**Authors**: *Wenbo Zhao, Xianming Liu, Zhiwei Zhong, Junjun Jiang, Wei Gao, Ge Li, Xiangyang Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.00204](https://doi.org/10.1109/CVPR52688.2022.00204)

**Abstract**:

Point clouds upsampling is a challenging issue to gener-ate dense and uniform point clouds from the given sparse input. Most existing methods either take the end-to-end su-pervised learning based manner, where large amounts of pairs of sparse input and dense ground-truth are exploited as supervision information; or treat up-scaling of different scale factors as independent tasks, and have to build multiple networks to handle upsampling with varying factors. In this paper, we propose a novel approach that achieves self-supervised and magnification-flexible point clouds upsampling simultaneously. We formulate point clouds upsampling as the task of seeking nearest projection points on the implicit surface for seed points. To this end, we define two implicit neural functions to estimate projection direction and distance respectively, which can be trained by two pretext learning tasks. Experimental results demonstrate that our self-supervised learning based scheme achieves competitive or even better performance than supervised learning based state-of-the-art methods. The source code is publicly available at https://github.com/xnowbzhaolsapcu.

----



[Go to the next page](CVPR-2022-list02.md)

[Go to the catalog section](README.md)