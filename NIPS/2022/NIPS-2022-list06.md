## [1000] What Makes Graph Neural Networks Miscalibrated?

**Authors**: *Hans Hao-Hsun Hsu, Yuesong Shen, Christian Tomani, Daniel Cremers*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5975754c7650dfee0682e06e1fec0522-Abstract-Conference.html)

**Abstract**:

Given the importance of getting calibrated predictions and reliable uncertainty estimations, various post-hoc calibration methods have been developed for neural networks on standard multi-class classification tasks. However, these methods are not well suited for calibrating graph neural networks (GNNs), which presents unique challenges such as accounting for the graph structure and the graph-induced correlations between the nodes. In this work, we conduct a systematic study on the calibration qualities of GNN node predictions. In particular, we identify five factors which influence the calibration of GNNs: general under-confident tendency, diversity of nodewise predictive distributions, distance to training nodes, relative confidence level, and neighborhood similarity. Furthermore, based on the insights from this study, we design a novel calibration method named Graph Attention Temperature Scaling (GATS), which is tailored for calibrating graph neural networks. GATS incorporates designs that address all the identified influential factors and produces nodewise temperature scaling using an attention-based architecture. GATS is accuracy-preserving, data-efficient, and expressive at the same time. Our experiments empirically verify the effectiveness of GATS, demonstrating that it can consistently achieve state-of-the-art calibration results on various graph datasets for different GNN backbones.

----

## [1001] Stochastic Adaptive Activation Function

**Authors**: *Kyungsu Lee, Jaeseung Yang, Haeyun Lee, Jae Youn Hwang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/59841d5dfa567f0db25755b391d1f41a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/59841d5dfa567f0db25755b391d1f41a-Abstract-Conference.html)

**Abstract**:

The simulation of human neurons and neurotransmission mechanisms has been realized in deep neural networks based on the theoretical implementations of activation functions. However, recent studies have reported that the threshold potential of neurons exhibits different values according to the locations and types of individual neurons, and that the activation functions have limitations in terms of representing this variability. Therefore, this study proposes a simple yet effective activation function that facilitates different thresholds and adaptive activations according to the positions of units and the contexts of inputs. Furthermore, the proposed activation function mathematically exhibits a more generalized form of Swish activation function, and thus we denoted it as Adaptive SwisH (ASH). ASH highlights informative features that exhibit large values in the top percentiles in an input, whereas it rectifies low values. Most importantly, ASH exhibits trainable, adaptive, and context-aware properties compared to other activation functions. Furthermore, ASH represents general formula of the previously studied activation function and provides a reasonable mathematical background for the superior performance. To validate the effectiveness and robustness of ASH, we implemented ASH into many deep learning models for various tasks, including classification, detection, segmentation, and image generation. Experimental analysis demonstrates that our activation function can provide the benefits of more accurate prediction and earlier convergence in many deep learning applications.

----

## [1002] ActionSense: A Multimodal Dataset and Recording Framework for Human Activities Using Wearable Sensors in a Kitchen Environment

**Authors**: *Joseph DelPreto, Chao Liu, Yiyue Luo, Michael Foshey, Yunzhu Li, Antonio Torralba, Wojciech Matusik, Daniela Rus*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5985e81d65605827ac35401999aea22a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/5985e81d65605827ac35401999aea22a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper introduces ActionSense, a multimodal dataset and recording framework with an emphasis on wearable sensing in a kitchen environment.  It provides rich, synchronized data streams along with ground truth data to facilitate learning pipelines that could extract insights about how humans interact with the physical world during activities of daily living, and help lead to more capable and collaborative robot assistants.  The wearable sensing suite captures motion, force, and attention information; it includes eye tracking with a first-person camera, forearm muscle activity sensors, a body-tracking system using 17 inertial sensors, finger-tracking gloves, and custom tactile sensors on the hands that use a matrix of conductive threads.  This is coupled with activity labels and with externally-captured data from multiple RGB cameras, a depth camera, and microphones.  The specific tasks recorded in ActionSense are designed to highlight lower-level physical skills and higher-level scene reasoning or action planning.  They include simple object manipulations (e.g., stacking plates), dexterous actions (e.g., peeling or cutting vegetables), and complex action sequences (e.g., setting a table or loading a dishwasher).  The resulting dataset and underlying experiment framework are available at https://action-sense.csail.mit.edu. Preliminary networks and analyses explore modality subsets and cross-modal correlations.  ActionSense aims to support applications including learning from demonstrations, dexterous robot control, cross-modal predictions, and fine-grained action segmentation. It could also help inform the next generation of smart textiles that may one day unobtrusively send rich data streams to in-home collaborative or autonomous robot assistants.

----

## [1003] Video compression dataset and benchmark of learning-based video-quality metrics

**Authors**: *Anastasia Antsiferova, Sergey Lavrushkin, Maksim Smirnov, Aleksandr Gushchin, Dmitriy S. Vatolin, Dmitriy L. Kulikov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/59ac9f01ea2f701310f3d42037546e4a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/59ac9f01ea2f701310f3d42037546e4a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Video-quality measurement is a critical task in video processing. Nowadays, many implementations of new encoding standards - such as AV1, VVC, and LCEVC - use deep-learning-based decoding algorithms with perceptual metrics that serve as optimization objectives. But investigations of the performance of modern video- and image-quality metrics commonly employ videos compressed using older standards, such as AVC. In this paper, we present a new benchmark for video-quality metrics that evaluates video compression. It is based on a new dataset consisting of about 2,500 streams encoded using different standards, including AVC, HEVC, AV1, VP9, and VVC.  Subjective scores were collected using crowdsourced pairwise comparisons. The list of evaluated metrics includes recent ones based on machine learning and neural networks. The results demonstrate that new no-reference metrics exhibit high correlation with subjective quality and approach the capability of top full-reference metrics.

----

## [1004] Scalable Distributional Robustness in a Class of Non-Convex Optimization with Guarantees

**Authors**: *Avinandan Bose, Arunesh Sinha, Tien Mai*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/59e02a1440e6667e01628ed4c325255c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/59e02a1440e6667e01628ed4c325255c-Abstract-Conference.html)

**Abstract**:

Distributionally robust optimization (DRO) has shown a lot of promise in providing robustness in learning as well as sample-based optimization problems. We endeavor to provide DRO solutions for a class of sum of fractionals, non-convex optimization which is used for decision making in prominent areas such as facility location and security games. In contrast to previous work, we find it more tractable to optimize the equivalent variance regularized form of DRO rather than the minimax form. We transform the variance regularized form to a mixed-integer second-order cone program (MISOCP), which, while guaranteeing global optimality, does not scale enough to solve problems with real-world datasets. We further propose two abstraction approaches based on clustering and stratified sampling to increase scalability, which we then use for real-world datasets. Importantly, we provide global optimality guarantees for our approach and show experimentally that our solution quality is better than the locally optimal ones achieved by state-of-the-art gradient-based methods. We experimentally compare our different approaches and baselines and reveal nuanced properties of a DRO solution.

----

## [1005] Prototypical VoteNet for Few-Shot 3D Point Cloud Object Detection

**Authors**: *Shizhen Zhao, Xiaojuan Qi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/59e73ff865b56cba6ab7f6b2cce1425d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/59e73ff865b56cba6ab7f6b2cce1425d-Abstract-Conference.html)

**Abstract**:

Most existing 3D point cloud object detection approaches heavily rely on large amounts of labeled training data. However, the labeling process is costly and time-consuming. This paper considers few-shot 3D point cloud object detection, where only a few annotated samples of novel classes are needed with abundant samples of base classes. To this end, we propose Prototypical VoteNet to recognize and localize novel instances, which incorporates two new modules: Prototypical Vote Module (PVM) and Prototypical Head Module (PHM). Specifically, as the 3D basic geometric structures can be shared among categories, PVM is designed to leverage class-agnostic geometric prototypes, which are learned from base classes, to refine local features of novel categories. Then PHM is proposed to utilize class prototypes to enhance the global feature of each object, facilitating subsequent object localization and classification, which is trained by the episodic training strategy. To evaluate the model in this new setting, we contribute two new benchmark datasets, FS-ScanNet and FS-SUNRGBD. We conduct extensive experiments to demonstrate the effectiveness of Prototypical VoteNet, and our proposed method shows significant and consistent improvements compared to baselines on two benchmark datasets.

----

## [1006] Learning from a Sample in Online Algorithms

**Authors**: *C. J. Argue, Alan M. Frieze, Anupam Gupta, Christopher Seiler*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a093120ff4776b4f0dc452e3e3b6652-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a093120ff4776b4f0dc452e3e3b6652-Abstract-Conference.html)

**Abstract**:

We consider three central problems in optimization: the restricted  assignment load-balancing problem, the Steiner tree network design  problem, and facility location clustering. We consider the online  setting, where the input arrives over time, and irrevocable decisions  must be made without knowledge of the future.  For all these problems, any online algorithm must incur a cost that is  approximately $\log |I|$ times the optimal cost in the worst-case,  where $|I|$ is the length of the input. But can we go beyond the  worst-case?  In this work we give algorithms that perform substantially  better when a $p$-fraction of the input is given as a sample: the  algorithm use this sample to \emph{learn} a good strategy to use  for the rest of the input.

----

## [1007] Robust Rent Division

**Authors**: *Dominik Peters, Ariel D. Procaccia, David Zhu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a1a10c2c2c9b9af1514687bc24b8f3d-Abstract-Conference.html)

**Abstract**:

In fair rent division, the problem is to assign rooms to roommates and fairly split the rent based on roommates' reported valuations for the rooms. Envy-free rent division is the most popular application on the fair division website Spliddit. The standard model assumes that agents can correctly report their valuations for each room. In practice, agents may be unsure about their valuations, for example because they have had only limited time to inspect the rooms. Our goal is to find a robust rent division that remains fair even if agent valuations are slightly different from the reported ones. We introduce the lexislack solution, which selects a rent division that remains envy-free for valuations within as large a radius as possible of the reported valuations. We also consider robustness notions for valuations that come from a probability distribution, and use results from learning theory to show how we can find rent divisions that (almost) maximize the probability of being envy-free, or that minimize the expected envy. We show that an almost optimal allocation can be identified based on polynomially many samples from the valuation distribution. Finding the best allocation given these samples is NP-hard, but in practice such an allocation can be found using integer linear programming.

----

## [1008] Efficient Dataset Distillation using Random Feature Approximation

**Authors**: *Noel Loo, Ramin M. Hasani, Alexander Amini, Daniela Rus*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a28f46993c19f428f482cc59db40870-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a28f46993c19f428f482cc59db40870-Abstract-Conference.html)

**Abstract**:

Dataset distillation compresses large datasets into smaller synthetic coresets which retain performance with the aim of reducing the storage and computational burden of processing the entire dataset. Today's best performing algorithm, \textit{Kernel Inducing Points} (KIP), which makes use of the correspondence between infinite-width neural networks and kernel-ridge regression, is prohibitively slow due to the exact computation of the neural tangent kernel matrix, scaling $O(|S|^2)$, with $|S|$ being the coreset size. To improve this, we propose a novel algorithm that uses a random feature approximation (RFA) of the Neural Network Gaussian Process (NNGP) kernel which reduces the kernel matrix computation to $O(|S|)$.  Our algorithm provides at least a 100-fold speedup over KIP and can run on a single GPU. Our new method, termed an RFA Distillation (RFAD), performs competitively with KIP and other dataset condensation algorithms in accuracy over a range of large-scale datasets, both in kernel regression and finite-width network training. We demonstrate the effectiveness of our approach on tasks involving model interpretability and privacy preservation.

----

## [1009] Scalable Sensitivity and Uncertainty Analyses for Causal-Effect Estimates of Continuous-Valued Interventions

**Authors**: *Andrew Jesson, Alyson Douglas, Peter Manshausen, Maëlys Solal, Nicolai Meinshausen, Philip Stier, Yarin Gal, Uri Shalit*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a29c3d172b80bab1238ddc227246c52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a29c3d172b80bab1238ddc227246c52-Abstract-Conference.html)

**Abstract**:

Estimating the effects of continuous-valued interventions from observational data is a critically important task for climate science, healthcare, and economics. Recent work focuses on designing neural network architectures and regularization functions to allow for scalable estimation of average and individual-level dose-response curves from high-dimensional, large-sample data. Such methodologies assume ignorability (observation of all confounding variables) and positivity (observation of all treatment levels for every covariate value describing a set of units), assumptions problematic in the continuous treatment regime. Scalable sensitivity and uncertainty analyses to understand the ignorance induced in causal estimates when these assumptions are relaxed are less studied. Here, we develop a continuous treatment-effect marginal sensitivity model (CMSM) and derive bounds that agree with the observed data and a researcher-defined level of hidden confounding. We introduce a scalable algorithm and uncertainty-aware deep models to derive and estimate these bounds for high-dimensional, large-sample observational data. We work in concert with climate scientists interested in the climatological impacts of human emissions on cloud properties using satellite observations from the past 15 years. This problem is known to be complicated by many unobserved confounders.

----

## [1010] Sparse Interaction Additive Networks via Feature Interaction Detection and Sparse Selection

**Authors**: *James Enouen, Yan Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a3674849d6d6d23ac088b9a2552f323-Abstract-Conference.html)

**Abstract**:

There is currently a large gap in performance between the statistically rigorous methods like linear regression or additive splines and the powerful deep methods using neural networks.  Previous works attempting to close this gap have failed to fully consider the exponentially growing number of feature combinations which deep networks consider automatically during training.  In this work, we develop a tractable selection algorithm to efficiently identify the necessary feature combinations by leveraging techniques in feature interaction detection.Our proposed Sparse Interaction Additive Networks (SIAN) construct a bridge from these simple and interpretable models to a fully connected neural network.  SIAN achieves competitive performance against state-of-the-art methods across multiple large-scale tabular datasets and consistently finds an optimal tradeoff between the modeling capacity of neural networks and the generalizability of simpler methods.

----

## [1011] Faster Stochastic Algorithms for Minimax Optimization under Polyak-{\L}ojasiewicz Condition

**Authors**: *Lesi Chen, Boyuan Yao, Luo Luo*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a4699b3d0bf7ba934fe10cdba5a8a32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a4699b3d0bf7ba934fe10cdba5a8a32-Abstract-Conference.html)

**Abstract**:

This paper considers stochastic first-order algorithms for minimax optimization under Polyak-{\L}ojasiewicz (PL) conditions. We propose SPIDER-GDA for solving the finite-sum problem of the form $\min_x \max_y f(x,y)\triangleq \frac{1}{n} \sum_{i=1}^n f_i(x,y)$, where the objective function $f(x,y)$ is $\mu_x$-PL in $x$ and $\mu_y$-PL in $y$; and each $f_i(x,y)$ is $L$-smooth. We prove SPIDER-GDA could find an $\epsilon$-approximate solution within ${\mathcal O}\left((n + \sqrt{n}\,\kappa_x\kappa_y^2)\log (1/\epsilon)\right)$ stochastic first-order oracle (SFO) complexity, which is better than the state-of-the-art method whose SFO upper bound is ${\mathcal O}\big((n + n^{2/3}\kappa_x\kappa_y^2)\log (1/\epsilon)\big)$, where $\kappa_x\triangleq L/\mu_x$ and $\kappa_y\triangleq L/\mu_y$.For the ill-conditioned case, we provide an accelerated algorithm to reduce the computational cost further. It achieves $\tilde{{\mathcal O}}\big((n+\sqrt{n}\,\kappa_x\kappa_y)\log^2 (1/\epsilon)\big)$ SFO upper bound when $\kappa_x\geq\sqrt{n}$. Our ideas also can be applied to the more general setting that the objective function only satisfies PL condition for one variable. Numerical experiments validate the superiority of proposed methods.

----

## [1012] Kantorovich Strikes Back! Wasserstein GANs are not Optimal Transport?

**Authors**: *Alexander Korotin, Alexander Kolesov, Evgeny Burnaev*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a5aacae31b6d41edf49bc43bccb7c4f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a5aacae31b6d41edf49bc43bccb7c4f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Wasserstein Generative Adversarial Networks (WGANs) are the popular generative models built on the theory of Optimal Transport (OT) and the Kantorovich duality. Despite the success of WGANs, it is still unclear how well the underlying OT dual solvers approximate the OT cost (Wasserstein-1 distance, W1) and the OT gradient needed to update the generator. In this paper, we address these questions. We construct 1-Lipschitz functions and use them to build ray monotone transport plans. This strategy yields pairs of continuous benchmark distributions with the analytically known OT plan, OT cost and OT gradient in high-dimensional spaces such as spaces of images. We thoroughly evaluate popular WGAN dual form solvers (gradient penalty, spectral normalization, entropic regularization, etc.) using these benchmark pairs. Even though these solvers perform well in WGANs, none of them faithfully compute W1 in high dimensions. Nevertheless, many provide a meaningful approximation of the OT gradient. These observations suggest that these solvers should not be treated as good estimators of W1 but to some extent they indeed can be used in variational problems requiring the minimization of W1.

----

## [1013] List-Decodable Sparse Mean Estimation via Difference-of-Pairs Filtering

**Authors**: *Ilias Diakonikolas, Daniel Kane, Sushrut Karmalkar, Ankit Pensia, Thanasis Pittas*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a5ddf0ab751861025c00700093c5677-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a5ddf0ab751861025c00700093c5677-Abstract-Conference.html)

**Abstract**:

We study the problem of list-decodable sparse mean estimation. Specifically, for a parameter $\alpha \in (0, 1/2)$, we are given $m$ points in $\mathbb{R}^n$, $\lfloor \alpha m \rfloor$ of which are i.i.d. samples from a distribution $D$ with unknown $k$-sparse mean $\mu$. No assumptions are made on the remaining points, which form the majority of the dataset. The goal is to return a small list of candidates containing a vector $\hat \mu$ such that $\|\hat \mu - \mu\|_2$ is small. Prior work had studied the problem of list-decodable mean estimation in the dense setting. In this work, we develop a novel, conceptually simpler technique for list-decodable mean estimation. As the main application of our approach, we provide the first sample and computationally efficient algorithm for list-decodable sparse mean estimation. In particular, for distributions with  ``certifiably bounded'' $t$-th moments in $k$-sparse directions and sufficiently light tails, our algorithm achieves error of $(1/\alpha)^{O(1/t)}$ with sample complexity $m = (k\log(n))^{O(t)}/\alpha$ and running time $\mathrm{poly}(mn^t)$. For the special case of Gaussian inliers, our algorithm achieves the optimal error guarantee $\Theta (\sqrt{\log(1/\alpha)})$ with quasi-polynomial complexity. We complement our upper bounds with nearly-matching statistical query and low-degree polynomial testing lower bounds.

----

## [1014] Distributional Convergence of the Sliced Wasserstein Process

**Authors**: *Jiaqi Xi, Jonathan Niles-Weed*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a5e9197ea547141b4977a5a198bbaac-Abstract-Conference.html)

**Abstract**:

Motivated by the statistical and computational challenges of computing Wasserstein distances in high-dimensional contexts, machine learning researchers have defined modified Wasserstein distances based on computing distances between one-dimensional projections of the measures. Different choices of how to aggregate these projected distances (averaging, random sampling, maximizing) give rise to different distances, requiring different statistical analyses. We define the \emph{Sliced Wasserstein Process}, a stochastic process defined by the empirical Wasserstein distance between projections of empirical probability measures to all one-dimensional subspaces, and prove a uniform distributional limit theorem for this process. As a result, we obtain a unified framework in which to prove sample complexity and distributional limit results for all Wasserstein distances based on one-dimensional projections. We illustrate these results on a number of examples where no distributional limits were previously known.

----

## [1015] VTC-LFC: Vision Transformer Compression with Low-Frequency Components

**Authors**: *Zhenyu Wang, Hao Luo, Pichao Wang, Feng Ding, Fan Wang, Hao Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a8177df23bdcc15a02a6739f5b9dd4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a8177df23bdcc15a02a6739f5b9dd4a-Abstract-Conference.html)

**Abstract**:

Although Vision transformers (ViTs) have recently dominated many vision tasks, deploying ViT models on resource-limited devices remains a challenging problem. To address such a challenge, several methods have been proposed to compress ViTs. Most of them borrow experience in convolutional neural networks (CNNs) and mainly focus on the spatial domain. However, the compression only in the spatial domain suffers from a dramatic performance drop without fine-tuning and is not robust to noise, as the noise in the spatial domain can easily confuse the pruning criteria, leading to some parameters/channels being pruned incorrectly. Inspired by recent findings that self-attention is a low-pass filter and low-frequency signals/components are more informative to ViTs, this paper proposes compressing ViTs with low-frequency components. Two metrics named low-frequency sensitivity (LFS) and low-frequency energy (LFE) are proposed for better channel pruning and token pruning. Additionally, a bottom-up cascade pruning scheme is applied to compress different dimensions jointly. Extensive experiments demonstrate that the proposed method could save 40% ï½ž 60% of the FLOPs in ViTs, thus significantly increasing the throughput on practical devices with less than 1% performance drop on ImageNet-1K.

----

## [1016] A Theory of PAC Learnability under Transformation Invariances

**Authors**: *Han Shao, Omar Montasser, Avrim Blum*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a829e299ebc1c1615ddb09e98fb6ce8-Abstract-Conference.html)

**Abstract**:

Transformation invariances are present in many real-world problems. For example, image classification is usually invariant to rotation and color transformation: a rotated car in a different color is still identified as a car. Data augmentation, which adds the transformed data into the training set and trains a model on the augmented data, is one commonly used technique to build these invariances into the learning process. However, it is unclear how data augmentation performs theoretically and what the optimal algorithm is in presence of transformation invariances. In this paper, we study PAC learnability under transformation invariances in three settings according to different levels of realizability: (i) A hypothesis fits the augmented data; (ii) A hypothesis fits only the original data and the transformed data lying in the support of the data distribution; (iii) Agnostic case. One interesting observation is that distinguishing between the original data and the transformed data is necessary to achieve optimal accuracy in setting (ii) and (iii), which implies that any algorithm not differentiating between the original and transformed data (including data augmentation) is not optimal. Furthermore, this type of algorithms can even ``harm'' the accuracy. In setting (i), although it is unnecessary to distinguish between the two data sets, data augmentation still does not perform optimally. Due to such a difference, we propose two combinatorial measures characterizing the optimal sample complexity in setting (i) and (ii)(iii) and provide the optimal algorithms.

----

## [1017] PALBERT: Teaching ALBERT to Ponder

**Authors**: *Nikita Balagansky, Daniil Gavrilov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5a9c1af5f76da0bd37903b6f23e96c74-Abstract-Conference.html)

**Abstract**:

Currently, pre-trained models can be considered the default choice for a wide range of NLP tasks. Despite their SoTA results, there is practical evidence that these models may require a different number of computing layers for different input sequences, since evaluating all layers leads to overconfidence in wrong predictions (namely overthinking). This problem can potentially be solved by implementing adaptive computation time approaches, which were first designed to improve inference speed. Recently proposed PonderNet may be a promising solution for performing an early exit by treating the exit layer's index as a latent variable. However, the originally proposed exit criterion, relying on sampling from trained posterior distribution on the probability of exiting from the $i$-th layer, introduces major variance in exit layer indices, significantly reducing the resulting model's performance. In this paper, we propose improving PonderNet with a novel deterministic Q-exit criterion and a revisited model architecture. We adapted the proposed mechanism to ALBERT and RoBERTa and compared it with recent methods for performing an early exit. We observed that the proposed changes can be considered significant improvements on the original PonderNet architecture and outperform PABEE on a wide range of GLUE tasks. In addition, we also performed an in-depth ablation study of the proposed architecture to further understand Lambda layers and their performance.

----

## [1018] Distributed Methods with Compressed Communication for Solving Variational Inequalities, with Theoretical Guarantees

**Authors**: *Aleksandr Beznosikov, Peter Richtárik, Michael Diskin, Max Ryabinin, Alexander V. Gasnikov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ac1428c23b5da5e66d029646ea3206d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ac1428c23b5da5e66d029646ea3206d-Abstract-Conference.html)

**Abstract**:

Variational inequalities in general and saddle point problems in particular are increasingly relevant in machine learning applications, including adversarial learning, GANs, transport and robust optimization. With increasing data and problem sizes necessary to train high performing models across various applications, we need to rely on parallel and distributed computing. However, in distributed training, communication among the compute nodes is a key bottleneck during training, and this problem is exacerbated for high dimensional and over-parameterized models. Due to these considerations, it is important to equip existing methods with strategies that would allow to reduce the volume of transmitted information during training while obtaining a model of comparable quality. In this paper, we present the first theoretically grounded distributed methods for solving variational inequalities and saddle point problems using compressed communication: MASHA1 and MASHA2. Our theory and methods allow for the use of both unbiased (such as Rand$k$; MASHA1) and contractive (such as Top$k$; MASHA2) compressors. New algorithms support bidirectional compressions, and also can be modified for stochastic setting with batches and for federated learning with partial participation of clients. We empirically validated our conclusions using two experimental setups: a standard bilinear min-max problem, and large-scale distributed adversarial training of transformers.

----

## [1019] Analyzing Data-Centric Properties for Graph Contrastive Learning

**Authors**: *Puja Trivedi, Ekdeep Singh Lubana, Mark Heimann, Danai Koutra, Jayaraman J. Thiagarajan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5adac7be735715604e8a4b0b2924a7e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5adac7be735715604e8a4b0b2924a7e4-Abstract-Conference.html)

**Abstract**:

Recent analyses of self-supervised learning (SSL) find the following data-centric properties to be critical for learning good representations: invariance to task-irrelevant semantics, separability of classes in some latent space, and recoverability of labels from augmented samples. However, given their discrete, non-Euclidean nature, graph datasets and graph SSL methods are unlikely to satisfy these properties. This raises the question: how do graph SSL methods, such as contrastive learning (CL), work well? To systematically probe this question, we perform a generalization analysis for CL when using generic graph augmentations (GGAs), with a focus on data-centric properties. Our analysis yields formal insights into the limitations of GGAs and the necessity of task-relevant augmentations. As we empirically show, GGAs do not induce task-relevant invariances on common benchmark datasets, leading to only marginal gains over naive, untrained baselines. Our theory motivates a synthetic data generation process that enables control over task-relevant information and boasts pre-defined optimal augmentations. This flexible benchmark helps us identify yet unrecognized limitations in advanced augmentation techniques (e.g., automated methods). Overall, our work rigorously contextualizes, both empirically and theoretically, the effects of data-centric properties on augmentation strategies and learning paradigms for graph SSL.

----

## [1020] Learning Bipartite Graphs: Heavy Tails and Multiple Components

**Authors**: *José Vinícius de Miranda Cardoso, Jiaxi Ying, Daniel P. Palomar*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5adff4d5402703418f7210a4004e1314-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5adff4d5402703418f7210a4004e1314-Abstract-Conference.html)

**Abstract**:

We investigate the problem of learning an undirected, weighted bipartite graph under the Gaussian Markov random field model, for which we present an optimization formulation along with an efficient algorithm based on the projected gradient descent. Motivated by practical applications, where outliers or heavy-tailed events are present, we extend the proposed learning scheme to the case in which the data follow a multivariate Student-$t$ distribution. As a result, the optimization program is no longer convex, but a verifiably convergent iterative algorithm is proposed based on the majorization-minimization framework. Finally, we propose an efficient and provably convergent algorithm for learning $k$-component bipartite graphs that leverages rank constraints of the underlying graph Laplacian matrix. The proposed estimators outperform state-of-the-art methods for bipartite graph learning, as evidenced by real-world experiments using financial time series data.

----

## [1021] Training Scale-Invariant Neural Networks on the Sphere Can Happen in Three Regimes

**Authors**: *Maxim Kodryan, Ekaterina Lobacheva, Maksim Nakhodnov, Dmitry P. Vetrov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5aea56eefab60e06f35016478e21aae6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5aea56eefab60e06f35016478e21aae6-Abstract-Conference.html)

**Abstract**:

A fundamental property of deep learning normalization techniques, such as batch normalization, is making the pre-normalization parameters scale invariant. The intrinsic domain of such parameters is the unit sphere, and therefore their gradient optimization dynamics can be represented via spherical optimization with varying effective learning rate (ELR), which was studied previously. However, the varying ELR may obscure certain characteristics of the intrinsic loss landscape structure. In this work, we investigate the properties of training scale-invariant neural networks directly on the sphere using a fixed ELR. We discover three regimes of such training depending on the ELR value: convergence, chaotic equilibrium, and divergence. We study these regimes in detail both on a theoretical examination of a toy example and on a thorough empirical analysis of real scale-invariant deep learning models. Each regime has unique features and reflects specific properties of the intrinsic loss landscape, some of which have strong parallels with previous research on both regular and scale-invariant neural networks training. Finally, we demonstrate how the discovered regimes are reflected in conventional training of normalized networks and how they can be leveraged to achieve better optima.

----

## [1022] Exploring the Whole Rashomon Set of Sparse Decision Trees

**Authors**: *Rui Xin, Chudi Zhong, Zhi Chen, Takuya Takagi, Margo I. Seltzer, Cynthia Rudin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5afaa8b4dd18eb1eed055d2d821b58ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5afaa8b4dd18eb1eed055d2d821b58ae-Abstract-Conference.html)

**Abstract**:

In any given machine learning problem, there may be many models that could explain the data almost equally well. However, most learning algorithms return only one of these models, leaving practitioners with no practical way to explore alternative models that might have desirable properties beyond what could be expressed within a loss function. The Rashomon set is the set of these all almost-optimal models. Rashomon sets can be extremely complicated, particularly for highly nonlinear function classes that allow complex interaction terms, such as decision trees. We provide the first technique for completely enumerating the Rashomon set for sparse decision trees; in fact, our work provides the first complete enumeration of any Rashomon set for a non-trivial problem with a highly nonlinear discrete function class. This allows the user an unprecedented level of control over model choice among all models that are approximately equally good. We represent the Rashomon set in a specialized data structure that supports efficient querying and sampling. We show three applications of the Rashomon set: 1) it can be used to study variable importance for the set of almost-optimal trees (as opposed to a single tree), 2) the Rashomon set for accuracy enables enumeration of the Rashomon sets for balanced accuracy and F1-score, and 3) the Rashomon set for a full dataset can be used to produce Rashomon sets constructed with only subsets of the data set. Thus, we are able to examine Rashomon sets across problems with a new lens, enabling users to choose models rather than be at the mercy of an algorithm that produces only a single model.

----

## [1023] Graph Self-supervised Learning with Accurate Discrepancy Learning

**Authors**: *Dongki Kim, Jinheon Baek, Sung Ju Hwang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b175f9e93873e3a10a6ce43dbb82e05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b175f9e93873e3a10a6ce43dbb82e05-Abstract-Conference.html)

**Abstract**:

Self-supervised learning of graph neural networks (GNNs) aims to learn an accurate representation of the graphs in an unsupervised manner, to obtain transferable representations of them for diverse downstream tasks. Predictive learning and contrastive learning are the two most prevalent approaches for graph self-supervised learning. However, they have their own drawbacks. While the predictive learning methods can learn the contextual relationships between neighboring nodes and edges, they cannot learn global graph-level similarities. Contrastive learning, while it can learn global graph-level similarities, its objective to maximize the similarity between two differently perturbed graphs may result in representations that cannot discriminate two similar graphs with different properties. To tackle such limitations, we propose a framework that aims to learn the exact discrepancy between the original and the perturbed graphs, coined as Discrepancy-based Self-supervised LeArning (D-SLA). Specifically, we create multiple perturbations of the given graph with varying degrees of similarity, and train the model to predict whether each graph is the original graph or the perturbed one. Moreover, we further aim to accurately capture the amount of discrepancy for each perturbed graph using the graph edit distance. We validate our D-SLA on various graph-related downstream tasks, including molecular property prediction, protein function prediction, and link prediction tasks, on which ours largely outperforms relevant baselines.

----

## [1024] Multi-Scale Adaptive Network for Single Image Denoising

**Authors**: *Yuanbiao Gou, Peng Hu, Jiancheng Lv, Joey Tianyi Zhou, Xi Peng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b288823575bb29654b0953a251e933b-Abstract-Conference.html)

**Abstract**:

Multi-scale architectures have shown effectiveness in a variety of tasks thanks to appealing cross-scale complementarity. However, existing architectures treat different scale features equally without considering the scale-specific characteristics, \textit{i.e.}, the within-scale characteristics are ignored in the architecture design. In this paper, we reveal this missing piece for multi-scale architecture design and accordingly propose a novel Multi-Scale Adaptive Network (MSANet) for single image denoising. Specifically, MSANet simultaneously embraces the within-scale characteristics and the cross-scale complementarity thanks to three novel neural blocks, \textit{i.e.}, adaptive feature block (AFeB), adaptive multi-scale block (AMB), and adaptive fusion block (AFuB). In brief, AFeB is designed to adaptively preserve image details and filter noises, which is highly expected for the features with mixed details and noises. AMB could enlarge the receptive field and aggregate the multi-scale information, which meets the need of contextually informative features. AFuB devotes to adaptively sampling and transferring the features from one scale to another scale, which fuses the multi-scale features with varying characteristics from coarse to fine. Extensive experiments on both three real and six synthetic noisy image datasets show the superiority of MSANet compared with 12 methods. The code could be accessed from https://github.com/XLearning-SCU/2022-NeurIPS-MSANet.

----

## [1025] A theory of weight distribution-constrained learning

**Authors**: *Weishun Zhong, Ben Sorscher, Daniel Lee, Haim Sompolinsky*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b2db6dfda4d7362b2101b2d12dac029-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b2db6dfda4d7362b2101b2d12dac029-Abstract-Conference.html)

**Abstract**:

A central question in computational neuroscience is how structure determines function in neural networks. Recent large-scale connectomic studies have started to provide a wealth of structural information such as the distribution of excitatory/inhibitory cell and synapse types as well as the distribution of synaptic weights in the brains of different species. The emerging high-quality large structural datasets raise the question of what general functional principles can be gleaned from them. Motivated by this question, we developed a statistical mechanical theory of learning in neural networks that incorporates structural information as constraints. We derived an analytical solution for the memory capacity of the perceptron, a basic feedforward model of supervised learning, with constraint on the distribution of its weights. Interestingly, the theory predicts that the reduction in capacity due to the constrained weight-distribution is related to the Wasserstein distance between the cumulative distribution function of the constrained weights and that of the standard normal distribution. To test the theoretical predictions, we use optimal transport theory and information geometry to develop an SGD-based algorithm to find weights that simultaneously learn the input-output task and satisfy the distribution constraint. We show that training in our algorithm can be interpreted as geodesic flows in the Wasserstein space of probability distributions. Given a parameterized family of weight distributions, our theory predicts the shape of the distribution with optimal parameters. We apply our theory to map out the experimental parameter landscape for the estimated distribution of synaptic weights in mammalian cortex and show that our theoryâ€™s prediction for optimal distribution is close to the experimentally measured value. We further developed a statistical mechanical theory for teacher-student perceptron rule learning and ask for the best way for the student to incorporate prior knowledge of the rule (i.e., the teacher). Our theory shows that it is beneficial for the learner to adopt different prior weight distributions during learning, and shows that distribution-constrained learning outperforms unconstrained and sign-constrained learning. Our theory and algorithm provide novel strategies for incorporating prior knowledge about weights into learning, and reveal a powerful connection between structure and function in neural networks.

----

## [1026] Hierarchical Normalization for Robust Monocular Depth Estimation

**Authors**: *Chi Zhang, Wei Yin, Billzb Wang, Gang Yu, Bin Fu, Chunhua Shen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b4a459db23e6db9be2a128380953d96-Abstract-Conference.html)

**Abstract**:

In this paper, we address monocular depth estimation with deep neural networks. To enable training of deep monocular estimation models with various sources of datasets, state-of-the-art methods adopt image-level normalization strategies to generate affine-invariant depth representations. However, learning with the image-level normalization mainly emphasizes the relations of pixel representations with the global statistic in the images, such as the structure of the scene, while the fine-grained depth difference may be overlooked. In this paper, we propose a novel multi-scale depth normalization method that hierarchically normalizes the depth representations based on spatial information and depth distributions. Compared with previous normalization strategies applied only at the holistic image level, the proposed hierarchical normalization can effectively preserve the fine-grained details and improve accuracy. We present two strategies that define the hierarchical normalization contexts in the depth domain and the spatial domain, respectively. Our extensive experiments show that the proposed normalization strategy remarkably outperforms previous normalization methods, and we set new state-of-the-art on five zero-shot transfer benchmark datasets.

----

## [1027] Deep Compression of Pre-trained Transformer Models

**Authors**: *Naigang Wang, Chi-Chun (Charlie) Liu, Swagath Venkataramani, Sanchari Sen, Chia-Yu Chen, Kaoutar El Maghraoui, Vijayalakshmi Srinivasan, Leland Chang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b5618e7d061748267d74478b7c5b1ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b5618e7d061748267d74478b7c5b1ab-Abstract-Conference.html)

**Abstract**:

Pre-trained transformer models have achieved remarkable success in natural language processing (NLP) and have recently become competitive alternatives to Convolution Neural Networks (CNN) and Recurrent Neural Networks (RNN) in vision and speech tasks, respectively. Due to excellent computational efficiency and scalability, transformer models can be trained on exceedingly large amounts of data; however, model sizes can grow tremendously. As high performance, large-scale, and pre-trained transformer models become available for users to download and fine-tune for customized downstream tasks, the deployment of these models becomes challenging due to the vast amount of operations and large memory footprint. To address this challenge, we introduce methods to deeply compress pre-trained transformer models across three major application domains: NLP, speech, and vision. Specifically, we quantize transformer backbones down to 4-bit and further achieve 50% fine-grained structural sparsity on pre-trained BERT, Wav2vec2.0 and Vision Transformer (ViT) models to achieve 16x compression while maintaining model accuracy. This is achieved by identifying the critical initialization for quantization/sparsity aware fine-tuning, as well as novel techniques including quantizers with zero-preserving format and scheduled dropout. These hardware-friendly techniques need only to be applied in the fine-tuning phase for downstream tasks; hence, are especially suitable for acceleration and deployment of pre-trained transformer models.

----

## [1028] Constrained Predictive Coding as a Biologically Plausible Model of the Cortical Hierarchy

**Authors**: *Siavash Golkar, Tiberiu Tesileanu, Yanis Bahroun, Anirvan M. Sengupta, Dmitri B. Chklovskii*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b5de8526aac159e37ff9547713677ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b5de8526aac159e37ff9547713677ed-Abstract-Conference.html)

**Abstract**:

Predictive coding (PC) has emerged as an influential normative model of neural computation with numerous extensions and applications. As such, much effort has been put into mapping PC faithfully onto the cortex, but there are issues that remain unresolved or controversial. In particular, current implementations often involve separate value and error neurons and require symmetric forward and backward weights across different brain regions. These features have not been experimentally confirmed. In this work, we show that the PC framework in the linear regime can be modified to map faithfully onto the cortical hierarchy in a manner compatible with empirical observations. By employing a disentangling-inspired constraint on hidden-layer neural activities, we derive an upper bound for the PC objective. Optimization of this upper bound leads to an algorithm that shows the same performance as the original objective and maps onto a biologically plausible network. The units of this network can be interpreted as multi-compartmental neurons with non-Hebbian learning rules, with a remarkable resemblance to recent experimental findings. There exist prior models which also capture these features, but they are phenomenological, while our work is a normative derivation. This allows us to determine which features are necessary for the functioning of the model. For instance, the network we derive does not involve one-to-one connectivity or signal multiplexing, which the phenomenological models require, indicating that these features are not necessary for learning in the cortex. The normative nature of our algorithm in the simplified linear case also allows us to prove interesting properties of the framework and analytically understand the computational role of our network's components. The parameters of our network have natural interpretations as physiological quantities in a multi-compartmental model of pyramidal neurons, providing a concrete link between PC and experimental measurements carried out in the cortex.

----

## [1029] Washing The Unwashable : On The (Im)possibility of Fairwashing Detection

**Authors**: *Ali Shahin Shamsabadi, Mohammad Yaghini, Natalie Dullerud, Sierra Calanda Wyllie, Ulrich Aïvodji, Aisha Alaagib, Sébastien Gambs, Nicolas Papernot*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b84864ff8474fd742c66f219b2eaac1-Abstract-Conference.html)

**Abstract**:

The use of black-box models (e.g., deep neural networks) in high-stakes decision-making systems, whose internal logic is complex, raises the need for providing explanations about their decisions. Model explanation techniques mitigate this problem by generating an interpretable and high-fidelity surrogate model (e.g., a logistic regressor or decision tree) to explain the logic of black-box models. In this work, we investigate the issue of fairwashing, in which model explanation techniques are manipulated to rationalize decisions taken by an unfair black-box model using deceptive surrogate models. More precisely, we theoretically characterize and analyze fairwashing, proving that this phenomenon is difficult to avoid due to an irreducible factor---the unfairness of the black-box model. Based on the theory developed, we propose a novel technique, called FRAUD-Detect (FaiRness AUDit Detection), to detect fairwashed models by measuring a divergence over subpopulation-wise fidelity measures of the interpretable model. We empirically demonstrate that this divergence is significantly larger in purposefully fairwashed interpretable models than in honest ones. Furthermore, we show that our detector is robust to an informed adversary trying to bypass our detector. The code implementing FRAUD-Detect is available at https://github.com/cleverhans-lab/FRAUD-Detect.

----

## [1030] Near-Optimal Collaborative Learning in Bandits

**Authors**: *Clémence Réda, Sattar Vakili, Emilie Kaufmann*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5b9bef4eae0f574cedbf9f4bf29d8ae7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5b9bef4eae0f574cedbf9f4bf29d8ae7-Abstract-Conference.html)

**Abstract**:

This paper introduces a general multi-agent bandit model in which each agent is facing a finite set of arms and may communicate with other agents through a central controller in order to identify -in pure exploration- or play -in regret minimization- its optimal arm. The twist is that the optimal arm for each agent is the arm with largest expected mixed reward, where the mixed reward of an arm is a weighted sum of the rewards of this arm for all agents. This makes communication between agents often necessary. This general setting allows to recover and extend several recent models for collaborative bandit learning, including the recently proposed federated learning with personalization [Shi et al., 2021]. In this paper, we provide new lower bounds on the sample complexity of pure exploration and on the regret. We then propose a near-optimal algorithm for pure exploration. This algorithm is based on phased elimination with two novel ingredients: a data-dependent sampling scheme within each phase, aimed at matching a relaxation of the lower bound.

----

## [1031] Hypothesis Testing for Differentially Private Linear Regression

**Authors**: *Daniel Alabi, Salil P. Vadhan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bc3356e0fa1753fff7e8d6628e71b22-Abstract-Conference.html)

**Abstract**:

In this work, we design differentially private hypothesis tests for the following problems in the general linear model: testing a linear relationship and testing for the presence of mixtures. The majority of our hypothesis tests are based on differentially private versions of the $F$-statistic for the general linear model framework, which are uniformly most powerful unbiased in the non-private setting. We also present another test for testing mixtures, based on the differentially private nonparametric tests of Couch, Kazan, Shi, Bray, and Groce (CCS 2019), which is especially suited for the small dataset regime. We show that the differentially private $F$-statistic converges to the asymptotic distribution of its non-private counterpart. As a corollary, the statistical power of the differentially private $F$-statistic converges to the statistical power of the non-private $F$-statistic. Through a suite of Monte Carlo based experiments, we show that our tests achieve desired \textit{significance levels} and have a high \textit{power} that approaches the power of the non-private tests as we increase sample sizes or the privacy-loss parameter. We also show when our tests outperform existing methods in the literature.

----

## [1032] Bayesian Optimistic Optimization: Optimistic Exploration for Model-based Reinforcement Learning

**Authors**: *Chenyang Wu, Tianci Li, Zongzhang Zhang, Yang Yu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bcb807ae43ad0851a6ba6162a866404-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bcb807ae43ad0851a6ba6162a866404-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) is a general framework for modeling sequential decision making problems, at the core of which lies the dilemma of exploitation and exploration. An agent failing to explore systematically will inevitably fail to learn efficiently. Optimism in the face of uncertainty (OFU) is a conventionally successful strategy for efficient exploration. An agent following the OFU principle explores actively and efficiently. However, when applied to model-based RL, it involves specifying a confidence set of the underlying model and solving a series of nonlinear constrained optimization, which can be computationally intractable. This paper proposes an algorithm, Bayesian optimistic optimization (BOO), which adopts a dynamic weighting technique for enforcing the constraint rather than explicitly solving a constrained optimization problem. BOO is a general algorithm proved to be sample-efficient for models in a finite-dimensional reproducing kernel Hilbert space. We also develop techniques for effective optimization and show through some simulation experiments that BOO is competitive with the existing algorithms.

----

## [1033] TokenMixup: Efficient Attention-guided Token-level Data Augmentation for Transformers

**Authors**: *Hyeong Kyu Choi, Joonmyung Choi, Hyunwoo J. Kim*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bd09a559a8c8e230697107b0f353d39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bd09a559a8c8e230697107b0f353d39-Abstract-Conference.html)

**Abstract**:

Mixup is a commonly adopted data augmentation technique for image classification. Recent advances in mixup methods primarily focus on mixing based on saliency. However, many saliency detectors require intense computation and are especially burdensome for parameter-heavy transformer models. To this end, we propose TokenMixup, an efficient attention-guided token-level data augmentation method that aims to maximize the saliency of a mixed set of tokens. TokenMixup provides ×15 faster saliency-aware data augmentation compared to gradient-based methods. Moreover, we introduce a variant of TokenMixup which mixes tokens within a single instance, thereby enabling multi-scale feature augmentation. Experiments show that our methods significantly improve the baseline models’ performance on CIFAR and ImageNet-1K, while being more efficient than previous methods. We also reach state-of-the-art performance on CIFAR-100 among from-scratch transformer models. Code is available at https://github.com/mlvlab/TokenMixup.

----

## [1034] Error Analysis of Tensor-Train Cross Approximation

**Authors**: *Zhen Qin, Alexander Lidiak, Zhexuan Gong, Gongguo Tang, Michael B. Wakin, Zhihui Zhu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bd9fbb3a5a985f80c16ddd0ec1dfc43-Abstract-Conference.html)

**Abstract**:

Tensor train decomposition is widely used in machine learning and quantum physics due to its concise representation of high-dimensional tensors, overcoming the curse of dimensionality. Cross approximation---originally developed for representing a matrix from a set of selected rows and columns---is an efficient method for constructing a tensor train decomposition of a tensor from few of its entries. While tensor train cross approximation has achieved remarkable performance in practical applications, its theoretical analysis, in particular regarding the error of the approximation, is so far lacking. To our knowledge, existing results only provide element-wise approximation accuracy guarantees, which lead to a very loose bound when extended to the entire tensor. In this paper, we bridge this gap by providing accuracy guarantees in terms of the entire tensor for both exact and noisy measurements. Our results illustrate how the choice of selected subtensors affects the quality of the cross approximation and that the approximation error caused by model error and/or measurement error may not grow exponentially with the order of the tensor. These results are verified by numerical experiments, and may have important implications for the usefulness of cross approximations for high-order tensors, such as those encountered in the description of quantum many-body states.

----

## [1035] Approximate Secular Equations for the Cubic Regularization Subproblem

**Authors**: *Yihang Gao, Man-Chung Yue, Michael Ng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html)

**Abstract**:

The cubic regularization method (CR) is a popular algorithm for unconstrained non-convex optimization. At each iteration, CR solves a cubically regularized quadratic problem, called the cubic regularization subproblem (CRS). One way to solve the CRS relies on solving the secular equation, whose computational bottleneck lies in the computation of all eigenvalues of the Hessian matrix. In this paper, we propose and analyze a novel CRS solver based on an approximate secular equation, which requires only some of the Hessian eigenvalues and is therefore much more efficient. Two approximate secular equations (ASEs) are developed. For both ASEs, we first study the existence and uniqueness of their roots and then establish an upper bound on the gap between the root and that of the standard secular equation. Such an upper bound can in turn be used to bound the distance from the approximate CRS solution based ASEs to the true CRS solution, thus offering a theoretical guarantee for our CRS solver. A desirable feature of our CRS solver is that it requires only matrix-vector multiplication but not matrix inversion, which makes it particularly suitable for high-dimensional applications of unconstrained non-convex optimization, such as low-rank recovery and deep learning. Numerical experiments with synthetic and real data-sets are conducted to investigate the practical performance of the proposed CRS solver. Experimental results show that the proposed solver outperforms two state-of-the-art methods.

----

## [1036] On the Theoretical Properties of Noise Correlation in Stochastic Optimization

**Authors**: *Aurélien Lucchi, Frank Proske, Antonio Orvieto, Francis R. Bach, Hans Kersting*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html)

**Abstract**:

Studying the properties of stochastic noise to optimize complex non-convex functions has been an active area of research in the field of machine learning. Prior work~\citep{zhou2019pgd, wei2019noise} has shown that the noise of stochastic gradient descent improves optimization by overcoming undesirable obstacles in the landscape. Moreover, injecting artificial Gaussian noise has become a popular idea to quickly escape saddle points. Indeed, in the absence of reliable gradient information, the noise is used to explore the landscape, but it is unclear what type of noise is optimal in terms of exploration ability. In order to narrow this gap in our knowledge, we study a general type of continuous-time non-Markovian process, based on fractional Brownian motion, that allows for the increments of the process to be correlated. This generalizes processes based on Brownian motion, such as the Ornstein-Uhlenbeck process. We demonstrate how to discretize such processes which gives rise to the new algorithm ``fPGD''. This method is a generalization of the known algorithms PGD and Anti-PGD~\citep{orvieto2022anti}. We study the properties of fPGD both theoretically and empirically, demonstrating that it possesses  exploration abilities that, in some cases, are favorable over PGD and Anti-PGD. These results open the field to novel ways to exploit noise for training machine learning models.

----

## [1037] Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models

**Authors**: *Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, Chaowei Xiao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5bf2b802e24106064dc547ae9283bb0c-Abstract-Conference.html)

**Abstract**:

Pre-trained vision-language models (e.g., CLIP) have shown promising zero-shot generalization in many downstream tasks with properly designed text prompts. Instead of relying on hand-engineered prompts, recent works learn prompts using the training data from downstream tasks. While effective, training on domain-specific data reduces a model's generalization capability to unseen new domains. In this work, we propose test-time prompt tuning (TPT), a method that can learn adaptive prompts on the fly with a single test sample. TPT optimizes the prompt by minimizing the entropy with confidence selection so that the model has consistent predictions across different augmented views of each test sample. In evaluating generalization to natural distribution shifts, TPT improves the zero-shot top-1 accuracy of CLIP by 3.6\% on average, surpassing previous prompt tuning approaches that require additional task-specific training data. In evaluating cross-dataset generalization with unseen categories, TPTperforms on par with the state-of-the-art approaches that use additional training data.

----

## [1038] SemMAE: Semantic-Guided Masking for Learning Masked Autoencoders

**Authors**: *Gang Li, Heliang Zheng, Daqing Liu, Chaoyue Wang, Bing Su, Changwen Zheng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c186016d0844767209dc36e9e61441b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c186016d0844767209dc36e9e61441b-Abstract-Conference.html)

**Abstract**:

Recently, significant progress has been made in masked image modeling to catch up to masked language modeling. However, unlike words in NLP, the lack of semantic decomposition of images still makes masked autoencoding (MAE) different between vision and language. In this paper, we explore a potential visual analogue of words, i.e., semantic parts, and we integrate semantic information into the training process of MAE by proposing a Semantic-Guided Masking strategy. Compared to widely adopted random masking, our masking strategy can gradually guide the network to learn various information, i.e., from intra-part patterns to inter-part relations. In particular, we achieve this in two steps. 1) Semantic part learning: we design a self-supervised part learning method to obtain semantic parts by leveraging and refining the multi-head attention of a ViT-based encoder. 2) Semantic-guided MAE (SemMAE) training: we design a masking strategy that varies from masking a portion of patches in each part to masking a portion of (whole) parts in an image. Extensive experiments on various vision tasks show that SemMAE can learn better image representation by integrating semantic information. In particular, SemMAE achieves 84.5% fine-tuning accuracy on ImageNet-1k, which outperforms the vanilla MAE by 1.4%. In the semantic segmentation and fine-grained recognition tasks, SemMAE also brings significant improvements and yields the state-of-the-art performance.

----

## [1039] BiT: Robustly Binarized Multi-distilled Transformer

**Authors**: *Zechun Liu, Barlas Oguz, Aasish Pappu, Lin Xiao, Scott Yih, Meng Li, Raghuraman Krishnamoorthi, Yashar Mehdad*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c1863f711c721648387ac2ef745facb-Abstract-Conference.html)

**Abstract**:

Modern pre-trained transformers have rapidly advanced the state-of-the-art in machine learning, but have also grown in parameters and computational complexity, making them increasingly difficult to deploy in resource-constrained environments. Binarization of the weights and activations of the network can significantly alleviate these issues, however, is technically challenging from an optimization perspective. In this work, we identify a series of improvements that enables binary transformers at a much higher accuracy than what was possible previously. These include a two-set binarization scheme, a novel elastic binary activation function with learned parameters, and a method to quantize a network to its limit by successively distilling higher precision models into lower precision students. These approaches allow for the first time, fully binarized transformer models that are at a practical level of accuracy, approaching a full-precision BERT baseline on the GLUE language understanding benchmark within as little as 5.9%. Code and models are available at:https://github.com/facebookresearch/bit.

----

## [1040] Perturbation Learning Based Anomaly Detection

**Authors**: *Jinyu Cai, Jicong Fan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c261ccdc44fbd32fbb344fa578a1844-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c261ccdc44fbd32fbb344fa578a1844-Abstract-Conference.html)

**Abstract**:

This paper presents a simple yet effective method for anomaly detection. The main idea is to learn small perturbations to perturb normal data and learn a classifier to classify the normal data and the perturbed data into two different classes. The perturbator and classifier are jointly learned using deep neural networks. Importantly, the perturbations should be as small as possible but the classifier is still able to recognize the perturbed data from unperturbed data. Therefore, the perturbed data are regarded as abnormal data and the classifier provides a decision boundary between the normal data and abnormal data, although the training data do not include any abnormal data.Compared with the state-of-the-art of anomaly detection, our method does not require any assumption about the shape (e.g. hypersphere) of the decision boundary and has fewer hyper-parameters to determine. Empirical studies on benchmark datasets verify the effectiveness and superiority of our method.

----

## [1041] Knowledge-Aware Bayesian Deep Topic Model

**Authors**: *Dongsheng Wang, Yishi Xu, Miaoge Li, Zhibin Duan, Chaojie Wang, Bo Chen, Mingyuan Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c60ee4d6e8faf0f3b2f2701c983dc8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c60ee4d6e8faf0f3b2f2701c983dc8c-Abstract-Conference.html)

**Abstract**:

We propose a Bayesian generative model for incorporating prior domain knowledge into hierarchical topic modeling. Although embedded topic models (ETMs) and its variants have gained promising performance in text analysis, they mainly focus on mining word co-occurrence patterns, ignoring potentially easy-to-obtain prior topic hierarchies that could help enhance topic coherence. While several knowledge-based topic models have recently been proposed, they are either only applicable to shallow hierarchies or sensitive to the quality of the provided prior knowledge. To this end, we develop a novel deep ETM that jointly models the documents and the given prior knowledge by embedding the words and topics into the same space. Guided by the provided domain knowledge, the proposed model tends to discover topic hierarchies that are organized into interpretable taxonomies. Moreover, with a technique for adapting a given graph, our extended version allows the structure of the prior knowledge to be fine-tuned to match the target corpus. Extensive experiments show that our proposed model efficiently integrates the prior knowledge and improves both hierarchical topic discovery and document representation.

----

## [1042] SelecMix: Debiased Learning by Contradicting-pair Sampling

**Authors**: *Inwoo Hwang, Sangjun Lee, Yunhyeok Kwak, Seong Joon Oh, Damien Teney, Jin-Hwa Kim, Byoung-Tak Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c6f928e3fc5f32ee29a1d916b68e6f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c6f928e3fc5f32ee29a1d916b68e6f5-Abstract-Conference.html)

**Abstract**:

Neural networks trained with ERM (empirical risk minimization) sometimes learn unintended decision rules, in particular when their training data is biased, i.e., when training labels are strongly correlated with undesirable features. To prevent a network from learning such features, recent methods augment training data such that examples displaying spurious correlations (i.e., bias-aligned examples) become a minority, whereas the other, bias-conflicting examples become prevalent. However, these approaches are sometimes difficult to train and scale to real-world data because they rely on generative models or disentangled representations. We propose an alternative based on mixup, a popular augmentation that creates convex combinations of training examples. Our method, coined SelecMix, applies mixup to contradicting pairs of examples, defined as showing either (i) the same label but dissimilar biased features, or (ii) different labels but similar biased features. Identifying such pairs requires comparing examples with respect to unknown biased features. For this, we utilize an auxiliary contrastive model with the popular heuristic that biased features are learned preferentially during training. Experiments on standard benchmarks demonstrate the effectiveness of the method, in particular when label noise complicates the identification of bias-conflicting examples.

----

## [1043] Are all Frames Equal? Active Sparse Labeling for Video Action Detection

**Authors**: *Aayush Jung Rana, Yogesh S. Rawat*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c81ea77a383cc2848d721224717fa4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c81ea77a383cc2848d721224717fa4b-Abstract-Conference.html)

**Abstract**:

Video action detection requires annotations at every frame, which drastically increases the labeling cost. In this work, we focus on efficient labeling of videos for action detection to minimize this cost. We propose active sparse labeling (ASL), a novel active learning strategy for video action detection. Sparse labeling will reduce the annotation cost but poses two main challenges; 1) how to estimate the utility of annotating a single frame for action detection as detection is performed at video level?, and 2) how these sparse labels can be used for action detection which require annotations on all the frames? This work attempts to address these challenges within a simple active learning framework. For the first challenge, we propose a novel frame-level scoring mechanism aimed at selecting most informative frames in a video. Next, we introduce a novel loss formulation which enables training of action detection model with these sparsely selected frames. We evaluate the proposed approach on two different action detection benchmark datasets, UCF-101-24 and J-HMDB-21, and observed that active sparse labeling can be very effective in saving annotation costs. We demonstrate that the proposed approach performs better than random selection, outperforming all other baselines, with performance comparable to supervised approach using merely 10% annotations.

----

## [1044] Unsupervised Domain Adaptation for Semantic Segmentation using Depth Distribution

**Authors**: *Quanliang Wu, Huajun Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5c882988ce5fac487974ee4f415b96a9-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed significant advancements made in the field of unsupervised domain adaptation for semantic segmentation. Depth information has been proved to be effective in building a bridge between synthetic datasets and real-world datasets. However, the existing methods may not pay enough attention to depth distribution in different categories, which makes it possible to use them for further improvement. Besides the existing methods that only use depth regression as an auxiliary task, we propose to use depth distribution density to support semantic segmentation. Therefore, considering the relationship among depth distribution density, depth and semantic segmentation, we also put forward a branch balance loss for these three subtasks in multi-task learning schemes. In addition, we also propose a spatial aggregation priors of pixels in different categories, which is used to refine the pseudo-labels for self-training, thus further improving the performance of the prediction model. Experiments on SYNTHIA-to-Cityscapes and SYNTHIA-to-Mapillary benchmarks show the effectiveness of our proposed method.

----

## [1045] P2P: Tuning Pre-trained Image Models for Point Cloud Analysis with Point-to-Pixel Prompting

**Authors**: *Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, Jiwen Lu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5cd6dc946ccc37ae6c9f4fc6b6181e1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5cd6dc946ccc37ae6c9f4fc6b6181e1d-Abstract-Conference.html)

**Abstract**:

Nowadays, pre-training big models on large-scale datasets has become a crucial topic in deep learning. The pre-trained models with high representation ability and transferability achieve a great success and dominate many downstream tasks in natural language processing and 2D vision. However, it is non-trivial to promote such a pretraining-tuning paradigm to the 3D vision, given the limited training data that are relatively inconvenient to collect. In this paper, we provide a new perspective of leveraging pre-trained 2D knowledge in 3D domain to tackle this problem, tuning pre-trained image models with the novel Point-to-Pixel prompting for point cloud analysis at a minor parameter cost. Following the principle of prompting engineering, we transform point clouds into colorful images with geometry-preserved projection and geometry-aware coloring to adapt to pre-trained image models, whose weights are kept frozen during the end-to-end optimization of point cloud analysis tasks. We conduct extensive experiments to demonstrate that cooperating with our proposed Point-to-Pixel Prompting, better pre-trained image model will lead to consistently better performance in 3D vision. Enjoying prosperous development from image pre-training field, our method attains 89.3% accuracy on the hardest setting of ScanObjectNN, surpassing conventional point cloud models with much fewer trainable parameters. Our framework also exhibits very competitive performance on ModelNet classification and ShapeNet Part Segmentation. Code is available at https://github.com/wangzy22/P2P.

----

## [1046] Finding Differences Between Transformers and ConvNets Using Counterfactual Simulation Testing

**Authors**: *Nataniel Ruiz, Sarah A. Bargal, Cihang Xie, Kate Saenko, Stan Sclaroff*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ce3a49415f78db65a714b4f05c62f4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ce3a49415f78db65a714b4f05c62f4e-Abstract-Conference.html)

**Abstract**:

Modern deep neural networks tend to be evaluated on static test sets. One shortcoming of this is the fact that these deep neural networks cannot be easily evaluated for robustness issues with respect to specific scene variations. For example, it is hard to study the robustness of these networks to variations of object scale, object pose, scene lighting and 3D occlusions. The main reason is that collecting real datasets with fine-grained naturalistic variations of sufficient scale can be extremely time-consuming and expensive. In this work, we present Counterfactual Simulation Testing, a counterfactual framework that allows us to study the robustness of neural networks with respect to some of these naturalistic variations by building realistic synthetic scenes that allow us to ask counterfactual questions to the models, ultimately providing answers to questions such as "Would your classification still be correct if the object were viewed from the top?" or "Would your classification still be correct if the object were partially occluded by another object?". Our method allows for a fair comparison of the robustness of recently released, state-of-the-art Convolutional Neural Networks and Vision Transformers, with respect to these naturalistic variations. We find evidence that ConvNext is more robust to pose and scale variations than Swin, that ConvNext generalizes better to our simulated domain and that Swin handles partial occlusion better than ConvNext. We also find that robustness for all networks improves with network scale and with data scale and variety. We release the Naturalistic Variation Object Dataset (NVD), a large simulated dataset of 272k images of everyday objects with naturalistic variations such as object pose, scale, viewpoint, lighting and occlusions. Project page: https://counterfactualsimulation.github.io

----

## [1047] In the Eye of the Beholder: Robust Prediction with Causal User Modeling

**Authors**: *Amir Feder, Guy Horowitz, Yoav Wald, Roi Reichart, Nir Rosenfeld*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5cebc89b113920dbff7c79854ba765a3-Abstract-Conference.html)

**Abstract**:

Accurately predicting the relevance of items to users is crucial to the success of many social platforms. Conventional approaches train models on logged historical data; but recommendation systems, media services, and online marketplaces all exhibit a constant influx of new content---making relevancy a moving target, to which standard predictive models are not robust. In this paper, we propose a learning framework for relevance prediction that is robust to changes in the data distribution. Our key observation is that robustness can be obtained by accounting for \emph{how users causally perceive the environment}. We model users as boundedly-rational decision makers whose causal beliefs are encoded by a causal graph, and show how minimal information regarding the graph can be used to contend with distributional changes. Experiments in multiple settings demonstrate the effectiveness of our approach.

----

## [1048] Variational inference via Wasserstein gradient flows

**Authors**: *Marc Lambert, Sinho Chewi, Francis R. Bach, Silvère Bonnabel, Philippe Rigollet*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d087955ee13fe9a7402eedec879b9c3-Abstract-Conference.html)

**Abstract**:

Along with Markov chain Monte Carlo (MCMC) methods, variational inference (VI) has emerged as a central computational approach to large-scale Bayesian inference. Rather than sampling from the true posterior $\pi$, VI aims at producing a simple but effective approximation $\hat \pi$ to $\pi$ for which summary statistics are easy to compute. However, unlike the well-studied MCMC methodology, algorithmic guarantees for VI are still relatively less well-understood. In this work, we propose principled methods for VI, in which $\hat \pi$ is taken to be a Gaussian or a mixture of Gaussians, which rest upon the theory of gradient flows on the Bures--Wasserstein space of Gaussian measures. Akin to MCMC, it comes with strong theoretical guarantees when $\pi$ is log-concave.

----

## [1049] projUNN: efficient method for training deep networks with unitary matrices

**Authors**: *Bobak Toussi Kiani, Randall Balestriero, Yann LeCun, Seth Lloyd*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d1a0188e18c1d74a0f8d6eb5ecede4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d1a0188e18c1d74a0f8d6eb5ecede4f-Abstract-Conference.html)

**Abstract**:

In learning with recurrent or very deep feed-forward networks, employing unitary matrices in each layer can be very effective at maintaining long-range stability. However, restricting network parameters to be unitary typically comes at the cost of expensive parameterizations or increased training runtime. We propose instead an efficient method based on rank-$k$ updates -- or their rank-$k$ approximation -- that maintains performance at a nearly optimal training runtime. We introduce two variants of this method, named Direct (projUNN-D) and Tangent (projUNN-T) projected Unitary Neural Networks, that can parameterize full $N$-dimensional unitary or orthogonal matrices with a training runtime scaling as $O(kN^2)$. Our method either projects low-rank gradients onto the closest unitary matrix (projUNN-T) or transports unitary matrices in the direction of the low-rank gradient (projUNN-D). Even in the fastest setting ($k=1$), projUNN is able to train a model's unitary parameters to reach comparable performances against baseline implementations. In recurrent neural network settings, projUNN closely matches or exceeds benchmarked results from prior unitary neural networks. Finally, we preliminarily explore projUNN in training orthogonal convolutional neural networks, which are currently unable to outperform state of the art models but can potentially enhance stability and robustness at large depth.

----

## [1050] Reinforcement Learning in a Birth and Death Process: Breaking the Dependence on the State Space

**Authors**: *Jonatha Anselmi, Bruno Gaujal, Louis-Sébastien Rebuffi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d2781cc34f459618a9a504761043055-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d2781cc34f459618a9a504761043055-Abstract-Conference.html)

**Abstract**:

In this paper, we revisit  the regret of  undiscounted  reinforcement learning in MDPs  with a birth and death structure. Specifically, we consider a controlled queue  with impatient jobs and the main objective is to optimize a trade-off between energy consumption and user-perceived performance. Within this setting, the diameter $D$ of the MDP is $\Omega(S^S)$, where $S$ is the number of states. Therefore, the existing lower and upper bounds on the regret at time $T$, of  order $O (\sqrt{DSAT})$ for MDPs with $S$ states and $A$ actions, may suggest that reinforcement learning is inefficient here. In our main result however, we exploit the structure of our MDPs to show that the regret of a slightly-tweaked version of the classical learning algorithm UCRL2 is in fact upper bounded by $\tilde{\mathcal{O}} (\sqrt{E_2AT})$ where $E_2$ is a weighted second moment of the stationary measure of a reference policy. Importantly, $E_2$ is bounded independently of $S$. Thus, our bound is asymptotically independent of the number of states and of the diameter. This result is based on a careful study of the number of visits performed by the learning algorithm to the states of the MDP, which is highly non-uniform.

----

## [1051] Multi-dataset Training of Transformers for Robust Action Recognition

**Authors**: *Junwei Liang, Enwei Zhang, Jun Zhang, Chunhua Shen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d2e24df9cfaad3189833b819c40b392-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d2e24df9cfaad3189833b819c40b392-Abstract-Conference.html)

**Abstract**:

We study the task of robust feature representations, aiming to generalize well on multiple datasets for action recognition. We build our method on Transformers for its efficacy. Although we have witnessed great progress for video action recognition in the past decade, it remains challenging yet valuable how to train a single model that can perform well across multiple datasets. Here, we propose a novel multi-dataset training paradigm, MultiTrain, with the design of two new loss terms, namely informative loss and projection loss, aiming tolearn robust representations for action recognition. In particular, the informative loss maximizes the expressiveness of the feature embedding while the projection loss for each dataset mines the intrinsic relations between classes across datasets. We verify the effectiveness of our method on five challenging datasets, Kinetics-400, Kinetics-700, Moments-in-Time, Activitynet and Something-something-v2 datasets. Extensive experimental results show that our method can consistently improve state-of-the-art performance. Code and models are released.

----

## [1052] An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning

**Authors**: *Xiu-Shen Wei, He-Yang Xu, Faen Zhang, Yuxin Peng, Wei Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d3b57e06e3fc45f077eb5c9f28156d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d3b57e06e3fc45f077eb5c9f28156d4-Abstract-Conference.html)

**Abstract**:

Semi-supervised few-shot learning consists in training a classifier to adapt to new tasks with limited labeled data and a fixed quantity of unlabeled data. Many sophisticated methods have been developed to address the challenges this problem comprises. In this paper, we propose a simple but quite effective approach to predict accurate negative pseudo-labels of unlabeled data from an indirect learning perspective, and then augment the extremely label-constrained support set in few-shot classification tasks. Our approach can be implemented in just few lines of code by only using off-the-shelf operations, yet it is able to outperform state-of-the-art methods on four benchmark datasets.

----

## [1053] Recipe for a General, Powerful, Scalable Graph Transformer

**Authors**: *Ladislav Rampásek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html)

**Abstract**:

We propose a recipe on how to build a general, powerful, scalable (GPS) graph Transformer with linear complexity and state-of-the-art results on a diverse set of benchmarks. Graph Transformers (GTs) have gained popularity in the field of graph representation learning with a variety of recent publications but they lack a common foundation about what constitutes a good positional or structural encoding, and what differentiates them. In this paper, we summarize the different types of encodings with a clearer definition and categorize them as being $\textit{local}$, $\textit{global}$ or $\textit{relative}$. The prior GTs are constrained to small graphs with a few hundred nodes, here we propose the first architecture with a complexity linear in the number of nodes and edges $O(N+E)$ by decoupling the local real-edge aggregation from the fully-connected Transformer. We argue that this decoupling does not negatively affect the expressivity, with our architecture being a universal function approximator on graphs. Our GPS recipe consists of choosing 3 main ingredients: (i) positional/structural encoding, (ii) local message-passing mechanism, and (iii) global attention mechanism. We provide a modular framework $\textit{GraphGPS}$ that supports multiple types of encodings and that provides efficiency and scalability both in small and large graphs. We test our architecture on 16 benchmarks and show highly competitive results in all of them, show-casing the empirical benefits gained by the modularity and the combination of different strategies.

----

## [1054] ALIFE: Adaptive Logit Regularizer and Feature Replay for Incremental Semantic Segmentation

**Authors**: *Youngmin Oh, Donghyeon Baek, Bumsub Ham*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d516fc09b53e9a7fade4fbad703e686-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d516fc09b53e9a7fade4fbad703e686-Abstract-Conference.html)

**Abstract**:

We address the problem of incremental semantic segmentation (ISS) recognizing novel object/stuff categories continually without forgetting previous ones that have been learned. The catastrophic forgetting problem is particularly severe in ISS, since pixel-level ground-truth labels are available only for the novel categories at training time. To address the problem, regularization-based methods exploit probability calibration techniques to learn semantic information from unlabeled pixels. While such techniques are effective, there is still a lack of theoretical understanding of them. Replay-based methods propose to memorize a small set of images for previous categories. They achieve state-of-the-art performance at the cost of large memory footprint. We propose in this paper a novel ISS method, dubbed ALIFE, that provides a better compromise between accuracy and efficiency. To this end, we first show an in-depth analysis on the calibration techniques to better understand the effects on ISS. Based on this, we then introduce an adaptive logit regularizer (ALI) that enables our model to better learn new categories, while retaining knowledge for previous ones. We also present a feature replay scheme that memorizes features, instead of images directly, in order to reduce memory requirements significantly. Since a feature extractor is changed continually, memorized features should also be updated at every incremental stage. To handle this, we introduce category-specific rotation matrices updating the features for each category separately. We demonstrate the effectiveness of our approach with extensive experiments on standard ISS benchmarks, and show that our method achieves a better trade-off in terms of accuracy and efficiency.

----

## [1055] Rare Gems: Finding Lottery Tickets at Initialization

**Authors**: *Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grinde, Alliot Nagle, Hongyi Wang, Eric P. Xing, Kangwook Lee, Dimitris S. Papailiopoulos*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d52b102ebd672023628cac20e9da5ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d52b102ebd672023628cac20e9da5ff-Abstract-Conference.html)

**Abstract**:

Large neural networks can be pruned to a small fraction of their original size, with little loss in accuracy, by following a time-consuming "train, prune, re-train" approach. Frankle & Carbin conjecture that we can avoid this by training lottery tickets, i.e., special sparse subnetworks found at initialization, that can be trained to high accuracy. However, a subsequent line of work presents concrete evidence that current algorithms for finding trainable networks at initialization, fail simple baseline comparisons, e.g., against training random sparse subnetworks. Finding lottery tickets that train to better accuracy compared to simple baselines remains an open problem. In this work, we resolve this open problem by proposing Gem-Miner which finds lottery tickets at initialization that beat current baselines. Gem-Miner finds lottery tickets trainable to accuracy competitive or better than Iterative Magnitude Pruning (IMP), and does so up to $19\times$ faster.

----

## [1056] Fast Vision Transformers with HiLo Attention

**Authors**: *Zizheng Pan, Jianfei Cai, Bohan Zhuang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d5f703ee1dedbfe324b1872f44db939-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d5f703ee1dedbfe324b1872f44db939-Abstract-Conference.html)

**Abstract**:

Vision Transformers (ViTs) have triggered the most recent and significant breakthroughs in computer vision. Their efficient designs are mostly guided by the indirect metric of computational complexity, i.e., FLOPs, which however has a clear gap with the direct metric such as throughput. Thus, we propose to use the direct speed evaluation on the target platform as the design principle for efficient ViTs. Particularly, we introduce LITv2, a simple and effective ViT which performs favourably against the existing state-of-the-art methods across a spectrum of different model sizes with faster speed. At the core of LITv2 is a novel self-attention mechanism, which we dub HiLo. HiLo is inspired by the insight that high frequencies in an image capture local fine details and low frequencies focus on global structures, whereas a multi-head self-attention layer neglects the characteristic of different frequencies. Therefore, we propose to disentangle the high/low frequency patterns in an attention layer by separating the heads into two groups, where one group encodes high frequencies via self-attention within each local window, and another group encodes low frequencies by performing global attention between the average-pooled low-frequency keys and values from each window and each query position in the input feature map. Benefiting from the efficient design for both groups, we show that HiLo is superior to the existing attention mechanisms by comprehensively benchmarking FLOPs, speed and memory consumption on GPUs and CPUs. For example, HiLo is 1.4× faster than spatial reduction attention and 1.6× faster than local window attention on CPUs. Powered by HiLo, LITv2 serves as a strong backbone for mainstream vision tasks including image classification, dense detection and segmentation. Code is available at https://github.com/ziplab/LITv2.

----

## [1057] Online Bipartite Matching with Advice: Tight Robustness-Consistency Tradeoffs for the Two-Stage Model

**Authors**: *Billy Jin, Will Ma*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d68a3f05ee2aae6a0fb2d94959082a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d68a3f05ee2aae6a0fb2d94959082a0-Abstract-Conference.html)

**Abstract**:

We study the two-stage vertex-weighted online bipartite matching problem of Feng, Niazadeh, and Saberi (SODA â€˜21) in a setting where the algorithm has access to a suggested matching that is recommended in the first stage. We evaluate an algorithm by its robustness $R$, which is its performance relative to that of the optimal offline matching, and its consistency $C$, which is its performance when the advice or the prediction given is correct.  We characterize for this problem the Pareto-efficient frontier between robustness and consistency, which is rare in the literature on advice-augmented algorithms, yet necessary for quantifying such an algorithm to be optimal. Specifically, we propose an algorithm that is $R$-robust and $C$-consistent for any $(R,C)$ with $0 \leq R \leq \frac{3}{4}$ and $\sqrt{1-R} + \sqrt{1-C} = 1$, and prove that no other algorithm can achieve a better tradeoff.

----

## [1058] Identifying good directions to escape the NTK regime and efficiently learn low-degree plus sparse polynomials

**Authors**: *Eshaan Nichani, Yu Bai, Jason D. Lee*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d6ae8ba43ecb378030753c4408ef9bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d6ae8ba43ecb378030753c4408ef9bd-Abstract-Conference.html)

**Abstract**:

A recent goal in the theory of deep learning is to identify how neural networks can escape the “lazy training,” or Neural Tangent Kernel (NTK) regime, where the network is coupled with its first order Taylor expansion at initialization. While the NTK is minimax optimal for learning dense polynomials (Ghorbani et al, 2021), it cannot learn features, and hence has poor sample complexity for learning many classes of functions including sparse polynomials. Recent works have thus aimed to identify settings where gradient based algorithms provably generalize better than the NTK. One such example is the “QuadNTK” approach of Bai & Lee (2020), which analyzes the second-order term in the Taylor expansion. Bai & Lee (2020) show that the second-order term can learn sparse polynomials efficiently; however, it sacrifices the ability to learn general dense polynomials.In this paper, we analyze how gradient descent on a two-layer neural network can escape the NTK regime by utilizing a spectral characterization of the NTK (Montanari & Zhong, 2020) and building on the QuadNTK approach. We first expand upon the spectral analysis to identify “good” directions in parameter space in which we can move without harming generalization. Next, we show that a wide two-layer neural network can jointly use the NTK and QuadNTK to fit target functions consisting of a dense low-degree term and a sparse high-degree term -- something neither the NTK nor the QuadNTK can do on their own. Finally, we construct a regularizer which encourages the parameter vector to move in the “good" directions, and show that gradient descent on the regularized loss will converge to a global minimizer, which also has low test error. This yields an end to end convergence and generalization guarantee with provable sample complexity improvement over both the NTK and QuadNTK on their own.

----

## [1059] Pure Transformers are Powerful Graph Learners

**Authors**: *Jinwoo Kim, Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee, Seunghoon Hong*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d84236751fe6d25dc06db055a3180b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d84236751fe6d25dc06db055a3180b0-Abstract-Conference.html)

**Abstract**:

We show that standard Transformers without graph-specific modifications can lead to promising results in graph learning both in theory and practice. Given a graph, we simply treat all nodes and edges as independent tokens, augment them with token embeddings, and feed them to a Transformer. With an appropriate choice of token embeddings, we prove that this approach is theoretically at least as expressive as an invariant graph network (2-IGN) composed of equivariant linear layers, which is already more expressive than all message-passing Graph Neural Networks (GNN). When trained on a large-scale graph dataset (PCQM4Mv2), our method coined Tokenized Graph Transformer (TokenGT) achieves significantly better results compared to GNN baselines and competitive results compared to Transformer variants with sophisticated graph-specific inductive bias. Our implementation is available at https://github.com/jw9730/tokengt.

----

## [1060] Orthogonal Transformer: An Efficient Vision Transformer Backbone with Token Orthogonalization

**Authors**: *Huaibo Huang, Xiaoqiang Zhou, Ran He*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5d8c01de2dc698c54201c1c7d0b86974-Abstract-Conference.html)

**Abstract**:

We present a general vision transformer backbone, called as Orthogonal Transformer, in pursuit of both efficiency and effectiveness. A major challenge for vision transformer is that self-attention, as the key element in capturing long-range dependency, is very computationally expensive for dense prediction tasks (e.g., object detection). Coarse global self-attention and local self-attention are then designed to reduce the cost, but they suffer from either neglecting local correlations or hurting global modeling. We present an orthogonal self-attention mechanism to alleviate these issues. Specifically, self-attention is computed in the orthogonal space that is reversible to the spatial domain but has much lower resolution. The capabilities of learning global dependency and exploring local correlations are maintained because every orthogonal token in self-attention can attend to the entire visual tokens. Remarkably, orthogonality is realized by constructing an endogenously orthogonal matrix that is friendly to neural networks and can be optimized as arbitrary orthogonal matrices. We also introduce Positional MLP to incorporate position information for arbitrary input resolutions as well as enhance the capacity of MLP. Finally, we develop a hierarchical architecture for Orthogonal Transformer. Extensive experiments demonstrate its strong performance on a broad range of vision tasks, including image classification, object detection, instance segmentation and semantic segmentation.

----

## [1061] Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & Out-of-Distribution Robustness

**Authors**: *Francesco Pinto, Harry Yang, Ser Nam Lim, Philip H. S. Torr, Puneet K. Dokania*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ddcfaad1cb72ce6f1a365e8f1ecf791-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ddcfaad1cb72ce6f1a365e8f1ecf791-Abstract-Conference.html)

**Abstract**:

We show that the effectiveness of the well celebrated Mixup can be further improved if instead of using it as the sole learning objective, it is utilized as an additional regularizer to the standard cross-entropy loss. This simple change not only improves accuracy but also significantly improves the quality of the predictive uncertainty estimation of Mixup in most cases under various forms of covariate shifts and out-of-distribution detection experiments. In fact, we observe that Mixup otherwise yields much degraded performance on detecting out-of-distribution samples possibly, as we show empirically, due to its tendency to learn models exhibiting high-entropy throughout; making it difficult to differentiate in-distribution samples from out-of-distribution ones. To show the efficacy of our approach (RegMixup), we provide thorough analyses and experiments on vision datasets (ImageNet & CIFAR-10/100) and compare it with a suite of recent approaches for reliable uncertainty estimation.

----

## [1062] NeurOLight: A Physics-Agnostic Neural Operator Enabling Parametric Photonic Device Simulation

**Authors**: *Jiaqi Gu, Zhengqi Gao, Chenghao Feng, Hanqing Zhu, Ray T. Chen, Duane S. Boning, David Z. Pan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ddfb189c022a317ff1c72e6639079de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ddfb189c022a317ff1c72e6639079de-Abstract-Conference.html)

**Abstract**:

Optical computing has become emerging technology in next-generation efficient artificial intelligence (AI) due to its ultra-high speed and efficiency. Electromagnetic field simulation is critical to the design, optimization, and validation of photonic devices and circuits.However, costly numerical simulation significantly hinders the scalability and turn-around time in the photonic circuit design loop. Recently, physics-informed neural networks were proposed to predict the optical field solution of a single instance of a partial differential equation (PDE) with predefined parameters. Their complicated PDE formulation and lack of efficient parametrization mechanism limit their flexibility and generalization in practical simulation scenarios. In this work, for the first time, a physics-agnostic neural operator-based framework, dubbed NeurOLight, is proposed to learn a family of frequency-domain Maxwell PDEs for ultra-fast parametric photonic device simulation. Specifically, we discretize different devices into a unified domain, represent parametric PDEs with a compact wave prior, and encode the incident light via masked source modeling. We design our model to have parameter-efficient cross-shaped NeurOLight blocks and adopt superposition-based augmentation for data-efficient learning. With those synergistic approaches, NeurOLight demonstrates 2-orders-of-magnitude faster simulation speed than numerical solvers and outperforms prior NN-based models by ~54% lower prediction error using ~44% fewer parameters.

----

## [1063] Learning the Structure of Large Networked Systems Obeying Conservation Laws

**Authors**: *Anirudh Rayas, Rajasekhar Anguluri, Gautam Dasarathy*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e0347e19c51cfd0f6fe52f371004dfc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e0347e19c51cfd0f6fe52f371004dfc-Abstract-Conference.html)

**Abstract**:

Many networked systems such as electric networks, the brain, and social networks of opinion dynamics are known to obey conservation laws. Examples of this phenomenon include the Kirchoff laws in electric networks and opinion consensus in social networks. Conservation laws in networked systems are modeled as balance equations of the form $X = B^\ast Y$, where the sparsity pattern of $B^\ast \in \mathbb{R}^{p\times p}$ captures the connectivity of the network on $p$ nodes, and  $Y, X \in \mathbb{R}^p$ are vectors of ''potentials'' and ''injected flows'' at the nodes respectively. The node potentials $Y$ cause flows across edges which aim to balance out the potential difference, and the flows $X$ injected at the nodes are extraneous to the network dynamics. In several practical systems, the network structure is often unknown and needs to be estimated from data to facilitate modeling, management, and control. To this end, one has access to samples of the node potentials $Y$, but only the statistics of the node injections $X$. Motivated by this important problem, we study the estimation of the sparsity structure of the matrix $B^\ast$ from $n$ samples of $Y$ under the assumption that the node injections $X$ follow a Gaussian distribution with a known covariance $\Sigma_X$. We propose a new $\ell_{1}$-regularized maximum likelihood estimator for tackling this problem in the high-dimensional regime where the size of the network may be vastly larger than the number of samples $n$. We show that this optimization problem is convex in the objective and admits a unique solution. Under a new mutual incoherence condition, we establish sufficient conditions on the triple $(n,p,d)$ for which exact sparsity recovery of $B^\ast$ is possible with high probability; $d$ is the degree of the underlying graph. We also establish guarantees for the recovery of $B^\ast$ in the element-wise maximum, Frobenius, and operator norms. Finally, we complement these theoretical results with experimental validation of the performance of the proposed estimator on synthetic and real-world data.

----

## [1064] FP8 Quantization: The Power of the Exponent

**Authors**: *Andrey Kuzmin, Mart van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, Tijmen Blankevoort*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e07476b6bd2497e1fbd11b8f0b2de3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e07476b6bd2497e1fbd11b8f0b2de3c-Abstract-Conference.html)

**Abstract**:

When quantizing neural networks for efficient inference, low-bit integers are the go-to format for efficiency. However, low-bit floating point numbers have an extra degree of freedom, assigning some bits to work on an exponential scale instead. This paper in-depth investigates this benefit of the floating point format for neural network inference. We detail the choices that can be made for the FP8 format, including the important choice of the number of bits for the mantissa and exponent, and show analytically in which settings these choices give better performance. Then we show how these findings translate to real networks, provide an efficient implementation for FP8 simulation, and a new algorithm that enables the learning of both the scale parameters and number of exponent bits in the FP8 format. Our chief conclusion is that when doing post-training quantization for a wide range of networks, the FP8 format is better than INT8 in terms of accuracy, and the choice of the number of exponent bits is driven by the severity of outliers in the network. We also conduct experiments with quantization-aware training where the difference in formats disappears as the network is trained to reduce the effect of outliers.

----

## [1065] Bridging the Gap Between Vision Transformers and Convolutional Neural Networks on Small Datasets

**Authors**: *Zhiying Lu, Hongtao Xie, Chuanbin Liu, Yongdong Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e0b46975d1bfe6030b1687b0ada1b85-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e0b46975d1bfe6030b1687b0ada1b85-Abstract-Conference.html)

**Abstract**:

There still remains an extreme performance gap between Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) when training from scratch on small datasets, which is concluded to the lack of inductive bias. In this paper, we further consider this problem and point out two weaknesses of ViTs in inductive biases, that is, the spatial relevance and diverse channel representation. First, on spatial aspect, objects are locally compact and relevant, thus fine-grained feature needs to be extracted from a token and its neighbors. While the lack of data hinders ViTs to attend the spatial relevance. Second, on channel aspect, representation exhibits diversity on different channels. But the scarce data can not enable ViTs to learn strong enough representation for accurate recognition. To this end, we propose Dynamic Hybrid Vision Transformer (DHVT) as the solution to enhance the two inductive biases. On spatial aspect, we adopt a hybrid structure, in which convolution is integrated into patch embedding and multi-layer perceptron module, forcing the model to capture the token features as well as their neighboring features. On channel aspect, we introduce a dynamic feature aggregation module in MLP and a brand new "head token" design in multi-head self-attention module to help re-calibrate channel representation and make different channel group representation interacts with each other. The fusion of weak channel representation forms a strong enough representation for classification. With this design, we successfully eliminate the performance gap between CNNs and ViTs, and our DHVT achieves a series of state-of-the-art performance with a lightweight model, 85.68% on CIFAR-100 with 22.8M parameters, 82.3% on ImageNet-1K with 24.0M parameters. Code is available at https://github.com/ArieSeirack/DHVT.

----

## [1066] Private Set Generation with Discriminative Information

**Authors**: *Dingfan Chen, Raouf Kerkouche, Mario Fritz*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e1a87dbb7e954b8d9d6c91f6db771eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e1a87dbb7e954b8d9d6c91f6db771eb-Abstract-Conference.html)

**Abstract**:

Differentially private data generation techniques have become a promising solution to the data privacy challenge –– it enables sharing of data while complying with rigorous privacy guarantees, which is essential for scientific progress in sensitive domains. Unfortunately, restricted by the inherent complexity of modeling high-dimensional distributions, existing private generative models are struggling with the utility of synthetic samples. In contrast to existing works that aim at fitting the complete data distribution, we directly optimize for a small set of samples that are representative of the distribution, which is generally an easier task and more suitable for private training. Moreover, we exploit discriminative information from downstream tasks to further ease the training. Our work provides an alternative view for differentially private generation of high-dimensional data and introduces a simple yet effective method that greatly improves the sample utility of state-of-the-art approaches.

----

## [1067] The First Optimal Algorithm for Smooth and Strongly-Convex-Strongly-Concave Minimax Optimization

**Authors**: *Dmitry Kovalev, Alexander V. Gasnikov*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e2ed801f62102f531d109d7c6e1b62f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e2ed801f62102f531d109d7c6e1b62f-Abstract-Conference.html)

**Abstract**:

In this paper, we revisit the smooth and strongly-convex-strongly-concave minimax optimization problem. Zhang et al. (2021) and Ibrahim et al. (2020) established the lower bound $\Omega\left(\sqrt{\kappa_x\kappa_y} \log \frac{1}{\epsilon}\right)$ on the number of gradient evaluations required to find an ϵ-accurate solution, where κx and κy are condition numbers for the strong convexity and strong concavity assumptions. However, the existing state-of-the-art methods do not match this lower bound: algorithms of Lin et al. (2020) and Wang and Li (2020) have gradient evaluation complexity $\mathcal{O}\left(\sqrt{\kappa_x\kappa_y} \log^3 \frac{1}{\epsilon}\right)$ and $\mathcal{O}\left( \sqrt{\kappa_x\kappa_y}\log^3 (\kappa_x\kappa_y)\log\frac{1}{\epsilon}\right)$, respectively. We fix this fundamental issue by providing the first algorithm with $\mathcal{O}\left(\sqrt{\kappa_x\kappa_y} \log \frac{1}{\epsilon}\right)$ gradient evaluation complexity. We design our algorithm in three steps: (i) we reformulate the original problem as a minimization problem via the pointwise conjugate function; (ii) we apply a specific variant of the proximal point algorithm to the reformulated problem; (iii) we compute the proximal operator inexactly using the optimal algorithm for operator norm reduction in monotone inclusions.

----

## [1068] Provable Defense against Backdoor Policies in Reinforcement Learning

**Authors**: *Shubham Kumar Bharti, Xuezhou Zhang, Adish Singla, Jerry Zhu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e67e6a814526079ad8505bf6d926fb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e67e6a814526079ad8505bf6d926fb6-Abstract-Conference.html)

**Abstract**:

We propose a provable defense mechanism against backdoor policies in reinforcement learning under subspace trigger assumption. A backdoor policy is a security threat where an adversary publishes a seemingly well-behaved policy which in fact allows hidden triggers. During deployment, the adversary can modify observed states in a particular way to trigger unexpected actions and harm the agent. We assume the agent does not have the resources to re-train a good policy. Instead, our defense mechanism sanitizes the backdoor policy by projecting observed states to a `safe subspace', estimated from a small number of interactions with a clean (non-triggered) environment. Our sanitized policy achieves $\epsilon$ approximate optimality in the presence of triggers, provided the number of clean interactions is $O\left(\frac{D}{(1-\gamma)^4 \epsilon^2}\right)$ where $\gamma$ is the discounting factor and $D$ is the dimension of state space. Empirically, we show that our sanitization defense performs well on two Atari game environments.

----

## [1069] Diffusion Models as Plug-and-Play Priors

**Authors**: *Alexandros Graikos, Nikolay Malkin, Nebojsa Jojic, Dimitris Samaras*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e6cec2a9520708381fe520246018e8b-Abstract-Conference.html)

**Abstract**:

We consider the problem of inferring high-dimensional data $x$ in a model that consists of a prior $p(x)$ and an auxiliary differentiable constraint $c(x,y)$ on $x$ given some additional information $y$. In this paper, the prior is an independently trained denoising diffusion generative model. The auxiliary constraint is expected to have a differentiable form, but can come from diverse sources. The possibility of such inference turns diffusion models into plug-and-play modules, thereby allowing a range of potential applications in adapting models to new domains and tasks, such as conditional generation or image segmentation. The structure of diffusion models allows us to perform approximate inference by iterating differentiation through the fixed denoising network enriched with different amounts of noise at each step. Considering many noised versions of $x$ in evaluation of its fitness is a novel search mechanism that may lead to new algorithms for solving combinatorial optimization problems. The code is available at https://github.com/AlexGraikos/diffusion_priors.

----

## [1070] CoupAlign: Coupling Word-Pixel with Sentence-Mask Alignments for Referring Image Segmentation

**Authors**: *Zicheng Zhang, Yi Zhu, Jianzhuang Liu, Xiaodan Liang, Wei Ke*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e773d319e310f1e4d695159484143b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e773d319e310f1e4d695159484143b8-Abstract-Conference.html)

**Abstract**:

Referring image segmentation aims at localizing all pixels of the visual objects described by a natural language sentence. Previous works learn to straightforwardly align the sentence embedding and pixel-level embedding for highlighting the referred objects, but ignore the semantic consistency of pixels within the same object, leading to incomplete masks and localization errors in predictions. To tackle this problem, we propose CoupAlign, a simple yet effective multi-level visual-semantic alignment method, to couple sentence-mask alignment with word-pixel alignment to enforce object mask constraint for achieving more accurate localization and segmentation. Specifically, the Word-Pixel Alignment (WPA) module performs early fusion of linguistic and pixel-level features in intermediate layers of the vision and language encoders. Based on the word-pixel aligned embedding, a set of mask proposals are generated to hypothesize possible objects. Then in the Sentence-Mask Alignment (SMA) module, the masks are weighted by the sentence embedding to localize the referred object, and finally projected back to aggregate the pixels for the target. To further enhance the learning of the two alignment modules, an auxiliary loss is designed to contrast the foreground and background pixels. By hierarchically aligning pixels and masks with linguistic features, our CoupAlign captures the pixel coherence at both visual and semantic levels, thus generating more accurate predictions. Extensive experiments on popular datasets (e.g., RefCOCO and G-Ref) show that our method achieves consistent improvements over state-of-the-art methods, e.g., about 2% oIoU increase on the validation and testing set of RefCOCO. Especially, CoupAlign has remarkable ability in distinguishing the target from multiple objects of the same class. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/CoupAlign.

----

## [1071] Saliency-Aware Neural Architecture Search

**Authors**: *Ramtin Hosseini, Pengtao Xie*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e84e4413268b713f0d4a1b23a9dae57-Abstract-Conference.html)

**Abstract**:

Recently a wide variety of NAS methods have been proposed and achieved considerable success in automatically identifying highly-performing architectures of neural networks for the sake of reducing the reliance on human experts. Existing NAS methods ignore the fact that different input data elements (e.g., image pixels) have different importance (or saliency) in determining the prediction outcome. They treat all data elements as being equally important and therefore lead to suboptimal performance. To address this problem, we propose an end-to-end framework which dynamically detects saliency of input data, reweights data using saliency maps, and searches  architectures on saliency-reweighted data. Our framework is based on four-level optimization, which performs four learning stages in a unified way. At the first stage, a model is trained with its architecture tentatively fixed. At the second stage, saliency maps are generated using the trained model. At the third stage, the model is retrained on saliency-reweighted data. At the fourth stage, the model is evaluated on a validation set and the architecture is updated by minimizing the validation loss. Experiments on several datasets demonstrate the effectiveness of our framework.

----

## [1072] VaiPhy: a Variational Inference Based Algorithm for Phylogeny

**Authors**: *Hazal Koptagel, Oskar Kviman, Harald Melin, Negar Safinianaini, Jens Lagergren*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5e956fef0946dc1e39760f94b78045fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5e956fef0946dc1e39760f94b78045fe-Abstract-Conference.html)

**Abstract**:

Phylogenetics is a classical methodology in computational biology that today has become highly relevant for medical investigation of single-cell data, e.g., in the context of development of cancer.  The exponential size of the tree space is unfortunately a formidable obstacle for current Bayesian phylogenetic inference using Markov chain Monte Carlo based methods since these rely on local operations. And although more recent variational inference (VI) based methods offer speed improvements, they rely on expensive auto-differentiation operations for learning the variational parameters. We propose VaiPhy, a remarkably fast VI based algorithm for approximate posterior inference in an \textit{augmented tree space}. VaiPhy produces marginal log-likelihood estimates on par with the state-of-the-art methods on real data, and is considerably faster since it does not require auto-differentiation. Instead, VaiPhy combines coordinate ascent update equations with two novel sampling schemes: (i) \textit{SLANTIS}, a proposal distribution for tree topologies in the augmented tree space, and (ii) the \textit{JC sampler}, the, to the best of our knowledge, first ever scheme for sampling branch lengths directly from the popular Jukes-Cantor model. We compare VaiPhy in terms of density estimation and runtime. Additionally, we evaluate the reproducibility of the baselines. We provide our code on GitHub: \url{https://github.com/Lagergren-Lab/VaiPhy}.

----

## [1073] A simple but strong baseline for online continual learning: Repeated Augmented Rehearsal

**Authors**: *Yaqian Zhang, Bernhard Pfahringer, Eibe Frank, Albert Bifet, Nick Jin Sean Lim, Yunzhe Jia*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ebbbac62b968254093023f1c95015d3-Abstract-Conference.html)

**Abstract**:

Online continual learning (OCL) aims to train neural networks incrementally from a non-stationary data stream with a single pass through data. Rehearsal-based methods attempt to approximate the observed input distributions over time with a small memory and revisit them later to avoid forgetting. Despite their strong empirical performance, rehearsal methods still suffer from a poor approximation of past dataâ€™s loss landscape with memory samples. This paper revisits the rehearsal dynamics in online settings. We provide theoretical insights on the inherent memory overfitting risk from the viewpoint of biased and dynamic empirical risk minimization, and examine the merits and limits of repeated rehearsal.Inspired by our analysis, a simple and intuitive baseline, repeated augmented rehearsal (RAR), is designed to address the underfitting-overfitting dilemma of online rehearsal. Surprisingly, across four rather different OCL benchmarks,this simple baseline outperforms vanilla rehearsal by  9\%-17\% and also significantly improves the state-of-the-art rehearsal-based methods MIR, ASER, and SCR. We also demonstrate that RAR successfully achieves an accurate approximation of the loss landscape of past data and high-loss ridge aversion in its learning trajectory. Extensive ablation studies are conducted to study the interplay between repeated and augmented rehearsal, and reinforcement learning (RL) is applied to dynamically adjust the hyperparameters of RAR to balance the stability-plasticity trade-off online.

----

## [1074] You Only Live Once: Single-Life Reinforcement Learning

**Authors**: *Annie S. Chen, Archit Sharma, Sergey Levine, Chelsea Finn*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ec4e93f2cec19d47ef852a0e1fb2c48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ec4e93f2cec19d47ef852a0e1fb2c48-Abstract-Conference.html)

**Abstract**:

Reinforcement learning algorithms are typically designed to learn a performant policy that can repeatedly and autonomously complete a task, usually starting from scratch. However, in many real-world situations, the goal might not be to learn a policy that can do the task repeatedly, but simply to perform a new task successfully once in a single trial. For example, imagine a disaster relief robot tasked with retrieving an item from a fallen building, where it cannot get direct supervision from humans. It must retrieve this  object within one test-time trial, and must do so while tackling unknown obstacles, though it may leverage knowledge it has of the building before the disaster. We formalize this problem setting, which we call single-life reinforcement learning (SLRL), where an agent must complete a task within a single episode without interventions, utilizing its prior experience while contending with some form of novelty. SLRL provides a natural setting to study the challenge of autonomously adapting to unfamiliar situations, and we find that algorithms designed for standard episodic reinforcement learning often struggle to recover from out-of-distribution states in this setting. Motivated by this observation, we propose an algorithm, Q-weighted adversarial learning (QWALE), which employs a distribution matching strategy that leverages the agent's prior experience as guidance in novel situations. Our experiments on several single-life continuous control problems indicate that methods based on our distribution matching formulation are 20-60% more successful because they can more quickly recover from novel states.

----

## [1075] Compressible-composable NeRF via Rank-residual Decomposition

**Authors**: *Jiaxiang Tang, Xiaokang Chen, Jingbo Wang, Gang Zeng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ed5c3c846f684a54975ad7a2525199f-Abstract-Conference.html)

**Abstract**:

Neural Radiance Field (NeRF) has emerged as a compelling method to represent 3D objects and scenes for photo-realistic rendering. However, its implicit representation causes difficulty in manipulating the models like the explicit mesh representation.Several recent advances in NeRF manipulation are usually restricted by a shared renderer network, or suffer from large model size. To circumvent the hurdle, in this paper, we present a neural field representation that enables efficient and convenient manipulation of models.To achieve this goal, we learn a hybrid tensor rank decomposition of the scene without neural networks. Motivated by the low-rank approximation property of the SVD algorithm, we propose a rank-residual learning strategy to encourage the preservation of primary information in lower ranks. The model size can then be dynamically adjusted by rank truncation to control the levels of detail, achieving near-optimal compression without extra optimization.Furthermore, different models can be arbitrarily transformed and composed into one scene by concatenating along the rank dimension.The growth of storage cost can also be mitigated by compressing the unimportant objects in the composed scene. We demonstrate that our method is able to achieve comparable rendering quality to state-of-the-art methods, while enabling extra capability of compression and composition.Code is available at https://github.com/ashawkey/CCNeRF.

----

## [1076] Data-Efficient Pipeline for Offline Reinforcement Learning with Limited Data

**Authors**: *Allen Nie, Yannis Flet-Berliac, Deon R. Jordan, William Steenbergen, Emma Brunskill*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ee7ed60a7e8169012224dec5fe0d27f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ee7ed60a7e8169012224dec5fe0d27f-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) can be used to improve future performance by leveraging historical data. There exist many different algorithms for offline RL, and it is well recognized that these algorithms, and their hyperparameter settings, can lead to decision policies with substantially differing performance. This prompts the need for pipelines that allow practitioners to systematically perform algorithm-hyperparameter selection for their setting. Critically, in most real-world settings, this pipeline must only involve the use of historical data. Inspired by statistical model selection methods for supervised learning, we introduce a task- and method-agnostic pipeline for automatically training, comparing, selecting, and deploying the best policy when the provided dataset is limited in size. In particular, our work highlights the importance of performing multiple data splits to produce more reliable algorithm-hyperparameter selection. While this is a common approach in supervised learning, to our knowledge, this has not been discussed in detail in the offline RL setting. We show it can have substantial impacts when the dataset is small. Compared to alternate approaches, our proposed pipeline outputs higher-performing deployed policies from a broad range of offline policy learning algorithms and across various simulation domains in healthcare, education, and robotics. This work contributes toward the development of a general-purpose meta-algorithm for automatic algorithm-hyperparameter selection for offline RL.

----

## [1077] Hardness in Markov Decision Processes: Theory and Practice

**Authors**: *Michelangelo Conserva, Paulo E. Rauber*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5eeb693f46d753e5fe24c97212c22bd2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5eeb693f46d753e5fe24c97212c22bd2-Abstract-Conference.html)

**Abstract**:

Meticulously analysing the empirical strengths and weaknesses of reinforcement learning methods in hard (challenging) environments is essential to inspire innovations and assess progress in the field. In tabular reinforcement learning, there is no well-established standard selection of environments to conduct such analysis, which is partially due to the lack of a widespread understanding of the rich theory of hardness of environments. The goal of this paper is to unlock the practical usefulness of this theory through four main contributions. First, we present a systematic survey of the theory of hardness, which also identifies promising research directions. Second, we introduce $\texttt{Colosseum}$, a pioneering package that enables empirical hardness analysis and implements a principled benchmark composed of environments that are diverse with respect to different measures of hardness. Third, we present an empirical analysis that provides new insights into computable measures. Finally, we benchmark five tabular agents in our newly proposed benchmark. While advancing the theoretical understanding of hardness in non-tabular reinforcement learning remains essential, our contributions in the tabular setting are intended as solid steps towards a principled non-tabular benchmark. Accordingly, we benchmark four agents in non-tabular versions of $\texttt{Colosseum}$ environments, obtaining results that demonstrate the generality of tabular hardness measures.

----

## [1078] Injecting Domain Knowledge from Empirical Interatomic Potentials to Neural Networks for Predicting Material Properties

**Authors**: *Zeren Shui, Daniel S. Karls, Mingjian Wen, Ilia A. Nikiforov, Ellad B. Tadmor, George Karypis*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5ef1df239d6640a27dd6ed9a59f518c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5ef1df239d6640a27dd6ed9a59f518c9-Abstract-Conference.html)

**Abstract**:

For decades, atomistic modeling has played a crucial role in predicting the behavior of materials in numerous fields ranging from nanotechnology to drug discovery. The most accurate methods in this domain are rooted in first-principles quantum mechanical calculations such as density functional theory (DFT). Because these methods have remained computationally prohibitive, practitioners have traditionally focused on defining physically motivated closed-form expressions known as empirical interatomic potentials (EIPs) that approximately model the interactions between atoms in materials. In recent years, neural network (NN)-based potentials trained on quantum mechanical (DFT-labeled) data have emerged as a more accurate alternative to conventional EIPs. However, the generalizability of these models relies heavily on the amount of labeled training data, which is often still insufficient to generate models suitable for general-purpose applications. In this paper, we propose two generic strategies that take advantage of unlabeled training instances to inject domain knowledge from conventional EIPs to NNs in order to increase their generalizability. The first strategy, based on weakly supervised learning, trains an auxiliary classifier on EIPs and selects the best-performing EIP to generate energies to supplement the ground-truth DFT energies in training the NN. The second strategy, based on transfer learning, first pretrains the NN on a large set of easily obtainable EIP energies, and then fine-tunes it on ground-truth DFT energies. Experimental results on three benchmark datasets demonstrate that the first strategy improves baseline NN performance by 5% to 51% while the second improves baseline performance by up to 55%. Combining them further boosts performance.

----

## [1079] Learning Modular Simulations for Homogeneous Systems

**Authors**: *Jayesh K. Gupta, Sai Vemprala, Ashish Kapoor*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html)

**Abstract**:

Complex systems are often decomposed into modular subsystems for engineering tractability. Although various equation based white-box modeling techniques make use of such structure, learning based methods have yet to incorporate these ideas broadly. We present a modular simulation framework for modeling homogeneous multibody dynamical systems, which combines ideas from graph neural networks and neural differential equations. We learn to model the individual dynamical subsystem as a neural ODE module. Full simulation of the composite system is orchestrated via spatio-temporal message passing between these modules. An arbitrary number of modules can be combined to simulate systems of a wide variety of coupling topologies. We evaluate our framework on a variety of systems and show that message passing allows coordination between multiple modules over time for accurate predictions and in certain cases, enables zero-shot generalization to new system configurations. Furthermore, we show that our models can be transferred to new system configurations with lower data requirement and training effort, compared to those trained from scratch.

----

## [1080] On Gap-dependent Bounds for Offline Reinforcement Learning

**Authors**: *Xinqi Wang, Qiwen Cui, Simon S. Du*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f5f7b6080dcadced61cf5d96f7c6dde-Abstract-Conference.html)

**Abstract**:

This paper presents a systematic study on gap-dependent sample complexity in offline reinforcement learning. Prior works showed when the density ratio between an optimal policy and the behavior policy is upper bounded (single policy coverage), then the agent can achieve an $O\left(\frac{1}{\epsilon^2}\right)$ rate, which is also minimax optimal. We show under the same single policy coverage assumption, the rate can be improved to $O\left(\frac{1}{\epsilon}\right)$ when there is a gap in the optimal $Q$-function. Furthermore, we show under a stronger uniform single policy coverage assumption, the sample complexity can be further improved to $O(1)$. Lastly, we also present nearly-matching lower bounds to complement our gap-dependent upper bounds.

----

## [1081] Semi-Discrete Normalizing Flows through Differentiable Tessellation

**Authors**: *Ricky T. Q. Chen, Brandon Amos, Maximilian Nickel*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f61939af1699c82dab00ed36c887968-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f61939af1699c82dab00ed36c887968-Abstract-Conference.html)

**Abstract**:

Mapping between discrete and continuous distributions is a difficult task and many have had to resort to heuristical approaches. We propose a tessellation-based approach that directly learns quantization boundaries in a continuous space, complete with exact likelihood evaluations. This is done through constructing normalizing flows on convex polytopes parameterized using a simple homeomorphism with an efficient log determinant Jacobian. We explore this approach in two application settings, mapping from discrete to continuous and vice versa. Firstly, a Voronoi dequantization allows automatically learning quantization boundaries in a multidimensional space. The location of boundaries and distances between regions can encode useful structural relations between the quantized discrete values. Secondly, a Voronoi mixture model has near-constant computation cost for likelihood evaluation regardless of the number of mixture components. Empirically, we show improvements over existing methods across a range of structured data modalities.

----

## [1082] Don't Pour Cereal into Coffee: Differentiable Temporal Logic for Temporal Action Segmentation

**Authors**: *Ziwei Xu, Yogesh S. Rawat, Yongkang Wong, Mohan S. Kankanhalli, Mubarak Shah*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f96a21345c138da929e99871fda138e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f96a21345c138da929e99871fda138e-Abstract-Conference.html)

**Abstract**:

We propose Differentiable Temporal Logic (DTL), a model-agnostic framework that introduces temporal constraints to deep networks. DTL treats the outputs of a network as a truth assignment of a temporal logic formula, and computes a temporal logic loss reflecting the consistency between the output and the constraints. We propose a comprehensive set of constraints, which are implicit in data annotations, and incorporate them with deep networks via DTL. We evaluate the effectiveness of DTL on the temporal action segmentation task and observe improved performance and reduced logical errors in the output of different task models. Furthermore, we provide an extensive analysis to visualize the desirable effects of DTL.

----

## [1083] Batch-Size Independent Regret Bounds for Combinatorial Semi-Bandits with Probabilistically Triggered Arms or Independent Arms

**Authors**: *Xutong Liu, Jinhang Zuo, Siwei Wang, Carlee Joe-Wong, John C. S. Lui, Wei Chen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f999db36b107f044089247bb41dbd90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f999db36b107f044089247bb41dbd90-Abstract-Conference.html)

**Abstract**:

In this paper, we study the combinatorial semi-bandits (CMAB) and focus on reducing the dependency of the batch-size $K$ in the regret bound, where $K$ is the total number of arms that can be pulled or triggered in each round. First, for the setting of CMAB with probabilistically triggered arms (CMAB-T), we discover a novel (directional) triggering probability and variance modulated (TPVM) condition that can replace the previously-used smoothness condition for various applications, such as cascading bandits, online network exploration and online influence maximization. Under this new condition, we propose a BCUCB-T algorithm with variance-aware confidence intervals and conduct regret analysis which reduces the $O(K)$ factor to $O(\log K)$ or $O(\log^2 K)$ in the regret bound, significantly improving the regret bounds for the above applications. Second, for the setting of non-triggering CMAB with independent arms, we propose a SESCB algorithm which leverages on the non-triggering version of the TPVM condition and completely removes the dependency on $K$ in the leading regret. As a valuable by-product, the regret analysis used in this paper can improve several existing results by a factor of $O(\log K)$. Finally, experimental evaluations show our superior performance compared with benchmark algorithms in different applications.

----

## [1084] Less-forgetting Multi-lingual Fine-tuning

**Authors**: *Yuren Mao, Yaobo Liang, Nan Duan, Haobo Wang, Kai Wang, Lu Chen, Yunjun Gao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5f9f9e4da57a94547491a39dc18f1696-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5f9f9e4da57a94547491a39dc18f1696-Abstract-Conference.html)

**Abstract**:

Multi-lingual fine-tuning (MLF), which fine-tunes a multi-lingual language model (MLLM) with multiple source languages, aims to gain good zero-shot performance on target languages. In MLF, the fine-tuned model tends to fit the source languages while forgetting its cross-lingual knowledge obtained from the pre-training stage. This forgetting phenomenon degenerates the zero-shot performance of MLF, which remains under-explored. To fill this gap, this paper proposes a multi-lingual fine-tuning method, dubbed Less-forgetting Multi-lingual Fine-tuning (LF-MLF). In LF-MLF, we cast multi-lingual fine-tuning as a constrained optimization problem, where the optimization objective is to minimize forgetting, and constraints are reducing the fine-tuning loss. The proposed method has superior zero-shot performance; furthermore, it can achieve the Pareto stationarity. Extensive experiments on Named Entity Recognition, Question Answering and Natural Language Inference back up our theoretical analysis and validate the superiority of our proposals.

----

## [1085] Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks

**Authors**: *Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5fa29a2f163ce2020769eca8956e2d77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5fa29a2f163ce2020769eca8956e2d77-Abstract-Conference.html)

**Abstract**:

The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure 80.59% accuracy under Square attack (2500 queries), while the best prior defense (i.e., adversarial training) only attains 67.44%. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets, bounds, norms, losses, and strategies. Moreover, AAA calibrates better without hurting the accuracy. Our code is available at https://github.com/Sizhe-Chen/AAA.

----

## [1086] Retaining Knowledge for Learning with Dynamic Definition

**Authors**: *Zichang Liu, Benjamin Coleman, Tianyi Zhang, Anshumali Shrivastava*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/5fcd540792da599adf1b932624e98f1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/5fcd540792da599adf1b932624e98f1f-Abstract-Conference.html)

**Abstract**:

Machine learning models are often deployed in settings where they must be constantly updated in response to the changes in class definitions while retaining high accuracy on previously learned definitions. A classical use case is fraud detection, where new fraud schemes come one after another. While such an update can be accomplished by re-training on the complete data, the process is inefficient and prevents real-time and on-device learning. On the other hand, efficient methods that incrementally learn from new data often result in the forgetting of previously-learned knowledge. We define this problem as Learning with Dynamic Definition (LDD) and demonstrate that popular models, such as the Vision Transformer and Roberta, exhibit substantial forgetting of past definitions.  We present the first practical and provable solution to LDD. Our proposal is a hash-based sparsity model \textit{RIDDLE} that solves evolving definitions by associating samples only to relevant parameters. We prove that our model is a universal function approximator and theoretically bounds the knowledge lost during the update process. On practical tasks with evolving class definition in vision and natural language processing, \textit{RIDDLE} outperforms baselines by up to 30\% on the original dataset while providing competitive accuracy on the update dataset.

----

## [1087] HUMANISE: Language-conditioned Human Motion Generation in 3D Scenes

**Authors**: *Zan Wang, Yixin Chen, Tengyu Liu, Yixin Zhu, Wei Liang, Siyuan Huang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6030db5195150ac86d942186f4abdad8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6030db5195150ac86d942186f4abdad8-Abstract-Conference.html)

**Abstract**:

Learning to generate diverse scene-aware and goal-oriented human motions in 3D scenes remains challenging due to the mediocre characters of the existing datasets on Human-Scene Interaction (HSI); they only have limited scale/quality and lack semantics. To fill in the gap, we propose a large-scale and semantic-rich synthetic HSI dataset, denoted as HUMANISE, by aligning the captured human motion sequences with various 3D indoor scenes. We automatically annotate the aligned motions with language descriptions that depict the action and the individual interacting objects; e.g., sit on the armchair near the desk. HUMANIZE thus enables a new generation task, language-conditioned human motion generation in 3D scenes. The proposed task is challenging as it requires joint modeling of the 3D scene, human motion, and natural language. To tackle this task, we present a novel scene-and-language conditioned generative model that can produce 3D human motions of the desirable action interacting with the specified objects. Our experiments demonstrate that our model generates diverse and semantically consistent human motions in 3D scenes.

----

## [1088] Keypoint-Guided Optimal Transport with Applications in Heterogeneous Domain Adaptation

**Authors**: *Xiang Gu, Yucheng Yang, Wei Zeng, Jian Sun, Zongben Xu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6091c5644d73637e3cccdcab52a7031f-Abstract-Conference.html)

**Abstract**:

Existing Optimal Transport (OT) methods mainly derive the optimal transport plan/matching under the criterion of transport cost/distance minimization, which may cause incorrect matching in some cases. In many applications, annotating a few matched keypoints across domains is reasonable or even effortless in annotation burden. It is valuable to investigate how to leverage the annotated keypoints to guide the correct matching in OT. In this paper, we propose a novel KeyPoint-Guided model by ReLation preservation (KPG-RL) that searches for the matching guided by the keypoints in OT. To impose the keypoints in OT, first, we propose a mask-based constraint of the transport plan that preserves the matching of keypoint pairs. Second, we propose to preserve the relation of each data point to the keypoints to guide the matching. The proposed KPG-RL model can be solved by the Sinkhorn's algorithm and is applicable even when distributions are supported in different spaces. We further utilize the relation preservation constraint in the Kantorovich Problem and Gromov-Wasserstein model to impose the guidance of keypoints in them. Meanwhile, the proposed KPG-RL model is extended to partial OT setting. As an application, we apply the proposed KPG-RL model to the heterogeneous domain adaptation. Experiments verified the effectiveness of the KPG-RL model.

----

## [1089] Sequence-to-Set Generative Models

**Authors**: *Longtao Tang, Ying Zhou, Yu Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6091f2bb355e960600f62566ac0e2862-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a sequence-to-set method that can transform any sequence generative model based on maximum likelihood to a set generative model where we can evaluate the utility/probability of any set. An efficient importance sampling algorithm is devised to tackle the computational challenge of learning our sequence-to-set model. We present GRU2Set, which is an instance of our sequence-to-set method and employs the famous GRU model as the sequence generative model.To further obtain permutation invariant representation of sets, we devise the SetNN model which is also an instance of the sequence-to-set model. A direct application of our models is to learn an order/set distribution from a collection of e-commerce orders, which is an essential step in many important operational decisions such as inventory arrangement for fast delivery. Based on the intuition that small-sized sets are usually easier to learn than large sets, we propose a size-bias trick that can help learn better set distributions with respect to the $\ell_1$-distance evaluation metric. Two e-commerce order datasets, TMALL and HKTVMALL, are used to conduct extensive experiments to show the effectiveness of our models. The experimental results demonstrate that our models can learn better set/order distributions from order data than the baselines. Moreover, no matter what model we use, applying the size-bias trick can always improve the quality of the set distribution learned from data.

----

## [1090] Near-Optimal Multi-Agent Learning for Safe Coverage Control

**Authors**: *Manish Prajapat, Matteo Turchetta, Melanie N. Zeilinger, Andreas Krause*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/60dc26558762425a465cb0409fc3dc52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/60dc26558762425a465cb0409fc3dc52-Abstract-Conference.html)

**Abstract**:

In multi-agent coverage control problems, agents navigate their environment to reach locations that maximize the coverage of some density. In practice, the density is rarely known $\textit{a priori}$, further complicating the original NP-hard problem. Moreover, in many applications, agents cannot visit arbitrary locations due to $\textit{a priori}$ unknown safety constraints. In this paper, we aim to efficiently learn the density to approximately solve the coverage problem while preserving the agents' safety. We first propose a conditionally linear submodular coverage function that facilitates theoretical analysis. Utilizing this structure, we develop MacOpt, a novel algorithm that efficiently trades off the exploration-exploitation dilemma due to partial observability, and show that it achieves sublinear regret. Next, we extend results on single-agent safe exploration to our multi-agent setting and propose SafeMac for safe coverage and exploration. We analyze SafeMac and give first of its kind results: near optimal coverage in finite time while provably guaranteeing safety. We extensively evaluate our algorithms on synthetic and real problems, including a bio-diversity monitoring task under safety constraints, where SafeMac outperforms competing methods.

----

## [1091] Network change point localisation under local differential privacy

**Authors**: *Mengchu Li, Thomas Berrett, Yi Yu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)

**Abstract**:

Network data are ubiquitous in our daily life, containing rich but often sensitive information. In this paper, we expand the current static analysis of privatised networks to a dynamic framework by considering a sequence of networks with potential change points. We investigate the fundamental limits in consistently localising change points under both node and edge privacy constraints, demonstrating interesting phase transition in terms of the signal-to-noise ratio condition, accompanied by polynomial-time algorithms. The private signal-to-noise ratio conditions quantify the costs of the privacy for change point localisation problems and exhibit a different scaling in the sparsity parameter compared to the non-private counterparts. Our algorithms are shown to be optimal under the edge LDP constraint up to log factors. Under node LDP constraint, a gap exists between our upper bound and lower bound and we leave it as an interesting open problem, echoing the challenges in high-dimensional statistical inference under LDP constraints.

----

## [1092] Independence Testing for Bounded Degree Bayesian Networks

**Authors**: *Arnab Bhattacharyya, Clément L. Canonne, Joy Qiping Yang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/611252d40f23c8b57a8bc9ffb577419b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/611252d40f23c8b57a8bc9ffb577419b-Abstract-Conference.html)

**Abstract**:

We study the following independence testing problem: given access to samples from a distribution $P$ over $\{0,1\}^n$, decide whether $P$ is a product distribution or whether it is $\varepsilon$-far in total variation distance from any product distribution. For arbitrary distributions, this problem requires $\exp(n)$ samples. We show in this work that if $P$ has a sparse structure, then in fact only linearly many samples are required.Specifically, if $P$  is Markov with respect to a Bayesian network whose underlying DAG has in-degree bounded by $d$, then $\tilde{\Theta}(2^{d/2}\cdot n/\varepsilon^2)$ samples are necessary and sufficient for independence testing.

----

## [1093] Beyond spectral gap: the role of the topology in decentralized learning

**Authors**: *Thijs Vogels, Hadrien Hendrikx, Martin Jaggi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/61162d94822d468ee6e92803340f2040-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/61162d94822d468ee6e92803340f2040-Abstract-Conference.html)

**Abstract**:

In data-parallel optimization of machine learning models, workers collaborate to improve their estimates of the model: more accurate gradients allow them to use larger learning rates and optimize faster. We consider the setting in which all workers sample from the same dataset, and communicate over a sparse graph (decentralized). In this setting, current theory fails to capture important aspects of real-world behavior. First, the ‘spectral gap’ of the communication graph is not predictive of its empirical performance in (deep) learning. Second, current theory does not explain that collaboration enables larger learning rates than training alone. In fact, it prescribes smaller learning rates, which further decrease as graphs become larger, failing to explain convergence in infinite graphs. This paper aims to paint an accurate picture of sparsely-connected distributed optimization when workers share the same data distribution. We quantify how the graph topology influences convergence in a quadratic toy problem and provide theoretical results for general smooth and (strongly) convex objectives. Our theory matches empirical observations in deep learning, and accurately describes the relative merits of different graph topologies.

----

## [1094] GriddlyJS: A Web IDE for Reinforcement Learning

**Authors**: *Christopher Bamford, Minqi Jiang, Mikayel Samvelyan, Tim Rocktäschel*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/611b896d447df43c898062358df4c114-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Progress in reinforcement learning (RL) research is often driven by the design of new, challenging environments---a costly undertaking requiring skills orthogonal to that of a typical machine learning researcher. The complexity of environment development has only increased with the rise of procedural-content generation (PCG) as the prevailing paradigm for producing varied environments capable of testing the robustness and generalization of RL agents. Moreover, existing environments often require complex build processes, making reproducing results difficult. To address these issues, we introduce GriddlyJS, a web-based Integrated Development Environment (IDE) based on the Griddly engine. GriddlyJS allows researchers to easily design and debug arbitrary, complex PCG grid-world environments, as well as visualize, evaluate, and record the performance of trained agent models. By connecting the RL workflow to the advanced functionality enabled by modern web standards, GriddlyJS allows publishing interactive agent-environment demos that reproduce experimental results directly to the web. To demonstrate the versatility of GriddlyJS, we use it to quickly develop a complex compositional puzzle-solving environment alongside arbitrary human-designed environment configurations and their solutions for use in a automatic curriculum learning and offline RL context. The GriddlyJS IDE is open source and freely available at https://griddly.ai.

----

## [1095] Periodic Graph Transformers for Crystal Material Property Prediction

**Authors**: *Keqiang Yan, Yi Liu, Yuchao Lin, Shuiwang Ji*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6145c70a4a4bf353a31ac5496a72a72d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6145c70a4a4bf353a31ac5496a72a72d-Abstract-Conference.html)

**Abstract**:

We consider representation learning on periodic graphs encoding crystal materials. Different from regular graphs, periodic graphs consist of a minimum unit cell repeating itself on a regular lattice in 3D space. How to effectively encode these periodic structures poses unique challenges not present in regular graph representation learning. In addition to being E(3) invariant, periodic graph representations need to be periodic invariant. That is, the learned representations should be invariant to shifts of cell boundaries as they are artificially imposed. Furthermore, the periodic repeating patterns need to be captured explicitly as lattices of different sizes and orientations may correspond to different materials. In this work, we propose a transformer architecture, known as Matformer, for periodic graph representation learning. Our Matformer is designed to be invariant to periodicity and can capture repeating patterns explicitly. In particular, Matformer encodes periodic patterns by efficient use of geometric distances between the same atoms in neighboring cells. Experimental results on multiple common benchmark datasets show that our Matformer outperforms baseline methods consistently. In addition, our results demonstrate the importance of periodic invariance and explicit repeating pattern encoding for crystal representation learning. Our code is publicly available at https://github.com/YKQ98/Matformer.

----

## [1096] Decentralized, Communication- and Coordination-free Learning in Structured Matching Markets

**Authors**: *Chinmay Maheshwari, Shankar Sastry, Eric Mazumdar*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/615ce9f03a2b0174d21ee1ffa272fadd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/615ce9f03a2b0174d21ee1ffa272fadd-Abstract-Conference.html)

**Abstract**:

We study the problem of online learning in competitive settings in the context of two-sided matching markets. In particular, one side of the market, the agents, must learn about their preferences over the other side, the firms, through repeated interaction while competing with other agents for successful matches. We propose a class of decentralized, communication- and coordination-free algorithms that agents can use to reach to their stable match in structured matching markets. In contrast to prior works, the proposed algorithms make decisions based solely on an agent's own history of play and requires no foreknowledge of the firms' preferences. Our algorithms are constructed by splitting up the statistical problem of learning one's preferences, from noisy observations, from the problem of competing for firms. We show that under realistic structural assumptions on the underlying preferences of the agents and firms, the proposed algorithms incur a regret which grows at most logarithmically in the time horizon. However, we note that in the worst case, it may grow exponentially in the size of the market.

----

## [1097] Improving GANs with A Dynamic Discriminator

**Authors**: *Ceyuan Yang, Yujun Shen, Yinghao Xu, Deli Zhao, Bo Dai, Bolei Zhou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6174c67b136621f3f2e4a6b1d3286f6b-Abstract-Conference.html)

**Abstract**:

Discriminator plays a vital role in training generative adversarial networks (GANs) via distinguishing real and synthesized samples. While the real data distribution remains the same, the synthesis distribution keeps varying because of the evolving generator, and thus effects a corresponding change of the bi-classification task assigned to the discriminator. We argue that a discriminator with an on-the-fly adjustment on its capacity can better accommodate such a time-varying task. A comprehensive empirical study confirms that the proposed training strategy, termed as DynamicD, improves the synthesis performance without incurring any additional computation cost or training objectives. Two capacity adjusting schemes are developed for training GANs under different data regimes: i) given a sufficient amount of training data, the discriminator benefits from a progressively increased learning capacity, and ii) when the training data is limited, gradually decreasing the layer width mitigates the over-fitting issue of the discriminator. Experiments on both 2D and 3D-aware image synthesis tasks conducted on a range of datasets substantiate the generalizability of our DynamicD as well as its substantial improvement over the baselines. Furthermore, DynamicD is synergistic to other discriminator-improving approaches (including data augmentation, regularizers, and pre-training), and brings continuous performance gain when combined with them for learning GANs. Code will be made publicly available.

----

## [1098] Deliberated Domain Bridging for Domain Adaptive Semantic Segmentation

**Authors**: *Lin Chen, Zhixiang Wei, Xin Jin, Huaian Chen, Miao Zheng, Kai Chen, Yi Jin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/61aa557643ae8709b6a4f41140b2234a-Abstract-Conference.html)

**Abstract**:

In unsupervised domain adaptation (UDA), directly adapting from the source to the target domain usually suffers significant discrepancies and leads to insufficient alignment. Thus, many UDA works attempt to vanish the domain gap gradually and softly via various intermediate spaces, dubbed domain bridging (DB). However, for dense prediction tasks such as domain adaptive semantic segmentation (DASS), existing solutions have mostly relied on rough style transfer and how to elegantly bridge domains is still under-explored. In this work, we resort to data mixing to establish a deliberated domain bridging (DDB) for DASS, through which the joint distributions of source and target domains are aligned and interacted with each in the intermediate space. At the heart of DDB lies a dual-path domain bridging step for generating two intermediate domains using the coarse-wise and the fine-wise data mixing techniques, alongside a cross-path knowledge distillation step for taking two complementary models trained on generated intermediate samples as ‘teachers’ to develop a superior ‘student’ in a multi-teacher distillation manner. These two optimization steps work in an alternating way and reinforce each other to give rise to DDB with strong adaptation power. Extensive experiments on adaptive segmentation tasks with different settings demonstrate that our DDB significantly outperforms state-of-the-art methods.

----

## [1099] Learning to Share in Networked Multi-Agent Reinforcement Learning

**Authors**: *Yuxuan Yi, Ge Li, Yaowei Wang, Zongqing Lu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/61d8577984e4ef0cba20966eb3ef2ed8-Abstract-Conference.html)

**Abstract**:

In this paper, we study the problem of networked multi-agent reinforcement learning (MARL), where a number of agents are deployed as a partially connected network and each interacts only with nearby agents. Networked MARL requires all agents to make decisions in a decentralized manner to optimize a global objective with restricted communication between neighbors over the network. Inspired by the fact that sharing plays a key role in human's learning of cooperation, we propose LToS, a hierarchically decentralized MARL framework that enables agents to learn to dynamically share reward with neighbors so as to encourage agents to cooperate on the global objective through collectives. For each agent, the high-level policy learns how to share reward with neighbors to decompose the global objective, while the low-level policy learns to optimize the local objective induced by the high-level policies in the neighborhood. The two policies form a bi-level optimization and learn alternately. We empirically demonstrate that LToS outperforms existing methods in both social dilemma and networked MARL scenarios across scales.

----

## [1100] Scalable and Efficient Non-adaptive Deterministic Group Testing

**Authors**: *Dariusz R. Kowalski, Dominik Pajak*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/61fc0928bc62ad9cf0cb5cab961fc178-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/61fc0928bc62ad9cf0cb5cab961fc178-Abstract-Conference.html)

**Abstract**:

Group Testing (GT) is about learning a (hidden) subset $K$, of size $k$, of some large domain $N$, of size $n \gg k$, using a sequence of queries. A result of a query provides some information about the intersection of the query with the unknown set $K$. The goal is to design efficient (polynomial time) and scalable (polylogarithmic number of queries per element in $K$) algorithms for constructing queries that allow to decode every hidden set $K$ based on the results of the queries. A vast majority of the previous work focused on randomized algorithms minimizing the number of queries; however, in case of large domains N, randomization may result in asignificant deviation from the expected precision of learning the set $K$. Others assumed unlimited computational power (existential results) or adaptiveness of queries (next query could be constructed taking into account the results of the previous queries) – the former approach is less practical due to non-efficiency, and the latter has several drawbacks including non-parallelization. To avoid all the abovementioned drawbacks, for Quantitative Group Testing (QGT) where query result is the size of its intersection with the hidden set, we present the first efficient and scalable non-adaptive deterministic algorithms for constructing queries and decoding a hidden set K from the results of the queries – these solutions do not use any randomization, adaptiveness or unlimited computational power.

----

## [1101] Structure-Preserving 3D Garment Modeling with Neural Sewing Machines

**Authors**: *Xipeng Chen, Guangrun Wang, Dizhong Zhu, Xiaodan Liang, Philip H. S. Torr, Liang Lin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/620317fb69899dbf58798d242a58d351-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/620317fb69899dbf58798d242a58d351-Abstract-Conference.html)

**Abstract**:

3D Garment modeling is a critical and challenging topic in the area of computer vision and graphics, with increasing attention focused on garment representation learning, garment reconstruction, and controllable garment manipulation, whereas existing methods were constrained to model garments under specific categories or with relatively simple topologies. In this paper, we propose a novel Neural Sewing Machine (NSM), a learning-based framework for structure-preserving 3D garment modeling, which is capable of learning representations for garments with diverse shapes and topologies and is successfully applied to 3D garment reconstruction and controllable manipulation. To model generic garments, we first obtain sewing pattern embedding via a unified sewing pattern encoding module, as the sewing pattern can accurately describe the intrinsic structure and the topology of the 3D garment. Then we use a 3D garment decoder to decode the sewing pattern embedding into a 3D garment using the UV-position maps with masks. To preserve the intrinsic structure of the predicted 3D garment, we introduce an inner-panel structure-preserving loss, an inter-panel structure-preserving loss, and a surface-normal loss in the learning process of our framework. We evaluate NSM on the public 3D garment dataset with sewing patterns with diverse garment shapes and categories. Extensive experiments demonstrate that the proposed NSM is capable of representing 3D garments under diverse garment shapes and topologies, realistically reconstructing 3D garments from 2D images with the preserved structure, and accurately manipulating the 3D garment categories, shapes, and topologies, outperforming the state-of-the-art methods by a clear margin.

----

## [1102] Rethinking Variational Inference for Probabilistic Programs with Stochastic Support

**Authors**: *Tim Reichelt, Luke Ong, Thomas Rainforth*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62203a74e233e933b160711e791e1a02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62203a74e233e933b160711e791e1a02-Abstract-Conference.html)

**Abstract**:

We introduce Support Decomposition Variational Inference (SDVI), a new variational inference (VI) approach for probabilistic programs with stochastic support. Existing approaches to this problem rely on designing a single global variational guide on a variable-by-variable basis, while maintaining the stochastic control flow of the original program. SDVI instead breaks the program down into sub-programs with static support, before automatically building separate sub-guides for each. This decomposition significantly aids in the construction of suitable variational families, enabling, in turn, substantial improvements in inference performance.

----

## [1103] Variance Reduced ProxSkip: Algorithm, Theory and Application to Federated Learning

**Authors**: *Grigory Malinovsky, Kai Yi, Peter Richtárik*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/622afc4edf2824a1b6aaf5afe153fa93-Abstract-Conference.html)

**Abstract**:

We study distributed optimization methods based on the {\em local training (LT)} paradigm, i.e., methods which achieve communication efficiency by performing richer local gradient-based training on the clients before (expensive) parameter averaging is allowed to take place. While these methods were first proposed about a decade ago, and form the algorithmic backbone of federated learning, there is an enormous gap between their practical performance, and our theoretical understanding. Looking back at the progress of the field, we {\em identify 5 generations of LT methods}: 1) heuristic, 2) homogeneous, 3) sublinear, 4) linear, and 5) accelerated. The 5${}^{\rm th}$ generation was initiated by the ProxSkip method of Mishchenko et al. (2022), whose analysis provided the first theoretical confirmation that LT is a communication acceleration mechanism. Inspired by this recent progress, we contribute to the 5${}^{\rm th}$ generation of LT methods by showing that it is possible to enhance ProxSkip further using {\em variance reduction}. While all previous theoretical results for LT methods ignore the cost of local work altogether, and are framed purely in terms of the number of communication rounds, we construct a method that can be substantially faster in terms of the {\em total training time} than the state-of-the-art method ProxSkip in theory and practice in the regime when local computation is sufficiently expensive. We characterize this threshold theoretically, and confirm our theoretical predictions with empirical results. Our treatment of variance reduction is generic, and can work with a large number of variance reduction techniques, which may lead to future applications in the future. Finally, we corroborate our theoretical results with carefully engineered proof-of-concept experiments.

----

## [1104] DreamShard: Generalizable Embedding Table Placement for Recommender Systems

**Authors**: *Daochen Zha, Louis Feng, Qiaoyu Tan, Zirui Liu, Kwei-Herng Lai, Bhargav Bhushanam, Yuandong Tian, Arun Kejariwal, Xia Hu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62302a24b04589f9f9cdd5b02c344b6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62302a24b04589f9f9cdd5b02c344b6c-Abstract-Conference.html)

**Abstract**:

We study embedding table placement for distributed recommender systems, which aims to partition and place the tables on multiple hardware devices (e.g., GPUs) to balance the computation and communication costs. Although prior work has explored learning-based approaches for the device placement of computational graphs, embedding table placement remains to be a challenging problem because of 1) the operation fusion of embedding tables, and 2) the generalizability requirement on unseen placement tasks with different numbers of tables and/or devices. To this end, we present DreamShard, a reinforcement learning (RL) approach for embedding table placement. DreamShard achieves the reasoning of operation fusion and generalizability with 1) a cost network to directly predict the costs of the fused operation, and 2) a policy network that is efficiently trained on an estimated Markov decision process (MDP) without real GPU execution, where the states and the rewards are estimated with the cost network. Equipped with sum and max representation reductions, the two networks can directly generalize to any unseen tasks with different numbers of tables and/or devices without fine-tuning. Extensive experiments show that DreamShard substantially outperforms the existing human expert and RNN-based strategies with up to 19% speedup over the strongest baseline on large-scale synthetic tables and our production tables. The code is available.

----

## [1105] Order-Invariant Cardinality Estimators Are Differentially Private

**Authors**: *Charlie Dickens, Justin Thaler, Daniel Ting*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/623307df18da128262aaf394cdcfb235-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/623307df18da128262aaf394cdcfb235-Abstract-Conference.html)

**Abstract**:

We consider privacy in the context of streaming algorithms for cardinality estimation.    We show that a large class of algorithms all satisfy $\epsilon$-differential privacy,     so long as (a) the algorithm is combined with a simple     down-sampling procedure, and (b) the input stream cardinality      is $\Omega(k/\epsilon)$. Here, $k$ is a certain parameter of the sketch    that is always at most the sketch size in bits, but is typically much smaller.    We also show that, even with no modification, algorithms in our    class satisfy $(\epsilon, \delta)$-differential privacy,    where $\delta$ falls exponentially with the stream cardinality.     Our analysis applies to essentially all popular cardinality estimation    algorithms, and substantially generalizes and tightens privacy bounds from earlier works.     Our approach is faster and exhibits a better utility-space    tradeoff than prior art.

----

## [1106] NeuForm: Adaptive Overfitting for Neural Shape Editing

**Authors**: *Connor Z. Lin, Niloy J. Mitra, Gordon Wetzstein, Leonidas J. Guibas, Paul Guerrero*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/623e5a86fcedca573d33390dd1173e6b-Abstract-Conference.html)

**Abstract**:

Neural representations are popular for representing shapes as they can be used for data cleanup, model completion, shape editing, and shape synthesis. Current neural representations can be categorized as either overfitting to a single object instance, or representing a collection of objects. However, neither allows accurate editing of neural scene representations: on the one hand, methods that overfit objects achieve highly accurate reconstructions but do not support editing, as they do not generalize to unseen object configurations; on the other hand, methods that represent a family of objects with variations do generalize but produce approximate reconstructions. We propose NeuForm to combine the advantages of both overfitted and generalizable representations by adaptively overfitting a generalizable representation to regions where reliable data is available, while using the generalizable representation everywhere else. We achieve this with a carefully designed architecture and an approach that blends the network weights of the two representations. We demonstrate edits that successfully reconfigure parts of human-made shapes, such as chairs, tables, and lamps, while preserving the accuracy of an overfitted shape representation. We compare with two state-of-the-art competitors and demonstrate clear improvements in terms of plausibility and fidelity of the resultant edits.

----

## [1107] Inductive Logical Query Answering in Knowledge Graphs

**Authors**: *Michael Galkin, Zhaocheng Zhu, Hongyu Ren, Jian Tang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6246e04dcf42baf7c71e3a65d3d93b55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6246e04dcf42baf7c71e3a65d3d93b55-Abstract-Conference.html)

**Abstract**:

Formulating and answering logical queries is a standard communication interface for knowledge graphs (KGs). Alleviating the notorious incompleteness of real-world KGs, neural methods achieved impressive results in link prediction and complex query answering tasks by learning representations of entities, relations, and queries. Still, most existing query answering methods rely on transductive entity embeddings and cannot generalize to KGs containing new entities without retraining entity embeddings. In this work, we study the inductive query answering task where inference is performed on a graph containing new entities with queries over both seen and unseen entities. To this end, we devise two mechanisms leveraging inductive node and relational structure representations powered by graph neural networks (GNNs).Experimentally, we show that inductive models are able to perform logical reasoning at inference time over unseen nodes generalizing to graphs up to 500% larger than training ones. Exploring the efficiency--effectiveness trade-off, we find the inductive relational structure representation method generally achieves higher performance, while the inductive node representation method is able to answer complex queries in the inference-only regime without any training on queries and scale to graphs of millions of nodes. Code is available at https://github.com/DeepGraphLearning/InductiveQE

----

## [1108] Drawing out of Distribution with Neuro-Symbolic Generative Models

**Authors**: *Yichao Liang, Josh Tenenbaum, Tuan Anh Le, N. Siddharth*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6248a3b8279a39b3668a8a7c0e29164d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6248a3b8279a39b3668a8a7c0e29164d-Abstract-Conference.html)

**Abstract**:

Learning general-purpose representations from perceptual inputs is a hallmark of human intelligence. For example, people can write out numbers or characters, or even draw doodles, by characterizing these tasks as different instantiations of the same generic underlying process---compositional arrangements of different forms of pen strokes. Crucially, learning to do one task, say writing, implies reasonable competence at another, say drawing, on account of this shared process. We present Drawing out of Distribution (DooD), a neuro-symbolic generative model of stroke-based drawing that can learn such general-purpose representations. In contrast to prior work, DooD operates directly on images, requires no supervision or expensive test-time inference, and performs unsupervised amortized inference with a symbolic stroke model that better enables both interpretability and generalization. We evaluate DooD on its ability to generalize across both data and tasks. We first perform zero-shot transfer from one dataset (e.g. MNIST) to another (e.g. Quickdraw), across five different datasets, and show that DooD clearly outperforms different baselines. An analysis of the learnt representations further highlights the benefits of adopting a symbolic stroke model. We then adopt a subset of the Omniglot challenge tasks, and evaluate its ability to generate new exemplars (both unconditionally and conditionally), and perform one-shot classification, showing that DooD matches the state of the art. Taken together, we demonstrate that DooD does indeed capture general-purpose representations across both data and task, and takes a further step towards building general and robust concept-learning systems.

----

## [1109] Unsupervised Object Representation Learning using Translation and Rotation Group Equivariant VAE

**Authors**: *Alireza Nasiri, Tristan Bepler*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/624fa2f9200f3df11a4a80f6d880ccc2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/624fa2f9200f3df11a4a80f6d880ccc2-Abstract-Conference.html)

**Abstract**:

In many imaging modalities, objects of interest can occur in a variety of locations and poses (i.e. are subject to translations and rotations in 2d or 3d), but the location and pose of an object does not change its semantics (i.e. the object's essence). That is, the specific location and rotation of an airplane in satellite imagery, or the 3d rotation of a chair in a natural image, or the rotation of a particle in a cryo-electron micrograph, do not change the intrinsic nature of those objects. Here, we consider the problem of learning semantic representations of objects that are invariant to pose and location in a fully unsupervised manner. We address shortcomings in previous approaches to this problem by introducing TARGET-VAE, a translation and rotation group-equivariant variational autoencoder framework. TARGET-VAE combines three core innovations: 1) a rotation and translation group-equivariant encoder architecture, 2) a structurally disentangled distribution over latent rotation, translation, and a rotation-translation-invariant semantic object representation, which are jointly inferred by the approximate inference network, and 3) a spatially equivariant generator network. In comprehensive experiments, we show that TARGET-VAE learns disentangled representations without supervision that significantly improve upon, and avoid the pathologies of, previous methods. When trained on images highly corrupted by rotation and translation, the semantic representations learned by TARGET-VAE are similar to those learned on consistently posed objects, dramatically improving clustering in the semantic latent space. Furthermore, TARGET-VAE is able to perform remarkably accurate unsupervised pose and location inference. We expect methods like TARGET-VAE will underpin future approaches for unsupervised object generation, pose prediction, and object detection. Our code is available at https://github.com/SMLC-NYSBC/TARGET-VAE.

----

## [1110] PointTAD: Multi-Label Temporal Action Detection with Learnable Query Points

**Authors**: *Jing Tan, Xiaotong Zhao, Xintian Shi, Bin Kang, Limin Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6255539f776ce988a81d3841eadc4cf9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6255539f776ce988a81d3841eadc4cf9-Abstract-Conference.html)

**Abstract**:

Traditional temporal action detection (TAD) usually handles untrimmed videos with small number of action instances from a single label (e.g., ActivityNet, THUMOS). However, this setting might be unrealistic as different classes of actions often co-occur in practice. In this paper, we focus on the task of multi-label temporal action detection that aims to localize all action instances from a multi-label untrimmed video. Multi-label TAD is more challenging as it requires for fine-grained class discrimination within a single video and precise localization of the co-occurring instances. To mitigate this issue, we extend the sparse query-based detection paradigm from the traditional TAD and propose the multi-label TAD framework of PointTAD. Specifically, our PointTAD introduces a small set of learnable query points to represent the important frames of each action instance. This point-based representation provides a flexible mechanism to localize the discriminative frames at boundaries and as well the important frames inside the action. Moreover, we perform the action decoding process with the Multi-level Interactive Module to capture both point-level and instance-level action semantics. Finally, our PointTAD employs an end-to-end trainable framework simply based on RGB input for easy deployment. We evaluate our proposed method on two popular benchmarks and introduce the new metric of detection-mAP for multi-label TAD. Our model outperforms all previous methods by a large margin under the detection-mAP metric, and also achieves promising results under the segmentation-mAP metric.

----

## [1111] Unpacking Reward Shaping: Understanding the Benefits of Reward Engineering on Sample Complexity

**Authors**: *Abhishek Gupta, Aldo Pacchiano, Yuexiang Zhai, Sham M. Kakade, Sergey Levine*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6255f22349da5f2126dfc0b007075450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6255f22349da5f2126dfc0b007075450-Abstract-Conference.html)

**Abstract**:

The success of reinforcement learning in a variety of challenging sequential decision-making problems has been much discussed, but often ignored in this discussion is the consideration of how the choice of reward function affects the behavior of these algorithms. Most practical RL algorithms require copious amounts of reward engineering in order to successfully solve challenging tasks. The idea of this type of ``reward-shaping'' has been often discussed in the literature and is used in practical instantiations, but there is relatively little formal characterization of how the choice of reward shaping can yield benefits in sample complexity for RL problems. In this work, we build on the framework of novelty-based exploration to provide a simple scheme for incorporating shaped rewards into RL along with an analysis tool to show that particular choices of reward shaping provably improve sample efficiency. We characterize the class of problems where these gains are expected to be significant and show how this can be connected to practical algorithms in the literature. We show that these results hold in practice in experimental evaluations as well, providing an insight into the mechanisms through which reward shaping can significantly improve the complexity of reinforcement learning while retaining asymptotic performance.

----

## [1112] LASSIE: Learning Articulated Shapes from Sparse Image Ensemble via 3D Part Discovery

**Authors**: *Chun-Han Yao, Wei-Chih Hung, Yuanzhen Li, Michael Rubinstein, Ming-Hsuan Yang, Varun Jampani*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6274d57365d7a6be06e58cad30d1b9da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6274d57365d7a6be06e58cad30d1b9da-Abstract-Conference.html)

**Abstract**:

Creating high-quality articulated 3D models of animals is challenging either via manual creation or using 3D scanning tools. Therefore, techniques to reconstruct articulated 3D objects from 2D images are crucial and highly useful. In this work, we propose a practical problem setting to estimate 3D pose and shape of animals given only a few (10-30) in-the-wild images of a particular animal species (say, horse). Contrary to existing works that rely on pre-defined template shapes, we do not assume any form of 2D or 3D ground-truth annotations, nor do we leverage any multi-view or temporal information. Moreover, each input image ensemble can contain animal instances with varying poses, backgrounds, illuminations, and textures. Our key insight is that 3D parts have much simpler shape compared to the overall animal and that they are robust w.r.t. animal pose articulations. Following these insights, we propose LASSIE, a novel optimization framework which discovers 3D parts in a self-supervised manner with minimal user intervention. A key driving force behind LASSIE is the enforcing of 2D-3D part consistency using self-supervisory deep features. Experiments on Pascal-Part and self-collected in-the-wild animal datasets demonstrate considerably better 3D reconstructions as well as both 2D and 3D part discovery compared to prior arts. Project page: https://chhankyao.github.io/lassie/

----

## [1113] Retrieval-Augmented Diffusion Models

**Authors**: *Andreas Blattmann, Robin Rombach, Kaan Oktay, Jonas Müller, Björn Ommer*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62868cc2fc1eb5cdf321d05b4b88510c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62868cc2fc1eb5cdf321d05b4b88510c-Abstract-Conference.html)

**Abstract**:

Novel architectures have recently improved generative image synthesis leading to excellent visual quality in various tasks. Much of this success is due to the scalability of these architectures and hence caused by a dramatic increase in model complexity and in the computational resources invested in training these models. Our work questions the underlying paradigm of compressing large training data into ever growing parametric representations. We rather present an orthogonal, semi-parametric approach. We complement comparably small diffusion or autoregressive models with a separate image database and a retrieval strategy. During training we retrieve a set of nearest neighbors from this external database for each training instance and condition the generative model on these informative samples. While the retrieval approach is providing the (local) content, the model is focusing on learning the composition of scenes based on this content. As demonstrated by our experiments, simply swapping the database for one with different contents transfers a trained model post-hoc to a novel domain. The evaluation shows competitive performance on tasks which the generative model has not been trained on, such as class-conditional synthesis, zero-shot stylization or text-to-image synthesis without requiring paired text-image data. With negligible memory and computational overhead for the external database and retrieval we can significantly reduce the parameter count of the generative model and still outperform the state-of-the-art.

----

## [1114] Logical Credal Networks

**Authors**: *Radu Marinescu, Haifeng Qian, Alexander G. Gray, Debarun Bhattacharjya, Francisco Barahona, Tian Gao, Ryan Riegel, Pravinda Sahu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62891522c00cf7323cbacb500e6cfc8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62891522c00cf7323cbacb500e6cfc8d-Abstract-Conference.html)

**Abstract**:

We introduce Logical Credal Networks (or LCNs for short) -- an expressive probabilistic logic that generalizes prior formalisms that combine logic and probability. Given imprecise information represented by probability bounds and conditional probability bounds on logic formulas, an LCN specifies a set of probability distributions over all its interpretations. Our approach allows propositional and first-order logic formulas with few restrictions, e.g., without requiring acyclicity. We also define a generalized Markov condition that allows us to identify implicit independence relations between atomic formulas. We evaluate our method on benchmark problems such as random networks, Mastermind games with uncertainty and credit card fraud detection. Our results show that the LCN outperforms existing approaches; its advantage lies in aggregating multiple sources of imprecise information.

----

## [1115] Neural Set Function Extensions: Learning with Discrete Functions in High Dimensions

**Authors**: *Nikolaos Karalias, Joshua Robinson, Andreas Loukas, Stefanie Jegelka*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6294a235c0b80f0a2b224375c546c750-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6294a235c0b80f0a2b224375c546c750-Abstract-Conference.html)

**Abstract**:

Integrating functions on discrete domains into neural networks is key to developing their capability to reason about discrete objects. But, discrete domains are (1) not naturally amenable to gradient-based optimization, and (2) incompatible with deep learning architectures that rely on representations in high-dimensional vector spaces. In this work, we address both difficulties for set functions, which capture many important discrete problems. First, we develop a framework for extending set functions onto low-dimensional continuous domains, where many extensions are naturally defined. Our framework subsumes many well-known extensions as special cases. Second, to avoid undesirable low-dimensional neural network bottlenecks, we convert low-dimensional extensions into representations in high-dimensional spaces, taking inspiration from the success of semidefinite programs for combinatorial optimization. Empirically, we observe benefits of our extensions for unsupervised neural combinatorial optimization, in particular with high-dimensional representations.

----

## [1116] Minimax-Optimal Multi-Agent RL in Markov Games With a Generative Model

**Authors**: *Gen Li, Yuejie Chi, Yuting Wei, Yuxin Chen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62b4fea131cfd5b7504eae356b75bbd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62b4fea131cfd5b7504eae356b75bbd8-Abstract-Conference.html)

**Abstract**:

This paper studies multi-agent reinforcement learning in Markov games, with the goal of learning Nash equilibria or coarse correlated equilibria (CCE) sample-optimally. All prior results suffer from at least one of the two obstacles: the curse of multiple agents and the barrier of long horizon, regardless of the sampling protocol in use. We take a step towards settling this problem, assuming access to a flexible sampling mechanism: the generative model. Focusing on non-stationary finite-horizon Markov games, we develop a fast learning algorithm called Q-FTRL and an adaptive sampling scheme that leverage the optimism principle in online adversarial learning (particularly the Follow-the-Regularized-Leader (FTRL) method). Our algorithm learns an $\varepsilon$-approximate CCE in a general-sum Markov game using  $$ \widetilde{O}\bigg( \frac{H^4 S \sum_{i=1}^m A_i}{\varepsilon^2} \bigg) $$ samples, where $m$ is the number of players, $S$ indicates the number of states, $H$ is the horizon, and $A_i$ denotes the number of actions for the $i$-th player. This is minimax-optimal (up to log factor) when $m$ is fixed. When applied to two-player zero-sum Markov games, our algorithm provably finds an $\varepsilon$-approximate Nash equilibrium with a minimal number of samples. Along the way, we derive a refined regret bound for FTRL that makes explicit the role of variance-type quantities, which might be of independent interest.

----

## [1117] Bi-directional Weakly Supervised Knowledge Distillation for Whole Slide Image Classification

**Authors**: *Linhao Qu, Xiaoyuan Luo, Manning Wang, Zhijian Song*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62c9aa4d48329a85d1e36d5b6d0a6a32-Abstract-Conference.html)

**Abstract**:

Computer-aided pathology diagnosis based on the classification of Whole Slide Image (WSI) plays an important role in clinical practice, and it is often formulated as a weakly-supervised Multiple Instance Learning (MIL) problem. Existing methods solve this problem from either a bag classification or an instance classification perspective. In this paper, we propose an end-to-end weakly supervised knowledge distillation framework (WENO) for WSI classification, which integrates a bag classifier and an instance classifier in a knowledge distillation framework to mutually improve the performance of both classifiers. Specifically, an attention-based bag classifier is used as the teacher network, which is trained with weak bag labels, and an instance classifier is used as the student network, which is trained using the normalized attention scores obtained from the teacher network as soft pseudo labels for the instances in positive bags. An instance feature extractor is shared between the teacher and the student to further enhance the knowledge exchange between them. In addition, we propose a hard positive instance mining strategy based on the output of the student network to force the teacher network to keep mining hard positive instances. WENO is a plug-and-play framework that can be easily applied to any existing attention-based bag classification methods. Extensive experiments on five datasets demonstrate the efficiency of WENO. Code is available at https://github.com/miccaiif/WENO.

----

## [1118] Private and Communication-Efficient Algorithms for Entropy Estimation

**Authors**: *Gecia Bravo Hermsdorff, Róbert Busa-Fekete, Mohammad Ghavamzadeh, Andrés Muñoz Medina, Umar Syed*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/62e5721247075dd097023d077d8e22f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/62e5721247075dd097023d077d8e22f7-Abstract-Conference.html)

**Abstract**:

Modern statistical estimation is often performed in a distributed setting where each sample belongs to single user who shares their data with a central server. Users are typically concerned with preserving the privacy of their sample, and also with minimizing the amount of data they must transmit to the server. We give improved private and communication-efficient algorithms for estimating several popular measures of the entropy of a distribution. All of our algorithms have constant communication cost and satisfy local differential privacy. For a joint distribution on many variables whose conditional independence graph is a tree, we describe algorithms for estimating Shannon entropy that require a number of samples that is linear in the number of variables, compared to the quadratic sample complexity of prior work. We also describe an algorithm for estimating Gini entropy whose sample complexity has no dependence on the support size of the distribution and can be implemented using a single round of concurrent communication between the users and the server, while the previously best-known algorithm has high communication cost and requires the server to facilitate interaction between the users. Finally, we describe an algorithm for estimating collision entropy that matches the space and sample complexity of the best known algorithm but generalizes it to the private and communication-efficient setting.

----

## [1119] PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient

**Authors**: *Weihan Cao, Yifan Zhang, Jianfei Gao, Anda Cheng, Ke Cheng, Jian Cheng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/631ad9ae3174bf4d6c0f6fdca77335a4-Abstract-Conference.html)

**Abstract**:

Knowledge distillation(KD) is a widely-used technique to train compact models in object detection. However, there is still a lack of study on how to distill between heterogeneous detectors. In this paper, we empirically find that better FPN features from a heterogeneous teacher detector can help the student although their detection heads and label assignments are different. However, directly aligning the feature maps to distill detectors suffers from two problems. First, the difference in feature magnitude between the teacher and the student could enforce overly strict constraints on the student. Second, the FPN stages and channels with large feature magnitude from the teacher model could dominate the gradient of distillation loss, which will overwhelm the effects of other features in KD and introduce much noise. To address the above issues, we propose to imitate features with Pearson Correlation Coefficient to focus on the relational information from the teacher and relax constraints on the magnitude of the features. Our method consistently outperforms the existing detection KD methods and works for both homogeneous and heterogeneous student-teacher pairs. Furthermore, it converges faster. With a powerful MaskRCNN-Swin detector as the teacher, ResNet-50 based RetinaNet and FCOS achieve 41.5% and 43.9% $mAP$ on COCO2017, which are 4.1% and 4.8% higher than the baseline, respectively.

----

## [1120] Off-Team Learning

**Authors**: *Brandon Cui, Hengyuan Hu, Andrei Lupu, Samuel Sokota, Jakob N. Foerster*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/631f99d8e860054410c239fc90d18270-Abstract-Conference.html)

**Abstract**:

Zero-shot coordination (ZSC) evaluates an algorithm by the performance of a team of agents that were trained independently under that algorithm. Off-belief learning (OBL) is a recent method that achieves state-of-the-art results in ZSC in the game Hanabi. However, the implementation of OBL relies on a belief model that experiences covariate shift. Moreover, during ad-hoc coordination, OBL or any other neural policy may experience test-time covariate shift. We present two methods addressing these issues. The first method, off-team belief learning (OTBL), attempts to improve the accuracy of the belief model of a target policy πT on a broader range of inputs by weighting trajectories approximately according to the distribution induced by a different policy πb. The second, off-team off-belief learning (OT-OBL), attempts to compute an OBL equilibrium, where fixed point error is weighted according to the distribution induced by cross-play between the training policy π and a different fixed policy πb instead of self-play of π. We investigate these methods in variants of Hanabi.

----

## [1121] NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis

**Authors**: *Jian Liang, Chenfei Wu, Xiaowei Hu, Zhe Gan, Jianfeng Wang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6358cd0cd6607fdf4870595795eb1710-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6358cd0cd6607fdf4870595795eb1710-Abstract-Conference.html)

**Abstract**:

Infinite visual synthesis aims to generate high-resolution images, long-duration videos, and even visual generation of infinite size. Some recent work tried to solve this task by first dividing data into processable patches and then training the models on them without considering the dependencies between patches. However, since they fail to model global dependencies between patches, the quality and consistency of the generation can be limited. To address this issue, we propose NUWA-Infinity, a patch-level \emph{``render-and-optimize''} strategy for infinite visual synthesis. Given a large image or a long video, NUWA-Infinity first splits it into non-overlapping patches and uses the ordered patch chain as a complete training instance, a rendering model autoregressively predicts each patch based on its contexts. Once a patch is predicted, it is optimized immediately and its hidden states are saved as contexts for the next \emph{``render-and-optimize''} process. This brings two advantages: ($i$) The autoregressive rendering process with information transfer between contexts provides an implicit global probabilistic distribution modeling; ($ii$) The timely optimization process alleviates the optimization stress of the model and helps convergence.  Based on the above designs, NUWA-Infinity shows a strong synthesis ability on high-resolution images and long-duration videos. The homepage link is \url{https://nuwa-infinity.microsoft.com}.

----

## [1122] SoftPatch: Unsupervised Anomaly Detection with Noisy Data

**Authors**: *Xi Jiang, Jianlin Liu, Jinbao Wang, Qiang Nie, Kai Wu, Yong Liu, Chengjie Wang, Feng Zheng*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/637a456d89289769ac1ab29617ef7213-Abstract-Conference.html)

**Abstract**:

Although mainstream unsupervised anomaly detection (AD) algorithms perform well in academic datasets, their performance is limited in practical application due to the ideal experimental setting of clean training data. Training with noisy data is an inevitable problem in real-world anomaly detection but is seldom discussed. This paper considers label-level noise in image sensory anomaly detection for the first time. To solve this problem, we proposed a memory-based unsupervised AD method, SoftPatch, which efficiently denoises the data at the patch level. Noise discriminators are utilized to generate outlier scores for patch-level noise elimination before coreset construction. The scores are then stored in the memory bank to soften the anomaly detection boundary. Compared with existing methods, SoftPatch maintains a strong modeling ability of normal data and alleviates the overconfidence problem in coreset. Comprehensive experiments in various noise scenes demonstrate that SoftPatch outperforms the state-of-the-art AD methods on the MVTecAD and BTAD benchmarks and is comparable to those methods under the setting without noise.

----

## [1123] Stability Analysis and Generalization Bounds of Adversarial Training

**Authors**: *Jiancong Xiao, Yanbo Fan, Ruoyu Sun, Jue Wang, Zhi-Quan Luo*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/637de5e2a7a77f741b0b84bd61c83125-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/637de5e2a7a77f741b0b84bd61c83125-Abstract-Conference.html)

**Abstract**:

In adversarial machine learning, deep neural networks can fit the adversarial examples on the training dataset but have poor generalization ability on the test set. This phenomenon is called robust overfitting, and it can be observed when adversarially training neural nets on common datasets, including SVHN, CIFAR-10, CIFAR-100, and ImageNet. In this paper, we study the robust overfitting issue of adversarial training by using tools from uniform stability. One major challenge is that the outer function (as a maximization of the inner function) is nonsmooth, so the standard technique (e.g., Hardt et al., 2016) cannot be applied. Our approach is to consider $\eta$-approximate smoothness: we show that the outer function satisfies this modified smoothness assumption with $\eta$ being a constant related to the adversarial perturbation $\epsilon$. Based on this, we derive stability-based generalization bounds for stochastic gradient descent (SGD) on the general class of $\eta$-approximate smooth functions, which covers the adversarial loss. Our results suggest that robust test accuracy decreases in $\epsilon$ when $T$ is large, with a speed between $\Omega(\epsilon\sqrt{T})$ and $\mathcal{O}(\epsilon T)$. This phenomenon is also observed in practice. Additionally, we show that a few popular techniques for adversarial training (\emph{e.g.,} early stopping, cyclic learning rate, and stochastic weight averaging) are stability-promoting in theory.

----

## [1124] LasUIE: Unifying Information Extraction with Latent Adaptive Structure-aware Generative Language Model

**Authors**: *Hao Fei, Shengqiong Wu, Jingye Li, Bobo Li, Fei Li, Libo Qin, Meishan Zhang, Min Zhang, Tat-Seng Chua*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63943ee9fe347f3d95892cf87d9a42e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63943ee9fe347f3d95892cf87d9a42e6-Abstract-Conference.html)

**Abstract**:

Universally modeling all typical information extraction tasks (UIE) with one generative language model (GLM) has revealed great potential by the latest study, where various IE predictions are unified into a linearized hierarchical expression under a GLM. Syntactic structure information, a type of effective feature which has been extensively utilized in IE community, should also be beneficial to UIE. In this work, we propose a novel structure-aware GLM, fully unleashing the power of syntactic knowledge for UIE. A heterogeneous structure inductor is explored to unsupervisedly induce rich heterogeneous structural representations by post-training an existing GLM. In particular, a structural broadcaster is devised to compact various latent trees into explicit high-order forests, helping to guide a better generation during decoding. We finally introduce a task-oriented structure fine-tuning mechanism, further adjusting the learned structures to most coincide with the end-task's need. Over 12 IE benchmarks across 7 tasks our system shows significant improvements over the baseline UIE system. Further in-depth analyses show that our GLM learns rich task-adaptive structural bias that greatly resolves the UIE crux, the long-range dependence issue and boundary identifying.

----

## [1125] STaR: Bootstrapping Reasoning With Reasoning

**Authors**: *Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html)

**Abstract**:

Generating step-by-step "chain-of-thought" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy by using only few-shot inference. We propose a technique to iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning. This technique, the "Self-Taught Reasoner" (STaR), relies on a simple loop: generate rationales to answer many questions, prompted with a few rationale examples; if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat. We show that STaR significantly improves performance on multiple datasets compared to a model fine-tuned to directly predict final answers, and performs comparably to fine-tuning a 30$\times$ larger state-of-the-art language model on CommensenseQA. Thus, STaR lets a model improve itself by learning from its own generated reasoning.

----

## [1126] TPU-KNN: K Nearest Neighbor Search at Peak FLOP/s

**Authors**: *Felix Chern, Blake Hechtman, Andy Davis, Ruiqi Guo, David Majnemer, Sanjiv Kumar*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/639d992f819c2b40387d4d5170b8ffd7-Abstract-Conference.html)

**Abstract**:

This paper presents a novel nearest neighbor search algorithm achieving TPU (Google Tensor Processing Unit) peak performance, outperforming state-of-the-art GPU algorithms with similar level of recall. The design of the proposed algorithm is motivated by an accurate accelerator performance model that takes into account both the  memory and instruction bottlenecks. Our algorithm comes with an analytical guarantee of recall in expectation and does not require maintaining sophisticated index data structure or tuning, making it suitable for applications with frequent updates. Our work is available in the open-source package of Jax and Tensorflow on TPU.

----

## [1127] Efficient Meta Reinforcement Learning for Preference-based Fast Adaptation

**Authors**: *Zhizhou Ren, Anji Liu, Yitao Liang, Jian Peng, Jianzhu Ma*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63b2b056f48653b7cff0d8d233c96a4d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63b2b056f48653b7cff0d8d233c96a4d-Abstract-Conference.html)

**Abstract**:

Learning new task-specific skills from a few trials is a fundamental challenge for artificial intelligence. Meta reinforcement learning (meta-RL) tackles this problem by learning transferable policies that support few-shot adaptation to unseen tasks. Despite recent advances in meta-RL, most existing methods require the access to the environmental reward function of new tasks to infer the task objective, which is not realistic in many practical applications. To bridge this gap, we study the problem of few-shot adaptation in the context of human-in-the-loop reinforcement learning. We develop a meta-RL algorithm that enables fast policy adaptation with preference-based feedback. The agent can adapt to new tasks by querying human's preference between behavior trajectories instead of using per-step numeric rewards. By extending techniques from information theory, our approach can design query sequences to maximize the information gain from human interactions while tolerating the inherent error of non-expert human oracle. In experiments, we extensively evaluate our method, Adaptation with Noisy OracLE (ANOLE), on a variety of meta-RL benchmark tasks and demonstrate substantial improvement over baseline algorithms in terms of both feedback efficiency and error tolerance.

----

## [1128] Weakly Supervised Representation Learning with Sparse Perturbations

**Authors**: *Kartik Ahuja, Jason S. Hartford, Yoshua Bengio*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63d3bae2c1f525745003f679e45bcf7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63d3bae2c1f525745003f679e45bcf7b-Abstract-Conference.html)

**Abstract**:

The theory of representation learning aims to build methods that provably invert the data generating process with minimal domain knowledge or any source of supervision. Most prior approaches require strong distributional assumptions on the latent variables and weak supervision (auxiliary information such as timestamps) to provide provable identification guarantees. In this work, we show that if one has weak supervision from observations generated by sparse perturbations of the latent variables--e.g. images in a reinforcement learning environment where actions move individual sprites--identification is achievable under unknown continuous latent distributions. We show that if the perturbations are applied only on mutually exclusive blocks of latents, we identify the latents up to those blocks. We also show that if these perturbation blocks overlap, we identify latents up to the smallest blocks shared across perturbations. Consequently, if there are blocks that intersect in one latent variable only, then such latents are identified up to permutation and scaling. We propose a natural estimation procedure based on this theory and illustrate it on low-dimensional synthetic and image-based experiments.

----

## [1129] A Multi-Resolution Framework for U-Nets with Applications to Hierarchical VAEs

**Authors**: *Fabian Falck, Christopher Williams, Dominic Danks, George Deligiannidis, Christopher Yau, Chris C. Holmes, Arnaud Doucet, Matthew Willetts*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63f471bb806bf5a3c0a19a99acf5a12a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63f471bb806bf5a3c0a19a99acf5a12a-Abstract-Conference.html)

**Abstract**:

U-Net architectures are ubiquitous in state-of-the-art deep learning, however their regularisation properties and relationship to wavelets are understudied. In this paper, we formulate a multi-resolution framework which identifies U-Nets as finite-dimensional truncations of models on an infinite-dimensional function space. We provide theoretical results which prove that average pooling corresponds to projection within the space of square-integrable functions and show that U-Nets with average pooling implicitly learn a Haar wavelet basis representation of the data. We then leverage our framework to identify state-of-the-art hierarchical VAEs (HVAEs), which have a U-Net architecture, as a type of two-step forward Euler discretisation of multi-resolution diffusion processes which flow from a point mass, introducing sampling instabilities. We also demonstrate that HVAEs learn a representation of time which allows for improved parameter efficiency through weight-sharing. We use this observation to achieve state-of-the-art HVAE performance with half the number of parameters of existing models, exploiting the properties of our continuous-time formulation.

----

## [1130] Watermarking for Out-of-distribution Detection

**Authors**: *Qizhou Wang, Feng Liu, Yonggang Zhang, Jing Zhang, Chen Gong, Tongliang Liu, Bo Han*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63fa7efdd3bcf944a4bd6e0ff6a50041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63fa7efdd3bcf944a4bd6e0ff6a50041-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection aims to identify OOD data based on representations extracted from well-trained deep models. However, existing methods largely ignore the reprogramming property of deep models and thus may not fully unleash their intrinsic strength: without modifying parameters of a well-trained deep model, we can reprogram this model for a new purpose via data-level manipulation (e.g., adding a specific feature perturbation). This property motivates us to reprogram a classification model to excel at OOD detection (a new task), and thus we propose a general methodology named watermarking in this paper. Specifically, we learn a unified pattern that is superimposed onto features of original data, and the model's detection capability is largely boosted after watermarking. Extensive experiments verify the effectiveness of watermarking, demonstrating the significance of the reprogramming property of deep models in OOD detection.

----

## [1131] K-LITE: Learning Transferable Visual Models with External Knowledge

**Authors**: *Sheng Shen, Chunyuan Li, Xiaowei Hu, Yujia Xie, Jianwei Yang, Pengchuan Zhang, Zhe Gan, Lijuan Wang, Lu Yuan, Ce Liu, Kurt Keutzer, Trevor Darrell, Anna Rohrbach, Jianfeng Gao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/63fef0802863f47775c3563e18cbba17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/63fef0802863f47775c3563e18cbba17-Abstract-Conference.html)

**Abstract**:

The new generation of state-of-the-art computer vision systems are trained from natural language supervision, ranging from simple object category names to descriptive captions. This form of supervision ensures high generality and usability of the learned visual models, based on the broad concept coverage achieved through large-scale data collection process. Alternatively, we argue that learning with external knowledge about images is a promising way which leverages a much more structured source of supervision and offers sample efficiency. In this paper, we propose K-LITE (Knowledge-augmented Language-Image Training and Evaluation), a simple strategy to leverage external knowledge for building transferable visual systems: In training, it enriches entities in natural language with WordNet and Wiktionary knowledge, leading to an efficient and scalable approach to learning image representations that uses knowledge about the visual concepts; In evaluation, the natural language is also augmented with external knowledge and then used to reference learned visual concepts (or describe new ones) to enable zero-shot and few-shot transfer of the pre-trained models. We study the performance of K-LITE on two important computer vision problems, image classification and object detection, benchmarking on 20 and 13 different existing datasets, respectively. The proposed knowledge-augmented models show significant improvement in transfer learning performance over existing methods. Our code is released at https://github.com/microsoft/klite.

----

## [1132] VeriDark: A Large-Scale Benchmark for Authorship Verification on the Dark Web

**Authors**: *Andrei Manolache, Florin Brad, Antonio Barbalau, Radu Tudor Ionescu, Marius Popescu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/64008fa30cba9b4d1ab1bd3bd3d57d61-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The Dark Web represents a hotbed for illicit activity, where users communicate on different market forums in order to exchange goods and services. Law enforcement agencies benefit from forensic tools that perform authorship analysis, in order to identify and profile users based on their textual content. However, authorship analysis has been traditionally studied using corpora featuring literary texts such as fragments from novels or fan fiction, which may not be suitable in a cybercrime context. Moreover, the few works that employ authorship analysis tools for cybercrime prevention usually employ ad-hoc experimental setups and datasets. To address these issues, we release VeriDark: a benchmark comprised of three large scale authorship verification datasets and one authorship identification dataset obtained from user activity from either Dark Web related Reddit communities or popular illicit Dark Web market forums. We evaluate competitive NLP baselines on the three datasets and perform an analysis of the predictions to better understand the limitations of such approaches. We make the datasets and baselines publicly available at https://github.com/bit-ml/VeriDark .

----

## [1133] EHRSQL: A Practical Text-to-SQL Benchmark for Electronic Health Records

**Authors**: *Gyubok Lee, Hyeonji Hwang, Seongsu Bae, Yeonsu Kwon, Woncheol Shin, Seongjun Yang, Minjoon Seo, Jong-Yeup Kim, Edward Choi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/643e347250cf9289e5a2a6c1ed5ee42e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/643e347250cf9289e5a2a6c1ed5ee42e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present a new text-to-SQL dataset for electronic health records (EHRs). The utterances were collected from 222 hospital staff, including physicians, nurses, insurance review and health records teams, and more. To construct the QA dataset on structured EHR data, we conducted a poll at a university hospital and templatized the responses to create seed questions. Then, we manually linked them to two open-source EHR databases—MIMIC-III and eICU—and included them with various time expressions and held-out unanswerable questions in the dataset, which were all collected from the poll. Our dataset poses a unique set of challenges: the model needs to 1) generate SQL queries that reflect a wide range of needs in the hospital, including simple retrieval and complex operations such as calculating survival rate, 2) understand various time expressions to answer time-sensitive questions in healthcare, and 3) distinguish whether a given question is answerable or unanswerable based on the prediction confidence. We believe our dataset, EHRSQL, could serve as a practical benchmark to develop and assess QA models on structured EHR data and take one step further towards bridging the gap between text-to-SQL research and its real-life deployment in healthcare. EHRSQL is available at https://github.com/glee4810/EHRSQL.

----

## [1134] Where do Models go Wrong? Parameter-Space Saliency Maps for Explainability

**Authors**: *Roman Levin, Manli Shu, Eitan Borgnia, Furong Huang, Micah Goldblum, Tom Goldstein*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6450ea28ebbc8437bc38775157818172-Abstract-Conference.html)

**Abstract**:

Conventional saliency maps highlight input features to which neural network predictions are highly sensitive. We take a different approach to saliency, in which we identify and analyze the network parameters, rather than inputs, which are responsible for erroneous decisions. We first verify that identified salient parameters are indeed responsible for misclassification by showing that turning these parameters off improves predictions on the associated samples more than turning off the same number of random or least salient parameters. We further validate the link between salient parameters and network misclassification errors by observing that fine-tuning a small number of the most salient parameters on a single sample results in error correction on other samples which were misclassified for similar reasons -- nearest neighbors in the saliency space. After validating our parameter-space saliency maps, we demonstrate that samples which cause similar parameters to malfunction are semantically similar. Further, we introduce an input-space saliency counterpart which reveals how image features cause specific network components to malfunction.

----

## [1135] Using Embeddings for Causal Estimation of Peer Influence in Social Networks

**Authors**: *Irina Cristali, Victor Veitch*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64587794695be22545d91c838243fcf8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/64587794695be22545d91c838243fcf8-Abstract-Conference.html)

**Abstract**:

We address the problem of using observational data to estimate peer contagion effects, the influence of treatments applied to individuals in a network on the outcomes of their neighbors. A main challenge to such estimation is that homophily - the tendency of connected units to share similar latent traits - acts as an unobserved confounder for contagion effects. Informally, it's hard to tell whether your friends have similar outcomes because they were influenced by your treatment, or whether it's due to some common trait that caused you to be friends in the first place. Because these common causes are not usually directly observed, they cannot be simply adjusted for. We describe an approach to perform the required adjustment using node embeddings learned from the network itself. The main aim is to perform this adjustment nonparametrically, without functional form assumptions on either the process that generated the network or the treatment assignment and outcome processes. The key contributions are to nonparametrically formalize the causal effect in a way that accounts for homophily, and to show how embedding methods can be used to identify and estimate this effect.

----

## [1136] Collaborative Learning by Detecting Collaboration Partners

**Authors**: *Shu Ding, Wei Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/646ca7b994bc46afe33d680dbe7ed67a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/646ca7b994bc46afe33d680dbe7ed67a-Abstract-Conference.html)

**Abstract**:

Massive amounts of data are naturally dispersed over different clients in many real-world applications, collaborative learning has been a promising paradigm that allows to learn models through collaboration among the clients.   However, leveraging these dispersed data to learn good models is still challenging since data over different clients are heterogeneous.   Previous works mainly focus on learning the centralized model for all clients or learning a personalized model for each client.  When there are numerous clients, the centralized model performs badly on some clients, while learning a personalized model for each client costs unaffordable computational resources.  In this paper, we propose the collaborative learning method to detect collaboration partners and adaptively learn $K$ models for numerous heterogeneous clients.  We theoretically prove that the model learned for each client is a good approximation of its personalized model.  Experimental results on real-world datasets verify the effectiveness of our method.

----

## [1137] Linear Label Ranking with Bounded Noise

**Authors**: *Dimitris Fotakis, Alkis Kalavasis, Vasilis Kontonis, Christos Tzamos*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/64792f7bd5d400c9ac310c6fef97ef2d-Abstract-Conference.html)

**Abstract**:

Label Ranking (LR) is the supervised task of learning a sorting function that maps feature vectors $x \in \mathbb{R}^d$ to rankings $\sigma(x) \in \mathbb S_k$ over a finite set of $k$ labels. We focus on the fundamental case of learning linear sorting functions (LSFs) under Gaussian marginals: $x$ is sampled from the $d$-dimensional standard normal and  the ground truth ranking $\sigma^\star(x)$ is the ordering induced by  sorting the coordinates of the vector $W^\star x$, where  $W^\star \in \mathbb{R}^{k \times d}$ is unknown. We consider learning LSFs in the presence of bounded noise: assuming that a noiseless example is of the form $(x, \sigma^\star(x))$, we observe $(x, \pi)$, where for any pair of elements $i \neq j$, the probability that the order of $i, j$ is different in $\pi$ than in  $\sigma^\star(x)$ is at most $\eta < 1/2$. We design efficient non-proper and proper learning algorithms that  learn hypotheses within normalized Kendall's Tau distance $\epsilon$ from the ground truth  with $N= \widetilde{O}(d\log(k)/\epsilon)$ labeled examples and runtime $\mathrm{poly}(N, k)$. For the more challenging top-$r$ disagreement loss, we give an efficient proper learning algorithm that achieves $\epsilon$ top-$r$ disagreement with the ground truth with $N = \widetilde{O}(d k r /\epsilon)$ samples and $\mathrm{poly}(N)$ runtime.

----

## [1138] Quo Vadis: Is Trajectory Forecasting the Key Towards Long-Term Multi-Object Tracking?

**Authors**: *Patrick Dendorfer, Vladimir Yugay, Aljosa Osep, Laura Leal-Taixé*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/647dc4a76b3efdd676f50f32949299a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/647dc4a76b3efdd676f50f32949299a8-Abstract-Conference.html)

**Abstract**:

Recent developments in monocular multi-object tracking have been very successful in tracking visible objects and bridging short occlusion gaps, mainly relying on data-driven appearance models. While significant advancements have been made in short-term tracking performance, bridging longer occlusion gaps remains elusive: state-of-the-art object trackers only bridge less than 10% of occlusions longer than three seconds. We suggest that the missing key is reasoning about future trajectories over a longer time horizon. Intuitively, the longer the occlusion gap, the larger the search space for possible associations. In this paper, we show that even a small yet diverse set of trajectory predictions for moving agents will significantly reduce this search space and thus improve long-term tracking robustness. Our experiments suggest that the crucial components of our approach are reasoning in a bird's-eye view space and generating a small yet diverse set of forecasts while accounting for their localization uncertainty. This way, we can advance state-of-the-art trackers on the MOTChallenge dataset and significantly improve their long-term tracking performance. This paper's source code and experimental data are available at https://github.com/dendorferpatrick/QuoVadis.

----

## [1139] Wasserstein Iterative Networks for Barycenter Estimation

**Authors**: *Alexander Korotin, Vage Egiazarian, Lingxiao Li, Evgeny Burnaev*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6489f2c6ac6420124fcef2a489615a97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6489f2c6ac6420124fcef2a489615a97-Abstract-Conference.html)

**Abstract**:

Wasserstein barycenters have become popular due to their ability to represent the average of probability measures in a geometrically meaningful way. In this paper, we present an algorithm to approximate the Wasserstein-2 barycenters of continuous measures via a generative model. Previous approaches rely on regularization (entropic/quadratic) which introduces bias or on input convex neural networks which are not expressive enough for large-scale tasks. In contrast, our algorithm does not introduce bias and allows using arbitrary neural networks. In addition, based on the celebrity faces dataset, we construct Ave, celeba! dataset which can be used for quantitative evaluation of barycenter algorithms by using standard metrics of generative models such as FID.

----

## [1140] Identifiability of deep generative models without auxiliary information

**Authors**: *Bohdan Kivva, Goutham Rajendran, Pradeep Ravikumar, Bryon Aragam*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/649f080d8891ab4d4b262cb9cd52e69a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/649f080d8891ab4d4b262cb9cd52e69a-Abstract-Conference.html)

**Abstract**:

We prove identifiability of a broad class of deep latent variable models that (a) have universal approximation capabilities and (b) are the decoders of variational autoencoders that are commonly used in practice. Unlike existing work, our analysis does not require weak supervision, auxiliary information, or conditioning in the latent space. Specifically, we show that for a broad class of generative (i.e. unsupervised) models with universal approximation capabilities, the side information $u$ is not necessary: We prove identifiability of the entire generative model where we do not observe $u$ and only observe the data $x$. The models we consider match autoencoder architectures used in practice that leverage mixture priors in the latent space and ReLU/leaky-ReLU activations in the encoder, such as VaDE and MFC-VAE. Our main result is an identifiability hierarchy that significantly generalizes previous work and exposes how different assumptions lead to different ``strengths'' of identifiability, and includes certain ``vanilla'' VAEs with isotropic Gaussian priors as a special case. For example, our weakest result establishes (unsupervised) identifiability up to an affine transformation, and thus partially resolves an open problem regarding model identifiability raised in prior work. These theoretical results are augmented with experiments on both simulated and real data.

----

## [1141] Task Discovery: Finding the Tasks that Neural Networks Generalize on

**Authors**: *Andrei Atanov, Andrei Filatov, Teresa Yeo, Ajay Sohmshetty, Amir Zamir*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64ad7b36b497f375ded2e6f15713ed4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/64ad7b36b497f375ded2e6f15713ed4c-Abstract-Conference.html)

**Abstract**:

When developing deep learning models, we usually decide what task we want to solve then search for a model that generalizes well on the task. An intriguing question would be: what if, instead of fixing the task and searching in the model space, we fix the model and search in the task space? Can we find tasks that the model generalizes on? How do they look, or do they indicate anything? These are the questions we address in this paper. We propose a task discovery framework that automatically finds examples of such tasks via optimizing a generalization-based quantity called agreement score. We demonstrate that one set of images can give rise to many tasks on which neural networks generalize well. These tasks are a reflection of the inductive biases of the learning framework and the statistical patterns present in the data, thus they can make a useful tool for analyzing the neural networks and their biases. As an example, we show that the discovered tasks can be used to automatically create ''adversarial train-test splits'' which make a model fail at test time, without changing the pixels or labels, but by only selecting how the datapoints should be split between the train and test sets. We end with a discussion on human-interpretability of the discovered tasks.

----

## [1142] Iron: Private Inference on Transformers

**Authors**: *Meng Hao, Hongwei Li, Hanxiao Chen, Pengzhi Xing, Guowen Xu, Tianwei Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html)

**Abstract**:

We initiate the study of private inference on Transformer-based models in the client-server setting, where clients have private inputs and servers hold proprietary models. Our main contribution is to provide several new secure protocols for matrix multiplication and complex non-linear functions like Softmax, GELU activations, and LayerNorm, which are critical components of Transformers. Specifically, we first propose a customized homomorphic encryption-based protocol for matrix multiplication that crucially relies on a novel compact packing technique. This design achieves $\sqrt{m} \times$ less communication ($m$ is the number of rows of the output matrix) over the most efficient work. Second, we design efficient protocols for three non-linear functions via integrating advanced underlying protocols and specialized optimizations. Compared to the state-of-the-art protocols, our recipes reduce about half of the communication and computation overhead. Furthermore, all protocols are numerically precise, which preserve the model accuracy of plaintext. These techniques together allow us to implement \Name, an efficient Transformer-based private inference framework. Experiments conducted on several real-world datasets and models demonstrate that \Name achieves $3 \sim 14\times$  less communication  and $3 \sim 11\times$ less runtime compared to the prior art.

----

## [1143] On Batch Teaching with Sample Complexity Bounded by VCD

**Authors**: *Farnam Mansouri, Hans Simon, Adish Singla, Sandra Zilles*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/64e52d01d26ad3914e556eeefb29a8ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/64e52d01d26ad3914e556eeefb29a8ac-Abstract-Conference.html)

**Abstract**:

In machine teaching, a concept is represented by (and inferred from) a small number of labeled examples. Various teaching models in the literature cast the interaction between teacher and learner in a way to obtain a small complexity (in terms of the number of examples required for teaching a concept) while obeying certain constraints that are meant to prevent unfair collusion between teacher and learner. In recent years, one major research goal has been to show interesting relationships between teaching complexity and the VC-dimension (VCD). So far, the only interesting relationship known from batch teaching settings is an upper bound quadratic in the VCD, on a parameter called recursive teaching dimension. The only known upper bound on teaching complexity that is linear in VCD was obtained in a model of teaching with sequences rather than batches.This paper is the first to provide an upper bound of VCD on a batch teaching complexity parameter. This parameter, called STDmin, is introduced here as a model of teaching that intuitively incorporates a notion of ``importance'' of an  example for a concept. In designing the STDmin teaching model, we argue that the standard notion of collusion-freeness from the literature may be inadequate for certain applications; we hence propose three desirable properties of teaching complexity and demonstrate that they are satisfied by STDmin.

----

## [1144] Approaching Quartic Convergence Rates for Quasi-Stochastic Approximation with Application to Gradient-Free Optimization

**Authors**: *Caio Kalil Lauand, Sean P. Meyn*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6530274c68e81047e1f4a2ceb0b8c0ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6530274c68e81047e1f4a2ceb0b8c0ef-Abstract-Conference.html)

**Abstract**:

Stochastic approximation is a foundation for many algorithms found in machine learning and optimization. It is in general slow to converge: the mean square error vanishes as $O(n^{-1})$. A deterministic counterpart known as quasi-stochastic approximation is a viable alternative in many applications, including gradient-free optimization and reinforcement learning. It was assumed in prior research that the optimal achievable convergence rate is $O(n^{-2})$. It is shown in this paper that through design it is possible to obtain far faster convergence, of order $O(n^{-4+\delta})$, with $\delta>0$ arbitrary. Two techniques are introduced for the first time to achieve this rate of convergence. The theory is also specialized within the context of gradient-free optimization, and tested on standard benchmarks. The main results are based on a combination of novel application of results from number theory and techniques adapted from stochastic approximation theory.

----

## [1145] PAC: Assisted Value Factorization with Counterfactual Predictions in Multi-Agent Reinforcement Learning

**Authors**: *Hanhan Zhou, Tian Lan, Vaneet Aggarwal*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65338cfb603d4871a2c38e53a3e039c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65338cfb603d4871a2c38e53a3e039c9-Abstract-Conference.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) has witnessed significant progress with the development of value function factorization methods. It allows optimizing a joint action-value function through the maximization of factorized per-agent utilities. In this paper, we show that in partially observable MARL problems, an agent's ordering over its own actions could impose concurrent constraints (across different states) on the representable function class, causing significant estimation errors during training. We tackle this limitation and propose PAC, a new framework leveraging Assistive information generated from Counterfactual Predictions of optimal joint action selection, which enable explicit assistance to value function factorization through a novel counterfactual loss. A variational inference-based information encoding method is developed to collect and encode the counterfactual predictions from an estimated baseline. To enable decentralized execution, we also derive factorized per-agent policies inspired by a maximum-entropy MARL framework. We evaluate the proposed PAC on multi-agent predator-prey and a set of StarCraft II micromanagement tasks. Empirical results demonstrate improved results of PAC over state-of-the-art value-based and policy-based multi-agent reinforcement learning algorithms on all benchmarks.

----

## [1146] Translation-equivariant Representation in Recurrent Networks with a Continuous Manifold of Attractors

**Authors**: *Wenhao Zhang, Ying Nian Wu, Si Wu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65384a01325fecbd364c835db872443c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65384a01325fecbd364c835db872443c-Abstract-Conference.html)

**Abstract**:

Equivariant representation is necessary for the brain and artificial perceptual systems to faithfully represent the stimulus under some (Lie) group transformations. However, it remains unknown how recurrent neural circuits in the brain represent the stimulus equivariantly, nor the neural representation of abstract group operators. The present study uses a one-dimensional (1D) translation group as an example to explore the general recurrent neural circuit mechanism of the equivariant stimulus representation. We found that a continuous attractor network (CAN), a canonical neural circuit model, self-consistently generates a continuous family of stationary population responses (attractors) that represents the stimulus equivariantly. Inspired by the Drosophila's compass circuit, we found that the 1D translation operators can be represented by extra speed neurons besides the CAN, where speed neurons' responses represent the moving speed (1D translation group parameter), and their feedback connections to the CAN represent the translation generator (Lie algebra). We demonstrated that the network responses are consistent with experimental data. Our model for the first time demonstrates how recurrent neural circuitry in the brain achieves equivariant stimulus representation.

----

## [1147] OpenXAI: Towards a Transparent Evaluation of Model Explanations

**Authors**: *Chirag Agarwal, Satyapriya Krishna, Eshika Saxena, Martin Pawelczyk, Nari Johnson, Isha Puri, Marinka Zitnik, Himabindu Lakkaraju*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/65398a0eba88c9b4a1c38ae405b125ef-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While several types of post hoc explanation methods have been proposed in recent literature, there is very little work on systematically benchmarking these methods. Here, we introduce OpenXAI, a comprehensive and extensible open-source framework for evaluating and benchmarking post hoc explanation methods. OpenXAI comprises of the following key components: (i) a flexible synthetic data generator and a collection of diverse real-world datasets, pre-trained models, and state-of-the-art feature attribution methods, (ii) open-source implementations of twenty-two quantitative metrics for evaluating faithfulness, stability (robustness), and fairness of explanation methods, and (iii) the first ever public XAI leaderboards to readily compare several explanation methods across a wide variety of metrics, models, and datasets. OpenXAI is easily extensible, as users can readily evaluate custom explanation methods and incorporate them into our leaderboards. Overall, OpenXAI provides an automated end-to-end pipeline that not only simplifies and standardizes the evaluation of post hoc explanation methods, but also promotes transparency and reproducibility in benchmarking these methods. While the first release of OpenXAI supports only tabular datasets, the explanation methods and metrics that we consider are general enough to be applicable to other data modalities. OpenXAI datasets and data loaders, implementations of state-of-the-art explanation methods and evaluation metrics, as well as leaderboards are publicly available at https://open-xai.github.io/. OpenXAI will be regularly updated to incorporate text and image datasets, other new metrics and explanation methods, and welcomes inputs from the community.

----

## [1148] Better Best of Both Worlds Bounds for Bandits with Switching Costs

**Authors**: *Idan Amir, Guy Azov, Tomer Koren, Roi Livni*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6590cb829f5ffef50050f3e5845fbb4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6590cb829f5ffef50050f3e5845fbb4c-Abstract-Conference.html)

**Abstract**:

We study best-of-both-worlds algorithms for bandits with switching cost, recently addressed by Rouyer et al., 2021. We introduce a surprisingly simple and effective algorithm that simultaneously achieves minimax optimal regret bound (up to logarithmic factors) of $\mathcal{O}(T^{2/3})$ in the oblivious adversarial setting and a bound of $\mathcal{O}(\min\{\log (T)/\Delta^2,T^{2/3}\})$ in the stochastically-constrained regime, both with (unit) switching costs, where $\Delta$ is the gap between the arms. In the stochastically constrained case, our bound improves over previous results due to Rouyer et al., 2021, that achieved regret of $\mathcal{O}(T^{1/3}/\Delta)$. We accompany our results with a lower bound showing that, in general, $\tilde{\mathcal{\Omega}}(\min\{1/\Delta^2,T^{2/3}\})$ switching cost regret is unavoidable in the stochastically-constrained case for algorithms with $\mathcal{O}(T^{2/3})$ worst-case switching cost regret.

----

## [1149] Uncertainty-Aware Hierarchical Refinement for Incremental Implicitly-Refined Classification

**Authors**: *Jian Yang, Kai Zhu, Kecheng Zheng, Yang Cao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65a723bf7d8dad838c09178270d30e80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65a723bf7d8dad838c09178270d30e80-Abstract-Conference.html)

**Abstract**:

Incremental implicitly-refined classification task aims at assigning hierarchical labels to each sample encountered at different phases. Existing methods tend to fail in generating hierarchy-invariant descriptors when the novel classes are inherited from the old ones. To address the issue, this paper, which explores the inheritance relations in the process of multi-level semantic increment, proposes an Uncertainty-Aware Hierarchical Refinement (UAHR) scheme. Specifically, our proposed scheme consists of a global representation extension strategy that enhances the discrimination of incremental representation by widening the corresponding margin distance, and a hierarchical distribution alignment strategy that refines the distillation process by explicitly determining the inheritance relationship of the incremental class. Particularly, the shifting subclasses are corrected under the guidance of hierarchical uncertainty, ensuring the consistency of the homogeneous features. Extensive experiments on widely used benchmarks (i.e., IIRC-CIFAR, IIRC-ImageNet-lite, IIRC-ImageNet-Subset, and IIRC-ImageNet-full) demonstrate the superiority of our proposed method over the state-of-the-art approaches.

----

## [1150] The Hessian Screening Rule

**Authors**: *Johan Larsson, Jonas Wallin*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65a925049647eab0aa06a9faf1cd470b-Abstract-Conference.html)

**Abstract**:

Predictor screening rules, which discard predictors before fitting a model, have had considerable impact on the speed with which sparse regression problems, such as the lasso, can be solved. In this paper we present a new screening rule for solving the lasso path: the Hessian Screening Rule. The rule uses second-order information from the model to provide both effective screening, particularly in the case of high correlation, as well as accurate warm starts. The proposed rule outperforms all alternatives we study on simulated data sets with both low and high correlation for (\ell_1)-regularized least-squares (the lasso) and logistic regression. It also performs best in general on the real data sets that we examine.

----

## [1151] Sharp Analysis of Stochastic Optimization under Global Kurdyka-Lojasiewicz Inequality

**Authors**: *Ilyas Fatkhullin, Jalal Etesami, Niao He, Negar Kiyavash*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65ae674df2fb642518ae8d2b5435e1b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65ae674df2fb642518ae8d2b5435e1b8-Abstract-Conference.html)

**Abstract**:

We study the complexity of finding the global solution to stochastic nonconvex optimization when the objective function satisfies global Kurdyka-{\L}ojasiewicz (KL) inequality and the queries from stochastic gradient oracles satisfy mild expected smoothness assumption.  We first introduce a general framework to analyze Stochastic Gradient Descent (SGD) and its associated nonlinear dynamics under the setting.  As a byproduct of our analysis, we obtain a sample complexity of  $\mathcal{O}(\epsilon^{-(4-\alpha)/\alpha})$ for SGD when the objective satisfies the so called $\alpha$-P{\L} condition, where $\alpha$ is the degree of gradient domination. Furthermore, we show that a modified SGD with variance reduction and restarting (PAGER) achieves an improved sample complexity of $\mathcal{O}(\epsilon^{-2/\alpha})$ when the objective satisfies the average smoothness assumption. This leads to the first optimal algorithm for the important case of $\alpha=1$ which appears in applications such as policy optimization in reinforcement learning.

----

## [1152] Plan To Predict: Learning an Uncertainty-Foreseeing Model For Model-Based Reinforcement Learning

**Authors**: *Zifan Wu, Chao Yu, Chen Chen, Jianye Hao, Hankz Hankui Zhuo*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65beb73449888fabcf601b3a3ef4b3a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65beb73449888fabcf601b3a3ef4b3a7-Abstract-Conference.html)

**Abstract**:

In Model-based Reinforcement Learning (MBRL), model learning is critical since an inaccurate model can bias policy learning via generating misleading samples. However, learning an accurate model can be difficult since the policy is continually updated and the induced distribution over visited states used for model learning shifts accordingly. Prior methods alleviate this issue by quantifying the uncertainty of model-generated samples. However, these methods only quantify the uncertainty passively after the samples were generated, rather than foreseeing the uncertainty before model trajectories fall into those highly uncertain regions. The resulting low-quality samples can induce unstable learning targets and hinder the optimization of the policy. Moreover, while being learned to minimize one-step prediction errors, the model is generally used to predict for multiple steps, leading to a mismatch between the objectives of model learning and model usage. To this end, we propose Plan To Predict (P2P), an MBRL framework that treats the model rollout process as a sequential decision making problem by reversely considering the model as a decision maker and the current policy as the dynamics. In this way, the model can quickly adapt to the current policy and foresee the multi-step future uncertainty when generating trajectories. Theoretically, we show that the performance of P2P can be guaranteed by approximately optimizing a lower bound of the true environment return. Empirical results demonstrate that P2P achieves state-of-the-art performance on several challenging benchmark tasks.

----

## [1153] Co-Modality Graph Contrastive Learning for Imbalanced Node Classification

**Authors**: *Yiyue Qian, Chunhui Zhang, Yiming Zhang, Qianlong Wen, Yanfang Ye, Chuxu Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65cbe3e21ac62553111d9ecf7d60c18e-Abstract-Conference.html)

**Abstract**:

Graph contrastive learning (GCL), leveraging graph augmentations to convert graphs into different views and further train graph neural networks (GNNs), has achieved considerable success on graph benchmark datasets. Yet, there are still some gaps in directly applying existing GCL methods to real-world data. First, handcrafted graph augmentations require trials and errors, but still can not yield consistent performance on multiple tasks. Second, most real-world graph data present class-imbalanced distribution but existing GCL methods are not immune to data imbalance. Therefore, this work proposes to explicitly tackle these challenges, via a principled framework called \textit{\textbf{C}o-\textbf{M}odality \textbf{G}raph \textbf{C}ontrastive \textbf{L}earning} (\textbf{CM-GCL}) to automatically generate contrastive pairs and further learn balanced representation over unlabeled data. Specifically, we design inter-modality GCL to automatically generate contrastive pairs (e.g., node-text) based on rich node content. Inspired by the fact that minority samples can be ``forgotten'' by pruning deep neural networks, we naturally extend network pruning to our GCL framework for mining minority nodes. Based on this, we co-train two pruned encoders (e.g., GNN and text encoder) in different modalities by pushing the corresponding node-text pairs together and the irrelevant node-text pairs away. Meanwhile, we propose intra-modality GCL by co-training non-pruned GNN and pruned GNN, to ensure node embeddings with similar attribute features stay closed. Last, we fine-tune the GNN encoder on downstream class-imbalanced node classification tasks. Extensive experiments demonstrate that our model significantly outperforms state-of-the-art baseline models and learns more balanced representations on real-world graphs. Our source code is available at https://github.com/graphprojects/CM-GCL.

----

## [1154] Efficiency Ordering of Stochastic Gradient Descent

**Authors**: *Jie Hu, Vishwaraj Doshi, Do Young Eun*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65ccdfe02045fa0b823c5fa7ffd56b66-Abstract-Conference.html)

**Abstract**:

We consider the stochastic gradient descent (SGD) algorithm driven by a general stochastic sequence, including i.i.d noise and random walk on an arbitrary graph, among others; and analyze it in the asymptotic sense. Specifically, we employ the notion of `efficiency ordering', a well-analyzed tool for comparing the performance of Markov Chain Monte Carlo (MCMC) samplers, for SGD algorithms in the form of Loewner ordering of covariance matrices associated with the scaled iterate errors in the long term. Using this ordering, we show that input sequences that are more efficient for MCMC sampling also lead to smaller covariance of the errors for SGD algorithms in the limit. This also suggests that an arbitrarily weighted MSE of SGD iterates in the limit becomes smaller when driven by more efficient chains. Our finding is of particular interest in applications such as decentralized optimization and swarm learning, where SGD is implemented in a random walk fashion on the underlying communication graph for cost issues and/or data privacy. We demonstrate how certain non-Markovian processes, for which typical mixing-time based non-asymptotic bounds are intractable, can outperform their Markovian counterparts in the sense of efficiency ordering for SGD. We show the utility of our method by applying it to gradient descent with shuffling and mini-batch gradient descent, reaffirming key results from existing literature under a unified framework. Empirically, we also observe efficiency ordering for variants of SGD such as accelerated SGD and Adam, open up the possibility of extending our notion of efficiency ordering to a broader family of stochastic optimization algorithms.

----

## [1155] Muffliato: Peer-to-Peer Privacy Amplification for Decentralized Optimization and Averaging

**Authors**: *Edwige Cyffers, Mathieu Even, Aurélien Bellet, Laurent Massoulié*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65d32185f73cbf4535449a792c63926f-Abstract-Conference.html)

**Abstract**:

Decentralized optimization is increasingly popular in machine learning for its scalability and efficiency. Intuitively, it should also provide better privacy guarantees, as nodes only observe the messages sent by their neighbors in the network graph. But formalizing and quantifying this gain is challenging: existing results are typically limited to Local Differential Privacy (LDP) guarantees that overlook the advantages of decentralization. In this work, we introduce pairwise network differential privacy, a relaxation of LDP that captures the fact that the privacy leakage from a node u to a node v may depend on their relative position in the graph. We then analyze the combination of local noise injection with (simple or randomized) gossip averaging protocols on fixed and random communication graphs. We also derive a differentially private decentralized optimization algorithm that alternates between local gradient descent steps and gossip averaging. Our results show that our algorithms amplify privacy guarantees as a function of the distance between nodes in the graph, matching the privacy-utility trade-off of the trusted curator, up to factors that explicitly depend on the graph topology. Remarkably, these factors become constant for expander graphs. Finally, we illustrate our privacy gains with experiments on synthetic and real-world datasets.

----

## [1156] Mining Multi-Label Samples from Single Positive Labels

**Authors**: *Youngin Cho, Daejin Kim, Mohammad Azam Khan, Jaegul Choo*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/65f54fdf62cd5614dc5715ae7ece4ef6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/65f54fdf62cd5614dc5715ae7ece4ef6-Abstract-Conference.html)

**Abstract**:

Conditional generative adversarial networks (cGANs) have shown superior results in class-conditional generation tasks. To simultaneously control multiple conditions, cGANs require multi-label training datasets, where multiple labels can be assigned to each data instance. Nevertheless, the tremendous annotation cost limits the accessibility of multi-label datasets in real-world scenarios. Therefore, in this study we explore the practical setting called the single positive setting, where each data instance is annotated by only one positive label with no explicit negative labels. To generate multi-label data in the single positive setting, we propose a novel sampling approach called single-to-multi-label (S2M) sampling, based on the Markov chain Monte Carlo method. As a widely applicable “add-on” method, our proposed S2M sampling method enables existing unconditional and conditional GANs to draw high-quality multi-label data with a minimal annotation cost. Extensive experiments on real image datasets verify the effectiveness and correctness of our method, even when compared to a model trained with fully annotated datasets.

----

## [1157] Sequential Information Design: Learning to Persuade in the Dark

**Authors**: *Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti, Francesco Trovò*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6604fbf7548524576c9ee2e30b0d5122-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6604fbf7548524576c9ee2e30b0d5122-Abstract-Conference.html)

**Abstract**:

We study a repeated information design problem faced by an informed sender who tries to influence the behavior of a self-interested receiver. We consider settings where the receiver faces a sequential decision making (SDM) problem. At each round, the sender observes the realizations of random events in the SDM problem. This begets the challenge of how to incrementally disclose such information to the receiver to persuade them to follow (desirable) action recommendations. We study the case in which the sender does not know random events probabilities, and, thus, they have to gradually learn them while persuading the receiver. Our goal is to design online learning algorithms that are no-regret for the sender, while at the same time being persuasive for the receiver. We start by providing a non-trivial polytopal approximation of the set of sender's persuasive information structures. This is crucial to design efficient learning algorithms. Next, we prove a negative result: no learning algorithm can be persuasive. Thus, we relax persuasiveness requirements by focusing on algorithms that guarantee that the receiver's regret in following recommendations grows sub-linearly. In the full-feedback setting---where the sender observes all random events realizations---, we provide an algorithm with $\tilde{O}(\sqrt{T})$ regret for both the sender and the receiver. Instead, in the bandit-feedback setting---where the sender only observes the realizations of random events actually occurring in the SDM problem---, we design an algorithm that, given an $\alpha \in [1/2, 1]$ as input, ensures $\tilde{O}({T^\alpha})$ and $\tilde{O}( T^{\max \{ \alpha, 1-\frac{\alpha}{2} \} })$ regrets for the sender and the receiver, respectively. This result is complemented by a lower bound showing that such a regrets trade-off is essentially tight.

----

## [1158] AutoML Two-Sample Test

**Authors**: *Jonas M. Kübler, Vincent Stimper, Simon Buchholz, Krikamol Muandet, Bernhard Schölkopf*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66247b78cb1aa7259dcf856a18c9e294-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66247b78cb1aa7259dcf856a18c9e294-Abstract-Conference.html)

**Abstract**:

Two-sample tests are important in statistics and machine learning, both as tools for scientific discovery as well as to detect distribution shifts.This led to the development of many sophisticated test procedures going beyond the standard supervised learning frameworks, whose usage can require specialized knowledge about two-sample testing. We use a simple test that takes the mean discrepancy of a witness function as the test statistic and prove that minimizing a squared loss leads to a witness with optimal testing power. This allows us to leverage recent advancements in AutoML. Without any user input about the problems at hand, and using the same method for all our experiments, our AutoML two-sample test achieves competitive performance on a diverse distribution shift benchmark as well as on challenging two-sample testing problems.

----

## [1159] Cross-Linked Unified Embedding for cross-modality representation learning

**Authors**: *Xinming Tu, Zhi-Jie Cao, Chenrui Xia, Sara Mostafavi, Ge Gao*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/662b1774ba8845fc1fa3d1fc0177ceeb-Abstract-Conference.html)

**Abstract**:

Multi-modal learning is essential for understanding information in the real world. Jointly learning from multi-modal data enables global integration of both shared and modality-specific information, but current strategies often fail when observa- tions from certain modalities are incomplete or missing for part of the subjects. To learn comprehensive representations based on such modality-incomplete data, we present a semi-supervised neural network model called CLUE (Cross-Linked Unified Embedding). Extending from multi-modal VAEs, CLUE introduces the use of cross-encoders to construct latent representations from modality-incomplete observations. Representation learning for modality-incomplete observations is common in genomics. For example, human cells are tightly regulated across multi- ple related but distinct modalities such as DNA, RNA, and protein, jointly defining a cell’s function. We benchmark CLUE on multi-modal data from single cell measurements, illustrating CLUE’s superior performance in all assessed categories of the NeurIPS 2021 Multimodal Single-cell Data Integration Competition. While we focus on analysis of single cell genomic datasets, we note that the proposed cross-linked embedding strategy could be readily applied to other cross-modality representation learning problems.

----

## [1160] Learning Individualized Treatment Rules with Many Treatments: A Supervised Clustering Approach Using Adaptive Fusion

**Authors**: *Haixu Ma, Donglin Zeng, Yufeng Liu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/663865ea167425c6c562cb0b6bcf76c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/663865ea167425c6c562cb0b6bcf76c7-Abstract-Conference.html)

**Abstract**:

Learning an optimal Individualized Treatment Rule (ITR) is a very important problem in precision medicine. This paper is concerned with the challenge when the number of treatment arms is large, and some groups of treatments in the large treatment space may work similarly for the patients. Motivated by the recent development of supervised clustering, we propose a novel adaptive fusion based method to cluster the treatments with similar treatment effects together and estimate the optimal ITR simultaneously through a single convex optimization. The problem is formulated as balancing \textit{loss}$+$\textit{penalty} terms with a tuning parameter, which allows the entire solution path of the treatment clustering process to be clearly visualized hierarchically. For computation, we propose an efficient  algorithm based on  accelerated proximal gradient and further conduct a novel group-lasso based algorithm for variable selection to boost the performance. Moreover, we demonstrate the theoretical guarantee of recovering the underlying true clustering structure of the treatments for our method. Finally, we demonstrate the superior performance of our method via both simulations and a real data application on cancer treatment, which may assist the decision making process for doctors.

----

## [1161] Robust Generalized Method of Moments: A Finite Sample Viewpoint

**Authors**: *Dhruv Rohatgi, Vasilis Syrgkanis*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66562bf632d45e83232437afaf2aa92b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66562bf632d45e83232437afaf2aa92b-Abstract-Conference.html)

**Abstract**:

For many inference problems in statistics and econometrics, the unknown parameter is identified by a set of moment conditions. A generic method of solving moment conditions is the Generalized Method of Moments (GMM). However, classical GMM estimation is potentially very sensitive to outliers. Robustified GMM estimators have been developed in the past, but suffer from several drawbacks: computational intractability, poor dimension-dependence, and no quantitative recovery guarantees in the presence of a constant fraction of outliers. In this work, we develop the first computationally efficient GMM estimator (under intuitive assumptions) that can tolerate a constant $\epsilon$ fraction of adversarially corrupted samples, and that has an $\ell_2$ recovery guarantee of $O(\sqrt{\epsilon})$. To achieve this, we draw upon and extend a recent line of work on algorithmic robust statistics for related but simpler problems such as mean estimation, linear regression and stochastic optimization. As a special case, we apply our algorithm to instrumental variables linear regression with heterogeneous treatment effects, and experimentally demonstrate that it can tolerate as much as $10$ -- $15\%$ corruption, significantly improving upon baseline methods.

----

## [1162] $k$-Sliced Mutual Information: A Quantitative Study of Scalability with Dimension

**Authors**: *Ziv Goldfeld, Kristjan H. Greenewald, Theshani Nuradha, Galen Reeves*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6662eaf34893d9827ddf60c29e9ad6af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6662eaf34893d9827ddf60c29e9ad6af-Abstract-Conference.html)

**Abstract**:

Sliced mutual information (SMI) is defined as an average of mutual information (MI) terms between one-dimensional random projections of the random variables. It serves as a surrogate measure of dependence to classic MI that preserves many of its properties but is more scalable to high dimensions. However, a quantitative characterization of how SMI itself and estimation rates thereof depend on the ambient dimension, which is crucial to the understanding of scalability, remain obscure. This work provides a multifaceted account of the dependence of SMI on dimension, under a broader framework termed $k$-SMI, which considers projections to $k$-dimensional subspaces. Using a new result on the continuity of differential entropy in the 2-Wasserstein metric, we derive sharp bounds on the error of Monte Carlo (MC)-based estimates of $k$-SMI, with explicit dependence on $k$ and the ambient dimension, revealing their interplay with the number of samples. We then combine the MC integrator with the neural estimation framework to provide an end-to-end $k$-SMI estimator, for which optimal convergence rates are established. We also explore asymptotics of the population $k$-SMI as dimension grows, providing Gaussian approximation results with a residual that decays under appropriate moment bounds. All our results trivially apply to SMI by setting $k=1$. Our theory is validated with numerical experiments and is applied to sliced InfoGAN, which altogether provide a comprehensive quantitative account of the scalability question of $k$-SMI, including SMI as a special case when $k=1$.

----

## [1163] What's the Harm? Sharp Bounds on the Fraction Negatively Affected by Treatment

**Authors**: *Nathan Kallus*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/666cccc6376058e251315b4de7e085b9-Abstract-Conference.html)

**Abstract**:

The fundamental problem of causal inference -- that we never observe counterfactuals -- prevents us from identifying how many might be negatively affected by a proposed intervention. If, in an A/B test, half of users click (or buy, or watch, or renew, etc.), whether exposed to the standard experience A or a new one B, hypothetically it could be because the change affects no one,  because the change positively affects half the user population to go from no-click to click while negatively affecting the other half, or something in between. While unknowable, this impact is clearly of material importance to the decision to implement a change or not, whether due to fairness, long-term, systemic, or operational considerations. We therefore derive the tightest-possible (i.e., sharp) bounds on the fraction negatively affected (and other related estimands) given data with only factual observations, whether experimental or observational. Naturally, the more we can stratify individuals by observable covariates, the tighter the sharp bounds. Since these bounds involve unknown functions that must be learned from data, we develop a robust inference algorithm that is efficient almost regardless of how and how fast these functions are learned, remains consistent when some are mislearned, and still gives valid conservative bounds when most are mislearned. Our methodology altogether therefore strongly supports credible conclusions: it avoids spuriously point-identifying this unknowable impact, focusing on the best bounds instead, and it permits exceedingly robust inference on these. We demonstrate our method in simulation studies and in a case study of career counseling for the unemployed.

----

## [1164] Bessel Equivariant Networks for Inversion of Transmission Effects in Multi-Mode Optical Fibres

**Authors**: *Joshua Mitton, Simon Peter Mekhail, Miles J. Padgett, Daniele Faccio, Marco Aversa, Roderick Murray-Smith*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html)

**Abstract**:

We develop a new type of model for solving the task of inverting the transmission effects of multi-mode optical fibres through the construction of an $\mathrm{SO}^{+}(2,1)$-equivariant neural network. This model takes advantage of the of the azimuthal correlations known to exist in fibre speckle patterns and naturally accounts for the difference in spatial arrangement between input and speckle patterns. In addition, we use a second post-processing network to remove circular artifacts, fill gaps, and sharpen the images, which is required due to the nature of optical fibre transmission. This two stage approach allows for the inspection of the predicted images produced by the more robust physically motivated equivariant model, which could be useful in a safety-critical application, or by the output of both models, which produces high quality images. Further, this model can scale to previously unachievable resolutions of imaging with multi-mode optical fibres and is demonstrated on $256 \times 256$ pixel images. This is a result of improving the trainable parameter requirement from $\mathcal{O}(N^4)$ to $\mathcal{O}(m)$, where $N$ is pixel size and $m$ is number of fibre modes. Finally, this model generalises to new images, outside of the set of training data classes, better than previous models.

----

## [1165] Training Subset Selection for Weak Supervision

**Authors**: *Hunter Lang, Aravindan Vijayaraghavan, David A. Sontag*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66720ca4e5a09ff83b55a117a6b2a86c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66720ca4e5a09ff83b55a117a6b2a86c-Abstract-Conference.html)

**Abstract**:

Existing weak supervision approaches use all the data covered by weak signals to train a classifier.  We show both theoretically and empirically that this is not always optimal.  Intuitively, there is a tradeoff between the amount of weakly-labeled data and the precision of the weak labels. We explore this tradeoff by combining pretrained data representations with the cut statistic to select (hopefully) high-quality subsets of the weakly-labeled training data. Subset selection applies to any label model and classifier and is very simple to plug in to existing weak supervision pipelines, requiring just a few lines of code. We show our subset selection method improves the performance of weak supervision for a wide range of label models, classifiers, and datasets.  Using less weakly-labeled data improves the accuracy of weak supervision pipelines by up to 19% (absolute) on benchmark tasks.

----

## [1166] Expansion and Shrinkage of Localization for Weakly-Supervised Semantic Segmentation

**Authors**: *Jinlong Li, Zequn Jie, Xu Wang, Xiaolin Wei, Lin Ma*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66738d21d3cddb8717ca52deff5a5546-Abstract-Conference.html)

**Abstract**:

Generating precise class-aware pseudo ground-truths, a.k.a, class activation maps (CAMs), is essential for Weakly-Supervised Semantic Segmentation. The original CAM method usually produces incomplete and inaccurate localization maps. To tackle with this issue, this paper proposes an Expansion and Shrinkage scheme based on the offset learning in the deformable convolution, to sequentially improve the recall and precision of the located object in the two respective stages. In the Expansion stage, an offset learning branch in a deformable convolution layer, referred to as expansion sampler'', seeks to sample increasingly less discriminative object regions, driven by an inverse supervision signal that maximizes image-level classification loss. The located more complete object region in the Expansion stage is then gradually narrowed down to the final object region during the Shrinkage stage. In the Shrinkage stage, the offset learning branch of another deformable convolution layer referred to as theshrinkage sampler'', is introduced to exclude the false positive background regions attended in the Expansion stage to improve the precision of the localization maps. We conduct various experiments on PASCAL VOC 2012 and MS COCO 2014 to well demonstrate the superiority of our method over other state-of-the-art methods for Weakly-Supervised Semantic Segmentation. The code is available at https://github.com/TyroneLi/ESOL_WSSS.

----

## [1167] No Free Lunch from Deep Learning in Neuroscience: A Case Study through Models of the Entorhinal-Hippocampal Circuit

**Authors**: *Rylan Schaeffer, Mikail Khona, Ila Fiete*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66808849a9f5d8e2d00dbdc844de6333-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66808849a9f5d8e2d00dbdc844de6333-Abstract-Conference.html)

**Abstract**:

Research in Neuroscience, as in many scientific disciplines, is undergoing a renaissance based on deep learning. Unique to Neuroscience, deep learning models can be used not only as a tool but interpreted as models of the brain. The central claims of recent deep learning-based models of brain circuits are that they make novel predictions about neural phenomena or shed light on the fundamental functions being optimized. We show, through the case-study of grid cells in the entorhinal-hippocampal circuit, that one may get neither. We begin by reviewing the principles of grid cell mechanism and function obtained from first-principles modeling efforts, then rigorously examine the claims of deep learning models of grid cells. Using large-scale architectural and hyperparameter sweeps and theory-driven experimentation, we demonstrate that the results of such models may be more strongly driven by particular, non-fundamental, and post-hoc implementation choices than fundamental truths about neural circuits or the loss function(s) they might optimize. We discuss why these models cannot be expected to produce accurate models of the brain without the addition of substantial amounts of inductive bias, an informal No Free Lunch result for Neuroscience. Based on first principles work, we provide hypotheses for what additional loss functions will produce grid cells more robustly. In conclusion, circumspection and transparency, together with biological knowledge, are warranted in building and interpreting deep learning models in Neuroscience.

----

## [1168] On the Importance of Gradient Norm in PAC-Bayesian Bounds

**Authors**: *Itai Gat, Yossi Adi, Alexander G. Schwing, Tamir Hazan*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6686e3f2e31a0db5bf90ab1cc2272b72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6686e3f2e31a0db5bf90ab1cc2272b72-Abstract-Conference.html)

**Abstract**:

Generalization bounds which assess the difference between the true risk and the empirical risk have been studied extensively. However, to obtain bounds, current techniques use strict assumptions such as a uniformly bounded or a Lipschitz loss function. To avoid these assumptions, in this paper, we follow an alternative approach: we relax uniform bounds assumptions by using on-average bounded loss and on-average bounded gradient norm assumptions. Following this relaxation, we propose a new generalization bound that exploits the contractivity of the log-Sobolev inequalities. These inequalities add an additional loss-gradient norm term to the generalization bound, which is intuitively a surrogate of the model complexity. We apply the proposed bound on Bayesian deep nets and empirically analyze the effect of this new loss-gradient norm term on different neural architectures.

----

## [1169] RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning

**Authors**: *Marc Rigter, Bruno Lacerda, Nick Hawes*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6691c5e4a199b72dffd9c90acb63bcd6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6691c5e4a199b72dffd9c90acb63bcd6-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) aims to find performant policies from logged data without further environment interaction. Model-based algorithms, which learn a model of the environment from the dataset and perform conservative policy optimisation within that model, have emerged as a promising approach to this problem. In this work, we present Robust Adversarial Model-Based Offline RL (RAMBO), a novel approach to model-based offline RL. We formulate the problem as a two-player zero sum game against an adversarial environment model. The model is trained to minimise the value function while still accurately predicting the transitions in the dataset, forcing the policy to act conservatively in areas not covered by the dataset. To approximately solve the two-player game, we alternate between optimising the policy and adversarially optimising the model. The problem formulation that we address is theoretically grounded, resulting in a probably approximately correct (PAC) performance guarantee and a pessimistic value function which lower bounds the value function in the true environment. We evaluate our approach on widely studied offline RL benchmarks, and demonstrate that it outperforms existing state-of-the-art baselines.

----

## [1170] Learning NP-Hard Multi-Agent Assignment Planning using GNN: Inference on a Random Graph and Provable Auction-Fitted Q-learning

**Authors**: *Hyunwook Kang, Taehwan Kwon, Jinkyoo Park, James R. Morrison*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66ad22a4a1d2e6fe6f6f6581fadeedbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66ad22a4a1d2e6fe6f6f6581fadeedbc-Abstract-Conference.html)

**Abstract**:

This paper explores the possibility of near-optimally solving multi-agent, multi-task NP-hard planning problems with time-dependent rewards using a learning-based algorithm. In particular, we consider a class of robot/machine scheduling problems called the multi-robot reward collection problem (MRRC). Such MRRC problems well model ride-sharing, pickup-and-delivery, and a variety of related problems. In representing the MRRC problem as a sequential decision-making problem, we observe that each state can be represented as an extension of probabilistic graphical models (PGMs), which we refer to as random PGMs. We then develop a mean-field inference method for random PGMs. We then propose (1) an order-transferable Q-function estimator and (2) an order-transferability-enabled auction to select a joint assignment in polynomial-time. These result in a reinforcement learning framework with at least $1-1/e$ optimality. Experimental results on solving MRRC problems highlight the near-optimality and transferability of the proposed methods. We also consider identical parallel machine scheduling problems (IPMS) and minimax multiple traveling salesman problems (minimax-mTSP).

----

## [1171] Improved techniques for deterministic l2 robustness

**Authors**: *Sahil Singla, Soheil Feizi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66c9de41210338c9581d5313125b7486-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66c9de41210338c9581d5313125b7486-Abstract-Conference.html)

**Abstract**:

Training convolutional neural networks (CNNs) with a strict 1-Lipschitz constraint under the l{2} norm is useful for adversarial robustness, interpretable gradients and stable training. 1-Lipschitz CNNs are usually designed by enforcing each layer to have an orthogonal Jacobian matrix (for all inputs) to prevent the gradients from vanishing during backpropagation. However, their performance often significantly lags behind that of heuristic methods to enforce Lipschitz constraints where the resulting CNN is not provably 1-Lipschitz. In this work, we reduce this gap by introducing (a) a procedure to certify robustness of 1-Lipschitz CNNs by replacing the last linear layer with a 1-hidden layer MLP that significantly improves their performance for both standard and provably robust accuracy, (b) a method to significantly reduce the training time per epoch for Skew Orthogonal Convolution (SOC) layers (>30\% reduction for deeper networks) and (c) a class of pooling layers using the mathematical property that the l{2} distance of an input to a manifold is 1-Lipschitz. Using these methods, we significantly advance the state-of-the-art for standard and provable robust accuracies on CIFAR-10 (gains of  +1.79\% and +3.82\%) and similarly on CIFAR-100 (+3.78\% and +4.75\% across all networks.

----

## [1172] Normalizing Flows for Knockoff-free Controlled Feature Selection

**Authors**: *Derek Hansen, Brian Manzo, Jeffrey Regier*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66f09010d989c83faeeac2617464b6a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66f09010d989c83faeeac2617464b6a4-Abstract-Conference.html)

**Abstract**:

Controlled feature selection aims to discover the features a response depends on while limiting the false discovery rate (FDR) to a predefined level. Recently, multiple deep-learning-based methods have been proposed to perform controlled feature selection through the Model-X knockoff framework. We demonstrate, however, that these methods often fail to control the FDR for two reasons. First, these methods often learn inaccurate models of features. Second, the "swap" property, which is required for knockoffs to be valid, is often not well enforced. We propose a new procedure called FlowSelect to perform controlled feature selection that does not suffer from either of these two problems. To more accurately model the features, FlowSelect uses normalizing flows, the state-of-the-art method for density estimation. Instead of enforcing the "swap" property, FlowSelect uses a novel MCMC-based procedure to calculate p-values for each feature directly. Asymptotically, FlowSelect computes valid p-values. Empirically, FlowSelect consistently controls the FDR on both synthetic and semi-synthetic benchmarks, whereas competing knockoff-based approaches do not. FlowSelect also demonstrates greater power on these benchmarks. Additionally, FlowSelect correctly infers the genetic variants associated with specific soybean traits from GWAS data.

----

## [1173] ReFactor GNNs: Revisiting Factorisation-based Models from a Message-Passing Perspective

**Authors**: *Yihong Chen, Pushkar Mishra, Luca Franceschi, Pasquale Minervini, Pontus Stenetorp, Sebastian Riedel*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/66f7a3df255c47b2e72f30b310a7e44a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/66f7a3df255c47b2e72f30b310a7e44a-Abstract-Conference.html)

**Abstract**:

Factorisation-based Models (FMs), such as DistMult, have enjoyed enduring success for Knowledge Graph Completion (KGC) tasks, often outperforming Graph Neural Networks (GNNs). However, unlike GNNs, FMs struggle to incorporate node features and generalise to unseen nodes in inductive settings. Our work bridges the gap between FMs and GNNs by proposing ReFactor GNNs. This new architecture draws upon $\textit{both}$ modelling paradigms, which previously were largely thought of as disjoint. Concretely, using a message-passing formalism, we show how FMs can be cast as GNNs by reformulating the gradient descent procedure as message-passing operations, which forms the basis of our ReFactor GNNs. Across a multitude of well-established KGC benchmarks, our ReFactor GNNs achieve comparable transductive performance to FMs, and state-of-the-art inductive performance while using an order of magnitude fewer parameters.

----

## [1174] Efficient Architecture Search for Diverse Tasks

**Authors**: *Junhong Shen, Mikhail Khodak, Ameet Talwalkar*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6724eae98f3917968d54c193ac0b45f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6724eae98f3917968d54c193ac0b45f1-Abstract-Conference.html)

**Abstract**:

While neural architecture search (NAS) has enabled automated machine learning (AutoML) for well-researched areas, its application to tasks beyond computer vision is still under-explored. As less-studied domains are precisely those where we expect AutoML to have the greatest impact, in this work we study NAS for efficiently solving diverse problems. Seeking an approach that is fast, simple, and broadly applicable, we fix a standard convolutional network (CNN) topology and propose to search for the right kernel sizes and dilations its operations should take on. This dramatically expands the model's capacity to extract features at multiple resolutions for different types of data while only requiring search over the operation space. To overcome the efficiency challenges of naive weight-sharing in this search space, we introduce DASH, a differentiable NAS algorithm that computes the mixture-of-operations using the Fourier diagonalization of convolution, achieving both a better asymptotic complexity and an up-to-10x search time speedup in practice. We evaluate DASH on ten tasks spanning a variety of application domains such as PDE solving, protein folding, and heart disease detection. DASH outperforms state-of-the-art AutoML methods in aggregate, attaining the best-known automated performance on seven tasks. Meanwhile, on six of the ten tasks, the combined search and retraining time is less than 2x slower than simply training a CNN backbone that is far less accurate.

----

## [1175] Beyond Not-Forgetting: Continual Learning with Backward Knowledge Transfer

**Authors**: *Sen Lin, Li Yang, Deliang Fan, Junshan Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6728fcf94660c59c938319a6833a6073-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6728fcf94660c59c938319a6833a6073-Abstract-Conference.html)

**Abstract**:

By learning a sequence of tasks continually, an agent in continual learning (CL) can improve the learning performance of both a new task and `old' tasks by leveraging the forward knowledge transfer and the backward knowledge transfer, respectively. However, most existing CL methods focus on addressing catastrophic forgetting in neural networks by minimizing the modification of the learnt model for old tasks. This inevitably limits the backward knowledge transfer from the new task to the old tasks, because judicious model updates could possibly improve the learning performance of the old tasks as well. To tackle this problem, we first theoretically analyze the conditions under which updating the learnt model of old tasks could be beneficial for CL and also lead to backward knowledge transfer, based on the gradient projection onto the input subspaces of old tasks. Building on the theoretical analysis, we next develop a ContinUal learning method with Backward knowlEdge tRansfer (CUBER), for a fixed capacity neural network without data replay. In particular, CUBER first characterizes the task correlation to identify the positively correlated old tasks in a layer-wise manner, and then selectively modifies the learnt model of the old tasks when learning the new task. Experimental studies show that CUBER can even achieve positive backward knowledge transfer on several existing CL benchmarks for the first time without data replay, where the related baselines still suffer from catastrophic forgetting (negative backward knowledge transfer). The superior performance of CUBER on the backward knowledge transfer also leads to higher accuracy accordingly.

----

## [1176] Inherently Explainable Reinforcement Learning in Natural Language

**Authors**: *Xiangyu Peng, Mark O. Riedl, Prithviraj Ammanabrolu*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/672e44a114a41d5f34b97459877c083d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/672e44a114a41d5f34b97459877c083d-Abstract-Conference.html)

**Abstract**:

We focus on the task of creating a reinforcement learning agent that is inherently explainable---with the ability to produce immediate local explanations by thinking out loud while performing a task and analyzing entire trajectories post-hoc to produce temporally extended explanations. This Hierarchically Explainable Reinforcement Learning agent (HEX-RL), operates in Interactive Fictions, text-based game environments in which an agent perceives and acts upon the world using textual natural language. These games are usually structured as puzzles or quests with long-term dependencies in which an agent must complete a sequence of actions to succeed---providing ideal environments in which to test an agent's ability to explain its actions. Our agent is designed to treat explainability as a first-class citizen, using an extracted symbolic knowledge graph-based state representation coupled with a Hierarchical Graph Attention mechanism that points to the facts in the internal graph representation that most influenced the choice of actions. Experiments show that this agent provides significantly improved explanations over strong baselines, as rated by human participants generally unfamiliar with the environment, while also matching state-of-the-art task performance.

----

## [1177] Conditional Independence Testing with Heteroskedastic Data and Applications to Causal Discovery

**Authors**: *Wiebke Günther, Urmi Ninad, Jonas Wahl, Jakob Runge*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6739d8df16b5bce3587ca5f18662a6aa-Abstract-Conference.html)

**Abstract**:

Conditional independence (CI) testing is frequently used in data analysis and machine learning for various scientific fields and it forms the basis of constraint-based causal discovery. Oftentimes, CI testing relies on strong, rather unrealistic assumptions. One of these assumptions is homoskedasticity, in other words, a constant conditional variance is assumed. We frame heteroskedasticity in a structural causal model framework and present an adaptation of the partial correlation CI test that works well in the presence of heteroskedastic noise, given that expert knowledge about the heteroskedastic relationships is available. Further, we provide theoretical consistency results for the proposed CI test which carry over to causal discovery under certain assumptions. Numerical causal discovery experiments demonstrate that the adapted partial correlation CI test outperforms the standard test in the presence of  heteroskedasticity and is on par for the homoskedastic case. Finally, we discuss the general challenges and limits as to how expert knowledge about heteroskedasticity can be accounted for in causal discovery.

----

## [1178] On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting

**Authors**: *Tomasz Korbak, Hady Elsahar, Germán Kruszewski, Marc Dymetman*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67496dfa96afddab795530cc7c69b57a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67496dfa96afddab795530cc7c69b57a-Abstract-Conference.html)

**Abstract**:

The availability of large pre-trained models is changing the landscape of Machine Learning research and practice, moving from a "training from scratch" to a "fine-tuning'' paradigm. While in some applications the goal is to "nudge'' the pre-trained distribution towards preferred outputs, in others it is to steer it towards a different distribution over the sample space. Two main paradigms have emerged to tackle this challenge: Reward Maximization (RM) and, more recently, Distribution Matching (DM). RM applies standard Reinforcement Learning (RL) techniques, such as Policy Gradients, to gradually increase the reward signal. DM prescribes to first make explicit the target distribution that the model is fine-tuned to approximate. Here we explore the theoretical connections between the two paradigms and show that methods such as KL-control developed in the RM paradigm can also be construed as belonging to DM. We further observe that while DM differs from RM, it can suffer from similar training difficulties, such as high gradient variance. We leverage connections between the two paradigms to import the concept of baseline into DM methods. We empirically validate the benefits of adding a baseline on an array of controllable language generation tasks such as constraining topic, sentiment, and gender distributions in texts sampled from a language model. We observe superior performance in terms of constraint satisfaction, stability, and sample efficiency.

----

## [1179] Ask4Help: Learning to Leverage an Expert for Embodied Tasks

**Authors**: *Kunal Pratap Singh, Luca Weihs, Alvaro Herrasti, Jonghyun Choi, Aniruddha Kembhavi, Roozbeh Mottaghi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/674ad201bc8fa74b3c9979230aa0c63b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/674ad201bc8fa74b3c9979230aa0c63b-Abstract-Conference.html)

**Abstract**:

Embodied AI agents continue to become more capable every year with the advent of new models, environments, and benchmarks, but are still far away from being performant and reliable enough to be deployed in real, user-facing, applications. In this paper, we ask: can we bridge this gap by enabling agents to ask for assistance from an expert such as a human being? To this end, we propose the Ask4Help policy that augments agents with the ability to request, and then use expert assistance. Ask4Help policies can be efficiently trained without modifying the original agent's parameters and learn a desirable trade-off between task performance and the amount of requested help, thereby reducing the cost of querying the expert. We evaluate Ask4Help on two different tasks -- object goal navigation and room rearrangement and see substantial improvements in performance using minimal help. On object navigation, an agent that achieves a $52\%$ success rate is raised to $86\%$ with $13\%$ help and for rearrangement, the state-of-the-art model with a $7\%$ success rate is dramatically improved to $90.4\%$ using $39\%$ help. Human trials with Ask4Help demonstrate the efficacy of our approach in practical scenarios.

----

## [1180] Human-AI Collaborative Bayesian Optimisation

**Authors**: *Arun Kumar A. V., Santu Rana, Alistair Shilton, Svetha Venkatesh*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6751611b394a3464cea53eed91cf163c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6751611b394a3464cea53eed91cf163c-Abstract-Conference.html)

**Abstract**:

Abstract Human-AI collaboration looks at harnessing the complementary strengths of both humans and AI. We propose a new method for human-AI collaboration in Bayesian optimisation where the optimum is mainly pursued by the Bayesian optimisation algorithm following complex computation, whilst getting occasional help from the accompanying expert having a deeper knowledge of the underlying physical phenomenon. We expect experts to have some understanding of the correlation structures of the experimental system, but not the location of the optimum. The expert provides feedback by either changing the current recommendation or providing her belief on the good and bad regions of the search space based on the current observations. Our proposed method takes such feedback to build a model that aligns with the expert’s model and then uses it for optimisation. We provide theoretical underpinning on why such an approach may be more efficient than the one without expert’s feedback. The empirical results show the robustness and superiority of our method with promising efficiency gains.

----

## [1181] Robust Learning against Relational Adversaries

**Authors**: *Yizhen Wang, Mohannad Alhanahnah, Xiaozhu Meng, Ke Wang, Mihai Christodorescu, Somesh Jha*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6752ced903c3f0265108caa10933965f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6752ced903c3f0265108caa10933965f-Abstract-Conference.html)

**Abstract**:

Test-time adversarial attacks have posed serious challenges to the robustness of machine-learning models, and in many settings the adversarial perturbation need not be bounded by small $\ell_p$-norms. Motivated by attacks in program analysis and security tasks, we investigate $\textit{relational adversaries}$, a broad class of attackers who create adversarial examples in a reflexive-transitive closure of a logical relation. We analyze the conditions for robustness against relational adversaries and investigate different levels of robustness-accuracy trade-off due to various patterns in a relation. Inspired by the insights, we propose $\textit{normalize-and-predict}$, a learning framework that leverages input normalization to achieve provable robustness. The framework solves the pain points of adversarial training against relational adversaries and can be combined with adversarial training for the benefits of both approaches. Guided by our theoretical findings, we apply our framework to source code authorship attribution and malware detection. Results of both tasks show our learning framework significantly improves the robustness of models against relational adversaries. In the process, it outperforms adversarial training, the most noteworthy defense mechanism, by a wide margin.

----

## [1182] Active Bayesian Causal Inference

**Authors**: *Christian Toth, Lars Lorch, Christian Knoll, Andreas Krause, Franz Pernkopf, Robert Peharz, Julius von Kügelgen*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/675e371eeeea99551ce47797ed6ed33e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/675e371eeeea99551ce47797ed6ed33e-Abstract-Conference.html)

**Abstract**:

Causal discovery and causal reasoning are classically treated as separate and consecutive tasks: one first infers the causal graph, and then uses it to estimate causal effects of interventions. However, such a two-stage approach is uneconomical, especially in terms of actively collected interventional data, since the causal query of interest may not require a fully-specified causal model. From a Bayesian perspective, it is also unnatural, since a causal query (e.g., the causal graph or some causal effect) can be viewed as a latent quantity subject to posterior inference—quantities that are not of direct interest ought to be marginalized out in this process, thus contributing to our overall uncertainty. In this work, we propose Active Bayesian Causal Inference (ABCI), a fully-Bayesian active learning framework for integrated causal discovery and reasoning, i.e., for jointly inferring a posterior over causal models and queries of interest. In our approach to ABCI, we focus on the class of causally-sufficient nonlinear additive Gaussian noise models, which we model using Gaussian processes. To capture the space of causal graphs, we use a continuous latent graph representation, allowing our approach to scale to practically relevant problem sizes. We sequentially design experiments that are maximally informative about our target causal query, collect the corresponding interventional data, update our beliefs, and repeat. Through simulations, we demonstrate that our approach is more data-efficient than existing methods that only focus on learning the full causal graph. This allows us to accurately learn downstream causal queries from fewer samples, while providing well-calibrated uncertainty estimates of the quantities of interest.

----

## [1183] Understanding and Improving Robustness of Vision Transformers through Patch-based Negative Augmentation

**Authors**: *Yao Qin, Chiyuan Zhang, Ting Chen, Balaji Lakshminarayanan, Alex Beutel, Xuezhi Wang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67662aa16456e0df65ab001136f92fd0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67662aa16456e0df65ab001136f92fd0-Abstract-Conference.html)

**Abstract**:

We investigate the robustness of vision transformers (ViTs) through the lens of their special patch-based architectural structure, i.e., they process an image as a sequence of image patches. We find that ViTs are surprisingly insensitive to patch-based transformations, even when the transformation largely destroys the original semantics and makes the image unrecognizable by humans. This indicates that ViTs heavily use features that survived such transformations but are generally not indicative of the semantic class to humans. Further investigations show that these features are useful but non-robust, as ViTs trained on them can achieve high in-distribution accuracy, but break down under distribution shifts. From this understanding, we ask: can training the model to rely less on these features improve ViT robustness and out-of-distribution performance? We use the images transformed with our patch-based operations as negatively augmented views and offer losses to regularize the training away from using non-robust features. This is a complementary view to existing research that mostly focuses on augmenting inputs with semantic-preserving transformations to enforce models' invariance. We show that patch-based negative augmentation consistently improves robustness of ViTs on ImageNet based robustness benchmarks across 20+ different experimental settings. Furthermore, we find our patch-based negative augmentation are complementary to traditional (positive) data augmentation techniques and batch-based negative examples in contrastive learning.

----

## [1184] LogiGAN: Learning Logical Reasoning via Adversarial Pre-training

**Authors**: *Xinyu Pi, Wanjun Zhong, Yan Gao, Nan Duan, Jian-Guang Lou*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/677ccf45da6d04ac8e76600821bd05ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/677ccf45da6d04ac8e76600821bd05ce-Abstract-Conference.html)

**Abstract**:

We present LogiGAN, an unsupervised adversarial pre-training framework for improving logical reasoning abilities of language models. Upon automatic identification of logical reasoning phenomena in massive text corpus via detection heuristics, we train language models to predict the masked-out logical statements. Inspired by the facilitation effect of reflective thinking in human learning, we analogically simulate the learning-thinking process with an adversarial Generator-Verifier architecture to assist logic learning. LogiGAN implements a novel sequential GAN approach that (a) circumvents the non-differentiable challenge of the sequential GAN by leveraging the Generator as a sentence-level generative likelihood scorer with a learning objective of reaching scoring consensus with the Verifier; (b) is computationally feasible for large-scale pre-training with arbitrary target length. Both base and large size language models pre-trained with LogiGAN demonstrate obvious performance improvement on 12 datasets requiring general reasoning abilities, revealing the fundamental role of logic in broad reasoning, as well as the effectiveness of LogiGAN. Ablation studies on LogiGAN components reveal the relative orthogonality between linguistic and logic abilities and suggest that reflective thinking's facilitation effect might also generalize to machine learning.

----

## [1185] Revisiting Non-Parametric Matching Cost Volumes for Robust and Generalizable Stereo Matching

**Authors**: *Kelvin Cheng, Tianfu Wu, Christopher G. Healey*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6794f555524c9069e26970a408d353cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6794f555524c9069e26970a408d353cc-Abstract-Conference.html)

**Abstract**:

Stereo matching is a classic challenging problem in computer vision, which has recently witnessed remarkable progress by Deep Neural Networks (DNNs). This paradigm shift leads to two interesting and entangled questions that have not been addressed well. First, it is unclear whether stereo matching DNNs that are trained from scratch really learn to perform matching well. This paper studies this problem from the lens of white-box adversarial attacks. It presents a method of learning stereo-constrained photometrically-consistent attacks, which by design are weaker adversarial attacks, and yet can cause catastrophic performance drop for those DNNs. This observation suggests that they may not actually learn to perform matching well in the sense that they should otherwise achieve potentially even better after stereo-constrained perturbations are introduced. Second, stereo matching DNNs are typically trained under the simulation-to-real (Sim2Real) pipeline due to the data hungriness of DNNs. Thus, alleviating the impacts of the Sim2Real photometric gap in stereo matching DNNs becomes a pressing need.  Towards joint adversarially robust and domain generalizable stereo matching, this paper proposes to learn DNN-contextualized binary-pattern-driven non-parametric cost-volumes. It leverages the perspective of learning the cost aggregation via DNNs, and presents a simple yet expressive design that is fully end-to-end trainable, without resorting to specific aggregation inductive biases. In experiments, the proposed method is tested in the SceneFlow dataset, the KITTI2015 dataset, and the Middlebury dataset. It significantly improves the adversarial robustness, while retaining accuracy performance comparable to state-of-the-art methods. It also shows a better Sim2Real generalizability. Our code and pretrained models are released at \href{https://github.com/kelkelcheng/AdversariallyRobustStereo}{this Github Repo}.

----

## [1186] On the Interpretability of Regularisation for Neural Networks Through Model Gradient Similarity

**Authors**: *Vincent Szolnoky, Viktor Andersson, Balázs Kulcsár, Rebecka Jörnsten*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67b0579a7298d9cf39c59404d867bdd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67b0579a7298d9cf39c59404d867bdd7-Abstract-Conference.html)

**Abstract**:

Most complex machine learning and modelling techniques are prone to over-fitting and may subsequently generalise poorly to future data. Artificial neural networks are no different in this regard and, despite having a level of implicit regularisation when trained with gradient descent, often require the aid of explicit regularisers. We introduce a new framework, Model Gradient Similarity (MGS), that (1) serves as a metric of regularisation, which can be used to monitor neural network training, (2) adds insight into how explicit regularisers, while derived from widely different principles, operate via the same mechanism underneath by increasing MGS, and (3) provides the basis for a new regularisation scheme which exhibits excellent performance, especially in challenging settings such as high levels of label noise or limited sample sizes.

----

## [1187] Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields

**Authors**: *Keqiang Sun, Shangzhe Wu, Zhaoyang Huang, Ning Zhang, Quan Wang, Hongsheng Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67b0e7c7c2a5780aeefe3b79caac106e-Abstract-Conference.html)

**Abstract**:

Capitalizing on the recent advances in image generation models, existing controllable face image synthesis methods are able to generate high-fidelity images with some levels of controllability, e.g., controlling the shapes, expressions, textures, and poses of the generated face images. However, these methods focus on 2D image generative models, which are prone to producing inconsistent face images under large expression and pose changes. In this paper, we propose a new NeRF-based conditional 3D face synthesis framework, which enables 3D controllability over the generated face images by imposing explicit 3D conditions from 3D face priors. At its core is a conditional Generative Occupancy Field (cGOF) that effectively enforces the shape of the generated face to commit to a given 3D Morphable Model (3DMM) mesh. To achieve accurate control over fine-grained 3D face shapes of the synthesized image, we additionally incorporate a 3D landmark loss as well as a volume warping loss into our synthesis algorithm. Experiments validate the effectiveness of the proposed method, which is able to generate high-fidelity face images and shows more precise 3D controllability than state-of-the-art 2D-based controllable face synthesis methods.

----

## [1188] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors**: *Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html)

**Abstract**:

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware---accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention, 3x speedup on GPT-2 (seq. length 1K), and 2.4x speedup on long-range arena (seq. length 1K-4K). FlashAttention, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

----

## [1189] Locally Hierarchical Auto-Regressive Modeling for Image Generation

**Authors**: *Tackgeun You, Saehoon Kim, Chiheon Kim, Doyup Lee, Bohyung Han*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67d60c2694f4fecd18fa04d1fa8c0a5c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67d60c2694f4fecd18fa04d1fa8c0a5c-Abstract-Conference.html)

**Abstract**:

We propose a locally hierarchical auto-regressive model with multiple resolutions of discrete codes. In the first stage of our algorithm, we represent an image with a pyramid of codes using Hierarchically Quantized Variational AutoEncoder (HQ-VAE), which disentangles the information contained in the multi-level codes. For an example of two-level codes, we create two separate pathways to carry high-level coarse structures of input images using top codes while compensating for missing fine details by constructing a residual connection for bottom codes. An appropriate selection of resizing operations for code embedding maps enables top codes to capture maximal information within images and the first stage algorithm achieves better performance on both vector quantization and image generation. The second stage adopts Hierarchically Quantized Transformer (HQ-Transformer) to process a sequence of local pyramids, which consist of a single top code and its corresponding bottom codes. Contrary to other hierarchical models, we sample bottom codes in parallel by exploiting the conditional independence assumption on the bottom codes. This assumption is naturally harvested from our first-stage model, HQ-VAE, where the bottom code learns to describe local details. On class-conditional and text-conditional generation benchmarks, our model shows competitive performance to previous AR models in terms of fidelity of generated images while enjoying lighter computational budgets.

----

## [1190] Learning Enhanced Representation for Tabular Data via Neighborhood Propagation

**Authors**: *Kounianhua Du, Weinan Zhang, Ruiwen Zhou, Yangkun Wang, Xilong Zhao, Jiarui Jin, Quan Gan, Zheng Zhang, David P. Wipf*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67e79c8e9b11f068a7cafd79505175c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67e79c8e9b11f068a7cafd79505175c0-Abstract-Conference.html)

**Abstract**:

Prediction over tabular data is an essential and fundamental problem in many important downstream tasks. However, existing methods either take a data instance of the table independently as input or do not fully utilize the multi-row features and labels to directly change and enhance the target data representations. In this paper, we propose to 1) construct a hypergraph from relevant data instance retrieval to model the cross-row and cross-column patterns of those instances, and 2) perform message Propagation to Enhance the target data instance representation for Tabular prediction tasks. Specifically, our specially-designed message propagation step benefits from 1) the fusion of label and features during propagation, and 2) locality-aware multiplicative high-order interaction between features. Experiments on two important tabular prediction tasks validate the superiority of the proposed PET model against other baselines. Additionally, we demonstrate the effectiveness of the model components and the feature enhancement ability of PET via various ablation studies and visualizations. The code is available at https://github.com/KounianhuaDu/PET.

----

## [1191] On global convergence of ResNets: From finite to infinite width using linear parameterization

**Authors**: *Raphaël Barboni, Gabriel Peyré, François-Xavier Vialard*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67ea314d1df751bbf99ab664ae3049a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67ea314d1df751bbf99ab664ae3049a5-Abstract-Conference.html)

**Abstract**:

Overparameterization is a key factor in the absence of convexity to explain global convergence of gradient descent (GD) for neural networks. Beside the well studied lazy regime, infinite width (mean field) analysis has been developed for shallow networks, using on convex optimization technics. To bridge the gap between the lazy and mean field regimes, we study Residual Networks (ResNets) in which the residual block has linear parameterization while still being nonlinear. Such ResNets admit both infinite depth and width limits, encoding residual blocks in a Reproducing Kernel Hilbert Space (RKHS). In this limit, we prove a local Polyak-Lojasiewicz inequality. Thus, every critical point is a global minimizer and a local convergence result of GD holds, retrieving the lazy regime. In contrast with other mean-field studies, it applies to both parametric and non-parametric cases under an expressivity condition on the residuals. Our analysis leads to a practical and quantified recipe: starting from a universal RKHS, Random Fourier Features are applied to obtain a finite dimensional parameterization satisfying with high-probability our expressivity condition.

----

## [1192] Spherization Layer: Representation Using Only Angles

**Authors**: *Hoyong Kim, Kangil Kim*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/67f30132d98e758f7b4e28c36091d86e-Abstract-Conference.html)

**Abstract**:

In neural network literature, angular similarity between feature vectors is frequently used for interpreting or re-using learned representations. However, the inner product in neural networks partially disperses information over the scales and angles of the involved input vectors and weight vectors. Therefore, when using only angular similarity on representations trained with the inner product, information loss occurs in downstream methods, which limits their performance. In this paper, we proposed the $\textit{spherization layer}$ to represent all information on angular similarity. The layer 1) maps the pre-activations of input vectors into the specific range of angles, 2) converts the angular coordinates of the vectors to Cartesian coordinates with an additional dimension, and 3) trains decision boundaries from hyperplanes, without bias parameters, passing through the origin. This approach guarantees that representation learning always occurs on the hyperspherical surface without the loss of any information unlike other projection-based methods. Furthermore, this method can be applied to any network by replacing an existing layer. We validate the functional correctness of the proposed method in a toy task, retention ability in well-known image classification tasks, and effectiveness in word analogy test and few-shot learning. Code is publicly available at https://github.com/GIST-IRR/spherization_layer

----

## [1193] On the Identifiability of Nonlinear ICA: Sparsity and Beyond

**Authors**: *Yujia Zheng, Ignavier Ng, Kun Zhang*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6801fa3fd290229efc490ee0cf1c5687-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6801fa3fd290229efc490ee0cf1c5687-Abstract-Conference.html)

**Abstract**:

Nonlinear independent component analysis (ICA) aims to recover the underlying independent latent sources from their observable nonlinear mixtures. How to make the nonlinear ICA model identifiable up to certain trivial indeterminacies is a long-standing problem in unsupervised learning. Recent breakthroughs reformulate the standard independence assumption of sources as conditional independence given some auxiliary variables (e.g., class labels and/or domain/time indexes) as weak supervision or inductive bias. However, nonlinear ICA with unconditional priors cannot benefit from such developments. We explore an alternative path and consider only assumptions on the mixing process, such as Structural Sparsity. We show that under specific instantiations of such constraints, the independent latent sources can be identified from their nonlinear mixtures up to a permutation and a component-wise transformation, thus achieving nontrivial identifiability of nonlinear ICA without auxiliary variables. We provide estimation methods and validate the theoretical results experimentally. The results on image data suggest that our conditions may hold in a number of practical data generating processes.

----

## [1194] Self-Supervised Visual Representation Learning with Semantic Grouping

**Authors**: *Xin Wen, Bingchen Zhao, Anlin Zheng, Xiangyu Zhang, Xiaojuan Qi*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html)

**Abstract**:

In this paper, we tackle the problem of learning visual representations from unlabeled scene-centric data. Existing works have demonstrated the potential of utilizing the underlying complex structure within scene-centric data; still, they commonly rely on hand-crafted objectness priors or specialized pretext tasks to build a learning framework, which may harm generalizability. Instead, we propose contrastive learning from data-driven semantic slots, namely SlotCon, for joint semantic grouping and representation learning. The semantic grouping is performed by assigning pixels to a set of learnable prototypes, which can adapt to each sample by attentive pooling over the feature and form new slots. Based on the learned data-dependent slots, a contrastive objective is employed for representation learning, which enhances the discriminability of features, and conversely facilitates grouping semantically coherent pixels together. Compared with previous efforts, by simultaneously optimizing the two coupled objectives of semantic grouping and contrastive learning, our approach bypasses the disadvantages of hand-crafted priors and is able to learn object/group-level representations from scene-centric images. Experiments show our approach effectively decomposes complex scenes into semantic groups for feature learning and significantly benefits downstream tasks, including object detection, instance segmentation, and semantic segmentation. Code is available at: https://github.com/CVMI-Lab/SlotCon.

----

## [1195] Optimistic Mirror Descent Either Converges to Nash or to Strong Coarse Correlated Equilibria in Bimatrix Games

**Authors**: *Ioannis Anagnostides, Gabriele Farina, Ioannis Panageas, Tuomas Sandholm*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/685d249ad59836727be209032f082bd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/685d249ad59836727be209032f082bd7-Abstract-Conference.html)

**Abstract**:

We show that, for any sufficiently small fixed $\epsilon > 0$, when both players in a general-sum two-player (bimatrix) game employ optimistic mirror descent (OMD) with smooth regularization, learning rate $\eta = O(\epsilon^2)$ and $T = \Omega(poly(1/\epsilon))$ repetitions, either the dynamics reach an $\epsilon$-approximate Nash equilibrium (NE), or the average correlated distribution of play is an $\Omega(poly(\epsilon))$-strong coarse correlated equilibrium (CCE): any possible unilateral deviation does not only leave the player worse, but will decrease its utility by $\Omega(poly(\epsilon))$. As an immediate consequence, when the iterates of OMD are bounded away from being Nash equilibria in a bimatrix game, we guarantee convergence to an \emph{exact} CCE after only $O(1)$ iterations. Our results reveal that uncoupled no-regret learning algorithms can converge to CCE in general-sum games remarkably faster than to NE in, for example, zero-sum games. To establish this, we show that when OMD does not reach arbitrarily close to a NE, the (cumulative) regret of both players is not only negative, but decays linearly with time. Given that regret is the canonical measure of performance in online learning, our results suggest that cycling behavior of no-regret learning algorithms in games can be justified in terms of efficiency.

----

## [1196] Discovered Policy Optimisation

**Authors**: *Chris Lu, Jakub Grudzien Kuba, Alistair Letcher, Luke Metz, Christian Schröder de Witt, Jakob N. Foerster*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/688c7a82e31653e7c256c6c29fd3b438-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/688c7a82e31653e7c256c6c29fd3b438-Abstract-Conference.html)

**Abstract**:

Tremendous progress has been made in reinforcement learning (RL) over the past decade. Most of these advancements came through the continual development of new algorithms, which were designed using a combination of mathematical derivations, intuitions, and experimentation. Such an approach of creating algorithms manually is limited by human understanding and ingenuity. In contrast, meta-learning provides a toolkit for automatic machine learning method optimisation, potentially addressing this flaw. However, black-box approaches which attempt to discover RL algorithms with minimal prior structure have thus far not outperformed existing hand-crafted algorithms. Mirror Learning, which includes RL algorithms, such as PPO, offers a potential middle-ground starting point: while every method in this framework comes with theoretical guarantees, components that differentiate them are subject to design. In this paper we explore the Mirror Learning space by meta-learning a “drift” function. We refer to the immediate result as Learnt Policy Optimisation (LPO). By analysing LPO we gain original insights into policy optimisation which we use to formulate a novel, closed-form RL algorithm, Discovered Policy Optimisation (DPO). Our experiments in Brax environments confirm state-of-the-art performance of LPO and DPO, as well as their transfer to unseen settings.

----

## [1197] Robust Testing in High-Dimensional Sparse Models

**Authors**: *Anand Jerry George, Clément L. Canonne*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/689cffc97600f9deb8374fc8fa918b8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/689cffc97600f9deb8374fc8fa918b8e-Abstract-Conference.html)

**Abstract**:

We consider the problem of robustly testing the norm of a high-dimensional sparse signal vector under two different observation models. In the first model, we are given $n$ i.i.d. samples from the distribution $\mathcal{N}\left(\theta,I_d\right)$ (with unknown $\theta$), of which a small fraction has been arbitrarily corrupted. Under the promise that $\|\theta\|_0\le s$, we want to correctly distinguish whether $\|\theta\|_2=0$ or $\|\theta\|_2>\gamma$, for some input parameter $\gamma>0$. We show that any algorithm for this task requires $n=\Omega\left(s\log\frac{ed}{s}\right)$ samples, which is tight up to logarithmic factors. We also extend our results to other common notions of sparsity, namely, $\|\theta\|_q\le s$ for any $0 < q < 2$. In the second observation model that we consider, the data is generated according to a sparse linear regression model, where the covariates are i.i.d. Gaussian and the regression coefficient (signal) is known to be $s$-sparse. Here too we assume that an $\epsilon$-fraction of the data is arbitrarily corrupted. We show that any algorithm that reliably tests the norm of the regression coefficient requires at least $n=\Omega\left(\min(s\log d,{1}/{\gamma^4})\right)$ samples. Our results show that the complexity of testing in these two settings significantly increases under robustness constraints. This is in line with the recent observations made in robust mean testing and robust covariance testing.

----

## [1198] Learning Consistency-Aware Unsigned Distance Functions Progressively from Raw Point Clouds

**Authors**: *Junsheng Zhou, Baorui Ma, Yu-Shen Liu, Yi Fang, Zhizhong Han*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/68d88dcd1e1917c74993902073f08e40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/68d88dcd1e1917c74993902073f08e40-Abstract-Conference.html)

**Abstract**:

Surface reconstruction for point clouds is an important task in 3D computer vision. Most of the latest methods resolve this problem by learning signed distance functions (SDF) from point clouds, which are limited to reconstructing shapes or scenes with closed surfaces. Some other methods tried to represent shapes or scenes with open surfaces using unsigned distance functions (UDF) which are learned from large scale ground truth unsigned distances. However, the learned UDF is hard to provide smooth distance fields near the surface due to the noncontinuous character of point clouds. In this paper, we propose a novel method to learn consistency-aware unsigned distance functions directly from raw point clouds. We achieve this by learning to move 3D queries to reach the surface with a field consistency constraint, where we also enable to progressively estimate a more accurate surface. Specifically, we train a neural network to gradually infer the relationship between 3D queries and the approximated surface by searching for the moving target of queries in a dynamic way, which results in a consistent field around the surface. Meanwhile, we introduce a polygonization algorithm to extract surfaces directly from the gradient field of the learned UDF. The experimental results in surface reconstruction for synthetic and real scan data show significant improvements over the state-of-the-art under the widely used benchmarks.

----

## [1199] Understanding Square Loss in Training Overparametrized Neural Network Classifiers

**Authors**: *Tianyang Hu, Jun Wang, Wenjia Wang, Zhenguo Li*

**Conference**: *nips 2022*

**URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/690ddbee6eef37933f4be0abeb7aff45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/690ddbee6eef37933f4be0abeb7aff45-Abstract-Conference.html)

**Abstract**:

Deep learning has achieved many breakthroughs in modern classification tasks. Numerous architectures have been proposed for different data structures but when it comes to the loss function, the cross-entropy loss is the predominant choice. Recently, several alternative losses have seen revived interests for deep classifiers. In particular, empirical evidence seems to promote square loss but a theoretical justification is still lacking. In this work, we contribute to the theoretical understanding of square loss in classification by systematically investigating how it performs for overparametrized neural networks in the neural tangent kernel (NTK) regime. Interesting properties regarding the generalization error, robustness, and calibration error are revealed. We consider two cases, according to whether classes are separable or not. In the general non-separable case, fast convergence rate is established for both misclassification rate and calibration error. When classes are separable, the misclassification rate improves to be exponentially fast. Further, the resulting margin is proven to be lower bounded away from zero, providing theoretical guarantees for robustness. We expect our findings to hold beyond the NTK regime and translate to practical settings. To this end, we conduct extensive empirical studies on practical neural networks, demonstrating the effectiveness of square loss in both synthetic low-dimensional data and real image data. Comparing to cross-entropy, square loss has comparable generalization error but noticeable advantages in robustness and model calibration.

----



[Go to the previous page](NIPS-2022-list05.md)

[Go to the next page](NIPS-2022-list07.md)

[Go to the catalog section](README.md)