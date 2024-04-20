## [2800] Text-Adaptive Multiple Visual Prototype Matching for Video-Text Retrieval

        **Authors**: *Chengzhi Lin, Ancong Wu, Junwei Liang, Jun Zhang, Wenhang Ge, Wei-Shi Zheng, Chunhua Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html)

        **Abstract**:

        Cross-modal retrieval between videos and texts has gained increasing interest because of the rapid emergence of videos on the web. Generally, a video contains rich instance and event information and the query text  only describes a part of the information. Thus, a video can have multiple different text descriptions and queries. We call it the Video-Text Correspondence Ambiguity problem. Current techniques mostly concentrate on mining local or multi-level alignment between contents of video and text (e.g., object to entity and action to verb). It is difficult for these methods to alleviate video-text correspondence ambiguity by describing a video using only one feature, which is required to be matched with multiple different text features at the same time. To address this problem, we propose a Text-Adaptive Multiple Visual Prototype Matching Model. It automatically captures multiple prototypes to describe a video by adaptive aggregation on video token features. Given a query text, the similarity is determined by the most similar prototype to find correspondence in the video, which is called text-adaptive matching.  To learn diverse prototypes for representing the rich information in videos, we propose a variance loss to encourage different prototypes to attend to different contents of the video.  Our method outperforms the state-of-the-art methods on four public video retrieval datasets.

        ----

        ## [2801] Asymptotically Unbiased Instance-wise Regularized Partial AUC Optimization: Theory and Algorithm

        **Authors**: *Huiyang Shao, Qianqian Xu, Zhiyong Yang, Shilong Bao, Qingming Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fc9f83d9925e6885e8f1ae1e17b3c44b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fc9f83d9925e6885e8f1ae1e17b3c44b-Abstract-Conference.html)

        **Abstract**:

        The Partial Area Under the ROC Curve (PAUC), typically including One-way Partial AUC (OPAUC) and Two-way Partial AUC (TPAUC), measures the average performance of a binary classifier within a specific false positive rate and/or true positive rate interval, which is a widely adopted measure when decision constraints must be considered. Consequently, PAUC optimization has naturally attracted increasing attention in the machine learning community within the last few years. Nonetheless, most of the existing methods could only optimize PAUC approximately, leading to inevitable biases that are not controllable. Fortunately, a recent work presents an unbiased formulation of the PAUC optimization problem via distributional robust optimization. However, it is based on the pair-wise formulation of AUC, which suffers from the limited scalability w.r.t. sample size and a slow convergence rate, especially for TPAUC. To address this issue, we present a simpler reformulation of the problem in an asymptotically unbiased and instance-wise manner. For both OPAUC and TPAUC, we come to a nonconvex strongly concave min-max regularized problem of instance-wise functions. On top of this, we employ an efficient solver that enjoys a linear per-iteration computational complexity w.r.t. the sample size and a time-complexity of $O(\epsilon^{-1/3})$ to reach a $\epsilon$ stationary point. Furthermore, we find that the min-max reformulation also facilitates the theoretical analysis of generalization error as a byproduct. Compared with the existing results, we present new error bounds that are much easier to prove and could deal with hypotheses with real-valued outputs. Finally, extensive experiments on several benchmark datasets demonstrate the effectiveness of our method.

        ----

        ## [2802] Universality of Group Convolutional Neural Networks Based on Ridgelet Analysis on Groups

        **Authors**: *Sho Sonoda, Isao Ishikawa, Masahiro Ikeda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fcc3dc27672a12510babe448d665e152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fcc3dc27672a12510babe448d665e152-Abstract-Conference.html)

        **Abstract**:

        We show the universality of depth-2 group convolutional neural networks (GCNNs) in a unified and constructive manner based on the ridgelet theory. Despite widespread use in applications, the approximation property of (G)CNNs has not been well investigated. The universality of (G)CNNs has been shown since the late 2010s. Yet, our understanding on how (G)CNNs represent functions is incomplete because the past universality theorems have been shown in a case-by-case manner by manually/carefully assigning the network parameters depending on the variety of convolution layers, and in an indirect manner by converting/modifying the (G)CNNs into other universal approximators such as invariant polynomials and fully-connected networks. In this study, we formulate a versatile depth-2 continuous GCNN $S[\gamma]$ as a nonlinear mapping between group representations, and  directly obtain an analysis operator, called the ridgelet trasform, that maps a given function $f$ to the network parameter $\gamma$ so that $S[\gamma]=f$. The proposed GCNN covers typical GCNNs such as the cyclic convolution on multi-channel images, networks on permutation-invariant inputs (Deep Sets), and $\mathrm{E}(n)$-equivariant networks. The closed-form expression of the ridgelet transform can describe how the network parameters are organized to represent a function. While it has been known only for fully-connected networks, this study is the first to obtain the ridgelet transform for GCNNs. By discretizing the closed-form expression, we can systematically generate a constructive proof of the $cc$-universality of finite GCNNs. In other words, our universality proofs are more unified and constructive than previous proofs.

        ----

        ## [2803] Error Correction Code Transformer

        **Authors**: *Yoni Choukroun, Lior Wolf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fcd3909db30887ce1da519c4468db668-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fcd3909db30887ce1da519c4468db668-Abstract-Conference.html)

        **Abstract**:

        Error correction code is a major part of the physical communication layer, ensuring the reliable transfer of data over noisy channels.Recently, neural decoders were shown to outperform classical decoding techniques.However, the existing neural approaches present strong overfitting, due to the exponential training complexity, or a restrictive inductive bias, due to reliance on Belief Propagation.Recently, Transformers have become methods of choice in many applications, thanks to their ability to represent complex interactions between elements.In this work, we propose to extend for the first time the Transformer architecture to the soft decoding of linear codes at arbitrary block lengths.We encode each channel's output dimension to a high dimension for a better representation of the bits' information to be processed separately.The element-wise processing allows the analysis of channel output reliability, while the algebraic code and the interaction between the bits are inserted into the model via an adapted masked self-attention module.The proposed approach demonstrates the power and flexibility of Transformers and outperforms existing state-of-the-art neural decoders by large margins, at a fraction of their time complexity.

        ----

        ## [2804] Towards Improving Calibration in Object Detection Under Domain Shift

        **Authors**: *Muhammad Akhtar Munir, Muhammad Haris Khan, M. Saquib Sarfraz, Mohsen Ali*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fcd812a51b8f8d05cfea22e3c9c4b369-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fcd812a51b8f8d05cfea22e3c9c4b369-Abstract-Conference.html)

        **Abstract**:

        With deep neural network based solution more readily being incorporated in real-world applications, it has been pressing requirement that predictions by such models, especially in safety-critical environments, be  highly accurate and well-calibrated. Although some techniques addressing DNN calibration have been proposed, they are only limited to visual classification applications and in-domain predictions. Unfortunately, very little to no attention is paid towards addressing calibration of DNN-based visual object detectors, that occupy similar space and importance in many decision making systems as their visual classification counterparts. In this work, we study the calibration of DNN-based object detection models, particularly under domain shift. To this end, we first propose a new, plug-and-play, train-time calibration loss for object detection (coined as TCD). It can be used with various application-specific loss functions as an auxiliary loss function to improve detection calibration. Second, we devise a new implicit technique for improving calibration in self-training based domain adaptive detectors, featuring a new uncertainty quantification mechanism for object detection. We demonstrate TCD is capable of enhancing calibration with notable margins (1) across different DNN-based object detection paradigms both in in-domain and out-of-domain predictions, and (2) in different domain-adaptive detectors across challenging adaptation scenarios. Finally, we empirically show that our implicit calibration technique can be used in tandem with TCD during adaptation to further boost calibration in diverse domain shift scenarios.

        ----

        ## [2805] Renyi Differential Privacy of Propose-Test-Release and Applications to Private and Robust Machine Learning

        **Authors**: *Jiachen T. Wang, Saeed Mahloujifar, Shouda Wang, Ruoxi Jia, Prateek Mittal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fcdffb372c9fa2ce757cf457415c7aab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fcdffb372c9fa2ce757cf457415c7aab-Abstract-Conference.html)

        **Abstract**:

        Propose-Test-Release (PTR) is a differential privacy framework that works with local sensitivity of functions, instead of their global sensitivity. This framework is typically used for releasing robust statistics such as median or trimmed mean in a differentially private manner. While PTR is a common framework introduced over a decade ago, using it in applications such as robust SGD where we need many adaptive robust queries is challenging. This is mainly due to the lack of \Renyi Differential Privacy (RDP) analysis, an essential ingredient underlying the moments accountant approach for differentially private deep learning. In this work, we generalize the standard PTR and derive the first RDP bound for it. We show that our RDP bound for PTR yields tighter DP guarantees than the directly analyzed $(\varepsilon, \delta)$-DP. We also derive the algorithm-specific privacy amplification bound of PTR under subsampling. We show that our bound is much tighter than the general upper bound and close to the lower bound. Our RDP bounds enable tighter privacy loss calculation for the composition of many adaptive runs of PTR. As an application of our analysis, we show that PTR and our theoretical results can be used to design differentially private variants for byzantine robust training algorithms that use robust statistics for gradients aggregation. We conduct experiments on the settings of label, feature, and gradient corruption across different datasets and architectures. We show that PTR-based private and robust training algorithm significantly improves the utility compared with the baseline.

        ----

        ## [2806] A Transformer-Based Object Detector with Coarse-Fine Crossing Representations

        **Authors**: *Zhishan Li, Ying Nie, Kai Han, Jianyuan Guo, Lei Xie, Yunhe Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fcfad93e2f30ab4c22f9ec5edfbb5cc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fcfad93e2f30ab4c22f9ec5edfbb5cc0-Abstract-Conference.html)

        **Abstract**:

        Transformer-based object detectors have shown competitive performance recently.  Compared with convolutional neural networks limited by the relatively small receptive fields, the advantage of transformer for visual tasks is the capacity to perceive long-range dependencies among all image patches, while the deficiency is that the local fine-grained information is not fully excavated. In this paper, we introduce the Coarse-grained and Fine-grained crossing representations to build an efficient Detection Transformer (CFDT). Specifically, we propose a local-global cross fusion module to establish the connection between local fine-grained features and global coarse-grained features. Besides, we propose a coarse-fine aware neck which enables detection tokens to interact with both coarse-grained and fine-grained features. Furthermore, an efficient feature integration module is presented for fusing multi-scale representations from different stages. Experimental results on the COCO dataset demonstrate the effectiveness of the proposed method. For instance, our CFDT achieves 48.1 AP with 173G FLOPs, which possesses higher accuracy and less computation compared with the state-of-the-art transformer-based detector ViDT. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/CFDT.

        ----

        ## [2807] Beyond Adult and COMPAS: Fair Multi-Class Prediction via Information Projection

        **Authors**: *Wael Alghamdi, Hsiang Hsu, Haewon Jeong, Hao Wang, Peter Michalák, Shahab Asoodeh, Flávio P. Calmon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of producing fair probabilistic classifiers for multi-class classification tasks. We formulate this problem in terms of ``projecting'' a pre-trained (and potentially unfair) classifier onto the set of models that satisfy target group-fairness requirements. The new, projected model is given by post-processing the outputs of the pre-trained classifier by a multiplicative factor. We provide a parallelizable, iterative algorithm for computing the projected classifier and derive both sample complexity and convergence guarantees. Comprehensive numerical comparisons with state-of-the-art benchmarks demonstrate that our approach maintains competitive performance in terms of accuracy-fairness trade-off curves, while achieving favorable runtime on large datasets. We also evaluate our method at scale on an open dataset with multiple classes, multiple intersectional groups, and over 1M samples.

        ----

        ## [2808] Explicit Tradeoffs between Adversarial and Natural Distributional Robustness

        **Authors**: *Mazda Moayeri, Kiarash Banihashem, Soheil Feizi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd62b65606f0f0d2af2c01623a224258-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd62b65606f0f0d2af2c01623a224258-Abstract-Conference.html)

        **Abstract**:

        Several existing works study either adversarial or natural distributional robustness of deep neural networks separately. In practice, however, models need to enjoy both types of robustness to ensure reliability. In this work, we bridge this gap and show that in fact, {\it explicit tradeoffs} exist between adversarial and natural distributional robustness. We first consider a simple linear regression setting on Gaussian data with disjoint sets of \emph{core} and \emph{spurious} features. In this setting, through theoretical and empirical analysis, we show that (i) adversarial training with $\ell_1$ and $\ell_2$ norms increases the model reliance on spurious features; (ii) For $\ell_\infty$ adversarial training, spurious reliance only occurs when the scale of the spurious features is larger than that of the core features; (iii) adversarial training can have {\it an unintended consequence} in reducing distributional robustness, specifically when spurious correlations are changed in the new test domain. Next, we present extensive empirical evidence, using a test suite of twenty adversarially trained models evaluated on five benchmark datasets (ObjectNet, RIVAL10, Salient ImageNet-1M, ImageNet-9, Waterbirds), that adversarially trained classifiers rely on backgrounds more than their standardly trained counterparts, validating our theoretical results. We also show that spurious correlations in training data (when preserved in the test domain) can {\it improve} adversarial robustness, revealing that previous claims that adversarial vulnerability is rooted in spurious correlations are incomplete.

        ----

        ## [2809] Learning Latent Seasonal-Trend Representations for Time Series Forecasting

        **Authors**: *Zhiyuan Wang, Xovee Xu, Weifeng Zhang, Goce Trajcevski, Ting Zhong, Fan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html)

        **Abstract**:

        Forecasting complex time series is ubiquitous and vital in a range of applications but challenging. Recent advances endeavor to achieve progress by incorporating various deep learning techniques (e.g., RNN and Transformer) into sequential models. However, clear patterns are still hard to extract since time series are often composed of several intricately entangled components. Motivated by the success of disentangled variational autoencoder in computer vision and classical time series decomposition, we plan to infer a couple of representations that depict seasonal and trend components of time series. To achieve this goal, we propose LaST, which, based on variational inference, aims to disentangle the seasonal-trend representations in the latent space. Furthermore, LaST supervises and disassociates representations from the perspectives of themselves and input reconstruction, and introduces a series of auxiliary objectives. Extensive experiments prove that LaST achieves state-of-the-art performance on time series forecasting task against the most advanced representation learning and end-to-end forecasting models. For reproducibility, our implementation is publicly available on Github.

        ----

        ## [2810] JAHS-Bench-201: A Foundation For Research On Joint Architecture And Hyperparameter Search

        **Authors**: *Archit Bansal, Danny Stoll, Maciej Janowski, Arber Zela, Frank Hutter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd78f2f65881c1c7ce47e26b040cf48f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd78f2f65881c1c7ce47e26b040cf48f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The past few years have seen the development of many benchmarks for Neural Architecture Search (NAS), fueling rapid progress in NAS research. However, recent work, which shows that good hyperparameter settings can be more important than using the best architecture, calls for a shift in focus towards Joint Architecture and Hyperparameter Search (JAHS). Therefore, we present JAHS-Bench-201, the first collection of surrogate benchmarks for JAHS, built to also facilitate research on multi-objective, cost-aware and (multi) multi-fidelity optimization algorithms. To the best of our knowledge, JAHS-Bench-201 is based on the most extensive dataset of neural network performance data in the public domain. It is composed of approximately 161 million data points and 20 performance metrics for three deep learning tasks, while featuring a 14-dimensional search and fidelity space that extends the popular NAS-Bench-201 space. With JAHS-Bench-201, we hope to democratize research on JAHS and lower the barrier to entry of an extremely compute intensive field, e.g., by reducing the compute time to run a JAHS algorithm from 5 days to only a few seconds.

        ----

        ## [2811] Capturing Graphs with Hypo-Elliptic Diffusions

        **Authors**: *Csaba Tóth, Darrick Lee, Celia Hacker, Harald Oberhauser*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd7f43f8689988f4ef056f192ec0589b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd7f43f8689988f4ef056f192ec0589b-Abstract-Conference.html)

        **Abstract**:

        Convolutional layers within graph neural networks operate by aggregating information about local neighbourhood structures; one common way to encode such substructures is through random walks. The distribution of these random walks evolves according to a diffusion equation defined using the graph Laplacian. We extend this approach by leveraging classic mathematical results about hypo-elliptic diffusions. This results in a novel tensor-valued graph operator, which we call the hypo-elliptic graph Laplacian. We provide theoretical guarantees and efficient low-rank approximation algorithms. In particular, this gives a structured approach to capture long-range dependencies on graphs that is robust to pooling. Besides the attractive theoretical properties, our experiments show that this method competes with graph transformers on datasets requiring long-range reasoning but scales only linearly in the number of edges as opposed to quadratically in nodes.

        ----

        ## [2812] A Spectral Approach to Item Response Theory

        **Authors**: *Duc Nguyen, Anderson Ye Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd88ea50ca8c1973db037462f116ff99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd88ea50ca8c1973db037462f116ff99-Abstract-Conference.html)

        **Abstract**:

        The Rasch model is one of the most fundamental models in item response theory and has wide-ranging applications from education testing to recommendation systems. In a universe with $n$ users and $m$ items, the Rasch model assumes that the binary response $X_{li} \in \{0,1\}$ of a user $l$ with parameter $\theta^*_l$ to an item $i$ with parameter $\beta^*_i$ (e.g., a user likes a movie, a student correctly solves a problem) is distributed as $\mathbb{P}(X_{li}=1) = 1/(1 + \exp(-(\theta^*_l - \beta^*_i)))$. In this paper, we propose a new item estimation algorithm for this celebrated model (i.e., to estimate $\beta^*$). The core of our algorithm is the computation of the stationary distribution of a Markov chain defined on an item-item graph. We complement our algorithmic contributions with finite-sample error guarantees, the first of their kind in the literature, showing that our algorithm is consistent and enjoys favorable optimality properties. We discuss practical modifications to accelerate and robustify the algorithm that practitioners can adopt. Experiments on synthetic and real-life datasets, ranging from small education testing datasets to large recommendation systems datasets show that our algorithm is scalable, accurate, and competitive with the most commonly used methods in the literature.

        ----

        ## [2813] FedSR: A Simple and Effective Domain Generalization Method for Federated Learning

        **Authors**: *A. Tuan Nguyen, Philip H. S. Torr, Ser Nam Lim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fd946a6c99541fddc3d64a3ea39a1bc2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fd946a6c99541fddc3d64a3ea39a1bc2-Abstract-Conference.html)

        **Abstract**:

        Federated Learning (FL) refers to the decentralized and privacy-preserving machine learning framework in which multiple clients collaborate (with the help of a central server) to train a global model without sharing their data. However, most existing FL methods only focus on maximizing the model's performance on the source clients' data (e.g., mobile users) without considering its generalization ability to unknown target data (e.g., a new user). In this paper, we incorporate the problem of Domain Generalization (DG) into Federated Learning to tackle the aforementioned issue. However, virtually all existing DG methods require a centralized setting where data is shared across the domains, which violates the principles of decentralized FL and hence not applicable. To this end, we propose a simple yet novel representation learning framework, namely FedSR, which enables domain generalization while still respecting the decentralized and privacy-preserving natures of this FL setting. Motivated by classical machine learning algorithms, we aim to learn a simple representation of the data for better generalization. In particular, we enforce an L2-norm regularizer on the representation and a conditional mutual information (between the representation and the data given the label) regularizer to encourage the model to only learn essential information (while ignoring spurious correlations such as the background). Furthermore, we provide theoretical connections between the above two objectives and representation alignment in domain generalization. Extensive experimental results suggest that our method significantly outperforms relevant baselines in this particular problem.

        ----

        ## [2814] SIXO: Smoothing Inference with Twisted Objectives

        **Authors**: *Dieterich Lawson, Allan Raventós, Andrew Warrington, Scott W. Linderman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fddc79681b2df2734c01444f9bc2a17e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fddc79681b2df2734c01444f9bc2a17e-Abstract-Conference.html)

        **Abstract**:

        Sequential Monte Carlo (SMC) is an inference algorithm for state space models that approximates the posterior by sampling from a sequence of target distributions. The target distributions are often chosen to be the filtering distributions, but these ignore information from future observations, leading to practical and theoretical limitations in inference and model learning.  We introduce SIXO, a method that instead learns target distributions that approximate the smoothing distributions, incorporating information from all observations. The key idea is to use density ratio estimation to fit functions that warp the filtering distributions into the smoothing distributions. We then use SMC with these learned targets to define a variational objective for model and proposal learning. SIXO yields provably tighter log marginal lower bounds and offers more accurate posterior inferences and parameter estimates in a variety of domains.

        ----

        ## [2815] Explicable Policy Search

        **Authors**: *Ze Gong, Yu Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fdff3c4130c24c40c88aa41eb52d2a27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fdff3c4130c24c40c88aa41eb52d2a27-Abstract-Conference.html)

        **Abstract**:

        Human teammates often form conscious and subconscious expectations of each other during interaction. Teaming success is contingent on whether such expectations can be met. Similarly, for an intelligent agent to operate beside a human, it must consider the human’s expectation of its behavior. Disregarding such expectations can lead to the loss of trust and degraded team performance. A key challenge here is that the human’s expectation may not align with the agent’s optimal behavior, e.g., due to the human’s partial or inaccurate understanding of the task domain. Prior work on explicable planning described the ability of agents to respect their human teammate’s expectations by trading off task performance for more expected or “explicable” behaviors. In this paper, we introduce Explicable Policy Search (EPS) to significantly extend such an ability to stochastic domains in a reinforcement learning (RL) setting with continuous state and action spaces. Furthermore, in contrast to the traditional RL methods, EPS must at the same time infer the human’s hidden expectations. Such inferences require information about the human’s belief about the domain dynamics and her reward model but directly querying them is impractical. We demonstrate that such information can be necessarily and sufficiently encoded by a surrogate reward function for EPS, which can be learned based on the human’s feedback on the agent’s behavior. The surrogate reward function is then used to reshape the agent’s reward function, which is shown to be equivalent to searching for an explicable policy. We evaluate EPS in a set of navigation domains with synthetic human models and in an autonomous driving domain with a user study. The results suggest that our method can generate explicable behaviors that reconcile task performance with human expectations intelligently and has real-world relevance in human-agent teaming domains.

        ----

        ## [2816] Exploring evolution-aware & -free protein language models as protein function predictors

        **Authors**: *Mingyang Hu, Fajie Yuan, Kevin Yang, Fusong Ju, Jin Su, Hui Wang, Fei Yang, Qiuyang Ding*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html)

        **Abstract**:

        Large-scale Protein Language Models (PLMs) have improved performance in protein prediction tasks, ranging from 3D structure prediction to various function predictions. In particular, AlphaFold, a ground-breaking AI system, could potentially reshape structural biology. However, the utility of the PLM module in AlphaFold, Evoformer, has not been explored beyond structure prediction. In this paper, we investigate the representation ability of three popular PLMs: ESM-1b (single sequence), MSA-Transformer (multiple sequence alignment), and Evoformer (structural), with a special focus on Evoformer. Specifically, we aim to answer the following key questions: (1) Does the Evoformer trained as part of AlphaFold produce representations amenable to predicting protein function?  (2) If yes, can Evoformer replace ESM-1b and MSA-Transformer? (3) How much do these PLMs rely on evolution-related protein data? In this regard, are they complementary to each other? We compare these models by empirical study along with new insights and conclusions. All code and datasets for reproducibility are available at https://github.com/elttaes/Revisiting-PLMs .

        ----

        ## [2817] Breaking Bad: A Dataset for Geometric Fracture and Reassembly

        **Authors**: *Silvia Sellán, Yun-Chun Chen, Ziyi Wu, Animesh Garg, Alec Jacobson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe18f2090bf1e0fd5a1ded5bdd7ca351-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe18f2090bf1e0fd5a1ded5bdd7ca351-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce Breaking Bad, a large-scale dataset of fractured objects. Our dataset consists of over one million fractured objects simulated from ten thousand base models. The fracture simulation is powered by a recent physically based algorithm that efficiently generates a variety of fracture modes of an object. Existing shape assembly datasets decompose objects according to semantically meaningful parts, effectively modeling the construction process. In contrast, Breaking Bad models the destruction process of how a geometric object naturally breaks into fragments. Our dataset serves as a benchmark that enables the study of fractured object reassembly and presents new challenges for geometric shape understanding. We analyze our dataset with several geometry measurements and benchmark three state-of-the-art shape assembly deep learning methods under various settings. Extensive experimental results demonstrate the difficulty of our dataset, calling on future research in model designs specifically for the geometric shape assembly task. We host our dataset at https://breaking-bad-dataset.github.io/.

        ----

        ## [2818] Fair and Optimal Decision Trees: A Dynamic Programming Approach

        **Authors**: *Jacobus G. M. van der Linden, Mathijs de Weerdt, Emir Demirovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html)

        **Abstract**:

        Interpretable and fair machine learning models are required for many applications, such as credit assessment and in criminal justice. Decision trees offer this interpretability, especially when they are small. Optimal decision trees are of particular interest because they offer the best performance possible for a given size. However, state-of-the-art algorithms for fair and optimal decision trees have scalability issues, often requiring several hours to find such trees even for small datasets. Previous research has shown that dynamic programming (DP) performs well for optimizing decision trees because it can exploit the tree structure. However, adding a global fairness constraint to a DP approach is not straightforward, because the global constraint violates the condition that subproblems should be independent. We show how such a constraint can be incorporated by introducing upper and lower bounds on final fairness values for partial solutions of subproblems, which enables early comparison and pruning. Our results show that our model can find fair and optimal trees several orders of magnitude faster than previous methods, and now also for larger datasets that were previously beyond reach. Moreover, we show that with this substantial improvement our method can find the full Pareto front in the trade-off between accuracy and fairness.

        ----

        ## [2819] Recommender Forest for Efficient Retrieval

        **Authors**: *Chao Feng, Wuchao Li, Defu Lian, Zheng Liu, Enhong Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe2fe749d329627f161484876630c689-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe2fe749d329627f161484876630c689-Abstract-Conference.html)

        **Abstract**:

        Recommender systems (RS) have to select the top-N items from a massive item set. For the sake of efficient recommendation, RS usually represents user and item as latent embeddings, and relies on approximate nearest neighbour search (ANNs) to retrieve the recommendation result. Despite the reduction of running time, the representation learning is independent of ANNs index construction; thus, the two operations can be incompatible, which results in potential loss of recommendation accuracy. To overcome the above problem, we propose the Recommender Forest (a.k.a., RecForest), which jointly learns latent embedding and index for efficient and high-fidelity recommendation. RecForest consists of multiple k-ary trees, each of which is a partition of the item set via hierarchical balanced clustering such that each item is uniquely represented by a path from the root to a leaf. Given such a data structure, an encoder-decoder based routing network is developed: it first encodes the context, i.e., user information, into hidden states; then, leveraging a transformer-based decoder, it identifies the top-N items via beam search. Compared with the existing methods, RecForest brings in the following advantages: 1) the false partition of the boundary items can be effectively alleviated by the use of multiple trees; 2) the routing operation becomes much more accurate thanks to the powerful transformer decoder; 3) the tree parameters are shared across different tree levels, making the index to be extremely memory-efficient. The experimental studies are performed on five popular recommendation datasets: with a significantly simplified training cost, RecForest outperforms competitive baseline approaches in terms of both recommendation accuracy and efficiency.

        ----

        ## [2820] Graph Few-shot Learning with Task-specific Structures

        **Authors**: *Song Wang, Chen Chen, Jundong Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe47dd3fd8e7eb43187d42d65083e383-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe47dd3fd8e7eb43187d42d65083e383-Abstract-Conference.html)

        **Abstract**:

        Graph few-shot learning is of great importance among various graph learning tasks. Under the few-shot scenario, models are often required to conduct classification given limited labeled samples. Existing graph few-shot learning methods typically leverage Graph Neural Networks (GNNs) and perform classification across a series of meta-tasks. Nevertheless, these methods generally rely on the original graph (i.e., the graph that the meta-task is sampled from) to learn node representations. Consequently, the learned representations for the same nodes are identical in all meta-tasks. Since the class sets are different across meta-tasks, node representations should be task-specific to promote classification performance. Therefore, to adaptively learn node representations across meta-tasks, we propose a novel framework that learns a task-specific structure for each meta-task. To handle the variety of nodes across meta-tasks, we extract relevant nodes and learn task-specific structures based on node influence and mutual information. In this way, we can learn node representations with the task-specific structure tailored for each meta-task. We further conduct extensive experiments on five node classification datasets under both single- and multiple-graph settings to validate the superiority of our framework over the state-of-the-art baselines.

        ----

        ## [2821] DASCO: Dual-Generator Adversarial Support Constrained Offline Reinforcement Learning

        **Authors**: *Quan Vuong, Aviral Kumar, Sergey Levine, Yevgen Chebotar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)

        **Abstract**:

        In offline RL, constraining the learned policy to remain close to the data is essential to prevent the policy from outputting out-of-distribution (OOD) actions with erroneously overestimated values. In principle, generative adversarial networks (GAN) can provide an elegant solution to do so, with the discriminator directly providing a probability that quantifies distributional shift. However, in practice, GAN-based offline RL methods have not outperformed alternative approaches, perhaps because the generator is trained to both fool the discriminator and maximize return - two objectives that are often at odds with each other. In this paper, we show that the issue of conflicting objectives can be resolved by training two generators: one that maximizes return, with the other capturing the "remainder" of the data distribution in the offline dataset, such that the mixture of the two is close to the behavior policy. We show that not only does having two generators enable an effective GAN-based offline RL method, but also approximates a support constraint, where the policy does not need to match the entire data distribution, but only the slice of the data that leads to high long term performance. We name our method DASCO, for Dual-Generator Adversarial Support Constrained Offline RL. On benchmark tasks that require learning from sub-optimal data, DASCO significantly outperforms prior methods that enforce distribution constraint.

        ----

        ## [2822] Beyond L1: Faster and Better Sparse Models with skglm

        **Authors**: *Quentin Bertrand, Quentin Klopfenstein, Pierre-Antoine Bannier, Gauthier Gidel, Mathurin Massias*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe5c31e525e9a26a1426ab0b589f42fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe5c31e525e9a26a1426ab0b589f42fe-Abstract-Conference.html)

        **Abstract**:

        We propose a new fast algorithm to estimate any sparse generalized linear model with convex or non-convex separable penalties. Our algorithm is able to solve problems with millions of samples and features in seconds, by relying on coordinate descent, working sets and Anderson acceleration.  It handles previously unaddressed models, and  is extensively shown to improve state-of-art algorithms. We provide a flexible, scikit-learn compatible package, which easily handles customized datafits and penalties.

        ----

        ## [2823] You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments

        **Authors**: *Keiran Paster, Sheila A. McIlraith, Jimmy Ba*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe90657b12193c7b52a3418bdc351807-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe90657b12193c7b52a3418bdc351807-Abstract-Conference.html)

        **Abstract**:

        Recently, methods such as Decision Transformer that reduce reinforcement learning to a prediction task and solve it via supervised learning (RvS) have become popular due to their simplicity, robustness to hyperparameters, and strong overall performance on offline RL tasks. However, simply conditioning a probabilistic model on a desired return and taking the predicted action can fail dramatically in stochastic environments since trajectories that result in a return may have only achieved that return due to luck. In this work, we describe the limitations of RvS approaches in stochastic environments and propose a solution. Rather than simply conditioning on returns, as is standard practice, our proposed method, ESPER, conditions on learned average returns which are independent from environment stochasticity. Doing so allows ESPER to achieve strong alignment between target return and expected performance in real environments. We demonstrate this in several challenging stochastic offline-RL tasks including the challenging puzzle game 2048, and Connect Four playing against a stochastic opponent. In all tested domains, ESPER achieves significantly better alignment between the target return and achieved return than simply conditioning on returns. ESPER also achieves higher maximum performance than even the value-based baselines.

        ----

        ## [2824] Cost-efficient Gaussian tensor network embeddings for tensor-structured inputs

        **Authors**: *Linjian Ma, Edgar Solomonik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe91414cdc6348bcb5710e81bcb72c08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe91414cdc6348bcb5710e81bcb72c08-Abstract-Conference.html)

        **Abstract**:

        This work discusses tensor network embeddings, which are random matrices ($S$) with tensor network structure. These embeddings have been used to perform dimensionality reduction of tensor network structured inputs $x$ and accelerate applications such as tensor decomposition and kernel regression. Existing works have designed embeddings for inputs $x$ with specific structures, such as the Kronecker product or Khatri-Rao product, such that the computational cost for calculating $Sx$ is efficient. We provide a systematic way to design tensor network embeddings consisting of Gaussian random tensors, such that for inputs with more general tensor network structures, both the sketch size (row size of $S$) and the sketching computational cost are low.We analyze general tensor network embeddings that can be reduced to a sequence of sketching matrices. We provide a sufficient condition to quantify the accuracy of such embeddings and derive sketching asymptotic cost lower bounds using embeddings that satisfy this condition and have a sketch size lower than any input dimension. We then provide an algorithm to efficiently sketch input data using such embeddings. The sketch size of the embedding used in the algorithm has a linear dependence on the number of sketching dimensions of the input. Assuming tensor contractions are performed with classical dense matrix multiplication algorithms, this algorithm achieves asymptotic cost within a factor of $O(\sqrt{m})$ of our cost lower bound, where $m$ is the sketch size. Further, when each tensor in the input has a dimension that needs to be sketched, this algorithm yields the optimal sketching asymptotic cost. We apply our sketching analysis to inexact tensor decomposition optimization algorithms. We provide a sketching algorithm for CP decomposition that is asymptotically faster than existing work in multiple regimes, and show the optimality of an existing algorithm for tensor train rounding.

        ----

        ## [2825] Neural Transmitted Radiance Fields

        **Authors**: *Chengxuan Zhu, Renjie Wan, Boxin Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fe989bb038b5dcc44181255dd6913e43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/fe989bb038b5dcc44181255dd6913e43-Abstract-Conference.html)

        **Abstract**:

        Neural radiance fields (NeRF) have brought tremendous progress to novel view synthesis. Though NeRF enables the rendering of subtle details in a scene by learning from a dense set of images, it also reconstructs the undesired reflections when we capture images through glass. As a commonly observed interference, the reflection would undermine the visibility of the desired transmitted scene behind glass by occluding the transmitted light rays. In this paper, we aim at addressing the problem of rendering novel transmitted views given a set of reflection-corrupted images. By introducing the transmission encoder and recurring edge constraints as guidance, our neural transmitted radiance fields can resist such reflection interference during rendering and reconstruct high-fidelity results even under sparse views. The proposed method achieves superior performance from the experiments on a newly collected dataset compared with state-of-the-art methods.

        ----

        ## [2826] Geoclidean: Few-Shot Generalization in Euclidean Geometry

        **Authors**: *Joy Hsu, Jiajun Wu, Noah D. Goodman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/feb34ce77fc8b94c85d12e608b23ce67-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/feb34ce77fc8b94c85d12e608b23ce67-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Euclidean geometry is among the earliest forms of mathematical thinking. While the geometric primitives underlying its constructions, such as perfect lines and circles, do not often occur in the natural world, humans rarely struggle to perceive and reason with them. Will computer vision models trained on natural images show the same sensitivity to Euclidean geometry? Here we explore these questions by studying few-shot generalization in the universe of Euclidean geometry constructions. We introduce Geoclidean, a domain-specific language for Euclidean geometry, and use it to generate two datasets of geometric concept learning tasks for benchmarking generalization judgements of humans and machines. We find that humans are indeed sensitive to Euclidean geometry and generalize strongly from a few visual examples of a geometric concept. In contrast, low-level and high-level visual features from standard computer vision models pretrained on natural images do not support correct generalization. Thus Geoclidean represents a novel few-shot generalization benchmark for geometric concept learning, where the performance of humans and of AI models diverge. The Geoclidean framework and dataset are publicly available for download.

        ----

        ## [2827] Enabling Detailed Action Recognition Evaluation Through Video Dataset Augmentation

        **Authors**: *Jihoon Chung, Yu Wu, Olga Russakovsky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ff52407b80dde0f0f45814db2738464c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/ff52407b80dde0f0f45814db2738464c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        It is well-known in the video understanding community that human action recognition models suffer from background bias, i.e., over-relying on scene cues in making their predictions. However, it is difficult to quantify this effect using existing evaluation frameworks. We introduce the Human-centric Analysis Toolkit (HAT), which enables evaluation of learned background bias without the need for new manual video annotation. It does so by automatically generating synthetically manipulated videos and leveraging the recent advances in image segmentation and video inpainting. Using HAT we perform an extensive analysis of 74 action recognition models trained on the Kinetics dataset. We confirm that all these models focus more on the scene background than on the human motion; further, we demonstrate that certain model design decisions (such as training with fewer frames per video or using dense as opposed to uniform temporal sampling) appear to worsen the background bias. We open-source HAT to enable the community to design more robust and generalizable human action recognition models.

        ----

        ## [2828] Unsupervised Skill Discovery via Recurrent Skill Training

        **Authors**: *Zheyuan Jiang, Jingyue Gao, Jianyu Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ff6b031d5bdc552b795175a0f3b35a50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ff6b031d5bdc552b795175a0f3b35a50-Abstract-Conference.html)

        **Abstract**:

        Being able to discover diverse useful skills without external reward functions is beneficial in reinforcement learning research. Previous unsupervised skill discovery approaches mainly train different skills in parallel. Although impressive results have been provided, we found that parallel training procedure can sometimes block exploration when the state visited by different skills overlap, which leads to poor state coverage and restricts the diversity of learned skills. In this paper, we take a deeper look into this phenomenon and propose a novel framework to address this issue, which we call Recurrent Skill Training (ReST). Instead of training all the skills in parallel, ReST trains different skills one after another recurrently, along with a state coverage based intrinsic reward. We conduct experiments on a number of challenging 2D navigation environments and robotic locomotion environments. Evaluation results show that our proposed approach outperforms previous parallel training approaches in terms of state coverage and skill diversity. Videos of the discovered skills are available at https://sites.google.com/view/neurips22-rest.

        ----

        ## [2829] Structural Kernel Search via Bayesian Optimization and Symbolical Optimal Transport

        **Authors**: *Matthias Bitzer, Mona Meister, Christoph Zimmer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ff7373914a96956f2a7cacbdf3b0b8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ff7373914a96956f2a7cacbdf3b0b8d8-Abstract-Conference.html)

        **Abstract**:

        Despite recent advances in automated machine learning, model selection is still a complex and computationally intensive process. For Gaussian processes (GPs), selecting the kernel is a crucial task, often done manually by the expert. Additionally, evaluating the model selection criteria for Gaussian processes typically scales cubically in the sample size, rendering kernel search particularly computationally expensive. We propose a novel, efficient search method through a general, structured kernel space. Previous methods solved this task via Bayesian optimization and relied on measuring the distance between GP's directly in function space to construct a kernel-kernel. We present an alternative approach by defining a kernel-kernel over the symbolic representation of the statistical hypothesis that is associated with a kernel. We empirically show that this leads to a computationally more efficient way of searching through a discrete kernel space.

        ----

        ## [2830] Robust Models are less Over-Confident

        **Authors**: *Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ff887781480973bd3cb6026feb378d1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ff887781480973bd3cb6026feb378d1e-Abstract-Conference.html)

        **Abstract**:

        Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustnessconfidencesevaluation

        ----

        ## [2831] Near-Optimal No-Regret Learning Dynamics for General Convex Games

        **Authors**: *Gabriele Farina, Ioannis Anagnostides, Haipeng Luo, Chung-Wei Lee, Christian Kroer, Tuomas Sandholm*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ffa1301939cc707d6e986e6c4124340b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ffa1301939cc707d6e986e6c4124340b-Abstract-Conference.html)

        **Abstract**:

        A recent line of work has established uncoupled learning dynamics such that, when employed by all players in a game, each player's regret after $T$ repetitions grows polylogarithmically in $T$, an exponential improvement over the traditional guarantees within the no-regret framework. However, so far these results have only been limited to certain classes of games with structured strategy spaces---such as normal-form and extensive-form games. The question as to whether $O(\mathrm{polylog} T)$ regret bounds can be obtained for general convex and compact strategy sets---as is the case in many fundamental models in economics and multiagent systems---while retaining efficient strategy updates is an important question. In this paper, we answer this in the positive by establishing the first uncoupled learning algorithm with $O(\log T)$ per-player regret in general convex games, that is, games with concave utility functions supported on arbitrary convex and compact strategy sets. Our learning dynamics are based on an instantiation of optimistic follow-the-regularized-leader over an appropriately lifted space using a self-concordant regularizer that is peculiarly not a barrier for the feasible region. Our learning dynamics are efficiently implementable given access to a proximal oracle for the convex strategy set, leading to $O(\log\log T)$ per-iteration complexity; we also give extensions when access to only a linear optimization oracle is assumed. Finally, we adapt our dynamics to guarantee $O(\sqrt{T})$ regret in the adversarial regime. Even in those special cases where prior results apply, our algorithm improves over the state-of-the-art regret bounds either in terms of the dependence on the number of iterations or on the dimension of the strategy sets.

        ----

        ## [2832] OTKGE: Multi-modal Knowledge Graph Embeddings via Optimal Transport

        **Authors**: *Zongsheng Cao, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ffdb280e7c7b4c4af30e04daf5a84b98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ffdb280e7c7b4c4af30e04daf5a84b98-Abstract-Conference.html)

        **Abstract**:

        Multi-modal knowledge graph embeddings (KGE) have caught more and more attention in learning representations of entities and relations for link prediction tasks. Different from previous uni-modal KGE approaches, multi-modal KGE can leverage expressive knowledge from a wealth of modalities (image, text, etc.), leading to more comprehensive representations of real-world entities. However, the critical challenge along this course lies in that the multi-modal embedding spaces are usually heterogeneous. In this sense, direct fusion will destroy the inherent spatial structure of different modal embeddings. To overcome this challenge, we revisit multi-modal KGE from a distributional alignment perspective and propose optimal transport knowledge graph embeddings (OTKGE). Specifically, we model the multi-modal fusion procedure as a transport plan moving different modal embeddings to a unified space by minimizing the Wasserstein distance between multi-modal distributions. Theoretically, we show that by minimizing the Wasserstein distance between the individual modalities and the unified embedding space, the final results are guaranteed to maintain consistency and comprehensiveness. Moreover, experimental results on well-established multi-modal knowledge graph completion benchmarks show that our OTKGE achieves state-of-the-art performance.

        ----

        ## [2833] ComMU: Dataset for Combinatorial Music Generation

        **Authors**: *Hyun Lee, Taehyun Kim, Hyolim Kang, Minjoo Ki, Hyeonchan Hwang, Kwanho Park, Sharang Han, Seon Joo Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/fff3ba5059aeeb88c324b6ba9b298166-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/fff3ba5059aeeb88c324b6ba9b298166-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Commercial adoption of automatic music composition requires the capability of generating diverse and high-quality music suitable for the desired context (e.g., music for romantic movies, action games, restaurants, etc.). In this paper, we introduce combinatorial music generation, a new task to create varying background music based on given conditions. Combinatorial music generation creates short samples of music with rich musical metadata, and combines them to produce a complete music. In addition, we introduce ComMU, the first symbolic music dataset consisting of short music samples and their corresponding 12 musical metadata for combinatorial music generation. Notable properties of ComMU are that (1) dataset is manually constructed by professional composers with an objective guideline that induces regularity, and (2) it has 12 musical metadata that embraces composers' intentions. Our results show that we can generate diverse high-quality music only with metadata, and that our unique metadata such as track-role and extended chord quality improves the capacity of the automatic composition. We highly recommend watching our video before reading the paper (https://pozalabs.github.io/ComMU/).

        ----

        

[Go to the previous page](NIPS-2022-list14.md)

[Go to the catalog section](README.md)