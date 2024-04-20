## [1400] Spotting the Unseen: Reciprocal Consensus Network Guided by Visual Archetypes

**Authors**: *Wenbo Hu, Hongjian Zhan, Xinchen Ma, Yue Lu, Ching Y. Suen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29149](https://doi.org/10.1609/aaai.v38i11.29149)

**Abstract**:

Humans often require only a few visual archetypes to spot novel objects. Based on this observation, we present a strategy rooted in ``spotting the unseen" by establishing dense correspondences between potential query image regions and a visual archetype, and we propose the Consensus Network (CoNet). Our method leverages relational patterns intra and inter images via Auto-Correlation Representation (ACR) and Mutual-Correlation Representation (MCR). Within each image, the ACR module is capable of encoding both local self-similarity and global context simultaneously.  Between the query and support images, the MCR module computes the cross-correlation across two image representations and introduces a reciprocal consistency constraint, which can incorporate to exclude outliers and enhance model robustness. To overcome the challenges of low-resource training data, particularly in one-shot learning scenarios, we incorporate an adaptive margin strategy to better handle diverse instances. The experimental results indicate the effectiveness of the proposed method across diverse domains such as object detection in natural scenes, and text spotting in both historical manuscripts and natural scenes, which demonstrates its sparkling generalization ability. Our code is available at: https://github.com/infinite-hwb/conet.

----

## [1401] Terrain Diffusion Network: Climatic-Aware Terrain Generation with Geological Sketch Guidance

**Authors**: *Zexin Hu, Kun Hu, Clinton Mo, Lei Pan, Zhiyong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29150](https://doi.org/10.1609/aaai.v38i11.29150)

**Abstract**:

Sketch-based terrain generation seeks to create realistic landscapes for virtual environments in various applications such as computer games, animation and virtual reality. Recently, deep learning based terrain generation has emerged, notably the ones based on generative adversarial networks (GAN). However, these methods often struggle to fulfill the requirements of flexible user control and maintain generative diversity for realistic terrain. Therefore, we propose a novel diffusion-based method, namely terrain diffusion network (TDN), which actively incorporates user guidance for enhanced controllability, taking into account terrain features like rivers, ridges, basins, and peaks. Instead of adhering to a conventional monolithic denoising process, which often compromises the fidelity of terrain details or the alignment with user control, a multi-level denoising scheme is proposed to generate more realistic terrains by taking into account fine-grained details, particularly those related to climatic patterns influenced by erosion and tectonic activities. Specifically, three terrain synthesisers are designed for structural, intermediate, and fine-grained level denoising purposes, which allow each synthesiser concentrate on a distinct terrain aspect. Moreover, to maximise the efficiency of our TDN, we further introduce terrain and sketch latent spaces for the synthesizers with pre-trained terrain autoencoders. Comprehensive experiments on a new dataset constructed from NASA Topology Images clearly demonstrate the effectiveness of our proposed method, achieving the state-of-the-art performance. Our code is available at https://github.com/TDNResearch/TDN.

----

## [1402] Energy Efficient Streaming Time Series Classification with Attentive Power Iteration

**Authors**: *Hao Huang, Tapan Shah, Scott Evans, Shinjae Yoo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29151](https://doi.org/10.1609/aaai.v38i11.29151)

**Abstract**:

Efficiently processing time series data streams in real-time on resource-constrained devices offers significant advantages in terms of enhanced computational energy efficiency and reduced time-related risks. We introduce an innovative streaming time series classification network that utilizes attentive power iteration, enabling real-time processing on resource-constrained devices. Our model continuously updates a compact representation of the entire time series, enhancing classification accuracy while conserving energy and processing time. Notably, it excels in streaming scenarios without requiring complete time series access, enabling swift decisions. Experimental results show that our approach excels in classification accuracy and energy efficiency, with over 70% less consumption and threefold faster task completion than benchmarks. This work advances real-time responsiveness, energy conservation, and operational effectiveness for constrained devices, contributing to optimizing various applications.

----

## [1403] SEC: More Accurate Clustering Algorithm via Structural Entropy

**Authors**: *Junyu Huang, Qilong Feng, Jiahui Wang, Ziyun Huang, Jinhui Xu, Jianxin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29152](https://doi.org/10.1609/aaai.v38i11.29152)

**Abstract**:

As one of the most popular machine learning tools in the field of unsupervised learning, clustering has been widely used in various practical applications. While numerous methods have been proposed for clustering, a commonly encountered issue is that the existing clustering methods rely heavily on local neighborhood information during the optimization process, which leads to suboptimal performance on real-world datasets. Besides, most existing clustering methods use Euclidean distances or densities to measure the similarity between data points. This could constrain the effectiveness of the algorithms for handling datasets with irregular patterns. Thus, a key challenge is how to effectively capture the global structural information in clustering instances to improve the clustering quality. In this paper, we propose a new clustering algorithm, called SEC. This algorithm uses the global structural information extracted from an encoding tree to guide the clustering optimization process. Based on the relation between data points in the instance, a sparse graph of the clustering instance can be constructed. By leveraging the sparse graph constructed, we propose an iterative encoding tree method, where hierarchical abstractions of the encoding tree are iteratively extracted as new clustering features to obtain better clustering results. To avoid the influence of easily misclustered data points located on the boundaries of the clustering partitions, which we call "fringe points", we propose an iterative pre-deletion and reassignment technique such that the algorithm can delete and reassign the "fringe points" to obtain more resilient and precise clustering results. Empirical experiments on both synthetic and real-world datasets demonstrate that our proposed algorithm outperforms state-of-the-art clustering methods and achieves better clustering performances. On average, the clustering accuracy (ACC) is increased by 1.7% and the normalized mutual information (NMI) by 7.9% compared with the current state-of-the-art (SOTA) algorithm on synthetic datasets. On real-world datasets, our method outperforms other clustering methods with an average increase of 12.3% in ACC and 5.2% in NMI, respectively.

----

## [1404] eTag: Class-Incremental Learning via Embedding Distillation and Task-Oriented Generation

**Authors**: *Libo Huang, Yan Zeng, Chuanguang Yang, Zhulin An, Boyu Diao, Yongjun Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29153](https://doi.org/10.1609/aaai.v38i11.29153)

**Abstract**:

Class incremental learning (CIL) aims to solve the notorious forgetting problem, which refers to the fact that once the network is updated on a new task, its performance on previously-learned tasks degenerates catastrophically. Most successful CIL methods store exemplars (samples of learned tasks) to train a feature extractor incrementally, or store prototypes (features of learned tasks) to estimate the incremental feature distribution. However, the stored exemplars would violate the data privacy concerns, while the fixed prototypes might not reasonably be consistent with the incremental feature distribution, hindering the exploration of real-world CIL applications. In this paper, we propose a data-free CIL method with embedding distillation and Task-oriented generation (eTag), which requires neither exemplar nor prototype. Embedding distillation prevents the feature extractor from forgetting by distilling the outputs from the networks' intermediate blocks. Task-oriented generation enables a lightweight generator to produce dynamic features, fitting the needs of the top incremental classifier. Experimental results confirm that the proposed eTag considerably outperforms state-of-the-art methods on several benchmark datasets.

----

## [1405] PPO-Clip Attains Global Optimality: Towards Deeper Understandings of Clipping

**Authors**: *Nai-Chieh Huang, Ping-Chun Hsieh, Kuo-Hao Ho, I-Chen Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29154](https://doi.org/10.1609/aaai.v38i11.29154)

**Abstract**:

Proximal Policy Optimization algorithm employing a clipped surrogate objective (PPO-Clip) is a prominent exemplar of the policy optimization methods. However, despite its remarkable empirical success, PPO-Clip lacks theoretical substantiation to date. In this paper, we contribute to the field by establishing the first global convergence results of a PPO-Clip variant in both tabular and neural function approximation settings. Our findings highlight the O(1/√T ) min-iterate convergence rate specifically in the context of neural function approximation. We tackle the inherent challenges in analyzing PPO-Clip through three central concepts: (i) We introduce a generalized version of the PPO-Clip objective, illuminated by its connection with the hinge loss. (ii) Employing entropic mirror descent, we establish asymptotic convergence for tabular PPO-Clip with direct policy parameterization. (iii) Inspired by the tabular analysis, we streamline convergence analysis by introducing a two-step policy improvement approach. This decouples policy search from complex neural policy parameterization using a regression-based update scheme. Furthermore, we gain deeper insights into the efficacy of PPO-Clip by interpreting these generalized objectives. Our theoretical findings also mark the first characterization of the influence of the clipping mechanism on PPO-Clip convergence. Importantly, the clipping range affects only the
pre-constant of the convergence rate.

----

## [1406] HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting

**Authors**: *Qihe Huang, Lei Shen, Ruixin Zhang, Jiahuan Cheng, Shouhong Ding, Zhengyang Zhou, Yang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29155](https://doi.org/10.1609/aaai.v38i11.29155)

**Abstract**:

Multivariate time series (MTS) prediction has been widely adopted in various scenarios. Recently, some methods have employed patching to enhance local semantics and improve model performance. However, length-fixed patch are prone to losing temporal boundary information, such as complete peaks and periods. Moreover, existing methods mainly focus on modeling long-term dependencies across patches, while paying little attention to other dimensions (e.g., short-term dependencies within patches and complex interactions among cross-variavle patches). To address these challenges, we propose a pure MLP-based HDMixer, aiming to acquire patches with richer semantic information and efficiently modeling hierarchical interactions. Specifically, we design a Length-Extendable Patcher (LEP) tailored to MTS, which enriches the boundary information of patches and alleviates semantic incoherence in series. Subsequently, we devise a Hierarchical Dependency Explorer (HDE) based on pure MLPs. This explorer effectively models short-term dependencies within patches, long-term dependencies across patches, and complex interactions among variables. Extensive experiments on 9 real-world datasets demonstrate the superiority of our approach. The code is available at https://github.com/hqh0728/HDMixer.

----

## [1407] Measuring Task Similarity and Its Implication in Fine-Tuning Graph Neural Networks

**Authors**: *Renhong Huang, Jiarong Xu, Xin Jiang, Chenglu Pan, Zhiming Yang, Chunping Wang, Yang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29156](https://doi.org/10.1609/aaai.v38i11.29156)

**Abstract**:

The paradigm of pre-training and fine-tuning graph neural networks has attracted wide research attention. In previous studies, the pre-trained models are viewed as universally versatile, and applied for a diverse range of downstream tasks. In many situations, however, this practice results in limited or even negative transfer. This paper, for the first time, emphasizes the specific application scope of graph pre-trained models: not all downstream tasks can effectively benefit from a graph pre-trained model. In light of this, we introduce the measure task consistency to quantify the similarity between graph pre-training and downstream tasks. This measure assesses the extent to which downstream tasks can benefit from specific pre-training tasks. Moreover, a novel fine-tuning strategy, Bridge-Tune, is proposed to further diminish the impact of the difference between pre-training and downstream tasks. The key innovation in Bridge-Tune is an intermediate step that bridges pre-training and downstream tasks. This step takes into account the task differences and further refines the pre-trained model. The superiority of the presented fine-tuning strategy is validated via numerous experiments with different pre-trained models and downstream tasks.

----

## [1408] Factorized Explainer for Graph Neural Networks

**Authors**: *Rundong Huang, Farhad Shirani, Dongsheng Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29157](https://doi.org/10.1609/aaai.v38i11.29157)

**Abstract**:

Graph Neural Networks (GNNs) have received increasing attention due to their ability to learn from graph-structured data. To open the black-box of these deep learning models, post-hoc instance-level explanation methods have been proposed to understand GNN predictions. These methods seek to discover substructures that explain the prediction behavior of a trained GNN.  In this paper, we show analytically that for a large class of explanation tasks, conventional approaches, which are based on the principle of graph information bottleneck (GIB), admit trivial solutions that do not align with the notion of explainability. Instead, we argue that a modified GIB principle may be used to avoid the aforementioned trivial solutions. We further introduce a novel factorized explanation model with theoretical performance guarantees. The modified GIB is used to analyze the structural properties of the proposed factorized explainer. We conduct extensive experiments on both synthetic and real-world datasets to validate the effectiveness of our proposed factorized explainer.

----

## [1409] Stochastic Bayesian Optimization with Unknown Continuous Context Distribution via Kernel Density Estimation

**Authors**: *Xiaobin Huang, Lei Song, Ke Xue, Chao Qian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29158](https://doi.org/10.1609/aaai.v38i11.29158)

**Abstract**:

Bayesian optimization (BO) is a sample-efficient method and has been widely used for optimizing expensive black-box functions. Recently, there has been a considerable interest in BO literature in optimizing functions that are affected by context variable in the environment, which is uncontrollable by decision makers. In this paper, we focus on the optimization of functions' expectations over continuous context variable, subject to an unknown distribution. To address this problem, we propose two algorithms that employ kernel density estimation to learn the probability density function (PDF) of continuous context variable online. The first algorithm is simpler, which directly optimizes the expectation under the estimated PDF. Considering that the estimated PDF may have high estimation error when the true distribution is complicated, we further propose the second algorithm that optimizes the distributionally robust objective. Theoretical results demonstrate that both algorithms have sub-linear Bayesian cumulative regret on the expectation objective. Furthermore, we conduct numerical experiments to empirically demonstrate the effectiveness of our algorithms.

----

## [1410] One Step Learning, One Step Review

**Authors**: *Xiaolong Huang, Qiankun Li, Xueran Li, Xuesong Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29159](https://doi.org/10.1609/aaai.v38i11.29159)

**Abstract**:

Visual fine-tuning has garnered significant attention with the rise of pre-trained vision models. The current prevailing method, full fine-tuning, suffers from the issue of knowledge forgetting as it focuses solely on fitting the downstream training set. In this paper, we propose a novel weight rollback-based fine-tuning method called OLOR (One step Learning, One step Review). OLOR combines fine-tuning with optimizers, incorporating a weight rollback term into the weight update term at each step. This ensures consistency in the weight range of upstream and downstream models, effectively mitigating knowledge forgetting and enhancing fine-tuning performance. In addition, a layer-wise penalty is presented to employ penalty decay and the diversified decay rate to adjust the weight rollback levels of layers for adapting varying downstream tasks. Through extensive experiments on various tasks such as image classification, object detection, semantic segmentation, and instance segmentation, we demonstrate the general applicability and state-of-the-art performance of our proposed OLOR. Code is available at https://github.com/rainbow-xiao/OLOR-AAAI-2024.

----

## [1411] Higher-Order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes

**Authors**: *Yiming Huang, Yujie Zeng, Qiang Wu, Linyuan Lü*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29160](https://doi.org/10.1609/aaai.v38i11.29160)

**Abstract**:

Despite the recent successes of vanilla Graph Neural Networks (GNNs) on various tasks, their foundation on pairwise networks inherently limits their capacity to discern latent higher-order interactions in complex systems.  To bridge this capability gap, we propose a novel approach exploiting the rich mathematical theory of simplicial complexes (SCs) - a robust tool for modeling higher-order interactions.  Current SC-based GNNs are burdened by high complexity and rigidity, and quantifying higher-order interaction strengths remains challenging.  Innovatively, we present a higher-order Flower-Petals (FP) model, incorporating FP Laplacians into SCs. Further, we introduce a Higher-order Graph Convolutional Network (HiGCN) grounded in FP Laplacians, capable of discerning intrinsic features across varying topological scales.  By employing learnable graph filters, a parameter group within each FP Laplacian domain, we can identify diverse patterns where the filters' weights serve as a quantifiable measure of higher-order interaction strengths.  The theoretical underpinnings of HiGCN's advanced expressiveness are rigorously demonstrated. Additionally, our empirical investigations reveal that the proposed model accomplishes state-of-the-art performance on a range of graph tasks and provides a scalable and flexible solution to explore higher-order interactions in graphs.  Codes and datasets are available at https://github.com/Yiminghh/HiGCN.

----

## [1412] Protein 3D Graph Structure Learning for Robust Structure-Based Protein Property Prediction

**Authors**: *Yufei Huang, Siyuan Li, Lirong Wu, Jin Su, Haitao Lin, Odin Zhang, Zihan Liu, Zhangyang Gao, Jiangbin Zheng, Stan Z. Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29161](https://doi.org/10.1609/aaai.v38i11.29161)

**Abstract**:

Protein structure-based property prediction has emerged as a promising approach for various biological tasks, such as protein function prediction and sub-cellular location estimation. The existing methods highly rely on experimental protein structure data and fail in scenarios where these data are unavailable. Predicted protein structures from AI tools (e.g., AlphaFold2) were utilized as alternatives. However, we observed that current practices, which simply employ accurately predicted structures during inference, suffer from notable degradation in prediction accuracy. While similar phenomena have been extensively studied in general fields (e.g., Computer Vision) as model robustness, their impact on protein property prediction remains unexplored. In this paper, we first investigate the reason behind the performance decrease when utilizing predicted structures, attributing it to the structure embedding bias from the perspective of structure representation learning. To study this problem, we identify a Protein 3D Graph Structure Learning Problem for Robust Protein Property Prediction (PGSL-RP3), collect benchmark datasets, and present a protein Structure embedding Alignment Optimization framework (SAO) to mitigate the problem of structure embedding bias between the predicted and experimental protein structures. Extensive experiments have shown that our framework is model-agnostic and effective in improving the property prediction of both predicted structures and experimental structures.

----

## [1413] Binding-Adaptive Diffusion Models for Structure-Based Drug Design

**Authors**: *Zhilin Huang, Ling Yang, Zaixi Zhang, Xiangxin Zhou, Yu Bao, Xiawu Zheng, Yuwei Yang, Yu Wang, Wenming Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29162](https://doi.org/10.1609/aaai.v38i11.29162)

**Abstract**:

Structure-based drug design (SBDD) aims to generate 3D ligand molecules that bind to specific protein targets. Existing 3D deep generative models including diffusion models have shown great promise for SBDD. However, it is complex to capture the essential protein-ligand interactions exactly in 3D space for molecular generation. To address this problem, we propose a novel framework, namely Binding-Adaptive Diffusion Models (BindDM). In BindDM, we adaptively extract subcomplex, the essential part of binding sites responsible for protein-ligand interactions. Then the selected protein-ligand subcomplex is processed with SE(3)-equivariant neural networks, and transmitted back to each atom of the complex for augmenting the target-aware 3D molecule diffusion generation with binding interaction information. We iterate this hierarchical complex-subcomplex process with cross-hierarchy interaction node for adequately fusing global binding context between the complex and its corresponding subcomplex. Empirical studies on the CrossDocked2020 dataset show BindDM can generate molecules with more realistic 3D structures and higher binding affinities towards the protein targets, with up to -5.92 Avg. Vina Score, while maintaining proper molecular properties. Our code is available at https://github.com/YangLing0818/BindDM

----

## [1414] Optimal Survival Trees: A Dynamic Programming Approach

**Authors**: *Tim Huisman, Jacobus G. M. van der Linden, Emir Demirovic*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29163](https://doi.org/10.1609/aaai.v38i11.29163)

**Abstract**:

Survival analysis studies and predicts the time of death, or other singular unrepeated events, based on historical data, while the true time of death for some instances is unknown. Survival trees enable the discovery of complex nonlinear relations in a compact human comprehensible model, by recursively splitting the population and predicting a distinct survival distribution in each leaf node. We use dynamic programming to provide the first survival tree method with optimality guarantees, enabling the assessment of the optimality gap of heuristics. We improve the scalability of our method through a special algorithm for computing trees up to depth two. The experiments show that our method's run time even outperforms some heuristics for realistic cases while obtaining similar out-of-sample performance with the state-of-the-art.

----

## [1415] ProCC: Progressive Cross-Primitive Compatibility for Open-World Compositional Zero-Shot Learning

**Authors**: *Fushuo Huo, Wenchao Xu, Song Guo, Jingcai Guo, Haozhao Wang, Ziming Liu, Xiaocheng Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29164](https://doi.org/10.1609/aaai.v38i11.29164)

**Abstract**:

Open-World Compositional Zero-shot Learning (OW-CZSL) aims to recognize novel compositions of state and object primitives in images with no priors on the compositional space, which induces a tremendously large output space containing all possible state-object compositions. Existing works either learn the joint compositional state-object embedding or predict simple primitives with separate classifiers. However, the former method heavily relies on external word embedding methods, and the latter ignores the interactions of interdependent primitives, respectively. In this paper, we revisit the primitive prediction approach and propose a novel method, termed Progressive Cross-primitive Compatibility (ProCC), to mimic the human learning process for OW-CZSL tasks. Specifically, the cross-primitive compatibility module explicitly learns to model the interactions of state and object features with the trainable memory units, which efficiently acquires cross-primitive visual attention to reason high-feasibility compositions, without the aid of external knowledge. Moreover, to alleviate the invalid cross-primitive interactions, especially for partial-supervision conditions (pCZSL), we design a progressive training paradigm to optimize the primitive classifiers conditioned on pre-trained features in an easy-to-hard manner. Extensive experiments on three widely used benchmark datasets demonstrate that our method outperforms other representative methods on both OW-CZSL and pCZSL settings by large margins.

----

## [1416] Non-exemplar Online Class-Incremental Continual Learning via Dual-Prototype Self-Augment and Refinement

**Authors**: *Fushuo Huo, Wenchao Xu, Jingcai Guo, Haozhao Wang, Yunfeng Fan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29165](https://doi.org/10.1609/aaai.v38i11.29165)

**Abstract**:

This paper investigates a new, practical, but challenging problem named Non-exemplar Online Class-incremental continual Learning (NO-CL), which aims to preserve the discernibility of base classes without buffering data examples and efficiently learn novel classes continuously in a single-pass (i.e., online) data stream. The challenges of this task are mainly two-fold: (1) Both base and novel classes suffer from severe catastrophic forgetting as no previous samples are available for replay. (2) As the online data can only be observed once, there is no way to fully re-train the whole model, e.g., re-calibrate the decision boundaries via prototype alignment or feature distillation. In this paper, we propose a novel Dual-prototype Self-augment and Refinement method (DSR) for NO-CL problem, which consists of two strategies: 1) Dual class prototypes: vanilla and high-dimensional prototypes are exploited to utilize the pre-trained information and obtain robust quasi-orthogonal representations rather than example buffers for both privacy preservation and memory reduction. 2) Self-augment and refinement: Instead of updating the whole network, we optimize high-dimensional prototypes alternatively with the extra projection module based on self-augment vanilla prototypes, through a bi-level optimization problem. Extensive experiments demonstrate the effectiveness and superiority of the proposed DSR in NO-CL.

----

## [1417] New Classes of the Greedy-Applicable Arm Feature Distributions in the Sparse Linear Bandit Problem

**Authors**: *Koji Ichikawa, Shinji Ito, Daisuke Hatano, Hanna Sumita, Takuro Fukunaga, Naonori Kakimura, Ken-ichi Kawarabayashi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29166](https://doi.org/10.1609/aaai.v38i11.29166)

**Abstract**:

We consider the sparse contextual bandit problem where arm feature affects reward through the inner product of sparse parameters. Recent studies have developed sparsity-agnostic algorithms based on the greedy arm selection policy. However, the analysis of these algorithms requires strong assumptions on the arm feature distribution to ensure that the greedily selected samples are sufficiently diverse; One of the most common assumptions, relaxed symmetry, imposes approximate origin-symmetry on the distribution, which cannot allow distributions that has origin-asymmetric support. In this paper, we show that the greedy algorithm is applicable to a wider range of the arm feature distributions from two aspects. Firstly, we show that a mixture distribution that has a greedy-applicable component is also greedy-applicable. Second, we propose new distribution classes, related to Gaussian mixture, discrete, and radial distribution, for which the sample diversity is guaranteed. The proposed classes can describe distributions with origin-asymmetric support and, in conjunction with the first claim, provide theoretical guarantees of the greedy policy for a very wide range of the arm feature distributions.

----

## [1418] Fairness without Demographics through Shared Latent Space-Based Debiasing

**Authors**: *Rashidul Islam, Huiyuan Chen, Yiwei Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29167](https://doi.org/10.1609/aaai.v38i11.29167)

**Abstract**:

Ensuring fairness in machine learning (ML) is crucial, particularly in applications that impact diverse populations. The majority of existing works heavily rely on the availability of protected features like race and gender. However, practical challenges such as privacy concerns and regulatory restrictions often prohibit the use of this data, limiting the scope of traditional fairness research. To address this, we introduce a Shared Latent Space-based Debiasing (SLSD) method that transforms data from both the target domain, which lacks protected features, and a separate source domain, which contains these features, into correlated latent representations. This allows for joint training of a cross-domain protected group estimator on the representations. We then debias the downstream ML model with an adversarial learning technique that leverages the group estimator. We also present a relaxed variant of SLSD, the R-SLSD, that occasionally accesses a small subset of protected features from the target domain during its training phase. Our extensive experiments on benchmark datasets demonstrate that our methods consistently outperform existing state-of-the-art models in standard group fairness metrics.

----

## [1419] TMPNN: High-Order Polynomial Regression Based on Taylor Map Factorization

**Authors**: *Andrei Ivanov, Stefan Maria Ailuro*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29168](https://doi.org/10.1609/aaai.v38i11.29168)

**Abstract**:

The paper presents Taylor Map Polynomial Neural Network (TMPNN), a novel form of very high-order polynomial regression, in which the same coefficients for a lower-to-moderate-order polynomial regression are iteratively reapplied so as to achieve a higher-order model without the number of coefficients to be fit exploding in the usual curse-of-dimensionality way. This method naturally implements multi-target regression and can capture internal relationships between targets. We also introduce an approach for model interpretation in the form of systems of differential equations. By benchmarking on Feynman regression, UCI, Friedman-1, and real-life industrial datasets, we demonstrate that the proposed method performs comparably to the state-of-the-art regression methods and outperforms them on specific tasks.

----

## [1420] Personalized Reinforcement Learning with a Budget of Policies

**Authors**: *Dmitry Ivanov, Omer Ben-Porat*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29169](https://doi.org/10.1609/aaai.v38i11.29169)

**Abstract**:

Personalization in machine learning (ML) tailors models' decisions to the individual characteristics of users. While this approach has seen success in areas like recommender systems, its expansion into high-stakes fields such as healthcare and autonomous driving is hindered by the extensive regulatory approval processes involved. To address this challenge, we propose a novel framework termed represented Markov Decision Processes (r-MDPs) that is designed to balance the need for personalization with the regulatory constraints. In an r-MDP, we cater to a diverse user population, each with unique preferences, through interaction with a small set of representative policies. Our objective is twofold: efficiently match each user to an appropriate representative policy and simultaneously optimize these policies to maximize overall social welfare. We develop two deep reinforcement learning algorithms that efficiently solve r-MDPs. These algorithms draw inspiration from the principles of classic K-means clustering and are underpinned by robust theoretical foundations. Our empirical investigations, conducted across a variety of simulated environments, showcase the algorithms' ability to facilitate meaningful personalization even under constrained policy budgets. Furthermore, they demonstrate scalability, efficiently adapting to larger policy budgets.

----

## [1421] Delivering Inflated Explanations

**Authors**: *Yacine Izza, Alexey Ignatiev, Peter J. Stuckey, João Marques-Silva*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29170](https://doi.org/10.1609/aaai.v38i11.29170)

**Abstract**:

In the quest for Explainable Artificial Intelligence (XAI) one of the  questions that frequently arises given a decision made by an AI system is, ``why was the decision made in this way?'' Formal approaches to explainability build a formal model of the AI system and use this to reason about the properties of the system.  Given a set of feature values for an instance to be explained, and a resulting decision, a formal abductive explanation  is a  set of features, such that if they take the given value will always lead to the same decision. This explanation is useful, it shows that only some features were used in making the final decision. But it is narrow, it only shows that if the selected features take their given values the decision is unchanged. It is possible that some features may change values and still lead to the same decision. In this paper we formally define  inflated explanations  which is a set of features, and for  each feature a set of values (always including the  value of the instance being explained), such that the decision will remain unchanged, for any of the values allowed for any of the  features in the (inflated) abductive explanation.
Inflated formal explanations are more informative than common abductive explanations since e.g. they allow us to see if the exact value of a feature is important, or it could be any nearby value.  Overall they allow us to better understand the role of each feature in the decision. We show that we can compute inflated explanations for not that much greater cost than abductive explanations, and that we can extend duality results for abductive explanations also to inflated explanations.

----

## [1422] Unified Framework for Diffusion Generative Models in SO(3): Applications in Computer Vision and Astrophysics

**Authors**: *Yesukhei Jagvaral, François Lanusse, Rachel Mandelbaum*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29171](https://doi.org/10.1609/aaai.v38i11.29171)

**Abstract**:

Diffusion-based generative models represent the current state-of-the-art for image generation. However, standard diffusion models are based on Euclidean geometry and do not translate directly to manifold-valued data. In this work, we develop extensions of both score-based generative models (SGMs) and Denoising Diffusion Probabilistic Models (DDPMs) to the Lie group of 3D rotations, SO(3). SO(3) is of particular interest in many disciplines such as robotics, biochemistry and astronomy/cosmology science. Contrary to more general Riemannian manifolds, SO(3) admits a tractable solution to heat diffusion, and allows us to implement efficient training of diffusion models. We apply both SO(3) DDPMs and SGMs to synthetic densities on SO(3) and demonstrate state-of-the-art results. Additionally, we demonstrate the practicality of our model on pose estimation tasks and in predicting correlated galaxy orientations for astrophysics/cosmology.

----

## [1423] GO-DICE: Goal-Conditioned Option-Aware Offline Imitation Learning via Stationary Distribution Correction Estimation

**Authors**: *Abhinav Jain, Vaibhav V. Unhelkar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29172](https://doi.org/10.1609/aaai.v38i11.29172)

**Abstract**:

Offline imitation learning (IL) refers to learning expert behavior solely from demonstrations, without any additional interaction with the environment. Despite significant advances in offline IL, existing techniques find it challenging to learn policies for long-horizon tasks and require significant re-training when task specifications change. Towards addressing these limitations, we present GO-DICE an offline IL technique for goal-conditioned long-horizon sequential tasks. GO-DICE discerns a hierarchy of sub-tasks from demonstrations and uses these to learn separate policies for sub-task transitions and action execution, respectively; this hierarchical policy learning facilitates long-horizon reasoning.Inspired by the expansive DICE-family of techniques, policy learning at both the levels transpires within the space of stationary distributions. Further, both policies are learnt with goal conditioning to minimize need for retraining when task goals change. Experimental results substantiate that GO-DICE outperforms recent baselines, as evidenced by a marked improvement in the completion rate of increasingly challenging pick-and-place Mujoco robotic tasks. GO-DICE is also capable of leveraging imperfect demonstration and partial task segmentation when available, both of which boost task performance relative to learning from expert demonstrations alone.

----

## [1424] Instance-Conditional Timescales of Decay for Non-Stationary Learning

**Authors**: *Nishant Jain, Pradeep Shenoy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29173](https://doi.org/10.1609/aaai.v38i11.29173)

**Abstract**:

Slow concept drift is a ubiquitous, yet under-studied problem in practical machine learning systems. In such settings, although recent data is more indicative of future data, naively prioritizing recent instances runs the risk of losing valuable information from the past. We propose an optimization-driven approach towards balancing instance importance over large training windows. First, we model instance relevance using a mixture of multiple timescales of decay, allowing us to capture rich temporal trends. Second, we learn an auxiliary scorer model that recovers the appropriate mixture of timescales as a function of the instance itself. Finally, we propose a nested optimization objective for learning the scorer, by which it maximizes forward transfer for the learned model. Experiments on a large real-world dataset of 39M photos over a 9 year period show upto 15% relative gains in accuracy compared to other robust learning baselines. We replicate our gains on two collections of real-world datasets for non-stationary learning, and extend our work to continual learning settings where, too, we beat SOTA methods by large margins.

----

## [1425] Universal Weak Coreset

**Authors**: *Ragesh Jaiswal, Amit Kumar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29174](https://doi.org/10.1609/aaai.v38i11.29174)

**Abstract**:

Coresets for k-means and k-median problems yield a small summary of the data, which preserves the clustering cost with respect to any set of k centers. Recently coresets have also been constructed for constrained k-means and k-median problems. However, the notion of coresets has the drawback that (i) they can only be applied in settings where the input points are allowed to have weights, and (ii) in general metric spaces, the size of the coresets can depend logarithmically on the number of points. The notion of weak coresets, which has less stringent requirements than coresets,  has been studied in the context of classical k-means and k-median problems. A weak coreset is a pair (J,S) of subsets of points, where S acts as a summary of the point set and J as a  set of potential centers. This pair satisfies the properties that (i) S is a good summary of the data as long as the k centers are chosen from J only, and (ii) there is a good choice of k centers in J with a cost close to the optimal cost.  We develop this framework, which we call universal weak coresets, for constrained clustering settings. In conjunction with recent coreset constructions for constrained settings, our designs give greater data compression, are conceptually simpler, and apply to a wide range of constrained k-median and k-means problems.

----

## [1426] Transportable Representations for Domain Generalization

**Authors**: *Kasra Jalaldoust, Elias Bareinboim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29175](https://doi.org/10.1609/aaai.v38i11.29175)

**Abstract**:

One key assumption in machine learning literature is that the testing and training data come from the same distribution, which is often violated in practice. The anchors that allow generalizations to take place are causal, and provenient in terms of the stability and modularity of the mechanisms underlying the system of variables. Building on the theory of causal transportability, we define the notion of ``transportable representations", and show that these representations are suitable candidates for the domain generalization task. Specifically, considering that the graphical assumptions about the underlying system are provided, the transportable representations can be characterized accordingly, and the distribution of label conditioned on the representation can be computed in terms of the source distributions. Finally, we relax the assumption of having access to the underlying graph by proving a graphical-invariance duality theorem, which delineates certain probabilistic invariances present in the source data as a sound and complete criterion for generalizable classification. Our findings provide a unifying theoretical basis for several existing approaches to the domain generalization problem.

----

## [1427] Meta-Learning-Based Adaptive Stability Certificates for Dynamical Systems

**Authors**: *Amit Jena, Dileep Kalathil, Le Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29176](https://doi.org/10.1609/aaai.v38i11.29176)

**Abstract**:

This paper addresses the problem of Neural Network (NN) based adaptive stability certification in a dynamical system. The state-of-the-art methods, such as Neural Lyapunov Functions (NLFs), use NN-based formulations to assess the stability of a non-linear dynamical system and compute a Region of Attraction (ROA) in the state space. However, under parametric uncertainty, if the values of system parameters vary over time, the NLF methods fail to adapt to such changes and may lead to conservative stability assessment performance. We circumvent this issue by integrating Model Agnostic Meta-learning (MAML) with NLFs and propose meta-NLFs. In this process, we train a meta-function that adapts to any parametric shifts and updates into an NLF for the system with new test-time parameter values. We demonstrate the stability assessment performance of meta-NLFs on some standard benchmark autonomous dynamical systems.

----

## [1428] Rethinking Dimensional Rationale in Graph Contrastive Learning from Causal Perspective

**Authors**: *Qirui Ji, Jiangmeng Li, Jie Hu, Rui Wang, Changwen Zheng, Fanjiang Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29177](https://doi.org/10.1609/aaai.v38i11.29177)

**Abstract**:

Graph contrastive learning is a general learning paradigm excelling at capturing invariant information from diverse perturbations in graphs. Recent works focus on exploring the structural rationale from graphs, thereby increasing the discriminability of the invariant information. However, such methods may incur in the mis-learning of graph models towards the interpretability of graphs, and thus the learned noisy and task-agnostic information interferes with the prediction of graphs. To this end, with the purpose of exploring the intrinsic rationale of graphs, we accordingly propose to capture the dimensional rationale from graphs, which has not received sufficient attention in the literature. The conducted exploratory experiments attest to the feasibility of the aforementioned roadmap. To elucidate the innate mechanism behind the performance improvement arising from the dimensional rationale, we rethink the dimensional rationale in graph contrastive learning from a causal perspective and further formalize the causality among the variables in the pre-training stage to build the corresponding structural causal model. On the basis of the understanding of the structural causal model, we propose the dimensional rationale-aware graph contrastive learning approach, which introduces a learnable dimensional rationale acquiring network and a redundancy reduction constraint. The learnable dimensional rationale acquiring network is updated by leveraging a bi-level meta-learning technique, and the redundancy reduction constraint disentangles the redundant features through a decorrelation process during learning. Empirically, compared with state-of-the-art methods, our method can yield significant performance boosts on various benchmarks with respect to discriminability and transferability. The code implementation of our method is available at https://github.com/ByronJi/DRGCL.

----

## [1429] MusER: Musical Element-Based Regularization for Generating Symbolic Music with Emotion

**Authors**: *Shulei Ji, Xinyu Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29178](https://doi.org/10.1609/aaai.v38i11.29178)

**Abstract**:

Generating music with emotion is an important task in automatic music generation, in which emotion is evoked through a variety of musical elements (such as pitch and duration) that change over time and collaborate with each other. However, prior research on deep learning-based emotional music generation has rarely explored the contribution of different musical elements to emotions, let alone the deliberate manipulation of these elements to alter the emotion of music, which is not conducive to fine-grained element-level control over emotions. To address this gap, we present a novel approach employing musical element-based regularization in the latent space to disentangle distinct elements, investigate their roles in distinguishing emotions, and further manipulate elements to alter musical emotions. Specifically, we propose a novel VQ-VAE-based model named MusER. MusER incorporates a regularization loss to enforce the correspondence between the musical element sequences and the specific dimensions of latent variable sequences, providing a new solution for disentangling discrete sequences. Taking advantage of the disentangled latent vectors, a two-level decoding strategy that includes multiple decoders attending to latent vectors with different semantics is devised to better predict the elements. By visualizing latent space, we conclude that MusER yields a disentangled and interpretable latent space and gain insights into the contribution of distinct elements to the emotional dimensions (i.e., arousal and valence). Experimental results demonstrate that MusER outperforms the state-of-the-art models for generating emotional music in both objective and subjective evaluation. Besides, we rearrange music through element transfer and attempt to alter the emotion of music by transferring emotion-distinguishable elements.

----

## [1430] FedFixer: Mitigating Heterogeneous Label Noise in Federated Learning

**Authors**: *Xinyuan Ji, Zhaowei Zhu, Wei Xi, Olga Gadyatskaya, Zilong Song, Yong Cai, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29179](https://doi.org/10.1609/aaai.v38i11.29179)

**Abstract**:

Federated Learning (FL) heavily depends on label quality for its performance. However, the label distribution among individual clients is always both noisy and heterogeneous. The high loss incurred by client-specific samples in heterogeneous label noise poses challenges for distinguishing between client-specific and noisy label samples, impacting the effectiveness of existing label noise learning approaches. To tackle this issue, we propose FedFixer, where the personalized model is introduced to cooperate with the global model to effectively select clean client-specific samples. In the dual models, updating the personalized model solely at a local level can lead to overfitting on noisy data due to limited samples, consequently affecting both the local and global models’ performance. To mitigate overfitting, we address this concern from two perspectives. Firstly, we employ a confidence regularizer to alleviate the impact of unconfident predictions caused by label noise. Secondly, a distance regularizer is implemented to constrain the disparity between the personalized and global models. We validate the effectiveness of FedFixer through extensive experiments on benchmark datasets. The results demonstrate that FedFixer can perform well in filtering noisy label samples on different clients, especially in highly heterogeneous label noise scenarios.

----

## [1431] Stratified GNN Explanations through Sufficient Expansion

**Authors**: *Yuwen Ji, Lei Shi, Zhimeng Liu, Ge Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29180](https://doi.org/10.1609/aaai.v38i11.29180)

**Abstract**:

Explaining the decisions made by Graph Neural Networks (GNNs) is vital for establishing trust and ensuring fairness in critical applications such as medicine and science. The prevalence of hierarchical structure in real-world graphs/networks raises an important question on GNN interpretability: "On each level of the graph structure, which specific fraction imposes the highest influence over the prediction?" Currently, the prevailing two categories of methods are incapable of achieving multi-level GNN explanation due to their flat or motif-centric nature. In this work, we formulate the problem of learning multi-level explanations out of GNN models and introduce a stratified explainer module, namely STFExplainer, that utilizes the concept of sufficient expansion to generate explanations on each stratum. Specifically, we learn a higher-level subgraph generator by leveraging both hierarchical structure and GNN-encoded input features. Experiment results on both synthetic and real-world datasets demonstrate the superiority of our stratified explainer on standard interpretability tasks and metrics such as fidelity and explanation recall, with an average improvement of 11% and 8% over the best alternative on each data type. The case study on material domains also confirms the value of our approach through detected multi-level graph patterns accurately reconstructing the knowledge-based ground truth.

----

## [1432] FedLPS: Heterogeneous Federated Learning for Multiple Tasks with Local Parameter Sharing

**Authors**: *Yongzhe Jia, Xuyun Zhang, Amin Beheshti, Wanchun Dou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29181](https://doi.org/10.1609/aaai.v38i11.29181)

**Abstract**:

Federated Learning (FL) has emerged as a promising solution in Edge Computing (EC) environments to process the proliferation of data generated by edge devices. By collaboratively optimizing the global machine learning models on distributed edge devices, FL circumvents the need for transmitting raw data and enhances user privacy. Despite practical successes, FL still confronts significant challenges including constrained edge device resources, multiple tasks deployment, and data heterogeneity. However, existing studies focus on mitigating the FL training costs of each single task whereas neglecting the resource consumption across multiple tasks in heterogeneous FL scenarios. In this paper, we propose Heterogeneous Federated Learning with Local Parameter Sharing (FedLPS) to fill this gap. FedLPS leverages principles from transfer learning to facilitate the deployment of multiple tasks on a single device by dividing the local model into a shareable encoder and task-specific encoders. To further reduce resource consumption, a channel-wise model pruning algorithm that shrinks the footprint of local models while accounting for both data and system heterogeneity is employed in FedLPS. Additionally, a novel heterogeneous model aggregation algorithm is proposed to aggregate the heterogeneous predictors in FedLPS. We implemented the proposed FedLPS on a real FL platform and compared it with state-of-the-art (SOTA) FL frameworks. The experimental results on five popular datasets and two modern DNN models illustrate that the proposed FedLPS significantly outperforms the SOTA FL frameworks by up to 4.88% and reduces the computational resource consumption by 21.3%. Our code is available at: https://github.com/jyzgh/FedLPS.

----

## [1433] Long-Tailed Partial Label Learning by Head Classifier and Tail Classifier Cooperation

**Authors**: *Yuheng Jia, Xiaorui Peng, Ran Wang, Min-Ling Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29182](https://doi.org/10.1609/aaai.v38i11.29182)

**Abstract**:

In partial label learning (PLL), each instance is associated with a set of candidate labels, among which only one is correct. The traditional PLL almost all implicitly assume that the distribution of the classes is balanced. However, in real-world applications, the distribution of the classes is imbalanced or long-tailed, leading to the long-tailed partial label learning problem. The previous methods solve this problem mainly by ameliorating the ability to learn in the tail classes, which will sacrifice the performance of the head classes. While keeping the performance of the head classes may degrade the performance of the tail classes. Therefore, in this paper, we construct two classifiers, i.e., a head classifier for keeping the performance of dominant classes and a tail classifier for improving the performance of the tail classes. Then, we propose a classifier weight estimation module to automatically estimate the shot belongingness (head class or tail class) of the samples and allocate the weights for the head classifier and tail classifier when making prediction. This cooperation improves the prediction ability for both the head classes and the tail classes. The experiments on the benchmarks demonstrate the proposed approach improves the accuracy of the SOTA methods by a substantial margin. Code and data are available at: https://github.com/pruirui/HTC-LTPLL.

----

## [1434] Which Is More Effective in Label Noise Cleaning, Correction or Filtering?

**Authors**: *Gaoxia Jiang, Jia Zhang, Xuefei Bai, Wenjian Wang, Deyu Meng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29183](https://doi.org/10.1609/aaai.v38i11.29183)

**Abstract**:

Most noise cleaning methods adopt one of the correction and filtering modes to build robust models. However, their effectiveness, applicability, and hyper-parameter insensitivity have not been carefully studied. We compare the two cleaning modes via a rebuilt error bound in noisy environments. At the dataset level, Theorem 5 implies that correction is more effective than filtering when the cleaned datasets have close noise rates. At the sample level, Theorem 6 indicates that confident label noises (large noise probabilities) are more suitable to be corrected, and unconfident noises (medium noise probabilities) should be filtered. Besides, an imperfect hyper-parameter may have fewer negative impacts on filtering than correction. Unlike existing methods with a single cleaning mode, the proposed Fusion cleaning framework of Correction and Filtering (FCF) combines the advantages of different modes to deal with diverse suspicious labels. Experimental results demonstrate that our FCF method can achieve state-of-the-art performance on benchmark datasets.

----

## [1435] Navigating Real-World Partial Label Learning: Unveiling Fine-Grained Images with Attributes

**Authors**: *Haoran Jiang, Zhihao Sun, Yingjie Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29184](https://doi.org/10.1609/aaai.v38i11.29184)

**Abstract**:

Partial label learning (PLL), a significant research area, addresses the challenge of annotating each sample with a candidate label set containing the true label when obtaining accurate labels is infeasible.  However, existing PLL methods often rely on generic datasets like CIFAR, where annotators  can readily differentiate candidate labels and are unlikely to confuse, making it less realistic for real-world partial label applications. In response, our research focuses on a rarely studied problem, PLL on fine-grained images with attributes. And we propose a novel framework called Shared to Learn, Distinct to Disambiguate (SoDisam). Within the candidate label set, the categories  may exhibit numerous shared attribute features, posing a challenge in accurately distinguishing them. Rather than perceiving it as an impediment, we capitalize on these shared attributes as definitive sources of supervision. This insight guides us to learn  attribute space visual representation to focus on the information from these shared attributes. Moreover, we introduce an attribute attention mechanism tailored to harness the remaining distinct  attributes. This mechanism directs the originally holistic feature towards specific regions, capturing corresponding discriminative features. In addition, a dynamic disambiguation module is introduced, continuously adjusting the two aforementioned mechanisms and achieve the final disambiguation process. Extensive experiments demonstrate the effectiveness of our approach on fine-grained partial label datasets. The proposed SoDisam framework not only addresses the challenges associated with fine-grained partial label learning but also provides a more realistic representation of real-world partial label scenarios.

----

## [1436] DHGCN: Dynamic Hop Graph Convolution Network for Self-Supervised Point Cloud Learning

**Authors**: *Jincen Jiang, Lizhi Zhao, Xuequan Lu, Wei Hu, Imran Razzak, Meili Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29185](https://doi.org/10.1609/aaai.v38i11.29185)

**Abstract**:

Recent works attempt to extend Graph Convolution Networks (GCNs) to point clouds for classification and segmentation tasks. These works tend to sample and group points to create smaller point sets locally and mainly focus on extracting local features through GCNs, while ignoring the relationship between point sets. In this paper, we propose the Dynamic Hop Graph Convolution Network (DHGCN) for explicitly learning the contextual relationships between the voxelized point parts, which are treated as graph nodes. Motivated by the intuition that the contextual information between point parts lies in the pairwise adjacent relationship, which can be depicted by the hop distance of the graph quantitatively, we devise a novel self-supervised part-level hop distance reconstruction task and design a novel loss function accordingly to facilitate training. In addition, we propose the Hop Graph Attention (HGA), which takes the learned hop distance as input for producing attention weights to allow edge features to contribute distinctively in aggregation. Eventually, the proposed DHGCN is a plug-and-play module that is compatible with point-based backbone networks. Comprehensive experiments on different backbones and tasks demonstrate that our self-supervised method achieves state-of-the-art performance. Our source codes are available at: https://github.com/Jinec98/DHGCN.

----

## [1437] FMRNet: Image Deraining via Frequency Mutual Revision

**Authors**: *Kui Jiang, Junjun Jiang, Xianming Liu, Xin Xu, Xianzheng Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29186](https://doi.org/10.1609/aaai.v38i11.29186)

**Abstract**:

The wavelet transform has emerged as a powerful tool in deciphering structural information within images. And now, the latest research suggests that combining the prowess of wavelet transform with neural networks can lead to unparalleled image deraining results. By harnessing the strengths of both the spatial domain and frequency space, this innovative approach is poised to revolutionize the field of image processing. The fascinating challenge of developing a comprehensive framework that takes into account the intrinsic frequency property and the correlation between rain residue and background is yet to be fully explored. In this work, we propose to investigate the potential relationships among rain-free and residue components at the frequency domain, forming a frequency mutual revision network (FMRNet) for image deraining. Specifically, we explore the  mutual representation of rain residue and background components at frequency domain, so as to better separate the rain layer from clean background while preserving structural textures of the degraded images. Meanwhile, the rain distribution prediction from the low-frequency coefficient, which can be seen as the degradation prior is used to refine the separation of rain residue and background components. Inversely, the updated rain residue is used to benefit the low-frequency rain distribution prediction, forming the multi-layer mutual learning. Extensive experiments demonstrate that our proposed FMRNet delivers significant performance gains for seven datasets on image deraining task, surpassing the state-of-the-art method ELFormer by 1.14 dB in PSNR on the Rain100L dataset, while with similar computation cost. Code and retrained models are available at https://github.com/kuijiang94/FMRNet.

----

## [1438] Racing Control Variable Genetic Programming for Symbolic Regression

**Authors**: *Nan Jiang, Yexiang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29187](https://doi.org/10.1609/aaai.v38i11.29187)

**Abstract**:

Symbolic regression, as one of the most crucial tasks in AI for science, discovers governing equations from experimental data. Popular approaches based on genetic programming, Monte Carlo tree search, or deep reinforcement learning learn symbolic regression from a fixed dataset. These methods require massive datasets and long training time especially when learning complex equations involving many variables. Recently, Control Variable Genetic Programming (CVGP) has been introduced which accelerates the regression process by discovering equations from designed control variable experiments. However, the set of experiments is fixed a-priori in CVGP and we observe that sub-optimal selection of experiment schedules delay the discovery process significantly. To overcome this limitation, we propose Racing Control Variable Genetic Programming (Racing-CVGP), which carries out multiple experiment schedules simultaneously. A selection scheme similar to that used in selecting good symbolic equations in the genetic programming process is implemented to ensure that promising experiment schedules eventually win over the average ones. The unfavorable schedules are terminated early to save time for the promising ones. We evaluate Racing-CVGP on several synthetic and real-world datasets corresponding to true physics laws. We demonstrate that Racing-CVGP outperforms CVGP and a series of symbolic regressors which discover equations from fixed datasets.

----

## [1439] Learning Diverse Risk Preferences in Population-Based Self-Play

**Authors**: *Yuhua Jiang, Qihan Liu, Xiaoteng Ma, Chenghao Li, Yiqin Yang, Jun Yang, Bin Liang, Qianchuan Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29188](https://doi.org/10.1609/aaai.v38i11.29188)

**Abstract**:

Among the remarkable successes of Reinforcement Learning (RL), self-play algorithms have played a crucial role in solving competitive games. However, current self-play RL methods commonly optimize the agent to maximize the expected win-rates against its current or historical copies, resulting in a limited strategy style and a tendency to get stuck in local optima. To address this limitation, it is important to improve the diversity of policies, allowing the agent to break stalemates and enhance its robustness when facing with different opponents.  In this paper, we present a novel perspective to promote diversity by considering that agents could have diverse risk preferences in the face of uncertainty. To achieve this, we introduce a novel reinforcement learning algorithm called Risk-sensitive Proximal Policy Optimization (RPPO), which smoothly interpolates between worst-case and best-case policy learning, enabling policy learning with desired risk preferences. Furthermore, by seamlessly integrating RPPO with population-based self-play, agents in the population optimize dynamic risk-sensitive objectives using experiences gained from playing against diverse opponents. Our empirical results demonstrate that our method achieves comparable or superior performance in competitive games and, importantly, leads to the emergence of diverse behavioral modes. Code is available at https://github.com/Jackory/RPBT.

----

## [1440] Deep Incomplete Multi-View Learning Network with Insufficient Label Information

**Authors**: *Zhangqi Jiang, Tingjin Luo, Xinyan Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29189](https://doi.org/10.1609/aaai.v38i11.29189)

**Abstract**:

Due to the efficiency of integrating semantic consensus and complementary information across different views, multi-view classification methods have attracted much attention in recent years. However, multi-view data often suffers from both the miss of view features and insufficient label information, which significantly decrease the performance of traditional multi-view classification methods in practice. Learning for such simultaneous lack of feature and label is crucial but rarely studied. To tackle these problems, we propose a novel Deep Incomplete Multi-view Learning Network (DIMvLN) by incorporating graph networks and semi-supervised learning in this paper. Specifically, DIMvLN firstly designs the deep graph networks to effectively recover missing data with assigning pseudo-labels of large amounts of unlabeled instances and refine the incomplete feature information. Meanwhile, to enhance the label information, a novel pseudo-label generation strategy with the similarity constraints of unlabeled instances is proposed to exploit additional supervisory information and guide the completion module to preserve more semantic information of absent multi-view data. Besides, we design view-specific representation extractors with the autoencoder structure and contrastive loss to learn high-level semantic representations for each view, promote cross-view consistencies and augment the separability between different categories. Finally, extensive experimental results demonstrate the effectiveness of our DIMvLN, attaining noteworthy performance improvements compared to state-of-the-art competitors on several public benchmark datasets. Code will be available at GitHub.

----

## [1441] Provably Convergent Federated Trilevel Learning

**Authors**: *Yang Jiao, Kai Yang, Tiancheng Wu, Chengtao Jian, Jianwei Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29190](https://doi.org/10.1609/aaai.v38i11.29190)

**Abstract**:

Trilevel learning, also called trilevel optimization (TLO), has been recognized as a powerful modelling tool for hierarchical decision process and widely applied in many machine learning applications, such as robust neural architecture search, hyperparameter optimization, and domain adaptation. Tackling TLO problems has presented a great challenge due to their nested decision-making structure. In addition, existing works on TLO face the following key challenges: 1) they all focus on the non-distributed setting, which may lead to privacy breach; 2) they do not offer any non-asymptotic convergence analysis which characterizes how fast an algorithm converges. To address the aforementioned challenges, this paper proposes an asynchronous federated trilevel optimization method to solve TLO problems. The proposed method utilizes u-cuts to construct a hyper-polyhedral approximation for the TLO problem and solve it in an asynchronous manner. We demonstrate that the proposed u-cuts are applicable to not only convex functions but also a wide range of non-convex functions that meet the u-weakly convex assumption. Furthermore, we theoretically analyze the non-asymptotic convergence rate for the proposed method by showing its iteration complexity to obtain ϵ-stationary point is upper bounded by O(1/ϵ²). Extensive experiments on real-world datasets have been conducted to elucidate the superiority of the proposed method, e.g., it has a faster convergence rate with a maximum acceleration of approximately 80%.

----

## [1442] Performative Federated Learning: A Solution to Model-Dependent and Heterogeneous Distribution Shifts

**Authors**: *Kun Jin, Tongxin Yin, Zhongzhu Chen, Zeyu Sun, Xueru Zhang, Yang Liu, Mingyan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29191](https://doi.org/10.1609/aaai.v38i11.29191)

**Abstract**:

We consider a federated learning (FL) system consisting of multiple clients and a server, where the clients aim to collaboratively learn a common decision model from their distributed data. Unlike the conventional FL framework that assumes the client's data is static, we consider scenarios where the clients' data distributions may be reshaped by the deployed decision model. In this work, we leverage the idea of distribution shift mappings in performative prediction to formalize this model-dependent data distribution shift and propose a performative FL framework. 
We first introduce necessary and sufficient conditions for the existence of a unique performative stable solution  and characterize its distance to the performative optimal solution. Then we propose the performative FedAvg algorithm and show that it converges to the performative stable solution at a rate of  O(1/T) under both full and partial participation schemes.
In particular, we use novel proof techniques and show how the clients' heterogeneity influences the convergence. Numerical results validate our analysis and provide valuable insights into real-world applications.

----

## [1443] Fractional Deep Reinforcement Learning for Age-Minimal Mobile Edge Computing

**Authors**: *Lyudong Jin, Ming Tang, Meng Zhang, Hao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29192](https://doi.org/10.1609/aaai.v38i11.29192)

**Abstract**:

Mobile edge computing (MEC) is a promising paradigm for real-time applications with intensive computational needs (e.g., autonomous driving), as it can reduce the processing delay. In this work, we focus on the timeliness of computational-intensive updates, measured by Age-of-Information (AoI), and study how to jointly optimize the task updating and offloading  policies for AoI with fractional form. Specifically, we consider edge load dynamics and formulate a task scheduling problem to minimize the expected time-average AoI. The uncertain edge load dynamics, the nature of the fractional objective, and hybrid continuous-discrete action space (due to the joint optimization) make this problem challenging and existing approaches not directly applicable. To this end, we propose a fractional reinforcement learning (RL) framework and prove its convergence. We further design a model-free fractional deep RL (DRL) algorithm, where each device makes scheduling decisions with the hybrid action space without knowing the system dynamics and decisions of other devices. Experimental results show that our proposed algorithms reduce the average AoI by up to 57.6% compared with several non-fractional benchmarks.

----

## [1444] Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs

**Authors**: *Tianyuan Jin, Hao-Lun Hsu, William Chang, Pan Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29193](https://doi.org/10.1609/aaai.v38i11.29193)

**Abstract**:

We study the multi-agent multi-armed bandit (MAMAB) problem, where agents are factored into overlapping groups. Each group represents a hyperedge, forming a hypergraph over the agents. At each round of interaction, the learner pulls a joint arm (composed of individual arms for each agent) and receives a reward according to the hypergraph structure. Specifically, we assume there is a local reward for each hyperedge, and the reward of the joint arm is the sum of these local rewards. Previous work introduced the multi-agent Thompson sampling (MATS) algorithm and derived a Bayesian regret bound. However, it remains an open problem how to derive a frequentist regret bound for Thompson sampling in this multi-agent setting. To address these issues, we propose an efficient variant of MATS, the epsilon-exploring Multi-Agent Thompson Sampling (eps-MATS) algorithm, which performs MATS exploration with probability epsilon while adopts a greedy policy otherwise. We prove that eps-MATS achieves a worst-case frequentist regret bound that is sublinear in both the time horizon and the local arm size. We also derive a lower bound for this setting, which implies our frequentist regret upper bound is optimal up to constant and logarithm terms, when the hypergraph is sufficiently sparse. Thorough experiments on standard MAMAB problems demonstrate the superior performance and the improved computational efficiency of eps-MATS compared with existing algorithms in the same setting.

----

## [1445] GLDL: Graph Label Distribution Learning

**Authors**: *Yufei Jin, Richard Gao, Yi He, Xingquan Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29194](https://doi.org/10.1609/aaai.v38i11.29194)

**Abstract**:

Label Distribution Learning (LDL), as a more general learning setting than generic single-label and multi-label learning, has been commonly used in computer vision and many other applications. To date, existing LDL approaches are designed and applied to data without considering the interdependence between instances. In this paper, we propose a Graph Label Distribution Learning (GLDL) framework, which explicitly models three types of relationships: instance-instance, label-label, and instance-label, to learn the label distribution for networked data. A label-label network is learned to capture label-to-label correlation, through which GLDL can accurately learn label distributions for nodes. Dual graph convolution network (GCN) Co-training with heterogeneous message passing ensures two GCNs, one focusing on instance-instance relationship and the other one targeting label-label correlation, are jointly trained such that instance-instance relationship can help induce label-label correlation and vice versa. Our theoretical study derives the error bound of GLDL. For verification, four benchmark datasets with label distributions for nodes are created using common graph benchmarks. The experiments show that considering dependency helps learn better label distributions for networked data, compared to state-of-the-art LDL baseline. In addition, GLDL not only outperforms simple GCN and graph attention networks (GAT) using distribution loss but is also superior to its variant considering label-label relationship as a static network. GLDL and its benchmarks are the first research endeavors to address LDL for graphs. Code and benchmark data are released for public access.

----

## [1446] Sterling: Synergistic Representation Learning on Bipartite Graphs

**Authors**: *Baoyu Jing, Yuchen Yan, Kaize Ding, Chanyoung Park, Yada Zhu, Huan Liu, Hanghang Tong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29195](https://doi.org/10.1609/aaai.v38i12.29195)

**Abstract**:

A fundamental challenge of bipartite graph representation learning is how to extract informative node embeddings. Self-Supervised Learning (SSL) is a promising paradigm to address this challenge. Most recent bipartite graph SSL methods are based on contrastive learning which learns embeddings by discriminating positive and negative node pairs. Contrastive learning usually requires a large number of negative node pairs, which could lead to computational burden and semantic errors. In this paper, we introduce a novel synergistic representation learning model (STERLING) to learn node embeddings without negative node pairs. STERLING preserves the unique local and global synergies in bipartite graphs. The local synergies are captured by maximizing the similarity of the inter-type and intra-type positive node pairs, and the global synergies are captured by maximizing the mutual information of co-clusters. Theoretical analysis demonstrates that STERLING could improve the connectivity between different node types in the embedding space. Extensive empirical evaluation on various benchmark datasets and tasks demonstrates the effectiveness of STERLING for extracting node embeddings.

----

## [1447] FoX: Formation-Aware Exploration in Multi-Agent Reinforcement Learning

**Authors**: *Yonghyeon Jo, Sunwoo Lee, Junghyuk Yeom, Seungyul Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29196](https://doi.org/10.1609/aaai.v38i12.29196)

**Abstract**:

Recently, deep multi-agent reinforcement learning (MARL) has gained significant popularity due to its success in various cooperative multi-agent tasks. However, exploration still remains a challenging problem in MARL due to the partial observability of the agents and the exploration space that can grow exponentially as the number of agents increases. Firstly, in order to address the scalability issue of the exploration space, we define a formation-based equivalence relation on the exploration space and aim to reduce the search space by exploring only meaningful states in different formations. Then, we propose a novel formation-aware exploration (FoX) framework that encourages partially observable agents to visit the states in diverse formations by guiding them to be well aware of their current formation solely based on their own observations. Numerical results show that the proposed FoX framework significantly outperforms the state-of-the-art MARL algorithms on Google Research Football (GRF) and sparse Starcraft II multi-agent challenge (SMAC) tasks.

----

## [1448] FLAME: A Small Language Model for Spreadsheet Formulas

**Authors**: *Harshit Joshi, Abishai Ebenezer, José Pablo Cambronero Sánchez, Sumit Gulwani, Aditya Kanade, Vu Le, Ivan Radicek, Gust Verbruggen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29197](https://doi.org/10.1609/aaai.v38i12.29197)

**Abstract**:

Spreadsheets are a vital tool for end-user data management. Using large language
models for formula authoring assistance in these environments can be difficult,
as these models are expensive to train and challenging to deploy due to their
size (up to billions of parameters). We present FLAME, a transformer-based model
trained exclusively on Excel formulas that leverages domain insights to achieve
competitive performance while being substantially smaller (60M parameters) and
training on two orders of magnitude less data. We curate a training dataset
using sketch deduplication, introduce an Excel-specific formula tokenizer, and
use domain-specific versions of masked span prediction and noisy auto-encoding
as pre-training objectives. We evaluate FLAME on formula repair, formula
completion, and similarity-based formula retrieval. FLAME can outperform much
larger models, such as the Davinci (175B) and Cushman (12B) variants of Codex
and CodeT5 (220M), in 10 of 14 evaluation settings for the repair and completion
tasks. For formula retrieval, FLAME outperforms CodeT5, CodeBERT, and
GraphCodeBERT.

----

## [1449] Towards Safe Policy Learning under Partial Identifiability: A Causal Approach

**Authors**: *Shalmali Joshi, Junzhe Zhang, Elias Bareinboim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29198](https://doi.org/10.1609/aaai.v38i12.29198)

**Abstract**:

Learning personalized treatment policies is a formative challenge in many real-world applications, including in healthcare, econometrics, artificial intelligence. However, the effectiveness of candidate policies is not always identifiable, i.e., it is not uniquely computable from the combination of the available data and assumptions about the generating mechanisms. This paper studies policy learning from data collected in various non-identifiable settings, i.e., (1) observational studies with unobserved confounding; (2) randomized experiments with partial observability; and (3) their combinations. We derive sharp, closed-formed bounds from observational and experimental data over the conditional treatment effects. Based on these novel bounds, we further characterize the problem of safe policy learning and develop an algorithm that trains a policy from data guaranteed to achieve, at least, the performance of the baseline policy currently deployed. Finally, we validate our proposed algorithm on synthetic data and a large clinical trial, demonstrating that it guarantees safe behaviors and robust performance.

----

## [1450] Patch-Wise Graph Contrastive Learning for Image Translation

**Authors**: *Chanyong Jung, Gihyun Kwon, Jong Chul Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29199](https://doi.org/10.1609/aaai.v38i12.29199)

**Abstract**:

Recently, patch-wise contrastive learning is drawing attention for the image translation by exploring the semantic correspondence between the input image and the output image. To further explore the patch-wise topology for high-level semantic understanding, here we exploit the graph neural network to capture the topology-aware features. Specifically, we construct the graph based on the patch-wise similarity from a pretrained encoder, whose adjacency matrix is shared to enhance the consistency of patch-wise relation between the input and the output. Then, we obtain the node feature from the graph neural network, and enhance the correspondence between the nodes by increasing mutual information using the contrastive loss. In order to capture the hierarchical semantic structure, we further propose the graph pooling. Experimental results demonstrate the state-of-art results for the image translation thanks to the semantic encoding by the constructed graphs.

----

## [1451] NN-Steiner: A Mixed Neural-Algorithmic Approach for the Rectilinear Steiner Minimum Tree Problem

**Authors**: *Andrew B. Kahng, Robert R. Nerem, Yusu Wang, Chien-Yi Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29200](https://doi.org/10.1609/aaai.v38i12.29200)

**Abstract**:

Recent years have witnessed rapid advances in the use of neural networks to solve combinatorial optimization problems. Nevertheless, designing the "right" neural model that can effectively handle a given optimization problem can be challenging, and often there is no theoretical understanding or justification of the resulting neural model. In this paper, we focus on the rectilinear Steiner minimum tree (RSMT) problem, which is of critical importance in IC layout design and as a result has attracted numerous heuristic approaches in the VLSI literature. Our contributions are two-fold. On the methodology front, we propose NN-Steiner which is a novel mixed neural-algorithmic framework for computing RSMTs that leverages the celebrated PTAS algorithmic framework of Arora to solve this problem (and other geometric optimization problems). Our NN-Steiner replaces key algorithmic components within Arora's PTAS by suitable neural components. In particular, NN-Steiner only needs four neural network (NN) components that are called repeatedly within an algorithmic framework. Crucially, each of the four NN components is only of bounded size independent of input size, and thus easy to train. Furthermore, as the NN component is learning a generic algorithmic step, once learned, the resulting mixed neural-algorithmic framework generalizes to much larger instances not seen in training. Our NN-Steiner, to our best knowledge, is the first neural architecture of bounded size that has capacity to approximately solve RSMT (and variants). On the empirical front, we show how NN-Steiner can be implemented and demonstrate the effectiveness of our resulting approach, especially in terms of generalization, by comparing with state-of-the-art methods (both neural and non-neural based).

----

## [1452] Measuring Self-Supervised Representation Quality for Downstream Classification Using Discriminative Features

**Authors**: *Neha Mukund Kalibhat, Kanika Narang, Hamed Firooz, Maziar Sanjabi, Soheil Feizi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29201](https://doi.org/10.1609/aaai.v38i12.29201)

**Abstract**:

Self-supervised learning (SSL) has shown impressive results in downstream classification tasks. However, there is limited work in understanding their failure modes and interpreting their learned representations. In this paper, we study the representation space of state-of-the-art self-supervised models including SimCLR, SwaV, MoCo, BYOL, DINO, SimSiam, VICReg and Barlow Twins. Without the use of class label information, we discover discriminative features that correspond to unique physical attributes in images, present mostly in correctly-classified representations. Using these features, we can compress the representation space by up to$40% without significantly affecting linear classification performance. We then propose Self-Supervised Representation Quality Score (or Q-Score), an unsupervised score that can reliably predict if a given sample is likely to be mis-classified during linear evaluation, achieving AUPRC of 91.45 on ImageNet-100 and 78.78 on ImageNet-1K. Q-Score can also be used as a regularization term on pre-trained encoders to remedy low-quality representations. Fine-tuning with Q-Score regularization can boost the linear probing accuracy of SSL models by up to 5.8% on ImageNet-100 and 3.7% on ImageNet-1K compared to their baselines. Finally, using gradient heatmaps and Salient ImageNet masks, we define a metric to quantify the interpretability of each representation. We show that discriminative features are strongly correlated to core attributes and, enhancing these features through Q-score regularization makes SSL representations more interpretable.

----

## [1453] Recall-Oriented Continual Learning with Generative Adversarial Meta-Model

**Authors**: *Haneol Kang, Dong-Wan Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29202](https://doi.org/10.1609/aaai.v38i12.29202)

**Abstract**:

The stability-plasticity dilemma is a major challenge in continual learning, as it involves balancing the conflicting objectives of maintaining performance on previous tasks while learning new tasks. In this paper, we propose the recalloriented continual learning framework to address this challenge. Inspired by the human brain’s ability to separate the mechanisms responsible for stability and plasticity, our framework consists of a two-level architecture where an inference network effectively acquires new knowledge and a generative network recalls past knowledge when necessary. In particular, to maximize the stability of past knowledge, we investigate the complexity of knowledge depending on different representations, and thereby introducing generative adversarial meta-model (GAMM) that incrementally learns task-specific parameters instead of input data samples of the task. Through our experiments, we show that our framework not only effectively learns new knowledge without any disruption but also achieves high stability of previous knowledge in both task-aware and task-agnostic learning scenarios. Our code is available at: https://github.com/bigdata-inha/recall-orientedcl-framework.

----

## [1454] Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study

**Authors**: *Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29203](https://doi.org/10.1609/aaai.v38i12.29203)

**Abstract**:

In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.

----

## [1455] Neural Oscillators for Generalization of Physics-Informed Machine Learning

**Authors**: *Taniya Kapoor, Abhishek Chandra, Daniel M. Tartakovsky, Hongrui Wang, Alfredo Núñez, Rolf P. B. J. Dollevoet*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29204](https://doi.org/10.1609/aaai.v38i12.29204)

**Abstract**:

A primary challenge of physics-informed machine learning (PIML) is its generalization beyond the training domain, especially when dealing with complex physical problems represented by partial differential equations (PDEs). This paper aims to enhance the generalization capabilities of PIML, facilitating practical, real-world applications where accurate predictions in unexplored regions are crucial. We leverage the inherent causality and temporal sequential characteristics of PDE solutions to fuse PIML models with recurrent neural architectures based on systems of ordinary differential equations, referred to as neural oscillators. Through effectively capturing long-time dependencies and mitigating the exploding and vanishing gradient problem, neural oscillators foster improved generalization in PIML tasks. Extensive experimentation involving time-dependent nonlinear PDEs and biharmonic beam equations demonstrates the efficacy of the proposed approach. Incorporating neural oscillators outperforms existing state-of-the-art methods on benchmark problems across various metrics. Consequently, the proposed method improves the generalization capabilities of PIML, providing accurate solutions for extrapolation and prediction beyond the training data.

----

## [1456] SHAP@k: Efficient and Probably Approximately Correct (PAC) Identification of Top-K Features

**Authors**: *Sanjay Kariyappa, Leonidas Tsepenekas, Freddy Lécué, Daniele Magazzeni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29205](https://doi.org/10.1609/aaai.v38i12.29205)

**Abstract**:

The SHAP framework provides a principled method to explain the predictions of a model by computing feature importance. Motivated by applications in finance, we introduce the Top-k Identification Problem (TkIP) (and its ordered variant TkIP- O), where the objective is to identify the subset (or ordered subset for TkIP-O) of k features corresponding to the highest SHAP values with PAC guarantees. While any sampling-based method that estimates SHAP values (such as KernelSHAP and SamplingSHAP) can be trivially adapted to solve TkIP, doing so is highly sample inefficient. Instead, we leverage the connection between SHAP values and multi-armed bandits (MAB) to show that both TkIP and TkIP-O can be reduced to variants of problems in MAB literature. This reduction allows us to use insights from the MAB literature to develop sample-efficient variants of KernelSHAP and SamplingSHAP. We propose KernelSHAP@k and SamplingSHAP@k for solving TkIP; along with KernelSHAP-O and SamplingSHAP-O to solve the ordering problem in TkIP-O. We perform extensive experiments using several credit-related datasets to show that our methods offer significant improvements of up to 40× in sample efficiency and 39× in runtime.

----

## [1457] Communication-Efficient Collaborative Regret Minimization in Multi-Armed Bandits

**Authors**: *Nikolai Karpov, Qin Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29206](https://doi.org/10.1609/aaai.v38i12.29206)

**Abstract**:

In this paper, we study the collaborative learning model, which concerns the tradeoff between parallelism and communication overhead in multi-agent multi-armed bandits.  For regret minimization in multi-armed bandits, we present the first set of tradeoffs between the number of rounds of communication between the agents and the regret of the collaborative learning process.

----

## [1458] Adversarially Balanced Representation for Continuous Treatment Effect Estimation

**Authors**: *Amirreza Kazemi, Martin Ester*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29207](https://doi.org/10.1609/aaai.v38i12.29207)

**Abstract**:

Individual treatment effect (ITE) estimation requires adjusting for the covariate shift between populations with different treatments, and deep representation learning has shown great promise in learning a balanced representation of covariates. However the existing methods mostly consider the scenario of binary treatments. In this paper, we consider the more practical and challenging scenario in which the treatment is a continuous variable (e.g. dosage of a medication), and we address the two main challenges of this setup. We propose the adversarial counterfactual regression network (ACFR) that adversarially minimizes the representation imbalance in terms of KL divergence, and also maintains the impact of the treatment value on the outcome prediction by leveraging an attention mechanism. 
Theoretically we demonstrate that ACFR objective function is grounded in an upper bound on counterfactual outcome prediction error. 
Our experimental evaluation on semi-synthetic datasets demonstrates the empirical superiority of ACFR over a range of state-of-the-art methods.

----

## [1459] Shaping Up SHAP: Enhancing Stability through Layer-Wise Neighbor Selection

**Authors**: *Gwladys Kelodjou, Laurence Rozé, Véronique Masson, Luis Galárraga, Romaric Gaudel, Maurice Tchuenté, Alexandre Termier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29208](https://doi.org/10.1609/aaai.v38i12.29208)

**Abstract**:

Machine learning techniques, such as deep learning and ensemble methods, are widely used in various domains due to their ability to handle complex real-world tasks. However, their black-box nature has raised multiple concerns about the fairness, trustworthiness, and transparency of computer-assisted decision-making. This has led to the emergence of local post-hoc explainability methods, which offer explanations for individual decisions made by black-box algorithms. Among these methods, Kernel SHAP is widely used due to its model-agnostic nature and its well-founded theoretical framework. Despite these strengths, Kernel SHAP suffers from high instability: different executions of the method with the same inputs can lead to significantly different explanations, which diminishes the relevance of the explanations. The contribution of this paper is two-fold. On the one hand, we show that Kernel SHAP's instability is caused by its stochastic neighbor selection procedure, which we adapt to achieve full stability without compromising explanation fidelity. On the other hand, we show that by restricting the neighbors generation to perturbations of size 1 -- which we call the coalitions of Layer 1 -- we obtain a novel feature-attribution method that is fully stable, computationally efficient, and still meaningful.

----

## [1460] IOFM: Using the Interpolation Technique on the Over-Fitted Models to Identify Clean-Annotated Samples

**Authors**: *Dongha Kim, Yongchan Choi, Kunwoong Kim, Ilsang Ohn, Yongdai Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29209](https://doi.org/10.1609/aaai.v38i12.29209)

**Abstract**:

Most recent state-of-the-art algorithms for handling noisy label problems are  based on the memorization effect, which is a phenomenon that deep neural networks (DNNs) memorize clean data before noisy ones. While the memorization effect can be a powerful tool, there are several cases where memorization effect does not occur. Examples are imbalanced class distributions and heavy contamination on labels. To address this limitation, we introduce a whole new approach called the interpolation with the over-fitted model (IOFM), which leverages over-fitted deep neural networks. The IOFM utilizes a new finding of over-fitted DNNs: for a given training sample, its neighborhoods chosen from the feature space are distributed differently on the original input space depending on the cleanness of the target sample. The IOFM has notable features in two aspects: 1) it yields superior results even when the training data are imbalanced or heavily noisy, 2) since we utilize over-fitted deep neural networks, a fine-tuning procedure to select the optimal training epoch, which is an essential yet sensitive factor for the success of the memorization effect, is not required, and thus, the IOFM can be used for non-experts. Through extensive experiments, we show that our method can serve as a promising alternative to existing solutions dealing with noisy labels, offering improved performance even in challenging situations.

----

## [1461] When Model Meets New Normals: Test-Time Adaptation for Unsupervised Time-Series Anomaly Detection

**Authors**: *Dongmin Kim, Sunghyun Park, Jaegul Choo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29210](https://doi.org/10.1609/aaai.v38i12.29210)

**Abstract**:

Time-series anomaly detection deals with the problem of detecting anomalous timesteps by learning normality from the sequence of observations. However, the concept of normality evolves over time, leading to a "new normal problem", where the distribution of normality can be changed due to the distribution shifts between training and test data. This paper highlights the prevalence of the new normal problem in unsupervised time-series anomaly detection studies. To tackle this issue, we propose a simple yet effective test-time adaptation strategy based on trend estimation and a self-supervised approach to learning new normalities during inference. Extensive experiments on real-world benchmarks demonstrate that incorporating the proposed strategy into the anomaly detector consistently improves the model's performances compared to the existing baselines, leading to robustness to the distribution shifts.

----

## [1462] Adaptive Shortcut Debiasing for Online Continual Learning

**Authors**: *Doyoung Kim, Dongmin Park, Yooju Shin, Jihwan Bang, Hwanjun Song, Jae-Gil Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29211](https://doi.org/10.1609/aaai.v38i12.29211)

**Abstract**:

We propose a novel framework DropTop that suppresses the shortcut bias in online continual learning (OCL) while being adaptive to the varying degree of the shortcut bias incurred by continuously changing environment. By the observed high-attention property of the shortcut bias, highly-activated features are considered candidates for debiasing. More importantly, resolving the limitation of the online environment where prior knowledge and auxiliary data are not ready, two novel techniques---feature map fusion and adaptive intensity shifting---enable us to automatically determine the appropriate level and proportion of the candidate shortcut features to be dropped. Extensive experiments on five benchmark datasets demonstrate that, when combined with various OCL algorithms, DropTop increases the average accuracy by up to 10.4% and decreases the forgetting by up to 63.2%.

----

## [1463] MetaMix: Meta-State Precision Searcher for Mixed-Precision Activation Quantization

**Authors**: *Han-Byul Kim, Joo Hyung Lee, Sungjoo Yoo, Hong-Seok Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29212](https://doi.org/10.1609/aaai.v38i12.29212)

**Abstract**:

Mixed-precision quantization of efficient networks often suffer from activation instability encountered in the exploration of bit selections. To address this problem, we propose a novel method called MetaMix which consists of bit selection and weight training phases. The bit selection phase iterates two steps, (1) the mixed-precision-aware weight update, and (2) the bit-search training with the fixed mixed-precision-aware weights, both of which combined reduce activation instability in mixed-precision quantization and contribute to fast and high-quality bit selection. The weight training phase exploits the weights and step sizes trained in the bit selection phase and fine-tunes them thereby offering fast training. Our experiments with efficient and hard-to-quantize networks, i.e., MobileNet v2 and v3, and ResNet-18 on ImageNet show that our proposed method pushes the boundary of mixed-precision quantization, in terms of accuracy vs. operations, by outperforming both mixed- and single-precision SOTA methods.

----

## [1464] Curved Representation Space of Vision Transformers

**Authors**: *Juyeop Kim, Junha Park, Songkuk Kim, Jong-Seok Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29213](https://doi.org/10.1609/aaai.v38i12.29213)

**Abstract**:

Neural networks with self-attention (a.k.a. Transformers) like ViT and Swin have emerged as a better alternative to traditional convolutional neural networks (CNNs). However, our understanding of how the new architecture works is still limited. In this paper, we focus on the phenomenon that Transformers show higher robustness against corruptions than CNNs, while not being overconfident. This is contrary to the intuition that robustness increases with confidence. We resolve this contradiction by empirically investigating how the output of the penultimate layer moves in the representation space as the input data moves linearly within a small area. In particular, we show the following. (1) While CNNs exhibit fairly linear relationship between the input and output movements, Transformers show nonlinear relationship for some data. For those data, the output of Transformers moves in a curved trajectory as the input moves linearly. (2) When a data is located in a curved region, it is hard to move it out of the decision region since the output moves along a curved trajectory instead of a straight line to the decision boundary, resulting in high robustness of Transformers. (3) If a data is slightly modified to jump out of the curved region, the movements afterwards become linear and the output goes to the decision boundary directly. In other words, there does exist a decision boundary near the data, which is hard to find only because of the curved representation space. This explains the underconfident prediction of Transformers. Also, we examine mathematical properties of the attention operation that induce nonlinear response to linear perturbation. Finally, we share our additional findings, regarding what contributes to the curved representation space of Transformers, and how the curvedness evolves during training.

----

## [1465] Robust Distributed Gradient Aggregation Using Projections onto Gradient Manifolds

**Authors**: *Kwang In Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29214](https://doi.org/10.1609/aaai.v38i12.29214)

**Abstract**:

We study the distributed gradient aggregation problem where individual clients contribute to learning a central model by sharing parameter gradients constructed from local losses. However, errors in some gradients, caused by low-quality data or adversaries, can degrade the learning process when naively combined. Existing robust gradient aggregation approaches assume that local data represent the global data-generating distribution, which may not always apply to heterogeneous (non-i.i.d.) client data. We propose a new algorithm that can robustly aggregate gradients from potentially heterogeneous clients. Our approach leverages the manifold structure inherent in heterogeneous client gradients and evaluates gradient anomaly degrees by projecting them onto this manifold. This algorithm is implemented as a simple and efficient method that accumulates random projections within the subspace defined by the nearest neighbors within a gradient cloud. Our experiments demonstrate consistent performance improvements over state-of-the-art robust aggregation algorithms.

----

## [1466] Stitching Sub-trajectories with Conditional Diffusion Model for Goal-Conditioned Offline RL

**Authors**: *Sungyoon Kim, Yunseon Choi, Daiki E. Matsunaga, Kee-Eung Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29215](https://doi.org/10.1609/aaai.v38i12.29215)

**Abstract**:

Offline Goal-Conditioned Reinforcement Learning (Offline GCRL) is an important problem in RL that focuses on acquiring diverse goal-oriented skills solely from pre-collected behavior datasets. 
In this setting, the reward feedback is typically absent except when the goal is achieved, 
which makes it difficult to learn policies especially from a finite dataset of suboptimal behaviors.
In addition, realistic scenarios involve long-horizon planning, which necessitates the 
extraction of useful skills within sub-trajectories.
Recently, the conditional diffusion model has been shown to be a promising approach to 
generate high-quality long-horizon plans for RL. 
However, their
practicality for the goal-conditioned setting is still limited due to a number of technical assumptions
made by the methods.
In this paper, we propose SSD (Sub-trajectory Stitching with Diffusion), 
a model-based offline GCRL method that leverages the conditional diffusion model to address these limitations. 
In summary, we use the diffusion model that generates future plans conditioned on the target goal and value, 
with the target value estimated from the goal-relabeled offline dataset.
We report state-of-the-art performance in the standard benchmark set of GCRL tasks, and 
demonstrate the capability to successfully stitch the segments of suboptimal trajectories in the offline data to generate high-quality plans.

----

## [1467] Cross-Class Feature Augmentation for Class Incremental Learning

**Authors**: *Taehoon Kim, Jaeyoo Park, Bohyung Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29216](https://doi.org/10.1609/aaai.v38i12.29216)

**Abstract**:

We propose a novel class incremental learning approach, which incorporates a feature augmentation technique motivated by adversarial attacks. We employ a classifier learned in the past to complement training examples of previous tasks. The proposed approach has an unique perspective to utilize the previous knowledge in class incremental learning since it augments features of arbitrary target classes using examples in other classes via adversarial attacks on a previously learned classifier. By allowing the Cross-Class Feature Augmentations (CCFA), each class in the old tasks conveniently populates samples in the feature space, which alleviates the collapse of the decision boundaries caused by sample deficiency for the previous tasks, especially when the number of stored exemplars is small. This idea can be easily incorporated into existing class incremental learning algorithms without any architecture modification. Extensive experiments on the standard benchmarks show that our method consistently outperforms existing class incremental learning methods by significant margins in various scenarios, especially under an environment with an extremely limited memory budget.

----

## [1468] Robust Policy Learning via Offline Skill Diffusion

**Authors**: *Woo Kyung Kim, Minjong Yoo, Honguk Woo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29217](https://doi.org/10.1609/aaai.v38i12.29217)

**Abstract**:

Skill-based reinforcement learning (RL) approaches have shown considerable promise, especially in solving long-horizon tasks via hierarchical structures. These skills, learned task-agnostically from offline datasets, can accelerate the policy learning process for new tasks. Yet, the application of these skills in different domains remains restricted due to their inherent dependency on the datasets, which poses a challenge when attempting to learn a skill-based policy via RL for a target domain different from the datasets' domains. In this paper, we present a novel offline skill learning framework DuSkill which employs a guided Diffusion model to generate versatile skills extended from the limited skills in datasets, thereby enhancing the robustness of policy learning for tasks in different domains. Specifically, we devise a guided diffusion-based skill decoder in conjunction with the hierarchical encoding to disentangle the skill embedding space into two distinct representations, one for encapsulating domain-invariant behaviors and the other for delineating the factors that induce domain variations in the behaviors. Our DuSkill framework enhances the diversity of skills learned offline, thus enabling to accelerate the learning procedure of high-level policies for different domains.
Through experiments, we show that DuSkill outperforms other skill-based imitation learning and RL algorithms for several long-horizon tasks, demonstrating its benefits in few-shot imitation and online RL.

----

## [1469] Relaxed Stationary Distribution Correction Estimation for Improved Offline Policy Optimization

**Authors**: *Woosung Kim, Donghyeon Ki, Byung-Jun Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29218](https://doi.org/10.1609/aaai.v38i12.29218)

**Abstract**:

One of the major challenges of offline reinforcement learning (RL) is dealing with distribution shifts that stem from the mismatch between the trained policy and the data collection policy. Stationary distribution correction estimation algorithms (DICE) have addressed this issue by regularizing the policy optimization with f-divergence between the state-action visitation distributions of the data collection policy and the optimized policy. While such regularization naturally integrates to derive an objective to get optimal state-action visitation, such an implicit policy optimization framework has shown limited performance in practice. We observe that the reduced performance is attributed to the biased estimate and the properties of conjugate functions of f-divergence regularization. In this paper, we improve the regularized implicit policy optimization framework by relieving the bias and reshaping the conjugate function by relaxing the constraints. We show that the relaxation adjusts the degree of involvement of the sub-optimal samples in optimization, and we derive a new offline RL algorithm that benefits from the relaxed framework, improving from a previous implicit policy optimization algorithm by a large margin.

----

## [1470] Structure-Aware Multimodal Sequential Learning for Visual Dialog

**Authors**: *Young-Jin Kim, Min-Jun Kim, Kyunghwan An, Jinwoo Ahn, Jaeseok Kim, Yu-Jung Heo, Du-Seong Chang, Eun-Sol Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29219](https://doi.org/10.1609/aaai.v38i12.29219)

**Abstract**:

With the ability to collect vast amounts of image and natural language data from the web, there has been a remarkable advancement in Large-scale Language Models (LLMs). This progress has led to the emergence of chatbots and dialogue systems capable of fluent conversations with humans. As the variety of devices enabling interactions between humans and agents expands, and the performance of text-based dialogue systems improves, there has been recently proposed research on visual dialog. However, visual dialog requires understanding sequences of pairs consisting of images and sentences, making it challenging to gather sufficient data for training large-scale models from the web. In this paper, we propose a new multimodal learning method leveraging existing large-scale models designed for each modality, to enable model training for visual dialog with small visual dialog datasets. The key ideas of our approach are: 1) storing the history or context during the progression of visual dialog in the form of spatiotemporal graphs, and 2) introducing small modulation blocks between modality-specific models and the graphs to align the semantic spaces. For implementation, we introduce a novel structure-aware cross-attention method, which retrieves relevant image and text knowledge for utterance generation from the pretrained models. For experiments, we achieved a new state-of-the-art performance on three visual dialog datasets, including the most challenging one COMET.

----

## [1471] A Class of Topological Pseudodistances for Fast Comparison of Persistence Diagrams

**Authors**: *Rolando Kindelan Nuñez, Mircea Petrache, Mauricio Cerda, Nancy Hitschfeld*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29220](https://doi.org/10.1609/aaai.v38i12.29220)

**Abstract**:

Persistence diagrams (PD)s play a central role in topological data analysis, and are used in an ever increasing variety of applications. The comparison of PD data requires computing distances among large sets of PDs, with metrics which are accurate, theoretically sound, and fast to compute. Especially for denser multi-dimensional PDs, such comparison metrics are lacking. While on the one hand, Wasserstein-type distances have high accuracy and theoretical guarantees, they incur high computational cost. On the other hand, distances between vectorizations such as Persistence Statistics (PS)s have lower computational cost, but lack the accuracy guarantees and theoretical properties of a true distance over PD space. In this work we introduce a class of pseudodistances called Extended Topological Pseudodistances (ETD)s, which have tunable complexity, and can approximate Sliced and classical Wasserstein distances at the high-complexity extreme, while being computationally lighter and close to Persistence Statistics at the lower complexity extreme, and thus allow users to interpolate between the two metrics. We build theoretical comparisons to show how to fit our new distances at an intermediate level between persistence vectorizations and Wasserstein distances. We also experimentally verify that ETDs outperform PSs in terms of accuracy and outperform Wasserstein and Sliced Wasserstein distances in terms of computational complexity.

----

## [1472] SALSA: Semantically-Aware Latent Space Autoencoder

**Authors**: *Kathryn E. Kirchoff, Travis Maxfield, Alexander Tropsha, Shawn M. Gomez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29221](https://doi.org/10.1609/aaai.v38i12.29221)

**Abstract**:

In deep learning for drug discovery, molecular representations are often based on sequences, known as SMILES, which allow for straightforward implementation of natural language processing methodologies, one being the sequence-to-sequence autoencoder. However, we observe that training an autoencoder solely on SMILES is insufficient to learn molecular representations that are semantically meaningful, where semantics are specified by the structural (graph-to-graph) similarities between molecules. We demonstrate by example that SMILES-based autoencoders may map structurally similar molecules to distant codes, resulting in an incoherent latent space that does not necessarily respect the semantic similarities between molecules. To address this shortcoming we propose Semantically-Aware Latent Space Autoencoder (SALSA) for molecular representations: a SMILES-based transformer autoencoder modified with a contrastive task aimed at learning graph-to-graph similarities between molecules. To accomplish this, we develop a novel dataset comprised of sets of structurally similar molecules and opt for a supervised contrastive loss that is able to incorporate full sets of positive samples. We evaluate semantic awareness of SALSA representations by comparing to its ablated counterparts, and show empirically that SALSA learns representations that maintain 1) structural awareness, 2) physicochemical awareness, 3) biological awareness, and 4) semantic continuity.

----

## [1473] Principle Component Trees and Their Persistent Homology

**Authors**: *Ben A. Kizaric, Daniel L. Pimentel-Alarcón*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29222](https://doi.org/10.1609/aaai.v38i12.29222)

**Abstract**:

Low dimensional models like PCA are often used to simplify complex datasets by learning a single approximating subspace. This paradigm has expanded to union of subspaces models, like those learned by subspace clustering. In this paper, we present Principal Component Trees (PCTs), a graph structure that generalizes these ideas to identify mixtures of components that together describe the subspace structure of high-dimensional datasets. Each node in a PCT corresponds to a principal component of the data, and the edges between nodes indicate the components that must be mixed to produce a subspace that approximates a portion of the data. In order to construct PCTs, we propose two angle-distribution hypothesis tests to detect subspace clusters in the data. To analyze, compare, and select the best PCT model, we define two persistent homology measures that describe their shape. We show our construction yields two key properties of PCTs, namely ancestral orthogonality and non-decreasing singular values. Our main theoretical results show that learning PCTs reduces to PCA under multivariate normality, and that PCTs are efficient parameterizations of intersecting union of subspaces. Finally, we use PCTs to analyze neural network latent space, word embeddings, and reference image datasets.

----

## [1474] Pantypes: Diverse Representatives for Self-Explainable Models

**Authors**: *Rune D. Kjærsgaard, Ahcène Boubekki, Line H. Clemmensen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29223](https://doi.org/10.1609/aaai.v38i12.29223)

**Abstract**:

Prototypical self-explainable classifiers have emerged to meet the growing demand for interpretable AI systems. These classifiers are designed to incorporate high transparency in their decisions by basing inference on similarity with learned prototypical objects. While these models are designed with diversity in mind, the learned prototypes often do not sufficiently represent all aspects of the input distribution, particularly those in low density regions. 
Such lack of sufficient data representation, known as representation bias, has been associated with various detrimental properties related to machine learning diversity and fairness. In light of this, we introduce pantypes, a new family of prototypical objects designed to capture the full diversity of the input distribution through a sparse set of objects. We show that pantypes can empower prototypical self-explainable models by occupying divergent regions of the latent space and thus fostering high diversity, interpretability and fairness.

----

## [1475] Shuffled Deep Regression

**Authors**: *Masahiro Kohjima*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29224](https://doi.org/10.1609/aaai.v38i12.29224)

**Abstract**:

Shuffled regression is the problem of learning regression models from shuffled data that consists of a set of input features and a set of target outputs where the correspondence between the input and output is unknown. This study proposes a new deep learning method for shuffled regression called Shuffled Deep Regression (SDR). We derive the sparse and stochastic variant of the Expectation-Maximization algorithm for SDR that iteratively updates discrete latent variables and the parameters of neural networks. The effectiveness of the proposal is confirmed by benchmark data experiments.

----

## [1476] Approximating the Shapley Value without Marginal Contributions

**Authors**: *Patrick Kolpaczki, Viktor Bengs, Maximilian Muschalik, Eyke Hüllermeier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29225](https://doi.org/10.1609/aaai.v38i12.29225)

**Abstract**:

The Shapley value, which is arguably the most popular approach for assigning a meaningful contribution value to players in a cooperative game, has recently been used intensively in explainable artificial intelligence. Its meaningfulness is due to axiomatic properties that only the Shapley value satisfies, which, however, comes at the expense of an exact computation growing exponentially with the number of agents. Accordingly, a number of works are devoted to the efficient approximation of the Shapley value, most of them revolve around the notion of an agent's marginal contribution. In this paper, we propose with SVARM and Stratified SVARM two parameter-free and domain-independent approximation algorithms based on a representation of the Shapley value detached from the notion of marginal contribution. We prove unmatched theoretical guarantees regarding their approximation quality and provide empirical results including synthetic games as well as common explainability use cases comparing ourselves with state-of-the-art methods.

----

## [1477] Improved Bandits in Many-to-One Matching Markets with Incentive Compatibility

**Authors**: *Fang Kong, Shuai Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29226](https://doi.org/10.1609/aaai.v38i12.29226)

**Abstract**:

Two-sided matching markets have been widely studied in the literature due to their rich applications. Since participants are usually uncertain about their preferences, online algorithms have recently been adopted to learn them through iterative interactions. An existing work initiates the study of this problem in a many-to-one setting with responsiveness. However, their results are far from optimal and lack guarantees of incentive compatibility. We first extend an existing algorithm for the one-to-one setting to this more general setting and show it achieves a near-optimal bound for player-optimal regret. Nevertheless, due to the substantial requirement for collaboration, a single player's deviation could lead to a huge increase in its own cumulative rewards and a linear regret for others. In this paper, we aim to enhance the regret bound in many-to-one markets while ensuring incentive compatibility. We first propose the adaptively explore-then-deferred-acceptance (AETDA) algorithm for responsiveness setting and derive an upper bound for player-optimal stable regret while demonstrating its guarantee of incentive compatibility. This result is a significant improvement over existing works. And to the best of our knowledge, it constitutes the first player-optimal guarantee in matching markets that offers such robust assurances. We also consider broader substitutable preferences, one of the most general conditions to ensure the existence of a stable matching and cover responsiveness. We devise an online DA (ODA) algorithm and establish an upper bound for the player-pessimal stable regret for this setting.

----

## [1478] Unknown-Aware Graph Regularization for Robust Semi-supervised Learning from Uncurated Data

**Authors**: *Heejo Kong, Suneung Kim, Ho-Joong Kim, Seong-Whan Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29227](https://doi.org/10.1609/aaai.v38i12.29227)

**Abstract**:

Recent advances in semi-supervised learning (SSL) have relied on the optimistic assumption that labeled and unlabeled data share the same class distribution. However, this assumption is often violated in real-world scenarios, where unlabeled data may contain out-of-class samples. SSL with such uncurated unlabeled data leads training models to be corrupted. In this paper, we propose a robust SSL method for learning from uncurated real-world data within the context of open-set semi-supervised learning (OSSL). Unlike previous works that rely on feature similarity distance, our method exploits uncertainty in logits. By leveraging task-dependent predictions of logits, our method is capable of robust learning even in the presence of highly correlated outliers. Our key contribution is to present an unknown-aware graph regularization (UAG), a novel technique that enhances the performance of uncertainty-based OSSL frameworks. The technique addresses not only the conflict between training objectives for inliers and outliers but also the limitation of applying the same training rule for all outlier classes, which are existed on previous uncertainty-based approaches. Extensive experiments demonstrate that UAG surpasses state-of-the-art OSSL methods by a large margin across various protocols. Codes are available at https://github.com/heejokong/UAGreg.

----

## [1479] Ghost Noise for Regularizing Deep Neural Networks

**Authors**: *Atli Kosson, Dongyang Fan, Martin Jaggi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29228](https://doi.org/10.1609/aaai.v38i12.29228)

**Abstract**:

Batch Normalization (BN) is widely used to stabilize the optimization process and improve the test performance of deep neural networks. The regularization effect of BN depends on the batch size and explicitly using smaller batch sizes with Batch Normalization, a method known as Ghost Batch Normalization (GBN), has been found to improve generalization in many settings. We investigate the effectiveness of GBN by disentangling the induced ``Ghost Noise'' from normalization and quantitatively analyzing the distribution of noise as well as its impact on model performance. Inspired by our analysis, we propose a new regularization technique called Ghost Noise Injection (GNI) that imitates the noise in GBN without incurring the detrimental train-test discrepancy effects of small batch training. We experimentally show that GNI can provide a greater generalization benefit than GBN. Ghost Noise Injection can also be beneficial in otherwise non-noisy settings such as layer-normalized networks, providing additional evidence of the usefulness of Ghost Noise in Batch Normalization as a regularizer.

----

## [1480] Zero-Shot Task Adaptation with Relevant Feature Information

**Authors**: *Atsutoshi Kumagai, Tomoharu Iwata, Yasuhiro Fujiwara*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29229](https://doi.org/10.1609/aaai.v38i12.29229)

**Abstract**:

We propose a method to learn prediction models such as classifiers for unseen target tasks where labeled and unlabeled data are absent but a few relevant input features for solving the tasks are given. Although machine learning requires data for training, data are often difficult to collect in practice. On the other hand, for many applications, a few relevant features would be more easily obtained. Although zero-shot learning or zero-shot domain adaptation use external knowledge to adapt to unseen classes or tasks without data, relevant features have not been used in existing studies. The proposed method improves the generalization performance on the target tasks, where there are no data but a few relevant features are given, by meta-learning from labeled data in related tasks. In the meta-learning phase, it is essential to simulate test phases on target tasks where prediction model learning is required without data. To this end, our neural network-based prediction model is meta-learned such that it correctly responds to perturbations of the relevant features on randomly generated synthetic data. By this modeling, the prediction model can explicitly learn the discriminability of the relevant features without real target data. When unlabeled training data are available in the target tasks, the proposed method can incorporate such data to boost the performance in a unified framework.  Our experiments demonstrate that the proposed method outperforms various existing methods with four real-world datasets.

----

## [1481] Friendly Attacks to Improve Channel Coding Reliability

**Authors**: *Anastasia Kurmukova, Deniz Gündüz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29230](https://doi.org/10.1609/aaai.v38i12.29230)

**Abstract**:

This paper introduces a novel approach called "friendly attack" aimed at enhancing the performance of error correction channel codes. Inspired by the concept of adversarial attacks, our method leverages the idea of introducing slight perturbations to the neural network input, resulting in a substantial impact on the network's performance. By introducing small perturbations to fixed-point modulated codewords before transmission, we effectively improve the decoder's performance without violating the input power constraint. The perturbation design is accomplished by a modified iterative fast gradient method. This study investigates various decoder architectures suitable for computing gradients to obtain the desired perturbations. Specifically, we consider belief propagation (BP) for LDPC codes; the error correcting code transformer, BP and neural BP (NBP) for polar codes, and neural BCJR for convolutional codes. We demonstrate that the proposed friendly attack method can improve the reliability across different channels, modulations, codes, and decoders. This method allows us to increase the reliability of communication with a legacy receiver by simply modifying the transmitted codeword appropriately.

----

## [1482] Evolving Parameterized Prompt Memory for Continual Learning

**Authors**: *Muhammad Rifki Kurniawan, Xiang Song, Zhiheng Ma, Yuhang He, Yihong Gong, Yang Qi, Xing Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29231](https://doi.org/10.1609/aaai.v38i12.29231)

**Abstract**:

Recent studies have demonstrated the potency of leveraging prompts in Transformers for continual learning (CL). Nevertheless, employing a discrete key-prompt bottleneck can lead to selection mismatches and inappropriate prompt associations during testing. Furthermore, this approach hinders adaptive prompting due to the lack of shareability among nearly identical instances at more granular level. To address these challenges, we introduce the Evolving Parameterized Prompt Memory (EvoPrompt), a novel method involving adaptive and continuous prompting attached to pre-trained Vision Transformer (ViT), conditioned on specific instance. We formulate a continuous prompt function as a neural bottleneck and encode the collection of prompts on network weights. We establish a paired prompt memory system consisting of a stable reference and a flexible working prompt memory. Inspired by linear mode connectivity, we progressively fuse the working prompt memory and reference prompt memory during inter-task periods, resulting in continually evolved prompt memory. This fusion involves aligning functionally equivalent prompts using optimal transport and aggregating them in parameter space with an adjustable bias based on prompt node attribution. Additionally, to enhance backward compatibility, we propose compositional classifier initialization, which leverages prior prototypes from pre-trained models to guide the initialization of new classifiers in a subspace-aware manner. Comprehensive experiments validate that our approach achieves state-of-the-art performance in both class and domain incremental learning scenarios.

----

## [1483] AesFA: An Aesthetic Feature-Aware Arbitrary Neural Style Transfer

**Authors**: *Joonwoo Kwon, Sooyoung Kim, Yuewei Lin, Shinjae Yoo, Jiook Cha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29232](https://doi.org/10.1609/aaai.v38i12.29232)

**Abstract**:

Neural style transfer (NST) has evolved significantly in recent years. Yet, despite its rapid progress and advancement, existing NST methods either struggle to transfer aesthetic information from a style effectively or suffer from high computational costs and inefficiencies in feature disentanglement due to using pre-trained models. This work proposes a lightweight but effective model, AesFA---Aesthetic Feature-Aware NST. The primary idea is to decompose the image via its frequencies to better disentangle aesthetic styles from the reference image while training the entire model in an end-to-end manner to exclude pre-trained models at inference completely. To improve the network's ability to extract more distinct representations and further enhance the stylization quality, this work introduces a new aesthetic feature: contrastive loss. Extensive experiments and ablations show the approach not only outperforms recent NST methods in terms of stylization quality, but it also achieves faster inference. Codes are available at https://github.com/Sooyyoungg/AesFA.

----

## [1484] HDformer: A Higher-Dimensional Transformer for Detecting Diabetes Utilizing Long-Range Vascular Signals

**Authors**: *Ella Lan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29233](https://doi.org/10.1609/aaai.v38i12.29233)

**Abstract**:

Diabetes mellitus is a global concern, and early detection can prevent serious complications. 50% of those with diabetes live undiagnosed, disproportionately afflicting low-income groups. Non-invasive methods have emerged for timely detection; however, their limited accuracy constrains clinical usage. In this research, we present a novel Higher Dimensional Transformer (HDformer), the first Transformer-based architecture which utilizes long-range photoplethysmography (PPG) to detect diabetes. The long-range PPG maximizes signal contextual information when compared to the less-than 30 second signals commonly used in existing research. To increase the computational efficiency of HDformer’s long-range processing, a new attention module, Time Square Attention (TSA), is invented to achieve linear computational complexity with respect to the token volume while retaining the local/global dependencies. TSA converts the 1D inputs into 2D representations, grouping the adjacent points into a single 2D token. It then generates dynamic patches and feeds them into a gated mixture-of-experts (MoE) network, optimizing the learning on different attention areas. HDformer achieves state-of-the-art results (sensitivity 98.4, accuracy 97.3, specificity 92.8, AUC 0.929) on the standard MIMIC-III dataset, surpassing existing research. Furthermore, we develop an end-to-end solution where a low-cost wearable is prototyped to connect with the HDformer in the Cloud via a mobile app. This scalable, convenient, and affordable approach provides instantaneous detection and continuous monitoring for individuals. It aids doctors in easily screening for diabetes and safeguards underprivileged communities. The enhanced versatility of HDformer allows for efficient processing and learning of long-range signals in general one-dimensional time-series sequences, particularly for all biomedical waveforms.

----

## [1485] Generative Model Perception Rectification Algorithm for Trade-Off between Diversity and Quality

**Authors**: *Guipeng Lan, Shuai Xiao, Jiachen Yang, Jiabao Wen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29234](https://doi.org/10.1609/aaai.v38i12.29234)

**Abstract**:

How to balance the diversity and quality of results from generative models through perception rectification poses a significant challenge. Abnormal perception in generative models is typically caused by two factors: inadequate model structure and imbalanced data distribution. In response to this issue, we propose the dynamic model perception rectification algorithm (DMPRA) for generalized generative models. The core idea is to gain a comprehensive perception of the data in the generative model by appropriately highlighting the low-density samples in the perception space, also known as the minor group samples. The entire process can be summarized as "search-evaluation-adjustment". To identify low-density regions in the data manifold within the perception space of generative models, we introduce a filtering method based on extended neighborhood sampling. Based on the informational value of samples from low-density regions, our proposed mechanism generates informative weights to assess the significance of these samples in correcting the models' perception. By using dynamic adjustment, DMPRA ensures simultaneous enhancement of diversity and quality in the presence of imbalanced data distribution. Experimental results indicate that the algorithm has effectively improved Generative Adversarial Nets (GANs), Normalizing Flows (Flows), Variational Auto-Encoders (VAEs), and Diffusion Models (Diffusion).

----

## [1486] CoLAL: Co-learning Active Learning for Text Classification

**Authors**: *Linh Le, Genghong Zhao, Xia Zhang, Guido Zuccon, Gianluca Demartini*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29235](https://doi.org/10.1609/aaai.v38i12.29235)

**Abstract**:

In the machine learning field, the challenge of effectively learning with limited data has become increasingly crucial. Active Learning (AL) algorithms play a significant role in this by enhancing model performance. We introduce a novel AL algorithm, termed Co-learning (CoLAL), designed to select the most diverse and representative samples within a training dataset.  This approach utilizes noisy labels and predictions made by the primary model on unlabeled data. By leveraging a probabilistic graphical model, we combine two multi-class classifiers into a binary one. This classifier determines if both the main and the peer models agree on a prediction. If they do, the unlabeled sample is assumed to be easy to classify and is thus not beneficial to increase the target model's performance. We prioritize data that represents the unlabeled set without overlapping  decision boundaries. The discrepancies between these boundaries can be estimated by the probability that two models result in the same prediction. Through theoretical analysis and experimental validation, we reveal that the integration of noisy labels into the peer model effectively identifies target model's potential inaccuracies. We evaluated the CoLAL method across seven benchmark datasets: four text datasets (AGNews, DBPedia, PubMed, SST-2) and text-based state-of-the-art (SOTA) baselines, and three image datasets (CIFAR100, MNIST, OpenML-155) and computer vision SOTA baselines. The results show that our CoLAL method significantly outperforms existing SOTA in text-based AL, and is competitive with SOTA image-based AL techniques.

----

## [1487] Doubly Perturbed Task Free Continual Learning

**Authors**: *Byung Hyun Lee, Min-hwan Oh, Se Young Chun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29236](https://doi.org/10.1609/aaai.v38i12.29236)

**Abstract**:

Task-free online continual learning (TF-CL) is a challenging problem where the model incrementally learns tasks without explicit task information. Although training with entire data from the past, present as well as future is considered as the gold standard, naive approaches in TF-CL with the current samples may be conflicted with learning with samples in the future, leading to catastrophic forgetting and poor plasticity. Thus, a proactive consideration of an unseen future sample in TF-CL becomes imperative. Motivated by this intuition, we propose a novel TF-CL framework considering future samples and show that injecting adversarial perturbations on both input data and decision-making is effective. Then, we propose a novel method named Doubly Perturbed Continual Learning (DPCL) to efficiently implement these input and decision-making perturbations. Specifically, for input perturbation, we propose an approximate perturbation method that injects noise into the input data as well as the feature vector and then interpolates the two perturbed samples. For decision-making process perturbation, we devise multiple stochastic classifiers. We also investigate a memory management scheme and learning rate scheduling reflecting our proposed double perturbations. We demonstrate that our proposed method outperforms the state-of-the-art baseline methods by large margins on various TF-CL benchmarks.

----

## [1488] OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models

**Authors**: *Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, Eunhyeok Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29237](https://doi.org/10.1609/aaai.v38i12.29237)

**Abstract**:

Large language models (LLMs) with hundreds of billions of parameters require powerful server-grade GPUs for inference, limiting their practical deployment. To address this challenge, we introduce the outlier-aware weight quantization (OWQ) method, which aims to minimize LLM's footprint through low-precision representation. OWQ prioritizes a small subset of structured weights sensitive to quantization, storing them in high-precision, while applying highly tuned quantization to the remaining dense weights. This sensitivity-aware mixed-precision scheme reduces the quantization error notably, and extensive experiments demonstrate that 3.1-bit models using OWQ perform comparably to 4-bit models optimized by OPTQ. Furthermore, OWQ incorporates a parameter-efficient fine-tuning for task-specific adaptation, called weak column tuning (WCT), enabling accurate task-specific LLM adaptation with minimal memory overhead in the optimized format. OWQ represents a notable advancement in the flexibility, efficiency, and practicality of LLM optimization literature. The source code is available at https://github.com/xvyaward/owq.

----

## [1489] DiSCO: Diffusion Schrödinger Bridge for Molecular Conformer Optimization

**Authors**: *Danyeong Lee, Dohoon Lee, Dongmin Bang, Sun Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29238](https://doi.org/10.1609/aaai.v38i12.29238)

**Abstract**:

The generation of energetically optimal 3D molecular conformers is crucial in cheminformatics and drug discovery. While deep generative models have been utilized for direct generation in Euclidean space, this approach encounters challenges, including the complexity of navigating a vast search space. Recent generative models that implement simplifications to circumvent these challenges have achieved state-of-the-art results, but this simplified approach unavoidably creates a gap between the generated conformers and the ground-truth conformational landscape. To bridge this gap, we introduce DiSCO: Diffusion Schrödinger Bridge for Molecular Conformer Optimization, a novel diffusion framework that enables direct learning of nonlinear diffusion processes in prior-constrained Euclidean space for the optimization of 3D molecular conformers. Through the incorporation of an SE(3)-equivariant Schrödinger bridge, we establish the roto-translational equivariance of the generated conformers. Our framework is model-agnostic and offers an easily implementable solution for the post hoc optimization of conformers produced by any generation method. Through comprehensive evaluations and analyses, we establish the strengths of our framework, substantiating the application of the Schrödinger bridge for molecular conformer optimization. First, our approach consistently outperforms four baseline approaches, producing conformers with higher diversity and improved quality. Then, we show that the intermediate conformers generated during our diffusion process exhibit valid and chemically meaningful characteristics. We also demonstrate the robustness of our method when starting from conformers of diverse quality, including those unseen during training. Lastly, we show that the precise generation of low-energy conformers via our framework helps in enhancing the downstream prediction of molecular properties. The code is available at https://github.com/Danyeong-Lee/DiSCO.

----

## [1490] Spear and Shield: Adversarial Attacks and Defense Methods for Model-Based Link Prediction on Continuous-Time Dynamic Graphs

**Authors**: *Dongjin Lee, Juho Lee, Kijung Shin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29239](https://doi.org/10.1609/aaai.v38i12.29239)

**Abstract**:

Real-world graphs are dynamic, constantly evolving with new interactions, such as financial transactions in financial networks. 
Temporal Graph Neural Networks (TGNNs) have been developed to effectively capture the evolving patterns in dynamic graphs.
While these models have demonstrated their superiority, being widely adopted in various important fields, their vulnerabilities against adversarial attacks remain largely unexplored.
In this paper, we propose T-SPEAR, a simple and effective adversarial attack method for link prediction on continuous-time dynamic graphs, focusing on investigating the vulnerabilities of TGNNs.
Specifically, before the training procedure of a victim model, which is a TGNN for link prediction, we inject edge perturbations to the data that are unnoticeable in terms of the four constraints we propose, and yet effective enough to cause malfunction of the victim model. 
Moreover, we propose a robust training approach T-SHIELD to mitigate the impact of adversarial attacks.
By using edge filtering and enforcing temporal smoothness to node embeddings, we enhance the robustness of the victim model.
Our experimental study shows that T-SPEAR significantly degrades the victim model's performance on link prediction tasks, and even more, our attacks are transferable to other TGNNs, which differ from the victim model assumed by the attacker.
Moreover, we demonstrate that T-SHIELD effectively filters out adversarial edges and exhibits robustness against adversarial attacks, surpassing the link prediction performance of the naive TGNN by up to 11.2% under T-SPEAR.
The code and datasets are available at https://github.com/wooner49/T-spear-shield

----

## [1491] The Choice of Noninformative Priors for Thompson Sampling in Multiparameter Bandit Models

**Authors**: *Jongyeong Lee, Chao-Kai Chiang, Masashi Sugiyama*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29240](https://doi.org/10.1609/aaai.v38i12.29240)

**Abstract**:

Thompson sampling (TS) has been known for its outstanding empirical performance supported by theoretical guarantees across various reward models in the classical stochastic multi-armed bandit problems.
Nonetheless, its optimality is often restricted to specific priors due to the common observation that TS is fairly insensitive to the choice of the prior when it comes to asymptotic regret bounds.
However, when the model contains multiple parameters, the optimality of TS highly depends on the choice of priors, which casts doubt on the generalizability of previous findings to other models. 
To address this gap, this study explores the impact of selecting noninformative priors, offering insights into the performance of TS when dealing with new models that lack theoretical understanding.
We first extend the regret analysis of TS to the model of uniform distributions with unknown supports, which would be the simplest non-regular model. 
Our findings reveal that changing noninformative priors can significantly affect the expected regret, aligning with previously known results in other multiparameter bandit models.
Although the uniform prior is shown to be optimal, we highlight the inherent limitation of its optimality, which is limited to specific parameterizations and emphasizes the significance of the invariance property of priors.
In light of this limitation, we propose a slightly modified TS-based policy, called TS with Truncation (TS-T), which can achieve the asymptotic optimality for the Gaussian models and the uniform models by using the reference prior and the Jeffreys prior that are invariant under one-to-one reparameterizations.
This policy provides an alternative approach to achieving optimality by employing fine-tuned truncation, which would be much easier than hunting for optimal priors in practice.

----

## [1492] Learning Uncertainty-Aware Temporally-Extended Actions

**Authors**: *Joongkyu Lee, Seung Joon Park, Yunhao Tang, Min-hwan Oh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29241](https://doi.org/10.1609/aaai.v38i12.29241)

**Abstract**:

In reinforcement learning, temporal abstraction in the action space, exemplified by action repetition, is a technique to facilitate policy learning through extended actions. However, a primary limitation in previous studies of action repetition is its potential to degrade performance, particularly when sub-optimal actions are repeated. This issue often negates the advantages of action repetition. To address this, we propose a novel algorithm named Uncertainty-aware Temporal Extension (UTE). UTE employs ensemble methods to accurately measure uncertainty during action extension. This feature allows policies to strategically choose between emphasizing exploration or adopting an uncertainty-averse approach, tailored to their specific needs. We demonstrate the effectiveness of UTE through experiments in Gridworld and Atari 2600 environments. Our findings show that UTE outperforms existing action repetition algorithms, effectively mitigating their inherent limitations and significantly enhancing policy learning efficiency.

----

## [1493] Any-Way Meta Learning

**Authors**: *Junhoo Lee, Yearim Kim, Hyunho Lee, Nojun Kwak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29242](https://doi.org/10.1609/aaai.v38i12.29242)

**Abstract**:

Although meta-learning seems promising performance in the realm of rapid adaptability, it is constrained by 
fixed cardinality. When faced with tasks of varying cardinalities that were unseen during training, 
the model lacks its ability. In this paper, we address and resolve this challenge 
by harnessing `label equivalence' emerged from stochastic numeric label assignments during episodic task sampling. Questioning what defines ``true" meta-learning, we introduce the ``any-way" learning paradigm, an innovative model training approach that liberates model from
fixed cardinality constraints. Surprisingly, this model not only matches but often outperforms traditional fixed-way models in terms of performance, convergence speed, and stability. This disrupts established notions
about domain generalization. Furthermore, we argue that the inherent 
label equivalence naturally lacks semantic information. To bridge this 
semantic information gap arising from label equivalence, we further propose a mechanism for infusing semantic class information into the model. This would enhance the model's comprehension and functionality. Experiments conducted on renowned architectures like MAML and ProtoNet affirm the effectiveness of our method.

----

## [1494] Mixed-Effects Contextual Bandits

**Authors**: *Kyungbok Lee, Myunghee Cho Paik, Min-hwan Oh, Gi-Soo Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29243](https://doi.org/10.1609/aaai.v38i12.29243)

**Abstract**:

We study a novel variant of a contextual bandit problem with multi-dimensional reward feedback formulated as a mixed-effects model, where the correlations between multiple feedback are induced by sharing stochastic coefficients called random effects. We propose a novel algorithm, Mixed-Effects Contextual UCB (ME-CUCB), achieving tildeO(d sqrt(mT)) regret bound after T rounds where d is the dimension of contexts and m is the dimension of outcomes, with either known or unknown covariance structure. This is a tighter regret bound than that of the naive canonical linear bandit algorithm ignoring the correlations among rewards. We prove a lower bound of Omega(d sqrt(mT)) matching the upper bound up to logarithmic factors. To our knowledge, this is the first work providing a regret analysis for mixed-effects models and algorithms involving weighted least-squares estimators. Our theoretical analysis faces a significant technical challenge in that the error terms do not constitute martingales since the weights depend on the rewards. We overcome this challenge by using covering numbers, of theoretical interest in its own right. We provide numerical experiments demonstrating the advantage of our proposed algorithm, supporting the theoretical claims.

----

## [1495] Proxyformer: Nyström-Based Linear Transformer with Trainable Proxy Tokens

**Authors**: *Sangho Lee, Hayun Lee, Dongkun Shin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29244](https://doi.org/10.1609/aaai.v38i12.29244)

**Abstract**:

Transformer-based models have demonstrated remarkable performance in various domains, including natural language processing, image processing and generative modeling. The most significant contributor to the successful performance of Transformer models is the self-attention mechanism, which allows for a comprehensive understanding of the interactions between tokens in the input sequence. However, there is a well-known scalability issue, the quadratic dependency (i.e. O(n^2)) of self-attention operations on the input sequence length n, making the handling of lengthy sequences challenging. To address this limitation, there has been a surge of research on efficient transformers, aiming to alleviate the quadratic dependency on the input sequence length. Among these, the Nyströmformer, which utilizes the Nyström method to decompose the attention matrix, achieves superior performance in both accuracy and throughput. However, its landmark selection exhibits redundancy, and the model incurs computational overhead when calculating the pseudo-inverse matrix. We propose a novel Nyström method-based transformer, called Proxyformer. Unlike the traditional approach of selecting landmarks from input tokens, the Proxyformer utilizes trainable neural memory, called proxy tokens, for landmarks. By integrating contrastive learning, input injection, and a specialized dropout for the decomposed matrix, Proxyformer achieves top-tier performance for long sequence tasks in the Long Range Arena benchmark.

----

## [1496] Multi-Architecture Multi-Expert Diffusion Models

**Authors**: *Yunsung Lee, JinYoung Kim, Hyojun Go, Myeongho Jeong, Shinhyeok Oh, Seungtaek Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29245](https://doi.org/10.1609/aaai.v38i12.29245)

**Abstract**:

In this paper, we address the performance degradation of efficient diffusion models by introducing Multi-architecturE Multi-Expert diffusion models (MEME). We identify the need for tailored operations at different time-steps in diffusion processes and leverage this insight to create compact yet high-performing models. MEME assigns distinct architectures to different time-step intervals, balancing convolution and self-attention operations based on observed frequency characteristics. We also introduce a soft interval assignment strategy for comprehensive training. Empirically, MEME operates 3.3 times faster than baselines while improving image generation quality (FID scores) by 0.62 (FFHQ) and 0.37 (CelebA). Though we validate the effectiveness of assigning more optimal architecture per time-step, where efficient models outperform the larger models, we argue that MEME opens a new design choice for diffusion models that can be easily applied in other scenarios, such as large multi-expert models.

----

## [1497] PC-Conv: Unifying Homophily and Heterophily with Two-Fold Filtering

**Authors**: *Bingheng Li, Erlin Pan, Zhao Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29246](https://doi.org/10.1609/aaai.v38i12.29246)

**Abstract**:

Recently, many carefully designed graph representation learning methods have achieved impressive performance on either strong heterophilic or homophilic graphs, but not both. Therefore, they are incapable of generalizing well across real-world graphs with different levels of homophily. This is attributed to their neglect of homophily in heterophilic graphs, and vice versa. In this paper, we propose a two-fold filtering mechanism to mine homophily in heterophilic graphs, and vice versa. In particular, we extend the graph heat equation to perform heterophilic aggregation of global information from a long distance. The resultant filter can be exactly approximated by the Possion-Charlier (PC) polynomials. To further exploit information at multiple orders, we introduce a powerful graph convolution PC-Conv and its instantiation PCNet for the node classification task.  Compared to the state-of-the-art GNNs, PCNet shows competitive performance on well-known homophilic and heterophilic graphs. Our implementation is available at https://github.com/uestclbh/PC-Conv.

----

## [1498] All Beings Are Equal in Open Set Recognition

**Authors**: *Chaohua Li, Enhao Zhang, Chuanxing Geng, Songcan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29247](https://doi.org/10.1609/aaai.v38i12.29247)

**Abstract**:

In open-set recognition (OSR), a promising strategy is exploiting pseudo-unknown data outside given K known classes as an additional K+1-th class to explicitly model potential open space. However, treating unknown classes without distinction is unequal for them relative to known classes due to the category-agnostic and scale-agnostic of the unknowns. This inevitably not only disrupts the inherent distributions of unknown classes but also incurs both class-wise and instance-wise imbalances between known and unknown classes. Ideally, the OSR problem should model the whole class space as K+∞, but enumerating all unknowns is impractical. Since the core of OSR is to effectively model the boundaries of known classes, this means just focusing on the unknowns nearing the boundaries of targeted known classes seems sufficient. Thus, as a compromise, we convert the open classes from infinite to K, with a novel concept Target-Aware Universum (TAU) and propose a simple yet effective framework Dual Contrastive Learning with Target-Aware Universum (DCTAU). In details, guided by the targeted known classes, TAU automatically expands the unknown classes from the previous 1 to K, effectively alleviating the distribution disruption and the imbalance issues mentioned above. Then, a novel Dual Contrastive (DC) loss is designed, where all instances irrespective of known or TAU are considered as positives to contrast with their respective negatives. Experimental results indicate DCTAU sets a new state-of-the-art.

----

## [1499] GxVAEs: Two Joint VAEs Generate Hit Molecules from Gene Expression Profiles

**Authors**: *Chen Li, Yoshihiro Yamanishi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29248](https://doi.org/10.1609/aaai.v38i12.29248)

**Abstract**:

The de novo generation of hit-like molecules that show bioactivity and drug-likeness is an important task in computer-aided drug discovery. Although artificial intelligence can generate molecules with desired chemical properties, most previous studies have ignored the influence of disease-related cellular environments. This study proposes a novel deep generative model called GxVAEs to generate hit-like molecules from gene expression profiles by leveraging two joint variational autoencoders (VAEs). The first VAE, ProfileVAE, extracts latent features from gene expression profiles. The extracted features serve as the conditions that guide the second VAE, which is called MolVAE, in generating hit-like molecules. GxVAEs bridge the gap between molecular generation and the cellular environment in a biological system, and produce molecules that are biologically meaningful in the context of specific diseases. Experiments and case studies on the generation of therapeutic molecules show that GxVAEs outperforms current state-of-the-art baselines and yield hit-like molecules with potential bioactivity and drug-like properties. We were able to successfully generate the potential molecular structures with therapeutic effects for various diseases from patients’ disease profiles.

----

## [1500] Towards Continual Learning Desiderata via HSIC-Bottleneck Orthogonalization and Equiangular Embedding

**Authors**: *Depeng Li, Tianqi Wang, Junwei Chen, Qining Ren, Kenji Kawaguchi, Zhigang Zeng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29249](https://doi.org/10.1609/aaai.v38i12.29249)

**Abstract**:

Deep neural networks are susceptible to catastrophic forgetting when trained on sequential tasks. Various continual learning (CL) methods often rely on exemplar buffers or/and network expansion for balancing model stability and plasticity, which, however, compromises their practical value due to privacy and memory concerns. Instead, this paper considers a strict yet realistic setting, where the training data from previous tasks is unavailable and the model size remains relatively constant during sequential training. To achieve such desiderata, we propose a conceptually simple yet effective method that attributes forgetting to layer-wise parameter overwriting and the resulting decision boundary distortion. This is achieved by the synergy between two key components: HSIC-Bottleneck Orthogonalization (HBO) implements non-overwritten parameter updates mediated by Hilbert-Schmidt independence criterion in an orthogonal space and EquiAngular Embedding (EAE) enhances decision boundary adaptation between old and new tasks with predefined basis vectors. Extensive experiments demonstrate that our method achieves competitive accuracy performance, even with absolute superiority of zero exemplar buffer and 1.02x the base model.

----

## [1501] Regroup Median Loss for Combating Label Noise

**Authors**: *Fengpeng Li, Kemou Li, Jinyu Tian, Jiantao Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29250](https://doi.org/10.1609/aaai.v38i12.29250)

**Abstract**:

The deep model training procedure requires large-scale datasets of annotated data. Due to the difficulty of annotating a large number of samples, label noise caused by incorrect annotations is inevitable, resulting in low model performance and poor model generalization. To combat label noise, current methods usually select clean samples based on the small-loss criterion and use these samples for training. Due to some noisy samples similar to clean ones, these small-loss criterion-based methods are still affected by label noise. To address this issue, in this work, we propose Regroup Median Loss (RML) to reduce the probability of selecting noisy samples and correct losses of noisy samples. RML randomly selects samples with the same label as the training samples based on a new loss processing method. Then, we combine the stable mean loss and the robust median loss through a proposed regrouping strategy to obtain robust loss estimation for noisy samples. To further improve the model performance against label noise, we propose a new sample selection strategy and build a semi-supervised method based on RML. Compared to state-of-the-art methods, for both the traditionally trained and semi-supervised models, RML achieves a significant improvement on synthetic and complex real-world datasets. The source is at https://github.com/Feng-peng-Li/Regroup-Loss-Median-to-Combat-Label-Noise.

----

## [1502] Parsing All Adverse Scenes: Severity-Aware Semantic Segmentation with Mask-Enhanced Cross-Domain Consistency

**Authors**: *Fuhao Li, Ziyang Gong, Yupeng Deng, Xianzheng Ma, Renrui Zhang, Zhenming Ji, Xiangwei Zhu, Hong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29251](https://doi.org/10.1609/aaai.v38i12.29251)

**Abstract**:

Although recent methods in Unsupervised Domain Adaptation (UDA) have achieved success in segmenting rainy or snowy scenes by improving consistency, they face limitations when dealing with more challenging scenarios like foggy and night scenes. We argue that these prior methods excessively focus on weather-specific features in adverse scenes, which exacerbates the existing domain gaps.
To address this issue, we propose a new metric to evaluate the severity of all adverse scenes and offer a novel perspective that enables task unification across all adverse scenarios. Our method focuses on Severity, allowing our model to learn more consistent features and facilitate domain distribution alignment, thereby alleviating domain gaps. Unlike the vague descriptions of consistency in previous methods, we introduce Cross-domain Consistency, which is quantified using the Structure Similarity Index Measure (SSIM) to measure the distance between the source and target domains. Specifically, our unified model consists of two key modules: the Merging Style Augmentation Module (MSA) and the Severity Perception Mask Module (SPM). The MSA module transforms all adverse scenes into augmented scenes, effectively eliminating weather-specific features and enhancing Cross-domain Consistency. The SPM module incorporates a Severity Perception mechanism, guiding a Mask operation that enables our model to learn highly consistent features from the augmented scenes. Our unified framework, named PASS (Parsing All adverSe Scenes), achieves significant performance improvements over state-of-the-art methods on widely-used benchmarks for all adverse scenes. Notably, the performance of PASS is superior to Semi-Unified models and even surpasses weather-specific models.

----

## [1503] Learning Spatially Collaged Fourier Bases for Implicit Neural Representation

**Authors**: *Jason Chun Lok Li, Chang Liu, Binxiao Huang, Ngai Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29252](https://doi.org/10.1609/aaai.v38i12.29252)

**Abstract**:

Existing approaches to Implicit Neural Representation (INR) can be interpreted as a global scene representation via a linear combination of Fourier bases of different frequencies. However, such universal basis functions can limit the representation capability in local regions where a specific component is unnecessary, resulting in unpleasant artifacts. To this end, we introduce a learnable spatial mask that effectively dispatches distinct Fourier bases into respective regions. This translates into collaging Fourier patches, thus enabling an accurate representation of complex signals. Comprehensive experiments demonstrate the superior reconstruction quality of the proposed approach over existing baselines across various INR tasks, including image fitting, video representation, and 3D shape representation. Our method outperforms all other baselines, improving the image fitting PSNR by over 3dB and 3D reconstruction to 98.81 IoU and 0.0011 Chamfer Distance.

----

## [1504] High-Dimensional Analysis for Generalized Nonlinear Regression: From Asymptotics to Algorithm

**Authors**: *Jian Li, Yong Liu, Weiping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29253](https://doi.org/10.1609/aaai.v38i12.29253)

**Abstract**:

Overparameterization often leads to benign overfitting, where deep neural networks can be trained to overfit the training data but still generalize well on unseen data. However, it lacks a generalized asymptotic framework for nonlinear regressions and connections to conventional complexity notions. In this paper, we propose a generalized high-dimensional analysis for nonlinear regression models, including various nonlinear feature mapping methods and subsampling. Specifically, we first provide an implicit regularization parameter and asymptotic equivalents related to a classical complexity notion, i.e., effective dimension. We then present a high-dimensional analysis for nonlinear ridge regression and extend it to ridgeless regression in the under-parameterized and over-parameterized regimes, respectively. We find that the limiting risks decrease with the effective dimension. Motivated by these theoretical findings, we propose an algorithm, namely RFRed, to improve generalization ability. Finally, we validate our theoretical findings and the proposed algorithm through several experiments.

----

## [1505] FedNS: A Fast Sketching Newton-Type Algorithm for Federated Learning

**Authors**: *Jian Li, Yong Liu, Weiping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29254](https://doi.org/10.1609/aaai.v38i12.29254)

**Abstract**:

Recent Newton-type federated learning algorithms have demonstrated linear convergence with respect to the communication rounds. However, communicating Hessian matrices is often unfeasible due to their quadratic communication complexity. In this paper, we introduce a novel approach to tackle this issue while still achieving fast convergence rates. Our proposed method, named as Federated Newton Sketch methods (FedNS), approximates the centralized Newton's method by communicating the sketched square-root Hessian instead of the exact Hessian. To enhance communication efficiency, we reduce the sketch size to match the effective dimension of the Hessian matrix. We provide convergence analysis based on statistical learning for the federated Newton sketch approaches. Specifically, our approaches reach super-linear convergence rates w.r.t. the communication rounds for the first time. We validate the effectiveness of our algorithms through various experiments, which coincide with our theoretical findings.

----

## [1506] Hierarchical Topology Isomorphism Expertise Embedded Graph Contrastive Learning

**Authors**: *Jiangmeng Li, Yifan Jin, Hang Gao, Wenwen Qiang, Changwen Zheng, Fuchun Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29255](https://doi.org/10.1609/aaai.v38i12.29255)

**Abstract**:

Graph contrastive learning (GCL) aims to align the positive features while differentiating the negative features in the latent space by minimizing a pair-wise contrastive loss. As the embodiment of an outstanding discriminative unsupervised graph representation learning approach, GCL achieves impressive successes in various graph benchmarks. However, such an approach falls short of recognizing the topology isomorphism of graphs, resulting in that graphs with relatively homogeneous node features cannot be sufficiently discriminated. By revisiting classic graph topology recognition works, we disclose that the corresponding expertise intuitively complements GCL methods. To this end, we propose a novel hierarchical topology isomorphism expertise embedded graph contrastive learning, which introduces knowledge distillations to empower GCL models to learn the hierarchical topology isomorphism expertise, including the graph-tier and subgraph-tier. On top of this, the proposed method holds the feature of plug-and-play, and we empirically demonstrate that the proposed method is universal to multiple state-of-the-art GCL models. The solid theoretical analyses are further provided to prove that compared with conventional GCL methods, our method acquires the tighter upper bound of Bayes classification error. We conduct extensive experiments on real-world benchmarks to exhibit the performance superiority of our method over candidate GCL methods, e.g., for the real-world graph representation learning experiments, the proposed method beats the state-of-the-art method by 0.23% on unsupervised representation learning setting, 0.43% on transfer learning setting. Our code is available at https://github.com/jyf123/HTML.

----

## [1507] Curriculum-Enhanced Residual Soft An-Isotropic Normalization for Over-Smoothness in Deep GNNs

**Authors**: *Jin Li, Qirong Zhang, Shuling Xu, Xinlong Chen, Longkun Guo, Yang-Geng Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29256](https://doi.org/10.1609/aaai.v38i12.29256)

**Abstract**:

Despite Graph neural networks' significant performance gain over many classic techniques in various graph-related downstream tasks, their successes are restricted in shallow models due to over-smoothness and the difficulties of optimizations among many other issues. In this paper, to alleviate the over-smoothing issue, we propose a soft graph normalization method to preserve the diversities of node embeddings and prevent indiscrimination due to possible over-closeness. Combined with residual connections, we analyze the reason why the method can effectively capture the knowledge in both input graph structures and node features even with deep networks. Additionally, inspired by Curriculum Learning that learns easy examples before the hard ones, we propose a novel label-smoothing-based learning framework to enhance the optimization of deep GNNs, which iteratively smooths labels in an auxiliary graph and constructs many gradual non-smooth tasks for extracting increasingly complex knowledge and gradually discriminating nodes from coarse to fine. The method arguably reduces the risk of overfitting and generalizes better results. Finally, extensive experiments are carried out to demonstrate the effectiveness and potential of the proposed model and learning framework through comparison with twelve existing baselines including the state-of-the-art methods on twelve real-world node classification benchmarks.

----

## [1508] Tensorized Label Learning on Anchor Graph

**Authors**: *Jing Li, Quanxue Gao, Qianqian Wang, Wei Xia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29257](https://doi.org/10.1609/aaai.v38i12.29257)

**Abstract**:

Graph-based multimedia data clustering has attracted much attention due to the impressive clustering performance for arbitrarily shaped multimedia data. However, existing graph-based clustering methods need post-processing to get labels for multimedia data with high computational complexity. Moreover, it is sub-optimal for label learning due to the fact that they exploit the complementary information embedded in data with different types pixel by pixel. To handle these problems, we present a novel label learning model with good interpretability for clustering. To be specific, our model decomposes anchor graph into the products of two matrices with orthogonal non-negative constraint to directly get soft label without any post-processing, which remarkably reduces the computational complexity. To well exploit the complementary information embedded in multimedia data, we introduce tensor Schatten p-norm regularization on the label tensor which is composed of soft labels of multimedia data. The solution can be obtained by iteratively optimizing four decoupled sub-problems, which can be solved more efficiently with good convergence. Experimental results on various datasets demonstrate the efficiency of our model.

----

## [1509] EMGAN: Early-Mix-GAN on Extracting Server-Side Model in Split Federated Learning

**Authors**: *Jingtao Li, Xing Chen, Li Yang, Adnan Siraj Rakin, Deliang Fan, Chaitali Chakrabarti*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29258](https://doi.org/10.1609/aaai.v38i12.29258)

**Abstract**:

Split Federated Learning (SFL) is an emerging edge-friendly version of Federated Learning (FL), where clients process a small portion of the entire model. While SFL was considered to be resistant to Model Extraction Attack (MEA) by design, a recent work shows it is not necessarily the case. In general, gradient-based MEAs are not effective on a target model that is changing, as is the case in training-from-scratch applications. In this work, we propose a strong MEA during the SFL training phase. The proposed Early-Mix-GAN (EMGAN) attack effectively exploits gradient queries regardless of data assumptions. EMGAN adopts three key components to address the problem of inconsistent gradients. Specifically, it employs (i) Early-learner approach for better adaptability, (ii) Multi-GAN approach to introduce randomness in generator training to mitigate mode collapse, and (iii) ProperMix to effectively augment the limited amount of synthetic data for a better approximation of the target domain data distribution. EMGAN achieves excellent results in extracting server-side models. With only 50 training samples, EMGAN successfully extracts a 5-layer server-side model of VGG-11 on CIFAR-10, with 7% less accuracy than the target model. With zero training data, the extracted model achieves 81.3% accuracy, which is significantly better than the 45.5% accuracy of the model extracted by the SoTA method. The code is available at "https://github.com/zlijingtao/SFL-MEA".

----

## [1510] Contrastive Continual Learning with Importance Sampling and Prototype-Instance Relation Distillation

**Authors**: *Jiyong Li, Dilshod Azizov, Yang Li, Shangsong Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29259](https://doi.org/10.1609/aaai.v38i12.29259)

**Abstract**:

Recently, because of the high-quality representations of contrastive learning methods, rehearsal-based contrastive continual learning has been proposed to explore how to continually learn transferable representation embeddings to avoid the catastrophic forgetting issue in traditional continual settings. Based on this framework, we propose Contrastive Continual Learning via Importance Sampling (CCLIS) to preserve knowledge by recovering previous data distributions with a new strategy for Replay Buffer Selection (RBS), which minimize estimated variance to save hard negative samples for representation learning with high quality. Furthermore, we present the Prototype-instance Relation Distillation (PRD) loss, a technique designed to maintain the relationship between prototypes and sample representations using a self-distillation process. Experiments on standard continual learning benchmarks reveal that our method notably outperforms existing baselines in terms of knowledge preservation and thereby effectively counteracts catastrophic forgetting in online contexts. The code is available at https://github.com/lijy373/CCLIS.

----

## [1511] Twice Class Bias Correction for Imbalanced Semi-supervised Learning

**Authors**: *Lan Li, Bowen Tao, Lu Han, De-Chuan Zhan, Han-Jia Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29260](https://doi.org/10.1609/aaai.v38i12.29260)

**Abstract**:

Differing from traditional semi-supervised learning, class-imbalanced semi-supervised learning presents two distinct challenges: (1) The imbalanced distribution of training samples leads to model bias towards certain classes, and (2) the distribution of unlabeled samples is unknown and potentially distinct from that of labeled samples, which further contributes to class bias in the pseudo-labels during the training. To address these dual challenges, we introduce a novel approach called Twice Class Bias Correction (TCBC). We begin by utilizing an estimate of the class distribution from the participating training samples to correct the model, enabling it to learn the posterior probabilities of samples under a class-balanced prior. This correction serves to alleviate the inherent class bias of the model. Building upon this foundation, we further estimate the class bias of the current model parameters during the training process. We apply a secondary correction to the model's pseudo-labels for unlabeled samples, aiming to make the assignment of pseudo-labels across different classes of unlabeled samples as equitable as possible. Through extensive experimentation on CIFAR10/100-LT, STL10-LT, and the sizable long-tailed dataset SUN397, we provide conclusive evidence that our proposed TCBC method reliably enhances the performance of class-imbalanced semi-supervised learning.

----

## [1512] Dynamic Regret of Adversarial MDPs with Unknown Transition and Linear Function Approximation

**Authors**: *Long-Fei Li, Peng Zhao, Zhi-Hua Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29261](https://doi.org/10.1609/aaai.v38i12.29261)

**Abstract**:

We study reinforcement learning (RL) in episodic MDPs with adversarial full-information losses and the unknown transition. Instead of the classical static regret, we adopt dynamic regret as the performance measure which benchmarks the learner's performance with changing policies, making it more suitable for non-stationary environments. The primary challenge is to handle the uncertainties of unknown transition and unknown non-stationarity of environments simultaneously. We propose a general framework to decouple the two sources of uncertainties and show the dynamic regret bound naturally decomposes into two terms, one due to constructing confidence sets to handle the unknown transition and the other due to choosing sub-optimal policies under the unknown non-stationarity. To this end, we first employ the two-layer online ensemble structure to handle the adaptation error due to the unknown non-stationarity, which is model-agnostic. Subsequently, we instantiate the framework to three fundamental MDP models, including tabular MDPs, linear MDPs and linear mixture MDPs, and present corresponding approaches to control the exploration error due to the unknown transition. We provide dynamic regret guarantees respectively and show they are optimal in terms of the number of episodes K and the non-stationarity P̄ᴋ by establishing matching lower bounds. To the best of our knowledge, this is the first work that achieves the dynamic regret exhibiting optimal dependence on K and P̄ᴋ without prior knowledge about the non-stationarity for adversarial MDPs with unknown transition.

----

## [1513] Feature Fusion from Head to Tail for Long-Tailed Visual Recognition

**Authors**: *Mengke Li, Zhikai Hu, Yang Lu, Weichao Lan, Yiu-ming Cheung, Hui Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29262](https://doi.org/10.1609/aaai.v38i12.29262)

**Abstract**:

The imbalanced distribution of long-tailed data presents a considerable challenge for deep learning models, as it causes them to prioritize the accurate classification of head classes but largely disregard tail classes. The biased decision boundary caused by inadequate semantic information in tail classes is one of the key factors contributing to their low recognition accuracy. To rectify this issue, we propose to augment tail classes by grafting the diverse semantic information from head classes, referred to as head-to-tail fusion (H2T). We replace a portion of feature maps from tail classes with those belonging to head classes. These fused features substantially enhance the diversity of tail classes. Both theoretical analysis and practical experimentation demonstrate that H2T can contribute to a more optimized solution for the decision boundary. We seamlessly integrate H2T in the classifier adjustment stage, making it a plug-and-play module. Its simplicity and ease of implementation allow for smooth integration with existing long-tailed recognition methods, facilitating a further performance boost. Extensive experiments on various long-tailed benchmarks demonstrate the effectiveness of the proposed H2T. The source code is available at https://github.com/Keke921/H2T.

----

## [1514] Narrowing the Gap between Supervised and Unsupervised Sentence Representation Learning with Large Language Model

**Authors**: *Mingxin Li, Richong Zhang, Zhijie Nie, Yongyi Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29263](https://doi.org/10.1609/aaai.v38i12.29263)

**Abstract**:

Sentence Representation Learning (SRL) is a fundamental task in Natural Language Processing (NLP), with the Contrastive Learning of Sentence Embeddings (CSE) being the mainstream technique due to its superior performance. An intriguing phenomenon in CSE is the significant performance gap between supervised and unsupervised methods, with their only difference lying in the training data. Previous works attribute this performance gap to differences in two representation properties (alignment and uniformity). However, since alignment and uniformity only measure the results, they fail to answer "What aspects of the training data contribute to the performance gap?" and "How can the performance gap be narrowed?", In this paper, we conduct empirical experiments to answer these "What" and "How" questions. We first answer the "What" question by thoroughly comparing the behavior of supervised and unsupervised CSE during their respective training processes. From the comparison, we identify the similarity pattern as a key factor to the performance gap, and introduce a metric, called Relative Fitting Difficulty (RFD), to measure the complexity of the similarity pattern. Then, based on the insights gained from the "What" question, we tackle the "How" question by increasing the pattern complexity of the training data. We achieve this by leveraging the In-Context Learning (ICL) capability of the Large Language Model (LLM) to generate data that simulates complex patterns. By utilizing the hierarchical patterns in the LLM-generated data, we effectively narrow the gap between supervised and unsupervised CSE. We release our codes and appendix at https://github.com/BDBC-KG-NLP/NGCSE.

----

## [1515] AdapterGNN: Parameter-Efficient Fine-Tuning Improves Generalization in GNNs

**Authors**: *Shengrui Li, Xueting Han, Jing Bai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29264](https://doi.org/10.1609/aaai.v38i12.29264)

**Abstract**:

Fine-tuning pre-trained models has recently yielded remarkable performance gains in graph neural networks (GNNs). In addition to pre-training techniques, inspired by the latest work in the natural language fields, more recent work has shifted towards applying effective fine-tuning approaches, such as parameter-efficient fine-tuning (PEFT). However, given the substantial differences between GNNs and transformer-based models, applying such approaches directly to GNNs proved to be less effective. In this paper, we present a comprehensive comparison of PEFT techniques for GNNs and propose a novel PEFT method specifically designed for GNNs, called AdapterGNN. AdapterGNN preserves the knowledge of the large pre-trained model and leverages highly expressive adapters for GNNs, which can adapt to downstream tasks effectively with only a few parameters, while also improving the model's generalization ability. Extensive experiments show that AdapterGNN achieves higher performance than other PEFT methods and is the only one consistently surpassing full fine-tuning (outperforming it by 1.6% and 5.7% in the chemistry and biology domains respectively, with only 5% and 4% of its parameters tuned) with lower generalization gaps. Moreover, we empirically show that a larger GNN model can have a worse generalization ability, which differs from the trend observed in large transformer-based models. Building upon this, we provide a theoretical justification for PEFT can improve generalization of GNNs by applying generalization bounds. Our code is available at https://github.com/Lucius-lsr/AdapterGNN.

----

## [1516] Robust Visual Imitation Learning with Inverse Dynamics Representations

**Authors**: *Siyuan Li, Xun Wang, Rongchang Zuo, Kewu Sun, Lingfei Cui, Jishiyu Ding, Peng Liu, Zhe Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29265](https://doi.org/10.1609/aaai.v38i12.29265)

**Abstract**:

Imitation learning (IL) has achieved considerable success in solving complex sequential decision-making problems. However, current IL methods mainly assume that the environment for learning policies is the same as the environment for collecting expert datasets. Therefore, these methods may fail to work when there are slight differences between the learning and expert environments, especially for challenging problems with high-dimensional image observations. However, in real-world scenarios, it is rare to have the chance to collect expert trajectories precisely in the target learning environment. To address this challenge, we propose a novel robust imitation learning approach, where we develop an inverse dynamics state representation learning objective to align the expert environment and the learning environment. With the abstract state representation, we design an effective reward function, which thoroughly measures the similarity between behavior data and expert data not only element-wise, but also from the trajectory level. We conduct extensive experiments to evaluate the proposed approach under various visual perturbations and in diverse visual control tasks. Our approach can achieve a near-expert performance in most environments, and significantly outperforms the state-of-the-art visual IL methods and robust IL methods.

----

## [1517] Cumulative Difference Learning VAE for Time-Series with Temporally Correlated Inflow-Outflow

**Authors**: *Tianchun Li, Chengxiang Wu, Pengyi Shi, Xiaoqian Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29266](https://doi.org/10.1609/aaai.v38i12.29266)

**Abstract**:

Time-series generation has crucial practical significance for decision-making under uncertainty. Existing methods have various limitations like accumulating errors over time, significantly impacting downstream tasks. We develop a novel generation method, DT-VAE, that incorporates generalizable domain knowledge, is mathematically justified, and significantly outperforms existing methods by mitigating error accumulation through a cumulative difference learning mechanism. We evaluate the performance of DT-VAE on several downstream tasks using both semi-synthetic and real time-series datasets, including benchmark datasets and our newly curated COVID-19 hospitalization datasets. The COVID-19 datasets enrich existing resources for time-series analysis. Additionally, we introduce Diverse Trend Preserving (DTP), a time-series clustering-based evaluation for direct and interpretable assessments of generated samples, serving as a valuable tool for evaluating time-series generative models.

----

## [1518] Federated X-armed Bandit

**Authors**: *Wenjie Li, Qifan Song, Jean Honorio, Guang Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29267](https://doi.org/10.1609/aaai.v38i12.29267)

**Abstract**:

This work establishes the first framework of federated X-armed bandit, where different clients face heterogeneous local objective functions defined on the same domain and are required to collaboratively figure out the global optimum. We propose the first federated algorithm for such problems, named Fed-PNE. By utilizing the topological structure of the global objective inside the hierarchical partitioning and the weak smoothness property, our algorithm achieves sublinear cumulative regret with respect to both the number of clients and the evaluation budget. Meanwhile, it only requires logarithmic communications between the central server and clients, protecting the client privacy. Experimental results on synthetic functions and real datasets validate the advantages of Fed-PNE over various centralized and federated baseline algorithms.

----

## [1519] Unsupervised Training Sequence Design: Efficient and Generalizable Agent Training

**Authors**: *Wenjun Li, Pradeep Varakantham*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29268](https://doi.org/10.1609/aaai.v38i12.29268)

**Abstract**:

To train generalizable Reinforcement Learning (RL) agents, researchers recently proposed the Unsupervised Environment Design (UED) framework, in which a teacher agent creates a very large number of training environments and a student agent trains on the experiences in these environments to be robust against unseen testing scenarios. For example, to train a student to master the “stepping over stumps” task, the teacher will create numerous training environments with varying stump heights and shapes. In this paper, we argue that UED neglects training efficiency and its need for very large number of environments (henceforth referred to as infinite horizon training) makes it less suitable to training robots and non-expert humans. In real-world applications where either creating new training scenarios is expensive or training efficiency is of critical importance, we want to maximize both the learning efficiency and learning outcome of the student. To achieve efficient finite horizon training, we propose a novel Markov Decision Process (MDP) formulation for the teacher agent, referred to as Unsupervised Training Sequence Design (UTSD). Specifically, we encode salient information from the student policy (e.g., behaviors and learning progress) into the teacher's state space, enabling the teacher to closely track the student's learning progress and consequently discover the optimal training sequences with finite lengths. Additionally, we explore the teacher's efficient adaptation to unseen students at test time by employing the context-based meta-learning approach, which leverages the teacher's past experiences with various students. Finally, we empirically demonstrate our teacher's capability to design efficient and effective training sequences for students with varying capabilities.

----

## [1520] Image Content Generation with Causal Reasoning

**Authors**: *Xiaochuan Li, Baoyu Fan, Runze Zhang, Liang Jin, Di Wang, Zhenhua Guo, Yaqian Zhao, Rengang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29269](https://doi.org/10.1609/aaai.v38i12.29269)

**Abstract**:

The emergence of ChatGPT has once again sparked research in generative artificial intelligence (GAI). While people have been amazed by the generated results, they have also noticed the reasoning potential reflected in the generated textual content.  However, this current ability for causal reasoning is primarily limited to the domain of language generation, such as in models like GPT-3. In visual modality, there is currently no equivalent research. Considering causal reasoning in visual content generation is significant. This is because visual information contains infinite granularity. Particularly, images can provide more intuitive and specific demonstrations for certain reasoning tasks, especially when compared to coarse-grained text. Hence, we propose a new image generation task called visual question answering with image (VQAI) and establish a dataset of the same name based on the classic Tom and Jerry animated series.  Additionally, we develop a new paradigm for image generation to tackle the challenges of this task.  Finally, we perform extensive experiments and analyses, including visualizations of the generated content and discussions on the potentials and limitations. The code and data are publicly available under the license of CC BY-NC-SA 4.0 for academic and non-commercial usage at: https://github.com/IEIT-AGI/MIX-Shannon/blob/main/projects/VQAI/lgd_vqai.md.

----

## [1521] Deep Active Learning with Noise Stability

**Authors**: *Xingjian Li, Pengkun Yang, Yangcheng Gu, Xueying Zhan, Tianyang Wang, Min Xu, Chengzhong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29270](https://doi.org/10.1609/aaai.v38i12.29270)

**Abstract**:

Uncertainty estimation for unlabeled data is crucial to active learning. With a deep neural network employed as the backbone model, the data selection process is highly challenging due to the potential over-confidence of the model inference. Existing methods resort to special learning fashions (e.g. adversarial) or auxiliary models to address this challenge. This tends to result in complex and inefficient pipelines, which would render the methods impractical. In this work, we propose a novel algorithm that leverages noise stability to estimate data uncertainty. The key idea is to measure the output derivation from the original observation when the model parameters are randomly perturbed by noise. We provide theoretical analyses by leveraging the small Gaussian noise theory and demonstrate that our method favors a subset with large and diverse gradients. Our method is generally applicable in various tasks, including computer vision, natural language processing, and structural data analysis. It achieves competitive performance compared against state-of-the-art active learning baselines.

----

## [1522] Distribution-Conditioned Adversarial Variational Autoencoder for Valid Instrumental Variable Generation

**Authors**: *Xinshu Li, Lina Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29271](https://doi.org/10.1609/aaai.v38i12.29271)

**Abstract**:

Instrumental variables (IVs), widely applied in economics and healthcare, enable consistent counterfactual prediction in the presence of hidden confounding factors, effectively addressing endogeneity issues. The prevailing IV-based counterfactual prediction methods typically rely on the availability of valid IVs (satisfying Relevance, Exclusivity, and Exogeneity), a requirement which often proves elusive in real-world scenarios. Various data-driven techniques are being developed to create valid IVs (or representations of IVs) from a pool of IV candidates. However, most of these techniques still necessitate the inclusion of valid IVs within the set of candidates. This paper proposes a distribution-conditioned adversarial variational autoencoder to tackle this challenge. Specifically: 1) for Relevance and Exclusivity, we deduce the corresponding evidence lower bound following the Bayesian network structure and build the variational autoencoder; accordingly, 2) for Exogeneity , we design an adversarial game to encourage latent factors originating from the marginal distribution, compelling the independence between IVs and other outcome-related factors. Extensive experimental results validate the effectiveness, stability and generality of our proposed model in generating valid IV factors in the absence of valid IV candidates.

----

## [1523] Agile Multi-Source-Free Domain Adaptation

**Authors**: *Xinyao Li, Jingjing Li, Fengling Li, Lei Zhu, Ke Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29272](https://doi.org/10.1609/aaai.v38i12.29272)

**Abstract**:

Efficiently utilizing rich knowledge in pretrained models has become a critical topic in the era of large models. This work focuses on adaptively utilize knowledge from multiple source-pretrained models to an unlabeled target domain without accessing the source data. Despite being a practically useful setting, existing methods require extensive parameter tuning over each source model, which is computationally expensive when facing abundant source domains or larger source models. To address this challenge, we propose a novel approach which is free of the parameter tuning over source backbones. Our technical contribution lies in the Bi-level ATtention ENsemble (Bi-ATEN) module, which learns both intra-domain weights and inter-domain ensemble weights to achieve a fine balance between instance specificity and domain consistency. By slightly tuning source bottlenecks, we achieve comparable or even superior performance on a challenging benchmark DomainNet with less than 3% trained parameters and 8 times of throughput compared with SOTA method. Furthermore, with minor modifications, the proposed module can be easily equipped to existing methods and gain more than 4% performance boost. Code is available at https://github.com/TL-UESTC/Bi-ATEN.

----

## [1524] Towards Effective and General Graph Unlearning via Mutual Evolution

**Authors**: *Xunkai Li, Yulin Zhao, Zhengyu Wu, Wentao Zhang, Rong-Hua Li, Guoren Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29273](https://doi.org/10.1609/aaai.v38i12.29273)

**Abstract**:

With the rapid advancement of AI applications, the growing needs for data privacy and model robustness have highlighted the importance of machine unlearning, especially in thriving graph-based scenarios. However, most existing graph unlearning strategies primarily rely on well-designed architectures or manual process, rendering them less user-friendly and posing challenges in terms of deployment efficiency. Furthermore, striking a balance between unlearning performance and framework generalization is also a pivotal concern. To address the above issues, we propose Mutual Evolution Graph Unlearning (MEGU), a new mutual evolution paradigm that simultaneously evolves the predictive and unlearning capacities of graph unlearning. By incorporating aforementioned two components, MEGU ensures complementary optimization in a unified training framework that aligns with the prediction and unlearning requirements. Extensive experiments on 9 graph benchmark datasets demonstrate the superior performance of MEGU in addressing unlearning requirements at the feature, node, and edge levels. Specifically, MEGU achieves average performance improvements of 2.7%, 2.5%, and 3.2% across these three levels of unlearning tasks when compared to state-of-the-art baselines. Furthermore, MEGU exhibits satisfactory training efficiency, reducing time and space overhead by an average of 159.8x and 9.6x, respectively, in comparison to retraining GNN from scratch.

----

## [1525] Component Fourier Neural Operator for Singularly Perturbed Differential Equations

**Authors**: *Ye Li, Ting Du, Yiwen Pang, Zhongyi Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29274](https://doi.org/10.1609/aaai.v38i12.29274)

**Abstract**:

Solving Singularly Perturbed Differential Equations (SPDEs) poses computational challenges arising from the rapid transitions in their solutions within thin regions. The effectiveness of deep learning in addressing differential equations motivates us to employ these methods for solving SPDEs. In this paper, we introduce Component Fourier Neural Operator (ComFNO), an innovative operator learning method that builds upon Fourier Neural Operator (FNO), while simultaneously incorporating valuable prior knowledge obtained from asymptotic analysis. Our approach is not limited to FNO and can be applied to other neural network frameworks, such as Deep Operator Network (DeepONet), leading to potential similar SPDEs solvers. Experimental results across diverse classes of SPDEs demonstrate that ComFNO significantly improves accuracy compared to vanilla FNO. Furthermore, ComFNO exhibits natural adaptability to diverse data distributions and performs well in few-shot scenarios, showcasing its excellent generalization ability in practical situations.

----

## [1526] Learning to Prompt Knowledge Transfer for Open-World Continual Learning

**Authors**: *Yujie Li, Xin Yang, Hao Wang, Xiangkun Wang, Tianrui Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29275](https://doi.org/10.1609/aaai.v38i12.29275)

**Abstract**:

This paper studies the problem of continual learning in an open-world scenario, referred to as Open-world Continual Learning (OwCL). OwCL is increasingly rising while it is highly challenging in two-fold: i) learning a sequence of tasks without forgetting knowns in the past, and ii) identifying unknowns (novel objects/classes) in the future. Existing OwCL methods suffer from the adaptability of task-aware boundaries between knowns and unknowns, and do not consider the mechanism of knowledge transfer. In this work, we propose Pro-KT, a novel prompt-enhanced knowledge transfer model for OwCL. Pro-KT includes two key components: (1) a prompt bank to encode and transfer both task-generic and task-specific knowledge, and (2) a task-aware open-set boundary to identify unknowns in the new tasks. Experimental results using two real-world datasets demonstrate that the proposed Pro-KT outperforms the state-of-the-art counterparts in both the detection of unknowns and the classification of knowns markedly. Code released at https://github.com/YujieLi42/Pro-KT.

----

## [1527] SPD-DDPM: Denoising Diffusion Probabilistic Models in the Symmetric Positive Definite Space

**Authors**: *Yunchen Li, Zhou Yu, Gaoqi He, Yunhang Shen, Ke Li, Xing Sun, Shaohui Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29276](https://doi.org/10.1609/aaai.v38i12.29276)

**Abstract**:

Symmetric positive definite(SPD) matrices have shown important value and applications in statistics and machine learning, such as FMRI analysis and traffic prediction. Previous works on SPD matrices mostly focus on discriminative models, where predictions are made directly on E(X|y), where y is a vector and X is an SPD matrix. However, these methods are challenging to handle for large-scale data. In this paper, inspired by denoising diffusion probabilistic model(DDPM), we propose a novel generative model, termed SPD-DDPM, by introducing Gaussian distribution in the SPD space to estimate E(X|y). Moreover, our model can estimate p(X) unconditionally and flexibly without giving y. On the one hand, the model conditionally learns p(X|y) and utilizes the mean of samples to obtain E(X|y) as a prediction. On the other hand, the model unconditionally learns the probability distribution of the data p(X) and generates samples that conform to this distribution. Furthermore, we propose a new SPD net which is much deeper than the previous networks and allows for the inclusion of conditional factors. Experiment results on toy data and real taxi data demonstrate that our models effectively fit the data distribution both unconditionally and conditionally.

----

## [1528] Backpropagation Through Agents

**Authors**: *Zhiyuan Li, Wenshuai Zhao, Lijun Wu, Joni Pajarinen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29277](https://doi.org/10.1609/aaai.v38i12.29277)

**Abstract**:

A fundamental challenge in multi-agent reinforcement learning (MARL) is to learn the joint policy in an extremely large search space, which grows exponentially with the number of agents. Moreover, fully decentralized policy factorization significantly restricts the search space, which may lead to sub-optimal policies. In contrast, the auto-regressive joint policy can represent a much richer class of joint policies by factorizing the joint policy into the product of a series of conditional individual policies. While such factorization introduces the action dependency among agents explicitly in sequential execution, it does not take full advantage of the dependency during learning. In particular, the subsequent agents do not give the preceding agents feedback about their decisions. In this paper, we propose a new framework Back-Propagation Through Agents (BPTA) that directly accounts for both agents' own policy updates and the learning of their dependent counterparts. This is achieved by propagating the feedback through action chains. With the proposed framework, our Bidirectional Proximal Policy Optimisation (BPPO) outperforms the state-of-the-art methods. Extensive experiments on matrix games, StarCraftII v2, Multi-agent MuJoCo, and Google Research Football demonstrate the effectiveness of the proposed method.

----

## [1529] Multi-Granularity Causal Structure Learning

**Authors**: *Jiaxuan Liang, Jun Wang, Guoxian Yu, Shuyin Xia, Guoyin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29278](https://doi.org/10.1609/aaai.v38i12.29278)

**Abstract**:

Unveiling, modeling, and comprehending the causal mechanisms underpinning natural phenomena stand as fundamental endeavors across myriad scientific disciplines. Meanwhile, new knowledge emerges when discovering causal relationships from data. Existing causal learning algorithms predominantly focus on the isolated effects of variables, overlook the intricate interplay of multiple variables and their collective behavioral patterns. Furthermore, the ubiquity of high-dimensional data exacts a substantial temporal cost for causal algorithms. In this paper, we develop a novel method called MgCSL (Multi-granularity Causal Structure Learning), which first leverages sparse auto-encoder to explore coarse-graining strategies and causal abstractions from micro-variables to macro-ones. MgCSL then takes multi-granularity variables as inputs to train multilayer perceptrons and to delve the causality between variables. To enhance the efficacy on high-dimensional data, MgCSL  introduces a simplified acyclicity constraint to adeptly search the directed acyclic graph among variables. Experimental results show that MgCSL outperforms competitive baselines, and finds out explainable causal connections on fMRI datasets.

----

## [1530] Inducing Clusters Deep Kernel Gaussian Process for Longitudinal Data

**Authors**: *Junjie Liang, Weijieying Ren, Hanifi Sahar, Vasant G. Honavar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29279](https://doi.org/10.1609/aaai.v38i12.29279)

**Abstract**:

We consider the problem of predictive modeling from irregularly and sparsely sampled longitudinal data with unknown, complex correlation structures and abrupt discontinuities. To address these challenges, we introduce a novel inducing clusters longitudinal deep kernel Gaussian Process (ICDKGP). ICDKGP approximates the data generating process by a zero-mean GP with a longitudinal deep kernel that models the unknown complex correlation structure in the data and a deterministic non-zero mean function to model the abrupt discontinuities. To improve the scalability and interpretability of ICDKGP, we introduce inducing clusters corresponding to centers of clusters in the training data. We formulate the training of ICDKGP as a constrained optimization problem and derive its evidence lower bound. We introduce a novel relaxation of the resulting problem which under rather mild assumptions yields a solution with error bounded relative to the original problem. We describe the results of extensive experiments demonstrating that ICDKGP substantially outperforms the state-of-the-art longitudinal methods on data with both smoothly and non-smoothly varying outcomes.

----

## [1531] Self-Supervised Multi-Modal Knowledge Graph Contrastive Hashing for Cross-Modal Search

**Authors**: *Meiyu Liang, Junping Du, Zhengyang Liang, Yongwang Xing, Wei Huang, Zhe Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29280](https://doi.org/10.1609/aaai.v38i12.29280)

**Abstract**:

Deep cross-modal hashing technology provides an effective and efficient cross-modal unified representation learning solution for cross-modal search. However, the existing methods neglect the implicit fine-grained multimodal knowledge relations between these modalities such as when the image contains information that is not directly described in the text. To tackle this problem, we propose a novel self-supervised multi-grained multi-modal knowledge graph contrastive hashing method for cross-modal search (CMGCH). Firstly, in order to capture implicit fine-grained cross-modal semantic associations, a multi-modal knowledge graph is constructed, which represents the implicit multimodal knowledge relations between the image and text as inter-modal and intra-modal semantic associations. Secondly, a cross-modal graph contrastive attention network is proposed to reason on the multi-modal knowledge graph to sufficiently learn the implicit fine-grained inter-modal and intra-modal knowledge relations. Thirdly, a cross-modal multi-granularity contrastive embedding learning mechanism is proposed, which fuses the global coarse-grained and local fine-grained embeddings by multihead attention mechanism for inter-modal and intra-modal contrastive learning, so as to enhance the cross-modal unified representations with stronger discriminativeness and semantic consistency preserving power. With the joint training of intra-modal and inter-modal contrast, the invariant and modal-specific information of different modalities can be maintained in the final unified cross-modal unified hash space. Extensive experiments on several cross-modal benchmark datasets demonstrate that the proposed CMGCH outperforms the state-of the-art methods.

----

## [1532] DC-NAS: Divide-and-Conquer Neural Architecture Search for Multi-Modal Classification

**Authors**: *Xinyan Liang, Pinhan Fu, Qian Guo, Keyin Zheng, Yuhua Qian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29281](https://doi.org/10.1609/aaai.v38i12.29281)

**Abstract**:

Neural architecture search-based multi-modal classification (NAS-MMC) methods can individually obtain the optimal classifier for different multi-modal data sets in an automatic manner. However, most existing NAS-MMC methods are dramatically time consuming due to the requirement for training and evaluating enormous models. In this paper, we propose an efficient evolutionary-based NAS-MMC method called divide-and-conquer neural architecture search (DC-NAS). Specifically, the evolved population is first divided into k+1 sub-populations, and then k sub-populations of them evolve on k small-scale data sets respectively that are obtained by splitting the entire data set using the k-fold stratified sampling technique; the remaining one evolves on the entire data set. To solve the sub-optimal fusion model problem caused by the training strategy of partial data, two kinds of sub-populations that are trained using partial data and entire data exchange the learned knowledge via two special knowledge bases. With the two techniques mentioned above, DC-NAS achieves the training time reduction and classification performance improvement. Experimental results show that DC-NAS achieves the state-of-the-art results in term of classification performance, training efficiency and the number of model parameters than the compared NAS-MMC methods on three popular multi-modal tasks including multi-label movie genre classification, action recognition with RGB and body joints and dynamic hand gesture recognition.

----

## [1533] Value at Adversarial Risk: A Graph Defense Strategy against Cost-Aware Attacks

**Authors**: *Junlong Liao, Wenda Fu, Cong Wang, Zhongyu Wei, Jiarong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29282](https://doi.org/10.1609/aaai.v38i12.29282)

**Abstract**:

Deep learning methods on graph data have achieved remarkable efficacy across a variety of real-world applications, such as social network analysis and transaction risk detection. Nevertheless, recent studies have illuminated a concerning fact: even the most expressive Graph Neural Networks (GNNs) are vulnerable to graph adversarial attacks. While several methods have been proposed to enhance the robustness of GNN models against adversarial attacks, few have focused on a simple yet realistic approach: valuing the adversarial risks and focused safeguards at the node level. This empowers defenders to allocate heightened security level to vulnerable nodes, while lower to robust nodes. With this new perspective, we propose a novel graph defense strategy RisKeeper, such that the adversarial risk can be directly kept in the input graph. We start at valuing the adversarial risk, by introducing a cost-aware projected gradient descent attack that takes into account both cost avoidance and compliance with costs budgets. Subsequently, we present a learnable approach to ascertain the ideal security level for each individual node by solving a bi-level optimization problem. Through extensive experiments on four real-world datasets, we demonstrate that our method achieves superior performance surpassing state-of-the-art methods. Our in-depth case studies provide further insights into vulnerable and robust structural patterns, serving as inspiration for practitioners to exercise heightened vigilance.

----

## [1534] Invariant Random Forest: Tree-Based Model Solution for OOD Generalization

**Authors**: *Yufan Liao, Qi Wu, Xing Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29283](https://doi.org/10.1609/aaai.v38i12.29283)

**Abstract**:

Out-Of-Distribution (OOD) generalization is an essential topic in machine learning. However, recent research is only focusing on the corresponding methods for neural networks. This paper introduces a novel and effective solution for OOD generalization of decision tree models, named Invariant Decision Tree (IDT). IDT enforces a penalty term with regard to the unstable/varying behavior of a split across different environments during the growth of the tree. Its ensemble version, the Invariant Random Forest (IRF), is constructed. Our proposed method is motivated by a theoretical result under mild conditions, and validated by numerical tests with both synthetic and real datasets. The superior performance compared to non-OOD tree models implies that considering OOD generalization for tree models is absolutely necessary and should be given more attention.

----

## [1535] Ahpatron: A New Budgeted Online Kernel Learning Machine with Tighter Mistake Bound

**Authors**: *Yun Liao, Junfan Li, Shizhong Liao, Qinghua Hu, Jianwu Dang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29284](https://doi.org/10.1609/aaai.v38i12.29284)

**Abstract**:

In this paper, we study the mistake bound of online kernel learning on a budget. We propose a new budgeted online kernel learning model, called Ahpatron, which significantly improves the mistake bound of previous work and resolves an open problem related to upper bounds of hypothesis space constraints. We first present an aggressive variant of Perceptron, named AVP, a model without budget, which uses an active updating rule. Then we design a new budget maintenance mechanism, which removes a half of examples, and projects the removed examples onto a hypothesis space spanned by the remaining examples. Ahpatron adopts the above mechanism to approximate AVP. Theoretical analyses prove that Ahpatron has tighter mistake bounds, and experimental results show that Ahpatron outperforms the state-of-the-art algorithms on the same or a smaller budget.

----

## [1536] Spiking NeRF: Representing the Real-World Geometry by a Discontinuous Representation

**Authors**: *Zhanfeng Liao, Yan Liu, Qian Zheng, Gang Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29285](https://doi.org/10.1609/aaai.v38i12.29285)

**Abstract**:

A crucial reason for the success of existing NeRF-based methods is to build a neural density field for the geometry representation via multiple perceptron layers (MLPs).
MLPs are continuous functions, however, real geometry or density field is frequently discontinuous at the interface between the air and the surface. Such a contrary brings the problem of unfaithful geometry representation. To this end, this paper proposes spiking NeRF, which leverages spiking neurons and a hybrid Artificial Neural Network (ANN)-Spiking Neural Network (SNN) framework to build a discontinuous density field for faithful geometry representation. Specifically, we first demonstrate the reason why continuous density fields will bring inaccuracy. Then, we propose to use the spiking neurons to build a discontinuous density field. We conduct a comprehensive analysis for the problem of existing spiking neuron models and then provide the numerical relationship between the parameter of the spiking neuron and the theoretical accuracy of geometry. Based on this, we propose a bounded spiking neuron to build the discontinuous density field. Our method achieves SOTA performance. The source code and the supplementary material are available at https://github.com/liaozhanfeng/Spiking-NeRF.

----

## [1537] Mitigating Label Noise through Data Ambiguation

**Authors**: *Julian Lienen, Eyke Hüllermeier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29286](https://doi.org/10.1609/aaai.v38i12.29286)

**Abstract**:

Label noise poses an important challenge in machine learning, especially in deep learning, in which large models with high expressive power dominate the field. Models of that kind are prone to memorizing incorrect labels, thereby harming generalization performance. Many methods have been proposed to address this problem, including robust loss functions and more complex label correction approaches. Robust loss functions are appealing due to their simplicity, but typically lack flexibility, while label correction usually adds substantial complexity to the training setup. In this paper, we suggest to address the shortcomings of both methodologies by "ambiguating" the target information, adding additional, complementary candidate labels in case the learner is not sufficiently convinced of the observed training label. More precisely, we leverage the framework of so-called superset learning to construct set-valued targets based on a confidence threshold, which deliver imprecise yet more reliable beliefs about the ground-truth, effectively helping the learner to suppress the memorization effect. In an extensive empirical evaluation, our method demonstrates favorable learning behavior on synthetic and real-world noise, confirming the effectiveness in detecting and correcting erroneous training labels.

----

## [1538] Episodic Return Decomposition by Difference of Implicitly Assigned Sub-trajectory Reward

**Authors**: *Haoxin Lin, Hongqiu Wu, Jiaji Zhang, Yihao Sun, Junyin Ye, Yang Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29287](https://doi.org/10.1609/aaai.v38i12.29287)

**Abstract**:

Real-world decision-making problems are usually accompanied by delayed rewards, which affects the sample efficiency of Reinforcement Learning, especially in the extremely delayed case where the only feedback is the episodic reward obtained at the end of an episode. Episodic return decomposition is a promising way to deal with the episodic-reward setting. Several corresponding algorithms have shown remarkable effectiveness of the learned step-wise proxy rewards from return decomposition. However, these existing methods lack either attribution or representation capacity, leading to inefficient decomposition in the case of long-term episodes. In this paper, we propose a novel episodic return decomposition method called Diaster (Difference of implicitly assigned sub-trajectory reward). Diaster decomposes any episodic reward into credits of two divided sub-trajectories at any cut point, and the step-wise proxy rewards come from differences in expectation. We theoretically and empirically verify that the decomposed proxy reward function can guide the policy to be nearly optimal. Experimental results show that our method outperforms previous state-of-the-art methods in terms of both sample efficiency and performance. The code is available at https://github.com/HxLyn3/Diaster.

----

## [1539] Jointly Modeling Spatio-Temporal Features of Tactile Signals for Action Classification

**Authors**: *Jimmy Lin, Junkai Li, Jiasi Gao, Weizhi Ma, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29288](https://doi.org/10.1609/aaai.v38i12.29288)

**Abstract**:

Tactile signals collected by wearable electronics are essential in modeling and understanding human behavior. One of the main applications of tactile signals is action classification, especially in healthcare and robotics. However, existing tactile classification methods fail to capture the spatial and temporal features of tactile signals simultaneously, which results in sub-optimal performances. In this paper, we design Spatio-Temporal Aware tactility Transformer (STAT) to utilize continuous tactile signals for action classification. We propose spatial and temporal embeddings along with a new temporal pretraining task in our model, which aims to enhance the transformer in modeling the spatio-temporal features of tactile signals. Specially, the designed temporal pretraining task is to differentiate the time order of tubelet inputs to model the temporal properties explicitly. Experimental results on a public action classification dataset demonstrate that our model outperforms state-of-the-art methods in all metrics.

----

## [1540] ERL-TD: Evolutionary Reinforcement Learning Enhanced with Truncated Variance and Distillation Mutation

**Authors**: *Qiuzhen Lin, Yangfan Chen, Lijia Ma, Wei-Neng Chen, Jianqiang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29289](https://doi.org/10.1609/aaai.v38i12.29289)

**Abstract**:

Recently, an emerging research direction called Evolutionary Reinforcement Learning (ERL) has been proposed, which combines evolutionary algorithm with reinforcement learning (RL) for tackling the tasks of sequential decision making. However, the recently proposed ERL algorithms often suffer from two challenges: the inaccuracy of policy estimation caused by the overestimation bias in RL and the insufficiency of exploration caused by inefficient mutations. To alleviate these problems, we propose an Evolutionary Reinforcement Learning algorithm enhanced with Truncated variance and Distillation mutation, called ERL-TD. We utilize multiple Q-networks to evaluate state-action pairs, so that multiple networks can provide more accurate evaluations for state-action pairs, in which the variance of evaluations can be adopted to control the overestimation bias in RL. Moreover, we propose a new distillation mutation to provide a promising mutation direction, which is different from traditional mutation generating a large number of random solutions. We evaluate ERL-TD on the continuous control benchmarks from the OpenAI Gym and DeepMind Control Suite. The experiments show that ERL-TD shows excellent performance and outperforms all baseline RL algorithms on the test suites.

----

## [1541] Hypergraph Neural Architecture Search

**Authors**: *Wei Lin, Xu Peng, Zhengtao Yu, Taisong Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29290](https://doi.org/10.1609/aaai.v38i12.29290)

**Abstract**:

In recent years, Hypergraph Neural Networks (HGNNs) have achieved considerable success by manually designing architectures, which are capable of extracting effective patterns with high-order interactions from non-Euclidean data. However, such mechanism is extremely inefficient, demanding tremendous human efforts to tune diverse model parameters. In this paper, we propose a novel Hypergraph Neural Architecture Search (HyperNAS) to automatically design the optimal HGNNs. The proposed model constructs a search space suitable for hypergraphs, and derives hypergraph architectures through differentiable search strategies. A hypergraph structure-aware distance criterion is introduced as a guideline for obtaining an optimal hypergraph architecture via the leave-one-out method. Experimental results for node classification on benchmark Cora, Citeseer, Pubmed citation networks and hypergraph datasets show that HyperNAS outperforms existing HGNNs models and graph NAS methods.

----

## [1542] Scaling Few-Shot Learning for the Open World

**Authors**: *Zhipeng Lin, Wenjing Yang, Haotian Wang, Haoang Chi, Long Lan, Ji Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29291](https://doi.org/10.1609/aaai.v38i12.29291)

**Abstract**:

Few-shot learning (FSL) aims to enable learning models with the ability to automatically adapt to novel (unseen) domains in open-world scenarios. Nonetheless, there exists a significant disparity between the vast number of new concepts encountered in the open world and the restricted available scale of existing FSL works, which primarily focus on a limited number of novel classes. Such a gap hinders the practical applicability of FSL in realistic scenarios. To bridge this gap, we propose a new problem named Few-Shot Learning with Many Novel Classes (FSL-MNC) by substantially enlarging the number of novel classes, exceeding the count in the traditional FSL setup by over 500-fold. This new problem exhibits two major challenges, including the increased computation overhead during meta-training and the degraded classification performance by the large number of classes during meta-testing. To overcome these challenges, we propose a Simple Hierarchy Pipeline (SHA-Pipeline). Due to the inefficiency of traditional protocols of EML, we re-design a lightweight training strategy to reduce the overhead brought by much more novel classes. To capture discriminative semantics across numerous novel classes, we effectively reconstruct and leverage the class hierarchy information during meta-testing. Experiments show that the proposed SHA-Pipeline significantly outperforms not only the ProtoNet baseline but also the state-of-the-art alternatives across different numbers of novel classes.

----

## [1543] Towards Inductive Robustness: Distilling and Fostering Wave-Induced Resonance in Transductive GCNs against Graph Adversarial Attacks

**Authors**: *Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29292](https://doi.org/10.1609/aaai.v38i12.29292)

**Abstract**:

Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs. Appendices can be found on arXiv version: https://arxiv.org/abs/2312.08651

----

## [1544] Attention-Induced Embedding Imputation for Incomplete Multi-View Partial Multi-Label Classification

**Authors**: *Chengliang Liu, Jinlong Jia, Jie Wen, Yabo Liu, Xiaoling Luo, Chao Huang, Yong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29293](https://doi.org/10.1609/aaai.v38i12.29293)

**Abstract**:

As a combination of emerging multi-view learning methods and traditional multi-label classification tasks, multi-view multi-label classification has shown broad application prospects. The diverse semantic information contained in heterogeneous data effectively enables the further development of multi-label classification. However, the widespread incompleteness problem on multi-view features and labels greatly hinders the practical application of multi-view multi-label classification. Therefore, in this paper, we propose an attention-induced missing instances imputation technique to enhance the generalization ability of the model. Different from existing incomplete multi-view completion methods, we attempt to approximate the latent features of missing instances in embedding space according to cross-view joint attention, instead of recovering missing views in kernel space or original feature space. Accordingly, multi-view completed features are dynamically weighted by the confidence derived from joint attention in the late fusion phase. In addition, we propose a multi-view multi-label classification framework based on label-semantic feature learning, utilizing the statistical weak label correlation matrix and graph attention network to guide the learning process of label-specific features. Finally, our model is compatible with missing multi-view and partial multi-label data simultaneously and extensive experiments on five datasets confirm the advancement and effectiveness of our embedding imputation method and multi-view multi-label classification model.

----

## [1545] Learning Temporal Resolution in Spectrogram for Audio Classification

**Authors**: *Haohe Liu, Xubo Liu, Qiuqiang Kong, Wenwu Wang, Mark D. Plumbley*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29294](https://doi.org/10.1609/aaai.v38i12.29294)

**Abstract**:

The audio spectrogram is a time-frequency representation that has been widely used for audio classification. One of the key attributes of the audio spectrogram is the temporal resolution, which depends on the hop size used in the Short-Time Fourier Transform (STFT). Previous works generally assume the hop size should be a constant value (e.g., 10 ms). However, a fixed temporal resolution is not always optimal for different types of sound. The temporal resolution affects not only classification accuracy but also computational cost. This paper proposes a novel method, DiffRes, that enables differentiable temporal resolution modeling for audio classification. Given a spectrogram calculated with a fixed hop size, DiffRes merges non-essential time frames while preserving important frames. DiffRes acts as a "drop-in" module between an audio spectrogram and a classifier and can be jointly optimized with the classification task. We evaluate DiffRes on five audio classification tasks, using mel-spectrograms as the acoustic features, followed by off-the-shelf classifier backbones. Compared with previous methods using the fixed temporal resolution, the DiffRes-based method can achieve the equivalent or better classification accuracy with at least 25% computational cost reduction. We further show that DiffRes can improve classification accuracy by increasing the temporal resolution of input acoustic features, without adding to the computational cost.

----

## [1546] Language-Guided Transformer for Federated Multi-Label Classification

**Authors**: *I-Jieh Liu, Ci-Siang Lin, Fu-En Yang, Yu-Chiang Frank Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29295](https://doi.org/10.1609/aaai.v38i12.29295)

**Abstract**:

Federated Learning (FL) is an emerging paradigm that enables multiple users to collaboratively train a robust model in a privacy-preserving manner without sharing their private data. Most existing approaches of FL only consider traditional single-label image classification, ignoring the impact when transferring the task to multi-label image classification. Nevertheless, it is still challenging for FL to deal with user heterogeneity in their local data distribution in the real-world FL scenario, and this issue becomes even more severe in multi-label image classification. Inspired by the recent success of Transformers in centralized settings, we propose a novel FL framework for multi-label classification. Since partial label correlation may be observed by local clients during training, direct aggregation of locally updated models would not produce satisfactory performances. Thus, we propose a novel FL framework of Language-Guided Transformer (FedLGT) to tackle this challenging task, which aims to exploit and transfer knowledge across different clients for learning a robust global model. Through extensive experiments on various multi-label datasets (e.g., FLAIR, MS-COCO, etc.), we show that our FedLGT is able to achieve satisfactory performance and outperforms standard FL techniques under multi-label FL scenarios. Code is available at https://github.com/Jack24658735/FedLGT.

----

## [1547] UPDP: A Unified Progressive Depth Pruner for CNN and Vision Transformer

**Authors**: *Ji Liu, Dehua Tang, Yuanxian Huang, Li Zhang, Xiaocheng Zeng, Dong Li, Mingjie Lu, Jinzhang Peng, Yu Wang, Fan Jiang, Lu Tian, Ashish Sirasao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29296](https://doi.org/10.1609/aaai.v38i12.29296)

**Abstract**:

Traditional channel-wise pruning methods by reducing network channels struggle to effectively prune efficient CNN models with depth-wise convolutional layers and certain efficient modules, such as popular inverted residual blocks. Prior depth pruning methods by reducing network depths are not suitable for pruning some efficient models due to the existence of some normalization layers. Moreover, finetuning subnet with directly removing activation layers would corrupt the original model weights, hindering the pruned model from achieving high performance. To address these issues, we propose a novel depth pruning method for efficient models. Our approach proposes a novel block pruning strategy and progressive training method for the subnet. Additionally, we extend our pruning method to vision transformer models. Experimental results demonstrate that our method consistently outperforms existing depth pruning methods across various pruning configurations. We obtained three pruned ConvNeXtV1 models with our method applying on ConvNeXtV1, which surpass most SOTA efficient models with comparable inference performance. Our method also achieves state-of-the-art pruning performance on the vision transformer model.

----

## [1548] FedASMU: Efficient Asynchronous Federated Learning with Dynamic Staleness-Aware Model Update

**Authors**: *Ji Liu, Juncheng Jia, Tianshi Che, Chao Huo, Jiaxiang Ren, Yang Zhou, Huaiyu Dai, Dejing Dou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29297](https://doi.org/10.1609/aaai.v38i12.29297)

**Abstract**:

As a promising approach to deal with distributed data, Federated Learning (FL) achieves major advancements in recent years. FL enables collaborative model training by exploiting the raw data dispersed in multiple edge devices. However, the data is generally non-independent and identically distributed, i.e., statistical heterogeneity, and the edge devices significantly differ in terms of both computation and communication capacity, i.e., system heterogeneity. The statistical heterogeneity leads to severe accuracy degradation while the system heterogeneity significantly prolongs the training process. In order to address the heterogeneity issue, we propose an Asynchronous Staleness-aware Model Update FL framework, i.e., FedASMU, with two novel methods. First, we propose an asynchronous FL system model with a dynamical model aggregation method between updated local models and the global model on the server for superior accuracy and high efficiency. Then, we propose an adaptive local model adjustment method by aggregating the fresh global model with local models on devices to further improve the accuracy. Extensive experimentation with 6 models and 5 public datasets demonstrates that FedASMU significantly outperforms baseline approaches in terms of accuracy (0.60% to 23.90% higher) and efficiency (3.54% to 97.98% faster).

----

## [1549] Towards Making Learnware Specification and Market Evolvable

**Authors**: *Jian-Dong Liu, Zhi-Hao Tan, Zhi-Hua Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29298](https://doi.org/10.1609/aaai.v38i12.29298)

**Abstract**:

The learnware paradigm aims to establish a market of numerous well-performed machine learning models, enabling users to leverage existing helpful models for their tasks instead of starting from scratch. Each learnware in the market is a model submitted by its developer, associated with a specification generated with the help of learnware market, representing the model's specialty and utility and enabling it to be identified for new user tasks. As the market continuously scales up, accommodating an ever-increasing number of learnwares, the critical challenge of the learnware paradigm is to effectively and efficiently identify the most helpful learnware(s) for a new user task without accessing the user's raw data. In this paper, to achieve increasingly accurate learnware characterization and identification along with a growing number of learnwares in the market, we propose an approach called Evolvable Learnware Specification with Index (ELSI). Specifically, based on the key idea of leveraging the task information within learnware specifications, we tackle the challenge of ascertaining the capabilities of models beyond their original training tasks, thereby enabling learnware specifications and the entire market to evolve continuously. Furthermore, through organizing learnwares and constructing specification indexes, we design a practical procedure to accurately and efficiently identify helpful learnwares without examining the entire market. Theoretical analysis and extensive experiments on a learnware market prototype encompassing thousands of models and covering six real-world scenarios validate the effectiveness and efficiency of our approach.

----

## [1550] TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning

**Authors**: *Jiexi Liu, Songcan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29299](https://doi.org/10.1609/aaai.v38i12.29299)

**Abstract**:

Learning universal time series representations applicable to various types of downstream tasks is challenging but valuable in real applications. Recently, researchers have attempted to leverage the success of self-supervised contrastive learning (SSCL) in Computer Vision(CV) and Natural Language Processing(NLP) to tackle time series representation. Nevertheless, due to the special temporal characteristics, relying solely on empirical guidance from other domains may be ineffective for time series and difficult to adapt to multiple downstream tasks. To this end, we review three parts involved in SSCL including 1) designing augmentation methods for positive pairs, 2) constructing (hard) negative pairs, and 3) designing SSCL loss. For 1) and 2), we find that unsuitable positive and negative pair construction may introduce inappropriate inductive biases, which neither preserve temporal properties nor provide sufficient discriminative features. For 3), just exploring segment- or instance-level semantics information is not enough for learning universal representation. To remedy the above issues, we propose a novel self-supervised framework named TimesURL. Specifically, we first introduce a frequency-temporal-based augmentation to keep the temporal property unchanged. And then, we construct double Universums as a special kind of hard negative to guide better contrastive learning. Additionally, we introduce time reconstruction as a joint optimization objective with contrastive learning to capture both segment-level and instance-level information. As a result, TimesURL can learn high-quality universal representations and achieve state-of-the-art performance in 6 different downstream tasks, including short- and long-term forecasting, imputation, classification, anomaly detection and transfer learning.

----

## [1551] Faster Stochastic Variance Reduction Methods for Compositional MiniMax Optimization

**Authors**: *Jin Liu, Xiaokang Pan, Junwen Duan, Hongdong Li, Youqi Li, Zhe Qu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29300](https://doi.org/10.1609/aaai.v38i12.29300)

**Abstract**:

This paper delves into the realm of stochastic optimization for compositional minimax optimization—a pivotal challenge across various machine learning domains, including deep AUC and reinforcement learning policy evaluation. Despite its significance, the problem of compositional minimax optimization is still under-explored. Adding to the complexity, current methods of compositional minimax optimization are plagued by sub-optimal complexities or heavy reliance on sizable batch sizes. To respond to these constraints, this paper introduces a novel method, called Nested STOchastic Recursive Momentum (NSTORM), which can achieve the optimal sample complexity and obtain the nearly accuracy solution, matching the existing minimax methods. We also demonstrate that NSTORM can achieve the same sample complexity under the Polyak-Lojasiewicz (PL)-condition—an insightful extension of its capabilities. Yet, NSTORM encounters an issue with its requirement for low learning rates, potentially constraining its real-world applicability in machine learning. To overcome this hurdle, we present ADAptive NSTORM (ADA-NSTORM) with adaptive learning rates. We demonstrate that ADA-NSTORM can achieve the same sample complexity but the experimental results show its more effectiveness. All the proposed complexities indicate that our proposed methods can match lower bounds to existing minimax optimizations, without requiring a large batch size in each iteration. Extensive experiments support the efficiency of our proposed methods.

----

## [1552] Sketched Newton Value Iteration for Large-Scale Markov Decision Processes

**Authors**: *Jinsong Liu, Chenghan Xie, Qi Deng, Dongdong Ge, Yinyu Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29301](https://doi.org/10.1609/aaai.v38i12.29301)

**Abstract**:

Value Iteration (VI) is one of the most classic algorithms for solving Markov Decision Processes (MDPs), which lays the foundations for various more advanced reinforcement learning algorithms, such as Q-learning. VI may take a large number of iterations to converge as it is a first-order method. In this paper, we introduce the Newton Value Iteration (NVI) algorithm, which eliminates the impact of action space dimension compared to some previous second-order methods. Consequently, NVI can efficiently handle MDPs with large action spaces. Building upon NVI, we propose a novel approach called Sketched Newton Value Iteration (SNVI) to tackle MDPs with both large state and action spaces. SNVI not only inherits the stability and fast convergence advantages of second-order algorithms, but also significantly reduces computational complexity, making it highly scalable. Extensive experiments demonstrate the superiority of our algorithms over traditional VI and previously proposed second-order VI algorithms.

----

## [1553] Beyond OOD State Actions: Supported Cross-Domain Offline Reinforcement Learning

**Authors**: *Jinxin Liu, Ziqi Zhang, Zhenyu Wei, Zifeng Zhuang, Yachen Kang, Sibo Gai, Donglin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29302](https://doi.org/10.1609/aaai.v38i12.29302)

**Abstract**:

Offline reinforcement learning (RL) aims to learn a policy using only pre-collected and fixed data. Although avoiding the time-consuming online interactions in RL, it poses challenges for out-of-distribution (OOD) state actions and often suffers from data inefficiency for training. Despite many efforts being devoted to addressing OOD state actions, the latter (data inefficiency) receives little attention in offline RL. To address this, this paper proposes the cross-domain offline RL, which assumes offline data incorporate additional source-domain data from varying transition dynamics (environments), and expects it to contribute to the offline data efficiency. To do so, we identify a new challenge of OOD transition dynamics, beyond the common OOD state actions issue, when utilizing cross-domain offline data. Then, we propose our method BOSA, which employs two support-constrained objectives to address the above OOD issues. Through extensive experiments in the cross-domain offline RL setting, we demonstrate BOSA can greatly improve offline data efficiency: using only 10% of the target data, BOSA could achieve 74.4% of the SOTA offline RL performance that uses 100% of the target data. Additionally, we also show BOSA can be effortlessly plugged into model-based offline RL and noising data augmentation techniques (used for generating source-domain data), which naturally avoids the potential dynamics mismatch between target-domain data and newly generated source-domain data.

----

## [1554] OVD-Explorer: Optimism Should Not Be the Sole Pursuit of Exploration in Noisy Environments

**Authors**: *Jinyi Liu, Zhi Wang, Yan Zheng, Jianye Hao, Chenjia Bai, Junjie Ye, Zhen Wang, Haiyin Piao, Yang Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29303](https://doi.org/10.1609/aaai.v38i12.29303)

**Abstract**:

In reinforcement learning, the optimism in the face of uncertainty (OFU) is a mainstream principle for directing exploration towards less explored areas, characterized by higher uncertainty. However, in the presence of environmental stochasticity (noise), purely optimistic exploration may lead to excessive probing of high-noise areas, consequently impeding exploration efficiency. Hence, in exploring noisy environments, while optimism-driven exploration serves as a foundation, prudent attention to alleviating unnecessary over-exploration in high-noise areas becomes beneficial. In this work, we propose Optimistic Value Distribution Explorer (OVD-Explorer) to achieve a noise-aware optimistic exploration for continuous control. OVD-Explorer proposes a new measurement of the policy's exploration ability considering noise in optimistic perspectives, and leverages gradient ascent to drive exploration. Practically, OVD-Explorer can be easily integrated with continuous control RL algorithms.  Extensive evaluations on the MuJoCo and GridChaos tasks demonstrate the superiority of OVD-Explorer in achieving noise-aware optimistic exploration.

----

## [1555] Rethinking Propagation for Unsupervised Graph Domain Adaptation

**Authors**: *Meihan Liu, Zeyu Fang, Zhen Zhang, Ming Gu, Sheng Zhou, Xin Wang, Jiajun Bu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29304](https://doi.org/10.1609/aaai.v38i12.29304)

**Abstract**:

Unsupervised Graph Domain Adaptation (UGDA) aims to transfer knowledge from a labelled source graph to an unlabelled target graph in order to address the distribution shifts between graph domains. Previous works have primarily focused on aligning data from the source and target graph in the representation space learned by graph neural networks (GNNs). However, the inherent generalization capability of GNNs has been largely overlooked. Motivated by our empirical analysis, we reevaluate the role of GNNs in graph domain adaptation and uncover the pivotal role of the propagation process in GNNs for adapting to different graph domains. We provide a comprehensive theoretical analysis of UGDA and derive a generalization bound for multi-layer GNNs. By formulating GNN Lipschitz for k-layer GNNs, we show that the target risk bound can be tighter by removing propagation layers in source graph and stacking multiple propagation layers in target graph. Based on the empirical and theoretical analysis mentioned above, we propose a simple yet effective approach called A2GNN for graph domain adaptation. Through extensive experiments on real-world datasets, we demonstrate the effectiveness of our proposed A2GNN framework.

----

## [1556] ECHO-GL: Earnings Calls-Driven Heterogeneous Graph Learning for Stock Movement Prediction

**Authors**: *Mengpu Liu, Mengying Zhu, Xiuyuan Wang, Guofang Ma, Jianwei Yin, Xiaolin Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29305](https://doi.org/10.1609/aaai.v38i12.29305)

**Abstract**:

Stock movement prediction serves an important role in quantitative trading. Despite advances in existing models that enhance stock movement prediction by incorporating stock relations, these prediction models face two limitations, i.e., constructing either insufficient or static stock relations, which fail to effectively capture the complex dynamic stock relations because such complex dynamic stock relations are influenced by various factors in the ever-changing financial market. To tackle the above limitations, we propose a novel stock movement prediction model ECHO-GL based on stock relations derived from earnings calls. ECHO-GL not only constructs comprehensive stock relations by exploiting the rich semantic information in the earnings calls but also captures the movement signals between related stocks based on multimodal and heterogeneous graph learning. Moreover, ECHO-GL customizes learnable stock stochastic processes based on the post earnings announcement drift (PEAD) phenomenon to generate the temporal stock price trajectory, which can be easily plugged into any investment strategy with different time horizons to meet investment demands. Extensive experiments on two financial datasets demonstrate the effectiveness of ECHO-GL on stock price movement prediction tasks together with high prediction accuracy and trading profitability.

----

## [1557] Decentralized Scheduling with QoS Constraints: Achieving O(1) QoS Regret of Multi-Player Bandits

**Authors**: *Qingsong Liu, Zhixuan Fang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29306](https://doi.org/10.1609/aaai.v38i12.29306)

**Abstract**:

We consider a decentralized multi-player multi-armed bandit (MP-MAB) problem where  players cannot observe the actions and rewards of other players and no explicit communication or coordination between players is possible.  Prior studies mostly focus on maximizing the sum of rewards of the players over time. However, the total reward maximization learning may lead to imbalanced reward among players, leading to poor Quality of Service (QoS) for some players. In contrast, our objective is to let each player n achieve a predetermined expected average reward over time, i.e., achieving a predetermined level of QoS. We develop a novel decentralized MP-MAB algorithm to accomplish this objective by leveraging the methodology of randomized matching.  We prove that our decentralized algorithm can ensure that all players have an O(1) QoS regret. We also reveal an analog between our MP-MAB model and the online wireless queuing systems, which builds a connection between QoS in MP-MAB learning and stability in queuing theory.

----

## [1558] ASWT-SGNN: Adaptive Spectral Wavelet Transform-Based Self-Supervised Graph Neural Network

**Authors**: *Ruyue Liu, Rong Yin, Yong Liu, Weiping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29307](https://doi.org/10.1609/aaai.v38i12.29307)

**Abstract**:

Graph Comparative Learning (GCL) is a self-supervised method that combines the advantages of Graph Convolutional Networks (GCNs) and comparative learning, making it promising for learning node representations. However, the GCN encoders used in these methods rely on the Fourier transform to learn fixed graph representations, which is inherently limited by the uncertainty principle involving spatial and spectral localization trade-offs. To overcome the inflexibility of existing methods and the computationally expensive eigen-decomposition and dense matrix multiplication, this paper proposes an Adaptive Spectral Wavelet Transform-based Self-Supervised Graph Neural Network (ASWT-SGNN). The proposed method employs spectral adaptive polynomials to approximate the filter function and optimize the wavelet using contrast loss. This design enables the creation of local filters in both spectral and spatial domains, allowing flexible aggregation of neighborhood information at various scales and facilitating controlled transformation between local and global information. Compared to existing methods, the proposed approach reduces computational complexity and addresses the limitation of graph convolutional neural networks, which are constrained by graph size and lack flexible control over the neighborhood aspect. Extensive experiments on eight benchmark datasets demonstrate that ASWT-SGNN accurately approximates the filter function in high-density spectral regions, avoiding costly eigen-decomposition. Furthermore, ASWT-SGNN achieves comparable performance to state-of-the-art models in node classification tasks.

----

## [1559] Density Matters: Improved Core-Set for Active Domain Adaptive Segmentation

**Authors**: *Shizhan Liu, Zhengkai Jiang, Yuxi Li, Jinlong Peng, Yabiao Wang, Weiyao Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29308](https://doi.org/10.1609/aaai.v38i12.29308)

**Abstract**:

Active domain adaptation has emerged as a solution to balance the expensive annotation cost and the performance of trained models in semantic segmentation. However, existing works usually ignore the correlation between selected samples and its local context in feature space, which leads to inferior usage of annotation budgets. In this work, we revisit the theoretical bound of the classical Core-set method and identify that the performance is closely related to the local sample distribution around selected samples. To estimate the density of local samples efficiently, we introduce a local proxy estimator with Dynamic Masked Convolution and develop a Density-aware Greedy algorithm to optimize the bound. Extensive experiments demonstrate the superiority of our approach. Moreover, with very few labels, our scheme achieves comparable performance to the fully supervised counterpart.

----

## [1560] RPSC: Robust Pseudo-Labeling for Semantic Clustering

**Authors**: *Sihang Liu, Wenming Cao, Ruigang Fu, Kaixiang Yang, Zhiwen Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29309](https://doi.org/10.1609/aaai.v38i12.29309)

**Abstract**:

Clustering methods achieve performance improvement by jointly learning representation and cluster assignment. However, they do not consider the confidence of pseudo-labels which are not optimal as supervised information, resulting into error accumulation. To address this issue, we propose a Robust Pseudo-labeling for Semantic Clustering (RPSC) approach, which includes two stages. In the first stage (RPSC-Self), we design a semantic pseudo-labeling scheme by using the consistency of samples, i.e., samples with same semantics should be close to each other in the embedding space. To exploit robust semantic pseudo-labels for self-supervised learning, we propose a soft contrastive loss (SCL) which encourage the model to believe high-confidence sematic pseudo-labels and be less driven by low-confidence pseudo-labels. In the second stage (RPSC-Semi), we first determine the semantic pseudo-label of a sample based on the distance between itself and cluster centers, followed by screening out reliable semantic pseudo-label by exploiting the consistency. These reliable pseudo-labels are used as supervised information in the pseudo-semi-supervised learning algorithm to further improve the performance. Experimental results show that RPSC outperforms 18 competitive clustering algorithms significantly on six challenging image benchmarks. In particular, RPSC achieves an accuracy of 0.688 on ImageNet-Dogs, which is an up to 24% improvement, compared with the second-best method. Meanwhile, we conduct ablation studies to investigate effects of different augmented strategies on RPSC as well as contributions of terms in SCL to clustering performance. Besides, experimental results indicate that SCL can be easily integrated into existing clustering methods and bring performance improvement.

----

## [1561] Sample-Level Cross-View Similarity Learning for Incomplete Multi-View Clustering

**Authors**: *Suyuan Liu, Junpu Zhang, Yi Wen, Xihong Yang, Siwei Wang, Yi Zhang, En Zhu, Chang Tang, Long Zhao, Xinwang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29310](https://doi.org/10.1609/aaai.v38i12.29310)

**Abstract**:

Incomplete multi-view clustering has attracted much attention due to its ability to handle partial multi-view data. Recently, similarity-based methods have been developed to explore the complete relationship among incomplete multi-view data. Although widely applied to partial scenarios, most of the existing approaches are still faced with two limitations. Firstly, fusing similarities constructed individually on each view fails to yield a complete unified similarity. Moreover, incomplete similarity generation may lead to anomalous similarity values with column sum constraints, affecting the final clustering results. To solve the above challenging issues, we propose a Sample-level Cross-view Similarity Learning (SCSL) method for Incomplete Multi-view Clustering. Specifically, we project all samples to the same dimension and simultaneously construct a complete similarity matrix across views based on the inter-view sample relationship and the intra-view sample relationship. In addition, a simultaneously learning consensus representation ensures the validity of the projection, which further enhances the quality of the similarity matrix through the graph Laplacian regularization. Experimental results on six benchmark datasets demonstrate the ability of SCSL in processing incomplete multi-view clustering tasks. Our code is publicly available at https://github.com/Tracesource/SCSL.

----

## [1562] UFDA: Universal Federated Domain Adaptation with Practical Assumptions

**Authors**: *Xinhui Liu, Zhenghao Chen, Luping Zhou, Dong Xu, Wei Xi, Gairui Bai, Yihan Zhao, Jizhong Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29311](https://doi.org/10.1609/aaai.v38i12.29311)

**Abstract**:

Conventional Federated Domain Adaptation (FDA) approaches usually demand an abundance of assumptions, which makes them significantly less feasible for real-world situations and introduces security hazards. This paper relaxes the assumptions from previous FDAs and studies a more practical scenario named Universal Federated Domain Adaptation (UFDA). It only requires the black-box model and the label set information of each source domain, while the label sets of different source domains could be inconsistent, and the target-domain label set is totally blind. Towards a more effective solution for our newly proposed UFDA scenario, we propose a corresponding methodology called Hot-Learning with Contrastive Label Disambiguation (HCLD). It particularly tackles UFDA's domain shifts and category gaps problems by using one-hot outputs from the black-box models of various source domains. Moreover, to better distinguish the shared and unknown classes, we further present a cluster-level strategy named Mutual-Voting Decision (MVD) to extract robust consensus knowledge across peer classes from both source and target domains. Extensive experiments on three benchmark datasets demonstrate that our method achieves comparable performance for our UFDA scenario with much fewer assumptions, compared to previous methodologies with comprehensive additional assumptions.

----

## [1563] Unify Named Entity Recognition Scenarios via Contrastive Real-Time Updating Prototype

**Authors**: *Yanhe Liu, Peng Wang, Wenjun Ke, Guozheng Li, Xiye Chen, Jiteng Zhao, Ziyu Shang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29312](https://doi.org/10.1609/aaai.v38i12.29312)

**Abstract**:

Supervised named entity recognition (NER) aims to classify entity mentions into a fixed number of pre-defined types. However, in real-world scenarios, unknown entity types are continually involved. Naive fine-tuning will result in catastrophic forgetting on old entity types. Existing continual methods usually depend on knowledge distillation to alleviate forgetting, which are less effective on long task sequences. Moreover, most of them are specific to the class-incremental scenario and cannot adapt to the online scenario, which is more common in practice. In this paper, we propose a unified framework called Contrastive Real-time Updating Prototype (CRUP) that can handle different scenarios for NER. Specifically, we train a Gaussian projection model by a regularized contrastive objective. After training on each batch, we store the mean vectors of representations belong to new entity types as their prototypes. Meanwhile, we update existing prototypes belong to old types only based on representations of the current batch. The final prototypes will be used for the nearest class mean classification. In this way, CRUP can handle different scenarios through its batch-wise learning. Moreover, CRUP can alleviate forgetting in continual scenarios only with current data instead of old data. To comprehensively evaluate CRUP, we construct extensive benchmarks based on various datasets. Experimental results show that CRUP significantly outperforms baselines in continual scenarios and is also competitive in the supervised scenario.

----

## [1564] Effect Size Estimation for Duration Recommendation in Online Experiments: Leveraging Hierarchical Models and Objective Utility Approaches

**Authors**: *Yu Liu, Runzhe Wan, James McQueen, Doug Hains, Jinxiang Gu, Rui Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29313](https://doi.org/10.1609/aaai.v38i12.29313)

**Abstract**:

The selection of the assumed effect size (AES) critically determines the duration of an experiment, and hence its accuracy and efficiency. Traditionally, experimenters determine AES based on domain knowledge. However, this method becomes impractical for online experimentation services managing numerous experiments, and  a more automated approach is hence of great  demand. We initiate the study of data-driven AES selection in for online experimentation services by introducing two  solutions. The first employs a three-layer Gaussian Mixture Model considering the heteroskedasticity across experiments, and it seeks to estimate the true expected effect size among  positive experiments. The second method, grounded in utility theory, aims to determine the optimal effect size by striking a balance between the experiment's cost and the precision of decision-making. Through comparisons with baseline methods using both simulated and real data, we showcase the superior performance of the proposed approaches.

----

## [1565] Causality-Inspired Invariant Representation Learning for Text-Based Person Retrieval

**Authors**: *Yu Liu, Guihe Qin, Haipeng Chen, Zhiyong Cheng, Xun Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29314](https://doi.org/10.1609/aaai.v38i12.29314)

**Abstract**:

Text-based Person Retrieval (TPR) aims to retrieve relevant images of specific pedestrians based on the given textual query. The mainstream approaches primarily leverage pretrained deep neural networks to learn the mapping of visual and textual modalities into a common latent space for cross-modality matching. Despite their remarkable achievements, existing efforts mainly focus on learning the statistical cross-modality correlation found in training data, other than the intrinsic causal correlation. As a result, they often struggle to retrieve accurately in the face of environmental changes such as illumination, pose, and occlusion, or when encountering images with similar attributes. In this regard, we pioneer the observation of TPR from a causal view. Specifically, we assume that each image is composed of a mixture of causal factors (which are semantically consistent with text descriptions) and non-causal factors (retrieval-irrelevant, e.g., background), and only the former can lead to reliable retrieval judgments. Our goal is to extract text-critical robust visual representation (i.e., causal factors) and establish domain invariant cross-modality correlations for accurate and reliable retrieval. However, causal/non-causal factors are unobserved, so we emphasize that ideal causal factors that can simulate causal scenes should satisfy two basic principles:1） Independence: being independent of non-causal factors, and 2）Sufficiency: being causally sufficient for TPR across different environments. Building on that, we propose an Invariant Representation Learning method for TPR (IRLT), that enforces the visual representations to satisfy the two aforementioned critical properties. Extensive experiments on three datasets clearly demonstrate the advantages of IRLT over leading baselines in terms of accuracy and generalization.

----

## [1566] Detection-Based Intermediate Supervision for Visual Question Answering

**Authors**: *Yuhang Liu, Daowan Peng, Wei Wei, Yuanyuan Fu, Wenfeng Xie, Dangyang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29315](https://doi.org/10.1609/aaai.v38i12.29315)

**Abstract**:

Recently, neural module networks (NMNs) have yielded ongoing success in answering compositional visual questions, especially those involving multi-hop visual and logical reasoning. NMNs decompose the complex question into several sub-tasks using instance-modules from the reasoning paths of that question and then exploit intermediate supervisions to guide answer prediction, thereby improving inference interpretability. However, their performance may be hindered due to sketchy modeling of intermediate supervisions. For instance, (1) a prior assumption that each instance-module refers to only one grounded object yet overlooks other potentially associated grounded objects, impeding full cross-modal alignment learning; (2) IoU-based intermediate supervisions may introduce noise signals as the bounding box overlap issue might guide the model's focus towards irrelevant objects. To address these issues, a novel method, Detection-based Intermediate Supervision (DIS), is proposed, which adopts a generative detection framework to facilitate multiple grounding supervisions via sequence generation. As such, DIS offers more comprehensive and accurate intermediate supervisions, thereby boosting answer prediction performance. Furthermore, by considering intermediate results, DIS enhances the consistency in answering compositional questions and their sub-questions. Extensive experiments demonstrate the superiority of our proposed DIS, showcasing both improved accuracy and state-of-the-art reasoning consistency compared to prior approaches.

----

## [1567] Text Diffusion with Reinforced Conditioning

**Authors**: *Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, Qi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i12.29316](https://doi.org/10.1609/aaai.v38i12.29316)

**Abstract**:

Diffusion models have demonstrated exceptional capability in generating high-quality images, videos, and audio. Due to their adaptiveness in iterative refinement, they provide a strong potential for achieving better non-autoregressive sequence generation. However, existing text diffusion models still fall short in their performance due to a challenge in handling the discreteness of language. This paper thoroughly analyzes text diffusion models and uncovers two significant limitations: degradation of self-conditioning during training and misalignment between training and sampling. Motivated by our findings, we propose a novel Text Diffusion model called TReC, which mitigates the degradation with Reinforced Conditioning and the misalignment by Time-Aware Variance Scaling. Our extensive experiments demonstrate the competitiveness of TReC against autoregressive, non-autoregressive, and diffusion baselines. Moreover, qualitative analysis shows its advanced ability to fully utilize the diffusion process in refining samples.

----

## [1568] Diffusion Language-Shapelets for Semi-supervised Time-Series Classification

**Authors**: *Zhen Liu, Wenbin Pei, Disen Lan, Qianli Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29317](https://doi.org/10.1609/aaai.v38i13.29317)

**Abstract**:

Semi-supervised time-series classification could effectively alleviate the issue of lacking labeled data. However, existing approaches usually ignore model interpretability, making it difficult for humans to understand the principles behind the predictions of a model. Shapelets are a set of discriminative subsequences that show high interpretability in time series classification tasks. Shapelet learning-based methods have demonstrated promising classification performance. Unfortunately, without enough labeled data, the shapelets learned by existing methods are often poorly discriminative, and even dissimilar to any subsequence of the original time series. To address this issue, we propose the Diffusion Language-Shapelets model (DiffShape) for semi-supervised time series classification. In DiffShape, a self-supervised diffusion learning mechanism is designed, which uses real subsequences as a condition. This helps to increase the similarity between the learned shapelets and real subsequences by using a large amount of unlabeled data. Furthermore, we introduce a contrastive language-shapelets learning strategy that improves the discriminability of the learned shapelets by incorporating the natural language descriptions of the time series. Experiments have been conducted on the UCR time series archive, and the results reveal that the proposed DiffShape method achieves state-of-the-art performance and exhibits superior interpretability over baselines.

----

## [1569] Decentralized Sum-of-Nonconvex Optimization

**Authors**: *Zhuanghua Liu, Bryan Kian Hsiang Low*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29318](https://doi.org/10.1609/aaai.v38i13.29318)

**Abstract**:

We consider the optimization problem of minimizing the sum-of-nonconvex function, i.e., a convex function that is the average of nonconvex components. The existing stochastic algorithms for such a problem only focus on a single machine and the centralized scenario. In this paper, we study the sum-of-nonconvex optimization in the decentralized setting. We present a new theoretical analysis of the PMGT-SVRG algorithm for this problem and prove the linear convergence of their approach. However, the convergence rate of the PMGT-SVRG algorithm has a linear dependency on the condition number, which is undesirable for the ill-conditioned problem. To remedy this issue, we propose an accelerated stochastic decentralized first-order algorithm by incorporating the techniques of acceleration, gradient tracking, and multi-consensus mixing into the SVRG algorithm. The convergence rate of the proposed method has a square-root dependency on the condition number. The numerical experiments validate the theoretical guarantee of our proposed algorithms on both synthetic and real-world datasets.

----

## [1570] Incremental Quasi-Newton Methods with Faster Superlinear Convergence Rates

**Authors**: *Zhuanghua Liu, Luo Luo, Bryan Kian Hsiang Low*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29319](https://doi.org/10.1609/aaai.v38i13.29319)

**Abstract**:

We consider the finite-sum optimization problem, where each component function is strongly convex and has Lipschitz continuous gradient and Hessian. The recently proposed incremental quasi-Newton method is based on BFGS update and achieves a local superlinear convergence rate that is dependent on the condition number of the problem. This paper proposes a more efficient quasi-Newton method by incorporating the symmetric rank-1 update into the incremental framework, which results in the condition-number-free local superlinear convergence rate. Furthermore, we can boost our method by applying the block update on the Hessian approximation, which leads to an even faster local convergence rate. The numerical experiments show the proposed methods significantly outperform the baseline methods.

----

## [1571] DART: Dual-Modal Adaptive Online Prompting and Knowledge Retention for Test-Time Adaptation

**Authors**: *Zichen Liu, Hongbo Sun, Yuxin Peng, Jiahuan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29320](https://doi.org/10.1609/aaai.v38i13.29320)

**Abstract**:

As an up-and-coming area, CLIP-based pre-trained vision-language models can readily facilitate downstream tasks through the zero-shot or few-shot fine-tuning manners. However, they still face critical challenges in test-time generalization due to the shifts between the training and test data distributions, hindering the further improvement of the performance. To address this crucial problem, the latest works have introduced Test-Time Adaptation (TTA) techniques to CLIP which dynamically learn text prompts using only test samples. However, their limited learning capacity due to the overlook of visual modality information, and the underutilization of knowledge in previously seen test samples result in reduced performance. In this paper, we propose a novel Dual-modal Adaptive online prompting and knowledge ReTention method called DART to overcome these challenges. To increase the learning capacity, DART captures knowledge from each test sample by learning class-specific text prompts and instance-level image prompts. Additionally, to fully leverage the knowledge from previously seen test samples, DART utilizes dual-modal knowledge retention prompts to adaptively retain the acquired knowledge, thereby enhancing the predictions on subsequent test samples. Extensive experiments on various large-scale benchmarks demonstrate the effectiveness of our proposed DART against state-of-the-art methods.

----

## [1572] Backdoor Attacks via Machine Unlearning

**Authors**: *Zihao Liu, Tianhao Wang, Mengdi Huai, Chenglin Miao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29321](https://doi.org/10.1609/aaai.v38i13.29321)

**Abstract**:

As a new paradigm to erase data from a model and protect user privacy, machine unlearning has drawn significant attention. However, existing studies on machine unlearning mainly focus on its effectiveness and efficiency, neglecting the security challenges introduced by this technique. In this paper, we aim to bridge this gap and study the possibility of conducting malicious attacks leveraging machine unlearning. Specifically, we consider the backdoor attack via machine unlearning, where an attacker seeks to inject a backdoor in the unlearned model by submitting malicious unlearning requests, so that the prediction made by the unlearned model can be changed when a particular trigger presents. In our study, we propose two attack approaches. The first attack approach does not require the attacker to poison any training data of the model. The attacker can achieve the attack goal only by requesting to unlearn a small subset of his contributed training data. The second approach allows the attacker to poison a few training instances with a pre-defined trigger upfront, and then activate the attack via submitting a malicious unlearning request. Both attack approaches are proposed with the goal of maximizing the attack utility while ensuring attack stealthiness. The effectiveness of the proposed attacks is demonstrated with different machine unlearning algorithms as well as different models on different datasets.

----

## [1573] Cooperative Knowledge Distillation: A Learner Agnostic Approach

**Authors**: *Michael J. Livanos, Ian Davidson, Stephen Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29322](https://doi.org/10.1609/aaai.v38i13.29322)

**Abstract**:

Knowledge distillation is a simple but powerful way to transfer knowledge between a teacher model to a student model. Existing work suffers from at least one of the following key limitations in terms of direction and scope of transfer which restrict its use: all knowledge is transferred from teacher to student regardless of whether or not that knowledge is useful, the student is the only one learning in this exchange, and typically distillation transfers knowledge only from a single teacher to a single student. We formulate a novel form of knowledge distillation in which many models can act as both students and teachers which we call cooperative distillation. The models cooperate as follows: a model (the student) identifies specific deficiencies in it's performance and searches for another model (the teacher) who encodes learned knowledge into instructional virtual instances via counterfactual instance generation. Because different models may have different strengths and weaknesses, all models can act as either students or teachers (cooperation) when appropriate and only distill knowledge in areas specific to their strengths (focus). Since counterfactuals as a paradigm are not tied to any specific algorithm, we can use this method to distill knowledge between learners of different architectures, algorithms, and even feature spaces. We demonstrate our approach not only outperforms baselines such as transfer learning, self-supervised learning, and multiple knowledge distillation algorithms on several datasets, but it can also be used in settings where the aforementioned techniques cannot.

----

## [1574] On the Convergence of an Adaptive Momentum Method for Adversarial Attacks

**Authors**: *Sheng Long, Wei Tao, Shuohao Li, Jun Lei, Jun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29323](https://doi.org/10.1609/aaai.v38i13.29323)

**Abstract**:

Adversarial examples are commonly created by solving a constrained optimization problem, typically using sign-based methods like Fast Gradient Sign Method (FGSM). These attacks can benefit from momentum with a constant parameter, such as Momentum Iterative FGSM (MI-FGSM), to enhance black-box transferability. However, the monotonic time-varying momentum parameter is required to guarantee convergence in theory, creating a theory-practice gap. Additionally, recent work shows that sign-based methods fail to converge to the optimum in several convex settings, exacerbating the issue. To address these concerns, we propose a novel method which incorporates both an innovative adaptive momentum parameter without monotonicity assumptions  and an adaptive step-size scheme that replaces the sign operation. Furthermore, we derive a regret upper bound for general convex functions. Experiments on multiple models demonstrate the efficacy of our method in generating adversarial examples with human-imperceptible noise while achieving high attack success rates, indicating its superiority over previous adversarial example generation methods.

----

## [1575] Layer Collaboration in the Forward-Forward Algorithm

**Authors**: *Guy Lorberbom, Itai Gat, Yossi Adi, Alexander G. Schwing, Tamir Hazan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29324](https://doi.org/10.1609/aaai.v38i13.29324)

**Abstract**:

Backpropagation, which uses the chain rule, is the de-facto standard algorithm for optimizing neural networks nowadays. Recently, Hinton (2022) proposed the forward-forward algorithm, a promising alternative that optimizes neural nets layer-by-layer, without propagating gradients throughout the network. Although such an approach has several advantages over back-propagation and shows promising results, the fact that each layer is being trained independently limits the optimization process. Specifically, it prevents the network's layers from collaborating to learn complex and rich features. In this work, we study layer collaboration in the forward-forward algorithm. We show that the current version of the forward-forward algorithm is suboptimal when considering information flow in the network, resulting in a lack of collaboration between layers of the network. We propose an improved version that supports layer collaboration to better utilize the network structure, while not requiring any additional assumptions or computations. We empirically demonstrate the efficacy of the proposed version when considering both information flow and objective metrics. Additionally, we provide a theoretical motivation for the proposed method, inspired by functional entropy theory.

----

## [1576] CGS-Mask: Making Time Series Predictions Intuitive for All

**Authors**: *Feng Lu, Wei Li, Yifei Sun, Cheng Song, Yufei Ren, Albert Y. Zomaya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29325](https://doi.org/10.1609/aaai.v38i13.29325)

**Abstract**:

Artificial intelligence (AI) has immense potential in time series prediction, but most explainable tools have limited capabilities in providing a systematic understanding of important features over time. These tools typically rely on evaluating a single time point, overlook the time ordering of inputs, and neglect the time-sensitive nature of time series applications. These factors make it difficult for users, particularly those without domain knowledge, to comprehend AI model decisions and obtain meaningful explanations. We propose CGS-Mask, a post-hoc and model-agnostic cellular genetic strip mask-based saliency approach to address these challenges. CGS-Mask uses consecutive time steps as a cohesive entity to evaluate the impact of features on the final prediction, providing binary and sustained feature importance scores over time. Our algorithm optimizes the mask population iteratively to obtain the optimal mask in a reasonable time. We evaluated CGS-Mask on synthetic and real-world datasets, and it outperformed state-of-the-art methods in elucidating the importance of features over time.  According to our pilot user study via a questionnaire survey, CGS-Mask is the most effective approach in presenting easily understandable time series prediction results, enabling users to comprehend the decision-making process of AI models with ease.

----

## [1577] Improving Expressive Power of Spectral Graph Neural Networks with Eigenvalue Correction

**Authors**: *Kangkang Lu, Yanhua Yu, Hao Fei, Xuan Li, Zixuan Yang, Zirui Guo, Meiyu Liang, Mengran Yin, Tat-Seng Chua*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29326](https://doi.org/10.1609/aaai.v38i13.29326)

**Abstract**:

In recent years, spectral graph neural networks, characterized by polynomial filters, have garnered increasing attention and have achieved remarkable performance in tasks such as node classification. These models typically assume that eigenvalues for the normalized Laplacian matrix are distinct from each other, thus expecting a polynomial filter to have a high fitting ability. However, this paper empirically observes that normalized Laplacian matrices frequently possess repeated eigenvalues. Moreover, we theoretically establish that the number of distinguishable eigenvalues plays a pivotal role in determining the expressive power of spectral graph neural networks. In light of this observation, we propose an eigenvalue correction strategy that can free polynomial filters from the constraints of repeated eigenvalue inputs. Concretely, the proposed eigenvalue correction strategy enhances the uniform distribution of eigenvalues, thus mitigating repeated eigenvalues, and improving the fitting capacity and expressive power of polynomial filters. Extensive experimental results on both synthetic and real-world datasets demonstrate the superiority of our method.

----

## [1578] UniADS: Universal Architecture-Distiller Search for Distillation Gap

**Authors**: *Liming Lu, Zhenghan Chen, Xiaoyu Lu, Yihang Rao, Lujun Li, Shuchao Pang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29327](https://doi.org/10.1609/aaai.v38i13.29327)

**Abstract**:

In this paper, we present UniADS, the first Universal Architecture-Distiller Search framework for co-optimizing student architecture and distillation policies. Teacher-student distillation gap limits the distillation gains. Previous approaches seek to discover the ideal student architecture while ignoring distillation settings. In UniADS, we construct a comprehensive search space encompassing an architectural search for student models, knowledge transformations in distillation strategies, distance functions, loss weights, and other vital settings. To efficiently explore the search space, we utilize the NSGA-II genetic algorithm for better crossover and mutation configurations and employ the Successive Halving algorithm for search space pruning, resulting in improved search efficiency and promising results. Extensive experiments are performed on different teacher-student pairs using CIFAR-100 and ImageNet datasets. The experimental results consistently demonstrate the superiority of our method over existing approaches. Furthermore, we provide a detailed analysis of the search results, examining the impact of each variable and extracting valuable insights and practical guidance for distillation design and implementation.

----

## [1579] NodeMixup: Tackling Under-Reaching for Graph Neural Networks

**Authors**: *Weigang Lu, Ziyu Guan, Wei Zhao, Yaming Yang, Long Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29328](https://doi.org/10.1609/aaai.v38i13.29328)

**Abstract**:

Graph Neural Networks (GNNs) have become mainstream methods for solving the semi-supervised node classification problem. However, due to the uneven location distribution of labeled nodes in the graph, labeled nodes are only accessible to a small portion of unlabeled nodes, leading to the under-reaching issue. In this study, we firstly reveal under-reaching by conducting an empirical investigation on various well-known graphs. Then, we demonstrate that under-reaching results in unsatisfactory distribution alignment between labeled and unlabeled nodes through systematic experimental analysis, significantly degrading GNNs' performance. To tackle under-reaching for GNNs, we propose an architecture-agnostic method dubbed NodeMixup. The fundamental idea is to (1) increase the reachability of labeled nodes by labeled-unlabeled pairs mixup, (2) leverage graph structures via fusing the neighbor connections of intra-class node pairs to improve performance gains of mixup, and (3) use neighbor label distribution similarity incorporating node degrees to determine sampling weights for node mixup. Extensive experiments demonstrate the efficacy of NodeMixup in assisting GNNs in handling under-reaching. The source code is available at https://github.com/WeigangLu/NodeMixup.

----

## [1580] Federated Learning with Extremely Noisy Clients via Negative Distillation

**Authors**: *Yang Lu, Lin Chen, Yonggang Zhang, Yiliang Zhang, Bo Han, Yiu-ming Cheung, Hanzi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29329](https://doi.org/10.1609/aaai.v38i13.29329)

**Abstract**:

Federated learning (FL) has shown remarkable success in cooperatively training deep models, while typically struggling with noisy labels. Advanced works propose to tackle label noise by a re-weighting strategy with a strong assumption, i.e., mild label noise. However, it may be violated in many real-world FL scenarios because of highly contaminated clients, resulting in extreme noise ratios, e.g., >90%. To tackle extremely noisy clients, we study the robustness of the re-weighting strategy, showing a pessimistic conclusion: minimizing the weight of clients trained over noisy data outperforms re-weighting strategies. To leverage models trained on noisy clients, we propose a novel approach, called negative distillation (FedNed). FedNed first identifies noisy clients and employs rather than discards the noisy clients in a knowledge distillation manner. In particular, clients identified as noisy ones are required to train models using noisy labels and pseudo-labels obtained by global models. The model trained on noisy labels serves as a ‘bad teacher’ in knowledge distillation, aiming to decrease the risk of providing incorrect information. Meanwhile, the model trained on pseudo-labels is involved in model aggregation if not identified as a noisy client. Consequently, through pseudo-labeling, FedNed gradually increases the trustworthiness of models trained on noisy clients, while leveraging all clients for model aggregation through negative distillation. To verify the efficacy of FedNed, we conduct extensive experiments under various settings, demonstrating that FedNed can consistently outperform baselines and achieve state-of-the-art performance.

----

## [1581] Decoupled Contrastive Multi-View Clustering with High-Order Random Walks

**Authors**: *Yiding Lu, Yijie Lin, Mouxing Yang, Dezhong Peng, Peng Hu, Xi Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29330](https://doi.org/10.1609/aaai.v38i13.29330)

**Abstract**:

In recent, some robust contrastive multi-view clustering (MvC) methods have been proposed, which construct data pairs from neighborhoods to alleviate the false negative issue, i.e., some intra-cluster samples are wrongly treated as negative pairs. Although promising performance has been achieved by these methods, the false negative issue is still far from addressed and the false positive issue emerges because all in- and out-of-neighborhood samples are simply treated as positive and negative, respectively. To address the issues, we propose a novel robust method, dubbed decoupled contrastive multi-view clustering with high-order random walks (DIVIDE). In brief, DIVIDE leverages random walks to progressively identify data pairs in a global instead of local manner. As a result, DIVIDE could identify in-neighborhood negatives and out-of-neighborhood positives. Moreover, DIVIDE embraces a novel MvC architecture to perform inter- and intra-view contrastive learning in different embedding spaces, thus boosting clustering performance and embracing the robustness against missing views. To verify the efficacy of DIVIDE, we carry out extensive experiments on four benchmark datasets comparing with nine state-of-the-art MvC methods in both complete and incomplete MvC settings. The code is released on https://github.com/XLearning-SCU/2024-AAAI-DIVIDE.

----

## [1582] Are You Concerned about Limited Function Evaluations: Data-Augmented Pareto Set Learning for Expensive Multi-Objective Optimization

**Authors**: *Yongfan Lu, Bingdong Li, Aimin Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29331](https://doi.org/10.1609/aaai.v38i13.29331)

**Abstract**:

Optimizing multiple conflicting black-box objectives simultaneously is a prevalent occurrence in many real-world applications, such as neural architecture search, and machine learning. These problems are known as expensive multi-objective optimization problems (EMOPs) when the function evaluations are computationally or financially costly. Multi-objective Bayesian optimization (MOBO) offers an efficient approach to discovering a set of Pareto optimal solutions. However, the data deficiency issue caused by limited function evaluations has posed a great challenge to current optimization methods. Moreover, most current methods tend to prioritize the quality of candidate solutions, while ignoring the quantity of promising samples. In order to tackle these issues, our paper proposes a novel multi-objective Bayesian optimization algorithm with a data augmentation strategy that provides ample high-quality samples for Pareto set learning (PSL). Specifically, it utilizes Generative Adversarial Networks (GANs) to enrich data and a dominance prediction model to screen out high-quality samples, mitigating the predicament of limited function evaluations in EMOPs. Additionally, we adopt the regularity model to expensive multi-objective Bayesian optimization for PSL. Experimental results on both synthetic and real-world problems demonstrate that our algorithm outperforms several state-of-the-art and classical algorithms.

----

## [1583] Autoregressive Omni-Aware Outpainting for Open-Vocabulary 360-Degree Image Generation

**Authors**: *Zhuqiang Lu, Kun Hu, Chaoyue Wang, Lei Bai, Zhiyong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29332](https://doi.org/10.1609/aaai.v38i13.29332)

**Abstract**:

A 360-degree (omni-directional) image provides an all-encompassing spherical view of a scene. Recently, there has been an increasing interest in synthesising 360-degree images from conventional narrow field of view (NFoV) images captured by digital cameras and smartphones, for providing immersive experiences in various scenarios such as virtual reality. Yet, existing methods typically fall short in synthesizing intricate visual details or ensure the generated images align consistently with user-provided prompts. In this study, autoregressive omni-aware generative network (AOG-Net) is proposed for 360-degree image generation by outpainting an incomplete 360-degree image progressively with NFoV and text guidances joinly or individually. This autoregressive scheme not only allows for deriving finer-grained and text-consistent patterns by dynamically generating and adjusting the process but also offers users greater flexibility to edit their conditions throughout the generation process. A global-local conditioning mechanism is devised to comprehensively formulate the outpainting guidance in each autoregressive step. Text guidances, omni-visual cues, NFoV inputs and omni-geometry are encoded and further formulated with cross-attention based transformers into a global stream and a local stream into a conditioned generative backbone model. As AOG-Net is compatible to leverage large-scale models for the conditional encoder and the generative prior, it enables the generation to use extensive open-vocabulary text guidances. Comprehensive experiments on two commonly used 360-degree image datasets for both indoor and outdoor settings demonstrate the state-of-the-art performance of our proposed method. Our code is available at https://github.com/zhuqiangLu/AOG-NET-360.

----

## [1584] Leveraging Diffusion Perturbations for Measuring Fairness in Computer Vision

**Authors**: *Nicholas Lui, Bryan Chia, William Berrios, Candace Ross, Douwe Kiela*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29333](https://doi.org/10.1609/aaai.v38i13.29333)

**Abstract**:

Computer vision models have been known to encode harmful biases, leading to the potentially unfair treatment of historically marginalized groups, such as people of color. However, there remains a lack of datasets balanced along demographic traits that can be used to evaluate the downstream fairness of these models. In this work, we demonstrate that diffusion models can be leveraged to create such a dataset. We first use a diffusion model to generate a large set of images depicting various occupations. Subsequently, each image is edited using inpainting to generate multiple variants, where each variant refers to a different perceived race. Using this dataset, we benchmark several vision-language models on a multi-class occupation classification task. We find that images generated with non-Caucasian labels have a significantly higher occupation misclassification rate than images generated with Caucasian labels, and that several misclassifications are suggestive of racial biases. We measure a model’s downstream fairness by computing the standard deviation in the probability of predicting the true occupation label across the different identity groups. Using this fairness metric, we find significant disparities between the evaluated vision-and-language models. We hope that our work demonstrates the potential value of diffusion methods for fairness evaluations.

----

## [1585] Three Heads Are Better than One: Complementary Experts for Long-Tailed Semi-supervised Learning

**Authors**: *Chengcheng Ma, Ismail Elezi, Jiankang Deng, Weiming Dong, Changsheng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29334](https://doi.org/10.1609/aaai.v38i13.29334)

**Abstract**:

We address the challenging problem of Long-Tailed Semi-Supervised Learning (LTSSL) where labeled data exhibit imbalanced class distribution and unlabeled data follow an unknown distribution. Unlike in balanced SSL, the generated pseudo-labels are skewed towards head classes, intensifying the training bias. Such a phenomenon is even amplified as more unlabeled data will be mislabeled as head classes when the class distribution of labeled and unlabeled datasets are mismatched. To solve this problem, we propose a novel method named ComPlementary Experts (CPE). Specifically, we train multiple experts to model various class distributions, each of them yielding high-quality pseudo-labels within one form of class distribution. Besides, we introduce Classwise Batch Normalization for CPE to avoid performance degradation caused by feature distribution mismatch between head and non-head classes. CPE achieves state-of-the-art performances on CIFAR-10-LT, CIFAR-100-LT, and STL-10-LT dataset benchmarks. For instance, on CIFAR-10-LT, CPE improves test accuracy by over >2.22% compared to baselines. Code is available at https://github.com/machengcheng2016/CPE-LTSSL.

----

## [1586] Discerning Temporal Difference Learning

**Authors**: *Jianfei Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29335](https://doi.org/10.1609/aaai.v38i13.29335)

**Abstract**:

Temporal difference learning (TD) is a foundational concept in reinforcement learning (RL), aimed at efficiently assessing a policy's value function. TD(λ), a potent variant, incorporates a memory trace to distribute the prediction error into the historical context. However, this approach often neglects the significance of historical states and the relative importance of propagating the TD error, influenced by challenges such as visitation imbalance or outcome noise. To address this, we propose a novel TD algorithm named discerning TD learning (DTD), which allows flexible emphasis functions—predetermined or adapted during training—to allocate efforts effectively across states. We establish the convergence properties of our method within a specific class of emphasis functions and showcase its promising potential for adaptation to deep RL contexts. Empirical results underscore that employing a judicious emphasis function not only improves value estimation but also expedites learning across diverse scenarios.

----

## [1587] One-Step Forward and Backtrack: Overcoming Zig-Zagging in Loss-Aware Quantization Training

**Authors**: *Lianbo Ma, Yuee Zhou, Jianlun Ma, Guo Yu, Qing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29336](https://doi.org/10.1609/aaai.v38i13.29336)

**Abstract**:

Weight quantization is an effective technique to compress deep neural networks for their deployment on edge devices with limited resources. Traditional loss-aware quantization methods commonly use the quantized gradient to replace the full-precision gradient. However, we discover that the gradient error will lead to an unexpected zig-zagging-like issue in the gradient descent learning procedures, where the gradient directions rapidly oscillate or zig-zag, and such issue seriously slows down the model convergence. Accordingly, this paper proposes a one-step forward and backtrack way for loss-aware quantization to get more accurate and stable gradient direction to defy this issue. During the gradient descent learning, a one-step forward search is designed to find the trial gradient of the next-step, which is adopted to adjust the gradient of current step towards the direction of fast convergence. After that, we backtrack the current step to update the full-precision and quantized weights through the current-step gradient and the trial gradient. A series of theoretical analysis and experiments on benchmark deep models have demonstrated the effectiveness and competitiveness of the proposed method, and our method especially outperforms others on the convergence performance.

----

## [1588] U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting

**Authors**: *Xiang Ma, Xuemei Li, Lexin Fang, Tianlong Zhao, Caiming Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29337](https://doi.org/10.1609/aaai.v38i13.29337)

**Abstract**:

Time series forecasting is a crucial task in various domains. Caused by factors such as trends, seasonality, or irregular fluctuations, time series often exhibits non-stationary. It obstructs stable feature propagation through deep layers, disrupts feature distributions, and complicates learning data distribution changes. As a result, many existing models struggle to capture the underlying patterns, leading to degraded forecasting performance. In this study, we tackle the challenge of non-stationarity in time series forecasting with our proposed framework called U-Mixer. By combining Unet and Mixer, U-Mixer effectively captures local temporal dependencies between different patches and channels separately to avoid the influence of distribution variations among channels, and merge low- and high-levels features to obtain comprehensive data representations. The key contribution is a novel stationarity correction method, explicitly restoring data distribution by constraining the difference in stationarity between the data before and after model processing to restore the non-stationarity information, while ensuring the temporal dependencies are preserved. Through extensive experiments on various real-world time series datasets, U-Mixer demonstrates its effectiveness and robustness, and achieves 14.5% and 7.7% improvements over state-of-the-art (SOTA) methods.

----

## [1589] Transformer-Based Video-Structure Multi-Instance Learning for Whole Slide Image Classification

**Authors**: *Yingfan Ma, Xiaoyuan Luo, Kexue Fu, Manning Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29338](https://doi.org/10.1609/aaai.v38i13.29338)

**Abstract**:

Pathological images play a vital role in clinical cancer diagnosis. Computer-aided diagnosis utilized on digital Whole Slide Images (WSIs) has been widely studied. The major challenge of using deep learning models for WSI analysis is the huge size of WSI images and existing methods struggle between end-to-end learning and proper modeling of contextual information. Most state-of-the-art methods utilize a two-stage strategy, in which they use a pre-trained model to extract features of small patches cut from a WSI and then input these features into a classification model. These methods can not perform end-to-end learning and consider contextual information at the same time. To solve this problem, we propose a framework that models a WSI as a pathologist's observing video and utilizes Transformer to process video clips with a divide-and-conquer strategy, which helps achieve both context-awareness and end-to-end learning. Extensive experiments on three public WSI datasets show that our proposed method outperforms existing SOTA methods in both WSI classification and positive region detection.

----

## [1590] PPIDSG: A Privacy-Preserving Image Distribution Sharing Scheme with GAN in Federated Learning

**Authors**: *Yuting Ma, Yuanzhi Yao, Xiaohua Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29339](https://doi.org/10.1609/aaai.v38i13.29339)

**Abstract**:

Federated learning (FL) has attracted growing attention since it allows for privacy-preserving collaborative training on decentralized clients without explicitly uploading sensitive data to the central server. However, recent works have revealed that it still has the risk of exposing private data to adversaries. In this paper, we conduct reconstruction attacks and enhance inference attacks on various datasets to better understand that sharing trained classification model parameters to a central server is the main problem of privacy leakage in FL. To tackle this problem, a privacy-preserving image distribution sharing scheme with GAN (PPIDSG) is proposed, which consists of a block scrambling-based encryption algorithm, an image distribution sharing method, and local classification training. Specifically, our method can capture the distribution of a target image domain which is transformed by the block encryption algorithm, and upload generator parameters to avoid classifier sharing with negligible influence on model performance. Furthermore, we apply a feature extractor to motivate model utility and train it separately from the classifier. The extensive experimental results and security analyses demonstrate the superiority of our proposed scheme compared to other state-of-the-art defense methods. The code is available at https://github.com/ytingma/PPIDSG.

----

## [1591] Hard Regularization to Prevent Deep Online Clustering Collapse without Data Augmentation

**Authors**: *Louis Mahon, Thomas Lukasiewicz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29340](https://doi.org/10.1609/aaai.v38i13.29340)

**Abstract**:

Online deep clustering refers to the joint use of a feature extraction network and a clustering model to assign cluster labels to each new data point or batch as it is processed. While faster and more versatile than offline methods, online clustering can easily reach the collapsed solution where the encoder maps all inputs to the same point and all are put into a single cluster. Successful existing models have employed various techniques to avoid this problem, most of which require data augmentation or which aim to make the average soft assignment across the dataset the same for each cluster. We propose a method that does not require data augmentation, and that, differently from existing methods, regularizes the hard assignments. Using a Bayesian framework, we derive an intuitive optimization objective that can be straightforwardly included in the training of the encoder network. Tested on four image datasets, it consistently avoids collapse more robustly than other methods and leads to more accurate clustering. We also conduct further experiments and analyses justifying our choice to regularize the hard cluster assignments. Code is available at https://github.com/Lou1sM/online_hard_clustering.

----

## [1592] Simple Weak Coresets for Non-decomposable Classification Measures

**Authors**: *Jayesh Malaviya, Anirban Dasgupta, Rachit Chhaya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29341](https://doi.org/10.1609/aaai.v38i13.29341)

**Abstract**:

While coresets have been growing in terms of their application, barring few exceptions, they have mostly been limited to unsupervised settings. We consider supervised classification problems, and non-decomposable evaluation measures in such settings. We show that stratified uniform sampling based coresets have excellent empirical performance that are backed by theoretical guarantees too. We focus on the F1 score and Matthews Correlation Coefficient, two widely used non-decomposable objective functions that are nontrivial to optimize for and show that uniform coresets attain a lower bound for coreset size, and have good empirical performance, comparable with ``smarter'' coreset construction strategies.

----

## [1593] One Self-Configurable Model to Solve Many Abstract Visual Reasoning Problems

**Authors**: *Mikolaj Malkinski, Jacek Mandziuk*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29342](https://doi.org/10.1609/aaai.v38i13.29342)

**Abstract**:

Abstract Visual Reasoning (AVR) comprises a wide selection of various problems similar to those used in human IQ tests. Recent years have brought dynamic progress in solving particular AVR tasks, however, in the contemporary literature AVR problems are largely dealt with in isolation, leading to highly specialized task-specific methods. With the aim of developing universal learning systems in the AVR domain, we propose the unified model for solving Single-Choice Abstract visual Reasoning tasks (SCAR), capable of solving various single-choice AVR tasks, without making any a priori assumptions about the task structure, in particular the number and location of panels. The proposed model relies on a novel Structure-Aware dynamic Layer (SAL), which adapts its weights to the structure of the considered AVR problem. Experiments conducted on Raven's Progressive Matrices, Visual Analogy Problems, and Odd One Out problems show that SCAR (SAL-based models, in general) effectively solves diverse AVR tasks, and its performance is on par with the state-of-the-art task-specific baselines. What is more, SCAR demonstrates effective knowledge reuse in multi-task and transfer learning settings. To our knowledge, this work is the first successful attempt to construct a general single-choice AVR solver relying on self-configurable architecture and unified solving method. With this work we aim to stimulate and foster progress on task-independent research paths in the AVR domain, with the long-term goal of development of a general AVR solver.

----

## [1594] Permutation-Based Hypothesis Testing for Neural Networks

**Authors**: *Francesca Mandel, Ian Barnett*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29343](https://doi.org/10.1609/aaai.v38i13.29343)

**Abstract**:

Neural networks are powerful predictive models, but they provide little insight into the nature of relationships between predictors and outcomes. Although numerous methods have been proposed to quantify the relative contributions of input features, statistical inference and hypothesis testing of feature associations remain largely unexplored. We propose a permutation-based approach to testing that uses the partial derivatives of the network output with respect to specific inputs to assess both the significance of input features and whether significant features are linearly associated with the network output. These tests, which can be flexibly applied to a variety of network architectures, enhance the explanatory power of neural networks, and combined with powerful predictive capability, extend the applicability of these models.

----

## [1595] Online Markov Decision Processes Configuration with Continuous Decision Space

**Authors**: *Davide Maran, Pierriccardo Olivieri, Francesco Emanuele Stradi, Giuseppe Urso, Nicola Gatti, Marcello Restelli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29344](https://doi.org/10.1609/aaai.v38i13.29344)

**Abstract**:

In this paper, we investigate the optimal online configuration of episodic Markov decision processes when the space of the possible configurations is continuous. Specifically, we study the interaction between a learner (referred to as the configurator) and an agent with a fixed, unknown policy, when the learner aims to minimize her losses by choosing transition functions in online fashion. The losses may be unrelated to the agent's rewards. This problem applies to many real-world scenarios where the learner seeks to manipulate the Markov decision process to her advantage. We study both deterministic and stochastic settings, where the losses are either fixed or sampled from an unknown probability distribution. We design two algorithms whose peculiarity is to rely on occupancy measures to explore with optimism the continuous space of transition functions, achieving constant regret in  deterministic settings and sublinear regret in stochastic settings, respectively. Moreover, we prove that the regret bound is tight with respect to any constant factor in deterministic settings. Finally, we compare the empiric performance of our algorithms with a baseline in synthetic experiments.

----

## [1596] GradTree: Learning Axis-Aligned Decision Trees with Gradient Descent

**Authors**: *Sascha Marton, Stefan Lüdtke, Christian Bartelt, Heiner Stuckenschmidt*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29345](https://doi.org/10.1609/aaai.v38i13.29345)

**Abstract**:

Decision Trees (DTs) are commonly used for many machine learning tasks due to their high degree of interpretability. However, learning a DT from data is a difficult optimization problem, as it is non-convex and non-differentiable. Therefore, common approaches learn DTs using a greedy growth algorithm that minimizes the impurity locally at each internal node. Unfortunately, this greedy procedure can lead to inaccurate trees. In this paper, we present a novel approach for learning hard, axis-aligned DTs with gradient descent. The proposed method uses backpropagation with a straight-through operator on a dense DT representation, to jointly optimize all tree parameters. Our approach outperforms existing methods on binary classification benchmarks and achieves competitive results for multi-class tasks. The implementation is available under: https://github.com/s-marton/GradTree

----

## [1597] Optimal Attack and Defense for Reinforcement Learning

**Authors**: *Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29346](https://doi.org/10.1609/aaai.v38i13.29346)

**Abstract**:

To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.

----

## [1598] QCS-SGM+: Improved Quantized Compressed Sensing with Score-Based Generative Models

**Authors**: *Xiangming Meng, Yoshiyuki Kabashima*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29347](https://doi.org/10.1609/aaai.v38i13.29347)

**Abstract**:

In practical compressed sensing (CS), the obtained measurements typically necessitate quantization to a limited number of bits prior to transmission or storage. This nonlinear quantization process poses significant recovery challenges, particularly with extreme coarse quantization such as 1-bit. Recently, an efficient algorithm called QCS-SGM was proposed for quantized CS (QCS) which utilizes score-based generative models (SGM) as an implicit prior. Due to the adeptness of SGM in capturing the intricate structures of natural signals, QCS-SGM substantially outperforms previous QCS methods. However, QCS-SGM is constrained to (approximately) row-orthogonal sensing matrices as the computation of the likelihood score becomes intractable otherwise. To address this limitation, we introduce an advanced variant of QCS-SGM, termed QCS-SGM+, capable of handling general matrices effectively. The key idea is a Bayesian inference perspective on the likelihood score computation, wherein expectation propagation is employed for its approximate computation.   Extensive experiments are conducted, demonstrating the substantial superiority of QCS-SGM+ over QCS-SGM for general sensing matrices beyond mere row-orthogonality.

----

## [1599] Learning Representations on the Unit Sphere: Investigating Angular Gaussian and Von Mises-Fisher Distributions for Online Continual Learning

**Authors**: *Nicolas Michel, Giovanni Chierchia, Romain Negrel, Jean-François Bercher*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i13.29348](https://doi.org/10.1609/aaai.v38i13.29348)

**Abstract**:

We use the maximum a posteriori estimation principle for learning representations distributed on the unit sphere. We propose to use the angular Gaussian distribution, which corresponds to a Gaussian projected on the unit-sphere and derive the associated loss function. We also consider the von Mises-Fisher distribution, which is the conditional of a Gaussian in the unit-sphere. The learned representations are pushed toward fixed directions, which are the prior means of the Gaussians; allowing for a learning strategy that is resilient to data drift. This makes it suitable for online continual learning, which is the problem of training neural networks on a continuous data stream, where multiple classification tasks are presented sequentially so that data from past tasks are no longer accessible, and data from the current task can be seen only once. To address this challenging scenario, we propose a memory-based representation learning technique equipped with our new loss functions. Our approach does not require negative data or knowledge of task boundaries and performs well with smaller batch sizes while being computationally efficient. We demonstrate with extensive experiments that the proposed method outperforms the current state-of-the-art methods on both standard evaluation scenarios and realistic scenarios with blurry task boundaries. For reproducibility, we use the same training pipeline for every compared method and share the code at https://github.com/Nicolas1203/ocl-fd.

----



[Go to the previous page](AAAI-2024-list07.md)

[Go to the next page](AAAI-2024-list09.md)

[Go to the catalog section](README.md)