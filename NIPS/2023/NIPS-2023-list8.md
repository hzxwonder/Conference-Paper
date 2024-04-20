## [0] To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning

**Authors**: *Ildus Sadrtdinov, Dmitrii Pozdeev, Dmitry P. Vetrov, Ekaterina Lobacheva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/336572db3e99930814d6b328d4220cb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/336572db3e99930814d6b328d4220cb6-Abstract-Conference.html)

**Abstract**:

Transfer learning and ensembling are two popular techniques for improving the performance and robustness of neural networks. Due to the high cost of pre-training, ensembles of models fine-tuned from a single pre-trained checkpoint are often used in practice. Such models end up in the same basin of the loss landscape, which we call the pre-train basin, and thus have limited diversity. In this work, we show that ensembles trained from a single pre-trained checkpoint may be improved by better exploring the pre-train basin, however, leaving the basin results in losing the benefits of transfer learning and in degradation of the ensemble quality. Based on the analysis of existing exploration methods, we propose a more effective modification of the Snapshot Ensembles (SSE) for transfer learning setup, StarSSE, which results in stronger ensembles and uniform model soups.

----

## [0] Statistical Limits of Adaptive Linear Models: Low-Dimensional Estimation and Inference

**Authors**: *Licong Lin, Mufang Ying, Suvrojit Ghosh, Koulik Khamaru, Cun-Hui Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3368e8f592b0d46ed85def795fd5168f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3368e8f592b0d46ed85def795fd5168f-Abstract-Conference.html)

**Abstract**:

Estimation and inference in statistics pose significant challenges when data are collected adaptively. Even in linear models, the Ordinary Least Squares (OLS) estimator may fail to exhibit asymptotic normality for single coordinate estimation and have inflated error. This issue is highlighted by a recent minimax lower bound, which shows that the error of estimating a  single coordinate can be enlarged by a multiple of $\sqrt{d}$ when data are allowed to be arbitrarily adaptive, compared with the case when they are i.i.d. Our work explores this striking difference in estimation performance between utilizing i.i.d. and adaptive data. We investigate how the degree of adaptivity in data collection impacts the performance of estimating a low-dimensional parameter component in high-dimensional linear models. We identify conditions on the data collection mechanism under which the estimation error for a low-dimensional parameter component matches its counterpart in the i.i.d. setting, up to a factor that depends on the degree of adaptivity. We show that OLS or OLS on centered data can achieve this matching error. In addition, we propose a novel estimator for single coordinate inference via solving a Two-stage Adaptive Linear Estimating equation (TALE). Under a weaker form of adaptivity in data collection, we establish an asymptotic normality property of the proposed estimator.

----

## [0] Future-Dependent Value-Based Off-Policy Evaluation in POMDPs

**Authors**: *Masatoshi Uehara, Haruka Kiyohara, Andrew Bennett, Victor Chernozhukov, Nan Jiang, Nathan Kallus, Chengchun Shi, Wen Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3380e8116452e0efbf36f35d95e88c94-Abstract-Conference.html)

**Abstract**:

We study off-policy evaluation (OPE) for partially observable MDPs (POMDPs) with general function approximation. Existing methods such as sequential importance sampling estimators and fitted-Q evaluation suffer from the curse of horizon in POMDPs. To circumvent this problem, we develop a novel model-free OPE method by introducing future-dependent value functions that take future proxies as inputs. Future-dependent value functions play similar roles as classical value functions in fully-observable MDPs. We derive a new off-policy Bellman equation for future-dependent value functions as conditional moment equations that use history proxies as instrumental variables. We further propose a minimax learning method to learn future-dependent value functions using the new Bellman equation. We obtain the PAC result, which implies our OPE estimator is close to the true policy value as long as futures and histories contain sufficient information about latent states, and the Bellman completeness. Our code is available at https://github.com/aiueola/neurips2023-future-dependent-ope

----

## [0] Visual Explanations of Image-Text Representations via Multi-Modal Information Bottleneck Attribution

**Authors**: *Ying Wang, Tim G. J. Rudner, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/339caf45a6fa281cae8adc6465343464-Abstract-Conference.html)

**Abstract**:

Vision-language pretrained models have seen remarkable success, but their application to safety-critical settings is limited by their lack of interpretability. To improve the interpretability of vision-language models such as CLIP, we propose a multi-modal information bottleneck (M2IB) approach that learns latent representations that compress irrelevant information while preserving relevant visual and textual features. We demonstrate how M2IB can be applied to attribution analysis of vision-language pretrained models, increasing attribution accuracy and improving the interpretability of such models when applied to safety-critical domains such as healthcare. Crucially, unlike commonly used unimodal attribution methods, M2IB does not require ground truth labels, making it possible to audit representations of vision-language pretrained models when multiple modalities but no ground-truth data is available. Using CLIP as an example, we demonstrate the effectiveness of M2IB attribution and show that it outperforms gradient-based, perturbation-based, and attention-based attribution methods both qualitatively and quantitatively.

----

## [0] PlanE: Representation Learning over Planar Graphs

**Authors**: *Radoslav Dimitrov, Zeyang Zhao, Ralph Abboud, Ismail Ilkan Ceylan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html)

**Abstract**:

Graph neural networks are prominent models for representation learning over graphs, where the idea is to iteratively compute representations of nodes of an input graph through a series of transformations in such a way that the learned graph function is isomorphism-invariant on graphs, which makes the learned representations graph invariants. On the other hand, it is well-known that graph invariants learned by these class of models are incomplete: there are pairs of non-isomorphic graphs which cannot be distinguished by standard graph neural networks. This is unsurprising given the computational difficulty of graph isomorphism testing on general graphs, but the situation begs to differ for special graph classes, for which efficient graph isomorphism testing algorithms are known, such as planar graphs. The goal of this work is to design architectures for efficiently learning complete invariants of planar graphs. Inspired by the classical planar graph isomorphism algorithm of Hopcroft and Tarjan, we propose PlanE as a framework for planar representation learning. PlanE includes architectures which can learn complete invariants over planar graphs while remaining practically scalable. We empirically validate the strong performance of the resulting model architectures on well-known planar graph benchmarks, achieving multiple state-of-the-art results.

----

## [0] Optimal Regret Is Achievable with Bounded Approximate Inference Error: An Enhanced Bayesian Upper Confidence Bound Framework

**Authors**: *Ziyi Huang, Henry Lam, Amirhossein Meisami, Haofeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html)

**Abstract**:

Bayesian bandit algorithms with approximate Bayesian inference have been widely used in real-world applications. However, there is a large discrepancy between the superior practical performance of these approaches and their theoretical justification. Previous research only indicates a negative theoretical result: Thompson sampling could have a worst-case linear regret $\Omega(T)$ with a constant threshold on the inference error measured by one $\alpha$-divergence. To bridge this gap, we propose an Enhanced Bayesian Upper Confidence Bound (EBUCB) framework that can efficiently accommodate bandit problems in the presence of approximate inference. Our theoretical analysis demonstrates that for Bernoulli multi-armed bandits, EBUCB can achieve the optimal regret order $O(\log T)$ if the inference error measured by two different $\alpha$-divergences is less than a constant, regardless of how large this constant is. To our best knowledge, our study provides the first theoretical regret bound that is better than $o(T)$ in the setting of constant approximate inference error. Furthermore, in concordance with the negative results in previous studies, we show that only one bounded $\alpha$-divergence is insufficient to guarantee a sub-linear regret.

----

## [0] Any-to-Any Generation via Composable Diffusion

**Authors**: *Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/33edf072fe44f19079d66713a1831550-Abstract-Conference.html)

**Abstract**:

We present Composable Diffusion (CoDi), a novel generative model capable of generating any combination of output modalities, such as language, image, video, or audio, from any combination of input modalities. Unlike existing generative AI systems, CoDi can generate multiple modalities in parallel and its input is not limited to a subset of modalities like text or image. Despite the absence of training datasets for many combinations of modalities, we propose to align modalities in both the input and output space. This allows CoDi to freely condition on any input combination and generate any group of modalities, even if they are not present in the training data. CoDi employs a novel composable generation strategy which involves building a shared multimodal space by bridging alignment in the diffusion process, enabling the synchronized generation of intertwined modalities, such as temporally aligned video and audio. Highly customizable and flexible, CoDi achieves strong joint-modality generation quality, and outperforms or is on par with the unimodal state-of-the-art for single-modality synthesis.

----

## [0] Rank-DETR for High Quality Object Detection

**Authors**: *Yifan Pu, Weicong Liang, Yiduo Hao, Yuhui Yuan, Yukang Yang, Chao Zhang, Han Hu, Gao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34074479ee2186a9f236b8fd03635372-Abstract-Conference.html)

**Abstract**:

Modern detection transformers (DETRs) use a set of object queries to predict a list of bounding boxes, sort them by their classification confidence scores, and select the top-ranked predictions as the final detection results for the given input image. A highly performant object detector requires accurate ranking for the bounding box predictions. For DETR-based detectors, the top-ranked bounding boxes suffer from less accurate localization quality due to the misalignment between classification scores and localization accuracy, thus impeding the construction of high-quality detectors. In this work, we introduce a simple and highly performant DETR-based object detector by proposing a series of rank-oriented designs, combinedly called Rank-DETR. Our key contributions include: (i) a rank-oriented architecture design that can prompt positive predictions and suppress the negative ones to ensure lower false positive rates, as well as (ii) a rank-oriented loss function and matching cost design that prioritizes predictions of more accurate localization accuracy during ranking to boost the AP under high IoU thresholds. We apply our method to improve the recent SOTA methods (e.g., H-DETR and DINO-DETR) and report strong COCO object detection results when using different backbones such as ResNet-$50$, Swin-T, and Swin-L, demonstrating the effectiveness of our approach. Code is available at \url{https://github.com/LeapLabTHU/Rank-DETR}.

----

## [0] Towards Efficient Pre-Trained Language Model via Feature Correlation Distillation

**Authors**: *Kun Huang, Xin Guo, Meng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html)

**Abstract**:

Knowledge Distillation (KD) has emerged as a promising approach for compressing large Pre-trained Language Models (PLMs). The performance of KD relies on how to effectively formulate and transfer the knowledge from the teacher model to the student model. Prior arts mainly focus on directly aligning output features from the transformer block, which may impose overly strict constraints on the student model's learning process and complicate the training process by introducing extra parameters and computational cost. Moreover, our analysis indicates that the different relations within self-attention, as adopted in other works, involves more computation complexities and can easily be constrained by the number of heads, potentially leading to suboptimal solutions.  To address these issues, we propose a novel approach that builds relationships directly from output features. Specifically, we introduce token-level and sequence-level relations concurrently  to fully exploit the knowledge from the teacher model. Furthermore, we propose a correlation-based distillation loss to alleviate the exact match properties inherent in traditional KL divergence or MSE loss functions. Our method, dubbed FCD, presents a simple yet effective method to compress various architectures (BERT, RoBERTa, and GPT) and model sizes (base-size and large-size).  Extensive experimental results demonstrate that our distilled, smaller language models significantly surpass existing KD methods across various NLP tasks.

----

## [0] Simple and Asymmetric Graph Contrastive Learning without Augmentations

**Authors**: *Teng Xiao, Huaisheng Zhu, Zhengyu Chen, Suhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3430bcc30cdaabd0bf6c5d0c31bda67c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3430bcc30cdaabd0bf6c5d0c31bda67c-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has shown superior performance in representation learning in graph-structured data. Despite their success, most existing GCL methods rely on prefabricated graph augmentation and homophily assumptions. Thus, they fail to generalize well to heterophilic graphs where connected nodes may have different class labels and dissimilar features. In this paper, we study the problem of conducting contrastive learning on homophilic and heterophilic graphs. We find that we can achieve promising performance simply by considering an asymmetric view of the neighboring nodes. The resulting simple algorithm, Asymmetric Contrastive Learning for Graphs (GraphACL), is easy to implement and does not rely on graph augmentations and homophily assumptions. We provide theoretical and empirical evidence that GraphACL can capture one-hop local neighborhood information and two-hop monophily similarity, which are both important for modeling heterophilic graphs. Experimental results show that the simple GraphACL significantly outperforms state-of-the-art graph contrastive learning and self-supervised learning methods on  homophilic and heterophilic graphs. The code of GraphACL is available at https://github.com/tengxiao1/GraphACL.

----

## [0] Large-Scale Distributed Learning via Private On-Device LSH

**Authors**: *Tahseen Rabbani, Marco Bornstein, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html)

**Abstract**:

Locality-sensitive hashing (LSH) based frameworks have been used efficiently to select weight vectors in a dense hidden layer with high cosine similarity to an input, enabling dynamic pruning.  While this type of scheme has been shown to improve computational training efficiency, existing algorithms require repeated randomized projection of the full layer weight, which is impractical for computational- and memory-constrained devices.  In a distributed setting, deferring LSH analysis to a centralized host is (i) slow if the device cluster is large and (ii) requires access to input data which is forbidden in a federated context.  Using a new family of hash functions, we develop the first private, personalized, and memory-efficient on-device LSH framework.Our framework enables privacy and personalization by allowing each device to generate hash tables, without the help of a central host, using device-specific hashing hyper-parameters (e.g., number of hash tables or hash length).Hash tables are generated with a compressed set of the full weights, and can be serially generated and discarded if the process is memory-intensive.This allows devices to avoid maintaining (i) the fully-sized model and (ii) large amounts of hash tables in local memory for LSH analysis. We prove several statistical and sensitivity properties of our hash functions, and experimentally demonstrate that our framework is competitive in training large scale recommender networks compared to other LSH frameworks which assume unrestricted on-device capacity.

----

## [0] Facilitating Graph Neural Networks with Random Walk on Simplicial Complexes

**Authors**: *Cai Zhou, Xiyuan Wang, Muhan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/345208bdbbb6104616311dfc1d093fe7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/345208bdbbb6104616311dfc1d093fe7-Abstract-Conference.html)

**Abstract**:

Node-level random walk has been widely used to improve Graph Neural Networks. However, there is limited attention to random walk on edge and, more generally, on $k$-simplices. This paper systematically analyzes how random walk on different orders of simplicial complexes (SC) facilitates GNNs in their theoretical expressivity. First, on $0$-simplices or node level, we establish a connection between existing positional encoding (PE) and structure encoding (SE) methods through the bridge of random walk. Second, on $1$-simplices or edge level, we bridge edge-level random walk and Hodge $1$-Laplacians and design corresponding edge PE respectively. In spatial domain, we directly make use of edge level random walk to construct EdgeRWSE. Based on spectral analysis of Hodge $1$-Laplcians, we propose Hodge1Lap, a permutation equivariant and expressive edge-level positional encoding. Third, we generalize our theory to random walk on higher-order simplices and propose the general principle to design PE on simplices based on random walk and Hodge Laplacians. Inter-level random walk is also introduced to unify a wide range of simplicial networks. Extensive experiments verify the effectiveness of our random walk-based methods.

----

## [0] Koopman Kernel Regression

**Authors**: *Petar Bevanda, Max Beier, Armin Lederer, Stefan Sosnowski, Eyke Hüllermeier, Sandra Hirche*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34678d08b36076de986df95c5bbba92f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34678d08b36076de986df95c5bbba92f-Abstract-Conference.html)

**Abstract**:

Many machine learning approaches for decision making, such as reinforcement learning, rely on simulators or predictive models to forecast the time-evolution of quantities of interest, e.g., the state of an agent or the reward of a policy. Forecasts of such complex phenomena are commonly described by highly nonlinear dynamical systems, making their use in optimization-based decision-making challenging.Koopman operator theory offers a beneficial paradigm for addressing this problem by characterizing forecasts via linear time-invariant (LTI) ODEs, turning multi-step forecasts into sparse matrix multiplication.Though there exists a variety of learning approaches, they usually lack crucial learning-theoretic guarantees, making the behavior of the obtained models with increasing data and dimensionality unclear.We address the aforementioned by deriving a universal Koopman-invariant reproducing kernel Hilbert space (RKHS) that solely spans transformations into LTI dynamical systems. The resulting Koopman Kernel Regression (KKR) framework enables the use of statistical learning tools from function approximation for novel convergence results and generalization error bounds under weaker assumptions than existing work. Our experiments demonstrate superior forecasting performance compared to Koopman operator and sequential data predictors in RKHS.

----

## [0] Diffusion Self-Guidance for Controllable Image Generation

**Authors**: *Dave Epstein, Allan Jabri, Ben Poole, Alexei A. Efros, Aleksander Holynski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html)

**Abstract**:

Large-scale generative models are capable of producing high-quality images from detailed prompts. However, many aspects of an image are difficult or impossible to convey through text. We introduce self-guidance, a method that provides precise control over properties of the generated image by guiding the internal representations of diffusion models. We demonstrate that the size, location, and appearance of objects can be extracted from these representations, and show how to use them to steer the sampling process. Self-guidance operates similarly to standard classifier guidance, but uses signals present in the pretrained model itself, requiring no additional models or training. We demonstrate the flexibility and effectiveness of self-guided generation through a wide range of challenging image manipulations, such as modifying the position or size of a single object (keeping the rest of the image unchanged), merging the appearance of objects in one image with the layout of another, composing objects from multiple images into one, and more. We also propose a new method for reconstruction using self-guidance, which allows extending our approach to editing real images.

----

## [0] Neural (Tangent Kernel) Collapse

**Authors**: *Mariia Seleznova, Dana Weitzner, Raja Giryes, Gitta Kutyniok, Hung-Hsu Chou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3477ca0ce484aa2fa42c1361ab601c25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3477ca0ce484aa2fa42c1361ab601c25-Abstract-Conference.html)

**Abstract**:

This work bridges two important concepts: the Neural Tangent Kernel (NTK), which captures the evolution of deep neural networks (DNNs) during training, and the Neural Collapse (NC) phenomenon, which refers to the emergence of symmetry and structure in the last-layer features of well-trained classification DNNs. We adopt the natural assumption that the empirical NTK develops a block structure aligned with the class labels, i.e., samples within the same class have stronger correlations than samples from different classes. Under this assumption, we derive the dynamics of DNNs trained with mean squared (MSE) loss and break them into interpretable phases. Moreover, we identify an invariant that captures the essence of the dynamics, and use it to prove the emergence of NC in DNNs with block-structured NTK. We provide large-scale numerical experiments on three common DNN architectures and three benchmark datasets to support our theory.

----

## [0] Fast Optimal Locally Private Mean Estimation via Random Projections

**Authors**: *Hilal Asi, Vitaly Feldman, Jelani Nelson, Huy L. Nguyen, Kunal Talwar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34822dab66c13f0100017b8ea373038a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34822dab66c13f0100017b8ea373038a-Abstract-Conference.html)

**Abstract**:

We study the problem of locally private mean estimation of high-dimensional vectors in the Euclidean ball. Existing algorithms for this problem either incur sub-optimal error or have high communication and/or run-time complexity. We propose a new algorithmic framework, namely ProjUnit, for private mean estimation that yields algorithms that are computationally efficient, have low communication complexity, and incur optimal error up to a $1+o(1)$-factor. Our framework is deceptively simple: each randomizer projects its input to a random low-dimensional subspace and then runs an optimal algorithm such a PrivUnitG in the lower dimensional space. We analyze the error of the algorithm in terms of properties of the random projection ensemble, and study two instantiations. We conduct several experiments for private mean estimation and private federated learning which demonstrate that our algorithms obtain nearly the same utility as optimal algorithms while having significantly lower communication and computational cost.

----

## [0] Expert load matters: operating networks at high accuracy and low manual effort

**Authors**: *Sara Sangalli, Ertunc Erdil, Ender Konukoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/348346383eb58ed19def02e233c408d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/348346383eb58ed19def02e233c408d6-Abstract-Conference.html)

**Abstract**:

In human-AI collaboration systems for critical applications, in order to ensure minimal error, users should set an operating point based on model confidence to determine when the decision should be delegated to human experts. Samples for which model confidence is lower than the operating point would be manually analysed by experts to avoid mistakes.Such systems can become truly useful only if they consider two aspects: models should be confident only for samples for which they are accurate, and the number of samples delegated to experts should be minimized.The latter aspect is especially crucial for applications where available expert time is limited and expensive, such as healthcare. The trade-off between the model accuracy and the number of samples delegated to experts can be represented by a curve that is similar to an ROC curve, which we refer to as confidence operating characteristic (COC) curve. In this paper, we argue that deep neural networks should be trained by taking into account both accuracy and expert load and, to that end, propose a new complementary loss function for classification that maximizes the area under this COC curve.This promotes simultaneously the increase in network accuracy and the reduction in number of samples delegated to humans.We perform experiments on multiple computer vision and medical image datasets for classification.Our results demonstrate that the proposed loss improves classification accuracy and delegates less number of decisions to experts, achieves better out-of-distribution samples detection and on par calibration performance compared to existing loss functions.

----

## [0] PRODIGY: Enabling In-context Learning Over Graphs

**Authors**: *Qian Huang, Hongyu Ren, Peng Chen, Gregor Krzmanc, Daniel Zeng, Percy Liang, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34dce0dc3121951dd0399ba02c0f0d06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34dce0dc3121951dd0399ba02c0f0d06-Abstract-Conference.html)

**Abstract**:

In-context learning is the ability of a pretrained model to  adapt to novel and diverse downstream tasks by conditioning on prompt examples, without optimizing any parameters. While large language models have demonstrated this ability, how in-context learning could be performed over graphs is unexplored. In this paper, we develop \textbf{Pr}etraining \textbf{O}ver \textbf{D}iverse \textbf{I}n-Context  \textbf{G}raph S\textbf{y}stems (PRODIGY), the first pretraining framework that enables in-context learning over graphs. The key idea of our framework is to formulate in-context learning over graphs with a novel \emph{prompt graph} representation, which connects prompt examples and queries. We then propose a graph neural network architecture over the prompt graph and a corresponding family of in-context pretraining objectives. With PRODIGY, the pretrained model can directly perform novel downstream classification tasks on unseen graphs via in-context learning. We provide empirical evidence of the effectiveness of our framework by showcasing its strong in-context learning performance on tasks involving citation networks and knowledge graphs. Our approach outperforms the in-context learning accuracy of contrastive pretraining baselines with hard-coded adaptation by 18\% on average across all setups. Moreover, it also outperforms standard finetuning with limited data by 33\% on average with in-context learning.

----

## [0] Towards Automated Circuit Discovery for Mechanistic Interpretability

**Authors**: *Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, Adrià Garriga-Alonso*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html)

**Abstract**:

Through considerable effort and intuition, several recent works have reverse-engineered nontrivial behaviors oftransformer models. This paper systematizes the mechanistic interpretability process they followed. First, researcherschoose a metric and dataset that elicit the desired model behavior. Then, they apply activation patching to find whichabstract neural network units are involved in the behavior. By varying the dataset, metric, and units underinvestigation, researchers can understand the functionality of each component.We automate one of the process' steps: finding the connections between the abstract neural network units that form a circuit. We propose several algorithms and reproduce previous interpretability results to validate them. Forexample, the ACDC algorithm rediscovered 5/5 of the component types in a circuit in GPT-2 Small that computes theGreater-Than operation. ACDC selected 68 of the 32,000 edges in GPT-2 Small, all of which were manually found byprevious work. Our code is available at https://github.com/ArthurConmy/Automatic-Circuit-Discovery

----

## [0] Find What You Want: Learning Demand-conditioned Object Attribute Space for Demand-driven Navigation

**Authors**: *Hongcheng Wang, Andy Guan Hong Chen, Xiaoqi Li, Mingdong Wu, Hao Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34e278fbbd7d6d7d788c98065988e1a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34e278fbbd7d6d7d788c98065988e1a9-Abstract-Conference.html)

**Abstract**:

The task of Visual Object Navigation (VON) involves an agent's ability to locate a particular object within a given scene. To successfully accomplish the VON task, two essential conditions must be fulfiled:  1) the user knows the name of the desired object; and 2) the user-specified object actually is present within the scene. To meet these conditions, a simulator can incorporate predefined object names and positions into the metadata of the scene. However, in real-world scenarios, it is often challenging to ensure that these conditions are always met. Humans in an unfamiliar environment may not know which objects are present in the scene, or they may mistakenly specify an object that is not actually present.  Nevertheless, despite these challenges, humans may still have a demand for an object, which could potentially be fulfilled by other objects present within the scene in an equivalent manner. Hence, this paper proposes Demand-driven Navigation (DDN), which leverages the user's demand as the task instruction and prompts the agent to find an object which matches the specified demand. DDN aims to relax the stringent conditions of VON by focusing on fulfilling the user's demand rather than relying solely on specified object names. This paper proposes a method of acquiring textual attribute features of objects by extracting common sense knowledge  from a large language model (LLM). These textual attribute features are subsequently aligned with visual attribute features using Contrastive Language-Image Pre-training (CLIP). Incorporating the visual attribute features as prior knowledge, enhances the navigation process. Experiments on AI2Thor with the ProcThor dataset demonstrate that the visual attribute features improve the agent's navigation performance and outperform the baseline methods commonly used in the VON and VLN task and methods with LLMs. The codes and demonstrations can be viewed at https://sites.google.com/view/demand-driven-navigation.

----

## [0] On the Convergence of No-Regret Learning Dynamics in Time-Varying Games

**Authors**: *Ioannis Anagnostides, Ioannis Panageas, Gabriele Farina, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/34f1c2e7ab91b6fa481ad0286a08ad02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/34f1c2e7ab91b6fa481ad0286a08ad02-Abstract-Conference.html)

**Abstract**:

Most of the literature on learning in games has focused on the restrictive setting where the underlying repeated game does not change over time. Much less is known about the convergence of no-regret learning algorithms in dynamic multiagent settings. In this paper, we characterize the convergence of optimistic gradient descent (OGD) in time-varying games. Our framework yields sharp convergence bounds for the equilibrium gap of OGD in zero-sum games parameterized on natural variation measures of the sequence of games, subsuming known results for static games. Furthermore, we establish improved second-order variation bounds under strong convexity-concavity, as long as each game is repeated multiple times. Our results also apply to time-varying general-sum multi-player games via a bilinear formulation of correlated equilibria, which has novel implications for meta-learning and for obtaining refined variation-dependent regret bounds, addressing questions left open in prior papers. Finally, we leverage our framework to also provide new insights on dynamic regret guarantees in static games.

----

## [0] Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design

**Authors**: *Ibrahim M. Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, Lucas Beyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3504a4fa45685d668ce92797fbbf1895-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3504a4fa45685d668ce92797fbbf1895-Abstract-Conference.html)

**Abstract**:

Scaling laws have been recently employed to derive compute-optimal model size (number of parameters) for a given compute duration. We advance and refine such methods to infer compute-optimal model shapes, such as width and depth, and successfully implement this in vision transformers. Our shape-optimized vision transformer, SoViT, achieves results competitive with models that exceed twice its size, despite being pre-trained with an equivalent amount of  compute. For example, SoViT-400m/14 achieves 90.3% fine-tuning accuracy on ILSRCV2012, surpassing the much larger ViT-g/14 and approaching ViT-G/14 under identical settings, with also less than half the inference cost. We conduct a thorough evaluation across multiple tasks, such as image classification, captioning, VQA and zero-shot transfer, demonstrating the effectiveness of our model across a broad range of domains and identifying limitations. Overall, our findings challenge the prevailing approach of blindly scaling up vision models and pave a path for a more informed scaling.

----

## [0] Expressive Sign Equivariant Networks for Spectral Geometric Learning

**Authors**: *Derek Lim, Joshua Robinson, Stefanie Jegelka, Haggai Maron*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3516aa3393f0279e04c099f724664f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3516aa3393f0279e04c099f724664f99-Abstract-Conference.html)

**Abstract**:

Recent work has shown the utility of developing machine learning models that respect the structure and symmetries of eigenvectors. These works promote sign invariance, since for any eigenvector v the negation -v is also an eigenvector. However, we show that sign invariance is theoretically limited for tasks such as building orthogonally equivariant models and learning node positional encodings for link prediction in graphs. In this work, we demonstrate the benefits of sign equivariance for these tasks. To obtain these benefits, we develop novel sign equivariant neural network architectures. Our models are based on a new analytic characterization of sign equivariant polynomials and thus inherit provable expressiveness properties. Controlled synthetic experiments show that our networks can achieve the theoretically predicted benefits of sign equivariant models.

----

## [0] FLAIR : a Country-Scale Land Cover Semantic Segmentation Dataset From Multi-Source Optical Imagery

**Authors**: *Anatol Garioud, Nicolas Gonthier, Loïc Landrieu, Apolline De Wit, Marion Valette, Marc Poupée, Sébastien Giordano, Boris Wattrelos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/353ca88f722cdd0c481b999428ae113a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce the French Land cover from Aerospace ImageRy (FLAIR), an extensive dataset from the French National Institute of Geographical and Forest Information (IGN) that provides a unique and rich resource for large-scale geospatial analysis. FLAIR contains high-resolution aerial imagery with a ground sample distance of 20 cm and over 20 billion individually labeled pixels for precise land-cover classification. The dataset also integrates temporal and spectral data from optical satellite time series. FLAIR thus combines data with varying spatial, spectral, and temporal resolutions across over 817 km² of acquisitions representing the full landscape diversity of France. This diversity makes FLAIR a valuable resource for the development and evaluation of novel methods for large-scale land-cover semantic segmentation and raises significant challenges in terms of computer vision, data fusion, and geospatial analysis. We also provide powerful uni- and multi-sensor baseline models that can be employed to assess algorithm's performance and for downstream applications.

----

## [0] Strategic Data Sharing between Competitors

**Authors**: *Nikita Tsoy, Nikola Konstantinov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/355091f86e3e2296fbeefa10676ddb17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/355091f86e3e2296fbeefa10676ddb17-Abstract-Conference.html)

**Abstract**:

Collaborative learning techniques have significantly advanced in recent years, enabling private model training across multiple organizations. Despite this opportunity, firms face a dilemma when considering data sharing with competitors—while collaboration can improve a company’s machine learning model, it may also benefit competitors and hence reduce profits. In this work, we introduce a general framework for analyzing this data-sharing trade-off. The framework consists of three components, representing the firms’ production decisions, the effect of additional data on model quality, and the data-sharing negotiation process, respectively. We then study an instantiation of the framework, based on a conventional market model from economic theory, to identify key factors that affect collaboration incentives. Our findings indicate a profound impact of market conditions on the data-sharing incentives. In particular, we find that reduced competition, in terms of the similarities between the firms’ products, and harder learning tasks foster collaboration.

----

## [0] Optimal Time Complexities of Parallel Stochastic Optimization Methods Under a Fixed Computation Model

**Authors**: *Alexander Tyurin, Peter Richtárik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3563abb1040f4e150f4242a7282cd1ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3563abb1040f4e150f4242a7282cd1ec-Abstract-Conference.html)

**Abstract**:

Parallelization is a popular strategy for improving the performance of methods. Optimization methods are no exception: design of efficient parallel optimization methods and tight analysis of their theoretical properties are important research endeavors. While the minimax complexities are well known for sequential optimization methods, the theory of parallel optimization methods is less explored. In this paper, we propose a new protocol that generalizes the classical oracle framework approach. Using this protocol, we establish minimax complexities for parallel optimization methods that have access to an unbiased stochastic gradient oracle with bounded variance. We consider a fixed computation model characterized by each worker requiring a fixed but worker-dependent time to calculate stochastic gradient. We prove lower bounds and develop optimal algorithms that attain them. Our results have surprising consequences for the literature of asynchronous optimization methods.

----

## [0] An ε-Best-Arm Identification Algorithm for Fixed-Confidence and Beyond

**Authors**: *Marc Jourdan, Rémy Degenne, Emilie Kaufmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/358e4a39b8ace4744fbad77e84a7e757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/358e4a39b8ace4744fbad77e84a7e757-Abstract-Conference.html)

**Abstract**:

We propose EB-TC$\varepsilon$, a novel sampling rule for $\varepsilon$-best arm identification in stochastic bandits.It is the first instance of Top Two algorithm analyzed for approximate best arm identification. EB-TC$\varepsilon$ is an *anytime* sampling rule that can therefore be employed without modification for fixed confidence or fixed budget identification (without prior knowledge of the budget).We provide three types of theoretical guarantees for EB-TC$\varepsilon$.First, we prove bounds on its expected sample complexity in the fixed confidence setting, notably showing its asymptotic optimality in combination with an adaptive tuning of its exploration parameter.We complement these findings with upper bounds on its probability of error at any time and for any slack parameter, which further yield upper bounds on its simple regret at any time.Finally, we show through numerical simulations that EB-TC$\varepsilon$ performs favorably compared to existing algorithms for different approximate best arm identification tasks.

----

## [0] Debiased and Denoised Entity Recognition from Distant Supervision

**Authors**: *Haobo Wang, Yiwen Dong, Ruixuan Xiao, Fei Huang, Gang Chen, Junbo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html)

**Abstract**:

While distant supervision has been extensively explored and exploited in NLP tasks like named entity recognition, a major obstacle stems from the inevitable noisy distant labels tagged unsupervisedly. A few past works approach this problem by adopting a self-training framework with a sample-selection mechanism. In this work, we innovatively identify two types of biases that were omitted by prior work, and these biases lead to inferior performance of the distant-supervised NER setup. First, we characterize the noise concealed in the distant labels as highly structural rather than fully randomized. Second, the self-training framework would ubiquitously introduce an inherent bias that causes erroneous behavior in both sample selection and eventually prediction. To cope with these problems, we propose a novel self-training framework, dubbed DesERT. This framework augments the conventional NER predicative pathway to a dual form that effectively adapts the sample-selection process to conform to its innate distributional-bias structure. The other crucial component of DesERT composes a debiased module aiming to enhance the token representations, hence the quality of the pseudo-labels. Extensive experiments are conducted to validate the DesERT. The results show that our framework establishes a new state-of-art performance, it achieves a +2.22% average F1 score improvement on five standardized benchmarking datasets. Lastly, DesERT demonstrates its effectiveness under a new DSNER benchmark where additional distant supervision comes from the ChatGPT model.

----

## [0] Accelerated Training via Incrementally Growing Neural Networks using Variance Transfer and Learning Rate Adaptation

**Authors**: *Xin Yuan, Pedro Savarese, Michael Maire*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/359ffa88712bd688963a0ca641d8330b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/359ffa88712bd688963a0ca641d8330b-Abstract-Conference.html)

**Abstract**:

We develop an approach to efficiently grow neural networks, within which parameterization and optimization strategies are designed by considering their effects on the training dynamics.  Unlike existing growing methods, which follow simple replication heuristics or utilize auxiliary gradient-based local optimization, we craft a parameterization scheme which dynamically stabilizes weight, activation, and gradient scaling as the architecture evolves, and maintains the inference functionality of the network.  To address the optimization difficulty resulting from imbalanced training effort distributed to subnetworks fading in at different growth phases, we propose a learning rate adaption mechanism that rebalances the gradient contribution of these separate subcomponents.  Experiments show that our method achieves comparable or better accuracy than training large fixed-size models, while saving a substantial portion of the original training computation budget.  We demonstrate that these gains translate into real wall-clock training speedups.

----

## [0] Likelihood-Based Diffusion Language Models

**Authors**: *Ishaan Gulrajani, Tatsunori B. Hashimoto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html)

**Abstract**:

Despite a growing interest in diffusion-based language models, existing work has not shown that these models can attain nontrivial likelihoods on standard language modeling benchmarks. In this work, we take the first steps towards closing the likelihood gap between autoregressive and diffusion-based language models, with the goal of building and releasing a diffusion model which outperforms a small but widely-known autoregressive model. We pursue this goal through algorithmic improvements, scaling laws, and increased compute. On the algorithmic front, we introduce several methodological improvements for the maximum-likelihood training of diffusion language models. We then study scaling laws for our diffusion models and find compute-optimal training regimes which differ substantially from autoregressive models. Using our methods and scaling analysis, we train and release Plaid 1B, a large diffusion language model which outperforms GPT-2 124M in likelihood on benchmark datasets and generates fluent samples in unconditional and zero-shot control settings.

----

## [0] Structural Pruning for Diffusion Models

**Authors**: *Gongfan Fang, Xinyin Ma, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html)

**Abstract**:

Generative modeling has recently undergone remarkable advancements, primarily propelled by the transformative implications of Diffusion Probabilistic Models (DPMs). The impressive capability of these models, however, often entails significant computational overhead during both training and inference. To tackle this challenge, we present Diff-Pruning, an efficient compression method tailored for learning lightweight diffusion models from pre-existing ones, without the need for extensive re-training. The essence of Diff-Pruning is encapsulated in a Taylor expansion over pruned timesteps, a process that disregards non-contributory diffusion steps and ensembles informative gradients to identify important weights. Our empirical assessment, undertaken across several datasets highlights two primary benefits of our proposed method: 1) Efficiency: it enables approximately a 50\% reduction in FLOPs at a mere 10% to 20% of the original training expenditure; 2) Consistency: the pruned diffusion models inherently preserve generative behavior congruent with their pre-trained models.

----

## [0] Fast and Regret Optimal Best Arm Identification: Fundamental Limits and Low-Complexity Algorithms

**Authors**: *Qining Zhang, Lei Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/35fdecdf8861bc15110d48fbec3193cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/35fdecdf8861bc15110d48fbec3193cf-Abstract-Conference.html)

**Abstract**:

This paper considers a stochastic Multi-Armed Bandit (MAB) problem with dual objectives: (i) quick identification and commitment to the optimal arm, and (ii) reward maximization throughout a sequence of $T$ consecutive rounds. Though each objective has been individually well-studied, i.e., best arm identification for (i) and regret minimization for (ii), the simultaneous realization of both objectives remains an open problem, despite its practical importance. This paper introduces \emph{Regret Optimal Best Arm Identification} (ROBAI) which aims to achieve these dual objectives. To solve ROBAI with both pre-determined stopping time and adaptive stopping time requirements, we present an algorithm called EOCP and its variants respectively, which not only achieve asymptotic optimal regret in both Gaussian and general bandits, but also commit to the optimal arm in $\mathcal{O}(\log T)$ rounds with pre-determined stopping time and $\mathcal{O}(\log^2 T)$ rounds with adaptive stopping time. We further characterize lower bounds on the commitment time (equivalent to the sample complexity) of ROBAI, showing that EOCP and its variants are sample optimal with pre-determined stopping time, and almost sample optimal with adaptive stopping time. Numerical results confirm our theoretical analysis and reveal an interesting ``over-exploration'' phenomenon carried by classic UCB algorithms, such that EOCP has smaller regret even though it stops exploration much earlier than UCB, i.e., $\mathcal{O}(\log T)$ versus $\mathcal{O}(T)$, which suggests over-exploration is unnecessary and potentially harmful to system performance.

----

## [0] Simultaneous embedding of multiple attractor manifolds in a recurrent neural network using constrained gradient optimization

**Authors**: *Haggai Agmon, Yoram Burak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html)

**Abstract**:

The storage of continuous variables in working memory is hypothesized to be sustained in the brain by the dynamics of recurrent neural networks (RNNs) whose steady states form continuous manifolds. In some cases, it is thought that the synaptic connectivity supports multiple attractor manifolds, each mapped to a different context or task. For example, in hippocampal area CA3, positions in distinct environments are represented by distinct sets of population activity patterns, each forming a continuum. It has been argued that the embedding of multiple continuous attractors in a single RNN inevitably causes detrimental interference: quenched noise in the synaptic connectivity disrupts the continuity of each attractor, replacing it by a discrete set of steady states that can be conceptualized as lying on local minima of an abstract energy landscape. Consequently, population activity patterns exhibit systematic drifts towards one of these discrete minima, thereby degrading the stored memory over time. Here we show that it is possible to dramatically attenuate these detrimental interference effects by adjusting the synaptic weights. Synaptic weight adjustment are derived from a loss function that quantifies the roughness of the energy landscape along each of the embedded attractor manifolds. By minimizing this loss function, the stability of states can be dramatically improved, without compromising the capacity.

----

## [0] Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization

**Authors**: *Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan S. Kankanhalli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/364071531ff2398e0fb8bae31f615b69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/364071531ff2398e0fb8bae31f615b69-Abstract-Conference.html)

**Abstract**:

Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/EnhancingACLvia_AIR.

----

## [0] Best Arm Identification with Fixed Budget: A Large Deviation Perspective

**Authors**: *Po-An Wang, Ruo-Chun Tzeng, Alexandre Proutière*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/364d565b4b726c607aa40e1632045873-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/364d565b4b726c607aa40e1632045873-Abstract-Conference.html)

**Abstract**:

We consider the problem of identifying the best arm in stochastic Multi-Armed Bandits (MABs) using a fixed sampling budget. Characterizing the minimal instance-specific error probability for this problem constitutes one of the important remaining open problems in MABs. When arms are selected using a static sampling strategy, the error probability decays exponentially with the number of samples at a rate that can be explicitly derived via Large Deviation techniques. Analyzing the performance of algorithms with adaptive sampling strategies is however much more challenging. In this paper, we establish a connection between the Large Deviation Principle (LDP) satisfied by the empirical proportions of arm draws and that satisfied by the empirical arm rewards. This connection holds for any adaptive algorithm, and is leveraged (i) to improve error probability upper bounds of some existing algorithms, such as the celebrated SR (Successive Rejects) algorithm \cite{audibert2010best}, and (ii) to devise and analyze new algorithms. In particular, we present CR (Continuous Rejects), a truly adaptive algorithm that can reject arms in {\it any} round based on the observed empirical gaps between the rewards of various arms. Applying our Large Deviation results, we prove that CR enjoys better performance guarantees than existing algorithms, including SR. Extensive numerical experiments confirm this observation.

----

## [0] Full-Atom Protein Pocket Design via Iterative Refinement

**Authors**: *Zaixi Zhang, Zepu Lu, Zhongkai Hao, Marinka Zitnik, Qi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/365a6f71486ecdfa7eb8d61cbe168782-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/365a6f71486ecdfa7eb8d61cbe168782-Abstract-Conference.html)

**Abstract**:

The design of \emph{de novo} functional proteins that bind with specific ligand molecules is crucial in various domains like therapeutics and bio-engineering. One vital yet challenging step is to design the protein pocket, the cavity region of protein where the ligand binds with. Existing methods suffer from inefficient generation, insufficient context modeling (ligand molecule), and incapability of generating sidechain atoms. To overcome the limitations, we propose a \textbf{F}ull-\textbf{A}tom \textbf{I}terative \textbf{R}efinement framework (\textbf{FAIR}) for protein pocket sequence (i.e., residue types) and 3D structure co-design. Generally, FAIR consists of two steps that follow a coarse-to-fine pipeline (backbone atoms to full atoms including sidechain) for full-atom generation. For efficiency, all residue types and structures are updated together in each round (i.e., full-shot refinement). In the first step, the residue types and backbone coordinates are updated with a hierarchical context encoder and two structure refinement modules capturing inter-residue and pocket-ligand interactions. The second step further models the sidechain atoms of pockets and updates residue types to achieve sequence-structure consistency. The structure of the binding ligand is also updated along with the above refinement iterations accounting for its flexibility. Finally, extensive evaluations showthat FAIR outperforms baselines in efficiently designing high-quality pocket sequences and structures. Specifically, the average improvements on AAR and RMSD are over 10$\%$.

----

## [0] Flow Matching for Scalable Simulation-Based Inference

**Authors**: *Jonas Wildberger, Maximilian Dax, Simon Buchholz, Stephen R. Green, Jakob H. Macke, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract-Conference.html)

**Abstract**:

Neural posterior estimation methods based on discrete normalizing flows have become established tools for simulation-based inference (SBI), but scaling them to high-dimensional problems can be challenging. Building on recent advances in generative modeling, we here present flow matching posterior estimation (FMPE), a technique for SBI using continuous normalizing flows. Like diffusion models, and in contrast to discrete flows, flow matching allows for unconstrained architectures, providing enhanced flexibility for complex data modalities. Flow matching, therefore, enables exact density evaluation, fast training, and seamless scalability to large architectures---making it ideal for SBI. We show that FMPE achieves competitive performance on an established SBI benchmark, and then demonstrate its improved scalability on a challenging scientific problem: for gravitational-wave inference, FMPE outperforms methods based on comparable discrete flows, reducing training time by 30\% with substantially improved accuracy. Our work underscores the potential of FMPE to enhance performance in challenging inference scenarios, thereby paving the way for more advanced applications to scientific problems.

----

## [0] Learning DAGs from Data with Few Root Causes

**Authors**: *Panagiotis Misiakos, Chris Wendler, Markus Püschel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/367ab3106d990825d5b47ce91db75a73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/367ab3106d990825d5b47ce91db75a73-Abstract-Conference.html)

**Abstract**:

We present a novel perspective and algorithm for learning directed acyclic graphs (DAGs) from data generated by a linear structural equation model (SEM). First, we show that a linear SEM can be viewed as a linear transform that, in prior work, computes the data from a dense input vector of random valued root causes (as we will call them) associated with the nodes. Instead, we consider the case of (approximately) few root causes and also introduce noise in the measurement of the data. Intuitively, this means that the DAG data is produced by few data generating events whose effect percolates through the DAG. We prove identifiability in this new setting and show that the true DAG is the global minimizer of the $L^0$-norm of the vector of root causes. For data satisfying the few root causes assumption, we show superior performance compared to prior DAG learning methods.

----

## [0] Robust Learning for Smoothed Online Convex Optimization with Feedback Delay

**Authors**: *Pengfei Li, Jianyi Yang, Adam Wierman, Shaolei Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36848567d39a5128e671ad04a6075374-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36848567d39a5128e671ad04a6075374-Abstract-Conference.html)

**Abstract**:

We study a general form of Smoothed Online Convex Optimization, a.k.a. SOCO, including multi-step switching costs and feedback delay. We propose a novel machine learning (ML) augmented online algorithm, Robustness-Constrained Learning (RCL), which combines untrusted ML predictions with a trusted expert online algorithm via constrained projection to robustify the ML prediction. Specifically, we prove that RCL is able to guarantee $(1+\lambda)$-competitiveness against any given expert for any $\lambda>0$, while also explicitly training the ML model in a robustification-aware manner to improve the average-case performance. Importantly, RCL is the first ML-augmented algorithm with a provable robustness guarantee in the case of multi-step switching cost and feedback delay. We demonstrate the improvement of RCL in both robustness and average performance using battery management as a case study.

----

## [0] Scalarization for Multi-Task and Multi-Domain Learning at Scale

**Authors**: *Amelie Royer, Tijmen Blankevoort, Babak Ehteshami Bejnordi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/368559ed8ede03b21f624feaeb3a5867-Abstract-Conference.html)

**Abstract**:

Training a single model on multiple input domains and/or output tasks allows for compressing information from multiple sources into a unified backbone hence improves model efficiency. It also enables potential positive knowledge transfer across tasks/domains, leading to improved accuracy and data-efficient training. However, optimizing such networks is a challenge, in particular due to discrepancies between the different tasks or domains: Despite several hypotheses and solutions proposed over the years, recent work has shown that uniform scalarization training, i.e., simply minimizing the average of the task losses,  yields on-par performance with more costly SotA optimization methods. This raises the issue of how well we understand the training dynamics of multi-task and multi-domain networks. In this work, we first devise a large-scale unified analysis of multi-domain and multi-task learning to better understand the dynamics of scalarization across varied task/domain combinations and model sizes. Following these insights, we then propose to leverage population-based training to efficiently search for the optimal scalarization weights when dealing with a large number of tasks or domains.

----

## [0] Causal discovery from observational and interventional data across multiple environments

**Authors**: *Adam Li, Amin Jaber, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/368cba57d00902c752eaa9e4770bbbbe-Abstract-Conference.html)

**Abstract**:

A fundamental problem in many sciences is the learning of causal structure underlying a system, typically through observation and experimentation. Commonly, one even collects data across multiple domains, such as gene sequencing from different labs, or neural recordings from different species. Although there exist methods for learning the equivalence class of causal diagrams from observational and experimental data, they are meant to operate in a single domain. In this paper, we develop a fundamental approach to structure learning in non-Markovian systems (i.e. when there exist latent confounders) leveraging observational and interventional data collected from multiple domains. Specifically, we start by showing that learning from observational data in multiple domains is equivalent to learning from interventional data with unknown targets in a single domain. But there are also subtleties when considering observational and experimental data. Using causal invariances derived from do-calculus, we define a property called S-Markov that connects interventional distributions from multiple-domains to graphical criteria on a selection diagram. Leveraging the S-Markov property, we introduce a new constraint-based causal discovery algorithm, S-FCI, that can learn from observational and interventional data from different domains. We prove that the algorithm is sound and subsumes existing constraint-based causal discovery algorithms.

----

## [0] Beyond Normal: On the Evaluation of Mutual Information Estimators

**Authors**: *Pawel Czyz, Frederic Grabowski, Julia E. Vogt, Niko Beerenwinkel, Alexander Marx*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36b80eae70ff629d667f210e13497edf-Abstract-Conference.html)

**Abstract**:

Mutual information is a general statistical dependency measure which has found applications in representation learning, causality, domain generalization and computational biology. However, mutual information estimators are typically evaluated on simple families of probability distributions, namely multivariate normal distribution and selected distributions with one-dimensional random variables. In this paper, we show how to construct a diverse family of distributions with known ground-truth mutual information and propose a language-independent benchmarking platform for mutual information estimators. We discuss the general applicability and limitations of classical and neural estimators in settings involving high dimensions, sparse interactions, long-tailed distributions, and high mutual information. Finally, we provide guidelines for practitioners on how to select appropriate estimator adapted to the difficulty of problem considered and issues one needs to consider when applying an estimator to a new data set.

----

## [0] Structured Semidefinite Programming for Recovering Structured Preconditioners

**Authors**: *Arun Jambulapati, Jerry Li, Christopher Musco, Kirankumar Shiragur, Aaron Sidford, Kevin Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html)

**Abstract**:

We develop a general framework for finding approximately-optimal preconditioners for solving linear systems. Leveraging this framework we obtain improved runtimes for fundamental preconditioning and linear system solving problems including:Diagonal preconditioning. We give an algorithm which, given positive definite $\mathbf{K} \in \mathbb{R}^{d \times d}$ with $\mathrm{nnz}(\mathbf{K})$ nonzero entries, computes an $\epsilon$-optimal diagonal preconditioner in time $\widetilde{O}(\mathrm{nnz}(\mathbf{K}) \cdot \mathrm{poly}(\kappa^\star,\epsilon^{-1}))$, where $\kappa^\star$ is the optimal condition number of the rescaled matrix.Structured linear systems. We give an algorithm which, given $\mathbf{M} \in \mathbb{R}^{d \times d}$ that is either the pseudoinverse of a graph Laplacian matrix or a constant spectral approximation of one, solves linear systems in $\mathbf{M}$ in $\widetilde{O}(d^2)$ time. Our diagonal preconditioning results improve state-of-the-art runtimes of $\Omega(d^{3.5})$ attained by general-purpose semidefinite programming, and our solvers improve state-of-the-art runtimes of $\Omega(d^{\omega})$ where $\omega > 2.3$ is the current matrix multiplication constant. We attain our results via new algorithms for a class of semidefinite programs (SDPs) we call matrix-dictionary approximation SDPs, which we leverage to solve an associated problem we call matrix-dictionary recovery.

----

## [0] Certifiably Robust Graph Contrastive Learning

**Authors**: *Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37050ebbbd7096719ab96cec19a4c69f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/37050ebbbd7096719ab96cec19a4c69f-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.

----

## [0] FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy

**Authors**: *Zuhao Yang, Yingfang Yuan, Yang Xu, Shuo Zhan, Huajun Bai, Kefan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37094fdc81632915a5738293cf9b7ad4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/37094fdc81632915a5738293cf9b7ad4-Abstract-Conference.html)

**Abstract**:

Measuring the distance between machine-produced and human language is a critical open problem. Inspired by empirical findings from psycholinguistics on the periodicity of entropy in language, we propose FACE, a set of metrics based on Fourier Analysis of the estimated Cross-Entropy of language, for measuring the similarity between model-generated and human-written languages. Based on an open-ended generation task and the experimental data from previous studies, we find that FACE can effectively identify the human-model gap, scales with model size, reflects the outcomes of different sampling methods for decoding, correlates well with other evaluation metrics and with human judgment scores.

----

## [0] 3D Copy-Paste: Physically Plausible Object Insertion for Monocular 3D Detection

**Authors**: *Yunhao Ge, Hong-Xing Yu, Cheng Zhao, Yuliang Guo, Xinyu Huang, Liu Ren, Laurent Itti, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html)

**Abstract**:

A major challenge in monocular 3D object detection is the limited diversity and quantity of objects in real datasets. While augmenting real scenes with virtual objects holds promise to improve both the diversity and quantity of the objects, it remains elusive due to the lack of an effective 3D object insertion method in complex real captured scenes. In this work, we study augmenting complex real indoor scenes with virtual objects for monocular 3D object detection. The main challenge is to automatically identify plausible physical properties for virtual assets (e.g., locations, appearances, sizes, etc.) in cluttered real scenes. To address this challenge, we propose a physically plausible indoor 3D object insertion approach to automatically copy virtual objects and paste them into real scenes. The resulting objects in scenes have 3D bounding boxes with plausible physical locations and appearances. In particular, our method first identifies physically feasible locations and poses for the inserted objects to prevent collisions with the existing room layout. Subsequently, it estimates spatially-varying illumination for the insertion location, enabling the immersive blending of the virtual objects into the original scene with plausible appearances and cast shadows. We show that our augmentation method significantly improves existing monocular 3D object models and achieves state-of-the-art performance. For the first time, we demonstrate that a physically plausible 3D object insertion, serving as a generative data augmentation technique, can lead to significant improvements for discriminative downstream tasks such as monocular 3D object detection. Project website: https://gyhandy.github.io/3D-Copy-Paste/.

----

## [0] Laughing Hyena Distillery: Extracting Compact Recurrences From Convolutions

**Authors**: *Stefano Massaroli, Michael Poli, Daniel Y. Fu, Hermann Kumbong, Rom N. Parnichkun, David W. Romero, Aman Timalsina, Quinn McIntyre, Beidi Chen, Atri Rudra, Ce Zhang, Christopher Ré, Stefano Ermon, Yoshua Bengio*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/371355cd42caaf83412c3fbef4688979-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/371355cd42caaf83412c3fbef4688979-Abstract-Conference.html)

**Abstract**:

Recent advances in attention-free sequence models rely on convolutions as alternatives to the attention operator at the core of Transformers. In particular, long convolution sequence models have achieved state-of-the-art performance in many domains, but incur a significant cost during auto-regressive inference workloads -- naively requiring a full pass (or caching of activations) over the input sequence for each generated token -- similarly to attention-based models. In this paper, we seek to enable $\mathcal O(1)$ compute and memory cost per token in any pre-trained long convolution architecture to reduce memory footprint and increase throughput during generation. Concretely, our methods consist in extracting low-dimensional linear state-space models from each convolution layer, building upon rational interpolation and model-order reduction techniques. We further introduce architectural improvements to convolution-based layers such as Hyena: by weight-tying the filters across channels into heads, we achieve higher pre-training quality and reduce the number of filters to be distilled. The resulting model achieves 10x higher throughput than Transformers and 1.5x higher than Hyena at 1.3B parameters, without any loss in quality after distillation.

----

## [0] Incomplete Multimodality-Diffused Emotion Recognition

**Authors**: *Yuanzhi Wang, Yong Li, Zhen Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/372cb7805eaccb2b7eed641271a30eec-Abstract-Conference.html)

**Abstract**:

Human multimodal emotion recognition (MER) aims to perceive and understand human emotions via various heterogeneous modalities, such as language, vision, and acoustic. Compared with unimodality, the complementary information in the multimodalities facilitates robust emotion understanding. Nevertheless,  in real-world scenarios, the missing modalities hinder multimodal understanding and result in degraded MER performance. In this paper, we propose an Incomplete Multimodality-Diffused emotion recognition (IMDer) method to mitigate the challenge of MER under incomplete multimodalities. To recover the missing modalities, IMDer exploits the score-based diffusion model that maps the input Gaussian noise into the desired distribution space of the missing modalities and recovers missing data abided by their original distributions. Specially, to reduce semantic ambiguity between the missing and the recovered modalities, the available modalities are embedded as the condition to guide and refine the diffusion-based recovering process. In contrast to previous work,  the diffusion-based modality recovery mechanism in IMDer allows to simultaneously reach both distribution consistency and semantic disambiguation. Feature visualization of the recovered modalities illustrates the consistent modality-specific distribution and semantic alignment. Besides, quantitative experimental results verify that IMDer obtains state-of-the-art MER accuracy under various missing modality patterns.

----

## [0] Diffusion-Based Probabilistic Uncertainty Estimation for Active Domain Adaptation

**Authors**: *Zhekai Du, Jingjing Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/374050dc3f211267bd6bf0ea24eae184-Abstract-Conference.html)

**Abstract**:

Active Domain Adaptation (ADA) has emerged as an attractive technique for assisting domain adaptation by actively annotating a small subset of target samples. Most ADA methods focus on measuring the target representativeness beyond traditional active learning criteria to handle the domain shift problem, while leaving the uncertainty estimation to be performed by an uncalibrated deterministic model. In this work, we introduce a probabilistic framework that captures both data-level and prediction-level uncertainties beyond a point estimate. Specifically, we use variational inference to approximate the joint posterior distribution of latent representation and model prediction. The variational objective of labeled data can be formulated by a variational autoencoder and a latent diffusion classifier, and the objective of unlabeled data can be implemented in a knowledge distillation framework. We utilize adversarial learning to ensure an invariant latent space. The resulting diffusion classifier enables efficient sampling of all possible predictions for each individual to recover the predictive distribution. We then leverage a t-test-based criterion upon the sampling and select informative unlabeled target samples based on the p-value, which encodes both prediction variability and cross-category ambiguity. Experiments on both ADA and Source-Free ADA settings show that our method provides more calibrated predictions than previous ADA methods and achieves favorable performance on three domain adaptation datasets.

----

## [0] Augmented Memory Replay-based Continual Learning Approaches for Network Intrusion Detection

**Authors**: *Suresh Kumar Amalapuram, Sumohana S. Channappayya, Bheemarjuna Reddy Tamma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3755a02b1035fbadd5f93a022170e46f-Abstract-Conference.html)

**Abstract**:

Intrusion detection is a form of anomalous activity detection in communication network traffic. Continual learning (CL) approaches to the intrusion detection task accumulate old knowledge while adapting to the latest threat knowledge. Previous works have shown the effectiveness of memory replay-based CL approaches for this task. In this work, we present two novel contributions to improve the performance of CL-based network intrusion detection in the context of class imbalance and scalability. First, we extend class balancing reservoir sampling (CBRS), a memory-based CL method, to address the problems of severe class imbalance for large datasets. Second, we propose a novel approach titled perturbation assistance for parameter approximation (PAPA) based on the Gaussian mixture model to reduce the number of \textit{virtual stochastic gradient descent (SGD) parameter} computations needed to discover maximally interfering samples for CL. We demonstrate that the proposed approaches perform remarkably better than the baselines on standard intrusion detection benchmarks created over shorter periods (KDDCUP'99, NSL-KDD, CICIDS-2017/2018, UNSW-NB15, and CTU-13) and a longer period with distribution shift (AnoShift). We also validated proposed approaches on standard continual learning benchmarks (SVHN, CIFAR-10/100, and CLEAR-10/100) and anomaly detection benchmarks (SMAP, SMD, and MSL). Further, the proposed PAPA approach significantly lowers the number of virtual SGD update operations, thus resulting in training time savings in the range of 12 to 40\% compared to the maximally interfered samples retrieval algorithm.

----

## [0] Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models

**Authors**: *Alvin Heng, Harold Soh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/376276a95781fa17c177b1ccdd0a03ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/376276a95781fa17c177b1ccdd0a03ac-Abstract-Conference.html)

**Abstract**:

The recent proliferation of large-scale text-to-image models has led to growing concerns that such models may be misused to generate harmful, misleading, and inappropriate content. Motivated by this issue, we derive a technique inspired by continual learning to selectively forget concepts in pretrained deep generative models. Our method, dubbed Selective Amnesia, enables controllable forgetting  where a user can specify how a concept should be forgotten. Selective Amnesia can be applied to conditional variational likelihood models, which encompass a variety of popular deep generative frameworks, including variational autoencoders and large-scale text-to-image diffusion models. Experiments across different models demonstrate that our approach induces forgetting on a variety of concepts, from entire classes in standard datasets to celebrity and nudity prompts in text-to-image models.

----

## [0] Structure from Duplicates: Neural Inverse Graphics from a Pile of Objects

**Authors**: *Tianhang Cheng, Wei-Chiu Ma, Kaiyu Guan, Antonio Torralba, Shenlong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3764a9c8abc84c7482f778fefc24f10b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3764a9c8abc84c7482f778fefc24f10b-Abstract-Conference.html)

**Abstract**:

Abstract Our world is full of identical objects (\emph{e.g.}, cans of coke, cars of same model). These duplicates, when seen together, provide additional and strong cues for us to effectively reason about 3D. Inspired by this observation, we introduce Structure from Duplicates (SfD), a novel inverse graphics framework that reconstructs geometry, material, and illumination from a single image containing multiple identical objects. SfD begins by identifying multiple instances of an object within an image, and then jointly estimates the 6DoF pose for all instances. An inverse graphics pipeline is subsequently employed to jointly reason about the shape, material of the object, and the environment light, while adhering to the shared geometry and material constraint across instances.Our primary contributions involve utilizing object duplicates as a robust prior for single-image inverse graphics and proposing an in-plane rotation-robust Structure from Motion (SfM) formulation for joint 6-DoF object pose estimation. By leveraging multi-view cues from a single image, SfD generates more realistic and detailed 3D reconstructions, significantly outperforming existing single image reconstruction models and multi-view reconstruction approaches with a similar or greater number of observations.

----

## [0] Weakly-Supervised Audio-Visual Segmentation

**Authors**: *Shentong Mo, Bhiksha Raj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/377b2e39e97e917b9e625b35241e33df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/377b2e39e97e917b9e625b35241e33df-Abstract-Conference.html)

**Abstract**:

Audio-visual segmentation is a challenging task that aims to predict pixel-level masks for sound sources in a video.  Previous work applied a comprehensive manually designed architecture with countless pixel-wise accurate masks as supervision.  However, these pixel-level masks are expensive and not available in all cases.  In this work, we aim to simplify the supervision as the instance-level annotation, $\textit{i.e.}$, weakly-supervised audio-visual segmentation.  We present a novel Weakly-Supervised Audio-Visual Segmentation framework, namely WS-AVS, that can learn multi-scale audio-visual alignment with multi-scale multiple-instance contrastive learning for audio-visual segmentation.  Extensive experiments on AVSBench demonstrate the effectiveness of our WS-AVS in the weakly-supervised audio-visual segmentation of single-source and multi-source scenarios.

----

## [0] Adversarial Examples Are Not Real Features

**Authors**: *Ang Li, Yifei Wang, Yiwen Guo, Yisen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/378b284f7f03274d1bf5322bb15c5c16-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/378b284f7f03274d1bf5322bb15c5c16-Abstract-Conference.html)

**Abstract**:

The existence of adversarial examples has been a mystery for years and attracted much interest. A well-known theory by \citet{ilyas2019adversarial} explains adversarial vulnerability from a data perspective by showing that one can extract non-robust features from adversarial examples and these features alone are useful for classification. However, the explanation remains quite counter-intuitive since non-robust features are mostly noise features to humans. In this paper, we re-examine the theory from a larger context by incorporating multiple learning paradigms. Notably, we find that contrary to their good usefulness under supervised learning, non-robust features attain poor usefulness when transferred to other self-supervised learning paradigms, such as contrastive learning, masked image modeling, and diffusion models. It reveals that non-robust features are not really as useful as robust or natural features that enjoy good transferability between these paradigms. Meanwhile, for robustness, we also show that naturally trained encoders from robust features are largely non-robust under AutoAttack. Our cross-paradigm examination suggests that the non-robust features are not really useful but more like paradigm-wise shortcuts, and robust features alone might be insufficient to attain reliable model robustness. Code is available at \url{https://github.com/PKU-ML/AdvNotRealFeatures}.

----

## [0] A Comprehensive Study on Text-attributed Graphs: Benchmarking and Rethinking

**Authors**: *Hao Yan, Chaozhuo Li, Ruosong Long, Chao Yan, Jianan Zhao, Wenwen Zhuang, Jun Yin, Peiyan Zhang, Weihao Han, Hao Sun, Weiwei Deng, Qi Zhang, Lichao Sun, Xing Xie, Senzhang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/37d00f567a18b478065f1a91b95622a0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Text-attributed graphs (TAGs) are prevalent in various real-world scenarios, where each node is associated with a text description. The cornerstone of representation learning on TAGs lies in the seamless integration of textual semantics within individual nodes and the topological connections across nodes. Recent advancements in pre-trained language models (PLMs) and graph neural networks (GNNs) have facilitated effective learning on TAGs, garnering increased research interest. However, the absence of meaningful benchmark datasets and standardized evaluation procedures for TAGs has impeded progress in this field. In this paper, we propose CS-TAG, a comprehensive and diverse collection of challenging benchmark datasets for TAGs. The CS-TAG datasets are notably large in scale and encompass a wide range of domains, spanning from citation networks to purchase graphs. In addition to building the datasets,  we conduct extensive benchmark experiments over CS-TAG with various learning paradigms, including PLMs, GNNs, PLM-GNN co-training methods, and the proposed novel topological pre-training of language models. In a nutshell, we provide an overview of the CS-TAG datasets, standardized evaluation procedures, and present baseline experiments. The entire CS-TAG project is publicly accessible at \url{https://github.com/sktsherlock/TAG-Benchmark}.

----

## [0] Online Ad Allocation with Predictions

**Authors**: *Fabian Spaeh, Alina Ene*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3815d62554efad0878fad6c1c30ffda0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3815d62554efad0878fad6c1c30ffda0-Abstract-Conference.html)

**Abstract**:

Display Ads and the generalized assignment problem are two well-studied online packing problems with important applications in ad allocation and other areas. In both problems, ad impressions arrive online and have to be allocated immediately to budget-constrained advertisers. Worst-case algorithms that achieve the ideal competitive ratio are known for both problems, but might act overly conservative given the predictable and usually tame nature of real-world input. Given this discrepancy, we develop an algorithm for both problems that incorporate machine-learned predictions and can thus improve the performance beyond the worst-case. Our algorithm is based on the work of Feldman et al. (2009) and similar in nature to Mahdian et al. (2007) who were the first to develop a learning-augmented algorithm for the related, but more structured Ad Words problem. We use a novel analysis to show that our algorithm is able to capitalize on a good prediction, while being robust against poor predictions. We experimentally evaluate our algorithm on synthetic and real-world data on a wide range of predictions. Our algorithm is consistently outperforming the worst-case algorithm without predictions.

----

## [0] Transfer Learning with Affine Model Transformation

**Authors**: *Shunya Minami, Kenji Fukumizu, Yoshihiro Hayashi, Ryo Yoshida*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3819a070922cc0d19f3d66ce108f28e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3819a070922cc0d19f3d66ce108f28e0-Abstract-Conference.html)

**Abstract**:

Supervised transfer learning has received considerable attention due to its potential to boost the predictive power of machine learning in scenarios where data are scarce. Generally, a given set of source models and a dataset from a target domain are used to adapt the pre-trained models to a target domain by statistically learning domain shift and domain-specific factors. While such procedurally and intuitively plausible methods have achieved great success in a wide range of real-world applications, the lack of a theoretical basis hinders further methodological development. This paper presents a general class of transfer learning regression called affine model transfer, following the principle of expected-square loss minimization. It is shown that the affine model transfer broadly encompasses various existing methods, including the most common procedure based on neural feature extractors. Furthermore, the current paper clarifies theoretical properties of the affine model transfer such as generalization error and excess risk. Through several case studies, we demonstrate the practical benefits of modeling and estimating inter-domain commonality and domain-specific factors separately with the affine-type transfer models.

----

## [0] Towards Robust and Expressive Whole-body Human Pose and Shape Estimation

**Authors**: *Hui En Pang, Zhongang Cai, Lei Yang, Qingyi Tao, Zhonghua Wu, Tianwei Zhang, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/381d36bf8e115cdeda48763c9cb77616-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/381d36bf8e115cdeda48763c9cb77616-Abstract-Conference.html)

**Abstract**:

Whole-body pose and shape estimation aims to jointly predict different behaviors (e.g., pose, hand gesture, facial expression) of the entire human body from a monocular image. Existing methods often exhibit suboptimal performance due to the complexity of in-the-wild scenarios. We argue that the prediction accuracy of these models is significantly affected by the quality of the bounding box, e.g., scale, alignment. The natural discrepancy between the ideal bounding box annotations and model detection results is particularly detrimental to the performance of whole-body pose and shape estimation.In this paper, we propose a novel framework to enhance the robustness of whole-body pose and shape estimation. Our framework incorporates three new modules to address the above challenges from three perspectives: (1) a Localization Module enhances the model's awareness of the subject's location and semantics within the image space; (2) a Contrastive Feature Extraction Module encourages the model to be invariant to robust augmentations by incorporating a contrastive loss and positive samples; (3) a Pixel Alignment Module ensures the reprojected mesh from the predicted camera and body model parameters are more accurate and pixel-aligned. We perform comprehensive experiments to demonstrate the effectiveness of our proposed framework on body, hands, face and whole-body benchmarks.

----

## [0] Neural Lad: A Neural Latent Dynamics Framework for Times Series Modeling

**Authors**: *Ting Li, Jianguo Li, Zhanxing Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/382a8606a85ca6ec7c06185a1a95ce8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/382a8606a85ca6ec7c06185a1a95ce8b-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equation (Neural ODE) is an elegant yet powerful framework to learn the temporal dynamics for time series modeling.However, we observe that existing Neural ODE forecasting models suffer from two disadvantages:i) controlling the latent states only through the linear transformation over the local change of the observed signals may be inadequate;ii) lacking the ability to capture the inherent periodical property in time series forecasting tasks;To overcome the two issues, we introduce a new neural ODE framework called \textbf{Neural Lad}, a \textbf{Neural} \textbf{La}tent \textbf{d}ynamics model in which the latent representations evolve with an ODE enhanced by the change of observed signal and seasonality-trend characterization. We incorporate the local change of input signal into the latent dynamics in an attention-based manner and design a residual architecture over basis expansion to depict the periodicity in the underlying dynamics. To accommodate the multivariate time series forecasting, we extend the  Neural Lad through learning an adaptive relationship between multiple time series.        Experiments demonstrate that our model can achieve better or comparable performance against existing neural ODE families and transformer variants in various datasets. Remarkably, the empirical superiority of Neural Lad is consistent across  short and long-horizon forecasting for both univariate, multivariate and even irregular sampled time series.

----

## [0] Weighted ROC Curve in Cost Space: Extending AUC to Cost-Sensitive Learning

**Authors**: *Huiyang Shao, Qianqian Xu, Zhiyong Yang, Peisong Wen, Peifeng Gao, Qingming Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38687a6a01f127d6db92561508a225b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38687a6a01f127d6db92561508a225b7-Abstract-Conference.html)

**Abstract**:

In this paper, we aim to tackle flexible cost requirements for long-tail datasets, where we need to construct a (a) cost-sensitive and (b) class-distribution robust learning framework. The misclassification cost and the area under the ROC curve (AUC) are popular metrics for (a) and (b), respectively. However, limited by their formulations, models trained with AUC cannot be applied to cost-sensitive decision problems, and models trained with fixed costs are sensitive to the class distribution shift. To address this issue, we present a new setting where costs are treated like a dataset to deal with arbitrarily unknown cost distributions. Moreover, we propose a novel weighted version of AUC where the cost distribution can be integrated into its calculation through decision thresholds.  To formulate this setting, we propose a novel bilevel paradigm to bridge weighted AUC (WAUC) and cost. The inner-level problem approximates the optimal threshold from sampling costs, and the outer-level problem minimizes the WAUC loss over the optimal threshold distribution. To optimize this bilevel paradigm, we employ a stochastic optimization algorithm (SACCL) to optimize it. Finally, experiment results show that our algorithm performs better than existing cost-sensitive learning methods and two-stage AUC decisions approach.

----

## [0] Dynamic Non-monotone Submodular Maximization

**Authors**: *Kiarash Banihashem, Leyla Biabani, Samira Goudarzi, MohammadTaghi Hajiaghayi, Peyman Jabbarzade, Morteza Monemizadeh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/387982dbf23d9975c7fc45813dd3dabc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/387982dbf23d9975c7fc45813dd3dabc-Abstract-Conference.html)

**Abstract**:

Maximizing submodular functions has been increasingly used in many applications of machine learning, such as data summarization, recommendation systems, and feature selection. Moreover, there has been a growing interest in both submodular maximization and dynamic algorithms. In 2020, Monemizadeh and Lattanzi, Mitrovic, Norouzi-Fard, Tarnawski, and Zadimoghaddam initiated developing dynamic algorithms for the monotone submodular maximization problem under the cardinality constraint $k$. In 2022, Chen and Peng studied the complexity of this problem and raised an important open question: "\emph{Can we extend [fully dynamic] results (algorithm or hardness) to non-monotone submodular maximization?}". We affirmatively answer their question by demonstrating a reduction from maximizing a non-monotone submodular function under the cardinality constraint $k$ to maximizing a monotone submodular function under the same constraint. Through this reduction, we obtain the first dynamic algorithms to solve the non-monotone submodular maximization problem under the cardinality constraint $k$. Our algorithms maintain an $(8+\epsilon)$-approximate of the solution and use expected amortized $O(\epsilon^{-3}k^3\log^3(n)\log(k))$ or $O(\epsilon^{-1}k^2\log^3(k))$ oracle queries per update, respectively. Furthermore, we showcase the benefits of our dynamic algorithm for video summarization and max-cut problems on several real-world data sets.

----

## [0] Semi-Implicit Denoising Diffusion Models (SIDDMs)

**Authors**: *Yanwu Xu, Mingming Gong, Shaoan Xie, Wei Wei, Matthias Grundmann, Kayhan Batmanghelich, Tingbo Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3882ca2c952276247fe9a993193b00e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3882ca2c952276247fe9a993193b00e4-Abstract-Conference.html)

**Abstract**:

Despite the proliferation of generative models, achieving fast sampling during inference without compromising sample diversity and quality remains challenging. Existing models such as Denoising Diffusion Probabilistic Models (DDPM) deliver high-quality, diverse samples but are slowed by an inherently high number of iterative steps. The Denoising Diffusion Generative Adversarial Networks (DDGAN) attempted to circumvent this limitation by integrating a GAN model for larger jumps in the diffusion process. However, DDGAN encountered scalability limitations when applied to large datasets. To address these limitations, we introduce a novel approach that tackles the problem by matching implicit and explicit factors. More specifically, our approach involves utilizing an implicit model to match the marginal distributions of noisy data and the explicit conditional distribution of the forward diffusion. This combination allows us to effectively match the joint denoising distributions. Unlike DDPM but similar to DDGAN, we do not enforce a parametric distribution for the reverse step, enabling us to take large steps during inference. Similar to the DDPM but unlike DDGAN, we take advantage of the exact form of the diffusion process. We demonstrate that our proposed method obtains comparable generative performance to diffusion-based models and vastly superior results to models with a small number of sampling steps.

----

## [0] Implicit Convolutional Kernels for Steerable CNNs

**Authors**: *Maksim Zhdanov, Nico Hoffmann, Gabriele Cesa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/389a55c90f839d58188060a42bb9138a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/389a55c90f839d58188060a42bb9138a-Abstract-Conference.html)

**Abstract**:

Steerable convolutional neural networks (CNNs) provide a general framework for building neural networks equivariant to translations and transformations of an origin-preserving group $G$, such as reflections and rotations. They rely on standard convolutions with $G$-steerable kernels obtained by analytically solving the group-specific equivariance constraint imposed onto the kernel space. As the solution is tailored to a particular group $G$, implementing a kernel basis does not generalize to other symmetry transformations, complicating the development of general group equivariant models. We propose using implicit neural representation via multi-layer perceptrons (MLPs) to parameterize $G$-steerable kernels. The resulting framework offers a simple and flexible way to implement Steerable CNNs and generalizes to any group $G$ for which a $G$-equivariant MLP can be built. We prove the effectiveness of our method on multiple tasks, including N-body simulations, point cloud classification and molecular property prediction.

----

## [0] Block Low-Rank Preconditioner with Shared Basis for Stochastic Optimization

**Authors**: *Jui-Nan Yen, Sai Surya Duvvuri, Inderjit S. Dhillon, Cho-Jui Hsieh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/389cfad711d2b1e2128e931feee80230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/389cfad711d2b1e2128e931feee80230-Abstract-Conference.html)

**Abstract**:

Adaptive methods with non-diagonal preconditioning have shown state-of-the-art results on various tasks. However, their computational complexity and memory requirement makes it challenging to scale these methods to modern neural network architectures. To address this challenge, some previous works have adopted block-diagonal preconditioners. However, the memory cost of storing the block-diagonal matrix remains substantial, leading to the use of smaller block sizes and ultimately resulting in suboptimal performance. To reduce the time and memory complexity without sacrificing performance, we propose approximating each diagonal block of the second moment matrix by low-rank matrices and enforcing the same basis for the blocks within each layer.  We provide theoretical justification for such sharing and design an algorithm to efficiently maintain this shared-basis block low-rank approximation during training. Our results on a deep autoencoder and a transformer benchmark demonstrate that the proposed method outperforms first-order methods with slightly more time and memory usage, while also achieving competitive or superior performance compared to other second-order methods with less time and memory usage.

----

## [0] Learning in the Presence of Low-dimensional Structure: A Spiked Random Matrix Perspective

**Authors**: *Jimmy Ba, Murat A. Erdogdu, Taiji Suzuki, Zhichao Wang, Denny Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38a1671ab0747b6ffe4d1c6ef117a3a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38a1671ab0747b6ffe4d1c6ef117a3a9-Abstract-Conference.html)

**Abstract**:

We consider the learning of a single-index target function $f_*: \mathbb{R}^d\to\mathbb{R}$ under spiked covariance data: $$f_*(\boldsymbol{x}) = \textstyle\sigma_*(\frac{1}{\sqrt{1+\theta}}\langle\boldsymbol{x},\boldsymbol{\mu}\rangle), ~~ \boldsymbol{x}\overset{\small\mathrm{i.i.d.}}{\sim}\mathcal{N}(0,\boldsymbol{I_d} + \theta\boldsymbol{\mu}\boldsymbol{\mu}^\top), ~~ \theta\asymp d^{\beta} \text{ for } \beta\in[0,1), $$ where the link function $\sigma_*:\mathbb{R}\to\mathbb{R}$ is a degree-$p$ polynomial with information exponent $k$ (defined as the lowest degree in the Hermite expansion of $\sigma_*$), and it depends on the projection of input $\boldsymbol{x}$ onto the spike (signal) direction $\boldsymbol{\mu}\in\mathbb{R}^d$. In the proportional asymptotic limit where the number of training examples $n$ and the dimensionality $d$ jointly diverge: $n,d\to\infty, n/d\to\psi\in(0,\infty)$, we ask the following question: how large should the spike magnitude $\theta$ (i.e., the strength of the low-dimensional component) be, in order for $(i)$ kernel methods, $(ii)$ neural networks optimized by gradient descent, to learn $f_*$? We show that for kernel ridge regression, $\beta\ge 1-\frac{1}{p}$ is both sufficient and necessary. Whereas for two-layer neural networks trained with gradient descent, $\beta>1-\frac{1}{k}$ suffices. Our results demonstrate that both kernel methods and neural networks benefit from low-dimensional structures in the data. Further, since $k\le p$ by definition, neural networks can adapt to such structures more effectively.

----

## [0] Efficient Neural Music Generation

**Authors**: *Max W. Y. Lam, Qiao Tian, Tang Li, Zongyu Yin, Siyuan Feng, Ming Tu, Yuliang Ji, Rui Xia, Mingbo Ma, Xuchen Song, Jitong Chen, Yuping Wang, Yuxuan Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38b23e2328096520e9c889ae03e372c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38b23e2328096520e9c889ae03e372c9-Abstract-Conference.html)

**Abstract**:

Recent progress in music generation has been remarkably advanced by the state-of-the-art MusicLM, which comprises a hierarchy of three LMs, respectively, for semantic, coarse acoustic, and fine acoustic modelings. Yet, sampling with the MusicLM requires processing through these LMs one by one to obtain the fine-grained acoustic tokens, making it computationally expensive and prohibitive for a real-time generation. Efficient music generation with a quality on par with MusicLM remains a significant challenge.In this paper, we present MeLoDy (M for music; L for LM; D for diffusion), an LM-guided diffusion model that generates music audios of state-of-the-art quality meanwhile reducing 95.7\% to 99.6\% forward passes in MusicLM, respectively, for sampling 10s to 30s music. MeLoDy inherits the highest-level LM from MusicLM for semantic modeling, and applies a novel dual-path diffusion (DPD) model and an audio VAE-GAN to efficiently decode the conditioning semantic tokens into waveform. DPD is proposed to simultaneously model the coarse and fine acoustics by incorporating the semantic information into segments of latents effectively via cross-attention at each denoising step. Our experimental results suggest the superiority of MeLoDy, not only in its practical advantages on sampling speed and infinitely continuable generation, but also in its state-of-the-art musicality, audio quality, and text correlation.Our samples are available at https://Efficient-MeLoDy.github.io/.

----

## [0] Crystal Structure Prediction by Joint Equivariant Diffusion

**Authors**: *Rui Jiao, Wenbing Huang, Peijia Lin, Jiaqi Han, Pin Chen, Yutong Lu, Yang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html)

**Abstract**:

Crystal Structure Prediction (CSP) is crucial in various scientific disciplines. While CSP can be addressed by employing currently-prevailing generative models (e.g. diffusion models), this task encounters unique challenges owing to the symmetric geometry of crystal structures---the invariance of translation, rotation, and periodicity. To incorporate the above symmetries, this paper proposes DiffCSP, a novel diffusion model to learn the structure distribution from stable crystals. To be specific, DiffCSP jointly generates the lattice and atom coordinates for each crystal by employing a periodic-E(3)-equivariant denoising model, to better model the crystal geometry. Notably, different from related equivariant generative approaches, DiffCSP leverages fractional coordinates other than Cartesian coordinates to represent crystals, remarkably promoting the diffusion and the generation process of atom positions. Extensive experiments verify that our DiffCSP remarkably outperforms existing CSP methods, with a much lower computation cost in contrast to DFT-based methods. Moreover, the superiority of DiffCSP is still observed when it is extended for ab initio crystal generation.

----

## [0] Understanding the detrimental class-level effects of data augmentation

**Authors**: *Polina Kirichenko, Mark Ibrahim, Randall Balestriero, Diane Bouchacourt, Shanmukha Ramakrishna Vedantam, Hamed Firooz, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38c05a5410a6ab7eeeb26c9dbebbc41b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38c05a5410a6ab7eeeb26c9dbebbc41b-Abstract-Conference.html)

**Abstract**:

Data augmentation (DA) encodes invariance and provides implicit regularization critical to a model's performance in image classification tasks. However, while DA improves average accuracy, recent studies have shown that its impact can be highly class dependent: achieving optimal average accuracy comes at the cost of significantly hurting individual class accuracy by as much as 20% on ImageNet. There has been little progress in resolving class-level accuracy drops due to a limited understanding of these effects. In this work, we present a framework for understanding how DA interacts with class-level learning dynamics. Using higher-quality multi-label annotations on ImageNet, we systematically categorize the affected classes and find that the majority are inherently ambiguous, co-occur, or involve fine-grained distinctions, while DA controls the model's bias towards one of the closely related classes. While many of the previously reported performance drops are explained by multi-label annotations, we identify other sources of accuracy degradations by analyzing class confusions. We show that simple class-conditional augmentation strategies informed by our framework improve performance on the negatively affected classes.

----

## [0] Optimal Guarantees for Algorithmic Reproducibility and Gradient Complexity in Convex Optimization

**Authors**: *Liang Zhang, Junchi Yang, Amin Karbasi, Niao He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38c5feed4b72c96f6cf925ccc9832ecf-Abstract-Conference.html)

**Abstract**:

Algorithmic reproducibility measures the deviation in outputs of machine learning algorithms upon minor changes in the training process. Previous work suggests that first-order methods would need to trade-off convergence rate (gradient complexity) for better reproducibility. In this work, we challenge this perception and demonstrate that both optimal reproducibility and near-optimal convergence guarantees can be achieved for smooth convex minimization and smooth convex-concave minimax problems under various error-prone oracle settings. Particularly, given the inexact initialization oracle, our regularization-based algorithms achieve the best of both worlds -- optimal reproducibility and near-optimal gradient complexity -- for minimization and minimax optimization. With the inexact gradient oracle, the near-optimal guarantees also hold for minimax optimization. Additionally, with the stochastic gradient oracle, we show that stochastic gradient descent ascent is optimal in terms of both reproducibility and gradient complexity. We believe our results contribute to an enhanced understanding of the reproducibility-convergence trade-off in the context of convex optimization.

----

## [0] Test-time Adaptation of Discriminative Models via Diffusion Generative Feedback

**Authors**: *Mihir Prabhudesai, Tsung-Wei Ke, Alexander C. Li, Deepak Pathak, Katerina Fragkiadaki*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38e511a690709603d4cc3a1c52b4a9fd-Abstract-Conference.html)

**Abstract**:

The advancements in generative modeling, particularly the advent of diffusion models, have sparked a fundamental question: how can these models be effectively used for discriminative tasks? In this work, we find that generative models can be great test-time adapters for discriminative models. Our method, Diffusion-TTA, adapts pre-trained discriminative models such as image classifiers, segmenters and depth predictors, to each unlabelled example in the test set using generative feedback from a diffusion model. We achieve this by modulating the conditioning of the diffusion model using the output of the discriminative model. We then maximize the image likelihood objective by backpropagating the gradients to discriminative modelâ€™s parameters. We show Diffusion-TTA significantly enhances the accuracy of various large-scale pre-trained discriminative models, such as, ImageNet classifiers, CLIP models, image pixel labellers and image depth predictors. Diffusion-TTA outperforms existing test-time adaptation methods, including TTT-MAE and TENT, and particularly shines in online adaptation setups, where the discriminative model is continually adapted to each example in the test set. We provide access to code, results, and visualizations on our website: diffusion-tta.github.io/

----

## [0] Fragment-based Pretraining and Finetuning on Molecular Graphs

**Authors**: *Kha-Dinh Luong, Ambuj K. Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38ec60a949c3538e5cbb337b1b386dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38ec60a949c3538e5cbb337b1b386dcf-Abstract-Conference.html)

**Abstract**:

Property prediction on molecular graphs is an important application of Graph Neural Networks (GNNs). Recently, unlabeled molecular data has become abundant, which facilitates the rapid development of self-supervised learning for GNNs in the chemical domain. In this work, we propose pretraining GNNs at the fragment level, a promising middle ground to overcome the limitations of node-level and graph-level pretraining. Borrowing techniques from recent work on principal subgraph mining, we obtain a compact vocabulary of prevalent fragments from a large pretraining dataset. From the extracted vocabulary, we introduce several fragment-based contrastive and predictive pretraining tasks. The contrastive learning task jointly pretrains two different GNNs: one on molecular graphs and the other on fragment graphs, which represents higher-order connectivity within molecules. By enforcing consistency between the fragment embedding and the aggregated embedding of the corresponding atoms from the molecular graphs, we ensure that the embeddings capture structural information at multiple resolutions. The structural information of fragment graphs is further exploited to extract auxiliary labels for graph-level predictive pretraining. We employ both the pretrained molecular-based and fragment-based GNNs for downstream prediction, thus utilizing the fragment information during finetuning. Our graph fragment-based pretraining (GraphFP) advances the performances on 5 out of 8 common molecular benchmarks and improves the performances on long-range biological benchmarks by at least 11.5%. Code is available at: https://github.com/lvkd84/GraphFP.

----

## [0] Characterizing Out-of-Distribution Error via Optimal Transport

**Authors**: *Yuzhe Lu, Yilong Qin, Runtian Zhai, Andrew Shen, Ketong Chen, Zhenlin Wang, Soheil Kolouri, Simon Stepputtis, Joseph Campbell, Katia P. Sycara*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/38fd51cf36f28566230a93a5fbeaabbf-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) data poses serious challenges in deployed machine learning models,so methods of predicting a model's performance on OOD data without labels are important for machine learning safety.While a number of methods have been proposed by prior work, they often underestimate the actual error, sometimes by a large margin, which greatly impacts their applicability to real tasks. In this work, we identify pseudo-label shift, or the difference between the predicted and true OOD label distributions, as a key indicator of this underestimation. Based on this observation, we introduce a novel method for estimating model performance by leveraging optimal transport theory, Confidence Optimal Transport (COT), and show that it provably provides more robust error estimates in the presence of pseudo-label shift. Additionally, we introduce an empirically-motivated variant of COT, Confidence Optimal Transport with Thresholding (COTT), which applies thresholding to the individual transport costs and further improves the accuracy of COT's error estimates. We evaluate COT and COTT on a variety of standard benchmarks that induce various types of distribution shift -- synthetic, novel subpopulation, and natural -- and show that our approaches significantly outperform existing state-of-the-art methods with up to 3x lower prediction errors.

----

## [0] Unsupervised Video Domain Adaptation for Action Recognition: A Disentanglement Perspective

**Authors**: *Pengfei Wei, Lingdong Kong, Xinghua Qu, Yi Ren, Zhiqiang Xu, Jing Jiang, Xiang Yin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39235c56aef13fb05a6adc95eb9d8d66-Abstract-Conference.html)

**Abstract**:

Unsupervised video domain adaptation is a practical yet challenging task. In this work, for the first time, we tackle it from a disentanglement view. Our key idea is to handle the spatial and temporal domain divergence separately through disentanglement. Specifically, we consider the generation of cross-domain videos from two sets of latent factors, one encoding the static information and another encoding the dynamic information. A Transfer Sequential VAE (TranSVAE) framework is then developed to model such generation. To better serve for adaptation, we propose several objectives to constrain the latent factors. With these constraints, the spatial divergence can be readily removed by disentangling the static domain-specific information out, and the temporal divergence is further reduced from both frame- and video-levels through adversarial learning. Extensive experiments on the UCF-HMDB, Jester, and Epic-Kitchens datasets verify the effectiveness and superiority of TranSVAE compared with several state-of-the-art approaches.

----

## [0] Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs Knowledge Editing in Language Models

**Authors**: *Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)

**Abstract**:

Language models learn a great quantity of factual information during pretraining, and recent work localizes this information to specific model weights like mid-layer MLP weights. In this paper, we find that we can change how a fact is stored in a model by editing weights that are in a different location than where existing methods suggest that the fact is stored. This is surprising because we would expect that localizing facts to specific model parameters would tell us where to manipulate knowledge in models, and this assumption has motivated past work on model editing methods. Specifically, we show that localization conclusions from representation denoising (also known as Causal Tracing) do not provide any insight into which model MLP layer would be best to edit in order to override an existing stored fact with a new one. This finding raises questions about how past work relies on Causal Tracing to select which model layers to edit. Next, we consider several variants of the editing problem, including erasing and amplifying facts. For one of our editing problems, editing performance does relate to localization results from representation denoising, but we find that which layer we edit is a far better predictor of performance. Our results suggest, counterintuitively, that better mechanistic understanding of how pretrained language models work may not always translate to insights about how to best change their behavior.

----

## [0] The Geometry of Neural Nets' Parameter Spaces Under Reparametrization

**Authors**: *Agustinus Kristiadi, Felix Dangel, Philipp Hennig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/395371f778ebd4854b88521100af30ad-Abstract-Conference.html)

**Abstract**:

Model reparametrization, which follows the change-of-variable rule of calculus, is a popular way to improve the training of neural nets. But it can also be problematic since it can induce inconsistencies in, e.g., Hessian-based flatness measures, optimization trajectories, and modes of probability densities. This complicates downstream analyses: e.g. one cannot definitively relate flatness with generalization since arbitrary reparametrization changes their relationship. In this work, we study the invariance of neural nets under reparametrization from the perspective of Riemannian geometry. From this point of view, invariance is an inherent property of any neural net if one explicitly represents the metric and uses the correct associated transformation rules. This is important since although the metric is always present, it is often implicitly assumed as identity, and thus dropped from the notation, then lost under reparametrization. We discuss implications for measuring the flatness of minima, optimization, and for probability-density maximization. Finally, we explore some interesting directions where invariance is useful.

----

## [0] A Dataset of Relighted 3D Interacting Hands

**Authors**: *Gyeongsik Moon, Shunsuke Saito, Weipeng Xu, Rohan Joshi, Julia Buffalini, Harley Bellan, Nicholas Rosen, Jesse Richardson, Mallorie Mize, Philippe de Bree, Tomas Simon, Bo Peng, Shubham Garg, Kevyn McPhail, Takaaki Shiratori*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/396beafa6feba781a7114780e6837253-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/396beafa6feba781a7114780e6837253-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The two-hand interaction is one of the most challenging signals to analyze due to the self-similarity, complicated articulations, and occlusions of hands. Although several datasets have been proposed for the two-hand interaction analysis, all of them do not achieve 1) diverse and realistic image appearances and 2) diverse and large-scale groundtruth (GT) 3D poses at the same time. In this work, we propose Re:InterHand, a dataset of relighted 3D interacting hands that achieve the two goals. To this end, we employ a state-of-the-art hand relighting network with our accurately tracked two-hand 3D poses. We compare our Re:InterHand with existing 3D interacting hands datasets and show the benefit of it. Our Re:InterHand is available in https://mks0601.github.io/ReInterHand/

----

## [0] Multi-Objective Intrinsic Reward Learning for Conversational Recommender Systems

**Authors**: *Zhendong Chu, Nan Wang, Hongning Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/396ea38391e8b96a3add6126006f1a53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/396ea38391e8b96a3add6126006f1a53-Abstract-Conference.html)

**Abstract**:

Conversational Recommender Systems (CRS) actively elicit user preferences to generate adaptive recommendations. Mainstream reinforcement learning-based CRS solutions heavily rely on handcrafted reward functions, which may not be aligned with user intent in CRS tasks. Therefore, the design of task-specific rewards is critical to facilitate CRS policy learning, which remains largely under-explored in the literature. In this work, we propose a novel approach to address this challenge by learning intrinsic rewards from interactions with users. Specifically, we formulate intrinsic reward learning as a multi-objective bi-level optimization problem. The inner level optimizes the CRS policy augmented by the learned intrinsic rewards, while the outer level drives the intrinsic rewards to optimize two CRS-specific objectives: maximizing the success rate and minimizing the number of turns to reach a successful recommendation}in conversations. To evaluate the effectiveness of our approach, we conduct extensive experiments on three public CRS benchmarks. The results show that our algorithm significantly improves CRS performance by exploiting informative learned intrinsic rewards.

----

## [0] Predict-then-Calibrate: A New Perspective of Robust Contextual LP

**Authors**: *Chunlin Sun, Linyu Liu, Xiaocheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html)

**Abstract**:

Contextual optimization, also known as predict-then-optimize or prescriptive analytics, considers an optimization problem with the presence of covariates (context or side information). The goal is to learn a prediction model (from the training data) that predicts the objective function from the covariates, and then in the test phase, solve the optimization problem with the covariates but without the observation of the objective function. In this paper, we consider a risk-sensitive version of the problem and propose a generic algorithm design paradigm called predict-then-calibrate. The idea is to first develop a prediction model without concern for the downstream risk profile or robustness guarantee, and then utilize calibration (or recalibration) methods to quantify the uncertainty of the prediction. While the existing methods suffer from either a restricted choice of the prediction model or strong assumptions on the underlying data, we show the disentangling of the prediction model and the calibration/uncertainty quantification has several advantages. First, it imposes no restriction on the prediction model and thus fully unleashes the potential of off-the-shelf machine learning methods. Second, the derivation of the risk and robustness guarantee can be made independent of the choice of the prediction model through a data-splitting idea. Third, our paradigm of predict-then-calibrate applies to both (risk-sensitive) robust and (risk-neutral) distributionally robust optimization (DRO) formulations. Theoretically, it gives new generalization bounds for the contextual LP problem and sheds light on the existing results of DRO for contextual LP. Numerical experiments further reinforce the advantage of the predict-then-calibrate paradigm in that an improvement on either the prediction model or the calibration model will lead to a better final performance.

----

## [0] INSPECT: A Multimodal Dataset for Patient Outcome Prediction of Pulmonary Embolisms

**Authors**: *Shih-Cheng Huang, Zepeng Huo, Ethan Steinberg, Chia-Chun Chiang, Curtis P. Langlotz, Matthew P. Lungren, Serena Yeung, Nigam Shah, Jason Alan Fries*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39736af1b9d87a1fddad9f84a6bcf64c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39736af1b9d87a1fddad9f84a6bcf64c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthesizing information from various data sources plays a crucial role in the practice of modern medicine. Current applications of artificial intelligence in medicine often focus on single-modality data due to a lack of publicly available, multimodal medical datasets. To address this limitation, we introduce INSPECT, which contains de-identified longitudinal records from a large cohort of pulmonary embolism (PE) patients, along with ground truth labels for multiple outcomes. INSPECT contains data from 19,402 patients, including CT images, sections of radiology reports, and structured electronic health record (EHR) data (including demographics, diagnoses, procedures, and vitals). Using our provided dataset, we develop and release a benchmark for evaluating several baseline modeling approaches on a variety of important PE related tasks. We evaluate image-only, EHR-only, and fused models. Trained models and the de-identified dataset are made available for non-commercial use under a data use agreement. To the best our knowledge, INSPECT is the largest multimodal dataset for enabling reproducible research on strategies for integrating 3D medical imaging and EHR data.

----

## [0] What Makes Good Examples for Visual In-Context Learning?

**Authors**: *Yuanhan Zhang, Kaiyang Zhou, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/398ae57ed4fda79d0781c65c926d667b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/398ae57ed4fda79d0781c65c926d667b-Abstract-Conference.html)

**Abstract**:

Large vision models with billions of parameters and trained on broad data have great potential in numerous downstream applications. However, these models are typically difficult to adapt due to their large parameter size and sometimes lack of accesss to their weights---entities able to develop large vision models often provide APIs only. In this paper, we study how to better utilize large vision models through the lens of in-context learning, a concept that has been well-known in natural language processing but has only been studied very recently in computer vision. In-context learning refers to the ability to perform inference on tasks never seen during training by simply conditioning on in-context examples (i.e., input-output pairs) without updating any internal model parameters. To demystify in-context learning in computer vision, we conduct an extensive research and identify a critical problem: downstream performance is highly sensitivie to the choice of visual in-context examples. To address this problem, we propose a prompt retrieval framework specifically for large vision models, allowing the selection of in-context examples to be fully automated. Concretely, we provide two implementations: (i) an unsupervised prompt retrieval method based on nearest example search using an off-the-shelf model, and (ii) a supervised prompt retrieval method, which trains a neural network to choose examples that directly maximize in-context learning performance. Both methods do not require access to the internal weights of large vision models. Our results demonstrate that our methods can bring non-trivial improvements to visual in-context learning in comparison to the commonly-used random selection. Code and models will be released.

----

## [0] Parameterizing Context: Unleashing the Power of Parameter-Efficient Fine-Tuning and In-Context Tuning for Continual Table Semantic Parsing

**Authors**: *Yongrui Chen, Shenyu Zhang, Guilin Qi, Xinnan Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/398b00a05b847ac65eb98c8e5e865fe8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/398b00a05b847ac65eb98c8e5e865fe8-Abstract-Conference.html)

**Abstract**:

Continual table semantic parsing aims to train a parser on a sequence of tasks, where each task requires the parser to translate natural language into SQL based on task-specific tables but only offers limited training examples. Conventional methods tend to suffer from overfitting with limited supervision, as well as catastrophic forgetting due to parameter updates.Despite recent advancements that partially alleviate these issues through semi-supervised data augmentation and retention of a few past examples, the performance is still limited by the volume of unsupervised data and stored examples.To overcome these challenges, this paper introduces a novel method integrating parameter-efficient fine-tuning (PEFT) and in-context tuning (ICT) for training a continual table semantic parser. Initially, we present a task-adaptive PEFT framework capable of fully circumventing catastrophic forgetting, which is achieved by freezing the pre-trained model backbone and fine-tuning small-scale prompts. Building on this, we propose a teacher-student framework-based solution. The teacher addresses the few-shot problem using ICT, which procures contextual information by demonstrating a few training examples. In turn, the student leverages the proposed PEFT framework to learn from the teacher's output distribution, and subsequently compresses and saves the contextual information to the prompts, eliminating the need to store any training examples.Experimental evaluations on two benchmarks affirm the superiority of our method over prevalent few-shot and continual learning baselines across various metrics.

----

## [0] Incentives in Federated Learning: Equilibria, Dynamics, and Mechanisms for Welfare Maximization

**Authors**: *Aniket Murhekar, Zhuowen Yuan, Bhaskar Ray Chaudhury, Bo Li, Ruta Mehta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39b77b5e422b4e070e2811b73ea9bcf7-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) has emerged as a powerful scheme to facilitate the collaborative learning of models amongst a set of agents holding their own private data.  Although the agents benefit from the global model trained on shared data, by participating in federated learning, they may also incur costs (related to privacy and communication) due to data sharing. In this paper, we model a collaborative FL framework, where every agent attempts to achieve an optimal trade-off between her learning payoff and data sharing cost. We show the existence of Nash equilibrium (NE) under mild assumptions on agents' payoff and costs. Furthermore, we show that agents can discover the NE via best response dynamics. However, some of the NE may be bad in terms of overall welfare for the agents, implying little incentive for some fraction of the agents to participate in the learning. To remedy this, we design a budget-balanced mechanism involving payments to the agents, that ensures that any $p$-mean welfare function of the agents' utilities is maximized at NE. In addition, we introduce a FL protocol FedBR-BG that incorporates our budget-balanced mechanism, utilizing best response dynamics. Our empirical validation on MNIST and CIFAR-10 substantiates our theoretical analysis. We show that FedBR-BG outperforms the basic best-response-based protocol without additional incentivization, the standard federated learning protocol FedAvg, as well as a recent baseline MWFed in terms of achieving superior $p$-mean welfare.

----

## [0] MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

**Authors**: *Mohammad Mozaffari, Sikan Li, Zhao Zhang, Maryam Mehri Dehnavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39bc6e3cbf5a1991d33dc10ebff9a9cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39bc6e3cbf5a1991d33dc10ebff9a9cf-Abstract-Conference.html)

**Abstract**:

This work proposes a Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 updates, called MKOR, that improves the training time and convergence properties of deep neural networks (DNNs). Second-order techniques, while enjoying higher convergence rates vs first-order counterparts, have cubic complexity with respect to either the model size and/or the training batch size. Hence they exhibit poor scalability and performance in transformer models, e.g. large language models (LLMs), because the batch sizes in these models  scale by the attention mechanism sequence length, leading to large model size and batch sizes. MKOR's complexity is quadratic with respect to the model size, alleviating the computation bottlenecks in  second-order methods. Because of their high computation complexity, state-of-the-art implementations of second-order methods can only afford to update the second order information infrequently, and thus do not fully exploit the promise of better convergence from these updates. By reducing the communication complexity of the second-order updates as well as achieving a linear communication complexity, MKOR increases the frequency of second order updates. We also propose a hybrid version of MKOR (called MKOR-H) that mid-training falls backs to a first order optimizer if the second order updates  no longer accelerate convergence.  Our experiments show that MKOR outperforms state -of-the-art first order methods, e.g. the LAMB optimizer, and best implementations of second-order methods, i.e. KAISA/KFAC, up to 2.57x and 1.85x respectively on BERT-Large-Uncased on 64 GPUs.

----

## [0] DFRD: Data-Free Robustness Distillation for Heterogeneous Federated Learning

**Authors**: *Kangyang Luo, Shuai Wang, Yexuan Fu, Xiang Li, Yunshi Lan, Ming Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is a privacy-constrained decentralized machine learning paradigm in which clients enable collaborative training without compromising private data. However, how to learn a robust global model in the data-heterogeneous and model-heterogeneous FL scenarios is challenging. To address it, we resort to data-free knowledge distillation to propose a new FL method (namely DFRD).DFRD equips a conditional generator on the server to approximate the training space of the local models uploaded by clients, and systematically investigates its training in terms of fidelity, transferability and diversity. To overcome the catastrophic forgetting of the global model caused by the distribution shifts of the generator across communication rounds, we maintain an exponential moving average copy of the generator on the server.  Additionally, we propose dynamic weighting and label sampling to accurately extract knowledge from local models. Finally, our extensive experiments on various image classification tasks illustrate that DFRD achieves significant performance gains compared to SOTA baselines.

----

## [0] SG×P : A Sorghum Genotype × Phenotype Prediction Dataset and Benchmark

**Authors**: *Zeyu Zhang, Robert Pless, Nadia Shakoor, Austin Carnahan, Abby Stylianou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39d02e8e23bafadd7cd405f2281bc05c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39d02e8e23bafadd7cd405f2281bc05c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large scale field-phenotyping approaches have the potential to solve important questions about the relationship of plant genotype to plant phenotype.  Computational approaches to measuring the phenotype (the observable plant features) are required to address the problem at a large scale, but machine learning approaches to extract phenotypes from sensor data have been hampered by limited access to (a) sufficiently large, organized multi-sensor datasets, (b) field trials that have a large scale and significant number of genotypes, (c) full genetic sequencing of those phenotypes, and (d) datasets sufficiently organized so that algorithm centered researchers can directly address the real biological problems.  To address this, we present SGxP, a novel benchmark dataset from a large-scale field trial consisting of the complete genotype of over 300 sorghum varieties, and time sequences of imagery from several field plots growing each variety, taken with RGB and laser 3D scanner imaging.  To lower the barrier to entry and facilitate further developments, we provide a set of well organized, multi-sensor imagery and corresponding genomic data.  We implement baseline deep learning based phenotyping approaches to create baseline results for individual sensors and multi-sensor fusion for detecting genetic mutations with known impacts.  We also provide and support an open-ended challenge by identifying thousands of genetic mutations whose phenotypic impacts are currently unknown.  A web interface for machine learning researchers and practitioners to share approaches, visualizations and hypotheses supports engagement with plant biologists to further the understanding of the sorghum genotype x phenotype relationship. The full dataset, leaderboard (including baseline results) and discussion forums can be found at http://sorghumsnpbenchmark.com.

----

## [0] Rank-N-Contrast: Learning Continuous Representations for Regression

**Authors**: *Kaiwen Zha, Peng Cao, Jeany Son, Yuzhe Yang, Dina Katabi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)

**Abstract**:

Deep regression models typically learn in an end-to-end fashion without explicitly emphasizing a regression-aware representation. Consequently, the learned representations exhibit fragmentation and fail to capture the continuous nature of sample orders, inducing suboptimal results across a wide range of regression tasks. To fill the gap, we propose Rank-N-Contrast (RNC), a framework that learns continuous representations for regression by contrasting samples against each other based on their rankings in the target space. We demonstrate, theoretically and empirically, that RNC guarantees the desired order of learned representations in accordance with the target orders, enjoying not only better performance but also significantly improved robustness, efficiency, and generalization. Extensive experiments using five real-world regression datasets that span computer vision, human-computer interaction, and healthcare verify that RNC achieves state-of-the-art performance, highlighting its intriguing properties including better data efficiency, robustness to spurious targets and data corruptions, and generalization to distribution shifts.

----

## [0] OpenGSL: A Comprehensive Benchmark for Graph Structure Learning

**Authors**: *Zhiyao Zhou, Sheng Zhou, Bochao Mao, Xuanyi Zhou, Jiawei Chen, Qiaoyu Tan, Daochen Zha, Yan Feng, Chun Chen, Can Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/39f8ef62e061042cca8c8f46d7e0e31b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/39f8ef62e061042cca8c8f46d7e0e31b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as the de facto standard for representation learning on graphs, owing to their ability to effectively integrate graph topology and node attributes. However, the inherent suboptimal nature of node connections, resulting from the complex and contingent formation process of graphs, presents significant challenges in modeling them effectively. To tackle this issue, Graph Structure Learning (GSL), a family of data-centric learning approaches, has garnered substantial attention in recent years. The core concept behind GSL is to jointly optimize the graph structure and the corresponding GNN models. Despite the proposal of numerous GSL methods, the progress in this field remains unclear due to inconsistent experimental protocols, including variations in datasets, data processing techniques, and splitting strategies. In this paper, we introduce OpenGSL, the first comprehensive benchmark for GSL, aimed at addressing this gap. OpenGSL enables a fair comparison among state-of-the-art GSL methods by evaluating them across various popular datasets using uniform data processing and splitting strategies. Through extensive experiments, we observe that existing GSL methods do not consistently outperform vanilla GNN counterparts. We also find that there is no significant correlation between the homophily of the learned structure and task performance, challenging the common belief. Moreover, we observe that the learned graph structure demonstrates a strong generalization ability across different GNN models, despite the high computational and space consumption. We hope that our open-sourced library will facilitate rapid and equitable evaluation and inspire further innovative research in this field. The code of the benchmark can be found in https://github.com/OpenGSL/OpenGSL.

----

## [0] Global Optimality in Bivariate Gradient-based DAG Learning

**Authors**: *Chang Deng, Kevin Bello, Pradeep Ravikumar, Bryon Aragam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a02b6df276223b68c69ca572cb3c4a8-Abstract-Conference.html)

**Abstract**:

Recently, a new class of non-convex optimization problems motivated by the statistical problem of learning an acyclic directed graphical model from data has attracted significant interest. While existing work uses standard first-order optimization schemes to solve this problem, proving the global optimality of such approaches has proven elusive. The difficulty lies in the fact that unlike other non-convex problems in the literature, this problem is not "benign", and possesses multiple spurious solutions that standard approaches can easily get trapped in. In this paper, we prove that a simple path-following optimization scheme globally converges to the global minimum of the population loss in the bivariate setting.

----

## [0] Perceptual adjustment queries and an inverted measurement paradigm for low-rank metric learning

**Authors**: *Austin Xu, Andrew D. McRae, Jingyan Wang, Mark A. Davenport, Ashwin Pananjady*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a07c3a67cfe50d3236b71fb674c7f30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a07c3a67cfe50d3236b71fb674c7f30-Abstract-Conference.html)

**Abstract**:

We introduce a new type of query mechanism for collecting human feedback, called the perceptual adjustment query (PAQ). Being both informative and cognitively lightweight, the PAQ adopts an inverted measurement scheme, and combines advantages from both cardinal and ordinal queries. We showcase the PAQ in the metric learning problem, where we collect PAQ measurements to learn an unknown Mahalanobis distance. This gives rise to a high-dimensional, low-rank matrix estimation problem to which standard matrix estimators cannot be applied. Consequently, we develop a two-stage estimator for metric learning from PAQs, and provide sample complexity guarantees for this estimator. We present numerical simulations demonstrating the performance of the estimator and its notable properties.

----

## [0] Joint processing of linguistic properties in brains and language models

**Authors**: *Subba Reddy Oota, Manish Gupta, Mariya Toneva*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a0e2de215bd17c39ad08ba1d16c1b12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a0e2de215bd17c39ad08ba1d16c1b12-Abstract-Conference.html)

**Abstract**:

Language models have been shown to be very effective in predicting brain recordings of subjects experiencing complex language stimuli. For a deeper understanding of this alignment, it is important to understand the correspondence between the detailed processing of linguistic information by the human brain versus language models. We investigate this correspondence via a direct approach, in which we eliminate information related to specific linguistic properties in the language model representations and observe how this intervention affects the alignment with fMRI brain recordings obtained while participants listened to a story. We investigate a range of linguistic properties (surface, syntactic, and semantic) and find that the elimination of each one results in a significant decrease in brain alignment. Specifically, we find that syntactic properties (i.e. Top Constituents and Tree Depth) have the largest effect on the trend of brain alignment across model layers. These findings provide clear evidence for the role of specific linguistic information in the alignment between brain and language models, and open new avenues for mapping the joint information processing in both systems. We make the code publicly available https://github.com/subbareddy248/lingprop-brain-alignment.

----

## [0] S3: Increasing GPU Utilization during Generative Inference for Higher Throughput

**Authors**: *Yunho Jin, Chun-Feng Wu, David Brooks, Gu-Yeon Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a13be0c5dae69e0f08065f113fb10b8-Abstract-Conference.html)

**Abstract**:

Generating texts with a large language model (LLM) consumes massive amounts of memory. Apart from the already-large model parameters, the key/value (KV) cache that holds information about previous tokens in a sequence can grow to be even larger than the model itself. This problem is exacerbated in one of the current LLM serving frameworks which reserves the maximum sequence length of memory for the KV cache to guarantee generating a complete sequence as they do not know the output sequence length. This restricts us to use a smaller batch size leading to lower GPU utilization and above all, lower throughput. We argue that designing a system with a priori knowledge of the output sequence can mitigate this problem. To this end, we propose $S^3$, which predicts the output sequence length, schedules generation queries based on the prediction to increase device resource utilization and throughput, and handle mispredictions. Our proposed method achieves 6.49Ã— throughput over those systems that assume the worst case for the output sequence length.

----

## [0] Disentangling Cognitive Diagnosis with Limited Exercise Labels

**Authors**: *Xiangzhi Chen, Le Wu, Fei Liu, Lei Chen, Kun Zhang, Richang Hong, Meng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a14ae9951e8153a8fc814b5f506b5b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a14ae9951e8153a8fc814b5f506b5b7-Abstract-Conference.html)

**Abstract**:

Cognitive diagnosis is an important task in intelligence education, which aims at measuring studentsâ€™ proficiency in specific knowledge concepts. Given a fully labeled exercise-concept matrix, most existing models focused on mining students' response records for cognitive diagnosis. Despite their success, due to the huge cost of labeling exercises, a more practical scenario is that limited exercises are labeled with concepts. Performing cognitive diagnosis with limited exercise labels is under-explored and remains pretty much open. In this paper, we propose Disentanglement based Cognitive Diagnosis (DCD) to address the challenges of limited exercise labels. Specifically, we utilize students' response records to model student proficiency, exercise difficulty and exercise label distribution.   Then, we introduce two novel modules - group-based disentanglement and limited-labeled alignment modules - to disentangle the factors relevant to concepts and align them with real limited labels.  Particularly, we introduce the tree-like structure of concepts with negligible cost for group-based disentangling, as concepts of different levels exhibit different independence relationships.Extensive experiments on widely used benchmarks demonstrate the superiority of our proposed model.

----

## [0] Energy-Based Sliced Wasserstein Distance

**Authors**: *Khai Nguyen, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a23caeb904c822575fa56fb114ca499-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a23caeb904c822575fa56fb114ca499-Abstract-Conference.html)

**Abstract**:

The sliced Wasserstein (SW) distance has been widely recognized as a statistically effective and computationally efficient metric between two probability measures. A key component of the SW distance is the slicing distribution. There are two existing approaches for choosing this distribution. The first approach is using a fixed prior distribution. The second approach is optimizing for the best distribution which belongs to a parametric family of distributions and can maximize the expected distance. However, both approaches have their limitations. A fixed prior distribution is non-informative in terms of highlighting projecting directions that can discriminate two general probability measures. Doing optimization for the best distribution is often expensive and unstable. Moreover, designing the parametric family of the candidate distribution could be easily misspecified. To address the issues, we propose to design the slicing distribution as an energy-based distribution that is parameter-free and has the density proportional to an energy function of the projected one-dimensional Wasserstein distance. We then derive a novel sliced Wasserstein variant, energy-based sliced Waserstein (EBSW) distance, and investigate its topological, statistical, and computational properties via importance sampling, sampling importance resampling, and Markov Chain methods. Finally, we conduct experiments on point-cloud gradient flow, color transfer, and point-cloud reconstruction to show the favorable performance of the EBSW.

----

## [0] E2PNet: Event to Point Cloud Registration with Spatio-Temporal Representation Learning

**Authors**: *Xiuhong Lin, Changjie Qiu, Zhipeng Cai, Siqi Shen, Yu Zang, Weiquan Liu, Xuesheng Bian, Matthias Müller, Cheng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a2d1bf9bc0a9794cf82c1341a7a75e6-Abstract-Conference.html)

**Abstract**:

Event cameras have emerged as a promising vision sensor in recent years due to their unparalleled temporal resolution and dynamic range. While registration of 2D RGB images to 3D point clouds is a long-standing problem in computer vision, no prior work studies 2D-3D registration for event cameras. To this end, we propose E2PNet, the first learning-based method for event-to-point cloud registration.The core of E2PNet is a novel feature representation network called Event-Points-to-Tensor (EP2T), which encodes event data into a 2D grid-shaped feature tensor. This grid-shaped feature enables matured RGB-based frameworks to be easily used for event-to-point cloud registration, without changing hyper-parameters and the training procedure. EP2T treats the event input as spatio-temporal point clouds. Unlike standard 3D learning architectures that treat all dimensions of point clouds equally, the novel sampling and information aggregation modules in EP2T are designed to handle the inhomogeneity of the spatial and temporal dimensions. Experiments on the MVSEC and VECtor datasets demonstrate the superiority of E2PNet over hand-crafted and other learning-based methods. Compared to RGB-based registration, E2PNet is more robust to extreme illumination or fast motion due to the use of event data. Beyond 2D-3D registration, we also show the potential of EP2T for other vision tasks such as flow estimation, event-to-image reconstruction and object recognition. The source code can be found at: https://github.com/Xmu-qcj/E2PNet.

----

## [0] Pengi: An Audio Language Model for Audio Tasks

**Authors**: *Soham Deshmukh, Benjamin Elizalde, Rita Singh, Huaming Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a2e5889b4bbef997ddb13b55d5acf77-Abstract-Conference.html)

**Abstract**:

In the domain of audio processing, Transfer Learning has facilitated the rise of Self-Supervised Learning and Zero-Shot Learning techniques. These approaches have led to the development of versatile models capable of tackling a wide array of tasks, while delivering state-of-the-art performance. However, current models inherently lack the capacity to produce the requisite language for open-ended tasks, such as Audio Captioning or Audio Question Answering. We introduce Pengi, a novel Audio Language Model that leverages Transfer Learning by framing all audio tasks as text-generation tasks. It takes as input, an audio recording, and text, and generates free-form text as output. The input audio is represented as a sequence of continuous embeddings by an audio encoder. A text encoder does the same for the corresponding text input. Both sequences are combined as a prefix to prompt a pre-trained frozen language model. The unified architecture of Pengi enables open-ended tasks and close-ended tasks without any additional fine-tuning or task-specific extensions. When evaluated on 21 downstream tasks, our approach yields state-of-the-art performance in several of them. Our results show that connecting language models with audio models is a major step towards general-purpose audio understanding.

----

## [0] Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift

**Authors**: *Yongduo Sui, Qitian Wu, Jiancan Wu, Qing Cui, Longfei Li, Jun Zhou, Xiang Wang, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a33ddacb2798fc7d83b8334d552e05a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a33ddacb2798fc7d83b8334d552e05a-Abstract-Conference.html)

**Abstract**:

The issue of distribution shifts is emerging as a critical concern in graph representation learning. From the perspective of invariant learning and stable learning, a recently well-established paradigm for out-of-distribution generalization, stable features of the graph are assumed to causally determine labels, while environmental features tend to be unstable and can lead to the two primary types of distribution shifts. The correlation shift is often caused by the spurious correlation between environmental features and labels that differs between the training and test data; the covariate shift often stems from the presence of new environmental features in test data. However, most strategies, such as invariant learning or graph augmentation, typically struggle with limited training environments or perturbed stable features, thus exposing limitations in handling the problem of covariate shift. To address this challenge, we propose a simple-yet-effective data augmentation strategy, Adversarial Invariant Augmentation (AIA), to handle the covariate shift on graphs. Specifically, given the training data, AIA aims to extrapolate and generate new environments, while concurrently preserving the original stable features during the augmentation process. Such a design equips the graph classification model with an enhanced capability to identify stable features in new environments, thereby effectively tackling the covariate shift in data. Extensive experiments with in-depth empirical analysis demonstrate the superiority of our approach. The implementation codes are publicly available at https://github.com/yongduosui/AIA.

----

## [0] Adaptive recurrent vision performs zero-shot computation scaling to unseen difficulty levels

**Authors**: *Vijay Veerabadran, Srinivas Ravishankar, Yuan Tang, Ritik Raina, Virginia de Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a40e042c66e84659249f3254460c123-Abstract-Conference.html)

**Abstract**:

Humans solving algorithmic (or) reasoning problems typically exhibit solution times that grow as a function of problem difficulty. Adaptive recurrent neural networks have been shown to exhibit this property for various language-processing tasks. However, little work has been performed to assess whether such adaptive computation can also enable vision models to extrapolate solutions beyond their training distribution's difficulty level, with prior work focusing on very simple tasks. In this study, we investigate a critical functional role of such adaptive processing using recurrent neural networks: to dynamically scale computational resources conditional on input requirements that allow for zero-shot generalization to novel difficulty levels not seen during training using two challenging visual reasoning tasks: PathFinder and Mazes. We combine convolutional recurrent neural networks (ConvRNNs) with a learnable halting mechanism based on Graves (2016). We explore various implementations of such adaptive ConvRNNs (AdRNNs) ranging from tying weights across layers to more sophisticated biologically inspired recurrent networks that possess lateral connections and gating. We show that 1) AdRNNs learn to dynamically halt processing early (or late) to solve easier (or harder) problems, 2) these RNNs zero-shot generalize to more difficult problem settings not shown during training by dynamically increasing the number of recurrent iterations at test time. Our study provides modeling evidence supporting the hypothesis that recurrent processing enables the functional advantage of adaptively allocating compute resources conditional on input requirements and hence allowing generalization to harder difficulty levels of a visual reasoning problem without training.

----

## [0] NurViD: A Large Expert-Level Video Database for Nursing Procedure Activity Understanding

**Authors**: *Ming Hu, Lin Wang, Siyuan Yan, Don Ma, Qingli Ren, Peng Xia, Wei Feng, Peibo Duan, Lie Ju, Zongyuan Ge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a48b0eaba26ba862220a307a9edb0bb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The application of deep learning to nursing procedure activity understanding has the potential to greatly enhance the quality and safety of nurse-patient interactions. By utilizing the technique, we can facilitate training and education, improve quality control, and enable operational compliance monitoring. However, the development of automatic recognition systems in this field is currently hindered by the scarcity of appropriately labeled datasets. The existing video datasets pose several limitations: 1) these datasets are small-scale in size to support comprehensive investigations of nursing activity; 2) they primarily focus on single procedures, lacking expert-level annotations for various nursing procedures and action steps; and 3) they lack temporally localized annotations, which prevents the effective localization of targeted actions within longer video sequences. To mitigate these limitations, we propose NurViD, a large video dataset with expert-level annotation for nursing procedure activity understanding. NurViD consists of over 1.5k videos totaling 144 hours, making it approximately four times longer than the existing largest nursing activity datasets. Notably, it encompasses 51 distinct nursing procedures and 177 action steps, providing a much more comprehensive coverage compared to existing datasets that primarily focus on limited procedures. To evaluate the efficacy of current deep learning methods on nursing activity understanding, we establish three benchmarks on NurViD: procedure recognition on untrimmed videos, procedure and action recognition on trimmed videos, and action detection. Our benchmark and code will be available at https://github.com/minghu0830/NurViD-benchmark.

----

## [0] The Pick-to-Learn Algorithm: Empowering Compression for Tight Generalization Bounds and Improved Post-training Performance

**Authors**: *Dario Paccagnan, Marco C. Campi, Simone Garatti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a4f287883609241031e6818bd01133e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a4f287883609241031e6818bd01133e-Abstract-Conference.html)

**Abstract**:

Generalization bounds are valuable both for theory and applications. On the one hand, they shed light on the mechanisms that underpin the learning processes; on the other, they certify how well a learned model performs against unseen inputs.  In this work we build upon a recent breakthrough in compression theory to develop a new framework yielding tight generalization bounds of wide practical applicability.  The core idea is to embed any given learning algorithm into a suitably-constructed meta-algorithm (here called Pick-to-Learn, P2L) in order to instill desirable compression properties. When applied to the MNIST classification dataset and to a synthetic regression problem, P2L not only attains generalization bounds that compare favorably with the state of the art (test-set and PAC-Bayes bounds), but it also learns models with better post-training performance.

----

## [0] Orthogonal Non-negative Tensor Factorization based Multi-view Clustering

**Authors**: *Jing Li, Quanxue Gao, Qianqian Wang, Ming Yang, Wei Xia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a5b75ce6cbd3aaaa32d6e935ffc4cff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a5b75ce6cbd3aaaa32d6e935ffc4cff-Abstract-Conference.html)

**Abstract**:

Multi-view clustering (MVC) based on non-negative matrix factorization (NMF) and its variants have attracted much attention due to their advantages in clustering interpretability. However, existing NMF-based multi-view clustering methods perform NMF on each view respectively and ignore the impact of between-view. Thus, they can't well exploit the within-view spatial structure and between-view complementary information. To resolve this issue, we present orthogonal non-negative tensor factorization (Orth-NTF) and develop a novel multi-view clustering based on Orth-NTF with one-side orthogonal constraint. Our model directly performs Orth-NTF on the 3rd-order tensor which is composed of anchor graphs of views. Thus, our model directly considers the between-view relationship. Moreover, we use the tensor Schatten $p$-norm regularization as a rank approximation of the 3rd-order tensor which characterizes the cluster structure of multi-view data and exploits the between-view complementary information. In addition, we provide an optimization algorithm for the proposed method and prove mathematically that the algorithm always converges to the stationary KKT point. Extensive experiments on various benchmark datasets indicate that our proposed method is able to achieve satisfactory clustering performance.

----



[Go to the previous page](NIPS-2023-list700.md)

[Go to the next page](NIPS-2023-list702.md)

[Go to the catalog section](README.md)