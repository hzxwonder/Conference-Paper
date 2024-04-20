## [0] Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback

**Authors**: *Jaskirat Singh, Liang Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfd0bd56e8a6f82d1619f5d093d5f9ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfd0bd56e8a6f82d1619f5d093d5f9ca-Abstract-Conference.html)

**Abstract**:

The field of text-conditioned image generation has made unparalleled progress with the recent advent of latent diffusion models. While revolutionary, as the complexity of given text input increases, the current state of art diffusion models may still fail in generating images that accurately convey the semantics of the given prompt. Furthermore, such misalignments are often left undetected by pretrained multi-modal models such as CLIP.  To address these problems, in this paper, we explore a simple yet effective decompositional approach towards both evaluation and improvement of text-to-image alignment. In particular,  we first introduce a Decompositional-Alignment-Score which given a complex caption decomposes it into a set of disjoint assertions. The alignment of each assertion with generated images is then measured using a VQA model. Finally, alignment scores for different assertions are combined aposteriori to give the final text-to-image alignment score. Experimental analysis reveals that the proposed alignment metric shows a significantly higher correlation with human ratings as opposed to traditional CLIP, BLIP scores. Furthermore, we also find that the assertion level alignment scores also provide useful feedback which can then be used in a simple iterative procedure to gradually increase the expressivity of different assertions in the final image outputs. Human user studies indicate that the proposed approach surpasses previous state-of-the-art by 8.7% in overall text-to-image alignment accuracy.

----

## [0] Distributed Personalized Empirical Risk Minimization

**Authors**: *Yuyang Deng, Mohammad Mahdi Kamani, Pouria Mahdavinia, Mehrdad Mahdavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfee09496a5a8b0b01d9d4c589758832-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfee09496a5a8b0b01d9d4c589758832-Abstract-Conference.html)

**Abstract**:

This paper advocates a new paradigm  Personalized Empirical Risk Minimization (PERM) to facilitate learning from heterogeneous  data sources without imposing stringent constraints on computational resources shared by participating devices. In PERM, we aim at  learning  a distinct model for each client by personalizing the aggregation of local empirical losses by effectively estimating the  statistical discrepancy among data distributions, which entails  optimal statistical accuracy for all local distributions and overcomes the data heterogeneity issue.  To learn personalized models at scale,  we propose a distributed algorithm that replaces the standard model averaging with model shuffling to simultaneously optimize PERM objectives for all devices. This also allows to learn distinct model architectures (e.g., neural networks with different number of parameters) for  different clients, thus confining to  underlying  memory and compute resources of individual clients. We rigorously analyze the convergence of proposed algorithm and  conduct experiments  that corroborates the effectiveness of proposed paradigm.

----

## [0] Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis

**Authors**: *Zhiyu Jin, Xuli Shen, Bin Li, Xiangyang Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0378e0c642b1d292fcb224e8d5a39b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0378e0c642b1d292fcb224e8d5a39b3-Abstract-Conference.html)

**Abstract**:

Diffusion models (DMs) have recently gained attention with state-of-the-art performance in text-to-image synthesis. Abiding by the tradition in deep learning, DMs are trained and evaluated on the images with fixed sizes. However, users are demanding for various images with specific sizes and various aspect ratio. This paper focuses on adapting text-to-image diffusion models to handle such variety while maintaining visual fidelity. First we observe that, during the synthesis, lower resolution images suffer from incomplete object portrayal, while higher resolution images exhibit repetitively disordered presentation. Next, we establish a statistical relationship indicating that attention entropy changes with token quantity, suggesting that models aggregate spatial information in proportion to image resolution. The subsequent interpretation on our observations is that objects are incompletely depicted due to limited spatial information for low resolutions, while repetitively disorganized presentation arises from redundant spatial information for high resolutions. From this perspective, we propose a scaling factor to alleviate the change of attention entropy and mitigate the defective pattern observed. Extensive experimental results validate the efficacy of the proposed scaling factor, enabling models to achieve better visual effects, image quality, and text alignment. Notably, these improvements are achieved without additional training or fine-tuning techniques.

----

## [0] Enhancing Sharpness-Aware Optimization Through Variance Suppression

**Authors**: *Bingcong Li, Georgios B. Giannakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) has well documented merits in enhancing generalization of deep neural networks, even without sizable data augmentation. Embracing the geometry of the loss function, where neighborhoods of 'flat minima' heighten generalization ability, SAM seeks 'flat valleys' by minimizing the maximum loss caused by an adversary perturbing parameters within the neighborhood.Although critical to account for sharpness of the loss function, such an 'over-friendly adversary' can curtail the outmost level of generalization. The novel approach of this contribution fosters stabilization of adversaries through variance suppression (VaSSO) to avoid such friendliness. VaSSO's provable stability safeguards its numerical improvement over SAM in model-agnostic tasks, including image classification and machine translation. In addition, experiments confirm that VaSSO endows SAM with robustness against high levels of label noise. Code is available at https://github.com/BingcongLi/VaSSO.

----

## [0] Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards

**Authors**: *Bo Xue, Yimu Wang, Yuanyu Wan, Jinfeng Yi, Lijun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html)

**Abstract**:

This paper investigates the problem of generalized linear bandits with heavy-tailed rewards, whose $(1+\epsilon)$-th moment is bounded for some $\epsilon\in (0,1]$. Although there exist methods for generalized linear bandits, most of them focus on bounded or sub-Gaussian rewards and are not well-suited for many real-world scenarios, such as financial markets and web-advertising. To address this issue, we propose two novel algorithms based on truncation and mean of medians. These algorithms achieve an almost optimal regret bound of $\widetilde{O}(dT^{\frac{1}{1+\epsilon}})$, where $d$ is the dimension of contextual information and $T$ is the time horizon. Our truncation-based algorithm supports online learning, distinguishing it from existing truncation-based approaches. Additionally, our mean-of-medians-based algorithm requires only $O(\log T)$ rewards and one estimator per epoch, making it more practical. Moreover, our algorithms improve the regret bounds by a logarithmic factor compared to existing algorithms when $\epsilon=1$. Numerical experimental results confirm the merits of our algorithms.

----

## [0] Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions

**Authors**: *Kai Tan, Pierre C. Bellec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0ac27bf3327c9cb99cc5f548db4f73a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0ac27bf3327c9cb99cc5f548db4f73a-Abstract-Conference.html)

**Abstract**:

This paper investigates the asymptotic distribution of the maximum-likelihood estimate (MLE) in multinomial logistic models in the high-dimensional regime where dimension and sample size are of the same order. While classical large-sample theory provides asymptotic normality of the MLE under certain conditions, such classical results are expected to fail in high-dimensions as documented for the binary logistic case in the seminal work of Sur and Cand√®s [2019]. We address this issue in classification problems with 3 or more classes, by developing asymptotic normality and asymptotic chi-square results for the multinomial logistic MLE (also known as cross-entropy minimizer) on null covariates. Our theory leads to a new methodology to test the significance of a given feature. Extensive simulation studies on synthetic data corroborate these asymptotic results and confirm the validity of proposed p-values for testing the significance of a given feature.

----

## [0] Why think step by step? Reasoning emerges from the locality of experience

**Authors**: *Ben Prystawski, Michael Li, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0af79ad53a336b4c4b4f7e2a68eb609-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0af79ad53a336b4c4b4f7e2a68eb609-Abstract-Conference.html)

**Abstract**:

Humans have a powerful and mysterious capacity to reason. Working through a set of mental steps enables us to make inferences we would not be capable of making directly even though we get no additional data from the world. Similarly, when large language models generate intermediate steps (a chain of thought) before answering a question, they often produce better answers than they would directly. We investigate why and how chain-of-thought reasoning is useful in language models, testing the hypothesis that reasoning is effective when training data consists of overlapping local clusters of variables that influence each other strongly. These training conditions enable the chaining of accurate local inferences to estimate relationships between variables that were not seen together in training. We prove that there will exist a "reasoning gap", where reasoning through intermediate variables reduces bias, for the simple case of an autoregressive density estimator trained on local samples from a chain-structured probabilistic model. We then test our hypothesis experimentally in more complex models, training an autoregressive language model on samples from Bayes nets but only including a subset of variables in each sample. We test language modelsâ€™ ability to match conditional probabilities with and without intermediate reasoning steps, finding that intermediate steps are only helpful when the training data is locally structured with respect to dependencies between variables. The combination of locally structured observations and reasoning is much more data-efficient than training on all variables. Our results illustrate how the effectiveness of reasoning step by step is rooted in the local statistical structure of the training data.

----

## [0] Analyzing Generalization of Neural Networks through Loss Path Kernels

**Authors**: *Yilan Chen, Wei Huang, Hao Wang, Charlotte Loh, Akash Srivastava, Lam M. Nguyen, Lily Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0b6f389739496e363a89155c9448a8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0b6f389739496e363a89155c9448a8a-Abstract-Conference.html)

**Abstract**:

Deep neural networks have been increasingly used in real-world applications, making it critical to ensure their ability to adapt to new, unseen data. In this paper, we study the generalization capability of neural networks trained with (stochastic) gradient flow. We establish a new connection between the loss dynamics of gradient flow and general kernel machines by proposing a new kernel, called loss path kernel. This kernel measures the similarity between two data points by evaluating the agreement between loss gradients along the path determined by the gradient flow. Based on this connection, we derive a new generalization upper bound that applies to general neural network architectures. This new bound is tight and strongly correlated with the true generalization error. We apply our results to guide the design of neural architecture search (NAS) and demonstrate favorable performance compared with state-of-the-art NAS algorithms through numerical experiments.

----

## [0] Operation-Level Early Stopping for Robustifying Differentiable NAS

**Authors**: *Shen Jiang, Zipeng Ji, Guanghui Zhu, Chunfeng Yuan, Yihua Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0bc6dbcbcc957b2aeadb20c39ba7f05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0bc6dbcbcc957b2aeadb20c39ba7f05-Abstract-Conference.html)

**Abstract**:

Differentiable NAS (DARTS) is a simple and efficient neural architecture search method that has been extensively adopted in various machine learning tasks.% Nevertheless, DARTS still encounters several robustness issues, mainly the domination of skip connections.% The resulting architectures are full of parametric-free operations, leading to performance collapse.% Existing methods suggest that the skip connection has additional advantages in optimization compared to other parametric operations and propose to alleviate the domination of skip connections by eliminating these additional advantages.% In this paper, we analyze this issue from a simple and straightforward perspective and propose that the domination of skip connections results from parametric operations overfitting the training data while architecture parameters are trained on the validation data, leading to undesired behaviors.% Based on this observation, we propose the operation-level early stopping (OLES) method to overcome this issue and robustify DARTS without introducing any computation overhead.% Extensive experimental results can verify our hypothesis and the effectiveness of OLES.

----

## [0] Training on Foveated Images Improves Robustness to Adversarial Attacks

**Authors**: *Muhammad Shah, Aqsa Kashaf, Bhiksha Raj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0c256700465c158de71081b4cf5e8c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0c256700465c158de71081b4cf5e8c3-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) have been shown to be vulnerable to adversarial attacks-- subtle,  perceptually indistinguishable perturbations of inputs that change the response of the model. In the context of vision, we hypothesize that an important contributor to the robustness of human visual perception is constant exposure to low-fidelity visual stimuli in our peripheral vision. To investigate this hypothesis, we develop RBlur, an image transform that simulates the loss in fidelity of peripheral vision by blurring the image and reducing its color saturation based on the distance from a given fixation point. We show that compared to DNNs trained on the original images, DNNs trained on images transformed by RBlur are substantially more robust to adversarial attacks, as well as other, non-adversarial, corruptions, achieving up to 25% higher accuracy on perturbed data.

----

## [0] Label Poisoning is All You Need

**Authors**: *Rishi D. Jha, Jonathan Hayase, Sewoong Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html)

**Abstract**:

In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.

----

## [0] Learning Trajectories are Generalization Indicators

**Authors**: *Jingwen Fu, Zhizheng Zhang, Dacheng Yin, Yan Lu, Nanning Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0da54d3dbc0107692da952358965f5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0da54d3dbc0107692da952358965f5f-Abstract-Conference.html)

**Abstract**:

This paper explores the connection between learning trajectories of Deep Neural Networks (DNNs) and their generalization capabilities when optimized using (stochastic) gradient descent algorithms. Instead of concentrating solely on the generalization error of the DNN post-training, we present a novel perspective for analyzing generalization error by investigating the contribution of each update step to the change in generalization error. This perspective enable a more direct comprehension of how the learning trajectory influences generalization error. Building upon this analysis, we propose a new generalization bound that incorporates more extensive trajectory information.Our proposed generalization bound depends on the complexity of learning trajectory and the ratio between the bias and diversity of training set. Experimental observations reveal that our method effectively captures the generalization error throughout the training process. Furthermore, our approach can also track changes in generalization error when adjustments are made to learning rates and label noise levels. These results demonstrate that learning trajectory information is a valuable indicator of a model's generalization capabilities.

----

## [0] CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection

**Authors**: *Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e10a6a906ef323efaf708f76cf3c1d1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e10a6a906ef323efaf708f76cf3c1d1e-Abstract-Conference.html)

**Abstract**:

Deriving reliable region-word alignment from image-text pairs is critical to learnobject-level vision-language representations for open-vocabulary object detection.Existing methods typically rely on pre-trained or self-trained vision-languagemodels for alignment, which are prone to limitations in localization accuracy orgeneralization capabilities. In this paper, we propose CoDet, a novel approachthat overcomes the reliance on pre-aligned vision-language space by reformulatingregion-word alignment as a co-occurring object discovery problem. Intuitively, bygrouping images that mention a shared concept in their captions, objects corresponding to the shared concept shall exhibit high co-occurrence among the group.CoDet then leverages visual similarities to discover the co-occurring objects andalign them with the shared concept. Extensive experiments demonstrate that CoDethas superior performances and compelling scalability in open-vocabulary detection,e.g., by scaling up the visual backbone, CoDet achieves 37.0 $AP^m_{novel}$ and 44.7 $AP^m_{all}$ on OV-LVIS, surpassing the previous SoTA by 4.2 $AP^m_{novel}$ and 9.8 $AP^m_{all}$. Code is available at https://github.com/CVMI-Lab/CoDet.

----

## [0] Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards

**Authors**: *Alexandre Ramé, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, Matthieu Cord*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e12a3b98b67e8395f639fde4c2b03168-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e12a3b98b67e8395f639fde4c2b03168-Abstract-Conference.html)

**Abstract**:

Foundation models are first pre-trained on vast unsupervised datasets and then fine-tuned on labeled data. Reinforcement learning, notably from human feedback (RLHF), can further align the network with the intended usage. Yet the imperfections in the proxy reward may hinder the training and lead to suboptimal results; the diversity of objectives in real-world tasks and human opinions exacerbate the issue. This paper proposes embracing the heterogeneity of diverse rewards by following a multi-policy strategy. Rather than focusing on a single a priori reward, we aim for Pareto-optimal generalization across the entire space of preferences. To this end, we propose rewarded soup, first specializing multiple networks independently (one for each proxy reward) and then interpolating their weights linearly. This succeeds empirically because we show that the weights remain linearly connected when fine-tuned on diverse rewards from a shared pre-trained initialization. We demonstrate the effectiveness of our approach for text-to-text (summarization, Q&A, helpful assistant, review), text-image (image captioning, text-to-image generation, visual grounding), and control (locomotion) tasks. We hope to enhance the alignment of deep models, and how they interact with the world in all its diversity.

----

## [0] Optimal Block-wise Asymmetric Graph Construction for Graph-based Semi-supervised Learning

**Authors**: *Zixing Song, Yifei Zhang, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html)

**Abstract**:

Graph-based semi-supervised learning (GSSL) serves as a powerful tool to model the underlying manifold structures of samples in high-dimensional spaces. It involves two phases: constructing an affinity graph from available data and inferring labels for unlabeled nodes on this graph. While numerous algorithms have been developed for label inference, the crucial graph construction phase has received comparatively less attention, despite its significant influence on the subsequent phase. In this paper, we present an optimal asymmetric graph structure for the label inference phase with theoretical motivations. Unlike existing graph construction methods, we differentiate the distinct roles that labeled nodes and unlabeled nodes could play. Accordingly, we design an efficient block-wise graph learning algorithm with a global convergence guarantee. Other benefits induced by our method, such as enhanced robustness to noisy node features, are explored as well. Finally, we perform extensive experiments on synthetic and real-world datasets to demonstrate its superiority to the state-of-the-art graph construction methods in GSSL.

----

## [0] On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence

**Authors**: *Achraf Azize, Marc Jourdan, Aymen Al Marjani, Debabrota Basu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e14de1a0ebc31d9b989f5f5528c125bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e14de1a0ebc31d9b989f5f5528c125bb-Abstract-Conference.html)

**Abstract**:

Best Arm Identification (BAI) problems are progressively used for data-sensitive applications, such as designing adaptive clinical trials, tuning hyper-parameters, and conducting user studies to name a few. Motivated by the data privacy concerns invoked by these applications, we study the problem of BAI with fixed confidence under $\epsilon$-global Differential Privacy (DP). First, to quantify the cost of privacy, we derive a lower bound on the sample complexity of any $\delta$-correct BAI algorithm satisfying $\epsilon$-global DP. Our lower bound suggests the existence of two privacy regimes depending on the privacy budget $\epsilon$. In the high-privacy regime (small $\epsilon$), the hardness depends on a coupled effect of privacy and a novel information-theoretic quantity, called the Total Variation Characteristic Time. In the low-privacy regime (large $\epsilon$), the sample complexity lower bound reduces to the classical non-private lower bound. Second, we propose AdaP-TT, an $\epsilon$-global DP variant of the Top Two algorithm. AdaP-TT runs in *arm-dependent adaptive episodes* and adds *Laplace noise* to ensure a good privacy-utility trade-off. We derive an asymptotic upper bound on the sample complexity of AdaP-TT that matches with the lower bound up to multiplicative constants in the high-privacy regime. Finally, we provide an experimental analysis of AdaP-TT that validates our theoretical results.

----

## [0] COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs

**Authors**: *Tiep Le, Vasudev Lal, Phillip Howard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e14e4cb8266184ceb234973dfe07faed-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e14e4cb8266184ceb234973dfe07faed-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation. We make our code and the COCO-Counterfactuals dataset publicly available.

----

## [0] BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis

**Authors**: *Zelin Ni, Hang Yu, Shizhan Liu, Jianguo Li, Weiyao Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html)

**Abstract**:

Bases have become an integral part of modern deep learning-based models for time series forecasting due to their ability to act as feature extractors or future references. To be effective, a basis must be tailored to the specific set of time series data and exhibit distinct correlation with each time series within the set. However, current state-of-the-art methods are limited in their ability to satisfy both of these requirements simultaneously. To address this challenge, we propose BasisFormer, an end-to-end time series forecasting architecture that leverages learnable and interpretable bases. This architecture comprises three components: First, we acquire bases through adaptive self-supervised learning, which treats the historical and future sections of the time series as two distinct views and employs contrastive learning. Next, we design a Coef module that calculates the similarity coefficients between the time series and bases in the historical view via bidirectional cross-attention. Finally, we present a Forecast module that selects and consolidates the bases in the future view based on the similarity coefficients, resulting in accurate future predictions. Through extensive experiments on six datasets, we demonstrate that BasisFormer outperforms previous state-of-the-art methods by 11.04% and 15.78% respectively for univariate and multivariate forecasting tasks. Code isavailable at: https://github.com/nzl5116190/Basisformer.

----

## [0] Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior

**Authors**: *Shashank Subramanian, Peter Harrington, Kurt Keutzer, Wahid Bhimji, Dmitriy Morozov, Michael W. Mahoney, Amir Gholami*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e15790966a4a9d85d688635c88ee6d8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e15790966a4a9d85d688635c88ee6d8a-Abstract-Conference.html)

**Abstract**:

Pre-trained machine learning (ML) models have shown great performance for awide range of applications, in particular in natural language processing (NLP)and computer vision (CV). Here, we study how pre-training could be used forscientific machine learning (SciML) applications, specifically in the context oftransfer learning. We study the transfer behavior of these models as (i) the pretrainedmodel size is scaled, (ii) the downstream training dataset size is scaled,(iii) the physics parameters are systematically pushed out of distribution, and (iv)how a single model pre-trained on a mixture of different physics problems canbe adapted to various downstream applications. We find that—when fine-tunedappropriately—transfer learning can help reach desired accuracy levels with ordersof magnitude fewer downstream examples (across different tasks that can even beout-of-distribution) than training from scratch, with consistent behaviour across awide range of downstream examples. We also find that fine-tuning these modelsyields more performance gains as model size increases, compared to training fromscratch on new downstream tasks. These results hold for a broad range of PDElearning tasks. All in all, our results demonstrate the potential of the “pre-train andfine-tune” paradigm for SciML problems, demonstrating a path towards buildingSciML foundation models. Our code is available as open-source.

----

## [0] Hyperbolic Space with Hierarchical Margin Boosts Fine-Grained Learning from Coarse Labels

**Authors**: *Shu-Lin Xu, Yifan Sun, Faen Zhang, Anqi Xu, Xiu-Shen Wei, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e17e11960843febbc2dd22d3c7d79144-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e17e11960843febbc2dd22d3c7d79144-Abstract-Conference.html)

**Abstract**:

Learning fine-grained embeddings from coarse labels is a challenging task due to limited label granularity supervision, i.e., lacking the detailed distinctions required for fine-grained tasks. The task becomes even more demanding when attempting few-shot fine-grained recognition, which holds practical significance in various applications. To address these challenges, we propose a novel method that embeds visual embeddings into a hyperbolic space and enhances their discriminative ability with a hierarchical cosine margins manner. Specifically, the hyperbolic space offers distinct advantages, including the ability to capture hierarchical relationships and increased expressive power, which favors modeling fine-grained objects. Based on the hyperbolic space, we further enforce relatively large/small similarity margins between coarse/fine classes, respectively, yielding the so-called hierarchical cosine margins manner. While enforcing similarity margins in the regular Euclidean space has become popular for deep embedding learning, applying it to the hyperbolic space is non-trivial and validating the benefit for coarse-to-fine generalization is valuable. Extensive experiments conducted on five benchmark datasets showcase the effectiveness of our proposed method, yielding state-of-the-art results surpassing competing methods.

----

## [0] PromptIR: Prompting for All-in-One Image Restoration

**Authors**: *Vaishnav Potlapalli, Syed Waqas Zamir, Salman H. Khan, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html)

**Abstract**:

Image restoration involves recovering a high-quality clean image from its degraded version. Deep learning-based methods have significantly improved image restoration performance, however, they have limited generalization ability to different degradation types and levels. This restricts their real-world application since it requires training individual models for each specific degradation and knowing the input degradation type to apply the relevant model. We present a prompt-based learning approach, PromptIR, for All-In-One image restoration that can effectively restore images from various types and levels of degradation. In particular, our method uses prompts to encode degradation-specific information, which is then used to dynamically guide the restoration network. This allows our method to generalize to different degradation types and levels, while still achieving state-of-the-art results on image denoising, deraining, and dehazing.  Overall, PromptIR offers a generic and efficient plugin module with few lightweight prompts that can be used to restore images of various types and levels of degradation with no prior information on the corruptions present in the image. Our code and pre-trained models are available here: https://github.com/va1shn9v/PromptIR

----

## [0] Creating a Public Repository for Joining Private Data

**Authors**: *James Cook, Milind Shyani, Nina Mishra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e19560e93418dd0d6498bd3b2de856cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e19560e93418dd0d6498bd3b2de856cd-Abstract-Conference.html)

**Abstract**:

How can one publish a dataset with sensitive attributes in a way that both preserves privacy and enables joins with other datasets on those same sensitive attributes? This problem arises in many contexts, e.g., a hospital and an airline may want to jointly determine whether people who take long-haul flights are more likely to catch respiratory infections. If they join their data by a common keyed user identifier such as email address, they can determine the answer, though it breaks privacy.  This paper shows how the hospital can generate a private sketch and how the airline can privately join with the hospital's sketch by email address. The proposed solution satisfies pure differential privacy and gives approximate answers to linear queries and optimization problems over those joins. Whereas prior work such as secure function evaluation requires sender/receiver interaction, a distinguishing characteristic of the proposed approach is that it is non-interactive. Consequently, the sketch can be published to a repository for any organization to join with, facilitating data discovery. The accuracy of the method is demonstrated through both theoretical analysis and extensive empirical evidence.

----

## [0] What Truly Matters in Trajectory Prediction for Autonomous Driving?

**Authors**: *Tran Phong, Haoran Wu, Cunjun Yu, Panpan Cai, Sifa Zheng, David Hsu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e197fe307eb3467035f892dc100d570a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e197fe307eb3467035f892dc100d570a-Abstract-Conference.html)

**Abstract**:

Trajectory prediction plays a vital role in the performance of autonomous driving systems, and prediction accuracy, such as average displacement error (ADE) or final displacement error (FDE), is widely used as a performance metric. However, a significant disparity exists between the accuracy of predictors on fixed datasets and driving performance when the predictors are used downstream for vehicle control, because of a dynamics gap. In the real world, the prediction algorithm influences the behavior of the ego vehicle, which, in turn, influences the behaviors of other vehicles nearby. This interaction results in predictor-specific dynamics that directly impacts prediction results. In fixed datasets, since other vehicles' responses are predetermined, this interaction effect is lost, leading to a significant dynamics gap. This paper studies the overlooked significance of this dynamics gap. We also examine several other factors contributing to the disparity between prediction performance and driving performance. The findings highlight the trade-off between the predictor's computational efficiency and prediction accuracy in determining real-world driving performance. In summary,  an interactive, task-driven evaluation protocol for trajectory prediction is crucial to capture its effectiveness for autonomous driving. Source code along with experimental settings is available online (https://whatmatters23.github.io/).

----

## [0] AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models

**Authors**: *Yuancheng Wang, Zeqian Ju, Xu Tan, Lei He, Zhizheng Wu, Jiang Bian, Sheng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1b619a9e241606a23eb21767f16cf81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1b619a9e241606a23eb21767f16cf81-Abstract-Conference.html)

**Abstract**:

Audio editing is applicable for various purposes, such as adding background sound effects, replacing a musical instrument, and repairing damaged audio. Recently, some diffusion-based methods achieved zero-shot audio editing by using a diffusion and denoising process conditioned on the text description of the output audio. However, these methods still have some problems: 1) they have not been trained on editing tasks and cannot ensure good editing effects; 2) they can erroneously modify audio segments that do not require editing; 3) they need a complete description of the output audio, which is not always available or necessary in practical scenarios. In this work, we propose AUDIT, an instruction-guided audio editing model based on latent diffusion models. Specifically, \textbf{AUDIT} has three main design features: 1) we construct triplet training data (instruction, input audio, output audio) for different audio editing tasks and train a diffusion model using instruction and input (to be edited) audio as conditions and generating output (edited) audio; 2) it can automatically learn to only modify segments that need to be edited by comparing the difference between the input and output audio; 3) it only needs edit instructions instead of full target audio descriptions as text input. AUDIT achieves state-of-the-art results in both objective and subjective metrics for several audio editing tasks (e.g., adding, dropping, replacement, inpainting, super-resolution). Demo samples are available at https://audit-demopage.github.io/.

----

## [0] An Optimization-based Approach To Node Role Discovery in Networks: Approximating Equitable Partitions

**Authors**: *Michael Scholkemper, Michael T. Schaub*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1c73e9595126794186536cfbbed012f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1c73e9595126794186536cfbbed012f-Abstract-Conference.html)

**Abstract**:

Similar to community detection, partitioning the nodes of a complex network according to their structural roles aims to identify fundamental building blocks of a network, which can be used, e.g., to find simplified descriptions of the network connectivity, to derive reduced order models for dynamical processes unfolding on processes, or as ingredients for various network analysis and graph mining tasks. In this work, we offer a fresh look on the problem of role extraction and its differences to community detection and present a definition of node roles and two associated optimization problems (cost functions) grounded in ideas related to graph-isomorphism tests, the Weisfeiler-Leman algorithm and equitable partitions. We present theoretical guarantees and validate our approach via a novel “role-infused partition benchmark”, a network model from which we can sample networks in which nodes are endowed with different roles in a stochastic way.

----

## [0] Robust Model Reasoning and Fitting via Dual Sparsity Pursuit

**Authors**: *Xingyu Jiang, Jiayi Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1de63ec74f40d3234c4e053f3528e18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1de63ec74f40d3234c4e053f3528e18-Abstract-Conference.html)

**Abstract**:

In this paper, we contribute to solving a threefold problem: outlier rejection, true model reasoning and parameter estimation with a unified optimization modeling. To this end, we first pose this task as a sparse subspace recovering problem, to search a maximum of independent bases under an over-embedded data space. Then we convert the objective into a continuous optimization paradigm that estimates sparse solutions for both bases and errors. Wherein a fast and robust solver is proposed to accurately estimate the sparse subspace parameters and error entries, which is implemented by a proximal approximation method under the alternating optimization framework with the ``optimal'' sub-gradient descent. Extensive experiments regarding known and unknown model fitting on synthetic and challenging real datasets have demonstrated the superiority of our method against the state-of-the-art. We also apply our method to multi-class multi-model fitting and loop closure detection, and achieve promising results both in accuracy and efficiency. Code is released at: https://github.com/StaRainJ/DSP.

----

## [0] Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance

**Authors**: *Jonathan Crabbé, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html)

**Abstract**:

Interpretability methods are valuable only if their explanations faithfully describe the explained model. In this work, we consider neural networks whose predictions are invariant under a specific symmetry group. This includes popular architectures, ranging from convolutional to graph neural networks. Any explanation that faithfully explains this type of model needs to be in agreement with this invariance property. We formalize this intuition through the notion of explanation invariance and equivariance by leveraging the formalism from geometric deep learning. Through this rigorous formalism, we derive (1) two metrics to measure the robustness of any interpretability method with respect to the model symmetry group; (2) theoretical robustness guarantees for some popular interpretability methods and (3) a systematic approach to increase the invariance of any interpretability method with respect to a symmetry group. By empirically measuring our metrics for explanations of models associated with various modalities and symmetry groups, we derive a set of 5 guidelines to allow users and developers of interpretability methods to produce robust explanations.

----

## [0] Back-Modality: Leveraging Modal Transformation for Data Augmentation

**Authors**: *Zhi Li, Yifan Liu, Yin Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e20a65c7308b7b94ed1178eebc45bf76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e20a65c7308b7b94ed1178eebc45bf76-Abstract-Conference.html)

**Abstract**:

We introduce Back-Modality, a novel data augmentation schema predicated on modal transformation. Data from an initial modality undergoes transformation to an intermediate modality, followed by a reverse transformation. This framework serves dual roles. On one hand, it operates as a general data augmentation strategy. On the other hand, it allows for other augmentation techniques, suitable for the intermediate modality, to enhance the initial modality. For instance, data augmentation methods applicable to pure text can be employed to augment images, thereby facilitating the cross-modality of data augmentation techniques. To validate the viability and efficacy of our framework, we proffer three instantiations of Back-Modality: back-captioning, back-imagination, and back-speech. Comprehensive evaluations across tasks such as image classification, sentiment classification, and textual entailment demonstrate that our methods consistently enhance performance under data-scarce circumstances.

----

## [0] Gradient-Based Feature Learning under Structured Data

**Authors**: *Alireza Mousavi Hosseini, Denny Wu, Taiji Suzuki, Murat A. Erdogdu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e21955c93dede886af1d0d362c756757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e21955c93dede886af1d0d362c756757-Abstract-Conference.html)

**Abstract**:

Recent works have demonstrated that the sample complexity of gradient-based learning of single index models, i.e. functions that depend on a 1-dimensional projection of the input data, is governed by their information exponent. However, these results are only concerned with isotropic data, while in practice the input often contains additional structure which can implicitly guide the algorithm. In this work, we investigate the effect of a spiked covariance structure and reveal several interesting phenomena. First, we show that in the anisotropic setting, the commonly used spherical gradient dynamics may fail to recover the true direction, even when the spike is perfectly aligned with the target direction. Next, we show that appropriate weight normalization that is reminiscent of batch normalization can alleviate this issue. Further, by exploiting the alignment between the (spiked) input covariance and the target, we obtain improved sample complexity compared to the isotropic case. In particular, under the spiked model with a suitably large spike, the sample complexity of gradient-based training can be made independent of the information exponent while also outperforming lower bounds for rotationally invariant kernel methods.

----

## [0] Does Invariant Graph Learning via Environment Augmentation Learn Invariance?

**Authors**: *Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, James Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e21a7b668ce3ea2c9c964c52d1c9f161-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e21a7b668ce3ea2c9c964c52d1c9f161-Abstract-Conference.html)

**Abstract**:

Invariant graph representation learning aims to learn the invariance among data from different environments for out-of-distribution generalization on graphs. As the graph environment partitions are usually expensive to obtain, augmenting the environment information has become the de facto approach. However, the usefulness of the augmented environment information has never been verified. In this work, we find that it is fundamentally impossible to learn invariant graph representations via environment augmentation without additional assumptions. Therefore, we develop a set of minimal assumptions, including variation sufficiency and variation consistency, for feasible invariant graph learning. We then propose a new framework Graph invAriant Learning Assistant (GALA). GALA incorporates an assistant model that needs to be sensitive to graph environment changes or distribution shifts. The correctness of the proxy predictions by the assistant model hence can differentiate the variations in spurious subgraphs. We show that extracting the maximally invariant subgraph to the proxy predictions provably identifies the underlying invariant subgraph for successful OOD generalization under the established minimal assumptions. Extensive experiments on datasets including DrugOOD with various graph distribution shifts confirm the effectiveness of GALA.

----

## [0] Mitigating Test-Time Bias for Fair Image Retrieval

**Authors**: *Fanjie Kong, Shuai Yuan, Weituo Hao, Ricardo Henao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e24570da4fa1c005b189104250993aee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e24570da4fa1c005b189104250993aee-Abstract-Conference.html)

**Abstract**:

We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result while maintaining satisfactory retrieval performance. The source code is publicly available at \url{https://github.com/timqqt/FairTextbasedImageRetrieval}.

----

## [0] Lower Bounds on Adaptive Sensing for Matrix Recovery

**Authors**: *Praneeth Kacham, David P. Woodruff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e258bb98cc032ab6ae9053db453431f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e258bb98cc032ab6ae9053db453431f7-Abstract-Conference.html)

**Abstract**:

We study lower bounds on adaptive sensing algorithms for recovering low rank matrices using linear measurements. Given an $n \times n$ matrix $A$, a general linear measurement $S(A)$, for an $n \times n$ matrix $S$, is just the inner product of $S$ and $A$, each treated as $n^2$-dimensional vectors. By performing as few linear measurements as possible on a rank-$r$ matrix $A$, we hope to construct a matrix $\hat{A}$ that satisfies $|A - \hat{A}|\_F^2 \le c |A|\_F^2$, for a small constant $c$. Here $|A|\_F$ denotes the Frobenius norm $(\sum_{i,j} A_{i,j}^2)^{1/2}$. It is commonly assumed that when measuring $A$ with $S$, the response is corrupted with an independent Gaussian random variable of mean $0$ and variance $\sigma^2$. CandÃ¨s and Plan (IEEE Trans. Inform. Theory 2011) study non-adaptive algorithms for low rank matrix recovery using random linear measurements. They use the restricted isometry property (RIP) of Random Gaussian Matrices to give tractable algorithms to estimate $A$ from the measurements.At the edge of the noise level where recovery is information-theoretically feasible, it is known that their non-adaptive algorithms need to perform $\Omega(n^2)$ measurements, which amounts to reading the entire matrix. An important question is whether adaptivity helps in decreasing the overall number of measurements. While for the related problem of sparse recovery, adaptive algorithms have been extensively studied, as far as we are aware adaptive algorithms and lower bounds on them seem largely unexplored for matrix recovery. We show that any adaptive algorithm that uses $k$ linear measurements in each round and outputs an approximation as in (1) with probability $\ge 9/10$ must run for $t = \Omega(\log(n^2/k)/\log\log n)$ rounds. Our lower bound shows that any adaptive algorithm which uses $n^{2-\beta}$ ($\beta > 0$ is arbitrary constant) linear measurements in each round must run for $\Omega(\log n/\log\log n)$ rounds. Our techniques also readily extend to obtain lower bounds on adaptive algorithms for tensor recovery. Our hard distribution also allows us to give a measurement-vs-rounds trade-off for many sensing problems in numerical linear algebra, such as spectral norm low rank approximation, Frobenius norm low rank approximation, singular vector approximation, and more.

----

## [0] Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction

**Authors**: *Anagh Malik, Parsa Mirdehghan, Sotiris Nousias, Kyros Kutulakos, David B. Lindell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html)

**Abstract**:

Neural radiance fields (NeRFs) have become a ubiquitous tool for modeling scene appearance and geometry from multiview imagery. Recent work has also begun to explore how to use additional supervision from lidar or depth sensor measurements in the NeRF framework. However, previous lidar-supervised NeRFs focus on rendering conventional camera imagery and use lidar-derived point cloud data as auxiliary supervision; thus, they fail to incorporate the underlying image formation model of the lidar. Here, we propose a novel method for rendering transient NeRFs that take as input the raw, time-resolved photon count histograms measured by a single-photon lidar system, and we seek to render such histograms from novel views. Different from conventional NeRFs, the approach relies on a time-resolved version of the volume rendering equation to render the lidar measurements and capture transient light transport phenomena at picosecond timescales. We evaluate our method on a first-of-its-kind dataset of simulated and captured transient multiview scans from a prototype single-photon lidar. Overall, our work brings NeRFs to a new dimension of imaging at transient timescales, newly enabling rendering of transient imagery from novel views. Additionally, we show that our approach recovers improved geometry and conventional appearance compared to point cloud-based supervision when training on few input viewpoints. Transient NeRFs may be especially useful for applications which seek to simulate raw lidar measurements for downstream tasks in autonomous driving, robotics, and remote sensing.

----

## [0] An Exploration-by-Optimization Approach to Best of Both Worlds in Linear Bandits

**Authors**: *Shinji Ito, Kei Takemura*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e262fc23ec7275230ee77c55d0cc9555-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e262fc23ec7275230ee77c55d0cc9555-Abstract-Conference.html)

**Abstract**:

In this paper, we consider how to construct best-of-both-worlds linear bandit algorithms that achieve nearly optimal performance for both stochastic and adversarial environments.  For this purpose, we show that a natural approach referred to as exploration by optimization [Lattimore and Szepesv√°ri, 2020] works well. Specifically, an algorithm constructed using this approach achieves $O(d \sqrt{ T \log{T}})$-regret in adversarial environments and $O(\frac{d^2 \log T}{\Delta_{\min}} )$-regret in stochastic environments.  Symbols $d$, $T$ and $\Delta_{\min}$ here represent the dimensionality of the action set, the time horizon, and the minimum sub-optimality gap, respectively.  We also show that this algorithm has even better theoretical guarantees for important special cases including the multi-armed bandit problem and multitask bandits.

----

## [0] The expressive power of pooling in Graph Neural Networks

**Authors**: *Filippo Maria Bianchi, Veronica Lachi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html)

**Abstract**:

In Graph Neural Networks (GNNs), hierarchical pooling operators generate local summaries of the data by coarsening the graph structure and the vertex features. Considerable attention has been devoted to analyzing the expressive power of message-passing (MP) layers in GNNs, while a study on how graph pooling affects the expressiveness of a GNN is still lacking. Additionally, despite the recent advances in the design of pooling operators, there is not a principled criterion to compare them. In this work, we derive sufficient conditions for a pooling operator to fully preserve the expressive power of the MP layers before it. These conditions serve as a universal and theoretically-grounded criterion for choosing among existing pooling operators or designing new ones. Based on our theoretical findings, we analyze several existing pooling operators and identify those that fail to satisfy the expressiveness conditions. Finally, we introduce an experimental setup to verify empirically the expressive power of a GNN equipped with pooling layers, in terms of its capability to perform a graph isomorphism test.

----

## [0] Cal-DETR: Calibrated Detection Transformer

**Authors**: *Muhammad Akhtar Munir, Salman H. Khan, Muhammad Haris Khan, Mohsen Ali, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html)

**Abstract**:

Albeit revealing impressive predictive performance for several computer vision tasks, deep neural networks (DNNs) are prone to making overconfident predictions. This limits the adoption and wider utilization of DNNs in many safety-critical applications. There have been recent efforts toward calibrating DNNs, however, almost all of them focus on the classification task. Surprisingly, very little attention has been devoted to calibrating modern DNN-based object detectors, especially detection transformers, which have recently demonstrated promising detection performance and are influential in many decision-making systems. In this work, we address the problem by proposing a mechanism for calibrated detection transformers (Cal-DETR), particularly for Deformable-DETR, UP-DETR, and DINO. We pursue the train-time calibration route and make the following contributions. First, we propose a simple yet effective approach for quantifying uncertainty in transformer-based object detectors. Second, we develop an uncertainty-guided logit modulation mechanism that leverages the uncertainty to modulate the class logits. Third, we develop a logit mixing approach that acts as a regularizer with detection-specific losses and is also complementary to the uncertainty-guided logit modulation technique to further improve the calibration performance. Lastly, we conduct extensive experiments across three in-domain and four out-domain scenarios. Results corroborate the effectiveness of Cal-DETR against the competing train-time methods in calibrating both in-domain and out-domain detections while maintaining or even improving the detection performance. Our codebase and pre-trained models can be accessed at \url{https://github.com/akhtarvision/cal-detr}.

----

## [0] Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory

**Authors**: *Minhak Song, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e2a9256bd816ab9e082dfaa22f1f62a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e2a9256bd816ab9e082dfaa22f1f62a2-Abstract-Conference.html)

**Abstract**:

Cohen et al. (2021) empirically study the evolution of the largest eigenvalue of the loss Hessian, also known as sharpness, along the gradient descent (GD) trajectory and observe the Edge of Stability (EoS) phenomenon. The sharpness increases at the early phase of training (referred to as progressive sharpening), and eventually saturates close to the threshold of $2 / \text{(step size)}$. In this paper, we start by demonstrating through empirical studies that when the EoS phenomenon occurs, different GD trajectories (after a proper reparameterization) align on a specific bifurcation diagram independent of initialization. We then rigorously prove this trajectory alignment phenomenon for a two-layer fully-connected linear network and a single-neuron nonlinear network trained with a single data point. Our trajectory alignment analysis establishes both progressive sharpening and EoS phenomena, encompassing and extending recent findings in the literature.

----

## [0] OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents

**Authors**: *Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks. However, the datasets used to train these models have not been released, and the collection process has not been fully specified.  We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELICS, we train on the dataset vision and language models of 9 and 80 billion parameters, IDEFICS-9B and IDEFICS, and obtain competitive performance on different multimodal benchmarks. We release our dataset, models and code.

----

## [0] ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets

**Authors**: *Damien Teney, Yong Lin, Seong Joon Oh, Ehsan Abbasnejad*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html)

**Abstract**:

Several studies have compared the in-distribution (ID) and out-of-distribution (OOD) performance of models in computer vision and NLP. They report a frequent positive correlation and some surprisingly never even observe an inverse correlation indicative of a necessary trade-off. The possibility of inverse patterns is important to determine whether ID performance can serve as a proxy for OOD generalization capabilities.This paper shows that inverse correlations between ID and OOD performance do happen with multiple real-world datasets, not only in artificial worst-case settings. We explain theoretically how these cases arise and how past studies missed them because of improper methodologies that examined a biased selection of models.Our observations lead to recommendations that contradict those found in much of the current literature.- High OOD performance sometimes requires trading off ID performance.- Focusing on ID performance alone may not lead to optimal OOD performance. It may produce diminishing (eventually negative) returns in OOD performance.- In these cases, studies on OOD generalization that use ID performance for model selection (a common recommended practice) will necessarily miss the best-performing models, making these studies blind to a whole range of phenomena.

----

## [0] On Generalization Bounds for Projective Clustering

**Authors**: *Maria Sofia Bucarelli, Matilde Fjeldsø Larsen, Chris Schwiegelshohn, Mads Toftrup*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e30bf4765ae6b16a87fb4d7b0b3b3dec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e30bf4765ae6b16a87fb4d7b0b3b3dec-Abstract-Conference.html)

**Abstract**:

Given a set of points, clustering consists of finding a partition of a point set into $k$ clusters such that the center to which a point is assigned is as close as possible. Most commonly, centers are points themselves, which leads to the famous $k$-median and $k$-means objectives. One may also choose centers to be $j$ dimensional subspaces, which gives rise to subspace clustering. In this paper, we consider learning bounds for these problems. That is, given a set of $n$ samples $P$ drawn independently from some unknown, but fixed distribution $\mathcal{D}$, how quickly does a solution computed on $P$ converge to the optimal clustering of $\mathcal{D}$?We give several near optimal results. In particular, 1. For center-based objectives, we show a convergence rate of $\tilde{O}\left(\sqrt{{k}/{n}}\right)$. This matches the known optimal bounds of [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] and [Bartlett, Linder, and Lugosi, IEEE Trans. Inf. Theory 1998] for $k$-means and extends it to other important objectives such as $k$-median. 2. For subspace clustering with $j$-dimensional subspaces, we show a convergence rate of $\tilde{O}\left(\sqrt{{(kj^2)}/{n}}\right)$. These are the first provable bounds for most of these problems. For the specific case of projective clustering, which generalizes $k$-means, we show a converge rate of $\Omega\left(\sqrt{{(kj)}/{n}}\right)$ is necessary, thereby proving that the bounds from [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] are essentially optimal.

----

## [0] Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity

**Authors**: *Tianqin Li, Ziqi Wen, Yangfan Li, Tai Sing Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e31c16c7b3e0ccee5159ae5443154fac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e31c16c7b3e0ccee5159ae5443154fac-Abstract-Conference.html)

**Abstract**:

Current deep-learning models for object recognition are known to be heavily biased toward texture. In contrast, human visual systems are known to be biased toward shape and structure. What could be the design principles in human visual systems that led to this difference? How could we introduce more shape bias into the deep learning models? In this paper, we report that sparse coding, a ubiquitous principle in the brain,  can in itself introduce shape bias into the network. We found that enforcing the sparse coding constraint using a non-differential Top-K operation  can lead to the emergence of structural encoding in neurons in convolutional neural networks,  resulting in a smooth decomposition of objects into parts and subparts and endowing the networks with shape bias.  We demonstrated this emergence of shape bias and its functional benefits for different network structures with various datasets. For object recognition convolutional neural networks, the shape bias leads to greater robustness against style and pattern change distraction. For the image synthesis generative adversary networks,  the emerged shape bias leads to more coherent and decomposable structures in the synthesized images. Ablation studies suggest that sparse codes tend to encode structures, whereas the more distributed codes tend to favor texture. Our code is host at the github repository: https://topk-shape-bias.github.io/

----

## [0] Resilient Constrained Learning

**Authors**: *Ignacio Hounie, Alejandro Ribeiro, Luiz F. O. Chamon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e32349fe7e3cd4f9ef598c2b7b7a31f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e32349fe7e3cd4f9ef598c2b7b7a31f4-Abstract-Conference.html)

**Abstract**:

When deploying machine learning solutions, they must satisfy multiple requirements beyond accuracy, such as fairness, robustness, or safety. These requirements are imposed during training either implicitly, using penalties, or explicitly, using constrained optimization methods based on Lagrangian duality. Either way, specifying requirements is hindered by the presence of compromises and limited prior knowledge about the data. Furthermore, their impact on performance can often only be evaluated by actually solving the learning problem. This paper presents a constrained learning approach that adapts the requirements while simultaneously solving the learning task. To do so, it relaxes the learning constraints in a way that contemplates how much they affect the task at hand by balancing the performance gains obtained from the relaxation against a user-defined cost of that relaxation. We call this approach resilient constrained learning after the term used to describe ecological systems that adapt to disruptions by modifying their operation. We show conditions under which this balance can be achieved and introduce a practical algorithm to compute it, for which we derive approximation and generalization guarantees. We showcase the advantages of this resilient learning method in image classification tasks involving multiple potential invariances and in federated learning under distribution shift.

----

## [0] Recovering Simultaneously Structured Data via Non-Convex Iteratively Reweighted Least Squares

**Authors**: *Christian Kümmerle, Johannes Maly*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e33a4d41305fb34316df6f3fa8a0e58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e33a4d41305fb34316df6f3fa8a0e58c-Abstract-Conference.html)

**Abstract**:

We propose a new algorithm for the problem of recovering data that adheres to multiple, heterogenous low-dimensional structures from linear observations. Focussing on data matrices that are simultaneously row-sparse and low-rank, we propose and analyze an iteratively reweighted least squares (IRLS) algorithm that is able to leverage both structures. In particular, it optimizes a combination of non-convex surrogates for row-sparsity and rank, a balancing of which is built into the algorithm. We prove locally quadratic convergence of the iterates to a simultaneously structured data matrix in a regime of minimal sample complexity (up to constants and a logarithmic factor), which is known to be impossible for a combination of convex surrogates. In experiments, we show that the IRLS method exhibits favorable empirical convergence, identifying simultaneously row-sparse and low-rank matrices from fewer measurements than state-of-the-art methods.

----

## [0] Error Bounds for Learning with Vector-Valued Random Features

**Authors**: *Samuel Lanthaler, Nicholas H. Nelsen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e34d908241aef40440e61d2a27715424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e34d908241aef40440e61d2a27715424-Abstract-Conference.html)

**Abstract**:

This paper provides a comprehensive error analysis of learning with vector-valued random features (RF). The theory is developed for RF ridge regression in a fully general infinite-dimensional input-output setting, but nonetheless applies to and improves existing finite-dimensional analyses. In contrast to comparable work in the literature, the approach proposed here relies on a direct analysis of the underlying risk functional and completely avoids the explicit RF ridge regression solution formula in terms of random matrices. This removes the need for concentration results in random matrix theory or their generalizations to random operators. The main results established in this paper include strong consistency of vector-valued RF estimators under model misspecification and minimax optimal convergence rates in the well-specified setting. The parameter complexity (number of random features) and sample complexity (number of labeled data) required to achieve such rates are comparable with Monte Carlo intuition and free from logarithmic factors.

----

## [0] CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection

**Authors**: *Yang Cao, Yihan Zeng, Hang Xu, Dan Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e352b765e625934ce86919995e2371aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e352b765e625934ce86919995e2371aa-Abstract-Conference.html)

**Abstract**:

Open-vocabulary 3D Object Detection (OV-3DDet) aims to detect objects from an arbitrary list of categories within a 3D scene, which remains seldom explored in the literature. There are primarily two fundamental problems in OV-3DDet, i.e., localizing and classifying novel objects. This paper aims at addressing the two problems simultaneously via a unified framework, under the condition of limited base categories. To localize novel 3D objects, we propose an effective 3D Novel Object Discovery strategy, which utilizes both the 3D box geometry priors and 2D semantic open-vocabulary priors to generate pseudo box labels of the novel objects. To classify novel object boxes, we further develop a cross-modal alignment module based on discovered novel boxes, to align feature spaces between 3D pointcloud and image/text modalities. Specifically, the alignment process contains a class-agnostic and a class-discriminative alignment, incorporating not only the base objects with annotations but also the increasingly discovered novel objects, resulting in an iteratively enhanced alignment. The novel box discovery and crossmodal alignment are jointly learned to collaboratively benefit each other. Thenovel object discovery can directly impact the cross-modal alignment, while a better feature alignment can, in turn, boost the localization capability, leading to a unified OV-3DDet framework, named CoDA, for simultaneous novel object localization and classification. Extensive experiments on two challenging datasets (i.e., SUN-RGBD and ScanNet) demonstrate the effectiveness of our method and also show a significant mAP improvement upon the best-performing alternative method by 80%. Codes and pre-trained models are released on the project page.

----

## [0] Don't blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

**Authors**: *Aahlad Manas Puli, Lily H. Zhang, Yoav Wald, Rajesh Ranganath*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e35460304fdf6df523f068a59aaf8829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e35460304fdf6df523f068a59aaf8829-Abstract-Conference.html)

**Abstract**:

Common explanations for shortcut learning assume that the shortcut improves prediction only under the training distribution. Thus, models trained in the typical way by minimizing log-loss using gradient descent, which we call default-ERM, should utilize the shortcut. However, even when the stable feature determines the label in the training distribution and the shortcut does not provide any additional information, like in perception tasks, default-ERM exhibits shortcut learning. Why are such solutions preferred when the loss can be driven to zero when using the stable feature alone? By studying a linear perception task, we show that default-ERM’s preference for maximizing the margin, even without overparameterization, leads to models that depend more on the shortcut than the stable feature. This insight suggests that default-ERM’s implicit inductive bias towards max-margin may be unsuitable for perception tasks. Instead, we consider inductive biases toward uniform margins. We show that uniform margins guarantee sole dependence on the perfect stable feature in the linear perception task and suggest alternative loss functions, termed margin control (MARG-CTRL), that encourage uniform-margin solutions. MARG-CTRL techniques mitigate shortcut learning on a variety of vision and language tasks, showing that changing inductive biases can remove the need for complicated shortcut-mitigating methods in perception tasks.

----

## [0] Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer

**Authors**: *Yuandong Tian, Yiping Wang, Beidi Chen, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e359ebe56ba306b674e8952349c6049e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e359ebe56ba306b674e8952349c6049e-Abstract-Conference.html)

**Abstract**:

Transformer architecture has shown impressive performance in multiple research domains and has become the backbone of many neural network models. However, there is limited understanding on how it works. In particular, with a simple predictive loss,  how the representation emerges from the gradient \emph{training dynamics} remains a mystery. In this paper, for 1-layer transformer with one self-attention layer plus one decoder layer, we analyze its SGD training dynamics for the task of next token prediction in a mathematically rigorous manner. We open the black box of the dynamic process of how the self-attention layer combines input tokens, and reveal the nature of underlying inductive bias. More specifically, with the assumption (a) no positional encoding, (b) long input sequence, and (c) the decoder layer learns faster than the self-attention layer, we prove that self-attention acts as a \emph{discriminative scanning algorithm}:  starting from uniform attention, it gradually attends more to distinct key tokens for a specific next token to be predicted, and pays less attention to common key tokens that occur across different next tokens. Among distinct tokens, it progressively drops attention weights, following the order of low to high co-occurrence between the key and the query token in the training set. Interestingly, this procedure does not lead to winner-takes-all, but stops due to a \emph{phase transition} that is controllable by the learning rate of the decoder layer, leaving (almost) fixed token combination. We verify this \textbf{\emph{scan and snap}} dynamics on synthetic and real-world data (WikiText-103).

----

## [0] Stein Π-Importance Sampling

**Authors**: *Congye Wang, Ye Chen, Heishiro Kanagawa, Chris J. Oates*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html)

**Abstract**:

Stein discrepancies have emerged as a powerful tool for retrospective improvement of Markov chain Monte Carlo output.  However, the question of how to design Markov chains that are well-suited to such post-processing has yet to be addressed.  This paper studies Stein importance sampling, in which weights are assigned to the states visited by a $\Pi$-invariant Markov chain to obtain a consistent approximation of $P$, the intended target.  Surprisingly, the optimal choice of $\Pi$ is not identical to the target $P$; we therefore propose an explicit construction for $\Pi$ based on a novel variational argument.  Explicit conditions for convergence of Stein $\Pi$-Importance Sampling are established.  For $\approx 70$% of tasks in the PosteriorDB benchmark, a significant improvement over the analogous post-processing of $P$-invariant Markov chains is reported.

----

## [0] GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction

**Authors**: *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html)

**Abstract**:

This paper aims to efficiently enable Large Language Models (LLMs) to use multi-modal tools.The advanced proprietary LLMs, such as ChatGPT and GPT-4, have shown great potential for tool usage through sophisticated prompt engineering.Nevertheless, these models typically rely on prohibitive computational costs and publicly inaccessible data.To address these challenges, we propose the GPT4Tools based on self-instruct to enable open-source LLMs, such as LLaMA and OPT, to use tools.It generates an instruction-following dataset by prompting an advanced teacher with various multi-modal contexts.By using the Low-Rank Adaptation (LoRA) optimization, our approach facilitates the open-source LLMs to solve a range of visual problems, including visual comprehension and image generation.Moreover, we provide a benchmark to evaluate the ability of LLMs to use tools, which is performed in both zero-shot and fine-tuning ways.Extensive experiments demonstrate the effectiveness of our method on various language models, which not only significantly improves the accuracy of invoking seen tools, but also enables the zero-shot capacity for unseen tools.

----

## [0] Reinforcement Learning with Fast and Forgetful Memory

**Authors**: *Steven D. Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html)

**Abstract**:

Nearly all real world tasks are inherently partially observable, necessitating the use of memory in Reinforcement Learning (RL). Most model-free approaches summarize the trajectory into a latent Markov state using memory models borrowed from Supervised Learning (SL), even though RL tends to exhibit different training and efficiency characteristics. Addressing this discrepancy, we introduce Fast and Forgetful Memory, an algorithm-agnostic memory model designed specifically for RL. Our approach constrains the model search space via strong structural priors inspired by computational psychology. It is a drop-in replacement for recurrent neural networks (RNNs) in recurrent RL algorithms, achieving greater reward than RNNs across various recurrent benchmarks and algorithms without changing any hyperparameters. Moreover, Fast and Forgetful Memory exhibits training speeds two orders of magnitude faster than RNNs, attributed to its logarithmic time and linear space complexity. Our implementation is available at https://github.com/proroklab/ffm.

----

## [0] Systematic Visual Reasoning through Object-Centric Relational Abstraction

**Authors**: *Taylor Webb, Shanka Subhra Mondal, Jonathan D. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3cdc587873dd1d00ac78f0c1f9aa60c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3cdc587873dd1d00ac78f0c1f9aa60c-Abstract-Conference.html)

**Abstract**:

Human visual reasoning is characterized by an ability to identify abstract patterns from only a small number of examples, and to systematically generalize those patterns to novel inputs. This capacity depends in large part on our ability to represent complex visual inputs in terms of both objects and relations. Recent work in computer vision has introduced models with the capacity to extract object-centric representations, leading to the ability to process multi-object visual inputs, but falling short of the systematic generalization displayed by human reasoning. Other recent models have employed inductive biases for relational abstraction to achieve systematic generalization of learned abstract rules, but have generally assumed the presence of object-focused inputs. Here, we combine these two approaches, introducing Object-Centric Relational Abstraction (OCRA), a model that extracts explicit representations of both objects and abstract relations, and achieves strong systematic generalization in tasks (including a novel dataset, CLEVR-ART, with greater visual complexity) involving complex visual displays.

----

## [0] In-Context Impersonation Reveals Large Language Models' Strengths and Biases

**Authors**: *Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, Zeynep Akata*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html)

**Abstract**:

In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs' impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs' biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their strengths and hidden biases. Our code is available at https://github.com/ExplainableML/in-context-impersonation.

----

## [0] The s-value: evaluating stability with respect to distributional shifts

**Authors**: *Suyash Gupta, Dominik Rothenhäusler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3fea99df80195b316cefa7aa6099cd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3fea99df80195b316cefa7aa6099cd5-Abstract-Conference.html)

**Abstract**:

Common statistical measures of uncertainty such as $p$-values and confidence intervals quantify the uncertainty due to sampling, that is, the uncertainty due to not observing the full population. However, sampling is not the only source of uncertainty. In practice, distributions change between locations and across time. This makes it difficult to gather knowledge that transfers across data sets. We propose a measure of instability that quantifies the distributional instability of a statistical parameter with respect to Kullback-Leibler divergence, that is, the sensitivity of the parameter under general distributional perturbations within a Kullback-Leibler divergence ball. In addition, we quantify the instability of parameters with respect to directional or variable-specific shifts. Measuring instability with respect to directional shifts can be used to detect under which kind of distribution shifts a statistical conclusion might be reversed. We discuss how such knowledge can inform data collection for transfer learning of statistical parameters under shifted distributions. We evaluate the performance of the proposed measure on real data and show that it can elucidate the distributional instability of a parameter with respect to certain shifts and can be used to improve estimation accuracy under shifted distributions.

----

## [0] When Does Optimizing a Proper Loss Yield Calibration?

**Authors**: *Jaroslaw Blasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html)

**Abstract**:

Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth. Under what circumstances does optimizing proper loss  over a restricted family yield calibrated models? What precise calibration guarantees does it give? In this work, we provide a rigorous answer to these questions. We replace the global optimality with a local optimality condition stipulating that the (proper) loss of the predictor cannot be reduced much by post-processing its predictions with a certain family of Lipschitz functions. We show that any predictor with this local optimality satisfies smooth calibration as defined in [Kakade and Foster, 2008, BÅ‚asiok et al., 2023]. Local optimality is plausibly satisfied by well-trained DNNs, which suggests an explanation for why they are calibrated from proper loss minimization alone. Finally, we show that the connection between local optimality and calibration error goes both ways: nearly calibrated predictors are also nearly locally optimal.

----

## [0] Language Is Not All You Need: Aligning Perception with Language Models

**Authors**: *Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Nils Johan Bertil Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e425b75bac5742a008d643826428787c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e425b75bac5742a008d643826428787c-Abstract-Conference.html)

**Abstract**:

A big convergence of language, multimodal perception, action, and world modeling is a key step toward artificial general intelligence. In this work, we introduce KOSMOS-1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot). Specifically, we train KOSMOS-1 from scratch on web-scale multi-modal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data. We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning. Experimental results show that KOSMOS-1 achieves impressive performance on (i) language understanding, generation, and even OCR-free NLP (directly fed with document images), (ii) perception-language tasks, including multimodal dialogue, image captioning, visual question answering, and (iii) vision tasks, such as image recognition with descriptions (specifying classification via text instructions). We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language. In addition, we introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning capability of MLLMs.

----

## [0] Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources

**Authors**: *Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection discerns OOD data where the predictor cannot make valid predictions as in-distribution (ID) data, thereby increasing the reliability of open-world classification. However, it is typically hard to collect real out-of-distribution (OOD) data for training a predictor capable of discerning ID and OOD patterns. This obstacle gives rise to data generation-based learning methods, synthesizing OOD data via data generators for predictor training without requiring any real OOD data. Related methods typically pre-train a generator on ID data and adopt various selection procedures to find those data likely to be the OOD cases. However, generated data may still coincide with ID semantics, i.e., mistaken OOD generation remains, confusing the predictor between ID and OOD data. To this end, we suggest that generated data (with mistaken OOD generation) can be used to devise an auxiliary OOD detection task to facilitate real OOD detection. Specifically, we can ensure that learning from such an auxiliary task is beneficial if the ID and the OOD parts have disjoint supports, with the help of a well-designed training procedure for the predictor. Accordingly, we propose a powerful data generation-based learning method named Auxiliary Task-based OOD Learning (ATOL) that can relieve the mistaken OOD generation. We conduct extensive experiments under various OOD detection setups, demonstrating the effectiveness of our method against its advanced counterparts.

----

## [0] Robust covariance estimation with missing values and cell-wise contamination

**Authors**: *Grégoire Pacreau, Karim Lounici*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html)

**Abstract**:

Large datasets are often affected by cell-wise outliers in the form of missing or erroneous data. However, discarding any samples containing outliers may result in a dataset that is too small to accurately estimate the covariance matrix. Moreover, the robust procedures designed to address this problem require the invertibility of the covariance operator and thus are not effective on high-dimensional data. In this paper, we propose an unbiased estimator for the covariance in the presence of missing values that does not require any imputation step and still achieves near minimax statistical accuracy with the operator norm. We also advocate for its use in combination with cell-wise outlier detection methods to tackle cell-wise contamination in a high-dimensional and low-rank setting, where state-of-the-art methods may suffer from numerical instability and long computation times. To complement our theoretical findings, we conducted an experimental study which demonstrates the superiority of our approach over the state of the art both in low and high dimension settings.

----

## [0] Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models

**Authors**: *Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4667dd0a5a54b74019b72b677ed8ec1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4667dd0a5a54b74019b72b677ed8ec1-Abstract-Conference.html)

**Abstract**:

Diffusion models are powerful, but they require a lot of time and data to train. We propose Patch Diffusion, a generic patch-wise training framework, to significantly reduce the training time costs while improving data efficiency, which thus helps democratize diffusion model training to broader users. At the core of our innovations is a new conditional score function at the patch level, where the patch location in the original image is included as additional coordinate channels, while the patch size is randomized and diversified throughout training to encode the cross-region dependency at multiple scales. Sampling with our method is as easy as in the original diffusion model. Through Patch Diffusion, we could achieve $\mathbf{\ge 2\times}$ faster training, while maintaining comparable or better generation quality. Patch Diffusion meanwhile improves the performance of diffusion models trained on relatively small datasets, $e.g.$, as few as 5,000 images to train from scratch. We achieve outstanding FID scores in line with state-of-the-art benchmarks: 1.77 on CelebA-64$\times$64, 1.93 on AFHQv2-Wild-64$\times$64, and 2.72 on ImageNet-256$\times$256. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Patch-Diffusion.

----

## [0] Clustering the Sketch: Dynamic Compression for Embedding Tables

**Authors**: *Henry Ling-Hei Tsang, Thomas D. Ahle*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html)

**Abstract**:

Embedding tables are used by machine learning systems  to work with categorical features.In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training.We suggest Clustered Compositional Embeddings (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like The Hashing Trick and Compositional Embeddings [Shi et al., 2020].Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but \emph{dynamically} like hashing-based methods, so it can be used during training.Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.

----

## [0] Dynamic Personalized Federated Learning with Adaptive Differential Privacy

**Authors**: *Xiyuan Yang, Wenke Huang, Mang Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4724af0e2a0d52ce5a0a4e084b87f59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4724af0e2a0d52ce5a0a4e084b87f59-Abstract-Conference.html)

**Abstract**:

Personalized federated learning with differential privacy has been considered a feasible solution to address non-IID distribution of data and privacy leakage risks. However, current personalized federated learning methods suffer from inflexible personalization and convergence difficulties due to two main factors: 1) Firstly, we observe that the prevailing personalization methods mainly achieve this by personalizing a fixed portion of the model, which lacks flexibility. 2) Moreover, we further demonstrate that the default gradient calculation is sensitive to the widely-used clipping operations in differential privacy, resulting in difficulties in convergence. Considering that Fisher information values can serve as an effective measure for estimating the information content of parameters by reflecting the model sensitivity to parameters, we aim to leverage this property to address the aforementioned challenges. In this paper, we propose a novel federated learning method with Dynamic Fisher Personalization and Adaptive Constraint (FedDPA) to handle these challenges. Firstly, by using layer-wise Fisher information to measure the information content of local parameters, we retain local parameters with high Fisher values during the personalization process, which are considered informative, simultaneously prevent these parameters from noise perturbation. Secondly, we introduce an adaptive approach by applying differential constraint strategies to personalized parameters and shared parameters identified in the previous for better convergence.  Our method boosts performance through flexible personalization while mitigating the slow convergence caused by clipping operations. Experimental results on CIFAR-10, FEMNIST and SVHN dataset demonstrate the effectiveness of our approach in achieving better performance and robustness against clipping, under personalized federated learning with differential privacy.

----

## [0] Bias in Evaluation Processes: An Optimization-Based Model

**Authors**: *L. Elisa Celis, Amit Kumar, Anay Mehrotra, Nisheeth K. Vishnoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4748b6b6ca49f04b6a8cfce1d5f9a70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4748b6b6ca49f04b6a8cfce1d5f9a70-Abstract-Conference.html)

**Abstract**:

Biases with respect to socially-salient attributes of individuals have been well documented in evaluation processes used in settings such as admissions and hiring. We view such an evaluation process as a transformation of a  distribution of the true utility of an individual for a task to an observed distribution and model it as a solution to a loss minimization problem subject to an information constraint. Our model has two parameters that have been identified as factors leading to biases: the resource-information trade-off parameter in the information constraint and the risk-averseness parameter in the loss function.  We characterize the distributions that arise from our model and study the effect of the parameters on the observed distribution. The outputs of our model enrich the class of distributions that can be used to capture variation across groups in the observed evaluations. We empirically validate our model by fitting real-world datasets and use it to study the effect of interventions in a downstream selection task. These results contribute to an understanding of the emergence of bias in evaluation processes and provide tools to guide the deployment of interventions to mitigate biases.

----

## [0] Data Minimization at Inference Time

**Authors**: *Cuong Tran, Nando Fioretto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e48880ea81caa7836e6a0694049093ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e48880ea81caa7836e6a0694049093ae-Abstract-Conference.html)

**Abstract**:

In high-stakes domains such as legal, banking, hiring, and healthcare, learning models frequently rely on sensitive user information for inference, necessitating the complete set of features. This not only poses significant privacy risks for individuals but also demands substantial human effort from organizations to verify information accuracy. This study asks whether it is necessary to use all input features for accurate predictions at inference time. The paper demonstrates that, in a personalized setting, individuals may only need to disclose a small subset of features without compromising decision-making accuracy. The paper also provides an efficient sequential algorithm to determine the appropriate attributes for each individual to provide. Evaluations across various learning tasks show that individuals can potentially report as little as 10\% of their information while maintaining the same accuracy level as a model that employs the full set of user information.

----

## [0] Learning Adaptive Tensorial Density Fields for Clean Cryo-ET Reconstruction

**Authors**: *Yuanhao Wang, Ramzi Idoughi, Wolfgang Heidrich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4be7e9867ef163563f4a5e90cec478f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4be7e9867ef163563f4a5e90cec478f-Abstract-Conference.html)

**Abstract**:

We present a novel learning-based framework for reconstructing 3D structures from tilt-series cryo-Electron Tomography (cryo-ET) data. Cryo-ET is a powerful imaging technique that can achieve near-atomic resolutions. Still, it suffers from challenges such as missing-wedge acquisition, large data size, and high noise levels. Our framework addresses these challenges by using an adaptive tensorial-based representation for the 3D density field of the scanned sample. First, we optimize a quadtree structure to partition the volume of interest. Then, we learn a vector-matrix factorization of the tensor representing the density field in each node. Moreover, we use a loss function that combines a differentiable tomographic formation model with three regularization terms: total variation, boundary consistency constraint, and an isotropic Fourier prior. Our framework allows us to query the density at any location using the learned representation and obtain a high-quality 3D tomogram. We demonstrate the superiority of our framework over existing methods using synthetic and real data. Thus, our framework boosts the quality of the reconstruction while reducing the computation time and the memory footprint. The code is available at https://github.com/yuanhaowang1213/adaptivetensordf.

----

## [0] Resetting the Optimizer in Deep RL: An Empirical Study

**Authors**: *Kavosh Asadi, Rasool Fakoor, Shoham Sabach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html)

**Abstract**:

We focus on the task of approximating the optimal value function in deep reinforcement learning. This iterative process is comprised of solving a sequence of optimization problems where the loss function changes per iteration. The common approach to solving this sequence of problems is to employ modern variants of the stochastic gradient descent algorithm such as Adam. These optimizers maintain their own internal parameters such as estimates of the first-order and the second-order moments of the gradient, and update them over time. Therefore, information obtained in previous iterations is used to solve the optimization problem in the current iteration. We demonstrate that this can contaminate the moment estimates because the optimization landscape can change arbitrarily from one iteration to the next one. To hedge against this negative effect, a simple idea is to reset the internal parameters of the optimizer when starting a new iteration. We empirically investigate this resetting idea by employing various optimizers in conjunction with the Rainbow algorithm. We demonstrate that this simple modification significantly improves the performance of deep RL on the Atari benchmark.

----

## [0] Why Does Sharpness-Aware Minimization Generalize Better Than SGD?

**Authors**: *Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, Quanquan Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4d3fe32495088805bbbb4f1de63e947-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4d3fe32495088805bbbb4f1de63e947-Abstract-Conference.html)

**Abstract**:

The challenge of overfitting, in which the model memorizes the training data and fails to generalize to test data, has become increasingly significant in the training of large neural networks. To tackle this challenge, Sharpness-Aware Minimization (SAM) has emerged as a promising training method, which can improve the generalization of neural networks even in the presence of label noise. However, a deep understanding of how SAM works, especially in the setting of nonlinear neural networks and classification tasks, remains largely missing. This paper fills this gap by demonstrating why SAM generalizes better than Stochastic Gradient Descent (SGD) for a certain data model and two-layer convolutional ReLU networks. The loss landscape of our studied problem is nonsmooth, thus current explanations for the success of SAM based on the Hessian information are insufficient. Our result explains the benefits of SAM, particularly its ability to prevent noise learning in the early stages, thereby facilitating more effective learning of features. Experiments on both synthetic and real data corroborate our theory.

----

## [0] Grassmann Manifold Flows for Stable Shape Generation

**Authors**: *Ryoma Yataka, Kazuki Hirashima, Masashi Shiraishi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e50e253e21cbcdcd200394f61d73acc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e50e253e21cbcdcd200394f61d73acc8-Abstract-Conference.html)

**Abstract**:

Recently, studies on machine learning have focused on methods that use symmetry implicit in a specific manifold as an inductive bias.Grassmann manifolds provide the ability to handle fundamental shapes represented as shape spaces, enabling stable shape analysis. In this paper, we present a novel approach in which we establish the theoretical foundations for learning distributions on the Grassmann manifold via continuous normalization flows, with the explicit goal of generating stable shapes.Our approach facilitates more robust generation by effectively eliminating the influence of extraneous transformations, such as rotations and inversions, through learning and generating within a Grassmann manifold designed to accommodate the essential shape information of the object.The experimental results indicated that the proposed method could generate high-quality samples by capturing the data structure.Furthermore, the proposed method significantly outperformed state-of-the-art methods in terms of the log-likelihood or evidence lower bound.The results obtained are expected to stimulate further research in this field, leading to advances for stable shape generation and analysis.

----

## [0] Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack

**Authors**: *Pratik Karmakar, Debabrota Basu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5440ffceaf4831b5f98652b8a27ffde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5440ffceaf4831b5f98652b8a27ffde-Abstract-Conference.html)

**Abstract**:

We study design of black-box model extraction attacks that can *send minimal number of queries from* a *publicly available dataset* to a target ML model through a predictive API with an aim *to create an informative and distributionally equivalent replica* of the target.First, we define *distributionally equivalent* and *Max-Information model extraction* attacks, and reduce them into a variational optimisation problem. The attacker sequentially solves this optimisation problem to select the most informative queries that simultaneously maximise the entropy and reduce the mismatch between the target and the stolen models. This leads to *an active sampling-based query selection algorithm*, Marich, which is *model-oblivious*. Then, we evaluate Marich on different text and image data sets, and different models, including CNNs and BERT. Marich extracts models that achieve $\sim 60-95\%$ of true model's accuracy and uses $\sim 1,000 - 8,500$ queries from the publicly available datasets, which are different from the private training datasets. Models extracted by Marich yield prediction distributions, which are $\sim2-4\times$ closer to the target's distribution in comparison to the existing active sampling-based attacks. The extracted models also lead to 84-96$\%$ accuracy under membership inference attacks. Experimental results validate that Marich is *query-efficient*, and capable of performing task-accurate, high-fidelity, and informative model extraction.

----

## [0] Evaluating Post-hoc Explanations for Graph Neural Networks via Robustness Analysis

**Authors**: *Junfeng Fang, Wei Liu, Yuan Gao, Zemin Liu, An Zhang, Xiang Wang, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e55c2f3fdde519014c879aa3554414c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e55c2f3fdde519014c879aa3554414c0-Abstract-Conference.html)

**Abstract**:

This work studies the evaluation of explaining graph neural networks (GNNs), which is crucial to the credibility of post-hoc explainability in practical usage. Conventional evaluation metrics, and even explanation methods -- which mainly follow the paradigm of feeding the explanatory subgraph and measuring output difference -- always suffer from the notorious out-of-distribution (OOD) issue. In this work, we endeavor to confront the issue by introducing a novel evaluation metric, termed OOD-resistant Adversarial Robustness (OAR). Specifically, we draw inspiration from the notion of adversarial robustness and evaluate post-hoc explanation subgraphs by calculating their robustness under attack. On top of that, an elaborate OOD reweighting block is inserted into the pipeline to confine the evaluation process to the original data distribution. For applications involving large datasets, we further devise a Simplified version of OAR (SimOAR), which achieves a significant improvement in computational efficiency at the cost of a small amount of performance. Extensive empirical studies validate the effectiveness of our OAR and SimOAR.

----

## [0] A Unifying Perspective on Multi-Calibration: Game Dynamics for Multi-Objective Learning

**Authors**: *Nika Haghtalab, Michael I. Jordan, Eric Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e55edcdb01ac45c839a602f96e09fbcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e55edcdb01ac45c839a602f96e09fbcb-Abstract-Conference.html)

**Abstract**:

We provide a unifying framework for the design and analysis of multi-calibrated predictors. By placing the multi-calibration problem in the general setting of multi-objective learning---where learning guarantees must hold simultaneously over a set of distributions and loss functions---we exploit connections to game dynamics to achieve state-of-the-art guarantees for a diverse set of multi-calibration learning problems. In addition to shedding light on existing multi-calibration guarantees and greatly simplifying their analysis, our approach also yields improved guarantees, such as error tolerances that scale with the square-root of group size versus the constant tolerances guaranteed by prior works, and improving the complexity of $k$-class multi-calibration by an exponential factor of $k$ versus Gopalan et al.. Beyond multi-calibration, we use these game dynamics to address emerging considerations in the study of group fairness and multi-distribution learning.

----

## [0] Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts

**Authors**: *Emanuele Marconato, Stefano Teso, Antonio Vergari, Andrea Passerini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html)

**Abstract**:

Neuro-Symbolic (NeSy) predictive models hold the promise of improved compliance with given constraints, systematic generalization, and interpretability, as they allow to infer labels that are consistent with some prior knowledge by reasoning over high-level concepts extracted from sub-symbolic inputs. It was recently shown that NeSy predictors are affected by reasoning shortcuts: they can attain high accuracy but by leveraging concepts with \textit{unintended semantics}, thus coming short of their promised advantages. Yet, a systematic characterization of reasoning shortcuts and of potential mitigation strategies is missing. This work fills this gap by characterizing them as unintended optima of the learning objective and identifying four key conditions behind their occurrence. Based on this, we derive several natural mitigation strategies, and analyze their efficacy both theoretically and empirically. Our analysis shows reasoning shortcuts are difficult to deal with, casting doubts on the trustworthiness and interpretability of existing NeSy solutions.

----

## [0] Contrastive Moments: Unsupervised Halfspace Learning in Polynomial Time

**Authors**: *Xinyuan Cao, Santosh S. Vempala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5a71ba556c84fef542aaace56b6cfe9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5a71ba556c84fef542aaace56b6cfe9-Abstract-Conference.html)

**Abstract**:

We give a polynomial-time algorithm for learning high-dimensional halfspaces with margins in $d$-dimensional space to within desired Total Variation (TV) distance when the ambient distribution is an unknown affine transformation of the $d$-fold product of an (unknown) symmetric one-dimensional logconcave distribution, and the halfspace is introduced by deleting at least an $\epsilon$ fraction of the data in one of the component distributions. Notably, our algorithm does not need labels and establishes the unique (and efficient) identifiability of the hidden halfspace under this distributional assumption.  The sample and time complexity of the algorithm are polynomial in the dimension and $1/\epsilon$. The algorithm uses only the first two moments of *suitable re-weightings* of the empirical distribution, which we call *contrastive moments*; its analysis uses classical facts about generalized Dirichlet polynomials and relies crucially on a new monotonicity property of the moment ratio of truncations of logconcave distributions. Such algorithms, based only on first and second moments were suggested in earlier work, but hitherto eluded rigorous guarantees.Prior work addressed the special case when the underlying distribution is Gaussian via Non-Gaussian Component Analysis. We improve on this by providing polytime guarantees based on TV distance, in place of existing moment-bound guarantees that can be super-polynomial. Our work is also the first to go beyond Gaussians in this setting.

----

## [0] Boosting Spectral Clustering on Incomplete Data via Kernel Correction and Affinity Learning

**Authors**: *Fangchen Yu, Runze Zhao, Zhan Shi, Yiwen Lu, Jicong Fan, Yicheng Zeng, Jianfeng Mao, Wenye Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html)

**Abstract**:

Spectral clustering has gained popularity for clustering non-convex data due to its simplicity and effectiveness. It is essential to construct a similarity graph using a high-quality affinity measure that models the local neighborhood relations among the data samples. However, incomplete data can lead to inaccurate affinity measures, resulting in degraded clustering performance. To address these issues, we propose an imputation-free framework with two novel approaches to improve spectral clustering on incomplete data. Firstly, we introduce a new kernel correction method that enhances the quality of the kernel matrix estimated on incomplete data with a theoretical guarantee, benefiting classical spectral clustering on pre-defined kernels. Secondly, we develop a series of affinity learning methods that equip the self-expressive framework with $\ell_p$-norm to construct an intrinsic affinity matrix with an adaptive extension. Our methods outperform existing data imputation and distance calibration techniques on benchmark datasets, offering a promising solution to spectral clustering on incomplete data in various real-world applications.

----

## [0] DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation

**Authors**: *Qingkai Fang, Yan Zhou, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)

**Abstract**:

Direct speech-to-speech translation (S2ST) translates speech from one language into another using a single model. However, due to the presence of linguistic and acoustic diversity, the target speech follows a complex multimodal distribution, posing challenges to achieving both high-quality translations and fast decoding speeds for S2ST models. In this paper, we propose DASpeech, a non-autoregressive direct S2ST model which realizes both fast and high-quality S2ST. To better capture the complex distribution of the target speech, DASpeech adopts the two-pass architecture to decompose the generation process into two steps, where a linguistic decoder first generates the target text, and an acoustic decoder then generates the target speech based on the hidden states of the linguistic decoder. Specifically, we use the decoder of DA-Transformer as the linguistic decoder, and use FastSpeech 2 as the acoustic decoder. DA-Transformer models translations with a directed acyclic graph (DAG). To consider all potential paths in the DAG during training, we calculate the expected hidden states for each target token via dynamic programming, and feed them into the acoustic decoder to predict the target mel-spectrogram. During inference, we select the most probable path and take hidden states on that path as input to the acoustic decoder. Experiments on the CVSS Fr$\rightarrow$En benchmark demonstrate that DASpeech can achieve comparable or even better performance than the state-of-the-art S2ST model Translatotron 2, while preserving up to 18.53$\times$ speedup compared to the autoregressive baseline. Compared with the previous non-autoregressive S2ST model, DASpeech does not rely on knowledge distillation and iterative decoding, achieving significant improvements in both translation quality and decoding speed. Furthermore, DASpeech shows the ability to preserve the speaker's voice of the source speech during translation.

----

## [0] Learning Large-scale Neural Fields via Context Pruned Meta-Learning

**Authors**: *Jihoon Tack, Subin Kim, Sihyun Yu, Jaeho Lee, Jinwoo Shin, Jonathan Richard Schwarz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b5c402bb7bd5e60bede6961d6fe39e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b5c402bb7bd5e60bede6961d6fe39e-Abstract-Conference.html)

**Abstract**:

We introduce an efficient optimization-based meta-learning technique for large-scale neural field training by realizing significant memory savings through automated online context point selection. This is achieved by focusing each learning step on the subset of data with the highest expected immediate improvement in model quality, resulting in the almost instantaneous modeling of global structure and subsequent refinement of high-frequency details. We further improve the quality of our meta-learned initialization by introducing a bootstrap correction resulting in the minimization of any error introduced by reduced context sets while simultaneously mitigating the well-known myopia of optimization-based meta-learning. Finally, we show how gradient re-scaling at meta-test time allows the learning of extremely high-quality neural fields in significantly shortened optimization procedures. Our framework is model-agnostic, intuitive, straightforward to implement, and shows significant reconstruction improvements for a wide range of signals. We provide an extensive empirical evaluation on nine datasets across multiple multiple modalities, demonstrating state-of-the-art results while providing additional insight through careful analysis of the algorithmic components constituting our method. Code is available at https://github.com/jihoontack/GradNCP

----

## [0] AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation

**Authors**: *Daiki E. Matsunaga, Jongmin Lee, Jaeseok Yoon, Stefanos Leonardos, Pieter Abbeel, Kee-Eung Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b6eb1dbabff82838d5e99f62de37c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b6eb1dbabff82838d5e99f62de37c8-Abstract-Conference.html)

**Abstract**:

One of the main challenges in offline Reinforcement Learning (RL) is the distribution shift that arises from the learned policy deviating from the data collection policy. This is often addressed by avoiding out-of-distribution (OOD) actions during policy improvement as their presence can lead to substantial performance degradation. This challenge is amplified in the offline Multi-Agent RL (MARL) setting since the joint action space grows exponentially with the number of agents.To avoid this curse of dimensionality, existing MARL methods adopt either value decomposition methods or fully decentralized training of individual agents. However, even when combined with standard conservatism principles, these methods can still result in the selection of OOD joint actions in offline MARL. To this end, we introduce AlberDICE,an offline MARL algorithm that alternatively performs centralized training of individual agents based on stationary distribution optimization. AlberDICE circumvents the exponential complexity of MARL by computing the best response of one agent at a time while effectively avoiding OOD joint action selection. Theoretically, we show that the alternating optimization procedure converges to Nash policies. In the experiments, we demonstrate that AlberDICE significantly outperforms baseline algorithms on a standard suite of MARL benchmarks.

----

## [0] Approximate inference of marginals using the IBIA framework

**Authors**: *Shivani Bathla, Vinita Vasudevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5beb17e56bbb8fd562efeefab79425f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5beb17e56bbb8fd562efeefab79425f-Abstract-Conference.html)

**Abstract**:

Exact inference of marginals in probabilistic graphical models (PGM) is known to be intractable, necessitating the use of approximate methods. Most of the existing variational techniques perform iterative message passing in loopy graphs which is slow to converge for many benchmarks. In this paper, we propose a new algorithm for marginal inference that is based on the incremental build-infer-approximate (IBIA) paradigm. Our algorithm converts the PGM into a sequence of linked clique tree forests (SLCTF) with bounded clique sizes, and then uses a heuristic belief update algorithm to infer the marginals. For the special case of Bayesian networks, we show that if the incremental build step in IBIA uses the topological order of variables then (a) the prior marginals are consistent in all CTFs in the SLCTF  and (b) the posterior marginals are consistent once all evidence variables are added to the SLCTF. In our approach, the belief propagation step is non-iterative and the accuracy-complexity trade-off is controlled using user-defined clique size bounds. Results for several benchmark sets from recent UAI competitions show that our method gives either better or comparable accuracy than existing variational and sampling based methods, with smaller runtimes.

----

## [0] HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation

**Authors**: *Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, David Bull*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html)

**Abstract**:

Learning-based video compression is currently a popular research topic, offering the potential to compete with conventional standard video codecs. In this context, Implicit Neural Representations (INRs) have previously been used to represent and compress image and video content, demonstrating relatively high decoding speed compared to other methods. However, existing INR-based methods have failed to deliver rate quality performance comparable with the state of the art in video compression. This is mainly due to the simplicity of the employed network architectures, which limit their representation capability. In this paper, we propose HiNeRV, an INR that combines light weight layers with novel hierarchical positional encodings. We employs depth-wise convolutional, MLP and interpolation layers to build the deep and wide network architecture with high capacity. HiNeRV is also a unified representation encoding videos in both frames and patches at the same time, which offers higher performance and flexibility than existing methods. We further build a video codec based on HiNeRV and a refined pipeline for training, pruning and quantization that can better preserve HiNeRV's performance during lossy model compression. The proposed method has been evaluated on both UVG and MCL-JCV datasets for video compression, demonstrating significant improvement over all existing INRs baselines and competitive performance when compared to learning-based codecs (72.3\% overall bit rate saving over HNeRV and 43.4\% over DCVC on the UVG dataset, measured in PSNR).

----

## [0] Bicriteria Approximation Algorithms for the Submodular Cover Problem

**Authors**: *Wenjing Chen, Victoria G. Crawford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5eaf67f3405be58cd12848a89cd8ace-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5eaf67f3405be58cd12848a89cd8ace-Abstract-Conference.html)

**Abstract**:

In this paper, we consider the optimization problem Submodular Cover (SCP), which is to find a minimum cardinality subset of a finite universe $U$ such that the value of a submodular function $f$ is above an input threshold $\tau$. In particular, we consider several variants of SCP including the general case, the case where $f$ is additionally assumed to be monotone, and finally the case where $f$ is a regularized monotone submodular function. Our most significant contributions are that: (i) We propose a scalable algorithm for monotone SCP that achieves nearly the same approximation guarantees as the standard greedy algorithm in significantly faster time; (ii) We are the first to develop an algorithm for general SCP that achieves a solution arbitrarily close to being feasible; and finally (iii) we are the first to develop algorithms for regularized SCP. Our algorithms are then demonstrated to be effective in an extensive experimental section on data summarization and graph cut, two applications of SCP.

----

## [0] Responsible AI (RAI) Games and Ensembles

**Authors**: *Yash Gupta, Runtian Zhai, Arun Sai Suggala, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6057bf047bcc5f86ebf4e8db6e24a1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6057bf047bcc5f86ebf4e8db6e24a1f-Abstract-Conference.html)

**Abstract**:

Several recent works have studied the societal effects of AI; these include issues such as fairness, robustness, and safety.  In many of these objectives, a learner seeks to minimize its worst-case loss over a set of predefined distributions (known as uncertainty sets), with usual examples being perturbed versions of the empirical distribution. In other words, the aforementioned problems can be written as min-max problems over these uncertainty sets. In this work, we provide a general framework for studying these problems, which we refer to as Responsible AI (RAI) games. We provide two classes of algorithms for solving these games:  (a) game-play based algorithms, and (b) greedy stagewise estimation algorithms. The former class is motivated by online learning and game theory, whereas the latter class is motivated by the classical statistical literature on boosting, and regression. We empirically demonstrate the applicability and competitive performance of our techniques for solving several RAI problems, particularly around subpopulation shift.

----

## [0] May the Force be with You: Unified Force-Centric Pre-Training for 3D Molecular Conformations

**Authors**: *Rui Feng, Qi Zhu, Huan Tran, Binghong Chen, Aubrey Toland, Rampi Ramprasad, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)

**Abstract**:

Recent works have shown the promise of learning pre-trained models for 3D molecular representation.However, existing pre-training models focus predominantly on equilibrium data and largely overlook off-equilibrium conformations.It is challenging to extend these methods to off-equilibrium data because their training objective relies on assumptions ofconformations being the local energy minima. We address this gap by proposing a force-centric pretraining model for 3D molecular conformations covering both equilibrium and off-equilibrium data.For off-equilibrium data, our model learns directly from their atomic forces. For equilibrium data, we introduce zero-force regularization and forced-based denoising techniques to approximate near-equilibrium forces.We obtain a unified pre-trained model for 3D molecular representation with over 15 million diverse conformations. Experiments show that, with our pre-training objective, we increase forces accuracy by around 3 times compared to the un-pre-trained Equivariant Transformer model. By incorporating regularizations on equilibrium data, we solved the problem of unstable MD simulations in vanilla Equivariant Transformers, achieving state-of-the-art simulation performance with 2.45 times faster inference time than NequIP.  As a powerful molecular encoder, our pre-trained model achieves on-par performance with state-of-the-art property prediction tasks.

----

## [0] Deep Fractional Fourier Transform

**Authors**: *Hu Yu, Jie Huang, Lingzhi Li, Man Zhou, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e66309ead63bc1410d2df261a28f602d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e66309ead63bc1410d2df261a28f602d-Abstract-Conference.html)

**Abstract**:

Existing deep learning-based computer vision methods usually operate in the spatial and frequency domains, which are two orthogonal \textbf{individual} perspectives for image processing.In this paper, we introduce a new spatial-frequency analysis tool, Fractional Fourier Transform (FRFT), to provide comprehensive \textbf{unified} spatial-frequency perspectives.The FRFT is a unified continuous spatial-frequency transform that simultaneously reflects an image's spatial and frequency representations, making it optimal for processing non-stationary image signals.We explore the properties of the FRFT for image processing and present a fast implementation of the 2D FRFT, which facilitates its widespread use.Based on these explorations, we introduce a simple yet effective operator, Multi-order FRactional Fourier Convolution (MFRFC), which exhibits the remarkable merits of processing images from more perspectives in the spatial-frequency plane. Our proposed MFRFC is a general and basic operator that can be easily integrated into various tasks for performance improvement.We experimentally evaluate the MFRFC on various computer vision tasks, including object detection, image classification, guided super-resolution, denoising, dehazing, deraining, and low-light enhancement. Our proposed MFRFC consistently outperforms baseline methods by significant margins across all tasks.

----

## [0] Survival Permanental Processes for Survival Analysis with Time-Varying Covariates

**Authors**: *Hideaki Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html)

**Abstract**:

Survival or time-to-event data with time-varying covariates are common in practice, and exploring the non-stationarity in covariates is essential to accurately analyzing the nonlinear dependence of time-to-event outcomes on covariates. Traditional survival analysis methods such as Cox proportional hazards model have been extended to address the time-varying covariates through a counting process formulation, although sophisticated machine learning methods that can accommodate time-varying covariates have been limited. In this paper, we propose a non-parametric Bayesian survival model to analyze the nonlinear dependence of time-to-event outcomes on time-varying covariates. We focus on a computationally feasible Cox process called permanental process, which assumes the square root of hazard function to be generated from a Gaussian process, and tailor it for survival data with time-varying covariates. We verify that the proposed model holds with the representer theorem, a beneficial property for functional analysis, which offers us a fast Bayesian estimation algorithm that scales linearly with the number of observed events without relying on Markov Chain Monte Carlo computation. We evaluate our algorithm on synthetic and real-world data, and show that it achieves comparable predictive accuracy while being tens to hundreds of times faster than state-of-the-art methods.

----

## [0] Learn to Categorize or Categorize to Learn? Self-Coding for Generalized Category Discovery

**Authors**: *Sarah Rastegar, Hazel Doughty, Cees Snoek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6789e468c65a7816760a00a487d3c4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6789e468c65a7816760a00a487d3c4e-Abstract-Conference.html)

**Abstract**:

In the quest for unveiling novel categories at test time, we confront the inherent limitations of traditional supervised recognition models that are restricted by a predefined category set. While strides have been made in the realms of self-supervised and open-world learning towards test-time category discovery, a crucial yet often overlooked question persists: what exactly delineates a category? In this paper, we conceptualize a category through the lens of optimization, viewing it as an optimal solution to a well-defined problem. Harnessing this unique conceptualization, we propose a novel, efficient and self-supervised method capable of discovering previously unknown categories at test time. A salient feature of our approach is the assignment of minimum length category codes to individual data instances, which encapsulates the implicit category hierarchy prevalent in real-world datasets. This mechanism affords us enhanced control over category granularity, thereby equipping our model to handle fine-grained categories adeptly. Experimental evaluations, bolstered by state-of-the-art benchmark comparisons, testify to the efficacy of our solution in managing unknown categories at test time. Furthermore, we fortify our proposition with a theoretical foundation, providing proof of its optimality. Our code is available at: https://github.com/SarahRastegar/InfoSieve.

----

## [0] Training Chain-of-Thought via Latent-Variable Inference

**Authors**: *Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e69a9560c450ca76584d9eb37e7f5ae8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e69a9560c450ca76584d9eb37e7f5ae8-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) solve problems more accurately and interpretably when instructed to work out the answer step by step using a "chain-of-thought" (CoT) prompt. One can also improve LLMs' performance on a specific task by supervised fine-tuning, i.e., by using gradient ascent on some tunable parameters to maximize the average log-likelihood of correct answers from a labeled training set. Naively combining CoT with supervised tuning requires supervision not just of the correct answers, but also of detailed rationales that lead to those answers; these rationales are expensive to produce by hand. Instead, we propose a fine-tuning strategy that tries to maximize the \emph{marginal} log-likelihood of generating a correct answer using CoT prompting, approximately averaging over all possible rationales. The core challenge is sampling from the posterior over rationales conditioned on the correct answer; we address it using a simple Markov-chain Monte Carlo (MCMC) expectation-maximization (EM) algorithm inspired by the self-taught reasoner (STaR), memoized wake-sleep, Markovian score climbing, and persistent contrastive divergence. This algorithm also admits a novel control-variate technique that drives the variance of our gradient estimates to zero as the model improves. Applying our technique to GSM8K and the tasks in BIG-Bench Hard, we find that this MCMC-EM fine-tuning technique typically improves the model's accuracy on held-out examples more than STaR or prompt-tuning with or without CoT.

----

## [0] VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset

**Authors**: *Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, Jing Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6b2b48b5ed90d07c305932729927781-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6b2b48b5ed90d07c305932729927781-Abstract-Conference.html)

**Abstract**:

Vision and text have been fully explored  in contemporary video-text foundational models, while other modalities such as audio and subtitles in videos have not received sufficient attention. In this paper, we resort to establish connections between multi-modality video tracks, including Vision, Audio, and Subtitle, and Text by exploring an automatically generated large-scale omni-modality video caption dataset called VAST-27M. Specifically, we first collect 27 million open-domain video clips and separately train a vision and an audio captioner to generate vision and audio captions. Then, we employ an off-the-shelf Large Language Model (LLM) to integrate the generated captions, together with subtitles and instructional prompts into omni-modality captions. Based on the proposed VAST-27M dataset, we train an omni-modality video-text foundational model named VAST, which can perceive and process vision, audio, and subtitle modalities from video, and better support various tasks including  vision-text, audio-text, and multi-modal video-text tasks (retrieval, captioning and QA). Extensive experiments have been conducted to demonstrate the effectiveness of our proposed VAST-27M corpus and VAST foundation model. VAST achieves 22 new state-of-the-art results on various cross-modality benchmarks.

----

## [0] Bayesian Learning via Q-Exponential Process

**Authors**: *Shuyi Li, Michael O'Connor, Shiwei Lan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html)

**Abstract**:

Regularization is one of the most fundamental topics in optimization, statistics and machine learning. To get sparsity in estimating a parameter $u\in\mathbb{R}^d$, an $\ell_q$ penalty term, $\Vert u\Vert_q$, is usually added to the objective function. What is the probabilistic distribution corresponding to such $\ell_q$ penalty? What is the \emph{correct} stochastic process corresponding to $\Vert u\Vert_q$ when we model functions $u\in L^q$? This is important for statistically modeling high-dimensional objects such as images, with penalty to preserve certainty properties, e.g. edges in the image.In this work, we generalize the $q$-exponential distribution (with density proportional to) $\exp{(- \frac{1}{2}|u|^q)}$ to a stochastic process named \emph{$Q$-exponential (Q-EP) process} that corresponds to the $L_q$ regularization of functions. The key step is to specify consistent multivariate $q$-exponential distributions by choosing from a large family of elliptic contour distributions. The work is closely related to Besov process which is usually defined in terms of series. Q-EP can be regarded as a definition of Besov process with explicit probabilistic formulation, direct control on the correlation strength, and tractable prediction formula. From the Bayesian perspective, Q-EP provides a flexible prior on functions with sharper penalty ($q<2$) than the commonly used Gaussian process (GP, $q=2$).We compare GP, Besov and Q-EP in modeling functional data, reconstructing images and solving inverse problems and demonstrate the advantage of our proposed methodology.

----

## [0] Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective

**Authors**: *Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html)

**Abstract**:

There are a number of diverging hypotheses about the neural text degeneration problem, i.e., generating repetitive and dull loops, which makes this problem both interesting and confusing. In this work, we aim to advance our understanding by presenting a straightforward and fundamental explanation from the data perspective. Our preliminary investigation reveals a strong correlation between the degeneration issue and the presence of repetitions in training data. Subsequent experiments also demonstrate that by selectively dropping out the attention to repetitive words in training data, degeneration can be significantly minimized. Furthermore, our empirical analysis illustrates that prior works addressing the degeneration issue from various standpoints, such as the high-inflow words, the likelihood objective, and the self-reinforcement phenomenon, can be interpreted by one simple explanation. That is, penalizing the repetitions in training data is a common and fundamental factor for their effectiveness. Moreover, our experiments reveal that penalizing the repetitions in training data remains critical even when considering larger model sizes and instruction tuning.

----

## [0] Facing Off World Model Backbones: RNNs, Transformers, and S4

**Authors**: *Fei Deng, Junyeong Park, Sungjin Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html)

**Abstract**:

World models are a fundamental component in model-based reinforcement learning (MBRL). To perform temporally extended and consistent simulations of the future in partially observable environments, world models need to possess long-term memory. However, state-of-the-art MBRL agents, such as Dreamer, predominantly employ recurrent neural networks (RNNs) as their world model backbone, which have limited memory capacity. In this paper, we seek to explore alternative world model backbones for improving long-term memory. In particular, we investigate the effectiveness of Transformers and Structured State Space Sequence (S4) models, motivated by their remarkable ability to capture long-range dependencies in low-dimensional sequences and their complementary strengths. We propose S4WM, the first world model compatible with parallelizable SSMs including S4 and its variants. By incorporating latent variable modeling, S4WM can efficiently generate high-dimensional image sequences through latent imagination. Furthermore, we extensively compare RNN-, Transformer-, and S4-based world models across four sets of environments, which we have tailored to assess crucial memory capabilities of world models, including long-term imagination, context-dependent recall, reward prediction, and memory-based reasoning. Our findings demonstrate that S4WM outperforms Transformer-based world models in terms of long-term memory, while exhibiting greater efficiency during training and imagination. These results pave the way for the development of stronger MBRL agents.

----

## [0] STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events

**Authors**: *Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel Aleksander Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, Tuomas Virtanen, Yuki Mitsufuji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c9671ed3b3106b71cafda3ba225c1a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c9671ed3b3106b71cafda3ba225c1a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While direction of arrival (DOA) of sound events is generally estimated from multichannel audio data recorded in a microphone array, sound events usually derive from visually perceptible source objects, e.g., sounds of footsteps come from the feet of a walker. This paper proposes an audio-visual sound event localization and detection (SELD) task, which uses multichannel audio and video information to estimate the temporal activation and DOA of target sound events. Audio-visual SELD systems can detect and localize sound events using signals from a microphone array and audio-visual correspondence. We also introduce an audio-visual dataset, Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23), which consists of multichannel audio data recorded with a microphone array, video data, and spatiotemporal annotation of sound events. Sound scenes in STARSS23 are recorded with instructions, which guide recording participants to ensure adequate activity and occurrences of sound events. STARSS23 also serves human-annotated temporal activation labels and human-confirmed DOA labels, which are based on tracking results of a motion capture system. Our benchmark results demonstrate the benefits of using visual object positions in audio-visual SELD tasks. The data is available at https://zenodo.org/record/7880637.

----

## [0] Inserting Anybody in Diffusion Models via Celeb Basis

**Authors**: *Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, Huicheng Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html)

**Abstract**:

Exquisite demand exists for customizing the pretrained large text-to-image model, $e.g.$ Stable Diffusion, to generate innovative concepts, such as the users themselves. However, the newly-added concept from previous customization methods often shows weaker combination abilities than the original ones even given several images during training. We thus propose a new personalization method that allows for the seamless integration of a unique individual into the pre-trained diffusion model using just $one\ facial\ photograph$ and only $1024\ learnable\ parameters$ under $3\ minutes$. So we can effortlessly generate stunning images of this person in any pose or position, interacting with anyone and doing anything imaginable from text prompts. To achieve this, we first analyze and build a well-defined celeb basis from the embedding space of the pre-trained large text encoder. Then, given one facial photo as the target identity, we generate its own embedding by optimizing the weight of this basis and locking all other parameters. Empowered by the proposed celeb basis, the new identity in our customized model showcases a better concept combination ability than previous personalization methods. Besides, our model can also learn several new identities at once and interact with each other where the previous customization model fails to. Project page is at: http://celeb-basis.github.io. Code is at: https://github.com/ygtxr1997/CelebBasis.

----

## [0] Scaling Open-Vocabulary Object Detection

**Authors**: *Matthias Minderer, Alexey A. Gritsenko, Neil Houlsby*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html)

**Abstract**:

Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling. Code and checkpoints are available on GitHub.

----

## [0] Formulating Discrete Probability Flow Through Optimal Transport

**Authors**: *Pengze Zhang, Hubery Yin, Chen Li, Xiaohua Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6e706454d72c18582b9c1ff70b11f7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6e706454d72c18582b9c1ff70b11f7d-Abstract-Conference.html)

**Abstract**:

Continuous diffusion models are commonly acknowledged to display a deterministic probability flow, whereas discrete diffusion models do not. In this paper, we aim to establish the fundamental theory for the probability flow of discrete diffusion models. Specifically, we first prove that the continuous probability flow is the Monge optimal transport map under certain conditions, and also present an equivalent evidence for discrete cases.  In view of these findings, we are then able to define the discrete probability flow in line with the principles of optimal transport. Finally, drawing upon our newly established definitions, we propose a novel sampling method that surpasses previous discrete diffusion models in its ability to generate more certain outcomes. Extensive experiments on the synthetic toy dataset and the CIFAR-10 dataset have validated the effectiveness of our proposed discrete probability flow. Code is released at: https://github.com/PangzeCheung/Discrete-Probability-Flow.

----

## [0] Successor-Predecessor Intrinsic Exploration

**Authors**: *Changmin Yu, Neil Burgess, Maneesh Sahani, Samuel J. Gershman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6f2b968c4ee8ba260cd7077e39590dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6f2b968c4ee8ba260cd7077e39590dd-Abstract-Conference.html)

**Abstract**:

Exploration is essential in reinforcement learning, particularly in environments where external rewards are sparse. Here we focus on exploration with intrinsic rewards, where the agent transiently augments the external rewards with self-generated intrinsic rewards. Although the study of intrinsic rewards has a long history, existing methods focus on composing the intrinsic reward based on measures of future prospects of states, ignoring the information contained in the retrospective structure of transition sequences. Here we argue that the agent can utilise retrospective information to generate explorative behaviour with structure-awareness, facilitating efficient exploration based on global instead of local information. We propose Successor-Predecessor Intrinsic Exploration (SPIE), an exploration algorithm based on a novel intrinsic reward combining prospective and retrospective information. We show that SPIE yields more efficient and ethologically plausible exploratory behaviour in environments with sparse rewards and bottleneck states  than competing methods. We also implement SPIE in deep reinforcement learning agents, and show that the resulting agent achieves stronger empirical performance than existing methods on sparse-reward Atari games.

----

## [0] TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph

**Authors**: *Xueyuan Lin, Haihong E, Chengjin Xu, Gengxian Zhou, Haoran Luo, Tianyi Hu, Fenglong Su, Ningyuan Li, Mingzhi Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e71a42c64851834013e2658b69d7fe93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e71a42c64851834013e2658b69d7fe93-Abstract-Conference.html)

**Abstract**:

Multi-hop logical reasoning over knowledge graph plays a fundamental role in many artificial intelligence tasks.  Recent complex query embedding methods for reasoning focus on static KGs, while temporal knowledge graphs have not been fully explored.  Reasoning over TKGs has two challenges: 1. The query should answer entities or timestamps; 2. The operators should consider both set logic on entity set and temporal logic on timestamp set.To bridge this gap, we introduce the multi-hop logical reasoning problem on TKGs and then propose the first temporal complex query embedding named Temporal Feature-Logic Embedding framework (TFLEX) to answer the temporal complex queries.  Specifically, we utilize fuzzy logic to compute the logic part of the Temporal Feature-Logic embedding, thus naturally modeling all first-order logic operations on the entity set.  In addition, we further extend fuzzy logic on timestamp set to cope with three extra temporal operators (After, Before and Between).Experiments on numerous query patterns demonstrate the effectiveness of our method.

----

## [0] StyleGAN knows Normal, Depth, Albedo, and More

**Authors**: *Anand Bhattad, Daniel McKee, Derek Hoiem, David A. Forsyth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html)

**Abstract**:

Intrinsic images, in the original sense, are image-like maps of scene properties like depth, normal, albedo, or shading. This paper demonstrates that StyleGAN can easily be induced to produce intrinsic images.  The procedure is straightforward. We show that if StyleGAN produces $G({\bf w})$ from latent ${\bf w}$, then for each type of intrinsic image, there is a fixed offset ${\bf d}_c$ so that $G({\bf w}+{\bf d}_c)$ is that type of intrinsic image for $G({\bf w})$. Here ${\bf d}_c$ is {\em independent of ${\bf w}$}.  The StyleGAN we used was pretrained by others, so this property is not some accident of our training regime.  We show that there are image transformations StyleGAN will {\em not} produce in this fashion, so StyleGAN is not a generic image regression engine.  It is conceptually exciting that an image generator should ``know'' and represent intrinsic images. There may also be practical advantages to using a generative model to produce intrinsic images. The intrinsic images obtained from StyleGAN compare well both qualitatively and quantitatively with those obtained by using SOTA image regression techniques; but StyleGAN's intrinsic images are robust to relighting effects, unlike SOTA methods.

----

## [0] Are Vision Transformers More Data Hungry Than Newborn Visual Systems?

**Authors**: *Lalit Pandey, Samantha M. W. Wood, Justin N. Wood*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e75dce944052276caf89c17aca8963d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e75dce944052276caf89c17aca8963d3-Abstract-Conference.html)

**Abstract**:

Vision transformers (ViTs) are top-performing models on many computer vision benchmarks and can accurately predict human behavior on object recognition tasks. However, researchers question the value of using ViTs as models of biological learning because ViTs are thought to be more “data hungry” than brains, with ViTs requiring more training data than brains to reach similar levels of performance. To test this assumption, we directly compared the learning abilities of ViTs and animals, by performing parallel controlled-rearing experiments on ViTs and newborn chicks. We first raised chicks in impoverished visual environments containing a single object, then simulated the training data available in those environments by building virtual animal chambers in a video game engine. We recorded the first-person images acquired by agents moving through the virtual chambers and used those images to train self-supervised ViTs that leverage time as a teaching signal, akin to biological visual systems. When ViTs were trained “through the eyes” of newborn chicks, the ViTs solved the same view-invariant object recognition tasks as the chicks. Thus, ViTs were not more data hungry than newborn chicks: both learned view-invariant object representations in impoverished visual environments. The flexible and generic attention-based learning mechanism in ViTs—combined with the embodied data streams available to newborn animals—appears sufficient to drive the development of animal-like object recognition.

----

## [0] How to Scale Your EMA

**Authors**: *Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html)

**Abstract**:

Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enabled by a scaling rule, for example, in stochastic gradient descent, one should scale the learning rate linearly with the batch size. Another important machine learning tool is the model EMA, a functional copy of a target model, whose parameters move towards those of its target model according to an Exponential Moving Average (EMA) at a rate parameterized by a momentum hyperparameter. This model EMA can improve the robustness and generalization of supervised learning, stabilize pseudo-labeling, and provide a learning signal for Self-Supervised Learning (SSL). Prior works have not considered the optimization of the model EMA when performing scaling, leading to different training dynamics across batch sizes and lower model performance. In this work, we provide a scaling rule for optimization in the presence of a model EMA and demonstrate the rule's validity across a range of architectures, optimizers, and data modalities.  We also show the rule's validity where the model EMA contributes to the optimization of the target model, enabling us to train EMA-based pseudo-labeling and SSL methods at small and large batch sizes. For SSL, we enable training of BYOL up to batch size 24,576 without sacrificing performance, a 6$\times$ wall-clock time reduction under idealized hardware settings.

----

## [0] Unsupervised Graph Neural Architecture Search with Disentangled Self-Supervision

**Authors**: *Zeyang Zhang, Xin Wang, Ziwei Zhang, Guangyao Shen, Shiqi Shen, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e78399fc43dbb2d87b7e1e6906ce5baf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e78399fc43dbb2d87b7e1e6906ce5baf-Abstract-Conference.html)

**Abstract**:

The existing graph neural architecture search (GNAS) methods heavily rely on supervised labels during the search process, failing to handle ubiquitous scenarios where supervisions are not available. In this paper, we study the problem of unsupervised graph neural architecture search, which remains unexplored in the literature. The key problem is to discover the latent graph factors that drive the formation of graph data as well as the underlying relations between the factors and the optimal neural architectures. Handling this problem is challenging given that the latent graph factors together with architectures are highly entangled due to the nature of the graph and the complexity of the neural architecture search process. To address the challenge, we propose a novel Disentangled Self-supervised Graph Neural Architecture Search (DSGAS) model, which is able to discover the optimal architectures capturing various latent graph factors in a self-supervised fashion based on unlabeled graph data. Specifically, we first design a disentangled graph super-network capable of incorporating multiple architectures with factor-wise disentanglement, which are optimized simultaneously. Then, we estimate the performance of architectures under different factors by our proposed self-supervised training with joint architecture-graph disentanglement. Finally, we propose a contrastive search with architecture augmentations to discover architectures with factor-specific expertise. Extensive experiments on 11 real-world datasets demonstrate that the proposed model is able to achieve state-of-the-art performance against several baseline methods in an unsupervised manner.

----

## [0] Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots

**Authors**: *Ruixiang (Ryan) Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html)

**Abstract**:

In the field of natural language processing, the prevalent approach involves fine-tuning pretrained language models (PLMs) using local samples. Recent research has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries can embed malicious prediction behaviors by manipulating a few training samples. In this study, our objective is to develop a backdoor-resistant tuning procedure that yields a backdoor-free model, no matter whether the fine-tuning dataset contains poisoned samples. To this end, we propose and integrate an \emph{honeypot module} into the original PLM, specifically designed to absorb backdoor information exclusively. Our design is motivated by the observation that lower-layer representations in PLMs carry sufficient backdoor features while carrying minimal information about the original tasks. Consequently, we can impose penalties on the information acquired by the honeypot module to inhibit backdoor creation during the fine-tuning process of the stem network. Comprehensive experiments conducted on benchmark datasets substantiate the effectiveness and robustness of our defensive strategy. Notably, these results indicate a substantial reduction in the attack success rate ranging from 10\% to 40\% when compared to prior state-of-the-art methods.

----

## [0] Learning Large-Scale MTP2 Gaussian Graphical Models via Bridge-Block Decomposition

**Authors**: *Xiwen Wang, Jiaxi Ying, Daniel P. Palomar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7e506bc5a94768243083216fe51d98b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7e506bc5a94768243083216fe51d98b-Abstract-Conference.html)

**Abstract**:

This paper studies the problem of learning the large-scale Gaussian graphical models that are multivariate totally positive of order two ($\text{MTP}_2$). By introducing the concept of bridge, which commonly exists in large-scale sparse graphs, we show that the entire problem can be equivalently optimized through (1) several smaller-scaled sub-problems induced by a \emph{bridge-block decomposition} on the thresholded sample covariance graph and (2) a set of explicit solutions on entries corresponding to  \emph{bridges}. From practical aspect, this simple and provable discipline can be applied to break down a large problem into small tractable ones, leading to enormous reduction on the computational complexity and substantial improvements for all existing algorithms.  The synthetic and real-world experiments demonstrate that our proposed method presents a significant speed-up compared to the state-of-the-art benchmarks.

----



[Go to the previous page](NIPS-2023-list3100.md)

[Go to the next page](NIPS-2023-list3102.md)

[Go to the catalog section](README.md)