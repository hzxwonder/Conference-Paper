## [1200] Multi-Agent Reinforcement Learning is a Sequence Modeling Problem

        **Authors**: *Muning Wen, Jakub Grudzien Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html)

        **Abstract**:

        Large sequence models (SM) such as GPT series and BERT have displayed outstanding performance and generalization capabilities in natural language process, vision and recently reinforcement learning. A natural follow-up question is how to abstract multi-agent decision making also as an sequence modeling problem and benefit from the prosperous development of the SMs. In this paper, we introduce a novel architecture named Multi-Agent Transformer (MAT) that effectively casts cooperative multi-agent reinforcement learning (MARL) into SM problems wherein the objective is to map agents'  observation sequences to agents' optimal action sequences. Our goal is to build the bridge between MARL and SMs so that the modeling power of modern sequence models can be unleashed for MARL. Central to our MAT is an encoder-decoder architecture which leverages the multi-agent advantage decomposition theorem to transform the joint policy search problem into a sequential decision making process; this renders only linear time complexity for multi-agent problems and, most importantly, endows MAT with monotonic performance improvement guarantee. Unlike prior arts such as Decision Transformer fit only pre-collected offline data, MAT is trained by online trial and error from the environment in an on-policy fashion. To validate MAT, we conduct extensive experiments on StarCraftII, Multi-Agent MuJoCo, Dexterous Hands Manipulation, and Google Research Football benchmarks. Results demonstrate that MAT achieves superior performance and data efficiency compared to strong baselines including MAPPO and HAPPO. Furthermore, we demonstrate that MAT is an excellent few-short learner on unseen tasks regardless of changes in the number of agents.See our project page at https://sites.google.com/view/multi-agent-transformer.

        ----

        ## [1201] Non-stationary Bandits with Knapsacks

        **Authors**: *Shang Liu, Jiashuo Jiang, Xiaocheng Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69469da823348084ca8933368ecbf676-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69469da823348084ca8933368ecbf676-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the problem of bandits with knapsacks (BwK) in a non-stationary environment. The BwK problem generalizes the multi-arm bandit (MAB) problem to model the resource consumption associated with playing each arm. At each time, the decision maker/player chooses to play an arm, and s/he will receive a reward and consume certain amount of resource from each of the multiple resource types. The objective is to maximize the cumulative reward over a finite horizon subject to some knapsack constraints on the resources. Existing works study the BwK problem under either a stochastic or adversarial environment. Our paper considers a non-stationary environment which continuously interpolates between these two extremes. We first show that the traditional notion of variation budget is insufficient to characterize the non-stationarity of the BwK problem for a sublinear regret due to the presence of the constraints, and then we propose a new notion of global non-stationarity measure. We employ both non-stationarity measures to derive upper and lower bounds for the problem. Our results are based on a primal-dual analysis of the underlying linear programs and highlight the interplay between the constraints and the non-stationarity. Finally, we also extend the non-stationarity measure to the problem of online convex optimization with constraints and obtain new regret bounds accordingly.

        ----

        ## [1202] Fast Bayesian Inference with Batch Bayesian Quadrature via Kernel Recombination

        **Authors**: *Masaki Adachi, Satoshi Hayakawa, Martin Jørgensen, Harald Oberhauser, Michael A. Osborne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/697200c9d1710c2799720b660abd11bb-Abstract-Conference.html)

        **Abstract**:

        Calculation of Bayesian posteriors and model evidences typically requires numerical integration. Bayesian quadrature (BQ), a surrogate-model-based approach to numerical integration, is capable of superb sample efficiency, but its lack of parallelisation has hindered its practical applications. In this work, we propose a parallelised (batch) BQ method, employing techniques from kernel quadrature, that possesses an empirically exponential convergence rate.Additionally, just as with Nested Sampling, our method permits simultaneous inference of both posteriors and model evidence.Samples from our BQ surrogate model are re-selected to give a sparse set of samples, via a kernel recombination algorithm, requiring negligible additional time to increase the batch size.Empirically, we find that our approach significantly outperforms the sampling efficiency of both state-of-the-art BQ techniques and Nested Sampling in various real-world datasets, including lithium-ion battery analytics.

        ----

        ## [1203] Are Two Heads the Same as One? Identifying Disparate Treatment in Fair Neural Networks

        **Authors**: *Michael Lohaus, Matthäus Kleindessner, Krishnaram Kenthapadi, Francesco Locatello, Chris Russell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/698c05933e5f7fde98e567a669d2c752-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/698c05933e5f7fde98e567a669d2c752-Abstract-Conference.html)

        **Abstract**:

        We show that deep networks trained to satisfy demographic parity often do so through a form of race or gender awareness, and that the more we force a network to be fair, the more accurately we can recover race or gender from the internal state of the network. Based on this observation, we investigate an alternative fairness approach: we add a second classification head to the network to explicitly predict the protected attribute (such as race or gender) alongside the original task. After training the two-headed network, we enforce demographic parity by merging the two heads, creating a network with the same architecture as the original network. We establish a close relationship between existing approaches and our approach by showing (1) that the decisions of a fair classifier are well-approximated by our approach, and (2) that an unfair and optimally accurate classifier can be recovered from a fair classifier and our second head  predicting the protected attribute. We use our explicit formulation to argue that the existing fairness approaches, just as ours, demonstrate disparate treatment and that they are likely to be unlawful in a wide range of scenarios under US law.

        ----

        ## [1204] Provable Generalization of Overparameterized Meta-learning Trained with SGD

        **Authors**: *Yu Huang, Yingbin Liang, Longbo Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69a076724e7228aba0272305bb98727e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69a076724e7228aba0272305bb98727e-Abstract-Conference.html)

        **Abstract**:

        Despite the empirical success of deep meta-learning, theoretical understanding of overparameterized meta-learning is still limited. This paper studies the generalization of a widely used meta-learning approach, Model-Agnostic Meta-Learning (MAML), which aims to find a good initialization for fast adaptation to new tasks. Under a mixed linear regression model, we analyze the generalization properties of MAML trained with SGD in the overparameterized regime. We provide both upper and lower bounds for the excess risk of MAML, which captures how SGD dynamics affect these generalization bounds. With such sharp characterizations, we further explore how various learning parameters impact the generalization capability of  overparameterized MAML, including explicitly identifying typical data and task distributions that can achieve diminishing generalization error with overparameterization, and characterizing the impact of adaptation learning rate on both excess risk and the early stopping time. Our theoretical findings are further validated by experiments.

        ----

        ## [1205] When Do Flat Minima Optimizers Work?

        **Authors**: *Jean Kaddour, Linqing Liu, Ricardo Silva, Matt J. Kusner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69b5534586d6c035a96b49c86dbeece8-Abstract-Conference.html)

        **Abstract**:

        Recently, flat-minima optimizers, which seek to find parameters in low-loss neighborhoods, have been shown to improve a neural network's generalization performance over stochastic and adaptive gradient-based optimizers. Two methods have received significant attention due to their scalability: 1. Stochastic Weight Averaging (SWA), and 2. Sharpness-Aware Minimization (SAM). However, there has been limited investigation into their properties and no systematic benchmarking of them across different domains. We fill this gap here by comparing the loss surfaces of the models trained with each method and through broad benchmarking across computer vision, natural language processing, and graph representation learning tasks. We discover several surprising findings from these results, which we hope will help researchers further improve deep learning optimizers, and practitioners identify the right optimizer for their problem.

        ----

        ## [1206] Fast Instrument Learning with Faster Rates

        **Authors**: *Ziyu Wang, Yuhao Zhou, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69b6de0de842bfedbc40ed6e162b4233-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69b6de0de842bfedbc40ed6e162b4233-Abstract-Conference.html)

        **Abstract**:

        We investigate nonlinear instrumental variable (IV) regression given high-dimensional instruments. We propose a simple algorithm which combines kernelized IV methods and an arbitrary, adaptive regression algorithm, accessed as a black box. Our algorithm enjoys faster-rate convergence and adapts to the dimensionality of informative latent features, while avoiding an expensive minimax optimization procedure, which has been necessary to establish similar guarantees. It further brings the benefit of flexible machine learning models to quasi-Bayesian uncertainty quantification, likelihood-based model selection, and model averaging. Simulation studies demonstrate the competitive performance of our method.

        ----

        ## [1207] Sparse Gaussian Process Hyperparameters: Optimize or Integrate?

        **Authors**: *Vidhi Lalchand, Wessel P. Bruinsma, David R. Burt, Carl Edward Rasmussen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69c49f75ca31620f1f0d38093d9f3d9b-Abstract-Conference.html)

        **Abstract**:

        The kernel function and its hyperparameters are the central model selection choice in a Gaussian process (Rasmussen and Williams, 2006).Typically, the hyperparameters of the kernel are chosen by maximising the marginal likelihood, an approach known as Type-II maximum likelihood (ML-II). However, ML-II does not account for hyperparameter uncertainty, and it is well-known that this can lead to severely biased estimates and an underestimation of predictive uncertainty. While there are several works which employ fully Bayesian characterisation of GPs, relatively few propose such approaches for the sparse GPs paradigm. In this work we propose an algorithm for sparse Gaussian process regression which leverages MCMC to sample from the hyperparameter posterior within the variational inducing point framework of (Titsias, 2009). This work is closely related to (Hensman et al, 2015b) but side-steps the need to sample the inducing points, thereby significantly improving sampling efficiency in the Gaussian likelihood case. We compare this scheme against natural baselines in literature along with stochastic variational GPs (SVGPs) along with an extensive computational analysis.

        ----

        ## [1208] HierSpeech: Bridging the Gap between Text and Speech by Hierarchical Variational Inference using Self-supervised Representations for Speech Synthesis

        **Authors**: *Sang-Hoon Lee, Seung-Bin Kim, Ji-Hyun Lee, Eunwoo Song, Min-Jae Hwang, Seong-Whan Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69c754f571806bf15add18556ff39b4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69c754f571806bf15add18556ff39b4f-Abstract-Conference.html)

        **Abstract**:

        This paper presents HierSpeech, a high-quality end-to-end text-to-speech (TTS) system based on a hierarchical conditional variational autoencoder (VAE) utilizing self-supervised speech representations. Recently, single-stage TTS systems, which directly generate raw speech waveform from text, have been getting interest thanks to their ability in generating high-quality audio within a fully end-to-end training pipeline. However, there is still a room for improvement in the conventional TTS systems. Since it is challenging to infer both the linguistic and acoustic attributes from the text directly, missing the details of attributes, specifically linguistic information, is inevitable, which results in mispronunciation and over-smoothing problem in their synthetic speech. To address the aforementioned problem, we leverage self-supervised speech representations as additional linguistic representations to bridge an information gap between text and speech. Then, the hierarchical conditional VAE is adopted to connect these representations and to learn each attribute hierarchically by improving the linguistic capability in latent representations. Compared with the state-of-the-art TTS system, HierSpeech achieves +0.303 comparative mean opinion score, and reduces the phoneme error rate of synthesized speech from 9.16% to 5.78% on the VCTK dataset. Furthermore, we extend our model to HierSpeech-U, an untranscribed text-to-speech system. Specifically, HierSpeech-U can adapt to a novel speaker by utilizing self-supervised speech representations without text transcripts. The experimental results reveal that our method outperforms publicly available TTS models, and show the effectiveness of speaker adaptation with untranscribed speech.

        ----

        ## [1209] Sharing Knowledge for Meta-learning with Feature Descriptions

        **Authors**: *Tomoharu Iwata, Atsutoshi Kumagai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69ce18ad9f53f28e8e7ac1649ae02337-Abstract-Conference.html)

        **Abstract**:

        Language is an important tool for humans to share knowledge. We propose a meta-learning method that shares knowledge across supervised learning tasks using feature descriptions written in natural language, which have not been used in the existing meta-learning methods. The proposed method improves the predictive performance on unseen tasks with a limited number of labeled data by meta-learning from various tasks. With the feature descriptions, we can find relationships across tasks even when their feature spaces are different. The feature descriptions are encoded using a language model pretrained with a large corpus, which enables us to incorporate human knowledge stored in the corpus into meta-learning. In our experiments, we demonstrate that the proposed method achieves better predictive performance than the existing meta-learning methods using a wide variety of real-world datasets provided by the statistical office of the EU and Japan.

        ----

        ## [1210] A Variant of Anderson Mixing with Minimal Memory Size

        **Authors**: *Fuchao Wei, Chenglong Bao, Yang Liu, Guangwen Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69deee78acc4af72e6e0fca82780b3a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69deee78acc4af72e6e0fca82780b3a4-Abstract-Conference.html)

        **Abstract**:

        Anderson mixing (AM) is a useful method that can accelerate fixed-point iterations by exploring the information from historical iterations. Despite its numerical success in various applications, the memory requirement in AM remains a bottleneck when solving large-scale optimization problems in a resource-limited machine. To address this problem, we propose a novel variant of AM method, called Min-AM, by storing only one vector pair, that is the minimal memory size requirement in AM. Our method forms a symmetric approximation to the inverse Hessian matrix and is proved to be equivalent to the full-memory Type-I AM for solving strongly convex quadratic optimization. Moreover, for general nonlinear optimization problems, we establish the convergence properties of Min-AM under reasonable assumptions and show that the mixing parameters can be adaptively chosen by estimating the eigenvalues of the Hessian. Finally, we extend Min-AM to solve stochastic programming problems. Experimental results on logistic regression and network training problems validate the effectiveness of the proposed Min-AM.

        ----

        ## [1211] AdaptFormer: Adapting Vision Transformers for Scalable Visual Recognition

        **Authors**: *Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, Ping Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69e2f49ab0837b71b0e0cb7c555990f8-Abstract-Conference.html)

        **Abstract**:

        Pretraining Vision Transformers (ViTs) has achieved great success in visual recognition. A following scenario is to adapt a ViT to various image and video recognition tasks. The adaptation is challenging because of heavy computation and memory storage. Each model needs an independent and complete finetuning process to adapt to different tasks, which limits its transferability to different visual domains.To address this challenge, we propose an effective adaptation approach for Transformer, namely AdaptFormer, which can adapt the pre-trained ViTs into many different image and video tasks efficiently.It possesses several benefits more appealing than prior arts.Firstly, AdaptFormer introduces lightweight modules that only add less than 2% extra parameters to a ViT, while it is able to increase the ViT's transferability without updating its original pre-trained parameters, significantly outperforming the existing 100\% fully fine-tuned models on action recognition benchmarks.Secondly, it can be plug-and-play in different Transformers and scalable to many visual tasks.Thirdly, extensive experiments on five image and video datasets show that AdaptFormer largely improves ViTs in the target domains. For example, when updating just 1.5% extra parameters, it achieves about 10% and 19% relative improvement compared to the fully fine-tuned models on Something-Something~v2 and HMDB51, respectively. Code is available at https://github.com/ShoufaChen/AdaptFormer.

        ----

        ## [1212] Symmetry Teleportation for Accelerated Optimization

        **Authors**: *Bo Zhao, Nima Dehmamy, Robin Walters, Rose Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/69f7750aa28f75fddf101da038f8b529-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/69f7750aa28f75fddf101da038f8b529-Abstract-Conference.html)

        **Abstract**:

        Existing gradient-based optimization methods update parameters locally, in a direction that minimizes the loss function. We study a different approach, symmetry teleportation, that allows parameters to travel a large distance on the loss level set, in order to improve the convergence speed in subsequent steps. Teleportation exploits symmetries in the loss landscape of optimization problems. We derive loss-invariant group actions for test functions in optimization and multi-layer neural networks, and prove a necessary condition for teleportation to improve convergence rate. We also show that our algorithm is closely related to second order methods. Experimentally, we show that teleportation improves the convergence speed of gradient descent and AdaGrad for several optimization problems including test functions, multi-layer regressions, and MNIST classification.

        ----

        ## [1213] Wasserstein Logistic Regression with Mixed Features

        **Authors**: *Aras Selvi, Mohammad Reza Belbasi, Martin Haugh, Wolfram Wiesemann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a13cffb5ec4128324f64a186785215b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a13cffb5ec4128324f64a186785215b-Abstract-Conference.html)

        **Abstract**:

        Recent work has leveraged the popular distributionally robust optimization paradigm to combat overfitting in classical logistic regression. While the resulting classification scheme displays a promising performance in numerical experiments, it is inherently limited to numerical features. In this paper, we show that distributionally robust logistic regression with mixed (\emph{i.e.}, numerical and categorical) features, despite amounting to an optimization problem of exponential size, admits a polynomial-time solution scheme. We subsequently develop a practically efficient cutting plane approach that solves the problem as a sequence of polynomial-time solvable exponential conic programs. Our method retains many of the desirable theoretical features of previous works, but---in contrast to the literature---it does not admit an equivalent representation as a regularized logistic regression, that is, it represents a genuinely novel variant of the logistic regression problem. We show that our method outperforms both the unregularized and the regularized logistic regression on categorical as well as mixed-feature benchmark instances.

        ----

        ## [1214] TaiSu: A 166M Large-scale High-Quality Dataset for Chinese Vision-Language Pre-training

        **Authors**: *Yulong Liu, Guibo Zhu, Bin Zhu, Qi Song, Guojing Ge, Haoran Chen, Guanhui Qiao, Ru Peng, Lingxiang Wu, Jinqiao Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Vision-Language Pre-training (VLP) has been shown to be an efficient method to improve the performance of models on different vision-and-language downstream tasks. Substantial studies have shown that neural networks may be able to learn some general rules about language and visual concepts from a large-scale weakly labeled image-text dataset. However, most of the public cross-modal datasets that contain more than 100M image-text pairs are in English; there is a lack of available large-scale and high-quality Chinese VLP datasets. In this work, we propose a new framework for automatic dataset acquisition and cleaning with which we construct a new large-scale and high-quality cross-modal dataset named as TaiSu, containing 166 million images and 219 million Chinese captions. Compared with the recently released Wukong dataset, our dataset is achieved with much stricter restrictions on the semantic correlation of image-text pairs. We also propose to combine texts collected from the web with texts generated by a pre-trained image-captioning model. To the best of our knowledge, TaiSu is currently the largest publicly accessible Chinese cross-modal dataset. Furthermore, we test our dataset on several vision-language downstream tasks. TaiSu outperforms BriVL by a large margin on the zero-shot image-text retrieval task and zero-shot image classification task. TaiSu also shows better performance than Wukong on the image-retrieval task without using image augmentation for training. Results demonstrate that TaiSu can serve as a promising VLP dataset, both for understanding and generative tasks. More information can be referred to https://github.com/ksOAn6g5/TaiSu.

        ----

        ## [1215] Robust Bayesian Regression via Hard Thresholding

        **Authors**: *Zheyi Fan, Zhaohui Li, Qingpei Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a4262293ca91c5af2dfab24bd343b43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a4262293ca91c5af2dfab24bd343b43-Abstract-Conference.html)

        **Abstract**:

        By combining robust regression and prior information, we develop an effective robust regression method that can resist adaptive adversarial attacks. Due to the widespread existence of noise and data corruption, it is necessary to recover the true regression parameters when a certain proportion of the response variables have been corrupted. Methods to overcome this problem often involve robust least-squares regression. However, few methods achieve good performance when dealing with severe adaptive adversarial attacks. Based on the combination of prior information and robust regression via hard thresholding, this paper proposes an algorithm that improves the breakdown point when facing adaptive adversarial attacks. Furthermore, to improve the robustness and reduce the estimation error caused by the inclusion of a prior, the idea of Bayesian reweighting is used to construct a more robust algorithm. We prove the theoretical convergence of proposed algorithms under mild conditions. Extensive experiments show that, under different dataset attacks, our algorithms achieve state-of-the-art results compared with other benchmark algorithms, demonstrating the robustness of the proposed approach.

        ----

        ## [1216] Trajectory Inference via Mean-field Langevin in Path Space

        **Authors**: *Lénaïc Chizat, Stephen Zhang, Matthieu Heitz, Geoffrey Schiebinger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a5181cfe76f67b37a7e1bb19837abdf-Abstract-Conference.html)

        **Abstract**:

        Trajectory inference aims at recovering the dynamics of a population from snapshots of its temporal marginals. To solve this task, a min-entropy estimator relative to the Wiener measure in path space was introduced in [Lavenant et al., 2021], and shown to consistently recover the dynamics of a large class of drift-diffusion processes from the solution of an infinite dimensional convex optimization problem. In this paper, we introduce a grid-free algorithm to compute this estimator. Our method consists in a family of point clouds (one per snapshot) coupled via Schrödinger bridges which evolve with noisy gradient descent. We study the mean-field limit of the dynamics and prove its global convergence to the desired estimator. Overall, this leads to an inference method with end-to-end theoretical guarantees that solves an interpretable model for trajectory inference. We also present how to adapt the method to deal with mass variations, a useful extension when dealing with single cell RNA-sequencing data where cells can branch and die.

        ----

        ## [1217] SwinTrack: A Simple and Strong Baseline for Transformer Tracking

        **Authors**: *Liting Lin, Heng Fan, Zhipeng Zhang, Yong Xu, Haibin Ling*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a5c23219f401f3efd322579002dbb80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a5c23219f401f3efd322579002dbb80-Abstract-Conference.html)

        **Abstract**:

        Recently Transformer has been largely explored in tracking and shown state-of-the-art (SOTA) performance. However, existing efforts mainly focus on fusing and enhancing features generated by convolutional neural networks (CNNs). The potential of Transformer in representation learning remains under-explored. In this paper, we aim to further unleash the power of Transformer by proposing a simple yet efficient fully-attentional tracker, dubbed SwinTrack, within classic Siamese framework. In particular, both representation learning and feature fusion in SwinTrack leverage the Transformer architecture, enabling better feature interactions for tracking than pure CNN or hybrid CNN-Transformer frameworks. Besides, to further enhance robustness, we present a novel motion token that embeds historical target trajectory to improve tracking by providing temporal context. Our motion token is lightweight with negligible computation but brings clear gains. In our thorough experiments, SwinTrack exceeds existing approaches on multiple benchmarks. Particularly, on the challenging LaSOT, SwinTrack sets a new record with 0.713 SUC score. It also achieves SOTA results on other benchmarks. We expect SwinTrack to serve as a solid baseline for Transformer tracking and facilitate future research. Our codes and results are released at https://github.com/LitingLin/SwinTrack.

        ----

        ## [1218] Unknown-Aware Domain Adversarial Learning for Open-Set Domain Adaptation

        **Authors**: *JoonHo Jang, Byeonghu Na, DongHyeok Shin, Mingi Ji, Kyungwoo Song, Il-Chul Moon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6a934325ec64639ba83b492b9c317085-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6a934325ec64639ba83b492b9c317085-Abstract-Conference.html)

        **Abstract**:

        Open-Set Domain Adaptation (OSDA) assumes that a target domain contains unknown classes, which are not discovered in a source domain. Existing domain adversarial learning methods are not suitable for OSDA because distribution matching with $\textit{unknown}$ classes leads to negative transfer. Previous OSDA methods have focused on matching the source and the target distribution by only utilizing $\textit{known}$ classes. However, this $\textit{known}$-only matching may fail to learn the target-$\textit{unknown}$ feature space. Therefore, we propose Unknown-Aware Domain Adversarial Learning (UADAL), which $\textit{aligns}$ the source and the target-$\textit{known}$ distribution while simultaneously $\textit{segregating}$ the target-$\textit{unknown}$ distribution in the feature alignment procedure. We provide theoretical analyses on the optimized state of the proposed $\textit{unknown-aware}$ feature alignment, so we can guarantee both $\textit{alignment}$ and $\textit{segregation}$ theoretically. Empirically, we evaluate UADAL on the benchmark datasets, which shows that UADAL outperforms other methods with better feature alignments by reporting state-of-the-art performances.

        ----

        ## [1219] Learning Chaotic Dynamics in Dissipative Systems

        **Authors**: *Zongyi Li, Miguel Liu-Schiaffini, Nikola B. Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, Anima Anandkumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ad68277e27b42c60ac228c9859fc1a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ad68277e27b42c60ac228c9859fc1a2-Abstract-Conference.html)

        **Abstract**:

        Chaotic systems are notoriously challenging to predict because of their sensitivity to perturbations and errors due to time stepping. Despite this unpredictable behavior, for many dissipative systems the statistics of the long term trajectories are governed by an invariant measure supported on a set, known as the global attractor; for many problems this set is finite dimensional, even if the state space is infinite dimensional. For Markovian systems, the statistical properties of long-term trajectories are uniquely determined by the solution operator that maps the evolution of the system over arbitrary positive time increments. In this work, we propose a machine learning framework to learn the underlying solution operator for dissipative chaotic systems, showing that the resulting learned operator accurately captures short-time trajectories and long-time statistical behavior. Using this framework, we are able to predict various statistics of the invariant measure for the turbulent Kolmogorov Flow dynamics with Reynolds numbers up to $5000$.

        ----

        ## [1220] Poisson Flow Generative Models

        **Authors**: *Yilun Xu, Ziming Liu, Max Tegmark, Tommi S. Jaakkola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ad68a54eaa8f9bf6ac698b02ec05048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ad68a54eaa8f9bf6ac698b02ec05048-Abstract-Conference.html)

        **Abstract**:

        We propose a new "Poisson flow" generative model~(PFGM) that maps a uniform distribution on a high-dimensional hemisphere into any data distribution. We interpret the data points as electrical charges on the $z=0$ hyperplane in a space augmented with an additional dimension $z$, generating a high-dimensional electric field (the gradient of the solution to Poisson equation). We prove that if these charges flow upward along electric field lines, their initial distribution in the $z=0$ plane transforms into a distribution on the hemisphere of radius $r$ that becomes uniform in the $r \to\infty$ limit. To learn the bijective transformation, we estimate the normalized field in the augmented space. For sampling, we devise a backward ODE that is anchored by the physically meaningful additional dimension: the samples hit the (unaugmented) data manifold when the $z$ reaches zero. Experimentally, PFGM achieves current state-of-the-art performance among the normalizing flow models on CIFAR-10, with an Inception score of $9.68$ and a FID score of $2.35$. It also performs on par with the state-of-the-art SDE approaches while offering $10\times $ to $20 \times$ acceleration on image generation tasks. Additionally, PFGM appears more tolerant of estimation errors on a weaker network architecture and robust to the step size in the Euler method. The code is available at https://github.com/Newbeeer/poisson_flow .

        ----

        ## [1221] Bounding and Approximating Intersectional Fairness through Marginal Fairness

        **Authors**: *Mathieu Molina, Patrick Loiseau*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ae7df1f40f5faeda474b36b61197822-Abstract-Conference.html)

        **Abstract**:

        Discrimination in machine learning often arises along multiple dimensions (a.k.a. protected attributes); it is then desirable to ensure \emph{intersectional fairness}---i.e., that no subgroup is discriminated against. It is known that ensuring \emph{marginal fairness} for every dimension independently is not sufficient in general. Due to the exponential number of subgroups, however, directly measuring intersectional fairness from data is impossible. In this paper, our primary goal is to understand in detail the relationship between marginal and intersectional fairness through statistical analysis. We first identify a set of sufficient conditions under which an exact relationship can be obtained. Then, we prove bounds (easily computable through marginal fairness and other meaningful statistical quantities) in high-probability on intersectional fairness in the general case. Beyond their descriptive value, we show that these theoretical bounds can be leveraged to derive a heuristic improving the approximation and bounds of intersectional fairness by choosing, in a relevant manner, protected attributes for which we describe intersectional subgroups. Finally, we test the performance of our approximations and bounds on real and synthetic data-sets.

        ----

        ## [1222] Semi-infinitely Constrained Markov Decision Processes

        **Authors**: *Liangyu Zhang, Yang Peng, Wenhao Yang, Zhihua Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6aef8bffb372096ee73d98da30119f89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6aef8bffb372096ee73d98da30119f89-Abstract-Conference.html)

        **Abstract**:

        We propose a generalization of constrained Markov decision processes (CMDPs) that we call the \emph{semi-infinitely constrained Markov decision process} (SICMDP).Particularly, in a SICMDP model, we impose a continuum of constraints instead of a finite number of constraints as in the case of ordinary CMDPs.We also devise a reinforcement learning algorithm for SICMDPs that we call SI-CRL.We first transform the reinforcement learning problem into a linear semi-infinitely programming (LSIP) problem and then use the dual exchange method in the LSIP literature to solve it.To the best of our knowledge, we are the first to apply tools from semi-infinitely programming (SIP) to solve reinforcement learning problems.We present theoretical analysis for SI-CRL, identifying its sample complexity and iteration complexity.We also conduct extensive numerical examples to illustrate the SICMDP model and validate the SI-CRL algorithm.

        ----

        ## [1223] Latent Planning via Expansive Tree Search

        **Authors**: *Robert Gieselmann, Florian T. Pokorny*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6af779991368999ab3da0d366c208fba-Abstract-Conference.html)

        **Abstract**:

        Planning enables autonomous agents to solve complex decision-making problems by evaluating predictions of the future. However, classical planning algorithms often become infeasible in real-world settings where state spaces are high-dimensional and transition dynamics unknown. The idea behind latent planning is to simplify the decision-making task by mapping it to a lower-dimensional embedding space. Common latent planning strategies are based on trajectory optimization techniques such as shooting or collocation, which are prone to failure in long-horizon and highly non-convex settings. In this work, we study long-horizon goal-reaching scenarios from visual inputs and formulate latent planning as an explorative tree search. Inspired by classical sampling-based motion planning algorithms, we design a method which iteratively grows and optimizes a tree representation of visited areas of the latent space. To encourage fast exploration, the sampling of new states is biased towards sparsely represented regions within the estimated data support. Our method, called Expansive Latent Space Trees (ELAST), relies on self-supervised training via contrastive learning to obtain (a) a latent state representation and (b) a latent transition density model. We embed ELAST into a model-predictive control scheme and demonstrate significant performance improvements compared to existing baselines given challenging visual control tasks in simulation, including the navigation for a deformable object.

        ----

        ## [1224] Invertible Monotone Operators for Normalizing Flows

        **Authors**: *Byeongkeun Ahn, Chiyoon Kim, Youngjoon Hong, Hyunwoo J. Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6afae862e1abc2ea396c5e842be54a52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6afae862e1abc2ea396c5e842be54a52-Abstract-Conference.html)

        **Abstract**:

        Normalizing flows model probability distributions by learning invertible transformations that transfer a simple distribution into complex distributions. Since the architecture of ResNet-based normalizing flows is more flexible than that of coupling-based models, ResNet-based normalizing flows have been widely studied in recent years. Despite their architectural flexibility, it is well-known that the current ResNet-based models suffer from constrained Lipschitz constants. In this paper, we propose the monotone formulation to overcome the issue of the Lipschitz constants using monotone operators and provide an in-depth theoretical analysis. Furthermore, we construct an activation function called Concatenated Pila (CPila) to improve gradient flow. The resulting model, Monotone Flows, exhibits an excellent performance on multiple density estimation benchmarks (MNIST, CIFAR-10, ImageNet32, ImageNet64). Code is available at https://github.com/mlvlab/MonotoneFlows.

        ----

        ## [1225] Dynamic Sparse Network for Time Series Classification: Learning What to "See"

        **Authors**: *Qiao Xiao, Boqian Wu, Yu Zhang, Shiwei Liu, Mykola Pechenizkiy, Elena Mocanu, Decebal Constantin Mocanu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html)

        **Abstract**:

        The receptive field (RF), which determines the region of time series to be “seen” and used, is critical to improve the performance for time series classification (TSC). However, the variation of signal scales across and within time series data, makes it challenging to decide on proper RF sizes for TSC. In this paper, we propose a dynamic sparse network (DSN) with sparse connections for TSC, which can learn to cover various RF without cumbersome hyper-parameters tuning. The kernels in each sparse layer are sparse and can be explored under the constraint regions by dynamic sparse training, which makes it possible to reduce the resource cost. The experimental results show that the proposed DSN model can achieve state-of-art performance on both univariate and multivariate TSC datasets with less than 50% computational cost compared with recent baseline methods, opening the path towards more accurate resource-aware methods for time series analyses. Our code is publicly available at: https://github.com/QiaoXiao7282/DSN.

        ----

        ## [1226] Learning to Sample and Aggregate: Few-shot Reasoning over Temporal Knowledge Graphs

        **Authors**: *Ruijie Wang, Zheng Li, Dachun Sun, Shengzhong Liu, Jinning Li, Bing Yin, Tarek F. Abdelzaher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6b295b08549c0441914e391651423477-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6b295b08549c0441914e391651423477-Abstract-Conference.html)

        **Abstract**:

        In this paper, we investigate a realistic but underexplored problem, called few-shot temporal knowledge graph reasoning, that aims to predict future facts for newly emerging entities based on extremely limited observations in evolving graphs. It offers practical value in applications that need to derive instant new knowledge about new entities in temporal knowledge graphs (TKGs) with minimal supervision. The challenges mainly come from the few-shot and time shift properties of new entities. First, the limited observations associated with them are insufficient for training a model from scratch. Second, the potentially dynamic distributions from the initially observable facts to the future facts ask for explicitly modeling the evolving characteristics of new entities. We correspondingly propose a novel Meta Temporal Knowledge Graph Reasoning (MetaTKGR) framework. Unlike prior work that relies on rigid neighborhood aggregation schemes to enhance low-data entity representation, MetaTKGR dynamically adjusts the strategies of sampling and aggregating neighbors from recent facts for new entities, through temporally supervised signals on future facts as instant feedback. Besides, such a meta temporal reasoning procedure goes beyond existing meta-learning paradigms on static knowledge graphs that fail to handle temporal adaptation with large entity variance. We further provide a theoretical analysis and propose a temporal adaptation regularizer to stabilize the meta temporal reasoning over time. Empirically, extensive experiments on three real-world TKGs demonstrate the superiority of MetaTKGR over eight state-of-the-art baselines by a large margin.

        ----

        ## [1227] Evaluating Robustness to Dataset Shift via Parametric Robustness Sets

        **Authors**: *Nikolaj Thams, Michael Oberst, David A. Sontag*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6b7f9d9c1217a748391800871ff7d17d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6b7f9d9c1217a748391800871ff7d17d-Abstract-Conference.html)

        **Abstract**:

        We give a method for proactively identifying small, plausible shifts in distribution which lead to large differences in model performance.  These shifts are defined via parametric changes in the causal mechanisms of observed variables, where constraints on parameters yield a "robustness set" of plausible distributions and a corresponding worst-case loss over the set. While the loss under an individual parametric shift can be estimated via reweighting techniques such as importance sampling, the resulting worst-case optimization problem is non-convex, and the estimate may suffer from large variance. For small shifts, however, we can construct a local second-order approximation to the loss under shift and cast the problem of finding a worst-case shift as a particular non-convex quadratic optimization problem, for which efficient algorithms are available.  We demonstrate that this second-order approximation can be estimated directly for shifts in conditional exponential family models, and we bound the approximation error. We apply our approach to a computer vision task (classifying gender from images), revealing sensitivity to shifts in non-causal attributes.

        ----

        ## [1228] CogView2: Faster and Better Text-to-Image Generation via Hierarchical Transformers

        **Authors**: *Ming Ding, Wendi Zheng, Wenyi Hong, Jie Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6baec7c4ba0a8734ccbd528a8090cb1f-Abstract-Conference.html)

        **Abstract**:

        Development of transformer-based text-to-image models is impeded by its slow generation and complexity, for high-resolution images. In this work, we put forward a solution based on hierarchical transformers and local parallel autoregressive generation.  We pretrain a 6B-parameter transformer with a simple and flexible self-supervised task, a cross-modal general language model (CogLM), and fine-tune it for fast super-resolution. The new text-to-image system, CogView2, shows very competitive generation compared to concurrent state-of-the-art DALL-E-2, and naturally supports interactive text-guided editing on images.

        ----

        ## [1229] Recursive Reasoning in Minimax Games: A Level $k$ Gradient Play Method

        **Authors**: *Zichu Liu, Lacra Pavel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6bdc60ac98a8eda10922d35eced6f969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6bdc60ac98a8eda10922d35eced6f969-Abstract-Conference.html)

        **Abstract**:

        Despite the success of generative adversarial networks (GANs) in generating visually appealing images, they are notoriously challenging to train. In order to stabilize the learning dynamics in minimax games, we propose a novel recursive reasoning algorithm: Level $k$ Gradient Play (Lv.$k$ GP) algorithm. Our algorithm does not require sophisticated heuristics or second-order information, as do existing algorithms based on predictive updates. We show that as k increases, Lv.$k$ GP converges asymptotically towards an accurate estimation of players' future strategy.Moreover, we justify that Lv.$\infty$ GP naturally generalizes a line of provably convergent game dynamics which rely on predictive updates. Furthermore, we provide its local convergence property in nonconvex-nonconcave zero-sum games and global convergence in bilinear and quadratic games. By combining Lv.$k$ GP with Adam optimizer, our algorithm shows a clear advantage in terms of performance and computational overhead compared to other methods. Using a single Nvidia RTX3090 GPU and 30 times fewer parameters than BigGAN on CIFAR-10, we achieve an FID of 10.17 for unconditional image generation within 30 hours, allowing GAN training on common computational resources to reach state-of-the-art performance.

        ----

        ## [1230] When to Ask for Help: Proactive Interventions in Autonomous Reinforcement Learning

        **Authors**: *Annie Xie, Fahim Tajwar, Archit Sharma, Chelsea Finn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference.html)

        **Abstract**:

        A long-term goal of reinforcement learning is to design agents that can autonomously interact and learn in the world. A critical challenge to such autonomy is the presence of irreversible states which require external assistance to recover from, such as when a robot arm has pushed an object off of a table. While standard agents require constant monitoring to decide when to intervene, we aim to design proactive agents that can request human intervention only when needed. To this end, we propose an algorithm that efficiently learns to detect and avoid states that are irreversible, and proactively asks for help in case the agent does enter them. On a suite of continuous control environments with unknown irreversible states, we find that our algorithm exhibits better sample- and intervention-efficiency compared to existing methods.

        ----

        ## [1231] Reinforcement Learning with Neural Radiance Fields

        **Authors**: *Danny Driess, Ingmar Schubert, Pete Florence, Yunzhu Li, Marc Toussaint*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6c294f059e3d77d58dbb8fe48f21fe00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6c294f059e3d77d58dbb8fe48f21fe00-Abstract-Conference.html)

        **Abstract**:

        It is a long-standing problem to find effective representations for training reinforcement learning (RL) agents. This paper demonstrates that learning state representations with supervision from Neural Radiance Fields (NeRFs) can improve the performance of RL compared to other learned representations or even low-dimensional, hand-engineered state information. Specifically, we propose to train an encoder that maps multiple image observations to a latent space describing the objects in the scene. The decoder built from a latent-conditioned NeRF serves as the supervision signal to learn the latent space. An RL algorithm then operates on the learned latent space as its state representation. We call this NeRF-RL. Our experiments indicate that NeRF as supervision leads to a latent space better suited for the downstream RL tasks involving robotic object manipulations like hanging mugs on hooks, pushing objects, or opening doors.Video: https://dannydriess.github.io/nerf-rl

        ----

        ## [1232] Function Classes for Identifiable Nonlinear Independent Component Analysis

        **Authors**: *Simon Buchholz, Michel Besserve, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6c5da478b9d13f541993d67897a0bb30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6c5da478b9d13f541993d67897a0bb30-Abstract-Conference.html)

        **Abstract**:

        Unsupervised learning of latent variable models (LVMs) is widely used to represent data in machine learning. When such model reflects the ground truth factors and the mechanisms mapping them to observations, there is reason to expect that such models allow generalisation in downstream tasks. It is however well known that such identifiability guaranties are typically not achievable without putting constraints on the model class. This is notably the case for nonlinear Independent Component Analysis, in which the LVM maps statistically independent variables to observations via a deterministic nonlinear function. Several families of spurious solutions fitting perfectly the data, but that do not correspond to the ground truth factors can be constructed in generic settings. However, recent work suggests that constraining the function class of such models may promote identifiability. Specifically, function classes with constraints on their partial derivatives, gathered in the Jacobian matrix, have been proposed, such as orthogonal coordinate transformations (OCT), which impose orthogonality of the Jacobian columns. In the present work, we prove that a subclass of these transformations, conformal maps, is identifiable and provide novel theoretical results suggesting that OCTs have properties that prevent families of spurious solutions to spoil identifiability in a generic setting.

        ----

        ## [1233] Self-supervised Heterogeneous Graph Pre-training Based on Structural Clustering

        **Authors**: *Yaming Yang, Ziyu Guan, Zhe Wang, Wei Zhao, Cai Xu, Weigang Lu, Jianbin Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6c7297baffe5c85ea1d9e1ccb1222ab8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6c7297baffe5c85ea1d9e1ccb1222ab8-Abstract-Conference.html)

        **Abstract**:

        Recent self-supervised pre-training methods on Heterogeneous Information Networks (HINs) have shown promising competitiveness over traditional semi-supervised Heterogeneous Graph Neural Networks (HGNNs). Unfortunately, their performance heavily depends on careful customization of various strategies for generating high-quality positive examples and negative examples, which notably limits their flexibility and generalization ability. In this work, we present SHGP, a novel Self-supervised Heterogeneous Graph Pre-training approach, which does not need to generate any positive examples or negative examples. It consists of two modules that share the same attention-aggregation scheme. In each iteration, the Att-LPA module produces pseudo-labels through structural clustering, which serve as the self-supervision signals to guide the Att-HGNN module to learn object embeddings and attention coefficients. The two modules can effectively utilize and enhance each other, promoting the model to learn discriminative embeddings. Extensive experiments on four real-world datasets demonstrate the superior effectiveness of SHGP against state-of-the-art unsupervised baselines and even semi-supervised baselines. We release our source code at: https://github.com/kepsail/SHGP.

        ----

        ## [1234] Learning Audio-Visual Dynamics Using Scene Graphs for Audio Source Separation

        **Authors**: *Moitreya Chatterjee, Narendra Ahuja, Anoop Cherian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6c92839f0f9cddc96c694712a7143b09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6c92839f0f9cddc96c694712a7143b09-Abstract-Conference.html)

        **Abstract**:

        There exists an unequivocal distinction between the sound produced by a static source and that produced by a moving one, especially when the source moves towards or away from the microphone. In this paper, we propose to use this connection between audio and visual dynamics for solving two challenging tasks simultaneously, namely: (i) separating audio sources from a mixture using visual cues, and (ii) predicting the 3D visual motion of a sounding source using its separated audio. Towards this end, we present Audio Separator and Motion Predictor (ASMP) -- a deep learning framework that leverages the 3D structure of the scene and the motion of sound sources for better audio source separation. At the heart of ASMP is a 2.5D scene graph capturing various objects in the video and their pseudo-3D spatial proximities. This graph is constructed by registering together 2.5D monocular depth predictions from the 2D video frames and associating the 2.5D scene regions with the outputs of an object detector applied on those frames. The ASMP task is then mathematically modeled as the joint problem of: (i) recursively segmenting the 2.5D scene graph into several sub-graphs, each associated with a constituent sound in the input audio mixture (which is then separated) and (ii) predicting the 3D motions of the corresponding sound sources from the separated audio. To empirically evaluate ASMP, we present experiments on two challenging audio-visual datasets, viz. Audio Separation in the Wild (ASIW) and Audio Visual Event (AVE). Our results demonstrate that ASMP achieves a clear improvement in source separation quality, outperforming prior works on both datasets, while also estimating the direction of motion of the sound sources better than other methods.

        ----

        ## [1235] On the Sample Complexity of Stabilizing LTI Systems on a Single Trajectory

        **Authors**: *Yang Hu, Adam Wierman, Guannan Qu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ca5d2665de83394f437dad0c3746907-Abstract-Conference.html)

        **Abstract**:

        Stabilizing an unknown dynamical system is one of the central problems in control theory. In this paper, we study the sample complexity of the learn-to-stabilize problem in Linear Time-Invariant (LTI) systems on a single trajectory. Current state-of-the-art approaches require a sample complexity linear in $n$, the state dimension, which incurs a state norm that blows up exponentially in $n$. We propose a novel algorithm based on spectral decomposition that only needs to learn ``a small part'' of the dynamical matrix acting on its unstable subspace. We show that, under proper assumptions, our algorithm stabilizes an LTI system on a single trajectory with $O(k \log n)$ samples, where $k$ is the instability index of the system. This represents the first sub-linear sample complexity result for the stabilization of LTI systems under the regime when $k = o(n)$.

        ----

        ## [1236] coVariance Neural Networks

        **Authors**: *Saurabh Sihag, Gonzalo Mateos, Corey McMillan, Alejandro Ribeiro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cb00ce1a21a090a3dae04cebebd8341-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cb00ce1a21a090a3dae04cebebd8341-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNN) are an effective framework that exploit inter-relationships within graph-structured data for learning. Principal component analysis (PCA) involves the projection of data on the eigenspace of the covariance matrix and draws similarities with the graph convolutional filters in GNNs. Motivated by this observation, we study a GNN architecture, called coVariance neural network (VNN), that operates on sample covariance matrices as graphs. We theoretically establish the stability of VNNs to perturbations in the covariance matrix, thus, implying an advantage over standard PCA-based data analysis approaches that are prone to instability due to principal components associated with close eigenvalues. Our experiments on real-world datasets validate our theoretical results and show that VNN performance is indeed more stable than PCA-based statistical approaches. Moreover, our experiments on multi-resolution datasets also demonstrate that VNNs are amenable to transferability of performance over covariance matrices of different dimensions; a feature that is infeasible for PCA-based approaches.

        ----

        ## [1237] Taming Fat-Tailed ("Heavier-Tailed" with Potentially Infinite Variance) Noise in Federated Learning

        **Authors**: *Haibo Yang, Peiwen Qiu, Jia Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cb7246003d556c4d1cbf9c17c392ee3-Abstract-Conference.html)

        **Abstract**:

        In recent years, federated learning (FL) has emerged as an important distributed machine learning paradigm to collaboratively learn a global model with multiple clients, while keeping data local and private. However, a key assumption in most existing works on FL algorithms' convergence analysis is that the noise in stochastic first-order information has a finite variance. Although this assumption covers all light-tailed (i.e., sub-exponential) and some heavy-tailed noise distributions (e.g., log-normal, Weibull, and some Pareto distributions), it fails for many fat-tailed noise distributions (i.e., ``heavier-tailed'' with potentially infinite variance) that have been empirically observed in the FL literature. To date, it remains unclear whether one can design convergent algorithms for FL systems that experience fat-tailed noise. This motivates us to fill this gap in this paper by proposing an algorithmic framework called $\mathsf{FAT}$-$\mathsf{Clipping}~$ (\ul{f}ederated \ul{a}veraging with \ul{t}wo-sided learning rates and \ul{clipping}), which contains two variants: $\mathsf{FAT}$-$\mathsf{Clipping}~$ per-round ($\mathsf{FAT}$-$\mathsf{Clipping}$-$\mathsf{PR}$) and $\mathsf{FAT}$-$\mathsf{Clipping}~$ per-iteration ($\mathsf{FAT}$-$\mathsf{Clipping}$-$\mathsf{PI}$). Specifically, for the largest $\alpha \in (1,2]$ such that the fat-tailed noise in FL still has a bounded $\alpha$-moment, we show that both variants achieve $\mathcal{O}((mT)^{\frac{2-\alpha}{\alpha}})$ and $\mathcal{O}((mT)^{\frac{1-\alpha}{3\alpha-2}})$ convergence rates in the strongly-convex and general non-convex settings, respectively, where $m$ and $T$ are the numbers of clients and communication rounds. Moreover, at the expense of more clipping operations compared to $\mathsf{FAT}$-$\mathsf{Clipping}$-$\mathsf{PR}$, $\mathsf{FAT}$-$\mathsf{Clipping}$-$\mathsf{PI}~$ further enjoys a linear speedup effect with respect to the number of local updates at each client and being lower-bound-matching (i.e., order-optimal). Collectively, our results advance the understanding of designing efficient algorithms for FL systems that exhibit fat-tailed first-order oracle information.

        ----

        ## [1238] Exploring Figure-Ground Assignment Mechanism in Perceptual Organization

        **Authors**: *Wei Zhai, Yang Cao, Jing Zhang, Zheng-Jun Zha*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cc31b44d88dce8380d36e81485cd07f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cc31b44d88dce8380d36e81485cd07f-Abstract-Conference.html)

        **Abstract**:

        Perceptual organization is a challenging visual task that aims to perceive and group the individual visual element so that it is easy to understand the meaning of the scene as a whole. Most recent methods building upon advanced Convolutional Neural Network (CNN) come from learning discriminative representation and modeling context hierarchically. However, when the visual appearance difference between foreground and background is obscure, the performance of existing methods degrades significantly due to the visual ambiguity in the discrimination process. In this paper, we argue that the figure-ground assignment mechanism, which conforms to human vision cognitive theory, can be explored to empower CNN to achieve a robust perceptual organization despite visual ambiguity. Specifically, we present a novel Figure-Ground-Aided (FGA) module to learn the configural statistics of the visual scene and leverage it for the reduction of visual ambiguity. Particularly, we demonstrate the benefit of using stronger supervisory signals by teaching (FGA) module to perceive configural cues, \ie, convexity and lower region, that human deem important for the perceptual organization. Furthermore, an Interactive Enhancement Module (IEM) is devised to leverage such configural priors to assist representation learning, thereby achieving robust perception organization with complex visual ambiguities. In addition, a well-founded visual segregation test is designed to validate the capability of the proposed FGA mechanism explicitly. Comprehensive evaluation results demonstrate our proposed FGA mechanism can effectively enhance the capability of perception organization on various baseline models. Nevertheless, the model augmented via our proposed FGA mechanism also outperforms state-of-the-art approaches on four challenging real-world applications.

        ----

        ## [1239] Two-Stream Network for Sign Language Recognition and Translation

        **Authors**: *Yutong Chen, Ronglai Zuo, Fangyun Wei, Yu Wu, Shujie Liu, Brian Mak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cd3ac24cdb789beeaa9f7145670fcae-Abstract-Conference.html)

        **Abstract**:

        Sign languages are visual languages using manual articulations and non-manual elements to convey information. For sign language recognition and translation, the majority of existing approaches directly encode RGB videos into hidden representations. RGB videos, however, are raw signals with substantial visual redundancy, leading the encoder to overlook the key information for sign language understanding. To mitigate this problem and better incorporate domain knowledge, such as handshape and body movement, we introduce a dual visual encoder containing two separate streams to model both the raw videos and the keypoint sequences generated by an off-the-shelf keypoint estimator. To make the two streams interact with each other, we explore a variety of techniques, including bidirectional lateral connection, sign pyramid network with auxiliary supervision, and frame-level self-distillation. The resulting model is called TwoStream-SLR, which is competent for sign language recognition (SLR). TwoStream-SLR is extended to a sign language translation (SLT) model, TwoStream-SLT, by simply attaching an extra translation network. Experimentally, our TwoStream-SLR and TwoStream-SLT achieve state-of-the-art performance on SLR and SLT tasks across a series of datasets including Phoenix-2014, Phoenix-2014T, and CSL-Daily.

        ----

        ## [1240] VisFIS: Visual Feature Importance Supervision with Right-for-the-Right-Reason Objectives

        **Authors**: *Zhuofan Ying, Peter Hase, Mohit Bansal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cdb2cbb2083477cca5243843d6dad06-Abstract-Conference.html)

        **Abstract**:

        Many past works aim to improve visual reasoning in models by supervising feature importance (estimated by model explanation techniques) with human annotations such as highlights of important image regions. However, recent work has shown that performance gains from feature importance (FI) supervision for Visual Question Answering (VQA) tasks persist even with random supervision, suggesting that these methods do not meaningfully align model FI with human FI. In this paper, we show that model FI supervision can meaningfully improve VQA model accuracy as well as performance on several Right-for-the-Right-Reason (RRR) metrics by optimizing for four key model objectives: (1) accurate predictions given limited but sufficient information (Sufficiency); (2) max-entropy predictions given no important information (Uncertainty); (3) invariance of predictions to changes in unimportant features (Invariance); and (4) alignment between model FI explanations and human FI explanations (Plausibility). Our best performing method, Visual Feature Importance Supervision (VISFIS), outperforms strong baselines on benchmark VQA datasets in terms of both in-distribution and out-of-distribution accuracy. While past work suggests that the mechanism for improved accuracy is through improved explanation plausibility, we show that this relationship depends crucially on explanation faithfulness (whether explanations truly represent the model’s internal reasoning). Predictions are more accurate when explanations are plausible and faithful, and not when they are plausible but not faithful. Lastly, we show that, surprisingly, RRR metrics are not predictive of out-of-distribution model accuracy when controlling for a model’s in-distribution accuracy, which calls into question the value of these metrics for evaluating model reasoning.

        ----

        ## [1241] Operative dimensions in unconstrained connectivity of recurrent neural networks

        **Authors**: *Renate Krause, Matthew Cook, Sepp Kollmorgen, Valerio Mante, Giacomo Indiveri*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6cdd4ce9330025967dd1ed0bed3010f5-Abstract-Conference.html)

        **Abstract**:

        Recurrent Neural Networks (RNN) are commonly used models to study neural computation. However, a comprehensive understanding of how dynamics in RNN emerge from the underlying connectivity is largely lacking. Previous work derived such an understanding for RNN fulfilling very specific constraints on their connectivity, but it is unclear whether the resulting insights apply more generally. Here we study how network dynamics are related to network connectivity in RNN trained without any specific constraints on several tasks previously employed in neuroscience. Despite the apparent high-dimensional connectivity of these RNN, we show that a low-dimensional, functionally relevant subspace of the weight matrix can be found through the identification of \textit{operative} dimensions, which we define as components of the connectivity whose removal has a large influence on local RNN dynamics. We find that a weight matrix built from only a few operative dimensions is sufficient for the RNN to operate with the original performance, implying that much of the high-dimensional structure of the trained connectivity is functionally irrelevant. The existence of a low-dimensional, operative subspace in the weight matrix simplifies the challenge of linking connectivity to network dynamics and suggests that independent network functions may be placed in specific, separate subspaces of the weight matrix to avoid catastrophic forgetting in continual learning.

        ----

        ## [1242] Batch size-invariance for policy optimization

        **Authors**: *Jacob Hilton, Karl Cobbe, John Schulman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ceb6c2150bbf46fd75528a6cd6be793-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ceb6c2150bbf46fd75528a6cd6be793-Abstract-Conference.html)

        **Abstract**:

        We say an algorithm is batch size-invariant if changes to the batch size can largely be compensated for by changes to other hyperparameters. Stochastic gradient descent is well-known to have this property at small batch sizes, via the learning rate. However, some policy optimization algorithms (such as PPO) do not have this property, because of how they control the size of policy updates. In this work we show how to make these algorithms batch size-invariant. Our key insight is to decouple the proximal policy (used for controlling policy updates) from the behavior policy (used for off-policy corrections). Our experiments help explain why these algorithms work, and additionally show how they can make more efficient use of stale data.

        ----

        ## [1243] Inference and Sampling for Archimax Copulas

        **Authors**: *Yuting Ng, Ali Hasan, Vahid Tarokh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d00071564ec447466fc4577743cf1b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d00071564ec447466fc4577743cf1b3-Abstract-Conference.html)

        **Abstract**:

        Understanding multivariate dependencies in both the bulk and the tails of a distribution is an important problem for many applications, such as ensuring algorithms are robust to observations that are infrequent but have devastating effects. Archimax copulas are a family of distributions endowed with a precise representation that allows simultaneous modeling of the bulk and the tails of a distribution. Rather than separating the two as is typically done in practice, incorporating additional information from the bulk may improve inference of the tails, where observations are limited. Building on the stochastic representation of Archimax copulas, we develop a non-parametric inference method and sampling algorithm. Our proposed methods, to the best of our knowledge, are the first that allow for highly flexible and scalable inference and sampling algorithms, enabling the increased use of Archimax copulas in practical settings. We experimentally compare to state-of-the-art density modeling techniques, and the results suggest that the proposed method effectively extrapolates to the tails while scaling to higher dimensional data. Our findings suggest that the proposed algorithms can be used in a variety of applications where understanding the interplay between the bulk and the tails of a distribution is necessary, such as healthcare and safety.

        ----

        ## [1244] Neural Shape Deformation Priors

        **Authors**: *Jiapeng Tang, Lev Markhasin, Bi Wang, Justus Thies, Matthias Nießner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d09ef61aeb76be676b358f6f87b3484-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d09ef61aeb76be676b358f6f87b3484-Abstract-Conference.html)

        **Abstract**:

        We present Neural Shape Deformation Priors, a novel method for shape manipulation that predicts mesh deformations of non-rigid objects from user-provided handle movements. State-of-the-art methods cast this problem as an optimization task, where the input source mesh is iteratively deformed to minimize an objective function according to hand-crafted regularizers such as ARAP. In this work, we learn the deformation behavior based on the underlying geometric properties of a shape, while leveraging a large-scale dataset containing a diverse set of non-rigid deformations. Specifically, given a source mesh and desired target locations of handles that describe the partial surface deformation, we predict a continuous deformation field that is defined in 3D space to describe the space deformation. To this end, we introduce transformer-based deformation networks that represent a shape deformation as a composition of local surface deformations. It learns a set of local latent codes anchored in 3D space, from which we can learn a set of continuous deformation functions for local surfaces. Our method can be applied to challenging deformations and generalizes well to unseen deformations. We validate our approach in experiments using the DeformingThing4D dataset, and compare to both classic optimization-based and recent neural network-based methods.

        ----

        ## [1245] The Curse of Unrolling: Rate of Differentiating Through Optimization

        **Authors**: *Damien Scieur, Gauthier Gidel, Quentin Bertrand, Fabian Pedregosa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d53193a098b982229340a7c3eb0ecbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d53193a098b982229340a7c3eb0ecbf-Abstract-Conference.html)

        **Abstract**:

        Computing the Jacobian of the solution of an optimization problem is a central problem in machine learning, with applications in hyperparameter optimization, meta-learning, optimization as a layer, and dataset distillation, to name a few. Unrolled differentiation is a popular heuristic that approximates the solution using an iterative solver and differentiates it through the computational path. This work provides a non-asymptotic convergence-rate analysis of this approach on quadratic objectives for gradient descent and the Chebyshev method. We show that to ensure convergence of the Jacobian, we can either 1) choose a large learning rate leading to a fast asymptotic convergence but accept that the algorithm may have an arbitrarily long burn-in phase or 2) choose a smaller learning rate leading to an immediate but slower convergence. We refer to this phenomenon as the curse of unrolling.Finally, we discuss open problems relative to this approach, such as deriving a practical update rule for the optimal unrolling strategy and making novel connections with the field of Sobolev orthogonal polynomials.

        ----

        ## [1246] A Deep Learning Dataloader with Shared Data Preparation

        **Authors**: *Jian Xie, Jingwei Xu, Guochang Wang, Yuan Yao, Zenan Li, Chun Cao, Hanghang Tong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d538a6e667960b168d3d947eb6207a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d538a6e667960b168d3d947eb6207a6-Abstract-Conference.html)

        **Abstract**:

        Executing a family of Deep Neural Networks (DNNs) training jobs on the same or similar datasets in parallel is typical in current deep learning scenarios. It is time-consuming and resource-intensive because each job repetitively prepares (i.e., loads and preprocesses) the data independently, causing redundant consumption of I/O and computations. Although the page cache or a centralized cache component can alleviate the redundancies by reusing the data prep work, each job's data sampled uniformly at random presents a low sampling locality in the shared dataset that causes the heavy cache thrashing. Prior work tries to solve the problem by enforcing all training jobs iterating over the dataset in the same order and requesting each data in lockstep, leading to strong constraints: all jobs must have the same dataset and run simultaneously. In this paper, we propose a dependent sampling algorithm (DSA) and domain-specific cache policy to relax the constraints. Besides, a novel tree data structure is designed to efficiently implement DSA. Based on the proposed technologies, we implemented a prototype system, named Joader, which can share data prep work as long as the datasets share partially. We evaluate the proposed Joader in practical scenarios, showing a greater versatility and superiority over training speed improvement (up to 500% in ResNet18).

        ----

        ## [1247] Learning Physics Constrained Dynamics Using Autoencoders

        **Authors**: *Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, Peter J. Ramadge*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d5e035724687454549b97d6c805dc84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d5e035724687454549b97d6c805dc84-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of estimating states (e.g., position and velocity) and physical parameters (e.g., friction, elasticity) from a sequence of observations when provided a dynamic equation that describes the behavior of the system. The dynamic equation can arise from first principles (e.g., Newtonâ€™s laws) and provide useful cues for learning, but its physical parameters are unknown. To address this problem, we propose a model that estimates states and physical parameters of the system using two main components. First, an autoencoder compresses a sequence of observations (e.g., sensor measurements, pixel images) into a sequence for the state representation that is consistent with physics by including a simulation of the dynamic equation. Second, an estimator is coupled with the autoencoder to predict the values of the physical parameters. We also theoretically and empirically show that using Fourier feature mappings improves generalization of the estimator in predicting physical parameters compared to raw state sequences. In our experiments on three visual and one sensor measurement tasks, our model imposes interpretability on latent states and achieves improved generalization performance for long-term prediction of system dynamics over state-of-the-art baselines.

        ----

        ## [1248] Variational Model Perturbation for Source-Free Domain Adaptation

        **Authors**: *Mengmeng Jing, Xiantong Zhen, Jingjing Li, Cees Snoek*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d7a9f292360193eb530d693f7941c73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d7a9f292360193eb530d693f7941c73-Abstract-Conference.html)

        **Abstract**:

        We aim for source-free domain adaptation, where the task is to deploy a model pre-trained on source domains to target domains. The challenges stem from the distribution shift from the source to the target domain, coupled with the unavailability of any source data and labeled target data for optimization. Rather than fine-tuning the model by updating the parameters, we propose to perturb the source model to achieve adaptation to target domains. We introduce perturbations into the model parameters by variational Bayesian inference in a probabilistic framework. By doing so, we can effectively adapt the model to the target domain while largely preserving the discriminative ability. Importantly, we demonstrate the theoretical connection to learning Bayesian neural networks, which proves the generalizability of the perturbed model to target domains. To enable more efficient optimization, we further employ a parameter sharing strategy, which substantially reduces the learnable parameters compared to a fully Bayesian neural network. Our model perturbation provides a new probabilistic way for domain adaptation which enables efficient adaptation to target domains while maximally preserving knowledge in source models. Experiments on several source-free benchmarks under three different evaluation settings verify the effectiveness of the proposed variational model perturbation for source-free domain adaptation.

        ----

        ## [1249] On the non-universality of deep learning: quantifying the cost of symmetry

        **Authors**: *Emmanuel Abbe, Enric Boix-Adserà*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6d9aac9407bcb1a5957401fa0b8de693-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6d9aac9407bcb1a5957401fa0b8de693-Abstract-Conference.html)

        **Abstract**:

        We prove limitations on what neural networks trained by noisy gradient descent (GD) can efficiently learn. Our results apply whenever GD training is equivariant, which holds for many standard architectures and initializations. As applications, (i) we characterize the functions that fully-connected networks can weak-learn on the binary hypercube and unit sphere, demonstrating that depth-2 is as powerful as any other depth for this task; (ii) we extend the merged-staircase necessity result for learning with latent low-dimensional structure [ABM22] to beyond the mean-field regime. Under cryptographic assumptions, we also show hardness results for learning with fully-connected networks trained by stochastic gradient descent (SGD).

        ----

        ## [1250] Sharper Convergence Guarantees for Asynchronous SGD for Distributed and Federated Learning

        **Authors**: *Anastasia Koloskova, Sebastian U. Stich, Martin Jaggi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6db3ea527f53682657b3d6b02a841340-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6db3ea527f53682657b3d6b02a841340-Abstract-Conference.html)

        **Abstract**:

        We study the asynchronous stochastic gradient descent algorithm, for distributed training over $n$ workers that might be heterogeneous. In this algorithm, workers compute stochastic gradients in parallel at their own pace and return them to the server without any synchronization.Existing convergence rates of this algorithm for non-convex smooth objectives depend on the maximum delay $\tau_{\max}$ and reach an $\epsilon$-stationary point after $O\!\left(\sigma^2\epsilon^{-2}+ \tau_{\max}\epsilon^{-1}\right)$ iterations,  where $\sigma$ is the variance of stochastic gradients. In this work (i) we obtain a tighter convergence rate of $O\!\left(\sigma^2\epsilon^{-2}+ \sqrt{\tau_{\max}\tau_{avg}}\epsilon^{-1}\right)$ *without any change in the algorithm* where $\tau_{avg}$ is the average delay, which can be significantly smaller than $\tau_{\max}$. We also provide (ii) a simple delay-adaptive learning rate scheme, under which asynchronous SGD achieves a convergence rate of $O\!\left(\sigma^2\epsilon^{-2}+ \tau_{avg}\epsilon^{-1}\right)$, and does not require any extra hyperparameter tuning nor extra communications. Our result allows to show *for the first time* that asynchronous SGD is *always faster* than mini-batch SGD. In addition, (iii) we consider the case of heterogeneous functions motivated by federated learning applications and improve the convergence rate by proving a weaker dependence on the maximum delay compared to prior works.

        ----

        ## [1251] A Unified Framework for Alternating Offline Model Training and Policy Learning

        **Authors**: *Shentao Yang, Shujian Zhang, Yihao Feng, Mingyuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6dc02cf4905e873ca6fd0dfc7907e230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6dc02cf4905e873ca6fd0dfc7907e230-Abstract-Conference.html)

        **Abstract**:

        In offline model-based reinforcement learning (offline MBRL), we learn a dynamic model from historically collected data, and subsequently utilize the learned model and fixed datasets for policy learning, without further interacting with the environment. Offline MBRL algorithms can improve the efficiency and stability of policy learning over the model-free algorithms. However, in most of the existing offline MBRL algorithms, the learning objectives for the dynamic models and the policies are isolated from each other. Such an objective mismatch may lead to inferior performance of the learned agents. In this paper, we address this issue by developing an iterative offline MBRL framework, where we maximize a lower bound of the true expected return, by alternating between dynamic-model training and policy learning. With the proposed unified model-policy learning framework, we achieve competitive performance on a wide range of continuous-control offline reinforcement learning datasets. Source code is released at https://github.com/Shentao-YANG/AMPL_NeurIPS2022.

        ----

        ## [1252] Adjoint-aided inference of Gaussian process driven differential equations

        **Authors**: *Paterne Gahungu, Christopher W. Lanyon, Mauricio A. Álvarez, Engineer Bainomugisha, Michael T. Smith, Richard Wilkinson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6dd16c884345ad63e4708367222410e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6dd16c884345ad63e4708367222410e5-Abstract-Conference.html)

        **Abstract**:

        Linear systems occur throughout engineering and the sciences, most notably as differential equations. In many cases the forcing function for the system is unknown, and interest lies in using noisy observations of the system to infer the forcing, as well as other unknown parameters. In differential equations, the forcing function is an unknown function of the independent variables (typically time and space), and can be modelled as a Gaussian process (GP). In this paper we show how the adjoint of a linear system can be used to efficiently infer forcing functions modelled as GPs, after using a truncated basis expansion of the GP kernel. We show how exact conjugate Bayesian inference for the truncated GP can be achieved, in many cases with substantially lower computation than would be required using MCMC methods. We demonstrate the approach on systems of both ordinary and partial differential equations, and show that the basis expansion approach approximates well the true forcing  with a modest number of basis vectors. Finally, we  show how to infer point estimates for the non-linear model parameters, such as the kernel length-scales, using Bayesian optimisation.

        ----

        ## [1253] BOME! Bilevel Optimization Made Easy: A Simple First-Order Approach

        **Authors**: *Bo Liu, Mao Ye, Stephen Wright, Peter Stone, Qiang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html)

        **Abstract**:

        Bilevel optimization (BO) is useful for solving a variety of important machine learning problems including but not limited to hyperparameter optimization, meta-learning, continual learning, and reinforcement learning.Conventional BO methods need to differentiate through the low-level optimization process with implicit differentiation, which requires expensive calculations related to the Hessian matrix. There has been a recent quest for first-order methods for BO, but the methods proposed to date tend to be complicated and impractical for large-scale deep learning applications. In this work, we propose a simple first-order BO algorithm that depends only on first-order gradient information, requires no implicit differentiation, and is practical and efficient for large-scale non-convex functions in deep learning. We provide non-asymptotic convergence analysis of the proposed method to stationary points for non-convex objectives and present empirical results that show its superior practical performance.

        ----

        ## [1254] Mirror Descent with Relative Smoothness in Measure Spaces, with application to Sinkhorn and EM

        **Authors**: *Pierre-Cyril Aubin-Frankowski, Anna Korba, Flavien Léger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6e3daaeca6be8579573f69082b2dd58b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6e3daaeca6be8579573f69082b2dd58b-Abstract-Conference.html)

        **Abstract**:

        Many problems in machine learning can be formulated as optimizing a convex functional over a vector space of measures. This paper studies the convergence of the mirror descent algorithm in this infinite-dimensional setting. Defining Bregman divergences through directional derivatives, we derive the convergence of the scheme for relatively smooth and convex pairs of functionals. Such assumptions allow to handle non-smooth functionals such as the Kullback--Leibler (KL) divergence. Applying our result to joint distributions and KL, we show that Sinkhorn's primal iterations for entropic optimal transport in the continuous setting correspond to a mirror descent, and we obtain a new proof of its (sub)linear convergence. We also show that Expectation Maximization (EM) can always formally be written as a mirror descent. When optimizing only on the latent distribution while fixing the mixtures parameters -- which corresponds to the Richardson--Lucy deconvolution scheme in signal processing -- we derive sublinear rates of convergence.

        ----

        ## [1255] Peer Prediction for Learning Agents

        **Authors**: *Shi Feng, Fang-Yi Yu, Yiling Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6e469fbdc43ade121170f61096f4458b-Abstract-Conference.html)

        **Abstract**:

        Peer prediction refers to a collection of mechanisms for eliciting information from human agents when direct verification of the obtained information is unavailable. They are designed to have a game-theoretic equilibrium where everyone reveals their private information truthfully. This result holds under the assumption that agents are Bayesian and they each adopt a fixed strategy across all tasks. Human agents however are observed in many domains to exhibit learning behavior in sequential settings. In this paper, we explore the dynamics of sequential peer prediction mechanisms when participants are learning agents. We first show that the notion of no regret alone for the agents’ learning algorithms cannot guarantee convergence to the truthful strategy. We then focus on a family of learning algorithms where strategy updates only depend on agents’ cumulative rewards and prove that agents' strategies in the popular Correlated Agreement (CA) mechanism converge to truthful reporting when they use algorithms from this family. This family of algorithms is not necessarily no-regret, but includes several familiar no-regret learning algorithms (e.g multiplicative weight update and Follow the Perturbed Leader) as special cases. Simulation of several algorithms in this family as well as the $\epsilon$-greedy algorithm, which is outside of this family, shows convergence to the truthful strategy in the CA mechanism.

        ----

        ## [1256] Visual Clues: Bridging Vision and Language Foundations for Image Paragraph Captioning

        **Authors**: *Yujia Xie, Luowei Zhou, Xiyang Dai, Lu Yuan, Nguyen Bach, Ce Liu, Michael Zeng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6e4df3406bcf04443ea26d5695454355-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6e4df3406bcf04443ea26d5695454355-Abstract-Conference.html)

        **Abstract**:

        People say, "A picture is worth a thousand words". Then how can we get the rich information out of the image? We argue that by using visual clues to bridge large pretrained vision foundation models and language models, we can do so without any extra cross-modal training. Thanks to the strong zero-shot capability of foundation models, we start by constructing a rich semantic representation of the image (e.g., image tags, object attributes / locations, captions) as a structured textual prompt, called visual clues, using a vision foundation model. Based on visual clues, we use large language model to produce a series of comprehensive descriptions for the visual content, which is then verified by the vision model again to select the candidate that aligns best with the image. We evaluate the quality of generated descriptions by quantitative and qualitative measurement. The results demonstrate the effectiveness of such a structured semantic representation.

        ----

        ## [1257] APT-36K: A Large-scale Benchmark for Animal Pose Estimation and Tracking

        **Authors**: *Yuxiang Yang, Junjie Yang, Yufei Xu, Jing Zhang, Long Lan, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/6e566c91d381bd7a45647d9a90838817-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Animal pose estimation and tracking (APT) is a fundamental task for detecting and tracking animal keypoints from a sequence of video frames. Previous animal-related datasets focus either on animal tracking or single-frame animal pose estimation, and never on both aspects. The lack of APT datasets hinders the development and evaluation of video-based animal pose estimation and tracking methods, limiting the applications in real world, e.g., understanding animal behavior in wildlife conservation. To fill this gap, we make the first step and propose APT-36K, i.e., the first large-scale benchmark for animal pose estimation and tracking. Specifically, APT-36K consists of 2,400 video clips collected and filtered from 30 animal species with 15 frames for each video, resulting in 36,000 frames in total. After manual annotation and careful double-check, high-quality keypoint and tracking annotations are provided for all the animal instances. Based on APT-36K, we benchmark several representative models on the following three tracks: (1) supervised animal pose estimation on a single frame under intra- and inter-domain transfer learning settings, (2) inter-species domain generalization test for unseen animals, and (3) animal pose estimation with animal tracking. Based on the experimental results, we gain some empirical insights and show that APT-36K provides a useful animal pose estimation and tracking benchmark, offering new challenges and opportunities for future research. The code and dataset will be made publicly available at https://github.com/pandorgan/APT-36K.

        ----

        ## [1258] ShuffleMixer: An Efficient ConvNet for Image Super-Resolution

        **Authors**: *Long Sun, Jinshan Pan, Jinhui Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6e60a9023d2c63f7f0856910129ae753-Abstract-Conference.html)

        **Abstract**:

        Lightweight and efficiency are critical drivers for the practical application of image super-resolution (SR) algorithms. We propose a simple and effective approach, ShuffleMixer, for lightweight image super-resolution that explores large convolution and channel split-shuffle operation. In contrast to previous SR models that simply stack multiple small kernel convolutions or complex operators to learn representations, we explore a large kernel ConvNet for mobile-friendly SR design. Specifically, we develop a large depth-wise convolution and two projection layers based on channel splitting and shuffling as the basic component to mix features efficiently. Since the contexts of natural images are strongly locally correlated, using large depth-wise convolutions only is insufficient to reconstruct fine details. To overcome this problem while maintaining the efficiency of the proposed module, we introduce Fused-MBConvs into the proposed network to model the local connectivity of different features. Experimental results demonstrate that the proposed ShuffleMixer is about $3 \times$ smaller than the state-of-the-art efficient SR methods, e.g. CARN, in terms of model parameters and FLOPs while achieving competitive performance.

        ----

        ## [1259] Neural Topological Ordering for Computation Graphs

        **Authors**: *Mukul Gagrani, Corrado Rainone, Yang Yang, Harris Teague, Wonseok Jeon, Roberto Bondesan, Herke van Hoof, Christopher Lott, Weiliang Will Zeng, Piero Zappi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6ef586bdf0af0b609b1d0386a3ce0e4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6ef586bdf0af0b609b1d0386a3ce0e4b-Abstract-Conference.html)

        **Abstract**:

        Recent works on machine learning for combinatorial optimization have shown that learning based approaches can outperform heuristic methods in terms of speed and performance. In this paper, we consider the problem of finding an optimal topological order on a directed acyclic graph (DAG) with focus on the memory minimization problem which arises in compilers. We propose an end-to-end machine learning based approach for topological ordering using an encoder-decoder framework. Our encoder is a novel attention based graph neural network architecture called \emph{Topoformer} which uses different topological transforms of a DAG for message passing. The node embeddings produced by the encoder are converted into node priorities which are used by the decoder to generate a probability distribution over topological orders. We train our model on a dataset of synthetically generated graphs called layered graphs. We show that our model outperforms, or is on-par, with several topological ordering baselines while being significantly faster on synthetic graphs with up to 2k nodes. We also train and test our model on a set of real-world computation graphs, showing performance improvements.

        ----

        ## [1260] Probable Domain Generalization via Quantile Risk Minimization

        **Authors**: *Cian Eastwood, Alexander Robey, Shashank Singh, Julius von Kügelgen, Hamed Hassani, George J. Pappas, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f11132f6ecbbcafafdf6decfc98f7be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f11132f6ecbbcafafdf6decfc98f7be-Abstract-Conference.html)

        **Abstract**:

        Domain generalization (DG) seeks predictors which perform well on unseen test distributions by leveraging data drawn from multiple related training distributions or domains. To achieve this, DG is commonly formulated as an average- or worst-case problem over the set of possible domains. However, predictors that perform well on average lack robustness while predictors that perform well in the worst case tend to be overly-conservative. To address this, we propose a new probabilistic framework for DG where the goal is to learn predictors that perform well with high probability. Our key idea is that distribution shifts seen during training should inform us of probable shifts at test time, which we realize by explicitly relating training and test domains as draws from the same underlying meta-distribution. To achieve probable DG, we propose a new optimization problem called Quantile Risk Minimization (QRM). By minimizing the $\alpha$-quantile of predictor's risk distribution over domains, QRM seeks predictors that perform well with probability $\alpha$. To solve QRM in practice, we propose the Empirical QRM (EQRM) algorithm and provide: (i) a generalization bound for EQRM; and (ii) the conditions under which EQRM recovers the causal predictor as $\alpha \to 1$. In our experiments, we introduce a more holistic quantile-focused evaluation protocol for DG, and demonstrate that EQRM outperforms state-of-the-art baselines on datasets from WILDS and DomainBed.

        ----

        ## [1261] Locating and Editing Factual Associations in GPT

        **Authors**: *Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html)

        **Abstract**:

        We analyze the storage and recall of factual associations in autoregressive transformer language models, finding evidence that these associations correspond to localized, directly-editable computations. We first develop a causal intervention for identifying neuron activations that are decisive in a model's factual predictions. This reveals a distinct set of steps in middle-layer feed-forward modules that mediate factual predictions while processing subject tokens. To test our hypothesis that these computations correspond to factual association recall, we modify feed-forward weights to update specific factual associations using Rank-One Model Editing (ROME).  We find that ROME is effective on a standard zero-shot relation extraction (zsRE) model-editing task, comparable to existing methods. To perform a more sensitive evaluation, we also evaluate ROME on a new dataset of counterfactual assertions, on which it simultaneously maintains both specificity and generalization, whereas other methods sacrifice one or another. Our results confirm an important role for mid-layer feed-forward modules in storing factual associations and suggest that direct manipulation of computational mechanisms may be a feasible approach for model editing. The code, dataset, visualizations, and an interactive demo notebook are available in the supplemental materials.

        ----

        ## [1262] S-PIFu: Integrating Parametric Human Models with PIFu for Single-view Clothed Human Reconstruction

        **Authors**: *Kennard Yanting Chan, Guosheng Lin, Haiyu Zhao, Weisi Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f32db03ef5211f66101ec5972ea9da5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f32db03ef5211f66101ec5972ea9da5-Abstract-Conference.html)

        **Abstract**:

        We present three novel strategies to incorporate a parametric body model into a pixel-aligned implicit model for single-view clothed human reconstruction. Firstly, we introduce ray-based sampling, a novel technique that transforms a parametric model into a set of highly informative, pixel-aligned 2D feature maps. Next, we propose a new type of feature based on blendweights. Blendweight-based labels serve as soft human parsing labels and help to improve the structural fidelity of reconstructed meshes. Finally, we show how we can extract and capitalize on body part orientation information from a parametric model to further improve reconstruction quality. Together, these three techniques form our S-PIFu framework, which significantly outperforms state-of-the-arts methods in all metrics. Our code is available at https://github.com/kcyt/SPIFu.

        ----

        ## [1263] A composable machine-learning approach for steady-state simulations on high-resolution grids

        **Authors**: *Rishikesh Ranade, Chris Hill, Lalit Ghule, Jay Pathak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f697c55beafa9049ab8334400dd1ff5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f697c55beafa9049ab8334400dd1ff5-Abstract-Conference.html)

        **Abstract**:

        In this paper we show that our Machine Learning (ML) approach, CoMLSim (Composable Machine Learning Simulator), can  simulate PDEs on highly-resolved grids with higher accuracy and generalization to out-of-distribution source terms and geometries than traditional ML baselines. Our unique approach combines key principles of traditional PDE solvers with local-learning and low-dimensional manifold techniques to iteratively simulate PDEs on large computational domains. The proposed approach is validated on more than 5 steady-state PDEs across different PDE conditions on highly-resolved grids and comparisons are made with the commercial solver, Ansys Fluent as well as 4 other state-of-the-art ML methods. The numerical experiments show that our approach outperforms ML baselines in terms of 1) accuracy across quantitative metrics and 2) generalization to out-of-distribution conditions as well as domain sizes. Additionally, we provide results for a large number of ablations experiments conducted to highlight components of our approach that strongly influence the results. We conclude that our local-learning and iterative-inferencing approach reduces the challenge of generalization that most ML models face.

        ----

        ## [1264] Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models

        **Authors**: *Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, Xianglong Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f6db140de9c9f111b12ef8a216320a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f6db140de9c9f111b12ef8a216320a9-Abstract-Conference.html)

        **Abstract**:

        Transformer architecture has become the fundamental element of the widespread natural language processing~(NLP) models. With the trends of large NLP models, the increasing memory and computation costs hinder their efficient deployment on resource-limited devices. Therefore, transformer quantization attracts wide research interest. Recent work recognizes that structured outliers are the critical bottleneck for quantization performance. However, their proposed methods increase the computation overhead and still leave the outliers there. To fundamentally address this problem, this paper delves into the inherent inducement and importance of the outliers. We discover that $\boldsymbol \gamma$ in LayerNorm (LN) acts as a sinful amplifier for the outliers, and the importance of outliers varies greatly where some outliers provided by a few tokens cover a large area but can be clipped sharply without negative impacts. Motivated by these findings, we propose an outlier suppression framework including two components: Gamma Migration and Token-Wise Clipping. The Gamma Migration migrates the outlier amplifier to subsequent modules in an equivalent transformation, contributing to a more quantization-friendly model without any extra burden. The Token-Wise Clipping takes advantage of the large variance of token range and designs a token-wise coarse-to-fine pipeline, obtaining a clipping range with minimal final quantization loss in an efficient way. This framework effectively suppresses the outliers and can be used in a plug-and-play mode. Extensive experiments prove that our framework surpasses the existing works and, for the first time, pushes the 6-bit post-training BERT quantization to the full-precision (FP) level. Our code is available at https://github.com/wimh966/outlier_suppression.

        ----

        ## [1265] A Single-timescale Analysis for Stochastic Approximation with Multiple Coupled Sequences

        **Authors**: *Han Shen, Tianyi Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f6dd92b03ff9be7468a6104611c9187-Abstract-Conference.html)

        **Abstract**:

        Stochastic approximation (SA) with multiple coupled sequences has found broad applications in machine learning such as bilevel learning and reinforcement learning (RL). In this paper, we study the finite-time convergence of nonlinear SA with multiple coupled sequences. Different from existing multi-timescale analysis, we seek scenarios where a fine-grained analysis can provide a tight performance guarantee for single-timescale multi-sequence SA (STSA). At the heart of our analysis is the smoothness property of the fixed points in multi-sequence SA that holds in many applications. When all sequences have strongly monotone increments, we establish the iteration complexity of $\mathcal{O}(\epsilon^{-1})$ to achieve $\epsilon$-accuracy, which improves the existing $\mathcal{O}(\epsilon^{-1.5})$ complexity for two coupled sequences. When the main sequence does not have a strongly monotone increment, we establish the iteration complexity of $\mathcal{O}(\epsilon^{-2})$. We showcase the power of our result by applying it to stochastic bilevel and compositional optimization problems, as well as RL problems, all of which recover the best known or lead to improvements over their existing guarantees.

        ----

        ## [1266] Bayesian Risk Markov Decision Processes

        **Authors**: *Yifan Lin, Yuxuan Ren, Enlu Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f7d90b1198fec96defd80b5ebd5bc81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f7d90b1198fec96defd80b5ebd5bc81-Abstract-Conference.html)

        **Abstract**:

        We consider finite-horizon Markov Decision Processes where parameters, such as transition probabilities, are unknown and estimated from data. The popular distributionally robust approach to addressing the parameter uncertainty can sometimes be overly conservative. In this paper, we propose a new formulation, Bayesian risk Markov decision process (BR-MDP), to address parameter uncertainty in MDPs, where a risk functional is applied in nested form to the expected total cost with respect to the Bayesian posterior distributions of the unknown parameters. The proposed formulation provides more flexible risk attitudes towards parameter uncertainty and takes into account the availability of data in future time stages. To solve the proposed formulation with the conditional value-at-risk (CVaR) risk functional, we propose an efficient approximation algorithm by deriving an analytical approximation of the value function and utilizing the convexity of CVaR. We demonstrate the empirical performance of the BR-MDP formulation and proposed algorithms on a gamblerâ€™s betting problem and an inventory control problem.

        ----

        ## [1267] All Politics is Local: Redistricting via Local Fairness

        **Authors**: *Shao-Heng Ko, Erin Taylor, Pankaj K. Agarwal, Kamesh Munagala*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6f7fa4df2c8a79c164d3697898a32bd9-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose to use the concept of local fairness for auditing and ranking redistricting plans. Given a redistricting plan, a deviating group is a population-balanced contiguous region in which a majority of individuals are of the same interest and in the minority of their respective districts;  such a set of individuals have a justified complaint with how the redistricting plan was drawn. A redistricting plan with no deviating groups is called locally fair. We show that the problem of auditing a given plan for local fairness is NP-complete. We present an MCMC approach for auditing as well as ranking redistricting plans. We also present a dynamic programming based algorithm for the auditing problem that we use to demonstrate the efficacy of our MCMC approach. Using these tools, we test local fairness on real-world election data, showing that it is indeed possible to find plans that are almost or exactly locally fair. Further, we show that such plans can be generated while sacrificing very little in terms of compactness and existing fairness measures such as competitiveness of the districts or seat shares of the plans.

        ----

        ## [1268] Confident Adaptive Language Modeling

        **Authors**: *Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, Donald Metzler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract-Conference.html)

        **Abstract**:

        Recent advances in Transformer-based large language models (LLMs) have led to significant performance improvements across many tasks. These gains come with a drastic increase in the models' size, potentially leading to slow and costly use at inference time. In practice, however, the series of generations made by LLMs is composed of varying levels of difficulty. While certain predictions truly benefit from the models' full capacity, other continuations are more trivial and can be solved with reduced compute. In this work, we introduce Confident Adaptive Language Modeling (CALM), a framework for dynamically allocating different amounts of compute per input and generation timestep. Early exit decoding involves several challenges that we address here, such as: (1) what confidence measure to use; (2) connecting sequence-level constraints to local per-token exit decisions; and (3) attending back to missing hidden representations due to early exits in previous tokens. Through theoretical analysis and empirical experiments on three diverse text generation tasks, we demonstrate the efficacy of our framework in reducing compute---potential speedup of up to $\times 3$---while provably maintaining high performance.

        ----

        ## [1269] Submodular Maximization in Clean Linear Time

        **Authors**: *Wenxin Li, Moran Feldman, Ehsan Kazemi, Amin Karbasi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6faf3b8ed0df532c14d0fc009e451b6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6faf3b8ed0df532c14d0fc009e451b6d-Abstract-Conference.html)

        **Abstract**:

        In this paper, we provide the first deterministic algorithm that achieves $1/2$-approximation for monotone submodular maximization subject to a knapsack constraint, while making a number of queries that scales only linearly with the size of the ground set $n$. Moreover, our result automatically paves the way for developing a linear-time deterministic algorithm that achieves the tight $1-1/e$ approximation guarantee for monotone submodular maximization under a cardinality (size) constraint. To complement our positive results, we also show strong information-theoretic lower bounds. More specifically, we show that when the maximum cardinality allowed for a solution is constant, no deterministic or randomized algorithm making a sub-linear number of function evaluations can guarantee any constant approximation ratio. Furthermore,  when the constraint allows the selection of a constant fraction of the ground set, we show that any algorithm making fewer than $\Omega(n/\log(n))$ function evaluations cannot perform better than an algorithm that simply outputs a uniformly random subset of the ground set of the right size. We extend our results to the general case of maximizing a monotone submodular function subject to the intersection of a $p$-set system and multiple knapsack constraints. Finally, we evaluate the performance of our algorithms on multiple real-life applications, including movie recommendation, location summarization, Twitter text summarization, and video summarization.

        ----

        ## [1270] Parameters or Privacy: A Provable Tradeoff Between Overparameterization and Membership Inference

        **Authors**: *Jasper Tan, Blake Mason, Hamid Javadi, Richard G. Baraniuk*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fb83b240844d0e3eb8d457072a071ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fb83b240844d0e3eb8d457072a071ad-Abstract-Conference.html)

        **Abstract**:

        A surprising phenomenon in modern machine learning is the ability of a highly overparameterized model to generalize well (small error on the test data) even when it is trained to memorize the training data (zero error on the training data). This has led to an arms race towards increasingly overparameterized models (c.f., deep learning). In this paper, we study an underexplored hidden cost of overparameterization: the fact that overparameterized models may be more vulnerable to privacy attacks, in particular the membership inference attack that predicts the (potentially sensitive) examples used to train a model. We significantly extend the relatively few empirical results on this problem by theoretically proving for an overparameterized linear regression model in the Gaussian data setting that membership inference vulnerability increases with the number of parameters. Moreover, a range of empirical studies indicates that more complex, nonlinear models exhibit the same behavior. Finally, we extend our analysis towards ridge-regularized linear regression and show in the Gaussian data setting that increased regularization also increases membership inference vulnerability in the overparameterized regime.

        ----

        ## [1271] EF-BV: A Unified Theory of Error Feedback and Variance Reduction Mechanisms for Biased and Unbiased Compression in Distributed Optimization

        **Authors**: *Laurent Condat, Kai Yi, Peter Richtárik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fb9ea5197c0b8ece8a64220fb82cdfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fb9ea5197c0b8ece8a64220fb82cdfe-Abstract-Conference.html)

        **Abstract**:

        In distributed or federated optimization and learning, communication between the different computing units is often the bottleneck and gradient compression is widely used to reduce the number of bits sent within each communication round of iterative methods. There are two classes of compression operators and separate algorithms making use of them. In the case of unbiased random compressors with bounded variance (e.g., rand-k), the DIANA algorithm of Mishchenko et al. (2019), which implements a variance reduction technique for handling the variance introduced by compression, is the current state of the art. In the case of biased and contractive compressors (e.g., top-k), the EF21 algorithm of Richt√°rik et al. (2021), which instead implements an error-feedback mechanism, is the current state of the art. These two classes of compression schemes and algorithms are distinct, with different analyses and proof techniques. In this paper, we unify them into a single framework and propose a new algorithm, recovering DIANA and EF21 as particular cases. Our general approach works with a new, larger class of compressors, which has two parameters, the bias and the variance, and includes unbiased and biased compressors as particular cases. This allows us to inherit the best of the two worlds: like EF21 and unlike DIANA, biased compressors, like top-k, whose good performance in practice is recognized, can be used. And like DIANA and unlike EF21, independent randomness at the compressors allows to mitigate the effects of compression, with the convergence rate improving when the number of parallel workers is large. This is the first time that an algorithm with all these features is proposed. We prove its linear convergence under certain conditions. Our approach takes a step towards better understanding of two so-far distinct worlds of communication-efficient distributed learning.

        ----

        ## [1272] DataMUX: Data Multiplexing for Neural Networks

        **Authors**: *Vishvak Murahari, Carlos E. Jimenez, Runzhe Yang, Karthik Narasimhan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fc46679a7ba2ec82183cf01b80e5133-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fc46679a7ba2ec82183cf01b80e5133-Abstract-Conference.html)

        **Abstract**:

        In this paper, we introduce \emph{data multiplexing} (DataMUX), a technique that enables deep neural networks to process multiple inputs simultaneously using a single compact representation. DataMUX demonstrates that neural networks are  capable of generating accurate predictions over \emph{mixtures} of inputs, resulting in increased inference throughput with minimal extra memory requirements. Our approach uses two key components -- 1) a multiplexing layer that performs a fixed linear transformation to each input before combining them to create a "mixed" representation of the same size as a single input, which is then processed by the base network, and 2) a demultiplexing layer that converts the base network's output back into independent representations before producing predictions for each input. We show the viability of DataMUX for different architectures (Transformers, and to a much lesser extent MLPs and CNNs) across six different tasks spanning sentence classification, named entity recognition and image classification. For instance, DataMUX for Transformers can multiplex up to 20x/40x inputs, achieving up to 11x/18x increase in inference throughput with absolute performance drops of $<2\%$ and $<4\%$ respectively compared to a vanilla Transformer on MNLI, a natural language inference task. We also provide a theoretical construction for multiplexing in self-attention networks and analyze the effect of various design elements in DataMUX.

        ----

        ## [1273] Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators

        **Authors**: *Yuhan Helena Liu, Stephen Smith, Stefan Mihalas, Eric Shea-Brown, Uygar Sümbül*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fca3ed3c54ffeae947ae668a0841ab2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fca3ed3c54ffeae947ae668a0841ab2-Abstract-Conference.html)

        **Abstract**:

        The spectacular successes of recurrent neural network models where key parameters are adjusted via backpropagation-based gradient descent have inspired much thought as to how biological neuronal networks might solve the corresponding synaptic credit assignment problem [1, 2, 3]. There is so far little agreement, however, as to how biological networks could implement the necessary backpropagation through time, given widely recognized constraints of biological synaptic network signaling architectures. Here, we propose that extra-synaptic diffusion of local neuromodulators such as neuropeptides may afford an effective mode of backpropagation lying within the bounds of biological plausibility. Going beyond existing temporal truncation-based gradient approximations [4, 5, 6], our approximate gradient-based update rule, ModProp, propagates credit information through arbitrary time steps. ModProp suggests that modulatory signals can act on receiving cells by convolving their eligibility traces via causal, time-invariant and synapse-type-specific filter taps. Our mathematical analysis of ModProp learning, together with simulation results on benchmark temporal tasks, demonstrate the advantage of ModProp over existing biologically-plausible temporal credit assignment rules. These results suggest a potential neuronal mechanism for signaling credit information related to recurrent interactions over a longer time horizon. Finally, we derive an in-silico implementation of ModProp that could serve as a low-complexity and causal alternative to backpropagation through time.

        ----

        ## [1274] Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering

        **Authors**: *Yongyi Su, Xun Xu, Kui Jia*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fcc2190f456464160921e98393bf50e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fcc2190f456464160921e98393bf50e-Abstract-Conference.html)

        **Abstract**:

        Deploying models on target domain data subject to distribution shift requires adaptation. Test-time training (TTT) emerges as a solution to this adaptation under a realistic scenario where access to full source domain data is not available and instant inference on target domain is required. Despite many efforts into TTT, there is a confusion over the experimental settings, thus leading to unfair comparisons. In this work, we first revisit TTT assumptions and categorize TTT protocols by two key factors. Among the multiple protocols, we adopt a realistic sequential test-time training (sTTT) protocol, under which we further develop a test-time anchored clustering (TTAC) approach to enable stronger test-time feature learning. TTAC discovers clusters in both source and target domain and match the target clusters to the source ones to improve generalization. Pseudo label filtering and iterative updating are developed to improve the effectiveness and efficiency of anchored clustering. We demonstrate that under all TTT protocols TTAC consistently outperforms the state-of-the-art methods on six TTT datasets. We hope this work will provide a fair benchmarking of TTT methods and future research should be compared within respective protocols. A demo code is available at https://github.com/Gorilla-Lab-SCUT/TTAC.

        ----

        ## [1275] Label Noise in Adversarial Training: A Novel Perspective to Study Robust Overfitting

        **Authors**: *Chengyu Dong, Liyuan Liu, Jingbo Shang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fe6a2ba2594521d15af3b1f2162d79c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fe6a2ba2594521d15af3b1f2162d79c-Abstract-Conference.html)

        **Abstract**:

        We show that label noise exists in adversarial training. Such label noise is due to the mismatch between the true label distribution of adversarial examples and the label inherited from clean examples â€“ the true label distribution is distorted by the adversarial perturbation, but is neglected by the common practice that inherits labels from clean examples. Recognizing label noise sheds insights on the prevalence of robust overfitting in adversarial training, and explains its intriguing dependence on perturbation radius and data quality. Also, our label noise perspective aligns well with our observations of the epoch-wise double descent in adversarial training. Guided by our analyses, we proposed a method to automatically calibrate the label to address the label noise and robust overfitting. Our method achieves consistent performance improvements across various models and datasets without introducing new hyper-parameters or additional tuning.

        ----

        ## [1276] Active Labeling: Streaming Stochastic Gradients

        **Authors**: *Vivien Cabannes, Francis R. Bach, Vianney Perchet, Alessandro Rudi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/6fee03d84375a159ecd3769ebbacae83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/6fee03d84375a159ecd3769ebbacae83-Abstract-Conference.html)

        **Abstract**:

        The workhorse of machine learning is stochastic gradient descent.To access stochastic gradients, it is common to consider iteratively input/output pairs of a training dataset.Interestingly, it appears that one does not need full supervision to access stochastic gradients, which is the main motivation of this paper.After formalizing the "active labeling" problem, which focuses on active learning with partial supervision, we provide a streaming technique that provably minimizes the ratio of generalization error over the number of samples.We illustrate our technique in depth for robust regression.

        ----

        ## [1277] CEBaB: Estimating the Causal Effects of Real-World Concepts on NLP Model Behavior

        **Authors**: *Eldar David Abraham, Karel D'Oosterlinck, Amir Feder, Yair Ori Gat, Atticus Geiger, Christopher Potts, Roi Reichart, Zhengxuan Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/701ec28790b29a5bc33832b7bdc4c3b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/701ec28790b29a5bc33832b7bdc4c3b6-Abstract-Conference.html)

        **Abstract**:

        The increasing size and complexity of modern ML systems has improved their predictive capabilities but made their behavior harder to explain. Many techniques for model explanation have been developed in response, but we lack clear criteria for assessing these techniques. In this paper, we cast model explanation as the causal inference problem of estimating causal effects of real-world concepts on the output behavior of ML models given actual input data. We introduce CEBaB, a new benchmark dataset for assessing concept-based explanation methods in Natural Language Processing (NLP). CEBaB consists of short restaurant reviews with human-generated counterfactual reviews in which an aspect (food, noise, ambiance, service) of the dining experience was modified. Original and counterfactual reviews are annotated with multiply-validated sentiment ratings at the aspect-level and review-level. The rich structure of CEBaB allows us to go beyond input features to study the effects of abstract, real-world concepts on model behavior. We use CEBaB to compare the quality of a range of concept-based explanation methods covering different assumptions and conceptions of the problem, and we seek to establish natural metrics for comparative assessments of these methods.

        ----

        ## [1278] TOIST: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation

        **Authors**: *Pengfei Li, Beiwen Tian, Yongliang Shi, Xiaoxue Chen, Hao Zhao, Guyue Zhou, Ya-Qin Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70270a1bc28ecb2a2aefad566c5e556b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70270a1bc28ecb2a2aefad566c5e556b-Abstract-Conference.html)

        **Abstract**:

        Current referring expression comprehension algorithms can effectively detect or segment objects indicated by nouns, but how to understand verb reference is still under-explored. As such, we study the challenging problem of task oriented detection, which aims to find objects that best afford an action indicated by verbs like sit comfortably on. Towards a finer localization that better serves downstream applications like robot interaction, we extend the problem into task oriented instance segmentation. A unique requirement of this task is to select preferred candidates among possible alternatives. Thus we resort to the transformer architecture which naturally models pair-wise query relationships with attention, leading to the TOIST method. In order to leverage pre-trained noun referring expression comprehension models and the fact that we can access privileged noun ground truth during training, a novel noun-pronoun distillation framework is proposed. Noun prototypes are generated in an unsupervised manner and contextual pronoun features are trained to select prototypes. As such, the network remains noun-agnostic during inference. We evaluate TOIST on the large-scale task oriented dataset COCO-Tasks and achieve +10.7% higher $\rm{mAP^{box}}$ than the best-reported results. The proposed noun-pronoun distillation can boost $\rm{mAP^{box}}$ and $\rm{mAP^{mask}}$ by +2.6% and +3.6%. Codes and models are publicly available.

        ----

        ## [1279] Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning

        **Authors**: *Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Y. Zou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/702f4db7543a7432431df588d57bc7c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/702f4db7543a7432431df588d57bc7c9-Abstract-Conference.html)

        **Abstract**:

        We present modality gap, an intriguing geometric phenomenon of the representation space of multi-modal models. Specifically, we show that different data modalities (e.g. images and text) are embedded at arm's length in their shared representation in multi-modal models such as CLIP. Our systematic analysis demonstrates that this gap is caused by a combination of model initialization and contrastive learning optimization. In model initialization, we show empirically and theoretically that the representation of a common deep neural network is restricted to a narrow cone. As a consequence, in a multi-modal model with two encoders, the representations of the two modalities are clearly apart when the model is initialized. During optimization,  contrastive learning keeps the different modalities separate by a certain distance, which is influenced by the temperature parameter in the loss function. Our experiments further demonstrate that varying the modality gap distance has a significant impact in improving the model's downstream zero-shot classification performance and fairness.

        ----

        ## [1280] $\alpha$-ReQ : Assessing Representation Quality in Self-Supervised Learning by measuring eigenspectrum decay

        **Authors**: *Kumar Krishna Agrawal, Arnab Kumar Mondal, Arna Ghosh, Blake A. Richards*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70596d70542c51c8d9b4e423f4bf2736-Abstract-Conference.html)

        **Abstract**:

        Self-Supervised Learning (SSL) with large-scale unlabelled datasets enables learning useful representations for multiple downstream tasks. However, assessing the quality of such representations efficiently poses nontrivial challenges. Existing approaches train linear probes (with frozen features) to evaluate performance on a given task. This is expensive both computationally, since it requires retraining a new prediction head for each downstream task, and statistically, requires task-specific labels for multiple tasks. This poses a natural question, how do we efficiently determine the "goodness" of representations learned with SSL across a wide range of potential downstream tasks? In particular, a task-agnostic statistical measure of representation quality, that predicts generalization without explicit downstream task evaluation, would be highly desirable.    In this work, we analyze characteristics of learned representations $\mathbf{f_\theta}$, in well-trained neural networks with canonical architectures \& across SSL objectives. We observe that the eigenspectrum of the empirical feature covariance $\mathrm{Cov}(\mathbf{f_\theta}$) can be well approximated with the family of power-law distribution. We analytically and empirically (using multiple datasets, e.g. CIFAR, STL10, MIT67, ImageNet) demonstrate that the decay coefficient $\alpha$ serves as a measure of representation quality for tasks that are solvable with a linear readout, i.e. there exist well-defined intervals for $\alpha$ where models exhibit excellent downstream generalization. Furthermore, our experiments suggest that key design parameters in SSL algorithms, such as BarlowTwins, implicitly modulate the decay coefficient of the eigenspectrum ($\alpha$). As $\alpha$ depends only on the features themselves, this measure for model selection with hyperparameter tuning for BarlowTwins enables search with less compute.

        ----

        ## [1281] When Combinatorial Thompson Sampling meets Approximation Regret

        **Authors**: *Pierre Perrault*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/706317675344a9e955814b8307bc20bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/706317675344a9e955814b8307bc20bf-Abstract-Conference.html)

        **Abstract**:

        We study the Combinatorial Thompson Sampling policy (CTS) for combinatorial multi-armed bandit problems (CMAB), within an approximation regret setting. Although CTS has attracted a lot of interest, it has a drawback that other usual CMAB policies do not have when considering non-exact oracles: for some oracles, CTS has a poor approximation regret (scaling linearly with the time horizon $T$) [Wang and Chen, 2018]. A study is then necessary to discriminate the oracles on which CTS could learn. This study was started by Kong et al. [2021]: they gave the first approximation regret analysis of CTS for the greedy oracle, obtaining an upper bound of order $\mathcal{O}{\left(\log(T)/\Delta^2\right)}$, where $\Delta$ is some minimal reward gap. In this paper, our objective is to push this study further than the simple case of the greedy oracle. We provide the first $\mathcal{O}{\left(\log(T)/\Delta\right)}$ approximation regret upper bound for CTS, obtained under a specific condition on the approximation oracle, allowing a reduction to the exact oracle analysis. We thus term this condition Reduce2Exact, and observe that it is satisfied in many concrete examples. Moreover, it can be extended to the probabilistically triggered arms setting, thus capturing even more problems, such as online influence maximization.

        ----

        ## [1282] Pruning has a disparate impact on model accuracy

        **Authors**: *Cuong Tran, Ferdinando Fioretto, Jung-Eun Kim, Rakshit Naidu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7087c949df293f13c0052ac825936e6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7087c949df293f13c0052ac825936e6f-Abstract-Conference.html)

        **Abstract**:

        Network pruning is a widely-used compression technique that is able to significantly scale down overparameterized models with minimal loss of accuracy. This paper shows that pruning may create or exacerbate disparate impacts. The paper sheds light on the factors to cause such disparities, suggesting differences in gradient norms and distance to decision boundary across groups to be responsible for this critical issue. It analyzes these factors in detail, providing both theoretical and empirical support, and proposes a simple, yet effective, solution that mitigates the disparate impacts caused by pruning.

        ----

        ## [1283] Sequence Model Imitation Learning with Unobserved Contexts

        **Authors**: *Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell, Zhiwei Steven Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html)

        **Abstract**:

        We consider imitation learning problems where the learner's ability to mimic the expert increases throughout the course of an episode as more information is revealed. One example of this is when the expert has access to privileged information: while the learner might not be able to accurately reproduce expert behavior early on in an episode, by considering the entire history of states and actions, they might be able to eventually identify the hidden context and act as the expert would. We prove that on-policy imitation learning algorithms (with or without access to a queryable expert) are better equipped to handle these sorts of asymptotically realizable problems than off-policy methods. This is because on-policy algorithms provably learn to recover from their initially suboptimal actions, while off-policy methods treat their suboptimal past actions as though they came from the expert. This often manifests as a latching behavior: a naive repetition of past actions. We conduct experiments in a toy bandit domain that show that there exist sharp phase transitions of whether off-policy approaches are able to match expert performance asymptotically, in contrast to the uniformly good performance of on-policy approaches. We demonstrate that on several continuous control tasks, on-policy approaches are able to use history to identify the context while off-policy approaches actually perform worse when given access to history.

        ----

        ## [1284] Tikhonov Regularization is Optimal Transport Robust under Martingale Constraints

        **Authors**: *Jiajin Li, Sirui Lin, Jose H. Blanchet, Viet Anh Nguyen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/708fdc7911f11585ee7161518e509ae6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/708fdc7911f11585ee7161518e509ae6-Abstract-Conference.html)

        **Abstract**:

        Distributionally robust optimization (DRO) has been shown to offer a principled way to regularize learning models. In this paper, we find that Tikhonov regularization is distributionally robust in an optimal transport sense (i.e. if an adversary chooses distributions in a suitable optimal transport neighborhood of the empirical measure), provided that suitable martingale constraints are also imposed. Further, we introduce a relaxation of the martingale constraints which not only provide a unified viewpoint to a class of existing robust methods but also lead to new regularization tools. To realize these novel tools,  provably efficient computational algorithms are proposed. As a byproduct, the strong duality theorem proved in this paper can be potentially applied to other problems of independent interest.

        ----

        ## [1285] Policy Optimization with Linear Temporal Logic Constraints

        **Authors**: *Cameron Voloshin, Hoang Minh Le, Swarat Chaudhuri, Yisong Yue*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)

        **Abstract**:

        We study the problem of policy optimization (PO) with linear temporal logic (LTL) constraints. The language of LTL allows flexible description of tasks that may be unnatural to encode as a scalar cost function. We consider LTL-constrained PO as a systematic framework, decoupling task specification from policy selection, and an alternative to the standard of cost shaping. With access to a generative model, we develop a model-based approach that enjoys a sample complexity analysis for guaranteeing both task satisfaction and cost optimality (through a reduction to a reachability problem). Empirically, our algorithm can achieve strong  performance even in low sample regimes.

        ----

        ## [1286] Merging Models with Fisher-Weighted Averaging

        **Authors**: *Michael Matena, Colin Raffel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70c26937fbf3d4600b69a129031b66ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70c26937fbf3d4600b69a129031b66ec-Abstract-Conference.html)

        **Abstract**:

        Averaging the parameters of models that have the same architecture and initialization can provide a means of combining their respective capabilities. In this paper, we take the perspective that this "merging" operation can be seen as choosing parameters that approximately maximize the joint likelihood of the posteriors of the models' parameters. Computing a simple average of the models' parameters therefore corresponds to making an isotropic Gaussian approximation to their posteriors. We develop an alternative merging procedure based on the Laplace approximation where we approximate each model's posterior as a Gaussian distribution whose precision matrix corresponds to its Fisher information. We first show that our "Fisher merging" technique provides a performance boost in settings where simple parameter averaging is currently used -- specifically, robust fine-tuning and model ensembling. Then, we compare merging to standard gradient-based transfer learning and demonstrate that merging enables a fundamentally different method for transferring capabilities across models. Specifically, we show that Fisher merging is competitive with gradient-based transfer learning approaches (while being significantly cheaper) in intermediate-task training and domain-adaptive pre-training. We also show that our merging procedure makes it possible to combine models in previously unexplored ways. We release our code to facilitate future research into methods for merging models.

        ----

        ## [1287] Learning to Generate Inversion-Resistant Model Explanations

        **Authors**: *Hoyong Jeong, Suyoung Lee, Sung Ju Hwang, Sooel Son*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70d638f3177d2f0bbdd9f400b43f0683-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70d638f3177d2f0bbdd9f400b43f0683-Abstract-Conference.html)

        **Abstract**:

        The wide adoption of deep neural networks (DNNs) in mission-critical applications has spurred the need for interpretable models that provide explanations of the model's decisions. Unfortunately, previous studies have demonstrated that model explanations facilitate information leakage, rendering DNN models vulnerable to model inversion attacks. These attacks enable the adversary to reconstruct original images based on model explanations, thus leaking privacy-sensitive features. To this end, we present Generative Noise Injector for Model Explanations (GNIME), a novel defense framework that perturbs model explanations to minimize the risk of model inversion attacks while preserving the interpretabilities of the generated explanations. Specifically, we formulate the defense training as a two-player minimax game between the inversion attack network on the one hand, which aims to invert model explanations, and the noise generator network on the other, which aims to inject perturbations to tamper with model inversion attacks. We demonstrate that GNIME significantly decreases the information leakage in model explanations, decreasing transferable classification accuracy in facial recognition models by up to 84.8% while preserving the original functionality of model explanations.

        ----

        ## [1288] Unsupervised Multi-View Object Segmentation Using Radiance Field Propagation

        **Authors**: *Xinhang Liu, Jiaben Chen, Huai Yu, Yu-Wing Tai, Chi-Keung Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70de9e3948645a1be2de657f14d85c6d-Abstract-Conference.html)

        **Abstract**:

        We present radiance field propagation (RFP), a novel approach to segmenting objects in 3D during reconstruction given only unlabeled multi-view images of a scene. RFP is derived from emerging neural radiance field-based techniques, which jointly encodes semantics with appearance and geometry. The core of our method is a novel propagation strategy for individual objects' radiance fields with a bidirectional photometric loss, enabling an unsupervised partitioning of a scene into salient or meaningful regions corresponding to different object instances. To better handle complex scenes with multiple objects and occlusions, we further propose an iterative expectation-maximization algorithm to refine object masks. To the best of our knowledge, RFP is the first unsupervised approach for tackling 3D scene object segmentation for neural radiance field (NeRF) without any supervision, annotations, or other cues such as 3D bounding boxes and prior knowledge of object class. Experiments demonstrate that RFP achieves feasible segmentation results that are more accurate than previous unsupervised image/scene segmentation approaches, and are comparable to existing supervised NeRF-based methods. The segmented object representations enable individual 3D object editing operations. Codes and datasets will be made publicly available.

        ----

        ## [1289] Beyond Mahalanobis Distance for Textual OOD Detection

        **Authors**: *Pierre Colombo, Eduardo Dadalto Câmara Gomes, Guillaume Staerman, Nathan Noiry, Pablo Piantanida*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/70fa5df8e3300dc30bf19bee44a56155-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/70fa5df8e3300dc30bf19bee44a56155-Abstract-Conference.html)

        **Abstract**:

        As the number of AI systems keeps growing, it is fundamental to implement and develop efficient control mechanisms to ensure the safe and proper functioning of machine learning (ML) systems. Reliable out-of-distribution (OOD) detection aims to detect test samples that are statistically far from the training distribution, as they might cause failures of in-production systems. In this paper, we propose a new detector called TRUSTED. Different from previous works, TRUSTED key components (i) include a novel OOD score relying on the concept of statistical data depth, (ii) rely on the ideaâ€™s full potential that all hidden layers of the network carry information regarding OOD. Our extensive experiments, comparing over 51k model configurations including different checkpoints, seed and various datasets, demonstrate that TRUSTED achieve state-of-the-art performances by producing an improvement of over 3 AUROC points.

        ----

        ## [1290] FasterRisk: Fast and Accurate Interpretable Risk Scores

        **Authors**: *Jiachang Liu, Chudi Zhong, Boxuan Li, Margo I. Seltzer, Cynthia Rudin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7103444259031cc58051f8c9a4868533-Abstract-Conference.html)

        **Abstract**:

        Over the last century, risk scores have been the most popular form of predictive model used in healthcare and criminal justice. Risk scores are sparse linear models with integer coefficients; often these models can be memorized or placed on an index card. Typically, risk scores have been created either without data or by rounding logistic regression coefficients, but these methods do not reliably produce high-quality risk scores. Recent work used mathematical programming, which is computationally slow. We introduce an approach for efficiently producing a collection of high-quality risk scores learned from data. Specifically, our approach  produces a pool of almost-optimal sparse continuous solutions, each with a different support set, using a beam-search algorithm. Each of these continuous solutions is transformed into a separate risk score through a "star ray" search, where a range of multipliers are considered before rounding the coefficients sequentially to maintain low logistic loss. Our algorithm returns all of these high-quality risk scores for the user to consider. This method completes within minutes and can be valuable in a broad variety of applications.

        ----

        ## [1291] Graph Learning Assisted Multi-Objective Integer Programming

        **Authors**: *Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang, Abhishek Gupta, Mingyan Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/710aae9186778a91b656e609778f7898-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/710aae9186778a91b656e609778f7898-Abstract-Conference.html)

        **Abstract**:

        Objective-space decomposition algorithms (ODAs) are widely studied for solving multi-objective integer programs. However, they often encounter difficulties in handling scalarized problems, which could cause infeasibility or repetitive nondominated points and thus induce redundant runtime. To mitigate the issue, we present a graph neural network (GNN) based method to learn the reduction rule in the ODA. We formulate the algorithmic procedure of generic ODAs as a Markov decision process, and parameterize the policy (reduction rule) with a novel two-stage GNN to fuse information from variables, constraints and especially objectives for better state representation. We train our model with imitation learning and deploy it on a state-of-the-art ODA. Results show that our method significantly improves the solving efficiency of the ODA. The learned policy generalizes fairly well to larger problems or more objectives, and the proposed GNN outperforms existing ones for integer programming in terms of test and generalization accuracy.

        ----

        ## [1292] Revisiting Sliced Wasserstein on Images: From Vectorization to Convolution

        **Authors**: *Khai Nguyen, Nhat Ho*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/712a2cefb5c84274061e2dd4f6464f31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/712a2cefb5c84274061e2dd4f6464f31-Abstract-Conference.html)

        **Abstract**:

        The conventional sliced Wasserstein is defined between two probability measures that have realizations as \textit{vectors}. When comparing two probability measures over images, practitioners first need to vectorize images and then project them to one-dimensional space by using matrix multiplication between the sample matrix and the projection matrix. After that, the sliced Wasserstein is evaluated by averaging the two corresponding one-dimensional projected probability measures. However, this approach has two limitations. The first limitation is that the spatial structure of images is not captured efficiently by the vectorization step; therefore, the later slicing process becomes harder to gather the discrepancy information. The second limitation is memory inefficiency since each slicing direction is a vector that has the same dimension as the images. To address these limitations, we propose novel slicing methods for sliced Wasserstein between probability measures over images that are based on the convolution operators. We derive \emph{convolution sliced Wasserstein} (CSW) and its variants via incorporating stride, dilation, and non-linear activation function into the convolution operators. We investigate the metricity of CSW as well as its sample complexity, its computational complexity, and its connection to conventional sliced Wasserstein distances. Finally, we demonstrate the favorable performance of CSW over the conventional sliced Wasserstein in comparing probability measures over images and in training deep generative modeling on images.

        ----

        ## [1293] SignRFF: Sign Random Fourier Features

        **Authors**: *Xiaoyun Li, Ping Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/718a3c5cf135894db6e718725f52ef9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/718a3c5cf135894db6e718725f52ef9a-Abstract-Conference.html)

        **Abstract**:

        The industry practice has been moving to embedding based retrieval (EBR). For example, in many applications, the embedding vectors are trained by some form of two-tower models. During serving phase, candidates (embedding vectors) are retrieved according to the rankings of cosine similarities either exhaustively or by  approximate near neighbor (ANN) search algorithms. For those applications, it is natural to apply ``sign random projections'' (SignRP) or variants, on the trained embedding vectors to facilitate efficient data storage and cosine distance computations. SignRP is also one of the standard indexing schemes for conducting approximate near neighbor search. In the literature, SignRP has been   popular and, to an extent, becomes the default method for ``locality sensitive hashing'' (LSH). In this paper, we propose ``sign random Fourier features'' (SignRFF) as an alternative to SignRP. The original method of random Fourier features (RFF) is a standard technique for approximating the Gaussian kernel (as opposed to the linear cosine kernel), in the literature of large-scale machine learning. Basically, RFF applies a simple nonlinear transformation on the samples generated by random projections (RP). Thus, in the pipeline of EBR, it is straightforward to replace SignRP by SignRFF. This paper explains, in a principled manner, why it makes sense to do so. In this paper, a new analytical measure called \textbf{Ranking Efficiency (RE)} is developed, which in retrospect is closely related to the ``two-sample mean'' $t$-test statistic for binomial  variables. RE provides a systematic and unified framework for comparing different LSH methods. We compare our proposed SignRP with SignRP, KLSH (kernel LSH), as well SQ-RFF (which is another 1-bit coding scheme for RFF). According to the RE expression, SignRFF consistently outperforms KLSH (for Gaussian kernel) and SQ-RFF. SignRFF also outperforms SignRP in the relatively high similarity region. The theoretical comparison results are consistent with our empirical findings. In addition, experiments are conducted to compare SignRFF with a wide range of data-dependent and deep learning based hashing methods and show the advantage of SignRFF with a sufficient number of hash bits.

        ----

        ## [1294] The Role of Baselines in Policy Gradient Optimization

        **Authors**: *Jincheng Mei, Wesley Chung, Valentin Thomas, Bo Dai, Csaba Szepesvári, Dale Schuurmans*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html)

        **Abstract**:

        We study the effect of baselines in on-policy stochastic policy gradient optimization, and close the gap between the theory and practice of policy optimization methods. Our first contribution is to show that the \emph{state value} baseline allows on-policy stochastic \emph{natural} policy gradient (NPG) to converge to a globally optimal policy at an $O(1/t)$ rate, which was not previously known. The analysis relies on two novel findings: the expected progress of the NPG update satisfies a stochastic version of the non-uniform \L{}ojasiewicz (N\L{}) inequality, and with probability 1 the state value baseline prevents the optimal action's probability from vanishing, thus ensuring sufficient exploration. Importantly, these results provide a new understanding of the role of baselines in stochastic policy gradient: by showing that the variance of natural policy gradient estimates remains unbounded with or without a baseline, we find that variance reduction \emph{cannot} explain their utility in this setting. Instead, the analysis reveals that the primary effect of the value baseline is to \textbf{reduce the aggressiveness of the updates} rather than their variance. That is, we demonstrate that a finite variance is \emph{not necessary} for almost sure convergence of stochastic NPG, while controlling update aggressiveness is both necessary and sufficient. Additional experimental results verify these theoretical findings.

        ----

        ## [1295] A Rotated Hyperbolic Wrapped Normal Distribution for Hierarchical Representation Learning

        **Authors**: *Seunghyuk Cho, Juyong Lee, Jaesik Park, Dongwoo Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71ad539a57b1fd49b19e5c80070cb8b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71ad539a57b1fd49b19e5c80070cb8b9-Abstract-Conference.html)

        **Abstract**:

        We present a rotated hyperbolic wrapped normal distribution (RoWN), a simple yet effective alteration of a hyperbolic wrapped normal distribution (HWN). The HWN expands the domain of probabilistic modeling from Euclidean to hyperbolic space, where a tree can be embedded with arbitrary low distortion in theory. In this work, we analyze the geometric properties of the diagonal HWN, a standard choice of distribution in probabilistic modeling. The analysis shows that the distribution is inappropriate to represent the data points at the same hierarchy level through their angular distance with the same norm in the Poincar\'e disk model. We then empirically verify the presence of limitations of HWN, and show how RoWN, the proposed distribution, can alleviate the limitations on various hierarchical datasets, including noisy synthetic binary tree, WordNet, and Atari 2600 Breakout. The code is available at https://github.com/ml-postech/RoWN.

        ----

        ## [1296] Private Graph All-Pairwise-Shortest-Path Distance Release with Improved Error Rate

        **Authors**: *Chenglin Fan, Ping Li, Xiaoyun Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71b17f00017da0d73823ccf7fbce2d4f-Abstract-Conference.html)

        **Abstract**:

        Releasing all pairwise shortest path (APSP) distances between vertices on general graphs under weight Differential Privacy (DP) is known as a challenging task. In previous work, to achieve DP with some fixed budget, with high probability the maximal absolute error among all published pairwise distances is roughly O(n) where n is the number of nodes. It was shown that this error could be reduced for some special graphs, which, however, is hard for general graphs. Therefore, whether the approximation error can be reduced to sublinear is posted as an interesting open problem.In this paper, we break the linear barrier on the distance approximation error of previous result, by proposing an algorithm that releases a constructed synthetic graph privately. Computing all pairwise distances on the constructed graph only introduces O(n^{1/2}) error in answering all pairwise shortest path distances for fixed privacy parameter. Our method is based on a novel graph diameter (link length) augmentation via constructing ``shortcuts'' for the paths. By adding a set of shortcut edges to the original graph, we show that any node pair has a shortest path with link length O(n^{1/2}). Then by adding noises with some positive mean to the edge weights, the new graph is differentially private and can be published to answer all pairwise shortest path distances with O(n^{1/2}) approximation error using standard APSP computation. Numerical examples are also provided.Additionally, we also consider the graph with small feedback vertex set number. A feedback vertex set (FVS) of a graph is a set of vertices whose removal leaves a graph without cycles, and the feedback vertex set number of a graph, k, is the size of a smallest feedback vertex set. We propose a DP algorithm with error rate O(k), which improves the error of general graphs provided k=o(n^{1/2}).

        ----

        ## [1297] Multi-Fidelity Best-Arm Identification

        **Authors**: *Riccardo Poiani, Alberto Maria Metelli, Marcello Restelli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71c31ebf577ffdad5f4a74156daad518-Abstract-Conference.html)

        **Abstract**:

        In several real-world applications, a learner has access to multiple environment simulators, each with a different precision (e.g., simulation accuracy) and cost (e.g., computational time). In such a scenario, the learner faces the trade-off between selecting expensive accurate simulators or preferring cheap imprecise ones. We formalize this setting as a multi-fidelity variant of the stochastic best-arm identification problem, where querying the original arm is expensive, but multiple and biased approximations (i.e., fidelities) are available at lower costs. The learner's goal, in this setting, is to sequentially choose which simulator to query in order to minimize the total cost, while guaranteeing to identify the optimal arm with high probability. We first derive a lower bound on the identification cost, assuming that the maximum bias of each fidelity is known to the learner. Then, we propose a novel algorithm, Iterative Imprecise Successive Elimination (IISE), which provably reduces the total cost w.r.t. algorithms that ignore the multi-fidelity structure and whose cost complexity upper bound mimics the structure of the lower bound. Furthermore, we show that the cost complexity of IISE can be further reduced when the agent has access to a more fine-grained knowledge of the error introduced by the approximators.Finally, we numerically validate IISE, showing the benefits of our method in simulated domains.

        ----

        ## [1298] SemiFL: Semi-Supervised Federated Learning for Unlabeled Clients with Alternate Training

        **Authors**: *Enmao Diao, Jie Ding, Vahid Tarokh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71c3451f6cd6a4f82bb822db25cea4fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71c3451f6cd6a4f82bb822db25cea4fd-Abstract-Conference.html)

        **Abstract**:

        Federated Learning allows the training of machine learning models by using the computation and private data resources of many distributed clients. Most existing results on Federated Learning (FL) assume the clients have ground-truth labels. However, in many practical scenarios, clients may be unable to label task-specific data due to a lack of expertise or resource. We propose SemiFL to address the problem of combining communication-efficient FL such as FedAvg with Semi-Supervised Learning (SSL). In SemiFL, clients have completely unlabeled data and can train multiple local epochs to reduce communication costs, while the server has a small amount of labeled data. We provide a theoretical understanding of the success of data augmentation-based SSL methods to illustrate the bottleneck of a vanilla combination of communication-efficient FL with SSL. To address this issue, we propose alternate training to 'fine-tune global model with labeled data' and 'generate pseudo-labels with the global model.' We conduct extensive experiments and demonstrate that our approach significantly improves the performance of a labeled server with unlabeled clients training with multiple local epochs. Moreover, our method outperforms many existing SSFL baselines and performs competitively with the state-of-the-art FL and SSL results.

        ----

        ## [1299] RankFeat: Rank-1 Feature Removal for Out-of-distribution Detection

        **Authors**: *Yue Song, Nicu Sebe, Wei Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71c9eb0913e6c7fda3afd69c914b1a0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71c9eb0913e6c7fda3afd69c914b1a0c-Abstract-Conference.html)

        **Abstract**:

        The task of out-of-distribution (OOD) detection is crucial for deploying machine learning models in real-world settings. In this paper, we observe that the singular value distributions of the in-distribution (ID) and OOD features are quite different: the OOD feature matrix tends to have a larger dominant singular value than the ID feature, and the class predictions of OOD samples are largely determined by it. This observation motivates us to propose RankFeat, a simple yet effective post hoc approach for OOD detection by removing the rank-1 matrix composed of the largest singular value and the associated singular vectors from the high-level feature. RankFeat achieves state-of-the-art performance and reduces the average false positive rate (FPR95) by 17.90% compared with the previous best method. Extensive ablation studies and comprehensive theoretical analyses are presented to support the empirical results.

        ----

        ## [1300] Jump Self-attention: Capturing High-order Statistics in Transformers

        **Authors**: *Haoyi Zhou, Siyang Xiao, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71ec377d5df1fc61ee7770857820519b-Abstract-Conference.html)

        **Abstract**:

        The recent success of Transformer has benefited many real-world applications, with its capability of building long dependency through pairwise dot-products. However, the strong assumption that elements are directly attentive to each other limits the performance of tasks with high-order dependencies such as natural language understanding and Image captioning. To solve such problems, we are the first to define the Jump Self-attention (JAT) to build Transformers. Inspired by the pieces moving of English Draughts, we introduce the spectral convolutional technique to calculate JAT on the dot-product feature map. This technique allows JAT's propagation in each self-attention head and is interchangeable with the canonical self-attention. We further develop the higher-order variants under the multi-hop assumption to increase the generality. Moreover, the proposed architecture is compatible with the pre-trained models. With extensive experiments, we empirically show that our methods significantly increase the performance on ten different tasks.

        ----

        ## [1301] Learning Infinite-Horizon Average-Reward Restless Multi-Action Bandits via Index Awareness

        **Authors**: *Guojun Xiong, Shufan Wang, Jian Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/71f003060ce1e8b6b4856023b67cda5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/71f003060ce1e8b6b4856023b67cda5d-Abstract-Conference.html)

        **Abstract**:

        We consider the online restless bandits with average-reward and multiple actions, where the state of each arm evolves according to a Markov decision process (MDP), and the reward of pulling an arm depends on both the current state of the corresponding MDP and the action taken.  Since finding the optimal control is typically intractable for restless bandits, existing learning algorithms are often computationally expensive or with a regret bound that is exponential in the number of arms and states.  In this paper, we advocate \textit{index-aware reinforcement learning} (RL) solutions to design RL algorithms operating on a much smaller dimensional subspace by exploiting the inherent structure in restless bandits.  Specifically, we first propose novel index policies to address dimensionality concerns, which are provably optimal.  We then leverage the indices to develop two low-complexity index-aware RL algorithms, namely, (i) GM-R2MAB, which has access to a generative model; and (ii) UC-R2MAB, which learns the model using an upper confidence style online exploitation method.  We prove that both algorithms achieve a sub-linear regret that is only polynomial in the number of arms and states.  A key differentiator between our algorithms and existing ones stems from the fact that our RL algorithms contain a novel exploitation that leverages our proposed provably optimal index policies for decision-makings.

        ----

        ## [1302] STNDT: Modeling Neural Population Activity with Spatiotemporal Transformers

        **Authors**: *Trung Le, Eli Shlizerman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72163d1c3c1726f1c29157d06e9e93c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/72163d1c3c1726f1c29157d06e9e93c1-Abstract-Conference.html)

        **Abstract**:

        Modeling neural population dynamics underlying noisy single-trial spiking activities is essential for relating neural observation and behavior. A recent non-recurrent method - Neural Data Transformers (NDT) - has shown great success in capturing neural dynamics with low inference latency without an explicit dynamical model. However, NDT focuses on modeling the temporal evolution of the population activity while neglecting the rich covariation between individual neurons. In this paper we introduce SpatioTemporal Neural Data Transformer (STNDT), an NDT-based architecture that explicitly models responses of individual neurons in the population across time and space to uncover their underlying firing rates. In addition, we propose a contrastive learning loss that works in accordance with mask modeling objective to further improve the predictive performance. We show that our model achieves state-of-the-art performance on ensemble level in estimating neural activities across four neural datasets, demonstrating its capability to capture autonomous and non-autonomous dynamics spanning different cortical regions while being completely agnostic to the specific behaviors at hand. Furthermore, STNDT spatial attention mechanism reveals consistently important subsets of neurons that play a vital role in driving the response of the entire population, providing interpretability and key insights into how the population of neurons performs computation.

        ----

        ## [1303] ProtoVAE: A Trustworthy Self-Explainable Prototypical Variational Model

        **Authors**: *Srishti Gautam, Ahcène Boubekki, Stine Hansen, Suaiba Amina Salahuddin, Robert Jenssen, Marina M.-C. Höhne, Michael Kampffmeyer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/722f3f9298a961d2639eadd3f14a2816-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/722f3f9298a961d2639eadd3f14a2816-Abstract-Conference.html)

        **Abstract**:

        The need for interpretable models has fostered the development of self-explainable classifiers. Prior approaches are either based on multi-stage optimization schemes, impacting the predictive performance of the model, or produce explanations that are not transparent, trustworthy or do not capture the diversity of the data. To address these shortcomings, we propose ProtoVAE, a variational autoencoder-based framework that learns class-specific prototypes in an end-to-end manner and enforces trustworthiness and diversity by regularizing the representation space and introducing an orthonormality constraint. Finally, the model is designed to be transparent by directly incorporating the prototypes into the decision process. Extensive comparisons with previous self-explainable approaches demonstrate the superiority of ProtoVAE, highlighting its ability to generate trustworthy and diverse explanations, while not degrading predictive performance.

        ----

        ## [1304] If Influence Functions are the Answer, Then What is the Question?

        **Authors**: *Juhan Bae, Nathan Ng, Alston Lo, Marzyeh Ghassemi, Roger B. Grosse*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7234e0c36fdbcb23e7bd56b68838999b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7234e0c36fdbcb23e7bd56b68838999b-Abstract-Conference.html)

        **Abstract**:

        Influence functions efficiently estimate the effect of removing a single training data point on a model's learned parameters. While influence estimates align well with leave-one-out retraining for linear models, recent works have shown this alignment is often poor in neural networks. In this work, we investigate the specific factors that cause this discrepancy by decomposing it into five separate terms. We study the contributions of each term on a variety of architectures and datasets and how they vary with factors such as network width and training time. While practical influence function estimates may be a poor match to leave-one-out retraining for nonlinear networks, we show that they are often a good approximation to a different object we term the proximal Bregman response function (PBRF). Since the PBRF can still be used to answer many of the questions motivating influence functions, such as identifying influential or mislabeled examples, our results suggest that current algorithms for influence function estimation give more informative results than previous error analyses would suggest.

        ----

        ## [1305] FACT: Learning Governing Abstractions Behind Integer Sequences

        **Authors**: *Peter Belcak, Ard Kastrati, Flavio Schenker, Roger Wattenhofer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72372ec86dd49238900fc0b68bad63f8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/72372ec86dd49238900fc0b68bad63f8-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Integer sequences are of central importance to the modeling of concepts admitting complete finitary descriptions. We introduce a novel view on the learning of such concepts and lay down a set of benchmarking tasks aimed at conceptual understanding by machine learning models. These tasks indirectly assess model ability to abstract, and challenge them to reason both interpolatively and extrapolatively from the knowledge gained by observing representative examples. To further aid research in knowledge representation and reasoning, we present FACT, the Finitary Abstraction Comprehension Toolkit. The toolkit surrounds a large dataset of integer sequences comprising both organic and synthetic entries, a library for data pre-processing and generation, a set of model performance evaluation tools, and a collection of baseline model implementations, enabling the making of the future advancements with ease.

        ----

        ## [1306] SAPipe: Staleness-Aware Pipeline for Data Parallel DNN Training

        **Authors**: *Yangrui Chen, Cong Xie, Meng Ma, Juncheng Gu, Yanghua Peng, Haibin Lin, Chuan Wu, Yibo Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/725ce5f2b1a8e2e0ac66994e7fefe375-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/725ce5f2b1a8e2e0ac66994e7fefe375-Abstract-Conference.html)

        **Abstract**:

        Data parallelism across multiple machines is widely adopted for accelerating distributed deep learning, but it is hard to achieve linear speedup due to the heavy communication. In this paper, we propose SAPipe, a performant system that pushes the training speed of data parallelism to its fullest extent. By introducing partial staleness, the communication overlaps the computation with minimal staleness in SAPipe. To mitigate additional problems incurred by staleness, SAPipe adopts staleness compensation techniques including weight prediction and delay compensation with provably lower error bounds. Additionally, SAPipe presents an algorithm-system co-design with runtime optimization to minimize system overhead for the staleness training pipeline and staleness compensation. We have implemented SAPipe in the BytePS framework, compatible to both TensorFlow and PyTorch. Our experiments show that SAPipe achieves up to 157% speedups over BytePS (non-stale), and outperforms PipeSGD in accuracy by up to 13.7%.

        ----

        ## [1307] Probing Classifiers are Unreliable for Concept Removal and Detection

        **Authors**: *Abhinav Kumar, Chenhao Tan, Amit Sharma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/725f5e8036cc08adeba4a7c3bcbc6f2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/725f5e8036cc08adeba4a7c3bcbc6f2c-Abstract-Conference.html)

        **Abstract**:

        Neural network models trained on text data have been found to encode undesirable linguistic or sensitive concepts in their representation. Removing such concepts is non-trivial because of a complex relationship between the concept, text input, and the learnt representation. Recent work has proposed post-hoc and adversarial methods to remove such unwanted concepts from a model's representation. Through an extensive theoretical and empirical analysis, we show that these methods can be counter-productive: they are unable to remove the concepts entirely, and in the worst case may end up destroying all task-relevant features. The reason is the methods' reliance on a probing classifier as a proxy for the concept. Even under the most favorable conditions for learning a probing classifier when a concept's relevant features in representation space alone can provide 100% accuracy, we prove that a probing classifier is likely to use non-concept features and thus post-hoc or adversarial methods will fail to remove the concept correctly. These theoretical implications are confirmed by experiments on models trained on synthetic, Multi-NLI, and Twitter datasets. For sensitive applications of concept removal such as fairness, we recommend caution against using these methods and propose a spuriousness metric to gauge the quality of the final classifier.

        ----

        ## [1308] SCL-WC: Cross-Slide Contrastive Learning for Weakly-Supervised Whole-Slide Image Classification

        **Authors**: *Xiyue Wang, Jinxi Xiang, Jun Zhang, Sen Yang, Zhongyi Yang, Ming-Hui Wang, Jing Zhang, Wei Yang, Junzhou Huang, Xiao Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/726204cea3ec27790a644e5b379175e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/726204cea3ec27790a644e5b379175e3-Abstract-Conference.html)

        **Abstract**:

        Weakly-supervised whole-slide image (WSI) classification (WSWC) is a challenging task where a large number of unlabeled patches (instances) exist within each WSI (bag) while only a slide label is given. Despite recent progress for the multiple instance learning (MIL)-based WSI analysis,  the major limitation is that it usually focuses on the easy-to-distinguish diagnosis-positive regions while ignoring positives that occupy a small ratio in the entire WSI. To obtain more discriminative features, we propose a novel weakly-supervised classification method based on cross-slide contrastive learning (called SCL-WC), which depends on task-agnostic self-supervised feature pre-extraction and task-specific weakly-supervised feature refinement and aggregation for WSI-level prediction. To enable both intra-WSI and inter-WSI information interaction, we propose a positive-negative-aware module (PNM) and a weakly-supervised cross-slide contrastive learning (WSCL) module, respectively. The WSCL aims to pull WSIs with the same disease types closer and push different WSIs away. The PNM aims to facilitate the separation of tumor-like patches and normal ones within each WSI. Extensive experiments demonstrate state-of-the-art performance of our method in three different classification tasks (e.g., over 2% of AUC in Camelyon16, 5% of F1 score in BRACS, and 3% of AUC in DiagSet). Our method also shows superior flexibility and scalability in weakly-supervised localization and semi-supervised classification experiments (e.g., first place in the BRIGHT challenge). Our code will be available at https://github.com/Xiyue-Wang/SCL-WC.

        ----

        ## [1309] Reproducibility in Optimization: Theoretical Framework and Limits

        **Authors**: *Kwangjun Ahn, Prateek Jain, Ziwei Ji, Satyen Kale, Praneeth Netrapalli, Gil I. Shamir*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html)

        **Abstract**:

        We initiate a formal study of reproducibility in optimization. We define a quantitative measure of reproducibility of optimization procedures in the face of noisy or error-prone operations such as inexact or stochastic gradient computations or inexact initialization. We then analyze several convex optimization settings of interest such as smooth, non-smooth, and strongly-convex objective functions and establish tight bounds on the limits of reproducibility in each setting. Our analysis reveals a fundamental trade-off between computation and reproducibility: more computation is necessary (and sufficient) for better reproducibility.

        ----

        ## [1310] Hierarchical classification at multiple operating points

        **Authors**: *Jack Valmadre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/727855c31df8821fd18d41c23daebf10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/727855c31df8821fd18d41c23daebf10-Abstract-Conference.html)

        **Abstract**:

        Many classification problems consider classes that form a hierarchy. Classifiers that are aware of this hierarchy may be able to make confident predictions at a coarse level despite being uncertain at the fine-grained level. While it is generally possible to vary the granularity of predictions using a threshold at inference time, most contemporary work considers only leaf-node prediction, and almost no prior work has compared methods at multiple operating points. We present an efficient algorithm to produce operating characteristic curves for any method that assigns a score to every class in the hierarchy. Applying this technique to evaluate existing methods reveals that top-down classifiers are dominated by a naive flat softmax classifier across the entire operating range. We further propose two novel loss functions and show that a soft variant of the structured hinge loss is able to significantly outperform the flat baseline. Finally, we investigate the poor accuracy of top-down classifiers and demonstrate that they perform relatively well on unseen classes.

        ----

        ## [1311] Kernel Multimodal Continuous Attention

        **Authors**: *Alexander Moreno, Zhenke Wu, Supriya Nagesh, Walter H. Dempsey, James M. Rehg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/727a5a5c77be15d053b47b7c391800c2-Abstract-Conference.html)

        **Abstract**:

        Attention mechanisms take an expectation of a data representation with respect to probability weights. Recently, (Martins et al. 2020, 2021) proposed continuous attention mechanisms, focusing on unimodal attention densities from the exponential and deformed exponential families: the latter has sparse support. (Farinhas et al 2021) extended this to to multimodality via Gaussian mixture attention densities. In this paper, we extend this to kernel exponential families (Canu and Smola 2006) and our new sparse counterpart, kernel deformed exponential families. Theoretically, we show new existence results for both kernel exponential and deformed exponential families, and that the deformed case has similar approximation capabilities to kernel exponential families. Lacking closed form expressions for the context vector, we use numerical integration: we show exponential convergence for both kernel exponential and deformed exponential families. Experiments show that kernel continuous attention often outperforms unimodal continuous attention, and the sparse variant tends to highlight peaks of time series.

        ----

        ## [1312] VisCo Grids: Surface Reconstruction with Viscosity and Coarea Grids

        **Authors**: *Albert Pumarola, Artsiom Sanakoyeu, Lior Yariv, Ali K. Thabet, Yaron Lipman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7283957fa5aa3bd79870ac8753f2f742-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7283957fa5aa3bd79870ac8753f2f742-Abstract-Conference.html)

        **Abstract**:

        Surface reconstruction has been seeing a lot of progress lately by utilizing Implicit Neural Representations (INRs). Despite their success, INRs often introduce hard to control inductive bias (i.e., the solution surface can exhibit unexplainable behaviours), have costly inference, and are slow to train.  The goal of this work is to show that replacing neural networks with simple grid functions, along with two novel geometric priors achieve comparable results to INRs, with instant inference, and improved training times. To that end we introduce VisCo Grids: a grid-based surface reconstruction method incorporating Viscosity and Coarea priors. Intuitively, the Viscosity prior replaces the smoothness inductive bias of INRs, while the Coarea favors a minimal area solution. Experimenting with VisCo Grids on a standard reconstruction baseline provided comparable results to the best performing INRs on this dataset.

        ----

        ## [1313] Exact Shape Correspondence via 2D graph convolution

        **Authors**: *Barakeel Fanseu Kamhoua, Lin Zhang, Yongqiang Chen, Han Yang, Kaili Ma, Bo Han, Bo Li, James Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72d9a23e3895b5670e650c2e742065c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/72d9a23e3895b5670e650c2e742065c9-Abstract-Conference.html)

        **Abstract**:

        For exact 3D shape correspondence (matching or alignment), i.e., the task of matching each point on a shape to its exact corresponding point on the other shape (or to be more specific, matching at geodesic error 0), most existing methods do not perform well due to two main problems. First, on nearly-isometric shapes (i.e., low noise levels), most existing methods use the eigen-vectors (eigen-functions) of the Laplace Beltrami Operator (LBO) or other shape descriptors to update an initialized correspondence which is not exact, leading to an accumulation of update errors. Thus, though the final correspondence may generally be smooth, it is generally inexact. Second, on non-isometric shapes (noisy shapes), existing methods are generally not robust to noise as they usually assume near-isometry. In addition, existing methods that attempt to address the non-isometric shape problem (e.g., GRAMPA) are generally computationally expensive and do not generalise to nearly-isometric shapes. To address these two problems, we propose a 2D graph convolution-based framework called 2D-GEM. 2D-GEM is robust to noise on non-isometric shapes and with a few additional constraints, it also addresses the errors in the update on nearly-isometric shapes. We demonstrate the effectiveness of 2D-GEM by achieving a high accuracy of 90.5$\%$ at geodesic error 0 on the non-isometric benchmark SHREC16, i.e., TOPKIDS (while being much faster than GRAMPA), and on nearly-isometric benchmarks by achieving a high accuracy of 92.5$\%$ on TOSCA and 84.9$\%$ on SCAPE at geodesic error 0.

        ----

        ## [1314] Posterior Matching for Arbitrary Conditioning

        **Authors**: *Ryan R. Strauss, Junier B. Oliva*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72dad0866fa5b0ef20cec94b8bd5763a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/72dad0866fa5b0ef20cec94b8bd5763a-Abstract-Conference.html)

        **Abstract**:

        Arbitrary conditioning is an important problem in unsupervised learning, where we seek to model the conditional densities $p(\mathbf{x}_u \mid \mathbf{x}_o)$ that underly some data, for all possible non-intersecting subsets $o, u \subset \{1, \dots , d\}$. However, the vast majority of density estimation only focuses on modeling the joint distribution $p(\mathbf{x})$, in which important conditional dependencies between features are opaque. We propose a simple and general framework, coined Posterior Matching, that enables Variational Autoencoders (VAEs) to perform arbitrary conditioning, without modification to the VAE itself. Posterior Matching applies to the numerous existing VAE-based approaches to joint density estimation, thereby circumventing the specialized models required by previous approaches to arbitrary conditioning. We find that Posterior Matching is comparable or superior to current state-of-the-art methods for a variety of tasks with an assortment of VAEs (e.g.~discrete, hierarchical, VaDE).

        ----

        ## [1315] CARD: Classification and Regression Diffusion Models

        **Authors**: *Xizewen Han, Huangjie Zheng, Mingyuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/72dad95a24fae750f8ab1cb3dab5e58d-Abstract-Conference.html)

        **Abstract**:

        Learning the distribution of a continuous or categorical response variable y given its covariates x is a fundamental problem in statistics and machine learning. Deep neural network-based supervised learning algorithms have made great progress in predicting the mean of y given x, but they are often criticized for their ability to accurately capture the uncertainty of their predictions. In this paper, we introduce classification and regression diffusion (CARD) models, which combine a denoising diffusion-based conditional generative model and a pre-trained conditional mean estimator, to accurately predict the distribution of y given x.  We demonstrate the outstanding ability of CARD in conditional distribution prediction with both toy examples and real-world datasets, the experimental results on which show that CARD, in general, outperforms state-of-the-art methods, including Bayesian neural network-based one, designed for uncertainty estimation, especially when the conditional distribution of y given x is multi-modal. In addition, we utilize the stochastic nature of the generative model outputs to obtain a finer granularity in model confidence assessment at the instance level for classification tasks.

        ----

        ## [1316] What Can the Neural Tangent Kernel Tell Us About Adversarial Robustness?

        **Authors**: *Nikolaos Tsilivis, Julia Kempe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/72f9c316440c384a95c88022fd78f066-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/72f9c316440c384a95c88022fd78f066-Abstract-Conference.html)

        **Abstract**:

        The adversarial vulnerability of neural nets, and subsequent techniques to create robust models have attracted significant attention; yet we still lack a full understanding of this phenomenon. Here, we study adversarial examples of trained neural networks through analytical tools afforded by recent theory advances connecting neural networks and kernel methods, namely the Neural Tangent Kernel (NTK), following a growing body of work that leverages the NTK approximation to successfully analyze important deep learning phenomena and design algorithms for new applications. We show how NTKs allow to generate adversarial examples in a training-free'' fashion, and demonstrate that they transfer to fool their finite-width neural net counterparts in thelazy'' regime. We leverage this connection to provide an alternative view on robust and non-robust features, which have been suggested to underlie the adversarial brittleness of neural nets. Specifically, we define and study features induced by the eigendecomposition of the kernel to better understand the role of robust and non-robust features, the reliance on both for standard classification and the robustness-accuracy trade-off. We find that such features are surprisingly consistent across architectures, and that robust features tend to correspond to the largest eigenvalues of the model, and thus are learned early during training. Our framework allows us to identify and visualize non-robust yet useful features. Finally, we shed light on the robustness mechanism underlying adversarial training of neural nets used in practice: quantifying the evolution of the associated empirical NTK, we demonstrate that its dynamics falls much earlier into the ``lazy'' regime and manifests a much stronger form of the well known bias to prioritize learning features within the top eigenspaces of the kernel, compared to standard training.

        ----

        ## [1317] Benefits of Permutation-Equivariance in Auction Mechanisms

        **Authors**: *Tian Qin, Fengxiang He, Dingfeng Shi, Wenbing Huang, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/730d61b4d9ff794a028fa3a25b9b891d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/730d61b4d9ff794a028fa3a25b9b891d-Abstract-Conference.html)

        **Abstract**:

        Designing an incentive-compatible auction mechanism that maximizes the auctioneer's revenue while minimizes the bidders’ ex-post regret is an important yet intricate problem in economics. Remarkable progress has been achieved through learning the optimal auction mechanism by neural networks. In this paper, we consider the popular additive valuation and symmetric valuation setting; i.e., the valuation for a set of items is defined as the sum of all items’ valuations in the set, and the valuation distribution is invariant when the bidders and/or the items are permutated. We prove that permutation-equivariant neural networks have significant advantages: the permutation-equivariance decreases the expected ex-post regret, improves the model generalizability, while maintains the expected revenue invariant. This implies that the permutation-equivariance helps approach the theoretically optimal dominant strategy incentive compatible condition, and reduces the required sample complexity for desired generalization. Extensive experiments fully support our theory. To our best knowledge, this is the first work towards understanding the benefits of permutation-equivariance in auction mechanisms.

        ----

        ## [1318] MoCoDA: Model-based Counterfactual Data Augmentation

        **Authors**: *Silviu Pitis, Elliot Creager, Ajay Mandlekar, Animesh Garg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7314e20a73542bbfff25030d1185ce88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7314e20a73542bbfff25030d1185ce88-Abstract-Conference.html)

        **Abstract**:

        The number of states in a dynamic process is exponential in the number of objects, making  reinforcement learning (RL) difficult in complex, multi-object domains. For agents to scale to the real world, they will need to react to and reason about unseen combinations of objects. We argue that the ability to recognize and use local factorization in transition dynamics is a key element in unlocking the power of multi-object reasoning. To this end, we show that (1) known local structure in the environment transitions is sufficient for an exponential reduction in the sample complexity of training a dynamics model, and (2) a locally factored dynamics model provably generalizes out-of-distribution to unseen states and actions. Knowing the local structure also allows us to predict which unseen states and actions this dynamics model will generalize to. We propose to leverage these observations in a novel Model-based Counterfactual Data Augmentation (MoCoDA) framework. MoCoDA applies a learned locally factored dynamics model to an augmented distribution of states and actions to generate counterfactual transitions for RL. MoCoDA works with a broader set of local structures than prior work and allows for direct control over the augmented training distribution. We show that MoCoDA enables RL agents to learn policies that generalize to unseen states and actions. We use MoCoDA to train an offline RL agent to solve an out-of-distribution robotics manipulation task on which standard offline RL algorithms fail.

        ----

        ## [1319] SkinCon: A skin disease dataset densely annotated by domain experts for fine-grained debugging and analysis

        **Authors**: *Roxana Daneshjou, Mert Yüksekgönül, Zhuo Ran Cai, Roberto A. Novoa, James Y. Zou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/7318b51b52078e3af28197e725f5068a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        For the deployment of artificial intelligence (AI) in high risk settings, such as healthcare, methods that provide interpretability/explainability or allow fine-grained error analysis are critical. Many recent methods for interpretability/explainability and fine-grained error analysis use concepts, which are meta-labels which are semantically meaningful to humans.  However, there are only a few datasets that include concept-level meta-labels and most of these meta-labels are relevant for natural images that do not require domain expertise. Previous densely annotated datasets in medicine focused on meta-labels that are relevant to a single disease such as osteoarthritis or melanoma. In dermatology, skin disease is described using an established clinical lexicon that allow clinicians to describe physical exam findings to one another. To provide the first medical dataset densely annotated by domain experts to provide annotations useful across multiple disease processes, we developed SkinCon: a skin disease dataset densely annotated by dermatologists. SkinCon includes 3230 images from the Fitzpatrick 17k skin disease dataset densely annotated with 48 clinical concepts, 22 of which have at least 50 images representing the concept. The concepts used were chosen by two dermatologists considering the clinical descriptor terms used to describe skin lesions. Examples include "plaque", "scale", and "erosion". These same concepts were also used to label 656 skin disease images from the Diverse Dermatology Images dataset, providing an additional external dataset with diverse skin tone representations. We review the potential applications for the SkinCon dataset, such as probing models, concept-based explanations, concept bottlenecks, error analysis, and slice discovery. Furthermore, we use SkinCon to demonstrate two of these use cases: debugging mistakes of an existing dermatology AI model with concepts and developing interpretable models with post-hoc concept bottleneck models.

        ----

        ## [1320] A permutation-free kernel two-sample test

        **Authors**: *Shubhanshu Shekhar, Ilmun Kim, Aaditya Ramdas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/731b952bdc833485eb72f458cdd5c489-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/731b952bdc833485eb72f458cdd5c489-Abstract-Conference.html)

        **Abstract**:

        The kernel Maximum Mean Discrepancy~(MMD) is a popular multivariate distance metric between distributions. The usual kernel-MMD test statistic (for two-sample testing) is a degenerate U-statistic under the null, and thus it has an intractable limiting null distribution. Hence, the standard approach for designing a level-$(1-\alpha)$ two-sample test using this statistic involves selecting the rejection threshold as the $(1-\alpha)$-quantile of the permutation distribution. The resulting nonparametric test has finite-sample validity but suffers from large computational cost, since the test statistic must be recomputed for every permutation. We propose the cross-MMD, a new quadratic time MMD test statistic based on sample-splitting and studentization. We prove that under mild assumptions, it has a standard normal limiting distribution under the null. Importantly, we also show that the resulting test is consistent against any fixed alternative, and when using the Gaussian kernel, it has minimax rate-optimal power against local alternatives.  For large sample-sizes, our new cross-MMD provides a significant speedup over the MMD, for only a slight loss in power.

        ----

        ## [1321] Simple Unsupervised Object-Centric Learning for Complex and Naturalistic Videos

        **Authors**: *Gautam Singh, Yi-Fu Wu, Sungjin Ahn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/735c847a07bf6dd4486ca1ace242a88c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/735c847a07bf6dd4486ca1ace242a88c-Abstract-Conference.html)

        **Abstract**:

        Unsupervised object-centric learning aims to represent the modular, compositional, and causal structure of a scene as a set of object representations and thereby promises to resolve many critical limitations of traditional single-vector representations such as poor systematic generalization. Although there have been many remarkable advances in recent years, one of the most critical problems in this direction has been that previous methods work only with simple and synthetic scenes but not with complex and naturalistic images or videos. In this paper, we propose STEVE, an unsupervised model for object-centric learning in videos. Our proposed model makes a significant advancement by demonstrating its effectiveness on various complex and naturalistic videos unprecedented in this line of research. Interestingly, this is achieved by neither adding complexity to the model architecture nor introducing a new objective or weak supervision. Rather, it is achieved by a surprisingly simple architecture that uses a transformer-based image decoder conditioned on slots and the learning objective is simply to reconstruct the observation. Our experiment results on various complex and naturalistic videos show significant improvements compared to the previous state-of-the-art.

        ----

        ## [1322] Communication-efficient distributed eigenspace estimation with arbitrary node failures

        **Authors**: *Vasileios Charisopoulos, Anil Damle*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/73b038fffc99ae11056e936f9a299508-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/73b038fffc99ae11056e936f9a299508-Abstract-Conference.html)

        **Abstract**:

        We develop an eigenspace estimation algorithm for distributed environments with arbitrary node failures, where a subset of computing nodes can return structurally valid but otherwise arbitrarily chosen responses. Notably, this setting encompasses several important scenarios that arise in distributed computing and data-collection environments such as silent/soft errors, outliers or corrupted data at certain nodes, and adversarial responses. Our estimator builds upon and matches the performance of a recently proposed non-robust estimator up to an additive $\tilde{O}(\sigma \sqrt{\alpha})$ error, where $\sigma^2$ is the variance of the existing estimator and $\alpha$ is the fraction of corrupted nodes.

        ----

        ## [1323] On Uncertainty, Tempering, and Data Augmentation in Bayesian Classification

        **Authors**: *Sanyam Kapoor, Wesley J. Maddox, Pavel Izmailov, Andrew Gordon Wilson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/73e018a0123b35a3e64269526f9096c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/73e018a0123b35a3e64269526f9096c9-Abstract-Conference.html)

        **Abstract**:

        Aleatoric uncertainty captures the inherent randomness of the data, such as measurement noise. In Bayesian regression, we often use a Gaussian observation model, where we control the level of aleatoric uncertainty with a noise variance parameter. By contrast, for Bayesian classification we use a categorical distribution with no mechanism to represent our beliefs about aleatoric uncertainty. Our work shows that explicitly accounting for aleatoric uncertainty significantly improves the performance of Bayesian neural networks. We note that many standard benchmarks, such as CIFAR-10, have essentially no aleatoric uncertainty. Moreover, we show that data augmentation in approximate inference softens the likelihood, leading to underconfidence and misrepresenting our beliefs about aleatoric uncertainty. Accordingly, we find that a cold posterior, tempered by a power greater than one, often more honestly reflects our beliefs about aleatoric uncertainty than no tempering --- providing an explicit link between data augmentation and cold posteriors. We further show that we can match or exceed the performance of posterior tempering by using a Dirichlet observation model, where we explicitly control the level of aleatoric uncertainty, without any need for tempering.

        ----

        ## [1324] Optimal Binary Classification Beyond Accuracy

        **Authors**: *Shashank Singh, Justin T. Khim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7412b6288499d68769430839b98ff1c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7412b6288499d68769430839b98ff1c1-Abstract-Conference.html)

        **Abstract**:

        The vast majority of statistical theory on binary classification characterizes performance in terms of accuracy. However, accuracy is known in many cases to poorly reflect the practical consequences of classification error, most famously in imbalanced binary classification, where data are dominated by samples from one of two classes. The first part of this paper derives a novel generalization of the Bayes-optimal classifier from accuracy to any performance metric computed from the confusion matrix. Specifically, this result (a) demonstrates that stochastic classifiers sometimes outperform the best possible deterministic classifier and (b) removes an empirically unverifiable absolute continuity assumption that is poorly understood but pervades existing results. We then demonstrate how to use this generalized Bayes classifier to obtain regret bounds in terms of the error of estimating regression functions under uniform loss. Finally, we use these results to develop some of the first finite-sample statistical guarantees specific to imbalanced binary classification. Specifically, we demonstrate that optimal classification performance depends on properties of class imbalance, such as a novel notion called Uniform Class Imbalance, that have not previously been formalized. We further illustrate these contributions numerically in the case of $k$-nearest neighbor classification.

        ----

        ## [1325] Information-Theoretic GAN Compression with Variational Energy-based Model

        **Authors**: *Minsoo Kang, Hyewon Yoo, Eunhee Kang, Sehwan Ki, Hyong-Euk Lee, Bohyung Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7416573f05b50beac6d0aef3abc805c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7416573f05b50beac6d0aef3abc805c0-Abstract-Conference.html)

        **Abstract**:

        We propose an information-theoretic knowledge distillation approach for the compression of generative adversarial networks, which aims to maximize the mutual information between teacher and student networks via a variational optimization based on an energy-based model. Because the direct computation of the mutual information in continuous domains is intractable, our approach alternatively optimizes the student network by maximizing the variational lower bound of the mutual information. To achieve a tight lower bound, we introduce an energy-based model relying on a deep neural network to represent a flexible variational distribution that deals with high-dimensional images and consider spatial dependencies between pixels, effectively. Since the proposed method is a generic optimization algorithm, it can be conveniently incorporated into arbitrary generative adversarial networks and even dense prediction networks, e.g., image enhancement models. We demonstrate that the proposed algorithm achieves outstanding performance in model compression of generative adversarial networks consistently when combined with several existing models.

        ----

        ## [1326] On Learning and Refutation in Noninteractive Local Differential Privacy

        **Authors**: *Alexander Edmonds, Aleksandar Nikolov, Toniann Pitassi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7418d4cfa9c095d8bd06af7deb95ad54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7418d4cfa9c095d8bd06af7deb95ad54-Abstract-Conference.html)

        **Abstract**:

        We study two basic statistical tasks in  non-interactive local differential privacy (LDP): *learning* and *refutation*: learning requires finding a concept that best fits an unknown target function (from labelled samples drawn from a distribution), whereas  refutation requires distinguishing between data distributions that are well-correlated with some concept in the class, versus distributions where the labels are random. Our main result is a complete characterization of the sample complexity of agnostic PAC learning for non-interactive LDP protocols. We show that the optimal sample complexity for any concept class is captured by the approximate $\gamma_2$ norm of a natural matrix associated with the class. Combined with previous work, this gives an *equivalence* between agnostic learning and refutation in the agnostic setting.

        ----

        ## [1327] Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters

        **Authors**: *Seyed Kamyar Seyed Ghasemipour, Shixiang Shane Gu, Ofir Nachum*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7423902b5534e2b267438c85444a54b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7423902b5534e2b267438c85444a54b1-Abstract-Conference.html)

        **Abstract**:

        Motivated by the success of ensembles for uncertainty estimation in supervised learning, we take a renewed look at how ensembles of $Q$-functions can be leveraged as the primary source of pessimism for offline reinforcement learning (RL). We begin by identifying a critical flaw in a popular algorithmic choice used by many ensemble-based RL algorithms, namely the use of shared pessimistic target values when computing each ensemble member's Bellman error. Through theoretical analyses and construction of examples in toy MDPs, we demonstrate that shared pessimistic targets can paradoxically lead to value estimates that are effectively optimistic. Given this result, we propose MSG, a practical offline RL algorithm that trains an ensemble of $Q$-functions with independently computed targets based on completely separate networks, and optimizes a policy with respect to the lower confidence bound of predicted action values. Our experiments on the popular D4RL and RL Unplugged offline RL benchmarks demonstrate that on challenging domains such as antmazes, MSG with deep ensembles surpasses highly well-tuned state-of-the-art methods by a wide margin. Additionally, through ablations on benchmarks domains, we verify the critical significance of using independently trained $Q$-functions, and study the role of ensemble size. Finally, as using separate networks per ensemble member can become computationally costly with larger neural network architectures, we investigate whether efficient ensemble approximations developed for supervised learning can be similarly effective, and demonstrate that they do not match the performance and robustness of MSG with separate networks, highlighting the need for new efforts into efficient uncertainty estimation directed at RL.

        ----

        ## [1328] Private Synthetic Data for Multitask Learning and Marginal Queries

        **Authors**: *Giuseppe Vietri, Cédric Archambeau, Sergül Aydöre, William Brown, Michael Kearns, Aaron Roth, Amaresh Ankit Siva, Shuai Tang, Zhiwei Steven Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7428310c0f97f1c6bb2ef1be99c1ec2a-Abstract-Conference.html)

        **Abstract**:

        We provide a differentially private algorithm for producing  synthetic data simultaneously useful for multiple tasks: marginal queries and multitask machine learning (ML). A key innovation in our algorithm is the ability to directly handle numerical features, in contrast to a number of related prior approaches which require numerical features to be first converted into {high cardinality} categorical features via {a binning strategy}. Higher binning granularity is required for better accuracy, but this negatively impacts scalability. Eliminating the need for binning allows us to produce synthetic data preserving large numbers of statistical queries such as marginals on numerical features, and class conditional linear threshold queries. Preserving the latter means that the fraction of points of each class label above a particular half-space is roughly the same in both the real and synthetic data. This is the property that is needed to train a linear classifier in a multitask setting. Our algorithm also allows us to produce high quality synthetic data for mixed marginal queries, that combine both categorical  and numerical features. Our method consistently runs 2-5x faster than the best comparable techniques, and provides significant accuracy improvements in both marginal queries and linear prediction tasks for mixed-type datasets.

        ----

        ## [1329] Sample-Efficient Reinforcement Learning of Partially Observable Markov Games

        **Authors**: *Qinghua Liu, Csaba Szepesvári, Chi Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/743459dae9b2c5d2904e5432d5298128-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/743459dae9b2c5d2904e5432d5298128-Abstract-Conference.html)

        **Abstract**:

        This paper considers the challenging tasks of Multi-Agent Reinforcement Learning (MARL) under partial observability, where each agent only sees her own individual observations and actions that reveal incomplete information about the underlying state of system. This paper studies these tasks under the general model of multiplayer general-sum Partially Observable Markov Games (POMGs), which is significantly larger than the standard model of Imperfect Information Extensive-Form Games (IIEFGs). We identify a rich subclass of POMGs---weakly revealing POMGs---in which sample-efficient learning is tractable. In the self-play setting, we prove that a simple algorithm combining optimism and Maximum Likelihood Estimation (MLE) is sufficient to find approximate Nash equilibria, correlated equilibria, as well as coarse correlated equilibria of weakly revealing POMGs, in a polynomial number of samples when the number of agents is small. In the setting of playing against adversarial opponents, we show that a variant of our optimistic MLE algorithm is capable of achieving sublinear regret when being compared against the optimal maximin policies. To our best knowledge, this work provides the first line of sample-efficient results for learning POMGs.

        ----

        ## [1330] Advancing Model Pruning via Bi-level Optimization

        **Authors**: *Yihua Zhang, Yuguang Yao, Parikshit Ram, Pu Zhao, Tianlong Chen, Mingyi Hong, Yanzhi Wang, Sijia Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/749252feedd44f7f10d47ec1d674a2f8-Abstract-Conference.html)

        **Abstract**:

        The deployment constraints in practical applications necessitate the pruning of large-scale deep learning models, i.e., promoting their weight sparsity. As illustrated by the Lottery Ticket Hypothesis (LTH), pruning also has the potential of improving their generalization ability. At the core of LTH, iterative magnitude pruning (IMP) is the predominant pruning method to successfully find ‘winning tickets’. Yet, the computation cost of IMP grows prohibitively as the targeted pruning ratio increases. To reduce the computation overhead, various efficient ‘one-shot’ pruning methods have been developed, but these schemes are usually unable to find winning tickets as good as IMP. This raises the question of how to close the gap between pruning accuracy and pruning efficiency? To tackle it, we pursue the algorithmic advancement of model pruning. Specifically, we formulate the pruning problem from a fresh and novel viewpoint, bi-level optimization (BLO). We show that the BLO interpretation provides a technically-grounded optimization base for an efficient implementation of the pruning-retraining learning paradigm used in IMP. We also show that the proposed bi-level optimization-oriented pruning method (termed BiP) is a special class of BLO problems with a bi-linear problem structure. By leveraging such bi-linearity, we theoretically show that BiP can be solved as easily as first-order optimization, thus inheriting the computation efficiency. Through extensive experiments on both structured and unstructured pruning with 5 model architectures and 4 data sets, we demonstrate that BiP can find better winning tickets than IMP in most cases, and is computationally as efficient as the one-shot pruning schemes, demonstrating $2-7\times$ speedup over IMP for the same level of model accuracy and sparsity.

        ----

        ## [1331] Optimal Positive Generation via Latent Transformation for Contrastive Learning

        **Authors**: *Yinqi Li, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/74a31a3b862eb7f01defbbed8e5f0c69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/74a31a3b862eb7f01defbbed8e5f0c69-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning, which learns to contrast positive with negative pairs of samples, has been popular for self-supervised visual representation learning. Although great effort has been made to design proper positive pairs through data augmentation, few works attempt to generate optimal positives for each instance. Inspired by semantic consistency and computational advantage in latent space of pretrained generative models, this paper proposes to learn instance-specific latent transformations to generate Contrastive Optimal Positives (COP-Gen) for self-supervised contrastive learning. Specifically, we formulate COP-Gen as an instance-specific latent space navigator which minimizes the mutual information between the generated positive pair subject to the semantic consistency constraint. Theoretically, the learned latent transformation creates optimal positives for contrastive learning, which removes as much nuisance information as possible while preserving the semantics. Empirically, using generated positives by COP-Gen consistently outperforms other latent transformation methods and even real-image-based methods in self-supervised contrastive learning.

        ----

        ## [1332] MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge

        **Authors**: *Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, Anima Anandkumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5cc5910f64938cac4526a90-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Autonomous agents have made great strides in specialist domains like Atari games and Go. However, they typically learn tabula rasa in isolated environments with limited and manually conceived objectives, thus failing to generalize across a wide spectrum of tasks and capabilities. Inspired by how humans continually learn and adapt in the open world, we advocate a trinity of ingredients for building generalist agents: 1) an environment that supports a multitude of tasks and goals, 2) a large-scale database of multimodal knowledge, and 3) a flexible and scalable agent architecture. We introduce MineDojo, a new framework built on the popular Minecraft game that features a simulation suite with thousands of diverse open-ended tasks and an internet-scale knowledge base with Minecraft videos, tutorials, wiki pages, and forum discussions. Using MineDojo's data, we propose a novel agent learning algorithm that leverages large pre-trained video-language models as a learned reward function. Our agent is able to solve a variety of open-ended tasks specified in free-form language without any manually designed dense shaping reward. We open-source the simulation suite, knowledge bases, algorithm implementation, and pretrained models (https://minedojo.org) to promote research towards the goal of generally capable embodied agents.

        ----

        ## [1333] The price of unfairness in linear bandits with biased feedback

        **Authors**: *Solenne Gaucher, Alexandra Carpentier, Christophe Giraud*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/74bb24dca8334adce292883b4b651eda-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the problem of fair sequential decision making with biased linear bandit feedback. At each round, a player selects an action described by a covariate and by a sensitive attribute. The perceived reward is a linear combination of the covariates of the chosen action, but the player only observes a biased evaluation of this reward, depending on the sensitive attribute. To characterize the difficulty of this problem, we design a phased elimination algorithm that corrects the unfair evaluations, and establish upper bounds on its regret. We show that the worst-case regret is smaller than $\mathcal{O}(\kappa_* ^{1/3}\log(T)^{1/3}T^{2/3})$, where $\kappa_*$ is an explicit geometrical constant characterizing the difficulty of bias estimation. We prove lower bounds on the worst-case regret for some sets of actions showing that this rate is tight up to a possible sub-logarithmic factor. We also derive gap-dependent upper bounds on the regret, and  matching lower bounds for some problem instance. Interestingly, these results reveal a transition between a regime where the problem is as difficult as its unbiased counterpart, and a regime where it can be much harder.

        ----

        ## [1334] DeepFoids: Adaptive Bio-Inspired Fish Simulation with Deep Reinforcement Learning

        **Authors**: *Yuko Ishiwaka, Xiao S. Zeng, Shun Ogawa, Donovan Westwater, Tadayuki Tone, Masaki Nakada*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/74fa9e6bc36aa567fe7cf002b733a30d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/74fa9e6bc36aa567fe7cf002b733a30d-Abstract-Conference.html)

        **Abstract**:

        Our goal is to synthesize realistic underwater scenes with various fish species in different fish cages, which can be utilized to train computer vision models to automate fish counting and sizing tasks. It is a challenging problem to prepare a sufficiently diverse labeled dataset of images from aquatic environments. We solve this challenge by introducing an adaptive bio-inspired fish simulation. The behavior of caged fish changes based on the species, size and number of fish, and the size and shape of the cage, among other variables. However, a method to autonomously achieve schooling behavior for caged fish did not exist. In this paper, we propose a method for achieving schooling behavior for any given combination of variables, using multi-agent deep reinforcement learning (DRL) in various fish cages in arbitrary environments. Furthermore, to visually reproduce the underwater scene in different locations and seasons, we incorporate a physically-based underwater simulation.

        ----

        ## [1335] Truncated Matrix Power Iteration for Differentiable DAG Learning

        **Authors**: *Zhen Zhang, Ignavier Ng, Dong Gong, Yuhang Liu, Ehsan Abbasnejad, Mingming Gong, Kun Zhang, Javen Qinfeng Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html)

        **Abstract**:

        Recovering underlying Directed Acyclic Graph (DAG) structures from observational data is highly challenging due to the combinatorial nature of the DAG-constrained optimization problem. Recently, DAG learning has been cast as a continuous optimization problem by characterizing the DAG constraint as a smooth equality one, generally based on polynomials over adjacency matrices. Existing methods place very small coefficients on high-order polynomial terms for stabilization, since they argue that large coefficients on the higher-order terms are harmful due to numeric exploding. On the contrary, we discover that large coefficients on higher-order terms are beneficial for DAG learning, when the spectral radiuses of the adjacency matrices are small, and that larger coefficients for higher-order terms can approximate the DAG constraints much better than the small counterparts. Based on this, we propose a novel DAG learning method with efficient truncated matrix power iteration to approximate geometric series based DAG constraints. Empirically, our DAG learning method outperforms the previous state-of-the-arts in various settings, often by a factor of $3$ or more in terms of structural Hamming distance.

        ----

        ## [1336] Learning Debiased Classifier with Biased Committee

        **Authors**: *Nayeong Kim, Sehyun Hwang, Sungsoo Ahn, Jaesik Park, Suha Kwak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/750046157471c56235a781f2eff6e226-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/750046157471c56235a781f2eff6e226-Abstract-Conference.html)

        **Abstract**:

        Neural networks are prone to be biased towards spurious correlations between classes and latent attributes exhibited in a major portion of training data, which ruins their generalization capability. We propose a new method for training debiased classifiers with no spurious attribute label. The key idea is to employ a committee of classifiers as an auxiliary module that identifies bias-conflicting data, i.e., data without spurious correlation, and assigns large weights to them when training the main classifier. The committee is learned as a bootstrapped ensemble so that a majority of its classifiers are biased as well as being diverse, and intentionally fail to predict classes of bias-conflicting data accordingly. The consensus within the committee on prediction difficulty thus provides a reliable cue for identifying and weighting bias-conflicting data. Moreover, the committee is also trained with knowledge transferred from the main classifier so that it gradually becomes debiased along with the main classifier and emphasizes more difficult data as training progresses. On five real-world datasets, our method outperforms prior arts using no spurious attribute label like ours and even surpasses those relying on bias labels occasionally. Our code is available at https://github.com/nayeong-v-kim/LWBC.

        ----

        ## [1337] Modeling the Machine Learning Multiverse

        **Authors**: *Samuel J. Bell, Onno Kampman, Jesse Dodge, Neil D. Lawrence*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/750337e1301941f81ae31a90e0a1c181-Abstract-Conference.html)

        **Abstract**:

        Amid mounting concern about the reliability and credibility of machine learning research, we present a principled framework for making robust and generalizable claims: the multiverse analysis. Our framework builds upon the multiverse analysis introduced in response to psychology's own reproducibility crisis. To efficiently explore high-dimensional and often continuous ML search spaces, we model the multiverse with a Gaussian Process surrogate and apply Bayesian experimental design. Our framework is designed to facilitate drawing robust scientific conclusions about model performance, and thus our approach focuses on exploration rather than conventional optimization. In the first of two case studies, we investigate disputed claims about the relative merit of adaptive optimizers.  Second, we synthesize conflicting research on the effect of learning rate on the large batch training generalization gap. For the machine learning community, the multiverse analysis is a simple and effective technique for identifying robust claims, for increasing transparency, and a step toward improved reproducibility.

        ----

        ## [1338] Label-Aware Global Consistency for Multi-Label Learning with Single Positive Labels

        **Authors**: *Ming-Kun Xie, Jiahao Xiao, Sheng-Jun Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/751ef1e7f557a8a88f0837b61bf5070f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/751ef1e7f557a8a88f0837b61bf5070f-Abstract-Conference.html)

        **Abstract**:

        In single positive multi-label learning (SPML), only one of multiple positive labels is observed for each instance. The previous work trains the model by simply treating unobserved labels as negative ones, and designs the regularization to constrain the number of expected positive labels. However, in many real-world scenarios, the true number of positive labels is unavailable, making such methods less applicable. In this paper, we propose to solve SPML problems by designing a Label-Aware global Consistency (LAC) regularization, which leverages the manifold structure information to enhance the recovery of potential positive labels. On one hand, we first perform pseudo-labeling for each unobserved label based on its prediction probability. The consistency regularization is then imposed on model outputs to balance the fitting of identified labels and exploring of potential positive labels. On the other hand, by enforcing label-wise embeddings to maintain global consistency, LAC loss encourages the model to learn more distinctive representations, which is beneficial for recovering the information of potential positive labels. Experiments on multiple benchmark datasets validate that the proposed method can achieve state-of-the-art performance for solving SPML tasks.

        ----

        ## [1339] Unifying Voxel-based Representation with Transformer for 3D Object Detection

        **Authors**: *Yanwei Li, Yilun Chen, Xiaojuan Qi, Zeming Li, Jian Sun, Jiaya Jia*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/752df938681b2cf15e5fc9689f0bcf3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/752df938681b2cf15e5fc9689f0bcf3a-Abstract-Conference.html)

        **Abstract**:

        In this work, we present a unified framework for multi-modality 3D object detection, named UVTR. The proposed method aims to unify multi-modality representations in the voxel space for accurate and robust single- or cross-modality 3D detection. To this end, the modality-specific space is first designed to represent different inputs in the voxel feature space. Different from previous work, our approach preserves the voxel space without height compression to alleviate semantic ambiguity and enable spatial connections. To make full use of the inputs from different sensors, the cross-modality interaction is then proposed, including knowledge transfer and modality fusion. In this way, geometry-aware expressions in point clouds and context-rich features in images are well utilized for better performance and robustness. The transformer decoder is applied to efficiently sample features from the unified space with learnable positions, which facilitates object-level interactions. In general, UVTR presents an early attempt to represent different modalities in a unified framework. It surpasses previous work in single- or multi-modality entries. The proposed method achieves leading performance in the nuScenes test set for both object detection and the following object tracking task. Code is made publicly available at https://github.com/dvlab-research/UVTR.

        ----

        ## [1340] UDC: Unified DNAS for Compressible TinyML Models for Neural Processing Units

        **Authors**: *Igor Fedorov, Ramon Matas Navarro, Hokchhay Tann, Chuteng Zhou, Matthew Mattina, Paul N. Whatmough*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/753d9584b57ba01a10482f1ea7734a89-Abstract-Conference.html)

        **Abstract**:

        Deploying TinyML models on low-cost IoT hardware is very challenging, due to limited device memory capacity. Neural processing unit (NPU) hardware address the memory challenge by using model compression to exploit weight quantization and sparsity to fit more parameters in the same footprint. However, designing compressible neural networks (NNs) is challenging, as it expands the design space across which we must make balanced trade-offs. This paper demonstrates Unified DNAS for Compressible (UDC) NNs, which explores a large search space to generate state-of-the-art compressible NNs for NPU. ImageNet results show UDC networks are up to 3.35x smaller (iso-accuracy) or 6.25% more accurate (iso-model size) than previous work.

        ----

        ## [1341] Measures of Information Reflect Memorization Patterns

        **Authors**: *Rachit Bansal, Danish Pruthi, Yonatan Belinkov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/753fec797a22f71baf7106833734fdf3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/753fec797a22f71baf7106833734fdf3-Abstract-Conference.html)

        **Abstract**:

        Neural networks are known to exploit spurious artifacts (or shortcuts) that co-occur with a target label, exhibiting heuristic memorization. On the other hand, networks have been shown to memorize training examples, resulting in example-level memorization. These kinds of memorization impede generalization of networks beyond their training distributions. Detecting such memorization could be challenging, often requiring researchers to curate tailored test sets. In this work, we hypothesize—and subsequently show—that the diversity in the activation patterns of different neurons is reflective of model generalization and memorization. We quantify the diversity in the neural activations through information-theoretic measures and find support for our hypothesis in experiments spanning several natural language and vision tasks. Importantly, we discover that information organization points to the two forms of memorization, even for neural activations computed on unlabeled in-distribution examples. Lastly, we demonstrate the utility of our findings for the problem of model selection.

        ----

        ## [1342] Fine-Grained Analysis of Stability and Generalization for Modern Meta Learning Algorithms

        **Authors**: *Jiechao Guan, Yong Liu, Zhiwu Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/754e862a9329c5af4c4420d9f2e08c42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/754e862a9329c5af4c4420d9f2e08c42-Abstract-Conference.html)

        **Abstract**:

        The support/query episodic training strategy has been widely applied in modern meta learning algorithms. Supposing the $n$ training episodes and the test episodes are sampled independently from the same environment, previous work has derived a generalization bound of $O(1/\sqrt{n})$ for smooth non-convex functions via algorithmic stability analysis. In this paper, we provide fine-grained analysis of stability and generalization for modern meta learning algorithms by considering more general situations. Firstly, we develop matching lower and upper stability bounds for meta learning algorithms with two types of loss functions: (1) nonsmooth convex functions with $\alpha$-H{\"o}lder continuous subgradients $(\alpha \in [0,1))$; (2) smooth (including convex and non-convex) functions. Our tight stability bounds show that, in the nonsmooth convex case, meta learning algorithms can be inherently less stable than in the smooth convex case. For the smooth non-convex functions, our stability bound is sharper than the existing one, especially in the setting where the number of iterations is larger than the number $n$ of training episodes. Secondly, we derive improved generalization bounds for meta learning algorithms that hold with high probability. Specifically, we first demonstrate that, under the independent episode environment assumption, the generalization bound of $O(1/\sqrt{n})$ via algorithmic stability analysis is near optimal. To attain faster convergence rate, we show how to yield a deformed generalization bound of $O(\ln{n}/n)$ with the curvature condition of loss functions. Finally, we obtain a generalization bound for meta learning with dependent episodes whose dependency relation is characterized by a graph. Experiments on regression problems are conducted to verify our theoretical results.

        ----

        ## [1343] On Scrambling Phenomena for Randomly Initialized Recurrent Networks

        **Authors**: *Vaggos Chatziafratis, Ioannis Panageas, Clayton Sanford, Stelios Stavroulakis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/755acd0c7c07180d78959b6d89768207-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/755acd0c7c07180d78959b6d89768207-Abstract-Conference.html)

        **Abstract**:

        Recurrent Neural Networks (RNNs) frequently exhibit complicated dynamics, and their sensitivity to the initialization process often renders them notoriously hard to train. Recent works have shed light on such phenomena analyzing when exploding or vanishing gradients may occur, either of which is detrimental for training dynamics. In this paper, we point to a formal connection between RNNs and chaotic dynamical systems and prove a qualitatively stronger phenomenon about RNNs than what exploding gradients seem to suggest. Our main result proves that under standard initialization (e.g., He, Xavier etc.), RNNs will exhibit \textit{Li-Yorke chaos} with \textit{constant} probability \textit{independent} of the network's width. This explains the experimentally observed phenomenon of \textit{scrambling}, under which trajectories of nearby points may appear to be arbitrarily close during some timesteps, yet will be far away in future timesteps. In stark contrast to their feedforward counterparts, we show that chaotic behavior in RNNs is preserved under small perturbations and that their expressive power remains exponential in the number of feedback iterations. Our technical arguments rely on viewing RNNs as random walks under non-linear activations, and studying the existence of certain types of higher-order fixed points called \textit{periodic points} in order to establish phase transitions from order to chaos.

        ----

        ## [1344] Learning to Branch with Tree MDPs

        **Authors**: *Lara Scavuzzo, Feng Yang Chen, Didier Chételat, Maxime Gasse, Andrea Lodi, Neil Yorke-Smith, Karen I. Aardal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/756d74cd58592849c904421e3b2ec7a4-Abstract-Conference.html)

        **Abstract**:

        State-of-the-art Mixed Integer Linear Programming (MILP) solvers combine systematic tree search with a plethora of hard-coded heuristics, such as branching rules. While approaches to learn branching strategies have received increasing attention and have shown very promising results, most of the literature focuses on learning fast approximations of the \emph{strong branching} rule. Instead, we propose to learn branching rules from scratch with Reinforcement Learning (RL). We revisit the work of Etheve et al. (2020) and propose a generalization of Markov Decisions Processes (MDP), which we call \emph{tree MDP}, that provides a more suitable formulation of the branching problem. We derive a policy gradient theorem for tree MDPs that exhibits a better credit assignment compared to its temporal counterpart. We demonstrate through computational experiments that this new framework is suitable to tackle the learning-to-branch problem in MILP, and improves the learning convergence.

        ----

        ## [1345] Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs

        **Authors**: *Cristian Bodnar, Francesco Di Giovanni, Benjamin Paul Chamberlain, Pietro Lió, Michael M. Bronstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html)

        **Abstract**:

        Cellular sheaves equip graphs with a ``geometrical'' structure by assigning vector spaces and linear maps to nodes and edges. Graph Neural Networks (GNNs) implicitly assume a graph with a trivial underlying sheaf. This choice is reflected in the structure of the graph Laplacian operator, the properties of the associated diffusion equation, and the characteristics of the convolutional models that discretise this equation. In this paper, we use cellular sheaf theory to show that the underlying geometry of the graph is deeply linked with the performance of GNNs in heterophilic settings and their oversmoothing behaviour. By considering a hierarchy of increasingly general sheaves, we study how the ability of the sheaf diffusion process to achieve linear separation of the classes in the infinite time limit expands. At the same time, we prove that when the sheaf is non-trivial, discretised parametric diffusion processes have greater control than GNNs over their asymptotic behaviour. On the practical side, we study how sheaves can be learned from data. The resulting sheaf diffusion models have many desirable properties that address the limitations of classical graph diffusion equations (and corresponding GNN models) and obtain competitive results in heterophilic settings. Overall, our work provides new connections between GNNs and algebraic topology and would be of interest to both fields.

        ----

        ## [1346] pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models

        **Authors**: *Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang, Jiliang Tang, Weiqi Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/75ca2b23d9794f02a92449af65a57556-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Knowledge tracing (KT) is the task of using students' historical learning interaction data to model their knowledge mastery over time so as to make predictions on their future interaction performance. Recently, remarkable progress has been made of using various deep learning techniques to solve the KT problem. However, the success behind deep learning based knowledge tracing (DLKT) approaches is still left somewhat unknown and proper measurement and analysis of these DLKT approaches remain a challenge. First, data preprocessing procedures in existing works are often private and custom, which limits experimental standardization. Furthermore, existing DLKT studies often differ in terms of the evaluation protocol and are far away real-world educational contexts. To address these problems, we introduce a comprehensive python based benchmark platform, \textsc{pyKT}, to guarantee valid comparisons across DLKT methods via thorough evaluations. The \textsc{pyKT} library consists of a standardized set of integrated data preprocessing procedures on 7 popular datasets across different domains, and 10 frequently compared DLKT model implementations for transparent experiments. Results from our fine-grained and rigorous empirical KT studies yield a set of observations and suggestions for effective DLKT, e.g., wrong evaluation setting may cause label leakage that generally leads to performance inflation; and the improvement of many DLKT approaches is minimal compared to the very first DLKT model proposed by Piech et al. \cite{piech2015deep}. We have open sourced \textsc{pyKT} and our experimental results at \url{https://pykt.org/}. We welcome contributions from other research groups and practitioners.

        ----

        ## [1347] Adversarial Unlearning: Reducing Confidence Along Adversarial Directions

        **Authors**: *Amrith Setlur, Benjamin Eysenbach, Virginia Smith, Sergey Levine*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/75f1a165c7561e028c41d42fa6286a76-Abstract-Conference.html)

        **Abstract**:

        Supervised learning methods trained with maximum likelihood objectives often overfit on training data. Most regularizers that prevent overfitting look to increase confidence on additional examples (e.g., data augmentation, adversarial training), or reduce it on training data (e.g., label smoothing). In this work we propose a complementary regularization strategy that reduces confidence on self-generated examples. The method, which we call RCAD (Reducing Confidence along Adversarial Directions), aims to reduce confidence on out-of-distribution examples lying along directions adversarially chosen to increase training loss. In contrast to adversarial training, RCAD does not try to robustify the model to output the original label, but rather regularizes it to have reduced confidence on points generated using much larger perturbations than in conventional adversarial training. RCAD can be easily integrated into training pipelines with a few lines of code. Despite its simplicity, we find on many classification benchmarks that RCAD can be added to existing techniques (e.g., label smoothing, MixUp training) to increase test accuracy by 1-3% in absolute value, with more significant gains in the low data regime. We also provide a theoretical analysis that helps to explain these benefits in simplified settings, showing that RCAD can provably help the model unlearn spurious features in the training data.

        ----

        ## [1348] How Would The Viewer Feel? Estimating Wellbeing From Video Scenarios

        **Authors**: *Mantas Mazeika, Eric Tang, Andy Zou, Steven Basart, Jun Shern Chan, Dawn Song, David A. Forsyth, Jacob Steinhardt, Dan Hendrycks*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/75ff01252ab45ce278cb060effce4ca1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/75ff01252ab45ce278cb060effce4ca1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In recent years, deep neural networks have demonstrated increasingly strong abilities to recognize objects and activities in videos. However, as video understanding becomes widely used in real-world applications, a key consideration is developing human-centric systems that understand not only the content of the video but also how it would affect the wellbeing and emotional state of viewers. To facilitate research in this setting, we introduce two large-scale datasets with over 60,000 videos manually annotated for emotional response and subjective wellbeing. The Video Cognitive Empathy (VCE) dataset contains annotations for distributions of fine-grained emotional responses, allowing models to gain a detailed understanding of affective states. The Video to Valence (V2V) dataset contains annotations of relative pleasantness between videos, which enables predicting a continuous spectrum of wellbeing. In experiments, we show how video models that are primarily trained to recognize actions and find contours of objects can be repurposed to understand human preferences and the emotional content of videos. Although there is room for improvement, predicting wellbeing and emotional response is on the horizon for state-of-the-art models. We hope our datasets can help foster further advances at the intersection of commonsense video understanding and human preference learning.

        ----

        ## [1349] On Elimination Strategies for Bandit Fixed-Confidence Identification

        **Authors**: *Andrea Tirinzoni, Rémy Degenne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html)

        **Abstract**:

        Elimination algorithms for bandit identification, which prune the plausible correct answers sequentially until only one remains, are computationally convenient since they reduce the problem size over time. However, existing elimination strategies are often not fully adaptive (they update their sampling rule infrequently) and are not easy to extend to combinatorial settings, where the set of answers is exponentially large in the problem dimension. On the other hand, most existing fully-adaptive strategies to tackle general identification problems are computationally demanding since they repeatedly test the correctness of every answer, without ever reducing the problem size. We show that adaptive methods can be modified to use elimination in both their stopping and sampling rules, hence obtaining the best of these two worlds: the algorithms (1) remain fully adaptive, (2) suffer a sample complexity that is never worse of their non-elimination counterpart, and (3) provably eliminate certain wrong answers early. We confirm these benefits experimentally, where elimination improves significantly the computational complexity of adaptive methods on common tasks like best-arm identification in linear bandits.

        ----

        ## [1350] When Adversarial Training Meets Vision Transformers: Recipes from Training to Architecture

        **Authors**: *Yichuan Mo, Dongxian Wu, Yifei Wang, Yiwen Guo, Yisen Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/760b5def8dcb1156aac454e9c0f5f406-Abstract-Conference.html)

        **Abstract**:

        Vision Transformers (ViTs) have recently achieved competitive performance in broad vision tasks. Unfortunately, on popular threat models, naturally trained ViTs are shown to provide no more adversarial robustness than convolutional neural networks (CNNs). Adversarial training is still required for ViTs to defend against such adversarial attacks. In this paper, we provide the first and comprehensive study on the adversarial training recipe of ViTs via extensive evaluation of various training techniques across benchmark datasets. We find that pre-training and SGD optimizer are necessary for ViTs' adversarial training. Further considering ViT as a new type of model architecture, we investigate its adversarial robustness from the perspective of its unique architectural components. We find, when randomly masking gradients from some attention blocks or masking perturbations on some patches during adversarial training, the adversarial robustness of ViTs can be remarkably improved, which may potentially open up a line of work to explore the architectural information inside the newly designed models like ViTs. Our code is available at https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers.

        ----

        ## [1351] Escaping from the Barren Plateau via Gaussian Initializations in Deep Variational Quantum Circuits

        **Authors**: *Kaining Zhang, Liu Liu, Min-Hsiu Hsieh, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7611a3cb5d733e628081431445cb01fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7611a3cb5d733e628081431445cb01fd-Abstract-Conference.html)

        **Abstract**:

        Variational quantum circuits have been widely employed in quantum simulation and quantum machine learning in recent years. However, quantum circuits with random structures have poor trainability due to the exponentially vanishing gradient with respect to the circuit depth and the qubit number. This result leads to a general standpoint that deep quantum circuits would not be feasible for practical tasks. In this work, we propose an initialization strategy with theoretical guarantees for the vanishing gradient problem in general deep quantum circuits. Specifically, we prove that under proper Gaussian initialized parameters, the norm of the gradient decays at most polynomially when the qubit number and the circuit depth increase. Our theoretical results hold for both the local and the global observable cases, where the latter was believed to have vanishing gradients even for very shallow circuits. Experimental results verify our theoretical findings in quantum simulation and quantum chemistry.

        ----

        ## [1352] A sharp NMF result with applications in network modeling

        **Authors**: *Jiashun Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/764651d0887f997fc1e40ff97c8b12e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/764651d0887f997fc1e40ff97c8b12e6-Abstract-Conference.html)

        **Abstract**:

        Given an $n \times n$  non-negative rank-$K$ matrix $\Omega$ where $m$ eigenvalues are negative, when can we write $\Omega = Z P Z'$ for non-negative matrices $Z \in \mathbb{R}^{n, K}$ and $P \in \mathbb{R}^{K, K}$?  While most existing works focused on the case of $m = 0$, our primary interest is on the case of  general $m$. With new proof ideas we develop, we present sharp results on when the NMF problem is solvable, which significantly extend existing results on this topic. The NMF problem is partially motivated by applications in network modeling.   For a network with $K$ communities,  rank-$K$ models are popular, with many proposals. The DCMM model is a recent rank-$K$ model which is especially useful and interpretable in practice. To enjoy such properties,  it is of interest to study when a rank-$K$ model can be rewritten as a DCMM model. Using our NMF results, we show that for a rank-$K$ model with parameters in the most interesting range, we can always rewrite it as a DCMM model.

        ----

        ## [1353] Decoupling Classifier for Boosting Few-shot Object Detection and Instance Segmentation

        **Authors**: *Bin-Bin Gao, Xiaochen Chen, Zhongyi Huang, Congchong Nie, Jun Liu, Jinxiang Lai, Guannan Jiang, Xi Wang, Chengjie Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html)

        **Abstract**:

        This paper focus on few-shot object detection~(FSOD) and instance segmentation~(FSIS), which requires a model to quickly adapt to novel classes with a few labeled instances. The existing methods severely suffer from bias classification because of the missing label issue which naturally exists in an instance-level few-shot scenario and is first formally proposed by us. Our analysis suggests that the standard classification head of most FSOD or FSIS models needs to be decoupled to mitigate the bias classification. Therefore, we propose an embarrassingly simple but effective method that decouples the standard classifier into two heads. Then, these two individual heads are capable of independently addressing clear positive samples and noisy negative samples which are caused by the missing label. In this way, the model can effectively learn novel classes while mitigating the effects of noisy negative samples. Without bells and whistles, our model without any additional computation cost and parameters consistently outperforms its baseline and state-of-the-art by a large margin on PASCAL VOC and MS-COCO benchmarks for FSOD and FSIS tasks.\footnote{\url{https://csgaobb.github.io/Projects/DCFS}.}

        ----

        ## [1354] Private Estimation with Public Data

        **Authors**: *Alex Bie, Gautam Kamath, Vikrant Singhal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/765ec49952dd0140ac754d6d3f9bc899-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/765ec49952dd0140ac754d6d3f9bc899-Abstract-Conference.html)

        **Abstract**:

        We initiate the study of differentially private (DP) estimation with access to a small amount of public data. For private estimation of $d$-dimensional Gaussians, we assume that the public data comes from a Gaussian that may have vanishing similarity in total variation distance with the underlying Gaussian of the private data. We show that under the constraints of pure or concentrated DP, $d+1$ public data samples are sufficient to remove any dependence on the range parameters of the private data distribution from the private sample complexity, which is known to be otherwise necessary without public data. For separated Gaussian mixtures, we assume that the underlying public and private distributions are the same, and we consider two settings: (1) when given a dimension-independent amount of public data, the private sample complexity can be improved polynomially in terms of the number of mixture components, and any dependence on the range parameters of the distribution can be removed in the approximate DP case; (2) when given an amount of public data linear in the dimension, the private sample complexity can be made independent of range parameters even under concentrated DP, and additional improvements can be made to the overall sample complexity.

        ----

        ## [1355] Pre-activation Distributions Expose Backdoor Neurons

        **Authors**: *Runkai Zheng, Rongjun Tang, Jianze Li, Li Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76917808731dae9e6d62c2a7a6afb542-Abstract-Conference.html)

        **Abstract**:

        Convolutional neural networks (CNN) can be manipulated to perform specific behaviors when encountering a particular trigger pattern without affecting the performance on normal samples, which is referred to as backdoor attack. The backdoor attack is usually achieved by injecting a small proportion of poisoned samples into the training set, through which the victim trains a model embedded with the designated backdoor. In this work, we demonstrate that backdoor neurons are exposed by their pre-activation distributions, where populations from benign data and poisoned data show significantly different moments. This property is shown to be attack-invariant and allows us to efficiently locate backdoor neurons. On this basis, we make several proper assumptions on the neuron activation distributions, and propose two backdoor neuron detection strategies based on (1) the differential entropy of the neurons, and (2) the Kullback-Leibler divergence between the benign sample distribution and a poisoned statistics based hypothetical distribution. Experimental results show that our proposed defense strategies are both efficient and effective against various backdoor attacks.

        ----

        ## [1356] DaDA: Distortion-aware Domain Adaptation for Unsupervised Semantic Segmentation

        **Authors**: *Sujin Jang, Joohan Na, Dokwan Oh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76931eaba1fcb55b70cde7d0de0161ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76931eaba1fcb55b70cde7d0de0161ef-Abstract-Conference.html)

        **Abstract**:

        Distributional shifts in photometry and texture have been extensively studied for unsupervised domain adaptation, but their counterparts in optical distortion have been largely neglected. In this work, we tackle the task of unsupervised domain adaptation for semantic image segmentation where unknown optical distortion exists between source and target images. To this end, we propose a distortion-aware domain adaptation (DaDA) framework that boosts the unsupervised segmentation performance. We first present a relative distortion learning (RDL) approach that is capable of modeling domain shifts in fine-grained geometric deformation based on diffeomorphic transformation. Then, we demonstrate that applying additional global affine transformations to the diffeomorphically transformed source images can further improve the segmentation adaptation. Besides, we find that our distortion-aware adaptation method helps to enhance self-supervised learning by providing higher-quality initial models and pseudo labels. To evaluate, we propose new distortion adaptation benchmarks, where rectilinear source images and fisheye target images are used for unsupervised domain adaptation. Extensive experimental results highlight the effectiveness of our approach over state-of-the-art methods under unknown relative distortion across domains. Datasets and more information are available at https://sait-fdd.github.io/.

        ----

        ## [1357] Large-batch Optimization for Dense Visual Predictions: Training Faster R-CNN in 42 Minutes

        **Authors**: *Zeyue Xue, Jianming Liang, Guanglu Song, Zhuofan Zong, Liang Chen, Yu Liu, Ping Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76bea0a1cf7bf9b78f842009f6de15a1-Abstract-Conference.html)

        **Abstract**:

        Training a large-scale deep neural network in a large-scale dataset is challenging and time-consuming. The recent breakthrough of large-batch optimization is a promising way to tackle this challenge. However, although the current advanced algorithms such as LARS and LAMB succeed in classification models, the complicated pipelines of dense visual predictions such as object detection and segmentation still suffer from the heavy performance drop in the large-batch training regime. To address this challenge, we propose a simple yet effective algorithm, named Adaptive Gradient Variance Modulator (AGVM), which can train dense visual predictors with very large batch size, enabling several benefits more appealing than prior arts. Firstly, AGVM can align the gradient variances between different modules in the dense visual predictors, such as backbone, feature pyramid network (FPN), detection, and segmentation heads. We show that training with a large batch size can fail with the gradient variances misaligned among them, which is a phenomenon primarily overlooked in previous work. Secondly, AGVM is a plug-and-play module that generalizes well to many different architectures (e.g., CNNs and Transformers) and different tasks (e.g., object detection, instance segmentation, semantic segmentation, and panoptic segmentation). It is also compatible with different optimizers (e.g., SGD and AdamW). Thirdly, a theoretical analysis of AGVM is provided. Extensive experiments on the COCO and ADE20K datasets demonstrate the superiority of AGVM. For example, AGVM demonstrates more stable generalization performance than prior arts under extremely large batch size (i.e., 10k). AGVM can train Faster R-CNN+ResNet50 in 4.2 minutes without losing performance. It enables training an object detector with one billion parameters in just 3.5 hours, reducing the training time by 20.9Ã—, whilst achieving 62.2 mAP on COCO. The deliverables will be released at https://github.com/Sense-X/AGVM.

        ----

        ## [1358] Most Activation Functions Can Win the Lottery Without Excessive Depth

        **Authors**: *Rebekka Burkholz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76bf7786d311217077bc8bb021946cd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76bf7786d311217077bc8bb021946cd9-Abstract-Conference.html)

        **Abstract**:

        The strong lottery ticket hypothesis has highlighted the potential for training deep neural networks by pruning, which has inspired interesting practical and theoretical insights into how neural networks can represent functions. For networks with ReLU activation functions, it has been proven that a target network with depth L can be approximated by the subnetwork of a randomly initialized neural network that has double the target's depth 2L and is wider by a logarithmic factor. We show that a depth L+1 is sufficient. This result indicates that we can expect to find lottery tickets at realistic, commonly used depths while only requiring logarithmic overparametrization. Our novel construction approach applies to a large class of activation functions and is not limited to ReLUs. Code is available on Github (RelationalML/LT-existence).

        ----

        ## [1359] Rapid Model Architecture Adaption for Meta-Learning

        **Authors**: *Yiren Zhao, Xitong Gao, Ilia Shumailov, Nicolò Fusi, Robert D. Mullins*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76df3f555683bc6c4b988bb81b930d5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76df3f555683bc6c4b988bb81b930d5b-Abstract-Conference.html)

        **Abstract**:

        Network Architecture Search (NAS) methods have recently gathered much attention. They design networks with better performance and use a much shorter search time compared to traditional manual tuning. Despite their efficiency in model deployments, most NAS algorithms target a single task on a fixed hardware system. However, real-life few-shot learning environments often cover a great number of tasks ($T$) and deployments on a wide variety of hardware platforms ($H$). The combinatorial search complexity $T \times H$ creates a fundamental search efficiency challenge if one naively applies existing NAS methods to these scenarios. To overcome this issue, we show, for the first time, how to rapidly adapt model architectures to new tasks in a \emph{many-task many-hardware} few-shot learning setup by integrating Model Agnostic Meta Learning (MAML) into the NAS flow. The proposed NAS method (H-Meta-NAS) is hardware-aware and performs optimisation in the MAML framework. MetaNAS shows a Pareto dominance compared to a variety of NAS and manual baselines in popular few-shot learning benchmarks with various hardware platforms and constraints. In particular, on the 5-way 1-shot Mini-ImageNet classification task,  the proposed method outperforms the best manual baseline by a large margin ($5.21\%$ in accuracy) using $60\%$ less computation.

        ----

        ## [1360] Exploring through Random Curiosity with General Value Functions

        **Authors**: *Aditya Ramesh, Louis Kirsch, Sjoerd van Steenkiste, Jürgen Schmidhuber*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/76e57c3c6b3e06f332a4832ddd6a9a12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/76e57c3c6b3e06f332a4832ddd6a9a12-Abstract-Conference.html)

        **Abstract**:

        Efficient exploration in reinforcement learning is a challenging problem commonly addressed through intrinsic rewards. Recent prominent approaches are based on state novelty or variants of artificial curiosity. However, directly applying them to partially observable environments can be ineffective and lead to premature dissipation of intrinsic rewards. Here we propose random curiosity with general value functions (RC-GVF), a novel intrinsic reward function that draws upon connections between these distinct approaches. Instead of using only the current observation’s novelty or a curiosity bonus for failing to predict precise environment dynamics, RC-GVF derives intrinsic rewards through predicting temporally extended general value functions. We demonstrate that this improves exploration in a hard-exploration diabolical lock problem. Furthermore, RC-GVF significantly outperforms previous methods in the absence of ground-truth episodic counts in the partially observable MiniGrid environments. Panoramic observations on MiniGrid further boost RC-GVF's performance such that it is competitive to baselines exploiting privileged information in form of episodic counts.

        ----

        ## [1361] Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition

        **Authors**: *Dae Ha Kim, Byung Cheol Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7715137660f3e38785eb8d46261e89da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7715137660f3e38785eb8d46261e89da-Abstract-Conference.html)

        **Abstract**:

        Identity-invariant facial expression recognition (FER) has been one of the challenging computer vision tasks. Since conventional FER schemes do not explicitly address the inter-identity variation of facial expressions, their neural network models still operate depending on facial identity. This paper proposes to quantify the inter-identity variation by utilizing pairs of similar expressions explored through a specific matching process. We formulate the identity matching process as an Optimal Transport (OT) problem. Specifically, to find pairs of similar expressions from different identities, we define the inter-feature similarity as a transportation cost. Then, optimal identity matching to find the optimal flow with minimum transportation cost is performed by Sinkhorn-Knopp iteration. The proposed matching method is not only easy to plug in to other models, but also requires only acceptable computational overhead. Extensive simulations prove that the proposed FER method improves the PCC/CCC performance by up to 10% or more compared to the runner-up on wild datasets. The source code and software demo are available at https://github.com/kdhht2334/ELIM_FER.

        ----

        ## [1362] Unsupervised Learning under Latent Label Shift

        **Authors**: *Manley Roberts, Pranav Mani, Saurabh Garg, Zachary C. Lipton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/771e09dd204ea339da0d8114c48afd21-Abstract-Conference.html)

        **Abstract**:

        What sorts of structure might enable a learner to discover classes from unlabeled data? Traditional approaches rely on feature-space similarity and heroic assumptions on the data. In this paper, we introduce unsupervised learning under Latent Label Shift (LLS), where the label marginals $p_d(y)$ shift but the class conditionals $p(x|y)$ do not. This work instantiates a new principle for identifying classes: elements that shift together group together. For finite input spaces, we establish an isomorphism between LLS and topic modeling: inputs correspond to words, domains to documents, and labels to topics. Addressing continuous data, we prove that when each label's support contains a separable region, analogous to an anchor word, oracle access to $p(d|x)$ suffices to identify $p_d(y)$ and $p_d(y|x)$ up to permutation. Thus motivated, we introduce a practical algorithm that leverages domain-discriminative models as follows: (i) push examples through domain discriminator $p(d|x)$; (ii) discretize the data by clustering examples in $p(d|x)$ space; (iii) perform non-negative matrix factorization on the discrete data; (iv) combine the recovered $p(y|d)$ with the discriminator outputs $p(d|x)$ to compute $p_d(y|x) \; \forall d$. With semisynthetic experiments, we show that our algorithm can leverage domain information to improve upon competitiveunsupervised classification methods. We reveal a failure mode of standard unsupervised classification methods when data-space similarity does not indicate true groupings, and show empirically that our method better handles this case. Our results establish a deep connection between distribution shift and topic modeling, opening promising lines for future work.

        ----

        ## [1363] SHINE: SubHypergraph Inductive Neural nEtwork

        **Authors**: *Yuan Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7721f1fea280e9ffae528dc78c732576-Abstract-Conference.html)

        **Abstract**:

        Hypergraph neural networks can model multi-way connections among nodes of the graphs, which are common in real-world applications such as genetic medicine. In particular, genetic pathways or gene sets encode molecular functions driven by multiple genes, naturally represented as hyperedges. Thus, hypergraph-guided embedding can capture functional relations in learned representations. Existing hypergraph neural network models often focus on node-level or graph-level inference. There is an unmet need in learning powerful representations of subgraphs of hypergraphs in real-world applications. For example, a cancer patient can be viewed as a subgraph of genes harboring mutations in the patient, while all the genes are connected by hyperedges that correspond to pathways representing specific molecular functions. For accurate inductive subgraph prediction, we propose SubHypergraph Inductive Neural nEtwork (SHINE). SHINE uses informative genetic pathways that encode molecular functions as hyperedges to connect genes as nodes. SHINE jointly optimizes the objectives of end-to-end subgraph classification and hypergraph nodes' similarity regularization. SHINE simultaneously learns representations for both genes and pathways using strongly dual attention message passing. The learned representations are aggregated via a subgraph attention layer and used to train a multilayer perceptron for subgraph inferencing. We evaluated SHINE against a wide array of state-of-the-art (hyper)graph neural networks, XGBoost, NMF and polygenic risk score models, using large scale NGS and curated datasets. SHINE outperformed all comparison models significantly, and yielded interpretable disease models with functional insights.

        ----

        ## [1364] Efficient Aggregated Kernel Tests using Incomplete $U$-statistics

        **Authors**: *Antonin Schrab, Ilmun Kim, Benjamin Guedj, Arthur Gretton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/774164b966cc277c82a960934445140d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/774164b966cc277c82a960934445140d-Abstract-Conference.html)

        **Abstract**:

        We propose a series of computationally efficient, nonparametric tests for the two-sample, independence and goodness-of-fit problems, using the  Maximum Mean Discrepancy (MMD), Hilbert Schmidt Independence Criterion (HSIC), and Kernel Stein Discrepancy (KSD), respectively. Our test statistics are incomplete $U$-statistics, with a computational cost that interpolates between linear time in the number of samples, and quadratic time, as associated with classical $U$-statistic tests. The three proposed tests aggregate over several kernel bandwidths to detect departures from the null on various scales: we call the resulting tests MMDAggInc, HSICAggInc and KSDAggInc. This procedure provides a solution to the fundamental kernel selection problem as we can aggregate a large number of kernels with several bandwidths without incurring a significant loss of test power. For the test thresholds, we derive a quantile bound for wild bootstrapped incomplete $U$-statistics, which is of independent interest. We derive non-asymptotic uniform separation rates for MMDAggInc and HSICAggInc, and quantify exactly the trade-off between computational efficiency and the attainable rates: this result is novel for tests based on incomplete $U$-statistics, to our knowledge. We further show that in the quadratic-time case, the wild bootstrap incurs no penalty to test power over more widespread permutation-based approaches, since both attain the same minimax optimal rates (which in turn match the rates that use oracle quantiles). We support our claims with numerical experiments on the trade-off between computational efficiency and test power. In all three testing frameworks, our proposed linear-time tests outperform the current linear-time state-of-the-art tests (or at least match their test power).

        ----

        ## [1365] Influencing Long-Term Behavior in Multiagent Reinforcement Learning

        **Authors**: *Dong-Ki Kim, Matthew Riemer, Miao Liu, Jakob N. Foerster, Michael Everett, Chuangchuang Sun, Gerald Tesauro, Jonathan P. How*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7749f9c0d5ff109231be21e910a3ced2-Abstract-Conference.html)

        **Abstract**:

        The main challenge of multiagent reinforcement learning is the difficulty of learning useful policies in the presence of other simultaneously learning agents whose changing behaviors jointly affect the environment's transition and reward dynamics. An effective approach that has recently emerged for addressing this non-stationarity is for each agent to anticipate the learning of other agents and influence the evolution of future policies towards desirable behavior for its own benefit. Unfortunately, previous approaches for achieving this suffer from myopic evaluation, considering only a finite number of policy updates. As such, these methods can only influence transient future policies rather than achieving the promise of scalable equilibrium selection approaches that influence the behavior at convergence. In this paper, we propose a principled framework for considering the limiting policies of other agents as time approaches infinity. Specifically, we develop a new optimization objective that maximizes each agent's average reward by directly accounting for the impact of its behavior on the limiting set of policies that other agents will converge to. Our paper characterizes desirable solution concepts within this problem setting and provides practical approaches for optimizing over possible outcomes. As a result of our farsighted objective, we demonstrate better long-term performance than state-of-the-art baselines across a suite of diverse multiagent benchmark domains.

        ----

        ## [1366] Quantized Training of Gradient Boosting Decision Trees

        **Authors**: *Yu Shi, Guolin Ke, Zhuoming Chen, Shuxin Zheng, Tie-Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77911ed9e6e864ca1a3d165b2c3cb258-Abstract-Conference.html)

        **Abstract**:

        Recent years have witnessed significant success in Gradient Boosting Decision Trees (GBDT) for a wide range of machine learning applications. Generally, a consensus about GBDT's training algorithms is gradients and statistics are computed based on high-precision floating points. In this paper, we investigate an essentially important question which has been largely ignored by the previous literature - how many bits are needed for representing gradients in training GBDT? To solve this mystery, we propose to quantize all the high-precision gradients in a very simple yet effective way in the GBDT's training algorithm. Surprisingly, both our theoretical analysis and empirical studies show that the necessary precisions of gradients without hurting any performance can be quite low, e.g., 2 or 3 bits. With low-precision gradients, most arithmetic operations in GBDT training can be replaced by integer operations of 8, 16, or 32 bits. Promisingly, these findings may pave the way for much more efficient training of GBDT from several aspects: (1) speeding up the computation of gradient statistics in histograms; (2) compressing the communication cost of high-precision statistical information during distributed training; (3) the inspiration of utilization and development of hardware architectures which well support low-precision computation for GBDT training. Benchmarked on CPUs, GPUs, and distributed clusters, we observe up to 2$\times$ speedup of our simple quantization strategy compared with SOTA GBDT systems on extensive datasets, demonstrating the effectiveness and potential of the low-precision training of GBDT. The code will be released to the official repository of LightGBM.

        ----

        ## [1367] Convergent Representations of Computer Programs in Human and Artificial Neural Networks

        **Authors**: *Shashank Srikant, Ben Lipkin, Anna A. Ivanova, Evelina Fedorenko, Una-May O'Reilly*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77b5aaf2826c95c98e5eb4ab830073de-Abstract-Conference.html)

        **Abstract**:

        What aspects of computer programs are represented by the human brain during comprehension? We leverage brain recordings derived from functional magnetic resonance imaging (fMRI) studies of programmers comprehending Python code to evaluate the properties and code-related information encoded in the neural signal. We first evaluate a selection of static and dynamic code properties, such as abstract syntax tree (AST)-related and runtime-related metrics. Then, to learn whether brain representations encode fine-grained information about computer programs, we train a probe to align brain recordings with representations learned by a suite of ML models. We find that both the Multiple Demand and Language systems--brain systems which are responsible for very different cognitive tasks, encode specific code properties and uniquely align with machine learned representations of code. These findings suggest at least two distinct neural mechanisms mediating computer program comprehension and evaluation, prompting the design of code model objectives that go beyond static language modeling.We make all the corresponding code, data, and analysis publicly available at https://github.com/ALFA-group/code-representations-ml-brain

        ----

        ## [1368] QC-StyleGAN - Quality Controllable Image Generation and Manipulation

        **Authors**: *Dat Viet Thanh Nguyen, Phong Tran The, Tan M. Dinh, Cuong Pham, Anh Tran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77b6da696f6fe10ef3cbe748bd8526b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77b6da696f6fe10ef3cbe748bd8526b3-Abstract-Conference.html)

        **Abstract**:

        The introduction of high-quality image generation models, particularly the StyleGAN family, provides a powerful tool to synthesize and manipulate images. However, existing models are built upon high-quality (HQ) data as desired outputs, making them unfit for in-the-wild low-quality (LQ) images, which are common inputs for manipulation. In this work, we bridge this gap by proposing a novel GAN structure that allows for generating images with controllable quality. The network can synthesize various image degradation and restore the sharp image via a quality control code. Our proposed QC-StyleGAN can directly edit LQ images without altering their quality by applying GAN inversion and manipulation techniques. It also provides for free an image restoration solution that can handle various degradations, including noise, blur, compression artifacts, and their mixtures. Finally, we demonstrate numerous other applications such as image degradation synthesis, transfer, and interpolation.

        ----

        ## [1369] Retrieve, Reason, and Refine: Generating Accurate and Faithful Patient Instructions

        **Authors**: *Fenglin Liu, Bang Yang, Chenyu You, Xian Wu, Shen Ge, Zhangdaihong Liu, Xu Sun, Yang Yang, David A. Clifton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77c08a6e68ae25433f1d117283c0e312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77c08a6e68ae25433f1d117283c0e312-Abstract-Conference.html)

        **Abstract**:

        The "Patient Instruction" (PI), which contains critical instructional information provided both to carers and to the patient at the time of discharge, is essential for the patient to manage their condition outside hospital. An accurate and easy-to-follow PI can improve the self-management of patients which can in turn reduce hospital readmission rates. However, writing an appropriate PI can be extremely time consuming for physicians, and is subject to being incomplete or error-prone for (potentially overworked) physicians. Therefore, we propose a new task that can provide an objective means of avoiding incompleteness, while reducing clinical workload: the automatic generation of the PI, which is imagined as being a document that the clinician can review, modify, and approve as necessary (rather than taking the human "out of the loop"). We build a benchmark clinical dataset and propose the Re$^3$Writer, which imitates the working patterns of physicians to first retrieve related working experience from historical PIs written by physicians, then reason related medical knowledge. Finally, it refines the retrieved working experience and reasoned medical knowledge to extract useful information, which is used to generate the PI for previously-unseen patient according to their health records during hospitalization. Our experiments show that, using our method, the performance of 6 different models can be substantially boosted across all metrics, with up to 20%, 11%, and 19% relative improvements in BLEU-4, ROUGE-L, and METEOR, respectively. Meanwhile, we show results from human evaluations to measure the effectiveness in terms of its usefulness for clinical practice. The code is available at https://github.com/AI-in-Health/Patient-Instructions.

        ----

        ## [1370] Data Distributional Properties Drive Emergent In-Context Learning in Transformers

        **Authors**: *Stephanie C. Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya K. Singh, Pierre H. Richemond, James L. McClelland, Felix Hill*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77c6ccacfd9962e2307fc64680fc5ace-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77c6ccacfd9962e2307fc64680fc5ace-Abstract-Conference.html)

        **Abstract**:

        Large transformer-based models are able to perform in-context few-shot learning, without being explicitly trained for it. This observation raises the question: what aspects of the training regime lead to this emergent behavior? Here, we show that this behavior is driven by the distributions of the training data itself. In-context learning emerges when the training data exhibits particular distributional properties such as burstiness (items appear in clusters rather than being uniformly distributed over time) and having a large number of rarely occurring classes. In-context learning also emerges more strongly when item meanings or interpretations are dynamic rather than fixed. These properties are exemplified by natural language, but are also inherent to naturalistic data in a wide range of other domains. They also depart significantly from the uniform, i.i.d. training distributions typically used for standard supervised learning. In our initial experiments, we found that in-context learning traded off against more conventional weight-based learning, and models were unable to achieve both simultaneously. However, our later experiments uncovered that the two modes of learning could co-exist in a single model when it was trained on data following a skewed Zipfian distribution -- another common property of naturalistic data, including language. In further experiments, we found that naturalistic data distributions were only able to elicit in-context learning in transformers, and not in recurrent models. Our findings indicate how the transformer architecture works together with particular properties of the training data to drive the intriguing emergent in-context learning behaviour of large language models, and indicate how future work might encourage both in-context and in-weights learning in domains beyond language.

        ----

        ## [1371] Algorithms that Approximate Data Removal: New Results and Limitations

        **Authors**: *Vinith M. Suriyakumar, Ashia C. Wilson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html)

        **Abstract**:

        We study the problem of deleting user data from machine learning models trained using empirical risk minimization (ERM). Our focus is on learning algorithms which return the empirical risk minimizer and approximate unlearning algorithms that comply with deletion requests that come in an online manner. Leveraging the infintesimal jacknife, we develop an online unlearning algorithm that is both computationally and memory efficient. Unlike prior memory efficient unlearning algorithms, we target ERM trained models that minimize objectives with non-smooth regularizers, such as the commonly used $\ell_1$, elastic net, or nuclear norm penalties. We also provide generalization, deletion capacity, and unlearning guarantees that are consistent with state of the art methods. Across a variety of benchmark datasets, our algorithm empirically improves upon the runtime of prior methods while maintaining the same memory requirements and test accuracy. Finally, we open a new direction of inquiry by proving that all approximate unlearning algorithms introduced so far fail to unlearn in problem settings where common hyperparameter tuning methods, such as cross-validation, have been used to select models.

        ----

        ## [1372] LOT: Layer-wise Orthogonal Training on Improving l2 Certified Robustness

        **Authors**: *Xiaojun Xu, Linyi Li, Bo Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77d52754ff6b2de5a5d96ee921b6b3cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77d52754ff6b2de5a5d96ee921b6b3cd-Abstract-Conference.html)

        **Abstract**:

        Recent studies show that training deep neural networks (DNNs) with Lipschitz constraints are able to enhance adversarial robustness and other model properties such as stability. In this paper, we propose a layer-wise orthogonal training method (LOT) to effectively train 1-Lipschitz convolution layers via parametrizing an orthogonal matrix with an unconstrained matrix. We then efficiently compute the inverse square root of a convolution kernel by transforming the input domain to the Fourier frequency domain. On the other hand, as existing works show that semi-supervised training helps improve empirical robustness, we aim to bridge the gap and prove that semi-supervised learning also improves the certified robustness of Lipschitz-bounded models. We conduct comprehensive evaluations for LOT under different settings. We show that LOT significantly outperforms baselines regarding deterministic l2 certified robustness, and scales to deeper neural networks. Under the supervised scenario, we improve the state-of-the-art certified robustness for all architectures (e.g. from 59.04% to 63.50% on CIFAR-10 and from 32.57% to 34.59% on CIFAR-100 at radius $\rho=36/255$ for 40-layer networks). With semi-supervised learning over unlabelled data, we are able to improve state-of-the-art certified robustness on CIFAR-10 at $\rho=108/255$ from 36.04% to 42.39%. In addition, LOT consistently outperforms baselines on different model architectures with only 1/3 evaluation time.

        ----

        ## [1373] Lottery Tickets on a Data Diet: Finding Initializations with Sparse Trainable Networks

        **Authors**: *Mansheej Paul, Brett W. Larsen, Surya Ganguli, Jonathan Frankle, Gintare Karolina Dziugaite*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77dd8e90fe833eba5fae86cf017d7a56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77dd8e90fe833eba5fae86cf017d7a56-Abstract-Conference.html)

        **Abstract**:

        A striking observation about iterative magnitude pruning (IMP; Frankle et al. 2020) is that—after just a few hundred steps of dense training—the method can find a sparse sub-network that can be trained to the same accuracy as the dense network. However, the same does not hold at step 0, i.e. random initialization. In this work, we seek to understand how this early phase of pre-training leads to a good initialization for IMP both through the lens of the data distribution and the loss landscape geometry. Empirically we observe that, holding the number of pre-training iterations constant, training on a small fraction of (randomly chosen) data suffices to obtain an equally good initialization for IMP. We additionally observe that by pre-training only on "easy" training data, we can decrease the number of steps necessary to find a good initialization for IMP compared to training on the full dataset or a randomly chosen subset. Finally, we identify novel properties of the loss landscape of dense networks that are predictive of IMP performance, showing in particular that more examples being linearly mode connected in the dense network correlates well with good initializations for IMP. Combined, these results provide new insight into the role played by the early phase training in IMP.

        ----

        ## [1374] Regularized Molecular Conformation Fields

        **Authors**: *Lihao Wang, Yi Zhou, Yiqun Wang, Xiaoqing Zheng, Xuanjing Huang, Hao Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77e5109bdf9f337e11e004c22c8ac89d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77e5109bdf9f337e11e004c22c8ac89d-Abstract-Conference.html)

        **Abstract**:

        Predicting energetically favorable 3-dimensional conformations of organic molecules frommolecular graph plays a fundamental role in computer-aided drug discovery research.However, effectively exploring the high-dimensional conformation space to identify (meta) stable conformers is anything but trivial.In this work, we introduce RMCF, a novel framework to generate a diverse set of low-energy molecular conformations through samplingfrom a regularized molecular conformation field.We develop a data-driven molecular segmentation algorithm to automatically partition each molecule into several structural building blocks to reduce the modeling degrees of freedom.Then, we employ a Markov Random Field to learn the joint probability distribution of fragment configurations and inter-fragment dihedral angles, which enables us to sample from different low-energy regions of a conformation space.Our model constantly outperforms state-of-the-art models for the conformation generation task on the GEOM-Drugs dataset.We attribute the success of RMCF to modeling in a regularized feature space and learning a global fragment configuration distribution for effective sampling.The proposed method could be generalized to deal with larger biomolecular systems.

        ----

        ## [1375] Reconstruction on Trees and Low-Degree Polynomials

        **Authors**: *Frederic Koehler, Elchanan Mossel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77e6814d32a86b76123bd10aa7e2ad81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77e6814d32a86b76123bd10aa7e2ad81-Abstract-Conference.html)

        **Abstract**:

        The study of Markov processes and broadcasting on trees has deep connections to a variety of areas including statistical physics, graphical models, phylogenetic reconstruction, Markov Chain Monte Carlo, and community detection in random graphs. Notably, the celebrated Belief Propagation (BP) algorithm achieves Bayes-optimal performance for the reconstruction problem of predicting the value of the Markov process at the root of the tree from its values at the leaves.Recently, the analysis of low-degree polynomials has emerged as a valuable tool for predicting computational-to-statistical gaps. In this work, we investigate the performance of low-degree polynomials for the reconstruction problem on trees. Perhaps surprisingly, we show that there are simple tree models with $N$ leaves and bounded arity where (1) nontrivial reconstruction of the root value is possible with a simple polynomial time algorithm and with robustness to noise, but not with any polynomial of degree $N^{c}$ for $c > 0$ a constant depending only on the arity, and (2) when the tree is unknown and given multiple samples with correlated root assignments, nontrivial reconstruction of the root value is possible with a simple Statistical Query algorithm but not with any polynomial of degree $N^c$. These results clarify some of the limitations of low-degree polynomials vs. polynomial time algorithms for Bayesian estimation problems. They also complement recent work of Moitra, Mossel, and Sandon who studied the circuit complexity of Belief Propagation.  As a consequence of our main result, we are able to prove a result of independent interest regarding the performance of RBF kernel ridge regression for learning to predict the root coloration: for some $c' > 0$ depending only on the arity, $\exp(N^{c'})$ many samples are needed for the kernel regression to obtain nontrivial correlation with the true regression function (BP). We pose related open questions about low-degree polynomials and the Kesten-Stigum threshold.

        ----

        ## [1376] Lower Bounds and Nearly Optimal Algorithms in Distributed Learning with Communication Compression

        **Authors**: *Xinmeng Huang, Yiming Chen, Wotao Yin, Kun Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/77f2d0c271e508278ea13e24cd8773d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/77f2d0c271e508278ea13e24cd8773d5-Abstract-Conference.html)

        **Abstract**:

        Recent advances in distributed optimization and learning have shown that communication compression is one of the most effective means of reducing communication. While there have been many results for convergence rates with compressed communication, a lower bound is still missing.Analyses of algorithms with communication compression have identified two abstract properties that guarantee convergence: the unbiased property or the contractive property. They can be applied either unidirectionally (compressing messages from worker to server) or bidirectionally. In the smooth and non-convex stochastic regime, this paper establishes a lower bound for distributed algorithms whether using unbiased or contractive compressors in unidirection or bidirection. To close the gap between this lower bound and the best existing upper bound, we further propose an algorithm, NEOLITHIC, that almost reaches our lower bound (except for a logarithm factor) under mild conditions. Our results also show that using contractive compressors in bidirection can yield iterative methods that converge as fast as those using unbiased compressors unidirectionally. We report experimental results that validate our findings.

        ----

        ## [1377] Memory safe computations with XLA compiler

        **Authors**: *Artem Artemev, Yuze An, Tilman Roeder, Mark van der Wilk*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/782b6152c04e9948c2cb3833e9a288ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/782b6152c04e9948c2cb3833e9a288ef-Abstract-Conference.html)

        **Abstract**:

        Software packages like TensorFlow and PyTorch are designed to support linear algebra operations, and their speed and usability determine their success. However, by prioritising speed, they often neglect memory requirements. As a consequence, the implementations of memory-intensive algorithms that are convenient in terms of software design can often not be run for large problems due to memory overflows. Memory-efficient solutions require complex programming approaches with significant logic outside the computational framework. This impairs the adoption and use of such algorithms. To address this, we developed an XLA compiler extension that adjusts the computational data-flow representation of an algorithm according to a user-specified memory limit. We show that k-nearest neighbour, sparse Gaussian process regression methods and Transformers can be run on a single device at a much larger scale, where standard implementations would have failed. Our approach leads to better use of hardware resources. We believe that further focus on removing memory constraints at a compiler level will widen the range of machine learning methods that can be developed in the future.

        ----

        ## [1378] Towards Theoretically Inspired Neural Initialization Optimization

        **Authors**: *Yibo Yang, Hong Wang, Haobo Yuan, Zhouchen Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html)

        **Abstract**:

        Automated machine learning has been widely explored to reduce human efforts in designing neural architectures and looking for proper hyperparameters. In the domain of neural initialization, however, similar automated techniques have rarely been studied. Most existing initialization methods are handcrafted and highly dependent on specific architectures. In this paper, we propose a differentiable quantity, named GradCoisne, with theoretical insights to evaluate the initial state of a neural network. Specifically, GradCosine is the cosine similarity of sample-wise gradients with respect to the initialized parameters. By analyzing the sample-wise optimization landscape, we show that both the training and test performance of a network can be improved by maximizing GradCosine under gradient norm constraint. Based on this observation, we further propose the neural initialization optimization (NIO) algorithm. Generalized from the sample-wise analysis into the real batch setting, NIO is able to automatically look for a better initialization with negligible cost compared with the training time. With NIO, we improve the classification performance of a variety of neural architectures on CIFAR10, CIFAR-100, and ImageNet. Moreover, we find that our method can even help to train large vision Transformer architecture without warmup.

        ----

        ## [1379] AnimeRun: 2D Animation Visual Correspondence from Open Source 3D Movies

        **Authors**: *Li Siyao, Yuhang Li, Bo Li, Chao Dong, Ziwei Liu, Chen Change Loy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/78b23d272f58fe3789ab490ebf080fa5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/78b23d272f58fe3789ab490ebf080fa5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Visual correspondence of 2D animation is the core of many applications and deserves careful study. Existing correspondence datasets for 2D cartoon suffer from simple frame composition and monotonic movements, making them  insufficient to simulate real animations. In this work, we present a new 2D animation visual correspondence dataset, AnimeRun, by converting open source 3D movies to full scenes in 2D style, including simultaneous moving background and interactions of multiple subjects. Statistics show that our proposed dataset not only resembles real anime more in image composition, but also possesses richer and more complex motion patterns compared to existing datasets. With this dataset, we establish a comprehensive benchmark by evaluating several existing optical flow and segment matching methods, and analyze shortcomings of these methods on animation data. Data are available at https://lisiyao21.github.io/projects/AnimeRun.

        ----

        ## [1380] Uncertainty Estimation Using Riemannian Model Dynamics for Offline Reinforcement Learning

        **Authors**: *Guy Tennenholtz, Shie Mannor*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/78e36c70d5051e9e271b00289624d709-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/78e36c70d5051e9e271b00289624d709-Abstract-Conference.html)

        **Abstract**:

        Model-based offline reinforcement learning approaches generally rely on bounds of model error. Estimating these bounds is usually achieved through uncertainty estimation methods. In this work, we combine parametric and nonparametric methods for uncertainty estimation through a novel latent space based metric. In particular, we build upon recent advances in Riemannian geometry of generative models to construct a pullback metric of an encoder-decoder based forward model. Our proposed metric measures both the quality of out-of-distribution samples as well as the discrepancy of examples in the data. We leverage our combined method for uncertainty estimation in a pessimistic model-based framework, showing a significant improvement upon contemporary model-based offline approaches on continuous control and autonomous driving benchmarks.

        ----

        ## [1381] Improved Convergence Rate of Stochastic Gradient Langevin Dynamics with Variance Reduction and its Application to Optimization

        **Authors**: *Yuri Kinoshita, Taiji Suzuki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/78e839f96568985d18463044a064ea0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/78e839f96568985d18463044a064ea0f-Abstract-Conference.html)

        **Abstract**:

        The stochastic gradient Langevin Dynamics is one of the most fundamental algorithms to solve sampling problems and non-convex optimization appearing in several machine learning applications. Especially, its variance reduced versions have nowadays gained particular attention. In this paper, we study two variants of this kind, namely, the Stochastic Variance Reduced Gradient Langevin Dynamics and the Stochastic Recursive Gradient Langevin Dynamics. We prove their convergence to the objective distribution in terms of KL-divergence under the sole assumptions of smoothness and Log-Sobolev inequality which are weaker conditions than those used in prior works for these algorithms. With the batch size and the inner loop length set to $\sqrt{n}$, the gradient complexity to achieve an $\epsilon$-precision is $\tilde{O}((n+dn^{1/2}\epsilon^{-1})\gamma^2 L^2\alpha^{-2})$, which is an improvement from any previous analyses. We also show some essential applications of our result to non-convex optimization.

        ----

        ## [1382] Practical Adversarial Attacks on Spatiotemporal Traffic Forecasting Models

        **Authors**: *Fan Liu, Hao Liu, Wenzhao Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79081c95482707d2db390542614e29cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79081c95482707d2db390542614e29cd-Abstract-Conference.html)

        **Abstract**:

        Machine learning based traffic forecasting models leverage sophisticated spatiotemporal auto-correlations to provide accurate predictions of city-wide traffic states. However, existing methods assume a reliable and unbiased forecasting environment, which is not always available in the wild. In this work, we investigate the vulnerability of spatiotemporal traffic forecasting models and propose a practical adversarial spatiotemporal attack framework. Specifically, instead of simultaneously attacking all geo-distributed data sources, an iterative gradient guided node saliency method is proposed to identify the time-dependent set of victim nodes. Furthermore, we devise a spatiotemporal gradient descent based scheme to generate real-valued adversarial traffic states under a perturbation constraint.Meanwhile, we theoretically demonstrate the worst performance bound of adversarial traffic forecasting attacks. Extensive experiments on two real-world datasets show that the proposed two-step framework achieves up to 67.8% performance degradation on various advanced spatiotemporal forecasting models. Remarkably, we also show that adversarial training with our proposed attacks can significantly improve the robustness of spatiotemporal traffic forecasting models.

        ----

        ## [1383] Efficient learning of nonlinear prediction models with time-series privileged information

        **Authors**: *Bastian Jung, Fredrik D. Johansson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/792845ddfe4047d7066348e52e46b74d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/792845ddfe4047d7066348e52e46b74d-Abstract-Conference.html)

        **Abstract**:

        In domains where sample sizes are limited, efficient learning algorithms are critical. Learning using privileged information (LuPI) offers increased sample efficiency by allowing prediction models access to auxiliary information at training time which is unavailable when the models are used. In recent work, it was shown that for prediction in linear-Gaussian dynamical systems, a LuPI learner with access to intermediate time series data is never worse and often better in expectation than any unbiased classical learner. We provide new insights into this analysis and generalize it to nonlinear prediction tasks in latent dynamical systems, extending theoretical guarantees to the case where the map connecting latent variables and observations is known up to a linear transform. In addition, we propose algorithms based on random features and representation learning for the case when this map is unknown. A suite of empirical results confirm theoretical findings and show the potential of using privileged time-series information in nonlinear prediction.

        ----

        ## [1384] Layer Freezing & Data Sieving: Missing Pieces of a Generic Framework for Sparse Training

        **Authors**: *Geng Yuan, Yanyu Li, Sheng Li, Zhenglun Kong, Sergey Tulyakov, Xulong Tang, Yanzhi Wang, Jian Ren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/794a425a2e47e05d29d30f79b79a692d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/794a425a2e47e05d29d30f79b79a692d-Abstract-Conference.html)

        **Abstract**:

        Recently, sparse training has emerged as a promising paradigm for efficient deep learning on edge devices. The current research mainly devotes the efforts to reducing training costs by further increasing model sparsity. However, increasing sparsity is not always ideal since it will inevitably introduce severe accuracy degradation at an extremely high sparsity level. This paper intends to explore other possible directions to effectively and efficiently reduce sparse training costs while preserving accuracy. To this end, we investigate two techniques, namely, layer freezing and data sieving. First, the layer freezing approach has shown its success in dense model training and fine-tuning, yet it has never been adopted in the sparse training domain. Nevertheless, the unique characteristics of sparse training may hinder the incorporation of layer freezing techniques. Therefore, we analyze the feasibility and potentiality of using the layer freezing technique in sparse training and find it has the potential to save considerable training costs. Second, we propose a data sieving method for dataset-efficient training, which further reduces training costs by ensuring only a partial dataset is used throughout the entire training process. We show that both techniques can be well incorporated into the sparse training algorithm to form a generic framework, which we dub SpFDE. Our extensive experiments demonstrate that SpFDE can significantly reduce training costs while preserving accuracy from three dimensions: weight sparsity, layer freezing, and dataset sieving. Our code and models will be released.

        ----

        ## [1385] ETAB: A Benchmark Suite for Visual Representation Learning in Echocardiography

        **Authors**: *Ahmed M. Alaa, Anthony Philippakis, David A. Sontag*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/796501434d0dc3a039d5b91261f7f889-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/796501434d0dc3a039d5b91261f7f889-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Echocardiography is one of the most commonly used diagnostic imaging modalities in cardiology. Application of deep learning models to echocardiograms can enable automated identification of cardiac structures, estimation of cardiac function, and prediction of clinical outcomes. However, a major hindrance to realizing the full potential of deep learning is the lack of large-scale, fully curated and annotated data sets required for supervised training. High-quality pre-trained representations that can transfer useful visual features of echocardiograms to downstream tasks can help adapt deep learning models to new setups using fewer examples. In this paper, we design a suite of benchmarks that can be used to pre-train and evaluate echocardiographic representations with respect to various clinically-relevant tasks using publicly accessible data sets. In addition, we develop a unified evaluation protocol---which we call the echocardiographic task adaptation benchmark (ETAB)---that measures how well a visual representation of echocardiograms generalizes to common downstream tasks of interest. We use our benchmarking framework to evaluate state-of-the-art vision modeling pipelines. We envision that our standardized, publicly accessible benchmarks would encourage future research and expedite progress in applying deep learning to high-impact problems in cardiovascular medicine.

        ----

        ## [1386] Trustworthy Monte Carlo

        **Authors**: *Juha Harviainen, Mikko Koivisto, Petteri Kaski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/798fd0fd7f00f77bdbd7c19ea329bf6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/798fd0fd7f00f77bdbd7c19ea329bf6b-Abstract-Conference.html)

        **Abstract**:

        Monte Carlo integration is a key technique for designing randomized approximation schemes for counting problems, with applications, e.g., in machine learning and statistical physics. The technique typically enables massively parallel computation, however, with the risk that some of the delegated computations contain spontaneous or adversarial errors. We present an orchestration of the computations such that the outcome is accompanied with a proof of correctness that can be verified with substantially less computational resources than it takes to run the computations from scratch with state-of-the-art algorithms. Specifically, we adopt an algebraic proof system developed in computational complexity theory, in which the proof is represented by a polynomial; evaluating the polynomial at a random point amounts to a verification of the proof with probabilistic guarantees. We give examples of known Monte Carlo estimators that admit verifiable extensions with moderate computational overhead: for the permanent of zero--one matrices, for the model count of disjunctive normal form formulas, and for the gradient of logistic regression models. We also discuss the prospects and challenges of engineering efficient verifiable approximation schemes more generally.

        ----

        ## [1387] Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity

        **Authors**: *Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79a0c8e7ae8e403e39341ea6b0ba4c21-Abstract-Conference.html)

        **Abstract**:

        In response to subtle adversarial examples flipping classifications of neural network models, recent research has promoted certified robustness as a solution. There, invariance of predictions to all norm-bounded attacks is achieved through randomised smoothing of network inputs. Today's state-of-the-art certifications make optimal use of the class output scores at the input instance under test: no better radius of certification (under the $L_2$ norm) is possible given only these score. However, it is an open question as to whether such lower bounds can be improved using local information around the instance under test.  In this work, we demonstrate how today's ``optimal'' certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than $80 \%$ of Tiny-Imagenet instances, yielding an on average $5\%$ increase in the associated certification. When incorporating training time processes that enhance the certified radius, our technique shows even more promising results, with a uniform $4$ percentage point increase in the achieved certified radius.

        ----

        ## [1388] An $\alpha$-No-Regret Algorithm For Graphical Bilinear Bandits

        **Authors**: *Geovani Rizk, Igor Colin, Albert Thomas, Rida Laraki, Yann Chevaleyre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79a10a4977d1e21c319060e125406bd6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79a10a4977d1e21c319060e125406bd6-Abstract-Conference.html)

        **Abstract**:

        We propose the first regret-based approach to the \emph{Graphical Bilinear Bandits} problem, where $n$ agents in a graph play a stochastic bilinear bandit game with each of their neighbors. This setting reveals a combinatorial NP-hard problem that prevents the use of any existing regret-based algorithm in the (bi-)linear bandit literature. In this paper, we fill this gap and present the first regret-based algorithm for graphical bilinear bandits using the principle of optimism in the face of uncertainty. Theoretical analysis of this new method yields an upper bound of $\tilde{O}(\sqrt{T})$ on the $\alpha$-regret and evidences the impact of the graph structure on the rate of convergence. Finally, we show through various experiments the validity of our approach.

        ----

        ## [1389] A Unified Analysis of Federated Learning with Arbitrary Client Participation

        **Authors**: *Shiqiang Wang, Mingyue Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79ba1b827d3fc58e129d1cbfc8ff69f2-Abstract-Conference.html)

        **Abstract**:

        Federated learning (FL) faces challenges of intermittent client availability and computation/communication efficiency. As a result, only a small subset of clients can participate in FL at a given time. It is important to understand how partial client participation affects convergence, but most existing works have either considered idealized participation patterns or obtained results with non-zero optimality error for generic patterns. In this paper, we provide a unified convergence analysis for FL with arbitrary client participation. We first introduce a generalized version of federated averaging (FedAvg) that amplifies parameter updates at an interval of multiple FL rounds. Then, we present a novel analysis that captures the effect of client participation in a single term. By analyzing this term, we obtain convergence upper bounds for a wide range of participation patterns, including both non-stochastic and stochastic cases, which match either the lower bound of stochastic gradient descent (SGD) or the state-of-the-art results in specific settings. We also discuss various insights, recommendations, and experimental results.

        ----

        ## [1390] Deconfounded Representation Similarity for Comparison of Neural Networks

        **Authors**: *Tianyu Cui, Yogesh Kumar, Pekka Marttinen, Samuel Kaski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79cbf4f96c2bcc67267421154da689dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79cbf4f96c2bcc67267421154da689dd-Abstract-Conference.html)

        **Abstract**:

        Similarity metrics such as representational similarity analysis (RSA) and centered kernel alignment (CKA) have been used to understand neural networks by comparing their layer-wise representations. However, these metrics are confounded by the population structure of data items in the input space, leading to inconsistent conclusions about the \emph{functional} similarity between neural networks, such as spuriously high similarity of completely random neural networks and inconsistent domain relations in transfer learning. We introduce a simple and generally applicable fix to adjust for the confounder with covariate adjustment regression, which improves the ability of CKA and RSA to reveal functional similarity and also retains the intuitive invariance properties of the original similarity measures. We show that deconfounding the similarity metrics increases the resolution of detecting functionally similar neural networks across domains. Moreover, in real-world applications, deconfounding improves the consistency between CKA and domain similarity in transfer learning, and increases the correlation between CKA and model out-of-distribution accuracy similarity.

        ----

        ## [1391] Fairness without Demographics through Knowledge Distillation

        **Authors**: *Junyi Chai, Taeuk Jang, Xiaoqian Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79dc391a2c1067e9ac2b764e31a60377-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79dc391a2c1067e9ac2b764e31a60377-Abstract-Conference.html)

        **Abstract**:

        Most of existing work on fairness assumes available demographic information in the training set. In practice, due to legal or privacy concerns, when demographic information is not available in the training set, it is crucial to find alternative objectives to ensure fairness. Existing work on fairness without demographics follows Rawlsian Max-Min fairness objectives. However, such constraints could be too strict to improve group fairness, and could lead to a great decrease in accuracy. In light of these limitations, in this paper, we propose to solve the problem from a new perspective, i.e., through knowledge distillation. Our method uses soft label from an overfitted teacher model as an alternative, and we show from preliminary experiments that soft labelling is beneficial for improving fairness. We analyze theoretically the fairness of our method, and we show that our method can be treated as an error-based reweighing. Experimental results on three datasets show that our method outperforms state-of-the-art alternatives, with notable improvements in group fairness and with relatively small decrease in accuracy.

        ----

        ## [1392] Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch

        **Authors**: *Hossein Souri, Liam Fowl, Rama Chellappa, Micah Goldblum, Tom Goldstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/79eec295a3cd5785e18c61383e7c996b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/79eec295a3cd5785e18c61383e7c996b-Abstract-Conference.html)

        **Abstract**:

        As the curation of data for machine learning becomes increasingly automated, dataset tampering is a mounting threat.  Backdoor attackers tamper with training data to embed a vulnerability in models that are trained on that data. This vulnerability is then activated at inference time by placing a "trigger'' into the model's input. Typical backdoor attacks insert the trigger directly into the training data, although the presence of such an attack may be visible upon inspection. In contrast, the Hidden Trigger Backdoor Attack achieves poisoning without placing a trigger into the training data at all.   However, this hidden trigger attack is ineffective at poisoning neural networks trained from scratch.  We develop a new hidden trigger attack,  Sleeper Agent, which employs gradient matching, data selection, and target model re-training during the crafting process.  Sleeper Agent is the first hidden trigger backdoor attack to be effective against neural networks trained from scratch. We demonstrate its effectiveness on ImageNet and in black-box settings. Our implementation code can be found at: https://github.com/hsouri/Sleeper-Agent.

        ----

        ## [1393] Dynamic Pricing with Monotonicity Constraint under Unknown Parametric Demand Model

        **Authors**: *Su Jia, Andrew A. Li, R. Ravi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a0f8055c838df8e62329a76c7c6403d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a0f8055c838df8e62329a76c7c6403d-Abstract-Conference.html)

        **Abstract**:

        We consider the Continuum Bandit problem where the goal is to find the optimal action under an unknown reward function, with an additional monotonicity constraint (or, "markdown" constraint) that requires that the action sequence be non-increasing. This problem faithfully models a natural single-product dynamic pricing problem, called "markdown pricing", where the objective is to adaptively reduce the price over a finite sales horizon to maximize expected revenues. Jia et al '21 and Chen '21 independently showed a tight $T^{3/4}$ regret bound over $T$ rounds under *minimal* assumptions of unimodality and Lipschitzness in the reward (or, "revenue") function. This bound shows that the demand learning in markdown pricing is harder than unconstrained (i.e., without the monotonicity constraint) pricing under unknown demand which suffers regret only of the order of $T^{2/3}$ under the same assumptions (Kleinberg '04). However, in practice the demand functions are usually assumed to have certain functional forms (e.g. linear or exponential), rendering the demand-learning easier and suggesting lower regret bounds. We investigate two fundamental questions, assuming the underlying demand curve comes from a given parametric family: (1) Can we improve the $T^{3/4}$ regret bound for markdown pricing, under extra assumptions on the functional forms of the demand functions? (2) Is markdown pricing still harder than unconstrained pricing, under these additional assumptions? To answer these, we introduce a concept called markdown dimension that measures the complexity of the parametric family and present tight regret bounds under this framework, thereby completely settling the aforementioned questions.

        ----

        ## [1394] A Win-win Deal: Towards Sparse and Robust Pre-trained Language Models

        **Authors**: *Yuanxin Liu, Fandong Meng, Zheng Lin, Jiangnan Li, Peng Fu, Yanan Cao, Weiping Wang, Jie Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a27143ea615262a0c122eb179c9b7a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a27143ea615262a0c122eb179c9b7a6-Abstract-Conference.html)

        **Abstract**:

        Despite the remarkable success of pre-trained language models (PLMs), they still face two challenges: First, large-scale PLMs are inefficient in terms of memory footprint and computation. Second, on the downstream tasks, PLMs tend to rely on the dataset bias and struggle to generalize to out-of-distribution (OOD) data. In response to the efficiency problem, recent studies show that dense PLMs can be replaced with sparse subnetworks without hurting the performance. Such subnetworks can be found in three scenarios: 1) the fine-tuned PLMs, 2) the raw PLMs and then fine-tuned in isolation, and even inside 3) PLMs without any parameter fine-tuning. However, these results are only obtained in the in-distribution (ID) setting. In this paper, we extend the study on PLMs subnetworks to the OOD setting, investigating whether sparsity and robustness to dataset bias can be achieved simultaneously. To this end, we conduct extensive experiments with the pre-trained BERT model on three natural language understanding (NLU) tasks. Our results demonstrate that \textbf{sparse and robust subnetworks (SRNets) can consistently be found in BERT}, across the aforementioned three scenarios, using different training and compression methods. Furthermore, we explore the upper bound of SRNets using the OOD information and show that \textbf{there exist sparse and almost unbiased BERT subnetworks}. Finally, we present 1) an analytical study that provides insights on how to promote the efficiency of SRNets searching process and 2) a solution to improve subnetworks' performance at high sparsity. The code is available at \url{https://github.com/llyx97/sparse-and-robust-PLM}.

        ----

        ## [1395] Asymptotics of smoothed Wasserstein distances in the small noise regime

        **Authors**: *Yunzi Ding, Jonathan Niles-Weed*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a41a2bc087b6d1981235d1718e53f51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a41a2bc087b6d1981235d1718e53f51-Abstract-Conference.html)

        **Abstract**:

        We study the behavior of the Wasserstein-$2$ distance between discrete measures $\mu$ and $\nu$ in $\mathbb{R}^d$ when both measures are smoothed by small amounts of Gaussian noise. This procedure, known as Gaussian-smoothed optimal transport, has recently attracted attention as a statistically attractive alternative to the unregularized Wasserstein distance. We give precise bounds on the approximation properties of this proposal in the small noise regime, and establish the existence of a phase transition: we show that, if the optimal transport plan from $\mu$ to $\nu$ is unique and a perfect matching, there exists a critical threshold such that the difference between $W_2(\mu, \nu)$ and the Gaussian-smoothed OT distance $W_2(\mu \ast \mathcal{N}_\sigma, \nu\ast \mathcal{N}_\sigma)$ scales like $\exp(-c /\sigma^2)$ for $\sigma$ below the threshold, and scales like $\sigma$ above it. These results establish that for $\sigma$ sufficiently small, the smoothed Wasserstein distance approximates the unregularized distance exponentially well.

        ----

        ## [1396] Fine-tuning Language Models over Slow Networks using Activation Quantization with Guarantees

        **Authors**: *Jue Wang, Binhang Yuan, Luka Rimanic, Yongjun He, Tri Dao, Beidi Chen, Christopher Ré, Ce Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a43b8eb92cd5f652b78eeee3fb6f910-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a43b8eb92cd5f652b78eeee3fb6f910-Abstract-Conference.html)

        **Abstract**:

        Communication compression is a crucial technique for modern distributed learning systems to alleviate their communication bottlenecks over slower networks. Despite recent intensive studies of gradient compression for data parallel-style training, compressing the activations for models trained with pipeline parallelism is still an open problem. In this paper, we propose AQ-SGD, a novel activation compression algorithm for communication-efficient pipeline parallelism training over slow networks. Different from previous efforts in activation compression, instead of compressing activation values directly, AQ-SGD compresses the changes of the activations. This allows us to show, to the best of our knowledge for the first time, that one can still achieve $O(1/\sqrt{T})$ convergence rate for non-convex objectives under activation compression, without making assumptions on gradient unbiasedness that do not hold for deep learning models with non-linear activation functions. We then show that AQ-SGD can be optimized and implemented efficiently, without additional end-to-end runtime overhead. We evaluated AQ-SGD to fine-tune language models with up to 1.5 billion parameters, compressing activation to 2-4 bits. AQ-SGD provides up to $4.3\times$ end-to-end speed-up in slower networks, without sacrificing model quality. Moreover, we also show that AQ-SGD can be combined with state-of-the-art gradient compression algorithms to enable end-to-end communication compression: All communications between machines, including model gradients, forward activations, and backward gradients are compressed into lower precision. This provides up to $4.9\times$ end-to-end speed-up, without sacrificing model quality.

        ----

        ## [1397] Pareto Set Learning for Expensive Multi-Objective Optimization

        **Authors**: *Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, Qingfu Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract-Conference.html)

        **Abstract**:

        Expensive multi-objective optimization problems can be found in many real-world applications, where their objective function evaluations involve expensive computations or physical experiments. It is desirable to obtain an approximate Pareto front with a limited evaluation budget. Multi-objective Bayesian optimization (MOBO) has been widely used for finding a finite set of Pareto optimal solutions. However, it is well-known that the whole Pareto set is on a continuous manifold and can contain infinite solutions. The structural properties of the Pareto set are not well exploited in existing MOBO methods, and the finite-set approximation may not contain the most preferred solution(s) for decision-makers. This paper develops a novel learning-based method to approximate the whole Pareto set for MOBO, which generalizes the decomposition-based multi-objective optimization algorithm (MOEA/D) from finite populations to models. We design a simple and powerful acquisition search method based on the learned Pareto set, which naturally supports batch evaluation. In addition, with our proposed model, decision-makers can readily explore any trade-off area in the approximate Pareto set for flexible decision-making. This work represents the first attempt to model the Pareto set for expensive multi-objective optimization. Experimental results on different synthetic and real-world problems demonstrate the effectiveness of our proposed method.

        ----

        ## [1398] Non-monotonic Resource Utilization in the Bandits with Knapsacks Problem

        **Authors**: *Raunak Kumar, Robert Kleinberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a62d9a4c03377d1175b8859b4cc16d4-Abstract-Conference.html)

        **Abstract**:

        Bandits with knapsacks (BwK) is an influential model of sequential decision-making under uncertainty that incorporates resource consumption constraints. In each round, the decision-maker observes an outcome consisting of a reward and a vector of nonnegative resource consumptions, and the budget of each resource is decremented by its consumption. In this paper we introduce a natural generalization of the stochastic BwK problem that allows non-monotonic resource utilization. In each round, the decision-maker observes an outcome consisting of a reward and a vector of resource drifts that can be positive, negative or zero, and the budget of each resource is incremented by its drift. Our main result is a Markov decision process (MDP) policy that has constant regret against a linear programming (LP) relaxation when the decision-maker knows the true outcome distributions. We build upon this to develop a learning algorithm that has logarithmic regret against the same LP relaxation when the decision-maker does not know the true outcome distributions. We also present a reduction from BwK to our model that shows our regret bound matches existing results.

        ----

        ## [1399] Efficient identification of informative features in simulation-based inference

        **Authors**: *Jonas Beck, Michael Deistler, Yves Bernaerts, Jakob H. Macke, Philipp Berens*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html)

        **Abstract**:

        Simulation-based Bayesian inference (SBI) can be used to estimate the parameters of complex mechanistic models given observed model outputs without requiring access to explicit likelihood evaluations. A prime example for the application of SBI in neuroscience involves estimating the parameters governing the response dynamics of Hodgkin-Huxley (HH) models from electrophysiological measurements, by inferring a posterior over the parameters that is consistent with a set of observations. To this end, many SBI methods employ a set of summary statistics or scientifically interpretable features to estimate a surrogate likelihood or posterior. However, currently, there is no way to identify how much each summary statistic or feature contributes to reducing posterior uncertainty. To address this challenge, one could simply compare the posteriors with and without a given feature included in the inference process. However, for large or nested feature sets, this would necessitate repeatedly estimating the posterior, which is computationally expensive or even prohibitive. Here, we provide a more efficient approach based on the SBI method neural likelihood estimation (NLE): We show that one can marginalize the trained surrogate likelihood post-hoc before inferring the posterior to assess the contribution of a feature. We demonstrate the usefulness of our method by identifying the most important features for inferring parameters of an example HH neuron model. Beyond neuroscience, our method is generally applicable to SBI workflows that rely on data features for inference used in other scientific fields.

        ----

        

[Go to the previous page](NIPS-2022-list06.md)

[Go to the next page](NIPS-2022-list08.md)

[Go to the catalog section](README.md)