## [0] Collaborative Score Distillation for Consistent Visual Editing

**Authors**: *Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7fd2c0a1a6f956c94024e955b34cc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7fd2c0a1a6f956c94024e955b34cc43-Abstract-Conference.html)

**Abstract**:

Generative priors of large-scale text-to-image diffusion models enable a wide range of new generation and editing applications on diverse visual modalities. However, when adapting these priors to complex visual modalities, often represented as multiple images (e.g., video or 3D scene), achieving consistency across a set of images is challenging. In this paper, we address this challenge with a novel method, Collaborative Score Distillation (CSD). CSD is based on the Stein Variational Gradient Descent (SVGD). Specifically, we propose to consider multiple samples as “particles” in the SVGD update and combine their score functions to distill generative priors over a set of images synchronously. Thus, CSD facilitates the seamless integration of information across 2D images, leading to a consistent visual synthesis across multiple samples. We show the effectiveness of CSD in a variety of editing tasks, encompassing the visual editing of panorama images, videos, and 3D scenes. Our results underline the competency of CSD as a versatile method for enhancing inter-sample consistency, thereby broadening the applicability of text-to-image diffusion models.

----

## [0] FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout

**Authors**: *Irene Wang, Prashant J. Nair, Divya Mahajan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) allows machine learning models to train locally on individual mobile devices, synchronizing model updates via a shared server. This approach safeguards user privacy; however, it also generates a heterogeneous training environment due to the varying performance capabilities across devices. As a result, “straggler” devices with lower performance often dictate the overalltraining time in FL. In this work, we aim to alleviate this performance bottleneck due to stragglers by dynamically balancing the training load across the system. We introduce Invariant Dropout, a method that extracts a sub-model based on the weight update threshold, thereby minimizing potential impacts on accuracy. Building on this dropout technique, we develop an adaptive training framework, Federated Learning using Invariant Dropout (FLuID). FLuID offers a lightweight sub-model extraction to regulate computational intensity, thereby reducing the load on straggler devices without affecting model quality. Our method leverages neuron updates from non-straggler devices to construct a tailored sub-model for each straggler based on client performance profiling. Furthermore, FLuID can dynamically adapt to changes in stragglers as runtime conditions shift. We evaluate FLuID using five real-world mobile clients. The evaluations show that Invariant Dropout maintains baseline model efficiency while alleviating the performance bottleneck of stragglers through a dynamic, runtime approach.

----

## [0] Learning to Augment Distributions for Out-of-distribution Detection

**Authors**: *Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e812af67a942c21dd0104bd929f99da1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e812af67a942c21dd0104bd929f99da1-Abstract-Conference.html)

**Abstract**:

Open-world classification systems should discern out-of-distribution (OOD) data whose labels deviate from those of in-distribution (ID) cases, motivating recent studies in OOD detection. Advanced works, despite their promising progress, may still fail in the open world, owing to the lacking knowledge about unseen OOD data in advance. Although one can access auxiliary OOD data (distinct from unseen ones) for model training, it remains to analyze how such auxiliary data will work in the open world. To this end, we delve into such a problem from a learning theory perspective, finding that the distribution discrepancy between the auxiliary and the unseen real OOD data is the key to affect the open-world detection performance. Accordingly, we propose Distributional-Augmented OOD Learning (DAOL), alleviating the OOD distribution discrepancy by crafting an OOD distribution set that contains all distributions in a Wasserstein ball centered on the auxiliary OOD distribution. We justify that the predictor trained over the worst OOD data in the ball can shrink the OOD distribution discrepancy, thus improving the open-world detection performance given only the auxiliary OOD data. We conduct extensive evaluations across representative OOD detection setups, demonstrating the superiority of our DAOL over its advanced counterparts.

----

## [0] Covariance-adaptive best arm identification

**Authors**: *El Mehdi Saad, Gilles Blanchard, Nicolas Verzelen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e82ef7865f29b40640f486bbbe7959a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e82ef7865f29b40640f486bbbe7959a7-Abstract-Conference.html)

**Abstract**:

We consider the problem of best arm identification in the multi-armed bandit model, under fixed confidence. Given a confidence input $\delta$, the goal is to identify the arm with the highest mean reward with a probability of at least $1 - \delta$, while minimizing the number of arm pulls. While the literature provides solutions to this problem under the assumption of independent arms distributions, we propose a more flexible scenario where arms can be dependent and rewards can be sampled simultaneously. This framework allows the learner to estimate the covariance among the arms distributions, enabling a more efficient identification of the best arm. The relaxed setting we propose is relevant in various applications, such as clinical trials, where similarities between patients or drugs suggest underlying correlations in the outcomes. We introduce new algorithms that adapt to the unknown covariance of the arms and demonstrate through theoretical guarantees that substantial improvement can be achieved over the standard setting. Additionally, we provide new lower bounds for the relaxed setting and present numerical simulations that support their theoretical findings.

----

## [0] What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation

**Authors**: *Benedikt Blumenstiel, Johannes Jakubik, Hilde Kühne, Michael Vössing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e82f45e480f5f44d696ba15dad88f9a3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e82f45e480f5f44d696ba15dad88f9a3-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While semantic segmentation has seen tremendous improvements in the past, there are still significant labeling efforts necessary and the problem of limited generalization to classes that have not been present during training. To address this problem, zero-shot semantic segmentation makes use of large self-supervised vision-language models, allowing zero-shot transfer to unseen classes. In this work, we build a benchmark for Multi-domain Evaluation of Zero-Shot Semantic Segmentation (MESS), which allows a holistic analysis of performance across a wide range of domain-specific datasets such as medicine, engineering, earth monitoring, biology, and agriculture. To do this, we reviewed 120 datasets, developed a taxonomy, and classified the datasets according to the developed taxonomy. We select a representative subset consisting of 22 datasets and propose it as the MESS benchmark. We evaluate eight recently published models on the proposed MESS benchmark and analyze characteristics for the performance of zero-shot transfer models. The toolkit is available at https://github.com/blumenstiel/MESS.

----

## [0] Swarm Reinforcement Learning for Adaptive Mesh Refinement

**Authors**: *Niklas Freymuth, Philipp Dahlinger, Tobias Würth, Simon Reisch, Luise Kärger, Gerhard Neumann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html)

**Abstract**:

The Finite Element Method, an important technique in engineering, is aided by Adaptive Mesh Refinement (AMR), which dynamically refines mesh regions to allow for a favorable trade-off between computational speed and simulation accuracy. Classical methods for AMR depend on task-specific heuristics or expensive error estimators, hindering their use for complex simulations. Recent learned AMR methods tackle these problems, but so far scale only to simple toy examples. We formulate AMR as a novel Adaptive Swarm Markov Decision Process in which a mesh is modeled as a system of simple collaborating agents that may split into multiple new agents. This framework allows for a spatial reward formulation that simplifies the credit assignment problem, which we combine with Message Passing Networks to propagate information between neighboring mesh elements. We experimentally validate the effectiveness of our approach, Adaptive Swarm Mesh Refinement (ASMR), showing that it learns reliable, scalable, and efficient refinement strategies on a set of challenging problems. Our approach significantly speeds up computation, achieving up to 30-fold improvement compared to uniform refinements in complex simulations. Additionally, we outperform learned baselines and achieve a refinement quality that is on par with a traditional error-based AMR strategy without expensive oracle information about the error signal.

----

## [0] Fast Projected Newton-like Method for Precision Matrix Estimation under Total Positivity

**Authors**: *Jianfeng Cai, José Vinícius de Miranda Cardoso, Daniel P. Palomar, Jiaxi Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e878c8f38381d0964677fb9536c494ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e878c8f38381d0964677fb9536c494ee-Abstract-Conference.html)

**Abstract**:

We study the problem of estimating precision matrices in Gaussian distributions that are multivariate totally positive of order two ($\mathrm{MTP}_2$). The precision matrix in such a distribution is an M-matrix. This problem can be formulated as a sign-constrained log-determinant program. Current algorithms are designed using the block coordinate descent method or the proximal point algorithm, which becomes computationally challenging in high-dimensional cases due to the requirement to solve numerous nonnegative quadratic programs or large-scale linear systems. To address this issue, we propose a novel algorithm based on the two-metric projection method, incorporating a carefully designed search direction and variable partitioning scheme. Our algorithm substantially reduces computational complexity, and its theoretical convergence is established. Experimental results on synthetic and real-world datasets demonstrate that our proposed algorithm provides a significant improvement in computational efficiency compared to the state-of-the-art methods.

----

## [0] BanditPAM++: Faster k-medoids Clustering

**Authors**: *Mo Tiwari, Ryan Kang, Donghyun Lee, Sebastian Thrun, Ilan Shomorony, Martin J. Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e885e5bc6e13b9dd8f80bc5482b1fa2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e885e5bc6e13b9dd8f80bc5482b1fa2f-Abstract-Conference.html)

**Abstract**:

Clustering is a fundamental task in data science with wide-ranging applications. In $k$-medoids clustering, cluster centers must be actual datapoints and arbitrary distance metrics may be used; these features allow for greater interpretability of the cluster centers and the clustering of exotic objects in $k$-medoids clustering, respectively. $k$-medoids clustering has recently grown in popularity due to the discovery of more efficient $k$-medoids algorithms. In particular, recent research has proposed BanditPAM, a randomized $k$-medoids algorithm with state-of-the-art complexity and clustering accuracy. In this paper, we present BanditPAM++, which accelerates BanditPAM via two algorithmic improvements, and is $O(k)$ faster than BanditPAM in complexity and substantially faster than BanditPAM in wall-clock runtime. First, we demonstrate that BanditPAM has a special structure that allows the reuse of clustering information $\textit{within}$ each iteration. Second, we demonstrate that BanditPAM has additional structure that permits the reuse of information $\textit{across}$ different iterations. These observations inspire our proposed algorithm, BanditPAM++, which returns the same clustering solutions as BanditPAM but often several times faster. For example, on the CIFAR10 dataset, BanditPAM++ returns the same results as BanditPAM but runs over 10$\times$ faster. Finally, we provide a high-performance C++ implementation of BanditPAM++, callable from Python and R, that may be of interest to practitioners at https://github.com/motiwari/BanditPAM. Auxiliary code to reproduce all of our experiments via a one-line script is available at https://github.com/ThrunGroup/BanditPAM_plusplus_experiments.

----

## [0] Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks

**Authors**: *Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, Jordan Terry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8916198466e8ef218a2185a491b49fa-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8916198466e8ef218a2185a491b49fa-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present the Minigrid and Miniworld libraries which provide a suite of goal-oriented 2D and 3D environments. The libraries were explicitly created with a minimalistic design paradigm to allow users to rapidly develop new environments for a wide range of research-specific needs. As a result, both have received widescale adoption by the RL community, facilitating research in a wide range of areas. In this paper, we outline the design philosophy, environment details, and their world generation API.  We also showcase the additional capabilities brought by the unified API between Minigrid and Miniworld through case studies on transfer learning (for both RL agents and humans) between the different observation spaces. The source code of Minigrid and Miniworld can be found at https://github.com/Farama-Foundation/Minigrid and https://github.com/Farama-Foundation/Miniworld along with their documentation at https://minigrid.farama.org/ and https://miniworld.farama.org/.

----

## [0] Cross-Domain Policy Adaptation via Value-Guided Data Filtering

**Authors**: *Kang Xu, Chenjia Bai, Xiaoteng Ma, Dong Wang, Bin Zhao, Zhen Wang, Xuelong Li, Wei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ad87f1076fb0f75d89a45828f186b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ad87f1076fb0f75d89a45828f186b0-Abstract-Conference.html)

**Abstract**:

Generalizing policies across different domains with dynamics mismatch poses a significant challenge in reinforcement learning. For example, a robot learns the policy in a simulator, but when it is deployed in the real world, the dynamics of the environment may be different. Given the source and target domain with dynamics mismatch, we consider the online dynamics adaptation problem, in which case the agent can access sufficient source domain data while online interactions with the target domain are limited. Existing research has attempted to solve the problem from the dynamics discrepancy perspective. In this work, we reveal the limitations of these methods and explore the problem from the value difference perspective via a novel insight on the value consistency across domains. Specifically, we present the Value-Guided Data Filtering (VGDF) algorithm, which selectively shares transitions from the source domain based on the proximity of paired value targets across the two domains. Empirical results on various environments with kinematic and morphology shifts demonstrate that our method achieves superior performance compared to prior approaches.

----

## [0] Connecting Certified and Adversarial Training

**Authors**: *Yuhao Mao, Mark Niklas Müller, Marc Fischer, Martin T. Vechev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8b0c97b34fdaf58b2f48f8cca85e76a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8b0c97b34fdaf58b2f48f8cca85e76a-Abstract-Conference.html)

**Abstract**:

Training certifiably robust neural networks remains a notoriously hard problem.While adversarial training optimizes under-approximations of the worst-case loss, which leads to insufficient regularization for certification, sound certified training methods, optimize loose over-approximations, leading to over-regularization and poor (standard) accuracy.In this work, we propose TAPS, an (unsound) certified training method that combines IBP and PGD training to optimize more precise, although not necessarily sound, worst-case loss approximations, reducing over-regularization and increasing certified and standard accuracies.Empirically, TAPS achieves a new state-of-the-art in many settings, e.g., reaching a certified accuracy of $22$% on TinyImageNet for $\ell_\infty$-perturbations with radius $\epsilon=1/255$. We make our implementation and networks public at https://github.com/eth-sri/taps.

----

## [0] Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning

**Authors**: *Akhil Bagaria, Ben Abbatematteo, Omer Gottesman, Matt Corsaro, Sreehari Rammohan, George Dimitri Konidaris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html)

**Abstract**:

An agent learning an option in hierarchical reinforcement learning must solve three problems: identify the option's subgoal (termination condition), learn a policy, and learn where that policy will succeed (initiation set). The termination condition is typically identified first, but the option policy and initiation set must be learned simultaneously, which is challenging because the initiation set depends on the option policy, which changes as the agent learns. Consequently, data obtained from option execution becomes invalid over time, leading to an inaccurate initiation set that subsequently harms downstream task performance. We highlight three issues---data non-stationarity, temporal credit assignment, and pessimism---specific to learning initiation sets, and propose to address them using tools from off-policy value estimation and classification. We show that our method learns higher-quality initiation sets faster than existing methods (in MiniGrid and Montezuma's Revenge), can automatically discover promising grasps for robot manipulation (in Robosuite), and improves the performance of a state-of-the-art option discovery method in a challenging maze navigation task in MuJoCo.

----

## [0] Alignment with human representations supports robust few-shot learning

**Authors**: *Ilia Sucholutsky, Tom Griffiths*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ddc03b001d4c4b44b29bc1167e7fdd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ddc03b001d4c4b44b29bc1167e7fdd-Abstract-Conference.html)

**Abstract**:

Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both natural adversarial attacks and domain shifts. Our results suggest that human-alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well.

----

## [0] ReMaX: Relaxing for Better Training on Efficient Panoptic Segmentation

**Authors**: *Shuyang Sun, Weijun Wang, Andrew G. Howard, Qihang Yu, Philip H. S. Torr, Liang-Chieh Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8e30fda5ab87ea93360a36288ac0145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8e30fda5ab87ea93360a36288ac0145-Abstract-Conference.html)

**Abstract**:

This paper presents a new mechanism to facilitate the training of mask transformers for efficient panoptic segmentation, democratizing its deployment. We observe that due to the high complexity in the training objective of panoptic segmentation, it will inevitably lead to much higher penalization on false positive. Such unbalanced loss makes the training process of the end-to-end mask-transformer based architectures difficult, especially for efficient models. In this paper, we present ReMaX that adds relaxation to mask predictions and class predictions during the training phase for panoptic segmentation. We demonstrate that via these simple relaxation techniques during training, our model can be consistently improved by a clear margin without any extra computational cost on inference. By combining our method with efficient backbones like MobileNetV3-Small, our method achieves new state-of-the-art results for efficient panoptic segmentation on COCO, ADE20K and Cityscapes. Code and pre-trained checkpoints will be available at https://github.com/google-research/deeplab2.

----

## [0] The Behavior and Convergence of Local Bayesian Optimization

**Authors**: *Kaiwen Wu, Kyurae Kim, Roman Garnett, Jacob Gardner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html)

**Abstract**:

A recent development in Bayesian optimization is the use of local optimization strategies, which can deliver strong empirical performance on high-dimensional problems compared to traditional global strategies. The "folk wisdom" in the literature is that the focus on local optimization sidesteps the curse of dimensionality; however, little is known concretely about the expected behavior or convergence of Bayesian local optimization routines. We first study the behavior of the local approach, and find that the statistics of individual local solutions of Gaussian process sample paths are surprisingly good compared to what we would expect to recover from global methods. We then present the first rigorous analysis of such a Bayesian local optimization algorithm recently proposed by MÃ¼ller et al. (2021), and derive convergence rates in both the noisy and noiseless settings.

----

## [0] Contrastive Sampling Chains in Diffusion Models

**Authors**: *Junyu Zhang, Daochang Liu, Shichao Zhang, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ff788779f2e9e74ccd0d6b84607437-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ff788779f2e9e74ccd0d6b84607437-Abstract-Conference.html)

**Abstract**:

The past few years have witnessed great success in the use of diffusion models (DMs) to generate high-fidelity images with the help of stochastic differential equations (SDEs). However, discretization error is an inevitable limitation when utilizing numerical solvers to solve SDEs. To address this limitation, we provide a theoretical analysis demonstrating that an appropriate combination of the contrastive loss and score matching serves as an upper bound of the KL divergence between the true data distribution and the model distribution. To obtain this bound, we utilize a contrastive loss to construct a contrastive sampling chain to fine-tuning the pre-trained DM. In this manner, our method reduces the discretization error and thus yields a smaller gap between the true data distribution and our model distribution. Moreover, the presented method can be applied to fine-tuning various pre-trained DMs, both with or without fast sampling algorithms, contributing to better sample quality or slightly faster sampling speeds. To validate the efficacy of our method, we conduct comprehensive experiments. For example, on CIFAR10, when applied to a pre-trained EDM, our method improves the FID from 2.04 to 1.88 with 35 neural function evaluations (NFEs), and reduces NFEs from 35 to 25 to achieve the same 2.04 FID.

----

## [0] Effective Robustness against Natural Distribution Shifts for Models with Different Training Data

**Authors**: *Zhouxing Shi, Nicholas Carlini, Ananth Balashankar, Ludwig Schmidt, Cho-Jui Hsieh, Alex Beutel, Yao Qin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9000ecb86d45c442a1d38fae68dd8fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9000ecb86d45c442a1d38fae68dd8fb-Abstract-Conference.html)

**Abstract**:

``Effective robustness'' measures the extra out-of-distribution (OOD) robustness beyond what can be predicted from the in-distribution (ID) performance. Existing effective robustness evaluations typically use a single test set such as ImageNet to evaluate the ID accuracy. This becomes problematic when evaluating models trained on different data distributions, e.g., comparing models trained on ImageNet vs. zero-shot language-image pre-trained models trained on LAION. In this paper, we propose a new evaluation metric to evaluate and compare the effective robustness of models trained on different data. To do this, we control for the accuracy on multiple ID test sets that cover the training distributions for all the evaluated models. Our new evaluation metric provides a better estimate of effective robustness when there are models with different training data. It may also explain the surprising effective robustness gains of zero-shot CLIP-like models exhibited in prior works that used ImageNet as the only ID test set, while the gains diminish under our new evaluation. Additional artifacts including interactive visualizations are provided at https://shizhouxing.github.io/effective-robustness.

----

## [0] Tailoring Self-Attention for Graph via Rooted Subtrees

**Authors**: *Siyuan Huang, Yunchong Song, Jiayue Zhou, Zhouhan Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e90ba1fc564a69809d7391bf76a5f087-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e90ba1fc564a69809d7391bf76a5f087-Abstract-Conference.html)

**Abstract**:

Attention mechanisms have made significant strides in graph learning, yet they still exhibit notable limitations: local attention faces challenges in capturing long-range information due to the inherent problems of the message-passing scheme, while global attention cannot reflect the hierarchical neighborhood structure and fails to capture fine-grained local information. In this paper, we propose a novel multi-hop graph attention mechanism, named Subtree Attention (STA), to address the aforementioned issues. STA seamlessly bridges the fully-attentional structure and the rooted subtree, with theoretical proof that STA approximates the global attention under extreme settings. By allowing direct computation of attention weights among multi-hop neighbors, STA mitigates the inherent problems in existing graph attention mechanisms. Further we devise an efficient form for STA by employing kernelized softmax, which yields a linear time complexity. Our resulting GNN architecture, the STAGNN, presents a simple yet performant STA-based graph neural network leveraging a hop-aware attention strategy. Comprehensive evaluations on ten node classification datasets demonstrate that STA-based models outperform existing graph transformers and mainstream GNNs. The codeis available at https://github.com/LUMIA-Group/SubTree-Attention.

----

## [0] Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective

**Authors**: *Zeyuan Yin, Eric P. Xing, Zhiqiang Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html)

**Abstract**:

We present a new dataset condensation framework termed Squeeze, Recover and Relabel (SRe$^2$L) that decouples the bilevel optimization of model and synthetic data during training, to handle varying scales of datasets, model architectures and image resolutions for efficient dataset condensation. The proposed method demonstrates flexibility across diverse dataset scales and exhibits multiple advantages in terms of arbitrary resolutions of synthesized images, low training cost and memory consumption with high-resolution synthesis, and the ability to scale up to arbitrary evaluation network architectures. Extensive experiments are conducted on Tiny-ImageNet and full ImageNet-1K datasets. Under 50 IPC, our approach achieves the highest 42.5\% and 60.8\% validation accuracy on Tiny-ImageNet and ImageNet-1K, outperforming all previous state-of-the-art methods by margins of 14.5\% and 32.9\%, respectively. Our approach also surpasses MTT in terms of speed by approximately 52$\times$ (ConvNet-4) and 16$\times$ (ResNet-18) faster with less memory consumption of 11.6$\times$ and 6.4$\times$ during data synthesis. Our code and condensed datasets of 50, 200 IPC with 4K recovery budget are available at https://github.com/VILA-Lab/SRe2L.

----

## [0] Disentangled Wasserstein Autoencoder for T-Cell Receptor Engineering

**Authors**: *Tianxiao Li, Hongyu Guo, Filippo Grazioli, Mark Gerstein, Martin Renqiang Min*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html)

**Abstract**:

In protein biophysics, the separation between the functionally important residues (forming the active site or binding surface) and those that create the overall structure (the fold) is a well-established and fundamental concept. Identifying and modifying those functional sites is critical for protein engineering but computationally non-trivial, and requires significant domain knowledge. To automate this process from a data-driven perspective, we propose a disentangled Wasserstein autoencoder with an auxiliary classifier, which isolates the function-related patterns from the rest with theoretical guarantees. This enables one-pass protein sequence editing and improves the understanding of the resulting sequences and editing actions involved. To demonstrate its effectiveness, we apply it to T-cell receptors (TCRs), a well-studied structure-function case. We show that our method can be used to alter the function of TCRs without changing the structural backbone, outperforming several competing methods in generation quality and efficiency, and requiring only 10\% of the running time needed by baseline models. To our knowledge, this is the first approach that utilizes disentangled representations for TCR engineering.

----

## [0] Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser

**Authors**: *Yung-Hsuan Lai, Yen-Chun Chen, Frank Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95e9f0c127aa1cfa2628adb2f3cb107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95e9f0c127aa1cfa2628adb2f3cb107-Abstract-Conference.html)

**Abstract**:

Audio-visual learning has been a major pillar of multi-modal machine learning, where the community mostly focused on its $\textit{modality-aligned}$ setting, $\textit{i.e.}$, the audio and visual modality are $\textit{both}$ assumed to signal the prediction target.With the Look, Listen, and Parse dataset (LLP), we investigate the under-explored $\textit{unaligned}$ setting, where the goal is to recognize audio and visual events in a video with only weak labels observed.Such weak video-level labels only tell what events happen without knowing the modality they are perceived (audio, visual, or both).To enhance learning in this challenging setting, we incorporate large-scale contrastively pre-trained models as the modality teachers. A simple, effective, and generic method, termed $\textbf{V}$isual-$\textbf{A}$udio $\textbf{L}$abel Elab$\textbf{or}$ation (VALOR), is innovated to harvest modality labels for the training events.Empirical studies show that the harvested labels significantly improve an attentional baseline by $\textbf{8.0}$ in average F-score (Type@AV).Surprisingly, we found that modality-independent teachers outperform their modality-fused counterparts since they are noise-proof from the other potentially unaligned modality.Moreover, our best model achieves the new state-of-the-art on all metrics of LLP by a substantial margin ($\textbf{+5.4}$ F-score for Type@AV). VALOR is further generalized to Audio-Visual Event Localization and achieves the new state-of-the-art as well.

----

## [0] Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation

**Authors**: *Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95eb5206c867be843fbc14bbfe8c10e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95eb5206c867be843fbc14bbfe8c10e-Abstract-Conference.html)

**Abstract**:

This paper studies the problem of weakly open-vocabulary semantic segmentation (WOVSS), which learns to segment objects of arbitrary classes using mere image-text pairs. Existing works turn to enhance the vanilla vision transformer by introducing explicit grouping recognition, i.e., employing several group tokens/centroids to cluster the image tokens and perform the group-text alignment. Nevertheless, these methods suffer from a granularity inconsistency regarding the usage of group tokens, which are aligned in the all-to-one v.s. one-to-one manners during the training and inference phases, respectively. We argue that this discrepancy arises from the lack of elaborate supervision for each group token. To bridge this granularity gap, this paper explores explicit supervision for the group tokens from the prototypical knowledge. To this end, this paper proposes the non-learnable prototypical regularization (NPR) where non-learnable prototypes are estimated from source features to serve as supervision and enable contrastive matching of the group tokens. This regularization encourages the group tokens to segment objects with less redundancy and capture more comprehensive semantic regions, leading to increased compactness and richness. Based on NPR, we propose the prototypical guidance segmentation network (PGSeg) that incorporates multi-modal regularization by leveraging prototypical sources from both images and texts at different levels, progressively enhancing the segmentation capability with diverse prototypical patterns. Experimental results show that our proposed method achieves state-of-the-art performance on several benchmark datasets.

----

## [0] A Theoretical Analysis of Optimistic Proximal Policy Optimization in Linear Markov Decision Processes

**Authors**: *Han Zhong, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html)

**Abstract**:

The proximal policy optimization (PPO) algorithm stands as one of the most prosperous methods in the field of reinforcement learning (RL). Despite its success, the theoretical understanding of PPO remains deficient. Specifically, it is unclear whether PPO or its optimistic variants can effectively solve linear Markov decision processes (MDPs), which are arguably the simplest models in RL with function approximation.     To bridge this gap, we propose an optimistic variant of PPO for episodic adversarial linear MDPs with full-information feedback, and establish a $\tilde{\mathcal{O}}(d^{3/4}H^2K^{3/4})$ regret for it. Here $d$ is the ambient dimension of linear MDPs, $H$ is the length of each episode, and $K$ is the number of episodes. Compared with existing policy-based algorithms, we achieve the state-of-the-art regret bound in both stochastic linear MDPs and adversarial linear MDPs with full information. Additionally, our algorithm design features a novel multi-batched updating mechanism and the theoretical analysis utilizes a new covering number argument of value and policy classes, which might be of independent interest.

----

## [0] Uncertainty-Aware Instance Reweighting for Off-Policy Learning

**Authors**: *Xiaoying Zhang, Junpu Chen, Hongning Wang, Hong Xie, Yang Liu, John C. S. Lui, Hang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e97ac22927560eb2de6b658498cbc575-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e97ac22927560eb2de6b658498cbc575-Abstract-Conference.html)

**Abstract**:

Off-policy learning, referring to the procedure of policy optimization with access only to logged feedback data, has shown importance in various important real-world applications, such as search engines and recommender systems. While the ground-truth logging policy is usually unknown, previous work simply takes its estimated value for the off-policy learning, ignoring the negative impact from both high bias and high variance resulted from such an estimator. And these impact is often magnified on samples with small and inaccurately estimated logging probabilities. The contribution of this work is to explicitly model the uncertainty in the estimated logging policy, and propose an Uncertainty-aware Inverse Propensity Score estimator (UIPS) for improved off-policy learning, with a theoretical convergence guarantee. Experiment results on the synthetic and real-world recommendation datasets demonstrate that UIPS significantly improves the quality of the discovered policy, when compared against an extensive list of state-of-the-art baselines.

----

## [0] Model-free Posterior Sampling via Learning Rate Randomization

**Authors**: *Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Rémi Munos, Alexey Naumov, Pierre Perrault, Michal Valko, Pierre Ménard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e985dfca10e1167c0836a70880ef0858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e985dfca10e1167c0836a70880ef0858-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce Randomized Q-learning (RandQL), a novel randomized model-free algorithm for regret minimization in episodic Markov Decision Processes (MDPs). To the best of our knowledge, RandQL is the first tractable model-free posterior sampling-based algorithm. We analyze the performance of RandQL in both tabular and non-tabular metric space settings. In tabular MDPs, RandQL achieves a regret bound of order $\widetilde{\mathcal{O}}(\sqrt{H^{5}SAT})$, where $H$ is the planning horizon, $S$ is the number of states, $A$ is the number of actions, and $T$ is the number of episodes. For a metric state-action space, RandQL enjoys a regret bound of order $\widetilde{\mathcal{O}}(H^{5/2} T^{(d_z+1)/(d_z+2)})$, where $d_z$ denotes the zooming dimension. Notably, RandQL achieves optimistic exploration without using bonuses, relying instead on a novel idea of learning rate randomization. Our empirical study shows that RandQL outperforms existing approaches on baseline exploration environments.

----

## [0] TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

**Authors**: *Preetha Vijayan, Prashant Shivaram Bhat, Bahram Zonooz, Elahe Arani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html)

**Abstract**:

Continual learning (CL) has remained a persistent challenge for deep neural networks due to catastrophic forgetting (CF) of previously learned tasks. Several techniques such as weight regularization, experience rehearsal, and parameter isolation have been proposed to alleviate CF. Despite their relative success, these research directions have predominantly remained orthogonal and suffer from several shortcomings, while missing out on the advantages of competing strategies. On the contrary, the brain continually learns, accommodates, and transfers knowledge across tasks by simultaneously leveraging several neurophysiological processes, including neurogenesis, active forgetting, neuromodulation, metaplasticity, experience rehearsal, and context-dependent gating, rarely resulting in CF. Inspired by how the brain exploits multiple mechanisms concurrently, we propose TriRE, a novel CL paradigm that encompasses retaining the most prominent neurons for each task, revising and solidifying the extracted knowledge of current and past tasks, and actively promoting less active neurons for subsequent tasks through rewinding and relearning. Across CL settings, TriRE significantly reduces task interference and surpasses different CL approaches considered in isolation.

----

## [0] Implicit Variational Inference for High-Dimensional Posteriors

**Authors**: *Anshuk Uppal, Kristoffer Stensbo-Smidt, Wouter Boomsma, Jes Frellsen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e99be8b1f637996eaf1154f2f4cb6f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e99be8b1f637996eaf1154f2f4cb6f49-Abstract-Conference.html)

**Abstract**:

In variational inference, the benefits of Bayesian models rely on accurately capturing the true posterior distribution. We propose using neural samplers that specify implicit distributions, which are well-suited for approximating complex multimodal and correlated posteriors in high-dimensional spaces. Our approach introduces novel bounds for approximate inference using implicit distributions by locally linearising the neural sampler. This is distinct from existing methods that rely on additional discriminator networks and unstable adversarial objectives. Furthermore, we present a new sampler architecture that, for the first time, enables implicit distributions over tens of millions of latent variables, addressing computational concerns by using differentiable numerical approximations. We empirically show that our method is capable of recovering correlations across layers in large Bayesian neural networks, a property that is crucial for a network's performance but notoriously challenging to achieve. To the best of our knowledge, no other method has been shown to accomplish this task for such large models. Through experiments in downstream tasks, we demonstrate that our expressive posteriors outperform state-of-the-art uncertainty quantification methods, validating the effectiveness of our training algorithm and the quality of the learned implicit approximation.

----

## [0] k-Median Clustering via Metric Embedding: Towards Better Initialization with Differential Privacy

**Authors**: *Chenglin Fan, Ping Li, Xiaoyun Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9a612969b4df241ff0d8273656bd5a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9a612969b4df241ff0d8273656bd5a4-Abstract-Conference.html)

**Abstract**:

In clustering algorithms, the choice of initial centers is crucial for the quality of the learned clusters. We propose a new initialization scheme for the $k$-median problem in the general metric space (e.g., discrete space induced by graphs), based on the construction of metric embedding tree structure of the data. We propose a novel and efficient search algorithm, for good initial centers that can be used subsequently for the local search algorithm. The so-called HST initialization method can produce initial centers achieving lower error than those from another popular method $k$-median++, also with higher efficiency when $k$ is not too small. Our HST initialization can also be easily extended to the setting of differential privacy (DP) to generate private initial centers. We show that the error of applying DP local search followed by our private HST initialization improves previous results on the approximation error, and approaches the lower bound within a small factor. Experiments demonstrate the effectiveness of our proposed methods.

----

## [0] Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift

**Authors**: *Xingdong Feng, Xin He, Caixing Wang, Chao Wang, Jingnan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9b0ae84d6879b30c78cb8537466a4e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9b0ae84d6879b30c78cb8537466a4e0-Abstract-Conference.html)

**Abstract**:

Covariate shift occurs prevalently in practice, where the input distributions of the source and target data are substantially different. Despite its practical importance in various learning problems, most of the existing methods only focus on some specific learning tasks and are not well validated theoretically and numerically. To tackle this problem, we propose a unified analysis of general nonparametric methods in a reproducing kernel Hilbert space (RKHS) under covariate shift.  Our theoretical results are established for a general loss belonging to a rich loss function family, which includes many commonly used methods as special cases, such as mean regression, quantile regression, likelihood-based classification, and margin-based classification. Two types of covariate shift problems are the focus of this paper and the sharp convergence rates are established for a general loss function to provide a unified theoretical analysis, which concurs with the optimal results in literature where the squared loss is used. Extensive numerical studies on synthetic and real examples confirm our theoretical findings and further illustrate the effectiveness of our proposed method.

----

## [0] Learning Functional Transduction

**Authors**: *Mathieu Chalvidal, Thomas Serre, Rufin VanRullen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9b8a3362a6d9a7f9f842bd2d919e1a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9b8a3362a6d9a7f9f842bd2d919e1a0-Abstract-Conference.html)

**Abstract**:

Research in statistical learning has polarized into two general approaches to perform regression analysis: Transductive methods construct estimates directly based on exemplar data using generic relational principles which might suffer from the curse of dimensionality. Conversely, inductive methods can potentially fit highly complex functions at the cost of compute-intensive solution searches. In this work, we leverage the theory of vector-valued Reproducing Kernel Banach Spaces (RKBS) to propose a hybrid approach: We show that transductive regression systems can be meta-learned with gradient descent to form efficient in-context neural approximators of function defined over both finite and infinite-dimensional spaces (operator regression). Once trained, our Transducer can almost instantaneously capture new functional relationships and produce original image estimates, given a few pairs of input and output examples. We demonstrate the benefit of our meta-learned transductive approach to model physical systems influenced by varying external factors with little data at a fraction of the usual deep learning training costs for partial differential equations and climate modeling applications.

----

## [0] Gaussian Membership Inference Privacy

**Authors**: *Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html)

**Abstract**:

We propose a novel and practical privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model.  Consequently, $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). In particular, we derive a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP) by theoretically analyzing likelihood ratio-based membership inference attacks on stochastic gradient descent (SGD). Our analysis highlights that models trained with standard SGD already offer an elementary level of MIP.  Additionally, we show how $f$-MIP can be amplified by adding noise to gradient updates. Our analysis further yields an analytical membership inference attack that offers two distinct advantages over previous approaches. First, unlike existing state-of-the-art attacks that require training hundreds of shadow models, our attack does not require any shadow model.  Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, we quantify how various hyperparameters (e.g., batch size, number of model parameters) and specific data characteristics determine an attacker's ability to accurately infer a point's membership in the training set. We demonstrate the effectiveness of our method on models trained on vision and tabular datasets.

----

## [0] Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder

**Authors**: *Huiwon Jang, Jihoon Tack, Daewon Choi, Jongheon Jeong, Jinwoo Shin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9df55bf67e499635908395931ed6ea9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9df55bf67e499635908395931ed6ea9-Abstract-Conference.html)

**Abstract**:

Despite its practical importance across a wide range of modalities, recent advances in self-supervised learning (SSL) have been primarily focused on a few well-curated domains, e.g., vision and language, often relying on their domain-specific knowledge. For example, Masked Auto-Encoder (MAE) has become one of the popular architectures in these domains, but less has explored its potential in other modalities. In this paper, we develop MAE as a unified, modality-agnostic SSL framework. In turn, we argue meta-learning as a key to interpreting MAE as a modality-agnostic learner, and propose enhancements to MAE from the motivation to jointly improve its SSL across diverse modalities, coined MetaMAE as a result. Our key idea is to view the mask reconstruction of MAE as a meta-learning task: masked tokens are predicted by adapting the Transformer meta-learner through the amortization of unmasked tokens. Based on this novel interpretation, we propose to integrate two advanced meta-learning techniques. First, we adapt the amortized latent of the Transformer encoder using gradient-based meta-learning to enhance the reconstruction. Then, we maximize the alignment between amortized and adapted latents through task contrastive learning which guides the Transformer encoder to better encode the task-specific knowledge. Our experiment demonstrates the superiority of MetaMAE in the modality-agnostic SSL benchmark (called DABS), significantly outperforming prior baselines.

----

## [0] Dual Self-Awareness Value Decomposition Framework without Individual Global Max for Cooperative MARL

**Authors**: *Zhiwei Xu, Bin Zhang, Dapeng Li, Guangchong Zhou, Zeren Zhang, Guoliang Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9e140df6de01afb672cb859d203c307-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9e140df6de01afb672cb859d203c307-Abstract-Conference.html)

**Abstract**:

Value decomposition methods have gained popularity in the field of cooperative multi-agent reinforcement learning. However, almost all existing methods follow the principle of Individual Global Max (IGM) or its variants, which limits their problem-solving capabilities. To address this, we propose a dual self-awareness value decomposition framework, inspired by the notion of dual self-awareness in psychology, that entirely rejects the IGM premise. Each agent consists of an ego policy for action selection and an alter ego value function to solve the credit assignment problem. The value function factorization can ignore the IGM assumption by utilizing an explicit search procedure. On the basis of the above, we also suggest a novel anti-ego exploration mechanism to avoid the algorithm becoming stuck in a local optimum. As the first fully IGM-free value decomposition method, our proposed framework achieves desirable performance in various cooperative tasks.

----

## [0] DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

**Authors**: *Mintong Kang, Dawn Song, Bo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea0b28cbbd0cbc45ec4ac38e92da9cb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea0b28cbbd0cbc45ec4ac38e92da9cb2-Abstract-Conference.html)

**Abstract**:

Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20\% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10\% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.

----

## [0] Squared Neural Families: A New Class of Tractable Density Models

**Authors**: *Russell Tsuchida, Cheng Soon Ong, Dino Sejdinovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html)

**Abstract**:

Flexible models for probability distributions are an essential ingredient in many machine learning tasks. We develop and investigate a new class of probability distributions, which we call a Squared Neural Family (SNEFY), formed by squaring the 2-norm of a neural network and normalising it with respect to a base measure. Following the reasoning similar to the well established connections between infinitely wide neural networks and Gaussian processes, we show that SNEFYs admit closed form normalising constants in many cases of interest, thereby resulting in flexible yet fully tractable density models. SNEFYs strictly generalise classical exponential families, are closed under conditioning, and have tractable marginal distributions. Their utility is illustrated on a variety of density estimation, conditional density estimation, and density estimation with missing data tasks.

----

## [0] Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation

**Authors**: *Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, Shenghua Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html)

**Abstract**:

We present a novel alignment-before-generation approach to tackle the challenging task of generating general 3D shapes based on 2D images or texts. Directly learning a conditional generative model from images or texts to 3D shapes is prone to producing inconsistent results with the conditions because 3D shapes have an additional dimension whose distribution significantly differs from that of 2D images and texts. To bridge the domain gap among the three modalities and facilitate multi-modal-conditioned 3D shape generation, we explore representing 3D shapes in a shape-image-text-aligned space. Our framework comprises two models: a Shape-Image-Text-Aligned Variational Auto-Encoder (SITA-VAE) and a conditional Aligned Shape Latent Diffusion Model (ASLDM). The former model encodes the 3D shapes into the shape latent space aligned to the image and text and reconstructs the fine-grained 3D neural fields corresponding to given shape embeddings via the transformer-based decoder. The latter model learns a probabilistic mapping function from the image or text space to the latent shape space. Our extensive experiments demonstrate that our proposed approach can generate higher-quality and more diverse 3D shapes that better semantically conform to the visual or textural conditional inputs, validating the effectiveness of the shape-image-text-aligned space for cross-modality 3D shape generation.

----

## [0] Estimating Riemannian Metric with Noise-Contaminated Intrinsic Distance

**Authors**: *Jiaming Qiu, Xiongtao Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea5cb7d9fd2deb0554def3552962d276-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea5cb7d9fd2deb0554def3552962d276-Abstract-Conference.html)

**Abstract**:

We extend metric learning by studying the Riemannian manifold structure of the underlying data space induced by similarity measures between data points. The key quantity of interest here is the Riemannian metric, which characterizes the Riemannian geometry and defines straight lines and derivatives on the manifold. Being able to estimate the Riemannian metric allows us to gain insights into the underlying manifold and compute geometric features such as the geodesic curves. We model the observed similarity measures as noisy responses generated from a function of the intrinsic geodesic distance between data points. A new local regression approach is proposed to learn the Riemannian metric tensor and its derivatives based on a Taylor expansion for the squared geodesic distances, accommodating different types of data such as continuous, binary, or comparative responses. We develop theoretical foundation for our method by deriving the rates of convergence for the asymptotic bias and variance of the estimated metric tensor. The proposed method is shown to be versatile in simulation studies and real data applications involving taxi trip time in New York City and MNIST digits.

----

## [0] Inner Product-based Neural Network Similarity

**Authors**: *Wei Chen, Zichen Miao, Qiang Qiu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea7623ff02edffe68866f88da2667592-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea7623ff02edffe68866f88da2667592-Abstract-Conference.html)

**Abstract**:

Analyzing representational similarity among neural networks (NNs) is essential for interpreting or transferring deep models. In application scenarios where numerous NN models are learned, it becomes crucial to assess model similarities in computationally efficient ways. In this paper, we propose a new paradigm for reducing NN representational similarity to filter subspace distance. Specifically, when convolutional filters are decomposed as a linear combination of a set of filter subspace elements, denoted as filter atoms, and have those decomposed atom coefficients shared across networks, NN representational similarity can be significantly simplified as calculating the cosine distance among respective filter atoms, to achieve millions of times computation reduction over popular probing-based methods. We provide both theoretical and empirical evidence that such simplified filter subspace-based similarity preserves a strong linear correlation with other popular probing-based metrics, while being significantly more efficient to obtain and robust to probing data. We further validate the effectiveness of the proposed method in various application scenarios where numerous models exist, such as federated and continual learning as well as analyzing training dynamics. We hope our findings can help further explorations of real-time large-scale representational similarity analysis in neural networks.

----

## [0] State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory

**Authors**: *Shida Wang, Beichen Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea8608c6258450e75b3443ec8022fb2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea8608c6258450e75b3443ec8022fb2e-Abstract-Conference.html)

**Abstract**:

State-space models have gained popularity in sequence modelling due to their simple and efficient network structures. However, the absence of nonlinear activation along the temporal direction limits the model's capacity. In this paper, we prove that stacking state-space models with layer-wise nonlinear activation is sufficient to approximate any continuous sequence-to-sequence relationship. Our findings demonstrate that the addition of layer-wise nonlinear activation enhances the model's capacity to learn complex sequence patterns. Meanwhile, it can be seen both theoretically and empirically that the state-space models do not fundamentally resolve the issue of exponential decaying memory. Theoretical results are justified by numerical verifications.

----

## [0] Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory

**Authors**: *Ankur Sikarwar, Mengmi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Working memory (WM), a fundamental cognitive process facilitating the temporary storage, integration, manipulation, and retrieval of information, plays a vital role in reasoning and decision-making tasks. Robust benchmark datasets that capture the multifaceted nature of WM are crucial for the effective development and evaluation of AI WM models. Here, we introduce a comprehensive Working Memory (WorM) benchmark dataset for this purpose. WorM comprises 10 tasks and a total of 1 million trials, assessing 4 functionalities, 3 domains, and 11 behavioral and neural characteristics of WM. We jointly trained and tested state-of-the-art recurrent neural networks and transformers on all these tasks. We also include human behavioral benchmarks as an upper bound for comparison. Our results suggest that AI models replicate some characteristics of WM in the brain, most notably primacy and recency effects, and neural clusters and correlates specialized for different domains and functionalities of WM. In the experiments, we also reveal some limitations in existing models to approximate human behavior. This dataset serves as a valuable resource for communities in cognitive psychology, neuroscience, and AI, offering a standardized framework to compare and enhance WM models, investigate WM's neural underpinnings, and develop WM models with human-like capabilities. Our source code and data are available at: https://github.com/ZhangLab-DeepNeuroCogLab/WorM

----

## [0] Many-body Approximation for Non-negative Tensors

**Authors**: *Kazu Ghalamkari, Mahito Sugiyama, Yoshinobu Kawahara*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html)

**Abstract**:

We present an alternative approach to decompose non-negative tensors, called many-body approximation. Traditional decomposition methods assume low-rankness in the representation, resulting in difficulties in global optimization and target rank selection. We avoid these problems by energy-based modeling of tensors, where a tensor and its mode correspond to a probability distribution and a random variable, respectively. Our model can be globally optimized in terms of the KL divergence minimization by taking the interaction between variables (that is, modes), into account that can be tuned more intuitively than ranks. Furthermore, we visualize interactions between modes as tensor networks and reveal a nontrivial relationship between many-body approximation and low-rank approximation. We demonstrate the effectiveness of our approach in tensor completion and approximation.

----

## [0] Breaking the Communication-Privacy-Accuracy Tradeoff with f-Differential Privacy

**Authors**: *Richeng Jin, Zhonggen Su, Caijun Zhong, Zhaoyang Zhang, Tony Q. S. Quek, Huaiyu Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ead13878cd158f013becb6a559a60364-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ead13878cd158f013becb6a559a60364-Abstract-Conference.html)

**Abstract**:

We consider a federated data analytics problem in which a server coordinates the collaborative data analysis of multiple users with privacy concerns and limited communication capability. The commonly adopted compression schemes introduce information loss into local data while improving communication efficiency, and it remains an open problem whether such discrete-valued mechanisms provide any privacy protection. In this paper, we study the local differential privacy guarantees of discrete-valued mechanisms with finite output space through the lens of $f$-differential privacy (DP). More specifically, we advance the existing literature by deriving tight $f$-DP guarantees for a variety of discrete-valued mechanisms, including the binomial noise and the binomial mechanisms that are proposed for privacy preservation, and the sign-based methods that are proposed for data compression, in closed-form expressions. We further investigate the amplification in privacy by sparsification and propose a ternary stochastic compressor. By leveraging compression for privacy amplification, we improve the existing methods by removing the dependency of accuracy (in terms of mean square error) on communication cost in the popular use case of distributed mean estimation, therefore breaking the three-way tradeoff between privacy, communication, and accuracy.

----

## [0] Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation

**Authors**: *Haoran Chen, Xintong Han, Zuxuan Wu, Yu-Gang Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html)

**Abstract**:

Most existing methods for unsupervised domain adaptation (UDA) rely on a shared network to extract domain-invariant features. However, when facing multiple source domains, optimizing such a network involves updating the parameters of the entire network, making it both computationally expensive and challenging, particularly when coupled with min-max objectives. Inspired by recent advances in prompt learning that adapts high-capacity models for downstream tasks in a computationally economic way, we introduce Multi-Prompt Alignment (MPA), a simple yet efficient framework for multi-source UDA. Given a source and target domain pair, MPA first trains an individual prompt to minimize the domain gap through a contrastive loss. Then, MPA denoises the learned prompts through an auto-encoding process and aligns them by maximizing the agreement of all the reconstructed prompts. Moreover, we show that the resulting subspace acquired from the auto-encoding process can easily generalize to a streamlined set of target domains, making our method more efficient for practical usage. Extensive experiments show that MPA achieves state-of-the-art results on three popular datasets with an impressive average accuracy of 54.1% on DomainNet.

----

## [0] Characterization and Learning of Causal Graphs with Small Conditioning Sets

**Authors**: *Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eaef3b49866b942041a34bb8da397eb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eaef3b49866b942041a34bb8da397eb7-Abstract-Conference.html)

**Abstract**:

Constraint-based causal discovery algorithms learn part of the causal graph structure by systematically testing conditional independences  observed in the data. These algorithms, such as the PC algorithm and its variants, rely on graphical characterizations of the so-called equivalence class of causal graphs proposed by Pearl. However, constraint-based causal discovery algorithms struggle when data is limited since conditional independence tests quickly lose their statistical power, especially when the conditioning set is large. To address this, we propose using conditional independence tests where the size of the conditioning set is upper bounded by some integer k for robust causal discovery. The existing graphical characterizations of the equivalence classes of causal graphs are not applicable when we cannot leverage all the conditional independence statements. We first define the notion of k-Markov equivalence: Two causal graphs are k-Markov equivalent if they entail the same conditional independence constraints where the conditioning set size is upper bounded by k. We propose a novel representation that allows us to graphically characterize k-Markov equivalence between two causal graphs. We propose a sound constraint-based algorithm called the k-PC algorithm for learning this equivalence class. Finally, we conduct synthetic, and semi-synthetic experiments to demonstrate that the k-PC algorithm enables more robust causal discovery in the small sample regime compared to the baseline algorithms.

----

## [0] Finite Population Regression Adjustment and Non-asymptotic Guarantees for Treatment Effect Estimation

**Authors**: *Mehrdad Ghadiri, David Arbour, Tung Mai, Cameron Musco, Anup B. Rao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html)

**Abstract**:

The design and analysis of randomized experiments is fundamental to many areas, from the physical and social sciences to industrial settings. Regression adjustment is a popular technique to reduce the variance of estimates obtained from experiments, by utilizing information contained in auxiliary covariates. While there is a large literature within the statistics community studying various approaches to regression adjustment and their asymptotic properties, little focus has been given to approaches in the finite population setting with non-asymptotic accuracy bounds. Further, prior work typically assumes that an entire population is exposed to an experiment, whereas practitioners often seek to minimize the number of subjects exposed to an experiment, for ethical and pragmatic reasons.In this work, we study the problems of estimating the sample mean, individual treatment effects, and average treatment effect with regression adjustment. We propose approaches that use techniques from randomized numerical linear algebra to sample a subset of the population on which to perform an experiment. We give non-asymptotic accuracy bounds for our methods and demonstrate that they compare favorably with prior approaches.

----

## [0] P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting

**Authors**: *Sungwon Kim, Kevin J. Shih, Rohan Badlani, João Felipe Santos, Evelina Bakhturina, Mikyas Desta, Rafael Valle, Sungroh Yoon, Bryan Catanzaro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html)

**Abstract**:

While recent large-scale neural codec language models have shown significant improvement in zero-shot TTS by training on thousands of hours of data, they suffer from drawbacks such as a lack of robustness, slow sampling speed similar to previous autoregressive TTS methods, and reliance on pre-trained neural codec representations. Our work proposes P-Flow, a fast and data-efficient zero-shot TTS model that uses speech prompts for speaker adaptation. P-Flow comprises a speech-prompted text encoder for speaker adaptation and a flow matching generative decoder for high-quality and fast speech synthesis. Our speech-prompted text encoder uses speech prompts and text input to generate speaker-conditional text representation. The flow matching generative decoder uses the speaker-conditional output to synthesize high-quality personalized speech significantly faster than in real-time. Unlike the neural codec language models, we specifically train P-Flow on LibriTTS dataset using a continuous mel-representation. Through our training method using continuous speech prompts, P-Flow matches the speaker similarity performance of the large-scale zero-shot TTS models with two orders of magnitude less training data and has more than 20$\times$ faster sampling speed. Our results show that P-Flow has better pronunciation and is preferred in human likeness and speaker similarity to its recent state-of-the-art counterparts, thus defining P-Flow as an attractive and desirable alternative. We provide audio samples on our demo page: [https://research.nvidia.com/labs/adlr/projects/pflow](https://research.nvidia.com/labs/adlr/projects/pflow)

----

## [0] Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability

**Authors**: *Jingfeng Wu, Vladimir Braverman, Jason D. Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)

**Abstract**:

Recent research has observed that in machine learning optimization, gradient descent (GD) often operates at the edge of stability (EoS) [Cohen et al., 2021], where the stepsizes are set to be large, resulting in non-monotonic losses induced by the GD iterates. This paper studies the convergence and implicit bias of constant-stepsize GD for logistic regression on linearly separable data in the EoS regime. Despite the presence of local oscillations, we prove that the logistic loss can be minimized by GD with any constant stepsize over a long time scale. Furthermore, we prove that with any constant stepsize, the GD iterates tend to infinity when projected to a max-margin direction (the hard-margin SVM direction) and converge to a fixed vector that minimizes a strongly convex potential when projected to the orthogonal complement of the max-margin direction. In contrast, we also show that in the EoS regime, GD iterates may diverge catastrophically under the exponential loss, highlighting the superiority of the logistic loss. These theoretical findings are in line with numerical simulations and complement existing theories on the convergence and implicit bias of GD for logistic regression, which are only applicable when the stepsizes are sufficiently small.

----

## [0] Two Sides of One Coin: the Limits of Untuned SGD and the Power of Adaptive Methods

**Authors**: *Junchi Yang, Xiang Li, Ilyas Fatkhullin, Niao He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html)

**Abstract**:

The classical analysis of Stochastic Gradient Descent (SGD) with polynomially decaying stepsize $\eta_t = \eta/\sqrt{t}$ relies on well-tuned $\eta$ depending on  problem parameters such as Lipschitz smoothness constant, which is often unknown in practice. In this work, we prove that SGD with arbitrary $\eta > 0$, referred to as untuned SGD, still attains an order-optimal convergence rate $\widetilde{\mathcal{O}}(T^{-1/4})$ in terms of gradient norm for minimizing smooth objectives. Unfortunately, it comes at the expense of a catastrophic exponential dependence on the smoothness constant, which we show is unavoidable for this scheme even in the noiseless setting. We then examine three families of adaptive methods — Normalized SGD (NSGD), AMSGrad, and AdaGrad — unveiling their power in preventing such exponential dependency in  the absence of information about the smoothness parameter and boundedness of stochastic gradients. Our results provide  theoretical justification for the advantage of adaptive methods over untuned SGD in alleviating the issue with large gradients.

----

## [0] Theoretical Analysis of the Inductive Biases in Deep Convolutional Networks

**Authors**: *Zihao Wang, Lei Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html)

**Abstract**:

In this paper, we provide a theoretical analysis of the inductive biases in convolutional neural networks (CNNs). We start by examining the universality of CNNs, i.e., the ability to approximate any continuous functions.  We prove that a depth of $\mathcal{O}(\log d)$ suffices for deep CNNs to achieve this universality, where $d$ in the input dimension.  Additionally, we establish  that learning sparse functions with  CNNs requires only $\widetilde{\mathcal{O}}(\log^2d)$ samples, indicating that deep CNNs can efficiently capture {\em long-range} sparse correlations. These results are made possible through a novel combination of the multichanneling and downsampling when increasing the network depth. We also delve into the distinct roles  of weight sharing and locality in CNNs. To this end, we compare the performance of CNNs, locally-connected networks (LCNs), and fully-connected networks (FCNs) on a simple regression task, where LCNs can be viewed as CNNs without weight sharing. On the one hand,  we  prove that LCNs require ${\Omega}(d)$ samples while CNNs need only $\widetilde{\mathcal{O}}(\log^2d)$ samples,  highlighting the  critical role of weight sharing. On the other hand, we prove that FCNs require $\Omega(d^2)$ samples, whereas LCNs need only $\widetilde{\mathcal{O}}(d)$ samples,  underscoring the importance  of locality. These provable separations quantify the difference between the two biases, and the major observation behind our proof is that weight sharing and locality break different symmetries in the learning process.

----

## [0] Object Reprojection Error (ORE): Camera pose benchmarks from lightweight tracking annotations

**Authors**: *Xingyu Chen, Weiyao Wang, Hao Tang, Matt Feiszli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb206443c93d07da8b1974b768d8a0d4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb206443c93d07da8b1974b768d8a0d4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

3D spatial understanding is highly valuable in the context of semantic modeling of environments, agents, and their relationships.  Semantic modeling approaches employed on monocular video often ingest outputs from off-the-shelf SLAM/SfM pipelines, which are anecdotally observed to perform poorly or fail completely on some fraction of the videos of interest.  These target videos may vary widely in complexity of scenes, activities, camera trajectory, etc.  Unfortunately, such semantically-rich video data often comes with no ground-truth 3D information, and in practice it is prohibitively costly or impossible to obtain ground truth reconstructions or camera pose post-hoc.  This paper proposes a novel evaluation protocol, Object Reprojection Error (ORE) to benchmark camera trajectories; ORE computes reprojection error for static objects within the video and requires only lightweight object tracklet annotations.  These annotations are easy to gather on new or existing video, enabling ORE to be calculated on essentially arbitrary datasets.  We show that ORE maintains high rank correlation with standard metrics based on groundtruth.  Leveraging ORE, we source videos and annotations from Ego4D-EgoTracks, resulting in EgoStatic, a large-scale diverse dataset for evaluating camera trajectories in-the-wild.

----

## [0] Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition

**Authors**: *Samuel Dooley, Rhea Sukthanker, John P. Dickerson, Colin White, Frank Hutter, Micah Goldblum*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html)

**Abstract**:

Face recognition systems are widely deployed in safety-critical applications, including law enforcement, yet they exhibit bias across a range of socio-demographic dimensions, such as gender and race.  Conventional wisdom dictates that model biases arise from biased training data.  As a consequence, previous works on bias mitigation largely focused on pre-processing the training data, adding penalties to prevent bias from effecting the model during training, or post-processing predictions to debias them, yet these approaches have shown limited success on hard problems such as face recognition.  In our work, we discover that biases are actually inherent to neural network architectures themselves.  Following this reframing, we conduct the first neural architecture search for fairness, jointly with a search for hyperparameters. Our search outputs a suite of models which Pareto-dominate all other high-performance architectures and existing bias mitigation methods in terms of accuracy and fairness, often by large margins, on the two most widely used datasets for face identification, CelebA and VGGFace2. Furthermore, these models generalize to other datasets and sensitive attributes. We release our code, models and raw data files at https://github.com/dooleys/FR-NAS.

----

## [0] H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation

**Authors**: *Yanjie Ze, Yuyao Liu, Ruizhe Shi, Jiaxin Qin, Zhecheng Yuan, Jiashun Wang, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb4b1f7feadcd124a59de6ff7b9196f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb4b1f7feadcd124a59de6ff7b9196f3-Abstract-Conference.html)

**Abstract**:

Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human $\textbf{H}$and-$\textbf{In}$formed visual representation learning framework to solve difficult $\textbf{Dex}$terous manipulation tasks ($\textbf{H-InDex}$) with reinforcement learning. Our framework consists of three stages: $\textit{(i)}$ pre-training representations with 3D human hand pose estimation, $\textit{(ii)}$ offline adapting representations with self-supervised keypoint detection, and $\textit{(iii)}$ reinforcement learning with exponential moving average BatchNorm. The last two stages only modify $0.36$% parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study $\textbf{12}$ challenging dexterous manipulation tasks and find that $\textbf{H-InDex}$ largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code and videos are available at https://yanjieze.com/H-InDex .

----

## [0] DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets

**Authors**: *Lazar Atanackovic, Alexander Tong, Bo Wang, Leo J. Lee, Yoshua Bengio, Jason S. Hartford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html)

**Abstract**:

One of the grand challenges of cell biology is inferring the gene regulatory network (GRN) which describes interactions between genes and their products that control gene expression and cellular function. We can treat this as a causal discovery problem but with two non-standard challenges: (1) regulatory networks are inherently cyclic so we should not model a GRN as a directed acyclic graph (DAG), and (2) observations have significant measurement noise so for typical sample sizes, there will always be a large equivalence class of graphs that are likely given the data, and we want methods that capture this uncertainty. Existing methods either focus on challenge (1), identifying cyclic structure from dynamics, or on challenge (2) learning complex Bayesian posteriors over directed acyclic graphs, but not both. In this paper we leverage the fact that it is possible to estimate the ``velocity'' of the expression of a gene with RNA velocity techniques to develop an approach that addresses both challenges. Because we have access to velocity information, we can treat the Bayesian structure learning problem as a problem of sparse identification of a dynamical system, capturing cyclic feedback loops through time. We leverage Generative Flow Networks (GFlowNets) to estimate the posterior distribution over the combinatorial space of possible sparse dependencies. Our results indicate that our method learns posteriors that better encapsulate the distributions of cyclic structures compared to counterpart state-of-the-art Bayesian structure learning approaches.

----

## [0] RaLEs: a Benchmark for Radiology Language Evaluations

**Authors**: *Juanma Zambrano Chaves, Nandita Bhaskhar, Maayane Attias, Jean-Benoit Delbrouck, Daniel L. Rubin, Andreas M. Loening, Curtis P. Langlotz, Akshay Chaudhari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5683d06bdef51ed4dff644908eef4b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5683d06bdef51ed4dff644908eef4b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The radiology report is the main form of communication between radiologists and other clinicians. Prior work in natural language processing in radiology reports has shown the value of developing methods tailored for individual tasks such as identifying reports with critical results or disease detection. Meanwhile, English and biomedical natural language understanding benchmarks such as the General Language Understanding and Evaluation as well as Biomedical Language Understanding and Reasoning Benchmark have motivated the development of models that can be easily adapted to address many tasks in those domains. Here, we characterize the radiology report as a distinct domain and introduce RaLEs, the Radiology Language Evaluations, as a benchmark for natural language understanding and generation in radiology. RaLEs is comprised of seven natural language understanding and generation evaluations including the extraction of anatomical and disease entities and their relations, procedure selection, and report summarization. We characterize the performance of models designed for the general, biomedical, clinical and radiology domains across these tasks. We find that advances in the general and biomedical domains do not necessarily translate to radiology, and that improved models from the general domain can perform comparably to smaller clinical-specific models. The limited performance of existing pre-trained models on RaLEs highlights the opportunity to improve domain-specific self-supervised models for natural language processing in radiology. We propose RaLEs as a benchmark to promote and track the development of such domain-specific radiology language models.

----

## [0] AutoGO: Automated Computation Graph Optimization for Neural Network Evolution

**Authors**: *Mohammad Salameh, Keith G. Mills, Negar Hassanpour, Fred X. Han, Shuting Zhang, Wei Lu, Shangling Jui, Chunhua Zhou, Fengyu Sun, Di Niu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html)

**Abstract**:

Optimizing Deep Neural Networks (DNNs) to obtain high-quality models for efficient real-world deployment has posed multi-faceted challenges to machine learning engineers. Existing methods either search for neural architectures in heuristic design spaces or apply low-level adjustments to computation primitives to improve inference efficiency on hardware. We present Automated Graph Optimization (AutoGO), a framework to evolve neural networks in a low-level Computation Graph (CG) of primitive operations to improve both its performance and hardware friendliness. Through a tokenization scheme, AutoGO performs variable-sized segment mutations, making both primitive changes and larger-grained changes to CGs. We introduce our segmentation and mutation algorithms, efficient frequent segment mining technique, as well as a pretrained context-aware predictor to estimate the impact of segment replacements. Extensive experimental results show that AutoGO can automatically evolve several typical large convolutional networks to achieve significant task performance improvement and FLOPs reduction on a range of CV tasks, ranging from Classification, Semantic Segmentation, Human Pose Estimation, to Super Resolution, yet without introducing any newer primitive operations. We also demonstrate the lightweight deployment results of AutoGO-optimized super-resolution and denoising U-Nets on a cycle simulator for a Neural Processing Unit (NPU), achieving PSNR improvement and latency/power reduction simultaneously. Code available at https://github.com/Ascend-Research/AutoGO.

----

## [0] Where Did I Come From? Origin Attribution of AI-Generated Images

**Authors**: *Zhenting Wang, Chen Chen, Yi Zeng, Lingjuan Lyu, Shiqing Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebb4c188fafe7da089b41a9f615ad84d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebb4c188fafe7da089b41a9f615ad84d-Abstract-Conference.html)

**Abstract**:

Image generation techniques have been gaining increasing attention recently, but concerns have been raised about the potential misuse and intellectual property (IP) infringement associated with image generation models. It is, therefore, necessary to analyze the origin of images by inferring if a specific image was generated by a particular model, i.e., origin attribution. Existing methods only focus on specific types of generative models and require additional procedures during the training phase or generation phase. This makes them unsuitable for pre-trained models that lack these specific operations and may impair generation quality. To address this problem, we first develop an alteration-free and model-agnostic origin attribution method via reverse-engineering on image generation models, i.e., inverting the input of a particular model for a specific image. Given a particular model, we first analyze the differences in the hardness of reverse-engineering tasks for generated samples of the given model and other images. Based on our analysis, we then propose a method that utilizes the reconstruction loss of reverse-engineering to infer the origin. Our proposed method effectively distinguishes between generated images of a specific generative model and other images, i.e., images generated by other models and real images.

----

## [0] Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy

**Authors**: *Dongmin Park, Seola Choi, Doyoung Kim, Hwanjun Song, Jae-Gil Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebb6bee50913ba7e1efeb91a1d47a002-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebb6bee50913ba7e1efeb91a1d47a002-Abstract-Conference.html)

**Abstract**:

Data pruning, which aims to downsize a large training set into a small informative subset, is crucial for reducing the enormous computational costs of modern deep learning. Though large-scale data collections invariably contain annotation noise and numerous robust learning methods have been developed, data pruning for the noise-robust learning scenario has received little attention. With state-of-the-art Re-labeling methods that self-correct erroneous labels while training, it is challenging to identify which subset induces the most accurate re-labeling of erroneous labels in the entire training set. In this paper, we formalize the problem of data pruning with re-labeling. We first show that the likelihood of a training example being correctly re-labeled is proportional to the prediction confidence of its neighborhood in the subset. Therefore, we propose a novel data pruning algorithm, Prune4Rel, that finds a subset maximizing the total neighborhood confidence of all training examples, thereby maximizing the re-labeling accuracy and generalization performance. Extensive experiments on four real and one synthetic noisy datasets show that Prune4Rel outperforms the baselines with Re-labeling models by up to 9.1% as well as those with a standard model by up to 21.6%.

----

## [0] WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

**Authors**: *Sebastian Gerard, Yu Zhao, Josephine Sullivan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present a multi-temporal, multi-modal remote-sensing dataset for predicting how active wildfires will spread at a resolution of 24 hours. The dataset consists of 13607 images across 607 fire events in the United States from January 2018 to October 2021. For each fire event, the dataset contains a full time series of daily observations, containing detected active fires and variables related to fuel, topography and weather conditions. The dataset is challenging due to: a) its inputs being multi-temporal, b) the high number of 23 multi-modal input channels, c) highly imbalanced labels and d) noisy labels, due to smoke, clouds, and inaccuracies in the active fire detection. The underlying complexity of the physical processes adds to these challenges. Compared to existing public datasets in this area, WildfireSpreadTS allows for multi-temporal modeling of spreading wildfires, due to its time series structure. Furthermore, we provide additional input modalities and a high spatial resolution of 375m for the active fire maps. We publish this dataset to encourage further research on this important task with multi-temporal, noise-resistant or generative methods, uncertainty estimation or advanced optimization techniques that deal with the high-dimensional input space.

----

## [0] Augmenting Language Models with Long-Term Memory

**Authors**: *Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html)

**Abstract**:

Existing large language models (LLMs) can only afford fix-sized inputs due to the input length limit, preventing them from utilizing rich long-context information from past inputs. To address this, we propose a framework, Language Models Augmented with Long-Term Memory (LongMem), which enables LLMs to memorize long history. We design a novel decoupled network architecture with the original backbone LLM frozen as a memory encoder and an adaptive residual side-network as a memory retriever and reader. Such a decoupled memory design can easily cache and update long-term past contexts for memory retrieval without suffering from memory staleness. Enhanced with memory-augmented adaptation training, LongMem can thus memorize long past context and use long-term memory for language modeling. The proposed memory retrieval module can handle unlimited-length context in its memory bank to benefit various downstream tasks. Typically, LongMem can enlarge the long-form memory to 65k tokens and thus cache many-shot extra demonstration examples as long-form memory for in-context learning. Experiments show that our method outperforms strong long-context models on ChapterBreak, a challenging long-context modeling benchmark, and achieves remarkable improvements on memory-augmented in-context learning over LLMs. The results demonstrate that the proposed method is effective in helping language models to memorize and utilize long-form contents.

----

## [0] Expressivity-Preserving GNN Simulation

**Authors**: *Fabian Jogl, Maximilian Thiessen, Thomas Gärtner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebf95a6f3c575322da15d4fd0fc2b3c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebf95a6f3c575322da15d4fd0fc2b3c8-Abstract-Conference.html)

**Abstract**:

We systematically investigate graph transformations that enable standard message passing to simulate state-of-the-art graph neural networks (GNNs) without loss of expressivity. Using these, many state-of-the-art GNNs can be implemented with message passing operations from standard libraries, eliminating many sources of implementation issues and allowing for better code optimization. We distinguish between weak and strong simulation: weak simulation achieves the same expressivity only after several message passing steps while strong simulation achieves this after every message passing step. Our contribution leads to a direct way to translate common operations of non-standard GNNs to graph transformations that allow for strong or weak simulation. Our empirical evaluation shows competitive predictive performance of message passing on transformed graphs for various molecular benchmark datasets, in several cases surpassing the original GNNs.

----

## [0] Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?

**Authors**: *Fan Yao, Chuanhao Li, Karthik Abinav Sankararaman, Yiming Liao, Yan Zhu, Qifan Wang, Hongning Wang, Haifeng Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebfabf372037aaa4a8d92c9b457ece3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebfabf372037aaa4a8d92c9b457ece3e-Abstract-Conference.html)

**Abstract**:

The past decade has witnessed the flourishing of a new profession as media content creators, who rely on revenue streams from online content recommendation platforms. The reward mechanism employed by these platforms creates a competitive environment among creators which affects their production choices and, consequently, content distribution and system welfare. It is thus crucial to design the platform's reward mechanism in order to steer the creators' competition towards a desirable welfare outcome in the long run. This work makes two major contributions in this regard: first, we uncover a fundamental limit about a class of widely adopted mechanisms, coined \emph{Merit-based Monotone Mechanisms}, by showing that they inevitably lead to a constant fraction loss of the optimal welfare. To circumvent this limitation, we introduce \emph{Backward Rewarding Mechanisms} (BRMs) and show that the competition game resultant from BRMs possesses a potential game structure. BRMs thus naturally induce strategic creators' collective behaviors towards optimizing the potential function, which can be designed to match any given welfare metric. In addition, the class of BRM can be parameterized so that it allows the platform to directly optimize welfare within the feasible mechanism space even when the welfare metric is not explicitly defined.

----

## [0] Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data

**Authors**: *Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina K. Ramirez, Greggory Heller, Severine Durand, Shawn R. Olsen, Stefan Mihalas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html)

**Abstract**:

Recent advances in neuroscientific experimental techniques have enabled us to simultaneously record the activity of thousands of neurons across multiple brain regions. This has led to a growing need for computational tools capable of analyzing how task-relevant information is represented and communicated between several brain regions. Partial information decompositions (PIDs) have emerged as one such tool, quantifying how much unique, redundant and synergistic information two or more brain regions carry about a task-relevant message. However, computing PIDs is computationally challenging in practice, and statistical issues such as the bias and variance of estimates remain largely unexplored. In this paper, we propose a new method for efficiently computing and estimating a PID definition on multivariate Gaussian distributions. We show empirically that our method satisfies an intuitive additivity property, and recovers the ground truth in a battery of canonical examples, even at high dimensionality. We also propose and evaluate, for the first time, a method to correct the bias in PID estimates at finite sample sizes. Finally, we demonstrate that our Gaussian PID effectively characterizes inter-areal interactions in the mouse brain, revealing higher redundancy between visual areas when a stimulus is behaviorally relevant.

----

## [0] Unconstrained Dynamic Regret via Sparse Coding

**Authors**: *Zhiyu Zhang, Ashok Cutkosky, Yannis Paschalidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec2833cda146c277cdaa39066764f25c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec2833cda146c277cdaa39066764f25c-Abstract-Conference.html)

**Abstract**:

Motivated by the challenge of nonstationarity in sequential decision making, we study Online Convex Optimization (OCO) under the coupling of two problem structures: the domain is unbounded, and the comparator sequence $u_1,\ldots,u_T$ is arbitrarily time-varying. As no algorithm can guarantee low regret simultaneously against all comparator sequences, handling this setting requires moving from minimax optimality to comparator adaptivity. That is, sensible regret bounds should depend on certain complexity measures of the comparator relative to one's prior knowledge. This paper achieves a new type of such adaptive regret bounds leveraging a sparse coding framework. The complexity of the comparator is measured by its energy and its sparsity on a user-specified dictionary, which offers considerable versatility. For example, equipped with a wavelet dictionary, our framework improves the state-of-the-art bound (Jacobsen & Cutkosky, 2022) by adapting to both ($i$) the magnitude of the comparator average $||\bar u||=||\sum_{t=1}^Tu_t/T||$, rather than the maximum $\max_t||u_t||$; and ($ii$) the comparator variability $\sum_{t=1}^T||u_t-\bar u||$, rather than the uncentered sum $\sum_{t=1}^T||u_t||$. Furthermore, our proof is simpler due to decoupling function approximation from regret minimization.

----

## [0] Efficient Test-Time Adaptation for Super-Resolution with Second-Order Degradation and Reconstruction

**Authors**: *Zeshuai Deng, Zhuokun Chen, Shuaicheng Niu, Thomas H. Li, Bohan Zhuang, Mingkui Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html)

**Abstract**:

Image super-resolution (SR) aims to learn a mapping from low-resolution (LR) to high-resolution (HR) using paired HR-LR training images. Conventional SR methods typically gather the paired training data by synthesizing LR images from HR images using a predetermined degradation model, e.g., Bicubic down-sampling. However, the realistic degradation type of test images may mismatch with the training-time degradation type due to the dynamic changes of the real-world scenarios, resulting in inferior-quality SR images. To address this, existing methods attempt to estimate the degradation model and train an image-specific model, which, however, is quite time-consuming and impracticable to handle rapidly changing domain shifts. Moreover, these methods largely concentrate on the estimation of one degradation type (e.g., blur degradation), overlooking other degradation types like noise and JPEG in real-world test-time scenarios, thus limiting their practicality. To tackle these problems, we present an efficient test-time adaptation framework for SR, named SRTTA, which is able to quickly adapt SR models to test domains with different/unknown degradation types. Specifically, we design a second-order degradation scheme to construct paired data based on the degradation type of the test image, which is predicted by a pre-trained degradation classifier. Then, we adapt the SR model by implementing feature-level reconstruction learning from the initial test image to its second-order degraded counterparts, which helps the SR model generate plausible HR images. Extensive experiments are conducted on newly synthesized corrupted DIV2K datasets with 8 different degradations and several real-world datasets, demonstrating that our SRTTA framework achieves an impressive improvement over existing methods with satisfying speed. The source code is available at https://github.com/DengZeshuai/SRTTA.

----

## [0] Replicability in Reinforcement Learning

**Authors**: *Amin Karbasi, Grigoris Velegkas, Lin Yang, Felix Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html)

**Abstract**:

We initiate the mathematical study of replicability as an   algorithmic property in the context of reinforcement learning (RL).  We focus on the fundamental setting of discounted tabular MDPs with access to a generative model.  Inspired by Impagliazzo et al. [2022], we say that an RL algorithm is replicable if,  with high probability,  it outputs the exact same policy  after two executions on i.i.d. samples drawn from the generator  when its internal randomness  is the same.  We first provide   an efficient $\rho$-replicable algorithm for $(\varepsilon, \delta)$-optimal policy estimation  with sample and time complexity $\widetilde O\left(\frac{N^3\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$,  where $N$ is the number of state-action pairs.  Next,  for the subclass of deterministic algorithms,  we provide a lower bound of order $\Omega\left(\frac{N^3}{(1-\gamma)^3\cdot\varepsilon^2\cdot\rho^2}\right)$.  Then, we study a relaxed version of replicability proposed  by Kalavasis et al. [2023] called TV indistinguishability.  We design a computationally efficient TV indistinguishable algorithm for policy estimation  whose sample complexity is $\widetilde O\left(\frac{N^2\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.  At the cost of $\exp(N)$ running time,  we transform these TV indistinguishable algorithms to $\rho$-replicable ones without increasing their sample complexity.  Finally,  we introduce the notion of approximate-replicability  where we only require that two outputted policies are close  under an appropriate statistical divergence (e.g., Renyi)  and show an improved sample complexity of $\widetilde O\left(\frac{N\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.

----

## [0] Probabilistic Invariant Learning with Randomized Linear Classifiers

**Authors**: *Leonardo Cotta, Gal Yehuda, Assaf Schuster, Chris J. Maddison*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec4f0b0a7557d6a51c42308800f2c23a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec4f0b0a7557d6a51c42308800f2c23a-Abstract-Conference.html)

**Abstract**:

Designing models that are both expressive and preserve known invariances of tasks is an increasingly hard problem. Existing solutions tradeoff invariance for computational or memory resources. In this work, we show how to leverage randomness and design models that are both expressive and invariant but use less resources. Inspired by randomized algorithms, our key insight is that accepting probabilistic notions of universal approximation and invariance can reduce our resource requirements. More specifically, we propose a class of binary classification models called Randomized Linear Classifiers (RLCs). We give parameter and sample size conditions in which RLCs can, with high probability, approximate any (smooth) function while preserving invariance to compact group transformations. Leveraging this result, we design three RLCs that are provably probabilistic invariant for classification tasks over sets, graphs, and spherical data. We show how these models can achieve probabilistic invariance and universality using less resources than (deterministic) neural networks and their invariant counterparts. Finally, we empirically demonstrate the benefits of this new class of models on invariant tasks where deterministic invariant neural networks are known to struggle.

----

## [0] FedNAR: Federated Optimization with Normalized Annealing Regularization

**Authors**: *Junbo Li, Ang Li, Chong Tian, Qirong Ho, Eric P. Xing, Hongyi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec52572b9e16b91edff5dc70e2642240-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec52572b9e16b91edff5dc70e2642240-Abstract-Conference.html)

**Abstract**:

Weight decay is a standard technique to improve generalization performance in modern deep neural network optimization, and is also widely adopted in federated learning (FL) to prevent overfitting in local clients. In this paper, we first explore the choices of weight decay and identify that weight decay value appreciably influences the convergence of existing FL algorithms. While preventing overfitting is crucial, weight decay can introduce a different optimization goal towards the global objective, which is further amplified in FL due to multiple local updates and heterogeneous data distribution.To address this challenge, we develop {\it Federated optimization with Normalized Annealing Regularization} (FedNAR), a simple yet effective and versatile algorithmic plug-in that can be seamlessly integrated into any existing FL algorithms. Essentially, we regulate the magnitude of each update by performing co-clipping of the gradient and weight decay.We provide a comprehensive theoretical analysis of FedNAR's convergence rate and conduct extensive experiments on both vision and language datasets with different backbone federated optimization algorithms. Our experimental results consistently demonstrate that incorporating FedNAR into existing FL algorithms leads to accelerated convergence and heightened model accuracy. Moreover, FedNAR exhibits resilience in the face of various hyperparameter configurations. Specifically, FedNAR has the ability to self-adjust the weight decay when the initial specification is not optimal, while the accuracy of traditional FL algorithms would markedly decline. Our codes are released at \href{https://anonymous.4open.science/r/fednar-BE8F}{https://anonymous.4open.science/r/fednar-BE8F}.

----

## [0] How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources

**Authors**: *Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, Hannaneh Hajishirzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec6413875e4ab08d7bc4d8e225263398-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec6413875e4ab08d7bc4d8e225263398-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In this work we explore recent advances in instruction-tuning language models on a range of open instruction-following datasets. Despite recent claims that open models can be on par with state-of-the-art proprietary models, these claims are often accompanied by limited evaluation, making it difficult to compare models across the board and determine the utility of various resources. We provide a large set of instruction-tuned models from 6.7B to 65B parameters in size, trained on 12 instruction datasets ranging from manually curated (e.g., OpenAssistant) to synthetic and distilled (e.g., Alpaca) and systematically evaluate them on their factual knowledge, reasoning, multilinguality, coding, safety, and open-ended instruction following abilities through a collection of automatic, model-based, and human-based metrics. We further introduce Tülu, our best performing instruction-tuned model suite finetuned on a combination of high-quality open resources.Our experiments show that different instruction-tuning datasets can uncover or enhance specific skills, while no single dataset (or combination) provides the best performance across all evaluations. Interestingly, we find that model and human preference-based evaluations fail to reflect differences in model capabilities exposed by benchmark-based evaluations, suggesting the need for the type of systemic evaluation performed in this work. Our evaluations show that the best model in any given evaluation reaches on average 87% of ChatGPT performance, and 73% of GPT-4 performance, suggesting that further investment in building better base models and instruction-tuning data is required to close the gap. We release our instruction-tuned models, including a fully finetuned 65B Tülu, along with our code, data, and evaluation framework to facilitate future research.

----

## [0] Trial matching: capturing variability with data-constrained spiking neural networks

**Authors**: *Christos Sourmpis, Carl C. H. Petersen, Wulfram Gerstner, Guillaume Bellec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec702dd6e83b2113a43614685a7e2ac6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec702dd6e83b2113a43614685a7e2ac6-Abstract-Conference.html)

**Abstract**:

Simultaneous behavioral and electrophysiological recordings call for new methods to reveal the interactions between neural activity and behavior. A milestone would be an interpretable model of the co-variability of spiking activity and behavior across trials. Here, we model a mouse cortical sensory-motor pathway in a tactile detection task reported by licking with a large recurrent spiking neural network (RSNN), fitted to the recordings via gradient-based optimization. We focus specifically on the difficulty to match the trial-to-trial variability in the data. Our solution relies on optimal transport to define a distance between the distributions of generated and recorded trials. The technique is applied to artificial data and neural recordings covering six cortical areas. We find that the resulting RSNN can generate realistic cortical activity and predict jaw movements across the main modes of trial-to-trial variability. Our analysis also identifies an unexpected mode of variability in the data corresponding to task-irrelevant movements of the mouse.

----

## [0] Decentralized Randomly Distributed Multi-agent Multi-armed Bandit with Heterogeneous Rewards

**Authors**: *Mengfan Xu, Diego Klabjan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html)

**Abstract**:

We study a decentralized multi-agent multi-armed bandit problem in which multiple clients are connected by time dependent random graphs provided by an environment. The reward distributions of each arm vary across clients and rewards are generated independently over time by an environment based on distributions that include both sub-exponential and sub-gaussian distributions. Each client pulls an arm and communicates with neighbors based on the graph provided by the environment. The goal is to minimize the overall regret of the entire system through collaborations. To this end, we introduce a novel algorithmic framework, which first provides robust simulation methods for generating random graphs using rapidly mixing markov chains or the random graph model, and then combines an averaging-based consensus approach with a newly proposed weighting technique and the upper confidence bound to deliver a UCB-type solution. Our algorithms account for the randomness in the graphs, removing the conventional doubly stochasticity assumption, and only require the knowledge of the number of clients at initialization. We derive optimal instance-dependent regret upper bounds of order $\log{T}$ in both sub-gaussian and sub-exponential environments, and a nearly optimal instance-free regret upper bound of order $\sqrt{T}\log T$ up to a $\log T$ factor. Importantly, our regret bounds hold with high probability and capture graph randomness, whereas prior works consider expected regret under assumptions and require more stringent reward distributions.

----

## [0] (Amplified) Banded Matrix Factorization: A unified approach to private training

**Authors**: *Christopher A. Choquette-Choo, Arun Ganesh, Ryan McKenna, H. Brendan McMahan, John Rush, Abhradeep Guha Thakurta, Zheng Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecc28b4ce9b39f5f23c3efb03e25b7bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecc28b4ce9b39f5f23c3efb03e25b7bf-Abstract-Conference.html)

**Abstract**:

Matrix factorization (MF) mechanisms for differential privacy (DP) have substantially improved the state-of-the-art in privacy-utility-computation tradeoffs for ML applications in a variety of scenarios, but in both the centralized and federated settings there remain instances where either MF cannot be easily applied, or other algorithms provide better tradeoffs (typically, as $\epsilon$ becomes small).In this work, we show how MF can subsume prior state-of-the-art algorithms in both federated and centralized training settings, across all privacy budgets. The key technique throughout is the construction of MF mechanisms with banded matrices (lower-triangular matrices with at most $\hat{b}$ nonzero bands including the main diagonal). For cross-device federated learning (FL), this enables multiple-participations with a relaxed device participation schema compatible with practical FL infrastructure (as demonstrated by a production deployment).  In the centralized setting, we prove that banded matrices enjoy the same privacy amplification results as the ubiquitous DP-SGD algorithm, but can provide strictly better performance  in most scenarios---this lets us always at least match DP-SGD, and often outperform it

----

## [0] Anchor Data Augmentation

**Authors**: *Nora Schneider, Shirin Goshtasbpour, Fernando Pérez-Cruz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecc9b6dfdbe374c0a3364ff81cd28642-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecc9b6dfdbe374c0a3364ff81cd28642-Abstract-Conference.html)

**Abstract**:

We propose a novel algorithm for data augmentation in nonlinear over-parametrized regression. Our data augmentation algorithm borrows from the literature on causality. Contrary to the current state-of-the-art solutions that rely on modifications of Mixup algorithm, we extend the recently proposed distributionally robust Anchor regression (AR) method for data augmentation. Our Anchor Data Augmentation (ADA) uses several replicas of the modified samples in AR to provide more training examples, leading to more robust regression predictions. We apply ADA to linear and nonlinear regression problems using neural networks. ADA is competitive with state-of-the-art C-Mixup solutions.

----

## [0] The noise level in linear regression with dependent data

**Authors**: *Ingvar M. Ziemann, Stephen Tu, George J. Pappas, Nikolai Matni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecffd829f90b0a4b6aa017b6df15904f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecffd829f90b0a4b6aa017b6df15904f-Abstract-Conference.html)

**Abstract**:

We derive upper bounds for random design linear regression with dependent ($\beta$-mixing) data absent any realizability assumptions.  In contrast to the strictly realizable martingale noise regime, no sharp \emph{instance-optimal} non-asymptotics are available in the literature. Up to constant factors, our analysis correctly recovers the variance term predicted by the Central Limit Theorem---the noise level of the problem---and thus exhibits graceful degradation as we introduce misspecification. Past a burn-in, our result is sharp in the moderate deviations regime, and in particular does not inflate the leading order term by mixing time factors.

----

## [0] SafeDICE: Offline Safe Imitation Learning with Non-Preferred Demonstrations

**Authors**: *Youngsoo Jang, Geon-Hyeong Kim, Jongmin Lee, Sungryull Sohn, Byoungjip Kim, Honglak Lee, Moontae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed2fb79f2664c3d9ba878be7e575b2af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed2fb79f2664c3d9ba878be7e575b2af-Abstract-Conference.html)

**Abstract**:

We consider offline safe imitation learning (IL), where the agent aims to learn the safe policy that mimics preferred behavior while avoiding non-preferred behavior from non-preferred demonstrations and unlabeled demonstrations. This problem setting corresponds to various real-world scenarios, where satisfying safety constraints is more important than maximizing the expected return. However, it is very challenging to learn the policy to avoid constraint-violating (i.e. non-preferred) behavior, as opposed to standard imitation learning which learns the policy to mimic given demonstrations. In this paper, we present a hyperparameter-free offline safe IL algorithm, SafeDICE, that learns safe policy by leveraging the non-preferred demonstrations in the space of stationary distributions. Our algorithm directly estimates the stationary distribution corrections of the policy that imitate the demonstrations excluding the non-preferred behavior. In the experiments, we demonstrate that our algorithm learns a more safe policy that satisfies the cost constraint without degrading the reward performance, compared to baseline algorithms.

----

## [0] Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

**Authors**: *Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) can achieve strong performance on many tasks by producing step-by-step reasoning before giving a final output, often referred to as chain-of-thought reasoning (CoT). It is tempting to interpret these CoT explanations as the LLM's process for solving a task. This level of transparency into LLMs' predictions would yield significant safety benefits. However, we find that CoT explanations can systematically misrepresent the true reason for a model's prediction. We demonstrate that CoT explanations can be heavily influenced by adding biasing features to model inputs—e.g., by reordering the multiple-choice options in a few-shot prompt to make the answer always "(A)"—which models systematically fail to mention in their explanations. When we bias models toward incorrect answers, they frequently generate CoT explanations rationalizing those answers. This causes accuracy to drop by as much as 36% on a suite of 13 tasks from BIG-Bench Hard, when testing with GPT-3.5 from OpenAI and Claude 1.0 from Anthropic. On a social-bias task, model explanations justify giving answers in line with stereotypes without mentioning the influence of these social biases. Our findings indicate that CoT explanations can be plausible yet misleading, which risks increasing our trust in LLMs without guaranteeing their safety. Building more transparent and explainable systems will require either improving CoT faithfulness through targeted efforts or abandoning CoT in favor of alternative methods.

----

## [0] Static and Sequential Malicious Attacks in the Context of Selective Forgetting

**Authors**: *Chenxu Zhao, Wei Qian, Rex Ying, Mengdi Huai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed4bacc8c7ca1ee0e1d4e0ef376b7ac7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed4bacc8c7ca1ee0e1d4e0ef376b7ac7-Abstract-Conference.html)

**Abstract**:

With the growing demand for the right to be forgotten, there is an increasing need for machine learning models to forget sensitive data and its impact. To address this, the paradigm of selective forgetting (a.k.a machine unlearning) has been extensively studied, which aims to remove the impact of requested data from a well-trained model without retraining from scratch. Despite its significant success, limited attention has been given to the security vulnerabilities of the unlearning system concerning malicious data update requests. Motivated by this, in this paper, we explore the possibility and feasibility of malicious data update requests during the unlearning process. Specifically, we first propose a new class of malicious selective forgetting attacks, which involves a static scenario where all the malicious data update requests are provided by the adversary at once. Additionally, considering the sequential setting where the data update requests arrive sequentially, we also design a novel framework for sequential forgetting attacks, which is formulated as a stochastic optimal control problem. We also propose novel optimization algorithms that can find the effective malicious data update requests. We perform theoretical analyses for the proposed selective forgetting attacks, and extensive experimental results validate the effectiveness of our proposed selective forgetting attacks. The source code is available in the supplementary material.

----

## [0] Language-based Action Concept Spaces Improve Video Self-Supervised Learning

**Authors**: *Kanchana Ranasinghe, Michael S. Ryoo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html)

**Abstract**:

Recent contrastive language image pre-training has led to learning highly transferable and robust image representations. However, adapting these models to video domain with minimal supervision remains an open problem. We explore a simple step in that direction, using language tied self-supervised learning to adapt an image CLIP model to the video domain. A backbone modified for temporal modeling is trained under self-distillation settings with train objectives operating in an action concept space. Feature vectors of various action concepts extracted from a language encoder using relevant textual prompts construct this space. A large language model aware of actions and their attributes generates the relevant textual prompts.We introduce two train objectives, concept distillation and concept alignment, that retain generality of original representations while enforcing relations between actions and their attributes. Our approach improves zero-shot and linear probing performance on three action recognition benchmarks.

----

## [0] TRIAGE: Characterizing and auditing training data for improved regression

**Authors**: *Nabeel Seedat, Jonathan Crabbé, Zhaozhi Qian, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed687a5f52b651b19e7c18f702907b8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed687a5f52b651b19e7c18f702907b8b-Abstract-Conference.html)

**Abstract**:

Data quality is crucial for robust machine learning algorithms, with the recent interest in data-centric AI emphasizing the importance of training data characterization. However, current data characterization methods are largely focused on classification settings, with regression settings largely understudied. To address this, we introduce TRIAGE, a novel data characterization framework tailored to regression tasks and compatible with a broad class of regressors. TRIAGE utilizes conformal predictive distributions to provide a model-agnostic scoring method, the TRIAGE score. We operationalize the score to analyze individual samples' training dynamics and characterize samples as under-, over-, or well-estimated by the model. We show that TRIAGE's characterization is consistent and highlight its utility to improve performance via data sculpting/filtering, in multiple regression settings. Additionally, beyond sample level, we show TRIAGE enables new approaches to dataset selection and feature acquisition. Overall, TRIAGE highlights the value unlocked by data characterization in real-world regression applications.

----

## [0] ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling

**Authors**: *Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, Aditya Grover*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed73c36e771881b232ef35fa3a1dec14-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed73c36e771881b232ef35fa3a1dec14-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Modeling weather and climate is an essential endeavor to understand the near- and long-term impacts of climate change, as well as to inform technology and policymaking for adaptation and mitigation efforts. In recent years, there has been a surging interest in applying data-driven methods based on machine learning for solving core problems such as weather forecasting and climate downscaling. Despite promising results, much of this progress has been impaired due to the lack of large-scale, open-source efforts for reproducibility, resulting in the use of inconsistent or underspecified datasets, training setups, and evaluations by both domain scientists and artificial intelligence researchers. We introduce ClimateLearn, an open-source PyTorch library that vastly simplifies the training and evaluation of machine learning models for data-driven climate science. ClimateLearn consists of holistic pipelines for dataset processing (e.g., ERA5, CMIP6, PRISM), implementing state-of-the-art deep learning models (e.g., Transformers, ResNets), and quantitative and qualitative evaluation for standard weather and climate modeling tasks. We supplement these functionalities with extensive documentation, contribution guides, and quickstart tutorials to expand access and promote community growth. We have also performed comprehensive forecasting and downscaling experiments to showcase the capabilities and key features of our library. To our knowledge, ClimateLearn is the first large-scale, open-source effort for bridging research in weather and climate modeling with modern machine learning systems. Our library is available publicly at https://github.com/aditya-grover/climate-learn.

----

## [0] Advice Querying under Budget Constraint for Online Algorithms

**Authors**: *Ziyad Benomar, Vianney Perchet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eda830e16044587b5082a853c4f25a90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eda830e16044587b5082a853c4f25a90-Abstract-Conference.html)

**Abstract**:

Several problems have been extensively studied in the learning-augmented setting, where the algorithm has access to some, possibly incorrect, predictions. However, it is assumed in most works that the predictions are provided to the algorithm as input, with no constraint on their size. In this paper, we consider algorithms with access to a limited number of predictions, that they can request at any time during their execution. We study three classical problems in competitive analysis, the ski rental problem, the secretary problem, and the non-clairvoyant job scheduling. We address the question of when to query predictions and how to use them.

----

## [0] DIFFER: Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning

**Authors**: *Xunhan Hu, Jian Zhao, Wengang Zhou, Ruili Feng, Houqiang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html)

**Abstract**:

Cooperative multi-agent reinforcement learning (MARL) is a challenging task, as agents must learn complex and diverse individual strategies from a shared team reward. However, existing methods struggle to distinguish and exploit important individual experiences, as they lack an effective way to decompose the team reward into individual rewards. To address this challenge, we propose DIFFER, a powerful theoretical framework for decomposing individual rewards to enable fair experience replay in MARL.By enforcing the invariance of network gradients, we establish a partial differential equation whose solution yields the underlying individual reward function. The individual TD-error can then be computed from the solved closed-form individual rewards, indicating the importance of each piece of experience in the learning task and guiding the training process. Our method elegantly achieves an equivalence to the original learning framework when individual experiences are homogeneous, while also adapting to achieve more muscular efficiency and fairness when diversity is observed.Our extensive experiments on popular benchmarks validate the effectiveness of our theory and method, demonstrating significant improvements in learning efficiency and fairness. Code is available in supplement material.

----

## [0] Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

**Authors**: *Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html)

**Abstract**:

Transformer models have been widely adopted in various domains over the last years and especially large language models have advanced the field of AI significantly. Due to their size, the capability of these networks has increased tremendously, but this has come at the cost of a significant increase in necessary compute. Quantization is one of the most effective ways for reducing the computational time and memory consumption of neural networks. Many studies have shown, however, that modern transformer models tend to learn strong outliers in their activations, making them difficult to quantize. To retain acceptable performance, the existence of these outliers requires activations to be in higher-bitwidth or the use of different numeric formats, extra fine-tuning, or other workarounds. We show that strong outliers are related to very specific behavior of attention heads that try to learn a "no-op", or just a partial update of the residual. To achieve the exact zeros needed in the attention matrix for a no-update, the input to the softmax is pushed to be larger and larger during training, causing outliers in other parts of the network. Based on these observations, we propose two simple (independent) modifications to the attention mechanism - clipped softmax and gated attention. We empirically show that models pre-trained using our methods learn significantly smaller outliers while maintaining and sometimes even improving the floating-point task performance. This enables us to quantize transformers to full INT8 quantization of the activations without any additional effort. We demonstrate the effectiveness of our methods on both language models (BERT, OPT) and vision transformers.

----

## [0] Adversarial Training from Mean Field Perspective

**Authors**: *Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edcd1aa172dceda2ea9d45a48f25d3e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edcd1aa172dceda2ea9d45a48f25d3e3-Abstract-Conference.html)

**Abstract**:

Although adversarial training is known to be effective against adversarial examples, training dynamics are not well understood. In this study, we present the first theoretical analysis of adversarial training in random deep neural networks without any assumptions on data distributions. We introduce a new theoretical framework based on mean field theory, which addresses the limitations of existing mean field-based approaches. Based on the framework, we derive the (empirically tight) upper bounds of $\ell_q$ norm-based adversarial loss with $\ell_p$ norm-based adversarial examples for various values of $p$ and $q$. Moreover, we prove that networks without shortcuts are generally not adversarially trainable and that adversarial training reduces network capacity. We also show that the network width alleviates these issues. Furthermore, the various impacts of input and output dimensions on the upper bounds and time evolution of weight variance are presented.

----

## [0] MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting

**Authors**: *Felix Biggs, Antonin Schrab, Arthur Gretton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html)

**Abstract**:

We propose novel statistics which maximise the power  of a two-sample test based on the Maximum Mean Discrepancy (MMD), byadapting over the set of kernels used in defining it.For finite sets, this reduces to combining (normalised) MMD values under each of these kernels via a weighted soft maximum.Exponential concentration bounds are proved for our proposed statistics under the null and alternative.We further show how these kernels can be chosen in a data-dependent but permutation-independent way, in a well-calibrated test, avoiding data splitting.This technique applies more broadly to general permutation-based MMD testing, and includes the use of deep kernels with features learnt using unsupervised models such as auto-encoders.We highlight the applicability of our MMD-Fuse tests on both synthetic low-dimensional and real-world high-dimensional data, and compare its performance in terms of power against current state-of-the-art kernel tests.

----

## [0] DAC-DETR: Divide the Attention Layers and Conquer

**Authors**: *Zhengdong Hu, Yifan Sun, Jingdong Wang, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html)

**Abstract**:

This paper reveals a characteristic of DEtection Transformer (DETR) that negatively impacts its training efficacy, i.e., the cross-attention and self-attention layers in DETR decoder have contrary impacts on the object queries (though both impacts are important). Specifically, we observe the cross-attention tends to gather multiple queries around the same object, while the self-attention disperses these queries far away. To improve the training efficacy, we propose a Divide-And-Conquer DETR (DAC-DETR) that divides the cross-attention out from this contrary for better conquering. During training, DAC-DETR employs an auxiliary decoder that focuses on learning the cross-attention layers. The auxiliary decoder, while sharing all the other parameters, has NO self-attention layers and employs one-to-many label assignment to improve the gathering effect. Experiments show that DAC-DETR brings remarkable improvement over popular DETRs. For example, under the 12 epochs training scheme on MS-COCO, DAC-DETR improves Deformable DETR (ResNet-50) by +3.4 AP and achieves 50.9 (ResNet-50) / 58.1 AP (Swin-Large) based on some popular methods (i.e., DINO and an IoU-related loss). Our code will be made available at https://github.com/huzhengdongcs/DAC-DETR.

----

## [0] Streaming Algorithms and Lower Bounds for Estimating Correlation Clustering Cost

**Authors**: *Sepehr Assadi, Vihan Shah, Chen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1a1ecc92f35702b5c29dad3dc909ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1a1ecc92f35702b5c29dad3dc909ea-Abstract-Conference.html)

**Abstract**:

Correlation clustering is a fundamental optimization problem at the intersection of machine learning and theoretical computer science. Motivated by applications to big data processing, recent years have witnessed a flurry of results on this problem in the streaming model. In this model, the algorithm needs to process the input $n$-vertex graph by making one or few passes over the stream of its edges and using a limited memory, much smaller than the input size. All previous work on streaming correlation clustering have focused on semi-streaming algorithms with $\Omega(n)$ memory, whereas in this work, we study streaming algorithms with much smaller memory requirement of only $\text{polylog}{(n)}$ bits. This stringent memory requirement is in the same spirit of classical streaming algorithms that instead of recovering a full solution to the problem---which can be prohibitively large with such small memory as is the case in our problem---, aimed to learn certain statistical properties of their inputs. In our case, this translates to determining the ``(correlation) clusterability'' of input graphs, or more precisely, estimating the cost of the optimal correlation clustering solution. As our main result, we present two novel algorithms that in only $\text{polylog}{(n)}$ space are able to estimate the optimal correlation clustering cost up to some constant multiplicative factor plus some extra additive error. One of the algorithms outputs a $3$-multiplicative approximation plus $o(n^2)$ additive approximation, and the other one improves the additive error further down at the cost of increasing the multiplicative factor to some large constant. We then present new lower bounds that justify this mix of both multiplicative and additive error approximation in our algorithms.

----

## [0] Episodic Multi-Task Learning with Heterogeneous Neural Processes

**Authors**: *Jiayi Shen, Xiantong Zhen, Qi Wang, Marcel Worring*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html)

**Abstract**:

This paper focuses on the data-insufficiency problem in multi-task learning within an episodic training setup. Specifically, we explore the potential of heterogeneous information across tasks and meta-knowledge among episodes to effectively tackle each task with limited data. Existing meta-learning methods often fail to take advantage of crucial heterogeneous information in a single episode, while multi-task learning models neglect reusing experience from earlier episodes. To address the problem of insufficient data, we develop Heterogeneous Neural Processes (HNPs) for the episodic multi-task setup. Within the framework of hierarchical Bayes, HNPs effectively capitalize on prior experiences as meta-knowledge and capture task-relatedness among heterogeneous tasks, mitigating data-insufficiency. Meanwhile, transformer-structured inference modules are designed to enable efficient inferences toward meta-knowledge and task-relatedness. In this way, HNPs can learn more powerful functional priors for adapting to novel heterogeneous tasks in each meta-test episode. Experimental results show the superior performance of the proposed HNPs over typical baselines, and ablation studies verify the effectiveness of the designed inference modules.

----

## [0] Knowledge Distillation Performs Partial Variance Reduction

**Authors**: *Mher Safaryan, Alexandra Peste, Dan Alistarh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html)

**Abstract**:

Knowledge distillation is a popular approach for enhancing the performance of "student" models, with lower representational capacity, by taking advantage of more powerful "teacher" models. Despite its apparent simplicity, the underlying mechanics behind knowledge distillation (KD) are not yet fully understood. In this work, we shed new light on the inner workings of this method, by examining it from an optimization perspective. Specifically, we show that, in the context of linear and deep linear models, KD can be interpreted as a novel type of stochastic variance reduction mechanism. We provide a detailed convergence analysis of the resulting dynamics, which hold under standard assumptions for both strongly-convex and non-convex losses, showing that KD acts as a form of \emph{partial variance reduction}, which can reduce the stochastic gradient noise, but may not eliminate it completely, depending on the properties of the ``teacher'' model. Our analysis puts further emphasis on the need for careful parametrization of KD, in particular w.r.t. the weighting of the distillation loss, and is validated empirically on both linear models and deep neural networks.

----

## [0] Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels

**Authors**: *Zifu Wang, Xuefei Ning, Matthew B. Blaschko*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html)

**Abstract**:

Intersection over Union (IoU) losses are surrogates that directly optimize the Jaccard index. Leveraging IoU losses as part of the loss function have demonstrated superior performance in semantic segmentation tasks compared to optimizing pixel-wise losses such as the cross-entropy loss alone. However, we identify a lack of flexibility in these losses to support vital training techniques like label smoothing, knowledge distillation, and semi-supervised learning, mainly due to their inability to process soft labels. To address this, we introduce Jaccard Metric Losses (JMLs), which are identical to the soft Jaccard loss in standard settings with hard labels but are fully compatible with soft labels. We apply JMLs to three prominent use cases of soft labels: label smoothing, knowledge distillation and semi-supervised learning, and demonstrate their potential to enhance model accuracy and calibration. Our experiments show consistent improvements over the cross-entropy loss across 4 semantic segmentation datasets (Cityscapes, PASCAL VOC, ADE20K, DeepGlobe Land) and 13 architectures, including classic CNNs and recent vision transformers. Remarkably, our straightforward approach significantly outperforms state-of-the-art knowledge distillation and semi-supervised learning methods. The code is available at \href{https://github.com/zifuwanggg/JDTLosses}{https://github.com/zifuwanggg/JDTLosses}.

----

## [0] Towards Stable Backdoor Purification through Feature Shift Tuning

**Authors**: *Rui Min, Zeyu Qin, Li Shen, Minhao Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html)

**Abstract**:

It has been widely observed that deep neural networks (DNN) are vulnerable to backdoor attacks where attackers could manipulate the model behavior maliciously by tampering with a small set of training samples. Although a line of defense methods is proposed to mitigate this threat, they either require complicated modifications to the training process or heavily rely on the specific model architecture, which makes them hard to deploy into real-world applications. Therefore, in this paper, we instead start with fine-tuning, one of the most common and easy-to-deploy backdoor defenses, through comprehensive evaluations against diverse attack scenarios. Observations made through initial experiments show that in contrast to the promising defensive results on high poisoning rates, vanilla tuning methods completely fail at low poisoning rate scenarios. Our analysis shows that with the low poisoning rate, the entanglement between backdoor and clean features undermines the effect of tuning-based defenses. Therefore, it is necessary to disentangle the backdoor and clean features in order to improve backdoor purification. To address this, we introduce Feature Shift Tuning (FST), a method for tuning-based backdoor purification. Specifically, FST encourages feature shifts by actively deviating the classifier weights from the originally compromised weights. Extensive experiments demonstrate that our FST provides consistently stable performance under different attack settings. Without complex parameter adjustments, FST also achieves much lower tuning costs, only $10$ epochs. Our codes are available at https://github.com/AISafety-HKUST/stable_backdoor_purification.

----

## [0] Scalable 3D Captioning with Pretrained Models

**Authors**: *Tiange Luo, Chris Rockwell, Honglak Lee, Justin Johnson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee4814f9bce0cae7991d3341bb081b55-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee4814f9bce0cae7991d3341bb081b55-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Cap3D, an automatic approach for generating descriptive text for 3D objects. This approach utilizes pretrained models from image captioning, image-text alignment, and LLM to consolidate captions from multiple views of a 3D asset, completely side-stepping the time-consuming and costly process of manual annotation. We apply Cap3D to the recently introduced large-scale 3D dataset, Objaverse, resulting in 660k 3D-text pairs. Our evaluation, conducted using 41k human annotations from the same dataset, demonstrates that Cap3D surpasses human-authored descriptions in terms of quality, cost, and speed. Through effective prompt engineering, Cap3D rivals human performance in generating geometric descriptions on 17k collected annotations from the ABO dataset. Finally, we finetune Text-to-3D models on Cap3D and human captions, and show Cap3D outperforms; and benchmark the SOTA including Point·E, Shape·E, and DreamFusion.

----

## [0] Langevin Quasi-Monte Carlo

**Authors**: *Sifan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html)

**Abstract**:

Langevin Monte Carlo (LMC) and its stochastic gradient versions are powerful algorithms for sampling from complex high-dimensional distributions. To sample from a distribution with density $\pi(\theta)\propto \exp(-U(\theta)) $, LMC iteratively generates the next sample by taking a step in the gradient direction $\nabla U$ with added Gaussian perturbations. Expectations w.r.t. the target distribution $\pi$ are estimated by averaging over LMC samples. In ordinary Monte Carlo, it is well known that the estimation error can be substantially reduced by replacing independent random samples by quasi-random samples like low-discrepancy sequences. In this work, we show that the estimation error of LMC can also be reduced by using quasi-random samples. Specifically, we propose to use completely uniformly distributed (CUD) sequences with certain low-discrepancy property to generate the Gaussian perturbations. Under smoothness and convexity conditions, we prove that LMC with a low-discrepancy CUD sequence achieves smaller error than standard LMC. The theoretical analysis is supported by compelling numerical experiments, which demonstrate the effectiveness of our approach.

----

## [0] LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting

**Authors**: *Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, Roger Zimmermann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Road traffic forecasting plays a critical role in smart city initiatives and has experienced significant advancements thanks to the power of deep learning in capturing non-linear patterns of traffic data. However, the promising results achieved on current public datasets may not be applicable to practical scenarios due to limitations within these datasets. First, the limited sizes of them may not reflect the real-world scale of traffic networks. Second, the temporal coverage of these datasets is typically short, posing hurdles in studying long-term patterns and acquiring sufficient samples for training deep models. Third, these datasets often lack adequate metadata for sensors, which compromises the reliability and interpretability of the data. To mitigate these limitations, we introduce the LargeST benchmark dataset. It encompasses a total number of 8,600 sensors in California with a 5-year time coverage and includes comprehensive metadata. Using LargeST, we perform in-depth data analysis to extract data insights, benchmark well-known baselines in terms of their performance and efficiency, and identify challenges as well as opportunities for future research. We release the datasets and baseline implementations at: https://github.com/liuxu77/LargeST.

----

## [0] What Can We Learn from Unlearnable Datasets?

**Authors**: *Pedro Sandoval Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html)

**Abstract**:

In an era of widespread web scraping, unlearnable dataset methods have the potential to protect data privacy by preventing deep neural networks from generalizing. But in addition to a number of practical limitations that make their use unlikely, we make a number of findings that call into question their ability to safeguard data. First, it is widely believed that neural networks trained on unlearnable datasets only learn shortcuts, simpler rules that are not useful for generalization. In contrast, we find that networks actually can learn useful features that can be reweighed for high test performance, suggesting that image protection is not assured. Unlearnable datasets are also believed to induce learning shortcuts through linear separability of added perturbations. We provide a counterexample, demonstrating that linear separability of perturbations is not a necessary condition. To emphasize why linearly separable perturbations should not be relied upon, we propose an orthogonal projection attack which allows learning from unlearnable datasets published in ICML 2021 and ICLR 2023. Our proposed attack is significantly less complex than recently proposed techniques.

----

## [0] Language Models Meet World Models: Embodied Experiences Enhance Language Models

**Authors**: *Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, Zhiting Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee6630dcbcff857026e474fc857aa9f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee6630dcbcff857026e474fc857aa9f0-Abstract-Conference.html)

**Abstract**:

While large language models (LMs) have shown remarkable capabilities across numerous tasks, they often struggle with simple reasoning and planning in physical environments, such as understanding object permanence or planning household activities. The limitation arises from the fact that LMs are trained only on written text and miss essential embodied knowledge and skills. In this paper, we propose a new paradigm of enhancing LMs by finetuning them with world models, to gain diverse embodied knowledge while retaining their general language capabilities. Our approach deploys an embodied agent in a world model, particularly a simulator of the physical world (VirtualHome), and acquires a diverse set of embodied experiences through both goal-oriented planning and random exploration. These experiences are then used to finetune LMs to teach diverse abilities of reasoning and acting in the physical world, e.g., planning and completing goals, object permanence and tracking, etc. Moreover, it is desirable to preserve the generality of LMs during finetuning, which facilitates generalizing the embodied knowledge across tasks rather than being tied to specific simulations. We thus further introduce the classical elastic weight consolidation (EWC) for selective weight updates, combined with low-rank adapters (LoRA) for training efficiency. Extensive experiments show our approach substantially improves base LMs on 18 downstream tasks by 64.28% on average. In particular, the small LMs (1.3B, 6B, and 13B) enhanced by our approach match or even outperform much larger LMs (e.g., ChatGPT).

----

## [0] Activity Grammars for Temporal Action Segmentation

**Authors**: *Dayoung Gong, Joonseok Lee, Deunsol Jung, Suha Kwak, Minsu Cho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee6c4b99b4c0d3d60efd22c1ecdd9891-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee6c4b99b4c0d3d60efd22c1ecdd9891-Abstract-Conference.html)

**Abstract**:

Sequence prediction on temporal data requires the ability to understand compositional structures of multi-level semantics beyond individual and contextual properties of parts. The task of temporal action segmentation remains challenging for the reason, aiming at translating an untrimmed activity video into a sequence of action segments. This paper addresses the problem by introducing an effective activity grammar to guide neural predictions for temporal action segmentation.  We propose a novel grammar induction algorithm, dubbed KARI, that extracts a powerful context-free grammar from action sequence data. We also develop an efficient generalized parser, dubbed BEP, that transforms frame-level probability distributions into a reliable sequence of actions according to the induced grammar with recursive rules. Our approach can be combined with any neural network for temporal action segmentation to enhance the sequence prediction and discover its compositional structure. Experimental results demonstrate that our method significantly improves temporal action segmentation in terms of both performance and interpretability on two standard benchmarks, Breakfast and 50 Salads.

----

## [0] SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection

**Authors**: *Daehyun Kim, Sungyong Baik, Tae Hyun Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html)

**Abstract**:

Visual anomaly detection, the task of detecting abnormal characteristics in images, is challenging due to the rarity and unpredictability of anomalies. In order to reliably model the distribution of normality and detect anomalies, a few works have attempted to exploit the density estimation ability of normalizing flow (NF). However, previous NF-based methods have relied solely on the capability of NF and forcibly transformed the distribution of all features to a single distribution (e.g., unit normal distribution), when features can have different semantic information and thus follow different distributions. We claim that forcibly learning to transform such diverse distributions to a single distribution with a single network will cause the learning difficulty, limiting the capacity of a network to discriminate normal and abnormal data. As such, we propose to transform the distribution of features at each location of a given image to different distributions. In particular, we train NF to map normal data distribution to distributions with the same mean but different variances at each location of the given image. To enhance the discriminability, we also train NF to map abnormal data distribution to a distribution with a mean that is different from that of normal data, where abnormal data is synthesized with data augmentation. The experimental results outline the effectiveness of the proposed framework in improving the density modeling and thus anomaly detection performance.

----

## [0] AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling

**Authors**: *Sachin Chauhan, Zeel Bharatkumar Patel, Sayan Ranu, Rijurekha Sen, Nipun Batra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee799aff607fcf39c01df6391e96f92c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee799aff607fcf39c01df6391e96f92c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Air pollution poses serious health concerns  in developing countries, such as India, necessitating large-scale measurement for correlation analysis, policy recommendations, and informed decision-making. However, fine-grained data collection is costly.  Specifically, static sensors for pollution measurement cost several thousand dollars per unit, leading to inadequate deployment and coverage. To complement the existing sparse static sensor network, we propose a mobile sensor network utilizing lower-cost PM2.5 sensors mounted on public buses in the Delhi-NCR region of India. Through this exercise, we introduce a novel dataset AirDelhi comprising PM2.5 and PM10 measurements. This dataset is made publicly available, at https://www.cse.iitd.ac.in/pollutiondata, serving as a valuable resource for machine learning (ML) researchers and environmentalists. We present three key contributions with the release of this dataset. Firstly, through in-depth statistical analysis, we demonstrate that the released dataset significantly differs from existing pollution datasets, highlighting its uniqueness and potential for new insights. Secondly, the dataset quality been validated against existing expensive sensors. Thirdly, we conduct a benchmarking exercise (https://github.com/sachin-iitd/DelhiPMDatasetBenchmark), evaluating state-of-the-art methods for interpolation, feature imputation, and forecasting on this dataset, which is the largest publicly available PM dataset to date. The results of the benchmarking exercise underscore the substantial disparities in accuracy between the proposed dataset and other publicly available datasets. This finding highlights the complexity and richness of our dataset, emphasizing its value for advancing research in the field of air pollution.

----

## [0] On Convergence of Polynomial Approximations to the Gaussian Mixture Entropy

**Authors**: *Caleb Dahlke, Jason Pacheco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee860a9fa65a55a335754c557a5211de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee860a9fa65a55a335754c557a5211de-Abstract-Conference.html)

**Abstract**:

Gaussian mixture models (GMMs) are fundamental to machine learning due to their flexibility as approximating densities. However, uncertainty quantification of GMMs remains a challenge as differential entropy lacks a closed form.  This paper explores polynomial approximations, specifically Taylor and Legendre, to the GMM entropy from a theoretical and practical perspective. We provide new analysis of a widely used approach due to Huber et al.(2008) and show that the series diverges under simple conditions.  Motivated by this divergence we provide a novel Taylor series that is provably convergent to the true entropy of any GMM.  We demonstrate a method for selecting a center such that the series converges from below, providing a lower bound on GMM entropy. Furthermore, we demonstrate that orthogonal polynomial series result in more accurate polynomial approximations. Experimental validation supports our theoretical results while showing that our method is comparable in computation to Huber et al. We also show that in application, the use of these polynomial approximations, such as in Nonparametric Variational Inference by Gershamn et al. (2012), rely on the convergence of the methods in computing accurate approximations. This work contributes useful analysis to existing methods while introducing a novel approximation supported by firm theoretical guarantees.

----

## [0] CEIL: Generalized Contextual Imitation Learning

**Authors**: *Jinxin Liu, Li He, Yachen Kang, Zifeng Zhuang, Donglin Wang, Huazhe Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee90fb9511b263f2ff971be9b374f9ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee90fb9511b263f2ff971be9b374f9ee-Abstract-Conference.html)

**Abstract**:

In this paper, we present ContExtual Imitation Learning (CEIL), a general and broadly applicable algorithm for imitation learning (IL). Inspired by the formulation of hindsight information matching, we derive CEIL by explicitly learning a hindsight embedding function together with a contextual policy using the hindsight embeddings. To achieve the expert matching objective for IL, we advocate for optimizing a contextual variable such that it biases the contextual policy towards mimicking expert behaviors. Beyond the typical learning from demonstrations (LfD) setting, CEIL is a generalist that can be effectively applied to multiple settings including: 1) learning from observations (LfO), 2) offline IL, 3) cross-domain IL (mismatched experts), and 4) one-shot IL settings. Empirically, we evaluate CEIL on the popular MuJoCo tasks (online) and the D4RL dataset (offline). Compared to prior state-of-the-art baselines, we show that CEIL is more sample-efficient in most online IL tasks and achieves better or competitive performances in offline tasks.

----



[Go to the previous page](NIPS-2023-list3200.md)

[Go to the next page](NIPS-2023-list3202.md)

[Go to the catalog section](README.md)