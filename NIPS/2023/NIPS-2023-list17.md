## [3200] Collaborative Score Distillation for Consistent Visual Editing

        **Authors**: *Subin Kim, Kyungmin Lee, June Suk Choi, Jongheon Jeong, Kihyuk Sohn, Jinwoo Shin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7fd2c0a1a6f956c94024e955b34cc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7fd2c0a1a6f956c94024e955b34cc43-Abstract-Conference.html)

        **Abstract**:

        Generative priors of large-scale text-to-image diffusion models enable a wide range of new generation and editing applications on diverse visual modalities. However, when adapting these priors to complex visual modalities, often represented as multiple images (e.g., video or 3D scene), achieving consistency across a set of images is challenging. In this paper, we address this challenge with a novel method, Collaborative Score Distillation (CSD). CSD is based on the Stein Variational Gradient Descent (SVGD). Specifically, we propose to consider multiple samples as “particles” in the SVGD update and combine their score functions to distill generative priors over a set of images synchronously. Thus, CSD facilitates the seamless integration of information across 2D images, leading to a consistent visual synthesis across multiple samples. We show the effectiveness of CSD in a variety of editing tasks, encompassing the visual editing of panorama images, videos, and 3D scenes. Our results underline the competency of CSD as a versatile method for enhancing inter-sample consistency, thereby broadening the applicability of text-to-image diffusion models.

        ----

        ## [3201] FLuID: Mitigating Stragglers in Federated Learning using Invariant Dropout

        **Authors**: *Irene Wang, Prashant J. Nair, Divya Mahajan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html)

        **Abstract**:

        Federated Learning (FL) allows machine learning models to train locally on individual mobile devices, synchronizing model updates via a shared server. This approach safeguards user privacy; however, it also generates a heterogeneous training environment due to the varying performance capabilities across devices. As a result, “straggler” devices with lower performance often dictate the overalltraining time in FL. In this work, we aim to alleviate this performance bottleneck due to stragglers by dynamically balancing the training load across the system. We introduce Invariant Dropout, a method that extracts a sub-model based on the weight update threshold, thereby minimizing potential impacts on accuracy. Building on this dropout technique, we develop an adaptive training framework, Federated Learning using Invariant Dropout (FLuID). FLuID offers a lightweight sub-model extraction to regulate computational intensity, thereby reducing the load on straggler devices without affecting model quality. Our method leverages neuron updates from non-straggler devices to construct a tailored sub-model for each straggler based on client performance profiling. Furthermore, FLuID can dynamically adapt to changes in stragglers as runtime conditions shift. We evaluate FLuID using five real-world mobile clients. The evaluations show that Invariant Dropout maintains baseline model efficiency while alleviating the performance bottleneck of stragglers through a dynamic, runtime approach.

        ----

        ## [3202] Learning to Augment Distributions for Out-of-distribution Detection

        **Authors**: *Qizhou Wang, Zhen Fang, Yonggang Zhang, Feng Liu, Yixuan Li, Bo Han*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e812af67a942c21dd0104bd929f99da1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e812af67a942c21dd0104bd929f99da1-Abstract-Conference.html)

        **Abstract**:

        Open-world classification systems should discern out-of-distribution (OOD) data whose labels deviate from those of in-distribution (ID) cases, motivating recent studies in OOD detection. Advanced works, despite their promising progress, may still fail in the open world, owing to the lacking knowledge about unseen OOD data in advance. Although one can access auxiliary OOD data (distinct from unseen ones) for model training, it remains to analyze how such auxiliary data will work in the open world. To this end, we delve into such a problem from a learning theory perspective, finding that the distribution discrepancy between the auxiliary and the unseen real OOD data is the key to affect the open-world detection performance. Accordingly, we propose Distributional-Augmented OOD Learning (DAOL), alleviating the OOD distribution discrepancy by crafting an OOD distribution set that contains all distributions in a Wasserstein ball centered on the auxiliary OOD distribution. We justify that the predictor trained over the worst OOD data in the ball can shrink the OOD distribution discrepancy, thus improving the open-world detection performance given only the auxiliary OOD data. We conduct extensive evaluations across representative OOD detection setups, demonstrating the superiority of our DAOL over its advanced counterparts.

        ----

        ## [3203] Covariance-adaptive best arm identification

        **Authors**: *El Mehdi Saad, Gilles Blanchard, Nicolas Verzelen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e82ef7865f29b40640f486bbbe7959a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e82ef7865f29b40640f486bbbe7959a7-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of best arm identification in the multi-armed bandit model, under fixed confidence. Given a confidence input $\delta$, the goal is to identify the arm with the highest mean reward with a probability of at least $1 - \delta$, while minimizing the number of arm pulls. While the literature provides solutions to this problem under the assumption of independent arms distributions, we propose a more flexible scenario where arms can be dependent and rewards can be sampled simultaneously. This framework allows the learner to estimate the covariance among the arms distributions, enabling a more efficient identification of the best arm. The relaxed setting we propose is relevant in various applications, such as clinical trials, where similarities between patients or drugs suggest underlying correlations in the outcomes. We introduce new algorithms that adapt to the unknown covariance of the arms and demonstrate through theoretical guarantees that substantial improvement can be achieved over the standard setting. Additionally, we provide new lower bounds for the relaxed setting and present numerical simulations that support their theoretical findings.

        ----

        ## [3204] What a MESS: Multi-Domain Evaluation of Zero-Shot Semantic Segmentation

        **Authors**: *Benedikt Blumenstiel, Johannes Jakubik, Hilde Kühne, Michael Vössing*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e82f45e480f5f44d696ba15dad88f9a3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e82f45e480f5f44d696ba15dad88f9a3-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        While semantic segmentation has seen tremendous improvements in the past, there are still significant labeling efforts necessary and the problem of limited generalization to classes that have not been present during training. To address this problem, zero-shot semantic segmentation makes use of large self-supervised vision-language models, allowing zero-shot transfer to unseen classes. In this work, we build a benchmark for Multi-domain Evaluation of Zero-Shot Semantic Segmentation (MESS), which allows a holistic analysis of performance across a wide range of domain-specific datasets such as medicine, engineering, earth monitoring, biology, and agriculture. To do this, we reviewed 120 datasets, developed a taxonomy, and classified the datasets according to the developed taxonomy. We select a representative subset consisting of 22 datasets and propose it as the MESS benchmark. We evaluate eight recently published models on the proposed MESS benchmark and analyze characteristics for the performance of zero-shot transfer models. The toolkit is available at https://github.com/blumenstiel/MESS.

        ----

        ## [3205] Swarm Reinforcement Learning for Adaptive Mesh Refinement

        **Authors**: *Niklas Freymuth, Philipp Dahlinger, Tobias Würth, Simon Reisch, Luise Kärger, Gerhard Neumann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html)

        **Abstract**:

        The Finite Element Method, an important technique in engineering, is aided by Adaptive Mesh Refinement (AMR), which dynamically refines mesh regions to allow for a favorable trade-off between computational speed and simulation accuracy. Classical methods for AMR depend on task-specific heuristics or expensive error estimators, hindering their use for complex simulations. Recent learned AMR methods tackle these problems, but so far scale only to simple toy examples. We formulate AMR as a novel Adaptive Swarm Markov Decision Process in which a mesh is modeled as a system of simple collaborating agents that may split into multiple new agents. This framework allows for a spatial reward formulation that simplifies the credit assignment problem, which we combine with Message Passing Networks to propagate information between neighboring mesh elements. We experimentally validate the effectiveness of our approach, Adaptive Swarm Mesh Refinement (ASMR), showing that it learns reliable, scalable, and efficient refinement strategies on a set of challenging problems. Our approach significantly speeds up computation, achieving up to 30-fold improvement compared to uniform refinements in complex simulations. Additionally, we outperform learned baselines and achieve a refinement quality that is on par with a traditional error-based AMR strategy without expensive oracle information about the error signal.

        ----

        ## [3206] Fast Projected Newton-like Method for Precision Matrix Estimation under Total Positivity

        **Authors**: *Jianfeng Cai, José Vinícius de Miranda Cardoso, Daniel P. Palomar, Jiaxi Ying*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e878c8f38381d0964677fb9536c494ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e878c8f38381d0964677fb9536c494ee-Abstract-Conference.html)

        **Abstract**:

        We study the problem of estimating precision matrices in Gaussian distributions that are multivariate totally positive of order two ($\mathrm{MTP}_2$). The precision matrix in such a distribution is an M-matrix. This problem can be formulated as a sign-constrained log-determinant program. Current algorithms are designed using the block coordinate descent method or the proximal point algorithm, which becomes computationally challenging in high-dimensional cases due to the requirement to solve numerous nonnegative quadratic programs or large-scale linear systems. To address this issue, we propose a novel algorithm based on the two-metric projection method, incorporating a carefully designed search direction and variable partitioning scheme. Our algorithm substantially reduces computational complexity, and its theoretical convergence is established. Experimental results on synthetic and real-world datasets demonstrate that our proposed algorithm provides a significant improvement in computational efficiency compared to the state-of-the-art methods.

        ----

        ## [3207] BanditPAM++: Faster k-medoids Clustering

        **Authors**: *Mo Tiwari, Ryan Kang, Donghyun Lee, Sebastian Thrun, Ilan Shomorony, Martin J. Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e885e5bc6e13b9dd8f80bc5482b1fa2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e885e5bc6e13b9dd8f80bc5482b1fa2f-Abstract-Conference.html)

        **Abstract**:

        Clustering is a fundamental task in data science with wide-ranging applications. In $k$-medoids clustering, cluster centers must be actual datapoints and arbitrary distance metrics may be used; these features allow for greater interpretability of the cluster centers and the clustering of exotic objects in $k$-medoids clustering, respectively. $k$-medoids clustering has recently grown in popularity due to the discovery of more efficient $k$-medoids algorithms. In particular, recent research has proposed BanditPAM, a randomized $k$-medoids algorithm with state-of-the-art complexity and clustering accuracy. In this paper, we present BanditPAM++, which accelerates BanditPAM via two algorithmic improvements, and is $O(k)$ faster than BanditPAM in complexity and substantially faster than BanditPAM in wall-clock runtime. First, we demonstrate that BanditPAM has a special structure that allows the reuse of clustering information $\textit{within}$ each iteration. Second, we demonstrate that BanditPAM has additional structure that permits the reuse of information $\textit{across}$ different iterations. These observations inspire our proposed algorithm, BanditPAM++, which returns the same clustering solutions as BanditPAM but often several times faster. For example, on the CIFAR10 dataset, BanditPAM++ returns the same results as BanditPAM but runs over 10$\times$ faster. Finally, we provide a high-performance C++ implementation of BanditPAM++, callable from Python and R, that may be of interest to practitioners at https://github.com/motiwari/BanditPAM. Auxiliary code to reproduce all of our experiments via a one-line script is available at https://github.com/ThrunGroup/BanditPAM_plusplus_experiments.

        ----

        ## [3208] Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks

        **Authors**: *Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, Jordan Terry*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8916198466e8ef218a2185a491b49fa-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8916198466e8ef218a2185a491b49fa-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present the Minigrid and Miniworld libraries which provide a suite of goal-oriented 2D and 3D environments. The libraries were explicitly created with a minimalistic design paradigm to allow users to rapidly develop new environments for a wide range of research-specific needs. As a result, both have received widescale adoption by the RL community, facilitating research in a wide range of areas. In this paper, we outline the design philosophy, environment details, and their world generation API.  We also showcase the additional capabilities brought by the unified API between Minigrid and Miniworld through case studies on transfer learning (for both RL agents and humans) between the different observation spaces. The source code of Minigrid and Miniworld can be found at https://github.com/Farama-Foundation/Minigrid and https://github.com/Farama-Foundation/Miniworld along with their documentation at https://minigrid.farama.org/ and https://miniworld.farama.org/.

        ----

        ## [3209] Cross-Domain Policy Adaptation via Value-Guided Data Filtering

        **Authors**: *Kang Xu, Chenjia Bai, Xiaoteng Ma, Dong Wang, Bin Zhao, Zhen Wang, Xuelong Li, Wei Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ad87f1076fb0f75d89a45828f186b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ad87f1076fb0f75d89a45828f186b0-Abstract-Conference.html)

        **Abstract**:

        Generalizing policies across different domains with dynamics mismatch poses a significant challenge in reinforcement learning. For example, a robot learns the policy in a simulator, but when it is deployed in the real world, the dynamics of the environment may be different. Given the source and target domain with dynamics mismatch, we consider the online dynamics adaptation problem, in which case the agent can access sufficient source domain data while online interactions with the target domain are limited. Existing research has attempted to solve the problem from the dynamics discrepancy perspective. In this work, we reveal the limitations of these methods and explore the problem from the value difference perspective via a novel insight on the value consistency across domains. Specifically, we present the Value-Guided Data Filtering (VGDF) algorithm, which selectively shares transitions from the source domain based on the proximity of paired value targets across the two domains. Empirical results on various environments with kinematic and morphology shifts demonstrate that our method achieves superior performance compared to prior approaches.

        ----

        ## [3210] Connecting Certified and Adversarial Training

        **Authors**: *Yuhao Mao, Mark Niklas Müller, Marc Fischer, Martin T. Vechev*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8b0c97b34fdaf58b2f48f8cca85e76a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8b0c97b34fdaf58b2f48f8cca85e76a-Abstract-Conference.html)

        **Abstract**:

        Training certifiably robust neural networks remains a notoriously hard problem.While adversarial training optimizes under-approximations of the worst-case loss, which leads to insufficient regularization for certification, sound certified training methods, optimize loose over-approximations, leading to over-regularization and poor (standard) accuracy.In this work, we propose TAPS, an (unsound) certified training method that combines IBP and PGD training to optimize more precise, although not necessarily sound, worst-case loss approximations, reducing over-regularization and increasing certified and standard accuracies.Empirically, TAPS achieves a new state-of-the-art in many settings, e.g., reaching a certified accuracy of $22$% on TinyImageNet for $\ell_\infty$-perturbations with radius $\epsilon=1/255$. We make our implementation and networks public at https://github.com/eth-sri/taps.

        ----

        ## [3211] Effectively Learning Initiation Sets in Hierarchical Reinforcement Learning

        **Authors**: *Akhil Bagaria, Ben Abbatematteo, Omer Gottesman, Matt Corsaro, Sreehari Rammohan, George Dimitri Konidaris*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html)

        **Abstract**:

        An agent learning an option in hierarchical reinforcement learning must solve three problems: identify the option's subgoal (termination condition), learn a policy, and learn where that policy will succeed (initiation set). The termination condition is typically identified first, but the option policy and initiation set must be learned simultaneously, which is challenging because the initiation set depends on the option policy, which changes as the agent learns. Consequently, data obtained from option execution becomes invalid over time, leading to an inaccurate initiation set that subsequently harms downstream task performance. We highlight three issues---data non-stationarity, temporal credit assignment, and pessimism---specific to learning initiation sets, and propose to address them using tools from off-policy value estimation and classification. We show that our method learns higher-quality initiation sets faster than existing methods (in MiniGrid and Montezuma's Revenge), can automatically discover promising grasps for robot manipulation (in Robosuite), and improves the performance of a state-of-the-art option discovery method in a challenging maze navigation task in MuJoCo.

        ----

        ## [3212] Alignment with human representations supports robust few-shot learning

        **Authors**: *Ilia Sucholutsky, Tom Griffiths*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ddc03b001d4c4b44b29bc1167e7fdd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ddc03b001d4c4b44b29bc1167e7fdd-Abstract-Conference.html)

        **Abstract**:

        Should we care whether AI systems have representations of the world that are similar to those of humans? We provide an information-theoretic analysis that suggests that there should be a U-shaped relationship between the degree of representational alignment with humans and performance on few-shot learning tasks. We confirm this prediction empirically, finding such a relationship in an analysis of the performance of 491 computer vision models. We also show that highly-aligned models are more robust to both natural adversarial attacks and domain shifts. Our results suggest that human-alignment is often a sufficient, but not necessary, condition for models to make effective use of limited data, be robust, and generalize well.

        ----

        ## [3213] ReMaX: Relaxing for Better Training on Efficient Panoptic Segmentation

        **Authors**: *Shuyang Sun, Weijun Wang, Andrew G. Howard, Qihang Yu, Philip H. S. Torr, Liang-Chieh Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8e30fda5ab87ea93360a36288ac0145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8e30fda5ab87ea93360a36288ac0145-Abstract-Conference.html)

        **Abstract**:

        This paper presents a new mechanism to facilitate the training of mask transformers for efficient panoptic segmentation, democratizing its deployment. We observe that due to the high complexity in the training objective of panoptic segmentation, it will inevitably lead to much higher penalization on false positive. Such unbalanced loss makes the training process of the end-to-end mask-transformer based architectures difficult, especially for efficient models. In this paper, we present ReMaX that adds relaxation to mask predictions and class predictions during the training phase for panoptic segmentation. We demonstrate that via these simple relaxation techniques during training, our model can be consistently improved by a clear margin without any extra computational cost on inference. By combining our method with efficient backbones like MobileNetV3-Small, our method achieves new state-of-the-art results for efficient panoptic segmentation on COCO, ADE20K and Cityscapes. Code and pre-trained checkpoints will be available at https://github.com/google-research/deeplab2.

        ----

        ## [3214] The Behavior and Convergence of Local Bayesian Optimization

        **Authors**: *Kaiwen Wu, Kyurae Kim, Roman Garnett, Jacob Gardner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8f4eae0a41cab67fdead3aa6b77f083-Abstract-Conference.html)

        **Abstract**:

        A recent development in Bayesian optimization is the use of local optimization strategies, which can deliver strong empirical performance on high-dimensional problems compared to traditional global strategies. The "folk wisdom" in the literature is that the focus on local optimization sidesteps the curse of dimensionality; however, little is known concretely about the expected behavior or convergence of Bayesian local optimization routines. We first study the behavior of the local approach, and find that the statistics of individual local solutions of Gaussian process sample paths are surprisingly good compared to what we would expect to recover from global methods. We then present the first rigorous analysis of such a Bayesian local optimization algorithm recently proposed by MÃ¼ller et al. (2021), and derive convergence rates in both the noisy and noiseless settings.

        ----

        ## [3215] Contrastive Sampling Chains in Diffusion Models

        **Authors**: *Junyu Zhang, Daochang Liu, Shichao Zhang, Chang Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e8ff788779f2e9e74ccd0d6b84607437-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e8ff788779f2e9e74ccd0d6b84607437-Abstract-Conference.html)

        **Abstract**:

        The past few years have witnessed great success in the use of diffusion models (DMs) to generate high-fidelity images with the help of stochastic differential equations (SDEs). However, discretization error is an inevitable limitation when utilizing numerical solvers to solve SDEs. To address this limitation, we provide a theoretical analysis demonstrating that an appropriate combination of the contrastive loss and score matching serves as an upper bound of the KL divergence between the true data distribution and the model distribution. To obtain this bound, we utilize a contrastive loss to construct a contrastive sampling chain to fine-tuning the pre-trained DM. In this manner, our method reduces the discretization error and thus yields a smaller gap between the true data distribution and our model distribution. Moreover, the presented method can be applied to fine-tuning various pre-trained DMs, both with or without fast sampling algorithms, contributing to better sample quality or slightly faster sampling speeds. To validate the efficacy of our method, we conduct comprehensive experiments. For example, on CIFAR10, when applied to a pre-trained EDM, our method improves the FID from 2.04 to 1.88 with 35 neural function evaluations (NFEs), and reduces NFEs from 35 to 25 to achieve the same 2.04 FID.

        ----

        ## [3216] Effective Robustness against Natural Distribution Shifts for Models with Different Training Data

        **Authors**: *Zhouxing Shi, Nicholas Carlini, Ananth Balashankar, Ludwig Schmidt, Cho-Jui Hsieh, Alex Beutel, Yao Qin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9000ecb86d45c442a1d38fae68dd8fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9000ecb86d45c442a1d38fae68dd8fb-Abstract-Conference.html)

        **Abstract**:

        ``Effective robustness'' measures the extra out-of-distribution (OOD) robustness beyond what can be predicted from the in-distribution (ID) performance. Existing effective robustness evaluations typically use a single test set such as ImageNet to evaluate the ID accuracy. This becomes problematic when evaluating models trained on different data distributions, e.g., comparing models trained on ImageNet vs. zero-shot language-image pre-trained models trained on LAION. In this paper, we propose a new evaluation metric to evaluate and compare the effective robustness of models trained on different data. To do this, we control for the accuracy on multiple ID test sets that cover the training distributions for all the evaluated models. Our new evaluation metric provides a better estimate of effective robustness when there are models with different training data. It may also explain the surprising effective robustness gains of zero-shot CLIP-like models exhibited in prior works that used ImageNet as the only ID test set, while the gains diminish under our new evaluation. Additional artifacts including interactive visualizations are provided at https://shizhouxing.github.io/effective-robustness.

        ----

        ## [3217] Tailoring Self-Attention for Graph via Rooted Subtrees

        **Authors**: *Siyuan Huang, Yunchong Song, Jiayue Zhou, Zhouhan Lin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e90ba1fc564a69809d7391bf76a5f087-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e90ba1fc564a69809d7391bf76a5f087-Abstract-Conference.html)

        **Abstract**:

        Attention mechanisms have made significant strides in graph learning, yet they still exhibit notable limitations: local attention faces challenges in capturing long-range information due to the inherent problems of the message-passing scheme, while global attention cannot reflect the hierarchical neighborhood structure and fails to capture fine-grained local information. In this paper, we propose a novel multi-hop graph attention mechanism, named Subtree Attention (STA), to address the aforementioned issues. STA seamlessly bridges the fully-attentional structure and the rooted subtree, with theoretical proof that STA approximates the global attention under extreme settings. By allowing direct computation of attention weights among multi-hop neighbors, STA mitigates the inherent problems in existing graph attention mechanisms. Further we devise an efficient form for STA by employing kernelized softmax, which yields a linear time complexity. Our resulting GNN architecture, the STAGNN, presents a simple yet performant STA-based graph neural network leveraging a hop-aware attention strategy. Comprehensive evaluations on ten node classification datasets demonstrate that STA-based models outperform existing graph transformers and mainstream GNNs. The codeis available at https://github.com/LUMIA-Group/SubTree-Attention.

        ----

        ## [3218] Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective

        **Authors**: *Zeyuan Yin, Eric P. Xing, Zhiqiang Shen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html)

        **Abstract**:

        We present a new dataset condensation framework termed Squeeze, Recover and Relabel (SRe$^2$L) that decouples the bilevel optimization of model and synthetic data during training, to handle varying scales of datasets, model architectures and image resolutions for efficient dataset condensation. The proposed method demonstrates flexibility across diverse dataset scales and exhibits multiple advantages in terms of arbitrary resolutions of synthesized images, low training cost and memory consumption with high-resolution synthesis, and the ability to scale up to arbitrary evaluation network architectures. Extensive experiments are conducted on Tiny-ImageNet and full ImageNet-1K datasets. Under 50 IPC, our approach achieves the highest 42.5\% and 60.8\% validation accuracy on Tiny-ImageNet and ImageNet-1K, outperforming all previous state-of-the-art methods by margins of 14.5\% and 32.9\%, respectively. Our approach also surpasses MTT in terms of speed by approximately 52$\times$ (ConvNet-4) and 16$\times$ (ResNet-18) faster with less memory consumption of 11.6$\times$ and 6.4$\times$ during data synthesis. Our code and condensed datasets of 50, 200 IPC with 4K recovery budget are available at https://github.com/VILA-Lab/SRe2L.

        ----

        ## [3219] Disentangled Wasserstein Autoencoder for T-Cell Receptor Engineering

        **Authors**: *Tianxiao Li, Hongyu Guo, Filippo Grazioli, Mark Gerstein, Martin Renqiang Min*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html)

        **Abstract**:

        In protein biophysics, the separation between the functionally important residues (forming the active site or binding surface) and those that create the overall structure (the fold) is a well-established and fundamental concept. Identifying and modifying those functional sites is critical for protein engineering but computationally non-trivial, and requires significant domain knowledge. To automate this process from a data-driven perspective, we propose a disentangled Wasserstein autoencoder with an auxiliary classifier, which isolates the function-related patterns from the rest with theoretical guarantees. This enables one-pass protein sequence editing and improves the understanding of the resulting sequences and editing actions involved. To demonstrate its effectiveness, we apply it to T-cell receptors (TCRs), a well-studied structure-function case. We show that our method can be used to alter the function of TCRs without changing the structural backbone, outperforming several competing methods in generation quality and efficiency, and requiring only 10\% of the running time needed by baseline models. To our knowledge, this is the first approach that utilizes disentangled representations for TCR engineering.

        ----

        ## [3220] Modality-Independent Teachers Meet Weakly-Supervised Audio-Visual Event Parser

        **Authors**: *Yung-Hsuan Lai, Yen-Chun Chen, Frank Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95e9f0c127aa1cfa2628adb2f3cb107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95e9f0c127aa1cfa2628adb2f3cb107-Abstract-Conference.html)

        **Abstract**:

        Audio-visual learning has been a major pillar of multi-modal machine learning, where the community mostly focused on its $\textit{modality-aligned}$ setting, $\textit{i.e.}$, the audio and visual modality are $\textit{both}$ assumed to signal the prediction target.With the Look, Listen, and Parse dataset (LLP), we investigate the under-explored $\textit{unaligned}$ setting, where the goal is to recognize audio and visual events in a video with only weak labels observed.Such weak video-level labels only tell what events happen without knowing the modality they are perceived (audio, visual, or both).To enhance learning in this challenging setting, we incorporate large-scale contrastively pre-trained models as the modality teachers. A simple, effective, and generic method, termed $\textbf{V}$isual-$\textbf{A}$udio $\textbf{L}$abel Elab$\textbf{or}$ation (VALOR), is innovated to harvest modality labels for the training events.Empirical studies show that the harvested labels significantly improve an attentional baseline by $\textbf{8.0}$ in average F-score (Type@AV).Surprisingly, we found that modality-independent teachers outperform their modality-fused counterparts since they are noise-proof from the other potentially unaligned modality.Moreover, our best model achieves the new state-of-the-art on all metrics of LLP by a substantial margin ($\textbf{+5.4}$ F-score for Type@AV). VALOR is further generalized to Audio-Visual Event Localization and achieves the new state-of-the-art as well.

        ----

        ## [3221] Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation

        **Authors**: *Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e95eb5206c867be843fbc14bbfe8c10e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e95eb5206c867be843fbc14bbfe8c10e-Abstract-Conference.html)

        **Abstract**:

        This paper studies the problem of weakly open-vocabulary semantic segmentation (WOVSS), which learns to segment objects of arbitrary classes using mere image-text pairs. Existing works turn to enhance the vanilla vision transformer by introducing explicit grouping recognition, i.e., employing several group tokens/centroids to cluster the image tokens and perform the group-text alignment. Nevertheless, these methods suffer from a granularity inconsistency regarding the usage of group tokens, which are aligned in the all-to-one v.s. one-to-one manners during the training and inference phases, respectively. We argue that this discrepancy arises from the lack of elaborate supervision for each group token. To bridge this granularity gap, this paper explores explicit supervision for the group tokens from the prototypical knowledge. To this end, this paper proposes the non-learnable prototypical regularization (NPR) where non-learnable prototypes are estimated from source features to serve as supervision and enable contrastive matching of the group tokens. This regularization encourages the group tokens to segment objects with less redundancy and capture more comprehensive semantic regions, leading to increased compactness and richness. Based on NPR, we propose the prototypical guidance segmentation network (PGSeg) that incorporates multi-modal regularization by leveraging prototypical sources from both images and texts at different levels, progressively enhancing the segmentation capability with diverse prototypical patterns. Experimental results show that our proposed method achieves state-of-the-art performance on several benchmark datasets.

        ----

        ## [3222] A Theoretical Analysis of Optimistic Proximal Policy Optimization in Linear Markov Decision Processes

        **Authors**: *Han Zhong, Tong Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9721921b799b6ea98d37f9e77f1a7fe-Abstract-Conference.html)

        **Abstract**:

        The proximal policy optimization (PPO) algorithm stands as one of the most prosperous methods in the field of reinforcement learning (RL). Despite its success, the theoretical understanding of PPO remains deficient. Specifically, it is unclear whether PPO or its optimistic variants can effectively solve linear Markov decision processes (MDPs), which are arguably the simplest models in RL with function approximation.     To bridge this gap, we propose an optimistic variant of PPO for episodic adversarial linear MDPs with full-information feedback, and establish a $\tilde{\mathcal{O}}(d^{3/4}H^2K^{3/4})$ regret for it. Here $d$ is the ambient dimension of linear MDPs, $H$ is the length of each episode, and $K$ is the number of episodes. Compared with existing policy-based algorithms, we achieve the state-of-the-art regret bound in both stochastic linear MDPs and adversarial linear MDPs with full information. Additionally, our algorithm design features a novel multi-batched updating mechanism and the theoretical analysis utilizes a new covering number argument of value and policy classes, which might be of independent interest.

        ----

        ## [3223] Uncertainty-Aware Instance Reweighting for Off-Policy Learning

        **Authors**: *Xiaoying Zhang, Junpu Chen, Hongning Wang, Hong Xie, Yang Liu, John C. S. Lui, Hang Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e97ac22927560eb2de6b658498cbc575-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e97ac22927560eb2de6b658498cbc575-Abstract-Conference.html)

        **Abstract**:

        Off-policy learning, referring to the procedure of policy optimization with access only to logged feedback data, has shown importance in various important real-world applications, such as search engines and recommender systems. While the ground-truth logging policy is usually unknown, previous work simply takes its estimated value for the off-policy learning, ignoring the negative impact from both high bias and high variance resulted from such an estimator. And these impact is often magnified on samples with small and inaccurately estimated logging probabilities. The contribution of this work is to explicitly model the uncertainty in the estimated logging policy, and propose an Uncertainty-aware Inverse Propensity Score estimator (UIPS) for improved off-policy learning, with a theoretical convergence guarantee. Experiment results on the synthetic and real-world recommendation datasets demonstrate that UIPS significantly improves the quality of the discovered policy, when compared against an extensive list of state-of-the-art baselines.

        ----

        ## [3224] Model-free Posterior Sampling via Learning Rate Randomization

        **Authors**: *Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Rémi Munos, Alexey Naumov, Pierre Perrault, Michal Valko, Pierre Ménard*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e985dfca10e1167c0836a70880ef0858-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e985dfca10e1167c0836a70880ef0858-Abstract-Conference.html)

        **Abstract**:

        In this paper, we introduce Randomized Q-learning (RandQL), a novel randomized model-free algorithm for regret minimization in episodic Markov Decision Processes (MDPs). To the best of our knowledge, RandQL is the first tractable model-free posterior sampling-based algorithm. We analyze the performance of RandQL in both tabular and non-tabular metric space settings. In tabular MDPs, RandQL achieves a regret bound of order $\widetilde{\mathcal{O}}(\sqrt{H^{5}SAT})$, where $H$ is the planning horizon, $S$ is the number of states, $A$ is the number of actions, and $T$ is the number of episodes. For a metric state-action space, RandQL enjoys a regret bound of order $\widetilde{\mathcal{O}}(H^{5/2} T^{(d_z+1)/(d_z+2)})$, where $d_z$ denotes the zooming dimension. Notably, RandQL achieves optimistic exploration without using bonuses, relying instead on a novel idea of learning rate randomization. Our empirical study shows that RandQL outperforms existing approaches on baseline exploration environments.

        ----

        ## [3225] TriRE: A Multi-Mechanism Learning Paradigm for Continual Knowledge Retention and Promotion

        **Authors**: *Preetha Vijayan, Prashant Shivaram Bhat, Bahram Zonooz, Elahe Arani*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html)

        **Abstract**:

        Continual learning (CL) has remained a persistent challenge for deep neural networks due to catastrophic forgetting (CF) of previously learned tasks. Several techniques such as weight regularization, experience rehearsal, and parameter isolation have been proposed to alleviate CF. Despite their relative success, these research directions have predominantly remained orthogonal and suffer from several shortcomings, while missing out on the advantages of competing strategies. On the contrary, the brain continually learns, accommodates, and transfers knowledge across tasks by simultaneously leveraging several neurophysiological processes, including neurogenesis, active forgetting, neuromodulation, metaplasticity, experience rehearsal, and context-dependent gating, rarely resulting in CF. Inspired by how the brain exploits multiple mechanisms concurrently, we propose TriRE, a novel CL paradigm that encompasses retaining the most prominent neurons for each task, revising and solidifying the extracted knowledge of current and past tasks, and actively promoting less active neurons for subsequent tasks through rewinding and relearning. Across CL settings, TriRE significantly reduces task interference and surpasses different CL approaches considered in isolation.

        ----

        ## [3226] Implicit Variational Inference for High-Dimensional Posteriors

        **Authors**: *Anshuk Uppal, Kristoffer Stensbo-Smidt, Wouter Boomsma, Jes Frellsen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e99be8b1f637996eaf1154f2f4cb6f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e99be8b1f637996eaf1154f2f4cb6f49-Abstract-Conference.html)

        **Abstract**:

        In variational inference, the benefits of Bayesian models rely on accurately capturing the true posterior distribution. We propose using neural samplers that specify implicit distributions, which are well-suited for approximating complex multimodal and correlated posteriors in high-dimensional spaces. Our approach introduces novel bounds for approximate inference using implicit distributions by locally linearising the neural sampler. This is distinct from existing methods that rely on additional discriminator networks and unstable adversarial objectives. Furthermore, we present a new sampler architecture that, for the first time, enables implicit distributions over tens of millions of latent variables, addressing computational concerns by using differentiable numerical approximations. We empirically show that our method is capable of recovering correlations across layers in large Bayesian neural networks, a property that is crucial for a network's performance but notoriously challenging to achieve. To the best of our knowledge, no other method has been shown to accomplish this task for such large models. Through experiments in downstream tasks, we demonstrate that our expressive posteriors outperform state-of-the-art uncertainty quantification methods, validating the effectiveness of our training algorithm and the quality of the learned implicit approximation.

        ----

        ## [3227] k-Median Clustering via Metric Embedding: Towards Better Initialization with Differential Privacy

        **Authors**: *Chenglin Fan, Ping Li, Xiaoyun Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9a612969b4df241ff0d8273656bd5a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9a612969b4df241ff0d8273656bd5a4-Abstract-Conference.html)

        **Abstract**:

        In clustering algorithms, the choice of initial centers is crucial for the quality of the learned clusters. We propose a new initialization scheme for the $k$-median problem in the general metric space (e.g., discrete space induced by graphs), based on the construction of metric embedding tree structure of the data. We propose a novel and efficient search algorithm, for good initial centers that can be used subsequently for the local search algorithm. The so-called HST initialization method can produce initial centers achieving lower error than those from another popular method $k$-median++, also with higher efficiency when $k$ is not too small. Our HST initialization can also be easily extended to the setting of differential privacy (DP) to generate private initial centers. We show that the error of applying DP local search followed by our private HST initialization improves previous results on the approximation error, and approaches the lower bound within a small factor. Experiments demonstrate the effectiveness of our proposed methods.

        ----

        ## [3228] Towards a Unified Analysis of Kernel-based Methods Under Covariate Shift

        **Authors**: *Xingdong Feng, Xin He, Caixing Wang, Chao Wang, Jingnan Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9b0ae84d6879b30c78cb8537466a4e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9b0ae84d6879b30c78cb8537466a4e0-Abstract-Conference.html)

        **Abstract**:

        Covariate shift occurs prevalently in practice, where the input distributions of the source and target data are substantially different. Despite its practical importance in various learning problems, most of the existing methods only focus on some specific learning tasks and are not well validated theoretically and numerically. To tackle this problem, we propose a unified analysis of general nonparametric methods in a reproducing kernel Hilbert space (RKHS) under covariate shift.  Our theoretical results are established for a general loss belonging to a rich loss function family, which includes many commonly used methods as special cases, such as mean regression, quantile regression, likelihood-based classification, and margin-based classification. Two types of covariate shift problems are the focus of this paper and the sharp convergence rates are established for a general loss function to provide a unified theoretical analysis, which concurs with the optimal results in literature where the squared loss is used. Extensive numerical studies on synthetic and real examples confirm our theoretical findings and further illustrate the effectiveness of our proposed method.

        ----

        ## [3229] Learning Functional Transduction

        **Authors**: *Mathieu Chalvidal, Thomas Serre, Rufin VanRullen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9b8a3362a6d9a7f9f842bd2d919e1a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9b8a3362a6d9a7f9f842bd2d919e1a0-Abstract-Conference.html)

        **Abstract**:

        Research in statistical learning has polarized into two general approaches to perform regression analysis: Transductive methods construct estimates directly based on exemplar data using generic relational principles which might suffer from the curse of dimensionality. Conversely, inductive methods can potentially fit highly complex functions at the cost of compute-intensive solution searches. In this work, we leverage the theory of vector-valued Reproducing Kernel Banach Spaces (RKBS) to propose a hybrid approach: We show that transductive regression systems can be meta-learned with gradient descent to form efficient in-context neural approximators of function defined over both finite and infinite-dimensional spaces (operator regression). Once trained, our Transducer can almost instantaneously capture new functional relationships and produce original image estimates, given a few pairs of input and output examples. We demonstrate the benefit of our meta-learned transductive approach to model physical systems influenced by varying external factors with little data at a fraction of the usual deep learning training costs for partial differential equations and climate modeling applications.

        ----

        ## [3230] Gaussian Membership Inference Privacy

        **Authors**: *Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9df36b21ff4ee211a8b71ee8b7e9f57-Abstract-Conference.html)

        **Abstract**:

        We propose a novel and practical privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model.  Consequently, $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). In particular, we derive a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP) by theoretically analyzing likelihood ratio-based membership inference attacks on stochastic gradient descent (SGD). Our analysis highlights that models trained with standard SGD already offer an elementary level of MIP.  Additionally, we show how $f$-MIP can be amplified by adding noise to gradient updates. Our analysis further yields an analytical membership inference attack that offers two distinct advantages over previous approaches. First, unlike existing state-of-the-art attacks that require training hundreds of shadow models, our attack does not require any shadow model.  Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, we quantify how various hyperparameters (e.g., batch size, number of model parameters) and specific data characteristics determine an attacker's ability to accurately infer a point's membership in the training set. We demonstrate the effectiveness of our method on models trained on vision and tabular datasets.

        ----

        ## [3231] Modality-Agnostic Self-Supervised Learning with Meta-Learned Masked Auto-Encoder

        **Authors**: *Huiwon Jang, Jihoon Tack, Daewon Choi, Jongheon Jeong, Jinwoo Shin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9df55bf67e499635908395931ed6ea9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9df55bf67e499635908395931ed6ea9-Abstract-Conference.html)

        **Abstract**:

        Despite its practical importance across a wide range of modalities, recent advances in self-supervised learning (SSL) have been primarily focused on a few well-curated domains, e.g., vision and language, often relying on their domain-specific knowledge. For example, Masked Auto-Encoder (MAE) has become one of the popular architectures in these domains, but less has explored its potential in other modalities. In this paper, we develop MAE as a unified, modality-agnostic SSL framework. In turn, we argue meta-learning as a key to interpreting MAE as a modality-agnostic learner, and propose enhancements to MAE from the motivation to jointly improve its SSL across diverse modalities, coined MetaMAE as a result. Our key idea is to view the mask reconstruction of MAE as a meta-learning task: masked tokens are predicted by adapting the Transformer meta-learner through the amortization of unmasked tokens. Based on this novel interpretation, we propose to integrate two advanced meta-learning techniques. First, we adapt the amortized latent of the Transformer encoder using gradient-based meta-learning to enhance the reconstruction. Then, we maximize the alignment between amortized and adapted latents through task contrastive learning which guides the Transformer encoder to better encode the task-specific knowledge. Our experiment demonstrates the superiority of MetaMAE in the modality-agnostic SSL benchmark (called DABS), significantly outperforming prior baselines.

        ----

        ## [3232] Dual Self-Awareness Value Decomposition Framework without Individual Global Max for Cooperative MARL

        **Authors**: *Zhiwei Xu, Bin Zhang, Dapeng Li, Guangchong Zhou, Zeren Zhang, Guoliang Fan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e9e140df6de01afb672cb859d203c307-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e9e140df6de01afb672cb859d203c307-Abstract-Conference.html)

        **Abstract**:

        Value decomposition methods have gained popularity in the field of cooperative multi-agent reinforcement learning. However, almost all existing methods follow the principle of Individual Global Max (IGM) or its variants, which limits their problem-solving capabilities. To address this, we propose a dual self-awareness value decomposition framework, inspired by the notion of dual self-awareness in psychology, that entirely rejects the IGM premise. Each agent consists of an ego policy for action selection and an alter ego value function to solve the credit assignment problem. The value function factorization can ignore the IGM assumption by utilizing an explicit search procedure. On the basis of the above, we also suggest a novel anti-ego exploration mechanism to avoid the algorithm becoming stuck in a local optimum. As the first fully IGM-free value decomposition method, our proposed framework achieves desirable performance in various cooperative tasks.

        ----

        ## [3233] DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification

        **Authors**: *Mintong Kang, Dawn Song, Bo Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea0b28cbbd0cbc45ec4ac38e92da9cb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea0b28cbbd0cbc45ec4ac38e92da9cb2-Abstract-Conference.html)

        **Abstract**:

        Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20\% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10\% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.

        ----

        ## [3234] Squared Neural Families: A New Class of Tractable Density Models

        **Authors**: *Russell Tsuchida, Cheng Soon Ong, Dino Sejdinovic*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea13534ee239bb3977795b8cc855bacc-Abstract-Conference.html)

        **Abstract**:

        Flexible models for probability distributions are an essential ingredient in many machine learning tasks. We develop and investigate a new class of probability distributions, which we call a Squared Neural Family (SNEFY), formed by squaring the 2-norm of a neural network and normalising it with respect to a base measure. Following the reasoning similar to the well established connections between infinitely wide neural networks and Gaussian processes, we show that SNEFYs admit closed form normalising constants in many cases of interest, thereby resulting in flexible yet fully tractable density models. SNEFYs strictly generalise classical exponential families, are closed under conditioning, and have tractable marginal distributions. Their utility is illustrated on a variety of density estimation, conditional density estimation, and density estimation with missing data tasks.

        ----

        ## [3235] Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation

        **Authors**: *Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, Shenghua Gao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html)

        **Abstract**:

        We present a novel alignment-before-generation approach to tackle the challenging task of generating general 3D shapes based on 2D images or texts. Directly learning a conditional generative model from images or texts to 3D shapes is prone to producing inconsistent results with the conditions because 3D shapes have an additional dimension whose distribution significantly differs from that of 2D images and texts. To bridge the domain gap among the three modalities and facilitate multi-modal-conditioned 3D shape generation, we explore representing 3D shapes in a shape-image-text-aligned space. Our framework comprises two models: a Shape-Image-Text-Aligned Variational Auto-Encoder (SITA-VAE) and a conditional Aligned Shape Latent Diffusion Model (ASLDM). The former model encodes the 3D shapes into the shape latent space aligned to the image and text and reconstructs the fine-grained 3D neural fields corresponding to given shape embeddings via the transformer-based decoder. The latter model learns a probabilistic mapping function from the image or text space to the latent shape space. Our extensive experiments demonstrate that our proposed approach can generate higher-quality and more diverse 3D shapes that better semantically conform to the visual or textural conditional inputs, validating the effectiveness of the shape-image-text-aligned space for cross-modality 3D shape generation.

        ----

        ## [3236] Estimating Riemannian Metric with Noise-Contaminated Intrinsic Distance

        **Authors**: *Jiaming Qiu, Xiongtao Dai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea5cb7d9fd2deb0554def3552962d276-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea5cb7d9fd2deb0554def3552962d276-Abstract-Conference.html)

        **Abstract**:

        We extend metric learning by studying the Riemannian manifold structure of the underlying data space induced by similarity measures between data points. The key quantity of interest here is the Riemannian metric, which characterizes the Riemannian geometry and defines straight lines and derivatives on the manifold. Being able to estimate the Riemannian metric allows us to gain insights into the underlying manifold and compute geometric features such as the geodesic curves. We model the observed similarity measures as noisy responses generated from a function of the intrinsic geodesic distance between data points. A new local regression approach is proposed to learn the Riemannian metric tensor and its derivatives based on a Taylor expansion for the squared geodesic distances, accommodating different types of data such as continuous, binary, or comparative responses. We develop theoretical foundation for our method by deriving the rates of convergence for the asymptotic bias and variance of the estimated metric tensor. The proposed method is shown to be versatile in simulation studies and real data applications involving taxi trip time in New York City and MNIST digits.

        ----

        ## [3237] Inner Product-based Neural Network Similarity

        **Authors**: *Wei Chen, Zichen Miao, Qiang Qiu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea7623ff02edffe68866f88da2667592-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea7623ff02edffe68866f88da2667592-Abstract-Conference.html)

        **Abstract**:

        Analyzing representational similarity among neural networks (NNs) is essential for interpreting or transferring deep models. In application scenarios where numerous NN models are learned, it becomes crucial to assess model similarities in computationally efficient ways. In this paper, we propose a new paradigm for reducing NN representational similarity to filter subspace distance. Specifically, when convolutional filters are decomposed as a linear combination of a set of filter subspace elements, denoted as filter atoms, and have those decomposed atom coefficients shared across networks, NN representational similarity can be significantly simplified as calculating the cosine distance among respective filter atoms, to achieve millions of times computation reduction over popular probing-based methods. We provide both theoretical and empirical evidence that such simplified filter subspace-based similarity preserves a strong linear correlation with other popular probing-based metrics, while being significantly more efficient to obtain and robust to probing data. We further validate the effectiveness of the proposed method in various application scenarios where numerous models exist, such as federated and continual learning as well as analyzing training dynamics. We hope our findings can help further explorations of real-time large-scale representational similarity analysis in neural networks.

        ----

        ## [3238] State-space models with layer-wise nonlinearity are universal approximators with exponential decaying memory

        **Authors**: *Shida Wang, Beichen Xue*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea8608c6258450e75b3443ec8022fb2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea8608c6258450e75b3443ec8022fb2e-Abstract-Conference.html)

        **Abstract**:

        State-space models have gained popularity in sequence modelling due to their simple and efficient network structures. However, the absence of nonlinear activation along the temporal direction limits the model's capacity. In this paper, we prove that stacking state-space models with layer-wise nonlinear activation is sufficient to approximate any continuous sequence-to-sequence relationship. Our findings demonstrate that the addition of layer-wise nonlinear activation enhances the model's capacity to learn complex sequence patterns. Meanwhile, it can be seen both theoretically and empirically that the state-space models do not fundamentally resolve the issue of exponential decaying memory. Theoretical results are justified by numerical verifications.

        ----

        ## [3239] Decoding the Enigma: Benchmarking Humans and AIs on the Many Facets of Working Memory

        **Authors**: *Ankur Sikarwar, Mengmi Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea8758dbe6cc5e6e1764c009acb4c31e-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Working memory (WM), a fundamental cognitive process facilitating the temporary storage, integration, manipulation, and retrieval of information, plays a vital role in reasoning and decision-making tasks. Robust benchmark datasets that capture the multifaceted nature of WM are crucial for the effective development and evaluation of AI WM models. Here, we introduce a comprehensive Working Memory (WorM) benchmark dataset for this purpose. WorM comprises 10 tasks and a total of 1 million trials, assessing 4 functionalities, 3 domains, and 11 behavioral and neural characteristics of WM. We jointly trained and tested state-of-the-art recurrent neural networks and transformers on all these tasks. We also include human behavioral benchmarks as an upper bound for comparison. Our results suggest that AI models replicate some characteristics of WM in the brain, most notably primacy and recency effects, and neural clusters and correlates specialized for different domains and functionalities of WM. In the experiments, we also reveal some limitations in existing models to approximate human behavior. This dataset serves as a valuable resource for communities in cognitive psychology, neuroscience, and AI, offering a standardized framework to compare and enhance WM models, investigate WM's neural underpinnings, and develop WM models with human-like capabilities. Our source code and data are available at: https://github.com/ZhangLab-DeepNeuroCogLab/WorM

        ----

        ## [3240] Many-body Approximation for Non-negative Tensors

        **Authors**: *Kazu Ghalamkari, Mahito Sugiyama, Yoshinobu Kawahara*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html)

        **Abstract**:

        We present an alternative approach to decompose non-negative tensors, called many-body approximation. Traditional decomposition methods assume low-rankness in the representation, resulting in difficulties in global optimization and target rank selection. We avoid these problems by energy-based modeling of tensors, where a tensor and its mode correspond to a probability distribution and a random variable, respectively. Our model can be globally optimized in terms of the KL divergence minimization by taking the interaction between variables (that is, modes), into account that can be tuned more intuitively than ranks. Furthermore, we visualize interactions between modes as tensor networks and reveal a nontrivial relationship between many-body approximation and low-rank approximation. We demonstrate the effectiveness of our approach in tensor completion and approximation.

        ----

        ## [3241] Breaking the Communication-Privacy-Accuracy Tradeoff with f-Differential Privacy

        **Authors**: *Richeng Jin, Zhonggen Su, Caijun Zhong, Zhaoyang Zhang, Tony Q. S. Quek, Huaiyu Dai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ead13878cd158f013becb6a559a60364-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ead13878cd158f013becb6a559a60364-Abstract-Conference.html)

        **Abstract**:

        We consider a federated data analytics problem in which a server coordinates the collaborative data analysis of multiple users with privacy concerns and limited communication capability. The commonly adopted compression schemes introduce information loss into local data while improving communication efficiency, and it remains an open problem whether such discrete-valued mechanisms provide any privacy protection. In this paper, we study the local differential privacy guarantees of discrete-valued mechanisms with finite output space through the lens of $f$-differential privacy (DP). More specifically, we advance the existing literature by deriving tight $f$-DP guarantees for a variety of discrete-valued mechanisms, including the binomial noise and the binomial mechanisms that are proposed for privacy preservation, and the sign-based methods that are proposed for data compression, in closed-form expressions. We further investigate the amplification in privacy by sparsification and propose a ternary stochastic compressor. By leveraging compression for privacy amplification, we improve the existing methods by removing the dependency of accuracy (in terms of mean square error) on communication cost in the popular use case of distributed mean estimation, therefore breaking the three-way tradeoff between privacy, communication, and accuracy.

        ----

        ## [3242] Multi-Prompt Alignment for Multi-Source Unsupervised Domain Adaptation

        **Authors**: *Haoran Chen, Xintong Han, Zuxuan Wu, Yu-Gang Jiang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eadeef7c51ad86989cc3b311cb49ec89-Abstract-Conference.html)

        **Abstract**:

        Most existing methods for unsupervised domain adaptation (UDA) rely on a shared network to extract domain-invariant features. However, when facing multiple source domains, optimizing such a network involves updating the parameters of the entire network, making it both computationally expensive and challenging, particularly when coupled with min-max objectives. Inspired by recent advances in prompt learning that adapts high-capacity models for downstream tasks in a computationally economic way, we introduce Multi-Prompt Alignment (MPA), a simple yet efficient framework for multi-source UDA. Given a source and target domain pair, MPA first trains an individual prompt to minimize the domain gap through a contrastive loss. Then, MPA denoises the learned prompts through an auto-encoding process and aligns them by maximizing the agreement of all the reconstructed prompts. Moreover, we show that the resulting subspace acquired from the auto-encoding process can easily generalize to a streamlined set of target domains, making our method more efficient for practical usage. Extensive experiments show that MPA achieves state-of-the-art results on three popular datasets with an impressive average accuracy of 54.1% on DomainNet.

        ----

        ## [3243] Characterization and Learning of Causal Graphs with Small Conditioning Sets

        **Authors**: *Murat Kocaoglu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eaef3b49866b942041a34bb8da397eb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eaef3b49866b942041a34bb8da397eb7-Abstract-Conference.html)

        **Abstract**:

        Constraint-based causal discovery algorithms learn part of the causal graph structure by systematically testing conditional independences  observed in the data. These algorithms, such as the PC algorithm and its variants, rely on graphical characterizations of the so-called equivalence class of causal graphs proposed by Pearl. However, constraint-based causal discovery algorithms struggle when data is limited since conditional independence tests quickly lose their statistical power, especially when the conditioning set is large. To address this, we propose using conditional independence tests where the size of the conditioning set is upper bounded by some integer k for robust causal discovery. The existing graphical characterizations of the equivalence classes of causal graphs are not applicable when we cannot leverage all the conditional independence statements. We first define the notion of k-Markov equivalence: Two causal graphs are k-Markov equivalent if they entail the same conditional independence constraints where the conditioning set size is upper bounded by k. We propose a novel representation that allows us to graphically characterize k-Markov equivalence between two causal graphs. We propose a sound constraint-based algorithm called the k-PC algorithm for learning this equivalence class. Finally, we conduct synthetic, and semi-synthetic experiments to demonstrate that the k-PC algorithm enables more robust causal discovery in the small sample regime compared to the baseline algorithms.

        ----

        ## [3244] Finite Population Regression Adjustment and Non-asymptotic Guarantees for Treatment Effect Estimation

        **Authors**: *Mehrdad Ghadiri, David Arbour, Tung Mai, Cameron Musco, Anup B. Rao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eaf5d2cdb582c058a078d4fdf52a20f9-Abstract-Conference.html)

        **Abstract**:

        The design and analysis of randomized experiments is fundamental to many areas, from the physical and social sciences to industrial settings. Regression adjustment is a popular technique to reduce the variance of estimates obtained from experiments, by utilizing information contained in auxiliary covariates. While there is a large literature within the statistics community studying various approaches to regression adjustment and their asymptotic properties, little focus has been given to approaches in the finite population setting with non-asymptotic accuracy bounds. Further, prior work typically assumes that an entire population is exposed to an experiment, whereas practitioners often seek to minimize the number of subjects exposed to an experiment, for ethical and pragmatic reasons.In this work, we study the problems of estimating the sample mean, individual treatment effects, and average treatment effect with regression adjustment. We propose approaches that use techniques from randomized numerical linear algebra to sample a subset of the population on which to perform an experiment. We give non-asymptotic accuracy bounds for our methods and demonstrate that they compare favorably with prior approaches.

        ----

        ## [3245] P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting

        **Authors**: *Sungwon Kim, Kevin J. Shih, Rohan Badlani, João Felipe Santos, Evelina Bakhturina, Mikyas Desta, Rafael Valle, Sungroh Yoon, Bryan Catanzaro*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html)

        **Abstract**:

        While recent large-scale neural codec language models have shown significant improvement in zero-shot TTS by training on thousands of hours of data, they suffer from drawbacks such as a lack of robustness, slow sampling speed similar to previous autoregressive TTS methods, and reliance on pre-trained neural codec representations. Our work proposes P-Flow, a fast and data-efficient zero-shot TTS model that uses speech prompts for speaker adaptation. P-Flow comprises a speech-prompted text encoder for speaker adaptation and a flow matching generative decoder for high-quality and fast speech synthesis. Our speech-prompted text encoder uses speech prompts and text input to generate speaker-conditional text representation. The flow matching generative decoder uses the speaker-conditional output to synthesize high-quality personalized speech significantly faster than in real-time. Unlike the neural codec language models, we specifically train P-Flow on LibriTTS dataset using a continuous mel-representation. Through our training method using continuous speech prompts, P-Flow matches the speaker similarity performance of the large-scale zero-shot TTS models with two orders of magnitude less training data and has more than 20$\times$ faster sampling speed. Our results show that P-Flow has better pronunciation and is preferred in human likeness and speaker similarity to its recent state-of-the-art counterparts, thus defining P-Flow as an attractive and desirable alternative. We provide audio samples on our demo page: [https://research.nvidia.com/labs/adlr/projects/pflow](https://research.nvidia.com/labs/adlr/projects/pflow)

        ----

        ## [3246] Implicit Bias of Gradient Descent for Logistic Regression at the Edge of Stability

        **Authors**: *Jingfeng Wu, Vladimir Braverman, Jason D. Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb189151ced0ff808abafd16a51fec92-Abstract-Conference.html)

        **Abstract**:

        Recent research has observed that in machine learning optimization, gradient descent (GD) often operates at the edge of stability (EoS) [Cohen et al., 2021], where the stepsizes are set to be large, resulting in non-monotonic losses induced by the GD iterates. This paper studies the convergence and implicit bias of constant-stepsize GD for logistic regression on linearly separable data in the EoS regime. Despite the presence of local oscillations, we prove that the logistic loss can be minimized by GD with any constant stepsize over a long time scale. Furthermore, we prove that with any constant stepsize, the GD iterates tend to infinity when projected to a max-margin direction (the hard-margin SVM direction) and converge to a fixed vector that minimizes a strongly convex potential when projected to the orthogonal complement of the max-margin direction. In contrast, we also show that in the EoS regime, GD iterates may diverge catastrophically under the exponential loss, highlighting the superiority of the logistic loss. These theoretical findings are in line with numerical simulations and complement existing theories on the convergence and implicit bias of GD for logistic regression, which are only applicable when the stepsizes are sufficiently small.

        ----

        ## [3247] Two Sides of One Coin: the Limits of Untuned SGD and the Power of Adaptive Methods

        **Authors**: *Junchi Yang, Xiang Li, Ilyas Fatkhullin, Niao He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb1a323fa10d4102ff13422476a744ff-Abstract-Conference.html)

        **Abstract**:

        The classical analysis of Stochastic Gradient Descent (SGD) with polynomially decaying stepsize $\eta_t = \eta/\sqrt{t}$ relies on well-tuned $\eta$ depending on  problem parameters such as Lipschitz smoothness constant, which is often unknown in practice. In this work, we prove that SGD with arbitrary $\eta > 0$, referred to as untuned SGD, still attains an order-optimal convergence rate $\widetilde{\mathcal{O}}(T^{-1/4})$ in terms of gradient norm for minimizing smooth objectives. Unfortunately, it comes at the expense of a catastrophic exponential dependence on the smoothness constant, which we show is unavoidable for this scheme even in the noiseless setting. We then examine three families of adaptive methods — Normalized SGD (NSGD), AMSGrad, and AdaGrad — unveiling their power in preventing such exponential dependency in  the absence of information about the smoothness parameter and boundedness of stochastic gradients. Our results provide  theoretical justification for the advantage of adaptive methods over untuned SGD in alleviating the issue with large gradients.

        ----

        ## [3248] Theoretical Analysis of the Inductive Biases in Deep Convolutional Networks

        **Authors**: *Zihao Wang, Lei Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb1bad7a84ef68a64f1afd6577725d45-Abstract-Conference.html)

        **Abstract**:

        In this paper, we provide a theoretical analysis of the inductive biases in convolutional neural networks (CNNs). We start by examining the universality of CNNs, i.e., the ability to approximate any continuous functions.  We prove that a depth of $\mathcal{O}(\log d)$ suffices for deep CNNs to achieve this universality, where $d$ in the input dimension.  Additionally, we establish  that learning sparse functions with  CNNs requires only $\widetilde{\mathcal{O}}(\log^2d)$ samples, indicating that deep CNNs can efficiently capture {\em long-range} sparse correlations. These results are made possible through a novel combination of the multichanneling and downsampling when increasing the network depth. We also delve into the distinct roles  of weight sharing and locality in CNNs. To this end, we compare the performance of CNNs, locally-connected networks (LCNs), and fully-connected networks (FCNs) on a simple regression task, where LCNs can be viewed as CNNs without weight sharing. On the one hand,  we  prove that LCNs require ${\Omega}(d)$ samples while CNNs need only $\widetilde{\mathcal{O}}(\log^2d)$ samples,  highlighting the  critical role of weight sharing. On the other hand, we prove that FCNs require $\Omega(d^2)$ samples, whereas LCNs need only $\widetilde{\mathcal{O}}(d)$ samples,  underscoring the importance  of locality. These provable separations quantify the difference between the two biases, and the major observation behind our proof is that weight sharing and locality break different symmetries in the learning process.

        ----

        ## [3249] Object Reprojection Error (ORE): Camera pose benchmarks from lightweight tracking annotations

        **Authors**: *Xingyu Chen, Weiyao Wang, Hao Tang, Matt Feiszli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb206443c93d07da8b1974b768d8a0d4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb206443c93d07da8b1974b768d8a0d4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        3D spatial understanding is highly valuable in the context of semantic modeling of environments, agents, and their relationships.  Semantic modeling approaches employed on monocular video often ingest outputs from off-the-shelf SLAM/SfM pipelines, which are anecdotally observed to perform poorly or fail completely on some fraction of the videos of interest.  These target videos may vary widely in complexity of scenes, activities, camera trajectory, etc.  Unfortunately, such semantically-rich video data often comes with no ground-truth 3D information, and in practice it is prohibitively costly or impossible to obtain ground truth reconstructions or camera pose post-hoc.  This paper proposes a novel evaluation protocol, Object Reprojection Error (ORE) to benchmark camera trajectories; ORE computes reprojection error for static objects within the video and requires only lightweight object tracklet annotations.  These annotations are easy to gather on new or existing video, enabling ORE to be calculated on essentially arbitrary datasets.  We show that ORE maintains high rank correlation with standard metrics based on groundtruth.  Leveraging ORE, we source videos and annotations from Ego4D-EgoTracks, resulting in EgoStatic, a large-scale diverse dataset for evaluating camera trajectories in-the-wild.

        ----

        ## [3250] Rethinking Bias Mitigation: Fairer Architectures Make for Fairer Face Recognition

        **Authors**: *Samuel Dooley, Rhea Sukthanker, John P. Dickerson, Colin White, Frank Hutter, Micah Goldblum*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html)

        **Abstract**:

        Face recognition systems are widely deployed in safety-critical applications, including law enforcement, yet they exhibit bias across a range of socio-demographic dimensions, such as gender and race.  Conventional wisdom dictates that model biases arise from biased training data.  As a consequence, previous works on bias mitigation largely focused on pre-processing the training data, adding penalties to prevent bias from effecting the model during training, or post-processing predictions to debias them, yet these approaches have shown limited success on hard problems such as face recognition.  In our work, we discover that biases are actually inherent to neural network architectures themselves.  Following this reframing, we conduct the first neural architecture search for fairness, jointly with a search for hyperparameters. Our search outputs a suite of models which Pareto-dominate all other high-performance architectures and existing bias mitigation methods in terms of accuracy and fairness, often by large margins, on the two most widely used datasets for face identification, CelebA and VGGFace2. Furthermore, these models generalize to other datasets and sensitive attributes. We release our code, models and raw data files at https://github.com/dooleys/FR-NAS.

        ----

        ## [3251] H-InDex: Visual Reinforcement Learning with Hand-Informed Representations for Dexterous Manipulation

        **Authors**: *Yanjie Ze, Yuyao Liu, Ruizhe Shi, Jiaxin Qin, Zhecheng Yuan, Jiashun Wang, Huazhe Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb4b1f7feadcd124a59de6ff7b9196f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb4b1f7feadcd124a59de6ff7b9196f3-Abstract-Conference.html)

        **Abstract**:

        Human hands possess remarkable dexterity and have long served as a source of inspiration for robotic manipulation. In this work, we propose a human $\textbf{H}$and-$\textbf{In}$formed visual representation learning framework to solve difficult $\textbf{Dex}$terous manipulation tasks ($\textbf{H-InDex}$) with reinforcement learning. Our framework consists of three stages: $\textit{(i)}$ pre-training representations with 3D human hand pose estimation, $\textit{(ii)}$ offline adapting representations with self-supervised keypoint detection, and $\textit{(iii)}$ reinforcement learning with exponential moving average BatchNorm. The last two stages only modify $0.36$% parameters of the pre-trained representation in total, ensuring the knowledge from pre-training is maintained to the full extent. We empirically study $\textbf{12}$ challenging dexterous manipulation tasks and find that $\textbf{H-InDex}$ largely surpasses strong baseline methods and the recent visual foundation models for motor control. Code and videos are available at https://yanjieze.com/H-InDex .

        ----

        ## [3252] DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets

        **Authors**: *Lazar Atanackovic, Alexander Tong, Bo Wang, Leo J. Lee, Yoshua Bengio, Jason S. Hartford*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html)

        **Abstract**:

        One of the grand challenges of cell biology is inferring the gene regulatory network (GRN) which describes interactions between genes and their products that control gene expression and cellular function. We can treat this as a causal discovery problem but with two non-standard challenges: (1) regulatory networks are inherently cyclic so we should not model a GRN as a directed acyclic graph (DAG), and (2) observations have significant measurement noise so for typical sample sizes, there will always be a large equivalence class of graphs that are likely given the data, and we want methods that capture this uncertainty. Existing methods either focus on challenge (1), identifying cyclic structure from dynamics, or on challenge (2) learning complex Bayesian posteriors over directed acyclic graphs, but not both. In this paper we leverage the fact that it is possible to estimate the ``velocity'' of the expression of a gene with RNA velocity techniques to develop an approach that addresses both challenges. Because we have access to velocity information, we can treat the Bayesian structure learning problem as a problem of sparse identification of a dynamical system, capturing cyclic feedback loops through time. We leverage Generative Flow Networks (GFlowNets) to estimate the posterior distribution over the combinatorial space of possible sparse dependencies. Our results indicate that our method learns posteriors that better encapsulate the distributions of cyclic structures compared to counterpart state-of-the-art Bayesian structure learning approaches.

        ----

        ## [3253] RaLEs: a Benchmark for Radiology Language Evaluations

        **Authors**: *Juanma Zambrano Chaves, Nandita Bhaskhar, Maayane Attias, Jean-Benoit Delbrouck, Daniel L. Rubin, Andreas M. Loening, Curtis P. Langlotz, Akshay Chaudhari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5683d06bdef51ed4dff644908eef4b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5683d06bdef51ed4dff644908eef4b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The radiology report is the main form of communication between radiologists and other clinicians. Prior work in natural language processing in radiology reports has shown the value of developing methods tailored for individual tasks such as identifying reports with critical results or disease detection. Meanwhile, English and biomedical natural language understanding benchmarks such as the General Language Understanding and Evaluation as well as Biomedical Language Understanding and Reasoning Benchmark have motivated the development of models that can be easily adapted to address many tasks in those domains. Here, we characterize the radiology report as a distinct domain and introduce RaLEs, the Radiology Language Evaluations, as a benchmark for natural language understanding and generation in radiology. RaLEs is comprised of seven natural language understanding and generation evaluations including the extraction of anatomical and disease entities and their relations, procedure selection, and report summarization. We characterize the performance of models designed for the general, biomedical, clinical and radiology domains across these tasks. We find that advances in the general and biomedical domains do not necessarily translate to radiology, and that improved models from the general domain can perform comparably to smaller clinical-specific models. The limited performance of existing pre-trained models on RaLEs highlights the opportunity to improve domain-specific self-supervised models for natural language processing in radiology. We propose RaLEs as a benchmark to promote and track the development of such domain-specific radiology language models.

        ----

        ## [3254] AutoGO: Automated Computation Graph Optimization for Neural Network Evolution

        **Authors**: *Mohammad Salameh, Keith G. Mills, Negar Hassanpour, Fred X. Han, Shuting Zhang, Wei Lu, Shangling Jui, Chunhua Zhou, Fengyu Sun, Di Niu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eb5d9195b201ec7ba66c8e20b396d349-Abstract-Conference.html)

        **Abstract**:

        Optimizing Deep Neural Networks (DNNs) to obtain high-quality models for efficient real-world deployment has posed multi-faceted challenges to machine learning engineers. Existing methods either search for neural architectures in heuristic design spaces or apply low-level adjustments to computation primitives to improve inference efficiency on hardware. We present Automated Graph Optimization (AutoGO), a framework to evolve neural networks in a low-level Computation Graph (CG) of primitive operations to improve both its performance and hardware friendliness. Through a tokenization scheme, AutoGO performs variable-sized segment mutations, making both primitive changes and larger-grained changes to CGs. We introduce our segmentation and mutation algorithms, efficient frequent segment mining technique, as well as a pretrained context-aware predictor to estimate the impact of segment replacements. Extensive experimental results show that AutoGO can automatically evolve several typical large convolutional networks to achieve significant task performance improvement and FLOPs reduction on a range of CV tasks, ranging from Classification, Semantic Segmentation, Human Pose Estimation, to Super Resolution, yet without introducing any newer primitive operations. We also demonstrate the lightweight deployment results of AutoGO-optimized super-resolution and denoising U-Nets on a cycle simulator for a Neural Processing Unit (NPU), achieving PSNR improvement and latency/power reduction simultaneously. Code available at https://github.com/Ascend-Research/AutoGO.

        ----

        ## [3255] Where Did I Come From? Origin Attribution of AI-Generated Images

        **Authors**: *Zhenting Wang, Chen Chen, Yi Zeng, Lingjuan Lyu, Shiqing Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebb4c188fafe7da089b41a9f615ad84d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebb4c188fafe7da089b41a9f615ad84d-Abstract-Conference.html)

        **Abstract**:

        Image generation techniques have been gaining increasing attention recently, but concerns have been raised about the potential misuse and intellectual property (IP) infringement associated with image generation models. It is, therefore, necessary to analyze the origin of images by inferring if a specific image was generated by a particular model, i.e., origin attribution. Existing methods only focus on specific types of generative models and require additional procedures during the training phase or generation phase. This makes them unsuitable for pre-trained models that lack these specific operations and may impair generation quality. To address this problem, we first develop an alteration-free and model-agnostic origin attribution method via reverse-engineering on image generation models, i.e., inverting the input of a particular model for a specific image. Given a particular model, we first analyze the differences in the hardness of reverse-engineering tasks for generated samples of the given model and other images. Based on our analysis, we then propose a method that utilizes the reconstruction loss of reverse-engineering to infer the origin. Our proposed method effectively distinguishes between generated images of a specific generative model and other images, i.e., images generated by other models and real images.

        ----

        ## [3256] Robust Data Pruning under Label Noise via Maximizing Re-labeling Accuracy

        **Authors**: *Dongmin Park, Seola Choi, Doyoung Kim, Hwanjun Song, Jae-Gil Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebb6bee50913ba7e1efeb91a1d47a002-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebb6bee50913ba7e1efeb91a1d47a002-Abstract-Conference.html)

        **Abstract**:

        Data pruning, which aims to downsize a large training set into a small informative subset, is crucial for reducing the enormous computational costs of modern deep learning. Though large-scale data collections invariably contain annotation noise and numerous robust learning methods have been developed, data pruning for the noise-robust learning scenario has received little attention. With state-of-the-art Re-labeling methods that self-correct erroneous labels while training, it is challenging to identify which subset induces the most accurate re-labeling of erroneous labels in the entire training set. In this paper, we formalize the problem of data pruning with re-labeling. We first show that the likelihood of a training example being correctly re-labeled is proportional to the prediction confidence of its neighborhood in the subset. Therefore, we propose a novel data pruning algorithm, Prune4Rel, that finds a subset maximizing the total neighborhood confidence of all training examples, thereby maximizing the re-labeling accuracy and generalization performance. Extensive experiments on four real and one synthetic noisy datasets show that Prune4Rel outperforms the baselines with Re-labeling models by up to 9.1% as well as those with a standard model by up to 21.6%.

        ----

        ## [3257] WildfireSpreadTS: A dataset of multi-modal time series for wildfire spread prediction

        **Authors**: *Sebastian Gerard, Yu Zhao, Josephine Sullivan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present a multi-temporal, multi-modal remote-sensing dataset for predicting how active wildfires will spread at a resolution of 24 hours. The dataset consists of 13607 images across 607 fire events in the United States from January 2018 to October 2021. For each fire event, the dataset contains a full time series of daily observations, containing detected active fires and variables related to fuel, topography and weather conditions. The dataset is challenging due to: a) its inputs being multi-temporal, b) the high number of 23 multi-modal input channels, c) highly imbalanced labels and d) noisy labels, due to smoke, clouds, and inaccuracies in the active fire detection. The underlying complexity of the physical processes adds to these challenges. Compared to existing public datasets in this area, WildfireSpreadTS allows for multi-temporal modeling of spreading wildfires, due to its time series structure. Furthermore, we provide additional input modalities and a high spatial resolution of 375m for the active fire maps. We publish this dataset to encourage further research on this important task with multi-temporal, noise-resistant or generative methods, uncertainty estimation or advanced optimization techniques that deal with the high-dimensional input space.

        ----

        ## [3258] Augmenting Language Models with Long-Term Memory

        **Authors**: *Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, Furu Wei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebd82705f44793b6f9ade5a669d0f0bf-Abstract-Conference.html)

        **Abstract**:

        Existing large language models (LLMs) can only afford fix-sized inputs due to the input length limit, preventing them from utilizing rich long-context information from past inputs. To address this, we propose a framework, Language Models Augmented with Long-Term Memory (LongMem), which enables LLMs to memorize long history. We design a novel decoupled network architecture with the original backbone LLM frozen as a memory encoder and an adaptive residual side-network as a memory retriever and reader. Such a decoupled memory design can easily cache and update long-term past contexts for memory retrieval without suffering from memory staleness. Enhanced with memory-augmented adaptation training, LongMem can thus memorize long past context and use long-term memory for language modeling. The proposed memory retrieval module can handle unlimited-length context in its memory bank to benefit various downstream tasks. Typically, LongMem can enlarge the long-form memory to 65k tokens and thus cache many-shot extra demonstration examples as long-form memory for in-context learning. Experiments show that our method outperforms strong long-context models on ChapterBreak, a challenging long-context modeling benchmark, and achieves remarkable improvements on memory-augmented in-context learning over LLMs. The results demonstrate that the proposed method is effective in helping language models to memorize and utilize long-form contents.

        ----

        ## [3259] Expressivity-Preserving GNN Simulation

        **Authors**: *Fabian Jogl, Maximilian Thiessen, Thomas Gärtner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebf95a6f3c575322da15d4fd0fc2b3c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebf95a6f3c575322da15d4fd0fc2b3c8-Abstract-Conference.html)

        **Abstract**:

        We systematically investigate graph transformations that enable standard message passing to simulate state-of-the-art graph neural networks (GNNs) without loss of expressivity. Using these, many state-of-the-art GNNs can be implemented with message passing operations from standard libraries, eliminating many sources of implementation issues and allowing for better code optimization. We distinguish between weak and strong simulation: weak simulation achieves the same expressivity only after several message passing steps while strong simulation achieves this after every message passing step. Our contribution leads to a direct way to translate common operations of non-standard GNNs to graph transformations that allow for strong or weak simulation. Our empirical evaluation shows competitive predictive performance of message passing on transformed graphs for various molecular benchmark datasets, in several cases surpassing the original GNNs.

        ----

        ## [3260] Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?

        **Authors**: *Fan Yao, Chuanhao Li, Karthik Abinav Sankararaman, Yiming Liao, Yan Zhu, Qifan Wang, Hongning Wang, Haifeng Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ebfabf372037aaa4a8d92c9b457ece3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ebfabf372037aaa4a8d92c9b457ece3e-Abstract-Conference.html)

        **Abstract**:

        The past decade has witnessed the flourishing of a new profession as media content creators, who rely on revenue streams from online content recommendation platforms. The reward mechanism employed by these platforms creates a competitive environment among creators which affects their production choices and, consequently, content distribution and system welfare. It is thus crucial to design the platform's reward mechanism in order to steer the creators' competition towards a desirable welfare outcome in the long run. This work makes two major contributions in this regard: first, we uncover a fundamental limit about a class of widely adopted mechanisms, coined \emph{Merit-based Monotone Mechanisms}, by showing that they inevitably lead to a constant fraction loss of the optimal welfare. To circumvent this limitation, we introduce \emph{Backward Rewarding Mechanisms} (BRMs) and show that the competition game resultant from BRMs possesses a potential game structure. BRMs thus naturally induce strategic creators' collective behaviors towards optimizing the potential function, which can be designed to match any given welfare metric. In addition, the class of BRM can be parameterized so that it allows the platform to directly optimize welfare within the feasible mechanism space even when the welfare metric is not explicitly defined.

        ----

        ## [3261] Gaussian Partial Information Decomposition: Bias Correction and Application to High-dimensional Data

        **Authors**: *Praveen Venkatesh, Corbett Bennett, Sam Gale, Tamina K. Ramirez, Greggory Heller, Severine Durand, Shawn R. Olsen, Stefan Mihalas*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec0bff8bf4b11e36f874790046dfdb65-Abstract-Conference.html)

        **Abstract**:

        Recent advances in neuroscientific experimental techniques have enabled us to simultaneously record the activity of thousands of neurons across multiple brain regions. This has led to a growing need for computational tools capable of analyzing how task-relevant information is represented and communicated between several brain regions. Partial information decompositions (PIDs) have emerged as one such tool, quantifying how much unique, redundant and synergistic information two or more brain regions carry about a task-relevant message. However, computing PIDs is computationally challenging in practice, and statistical issues such as the bias and variance of estimates remain largely unexplored. In this paper, we propose a new method for efficiently computing and estimating a PID definition on multivariate Gaussian distributions. We show empirically that our method satisfies an intuitive additivity property, and recovers the ground truth in a battery of canonical examples, even at high dimensionality. We also propose and evaluate, for the first time, a method to correct the bias in PID estimates at finite sample sizes. Finally, we demonstrate that our Gaussian PID effectively characterizes inter-areal interactions in the mouse brain, revealing higher redundancy between visual areas when a stimulus is behaviorally relevant.

        ----

        ## [3262] Unconstrained Dynamic Regret via Sparse Coding

        **Authors**: *Zhiyu Zhang, Ashok Cutkosky, Yannis Paschalidis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec2833cda146c277cdaa39066764f25c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec2833cda146c277cdaa39066764f25c-Abstract-Conference.html)

        **Abstract**:

        Motivated by the challenge of nonstationarity in sequential decision making, we study Online Convex Optimization (OCO) under the coupling of two problem structures: the domain is unbounded, and the comparator sequence $u_1,\ldots,u_T$ is arbitrarily time-varying. As no algorithm can guarantee low regret simultaneously against all comparator sequences, handling this setting requires moving from minimax optimality to comparator adaptivity. That is, sensible regret bounds should depend on certain complexity measures of the comparator relative to one's prior knowledge. This paper achieves a new type of such adaptive regret bounds leveraging a sparse coding framework. The complexity of the comparator is measured by its energy and its sparsity on a user-specified dictionary, which offers considerable versatility. For example, equipped with a wavelet dictionary, our framework improves the state-of-the-art bound (Jacobsen & Cutkosky, 2022) by adapting to both ($i$) the magnitude of the comparator average $||\bar u||=||\sum_{t=1}^Tu_t/T||$, rather than the maximum $\max_t||u_t||$; and ($ii$) the comparator variability $\sum_{t=1}^T||u_t-\bar u||$, rather than the uncentered sum $\sum_{t=1}^T||u_t||$. Furthermore, our proof is simpler due to decoupling function approximation from regret minimization.

        ----

        ## [3263] Efficient Test-Time Adaptation for Super-Resolution with Second-Order Degradation and Reconstruction

        **Authors**: *Zeshuai Deng, Zhuokun Chen, Shuaicheng Niu, Thomas H. Li, Bohan Zhuang, Mingkui Tan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec3d49763c653ad7c8d587f52220c129-Abstract-Conference.html)

        **Abstract**:

        Image super-resolution (SR) aims to learn a mapping from low-resolution (LR) to high-resolution (HR) using paired HR-LR training images. Conventional SR methods typically gather the paired training data by synthesizing LR images from HR images using a predetermined degradation model, e.g., Bicubic down-sampling. However, the realistic degradation type of test images may mismatch with the training-time degradation type due to the dynamic changes of the real-world scenarios, resulting in inferior-quality SR images. To address this, existing methods attempt to estimate the degradation model and train an image-specific model, which, however, is quite time-consuming and impracticable to handle rapidly changing domain shifts. Moreover, these methods largely concentrate on the estimation of one degradation type (e.g., blur degradation), overlooking other degradation types like noise and JPEG in real-world test-time scenarios, thus limiting their practicality. To tackle these problems, we present an efficient test-time adaptation framework for SR, named SRTTA, which is able to quickly adapt SR models to test domains with different/unknown degradation types. Specifically, we design a second-order degradation scheme to construct paired data based on the degradation type of the test image, which is predicted by a pre-trained degradation classifier. Then, we adapt the SR model by implementing feature-level reconstruction learning from the initial test image to its second-order degraded counterparts, which helps the SR model generate plausible HR images. Extensive experiments are conducted on newly synthesized corrupted DIV2K datasets with 8 different degradations and several real-world datasets, demonstrating that our SRTTA framework achieves an impressive improvement over existing methods with satisfying speed. The source code is available at https://github.com/DengZeshuai/SRTTA.

        ----

        ## [3264] Replicability in Reinforcement Learning

        **Authors**: *Amin Karbasi, Grigoris Velegkas, Lin Yang, Felix Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec4d2e436794d1bf55ca83f5ebb31887-Abstract-Conference.html)

        **Abstract**:

        We initiate the mathematical study of replicability as an   algorithmic property in the context of reinforcement learning (RL).  We focus on the fundamental setting of discounted tabular MDPs with access to a generative model.  Inspired by Impagliazzo et al. [2022], we say that an RL algorithm is replicable if,  with high probability,  it outputs the exact same policy  after two executions on i.i.d. samples drawn from the generator  when its internal randomness  is the same.  We first provide   an efficient $\rho$-replicable algorithm for $(\varepsilon, \delta)$-optimal policy estimation  with sample and time complexity $\widetilde O\left(\frac{N^3\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$,  where $N$ is the number of state-action pairs.  Next,  for the subclass of deterministic algorithms,  we provide a lower bound of order $\Omega\left(\frac{N^3}{(1-\gamma)^3\cdot\varepsilon^2\cdot\rho^2}\right)$.  Then, we study a relaxed version of replicability proposed  by Kalavasis et al. [2023] called TV indistinguishability.  We design a computationally efficient TV indistinguishable algorithm for policy estimation  whose sample complexity is $\widetilde O\left(\frac{N^2\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.  At the cost of $\exp(N)$ running time,  we transform these TV indistinguishable algorithms to $\rho$-replicable ones without increasing their sample complexity.  Finally,  we introduce the notion of approximate-replicability  where we only require that two outputted policies are close  under an appropriate statistical divergence (e.g., Renyi)  and show an improved sample complexity of $\widetilde O\left(\frac{N\cdot\log(1/\delta)}{(1-\gamma)^5\cdot\varepsilon^2\cdot\rho^2}\right)$.

        ----

        ## [3265] Probabilistic Invariant Learning with Randomized Linear Classifiers

        **Authors**: *Leonardo Cotta, Gal Yehuda, Assaf Schuster, Chris J. Maddison*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec4f0b0a7557d6a51c42308800f2c23a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec4f0b0a7557d6a51c42308800f2c23a-Abstract-Conference.html)

        **Abstract**:

        Designing models that are both expressive and preserve known invariances of tasks is an increasingly hard problem. Existing solutions tradeoff invariance for computational or memory resources. In this work, we show how to leverage randomness and design models that are both expressive and invariant but use less resources. Inspired by randomized algorithms, our key insight is that accepting probabilistic notions of universal approximation and invariance can reduce our resource requirements. More specifically, we propose a class of binary classification models called Randomized Linear Classifiers (RLCs). We give parameter and sample size conditions in which RLCs can, with high probability, approximate any (smooth) function while preserving invariance to compact group transformations. Leveraging this result, we design three RLCs that are provably probabilistic invariant for classification tasks over sets, graphs, and spherical data. We show how these models can achieve probabilistic invariance and universality using less resources than (deterministic) neural networks and their invariant counterparts. Finally, we empirically demonstrate the benefits of this new class of models on invariant tasks where deterministic invariant neural networks are known to struggle.

        ----

        ## [3266] FedNAR: Federated Optimization with Normalized Annealing Regularization

        **Authors**: *Junbo Li, Ang Li, Chong Tian, Qirong Ho, Eric P. Xing, Hongyi Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec52572b9e16b91edff5dc70e2642240-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec52572b9e16b91edff5dc70e2642240-Abstract-Conference.html)

        **Abstract**:

        Weight decay is a standard technique to improve generalization performance in modern deep neural network optimization, and is also widely adopted in federated learning (FL) to prevent overfitting in local clients. In this paper, we first explore the choices of weight decay and identify that weight decay value appreciably influences the convergence of existing FL algorithms. While preventing overfitting is crucial, weight decay can introduce a different optimization goal towards the global objective, which is further amplified in FL due to multiple local updates and heterogeneous data distribution.To address this challenge, we develop {\it Federated optimization with Normalized Annealing Regularization} (FedNAR), a simple yet effective and versatile algorithmic plug-in that can be seamlessly integrated into any existing FL algorithms. Essentially, we regulate the magnitude of each update by performing co-clipping of the gradient and weight decay.We provide a comprehensive theoretical analysis of FedNAR's convergence rate and conduct extensive experiments on both vision and language datasets with different backbone federated optimization algorithms. Our experimental results consistently demonstrate that incorporating FedNAR into existing FL algorithms leads to accelerated convergence and heightened model accuracy. Moreover, FedNAR exhibits resilience in the face of various hyperparameter configurations. Specifically, FedNAR has the ability to self-adjust the weight decay when the initial specification is not optimal, while the accuracy of traditional FL algorithms would markedly decline. Our codes are released at \href{https://anonymous.4open.science/r/fednar-BE8F}{https://anonymous.4open.science/r/fednar-BE8F}.

        ----

        ## [3267] How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources

        **Authors**: *Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, Hannaneh Hajishirzi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec6413875e4ab08d7bc4d8e225263398-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec6413875e4ab08d7bc4d8e225263398-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In this work we explore recent advances in instruction-tuning language models on a range of open instruction-following datasets. Despite recent claims that open models can be on par with state-of-the-art proprietary models, these claims are often accompanied by limited evaluation, making it difficult to compare models across the board and determine the utility of various resources. We provide a large set of instruction-tuned models from 6.7B to 65B parameters in size, trained on 12 instruction datasets ranging from manually curated (e.g., OpenAssistant) to synthetic and distilled (e.g., Alpaca) and systematically evaluate them on their factual knowledge, reasoning, multilinguality, coding, safety, and open-ended instruction following abilities through a collection of automatic, model-based, and human-based metrics. We further introduce Tülu, our best performing instruction-tuned model suite finetuned on a combination of high-quality open resources.Our experiments show that different instruction-tuning datasets can uncover or enhance specific skills, while no single dataset (or combination) provides the best performance across all evaluations. Interestingly, we find that model and human preference-based evaluations fail to reflect differences in model capabilities exposed by benchmark-based evaluations, suggesting the need for the type of systemic evaluation performed in this work. Our evaluations show that the best model in any given evaluation reaches on average 87% of ChatGPT performance, and 73% of GPT-4 performance, suggesting that further investment in building better base models and instruction-tuning data is required to close the gap. We release our instruction-tuned models, including a fully finetuned 65B Tülu, along with our code, data, and evaluation framework to facilitate future research.

        ----

        ## [3268] Trial matching: capturing variability with data-constrained spiking neural networks

        **Authors**: *Christos Sourmpis, Carl C. H. Petersen, Wulfram Gerstner, Guillaume Bellec*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec702dd6e83b2113a43614685a7e2ac6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec702dd6e83b2113a43614685a7e2ac6-Abstract-Conference.html)

        **Abstract**:

        Simultaneous behavioral and electrophysiological recordings call for new methods to reveal the interactions between neural activity and behavior. A milestone would be an interpretable model of the co-variability of spiking activity and behavior across trials. Here, we model a mouse cortical sensory-motor pathway in a tactile detection task reported by licking with a large recurrent spiking neural network (RSNN), fitted to the recordings via gradient-based optimization. We focus specifically on the difficulty to match the trial-to-trial variability in the data. Our solution relies on optimal transport to define a distance between the distributions of generated and recorded trials. The technique is applied to artificial data and neural recordings covering six cortical areas. We find that the resulting RSNN can generate realistic cortical activity and predict jaw movements across the main modes of trial-to-trial variability. Our analysis also identifies an unexpected mode of variability in the data corresponding to task-irrelevant movements of the mouse.

        ----

        ## [3269] Decentralized Randomly Distributed Multi-agent Multi-armed Bandit with Heterogeneous Rewards

        **Authors**: *Mengfan Xu, Diego Klabjan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html)

        **Abstract**:

        We study a decentralized multi-agent multi-armed bandit problem in which multiple clients are connected by time dependent random graphs provided by an environment. The reward distributions of each arm vary across clients and rewards are generated independently over time by an environment based on distributions that include both sub-exponential and sub-gaussian distributions. Each client pulls an arm and communicates with neighbors based on the graph provided by the environment. The goal is to minimize the overall regret of the entire system through collaborations. To this end, we introduce a novel algorithmic framework, which first provides robust simulation methods for generating random graphs using rapidly mixing markov chains or the random graph model, and then combines an averaging-based consensus approach with a newly proposed weighting technique and the upper confidence bound to deliver a UCB-type solution. Our algorithms account for the randomness in the graphs, removing the conventional doubly stochasticity assumption, and only require the knowledge of the number of clients at initialization. We derive optimal instance-dependent regret upper bounds of order $\log{T}$ in both sub-gaussian and sub-exponential environments, and a nearly optimal instance-free regret upper bound of order $\sqrt{T}\log T$ up to a $\log T$ factor. Importantly, our regret bounds hold with high probability and capture graph randomness, whereas prior works consider expected regret under assumptions and require more stringent reward distributions.

        ----

        ## [3270] (Amplified) Banded Matrix Factorization: A unified approach to private training

        **Authors**: *Christopher A. Choquette-Choo, Arun Ganesh, Ryan McKenna, H. Brendan McMahan, John Rush, Abhradeep Guha Thakurta, Zheng Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecc28b4ce9b39f5f23c3efb03e25b7bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecc28b4ce9b39f5f23c3efb03e25b7bf-Abstract-Conference.html)

        **Abstract**:

        Matrix factorization (MF) mechanisms for differential privacy (DP) have substantially improved the state-of-the-art in privacy-utility-computation tradeoffs for ML applications in a variety of scenarios, but in both the centralized and federated settings there remain instances where either MF cannot be easily applied, or other algorithms provide better tradeoffs (typically, as $\epsilon$ becomes small).In this work, we show how MF can subsume prior state-of-the-art algorithms in both federated and centralized training settings, across all privacy budgets. The key technique throughout is the construction of MF mechanisms with banded matrices (lower-triangular matrices with at most $\hat{b}$ nonzero bands including the main diagonal). For cross-device federated learning (FL), this enables multiple-participations with a relaxed device participation schema compatible with practical FL infrastructure (as demonstrated by a production deployment).  In the centralized setting, we prove that banded matrices enjoy the same privacy amplification results as the ubiquitous DP-SGD algorithm, but can provide strictly better performance  in most scenarios---this lets us always at least match DP-SGD, and often outperform it

        ----

        ## [3271] Anchor Data Augmentation

        **Authors**: *Nora Schneider, Shirin Goshtasbpour, Fernando Pérez-Cruz*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecc9b6dfdbe374c0a3364ff81cd28642-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecc9b6dfdbe374c0a3364ff81cd28642-Abstract-Conference.html)

        **Abstract**:

        We propose a novel algorithm for data augmentation in nonlinear over-parametrized regression. Our data augmentation algorithm borrows from the literature on causality. Contrary to the current state-of-the-art solutions that rely on modifications of Mixup algorithm, we extend the recently proposed distributionally robust Anchor regression (AR) method for data augmentation. Our Anchor Data Augmentation (ADA) uses several replicas of the modified samples in AR to provide more training examples, leading to more robust regression predictions. We apply ADA to linear and nonlinear regression problems using neural networks. ADA is competitive with state-of-the-art C-Mixup solutions.

        ----

        ## [3272] The noise level in linear regression with dependent data

        **Authors**: *Ingvar M. Ziemann, Stephen Tu, George J. Pappas, Nikolai Matni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ecffd829f90b0a4b6aa017b6df15904f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ecffd829f90b0a4b6aa017b6df15904f-Abstract-Conference.html)

        **Abstract**:

        We derive upper bounds for random design linear regression with dependent ($\beta$-mixing) data absent any realizability assumptions.  In contrast to the strictly realizable martingale noise regime, no sharp \emph{instance-optimal} non-asymptotics are available in the literature. Up to constant factors, our analysis correctly recovers the variance term predicted by the Central Limit Theorem---the noise level of the problem---and thus exhibits graceful degradation as we introduce misspecification. Past a burn-in, our result is sharp in the moderate deviations regime, and in particular does not inflate the leading order term by mixing time factors.

        ----

        ## [3273] SafeDICE: Offline Safe Imitation Learning with Non-Preferred Demonstrations

        **Authors**: *Youngsoo Jang, Geon-Hyeong Kim, Jongmin Lee, Sungryull Sohn, Byoungjip Kim, Honglak Lee, Moontae Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed2fb79f2664c3d9ba878be7e575b2af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed2fb79f2664c3d9ba878be7e575b2af-Abstract-Conference.html)

        **Abstract**:

        We consider offline safe imitation learning (IL), where the agent aims to learn the safe policy that mimics preferred behavior while avoiding non-preferred behavior from non-preferred demonstrations and unlabeled demonstrations. This problem setting corresponds to various real-world scenarios, where satisfying safety constraints is more important than maximizing the expected return. However, it is very challenging to learn the policy to avoid constraint-violating (i.e. non-preferred) behavior, as opposed to standard imitation learning which learns the policy to mimic given demonstrations. In this paper, we present a hyperparameter-free offline safe IL algorithm, SafeDICE, that learns safe policy by leveraging the non-preferred demonstrations in the space of stationary distributions. Our algorithm directly estimates the stationary distribution corrections of the policy that imitate the demonstrations excluding the non-preferred behavior. In the experiments, we demonstrate that our algorithm learns a more safe policy that satisfies the cost constraint without degrading the reward performance, compared to baseline algorithms.

        ----

        ## [3274] Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

        **Authors**: *Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html)

        **Abstract**:

        Large Language Models (LLMs) can achieve strong performance on many tasks by producing step-by-step reasoning before giving a final output, often referred to as chain-of-thought reasoning (CoT). It is tempting to interpret these CoT explanations as the LLM's process for solving a task. This level of transparency into LLMs' predictions would yield significant safety benefits. However, we find that CoT explanations can systematically misrepresent the true reason for a model's prediction. We demonstrate that CoT explanations can be heavily influenced by adding biasing features to model inputs—e.g., by reordering the multiple-choice options in a few-shot prompt to make the answer always "(A)"—which models systematically fail to mention in their explanations. When we bias models toward incorrect answers, they frequently generate CoT explanations rationalizing those answers. This causes accuracy to drop by as much as 36% on a suite of 13 tasks from BIG-Bench Hard, when testing with GPT-3.5 from OpenAI and Claude 1.0 from Anthropic. On a social-bias task, model explanations justify giving answers in line with stereotypes without mentioning the influence of these social biases. Our findings indicate that CoT explanations can be plausible yet misleading, which risks increasing our trust in LLMs without guaranteeing their safety. Building more transparent and explainable systems will require either improving CoT faithfulness through targeted efforts or abandoning CoT in favor of alternative methods.

        ----

        ## [3275] Static and Sequential Malicious Attacks in the Context of Selective Forgetting

        **Authors**: *Chenxu Zhao, Wei Qian, Rex Ying, Mengdi Huai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed4bacc8c7ca1ee0e1d4e0ef376b7ac7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed4bacc8c7ca1ee0e1d4e0ef376b7ac7-Abstract-Conference.html)

        **Abstract**:

        With the growing demand for the right to be forgotten, there is an increasing need for machine learning models to forget sensitive data and its impact. To address this, the paradigm of selective forgetting (a.k.a machine unlearning) has been extensively studied, which aims to remove the impact of requested data from a well-trained model without retraining from scratch. Despite its significant success, limited attention has been given to the security vulnerabilities of the unlearning system concerning malicious data update requests. Motivated by this, in this paper, we explore the possibility and feasibility of malicious data update requests during the unlearning process. Specifically, we first propose a new class of malicious selective forgetting attacks, which involves a static scenario where all the malicious data update requests are provided by the adversary at once. Additionally, considering the sequential setting where the data update requests arrive sequentially, we also design a novel framework for sequential forgetting attacks, which is formulated as a stochastic optimal control problem. We also propose novel optimization algorithms that can find the effective malicious data update requests. We perform theoretical analyses for the proposed selective forgetting attacks, and extensive experimental results validate the effectiveness of our proposed selective forgetting attacks. The source code is available in the supplementary material.

        ----

        ## [3276] Language-based Action Concept Spaces Improve Video Self-Supervised Learning

        **Authors**: *Kanchana Ranasinghe, Michael S. Ryoo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed67dff7cb96e7e86c4d91c0d5db49bb-Abstract-Conference.html)

        **Abstract**:

        Recent contrastive language image pre-training has led to learning highly transferable and robust image representations. However, adapting these models to video domain with minimal supervision remains an open problem. We explore a simple step in that direction, using language tied self-supervised learning to adapt an image CLIP model to the video domain. A backbone modified for temporal modeling is trained under self-distillation settings with train objectives operating in an action concept space. Feature vectors of various action concepts extracted from a language encoder using relevant textual prompts construct this space. A large language model aware of actions and their attributes generates the relevant textual prompts.We introduce two train objectives, concept distillation and concept alignment, that retain generality of original representations while enforcing relations between actions and their attributes. Our approach improves zero-shot and linear probing performance on three action recognition benchmarks.

        ----

        ## [3277] TRIAGE: Characterizing and auditing training data for improved regression

        **Authors**: *Nabeel Seedat, Jonathan Crabbé, Zhaozhi Qian, Mihaela van der Schaar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed687a5f52b651b19e7c18f702907b8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed687a5f52b651b19e7c18f702907b8b-Abstract-Conference.html)

        **Abstract**:

        Data quality is crucial for robust machine learning algorithms, with the recent interest in data-centric AI emphasizing the importance of training data characterization. However, current data characterization methods are largely focused on classification settings, with regression settings largely understudied. To address this, we introduce TRIAGE, a novel data characterization framework tailored to regression tasks and compatible with a broad class of regressors. TRIAGE utilizes conformal predictive distributions to provide a model-agnostic scoring method, the TRIAGE score. We operationalize the score to analyze individual samples' training dynamics and characterize samples as under-, over-, or well-estimated by the model. We show that TRIAGE's characterization is consistent and highlight its utility to improve performance via data sculpting/filtering, in multiple regression settings. Additionally, beyond sample level, we show TRIAGE enables new approaches to dataset selection and feature acquisition. Overall, TRIAGE highlights the value unlocked by data characterization in real-world regression applications.

        ----

        ## [3278] ClimateLearn: Benchmarking Machine Learning for Weather and Climate Modeling

        **Authors**: *Tung Nguyen, Jason Jewik, Hritik Bansal, Prakhar Sharma, Aditya Grover*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ed73c36e771881b232ef35fa3a1dec14-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ed73c36e771881b232ef35fa3a1dec14-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Modeling weather and climate is an essential endeavor to understand the near- and long-term impacts of climate change, as well as to inform technology and policymaking for adaptation and mitigation efforts. In recent years, there has been a surging interest in applying data-driven methods based on machine learning for solving core problems such as weather forecasting and climate downscaling. Despite promising results, much of this progress has been impaired due to the lack of large-scale, open-source efforts for reproducibility, resulting in the use of inconsistent or underspecified datasets, training setups, and evaluations by both domain scientists and artificial intelligence researchers. We introduce ClimateLearn, an open-source PyTorch library that vastly simplifies the training and evaluation of machine learning models for data-driven climate science. ClimateLearn consists of holistic pipelines for dataset processing (e.g., ERA5, CMIP6, PRISM), implementing state-of-the-art deep learning models (e.g., Transformers, ResNets), and quantitative and qualitative evaluation for standard weather and climate modeling tasks. We supplement these functionalities with extensive documentation, contribution guides, and quickstart tutorials to expand access and promote community growth. We have also performed comprehensive forecasting and downscaling experiments to showcase the capabilities and key features of our library. To our knowledge, ClimateLearn is the first large-scale, open-source effort for bridging research in weather and climate modeling with modern machine learning systems. Our library is available publicly at https://github.com/aditya-grover/climate-learn.

        ----

        ## [3279] Advice Querying under Budget Constraint for Online Algorithms

        **Authors**: *Ziyad Benomar, Vianney Perchet*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eda830e16044587b5082a853c4f25a90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eda830e16044587b5082a853c4f25a90-Abstract-Conference.html)

        **Abstract**:

        Several problems have been extensively studied in the learning-augmented setting, where the algorithm has access to some, possibly incorrect, predictions. However, it is assumed in most works that the predictions are provided to the algorithm as input, with no constraint on their size. In this paper, we consider algorithms with access to a limited number of predictions, that they can request at any time during their execution. We study three classical problems in competitive analysis, the ski rental problem, the secretary problem, and the non-clairvoyant job scheduling. We address the question of when to query predictions and how to use them.

        ----

        ## [3280] DIFFER: Decomposing Individual Reward for Fair Experience Replay in Multi-Agent Reinforcement Learning

        **Authors**: *Xunhan Hu, Jian Zhao, Wengang Zhou, Ruili Feng, Houqiang Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html)

        **Abstract**:

        Cooperative multi-agent reinforcement learning (MARL) is a challenging task, as agents must learn complex and diverse individual strategies from a shared team reward. However, existing methods struggle to distinguish and exploit important individual experiences, as they lack an effective way to decompose the team reward into individual rewards. To address this challenge, we propose DIFFER, a powerful theoretical framework for decomposing individual rewards to enable fair experience replay in MARL.By enforcing the invariance of network gradients, we establish a partial differential equation whose solution yields the underlying individual reward function. The individual TD-error can then be computed from the solved closed-form individual rewards, indicating the importance of each piece of experience in the learning task and guiding the training process. Our method elegantly achieves an equivalence to the original learning framework when individual experiences are homogeneous, while also adapting to achieve more muscular efficiency and fairness when diversity is observed.Our extensive experiments on popular benchmarks validate the effectiveness of our theory and method, demonstrating significant improvements in learning efficiency and fairness. Code is available in supplement material.

        ----

        ## [3281] Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing

        **Authors**: *Yelysei Bondarenko, Markus Nagel, Tijmen Blankevoort*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edbcb7583fd8921dad78adecfe06a99b-Abstract-Conference.html)

        **Abstract**:

        Transformer models have been widely adopted in various domains over the last years and especially large language models have advanced the field of AI significantly. Due to their size, the capability of these networks has increased tremendously, but this has come at the cost of a significant increase in necessary compute. Quantization is one of the most effective ways for reducing the computational time and memory consumption of neural networks. Many studies have shown, however, that modern transformer models tend to learn strong outliers in their activations, making them difficult to quantize. To retain acceptable performance, the existence of these outliers requires activations to be in higher-bitwidth or the use of different numeric formats, extra fine-tuning, or other workarounds. We show that strong outliers are related to very specific behavior of attention heads that try to learn a "no-op", or just a partial update of the residual. To achieve the exact zeros needed in the attention matrix for a no-update, the input to the softmax is pushed to be larger and larger during training, causing outliers in other parts of the network. Based on these observations, we propose two simple (independent) modifications to the attention mechanism - clipped softmax and gated attention. We empirically show that models pre-trained using our methods learn significantly smaller outliers while maintaining and sometimes even improving the floating-point task performance. This enables us to quantize transformers to full INT8 quantization of the activations without any additional effort. We demonstrate the effectiveness of our methods on both language models (BERT, OPT) and vision transformers.

        ----

        ## [3282] Adversarial Training from Mean Field Perspective

        **Authors**: *Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edcd1aa172dceda2ea9d45a48f25d3e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edcd1aa172dceda2ea9d45a48f25d3e3-Abstract-Conference.html)

        **Abstract**:

        Although adversarial training is known to be effective against adversarial examples, training dynamics are not well understood. In this study, we present the first theoretical analysis of adversarial training in random deep neural networks without any assumptions on data distributions. We introduce a new theoretical framework based on mean field theory, which addresses the limitations of existing mean field-based approaches. Based on the framework, we derive the (empirically tight) upper bounds of $\ell_q$ norm-based adversarial loss with $\ell_p$ norm-based adversarial examples for various values of $p$ and $q$. Moreover, we prove that networks without shortcuts are generally not adversarially trainable and that adversarial training reduces network capacity. We also show that the network width alleviates these issues. Furthermore, the various impacts of input and output dimensions on the upper bounds and time evolution of weight variance are presented.

        ----

        ## [3283] MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting

        **Authors**: *Felix Biggs, Antonin Schrab, Arthur Gretton*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edd00cead3425393baf13004de993017-Abstract-Conference.html)

        **Abstract**:

        We propose novel statistics which maximise the power  of a two-sample test based on the Maximum Mean Discrepancy (MMD), byadapting over the set of kernels used in defining it.For finite sets, this reduces to combining (normalised) MMD values under each of these kernels via a weighted soft maximum.Exponential concentration bounds are proved for our proposed statistics under the null and alternative.We further show how these kernels can be chosen in a data-dependent but permutation-independent way, in a well-calibrated test, avoiding data splitting.This technique applies more broadly to general permutation-based MMD testing, and includes the use of deep kernels with features learnt using unsupervised models such as auto-encoders.We highlight the applicability of our MMD-Fuse tests on both synthetic low-dimensional and real-world high-dimensional data, and compare its performance in terms of power against current state-of-the-art kernel tests.

        ----

        ## [3284] DAC-DETR: Divide the Attention Layers and Conquer

        **Authors**: *Zhengdong Hu, Yifan Sun, Jingdong Wang, Yi Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/edd0d433f8a1a51aa11237a6543fc280-Abstract-Conference.html)

        **Abstract**:

        This paper reveals a characteristic of DEtection Transformer (DETR) that negatively impacts its training efficacy, i.e., the cross-attention and self-attention layers in DETR decoder have contrary impacts on the object queries (though both impacts are important). Specifically, we observe the cross-attention tends to gather multiple queries around the same object, while the self-attention disperses these queries far away. To improve the training efficacy, we propose a Divide-And-Conquer DETR (DAC-DETR) that divides the cross-attention out from this contrary for better conquering. During training, DAC-DETR employs an auxiliary decoder that focuses on learning the cross-attention layers. The auxiliary decoder, while sharing all the other parameters, has NO self-attention layers and employs one-to-many label assignment to improve the gathering effect. Experiments show that DAC-DETR brings remarkable improvement over popular DETRs. For example, under the 12 epochs training scheme on MS-COCO, DAC-DETR improves Deformable DETR (ResNet-50) by +3.4 AP and achieves 50.9 (ResNet-50) / 58.1 AP (Swin-Large) based on some popular methods (i.e., DINO and an IoU-related loss). Our code will be made available at https://github.com/huzhengdongcs/DAC-DETR.

        ----

        ## [3285] Streaming Algorithms and Lower Bounds for Estimating Correlation Clustering Cost

        **Authors**: *Sepehr Assadi, Vihan Shah, Chen Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1a1ecc92f35702b5c29dad3dc909ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1a1ecc92f35702b5c29dad3dc909ea-Abstract-Conference.html)

        **Abstract**:

        Correlation clustering is a fundamental optimization problem at the intersection of machine learning and theoretical computer science. Motivated by applications to big data processing, recent years have witnessed a flurry of results on this problem in the streaming model. In this model, the algorithm needs to process the input $n$-vertex graph by making one or few passes over the stream of its edges and using a limited memory, much smaller than the input size. All previous work on streaming correlation clustering have focused on semi-streaming algorithms with $\Omega(n)$ memory, whereas in this work, we study streaming algorithms with much smaller memory requirement of only $\text{polylog}{(n)}$ bits. This stringent memory requirement is in the same spirit of classical streaming algorithms that instead of recovering a full solution to the problem---which can be prohibitively large with such small memory as is the case in our problem---, aimed to learn certain statistical properties of their inputs. In our case, this translates to determining the ``(correlation) clusterability'' of input graphs, or more precisely, estimating the cost of the optimal correlation clustering solution. As our main result, we present two novel algorithms that in only $\text{polylog}{(n)}$ space are able to estimate the optimal correlation clustering cost up to some constant multiplicative factor plus some extra additive error. One of the algorithms outputs a $3$-multiplicative approximation plus $o(n^2)$ additive approximation, and the other one improves the additive error further down at the cost of increasing the multiplicative factor to some large constant. We then present new lower bounds that justify this mix of both multiplicative and additive error approximation in our algorithms.

        ----

        ## [3286] Episodic Multi-Task Learning with Heterogeneous Neural Processes

        **Authors**: *Jiayi Shen, Xiantong Zhen, Qi Wang, Marcel Worring*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1e549d6fb7c58ed06557bfc264335c-Abstract-Conference.html)

        **Abstract**:

        This paper focuses on the data-insufficiency problem in multi-task learning within an episodic training setup. Specifically, we explore the potential of heterogeneous information across tasks and meta-knowledge among episodes to effectively tackle each task with limited data. Existing meta-learning methods often fail to take advantage of crucial heterogeneous information in a single episode, while multi-task learning models neglect reusing experience from earlier episodes. To address the problem of insufficient data, we develop Heterogeneous Neural Processes (HNPs) for the episodic multi-task setup. Within the framework of hierarchical Bayes, HNPs effectively capitalize on prior experiences as meta-knowledge and capture task-relatedness among heterogeneous tasks, mitigating data-insufficiency. Meanwhile, transformer-structured inference modules are designed to enable efficient inferences toward meta-knowledge and task-relatedness. In this way, HNPs can learn more powerful functional priors for adapting to novel heterogeneous tasks in each meta-test episode. Experimental results show the superior performance of the proposed HNPs over typical baselines, and ablation studies verify the effectiveness of the designed inference modules.

        ----

        ## [3287] Knowledge Distillation Performs Partial Variance Reduction

        **Authors**: *Mher Safaryan, Alexandra Peste, Dan Alistarh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation is a popular approach for enhancing the performance of "student" models, with lower representational capacity, by taking advantage of more powerful "teacher" models. Despite its apparent simplicity, the underlying mechanics behind knowledge distillation (KD) are not yet fully understood. In this work, we shed new light on the inner workings of this method, by examining it from an optimization perspective. Specifically, we show that, in the context of linear and deep linear models, KD can be interpreted as a novel type of stochastic variance reduction mechanism. We provide a detailed convergence analysis of the resulting dynamics, which hold under standard assumptions for both strongly-convex and non-convex losses, showing that KD acts as a form of \emph{partial variance reduction}, which can reduce the stochastic gradient noise, but may not eliminate it completely, depending on the properties of the ``teacher'' model. Our analysis puts further emphasis on the need for careful parametrization of KD, in particular w.r.t. the weighting of the distillation loss, and is validated empirically on both linear models and deep neural networks.

        ----

        ## [3288] Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels

        **Authors**: *Zifu Wang, Xuefei Ning, Matthew B. Blaschko*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html)

        **Abstract**:

        Intersection over Union (IoU) losses are surrogates that directly optimize the Jaccard index. Leveraging IoU losses as part of the loss function have demonstrated superior performance in semantic segmentation tasks compared to optimizing pixel-wise losses such as the cross-entropy loss alone. However, we identify a lack of flexibility in these losses to support vital training techniques like label smoothing, knowledge distillation, and semi-supervised learning, mainly due to their inability to process soft labels. To address this, we introduce Jaccard Metric Losses (JMLs), which are identical to the soft Jaccard loss in standard settings with hard labels but are fully compatible with soft labels. We apply JMLs to three prominent use cases of soft labels: label smoothing, knowledge distillation and semi-supervised learning, and demonstrate their potential to enhance model accuracy and calibration. Our experiments show consistent improvements over the cross-entropy loss across 4 semantic segmentation datasets (Cityscapes, PASCAL VOC, ADE20K, DeepGlobe Land) and 13 architectures, including classic CNNs and recent vision transformers. Remarkably, our straightforward approach significantly outperforms state-of-the-art knowledge distillation and semi-supervised learning methods. The code is available at \href{https://github.com/zifuwanggg/JDTLosses}{https://github.com/zifuwanggg/JDTLosses}.

        ----

        ## [3289] Towards Stable Backdoor Purification through Feature Shift Tuning

        **Authors**: *Rui Min, Zeyu Qin, Li Shen, Minhao Cheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee37d51b3c003d89acba2363dde256af-Abstract-Conference.html)

        **Abstract**:

        It has been widely observed that deep neural networks (DNN) are vulnerable to backdoor attacks where attackers could manipulate the model behavior maliciously by tampering with a small set of training samples. Although a line of defense methods is proposed to mitigate this threat, they either require complicated modifications to the training process or heavily rely on the specific model architecture, which makes them hard to deploy into real-world applications. Therefore, in this paper, we instead start with fine-tuning, one of the most common and easy-to-deploy backdoor defenses, through comprehensive evaluations against diverse attack scenarios. Observations made through initial experiments show that in contrast to the promising defensive results on high poisoning rates, vanilla tuning methods completely fail at low poisoning rate scenarios. Our analysis shows that with the low poisoning rate, the entanglement between backdoor and clean features undermines the effect of tuning-based defenses. Therefore, it is necessary to disentangle the backdoor and clean features in order to improve backdoor purification. To address this, we introduce Feature Shift Tuning (FST), a method for tuning-based backdoor purification. Specifically, FST encourages feature shifts by actively deviating the classifier weights from the originally compromised weights. Extensive experiments demonstrate that our FST provides consistently stable performance under different attack settings. Without complex parameter adjustments, FST also achieves much lower tuning costs, only $10$ epochs. Our codes are available at https://github.com/AISafety-HKUST/stable_backdoor_purification.

        ----

        ## [3290] Scalable 3D Captioning with Pretrained Models

        **Authors**: *Tiange Luo, Chris Rockwell, Honglak Lee, Justin Johnson*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee4814f9bce0cae7991d3341bb081b55-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee4814f9bce0cae7991d3341bb081b55-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce Cap3D, an automatic approach for generating descriptive text for 3D objects. This approach utilizes pretrained models from image captioning, image-text alignment, and LLM to consolidate captions from multiple views of a 3D asset, completely side-stepping the time-consuming and costly process of manual annotation. We apply Cap3D to the recently introduced large-scale 3D dataset, Objaverse, resulting in 660k 3D-text pairs. Our evaluation, conducted using 41k human annotations from the same dataset, demonstrates that Cap3D surpasses human-authored descriptions in terms of quality, cost, and speed. Through effective prompt engineering, Cap3D rivals human performance in generating geometric descriptions on 17k collected annotations from the ABO dataset. Finally, we finetune Text-to-3D models on Cap3D and human captions, and show Cap3D outperforms; and benchmark the SOTA including Point·E, Shape·E, and DreamFusion.

        ----

        ## [3291] Langevin Quasi-Monte Carlo

        **Authors**: *Sifan Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee56aa4fe26a189782f507d843fd5272-Abstract-Conference.html)

        **Abstract**:

        Langevin Monte Carlo (LMC) and its stochastic gradient versions are powerful algorithms for sampling from complex high-dimensional distributions. To sample from a distribution with density $\pi(\theta)\propto \exp(-U(\theta)) $, LMC iteratively generates the next sample by taking a step in the gradient direction $\nabla U$ with added Gaussian perturbations. Expectations w.r.t. the target distribution $\pi$ are estimated by averaging over LMC samples. In ordinary Monte Carlo, it is well known that the estimation error can be substantially reduced by replacing independent random samples by quasi-random samples like low-discrepancy sequences. In this work, we show that the estimation error of LMC can also be reduced by using quasi-random samples. Specifically, we propose to use completely uniformly distributed (CUD) sequences with certain low-discrepancy property to generate the Gaussian perturbations. Under smoothness and convexity conditions, we prove that LMC with a low-discrepancy CUD sequence achieves smaller error than standard LMC. The theoretical analysis is supported by compelling numerical experiments, which demonstrate the effectiveness of our approach.

        ----

        ## [3292] LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting

        **Authors**: *Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, Roger Zimmermann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee57cd73a76bd927ffca3dda1dc3b9d4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Road traffic forecasting plays a critical role in smart city initiatives and has experienced significant advancements thanks to the power of deep learning in capturing non-linear patterns of traffic data. However, the promising results achieved on current public datasets may not be applicable to practical scenarios due to limitations within these datasets. First, the limited sizes of them may not reflect the real-world scale of traffic networks. Second, the temporal coverage of these datasets is typically short, posing hurdles in studying long-term patterns and acquiring sufficient samples for training deep models. Third, these datasets often lack adequate metadata for sensors, which compromises the reliability and interpretability of the data. To mitigate these limitations, we introduce the LargeST benchmark dataset. It encompasses a total number of 8,600 sensors in California with a 5-year time coverage and includes comprehensive metadata. Using LargeST, we perform in-depth data analysis to extract data insights, benchmark well-known baselines in terms of their performance and efficiency, and identify challenges as well as opportunities for future research. We release the datasets and baseline implementations at: https://github.com/liuxu77/LargeST.

        ----

        ## [3293] What Can We Learn from Unlearnable Datasets?

        **Authors**: *Pedro Sandoval Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee5bb72130c332c3d4bf8d231e617506-Abstract-Conference.html)

        **Abstract**:

        In an era of widespread web scraping, unlearnable dataset methods have the potential to protect data privacy by preventing deep neural networks from generalizing. But in addition to a number of practical limitations that make their use unlikely, we make a number of findings that call into question their ability to safeguard data. First, it is widely believed that neural networks trained on unlearnable datasets only learn shortcuts, simpler rules that are not useful for generalization. In contrast, we find that networks actually can learn useful features that can be reweighed for high test performance, suggesting that image protection is not assured. Unlearnable datasets are also believed to induce learning shortcuts through linear separability of added perturbations. We provide a counterexample, demonstrating that linear separability of perturbations is not a necessary condition. To emphasize why linearly separable perturbations should not be relied upon, we propose an orthogonal projection attack which allows learning from unlearnable datasets published in ICML 2021 and ICLR 2023. Our proposed attack is significantly less complex than recently proposed techniques.

        ----

        ## [3294] Language Models Meet World Models: Embodied Experiences Enhance Language Models

        **Authors**: *Jiannan Xiang, Tianhua Tao, Yi Gu, Tianmin Shu, Zirui Wang, Zichao Yang, Zhiting Hu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee6630dcbcff857026e474fc857aa9f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee6630dcbcff857026e474fc857aa9f0-Abstract-Conference.html)

        **Abstract**:

        While large language models (LMs) have shown remarkable capabilities across numerous tasks, they often struggle with simple reasoning and planning in physical environments, such as understanding object permanence or planning household activities. The limitation arises from the fact that LMs are trained only on written text and miss essential embodied knowledge and skills. In this paper, we propose a new paradigm of enhancing LMs by finetuning them with world models, to gain diverse embodied knowledge while retaining their general language capabilities. Our approach deploys an embodied agent in a world model, particularly a simulator of the physical world (VirtualHome), and acquires a diverse set of embodied experiences through both goal-oriented planning and random exploration. These experiences are then used to finetune LMs to teach diverse abilities of reasoning and acting in the physical world, e.g., planning and completing goals, object permanence and tracking, etc. Moreover, it is desirable to preserve the generality of LMs during finetuning, which facilitates generalizing the embodied knowledge across tasks rather than being tied to specific simulations. We thus further introduce the classical elastic weight consolidation (EWC) for selective weight updates, combined with low-rank adapters (LoRA) for training efficiency. Extensive experiments show our approach substantially improves base LMs on 18 downstream tasks by 64.28% on average. In particular, the small LMs (1.3B, 6B, and 13B) enhanced by our approach match or even outperform much larger LMs (e.g., ChatGPT).

        ----

        ## [3295] Activity Grammars for Temporal Action Segmentation

        **Authors**: *Dayoung Gong, Joonseok Lee, Deunsol Jung, Suha Kwak, Minsu Cho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee6c4b99b4c0d3d60efd22c1ecdd9891-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee6c4b99b4c0d3d60efd22c1ecdd9891-Abstract-Conference.html)

        **Abstract**:

        Sequence prediction on temporal data requires the ability to understand compositional structures of multi-level semantics beyond individual and contextual properties of parts. The task of temporal action segmentation remains challenging for the reason, aiming at translating an untrimmed activity video into a sequence of action segments. This paper addresses the problem by introducing an effective activity grammar to guide neural predictions for temporal action segmentation.  We propose a novel grammar induction algorithm, dubbed KARI, that extracts a powerful context-free grammar from action sequence data. We also develop an efficient generalized parser, dubbed BEP, that transforms frame-level probability distributions into a reliable sequence of actions according to the induced grammar with recursive rules. Our approach can be combined with any neural network for temporal action segmentation to enhance the sequence prediction and discover its compositional structure. Experimental results demonstrate that our method significantly improves temporal action segmentation in terms of both performance and interpretability on two standard benchmarks, Breakfast and 50 Salads.

        ----

        ## [3296] SANFlow: Semantic-Aware Normalizing Flow for Anomaly Detection

        **Authors**: *Daehyun Kim, Sungyong Baik, Tae Hyun Kim*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee74a6ade401e200985e2421b20bbae4-Abstract-Conference.html)

        **Abstract**:

        Visual anomaly detection, the task of detecting abnormal characteristics in images, is challenging due to the rarity and unpredictability of anomalies. In order to reliably model the distribution of normality and detect anomalies, a few works have attempted to exploit the density estimation ability of normalizing flow (NF). However, previous NF-based methods have relied solely on the capability of NF and forcibly transformed the distribution of all features to a single distribution (e.g., unit normal distribution), when features can have different semantic information and thus follow different distributions. We claim that forcibly learning to transform such diverse distributions to a single distribution with a single network will cause the learning difficulty, limiting the capacity of a network to discriminate normal and abnormal data. As such, we propose to transform the distribution of features at each location of a given image to different distributions. In particular, we train NF to map normal data distribution to distributions with the same mean but different variances at each location of the given image. To enhance the discriminability, we also train NF to map abnormal data distribution to a distribution with a mean that is different from that of normal data, where abnormal data is synthesized with data augmentation. The experimental results outline the effectiveness of the proposed framework in improving the density modeling and thus anomaly detection performance.

        ----

        ## [3297] AirDelhi: Fine-Grained Spatio-Temporal Particulate Matter Dataset From Delhi For ML based Modeling

        **Authors**: *Sachin Chauhan, Zeel Bharatkumar Patel, Sayan Ranu, Rijurekha Sen, Nipun Batra*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee799aff607fcf39c01df6391e96f92c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee799aff607fcf39c01df6391e96f92c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Air pollution poses serious health concerns  in developing countries, such as India, necessitating large-scale measurement for correlation analysis, policy recommendations, and informed decision-making. However, fine-grained data collection is costly.  Specifically, static sensors for pollution measurement cost several thousand dollars per unit, leading to inadequate deployment and coverage. To complement the existing sparse static sensor network, we propose a mobile sensor network utilizing lower-cost PM2.5 sensors mounted on public buses in the Delhi-NCR region of India. Through this exercise, we introduce a novel dataset AirDelhi comprising PM2.5 and PM10 measurements. This dataset is made publicly available, at https://www.cse.iitd.ac.in/pollutiondata, serving as a valuable resource for machine learning (ML) researchers and environmentalists. We present three key contributions with the release of this dataset. Firstly, through in-depth statistical analysis, we demonstrate that the released dataset significantly differs from existing pollution datasets, highlighting its uniqueness and potential for new insights. Secondly, the dataset quality been validated against existing expensive sensors. Thirdly, we conduct a benchmarking exercise (https://github.com/sachin-iitd/DelhiPMDatasetBenchmark), evaluating state-of-the-art methods for interpolation, feature imputation, and forecasting on this dataset, which is the largest publicly available PM dataset to date. The results of the benchmarking exercise underscore the substantial disparities in accuracy between the proposed dataset and other publicly available datasets. This finding highlights the complexity and richness of our dataset, emphasizing its value for advancing research in the field of air pollution.

        ----

        ## [3298] On Convergence of Polynomial Approximations to the Gaussian Mixture Entropy

        **Authors**: *Caleb Dahlke, Jason Pacheco*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee860a9fa65a55a335754c557a5211de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee860a9fa65a55a335754c557a5211de-Abstract-Conference.html)

        **Abstract**:

        Gaussian mixture models (GMMs) are fundamental to machine learning due to their flexibility as approximating densities. However, uncertainty quantification of GMMs remains a challenge as differential entropy lacks a closed form.  This paper explores polynomial approximations, specifically Taylor and Legendre, to the GMM entropy from a theoretical and practical perspective. We provide new analysis of a widely used approach due to Huber et al.(2008) and show that the series diverges under simple conditions.  Motivated by this divergence we provide a novel Taylor series that is provably convergent to the true entropy of any GMM.  We demonstrate a method for selecting a center such that the series converges from below, providing a lower bound on GMM entropy. Furthermore, we demonstrate that orthogonal polynomial series result in more accurate polynomial approximations. Experimental validation supports our theoretical results while showing that our method is comparable in computation to Huber et al. We also show that in application, the use of these polynomial approximations, such as in Nonparametric Variational Inference by Gershamn et al. (2012), rely on the convergence of the methods in computing accurate approximations. This work contributes useful analysis to existing methods while introducing a novel approximation supported by firm theoretical guarantees.

        ----

        ## [3299] CEIL: Generalized Contextual Imitation Learning

        **Authors**: *Jinxin Liu, Li He, Yachen Kang, Zifeng Zhuang, Donglin Wang, Huazhe Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ee90fb9511b263f2ff971be9b374f9ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ee90fb9511b263f2ff971be9b374f9ee-Abstract-Conference.html)

        **Abstract**:

        In this paper, we present ContExtual Imitation Learning (CEIL), a general and broadly applicable algorithm for imitation learning (IL). Inspired by the formulation of hindsight information matching, we derive CEIL by explicitly learning a hindsight embedding function together with a contextual policy using the hindsight embeddings. To achieve the expert matching objective for IL, we advocate for optimizing a contextual variable such that it biases the contextual policy towards mimicking expert behaviors. Beyond the typical learning from demonstrations (LfD) setting, CEIL is a generalist that can be effectively applied to multiple settings including: 1) learning from observations (LfO), 2) offline IL, 3) cross-domain IL (mismatched experts), and 4) one-shot IL settings. Empirically, we evaluate CEIL on the popular MuJoCo tasks (online) and the D4RL dataset (offline). Compared to prior state-of-the-art baselines, we show that CEIL is more sample-efficient in most online IL tasks and achieves better or competitive performances in offline tasks.

        ----

        ## [3300] Entropic Neural Optimal Transport via Diffusion Processes

        **Authors**: *Nikita Gushchin, Alexander Kolesov, Alexander Korotin, Dmitry P. Vetrov, Evgeny Burnaev*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeac51414a11484d048432f614d5bb1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeac51414a11484d048432f614d5bb1b-Abstract-Conference.html)

        **Abstract**:

        We propose a novel neural algorithm for the fundamental problem of computing the entropic optimal transport (EOT) plan between probability distributions which are accessible by samples. Our algorithm is based on the saddle point reformulation of the dynamic version of EOT which is known as the Schr√∂dinger Bridge problem. In contrast to the prior methods for large-scale EOT, our algorithm is end-to-end and consists of a single learning step, has fast inference procedure, and allows handling small values of the entropy regularization coefficient which is of particular importance in some applied problems. Empirically, we show the performance of the method on several large-scale EOT tasks. The code for the ENOT solver can be found at https://github.com/ngushchin/EntropicNeuralOptimalTransport

        ----

        ## [3301] On the Convergence to a Global Solution of Shuffling-Type Gradient Algorithms

        **Authors**: *Lam M. Nguyen, Trang H. Tran*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeb57fdf745eb31a3c7ef22c59a4661d-Abstract-Conference.html)

        **Abstract**:

        Stochastic gradient descent (SGD) algorithm is the method of choice in many machine learning tasks thanks to its scalability and efficiency in dealing with large-scale problems. In this paper, we focus on the shuffling version of SGD which matches the mainstream practical heuristics. We show the convergence to a global solution of shuffling SGD for a class of non-convex functions under over-parameterized settings. Our analysis employs more relaxed non-convex assumptions than previous literature. Nevertheless, we maintain the desired computational complexity as shuffling SGD has achieved in the general convex setting.

        ----

        ## [3302] Evaluating Robustness and Uncertainty of Graph Models Under Structural Distributional Shifts

        **Authors**: *Gleb Bazhenov, Denis Kuznedelev, Andrey Malinin, Artem Babenko, Liudmila Prokhorenkova*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eec7fee9a8595ca964b9a11562767345-Abstract-Conference.html)

        **Abstract**:

        In reliable decision-making systems based on machine learning, models have to be robust to distributional shifts or provide the uncertainty of their predictions. In node-level problems of graph learning, distributional shifts can be especially complex since the samples are interdependent. To evaluate the performance of graph models, it is important to test them on diverse and meaningful distributional shifts. However, most graph benchmarks considering distributional shifts for node-level problems focus mainly on node features, while structural properties are also essential for graph problems. In this work, we propose a general approach for inducing diverse distributional shifts based on graph structure. We use this approach to create data splits according to several structural node properties: popularity, locality, and density. In our experiments, we thoroughly evaluate the proposed distributional shifts and show that they can be quite challenging for existing graph models. We also reveal that simple models often outperform more sophisticated methods on the considered structural shifts. Finally, our experiments provide evidence that there is a trade-off between the quality of learned representations for the base classification task under structural distributional shift and the ability to separate the nodes from different distributions using these representations.

        ----

        ## [3303] Learning Causal Models under Independent Changes

        **Authors**: *Sarah Mameche, David Kaltenpoth, Jilles Vreeken*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eee6efe709623f36483e3fbb0bb513dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eee6efe709623f36483e3fbb0bb513dd-Abstract-Conference.html)

        **Abstract**:

        In many scientific applications, we observe a system in different conditions in which its components may change, rather than in isolation. In our work, we are interested in explaining the generating process of such a multi-context system using a finite mixture of causal mechanisms. Recent work shows that this causal model is identifiable from data, but is limited to settings where the sparse mechanism shift hypothesis holds and only a subset of the causal conditionals change. As this assumption is not easily verifiable in practice, we study the more general principle that mechanism shifts are independent, which we formalize using the algorithmic notion of independence. We introduce an approach for causal discovery beyond partially directed graphs using Gaussian Process models, and give conditions under which we provably identify the correct causal model. In our experiments, we show that our method performs well in a range of synthetic settings, on realistic gene expression simulations, as well as on real-world cell signaling data.

        ----

        ## [3304] Universality and Limitations of Prompt Tuning

        **Authors**: *Yihan Wang, Jatin Chauhan, Wei Wang, Cho-Jui Hsieh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eef6aecfe050b556c6a48d9c16b15558-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eef6aecfe050b556c6a48d9c16b15558-Abstract-Conference.html)

        **Abstract**:

        Despite the demonstrated empirical efficacy of prompt tuning to adapt a pretrained language model for a new task, the theoretical underpinnings of the difference between "tuning parameters before the input" against "the tuning of model weights" are limited. We thus take one of the first steps to understand the role of soft-prompt tuning for transformer-based architectures. By considering a general purpose architecture, we analyze prompt tuning from the lens of both: universal approximation and limitations with finite-depth fixed-weight pretrained transformers for continuous-valued functions. Our universality result guarantees the existence of a strong transformer with a prompt to approximate any sequence-to-sequence function in the set of Lipschitz functions. The limitations of prompt tuning for limited-depth transformers are first proved by constructing a set of datasets, that cannot be memorized by a prompt of any length for a given single encoder layer. We also provide a lower bound on the required number of tunable prompt parameters and compare the result with the number of parameters required for a low-rank update (based on LoRA) for a single-layer setting. We finally extend our analysis to multi-layer settings by providing sufficient conditions under which the transformer can at best learn datasets from invertible functions only. Our theoretical claims are also corroborated by empirical results.

        ----

        ## [3305] Evaluating Neuron Interpretation Methods of NLP Models

        **Authors**: *Yimin Fan, Fahim Dalvi, Nadir Durrani, Hassan Sajjad*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eef6cb60fd59b32d35718e176b4b08d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eef6cb60fd59b32d35718e176b4b08d6-Abstract-Conference.html)

        **Abstract**:

        Neuron interpretation offers valuable insights into how knowledge is structured within a deep neural network model. While a number of neuron interpretation methods have been proposed in the literature, the field lacks a comprehensive comparison among these methods. This gap hampers progress due to the absence of standardized metrics and benchmarks. The commonly used evaluation metric has limitations, and creating ground truth annotations for neurons is impractical. Addressing these challenges, we propose an evaluation framework based on voting theory. Our hypothesis posits that neurons consistently identified by different methods carry more significant information. We rigorously assess our framework across a diverse array of neuron interpretation methods. Notable findings include: i) despite the theoretical differences among the methods, neuron ranking methods share over 60% of their rankings when identifying salient neurons, ii) the neuron interpretation methods are most sensitive to the last layer representations, iii) Probeless neuron ranking emerges as the most consistent method.

        ----

        ## [3306] How Re-sampling Helps for Long-Tail Learning?

        **Authors**: *Jiang-Xin Shi, Tong Wei, Yuke Xiang, Yu-Feng Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/eeffa70bcbbd43f6bd067edebc6595e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/eeffa70bcbbd43f6bd067edebc6595e8-Abstract-Conference.html)

        **Abstract**:

        Long-tail learning has received significant attention in recent years due to the challenge it poses with extremely imbalanced datasets. In these datasets, only a few classes (known as the head classes) have an adequate number of training samples, while the rest of the classes (known as the tail classes) are infrequent in the training data. Re-sampling is a classical and widely used approach for addressing class imbalance issues. Unfortunately, recent studies claim that re-sampling brings negligible performance improvements in modern long-tail learning tasks. This paper aims to investigate this phenomenon systematically. Our research shows that re-sampling can considerably improve generalization when the training images do not contain semantically irrelevant contexts. In other scenarios, however, it can learn unexpected spurious correlations between irrelevant contexts and target labels. We design experiments on two homogeneous datasets, one containing irrelevant context and the other not, to confirm our findings. To prevent the learning of spurious correlations, we propose a new context shift augmentation module that generates diverse training images for the tail class by maintaining a context bank extracted from the head-class images. Experiments demonstrate that our proposed module can boost the generalization and outperform other approaches, including class-balanced re-sampling, decoupled classifier re-training, and data augmentation methods. The source code is available at https://www.lamda.nju.edu.cn/code_CSA.ashx.

        ----

        ## [3307] FIND: A Function Description Benchmark for Evaluating Interpretability Methods

        **Authors**: *Sarah Schwettmann, Tamar Rott Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, Antonio Torralba*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0164c1112f56246224af540857348f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0164c1112f56246224af540857348f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Labeling neural network submodules with human-legible descriptions is useful for many downstream tasks: such descriptions can surface failures, guide interventions, and perhaps even explain important model behaviors. To date, most mechanistic descriptions of trained networks have involved small models, narrowly delimited phenomena, and large amounts of human labor. Labeling all human-interpretable sub-computations in models of increasing size and complexity will almost certainly require tools that can generate and validate descriptions automatically. Recently, techniques that use learned models in-the-loop for labeling have begun to gain traction, but methods for evaluating their efficacy are limited and ad-hoc. How should we validate and compare open-ended labeling tools? This paper introduces FIND (Function INterpretation and Description), a benchmark suite for evaluating the building blocks of automated interpretability methods. FIND contains functions that resemble components of trained neural networks, and accompanying descriptions of the kind we seek to generate. The functions are procedurally constructed across textual and numeric domains, and involve a range of real-world complexities, including noise, composition, approximation, and bias. We evaluate methods that use pretrained language models (LMs) to produce code-based and natural language descriptions of function behavior. Additionally, we introduce a new interactive method in which an Automated Interpretability Agent (AIA) generates function descriptions. We find that an AIA, built with an off-the-shelf LM augmented with black-box access to functions, can sometimes infer function structureâ€”acting as a scientist by forming hypotheses, proposing experiments, and updating descriptions in light of new data. However, FIND also reveals that LM-based descriptions capture global function behavior while missing local details. These results suggest that FIND will be useful for characterizing the performance of more sophisticated interpretability methods before they are applied to real-world models.

        ----

        ## [3308] EgoTracks: A Long-term Egocentric Visual Object Tracking Dataset

        **Authors**: *Hao Tang, Kevin J. Liang, Kristen Grauman, Matt Feiszli, Weiyao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef01d91aa87e7701aa9c8dc66a2d5bdb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef01d91aa87e7701aa9c8dc66a2d5bdb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Visual object tracking is a key component to many egocentric vision problems. However, the full spectrum of challenges of egocentric tracking faced by an embodied AI is underrepresented in many existing datasets; these tend to focus on relatively short, third-person videos. Egocentric video has several distinguishing characteristics from those commonly found in past datasets: frequent large camera motions and hand interactions with objects commonly lead to occlusions or objects exiting the frame, and object appearance can change rapidly due to widely different points of view, scale, or object states. Embodied tracking is also naturally long-term, and being able to consistently (re-)associate objects to their appearances and disappearances over as long as a lifetime is critical. Previous datasets under-emphasize this re-detection problem, and their "framed" nature has led to adoption of various spatiotemporal priors that we find do not necessarily generalize to egocentric video. We thus introduce EgoTracks, a new dataset for long-term egocentric visual object tracking. Sourced from the Ego4D dataset, this new dataset presents a significant challenge to recent state-of-the-art single-object tracking models, which we find score poorly on traditional tracking metrics for our new dataset, compared to popular benchmarks. We further show improvements that can be made to a STARK tracker to significantly increase its performance on egocentric data, resulting in a baseline model we call EgoSTARK. We publicly release our annotations and benchmark, hoping our dataset leads to further advancements in tracking.

        ----

        ## [3309] Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models

        **Authors**: *Andrew Luo, Maggie Henderson, Leila Wehbe, Michael J. Tarr*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html)

        **Abstract**:

        A long standing goal in neuroscience has been to elucidate the functional organization of the brain. Within higher visual cortex, functional accounts have remained relatively coarse, focusing on regions of interest (ROIs) and taking the form of selectivity for broad categories such as faces, places, bodies, food, or words. Because the identification of such ROIs has typically relied on manually assembled stimulus sets consisting of isolated objects in non-ecological contexts, exploring functional organization without robust a priori hypotheses has been challenging. To overcome these limitations, we introduce a data-driven approach in which we synthesize images predicted to activate a given brain region using paired natural images and fMRI recordings, bypassing the need for category-specific stimuli. Our approach -- Brain Diffusion for Visual Exploration ("BrainDiVE") -- builds on recent generative methods by combining large-scale diffusion models with brain-guided image synthesis. Validating our method, we demonstrate the ability to synthesize preferred images with appropriate semantic specificity for well-characterized category-selective ROIs. We then show that BrainDiVE can characterize differences between ROIs selective for the same high-level category. Finally we identify novel functional subdivisions within these ROIs, validated with behavioral data. These  results advance our understanding of the fine-grained functional organization of human visual cortex, and provide well-specified constraints for further examination of cortical organization using hypothesis-driven methods.

        ----

        ## [3310] Query-based Temporal Fusion with Explicit Motion for 3D Object Detection

        **Authors**: *Jinghua Hou, Zhe Liu, Dingkang Liang, Zhikang Zou, Xiaoqing Ye, Xiang Bai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef0dcb44a47185f5bacac62571f6e920-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef0dcb44a47185f5bacac62571f6e920-Abstract-Conference.html)

        **Abstract**:

        Effectively utilizing temporal information to improve 3D detection performance is vital for autonomous driving vehicles. Existing methods either conduct temporal fusion based on the dense BEV features or sparse 3D proposal features. However, the former does not pay more attention to foreground objects, leading to more computation costs and sub-optimal performance. The latter implements time-consuming operations to generate sparse 3D proposal features, and the performance is limited by the quality of 3D proposals. In this paper, we propose a simple and effective Query-based Temporal Fusion Network (QTNet). The main idea is to exploit the object queries in previous frames to enhance the representation of current object queries by the proposed Motion-guided Temporal Modeling (MTM) module, which utilizes the spatial position information of object queries along the temporal dimension to construct their relevance between adjacent frames reliably. Experimental results show our proposed QTNet outperforms BEV-based or proposal-based manners on the nuScenes dataset. Besides, the MTM is a plug-and-play module, which can be integrated into some advanced LiDAR-only or multi-modality 3D detectors and even brings new SOTA performance with negligible computation cost and latency on the nuScenes dataset. These experiments powerfully illustrate the superiority and generalization of our method. The code is available at https://github.com/AlmoonYsl/QTNet.

        ----

        ## [3311] Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection

        **Authors**: *Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan S. Kankanhalli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef4f2a0232a246b8a502135175e08953-Abstract-Conference.html)

        **Abstract**:

        Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/EfficientACLvia_RCS.

        ----

        ## [3312] A Finite-Sample Analysis of Payoff-Based Independent Learning in Zero-Sum Stochastic Games

        **Authors**: *Zaiwei Chen, Kaiqing Zhang, Eric Mazumdar, Asuman E. Ozdaglar, Adam Wierman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef62614753535977071395fb1f1435be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef62614753535977071395fb1f1435be-Abstract-Conference.html)

        **Abstract**:

        In this work, we study two-player zero-sum stochastic games and develop a variant of the smoothed best-response learning dynamics that combines independent learning dynamics for matrix games with the minimax value iteration for stochastic games. The resulting learning dynamics are payoff-based, convergent, rational, and symmetric between the two players. Our theoretical results present to the best of our knowledge the first last-iterate finite-sample analysis of such independent learning dynamics. To establish the results, we develop a coupled Lyapunov drift approach to capture the evolution of multiple sets of coupled and stochastic iterates, which might be of independent interest.

        ----

        ## [3313] Compact Neural Volumetric Video Representations with Dynamic Codebooks

        **Authors**: *Haoyu Guo, Sida Peng, Yunzhi Yan, Linzhan Mou, Yujun Shen, Hujun Bao, Xiaowei Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef63b00ad8475605b2eaf520747f61d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef63b00ad8475605b2eaf520747f61d4-Abstract-Conference.html)

        **Abstract**:

        This paper addresses the challenge of representing high-fidelity volumetric videos with low storage cost. Some recent feature grid-based methods have shown superior performance of fast learning implicit neural representations from input 2D images. However, such explicit representations easily lead to large model sizes when modeling dynamic scenes. To solve this problem, our key idea is reducing the spatial and temporal redundancy of feature grids, which intrinsically exist due to the self-similarity of scenes. To this end, we propose a novel neural representation, named dynamic codebook, which first merges similar features for the model compression and then compensates for the potential decline in rendering quality by a set of dynamic codes. Experiments on the NHR and DyNeRF datasets demonstrate that the proposed approach achieves state-of-the-art rendering quality, while being able to achieve more storage efficiency. The source code is available at https://github.com/zju3dv/compact_vv.

        ----

        ## [3314] Towards Label-free Scene Understanding by Vision Foundation Models

        **Authors**: *Runnan Chen, Youquan Liu, Lingdong Kong, Nenglun Chen, Xinge Zhu, Yuexin Ma, Tongliang Liu, Wenping Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef6c94e9cf4d169298479ee2e230ee13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef6c94e9cf4d169298479ee2e230ee13-Abstract-Conference.html)

        **Abstract**:

        Vision foundation models such as Contrastive Vision-Language Pre-training (CLIP) and Segment Anything (SAM) have demonstrated impressive zero-shot performance on image classification and segmentation tasks. However, the incorporation of CLIP and SAM for label-free scene understanding has yet to be explored. In this paper, we investigate the potential of vision foundation models in enabling networks to comprehend 2D and 3D worlds without labelled data. The primary challenge lies in effectively supervising networks under extremely noisy pseudo labels, which are generated by CLIP and further exacerbated during the propagation from the 2D to the 3D domain. To tackle these challenges, we propose a novel Cross-modality Noisy Supervision (CNS) method that leverages the strengths of CLIP and SAM to supervise 2D and 3D networks simultaneously. In particular, we introduce a prediction consistency regularization to co-train 2D and 3D networks, then further impose the networks' latent space consistency using the SAM's robust feature representation. Experiments conducted on diverse indoor and outdoor datasets demonstrate the superior performance of our method in understanding 2D and 3D open environments. Our 2D and 3D network achieves label-free semantic segmentation with 28.4\% and 33.5\% mIoU on ScanNet, improving 4.7\% and 7.9\%, respectively. For nuImages and nuScenes datasets, the performance is 22.1\% and 26.8\% with improvements of 3.5\% and 6.0\%, respectively. Code is available. (https://github.com/runnanchen/Label-Free-Scene-Understanding)

        ----

        ## [3315] Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions

        **Authors**: *Vladimir Feinberg, Xinyi Chen, Y. Jennifer Sun, Rohan Anil, Elad Hazan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef72fa6579401ffff9da246a5014f055-Abstract-Conference.html)

        **Abstract**:

        Adaptive regularization methods that exploit more than the diagonal entries exhibit state of the art performance for many tasks, but can be prohibitive in terms of memory and running time. We find the spectra of the Kronecker-factored gradient covariance matrix in deep learning (DL) training tasks are concentrated on a small leading eigenspace that changes throughout training, motivating a low-rank sketching approach. We describe a generic method for reducing memory and compute requirements of maintaining a matrix preconditioner using the Frequent Directions (FD) sketch. While previous approaches have explored applying FD for second-order optimization, we present a novel analysis which allows efficient interpolation between resource requirements and the degradation in regret guarantees with rank $k$: in the online convex optimization (OCO) setting over dimension $d$, we match full-matrix $d^2$ memory regret using only $dk$ memory up to additive error in the bottom $d-k$ eigenvalues of the gradient covariance. Further, we show extensions of our work to Shampoo, resulting in a method competitive in quality with Shampoo and Adam, yet requiring only sub-linear memory for tracking second moments.

        ----

        ## [3316] SatBird: a Dataset for Bird Species Distribution Modeling using Remote Sensing and Citizen Science Data

        **Authors**: *Mélisande Teng, Amna Elmustafa, Benjamin Akera, Yoshua Bengio, Hager Radi Abdelwahed, Hugo Larochelle, David Rolnick*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef7653bbc4655305efb89a32362e332a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef7653bbc4655305efb89a32362e332a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Biodiversity is declining at an unprecedented rate, impacting ecosystem services necessary to ensure food, water, and human health and well-being. Understanding the distribution of species and their habitats is crucial for conservation policy planning. However, traditional methods in ecology for species distribution models (SDMs) generally focus either on narrow sets of species or narrow geographical areas and there remain significant knowledge gaps about the distribution of species. A major reason for this is the limited availability of data traditionally used, due to the prohibitive amount of effort and expertise required for traditional field monitoring. The wide availability of remote sensing data and the growing adoption of citizen science tools to collect species observations data at low cost offer an opportunity for improving biodiversity monitoring and enabling the modelling of complex ecosystems. We introduce a novel task for mapping bird species to their habitats by predicting species encounter rates from satellite images, and present SatBird, a satellite dataset of locations in the USA with labels derived from presence-absence observation data from the citizen science database eBird, considering summer (breeding) and winter seasons. We also provide a dataset in Kenya representing low-data regimes. We additionally provide environmental data and species range maps for each location.  We benchmark a set of baselines on our dataset, including SOTA models for remote sensing tasks. SatBird opens up possibilities for scalably modelling properties of ecosystems worldwide.

        ----

        ## [3317] DiffComplete: Diffusion-based Generative 3D Shape Completion

        **Authors**: *Ruihang Chu, Enze Xie, Shentong Mo, Zhenguo Li, Matthias Nießner, Chi-Wing Fu, Jiaya Jia*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ef7bd1f9cbf8a5ab7ddcaccd50699c90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ef7bd1f9cbf8a5ab7ddcaccd50699c90-Abstract-Conference.html)

        **Abstract**:

        We introduce a new diffusion-based approach for shape completion on 3D range scans. Compared with prior deterministic and probabilistic methods, we strike a balance between realism, multi-modality, and high fidelity. We propose DiffComplete by casting shape completion as a generative task conditioned on the incomplete shape. Our key designs are two-fold. First, we devise a hierarchical feature aggregation mechanism to inject conditional features in a spatially-consistent manner. So, we can capture both local details and broader contexts of the conditional inputs to control the shape completion. Second, we propose an occupancy-aware fusion strategy in our model to enable the completion of multiple partial shapes and introduce higher flexibility on the input conditions. DiffComplete sets a new SOTA performance (e.g., 40% decrease on $l_1$ error) on two large-scale 3D shape completion benchmarks. Our completed shapes not only have a realistic outlook compared with the deterministic methods but also exhibit high similarity to the ground truths compared with the probabilistic alternatives. Further, DiffComplete has strong generalizability on objects of entirely unseen classes for both synthetic and real data, eliminating the need for model re-training in various applications.

        ----

        ## [3318] Bayesian Risk-Averse Q-Learning with Streaming Observations

        **Authors**: *Yuhao Wang, Enlu Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html)

        **Abstract**:

        We consider a robust reinforcement learning problem, where a learning agent learns from a simulated training environment. To account for the model mis-specification between this training environment and the true environment due to lack of data, we adopt a formulation of Bayesian risk MDP (BRMDP) with infinite horizon, which uses Bayesian posterior to estimate the transition model and impose a risk functional to account for the model uncertainty. Observations from the real environment that is out of the agent's control arrive periodically and are utilized by the agent to update the Bayesian posterior to reduce model uncertainty. We theoretically demonstrate that BRMDP balances the trade-off between robustness and conservativeness, and we further develop a multi-stage Bayesian risk-averse Q-learning algorithm to solve BRMDP with streaming observations from real environment. The proposed algorithm learns a risk-averse yet optimal policy that depends on the availability of real-world observations. We provide a theoretical guarantee of strong convergence for the proposed algorithm.

        ----

        ## [3319] On the Planning Abilities of Large Language Models - A Critical Investigation

        **Authors**: *Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, Subbarao Kambhampati*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efb2072a358cefb75886a315a6fcf880-Abstract-Conference.html)

        **Abstract**:

        Intrigued by the claims of emergent reasoning capabilities in LLMs trained on general web corpora, in this paper, we set out to investigate their planning capabilities. We aim to evaluate (1) the effectiveness of LLMs in generating plans autonomously in commonsense planning tasks and (2) the potential of LLMs as a source of heuristic guidance for other agents (AI planners) in their planning tasks. We conduct a systematic study by generating a suite of instances on domains similar to the ones employed in the International Planning Competition and evaluate LLMs in two distinct modes: autonomous and heuristic. Our findings reveal that LLMsâ€™ ability to generate executable plans autonomously is rather limited, with the best model (GPT-4) having an average success rate of ~12% across the domains. However, the results in the heuristic mode show more promise. In the heuristic mode, we demonstrate that LLM-generated plans can improve the search process for underlying sound planners and additionally show that external verifiers can help provide feedback on the generated plans and back-prompt the LLM for better plan generation.

        ----

        ## [3320] Is RLHF More Difficult than Standard RL? A Theoretical Perspective

        **Authors**: *Yuanhao Wang, Qinghua Liu, Chi Jin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efb9629755e598c4f261c44aeb6fde5e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efb9629755e598c4f261c44aeb6fde5e-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning from Human Feedback (RLHF) learns from preference signals, while standard Reinforcement Learning (RL) directly learns from reward signals. Preferences arguably contain less information than rewards, which makes preference-based RL seemingly more difficult. This paper theoretically proves that, for a wide range of preference models, we can solve preference-based RL directly using existing algorithms and techniques for reward-based RL, with small or no extra costs. Specifically, (1) for preferences that are drawn from reward-based probabilistic models, we reduce the problem to robust reward-based RL that can tolerate small errors in rewards; (2) for general arbitrary preferences where the objective is to find the von Neumann winner, we reduce the problem to multiagent reward-based RL which finds Nash equilibria for factored Markov games under a restricted set of policies. The latter case can be further reduce to adversarial MDP when preferences only depend on the final state. We instantiate all reward-based RL subroutines by concrete provable algorithms, and apply our theory to a large class of models including tabular MDPs and MDPs with generic function approximation. We further provide guarantees when K-wise comparisons are available.

        ----

        ## [3321] How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model

        **Authors**: *Michael Hanna, Ollie Liu, Alexandre Variengien*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efbba7719cc5172d175240f24be11280-Abstract-Conference.html)

        **Abstract**:

        Pre-trained language models can be surprisingly adept at tasks they were not explicitly trained on, but how they implement these capabilities is poorly understood. In this paper, we investigate the basic mathematical abilities often acquired by pre-trained language models. Concretely, we use mechanistic interpretability techniques to explain the (limited) mathematical abilities of GPT-2 small. As a case study, we examine its ability to take in sentences such as "The war lasted from the year 1732 to the year 17", and predict valid two-digit end years (years > 32). We first identify a circuit, a small subset of GPT-2 small's computational graph that computes this task's output. Then, we explain the role of each circuit component, showing that GPT-2 small's final multi-layer perceptrons boost the probability of end years greater than the start year. Finally, we find related tasks that activate our circuit. Our results suggest that GPT-2 small computes greater-than using a complex but general mechanism that activates across diverse contexts.

        ----

        ## [3322] NAVI: Category-Agnostic Image Collections with High-Quality 3D Shape and Pose Annotations

        **Authors**: *Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, André Araújo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, Howard Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efc90033e6e1b05485312dd09fe302b8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/efc90033e6e1b05485312dd09fe302b8-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recent advances in neural reconstruction enable high-quality 3D object reconstruction from casually captured image collections. Current techniques mostly analyze their progress on relatively simple image collections where SfM techniques can provide ground-truth (GT) camera poses. We note that SfM techniques tend to fail on in-the-wild image collections such as image search results with varying backgrounds and illuminations. To enable systematic research progress on 3D reconstruction from casual image captures, we propose `NAVI': a new dataset of category-agnostic image collections of objects with high-quality 3D scans along with per-image 2D-3D alignments providing near-perfect GT camera parameters. These 2D-3D alignments allow us to extract accurate derivative annotations such as dense pixel correspondences, depth and segmentation maps. We demonstrate the use of NAVI image collections on different problem settings and show that NAVI enables more thorough evaluations that were not possible with existing datasets. We believe NAVI is beneficial for systematic research progress on 3D reconstruction and correspondence estimation.

        ----

        ## [3323] Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation

        **Authors**: *Jia-Xing Zhong, Ta Ying Cheng, Yuhang He, Kai Lu, Kaichen Zhou, Andrew Markham, Niki Trigoni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efca456a4e861f3b47455c44bb134424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efca456a4e861f3b47455c44bb134424-Abstract-Conference.html)

        **Abstract**:

        A truly generalizable approach to rigid segmentation and motion estimation is fundamental to 3D understanding of articulated objects and moving scenes. In view of the closely intertwined relationship between segmentation and motion estimates, we present an SE(3) equivariant architecture and a training strategy to tackle this task in an unsupervised manner. Our architecture is composed of two interconnected, lightweight heads. These heads predict segmentation masks using point-level invariant features and estimate motion from SE(3) equivariant features, all without the need for category information. Our training strategy is unified and can be implemented online, which jointly optimizes the predicted segmentation and motion by leveraging the interrelationships among scene flow, segmentation mask, and rigid transformations. We conduct experiments on four datasets to demonstrate the superiority of our method. The results show that our method excels in both model performance and computational efficiency, with only 0.25M parameters and 0.92G FLOPs. To the best of our knowledge, this is the first work designed for category-agnostic part-level SE(3) equivariance in dynamic point clouds.

        ----

        ## [3324] Suggesting Variable Order for Cylindrical Algebraic Decomposition via Reinforcement Learning

        **Authors**: *Fuqi Jia, Yuhang Dong, Minghao Liu, Pei Huang, Feifei Ma, Jian Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efcb5b06ce8bb672ffa26b9dc5cdd0f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efcb5b06ce8bb672ffa26b9dc5cdd0f9-Abstract-Conference.html)

        **Abstract**:

        Cylindrical Algebraic Decomposition (CAD) is one of the pillar algorithms of symbolic computation, and its worst-case complexity is double exponential to the number of variables. Researchers found that variable order dramatically affects efficiency and proposed various heuristics. The existing learning-based methods are all supervised learning methods that cannot cope with diverse polynomial sets.This paper proposes two Reinforcement Learning (RL) approaches combined with Graph Neural Networks (GNN) for Suggesting Variable Order (SVO). One is GRL-SVO(UP), a branching heuristic integrated with CAD. The other is GRL-SVO(NUP), a fast heuristic providing a total order directly. We generate a random dataset and collect a real-world dataset from SMT-LIB. The experiments show that our approaches outperform state-of-the-art learning-based heuristics and are competitive with the best expert-based heuristics. Interestingly, our models show a strong generalization ability, working well on various datasets even if they are only trained on a 3-var random dataset. The source code and data are available at https://github.com/dongyuhang22/GRL-SVO.

        ----

        ## [3325] GLOBER: Coherent Non-autoregressive Video Generation via GLOBal Guided Video DecodER

        **Authors**: *Mingzhen Sun, Weining Wang, Zihan Qin, Jiahui Sun, Sihan Chen, Jing Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efe36e55d80a94d1726f660b8d237a0f-Abstract-Conference.html)

        **Abstract**:

        Video generation necessitates both global coherence and local realism. This work presents a novel non-autoregressive method GLOBER, which first generates global features to obtain comprehensive global guidance and then synthesizes video frames based on the global features to generate coherent videos. Specifically, we propose a video auto-encoder, where a video encoder encodes videos into global features, and a video decoder, built on a diffusion model, decodes the global features and synthesizes video frames in a non-autoregressive manner. To achieve maximum flexibility, our video decoder perceives temporal information through normalized frame indexes, which enables it to synthesize arbitrary sub video clips with predetermined starting and ending frame indexes. Moreover, a novel adversarial loss is introduced to improve the global coherence and local realism between the synthesized video frames. Finally, we employ a diffusion-based video generator to fit the global features outputted by the video encoder for video generation. Extensive experimental results demonstrate the effectiveness and efficiency of our proposed method, and new state-of-the-art results have been achieved on multiple benchmarks.

        ----

        ## [3326] Dense and Aligned Captions (DAC) Promote Compositional Reasoning in VL Models

        **Authors**: *Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogério Feris, Shimon Ullman, Leonid Karlinsky*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/efe406d6d2674d176cdcd958ce605d17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/efe406d6d2674d176cdcd958ce605d17-Abstract-Conference.html)

        **Abstract**:

        Vision and Language (VL) models offer an effective method for aligning representation spaces of images and text allowing for numerous applications such as cross-modal retrieval, visual and multi-hop question answering, captioning, and many more. However, the aligned image-text spaces learned by all the popular VL models are still suffering from the so-called 'object bias' - their representations behave as 'bags of nouns' mostly ignoring or downsizing the attributes, relations, and states of objects described/appearing in texts/images. Although some great attempts at fixing these `compositional reasoning' issues were proposed in the recent literature, the problem is still far from being solved. In this paper, we uncover two factors limiting the VL models' compositional reasoning performance. These two factors are properties of the paired VL dataset used for finetuning (or pre-training) the VL model: (i) the caption quality, or in other words 'image-alignment', of the texts; and (ii) the 'density' of the captions in the sense of mentioning all the details appearing on the image. We propose a fine-tuning approach for automatically treating these factors on a standard collection of paired VL data (CC3M). Applied to CLIP, we demonstrate its significant compositional reasoning performance increase of up to $\sim27$\% over the base model, up to $\sim20$\% over the strongest baseline, and by $6.7$\% on average. Our code is provided in the Supplementary and would be released upon acceptance.

        ----

        ## [3327] Latent SDEs on Homogeneous Spaces

        **Authors**: *Sebastian Zeng, Florian Graf, Roland Kwitt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0172a5da5a2611e3dc0fe9c6e9a7480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0172a5da5a2611e3dc0fe9c6e9a7480-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of variational Bayesian inference in a latent variable model where a (possibly complex) observed stochastic process is governed by the unobserved solution of a latent stochastic differential equation (SDE). Motivated by the challenges that arise when trying to learn a latent SDE in $\mathbb{R}^n$ from large-scale data, such as efficient gradient computation, we take a step back and study a specific subclass instead. In our case, the SDE evolves inside a homogeneous latent space and is induced by stochastic dynamics of the corresponding (matrix) Lie group. In the context of learning problems, SDEs on the $n$-dimensional unit sphere are arguably the most relevant incarnation of this setup. For variational inference, the sphere not only facilitates using a uniform prior on the initial state of the SDE, but we also obtain a particularly simple and intuitive expression for the KL divergence between the approximate posterior and prior process in the evidence lower bound. We provide empirical evidence that a latent SDE of the proposed type can be learned efficiently by means of an existing one-step geometric Euler-Maruyama scheme. Despite restricting ourselves to a less diverse class of SDEs, we achieve competitive or even state-of-the-art performance on a collection of time series interpolation and classification benchmarks.

        ----

        ## [3328] Balancing Risk and Reward: A Batched-Bandit Strategy for Automated Phased Release

        **Authors**: *Yufan Li, Jialiang Mao, Iavor Bojinov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f02a7dd6bd3d038b51d092d99e74c638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f02a7dd6bd3d038b51d092d99e74c638-Abstract-Conference.html)

        **Abstract**:

        Phased releases are a common strategy in the technology industry for gradually releasing new products or updates through a sequence of A/B tests in which the number of treated units gradually grows until full deployment or deprecation. Performing phased releases in a principled way requires selecting the proportion of units assigned to the new release in a way that balances the risk of an adverse effect with the need to iterate and learn from the experiment rapidly. In this paper, we formalize this problem and propose an algorithm that automatically determines the release percentage at each stage in the schedule, balancing the need to control risk while maximizing ramp-up speed. Our framework models the challenge as a constrained batched bandit problem that ensures that our pre-specified experimental budget is not depleted with high probability. Our proposed algorithm leverages an adaptive Bayesian approach in which the maximal number of units assigned to the treatment is determined by the posterior distribution, ensuring that the probability of depleting the remaining budget is low. Notably, our approach analytically solves the ramp sizes by inverting probability bounds, eliminating the need for challenging rare-event Monte Carlo simulation. It only requires computing means and variances of outcome subsets, making it highly efficient and parallelizable.

        ----

        ## [3329] Preconditioning Matters: Fast Global Convergence of Non-convex Matrix Factorization via Scaled Gradient Descent

        **Authors**: *Xixi Jia, Hailin Wang, Jiangjun Peng, Xiangchu Feng, Deyu Meng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html)

        **Abstract**:

        Low-rank matrix factorization (LRMF) is a canonical problem in non-convex optimization, the objective function to be minimized is non-convex and even non-smooth, which makes the global convergence guarantee of gradient-based algorithm quite challenging. Recent work made a breakthrough on proving that standard gradient descent converges to the $\varepsilon$-global minima after $O( \frac{d \kappa^2}{\tau^2} {\rm ln} \frac{d \sigma_d}{\tau} + \frac{d \kappa^2}{\tau^2} {\rm ln} \frac{\sigma_d}{\varepsilon})$ iterations from small initialization with a very small learning rate (both are related to the small constant $\tau$). While the dependence of the convergence on the \textit{condition number} $\kappa$ and \textit{small learning rate} makes it not practical especially for ill-conditioned LRMF problem.In this paper, we show that precondition helps in accelerating the convergence and prove that the scaled gradient descent (ScaledGD) and its variant, alternating scaled gradient descent (AltScaledGD) converge to an $\varepsilon$-global minima after $O( {\rm ln} \frac{d}{\delta} + {\rm ln} \frac{d}{\varepsilon})$ iterations from general random initialization. Meanwhile, for small initialization as in gradient descent, both ScaledGD and AltScaledGD converge to $\varepsilon$-global minima after only $O({\rm ln} \frac{d}{\varepsilon})$ iterations. Furthermore, we prove that as a proximity to the alternating minimization, AltScaledGD converges faster than ScaledGD, its global convergence does not rely on small learning rate and small initialization, which certificates the advantages of AltScaledGD in LRMF.

        ----

        ## [3330] Deep Insights into Noisy Pseudo Labeling on Graph Data

        **Authors**: *Botao Wang, Jia Li, Yang Liu, Jiashun Cheng, Yu Rong, Wenjia Wang, Fugee Tsung*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0318ba897cee71ce200e408dea6062e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0318ba897cee71ce200e408dea6062e-Abstract-Conference.html)

        **Abstract**:

        Pseudo labeling (PL) is a wide-applied strategy to enlarge the labeled dataset by self-annotating the potential samples during the training process. Several works have shown that it can improve the graph learning model performance in general. However, we notice that the incorrect labels can be fatal to the graph training process. Inappropriate PL may result in the performance degrading, especially on graph data where the noise can propagate. Surprisingly, the corresponding error is seldom theoretically analyzed in the literature. In this paper, we aim to give deep insights of PL on graph learning models. We first present the error analysis of PL strategy by showing that the error is bounded by the confidence of PL threshold and consistency of multi-view prediction. Then, we theoretically illustrate the effect of PL on convergence property. Based on the analysis, we propose a cautious pseudo labeling methodology in which we pseudo label the samples with highest confidence and multi-view consistency. Finally, extensive experiments demonstrate that the proposed strategy improves graph learning process and outperforms other PL strategies on link prediction and node classification tasks.

        ----

        ## [3331] Predicting a Protein's Stability under a Million Mutations

        **Authors**: *Jeffrey Ouyang-Zhang, Daniel Jesus Diaz, Adam R. Klivans, Philipp Krähenbühl*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f03cb785864596fa5901f1359d23fd81-Abstract-Conference.html)

        **Abstract**:

        Stabilizing proteins is a foundational step in protein engineering. However, the evolutionary pressure of all extant proteins makes identifying the scarce number of mutations that will improve thermodynamic stability challenging. Deep learning has recently emerged as a powerful tool for identifying promising mutations.Existing approaches, however, are computationally expensive, as the number of model inferences scales with the number of mutations queried. Our main contribution is a simple, parallel decoding algorithm.Mutate Everything is capable of predicting the effect of all single and double mutations in one forward pass. It is even versatile enough to predict higher-order mutations with minimal computational overhead.We build Mutate Everything on top of ESM2 and AlphaFold, neither of which were trained to predict thermodynamic stability.We trained on the Mega-Scale cDNA proteolysis dataset and achieved state-of-the-art performance on single and higher-order mutations on S669, ProTherm, and ProteinGym datasets.Our code is available at https://github.com/jozhang97/MutateEverything.

        ----

        ## [3332] Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems

        **Authors**: *Fiona Lippert, Bart Kranstauber, Emiel van Loon, Patrick Forré*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)

        **Abstract**:

        Probabilistic inference in high-dimensional state-space models is computationally challenging. For many spatiotemporal systems, however, prior knowledge about the dependency structure of state variables is available. We leverage this structure to develop a computationally efficient approach to state estimation and learning in graph-structured state-space models with (partially) unknown dynamics and limited historical data. Building on recent methods that combine ideas from deep learning with principled inference in Gaussian Markov random fields (GMRF), we reformulate graph-structured state-space models as Deep GMRFs defined by simple spatial and temporal graph layers. This results in a flexible spatiotemporal prior that can be learned efficiently from a single time sequence via variational inference. Under linear Gaussian assumptions, we retain a closed-form posterior, which can be sampled efficiently using the conjugate gradient method, scaling favourably compared to classical Kalman filter based approaches.

        ----

        ## [3333] Computational Complexity of Learning Neural Networks: Smoothness and Degeneracy

        **Authors**: *Amit Daniely, Nati Srebro, Gal Vardi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html)

        **Abstract**:

        Understanding when neural networks can be learned efficientlyis a fundamental question in learning theory.Existing hardness results suggest that assumptions on both the input distribution and the network's weights are necessary for obtaining efficient algorithms. Moreover, it was previously shown that depth-$2$ networks can be efficiently learned under the assumptions that the input distribution is Gaussian, and the weight matrix is non-degenerate. In this work, we study whether such assumptions may suffice for learning deeper networks and prove negative results. We show that learning depth-$3$ ReLU networks under the Gaussian input distribution is hard even in the smoothed-analysis framework, where a random noise is added to the network's parameters. It implies that learning depth-$3$ ReLU networks under the Gaussian distribution is hard even if the weight matrices are non-degenerate. Moreover, we consider depth-$2$ networks, and show hardness of learning in the smoothed-analysis framework, where both the network parameters and the input distribution are smoothed. Our hardness results are under a well-studied assumption on the existence of local pseudorandom generators.

        ----

        ## [3334] LoCoOp: Few-Shot Out-of-Distribution Detection via Prompt Learning

        **Authors**: *Atsuyuki Miyai, Qing Yu, Go Irie, Kiyoharu Aizawa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0606b882692637835e8ac981089eccd-Abstract-Conference.html)

        **Abstract**:

        We present a novel vision-language prompt learning approach for few-shot out-of-distribution (OOD) detection. Few-shot OOD detection aims to detect OOD images from classes that are unseen during training using only a few labeled in-distribution (ID) images. While prompt learning methods such as CoOp have shown effectiveness and efficiency in few-shot ID classification, they still face limitations in OOD detection due to the potential presence of ID-irrelevant information in text embeddings. To address this issue, we introduce a new approach called $\textbf{Lo}$cal regularized $\textbf{Co}$ntext $\textbf{Op}$timization (LoCoOp), which performs OOD regularization that utilizes the portions of CLIP local features as OOD features during training. CLIP's local features have a lot of ID-irrelevant nuisances ($\textit{e.g.}$, backgrounds), and by learning to push them away from the ID class text embeddings, we can remove the nuisances in the ID class text embeddings and enhance the separation between ID and OOD. Experiments on the large-scale ImageNet OOD detection benchmarks demonstrate the superiority of our LoCoOp over zero-shot, fully supervised detection methods and prompt learning methods. Notably, even in a one-shot setting -- just one label per class, LoCoOp outperforms existing zero-shot and fully supervised detection methods. The code is available via https://github.com/AtsuMiyai/LoCoOp.

        ----

        ## [3335] AdANNS: A Framework for Adaptive Semantic Search

        **Authors**: *Aniket Rege, Aditya Kusupati, Sharan Ranjit S, Alan Fan, Qingqing Cao, Sham M. Kakade, Prateek Jain, Ali Farhadi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f062da1973ac9ac61fc6d44dd7fa309f-Abstract-Conference.html)

        **Abstract**:

        Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture tail queries and data points, learned representations typically are _rigid, high-dimensional_ vectors that are generally used as-is in the entire ANNS pipeline and can lead to computationally expensive retrieval. In this paper, we argue that instead of rigid representations, different stages of ANNS can leverage _adaptive representations_ of varying capacities to achieve significantly better accuracy-compute trade-offs, i.e., stages of ANNS that can get away with more approximate computation should use a lower-capacity representation of the same data point. To this end, we introduce AdANNS, a novel ANNS design framework that explicitly leverages the flexibility of Matryoshka Representations. We demonstrate state-of-the-art accuracy-compute trade-offs using novel AdANNS-based key ANNS building blocks like search data structures (AdANNS-IVF) and quantization (AdANNS-OPQ). For example on ImageNet retrieval, AdANNS-IVF is up to $\mathbf{1.5}$% more accurate than the rigid representations-based IVF at the same compute budget; and matches accuracy while being up to $\mathbf{90}\times$ faster in _wall-clock time_. For Natural Questions, $32$-byte AdANNS-OPQ matches the accuracy of the $64$-byte OPQ baseline constructed using rigid representations -- _same accuracy at half the cost!_ We further show that the gains from AdANNS translate to modern-day composite ANNS indices that combine search structures and quantization. Finally, we demonstrate that AdANNS can enable inference-time adaptivity for compute-aware search on ANNS indices built non-adaptively on matryoshka representations. Code is open-sourced at https://github.com/RAIVNLab/AdANNS.

        ----

        ## [3336] When Do Neural Nets Outperform Boosted Trees on Tabular Data?

        **Authors**: *Duncan C. McElfresh, Sujay Khandagale, Jonathan Valverde, Vishak Prasad C., Ganesh Ramakrishnan, Micah Goldblum, Colin White*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Tabular data is one of the most commonly used types of data in machine learning. Despite recent advances in neural nets (NNs) for tabular data, there is still an active discussion on whether or not NNs generally outperform gradient-boosted decision trees (GBDTs) on tabular data, with several recent works arguing either that GBDTs consistently outperform NNs on tabular data, or vice versa. In this work, we take a step back and question the importance of this debate. To this end, we conduct the largest tabular data analysis to date, comparing 19 algorithms across 176 datasets, and we find that the 'NN vs. GBDT' debate is overemphasized: for a surprisingly high number of datasets, either the performance difference between GBDTs and NNs is negligible, or light hyperparameter tuning on a GBDT is more important than choosing between NNs and GBDTs. Next, we analyze dozens of metafeatures to determine what \emph{properties} of a dataset make NNs or GBDTs better-suited to perform well. For example, we find that GBDTs are much better than NNs at handling skewed or heavy-tailed feature distributions and other forms of dataset irregularities. Our insights act as a guide for practitioners to determine which techniques may work best on their dataset. Finally, with the goal of accelerating tabular data research, we release the TabZilla Benchmark Suite: a collection of the 36 'hardest' of the datasets we study. Our benchmark suite, codebase, and all raw results are available at https://github.com/naszilla/tabzilla.

        ----

        ## [3337] Adversarially Robust Distributed Count Tracking via Partial Differential Privacy

        **Authors**: *Zhongzheng Xiong, Xiaoyi Zhu, Zengfeng Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0722b58f02d7793acf7d328928f933a-Abstract-Conference.html)

        **Abstract**:

        We study the distributed tracking model, also known as distributed functional monitoring. This model involves $k$ sites each receiving a stream of items and communicating with the central server. The server's task is to track a function of all items received thus far continuously, with minimum communication cost. For count tracking, it is known that there is a $\sqrt{k}$ gap in communication between deterministic and randomized algorithms. However, existing randomized algorithms assume an "oblivious adversary" who constructs the entire input streams before the algorithm starts. Here we consider adaptive adversaries who can choose new items based on previous answers from the algorithm. Deterministic algorithms are trivially robust to adaptive adversaries, while randomized ones may not. Therefore, we investigate whether the $\sqrt{k}$ advantage of randomized algorithms is from randomness itself or the oblivious adversary assumption. We provide an affirmative answer to this question by giving a robust algorithm with optimal communication. Existing robustification techniques do not yield optimal bounds due to the inherent challenges of the distributed nature of the problem. To address this, we extend the differential privacy framework by introducing "partial differential privacy" and proving a new generalization theorem. This theorem may have broader applications beyond robust count tracking, making it of independent interest.

        ----

        ## [3338] Energy-Based Cross Attention for Bayesian Context Update in Text-to-Image Diffusion Models

        **Authors**: *Geon Yeong Park, Jeongsol Kim, Beomsu Kim, Sang Wan Lee, Jong Chul Ye*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0878b7efa656b3bbd407c9248d13751-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0878b7efa656b3bbd407c9248d13751-Abstract-Conference.html)

        **Abstract**:

        Despite the remarkable performance of text-to-image diffusion models in image generation tasks, recent studies have raised the issue that generated images sometimes cannot capture the intended semantic contents of the text prompts, which phenomenon is often called semantic misalignment. To address this, here we present a novel energy-based model (EBM) framework for adaptive context control by modeling the posterior of context vectors. Specifically, we first formulate EBMs of latent image representations and text embeddings in each cross-attention layer of the denoising autoencoder. Then, we obtain the gradient of the log posterior of context vectors, which can be updated and transferred to the subsequent cross-attention layer, thereby implicitly minimizing a nested hierarchy of energy functions. Our latent EBMs further allow zero-shot compositional generation as a linear combination of cross-attention outputs from different contexts. Using extensive experiments, we demonstrate that the proposed method is highly effective in handling various image generation tasks, including multi-concept generation, text-guided image inpainting, and real and synthetic image editing. Code: https://github.com/EnergyAttention/Energy-Based-CrossAttention.

        ----

        ## [3339] Optimal Algorithms for the Inhomogeneous Spiked Wigner Model

        **Authors**: *Aleksandr Pak, Justin Ko, Florent Krzakala*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0a6b46b0183a62a2db973014e3429f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0a6b46b0183a62a2db973014e3429f4-Abstract-Conference.html)

        **Abstract**:

        We study a spiked Wigner problem with an inhomogeneous noise profile. Our aim in this problem is to recover the signal passed through an inhomogeneous low-rank matrix channel. While the information-theoretic performances are well-known, we focus on the algorithmic problem. First, we derive an approximate message-passing algorithm (AMP) for the inhomogeneous problem and show that its rigorous state evolution coincides with the information-theoretic optimal Bayes fixed-point equations. Second, we deduce a simple and efficient spectral method that outperforms PCA and is shown to match the information-theoretic transition.

        ----

        ## [3340] Learning Repeatable Speech Embeddings Using An Intra-class Correlation Regularizer

        **Authors**: *Jianwei Zhang, Suren Jayasuriya, Visar Berisha*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0aa7e9e67515fa0c607c2959ccda6a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0aa7e9e67515fa0c607c2959ccda6a0-Abstract-Conference.html)

        **Abstract**:

        A good supervised embedding for a specific machine learning task is only sensitive to changes in the label of interest and is invariant to other confounding factors. We leverage the concept of repeatability from measurement theory to describe this property and propose to use the intra-class correlation coefficient (ICC) to evaluate the repeatability of embeddings. We then propose a novel regularizer, the ICC regularizer, as a complementary component for contrastive losses to guide deep neural networks to produce embeddings with higher repeatability. We use simulated data to explain why the ICC regularizer works better on minimizing the intra-class variance than the contrastive loss alone. We implement the ICC regularizer and apply it to three speech tasks: speaker verification, voice style conversion, and a clinical application for detecting dysphonic voice. The experimental results demonstrate that adding an ICC regularizer can improve the repeatability of learned embeddings compared to only using the contrastive loss; further, these embeddings lead to improved performance in these downstream tasks.

        ----

        ## [3341] Momentum Provably Improves Error Feedback!

        **Authors**: *Ilyas Fatkhullin, Alexander Tyurin, Peter Richtárik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0b1515be276f6ba82b4f2b25e50bef0-Abstract-Conference.html)

        **Abstract**:

        Due to the high communication overhead when training machine learning models in a distributed environment, modern algorithms invariably rely on lossy communication compression. However, when untreated, the errors caused by compression propagate, and can lead to severely unstable behavior, including exponential divergence. Almost a decade ago, Seide et al. [2014] proposed an error feedback (EF) mechanism, which we refer to as EF14, as an immensely effective heuristic for mitigating this issue. However, despite steady algorithmic and theoretical  advances in the EF field in the last decade, our understanding is far from complete. In this work we address one of the most pressing issues. In particular, in the canonical nonconvex setting, all known variants of EF rely on very large batch sizes to converge, which can be prohibitive in practice. We propose a surprisingly simple fix which removes this issue both theoretically, and in practice: the application of Polyak's momentum to the latest incarnation of EF due to Richt√°rik et al. [2021] known as EF21. Our algorithm, for which we coin the name EF21-SGDM, improves the communication and sample complexities of previous error feedback algorithms under standard smoothness and bounded variance assumptions, and does not require any further strong assumptions such as bounded gradient dissimilarity. Moreover, we propose a double momentum version of our method that improves the complexities even further. Our proof seems to be novel even when compression is removed form the method, and as such, our proof technique is of independent interest in the study of nonconvex stochastic optimization enriched with Polyak's momentum.

        ----

        ## [3342] Optimal Convergence Rate for Exact Policy Mirror Descent in Discounted Markov Decision Processes

        **Authors**: *Emmeran Johnson, Ciara Pike-Burke, Patrick Rebeschini*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f0d7b528c31bc3f9a0d5bab515ed6ed5-Abstract-Conference.html)

        **Abstract**:

        Policy Mirror Descent (PMD) is a general family of algorithms that covers a wide range of novel and fundamental methods in reinforcement learning. Motivated by the instability of policy iteration (PI) with inexact policy evaluation, unregularised PMD algorithmically regularises the policy improvement step of PI without regularising the objective function. With exact policy evaluation, PI is known to converge linearly with a rate given by the discount factor $\gamma$ of a Markov Decision Process. In this work, we bridge the gap between PI and PMD with exact policy evaluation and show that the dimension-free $\gamma$-rate of PI can be achieved by the general family of unregularised PMD algorithms under an adaptive step-size. We show that both the rate and step-size are unimprovable for PMD: we provide matching lower bounds that demonstrate that the $\gamma$-rate is optimal for PMD methods as well as PI and that the adaptive step-size is necessary to achieve it. Our work is the first to relate PMD to rate-optimality and step-size necessity. Our study of the convergence of PMD avoids the use of the performance difference lemma, which leads to a direct analysis of independent interest. We also extend the analysis to the inexact setting and establish the first dimension-optimal sample complexity for unregularised PMD under a generative model, improving upon the best-known result.

        ----

        ## [3343] Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models

        **Authors**: *Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhihua Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html)

        **Abstract**:

        Due to the ease of training, ability to scale, and high sample quality, diffusion models (DMs) have become the preferred option for generative modeling, with numerous pre-trained models available for a wide variety of datasets. Containing intricate information about data distributions, pre-trained DMs are valuable assets for downstream applications. In this work, we consider learning from pre-trained DMs and transferring their knowledge to other generative models in a data-free fashion. Specifically, we propose a general framework called Diff-Instruct to instruct the training of arbitrary generative models as long as the generated samples are differentiable with respect to the model parameters. Our proposed Diff-Instruct is built on a rigorous mathematical foundation where the instruction process directly corresponds to minimizing a novel divergence we call Integral Kullback-Leibler (IKL) divergence. IKL is tailored for DMs by calculating the integral of the KL divergence along a diffusion process, which we show to be more robust in comparing distributions with misaligned supports. We also reveal non-trivial connections of our method to existing works such as DreamFusion \citep{poole2022dreamfusion}, and generative adversarial training. To demonstrate the effectiveness and universality of Diff-Instruct, we consider two scenarios: distilling pre-trained diffusion models and refining existing GAN models. The experiments on distilling pre-trained diffusion models show that Diff-Instruct results in state-of-the-art single-step diffusion-based models. The experiments on refining GAN models show that the Diff-Instruct can consistently improve the pre-trained generators of GAN models across various settings. Our official code is released through \url{https://github.com/pkulwj1994/diff_instruct}.

        ----

        ## [3344] Characterization of Overfitting in Robust Multiclass Classification

        **Authors**: *Jingyuan Xu, Weiwei Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f144ab9985c739a5091ec188a2688644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f144ab9985c739a5091ec188a2688644-Abstract-Conference.html)

        **Abstract**:

        This paper considers the following question: Given the number of classes m, the number of robust accuracy queries k, and the number of test examples in the dataset n, how much can adaptive algorithms robustly overfit the test dataset? We solve this problem by equivalently giving near-matching upper and lower bounds of the robust overfitting bias in multiclass classification problems.

        ----

        ## [3345] Expanding Small-Scale Datasets with Guided Imagination

        **Authors**: *Yifan Zhang, Daquan Zhou, Bryan Hooi, Kai Wang, Jiashi Feng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f188a55392d3a7509b0b27f8d24364bb-Abstract-Conference.html)

        **Abstract**:

        The power of DNNs relies heavily on the quantity and quality of training data. However, collecting and annotating data on a large scale is often expensive and time-consuming. To address this issue, we explore a new task, termed dataset expansion, aimed at expanding a ready-to-use small dataset by automatically creating new labeled samples. To this end, we present a Guided Imagination Framework (GIF) that leverages cutting-edge generative models like DALL-E2 and Stable Diffusion (SD) to "imagine" and create informative new data from the input seed data. Specifically, GIF conducts data imagination by optimizing the latent features of the seed data in the semantically meaningful space of the prior model, resulting in the creation of photo-realistic images with new content. To guide the imagination towards creating informative samples for model training, we introduce two key criteria, i.e., class-maintained information boosting and sample diversity promotion. These criteria are verified to be essential for effective dataset expansion: GIF-SD obtains 13.5% higher model accuracy on natural image datasets than unguided expansion with SD. With these essential criteria, GIF successfully expands small datasets in various scenarios, boosting model accuracy by 36.9% on average over six natural image datasets and by 13.5% on average over three medical datasets. The source code is available at https://github.com/Vanint/DatasetExpansion.

        ----

        ## [3346] Parallel-mentoring for Offline Model-based Optimization

        **Authors**: *Can Chen, Christopher Beckham, Zixuan Liu, Xue (Steve) Liu, Chris Pal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f189e7580acad0fc7fd45405817ddee3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f189e7580acad0fc7fd45405817ddee3-Abstract-Conference.html)

        **Abstract**:

        We study offline model-based optimization to maximize a black-box objective function with a static dataset of designs and scores. These designs encompass a variety of domains, including materials, robots, DNA sequences, and proteins. A common approach trains a proxy on the static dataset and performs gradient ascent to obtain new designs. However, this often results in poor designs due to the proxy inaccuracies for out-of-distribution designs. Recent studies indicate that (a) gradient ascent with a mean ensemble of proxies generally outperforms simple gradient ascent, and (b) a trained proxy provides weak ranking supervision signals for design selection. Motivated by (a) and (b), we propose $\textit{parallel-mentoring}$ as an effective and novel method that facilitates mentoring among proxies, creating a more robust ensemble to mitigate the out-of-distribution issue. We focus on the three-proxy case in the main paper and our method consists of two modules. The first module, $\textit{voting-based pairwise supervision}$, operates on three parallel proxies and captures their ranking supervision signals as pairwise comparison labels. These labels are combined through majority voting to generate consensus labels, which incorporates ranking supervision signals from all proxies and enables mutual mentoring. Yet, label noise arises due to possible incorrect consensus. To alleviate this, we introduce an $\textit{adaptive soft-labeling}$ module with soft-labels initialized as consensus labels. Based on bi-level optimization, this module fine-tunes proxies in the inner level and learns more accurate labels in the outer level to adaptively mentor proxies, resulting in a more robust ensemble. Experiments validate the effectiveness of our method. Our code is available here.

        ----

        ## [3347] Nominality Score Conditioned Time Series Anomaly Detection by Point/Sequential Reconstruction

        **Authors**: *Chih-Yu Lai, Fan-Keng Sun, Zhengqi Gao, Jeffrey H. Lang, Duane S. Boning*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1cf02ce09757f57c3b93c0db83181e0-Abstract-Conference.html)

        **Abstract**:

        Time series anomaly detection is challenging due to the complexity and variety of patterns that can occur. One major difficulty arises from modeling time-dependent relationships to find contextual anomalies while maintaining detection accuracy for point anomalies. In this paper, we propose a framework for unsupervised time series anomaly detection that utilizes point-based and sequence-based reconstruction models. The point-based model attempts to quantify point anomalies, and the sequence-based model attempts to quantify both point and contextual anomalies. Under the formulation that the observed time point is a two-stage deviated value from a nominal time point, we introduce a nominality score calculated from the ratio of a combined value of the reconstruction errors. We derive an induced anomaly score by further integrating the nominality score and anomaly score, then theoretically prove the superiority of the induced anomaly score over the original anomaly score under certain conditions. Extensive studies conducted on several public datasets show that the proposed framework outperforms most state-of-the-art baselines for time series anomaly detection.

        ----

        ## [3348] Frequency-domain MLPs are More Effective Learners in Time Series Forecasting

        **Authors**: *Kun Yi, Qi Zhang, Wei Fan, Shoujin Wang, Pengyang Wang, Hui He, Ning An, Defu Lian, Longbing Cao, Zhendong Niu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html)

        **Abstract**:

        Time series forecasting has played the key role in different industrial, including finance, traffic, energy, and healthcare domains. While existing literatures have designed many sophisticated architectures based on RNNs, GNNs, or Transformers, another kind of approaches based on multi-layer perceptrons (MLPs) are proposed with simple structure, low complexity, and superior performance. However, most MLP-based forecasting methods suffer from the point-wise mappings and information bottleneck, which largely hinders the forecasting performance. To overcome this problem, we explore a novel direction of applying MLPs in the frequency domain for time series forecasting. We investigate the learned patterns of frequency-domain MLPs and discover their two inherent characteristic benefiting forecasting, (i) global view: frequency spectrum makes MLPs own a complete view for signals and learn global dependencies more easily, and (ii) energy compaction: frequency-domain MLPs concentrate on smaller key part of frequency components with compact signal energy. Then, we propose FreTS, a simple yet effective architecture built upon Frequency-domain MLPs for Time Series forecasting. FreTS mainly involves two stages, (i) Domain Conversion, that transforms time-domain signals into complex numbers of frequency domain; (ii) Frequency Learning, that performs our redesigned MLPs for the learning of real and imaginary part of frequency components. The above stages operated on both inter-series and intra-series scales further contribute to channel-wise and time-wise dependency learning. Extensive experiments on 13 real-world benchmarks (including 7 benchmarks for short-term forecasting and 6 benchmarks for long-term forecasting) demonstrate our consistent superiority over state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FreTS.

        ----

        ## [3349] Q-DM: An Efficient Low-bit Quantized Diffusion Model

        **Authors**: *Yanjing Li, Sheng Xu, Xianbin Cao, Xiao Sun, Baochang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f1ee1cca0721de55bb35cf28ab95e1b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f1ee1cca0721de55bb35cf28ab95e1b4-Abstract-Conference.html)

        **Abstract**:

        Denoising diffusion generative models are capable of generating high-quality data, but suffers from the computation-costly generation process, due to a iterative noise estimation using full-precision  networks. As an intuitive solution, quantization can significantly reduce the computational and memory consumption by low-bit parameters and operations. However, low-bit noise estimation networks in diffusion models (DMs) remain unexplored yet and perform much worse than the full-precision counterparts as observed in our experimental studies. In this paper, we first identify that the bottlenecks of low-bit quantized DMs come from a large distribution oscillation  on activations  and accumulated quantization error caused by the multi-step denoising process. To address these issues, we first develop a Timestep-aware Quantization (TaQ) method and a Noise-estimating Mimicking (NeM) scheme for low-bit quantized DMs (Q-DM) to effectively eliminate such oscillation and accumulated error respectively, leading to well-performed low-bit DMs. In this way, we propose an efficient Q-DM to calculate low-bit DMs by considering both training and inference process in the same framework. We evaluate our methods on popular DDPM and DDIM models. Extensive experimental results show that our method achieves a much better performance than the prior arts. For example, the 4-bit Q-DM theoretically accelerates the 1000-step DDPM by 7.8x and achieves a FID score of 5.17, on the unconditional CIFAR-10 dataset.

        ----

        ## [3350] Beyond Exponential Graph: Communication-Efficient Topologies for Decentralized Learning via Finite-time Convergence

        **Authors**: *Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, Makoto Yamada*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f201b3f3d0f08c6ab46c36b9052c1b64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f201b3f3d0f08c6ab46c36b9052c1b64-Abstract-Conference.html)

        **Abstract**:

        Decentralized learning has recently been attracting increasing attention for its applications in parallel computation and privacy preservation. Many recent studies stated that the underlying network topology with a faster consensus rate (a.k.a. spectral gap) leads to a better convergence rate and accuracy for decentralized learning. However, a topology with a fast consensus rate, e.g., the exponential graph, generally has a large maximum degree, which incurs significant communication costs. Thus, seeking topologies with both a fast consensus rate and small maximum degree is important. In this study, we propose a novel topology combining both a fast consensus rate and small maximum degree called the Base-$\left(k+1\right)$ Graph. Unlike the existing topologies, the Base-$\left(k+1\right)$ Graph enables all nodes to reach the exact consensus after a finite number of iterations for any number of nodes and maximum degree $k$. Thanks to this favorable property, the Base-$\left(k+1\right)$ Graph endows Decentralized SGD (DSGD) with both a faster convergence rate and more communication efficiency than the exponential graph. We conducted experiments with various topologies, demonstrating that the Base-$\left(k+1\right)$ Graph enables various decentralized learning methods to achieve higher accuracy with better communication efficiency than the existing topologies. Our code is available at https://github.com/yukiTakezawa/BaseGraph.

        ----

        ## [3351] Alternating Updates for Efficient Transformers

        **Authors**: *Cenk Baykal, Dylan J. Cutler, Nishanth Dikkala, Nikhil Ghosh, Rina Panigrahy, Xin Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2059277ac6ce66e7e5543001afa8bb5-Abstract-Conference.html)

        **Abstract**:

        It has been well established that increasing scale in deep transformer networks leads to improved quality and performance. However, this increase in scale often comes with prohibitive increases in compute cost and inference latency. We introduce Alternating Updates (AltUp), a simple-to-implement method to increase a model's capacity without the computational burden. AltUp enables the widening of the learned representation, i.e., the token embedding, while only incurring a negligible increase in latency. AltUp achieves this by working on a subblock of the widened representation at each layer and using a predict-and-correct mechanism to update the inactivated blocks. We present extensions of AltUp, such as its applicability to the sequence dimension, and demonstrate how AltUp can be synergistically combined with existing approaches, such as Sparse Mixture-of-Experts models, to obtain efficient models with even higher capacity. Our experiments on benchmark transformer models and language tasks demonstrate the consistent effectiveness of AltUp on a diverse set of scenarios. Notably, on SuperGLUE and SQuAD benchmarks, AltUp enables up to $87\%$ speedup relative to the dense baselines at the same accuracy.

        ----

        ## [3352] Interpretable Prototype-based Graph Information Bottleneck

        **Authors**: *Sangwoo Seo, Sungwon Kim, Chanyoung Park*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f224f056694bcfe465c5d84579785761-Abstract-Conference.html)

        **Abstract**:

        The success of Graph Neural Networks (GNNs) has led to a need for understanding their decision-making process and providing explanations for their predictions, which has given rise to explainable AI (XAI) that offers transparent explanations for black-box models. Recently, the use of prototypes has successfully improved the explainability of models by learning prototypes to imply training graphs that affect the prediction. However, these approaches tend to provide prototypes with excessive information from the entire graph, leading to the exclusion of key substructures or the inclusion of irrelevant substructures, which can limit both the interpretability and the performance of the model in downstream tasks. In this work, we propose a novel framework of explainable GNNs, called interpretable Prototype-based Graph Information Bottleneck (PGIB) that incorporates prototype learning within the information bottleneck framework to provide prototypes with the key subgraph from the input graph that is important for the model prediction. This is the first work that incorporates prototype learning into the process of identifying the key subgraphs that have a critical impact on the prediction performance. Extensive experiments, including qualitative analysis, demonstrate that PGIB outperforms state-of-the-art methods in terms of both prediction performance and explainability.

        ----

        ## [3353] Self-Chained Image-Language Model for Video Localization and Question Answering

        **Authors**: *Shoubin Yu, Jaemin Cho, Prateek Yadav, Mohit Bansal*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f22a9af8dbb348952b08bd58d4734b50-Abstract-Conference.html)

        **Abstract**:

        Recent studies have shown promising results on utilizing large pre-trained image-language models for video question answering. While these image-language models can efficiently bootstrap the representation learning of video-language models, they typically concatenate uniformly sampled video frames as visual inputs without explicit language-aware, temporal modeling. When only a portion of a video input is relevant to the language query, such uniform frame sampling can often lead to missing important visual cues. Although humans often find a video moment to focus on and rewind the moment to answer questions, training a query-aware video moment localizer often requires expensive annotations and high computational costs. To address this issue, we propose Self-Chained Video Localization-Answering (SeViLA), a novel framework that leverages a single image-language model (BLIP- 2) to tackle both temporal keyframe localization and question answering on videos. SeViLA framework consists of two modules: Localizer and Answerer, where both are parameter-efficiently fine-tuned from BLIP-2. We propose two ways of chaining these modules for cascaded inference and self-refinement. First, in the forward chain, the Localizer finds multiple language-aware keyframes in a video, which the Answerer uses to predict the answer. Second, in the reverse chain, the Answerer generates keyframe pseudo-labels to refine the Localizer, alleviating the need for expensive video moment localization annotations. Our SeViLA framework outperforms several strong baselines/previous works on five challenging video question answering and event prediction benchmarks, and achieves the state-of-the-art in both fine-tuning (NExT-QA and STAR) and zero-shot (NExT-QA, STAR, How2QA, and VLEP) settings. We show a comprehensive analysis of our framework, including the impact of Localizer, comparisons of Localizer with other temporal localization models, pre-training/self-refinement of Localizer, and varying the number of keyframes.

        ----

        ## [3354] The Tunnel Effect: Building Data Representations in Deep Neural Networks

        **Authors**: *Wojciech Masarczyk, Mateusz Ostaszewski, Ehsan Imani, Razvan Pascanu, Piotr Milos, Tomasz Trzcinski*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f249db9ab5975586f36df46f8958c008-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f249db9ab5975586f36df46f8958c008-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks are widely known for their remarkable effectiveness across various tasks, with the consensus that deeper networks implicitly learn more complex data representations. This paper shows that sufficiently deep networks trained for supervised image classification split into two distinct parts that contribute to the resulting data representations differently. The initial layers create linearly-separable representations, while the subsequent layers, which we refer to as \textit{the tunnel}, compress these representations and have a minimal impact on the overall performance. We explore the tunnel's behavior through comprehensive empirical studies, highlighting that it emerges early in the training process. Its depth depends on the relation between the network's capacity and task complexity. Furthermore, we show that the tunnel degrades out-of-distribution generalization and discuss its implications for continual learning.

        ----

        ## [3355] Restart Sampling for Improving Generative Processes

        **Authors**: *Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, Tommi S. Jaakkola*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2543511e5f4d4764857f9ad833a977d-Abstract-Conference.html)

        **Abstract**:

        Generative processes that involve solving differential equations, such as diffusion models, frequently necessitate balancing speed and quality. ODE-based samplers are fast but plateau in performance while SDE-based samplers deliver higher sample quality at the cost of increased sampling time.  We attribute this difference to sampling errors: ODE-samplers involve smaller discretization errors while stochasticity in SDE contracts accumulated errors. Based on these findings, we propose a novel sampling algorithm called \textit{Restart} in order to better balance discretization errors and contraction. The sampling method alternates between adding substantial noise in additional forward steps and strictly following a backward ODE. Empirically, Restart sampler surpasses previous SDE and ODE samplers in both speed and accuracy. Restart not only outperforms the previous best SDE results, but also accelerates the sampling speed by 10-fold / 2-fold on CIFAR-10 / ImageNet $64{\times} 64$. In addition, it attains significantly better sample quality than ODE samplers within comparable sampling times. Moreover, Restart better balances text-image alignment/visual quality versus diversity than previous samplers in the large-scale text-to-image Stable Diffusion model pre-trained on LAION $512{\times} 512$.  Code is available at https://github.com/Newbeeer/diffusion_restart_sampling

        ----

        ## [3356] Constructing Non-isotropic Gaussian Diffusion Model Using Isotropic Gaussian Diffusion Model for Image Editing

        **Authors**: *Xi Yu, Xiang Gu, Haozhi Liu, Jian Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f25602918e8a0d0c86e3c752ecfbbaa1-Abstract-Conference.html)

        **Abstract**:

        Score-based diffusion models (SBDMs) have achieved state-of-the-art results in image generation. In this paper, we propose a Non-isotropic Gaussian Diffusion Model (NGDM) for image editing, which requires editing the source image while preserving the image regions irrelevant to the editing task. We construct NGDM by adding independent Gaussian noises with different variances to different image pixels. Instead of specifically training the NGDM, we rectify the NGDM into an isotropic Gaussian diffusion model with different pixels having different total forward diffusion time. We propose to reverse the diffusion by designing a sampling method that starts at different time for different pixels for denoising to generate images using the pre-trained isotropic Gaussian diffusion model. Experimental results show that NGDM achieves state-of-the-art performance for image editing tasks, considering the trade-off between the fidelity to the source image and alignment with the desired editing target.

        ----

        ## [3357] Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models

        **Authors**: *Haonan Duan, Adam Dziedzic, Nicolas Papernot, Franziska Boenisch*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f26119b4ffe38c24d97e4c49d334b99e-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) are excellent in-context learners. However, the sensitivity of data contained in prompts raises privacy concerns. Our work first shows that these concerns are valid: we instantiate a simple but highly effective membership inference attack against the data used to prompt LLMs. To address this vulnerability, one could forego prompting and resort to fine-tuning LLMs with known algorithms for private gradient descent. However, this comes at the expense of the practicality and efficiency offered by prompting. Therefore, we propose to privately learn to prompt. We first show that soft prompts can be obtained privately through gradient descent on downstream data. However, this is not the case for discrete prompts. Thus, we orchestrate a noisy vote among an ensemble of LLMs presented with different prompts, i.e., a flock of stochastic parrots. The vote privately transfers the flock's knowledge into a single public prompt. We show that LLMs prompted with our private algorithms closely match the non-private baselines. For example, using GPT3 as the base model, we achieve a downstream accuracy of 92.7% on the sst2 dataset with $(\varepsilon=0.147, \delta=10^{-6})$-differential privacy vs. 95.2% for the non-private baseline. Through our experiments, we also show that our prompt-based approach is easily deployed with existing commercial~APIs.

        ----

        ## [3358] Dataset Diffusion: Diffusion-based Synthetic Data Generation for Pixel-Level Semantic Segmentation

        **Authors**: *Quang Nguyen, Truong Vu, Anh Tran, Khoi Nguyen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2957e48240c1d90e62b303574871b47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2957e48240c1d90e62b303574871b47-Abstract-Conference.html)

        **Abstract**:

        Preparing training data for deep vision models is a labor-intensive task. To address this, generative models have emerged as an effective solution for generating synthetic data. While current generative models produce image-level category labels, we propose a novel method for generating pixel-level semantic segmentation labels using the text-to-image generative model Stable Diffusion (SD). By utilizing the text prompts, cross-attention, and self-attention of SD, we introduce three new techniques: class-prompt appending, class-prompt cross-attention, and self-attention exponentiation. These techniques enable us to generate segmentation maps corresponding to synthetic images. These maps serve as pseudo-labels for training semantic segmenters, eliminating the need for labor-intensive pixel-wise annotation. To account for the imperfections in our pseudo-labels, we incorporate uncertainty regions into the segmentation, allowing us to disregard loss from those regions. We conduct evaluations on two datasets, PASCAL VOC and MSCOCO, and our approach significantly outperforms concurrent work. Our benchmarks and code will be released at https://github.com/VinAIResearch/Dataset-Diffusion.

        ----

        ## [3359] ASL Citizen: A Community-Sourced Dataset for Advancing Isolated Sign Language Recognition

        **Authors**: *Aashaka Desai, Lauren Berger, Fyodor Minakov, Nessa Milano, Chinmay Singh, Kriston Pumphrey, Richard E. Ladner, Hal Daumé III, Alex X. Lu, Naomi Caselli, Danielle Bragg*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f29cf8f8b4996a4a453ef366cf496354-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Sign languages are used as a primary language by approximately 70 million D/deaf people world-wide. However, most communication technologies operate in spoken and written languages, creating inequities in access. To help tackle this problem, we release ASL Citizen, the first crowdsourced Isolated Sign Language Recognition (ISLR) dataset, collected with consent and containing 83,399 videos for 2,731 distinct signs filmed by 52 signers in a variety of environments. We propose that this dataset be used for sign language dictionary retrieval for American Sign Language (ASL), where a user demonstrates a sign to their webcam to retrieve matching signs from a dictionary. We show that training supervised machine learning classifiers with our dataset advances the state-of-the-art on metrics relevant for dictionary retrieval, achieving 63\% accuracy and a recall-at-10 of 91\%, evaluated entirely on videos of users who are not present in the training or validation sets.

        ----

        ## [3360] Learning-to-Rank Meets Language: Boosting Language-Driven Ordering Alignment for Ordinal Classification

        **Authors**: *Rui Wang, Peipei Li, Huaibo Huang, Chunshui Cao, Ran He, Zhaofeng He*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2a11632520f4b7473d7838f074a7d25-Abstract-Conference.html)

        **Abstract**:

        We present a novel language-driven ordering alignment method for ordinal classification. The labels in ordinal classification contain additional ordering relations, making them prone to overfitting when relying solely on training data. Recent developments in pre-trained vision-language models inspire us to leverage the rich ordinal priors in human language by converting the original task into a vision-language alignment task. Consequently, we propose L2RCLIP, which fully utilizes the language priors from two perspectives. First, we introduce a complementary prompt tuning technique called RankFormer, designed to enhance the ordering relation of original rank prompts. It employs token-level attention with residual-style prompt blending in the word embedding space. Second, to further incorporate language priors, we revisit the approximate bound optimization of vanilla cross-entropy loss and restructure it within the cross-modal embedding space. Consequently, we propose a cross-modal ordinal pairwise loss to refine the CLIP feature space, where texts and images maintain both semantic alignment and ordering alignment.  Extensive experiments on three ordinal classification tasks, including facial age estimation, historical color image (HCI) classification, and aesthetic assessment demonstrate its promising performance.

        ----

        ## [3361] GAUCHE: A Library for Gaussian Processes in Chemistry

        **Authors**: *Ryan-Rhys Griffiths, Leo Klarner, Henry B. Moss, Aditya Ravuri, Sang Truong, Yuanqi Du, Samuel Stanton, Gary Tom, Bojana Rankovic, Arian Rokkum Jamasb, Aryan Deshwal, Julius Schwartz, Austin Tripp, Gregory Kell, Simon Frieder, Anthony Bourached, Alex Chan, Jacob Moss, Chengzhi Guo, Johannes Peter Dürholt, Saudamini Chaurasia, Ji Won Park, Felix Strieth-Kalthoff, Alpha A. Lee, Bingqing Cheng, Alán Aspuru-Guzik, Philippe Schwaller, Jian Tang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2b1b2e974fa5ea622dd87f22815f423-Abstract-Conference.html)

        **Abstract**:

        We introduce GAUCHE, an open-source library for GAUssian processes in CHEmistry. Gaussian processes have long been a cornerstone of probabilistic machine learning, affording particular advantages for uncertainty quantification and Bayesian optimisation. Extending Gaussian processes to molecular representations, however, necessitates kernels defined over structured inputs such as graphs, strings and bit vectors. By providing such kernels in a modular, robust and easy-to-use framework, we seek to enable expert chemists and materials scientists to make use of state-of-the-art black-box optimization techniques. Motivated by scenarios frequently encountered in practice, we showcase applications for GAUCHE in molecular discovery, chemical reaction optimisation and protein design. The codebase is made available at https://github.com/leojklarner/gauche.

        ----

        ## [3362] Exponentially Convergent Algorithms for Supervised Matrix Factorization

        **Authors**: *Joowon Lee, Hanbaek Lyu, Weixin Yao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2c80b3c9cf8102d38c4b21af25d9740-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2c80b3c9cf8102d38c4b21af25d9740-Abstract-Conference.html)

        **Abstract**:

        Supervised matrix factorization (SMF) is a classical machine learning method that simultaneously seeks feature extraction and classification tasks, which are not necessarily a priori aligned objectives. Our goal is to use SMF to learn low-rank latent factors that offer interpretable, data-reconstructive, and class-discriminative features, addressing challenges posed by high-dimensional data. Training SMF model involves solving a nonconvex and possibly constrained optimization with at least three blocks of parameters. Known algorithms are either heuristic or provide weak convergence guarantees for special cases. In this paper, we provide a novel framework that `lifts' SMF as a low-rank matrix estimation problem in a combined factor space and propose an efficient algorithm that provably converges exponentially fast to a global minimizer of the objective with arbitrary initialization under mild assumptions. Our framework applies to a wide range of SMF-type problems for multi-class classification with auxiliary features. To showcase an application, we demonstrate that our algorithm successfully identified well-known cancer-associated gene groups for various cancers.

        ----

        ## [3363] InfoCD: A Contrastive Chamfer Distance Loss for Point Cloud Completion

        **Authors**: *Fangzhou Lin, Yun Yue, Ziming Zhang, Songlin Hou, Kazunori D. Yamada, Vijaya Kolachalama, Venkatesh Saligrama*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f2ea1943896474b7cd9796b93e526f6f-Abstract-Conference.html)

        **Abstract**:

        A point cloud is a discrete set of data points sampled from a 3D geometric surface. Chamfer distance (CD) is a popular metric and training loss to measure the distances between point clouds, but also well known to be sensitive to outliers. To address this issue, in this paper we propose InfoCD, a novel contrastive Chamfer distance loss to learn to spread the matched points for better distribution alignments between point clouds as well as accounting for a surface similarity estimator. We show that minimizing InfoCD is equivalent to maximizing a lower bound of the mutual information between the underlying geometric surfaces represented by the point clouds, leading to a regularized CD metric which is robust and computationally efficient for deep learning. We conduct comprehensive experiments for point cloud completion using InfoCD and observe significant improvements consistently over all the popular baseline networks trained with CD-based losses, leading to new state-of-the-art results on several benchmark datasets. Demo code is available at https://github.com/Zhang-VISLab/NeurIPS2023-InfoCD.

        ----

        ## [3364] Differentially Private Statistical Inference through β-Divergence One Posterior Sampling

        **Authors**: *Jack Jewson, Sahra Ghalebikesabi, Chris C. Holmes*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3024ea88cec9f45a411cf4d51ab649c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3024ea88cec9f45a411cf4d51ab649c-Abstract-Conference.html)

        **Abstract**:

        Differential privacy guarantees allow the results of a statistical analysis involving sensitive data to be released without compromising the privacy of any individual taking part. Achieving such guarantees generally requires the injection of noise, either directly into parameter estimates or into the estimation process. Instead of artificially introducing perturbations, sampling from Bayesian posterior distributions has been shown to be a special case of the exponential mechanism, producing consistent,and efficient private estimates without altering the data generative process. The application of current approaches has, however, been limited by their strong bounding assumptions which do not hold for basic models, such as simple linear regressors.To ameliorate this, we propose $\beta$D-Bayes, a posterior sampling  scheme from a generalised posterior targeting the minimisation of the $\beta$-divergence between the model and the data generating process. This provides private estimation that is generally applicable without requiring changes to the underlying model and consistently learns the data generating parameter. We show that $\beta$D-Bayes produces more precise inference estimation for the same privacy guarantees, and further facilitates differentially private estimation of complex classifiers, and continuous regression models such as neural networks, which goes beyond what has been currently possible with private posterior sampling.

        ----

        ## [3365] Doubly Robust Augmented Transfer for Meta-Reinforcement Learning

        **Authors**: *Yuankun Jiang, Nuowen Kan, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f31bf160569618084ba9bdc2a8de29d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f31bf160569618084ba9bdc2a8de29d0-Abstract-Conference.html)

        **Abstract**:

        Meta-reinforcement learning (Meta-RL), though enabling a fast adaptation to learn new skills by exploiting the common structure shared among different tasks, suffers performance degradation in the sparse-reward setting. Current hindsight-based sample transfer approaches can alleviate this issue by transferring relabeled trajectories from other tasks to a new task so as to provide informative experience for the target reward function, but are unfortunately constrained with the unrealistic assumption that tasks differ only in reward functions. In this paper, we propose a doubly robust augmented transfer (DRaT) approach, aiming at addressing the more general sparse reward meta-RL scenario with both dynamics mismatches and varying reward functions across tasks. Specifically, we design a doubly robust augmented estimator for efficient value-function evaluation, which tackles dynamics mismatches with the optimal importance weight of transition distributions achieved by minimizing the theoretically derived upper bound of mean squared error (MSE) between the estimated values of transferred samples and their true values in the target task. Due to its intractability, we then propose an interval-based approximation to this optimal importance weight, which is guaranteed to cover the optimum with a constrained and sample-independent upper bound on the MSE approximation error. Based on our theoretical findings, we finally develop a DRaT algorithm for transferring informative samples across tasks during the training of meta-RL. We implement DRaT on an off-policy meta-RL baseline, and empirically show that it significantly outperforms other hindsight-based approaches on various sparse-reward MuJoCo locomotion tasks with varying dynamics and reward functions.

        ----

        ## [3366] Evaluating Open-QA Evaluation

        **Authors**: *Cunxiang Wang, Sirui Cheng, Qipeng Guo, Yuanhao Yue, Bowen Ding, Zhikun Xu, Yidong Wang, Xiangkun Hu, Zheng Zhang, Yue Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f323d594aa5d2c68154433a131c07959-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f323d594aa5d2c68154433a131c07959-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        This study focuses on the evaluation of the Open Question Answering (Open-QA) task, which can directly estimate the factuality of large language models (LLMs). Current automatic evaluation methods have shown limitations, indicating that human evaluation still remains the most reliable approach. We introduce a new task, QA Evaluation (QA-Eval) and the corresponding dataset EVOUNA, designed to assess the accuracy of AI-generated answers in relation to standard answers within Open-QA. Our evaluation of these methods utilizes human-annotated results to measure their performance. Specifically, the work investigates methods that show high correlation with human evaluations, deeming them more reliable. We also discuss the pitfalls of current methods and methods to improve LLM-based evaluators. We believe this new QA-Eval task and corresponding dataset EVOUNA will facilitate the development of more effective automatic evaluation tools and prove valuable for future research in this area. All resources are available at https://github.com/wangcunxiang/QA-Eval and it is under the Apache-2.0 License.

        ----

        ## [3367] Efficiently incorporating quintuple interactions into geometric deep learning force fields

        **Authors**: *Zun Wang, Guoqing Liu, Yichi Zhou, Tong Wang, Bin Shao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f32b13bfc384b3b1d52d675b05f2bece-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f32b13bfc384b3b1d52d675b05f2bece-Abstract-Conference.html)

        **Abstract**:

        Machine learning force fields (MLFFs) have instigated a groundbreaking shift in molecular dynamics (MD) simulations across a wide range of fields, such as physics, chemistry, biology, and materials science. Incorporating higher order many-body interactions can enhance the expressiveness and accuracy of models. Recent models have achieved this by explicitly including up to four-body interactions. However, five-body interactions, which have relevance in various fields, are still challenging to incorporate efficiently into MLFFs. In this work, we propose the quintuple network (QuinNet), an end-to-end graph neural network that efficiently expresses many-body interactions up to five-body interactions with \emph{ab initio} accuracy. By analyzing the topology of diverse many-body interactions, we design the model architecture to efficiently and explicitly represent these interactions. We evaluate QuinNet on public datasets of small molecules, such as MD17 and its revised version, and show that it is compatible with other state-of-the-art models on these benchmarks. Moreover, QuinNet surpasses many leading models on larger and more complex molecular systems, such as MD22 and Chignolin, without increasing the computational complexity. We also use QuinNet as a force field for molecular dynamics (MD) simulations to demonstrate its accuracy and stability, and conduct an ablation study to elucidate the significance of five-body interactions. We open source our implementation at https://github.com/Zun-Wang/QuinNet.

        ----

        ## [3368] Spectral Entry-wise Matrix Estimation for Low-Rank Reinforcement Learning

        **Authors**: *Stefan Stojanovic, Yassir Jedra, Alexandre Proutière*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f334c3375bd3744e98a0ca8eaa2403b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f334c3375bd3744e98a0ca8eaa2403b0-Abstract-Conference.html)

        **Abstract**:

        We study matrix estimation problems arising in reinforcement learning with low-rank structure. In low-rank bandits, the matrix to be recovered specifies the expected arm rewards, and for low-rank Markov Decision Processes (MDPs), it characterizes the transition kernel of the MDP. In both cases, each entry of the matrix carries important information, and we seek estimation methods with low entry-wise prediction error. Importantly, these methods further need to accommodate for inherent correlations in the available data (e.g. for MDPs, the data consists of system trajectories). We investigate the performance of  simple spectral-based matrix estimation approaches: we show that they efficiently recover the singular subspaces of the matrix and exhibit nearly-minimal entry-wise prediction error. These new results on low-rank matrix estimation make it possible to devise reinforcement learning algorithms that fully exploit the underlying low-rank structure. We provide two examples of such algorithms: a regret minimization algorithm for low-rank bandit problems, and a best policy identification algorithm for low-rank MDPs. Both algorithms yield state-of-the-art performance guarantees.

        ----

        ## [3369] Function Space Bayesian Pseudocoreset for Bayesian Neural Networks

        **Authors**: *Balhae Kim, Hyungi Lee, Juho Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f36a180277bd3d5781dc02245f9d5f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f36a180277bd3d5781dc02245f9d5f52-Abstract-Conference.html)

        **Abstract**:

        A Bayesian pseudocoreset is a compact synthetic dataset summarizing essential information of a large-scale dataset and thus can be used as a proxy dataset for scalable Bayesian inference. Typically, a Bayesian pseudocoreset is constructed by minimizing a divergence measure between the posterior conditioning on the pseudocoreset and the posterior conditioning on the full dataset. However, evaluating the divergence can be challenging, particularly for the models like deep neural networks having high-dimensional parameters. In this paper, we propose a novel Bayesian pseudocoreset construction method that operates on a function space. Unlike previous methods, which construct and match the coreset and full data posteriors in the space of model parameters (weights), our method constructs variational approximations to the coreset posterior on a function space and matches it to the full data posterior in the function space. By working directly on the function space, our method could bypass several challenges that may arise when working on a weight space, including limited scalability and multi-modality issue. Through various experiments, we demonstrate that the Bayesian pseudocoresets constructed from our method enjoys enhanced uncertainty quantification and better robustness across various model architectures.

        ----

        ## [3370] One-step differentiation of iterative algorithms

        **Authors**: *Jérôme Bolte, Edouard Pauwels, Samuel Vaiter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3716db40060004d0629d4051b2c57ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3716db40060004d0629d4051b2c57ab-Abstract-Conference.html)

        **Abstract**:

        In appropriate frameworks, automatic differentiation is transparent to the user, at the cost of being a significant computational burden when the number of operations is large. For iterative algorithms, implicit differentiation alleviates this issue but requires custom implementation of Jacobian evaluation. In this paper, we study one-step differentiation, also known as Jacobian-free backpropagation, a method as easy as automatic differentiation and as performant as implicit differentiation for fast algorithms (e.g. superlinear optimization methods). We provide a complete theoretical approximation analysis with specific examples (Newton's method, gradient descent) along with its consequences in bilevel optimization. Several numerical examples illustrate the well-foundness of the one-step estimator.

        ----

        ## [3371] Adaptive Principal Component Regression with Applications to Panel Data

        **Authors**: *Anish Agarwal, Keegan Harris, Justin Whitehouse, Steven Z. Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f37265d7493377170a3b4ba91823119a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f37265d7493377170a3b4ba91823119a-Abstract-Conference.html)

        **Abstract**:

        Principal component regression (PCR) is a popular technique for fixed-design error-in-variables regression, a generalization of the linear regression setting in which the observed covariates are corrupted with random noise. We provide the first time-uniform finite sample guarantees for online (regularized) PCR whenever data is collected adaptively. Since the proof techniques for PCR in the fixed design setting do not readily extend to the online setting, our results rely on adapting tools from modern martingale concentration to the error-in-variables setting. As an application of our bounds, we provide a framework for counterfactual estimation of unit-specific treatment effects in panel data settings when interventions are assigned adaptively. Our framework may be thought of as a generalization of the synthetic interventions framework where data is collected via an adaptive intervention assignment policy.

        ----

        ## [3372] VisAlign: Dataset for Measuring the Alignment between AI and Humans in Visual Perception

        **Authors**: *Jiyoung Lee, Seungho Kim, Seunghyun Won, Joonseok Lee, Marzyeh Ghassemi, James Thorne, Jaeseok Choi, O.-Kil Kwon, Edward Choi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f37aba0f53fdb59f53254fe9098b2177-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f37aba0f53fdb59f53254fe9098b2177-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        AI alignment refers to models acting towards human-intended goals, preferences, or ethical principles. Analyzing the similarity between models and humans can be a proxy measure for ensuring AI safety. In this paper, we focus on the models' visual perception alignment with humans, further referred to as AI-human visual alignment. Specifically, we propose a new dataset for measuring AI-human visual alignment in terms of image classification. In order to evaluate AI-human visual alignment, a dataset should encompass samples with various scenarios and have gold human perception labels. Our dataset consists of three groups of samples, namely Must-Act (i.e., Must-Classify), Must-Abstain, and Uncertain, based on the quantity and clarity of visual information in an image and further divided into eight categories. All samples have a gold human perception label; even Uncertain (e.g., severely blurry) sample labels were obtained via crowd-sourcing. The validity of our dataset is verified by sampling theory, statistical theories related to survey design, and experts in the related fields. Using our dataset, we analyze the visual alignment and reliability of five popular visual perception models and seven abstention methods. Our code and data is available at https://github.com/jiyounglee-0523/VisAlign.

        ----

        ## [3373] The Best of Both Worlds in Network Population Games: Reaching Consensus and Convergence to Equilibrium

        **Authors**: *Shuyue Hu, Harold Soh, Georgios Piliouras*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f39931608cdc52d7d9f8ba7003af9136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f39931608cdc52d7d9f8ba7003af9136-Abstract-Conference.html)

        **Abstract**:

        Reaching consensus and convergence to equilibrium are two major challenges of multi-agent systems. Although each has attracted significant attention, relatively few studies address both challenges at the same time. This paper examines the connection between the notions of consensus and equilibrium in a multi-agent system where multiple interacting sub-populations coexist. We argue that consensus can be seen as an intricate component of intra-population stability, whereas equilibrium can be seen as encoding inter-population stability. We show that smooth fictitious play, a well-known learning model in game theory, can achieve both consensus and convergence to equilibrium in diverse multi-agent settings. Moreover, we show that the consensus formation process plays a crucial role in the seminal thorny problem of equilibrium selection in multi-agent learning.

        ----

        ## [3374] L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors

        **Authors**: *Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html)

        **Abstract**:

        Language-based colorization produces plausible and visually pleasing colors under the guidance of user-friendly natural language descriptions. Previous methods implicitly assume that users provide comprehensive color descriptions for most of the objects in the image, which leads to suboptimal performance. In this paper, we propose a unified model to perform language-based colorization with any-level descriptions. We leverage the pretrained cross-modality generative model for its robust language understanding and rich color priors to handle the inherent ambiguity of any-level descriptions. We further design modules to align with input conditions to preserve local spatial structures and prevent the ghosting effect. With the proposed novel sampling strategy, our model achieves instance-aware colorization in diverse and complex scenarios. Extensive experimental results demonstrate our advantages of effectively handling any-level descriptions and outperforming both language-based and automatic colorization methods. The code and pretrained modelsare available at: https://github.com/changzheng123/L-CAD.

        ----

        ## [3375] Convolutional Neural Operators for robust and accurate learning of PDEs

        **Authors**: *Bogdan Raonic, Roberto Molinaro, Tim De Ryck, Tobias Rohner, Francesca Bartolucci, Rima Alaifari, Siddhartha Mishra, Emmanuel de Bézenac*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html)

        **Abstract**:

        Although very successfully used in conventional machine learning, convolution based neural network architectures -- believed to be inconsistent in function space -- have been largely ignored in the context of learning solution operators of PDEs. Here, we present novel adaptations for convolutional neural networks to demonstrate that they are indeed able to process functions as inputs and outputs. The resulting architecture, termed as convolutional neural operators (CNOs), is designed specifically to preserve its underlying continuous nature, even when implemented in a discretized form on a computer. We prove a universality theorem to show that CNOs can approximate operators arising in PDEs to desired accuracy. CNOs are tested on a novel suite of benchmarks, encompassing a diverse set of PDEs with multi-scale solutions and are observed to significantly outperform baselines, paving the way for an alternative framework for robust and accurate operator learning.

        ----

        ## [3376] Neural Image Compression: Generalization, Robustness, and Spectral Biases

        **Authors**: *Kelsey Lieberman, James Diffenderfer, Charles Godfrey, Bhavya Kailkhura*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3c5e56274140e0420baa3916c529210-Abstract-Conference.html)

        **Abstract**:

        Recent advances in neural image compression (NIC) have produced models that are starting to outperform classic codecs. While this has led to growing excitement about using NIC in real-world applications, the successful adoption of any machine learning system in the wild requires it to generalize (and be robust) to unseen distribution shifts at deployment. Unfortunately, current research lacks comprehensive datasets and informative tools to evaluate and understand NIC performance in real-world settings. To bridge this crucial gap, first, this paper presents a comprehensive benchmark suite to evaluate the out-of-distribution (OOD) performance of image compression methods. Specifically, we provide CLIC-C and Kodak-C by introducing 15 corruptions to the popular CLIC and Kodak benchmarks. Next, we propose spectrally-inspired inspection tools to gain deeper insight into errors introduced by image compression methods as well as their OOD performance. We then carry out a detailed performance comparison of several classic codecs and NIC variants, revealing intriguing findings that challenge our current understanding of the strengths and limitations of NIC. Finally, we corroborate our empirical findings with theoretical analysis, providing an in-depth view of the OOD performance of NIC and its dependence on the spectral properties of the data. Our benchmarks, spectral inspection tools, and findings provide a crucial bridge to the real-world adoption of NIC. We hope that our work will propel future efforts in designing robust and generalizable NIC methods. Code and data will be made available at https://github.com/klieberman/ood_nic.

        ----

        ## [3377] Estimating Koopman operators with sketching to provably learn large scale dynamical systems

        **Authors**: *Giacomo Meanti, Antoine Chatalic, Vladimir Kostic, Pietro Novelli, Massimiliano Pontil, Lorenzo Rosasco*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3d1e34a15c0af0954ae36a7f811c754-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3d1e34a15c0af0954ae36a7f811c754-Abstract-Conference.html)

        **Abstract**:

        The theory of Koopman operators allows to deploy non-parametric machine learning algorithms to predict and analyze complex dynamical systems.Estimators such as principal component regression (PCR) or reduced rank regression (RRR) in kernel spaces can be shown to provably learn Koopman operators from finite empirical observations of the system's time evolution. Scaling these approaches to very long trajectories is a challenge and requires introducing suitable approximations to make computations feasible. In this paper, we boost the efficiency of different kernel-based Koopman operator estimators using random projections (sketching).We derive, implement and test the new ``sketched'' estimators with extensive experiments on synthetic and large-scale molecular dynamics datasets. Further, we establish non asymptotic error bounds giving a sharp characterization of the trade-offs between statistical learning rates and computational efficiency.Our empirical and theoretical analysis shows that the proposed estimators provide a sound and efficient way to learn large scale dynamical systems.In particular our experiments indicate that the proposed estimators retain the same accuracy of PCR or RRR, while being much faster.

        ----

        ## [3378] Self-Adaptive Motion Tracking against On-body Displacement of Flexible Sensors

        **Authors**: *Chengxu Zuo, Jiawei Fang, Shihui Guo, Yipeng Qin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3da4165893c2465fd7e8df453c41ffa-Abstract-Conference.html)

        **Abstract**:

        Flexible sensors are promising for ubiquitous sensing of human status due to their flexibility and easy integration as wearable systems. However, on-body displacement of sensors is inevitable since the device cannot be firmly worn at a fixed position across different sessions. This displacement issue causes complicated patterns and significant challenges to subsequent machine learning algorithms. Our work proposes a novel self-adaptive motion tracking network to address this challenge. Our network consists of three novel components: i) a light-weight learnable Affine Transformation layer whose parameters can be tuned to efficiently adapt to unknown displacements; ii) a Fourier-encoded LSTM network for better pattern identification; iii) a novel sequence discrepancy loss equipped with auxiliary regressors for unsupervised tuning of Affine Transformation parameters.

        ----

        ## [3379] Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning

        **Authors**: *Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, Xiangyang Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f3f2ff9579ba6deeb89caa2fe1f0b99c-Abstract-Conference.html)

        **Abstract**:

        Offline multi-agent reinforcement learning is challenging due to the coupling effect of both distribution shift issue common in offline setting and the high dimension issue common in multi-agent setting, making the action out-of-distribution (OOD) and value overestimation phenomenon excessively severe. To mitigate this problem, we propose a novel multi-agent offline RL algorithm, named CounterFactual Conservative Q-Learning (CFCQL) to conduct conservative value estimation. Rather than regarding all the agents as a high dimensional single one and directly applying single agent conservative methods to it, CFCQL calculates conservative regularization for each agent separately in a counterfactual way and then linearly combines them to realize an overall conservative value estimation. We prove that it still enjoys the underestimation property and the performance guarantee as those single agent conservative methods do, but the induced regularization and safe policy improvement bound are independent of the agent number, which is therefore theoretically superior to the direct treatment referred to above, especially when the agent number is large. We further conduct experiments on four environments including both discrete and continuous action settings on both existing and our man-made datasets, demonstrating that CFCQL outperforms existing methods on most datasets and even with a remarkable margin on some of them.

        ----

        ## [3380] Black-Box Differential Privacy for Interactive ML

        **Authors**: *Haim Kaplan, Yishay Mansour, Shay Moran, Kobbi Nissim, Uri Stemmer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f418594e90047a10f4c158f70d6701cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f418594e90047a10f4c158f70d6701cc-Abstract-Conference.html)

        **Abstract**:

        In this work we revisit an interactive variant of joint differential privacy, recently introduced by Naor et al. [2023], and generalize it towards handling online processes in which existing privacy definitions seem too restrictive. We study basic properties of this definition and demonstrate that it satisfies (suitable variants) of group privacy, composition, and post processing.In order to demonstrate the advantages of this privacy definition compared to traditional forms of differential privacy,we consider the basic setting of online classification. We show that any (possibly non-private) learning rule can be effectively transformed to a private learning rule with only a polynomial overhead in the mistake bound. This demonstrates a stark difference with traditional forms of differential privacy, such as the one studied  by Golowich and Livni [2021], where only a double exponential overhead in the mistake bound is known (via an information theoretic upper bound).

        ----

        ## [3381] Mnemosyne: Learning to Train Transformers with Transformers

        **Authors**: *Deepali Jain, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Sumeet Singh, Vikas Sindhwani, Tingnan Zhang, Jie Tan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f41b6e5af73421e46ceed9cb036e72e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f41b6e5af73421e46ceed9cb036e72e7-Abstract-Conference.html)

        **Abstract**:

        In this work, we propose a new class of learnable optimizers, called Mnemosyne. It is based on the novel spatio-temporal low-rank implicit attention Transformers that can learn to train entire neural network architectures, including other Transformers, without any task-specific optimizer tuning. We show that Mnemosyne: (a) outperforms popular LSTM optimizers (also with new feature engineering to mitigate catastrophic forgetting of LSTMs), (b) can successfully train Transformers while using simple meta-training strategies that require minimal computational resources, (c) matches accuracy-wise SOTA hand-designed optimizers with carefully tuned hyper-parameters (often producing top performing models). Furthermore, Mnemosyne provides space complexity comparable to that of its hand-designed first-order counterparts, which allows it to scale to training larger sets of parameters. We conduct an extensive empirical evaluation of Mnemosyne on: (a) fine-tuning a wide range of Vision Transformers (ViTs) from medium-size architectures to massive ViT-Hs (36 layers, 16 heads), (b) pre-training BERT models and (c) soft prompt-tuning large 11B+ T5XXL models. We complement our results with a comprehensive theoretical analysis of the compact associative memory used by Mnemosyne which we believe was never done before.

        ----

        ## [3382] M2Hub: Unlocking the Potential of Machine Learning for Materials Discovery

        **Authors**: *Yuanqi Du, Yingheng Wang, Yining Huang, Jianan Canal Li, Yanqiao Zhu, Tian Xie, Chenru Duan, John M. Gregoire, Carla Pedro Gomes*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f43380ca3f86cd989f3269583c3c8b55-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f43380ca3f86cd989f3269583c3c8b55-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce M$^2$Hub, a toolkit for advancing machine learning in materials discovery. Machine learning has achieved remarkable progress in modeling molecular structures, especially biomolecules for drug discovery. However, the development of machine learning approaches for modeling materials structures lag behind, which is partly due to the lack of an integrated platform that enables access to diverse tasks for materials discovery. To bridge this gap, M$^2$Hub will enable easy access to materials discovery tasks, datasets, machine learning methods, evaluations, and benchmark results that cover the entire workflow. Specifically, the first release of M$^2$Hub focuses on three key stages in materials discovery: virtual screening, inverse design, and molecular simulation, including 9 datasets that covers 6 types of materials with 56 tasks across 8 types of material properties. We further provide 2 synthetic datasets for the purpose of generative tasks on materials. In addition to random data splits, we also provide 3 additional data partitions to reflect the real-world materials discovery scenarios. State-of-the-art machine learning methods (including those are suitable for materials structures but never compared in the literature) are benchmarked on representative tasks. Our codes and library are publicly available at \url{https://github.com/yuanqidu/M2Hub}.

        ----

        ## [3383] PoET: A generative model of protein families as sequences-of-sequences

        **Authors**: *Timothy F. Truong Jr., Tristan Bepler*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4366126eba252699b280e8f93c0ab2f-Abstract-Conference.html)

        **Abstract**:

        Generative protein language models are a natural way to design new proteins with desired functions. However, current models are either difficult to direct to produce a protein from a specific family of interest, or must be trained on a large multiple sequence alignment (MSA) from the specific family of interest, making them unable to benefit from transfer learning across families. To address this, we propose Protein Evolutionary Transformer (PoET), an autoregressive generative model of whole protein families that learns to generate sets of related proteins as sequences-of-sequences across tens of millions of natural protein sequence clusters. PoET can be used as a retrieval-augmented language model to generate and score arbitrary modifications conditioned on any protein family of interest, and can extrapolate from short context lengths to generalize well even for small families. This is enabled by a unique Transformer layer; we model tokens sequentially within sequences while attending between sequences order invariantly, allowing PoET to scale to context lengths beyond those used during training. In extensive experiments on deep mutational scanning datasets, we show that PoET outperforms existing protein language models and evolutionary sequence models for variant function prediction across proteins of all MSA depths. We also demonstrate PoET's ability to controllably generate new protein sequences.

        ----

        ## [3384] BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization

        **Authors**: *Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f445ba15f0f05c26e1d24f908ea78d60-Abstract-Conference.html)

        **Abstract**:

        Despite the success of neural-based combinatorial optimization methods for end-to-end heuristic learning, out-of-distribution generalization remains a challenge. In this paper, we present a novel formulation of Combinatorial Optimization Problems (COPs) as Markov Decision Processes (MDPs) that effectively leverages common symmetries of COPs to improve out-of-distribution robustness. Starting from a direct MDP formulation of a constructive method, we introduce a generic way to reduce the state space, based on Bisimulation Quotienting (BQ) in MDPs. Then, for COPs with a recursive nature, we specialize the bisimulation and show how the reduced state exploits the symmetries of these problems and facilitates MDP solving. Our approach is principled and we prove that an optimal policy for the proposed BQ-MDP actually solves the associated COPs. We illustrate our approach on five classical problems: the Euclidean and Asymmetric Traveling Salesman, Capacitated Vehicle Routing, Orienteering and Knapsack Problems. Furthermore, for each problem, we introduce a simple attention-based policy network for the BQ-MDPs, which we train by imitation of (near) optimal solutions of small instances from a single distribution. We obtain new state-of-the-art results for the five COPs on both synthetic and realistic benchmarks. Notably, in contrast to most existing neural approaches, our learned policies show excellent generalization performance to much larger instances than seen during training, without any additional search procedure. Our code is available at: link.

        ----

        ## [3385] Turbulence in Focus: Benchmarking Scaling Behavior of 3D Volumetric Super-Resolution with BLASTNet 20 Data

        **Authors**: *Wai Tong Chung, Bassem Akoush, Pushan Sharma, Alex Tamkin, Ki Sung Jung, Jacqueline Chen, Jack Guo, Davy Brouzet, Mohsen Talei, Bruno Savard, Alexei Y. Poludnenko, Matthias Ihme*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f458af2455b1e12608c2a16c308d663d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f458af2455b1e12608c2a16c308d663d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Analysis of compressible turbulent flows is essential for applications related to propulsion, energy generation, and the environment. Here, we present BLASTNet 2.0, a 2.2 TB network-of-datasets containing 744 full-domain samples from 34 high-fidelity direct numerical simulations, which addresses the current limited availability of 3D high-fidelity reacting and non-reacting compressible turbulent flow simulation data.  With this data, we benchmark a total of 49 variations of five deep learning approaches for 3D super-resolution - which can be applied for improving scientific imaging, simulations, turbulence models, as well as in computer vision  applications. We perform neural scaling analysis on these models to examine the performance of different machine learning (ML) approaches, including two scientific ML techniques. We demonstrate that (i) predictive performance can scale with model size and cost, (ii) architecture matters significantly, especially for smaller models, and (iii) the benefits of physics-based losses can persist with increasing model size. The outcomes of this benchmark study are anticipated to offer insights that can aid the design of 3D super-resolution models, especially for turbulence models, while this data is expected to foster ML methods for a broad range of flow physics applications. This data is publicly available with download links and browsing tools consolidated at https://blastnet.github.io.

        ----

        ## [3386] Neural Functional Transformers

        **Authors**: *Allan Zhou, Kaien Yang, Yiding Jiang, Kaylee Burns, Winnie Xu, Samuel Sokota, J. Zico Kolter, Chelsea Finn*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4757db82a02eea015670ecca605d5cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4757db82a02eea015670ecca605d5cc-Abstract-Conference.html)

        **Abstract**:

        The recent success of neural networks as implicit representation of data has driven growing interest in neural functionals: models that can process other neural networks as input by operating directly over their weight spaces. Nevertheless, constructing expressive and efficient neural functional architectures that can handle high-dimensional weight-space objects remains challenging. This paper uses the attention mechanism to define a novel set of permutation equivariant weight-space layers and composes them into deep equivariant models called neural functional Transformers (NFTs). NFTs respect weight-space permutation symmetries while incorporating the advantages of attention, which have exhibited remarkable success across multiple domains. In experiments processing the weights of feedforward MLPs and CNNs, we find that NFTs match or exceed the performance of prior weight-space methods. We also leverage NFTs to develop Inr2Array, a novel method for computing permutation invariant latent representations from the weights of implicit neural representations (INRs). Our proposed method improves INR classification accuracy by up to $+17\\%$ over existing methods. We provide an implementation of our layers at https://github.com/AllanYangZhou/nfn.

        ----

        ## [3387] LinkerNet: Fragment Poses and Linker Co-Design with 3D Equivariant Diffusion

        **Authors**: *Jiaqi Guan, Xingang Peng, Peiqi Jiang, Yunan Luo, Jian Peng, Jianzhu Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4821075019a058700f6e6738eea1365-Abstract-Conference.html)

        **Abstract**:

        Targeted protein degradation techniques, such as PROteolysis TArgeting Chimeras (PROTACs), have emerged as powerful tools for selectively removing disease-causing proteins. One challenging problem in this field is designing a linker to connect different molecular fragments to form a stable drug-candidate molecule. Existing models for linker design assume that the relative positions of the fragments are known, which may not be the case in real scenarios. In this work, we address a more general problem where the poses of the fragments are unknown in 3D space. We develop a 3D equivariant diffusion model that jointly learns the generative process of both fragment poses and the 3D structure of the linker. By viewing fragments as rigid bodies, we design a fragment pose prediction module inspired by the Newton-Euler equations in rigid body mechanics. Empirical studies on ZINC and PROTAC-DB datasets demonstrate that our model can generate chemically valid, synthetically-accessible,  and low-energy molecules under both unconstrained and constrained generation settings.

        ----

        ## [3388] One Risk to Rule Them All: A Risk-Sensitive Perspective on Model-Based Offline Reinforcement Learning

        **Authors**: *Marc Rigter, Bruno Lacerda, Nick Hawes*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f49287371916715b9209fa41a275851e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f49287371916715b9209fa41a275851e-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (RL) is suitable for safety-critical domains where online exploration is not feasible. In such domains, decision-making should take into consideration the risk of catastrophic outcomes. In other words, decision-making should be risk-averse. An additional challenge of offline RL is avoiding distributional shift, i.e. ensuring that  state-action pairs visited by the policy remain near those in the dataset. Previous offline RL algorithms that consider risk combine offline RL techniques (to avoid distributional shift), with risk-sensitive RL algorithms (to achieve risk-aversion). In this work, we propose risk-aversion as a mechanism to jointly address both of these issues. We propose a model-based approach, and use an ensemble of models to estimate epistemic uncertainty, in addition to aleatoric uncertainty. We train a policy that is risk-averse, and avoids high uncertainty actions. Risk-aversion to epistemic uncertainty prevents distributional shift, as areas not covered by the dataset have high epistemic uncertainty. Risk-aversion to aleatoric uncertainty discourages actions that are risky due to environment stochasticity. Thus, by considering epistemic uncertainty via a model ensemble and introducing risk-aversion, our algorithm (1R2R) avoids distributional shift in addition to achieving risk-aversion to aleatoric risk. Our experiments show that 1R2R achieves strong performance on deterministic benchmarks, and outperforms existing approaches for risk-sensitive objectives in stochastic domains.

        ----

        ## [3389] Monarch Mixer: A Simple Sub-Quadratic GEMM-Based Architecture

        **Authors**: *Daniel Y. Fu, Simran Arora, Jessica Grogan, Isys Johnson, Evan Sabri Eyuboglu, Armin W. Thomas, Benjamin Spector, Michael Poli, Atri Rudra, Christopher Ré*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f498c1ce6bff52eb04febf87438dd84b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f498c1ce6bff52eb04febf87438dd84b-Abstract-Conference.html)

        **Abstract**:

        Machine learning models are increasingly being scaled in both sequence length and model dimension to reach longer contexts and better performance. However, existing architectures such as Transformers scale quadratically along both these axes. We ask: are there performant architectures that can scale sub-quadratically along sequence length and model dimension? We introduce Monarch Mixer (M2), a new architecture that uses the same sub-quadratic primitive along both sequence length and model dimension: Monarch matrices, a simple class of expressive structured matrices that captures many linear transforms, achieves high hardware efficiency on GPUs, and scales sub-quadratically. As a proof of concept, we explore the performance of M2 in three domains: non-causal BERT-style language modeling, ViT-style image classification, and causal GPT-style language modeling. For non-causal BERT-style modeling, M2 matches BERT-base and BERT-large in downstream GLUE quality with up to 27% fewer parameters, and achieves up to 9.1$\times$ higher throughput at sequence length 4K. On ImageNet, M2 outperforms ViT-b by 1% in accuracy, with only half the parameters. Causal GPT-style models introduce a technical challenge: enforcing causality via masking introduces a quadratic bottleneck. To alleviate this bottleneck, we develop a novel theoretical view of Monarch matrices based on multivariate polynomial evaluation and interpolation, which lets us parameterize M2 to be causal while remaining sub-quadratic. Using this parameterization, M2 matches GPT-style Transformers at 360M parameters in pretraining perplexity on The PILE—showing for the first time that it may be possible to match Transformer quality without attention or MLPs.

        ----

        ## [3390] Bypassing spike sorting: Density-based decoding using spike localization from dense multielectrode probes

        **Authors**: *Yizi Zhang, Tianxiao He, Julien Boussard, Charles Windolf, Olivier Winter, Eric Trautmann, Noam Roth, Hailey Barrell, Mark Churchland, Nicholas A Steinmetz, Erdem Varol, Cole L. Hurwitz, Liam Paninski*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f499387f191d6be56e68966181095878-Abstract-Conference.html)

        **Abstract**:

        Neural decoding and its applications to brain computer interfaces (BCI) are essential for understanding the association between neural activity and behavior. A prerequisite for many decoding approaches is spike sorting, the assignment of action potentials (spikes) to individual neurons. Current spike sorting algorithms, however, can be inaccurate and do not properly model uncertainty of spike assignments, therefore discarding information that could potentially improve decoding performance. Recent advances in high-density probes (e.g., Neuropixels) and computational methods now allow for extracting a rich set of spike features from unsorted data; these features can in turn be used to directly decode behavioral correlates. To this end, we propose a spike sorting-free decoding method that directly models the distribution of extracted spike features using a mixture of Gaussians (MoG) encoding the uncertainty of spike assignments, without aiming to solve the spike clustering problem explicitly. We allow the mixing proportion of the MoG to change over time in response to the behavior and develop variational inference methods to fit the resulting model and to perform decoding. We benchmark our method with an extensive suite of recordings from different animals and probe geometries, demonstrating that our proposed decoder can consistently outperform current methods based on thresholding (i.e. multi-unit activity) and spike sorting. Open source code is available at https://github.com/yzhang511/density_decoding.

        ----

        ## [3391] SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models

        **Authors**: *Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhi-Ming Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4a6806490d31216a3ba667eb240c897-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4a6806490d31216a3ba667eb240c897-Abstract-Conference.html)

        **Abstract**:

        Diffusion Probabilistic Models (DPMs) have achieved considerable success in generation tasks. As sampling from DPMs is equivalent to solving diffusion SDE or ODE which is time-consuming, numerous fast sampling methods built upon improved differential equation solvers are proposed. The majority of such techniques consider solving the diffusion ODE due to its superior efficiency. However, stochastic sampling could offer additional advantages in generating diverse and high-quality data. In this work, we engage in a comprehensive analysis of stochastic sampling from two aspects: variance-controlled diffusion SDE and linear multi-step SDE solver. Based on our analysis, we propose SA-Solver, which is an improved efficient stochastic Adams method for solving diffusion SDE to generate data with high quality. Our experiments show that SA-Solver achieves: 1) improved or comparable performance compared with the existing state-of-the-art (SOTA) sampling methods for few-step sampling; 2) SOTA FID on substantial benchmark datasets under a suitable number of function evaluations (NFEs).

        ----

        ## [3392] Social Motion Prediction with Cognitive Hierarchies

        **Authors**: *Wentao Zhu, Jason Qin, Yuke Lou, Hang Ye, Xiaoxuan Ma, Hai Ci, Yizhou Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b52b45a677d855dee0ca9ba1ddf638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b52b45a677d855dee0ca9ba1ddf638-Abstract-Conference.html)

        **Abstract**:

        Humans exhibit a remarkable capacity for anticipating the actions of others and planning their own actions accordingly. In this study, we strive to replicate this ability by addressing the social motion prediction problem. We introduce a new benchmark, a novel formulation, and a cognition-inspired framework. We present Wusi, a 3D multi-person motion dataset under the context of team sports, which features intense and strategic human interactions and diverse pose distributions. By reformulating the problem from a multi-agent reinforcement learning perspective, we incorporate behavioral cloning and generative adversarial imitation learning to boost learning efficiency and generalization. Furthermore, we take into account the cognitive aspects of the human social action planning process and develop a cognitive hierarchy framework to predict strategic human social interactions. We conduct comprehensive experiments to validate the effectiveness of our proposed dataset and approach.

        ----

        ## [3393] Unbounded Differentially Private Quantile and Maximum Estimation

        **Authors**: *David Durfee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b6ef2a78684dca2fb3f1c09372e041-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b6ef2a78684dca2fb3f1c09372e041-Abstract-Conference.html)

        **Abstract**:

        In this work we consider the problem of differentially private computation ofquantiles for the data, especially the highest quantiles such as maximum, butwith an unbounded range for the dataset. We show that this can be doneefficiently through a simple invocation of $\texttt{AboveThreshold}$, asubroutine that is iteratively called in the fundamental Sparse VectorTechnique, even when there is no upper bound on the data. In particular, weshow that this procedure can give more accurate and robust estimates on thehighest quantiles with applications towards clipping that is essential fordifferentially private sum and mean estimation. In addition, we show how twoinvocations can handle the fully unbounded data setting. Within our study, weshow that an improved analysis of $\texttt{AboveThreshold}$ can improve theprivacy guarantees for the widely used Sparse Vector Technique that is ofindependent interest. We give a more general characterization of privacy lossfor $\texttt{AboveThreshold}$ which we immediately apply to our method forimproved privacy guarantees. Our algorithm only requires one $O(n)$ passthrough the data, which can be unsorted, and each subsequent query takes $O(1)$time. We empirically compare our unbounded algorithm with the state-of-the-artalgorithms in the bounded setting. For inner quantiles, we find that our methodoften performs better on non-synthetic datasets. For the maximal quantiles,which we apply to differentially private sum computation, we find that ourmethod performs significantly better.

        ----

        ## [3394] How to Turn Your Knowledge Graph Embeddings into Generative Models

        **Authors**: *Lorenzo Loconte, Nicola Di Mauro, Robert Peharz, Antonio Vergari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b768188be63b8d2680a46934fd295a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b768188be63b8d2680a46934fd295a-Abstract-Conference.html)

        **Abstract**:

        Some of the most successful knowledge graph embedding (KGE) models for link prediction – CP, RESCAL, TuckER, ComplEx – can be interpreted as energy-based models. Under this perspective they are not amenable for exact maximum-likelihood estimation (MLE), sampling and struggle to integrate logical constraints. This work re-interprets the score functions of these KGEs as circuits – constrained computational graphs allowing efficient marginalisation. Then, we design two recipes to obtain efficient generative circuit models by either restricting their activations to be non-negative or squaring their outputs. Our interpretation comes with little or no loss of performance for link prediction, while the circuits framework unlocks exact learning by MLE, efficient sampling of new triples, and guarantee that logical constraints are satisfied by design. Furthermore, our models scale more gracefully than the original KGEs on graphs with millions of entities.

        ----

        ## [3395] Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer

        **Authors**: *Zikai Xiao, Zihan Chen, Songshang Liu, Hualiang Wang, Yang Feng, Jin Hao, Joey Tianyi Zhou, Jian Wu, Howard H. Yang, Zuozhu Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html)

        **Abstract**:

        Data privacy and long-tailed distribution are the norms rather than the exception in many real-world tasks. This paper investigates a federated long-tailed learning (Fed-LT) task in which each client holds a locally heterogeneous dataset; if the datasets can be globally aggregated, they jointly exhibit a long-tailed distribution. Under such a setting, existing federated optimization and/or centralized long-tailed learning methods hardly apply due to challenges in (a) characterizing the global long-tailed distribution under privacy constraints and (b) adjusting the local learning strategy to cope with the head-tail imbalance. In response, we propose a method termed $\texttt{Fed-GraB}$, comprised of a Self-adjusting Gradient Balancer (SGB) module that re-weights clients' gradients in a closed-loop manner, based on the feedback of global long-tailed distribution evaluated by a Direct Prior Analyzer (DPA) module. Using $\texttt{Fed-GraB}$, clients can effectively alleviate the distribution drift caused by data heterogeneity during the model training process and obtain a global model with better performance on the minority classes while maintaining the performance of the majority classes. Extensive experiments demonstrate that $\texttt{Fed-GraB}$ achieves state-of-the-art performance on representative datasets such as CIFAR-10-LT, CIFAR-100-LT, ImageNet-LT, and iNaturalist.

        ----

        ## [3396] CityRefer: Geography-aware 3D Visual Grounding Dataset on City-scale Point Cloud Data

        **Authors**: *Taiki Miyanishi, Fumiya Kitamori, Shuhei Kurita, Jungdae Lee, Motoaki Kawanabe, Nakamasa Inoue*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4cef76305dcad4efd3537da087ff520-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        City-scale 3D point cloud is a promising way to express detailed and complicated outdoor structures. It encompasses both the appearance and geometry features of segmented city components, including cars, streets, and buildings that can be utilized for attractive applications such as user-interactive navigation of autonomous vehicles and drones. However, compared to the extensive text annotations available for images and indoor scenes, the scarcity of text annotations for outdoor scenes poses a significant challenge for achieving these applications. To tackle this problem, we introduce the CityRefer dataset for city-level visual grounding. The dataset consists of 35k natural language descriptions of 3D objects appearing in SensatUrban city scenes and 5k landmarks labels synchronizing with OpenStreetMap. To ensure the quality and accuracy of the dataset, all descriptions and labels in the CityRefer dataset are manually verified. We also have developed a baseline system that can learn encoded language descriptions, 3D object instances, and geographical information about the city's landmarks to perform visual grounding on the CityRefer dataset. To the best of our knowledge, the CityRefer dataset is the largest city-level visual grounding dataset for localizing specific 3D objects.

        ----

        ## [3397] GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image

        **Authors**: *Mingjian Zhu, Hanting Chen, Qiangyu Yan, Xudong Huang, Guanyu Lin, Wei Li, Zhijun Tu, Hailin Hu, Jie Hu, Yunhe Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4d4a021f9051a6c18183b059117e8b5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The extraordinary ability of generative models to generate photographic images has intensified concerns about the spread of disinformation, thereby leading to the demand for detectors capable of distinguishing between AI-generated fake images and real images. However, the lack of large datasets containing images from the most advanced image generators poses an obstacle to the development of such detectors. In this paper, we introduce the GenImage dataset, which has the following advantages: 1) Plenty of Images, including over one million pairs of AI-generated fake images and collected real images. 2) Rich Image Content, encompassing a broad range of image classes. 3) State-of-the-art Generators, synthesizing images with advanced diffusion models and GANs. The aforementioned advantages allow the detectors trained on GenImage to undergo a thorough evaluation and demonstrate strong applicability to diverse images. We conduct a comprehensive analysis of the dataset and propose two tasks for evaluating the detection method in resembling real-world scenarios. The cross-generator image classification task measures the performance of a detector trained on one generator when tested on the others. The degraded image classification task assesses the capability of the detectors in handling degraded images such as low-resolution, blurred, and compressed images. With the GenImage dataset, researchers can effectively expedite the development and evaluation of superior AI-generated image detectors in comparison to prevailing methodologies.

        ----

        ## [3398] On Differentially Private Sampling from Gaussian and Product Distributions

        **Authors**: *Badih Ghazi, Xiao Hu, Ravi Kumar, Pasin Manurangsi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4eaa4b8f2d08edb3f0af990d56134ea-Abstract-Conference.html)

        **Abstract**:

        We study the problem, where given a dataset of $n$ i.i.d. samples from an unknown distribution $P$, we seek to generate a sample from a distribution that is close to $P$ in total variation distance, under the constraint of differential privacy. We study the settings where $P$ is a multi-dimensional Gaussian distribution with different assumptions: known covariance, unknown bounded covariance, and unknown unbounded covariance. We present new differentially private sampling algorithms, and show that they achieve near-optimal sample complexity in the first two settings. Moreover, when $P$ is a product distribution on the binary hypercube, we obtain a pure-DP algorithm whereas only an approximate-DP algorithm (with slightly worse sample complexity) was previously known.

        ----

        ## [3399] MedSat: A Public Health Dataset for England Featuring Medical Prescriptions and Satellite Imagery

        **Authors**: *Sanja Scepanovic, Ivica Obadic, Sagar Joglekar, Laura Giustarini, Cristiano Nattero, Daniele Quercia, Xiaoxiang Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f4fdf676c3b21f20f8c391d929188386-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f4fdf676c3b21f20f8c391d929188386-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        As extreme weather events become more frequent, understanding their impact on human health becomes increasingly crucial. However, the utilization of Earth Observation to effectively analyze the environmental context in relation to health remains limited. This limitation is primarily due to the lack of fine-grained spatial and temporal data in public and population health studies, hindering a comprehensive understanding of health outcomes. Additionally, obtaining appropriate environmental indices across different geographical levels and timeframes poses a challenge. For the years 2019 (pre-COVID) and 2020 (COVID), we collected spatio-temporal indicators for all Lower Layer Super Output Areas in England. These indicators included: i) 111 sociodemographic features linked to health in existing literature, ii) 43 environmental point features (e.g., greenery and air pollution levels), iii) 4 seasonal composite satellite images each with 11 bands, and iv) prescription prevalence associated with five medical conditions (depression, anxiety, diabetes, hypertension, and asthma), opioids and total prescriptions. We combined these indicators into a single MedSat dataset, the availability of which presents an opportunity for the machine learning community to develop new techniques specific to public health. These techniques would address challenges such as handling large and complex data volumes, performing effective feature engineering on environmental and sociodemographic factors, capturing spatial and temporal dependencies in the models, addressing imbalanced data distributions, developing novel computer vision methods for health modeling based on satellite imagery, ensuring model explainability, and achieving generalization beyond the specific geographical region.

        ----

        

[Go to the previous page](NIPS-2023-list16.md)

[Go to the next page](NIPS-2023-list18.md)

[Go to the catalog section](README.md)