## [1000] Rectifying the Shortcut Learning of Background for Few-Shot Learning

**Authors**: *Xu Luo, Longhui Wei, Liangjian Wen, Jinrong Yang, Lingxi Xie, Zenglin Xu, Qi Tian*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6cfe0e6127fa25df2a0ef2ae1067d915-Abstract.html)

**Abstract**:

The category gap between training and evaluation has been characterised as one of the main obstacles to the success of Few-Shot Learning (FSL). In this paper, we for the first time empirically identify image background, common in realistic images, as a shortcut knowledge helpful for in-class classification but ungeneralizable beyond training categories in FSL. A novel framework, COSOC, is designed to tackle this problem by extracting foreground objects in images at both training and evaluation without any extra supervision. Extensive experiments carried on inductive FSL tasks demonstrate the effectiveness of our approaches.

----

## [1001] SEAL: Self-supervised Embodied Active Learning using Exploration and 3D Consistency

**Authors**: *Devendra Singh Chaplot, Murtaza Dalal, Saurabh Gupta, Jitendra Malik, Ruslan Salakhutdinov*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6d0c932802f6953f70eb20931645fa40-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6d0c932802f6953f70eb20931645fa40-Abstract.html)

**Abstract**:

In this paper, we explore how we can build upon the data and models of Internet images and use them to adapt to robot vision without requiring any extra labels. We present a framework called Self-supervised Embodied Active Learning (SEAL). It utilizes perception models trained on internet images to learn an active exploration policy. The observations gathered by this exploration policy are labelled using 3D consistency and used to improve the perception model. We build and utilize 3D semantic maps to learn both action and perception in a completely self-supervised manner. The semantic map is used to compute an intrinsic motivation reward for training the exploration policy and for labelling the agent observations using spatio-temporal 3D consistency and label propagation. We demonstrate that the SEAL framework can be used to close the action-perception loop: it improves object detection and instance segmentation performance of a pretrained perception model by just moving around in training environments and the improved perception model can be used to improve Object Goal Navigation.

----

## [1002] Sifting through the noise: Universal first-order methods for stochastic variational inequalities

**Authors**: *Kimon Antonakopoulos, Thomas Pethick, Ali Kavis, Panayotis Mertikopoulos, Volkan Cevher*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6d65b5ac1f4ec80b9a7309311f4f9b13-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6d65b5ac1f4ec80b9a7309311f4f9b13-Abstract.html)

**Abstract**:

We examine a flexible algorithmic framework for solving monotone variational inequalities in the presence of randomness and uncertainty. The proposed template encompasses a wide range of popular first-order methods, including dual averaging, dual extrapolation and optimistic gradient algorithms – both adaptive and non-adaptive. Our first result is that the algorithm achieves the optimal rates of convergence for cocoercive problems when the profile of the randomness is known to the optimizer: $\mathcal{O}(1/\sqrt{T})$ for absolute noise profiles, and $\mathcal{O}(1/T)$ for relative ones. Subsequently, we drop all prior knowledge requirements (the absolute/relative variance of the randomness affecting the problem, the operator's cocoercivity constant, etc.), and we analyze an adaptive instance of the method that gracefully interpolates between the above rates – i.e. it achieves $\mathcal{O}(1/\sqrt{T})$ and $\mathcal{O}(1/T)$ in the absolute and relative cases, respectively. To our knowledge, this is the first universality result of its kind in the literature and, somewhat surprisingly, it shows that an extra-gradient proxy step is not required to achieve optimal rates.

----

## [1003] Accommodating Picky Customers: Regret Bound and Exploration Complexity for Multi-Objective Reinforcement Learning

**Authors**: *Jingfeng Wu, Vladimir Braverman, Lin Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6d7d394c9d0c886e9247542e06ebb705-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6d7d394c9d0c886e9247542e06ebb705-Abstract.html)

**Abstract**:

In this paper we consider multi-objective reinforcement learning where the objectives are balanced using preferences. In practice, the preferences are often given in an adversarial manner, e.g., customers can be picky in many applications. We formalize this problem as an episodic learning problem on a Markov decision process, where transitions are unknown and a reward function is the inner product of a preference vector with pre-specified multi-objective reward functions. We consider two settings. In the online setting, the agent receives a (adversarial) preference every episode and proposes policies to interact with the environment. We provide a model-based algorithm that achieves a nearly minimax optimal regret bound $\widetilde{\mathcal{O}}\bigl(\sqrt{\min\{d,S\}\cdot H^2 SAK}\bigr)$, where $d$ is the number of objectives, $S$ is the number of states, $A$ is the number of actions, $H$ is the length of the horizon, and $K$ is the number of episodes. Furthermore, we consider preference-free exploration, i.e., the agent first interacts with the environment without specifying any preference and then is able to accommodate arbitrary preference vector up to $\epsilon$ error. Our proposed algorithm is provably efficient with a nearly optimal trajectory complexity $\widetilde{\mathcal{O}}\bigl({\min\{d,S\}\cdot H^3 SA}/{\epsilon^2}\bigr)$. This result partly resolves an open problem raised by \citet{jin2020reward}.

----

## [1004] Exact Privacy Guarantees for Markov Chain Implementations of the Exponential Mechanism with Artificial Atoms

**Authors**: *Jeremy Seeman, Matthew Reimherr, Aleksandra B. Slavkovic*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6d96718a701f5bfba283bbdc71dfa5c4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6d96718a701f5bfba283bbdc71dfa5c4-Abstract.html)

**Abstract**:

Implementations of the exponential mechanism in differential privacy often require sampling from intractable distributions. When approximate procedures like Markov chain Monte Carlo (MCMC) are used, the end result incurs costs to both privacy and accuracy. Existing work has examined these effects asymptotically, but implementable finite sample results are needed in practice so that users can specify privacy budgets in advance and implement samplers with exact privacy guarantees. In this paper, we use tools from ergodic theory and perfect simulation to design exact finite runtime sampling algorithms for the exponential mechanism by introducing an intermediate modified target distribution using artificial atoms. We propose an additional modification of this sampling algorithm that maintains its $\epsilon$-DP guarantee and has improved runtime at the cost of some utility. We then compare these methods in scenarios where we can explicitly calculate a $\delta$ cost (as in $(\epsilon, \delta)$-DP) incurred when using standard MCMC techniques. Much as there is a well known trade-off between privacy and utility, we demonstrate that there is also a trade-off between privacy guarantees and runtime.

----

## [1005] The Emergence of Objectness: Learning Zero-shot Segmentation from Videos

**Authors**: *Runtao Liu, Zhirong Wu, Stella X. Yu, Stephen Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6d9cb7de5e8ac30bd5e8734bc96a35c1-Abstract.html)

**Abstract**:

Humans can easily detect and segment moving objects simply by observing how they move, even without knowledge of object semantics. Inspired by this, we develop a zero-shot unsupervised approach for learning object segmentations. The model comprises two visual pathways: an appearance pathway that segments individual RGB images into coherent object regions, and a motion pathway that predicts the flow vector for each region between consecutive video frames. The two pathways jointly reconstruct a new representation called segment flow. This decoupled representation of appearance and motion is trained in a self-supervised manner to reconstruct one frame from another.When pretrained on an unlabeled video corpus, the model can be useful for a variety of applications, including 1) primary object segmentation from a single image in a zero-shot fashion; 2) moving object segmentation from a video with unsupervised test-time adaptation; 3) image semantic segmentation by supervised fine-tuning on a labeled image dataset. We demonstrate encouraging experimental results on all of these tasks using  pretrained models.

----

## [1006] Direct Multi-view Multi-person 3D Pose Estimation

**Authors**: *Tao Wang, Jianfeng Zhang, Yujun Cai, Shuicheng Yan, Jiashi Feng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6da9003b743b65f4c0ccd295cc484e57-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6da9003b743b65f4c0ccd295cc484e57-Abstract.html)

**Abstract**:

We present Multi-view Pose transformer (MvP) for estimating multi-person 3D poses from multi-view images. Instead of estimating 3D joint locations from costly volumetric representation or reconstructing the per-person 3D pose from multiple detected 2D poses as in previous methods, MvP directly regresses the multi-person 3D  poses  in  a  clean  and  efficient  way,  without  relying  on  intermediate  tasks. Specifically, MvP represents skeleton joints as learnable query embeddings and let them progressively attend to and reason over the multi-view information from the input images to directly regress the actual 3D joint locations. To improve the accuracy of such a simple pipeline, MvP presents a hierarchical scheme to concisely represent query embeddings of multi-person skeleton joints and introduces an input-dependent query adaptation approach. Further, MvP designs a novel geometrically guided attention mechanism, called projective attention, to more precisely fuse the cross-view information for each joint. MvP also introduces a RayConv operation to integrate the view-dependent camera geometry into the feature representations for augmenting the projective attention.  We show experimentally that our MvP model outperforms the state-of-the-art methods on several benchmarks while being much more efficient. Notably, it achieves 92.3% AP25 on the challenging Panoptic dataset, improving upon the previous best approach [35] by 9.8%. MvP is general and also extendable to recovering human mesh represented by the SMPL model, thus useful for modeling multi-person body shapes. Code and models are available at https://github.com/sail-sg/mvp.

----

## [1007] MST: Masked Self-Supervised Transformer for Visual Representation

**Authors**: *Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, Jinqiao Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6dbbe6abe5f14af882ff977fc3f35501-Abstract.html)

**Abstract**:

Transformer has been widely used for self-supervised pre-training in Natural Language Processing (NLP) and achieved great success. However, it has not been fully explored in visual self-supervised learning. Meanwhile, previous methods only consider the high-level feature and learning representation from a global perspective, which may fail to transfer to the downstream dense prediction tasks focusing on local features. In this paper, we present a novel Masked Self-supervised Transformer approach named MST, which can explicitly capture the local context of an image while preserving the global semantic information. Specifically, inspired by the Masked Language Modeling (MLM) in NLP, we propose a masked token strategy based on the multi-head self-attention map, which dynamically masks some tokens of local patches without damaging the crucial structure for self-supervised learning. More importantly, the masked tokens together with the remaining tokens are further recovered by a global image decoder, which preserves the spatial information of the image and is more friendly to the downstream dense prediction tasks. The experiments on multiple datasets demonstrate the effectiveness and generality of the proposed method. For instance, MST achieves Top-1 accuracy of 76.9% with DeiT-S only using 300-epoch pre-training by linear evaluation, which outperforms supervised methods with the same epoch by 0.4% and its comparable variant DINO by 1.0%. For dense prediction tasks, MST also achieves 42.7% mAP on MS COCO object detection and 74.04% mIoU on Cityscapes segmentation only with 100-epoch pre-training.

----

## [1008] Exploiting Opponents Under Utility Constraints in Sequential Games

**Authors**: *Martino Bernasconi de Luca, Federico Cacciamani, Simone Fioravanti, Nicola Gatti, Alberto Marchesi, Francesco Trovò*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6de0f2761a44ff1e2ca60131058d8297-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6de0f2761a44ff1e2ca60131058d8297-Abstract.html)

**Abstract**:

Recently, game-playing agents based on AI techniques have demonstrated super-human performance in several sequential games, such as chess, Go, and poker. Surprisingly, the multi-agent learning techniques that allowed to reach these achievements do not take into account the actual behavior of the human player, potentially leading to an impressive gap in performances. In this paper, we address the problem of designing artificial agents that learn how to effectively exploit unknown human opponents while playing repeatedly against them in an online fashion. We study the case in which the agent's strategy during each repetition of the game is subject to constraints ensuring that the human's expected utility is within some lower and upper thresholds. Our framework encompasses several real-world problems, such as human engagement in repeated game playing and human education by means of serious games. As a first result, we formalize a set of linear inequalities encoding the conditions that the agent's strategy must satisfy at each iteration in order to do not violate the given bounds for the human's expected utility. Then, we use such formulation in an upper confidence bound algorithm, and we prove that the resulting procedure suffers from sublinear regret and guarantees that the constraints are satisfied with high probability at each iteration. Finally, we empirically evaluate the convergence of our algorithm on standard testbeds of sequential games.

----

## [1009] A Compositional Atlas of Tractable Circuit Operations for Probabilistic Inference

**Authors**: *Antonio Vergari, YooJung Choi, Anji Liu, Stefano Teso, Guy Van den Broeck*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e01383fd96a17ae51cc3e15447e7533-Abstract.html)

**Abstract**:

Circuit representations are becoming the lingua franca to express and reason about tractable generative and discriminative models. In this paper, we show how complex inference scenarios for these models that commonly arise in machine learning---from  computing the expectations of decision tree ensembles to information-theoretic divergences of sum-product networks---can be represented in terms of tractable modular operations over circuits. Specifically, we characterize the tractability of simple transformations---sums, products, quotients, powers, logarithms, and exponentials---in terms of sufficient structural constraints of the circuits they operate on, and present novel hardness results for the cases in which these properties are not satisfied. Building on these operations, we derive a unified framework for reasoning about tractable models that generalizes several results in the literature and opens up novel tractable inference scenarios.

----

## [1010] Demystifying and Generalizing BinaryConnect

**Authors**: *Tim Dockhorn, Yaoliang Yu, Eyyüb Sari, Mahdi Zolnouri, Vahid Partovi Nia*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e0cf80a83327822a972bcde3c1d9740-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e0cf80a83327822a972bcde3c1d9740-Abstract.html)

**Abstract**:

BinaryConnect (BC) and its many variations have become the de facto standard for neural network quantization. However, our understanding of the inner workings of BC is still quite limited. We attempt to close this gap in four different aspects: (a) we show that existing quantization algorithms, including post-training quantization, are surprisingly similar to each other; (b) we argue for proximal maps as a natural family of quantizers that is both easy to design and analyze; (c) we refine the observation that BC is a special case of dual averaging, which itself is a special case of the generalized conditional gradient algorithm; (d) consequently, we propose ProxConnect (PC) as a generalization of BC and we prove its convergence properties by exploiting the established connections. We conduct experiments on CIFAR-10 and ImageNet, and verify that PC achieves competitive performance.

----

## [1011] CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator

**Authors**: *Alek Dimitriev, Mingyuan Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e16656a6ee1de7232164767ccfa7920-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e16656a6ee1de7232164767ccfa7920-Abstract.html)

**Abstract**:

Accurately backpropagating the gradient through categorical variables is a challenging task that arises in various domains, such as training discrete latent variable models. To this end, we propose CARMS, an unbiased estimator for categorical random variables based on multiple mutually negatively correlated (jointly antithetic) samples. CARMS combines REINFORCE with copula based sampling to avoid duplicate samples and reduce its variance, while keeping the estimator unbiased using importance sampling. It generalizes both the ARMS antithetic estimator for binary variables, which is CARMS for two categories, as well as LOORF/VarGrad, the leave-one-out REINFORCE estimator, which is CARMS with independent samples.  We evaluate CARMS on several benchmark datasets on a generative modeling task, as well as a structured output prediction task, and find it to outperform competing methods including a strong self-control baseline. The code is publicly available.

----

## [1012] Learning to Learn Dense Gaussian Processes for Few-Shot Learning

**Authors**: *Ze Wang, Zichen Miao, Xiantong Zhen, Qiang Qiu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e2713a6efee97bacb63e52c54f0ada0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e2713a6efee97bacb63e52c54f0ada0-Abstract.html)

**Abstract**:

Gaussian processes with deep neural networks demonstrate to be a strong learner for few-shot learning since they combine the strength of deep learning and kernels while being able to well capture uncertainty. However, it remains an open problem to leverage the shared knowledge provided by related tasks. In this paper, we propose to learn Gaussian processes with dense inducing variables by meta-learning for few-shot learning. In contrast to sparse Gaussian processes, we define a set of dense inducing variables to be of a much larger size than the support set in each task, which collects prior knowledge from experienced tasks. The dense inducing variables specify a shared Gaussian process prior over prediction functions of all tasks, which are learned in a variational inference framework and offer a strong inductive bias for learning new tasks. To achieve task-specific prediction functions, we propose to adapt the inducing variables to each task by efficient gradient descent. We conduct extensive experiments on common benchmark datasets for a variety of few-shot learning tasks. Our dense Gaussian processes present significant improvements over vanilla Gaussian processes and comparable or even better performance with state-of-the-art methods.

----

## [1013] Stochastic Solutions for Linear Inverse Problems using the Prior Implicit in a Denoiser

**Authors**: *Zahra Kadkhodaie, Eero P. Simoncelli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e28943943dbed3c7f82fc05f269947a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e28943943dbed3c7f82fc05f269947a-Abstract.html)

**Abstract**:

Deep neural networks have provided state-of-the-art solutions for problems such as image denoising, which implicitly rely on a prior probability model of natural images. Two recent lines of work – Denoising Score Matching and Plug-and-Play – propose methodologies for drawing samples from this implicit prior and using it to solve inverse problems, respectively. Here, we develop a parsimonious and robust generalization of these ideas. We rely on a classic statistical result that shows the least-squares solution for removing additive Gaussian noise can be written directly in terms of the gradient of the log of the noisy signal density. We use this to derive a stochastic coarse-to-fine gradient ascent procedure for drawing high-probability samples from the implicit prior embedded within a CNN trained to perform blind denoising. A generalization of this algorithm to constrained sampling provides a method for using the implicit prior to solve any deterministic linear inverse problem, with no additional training, thus extending the power of supervised learning for denoising to a much broader set of problems. The algorithm relies on minimal assumptions and exhibits robust convergence over a wide range of parameter choices. To demonstrate the generality of our method, we use it to obtain state-of-the-art levels of unsupervised performance for deblurring, super-resolution, and compressive sensing.

----

## [1014] Towards Stable and Robust AdderNets

**Authors**: *Minjing Dong, Yunhe Wang, Xinghao Chen, Chang Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e3197aae95c2ff8fcab35cb730f6a86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e3197aae95c2ff8fcab35cb730f6a86-Abstract.html)

**Abstract**:

Adder neural network (AdderNet) replaces the original convolutions with massive multiplications by cheap additions while achieving comparable performance thus yields a series of energy-efficient neural networks. Compared with convolutional neural networks (CNNs), the training of AdderNets is much more sophisticated including several techniques for adjusting gradient and batch normalization. In addition, variances of both weights and activations in resulting adder networks are very enormous which limits its performance and the potential for applying to other tasks. To enhance the stability and robustness of AdderNets, we first thoroughly analyze the variance estimation of weight parameters and output features of an arbitrary adder layer. Then, we develop a weight normalization scheme for adaptively optimizing the weight distribution of AdderNets during the training procedure, which can reduce the perturbation on running mean and variance in batch normalization layers. Meanwhile, the proposed weight normalization can also be utilized to enhance the adversarial robustness of resulting networks. Experiments conducted on several benchmarks demonstrate the superiority of the proposed approach for generating AdderNets with higher performance.

----

## [1015] Representing Long-Range Context for Graph Neural Networks with Global Attention

**Authors**: *Zhanghao Wu, Paras Jain, Matthew A. Wright, Azalia Mirhoseini, Joseph E. Gonzalez, Ion Stoica*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e67691b60ed3e4a55935261314dd534-Abstract.html)

**Abstract**:

Graph neural networks are powerful architectures for structured datasets. However, current methods struggle to represent long-range dependencies. Scaling the depth or width of GNNs is insufficient to broaden receptive fields as larger GNNs encounter optimization instabilities such as vanishing gradients and representation oversmoothing, while pooling-based approaches have yet to become as universally useful as in computer vision. In this work, we propose the use of Transformer-based self-attention to learn long-range pairwise relationships, with a novel “readout” mechanism to obtain a global graph embedding. Inspired by recent computer vision results that find position-invariant attention performant in learning long-range relationships, our method, which we call GraphTrans, applies a permutation-invariant Transformer module after a standard GNN module. This simple architecture leads to state-of-the-art results on several graph classification tasks, outperforming methods that explicitly encode graph structure. Our results suggest that purely-learning-based approaches without graph structure may be suitable for learning high-level, long-range relationships on graphs. Code for GraphTrans is available at https://github.com/ucbrise/graphtrans.

----

## [1016] Beyond Bandit Feedback in Online Multiclass Classification

**Authors**: *Dirk van der Hoeven, Federico Fusco, Nicolò Cesa-Bianchi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e79ed05baec2754e25b4eac73a332d2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e79ed05baec2754e25b4eac73a332d2-Abstract.html)

**Abstract**:

We study the problem of online multiclass classification in a setting where the learner's feedback is determined by an arbitrary directed graph. While including bandit feedback as a special case, feedback graphs allow a much richer set of applications, including filtering and label efficient classification.We introduce \textproc{Gappletron}, the first online multiclass algorithm that works with arbitrary feedback graphs. For this new algorithm,we prove surrogate regret bounds that hold, both in expectation and with high probability, for a large class of surrogate losses. Our bounds are of order $B\sqrt{\rho KT}$, where $B$ is the diameter of the prediction space, $K$ is the number of classes, $T$ is the time horizon, and $\rho$ is the domination number (a graph-theoretic parameter affecting the amount of exploration). In the full information case, we show that \textproc{Gappletron} achieves a constant surrogate regret of order $B^2K$. We also prove a general lower bound of order $\max\big\{B^2K,\sqrt{T}\big\}$ showing that our upper bounds are not significantly improvable. Experiments on synthetic data show that for various feedback graphs our algorithm is competitive against known baselines.

----

## [1017] Learning Student-Friendly Teacher Networks for Knowledge Distillation

**Authors**: *Dae Young Park, Moon-Hyun Cha, Changwook Jeong, Daesin Kim, Bohyung Han*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e7d2da6d3953058db75714ac400b584-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e7d2da6d3953058db75714ac400b584-Abstract.html)

**Abstract**:

We propose a novel knowledge distillation approach to facilitate the transfer of dark knowledge from a teacher to a student. Contrary to most of the existing methods that rely on effective training of student models given pretrained teachers, we aim to learn the teacher models that are friendly to students and, consequently, more appropriate for knowledge transfer. In other words, at the time of optimizing a teacher model, the proposed algorithm learns the student branches jointly to obtain student-friendly representations. Since the main goal of our approach lies in training teacher models and the subsequent knowledge distillation procedure is straightforward, most of the existing knowledge distillation methods can adopt this technique to improve the performance of diverse student models in terms of accuracy and convergence speed. The proposed algorithm demonstrates outstanding accuracy in several well-known knowledge distillation techniques with various combinations of teacher and student models even in the case that their architectures are heterogeneous and there is no prior knowledge about student models at the time of training teacher networks

----

## [1018] Implicit Transformer Network for Screen Content Image Continuous Super-Resolution

**Authors**: *Jingyu Yang, Sheng Shen, Huanjing Yue, Kun Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e7d5d259be7bf56ed79029c4e621f44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e7d5d259be7bf56ed79029c4e621f44-Abstract.html)

**Abstract**:

Nowadays, there is an explosive growth of screen contents due to the wide application of screen sharing, remote cooperation, and online education. To match the limited terminal bandwidth,  high-resolution (HR) screen contents may be downsampled and compressed.  At the receiver side, the super-resolution (SR)of low-resolution (LR) screen content images (SCIs) is highly demanded by the HR display or by the users to zoom in for detail observation.  However, image SR methods mostly designed for natural images do not generalize well for SCIs due to the very different image characteristics as well as the requirement of SCI browsing at arbitrary scales. To this end, we propose a novel Implicit Transformer Super-Resolution Network (ITSRN) for SCISR. For high-quality continuous SR at arbitrary ratios, pixel values at query coordinates are inferred from image features at key coordinates by the proposed implicit transformer and an implicit position encoding scheme is proposed to aggregate similar neighboring pixel values to the query one. We construct benchmark SCI1K and SCI1K-compression datasets withLR and HR SCI pairs.  Extensive experiments show that the proposed ITSRN significantly outperforms several competitive continuous and discrete SR methods for both compressed and uncompressed SCIs.

----

## [1019] Channel Permutations for N: M Sparsity

**Authors**: *Jeff Pool, Chong Yu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6e8404c3b93a9527c8db241a1846599a-Abstract.html)

**Abstract**:

We introduce channel permutations as a method to maximize the accuracy of N:M sparse networks. N:M sparsity requires N out of M consecutive elements to be zero and has been shown to maintain accuracy for many models and tasks with a simple prune and fine-tune workflow. By permuting weight matrices along their channel dimension and adjusting the surrounding layers appropriately, we demonstrate accuracy recovery for even small, parameter-efficient networks, without affecting inference run-time. We also present both a quality metric to simplify judging permutations as well as efficient methods to search for high-quality permutations, including two optimizations to escape local minima. Finally, we share an ablation study to show the importance of each part of our search algorithm, experimental results showing correlation between our quality metric and final network accuracy, improved sparse network accuracy using our techniques with insignificant overhead to training time, and the transformation of unstructured to structured sparse workloads. Code to use these techniques when generating a 2:4 sparse network is available at https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity.

----

## [1020] Curriculum Learning for Vision-and-Language Navigation

**Authors**: *Jiwen Zhang, Zhongyu Wei, Jianqing Fan, Jiajie Peng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f0442558302a6ededff195daf67f79b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f0442558302a6ededff195daf67f79b-Abstract.html)

**Abstract**:

Vision-and-Language Navigation (VLN) is a task where an agent navigates in an embodied indoor environment under human instructions. Previous works ignore the distribution of sample difficulty and we argue that this potentially degrade their agent performance. To tackle this issue, we propose a novel curriculum- based training paradigm for VLN tasks that can balance human prior knowledge and agent learning progress about training samples. We develop the principle of curriculum design and re-arrange the benchmark Room-to-Room (R2R) dataset to make it suitable for curriculum training. Experiments show that our method is model-agnostic and can significantly improve the performance, the generalizability, and the training efficiency of current state-of-the-art navigation agents without increasing model complexity.

----

## [1021] Better Algorithms for Individually Fair k-Clustering

**Authors**: *Maryam Negahbani, Deeparnab Chakrabarty*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f221fcb5c504fe96789df252123770b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f221fcb5c504fe96789df252123770b-Abstract.html)

**Abstract**:

We study data clustering problems with $\ell_p$-norm objectives (e.g. \textsc{$k$-Median} and \textsc{$k$-Means}) in the context of individual fairness. The dataset consists of $n$ points, and we want to find $k$ centers such that (a) the objective is minimized, while (b) respecting the individual fairness constraint that every point $v$ has a center within a distance at most $r(v)$, where $r(v)$ is $v$'s distance to its $(n/k)$th nearest point. Jung, Kannan, and Lutz [FORC 2020] introduced this concept and designed a clustering algorithm with provable (approximate) fairness and objective guarantees for the $\ell_\infty$ or \textsc{$k$-Center} objective.  Mahabadi and Vakilian [ICML 2020] revisited this problem to give a local-search algorithm for all $\ell_p$-norms. Empirically, their algorithms outperform Jung et. al.'s by a large margin in terms of cost (for \textsc{$k$-Median} and \textsc{$k$-Means}), but they incur a reasonable loss in fairness. In this paper, our main contribution is to use Linear Programming (LP) techniques to obtain better algorithms for this problem, both in theory and in practice. We prove that by modifying known LP rounding techniques, one gets a worst-case guarantee on the objective which is much better than in MV20, and empirically, this objective is extremely close to the optimal.  Furthermore, our theoretical fairness guarantees are comparable with MV20 in theory, and empirically, we obtain noticeably fairer solutions.Although solving the LP {\em exactly} might be prohibitive, we demonstrate that in practice, a simple sparsification technique drastically improves the run-time of our algorithm.

----

## [1022] Video Instance Segmentation using Inter-Frame Communication Transformers

**Authors**: *Sukjun Hwang, Miran Heo, Seoung Wug Oh, Seon Joo Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f2688a5fce7d48c8d19762b88c32c3b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f2688a5fce7d48c8d19762b88c32c3b-Abstract.html)

**Abstract**:

We propose a novel end-to-end solution for video instance segmentation (VIS) based on transformers. Recently, the per-clip pipeline shows superior performance over per-frame methods leveraging richer information from multiple frames. However, previous per-clip models require heavy computation and memory usage to achieve frame-to-frame communications, limiting practicality.In this work, we propose Inter-frame Communication Transformers (IFC), which significantly reduces the overhead for information-passing between frames by efficiently encoding the context within the input clip.Specifically, we propose to utilize concise memory tokens as a means of conveying information as well as summarizing each frame scene.The features of each frame are enriched and correlated with other frames through exchange of information between the precisely encoded memory tokens.We validate our method on the latest benchmark sets and achieved state-of-the-art performance (AP 42.6 on YouTube-VIS 2019 val set using the offline inference) while having a considerably fast runtime (89.4 FPS). Our method can also be applied to near-online inference for processing a video in real-time with only a small delay.The code is available at https://github.com/sukjunhwang/IFC

----

## [1023] Progressive Coordinate Transforms for Monocular 3D Object Detection

**Authors**: *Li Wang, Li Zhang, Yi Zhu, Zhi Zhang, Tong He, Mu Li, Xiangyang Xue*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f3ef77ac0e3619e98159e9b6febf557-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f3ef77ac0e3619e98159e9b6febf557-Abstract.html)

**Abstract**:

Recognizing and localizing objects in the 3D space is a crucial ability for an AI agent to perceive its surrounding environment. While significant progress has been achieved with expensive LiDAR point clouds, it poses a great challenge for 3D object detection given only a monocular image. While there exist different alternatives for tackling this problem, it is found that they are either equipped with heavy networks to fuse RGB and depth information or empirically ineffective to process millions of pseudo-LiDAR points. With in-depth examination, we realize that these limitations are rooted in inaccurate object localization. In this paper, we propose a novel and lightweight approach, dubbed {\em Progressive Coordinate Transforms} (PCT) to facilitate learning coordinate representations. Specifically, a localization boosting mechanism with confidence-aware loss is introduced to progressively refine the localization prediction. In addition, semantic image representation is also exploited to compensate for the usage of patch proposals. Despite being lightweight and simple, our strategy allows us to establish a new state-of-the-art among the monocular 3D detectors on the competitive KITTI benchmark. At the same time, our proposed PCT shows great generalization to most coordinate-based 3D detection frameworks.

----

## [1024] Structured Reordering for Modeling Latent Alignments in Sequence Transduction

**Authors**: *Bailin Wang, Mirella Lapata, Ivan Titov*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f46dd176364ccec308c2760189a4605-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f46dd176364ccec308c2760189a4605-Abstract.html)

**Abstract**:

Despite success in many domains, neural models struggle in settings where train and test examples are drawn from different distributions. In particular, in contrast to humans, conventional sequence-to-sequence (seq2seq) models fail to generalize systematically, i.e., interpret sentences representing novel combinations of concepts (e.g., text segments) seen in training. Traditional grammar formalisms excel in such settings by implicitly encoding alignments between input and output segments, but are hard to scale and maintain.  Instead of engineering a grammar, we directly model segment-to-segment alignments as discrete structured latent variables within a neural seq2seq model. To efficiently explore the large space of alignments, we introduce a reorder-first align-later framework whose central component is a neural reordering module producing separable permutations. We present an efficient dynamic programming algorithm performing exact marginal inference of separable permutations, and, thus, enabling end-to-end differentiable training of our model.  The resulting seq2seq model exhibits better systematic generalization than standard models on synthetic problems and NLP tasks (i.e., semantic parsing and machine translation).

----

## [1025] A universal probabilistic spike count model reveals ongoing modulation of neural variability

**Authors**: *David Liu, Máté Lengyel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f5216f8d89b086c18298e043bfe48ed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f5216f8d89b086c18298e043bfe48ed-Abstract.html)

**Abstract**:

Neural responses are variable: even under identical experimental conditions, single neuron and population responses typically differ from trial to trial and across time. Recent work has demonstrated that this variability has predictable structure, can be modulated by sensory input and behaviour, and bears critical signatures of the underlying network dynamics and computations. However, current methods for characterising neural variability are primarily geared towards sensory coding in the laboratory: they require trials with repeatable experimental stimuli and behavioural covariates. In addition, they make strong assumptions about the parametric form of variability, rely on assumption-free but data-inefficient histogram-based approaches, or are altogether ill-suited for capturing variability modulation by covariates. Here we present a universal probabilistic spike count model that eliminates these shortcomings. Our method builds on sparse Gaussian processes and can model arbitrary spike count distributions (SCDs) with flexible dependence on observed as well as latent covariates, using scalable variational inference to jointly infer the covariate-to-SCD mappings and latent trajectories in a data efficient way. Without requiring repeatable trials, it can flexibly capture covariate-dependent joint SCDs, and provide interpretable latent causes underlying the statistical dependencies between neurons. We apply the model to recordings from a canonical non-sensory neural population: head direction cells in the mouse. We find that variability in these cells defies a simple parametric relationship with mean spike count as assumed in standard models, its modulation by external covariates can be comparably strong to that of the mean firing rate, and slow low-dimensional latent factors explain away neural correlations. Our approach paves the way to understanding the mechanisms and computations underlying neural variability under naturalistic conditions, beyond the realm of sensory coding with repeatable stimuli.

----

## [1026] Bellman Eluder Dimension: New Rich Classes of RL Problems, and Sample-Efficient Algorithms

**Authors**: *Chi Jin, Qinghua Liu, Sobhan Miryoosefi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6f5e4e86a87220e5d361ad82f1ebc335-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6f5e4e86a87220e5d361ad82f1ebc335-Abstract.html)

**Abstract**:

Finding the minimal structural assumptions that empower sample-efficient learning is one of the most important research directions in Reinforcement Learning (RL). This paper advances our understanding of this fundamental question by introducing a new complexity measure—Bellman Eluder (BE) dimension. We show that the family of RL problems of low BE dimension is remarkably rich, which subsumes a vast majority of existing tractable RL problems including but not limited to tabular MDPs, linear MDPs, reactive POMDPs, low Bellman rank problems as well as low Eluder dimension problems. This paper further designs a new optimization-based algorithm— GOLF, and reanalyzes a hypothesis elimination-based algorithm—OLIVE (proposed in Jiang et al. (2017)). We prove that both algorithms learn the near-optimal policies of low BE dimension problems in a number of samples that is polynomial in all relevant parameters, but independent of the size of state-action space. Our regret and sample complexity results match or improve the best existing results for several well-known subclasses of low BE dimension problems.

----

## [1027] Detecting Anomalous Event Sequences with Temporal Point Processes

**Authors**: *Oleksandr Shchur, Ali Caner Türkmen, Tim Januschowski, Jan Gasthaus, Stephan Günnemann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6faa8040da20ef399b63a72d0e4ab575-Abstract.html)

**Abstract**:

Automatically detecting anomalies in event data can provide substantial value in domains such as healthcare, DevOps, and information security. In this paper, we frame the problem of detecting anomalous continuous-time event sequences as out-of-distribution (OOD) detection for temporal point processes (TPPs). First, we show how this problem can be approached using goodness-of-fit (GoF) tests. We then demonstrate the limitations of popular GoF statistics for TPPs and propose a new test that addresses these shortcomings. The proposed method can be combined with various TPP models, such as neural TPPs, and is easy to implement. In our experiments, we show that the proposed statistic excels at both traditional GoF testing, as well as at detecting anomalies in simulated and real-world data.

----

## [1028] HNPE: Leveraging Global Parameters for Neural Posterior Estimation

**Authors**: *Pedro Rodrigues, Thomas Moreau, Gilles Louppe, Alexandre Gramfort*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6fbd841e2e4b2938351a4f9b68f12e6b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6fbd841e2e4b2938351a4f9b68f12e6b-Abstract.html)

**Abstract**:

Inferring the parameters of a stochastic model based on experimental observations is central to the scientific method. A particularly challenging setting is when the model is strongly indeterminate, i.e. when distinct sets of parameters yield identical observations. This arises in many practical situations, such as when inferring the distance and power of a radio source (is the source close and weak or far and strong?) or when estimating the amplifier gain and underlying brain activity of an electrophysiological experiment. In this work, we present hierarchical neural posterior estimation (HNPE), a novel method for cracking such indeterminacy by exploiting additional information conveyed by an auxiliary set of observations sharing global parameters. Our method extends recent developments in simulation-based inference (SBI) based on normalizing flows to Bayesian hierarchical models. We validate quantitatively our proposal on a motivating example amenable to analytical solutions and then apply it to invert a well known non-linear model from computational neuroscience, using both simulated and real EEG data.

----

## [1029] Alignment Attention by Matching Key and Query Distributions

**Authors**: *Shujian Zhang, Xinjie Fan, Huangjie Zheng, Korawat Tanwisuth, Mingyuan Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6fd6b030c6afec018415662d0db43f9d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6fd6b030c6afec018415662d0db43f9d-Abstract.html)

**Abstract**:

The neural attention mechanism has been incorporated into deep neural networks to achieve state-of-the-art performance in various domains. Most such models use multi-head self-attention which is appealing for the ability to attend to information from different perspectives. This paper introduces alignment attention that explicitly encourages self-attention to match the distributions of the key and query within each head. The resulting alignment attention networks can be optimized as an unsupervised regularization in the existing attention framework. It is simple to convert any models with self-attention, including pre-trained ones, to the proposed alignment attention. On a variety of language understanding tasks, we show the effectiveness of our method in accuracy, uncertainty estimation, generalization across domains, and robustness to adversarial attacks. We further demonstrate the general applicability of our approach on graph attention and visual question answering, showing the great potential of incorporating our alignment method into various attention-related tasks.

----

## [1030] Settling the Variance of Multi-Agent Policy Gradients

**Authors**: *Jakub Grudzien Kuba, Muning Wen, Linghui Meng, Shangding Gu, Haifeng Zhang, David Mguni, Jun Wang, Yaodong Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6fe6a8a6e6cb710584efc4af0c34ce50-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6fe6a8a6e6cb710584efc4af0c34ce50-Abstract.html)

**Abstract**:

Policy gradient (PG) methods are popular reinforcement learning (RL) methods where a baseline is often applied to reduce the variance of gradient estimates. In multi-agent RL (MARL), although the PG theorem can be naturally extended, the effectiveness of multi-agent PG (MAPG)  methods degrades as the variance of gradient estimates increases rapidly with the number of agents.  In this paper, we offer a rigorous analysis of MAPG methods by, firstly, quantifying the contributions of the number of agents and agents' explorations to the variance of MAPG estimators. Based on this analysis, we derive the optimal baseline (OB) that achieves the minimal variance. In comparison to the OB, we measure the excess variance of existing MARL algorithms such as vanilla MAPG and COMA. Considering using deep neural networks,  we also propose a surrogate version of OB, which can be seamlessly plugged into any existing PG methods in MARL.   On benchmarks of Multi-Agent MuJoCo and StarCraft challenges, our OB technique effectively stabilises training and improves the performance of multi-agent PPO  and COMA algorithms by a significant margin.  Code is released at  \url{https://github.com/morning9393/Optimal-Baseline-for-Multi-agent-Policy-Gradients}.

----

## [1031] For high-dimensional hierarchical models, consider exchangeability of effects across covariates instead of across datasets

**Authors**: *Brian L. Trippe, Hilary K. Finucane, Tamara Broderick*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/6ffad86b9a8dd4a3e98df1b0830d1c8c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6ffad86b9a8dd4a3e98df1b0830d1c8c-Abstract.html)

**Abstract**:

Hierarchical Bayesian methods enable information sharing across regression problems on multiple groups of data. While standard practice is to model regression parameters (effects) as (1) exchangeable across the groups and (2) correlated to differing degrees across covariates, we show that this approach exhibits poor statistical performance when the number of covariates exceeds the number of groups. For instance, in statistical genetics, we might regress dozens of traits (defining groups) for thousands of individuals (responses) on up to millions of genetic variants (covariates). When an analyst has more covariates than groups, we argue that it is often preferable to instead model effects as (1) exchangeable across covariates and (2) correlated to differing degrees across groups. To this end, we propose a hierarchical model expressing our alternative perspective. We devise an empirical Bayes estimator for learning the degree of correlation between groups. We develop theory that demonstrates that our method outperforms the classic approach when the number of covariates dominates the number of groups, and corroborate this result empirically on several high-dimensional multiple regression and classification problems.

----

## [1032] Efficient Algorithms for Learning Depth-2 Neural Networks with General ReLU Activations

**Authors**: *Pranjal Awasthi, Alex Tang, Aravindan Vijayaraghavan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/700fdb2ba62d4554dc268c65add4b16e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/700fdb2ba62d4554dc268c65add4b16e-Abstract.html)

**Abstract**:

We present polynomial time and sample efficient algorithms for learning an unknown depth-2 feedforward neural network with general ReLU activations, under mild non-degeneracy assumptions. In particular, we consider learning an unknown network of the form $f(x) = {a}^{\mathsf{T}}\sigma({W}^\mathsf{T}x+b)$, where $x$ is drawn from the Gaussian distribution, and $\sigma(t) = \max(t,0)$ is the ReLU activation. Prior works for learning networks with ReLU activations assume that the bias ($b$) is zero. In order to deal with the presence of the bias terms, our proposed algorithm consists of robustly decomposing multiple higher order tensors arising from the Hermite expansion of the function $f(x)$. Using these ideas we also establish identifiability of the network parameters under very mild assumptions.

----

## [1033] Controllable and Compositional Generation with Latent-Space Energy-Based Models

**Authors**: *Weili Nie, Arash Vahdat, Anima Anandkumar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/701d804549a4a23d3cae801dac6c2c75-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/701d804549a4a23d3cae801dac6c2c75-Abstract.html)

**Abstract**:

Controllable generation is one of the key requirements for successful adoption of deep generative models in real-world applications, but it still remains as a great challenge. In particular, the compositional ability to generate novel concept combinations is out of reach for most current models. In this work, we use energy-based models (EBMs) to handle compositional generation over a set of attributes. To make them scalable to high-resolution image generation, we introduce an EBM in the latent space of a pre-trained generative model such as StyleGAN. We propose a novel EBM formulation representing the joint distribution of data and attributes together, and we show how sampling from it is formulated as solving an ordinary differential equation (ODE). Given a pre-trained generator, all we need for controllable generation is to train an attribute classifier. Sampling with ODEs is done efficiently in the latent space and is robust to hyperparameters. Thus, our method is simple, fast to train, and efficient to sample. Experimental results show that our method outperforms the state-of-the-art in both conditional sampling and sequential editing. In compositional generation, our method excels at zero-shot generation of unseen attribute combinations. Also, by composing energy functions with logical operators, this work is the first to achieve such compositionality in generating photo-realistic images of resolution 1024x1024.

----

## [1034] Reverse-Complement Equivariant Networks for DNA Sequences

**Authors**: *Vincent Mallet, Jean-Philippe Vert*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/706608cfdbcc1886bb7eea5513f90133-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/706608cfdbcc1886bb7eea5513f90133-Abstract.html)

**Abstract**:

As DNA sequencing technologies keep improving in scale and cost, there is a growing need to develop machine learning models to analyze DNA sequences, e.g., to decipher regulatory signals from DNA fragments bound by a particular protein of interest.  As a double helix made of two complementary strands, a DNA fragment can be sequenced as two equivalent, so-called reverse complement (RC) sequences of nucleotides. To take into account this inherent symmetry of the data in machine learning models can facilitate learning. In this sense, several authors have recently proposed particular RC-equivariant convolutional neural networks (CNNs). However, it remains unknown whether other RC-equivariant architecture exist, which could potentially increase the set of basic models adapted to DNA sequences for practitioners. Here, we close this gap by characterizing the set of all linear RC-equivariant layers, and show in particular that new architectures exist beyond the ones already explored. We further discuss RC-equivariant pointwise nonlinearities adapted to different architectures, as well as RC-equivariant embeddings of $k$-mers as an alternative to one-hot encoding of nucleotides. We show experimentally that the new architectures can outperform existing ones.

----

## [1035] Provably Efficient Reinforcement Learning with Linear Function Approximation under Adaptivity Constraints

**Authors**: *Tianhao Wang, Dongruo Zhou, Quanquan Gu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70a32110fff0f26d301e58ebbca9cb9f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70a32110fff0f26d301e58ebbca9cb9f-Abstract.html)

**Abstract**:

We study reinforcement learning (RL) with linear function approximation under the adaptivity constraint. We consider two popular limited adaptivity models: the batch learning model and the rare policy switch model, and propose two efficient online RL algorithms for episodic linear Markov decision processes, where the transition probability and the reward function can be represented as a linear function of some known feature mapping. In specific, for the batch learning model, our proposed LSVI-UCB-Batch algorithm achieves an $\tilde O(\sqrt{d^3H^3T} + dHT/B)$ regret, where $d$ is the dimension of the feature mapping, $H$ is the episode length, $T$ is the number of interactions and $B$ is the number of batches. Our result suggests that it suffices to use only $\sqrt{T/dH}$ batches to obtain $\tilde O(\sqrt{d^3H^3T})$ regret. For the rare policy switch model, our proposed LSVI-UCB-RareSwitch algorithm enjoys an $\tilde O(\sqrt{d^3H^3T[1+T/(dH)]^{dH/B}})$ regret, which implies that $dH\log T$ policy switches suffice to obtain the $\tilde O(\sqrt{d^3H^3T})$ regret. Our algorithms achieve the same regret as the LSVI-UCB algorithm \citep{jin2020provably}, yet with a substantially smaller amount of adaptivity. We also establish a lower bound for the batch learning model, which suggests that the dependency on $B$ in our regret bound is tight.

----

## [1036] Nonsmooth Implicit Differentiation for Machine-Learning and Optimization

**Authors**: *Jérôme Bolte, Tam Le, Edouard Pauwels, Antonio Silveti-Falls*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70afbf2259b4449d8ae1429e054df1b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70afbf2259b4449d8ae1429e054df1b1-Abstract.html)

**Abstract**:

In view of training increasingly complex learning architectures, we establish a nonsmooth implicit function theorem with an operational calculus. Our result applies to most practical problems (i.e., definable problems) provided that a nonsmooth form of the classical invertibility condition is fulfilled. This approach allows for formal subdifferentiation: for instance, replacing derivatives by Clarke Jacobians in the usual differentiation formulas is fully justified for a wide class of nonsmooth problems. Moreover this calculus is entirely compatible with algorithmic differentiation (e.g., backpropagation). We provide several applications such as training deep equilibrium networks, training neural nets with conic optimization layers, or hyperparameter-tuning for nonsmooth Lasso-type models. To show the sharpness of our assumptions, we present numerical experiments showcasing the extremely pathological gradient dynamics one can encounter when applying implicit algorithmic differentiation without any hypothesis.

----

## [1037] Heuristic-Guided Reinforcement Learning

**Authors**: *Ching-An Cheng, Andrey Kolobov, Adith Swaminathan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70d31b87bd021441e5e6bf23eb84a306-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70d31b87bd021441e5e6bf23eb84a306-Abstract.html)

**Abstract**:

We provide a framework to accelerate reinforcement learning (RL) algorithms by heuristics that are constructed by domain knowledge or offline data.  Tabula rasa RL algorithms require environment interactions or computation that scales with the horizon of the sequential decision-making task.  Using our framework, we show how heuristic-guided RL induces a much shorter horizon sub-problem that provably solves the original task. Our framework can be viewed as a horizon-based regularization for controlling bias and variance in RL under a finite interaction budget.  In theory, we characterize the properties of a good heuristic and the resulting impact on RL acceleration. In particular, we introduce the novel concept of an improvable heuristic that can allow any RL agent to conservatively extrapolate beyond its prior knowledge.  In practice, we instantiate our framework to accelerate several state-of-the-art algorithms in simulated robotic control tasks and procedurally generated games. Our framework complements the rich literature on warm-starting RL using expert demonstrations or exploratory data-sets, and creates a unified channel to inject prior knowledge into RL.

----

## [1038] Statistical Undecidability in Linear, Non-Gaussian Causal Models in the Presence of Latent Confounders

**Authors**: *Konstantin Genin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70d355680e628fe1c552221f690d8da4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70d355680e628fe1c552221f690d8da4-Abstract.html)

**Abstract**:

If causal relationships are linear and acyclic and noise terms are independent and Gaussian, causal orientation is not identified from observational data --- even if faithfulness is satisfied (Spirtes et al., 2002). Shimizu et al. (2006) showed that acyclic, linear, {\bf non}-Gaussian (LiNGAM) causal models {\em are} identified from observational data, so long as no latent confounders are present. That holds even when faithfulness fails. Genin and Mayo-Wilson (2020) refine that result:  not only are causal relationships identified, but causal orientation is {\em statistically decidable}. That means that for every $\epsilon>0,$ there is a method that converges in probability to the correct orientation and, at every sample size, outputs an incorrect orientation with probability less than $\epsilon.$ These results naturally raise questions about what happens in the presence of latent confounders. Hoyer et al. (2008) and Salehkaleybar et al. (2020) show that, although the causal model is not uniquely identified, causal orientation among observed variables is identified in the presence of latent confounders, so long as faithfulness is satisfied. This paper refines these results: although it is possible to converge to the right orientation in the limit, causal orientation is no longer statistically decidable---it is not possible to converge to the correct orientation with finite-sample bounds on the probability of orientation errors, even if faithfulness is satisfied. However, that limiting result suggests several adjustments to the LiNGAM model that may recover decidability.

----

## [1039] A novel notion of barycenter for probability distributions based on optimal weak mass transport

**Authors**: *Elsa Cazelles, Felipe A. Tobar, Joaquín Fontbona*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70d5212dd052b2ef06e5e562f6f9ab9c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70d5212dd052b2ef06e5e562f6f9ab9c-Abstract.html)

**Abstract**:

We introduce weak barycenters of a family of probability distributions, based on the recently developed notion of optimal weak transport of mass by Gozlan et al. (2017) and Backhoff-Veraguas et al. (2020). We provide a theoretical analysis of this object and discuss its interpretation in the light of convex ordering between probability measures. In particular, we show that, rather than averaging the input distributions in a geometric way (as the Wasserstein barycenter based on classic optimal transport does) weak barycenters extract common geometric information shared by all the input distributions, encoded as a latent random variable that underlies all of them. We also provide an iterative algorithm to compute a weak barycenter for a finite family of input distributions, and a stochastic algorithm that computes them for arbitrary populations of laws.  The latter approach is particularly well suited for the streaming setting, i.e., when distributions are observed sequentially. The notion of weak barycenter and our approaches to compute it are illustrated on synthetic examples, validated on 2D real-world data and compared to standard Wasserstein barycenters.

----

## [1040] Temporal-attentive Covariance Pooling Networks for Video Recognition

**Authors**: *Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu, Peihua Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70efdf2ec9b086079795c442636b55fb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70efdf2ec9b086079795c442636b55fb-Abstract.html)

**Abstract**:

For video recognition task, a global representation summarizing the whole contents of the video snippets plays an important role for the final performance. However, existing video architectures usually generate it by using a simple, global average pooling (GAP) method, which has limited ability to capture complex dynamics of videos. For image recognition task, there exist evidences showing that covariance pooling has stronger representation ability than GAP. Unfortunately, such plain covariance pooling used in image recognition is an orderless representative, which cannot model spatio-temporal structure inherent in videos. Therefore, this paper proposes a Temporal-attentive Covariance Pooling (TCP), inserted at the end of deep architectures, to produce powerful video representations. Specifically, our TCP first develops a temporal attention module to adaptively calibrate spatio-temporal features for the succeeding covariance pooling, approximatively producing attentive covariance representations. Then, a temporal covariance pooling performs temporal pooling of the attentive covariance representations to characterize both intra-frame correlations and inter-frame cross-correlations of the calibrated features. As such, the proposed TCP can capture complex temporal dynamics. Finally, a fast matrix power normalization is introduced to exploit geometry of covariance representations. Note that our TCP is model-agnostic and can be flexibly integrated into any video architectures, resulting in TCPNet for effective video recognition. The extensive experiments on six benchmarks (e.g., Kinetics, Something-Something V1 and Charades) using various video architectures show our TCPNet is clearly superior to its counterparts, while having strong generalization ability. The source code is publicly available.

----

## [1041] Revisiting Smoothed Online Learning

**Authors**: *Lijun Zhang, Wei Jiang, Shiyin Lu, Tianbao Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/70fc5f043205720a49d973d280eb83e7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/70fc5f043205720a49d973d280eb83e7-Abstract.html)

**Abstract**:

In this paper, we revisit the problem of smoothed online learning, in which the online learner suffers both a hitting cost and a switching cost, and target two performance metrics: competitive ratio and dynamic regret with switching cost. To bound the competitive ratio, we assume the hitting cost is known to the learner in each round, and investigate the simple idea of balancing the two costs by an optimization problem. Surprisingly, we find that minimizing the hitting cost alone is $\max(1, \frac{2}{\alpha})$-competitive for $\alpha$-polyhedral functions and $1 + \frac{4}{\lambda}$-competitive for $\lambda$-quadratic growth functions, both of which improve state-of-the-art results significantly. Moreover, when the hitting cost is both convex and $\lambda$-quadratic growth, we reduce the competitive ratio to $1 + \frac{2}{\sqrt{\lambda}}$  by minimizing the weighted sum of the hitting cost and the switching cost. To bound the dynamic regret with switching cost, we follow the standard setting of online convex optimization, in which the hitting cost is convex but hidden from the learner before making predictions. We modify Ader, an existing algorithm designed for dynamic regret, slightly to take into account the switching cost when measuring the performance. The proposed algorithm, named as Smoothed Ader, attains an optimal $O(\sqrt{T(1+P_T)})$ bound for dynamic regret with switching cost, where $P_T$ is the path-length of the comparator sequence. Furthermore, if the hitting cost is accessible in the beginning of each round, we obtain a similar guarantee without the bounded gradient condition, and establish an $\Omega(\sqrt{T(1+P_T)})$ lower bound to confirm the optimality.

----

## [1042] Marginalised Gaussian Processes with Nested Sampling

**Authors**: *Fergus Simpson, Vidhi Lalchand, Carl Edward Rasmussen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/712a67567ec10c52c2b966224cf94d1e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/712a67567ec10c52c2b966224cf94d1e-Abstract.html)

**Abstract**:

Gaussian Process models are a rich distribution over functions with inductive biases controlled by a kernel function. Learning occurs through optimisation of the kernel hyperparameters using the marginal likelihood as the objective. This work proposes nested sampling as a means of marginalising kernel hyperparameters,  because it is a technique that is well-suited to exploring complex, multi-modal distributions. We benchmark against Hamiltonian Monte Carlo on time-series and two-dimensional regression tasks, finding that a principled approach to quantifying hyperparameter uncertainty substantially improves the quality of prediction intervals.

----

## [1043] Provable Benefits of Actor-Critic Methods for Offline Reinforcement Learning

**Authors**: *Andrea Zanette, Martin J. Wainwright, Emma Brunskill*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/713fd63d76c8a57b16fc433fb4ae718a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/713fd63d76c8a57b16fc433fb4ae718a-Abstract.html)

**Abstract**:

Actor-critic methods are widely used in offline reinforcement learningpractice, but are not so well-understood theoretically. We propose a newoffline actor-critic algorithm that naturally incorporates the pessimism principle, leading to several key advantages compared to the state of the art. The algorithm can operate when the Bellman evaluation operator is closed with respect to the action value function of the actor's policies; this is a more general setting than the low-rank MDP model. Despite the added generality, the procedure is computationally tractable as it involves the solution of a sequence of second-order programs.We prove an upper bound on the suboptimality gap of the policy returned by the procedure that depends on the data coverage of any arbitrary, possibly data dependent comparator policy.The achievable guarantee is complemented with a minimax lower bound that is matching up to logarithmic factors.

----

## [1044] Bayesian Bellman Operators

**Authors**: *Mattie Fellows, Kristian Hartikainen, Shimon Whiteson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7180cffd6a8e829dacfc2a31b3f72ece-Abstract.html)

**Abstract**:

We introduce a novel perspective on Bayesian reinforcement learning (RL); whereas existing approaches infer a posterior over the transition distribution or Q-function, we characterise the uncertainty in the Bellman operator. Our Bayesian Bellman operator (BBO) framework is motivated by the insight that when bootstrapping is introduced, model-free approaches actually infer a posterior over Bellman operators, not value functions. In this paper, we use BBO to provide a rigorous theoretical analysis of model-free Bayesian RL to better understand its relationship to established frequentist RL methodologies. We prove that Bayesian solutions are consistent with frequentist RL solutions, even when approximate inference is used, and derive conditions for which  convergence properties hold. Empirically, we demonstrate that algorithms derived from the BBO framework have sophisticated deep exploration properties that enable them to solve continuous control tasks at which state-of-the-art regularised actor-critic algorithms fail catastrophically.

----

## [1045] Uncertainty Calibration for Ensemble-Based Debiasing Methods

**Authors**: *Ruibin Xiong, Yimeng Chen, Liang Pang, Xueqi Cheng, Zhi-Ming Ma, Yanyan Lan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/71a8b2ffe0b594a5c1b3c28090384fd7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/71a8b2ffe0b594a5c1b3c28090384fd7-Abstract.html)

**Abstract**:

Ensemble-based debiasing methods have been shown effective in mitigating the reliance of classifiers on specific dataset bias, by exploiting the output of a bias-only model to adjust the learning target. In this paper, we focus on the bias-only model in these ensemble-based methods, which plays an important role but has not gained much attention in the existing literature. Theoretically, we prove that the debiasing performance can be damaged by inaccurate uncertainty estimations of the bias-only model. Empirically, we show that existing bias-only models fall short in producing accurate uncertainty estimations. Motivated by these findings, we propose to conduct calibration on the bias-only model, thus achieving a three-stage ensemble-based debiasing framework, including bias modeling, model calibrating, and debiasing. Experimental results on NLI and fact verification tasks show that our proposed three-stage debiasing framework consistently outperforms the traditional two-stage one in out-of-distribution accuracy.

----

## [1046] Provably Faster Algorithms for Bilevel Optimization

**Authors**: *Junjie Yang, Kaiyi Ji, Yingbin Liang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/71cc107d2e0408e60a3d3c44f47507bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/71cc107d2e0408e60a3d3c44f47507bd-Abstract.html)

**Abstract**:

Bilevel optimization has been widely applied in many important machine learning applications such as hyperparameter optimization and meta-learning. Recently, several momentum-based algorithms have been proposed to solve bilevel optimization problems faster. However, those momentum-based algorithms do not achieve provably better computational complexity than $\mathcal{\widetilde O}(\epsilon^{-2})$ of the SGD-based algorithm. In this paper, we propose two new algorithms for bilevel optimization, where the first algorithm adopts momentum-based recursive iterations, and the second algorithm adopts recursive gradient estimations in nested loops to decrease the variance. We show that both algorithms achieve the complexity of $\mathcal{\widetilde O}(\epsilon^{-1.5})$, which outperforms all existing algorithms by the order of magnitude. Our experiments validate our theoretical results and demonstrate the superior empirical performance of our algorithms in hyperparameter applications.

----

## [1047] Neo-GNNs: Neighborhood Overlap-aware Graph Neural Networks for Link Prediction

**Authors**: *Seongjun Yun, Seoyoon Kim, Junhyun Lee, Jaewoo Kang, Hyunwoo J. Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/71ddb91e8fa0541e426a54e538075a5a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/71ddb91e8fa0541e426a54e538075a5a-Abstract.html)

**Abstract**:

Graph Neural Networks (GNNs) have been widely applied to various fields for learning over graph-structured data. They have shown significant improvements over traditional heuristic methods in various tasks such as node classification and graph classification. However, since GNNs heavily rely on smoothed node features rather than graph structure, they often show poor performance than simple heuristic methods in link prediction where the structural information, e.g., overlapped neighborhoods, degrees, and shortest paths, is crucial. To address this limitation, we propose Neighborhood Overlap-aware Graph Neural Networks (Neo-GNNs) that learn useful structural features from an adjacency matrix and estimate overlapped neighborhoods for link prediction. Our Neo-GNNs generalize neighborhood overlap-based heuristic methods and handle overlapped multi-hop neighborhoods. Our extensive experiments on Open Graph Benchmark datasets (OGB) demonstrate that Neo-GNNs consistently achieve state-of-the-art performance in link prediction.

----

## [1048] Self-Supervised Multi-Object Tracking with Cross-input Consistency

**Authors**: *Favyen Bastani, Songtao He, Samuel Madden*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/71e09b16e21f7b6919bbfc43f6a5b2f0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/71e09b16e21f7b6919bbfc43f6a5b2f0-Abstract.html)

**Abstract**:

In this paper, we propose a self-supervised learning procedure for training a robust multi-object tracking (MOT) model given only unlabeled video. While several self-supervisory learning signals have been proposed in prior work on single-object tracking, such as color propagation and cycle-consistency, these signals are not effective for training RNN models, which are needed to achieve accurate MOT: they yield degenerate models that, for instance, always match new detections to tracks with the closest initial detections. We propose a novel self-supervisory signal that we call cross-input consistency: we construct two distinct inputs for the same sequence of video, by hiding different information about the sequence in each input. We then compute tracks in that sequence by applying an RNN model independently on each input, and train the model to produce consistent tracks across the two inputs. We evaluate our unsupervised method on MOT17 and KITTI --- remarkably, we find that, despite training only on unlabeled video, our unsupervised approach outperforms four supervised methods published in the last 1--2 years, including Tracktor++, FAMNet, GSM, and mmMOT.

----

## [1049] Tree in Tree: from Decision Trees to Decision Graphs

**Authors**: *Bingzhao Zhu, Mahsa Shoaran*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html)

**Abstract**:

Decision trees have been widely used as classifiers in many machine learning applications thanks to their lightweight and interpretable decision process. This paper introduces Tree in Tree decision graph (TnT), a framework that extends the conventional decision tree to a more generic and powerful directed acyclic graph. TnT constructs decision graphs by recursively growing decision trees inside the internal or leaf nodes instead of greedy training. The time complexity of TnT is linear to the number of nodes in the graph, therefore it can construct decision graphs on large datasets. Compared to decision trees, we show that TnT achieves better classification performance with reduced model size, both as a stand-alone classifier and as a base-estimator in bagging/AdaBoost ensembles. Our proposed model is a novel, more efficient and accurate alternative to the widely-used decision trees.

----

## [1050] Test-time Collective Prediction

**Authors**: *Celestine Mendler-Dünner, Wenshuo Guo, Stephen Bates, Michael I. Jordan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/722caafb4825ef5d8670710fa29087cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/722caafb4825ef5d8670710fa29087cf-Abstract.html)

**Abstract**:

An increasingly common setting in machine learning involves multiple parties, each with their own data, who want to jointly make predictions on future test points. Agents wish to benefit from the collective expertise of the full set of agents to make better predictions than they would individually, but may not be willing to release labeled data or model parameters. In this work, we explore a decentralized mechanism to make collective predictions at test time, that is inspired by the literature in social science on human consensus-making. Building on a query model to facilitate information exchange among agents, our approach leverages each agent’s pre-trained model without relying on external validation, model retraining, or data pooling. A theoretical analysis shows that our approach recovers inverse mean-squared-error (MSE) weighting in the large-sample limit which is known to be the optimal way to combine independent, unbiased estimators. Empirically, we demonstrate that our scheme effectively combines models with differing quality across the input space: the proposed consensus prediction achieves significant gains over classical model averaging, and even outperforms weighted averaging schemes that have access to additional validation data. Finally, we propose a decentralized Jackknife procedure as a tool to evaluate the sensitivity of the collective predictions with respect to a single agent's opinion.

----

## [1051] A Continuous Mapping For Augmentation Design

**Authors**: *Keyu Tian, Chen Lin, Ser-Nam Lim, Wanli Ouyang, Puneet K. Dokania, Philip H. S. Torr*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7230b2b03e2da37352abf1a659545b44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7230b2b03e2da37352abf1a659545b44-Abstract.html)

**Abstract**:

Automated data augmentation (ADA) techniques have played an important role in boosting the performance of deep models. Such techniques mostly aim to optimize a parameterized distribution over a discrete augmentation space. Thus, are restricted by the discretization of the search space which normally is handcrafted. To overcome the limitations, we take the first step to constructing a continuous mapping from $\mathbb{R}^d$ to image transformations (an augmentation space). Using this mapping, we take a novel approach where 1) we pose the ADA as a continuous optimization problem over the parameters of the augmentation distribution; and 2) use Stochastic Gradient Langevin Dynamics to learn and sample augmentations. This allows us to potentially explore the space of infinitely many possible augmentations, which otherwise was not possible due to the discretization of the space. This view of ADA is radically different from the standard discretization based view of ADA, and it opens avenues for utilizing the vast efficient gradient-based algorithms available for continuous optimization problems. Results over multiple benchmarks demonstrate the efficiency improvement of this work compared with previous methods.

----

## [1052] Neural Routing by Memory

**Authors**: *Kaipeng Zhang, Zhenqiang Li, Zhifeng Li, Wei Liu, Yoichi Sato*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7241bd19bb709da0f46807bde88aed25-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7241bd19bb709da0f46807bde88aed25-Abstract.html)

**Abstract**:

Recent Convolutional Neural Networks (CNNs) have achieved significant success by stacking multiple convolutional blocks, named procedures in this paper, to extract semantic features. However, they use the same procedure sequence for all inputs, regardless of the intermediate features.This paper proffers a simple yet effective idea of constructing parallel procedures and assigning similar intermediate features to the same specialized procedures in a divide-and-conquer fashion. It relieves each procedure's learning difficulty and thus leads to superior performance. Specifically, we propose a routing-by-memory mechanism for existing CNN architectures. In each stage of the network, we introduce parallel Procedural Units (PUs). A PU consists of a memory head and a procedure. The memory head maintains a summary of a type of features. For an intermediate feature, we search its closest memory and forward it to the corresponding procedure in both training and testing. In this way, different procedures are tailored to different features and therefore tackle them better.Networks with the proposed mechanism can be trained efficiently using a four-step training strategy. Experimental results show that our method improves VGGNet, ResNet, and EfficientNet's accuracies on Tiny ImageNet, ImageNet, and CIFAR-100 benchmarks with a negligible extra computational cost.

----

## [1053] GeoMol: Torsional Geometric Generation of Molecular 3D Conformer Ensembles

**Authors**: *Octavian Ganea, Lagnajit Pattanaik, Connor W. Coley, Regina Barzilay, Klavs F. Jensen, William H. Green Jr., Tommi S. Jaakkola*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/725215ed82ab6306919b485b81ff9615-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/725215ed82ab6306919b485b81ff9615-Abstract.html)

**Abstract**:

Prediction of a moleculeâ€™s 3D conformer ensemble from the molecular graph holds a key role in areas of cheminformatics and drug discovery. Existing generative models have several drawbacks including lack of modeling important molecular geometry elements (e.g., torsion angles), separate optimization stages prone to error accumulation, and the need for structure fine-tuning based on approximate classical force-fields or computationally expensive methods. We propose GEOMOL --- an end-to-end, non-autoregressive, and SE(3)-invariant machine learning approach to generate distributions of low-energy molecular 3D conformers. Leveraging the power of message passing neural networks (MPNNs) to capture local and global graph information, we predict local atomic 3D structures and torsion angles, avoid- ing unnecessary over-parameterization of the geometric degrees of freedom (e.g., one angle per non-terminal bond). Such local predictions suffice both for both the training loss computation and for the full deterministic conformer assembly (at test time). We devise a non-adversarial optimal transport based loss function to promote diverse conformer generation. GEOMOL predominantly outperforms popular open-source, commercial, or state-of-the-art machine learning (ML) models, while achieving significant speed-ups. We expect such differentiable 3D structure generators to significantly impact molecular modeling and related applications.

----

## [1054] CANITA: Faster Rates for Distributed Convex Optimization with Communication Compression

**Authors**: *Zhize Li, Peter Richtárik*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7274a60c83145b1082be9caa91926ecf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7274a60c83145b1082be9caa91926ecf-Abstract.html)

**Abstract**:

Due to the high communication cost in distributed and federated learning, methods relying on compressed communication are becoming increasingly popular. Besides, the best theoretically and practically performing gradient-type methods invariably rely on some form of acceleration/momentum to reduce the number of communications (faster convergence), e.g., Nesterov's accelerated gradient descent [31, 32] and Adam [14]. In order to combine the benefits of communication compression and convergence acceleration, we propose a \emph{compressed and accelerated} gradient method based on ANITA [20] for distributed optimization, which we call CANITA. Our CANITA achieves the \emph{first accelerated rate} $O\bigg(\sqrt{\Big(1+\sqrt{\frac{\omega^3}{n}}\Big)\frac{L}{\epsilon}} + \omega\big(\frac{1}{\epsilon}\big)^{\frac{1}{3}}\bigg)$, which improves upon the state-of-the-art non-accelerated rate  $O\left((1+\frac{\omega}{n})\frac{L}{\epsilon} + \frac{\omega^2+\omega}{\omega+n}\frac{1}{\epsilon}\right)$ of DIANA [12] for distributed general convex problems, where $\epsilon$ is the target error,  $L$ is the smooth parameter of the objective, $n$ is the number of machines/devices, and $\omega$ is the compression parameter (larger $\omega$ means more compression can be applied, and no compression implies $\omega=0$). Our results show that as long as the number of devices $n$ is large (often true in distributed/federated learning), or the compression $\omega$ is not very high,  CANITA achieves the faster convergence rate $O\Big(\sqrt{\frac{L}{\epsilon}}\Big)$, i.e., the number of communication rounds is $O\Big(\sqrt{\frac{L}{\epsilon}}\Big)$ (vs. $O\big(\frac{L}{\epsilon}\big)$ achieved by previous works). As a result, CANITA enjoys the advantages of both compression (compressed communication in each round) and acceleration (much fewer communication rounds).

----

## [1055] Drop-DTW: Aligning Common Signal Between Sequences While Dropping Outliers

**Authors**: *Nikita Dvornik, Isma Hadji, Konstantinos G. Derpanis, Animesh Garg, Allan D. Jepson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/729c68884bd359ade15d5f163166738a-Abstract.html)

**Abstract**:

In this work, we consider the problem of sequence-to-sequence alignment for signals containing outliers.  Assuming the absence of outliers, the standard Dynamic Time Warping (DTW) algorithm efficiently computes the optimal alignment between two (generally) variable-length sequences.  While DTW is robust to temporal shifts and dilations of the signal, it fails to align sequences in a meaningful way in the presence of outliers that can be arbitrarily interspersed in the sequences.  To address this problem, we introduce Drop-DTW, a novel algorithm that aligns the common signal between the sequences while automatically dropping the outlier elements from the matching.  The entire procedure is implemented as a single dynamic program that is efficient and fully differentiable.  In our experiments, we show that Drop-DTW is a robust similarity measure for sequence retrieval and demonstrate its effectiveness as a training loss on diverse applications. With Drop-DTW, we address temporal step localization on instructional videos, representation learning from noisy videos, and cross-modal representation learning for audio-visual retrieval and localization. In all applications, we take a weakly- or unsupervised approach and demonstrate state-of-the-art results under these settings.

----

## [1056] Safe Reinforcement Learning with Natural Language Constraints

**Authors**: *Tsung-Yen Yang, Michael Y. Hu, Yinlam Chow, Peter J. Ramadge, Karthik Narasimhan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/72f67e70f6b7cdc4cc893edaddf0c4c6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/72f67e70f6b7cdc4cc893edaddf0c4c6-Abstract.html)

**Abstract**:

While safe reinforcement learning (RL) holds great promise for many practical applications like robotics or autonomous cars, current approaches require specifying constraints in mathematical form. Such specifications demand domain expertise, limiting the adoption of safe RL. In this paper, we propose learning to interpret natural language constraints for safe RL. To this end, we first introduce HAZARDWORLD, a new multi-task benchmark that requires an agent to optimize reward while not violating constraints specified in free-form text. We then develop an agent with a modular architecture that can interpret and adhere to such textual constraints while learning new tasks. Our model consists of (1) a constraint interpreter that encodes textual constraints into spatial and temporal representations of forbidden states, and (2) a policy network that uses these representations to produce a policy achieving minimal constraint violations during training. Across different domains in HAZARDWORLD, we show that our method achieves higher rewards (up to11x) and fewer constraint violations (by 1.8x) compared to existing approaches. However, in terms of absolute performance, HAZARDWORLD still poses significant challenges for agents to learn efficiently, motivating the need for future work.

----

## [1057] Compositional Modeling of Nonlinear Dynamical Systems with ODE-based Random Features

**Authors**: *Thomas M. McDonald, Mauricio A. Álvarez*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/72fe6f9fdab5f4d465ac6da028e4544c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/72fe6f9fdab5f4d465ac6da028e4544c-Abstract.html)

**Abstract**:

Effectively modeling phenomena present in highly nonlinear dynamical systems whilst also accurately quantifying uncertainty is a challenging task, which often requires problem-specific techniques. We present a novel, domain-agnostic approach to tackling this problem, using compositions of physics-informed random features, derived from ordinary differential equations. The architecture of our model leverages recent advances in approximate inference for deep Gaussian processes, such as layer-wise weight-space approximations which allow us to incorporate random Fourier features, and stochastic variational inference for approximate Bayesian inference. We provide evidence that our model is capable of capturing highly nonlinear behaviour in real-world multivariate time series data. In addition, we find that our approach achieves comparable performance to a number of other probabilistic models on benchmark regression tasks.

----

## [1058] Implicit Semantic Response Alignment for Partial Domain Adaptation

**Authors**: *Wenxiao Xiao, Zhengming Ding, Hongfu Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/731b03008e834f92a03085ef47061c4a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/731b03008e834f92a03085ef47061c4a-Abstract.html)

**Abstract**:

Partial Domain Adaptation (PDA) addresses the unsupervised domain adaptation problem where the target label space is a subset of the source label space. Most state-of-art PDA methods tackle the inconsistent label space by assigning weights to classes or individual samples, in an attempt to discard the source data that belongs to the irrelevant classes. However, we believe samples from those extra categories would still contain valuable information to promote positive transfer. In this paper, we propose the Implicit Semantic Response Alignment to explore the intrinsic relationships among different categories by applying a weighted schema on the feature level. Specifically, we design a class2vec module to extract the implicit semantic topics from the visual features. With an attention layer, we calculate the semantic response according to each implicit semantic topic. Then semantic responses of source and target data are aligned to retain the relevant information contained in multiple categories by weighting the features, instead of samples. Experiments on several cross-domain benchmark datasets demonstrate the effectiveness of our method over the state-of-the-art PDA methods. Moreover, we elaborate in-depth analyses to further explore implicit semantic alignment.

----

## [1059] ToAlign: Task-Oriented Alignment for Unsupervised Domain Adaptation

**Authors**: *Guoqiang Wei, Cuiling Lan, Wenjun Zeng, Zhizheng Zhang, Zhibo Chen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/731c83db8d2ff01bdc000083fd3c3740-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/731c83db8d2ff01bdc000083fd3c3740-Abstract.html)

**Abstract**:

Unsupervised domain adaptive classifcation intends to improve the classifcation performance on unlabeled target domain. To alleviate the adverse effect of domain shift, many approaches align the source and target domains in the feature space. However, a feature is usually taken as a whole for alignment without explicitly making domain alignment proactively serve the classifcation task, leading to sub-optimal solution. In this paper, we propose an effective Task-oriented Alignment (ToAlign) for unsupervised domain adaptation (UDA). We study what features should be aligned across domains and propose to make the domain alignment proactively serve classifcation by performing feature decomposition and alignment under the guidance of the prior knowledge induced from the classifcation task itself. Particularly, we explicitly decompose a feature in the source domain into a task-related/discriminative feature that should be aligned, and a task-irrelevant feature that should be avoided/ignored, based on the classifcation meta-knowledge. Extensive experimental results on various benchmarks (e.g., Offce-Home, Visda-2017, and DomainNet) under different domain adaptation settings demonstrate the effectiveness of ToAlign which helps achieve the state-of-the-art performance. The code is publicly available at https://github.com/microsoft/UDA.

----

## [1060] Prior-independent Dynamic Auctions for a Value-maximizing Buyer

**Authors**: *Yuan Deng, Hanrui Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/735143e9ff8c47def504f1ba0442df98-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/735143e9ff8c47def504f1ba0442df98-Abstract.html)

**Abstract**:

We study prior-independent dynamic auction design with production costs for a value-maximizing buyer, a paradigm that is becoming prevalent recently following the development of automatic bidding algorithms in advertising platforms. In contrast to a utility-maximizing buyer, who maximizes the difference between her total value and total payment, a value-maximizing buyer aims to maximize her total value subject to a return on investment (ROI) constraint. Our main result is a dynamic mechanism with regret $\tilde{O}(T^{2/3})$, where $T$ is the time horizon, against the first-best benchmark, i.e., the maximum amount of revenue the seller can extract assuming all values of the buyer are publicly known.

----

## [1061] Safe Reinforcement Learning by Imagining the Near Future

**Authors**: *Garrett Thomas, Yuping Luo, Tengyu Ma*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/73b277c11266681122132d024f53a75b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/73b277c11266681122132d024f53a75b-Abstract.html)

**Abstract**:

Safe reinforcement learning is a promising path toward applying reinforcement learning algorithms to real-world problems, where suboptimal behaviors may lead to actual negative consequences. In this work, we focus on the setting where unsafe states can be avoided by planning ahead a short time into the future. In this setting, a model-based agent with a sufficiently accurate model can avoid unsafe states.We devise a model-based algorithm that heavily penalizes unsafe trajectories, and derive guarantees that our algorithm can avoid unsafe states under certain assumptions. Experiments demonstrate that our algorithm can achieve competitive rewards with fewer safety violations in several continuous control tasks.

----

## [1062] Contrastive Active Inference

**Authors**: *Pietro Mazzaglia, Tim Verbelen, Bart Dhoedt*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/73c730319cf839f143bf40954448ce39-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/73c730319cf839f143bf40954448ce39-Abstract.html)

**Abstract**:

Active inference is a unifying theory for perception and action resting upon the idea that the brain maintains an internal model of the world by minimizing free energy. From a behavioral perspective, active inference agents can be seen as self-evidencing beings that act to fulfill their optimistic predictions, namely preferred outcomes or goals. In contrast, reinforcement learning requires human-designed rewards to accomplish any desired outcome. Although active inference could provide a more natural self-supervised objective for control, its applicability has been limited because of the shortcomings in scaling the approach to complex environments. In this work, we propose a contrastive objective for active inference that strongly reduces the computational burden in learning the agent's generative model and planning future actions. Our method performs notably better than likelihood-based active inference in image-based tasks, while also being computationally cheaper and easier to train. We compare to reinforcement learning agents that have access to human-designed reward functions, showing that our approach closely matches their performance. Finally, we also show that contrastive methods perform significantly better in the case of distractors in the environment and that our method is able to generalize goals to variations in the background.

----

## [1063] Overparameterization Improves Robustness to Covariate Shift in High Dimensions

**Authors**: *Nilesh Tripuraneni, Ben Adlam, Jeffrey Pennington*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/73fed7fd472e502d8908794430511f4d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/73fed7fd472e502d8908794430511f4d-Abstract.html)

**Abstract**:

A significant obstacle in the development of robust machine learning models is \emph{covariate shift}, a form of distribution shift that occurs when the input distributions of the training and test sets differ while the conditional label distributions remain the same. Despite the prevalence of covariate shift in real-world applications, a theoretical understanding in the context of modern machine learning has remained lacking. In this work, we examine the exact high-dimensional asymptotics of random feature regression under covariate shift and present a precise characterization of the limiting test error, bias, and variance in this setting. Our results motivate a natural partial order over covariate shifts that provides a sufficient condition for determining when the shift will harm (or even help) test performance. We find that overparameterized models exhibit enhanced robustness to covariate shift, providing one of the first theoretical explanations for this ubiquitous empirical phenomenon. Additionally, our analysis reveals an exact linear relationship between the in-distribution and out-of-distribution generalization performance, offering an explanation for this surprising recent observation.

----

## [1064] Logarithmic Regret in Feature-based Dynamic Pricing

**Authors**: *Jianyu Xu, Yu-Xiang Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/742141ceda6b8f6786609d31c8ef129f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/742141ceda6b8f6786609d31c8ef129f-Abstract.html)

**Abstract**:

Feature-based dynamic pricing is an increasingly popular model of setting prices for highly differentiated products with applications in digital marketing, online sales, real estate and so on. The problem was formally studied as an online learning problem [Javanmard & Nazerzadeh, 2019] where a seller needs to propose prices on the fly for a sequence of $T$ products based on their features $x$ while having a small regret relative to the best ---"omniscient"--- pricing strategy she could have come up with in hindsight. We revisit this problem and provide two algorithms (EMLP and ONSP) for stochastic and adversarial feature settings, respectively, and prove the optimal $O(d\log{T})$ regret bounds for both. In comparison, the best existing results are $O\left(\min\left\{\frac{1}{\lambda_{\min}^2}\log{T}, \sqrt{T}\right\}\right)$ and $O(T^{2/3})$ respectively, with $\lambda_{\min}$ being the smallest eigenvalue of $\mathbb{E}[xx^T]$ that could be arbitrarily close to $0$.  We also prove an $\Omega(\sqrt{T})$ information-theoretic lower bound for a slightly more general setting, which demonstrates that "knowing-the-demand-curve" leads to an exponential improvement in feature-based dynamic pricing.

----

## [1065] Dimension-free empirical entropy estimation

**Authors**: *Doron Cohen, Aryeh Kontorovich, Aaron Koolyk, Geoffrey Wolfer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/74378afe5e8b20910cf1f939e57f0480-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/74378afe5e8b20910cf1f939e57f0480-Abstract.html)

**Abstract**:

We seek an entropy estimator for discrete distributions with fully empirical accuracy bounds. As stated, this goal is infeasible without some prior assumptions on the distribution. We discover that a certain information moment assumption renders the problem feasible. We argue that the moment assumption is natural and, in some sense, {\em minimalistic} --- weaker than finite support or tail decay conditions. Under the moment assumption, we provide the first finite-sample entropy estimates for infinite alphabets, nearly recovering the known minimax rates. Moreover, we demonstrate that our empirical bounds are significantly sharper than the state-of-the-art bounds, for various natural distributions and non-trivial sample regimes. Along the way, we give a dimension-free analogue of the Cover-Thomas result on entropy continuity (with respect to total variation distance) for finite alphabets, which may be of independent interest.

----

## [1066] Towards Biologically Plausible Convolutional Networks

**Authors**: *Roman Pogodin, Yash Mehta, Timothy P. Lillicrap, Peter E. Latham*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/746b02b6680562f44ad7526675bac026-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/746b02b6680562f44ad7526675bac026-Abstract.html)

**Abstract**:

Convolutional networks are ubiquitous in deep learning. They are particularly useful for images, as they reduce the number of parameters, reduce training time, and increase accuracy. However, as a model of the brain they are seriously problematic, since they require weight sharing - something real neurons simply cannot do. Consequently, while neurons in the brain can be locally connected (one of the features of convolutional networks), they cannot be convolutional. Locally connected but non-convolutional networks, however, significantly underperform convolutional ones. This is troublesome for studies that use convolutional networks to explain activity in the visual system. Here we study plausible alternatives to weight sharing that aim at the same regularization principle, which is to make each neuron within a pool react similarly to identical inputs. The most natural way to do that is by showing the network multiple translations of the same image, akin to saccades in animal vision. However, this approach requires many translations, and doesn't remove the performance gap. We propose instead to add lateral connectivity to a locally connected network, and allow learning via Hebbian plasticity. This requires the network to pause occasionally for a sleep-like phase of "weight sharing". This method enables locally connected networks to achieve nearly convolutional performance on ImageNet and improves their fit to the ventral stream data, thus supporting convolutional networks as a model of the visual stream.

----

## [1067] DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification

**Authors**: *Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/747d3443e319a22747fbb873e8b2f9f2-Abstract.html)

**Abstract**:

Attention is sparse in vision transformers. We observe the final prediction in vision transformers is only based on a subset of most informative tokens, which is sufficient for accurate image recognition. Based on this observation, we propose a dynamic token sparsification framework to prune redundant tokens progressively and dynamically based on the input. Specifically, we devise a lightweight prediction module to estimate the importance score of each token given the current features. The module is added to different layers to prune redundant tokens hierarchically. To optimize the prediction module in an end-to-end manner, we propose an attention masking strategy to differentiably prune a token by blocking its interactions with other tokens. Benefiting from the nature of self-attention, the unstructured sparse tokens are still hardware friendly, which makes our framework easy to achieve actual speed-up. By hierarchically pruning 66% of the input tokens, our method greatly reduces 31% $\sim$ 37%  FLOPs and improves the throughput by over 40% while the drop of accuracy is within 0.5% for various vision transformers. Equipped with the dynamic token sparsification framework,  DynamicViT models can achieve very competitive complexity/accuracy trade-offs compared to state-of-the-art CNNs and vision transformers on ImageNet. Code is available at https://github.com/raoyongming/DynamicViT

----

## [1068] Learning Transferable Adversarial Perturbations

**Authors**: *Krishna Kanth Nakka, Mathieu Salzmann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7486cef2522ee03547cfb970a404a874-Abstract.html)

**Abstract**:

While effective, deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, recent work has shown that such attacks could be generated by another deep network, leading to significant speedups over optimization-based perturbations. However, the ability of such generative methods to generalize to different test-time situations has not been systematically studied. In this paper, we, therefore, investigate the transferability of generated perturbations when the conditions at inference time differ from the training ones in terms of the target architecture, target data, and target task. Specifically, we identify the mid-level features extracted by the intermediate layers of DNNs as common ground across different architectures, datasets, and tasks. This lets us introduce a loss function based on such mid-level features to learn an effective, transferable perturbation generator. Our experiments demonstrate that our approach outperforms the state-of-the-art universal and transferable attack strategies.

----

## [1069] PortaSpeech: Portable and High-Quality Generative Text-to-Speech

**Authors**: *Yi Ren, Jinglin Liu, Zhou Zhao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/748d6b6ed8e13f857ceaa6cfbdca14b8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/748d6b6ed8e13f857ceaa6cfbdca14b8-Abstract.html)

**Abstract**:

Non-autoregressive text-to-speech (NAR-TTS) models such as FastSpeech 2 and Glow-TTS can synthesize high-quality speech from the given text in parallel. After analyzing two kinds of generative NAR-TTS models (VAE and normalizing flow), we find that: VAE is good at capturing the long-range semantics features (e.g., prosody) even with small model size but suffers from blurry and unnatural results; and normalizing flow is good at reconstructing the frequency bin-wise details but performs poorly when the number of model parameters is limited. Inspired by these observations, to generate diverse speech with natural details and rich prosody using a lightweight architecture, we propose PortaSpeech, a portable and high-quality generative text-to-speech model. Specifically, 1) to model both the prosody and mel-spectrogram details accurately, we adopt a lightweight VAE with an enhanced prior followed by a flow-based post-net with strong conditional inputs as the main architecture. 2) To further compress the model size and memory footprint, we introduce the grouped parameter sharing mechanism to the affine coupling layers in the post-net. 3) To improve the expressiveness of synthesized speech and reduce the dependency on accurate fine-grained alignment between text and speech, we propose a linguistic encoder with mixture alignment combining hard word-level alignment and soft phoneme-level alignment, which explicitly extracts word-level semantic information.  Experimental results show that PortaSpeech outperforms other TTS models in both voice quality and prosody modeling in terms of subjective and objective evaluation metrics, and shows only a slight performance degradation when reducing the model parameters to 6.7M (about 4x model size and 3x runtime memory compression ratio compared with FastSpeech 2). Our extensive ablation studies demonstrate that each design in PortaSpeech is effective.

----

## [1070] Exponential Graph is Provably Efficient for Decentralized Deep Training

**Authors**: *Bicheng Ying, Kun Yuan, Yiming Chen, Hanbin Hu, Pan Pan, Wotao Yin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/74e1ed8b55ea44fd7dbb685c412568a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/74e1ed8b55ea44fd7dbb685c412568a4-Abstract.html)

**Abstract**:

Decentralized SGD is an emerging training method for deep learning known for its much less (thus faster) communication per iteration, which relaxes the averaging step in parallel SGD to inexact averaging. The less exact the averaging is, however, the more the total iterations the training needs to take. Therefore, the key to making decentralized SGD efficient is to realize nearly-exact averaging using little communication. This requires a skillful choice of communication topology, which is an under-studied topic in decentralized optimization.In this paper, we study so-called exponential graphs where every node is connected to $O(\log(n))$ neighbors and $n$ is the total number of nodes. This work proves such graphs can lead to both fast communication and effective averaging simultaneously. We also discover that a sequence of $\log(n)$ one-peer exponential graphs, in which each node communicates to one single neighbor per iteration, can together achieve exact averaging. This favorable property enables one-peer exponential graph to average as effective as its static counterpart but communicates more efficiently. We apply these exponential graphs in decentralized (momentum) SGD to obtain the state-of-the-art balance between per-iteration communication and iteration complexity among all commonly-used topologies. Experimental results on a variety of tasks and models demonstrate that decentralized (momentum) SGD over exponential graphs promises both fast and high-quality training. Our code is implemented through BlueFog and available at https://github.com/Bluefog-Lib/NeurIPS2021-Exponential-Graph.

----

## [1071] CLIP-It! Language-Guided Video Summarization

**Authors**: *Medhini Narasimhan, Anna Rohrbach, Trevor Darrell*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7503cfacd12053d309b6bed5c89de212-Abstract.html)

**Abstract**:

A generic video summary is an abridged version of a video that conveys the whole story and features the most important scenes. Yet the importance of scenes in a video is often subjective, and users should have the option of customizing the summary by using natural language to specify what is important to them. Further, existing models for fully automatic generic summarization have not exploited available language models, which can serve as an effective prior for saliency. This work introduces CLIP-It, a single framework for addressing both generic and query-focused video summarization, typically approached separately in the literature. We propose a language-guided multimodal transformer that learns to score frames in a video based on their importance relative to one another and their correlation with a user-defined query (for query-focused summarization) or an automatically generated dense video caption (for generic video summarization). Our model can be extended to the unsupervised setting by training without ground-truth supervision. We outperform baselines and prior work by a significant margin on both standard video summarization datasets (TVSum and SumMe) and a query-focused video summarization dataset (QFVS). Particularly, we achieve large improvements in the transfer setting, attesting to our method's strong generalization capabilities.

----

## [1072] Learning Treatment Effects in Panels with General Intervention Patterns

**Authors**: *Vivek F. Farias, Andrew A. Li, Tianyi Peng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7504adad8bb96320eb3afdd4df6e1f60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7504adad8bb96320eb3afdd4df6e1f60-Abstract.html)

**Abstract**:

The problem of causal inference with panel data is a central econometric question. The following is a fundamental version of this problem: Let $M^*$ be a low rank matrix and $E$ be a zero-mean noise matrix. For a `treatment' matrix $Z$ with entries in $\{0,1\}$ we observe the matrix $O$ with entries $O_{ij} := M^*_{ij} + E_{ij} + \mathcal{T}_{ij} Z_{ij}$ where $\mathcal{T}_{ij} $ are unknown, heterogenous treatment effects. The problem requires we estimate the average treatment effect $\tau^* := \sum_{ij} \mathcal{T}_{ij} Z_{ij} / \sum_{ij} Z_{ij}$. The synthetic control paradigm provides an approach to estimating $\tau^*$ when $Z$ places support on a single row. This paper extends that framework to allow rate-optimal recovery of $\tau^*$ for general $Z$, thus broadly expanding its applicability. Our guarantees are the first of their type in this general setting. Computational experiments on synthetic and real-world data show a substantial advantage over competing estimators.

----

## [1073] Lossy Compression for Lossless Prediction

**Authors**: *Yann Dubois, Benjamin Bloem-Reddy, Karen Ullrich, Chris J. Maddison*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7535bbb91c8fde347ad861f293126633-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7535bbb91c8fde347ad861f293126633-Abstract.html)

**Abstract**:

Most data is automatically collected and only ever "seen" by algorithms. Yet, data compressors preserve perceptual fidelity rather than just the information needed by algorithms performing downstream tasks. In this paper, we characterize the bit-rate required to ensure high performance on all predictive tasks that are invariant under a set of transformations, such as data augmentations. Based on our theory, we design unsupervised objectives for training neural compressors. Using these objectives, we train a generic image compressor that achieves substantial rate savings (more than 1000x on ImageNet) compared to JPEG on 8 datasets, without decreasing downstream classification performance.

----

## [1074] From Optimality to Robustness: Adaptive Re-Sampling Strategies in Stochastic Bandits

**Authors**: *Dorian Baudry, Patrick Saux, Odalric-Ambrym Maillard*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/75429d136f65d2d6168b9b6c5f6ec951-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/75429d136f65d2d6168b9b6c5f6ec951-Abstract.html)

**Abstract**:

The stochastic multi-arm bandit problem has been extensively studied under standard assumptions on the arm's distribution (e.g bounded with known support, exponential family, etc). These assumptions are suitable for many real-world problems but sometimes they require knowledge (on tails for instance) that may not be precisely accessible to the practitioner, raising the question of the robustness of bandit algorithms to model misspecification. In this paper we study a generic \emph{Dirichlet Sampling} (DS) algorithm, based on pairwise comparisons of empirical indices computed with \textit{re-sampling} of the arms' observations and a data-dependent \textit{exploration bonus}. We show that different variants of this strategy achieve provably optimal regret guarantees when the distributions are bounded and logarithmic regret for semi-bounded distributions with a mild quantile condition. We also show that a  simple tuning achieve robustness with respect to a large class of unbounded distributions, at the cost of slightly worse than logarithmic asymptotic regret. We finally provide numerical experiments showing the merits of DS in a decision-making problem on synthetic agriculture data.

----

## [1075] CCVS: Context-aware Controllable Video Synthesis

**Authors**: *Guillaume Le Moing, Jean Ponce, Cordelia Schmid*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/757b505cfd34c64c85ca5b5690ee5293-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/757b505cfd34c64c85ca5b5690ee5293-Abstract.html)

**Abstract**:

This presentation introduces a self-supervised learning approach to the synthesis of new videos clips from old ones, with several new key elements for improved spatial resolution and realism: It conditions the synthesis process on contextual information for temporal continuity and ancillary information for fine control. The prediction model is doubly autoregressive, in the latent space of an autoencoder for forecasting, and in image space for updating contextual information, which is also used to enforce spatio-temporal consistency through a learnable optical flow module. Adversarial training of the autoencoder in the appearance and temporal domains is used to further improve the realism of its output. A quantizer inserted between the encoder and the transformer in charge of forecasting future frames in latent space (and its inverse inserted between the transformer and the decoder) adds even more flexibility by affording simple mechanisms for handling multimodal ancillary information for controlling the synthesis process (e.g., a few sample frames, an audio track, a trajectory in image space) and taking into account the intrinsically uncertain nature of the future by allowing multiple predictions. Experiments with an implementation of the proposed approach give very good qualitative and quantitative results on multiple tasks and standard benchmarks.

----

## [1076] An Online Riemannian PCA for Stochastic Canonical Correlation Analysis

**Authors**: *Zihang Meng, Rudrasis Chakraborty, Vikas Singh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)

**Abstract**:

We present an efficient stochastic algorithm (RSG+) for canonical correlation analysis (CCA) using a reparametrization of the projection matrices. We show how this reparametrization (into structured matrices), simple in hindsight, directly presents an opportunity to repurpose/adjust mature techniques for numerical optimization on Riemannian manifolds. Our developments nicely complement existing methods for this problem which either require $O(d^3)$ time complexity per iteration with $O(\frac{1}{\sqrt{t}})$ convergence rate (where $d$ is the dimensionality) or only extract the top $1$ component with $O(\frac{1}{t})$ convergence rate. In contrast, our algorithm offers a strict improvement for this classical problem: it achieves $O(d^2k)$ runtime complexity per iteration for extracting the top $k$ canonical components with $O(\frac{1}{t})$ convergence rate. While the paper primarily focuses on the formulation and technical analysis of its properties, our experiments show that the empirical behavior on common datasets is quite promising, We also explore a potential application in training fair models where the label of protected attribute is missing or otherwise unavailable.

----

## [1077] Predify: Augmenting deep neural networks with brain-inspired predictive coding dynamics

**Authors**: *Bhavin Choksi, Milad Mozafari, Callum Biggs O'May, Benjamin Ador, Andrea Alamia, Rufin VanRullen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/75c58d36157505a600e0695ed0b3a22d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/75c58d36157505a600e0695ed0b3a22d-Abstract.html)

**Abstract**:

Deep neural networks excel at image classification, but their performance is far less robust to input perturbations than human perception. In this work we explore whether this shortcoming may be partly addressed by incorporating brain-inspired recurrent dynamics in deep convolutional networks. We take inspiration from a popular framework in neuroscience: "predictive coding". At each layer of the hierarchical model, generative feedback "predicts" (i.e., reconstructs) the pattern of activity in the previous layer. The reconstruction errors are used to iteratively update the networkâ€™s representations across timesteps, and to optimize the network's feedback weights over the natural image dataset--a form of unsupervised training. We show that implementing this strategy into two popular networks, VGG16 and EfficientNetB0, improves their robustness against various corruptions and adversarial attacks. We hypothesize that other feedforward networks could similarly benefit from the proposed framework. To promote research in this direction, we provide an open-sourced PyTorch-based package called \textit{Predify}, which can be used to implement and investigate the impacts of the predictive coding dynamics in any convolutional neural network.

----

## [1078] Deep Extrapolation for Attribute-Enhanced Generation

**Authors**: *Alvin Chan, Ali Madani, Ben Krause, Nikhil Naik*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/75da5036f659fe64b53f3d9b39412967-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/75da5036f659fe64b53f3d9b39412967-Abstract.html)

**Abstract**:

Attribute extrapolation in sample generation is challenging for deep neural networks operating beyond the training distribution. We formulate a new task for extrapolation in sequence generation, focusing on natural language and proteins, and propose GENhance, a generative framework that enhances attributes through a learned latent space. Trained on movie reviews and a computed protein stability dataset, GENhance can generate strongly-positive text reviews and highly stable protein sequences without being exposed to similar data during training. We release our benchmark tasks and models to contribute to the study of generative modeling extrapolation and data-driven design in biology and chemistry.

----

## [1079] Generalized DataWeighting via Class-Level Gradient Manipulation

**Authors**: *Can Chen, Shuhao Zheng, Xi Chen, Erqun Dong, Xue (Steve) Liu, Hao Liu, Dejing Dou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/75ebb02f92fc30a8040bbd625af999f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/75ebb02f92fc30a8040bbd625af999f1-Abstract.html)

**Abstract**:

Label noise and class imbalance are two major issues coexisting in real-world datasets. To alleviate the two issues, state-of-the-art methods reweight each instance by leveraging a small amount of clean and unbiased data. Yet, these methods overlook class-level information within each instance, which can be further utilized to improve performance. To this end, in this paper, we propose Generalized Data Weighting (GDW) to simultaneously mitigate label noise and class imbalance by manipulating gradients at the class level. To be specific, GDW unrolls the loss gradient to class-level gradients by the chain rule and reweights the flow of each gradient separately. In this way, GDW achieves remarkable performance improvement on both issues. Aside from the performance gain, GDW efficiently obtains class-level weights without introducing any extra computational cost compared with instance weighting methods. Specifically, GDW performs a gradient descent step on class-level weights, which only relies on intermediate gradients. Extensive experiments in various settings verify the effectiveness of GDW. For example, GDW outperforms state-of-the-art methods by $2.56\%$ under the $60\%$ uniform noise setting in CIFAR10. Our code is available at https://github.com/GGchen1997/GDW-NIPS2021.

----

## [1080] Slow Learning and Fast Inference: Efficient Graph Similarity Computation via Knowledge Distillation

**Authors**: *Can Qin, Handong Zhao, Lichen Wang, Huan Wang, Yulun Zhang, Yun Fu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/75fc093c0ee742f6dddaa13fff98f104-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/75fc093c0ee742f6dddaa13fff98f104-Abstract.html)

**Abstract**:

Graph Similarity Computation (GSC) is essential to wide-ranging graph applications such as retrieval, plagiarism/anomaly detection, etc. The exact computation of graph similarity, e.g., Graph Edit Distance (GED), is an NP-hard problem that cannot be exactly solved within an adequate time given large graphs. Thanks to the strong representation power of graph neural network (GNN), a variety of GNN-based inexact methods emerged. To capture the subtle difference across graphs, the key success is designing the dense interaction with features fusion at the early stage, which, however, is a trade-off between speed and accuracy. For slow learning of graph similarity, this paper proposes a novel early-fusion approach by designing a co-attention-based feature fusion network on multilevel GNN features. To further improve the speed without much accuracy drop, we introduce an efficient GSC solution by distilling the knowledge from the slow early-fusion model to the student one for fast inference. Such a student model also enables the offline collection of individual graph embeddings, speeding up the inference time in orders. To address the instability through knowledge transfer, we decompose the dynamic joint embedding into the static pseudo individual ones for precise teacher-student alignment. The experimental analysis on the real-world datasets demonstrates the superiority of our approach over the state-of-the-art methods on both accuracy and efficiency. Particularly, we speed up the prior art by more than 10x on the benchmark AIDS data.

----

## [1081] Meta Learning Backpropagation And Improving It

**Authors**: *Louis Kirsch, Jürgen Schmidhuber*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7608de7a475c0c878f60960d72a92654-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7608de7a475c0c878f60960d72a92654-Abstract.html)

**Abstract**:

Many concepts have been proposed for meta learning with neural networks (NNs), e.g., NNs that learn to reprogram fast weights, Hebbian plasticity, learned learning rules, and meta recurrent NNs. Our Variable Shared Meta Learning (VSML) unifies the above and demonstrates that simple weight-sharing and sparsity in an NN is sufficient to express powerful learning algorithms (LAs) in a reusable fashion. A simple implementation of VSML where the weights of a neural network are replaced by tiny LSTMs allows for implementing the backpropagation LA solely by running in forward-mode. It can even meta learn new LAs that differ from online backpropagation and generalize to datasets outside of the meta training distribution without explicit gradient calculation. Introspection reveals that our meta learned LAs learn through fast association in a way that is qualitatively different from gradient descent.

----

## [1082] Posterior Meta-Replay for Continual Learning

**Authors**: *Christian Henning, Maria R. Cervera, Francesco D'Angelo, Johannes von Oswald, Regina Traber, Benjamin Ehret, Seijin Kobayashi, Benjamin F. Grewe, João Sacramento*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/761b42cfff120aac30045f7a110d0256-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/761b42cfff120aac30045f7a110d0256-Abstract.html)

**Abstract**:

Learning a sequence of tasks without access to i.i.d. observations is a widely studied form of continual learning (CL) that remains challenging. In principle, Bayesian learning directly applies to this setting, since recursive and one-off Bayesian updates yield the same result. In practice, however, recursive updating often leads to poor trade-off solutions across tasks because approximate inference is necessary for most models of interest. Here, we describe an alternative Bayesian approach where task-conditioned parameter distributions are continually inferred from data. We offer a practical deep learning implementation of our framework based on probabilistic task-conditioned hypernetworks, an approach we term posterior meta-replay. Experiments on standard benchmarks show that our probabilistic hypernetworks compress sequences of posterior parameter distributions with virtually no forgetting. We obtain considerable performance gains compared to existing Bayesian CL methods, and identify task inference as our major limiting factor. This limitation has several causes that are independent of the considered sequential setting, opening up new avenues for progress in CL.

----

## [1083] Optimizing Reusable Knowledge for Continual Learning via Metalearning

**Authors**: *Julio Hurtado, Alain Raymond-Saez, Alvaro Soto*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/761e6675f9e54673cc778e7fdb2823d2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/761e6675f9e54673cc778e7fdb2823d2-Abstract.html)

**Abstract**:

When learning tasks over time, artificial neural networks suffer from a problem known as Catastrophic Forgetting (CF). This happens when the weights of a network are overwritten during the training of a new task causing forgetting of old information. To address this issue, we propose MetA Reusable Knowledge or MARK, a new method that fosters weight reusability instead of overwriting when learning a new task. Specifically, MARK keeps a set of shared weights among tasks. We envision these shared weights as a common Knowledge Base (KB) that is not only used to learn new tasks, but also enriched with new knowledge as the model learns new tasks. Key components behind MARK are two-fold. On the one hand, a metalearning approach provides the key mechanism to incrementally enrich the KB with new knowledge and to foster weight reusability among tasks. On the other hand, a set of trainable masks provides the key mechanism to selectively choose from the KB relevant weights to solve each task. By using MARK, we achieve state of the art results in several popular benchmarks, surpassing the best performing methods in terms of average accuracy by over 10% on the 20-Split-MiniImageNet dataset, while achieving almost zero forgetfulness using 55% of the number of parameters. Furthermore, an ablation study provides evidence that, indeed, MARK is learning reusable knowledge that is selectively used by each task.

----

## [1084] A sampling-based circuit for optimal decision making

**Authors**: *Camille E. Rullán Buxó, Cristina Savin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/76444b3132fda0e2aca778051d776f1c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/76444b3132fda0e2aca778051d776f1c-Abstract.html)

**Abstract**:

Many features of human and animal behavior can be understood in the framework of Bayesian inference and optimal decision making, but the biological substrate of such processes is not fully understood. Neural sampling provides a flexible code for probabilistic inference in high dimensions and explains key features of sensory responses under experimental manipulations of uncertainty. However, since it encodes uncertainty implicitly, across time and neurons, it remains unclear how such representations can be used for decision making. Here we propose a spiking network model that maps neural samples of a task-specific marginal distribution into an instantaneous representation of uncertainty via a procedure inspired by online  kernel density estimation, so that its output can be readily used for decision making. Our model is consistent with experimental results at the level of single neurons and populations, and makes predictions for how neural responses and decisions could be modulated by uncertainty and prior biases. More generally, our work brings together conflicting perspectives on probabilistic brain computation.

----

## [1085] Compressed Video Contrastive Learning

**Authors**: *Yuqi Huo, Mingyu Ding, Haoyu Lu, Nanyi Fei, Zhiwu Lu, Ji-Rong Wen, Ping Luo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7647966b7343c29048673252e490f736-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7647966b7343c29048673252e490f736-Abstract.html)

**Abstract**:

This work concerns self-supervised video representation learning (SSVRL), one topic that has received much attention recently. Since videos are storage-intensive and contain a rich source of visual content, models designed for SSVRL are expected to be storage- and computation-efficient, as well as effective. However, most existing methods only focus on one of the two objectives, failing to consider both at the same time. In this work, for the first time, the seemingly contradictory goals are simultaneously achieved by exploiting compressed videos and capturing mutual information between two input streams. Specifically, a novel Motion Vector based Cross Guidance Contrastive learning approach (MVCGC) is proposed. For storage and computation efficiency, we choose to directly decode RGB frames and motion vectors (that resemble low-resolution optical flows) from compressed videos on-the-fly. To enhance the representation ability of the motion vectors, hence the effectiveness of our method, we design a cross guidance contrastive learning algorithm based on multi-instance InfoNCE loss, where motion vectors can take supervision signals from RGB frames and vice versa. Comprehensive experiments on two downstream tasks show that our MVCGC yields new state-of-the-art while being significantly more efficient than its competitors.

----

## [1086] Uniform-PAC Bounds for Reinforcement Learning with Linear Function Approximation

**Authors**: *Jiafan He, Dongruo Zhou, Quanquan Gu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7695ea769f021803c508817dd374bb27-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7695ea769f021803c508817dd374bb27-Abstract.html)

**Abstract**:

We study reinforcement learning (RL) with linear function approximation. Existing algorithms for this problem only have high-probability regret and/or Probably Approximately Correct (PAC) sample complexity guarantees, which cannot guarantee the convergence to the optimal policy. In this paper, in order to overcome the limitation of existing algorithms, we propose a new algorithm called FLUTE, which enjoys uniform-PAC convergence to the optimal policy with high probability. The uniform-PAC guarantee is the strongest possible guarantee for reinforcement learning in the literature, which can directly imply both PAC and high probability regret bounds, making our algorithm superior to all existing algorithms with linear function approximation. At the core of our algorithm is a novel minimax value function estimator and a multi-level partition scheme to select the training samples from historical observations. Both of these techniques are new and of independent interest.

----

## [1087] Attention Bottlenecks for Multimodal Fusion

**Authors**: *Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/76ba9f564ebbc35b1014ac498fafadd0-Abstract.html)

**Abstract**:

Humans perceive the world by concurrently processing and fusing high-dimensional inputs from multiple modalities such as vision and audio.  Machine perception models, in stark contrast, are typically modality-specific and optimised for unimodal benchmarks.A common approach for building multimodal models is to simply combine multiple of these modality-specific architectures using late-stage fusion of final representations or predictions ('late-fusion').Instead, we introduce a novel transformer based architecture that uses 'attention bottlenecks' for modality fusion at multiple layers. Compared to traditional pairwise self-attention,  these bottlenecks force information between different modalities to pass through a small number of '`bottleneck' latent units, requiring the model to collate and condense the most relevant information in each modality and only share what is necessary. We find that such a strategy improves fusion performance, at the same time reducing computational cost. We conduct thorough ablation studies, and achieve state-of-the-art results on multiple audio-visual classification benchmarks including Audioset, Epic-Kitchens and VGGSound. All code and models will be released.

----

## [1088] Convergence of adaptive algorithms for constrained weakly convex optimization

**Authors**: *Ahmet Alacaoglu, Yura Malitsky, Volkan Cevher*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/76c073d8a82d9ddaf993300be03ac70f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/76c073d8a82d9ddaf993300be03ac70f-Abstract.html)

**Abstract**:

We analyze the adaptive first order algorithm AMSGrad, for solving a constrained stochastic optimization problem with a weakly convex objective. We prove the $\mathcal{\tilde O}(t^{-1/2})$ rate of convergence for the squared norm of the gradient of Moreau envelope, which is the standard stationarity measure for this class of problems. It matches the known rates that adaptive algorithms enjoy for the specific case of unconstrained smooth nonconvex stochastic optimization. Our analysis works with mini-batch size of $1$, constant first and second order moment parameters, and possibly unbounded optimization domains. Finally, we illustrate the applications and extensions of our results to specific problems and algorithms.

----

## [1089] On the Convergence of Step Decay Step-Size for Stochastic Optimization

**Authors**: *Xiaoyu Wang, Sindri Magnússon, Mikael Johansson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/76c538125fc5c9ec6ad1d05650a57de5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/76c538125fc5c9ec6ad1d05650a57de5-Abstract.html)

**Abstract**:

The convergence of stochastic gradient descent is highly dependent on the step-size, especially on non-convex problems such as neural network training. Step decay step-size schedules (constant and then cut) are widely used in practice because of their excellent convergence and generalization qualities, but their theoretical properties are not yet well understood. We provide convergence results for step decay in the non-convex regime, ensuring that the gradient norm vanishes at an $\mathcal{O}(\ln T/\sqrt{T})$ rate. We also provide near-optimal (and sometimes provably tight) convergence guarantees for general, possibly non-smooth, convex and strongly convex problems. The practical efficiency of the step decay step-size is demonstrated in several large-scale deep neural network training tasks.

----

## [1090] BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation

**Authors**: *Mingguo He, Zhewei Wei, Zengfeng Huang, Hongteng Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/76f1cfd7754a6e4fc3281bcccb3d0902-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/76f1cfd7754a6e4fc3281bcccb3d0902-Abstract.html)

**Abstract**:

Many representative graph neural networks, $e.g.$, GPR-GNN and ChebNet, approximate graph convolutions with graph spectral filters. However, existing work either applies predefined filter weights or learns them without necessary constraints, which may lead to oversimplified or ill-posed filters. To overcome these issues, we propose $\textit{BernNet}$, a novel graph neural network with theoretical support that provides a simple but effective scheme for designing and learning arbitrary graph spectral filters. In particular, for any filter over the normalized Laplacian spectrum of a graph, our BernNet estimates it by an order-$K$ Bernstein polynomial approximation and designs its spectral property by setting the coefficients of the Bernstein basis. Moreover, we can learn the coefficients (and the corresponding filter weights) based on observed graphs and their associated signals and thus achieve the BernNet specialized for the data. Our experiments demonstrate that BernNet can learn arbitrary spectral filters, including complicated band-rejection and comb filters, and it achieves superior performance in real-world graph modeling tasks. Code is available at https://github.com/ivam-he/BernNet.

----

## [1091] Co-evolution Transformer for Protein Contact Prediction

**Authors**: *He Zhang, Fusong Ju, Jianwei Zhu, Liang He, Bin Shao, Nanning Zheng, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/770f8e448d07586afbf77bb59f698587-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/770f8e448d07586afbf77bb59f698587-Abstract.html)

**Abstract**:

Proteins are the main machinery of life and protein functions are largely determined by their 3D structures. The measurement of the pairwise proximity between amino acids of a protein, known as inter-residue contact map, well characterizes the structural information of a protein. Protein contact prediction (PCP) is an essential building block of many protein structure related applications.  The prevalent approach to contact prediction is based on estimating the inter-residue contacts using hand-crafted coevolutionary features derived from multiple sequence alignments (MSAs). To mitigate the information loss caused by hand-crafted features, some recently proposed methods try to learn residue co-evolutions directly from MSAs. These methods generally derive coevolutionary features by aggregating the learned residue representations from individual sequences with equal weights, which is inconsistent with the premise that residue co-evolutions are a reflection of collective covariation patterns of numerous homologous proteins.  Moreover, non-homologous residues and gaps commonly exist in MSAs. By aggregating features from all homologs equally, the non-homologous information may cause misestimation of the residue co-evolutions.  To overcome these issues, we propose an attention-based architecture, Co-evolution Transformer (CoT), for PCP. CoT jointly considers the information from all homologous sequences in the MSA to better capture global coevolutionary patterns. To mitigate the influence of the non-homologous information, CoT selectively aggregates the features from different homologs by assigning smaller weights to non-homologous sequences or residue pairs.  Extensive experiments on two rigorous benchmark datasets demonstrate the effectiveness of CoT. In particular, CoT achieves a $51.6\%$ top-L long-range precision score for the Free Modeling (FM) domains on the CASP14 benchmark, which outperforms the winner group of CASP14 contact prediction challenge by $9.8\%$.

----

## [1092] Unsupervised Foreground Extraction via Deep Region Competition

**Authors**: *Peiyu Yu, Sirui Xie, Xiaojian Ma, Yixin Zhu, Ying Nian Wu, Song-Chun Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/77369e37b2aa1404f416275183ab055f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/77369e37b2aa1404f416275183ab055f-Abstract.html)

**Abstract**:

We present Deep Region Competition (DRC), an algorithm designed to extract foreground objects from images in a fully unsupervised manner. Foreground extraction can be viewed as a special case of generic image segmentation that focuses on identifying and disentangling objects from the background. In this work, we rethink the foreground extraction by reconciling energy-based prior with generative image modeling in the form of Mixture of Experts (MoE), where we further introduce the learned pixel re-assignment as the essential inductive bias to capture the regularities of background regions. With this modeling, the foreground-background partition can be naturally found through Expectation-Maximization (EM). We show that the proposed method effectively exploits the interaction between the mixture components during the partitioning process, which closely connects to region competition, a seminal approach for generic image segmentation. Experiments demonstrate that DRC exhibits more competitive performances on complex real-world data and challenging multi-object scenes compared with prior methods. Moreover, we show empirically that DRC can potentially generalize to novel foreground objects even from categories unseen during training.

----

## [1093] Leveraging Spatial and Temporal Correlations in Sparsified Mean Estimation

**Authors**: *Divyansh Jhunjhunwala, Ankur Mallick, Advait Gadhikar, Swanand Kadhe, Gauri Joshi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/77b88288ebae7b17b7c8610a48c40dd1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/77b88288ebae7b17b7c8610a48c40dd1-Abstract.html)

**Abstract**:

We study the problem of estimating at a central server the mean of a set of vectors distributed across several nodes (one vector per node). When the vectors are high-dimensional, the communication cost of sending entire vectors may be prohibitive, and it may be imperative for them to use sparsification techniques. While most existing work on sparsified mean estimation is agnostic to the characteristics of the data vectors, in many practical applications such as federated learning, there may be spatial correlations (similarities in the vectors sent by different nodes) or temporal correlations (similarities in the data sent by a single node over different iterations of the algorithm) in the data vectors. We leverage these correlations by simply modifying the decoding method used by the server to estimate the mean. We provide an analysis of the resulting estimation error as well as experiments for PCA, K-Means and Logistic Regression, which show that our estimators consistently outperform more sophisticated and expensive sparsification methods.

----

## [1094] Last-iterate Convergence in Extensive-Form Games

**Authors**: *Chung-Wei Lee, Christian Kroer, Haipeng Luo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/77bb14f6132ea06dea456584b7d5581e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/77bb14f6132ea06dea456584b7d5581e-Abstract.html)

**Abstract**:

Regret-based algorithms are highly efficient at finding approximate Nash equilibria in sequential games such as poker games. However, most regret-based algorithms, including counterfactual regret minimization (CFR) and its variants, rely on iterate averaging to achieve convergence.  Inspired by recent advances on last-iterate convergence of optimistic algorithms in zero-sum normal-form games, we study this phenomenon in sequential games, and provide a comprehensive study of last-iterate convergence for zero-sum extensive-form games with perfect recall (EFGs), using various optimistic regret-minimization algorithms over treeplexes. This includes algorithms using the vanilla entropy or squared Euclidean norm regularizers, as well as their dilated versions which admit more efficient implementation. In contrast to CFR, we show that all of these algorithms enjoy last-iterate convergence, with some of them even converging exponentially fast. We also provide experiments to further support our theoretical results.

----

## [1095] Class-Incremental Learning via Dual Augmentation

**Authors**: *Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/77ee3bc58ce560b86c2b59363281e914-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/77ee3bc58ce560b86c2b59363281e914-Abstract.html)

**Abstract**:

Deep learning systems typically suffer from catastrophic forgetting of past knowledge when acquiring new skills continually. In this paper, we emphasize two dilemmas, representation bias and classifier bias in class-incremental learning, and present a simple and novel approach that employs explicit class augmentation (classAug) and implicit semantic augmentation (semanAug) to address the two biases, respectively. On the one hand, we propose to address the representation bias by learning transferable and diverse representations. Specifically, we investigate the feature representations in incremental learning based on spectral analysis and present a simple technique called classAug, to let the model see more classes during training for learning representations transferable across classes. On the other hand, to overcome the classifier bias, semanAug implicitly involves the simultaneous generating of an infinite number of instances of old classes in the deep feature space, which poses tighter constraints to maintain the decision boundary of previously learned classes. Without storing any old samples, our method can perform comparably with representative data replay based approaches.

----

## [1096] Robust and Fully-Dynamic Coreset for Continuous-and-Bounded Learning (With Outliers) Problems

**Authors**: *Zixiu Wang, Yiwen Guo, Hu Ding*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7806689d934e610d660caf5536fea0b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7806689d934e610d660caf5536fea0b2-Abstract.html)

**Abstract**:

In many machine learning tasks, a common approach for dealing with large-scale data is to build a small summary, {\em e.g.,} coreset,  that can efficiently represent the original input. However, real-world datasets usually contain outliers and most existing coreset construction methods are not resilient against outliers (in particular, an outlier can be located arbitrarily in the space by an adversarial attacker). In this paper, we propose a novel robust coreset method for the {\em continuous-and-bounded learning} problems (with outliers) which includes a broad range of popular optimization objectives in machine learning, {\em e.g.,} logistic regression and $ k $-means clustering. Moreover, our robust coreset  can be efficiently maintained in fully-dynamic environment. To the best of our knowledge, this is the first robust and fully-dynamic coreset construction method for these optimization problems. Another highlight is that our coreset size can depend on the doubling dimension of the parameter space, rather than the VC dimension of the objective function which could be very large or even challenging to compute. Finally, we conduct the experiments on real-world datasets to evaluate the effectiveness of our proposed robust coreset method.

----

## [1097] Rethinking and Reweighting the Univariate Losses for Multi-Label Ranking: Consistency and Generalization

**Authors**: *Guoqiang Wu, Chongxuan Li, Kun Xu, Jun Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/781397bc0630d47ab531ea850bddcf63-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/781397bc0630d47ab531ea850bddcf63-Abstract.html)

**Abstract**:

The (partial) ranking loss is a commonly used evaluation measure for multi-label classification, which is usually optimized with convex surrogates for computational efficiency. Prior theoretical efforts on multi-label ranking mainly focus on (Fisher) consistency analyses. However, there is a gap between existing theory and practice --- some inconsistent pairwise losses can lead to promising performance, while some consistent univariate losses usually have no clear superiority in practice. To take a step towards filling up this gap, this paper presents a systematic study from two complementary perspectives of consistency and generalization error bounds of learning algorithms. We theoretically find two key factors of the distribution (or dataset) that affect the learning guarantees of algorithms: the instance-wise class imbalance and the label size $c$. Specifically, in an extremely imbalanced case, the algorithm with the consistent univariate loss has an error bound of $O(c)$, while the one with the inconsistent pairwise loss depends on $O(\sqrt{c})$ as shown in prior work. This may shed light on the superior performance of pairwise methods in practice, where real datasets are usually highly imbalanced. Moreover, we present an inconsistent reweighted univariate loss-based algorithm that enjoys an error bound of $O(\sqrt{c})$ for promising performance as well as the computational efficiency of univariate losses. Finally, experimental results confirm our theoretical findings.

----

## [1098] Fair Clustering Under a Bounded Cost

**Authors**: *Seyed A. Esmaeili, Brian Brubach, Aravind Srinivasan, John Dickerson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/781877bda0783aac5f1cf765c128b437-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/781877bda0783aac5f1cf765c128b437-Abstract.html)

**Abstract**:

Clustering is a fundamental unsupervised learning problem where a dataset is partitioned into clusters that consist of nearby points in a metric space. A recent variant, fair clustering, associates a color with each point representing its group membership and requires that each color has (approximately) equal representation in each cluster to satisfy group fairness. In this model, the cost of the clustering objective increases due to enforcing fairness in the algorithm. The relative increase in the cost, the ```````''price of fairness,'' can indeed be unbounded. Therefore, in this paper we propose to treat an upper bound on the clustering objective as a constraint on the clustering problem, and to maximize equality of representation subject to it. We consider two fairness objectives: the group utilitarian objective and the group egalitarian objective, as well as the group leximin objective which generalizes the group egalitarian objective. We derive fundamental lower bounds on the approximation of the utilitarian and egalitarian objectives and introduce algorithms with provable guarantees for them. For the leximin objective we introduce an effective heuristic algorithm. We further derive impossibility results for other natural fairness objectives. We conclude with experimental results on real-world datasets that demonstrate the validity of our algorithms.

----

## [1099] Improving Calibration through the Relationship with Adversarial Robustness

**Authors**: *Yao Qin, Xuezhi Wang, Alex Beutel, Ed H. Chi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/78421a2e0e1168e5cd1b7a8d23773ce6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/78421a2e0e1168e5cd1b7a8d23773ce6-Abstract.html)

**Abstract**:

Neural networks lack adversarial robustness, i.e., they are vulnerable to adversarial examples that through small perturbations to inputs cause incorrect predictions. Further, trust is undermined when models give miscalibrated predictions, i.e.,  the predicted probability is not a good indicator of how much we should trust our model. In this paper, we study the connection between adversarial robustness and calibration and find that the inputs for which the model is sensitive to small perturbations (are easily attacked) are more likely to have poorly calibrated predictions. Based on this insight, we examine if calibration can be improved by addressing those adversarially unrobust inputs. To this end, we propose Adversarial Robustness based Adaptive Label Smoothing (AR-AdaLS) that integrates the correlations of adversarial robustness and calibration into training by adaptively softening labels for an example based on how easily it can be attacked by an adversary. We find that our method, taking the adversarial robustness of the in-distribution data into consideration, leads to better calibration over the model even under distributional shifts. In addition, AR-AdaLS can also be applied to an ensemble model to further improve model calibration.

----

## [1100] Credal Self-Supervised Learning

**Authors**: *Julian Lienen, Eyke Hüllermeier*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7866c91c59f8bffc92a79a7cd09f9af9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7866c91c59f8bffc92a79a7cd09f9af9-Abstract.html)

**Abstract**:

Self-training is an effective approach to semi-supervised learning. The key idea is to let the learner itself iteratively generate "pseudo-supervision" for unlabeled instances based on its current hypothesis. In combination with consistency regularization, pseudo-labeling has shown promising performance in various domains, for example in computer vision. To account for the hypothetical nature of the pseudo-labels, these are commonly provided in the form of probability distributions. Still, one may argue that even a probability distribution represents an excessive level of informedness, as it suggests that the learner precisely knows the ground-truth conditional probabilities. In our approach, we therefore allow the learner to label instances in the form of credal sets, that is, sets of (candidate) probability distributions. Thanks to this increased expressiveness, the learner is able to represent uncertainty and a lack of knowledge in a more flexible and more faithful manner. To learn from weakly labeled data of that kind, we leverage methods that have recently been proposed in the realm of so-called superset learning. In an exhaustive empirical evaluation, we compare our methodology to state-of-the-art self-supervision approaches, showing competitive to superior performance especially in low-label scenarios incorporating a high degree of uncertainty.

----

## [1101] Spot the Difference: Detection of Topological Changes via Geometric Alignment

**Authors**: *Steffen Czolbe, Aasa Feragen, Oswin Krause*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7867d6557b82ed3b5d61e6591a2a2fd3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7867d6557b82ed3b5d61e6591a2a2fd3-Abstract.html)

**Abstract**:

Geometric alignment appears in a variety of applications, ranging from domain adaptation, optimal transport, and normalizing flows in machine learning; optical flow and learned augmentation in computer vision and deformable registration within biomedical imaging. A recurring challenge is the alignment of domains whose topology is not the same; a problem that is routinely ignored, potentially introducing bias in downstream analysis. As a first step towards solving such alignment problems, we propose an unsupervised algorithm for the detection of changes in image topology. The model is based on a conditional variational auto-encoder and detects topological changes between two images during the registration step. We account for both topological changes in the image under spatial variation and unexpected transformations. Our approach is validated on two tasks and datasets: detection of topological changes in microscopy images of cells, and unsupervised anomaly detection brain imaging.

----

## [1102] Rethinking the Variational Interpretation of Accelerated Optimization Methods

**Authors**: *Peiyuan Zhang, Antonio Orvieto, Hadi Daneshmand*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/788d986905533aba051261497ecffcbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/788d986905533aba051261497ecffcbb-Abstract.html)

**Abstract**:

The continuous-time model of Nesterov's momentum provides a thought-provoking perspective for understanding the nature of the acceleration phenomenon in convex optimization. One of the main ideas in this line of research comes from the field of classical mechanics and proposes to link Nesterov's trajectory to the solution of a set of Euler-Lagrange equations relative to the so-called Bregman Lagrangian. In the last years, this approach led to the discovery of many new (stochastic) accelerated algorithms and provided a solid theoretical foundation for the design of structure-preserving accelerated methods. In this work, we revisit this idea and provide an in-depth analysis of the action relative to the Bregman Lagrangian from the point of view of calculus of variations. Our main finding is that, while Nesterov's method is a stationary point for the action, it is often not a minimizer but instead a saddle point for this functional in the space of differentiable curves. This finding challenges the main intuition behind the variational interpretation of Nesterov's method and provides additional insights into the intriguing geometry of accelerated paths.

----

## [1103] Linear and Kernel Classification in the Streaming Model: Improved Bounds for Heavy Hitters

**Authors**: *Arvind V. Mahankali, David P. Woodruff*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/78ccad7da4c2fc2646d1848e965794c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/78ccad7da4c2fc2646d1848e965794c5-Abstract.html)

**Abstract**:

We study linear and kernel classification in the streaming model. For linear classification, we improve upon the algorithm of (Tai, et al. 2018), which solves the $\ell_1$ point query problem on the optimal weight vector $w_* \in \mathbb{R}^d$ in sublinear space. We first give an algorithm solving the more difficult $\ell_2$ point query problem on $w_*$, also in sublinear space. We also give an algorithm which solves the $\ell_2$ heavy hitter problem on $w_*$, in sublinear space and running time. Finally, we give an algorithm which can $\textit{deterministically}$ solve the $\ell_1$ point query problem on $w_*$, with sublinear space improving upon that of (Tai, et al. 2018). For kernel classification, if $w_* \in \mathbb{R}^{d^p}$ is the optimal weight vector classifying points in the stream according to their $p^{th}$-degree polynomial kernel, then we give an algorithm solving the $\ell_2$ point query problem on $w_*$ in $\text{poly}(\frac{p \log d}{\varepsilon})$ space, and an algorithm solving the $\ell_2$ heavy hitter problem in $\text{poly}(\frac{p \log d}{\varepsilon})$ space and running time. Note that our space and running time are polynomial in $p$, making our algorithms well-suited to high-degree polynomial kernels and the Gaussian kernel (approximated by the polynomial kernel of degree $p = \Theta(\log T)$).

----

## [1104] A PAC-Bayes Analysis of Adversarial Robustness

**Authors**: *Paul Viallard, Guillaume Vidot, Amaury Habrard, Emilie Morvant*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/78e8dffe65a2898eef68a33b8db35b78-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/78e8dffe65a2898eef68a33b8db35b78-Abstract.html)

**Abstract**:

We propose the first general PAC-Bayesian generalization bounds for adversarial robustness, that estimate, at test time, how much a model will be invariant to imperceptible perturbations in the input. Instead of deriving a worst-case analysis of the risk of a hypothesis over all the possible perturbations, we leverage the PAC-Bayesian framework to bound the averaged risk on the perturbations for majority votes (over the whole class of hypotheses). Our theoretically founded analysis has the advantage to provide general bounds (i) that are valid for any kind of attacks (i.e., the adversarial attacks), (ii) that are tight thanks to the PAC-Bayesian framework, (iii) that can be directly minimized during the learning phase to obtain a robust model on different attacks at test time.

----

## [1105] SE(3)-equivariant prediction of molecular wavefunctions and electronic densities

**Authors**: *Oliver T. Unke, Mihail Bogojeski, Michael Gastegger, Mario Geiger, Tess E. Smidt, Klaus-Robert Müller*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/78f1893678afbeaa90b1fa01b9cfb860-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/78f1893678afbeaa90b1fa01b9cfb860-Abstract.html)

**Abstract**:

Machine learning has enabled the prediction of quantum chemical properties with high accuracy and efficiency, allowing to bypass computationally costly ab initio calculations. Instead of training on a fixed set of properties, more recent approaches attempt to learn the electronic wavefunction (or density) as a central quantity of atomistic systems, from which all other observables can be derived. This is complicated by the fact that wavefunctions transform non-trivially under molecular rotations, which makes them a challenging prediction target. To solve this issue, we introduce general SE(3)-equivariant operations and building blocks for constructing deep learning architectures for geometric point cloud data and apply them to reconstruct wavefunctions of atomistic systems with unprecedented accuracy. Our model achieves speedups of over three orders of magnitude compared to ab initio methods and reduces prediction errors by up to two orders of magnitude compared to the previous state-of-the-art. This accuracy makes it possible to derive properties such as energies and forces directly from the wavefunction in an end-to-end manner. We demonstrate the potential of our approach in a transfer learning application, where a model trained on low accuracy reference wavefunctions implicitly learns to correct for electronic many-body interactions from observables computed at a higher level of theory. Such machine-learned wavefunction surrogates pave the way towards novel semi-empirical methods, offering resolution at an electronic level while drastically decreasing computational cost. Additionally, the predicted wavefunctions can serve as initial guess in conventional ab initio methods, decreasing the number of iterations required to arrive at a converged solution, thus leading to significant speedups without any loss of accuracy or robustness. While we focus on physics applications in this contribution, the proposed equivariant framework for deep learning on point clouds is promising also beyond, say, in computer vision or graphics.

----

## [1106] Modified Frank Wolfe in Probability Space

**Authors**: *Carson Kent, Jiajin Li, José H. Blanchet, Peter W. Glynn*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/79121bb953a3bd47c076f20234bafd2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/79121bb953a3bd47c076f20234bafd2e-Abstract.html)

**Abstract**:

We propose a novel Frank-Wolfe (FW) procedure for the optimization of infinite-dimensional functionals of probability measures - a task which arises naturally in a wide range of areas including statistical learning (e.g. variational inference) and artificial intelligence (e.g. generative adversarial networks). Our FW procedure takes advantage of Wasserstein gradient flows and strong duality results recently developed in Distributionally Robust Optimization so that gradient steps (in the Wasserstein space) can be efficiently computed using finite-dimensional, convex optimization methods. We show how to choose the step sizes in order to guarantee exponentially fast iteration convergence, under mild assumptions on the functional to optimize. We apply our algorithm to a range of functionals arising from applications in nonparametric estimation.

----

## [1107] Bayesian Optimization of Function Networks

**Authors**: *Raul Astudillo, Peter I. Frazier*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/792c7b5aae4a79e78aaeda80516ae2ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/792c7b5aae4a79e78aaeda80516ae2ac-Abstract.html)

**Abstract**:

We consider Bayesian optimization of the output of a network of functions, where each function takes as input the output of its parent nodes, and where the network takes significant time to evaluate. Such problems arise, for example, in reinforcement learning, engineering design, and manufacturing.  While the standard Bayesian optimization approach observes only the final output, our approach delivers greater query efficiency by leveraging information that the former ignores: intermediate output within the network. This is achieved by modeling the nodes of the network using Gaussian processes and choosing the points to evaluate using, as our acquisition function, the expected improvement computed with respect to the implied posterior on the objective. Although the non-Gaussian nature of this posterior prevents computing our acquisition function in closed form, we show that it can be efficiently maximized via sample average approximation. In addition, we prove that our method is asymptotically consistent, meaning that it finds a globally optimal solution as the number of evaluations grows to infinity, thus generalizing previously known convergence results for the expected improvement. Notably, this holds even though our method might not evaluate the domain densely, instead leveraging problem structure to leave regions unexplored. Finally, we show that our approach dramatically outperforms standard Bayesian optimization methods in several synthetic and real-world problems.

----

## [1108] Look at What I'm Doing: Self-Supervised Spatial Grounding of Narrations in Instructional Videos

**Authors**: *Reuben Tan, Bryan A. Plummer, Kate Saenko, Hailin Jin, Bryan Russell*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/792dd774336314c3c27a04bb260cf2cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/792dd774336314c3c27a04bb260cf2cf-Abstract.html)

**Abstract**:

We introduce the task of spatially localizing narrated interactions in videos. Key to our approach is the ability to learn to spatially localize interactions with self-supervision on a large corpus of videos with accompanying transcribed narrations. To achieve this goal, we propose a multilayer cross-modal attention network that enables effective optimization of a contrastive loss during training. We introduce a divided strategy that alternates between computing inter- and intra-modal attention across the visual and natural language modalities, which allows effective training via directly contrasting the two modalities' representations. We demonstrate the effectiveness of our approach by self-training on the HowTo100M instructional video dataset and evaluating on a newly collected dataset of localized described interactions in the YouCook2 dataset. We show that our approach outperforms alternative baselines, including shallow co-attention and full cross-modal attention. We also apply our approach to grounding phrases in images with weak supervision on Flickr30K and show that stacking multiple attention layers is effective and, when combined with a word-to-region loss, achieves state of the art on recall-at-one and pointing hand accuracies.

----

## [1109] RETRIEVE: Coreset Selection for Efficient and Robust Semi-Supervised Learning

**Authors**: *KrishnaTeja Killamsetty, Xujiang Zhao, Feng Chen, Rishabh K. Iyer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/793bc52a941b3951dfdb85fb04f9fd06-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/793bc52a941b3951dfdb85fb04f9fd06-Abstract.html)

**Abstract**:

Semi-supervised learning (SSL) algorithms have had great success in recent years in limited labeled data regimes. However, the current state-of-the-art SSL algorithms are computationally expensive and entail significant compute time and energy requirements. This can prove to be a huge limitation for many smaller companies and academic groups. Our main insight is that training on a subset of unlabeled data instead of entire unlabeled data enables the current SSL algorithms to converge faster, significantly reducing computational costs. In this work, we propose RETRIEVE, a coreset selection framework for efficient and robust semi-supervised learning. RETRIEVE selects the coreset by solving a mixed discrete-continuous bi-level optimization problem such that the selected coreset minimizes the labeled set loss. We use a one-step gradient approximation and show that the discrete optimization problem is approximately submodular, enabling simple greedy algorithms to obtain the coreset. We empirically demonstrate on several real-world datasets that existing SSL algorithms like VAT, Mean-Teacher, FixMatch, when used with RETRIEVE, achieve a) faster training times,  b) better performance when unlabeled data consists of Out-of-Distribution (OOD) data and imbalance. More specifically, we show that with minimal accuracy degradation, RETRIEVE achieves a speedup of around $3\times$ in the traditional SSL setting and achieves a speedup of $5\times$ compared to state-of-the-art (SOTA) robust SSL algorithms in the case of imbalance and OOD data. RETRIEVE is available as a part of the CORDS toolkit: https://github.com/decile-team/cords.

----

## [1110] Collaborating with Humans without Human Data

**Authors**: *DJ Strouse, Kevin R. McKee, Matt M. Botvinick, Edward Hughes, Richard Everett*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/797134c3e42371bb4979a462eb2f042a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/797134c3e42371bb4979a462eb2f042a-Abstract.html)

**Abstract**:

Collaborating with humans requires rapidly adapting to their individual strengths, weaknesses, and preferences. Unfortunately, most standard multi-agent reinforcement learning techniques, such as self-play (SP) or population play (PP), produce agents that overfit to their training partners and do not generalize well to humans. Alternatively, researchers can collect human data, train a human model using behavioral cloning, and then use that model to train "human-aware" agents ("behavioral cloning play", or BCP). While such an approach can improve the generalization of agents to new human co-players, it involves the onerous and expensive step of collecting large amounts of human data first. Here, we study the problem of how to train agents that collaborate well with human partners without using human data. We argue that the crux of the problem is to produce a diverse set of training partners. Drawing inspiration from successful multi-agent approaches in competitive domains, we find that a surprisingly simple approach is highly effective. We train our agent partner as the best response to a population of self-play agents and their past checkpoints taken throughout training, a method we call Fictitious Co-Play (FCP). Our experiments focus on a two-player collaborative cooking simulator that has recently been proposed as a challenge problem for coordination with humans. We find that FCP agents score significantly higher than SP, PP, and BCP when paired with novel agent and human partners. Furthermore, humans also report a strong subjective preference to partnering with FCP agents over all baselines.

----

## [1111] Training Feedback Spiking Neural Networks by Implicit Differentiation on the Equilibrium State

**Authors**: *Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Yisen Wang, Zhouchen Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/79a49b3e3762632813f9e35f4ba53d6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/79a49b3e3762632813f9e35f4ba53d6c-Abstract.html)

**Abstract**:

Spiking neural networks (SNNs) are brain-inspired models that enable energy-efficient implementation on neuromorphic hardware. However, the supervised training of SNNs remains a hard problem due to the discontinuity of the spiking neuron model. Most existing methods imitate the backpropagation framework and feedforward architectures for artificial neural networks, and use surrogate derivatives or compute gradients with respect to the spiking time to deal with the problem. These approaches either accumulate approximation errors or only propagate information limitedly through existing spikes, and usually require information propagation along time steps with large memory costs and biological implausibility. In this work, we consider feedback spiking neural networks, which are more brain-like, and propose a novel training method that does not rely on the exact reverse of the forward computation. First, we show that the average firing rates of SNNs with feedback connections would gradually evolve to an equilibrium state along time, which follows a fixed-point equation. Then by viewing the forward computation of feedback SNNs as a black-box solver for this equation, and leveraging the implicit differentiation on the equation, we can compute the gradient for parameters without considering the exact forward procedure. In this way, the forward and backward procedures are decoupled and therefore the problem of non-differentiable spiking functions is avoided. We also briefly discuss the biological plausibility of implicit differentiation, which only requires computing another equilibrium. Extensive experiments on MNIST, Fashion-MNIST, N-MNIST, CIFAR-10, and CIFAR-100 demonstrate the superior performance of our method for feedback models with fewer neurons and parameters in a small number of time steps. Our code is available at https://github.com/pkuxmq/IDE-FSNN.

----

## [1112] Online Selective Classification with Limited Feedback

**Authors**: *Aditya Gangrade, Anil Kag, Ashok Cutkosky, Venkatesh Saligrama*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/79b6245ff93841eb8c120cec9bf8be14-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/79b6245ff93841eb8c120cec9bf8be14-Abstract.html)

**Abstract**:

Motivated by applications to resource-limited and safety-critical domains, we study selective classification in the online learning model, wherein a predictor may abstain from classifying an instance. For example, this may model an adaptive decision to invoke more resources on this instance. Two salient aspects of the setting we consider are that the data may be non-realisable, due to which abstention may be a valid long-term action, and that feedback is only received when the learner abstains, which models the fact that reliable labels are only available when the resource intensive processing is invoked.Within this framework, we explore strategies that make few mistakes, while not abstaining too many times more than the best-in-hindsight error-free classifier from a given class. That is, the one that makes no mistakes, while abstaining the fewest number of times. We construct simple versioning-based schemes for any $\mu \in (0,1],$ that make most $T^\mu$ mistakes while incurring $\tilde{O}(T^{1-\mu})$ excess abstention against adaptive adversaries. We further show that this dependence on $T$ is tight, and provide illustrative experiments on realistic datasets.

----

## [1113] Controlled Text Generation as Continuous Optimization with Multiple Constraints

**Authors**: *Sachin Kumar, Eric Malmi, Aliaksei Severyn, Yulia Tsvetkov*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/79ec2a4246feb2126ecf43c4a4418002-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/79ec2a4246feb2126ecf43c4a4418002-Abstract.html)

**Abstract**:

As large-scale language model pretraining pushes the state-of-the-art in text generation, recent work has turned to controlling attributes of the text such models generate. While modifying the pretrained models via fine-tuning remains the popular approach, it incurs a significant computational cost and can be infeasible due to a lack of appropriate data. As an alternative, we propose \textsc{MuCoCO}---a flexible and modular algorithm for controllable inference from pretrained models. We formulate the decoding process as an optimization problem that allows for multiple attributes we aim to control to be easily incorporated as differentiable constraints. By relaxing this discrete optimization to a continuous one, we make use of Lagrangian multipliers and gradient-descent-based techniques to generate the desired text. We evaluate our approach on controllable machine translation and style transfer with multiple sentence-level attributes and observe significant improvements over baselines.

----

## [1114] S$^3$: Sign-Sparse-Shift Reparametrization for Effective Training of Low-bit Shift Networks

**Authors**: *Xinlin Li, Bang Liu, Yaoliang Yu, Wulong Liu, Chunjing Xu, Vahid Partovi Nia*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7a1d9028a78f418cb8f01909a348d9b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7a1d9028a78f418cb8f01909a348d9b2-Abstract.html)

**Abstract**:

Shift neural networks reduce computation complexity by removing expensive multiplication operations and quantizing continuous weights into low-bit discrete values, which are fast and energy-efficient compared to conventional neural networks. However, existing shift networks are sensitive to the weight initialization and yield a degraded performance caused by vanishing gradient and weight sign freezing problem. To address these issues, we propose S$^3$ re-parameterization, a novel technique for training low-bit shift networks. Our method decomposes a discrete parameter in a sign-sparse-shift 3-fold manner. This way, it efficiently learns a low-bit network with weight dynamics similar to full-precision networks and insensitive to weight initialization. Our proposed training method pushes the boundaries of shift neural networks and shows 3-bit shift networks compete with their full-precision counterparts in terms of top-1 accuracy on ImageNet.

----

## [1115] Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions

**Authors**: *Mathias Niepert, Pasquale Minervini, Luca Franceschi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html)

**Abstract**:

Combining discrete probability distributions and combinatorial optimization problems with neural network components has numerous applications but poses several challenges. We propose Implicit Maximum Likelihood Estimation (I-MLE), a framework for end-to-end learning of models combining discrete exponential family distributions and differentiable neural components. I-MLE is widely applicable as it only requires the ability to compute the most probable states and does not rely on smooth relaxations. The framework encompasses several approaches such as perturbation-based implicit differentiation and recent methods to differentiate through black-box combinatorial solvers. We introduce a novel class of noise distributions for approximating marginals via perturb-and-MAP. Moreover, we show that I-MLE simplifies to maximum likelihood estimation when used in some recently studied learning settings that involve combinatorial solvers. Experiments on several datasets suggest that I-MLE is competitive with and often outperforms existing approaches which rely on problem-specific relaxations.

----

## [1116] Scaling up Continuous-Time Markov Chains Helps Resolve Underspecification

**Authors**: *Alkis Gotovos, Rebekka Burkholz, John Quackenbush, Stefanie Jegelka*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7a50d83a1e70e9d96c3357438aed7a44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7a50d83a1e70e9d96c3357438aed7a44-Abstract.html)

**Abstract**:

Modeling the time evolution of discrete sets of items (e.g., genetic mutations) is a fundamental problem in many biomedical applications. We approach this problem through the lens of continuous-time Markov chains, and show that the resulting learning task is generally underspecified in the usual setting of cross-sectional data. We explore a perhaps surprising remedy: including a number of additional independent items can help determine time order, and hence resolve underspecification. This is in sharp contrast to the common practice of limiting the analysis to a small subset of relevant items, which is followed largely due to poor scaling of existing methods. To put our theoretical insight into practice, we develop an approximate likelihood maximization method for learning continuous-time Markov chains, which can scale to hundreds of items and is orders of magnitude faster than previous methods. We demonstrate the effectiveness of our approach on synthetic and real cancer data.

----

## [1117] Do Neural Optimal Transport Solvers Work? A Continuous Wasserstein-2 Benchmark

**Authors**: *Alexander Korotin, Lingxiao Li, Aude Genevay, Justin M. Solomon, Alexander Filippov, Evgeny Burnaev*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7a6a6127ff85640ec69691fb0f7cb1a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7a6a6127ff85640ec69691fb0f7cb1a2-Abstract.html)

**Abstract**:

Despite the recent popularity of neural network-based solvers for optimal transport (OT), there is no standard quantitative way to evaluate their performance. In this paper, we address this issue for quadratic-cost transport---specifically, computation of the Wasserstein-2 distance, a commonly-used formulation of optimal transport in machine learning. To overcome the challenge of computing ground truth transport maps between continuous measures needed to assess these solvers, we use input-convex neural networks (ICNN) to construct pairs of measures whose ground truth OT maps can be obtained analytically. This strategy yields pairs of continuous benchmark measures in high-dimensional spaces such as spaces of images. We thoroughly evaluate existing optimal transport solvers using these benchmark measures. Even though these solvers perform well in downstream tasks, many do not faithfully recover optimal transport maps. To investigate the cause of this discrepancy, we further test the solvers in a setting of image generation. Our study reveals crucial limitations of existing solvers and shows that increased OT accuracy does not necessarily correlate to better results downstream.

----

## [1118] Linear Convergence in Federated Learning: Tackling Client Heterogeneity and Sparse Gradients

**Authors**: *Aritra Mitra, Rayana H. Jaafar, George J. Pappas, Hamed Hassani*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7a6bda9ad6ffdac035c752743b7e9d0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7a6bda9ad6ffdac035c752743b7e9d0e-Abstract.html)

**Abstract**:

We consider a standard federated learning (FL) setup where a group of clients periodically coordinate with a central server to train a statistical model. We develop a general algorithmic framework called FedLin to tackle some of the key challenges intrinsic to FL, namely objective heterogeneity, systems heterogeneity, and infrequent and imprecise communication. Our framework is motivated by the observation that under these challenges, various existing FL algorithms suffer from a fundamental speed-accuracy conflict: they either guarantee linear convergence but to an incorrect point, or convergence to the global minimum but at a sub-linear rate, i.e., fast convergence comes at the expense of accuracy. In contrast, when the clients' local loss functions are smooth and strongly convex, we show that FedLin guarantees linear convergence to the global minimum, despite arbitrary objective and systems heterogeneity. We then establish matching upper and lower bounds on the convergence rate of FedLin that highlight the effects of infrequent, periodic communication. Finally, we show that FedLin preserves linear convergence rates under aggressive gradient sparsification, and quantify the effect of the compression level on the convergence rate. Notably, our work is the first to provide tight linear convergence rate guarantees, and constitutes the first comprehensive analysis of gradient sparsification in FL.

----

## [1119] On the Convergence of Prior-Guided Zeroth-Order Optimization Algorithms

**Authors**: *Shuyu Cheng, Guoqiang Wu, Jun Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7aaece81f2d731fbf8ee0ad3521002ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7aaece81f2d731fbf8ee0ad3521002ac-Abstract.html)

**Abstract**:

Zeroth-order (ZO) optimization is widely used to handle challenging tasks, such as query-based black-box adversarial attacks and reinforcement learning. Various attempts have been made to integrate prior information into the gradient estimation procedure based on finite differences, with promising empirical results. However, their convergence properties are not well understood. This paper makes an attempt to fill up this gap by analyzing the convergence of prior-guided ZO algorithms under a greedy descent framework with various gradient estimators. We provide a convergence guarantee for the prior-guided random gradient-free (PRGF) algorithms. Moreover, to further accelerate over greedy descent methods, we present a new accelerated random search (ARS) algorithm that incorporates prior information, together with a convergence analysis. Finally, our theoretical results are confirmed by experiments on several numerical benchmarks as well as adversarial attacks.

----

## [1120] Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning

**Authors**: *Milad Abdollahzadeh, Touba Malekzadeh, Ngai-Man Cheung*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b3403f79b478699224bb449509694cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b3403f79b478699224bb449509694cf-Abstract.html)

**Abstract**:

Multimodal meta-learning is a recent problem that extends conventional few-shot meta-learning by generalizing its setup to diverse multimodal task distributions. This setup makes a step towards mimicking how humans make use of a diverse set of prior skills to learn new skills. Previous work has achieved encouraging performance. In particular, in spite of the diversity of the multimodal tasks, previous work claims that a single meta-learner trained on a multimodal distribution can sometimes outperform multiple specialized meta-learners trained on individual unimodal distributions. The improvement is attributed to knowledge transfer between different modes of task distributions. However, there is no deep investigation to verify and understand the knowledge transfer between multimodal tasks. Our work makes two contributions to multimodal meta-learning. First, we propose a method to quantify knowledge transfer between tasks of different modes at a micro-level. Our quantitative, task-level analysis is inspired by the recent transference idea from multi-task learning. Second, inspired by hard parameter sharing in multi-task learning and a new interpretation of related work, we propose a new multimodal meta-learner that outperforms existing work by considerable margins. While the major focus is on multimodal meta-learning, our work also attempts to shed light on task interaction in conventional meta-learning. The code for this project is available at https://miladabd.github.io/KML.

----

## [1121] Dynamic Sasvi: Strong Safe Screening for Norm-Regularized Least Squares

**Authors**: *Hiroaki Yamada, Makoto Yamada*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b5b23f4aadf9513306bcd59afb6e4c9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b5b23f4aadf9513306bcd59afb6e4c9-Abstract.html)

**Abstract**:

A recently introduced technique, called safe screening,'' for a sparse optimization problem allows us to identify irrelevant variables in the early stages of optimization. In this paper, we first propose a flexible framework for safe screening based on the Fenchel--Rockafellar duality and then derive a strong safe screening rule for norm-regularized least squares using the proposed framework. We refer to the proposed screening rule for norm-regularized least squares asdynamic Sasvi'' because it can be interpreted as a generalization of Sasvi. Unlike the original Sasvi, it does not require the exact solution of a more strongly regularized problem; hence, it works safely in practice. We show that our screening rule always eliminates more features compared with the existing state-of-the-art methods.

----

## [1122] What Matters for Adversarial Imitation Learning?

**Authors**: *Manu Orsini, Anton Raichuk, Léonard Hussenot, Damien Vincent, Robert Dadashi, Sertan Girgin, Matthieu Geist, Olivier Bachem, Olivier Pietquin, Marcin Andrychowicz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b647a7d88f4d6319bf0d600d168dbeb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b647a7d88f4d6319bf0d600d168dbeb-Abstract.html)

**Abstract**:

Adversarial imitation learning has become a popular framework for imitation in continuous control. Over the years, several variations of its components were proposed to enhance the performance of the learned policies as well as the sample complexity of the algorithm. In practice, these choices are rarely tested all together in rigorous empirical studies.It is therefore difficult to discuss and understand what choices, among the high-level algorithmic options as well as  low-level implementation details, matter. To tackle this issue, we implement more than 50 of these choices in a generic adversarial imitation learning frameworkand investigate their impacts in a large-scale study (>500k trained agents) with both synthetic and human-generated demonstrations. We analyze the key results and highlight the most surprising findings.

----

## [1123] Sequential Causal Imitation Learning with Unobserved Confounders

**Authors**: *Daniel Kumor, Junzhe Zhang, Elias Bareinboim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b670d553471ad0fd7491c75bad587ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b670d553471ad0fd7491c75bad587ff-Abstract.html)

**Abstract**:

"Monkey see monkey do" is an age-old adage, referring to naive imitation without a deep understanding of a system's underlying mechanics. Indeed, if a demonstrator has access to information unavailable to the imitator (monkey), such as a different set of sensors, then no matter how perfectly the imitator models its perceived environment (See), attempting to directly reproduce the demonstrator's behavior (Do) can lead to poor outcomes. Imitation learning in the presence of a mismatch between demonstrator and imitator has been studied in the literature under the rubric of causal imitation learning  (Zhang et. al. 2020), but existing solutions are limited to single-stage decision-making. This paper investigates the problem of causal imitation learning in sequential settings, where the imitator must make multiple decisions per episode. We develop a graphical criterion that is both necessary and sufficient for determining the feasibility of causal imitation, providing conditions when an imitator can match a demonstrator's performance despite differing capabilities. Finally, we provide an efficient algorithm for determining imitability, and corroborate our theory with simulations.

----

## [1124] Topic Modeling Revisited: A Document Graph-based Neural Network Perspective

**Authors**: *Dazhong Shen, Chuan Qin, Chao Wang, Zheng Dong, Hengshu Zhu, Hui Xiong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b6982e584636e6a1cda934f1410299c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b6982e584636e6a1cda934f1410299c-Abstract.html)

**Abstract**:

Most topic modeling approaches are based on the bag-of-words assumption, where each word is required to be conditionally independent in the same document. As a result, both of the generative story and the topic formulation have totally ignored the semantic dependency among words, which is important for improving the semantic comprehension and model interpretability. To this end, in this paper, we revisit the task of topic modeling by transforming each document into a directed graph with word dependency as edges between word nodes, and develop a novel approach, namely Graph Neural Topic Model (GNTM). Specifically, in GNTM, a well-defined probabilistic generative story is designed to model both the graph structure and word sets with multinomial distributions on the vocabulary and word dependency edge set as the topics. Meanwhile, a Neural Variational Inference (NVI) approach is proposed to learn our model with graph neural networks to encode the document graphs. Besides, we theoretically demonstrate that Latent Dirichlet Allocation (LDA) can be derived from GNTM as a special case with similar objective functions. Finally, extensive experiments on four benchmark datasets have clearly demonstrated the effectiveness and interpretability of GNTM compared with state-of-the-art baselines.

----

## [1125] Hard-Attention for Scalable Image Classification

**Authors**: *Athanasios Papadopoulos, Pawel Korus, Nasir D. Memon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b7916dd2de56297aa29cccb2bbf48d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b7916dd2de56297aa29cccb2bbf48d4-Abstract.html)

**Abstract**:

Can we leverage high-resolution information without the unsustainable quadratic complexity to input scale? We propose Traversal Network (TNet), a novel multi-scale hard-attention architecture, which traverses image scale-space in a top-down fashion, visiting only the most informative image regions along the way. TNet offers an adjustable trade-off between accuracy and complexity, by changing the number of attended image locations. We compare our model against hard-attention baselines on ImageNet, achieving higher accuracy with less resources (FLOPs, processing time and memory). We further test our model on fMoW dataset, where we process satellite images of size up to $896 \times 896$ px, getting up to $2.5$x faster processing compared to baselines operating on the same resolution, while achieving higher accuracy as well. TNet is modular, meaning that most classification models could be adopted as its backbone for feature extraction, making the reported performance gains orthogonal to benefits offered by existing optimized deep models. Finally, hard-attention guarantees a degree of interpretability to our model's predictions, without any extra cost beyond inference.

----

## [1126] Fast Routing under Uncertainty: Adaptive Learning in Congestion Games via Exponential Weights

**Authors**: *Dong Quan Vu, Kimon Antonakopoulos, Panayotis Mertikopoulos*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7b86f36d139d8581d4b5a4f155ba431c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7b86f36d139d8581d4b5a4f155ba431c-Abstract.html)

**Abstract**:

We examine an adaptive learning framework for nonatomic congestion games where the players' cost functions may be subject to exogenous fluctuations (e.g., due to disturbances in the network, variations in the traffic going through a link). In this setting, the popular multiplicative/ exponential weights algorithm enjoys an $\mathcal{O}(1/\sqrt{T})$ equilibrium convergence rate; however, this rate is suboptimal in static environments---i.e., when the network is not subject to randomness. In this static regime, accelerated algorithms achieve an $\mathcal{O}(1/T^{2})$ convergence speed, but they fail to converge altogether in stochastic problems. To fill this gap, we propose a novel,  adaptive exponential weights method---dubbed AdaWeight---that seamlessly interpolates between the $\mathcal{O}(1/T^{2})$  and $\mathcal{O}(1/\sqrt{T})$ rates in the static and stochastic regimes respectively. Importantly, this "best-of-both-worlds" guarantee does not require any prior knowledge of the problem's parameters or tuning by the optimizer; in addition, the method's convergence speed depends subquadratically on the size of the network (number of vertices and edges), so it scales gracefully to large, real-life urban networks.

----

## [1127] Profiling Pareto Front With Multi-Objective Stein Variational Gradient Descent

**Authors**: *Xingchao Liu, Xin Tong, Qiang Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html)

**Abstract**:

Finding diverse and representative Pareto solutions from the Pareto front is a key challenge in multi-objective optimization (MOO). In this work, we propose a novel gradient-based algorithm for profiling Pareto front by using Stein variational gradient descent (SVGD). We also provide a counterpart of our method based on Langevin dynamics. Our methods iteratively update a set of points in a parallel fashion to push them towards the Pareto front using multiple gradient descent, while encouraging the diversity between the particles by using the repulsive force mechanism in SVGD, or diffusion noise in Langevin dynamics. Compared with existing gradient-based methods that require predefined preference functions, our method can work efficiently in high dimensional problems, and can obtain more diverse solutions evenly distributed in the Pareto front. Moreover, our methods are theoretically guaranteed to converge to the Pareto front. We demonstrate the effectiveness of our method, especially the SVGD algorithm, through extensive experiments, showing its superiority over existing gradient-based algorithms.

----

## [1128] MAP Propagation Algorithm: Faster Learning with a Team of Reinforcement Learning Agents

**Authors**: *Stephen Chung*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c05147f3029c97ce26c0cb0b2469fca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c05147f3029c97ce26c0cb0b2469fca-Abstract.html)

**Abstract**:

Nearly all state-of-the-art deep learning algorithms rely on error backpropagation, which is generally regarded as biologically implausible. An alternative way of training an artificial neural network is through treating each unit in the network as a reinforcement learning agent, and thus the network is considered as a team of agents. As such, all units can be trained by REINFORCE, a local learning rule modulated by a global signal that is more consistent with biologically observed forms of synaptic plasticity. Although this learning rule follows the gradient of return in expectation, it suffers from high variance and thus the low speed of learning, rendering it impractical to train deep networks. We therefore propose a novel algorithm called MAP propagation to reduce this variance significantly while retaining the local property of the learning rule. Experiments demonstrated that MAP propagation could solve common reinforcement learning tasks at a similar speed to backpropagation when applied to an actor-critic network. Our work thus allows for the broader application of teams of agents in deep reinforcement learning.

----

## [1129] TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up

**Authors**: *Yifan Jiang, Shiyu Chang, Zhangyang Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c220a2091c26a7f5e9f1cfb099511e3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c220a2091c26a7f5e9f1cfb099511e3-Abstract.html)

**Abstract**:

The recent explosive interest on transformers has suggested their potential to become powerful ``universal" models for computer vision tasks, such as classification, detection, and segmentation. While those attempts mainly study the discriminative models, we explore transformers on some more notoriously difficult vision tasks, e.g., generative adversarial networks (GANs).  Our goal is to conduct the first pilot study in building a GAN \textit{completely free of convolutions}, using only pure transformer-based architectures. Our vanilla GAN architecture, dubbed \textbf{TransGAN}, consists of a memory-friendly transformer-based generator that progressively increases feature resolution, and correspondingly a multi-scale discriminator to capture simultaneously semantic contexts and low-level textures. On top of them, we introduce the new module of grid self-attention for alleviating the memory bottleneck further, in order to scale up TransGAN to high-resolution generation. We also develop a unique training recipe including a series of techniques that can mitigate the training instability issues of TransGAN, such as data augmentation, modified normalization, and relative position encoding. Our best architecture achieves highly competitive performance compared to current state-of-the-art GANs using convolutional backbones. Specifically, TransGAN sets \textbf{new state-of-the-art} inception score of 10.43 and FID of 18.28 on STL-10. It also reaches the inception score of 9.02 and FID of 9.26  on CIFAR-10, and 5.28 FID on CelebA $\mathbf{128} \times \mathbf{128}$, respectively: both on par with the current best results and outperforming StyleGAN-V2. When it comes to higher-resolution (e.g. $\mathbf{256} \times \mathbf{256}$) generation tasks, such as on CelebA-HQ and LSUN-Church, TransGAN continues to produce diverse visual examples with high fidelity and impressive texture details. In addition, we dive deep into the transformer-based generation models to understand how their behaviors differ from convolutional ones, by visualizing training dynamics. The code is available at: https://github.com/VITA-Group/TransGAN.

----

## [1130] A Central Limit Theorem for Differentially Private Query Answering

**Authors**: *Jinshuo Dong, Weijie J. Su, Linjun Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c2c48a32443ad8f805e48520f3b26a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c2c48a32443ad8f805e48520f3b26a4-Abstract.html)

**Abstract**:

Perhaps the single most important use case for differential privacy is to privately answer numerical queries, which is usually achieved by adding noise to the answer vector. The central question is, therefore, to understand which noise distribution optimizes the privacy-accuracy trade-off, especially when the dimension of the answer vector is high. Accordingly, an extensive literature has been dedicated to the question and the upper and lower bounds have been successfully matched up to constant factors (Bun et al.,2018; Steinke & Ullman, 2017). In this paper, we take a novel approach to address this important optimality question. We first demonstrate an intriguing central limit theorem phenomenon in the high-dimensional regime. More precisely, we prove that a mechanism is approximately Gaussian Differentially Private (Dong et al., 2021) if the added noise satisfies certain conditions. In particular, densities proportional to $\mathrm{e}^{-\|x\|_p^\alpha}$, where $\|x\|_p$ is the standard $\ell_p$-norm, satisfies the conditions. Taking this perspective, we make use of the Cramer--Rao inequality and show an "uncertainty principle"-style result: the product of privacy parameter and the $\ell_2$-loss of the mechanism is lower bounded by the dimension. Furthermore, the Gaussian mechanism achieves the constant-sharp optimal privacy-accuracy trade-off among all such noises. Our findings are corroborated by numerical experiments.

----

## [1131] Differential Privacy Dynamics of Langevin Diffusion and Noisy Gradient Descent

**Authors**: *Rishav Chourasia, Jiayuan Ye, Reza Shokri*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c6c1a7bfde175bed616b39247ccace1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c6c1a7bfde175bed616b39247ccace1-Abstract.html)

**Abstract**:

What is the information leakage of an iterative randomized learning algorithm about its training data, when the internal state of the algorithm is \emph{private}? How much is the contribution of each specific training epoch to the information leakage through the released model? We study this problem for noisy gradient descent algorithms, and model the \emph{dynamics} of R\'enyi differential privacy loss throughout the training process.  Our analysis traces a provably \emph{tight} bound on the R\'enyi divergence between the pair of probability distributions over parameters of models trained on neighboring datasets.  We prove that the privacy loss converges exponentially fast, for smooth and strongly convex loss functions, which is a significant improvement over composition theorems (which over-estimate the privacy loss by upper-bounding its total value over all intermediate gradient computations). For Lipschitz, smooth, and strongly convex loss functions, we prove optimal utility with a small gradient complexity for noisy gradient descent algorithms.

----

## [1132] Data driven semi-supervised learning

**Authors**: *Maria-Florina Balcan, Dravyansh Sharma*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c93ebe873ef213123c8af4b188e7558-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c93ebe873ef213123c8af4b188e7558-Abstract.html)

**Abstract**:

We consider a novel data driven approach for designing semi-supervised learning algorithms that can effectively learn with only a small number of labeled examples. We focus on graph-based techniques, where the unlabeled examples are connected in a graph  under the implicit assumption that similar nodes likely have similar labels. Over the past two decades, several elegant graph-based semi-supervised learning algorithms for inferring the labels of the unlabeled examples given the graph and a few labeled examples have been proposed. However, the problem of how to create the graph (which impacts the practical usefulness of these methods significantly) has been relegated to heuristics and domain-specific art, and no general principles have been proposed. In this work we present a  novel data driven approach for learning the graph and provide strong formal guarantees in both the distributional and online learning formalizations. We show how to leverage problem instances coming from an underlying problem domain to learn the graph hyperparameters for commonly used parametric families of graphs that provably perform well on new instances from the same domain. We obtain low regret and efficient algorithms in the online setting, and generalization guarantees in the distributional setting. We also show how to combine several very different similarity metrics and learn multiple  hyperparameters, our results hold for large classes of problems. We expect some of the tools and techniques we develop along the way to be of independent interest, for data driven algorithms more generally.

----

## [1133] Meta-Learning via Learning with Distributed Memory

**Authors**: *Sudarshan Babu, Pedro Savarese, Michael Maire*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7c9e9afa5a9dc68ccaf27d9effeb9383-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7c9e9afa5a9dc68ccaf27d9effeb9383-Abstract.html)

**Abstract**:

We demonstrate that efficient meta-learning can be achieved via end-to-end training of deep neural networks with memory distributed across layers.  The persistent state of this memory assumes the entire burden of guiding task adaptation.  Moreover, its distributed nature is instrumental in orchestrating adaptation.  Ablation experiments demonstrate that providing relevant feedback to memory units distributed across the depth of the network enables them to guide adaptation throughout the entire network.  Our results show that this is a successful strategy for simplifying meta-learning -- often cast as a bi-level optimization problem -- to standard end-to-end training, while outperforming gradient-based, prototype-based, and other memory-based meta-learning strategies.  Additionally, our adaptation strategy naturally handles online learning scenarios with a significant delay between observing a sample and its corresponding label -- a setting in which other approaches struggle.  Adaptation via distributed memory is effective across a wide range of learning tasks, ranging from classification to online few-shot semantic segmentation.

----

## [1134] Physics-Integrated Variational Autoencoders for Robust and Interpretable Generative Modeling

**Authors**: *Naoya Takeishi, Alexandros Kalousis*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7ca57a9f85a19a6e4b9a248c1daca185-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ca57a9f85a19a6e4b9a248c1daca185-Abstract.html)

**Abstract**:

Integrating physics models within machine learning models holds considerable promise toward learning robust models with improved interpretability and abilities to extrapolate. In this work, we focus on the integration of incomplete physics models into deep generative models. In particular, we introduce an architecture of variational autoencoders (VAEs) in which a part of the latent space is grounded by physics. A key technical challenge is to strike a balance between the incomplete physics and trainable components such as neural networks for ensuring that the physics part is used in a meaningful manner. To this end, we propose a regularized learning method that controls the effect of the trainable components and preserves the semantics of the physics-based latent variables as intended. We not only demonstrate generative performance improvements over a set of synthetic and real-world datasets, but we also show that we learn robust models that can consistently extrapolate beyond the training distribution in a meaningful manner. Moreover, we show that we can control the generative process in an interpretable manner.

----

## [1135] Characterizing the risk of fairwashing

**Authors**: *Ulrich Aïvodji, Hiromi Arai, Sébastien Gambs, Satoshi Hara*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7caf5e22ea3eb8175ab518429c8589a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7caf5e22ea3eb8175ab518429c8589a4-Abstract.html)

**Abstract**:

Fairwashing refers to the risk that an unfair black-box model can be explained by a fairer model through post-hoc explanation manipulation. In this paper, we investigate the capability of fairwashing attacks by analyzing their fidelity-unfairness trade-offs. In particular, we show that fairwashed explanation models can generalize beyond the suing group (i.e., data points that are being explained), meaning that a fairwashed explainer can be used to rationalize subsequent unfair decisions of a black-box model. We also demonstrate that fairwashing attacks can transfer across black-box models, meaning that other black-box models can perform fairwashing without explicitly using their predictions. This generalization and transferability of fairwashing attacks imply that their detection will be difficult in practice. Finally, we propose an approach to quantify the risk of fairwashing, which is based on the computation of the range of the unfairness of high-fidelity explainers.

----

## [1136] Qimera: Data-free Quantization with Synthetic Boundary Supporting Samples

**Authors**: *Kanghyun Choi, Deokki Hong, Noseong Park, Youngsok Kim, Jinho Lee*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7cc234202e98d2722580858573fd0817-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7cc234202e98d2722580858573fd0817-Abstract.html)

**Abstract**:

Model quantization is known as a promising method to compress deep neural networks, especially for inferences on lightweight mobile or edge devices. However, model quantization usually requires access to the original training data to maintain the accuracy of the full-precision models, which is often infeasible in real-world scenarios for security and privacy issues.A popular approach to perform quantization without access to the original data is to use synthetically generated samples, based on batch-normalization statistics or adversarial learning.However, the drawback of such approaches is that they primarily rely on random noise input to the generator to attain diversity of the synthetic samples. We find that this is often insufficient to capture the distribution of the original data, especially around the decision boundaries.To this end, we propose Qimera, a method that uses superposed latent embeddings to generate synthetic boundary supporting samples.For the superposed embeddings to better reflect the original distribution, we also propose using an additional disentanglement mapping layer and extracting information from the full-precision model.The experimental results show that Qimera achieves state-of-the-art performances for various settings on data-free quantization. Code is available at https://github.com/iamkanghyunchoi/qimera.

----

## [1137] Embedding Principle of Loss Landscape of Deep Neural Networks

**Authors**: *Yaoyu Zhang, Zhongwang Zhang, Tao Luo, Zhi-Qin John Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7cc532d783a7461f227a5da8ea80bfe1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7cc532d783a7461f227a5da8ea80bfe1-Abstract.html)

**Abstract**:

Understanding the structure of loss landscape of deep neural networks (DNNs) is obviously important. In this work, we prove an embedding principle that the loss landscape of a DNN "contains" all the critical points of all the narrower DNNs. More precisely, we propose a critical embedding such that any critical point, e.g., local or global minima, of a narrower DNN can be embedded to a critical point/affine subspace of the target DNN with higher degeneracy and preserving the DNN output function. Note that, given any training data, differentiable loss function and differentiable activation function, this embedding structure of critical points holds.This general structure of DNNs is starkly different from other nonconvex problems such as protein-folding.Empirically, we find that a wide DNN is often attracted by highly-degenerate critical points that are embedded from narrow DNNs. The embedding principle provides a new perspective to study the general easy optimization of wide DNNs and unravels a potential implicit low-complexity regularization during the training.Overall, our work provides a skeleton for the study of loss landscape of DNNs and its implication, by which a more exact and comprehensive understanding can be anticipated in the near future.

----

## [1138] Adversarial Reweighting for Partial Domain Adaptation

**Authors**: *Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7ce3284b743aefde80ffd9aec500e085-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ce3284b743aefde80ffd9aec500e085-Abstract.html)

**Abstract**:

Partial domain adaptation (PDA) has gained much attention due to its practical setting. The current PDA methods usually adapt the feature extractor by aligning the target and reweighted source domain distributions. In this paper, we experimentally find that the feature adaptation by the reweighted distribution alignment in some state-of-the-art PDA methods is not robust to the ``noisy'' weights of source domain data, leading to negative domain transfer on some challenging benchmarks. To tackle the challenge of negative domain transfer, we propose a novel Adversarial Reweighting (AR) approach that adversarially learns the weights of source domain data to align the source and target domain distributions, and the transferable deep recognition network is learned on the reweighted source domain data. Based on this idea, we propose a training algorithm that alternately updates the parameters of the network and optimizes the weights of source domain data. Extensive experiments show that our method achieves state-of-the-art results on the benchmarks of ImageNet-Caltech, Office-Home, VisDA-2017, and DomainNet. Ablation studies also confirm the effectiveness of our approach.

----

## [1139] M-FAC: Efficient Matrix-Free Approximations of Second-Order Information

**Authors**: *Elias Frantar, Eldar Kurtic, Dan Alistarh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7cfd5df443b4eb0d69886a583b33de4c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7cfd5df443b4eb0d69886a583b33de4c-Abstract.html)

**Abstract**:

Efficiently approximating local curvature information of the loss function is a useful tool for the optimization and compression of deep neural networks. Yet, most existing methods to approximate second-order information have high computational or storage costs, limiting their practicality. In this work, we investigate matrix-free approaches for estimating Inverse-Hessian Vector Products (IHVPs) for the case when the Hessian can be approximated as a sum of rank-one matrices, as in the classic approximation of the Hessian by the empirical Fisher matrix. The first algorithm we propose is tailored towards network compression and can compute the IHVP for dimension $d$ given a fixed set of $m$ rank-one matrices using $O(dm^2)$ precomputation, $O(dm)$ cost for computing the IHVP and query cost $O(m)$ for computing any single element of the inverse Hessian approximation. The second algorithm targets an optimization setting, where we wish to compute the product between the inverse Hessian, estimated over a sliding  window of optimization steps, and a given gradient direction. We give an algorithm with cost $O(dm + m^2)$ for computing the IHVP and $O(dm + m^3)$ for adding or removing any gradient from the sliding window. We show that both algorithms yield competitive results for network pruning and optimization, respectively, with significantly lower computational overhead relative to existing second-order methods.

----

## [1140] Graph Adversarial Self-Supervised Learning

**Authors**: *Longqi Yang, Liangliang Zhang, Wenjing Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d3010c11d08cf990b7614d2c2ca9098-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d3010c11d08cf990b7614d2c2ca9098-Abstract.html)

**Abstract**:

This paper studies a long-standing problem of learning the representations of a whole graph without human supervision. The recent self-supervised learning methods train models to be invariant to the transformations (views) of the inputs. However, designing these views requires the experience of human experts. Inspired by adversarial training, we propose an adversarial self-supervised learning (\texttt{GASSL}) framework for learning unsupervised representations of graph data without any handcrafted views. \texttt{GASSL} automatically generates challenging views by adding perturbations to the input and are adversarially trained with respect to the encoder. Our method optimizes the min-max problem and utilizes a gradient accumulation strategy to accelerate the training process. Experimental on ten graph classification datasets show that the proposed approach is superior to state-of-the-art self-supervised learning baselines, which are competitive with supervised models.

----

## [1141] Anti-Backdoor Learning: Training Clean Models on Poisoned Data

**Authors**: *Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, Xingjun Ma*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d38b1e9bd793d3f45e0e212a729a93c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d38b1e9bd793d3f45e0e212a729a93c-Abstract.html)

**Abstract**:

Backdoor attack has emerged as a major security threat to deep neural networks (DNNs). While existing defense methods have demonstrated promising results on detecting or erasing backdoors, it is still not clear whether robust training methods can be devised to prevent the backdoor triggers being injected into the trained model in the first place. In this paper, we introduce the concept of \emph{anti-backdoor learning}, aiming to train \emph{clean} models given backdoor-poisoned data. We frame the overall learning process as a dual-task of learning the \emph{clean} and the \emph{backdoor} portions of data. From this view, we identify two inherent characteristics of backdoor attacks as their weaknesses: 1) the models learn backdoored data much faster than learning with clean data, and the stronger the attack the faster the model converges on backdoored data; 2) the backdoor task is tied to a specific class (the backdoor target class). Based on these two weaknesses, we propose a general learning scheme, Anti-Backdoor Learning (ABL), to automatically prevent backdoor attacks during training. ABL introduces a two-stage \emph{gradient ascent} mechanism for standard training to 1) help isolate backdoor examples at an early training stage, and 2) break the correlation between backdoor examples and the target class at a later training stage. Through extensive experiments on multiple benchmark datasets against 10 state-of-the-art attacks, we empirically show that ABL-trained models on backdoor-poisoned data achieve the same performance as they were trained on purely clean data. Code is available at \url{https://github.com/bboylyg/ABL}.

----

## [1142] Locally Most Powerful Bayesian Test for Out-of-Distribution Detection using Deep Generative Models

**Authors**: *Keunseo Kim, Juncheol Shin, Heeyoung Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d3e28d14440d6c07f73b7557e3d9602-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d3e28d14440d6c07f73b7557e3d9602-Abstract.html)

**Abstract**:

Several out-of-distribution (OOD) detection scores have been recently proposed for deep generative models because the direct use of the likelihood threshold for OOD detection has been shown to be problematic. In this paper, we propose a new OOD score based on a Bayesian hypothesis test called the locally most powerful Bayesian test (LMPBT). The LMPBT is locally most powerful in that the alternative hypothesis (the representative parameter for the OOD sample) is specified to maximize the probability that the Bayes factor exceeds the evidence threshold in favor of the alternative hypothesis provided that the parameter specified under the alternative hypothesis is in the neighborhood of the parameter specified under the null hypothesis. That is, under this neighborhood parameter condition, the test with the proposed alternative hypothesis maximizes the probability of correct detection of OOD samples. We also propose numerical strategies for more efficient and reliable computation of the LMPBT for practical application to deep generative models. Evaluations conducted of the OOD detection performance of the LMPBT on various benchmark datasets demonstrate its superior performance over existing OOD detection methods.

----

## [1143] Stable Neural ODE with Lyapunov-Stable Equilibrium Points for Defending Against Adversarial Attacks

**Authors**: *Qiyu Kang, Yang Song, Qinxu Ding, Wee Peng Tay*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d5430cf85f78c4b7aa09813b14bce0d-Abstract.html)

**Abstract**:

Deep neural networks (DNNs) are well-known to be vulnerable to adversarial attacks, where malicious human-imperceptible perturbations are included in the input to the deep network to fool it into making a wrong classification. Recent studies have demonstrated that neural Ordinary Differential Equations (ODEs) are intrinsically more robust against adversarial attacks compared to vanilla DNNs. In this work, we propose a neural ODE with Lyapunov-stable equilibrium points for defending against adversarial attacks (SODEF). By ensuring that the equilibrium points of the ODE solution used as part of SODEF are Lyapunov-stable, the ODE solution for an input with a small perturbation converges to the same solution as the unperturbed input. We provide theoretical results that give insights into the stability of SODEF as well as the choice of regularizers to ensure its stability. Our analysis suggests that our proposed regularizers force the extracted feature points to be within a neighborhood of the Lyapunov-stable equilibrium points of the SODEF ODE. SODEF is compatible with many defense methods and can be applied to any neural network's final regressor layer to enhance its stability against adversarial attacks.

----

## [1144] Robust Compressed Sensing MRI with Deep Generative Priors

**Authors**: *Ajil Jalal, Marius Arvinte, Giannis Daras, Eric Price, Alexandros G. Dimakis, Jonathan I. Tamir*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d6044e95a16761171b130dcb476a43e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d6044e95a16761171b130dcb476a43e-Abstract.html)

**Abstract**:

The CSGM framework (Bora-Jalal-Price-Dimakis'17) has shown that deepgenerative priors can be powerful tools for solving inverse problems.However, to date this framework has been empirically successful only oncertain datasets (for example, human faces and MNIST digits), and itis known to perform poorly on out-of-distribution samples. In thispaper, we present the first successful application of the CSGMframework on clinical MRI data. We train a generative prior on brainscans from the fastMRI dataset, and show that posterior sampling viaLangevin dynamics achieves high quality reconstructions. Furthermore,our experiments and theory show that posterior sampling is robust tochanges in the ground-truth distribution and measurement process.Our code and models are available at: \url{https://github.com/utcsilab/csgm-mri-langevin}.

----

## [1145] H-NeRF: Neural Radiance Fields for Rendering and Temporal Reconstruction of Humans in Motion

**Authors**: *Hongyi Xu, Thiemo Alldieck, Cristian Sminchisescu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d62a275027741d98073d42b8f735c68-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d62a275027741d98073d42b8f735c68-Abstract.html)

**Abstract**:

We present neural radiance fields for rendering and temporal (4D) reconstruction of humans in motion (H-NeRF), as captured by a sparse set of cameras or even from a monocular video. Our approach combines ideas from neural scene representation, novel-view synthesis, and implicit statistical geometric human representations, coupled using novel loss functions. Instead of learning a radiance field with a uniform occupancy prior, we constrain it by a structured implicit human body model, represented using signed distance functions. This allows us to robustly fuse information from sparse views and generalize well beyond the poses or views observed in training. Moreover, we apply geometric constraints to co-learn the structure of the observed subject -- including both body and clothing -- and to regularize the radiance field to geometrically plausible solutions. Extensive experiments on multiple datasets demonstrate the robustness and the accuracy of our approach, its generalization capabilities significantly outside a small training set of poses and views, and statistical extrapolation beyond the observed shape.

----

## [1146] DOBF: A Deobfuscation Pre-Training Objective for Programming Languages

**Authors**: *Marie-Anne Lachaux, Baptiste Rozière, Marc Szafraniec, Guillaume Lample*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7d6548bdc0082aacc950ed35e91fcccb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7d6548bdc0082aacc950ed35e91fcccb-Abstract.html)

**Abstract**:

Recent advances in self-supervised learning have dramatically improved the state of the art on a wide variety of tasks. However, research in language model pre-training has mostly focused on natural languages, and it is unclear whether models like BERT and its variants provide the best pre-training when applied to other modalities, such as source code. In this paper, we introduce a new pre-training objective, DOBF, that leverages the structural aspect of programming languages and pre-trains a model to recover the original version of obfuscated source code. We show that models pre-trained with DOBF significantly outperform existing approaches on multiple downstream tasks, providing relative improvements of up to 12.2% in unsupervised code translation, and 5.3% in natural language code search. Incidentally, we found that our pre-trained model is able to deobfuscate fully obfuscated source files, and to suggest descriptive variable names.

----

## [1147] Detecting Errors and Estimating Accuracy on Unlabeled Data with Self-training Ensembles

**Authors**: *Jiefeng Chen, Frederick Liu, Besim Avci, Xi Wu, Yingyu Liang, Somesh Jha*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7dd3ed2e12d7967b656d156d50308263-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7dd3ed2e12d7967b656d156d50308263-Abstract.html)

**Abstract**:

When a deep learning model is deployed in the wild, it can encounter test data drawn from distributions different from the training data distribution and suffer drop in performance. For safe deployment, it is essential to estimate the accuracy of the pre-trained model on the test data. However, the labels for the test inputs are usually not immediately available in practice, and obtaining them can be expensive. This observation leads to two challenging tasks: (1) unsupervised accuracy estimation, which aims to estimate the accuracy of a pre-trained classifier on a set of unlabeled test inputs; (2) error detection, which aims to identify mis-classified test inputs. In this paper, we propose a principled and practically effective framework that simultaneously addresses the two tasks. The proposed framework iteratively learns an ensemble of models to identify mis-classified data points and performs self-training to improve the ensemble with the identified points. Theoretical analysis demonstrates that our framework enjoys provable guarantees for both accuracy estimation and error detection under mild conditions readily satisfied by practical deep learning models. Along with the framework, we proposed and experimented with two instantiations and achieved state-of-the-art results on 59 tasks. For example, on iWildCam, one instantiation reduces the estimation error for unsupervised accuracy estimation by at least 70% and improves the F1 score for error detection by at least 4.7% compared to existing methods.

----

## [1148] Exploiting Chain Rule and Bayes' Theorem to Compare Probability Distributions

**Authors**: *Huangjie Zheng, Mingyuan Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7e0ff37942c2de60cbcbd27041196ce3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7e0ff37942c2de60cbcbd27041196ce3-Abstract.html)

**Abstract**:

To measure the difference between two probability distributions, referred to as the source and target, respectively, we exploit both the chain rule and Bayes' theorem to construct conditional transport (CT), which is constituted by both a forward component and a backward one. The forward CT is the expected cost of moving a source data point to a target one, with their joint distribution defined by the product of the source probability density function (PDF) and a source-dependent conditional distribution, which is related to the target PDF via Bayes' theorem. The backward CT is defined by reversing the direction. The CT cost can be approximated by replacing the source and target PDFs with their discrete empirical distributions supported on mini-batches, making it amenable to implicit distributions and stochastic gradient descent-based optimization. When applied to train a generative model, CT is shown to strike a good balance between mode-covering and mode-seeking behaviors and strongly resist mode collapse. On a wide variety of benchmark datasets for generative modeling, substituting the default statistical distance of an existing generative adversarial network with CT is shown to consistently improve the performance. PyTorch code is provided.

----

## [1149] Actively Identifying Causal Effects with Latent Variables Given Only Response Variable Observable

**Authors**: *Tian-Zuo Wang, Zhi-Hua Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7e7e69ea3384874304911625ac34321c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7e7e69ea3384874304911625ac34321c-Abstract.html)

**Abstract**:

In many real tasks, it is generally desired to study the causal effect on a specific target (response variable) only, with no need to identify the thorough causal effects involving all variables. In this paper, we attempt to identify such effects by a few active interventions where only the response variable is observable. This task is challenging because the causal graph is unknown and even there may exist latent confounders. To learn the necessary structure for identifying the effects, we provide the graphical characterization that allows us to efficiently estimate all possible causal effects in a partially mixed ancestral graph (PMAG) by generalized back-door criterion. The characterization guides learning a local structure with the interventional data. Theoretical analysis and empirical studies validate the effectiveness and efficiency of our proposed approach.

----

## [1150] Interventional Sum-Product Networks: Causal Inference with Tractable Probabilistic Models

**Authors**: *Matej Zecevic, Devendra Singh Dhami, Athresh Karanam, Sriraam Natarajan, Kristian Kersting*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7eb7eabbe9bd03c2fc99881d04da9cbd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7eb7eabbe9bd03c2fc99881d04da9cbd-Abstract.html)

**Abstract**:

While probabilistic models are an important tool for studying causality, doing so suffers from the intractability of inference. As a step towards tractable causal models, we consider the problem of learning interventional distributions using sum-product networks (SPNs) that are over-parameterized by gate functions, e.g., neural networks. Providing an arbitrarily intervened causal graph as input, effectively subsuming Pearl's do-operator, the gate function predicts the parameters of the SPN. The resulting interventional SPNs are motivated and illustrated by a structural causal model themed around personal health. Our empirical evaluation against competing methods from both generative and causal modelling demonstrates that interventional SPNs indeed are both expressive and causally adequate.

----

## [1151] PettingZoo: Gym for Multi-Agent Reinforcement Learning

**Authors**: *Justin K. Terry, Benjamin Black, Nathaniel Grammel, Mario Jayakumar, Ananth Hari, Ryan Sullivan, Luis S. Santos, Clemens Dieffendahl, Caroline Horsch, Rodrigo Perez-Vicente, Niall L. Williams, Yashas Lokesh, Praveen Ravi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7ed2d3454c5eea71148b11d0c25104ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ed2d3454c5eea71148b11d0c25104ff-Abstract.html)

**Abstract**:

This paper introduces the PettingZoo library and the accompanying Agent Environment Cycle ("AEC") games model. PettingZoo is a library of diverse sets of multi-agent environments with a universal, elegant Python API. PettingZoo was developed with the goal of accelerating research in Multi-Agent Reinforcement Learning ("MARL"), by making work more interchangeable, accessible and reproducible akin to what OpenAI's Gym library did for single-agent reinforcement learning. PettingZoo's API, while inheriting many features of Gym, is unique amongst MARL APIs in that it's based around the novel AEC games model. We argue, in part through case studies on major problems in popular MARL environments, that the popular game models are poor conceptual models of the games commonly used with MARL, that they promote severe bugs that are hard to detect, and that the AEC games model addresses these problems.

----

## [1152] Parametric Complexity Bounds for Approximating PDEs with Neural Networks

**Authors**: *Tanya Marwah, Zachary C. Lipton, Andrej Risteski*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7edccc661418aeb5761dbcdc06ad490c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7edccc661418aeb5761dbcdc06ad490c-Abstract.html)

**Abstract**:

Recent experiments have shown that deep networks can approximate solutions to high-dimensional PDEs, seemingly escaping the curse of dimensionality. However, questions regarding the theoretical basis for such approximations, including the required network size remain open. In this paper, we investigate the representational power of neural networks for approximating solutions to linear elliptic PDEs with Dirichlet boundary conditions. We prove that when a PDE's coefficients are representable by small neural networks, the parameters required to approximate its solution scale polynomially with the input dimension $d$ and proportionally to the parameter counts of the coefficient networks. To this end, we develop a proof technique that simulates gradient descent (in an appropriate Hilbert space) by growing a neural network architecture whose iterates each participate as sub-networks in their (slightly larger) successors, and converge to the solution of the PDE.

----

## [1153] Learning-to-learn non-convex piecewise-Lipschitz functions

**Authors**: *Maria-Florina Balcan, Mikhail Khodak, Dravyansh Sharma, Ameet Talwalkar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7ee6f2b3b68a212d3b7a4f6557eb8cc7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ee6f2b3b68a212d3b7a4f6557eb8cc7-Abstract.html)

**Abstract**:

We analyze the meta-learning of the initialization and step-size of learning algorithms for piecewise-Lipschitz functions, a non-convex setting with applications to both machine learning and algorithms. Starting from recent regret bounds for the exponential forecaster on losses with dispersed discontinuities, we generalize them to be initialization-dependent and then use this result to propose a practical meta-learning procedure that learns both the initialization and the step-size of the algorithm from multiple online learning tasks. Asymptotically, we guarantee that the average regret across tasks scales with a natural notion of task-similarity that measures the amount of overlap between near-optimal regions of different tasks. Finally, we instantiate the method and its guarantee in two important settings: robust meta-learning and multi-task data-driven algorithm design.

----

## [1154] Uncertain Decisions Facilitate Better Preference Learning

**Authors**: *Cassidy Laidlaw, Stuart Russell*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7f141cf8e7136ce8701dc6636c2a6fe4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7f141cf8e7136ce8701dc6636c2a6fe4-Abstract.html)

**Abstract**:

Existing observational approaches for learning human preferences, such as inverse reinforcement learning, usually make strong assumptions about the observability of the human's environment. However, in reality, people make many important decisions under uncertainty. To better understand preference learning in these cases, we study the setting of inverse decision theory (IDT), a previously proposed framework where a human is observed making non-sequential binary decisions under uncertainty. In IDT, the human's preferences are conveyed through their loss function, which expresses a tradeoff between different types of mistakes. We give the first statistical analysis of IDT, providing conditions necessary to identify these preferences and characterizing the sample complexityâ€”the number of decisions that must be observed to learn the tradeoff the human is making to a desired precision. Interestingly, we show that it is actually easier to identify preferences when the decision problem is more uncertain. Furthermore, uncertain decision problems allow us to relax the unrealistic assumption that the human is an optimal decision maker but still identify their exact preferences; we give sample complexities in this suboptimal case as well. Our analysis contradicts the intuition that partial observability should make preference learning more difficult. It also provides a first step towards understanding and improving preference learning methods for uncertain and suboptimal humans.

----

## [1155] Decision Transformer: Reinforcement Learning via Sequence Modeling

**Authors**: *Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html)

**Abstract**:

We introduce a framework that abstracts Reinforcement Learning (RL) as a sequence modeling problem. This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that casts the problem of RL as conditional sequence modeling. Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer. By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks.

----

## [1156] Probability Paths and the Structure of Predictions over Time

**Authors**: *Zhiyuan (Jerry) Lin, Hao Sheng, Sharad Goel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7f53f8c6c730af6aeb52e66eb74d8507-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7f53f8c6c730af6aeb52e66eb74d8507-Abstract.html)

**Abstract**:

In settings ranging from weather forecasts to political prognostications to financial projections, probability estimates of future binary outcomes often evolve over time. For example, the estimated likelihood of rain on a specific day changes by the hour as new information becomes available. Given a collection of such probability paths, we introduce a Bayesian framework -- which we call the Gaussian latent information martingale, or GLIM -- for modeling the structure of dynamic predictions over time. Suppose, for example, that the likelihood of rain in a week is 50%, and consider two hypothetical scenarios. In the first, one expects the forecast to be equally likely to become either 25% or 75% tomorrow; in the second, one expects the forecast to stay constant for the next several days. A time-sensitive decision-maker might select a course of action immediately in the latter scenario, but may postpone their decision in the former, knowing that new information is imminent. We model these trajectories by assuming predictions update according to a latent process of information flow, which is inferred from historical data. In contrast to general methods for time series analysis, this approach preserves important properties of probability paths such as the martingale structure and appropriate amount of volatility and better quantifies future uncertainties around probability paths. We show that GLIM outperforms three popular baseline methods, producing better estimated posterior probability path distributions measured by three different metrics. By elucidating the dynamic structure of predictions over time, we hope to help individuals make more informed choices.

----

## [1157] Deep Extended Hazard Models for Survival Analysis

**Authors**: *Qixian Zhong, Jonas Mueller, Jane-Ling Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7f6caf1f0ba788cd7953d817724c2b6e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7f6caf1f0ba788cd7953d817724c2b6e-Abstract.html)

**Abstract**:

Unlike standard prediction tasks, survival analysis requires modeling right censored data, which must be treated with care. While deep neural networks excel in traditional supervised learning, it remains unclear how to best utilize these models in survival analysis. A key question asks which data-generating assumptions of traditional survival models should be retained and which should be made more flexible via the function-approximating capabilities of neural networks. Rather than estimating the survival function targeted by most existing methods, we introduce a Deep Extended Hazard (DeepEH) model to provide a flexible and general framework for deep survival analysis. The extended hazard model includes the conventional Cox proportional hazards  and accelerated failure time models as special cases, so DeepEH subsumes the popular Deep Cox proportional hazard (DeepSurv) and Deep Accelerated Failure Time (DeepAFT) models. We additionally provide theoretical support for the proposed DeepEH model by establishing consistency and convergence rate of the survival function estimator, which underscore the attractive feature that deep learning is able to detect low-dimensional structure of data in high-dimensional space. Numerical experiments also provide evidence that the proposed methods outperform existing statistical and deep learning approaches to survival analysis.

----

## [1158] TNASP: A Transformer-based NAS Predictor with a Self-evolution Framework

**Authors**: *Shun Lu, Jixiang Li, Jianchao Tan, Sen Yang, Ji Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7fa1575cbd7027c9a799983a485c3c2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7fa1575cbd7027c9a799983a485c3c2f-Abstract.html)

**Abstract**:

Predictor-based Neural Architecture Search (NAS) continues to be an important topic because it aims to mitigate the time-consuming search procedure of traditional NAS methods. A promising performance predictor determines the quality of final searched models in predictor-based NAS methods. Most existing predictor-based methodologies train model-based predictors under a proxy dataset setting, which may suffer from the accuracy decline and the generalization problem, mainly due to their poor abilities to represent spatial topology information of the graph structure data. Besides the poor encoding for spatial topology information, these works did not take advantage of the temporal information such as historical evaluations during training. Thus, we propose a Transformer-based NAS performance predictor, associated with a Laplacian matrix based positional encoding strategy, which better represents topology information and achieves better performance than previous state-of-the-art methods on NAS-Bench-101, NAS-Bench-201, and DARTS search space. Furthermore, we also propose a self-evolution framework that can fully utilize temporal information as guidance. This framework iteratively involves the evaluations of previously predicted results as constraints into current optimization iteration, thus further improving the performance of our predictor. Such framework is model-agnostic, thus can enhance performance on various backbone structures for the prediction task. Our proposed method helped us rank 2nd among all teams in CVPR 2021 NAS Competition Track 2: Performance Prediction Track.

----

## [1159] Automorphic Equivalence-aware Graph Neural Network

**Authors**: *Fengli Xu, Quanming Yao, Pan Hui, Yong Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/7ffb4e0ece07869880d51662a2234143-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/7ffb4e0ece07869880d51662a2234143-Abstract.html)

**Abstract**:

Distinguishing the automorphic equivalence of nodes in a graph plays an essential role in many scientific domains, e.g., computational biologist and social network analysis. However, existing graph neural networks (GNNs) fail to capture such an important property. To make GNN aware of automorphic equivalence, we first introduce a localized variant of this concept --- ego-centered automorphic equivalence (Ego-AE). Then, we design a novel variant of GNN, i.e., GRAPE, that uses learnable AE-aware aggregators to explicitly differentiate the Ego-AE of each node's neighbors with the aids of various subgraph templates. While the design of subgraph templates can be hard, we further propose a genetic algorithm to automatically search them from graph data. Moreover, we theoretically prove that GRAPE is expressive in terms of generating distinct representations for nodes with different Ego-AE features, which fills in a fundamental gap of existing GNN variants. Finally, we empirically validate our model on eight real-world graph data, including social network, e-commerce co-purchase network, and citation network, and show that it consistently outperforms existing GNNs. The source code is public available at https://github.com/tsinghua-fib-lab/GRAPE.

----

## [1160] Random Shuffling Beats SGD Only After Many Epochs on Ill-Conditioned Problems

**Authors**: *Itay Safran, Ohad Shamir*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/803ef56843860e4a48fc4cdb3065e8ce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/803ef56843860e4a48fc4cdb3065e8ce-Abstract.html)

**Abstract**:

Recently, there has been much interest in studying the convergence rates of without-replacement SGD, and proving that it is faster than with-replacement SGD in the worst case. However, known lower bounds ignore the problem's geometry, including its condition number, whereas the upper bounds explicitly depend on it. Perhaps surprisingly, we prove that when the condition number is taken into account, without-replacement SGD \emph{does not} significantly improve on with-replacement SGD in terms of worst-case bounds, unless the number of epochs (passes over the data) is larger than the condition number. Since many problems in machine learning and other areas are both ill-conditioned and involve large datasets, this indicates that without-replacement does not necessarily improve over with-replacement sampling for realistic iteration budgets. We show this by providing new lower and upper bounds which are tight (up to log factors), for quadratic problems with commuting quadratic terms, precisely quantifying the dependence on the problem parameters.

----

## [1161] Analytic Study of Families of Spurious Minima in Two-Layer ReLU Neural Networks: A Tale of Symmetry II

**Authors**: *Yossi Arjevani, Michael Field*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/806d926414ce19d907700e23177ab4ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/806d926414ce19d907700e23177ab4ff-Abstract.html)

**Abstract**:

We study the optimization problem associated with fitting two-layer ReLU neural networks with respect to the squared loss, where labels are generated by a target network. We make use of the rich symmetry structure to develop a novel set of tools for studying families of spurious minima. In contrast to existing approaches which operate in limiting regimes, our technique directly addresses the nonconvex loss landscape for finite number of inputs $d$ and neurons $k$, and provides analytic, rather than heuristic, information. In particular, we derive analytic estimates for the loss at different minima, and prove that, modulo $O(d^{-1/2})$-terms, the Hessian spectrum concentrates near small positive constants, with the exception of $\Theta(d)$ eigenvalues which grow linearly with~$d$. We further show that the Hessian spectrum at global and spurious minima coincide to $O(d^{-1/2})$-order, thus challenging our ability to argue about statistical generalization through local curvature. Lastly, our technique provides the exact \emph{fractional} dimensionality at which families of critical points turn from saddles into spurious minima. This makes possible the study of the creation and the annihilation of spurious minima using powerful tools from equivariant bifurcation theory.

----

## [1162] CAM-GAN: Continual Adaptation Modules for Generative Adversarial Networks

**Authors**: *Sakshi Varshney, Vinay Kumar Verma, P. K. Srijith, Lawrence Carin, Piyush Rai*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8073bd4ed0fe0c330290c58056a2cd5e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8073bd4ed0fe0c330290c58056a2cd5e-Abstract.html)

**Abstract**:

We present a continual learning approach for generative adversarial networks (GANs), by designing and leveraging parameter-efficient feature map transformations. Our approach is based on learning a set of global and task-specific parameters. The global parameters are fixed across tasks whereas the task-specific parameters act as local adapters for each task, and help in efficiently obtaining task-specific feature maps. Moreover, we propose an element-wise addition of residual bias in the transformed feature space, which further helps stabilize GAN training in such settings. Our approach also leverages task similarities based on the Fisher information matrix. Leveraging this knowledge from previous tasks significantly improves the model performance. In addition, the similarity measure also helps reduce the parameter growth in continual adaptation and helps to learn a compact model. In contrast to the recent approaches for continually-learned GANs, the proposed approach provides a memory-efficient way to perform effective continual data generation. Through extensive experiments on challenging and diverse datasets, we show that the feature-map-transformation approach outperforms state-of-the-art methods for continually-learned GANs, with substantially fewer parameters. The proposed method generates high-quality samples that can also improve the generative-replay-based continual learning for discriminative tasks.

----

## [1163] Structured Dropout Variational Inference for Bayesian Neural Networks

**Authors**: *Son Nguyen, Duong Nguyen, Khai Nguyen, Khoat Than, Hung Bui, Nhat Ho*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/80a160ff31266be2f93012a2a3eca713-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/80a160ff31266be2f93012a2a3eca713-Abstract.html)

**Abstract**:

Approximate inference in Bayesian deep networks exhibits a dilemma of how to yield high fidelity posterior approximations while maintaining computational efficiency and scalability. We tackle this challenge by introducing a novel variational structured approximation inspired by the Bayesian interpretation of Dropout regularization. Concretely, we focus on the inflexibility of the factorized structure in Dropout posterior and then propose an improved method called Variational Structured Dropout (VSD). VSD employs an orthogonal transformation to learn a structured representation on the variational Gaussian noise with plausible complexity, and consequently induces statistical dependencies in the approximate posterior. Theoretically, VSD successfully addresses the pathologies of previous Variational Dropout methods and thus offers a standard Bayesian justification. We further show that VSD induces an adaptive regularization term with several desirable properties which contribute to better generalization. Finally, we conduct extensive experiments on standard benchmarks to demonstrate the effectiveness of VSD over state-of-the-art variational methods on predictive accuracy, uncertainty estimation, and out-of-distribution detection.

----

## [1164] Neural Relightable Participating Media Rendering

**Authors**: *Quan Zheng, Gurprit Singh, Hans-Peter Seidel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/80f24ef493982c552b6943f1411f7e2c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/80f24ef493982c552b6943f1411f7e2c-Abstract.html)

**Abstract**:

Learning neural radiance fields of a scene has recently allowed realistic novel view synthesis of the scene, but they are limited to synthesize images under the original fixed lighting condition. Therefore, they are not flexible for the eagerly desired tasks like relighting, scene editing and scene composition. To tackle this problem, several recent methods propose to disentangle reflectance and illumination from the radiance field. These methods can cope with solid objects with opaque surfaces but participating media are neglected. Also, they take into account only direct illumination or at most one-bounce indirect illumination, thus suffer from energy loss due to ignoring the high-order indirect illumination. We propose to learn neural representations for participating media with a complete simulation of global illumination. We estimate direct illumination via ray tracing and compute indirect illumination with spherical harmonics. Our approach avoids computing the lengthy indirect bounces and does not suffer from energy loss. Our experiments on multiple scenes show that our approach achieves superior visual quality and numerical performance compared to state-of-the-art methods, and it can generalize to deal with solid objects with opaque surfaces as well.

----

## [1165] Efficient Neural Network Training via Forward and Backward Propagation Sparsification

**Authors**: *Xiao Zhou, Weizhong Zhang, Zonghao Chen, Shizhe Diao, Tong Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/80f2f15983422987ea30d77bb531be86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/80f2f15983422987ea30d77bb531be86-Abstract.html)

**Abstract**:

Sparse training is a natural idea to accelerate the training speed of deep neural networks and save the memory usage, especially since large modern neural networks are significantly over-parameterized.  However, most of the existing methods cannot achieve this goal in practice because the chain rule based gradient (w.r.t. structure parameters) estimators adopted by previous methods require dense computation at least in the backward propagation step.  This paper solves this problem by proposing an efficient sparse training method with completely sparse forward and backward passes. We first formulate the training process as a continuous minimization problem under global sparsity constraint. We then separate the optimization process into two steps, corresponding to weight update and structure parameter update. For the former step, we use the conventional chain rule, which can be sparse via exploiting the sparse structure.  For the latter step, instead of using the chain rule based gradient estimators as in existing methods, we propose a variance reduced policy gradient estimator, which only requires two forward passes without backward propagation, thus achieving completely sparse training. We prove that the variance of our gradient estimator is bounded. Extensive experimental results on real-world datasets demonstrate that compared to previous methods, our algorithm is much more effective in accelerating the training process, up to an order of magnitude faster.

----

## [1166] Learning to Ground Multi-Agent Communication with Autoencoders

**Authors**: *Toru Lin, Jacob Huh, Christopher Stauffer, Ser-Nam Lim, Phillip Isola*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/80fee67c8a4c4989bf8a580b4bbb0cd2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/80fee67c8a4c4989bf8a580b4bbb0cd2-Abstract.html)

**Abstract**:

Communication requires having a common language, a lingua franca, between agents. This language could emerge via a consensus process, but it may require many generations of trial and error. Alternatively, the lingua franca can be given by the environment, where agents ground their language in representations of the observed world. We demonstrate a simple way to ground language in learned representations, which facilitates decentralized multi-agent communication and coordination. We find that a standard representation learning algorithm -- autoencoding -- is sufficient for arriving at a grounded common language. When agents broadcast these representations, they learn to understand and respond to each other's utterances and achieve surprisingly strong task performance across a variety of multi-agent communication environments.

----

## [1167] Large-Scale Wasserstein Gradient Flows

**Authors**: *Petr Mokrov, Alexander Korotin, Lingxiao Li, Aude Genevay, Justin M. Solomon, Evgeny Burnaev*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/810dfbbebb17302018ae903e9cb7a483-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/810dfbbebb17302018ae903e9cb7a483-Abstract.html)

**Abstract**:

Wasserstein gradient flows provide a powerful means of understanding and solving many diffusion equations. Specifically, Fokker-Planck equations, which model the diffusion of probability measures, can be understood as gradient descent over entropy functionals in Wasserstein space. This equivalence, introduced by Jordan, Kinderlehrer and Otto, inspired the so-called JKO scheme to approximate these diffusion processes via an implicit discretization of the gradient flow in Wasserstein space. Solving the optimization problem associated with each JKO step, however, presents serious computational challenges. We introduce a scalable method to approximate Wasserstein gradient flows, targeted to machine learning applications. Our approach relies on input-convex neural networks (ICNNs) to discretize the JKO steps, which can be optimized by stochastic gradient descent. Contrarily to previous work, our method does not require domain discretization or particle simulation.  As a result, we can sample from the measure at each time step of the diffusion and compute its probability density. We demonstrate the performance of our algorithm by computing diffusions following the Fokker-Planck equation and apply it to unnormalized density sampling as well as nonlinear filtering.

----

## [1168] Who Leads and Who Follows in Strategic Classification?

**Authors**: *Tijana Zrnic, Eric Mazumdar, S. Shankar Sastry, Michael I. Jordan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/812214fb8e7066bfa6e32c626c2c688b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/812214fb8e7066bfa6e32c626c2c688b-Abstract.html)

**Abstract**:

As predictive models are deployed into the real world, they must increasingly contend with strategic behavior. A growing body of work on strategic classification treats this problem as a Stackelberg game: the decision-maker "leads" in the game by deploying a model, and the strategic agents "follow" by playing their best response to the deployed model. Importantly, in this framing, the burden of learning is placed solely on the decision-maker, while the agents’ best responses are implicitly treated as instantaneous. In this work, we argue that the order of play in strategic classification is fundamentally determined by the relative frequencies at which the decision-maker and the agents adapt to each other’s actions. In particular, by generalizing the standard model to allow both players to learn over time, we show that a decision-maker that makes updates faster than the agents can reverse the order of play, meaning that the agents lead and the decision-maker follows. We observe in standard learning settings that such a role reversal can be desirable for both the decision-maker and the strategic agents. Finally, we show that a decision-maker with the freedom to choose their update frequency can induce learning dynamics that converge to Stackelberg equilibria with either order of play.

----

## [1169] Unadversarial Examples: Designing Objects for Robust Vision

**Authors**: *Hadi Salman, Andrew Ilyas, Logan Engstrom, Sai Vemprala, Aleksander Madry, Ashish Kapoor*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/816a6db41f0e44644bc65808b6db5ca4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/816a6db41f0e44644bc65808b6db5ca4-Abstract.html)

**Abstract**:

We study a class of computer vision settings wherein one can modify the design of the objects being recognized. We develop a framework that leverages this capability---and deep networks' unusual sensitivity to input perturbations---to design ``robust objects,'' i.e., objects that are explicitly optimized to be confidently classified. Our framework yields improved performance on standard benchmarks, a simulated robotics environment, and physical-world experiments.

----

## [1170] Deep Jump Learning for Off-Policy Evaluation in Continuous Treatment Settings

**Authors**: *Hengrui Cai, Chengchun Shi, Rui Song, Wenbin Lu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/816b112c6105b3ebd537828a39af4818-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/816b112c6105b3ebd537828a39af4818-Abstract.html)

**Abstract**:

We consider off-policy evaluation (OPE) in continuous treatment settings, such as personalized dose-finding. In OPE, one aims to estimate the mean outcome under a new treatment decision rule using historical data generated by a different decision rule. Most existing works on OPE focus on discrete treatment settings. To handle continuous treatments, we develop a novel estimation method for OPE using deep jump learning. The key ingredient of our method lies in adaptively discretizing the treatment space using deep discretization, by leveraging deep learning and multi-scale change point detection. This allows us to apply existing OPE methods in discrete treatments to handle continuous treatments. Our method is further justified by theoretical results, simulations, and a real application to Warfarin Dosing.

----

## [1171] Attention Approximates Sparse Distributed Memory

**Authors**: *Trenton Bricken, Cengiz Pehlevan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8171ac2c5544a5cb54ac0f38bf477af4-Abstract.html)

**Abstract**:

While Attention has come to be an important mechanism in deep learning, there remains limited intuition for why it works so well. Here, we show that Transformer Attention can be closely related under certain data conditions to Kanerva's Sparse Distributed Memory (SDM), a biologically plausible associative memory model. We confirm that these conditions are satisfied in pre-trained GPT2 Transformer models. We discuss the implications of the Attention-SDM map and provide new computational and biological interpretations of Attention.

----

## [1172] Augmented Shortcuts for Vision Transformers

**Authors**: *Yehui Tang, Kai Han, Chang Xu, An Xiao, Yiping Deng, Chao Xu, Yunhe Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/818f4654ed39a1c147d1e51a00ffb4cb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/818f4654ed39a1c147d1e51a00ffb4cb-Abstract.html)

**Abstract**:

Transformer models have achieved great progress on computer vision tasks recently. The rapid development of vision transformers is mainly contributed by their high representation ability for extracting informative features from input images. However, the mainstream transformer models are designed with deep architectures, and the feature diversity will be continuously reduced as the depth increases, \ie, feature collapse. In this paper, we theoretically analyze the feature collapse phenomenon and study the relationship between shortcuts and feature diversity in these transformer models. Then, we present an augmented shortcut scheme, which inserts additional paths with learnable parameters in parallel on the original shortcuts. To save the computational costs, we further explore an efficient approach that uses the block-circulant projection to implement augmented shortcuts. Extensive experiments conducted on benchmark datasets demonstrate the effectiveness of the proposed method, which brings about 1% accuracy increase of the state-of-the-art visual transformers without obviously increasing their parameters and FLOPs.

----

## [1173] Finding Regions of Heterogeneity in Decision-Making via Expected Conditional Covariance

**Authors**: *Justin Lim, Christina X. Ji, Michael Oberst, Saul Blecker, Leora Horwitz, David A. Sontag*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/81930c54e08b6d26d9638dd2e4656dc1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/81930c54e08b6d26d9638dd2e4656dc1-Abstract.html)

**Abstract**:

Individuals often make different decisions when faced with the same context, due to personal preferences and background.  For instance, judges may vary in their leniency towards certain drug-related offenses, and doctors may vary in their preference for how to start treatment for certain types of patients.  With these examples in mind, we present an algorithm for identifying types of contexts (e.g., types of cases or patients) with high inter-decision-maker disagreement.  We formalize this as a causal inference problem, seeking a region where the assignment of decision-maker has a large causal effect on the decision.  Our algorithm finds such a region by maximizing an empirical objective, and we give a generalization bound for its performance. In a semi-synthetic experiment, we show that our algorithm recovers the correct region of heterogeneity accurately compared to baselines. Finally, we apply our algorithm to real-world healthcare datasets, recovering variation that aligns with existing clinical knowledge.

----

## [1174] Identifying and Benchmarking Natural Out-of-Context Prediction Problems

**Authors**: *David Madras, Richard S. Zemel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/81c2f886f91e18fe16d6f4e865877cb6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/81c2f886f91e18fe16d6f4e865877cb6-Abstract.html)

**Abstract**:

Deep learning systems frequently fail at out-of-context (OOC) prediction, the problem of making reliable predictions on uncommon or unusual inputs or subgroups of the training distribution. To this end, a number of benchmarks for measuring OOC performance have been recently introduced.  In this work, we introduce a framework unifying the literature on OOC performance measurement, and demonstrate how rich auxiliary information can be leveraged to identify candidate sets of OOC examples in existing datasets.  We present NOOCh: a suite of naturally-occurring "challenge sets", and show how varying notions of context can be used to probe specific OOC failure modes. Experimentally, we explore the tradeoffs between various learning approaches on these challenge sets and demonstrate how the choices made in designing OOC benchmarks can yield varying conclusions.

----

## [1175] Label Disentanglement in Partition-based Extreme Multilabel Classification

**Authors**: *Xuanqing Liu, Wei-Cheng Chang, Hsiang-Fu Yu, Cho-Jui Hsieh, Inderjit S. Dhillon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/81c8727c62e800be708dbf37c4695dff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/81c8727c62e800be708dbf37c4695dff-Abstract.html)

**Abstract**:

Partition-based methods are increasingly-used in extreme multi-label classification (XMC) problems due to their scalability to large output spaces (e.g., millions or more). However, existing methods partition the large label space into mutually exclusive clusters, which is sub-optimal when labels have multi-modality and rich semantics. For instance, the label “Apple” can be the fruit or the brand name, which leads to the following research question: can we disentangle these multi-modal labels with non-exclusive clustering tailored for downstream XMC tasks? In this paper, we show that the label assignment problem in partition-based XMC can be formulated as an optimization problem, with the objective of maximizing precision rates. This leads to an efficient algorithm to form  flexible and overlapped label clusters, and a method that can alternatively optimizes the cluster assignments and the model parameters for partition-based XMC. Experimental results on synthetic and real datasets show that our method can successfully disentangle multi-modal labels, leading to state-of-the-art (SOTA) results on four XMC benchmarks.

----

## [1176] Leveraging SE(3) Equivariance for Self-supervised Category-Level Object Pose Estimation from Point Clouds

**Authors**: *Xiaolong Li, Yijia Weng, Li Yi, Leonidas J. Guibas, A. Lynn Abbott, Shuran Song, He Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/81e74d678581a3bb7a720b019f4f1a93-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/81e74d678581a3bb7a720b019f4f1a93-Abstract.html)

**Abstract**:

Category-level object pose estimation aims to find 6D object poses of previously unseen object instances from known categories without access to object CAD models. To reduce the huge amount of pose annotations needed for category-level learning, we propose for the first time a self-supervised learning framework to estimate category-level 6D object pose from single 3D point clouds. During training, our method assumes no ground-truth pose annotations, no CAD models, and no multi-view supervision. The key to our method is to disentangle shape and pose through an invariant shape reconstruction module and an equivariant pose estimation module, empowered by SE(3) equivariant point cloud networks. The invariant shape reconstruction module learns to perform aligned reconstructions, yielding a category-level reference frame without using any annotations. In addition, the equivariant pose estimation module achieves category-level pose estimation accuracy that is comparable to some fully supervised methods. Extensive experiments demonstrate the effectiveness of our approach on both complete and partial depth point clouds from the ModelNet40 benchmark, and on real depth point clouds from the NOCS-REAL 275 dataset. The project page with code and visualizations can be found at: dragonlong.github.io/equi-pose.

----

## [1177] A Theoretical Analysis of Fine-tuning with Linear Teachers

**Authors**: *Gal Shachaf, Alon Brutzkus, Amir Globerson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html)

**Abstract**:

Fine-tuning is a common practice in deep learning, achieving excellent generalization results on downstream tasks using relatively little training data. Although widely used in practice, it is not well understood theoretically. Here we analyze the sample complexity of this scheme for regression with linear teachers in several settings. Intuitively, the success of fine-tuning depends on the similarity between the source tasks and the target task. But what is the right way of measuring this similarity? We show that the relevant measure has to do with the relation between the source task, the target task and the covariance structure of the target data. In the setting of linear regression, we show that under realistic settings there can be substantial sample complexity reduction when the above measure is low. For deep linear regression, we propose a novel result regarding the inductive bias of gradient-based training when the network is initialized with pretrained weights. Using this result we show that the similarity measure for this setting is also affected by the depth of the network. We conclude with results on shallow ReLU models, and analyze the dependence of sample complexity there on source and target tasks. We empirically demonstrate our results for both synthetic and realistic data.

----

## [1178] Overinterpretation reveals image classification model pathologies

**Authors**: *Brandon Carter, Siddhartha Jain, Jonas Mueller, David Gifford*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8217bb4e7fa0541e0f5e04fea764ab91-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8217bb4e7fa0541e0f5e04fea764ab91-Abstract.html)

**Abstract**:

Image classifiers are typically scored on their test set accuracy, but high accuracy can mask a subtle type of model failure. We find that high scoring convolutional neural networks (CNNs) on popular benchmarks exhibit troubling pathologies that allow them to display high accuracy even in the absence of semantically salient features. When a model provides a high-confidence decision without salient supporting input features, we say the classifier has overinterpreted its input, finding too much class-evidence in patterns that appear nonsensical to humans. Here, we demonstrate that neural networks trained on CIFAR-10 and ImageNet suffer from overinterpretation, and we find models on CIFAR-10 make confident predictions even when 95% of input images are masked and humans cannot discern salient features in the remaining pixel-subsets. We introduce Batched Gradient SIS, a new method for discovering sufficient input subsets for complex datasets, and use this method to show the sufficiency of border pixels in ImageNet for training and testing. Although these patterns portend potential model fragility in real-world deployment, they are in fact valid statistical patterns of the benchmark that alone suffice to attain high test accuracy. Unlike adversarial examples, overinterpretation relies upon unmodified image pixels.  We find ensembling and input dropout can each help mitigate overinterpretation.

----

## [1179] Neural Circuit Synthesis from Specification Patterns

**Authors**: *Frederik Schmitt, Christopher Hahn, Markus N. Rabe, Bernd Finkbeiner*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8230bea7d54bcdf99cdfe85cb07313d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8230bea7d54bcdf99cdfe85cb07313d5-Abstract.html)

**Abstract**:

We train hierarchical Transformers on the task of synthesizing hardware circuits directly out of high-level logical speciﬁcations in linear-time temporal logic (LTL). The LTL synthesis problem is a well-known algorithmic challenge with a long history and an annual competition is organized to track the improvement of algorithms and tooling over time. New approaches using machine learning might open a lot of possibilities in this area, but suffer from the lack of sufﬁcient amounts of training data. In this paper, we consider a method to generate large amounts of additional training data, i.e., pairs of speciﬁcations and circuits implementing them. We ensure that this synthetic data is sufﬁciently close to human-written speciﬁcations by mining common patterns from the speciﬁcations used in the synthesis competitions. We show that hierarchical Transformers trained on this synthetic data solve a signiﬁcant portion of problems from the synthesis competitions, and even out-of-distribution examples from a recent case study.

----

## [1180] Directional Message Passing on Molecular Graphs via Synthetic Coordinates

**Authors**: *Johannes Gasteiger, Chandan Yeshwanth, Stephan Günnemann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82489c9737cc245530c7a6ebef3753ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82489c9737cc245530c7a6ebef3753ec-Abstract.html)

**Abstract**:

Graph neural networks that leverage coordinates via directional message passing have recently set the state of the art on multiple molecular property prediction tasks. However, they rely on atom position information that is often unavailable, and obtaining it is usually prohibitively expensive or even impossible. In this paper we propose synthetic coordinates that enable the use of advanced GNNs without requiring the true molecular configuration. We propose two distances as synthetic coordinates: Distance bounds that specify the rough range of molecular configurations, and graph-based distances using a symmetric variant of personalized PageRank. To leverage both distance and angular information we propose a method of transforming normal graph neural networks into directional MPNNs. We show that with this transformation we can reduce the error of a normal graph neural network by 55% on the ZINC benchmark. We furthermore set the state of the art on ZINC and coordinate-free QM9 by incorporating synthetic coordinates in the SMP and DimeNet++ models. Our implementation is available online.

----

## [1181] Federated Multi-Task Learning under a Mixture of Distributions

**Authors**: *Othmane Marfoq, Giovanni Neglia, Aurélien Bellet, Laetitia Kameni, Richard Vidal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82599a4ec94aca066873c99b4c741ed8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82599a4ec94aca066873c99b4c741ed8-Abstract.html)

**Abstract**:

The increasing size of data generated by smartphones and IoT devices motivated the development of Federated Learning (FL), a framework for on-device collaborative training of machine learning models. First efforts in FL focused on learning a single global model with good average performance across clients, but the global model may be arbitrarily bad for a given client, due to the inherent heterogeneity of local data distributions. Federated multi-task learning (MTL) approaches can learn personalized models by formulating an opportune penalized optimization problem. The penalization term can capture complex relations among personalized models, but eschews clear statistical assumptions about local data distributions. In this work, we propose to study federated MTL under the flexible assumption that each local data distribution is a mixture of unknown underlying distributions. This assumption encompasses most of the existing personalized FL approaches and leads to federated EM-like algorithms for both client-server and fully decentralized settings. Moreover, it  provides a principled way to serve personalized models to clients not seen at training time. The algorithms' convergence is analyzed through a novel federated surrogate optimization framework, which can be of general interest. Experimental results on FL benchmarks show that our approach provides models with higher accuracy and fairness than state-of-the-art methods.

----

## [1182] Learning Generative Vision Transformer with Energy-Based Latent Space for Saliency Prediction

**Authors**: *Jing Zhang, Jianwen Xie, Nick Barnes, Ping Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8289889263db4a40463e3f358bb7c7a1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8289889263db4a40463e3f358bb7c7a1-Abstract.html)

**Abstract**:

Vision transformer networks have shown superiority in many computer vision tasks. In this paper, we take a step further by proposing a novel generative vision transformer with latent variables following an informative energy-based prior for salient object detection. Both the vision transformer network and the energy-based prior model are jointly trained via Markov chain Monte Carlo-based maximum likelihood estimation, in which the sampling from the intractable posterior and prior distributions of the latent variables are performed by Langevin dynamics. Further, with the generative vision transformer, we can easily obtain a pixel-wise uncertainty map from an image, which indicates the model confidence in predicting saliency from the image. Different from the existing generative models which define the prior distribution of the latent variables as a simple isotropic Gaussian distribution, our model uses an energy-based informative prior which can be more expressive to capture the latent space of the data. We apply the proposed framework to both RGB and RGB-D salient object detection tasks. Extensive experimental results show that our framework can achieve not only accurate saliency predictions but also meaningful uncertainty maps that are consistent with the human perception.

----

## [1183] Regularization in ResNet with Stochastic Depth

**Authors**: *Soufiane Hayou, Fadhel Ayed*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82ba9d6eee3f026be339bb287651c3d8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82ba9d6eee3f026be339bb287651c3d8-Abstract.html)

**Abstract**:

Regularization plays a major role in modern deep learning. From classic techniques such as L1, L2 penalties to other noise-based methods such as Dropout, regularization often yields better generalization properties by avoiding overfitting. Recently, Stochastic Depth (SD) has emerged as an alternative regularization technique for residual neural networks (ResNets) and has proven to boost the performance of ResNet on many tasks [Huang et al., 2016]. Despite the recent success of SD, little is known about this technique from a theoretical perspective. This paper provides a hybrid analysis combining perturbation analysis and signal propagation to shed light on different regularization effects of SD. Our analysis allows us to derive principled guidelines for choosing the survival rates used for training with SD.

----

## [1184] ResT: An Efficient Transformer for Visual Recognition

**Authors**: *Qinglong Zhang, Yu-Bin Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82c2559140b95ccda9c6ca4a8b981f1e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82c2559140b95ccda9c6ca4a8b981f1e-Abstract.html)

**Abstract**:

This paper presents an efficient multi-scale vision Transformer, called ResT, that capably served as a general-purpose backbone for image recognition. Unlike existing Transformer methods, which employ standard Transformer blocks to tackle raw images with a fixed resolution, our ResT have several advantages: (1) A memory-efficient multi-head self-attention is built, which compresses the memory by a simple depth-wise convolution, and projects the interaction across the attention-heads dimension while keeping the diversity ability of multi-heads; (2) Positional encoding is constructed as spatial attention, which is more flexible and can tackle with input images of arbitrary size without interpolation or fine-tune; (3) Instead of the straightforward tokenization at the beginning of each stage, we design the patch embedding as a stack of overlapping convolution operation with stride on the token map. We comprehensively validate ResT on image classification and downstream tasks. Experimental results show that the proposed ResT can outperform the recently state-of-the-art backbones by a large margin, demonstrating the potential of ResT as strong backbones. The code and models will be made publicly available at https://github.com/wofmanaf/ResT.

----

## [1185] Adversarial Examples for k-Nearest Neighbor Classifiers Based on Higher-Order Voronoi Diagrams

**Authors**: *Chawin Sitawarin, Evgenios M. Kornaropoulos, Dawn Song, David A. Wagner*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82ca5dd156cc926b2992f73c2896f761-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82ca5dd156cc926b2992f73c2896f761-Abstract.html)

**Abstract**:

Adversarial examples are a widely studied phenomenon in machine learning models. While most of the attention has been focused on neural networks, other practical models also suffer from this issue. In this work, we propose an algorithm for evaluating the adversarial robustness of $k$-nearest neighbor classification, i.e., finding a minimum-norm adversarial example. Diverging from previous proposals, we propose the first geometric approach by performing a search that expands outwards from a given input point. On a high level, the search radius expands to the nearby higher-order Voronoi cells until we find a cell that classifies differently from the input point. To scale the algorithm to a large $k$, we introduce approximation steps that find perturbation with smaller norm, compared to the baselines, in a variety of datasets. Furthermore, we analyze the structural properties of a dataset where our approach outperforms the competition.

----

## [1186] Adversarially Robust 3D Point Cloud Recognition Using Self-Supervisions

**Authors**: *Jiachen Sun, Yulong Cao, Christopher B. Choy, Zhiding Yu, Anima Anandkumar, Zhuoqing Morley Mao, Chaowei Xiao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82cadb0649a3af4968404c9f6031b233-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82cadb0649a3af4968404c9f6031b233-Abstract.html)

**Abstract**:

3D point cloud data is increasingly used in safety-critical applications such as autonomous driving. Thus, the robustness of 3D deep learning models against adversarial attacks becomes a major consideration. In this paper, we systematically study the impact of various self-supervised learning proxy tasks on different architectures and threat models for 3D point clouds with adversarial training. Specifically, we study MLP-based (PointNet), convolution-based (DGCNN), and transformer-based (PCT) 3D architectures. Through extensive experimentation, we demonstrate that appropriate applications of self-supervision can significantly enhance the robustness in 3D point cloud recognition, achieving considerable improvements compared to the standard adversarial training baseline. Our analysis reveals that local feature learning is desirable for adversarial robustness in point clouds since it limits the adversarial propagation between the point-level input perturbations and the model's final output. This insight also explains the success of DGCNN and the jigsaw proxy task in achieving stronger 3D adversarial robustness.

----

## [1187] Tuning Mixed Input Hyperparameters on the Fly for Efficient Population Based AutoRL

**Authors**: *Jack Parker-Holder, Vu Nguyen, Shaan Desai, Stephen J. Roberts*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82debd8a12b498e765a11a8e51159440-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82debd8a12b498e765a11a8e51159440-Abstract.html)

**Abstract**:

Despite a series of recent successes in reinforcement learning (RL), many RL algorithms remain sensitive to hyperparameters. As such, there has recently been interest in the field of AutoRL, which seeks to automate design decisions to create more general algorithms. Recent work suggests that population based approaches may be effective AutoRL algorithms, by learning hyperparameter schedules on the fly. In particular, the PB2 algorithm is able to achieve strong performance in RL tasks by formulating online hyperparameter optimization as time varying GP-bandit problem, while also providing theoretical guarantees. However, PB2 is only designed to work for \emph{continuous} hyperparameters, which severely limits its utility in practice. In this paper we introduce a new (provably) efficient hierarchical approach for optimizing \emph{both continuous and categorical} variables, using a new time-varying bandit algorithm specifically designed for the population based training regime. We evaluate our approach on the challenging Procgen benchmark, where we show that explicitly modelling dependence between data augmentation and other hyperparameters improves generalization.

----

## [1188] Neural Algorithmic Reasoners are Implicit Planners

**Authors**: *Andreea Deac, Petar Velickovic, Ognjen Milinkovic, Pierre-Luc Bacon, Jian Tang, Mladen Nikolic*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/82e9e7a12665240d13d0b928be28f230-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/82e9e7a12665240d13d0b928be28f230-Abstract.html)

**Abstract**:

Implicit planning has emerged as an elegant technique for combining learned models of the world with end-to-end model-free reinforcement learning. We study the class of implicit planners inspired by value iteration, an algorithm that is guaranteed to yield perfect policies in fully-specified tabular environments. We find that prior approaches either assume that the environment is provided in such a tabular form---which is highly restrictive---or infer "local neighbourhoods" of states to run value iteration over---for which we discover an algorithmic bottleneck effect. This effect is caused by explicitly running the planning algorithm based on scalar predictions in every state, which can be harmful to data efficiency if such scalars are improperly predicted. We propose eXecuted Latent Value Iteration Networks (XLVINs), which alleviate the above limitations. Our method performs all planning computations in a high-dimensional latent space, breaking the algorithmic bottleneck. It maintains alignment with value iteration by carefully leveraging neural graph-algorithmic reasoning and contrastive self-supervised learning. Across seven low-data settings---including classical control, navigation and Atari---XLVINs provide significant improvements to data efficiency against value iteration-based implicit planners, as well as relevant model-free baselines. Lastly, we empirically verify that XLVINs can closely align with value iteration.

----

## [1189] Self-Supervised Learning with Kernel Dependence Maximization

**Authors**: *Yazhe Li, Roman Pogodin, Danica J. Sutherland, Arthur Gretton*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/83004190b1793d7aa15f8d0d49a13eba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/83004190b1793d7aa15f8d0d49a13eba-Abstract.html)

**Abstract**:

We approach self-supervised learning of image representations from a statistical dependence perspective, proposing Self-Supervised Learning with the Hilbert-Schmidt Independence Criterion (SSL-HSIC). SSL-HSIC maximizes dependence between representations of transformations of an image and the image identity, while minimizing the kernelized variance of those representations. This framework yields a new understanding of InfoNCE, a variational lower bound on the mutual information (MI) between different transformations. While the MI itself is known to have pathologies which can result in learning meaningless representations, its bound is much better behaved: we show that it implicitly approximates SSL-HSIC (with a slightly different regularizer).Our approach also gives us insight into BYOL, a negative-free SSL method, since SSL-HSIC similarly learns local neighborhoods of samples. SSL-HSIC allows us to directly optimize statistical dependence in time linear in the batch size, without restrictive data assumptions or indirect mutual information estimators. Trained with or without a target network, SSL-HSIC matches the current state-of-the-art for standard linear evaluation on ImageNet, semi-supervised learning and transfer to other classification and vision tasks such as semantic segmentation, depth estimation and object recognition. Code is available at https://github.com/deepmind/ssl_hsic.

----

## [1190] CROCS: Clustering and Retrieval of Cardiac Signals Based on Patient Disease Class, Sex, and Age

**Authors**: *Dani Kiyasseh, Tingting Zhu, David A. Clifton*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html)

**Abstract**:

The process of manually searching for relevant instances in, and extracting information from, clinical databases underpin a multitude of clinical tasks. Such tasks include disease diagnosis, clinical trial recruitment, and continuing medical education. This manual search-and-extract process, however, has been hampered by the growth of large-scale clinical databases and the increased prevalence of unlabelled instances. To address this challenge, we propose a supervised contrastive learning framework, CROCS, where representations of cardiac signals associated with a set of patient-specific attributes (e.g., disease class, sex, age) are attracted to learnable embeddings entitled clinical prototypes. We exploit such prototypes for both the clustering and retrieval of unlabelled cardiac signals based on multiple patient attributes. We show that CROCS outperforms the state-of-the-art method, DTC, when clustering and also retrieves relevant cardiac signals from a large database. We also show that clinical prototypes adopt a semantically meaningful arrangement based on patient attributes and thus confer a high degree of interpretability.

----

## [1191] Representing Hyperbolic Space Accurately using Multi-Component Floats

**Authors**: *Tao Yu, Christopher De Sa*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/832353270aacb6e3322f493a66aaf5b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/832353270aacb6e3322f493a66aaf5b9-Abstract.html)

**Abstract**:

Hyperbolic space is particularly useful for embedding data with hierarchical structure; however, representing hyperbolic space with ordinary floating-point numbers greatly affects the performance due to its \emph{ineluctable} numerical errors. Simply increasing the precision of floats fails to solve the problem and incurs a high computation cost for simulating greater-than-double-precision floats on hardware such as GPUs, which does not support them. In this paper, we propose a simple, feasible-on-GPUs, and easy-to-understand solution for numerically accurate learning on hyperbolic space. We do this with a new approach to represent hyperbolic space using multi-component floating-point (MCF) in the Poincar{\'e} upper-half space model. Theoretically and experimentally we show our model has small numerical error, and on embedding tasks across various datasets, models represented by multi-component floating-points gain more capacity and run significantly faster on GPUs than prior work.

----

## [1192] Dimensionality Reduction for Wasserstein Barycenter

**Authors**: *Zachary Izzo, Sandeep Silwal, Samson Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8346db44a721fa863ca38180638bad3d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8346db44a721fa863ca38180638bad3d-Abstract.html)

**Abstract**:

The Wasserstein barycenter is a geometric construct which captures the notion of centrality among probability distributions, and which has found many applications in machine learning. However, most algorithms for finding even an approximate barycenter suffer an exponential dependence on the dimension $d$ of the underlying space of the distributions. In order to cope with this ``curse of dimensionality,'' we study dimensionality reduction techniques for the Wasserstein barycenter problem. When the barycenter is restricted to support of size $n$, we show that randomized dimensionality reduction can be used to map the problem to a space of dimension $O(\log n)$ independent of both $d$ and $k$, and that \emph{any} solution found in the reduced dimension will have its cost preserved up to arbitrary small error in the original space. We provide matching upper and lower bounds on the size of the reduced dimension, showing that our methods are optimal up to constant factors. We also provide a coreset construction for the Wasserstein barycenter problem that significantly decreases the number of input distributions. The coresets can be used in conjunction with random projections and thus further improve computation time. Lastly, our experimental results validate the speedup provided by dimensionality reduction while maintaining solution quality.

----

## [1193] Neural Population Geometry Reveals the Role of Stochasticity in Robust Perception

**Authors**: *Joel Dapello, Jenelle Feather, Hang Le, Tiago Marques, David D. Cox, Josh H. McDermott, James J. DiCarlo, SueYeon Chung*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8383f931b0cefcc631f070480ef340e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8383f931b0cefcc631f070480ef340e1-Abstract.html)

**Abstract**:

Adversarial examples are often cited by neuroscientists and machine learning researchers as an example of how computational models diverge from biological sensory systems. Recent work has proposed adding biologically-inspired components to visual neural networks as a way to improve their adversarial robustness. One surprisingly effective component for reducing adversarial vulnerability is response stochasticity, like that exhibited by biological neurons. Here, using recently developed geometrical techniques from computational neuroscience, we investigate how adversarial perturbations influence the internal representations of standard, adversarially trained, and biologically-inspired stochastic networks. We find distinct geometric signatures for each type of network, revealing different mechanisms for achieving robust representations. Next, we generalize these results to the auditory domain, showing that neural stochasticity also makes auditory models more robust to adversarial perturbations. Geometric analysis of the stochastic networks reveals overlap between representations of clean and adversarially perturbed stimuli, and quantitatively demonstrate that competing geometric effects of stochasticity mediate a tradeoff between adversarial and clean performance. Our results shed light on the strategies of robust perception utilized by adversarially trained and stochastic networks, and help explain how stochasticity may be beneficial to machine and biological computation.

----

## [1194] Unsupervised Learning of Compositional Energy Concepts

**Authors**: *Yilun Du, Shuang Li, Yash Sharma, Josh Tenenbaum, Igor Mordatch*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/838aac83e00e8c5ca0f839c96d6cb3be-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/838aac83e00e8c5ca0f839c96d6cb3be-Abstract.html)

**Abstract**:

Humans are able to rapidly understand scenes by utilizing concepts extracted from prior experience. Such concepts are diverse, and include global scene descriptors, such as the weather or lighting, as well as local scene descriptors, such as the color or size of a particular object. So far, unsupervised discovery of concepts has focused on either modeling the global scene-level or the local object-level factors of variation, but not both. In this work, we propose COMET, which discovers and represents concepts as separate energy functions, enabling us to represent both global concepts as well as objects under a unified framework.  COMET discovers energy functions through recomposing the input image, which we find captures independent factors without additional supervision. Sample generation in COMET is formulated as an optimization process on underlying energy functions, enabling us to generate images with permuted and composed concepts.  Finally, discovered visual concepts in COMET generalize well, enabling us to compose concepts between separate modalities of images as well as with other concepts discovered by a separate instance of COMET trained on a different dataset. Code and data available at https://energy-based-model.github.io/comet/.

----

## [1195] Nearly Horizon-Free Offline Reinforcement Learning

**Authors**: *Tongzheng Ren, Jialian Li, Bo Dai, Simon S. Du, Sujay Sanghavi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8396b14c5dff55d13eea57487bf8ed26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8396b14c5dff55d13eea57487bf8ed26-Abstract.html)

**Abstract**:

We revisit offline reinforcement learning on episodic time-homogeneous Markov Decision Processes (MDP). For tabular MDP with $S$ states and $A$ actions, or linear MDP with anchor points and feature dimension $d$, given the collected $K$ episodes data with minimum visiting probability of (anchor) state-action pairs $d_m$, we obtain nearly horizon $H$-free sample complexity bounds for offline reinforcement learning when the total reward is upper bounded by 1. Specifically:• For offline policy evaluation, we obtain an $\tilde{O}\left(\sqrt{\frac{1}{Kd_m}} \right)$ error bound for the plug-in estimator, which matches the lower bound up to logarithmic factors and does not have additional dependency on $\mathrm{poly}(H, S, A, d)$ in higher-order term.• For offline policy optimization, we obtain an $\tilde{O}\left(\sqrt{\frac{1}{Kd_m}} + \frac{\min(S, d)}{Kd_m}\right)$ sub-optimality gap for the empirical optimal policy, which approaches the lower bound up to logarithmic factors and a high-order term, improving upon the best known result by [Cui and Yang 2020] that has additional $\mathrm{poly} (H, S, d)$ factors in the main term.To the best of our knowledge, these are the first set of nearly horizon-free bounds for episodic time-homogeneous offline tabular MDP and linear MDP with anchor points. Central to our analysis is a simple yet effective recursion based method to bound a "total variance" term in the offline scenarios, which could be of individual interest.

----

## [1196] Combinatorial Optimization for Panoptic Segmentation: A Fully Differentiable Approach

**Authors**: *Ahmed Abbas, Paul Swoboda*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/83a368f54768f506b833130584455df4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/83a368f54768f506b833130584455df4-Abstract.html)

**Abstract**:

We propose a fully differentiable architecture for simultaneous semantic and instance segmentation (a.k.a. panoptic segmentation) consisting of a convolutional neural network and an asymmetric multiway cut problem solver. The latter solves a combinatorial optimization problem that elegantly incorporates semantic and boundary predictions to produce a panoptic labeling. Our formulation allows to directly maximize a smooth surrogate of the panoptic quality metric by backpropagating the gradient through the optimization problem. Experimental evaluation shows improvement by backpropagating through the optimization problem w.r.t. comparable approaches on Cityscapes and COCO datasets. Overall, our approach of combinatorial optimization for panoptic segmentation (COPS) shows the utility of using optimization in tandem with deep learning in a challenging large scale real-world problem and showcases benefits and insights into training such an architecture.

----

## [1197] Reinforcement Learning with State Observation Costs in Action-Contingent Noiselessly Observable Markov Decision Processes

**Authors**: *Hyunji Alex Nam, Scott L. Fleming, Emma Brunskill*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/83e8fe6279ad25f15b23c6298c6a3584-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/83e8fe6279ad25f15b23c6298c6a3584-Abstract.html)

**Abstract**:

Many real-world problems that require making optimal sequences of decisions under uncertainty involve costs when the agent wishes to obtain information about its environment. We design and analyze algorithms for reinforcement learning (RL) in Action-Contingent Noiselessly Observable MDPs (ACNO-MDPs), a special class of POMDPs in which the agent can choose to either (1) fully observe the state at a cost and then act; or (2) act without any immediate observation information, relying on past observations to infer the underlying state. ACNO-MDPs arise frequently in important real-world application domains like healthcare, in which clinicians must balance the value of information gleaned from medical tests (e.g., blood-based biomarkers) with the costs of gathering that information (e.g., the costs of labor and materials required to administer such tests). We develop a PAC RL algorithm for tabular ACNO-MDPs that provides substantially tighter bounds, compared to generic POMDP-RL algorithms, on the total number of episodes exhibiting worse than near-optimal performance. For continuous-state ACNO-MDPs, we propose a novel method of incorporating observation information that, when coupled with modern RL algorithms, yields significantly faster learning compared to other POMDP-RL algorithms in several simulated environments.

----

## [1198] Iterative Amortized Policy Optimization

**Authors**: *Joseph Marino, Alexandre Piché, Alessandro Davide Ialongo, Yisong Yue*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/83fa5a432ae55c253d0e60dbfa716723-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/83fa5a432ae55c253d0e60dbfa716723-Abstract.html)

**Abstract**:

Policy networks are a central feature of deep reinforcement learning (RL) algorithms for continuous control, enabling the estimation and sampling of high-value actions. From the variational inference perspective on RL, policy networks, when used with entropy or KL regularization, are a form of amortized optimization, optimizing network parameters rather than the policy distributions directly. However, direct amortized mappings can yield suboptimal policy estimates and restricted distributions, limiting performance and exploration. Given this perspective, we consider the more flexible class of iterative amortized optimizers. We demonstrate that the resulting technique, iterative amortized policy optimization, yields performance improvements over direct amortization on benchmark continuous control tasks.

----

## [1199] Revisiting the Calibration of Modern Neural Networks

**Authors**: *Matthias Minderer, Josip Djolonga, Rob Romijnders, Frances Hubis, Xiaohua Zhai, Neil Houlsby, Dustin Tran, Mario Lucic*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/8420d359404024567b5aefda1231af24-Abstract.html)

**Abstract**:

Accurate estimation of predictive uncertainty (model calibration) is essential for the safe application of neural networks. Many instances of miscalibration in modern neural networks have been reported, suggesting a trend that newer, more accurate models produce poorly calibrated predictions. Here, we revisit this question for recent state-of-the-art image classification models. We systematically relate model calibration and accuracy, and find that the most recent models, notably those not using convolutions, are among the best calibrated. Trends observed in prior model generations, such as decay of calibration with distribution shift or model size, are less pronounced in recent architectures. We also show that model size and amount of pretraining do not fully explain these differences, suggesting that architecture is a major determinant of calibration properties.

----



[Go to the previous page](NIPS-2021-list05.md)

[Go to the next page](NIPS-2021-list07.md)

[Go to the catalog section](README.md)