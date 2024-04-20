## [0] GAIA: Delving into Gradient-based Attribution Abnormality for Out-of-distribution Detection

**Authors**: *Jinggang Chen, Junjie Li, Xiaoyang Qu, Jianzong Wang, Jiguang Wan, Jing Xiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fcdccd419c4dc471fa3b73ec97b53789-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fcdccd419c4dc471fa3b73ec97b53789-Abstract-Conference.html)

**Abstract**:

Detecting out-of-distribution (OOD) examples is crucial to guarantee the reliability and safety of deep neural networks in real-world settings. In this paper, we offer an innovative perspective on quantifying the disparities between in-distribution (ID) and OOD data---analyzing the uncertainty that arises when models attempt to explain their predictive decisions. This perspective is motivated by our observation that gradient-based attribution methods encounter challenges in assigning feature importance to OOD data, thereby yielding divergent explanation patterns. Consequently, we investigate how attribution gradients lead to uncertain explanation outcomes and introduce two forms of abnormalities for OOD detection: the zero-deflation abnormality and the channel-wise average abnormality. We then propose GAIA, a simple and effective approach that incorporates Gradient Abnormality Inspection and Aggregation.  The effectiveness of GAIA is validated on both commonly utilized (CIFAR) and large-scale (ImageNet-1k) benchmarks. Specifically, GAIA reduces the average FPR95 by 23.10% on CIFAR10 and by 45.41% on CIFAR100 compared to advanced post-hoc methods.

----

## [0] Context-guided Embedding Adaptation for Effective Topic Modeling in Low-Resource Regimes

**Authors**: *Yishi Xu, Jianqiao Sun, Yudi Su, Xinyang Liu, Zhibin Duan, Bo Chen, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fce176458ff542940fa3ed16e6f9c852-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fce176458ff542940fa3ed16e6f9c852-Abstract-Conference.html)

**Abstract**:

Embedding-based neural topic models have turned out to be a superior option for low-resourced topic modeling. However, current approaches consider static word embeddings learnt from source tasks as general knowledge that can be transferred directly to the target task, discounting the dynamically changing nature of word meanings in different contexts, thus typically leading to sub-optimal results when adapting to new tasks with unfamiliar contexts. To settle this issue, we provide an effective method that centers on adaptively generating semantically tailored word embeddings for each task by fully exploiting contextual information. Specifically, we first condense the contextual syntactic dependencies of words into a semantic graph for each task, which is then modeled by a Variational Graph Auto-Encoder to produce task-specific word representations. On this basis, we further impose a learnable Gaussian mixture prior on the latent space of words to efficiently learn topic representations from a clustering perspective, which contributes to diverse topic discovery and fast adaptation to novel tasks. We have conducted a wealth of quantitative and qualitative experiments, and the results show that our approach comprehensively outperforms established topic models.

----

## [0] Discovering General Reinforcement Learning Algorithms with Adversarial Environment Design

**Authors**: *Matthew Thomas Jackson, Minqi Jiang, Jack Parker-Holder, Risto Vuorio, Chris Lu, Gregory Farquhar, Shimon Whiteson, Jakob N. Foerster*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fce2d8a485746f76aac7b5650db2679d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fce2d8a485746f76aac7b5650db2679d-Abstract-Conference.html)

**Abstract**:

The past decade has seen vast progress in deep reinforcement learning (RL) on the back of algorithms manually designed by human researchers. Recently, it has been shown that it is possible to meta-learn update rules, with the hope of discovering algorithms that can perform well on a wide range of RL tasks. Despite impressive initial results from algorithms such as Learned Policy Gradient (LPG), there remains a generalization gap when these algorithms are applied to unseen environments. In this work, we examine how characteristics of the meta-training distribution impact the generalization performance of these algorithms. Motivated by this analysis and building on ideas from Unsupervised Environment Design (UED), we propose a novel approach for automatically generating curricula to maximize the regret of a meta-learned optimizer, in addition to a novel approximation of regret, which we name algorithmic regret (AR). The result is our method, General RL Optimizers Obtained Via Environment Design (GROOVE). In a series of experiments, we show that GROOVE achieves superior generalization to LPG, and evaluate AR against baseline metrics from UED, identifying it as a critical component of environment design in this setting. We believe this approach is a step towards the discovery of truly general RL algorithms, capable of solving a wide range of real-world environments.

----

## [0] A Riemannian Exponential Augmented Lagrangian Method for Computing the Projection Robust Wasserstein Distance

**Authors**: *Bo Jiang, Ya-Feng Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd02779b6c8885efc69bab6dd9571cee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd02779b6c8885efc69bab6dd9571cee-Abstract-Conference.html)

**Abstract**:

Projection robust Wasserstein (PRW) distance is recently proposed to efficiently mitigate the curse of dimensionality in the classical Wasserstein distance.  In this paper, by equivalently reformulating the computation of the PRW distance as an optimization problem over the Cartesian product of the Stiefel manifold and the Euclidean space with additional nonlinear inequality constraints, we propose a Riemannian exponential augmented Lagrangian method (REALM) for solving this problem. Compared with the existing Riemannian exponential penalty-based approaches, REALM can potentially avoid too small penalty parameters and exhibit more stable numerical performance. To solve the subproblems in REALM efficiently, we design an inexact Riemannian Barzilai-Borwein method with Sinkhorn iteration  (iRBBS), which selects the stepsizes adaptively rather than tuning the stepsizes in efforts as done in the existing methods. We show that iRBBS can return an $\epsilon$-stationary point of the original PRW distance problem within  $\mathcal{O}(\epsilon^{-3})$ iterations, which matches the best known iteration complexity result. Extensive numerical results demonstrate that our proposed methods outperform the state-of-the-art solvers for computing the PRW distance.

----

## [0] Context Shift Reduction for Offline Meta-Reinforcement Learning

**Authors**: *Yunkai Gao, Rui Zhang, Jiaming Guo, Fan Wu, Qi Yi, Shaohui Peng, Siming Lan, Ruizhi Chen, Zidong Du, Xing Hu, Qi Guo, Ling Li, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd489a44f3bcb9f122e4931ef21d0c43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd489a44f3bcb9f122e4931ef21d0c43-Abstract-Conference.html)

**Abstract**:

Offline meta-reinforcement learning (OMRL) utilizes pre-collected offline datasets to enhance the agent's generalization ability on unseen tasks. However, the context shift problem arises due to the distribution discrepancy between the contexts used for training (from the behavior policy) and testing (from the exploration policy). The context shift problem leads to incorrect task inference and further deteriorates the generalization ability of the meta-policy. Existing OMRL methods either overlook this problem or attempt to mitigate it with additional information. In this paper, we propose a novel approach called Context Shift Reduction for OMRL (CSRO) to address the context shift problem with only offline datasets. The key insight of CSRO is to minimize the influence of policy in context during both the meta-training and meta-test phases.  During meta-training, we design a max-min mutual information representation learning mechanism to diminish the impact of the behavior policy on task representation. In the meta-test phase, we introduce the non-prior context collection strategy to reduce the effect of the exploration policy. Experimental results demonstrate that CSRO significantly reduces the context shift and improves the generalization ability, surpassing previous methods across various challenging domains.

----

## [0] Towards Data-Agnostic Pruning At Initialization: What Makes a Good Sparse Mask?

**Authors**: *Hoang Pham, The-Anh Ta, Shiwei Liu, Lichuan Xiang, Dung Le, Hongkai Wen, Long Tran-Thanh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd5013ea0c3f96931dec77174eaf9d80-Abstract-Conference.html)

**Abstract**:

Pruning at initialization (PaI) aims to remove weights of neural networks before training in pursuit of training efficiency besides the inference. While off-the-shelf PaI methods manage to find trainable subnetworks that  outperform random pruning, their performance in terms of both accuracy and computational reduction is far from satisfactory compared to post-training pruning and the understanding of PaI is missing.  For instance, recent studies show that existing PaI methods only able to find good layerwise sparsities not weights, as the discovered subnetworks are surprisingly resilient against layerwise random mask shuffling and weight re-initialization.In this paper, we study PaI from a brand-new perspective -- the topology of subnetworks. In particular, we propose a principled framework for analyzing the performance of Pruning and Initialization (PaI) methods with two quantities, namely, the number of effective paths and effective nodes. These quantities allow for a more comprehensive understanding of PaI methods, giving us an accurate assessment of different subnetworks at initialization. We systematically analyze the behavior of various PaI methods through our framework and observe a guiding principle for constructing effective subnetworks: *at a specific sparsity, the top-performing subnetwork always presents a good balance between the number of effective nodes and the number of effective paths.*Inspired by this observation, we present a novel data-agnostic pruning method by solving a multi-objective optimization problem. By conducting extensive experiments across different architectures and datasets, our results demonstrate that our approach outperforms state-of-the-art PaI methods while it is able to discover subnetworks that have much lower inference FLOPs (up to 3.4$\times$). Code will be fully released.

----

## [0] New Bounds for Hyperparameter Tuning of Regression Problems Across Instances

**Authors**: *Maria-Florina Balcan, Anh Nguyen, Dravyansh Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd62b65606f0f0d2af2c01623a224258-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd62b65606f0f0d2af2c01623a224258-Abstract-Conference.html)

**Abstract**:

The task of tuning regularization coefficients in regularized regression models with provable guarantees across problem instances still poses a significant challenge in the literature. This paper investigates the sample complexity of tuning regularization parameters in linear and logistic regressions under $\ell_1$ and $\ell_2$-constraints in the data-driven setting. For the linear regression problem, by more carefully exploiting the structure of the dual function class, we provide a new upper bound for the pseudo-dimension of the validation loss function class, which significantly improves the best-known results on the problem. Remarkably, we also instantiate the first matching lower bound, proving our results are tight. For tuning the regularization parameters of logistic regression, we introduce a new approach to studying the learning guarantee via an approximation of the validation loss function class. We examine the pseudo-dimension of the approximation class and construct a uniform error bound between the validation loss function class and its approximation, which allows us to instantiate the first learning guarantee for the problem of tuning logistic regression regularization coefficients.

----

## [0] Jailbroken: How Does LLM Safety Training Fail?

**Authors**: *Alexander Wei, Nika Haghtalab, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html)

**Abstract**:

Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of “jailbreak” attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model’s capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI’s GPT-4 and Anthropic’s Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models’ red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity—that safety mechanisms should be as sophisticated as the underlying model—and argues against the idea that scaling alone can resolve these safety failure modes.

----

## [0] Conditional Mutual Information for Disentangled Representations in Reinforcement Learning

**Authors**: *Mhairi Dunion, Trevor McInroe, Kevin Sebastian Luck, Josiah Hanna, Stefano V. Albrecht*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd750154df5f199f94df897975621306-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd750154df5f199f94df897975621306-Abstract-Conference.html)

**Abstract**:

Reinforcement Learning (RL) environments can produce training data with spurious correlations between features due to the amount of training data or its limited feature coverage. This can lead to RL agents encoding these misleading correlations in their latent representation, preventing the agent from generalising if the correlation changes within the environment or when deployed in the real world. Disentangled representations can improve robustness, but existing disentanglement techniques that minimise mutual information between features require independent features, thus they cannot disentangle correlated features. We propose an auxiliary task for RL algorithms that learns a disentangled representation of high-dimensional observations with correlated features by minimising the conditional mutual information between features in the representation. We demonstrate experimentally, using continuous control tasks, that our approach improves generalisation under correlation shifts, as well as improving the training performance of RL algorithms in the presence of correlated features.

----

## [0] Comparing Causal Frameworks: Potential Outcomes, Structural Models, Graphs, and Abstractions

**Authors**: *Duligur Ibeling, Thomas Icard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd83f4e0dcaf1c64ea15bbb1695bb40f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd83f4e0dcaf1c64ea15bbb1695bb40f-Abstract-Conference.html)

**Abstract**:

The aim of this paper is to make clear and precise the relationship between the Rubin causal model (RCM) and structural causal model (SCM) frameworks for causal inference. Adopting a neutral logical perspective, and drawing on previous work, we show what is required for an RCM to be representable by an SCM. A key result then shows that every RCM---including those that violate algebraic principles implied by the SCM framework---emerges as an abstraction of some representable RCM. Finally, we illustrate the power of this ameliorative perspective by pinpointing an important role for SCM principles in classic applications of RCMs; conversely, we offer a characterization of the algebraic constraints implied by a graph, helping to substantiate further comparisons between the two frameworks.

----

## [0] LogSpecT: Feasible Graph Learning Model from Stationary Signals with Recovery Guarantees

**Authors**: *Shangyuan Liu, Linglingzhi Zhu, Anthony Man-Cho So*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fd8872fcba4ba87312cdfe5ebba91ca9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fd8872fcba4ba87312cdfe5ebba91ca9-Abstract-Conference.html)

**Abstract**:

Graph learning from signals is a core task in graph signal processing (GSP). A significant subclass of graph signals called the stationary graph signals that broadens the concept of stationarity of data defined on regular domains to signals on graphs is gaining increasing popularity in the GSP community. The most commonly used model to learn graphs from these stationary signals is SpecT, which forms the foundation for nearly all the subsequent, more advanced models. Despite its strengths, the practical formulation of the model, known as rSpecT, has been identified to be susceptible to the choice of hyperparameters. More critically, it may suffer from infeasibility as an optimization problem. In this paper, we introduce the first condition that ensures the infeasibility of rSpecT and design a novel model called LogSpecT, along with its practical formulation rLogSpecT to overcome this issue. Contrary to rSpecT, our novel practical model rLogSpecT is always feasible. Furthermore, we provide recovery guarantees of rLogSpecT from modern optimization tools related to epi-convergence, which could be of independent interest and significant for various learning problems. To demonstrate the practical advantages of rLogSpecT, a highly efficient algorithm based on the linearized alternating direction method of multipliers (L-ADMM) that allows closed-form solutions for each subproblem is proposed with convergence guarantees. Extensive numerical results on both synthetic and real networks not only corroborate the stability of our proposed methods, but also highlight their comparable and even superior performance than existing models.

----

## [0] Depth-discriminative Metric Learning for Monocular 3D Object Detection

**Authors**: *Wonhyeok Choi, Mingyu Shin, Sunghoon Im*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fda257e65f46e21dbc117b20fd0aba3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fda257e65f46e21dbc117b20fd0aba3c-Abstract-Conference.html)

**Abstract**:

Monocular 3D object detection poses a significant challenge due to the lack of depth information in RGB images. Many existing methods strive to enhance the object depth estimation performance by allocating additional parameters for object depth estimation, utilizing extra modules or data. In contrast, we introduce a novel metric learning scheme that encourages the model to extract depth-discriminative features regardless of the visual attributes without increasing inference time and model size. Our method employs the distance-preserving function to organize the feature space manifold in relation to ground-truth object depth. The proposed $(K,B,\epsilon)$-quasi-isometric loss leverages predetermined pairwise distance restriction as guidance for adjusting the distance among object descriptors without disrupting the non-linearity of the natural feature manifold. Moreover, we introduce an auxiliary head for object-wise depth estimation, which enhances depth quality while maintaining the inference time. The broad applicability of our method is demonstrated through experiments that show improvements in overall performance when integrated into various baselines. The results show that our method consistently improves the performance of various baselines by 23.51\% and 5.78\% on average across KITTI and Waymo, respectively.

----

## [0] PLANNER: Generating Diversified Paragraph via Latent Language Diffusion Model

**Authors**: *Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Joshua M. Susskind, Navdeep Jaitly*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fdba5e0a9b57fce03e89cc0cad0a24e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fdba5e0a9b57fce03e89cc0cad0a24e9-Abstract-Conference.html)

**Abstract**:

Autoregressive models for text sometimes generate repetitive and low-quality output because errors accumulate during the steps of generation. This issue is often attributed to exposure bias -- the difference between how a model is trained, and how it is used during inference. Denoising diffusion models provide an alternative approach in which a model can revisit and revise its output. However, they can be computationally expensive and prior efforts on text have led to models that produce less fluent output compared to autoregressive models, especially for longer text and paragraphs. In this paper, we propose PLANNER, a model that combines latent semantic diffusion with autoregressive generation, to generate fluent text while exercising global control over paragraphs. The model achieves this by combining an autoregressive "decoding" module with a "planning" module  that uses latent diffusion to generate semantic paragraph embeddings in a coarse-to-fine manner. The proposed method is evaluated on various conditional generation tasks, and results on semantic generation, text completion and summarization show its effectiveness in generating high-quality long-form text in an efficient manner.

----

## [0] Generalized Bayesian Inference for Scientific Simulators via Amortized Cost Estimation

**Authors**: *Richard Gao, Michael Deistler, Jakob H. Macke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fdd565f63f49776bef620e0ce368a492-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fdd565f63f49776bef620e0ce368a492-Abstract-Conference.html)

**Abstract**:

Simulation-based inference (SBI) enables amortized Bayesian inference for simulators with implicit likelihoods. But when we are primarily interested in the quality of predictive simulations, or when the model cannot exactly reproduce the observed data (i.e., is misspecified), targeting the Bayesian posterior may be overly restrictive. Generalized Bayesian Inference (GBI) aims to robustify inference for (misspecified) simulator models, replacing the likelihood-function with a cost function that evaluates the goodness of parameters relative to data. However, GBI methods generally require running multiple simulations to estimate the cost function at each parameter value during inference, making the approach computationally infeasible for even moderately complex simulators. Here, we propose amortized cost estimation (ACE) for GBI to address this challenge: We train a neural network to approximate the cost function, which we define as the expected distance between simulations produced by a parameter and observed data. The trained network can then be used with MCMC to infer GBI posteriors for any observation without running additional simulations. We show that, on several benchmark tasks, ACE accurately predicts cost and provides predictive simulations that are closer to synthetic observations than other SBI methods, especially for misspecified simulators. Finally, we apply ACE to infer parameters of the Hodgkin-Huxley model given real intracellular recordings from the Allen Cell Types Database. ACE identifies better data-matching parameters while being an order of magnitude more simulation-efficient than a standard SBI method. In summary, ACE combines the strengths of SBI methods and GBI to perform robust and simulation-amortized inference for scientific simulators.

----

## [0] Harnessing the power of choices in decision tree learning

**Authors**: *Guy Blanc, Jane Lange, Chirag Pabbaraju, Colin Sullivan, Li-Yang Tan, Mo Tiwari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fddad60891bdf85aac8041f80ed022df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fddad60891bdf85aac8041f80ed022df-Abstract-Conference.html)

**Abstract**:

We propose a simple generalization of standard and empirically successful decision tree learning algorithms such as ID3, C4.5, and CART.   These algorithms, which have been central to machine learning for decades, are greedy in nature: they grow a decision tree by iteratively splitting on the best attribute.  Our algorithm, Top-$k$, considers the $k$ best attributes as possible splits instead of just the single best attribute. We demonstrate, theoretically and empirically, the power of this simple generalization.  We first prove a greediness hierarchy theorem showing that for every $k\in \mathbb{N}$, Top-$(k+1)$  can be dramatically more powerful than Top-$k$: there are data distributions for which the former achieves accuracy $1-\epsilon$, whereas the latter only achieves accuracy $\frac{1}{2}+\epsilon$.  We then show, through extensive experiments, that Top-$k$ outperforms the two main approaches to decision tree learning: classic greedy algorithms and more recent ``optimal decision tree'' algorithms.  On one hand, Top-$k$ consistently enjoys significant accuracy gains over greedy algorithms across a wide range of benchmarks.  On the other hand, Top-$k$ is markedly more scalable than optimal decision tree algorithms and is able to handle dataset and feature set sizes that remain far beyond the reach of these algorithms. The code to reproduce our results is available at https://github.com/SullivanC19/pydl8.5-topk.

----

## [0] Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals

**Authors**: *Tam Nguyen, Tan Nguyen, Richard G. Baraniuk*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fde1a69a5b6e554b2f1f727197d2651d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fde1a69a5b6e554b2f1f727197d2651d-Abstract-Conference.html)

**Abstract**:

Transformers have achieved remarkable success in a wide range of natural language processing and computer vision applications. However, the representation capacity of a deep transformer model is degraded due to the over-smoothing issue in which the token representations become identical when the model's depth grows. In this work, we show that self-attention layers in transformers minimize a functional which promotes smoothness, thereby causing token uniformity. We then propose a novel regularizer that penalizes the norm of the difference between the smooth output tokens from self-attention and the input tokens to preserve the fidelity of the tokens. Minimizing the resulting regularized energy functional, we derive the Neural Transformer with a Regularized Nonlocal Functional (NeuTRENO), a novel class of transformer models that can mitigate the over-smoothing issue. We empirically demonstrate the advantages of NeuTRENO over the baseline transformers and state-of-the-art methods in reducing the over-smoothing of token representations on various practical tasks, including object classification, image segmentation, and language modeling.

----

## [0] On the Role of Noise in the Sample Complexity of Learning Recurrent Neural Networks: Exponential Gaps for Long Sequences

**Authors**: *Alireza Fathollah Pour, Hassan Ashtiani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe03053bd2cf5b5c56de1e463bc53e1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe03053bd2cf5b5c56de1e463bc53e1a-Abstract-Conference.html)

**Abstract**:

We consider the class of noisy multi-layered sigmoid recurrent neural networks with $w$ (unbounded) weights for classification of sequences of length $T$, where independent noise distributed according to $\mathcal{N}(0,\sigma^2)$ is added to the output of each neuron in the network. Our main result shows that the sample complexity of PAC learning this class can be bounded by $O (w\log(T/\sigma))$. For the non-noisy version of the same class (i.e., $\sigma=0$), we prove a lower bound of $\Omega (wT)$ for the sample complexity.   Our results indicate an exponential gap in the dependence of sample complexity on $T$ for noisy versus non-noisy networks. Moreover, given the mild logarithmic dependence of the upper bound on $1/\sigma$, this gap still holds even for numerically negligible values of $\sigma$.

----

## [0] Provably Bounding Neural Network Preimages

**Authors**: *Suhas Kotha, Christopher Brix, J. Zico Kolter, Krishnamurthy Dvijotham, Huan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe061ec0ae03c5cf5b5323a2b9121bfd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe061ec0ae03c5cf5b5323a2b9121bfd-Abstract-Conference.html)

**Abstract**:

Most work on the formal verification of neural networks has focused on bounding the set of outputs that correspond to a given set of inputs (for example, bounded perturbations of a nominal input). However, many use cases of neural network verification require solving the inverse problem, or over-approximating the set of inputs that lead to certain outputs. We present the INVPROP algorithm for verifying properties over the preimage of a linearly constrained output set, which can be combined with branch-and-bound to increase precision. Contrary to other approaches, our efficient algorithm is GPU-accelerated and does not require a linear programming solver. We demonstrate our algorithm for identifying safe control regions for a dynamical system via backward reachability analysis, verifying adversarial robustness, and detecting out-of-distribution inputs to a neural network. Our results show that in certain settings, we find over-approximations over $2500\times$ tighter than prior work while being $2.5\times$ faster. By strengthening robustness verification with output constraints, we consistently verify more properties than the previous state-of-the-art on multiple benchmarks, including a large model with 167k neurons in VNN-COMP 2023. Our algorithm has been incorporated into the $\alpha,\beta$-CROWN verifier, available at https://abcrown.org.

----

## [0] Scaling Riemannian Diffusion Models

**Authors**: *Aaron Lou, Minkai Xu, Adam Farris, Stefano Ermon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe1ab2f77a9a0f224839cc9f1034a908-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe1ab2f77a9a0f224839cc9f1034a908-Abstract-Conference.html)

**Abstract**:

Riemannian diffusion models draw inspiration from standard Euclidean space diffusion models to learn distributions on general manifolds. Unfortunately, the additional geometric complexity renders the diffusion transition term inexpressible in closed form, so prior methods resort to imprecise approximations of the score matching training objective that degrade performance and preclude applications in high dimensions. In this work, we reexamine these approximations and propose several practical improvements. Our key observation is that most relevant manifolds are symmetric spaces, which are much more amenable to computation. By leveraging and combining various ans\"{a}tze, we can quickly compute relevant quantities to high precision. On low dimensional datasets, our correction produces a noticeable improvement and is competitive with other techniques. Additionally, we show that our method enables us to scale to high dimensional tasks on nontrivial manifolds, including $SU(n)$ lattices in the context of lattice quantum chromodynamics (QCD). Finally, we apply our models to contrastively learned hyperspherical embeddings, curbing the representation collapse problem in the projection head and closing the gap between theory and practice.

----

## [0] Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models

**Authors**: *Siyan Zhao, Aditya Grover*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe1c4991d57f37dfef62d01b3901ca54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe1c4991d57f37dfef62d01b3901ca54-Abstract-Conference.html)

**Abstract**:

Reinforcement learning presents an attractive paradigm to reason about several distinct aspects of sequential decision making, such as specifying complex goals, planning future observations and actions, and critiquing their utilities. However, the combined integration of these capabilities poses competing algorithmic challenges in retaining maximal expressivity while allowing for flexibility in modeling choices for efficient learning and inference. We present Decision Stacks, a generative framework that decomposes goal-conditioned policy agents into 3 generative modules. These modules simulate the temporal evolution of observations, rewards, and actions via independent generative models that can be learned in parallel via teacher forcing. Our framework guarantees both expressivity and flexibility in designing individual modules to account for key factors such as architectural bias, optimization objective and dynamics, transferrability across domains, and inference speed. Our empirical results demonstrate the effectiveness of Decision Stacks for offline policy optimization for several MDP and POMDP environments, outperforming existing methods and enabling flexible generative decision making.

----

## [0] Conformal Prediction for Uncertainty-Aware Planning with Diffusion Dynamics Model

**Authors**: *Jiankai Sun, Yiqi Jiang, Jianing Qiu, Parth Nobel, Mykel J. Kochenderfer, Mac Schwager*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe318a2b6c699808019a456b706cd845-Abstract-Conference.html)

**Abstract**:

Robotic applications often involve working in environments that are uncertain, dynamic, and partially observable. Recently, diffusion models have been proposed for learning trajectory prediction models trained from expert demonstrations, which can be used for planning in robot tasks. Such models have demonstrated a strong ability to overcome challenges such as multi-modal action distributions, high-dimensional output spaces, and training instability. It is crucial to quantify the uncertainty of these dynamics models when using them for planning. In this paper, we quantify the uncertainty of diffusion dynamics models using Conformal Prediction (CP). Given a finite number of exchangeable expert trajectory examples (called the “calibration set”), we use CP to obtain a set in the trajectory space (called the “coverage region”) that is guaranteed to contain the output of the diffusion model with a user-defined probability (called the “coverage level”). In PlanCP, inspired by concepts from conformal prediction, we modify the loss function for training the diffusion model to include a quantile term to encourage more robust performance across the variety of training examples. At test time, we then calibrate PlanCP with a conformal prediction process to obtain coverage sets for the trajectory prediction with guaranteed coverage level. We evaluate our algorithm on various planning tasks and model-based offline reinforcement learning tasks and show that it reduces the uncertainty of the learned trajectory prediction model. As a by-product, our algorithm PlanCP outperforms prior algorithms on existing offline RL benchmarks and challenging continuous planning tasks. Our method can be combined with most model-based planning approaches to produce uncertainty estimates of the closed-loop system.

----

## [0] Max-Sliced Mutual Information

**Authors**: *Dor Tsur, Ziv Goldfeld, Kristjan H. Greenewald*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe4da14f07561a232782820d30ea22f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe4da14f07561a232782820d30ea22f3-Abstract-Conference.html)

**Abstract**:

Quantifying dependence between high-dimensional random variables is central to statistical learning and inference. Two classical methods are canonical correlation analysis (CCA), which identifies maximally correlated projected versions of the original variables, and Shannon's mutual information, which is a universal dependence measure that also captures high-order dependencies. However, CCA only accounts for linear dependence, which may be insufficient for certain applications, while mutual information is often infeasible to compute/estimate in high dimensions. This work proposes a middle ground in the form of a scalable information-theoretic generalization of CCA, termed max-sliced mutual information (mSMI). mSMI equals the maximal mutual information between low-dimensional projections of the high-dimensional variables, which reduces back to CCA in the Gaussian case. It enjoys the best of both worlds: capturing intricate dependencies in the data while being amenable to fast computation and scalable estimation from samples. We show that mSMI retains favorable structural properties of Shannon's mutual information, like variational forms and identification of independence. We then study statistical estimation of mSMI, propose an efficiently computable neural estimator, and couple it with formal non-asymptotic error bounds. We present experiments that demonstrate the utility of mSMI for several tasks, encompassing independence testing, multi-view representation learning, algorithmic fairness, and generative modeling. We observe that mSMI consistently outperforms competing methods with little-to-no computational overhead.

----

## [0] Neural Data Transformer 2: Multi-context Pretraining for Neural Spiking Activity

**Authors**: *Joel Ye, Jennifer L. Collinger, Leila Wehbe, Robert Gaunt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe51de4e7baf52e743b679e3bdba7905-Abstract-Conference.html)

**Abstract**:

The neural population spiking activity recorded by intracortical brain-computer interfaces (iBCIs) contain rich structure. Current models of such spiking activity are largely prepared for individual experimental contexts, restricting data volume to that collectable within a single session and limiting the effectiveness of deep neural networks (DNNs). The purported challenge in aggregating neural spiking data is the pervasiveness of context-dependent shifts in the neural data distributions. However, large scale unsupervised pretraining by nature spans heterogeneous data, and has proven to be a fundamental recipe for successful representation learning across deep learning. We thus develop Neural Data Transformer 2 (NDT2), a spatiotemporal Transformer for neural spiking activity, and demonstrate that pretraining can leverage motor BCI datasets that span sessions, subjects, and experimental tasks. NDT2 enables rapid adaptation to novel contexts in downstream decoding tasks and opens the path to deployment of pretrained DNNs for iBCI control. Code: https://github.com/joel99/contextgeneralbci

----

## [0] Data Quality in Imitation Learning

**Authors**: *Suneel Belkhale, Yuchen Cui, Dorsa Sadigh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe692980c5d9732cf153ce27947653a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe692980c5d9732cf153ce27947653a7-Abstract-Conference.html)

**Abstract**:

In supervised learning, the question of data quality and curation has been sidelined in recent years in favor of increasingly more powerful and expressive models that can ingest internet-scale data. However, in offline learning for robotics, we simply lack internet scale data, and so high quality datasets are a necessity. This is especially true in imitation learning (IL), a sample efficient paradigm for robot learning using expert demonstrations. Policies learned through IL suffer from state distribution shift at test time due to compounding errors in action prediction, which leads to unseen states that the policy cannot recover from.Instead of designing new algorithms to address distribution shift, an alternative perspective is to develop new ways of assessing and curating datasets. There is growing evidence that the same IL algorithms can have substantially different performance across different datasets. This calls for a formalism for defining metrics of "data quality" that can further be leveraged for data curation.In this work, we take the first step toward formalizing data quality for imitation learning through the lens of distribution shift: a high quality dataset encourages the policy to stay in distribution at test time. We propose two fundamental properties that are necessary for a high quality datasets: i) action divergence: the mismatch between the expert and learned policy at certain states; and ii) transition diversity: the noise present in the system for a given state and action. We investigate the combined effect of these two key properties in imitation learning theoretically, and we empirically analyze models trained on a variety of different data sources. We show that state diversity is not always beneficial, and we demonstrate how action divergence and transition diversity interact in practice.

----

## [0] Align Your Prompts: Test-Time Prompting with Distribution Alignment for Zero-Shot Generalization

**Authors**: *Jameel Abdul Samadh, Hanan Gani, Noor Hussein, Muhammad Uzair Khattak, Muzammal Naseer, Fahad Shahbaz Khan, Salman H. Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe8debfd5a36ada52e038c8b2078b2ce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe8debfd5a36ada52e038c8b2078b2ce-Abstract-Conference.html)

**Abstract**:

The promising zero-shot generalization of vision-language models such as CLIP has led to their adoption using prompt learning for numerous downstream tasks. Previous works have shown test-time prompt tuning using entropy minimization to adapt text prompts for unseen domains. While effective, this overlooks the key cause for performance degradation to unseen domains -- distribution shift. In this work, we explicitly handle this problem by aligning the out-of-distribution (OOD) test sample statistics to those of the source data using prompt tuning. We use a single test sample to adapt multi-modal prompts at test time by minimizing the feature distribution shift to bridge the gap in the test domain. Evaluating against the domain generalization benchmark, our method improves zero-shot top-1 accuracy beyond existing prompt-learning techniques, with a 3.08% improvement over the baseline MaPLe. In cross-dataset generalization with unseen categories across 10 datasets, our method improves consistently across all datasets compared to the existing state-of-the-art. Our source code and models are available at https://jameelhassan.github.io/promptalign

----

## [0] SLM: A Smoothed First-Order Lagrangian Method for Structured Constrained Nonconvex Optimization

**Authors**: *Songtao Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fe90657b12193c7b52a3418bdc351807-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fe90657b12193c7b52a3418bdc351807-Abstract-Conference.html)

**Abstract**:

Functional constrained optimization (FCO) has emerged as a powerful tool for solving various machine learning problems. However, with the rapid increase in applications of neural networks in recent years, it has become apparent that both the objective and constraints often involve nonconvex functions, which poses significant challenges in obtaining high-quality solutions. In this work, we focus on a class of nonconvex FCO problems with nonconvex constraints, where the two optimization variables are nonlinearly coupled in the inequality constraint. Leveraging the primal-dual optimization framework, we propose a smoothed first-order Lagrangian method (SLM) for solving this class of problems. We establish the theoretical convergence guarantees of SLM to the Karush-Kuhn-Tucker (KKT) solutions through quantifying dual error bounds. By establishing connections between this structured FCO and equilibrium-constrained nonconvex problems (also known as bilevel optimization), we apply the proposed SLM to tackle bilevel optimization oriented problems where the lower-level problem is nonconvex. Numerical results obtained from both toy examples and hyper-data cleaning problems demonstrate the superiority of SLM compared to benchmark methods.

----

## [0] Auslan-Daily: Australian Sign Language Translation for Daily Communication and News

**Authors**: *Xin Shen, Shaozu Yuan, Hongwei Sheng, Heming Du, Xin Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/feb34ce77fc8b94c85d12e608b23ce67-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/feb34ce77fc8b94c85d12e608b23ce67-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Sign language translation (SLT) aims to convert a continuous sign language video clip into a spoken language. Considering different geographic regions generally have their own native sign languages, it is valuable to establish corresponding SLT datasets to support related communication and research. Auslan, as a sign language specific to Australia, still lacks a dedicated large-scale dataset for SLT.To fill this gap, we curate an Australian Sign Language translation dataset, dubbed Auslan-Daily, which is collected from the Auslan educational TV series and Auslan TV programs. The former involves daily communications among multiple signers in the wild, while the latter comprises sign language videos for up-to-date news, weather forecasts, and documentaries. In particular, Auslan-Daily has two main features: (1) the topics are diverse and signed by multiple signers, and (2) the scenes in our dataset are more complex, e.g., captured in various environments, gesture interference during multi-signers' interactions and various camera positions. With a collection of more than 45 hours of high-quality Auslan video materials, we invite Auslan experts to align different fine-grained visual and language pairs, including video $\leftrightarrow$ fingerspelling, video $\leftrightarrow$ gloss, and video $\leftrightarrow$ sentence. As a result, Auslan-Daily contains multi-grained annotations that can be utilized to accomplish various fundamental sign language tasks, such as signer detection, sign spotting, fingerspelling detection, isolated sign language recognition, sign language translation and alignment. Moreover, we benchmark results with state-of-the-art models for each task in Auslan-Daily. Experiments indicate that Auslan-Daily is a highly challenging SLT dataset, and we hope this dataset will contribute to the development of Auslan and the advancement of sign languages worldwide in a broader context. All datasets and benchmarks are available at Auslan-Daily.

----

## [0] Red Teaming Deep Neural Networks with Feature Synthesis Tools

**Authors**: *Stephen Casper, Tong Bu, Yuxiao Li, Jiawei Li, Kevin Zhang, Kaivalya Hariharan, Dylan Hadfield-Menell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/febe5c5c6973f713cc43bf0f7c90edbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/febe5c5c6973f713cc43bf0f7c90edbe-Abstract-Conference.html)

**Abstract**:

Interpretable AI tools are often motivated by the goal of understanding model behavior in out-of-distribution (OOD) contexts. Despite the attention this area of study receives, there are comparatively few cases where these tools have identified previously unknown bugs in models. We argue that this is due, in part, to a common feature of many interpretability methods: they analyze model behavior by using a particular dataset. This only allows for the study of the model in the context of features that the user can sample in advance. To address this, a growing body of research involves interpreting models using feature synthesis methods that do not depend on a dataset. In this paper, we benchmark the usefulness of interpretability tools for model debugging. Our key insight is that we can implant human-interpretable trojans into models and then evaluate these tools based on whether they can help humans discover them. This is analogous to finding OOD bugs, except the ground truth is known, allowing us to know when a user's interpretation is correct. We make four contributions. (1) We propose trojan discovery as an evaluation task for interpretability tools and introduce a benchmark with 12 trojans of 3 different types. (2) We demonstrate the difficulty of this benchmark with a preliminary evaluation of 16 state-of-the-art feature attribution/saliency tools. Even under ideal conditions, given direct access to data with the trojan trigger, these methods still often fail to identify bugs. (3) We evaluate 7 feature-synthesis methods on our benchmark. (4) We introduce and evaluate 2 new variants of the best-performing method from the previous evaluation.

----

## [0] Rehearsal Learning for Avoiding Undesired Future

**Authors**: *Tian Qin, Tian-Zuo Wang, Zhi-Hua Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fed1ea8dcc2a13f3835cc854e8c8294c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fed1ea8dcc2a13f3835cc854e8c8294c-Abstract-Conference.html)

**Abstract**:

Machine learning (ML) models have been widely used to make predictions. Instead of a predictive statement about future outcomes, in many situations we want to pursue a decision: what can we do to avoid the undesired future if an ML model predicts so? In this paper, we present a rehearsal learning framework, in which decisions that can persuasively avoid the happening of undesired outcomes can be found and recommended. Based on the influence relation, we characterize the generative process of variables with structural rehearsal models, consisting of a probabilistic graphical model called rehearsal graphs and structural equations, and find actionable decisions that can alter the outcome by reasoning under a Bayesian framework. Moreover, we present a probably approximately correct bound to quantify the associated risk of a decision. Experiments validate the effectiveness of the proposed rehearsal learning framework and the informativeness of the bound.

----

## [0] Understanding How Consistency Works in Federated Learning via Stage-wise Relaxed Initialization

**Authors**: *Yan Sun, Li Shen, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a distributed paradigm that coordinates massive local clients to collaboratively train a global model via stage-wise local training processes on the heterogeneous dataset.  Previous works have implicitly studied that FL suffers from the "client-drift" problem, which is caused by the inconsistent optimum across local clients. However, till now it still lacks solid theoretical analysis to explain the impact of this local inconsistency.   To alleviate the negative impact of the "client drift" and explore its substance in FL, in this paper, we first design an efficient FL algorithm FedInit, which allows employing the personalized relaxed initialization state at the beginning of each local training stage. Specifically, FedInit initializes the local state by moving away from the current global state towards the reverse direction of the latest local state. This relaxed initialization helps to revise the local divergence and enhance the local consistency level.  Moreover, to further understand how inconsistency disrupts performance in FL, we introduce the excess risk analysis and study the divergence term to investigate the test error of the proposed FedInit method. Our studies show that on the non-convex objectives, optimization error is not sensitive to this local inconsistency, while it mainly affects the generalization error bound in FedInit.   Extensive experiments are conducted to validate this conclusion. Our proposed FedInit could achieve state-of-the-art (SOTA) results compared to several advanced benchmarks without any additional costs. Meanwhile, stage-wise relaxed initialization could also be incorporated into the current advanced algorithms to achieve higher performance in the FL paradigm.

----

## [0] Errors-in-variables Fr\'echet Regression with Low-rank Covariate Approximation

**Authors**: *Dogyoon Song, Kyunghee Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff06c57ef80625386884906c2d2d2429-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff06c57ef80625386884906c2d2d2429-Abstract-Conference.html)

**Abstract**:

Fr\'echet regression has emerged as a promising approach for regression analysis involving non-Euclidean response variables. However, its practical applicability has been hindered by its reliance on ideal scenarios with abundant and noiseless covariate data. In this paper, we present a novel estimation method that tackles these limitations by leveraging the low-rank structure inherent in the covariate matrix. Our proposed framework combines the concepts of global Fr\'echet regression and principal component regression, aiming to improve the efficiency and accuracy of the regression estimator. By incorporating the low-rank structure, our method enables more effective modeling and estimation, particularly in high-dimensional and errors-in-variables regression settings. We provide a theoretical analysis of the proposed estimator's large-sample properties, including a comprehensive rate analysis of bias, variance, and additional variations due to measurement errors. Furthermore, our numerical experiments provide empirical evidence that supports the theoretical findings, demonstrating the superior performance of our approach. Overall, this work introduces a promising framework for regression analysis of non-Euclidean variables, effectively addressing the challenges associated with limited and noisy covariate data, with potential applications in diverse fields.

----

## [0] Coupled Reconstruction of Cortical Surfaces by Diffeomorphic Mesh Deformation

**Authors**: *Hao Zheng, Hongming Li, Yong Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff0da832a110c6537e885cdfbac80a94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff0da832a110c6537e885cdfbac80a94-Abstract-Conference.html)

**Abstract**:

Accurate reconstruction of cortical surfaces from brain magnetic resonance images (MRIs) remains a challenging task due to the notorious partial volume effect in brain MRIs and the cerebral cortex's thin and highly folded patterns. Although many promising deep learning-based cortical surface reconstruction methods have been developed, they typically fail to model the interdependence between inner (white matter) and outer (pial) cortical surfaces, which can help generate cortical surfaces with spherical topology. To robustly reconstruct the cortical surfaces with topological correctness, we develop a new deep learning framework to jointly reconstruct the inner, outer, and their in-between (midthickness) surfaces and estimate cortical thickness directly from 3D MRIs. Our method first estimates the midthickness surface and then learns three diffeomorphic flows jointly to optimize the midthickness surface and deform it inward and outward to the inner and outer cortical surfaces respectively, regularized by topological correctness. Our method also outputs a cortex thickness value for each surface vertex, estimated from its diffeomorphic deformation trajectory. Our method has been evaluated on two large-scale neuroimaging datasets, including ADNI and OASIS, achieving state-of-the-art cortical surface reconstruction performance in terms of accuracy, surface regularity, and computation efficiency.

----

## [0] Active representation learning for general task space with applications in robotics

**Authors**: *Yifang Chen, Yingbing Huang, Simon S. Du, Kevin G. Jamieson, Guanya Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff4039889b7f89635e9cbd5cefffa0d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff4039889b7f89635e9cbd5cefffa0d4-Abstract-Conference.html)

**Abstract**:

Representation learning based on multi-task pretraining has become a powerful approach in many domains. In particular, task-aware representation learning aims to learn an optimal representation for a specific target task by sampling data from a set of source tasks, while task-agnostic representation learning seeks to learn a universal representation for a class of tasks.  In this paper, we propose a general and versatile algorithmic and theoretic framework for \emph{active representation learning}, where the learner optimally chooses which source tasks to sample from. This framework, along with a tractable meta algorithm, allows most arbitrary target and source task spaces (from discrete to continuous), covers both task-aware and task-agnostic settings, and is compatible with deep representation learning practices. We provide several instantiations under this framework, from bilinear and feature-based nonlinear to general nonlinear cases. In the bilinear case, by leveraging the non-uniform spectrum of the task representation and the calibrated source-target relevance, we prove that the sample complexity to achieve $\varepsilon$-excess risk on target scales with $(k^*)^2 ||v^*||_2^2 \varepsilon^{-2}$ where $k^*$ is the effective dimension of the target and $||v^*||_2^2 \in (0,1]$ represents the connection between source and target space. Compared to the passive one, this can save up to $\frac{1}{d_W}$ of sample complexity, where $d_W$ is the task space dimension. Finally, we demonstrate different instantiations of our meta algorithm in synthetic datasets and robotics problems, from pendulum simulations to real-world drone flight datasets. On average, our algorithms outperform baselines by 20%-70%.

----

## [0] Model and Feature Diversity for Bayesian Neural Networks in Mutual Learning

**Authors**: *Van Cuong Pham, Cuong C. Nguyen, Trung Le, Dinh Phung, Gustavo Carneiro, Thanh-Toan Do*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff521f7570d6ed23217ba5780753a1f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff521f7570d6ed23217ba5780753a1f7-Abstract-Conference.html)

**Abstract**:

Bayesian Neural Networks (BNNs) offer probability distributions for model parameters, enabling uncertainty quantification in predictions. However, they often underperform compared to deterministic neural networks. Utilizing mutual learning can effectively enhance the performance of peer BNNs. In this paper, we propose a novel approach to improve BNNs performance through deep mutual learning. The proposed approaches aim to increase diversity in both network parameter distributions and feature distributions, promoting peer networks to acquire distinct features that capture different characteristics of the input, which enhances the effectiveness of mutual learning. Experimental results demonstrate significant improvements in the classification accuracy, negative log-likelihood, and expected calibration error when compared to traditional mutual learning for BNNs.

----

## [0] Fair Graph Distillation

**Authors**: *Qizhang Feng, Zhimeng Stephen Jiang, Ruiquan Li, Yicheng Wang, Na Zou, Jiang Bian, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff6540c54a847ef9114a332c101f5edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff6540c54a847ef9114a332c101f5edc-Abstract-Conference.html)

**Abstract**:

As graph neural networks (GNNs) struggle with large-scale graphs due to high computational demands, data distillation for graph data promises to alleviate this issue by distilling a large real graph into a smaller distilled graph while maintaining comparable prediction performance for GNNs trained on both graphs. However, we observe that GNNs trained on distilled graphs may exhibit more severe group fairness problems than those trained on real graphs. Motivated by this observation, we propose \textit{fair graph distillation} (\Algnameabbr), an approach for generating small distilled \textit{fair and informative} graphs based on the graph distillation method. The challenge lies in the deficiency of sensitive attributes for nodes in the distilled graph, making most debiasing methods (e.g., regularization and adversarial debiasing) intractable for distilled graphs. We develop a simple yet effective bias metric, called coherence, for distilled graphs. Based on the proposed coherence metric, we introduce a framework for fair graph distillation using a bi-level optimization algorithm. Extensive experiments demonstrate that the proposed algorithm can achieve better prediction performance-fairness trade-offs across various datasets and GNN architectures.

----

## [0] Optimal testing using combined test statistics across independent studies

**Authors**: *Lasse Vuursteen, Botond Szabó, Aad van der Vaart, Harry van Zanten*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff703bfaf652f00ae7b609ce0da3fde2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff703bfaf652f00ae7b609ce0da3fde2-Abstract-Conference.html)

**Abstract**:

Combining test statistics from independent trials or experiments is a popular method of meta-analysis. However, there is very limited theoretical understanding of the power of the combined test, especially in high-dimensional models considering composite hypotheses tests. We derive a mathematical framework to study standard {meta-analysis} testing approaches in the context of the many normal means model, which serves as the platform to investigate more complex models.We introduce a natural and mild restriction on the meta-level combination functions of the local trials. This allows us to mathematically quantify the cost of compressing $m$ trials into real-valued test statistics and combining these. We then derive minimax lower and matching upper bounds for the separation rates of standard combination methods for e.g. p-values and e-values, quantifying the loss relative to using the full, pooled data. We observe an elbow effect, revealing that in certain cases combining the locally optimal tests in each trial results in a sub-optimal {meta-analysis} method and develop approaches to achieve the global optima. We also explore the possible gains of allowing limited coordination between the trial designs. Our results connect meta-analysis with bandwidth constraint distributed inference and build on recent information theoretic developments in the latter field.

----

## [0] Regret-Optimal Model-Free Reinforcement Learning for Discounted MDPs with Short Burn-In Time

**Authors**: *Xiang Ji, Gen Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff887781480973bd3cb6026feb378d1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff887781480973bd3cb6026feb378d1e-Abstract-Conference.html)

**Abstract**:

A crucial problem in reinforcement learning is learning the optimal policy. We study this in tabular infinite-horizon discounted Markov decision processes under the online setting. The existing algorithms either fail to achieve regret optimality or have to incur a high memory and computational cost. In addition, existing optimal algorithms all require a long burn-in time in order to achieve optimal sample efficiency, i.e., their optimality is not guaranteed unless sample size surpasses a high threshold. We address both open problems by introducing a model-free algorithm that employs variance reduction and a novel technique that switches the execution policy in a slow-yet-adaptive manner. This is the first regret-optimal model-free algorithm in the discounted setting, with the additional benefit of a low burn-in time.

----

## [0] Convolutional State Space Models for Long-Range Spatiotemporal Modeling

**Authors**: *Jimmy T. H. Smith, Shalini De Mello, Jan Kautz, Scott W. Linderman, Wonmin Byeon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff9783ec29688387d44779d67d06ef66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff9783ec29688387d44779d67d06ef66-Abstract-Conference.html)

**Abstract**:

Effectively modeling long spatiotemporal sequences is challenging due to the need to model complex spatial correlations and long-range temporal dependencies simultaneously. ConvLSTMs attempt to address this by updating tensor-valued states with recurrent neural networks, but their sequential computation makes them slow to train. In contrast, Transformers can process an entire spatiotemporal sequence, compressed into tokens, in parallel. However, the cost of attention scales quadratically in length, limiting their scalability to longer sequences. Here, we address the challenges of prior methods and introduce convolutional state space models (ConvSSM) that combine the tensor modeling ideas of ConvLSTM with the long sequence modeling approaches of state space methods such as S4 and S5. First, we demonstrate how parallel scans can be applied to convolutional recurrences to achieve subquadratic parallelization and fast autoregressive generation. We then establish an equivalence between the dynamics of ConvSSMs and SSMs, which motivates parameterization and initialization strategies for modeling long-range dependencies. The result is ConvS5, an efficient ConvSSM variant for long-range spatiotemporal modeling. ConvS5 significantly outperforms Transformers and ConvLSTM on a long horizon Moving-MNIST experiment while training $3\times$ faster than ConvLSTM and generating samples $400\times$ faster than Transformers. In addition,  ConvS5 matches or exceeds the performance of state-of-the-art methods on challenging DMLab, Minecraft and Habitat prediction benchmarks and enables new directions for modeling long spatiotemporal sequences.

----

## [0] CRoSS: Diffusion Model Makes Controllable, Robust and Secure Image Steganography

**Authors**: *Jiwen Yu, Xuanyu Zhang, Youmin Xu, Jian Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ff99390b6e942fb1dd7023f787fb0a27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ff99390b6e942fb1dd7023f787fb0a27-Abstract-Conference.html)

**Abstract**:

Current image steganography techniques are mainly focused on cover-based methods, which commonly have the risk of leaking secret images and poor robustness against degraded container images. Inspired by recent developments in diffusion models, we discovered that two properties of diffusion models, the ability to achieve translation between two images without training, and robustness to noisy data, can be used to improve security and natural robustness in image steganography tasks. For the choice of diffusion model, we selected Stable Diffusion, a type of conditional diffusion model, and fully utilized the latest tools from open-source communities, such as LoRAs and ControlNets, to improve the controllability and diversity of container images. In summary, we propose a novel image steganography framework, named Controllable, Robust and Secure Image Steganography (CRoSS), which has significant advantages in controllability, robustness, and security compared to cover-based image steganography methods. These benefits are obtained without additional training. To our knowledge, this is the first work to introduce diffusion models to the field of image steganography. In the experimental section, we conducted detailed experiments to demonstrate the advantages of our proposed CRoSS framework in controllability, robustness, and security.

----

## [0] American Stories: A Large-Scale Structured Text Dataset of Historical US Newspapers

**Authors**: *Melissa Dell, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin, Leander Heldring*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ffeb860479ccae44d84c0de32acd693d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ffeb860479ccae44d84c0de32acd693d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Existing full text datasets of U.S. public domain newspapers do not recognize the often complex layouts of newspaper scans, and as a result the digitized content scrambles texts from articles, headlines, captions, advertisements, and other layout regions. OCR quality can also be low. This study develops a novel, deep learning pipeline for extracting full article texts from newspaper images and applies it to the nearly 20 million scans in Library of Congress's public domain Chronicling America collection. The pipeline includes layout detection, legibility classification, custom OCR, and association of article texts spanning multiple bounding boxes. To achieve high scalability, it is built with efficient architectures designed for mobile phones. The resulting American Stories dataset provides high quality data that could be used for pre-training a large language model to achieve better understanding of historical English and historical world knowledge. The dataset could also be added to the external database of a retrieval-augmented language model to make historical information - ranging from interpretations of political events to minutiae about the lives of people's ancestors - more widely accessible. Furthermore, structured article texts facilitate using transformer-based methods for popular social science applications like topic classification, detection of reproduced content, and news story clustering.  Finally, American Stories provides a massive silver quality dataset for innovating multimodal layout analysis models and other multimodal applications.

----



[Go to the previous page](NIPS-2023-list3500.md)

[Go to the catalog section](README.md)